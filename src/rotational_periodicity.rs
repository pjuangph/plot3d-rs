//! Utilities for detecting rotational periodicity in structured multi-block grids.
//!
//! This module mirrors the behaviour of the original Python tooling and is covered end-to-end by
//! the integration test in `tests/test_rotational_periodicity.rs`. Generate HTML documentation with
//! `cargo doc --open` to browse rendered versions of these notes alongside the Rust API surface.
//!
//! # Three-Phase Periodicity Algorithm
//!
//! The [`rotated_periodicity`] function (via `rotational_periodicity_core`) detects
//! periodic face pairs across a rotation angle in three phases:
//!
//! **Phase 1 — Full-face matching via parallel theta bucketing**
//!
//! All outer faces are inserted into a `FacePool` that buckets them by
//! their angular (theta) coordinate in cylindrical space. For each face,
//! the rotated counterpart's theta range is computed and candidate faces
//! in the matching bucket are tested with `full_face_match_transformed`.
//! When all four rotated corners match a candidate's corners within
//! tolerance, a [`PeriodicPair`] is recorded. This phase runs in parallel
//! across all outer faces.
//!
//! **Phase 2 — Split-face matching with corner pre-check**
//!
//! Remaining unmatched faces are tested for partial overlap. Before the
//! expensive `get_face_intersection`, a quick corner pre-check confirms
//! that at least one rotated corner lies near a candidate face. When a
//! partial match is found, both faces are split along the intersection
//! boundary. Matched sub-faces produce [`PeriodicPair`] records and
//! remnants re-enter the pool. This loop runs until convergence — there
//! is **no iteration limit** (earlier versions had a hardcoded limit of 50
//! which was insufficient; Phase 2 may need 100+ iterations).
//!
//! **Phase 3 — Edge-based matching**
//!
//! Any faces still unmatched after Phase 2 are tested using edge geometry.
//! Face edges are extracted and compared via `count_edge_matches` in the
//! `FacePool`. This catches degenerate or thin-strip faces that the
//! area-based intersection misses.
//!
//! Use [`verify_periodicity`] to confirm that every matched periodic pair
//! has coincident nodes after rotation.

use std::collections::HashSet;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::{
    block::Block,
    block_face_functions::{
        create_face_from_diagonals, match_faces_to_list, outer_face_records_to_list, reduce_blocks,
        Face, FaceAxis,
    },
    connectivity::get_face_intersection,
    face_pool::{count_edge_matches, extract_face_edges, FacePool},
    face_record::{FaceKey, FaceMatch, FaceRecord, MatchPoint, Orientation, OrientationPlane, PeriodicPair},
    utils::{apply_rotation, compute_min_gcd, distance3},
    Float,
};

/// Rotation matrix for the requested axis.
///
/// # Arguments
/// * `angle` - Rotation angle in radians.
/// * `axis` - Axis designator (`'x'`, `'y'`, `'z'`, case-insensitive).
///
/// # Returns
/// A 3×3 rotation matrix in row-major order.
pub fn create_rotation_matrix(angle: Float, axis: char) -> [[Float; 3]; 3] {
    match axis.to_ascii_lowercase() {
        'x' => [
            [1.0, 0.0, 0.0],
            [0.0, angle.cos(), -angle.sin()],
            [0.0, angle.sin(), angle.cos()],
        ],
        'y' => [
            [angle.cos(), 0.0, angle.sin()],
            [0.0, 1.0, 0.0],
            [-angle.sin(), 0.0, angle.cos()],
        ],
        'z' => [
            [angle.cos(), -angle.sin(), 0.0],
            [angle.sin(), angle.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ],
        _ => panic!("Unsupported rotation axis '{axis}'"),
    }
}

/// Rotate a block using a precomputed rotation matrix.
///
/// # Arguments
/// * `block` - Source block supplying the original coordinates.
/// * `rotation` - 3×3 rotation matrix in row-major order.
///
/// # Returns
/// A new [`Block`] whose nodes are the rotated copy of `block`.
pub fn rotate_block_with_matrix(block: &Block, rotation: [[Float; 3]; 3]) -> Block {
    crate::block_face_functions::rotate_block(block, rotation)
}

/// Detect rotational periodicity by reducing grids by the minimum shared GCD,
/// running the 3-phase matching algorithm, then scaling results back.
///
/// # Arguments
/// * `blocks` - Full-resolution blocks that define the geometry.
/// * `matched_faces` - Interfaces already known to match between blocks.
/// * `outer_faces` - Faces that remain exposed after connectivity processing.
/// * `periodic_direction` - Axis (`"i"`, `"j"`, or `"k"`) along which periodicity is expected.
/// * `rotation_axis` - Axis of rotation (`'x'`, `'y'`, or `'z'`).
/// * `rotation_angle` - Rotation angle in radians.
///
/// # Returns
/// `(periodic_pairs, outer_faces)` containing the periodic matches and the filtered outer faces.
pub fn rotational_periodicity(
    blocks: &[Block],
    matched_faces: &[FaceMatch],
    outer_faces: &[FaceRecord],
    periodic_direction: &str,
    rotation_axis: char,
    rotation_angle: Float,
) -> (Vec<PeriodicPair>, Vec<FaceRecord>) {
    let gcd_to_use = compute_min_gcd(blocks);

    let reduced_blocks = reduce_blocks(blocks, gcd_to_use);

    let mut matched_scaled = matched_faces.to_vec();
    for entry in &mut matched_scaled {
        entry.divide_indices(gcd_to_use);
    }

    let mut outer_scaled = outer_faces.to_vec();
    for dict in &mut outer_scaled {
        dict.divide_indices(gcd_to_use);
    }

    let (mut periodic_export, mut outer_export) = rotational_periodicity_core(
        &reduced_blocks,
        &matched_scaled,
        &outer_scaled,
        rotation_angle,
        periodic_direction,
        rotation_axis,
    );

    if gcd_to_use > 1 {
        for rec in &mut periodic_export {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }

        for dict in &mut outer_export {
            dict.scale_indices(gcd_to_use);
        }
    }

    (periodic_export, outer_export)
}

/// Core implementation shared by `rotational_periodicity` and `rotated_periodicity`.
///
/// Uses a 3-phase algorithm:
///   Phase 1 — Full-face matching via cylindrical bucketing (O(N log N))
///   Phase 2 — Split-face matching with corner pre-checks
///   Phase 3 — Edge-based matching for remaining faces
fn rotational_periodicity_core(
    blocks: &[Block],
    matched_faces: &[FaceMatch],
    outer_faces: &[FaceRecord],
    rotation_angle: Float,
    periodic_direction: &str,
    rotation_axis: char,
) -> (Vec<PeriodicPair>, Vec<FaceRecord>) {
    use crate::block_face_functions::full_face_match_transformed;

    let rot_forward = create_rotation_matrix(rotation_angle, rotation_axis);
    let rot_backward = create_rotation_matrix(-rotation_angle, rotation_axis);

    let transform_fwd = |p: [Float; 3]| apply_rotation(p, rot_forward);
    let transform_rev = |p: [Float; 3]| apply_rotation(p, rot_backward);

    let mut periodic_exports: Vec<FaceMatch> = Vec::new();
    let mut seen_pair_keys: HashSet<(FaceKey, FaceKey)> = HashSet::new();

    let outer_faces_all = outer_face_records_to_list(blocks, outer_faces, 1);
    let matched_faces_all = match_faces_to_list(blocks, matched_faces, 1);

    // Build the face pool with cylindrical metadata
    let mut pool = FacePool::new(outer_faces_all, rotation_axis);

    // Theta tolerance for candidate search: fraction of rotation angle, clamped to a
    // reasonable range to avoid excessively wide searches for few-blade machines.
    let theta_tol = (rotation_angle.abs() * 0.15 + 0.05).min(0.25);

    // ===== PHASE 1: Full-face matching via cylindrical bucketing =====
    {
        let active = pool.active_indices();
        let pb = make_progress_bar(
            active.len() as u64,
            "faces",
            "Rot. periodicity Phase 1 (corners)",
        );

        // Parallel search: each face_a independently finds its best candidate match.
        // The pool is read-only during this phase; matches are deduplicated serially afterwards.
        let phase1_matches: Vec<(FaceKey, FaceKey, FaceMatch)> = active
            .par_iter()
            .filter_map(|&idx_a| {
                pb.inc(1);
                if pool.is_consumed(idx_a) {
                    return None;
                }

                let face_a = &pool.faces[idx_a];
                let candidates = pool.find_rotational_candidates(idx_a, rotation_angle, theta_tol);

                for &idx_b in &candidates {
                    if idx_a == idx_b || pool.is_consumed(idx_b) {
                        continue;
                    }
                    let face_b = &pool.faces[idx_b];

                    if !faces_support_direction(face_a, face_b, periodic_direction) {
                        continue;
                    }
                    if face_a.const_type() == -1 || face_b.const_type() == -1 {
                        continue;
                    }

                    // Try both rotation directions for this candidate
                    for &rot_mat in &[rot_forward, rot_backward] {
                        let transform = |p: [Float; 3]| apply_rotation(p, rot_mat);
                        if let Some(orientation) =
                            full_face_match_transformed(face_a, face_b, transform, MATCH_TOL)
                        {
                            let key_a = face_a.index_key();
                            let key_b = face_b.index_key();
                            return Some((
                                key_a,
                                key_b,
                                FaceMatch {
                                    block1: FaceRecord::from_face(face_a),
                                    block2: FaceRecord::from_face(face_b),
                                    points: Vec::new(),
                                    orientation: Some(orientation),
                                },
                            ));
                        }
                    }
                }
                None
            })
            .collect();
        pb.finish_and_clear();

        // Apply Phase 1 matches serially, deduplicating faces matched by multiple threads
        let mut consumed_in_phase1: HashSet<FaceKey> = HashSet::new();
        for (key_a, key_b, fm) in phase1_matches {
            if consumed_in_phase1.contains(&key_a) || consumed_in_phase1.contains(&key_b) {
                continue;
            }
            let pair_key = ordered_pair(key_a, key_b);
            if seen_pair_keys.contains(&pair_key) {
                continue;
            }
            seen_pair_keys.insert(pair_key);
            consumed_in_phase1.insert(key_a);
            consumed_in_phase1.insert(key_b);
            pool.consume(key_a);
            pool.consume(key_b);
            periodic_exports.push(fm);
        }
    }

    // ===== PHASE 2: Split-face matching with corner pre-checks =====
    {
        let mut changed = true;
        let mut iteration = 0usize;
        let mut non_matching_p2: HashSet<(FaceKey, FaceKey)> = HashSet::new();

        while changed {
            changed = false;
            iteration += 1;

            let active = pool.active_indices();
            let pb = make_progress_bar(
                active.len() as u64,
                "faces",
                format!("Rot. periodicity Phase 2 pass {iteration}"),
            );

            let mut match_found = None; // (idx_a, idx_b, rot_matrix)

            'phase2_search: for &idx_a in &active {
                pb.inc(1);
                if pool.is_consumed(idx_a) {
                    continue;
                }
                let face_a = &pool.faces[idx_a];
                let candidates = pool.find_rotational_candidates(idx_a, rotation_angle, theta_tol);

                for &idx_b in &candidates {
                    if idx_a == idx_b || pool.is_consumed(idx_b) {
                        continue;
                    }
                    let face_b = &pool.faces[idx_b];

                    let key_pair = ordered_pair(face_a.index_key(), face_b.index_key());
                    if non_matching_p2.contains(&key_pair) {
                        continue;
                    }

                    if !faces_support_direction(face_a, face_b, periodic_direction) {
                        non_matching_p2.insert(key_pair);
                        continue;
                    }
                    if face_a.const_type() == -1 || face_b.const_type() == -1 {
                        non_matching_p2.insert(key_pair);
                        continue;
                    }

                    let block_idx_a = match face_a.block_index() {
                        Some(idx) => idx,
                        None => continue,
                    };
                    let block_idx_b = match face_b.block_index() {
                        Some(idx) => idx,
                        None => continue,
                    };
                    if block_idx_a >= blocks.len() || block_idx_b >= blocks.len() {
                        continue;
                    }

                    let block_b = &blocks[block_idx_b];

                    // Try both rotation directions for this candidate
                    let mut found_corners = false;
                    for &rot_matrix in &[rot_forward, rot_backward] {
                        let corners_hit = count_rotated_corners_on_face(
                            face_a, face_b, block_b, rot_matrix, MATCH_TOL,
                        );
                        if corners_hit >= 2 {
                            match_found = Some((idx_a, idx_b, rot_matrix));
                            found_corners = true;
                            break;
                        }
                    }
                    if found_corners {
                        break 'phase2_search;
                    }
                    non_matching_p2.insert(key_pair);
                }
            }
            pb.finish_and_clear();

            // Process the found match
            if let Some((idx_a, idx_b, rot_matrix)) = match_found {
                let face_a = pool.faces[idx_a].clone();
                let face_b = pool.faces[idx_b].clone();
                let block_idx_a = face_a.block_index().unwrap();
                let block_idx_b = face_b.block_index().unwrap();

                let block_a_rot = rotate_block_with_matrix(&blocks[block_idx_a], rot_matrix);
                let block_b = &blocks[block_idx_b];

                if try_split_match(
                    &face_a,
                    &face_b,
                    &block_a_rot,
                    block_b,
                    blocks,
                    &mut seen_pair_keys,
                    &mut periodic_exports,
                    &mut pool,
                ) {
                    changed = true;
                } else {
                    // Candidate had corners but failed intersection — skip it next time
                    let pair_key = ordered_pair(face_a.index_key(), face_b.index_key());
                    non_matching_p2.insert(pair_key);
                    changed = true; // restart to try next candidate
                }
            }
        }
    }

    // ===== PHASE 3: Edge-based matching =====
    {
        let mut changed_p3 = true;
        let mut iteration_p3 = 0usize;
        let mut non_matching_p3: HashSet<(FaceKey, FaceKey)> = HashSet::new();

        while changed_p3 {
            changed_p3 = false;
            iteration_p3 += 1;

            let active = pool.active_indices();
            let pb = make_progress_bar(
                active.len() as u64,
                "faces",
                format!("Rot. periodicity Phase 3 pass {iteration_p3}"),
            );

            let mut match_found: Option<(usize, usize, bool)> = None;

            'phase3_search: for (ai, &idx_a) in active.iter().enumerate() {
                pb.inc(1);
                if pool.is_consumed(idx_a) {
                    continue;
                }
                let face_a = &pool.faces[idx_a];

                let block_idx_a = match face_a.block_index() {
                    Some(idx) => idx,
                    None => continue,
                };
                if block_idx_a >= blocks.len() {
                    continue;
                }

                // Pre-compute edges for face_a on both rotated blocks
                let edges_a_fwd = extract_face_edges(
                    face_a,
                    &rotate_block_with_matrix(&blocks[block_idx_a], rot_forward),
                );
                let edges_a_rev = extract_face_edges(
                    face_a,
                    &rotate_block_with_matrix(&blocks[block_idx_a], rot_backward),
                );

                for &idx_b in &active[(ai + 1)..] {
                    if pool.is_consumed(idx_b) {
                        continue;
                    }
                    let face_b = &pool.faces[idx_b];

                    let key_pair = ordered_pair(face_a.index_key(), face_b.index_key());
                    if non_matching_p3.contains(&key_pair) {
                        continue;
                    }

                    if !faces_support_direction(face_a, face_b, periodic_direction) {
                        non_matching_p3.insert(key_pair);
                        continue;
                    }
                    if face_a.const_type() == -1 || face_b.const_type() == -1 {
                        non_matching_p3.insert(key_pair);
                        continue;
                    }

                    let block_idx_b = match face_b.block_index() {
                        Some(idx) => idx,
                        None => continue,
                    };
                    if block_idx_b >= blocks.len() {
                        continue;
                    }

                    let block_b = &blocks[block_idx_b];
                    // Extract edges for face_b on its original block
                    let edges_b = extract_face_edges(face_b, block_b);
                    if edges_b.is_empty() {
                        continue;
                    }

                    // Try forward rotation edges
                    for (edges_a, is_forward) in [(&edges_a_fwd, true), (&edges_a_rev, false)] {
                        if edges_a.is_empty() {
                            continue;
                        }
                        // Identity matrix since edges are already extracted from rotated block
                        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                        let match_count =
                            count_edge_matches(edges_a, &edges_b, identity, MATCH_TOL);
                        if match_count >= 2 {
                            match_found = Some((idx_a, idx_b, is_forward));
                            break 'phase3_search;
                        }
                    }

                    non_matching_p3.insert(key_pair);
                }
            }
            pb.finish_and_clear();

            // Process edge-based match
            if let Some((idx_a, idx_b, is_forward)) = match_found {
                let face_a = pool.faces[idx_a].clone();
                let face_b = pool.faces[idx_b].clone();
                let block_idx_a = face_a.block_index().unwrap();
                let block_idx_b = face_b.block_index().unwrap();

                let rot_matrix = if is_forward {
                    rot_forward
                } else {
                    rot_backward
                };
                let block_a_rot = rotate_block_with_matrix(&blocks[block_idx_a], rot_matrix);
                let block_b = &blocks[block_idx_b];

                if is_valid_face(&face_a, &block_a_rot) && is_valid_face(&face_b, block_b) {
                    // Try full face match first (edges may indicate perfect alignment)
                    let transform: &dyn Fn([Float; 3]) -> [Float; 3] = if is_forward {
                        &transform_fwd
                    } else {
                        &transform_rev
                    };
                    if let Some(orientation) =
                        full_face_match_transformed(&face_a, &face_b, transform, MATCH_TOL)
                    {
                        let key_a = face_a.index_key();
                        let key_b = face_b.index_key();
                        let pair_key = ordered_pair(key_a, key_b);
                        if !seen_pair_keys.contains(&pair_key) {
                            seen_pair_keys.insert(pair_key);
                            pool.consume(key_a);
                            pool.consume(key_b);
                            periodic_exports.push(FaceMatch {
                                block1: FaceRecord::from_face(&face_a),
                                block2: FaceRecord::from_face(&face_b),
                                points: Vec::new(),
                                orientation: Some(orientation),
                            });
                            changed_p3 = true;
                            continue;
                        }
                    }

                    // Fall back to node-level intersection
                    if try_split_match(
                        &face_a,
                        &face_b,
                        &block_a_rot,
                        block_b,
                        blocks,
                        &mut seen_pair_keys,
                        &mut periodic_exports,
                        &mut pool,
                    ) {
                        changed_p3 = true;
                    }
                }
            }
        }
    }

    // Filter out faces that are in matched_faces
    let matched_keys: HashSet<FaceKey> = matched_faces_all.iter().map(|f| f.index_key()).collect();
    let mut outer_exports = pool.drain_as_records();
    outer_exports.retain(|r| !matched_keys.contains(&r.index_key()));

    (periodic_exports, outer_exports)
}

/// Rotate the entire mesh by an arbitrary angle and recover periodic matches.
///
/// # Arguments
/// * `blocks` - Baseline blocks before rotation.
/// * `matched_faces` - Known face matches between blocks.
/// * `outer_faces` - Exposed faces supplied from connectivity.
/// * `rotation_angle_deg` - Rotation angle in degrees applied to the candidate block.
/// * `rotation_axis` - Axis about which the rotation occurs.
/// * `reduce_mesh` - When `true`, down-sample the mesh by a shared GCD prior to matching.
///
/// # Returns
/// `(periodic_pairs, outer_faces)` mirroring [`rotational_periodicity`], but driven by the supplied
/// angle instead of the blade count.
pub fn rotated_periodicity(
    blocks: &[Block],
    matched_faces: &[FaceMatch],
    outer_faces: &[FaceRecord],
    rotation_angle_deg: Float,
    rotation_axis: char,
    reduce_mesh: bool,
) -> (Vec<PeriodicPair>, Vec<FaceRecord>) {
    let mut gcd_to_use = 1usize;
    let mut working_blocks: Vec<Block> = blocks.to_vec();
    if reduce_mesh && !blocks.is_empty() {
        gcd_to_use = compute_min_gcd(blocks);
        working_blocks = reduce_blocks(blocks, gcd_to_use);
    }

    let mut matched_scaled = matched_faces.to_vec();
    for entry in &mut matched_scaled {
        entry.divide_indices(gcd_to_use);
    }

    let mut outer_scaled = outer_faces.to_vec();
    for dict in &mut outer_scaled {
        dict.divide_indices(gcd_to_use);
    }

    let rotation_angle_rad = rotation_angle_deg.to_radians();

    // Use "any" as periodic direction since rotated_periodicity supports all directions
    let (mut periodic_export, mut outer_export) = rotational_periodicity_core(
        &working_blocks,
        &matched_scaled,
        &outer_scaled,
        rotation_angle_rad,
        "any",
        rotation_axis,
    );

    if gcd_to_use > 1 {
        for rec in &mut periodic_export {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }
        for dict in &mut outer_export {
            dict.scale_indices(gcd_to_use);
        }
    }

    (periodic_export, outer_export)
}

/// Create a styled progress bar with consistent formatting.
fn make_progress_bar(total: u64, unit: &str, message: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new(total);
    let template =
        format!("{{msg}} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} {unit} ({{eta}} remaining)");
    pb.set_style(
        ProgressStyle::with_template(&template)
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(message.into());
    pb
}

/// Order a pair of keys so the smallest always comes first.
///
/// # Arguments
/// * `a`, `b` - Keys to order.
///
/// # Returns
/// `(min, max)` ensuring deterministic ordering.
fn ordered_pair(a: FaceKey, b: FaceKey) -> (FaceKey, FaceKey) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Check whether both faces are constant along the requested periodic direction.
///
/// # Arguments
/// * `face_a`, `face_b` - Candidate faces to compare.
/// * `direction` - Periodic direction string (`"i"`, `"j"`, or `"k"`).
///
/// # Returns
/// `true` when both faces hold constant indices along `direction`.
pub fn faces_support_direction(face_a: &Face, face_b: &Face, direction: &str) -> bool {
    let dir = direction.trim().to_ascii_lowercase();
    match dir.as_str() {
        "i" => face_a.imin() == face_a.imax() && face_b.imin() == face_b.imax(),
        "j" => face_a.jmin() == face_a.jmax() && face_b.jmin() == face_b.jmax(),
        "k" => face_a.kmin() == face_a.kmax() && face_b.kmin() == face_b.kmax(),
        "any" => faces_support_any(face_a, face_b),
        _ => false,
    }
}

/// Returns `true` when both faces hold a constant index along at least one axis.
///
/// # Arguments
/// * `face_a`, `face_b` - Faces tested for flatness along any axis.
///
/// # Returns
/// `true` when the faces are planar along a shared axis.
pub fn faces_support_any(face_a: &Face, face_b: &Face) -> bool {
    let a_planar = face_a.imin() == face_a.imax()
        || face_a.jmin() == face_a.jmax()
        || face_a.kmin() == face_a.kmax();
    let b_planar = face_b.imin() == face_b.imax()
        || face_b.jmin() == face_b.jmax()
        || face_b.kmin() == face_b.kmax();
    a_planar && b_planar
}

/// Check whether face indices fit within a block's dimensions.
///
/// Returns `false` when any structured index on the face exceeds the corresponding
/// block extent, which would cause an out-of-bounds access during node lookup.
fn is_valid_face(face: &Face, block: &Block) -> bool {
    face.imin() < block.imax
        && face.imax() < block.imax
        && face.jmin() < block.jmax
        && face.jmax() < block.jmax
        && face.kmin() < block.kmax
        && face.kmax() < block.kmax
}

/// Attempt a split-face intersection match between two faces after rotation.
///
/// If the intersection succeeds, records the match in `periodic_exports`, consumes the
/// originals from `pool`, and adds any split remnants back. Returns `true` on success.
///
/// Split remnants from face_a's block are re-created using the original (unrotated)
/// block so that their vertex coordinates and cylindrical metadata are correct for
/// subsequent matching passes.
#[allow(clippy::too_many_arguments)]
fn try_split_match(
    face_a: &Face,
    face_b: &Face,
    block_a_rot: &Block,
    block_b: &Block,
    blocks: &[Block],
    seen_pair_keys: &mut HashSet<(FaceKey, FaceKey)>,
    periodic_exports: &mut Vec<FaceMatch>,
    pool: &mut FacePool,
) -> bool {
    if !is_valid_face(face_a, block_a_rot) || !is_valid_face(face_b, block_b) {
        return false;
    }
    if let Some((pair_faces, match_points, splits)) =
        periodicity_check_with_points(face_a, face_b, block_a_rot, block_b, MATCH_TOL)
    {
        let pair_key = ordered_pair(pair_faces[0].index_key(), pair_faces[1].index_key());
        if seen_pair_keys.contains(&pair_key) {
            return false;
        }
        seen_pair_keys.insert(pair_key);

        let orientation =
            infer_orientation_from_match_points(&match_points, &pair_faces[0], &pair_faces[1]);

        // Derive lb/ub from first/last MatchPoint (iloc-style), matching Python's
        // _build_periodic_export which uses df.iloc[0] / df.iloc[-1].
        let b1_rec = if !match_points.is_empty() {
            let first = &match_points[0];
            let last = &match_points[match_points.len() - 1];
            FaceRecord {
                block_index: pair_faces[0].block_index().unwrap_or(usize::MAX),
                il: first.i1, jl: first.j1, kl: first.k1,
                ih: last.i1, jh: last.j1, kh: last.k1,
                id: pair_faces[0].id(),
                u_physical: None,
                v_physical: None,
            }
        } else {
            FaceRecord::from_face(&pair_faces[0])
        };
        let b2_rec = if !match_points.is_empty() {
            let first = &match_points[0];
            let last = &match_points[match_points.len() - 1];
            FaceRecord {
                block_index: pair_faces[1].block_index().unwrap_or(usize::MAX),
                il: first.i2, jl: first.j2, kl: first.k2,
                ih: last.i2, jh: last.j2, kh: last.k2,
                id: pair_faces[1].id(),
                u_physical: None,
                v_physical: None,
            }
        } else {
            FaceRecord::from_face(&pair_faces[1])
        };

        periodic_exports.push(FaceMatch {
            block1: b1_rec,
            block2: b2_rec,
            points: match_points,
            orientation,
        });

        let removal = collect_removal_keys(face_a, face_b, &pair_faces);
        for key in &removal {
            pool.consume(*key);
        }

        // Re-create split remnants from the rotated block using the original
        // (unrotated) block coordinates. Without this fix, remnants from face_a's
        // side would carry rotated vertex positions, causing their theta centroids
        // to be offset by the rotation angle and preventing subsequent matches.
        let block_idx_a = face_a.block_index().unwrap_or(usize::MAX);
        for s in splits {
            let bidx = s.block_index().unwrap_or(usize::MAX);
            if bidx == block_idx_a && bidx < blocks.len() {
                let mut fixed = create_face_from_diagonals(
                    &blocks[bidx],
                    s.imin(),
                    s.jmin(),
                    s.kmin(),
                    s.imax(),
                    s.jmax(),
                    s.kmax(),
                );
                fixed.set_block_index(bidx);
                if let Some(id) = s.id() {
                    fixed.set_id(id);
                }
                pool.add_face(fixed);
            } else {
                pool.add_face(s);
            }
        }
        return true;
    }
    false
}

/// Gather all face keys involved in a successful periodic match for removal.
///
/// # Arguments
/// * `face_a`, `face_b` - Faces that triggered the match.
/// * `pair_faces` - Matched faces returned by [`periodicity_check`].
///
/// # Returns
/// Sorted, deduplicated list of keys to remove from future consideration.
fn collect_removal_keys(face_a: &Face, face_b: &Face, pair_faces: &[Face]) -> Vec<FaceKey> {
    let mut keys = Vec::new();
    keys.push(face_a.index_key());
    keys.push(face_b.index_key());
    for f in pair_faces {
        keys.push(f.index_key());
    }
    keys.sort();
    keys.dedup();
    keys
}

/// Intersect two faces after rotation and return matching subfaces, match points, and splits.
///
/// # Arguments
/// * `face1`, `face2` - Faces inspected for overlap.
/// * `block1`, `block2` - Blocks providing geometric detail for each face.
/// * `tol` - Node coincidence tolerance.
///
/// # Returns
/// `Some((matched_faces, match_points, splits))` when overlap exists.
pub fn periodicity_check_with_points(
    face1: &Face,
    face2: &Face,
    block1: &Block,
    block2: &Block,
    tol: Float,
) -> Option<(Vec<Face>, Vec<MatchPoint>, Vec<Face>)> {
    let mut face_a = face1.clone();
    let mut face_b = face2.clone();
    let mut swapped = false;
    let (block_a, block_b) = if face_b.diagonal_length() < face_a.diagonal_length() {
        std::mem::swap(&mut face_a, &mut face_b);
        swapped = true;
        (block2, block1)
    } else {
        (block1, block2)
    };

    let (matches, mut split1, split2) =
        get_face_intersection(&face_a, &face_b, block_a, block_b, tol);
    if matches.len() < 4 {
        return None;
    }

    let bounds_a = match_bounds(&matches, true);
    let bounds_b = match_bounds(&matches, false);

    let mut out1 = create_face_from_diagonals(
        block_a, bounds_a.0, bounds_a.2, bounds_a.4, bounds_a.1, bounds_a.3, bounds_a.5,
    );
    out1.set_block_index(face_a.block_index().unwrap_or(usize::MAX));
    if let Some(id) = face_a.id() {
        out1.set_id(id);
    }

    let mut out2 = create_face_from_diagonals(
        block_b, bounds_b.0, bounds_b.2, bounds_b.4, bounds_b.1, bounds_b.3, bounds_b.5,
    );
    out2.set_block_index(face_b.block_index().unwrap_or(usize::MAX));
    if let Some(id) = face_b.id() {
        out2.set_id(id);
    }

    split1.extend(split2);

    let pair = if swapped {
        vec![out2, out1]
    } else {
        vec![out1, out2]
    };

    Some((pair, matches, split1))
}

/// Determine the bounds of matching points for either the first or second face.
///
/// # Arguments
/// * `matches` - Point-to-point matches returned by connectivity.
/// * `first` - When `true`, consider the first face indices; otherwise use the second.
///
/// # Returns
/// `(imin, imax, jmin, jmax, kmin, kmax)` describing the bounding box.
fn match_bounds(
    matches: &[crate::face_record::MatchPoint],
    first: bool,
) -> (usize, usize, usize, usize, usize, usize) {
    let mut i_lo = usize::MAX;
    let mut j_lo = usize::MAX;
    let mut k_lo = usize::MAX;
    let mut i_hi = 0usize;
    let mut j_hi = 0usize;
    let mut k_hi = 0usize;
    for m in matches {
        let (i, j, k) = if first {
            (m.i1, m.j1, m.k1)
        } else {
            (m.i2, m.j2, m.k2)
        };
        i_lo = i_lo.min(i);
        j_lo = j_lo.min(j);
        k_lo = k_lo.min(k);
        i_hi = i_hi.max(i);
        j_hi = j_hi.max(j);
        k_hi = k_hi.max(k);
    }
    (i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)
}

/// Fixed matching tolerance for node coincidence checks.
const MATCH_TOL: Float = 1e-6;

// ============================================================================
// Orientation inference from match points
// ============================================================================

/// Infer orientation (u_reversed, v_reversed, swapped) from node-level MatchPoints.
///
/// Examines how block1 parametric indices map to block2 parametric indices.
fn infer_orientation_from_match_points(
    points: &[MatchPoint],
    face1: &Face,
    face2: &Face,
) -> Option<Orientation> {
    if points.len() < 2 {
        return None;
    }

    let axis1 = face1.const_axis()?;
    let axis2 = face2.const_axis()?;

    // Extract parametric (u, v) for each face based on constant axis
    let to_uv1 = |p: &MatchPoint| -> (isize, isize) {
        match axis1 {
            FaceAxis::I => (p.j1 as isize, p.k1 as isize),
            FaceAxis::J => (p.i1 as isize, p.k1 as isize),
            FaceAxis::K => (p.i1 as isize, p.j1 as isize),
        }
    };
    let to_uv2 = |p: &MatchPoint| -> (isize, isize) {
        match axis2 {
            FaceAxis::I => (p.j2 as isize, p.k2 as isize),
            FaceAxis::J => (p.i2 as isize, p.k2 as isize),
            FaceAxis::K => (p.i2 as isize, p.j2 as isize),
        }
    };

    // Find two points where u1 differs (along u-direction)
    let (u1_0, v1_0) = to_uv1(&points[0]);
    let (u2_0, v2_0) = to_uv2(&points[0]);

    // Look for a point with different u1
    let mut u_pair = None;
    let mut v_pair = None;
    for p in points.iter().skip(1) {
        let (u1, v1) = to_uv1(p);
        let (u2, v2) = to_uv2(p);
        if u1 != u1_0 && u_pair.is_none() {
            u_pair = Some((u1 - u1_0, v1 - v1_0, u2 - u2_0, v2 - v2_0));
        }
        if v1 != v1_0 && v_pair.is_none() {
            v_pair = Some((u1 - u1_0, v1 - v1_0, u2 - u2_0, v2 - v2_0));
        }
        if u_pair.is_some() && v_pair.is_some() {
            break;
        }
    }

    // Determine orientation from the mapping
    // When u1 changes: if u2 also changes → not swapped; if v2 changes → swapped
    let u_info = u_pair?;
    let du1 = u_info.0; // delta_u1
    let du2 = u_info.2; // delta_u2
    let dv2_from_u = u_info.3; // delta_v2 when u1 changes

    let swapped = du2 == 0 && dv2_from_u != 0;

    if swapped {
        // u1 maps to v2, need to figure out v1 maps to u2
        let u_reversed = if let Some(v_info) = v_pair {
            // v1 changes → check u2
            v_info.1.signum() != v_info.2.signum()
        } else {
            false
        };
        let v_reversed = dv2_from_u != 0 && (du1.signum() != dv2_from_u.signum());
        Some(Orientation::from_flags(
            u_reversed,
            v_reversed,
            true,
            if axis1 == axis2 { OrientationPlane::InPlane } else { OrientationPlane::CrossPlane },
        ))
    } else {
        let u_reversed = du1 != 0 && du2 != 0 && (du1.signum() != du2.signum());
        let v_reversed = if let Some(v_info) = v_pair {
            let dv1 = v_info.1;
            let dv2 = v_info.3;
            dv1 != 0 && dv2 != 0 && (dv1.signum() != dv2.signum())
        } else {
            false
        };
        Some(Orientation::from_flags(
            u_reversed,
            v_reversed,
            false,
            if axis1 == axis2 { OrientationPlane::InPlane } else { OrientationPlane::CrossPlane },
        ))
    }
}

/// Count how many of face_a's corners (after rotation) land near grid points of face_b.
pub fn count_rotated_corners_on_face(
    face_a: &Face,
    face_b: &Face,
    block_b: &Block,
    rotation_matrix: [[Float; 3]; 3],
    tol: Float,
) -> usize {
    let corners_a = face_a.vertices();
    let mut count = 0;

    // Sample face_b's boundary nodes
    let axis_b = match face_b.const_axis() {
        Some(a) => a,
        None => return 0,
    };
    let mut face_b_nodes: Vec<[Float; 3]> = Vec::new();
    match axis_b {
        FaceAxis::I => {
            let ic = face_b.imin();
            for j in face_b.jmin()..=face_b.jmax() {
                for k in face_b.kmin()..=face_b.kmax() {
                    if j < block_b.jmax && k < block_b.kmax && ic < block_b.imax {
                        let (x, y, z) = block_b.xyz(ic, j, k);
                        face_b_nodes.push([x, y, z]);
                    }
                }
            }
        }
        FaceAxis::J => {
            let jc = face_b.jmin();
            for i in face_b.imin()..=face_b.imax() {
                for k in face_b.kmin()..=face_b.kmax() {
                    if i < block_b.imax && k < block_b.kmax && jc < block_b.jmax {
                        let (x, y, z) = block_b.xyz(i, jc, k);
                        face_b_nodes.push([x, y, z]);
                    }
                }
            }
        }
        FaceAxis::K => {
            let kc = face_b.kmin();
            for i in face_b.imin()..=face_b.imax() {
                for j in face_b.jmin()..=face_b.jmax() {
                    if i < block_b.imax && j < block_b.jmax && kc < block_b.kmax {
                        let (x, y, z) = block_b.xyz(i, j, kc);
                        face_b_nodes.push([x, y, z]);
                    }
                }
            }
        }
    }

    for corner in corners_a {
        let rotated = apply_rotation(*corner, rotation_matrix);
        if face_b_nodes.iter().any(|n| distance3(rotated, *n) <= tol) {
            count += 1;
        }
    }
    count
}

/// Compute the rotation angle and matrix from `face1` to `face2`.
///
/// This assumes the rotation axis is the x-direction, which is suitable
/// for faces within the same turbomachinery passage.
///
/// Reference: Linear Real Transforms (`M_ccMBMesh.F`, `computeLRT`).
///
/// # Arguments
/// * `face1` - Source face.
/// * `face2` - Target face.
///
/// # Returns
/// `(angle_radians, rotation_matrix_3x3)`. Returns `(0.0, zeros)` when the
/// faces are already aligned.
pub fn linear_real_transform(face1: &Face, face2: &Face) -> (Float, [[Float; 3]; 3]) {
    let zero_matrix = [[0.0; 3]; 3];

    let (lower1, upper1) = match face1.get_corners() {
        Some(c) => c,
        None => return (0.0, zero_matrix),
    };
    let (lower2, upper2) = match face2.get_corners() {
        Some(c) => c,
        None => return (0.0, zero_matrix),
    };

    // Diagonal vectors
    let d_to = [
        upper1[0] - lower1[0],
        upper1[1] - lower1[1],
        upper1[2] - lower1[2],
    ];
    let d_from = [
        upper2[0] - lower2[0],
        upper2[1] - lower2[1],
        upper2[2] - lower2[2],
    ];

    let ld_to = (d_to[0] * d_to[0] + d_to[1] * d_to[1] + d_to[2] * d_to[2]).sqrt();
    let ld_from = (d_from[0] * d_from[0] + d_from[1] * d_from[1] + d_from[2] * d_from[2]).sqrt();

    if ld_to < 1e-15 || ld_from < 1e-15 {
        return (0.0, zero_matrix);
    }

    let n_to = [d_to[0] / ld_to, d_to[1] / ld_to, d_to[2] / ld_to];
    let n_from = [
        d_from[0] / ld_from,
        d_from[1] / ld_from,
        d_from[2] / ld_from,
    ];

    let dot = n_to[0] * n_from[0] + n_to[1] * n_from[1] + n_to[2] * n_from[2];

    if (dot - 1.0).abs() < 1e-10 {
        // No rotation needed
        return (0.0, zero_matrix);
    }

    // Compute angle from y,z components (rotation about x-axis)
    let denom_to = (n_to[1] * n_to[1] + n_to[2] * n_to[2]).sqrt();
    let denom_from = (n_from[1] * n_from[1] + n_from[2] * n_from[2]).sqrt();

    if denom_to < 1e-15 || denom_from < 1e-15 {
        return (0.0, zero_matrix);
    }

    let cos_ang = (n_to[1] * n_from[1] + n_to[2] * n_from[2]) / (denom_to * denom_from);
    let sin_ang = (n_to[2] * n_from[1] - n_to[1] * n_from[2]) / (denom_to * denom_from);
    let mut ang = cos_ang.clamp(-1.0, 1.0).acos();
    if sin_ang < 0.0 {
        ang = -ang;
    }

    let rotation_matrix = [
        [1.0, 0.0, 0.0],
        [0.0, cos_ang, -sin_ang],
        [0.0, sin_ang, cos_ang],
    ];

    (ang, rotation_matrix)
}

