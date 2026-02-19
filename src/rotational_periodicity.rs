//! Utilities for detecting rotational periodicity in structured multi-block grids.
//!
//! This module mirrors the behaviour of the original Python tooling and is covered end-to-end by
//! the integration test in `tests/test_rotational_periodicity.rs`. Generate HTML documentation with
//! `cargo doc --open` to browse rendered versions of these notes alongside the Rust API surface.

use std::collections::HashSet;
use std::f64::consts::PI;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::{
    block::Block,
    block_face_functions::{
        create_face_from_diagonals, find_angular_bounding_faces, match_faces_to_list,
        outer_face_records_to_list, reduce_blocks, rotate_block, to_radius, Face,
    },
    connectivity::{get_face_intersection, FaceMatch, FaceRecord},
    utils::{apply_rotation, compute_min_gcd, distance3, FaceKey},
};

/// Rotation matrix for the requested axis.
///
/// # Arguments
/// * `angle` - Rotation angle in radians.
/// * `axis` - Axis designator (`'x'`, `'y'`, `'z'`, case-insensitive).
///
/// # Returns
/// A 3×3 rotation matrix in row-major order.
pub fn create_rotation_matrix(angle: f64, axis: char) -> [[f64; 3]; 3] {
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
pub fn rotate_block_with_matrix(block: &Block, rotation: [[f64; 3]; 3]) -> Block {
    crate::block_face_functions::rotate_block(block, rotation)
}

/// Exportable description of a periodic face pairing.
pub type PeriodicPair = FaceMatch;

/// Detect rotational periodicity after reducing grids by the minimum shared GCD.
/// A more versatile version is [`rotated_periodicity`].
///
/// # Arguments
/// * `blocks` - Full-resolution blocks that define the geometry.
/// * `outer_faces` - Faces that remain exposed after connectivity processing.
/// * `matched_faces` - Interfaces already known to match between blocks.
/// * `periodic_direction` - Axis (`"i"`, `"j"`, or `"k"`) along which periodicity is expected.
/// * `rotation_axis` - Axis of rotation (`'x'`, `'y'`, or `'z'`).
/// * `nblades` - Number of periodic copies; controls the rotation increment.
///
/// # Returns
/// Tuple of `(periodic_pairs, outer_faces)` where the first element lists periodic matches as
/// [`PeriodicPair`] records and the second contains the remaining outer faces.
///
/// # Testing
/// The integration test `tests/test_rotational_periodicity.rs::rotational_periodicity_test`
/// exercises this helper as part of the publicly documented workflow.
pub fn rotational_periodicity_fast(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    matched_faces: &[FaceMatch],
    periodic_direction: &str,
    rotation_axis: char,
    nblades: usize,
) -> (Vec<PeriodicPair>, Vec<FaceRecord>) {
    let gcd_to_use = compute_min_gcd(blocks);

    let reduced_blocks = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);

    let mut matched_scaled = matched_faces.to_vec();
    for entry in &mut matched_scaled {
        entry.divide_indices(gcd_to_use);
    }

    let mut outer_scaled = outer_faces.to_vec();
    for dict in &mut outer_scaled {
        dict.divide_indices(gcd_to_use);
    }

    let (mut periodic_export, mut outer_export) = rotational_periodicity(
        &reduced_blocks,
        &matched_scaled,
        &outer_scaled,
        periodic_direction,
        rotation_axis,
        nblades,
    );

    for rec in &mut periodic_export {
        rec.block1.scale_indices(gcd_to_use);
        rec.block2.scale_indices(gcd_to_use);
    }

    for dict in &mut outer_export {
        dict.scale_indices(gcd_to_use);
    }

    return (periodic_export, outer_export);
}

/// Identify rotationally periodic face pairs without pre-scaling the mesh.
///
/// # Arguments
/// * `blocks` - Blocks evaluated at their current resolution.
/// * `matched_faces` - Pre-existing matched face records.
/// * `outer_faces` - Remaining outer faces for each block.
/// * `periodic_direction` - Axis (`"i"`, `"j"`, or `"k"`) that should stay constant across matches.
/// * `rotation_axis` - Physical rotation axis (`'x'`, `'y'`, or `'z'`).
/// * `nblades` - Number of equally spaced instances expected in the periodic set.
///
/// # Returns
/// `(periodic_pairs, outer_faces)` containing the periodic matches and the filtered outer faces.
///
/// # Testing
/// See `tests/test_rotational_periodicity.rs::rotational_periodicity_test` for an end-to-end
/// example that builds the mesh, invokes this routine, and inspects the exported matches.
pub fn rotational_periodicity(
    blocks: &[Block],
    matched_faces: &[FaceMatch],
    outer_faces: &[FaceRecord],
    periodic_direction: &str,
    rotation_axis: char,
    nblades: usize,
) -> (Vec<PeriodicPair>, Vec<FaceRecord>) {
    let rotation_angle = if nblades == 0 {
        0.0
    } else {
        2.0 * PI / nblades as f64
    };
    let rot_forward = create_rotation_matrix(rotation_angle, rotation_axis);
    let rot_backward = create_rotation_matrix(-rotation_angle, rotation_axis);

    let mut periodic_pairs: Vec<(Face, Face)> = Vec::new();
    let mut periodic_exports: Vec<FaceMatch> = Vec::new();

    let mut outer_faces_all = outer_face_records_to_list(blocks, outer_faces, 1);
    let matched_faces_all = match_faces_to_list(blocks, matched_faces, 1);
    let mut seen_pair_keys: HashSet<(FaceKey, FaceKey)> = HashSet::new();

    // ===== PHASE 1: Full face matching with rotation (fast, corner-based) =====
    {
        use crate::block_face_functions::full_face_match_transformed;

        let transform_fwd = |p: [f64; 3]| apply_rotation(p, rot_forward);
        let transform_rev = |p: [f64; 3]| apply_rotation(p, rot_backward);
        let mut consumed_keys: HashSet<FaceKey> = HashSet::new();

        let n = outer_faces_all.len();
        for idx_a in 0..n {
            if consumed_keys.contains(&face_key(&outer_faces_all[idx_a])) {
                continue;
            }
            for idx_b in 0..n {
                if idx_a == idx_b {
                    continue;
                }
                if consumed_keys.contains(&face_key(&outer_faces_all[idx_b])) {
                    continue;
                }
                let face_a = &outer_faces_all[idx_a];
                let face_b = &outer_faces_all[idx_b];

                if !faces_support_direction(face_a, face_b, periodic_direction) {
                    continue;
                }
                if !faces_could_match_rotationally(face_a, face_b, rot_forward, rotation_axis, 0.1)
                    && !faces_could_match_rotationally(
                        face_a,
                        face_b,
                        rot_backward,
                        rotation_axis,
                        0.1,
                    )
                {
                    continue;
                }

                let tol = 1e-6;
                // Try forward rotation
                if let Some(orientation) =
                    full_face_match_transformed(face_a, face_b, &transform_fwd, tol)
                {
                    let key_a = face_key(face_a);
                    let key_b = face_key(face_b);
                    let pair_key = ordered_pair(key_a, key_b);
                    if !seen_pair_keys.contains(&pair_key) {
                        seen_pair_keys.insert(pair_key);
                        consumed_keys.insert(key_a);
                        consumed_keys.insert(key_b);
                        periodic_pairs.push((face_a.clone(), face_b.clone()));
                        periodic_exports.push(FaceMatch {
                            block1: FaceRecord::from_face(face_a),
                            block2: FaceRecord::from_face(face_b),
                            points: Vec::new(),
                            orientation: Some(orientation),
                        });
                        break; // face_a consumed, move to next
                    }
                    continue;
                }
                // Try backward rotation
                if let Some(orientation) =
                    full_face_match_transformed(face_a, face_b, &transform_rev, tol)
                {
                    let key_a = face_key(face_a);
                    let key_b = face_key(face_b);
                    let pair_key = ordered_pair(key_a, key_b);
                    if !seen_pair_keys.contains(&pair_key) {
                        seen_pair_keys.insert(pair_key);
                        consumed_keys.insert(key_a);
                        consumed_keys.insert(key_b);
                        periodic_pairs.push((face_a.clone(), face_b.clone()));
                        periodic_exports.push(FaceMatch {
                            block1: FaceRecord::from_face(face_a),
                            block2: FaceRecord::from_face(face_b),
                            points: Vec::new(),
                            orientation: Some(orientation),
                        });
                        break; // face_a consumed, move to next
                    }
                }
            }
        }

        // Remove Phase 1 consumed faces
        outer_faces_all.retain(|f| !consumed_keys.contains(&face_key(f)));
    }

    // ===== PHASE 2: Partial face matching with splitting (slow) =====
    let mut changed = true;
    let mut iteration = 0usize;
    while changed {
        changed = false;
        iteration += 1;
        let combos: Vec<(usize, usize)> = permutations_indices(outer_faces_all.len());

        let pb = ProgressBar::new(combos.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {pos}/{len} pairs ({eta} remaining)",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_message(format!("Periodicity pass {iteration}"));

        let mut removal_keys: Option<Vec<FaceKey>> = None;
        let mut new_splits: Vec<Face> = Vec::new();
        // The ' is the loop label
        'outer_loop: for (idx_a, idx_b) in combos {
            pb.inc(1);
            if idx_a >= outer_faces_all.len() || idx_b >= outer_faces_all.len() {
                continue;
            }
            let face_a = outer_faces_all[idx_a].clone();
            let face_b = outer_faces_all[idx_b].clone();

            if !faces_support_direction(&face_a, &face_b, periodic_direction) {
                continue;
            }

            // Cheap geometric pre-check
            if !faces_could_match_rotationally(&face_a, &face_b, rot_forward, rotation_axis, 0.1)
                && !faces_could_match_rotationally(
                    &face_a,
                    &face_b,
                    rot_backward,
                    rotation_axis,
                    0.1,
                )
            {
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
            let block_a_rot_fwd = rotate_block_with_matrix(&blocks[block_idx_a], rot_forward);
            if let Some((pair_faces, splits)) =
                periodicity_check(&face_a, &face_b, &block_a_rot_fwd, block_b)
            {
                let pair_key = ordered_pair(face_key(&pair_faces[0]), face_key(&pair_faces[1]));
                if seen_pair_keys.contains(&pair_key) {
                    continue;
                }
                seen_pair_keys.insert(pair_key);
                removal_keys = Some(collect_removal_keys(&face_a, &face_b, &pair_faces));
                periodic_pairs.push((pair_faces[0].clone(), pair_faces[1].clone()));
                periodic_exports.push(FaceMatch {
                    block1: FaceRecord::from_face(&pair_faces[0]),
                    block2: FaceRecord::from_face(&pair_faces[1]),
                    points: Vec::new(),
                    orientation: None,
                });
                new_splits = splits;
                changed = true;
                break 'outer_loop;
            }
            let block_a_rot_rev = rotate_block_with_matrix(&blocks[block_idx_a], rot_backward);
            if let Some((pair_faces, splits)) =
                periodicity_check(&face_a, &face_b, &block_a_rot_rev, block_b)
            {
                let pair_key = ordered_pair(face_key(&pair_faces[0]), face_key(&pair_faces[1]));
                if seen_pair_keys.contains(&pair_key) {
                    continue;
                }
                seen_pair_keys.insert(pair_key);
                removal_keys = Some(collect_removal_keys(&face_a, &face_b, &pair_faces));
                periodic_pairs.push((pair_faces[0].clone(), pair_faces[1].clone()));
                periodic_exports.push(FaceMatch {
                    block1: FaceRecord::from_face(&pair_faces[0]),
                    block2: FaceRecord::from_face(&pair_faces[1]),
                    points: Vec::new(),
                    orientation: None,
                });
                new_splits = splits;
                changed = true;
                break 'outer_loop;
            }
        }
        pb.finish_and_clear();

        if changed {
            if let Some(keys) = removal_keys {
                outer_faces_all = outer_faces_all
                    .into_iter()
                    .filter(|f| !keys.iter().any(|k| face_key(f) == *k))
                    .collect();
            }
            outer_faces_all.extend(new_splits.drain(..));
        }
    }

    let matched_keys: Vec<FaceKey> = matched_faces_all.iter().map(face_key).collect();
    outer_faces_all.retain(|f| !matched_keys.contains(&face_key(f)));

    let mut outer_exports = Vec::new();
    for face in &outer_faces_all {
        outer_exports.push(face.to_record());
    }

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
    rotation_angle_deg: f64,
    rotation_axis: char,
    reduce_mesh: bool,
) -> (Vec<PeriodicPair>, Vec<FaceRecord>) {
    let mut gcd_to_use = 1usize;
    let mut working_blocks: Vec<Block> = blocks.to_vec();
    if reduce_mesh && !blocks.is_empty() {
        gcd_to_use = compute_min_gcd(blocks);
        working_blocks = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    }

    let rotation_angle_rad = rotation_angle_deg.to_radians();
    let rotation_matrix_forward = create_rotation_matrix(rotation_angle_rad, rotation_axis);
    let rotation_matrix_reverse = create_rotation_matrix(-rotation_angle_rad, rotation_axis);

    let rotated_blocks_forward: Vec<Block> = working_blocks
        .iter()
        .map(|b| rotate_block_with_matrix(b, rotation_matrix_forward))
        .collect();
    let rotated_blocks_reverse: Vec<Block> = working_blocks
        .iter()
        .map(|b| rotate_block_with_matrix(b, rotation_matrix_reverse))
        .collect();

    let mut outer_faces_all = outer_face_records_to_list(&working_blocks, outer_faces, gcd_to_use);
    let matched_faces_all = match_faces_to_list(&working_blocks, matched_faces, gcd_to_use);

    // Detect angular boundaries for annular domains
    let (_, _, lower_angular, upper_angular) =
        find_angular_bounding_faces(&working_blocks, &outer_faces_all, rotation_axis, 1e-6);
    let use_angular = !lower_angular.is_empty() && !upper_angular.is_empty();
    let lower_keys: HashSet<FaceKey> = lower_angular.iter().map(face_key).collect();
    let upper_keys: HashSet<FaceKey> = upper_angular.iter().map(face_key).collect();

    let mut periodic_pairs: Vec<(Face, Face)> = Vec::new();
    let mut non_matching: HashSet<(usize, usize)> = HashSet::new();

    // ===== PHASE 1: Full face matching with rotation (fast, corner-based) =====
    {
        use crate::block_face_functions::full_face_match_transformed;

        let transform_fwd = |p: [f64; 3]| apply_rotation(p, rotation_matrix_forward);
        let transform_rev = |p: [f64; 3]| apply_rotation(p, rotation_matrix_reverse);
        let mut consumed_keys: HashSet<FaceKey> = HashSet::new();

        let n = outer_faces_all.len();
        for idx_a in 0..n {
            if consumed_keys.contains(&face_key(&outer_faces_all[idx_a])) {
                continue;
            }
            for idx_b in 0..n {
                if idx_a == idx_b {
                    continue;
                }
                if consumed_keys.contains(&face_key(&outer_faces_all[idx_b])) {
                    continue;
                }
                let face_a = &outer_faces_all[idx_a];
                let face_b = &outer_faces_all[idx_b];

                if !faces_support_any(face_a, face_b) {
                    continue;
                }
                if !faces_could_match_rotationally(
                    face_a,
                    face_b,
                    rotation_matrix_forward,
                    rotation_axis,
                    0.1,
                ) && !faces_could_match_rotationally(
                    face_a,
                    face_b,
                    rotation_matrix_reverse,
                    rotation_axis,
                    0.1,
                ) {
                    continue;
                }

                let tol = 1e-6;
                if let Some(_orientation) =
                    full_face_match_transformed(face_a, face_b, &transform_fwd, tol)
                {
                    consumed_keys.insert(face_key(face_a));
                    consumed_keys.insert(face_key(face_b));
                    periodic_pairs.push((face_a.clone(), face_b.clone()));
                    break;
                }
                if let Some(_orientation) =
                    full_face_match_transformed(face_a, face_b, &transform_rev, tol)
                {
                    consumed_keys.insert(face_key(face_a));
                    consumed_keys.insert(face_key(face_b));
                    periodic_pairs.push((face_a.clone(), face_b.clone()));
                    break;
                }
            }
        }

        // Remove Phase 1 consumed faces
        outer_faces_all.retain(|f| !consumed_keys.contains(&face_key(f)));
    }

    // ===== PHASE 2: Partial face matching with splitting (slow) =====
    let mut periodic_found = true;

    while periodic_found {
        periodic_found = false;
        let combos_all = permutations_indices(outer_faces_all.len());
        // If angular boundaries detected, only pair lower vs upper faces
        let combos: Vec<(usize, usize)> = combos_all
            .into_iter()
            .filter(|&(a, b)| {
                if !use_angular {
                    return true;
                }
                let key_a = face_key(&outer_faces_all[a]);
                let key_b = face_key(&outer_faces_all[b]);
                (lower_keys.contains(&key_a) && upper_keys.contains(&key_b))
                    || (upper_keys.contains(&key_a) && lower_keys.contains(&key_b))
                    // Also allow non-boundary faces to participate
                    || (!lower_keys.contains(&key_a)
                        && !upper_keys.contains(&key_a)
                        && !lower_keys.contains(&key_b)
                        && !upper_keys.contains(&key_b))
            })
            .filter(|pair| !non_matching.contains(pair))
            .collect();
        // Search for a match in parallel using rayon find_first
        let match_result = combos
            .par_iter()
            .find_first(|&&(idx_a, idx_b)| {
                if idx_a >= outer_faces_all.len() || idx_b >= outer_faces_all.len() {
                    return false;
                }
                let face_a = &outer_faces_all[idx_a];
                let face_b = &outer_faces_all[idx_b];

                if !faces_support_any(face_a, face_b) {
                    return false;
                }

                if !faces_could_match_rotationally(
                    face_a,
                    face_b,
                    rotation_matrix_forward,
                    rotation_axis,
                    0.1,
                ) && !faces_could_match_rotationally(
                    face_a,
                    face_b,
                    rotation_matrix_reverse,
                    rotation_axis,
                    0.1,
                ) {
                    return false;
                }

                let block_idx_a = match face_a.block_index() {
                    Some(idx) => idx,
                    None => return false,
                };
                let block_idx_b = match face_b.block_index() {
                    Some(idx) => idx,
                    None => return false,
                };

                if block_idx_a >= working_blocks.len() || block_idx_b >= working_blocks.len() {
                    return false;
                }

                let rotated_fwd = &rotated_blocks_forward[block_idx_a];
                let rotated_rev = &rotated_blocks_reverse[block_idx_a];
                let base = &working_blocks[block_idx_b];

                let valid_face = |face: &Face, block: &Block| -> bool {
                    face.imin() < block.imax
                        && face.imax() < block.imax
                        && face.jmin() < block.jmax
                        && face.jmax() < block.jmax
                        && face.kmin() < block.kmax
                        && face.kmax() < block.kmax
                };

                let fa_valid_fwd = valid_face(face_a, rotated_fwd);
                let fa_valid_rev = valid_face(face_a, rotated_rev);
                if (!fa_valid_fwd && !fa_valid_rev) || !valid_face(face_b, base) {
                    return false;
                }

                if fa_valid_fwd {
                    if periodicity_check(face_a, face_b, rotated_fwd, base).is_some() {
                        return true;
                    }
                }
                if fa_valid_rev {
                    if periodicity_check(face_a, face_b, rotated_rev, base).is_some() {
                        return true;
                    }
                }
                false
            })
            .copied();

        // If a match was found, re-run the matching (cheap: single pair) to get the result
        let mut outer_faces_to_remove: Vec<Face> = Vec::new();
        let mut split_faces: Vec<Face> = Vec::new();

        if let Some((idx_a, idx_b)) = match_result {
            let face_a = outer_faces_all[idx_a].clone();
            let face_b = outer_faces_all[idx_b].clone();

            let block_idx_a = face_a.block_index().unwrap();
            let block_idx_b = face_b.block_index().unwrap();

            let rotated_fwd = &rotated_blocks_forward[block_idx_a];
            let rotated_rev = &rotated_blocks_reverse[block_idx_a];
            let base = &working_blocks[block_idx_b];

            let matched = periodicity_check(&face_a, &face_b, rotated_fwd, base)
                .or_else(|| periodicity_check(&face_a, &face_b, rotated_rev, base));

            if let Some((pair_faces, splits)) = matched {
                periodic_pairs.push((pair_faces[0].clone(), pair_faces[1].clone()));
                outer_faces_to_remove.push(face_a);
                outer_faces_to_remove.push(face_b);
                outer_faces_to_remove.extend(pair_faces.into_iter());
                split_faces.extend(splits);
                periodic_found = true;
            }
        }

        if periodic_found {
            let removal_keys: HashSet<FaceKey> =
                outer_faces_to_remove.iter().map(face_key).collect();

            outer_faces_all = outer_faces_all
                .into_iter()
                .filter(|face| !removal_keys.contains(&face_key(face)))
                .collect();

            if !split_faces.is_empty() {
                outer_faces_all.extend(split_faces.into_iter());
            }

            non_matching.clear();
        }
    }

    let mut removal_keys: HashSet<FaceKey> = matched_faces_all.iter().map(face_key).collect();

    for (face_a, face_b) in &periodic_pairs {
        removal_keys.insert(face_key(face_a));
        removal_keys.insert(face_key(face_b));
    }
    outer_faces_all.retain(|face| !removal_keys.contains(&face_key(face)));

    // Remove duplicate periodic pairs (order-insensitive)
    let mut dedup: HashSet<(FaceKey, FaceKey)> = HashSet::new();
    periodic_pairs.retain(|(a, b)| {
        let key = ordered_pair(face_key(a), face_key(b));
        dedup.insert(key)
    });

    let mut periodic_exports: Vec<FaceMatch> = periodic_pairs
        .into_iter()
        .map(|(a, b)| FaceMatch {
            block1: FaceRecord::from_face(&a),
            block2: FaceRecord::from_face(&b),
            points: Vec::new(),
            orientation: None,
        })
        .collect();

    let mut outer_export: Vec<FaceRecord> = outer_faces_all.iter().map(Face::to_record).collect();

    if gcd_to_use > 1 {
        for rec in &mut periodic_exports {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }
        for dict in &mut outer_export {
            dict.scale_indices(gcd_to_use);
        }
    }

    (periodic_exports, outer_export)
}

/// Build a comparable key from face indices and block identifier.
#[inline]
fn face_key(face: &Face) -> FaceKey {
    face.index_key()
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
fn faces_support_direction(face_a: &Face, face_b: &Face, direction: &str) -> bool {
    let dir = direction.trim().to_ascii_lowercase();
    match dir.as_str() {
        "i" => face_a.imin() == face_a.imax() && face_b.imin() == face_b.imax(),
        "j" => face_a.jmin() == face_a.jmax() && face_b.jmin() == face_b.jmax(),
        "k" => face_a.kmin() == face_a.kmax() && face_b.kmin() == face_b.kmax(),
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
fn faces_support_any(face_a: &Face, face_b: &Face) -> bool {
    (face_a.imin() == face_a.imax() && face_b.imin() == face_b.imax())
        || (face_a.jmin() == face_a.jmax() && face_b.jmin() == face_b.jmax())
        || (face_a.kmin() == face_a.kmax() && face_b.kmin() == face_b.kmax())
}

/// Cheap geometric pre-check to reject obviously non-matching face pairs for
/// rotational periodicity. Performs four tests in order of cost:
///
/// 1. Same `const_type` (I/J/K-constant).
/// 2. Axial extent overlap along rotation axis.
/// 3. Radial extent overlap perpendicular to rotation axis.
/// 4. Rotated centroid proximity (within 1.5x max diagonal).
fn faces_could_match_rotationally(
    face1: &Face,
    face2: &Face,
    rotation_matrix: [[f64; 3]; 3],
    rotation_axis: char,
    tol_rel: f64,
) -> bool {
    // 1. Same const_type
    let ct1 = face1.const_type();
    let ct2 = face2.const_type();
    if ct1 != ct2 || ct1 == -1 {
        return false;
    }

    // 2. Axial extent overlap
    let (f1_ax_min, f1_ax_max) = face_axis_extent(face1, rotation_axis);
    let (f2_ax_min, f2_ax_max) = face_axis_extent(face2, rotation_axis);
    let axial_span = (f1_ax_max - f1_ax_min).max(f2_ax_max - f2_ax_min).max(1e-12);
    let tol_axial = tol_rel * axial_span;
    if f1_ax_max + tol_axial < f2_ax_min || f2_ax_max + tol_axial < f1_ax_min {
        return false;
    }

    // 3. Radial extent overlap
    let (r1_min, r1_max) = face_radial_extent(face1, rotation_axis);
    let (r2_min, r2_max) = face_radial_extent(face2, rotation_axis);
    let radial_span = (r1_max - r1_min).max(r2_max - r2_min).max(1e-12);
    let tol_radial = tol_rel * radial_span;
    if r1_max + tol_radial < r2_min || r2_max + tol_radial < r1_min {
        return false;
    }

    // 4. Rotated centroid proximity
    let c1 = face1.centroid();
    let c1_rot = apply_rotation(c1, rotation_matrix);
    let c2 = face2.centroid();
    let dist = distance3(c1_rot, c2);
    let max_diag = face1.diagonal_length().max(face2.diagonal_length());
    if dist > max_diag * 1.5 {
        return false;
    }

    true
}

/// Axial coordinate extent of a face along the given rotation axis.
fn face_axis_extent(face: &Face, axis: char) -> (f64, f64) {
    let idx = match axis.to_ascii_lowercase() {
        'x' => 0,
        'y' => 1,
        _ => 2,
    };
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for v in face.vertices() {
        min_v = min_v.min(v[idx]);
        max_v = max_v.max(v[idx]);
    }
    (min_v, max_v)
}

/// Radial extent of a face perpendicular to the given rotation axis.
fn face_radial_extent(face: &Face, axis: char) -> (f64, f64) {
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    for v in face.vertices() {
        let r = to_radius(v[0], v[1], v[2], axis);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
    }
    (min_r, max_r)
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
    keys.push(face_key(face_a));
    keys.push(face_key(face_b));
    for f in pair_faces {
        keys.push(face_key(f));
    }
    keys.sort();
    keys.dedup();
    keys
}

/// Attempt to intersect two faces after rotation and return the matching subfaces when successful.
///
/// # Arguments
/// * `face1`, `face2` - Faces inspected for overlap.
/// * `block1`, `block2` - Blocks providing geometric detail for each face.
///
/// # Returns
/// `Some((matched_faces, splits))` when an overlap exists, where `matched_faces` contains the
/// oriented interface pair and `splits` lists any child faces created during splitting. Returns
/// `None` when the faces do not meet the matching criteria.
fn periodicity_check(
    face1: &Face,
    face2: &Face,
    block1: &Block,
    block2: &Block,
) -> Option<(Vec<Face>, Vec<Face>)> {
    debug_assert!(face1.imin() < block1.imax);
    debug_assert!(face1.jmin() < block1.jmax);
    debug_assert!(face1.kmin() < block1.kmax);
    debug_assert!(face1.imax() < block1.imax);
    debug_assert!(face1.jmax() < block1.jmax);
    debug_assert!(face1.kmax() < block1.kmax);
    debug_assert!(face2.imin() < block2.imax);
    debug_assert!(face2.jmin() < block2.jmax);
    debug_assert!(face2.kmin() < block2.kmax);
    debug_assert!(face2.imax() < block2.imax);
    debug_assert!(face2.jmax() < block2.jmax);
    debug_assert!(face2.kmax() < block2.kmax);
    let mut face_a = face1.clone();
    let mut face_b = face2.clone();
    let mut swapped = false;
    if face_b.diagonal_length() < face_a.diagonal_length() {
        std::mem::swap(&mut face_a, &mut face_b);
        swapped = true;
    }

    let (matches, mut split1, split2) =
        get_face_intersection(&face_a, &face_b, block1, block2, MATCH_TOL);
    if matches.len() < 4 {
        return None;
    }

    let bounds_a = match_bounds(&matches, true);
    let bounds_b = match_bounds(&matches, false);

    let mut out1 = create_face_from_diagonals(
        block1, bounds_a.0, bounds_a.2, bounds_a.4, bounds_a.1, bounds_a.3, bounds_a.5,
    );
    out1.set_block_index(face_a.block_index().unwrap_or(usize::MAX));
    if let Some(id) = face_a.id() {
        out1.set_id(id);
    }

    let mut out2 = create_face_from_diagonals(
        block2, bounds_b.0, bounds_b.2, bounds_b.4, bounds_b.1, bounds_b.3, bounds_b.5,
    );
    out2.set_block_index(face_b.block_index().unwrap_or(usize::MAX));
    if let Some(id) = face_b.id() {
        out2.set_id(id);
    }

    split1.extend(split2);

    let pair = if swapped {
        vec![out2.clone(), out1.clone()]
    } else {
        vec![out1.clone(), out2.clone()]
    };

    Some((pair, split1))
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
    matches: &[crate::connectivity::MatchPoint],
    first: bool,
) -> (usize, usize, usize, usize, usize, usize) {
    let mut imin = usize::MAX;
    let mut jmin = usize::MAX;
    let mut kmin = usize::MAX;
    let mut imax = 0usize;
    let mut jmax = 0usize;
    let mut kmax = 0usize;
    for m in matches {
        let (i, j, k) = if first {
            (m.i1, m.j1, m.k1)
        } else {
            (m.i2, m.j2, m.k2)
        };
        imin = imin.min(i);
        jmin = jmin.min(j);
        kmin = kmin.min(k);
        imax = imax.max(i);
        jmax = jmax.max(j);
        kmax = kmax.max(k);
    }
    (imin, imax, jmin, jmax, kmin, kmax)
}

const MATCH_TOL: f64 = 1e-6;

/// Generate all permutations `(i, j)` for `len`, excluding pairs where `i == j`.
///
/// # Arguments
/// * `len` - Number of elements to permute.
///
/// # Returns
/// Vector of ordered index pairs suitable for exhaustive comparisons.
fn permutations_indices(len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..len {
        for j in 0..len {
            if i != j {
                out.push((i, j));
            }
        }
    }
    out
}

/// Compute the rotation angle and matrix from `face1` to `face2`.
///
/// This assumes the rotation axis is the x-direction, which is suitable
/// for faces within the same turbomachinery passage.
///
/// Reference: Linear Real Transforms from GlennHT (`M_ccMBMesh.F`, `computeLRT`).
///
/// # Arguments
/// * `face1` - Source face.
/// * `face2` - Target face.
///
/// # Returns
/// `(angle_radians, rotation_matrix_3x3)`. Returns `(0.0, zeros)` when the
/// faces are already aligned.
pub fn linear_real_transform(face1: &Face, face2: &Face) -> (f64, [[f64; 3]; 3]) {
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

    let cos_ang =
        (n_to[1] * n_from[1] + n_to[2] * n_from[2]) / (denom_to * denom_from);
    let sin_ang =
        (n_to[2] * n_from[1] - n_to[1] * n_from[2]) / (denom_to * denom_from);
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

/// Verify periodic face matches by checking diagonal corners after rotation.
///
/// Same algorithm as [`crate::connectivity::verify_connectivity`] but rotates
/// block1 by `±theta` before checking spatial consistency.
///
/// # Arguments
/// * `blocks` - Full-resolution blocks.
/// * `face_matches` - Periodic face matches to verify.
/// * `theta` - Rotation angle in radians.
/// * `rotation_axis` - Axis of rotation (`'x'`, `'y'`, or `'z'`).
/// * `tol` - Euclidean distance tolerance for corner matching.
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_periodicity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    theta: f64,
    rotation_axis: char,
    tol: f64,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    // Compute GCD and reduce blocks
    let gcd_to_use = compute_min_gcd(blocks);

    let reduced = reduce_blocks(blocks, gcd_to_use);

    // Build rotation matrices for +theta and -theta
    let rot_pos = create_rotation_matrix(theta, rotation_axis);
    let rot_neg = create_rotation_matrix(-theta, rotation_axis);

    // Pre-rotate all reduced blocks in both directions
    let rotated_pos: Vec<Block> = reduced.iter().map(|b| rotate_block(b, rot_pos)).collect();
    let rotated_neg: Vec<Block> = reduced.iter().map(|b| rotate_block(b, rot_neg)).collect();

    // Scale down face_match indices by GCD
    let mut scaled_matches: Vec<FaceMatch> = face_matches.to_vec();
    for fm in &mut scaled_matches {
        fm.divide_indices(gcd_to_use);
    }

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();

    for (idx, fm) in scaled_matches.iter().enumerate() {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= reduced.len() || b2.block_index >= reduced.len() {
            mismatched.push(face_matches[idx].clone());
            continue;
        }

        let block2 = &reduced[b2.block_index];

        // Fast path: if orientation is known from Phase 1, trust the match
        if fm.orientation.is_some() {
            verified.push(face_matches[idx].clone());
            continue;
        }

        // Slow path: no orientation — enumerate corner permutations with rotations
        let i_vals = [b2.imin, b2.imax];
        let j_vals = [b2.jmin, b2.jmax];
        let k_vals = [b2.kmin, b2.kmax];

        let mut unique_corners: Vec<(usize, usize, usize)> = Vec::new();
        {
            let mut seen = HashSet::new();
            for &i in &i_vals {
                for &j in &j_vals {
                    for &k in &k_vals {
                        if seen.insert((i, j, k)) {
                            unique_corners.push((i, j, k));
                        }
                    }
                }
            }
        }

        let mut found = false;
        let mut best_d_lower = f64::INFINITY;
        let mut best_d_upper = f64::INFINITY;

        // Try +theta rotation first, then -theta
        for rotated_blocks in [&rotated_pos, &rotated_neg] {
            if found {
                break;
            }

            let block1_rotated = &rotated_blocks[b1.block_index];

            // Block1 rotated diagonal coordinates
            let (x1_l, y1_l, z1_l) = block1_rotated.xyz(b1.imin, b1.jmin, b1.kmin);
            let (x1_u, y1_u, z1_u) = block1_rotated.xyz(b1.imax, b1.jmax, b1.kmax);

            // Check stored diagonal first
            let (x2_l, y2_l, z2_l) = block2.xyz(b2.imin, b2.jmin, b2.kmin);
            let (x2_u, y2_u, z2_u) = block2.xyz(b2.imax, b2.jmax, b2.kmax);

            let d_lower = ((x2_l - x1_l).powi(2) + (y2_l - y1_l).powi(2) + (z2_l - z1_l).powi(2)).sqrt();
            let d_upper = ((x2_u - x1_u).powi(2) + (y2_u - y1_u).powi(2) + (z2_u - z1_u).powi(2)).sqrt();

            if d_lower < best_d_lower {
                best_d_lower = d_lower;
            }
            if d_upper < best_d_upper {
                best_d_upper = d_upper;
            }

            if d_lower < tol && d_upper < tol {
                verified.push(face_matches[idx].clone());
                found = true;
                break;
            }

            // Try all permutations of block2's corners
            for &(il, jl, kl) in &unique_corners {
                for &(iu, ju, ku) in &unique_corners {
                    if (il, jl, kl) == (iu, ju, ku) {
                        continue;
                    }

                    let (x2_l, y2_l, z2_l) = block2.xyz(il, jl, kl);
                    let (x2_u, y2_u, z2_u) = block2.xyz(iu, ju, ku);

                    let dl = ((x2_l - x1_l).powi(2) + (y2_l - y1_l).powi(2) + (z2_l - z1_l).powi(2)).sqrt();
                    let du = ((x2_u - x1_u).powi(2) + (y2_u - y1_u).powi(2) + (z2_u - z1_u).powi(2)).sqrt();

                    if dl < best_d_lower {
                        best_d_lower = dl;
                    }
                    if du < best_d_upper {
                        best_d_upper = du;
                    }

                    if dl < tol && du < tol {
                        let mut corrected = face_matches[idx].clone();
                        corrected.block2.imin = il * gcd_to_use;
                        corrected.block2.jmin = jl * gcd_to_use;
                        corrected.block2.kmin = kl * gcd_to_use;
                        corrected.block2.imax = iu * gcd_to_use;
                        corrected.block2.jmax = ju * gcd_to_use;
                        corrected.block2.kmax = ku * gcd_to_use;
                        verified.push(corrected);
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
        }

        if !found {
            eprintln!(
                "verify_periodicity: MISMATCH at face_match index {}",
                idx
            );
            eprintln!(
                "  block1 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                face_matches[idx].block1.block_index,
                face_matches[idx].block1.imin, face_matches[idx].block1.jmin, face_matches[idx].block1.kmin,
                face_matches[idx].block1.imax, face_matches[idx].block1.jmax, face_matches[idx].block1.kmax,
            );
            eprintln!(
                "  block2 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                face_matches[idx].block2.block_index,
                face_matches[idx].block2.imin, face_matches[idx].block2.jmin, face_matches[idx].block2.kmin,
                face_matches[idx].block2.imax, face_matches[idx].block2.jmax, face_matches[idx].block2.kmax,
            );
            eprintln!(
                "  Closest rotated block1 corner dist to block2 lower: {:.6e}",
                best_d_lower
            );
            eprintln!(
                "  Closest rotated block1 corner dist to block2 upper: {:.6e}",
                best_d_upper
            );
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}
