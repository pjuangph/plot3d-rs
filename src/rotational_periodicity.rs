//! Utilities for detecting rotational periodicity in structured multi-block grids.
//!
//! This module mirrors the behaviour of the original Python tooling and is covered end-to-end by
//! the integration test in `tests/test_rotational_periodicity.rs`. Generate HTML documentation with
//! `cargo doc --open` to browse rendered versions of these notes alongside the Rust API surface.

use std::collections::HashSet;
use std::f64::consts::PI;

use crate::{
    block::Block,
    block_face_functions::{
        create_face_from_diagonals, match_faces_to_list, outer_face_records_to_list, Face,
    },
    connectivity::{get_face_intersection, FaceMatch, FaceRecord},
    utils::gcd_three,
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
    let mut gcds = Vec::new();
    for block in blocks {
        gcds.push(gcd_three(block.imax - 1, block.jmax - 1, block.kmax - 1));
    }
    let gcd_to_use = gcds.into_iter().min().unwrap_or(1).max(1);

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

    let mut changed = true;
    while changed {
        changed = false;
        let combos: Vec<(usize, usize)> = permutations_indices(outer_faces_all.len());

        let mut removal_keys: Option<Vec<FaceKey>> = None;
        let mut new_splits: Vec<Face> = Vec::new();
        // The ' is the loop label
        'outer_loop: for (idx_a, idx_b) in combos {
            if idx_a >= outer_faces_all.len() || idx_b >= outer_faces_all.len() {
                continue;
            }
            let face_a = outer_faces_all[idx_a].clone();
            let face_b = outer_faces_all[idx_b].clone();

            if !faces_support_direction(&face_a, &face_b, periodic_direction) {
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
                });
                new_splits = splits;
                changed = true;
                break 'outer_loop;
            }
        }

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
        let mut gcds = Vec::with_capacity(blocks.len());
        for block in blocks {
            gcds.push(gcd_three(block.imax - 1, block.jmax - 1, block.kmax - 1));
        }
        gcd_to_use = gcds.into_iter().min().unwrap_or(1).max(1);
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

    let mut periodic_pairs: Vec<(Face, Face)> = Vec::new();
    let mut non_matching: HashSet<(usize, usize)> = HashSet::new();
    let mut periodic_found = true;

    while periodic_found {
        periodic_found = false;
        let combos_all = permutations_indices(outer_faces_all.len());
        let combos: Vec<(usize, usize)> = combos_all
            .into_iter()
            .filter(|pair| !non_matching.contains(pair))
            .collect();
        let mut outer_faces_to_remove: Vec<Face> = Vec::new();
        let mut split_faces: Vec<Face> = Vec::new();

        'combo_loop: for (idx_a, idx_b) in combos {
            if idx_a >= outer_faces_all.len() || idx_b >= outer_faces_all.len() {
                continue;
            }

            let face_a = outer_faces_all[idx_a].clone();
            let face_b = outer_faces_all[idx_b].clone();

            if !faces_support_any(&face_a, &face_b) {
                non_matching.insert((idx_a, idx_b));
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

            if block_idx_a >= working_blocks.len() || block_idx_b >= working_blocks.len() {
                continue;
            }

            let rotated_block_forward = &rotated_blocks_forward[block_idx_a];
            let rotated_block_reverse = &rotated_blocks_reverse[block_idx_a];
            let base_block = &working_blocks[block_idx_b];

            let valid_face = |face: &Face, block: &Block| -> bool {
                face.imin() < block.imax
                    && face.imax() < block.imax
                    && face.jmin() < block.jmax
                    && face.jmax() < block.jmax
                    && face.kmin() < block.kmax
                    && face.kmax() < block.kmax
            };

            let face_a_valid_forward = valid_face(&face_a, rotated_block_forward);
            let face_a_valid_reverse = valid_face(&face_a, rotated_block_reverse);
            if (!face_a_valid_forward && !face_a_valid_reverse) || !valid_face(&face_b, base_block)
            {
                non_matching.insert((idx_a, idx_b));
                continue;
            }

            let mut matched = None;
            if face_a_valid_forward {
                if let Some((pair_faces, splits)) =
                    periodicity_check(&face_a, &face_b, rotated_block_forward, base_block)
                {
                    matched = Some((pair_faces, splits));
                }
            }
            if matched.is_none() && face_a_valid_reverse {
                if let Some((pair_faces, splits)) =
                    periodicity_check(&face_a, &face_b, rotated_block_reverse, base_block)
                {
                    matched = Some((pair_faces, splits));
                }
            }

            if let Some((pair_faces, splits)) = matched {
                periodic_pairs.push((pair_faces[0].clone(), pair_faces[1].clone()));
                outer_faces_to_remove.push(face_a);
                outer_faces_to_remove.push(face_b);
                outer_faces_to_remove.extend(pair_faces.into_iter());
                split_faces.extend(splits);
                periodic_found = true;
                break 'combo_loop;
            }

            non_matching.insert((idx_a, idx_b));
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

type FaceKey = (usize, usize, usize, usize, usize, usize, usize);

/// Build a comparable key from face indices and block identifier.
///
/// # Arguments
/// * `face` - Face used to derive the key.
///
/// # Returns
/// Tuple identifying the face location and owning block.
fn face_key(face: &Face) -> FaceKey {
    (
        face.block_index().unwrap_or(usize::MAX),
        face.imin(),
        face.jmin(),
        face.kmin(),
        face.imax(),
        face.jmax(),
        face.kmax(),
    )
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
