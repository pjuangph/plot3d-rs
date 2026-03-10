//! Block-to-block face connectivity detection.
//!
//! The core data types ([`FaceRecord`], [`FaceMatch`], [`MatchPoint`],
//! [`Orientation`]) live in [`crate::face_record`]. This module provides the
//! algorithms that populate them.
//!
//! # Three-Phase Connectivity Algorithm
//!
//! The [`connectivity_fast`] function detects face matches in three phases:
//!
//! **Phase 1 -- Full-face matching using canonical grids + permutation matrices**
//!
//! For every pair of outer faces, corner comparison is used as a quick
//! pre-filter. Candidate matches are then verified by extracting both
//! faces as canonical 2D grids (ascending index order) and trying all 8
//! [`PERMUTATION_MATRICES`] on face B. When `Face_B * permutation == Face_A`
//! within tolerance, the match is confirmed and the winning `permutation_index`
//! is stored in [`FaceMatch::orientation`].
//!
//! **Phase 2 -- Partial / split-face node-by-node matching**
//!
//! Remaining unmatched faces are tested for partial overlap via
//! [`get_face_intersection`]. When a partial match is found, both faces are
//! split along the intersection boundary. The matched sub-faces produce
//! [`FaceMatch`] records, while the leftover remnants are fed back into the
//! pool for subsequent iterations. This loop continues until no new matches
//! are discovered during a full pass.
//!
//! **Phase 3 -- Fresh-face validation with all-node AABB pre-checks**
//!
//! After Phase 2 converges, any remaining unmatched faces may still have
//! partial overlaps with faces that were *already matched* in Phases 1 or 2.
//! Phase 3 re-examines these by computing axis-aligned bounding boxes
//! (AABBs) using **all face nodes** (not just corners), then testing AABB
//! overlap before calling `get_face_intersection`. This catches edge cases
//! where two corners of a sub-face lie outside the bounding diagonal of
//! a candidate face, which the 2-corner AABB would miss.
//!
//! # Post-processing
//!
//! After matching, [`align_face_orientations`] can optionally correct the
//! diagonal corners of `block2` in each [`FaceMatch`] so that they encode
//! the detected orientation. This embeds the permutation directly into the
//! `il/jl/kl` and `ih/jh/kh` fields, which is the format expected by
//! solvers that use the GridPro/GlennHT diagonal convention.
//!
//! # Tolerance
//!
//! The default spatial tolerance used for vertex comparisons is
//! [`DEFAULT_TOL`] (1e-6). Both [`connectivity`] and [`connectivity_fast`]
//! accept an explicit tolerance parameter to override this default.
//!
//! # Verification
//!
//! Use [`verify_connectivity`] after running connectivity to confirm that
//! all matched face pairs have coincident nodes.

use std::collections::{HashMap, HashSet};

use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    block::Block,
    block_face_functions::{create_face_from_diagonals, get_outer_faces, split_face, Face},
    face_record::{match_point_bounds, FaceKey, FaceMatch, FaceRecord, MatchPoint, Orientation},
    verification::{extract_canonical_grid, try_all_permutations},
    Float,
};

const DEFAULT_TOL: Float = 1e-6;

/// Structured-grid node on a face, capturing indices and XYZ coordinate.
#[derive(Clone, Debug)]
struct FaceNode {
    i: usize,
    j: usize,
    k: usize,
    coord: [Float; 3],
}

/// Enumerate all nodes that belong to `face` on `block`.
///
/// # Arguments
/// * `face` - Face whose nodes should be sampled.
/// * `block` - Parent block providing Cartesian coordinates.
///
/// # Returns
/// Vector of [`FaceNode`] containing structured indices `(i, j, k)` and the
/// corresponding XYZ coordinate.
fn face_nodes(face: &Face, block: &Block) -> Vec<FaceNode> {
    let mut nodes = Vec::new();
    let i_vals: Vec<usize> = if face.imin() == face.imax() {
        vec![face.imin()]
    } else {
        (face.imin()..=face.imax()).collect()
    };
    let j_vals: Vec<usize> = if face.jmin() == face.jmax() {
        vec![face.jmin()]
    } else {
        (face.jmin()..=face.jmax()).collect()
    };
    let k_vals: Vec<usize> = if face.kmin() == face.kmax() {
        vec![face.kmin()]
    } else {
        (face.kmin()..=face.kmax()).collect()
    };
    for &i in &i_vals {
        for &j in &j_vals {
            for &k in &k_vals {
                if !(i < block.imax && j < block.jmax && k < block.kmax) {
                    continue;
                }
                let (x, y, z) = block.xyz(i, j, k);
                nodes.push(FaceNode {
                    i,
                    j,
                    k,
                    coord: [x, y, z],
                });
            }
        }
    }
    nodes
}

/// Locate the node whose coordinate is within `tol` of `target`.
///
/// Returns the first node that meets the tolerance, preferring the closest
/// distance. When no node matches, `None` is returned.
fn find_closest_node(nodes: &[FaceNode], target: [Float; 3], tol: Float) -> Option<&FaceNode> {
    let mut best: Option<(&FaceNode, Float)> = None;
    for node in nodes {
        let dx = node.coord[0] - target[0];
        let dy = node.coord[1] - target[1];
        let dz = node.coord[2] - target[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= tol {
            match best {
                Some((_, best_dist)) if dist >= best_dist => {}
                _ => best = Some((node, dist)),
            }
        }
    }
    best.map(|(node, _)| node)
}

/// Check whether the coincident nodes degenerate to an edge contact.
fn is_edge(points: &[MatchPoint]) -> bool {
    if points.is_empty() {
        return false;
    }
    let (i_lo, i_hi, j_lo, j_hi, k_lo, k_hi) = match_point_bounds(points, true);
    let const_count =
        usize::from(i_lo == i_hi) + usize::from(j_lo == j_hi) + usize::from(k_lo == k_hi);
    const_count >= 2
}

/// Filter matches so the provided key advances monotonically by 1.
///
/// When there are exactly 2 unique values, they are always kept regardless of
/// gap size. This handles the case where a small face (e.g. 2 nodes wide after
/// GCD reduction) matches a large face — the matching indices on the large face
/// may span a wide gap (e.g. [0, 113]) but are still a valid match. The
/// `is_edge()` check upstream has already verified this isn't a degenerate edge.
fn filter_block_increasing(
    points: &[MatchPoint],
    key: fn(&MatchPoint) -> usize,
) -> Vec<MatchPoint> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut unique_vals: Vec<usize> = points.iter().map(key).collect();
    unique_vals.sort_unstable();
    unique_vals.dedup();
    if unique_vals.len() <= 1 {
        return Vec::new();
    }
    // With only 2 unique values, contiguity is trivially satisfied — keep all.
    if unique_vals.len() == 2 {
        return points.to_vec();
    }
    let mut keep: HashSet<usize> = HashSet::new();
    for window in unique_vals.windows(2) {
        if window[1] == window[0] + 1 {
            keep.insert(window[0]);
            keep.insert(window[1]);
        }
    }
    points
        .iter()
        .filter(|p| keep.contains(&key(p)))
        .cloned()
        .collect()
}

/// Enforce monotonic progression along the non-constant axes of each face.
fn apply_axis_filters(points: Vec<MatchPoint>, face1: &Face, face2: &Face) -> Vec<MatchPoint> {
    let mut filtered = points;
    match face1.const_axis() {
        Some(crate::block_face_functions::FaceAxis::I) => {
            filtered = filter_block_increasing(&filtered, |p| p.j1);
            filtered = filter_block_increasing(&filtered, |p| p.k1);
        }
        Some(crate::block_face_functions::FaceAxis::J) => {
            filtered = filter_block_increasing(&filtered, |p| p.i1);
            filtered = filter_block_increasing(&filtered, |p| p.k1);
        }
        Some(crate::block_face_functions::FaceAxis::K) => {
            filtered = filter_block_increasing(&filtered, |p| p.i1);
            filtered = filter_block_increasing(&filtered, |p| p.j1);
        }
        None => {}
    }
    match face2.const_axis() {
        Some(crate::block_face_functions::FaceAxis::I) => {
            filtered = filter_block_increasing(&filtered, |p| p.j2);
            filtered = filter_block_increasing(&filtered, |p| p.k2);
        }
        Some(crate::block_face_functions::FaceAxis::J) => {
            filtered = filter_block_increasing(&filtered, |p| p.i2);
            filtered = filter_block_increasing(&filtered, |p| p.k2);
        }
        Some(crate::block_face_functions::FaceAxis::K) => {
            filtered = filter_block_increasing(&filtered, |p| p.i2);
            filtered = filter_block_increasing(&filtered, |p| p.j2);
        }
        None => {}
    }
    filtered
}

/// Build subfaces produced by the intersection region.
fn create_split_faces(
    face: &Face,
    block: &Block,
    points: &[MatchPoint],
    use_block1: bool,
) -> Vec<Face> {
    if points.is_empty() {
        return Vec::new();
    }
    let (i_lo, i_hi, j_lo, j_hi, k_lo, k_hi) = match_point_bounds(points, use_block1);
    let degeneracy =
        usize::from(i_lo == i_hi) + usize::from(j_lo == j_hi) + usize::from(k_lo == k_hi);
    if degeneracy != 1 {
        return Vec::new();
    }
    let mut split = split_face(face, block, i_lo, j_lo, k_lo, i_hi, j_hi, k_hi);
    for f in &mut split {
        if let Some(idx) = face.block_index() {
            f.set_block_index(idx);
        }
        if let Some(id) = face.id() {
            f.set_id(id);
        }
    }
    split
}

/// Compute the coincident nodes between two faces on separate blocks.
///
/// # Arguments
/// * `face1` - Candidate face on `block1`.
/// * `face2` - Candidate face on `block2`.
/// * `block1` / `block2` - Parent blocks.
/// * `tol` - Euclidean tolerance for node matching.
///
/// # Returns
/// Tuple containing:
/// 1. List of [`MatchPoint`]s.
/// 2. Split faces generated on `block1`.
/// 3. Split faces generated on `block2`.
pub fn get_face_intersection(
    face1: &Face,
    face2: &Face,
    block1: &Block,
    block2: &Block,
    tol: Float,
) -> (Vec<MatchPoint>, Vec<Face>, Vec<Face>) {
    let nodes1 = face_nodes(face1, block1);
    let nodes2 = face_nodes(face2, block2);
    let mut matches = Vec::new();
    for node1 in &nodes1 {
        if let Some(node2) = find_closest_node(&nodes2, node1.coord, tol) {
            matches.push(MatchPoint {
                i1: node1.i,
                j1: node1.j,
                k1: node1.k,
                i2: node2.i,
                j2: node2.j,
                k2: node2.k,
            });
        }
    }
    if matches.len() < 4 || is_edge(&matches) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let matches = apply_axis_filters(matches, face1, face2);
    if matches.len() < 4 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let split_faces1 = create_split_faces(face1, block1, &matches, true);
    let split_faces2 = create_split_faces(face2, block2, &matches, false);
    (matches, split_faces1, split_faces2)
}

// ---------------------------------------------------------------------------
// Orientation-aware MatchPoint generation
// ---------------------------------------------------------------------------

use crate::block_face_functions::FaceAxis;

/// Extract the (u, v) index ranges for a face based on its constant axis.
fn face_uv_ranges(
    face: &Face,
    axis: FaceAxis,
) -> (
    std::ops::RangeInclusive<usize>,
    std::ops::RangeInclusive<usize>,
) {
    match axis {
        FaceAxis::I => (face.jmin()..=face.jmax(), face.kmin()..=face.kmax()),
        FaceAxis::J => (face.imin()..=face.imax(), face.kmin()..=face.kmax()),
        FaceAxis::K => (face.imin()..=face.imax(), face.jmin()..=face.jmax()),
    }
}

/// Convert parametric (u, v) back to structured (i, j, k) given the constant axis.
fn uv_to_ijk(u: usize, v: usize, axis: FaceAxis, face: &Face) -> (usize, usize, usize) {
    match axis {
        FaceAxis::I => (face.imin(), u, v), // u=j, v=k
        FaceAxis::J => (u, face.jmin(), v), // u=i, v=k
        FaceAxis::K => (u, v, face.kmin()), // u=i, v=j
    }
}

/// Given a full face match with known orientation, enumerate all corresponding
/// node pairs by walking both grids in lock-step.
///
/// This avoids the O(N*M) closest-node search used for partial matches.
fn build_match_points_from_orientation(
    face1: &Face,
    face2: &Face,
    orientation: &Orientation,
) -> Vec<MatchPoint> {
    let Some(axis1) = face1.const_axis() else {
        return Vec::new();
    };
    let Some(axis2) = face2.const_axis() else {
        return Vec::new();
    };

    let (u1_range, v1_range) = face_uv_ranges(face1, axis1);
    let (u2_range, v2_range) = face_uv_ranges(face2, axis2);

    let u1_vals: Vec<usize> = u1_range.collect();
    let v1_vals: Vec<usize> = v1_range.collect();
    let u2_vals: Vec<usize> = u2_range.collect();
    let v2_vals: Vec<usize> = v2_range.collect();

    let mut points = Vec::with_capacity(u1_vals.len() * v1_vals.len());

    for (u_off, &u1) in u1_vals.iter().enumerate() {
        for (v_off, &v1) in v1_vals.iter().enumerate() {
            // Apply orientation mapping to get face2's (u, v) offsets
            let (u2_off, v2_off) = if orientation.swapped() {
                (v_off, u_off)
            } else {
                (u_off, v_off)
            };

            let u2_idx = if orientation.u_reversed() {
                u2_vals.len().saturating_sub(1).saturating_sub(u2_off)
            } else {
                u2_off
            };
            let v2_idx = if orientation.v_reversed() {
                v2_vals.len().saturating_sub(1).saturating_sub(v2_off)
            } else {
                v2_off
            };

            if u2_idx >= u2_vals.len() || v2_idx >= v2_vals.len() {
                continue;
            }

            let (i1, j1, k1) = uv_to_ijk(u1, v1, axis1, face1);
            let (i2, j2, k2) = uv_to_ijk(u2_vals[u2_idx], v2_vals[v2_idx], axis2, face2);

            points.push(MatchPoint {
                i1,
                j1,
                k1,
                i2,
                j2,
                k2,
            });
        }
    }
    points
}

// ---------------------------------------------------------------------------
// Phase 1: Fast full-face matching using corner comparison
// ---------------------------------------------------------------------------

/// Phase 1: Fast full-face matching using canonical grid + permutation matrices.
///
/// For each candidate block pair, uses corner comparison as a quick pre-filter,
/// then verifies the match using `extract_canonical_grid` + `try_all_permutations`.
/// Face A is the first face, Face B is multiplied by the permutation matrix,
/// and all points are compared within tolerance.
///
/// Returns `(matches, consumed_face_keys)`.
fn find_full_face_matches(
    blocks: &[Block],
    block_outer_faces: &[Vec<Face>],
    candidate_pairs: &[(usize, usize)],
    tol: Float,
) -> (Vec<FaceMatch>, HashSet<FaceKey>) {
    use crate::block_face_functions::full_face_match;
    use crate::verification::{extract_canonical_grid, try_all_permutations};

    let mut face_matches = Vec::new();
    let mut consumed: HashSet<FaceKey> = HashSet::new();

    for &(i, j) in candidate_pairs {
        for face_i in &block_outer_faces[i] {
            if consumed.contains(&face_i.index_key()) {
                continue;
            }
            for face_j in &block_outer_faces[j] {
                if consumed.contains(&face_j.index_key()) {
                    continue;
                }
                // Quick corner pre-filter
                if full_face_match(face_i, face_j, tol).is_none() {
                    continue;
                }

                // Build FaceRecords for canonical grid extraction
                let rec_a = FaceRecord::from_face(face_i);
                let rec_b = FaceRecord::from_face(face_j);

                // Extract canonical grids: Face A and Face B
                let (pts_a, nu_a, nv_a) = match extract_canonical_grid(&blocks[i], &rec_a) {
                    Some(g) => g,
                    None => continue,
                };
                let (pts_b, nu_b, nv_b) = match extract_canonical_grid(&blocks[j], &rec_b) {
                    Some(g) => g,
                    None => continue,
                };

                // Face B * permutation matrix, then compare with Face A
                if let Some(perm_idx) =
                    try_all_permutations(&pts_a, nu_a, nv_a, &pts_b, nu_b, nv_b, tol)
                {
                    let orientation = Orientation::from_perm_index(
                        perm_idx,
                        rec_a.constant_axis(),
                        rec_b.constant_axis(),
                    );

                    // Build match points from the verified orientation
                    let points = build_match_points_from_orientation(face_i, face_j, &orientation);

                    consumed.insert(face_i.index_key());
                    consumed.insert(face_j.index_key());

                    face_matches.push(FaceMatch {
                        block1: rec_a,
                        block2: rec_b,
                        points,
                        orientation: Some(orientation),
                    });
                    break; // face_i consumed, move on
                }
            }
        }
    }

    (face_matches, consumed)
}

// ---------------------------------------------------------------------------
// Phase 2: Slow partial-face matching with node-by-node comparison
// ---------------------------------------------------------------------------

/// Recursively match all faces between a pair of blocks.
///
/// # Arguments
/// * `block1` / `block2` - Blocks to compare.
/// * `block1_outer` / `block2_outer` - Mutable outer-face lists that will be
///   updated in-place as faces are split.
/// * `tol` - Node matching tolerance.
///
/// # Returns
/// Collection of match-point arrays, one entry per detected interface.
pub fn find_matching_blocks(
    block1: &Block,
    block2: &Block,
    block1_outer: &mut Vec<Face>,
    block2_outer: &mut Vec<Face>,
    tol: Float,
) -> Vec<Vec<MatchPoint>> {
    let mut matches = Vec::new();
    let mut i = 0;
    'outer: while i < block1_outer.len() {
        let mut j = 0;
        while j < block2_outer.len() {
            let face1 = block1_outer[i].clone();
            let face2 = block2_outer[j].clone();
            let (match_points, split1, split2) =
                get_face_intersection(&face1, &face2, block1, block2, tol);
            if !match_points.is_empty() {
                matches.push(match_points.clone());

                block1_outer.remove(i);
                block2_outer.remove(j);
                block1_outer.extend(split1);
                block2_outer.extend(split2);
                i = 0;
                continue 'outer;
            } else {
                j += 1;
            }
        }
        i += 1;
    }
    matches
}

/// Return `(i, j)` block index pairs whose axis-aligned bounding boxes overlap
/// or nearly touch within `tol`.
///
/// This replaces the former centroid-distance approach which only considered
/// the 6 nearest blocks and could miss neighbours for L-shaped or elongated
/// geometries.  AABB overlap is both more robust and more correct.
///
/// # Arguments
/// * `blocks` - All blocks in the assembly.
/// * `tol` - AABB expansion tolerance.  Blocks whose bounding boxes are within
///   this distance of touching are still considered candidates.
///
/// # Returns
/// Vector of `(i, j)` pairs with `i < j`.
fn candidate_neighbor_pairs(blocks: &[Block], tol: Float) -> Vec<(usize, usize)> {
    use rayon::prelude::*;

    let n = blocks.len();
    // Precompute AABBs: [xmin, xmax, ymin, ymax, zmin, zmax]
    let aabbs: Vec<[Float; 6]> = blocks
        .par_iter()
        .map(|b| {
            let mut xmin = Float::INFINITY;
            let mut xmax = Float::NEG_INFINITY;
            let mut ymin = Float::INFINITY;
            let mut ymax = Float::NEG_INFINITY;
            let mut zmin = Float::INFINITY;
            let mut zmax = Float::NEG_INFINITY;
            for &x in &b.x {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
            }
            for &y in &b.y {
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
            for &z in &b.z {
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
            [xmin, xmax, ymin, ymax, zmin, zmax]
        })
        .collect();

    let pairs: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let aabbs = &aabbs;
            ((i + 1)..n)
                .filter_map(move |j| {
                    let a = &aabbs[i];
                    let b = &aabbs[j];
                    if a[1] + tol >= b[0]
                        && b[1] + tol >= a[0]
                        && a[3] + tol >= b[2]
                        && b[3] + tol >= a[2]
                        && a[5] + tol >= b[4]
                        && b[5] + tol >= a[4]
                    {
                        Some((i, j))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    pairs
}

/// Connectivity computation performed on GCD-reduced blocks.
///
/// This is the main entry point for connectivity detection. It down-samples
/// all blocks by the minimum GCD of their dimensions, runs the three-phase
/// connectivity algorithm (see module docs), then scales indices back to
/// the original resolution.
///
/// # Arguments
/// * `blocks` - Original block list. Each block is down-sampled by the
///   smallest index GCD across the set.
///
/// # Returns
/// Tuple `(matches, outer_faces)` where `matches` enumerates face interfaces
/// and `outer_faces` records the remaining external surfaces at the original
/// resolution.
pub fn connectivity_fast(blocks: &[Block]) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    let gcd_to_use = crate::utils::compute_min_gcd(blocks);
    let reduced_blocks = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    let (mut matches, mut outer_faces) = connectivity(&reduced_blocks);
    // Scale back to original size
    for face in &mut matches {
        face.block1.scale_indices(gcd_to_use);
        face.block2.scale_indices(gcd_to_use);
    }
    for face in &mut outer_faces {
        face.scale_indices(gcd_to_use);
    }
    (matches, outer_faces)
}

/// Determine face-to-face connectivity and exterior faces for all blocks.
///
/// # Arguments
/// * `blocks` - Full-resolution blocks to analyse.
///
/// # Returns
/// Tuple `(matches, outer_faces)` representing matched interfaces and the
/// formatted list of outer faces.
pub fn connectivity(blocks: &[Block]) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    use rayon::prelude::*;

    // Parallelize outer face extraction per block.
    // Extract outer faces for each block. Degenerate face pairs (where
    // opposite sides coincide) are NOT added to the inter-block pool —
    // they are handled as self-matches later in this function (lines 856+).
    // This matches the Python behavior where get_outer_faces returns
    // (non_matching, matching) and only non_matching enters the pool.
    let mut block_outer_faces: Vec<Vec<Face>> = blocks
        .par_iter()
        .enumerate()
        .map(|(idx, block)| {
            let (faces, _degenerate_pairs) = get_outer_faces(block);
            faces
                .into_iter()
                .map(|mut f| {
                    f.set_block_index(idx);
                    f
                })
                .collect()
        })
        .collect();

    let combos = candidate_neighbor_pairs(blocks, DEFAULT_TOL);

    // ===== PHASE 1: Full face matching (fast, corner-based + interior verification) =====
    let (mut matches, consumed_keys) =
        find_full_face_matches(blocks, &block_outer_faces, &combos, DEFAULT_TOL);

    // Remove fully-matched faces from the outer face pools
    for faces in &mut block_outer_faces {
        faces.retain(|f| !consumed_keys.contains(&f.index_key()));
    }

    let mut matches_to_remove: HashSet<FaceKey> = consumed_keys;

    // ===== PHASE 2: Partial face matching (slow, node-by-node) =====
    // Iterate until convergence: after splitting faces for one block pair,
    // the remnants may match faces from other blocks that were previously
    // consumed. Repeat until no new matches are found.
    let mut phase2_round = 0;
    let mut phase2_changed = true;
    while phase2_changed {
        phase2_changed = false;
        phase2_round += 1;

        let pb = ProgressBar::new(combos.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {pos}/{len} pairs ({eta} remaining)",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_message(format!(
            "Connectivity (partial matching, round {})",
            phase2_round
        ));

        for &(i, j) in &combos {
            pb.inc(1);
            // candidate_neighbor_pairs guarantees i < j
            let (left, right) = block_outer_faces.split_at_mut(j);
            let (left, right) = (&mut left[i], &mut right[0]);

            // Skip if either block has no remaining unmatched faces
            if left.is_empty() || right.is_empty() {
                continue;
            }

            let mut match_points =
                find_matching_blocks(&blocks[i], &blocks[j], left, right, DEFAULT_TOL);
            for points in match_points.drain(..) {
                phase2_changed = true;
                let (i1lo, i1hi, j1lo, j1hi, k1lo, k1hi) = match_point_bounds(&points, true);
                let mut face1 =
                    create_face_from_diagonals(&blocks[i], i1lo, j1lo, k1lo, i1hi, j1hi, k1hi);
                face1.set_block_index(i);
                let (i2lo, i2hi, j2lo, j2hi, k2lo, k2hi) = match_point_bounds(&points, false);
                let mut face2 =
                    create_face_from_diagonals(&blocks[j], i2lo, j2lo, k2lo, i2hi, j2hi, k2hi);
                face2.set_block_index(j);
                matches_to_remove.insert(face1.index_key());
                matches_to_remove.insert(face2.index_key());

                let corner1 = FaceRecord::from_match_points(i, &points, true).unwrap();
                let corner2 = FaceRecord::from_match_points(j, &points, false).unwrap();
                matches.push(FaceMatch {
                    block1: corner1,
                    block2: corner2,
                    points,
                    orientation: None,
                });
            }
        }
        pb.finish_with_message(format!(
            "Connectivity round {} done (changed={})",
            phase2_round, phase2_changed
        ));
    }

    let mut outer_faces = Vec::new();
    for faces in &block_outer_faces {
        for face in faces {
            outer_faces.push(face.clone());
        }
    }
    // Free large temporaries now that we've extracted what we need
    drop(block_outer_faces);

    let mut seen = HashSet::new();
    outer_faces.retain(|face| seen.insert(face.index_key()));

    outer_faces.retain(|face| !matches_to_remove.contains(&face.index_key()));
    drop(matches_to_remove);

    let mut outer_faces_to_remove = HashSet::new();
    let mut by_block: HashMap<usize, Vec<&Face>> = HashMap::new();
    for face in &outer_faces {
        if let Some(idx) = face.block_index() {
            by_block.entry(idx).or_default().push(face);
        }
    }
    for faces in by_block.values() {
        for (a_idx, face_a) in faces.iter().enumerate() {
            let dims_a = [
                face_a.imin(),
                face_a.jmin(),
                face_a.kmin(),
                face_a.imax(),
                face_a.jmax(),
                face_a.kmax(),
            ];
            for (b_idx, face_b) in faces.iter().enumerate() {
                if a_idx == b_idx {
                    continue;
                }
                let dims_b = [
                    face_b.imin(),
                    face_b.jmin(),
                    face_b.kmin(),
                    face_b.imax(),
                    face_b.jmax(),
                    face_b.kmax(),
                ];
                let equal_components = dims_a
                    .iter()
                    .zip(dims_b.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                if equal_components == 5 {
                    let remove_key = if face_b.diagonal_length() > face_a.diagonal_length() {
                        face_b.index_key()
                    } else {
                        face_a.index_key()
                    };
                    outer_faces_to_remove.insert(remove_key);
                }
            }
        }
    }

    outer_faces.retain(|face| !outer_faces_to_remove.contains(&face.index_key()));

    let mut self_match_keys: HashSet<FaceKey> = HashSet::new();
    for (idx, block) in blocks.iter().enumerate() {
        let (_, self_matches) = get_outer_faces(block);
        for (face_a, face_b) in self_matches {
            let mut corner1 = FaceRecord {
                block_index: idx,
                il: face_a.imin(),
                jl: face_a.jmin(),
                kl: face_a.kmin(),
                ih: face_a.imax(),
                jh: face_a.jmax(),
                kh: face_a.kmax(),
                id: face_a.id(),
                u_physical: None,
                v_physical: None,
            };
            let corner2 = FaceRecord {
                block_index: idx,
                il: face_b.imin(),
                jl: face_b.jmin(),
                kl: face_b.kmin(),
                ih: face_b.imax(),
                jh: face_b.jmax(),
                kh: face_b.kmax(),
                id: face_b.id(),
                u_physical: None,
                v_physical: None,
            };
            // Track self-matched face keys so they can be removed from outer faces
            let mut fa = face_a.clone();
            fa.set_block_index(idx);
            let mut fb = face_b.clone();
            fb.set_block_index(idx);
            self_match_keys.insert(fa.index_key());
            self_match_keys.insert(fb.index_key());

            corner1.id = face_a.id();
            matches.push(FaceMatch {
                block1: corner1,
                block2: corner2,
                points: Vec::new(),
                orientation: None,
            });
        }
    }

    // Remove self-matched faces from outer faces
    outer_faces.retain(|face| !self_match_keys.contains(&face.index_key()));

    // ===== PHASE 3: Fresh-face validation for remaining outer faces =====
    // Some outer faces remain unmatched because the matching block's face pool
    // was consumed by an earlier combo in Phases 1–2.  Re-check each remaining
    // outer face against *fresh* (un-consumed) outer faces of overlapping blocks.
    {
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
        for &(i, j) in &combos {
            neighbors[i].push(j);
            neighbors[j].push(i);
        }

        // Precompute fresh outer faces and their AABBs for every block.
        // We compute the AABB from ALL face nodes (not just 2 diagonal corners)
        // because large curved faces can have interior extents far beyond their
        // corner coordinates.
        let fresh_all: Vec<Vec<Face>> = blocks
            .iter()
            .map(|block| {
                let (faces, _) = get_outer_faces(block);
                faces
            })
            .collect();

        // Precompute [xmin, xmax, ymin, ymax, zmin, zmax] for each fresh face.
        let fresh_aabbs: Vec<Vec<[Float; 6]>> = blocks
            .iter()
            .zip(fresh_all.iter())
            .map(|(block, faces)| {
                faces
                    .iter()
                    .map(|f| {
                        let nodes = face_nodes(f, block);
                        let mut aabb = [
                            Float::INFINITY,
                            Float::NEG_INFINITY,
                            Float::INFINITY,
                            Float::NEG_INFINITY,
                            Float::INFINITY,
                            Float::NEG_INFINITY,
                        ];
                        for n in &nodes {
                            aabb[0] = aabb[0].min(n.coord[0]);
                            aabb[1] = aabb[1].max(n.coord[0]);
                            aabb[2] = aabb[2].min(n.coord[1]);
                            aabb[3] = aabb[3].max(n.coord[1]);
                            aabb[4] = aabb[4].min(n.coord[2]);
                            aabb[5] = aabb[5].max(n.coord[2]);
                        }
                        aabb
                    })
                    .collect()
            })
            .collect();

        let pb = ProgressBar::new(outer_faces.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta} remaining)",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_message("Connectivity Phase 3 (fresh-face validation)");

        let mut phase3_keys: HashSet<FaceKey> = HashSet::new();

        for face in outer_faces.iter() {
            pb.inc(1);
            if phase3_keys.contains(&face.index_key()) {
                continue;
            }
            let bi = match face.block_index() {
                Some(v) => v,
                None => continue,
            };

            // Compute proper AABB of this face from all its nodes
            let face_nodes_list = face_nodes(face, &blocks[bi]);
            let mut fxn = Float::INFINITY;
            let mut fxx = Float::NEG_INFINITY;
            let mut fyn = Float::INFINITY;
            let mut fyx = Float::NEG_INFINITY;
            let mut fzn = Float::INFINITY;
            let mut fzx = Float::NEG_INFINITY;
            for n in &face_nodes_list {
                fxn = fxn.min(n.coord[0]);
                fxx = fxx.max(n.coord[0]);
                fyn = fyn.min(n.coord[1]);
                fyx = fyx.max(n.coord[1]);
                fzn = fzn.min(n.coord[2]);
                fzx = fzx.max(n.coord[2]);
            }

            for &bj in &neighbors[bi] {
                for (fi, ff) in fresh_all[bj].iter().enumerate() {
                    // AABB pre-check using precomputed all-node AABBs
                    let gaabb = &fresh_aabbs[bj][fi];
                    let tol_pre = 0.01;
                    if fxx + tol_pre < gaabb[0]
                        || gaabb[1] + tol_pre < fxn
                        || fyx + tol_pre < gaabb[2]
                        || gaabb[3] + tol_pre < fyn
                        || fzx + tol_pre < gaabb[4]
                        || gaabb[5] + tol_pre < fzn
                    {
                        continue;
                    }

                    let (pts, _, _) =
                        get_face_intersection(face, ff, &blocks[bi], &blocks[bj], DEFAULT_TOL);
                    if pts.is_empty() {
                        continue;
                    }
                    if let (Some(c1), Some(c2)) = (
                        FaceRecord::from_match_points(bi, &pts, true),
                        FaceRecord::from_match_points(bj, &pts, false),
                    ) {
                        matches.push(FaceMatch {
                            block1: c1,
                            block2: c2,
                            points: pts,
                            orientation: None,
                        });
                    }
                    phase3_keys.insert(face.index_key());
                    // Don't break — continue checking other neighbors' faces
                    // for split-face matches where multiple blocks cover parts
                    // of the same remaining face.
                }
            }
        }

        let n3 = phase3_keys.len();
        pb.finish_with_message(format!("Phase 3 done ({n3} new matches)"));

        outer_faces.retain(|f| !phase3_keys.contains(&f.index_key()));
    }

    let mut formatted = Vec::new();
    let mut id_counter = 1;
    for face in outer_faces {
        formatted.push(FaceRecord {
            block_index: face.block_index().unwrap_or(usize::MAX),
            il: face.imin(),
            jl: face.jmin(),
            kl: face.kmin(),
            ih: face.imax(),
            jh: face.jmax(),
            kh: face.kmax(),
            id: Some(id_counter),
            u_physical: None,
            v_physical: None,
        });
        id_counter += 1;
    }

    (matches, formatted)
}

/// Find the correct orientation for each face match using permutation matrices.
///
/// Extracts canonical grids for both faces and tries all 8 permutation
/// matrices on face B. Stores the winning `permutation_index` and plane
/// type in [`FaceMatch::orientation`].
///
/// # Arguments
/// * `blocks` - Block array providing geometry.
/// * `face_matches` - Verified face matches (from `verify_connectivity`).
/// * `tol` - Euclidean distance tolerance.
///
/// # Returns
/// `(aligned, rejected)` — aligned matches with corrected block2 diagonals,
/// and rejected matches where no orientation produces point-by-point equality.
pub fn align_face_orientations(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let mut aligned = Vec::new();
    let mut rejected = Vec::new();

    let pb = ProgressBar::new(face_matches.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} matches ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("Align orientations");

    for fm in face_matches {
        pb.inc(1);
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            rejected.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // Extract canonical 2D grids for both faces
        let grid_a = match extract_canonical_grid(block1, b1) {
            Some(g) => g,
            None => {
                rejected.push(fm.clone());
                continue;
            }
        };
        let grid_b = match extract_canonical_grid(block2, b2) {
            Some(g) => g,
            None => {
                rejected.push(fm.clone());
                continue;
            }
        };

        let (pts_a, nu_a, nv_a) = grid_a;
        let (pts_b, nu_b, nv_b) = grid_b;

        // Try all 8 permutation matrices to find the matching orientation
        if let Some(perm_idx) = try_all_permutations(&pts_a, nu_a, nv_a, &pts_b, nu_b, nv_b, tol) {
            let mut fm_out = fm.clone();
            fm_out.orientation = Some(Orientation::from_perm_index(
                perm_idx,
                b1.constant_axis(),
                b2.constant_axis(),
            ));
            aligned.push(fm_out);
        } else {
            eprintln!(
                "  align: REJECTED block {}↔{} — no permutation matches",
                b1.block_index, b2.block_index
            );
            rejected.push(fm.clone());
        }
    }

    pb.finish_with_message("Align orientations done");
    (aligned, rejected)
}

/// Derive lb/ub for both faces from the first/last MatchPoint.
///
/// The traversal order of MatchPoints encodes the orientation relationship
/// between the two faces, which is lost if we use min/max or spatial proximity.
fn derive_diagonal_from_match_points(
    fm: &FaceMatch,
    gcd: usize,
) -> Option<(FaceRecord, FaceRecord)> {
    let points = &fm.points;
    if points.is_empty() {
        return None;
    }

    let first = &points[0];
    let last = &points[points.len() - 1];

    let b1 = FaceRecord {
        block_index: fm.block1.block_index,
        il: first.i1 * gcd,
        jl: first.j1 * gcd,
        kl: first.k1 * gcd,
        ih: last.i1 * gcd,
        jh: last.j1 * gcd,
        kh: last.k1 * gcd,
        id: fm.block1.id,
        u_physical: None,
        v_physical: None,
    };
    let b2 = FaceRecord {
        block_index: fm.block2.block_index,
        il: first.i2 * gcd,
        jl: first.j2 * gcd,
        kl: first.k2 * gcd,
        ih: last.i2 * gcd,
        jh: last.j2 * gcd,
        kh: last.k2 * gcd,
        id: fm.block2.id,
        u_physical: None,
        v_physical: None,
    };

    Some((b1, b2))
}

/// Establish diagonal correspondence for face-match records.
///
/// Two paths depending on match origin:
///
/// **Phase 2/3 matches** (have MatchPoint data): Derives diagonal corners
/// from the first/last MatchPoint traversal indices. MatchPoint indices
/// live on the GCD-reduced grid, so they are scaled up by the mesh GCD
/// factor to produce full-resolution `FaceRecord` indices.
///
/// **Phase 1 / self-matches** (no MatchPoint data): Falls back to spatial
/// proximity search over block2's bounding-box corners to find the corner
/// nearest block1's lower-bound corner.
///
/// The resulting matches carry oriented `il/jl/kl` and `ih/jh/kh` corners
/// suitable for downstream verification via [`crate::verification::verify_connectivity`],
/// which uses all 8 permutation matrices to determine the correct orientation.
///
/// # Arguments
/// * `blocks` - Block array providing geometry (full resolution).
/// * `face_matches` - Matches to validate.
///
/// # Returns
/// Validated face matches with corrected diagonal indices.
pub fn face_matches_to_dict(blocks: &[Block], face_matches: &[FaceMatch]) -> Vec<FaceMatch> {
    // GCD factor: MatchPoint indices are on the reduced grid while FaceRecord
    // indices have already been scaled up by this factor.
    let gcd = crate::utils::compute_min_gcd(blocks);

    let mut matched_count = 0usize;
    let mut empty_count = 0usize;
    let mut spatial_count = 0usize;

    let result: Vec<FaceMatch> = face_matches
        .iter()
        .filter_map(|fm| {
            let b1 = &fm.block1;
            let b2 = &fm.block2;

            let block1 = blocks.get(b1.block_index)?;
            let block2 = blocks.get(b2.block_index)?;

            let mut result = fm.clone();

            if !fm.points.is_empty() {
                // Has MatchPoint data — use spatial proximity derivation
                if let Some((b1_new, b2_new)) = derive_diagonal_from_match_points(fm, gcd) {
                    result.block1 = b1_new;
                    result.block2 = b2_new;
                    matched_count += 1;
                } else {
                    empty_count += 1;
                }
            } else {
                // No MatchPoints (self-matches): spatial search
                // over block2's bounding-box corners.
                let (x1_l, y1_l, z1_l) = block1.xyz(b1.il, b1.jl, b1.kl);

                let i_vals = [b2.i_lo(), b2.i_hi()];
                let j_vals = [b2.j_lo(), b2.j_hi()];
                let k_vals = [b2.k_lo(), b2.k_hi()];

                let mut best_lower = (Float::MAX, b2.il, b2.jl, b2.kl);
                for &i in &i_vals {
                    for &j in &j_vals {
                        for &k in &k_vals {
                            let (x2, y2, z2) = block2.xyz(i, j, k);
                            let d =
                                ((x2 - x1_l).powi(2) + (y2 - y1_l).powi(2) + (z2 - z1_l).powi(2))
                                    .sqrt();
                            if d < best_lower.0 {
                                best_lower = (d, i, j, k);
                            }
                        }
                    }
                }
                result.block2.il = best_lower.1;
                result.block2.jl = best_lower.2;
                result.block2.kl = best_lower.3;

                let (x1_u, y1_u, z1_u) = block1.xyz(b1.ih, b1.jh, b1.kh);

                let mut best_upper = (Float::MAX, b2.ih, b2.jh, b2.kh);
                for &i in &i_vals {
                    for &j in &j_vals {
                        for &k in &k_vals {
                            let (x2, y2, z2) = block2.xyz(i, j, k);
                            let d =
                                ((x2 - x1_u).powi(2) + (y2 - y1_u).powi(2) + (z2 - z1_u).powi(2))
                                    .sqrt();
                            if d < best_upper.0 {
                                best_upper = (d, i, j, k);
                            }
                        }
                    }
                }
                result.block2.ih = best_upper.1;
                result.block2.jh = best_upper.2;
                result.block2.kh = best_upper.3;

                spatial_count += 1;
            }

            Some(result)
        })
        .collect();

    eprintln!(
        "  face_matches_to_dict: {} matched, {} empty, {} spatial",
        matched_count, empty_count, spatial_count
    );
    result
}
