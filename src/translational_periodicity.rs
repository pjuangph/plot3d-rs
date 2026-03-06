//! Translational periodicity detection for structured multi-block grids.
//!
//! Identifies periodic face pairs along a translational axis (x, y, or z).
//! The algorithm uses [`find_bounding_faces`] to collect faces on the min/max
//! extremes of the specified axis, then matches them using
//! `full_face_match_transformed` with a translation offset.
//!
//! [`find_bounding_faces`]: crate::block_analysis::find_bounding_faces
//!
//! While there is not yet a dedicated Rust integration test, the `tests/test_rotational_periodicity.rs`
//! fixture demonstrates the expected data flow for the periodicity modules and should be referenced
//! when extending this module. Run `cargo doc --open` to view these notes alongside the generated API
//! documentation.

use std::collections::HashSet;

use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    block::Block,
    block_analysis::find_bounding_faces,
    block_face_functions::{full_face_match_transformed, outer_face_records_to_list, Face},
    face_record::{FaceKey, FaceMatch, FaceRecord, Orientation, OrientationPlane},
    utils::compute_min_gcd,
    Float,
};

/// Default spatial tolerance for bounding-face detection and corner matching.
const DEFAULT_TOL: Float = 1e-6;

/// Minimum floor for the adaptive node-matching tolerance.
const ADAPTIVE_TOL_FLOOR: Float = 1e-4;

/// Compute corrected lb2/ub2 and orientation for a periodic face pair.
///
/// Mirrors Python's `_compute_periodic_lb_ub_orientation`:
///   1. Shifts face1's lb/ub corners by `shift_amount` along `shift_axis`.
///   2. Builds all face2 grid points within the face2 index range.
///   3. Finds the nearest face2 point to each shifted face1 corner (brute-force).
///   4. Computes the orientation vector by stepping along each face1 axis and
///      querying which face2 axis changes.
///
/// Returns `(corrected_lb2, corrected_ub2, orientation)` where indices are
/// `[i, j, k]` and orientation is Python's 1-indexed `[a, b, c]` vector.
fn compute_periodic_lb_ub_orientation(
    blk1: &Block,
    lb1: [usize; 3],
    ub1: [usize; 3],
    blk2: &Block,
    lb2_orig: [usize; 3],
    ub2_orig: [usize; 3],
    shift_axis: usize,
    shift_amount: Float,
) -> ([usize; 3], [usize; 3], [usize; 3]) {
    // face1 lb and ub corners (shifted)
    let (x1l, y1l, z1l) = blk1.xyz(lb1[0], lb1[1], lb1[2]);
    let (x1u, y1u, z1u) = blk1.xyz(ub1[0], ub1[1], ub1[2]);
    let mut p1_lb = [x1l, y1l, z1l];
    let mut p1_ub = [x1u, y1u, z1u];
    p1_lb[shift_axis] += shift_amount;
    p1_ub[shift_axis] += shift_amount;

    // face2 index range
    let lo2 = [
        lb2_orig[0].min(ub2_orig[0]),
        lb2_orig[1].min(ub2_orig[1]),
        lb2_orig[2].min(ub2_orig[2]),
    ];
    let hi2 = [
        lb2_orig[0].max(ub2_orig[0]),
        lb2_orig[1].max(ub2_orig[1]),
        lb2_orig[2].max(ub2_orig[2]),
    ];

    // Build all face2 grid points
    let mut indices2: Vec<[usize; 3]> = Vec::new();
    let mut coords2: Vec<[Float; 3]> = Vec::new();
    for i in lo2[0]..=hi2[0] {
        for j in lo2[1]..=hi2[1] {
            for k in lo2[2]..=hi2[2] {
                indices2.push([i, j, k]);
                let (x, y, z) = blk2.xyz(i, j, k);
                coords2.push([x, y, z]);
            }
        }
    }

    // Nearest-neighbor search (brute-force, face grids are small at GCD resolution)
    let nearest = |query: &[Float; 3]| -> usize {
        let mut best_idx = 0;
        let mut best_dist = Float::INFINITY;
        for (i, c) in coords2.iter().enumerate() {
            let d =
                (c[0] - query[0]).powi(2) + (c[1] - query[1]).powi(2) + (c[2] - query[2]).powi(2);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    };

    let corrected_lb2 = indices2[nearest(&p1_lb)];
    let corrected_ub2 = indices2[nearest(&p1_ub)];

    // Compute orientation: step along each face1 axis, find which face2 axis changes
    let dims1 = [
        (ub1[0] as isize - lb1[0] as isize).unsigned_abs() + 1,
        (ub1[1] as isize - lb1[1] as isize).unsigned_abs() + 1,
        (ub1[2] as isize - lb1[2] as isize).unsigned_abs() + 1,
    ];
    let step1: [isize; 3] = [
        if ub1[0] >= lb1[0] { 1 } else { -1 },
        if ub1[1] >= lb1[1] { 1 } else { -1 },
        if ub1[2] >= lb1[2] { 1 } else { -1 },
    ];
    let cdims2 = [
        (corrected_ub2[0] as isize - corrected_lb2[0] as isize).unsigned_abs() + 1,
        (corrected_ub2[1] as isize - corrected_lb2[1] as isize).unsigned_abs() + 1,
        (corrected_ub2[2] as isize - corrected_lb2[2] as isize).unsigned_abs() + 1,
    ];

    let mut orientation = [0usize; 3];
    for d1 in 0..3 {
        if dims1[d1] == 1 {
            // Constant axis on face1 → find constant axis on face2
            for d2 in 0..3 {
                if cdims2[d2] == 1 {
                    orientation[d1] = d2 + 1;
                    break;
                }
            }
        } else {
            // Step one index along face1 axis d1
            let mut next_idx1 = [lb1[0] as isize, lb1[1] as isize, lb1[2] as isize];
            next_idx1[d1] += step1[d1];
            let (nx, ny, nz) = blk1.xyz(
                next_idx1[0] as usize,
                next_idx1[1] as usize,
                next_idx1[2] as usize,
            );
            let mut p1_next = [nx, ny, nz];
            p1_next[shift_axis] += shift_amount;
            let face2_next = indices2[nearest(&p1_next)];
            // Find which face2 axis changed → that's the face2 axis for face1 axis d1
            for d2 in 0..3 {
                if face2_next[d2] != corrected_lb2[d2] && cdims2[d2] > 1 {
                    orientation[d1] = d2 + 1;
                    break;
                }
            }
        }
    }

    // Fill any missing entries (sanity fallback)
    let used: HashSet<usize> = orientation.iter().copied().filter(|&v| v != 0).collect();
    if used.len() < 3 {
        let missing_d1: Vec<usize> = (0..3).filter(|&d| orientation[d] == 0).collect();
        let missing_d2: Vec<usize> = (1..=3).filter(|d| !used.contains(d)).collect();
        for (d1, d2) in missing_d1.iter().zip(missing_d2.iter()) {
            orientation[*d1] = *d2;
        }
    }

    (corrected_lb2, corrected_ub2, orientation)
}

/// Get the coordinate value along `axis_idx` (0=x, 1=y, 2=z) at a given IJK index.
#[inline]
fn block_axis_val(block: &Block, ijk: [usize; 3], axis_idx: usize) -> Float {
    let (x, y, z) = block.xyz(ijk[0], ijk[1], ijk[2]);
    match axis_idx {
        0 => x,
        1 => y,
        _ => z,
    }
}

/// Convert the Python-style orientation vector `[a, b, c]` (1-indexed face2 axis
/// per face1 axis) into the Rust `Orientation` struct by examining the corrected
/// lb2/ub2 step directions relative to face1's lb1/ub1.
fn orientation_from_orient_vec(
    orient_vec: &[usize; 3],
    lb1: &[usize; 3],
    ub1: &[usize; 3],
    corrected_lb2: &[usize; 3],
    corrected_ub2: &[usize; 3],
) -> Orientation {
    // Find the two varying axes on face1 (dims > 1)
    let dims1: [usize; 3] = [
        (ub1[0] as isize - lb1[0] as isize).unsigned_abs() + 1,
        (ub1[1] as isize - lb1[1] as isize).unsigned_abs() + 1,
        (ub1[2] as isize - lb1[2] as isize).unsigned_abs() + 1,
    ];
    let varying1: Vec<usize> = (0..3).filter(|&d| dims1[d] > 1).collect();
    if varying1.len() != 2 {
        return Orientation::from_flags(false, false, false, OrientationPlane::InPlane);
    }

    // Determine constant axes for plane classification
    let const_axis1 = (0..3).find(|&d| dims1[d] == 1).unwrap_or(0);
    let dims2: [usize; 3] = [
        (corrected_ub2[0] as isize - corrected_lb2[0] as isize).unsigned_abs() + 1,
        (corrected_ub2[1] as isize - corrected_lb2[1] as isize).unsigned_abs() + 1,
        (corrected_ub2[2] as isize - corrected_lb2[2] as isize).unsigned_abs() + 1,
    ];
    let const_axis2 = (0..3).find(|&d| dims2[d] == 1).unwrap_or(0);
    let plane = if const_axis1 == const_axis2 {
        OrientationPlane::InPlane
    } else {
        OrientationPlane::CrossPlane
    };

    // Face1 varying axes are u (first) and v (second)
    let u1 = varying1[0];
    let v1 = varying1[1];
    // Each maps to a face2 axis via orient_vec
    let u2 = orient_vec[u1].wrapping_sub(1); // 0-indexed
    let v2 = orient_vec[v1].wrapping_sub(1);

    // Check if face2's u and v axes are swapped relative to face1's
    let swapped = u2 != u1 || v2 != v1;

    // Check reversal: face1 step direction vs face2 step direction
    let step1 = |d: usize| -> isize {
        if ub1[d] >= lb1[d] {
            1
        } else {
            -1
        }
    };
    let step2 = |d: usize| -> isize {
        if corrected_ub2[d] >= corrected_lb2[d] {
            1
        } else {
            -1
        }
    };

    let u_reversed = step1(u1) != step2(u2);
    let v_reversed = step1(v1) != step2(v2);

    Orientation::from_flags(u_reversed, v_reversed, swapped, plane)
}

/// Detect translational periodicity along an axis.
/// Discover translational periodicity along a chosen axis.
///
/// # Testing
/// End-to-end validation is planned to follow the pattern established in
/// `tests/test_rotational_periodicity.rs`. Until then, exercising this function in a binary or
/// ad-hoc script is recommended to mirror the original Python examples.
#[allow(clippy::too_many_arguments)]
pub fn translational_periodicity(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    delta: Option<Float>,
    translational_direction: &str,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let axis = translational_direction.trim().to_ascii_lowercase();
    assert!(matches!(axis.as_str(), "x" | "y" | "z"));
    let axis_idx: usize = match axis.as_str() {
        "x" => 0,
        "y" => 1,
        _ => 2,
    };

    let (lower_faces_records, upper_faces_records, _, _) =
        find_bounding_faces(blocks, outer_faces, &axis, "both", DEFAULT_TOL, DEFAULT_TOL);

    let gcd_to_use = compute_min_gcd(blocks);

    let blocks_reduced = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    // find_bounding_faces already returns records at reduced resolution,
    // so pass gcd=1 to avoid dividing indices a second time.
    let lower_faces = outer_face_records_to_list(&blocks_reduced, &lower_faces_records, 1);
    let upper_faces = outer_face_records_to_list(&blocks_reduced, &upper_faces_records, 1);

    let delta_axis = delta.unwrap_or_else(|| {
        let global_min = blocks_reduced
            .iter()
            .map(|b| {
                b.axis_slice(axis_idx)
                    .iter()
                    .cloned()
                    .fold(Float::INFINITY, Float::min)
            })
            .fold(Float::INFINITY, Float::min);
        let global_max = blocks_reduced
            .iter()
            .map(|b| {
                b.axis_slice(axis_idx)
                    .iter()
                    .cloned()
                    .fold(Float::NEG_INFINITY, Float::max)
            })
            .fold(Float::NEG_INFINITY, Float::max);
        global_max - global_min
    });

    let axis_char = axis.chars().next().unwrap();
    let blocks_up: Vec<Block> = blocks_reduced
        .iter()
        .map(|b| b.shifted(delta_axis, axis_char))
        .collect();
    let blocks_dn: Vec<Block> = blocks_reduced
        .iter()
        .map(|b| b.shifted(-delta_axis, axis_char))
        .collect();

    let mut periodic_matches = Vec::new();
    // Track original (pre-KDTree-correction) block2 FaceRecords so we can
    // remove the correct outer faces later. The corrected lb2/ub2 may differ
    // from the original outer face keys.
    let mut original_block2_recs: Vec<FaceRecord> = Vec::new();

    let lower_pool = dedup_faces(lower_faces);
    let upper_pool = dedup_faces(upper_faces);

    // ── Phase 1: Fast full-face matching via 4-corner comparison ──
    let corner_tol = node_tol_xyz.unwrap_or(DEFAULT_TOL);
    let shift_up = |mut p: [Float; 3]| -> [Float; 3] {
        p[axis_idx] += delta_axis;
        p
    };
    let shift_dn = |mut p: [Float; 3]| -> [Float; 3] {
        p[axis_idx] -= delta_axis;
        p
    };
    let mut consumed_lower = HashSet::<FaceKey>::new();
    let mut consumed_upper = HashSet::<FaceKey>::new();

    let pb1 = ProgressBar::new(lower_pool.len() as u64);
    pb1.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} faces ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb1.set_message("Translational Phase 1 (corners)");

    for face_l in &lower_pool {
        pb1.inc(1);
        if consumed_lower.contains(&face_l.index_key()) {
            continue;
        }
        let matched = upper_pool.iter().find_map(|face_u| {
            if consumed_upper.contains(&face_u.index_key()) {
                return None;
            }
            // Try lower shifted up vs upper original
            if let Some(orient) = full_face_match_transformed(face_l, face_u, shift_up, corner_tol)
            {
                return Some((face_u.clone(), orient));
            }
            // Try upper shifted down vs lower original
            if let Some(orient) = full_face_match_transformed(face_u, face_l, shift_dn, corner_tol)
            {
                return Some((face_u.clone(), orient));
            }
            None
        });
        if let Some((face_u, _orient)) = matched {
            consumed_lower.insert(face_l.index_key());
            consumed_upper.insert(face_u.index_key());

            let rec1 = FaceRecord::from_face(face_l);
            let mut rec2 = FaceRecord::from_face(&face_u);
            // Save original block2 record before KDTree correction
            original_block2_recs.push(rec2.clone());

            let lb1 = [rec1.il, rec1.jl, rec1.kl];
            let ub1 = [rec1.ih, rec1.jh, rec1.kh];
            let lb2_orig = [rec2.il, rec2.jl, rec2.kl];
            let ub2_orig = [rec2.ih, rec2.jh, rec2.kh];

            // Determine shift direction: face1_lb axis value < face2_lb → shift +delta
            let blk1_r = &blocks_reduced[rec1.block_index];
            let blk2_r = &blocks_reduced[rec2.block_index];
            let p1_val = block_axis_val(blk1_r, lb1, axis_idx);
            let p2_val = block_axis_val(blk2_r, lb2_orig, axis_idx);
            let shift_amt = if p1_val < p2_val {
                delta_axis
            } else {
                -delta_axis
            };

            let (corrected_lb2, corrected_ub2, _orient_vec) = compute_periodic_lb_ub_orientation(
                blk1_r, lb1, ub1, blk2_r, lb2_orig, ub2_orig, axis_idx, shift_amt,
            );
            rec2.il = corrected_lb2[0];
            rec2.jl = corrected_lb2[1];
            rec2.kl = corrected_lb2[2];
            rec2.ih = corrected_ub2[0];
            rec2.jh = corrected_ub2[1];
            rec2.kh = corrected_ub2[2];

            let orient = orientation_from_orient_vec(
                &_orient_vec,
                &lb1,
                &ub1,
                &corrected_lb2,
                &corrected_ub2,
            );
            periodic_matches.push(FaceMatch {
                block1: rec1,
                block2: rec2,
                points: Vec::new(),
                orientation: Some(orient),
            });
        }
    }
    pb1.finish_with_message("Translational Phase 1 done");

    // Build remainder pools for Phase 2
    let lower_remainder: Vec<Face> = lower_pool
        .iter()
        .filter(|f| !consumed_lower.contains(&f.index_key()))
        .cloned()
        .collect();
    let upper_remainder: Vec<Face> = upper_pool
        .iter()
        .filter(|f| !consumed_upper.contains(&f.index_key()))
        .cloned()
        .collect();

    // ── Phase 2: Centroid-sorted greedy node-by-node matching on remainder ──
    // Build upper candidate pool with in-plane centroids for nearest-first matching
    struct UpperCandidate {
        face: Face,
        centroid_2d: [Float; 2],
    }
    let mut upper_pool_2: Vec<UpperCandidate> = upper_remainder
        .into_iter()
        .map(|f| {
            let c = f.centroid();
            let centroid_2d = match axis_idx {
                0 => [c[1], c[2]],
                1 => [c[0], c[2]],
                _ => [c[0], c[1]],
            };
            UpperCandidate {
                face: f,
                centroid_2d,
            }
        })
        .collect();

    let pb2 = ProgressBar::new(lower_remainder.len() as u64);
    pb2.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} faces ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb2.set_message("Translational Phase 2 (nodes)");

    for face_l in &lower_remainder {
        pb2.inc(1);
        if upper_pool_2.is_empty() {
            break;
        }

        // Compute lower face's in-plane centroid for distance sorting
        let c = face_l.centroid();
        let lower_c2d = match axis_idx {
            0 => [c[1], c[2]],
            1 => [c[0], c[2]],
            _ => [c[0], c[1]],
        };

        // Sort upper candidates by distance to lower face centroid (nearest first)
        let mut indices: Vec<usize> = (0..upper_pool_2.len()).collect();
        indices.sort_by(|&a, &b| {
            let da = (upper_pool_2[a].centroid_2d[0] - lower_c2d[0]).powi(2)
                + (upper_pool_2[a].centroid_2d[1] - lower_c2d[1]).powi(2);
            let db = (upper_pool_2[b].centroid_2d[0] - lower_c2d[0]).powi(2)
                + (upper_pool_2[b].centroid_2d[1] - lower_c2d[1]).powi(2);
            da.partial_cmp(&db).unwrap()
        });

        let matched_idx = indices.iter().find_map(|&idx| {
            faces_translational_match(
                face_l,
                &upper_pool_2[idx].face,
                &blocks_reduced,
                &blocks_up,
                &blocks_dn,
                axis.as_str(),
                delta_axis,
                node_tol_xyz,
                min_shared_frac,
                min_shared_abs,
                stride_u,
                stride_v,
            )
            .map(|_mode| idx)
        });

        if let Some(idx) = matched_idx {
            let face_u = upper_pool_2.remove(idx);
            let rec1 = FaceRecord::from_face(face_l);
            let mut rec2 = FaceRecord::from_face(&face_u.face);
            // Save original block2 record before KDTree correction
            original_block2_recs.push(rec2.clone());

            let lb1 = [rec1.il, rec1.jl, rec1.kl];
            let ub1 = [rec1.ih, rec1.jh, rec1.kh];
            let lb2_orig = [rec2.il, rec2.jl, rec2.kl];
            let ub2_orig = [rec2.ih, rec2.jh, rec2.kh];

            let blk1_r = &blocks_reduced[rec1.block_index];
            let blk2_r = &blocks_reduced[rec2.block_index];
            let p1_val = block_axis_val(blk1_r, lb1, axis_idx);
            let p2_val = block_axis_val(blk2_r, lb2_orig, axis_idx);
            let shift_amt = if p1_val < p2_val {
                delta_axis
            } else {
                -delta_axis
            };

            let (corrected_lb2, corrected_ub2, orient_vec) = compute_periodic_lb_ub_orientation(
                blk1_r, lb1, ub1, blk2_r, lb2_orig, ub2_orig, axis_idx, shift_amt,
            );
            rec2.il = corrected_lb2[0];
            rec2.jl = corrected_lb2[1];
            rec2.kl = corrected_lb2[2];
            rec2.ih = corrected_ub2[0];
            rec2.jh = corrected_ub2[1];
            rec2.kh = corrected_ub2[2];

            let orient = orientation_from_orient_vec(
                &orient_vec,
                &lb1,
                &ub1,
                &corrected_lb2,
                &corrected_ub2,
            );
            periodic_matches.push(FaceMatch {
                block1: rec1,
                block2: rec2,
                points: Vec::new(),
                orientation: Some(orient),
            });
        }
    }
    pb2.finish_with_message("Translational Phase 2 done");

    // Free shifted block copies now that matching is complete
    drop(blocks_up);
    drop(blocks_dn);

    // Scale periodic matches back to original resolution FIRST so that
    // periodic_keys are at the same resolution as outer_faces (which are
    // already at original resolution from connectivity_fast).
    if gcd_to_use > 1 {
        for rec in &mut periodic_matches {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }
    }

    // Also scale original (pre-correction) block2 records back to original resolution
    if gcd_to_use > 1 {
        for rec in &mut original_block2_recs {
            rec.scale_indices(gcd_to_use);
        }
    }

    let mut periodic_keys = HashSet::new();
    for rec in &periodic_matches {
        periodic_keys.insert(rec.block1.index_key());
        periodic_keys.insert(rec.block2.index_key()); // corrected block2 keys
    }
    // Add original (pre-correction) block2 keys — these match the outer_faces entries
    for rec in &original_block2_recs {
        periodic_keys.insert(rec.index_key());
    }

    // outer_faces are already at original resolution — do NOT scale remaining.
    let mut remaining = Vec::new();
    for record in outer_faces {
        if !periodic_keys.contains(&record.index_key()) {
            remaining.push(record.clone());
        }
    }

    (periodic_matches, remaining)
}

/// Assess one lower/upper face combo and return the match mode when successful.
#[allow(clippy::too_many_arguments)]
fn faces_translational_match(
    face_l: &Face,
    face_u: &Face,
    blocks: &[Block],
    blocks_up: &[Block],
    blocks_dn: &[Block],
    axis: &str,
    delta_axis: Float,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
) -> Option<String> {
    let tol_pair = pair_tolerance(face_l, face_u, blocks, node_tol_xyz, axis);

    if orthogonal_precheck(
        face_l,
        face_u,
        &blocks_up[face_l.block_index().unwrap()],
        &blocks[face_u.block_index().unwrap()],
        delta_axis,
        tol_pair,
        axis,
        min_shared_frac,
        min_shared_abs,
    ) {
        return Some(format!("{axis}_precheck_lower_up"));
    }
    if face_l.touches_by_nodes(
        face_u,
        &blocks_up[face_l.block_index().unwrap()],
        &blocks[face_u.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some("lower_up_vs_upper_orig".to_string());
    }
    if face_l.touches_by_nodes(
        face_u,
        &blocks[face_l.block_index().unwrap()],
        &blocks_dn[face_u.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some("lower_orig_vs_upper_dn".to_string());
    }
    if orthogonal_precheck(
        face_u,
        face_l,
        &blocks_up[face_u.block_index().unwrap()],
        &blocks[face_l.block_index().unwrap()],
        delta_axis,
        tol_pair,
        axis,
        min_shared_frac,
        min_shared_abs,
    ) {
        return Some(format!("{axis}_precheck_upper_up"));
    }
    if face_u.touches_by_nodes(
        face_l,
        &blocks_up[face_u.block_index().unwrap()],
        &blocks[face_l.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some("upper_up_vs_lower_orig".to_string());
    }
    face_u
        .touches_by_nodes(
            face_l,
            &blocks[face_u.block_index().unwrap()],
            &blocks_dn[face_l.block_index().unwrap()],
            tol_pair,
            min_shared_frac,
            min_shared_abs,
            stride_u,
            stride_v,
        )
        .then(|| "upper_orig_vs_lower_dn".to_string())
}

/// Decide the XYZ tolerance for a particular face pair, optionally honoring a global override.
fn pair_tolerance(
    face_a: &Face,
    face_b: &Face,
    blocks: &[Block],
    override_tol: Option<Float>,
    axis: &str,
) -> Float {
    if let Some(tol) = override_tol {
        return tol;
    }
    let spacing_a = median_inplane_spacing(face_a, &blocks[face_a.block_index().unwrap()], axis);
    let spacing_b = median_inplane_spacing(face_b, &blocks[face_b.block_index().unwrap()], axis);
    (0.03 * spacing_a.max(spacing_b)).max(ADAPTIVE_TOL_FLOOR)
}

/// Compute a median edge length for the face in the non-periodic directions.
fn median_inplane_spacing(face: &Face, block: &Block, axis: &str) -> Float {
    let points = face.grid_points(block, 1, 1);
    if points.len() <= 1 {
        return 1.0;
    }
    let mut spacings = Vec::new();
    for window in points.windows(2) {
        let p0 = window[0];
        let p1 = window[1];
        let diff = match axis {
            "x" => [(p0[1] - p1[1]).abs(), (p0[2] - p1[2]).abs()],
            "y" => [(p0[0] - p1[0]).abs(), (p0[2] - p1[2]).abs()],
            _ => [(p0[0] - p1[0]).abs(), (p0[1] - p1[1]).abs()],
        };
        spacings.push(diff[0].hypot(diff[1]));
    }
    spacings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spacings[spacings.len() / 2]
}

/// Perform a quick planar projection test to reject clearly mismatched faces.
#[allow(clippy::too_many_arguments)]
fn orthogonal_precheck(
    face_a: &Face,
    face_b: &Face,
    block_a: &Block,
    block_b: &Block,
    delta: Float,
    tol: Float,
    axis: &str,
    min_shared_frac: Float,
    min_shared_abs: usize,
) -> bool {
    let mut pts_a = face_a.grid_points(block_a, 1, 1);
    let pts_b = face_b.grid_points(block_b, 1, 1);
    if pts_a.is_empty() || pts_b.is_empty() {
        return false;
    }
    match axis {
        "x" => pts_a.iter_mut().for_each(|p| p[0] += delta),
        "y" => pts_a.iter_mut().for_each(|p| p[1] += delta),
        _ => pts_a.iter_mut().for_each(|p| p[2] += delta),
    }

    let proj_a = project_plane(&pts_a, axis);
    let proj_b = project_plane(&pts_b, axis);

    let key_a: HashSet<(i64, i64)> = proj_a
        .iter()
        .map(|p| ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64))
        .collect();
    let key_b: HashSet<(i64, i64)> = proj_b
        .iter()
        .map(|p| ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64))
        .collect();

    let shared = key_a.intersection(&key_b).count();
    shared >= min_shared_abs
        && (shared as Float) >= min_shared_frac * (key_a.len().min(key_b.len()) as Float)
}

/// Project 3D points onto the plane orthogonal to `axis`.
fn project_plane(points: &[[Float; 3]], axis: &str) -> Vec<[Float; 2]> {
    points
        .iter()
        .map(|p| match axis {
            "x" => [p[1], p[2]],
            "y" => [p[0], p[2]],
            _ => [p[0], p[1]],
        })
        .collect()
}

/// Remove duplicate faces while preserving the first occurrence.
fn dedup_faces(mut faces: Vec<Face>) -> Vec<Face> {
    let mut seen = HashSet::new();
    faces.retain(|f| seen.insert(f.index_key()));
    faces
}
