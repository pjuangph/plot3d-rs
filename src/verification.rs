//! Gold-standard verification for connectivity and periodicity.
//!
//! Pure pass/fail checks — takes lb/ub as given from the pipeline,
//! does directed traversal, compares point-by-point.
//! Do NOT modify these to get better results.
//!
//! Checks:
//!   1. Face AREA must match: `di*dj*dk == di'*dj'*dk'`
//!   2. Every point from face1 (traversing lb→ub) must match the
//!      corresponding point from face2 (traversing lb→ub).

use crate::block::Block;
use crate::block_face_functions::{reduce_blocks, rotate_block};
use crate::face_record::{FaceMatch, FaceRecord};
use crate::rotational_periodicity::create_rotation_matrix;
use crate::utils::{apply_rotation, compute_min_gcd};
use crate::Float;
use std::collections::HashSet;

/// Compute face area (total number of points) from diagonal corners.
fn face_area(rec: &FaceRecord) -> usize {
    let di = if rec.ih >= rec.il { rec.ih - rec.il + 1 } else { rec.il - rec.ih + 1 };
    let dj = if rec.jh >= rec.jl { rec.jh - rec.jl + 1 } else { rec.jl - rec.jh + 1 };
    let dk = if rec.kh >= rec.kl { rec.kh - rec.kl + 1 } else { rec.kl - rec.kh + 1 };
    di * dj * dk
}

/// Build an inclusive range from `start` to `end`, stepping +1 or −1.
fn directed_range(start: usize, end: usize) -> Vec<usize> {
    if start <= end {
        (start..=end).collect()
    } else {
        (end..=start).rev().collect()
    }
}

/// Extract face points in the directed traversal order defined by the FaceRecord
/// diagonals (il,jl,kl) → (ih,jh,kh).
///
/// Point n from face A must match point n from face B — the diagonal
/// convention preserves the node-to-node mapping between blocks.
fn extract_face_points(block: &Block, rec: &FaceRecord) -> Vec<(Float, Float, Float)> {
    let il = rec.il.min(block.imax.saturating_sub(1));
    let ih = rec.ih.min(block.imax.saturating_sub(1));
    let jl = rec.jl.min(block.jmax.saturating_sub(1));
    let jh = rec.jh.min(block.jmax.saturating_sub(1));
    let kl = rec.kl.min(block.kmax.saturating_sub(1));
    let kh = rec.kh.min(block.kmax.saturating_sub(1));

    let i_range = directed_range(il, ih);
    let j_range = directed_range(jl, jh);
    let k_range = directed_range(kl, kh);

    let mut pts = Vec::with_capacity(i_range.len() * j_range.len() * k_range.len());
    for &i in &i_range {
        for &j in &j_range {
            for &k in &k_range {
                pts.push(block.xyz(i, j, k));
            }
        }
    }
    pts
}

/// Verify connectivity face matches by checking that block1's diagonal corners
/// spatially match block2's diagonal corners within tolerance.
///
/// Matches Python's `verify_connectivity` exactly:
///   1. Compute GCD, reduce blocks.
///   2. Scale down face_match indices by GCD.
///   3. For each match: check stored diagonal first, then try all ordered pairs
///      of block2's unique face corners. Correct block2's lb/ub if a permutation
///      matches, scaling indices back up by GCD.
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_connectivity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    // Step 1: Compute GCD and reduce blocks (same as Python)
    let gcd_to_use = compute_min_gcd(blocks);
    let reduced_blocks = reduce_blocks(blocks, gcd_to_use);

    // Step 2: Scale down face_match indices by GCD
    let scaled_matches: Vec<FaceMatch> = face_matches
        .iter()
        .map(|fm| {
            let mut sfm = fm.clone();
            sfm.block1.il /= gcd_to_use;
            sfm.block1.jl /= gcd_to_use;
            sfm.block1.kl /= gcd_to_use;
            sfm.block1.ih /= gcd_to_use;
            sfm.block1.jh /= gcd_to_use;
            sfm.block1.kh /= gcd_to_use;
            sfm.block2.il /= gcd_to_use;
            sfm.block2.jl /= gcd_to_use;
            sfm.block2.kl /= gcd_to_use;
            sfm.block2.ih /= gcd_to_use;
            sfm.block2.jh /= gcd_to_use;
            sfm.block2.kh /= gcd_to_use;
            sfm
        })
        .collect();

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();

    for (idx, sfm) in scaled_matches.iter().enumerate() {
        let b1 = &sfm.block1;
        let b2 = &sfm.block2;
        let b1_idx = b1.block_index;
        let b2_idx = b2.block_index;

        if b1_idx >= reduced_blocks.len() || b2_idx >= reduced_blocks.len() {
            mismatched.push(face_matches[idx].clone());
            continue;
        }

        let block1 = &reduced_blocks[b1_idx];
        let block2 = &reduced_blocks[b2_idx];

        // Block1 diagonal coordinates (from reduced blocks)
        let (x1_l, y1_l, z1_l) = block1.xyz(b1.il, b1.jl, b1.kl);
        let (x1_u, y1_u, z1_u) = block1.xyz(b1.ih, b1.jh, b1.kh);

        // Enumerate unique corners of block2's face
        let i2 = [b2.il, b2.ih];
        let j2 = [b2.jl, b2.jh];
        let k2 = [b2.kl, b2.kh];

        let mut unique_corners: Vec<(usize, usize, usize)> = Vec::new();
        let mut seen = HashSet::new();
        for &i in &i2 {
            for &j in &j2 {
                for &k in &k2 {
                    let key = (i, j, k);
                    if seen.insert(key) {
                        unique_corners.push(key);
                    }
                }
            }
        }

        // Check stored diagonal first
        let (x2_l, y2_l, z2_l) = block2.xyz(b2.il, b2.jl, b2.kl);
        let (x2_u, y2_u, z2_u) = block2.xyz(b2.ih, b2.jh, b2.kh);

        let dx = x2_l - x1_l;
        let dy = y2_l - y1_l;
        let dz = z2_l - z1_l;
        let d_lower = (dx * dx + dy * dy + dz * dz).sqrt();
        let dx = x2_u - x1_u;
        let dy = y2_u - y1_u;
        let dz = z2_u - z1_u;
        let d_upper = (dx * dx + dy * dy + dz * dz).sqrt();

        if d_lower < tol && d_upper < tol {
            verified.push(face_matches[idx].clone());
            continue;
        }

        // Try all permutations of block2's corners
        let mut found = false;
        let mut best_d_lower = d_lower;
        let mut best_d_upper = d_upper;

        for &corner_lower in &unique_corners {
            for &corner_upper in &unique_corners {
                if corner_lower == corner_upper {
                    continue;
                }

                let (il, jl, kl) = corner_lower;
                let (iu, ju, ku) = corner_upper;

                let (x2_l, y2_l, z2_l) = block2.xyz(il, jl, kl);
                let (x2_u, y2_u, z2_u) = block2.xyz(iu, ju, ku);

                let dx = x2_l - x1_l;
                let dy = y2_l - y1_l;
                let dz = z2_l - z1_l;
                let dl = (dx * dx + dy * dy + dz * dz).sqrt();
                let dx = x2_u - x1_u;
                let dy = y2_u - y1_u;
                let dz = z2_u - z1_u;
                let du = (dx * dx + dy * dy + dz * dz).sqrt();

                if dl < best_d_lower {
                    best_d_lower = dl;
                }
                if du < best_d_upper {
                    best_d_upper = du;
                }

                if dl < tol && du < tol {
                    let mut corrected = face_matches[idx].clone();
                    corrected.block2.il = il * gcd_to_use;
                    corrected.block2.jl = jl * gcd_to_use;
                    corrected.block2.kl = kl * gcd_to_use;
                    corrected.block2.ih = iu * gcd_to_use;
                    corrected.block2.jh = ju * gcd_to_use;
                    corrected.block2.kh = ku * gcd_to_use;
                    verified.push(corrected);
                    if b1_idx == b2_idx {
                        eprintln!(
                            "verify_connectivity: Self-match corrected for block index {}",
                            b1_idx
                        );
                    }
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        if !found {
            let orig = &face_matches[idx];
            let b1_orig = &orig.block1;
            let b2_orig = &orig.block2;
            eprintln!("verify_connectivity: MISMATCH at face_match index {}", idx);
            eprintln!(
                "  block1 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                b1_orig.block_index,
                b1_orig.il, b1_orig.jl, b1_orig.kl,
                b1_orig.ih, b1_orig.jh, b1_orig.kh
            );
            eprintln!(
                "  block2 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                b2_orig.block_index,
                b2_orig.il, b2_orig.jl, b2_orig.kl,
                b2_orig.ih, b2_orig.jh, b2_orig.kh
            );
            eprintln!(
                "  block1 lower xyz = ({:.6e}, {:.6e}, {:.6e})",
                x1_l, y1_l, z1_l
            );
            eprintln!(
                "  block1 upper xyz = ({:.6e}, {:.6e}, {:.6e})",
                x1_u, y1_u, z1_u
            );
            eprintln!(
                "  Closest block2 corner dist to block1 lower: {:.6e}",
                best_d_lower
            );
            eprintln!(
                "  Closest block2 corner dist to block1 upper: {:.6e}",
                best_d_upper
            );
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}

/// Brute-force connectivity verification: tries all 8 diagonal orientations of face2.
///
/// For each match, keeps face1 fixed and tries all 8 possible lb/ub assignments
/// for face2 (all combinations of lo/hi for each axis). If any orientation
/// produces a point-by-point match via directed traversal, the match is verified
/// and face2's lb/ub are set to the matching orientation.
///
/// This is the "true" solution — exhaustive search that finds the correct
/// orientation if one exists. Use this to validate that the pipeline is
/// finding all valid connectivities.
///
/// # Returns
/// `(verified, mismatched)` — verified matches have face2's lb/ub corrected.
pub fn connectivity_bruteforce(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let tol2 = tol * tol;
    let mut verified = Vec::new();
    let mut mismatched = Vec::new();
    let mut area_failures = 0usize;
    let mut point_failures = 0usize;

    for fm in face_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            mismatched.push(fm.clone());
            continue;
        }

        // 1. Area check
        let area1 = face_area(b1);
        let area2 = face_area(b2);
        if area1 != area2 {
            area_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // Face1 points — fixed orientation
        let pts1 = extract_face_points(block1, b1);

        // 2. Try all 8 diagonal orientations of face2
        let i_lo = b2.i_lo().min(block2.imax.saturating_sub(1));
        let i_hi = b2.i_hi().min(block2.imax.saturating_sub(1));
        let j_lo = b2.j_lo().min(block2.jmax.saturating_sub(1));
        let j_hi = b2.j_hi().min(block2.jmax.saturating_sub(1));
        let k_lo = b2.k_lo().min(block2.kmax.saturating_sub(1));
        let k_hi = b2.k_hi().min(block2.kmax.saturating_sub(1));

        let diagonals = [
            (i_lo, j_lo, k_lo, i_hi, j_hi, k_hi),
            (i_hi, j_lo, k_lo, i_lo, j_hi, k_hi),
            (i_lo, j_hi, k_lo, i_hi, j_lo, k_hi),
            (i_lo, j_lo, k_hi, i_hi, j_hi, k_lo),
            (i_hi, j_hi, k_lo, i_lo, j_lo, k_hi),
            (i_hi, j_lo, k_hi, i_lo, j_hi, k_lo),
            (i_lo, j_hi, k_hi, i_hi, j_lo, k_lo),
            (i_hi, j_hi, k_hi, i_lo, j_lo, k_lo),
        ];

        let mut found = false;
        for &(il, jl, kl, ih, jh, kh) in &diagonals {
            let mut rec2 = b2.clone();
            rec2.il = il;
            rec2.jl = jl;
            rec2.kl = kl;
            rec2.ih = ih;
            rec2.jh = jh;
            rec2.kh = kh;

            let pts2 = extract_face_points(block2, &rec2);
            if pts1.len() != pts2.len() {
                continue;
            }

            let worst = pts1
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
                .fold(0.0 as Float, Float::max);

            if worst <= tol2 {
                let mut fm_out = fm.clone();
                fm_out.block2.il = il;
                fm_out.block2.jl = jl;
                fm_out.block2.kl = kl;
                fm_out.block2.ih = ih;
                fm_out.block2.jh = jh;
                fm_out.block2.kh = kh;
                verified.push(fm_out);
                found = true;
                break;
            }
        }

        if !found {
            point_failures += 1;
            mismatched.push(fm.clone());
        }
    }

    eprintln!(
        "  connectivity_bruteforce: {} verified, {} area failures, {} point failures",
        verified.len(),
        area_failures,
        point_failures
    );
    (verified, mismatched)
}

/// Brute-force periodic verification: tries all 8 diagonal orientations × ±theta.
///
/// Same as `connectivity_bruteforce` but rotates face1 points by ±theta
/// before comparing. Tries all 16 combinations (8 diagonals × 2 rotations).
///
/// # Returns
/// `(verified, mismatched)` — verified matches have face2's lb/ub corrected.
pub fn periodicity_bruteforce(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    theta: Float,
    rotation_axis: char,
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let tol2 = tol * tol;
    let rot_pos = create_rotation_matrix(theta, rotation_axis);
    let rot_neg = create_rotation_matrix(-theta, rotation_axis);

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();
    let mut area_failures = 0usize;
    let mut point_failures = 0usize;

    for fm in face_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            mismatched.push(fm.clone());
            continue;
        }

        // 1. Area check
        let area1 = face_area(b1);
        let area2 = face_area(b2);
        if area1 != area2 {
            area_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // Face1 points — fixed orientation
        let pts1_raw = extract_face_points(block1, b1);

        // Pre-compute rotated versions of face1
        let pts1_pos: Vec<(Float, Float, Float)> = pts1_raw
            .iter()
            .map(|p| {
                let r = apply_rotation([p.0, p.1, p.2], rot_pos);
                (r[0], r[1], r[2])
            })
            .collect();
        let pts1_neg: Vec<(Float, Float, Float)> = pts1_raw
            .iter()
            .map(|p| {
                let r = apply_rotation([p.0, p.1, p.2], rot_neg);
                (r[0], r[1], r[2])
            })
            .collect();

        // 2. Try all 8 diagonal orientations × 2 rotations
        let i_lo = b2.i_lo().min(block2.imax.saturating_sub(1));
        let i_hi = b2.i_hi().min(block2.imax.saturating_sub(1));
        let j_lo = b2.j_lo().min(block2.jmax.saturating_sub(1));
        let j_hi = b2.j_hi().min(block2.jmax.saturating_sub(1));
        let k_lo = b2.k_lo().min(block2.kmax.saturating_sub(1));
        let k_hi = b2.k_hi().min(block2.kmax.saturating_sub(1));

        let diagonals = [
            (i_lo, j_lo, k_lo, i_hi, j_hi, k_hi),
            (i_hi, j_lo, k_lo, i_lo, j_hi, k_hi),
            (i_lo, j_hi, k_lo, i_hi, j_lo, k_hi),
            (i_lo, j_lo, k_hi, i_hi, j_hi, k_lo),
            (i_hi, j_hi, k_lo, i_lo, j_lo, k_hi),
            (i_hi, j_lo, k_hi, i_lo, j_hi, k_lo),
            (i_lo, j_hi, k_hi, i_hi, j_lo, k_lo),
            (i_hi, j_hi, k_hi, i_lo, j_lo, k_lo),
        ];

        let mut found = false;
        for &(il, jl, kl, ih, jh, kh) in &diagonals {
            let mut rec2 = b2.clone();
            rec2.il = il;
            rec2.jl = jl;
            rec2.kl = kl;
            rec2.ih = ih;
            rec2.jh = jh;
            rec2.kh = kh;

            let pts2 = extract_face_points(block2, &rec2);
            if pts1_pos.len() != pts2.len() {
                continue;
            }

            // Try +theta
            let worst_pos = pts1_pos
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
                .fold(0.0 as Float, Float::max);

            if worst_pos <= tol2 {
                let mut fm_out = fm.clone();
                fm_out.block2.il = il;
                fm_out.block2.jl = jl;
                fm_out.block2.kl = kl;
                fm_out.block2.ih = ih;
                fm_out.block2.jh = jh;
                fm_out.block2.kh = kh;
                verified.push(fm_out);
                found = true;
                break;
            }

            // Try -theta
            let worst_neg = pts1_neg
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
                .fold(0.0 as Float, Float::max);

            if worst_neg <= tol2 {
                let mut fm_out = fm.clone();
                fm_out.block2.il = il;
                fm_out.block2.jl = jl;
                fm_out.block2.kl = kl;
                fm_out.block2.ih = ih;
                fm_out.block2.jh = jh;
                fm_out.block2.kh = kh;
                verified.push(fm_out);
                found = true;
                break;
            }
        }

        if !found {
            point_failures += 1;
            mismatched.push(fm.clone());
        }
    }

    eprintln!(
        "  periodicity_bruteforce: {} verified, {} area failures, {} point failures",
        verified.len(),
        area_failures,
        point_failures
    );
    (verified, mismatched)
}

/// Verify periodic face matches by checking that rotated block1's diagonal corners
/// spatially match block2's diagonal corners within tolerance.
///
/// Matches Python's `verify_periodicity` exactly:
///   1. Compute GCD, reduce blocks.
///   2. Build rotation matrices for +theta and -theta.
///   3. Pre-rotate ALL reduced blocks in both directions.
///   4. Scale down face_match indices by GCD.
///   5. For each match: try +theta then -theta, check stored diagonal first,
///      then try all ordered pairs of block2's unique face corners.
///      Correct block2's lb/ub if a permutation matches, scaling back by GCD.
///
/// # Arguments
/// * `theta` - rotation angle in **radians** (Python takes degrees and converts)
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_periodicity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    theta: Float,
    rotation_axis: char,
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    // Step 1: Compute GCD and reduce blocks
    let gcd_to_use = compute_min_gcd(blocks);
    let reduced_blocks = reduce_blocks(blocks, gcd_to_use);

    // Step 2: Build rotation matrices for +theta and -theta
    let rotation_matrix_pos = create_rotation_matrix(theta, rotation_axis);
    let rotation_matrix_neg = create_rotation_matrix(-theta, rotation_axis);

    // Step 3: Pre-rotate ALL reduced blocks in both directions
    let rotated_blocks_pos: Vec<Block> = reduced_blocks
        .iter()
        .map(|b| rotate_block(b, rotation_matrix_pos))
        .collect();
    let rotated_blocks_neg: Vec<Block> = reduced_blocks
        .iter()
        .map(|b| rotate_block(b, rotation_matrix_neg))
        .collect();

    // Step 4: Scale down face_match indices by GCD
    let scaled_matches: Vec<FaceMatch> = face_matches
        .iter()
        .map(|fm| {
            let mut sfm = fm.clone();
            sfm.block1.il /= gcd_to_use;
            sfm.block1.jl /= gcd_to_use;
            sfm.block1.kl /= gcd_to_use;
            sfm.block1.ih /= gcd_to_use;
            sfm.block1.jh /= gcd_to_use;
            sfm.block1.kh /= gcd_to_use;
            sfm.block2.il /= gcd_to_use;
            sfm.block2.jl /= gcd_to_use;
            sfm.block2.kl /= gcd_to_use;
            sfm.block2.ih /= gcd_to_use;
            sfm.block2.jh /= gcd_to_use;
            sfm.block2.kh /= gcd_to_use;
            sfm
        })
        .collect();

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();

    for (idx, sfm) in scaled_matches.iter().enumerate() {
        let b1 = &sfm.block1;
        let b2 = &sfm.block2;
        let b1_idx = b1.block_index;
        let b2_idx = b2.block_index;

        if b1_idx >= reduced_blocks.len() || b2_idx >= reduced_blocks.len() {
            mismatched.push(face_matches[idx].clone());
            continue;
        }

        let block2 = &reduced_blocks[b2_idx];

        // Enumerate unique corners of block2's face
        let i2 = [b2.il, b2.ih];
        let j2 = [b2.jl, b2.jh];
        let k2 = [b2.kl, b2.kh];

        let mut unique_corners: Vec<(usize, usize, usize)> = Vec::new();
        let mut seen = HashSet::new();
        for &i in &i2 {
            for &j in &j2 {
                for &k in &k2 {
                    let key = (i, j, k);
                    if seen.insert(key) {
                        unique_corners.push(key);
                    }
                }
            }
        }

        let mut found = false;
        let mut best_d_lower: Float = Float::INFINITY;
        let mut best_d_upper: Float = Float::INFINITY;

        // Try +theta rotation first, then -theta
        for rotated_blocks in [&rotated_blocks_pos, &rotated_blocks_neg] {
            if found {
                break;
            }

            let block1_rotated = &rotated_blocks[b1_idx];

            // Block1 rotated diagonal coordinates
            let (x1_l, y1_l, z1_l) = block1_rotated.xyz(b1.il, b1.jl, b1.kl);
            let (x1_u, y1_u, z1_u) = block1_rotated.xyz(b1.ih, b1.jh, b1.kh);

            // Check stored diagonal first
            let (x2_l, y2_l, z2_l) = block2.xyz(b2.il, b2.jl, b2.kl);
            let (x2_u, y2_u, z2_u) = block2.xyz(b2.ih, b2.jh, b2.kh);

            let dx = x2_l - x1_l;
            let dy = y2_l - y1_l;
            let dz = z2_l - z1_l;
            let d_lower = (dx * dx + dy * dy + dz * dz).sqrt();
            let dx = x2_u - x1_u;
            let dy = y2_u - y1_u;
            let dz = z2_u - z1_u;
            let d_upper = (dx * dx + dy * dy + dz * dz).sqrt();

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
            for &corner_lower in &unique_corners {
                for &corner_upper in &unique_corners {
                    if corner_lower == corner_upper {
                        continue;
                    }

                    let (il, jl, kl) = corner_lower;
                    let (iu, ju, ku) = corner_upper;

                    let (x2_l, y2_l, z2_l) = block2.xyz(il, jl, kl);
                    let (x2_u, y2_u, z2_u) = block2.xyz(iu, ju, ku);

                    let dx = x2_l - x1_l;
                    let dy = y2_l - y1_l;
                    let dz = z2_l - z1_l;
                    let dl = (dx * dx + dy * dy + dz * dz).sqrt();
                    let dx = x2_u - x1_u;
                    let dy = y2_u - y1_u;
                    let dz = z2_u - z1_u;
                    let du = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dl < best_d_lower {
                        best_d_lower = dl;
                    }
                    if du < best_d_upper {
                        best_d_upper = du;
                    }

                    if dl < tol && du < tol {
                        let mut corrected = face_matches[idx].clone();
                        corrected.block2.il = il * gcd_to_use;
                        corrected.block2.jl = jl * gcd_to_use;
                        corrected.block2.kl = kl * gcd_to_use;
                        corrected.block2.ih = iu * gcd_to_use;
                        corrected.block2.jh = ju * gcd_to_use;
                        corrected.block2.kh = ku * gcd_to_use;
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
            let orig = &face_matches[idx];
            let b1_orig = &orig.block1;
            let b2_orig = &orig.block2;
            eprintln!("verify_periodicity: MISMATCH at face_match index {}", idx);
            eprintln!(
                "  block1 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                b1_orig.block_index,
                b1_orig.il, b1_orig.jl, b1_orig.kl,
                b1_orig.ih, b1_orig.jh, b1_orig.kh
            );
            eprintln!(
                "  block2 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                b2_orig.block_index,
                b2_orig.il, b2_orig.jl, b2_orig.kl,
                b2_orig.ih, b2_orig.jh, b2_orig.kh
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
