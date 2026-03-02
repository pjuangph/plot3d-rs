//! Gold-standard verification for connectivity and periodicity.
//!
//! Pure pass/fail checks -- takes lb/ub as given from the pipeline,
//! does directed traversal, compares point-by-point.
//! Do NOT modify these to get better results.
//!
//! # Verification checks
//!
//!   1. **Face area**: `di*dj*dk == di'*dj'*dk'` -- the total number of
//!      grid points on both faces must agree.
//!   2. **Point-by-point**: every point from face1 (traversing the
//!      diagonal `lb -> ub`) must match the corresponding point from
//!      face2 within the supplied tolerance.
//!
//! # Permutation testing approach
//!
//! When the stored diagonal of `block2` does not produce a point-by-point
//! match, [`try_all_permutations`] exhaustively tests all 8 orientation
//! permutations (see [`crate::face_record::PERMUTATION_MATRICES`]):
//!
//! 1. Extract face2's grid points **once** in canonical ascending order
//!    (both u and v increasing).
//! 2. For each of the 8 permutations (bit encoding:
//!    `perm_idx = u_reversed | (v_reversed << 1) | (swapped << 2)`),
//!    remap output indices back to canonical grid indices via index
//!    arithmetic, avoiding redundant coordinate extraction.
//! 3. Compare every remapped point against face1's points (held
//!    constant). Accept on the first permutation where all points match
//!    within tolerance.
//! 4. On success, reconstruct a corrected [`FaceRecord`] whose diagonal
//!    corners encode the winning orientation, and return the permutation
//!    index.
//!
//! This approach is used by both [`verify_connectivity`] and
//! [`verify_periodicity`]. The periodicity variant additionally rotates
//! block1 by +/- theta before comparison.

use crate::block::Block;
use crate::block_face_functions::{reduce_blocks, rotate_block};
use crate::face_record::{FaceMatch, FaceRecord};
use crate::rotational_periodicity::create_rotation_matrix;
use crate::utils::compute_min_gcd;
use crate::Float;

/// Compute face area (total number of points) from diagonal corners.
pub(crate) fn face_area(rec: &FaceRecord) -> usize {
    let di = if rec.ih >= rec.il { rec.ih - rec.il + 1 } else { rec.il - rec.ih + 1 };
    let dj = if rec.jh >= rec.jl { rec.jh - rec.jl + 1 } else { rec.jl - rec.jh + 1 };
    let dk = if rec.kh >= rec.kl { rec.kh - rec.kl + 1 } else { rec.kl - rec.kh + 1 };
    di * dj * dk
}

/// Build an inclusive range from `start` to `end`, stepping +1 or −1.
pub(crate) fn directed_range(start: usize, end: usize) -> Vec<usize> {
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
pub(crate) fn extract_face_points(block: &Block, rec: &FaceRecord) -> Vec<(Float, Float, Float)> {
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

/// Try all 8 permutations of face2 against face1 using index-based grid manipulation.
///
/// Extracts face2 points **once** in canonical (both axes ascending) order, then
/// applies each of the 8 permutations (flip-u, flip-v, swap) via index arithmetic
/// rather than re-extracting points for each permutation.
///
/// Bit encoding: `perm_idx = u_reversed | (v_reversed << 1) | (swapped << 2)`
///
/// Returns `(corrected_FaceRecord, permutation_index)` on match, or `None`.
pub(crate) fn try_all_permutations(
    pts1: &[(Float, Float, Float)],
    block2: &Block,
    rec2: &FaceRecord,
    tol: Float,
) -> Option<(FaceRecord, u8)> {
    let tol2 = tol * tol;

    // Normalize to ascending ranges, clamped to block dimensions
    let clamp = [
        block2.imax.saturating_sub(1),
        block2.jmax.saturating_sub(1),
        block2.kmax.saturating_sub(1),
    ];
    let lo = [
        rec2.il.min(rec2.ih).min(clamp[0]),
        rec2.jl.min(rec2.jh).min(clamp[1]),
        rec2.kl.min(rec2.kh).min(clamp[2]),
    ];
    let hi = [
        rec2.il.max(rec2.ih).min(clamp[0]),
        rec2.jl.max(rec2.jh).min(clamp[1]),
        rec2.kl.max(rec2.kh).min(clamp[2]),
    ];

    // Find constant axis
    let const_dim = (0..3).find(|&d| lo[d] == hi[d])?;
    let varying: Vec<usize> = (0..3).filter(|&d| d != const_dim).collect();
    let d0 = varying[0]; // "u" axis
    let d1 = varying[1]; // "v" axis
    let nu = hi[d0] - lo[d0] + 1;
    let nv = hi[d1] - lo[d1] + 1;

    // Extract face2 points once in canonical ascending order: u outer, v inner
    let mut grid = Vec::with_capacity(nu * nv);
    for u in 0..nu {
        for v in 0..nv {
            let mut idx = [0usize; 3];
            idx[const_dim] = lo[const_dim];
            idx[d0] = lo[d0] + u;
            idx[d1] = lo[d1] + v;
            grid.push(block2.xyz(idx[0], idx[1], idx[2]));
        }
    }

    // Try each of the 8 permutations
    for perm_idx in 0u8..8 {
        let u_rev = perm_idx & 1 != 0;
        let v_rev = perm_idx & 2 != 0;
        let swap = perm_idx & 4 != 0;

        // Output dimensions after potential swap
        let (out_nu, out_nv) = if swap { (nv, nu) } else { (nu, nv) };

        if pts1.len() != out_nu * out_nv {
            continue;
        }

        let mut ok = true;
        for ou in 0..out_nu {
            if !ok {
                break;
            }
            for ov in 0..out_nv {
                // Map output (ou, ov) back to canonical grid indices (gu, gv)
                let (gu, gv) = if swap { (ov, ou) } else { (ou, ov) };
                let gu = if u_rev { nu - 1 - gu } else { gu };
                let gv = if v_rev { nv - 1 - gv } else { gv };

                let p2 = grid[gu * nv + gv];
                let p1 = pts1[ou * out_nv + ov];
                let d2 =
                    (p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2) + (p1.2 - p2.2).powi(2);
                if d2 > tol2 {
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            // Reconstruct corrected FaceRecord with the matching diagonal
            let mut new_lo = lo;
            let mut new_hi = lo;
            new_lo[const_dim] = lo[const_dim];
            new_hi[const_dim] = lo[const_dim];

            if swap {
                new_lo[d0] = if u_rev { hi[d1] } else { lo[d1] };
                new_hi[d0] = if u_rev { lo[d1] } else { hi[d1] };
                new_lo[d1] = if v_rev { hi[d0] } else { lo[d0] };
                new_hi[d1] = if v_rev { lo[d0] } else { hi[d0] };
            } else {
                new_lo[d0] = if u_rev { hi[d0] } else { lo[d0] };
                new_hi[d0] = if u_rev { lo[d0] } else { hi[d0] };
                new_lo[d1] = if v_rev { hi[d1] } else { lo[d1] };
                new_hi[d1] = if v_rev { lo[d1] } else { hi[d1] };
            }

            let corrected = FaceRecord {
                block_index: rec2.block_index,
                il: new_lo[0],
                jl: new_lo[1],
                kl: new_lo[2],
                ih: new_hi[0],
                jh: new_hi[1],
                kh: new_hi[2],
                id: rec2.id,
                u_physical: None,
                v_physical: None,
            };

            return Some((corrected, perm_idx));
        }
    }

    None
}

/// Verify connectivity face matches using full directed point-by-point traversal.
///
/// Matches Python's `verify_connectivity` exactly:
///   1. Compute GCD, reduce blocks.
///   2. Scale down face_match indices by GCD.
///   3. For each match:
///      a. Dimension check: face_area(b1) == face_area(b2).
///      b. Extract ALL face1 points in directed order (held constant).
///      c. Check stored diagonal: extract face2 points, compare point-by-point.
///      d. Try all 8 permutations (4 direct + 4 transposed) via `try_all_permutations`.
///      e. If permutation matches: correct block2's lb/ub, scale back by GCD.
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_connectivity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    // Step 1: Compute GCD and reduce blocks
    let gcd_to_use = compute_min_gcd(blocks);
    let reduced_blocks = reduce_blocks(blocks, gcd_to_use);

    // Step 2: Scale down face_match indices by GCD
    let scaled_matches: Vec<FaceMatch> = face_matches
        .iter()
        .map(|fm| {
            let mut sfm = fm.clone();
            sfm.divide_indices(gcd_to_use);
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

        // Step 3a: Dimension check
        let n1 = face_area(b1);
        let n2 = face_area(b2);
        if n1 != n2 {
            let orig = &face_matches[idx];
            eprintln!("verify_connectivity: DIMENSION MISMATCH at index {}", idx);
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{}) n={}",
                orig.block1.block_index,
                orig.block1.il, orig.block1.jl, orig.block1.kl,
                orig.block1.ih, orig.block1.jh, orig.block1.kh, n1
            );
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{}) n={}",
                orig.block2.block_index,
                orig.block2.il, orig.block2.jl, orig.block2.kl,
                orig.block2.ih, orig.block2.jh, orig.block2.kh, n2
            );
            mismatched.push(face_matches[idx].clone());
            continue;
        }

        // Step 3b: Extract face1 points (held constant)
        let pts1 = extract_face_points(block1, b1);

        // Step 3c: Check stored diagonal first (point-by-point)
        let pts2 = extract_face_points(block2, b2);
        let worst = pts1
            .iter()
            .zip(pts2.iter())
            .map(|(a, b)| {
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
            })
            .fold(0.0 as Float, Float::max);

        if worst < tol {
            verified.push(face_matches[idx].clone());
            continue;
        }

        // Step 3d: Try all 8 permutations of block2's direction
        if let Some((perm, _perm_idx)) = try_all_permutations(&pts1, block2, b2, tol) {
            // Step 3e: Correct block2's lb/ub, scale back by GCD
            let mut corrected = face_matches[idx].clone();
            corrected.block2.il = perm.il * gcd_to_use;
            corrected.block2.jl = perm.jl * gcd_to_use;
            corrected.block2.kl = perm.kl * gcd_to_use;
            corrected.block2.ih = perm.ih * gcd_to_use;
            corrected.block2.jh = perm.jh * gcd_to_use;
            corrected.block2.kh = perm.kh * gcd_to_use;
            verified.push(corrected);
            if b1_idx == b2_idx {
                eprintln!(
                    "verify_connectivity: Self-match corrected for block index {}",
                    b1_idx
                );
            }
        } else {
            let orig = &face_matches[idx];
            let n_bad = pts1
                .iter()
                .zip(pts2.iter())
                .filter(|(a, b)| {
                    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
                        > tol
                })
                .count();
            eprintln!("verify_connectivity: POINT MISMATCH at index {}", idx);
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{})",
                orig.block1.block_index,
                orig.block1.il, orig.block1.jl, orig.block1.kl,
                orig.block1.ih, orig.block1.jh, orig.block1.kh
            );
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{})",
                orig.block2.block_index,
                orig.block2.il, orig.block2.jl, orig.block2.kl,
                orig.block2.ih, orig.block2.jh, orig.block2.kh
            );
            eprintln!(
                "  total points: {}, mismatched: {}, max dist: {:.6e}",
                pts1.len(), n_bad, worst
            );
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}

/// Verify periodic face matches using full directed point-by-point traversal with rotation.
///
/// Matches Python's `verify_periodicity` exactly:
///   1. Compute GCD, reduce blocks.
///   2. Build rotation matrices for +theta and -theta.
///   3. Pre-rotate ALL reduced blocks in both directions.
///   4. Scale down face_match indices by GCD.
///   5. For each match:
///      a. Dimension check: face_area(b1) == face_area(b2).
///      b. For each rotation [+theta, -theta]:
///         - Extract face1 points from rotated block1 (held constant).
///         - Check stored diagonal: extract face2 points, compare point-by-point.
///         - Try all 8 permutations via `try_all_permutations`.
///         - If match found: correct block2's lb/ub, scale back by GCD.
///      c. If neither rotation works: push to mismatched.
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
            sfm.divide_indices(gcd_to_use);
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

        // Step 5a: Dimension check
        let n1 = face_area(b1);
        let n2 = face_area(b2);
        if n1 != n2 {
            let orig = &face_matches[idx];
            eprintln!("verify_periodicity: DIMENSION MISMATCH at index {}", idx);
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{}) n={}",
                orig.block1.block_index,
                orig.block1.il, orig.block1.jl, orig.block1.kl,
                orig.block1.ih, orig.block1.jh, orig.block1.kh, n1
            );
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{}) n={}",
                orig.block2.block_index,
                orig.block2.il, orig.block2.jl, orig.block2.kl,
                orig.block2.ih, orig.block2.jh, orig.block2.kh, n2
            );
            mismatched.push(face_matches[idx].clone());
            continue;
        }

        let mut found = false;

        // Step 5b: Try +theta rotation first, then -theta
        for rotated_blocks in [&rotated_blocks_pos, &rotated_blocks_neg] {
            if found {
                break;
            }

            let block1_rotated = &rotated_blocks[b1_idx];

            // Check stored diagonal first (full point-by-point)
            let pts1 = extract_face_points(block1_rotated, b1);
            let pts2 = extract_face_points(block2, b2);
            let worst = pts1
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| {
                    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
                })
                .fold(0.0 as Float, Float::max);

            if worst < tol {
                verified.push(face_matches[idx].clone());
                found = true;
                break;
            }

            // Try all 8 permutations of block2's direction
            if let Some((perm, _perm_idx)) = try_all_permutations(&pts1, block2, b2, tol) {
                let mut corrected = face_matches[idx].clone();
                corrected.block2.il = perm.il * gcd_to_use;
                corrected.block2.jl = perm.jl * gcd_to_use;
                corrected.block2.kl = perm.kl * gcd_to_use;
                corrected.block2.ih = perm.ih * gcd_to_use;
                corrected.block2.jh = perm.jh * gcd_to_use;
                corrected.block2.kh = perm.kh * gcd_to_use;
                verified.push(corrected);
                found = true;
                break;
            }
        }

        if !found {
            let orig = &face_matches[idx];
            eprintln!("verify_periodicity: MISMATCH at index {}", idx);
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{})",
                orig.block1.block_index,
                orig.block1.il, orig.block1.jl, orig.block1.kl,
                orig.block1.ih, orig.block1.jh, orig.block1.kh
            );
            eprintln!(
                "  block {}: lb=({},{},{}) ub=({},{},{})",
                orig.block2.block_index,
                orig.block2.il, orig.block2.jl, orig.block2.kl,
                orig.block2.ih, orig.block2.jh, orig.block2.kh
            );
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}
