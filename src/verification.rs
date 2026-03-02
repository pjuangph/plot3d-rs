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

/// Generate all 8 traversal permutations for a face (4 direct + 4 transposed).
///
/// Matches Python's `_generate_permutations(lb, ub)`:
///   1. Find the constant axis (where il==ih, jl==jh, or kl==kh).
///   2. For the two varying axes d0, d1:
///      - 4 direct: all combos of forward/reverse for d0 and d1.
///      - 4 transposed: swap d0's values into d1's slot and vice versa.
///   3. Return 8 FaceRecords (constant axis unchanged).
pub(crate) fn generate_permutations(rec: &FaceRecord) -> Vec<FaceRecord> {
    let lb = [rec.il, rec.jl, rec.kl];
    let ub = [rec.ih, rec.jh, rec.kh];

    let mut perms = Vec::with_capacity(8);

    for dim in 0..3 {
        if lb[dim] == ub[dim] {
            // This is the constant axis
            let varying: Vec<usize> = (0..3).filter(|&d| d != dim).collect();
            let d0 = varying[0];
            let d1 = varying[1];
            let vals = [[lb[d0], ub[d0]], [lb[d1], ub[d1]]];

            // 4 direct permutations
            for s0 in 0..2usize {
                for s1 in 0..2usize {
                    let mut new_lb = lb;
                    let mut new_ub = ub;
                    new_lb[d0] = vals[0][s0];
                    new_ub[d0] = vals[0][1 - s0];
                    new_lb[d1] = vals[1][s1];
                    new_ub[d1] = vals[1][1 - s1];
                    perms.push(FaceRecord {
                        block_index: rec.block_index,
                        il: new_lb[0], jl: new_lb[1], kl: new_lb[2],
                        ih: new_ub[0], jh: new_ub[1], kh: new_ub[2],
                        id: rec.id,
                        u_physical: None,
                        v_physical: None,
                    });
                }
            }

            // 4 transposed permutations (swap which axis values go to d0 vs d1)
            for s0 in 0..2usize {
                for s1 in 0..2usize {
                    let mut new_lb = lb;
                    let mut new_ub = ub;
                    new_lb[d0] = vals[1][s0];    // d1's values → d0's slot
                    new_ub[d0] = vals[1][1 - s0];
                    new_lb[d1] = vals[0][s1];    // d0's values → d1's slot
                    new_ub[d1] = vals[0][1 - s1];
                    perms.push(FaceRecord {
                        block_index: rec.block_index,
                        il: new_lb[0], jl: new_lb[1], kl: new_lb[2],
                        ih: new_ub[0], jh: new_ub[1], kh: new_ub[2],
                        id: rec.id,
                        u_physical: None,
                        v_physical: None,
                    });
                }
            }

            break;
        }
    }

    perms
}

/// Try all 8 direction permutations of block2's face against block1's face points.
///
/// Matches Python's `_try_all_permutations`:
///   - Holds pts1 fixed (already extracted).
///   - For block2, tries all 8 permutations (4 direct + 4 transposed).
///   - Returns the first matching permuted FaceRecord, or None.
pub(crate) fn try_all_permutations(
    pts1: &[(Float, Float, Float)],
    block2: &Block,
    rec2: &FaceRecord,
    tol: Float,
) -> Option<FaceRecord> {
    let tol2 = tol * tol;

    for perm in generate_permutations(rec2) {
        let pts2 = extract_face_points(block2, &perm);
        if pts1.len() != pts2.len() {
            continue;
        }

        let worst = pts1
            .iter()
            .zip(pts2.iter())
            .map(|(a, b)| {
                (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)
            })
            .fold(0.0 as Float, Float::max);

        if worst <= tol2 {
            return Some(perm);
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
        if let Some(perm) = try_all_permutations(&pts1, block2, b2, tol) {
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
            if let Some(perm) = try_all_permutations(&pts1, block2, b2, tol) {
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
