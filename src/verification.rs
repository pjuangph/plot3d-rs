//! Gold-standard verification for connectivity and periodicity.
//!
//! # Permutation Matrix Approach
//!
//! When two block faces meet at an interface, their parametric (u, v)
//! coordinate systems may differ — flipped, transposed, or both. Rather
//! than re-extracting coordinates in every possible traversal order, we:
//!
//! 1. Extract both faces as **canonical 2D grids** (ascending index order).
//! 2. Apply the stored [`PERMUTATION_MATRICES`][perm] entry to face B's grid.
//! 3. Compare point-by-point within tolerance.
//!
//! The 8 pre-computed permutation matrices encode every possible orientation.
//! The `permutation_index` (0-7) is the only orientation data needed:
//!
//! ```text
//! perm_idx = u_reversed | (v_reversed << 1) | (swapped << 2)
//! ```
//!
//! - **0-3** (in-plane): same constant axis, direction flips only.
//! - **4-7** (cross-plane): different constant axes, loop order changes.
//!
//! [perm]: crate::face_record::PERMUTATION_MATRICES
//!
//! # Public API
//!
//! - [`extract_canonical_grid`] — extract face points as a 2D grid in ascending order
//! - [`apply_permutation`] — apply a permutation matrix to a 2D grid
//! - [`verify_match`] — compare two point arrays within tolerance
//! - [`try_all_permutations`] — find which permutation makes face B match face A
//! - [`verify_partial_match`] — count matching points when face B is smaller than face A
//! - [`determine_plane`] — classify face pair as in-plane or cross-plane
//! - [`verify_connectivity`] — verify connectivity face matches
//! - [`verify_periodicity`] — verify periodic face matches with rotation
//!
//! # JSON Export Convention
//!
//! When exporting to the **diagonal (lb/ub)** JSON format:
//!
//! - **In-plane matches** (perm 0-3): block2's `lb`/`ub` encodes traversal
//!   direction. `permutation_index` is set to **-1** (direction is fully
//!   recoverable from the bounds).
//! - **Cross-plane matches** (perm 4-7): ascending `lb`/`ub` bounds with the
//!   actual `permutation_index`, since bounds alone cannot encode an axis swap.

use crate::block::Block;
use crate::block_face_functions::{reduce_blocks, rotate_block};
use crate::face_record::{FaceMatch, FaceRecord, Orientation, PERMUTATION_MATRICES};
use crate::rotational_periodicity::create_rotation_matrix;
use crate::utils::compute_min_gcd;
use crate::Float;

// ── Core helpers: extract, permute, compare ─────────────────────────────

/// Extract face points as a canonical 2D grid (both axes ascending).
///
/// Finds the constant axis from the FaceRecord bounds, then extracts
/// points with the first varying axis as the outer loop (u) and the
/// second as the inner loop (v), both in ascending order.
///
/// Returns `(grid, nu, nv)` where `grid` has layout `grid[u * nv + v]`.
/// Returns `None` if no constant axis is found (degenerate face).
pub fn extract_canonical_grid(
    block: &Block,
    rec: &FaceRecord,
) -> Option<(Vec<(Float, Float, Float)>, usize, usize)> {
    let (raw_lo, raw_hi) = rec.bounds();
    let imax = [
        block.imax.saturating_sub(1),
        block.jmax.saturating_sub(1),
        block.kmax.saturating_sub(1),
    ];
    let lo = [
        raw_lo[0].min(imax[0]),
        raw_lo[1].min(imax[1]),
        raw_lo[2].min(imax[2]),
    ];
    let hi = [
        raw_hi[0].min(imax[0]),
        raw_hi[1].min(imax[1]),
        raw_hi[2].min(imax[2]),
    ];

    let const_dim = rec.constant_axis()?;
    let varying: Vec<usize> = (0..3).filter(|&d| d != const_dim).collect();
    let d0 = varying[0]; // u axis
    let d1 = varying[1]; // v axis
    let nu = hi[d0] - lo[d0] + 1;
    let nv = hi[d1] - lo[d1] + 1;

    let mut grid = Vec::with_capacity(nu * nv);
    for u in 0..nu {
        for v in 0..nv {
            let mut idx = [0usize; 3];
            idx[const_dim] = lo[const_dim];
            idx[d0] = lo[d0] + u;
            idx[d1] = lo[d1] + v;
            grid.push(block.xyz(idx[0], idx[1], idx[2]));
        }
    }

    Some((grid, nu, nv))
}

/// Apply a pre-computed permutation matrix to a 2D grid.
///
/// Uses [`PERMUTATION_MATRICES`] to transform `grid_b`'s (u, v) layout
/// to match `grid_a`'s layout. The matrix is looked up by `perm_idx` (0-7),
/// not recalculated.
///
/// Bit encoding: `perm_idx = u_reversed | (v_reversed << 1) | (swapped << 2)`
///
/// Returns `(permuted_grid, out_nu, out_nv)`.
pub fn apply_permutation(
    grid: &[(Float, Float, Float)],
    nu: usize,
    nv: usize,
    perm_idx: u8,
) -> (Vec<(Float, Float, Float)>, usize, usize) {
    let _mat = PERMUTATION_MATRICES[perm_idx as usize];

    let u_rev = perm_idx & 1 != 0;
    let v_rev = perm_idx & 2 != 0;
    let swap = perm_idx & 4 != 0;

    let (out_nu, out_nv) = if swap { (nv, nu) } else { (nu, nv) };

    let mut result = Vec::with_capacity(out_nu * out_nv);
    for ou in 0..out_nu {
        for ov in 0..out_nv {
            // Map output (ou, ov) back to canonical grid indices (gu, gv)
            let (gu, gv) = if swap { (ov, ou) } else { (ou, ov) };
            let gu = if u_rev { nu - 1 - gu } else { gu };
            let gv = if v_rev { nv - 1 - gv } else { gv };
            result.push(grid[gu * nv + gv]);
        }
    }

    (result, out_nu, out_nv)
}

/// Compare two point arrays within tolerance.
///
/// Returns `true` if all corresponding points are within `tol` Euclidean
/// distance. Returns `false` if lengths differ or any point exceeds tolerance.
pub fn verify_match(
    pts_a: &[(Float, Float, Float)],
    pts_b: &[(Float, Float, Float)],
    tol: Float,
) -> bool {
    if pts_a.len() != pts_b.len() {
        return false;
    }
    let tol2 = tol * tol;
    for (a, b) in pts_a.iter().zip(pts_b.iter()) {
        let d2 = (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2);
        if d2 > tol2 {
            return false;
        }
    }
    true
}

/// Count how many points of face B (small, after permutation) match face A (large).
///
/// Face A is the large face, face B is the small face. We apply the permutation
/// to face B and check how many of B's transformed points exist within face A
/// (within tolerance). If all of face B's points match, the larger face A
/// should be split.
///
/// Returns `(match_count, total_b_points)`.
pub fn verify_partial_match(
    grid_a: &[(Float, Float, Float)],
    grid_b_permuted: &[(Float, Float, Float)],
    tol: Float,
) -> (usize, usize) {
    let tol2 = tol * tol;
    let mut count = 0;
    for b in grid_b_permuted {
        for a in grid_a {
            let d2 = (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2);
            if d2 <= tol2 {
                count += 1;
                break;
            }
        }
    }
    (count, grid_b_permuted.len())
}

/// Determine if two faces are in-plane (same constant axis) or cross-plane.
pub fn determine_plane(
    rec_a: &FaceRecord,
    rec_b: &FaceRecord,
) -> crate::face_record::OrientationPlane {
    if rec_a.constant_axis() == rec_b.constant_axis() {
        crate::face_record::OrientationPlane::InPlane
    } else {
        crate::face_record::OrientationPlane::CrossPlane
    }
}

// ── Permutation search ──────────────────────────────────────────────────

/// Try all 8 permutation matrices on `grid_b` to find one that matches `grid_a`.
///
/// For each permutation index 0..8:
/// 1. Apply the permutation to `grid_b` via [`apply_permutation`].
/// 2. Check output shape matches `grid_a`'s shape.
/// 3. Compare point-by-point via [`verify_match`].
///
/// Returns `Some(perm_idx)` on the first match, or `None` if no permutation works.
pub fn try_all_permutations(
    grid_a: &[(Float, Float, Float)],
    nu_a: usize,
    nv_a: usize,
    grid_b: &[(Float, Float, Float)],
    nu_b: usize,
    nv_b: usize,
    tol: Float,
) -> Option<u8> {
    for perm_idx in 0u8..8 {
        let (permuted, out_nu, out_nv) = apply_permutation(grid_b, nu_b, nv_b, perm_idx);

        // Shape check — this is the key fix for cross-plane matches
        if out_nu != nu_a || out_nv != nv_a {
            continue;
        }

        if verify_match(grid_a, &permuted, tol) {
            return Some(perm_idx);
        }
    }
    None
}

// ── Connectivity verification ───────────────────────────────────────────

/// Verify connectivity face matches using permutation matrices.
///
/// GCD-reduce blocks and scale face-match indices to match.
fn prepare_reduced(blocks: &[Block], face_matches: &[FaceMatch]) -> (Vec<Block>, Vec<FaceMatch>) {
    let gcd_to_use = compute_min_gcd(blocks);
    let reduced_blocks = reduce_blocks(blocks, gcd_to_use);
    let scaled_matches: Vec<FaceMatch> = face_matches
        .iter()
        .map(|fm| {
            let mut sfm = fm.clone();
            sfm.divide_indices(gcd_to_use);
            sfm
        })
        .collect();
    (reduced_blocks, scaled_matches)
}

/// For each face match:
/// 1. GCD-reduce blocks and scale indices.
/// 2. Extract both faces as canonical 2D grids.
/// 3. Try stored `permutation_index` first (if available).
/// 4. Fall back to [`try_all_permutations`] if needed.
/// 5. On success, update the `FaceMatch` with the correct `permutation_index`.
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_connectivity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let (reduced_blocks, scaled_matches) = prepare_reduced(blocks, face_matches);

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

        // Extract canonical grids
        let grid_a = match extract_canonical_grid(block1, b1) {
            Some(g) => g,
            None => {
                mismatched.push(face_matches[idx].clone());
                continue;
            }
        };
        let grid_b = match extract_canonical_grid(block2, b2) {
            Some(g) => g,
            None => {
                mismatched.push(face_matches[idx].clone());
                continue;
            }
        };

        let (pts_a, nu_a, nv_a) = grid_a;
        let (pts_b, nu_b, nv_b) = grid_b;

        // Try stored permutation_index first (if available)
        let stored_perm = sfm.orientation.as_ref().map(|o| o.permutation_index());
        if let Some(perm_idx) = stored_perm {
            let (permuted, out_nu, out_nv) = apply_permutation(&pts_b, nu_b, nv_b, perm_idx);
            if out_nu == nu_a && out_nv == nv_a && verify_match(&pts_a, &permuted, tol) {
                verified.push(face_matches[idx].clone());
                continue;
            }
        }

        // Fall back: try all 8 permutations
        if let Some(perm_idx) = try_all_permutations(&pts_a, nu_a, nv_a, &pts_b, nu_b, nv_b, tol) {
            let mut corrected = face_matches[idx].clone();
            corrected.orientation = Some(Orientation::from_perm_index(
                perm_idx,
                b1.constant_axis(),
                b2.constant_axis(),
            ));
            verified.push(corrected);
        } else {
            let orig = &face_matches[idx];
            eprintln!("verify_connectivity: MISMATCH at index {}", idx);
            eprintln!(
                "  block {}: lo=({},{},{}) hi=({},{},{})",
                orig.block1.block_index,
                orig.block1.i_lo(),
                orig.block1.j_lo(),
                orig.block1.k_lo(),
                orig.block1.i_hi(),
                orig.block1.j_hi(),
                orig.block1.k_hi()
            );
            eprintln!(
                "  block {}: lo=({},{},{}) hi=({},{},{})",
                orig.block2.block_index,
                orig.block2.i_lo(),
                orig.block2.j_lo(),
                orig.block2.k_lo(),
                orig.block2.i_hi(),
                orig.block2.j_hi(),
                orig.block2.k_hi()
            );
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}

/// Verify periodic face matches using permutation matrices with rotation.
///
/// For each face match, rotates block1 by +/- theta and then uses the
/// same canonical grid + permutation approach as [`verify_connectivity`].
///
/// # Arguments
/// * `theta` - rotation angle in **radians**
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
    let (reduced_blocks, scaled_matches) = prepare_reduced(blocks, face_matches);

    let rotation_matrix_pos = create_rotation_matrix(theta, rotation_axis);
    let rotation_matrix_neg = create_rotation_matrix(-theta, rotation_axis);

    let rotated_blocks_pos: Vec<Block> = reduced_blocks
        .iter()
        .map(|b| rotate_block(b, rotation_matrix_pos))
        .collect();
    let rotated_blocks_neg: Vec<Block> = reduced_blocks
        .iter()
        .map(|b| rotate_block(b, rotation_matrix_neg))
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

        // Extract face B's canonical grid (unrotated)
        let grid_b = match extract_canonical_grid(block2, b2) {
            Some(g) => g,
            None => {
                mismatched.push(face_matches[idx].clone());
                continue;
            }
        };
        let (pts_b, nu_b, nv_b) = grid_b;

        let mut found = false;

        // Try +theta rotation first, then -theta
        for rotated_blocks in [&rotated_blocks_pos, &rotated_blocks_neg] {
            if found {
                break;
            }

            let block1_rotated = &rotated_blocks[b1_idx];

            // Extract face A's canonical grid (from rotated block)
            let grid_a = match extract_canonical_grid(block1_rotated, b1) {
                Some(g) => g,
                None => continue,
            };
            let (pts_a, nu_a, nv_a) = grid_a;

            // Try stored permutation_index first
            let stored_perm = sfm.orientation.as_ref().map(|o| o.permutation_index());
            if let Some(perm_idx) = stored_perm {
                let (permuted, out_nu, out_nv) = apply_permutation(&pts_b, nu_b, nv_b, perm_idx);
                if out_nu == nu_a && out_nv == nv_a && verify_match(&pts_a, &permuted, tol) {
                    verified.push(face_matches[idx].clone());
                    found = true;
                    break;
                }
            }

            // Fall back: try all 8 permutations
            if let Some(perm_idx) =
                try_all_permutations(&pts_a, nu_a, nv_a, &pts_b, nu_b, nv_b, tol)
            {
                let mut corrected = face_matches[idx].clone();
                corrected.orientation = Some(Orientation::from_perm_index(
                    perm_idx,
                    b1.constant_axis(),
                    b2.constant_axis(),
                ));
                verified.push(corrected);
                found = true;
                break;
            }
        }

        if !found {
            let orig = &face_matches[idx];
            eprintln!("verify_periodicity: MISMATCH at index {}", idx);
            eprintln!(
                "  block {}: lo=({},{},{}) hi=({},{},{})",
                orig.block1.block_index,
                orig.block1.i_lo(),
                orig.block1.j_lo(),
                orig.block1.k_lo(),
                orig.block1.i_hi(),
                orig.block1.j_hi(),
                orig.block1.k_hi()
            );
            eprintln!(
                "  block {}: lo=({},{},{}) hi=({},{},{})",
                orig.block2.block_index,
                orig.block2.i_lo(),
                orig.block2.j_lo(),
                orig.block2.k_lo(),
                orig.block2.i_hi(),
                orig.block2.j_hi(),
                orig.block2.k_hi()
            );
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}
