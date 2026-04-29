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
//! - [`verify_translational_periodicity`] — verify periodic face matches with translation
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
use crate::face_record::{
    FaceMatch, FaceRecord, Orientation, OrientationPlane, PERMUTATION_MATRICES,
};
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

/// Compute the maximum Euclidean distance between corresponding points.
///
/// Returns `Float::MAX` if the arrays differ in length.
fn max_point_distance(
    pts_a: &[(Float, Float, Float)],
    pts_b: &[(Float, Float, Float)],
) -> Float {
    if pts_a.len() != pts_b.len() {
        return Float::MAX;
    }
    let mut max_d2: Float = 0.0;
    for (a, b) in pts_a.iter().zip(pts_b.iter()) {
        let d2 = (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2);
        if d2 > max_d2 {
            max_d2 = d2;
        }
    }
    max_d2.sqrt()
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
pub fn determine_plane(rec_a: &FaceRecord, rec_b: &FaceRecord) -> OrientationPlane {
    if rec_a.constant_axis() == rec_b.constant_axis() {
        OrientationPlane::InPlane
    } else {
        OrientationPlane::CrossPlane
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
        let stored_perm = sfm.orientation.as_ref().map(|o| o.permutation_index);
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
            let plane = determine_plane(b1, b2);
            corrected.orientation = Some(Orientation {
                permutation_index: perm_idx,
                plane,
            });
            verified.push(corrected);
        } else {
            // Diagnostic dump gated on env var: in a cascade pipeline
            // (load.rs::load_mesh) this verifier is the FIRST stage —
            // matches that need translational or rotational verification
            // legitimately fail here and fall through. Routine misses
            // would otherwise spam stderr (~1k lines on CMC009 rf=1).
            // Set `PLOT3D_RS_VERIFY_CONNECTIVITY_VERBOSE=1` to debug.
            if std::env::var("PLOT3D_RS_VERIFY_CONNECTIVITY_VERBOSE").as_deref() == Ok("1") {
                let orig = &face_matches[idx];
                let ca1 = b1.constant_axis();
                let ca2 = b2.constant_axis();
                let axis_label = |a: Option<usize>| match a {
                    Some(0) => "I", Some(1) => "J", Some(2) => "K", _ => "?"
                };
                let cross_tag = if ca1 != ca2 { "CROSS-AXIS" } else { "SAME-AXIS" };
                let mut best_dist: Float = Float::MAX;
                for p in 0u8..8 {
                    let (permuted, out_nu, out_nv) = apply_permutation(&pts_b, nu_b, nv_b, p);
                    if out_nu != nu_a || out_nv != nv_a { continue; }
                    let d = max_point_distance(&pts_a, &permuted);
                    if d < best_dist { best_dist = d; }
                }
                eprintln!("verify_connectivity: MISMATCH at index {} [{}]", idx, cross_tag);
                eprintln!(
                    "  block {}: lo=({},{},{}) hi=({},{},{}) const={}",
                    orig.block1.block_index,
                    orig.block1.i_lo(), orig.block1.j_lo(), orig.block1.k_lo(),
                    orig.block1.i_hi(), orig.block1.j_hi(), orig.block1.k_hi(),
                    axis_label(ca1)
                );
                eprintln!(
                    "  block {}: lo=({},{},{}) hi=({},{},{}) const={}",
                    orig.block2.block_index,
                    orig.block2.i_lo(), orig.block2.j_lo(), orig.block2.k_lo(),
                    orig.block2.i_hi(), orig.block2.j_hi(), orig.block2.k_hi(),
                    axis_label(ca2)
                );
                eprintln!("  grid_a: {}x{}, grid_b: {}x{}, best_dist: {:.6e}", nu_a, nv_a, nu_b, nv_b, best_dist);
            }
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
        let mut best_dist: Float = Float::MAX;
        let mut best_dims: Option<(usize, usize, usize, usize)> = None;

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

            // Track grid dimensions for diagnostics
            if best_dims.is_none() {
                best_dims = Some((nu_a, nv_a, nu_b, nv_b));
            }

            // Try stored permutation_index first
            let stored_perm = sfm.orientation.as_ref().map(|o| o.permutation_index);
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
                let plane = determine_plane(b1, b2);
                corrected.orientation = Some(Orientation {
                    permutation_index: perm_idx,
                    plane,
                });
                verified.push(corrected);
                found = true;
                break;
            }

            // Track best distance for diagnostics
            for p in 0u8..8 {
                let (permuted, out_nu, out_nv) = apply_permutation(&pts_b, nu_b, nv_b, p);
                if out_nu != nu_a || out_nv != nv_a { continue; }
                let d = max_point_distance(&pts_a, &permuted);
                if d < best_dist { best_dist = d; }
            }
        }

        if !found {
            // Diagnostic gated on env var (cascade misses are routine).
            // Set `PLOT3D_RS_VERIFY_PERIODICITY_VERBOSE=1` to debug.
            if std::env::var("PLOT3D_RS_VERIFY_PERIODICITY_VERBOSE").as_deref() == Ok("1") {
                let orig = &face_matches[idx];
                let ca1 = b1.constant_axis();
                let ca2 = b2.constant_axis();
                let axis_label = |a: Option<usize>| match a {
                    Some(0) => "I", Some(1) => "J", Some(2) => "K", _ => "?"
                };
                let cross_tag = if ca1 != ca2 { "CROSS-AXIS" } else { "SAME-AXIS" };
                eprintln!("verify_periodicity: MISMATCH at index {} [{}]", idx, cross_tag);
                eprintln!(
                    "  block {}: lo=({},{},{}) hi=({},{},{}) const={}",
                    orig.block1.block_index,
                    orig.block1.i_lo(), orig.block1.j_lo(), orig.block1.k_lo(),
                    orig.block1.i_hi(), orig.block1.j_hi(), orig.block1.k_hi(),
                    axis_label(ca1)
                );
                eprintln!(
                    "  block {}: lo=({},{},{}) hi=({},{},{}) const={}",
                    orig.block2.block_index,
                    orig.block2.i_lo(), orig.block2.j_lo(), orig.block2.k_lo(),
                    orig.block2.i_hi(), orig.block2.j_hi(), orig.block2.k_hi(),
                    axis_label(ca2)
                );
                if let Some((nua, nva, nub, nvb)) = best_dims {
                    eprintln!("  grid_a: {}x{}, grid_b: {}x{}, best_dist: {:.6e}", nua, nva, nub, nvb, best_dist);
                }
            }
            let _ = best_dist; let _ = best_dims;
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}

/// Verify face-match list against the grid under TRANSLATIONAL
/// periodicity (e.g. blade-pitch in y, span height in z).
///
/// Mirror of [`verify_periodicity`] but uses **translation** instead
/// of **rotation**: shifts every block by `±delta` along `axis`
/// (`'x' | 'y' | 'z'`), then for each face_match tries all 8
/// permutations against the shifted-block grid.
///
/// Use this for face_matches that are translationally periodic — i.e.,
/// the two faces don't physically coincide in the original mesh, but
/// they DO coincide once one block is translated by the periodicity
/// vector. CMC009 has translational periodicity in **both Y (pitch)**
/// and **Z (span height)**; call once for each direction in a cascading
/// pipeline:
///
/// ```ignore
/// let (verified_y, leftover_after_y) =
///     verify_translational_periodicity(&blocks, &leftover_from_verify_connectivity, None, 'y', 1.0e-6);
/// let (verified_z, still_unverified) =
///     verify_translational_periodicity(&blocks, &leftover_after_y, None, 'z', 1.0e-6);
/// ```
///
/// `delta` is the magnitude of the translation along `axis`.
///
/// **`None` triggers PER-MATCH auto-detect**: for each face_match, the
/// shift is computed as the difference between the centroids of face A
/// and face B projected onto `axis`. This is the geometrically-correct
/// per-match displacement and works for both:
///   * **Same-block self-loops** (e.g. block 589's k=0 ↔ k=4 face,
///     where Δ_z = block-z-extent)
///   * **Cross-block translational pitch matches** (where Δ is the
///     blade pitch in y or the span height in z)
///
/// A globally-fixed `Some(delta)` is supported for callers that want
/// a single mesh-wide pitch — but the per-match auto-detect is
/// generally more robust because it adapts to whatever shift each
/// individual face_match actually requires.
///
/// # Returns
///
/// `(verified, mismatched)` — verified face_matches have their
/// [`FaceMatch::orientation`] populated with the winning
/// `permutation_index` and the appropriate
/// [`OrientationPlane`] discriminator (`InPlane` for same-axis matches,
/// `CrossPlane` for axis-swapped). Mismatched face_matches are
/// returned for the caller to retry against another verifier
/// (e.g., the rotational [`verify_periodicity`]).
pub fn verify_translational_periodicity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    delta: Option<Float>,
    axis: char,
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let (reduced_blocks, scaled_matches) = prepare_reduced(blocks, face_matches);

    let axis_idx = match axis {
        'x' | 'X' => 0usize,
        'y' | 'Y' => 1usize,
        'z' | 'Z' => 2usize,
        _ => panic!("verify_translational_periodicity: invalid axis {:?}", axis),
    };

    // Helper: face centroid along `axis_idx`. Walks the FaceRecord's
    // (lb..ub) range on the parent block and averages the coordinate.
    // Uses `extract_canonical_grid` indirectly by going through the
    // block's corner coords — cheap and avoids re-extracting the full
    // canonical grid just for centroid computation.
    let face_axis_centroid = |block: &Block, rec: &FaceRecord| -> Float {
        // FaceRecord has `i_lo, i_hi, j_lo, j_hi, k_lo, k_hi` accessors.
        // Walk the inclusive range and average. Reduced blocks have
        // small node counts so this is cheap.
        let (il, jh, kl) = (rec.i_lo(), rec.j_lo(), rec.k_lo());
        let (ih, jl, kh) = (rec.i_hi(), rec.j_hi(), rec.k_hi());
        // FaceRecord may have lb > ub for direction-flipped faces;
        // normalise to ascending for the centroid walk.
        let (i0, i1) = if il <= ih { (il, ih) } else { (ih, il) };
        let (j0, j1) = if jl <= jh { (jl, jh) } else { (jh, jl) };
        let (k0, k1) = if kl <= kh { (kl, kh) } else { (kh, kl) };
        let mut sum: Float = 0.0;
        let mut n: usize = 0;
        for k in k0..=k1 {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let (x, y, z) = block.xyz(i, j, k);
                    let v = match axis_idx {
                        0 => x,
                        1 => y,
                        _ => z,
                    };
                    sum += v;
                    n += 1;
                }
            }
        }
        sum / (n.max(1) as Float)
    };

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

        // Per-match Δ: when caller didn't pin a global delta, compute
        // the geometrically-required shift directly from the centroids
        // of face A vs face B (projected onto the requested axis).
        // This handles same-block self-loops and cross-block pitch
        // matches uniformly.
        let delta_axis = match delta {
            Some(d) => d,
            None => {
                let c1 = face_axis_centroid(block1, b1);
                let c2 = face_axis_centroid(block2, b2);
                // We want to shift block 1 onto block 2: Δ = c2 - c1.
                // The match loop below tries both +Δ and -Δ, so the
                // sign is irrelevant — store the absolute value.
                (c2 - c1).abs()
            }
        };
        // Skip degenerate Δ (≈0): means the two faces are already
        // approximately coincident along this axis — they wouldn't
        // need a translational verifier; let them fall through.
        if delta_axis.abs() < tol {
            mismatched.push(face_matches[idx].clone());
            continue;
        }
        let block1_shifted_pos = block1.shifted(delta_axis, axis);
        let block1_shifted_neg = block1.shifted(-delta_axis, axis);

        // Face B's canonical grid (un-shifted).
        let grid_b = match extract_canonical_grid(block2, b2) {
            Some(g) => g,
            None => {
                mismatched.push(face_matches[idx].clone());
                continue;
            }
        };
        let (pts_b, nu_b, nv_b) = grid_b;

        let mut found = false;
        let mut best_dist: Float = Float::MAX;
        let mut best_dims: Option<(usize, usize, usize, usize)> = None;

        // Try +delta translation first, then -delta.
        for block1_shifted in [&block1_shifted_pos, &block1_shifted_neg] {
            if found {
                break;
            }


            let grid_a = match extract_canonical_grid(block1_shifted, b1) {
                Some(g) => g,
                None => continue,
            };
            let (pts_a, nu_a, nv_a) = grid_a;

            if best_dims.is_none() {
                best_dims = Some((nu_a, nv_a, nu_b, nv_b));
            }

            // Try stored permutation_index first (fast-path for
            // already-verified matches).
            let stored_perm = sfm.orientation.as_ref().map(|o| o.permutation_index);
            if let Some(perm_idx) = stored_perm {
                let (permuted, out_nu, out_nv) =
                    apply_permutation(&pts_b, nu_b, nv_b, perm_idx);
                if out_nu == nu_a && out_nv == nv_a && verify_match(&pts_a, &permuted, tol) {
                    verified.push(face_matches[idx].clone());
                    found = true;
                    break;
                }
            }

            // Fall back: try all 8 permutations.
            if let Some(perm_idx) =
                try_all_permutations(&pts_a, nu_a, nv_a, &pts_b, nu_b, nv_b, tol)
            {
                let mut corrected = face_matches[idx].clone();
                let plane = determine_plane(b1, b2);
                corrected.orientation = Some(Orientation {
                    permutation_index: perm_idx,
                    plane,
                });
                verified.push(corrected);
                found = true;
                break;
            }

            // Diagnostic: track best distance across all 8 perms.
            for p in 0u8..8 {
                let (permuted, out_nu, out_nv) =
                    apply_permutation(&pts_b, nu_b, nv_b, p);
                if out_nu != nu_a || out_nv != nv_a {
                    continue;
                }
                let d = max_point_distance(&pts_a, &permuted);
                if d < best_dist {
                    best_dist = d;
                }
            }
        }

        if !found {
            // Quiet on miss — caller will retry against the next
            // verifier in the cascade. Keep diagnostics behind an
            // env var to avoid spamming production runs that legitimately
            // route some matches to a different verifier.
            if std::env::var("PLOT3D_RS_VERIFY_TRANSLATIONAL_VERBOSE").as_deref() == Ok("1") {
                let orig = &face_matches[idx];
                let ca1 = b1.constant_axis();
                let ca2 = b2.constant_axis();
                let axis_label = |a: Option<usize>| match a {
                    Some(0) => "I",
                    Some(1) => "J",
                    Some(2) => "K",
                    _ => "?",
                };
                let cross_tag = if ca1 != ca2 { "CROSS-AXIS" } else { "SAME-AXIS" };
                eprintln!(
                    "verify_translational_periodicity[{}, Δ_per_match={:+.3e}]: \
                     MISMATCH at index {} [{}]",
                    axis, delta_axis, idx, cross_tag,
                );
                eprintln!(
                    "  block {}: lo=({},{},{}) hi=({},{},{}) const={}",
                    orig.block1.block_index,
                    orig.block1.i_lo(), orig.block1.j_lo(), orig.block1.k_lo(),
                    orig.block1.i_hi(), orig.block1.j_hi(), orig.block1.k_hi(),
                    axis_label(ca1),
                );
                eprintln!(
                    "  block {}: lo=({},{},{}) hi=({},{},{}) const={}",
                    orig.block2.block_index,
                    orig.block2.i_lo(), orig.block2.j_lo(), orig.block2.k_lo(),
                    orig.block2.i_hi(), orig.block2.j_hi(), orig.block2.k_hi(),
                    axis_label(ca2),
                );
                if let Some((nua, nva, nub, nvb)) = best_dims {
                    eprintln!(
                        "  grid_a: {}x{}, grid_b: {}x{}, best_dist: {:.6e}",
                        nua, nva, nub, nvb, best_dist,
                    );
                }
            }
            // Suppress unused warning when env var is absent.
            let _ = best_dims;
            mismatched.push(face_matches[idx].clone());
        }
    }

    (verified, mismatched)
}
