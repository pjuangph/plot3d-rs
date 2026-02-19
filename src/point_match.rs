//! Point matching utilities for structured grid faces.

use crate::Float;

/// Find the indices of the closest point in a 2D face grid to `(x, y, z)`.
///
/// The face grid is represented by flat arrays `x2`, `y2`, `z2` of length
/// `nu * nv`, stored row-major (u varies fastest).
///
/// Returns `Some((u, v))` if the minimum distance is within `tol`,
/// otherwise `None`.
pub fn point_match(
    x: Float,
    y: Float,
    z: Float,
    x2: &[Float],
    y2: &[Float],
    z2: &[Float],
    nu: usize,
    _nv: usize,
    tol: Float,
) -> Option<(usize, usize)> {
    debug_assert_eq!(x2.len(), y2.len());
    debug_assert_eq!(x2.len(), z2.len());

    let mut best_dist = Float::INFINITY;
    let mut best_idx = 0usize;

    for (idx, ((&px, &py), &pz)) in x2.iter().zip(y2.iter()).zip(z2.iter()).enumerate() {
        let dx = x - px;
        let dy = y - py;
        let dz = z - pz;
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < best_dist {
            best_dist = d;
            best_idx = idx;
        }
    }

    if best_dist < tol {
        let u = best_idx % nu;
        let v = best_idx / nu;
        Some((u, v))
    } else {
        None
    }
}
