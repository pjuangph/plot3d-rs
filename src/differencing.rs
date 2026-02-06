//! Forward and backward differencing for structured grid edges.

use crate::block::Block;

/// Backward and forward difference pair: `(backward, forward)`.
/// Each is a displacement vector `[dx, dy, dz]`.
pub type DiffPair = ([f64; 3], [f64; 3]);

/// Differencing data at a single node on a 2D face.
#[derive(Clone, Debug)]
pub struct FaceDiff {
    pub p: usize,
    pub q: usize,
    /// Backward and forward differences along p.
    pub dp: DiffPair,
    /// Backward and forward differences along q.
    pub dq: DiffPair,
}

/// Differencing data at a single node in a 3D block.
#[derive(Clone, Debug)]
pub struct BlockDiff {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Backward and forward differences along i.
    pub di: DiffPair,
    /// Backward and forward differences along j.
    pub dj: DiffPair,
    /// Backward and forward differences along k.
    pub dk: DiffPair,
}

/// Compute forward and backward differences along each direction for a 2D face.
///
/// `x`, `y`, `z` are flat arrays of length `pmax * qmax`, stored row-major
/// (p varies fastest within each row of q).
pub fn find_face_edges(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    pmax: usize,
    qmax: usize,
) -> Vec<FaceDiff> {
    let idx = |p: usize, q: usize| -> usize { q * pmax + p };
    let mut result = Vec::with_capacity(pmax * qmax);

    for p in 0..pmax {
        for q in 0..qmax {
            let id = idx(p, q);

            // dp: backward and forward along p
            let dp_b = if p > 0 {
                let prev = idx(p - 1, q);
                [x[prev] - x[id], y[prev] - y[id], z[prev] - z[id]]
            } else {
                [0.0, 0.0, 0.0]
            };
            let dp_f = if p < pmax - 1 {
                let next = idx(p + 1, q);
                [x[next] - x[id], y[next] - y[id], z[next] - z[id]]
            } else {
                [0.0, 0.0, 0.0]
            };

            // dq: backward and forward along q
            let dq_b = if q > 0 {
                let prev = idx(p, q - 1);
                [x[prev] - x[id], y[prev] - y[id], z[prev] - z[id]]
            } else {
                [0.0, 0.0, 0.0]
            };
            let dq_f = if q < qmax - 1 {
                let next = idx(p, q + 1);
                [x[next] - x[id], y[next] - y[id], z[next] - z[id]]
            } else {
                [0.0, 0.0, 0.0]
            };

            result.push(FaceDiff {
                p,
                q,
                dp: (dp_b, dp_f),
                dq: (dq_b, dq_f),
            });
        }
    }
    result
}

/// Compute forward and backward differences along each direction for a 3D block.
pub fn find_edges(block: &Block) -> Vec<BlockDiff> {
    let (ni, nj, nk) = (block.imax, block.jmax, block.kmax);
    let mut result = Vec::with_capacity(ni * nj * nk);

    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let (cx, cy, cz) = block.xyz(i, j, k);

                // di: backward and forward along i
                let di_b = if i > 0 {
                    let (px, py, pz) = block.xyz(i - 1, j, k);
                    [px - cx, py - cy, pz - cz]
                } else {
                    [0.0, 0.0, 0.0]
                };
                let di_f = if i < ni - 1 {
                    let (px, py, pz) = block.xyz(i + 1, j, k);
                    [px - cx, py - cy, pz - cz]
                } else {
                    [0.0, 0.0, 0.0]
                };

                // dj: backward and forward along j
                let dj_b = if j > 0 {
                    let (px, py, pz) = block.xyz(i, j - 1, k);
                    [px - cx, py - cy, pz - cz]
                } else {
                    [0.0, 0.0, 0.0]
                };
                let dj_f = if j < nj - 1 {
                    let (px, py, pz) = block.xyz(i, j + 1, k);
                    [px - cx, py - cy, pz - cz]
                } else {
                    [0.0, 0.0, 0.0]
                };

                // dk: backward and forward along k
                let dk_b = if k > 0 {
                    let (px, py, pz) = block.xyz(i, j, k - 1);
                    [px - cx, py - cy, pz - cz]
                } else {
                    [0.0, 0.0, 0.0]
                };
                let dk_f = if k < nk - 1 {
                    let (px, py, pz) = block.xyz(i, j, k + 1);
                    [px - cx, py - cy, pz - cz]
                } else {
                    [0.0, 0.0, 0.0]
                };

                result.push(BlockDiff {
                    i,
                    j,
                    k,
                    di: (di_b, di_f),
                    dj: (dj_b, dj_f),
                    dk: (dk_b, dk_f),
                });
            }
        }
    }
    result
}
