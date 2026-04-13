//! Finite-volume geometry metrics for structured Plot3D blocks.
//!
//! This module computes the geometric quantities needed by finite-volume CFD
//! solvers on structured hexahedral grids:
//!
//! - **Cell volumes** via the Davies-Salmond divergence-theorem method
//! - **Face area vectors** (projected areas with direction) for I-, J-, K-faces
//! - **Cell centroids** (arithmetic mean of the 8 corner nodes)
//!
//! All outputs are flat `Vec<Float>` arrays in row-major (i-fastest) order so
//! they can be consumed directly by solver kernels without reshaping.

use crate::block::Block;
use crate::Float;

// ---------------------------------------------------------------------------
// Face area vector storage
// ---------------------------------------------------------------------------

/// Projected face-area vectors for all three face families in a structured block.
///
/// In a structured grid with node dimensions `(ni, nj, nk)`, there are three
/// families of interior/boundary faces:
///
/// | Family | Count | Constant index | Varies |
/// |--------|-------|----------------|--------|
/// | I-face | `ni * (nj-1) * (nk-1)` | i = 0..ni-1 | j, k |
/// | J-face | `(ni-1) * nj * (nk-1)` | j = 0..nj-1 | i, k |
/// | K-face | `(ni-1) * (nj-1) * nk` | k = 0..nk-1 | i, j |
///
/// Each face is a quadrilateral formed by 4 grid nodes.  The area vector is
/// half the cross product of the two diagonals of the quad:
///
/// ```text
///   S = 0.5 * (diagonal_1 x diagonal_2)
/// ```
///
/// The sign convention gives outward normals in the +i, +j, +k directions
/// respectively, so the area vector of I-face `(i,j,k)` points from cell
/// `(i-1,j,k)` toward cell `(i,j,k)`.
#[derive(Clone, Debug)]
pub struct FaceMetrics {
    // ------ I-faces (constant-i surfaces) ------
    // Dimensions: ni * (nj-1) * (nk-1)
    // Flat index: i + ni*j + ni*(nj-1)*k  where j in 0..nj-1, k in 0..nk-1
    /// x-component of I-face area vectors.
    pub si_x: Vec<Float>,
    /// y-component of I-face area vectors.
    pub si_y: Vec<Float>,
    /// z-component of I-face area vectors.
    pub si_z: Vec<Float>,

    // ------ J-faces (constant-j surfaces) ------
    // Dimensions: (ni-1) * nj * (nk-1)
    // Flat index: i + (ni-1)*j + (ni-1)*nj*k  where i in 0..ni-1, j in 0..nj, k in 0..nk-1
    /// x-component of J-face area vectors.
    pub sj_x: Vec<Float>,
    /// y-component of J-face area vectors.
    pub sj_y: Vec<Float>,
    /// z-component of J-face area vectors.
    pub sj_z: Vec<Float>,

    // ------ K-faces (constant-k surfaces) ------
    // Dimensions: (ni-1) * (nj-1) * nk
    // Flat index: i + (ni-1)*j + (ni-1)*(nj-1)*k  where i in 0..ni-1, j in 0..nj-1, k in 0..nk
    /// x-component of K-face area vectors.
    pub sk_x: Vec<Float>,
    /// y-component of K-face area vectors.
    pub sk_y: Vec<Float>,
    /// z-component of K-face area vectors.
    pub sk_z: Vec<Float>,
}

// ---------------------------------------------------------------------------
// Cell volumes
// ---------------------------------------------------------------------------

/// Compute the volume of every hexahedral cell in a structured block.
///
/// # Method
///
/// Each cell `(i,j,k)` is a hexahedron bounded by 8 corner nodes.  The volume
/// is computed using the divergence theorem applied to the identity field
/// `F = r` (position vector):
///
/// ```text
///   V = (1/3) * integral_over_surface( r . dS )
/// ```
///
/// For a hexahedron with 6 planar quadrilateral faces this becomes a sum over
/// the 6 faces.  Each face area vector `S_face` is computed as half the cross
/// product of the face diagonals, and the face centroid `r_face` is the
/// average of its 4 corner nodes:
///
/// ```text
///   V = (1/3) * sum_{f=0..5} ( r_face_f . S_face_f )
/// ```
///
/// where the sign convention has outward normals on the +i, +j, +k faces and
/// inward normals on the -i, -j, -k faces (handled by diagonal ordering).
///
/// This is algebraically equivalent to the Davies-Salmond method (AIAA J.,
/// vol. 23, no. 6, pp. 954-956, 1985) and is exact for trilinear hexahedra.
///
/// # Indexing
///
/// Returns a flat `Vec<Float>` of length `(ni-1) * (nj-1) * (nk-1)` where
/// `ni = block.imax`, etc.  Cell `(i,j,k)` with `0 <= i < ni-1` is stored at:
///
/// ```text
///   cell_id = i + (ni-1)*j + (ni-1)*(nj-1)*k
/// ```
pub fn compute_cell_volumes(block: &Block) -> Vec<Float> {
    let ni = block.imax;
    let nj = block.jmax;
    let nk = block.kmax;

    let nci = ni - 1; // number of cells in i
    let ncj = nj - 1;
    let nck = nk - 1;
    let ncells = nci * ncj * nck;

    let mut volumes = vec![0.0 as Float; ncells];

    // Closure: flat node index in block arrays
    let nidx = |i: usize, j: usize, k: usize| -> usize { (k * nj + j) * ni + i };

    // Closure: flat cell index in output array
    let cidx = |i: usize, j: usize, k: usize| -> usize { (k * ncj + j) * nci + i };

    for k in 0..nck {
        for j in 0..ncj {
            for i in 0..nci {
                // ----------------------------------------------------------
                // 8 corner nodes of hex cell (i,j,k)
                //
                //   n0 = (i,   j,   k  )    n4 = (i,   j,   k+1)
                //   n1 = (i+1, j,   k  )    n5 = (i+1, j,   k+1)
                //   n2 = (i,   j+1, k  )    n6 = (i,   j+1, k+1)
                //   n3 = (i+1, j+1, k  )    n7 = (i+1, j+1, k+1)
                // ----------------------------------------------------------
                let p = |ii: usize, jj: usize, kk: usize| -> [Float; 3] {
                    let id = nidx(ii, jj, kk);
                    [block.x[id], block.y[id], block.z[id]]
                };

                let n0 = p(i, j, k);
                let n1 = p(i + 1, j, k);
                let n2 = p(i, j + 1, k);
                let n3 = p(i + 1, j + 1, k);
                let n4 = p(i, j, k + 1);
                let n5 = p(i + 1, j, k + 1);
                let n6 = p(i, j + 1, k + 1);
                let n7 = p(i + 1, j + 1, k + 1);

                // ----------------------------------------------------------
                // 6 faces of the hex cell, each a quadrilateral.
                //
                // For each face we compute:
                //   face_centroid = (1/4) * sum of 4 corners
                //   face_area_vec = 0.5 * (diag1 x diag2)
                //
                // Diagonal ordering is chosen so the area vector points
                // outward for +i/+j/+k faces and inward for -i/-j/-k faces.
                // The divergence theorem then gives V = (1/3) * sum(r . S).
                // ----------------------------------------------------------

                let mut vol = 0.0 as Float;

                // Face list: (corner_a, corner_b, corner_c, corner_d)
                // Diagonals are a-c and b-d. Order matters for sign.
                //
                // I-low  face (i   const): nodes n0, n2, n6, n4  => diag n0-n6, n4-n2
                // I-high face (i+1 const): nodes n1, n3, n7, n5  => diag n1-n7, n3-n5
                // J-low  face (j   const): nodes n0, n1, n5, n4  => diag n0-n5, n1-n4
                // J-high face (j+1 const): nodes n2, n3, n7, n6  => diag n2-n7, n6-n3
                // K-low  face (k   const): nodes n0, n1, n3, n2  => diag n0-n3, n1-n2
                // K-high face (k+1 const): nodes n4, n5, n7, n6  => diag n4-n7, n6-n5

                let faces: [([Float; 3], [Float; 3], [Float; 3], [Float; 3]); 6] = [
                    // I-low:  outward normal points in -i direction
                    // We use diagonal order so cross product points -i,
                    // which when dotted with centroid and summed gives
                    // the correct signed contribution.
                    (n0, n4, n6, n2),
                    // I-high: outward normal points in +i direction
                    (n1, n3, n7, n5),
                    // J-low: outward normal points in -j direction
                    (n0, n1, n5, n4),
                    // J-high: outward normal points in +j direction
                    (n2, n6, n7, n3),
                    // K-low: outward normal points in -k direction
                    (n0, n2, n3, n1),
                    // K-high: outward normal points in +k direction
                    (n4, n5, n7, n6),
                ];

                for (a, b, c, d) in &faces {
                    // Face centroid (un-normalized, factor 1/4 absorbed later)
                    let cx = a[0] + b[0] + c[0] + d[0];
                    let cy = a[1] + b[1] + c[1] + d[1];
                    let cz = a[2] + b[2] + c[2] + d[2];

                    // Diagonals of the quad: a->c and b->d
                    let d1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let d2 = [d[0] - b[0], d[1] - b[1], d[2] - b[2]];

                    // Area vector = 0.5 * (d1 x d2)
                    let sx = 0.5 * (d1[1] * d2[2] - d1[2] * d2[1]);
                    let sy = 0.5 * (d1[2] * d2[0] - d1[0] * d2[2]);
                    let sz = 0.5 * (d1[0] * d2[1] - d1[1] * d2[0]);

                    // Contribution: (1/3) * (centroid/4) . area_vec
                    // = (1/12) * centroid_sum . area_vec
                    vol += cx * sx + cy * sy + cz * sz;
                }

                // Divide by 12: factor of (1/3) from divergence theorem
                // times (1/4) from un-normalized centroid sum.
                volumes[cidx(i, j, k)] = (vol / 12.0).abs();
            }
        }
    }

    volumes
}

// ---------------------------------------------------------------------------
// Face area vectors
// ---------------------------------------------------------------------------

/// Compute projected face-area vectors for all three face families.
///
/// # I-faces (constant-i surfaces)
///
/// An I-face at index `(i, j, k)` is the quadrilateral formed by the 4 nodes:
///
/// ```text
///   (i, j, k),  (i, j+1, k),  (i, j+1, k+1),  (i, j, k+1)
/// ```
///
/// There are `ni` such faces in the i-direction (including the two boundary
/// faces at `i=0` and `i=ni-1`), and `(nj-1) * (nk-1)` faces in each
/// i-plane, giving `ni * (nj-1) * (nk-1)` I-faces total.
///
/// The area vector is:
///
/// ```text
///   S_i = 0.5 * (diag1 x diag2)
/// ```
///
/// where the diagonals connect opposite corners of the quad.  The sign
/// convention points S_i in the +i direction (from cell `i-1` to cell `i`).
///
/// # J-faces and K-faces
///
/// Analogous construction for j-constant and k-constant surfaces.
///
/// # Indexing
///
/// Within each family the flat index is i-fastest:
///
/// - I-face `(i,j,k)`:  `i + ni * j + ni * (nj-1) * k`
///   with `i in 0..ni`, `j in 0..nj-1`, `k in 0..nk-1`
///
/// - J-face `(i,j,k)`:  `i + (ni-1) * j + (ni-1) * nj * k`
///   with `i in 0..ni-1`, `j in 0..nj`, `k in 0..nk-1`
///
/// - K-face `(i,j,k)`:  `i + (ni-1) * j + (ni-1) * (nj-1) * k`
///   with `i in 0..ni-1`, `j in 0..nj-1`, `k in 0..nk`
pub fn compute_face_metrics(block: &Block) -> FaceMetrics {
    let ni = block.imax;
    let nj = block.jmax;
    let nk = block.kmax;

    // Node index helper
    let nidx = |i: usize, j: usize, k: usize| -> usize { (k * nj + j) * ni + i };

    // ---- I-faces: ni * (nj-1) * (nk-1) ----
    let n_ifaces = ni * (nj - 1) * (nk - 1);
    let mut si_x = vec![0.0 as Float; n_ifaces];
    let mut si_y = vec![0.0 as Float; n_ifaces];
    let mut si_z = vec![0.0 as Float; n_ifaces];

    for k in 0..(nk - 1) {
        for j in 0..(nj - 1) {
            for i in 0..ni {
                // Quad corners on the i-constant plane:
                //   p0 = (i, j,   k  )
                //   p1 = (i, j+1, k  )
                //   p2 = (i, j+1, k+1)
                //   p3 = (i, j,   k+1)
                //
                // Diagonals: p0->p2 and p1->p3
                // Cross product gives area vector pointing in +i direction.
                let p0 = nidx(i, j, k);
                let p1 = nidx(i, j + 1, k);
                let p2 = nidx(i, j + 1, k + 1);
                let p3 = nidx(i, j, k + 1);

                let d1x = block.x[p2] - block.x[p0];
                let d1y = block.y[p2] - block.y[p0];
                let d1z = block.z[p2] - block.z[p0];

                let d2x = block.x[p3] - block.x[p1];
                let d2y = block.y[p3] - block.y[p1];
                let d2z = block.z[p3] - block.z[p1];

                let fid = i + ni * j + ni * (nj - 1) * k;
                si_x[fid] = 0.5 * (d1y * d2z - d1z * d2y);
                si_y[fid] = 0.5 * (d1z * d2x - d1x * d2z);
                si_z[fid] = 0.5 * (d1x * d2y - d1y * d2x);
            }
        }
    }

    // ---- J-faces: (ni-1) * nj * (nk-1) ----
    let n_jfaces = (ni - 1) * nj * (nk - 1);
    let mut sj_x = vec![0.0 as Float; n_jfaces];
    let mut sj_y = vec![0.0 as Float; n_jfaces];
    let mut sj_z = vec![0.0 as Float; n_jfaces];

    for k in 0..(nk - 1) {
        for j in 0..nj {
            for i in 0..(ni - 1) {
                // Quad corners on the j-constant plane:
                //   p0 = (i,   j, k  )
                //   p1 = (i,   j, k+1)
                //   p2 = (i+1, j, k+1)
                //   p3 = (i+1, j, k  )
                //
                // Diagonals: p0->p2 and p1->p3
                // Cross product gives area vector pointing in +j direction.
                let p0 = nidx(i, j, k);
                let p1 = nidx(i, j, k + 1);
                let p2 = nidx(i + 1, j, k + 1);
                let p3 = nidx(i + 1, j, k);

                let d1x = block.x[p2] - block.x[p0];
                let d1y = block.y[p2] - block.y[p0];
                let d1z = block.z[p2] - block.z[p0];

                let d2x = block.x[p3] - block.x[p1];
                let d2y = block.y[p3] - block.y[p1];
                let d2z = block.z[p3] - block.z[p1];

                let fid = i + (ni - 1) * j + (ni - 1) * nj * k;
                sj_x[fid] = 0.5 * (d1y * d2z - d1z * d2y);
                sj_y[fid] = 0.5 * (d1z * d2x - d1x * d2z);
                sj_z[fid] = 0.5 * (d1x * d2y - d1y * d2x);
            }
        }
    }

    // ---- K-faces: (ni-1) * (nj-1) * nk ----
    let n_kfaces = (ni - 1) * (nj - 1) * nk;
    let mut sk_x = vec![0.0 as Float; n_kfaces];
    let mut sk_y = vec![0.0 as Float; n_kfaces];
    let mut sk_z = vec![0.0 as Float; n_kfaces];

    for k in 0..nk {
        for j in 0..(nj - 1) {
            for i in 0..(ni - 1) {
                // Quad corners on the k-constant plane:
                //   p0 = (i,   j,   k)
                //   p1 = (i+1, j,   k)
                //   p2 = (i+1, j+1, k)
                //   p3 = (i,   j+1, k)
                //
                // Diagonals: p0->p2 and p1->p3
                // Cross product gives area vector pointing in +k direction.
                let p0 = nidx(i, j, k);
                let p1 = nidx(i + 1, j, k);
                let p2 = nidx(i + 1, j + 1, k);
                let p3 = nidx(i, j + 1, k);

                let d1x = block.x[p2] - block.x[p0];
                let d1y = block.y[p2] - block.y[p0];
                let d1z = block.z[p2] - block.z[p0];

                let d2x = block.x[p3] - block.x[p1];
                let d2y = block.y[p3] - block.y[p1];
                let d2z = block.z[p3] - block.z[p1];

                let fid = i + (ni - 1) * j + (ni - 1) * (nj - 1) * k;
                sk_x[fid] = 0.5 * (d1y * d2z - d1z * d2y);
                sk_y[fid] = 0.5 * (d1z * d2x - d1x * d2z);
                sk_z[fid] = 0.5 * (d1x * d2y - d1y * d2x);
            }
        }
    }

    FaceMetrics {
        si_x, si_y, si_z,
        sj_x, sj_y, sj_z,
        sk_x, sk_y, sk_z,
    }
}

// ---------------------------------------------------------------------------
// Cell centers
// ---------------------------------------------------------------------------

/// Compute the geometric center of every cell as the arithmetic mean of its
/// 8 corner node coordinates.
///
/// # Returns
///
/// A tuple `(xc, yc, zc)` of flat `Vec<Float>`, each of length
/// `(ni-1) * (nj-1) * (nk-1)`.  Cell `(i,j,k)` is stored at flat index:
///
/// ```text
///   cell_id = i + (ni-1)*j + (ni-1)*(nj-1)*k
/// ```
///
/// where `0 <= i < ni-1`, `0 <= j < nj-1`, `0 <= k < nk-1`.
///
/// The cell center is simply:
///
/// ```text
///   x_c = (1/8) * sum of x-coordinates of the 8 corner nodes
/// ```
///
/// (and analogously for y and z).  This is exact for parallelepipeds and a
/// reasonable approximation for mildly skewed hexahedra.
pub fn compute_cell_centers(block: &Block) -> (Vec<Float>, Vec<Float>, Vec<Float>) {
    let ni = block.imax;
    let nj = block.jmax;
    let nk = block.kmax;

    let nci = ni - 1;
    let ncj = nj - 1;
    let nck = nk - 1;
    let ncells = nci * ncj * nck;

    let mut xc = vec![0.0 as Float; ncells];
    let mut yc = vec![0.0 as Float; ncells];
    let mut zc = vec![0.0 as Float; ncells];

    // Node index helper
    let nidx = |i: usize, j: usize, k: usize| -> usize { (k * nj + j) * ni + i };

    // Cell index helper
    let cidx = |i: usize, j: usize, k: usize| -> usize { (k * ncj + j) * nci + i };

    let eighth: Float = 0.125;

    for k in 0..nck {
        for j in 0..ncj {
            for i in 0..nci {
                let cid = cidx(i, j, k);

                // Indices of the 8 corner nodes
                let n0 = nidx(i, j, k);
                let n1 = nidx(i + 1, j, k);
                let n2 = nidx(i, j + 1, k);
                let n3 = nidx(i + 1, j + 1, k);
                let n4 = nidx(i, j, k + 1);
                let n5 = nidx(i + 1, j, k + 1);
                let n6 = nidx(i, j + 1, k + 1);
                let n7 = nidx(i + 1, j + 1, k + 1);

                xc[cid] = eighth
                    * (block.x[n0] + block.x[n1] + block.x[n2] + block.x[n3]
                        + block.x[n4] + block.x[n5] + block.x[n6] + block.x[n7]);

                yc[cid] = eighth
                    * (block.y[n0] + block.y[n1] + block.y[n2] + block.y[n3]
                        + block.y[n4] + block.y[n5] + block.y[n6] + block.y[n7]);

                zc[cid] = eighth
                    * (block.z[n0] + block.z[n1] + block.z[n2] + block.z[n3]
                        + block.z[n4] + block.z[n5] + block.z[n6] + block.z[n7]);
            }
        }
    }

    (xc, yc, zc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;

    /// Build a unit cube block with `n` nodes per edge (n >= 2).
    /// Grid is uniformly spaced [0,1]^3.
    fn unit_cube_block(n: usize) -> Block {
        let total = n * n * n;
        let mut x = Vec::with_capacity(total);
        let mut y = Vec::with_capacity(total);
        let mut z = Vec::with_capacity(total);
        let h = 1.0 / (n as f64 - 1.0);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    x.push(i as f64 * h);
                    y.push(j as f64 * h);
                    z.push(k as f64 * h);
                }
            }
        }
        Block::new(n, n, n, x, y, z)
    }

    #[test]
    fn test_cell_volumes_unit_cube() {
        // A single cell: 2x2x2 nodes => 1 cell of volume 1.0
        let block = unit_cube_block(2);
        let vols = compute_cell_volumes(&block);
        assert_eq!(vols.len(), 1);
        assert!((vols[0] - 1.0).abs() < 1e-12, "Expected volume 1.0, got {}", vols[0]);
    }

    #[test]
    fn test_cell_volumes_subdivided() {
        // 3x3x3 nodes => 2x2x2 = 8 cells, each of volume 0.125
        let block = unit_cube_block(3);
        let vols = compute_cell_volumes(&block);
        assert_eq!(vols.len(), 8);
        for (idx, v) in vols.iter().enumerate() {
            assert!(
                (v - 0.125).abs() < 1e-12,
                "Cell {} expected volume 0.125, got {}",
                idx,
                v
            );
        }
    }

    #[test]
    fn test_cell_volumes_total() {
        // 4x4x4 nodes => 27 cells, total volume should be 1.0
        let block = unit_cube_block(4);
        let vols = compute_cell_volumes(&block);
        assert_eq!(vols.len(), 27);
        let total: f64 = vols.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "Total volume {}", total);
    }

    #[test]
    fn test_cell_centers_unit_cube() {
        // 2x2x2 nodes => 1 cell, center at (0.5, 0.5, 0.5)
        let block = unit_cube_block(2);
        let (xc, yc, zc) = compute_cell_centers(&block);
        assert_eq!(xc.len(), 1);
        assert!((xc[0] - 0.5).abs() < 1e-12);
        assert!((yc[0] - 0.5).abs() < 1e-12);
        assert!((zc[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_cell_centers_subdivided() {
        // 3x3x3 => 8 cells. Cell (0,0,0) should have center (0.25, 0.25, 0.25).
        let block = unit_cube_block(3);
        let (xc, yc, zc) = compute_cell_centers(&block);
        assert_eq!(xc.len(), 8);
        // Cell (0,0,0)
        assert!((xc[0] - 0.25).abs() < 1e-12);
        assert!((yc[0] - 0.25).abs() < 1e-12);
        assert!((zc[0] - 0.25).abs() < 1e-12);
        // Cell (1,1,1) at index 1 + 2*1 + 2*2*1 = 7
        assert!((xc[7] - 0.75).abs() < 1e-12);
        assert!((yc[7] - 0.75).abs() < 1e-12);
        assert!((zc[7] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_face_metrics_unit_cube() {
        // 2x2x2 nodes => each face family has certain counts.
        // I-faces: 2 * 1 * 1 = 2
        // J-faces: 1 * 2 * 1 = 2
        // K-faces: 1 * 1 * 2 = 2
        let block = unit_cube_block(2);
        let fm = compute_face_metrics(&block);

        assert_eq!(fm.si_x.len(), 2);
        assert_eq!(fm.sj_x.len(), 2);
        assert_eq!(fm.sk_x.len(), 2);

        // For a unit cube, I-faces should have area vector magnitude 1.0
        // pointing in x-direction: si_x = 1.0, si_y = 0, si_z = 0
        for idx in 0..2 {
            assert!(
                (fm.si_x[idx].abs() - 1.0).abs() < 1e-12,
                "I-face {} si_x = {}",
                idx,
                fm.si_x[idx]
            );
            assert!(fm.si_y[idx].abs() < 1e-12);
            assert!(fm.si_z[idx].abs() < 1e-12);
        }

        // J-faces: area vector in y-direction
        for idx in 0..2 {
            assert!(fm.sj_x[idx].abs() < 1e-12);
            assert!(
                (fm.sj_y[idx].abs() - 1.0).abs() < 1e-12,
                "J-face {} sj_y = {}",
                idx,
                fm.sj_y[idx]
            );
            assert!(fm.sj_z[idx].abs() < 1e-12);
        }

        // K-faces: area vector in z-direction
        for idx in 0..2 {
            assert!(fm.sk_x[idx].abs() < 1e-12);
            assert!(fm.sk_y[idx].abs() < 1e-12);
            assert!(
                (fm.sk_z[idx].abs() - 1.0).abs() < 1e-12,
                "K-face {} sk_z = {}",
                idx,
                fm.sk_z[idx]
            );
        }
    }

    #[test]
    fn test_face_metrics_count_subdivided() {
        // 4x3x5 nodes
        let ni = 4;
        let nj = 3;
        let nk = 5;
        let total = ni * nj * nk;
        let x: Vec<f64> = (0..total).map(|_| 0.0).collect();
        let y = x.clone();
        let z = x.clone();
        let block = Block::new(ni, nj, nk, x, y, z);
        let fm = compute_face_metrics(&block);

        assert_eq!(fm.si_x.len(), ni * (nj - 1) * (nk - 1));
        assert_eq!(fm.sj_x.len(), (ni - 1) * nj * (nk - 1));
        assert_eq!(fm.sk_x.len(), (ni - 1) * (nj - 1) * nk);
    }
}
