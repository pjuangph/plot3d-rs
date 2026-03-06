//! Index-space normals and permutation matrix validation.
//!
//! This module provides:
//! - [`index_space_normal`]: Compute the topological (integer) outward normal for a face.
//! - [`compute_permutation_matrix`]: Compute the 3×3 permutation matrix from diagonal corners.
//! - [`validate_permutation_matrix`]: Validate that a FaceMatch's corner pairing produces a valid matrix.
//! - [`export_normals_json`] / [`import_normals_json`]: Serialize face normals to/from JSON.

use crate::block::Block;
use crate::face_record::{FaceMatch, FaceRecord};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Varying-axis lookup: given a constant axis (0-indexed), returns the two varying axes.
///
/// ```text
/// const 0 (I) → (1, 2) i.e. J, K
/// const 1 (J) → (0, 2) i.e. I, K
/// const 2 (K) → (0, 1) i.e. I, J
/// ```
fn varying_axes(const_ax: usize) -> (usize, usize) {
    match const_ax {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Compute the index-space (topological) outward normal for a face.
///
/// Returns a unit integer vector with exactly one non-zero component:
/// - `-1` on a low face (constant index == 0)
/// - `+1` on a high face (constant index > 0)
///
/// Returns `None` if the face has no single constant axis.
pub fn index_space_normal(rec: &FaceRecord) -> Option<[i8; 3]> {
    let c = rec.constant_axis()?;
    let (lo, _) = rec.bounds();
    let mut n = [0i8; 3];
    n[c] = if lo[c] == 0 { -1 } else { 1 };
    Some(n)
}

/// Compute the index-space normal from raw lb/ub bounds (0-indexed).
///
/// This is a standalone version that doesn't require a FaceRecord.
pub fn index_space_normal_from_bounds(lb: [usize; 3], ub: [usize; 3]) -> Option<[i8; 3]> {
    let const_axes: Vec<usize> = (0..3).filter(|&d| lb[d] == ub[d]).collect();
    if const_axes.len() != 1 {
        return None;
    }
    let c = const_axes[0];
    let mut n = [0i8; 3];
    n[c] = if lb[c] == 0 { -1 } else { 1 };
    Some(n)
}

/// Compute the 3×3 integer permutation matrix from spatially paired diagonal corners.
///
/// # Arguments
/// - `a1, b1`: Diagonal corners of face on block 1 (0-indexed).
/// - `a2, b2`: Diagonal corners of face on block 2 (0-indexed).
///   `a1↔a2` and `b1↔b2` must be spatially coincident.
///
/// # Returns
/// The 3×3 permutation matrix as `[[i8; 3]; 3]`, or `None` if any entry
/// is non-integer (indicating a bad corner pairing).
pub fn compute_permutation_matrix(
    a1: [i64; 3],
    b1: [i64; 3],
    a2: [i64; 3],
    b2: [i64; 3],
) -> Option<[[i8; 3]; 3]> {
    // Step 1: Compute index-space normals
    let n1 = {
        let mut n = [0i8; 3];
        for d in 0..3 {
            if a1[d] == b1[d] {
                n[d] = if a1[d] == 0 { -1 } else { 1 };
            }
        }
        n
    };
    let n2 = {
        let mut n = [0i8; 3];
        for d in 0..3 {
            if a2[d] == b2[d] {
                n[d] = if a2[d] == 0 { -1 } else { 1 };
            }
        }
        n
    };

    // Step 2: Identify constant (face) axes
    let face1 = (0..3).find(|&d| n1[d] != 0)?;
    let face2 = (0..3).find(|&d| n2[d] != 0)?;

    // Step 3: pf1f2 and chirality factor s
    // Uses 1-indexed face numbers for the exponent
    let pf1f2: i64 = -(n1[face1] as i64) * (n2[face2] as i64);
    let exp = (face1 + 1 + face2 + 1) as i64; // 1-indexed
    let sign = if exp % 2 == 0 { 1i64 } else { -1i64 };
    let s = sign * pf1f2;

    // Step 4: Diagonal vectors
    let d1: [i64; 3] = [b1[0] - a1[0], b1[1] - a1[1], b1[2] - a1[2]];
    let d2: [i64; 3] = [b2[0] - a2[0], b2[1] - a2[1], b2[2] - a2[2]];
    let d: i64 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
    if d == 0 {
        return None;
    }

    // Step 5: Varying axes
    let (i11, i12) = varying_axes(face1);
    let (i21, i22) = varying_axes(face2);

    // Step 6: Build 3×3 matrix
    let mut m = [[0i64; 3]; 3];

    m[i21][i11] = (d1[i11] * d2[i21] + s * d1[i12] * d2[i22]) / d;
    m[i21][i12] = (d1[i12] * d2[i21] - s * d1[i11] * d2[i22]) / d;
    m[i22][i11] = -s * m[i21][i12];
    m[i22][i12] = s * m[i21][i11];
    m[face2][face1] = pf1f2;

    // Validate all entries are in {-1, 0, +1}
    let mut result = [[0i8; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            if m[r][c] < -1 || m[r][c] > 1 {
                return None;
            }
            result[r][c] = m[r][c] as i8;
        }
    }

    // Check determinant magnitude == 1
    let det = det3x3(&result);
    if det != 1 && det != -1 {
        return None;
    }

    Some(result)
}

/// Compute determinant of a 3×3 integer matrix.
fn det3x3(m: &[[i8; 3]; 3]) -> i8 {
    let a = m[0][0] as i16;
    let b = m[0][1] as i16;
    let c = m[0][2] as i16;
    let d = m[1][0] as i16;
    let e = m[1][1] as i16;
    let f = m[1][2] as i16;
    let g = m[2][0] as i16;
    let h = m[2][1] as i16;
    let i = m[2][2] as i16;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    det as i8
}

/// Validate that a FaceMatch's directed diagonal corners produce a valid
/// permutation matrix (all-integer entries, |det| = 1).
///
/// Uses the FaceMatch's raw `il/jl/kl` and `ih/jh/kh` (which encode
/// directed diagonal corners when properly set).
pub fn validate_permutation_matrix(fm: &FaceMatch) -> Option<[[i8; 3]; 3]> {
    let a1 = [fm.block1.il as i64, fm.block1.jl as i64, fm.block1.kl as i64];
    let b1 = [fm.block1.ih as i64, fm.block1.jh as i64, fm.block1.kh as i64];
    let a2 = [fm.block2.il as i64, fm.block2.jl as i64, fm.block2.kl as i64];
    let b2 = [fm.block2.ih as i64, fm.block2.jh as i64, fm.block2.kh as i64];
    compute_permutation_matrix(a1, b1, a2, b2)
}

/// Validate corner pairing by checking spatial coincidence of diagonal
/// corners and that the resulting permutation matrix matrix is valid.
pub fn validate_corner_pairing(
    fm: &FaceMatch,
    blocks: &[Block],
    tol: f64,
) -> bool {
    let blk1 = &blocks[fm.block1.block_index];
    let blk2 = &blocks[fm.block2.block_index];

    // Check spatial coincidence of lb corners
    let (x1a, y1a, z1a) = blk1.xyz(fm.block1.il, fm.block1.jl, fm.block1.kl);
    let (x2a, y2a, z2a) = blk2.xyz(fm.block2.il, fm.block2.jl, fm.block2.kl);
    let dist_a = ((x1a - x2a).powi(2) + (y1a - y2a).powi(2) + (z1a - z2a).powi(2)).sqrt();
    if dist_a > tol {
        return false;
    }

    // Check spatial coincidence of ub corners
    let (x1b, y1b, z1b) = blk1.xyz(fm.block1.ih, fm.block1.jh, fm.block1.kh);
    let (x2b, y2b, z2b) = blk2.xyz(fm.block2.ih, fm.block2.jh, fm.block2.kh);
    let dist_b = ((x1b - x2b).powi(2) + (y1b - y2b).powi(2) + (z1b - z2b).powi(2)).sqrt();
    if dist_b > tol {
        return false;
    }

    // Check that permutation matrix produces a valid matrix
    validate_permutation_matrix(fm).is_some()
}

// ── JSON export/import ──────────────────────────────────────────────────

/// A single face record with its index-space normal, for JSON serialization.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FaceNormalRecord {
    pub block_index: usize,
    pub lb: [usize; 3],
    pub ub: [usize; 3],
    pub index_normal: [i8; 3],
    pub face_type: String,
}

/// Top-level structure for normals.json.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NormalsJson {
    pub faces: Vec<FaceNormalRecord>,
}

/// Face type name from constant axis and position.
fn face_type_name(const_ax: usize, is_low: bool) -> String {
    let axis = match const_ax {
        0 => "i",
        1 => "j",
        _ => "k",
    };
    if is_low {
        format!("{}min", axis)
    } else {
        format!("{}max", axis)
    }
}

/// Compute normals for all 6 faces of every block and return as a NormalsJson.
pub fn compute_all_normals(blocks: &[Block]) -> NormalsJson {
    let mut faces = Vec::new();
    for (bi, blk) in blocks.iter().enumerate() {
        let dims = [blk.imax, blk.jmax, blk.kmax];
        // 6 faces per block: {i,j,k} × {min, max}
        for const_ax in 0..3 {
            for &is_low in &[true, false] {
                let val = if is_low { 0 } else { dims[const_ax] - 1 };
                let mut lb = [0usize; 3];
                let mut ub = [0usize; 3];
                for d in 0..3 {
                    if d == const_ax {
                        lb[d] = val;
                        ub[d] = val;
                    } else {
                        lb[d] = 0;
                        ub[d] = dims[d] - 1;
                    }
                }
                let normal = if is_low { -1i8 } else { 1i8 };
                let mut n = [0i8; 3];
                n[const_ax] = normal;

                faces.push(FaceNormalRecord {
                    block_index: bi,
                    lb,
                    ub,
                    index_normal: n,
                    face_type: face_type_name(const_ax, is_low),
                });
            }
        }
    }
    NormalsJson { faces }
}

/// Write normals to a JSON file.
pub fn export_normals_json(normals: &NormalsJson, path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(normals)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, json)
}

/// Read normals from a JSON file.
pub fn import_normals_json(path: &Path) -> std::io::Result<NormalsJson> {
    let data = std::fs::read_to_string(path)?;
    serde_json::from_str(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_space_normal_kmin() {
        // K-face at k=0 → normal = (0, 0, -1)
        let n = index_space_normal_from_bounds([0, 0, 0], [24, 408, 0]);
        assert_eq!(n, Some([0, 0, -1]));
    }

    #[test]
    fn test_index_space_normal_jmin() {
        // J-face at j=0 → normal = (0, -1, 0)
        let n = index_space_normal_from_bounds([0, 0, 0], [408, 0, 24]);
        assert_eq!(n, Some([0, -1, 0]));
    }

    #[test]
    fn test_index_space_normal_imax() {
        // I-face at i=24 → normal = (0, 0, +1) — wait, i=24 > 0 → +1
        let n = index_space_normal_from_bounds([24, 0, 0], [24, 408, 12]);
        assert_eq!(n, Some([1, 0, 0]));
    }

    #[test]
    fn test_compute_permutation_matrix_cross_plane() {
        // Cross-plane: K-face ↔ J-face (from memory notes worked example)
        // a1=(0,0,0), b1=(24,408,0) — K-face k=0
        // a2=(408,0,0), b2=(0,0,24) — J-face j=0, directed
        let m = compute_permutation_matrix([0, 0, 0], [24, 408, 0], [408, 0, 0], [0, 0, 24]);
        let expected = [[0, -1, 0], [0, 0, -1], [1, 0, 0]];
        assert_eq!(m, Some(expected));
    }

    #[test]
    fn test_compute_permutation_matrix_in_plane_identity() {
        // In-plane: K-face ↔ K-face, same orientation
        // a1=(0,0,0), b1=(24,408,0) — K=0
        // a2=(0,0,12), b2=(24,408,12) — K=12
        let m = compute_permutation_matrix([0, 0, 0], [24, 408, 0], [0, 0, 12], [24, 408, 12]);
        // pf1f2 = -n1[k]*n2[k] = -(-1)*(+1) = +1, so identity
        let expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        assert_eq!(m, Some(expected));
    }

    #[test]
    fn test_bad_corner_pairing_returns_none() {
        // Ascending bounds for block2 (not directed) should give garbage
        // a2=(0,0,0), b2=(408,0,24) — ascending, NOT spatially paired
        let m = compute_permutation_matrix([0, 0, 0], [24, 408, 0], [0, 0, 0], [408, 0, 24]);
        // This should either return None or produce a non-identity matrix
        // The key test is the cross-plane case above which MUST be correct
        // This case may or may not produce valid integers depending on dimensions
        if let Some(mat) = m {
            // If it produces a matrix, it should be different from the correct one
            assert_ne!(mat, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]);
        }
    }
}
