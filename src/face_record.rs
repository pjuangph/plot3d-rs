//! Core data types for face connectivity: [`FaceRecord`], [`FaceMatch`],
//! [`MatchPoint`], and [`Orientation`].
//!
//! # Diagonal Convention
//!
//! [`FaceRecord`] stores diagonal corners as `il/jl/kl` (first corner) and
//! `ih/jh/kh` (second corner). The ordering is **not** normalized: `il` can
//! be greater than `ih`, encoding that the I-axis is reversed on this face
//! relative to the matching face on the other block.
//!
//! This matches the GridPro/GlennHT connectivity convention and makes it
//! possible to reconstruct the orientation relationship between two matched
//! faces from the `FaceRecord` alone, without re-sampling block coordinates.
//!
//! When you need min/max values (e.g. for range iteration), use the
//! normalized accessors: [`FaceRecord::i_lo()`], [`FaceRecord::i_hi()`], etc.

use serde::{Deserialize, Serialize};

use crate::{block::Block, block_face_functions::Face};

/// Compact identifier for a face: `(block_index, il, jl, kl, ih, jh, kh)`.
pub type FaceKey = (usize, usize, usize, usize, usize, usize, usize);

/// Pointwise correspondence between two block faces.
#[derive(Clone, Debug, Serialize)]
pub struct MatchPoint {
    pub i1: usize,
    pub j1: usize,
    pub k1: usize,
    pub i2: usize,
    pub j2: usize,
    pub k2: usize,
}

/// Extract `(i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)` bounds from a slice of [`MatchPoint`]s.
///
/// When `use_block1` is true the block-1 indices (`i1/j1/k1`) are used;
/// otherwise the block-2 indices (`i2/j2/k2`).
pub fn match_point_bounds(
    points: &[MatchPoint],
    use_block1: bool,
) -> (usize, usize, usize, usize, usize, usize) {
    if use_block1 {
        (
            points.iter().map(|p| p.i1).min().unwrap(),
            points.iter().map(|p| p.i1).max().unwrap(),
            points.iter().map(|p| p.j1).min().unwrap(),
            points.iter().map(|p| p.j1).max().unwrap(),
            points.iter().map(|p| p.k1).min().unwrap(),
            points.iter().map(|p| p.k1).max().unwrap(),
        )
    } else {
        (
            points.iter().map(|p| p.i2).min().unwrap(),
            points.iter().map(|p| p.i2).max().unwrap(),
            points.iter().map(|p| p.j2).min().unwrap(),
            points.iter().map(|p| p.j2).max().unwrap(),
            points.iter().map(|p| p.k2).min().unwrap(),
            points.iter().map(|p| p.k2).max().unwrap(),
        )
    }
}

/// Compact record describing a face on a particular block.
///
/// # Diagonal Convention
///
/// The fields `il/jl/kl` and `ih/jh/kh` define the two diagonal corners
/// of this face on the block. These are **NOT** guaranteed to satisfy
/// `il <= ih`. The ordering encodes **orientation**: when `il > ih`, the
/// I-axis is reversed on this face relative to the matching face.
///
/// Use `i_lo()`/`i_hi()` when you need normalized min/max values
/// (e.g., for range iteration or face reconstruction).
#[derive(Clone, Debug, Serialize)]
pub struct FaceRecord {
    pub block_index: usize,
    /// I-index of the first diagonal corner.
    pub il: usize,
    /// J-index of the first diagonal corner.
    pub jl: usize,
    /// K-index of the first diagonal corner.
    pub kl: usize,
    /// I-index of the second diagonal corner.
    pub ih: usize,
    /// J-index of the second diagonal corner.
    pub jh: usize,
    /// K-index of the second diagonal corner.
    pub kh: usize,
    pub id: Option<usize>,
    /// Which physical axis ('x','y','z') the u-parameter primarily aligns with,
    /// and whether the physical coordinate increases as the u-index increases.
    /// `None` when not yet computed.
    #[serde(default)]
    pub u_physical: Option<(char, bool)>,
    /// Same for the v-parameter (second varying index of the face).
    #[serde(default)]
    pub v_physical: Option<(char, bool)>,
}

impl FaceRecord {
    /// Build a corner description from matching points.
    ///
    /// * `block_index` – Owning block index.
    /// * `points` – Matched nodes.
    /// * `first` – If `true` we use the indices from block1; otherwise block2.
    ///
    /// Returns `None` when `points` is empty.
    pub(crate) fn from_match_points(
        block_index: usize,
        points: &[MatchPoint],
        first: bool,
    ) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let il = points
            .iter()
            .map(|p| if first { p.i1 } else { p.i2 })
            .min()?;
        let jl = points
            .iter()
            .map(|p| if first { p.j1 } else { p.j2 })
            .min()?;
        let kl = points
            .iter()
            .map(|p| if first { p.k1 } else { p.k2 })
            .min()?;
        let ih = points
            .iter()
            .map(|p| if first { p.i1 } else { p.i2 })
            .max()?;
        let jh = points
            .iter()
            .map(|p| if first { p.j1 } else { p.j2 })
            .max()?;
        let kh = points
            .iter()
            .map(|p| if first { p.k1 } else { p.k2 })
            .max()?;
        Some(Self {
            block_index,
            il,
            jl,
            kl,
            ih,
            jh,
            kh,
            id: None,
            u_physical: None,
            v_physical: None,
        })
    }

    /// Construct a record from a Face instance.
    pub fn from_face(face: &Face) -> Self {
        Self {
            block_index: face.block_index().unwrap_or(usize::MAX),
            il: face.imin(),
            jl: face.jmin(),
            kl: face.kmin(),
            ih: face.imax(),
            jh: face.jmax(),
            kh: face.kmax(),
            id: face.id(),
            u_physical: None,
            v_physical: None,
        }
    }

    // -- Normalized accessors (for range iteration / face reconstruction) --

    /// Smallest I-index. Always `min(il, ih)`.
    #[inline]
    pub fn i_lo(&self) -> usize {
        self.il.min(self.ih)
    }
    /// Largest I-index. Always `max(il, ih)`.
    #[inline]
    pub fn i_hi(&self) -> usize {
        self.il.max(self.ih)
    }
    /// Smallest J-index.
    #[inline]
    pub fn j_lo(&self) -> usize {
        self.jl.min(self.jh)
    }
    /// Largest J-index.
    #[inline]
    pub fn j_hi(&self) -> usize {
        self.jl.max(self.jh)
    }
    /// Smallest K-index.
    #[inline]
    pub fn k_lo(&self) -> usize {
        self.kl.min(self.kh)
    }
    /// Largest K-index.
    #[inline]
    pub fn k_hi(&self) -> usize {
        self.kl.max(self.kh)
    }

    /// True when the I-axis is reversed (`il > ih`).
    #[inline]
    pub fn i_reversed(&self) -> bool {
        self.il > self.ih
    }
    /// True when the J-axis is reversed (`jl > jh`).
    #[inline]
    pub fn j_reversed(&self) -> bool {
        self.jl > self.jh
    }
    /// True when the K-axis is reversed (`kl > kh`).
    #[inline]
    pub fn k_reversed(&self) -> bool {
        self.kl > self.kh
    }

    /// Ascending bounds: `([lo_i, lo_j, lo_k], [hi_i, hi_j, hi_k])`.
    #[inline]
    pub fn bounds(&self) -> ([usize; 3], [usize; 3]) {
        (
            [self.i_lo(), self.j_lo(), self.k_lo()],
            [self.i_hi(), self.j_hi(), self.k_hi()],
        )
    }

    /// Index (0, 1, or 2) of the constant axis, or `None` if no axis is constant.
    #[inline]
    pub fn constant_axis(&self) -> Option<usize> {
        let (lo, hi) = self.bounds();
        (0..3).find(|&d| lo[d] == hi[d])
    }

    /// Returns the sorted (ascending) pair of face dimension spans.
    /// For a face with one constant axis, two spans are non-zero.
    /// Uses absolute differences so reversal of il/ih, jl/jh, kl/kh is handled.
    /// The constant axis does not need to match between paired faces
    /// (e.g. a constant-i face can match a constant-k face).
    pub fn face_dims(&self) -> (usize, usize) {
        let mut spans = [
            self.il.abs_diff(self.ih),
            self.jl.abs_diff(self.jh),
            self.kl.abs_diff(self.kh),
        ];
        spans.sort();
        (spans[1], spans[2])
    }

    /// Compute and fill in the physical direction metadata by sampling the block.
    ///
    /// For a face with one constant axis (e.g. K-constant), the two varying axes
    /// form u and v. We sample the block at the min and max corners of each
    /// varying axis to determine which physical axis (x, y, z) it primarily
    /// aligns with and whether it is increasing.
    pub fn compute_direction(&mut self, block: &Block) {
        // Determine which axis is constant (use normalized min/max)
        let i_const = self.i_lo() == self.i_hi();
        let j_const = self.j_lo() == self.j_hi();
        let k_const = self.k_lo() == self.k_hi();

        let (ilo, jlo, klo) = (self.i_lo(), self.j_lo(), self.k_lo());
        let (ihi, jhi, khi) = (self.i_hi(), self.j_hi(), self.k_hi());

        // Identify u and v varying axes
        // Convention: for K-const → u=I, v=J; for J-const → u=I, v=K; for I-const → u=J, v=K
        let (u_min_ijk, u_max_ijk, v_min_ijk, v_max_ijk) = if k_const || !i_const && !j_const {
            // K-constant (or all varying, default to K-const convention)
            (
                (ilo, jlo, klo),
                (ihi, jlo, klo),
                (ilo, jlo, klo),
                (ilo, jhi, klo),
            )
        } else if j_const {
            (
                (ilo, jlo, klo),
                (ihi, jlo, klo),
                (ilo, jlo, klo),
                (ilo, jlo, khi),
            )
        } else {
            // I-constant
            (
                (ilo, jlo, klo),
                (ilo, jhi, klo),
                (ilo, jlo, klo),
                (ilo, jlo, khi),
            )
        };

        // Sample block coordinates
        let (ux0, uy0, uz0) = block.xyz(u_min_ijk.0, u_min_ijk.1, u_min_ijk.2);
        let (ux1, uy1, uz1) = block.xyz(u_max_ijk.0, u_max_ijk.1, u_max_ijk.2);
        let (vx0, vy0, vz0) = block.xyz(v_min_ijk.0, v_min_ijk.1, v_min_ijk.2);
        let (vx1, vy1, vz1) = block.xyz(v_max_ijk.0, v_max_ijk.1, v_max_ijk.2);

        // Determine dominant physical axis for u
        let du = [(ux1 - ux0), (uy1 - uy0), (uz1 - uz0)];
        let abs_du = [du[0].abs(), du[1].abs(), du[2].abs()];
        let u_axis_idx = if abs_du[0] >= abs_du[1] && abs_du[0] >= abs_du[2] {
            0
        } else if abs_du[1] >= abs_du[2] {
            1
        } else {
            2
        };
        let u_axis = ['x', 'y', 'z'][u_axis_idx];
        let u_increasing = du[u_axis_idx] >= 0.0;

        // Determine dominant physical axis for v
        let dv = [(vx1 - vx0), (vy1 - vy0), (vz1 - vz0)];
        let abs_dv = [dv[0].abs(), dv[1].abs(), dv[2].abs()];
        let v_axis_idx = if abs_dv[0] >= abs_dv[1] && abs_dv[0] >= abs_dv[2] {
            0
        } else if abs_dv[1] >= abs_dv[2] {
            1
        } else {
            2
        };
        let v_axis = ['x', 'y', 'z'][v_axis_idx];
        let v_increasing = dv[v_axis_idx] >= 0.0;

        self.u_physical = Some((u_axis, u_increasing));
        self.v_physical = Some((v_axis, v_increasing));
    }

    /// Scale the index ranges by `factor`.
    pub fn scale_indices(&mut self, factor: usize) {
        if factor <= 1 {
            return;
        }
        self.il *= factor;
        self.jl *= factor;
        self.kl *= factor;
        self.ih *= factor;
        self.jh *= factor;
        self.kh *= factor;
    }

    /// Reduce the index ranges by `divisor`.
    pub fn divide_indices(&mut self, divisor: usize) {
        if divisor <= 1 {
            return;
        }
        self.il /= divisor;
        self.jl /= divisor;
        self.kl /= divisor;
        self.ih /= divisor;
        self.jh /= divisor;
        self.kh /= divisor;
    }

    /// Build a compact key tuple for set/map lookups.
    #[inline]
    pub fn index_key(&self) -> FaceKey {
        (
            self.block_index,
            self.il,
            self.jl,
            self.kl,
            self.ih,
            self.jh,
            self.kh,
        )
    }

    /// Reconstruct a Face from this record using the provided blocks.
    ///
    /// Uses normalized `i_lo()/i_hi()` values to ensure valid face geometry.
    pub fn to_face(&self, blocks: &[Block]) -> Option<Face> {
        let block = blocks.get(self.block_index)?;
        let mut face = crate::block_face_functions::create_face_from_diagonals(
            block,
            self.i_lo(),
            self.j_lo(),
            self.k_lo(),
            self.i_hi(),
            self.j_hi(),
            self.k_hi(),
        );
        face.set_block_index(self.block_index);
        if let Some(id) = self.id {
            face.set_id(id);
        }
        Some(face)
    }
}

/// Helper trait to print summaries of face records.
pub trait FaceRecordTraits {
    fn print(&self);
}

impl FaceRecordTraits for [FaceRecord] {
    fn print(&self) {
        for face in self {
            println!(
                "face block{} id {:?}: [{},{},{} → {},{},{}]",
                face.block_index, face.id, face.il, face.jl, face.kl, face.ih, face.jh, face.kh
            );
        }
    }
}

impl FaceRecordTraits for Vec<FaceRecord> {
    fn print(&self) {
        self.as_slice().print();
    }
}

/// The 8 canonical 2×2 permutation matrices for face orientation (legacy).
///
/// Retained for backward compatibility and serialization. New code should
/// use [`Orientation::matrix3x3`] which works in full (i, j, k) space.
///
/// Each matrix operates on parametric (u, v) coordinates. The index encodes:
/// ```text
/// index = u_reversed | (v_reversed << 1) | (swapped << 2)
/// ```
pub const PERMUTATION_MATRICES: [[[i8; 2]; 2]; 8] = [
    [[1, 0], [0, 1]],   // 0: identity
    [[-1, 0], [0, 1]],  // 1: u reversed
    [[1, 0], [0, -1]],  // 2: v reversed
    [[-1, 0], [0, -1]], // 3: both reversed
    [[0, 1], [1, 0]],   // 4: swapped
    [[0, -1], [1, 0]],  // 5: swap + u reversed
    [[0, 1], [-1, 0]],  // 6: swap + v reversed
    [[0, -1], [-1, 0]], // 7: swap + both reversed
];

/// Whether a face match is in-plane or cross-plane.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OrientationPlane {
    InPlane,
    CrossPlane,
}

/// Describes the orientation relationship between two matched block faces
/// using a 3×3 signed permutation matrix.
///
/// The matrix `M` (entries in {-1, 0, +1}, |det| = 1) maps block1's
/// (i, j, k) indices to block2's:
///
/// ```text
/// [i2]         [i1 - lb1_i]
/// [j2] = lb2 + M * [j1 - lb1_j]
/// [k2]         [k1 - lb1_k]
/// ```
///
/// This replaces the earlier 2×2 parametric (u, v) approach. The 3×3
/// matrix unifies in-plane and cross-plane matches and eliminates the
/// need for separate constant-axis detection, u/v extraction, and
/// i/j/k reconstruction.
///
/// # Backward compatibility
///
/// The legacy `permutation_index` (0-7) is available via
/// [`Orientation::permutation_index`] for serialization.
///
/// # Examples
///
/// ```
/// use plot3d::Orientation;
///
/// // Identity orientation for a K-constant face match
/// let m = [[1,0,0],[0,1,0],[0,0,1i8]];
/// let orient = Orientation::from_matrix(m);
/// assert_eq!(orient.permutation_index(), 0);
///
/// // Build from legacy flags (backward compat)
/// let orient2 = Orientation::from_perm_index(3, Some(2), Some(2));
/// assert_eq!(orient2.matrix3x3()[0][0], -1); // u reversed
/// assert_eq!(orient2.matrix3x3()[1][1], -1); // v reversed
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Orientation {
    /// 3×3 signed permutation matrix mapping block1 (i,j,k) → block2 (i,j,k).
    pub matrix: [[i8; 3]; 3],
}

impl Orientation {
    /// Construct from a 3×3 matrix directly.
    pub fn from_matrix(matrix: [[i8; 3]; 3]) -> Self {
        Self { matrix }
    }

    /// Build a 3×3 orientation matrix from face constant axes and a 2×2
    /// permutation index.
    ///
    /// `const_axis1` and `const_axis2` are the constant axis indices (0=I, 1=J, 2=K)
    /// for face1 and face2 respectively. If `None`, defaults to axis 2 (K).
    pub fn from_perm_index(
        perm_idx: u8,
        const_axis1: Option<usize>,
        const_axis2: Option<usize>,
    ) -> Self {
        let c1 = const_axis1.unwrap_or(2);
        let c2 = const_axis2.unwrap_or(2);
        let mat2 = PERMUTATION_MATRICES[perm_idx as usize];
        Self {
            matrix: expand_2x2_to_3x3(mat2, c1, c2),
        }
    }

    /// Construct from the legacy boolean flags plus constant-axis info.
    pub fn from_flags(
        u_reversed: bool,
        v_reversed: bool,
        swapped: bool,
        const_axis1: usize,
        const_axis2: usize,
    ) -> Self {
        let index = (u_reversed as u8) | ((v_reversed as u8) << 1) | ((swapped as u8) << 2);
        Self::from_perm_index(index, Some(const_axis1), Some(const_axis2))
    }

    /// Get the 3×3 matrix.
    #[inline]
    pub fn matrix3x3(&self) -> &[[i8; 3]; 3] {
        &self.matrix
    }

    /// Whether this is an in-plane or cross-plane match (derived from matrix).
    pub fn plane(&self) -> OrientationPlane {
        // In-plane: the constant axis maps to itself (diagonal entry is nonzero)
        // Cross-plane: constant axis maps to a different axis
        // We detect by checking if the matrix has any off-diagonal nonzero in
        // the constant-axis row/col. Simplest: check if it's block-diagonal.
        let c1 = self.const_axis_from();
        let c2 = self.const_axis_to();
        if c1 == c2 {
            OrientationPlane::InPlane
        } else {
            OrientationPlane::CrossPlane
        }
    }

    /// Compute the legacy 2×2 permutation index (0-7).
    pub fn permutation_index(&self) -> u8 {
        let c1 = self.const_axis_from();
        let vary1 = varying_axes(c1);
        let c2 = self.const_axis_to();
        let vary2 = varying_axes(c2);

        // The 2x2 sub-matrix maps (u1,v1) → (u2,v2)
        // u1 = vary1.0, v1 = vary1.1, u2 = vary2.0, v2 = vary2.1
        let m00 = self.matrix[vary2.0][vary1.0];
        let m01 = self.matrix[vary2.0][vary1.1];
        let m10 = self.matrix[vary2.1][vary1.0];
        let m11 = self.matrix[vary2.1][vary1.1];

        // swapped: m00==0 && m11==0 (diagonal is zero, off-diagonal is nonzero)
        let swapped = m00 == 0 && m11 == 0;

        let (u_reversed, v_reversed) = if swapped {
            // m01 maps u1→v2, m10 maps v1→u2
            (m10 < 0, m01 < 0)
        } else {
            (m00 < 0, m11 < 0)
        };

        (u_reversed as u8) | ((v_reversed as u8) << 1) | ((swapped as u8) << 2)
    }

    /// Which axis on face1 (block1) is constant — the row in M that has
    /// a nonzero entry only in the const_axis_to column.
    fn const_axis_from(&self) -> usize {
        // The constant axis from block1 is the column where the
        // constant-axis row of M has a nonzero entry.
        // Equivalently, find which column has exactly one nonzero entry
        // and that entry's row also has exactly one nonzero entry.
        for col in 0..3 {
            let nonzero_rows: Vec<usize> = (0..3)
                .filter(|&row| self.matrix[row][col] != 0)
                .collect();
            if nonzero_rows.len() == 1 {
                let row = nonzero_rows[0];
                let nonzero_cols: Vec<usize> = (0..3)
                    .filter(|&c| self.matrix[row][c] != 0)
                    .collect();
                if nonzero_cols.len() == 1 {
                    // This col maps to exactly one row, check if the OTHER
                    // two cols also map cleanly (valid permutation matrix)
                    return col;
                }
            }
        }
        2 // fallback
    }

    /// Which axis on face2 (block2) is constant.
    fn const_axis_to(&self) -> usize {
        for row in 0..3 {
            let nonzero_cols: Vec<usize> = (0..3)
                .filter(|&col| self.matrix[row][col] != 0)
                .collect();
            if nonzero_cols.len() == 1 {
                let col = nonzero_cols[0];
                let nonzero_rows: Vec<usize> = (0..3)
                    .filter(|&r| self.matrix[r][col] != 0)
                    .collect();
                if nonzero_rows.len() == 1 {
                    return row;
                }
            }
        }
        2 // fallback
    }

    /// Whether block2's u-axis is reversed relative to block1's (legacy compat).
    pub fn u_reversed(&self) -> bool {
        self.permutation_index() & 1 != 0
    }

    /// Whether block2's v-axis is reversed relative to block1's (legacy compat).
    pub fn v_reversed(&self) -> bool {
        self.permutation_index() & 2 != 0
    }

    /// Whether block2's u and v axes are transposed relative to block1's (legacy compat).
    pub fn swapped(&self) -> bool {
        self.permutation_index() & 4 != 0
    }

    /// Get the legacy 2×2 permutation matrix (for serialization compatibility).
    pub fn matrix2x2(&self) -> &[[i8; 2]; 2] {
        &PERMUTATION_MATRICES[self.permutation_index() as usize]
    }
}

/// Return the two varying axes for a given constant axis.
#[inline]
fn varying_axes(const_ax: usize) -> (usize, usize) {
    match const_ax {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Expand a 2×2 permutation matrix into a 3×3 by embedding it in the
/// varying-axis subspace. The constant-axis diagonal entry is set to ±1
/// based on whether the faces are on the same or different constant axes.
fn expand_2x2_to_3x3(mat2: [[i8; 2]; 2], const1: usize, const2: usize) -> [[i8; 3]; 3] {
    let mut m = [[0i8; 3]; 3];
    let (u1, v1) = varying_axes(const1);
    let (u2, v2) = varying_axes(const2);

    // Map varying axes: row=face2 axis, col=face1 axis
    m[u2][u1] = mat2[0][0];
    m[u2][v1] = mat2[0][1];
    m[v2][u1] = mat2[1][0];
    m[v2][v1] = mat2[1][1];

    // Map constant axis: face1's const → face2's const
    // Sign: normals should be anti-parallel at an interface
    // For simplicity and backward compat, use +1 (same side) or infer from context.
    // In practice the constant-axis entry is +1 for same-axis, or the
    // determinant-preserving value for cross-axis.
    // Map constant axis: face1's const → face2's const
    m[const2][const1] = 1;

    m
}

/// Aggregates the matching data between two faces.
///
/// Each entry stores the corner ranges (on both blocks) and every coincident
/// node that was found for that interface.
#[derive(Clone, Debug, Serialize)]
pub struct FaceMatch {
    pub block1: FaceRecord,
    pub block2: FaceRecord,
    pub points: Vec<MatchPoint>,
    /// Orientation relationship between block1 and block2 faces.
    /// `None` for legacy code paths or partial matches where orientation
    /// was not detected.
    #[serde(default)]
    pub orientation: Option<Orientation>,
}

impl FaceMatch {
    /// Downscale both participating face records by `divisor`.
    /// Note: MatchPoints are NOT scaled — they may be from full-resolution
    /// Phase 2/3 matching and should only be used with full-resolution blocks.
    pub fn divide_indices(&mut self, divisor: usize) {
        self.block1.divide_indices(divisor);
        self.block2.divide_indices(divisor);
    }

    /// Upscale both participating face records by `factor`.
    pub fn scale_indices(&mut self, factor: usize) {
        self.block1.scale_indices(factor);
        self.block2.scale_indices(factor);
    }
}

/// Helper trait to print summaries of face matches.
pub trait FaceMatchPrinter {
    fn print(&self);
}

impl FaceMatchPrinter for [FaceMatch] {
    fn print(&self) {
        for (idx, m) in self.iter().enumerate() {
            let block1 = &m.block1;
            let block2 = &m.block2;
            let node_count = m.points.len();
            let node_label = if node_count == 1 { "node" } else { "nodes" };
            println!(
                "match #{idx}: block{block1_idx:02} [{il1:03},{jl1:03},{kl1:03} -> {ih1:03},{jh1:03},{kh1:03}] <-> block{block2_idx:02} [{il2:03},{jl2:03},{kl2:03} -> {ih2:03},{jh2:03},{kh2:03}] ({node_count} {node_label})",
                block1_idx = block1.block_index,
                il1 = block1.il,
                jl1 = block1.jl,
                kl1 = block1.kl,
                ih1 = block1.ih,
                jh1 = block1.jh,
                kh1 = block1.kh,
                block2_idx = block2.block_index,
                il2 = block2.il,
                jl2 = block2.jl,
                kl2 = block2.kl,
                ih2 = block2.ih,
                jh2 = block2.jh,
                kh2 = block2.kh,
                node_count = node_count,
                node_label = node_label,
            );
        }
    }
}

impl FaceMatchPrinter for Vec<FaceMatch> {
    fn print(&self) {
        self.as_slice().print();
    }
}

/// Semantic alias for a periodic face pair (same structure as [`FaceMatch`]).
pub type PeriodicPair = FaceMatch;
