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

/// The 8 canonical 2x2 permutation matrices for face orientation.
///
/// Each matrix operates on parametric (u, v) coordinates. The index encodes:
/// - bit 0: `u_reversed`
/// - bit 1: `v_reversed`
/// - bit 2: `swapped` (transpose u and v)
///
/// The index is computed as:
/// ```text
/// index = u_reversed | (v_reversed << 1) | (swapped << 2)
/// ```
///
/// | Index | Matrix              | Effect            |
/// |:-----:|:-------------------:|:-----------------:|
/// |   0   | `[[ 1, 0],[ 0, 1]]`| identity          |
/// |   1   | `[[-1, 0],[ 0, 1]]`| flip u            |
/// |   2   | `[[ 1, 0],[ 0,-1]]`| flip v            |
/// |   3   | `[[-1, 0],[ 0,-1]]`| flip both         |
/// |   4   | `[[ 0, 1],[ 1, 0]]`| transpose         |
/// |   5   | `[[ 0,-1],[ 1, 0]]`| transpose + flip u|
/// |   6   | `[[ 0, 1],[-1, 0]]`| transpose + flip v|
/// |   7   | `[[ 0,-1],[-1, 0]]`| transpose + both  |
///
/// # Examples
///
/// ```
/// use plot3d::PERMUTATION_MATRICES;
///
/// // Identity (index 0): no reversal, no swap
/// assert_eq!(PERMUTATION_MATRICES[0], [[1, 0], [0, 1]]);
///
/// // Index 5 = u_reversed (bit 0) + swapped (bit 2) = 1 + 4
/// assert_eq!(PERMUTATION_MATRICES[5], [[0, -1], [1, 0]]);
///
/// // Verify the full table has exactly 8 entries
/// assert_eq!(PERMUTATION_MATRICES.len(), 8);
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
///
/// When two block faces share an interface, their constant axes may or may
/// not be the same. This distinction matters because cross-plane matches
/// require a parametric axis swap (bit 2 of `permutation_index`), while
/// in-plane matches only need reversal flags.
///
/// - [`InPlane`](OrientationPlane::InPlane): both faces have the same
///   constant axis (e.g., both K-constant). Only the 4 non-swap
///   permutations (indices 0-3) apply.
/// - [`CrossPlane`](OrientationPlane::CrossPlane): faces have different
///   constant axes (e.g., K-constant abutting J-constant). The full set of
///   8 permutations (indices 0-7) must be tested.
///
/// # Examples
///
/// ```
/// use plot3d::OrientationPlane;
///
/// let plane = OrientationPlane::InPlane;
/// assert_eq!(plane, OrientationPlane::InPlane);
///
/// let cross = OrientationPlane::CrossPlane;
/// assert_ne!(plane, cross);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OrientationPlane {
    InPlane,
    CrossPlane,
}

/// Describes the parametric orientation between two matched faces using a
/// permutation matrix index (0-7).
///
/// The permutation matrix transforms face2's parametric (u, v) coordinates
/// to align with face1's. The `plane` field indicates whether the faces share
/// the same constant axis (in-plane) or have different constant axes
/// (cross-plane).
///
/// Construct via [`Orientation::from_flags`] when you have individual boolean
/// flags, or set `permutation_index` directly when you already know the
/// encoded value.
///
/// # Bit layout
///
/// ```text
/// permutation_index = u_reversed | (v_reversed << 1) | (swapped << 2)
/// ```
///
/// # Examples
///
/// ```
/// use plot3d::{Orientation, OrientationPlane, PERMUTATION_MATRICES};
///
/// // Build from boolean flags: u reversed, v not reversed, axes swapped
/// let orient = Orientation::from_flags(true, false, true, OrientationPlane::CrossPlane);
/// assert_eq!(orient.permutation_index, 5); // 1 + 0 + 4
/// assert!(orient.u_reversed());
/// assert!(!orient.v_reversed());
/// assert!(orient.swapped());
///
/// // Retrieve the 2x2 matrix
/// let m = orient.matrix();
/// assert_eq!(*m, PERMUTATION_MATRICES[5]);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Orientation {
    /// Index (0-7) into [`PERMUTATION_MATRICES`].
    pub permutation_index: u8,
    /// Whether this is an in-plane or cross-plane match.
    pub plane: OrientationPlane,
    /// Optional declarative copy of the 2×2 permutation matrix as written
    /// in upstream connectivity files (e.g. `connectivity.json` from
    /// `plot3d_utilities`). When present, this is the authoritative
    /// orientation source: callers must convert it via
    /// [`Orientation::index_from_permutation_matrix`] to get the
    /// [`PERMUTATION_MATRICES`]-canonical index, and verifiers prefer it
    /// over `permutation_index`. None for legacy / synthesized matches
    /// where only the index was set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permutation_matrix: Option<[[i8; 2]; 2]>,
}

impl Orientation {
    /// Construct from the legacy boolean flags.
    pub fn from_flags(
        u_reversed: bool,
        v_reversed: bool,
        swapped: bool,
        plane: OrientationPlane,
    ) -> Self {
        let index = (u_reversed as u8) | ((v_reversed as u8) << 1) | ((swapped as u8) << 2);
        Self {
            permutation_index: index,
            plane,
            permutation_matrix: None,
        }
    }

    /// Whether block2's u-axis is reversed relative to block1's.
    pub fn u_reversed(&self) -> bool {
        self.permutation_index & 1 != 0
    }

    /// Whether block2's v-axis is reversed relative to block1's.
    pub fn v_reversed(&self) -> bool {
        self.permutation_index & 2 != 0
    }

    /// Whether block2's u and v axes are transposed relative to block1's.
    pub fn swapped(&self) -> bool {
        self.permutation_index & 4 != 0
    }

    /// Get the 2×2 permutation matrix for this orientation.
    pub fn matrix(&self) -> &[[i8; 2]; 2] {
        &PERMUTATION_MATRICES[self.permutation_index as usize]
    }

    /// Look up the canonical permutation index whose matrix in
    /// [`PERMUTATION_MATRICES`] equals `m`.  Returns `None` if `m` is not
    /// one of the 8 canonical permutations (e.g. a non-axis-aligned
    /// matrix or a typo in upstream-provided connectivity).
    ///
    /// This is the deterministic translation used when an upstream
    /// `connectivity.json` carries an explicit `permutation_matrix` —
    /// the verifiers consult this instead of brute-forcing through all
    /// 8 candidates.
    pub fn index_from_permutation_matrix(m: [[i8; 2]; 2]) -> Option<u8> {
        PERMUTATION_MATRICES
            .iter()
            .position(|c| *c == m)
            .map(|i| i as u8)
    }
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
