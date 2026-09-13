//! Complete node-for-node validation of a structured face correspondence.
//!
//! A conformal structured interface is not "four corners agree". It is:
//!
//! 1. two logically rectangular patch ranges, each with exactly one constant
//!    index;
//! 2. compatible dimensions under index reversal and/or axis exchange;
//! 3. one structured index mapping — one of the eight
//!    [`PERMUTATION_MATRICES`](crate::face_record::PERMUTATION_MATRICES);
//! 4. **every** node, interior nodes included, within tolerance under that
//!    mapping, on the grid actually being matched;
//! 5. bijective coverage, which the structured mapping gives by construction
//!    once the dimensions agree.
//!
//! Corners can *propose* a mapping (they only order the search here); they
//! cannot certify one. Exactly one mapping must pass: none means the patches
//! do not correspond, more than one means the geometry cannot tell the
//! mappings apart (a collapsed patch) and the result is reported as
//! ambiguous instead of guessed.

use crate::{
    block::Block,
    face_record::{FaceMatch, FaceRecord, MatchPoint, Orientation, OrientationPlane},
    Float,
};

/// A logically rectangular face patch on a block: inclusive ascending index
/// bounds with exactly one constant axis and at least two nodes along each
/// varying axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Patch {
    pub block: usize,
    pub lo: [usize; 3],
    pub hi: [usize; 3],
}

/// Why a pair of index ranges cannot be treated as two face patches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatchError {
    /// Not exactly one constant axis (a volume, an edge or a point).
    NotAFace { block: usize, lo: [usize; 3], hi: [usize; 3] },
    /// The range exceeds the block's dimensions.
    OutOfBlock { block: usize, hi: [usize; 3], dims: [usize; 3] },
    /// The block index exceeds the block list.
    BlockOutOfRange { block: usize, nblocks: usize },
}

impl std::fmt::Display for PatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchError::NotAFace { block, lo, hi } => {
                write!(f, "block {block} lo={lo:?} hi={hi:?} is not a face patch (exactly one constant axis required)")
            }
            PatchError::OutOfBlock { block, hi, dims } => {
                write!(f, "block {block} patch hi={hi:?} exceeds block dims {dims:?}")
            }
            PatchError::BlockOutOfRange { block, nblocks } => {
                write!(f, "block {block} out of range ({nblocks} blocks)")
            }
        }
    }
}

impl std::error::Error for PatchError {}

impl Patch {
    /// Build a patch, normalising the bounds to ascending order.
    pub fn new(block: usize, a: [usize; 3], b: [usize; 3]) -> Result<Patch, PatchError> {
        let lo = [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])];
        let hi = [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])];
        let constant = (0..3).filter(|&d| lo[d] == hi[d]).count();
        if constant != 1 {
            return Err(PatchError::NotAFace { block, lo, hi });
        }
        Ok(Patch { block, lo, hi })
    }

    pub fn from_record(rec: &FaceRecord) -> Result<Patch, PatchError> {
        let (lo, hi) = rec.bounds();
        Patch::new(rec.block_index, lo, hi)
    }

    /// Confirm the patch lies inside `blocks[self.block]`.
    pub fn check_in(&self, blocks: &[Block]) -> Result<(), PatchError> {
        let block = blocks.get(self.block).ok_or(PatchError::BlockOutOfRange {
            block: self.block,
            nblocks: blocks.len(),
        })?;
        let dims = [block.imax, block.jmax, block.kmax];
        if (0..3).any(|d| self.hi[d] >= dims[d]) {
            return Err(PatchError::OutOfBlock {
                block: self.block,
                hi: self.hi,
                dims,
            });
        }
        Ok(())
    }

    #[inline]
    pub fn const_axis(&self) -> usize {
        (0..3)
            .find(|&d| self.lo[d] == self.hi[d])
            .expect("Patch invariant: one constant axis")
    }

    /// The two varying axes in ascending order `(u_axis, v_axis)` — the same
    /// convention as [`crate::verification::extract_canonical_grid`].
    #[inline]
    pub fn uv_axes(&self) -> (usize, usize) {
        match self.const_axis() {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        }
    }

    /// Node counts `(nu, nv)` along the varying axes.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        let (ua, va) = self.uv_axes();
        (self.hi[ua] - self.lo[ua] + 1, self.hi[va] - self.lo[va] + 1)
    }

    /// Structured indices of parametric node `(u, v)`.
    #[inline]
    pub fn ijk(&self, u: usize, v: usize) -> [usize; 3] {
        let (ua, va) = self.uv_axes();
        let mut idx = self.lo;
        idx[ua] = self.lo[ua] + u;
        idx[va] = self.lo[va] + v;
        idx
    }

    /// Ascending [`FaceRecord`] for this patch.
    pub fn to_record(&self, id: Option<usize>) -> FaceRecord {
        FaceRecord {
            block_index: self.block,
            il: self.lo[0],
            jl: self.lo[1],
            kl: self.lo[2],
            ih: self.hi[0],
            jh: self.hi[1],
            kh: self.hi[2],
            id,
            u_physical: None,
            v_physical: None,
        }
    }
}

/// Dimensions `(nu, nv)` of patch B after applying permutation `perm` — what
/// patch A's dimensions must equal for the mapping to be admissible.
#[inline]
pub fn permuted_dims(perm: u8, nu_b: usize, nv_b: usize) -> (usize, usize) {
    if perm & 4 != 0 {
        (nv_b, nu_b)
    } else {
        (nu_b, nv_b)
    }
}

/// Map patch A's parametric `(u, v)` to patch B's under permutation `perm`.
///
/// This is the mapping [`crate::verification::apply_permutation`] realises
/// when it lays B's grid out in A's order, so a `permutation_index` certified
/// here is interchangeable with one produced by the verifiers.
#[inline]
pub fn map_uv(perm: u8, u: usize, v: usize, nu_b: usize, nv_b: usize) -> (usize, usize) {
    let (gu, gv) = if perm & 4 != 0 { (v, u) } else { (u, v) };
    let gu = if perm & 1 != 0 { nu_b - 1 - gu } else { gu };
    let gv = if perm & 2 != 0 { nv_b - 1 - gv } else { gv };
    (gu, gv)
}

/// One node pair's separation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeDiscrepancy {
    pub distance: Float,
    /// Structured indices on patch A (the transformed side).
    pub node_a: [usize; 3],
    /// Structured indices on patch B.
    pub node_b: [usize; 3],
}

/// A structured mapping under which every node pair is within tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedMapping {
    pub permutation_index: u8,
    pub plane: OrientationPlane,
    /// The node pair with the largest separation.
    pub worst: NodeDiscrepancy,
    pub nodes_checked: usize,
}

impl CertifiedMapping {
    pub fn orientation(&self) -> Orientation {
        Orientation {
            permutation_index: self.permutation_index,
            plane: self.plane.clone(),
            permutation_matrix: None,
        }
    }
}

/// Why no mapping could be certified.
#[derive(Clone, Debug, PartialEq)]
pub enum MappingFailure {
    /// No permutation makes the dimensions agree.
    IncompatibleDimensions {
        dims_a: (usize, usize),
        dims_b: (usize, usize),
    },
    /// Every admissible permutation has a node pair beyond tolerance. `worst`
    /// is the largest separation of the permutation that came closest.
    ExceedsTolerance {
        best_permutation: u8,
        worst: NodeDiscrepancy,
    },
    /// More than one permutation passes: the geometry cannot distinguish
    /// them (a collapsed patch, or a tolerance wider than the patch).
    Ambiguous { permutations: Vec<u8> },
}

impl std::fmt::Display for MappingFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MappingFailure::IncompatibleDimensions { dims_a, dims_b } => write!(
                f,
                "patch dimensions {dims_a:?} and {dims_b:?} admit no structured mapping"
            ),
            MappingFailure::ExceedsTolerance {
                best_permutation,
                worst,
            } => write!(
                f,
                "no structured mapping keeps every node within tolerance; permutation {best_permutation} comes closest, failing at A{:?} <-> B{:?} with separation {:e}",
                worst.node_a, worst.node_b, worst.distance
            ),
            MappingFailure::Ambiguous { permutations } => write!(
                f,
                "{} structured mappings {permutations:?} all fit within tolerance; the patch geometry cannot distinguish them",
                permutations.len()
            ),
        }
    }
}

struct PermScan {
    passed: bool,
    worst: NodeDiscrepancy,
}

#[allow(clippy::too_many_arguments)]
fn scan_permutation<F>(
    block_a: &Block,
    patch_a: &Patch,
    block_b: &Block,
    patch_b: &Patch,
    transform: &F,
    tol: Float,
    perm: u8,
    early_exit: bool,
) -> PermScan
where
    F: Fn([Float; 3]) -> [Float; 3],
{
    let (nu_a, nv_a) = patch_a.dims();
    let (nu_b, nv_b) = patch_b.dims();
    let tol2 = tol * tol;
    let mut passed = true;
    let mut worst = NodeDiscrepancy {
        distance: -1.0,
        node_a: patch_a.lo,
        node_b: patch_b.lo,
    };
    for u in 0..nu_a {
        for v in 0..nv_a {
            let ia = patch_a.ijk(u, v);
            let (ub, vb) = map_uv(perm, u, v, nu_b, nv_b);
            let ib = patch_b.ijk(ub, vb);
            let (xa, ya, za) = block_a.xyz(ia[0], ia[1], ia[2]);
            let (xb, yb, zb) = block_b.xyz(ib[0], ib[1], ib[2]);
            let ta = transform([xa, ya, za]);
            let d2 = (ta[0] - xb).powi(2) + (ta[1] - yb).powi(2) + (ta[2] - zb).powi(2);
            if !(d2 <= worst.distance) {
                worst = NodeDiscrepancy {
                    distance: d2,
                    node_a: ia,
                    node_b: ib,
                };
            }
            // NaN compares false, so a non-finite coordinate fails the node.
            if !(d2 <= tol2) {
                passed = false;
                if early_exit {
                    worst.distance = worst.distance.max(0.0).sqrt();
                    return PermScan { passed, worst };
                }
            }
        }
    }
    worst.distance = worst.distance.max(0.0).sqrt();
    PermScan { passed, worst }
}

/// Admissible permutations ordered by how well the four corners agree, so the
/// certifying scan usually succeeds on its first full pass. Corners only
/// propose; every candidate is still fully scanned.
fn corner_ordered_permutations<F>(
    block_a: &Block,
    patch_a: &Patch,
    block_b: &Block,
    patch_b: &Patch,
    transform: &F,
) -> Vec<u8>
where
    F: Fn([Float; 3]) -> [Float; 3],
{
    let (nu_a, nv_a) = patch_a.dims();
    let (nu_b, nv_b) = patch_b.dims();
    let corners_a = [(0, 0), (0, nv_a - 1), (nu_a - 1, 0), (nu_a - 1, nv_a - 1)];
    let mut scored: Vec<(Float, u8)> = (0u8..8)
        .filter(|&p| permuted_dims(p, nu_b, nv_b) == (nu_a, nv_a))
        .map(|p| {
            let score = corners_a
                .iter()
                .map(|&(u, v)| {
                    let ia = patch_a.ijk(u, v);
                    let (ub, vb) = map_uv(p, u, v, nu_b, nv_b);
                    let ib = patch_b.ijk(ub, vb);
                    let (xa, ya, za) = block_a.xyz(ia[0], ia[1], ia[2]);
                    let (xb, yb, zb) = block_b.xyz(ib[0], ib[1], ib[2]);
                    let ta = transform([xa, ya, za]);
                    (ta[0] - xb).powi(2) + (ta[1] - yb).powi(2) + (ta[2] - zb).powi(2)
                })
                .fold(0.0 as Float, Float::max);
            (score, p)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, p)| p).collect()
}

/// Certify that `transform(patch_a)` corresponds node-for-node to `patch_b`
/// under exactly one structured mapping, every node within `tol`.
///
/// Every admissible permutation is scanned; the corners only decide the
/// order. Returns the unique passing mapping, or why none (or more than one)
/// could be certified.
pub fn certify_correspondence<F>(
    block_a: &Block,
    patch_a: &Patch,
    block_b: &Block,
    patch_b: &Patch,
    transform: F,
    tol: Float,
) -> Result<CertifiedMapping, MappingFailure>
where
    F: Fn([Float; 3]) -> [Float; 3],
{
    let dims_a = patch_a.dims();
    let dims_b = patch_b.dims();
    let ordered = corner_ordered_permutations(block_a, patch_a, block_b, patch_b, &transform);
    if ordered.is_empty() {
        return Err(MappingFailure::IncompatibleDimensions { dims_a, dims_b });
    }
    let plane = if patch_a.const_axis() == patch_b.const_axis() {
        OrientationPlane::InPlane
    } else {
        OrientationPlane::CrossPlane
    };

    let mut passing: Vec<(u8, NodeDiscrepancy)> = Vec::new();
    for &perm in &ordered {
        let scan = scan_permutation(block_a, patch_a, block_b, patch_b, &transform, tol, perm, true);
        if scan.passed {
            passing.push((perm, scan.worst));
        }
    }

    match passing.len() {
        1 => {
            let (perm, worst) = passing.remove(0);
            Ok(CertifiedMapping {
                permutation_index: perm,
                plane,
                worst,
                nodes_checked: dims_a.0 * dims_a.1,
            })
        }
        0 => {
            let mut best: Option<(u8, NodeDiscrepancy)> = None;
            for &perm in &ordered {
                let scan =
                    scan_permutation(block_a, patch_a, block_b, patch_b, &transform, tol, perm, false);
                let better = match &best {
                    None => true,
                    Some((_, w)) => scan.worst.distance < w.distance,
                };
                if better {
                    best = Some((perm, scan.worst));
                }
            }
            let (best_permutation, worst) = best.expect("ordered is non-empty");
            Err(MappingFailure::ExceedsTolerance {
                best_permutation,
                worst,
            })
        }
        _ => Err(MappingFailure::Ambiguous {
            permutations: passing.into_iter().map(|(p, _)| p).collect(),
        }),
    }
}

/// Certify a single, already-proposed permutation instead of searching.
///
/// Use this when a discovery stage has produced an orientation that must be
/// confirmed as-is (a wrong proposal is a failure, never silently replaced
/// by a different orientation).
pub fn certify_permutation<F>(
    block_a: &Block,
    patch_a: &Patch,
    block_b: &Block,
    patch_b: &Patch,
    transform: F,
    tol: Float,
    perm: u8,
) -> Result<CertifiedMapping, MappingFailure>
where
    F: Fn([Float; 3]) -> [Float; 3],
{
    let dims_a = patch_a.dims();
    let dims_b = patch_b.dims();
    if permuted_dims(perm, dims_b.0, dims_b.1) != dims_a {
        return Err(MappingFailure::IncompatibleDimensions { dims_a, dims_b });
    }
    let scan = scan_permutation(block_a, patch_a, block_b, patch_b, &transform, tol, perm, false);
    if scan.passed {
        Ok(CertifiedMapping {
            permutation_index: perm,
            plane: if patch_a.const_axis() == patch_b.const_axis() {
                OrientationPlane::InPlane
            } else {
                OrientationPlane::CrossPlane
            },
            worst: scan.worst,
            nodes_checked: dims_a.0 * dims_a.1,
        })
    } else {
        Err(MappingFailure::ExceedsTolerance {
            best_permutation: perm,
            worst: scan.worst,
        })
    }
}

/// Every node pair of the certified mapping, in `u`-outer / `v`-inner order
/// over patch A. The first entry is A's `lo` corner, the last A's `hi`.
pub fn correspondence_points(patch_a: &Patch, patch_b: &Patch, perm: u8) -> Vec<MatchPoint> {
    let (nu_a, nv_a) = patch_a.dims();
    let (nu_b, nv_b) = patch_b.dims();
    let mut points = Vec::with_capacity(nu_a * nv_a);
    for u in 0..nu_a {
        for v in 0..nv_a {
            let ia = patch_a.ijk(u, v);
            let (ub, vb) = map_uv(perm, u, v, nu_b, nv_b);
            let ib = patch_b.ijk(ub, vb);
            points.push(MatchPoint {
                i1: ia[0],
                j1: ia[1],
                k1: ia[2],
                i2: ib[0],
                j2: ib[1],
                k2: ib[2],
            });
        }
    }
    points
}

/// Patch B's [`FaceRecord`] with directed diagonal corners: `il/jl/kl` is the
/// image of A's `lo` corner and `ih/jh/kh` the image of A's `hi` corner, so a
/// reversed axis shows as `il > ih` (the GridPro/GlennHT convention).
pub fn directed_record(patch_a: &Patch, patch_b: &Patch, perm: u8, id: Option<usize>) -> FaceRecord {
    let (nu_a, nv_a) = patch_a.dims();
    let (nu_b, nv_b) = patch_b.dims();
    let (u0, v0) = map_uv(perm, 0, 0, nu_b, nv_b);
    let (u1, v1) = map_uv(perm, nu_a - 1, nv_a - 1, nu_b, nv_b);
    let lb = patch_b.ijk(u0, v0);
    let ub = patch_b.ijk(u1, v1);
    FaceRecord {
        block_index: patch_b.block,
        il: lb[0],
        jl: lb[1],
        kl: lb[2],
        ih: ub[0],
        jh: ub[1],
        kh: ub[2],
        id,
        u_physical: None,
        v_physical: None,
    }
}

/// Assemble a [`FaceMatch`] for a certified correspondence: ascending record
/// for A, directed record for B, every node pair, and the certified
/// orientation.
pub fn build_face_match(patch_a: &Patch, patch_b: &Patch, mapping: &CertifiedMapping) -> FaceMatch {
    let perm = mapping.permutation_index;
    FaceMatch {
        block1: patch_a.to_record(None),
        block2: directed_record(patch_a, patch_b, perm, None),
        points: correspondence_points(patch_a, patch_b, perm),
        orientation: Some(mapping.orientation()),
    }
}

/// Re-certify a proposed [`FaceMatch`] (as produced by any discovery stage)
/// against `blocks` under `transform`, every node within `tol`.
///
/// If the proposal declares an orientation it must certify *as declared*;
/// otherwise the unique certifying orientation is searched for. Returns the
/// certified mapping so the caller can back-fill the orientation.
pub fn certify_face_match<F>(
    blocks: &[Block],
    fm: &FaceMatch,
    transform: F,
    tol: Float,
) -> Result<CertifiedMapping, String>
where
    F: Fn([Float; 3]) -> [Float; 3],
{
    let pa = Patch::from_record(&fm.block1).map_err(|e| e.to_string())?;
    pa.check_in(blocks).map_err(|e| e.to_string())?;
    let pb = Patch::from_record(&fm.block2).map_err(|e| e.to_string())?;
    pb.check_in(blocks).map_err(|e| e.to_string())?;
    let block_a = &blocks[pa.block];
    let block_b = &blocks[pb.block];
    let declared = fm.orientation.as_ref().map(|o| match o.permutation_matrix {
        Some(m) => Orientation::index_from_permutation_matrix(m).unwrap_or(o.permutation_index),
        None => o.permutation_index,
    });
    match declared {
        Some(perm) => certify_permutation(block_a, &pa, block_b, &pb, transform, tol, perm)
            .map_err(|e| e.to_string()),
        None => certify_correspondence(block_a, &pa, block_b, &pb, transform, tol)
            .map_err(|e| e.to_string()),
    }
}
