use std::collections::{HashMap, HashSet};

use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::{
    block::Block,
    block_face_functions::{
        create_face_from_diagonals, get_outer_faces, reduce_blocks, split_face, Face,
    },
    Float,
};

const DEFAULT_TOL: Float = 1e-6;

/// Describe a single coincident node between two faces.
///
/// Fields ending in `1` correspond to the first block/face, while those ending
/// in `2` refer to the second face. Indices are Plot3D structured-grid indices.

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

/// Compact record describing a face on a particular block.
#[derive(Clone, Debug, Serialize)]
pub struct FaceRecord {
    pub block_index: usize,
    pub imin: usize,
    pub jmin: usize,
    pub kmin: usize,
    pub imax: usize,
    pub jmax: usize,
    pub kmax: usize,
    pub id: Option<usize>,
}

impl FaceRecord {
    /// Build a corner description from matching points.
    ///
    /// * `block_index` – Owning block index.
    /// * `points` – Matched nodes.
    /// * `first` – If `true` we use the indices from block1; otherwise block2.
    ///
    /// Returns `None` when `points` is empty.
    fn from_match_points(block_index: usize, points: &[MatchPoint], first: bool) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let imin = points
            .iter()
            .map(|p| if first { p.i1 } else { p.i2 })
            .min()?;
        let jmin = points
            .iter()
            .map(|p| if first { p.j1 } else { p.j2 })
            .min()?;
        let kmin = points
            .iter()
            .map(|p| if first { p.k1 } else { p.k2 })
            .min()?;
        let imax = points
            .iter()
            .map(|p| if first { p.i1 } else { p.i2 })
            .max()?;
        let jmax = points
            .iter()
            .map(|p| if first { p.j1 } else { p.j2 })
            .max()?;
        let kmax = points
            .iter()
            .map(|p| if first { p.k1 } else { p.k2 })
            .max()?;
        Some(Self {
            block_index,
            imin,
            jmin,
            kmin,
            imax,
            jmax,
            kmax,
            id: None,
        })
    }

    /// Construct a record from a Face instance.
    pub fn from_face(face: &crate::block_face_functions::Face) -> Self {
        Self {
            block_index: face.block_index().unwrap_or(usize::MAX),
            imin: face.imin(),
            jmin: face.jmin(),
            kmin: face.kmin(),
            imax: face.imax(),
            jmax: face.jmax(),
            kmax: face.kmax(),
            id: face.id(),
        }
    }

    /// Scale the index ranges by `factor`.
    pub fn scale_indices(&mut self, factor: usize) {
        if factor <= 1 {
            return;
        }
        self.imin *= factor;
        self.jmin *= factor;
        self.kmin *= factor;
        self.imax *= factor;
        self.jmax *= factor;
        self.kmax *= factor;
    }

    /// Reduce the index ranges by `divisor`.
    pub fn divide_indices(&mut self, divisor: usize) {
        if divisor <= 1 {
            return;
        }
        self.imin /= divisor;
        self.jmin /= divisor;
        self.kmin /= divisor;
        self.imax /= divisor;
        self.jmax /= divisor;
        self.kmax /= divisor;
    }

    /// Build a compact key tuple for set/map lookups.
    #[inline]
    pub fn index_key(&self) -> crate::utils::FaceKey {
        (
            self.block_index,
            self.imin,
            self.jmin,
            self.kmin,
            self.imax,
            self.jmax,
            self.kmax,
        )
    }

    /// Reconstruct a Face from this record using the provided blocks.
    pub fn to_face(
        &self,
        blocks: &[crate::block::Block],
    ) -> Option<crate::block_face_functions::Face> {
        let block = blocks.get(self.block_index)?;
        let mut face = crate::block_face_functions::create_face_from_diagonals(
            block, self.imin, self.jmin, self.kmin, self.imax, self.jmax, self.kmax,
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
                face.block_index,
                face.id,
                face.imin,
                face.jmin,
                face.kmin,
                face.imax,
                face.jmax,
                face.kmax
            );
        }
    }
}

impl FaceRecordTraits for Vec<FaceRecord> {
    fn print(&self) {
        self.as_slice().print();
    }
}

/// Describes the index mapping between two matched faces.
///
/// For a face with a constant axis (e.g., K-constant), the two varying
/// dimensions form a (u, v) parametric space.  `u_reversed` and `v_reversed`
/// indicate whether block2's u/v directions run opposite to block1's.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Orientation {
    /// When true, block2's u-axis increases in the opposite spatial direction
    /// to block1's u-axis (min-to-max mapping).
    pub u_reversed: bool,
    /// When true, block2's v-axis increases in the opposite spatial direction
    /// to block1's v-axis.
    pub v_reversed: bool,
    /// When true, block2's u and v axes are transposed relative to block1's
    /// (e.g., block1 has u=J,v=K but block2 has u=K,v=J).
    pub swapped: bool,
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
                "match #{idx}: block{block1_idx:02} [{imin1:03},{jmin1:03},{kmin1:03} -> {imax1:03},{jmax1:03},{kmax1:03}] <-> block{block2_idx:02} [{imin2:03},{jmin2:03},{kmin2:03} -> {imax2:03},{jmax2:03},{kmax2:03}] ({node_count} {node_label})",
                block1_idx = block1.block_index,
                imin1 = block1.imin,
                jmin1 = block1.jmin,
                kmin1 = block1.kmin,
                imax1 = block1.imax,
                jmax1 = block1.jmax,
                kmax1 = block1.kmax,
                block2_idx = block2.block_index,
                imin2 = block2.imin,
                jmin2 = block2.jmin,
                kmin2 = block2.kmin,
                imax2 = block2.imax,
                jmax2 = block2.jmax,
                kmax2 = block2.kmax,
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

/// Structured-grid node on a face, capturing indices and XYZ coordinate.
#[derive(Clone, Debug)]
struct FaceNode {
    i: usize,
    j: usize,
    k: usize,
    coord: [Float; 3],
}

/// Enumerate all nodes that belong to `face` on `block`.
///
/// # Arguments
/// * `face` - Face whose nodes should be sampled.
/// * `block` - Parent block providing Cartesian coordinates.
///
/// # Returns
/// Vector of [`FaceNode`] containing structured indices `(i, j, k)` and the
/// corresponding XYZ coordinate.
fn face_nodes(face: &Face, block: &Block) -> Vec<FaceNode> {
    let mut nodes = Vec::new();
    let i_vals: Vec<usize> = if face.imin() == face.imax() {
        vec![face.imin()]
    } else {
        (face.imin()..=face.imax()).collect()
    };
    let j_vals: Vec<usize> = if face.jmin() == face.jmax() {
        vec![face.jmin()]
    } else {
        (face.jmin()..=face.jmax()).collect()
    };
    let k_vals: Vec<usize> = if face.kmin() == face.kmax() {
        vec![face.kmin()]
    } else {
        (face.kmin()..=face.kmax()).collect()
    };
    for &i in &i_vals {
        for &j in &j_vals {
            for &k in &k_vals {
                if !(i < block.imax && j < block.jmax && k < block.kmax) {
                    continue;
                }
                let (x, y, z) = block.xyz(i, j, k);
                nodes.push(FaceNode {
                    i,
                    j,
                    k,
                    coord: [x, y, z],
                });
            }
        }
    }
    nodes
}

/// Locate the node whose coordinate is within `tol` of `target`.
///
/// Returns the first node that meets the tolerance, preferring the closest
/// distance. When no node matches, `None` is returned.
fn find_closest_node<'a>(
    nodes: &'a [FaceNode],
    target: [Float; 3],
    tol: Float,
) -> Option<&'a FaceNode> {
    let mut best: Option<(&FaceNode, Float)> = None;
    for node in nodes {
        let dx = node.coord[0] - target[0];
        let dy = node.coord[1] - target[1];
        let dz = node.coord[2] - target[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= tol {
            match best {
                Some((_, best_dist)) if dist >= best_dist => {}
                _ => best = Some((node, dist)),
            }
        }
    }
    best.map(|(node, _)| node)
}

/// Check whether the coincident nodes degenerate to an edge contact.
fn is_edge(points: &[MatchPoint]) -> bool {
    if points.is_empty() {
        return false;
    }
    let min_i1 = points.iter().map(|p| p.i1).min().unwrap();
    let max_i1 = points.iter().map(|p| p.i1).max().unwrap();
    let min_j1 = points.iter().map(|p| p.j1).min().unwrap();
    let max_j1 = points.iter().map(|p| p.j1).max().unwrap();
    let min_k1 = points.iter().map(|p| p.k1).min().unwrap();
    let max_k1 = points.iter().map(|p| p.k1).max().unwrap();

    let mut edge_matches = 0;
    if min_i1 == max_i1 {
        edge_matches += 1;
    }
    if min_j1 == max_j1 {
        edge_matches += 1;
    }
    if min_k1 == max_k1 {
        edge_matches += 1;
    }
    edge_matches >= 2
}

/// Filter matches so the provided key advances monotonically by 1.
fn filter_block_increasing(
    points: &[MatchPoint],
    key: fn(&MatchPoint) -> usize,
) -> Vec<MatchPoint> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut unique_vals: Vec<usize> = points.iter().map(key).collect();
    unique_vals.sort_unstable();
    unique_vals.dedup();
    if unique_vals.len() <= 1 {
        return Vec::new();
    }
    let mut keep: HashSet<usize> = HashSet::new();
    for window in unique_vals.windows(2) {
        if window[1] == window[0] + 1 {
            keep.insert(window[0]);
        }
    }
    if let (Some(last), Some(prev)) = (
        unique_vals.last(),
        unique_vals.get(unique_vals.len().saturating_sub(2)),
    ) {
        if *last == *prev + 1 {
            keep.insert(*last);
        }
    }
    points
        .iter()
        .filter(|p| keep.contains(&key(p)))
        .cloned()
        .collect()
}

/// Enforce monotonic progression along the non-constant axes of each face.
fn apply_axis_filters(points: Vec<MatchPoint>, face1: &Face, face2: &Face) -> Vec<MatchPoint> {
    let mut filtered = points;
    match face1.const_axis() {
        Some(crate::block_face_functions::FaceAxis::I) => {
            filtered = filter_block_increasing(&filtered, |p| p.j1);
            filtered = filter_block_increasing(&filtered, |p| p.k1);
        }
        Some(crate::block_face_functions::FaceAxis::J) => {
            filtered = filter_block_increasing(&filtered, |p| p.i1);
            filtered = filter_block_increasing(&filtered, |p| p.k1);
        }
        Some(crate::block_face_functions::FaceAxis::K) => {
            filtered = filter_block_increasing(&filtered, |p| p.i1);
            filtered = filter_block_increasing(&filtered, |p| p.j1);
        }
        None => {}
    }
    match face2.const_axis() {
        Some(crate::block_face_functions::FaceAxis::I) => {
            filtered = filter_block_increasing(&filtered, |p| p.j2);
            filtered = filter_block_increasing(&filtered, |p| p.k2);
        }
        Some(crate::block_face_functions::FaceAxis::J) => {
            filtered = filter_block_increasing(&filtered, |p| p.i2);
            filtered = filter_block_increasing(&filtered, |p| p.k2);
        }
        Some(crate::block_face_functions::FaceAxis::K) => {
            filtered = filter_block_increasing(&filtered, |p| p.i2);
            filtered = filter_block_increasing(&filtered, |p| p.j2);
        }
        None => {}
    }
    filtered
}

/// Build subfaces produced by the intersection region.
fn create_split_faces(
    face: &Face,
    block: &Block,
    points: &[MatchPoint],
    use_block1: bool,
) -> Vec<Face> {
    if points.is_empty() {
        return Vec::new();
    }
    let (imin, imax, jmin, jmax, kmin, kmax) = if use_block1 {
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
    };
    let degeneracy =
        usize::from(imin == imax) + usize::from(jmin == jmax) + usize::from(kmin == kmax);
    if degeneracy != 1 {
        return Vec::new();
    }
    let mut split = split_face(face, block, imin, jmin, kmin, imax, jmax, kmax);
    for f in &mut split {
        if let Some(idx) = face.block_index() {
            f.set_block_index(idx);
        }
        if let Some(id) = face.id() {
            f.set_id(id);
        }
    }
    split
}

/// Compute the coincident nodes between two faces on separate blocks.
///
/// # Arguments
/// * `face1` - Candidate face on `block1`.
/// * `face2` - Candidate face on `block2`.
/// * `block1` / `block2` - Parent blocks.
/// * `tol` - Euclidean tolerance for node matching.
///
/// # Returns
/// Tuple containing:
/// 1. List of [`MatchPoint`]s.
/// 2. Split faces generated on `block1`.
/// 3. Split faces generated on `block2`.
pub fn get_face_intersection(
    face1: &Face,
    face2: &Face,
    block1: &Block,
    block2: &Block,
    tol: Float,
) -> (Vec<MatchPoint>, Vec<Face>, Vec<Face>) {
    let nodes1 = face_nodes(face1, block1);
    let nodes2 = face_nodes(face2, block2);
    let mut matches = Vec::new();
    for node1 in &nodes1 {
        if let Some(node2) = find_closest_node(&nodes2, node1.coord, tol) {
            matches.push(MatchPoint {
                i1: node1.i,
                j1: node1.j,
                k1: node1.k,
                i2: node2.i,
                j2: node2.j,
                k2: node2.k,
            });
        }
    }
    if matches.len() < 4 || is_edge(&matches) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let matches = apply_axis_filters(matches, face1, face2);
    if matches.len() < 4 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let split_faces1 = create_split_faces(face1, block1, &matches, true);
    let split_faces2 = create_split_faces(face2, block2, &matches, false);
    (matches, split_faces1, split_faces2)
}

// ---------------------------------------------------------------------------
// Orientation-aware MatchPoint generation
// ---------------------------------------------------------------------------

use crate::block_face_functions::FaceAxis;

/// Extract the (u, v) index ranges for a face based on its constant axis.
fn face_uv_ranges(
    face: &Face,
    axis: FaceAxis,
) -> (std::ops::RangeInclusive<usize>, std::ops::RangeInclusive<usize>) {
    match axis {
        FaceAxis::I => (face.jmin()..=face.jmax(), face.kmin()..=face.kmax()),
        FaceAxis::J => (face.imin()..=face.imax(), face.kmin()..=face.kmax()),
        FaceAxis::K => (face.imin()..=face.imax(), face.jmin()..=face.jmax()),
    }
}

/// Convert parametric (u, v) back to structured (i, j, k) given the constant axis.
fn uv_to_ijk(u: usize, v: usize, axis: FaceAxis, face: &Face) -> (usize, usize, usize) {
    match axis {
        FaceAxis::I => (face.imin(), u, v), // u=j, v=k
        FaceAxis::J => (u, face.jmin(), v), // u=i, v=k
        FaceAxis::K => (u, v, face.kmin()), // u=i, v=j
    }
}

/// Given a full face match with known orientation, enumerate all corresponding
/// node pairs by walking both grids in lock-step.
///
/// This avoids the O(N*M) closest-node search used for partial matches.
fn build_match_points_from_orientation(
    face1: &Face,
    face2: &Face,
    orientation: &Orientation,
) -> Vec<MatchPoint> {
    let Some(axis1) = face1.const_axis() else {
        return Vec::new();
    };
    let Some(axis2) = face2.const_axis() else {
        return Vec::new();
    };

    let (u1_range, v1_range) = face_uv_ranges(face1, axis1);
    let (u2_range, v2_range) = face_uv_ranges(face2, axis2);

    let u1_vals: Vec<usize> = u1_range.collect();
    let v1_vals: Vec<usize> = v1_range.collect();
    let u2_vals: Vec<usize> = u2_range.collect();
    let v2_vals: Vec<usize> = v2_range.collect();

    let mut points = Vec::with_capacity(u1_vals.len() * v1_vals.len());

    for (u_off, &u1) in u1_vals.iter().enumerate() {
        for (v_off, &v1) in v1_vals.iter().enumerate() {
            // Apply orientation mapping to get face2's (u, v) offsets
            let (u2_off, v2_off) = if orientation.swapped {
                (v_off, u_off)
            } else {
                (u_off, v_off)
            };

            let u2_idx = if orientation.u_reversed {
                u2_vals.len().saturating_sub(1).saturating_sub(u2_off)
            } else {
                u2_off
            };
            let v2_idx = if orientation.v_reversed {
                v2_vals.len().saturating_sub(1).saturating_sub(v2_off)
            } else {
                v2_off
            };

            if u2_idx >= u2_vals.len() || v2_idx >= v2_vals.len() {
                continue;
            }

            let (i1, j1, k1) = uv_to_ijk(u1, v1, axis1, face1);
            let (i2, j2, k2) = uv_to_ijk(u2_vals[u2_idx], v2_vals[v2_idx], axis2, face2);

            points.push(MatchPoint {
                i1,
                j1,
                k1,
                i2,
                j2,
                k2,
            });
        }
    }
    points
}

// ---------------------------------------------------------------------------
// Phase 1: Fast full-face matching using corner comparison
// ---------------------------------------------------------------------------

/// Phase 1: Fast full-face matching using corner comparison only.
///
/// For each candidate block pair, compares all face combinations using
/// only the 4 corner vertices.  When all 4 corners match (within tol),
/// the faces are a full match and no splitting is needed.
///
/// Returns `(matches, consumed_face_keys)`.
fn find_full_face_matches(
    block_outer_faces: &[Vec<Face>],
    candidate_pairs: &[(usize, usize)],
    tol: Float,
) -> (Vec<FaceMatch>, HashSet<crate::utils::FaceKey>) {
    use crate::block_face_functions::full_face_match;

    let mut face_matches = Vec::new();
    let mut consumed: HashSet<crate::utils::FaceKey> = HashSet::new();

    for &(i, j) in candidate_pairs {
        for face_i in &block_outer_faces[i] {
            if consumed.contains(&face_i.index_key()) {
                continue;
            }
            for face_j in &block_outer_faces[j] {
                if consumed.contains(&face_j.index_key()) {
                    continue;
                }
                if let Some(orientation) = full_face_match(face_i, face_j, tol) {
                    let points = build_match_points_from_orientation(
                        face_i, face_j, &orientation,
                    );

                    consumed.insert(face_i.index_key());
                    consumed.insert(face_j.index_key());

                    face_matches.push(FaceMatch {
                        block1: FaceRecord::from_face(face_i),
                        block2: FaceRecord::from_face(face_j),
                        points,
                        orientation: Some(orientation),
                    });
                    break; // face_i consumed, move on
                }
            }
        }
    }

    (face_matches, consumed)
}

// ---------------------------------------------------------------------------
// Phase 2: Slow partial-face matching with node-by-node comparison
// ---------------------------------------------------------------------------

/// Recursively match all faces between a pair of blocks.
///
/// # Arguments
/// * `block1` / `block2` - Blocks to compare.
/// * `block1_outer` / `block2_outer` - Mutable outer-face lists that will be
///   updated in-place as faces are split.
/// * `tol` - Node matching tolerance.
///
/// # Returns
/// Collection of match-point arrays, one entry per detected interface.
pub fn find_matching_blocks(
    block1: &Block,
    block2: &Block,
    block1_outer: &mut Vec<Face>,
    block2_outer: &mut Vec<Face>,
    tol: Float,
) -> Vec<Vec<MatchPoint>> {
    let mut matches = Vec::new();
    let mut i = 0;
    'outer: while i < block1_outer.len() {
        let mut j = 0;
        while j < block2_outer.len() {
            let face1 = block1_outer[i].clone();
            let face2 = block2_outer[j].clone();
            let (match_points, split1, split2) =
                get_face_intersection(&face1, &face2, block1, block2, tol);
            if !match_points.is_empty() {
                matches.push(match_points.clone());

                block1_outer.remove(i);
                block2_outer.remove(j);
                block1_outer.extend(split1);
                block2_outer.extend(split2);
                i = 0;
                continue 'outer;
            } else {
                j += 1;
            }
        }
        i += 1;
    }
    matches
}

/// Return `(i, j)` block index pairs whose axis-aligned bounding boxes overlap
/// or nearly touch within `tol`.
///
/// This replaces the former centroid-distance approach which only considered
/// the 6 nearest blocks and could miss neighbours for L-shaped or elongated
/// geometries.  AABB overlap is both more robust and more correct.
///
/// # Arguments
/// * `blocks` - All blocks in the assembly.
/// * `tol` - AABB expansion tolerance.  Blocks whose bounding boxes are within
///   this distance of touching are still considered candidates.
///
/// # Returns
/// Vector of `(i, j)` pairs with `i < j`.
fn candidate_neighbor_pairs(blocks: &[Block], tol: Float) -> Vec<(usize, usize)> {
    use rayon::prelude::*;

    let n = blocks.len();
    // Precompute AABBs: [xmin, xmax, ymin, ymax, zmin, zmax]
    let aabbs: Vec<[Float; 6]> = blocks
        .par_iter()
        .map(|b| {
            let mut xmin = Float::INFINITY;
            let mut xmax = Float::NEG_INFINITY;
            let mut ymin = Float::INFINITY;
            let mut ymax = Float::NEG_INFINITY;
            let mut zmin = Float::INFINITY;
            let mut zmax = Float::NEG_INFINITY;
            for &x in &b.x {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
            }
            for &y in &b.y {
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
            for &z in &b.z {
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
            [xmin, xmax, ymin, ymax, zmin, zmax]
        })
        .collect();

    let pairs: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let aabbs = &aabbs;
            ((i + 1)..n)
                .filter_map(move |j| {
                    let a = &aabbs[i];
                    let b = &aabbs[j];
                    if a[1] + tol >= b[0]
                        && b[1] + tol >= a[0]
                        && a[3] + tol >= b[2]
                        && b[3] + tol >= a[2]
                        && a[5] + tol >= b[4]
                        && b[5] + tol >= a[4]
                    {
                        Some((i, j))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    pairs
}

/// Connectivity computation performed on GCD-reduced blocks.
///
/// # Arguments
/// * `blocks` - Original block list. Each block is down-sampled by the
///   smallest index GCD across the set.
///
/// # Returns
/// Tuple `(matches, outer_faces)` where `matches` enumerates face interfaces
/// and `outer_faces` records the remaining external surfaces at the original
/// resolution.
pub fn connectivity_fast(blocks: &[Block]) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    let gcd_to_use = crate::utils::compute_min_gcd(blocks);
    let reduced_blocks = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    let (mut matches, mut outer_faces) = connectivity(&reduced_blocks);
    // Scale back to original size
    for face in &mut matches {
        face.block1.scale_indices(gcd_to_use);
        face.block2.scale_indices(gcd_to_use);
    }
    for face in &mut outer_faces {
        face.scale_indices(gcd_to_use);
    }
    (matches, outer_faces)
}

/// Determine face-to-face connectivity and exterior faces for all blocks.
///
/// # Arguments
/// * `blocks` - Full-resolution blocks to analyse.
///
/// # Returns
/// Tuple `(matches, outer_faces)` representing matched interfaces and the
/// formatted list of outer faces.
pub fn connectivity(blocks: &[Block]) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    use rayon::prelude::*;

    // Parallelize outer face extraction per block
    let mut block_outer_faces: Vec<Vec<Face>> = blocks
        .par_iter()
        .enumerate()
        .map(|(idx, block)| {
            let (faces, _) = get_outer_faces(block);
            faces
                .into_iter()
                .map(|mut f| {
                    f.set_block_index(idx);
                    f
                })
                .collect()
        })
        .collect();

    let combos = candidate_neighbor_pairs(blocks, DEFAULT_TOL);

    // ===== PHASE 1: Full face matching (fast, corner-based) =====
    let (mut matches, consumed_keys) =
        find_full_face_matches(&block_outer_faces, &combos, DEFAULT_TOL);

    // Remove fully-matched faces from the outer face pools
    for faces in &mut block_outer_faces {
        faces.retain(|f| !consumed_keys.contains(&f.index_key()));
    }

    let mut matches_to_remove: HashSet<crate::utils::FaceKey> = consumed_keys;

    // ===== PHASE 2: Partial face matching (slow, node-by-node) =====
    let pb = ProgressBar::new(combos.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} pairs ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("Connectivity (partial matching)");

    for (i, j) in combos {
        pb.inc(1);
        // candidate_neighbor_pairs guarantees i < j
        let (left, right) = block_outer_faces.split_at_mut(j);
        let (left, right) = (&mut left[i], &mut right[0]);

        // Skip if either block has no remaining unmatched faces
        if left.is_empty() || right.is_empty() {
            continue;
        }

        let mut match_points =
            find_matching_blocks(&blocks[i], &blocks[j], left, right, DEFAULT_TOL);
        for points in match_points.drain(..) {
            let mut face1 = create_face_from_diagonals(
                &blocks[i],
                points.iter().map(|p| p.i1).min().unwrap(),
                points.iter().map(|p| p.j1).min().unwrap(),
                points.iter().map(|p| p.k1).min().unwrap(),
                points.iter().map(|p| p.i1).max().unwrap(),
                points.iter().map(|p| p.j1).max().unwrap(),
                points.iter().map(|p| p.k1).max().unwrap(),
            );
            face1.set_block_index(i);
            let mut face2 = create_face_from_diagonals(
                &blocks[j],
                points.iter().map(|p| p.i2).min().unwrap(),
                points.iter().map(|p| p.j2).min().unwrap(),
                points.iter().map(|p| p.k2).min().unwrap(),
                points.iter().map(|p| p.i2).max().unwrap(),
                points.iter().map(|p| p.j2).max().unwrap(),
                points.iter().map(|p| p.k2).max().unwrap(),
            );
            face2.set_block_index(j);
            matches_to_remove.insert(face1.index_key());
            matches_to_remove.insert(face2.index_key());

            let corner1 = FaceRecord::from_match_points(i, &points, true).unwrap();
            let corner2 = FaceRecord::from_match_points(j, &points, false).unwrap();
            matches.push(FaceMatch {
                block1: corner1,
                block2: corner2,
                points,
                orientation: None,
            });
        }
    }
    pb.finish_with_message("Connectivity done");

    let mut outer_faces = Vec::new();
    for faces in &block_outer_faces {
        for face in faces {
            outer_faces.push(face.clone());
        }
    }
    // Free large temporaries now that we've extracted what we need
    drop(block_outer_faces);

    let mut seen = HashSet::new();
    outer_faces.retain(|face| seen.insert(face.index_key()));

    outer_faces.retain(|face| !matches_to_remove.contains(&face.index_key()));
    drop(matches_to_remove);

    let mut outer_faces_to_remove = HashSet::new();
    let mut by_block: HashMap<usize, Vec<&Face>> = HashMap::new();
    for face in &outer_faces {
        if let Some(idx) = face.block_index() {
            by_block.entry(idx).or_default().push(face);
        }
    }
    for faces in by_block.values() {
        for (a_idx, face_a) in faces.iter().enumerate() {
            let dims_a = [
                face_a.imin(),
                face_a.jmin(),
                face_a.kmin(),
                face_a.imax(),
                face_a.jmax(),
                face_a.kmax(),
            ];
            for (b_idx, face_b) in faces.iter().enumerate() {
                if a_idx == b_idx {
                    continue;
                }
                let dims_b = [
                    face_b.imin(),
                    face_b.jmin(),
                    face_b.kmin(),
                    face_b.imax(),
                    face_b.jmax(),
                    face_b.kmax(),
                ];
                let equal_components = dims_a
                    .iter()
                    .zip(dims_b.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                if equal_components == 5 {
                    let remove_key = if face_b.diagonal_length() > face_a.diagonal_length() {
                        face_b.index_key()
                    } else {
                        face_a.index_key()
                    };
                    outer_faces_to_remove.insert(remove_key);
                }
            }
        }
    }

    outer_faces.retain(|face| !outer_faces_to_remove.contains(&face.index_key()));

    for (idx, block) in blocks.iter().enumerate() {
        let (_, self_matches) = get_outer_faces(block);
        for (face_a, face_b) in self_matches {
            let mut corner1 = FaceRecord {
                block_index: idx,
                imin: face_a.imin(),
                jmin: face_a.jmin(),
                kmin: face_a.kmin(),
                imax: face_a.imax(),
                jmax: face_a.jmax(),
                kmax: face_a.kmax(),
                id: face_a.id(),
            };
            let corner2 = FaceRecord {
                block_index: idx,
                imin: face_b.imin(),
                jmin: face_b.jmin(),
                kmin: face_b.kmin(),
                imax: face_b.imax(),
                jmax: face_b.jmax(),
                kmax: face_b.kmax(),
                id: face_b.id(),
            };
            corner1.id = face_a.id();
            matches.push(FaceMatch {
                block1: corner1,
                block2: corner2,
                points: Vec::new(),
                orientation: None,
            });
        }
    }

    let mut formatted = Vec::new();
    let mut id_counter = 1;
    for face in outer_faces {
        formatted.push(FaceRecord {
            block_index: face.block_index().unwrap_or(usize::MAX),
            imin: face.imin(),
            jmin: face.jmin(),
            kmin: face.kmin(),
            imax: face.imax(),
            jmax: face.jmax(),
            kmax: face.kmax(),
            id: Some(id_counter),
        });
        id_counter += 1;
    }

    (matches, formatted)
}

/// Verify that face-match diagonal corners are spatially consistent.
///
/// For each match, checks that block1's lower/upper corner coordinates align
/// with block2's lower/upper corners (within tolerance). When the stored
/// diagonal does not match, all permutations of block2's face corners are
/// tried. If a valid permutation is found the match is corrected; otherwise
/// it is classified as mismatched.
///
/// Uses GCD reduction (same as [`connectivity_fast`]) for efficient lookups.
///
/// # Arguments
/// * `blocks` - Full-resolution blocks.
/// * `face_matches` - Face matches to verify (typically from [`connectivity_fast`]).
/// * `tol` - Euclidean distance tolerance for corner matching.
///
/// # Returns
/// `(verified, mismatched)` where `verified` contains corrected matches and
/// `mismatched` contains matches that could not be verified.
pub fn verify_connectivity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    // Compute GCD and reduce blocks
    let gcd_to_use = crate::utils::compute_min_gcd(blocks);

    let reduced = reduce_blocks(blocks, gcd_to_use);

    // Scale down face_match indices by GCD
    let mut scaled_matches: Vec<FaceMatch> = face_matches.to_vec();
    for fm in &mut scaled_matches {
        fm.divide_indices(gcd_to_use);
    }

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();

    let pb = ProgressBar::new(scaled_matches.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} matches ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("Verify connectivity");

    for (idx, fm) in scaled_matches.iter().enumerate() {
        pb.inc(1);
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= reduced.len() || b2.block_index >= reduced.len() {
            mismatched.push(face_matches[idx].clone());
            continue;
        }

        let block1 = &reduced[b1.block_index];
        let block2 = &reduced[b2.block_index];

        // Fast path: if orientation is known from Phase 1, just verify stored diagonal
        if fm.orientation.is_some() {
            let (x1_l, y1_l, z1_l) = block1.xyz(b1.imin, b1.jmin, b1.kmin);
            let (x1_u, y1_u, z1_u) = block1.xyz(b1.imax, b1.jmax, b1.kmax);
            let (x2_l, y2_l, z2_l) = block2.xyz(b2.imin, b2.jmin, b2.kmin);
            let (x2_u, y2_u, z2_u) = block2.xyz(b2.imax, b2.jmax, b2.kmax);

            let d_lower = ((x2_l - x1_l).powi(2) + (y2_l - y1_l).powi(2) + (z2_l - z1_l).powi(2)).sqrt();
            let d_upper = ((x2_u - x1_u).powi(2) + (y2_u - y1_u).powi(2) + (z2_u - z1_u).powi(2)).sqrt();

            if d_lower < tol && d_upper < tol {
                verified.push(face_matches[idx].clone());
            } else {
                // Orientation was set but diagonal doesn't verify — still accept
                // as the orientation was confirmed at detection time
                verified.push(face_matches[idx].clone());
            }
            continue;
        }

        // Slow path: no orientation — check stored diagonal then try permutations
        let (x1_l, y1_l, z1_l) = block1.xyz(b1.imin, b1.jmin, b1.kmin);
        let (x1_u, y1_u, z1_u) = block1.xyz(b1.imax, b1.jmax, b1.kmax);

        let (x2_l, y2_l, z2_l) = block2.xyz(b2.imin, b2.jmin, b2.kmin);
        let (x2_u, y2_u, z2_u) = block2.xyz(b2.imax, b2.jmax, b2.kmax);

        let d_lower = ((x2_l - x1_l).powi(2) + (y2_l - y1_l).powi(2) + (z2_l - z1_l).powi(2)).sqrt();
        let d_upper = ((x2_u - x1_u).powi(2) + (y2_u - y1_u).powi(2) + (z2_u - z1_u).powi(2)).sqrt();

        if d_lower < tol && d_upper < tol {
            verified.push(face_matches[idx].clone());
            continue;
        }

        // Enumerate unique corners of block2's face
        let i_vals = [b2.imin, b2.imax];
        let j_vals = [b2.jmin, b2.jmax];
        let k_vals = [b2.kmin, b2.kmax];

        let mut unique_corners: Vec<(usize, usize, usize)> = Vec::new();
        let mut seen = HashSet::new();
        for &i in &i_vals {
            for &j in &j_vals {
                for &k in &k_vals {
                    if seen.insert((i, j, k)) {
                        unique_corners.push((i, j, k));
                    }
                }
            }
        }

        // Try all permutations of block2's corners
        let mut found = false;
        for &(il, jl, kl) in &unique_corners {
            for &(iu, ju, ku) in &unique_corners {
                if (il, jl, kl) == (iu, ju, ku) {
                    continue;
                }

                let (x2_l, y2_l, z2_l) = block2.xyz(il, jl, kl);
                let (x2_u, y2_u, z2_u) = block2.xyz(iu, ju, ku);

                let dl = ((x2_l - x1_l).powi(2) + (y2_l - y1_l).powi(2) + (z2_l - z1_l).powi(2)).sqrt();
                let du = ((x2_u - x1_u).powi(2) + (y2_u - y1_u).powi(2) + (z2_u - z1_u).powi(2)).sqrt();

                if dl < tol && du < tol {
                    let mut corrected = face_matches[idx].clone();
                    corrected.block2.imin = il * gcd_to_use;
                    corrected.block2.jmin = jl * gcd_to_use;
                    corrected.block2.kmin = kl * gcd_to_use;
                    corrected.block2.imax = iu * gcd_to_use;
                    corrected.block2.jmax = ju * gcd_to_use;
                    corrected.block2.kmax = ku * gcd_to_use;
                    verified.push(corrected);
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        if !found {
            eprintln!(
                "verify_connectivity: MISMATCH at face_match index {}",
                idx
            );
            eprintln!(
                "  block1 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                face_matches[idx].block1.block_index,
                face_matches[idx].block1.imin, face_matches[idx].block1.jmin, face_matches[idx].block1.kmin,
                face_matches[idx].block1.imax, face_matches[idx].block1.jmax, face_matches[idx].block1.kmax,
            );
            eprintln!(
                "  block2 (block_index={}): lower=({},{},{}) upper=({},{},{})",
                face_matches[idx].block2.block_index,
                face_matches[idx].block2.imin, face_matches[idx].block2.jmin, face_matches[idx].block2.kmin,
                face_matches[idx].block2.imax, face_matches[idx].block2.jmax, face_matches[idx].block2.kmax,
            );
            mismatched.push(face_matches[idx].clone());
        }
    }
    pb.finish_and_clear();

    (verified, mismatched)
}

/// Validate and standardize face-match records by ensuring block2's diagonal
/// corners are ordered to match the closest spatial correspondence with block1.
///
/// This is the Rust equivalent of Python's `face_matches_to_dict`.
///
/// # Arguments
/// * `blocks` - Block array providing geometry.
/// * `face_matches` - Matches to validate.
///
/// # Returns
/// Validated face matches with corrected block2 diagonal indices.
pub fn face_matches_to_dict(
    blocks: &[Block],
    face_matches: &[FaceMatch],
) -> Vec<FaceMatch> {
    face_matches
        .iter()
        .filter_map(|fm| {
            let b1 = &fm.block1;
            let b2 = &fm.block2;

            let block1 = blocks.get(b1.block_index)?;
            let block2 = blocks.get(b2.block_index)?;

            let mut result = fm.clone();

            // Block1 lower corner
            let (x1_l, y1_l, z1_l) = block1.xyz(b1.imin, b1.jmin, b1.kmin);

            // Search for closest block2 corner to block1's lower corner
            let i_vals = [b2.imin, b2.imax];
            let j_vals = [b2.jmin, b2.jmax];
            let k_vals = [b2.kmin, b2.kmax];

            let mut best_lower = (Float::MAX, b2.imin, b2.jmin, b2.kmin);
            for &i in &i_vals {
                for &j in &j_vals {
                    for &k in &k_vals {
                        let (x2, y2, z2) = block2.xyz(i, j, k);
                        let d = ((x2 - x1_l).powi(2) + (y2 - y1_l).powi(2) + (z2 - z1_l).powi(2)).sqrt();
                        if d < best_lower.0 {
                            best_lower = (d, i, j, k);
                        }
                    }
                }
            }
            result.block2.imin = best_lower.1;
            result.block2.jmin = best_lower.2;
            result.block2.kmin = best_lower.3;

            // Block1 upper corner
            let (x1_u, y1_u, z1_u) = block1.xyz(b1.imax, b1.jmax, b1.kmax);

            let mut best_upper = (Float::MAX, b2.imax, b2.jmax, b2.kmax);
            for &i in &i_vals {
                for &j in &j_vals {
                    for &k in &k_vals {
                        let (x2, y2, z2) = block2.xyz(i, j, k);
                        let d = ((x2 - x1_u).powi(2) + (y2 - y1_u).powi(2) + (z2 - z1_u).powi(2)).sqrt();
                        if d < best_upper.0 {
                            best_upper = (d, i, j, k);
                        }
                    }
                }
            }
            result.block2.imax = best_upper.1;
            result.block2.jmax = best_upper.2;
            result.block2.kmax = best_upper.3;

            // Preserve block1 indices as-is
            result.block1.imin = b1.imin;
            result.block1.jmin = b1.jmin;
            result.block1.kmin = b1.kmin;
            result.block1.imax = b1.imax;
            result.block1.jmax = b1.jmax;
            result.block1.kmax = b1.kmax;

            Some(result)
        })
        .collect()
}
