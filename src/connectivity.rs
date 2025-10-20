use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::{
    block::Block,
    block_face_functions::{create_face_from_diagonals, get_outer_faces, split_face, Face},
};

const DEFAULT_TOL: f64 = 1e-6;

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

/// Aggregates the matching data between two faces.
///
/// Each entry stores the corner ranges (on both blocks) and every coincident
/// node that was found for that interface.
#[derive(Clone, Debug, Serialize)]
pub struct FaceMatch {
    pub block1: FaceRecord,
    pub block2: FaceRecord,
    pub points: Vec<MatchPoint>,
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
            println!(
                "match #{idx}: block{} [{},{},{} → {},{},{}] ↔ block{} [{},{},{} → {},{},{}] ({} nodes)",
                m.block1.block_index,
                m.block1.imin,
                m.block1.jmin,
                m.block1.kmin,
                m.block1.imax,
                m.block1.jmax,
                m.block1.kmax,
                m.block2.block_index,
                m.block2.imin,
                m.block2.jmin,
                m.block2.kmin,
                m.block2.imax,
                m.block2.jmax,
                m.block2.kmax,
                m.points.len()
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
    coord: [f64; 3],
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
    target: [f64; 3],
    tol: f64,
) -> Option<&'a FaceNode> {
    let mut best: Option<(&FaceNode, f64)> = None;
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
    tol: f64,
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
    tol: f64,
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

/// Generate block index pairs using centroid distance ordering.
///
/// # Arguments
/// * `blocks` - All blocks in the assembly.
/// * `nearest_nblocks` - Number of closest neighbours to include for each block.
///
/// # Returns
/// Vector of ordered `(i, j)` index pairs.
fn combinations_of_nearest_blocks(blocks: &[Block], nearest_nblocks: usize) -> Vec<(usize, usize)> {
    let centroids: Vec<(f64, f64, f64)> = blocks.iter().map(Block::centroid).collect();
    let mut combos = Vec::new();
    for (i, &ci) in centroids.iter().enumerate() {
        let mut distances: Vec<(usize, f64)> = centroids
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, cj)| {
                let dx = ci.0 - cj.0;
                let dy = ci.1 - cj.1;
                let dz = ci.2 - cj.2;
                (j, (dx * dx + dy * dy + dz * dz).sqrt())
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (j, dist) in distances.into_iter().take(nearest_nblocks) {
            if dist.is_finite() {
                combos.push((i, j));
            }
        }
    }
    combos
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
    let mut gcd_array = Vec::with_capacity(blocks.len());
    for block in blocks {
        let gcd = gcd_three(block.imax - 1, block.jmax - 1, block.kmax - 1);
        gcd_array.push(gcd);
    }
    let gcd_to_use = gcd_array.into_iter().min().unwrap_or(1).max(1);
    let reduced_blocks = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    let (mut matches, mut outer_faces) = connectivity(&reduced_blocks);
    // Scale back to origional size
    for face in &mut matches {
        face.block1.imin *= gcd_to_use;
        face.block1.jmin *= gcd_to_use;
        face.block1.kmin *= gcd_to_use;
        face.block1.imax *= gcd_to_use;
        face.block1.jmax *= gcd_to_use;
        face.block1.kmax *= gcd_to_use;

        face.block2.imin *= gcd_to_use;
        face.block2.jmin *= gcd_to_use;
        face.block2.kmin *= gcd_to_use;
        face.block2.imax *= gcd_to_use;
        face.block2.jmax *= gcd_to_use;
        face.block2.kmax *= gcd_to_use;
    }
    for face in &mut outer_faces {
        face.imin *= gcd_to_use;
        face.jmin *= gcd_to_use;
        face.kmin *= gcd_to_use;
        face.imax *= gcd_to_use;
        face.jmax *= gcd_to_use;
        face.kmax *= gcd_to_use;
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
    let mut block_outer_faces: Vec<Vec<Face>> = blocks
        .iter()
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

    let combos = combinations_of_nearest_blocks(blocks, 6);
    let mut matches = Vec::new();
    let mut matches_to_remove: HashSet<(usize, usize, usize, usize, usize, usize, usize)> =
        HashSet::new();

    for (i, j) in combos {
        if i == j {
            continue;
        }
        let (left, right) = if i < j {
            let (left, right) = block_outer_faces.split_at_mut(j);
            (&mut left[i], &mut right[0])
        } else {
            let (left, right) = block_outer_faces.split_at_mut(i);
            (&mut right[0], &mut left[j])
        };
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
            });
        }
    }

    let mut outer_faces = Vec::new();
    for faces in &block_outer_faces {
        for face in faces {
            outer_faces.push(face.clone());
        }
    }
    let mut seen = HashSet::new();
    outer_faces.retain(|face| seen.insert(face.index_key()));

    outer_faces.retain(|face| !matches_to_remove.contains(&face.index_key()));

    let mut outer_faces_to_remove = HashSet::new();
    let mut by_block: HashMap<usize, Vec<&Face>> = HashMap::new();
    for face in &outer_faces {
        if let Some(idx) = face.block_index() {
            by_block.entry(idx).or_default().push(face);
        }
    }
    for faces in by_block.values() {
        for (a_idx, face_a) in faces.iter().enumerate() {
            let key_a = face_a.index_key();
            for (b_idx, face_b) in faces.iter().enumerate() {
                if a_idx == b_idx {
                    continue;
                }
                let key_b = face_b.index_key();
                let same = key_a.0 == key_b.0
                    && key_a.2 == key_b.2
                    && key_a.3 == key_b.3
                    && key_a.5 == key_b.5
                    && key_a.6 == key_b.6;
                if same {
                    if face_b.diagonal_length() > face_a.diagonal_length() {
                        outer_faces_to_remove.insert(key_b);
                    } else {
                        outer_faces_to_remove.insert(key_a);
                    }
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

/// Greatest common divisor of two integers.
fn gcd_two(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Greatest common divisor of three integers.
fn gcd_three(a: usize, b: usize, c: usize) -> usize {
    gcd_two(gcd_two(a, b), c)
}
