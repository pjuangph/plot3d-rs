use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    block::Block,
    connectivity::{FaceMatch, FaceRecord},
};

const DEFAULT_TOL: f64 = 1e-8;

/// Enumeration describing which index remains constant over a structured face.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FaceAxis {
    /// I-direction is constant.
    I,
    /// J-direction is constant.
    J,
    /// K-direction is constant.
    K,
}

/// Quadrilateral face definition that mimics the Python implementation.
#[derive(Clone, Debug)]
pub struct Face {
    vertices: Vec<[f64; 3]>,
    indices: Vec<[usize; 3]>,
    centroid: [f64; 3],
    block_index: Option<usize>,
    id: Option<usize>,
}

impl Face {
    /// Create an empty face.
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(4),
            indices: Vec::with_capacity(4),
            centroid: [0.0; 3],
            block_index: None,
            id: None,
        }
    }

    /// Add a vertex and update the centroid.
    ///
    /// * `x`, `y`, `z` - Cartesian coordinates.
    /// * `i`, `j`, `k` - Structured-grid indices.
    pub fn add_vertex(&mut self, x: f64, y: f64, z: f64, i: usize, j: usize, k: usize) {
        self.vertices.push([x, y, z]);
        self.indices.push([i, j, k]);
        let n = self.vertices.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for v in &self.vertices {
            cx += v[0];
            cy += v[1];
            cz += v[2];
        }
        self.centroid = [cx / n, cy / n, cz / n];
    }

    /// Set the owning block index.
    pub fn set_block_index(&mut self, idx: usize) {
        self.block_index = Some(idx);
    }

    /// Set the application-defined identifier.
    pub fn set_id(&mut self, id: usize) {
        self.id = Some(id);
    }

    /// Identifier, if one has been assigned.
    pub fn id(&self) -> Option<usize> {
        self.id
    }

    /// Retrieve the centroid.
    pub fn centroid(&self) -> [f64; 3] {
        self.centroid
    }

    /// Owning block index, if present.
    pub fn block_index(&self) -> Option<usize> {
        self.block_index
    }

    /// Iterate over stored vertex indices `(i, j, k)`.
    pub fn indices(&self) -> &[[usize; 3]] {
        &self.indices
    }

    /// All I indices used by this face.
    pub fn i_values(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().map(|ijk| ijk[0])
    }

    /// All J indices used by this face.
    pub fn j_values(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().map(|ijk| ijk[1])
    }

    /// All K indices used by this face.
    pub fn k_values(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().map(|ijk| ijk[2])
    }

    fn min_max(dim: usize, indices: &[[usize; 3]]) -> (usize, usize) {
        let mut min_v = usize::MAX;
        let mut max_v = 0usize;
        for idx in indices {
            min_v = min_v.min(idx[dim]);
            max_v = max_v.max(idx[dim]);
        }
        (min_v, max_v)
    }

    /// Minimum I index among the vertices.
    pub fn imin(&self) -> usize {
        Self::min_max(0, &self.indices).0
    }
    /// Maximum I index among the vertices.
    pub fn imax(&self) -> usize {
        Self::min_max(0, &self.indices).1
    }
    /// Minimum J index among the vertices.
    pub fn jmin(&self) -> usize {
        Self::min_max(1, &self.indices).0
    }
    /// Maximum J index among the vertices.
    pub fn jmax(&self) -> usize {
        Self::min_max(1, &self.indices).1
    }
    /// Minimum K index among the vertices.
    pub fn kmin(&self) -> usize {
        Self::min_max(2, &self.indices).0
    }
    /// Maximum K index among the vertices.
    pub fn kmax(&self) -> usize {
        Self::min_max(2, &self.indices).1
    }

    /// Determine which index is constant, if the face is structured.
    pub fn const_axis(&self) -> Option<FaceAxis> {
        let i_same = self.imin() == self.imax();
        let j_same = self.jmin() == self.jmax();
        let k_same = self.kmin() == self.kmax();
        match (i_same, j_same, k_same) {
            (true, false, false) => Some(FaceAxis::I),
            (false, true, false) => Some(FaceAxis::J),
            (false, false, true) => Some(FaceAxis::K),
            _ => None,
        }
    }

    /// True when the face collapses to an edge.
    pub fn is_edge(&self) -> bool {
        let eq = [
            self.imin() == self.imax(),
            self.jmin() == self.jmax(),
            self.kmin() == self.kmax(),
        ];
        eq.iter().filter(|&&b| b).count() > 1
    }

    /// Compare index ranges with another face.
    pub fn index_equals(&self, other: &Face) -> bool {
        self.imin() == other.imin()
            && self.imax() == other.imax()
            && self.jmin() == other.jmin()
            && self.jmax() == other.jmax()
            && self.kmin() == other.kmin()
            && self.kmax() == other.kmax()
    }

    /// Length of the face diagonal between the extreme corner nodes.
    pub fn diagonal_length(&self) -> f64 {
        fn corner_index(face: &Face, imin: usize, jmin: usize, kmin: usize) -> usize {
            face.indices
                .iter()
                .enumerate()
                .find_map(|(idx, ijk)| {
                    if ijk[0] == imin && ijk[1] == jmin && ijk[2] == kmin {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .unwrap_or(0)
        }

        let min_idx = corner_index(self, self.imin(), self.jmin(), self.kmin());
        let max_idx = corner_index(self, self.imax(), self.jmax(), self.kmax());
        let p0 = self.vertices[min_idx];
        let p1 = self.vertices[max_idx];
        distance(p0, p1)
    }

    /// Compare vertex positions with a tolerance.
    pub fn vertices_equals(&self, other: &Face, tol: f64) -> bool {
        if self.vertices.len() != other.vertices.len() {
            return false;
        }
        let mut matched = vec![false; other.vertices.len()];
        for v in &self.vertices {
            let mut found = false;
            for (idx, o) in other.vertices.iter().enumerate() {
                if matched[idx] {
                    continue;
                }
                // deference and copy the values of v and o
                if distance(*v, *o) <= tol {
                    matched[idx] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }

    /// Structured face points (dense sampling) for node matching.
    ///
    /// * `block` - Parent block.
    /// * `stride_u`, `stride_v` - Sampling strides in parametric space.
    pub fn grid_points(&self, block: &Block, stride_u: usize, stride_v: usize) -> Vec<[f64; 3]> {
        let Some(axis) = self.const_axis() else {
            return self.vertices.clone();
        };
        let su = stride_u.max(1);
        let sv = stride_v.max(1);
        let mut pts = Vec::new();
        match axis {
            FaceAxis::I => {
                let i = self.imin();
                for j in (self.jmin()..=self.jmax()).step_by(su) {
                    for k in (self.kmin()..=self.kmax()).step_by(sv) {
                        pts.push(to_array(block.xyz(i, j, k)));
                    }
                }
            }
            FaceAxis::J => {
                let j = self.jmin();
                for i in (self.imin()..=self.imax()).step_by(su) {
                    for k in (self.kmin()..=self.kmax()).step_by(sv) {
                        pts.push(to_array(block.xyz(i, j, k)));
                    }
                }
            }
            FaceAxis::K => {
                let k = self.kmin();
                for i in (self.imin()..=self.imax()).step_by(su) {
                    for j in (self.jmin()..=self.jmax()).step_by(sv) {
                        pts.push(to_array(block.xyz(i, j, k)));
                    }
                }
            }
        }
        pts
    }

    /// Decide if another face shares enough nodes to be considered touching.
    ///
    /// * `other` - Candidate face.
    /// * `block_self`, `block_other` - Parent blocks.
    /// * `tol_xyz` - Distance tolerance for node equivalence.
    /// * `min_shared_frac` - Minimum fraction of shared nodes.
    /// * `min_shared_abs` - Minimum absolute number of shared nodes.
    /// * `stride_u`, `stride_v` - Sampling stride along the face grid.
    pub fn touches_by_nodes(
        &self,
        other: &Face,
        block_self: &Block,
        block_other: &Block,
        tol_xyz: f64,
        min_shared_frac: f64,
        min_shared_abs: usize,
        stride_u: usize,
        stride_v: usize,
    ) -> bool {
        let pts_self = self.grid_points(block_self, stride_u, stride_v);
        let pts_other = other.grid_points(block_other, stride_u, stride_v);
        if pts_self.is_empty() || pts_other.is_empty() {
            return false;
        }

        let q_self: HashSet<_> = pts_self
            .iter()
            .map(|p| quantize_point(*p, tol_xyz))
            .collect();
        let q_other: HashSet<_> = pts_other
            .iter()
            .map(|p| quantize_point(*p, tol_xyz))
            .collect();

        let shared = q_self.intersection(&q_other).count();
        if shared < min_shared_abs {
            return false;
        }

        let denom = pts_self.len().min(pts_other.len()) as f64;
        (shared as f64) / denom >= min_shared_frac
    }

    /// Export a [`FaceRecord`] representation mirroring the Python dictionary API.
    pub fn to_record(&self) -> FaceRecord {
        FaceRecord {
            block_index: self.block_index.unwrap_or(usize::MAX),
            imin: self.imin(),
            jmin: self.jmin(),
            kmin: self.kmin(),
            imax: self.imax(),
            jmax: self.jmax(),
            kmax: self.kmax(),
            id: self.id,
        }
    }

    pub fn index_key(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        (
            self.block_index.unwrap_or(usize::MAX),
            self.imin(),
            self.jmin(),
            self.kmin(),
            self.imax(),
            self.jmax(),
            self.kmax(),
        )
    }

    /// Scale all stored index values by `factor`.
    pub fn scale_indices(&mut self, factor: usize) {
        if factor <= 1 {
            return;
        }
        for idx in &mut self.indices {
            idx[0] *= factor;
            idx[1] *= factor;
            idx[2] *= factor;
        }
    }
}

/// Helper structure representing a structured face grid.
/// Dense representation of a structured face grid.
#[derive(Clone, Debug)]
pub struct StructuredFace {
    /// Face dimensions `(nu, nv)`.
    pub dims: (usize, usize),
    /// Flattened coordinates stored row-major in `u`.
    pub coords: Vec<[f64; 3]>,
}

impl StructuredFace {
    fn idx(&self, u: usize, v: usize) -> [f64; 3] {
        self.coords[v * self.dims.0 + u]
    }
}

#[derive(Copy, Clone, Debug)]
enum BlockFaceKind {
    IMin,
    IMax,
    JMin,
    JMax,
    KMin,
    KMax,
}

impl BlockFaceKind {
    fn all() -> [Self; 6] {
        [
            Self::IMin,
            Self::IMax,
            Self::JMin,
            Self::JMax,
            Self::KMin,
            Self::KMax,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::IMin => "imin",
            Self::IMax => "imax",
            Self::JMin => "jmin",
            Self::JMax => "jmax",
            Self::KMin => "kmin",
            Self::KMax => "kmax",
        }
    }

    fn dims(self, block: &Block) -> (usize, usize) {
        match self {
            Self::IMin | Self::IMax => (block.jmax, block.kmax),
            Self::JMin | Self::JMax => (block.imax, block.kmax),
            Self::KMin | Self::KMax => (block.imax, block.jmax),
        }
    }

    fn sample(self, block: &Block, u: usize, v: usize) -> [f64; 3] {
        match self {
            Self::IMin => to_array(block.xyz(0, u, v)),
            Self::IMax => to_array(block.xyz(block.imax - 1, u, v)),
            Self::JMin => to_array(block.xyz(u, 0, v)),
            Self::JMax => to_array(block.xyz(u, block.jmax - 1, v)),
            Self::KMin => to_array(block.xyz(u, v, 0)),
            Self::KMax => to_array(block.xyz(u, v, block.kmax - 1)),
        }
    }

    fn structured_face(self, block: &Block) -> StructuredFace {
        let dims = self.dims(block);
        let mut coords = Vec::with_capacity(dims.0 * dims.1);
        for v in 0..dims.1 {
            for u in 0..dims.0 {
                coords.push(self.sample(block, u, v));
            }
        }
        StructuredFace { dims, coords }
    }
}

/// Compute the Euclidean distance between two points.
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn quantize_point(p: [f64; 3], tol: f64) -> (i64, i64, i64) {
    let s = if tol > 0.0 { tol } else { DEFAULT_TOL };
    (
        (p[0] / s).round() as i64,
        (p[1] / s).round() as i64,
        (p[2] / s).round() as i64,
    )
}

/// Convert a tuple `(x, y, z)` into an array `[f64; 3]`.
fn to_array(p: (f64, f64, f64)) -> [f64; 3] {
    [p.0, p.1, p.2]
}

/// Deduplicate index pairs (order-agnostic).
///
/// # Arguments
/// * `pairs` - Candidate index tuples `(a, b)`.
///
/// # Returns
/// Deduplicated list preserving the original ordering of the input.
pub fn unique_pairs(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &(a, b) in pairs {
        if a == b {
            continue;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if seen.insert(key) {
            out.push((a, b));
        }
    }
    out
}

/// Compare two structured faces and determine whether they match.
///
/// # Arguments
/// * `face1` - Face sampled from the first block.
/// * `face2` - Face sampled from the second block.
/// * `tol` - Maximum Euclidean distance allowed between corner nodes.
///
/// # Returns
/// `(matches, flips)` where `flips` encodes `(flip_ud, flip_lr)` applied to `face2`.
pub fn faces_match(
    face1: &StructuredFace,
    face2: &StructuredFace,
    tol: f64,
) -> (bool, Option<(bool, bool)>) {
    if face1.dims != face2.dims {
        return (false, None);
    }
    let (ni, nj) = face1.dims;

    let corners = |f: &StructuredFace, flip_ud: bool, flip_lr: bool| -> [[f64; 3]; 4] {
        let map = |u: usize, v: usize| {
            let uu = if flip_ud { ni - 1 - u } else { u };
            let vv = if flip_lr { nj - 1 - v } else { v };
            f.idx(uu, vv)
        };
        [
            map(0, 0),
            map(0, nj - 1),
            map(ni - 1, 0),
            map(ni - 1, nj - 1),
        ]
    };

    let c1 = corners(face1, false, false);
    for flip_ud in [false, true] {
        for flip_lr in [false, true] {
            let c2 = corners(face2, flip_ud, flip_lr);
            if c1.iter().zip(&c2).all(|(a, b)| distance(*a, *b) <= tol) {
                return (true, Some((flip_ud, flip_lr)));
            }
        }
    }
    (false, None)
}

/// Determine whether any faces on two blocks match.
///
/// # Arguments
/// * `block1` - First block to compare.
/// * `block2` - Second block to compare.
/// * `tol` - Corner matching tolerance.
///
/// # Returns
/// `Some((face_name_block1, face_name_block2, (flip_ud, flip_lr)))` when matching faces are found.
pub fn find_matching_faces(
    block1: &Block,
    block2: &Block,
    tol: f64,
) -> Option<(&'static str, &'static str, (bool, bool))> {
    for f1 in BlockFaceKind::all() {
        let face1 = f1.structured_face(block1);
        for f2 in BlockFaceKind::all() {
            let face2 = f2.structured_face(block2);
            let (matched, flips) = faces_match(&face1, &face2, tol);
            if matched {
                return flips.map(|flip| (f1.name(), f2.name(), flip));
            }
        }
    }
    None
}

/// Build the six outer faces for a block and identify internal matches.
///
/// # Arguments
/// * `block` - Target plot3d block.
///
/// # Returns
/// Tuple containing the exterior faces and any internal matching face pairs.
pub fn get_outer_faces(block: &Block) -> (Vec<Face>, Vec<(Face, Face)>) {
    let mut faces = Vec::with_capacity(6);
    for kind in BlockFaceKind::all() {
        let mut face = Face::new();
        match kind {
            BlockFaceKind::IMin | BlockFaceKind::IMax => {
                let i = if matches!(kind, BlockFaceKind::IMin) {
                    0
                } else {
                    block.imax - 1
                };
                for j in [0, block.jmax - 1] {
                    for k in [0, block.kmax - 1] {
                        let (x, y, z) = block.xyz(i, j, k);
                        face.add_vertex(x, y, z, i, j, k);
                    }
                }
            }
            BlockFaceKind::JMin | BlockFaceKind::JMax => {
                let j = if matches!(kind, BlockFaceKind::JMin) {
                    0
                } else {
                    block.jmax - 1
                };
                for i in [0, block.imax - 1] {
                    for k in [0, block.kmax - 1] {
                        let (x, y, z) = block.xyz(i, j, k);
                        face.add_vertex(x, y, z, i, j, k);
                    }
                }
            }
            BlockFaceKind::KMin | BlockFaceKind::KMax => {
                let k = if matches!(kind, BlockFaceKind::KMin) {
                    0
                } else {
                    block.kmax - 1
                };
                for i in [0, block.imax - 1] {
                    for j in [0, block.jmax - 1] {
                        let (x, y, z) = block.xyz(i, j, k);
                        face.add_vertex(x, y, z, i, j, k);
                    }
                }
            }
        }
        faces.push(face);
    }

    let mut matching_pairs = Vec::new();
    let mut non_matching = Vec::new();
    for i in 0..faces.len() {
        let mut matched = false;
        for j in 0..faces.len() {
            if i == j {
                continue;
            }
            if faces[i].vertices_equals(&faces[j], DEFAULT_TOL) {
                matching_pairs.push((i, j));
                matched = true;
            }
        }
        if !matched {
            non_matching.push(faces[i].clone());
        }
    }

    let pairs = unique_pairs(&matching_pairs)
        .into_iter()
        .map(|(a, b)| (faces[a].clone(), faces[b].clone()))
        .collect();

    (non_matching, pairs)
}

/// Build a face from diagonal index pairs on a block.
///
/// # Arguments
/// * `block` - Parent block.
/// * `imin`, `jmin`, `kmin` - Lower corner indices.
/// * `imax`, `jmax`, `kmax` - Upper corner indices.
///
/// # Returns
/// New `Face` populated with the four corner nodes.
pub fn create_face_from_diagonals(
    block: &Block,
    imin: usize,
    jmin: usize,
    kmin: usize,
    imax: usize,
    jmax: usize,
    kmax: usize,
) -> Face {
    let mut face = Face::new();
    if imin == imax {
        let i = imin;
        for j in [jmin, jmax] {
            for k in [kmin, kmax] {
                let (x, y, z) = block.xyz(i, j, k);
                face.add_vertex(x, y, z, i, j, k);
            }
        }
    } else if jmin == jmax {
        let j = jmin;
        for i in [imin, imax] {
            for k in [kmin, kmax] {
                let (x, y, z) = block.xyz(i, j, k);
                face.add_vertex(x, y, z, i, j, k);
            }
        }
    } else if kmin == kmax {
        let k = kmin;
        for i in [imin, imax] {
            for j in [jmin, jmax] {
                let (x, y, z) = block.xyz(i, j, k);
                face.add_vertex(x, y, z, i, j, k);
            }
        }
    }
    face
}

/// Convert serialized face records back into `Face` instances.
///
/// # Arguments
/// * `blocks` - Blocks interpreted at the reduced resolution.
/// * `outer_faces` - Collection of serialized face records.
/// * `gcd` - Grid reduction factor applied to the blocks.
///
/// # Returns
/// Converted faces with block indices preserved.
pub fn outer_face_records_to_list(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    gcd: usize,
) -> Vec<Face> {
    let mut faces = Vec::new();
    for record in outer_faces {
        let block_idx = record.block_index;
        if block_idx >= blocks.len() {
            continue;
        }
        let block = &blocks[block_idx];
        let scale = gcd.max(1);
        let mut face = create_face_from_diagonals(
            block,
            record.imin / scale,
            record.jmin / scale,
            record.kmin / scale,
            record.imax / scale,
            record.jmax / scale,
            record.kmax / scale,
        );
        face.set_block_index(block_idx);
        if let Some(id) = record.id {
            face.set_id(id);
        }
        faces.push(face);
    }
    faces
}

/// Convert serialized matched faces to a flat `Face` list.
///
/// # Arguments
/// * `blocks` - Blocks interpreted at the reduced resolution.
/// * `matched_faces` - Matched face descriptors describing interfaces.
/// * `gcd` - Grid reduction factor applied to the blocks.
///
/// # Returns
/// Flattened list of faces representing every entry in `matched_faces`.
pub fn match_faces_to_list(blocks: &[Block], matched_faces: &[FaceMatch], gcd: usize) -> Vec<Face> {
    let mut out = Vec::new();
    for record in matched_faces {
        let f1 = outer_face_records_to_list(blocks, &[record.block1.clone()], gcd)
            .into_iter()
            .next();
        let f2 = outer_face_records_to_list(blocks, &[record.block2.clone()], gcd)
            .into_iter()
            .next();
        if let Some(face) = f1 {
            out.push(face);
        }
        if let Some(face) = f2 {
            out.push(face);
        }
    }
    out
}

/// Split a face into subfaces along the specified diagonal indices.
///
/// # Arguments
/// * `face_to_split` - Parent face to subdivide.
/// * `block` - Block providing geometry.
/// * `imin`, `jmin`, `kmin` - Lower split indices.
/// * `imax`, `jmax`, `kmax` - Upper split indices.
///
/// # Returns
/// Collection of child faces excluding edges and the centre face itself.
pub fn split_face(
    face_to_split: &Face,
    block: &Block,
    imin: usize,
    jmin: usize,
    kmin: usize,
    imax: usize,
    jmax: usize,
    kmax: usize,
) -> Vec<Face> {
    let center = create_face_from_diagonals(block, imin, jmin, kmin, imax, jmax, kmax);
    let mut faces = Vec::new();

    if kmin == kmax {
        faces.push(create_face_from_diagonals(
            block,
            imin,
            jmax,
            kmin,
            imax,
            face_to_split.jmax(),
            kmax,
        ));
        faces.push(create_face_from_diagonals(
            block,
            imin,
            face_to_split.jmin(),
            kmin,
            imax,
            jmin,
            kmax,
        ));
        faces.push(create_face_from_diagonals(
            block,
            face_to_split.imin(),
            face_to_split.jmin(),
            kmin,
            imin,
            face_to_split.jmax(),
            kmax,
        ));
        faces.push(create_face_from_diagonals(
            block,
            imax,
            face_to_split.jmin(),
            kmin,
            face_to_split.imax(),
            face_to_split.jmax(),
            kmax,
        ));
    } else if imin == imax {
        faces.push(create_face_from_diagonals(
            block,
            imin,
            jmin,
            kmax,
            imax,
            jmax,
            face_to_split.kmax(),
        ));
        faces.push(create_face_from_diagonals(
            block,
            imin,
            jmin,
            face_to_split.kmin(),
            imax,
            jmax,
            kmin,
        ));
        faces.push(create_face_from_diagonals(
            block,
            imin,
            face_to_split.jmin(),
            face_to_split.kmin(),
            imax,
            jmin,
            face_to_split.kmax(),
        ));
        faces.push(create_face_from_diagonals(
            block,
            imin,
            jmax,
            face_to_split.kmin(),
            imax,
            face_to_split.jmax(),
            face_to_split.kmax(),
        ));
    } else if jmin == jmax {
        faces.push(create_face_from_diagonals(
            block,
            imin,
            jmin,
            kmax,
            imax,
            jmax,
            face_to_split.kmax(),
        ));
        faces.push(create_face_from_diagonals(
            block,
            imin,
            jmin,
            face_to_split.kmin(),
            imax,
            jmax,
            kmin,
        ));
        faces.push(create_face_from_diagonals(
            block,
            face_to_split.imin(),
            jmin,
            face_to_split.kmin(),
            imin,
            jmax,
            face_to_split.kmax(),
        ));
        faces.push(create_face_from_diagonals(
            block,
            imax,
            jmin,
            face_to_split.kmin(),
            face_to_split.imax(),
            jmax,
            face_to_split.kmax(),
        ));
    }

    faces
        .into_iter()
        .filter_map(|mut face| {
            if face.is_edge() || face.index_equals(&center) {
                None
            } else {
                if let Some(idx) = face_to_split.block_index() {
                    face.set_block_index(idx);
                }
                Some(face)
            }
        })
        .collect()
}

/// Pick the face closest to a reference point.
///
/// # Arguments
/// * `faces` - Candidate faces.
/// * `point` - Cartesian reference location.
///
/// # Returns
/// Index of the nearest face or `None` when the list is empty.
pub fn find_face_nearest_point(faces: &[Face], point: [f64; 3]) -> Option<usize> {
    faces
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            distance(a.centroid(), point)
                .partial_cmp(&distance(b.centroid(), point))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

/// Reduce blocks by sampling every `factor` nodes along each axis.
///
/// # Arguments
/// * `blocks` - Blocks to down-sample.
/// * `factor` - Sampling step applied to i, j and k directions.
///
/// # Returns
/// New blocks reduced to consistent spacing.
pub fn reduce_blocks(blocks: &[Block], factor: usize) -> Vec<Block> {
    if factor <= 1 {
        return blocks.to_vec();
    }

    fn sampled_indices(max: usize, stride: usize) -> Vec<usize> {
        if max == 0 {
            return Vec::new();
        }
        let mut indices: Vec<usize> = (0..max).step_by(stride).collect();
        if let Some(&last) = indices.last() {
            if last != max - 1 {
                indices.push(max - 1);
            }
        } else {
            indices.push(max - 1);
        }
        indices
    }

    blocks
        .iter()
        .map(|block| {
            let i_idx = sampled_indices(block.imax, factor);
            let j_idx = sampled_indices(block.jmax, factor);
            let k_idx = sampled_indices(block.kmax, factor);

            let si = i_idx.len();
            let sj = j_idx.len();
            let sk = k_idx.len();

            let mut x = Vec::with_capacity(si * sj * sk);
            let mut y = Vec::with_capacity(si * sj * sk);
            let mut z = Vec::with_capacity(si * sj * sk);

            for &k in &k_idx {
                for &j in &j_idx {
                    for &i in &i_idx {
                        let (px, py, pz) = block.xyz(i, j, k);
                        x.push(px);
                        y.push(py);
                        z.push(pz);
                    }
                }
            }

            Block::new(si, sj, sk, x, y, z)
        })
        .collect()
}

/// Rotate a block using a 3×3 rotation matrix.
///
/// # Arguments
/// * `block` - Block to rotate.
/// * `rotation` - Row-major rotation matrix.
///
/// # Returns
/// Rotated block with identical dimensions.
pub fn rotate_block(block: &Block, rotation: [[f64; 3]; 3]) -> Block {
    let mut x = Vec::with_capacity(block.npoints());
    let mut y = Vec::with_capacity(block.npoints());
    let mut z = Vec::with_capacity(block.npoints());
    for k in 0..block.kmax {
        for j in 0..block.jmax {
            for i in 0..block.imax {
                let (px, py, pz) = block.xyz(i, j, k);
                x.push(rotation[0][0] * px + rotation[0][1] * py + rotation[0][2] * pz);
                y.push(rotation[1][0] * px + rotation[1][1] * py + rotation[1][2] * pz);
                z.push(rotation[2][0] * px + rotation[2][1] * py + rotation[2][2] * pz);
            }
        }
    }
    return Block::new(block.imax, block.jmax, block.kmax, x, y, z);
}

/// Compute the global bounds across all blocks.
///
/// # Arguments
/// * `blocks` - Collection of blocks to inspect.
///
/// # Returns
/// `(x_bounds, y_bounds, z_bounds)` or `None` when the list is empty.
pub fn get_outer_bounds(blocks: &[Block]) -> Option<((f64, f64), (f64, f64), (f64, f64))> {
    if blocks.is_empty() {
        return None;
    }
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut zmin = f64::INFINITY;
    let mut zmax = f64::NEG_INFINITY;
    for block in blocks {
        for &val in block.x_slice() {
            xmin = xmin.min(val);
            xmax = xmax.max(val);
        }
        for &val in block.y_slice() {
            ymin = ymin.min(val);
            ymax = ymax.max(val);
        }
        for &val in block.z_slice() {
            zmin = zmin.min(val);
            zmax = zmax.max(val);
        }
    }
    Some(((xmin, xmax), (ymin, ymax), (zmin, zmax)))
}

/// Options for the block connectivity calculation.
#[derive(Copy, Clone, Debug)]
pub struct BlockConnectionOptions {
    pub node_tol_xyz: f64,
    pub min_shared_frac: f64,
    pub min_shared_abs: usize,
    pub stride_u: usize,
    pub stride_v: usize,
    pub use_area_fallback: bool,
    pub area_min_overlap_frac: f64,
}

impl Default for BlockConnectionOptions {
    fn default() -> Self {
        Self {
            node_tol_xyz: 1e-7,
            min_shared_frac: 0.02,
            min_shared_abs: 4,
            stride_u: 1,
            stride_v: 1,
            use_area_fallback: false,
            area_min_overlap_frac: 0.01,
        }
    }
}

/// Connectivity matrices describing which faces touch between blocks.
///
/// # Arguments
/// * `blocks` - Original block list.
/// * `outer_faces` - Optional pre-computed outer faces (face records).
/// * `tol` - Compatibility parameter maintained for parity with Python (unused).
/// * `options` - Node matching thresholds and sampling strides.
///
/// # Returns
/// Four symmetric adjacency matrices for overall connectivity and each axis-specific match.
pub fn block_connection_matrix(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    tol: f64,
    options: BlockConnectionOptions,
) -> (Vec<Vec<i8>>, Vec<Vec<i8>>, Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let gcd = blocks
        .iter()
        .map(|b| {
            gcd_three(
                b.imax.saturating_sub(1),
                b.jmax.saturating_sub(1),
                b.kmax.saturating_sub(1),
            )
        })
        .filter(|&g| g > 0)
        .min()
        .unwrap_or(1);
    let reduced = reduce_blocks(blocks, gcd);

    let mut faces_by_block: Vec<Vec<Face>> = vec![Vec::new(); blocks.len()];
    if outer_faces.is_empty() {
        for (idx, block) in reduced.iter().enumerate() {
            let (faces, _) = get_outer_faces(block);
            faces_by_block[idx] = faces
                .into_iter()
                .map(|mut f| {
                    f.set_block_index(idx);
                    f
                })
                .collect();
        }
    } else {
        for face in outer_face_records_to_list(&reduced, outer_faces, gcd) {
            if let Some(idx) = face.block_index() {
                if idx < faces_by_block.len() {
                    faces_by_block[idx].push(face);
                }
            }
        }
    }

    let n = blocks.len();
    let mut connectivity = vec![vec![0i8; n]; n];
    let mut conn_i = vec![vec![0i8; n]; n];
    let mut conn_j = vec![vec![0i8; n]; n];
    let mut conn_k = vec![vec![0i8; n]; n];
    for i in 0..n {
        connectivity[i][i] = 1;
        conn_i[i][i] = 1;
        conn_j[i][i] = 1;
        conn_k[i][i] = 1;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let mut connected = false;
            for face_i in &faces_by_block[i] {
                for face_j in &faces_by_block[j] {
                    if face_i.touches_by_nodes(
                        face_j,
                        &reduced[i],
                        &reduced[j],
                        options.node_tol_xyz,
                        options.min_shared_frac,
                        options.min_shared_abs,
                        options.stride_u,
                        options.stride_v,
                    ) {
                        connectivity[i][j] = 1;
                        connectivity[j][i] = 1;
                        if face_i.const_axis() == Some(FaceAxis::I)
                            && face_j.const_axis() == Some(FaceAxis::I)
                        {
                            conn_i[i][j] = 1;
                            conn_i[j][i] = 1;
                        }
                        if face_i.const_axis() == Some(FaceAxis::J)
                            && face_j.const_axis() == Some(FaceAxis::J)
                        {
                            conn_j[i][j] = 1;
                            conn_j[j][i] = 1;
                        }
                        if face_i.const_axis() == Some(FaceAxis::K)
                            && face_j.const_axis() == Some(FaceAxis::K)
                        {
                            conn_k[i][j] = 1;
                            conn_k[j][i] = 1;
                        }
                        connected = true;
                        break;
                    }
                }
                if connected {
                    break;
                }
            }
            if !connected {
                connectivity[i][j] = -1;
                connectivity[j][i] = -1;
            }
        }
    }

    if tol.is_finite() {
        let _ = tol; // placeholder for parity with Python signature
    }

    (connectivity, conn_i, conn_j, conn_k)
}

fn gcd_two(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn gcd_three(a: usize, b: usize, c: usize) -> usize {
    gcd_two(gcd_two(a, b), c)
}

/// Standardise block orientation so that indices increase with coordinate values.
///
/// # Arguments
/// * `block` - Block whose axes may require flipping.
///
/// # Returns
/// New block with consistent orientation.
pub fn standardize_block_orientation(block: &Block) -> Block {
    let mut x = block.x.clone();
    let mut y = block.y.clone();
    let mut z = block.z.clone();
    let dims = (block.imax, block.jmax, block.kmax);

    let center_i = block.imax / 2;
    let center_j = block.jmax / 2;
    let center_k = block.kmax / 2;

    if block.imax > 1 {
        let delta =
            block.x_at(block.imax - 1, center_j, center_k) - block.x_at(0, center_j, center_k);
        if delta < 0.0 {
            flip_block_axis(&mut x, &mut y, &mut z, dims, 0);
        }
    }
    if block.jmax > 1 {
        let delta =
            block.y_at(center_i, block.jmax - 1, center_k) - block.y_at(center_i, 0, center_k);
        if delta < 0.0 {
            flip_block_axis(&mut x, &mut y, &mut z, dims, 1);
        }
    }
    if block.kmax > 1 {
        let delta =
            block.z_at(center_i, center_j, block.kmax - 1) - block.z_at(center_i, center_j, 0);
        if delta < 0.0 {
            flip_block_axis(&mut x, &mut y, &mut z, dims, 2);
        }
    }

    Block::new(block.imax, block.jmax, block.kmax, x, y, z)
}

fn flip_block_axis(
    x: &mut [f64],
    y: &mut [f64],
    z: &mut [f64],
    dims: (usize, usize, usize),
    axis: usize,
) {
    let (imax, jmax, kmax) = dims;
    match axis {
        0 => {
            for k in 0..kmax {
                for j in 0..jmax {
                    for i in 0..imax / 2 {
                        let idx1 = (k * jmax + j) * imax + i;
                        let idx2 = (k * jmax + j) * imax + (imax - 1 - i);
                        x.swap(idx1, idx2);
                        y.swap(idx1, idx2);
                        z.swap(idx1, idx2);
                    }
                }
            }
        }
        1 => {
            for k in 0..kmax {
                for j in 0..jmax / 2 {
                    for i in 0..imax {
                        let idx1 = (k * jmax + j) * imax + i;
                        let idx2 = (k * jmax + (jmax - 1 - j)) * imax + i;
                        x.swap(idx1, idx2);
                        y.swap(idx1, idx2);
                        z.swap(idx1, idx2);
                    }
                }
            }
        }
        2 => {
            for k in 0..kmax / 2 {
                for j in 0..jmax {
                    for i in 0..imax {
                        let idx1 = (k * jmax + j) * imax + i;
                        let idx2 = ((kmax - 1 - k) * jmax + j) * imax + i;
                        x.swap(idx1, idx2);
                        y.swap(idx1, idx2);
                        z.swap(idx1, idx2);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Simple collinearity test using the cross product.
///
/// # Arguments
/// * `v1` - First vector.
/// * `v2` - Second vector.
///
/// # Returns
/// `true` when the vectors are collinear.
pub fn check_collinearity(v1: [f64; 3], v2: [f64; 3]) -> bool {
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    cross.iter().all(|c| c.abs() <= f64::EPSILON)
}

/// Compute outward normals for the six faces of a block.
///
/// # Arguments
/// * `block` - Block whose outward normals are required.
///
/// # Returns
/// Tuple containing normals for `(Imin, Jmin, Kmin, Imax, Jmax, Kmax)`.
pub fn calculate_outward_normals(
    block: &Block,
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let i0 = to_array(block.xyz(0, 0, 0));
    let ij = to_array(block.xyz(0, block.jmax - 1, 0));
    let ik = to_array(block.xyz(0, 0, block.kmax - 1));
    let ni_min = cross(sub(ij, i0), sub(ik, i0));

    let i1 = to_array(block.xyz(block.imax - 1, 0, 0));
    let i1j = to_array(block.xyz(block.imax - 1, block.jmax - 1, 0));
    let i1k = to_array(block.xyz(block.imax - 1, 0, block.kmax - 1));
    let ni_max = cross(sub(i1j, i1), sub(i1k, i1));

    let j0 = to_array(block.xyz(0, 0, 0));
    let ji = to_array(block.xyz(block.imax - 1, 0, 0));
    let jk = to_array(block.xyz(0, 0, block.kmax - 1));
    let nj_min = cross(sub(ji, j0), sub(jk, j0));

    let j1 = to_array(block.xyz(0, block.jmax - 1, 0));
    let j1i = to_array(block.xyz(block.imax - 1, block.jmax - 1, 0));
    let j1k = to_array(block.xyz(0, block.jmax - 1, block.kmax - 1));
    let nj_max = cross(sub(j1i, j1), sub(j1k, j1));

    let k0 = to_array(block.xyz(0, 0, 0));
    let ki = to_array(block.xyz(block.imax - 1, 0, 0));
    let kj = to_array(block.xyz(0, block.jmax - 1, 0));
    let nk_min = cross(sub(ki, k0), sub(kj, k0));

    let k1 = to_array(block.xyz(0, 0, block.kmax - 1));
    let k1i = to_array(block.xyz(block.imax - 1, 0, block.kmax - 1));
    let k1j = to_array(block.xyz(0, block.jmax - 1, block.kmax - 1));
    let nk_max = cross(sub(k1i, k1), sub(k1j, k1));

    (ni_min, nj_min, nk_min, ni_max, nj_max, nk_max)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Identify outer faces on the extreme of the requested axis using BFS.
///
/// # Arguments
/// * `blocks` - All blocks in the system.
/// * `outer_faces_dicts` - Optional pre-computed outer faces in dictionary form.
/// * `direction` - Axis name (`"x"`, `"y"`, or `"z"`).
/// * `side` - Requested side (`"min"`, `"max"`, or `"both"`).
/// * `tol_rel` - Relative tolerance for plane selection.
/// * `node_tol_xyz` - Node matching tolerance for BFS linking.
///
/// # Returns
/// Tuple containing serialized faces and raw face objects for the lower and upper planes.
pub fn find_bounding_faces(
    blocks: &[Block],
    outer_faces_records: &[FaceRecord],
    direction: &str,
    side: &str,
    tol_rel: f64,
    node_tol_xyz: f64,
) -> (Vec<FaceRecord>, Vec<FaceRecord>, Vec<Face>, Vec<Face>) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let axis = match direction {
        "x" => FaceAxis::I,
        "y" => FaceAxis::J,
        _ => FaceAxis::K,
    };
    let want_min = side == "min" || side == "both";
    let want_max = side == "max" || side == "both";

    let gcd = blocks
        .iter()
        .map(|b| gcd_three(b.imax - 1, b.jmax - 1, b.kmax - 1))
        .filter(|&g| g > 0)
        .min()
        .unwrap_or(1);
    let reduced = reduce_blocks(blocks, gcd);

    let outer_faces = if outer_faces_records.is_empty() {
        reduced
            .iter()
            .enumerate()
            .flat_map(|(idx, block)| {
                let (faces, _) = get_outer_faces(block);
                faces.into_iter().map(move |mut f| {
                    f.set_block_index(idx);
                    f
                })
            })
            .collect::<Vec<_>>()
    } else {
        outer_face_records_to_list(&reduced, outer_faces_records, gcd)
    };

    let axis_range = global_axis_bounds(&reduced, axis).unwrap_or((0.0, 0.0));
    let tol_abs = tol_rel * (axis_range.0.abs() + axis_range.1.abs()).max(1.0);

    let mut lower = Vec::new();
    let mut upper = Vec::new();

    if want_min {
        lower = collect_boundary_faces(&outer_faces, &reduced, axis, true, tol_abs, node_tol_xyz);
    }
    if want_max {
        upper = collect_boundary_faces(&outer_faces, &reduced, axis, false, tol_abs, node_tol_xyz);
    }

    let lower_export = lower.iter().map(Face::to_record).collect();
    let upper_export = upper.iter().map(Face::to_record).collect();
    (lower_export, upper_export, lower, upper)
}

fn global_axis_bounds(blocks: &[Block], axis: FaceAxis) -> Option<(f64, f64)> {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for block in blocks {
        match axis {
            FaceAxis::I => {
                for &x in block.x_slice() {
                    min_val = min_val.min(x);
                    max_val = max_val.max(x);
                }
            }
            FaceAxis::J => {
                for &y in block.y_slice() {
                    min_val = min_val.min(y);
                    max_val = max_val.max(y);
                }
            }
            FaceAxis::K => {
                for &z in block.z_slice() {
                    min_val = min_val.min(z);
                    max_val = max_val.max(z);
                }
            }
        }
    }
    if min_val.is_finite() && max_val.is_finite() {
        Some((min_val, max_val))
    } else {
        None
    }
}

fn collect_boundary_faces(
    faces: &[Face],
    blocks: &[Block],
    axis: FaceAxis,
    is_min: bool,
    tol_abs: f64,
    node_tol_xyz: f64,
) -> Vec<Face> {
    if faces.is_empty() {
        return Vec::new();
    }

    let mut plane_value = if is_min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    for face in faces {
        for v in &face.vertices {
            let val = match axis {
                FaceAxis::I => v[0],
                FaceAxis::J => v[1],
                FaceAxis::K => v[2],
            };
            if is_min {
                plane_value = plane_value.min(val);
            } else {
                plane_value = plane_value.max(val);
            }
        }
    }
    if !plane_value.is_finite() {
        return Vec::new();
    }

    let mut plane_faces = Vec::new();
    for face in faces {
        let mut fmin = f64::INFINITY;
        let mut fmax = f64::NEG_INFINITY;
        for v in &face.vertices {
            let val = match axis {
                FaceAxis::I => v[0],
                FaceAxis::J => v[1],
                FaceAxis::K => v[2],
            };
            fmin = fmin.min(val);
            fmax = fmax.max(val);
        }
        let touches_plane = if is_min {
            (fmin - plane_value).abs() <= tol_abs
        } else {
            (fmax - plane_value).abs() <= tol_abs
        };
        let not_past = if is_min {
            (fmax - plane_value) <= tol_abs
        } else {
            (plane_value - fmin) <= tol_abs
        };
        if touches_plane && not_past {
            plane_faces.push(face.clone());
        }
    }

    let mut visited: HashSet<(usize, usize, usize, usize, usize, usize, usize)> = HashSet::new();
    let mut result = Vec::new();
    for seed in &plane_faces {
        let mut queue = VecDeque::new();
        queue.push_back(seed.clone());
        while let Some(face) = queue.pop_front() {
            let key = face.index_key();
            if !visited.insert(key) {
                continue;
            }
            result.push(face.clone());
            for candidate in &plane_faces {
                let cand_key = candidate.index_key();
                if visited.contains(&cand_key) {
                    continue;
                }
                let Some(a_idx) = face.block_index() else {
                    continue;
                };
                let Some(b_idx) = candidate.block_index() else {
                    continue;
                };
                if a_idx >= blocks.len() || b_idx >= blocks.len() {
                    continue;
                }
                if face.touches_by_nodes(
                    candidate,
                    &blocks[a_idx],
                    &blocks[b_idx],
                    node_tol_xyz,
                    0.02,
                    2,
                    1,
                    1,
                ) {
                    queue.push_back(candidate.clone());
                }
            }
        }
    }
    result
}

/// Find the block whose centroid is closest to an extrapolated target.
///
/// # Arguments
/// * `blocks` - Candidate blocks.
/// * `centroid` - Reference centroid for the entire assembly.
/// * `direction` - Axis name controlling the search direction.
/// * `minvalue` - When `true`, search toward the minimum extreme; otherwise the maximum.
///
/// # Returns
/// The selected block index and the target coordinates used for the comparison.
pub fn find_closest_block(
    blocks: &[Block],
    centroid: [f64; 3],
    direction: &str,
    minvalue: bool,
) -> Option<(usize, f64, f64, f64)> {
    let Some((xbounds, ybounds, zbounds)) = get_outer_bounds(blocks) else {
        return None;
    };
    let (target_x, target_y, target_z) = match direction {
        "x" => {
            let dx = xbounds.1 - xbounds.0;
            let x = if minvalue {
                xbounds.0 - 0.5 * dx
            } else {
                xbounds.1 + 0.5 * dx
            };
            (x, centroid[1], centroid[2])
        }
        "y" => {
            let dy = ybounds.1 - ybounds.0;
            let y = if minvalue {
                ybounds.0 - 0.5 * dy
            } else {
                ybounds.1 + 0.5 * dy
            };
            (centroid[0], y, centroid[2])
        }
        _ => {
            let dz = zbounds.1 - zbounds.0;
            let z = if minvalue {
                zbounds.0 - 0.5 * dz
            } else {
                zbounds.1 + 0.5 * dz
            };
            (centroid[0], centroid[1], z)
        }
    };
    let mut best_idx = None;
    let mut best_dist = f64::INFINITY;
    for (idx, block) in blocks.iter().enumerate() {
        let cx = block.x_slice().iter().sum::<f64>() / block.x_slice().len() as f64;
        let cy = block.y_slice().iter().sum::<f64>() / block.y_slice().len() as f64;
        let cz = block.z_slice().iter().sum::<f64>() / block.z_slice().len() as f64;
        let dist = distance([cx, cy, cz], [target_x, target_y, target_z]);
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx.map(|idx| (idx, target_x, target_y, target_z))
}

/// Graph helper: find a neighbour connected to both `a` and `b`.
///
/// # Arguments
/// * `graph` - Adjacency map.
/// * `a` - First node.
/// * `b` - Second node.
/// * `exclude` - Nodes that must not be returned.
///
/// # Returns
/// `Some(node)` when a mutual neighbour exists.
pub fn common_neighbor(
    graph: &HashMap<usize, HashSet<usize>>,
    a: usize,
    b: usize,
    exclude: &HashSet<usize>,
) -> Option<usize> {
    graph
        .get(&a)?
        .iter()
        .find(|&&n| {
            n != b && !exclude.contains(&n) && graph.get(&n).map_or(false, |s| s.contains(&b))
        })
        .copied()
}

/// Convert face matches into an adjacency map.
///
/// # Arguments
/// * `connectivities` - Matched face descriptions (block1/block2 pairings).
///
/// # Returns
/// Undirected adjacency map between block indices.
pub fn build_connectivity_graph(connectivities: &[FaceMatch]) -> HashMap<usize, HashSet<usize>> {
    let mut graph: HashMap<usize, HashSet<usize>> = HashMap::new();
    for pair in connectivities {
        let block1 = pair.block1.block_index;
        let block2 = pair.block2.block_index;
        if block1 == usize::MAX || block2 == usize::MAX {
            continue;
        }
        graph.entry(block1).or_default().insert(block2);
        graph.entry(block2).or_default().insert(block1);
    }
    graph
}
