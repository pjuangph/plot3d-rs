use std::collections::HashSet;

use crate::{
    block::Block,
    face_record::{FaceKey, FaceMatch, FaceRecord},
    geometry::{
        clip_sutherland_hodgman, distance, dominant_projection_axis, poly_area_2d,
        project_drop_axis, quad_normal_from_verts, quantize_point, to_array, vertex_aabb,
    },
    utils::{cross3, dot3, sub3, vec_norm3},
    Float,
};

const DEFAULT_TOL: Float = 1e-8;

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
    vertices: Vec<[Float; 3]>,
    indices: Vec<[usize; 3]>,
    centroid: [Float; 3],
    block_index: Option<usize>,
    id: Option<usize>,
}

/// Find the linear index of the vertex whose structured indices match `(i, j, k)`.
fn corner_index(face: &Face, i: usize, j: usize, k: usize) -> usize {
    face.indices
        .iter()
        .enumerate()
        .find_map(|(idx, ijk)| {
            if ijk[0] == i && ijk[1] == j && ijk[2] == k {
                Some(idx)
            } else {
                None
            }
        })
        .unwrap_or(0)
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
    pub fn add_vertex(&mut self, x: Float, y: Float, z: Float, i: usize, j: usize, k: usize) {
        self.vertices.push([x, y, z]);
        self.indices.push([i, j, k]);
        let n = self.vertices.len() as Float;
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
    pub fn centroid(&self) -> [Float; 3] {
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

    /// Integer constant-type matching the Python convention:
    /// `0` = I-constant, `1` = J-constant, `2` = K-constant, `-1` = none.
    pub fn const_type(&self) -> i8 {
        match self.const_axis() {
            Some(FaceAxis::I) => 0,
            Some(FaceAxis::J) => 1,
            Some(FaceAxis::K) => 2,
            None => -1,
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

    /// Read-only access to the stored vertex coordinates.
    pub fn vertices(&self) -> &[[Float; 3]] {
        &self.vertices
    }

    /// Return the spatial coordinates of the two diagonal corners.
    ///
    /// The "lower" corner is the vertex at `(IMIN, JMIN, KMIN)` and the
    /// "upper" corner is at `(IMAX, JMAX, KMAX)`.
    ///
    /// Returns `None` when the face has no vertices.
    pub fn get_corners(&self) -> Option<([Float; 3], [Float; 3])> {
        if self.vertices.is_empty() {
            return None;
        }
        let min_idx = corner_index(self, self.imin(), self.jmin(), self.kmin());
        let max_idx = corner_index(self, self.imax(), self.jmax(), self.kmax());
        Some((self.vertices[min_idx], self.vertices[max_idx]))
    }

    /// Return all four corner vertices in canonical parametric order.
    ///
    /// The ordering matches `create_face_from_diagonals()`:
    ///   - I-constant: `[(i, jmin, kmin), (i, jmin, kmax), (i, jmax, kmin), (i, jmax, kmax)]`
    ///   - J-constant: `[(imin, j, kmin), (imin, j, kmax), (imax, j, kmin), (imax, j, kmax)]`
    ///   - K-constant: `[(imin, jmin, k), (imin, jmax, k), (imax, jmin, k), (imax, jmax, k)]`
    ///
    /// Returns `None` if the face has fewer than 4 vertices or no constant axis.
    pub fn get_all_corners(&self) -> Option<[[Float; 3]; 4]> {
        if self.vertices.len() < 4 {
            return None;
        }
        let axis = self.const_axis()?;
        let (imin, imax) = (self.imin(), self.imax());
        let (jmin, jmax) = (self.jmin(), self.jmax());
        let (kmin, kmax) = (self.kmin(), self.kmax());

        let find = |ti: usize, tj: usize, tk: usize| -> Option<[Float; 3]> {
            self.indices.iter().enumerate().find_map(|(idx, ijk)| {
                if ijk[0] == ti && ijk[1] == tj && ijk[2] == tk {
                    Some(self.vertices[idx])
                } else {
                    None
                }
            })
        };

        match axis {
            FaceAxis::I => {
                let ic = imin; // I is constant
                Some([
                    find(ic, jmin, kmin)?,
                    find(ic, jmin, kmax)?,
                    find(ic, jmax, kmin)?,
                    find(ic, jmax, kmax)?,
                ])
            }
            FaceAxis::J => {
                let jc = jmin; // J is constant
                Some([
                    find(imin, jc, kmin)?,
                    find(imin, jc, kmax)?,
                    find(imax, jc, kmin)?,
                    find(imax, jc, kmax)?,
                ])
            }
            FaceAxis::K => {
                let kc = kmin; // K is constant
                Some([
                    find(imin, jmin, kc)?,
                    find(imin, jmax, kc)?,
                    find(imax, jmin, kc)?,
                    find(imax, jmax, kc)?,
                ])
            }
        }
    }

    /// Length of the face diagonal between the extreme corner nodes.
    pub fn diagonal_length(&self) -> Float {
        let min_idx = corner_index(self, self.imin(), self.jmin(), self.kmin());
        let max_idx = corner_index(self, self.imax(), self.jmax(), self.kmax());
        let p0 = self.vertices[min_idx];
        let p1 = self.vertices[max_idx];
        distance(p0, p1)
    }

    /// Median edge spacing of the face grid using the block's coordinates.
    ///
    /// Walks adjacent grid nodes along both parametric directions and returns
    /// the median of all edge lengths. Falls back to `1.0` for degenerate faces.
    pub fn median_edge_spacing(&self, block: &Block) -> Float {
        let axis = match self.const_axis() {
            Some(a) => a,
            None => return 1.0,
        };
        let mut spacings = Vec::new();
        match axis {
            FaceAxis::I => {
                let ic = self.imin();
                for j in self.jmin()..self.jmax() {
                    for k in self.kmin()..=self.kmax() {
                        if j + 1 <= self.jmax()
                            && ic < block.imax
                            && j + 1 < block.jmax
                            && k < block.kmax
                        {
                            let (x0, y0, z0) = block.xyz(ic, j, k);
                            let (x1, y1, z1) = block.xyz(ic, j + 1, k);
                            spacings.push(
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt(),
                            );
                        }
                    }
                }
                for k in self.kmin()..self.kmax() {
                    for j in self.jmin()..=self.jmax() {
                        if k + 1 <= self.kmax()
                            && ic < block.imax
                            && j < block.jmax
                            && k + 1 < block.kmax
                        {
                            let (x0, y0, z0) = block.xyz(ic, j, k);
                            let (x1, y1, z1) = block.xyz(ic, j, k + 1);
                            spacings.push(
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt(),
                            );
                        }
                    }
                }
            }
            FaceAxis::J => {
                let jc = self.jmin();
                for i in self.imin()..self.imax() {
                    for k in self.kmin()..=self.kmax() {
                        if i + 1 <= self.imax()
                            && i + 1 < block.imax
                            && jc < block.jmax
                            && k < block.kmax
                        {
                            let (x0, y0, z0) = block.xyz(i, jc, k);
                            let (x1, y1, z1) = block.xyz(i + 1, jc, k);
                            spacings.push(
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt(),
                            );
                        }
                    }
                }
                for k in self.kmin()..self.kmax() {
                    for i in self.imin()..=self.imax() {
                        if k + 1 <= self.kmax()
                            && i < block.imax
                            && jc < block.jmax
                            && k + 1 < block.kmax
                        {
                            let (x0, y0, z0) = block.xyz(i, jc, k);
                            let (x1, y1, z1) = block.xyz(i, jc, k + 1);
                            spacings.push(
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt(),
                            );
                        }
                    }
                }
            }
            FaceAxis::K => {
                let kc = self.kmin();
                for i in self.imin()..self.imax() {
                    for j in self.jmin()..=self.jmax() {
                        if i + 1 <= self.imax()
                            && i + 1 < block.imax
                            && j < block.jmax
                            && kc < block.kmax
                        {
                            let (x0, y0, z0) = block.xyz(i, j, kc);
                            let (x1, y1, z1) = block.xyz(i + 1, j, kc);
                            spacings.push(
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt(),
                            );
                        }
                    }
                }
                for j in self.jmin()..self.jmax() {
                    for i in self.imin()..=self.imax() {
                        if j + 1 <= self.jmax()
                            && i < block.imax
                            && j + 1 < block.jmax
                            && kc < block.kmax
                        {
                            let (x0, y0, z0) = block.xyz(i, j, kc);
                            let (x1, y1, z1) = block.xyz(i, j + 1, kc);
                            spacings.push(
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt(),
                            );
                        }
                    }
                }
            }
        }
        if spacings.is_empty() {
            return 1.0;
        }
        spacings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        spacings[spacings.len() / 2]
    }

    /// Compare vertex positions with a tolerance.
    pub fn vertices_equals(&self, other: &Face, tol: Float) -> bool {
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
    pub fn grid_points(&self, block: &Block, stride_u: usize, stride_v: usize) -> Vec<[Float; 3]> {
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
        tol_xyz: Float,
        min_shared_frac: Float,
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

        let denom = pts_self.len().min(pts_other.len()) as Float;
        (shared as Float) / denom >= min_shared_frac
    }

    /// Export a [`FaceRecord`] representation mirroring the Python dictionary API.
    pub fn to_record(&self) -> FaceRecord {
        FaceRecord {
            block_index: self.block_index.unwrap_or(usize::MAX),
            il: self.imin(),
            jl: self.jmin(),
            kl: self.kmin(),
            ih: self.imax(),
            jh: self.jmax(),
            kh: self.kmax(),
            id: self.id,
            u_physical: None,
            v_physical: None,
        }
    }

    pub fn index_key(&self) -> FaceKey {
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

    /// True when the face collapses to a single point (all three indices constant).
    pub fn is_point(&self) -> bool {
        self.imin() == self.imax() && self.jmin() == self.jmax() && self.kmin() == self.kmax()
    }

    /// Return parametric face size in index-space cells.
    ///
    /// For a face with one constant axis this returns the product of the two
    /// varying dimension ranges. For a degenerate or 3D region, returns the
    /// product of all three ranges.
    pub fn face_size(&self) -> usize {
        let di = self.imax().saturating_sub(self.imin()).max(1);
        let dj = self.jmax().saturating_sub(self.jmin()).max(1);
        let dk = self.kmax().saturating_sub(self.kmin()).max(1);
        if self.imin() == self.imax() {
            dj * dk
        } else if self.jmin() == self.jmax() {
            di * dk
        } else if self.kmin() == self.kmax() {
            di * dj
        } else {
            di * dj * dk
        }
    }

    /// Compute the (unnormalized) geometric normal using three corner points
    /// from the parent block, based on which axis is constant.
    pub fn normal(&self, block: &Block) -> [Float; 3] {
        let axis = match self.const_axis() {
            Some(a) => a,
            None => return [0.0, 0.0, 0.0],
        };

        let (p1, p2, p3) = match axis {
            FaceAxis::I => {
                let ic = self.imin();
                (
                    to_array(block.xyz(ic, self.jmin(), self.kmin())),
                    to_array(block.xyz(ic, self.jmax(), self.kmin())),
                    to_array(block.xyz(ic, self.jmin(), self.kmax())),
                )
            }
            FaceAxis::J => {
                let jc = self.jmin();
                (
                    to_array(block.xyz(self.imin(), jc, self.kmin())),
                    to_array(block.xyz(self.imax(), jc, self.kmin())),
                    to_array(block.xyz(self.imin(), jc, self.kmax())),
                )
            }
            FaceAxis::K => {
                let kc = self.kmin();
                (
                    to_array(block.xyz(self.imin(), self.jmin(), kc)),
                    to_array(block.xyz(self.imax(), self.jmin(), kc)),
                    to_array(block.xyz(self.imin(), self.jmax(), kc)),
                )
            }
        };

        cross3(sub3(p2, p1), sub3(p3, p1))
    }

    /// Shift all stored vertices by `(dx, dy, dz)` in place.
    pub fn shift(&mut self, dx: Float, dy: Float, dz: Float) {
        for v in &mut self.vertices {
            v[0] += dx;
            v[1] += dy;
            v[2] += dz;
        }
        self.centroid[0] += dx;
        self.centroid[1] += dy;
        self.centroid[2] += dz;
    }

    /// Find vertex-index correspondences between `self` and `other`.
    ///
    /// Returns pairs `[i_self, j_other]` where vertex `i_self` of this face
    /// matches vertex `j_other` of `other` within tolerance 1e-6.
    pub fn match_indices(&self, other: &Face) -> Vec<[usize; 2]> {
        let tol = 1e-6;
        let mut matched_other = vec![false; other.vertices.len()];
        let mut result = Vec::new();
        for (i, v_self) in self.vertices.iter().enumerate() {
            for (j, v_other) in other.vertices.iter().enumerate() {
                if matched_other[j] {
                    continue;
                }
                if (v_self[0] - v_other[0]).abs() < tol
                    && (v_self[1] - v_other[1]).abs() < tol
                    && (v_self[2] - v_other[2]).abs() < tol
                {
                    result.push([i, j]);
                    matched_other[j] = true;
                    break;
                }
            }
        }
        result
    }

    /// Compute the overlap area fraction between this face and `other` using
    /// Sutherland-Hodgman polygon clipping.
    ///
    /// Returns `intersection_area / min(area_self, area_other)`.
    /// Returns 0.0 if normals are not (anti-)parallel within `tol_angle_deg`
    /// or if the faces are not coplanar within `tol_plane_dist`.
    pub fn overlap_fraction(
        &self,
        other: &Face,
        tol_angle_deg: Float,
        tol_plane_dist: Float,
    ) -> Float {
        if self.vertices.len() < 3 || other.vertices.len() < 3 {
            return 0.0;
        }

        // AABB prefilter: quick rejection if bounding boxes don't overlap
        let (s_min, s_max) = vertex_aabb(&self.vertices);
        let (o_min, o_max) = vertex_aabb(&other.vertices);
        let diag = self
            .diagonal_length()
            .max(other.diagonal_length())
            .max(1e-12);
        let aabb_tol = tol_plane_dist * diag;
        if s_max[0] + aabb_tol < o_min[0]
            || o_max[0] + aabb_tol < s_min[0]
            || s_max[1] + aabb_tol < o_min[1]
            || o_max[1] + aabb_tol < s_min[1]
            || s_max[2] + aabb_tol < o_min[2]
            || o_max[2] + aabb_tol < s_min[2]
        {
            return 0.0;
        }

        let n1 = quad_normal_from_verts(&self.vertices);
        let n2 = quad_normal_from_verts(&other.vertices);
        let len1 = vec_norm3(n1);
        let len2 = vec_norm3(n2);
        if len1 < 1e-15 || len2 < 1e-15 {
            return 0.0;
        }

        // Check normal parallelism (allow anti-parallel)
        let cos_angle = dot3(n1, n2) / (len1 * len2);
        let angle_deg = cos_angle.abs().min(1.0).acos().to_degrees();
        if angle_deg > tol_angle_deg {
            return 0.0;
        }

        // Check coplanarity: all vertices of other within tol of self's plane
        let p0 = self.vertices[0];
        let n_hat = [n1[0] / len1, n1[1] / len1, n1[2] / len1];
        // Adaptive tolerance: scale by face diagonal if needed
        let diag = self
            .diagonal_length()
            .max(other.diagonal_length())
            .max(1e-12);
        let plane_tol = tol_plane_dist * diag;
        for v in &other.vertices {
            let d = dot3(sub3(*v, p0), n_hat).abs();
            if d > plane_tol {
                return 0.0;
            }
        }

        // Project to 2D
        let drop = dominant_projection_axis(n_hat);
        let poly_self = project_drop_axis(&self.vertices, drop);
        let poly_other = project_drop_axis(&other.vertices, drop);

        let area_self = poly_area_2d(&poly_self).abs();
        let area_other = poly_area_2d(&poly_other).abs();
        if area_self < 1e-30 || area_other < 1e-30 {
            return 0.0;
        }

        let clipped = clip_sutherland_hodgman(&poly_other, &poly_self);
        if clipped.len() < 3 {
            return 0.0;
        }

        let area_inter = poly_area_2d(&clipped).abs();
        area_inter / area_self.min(area_other)
    }

    /// Returns `true` if `overlap_fraction >= min_overlap_frac`.
    pub fn touches(
        &self,
        other: &Face,
        tol_angle_deg: Float,
        tol_plane_dist: Float,
        min_overlap_frac: Float,
    ) -> bool {
        self.overlap_fraction(other, tol_angle_deg, tol_plane_dist) >= min_overlap_frac
    }

    /// Fraction of shared grid nodes between this face and `other`.
    ///
    /// Uses quantized point comparison for robustness.
    pub fn shared_point_fraction(
        &self,
        other: &Face,
        block_self: &Block,
        block_other: &Block,
        tol_xyz: Float,
        stride_u: usize,
        stride_v: usize,
    ) -> Float {
        let pts_self = self.grid_points(block_self, stride_u, stride_v);
        let pts_other = other.grid_points(block_other, stride_u, stride_v);
        if pts_self.is_empty() || pts_other.is_empty() {
            return 0.0;
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
        let denom = pts_self.len().min(pts_other.len()) as Float;
        if denom == 0.0 {
            return 0.0;
        }
        (shared as Float) / denom
    }
}

/// Helper structure representing a structured face grid.
/// Dense representation of a structured face grid.
#[derive(Clone, Debug)]
pub struct StructuredFace {
    /// Face dimensions `(nu, nv)`.
    pub dims: (usize, usize),
    /// Flattened coordinates stored row-major in `u`.
    pub coords: Vec<[Float; 3]>,
}

impl StructuredFace {
    fn idx(&self, u: usize, v: usize) -> [Float; 3] {
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

    fn sample(self, block: &Block, u: usize, v: usize) -> [Float; 3] {
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

// Geometry functions (distance, quantize_point, to_array, quad_normal_from_verts,
// vertex_aabb, dominant_projection_axis, project_drop_axis, poly_area_2d,
// clip_sutherland_hodgman) are in crate::geometry.

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
    tol: Float,
) -> (bool, Option<(bool, bool)>) {
    if face1.dims != face2.dims {
        return (false, None);
    }
    let (ni, nj) = face1.dims;

    let corners = |f: &StructuredFace, flip_ud: bool, flip_lr: bool| -> [[Float; 3]; 4] {
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

/// Attempt a full (1:1) face match by comparing the 4 corner vertices.
///
/// Tries all 8 valid orientations (4 flip combinations + swap) of face_b's
/// corners against face_a's canonical corner ordering.  A match means all 4
/// corners are within `tol` of each other.
///
/// # Returns
/// `Some(Orientation)` describing how face_b's indices map to face_a's,
/// or `None` if no valid corner mapping exists within tolerance.
pub fn full_face_match(
    face_a: &Face,
    face_b: &Face,
    tol: Float,
) -> Option<crate::face_record::Orientation> {
    let corners_a = face_a.get_all_corners()?;
    let corners_b = face_b.get_all_corners()?;
    try_corner_permutations(&corners_a, &corners_b, tol)
}

/// Like [`full_face_match`] but applies a coordinate transformation to
/// face_a's corners before comparing.
///
/// `transform` maps `[Float; 3] -> [Float; 3]`, typically a rotation or
/// translation.
pub fn full_face_match_transformed<F>(
    face_a: &Face,
    face_b: &Face,
    transform: F,
    tol: Float,
) -> Option<crate::face_record::Orientation>
where
    F: Fn([Float; 3]) -> [Float; 3],
{
    let raw = face_a.get_all_corners()?;
    let corners_a = [
        transform(raw[0]),
        transform(raw[1]),
        transform(raw[2]),
        transform(raw[3]),
    ];
    let corners_b = face_b.get_all_corners()?;
    try_corner_permutations(&corners_a, &corners_b, tol)
}

/// Core logic: try all 8 orientation permutations of `cb` against `ca`.
///
/// The canonical corner ordering is:
///   `[0] = (u_min, v_min)`
///   `[1] = (u_min, v_max)`
///   `[2] = (u_max, v_min)`
///   `[3] = (u_max, v_max)`
fn try_corner_permutations(
    ca: &[[Float; 3]; 4],
    cb: &[[Float; 3]; 4],
    tol: Float,
) -> Option<crate::face_record::Orientation> {
    use crate::face_record::Orientation;

    // Each entry: (index permutation of cb, u_reversed, v_reversed, swapped)
    const PERMS: [([usize; 4], bool, bool, bool); 8] = [
        ([0, 1, 2, 3], false, false, false), // identity
        ([2, 3, 0, 1], true, false, false),  // u_reversed
        ([1, 0, 3, 2], false, true, false),  // v_reversed
        ([3, 2, 1, 0], true, true, false),   // both reversed
        ([0, 2, 1, 3], false, false, true),  // swapped
        ([2, 0, 3, 1], true, false, true),   // swapped + u_reversed
        ([1, 3, 0, 2], false, true, true),   // swapped + v_reversed
        ([3, 1, 2, 0], true, true, true),    // swapped + both
    ];

    for &(ref perm, u_rev, v_rev, swapped) in &PERMS {
        let all_match = perm
            .iter()
            .enumerate()
            .all(|(i, &j)| distance(ca[i], cb[j]) <= tol);
        if all_match {
            return Some(Orientation {
                u_reversed: u_rev,
                v_reversed: v_rev,
                swapped,
            });
        }
    }

    None
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
    tol: Float,
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
        let (si, sj, sk) = (
            record.i_lo() / scale,
            record.j_lo() / scale,
            record.k_lo() / scale,
        );
        let (ei, ej, ek) = (
            record.i_hi() / scale,
            record.j_hi() / scale,
            record.k_hi() / scale,
        );
        // Skip records whose scaled indices exceed the reduced block dimensions
        if ei >= block.imax || ej >= block.jmax || ek >= block.kmax {
            continue;
        }
        let mut face = create_face_from_diagonals(block, si, sj, sk, ei, ej, ek);
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
pub fn find_face_nearest_point(faces: &[Face], point: [Float; 3]) -> Option<usize> {
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
pub fn rotate_block(block: &Block, rotation: [[Float; 3]; 3]) -> Block {
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

// Block-level analysis functions (get_outer_bounds, block_connection_matrix,
// standardize_block_orientation, check_collinearity, calculate_outward_normals,
// find_bounding_faces, find_closest_block, common_neighbor,
// build_connectivity_graph) are in crate::block_analysis.

// Cylindrical coordinate helpers (to_theta, to_radius, find_angular_bounding_faces)
// are in crate::cylindrical.
