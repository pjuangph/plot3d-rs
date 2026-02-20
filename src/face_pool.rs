//! Cylindrical-coordinate face pool and edge matching utilities for
//! rotational periodicity detection.

use std::collections::HashSet;

use crate::{
    block::Block,
    block_face_functions::{to_radius, to_theta, Face, FaceAxis},
    face_record::{FaceKey, FaceRecord},
    utils::{apply_rotation, distance3},
    Float, PI,
};

/// Extract the axial coordinate (along the rotation axis) from a 3D point.
#[inline]
pub(crate) fn axial_coord(p: [Float; 3], rotation_axis: char) -> Float {
    match rotation_axis.to_ascii_lowercase() {
        'x' => p[0],
        'y' => p[1],
        _ => p[2],
    }
}

/// Precomputed cylindrical-coordinate metadata for a single face in the pool.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct CylindricalFaceInfo {
    pub theta_centroid: Float,
    pub axial_centroid: Float,
    pub radial_centroid: Float,
    pub theta_min: Float,
    pub theta_max: Float,
    pub axial_min: Float,
    pub axial_max: Float,
    pub radial_min: Float,
    pub radial_max: Float,
}

impl CylindricalFaceInfo {
    pub fn from_face(face: &Face, rotation_axis: char) -> Self {
        let verts = face.vertices();
        let c = face.centroid();
        let theta_c = to_theta(c[0], c[1], c[2], rotation_axis);
        let axial_c = axial_coord(c, rotation_axis);
        let radial_c = to_radius(c[0], c[1], c[2], rotation_axis);

        let mut theta_min = theta_c;
        let mut theta_max = theta_c;
        let mut axial_min = axial_c;
        let mut axial_max = axial_c;
        let mut radial_min = radial_c;
        let mut radial_max = radial_c;

        for v in verts {
            let th = to_theta(v[0], v[1], v[2], rotation_axis);
            let ax = axial_coord(*v, rotation_axis);
            let r = to_radius(v[0], v[1], v[2], rotation_axis);
            theta_min = theta_min.min(th);
            theta_max = theta_max.max(th);
            axial_min = axial_min.min(ax);
            axial_max = axial_max.max(ax);
            radial_min = radial_min.min(r);
            radial_max = radial_max.max(r);
        }

        Self {
            theta_centroid: theta_c,
            axial_centroid: axial_c,
            radial_centroid: radial_c,
            theta_min,
            theta_max,
            axial_min,
            axial_max,
            radial_min,
            radial_max,
        }
    }
}

/// A 1D line of grid points along one boundary of a structured face.
#[derive(Clone, Debug)]
pub(crate) struct StructuredEdge {
    pub coords: Vec<[Float; 3]>,
}

/// Result of comparing two structured edges (one rotated).
#[derive(Debug)]
pub(crate) enum EdgeMatchResult {
    None,
    Full,
    Partial,
}

/// Managed pool of outer faces with cylindrical-coordinate lookup.
pub(crate) struct FacePool {
    pub faces: Vec<Face>,
    cyl_info: Vec<CylindricalFaceInfo>,
    /// Indices into `faces` sorted by theta_centroid (ascending).
    theta_sorted: Vec<usize>,
    consumed: HashSet<FaceKey>,
    rotation_axis: char,
}

impl FacePool {
    pub fn new(faces: Vec<Face>, rotation_axis: char) -> Self {
        let cyl_info: Vec<CylindricalFaceInfo> = faces
            .iter()
            .map(|f| CylindricalFaceInfo::from_face(f, rotation_axis))
            .collect();
        let mut theta_sorted: Vec<usize> = (0..faces.len()).collect();
        theta_sorted.sort_by(|&a, &b| {
            cyl_info[a]
                .theta_centroid
                .partial_cmp(&cyl_info[b].theta_centroid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            faces,
            cyl_info,
            theta_sorted,
            consumed: HashSet::new(),
            rotation_axis,
        }
    }

    pub fn add_face(&mut self, face: Face) {
        let info = CylindricalFaceInfo::from_face(&face, self.rotation_axis);
        let idx = self.faces.len();
        self.faces.push(face);
        self.cyl_info.push(info);
        // Insert into sorted list at correct position
        let theta = self.cyl_info[idx].theta_centroid;
        let pos = self
            .theta_sorted
            .binary_search_by(|&i| {
                self.cyl_info[i]
                    .theta_centroid
                    .partial_cmp(&theta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|p| p);
        self.theta_sorted.insert(pos, idx);
    }

    pub fn consume(&mut self, key: FaceKey) {
        self.consumed.insert(key);
    }

    pub fn is_consumed(&self, idx: usize) -> bool {
        let face = &self.faces[idx];
        self.consumed.contains(&face.index_key())
    }

    /// Find face indices whose cylindrical extents could overlap with the given
    /// target theta and matching axial/radial ranges.
    pub fn find_candidates(
        &self,
        target_theta: Float,
        axial_range: (Float, Float),
        radial_range: (Float, Float),
        theta_tol: Float,
    ) -> Vec<usize> {
        let theta_lo = target_theta - theta_tol;
        let theta_hi = target_theta + theta_tol;

        let mut candidates = Vec::new();

        // Binary search for the starting position in sorted theta
        let start = self
            .theta_sorted
            .binary_search_by(|&i| {
                self.cyl_info[i]
                    .theta_centroid
                    .partial_cmp(&theta_lo)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|p| p);

        // Scan forward from start
        for &pool_idx in &self.theta_sorted[start..] {
            let info = &self.cyl_info[pool_idx];
            if info.theta_centroid > theta_hi {
                break;
            }
            if self.is_consumed(pool_idx) {
                continue;
            }
            // Check axial overlap
            if info.axial_max
                < axial_range.0 - 0.1 * (axial_range.1 - axial_range.0).abs().max(1e-12)
                || info.axial_min
                    > axial_range.1 + 0.1 * (axial_range.1 - axial_range.0).abs().max(1e-12)
            {
                continue;
            }
            // Check radial overlap
            if info.radial_max
                < radial_range.0 - 0.1 * (radial_range.1 - radial_range.0).abs().max(1e-12)
                || info.radial_min
                    > radial_range.1 + 0.1 * (radial_range.1 - radial_range.0).abs().max(1e-12)
            {
                continue;
            }
            candidates.push(pool_idx);
        }

        // Handle theta wrapping at ±π boundary
        // If theta_lo < -π, also search near +π end
        if theta_lo < -PI {
            let wrapped_lo = theta_lo + 2.0 * PI;
            let wrapped_hi = PI; // search from wrapped_lo to +π
            let ws = self
                .theta_sorted
                .binary_search_by(|&i| {
                    self.cyl_info[i]
                        .theta_centroid
                        .partial_cmp(&wrapped_lo)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|p| p);
            for &pool_idx in &self.theta_sorted[ws..] {
                let info = &self.cyl_info[pool_idx];
                if info.theta_centroid > wrapped_hi {
                    break;
                }
                if self.is_consumed(pool_idx) {
                    continue;
                }
                if info.axial_max
                    < axial_range.0 - 0.1 * (axial_range.1 - axial_range.0).abs().max(1e-12)
                    || info.axial_min
                        > axial_range.1
                            + 0.1 * (axial_range.1 - axial_range.0).abs().max(1e-12)
                {
                    continue;
                }
                if info.radial_max
                    < radial_range.0 - 0.1 * (radial_range.1 - radial_range.0).abs().max(1e-12)
                    || info.radial_min
                        > radial_range.1
                            + 0.1 * (radial_range.1 - radial_range.0).abs().max(1e-12)
                {
                    continue;
                }
                candidates.push(pool_idx);
            }
        }
        // If theta_hi > π, also search near -π end
        if theta_hi > PI {
            let wrapped_hi = theta_hi - 2.0 * PI;
            let wrapped_lo = -PI;
            let ws = self
                .theta_sorted
                .binary_search_by(|&i| {
                    self.cyl_info[i]
                        .theta_centroid
                        .partial_cmp(&wrapped_lo)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|p| p);
            for &pool_idx in &self.theta_sorted[ws..] {
                let info = &self.cyl_info[pool_idx];
                if info.theta_centroid > wrapped_hi {
                    break;
                }
                if self.is_consumed(pool_idx) {
                    continue;
                }
                if info.axial_max
                    < axial_range.0 - 0.1 * (axial_range.1 - axial_range.0).abs().max(1e-12)
                    || info.axial_min
                        > axial_range.1
                            + 0.1 * (axial_range.1 - axial_range.0).abs().max(1e-12)
                {
                    continue;
                }
                if info.radial_max
                    < radial_range.0 - 0.1 * (radial_range.1 - radial_range.0).abs().max(1e-12)
                    || info.radial_min
                        > radial_range.1
                            + 0.1 * (radial_range.1 - radial_range.0).abs().max(1e-12)
                {
                    continue;
                }
                candidates.push(pool_idx);
            }
        }

        candidates
    }

    /// Find candidates for a face at the given index by searching both +angle and -angle
    /// theta targets, deduplicating the results.
    pub fn find_rotational_candidates(
        &self,
        idx: usize,
        rotation_angle: Float,
        theta_tol: Float,
    ) -> Vec<usize> {
        let info = &self.cyl_info[idx];
        let target_fwd = info.theta_centroid + rotation_angle;
        let target_rev = info.theta_centroid - rotation_angle;
        let axial_range = (info.axial_min, info.axial_max);
        let radial_range = (info.radial_min, info.radial_max);
        let mut candidates =
            self.find_candidates(target_fwd, axial_range, radial_range, theta_tol);
        candidates.extend(self.find_candidates(target_rev, axial_range, radial_range, theta_tol));
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Collect all unconsumed face indices.
    pub fn active_indices(&self) -> Vec<usize> {
        (0..self.faces.len())
            .filter(|&i| !self.is_consumed(i))
            .collect()
    }

    /// Drain all unconsumed faces as FaceRecords.
    pub fn drain_as_records(&self) -> Vec<FaceRecord> {
        self.faces
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.is_consumed(*i))
            .map(|(_, f)| f.to_record())
            .collect()
    }
}

// ============================================================================
// Edge extraction and matching
// ============================================================================

/// Extract the 4 boundary edges of a structured face from its parent block.
pub(crate) fn extract_face_edges(face: &Face, block: &Block) -> Vec<StructuredEdge> {
    let axis = match face.const_axis() {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut edges = Vec::with_capacity(4);

    match axis {
        FaceAxis::I => {
            let ic = face.imin();
            edges.push(build_edge(
                block,
                (face.kmin()..=face.kmax())
                    .map(|k| (ic, face.jmin(), k))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.kmin()..=face.kmax())
                    .map(|k| (ic, face.jmax(), k))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.jmin()..=face.jmax())
                    .map(|j| (ic, j, face.kmin()))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.jmin()..=face.jmax())
                    .map(|j| (ic, j, face.kmax()))
                    .collect(),
            ));
        }
        FaceAxis::J => {
            let jc = face.jmin();
            edges.push(build_edge(
                block,
                (face.kmin()..=face.kmax())
                    .map(|k| (face.imin(), jc, k))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.kmin()..=face.kmax())
                    .map(|k| (face.imax(), jc, k))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.imin()..=face.imax())
                    .map(|i| (i, jc, face.kmin()))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.imin()..=face.imax())
                    .map(|i| (i, jc, face.kmax()))
                    .collect(),
            ));
        }
        FaceAxis::K => {
            let kc = face.kmin();
            edges.push(build_edge(
                block,
                (face.jmin()..=face.jmax())
                    .map(|j| (face.imin(), j, kc))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.jmin()..=face.jmax())
                    .map(|j| (face.imax(), j, kc))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.imin()..=face.imax())
                    .map(|i| (i, face.jmin(), kc))
                    .collect(),
            ));
            edges.push(build_edge(
                block,
                (face.imin()..=face.imax())
                    .map(|i| (i, face.jmax(), kc))
                    .collect(),
            ));
        }
    }
    edges
}

fn build_edge(block: &Block, ijk_list: Vec<(usize, usize, usize)>) -> StructuredEdge {
    let coords: Vec<[Float; 3]> = ijk_list
        .iter()
        .map(|&(i, j, k)| {
            let (x, y, z) = block.xyz(i, j, k);
            [x, y, z]
        })
        .collect();
    StructuredEdge { coords }
}

/// Compare two structured edges. Points of edge_a are rotated before comparison.
/// Returns Full if all points match, Partial if >=2 contiguous match, None otherwise.
pub(crate) fn match_edges(
    edge_a: &StructuredEdge,
    edge_b: &StructuredEdge,
    rotation_matrix: [[Float; 3]; 3],
    tol: Float,
) -> EdgeMatchResult {
    if edge_a.coords.is_empty() || edge_b.coords.is_empty() {
        return EdgeMatchResult::None;
    }

    let rotated_a: Vec<[Float; 3]> = edge_a
        .coords
        .iter()
        .map(|p| apply_rotation(*p, rotation_matrix))
        .collect();

    // Find longest contiguous run of point-to-point matches
    let mut best_len = 0usize;
    let mut cur_len = 0usize;

    for ra in &rotated_a {
        let has_match = edge_b.coords.iter().any(|pb| distance3(*ra, *pb) <= tol);
        if has_match {
            cur_len += 1;
            best_len = best_len.max(cur_len);
        } else {
            cur_len = 0;
        }
    }

    if best_len < 2 {
        EdgeMatchResult::None
    } else if best_len == rotated_a.len() && best_len == edge_b.coords.len() {
        EdgeMatchResult::Full
    } else {
        EdgeMatchResult::Partial
    }
}

/// Count how many edge pairs between two faces have Full or Partial matches.
pub(crate) fn count_edge_matches(
    edges_a: &[StructuredEdge],
    edges_b: &[StructuredEdge],
    rotation_matrix: [[Float; 3]; 3],
    tol: Float,
) -> usize {
    let mut count = 0;
    for ea in edges_a {
        for eb in edges_b {
            match match_edges(ea, eb, rotation_matrix, tol) {
                EdgeMatchResult::None => {}
                _ => count += 1,
            }
        }
    }
    count
}
