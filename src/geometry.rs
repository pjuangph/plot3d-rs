//! Pure geometry helpers for face intersection, normals, and polygon clipping.
//!
//! These functions operate on raw coordinates and have no dependency on
//! block or face data structures.

use crate::{
    utils::{cross3, distance3, sub3, vec_norm3},
    Float,
};

const DEFAULT_TOL: Float = 1e-8;

/// Compute the Euclidean distance between two points.
#[inline]
pub(crate) fn distance(a: [Float; 3], b: [Float; 3]) -> Float {
    distance3(a, b)
}

/// Bin width for a proximity grid: never narrower than the tolerance (so a
/// tolerance ball spans at most three bins per axis) and never so narrow
/// that `coordinate / bin` leaves the `i64` range.
fn grid_bin(tol: Float, max_abs: Float) -> Float {
    let s = if tol > 0.0 { tol } else { DEFAULT_TOL };
    let bin = s.max(max_abs * (2.0 as Float).powi(-30));
    if bin > 0.0 && bin.is_finite() {
        bin
    } else {
        1.0
    }
}

/// Proximity grid over 3-D points for "is any stored point within `tol` of
/// `p`" queries.
///
/// A query visits every bin the tolerance ball around `p` intersects and then
/// compares actual Euclidean distances. Bin membership never decides
/// coincidence: two points closer than `tol` that straddle a bin boundary are
/// still found. (The rounded-bin comparison this replaces rejected exactly
/// that pair.)
pub(crate) struct PointGrid3 {
    bin: Float,
    map: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
    pts: Vec<[Float; 3]>,
}

impl PointGrid3 {
    pub fn new(pts: &[[Float; 3]], tol: Float) -> Self {
        let max_abs = pts
            .iter()
            .flat_map(|p| p.iter())
            .filter(|c| c.is_finite())
            .fold(0.0 as Float, |a, c| a.max(c.abs()));
        let bin = grid_bin(tol, max_abs);
        let mut map: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
            std::collections::HashMap::with_capacity(pts.len());
        for (i, p) in pts.iter().enumerate() {
            map.entry(Self::key(p, bin)).or_default().push(i);
        }
        Self {
            bin,
            map,
            pts: pts.to_vec(),
        }
    }

    fn key(p: &[Float; 3], bin: Float) -> (i64, i64, i64) {
        (
            (p[0] / bin).floor() as i64,
            (p[1] / bin).floor() as i64,
            (p[2] / bin).floor() as i64,
        )
    }

    /// Index and distance of the nearest stored point within `tol` of `p`
    /// among those accepted by `pred`.
    pub fn nearest_within_where(
        &self,
        p: [Float; 3],
        tol: Float,
        mut pred: impl FnMut(usize) -> bool,
    ) -> Option<(usize, Float)> {
        let lo = Self::key(&[p[0] - tol, p[1] - tol, p[2] - tol], self.bin);
        let hi = Self::key(&[p[0] + tol, p[1] + tol, p[2] + tol], self.bin);
        let tol2 = tol * tol;
        let mut best: Option<(usize, Float)> = None;
        for kx in lo.0..=hi.0 {
            for ky in lo.1..=hi.1 {
                for kz in lo.2..=hi.2 {
                    let Some(ids) = self.map.get(&(kx, ky, kz)) else { continue };
                    for &i in ids {
                        if !pred(i) {
                            continue;
                        }
                        let q = self.pts[i];
                        let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                        if d2 <= tol2 && best.map_or(true, |(_, b)| d2 < b) {
                            best = Some((i, d2));
                        }
                    }
                }
            }
        }
        best.map(|(i, d2)| (i, d2.sqrt()))
    }

    pub fn nearest_within(&self, p: [Float; 3], tol: Float) -> Option<(usize, Float)> {
        self.nearest_within_where(p, tol, |_| true)
    }

    pub fn any_within(&self, p: [Float; 3], tol: Float) -> bool {
        self.nearest_within(p, tol).is_some()
    }

    /// Number of query points that have at least one stored point within
    /// `tol`.
    pub fn count_with_partner(&self, queries: &[[Float; 3]], tol: Float) -> usize {
        queries.iter().filter(|q| self.any_within(**q, tol)).count()
    }
}

/// Proximity grid over 2-D points; see [`PointGrid3`].
pub(crate) struct PointGrid2 {
    bin: Float,
    map: std::collections::HashMap<(i64, i64), Vec<usize>>,
    pts: Vec<[Float; 2]>,
}

impl PointGrid2 {
    pub fn new(pts: &[[Float; 2]], tol: Float) -> Self {
        let max_abs = pts
            .iter()
            .flat_map(|p| p.iter())
            .filter(|c| c.is_finite())
            .fold(0.0 as Float, |a, c| a.max(c.abs()));
        let bin = grid_bin(tol, max_abs);
        let mut map: std::collections::HashMap<(i64, i64), Vec<usize>> =
            std::collections::HashMap::with_capacity(pts.len());
        for (i, p) in pts.iter().enumerate() {
            map.entry(Self::key(p, bin)).or_default().push(i);
        }
        Self {
            bin,
            map,
            pts: pts.to_vec(),
        }
    }

    fn key(p: &[Float; 2], bin: Float) -> (i64, i64) {
        ((p[0] / bin).floor() as i64, (p[1] / bin).floor() as i64)
    }

    pub fn nearest_within_where(
        &self,
        p: [Float; 2],
        tol: Float,
        mut pred: impl FnMut(usize) -> bool,
    ) -> Option<(usize, Float)> {
        let lo = Self::key(&[p[0] - tol, p[1] - tol], self.bin);
        let hi = Self::key(&[p[0] + tol, p[1] + tol], self.bin);
        let tol2 = tol * tol;
        let mut best: Option<(usize, Float)> = None;
        for kx in lo.0..=hi.0 {
            for ky in lo.1..=hi.1 {
                let Some(ids) = self.map.get(&(kx, ky)) else { continue };
                for &i in ids {
                    if !pred(i) {
                        continue;
                    }
                    let q = self.pts[i];
                    let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                    if d2 <= tol2 && best.map_or(true, |(_, b)| d2 < b) {
                        best = Some((i, d2));
                    }
                }
            }
        }
        best.map(|(i, d2)| (i, d2.sqrt()))
    }

    pub fn nearest_within(&self, p: [Float; 2], tol: Float) -> Option<(usize, Float)> {
        self.nearest_within_where(p, tol, |_| true)
    }

    pub fn count_with_partner(&self, queries: &[[Float; 2]], tol: Float) -> usize {
        queries.iter().filter(|q| self.nearest_within(**q, tol).is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_finds_pairs_that_straddle_a_bin_boundary() {
        let tol: Float = 1e-3;
        // With bins `tol` wide the boundary sits at multiples of tol; place
        // one point just below it and the query just above, 0.9 tol apart.
        for n in [0i64, 1, 7, 1000, -3] {
            let boundary = n as Float * tol;
            let stored = [boundary - 0.05 * tol, 0.3, -0.7];
            let query = [boundary + 0.85 * tol, 0.3, -0.7];
            let grid = PointGrid3::new(&[stored], tol);
            assert!(grid.any_within(query, tol), "n={n}");
            // A rounding-based key would put them in different bins.
            let old_key = |x: Float| (x / tol).round() as i64;
            assert_ne!(old_key(stored[0]), old_key(query[0]), "n={n}");
            // And a point genuinely farther than tol is not found.
            assert!(!grid.any_within([boundary + 1.1 * tol, 0.3, -0.7], tol));
        }
        let g2 = PointGrid2::new(&[[0.95e-3, 2.0]], 1e-3);
        assert!(g2.nearest_within([1.05e-3, 2.0], 1e-3).is_some());
        assert!(g2.nearest_within([2.1e-3, 2.0], 1e-3).is_none());
    }

    #[test]
    fn grid_reports_the_nearest_partner_not_the_first_bin_hit() {
        let tol: Float = 0.5;
        let pts = [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let grid = PointGrid3::new(&pts, tol);
        let (i, d) = grid.nearest_within([0.12, 0.0, 0.0], tol).unwrap();
        assert_eq!(i, 2);
        assert!((d - 0.02).abs() < 1e-12);
    }
}

/// Convert a tuple `(x, y, z)` into an array `[Float; 3]`.
pub(crate) fn to_array(p: (Float, Float, Float)) -> [Float; 3] {
    [p.0, p.1, p.2]
}

/// Average unit normal of a quad given as vertex positions.
/// Splits into two triangles, normalises each to unit length, then averages.
/// This matches the Python `_quad_normal()` and correctly handles skew quads.
pub(crate) fn quad_normal_from_verts(verts: &[[Float; 3]]) -> [Float; 3] {
    if verts.len() < 3 {
        return [0.0, 0.0, 1.0];
    }
    let c1 = cross3(sub3(verts[1], verts[0]), sub3(verts[2], verts[0]));
    let len1 = vec_norm3(c1);
    let n1 = if len1 > 1e-30 {
        [c1[0] / len1, c1[1] / len1, c1[2] / len1]
    } else {
        [0.0, 0.0, 1.0]
    };
    if verts.len() < 4 {
        return n1;
    }
    let c2 = cross3(sub3(verts[2], verts[0]), sub3(verts[3], verts[0]));
    let len2 = vec_norm3(c2);
    let n2 = if len2 > 1e-30 {
        [c2[0] / len2, c2[1] / len2, c2[2] / len2]
    } else {
        n1
    };
    let avg = [
        (n1[0] + n2[0]) * 0.5,
        (n1[1] + n2[1]) * 0.5,
        (n1[2] + n2[2]) * 0.5,
    ];
    let len = vec_norm3(avg);
    if len < 1e-30 {
        return [0.0, 0.0, 1.0];
    }
    [avg[0] / len, avg[1] / len, avg[2] / len]
}

/// Compute the axis-aligned bounding box of a set of 3D vertices.
pub(crate) fn vertex_aabb(verts: &[[Float; 3]]) -> ([Float; 3], [Float; 3]) {
    let mut min = [Float::INFINITY; 3];
    let mut max = [Float::NEG_INFINITY; 3];
    for v in verts {
        for d in 0..3 {
            min[d] = min[d].min(v[d]);
            max[d] = max[d].max(v[d]);
        }
    }
    (min, max)
}

/// Return the axis index (0, 1, or 2) with the largest absolute normal component.
pub(crate) fn dominant_projection_axis(n: [Float; 3]) -> usize {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ax >= ay && ax >= az {
        0
    } else if ay >= az {
        1
    } else {
        2
    }
}

/// Project 3D points to 2D by dropping one axis.
pub(crate) fn project_drop_axis(pts: &[[Float; 3]], drop: usize) -> Vec<[Float; 2]> {
    let (a, b) = match drop {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    pts.iter().map(|p| [p[a], p[b]]).collect()
}

/// Signed area of a 2D polygon via the shoelace formula.
pub(crate) fn poly_area_2d(poly: &[[Float; 2]]) -> Float {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    area * 0.5
}

/// Sutherland-Hodgman convex polygon clipping.
///
/// Clips `subject` against each edge of `clipper`. Both are 2D polygons.
pub(crate) fn clip_sutherland_hodgman(
    subject: &[[Float; 2]],
    clipper: &[[Float; 2]],
) -> Vec<[Float; 2]> {
    if subject.is_empty() || clipper.is_empty() {
        return Vec::new();
    }

    let mut output = subject.to_vec();

    let cn = clipper.len();
    for i in 0..cn {
        if output.is_empty() {
            return output;
        }
        let edge_start = clipper[i];
        let edge_end = clipper[(i + 1) % cn];
        let input = output;
        output = Vec::new();

        let inside = |p: [Float; 2]| -> bool {
            (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1])
                - (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
                >= 0.0
        };

        let intersect = |p1: [Float; 2], p2: [Float; 2]| -> [Float; 2] {
            let d1x = p2[0] - p1[0];
            let d1y = p2[1] - p1[1];
            let d2x = edge_end[0] - edge_start[0];
            let d2y = edge_end[1] - edge_start[1];
            let denom = d1x * d2y - d1y * d2x;
            if denom.abs() < 1e-30 {
                return p1;
            }
            let t = ((edge_start[0] - p1[0]) * d2y - (edge_start[1] - p1[1]) * d2x) / denom;
            [p1[0] + t * d1x, p1[1] + t * d1y]
        };

        let n_in = input.len();
        for j in 0..n_in {
            let current = input[j];
            let prev = input[(j + n_in - 1) % n_in];
            if inside(current) {
                if !inside(prev) {
                    output.push(intersect(prev, current));
                }
                output.push(current);
            } else if inside(prev) {
                output.push(intersect(prev, current));
            }
        }
    }
    output
}
