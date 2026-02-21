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

/// Quantize a 3D point to integer grid coordinates for hashing.
pub(crate) fn quantize_point(p: [Float; 3], tol: Float) -> (i64, i64, i64) {
    let s = if tol > 0.0 { tol } else { DEFAULT_TOL };
    (
        (p[0] / s).round() as i64,
        (p[1] / s).round() as i64,
        (p[2] / s).round() as i64,
    )
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
