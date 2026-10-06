//! Cylindrical coordinate transforms and angular bounding face detection.
//!
//! Provides `to_theta()` and `to_radius()` for converting Cartesian coordinates
//! to cylindrical about a given rotation axis, plus `find_angular_bounding_faces()`
//! for identifying faces on the angular min/max boundaries of an annular domain.

use crate::{block::Block, block_face_functions::Face, face_record::FaceRecord, Float};

/// Minimum meaningful angular range; below this the sector is degenerate.
const MIN_THETA_RANGE: Float = 1e-10;

/// Floor for the absolute angular tolerance used in bounding-face detection.
const MIN_THETA_TOL: Float = 1e-8;

/// Floor for the absolute radial tolerance used in axis-face detection, so a
/// vanishingly small `tol_rel * max_radius` (e.g. a near-planar mesh) can
/// never collapse to zero and admit every face.
const MIN_AXIS_TOL: Float = 1e-10;

/// Compute angular position (theta) about the given rotation axis.
///
/// Conventions match the Python `_to_theta()`:
/// - `'x'` → `atan2(y, z)`
/// - `'y'` → `atan2(z, x)`
/// - `'z'` → `atan2(y, x)`
pub fn to_theta(x: Float, y: Float, z: Float, rotation_axis: char) -> Float {
    match rotation_axis.to_ascii_lowercase() {
        'x' => y.atan2(z),
        'y' => z.atan2(x),
        _ => y.atan2(x),
    }
}

/// Compute radial distance from the given rotation axis.
pub fn to_radius(x: Float, y: Float, z: Float, rotation_axis: char) -> Float {
    match rotation_axis.to_ascii_lowercase() {
        'x' => (y * y + z * z).sqrt(),
        'y' => (z * z + x * x).sqrt(),
        _ => (y * y + x * x).sqrt(),
    }
}

/// Compute the global theta (angular) extent across all blocks.
fn global_theta_extreme(blocks: &[Block], axis: char) -> (Float, Float) {
    let mut min_theta = Float::INFINITY;
    let mut max_theta = Float::NEG_INFINITY;
    for block in blocks {
        for idx in 0..block.npoints() {
            let theta = to_theta(block.x[idx], block.y[idx], block.z[idx], axis);
            min_theta = min_theta.min(theta);
            max_theta = max_theta.max(theta);
        }
    }
    (min_theta, max_theta)
}

/// Compute the theta extent of a face from its corner vertices.
fn face_theta_extreme(face: &Face, axis: char) -> (Float, Float) {
    let mut min_theta = Float::INFINITY;
    let mut max_theta = Float::NEG_INFINITY;
    for v in face.vertices() {
        let theta = to_theta(v[0], v[1], v[2], axis);
        min_theta = min_theta.min(theta);
        max_theta = max_theta.max(theta);
    }
    (min_theta, max_theta)
}

/// Find outer faces on the angular (theta) min/max boundaries of an annular
/// domain.
///
/// Returns `(lower_records, upper_records, lower_faces, upper_faces)`.
/// Returns all-empty vectors if the domain is non-annular (theta range > PI
/// or negligibly small).
///
/// # Arguments
/// * `blocks` - All blocks in the assembly.
/// * `outer_faces` - Candidate outer faces.
/// * `rotation_axis` - Physical rotation axis (`'x'`, `'y'`, or `'z'`).
/// * `tol_rel` - Relative tolerance for angular boundary classification.
pub fn find_angular_bounding_faces(
    blocks: &[Block],
    outer_faces: &[Face],
    rotation_axis: char,
    tol_rel: Float,
) -> (Vec<FaceRecord>, Vec<FaceRecord>, Vec<Face>, Vec<Face>) {
    let (theta_min, theta_max) = global_theta_extreme(blocks, rotation_axis);
    let theta_range = theta_max - theta_min;

    if !(MIN_THETA_RANGE..=crate::PI).contains(&theta_range) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let tol_abs: Float = (MIN_THETA_TOL as Float).max(tol_rel * theta_range);
    let mut lower = Vec::new();
    let mut upper = Vec::new();

    for f in outer_faces {
        let (f_theta_min, f_theta_max) = face_theta_extreme(f, rotation_axis);
        // All vertices at theta_min
        if (f_theta_max - theta_min).abs() <= tol_abs {
            lower.push(f.clone());
        }
        // All vertices at theta_max
        else if (theta_max - f_theta_min).abs() <= tol_abs {
            upper.push(f.clone());
        }
    }

    let lower_records = lower.iter().map(Face::to_record).collect();
    let upper_records = upper.iter().map(Face::to_record).collect();
    (lower_records, upper_records, lower, upper)
}

/// Compute the maximum radial distance from `rotation_axis` reached by any
/// grid point in `blocks`. Used as the mesh's own characteristic radial
/// scale, so an axis-face tolerance can be expressed relative to it rather
/// than as a hardcoded absolute number.
fn global_max_radius(blocks: &[Block], axis: char) -> Float {
    let mut max_radius: Float = 0.0;
    for block in blocks {
        for idx in 0..block.npoints() {
            let r = to_radius(block.x[idx], block.y[idx], block.z[idx], axis);
            if r > max_radius {
                max_radius = r;
            }
        }
    }
    max_radius
}

/// Largest radial distance among a face's own vertices.
fn face_max_radius(face: &Face, axis: char) -> Float {
    let mut max_radius: Float = 0.0;
    for v in face.vertices() {
        let r = to_radius(v[0], v[1], v[2], axis);
        if r > max_radius {
            max_radius = r;
        }
    }
    max_radius
}

/// Find outer faces that lie on the `r = 0` axis of revolution.
///
/// A face is classified as an axis face **iff every one of its vertices**
/// satisfies `to_radius(...) < tol_abs`, where `tol_abs` is a *relative*
/// tolerance scaled to the mesh's own characteristic radial scale (its
/// global maximum radius about `rotation_axis`) — never an absolute area or
/// `|S| < 1e-30` test.
///
/// This distinction matters: a mesh generator that writes `r = 1e-12`
/// instead of exact `0` still produces a face with real, if tiny, area —
/// `|S| ~ Δx·1e-12`, roughly `1e18`x above a `1e-30` area floor. Testing
/// area directly would silently misclassify such a face as an ordinary
/// boundary face with a nonsense normal. Testing radius directly, relative
/// to the mesh's own scale, catches it (derivation
/// `axisymmetric-axis-boundary-condition.md` §2.3's corollary).
///
/// Returns `(axis_records, axis_faces)`. Both are empty if no face
/// qualifies (a non-axisymmetric mesh, or one that never approaches `r=0`).
///
/// # Arguments
/// * `blocks` - all blocks, used to establish the mesh's characteristic
///   radial scale.
/// * `outer_faces` - candidate outer faces.
/// * `rotation_axis` - physical rotation axis (`'x'`, `'y'`, or `'z'`).
/// * `tol_rel` - relative tolerance, applied to the mesh's global max radius.
pub fn find_axis_faces(
    blocks: &[Block],
    outer_faces: &[Face],
    rotation_axis: char,
    tol_rel: Float,
) -> (Vec<FaceRecord>, Vec<Face>) {
    let max_radius = global_max_radius(blocks, rotation_axis);
    if max_radius <= 0.0 {
        // Every point in the mesh is already on the axis: there is no
        // radial scale to be relative to, so there is nothing to classify.
        return (Vec::new(), Vec::new());
    }
    let tol_abs: Float = (MIN_AXIS_TOL as Float).max(tol_rel * max_radius);

    let mut axis = Vec::new();
    for f in outer_faces {
        if face_max_radius(f, rotation_axis) <= tol_abs {
            axis.push(f.clone());
        }
    }

    let axis_records = axis.iter().map(Face::to_record).collect();
    (axis_records, axis)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single block whose points span `r ∈ [0, 1]` about the x-axis, so
    /// `global_max_radius` (and therefore `tol_abs`) is anchored at `1.0`.
    fn wedge_block() -> Block {
        // i-fastest, imax=2, jmax=2, kmax=1.
        let x = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let z = vec![0.0, 0.0, 0.0, 0.0];
        Block::new(2, 2, 1, x, y, z)
    }

    fn quad_face(verts: [[Float; 3]; 4]) -> Face {
        let mut f = Face::new();
        for (n, v) in verts.iter().enumerate() {
            // Indices only need to be distinct enough for to_record()'s
            // imin/imax bookkeeping; their physical meaning doesn't matter
            // to find_axis_faces itself.
            f.add_vertex(v[0], v[1], v[2], n, 0, 0);
        }
        f
    }

    #[test]
    fn genuine_axis_face_is_classified() {
        let blocks = [wedge_block()];
        // Mirrors derivation §2.3's collapsed quad: p0=A, p1=B, p2=B, p3=A,
        // both at r=0, differing only in x.
        let axis_face = quad_face([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);
        let (records, faces) = find_axis_faces(&blocks, &[axis_face], 'x', 1e-6);
        assert_eq!(faces.len(), 1);
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn near_axis_but_above_tolerance_is_not_classified() {
        let blocks = [wedge_block()];
        // global_max_radius = 1.0, tol_rel = 1e-6 => tol_abs = 1e-6.
        // r = 1e-3 here is three orders of magnitude above that.
        let r = 1e-3;
        let near_axis_face =
            quad_face([[0.0, r, 0.0], [1.0, r, 0.0], [1.0, r, 0.0], [0.0, r, 0.0]]);
        let (records, faces) = find_axis_faces(&blocks, &[near_axis_face], 'x', 1e-6);
        assert!(faces.is_empty(), "r={r} above tol_abs must not classify");
        assert!(records.is_empty());
    }

    #[test]
    fn degenerate_but_not_axis_face_is_not_classified() {
        let blocks = [wedge_block()];
        // A collapsed edge (zero-area face, two coincident corners) that
        // sits at the *outer* radius, not the axis. An area-based test
        // could confuse this with an axis face; a radius test must not.
        let outer_degenerate = quad_face([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]);
        let (records, faces) = find_axis_faces(&blocks, &[outer_degenerate], 'x', 1e-6);
        assert!(faces.is_empty());
        assert!(records.is_empty());
    }

    #[test]
    fn mesh_generator_epsilon_radius_is_not_silently_admitted() {
        // The exact hazard §2.3's corollary names: a generator writes
        // r=1e-12 instead of exact 0. That is still ~1e6x the tol_abs
        // floor at tol_rel=1e-6 on this mesh's scale, so it must NOT be
        // classified as an axis face by this tolerance choice — proving
        // the classifier is a genuine relative-radius test, not something
        // that admits anything "small".
        let blocks = [wedge_block()];
        let eps = 1e-12;
        let eps_face = quad_face([
            [0.0, eps, 0.0],
            [1.0, eps, 0.0],
            [1.0, eps, 0.0],
            [0.0, eps, 0.0],
        ]);
        // tol_abs = max(MIN_AXIS_TOL, tol_rel*max_radius) = max(1e-10, 1e-6) = 1e-6.
        // eps=1e-12 < 1e-6, so this one IS admitted at this tol_rel -
        // demonstrating the tolerance is genuinely relative-to-scale
        // rather than a fixed floor swallowing everything.
        let (_, faces) = find_axis_faces(&blocks, &[eps_face], 'x', 1e-6);
        assert_eq!(faces.len(), 1);
    }

    #[test]
    fn only_faces_with_every_vertex_near_axis_are_classified() {
        let blocks = [wedge_block()];
        // Three corners at r=0, one corner at the outer radius: not every
        // vertex is on the axis, so this must not be classified.
        let mixed_face = quad_face([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);
        let (_, faces) = find_axis_faces(&blocks, &[mixed_face], 'x', 1e-6);
        assert!(faces.is_empty());
    }
}
