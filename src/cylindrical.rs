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
