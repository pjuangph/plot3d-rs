//! Translational periodicity utilities that mirror the original Python implementation.
//!
//! While there is not yet a dedicated Rust integration test, the `tests/test_rotational_periodicity.rs`
//! fixture demonstrates the expected data flow for the periodicity modules and should be referenced
//! when extending this module. Run `cargo doc --open` to view these notes alongside the generated API
//! documentation.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::{
    block::Block,
    block_face_functions::{find_bounding_faces, outer_face_records_to_list, Face},
    connectivity::FaceRecord,
};

/// Exportable record for a translationally periodic face pair.
/// Serialized pairing of translation-periodic faces, including index mapping metadata.
#[derive(Clone, Debug, Serialize)]
pub struct TranslationalPairExport {
    pub block1: FaceRecord,
    pub block2: FaceRecord,
    pub mapping: HashMap<String, String>,
    pub mode: String,
}

/// Detailed periodic pair including Faces and index mapping.
/// Detailed pairing retaining the full face objects and the discovered mapping.
#[derive(Clone, Debug)]
pub struct TranslationalPair {
    pub face1: Face,
    pub face2: Face,
    pub mapping: HashMap<String, String>,
    pub mode: String,
}

/// Detect translational periodicity along an axis.
/// Discover translational periodicity along a chosen axis.
///
/// # Testing
/// End-to-end validation is planned to follow the pattern established in
/// `tests/test_rotational_periodicity.rs`. Until then, exercising this function in a binary or
/// ad-hoc script is recommended to mirror the original Python examples.
pub fn translational_periodicity(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    delta: Option<f64>,
    translational_direction: &str,
    node_tol_xyz: Option<f64>,
    min_shared_frac: f64,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
) -> (
    Vec<TranslationalPairExport>,
    Vec<TranslationalPair>,
    Vec<FaceRecord>,
) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let axis = translational_direction.trim().to_ascii_lowercase();
    assert!(matches!(axis.as_str(), "x" | "y" | "z"));

    let (lower_faces_records, upper_faces_records, _, _) =
        find_bounding_faces(blocks, outer_faces, &axis, "both", 1e-6, 1e-6);

    let mut gcd_array = Vec::new();
    for block in blocks {
        gcd_array.push(gcd_three(block.imax - 1, block.jmax - 1, block.kmax - 1));
    }
    let gcd_to_use = gcd_array.into_iter().min().unwrap_or(1).max(1);

    let blocks_reduced = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    let lower_faces = outer_face_records_to_list(&blocks_reduced, &lower_faces_records, 1);
    let upper_faces = outer_face_records_to_list(&blocks_reduced, &upper_faces_records, 1);

    let delta_axis = match axis.as_str() {
        "x" => {
            let min_x = blocks_reduced
                .iter()
                .map(|b| b.x_slice().iter().cloned().fold(f64::INFINITY, f64::min))
                .fold(f64::INFINITY, f64::min);
            let max_x = blocks_reduced
                .iter()
                .map(|b| {
                    b.x_slice()
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            delta.unwrap_or(max_x - min_x)
        }
        "y" => {
            let min_y = blocks_reduced
                .iter()
                .map(|b| b.y_slice().iter().cloned().fold(f64::INFINITY, f64::min))
                .fold(f64::INFINITY, f64::min);
            let max_y = blocks_reduced
                .iter()
                .map(|b| {
                    b.y_slice()
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            delta.unwrap_or(max_y - min_y)
        }
        _ => {
            let min_z = blocks_reduced
                .iter()
                .map(|b| b.z_slice().iter().cloned().fold(f64::INFINITY, f64::min))
                .fold(f64::INFINITY, f64::min);
            let max_z = blocks_reduced
                .iter()
                .map(|b| {
                    b.z_slice()
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            delta.unwrap_or(max_z - min_z)
        }
    };

    let blocks_up: Vec<Block> = blocks_reduced
        .iter()
        .map(|b| b.shifted(delta_axis, axis.chars().next().unwrap()))
        .collect();
    let blocks_dn: Vec<Block> = blocks_reduced
        .iter()
        .map(|b| b.shifted(-delta_axis, axis.chars().next().unwrap()))
        .collect();

    let mut periodic_exports = Vec::new();
    let mut periodic_pairs = Vec::new();

    let lower_pool = dedup_faces(lower_faces);
    let mut upper_pool = dedup_faces(upper_faces);

    for face_l in lower_pool.clone() {
        let candidate = upper_pool.iter().enumerate().find_map(|(idx, f)| {
            faces_translational_match(
                &face_l,
                f,
                &blocks_reduced,
                &blocks_up,
                &blocks_dn,
                axis.as_str(),
                delta_axis,
                node_tol_xyz,
                min_shared_frac,
                min_shared_abs,
                stride_u,
                stride_v,
            )
            .map(|mode| (idx, mode))
        });
        if let Some((pos, mode)) = candidate {
            let face_u = upper_pool.remove(pos);
            let mapping = mapping_minmax(&face_l, &face_u);

            periodic_exports.push(TranslationalPairExport {
                block1: FaceRecord::from_face(&face_l),
                block2: FaceRecord::from_face(&face_u),
                mapping: mapping.clone(),
                mode: mode.clone(),
            });
            periodic_pairs.push(TranslationalPair {
                face1: face_l.clone(),
                face2: face_u.clone(),
                mapping,
                mode,
            });
        }
    }

    let mut periodic_keys = HashSet::new();
    for rec in &periodic_exports {
        periodic_keys.insert(face_record_key(&rec.block1));
        periodic_keys.insert(face_record_key(&rec.block2));
    }

    let mut remaining = Vec::new();
    for record in outer_faces {
        if !periodic_keys.contains(&face_record_key(record)) {
            remaining.push(record.clone());
        }
    }

    if gcd_to_use > 1 {
        for rec in &mut periodic_exports {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }
        for pair in &mut periodic_pairs {
            pair.face1.scale_indices(gcd_to_use);
            pair.face2.scale_indices(gcd_to_use);
        }
        for record in &mut remaining {
            record.scale_indices(gcd_to_use);
        }
    }

    (periodic_exports, periodic_pairs, remaining)
}

/// Assess one lower/upper face combo and return the match mode when successful.
fn faces_translational_match(
    face_l: &Face,
    face_u: &Face,
    blocks: &[Block],
    blocks_up: &[Block],
    blocks_dn: &[Block],
    axis: &str,
    delta_axis: f64,
    node_tol_xyz: Option<f64>,
    min_shared_frac: f64,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
) -> Option<String> {
    let tol_pair = pair_tolerance(face_l, face_u, blocks, node_tol_xyz, axis);

    if orthogonal_precheck(
        face_l,
        face_u,
        &blocks_up[face_l.block_index().unwrap()],
        &blocks[face_u.block_index().unwrap()],
        delta_axis,
        tol_pair,
        axis,
        min_shared_frac,
        min_shared_abs,
    ) {
        return Some(format!("{axis}_precheck_lower_up"));
    }
    if face_l.touches_by_nodes(
        face_u,
        &blocks_up[face_l.block_index().unwrap()],
        &blocks[face_u.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some("lower_up_vs_upper_orig".to_string());
    }
    if face_l.touches_by_nodes(
        face_u,
        &blocks[face_l.block_index().unwrap()],
        &blocks_dn[face_u.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some("lower_orig_vs_upper_dn".to_string());
    }
    if orthogonal_precheck(
        face_u,
        face_l,
        &blocks_up[face_u.block_index().unwrap()],
        &blocks[face_l.block_index().unwrap()],
        delta_axis,
        tol_pair,
        axis,
        min_shared_frac,
        min_shared_abs,
    ) {
        return Some(format!("{axis}_precheck_upper_up"));
    }
    if face_u.touches_by_nodes(
        face_l,
        &blocks_up[face_u.block_index().unwrap()],
        &blocks[face_l.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some("upper_up_vs_lower_orig".to_string());
    }
    face_u
        .touches_by_nodes(
            face_l,
            &blocks[face_u.block_index().unwrap()],
            &blocks_dn[face_l.block_index().unwrap()],
            tol_pair,
            min_shared_frac,
            min_shared_abs,
            stride_u,
            stride_v,
        )
        .then(|| "upper_orig_vs_lower_dn".to_string())
}

/// Decide the XYZ tolerance for a particular face pair, optionally honoring a global override.
fn pair_tolerance(
    face_a: &Face,
    face_b: &Face,
    blocks: &[Block],
    override_tol: Option<f64>,
    axis: &str,
) -> f64 {
    if let Some(tol) = override_tol {
        return tol;
    }
    let spacing_a = median_inplane_spacing(face_a, &blocks[face_a.block_index().unwrap()], axis);
    let spacing_b = median_inplane_spacing(face_b, &blocks[face_b.block_index().unwrap()], axis);
    (0.03 * spacing_a.max(spacing_b)).max(1e-4)
}

/// Compute a median edge length for the face in the non-periodic directions.
fn median_inplane_spacing(face: &Face, block: &Block, axis: &str) -> f64 {
    let points = face.grid_points(block, 1, 1);
    if points.len() <= 1 {
        return 1.0;
    }
    let mut spacings = Vec::new();
    for window in points.windows(2) {
        let p0 = window[0];
        let p1 = window[1];
        let diff = match axis {
            "x" => [(p0[1] - p1[1]).abs(), (p0[2] - p1[2]).abs()],
            "y" => [(p0[0] - p1[0]).abs(), (p0[2] - p1[2]).abs()],
            _ => [(p0[0] - p1[0]).abs(), (p0[1] - p1[1]).abs()],
        };
        spacings.push(diff[0].hypot(diff[1]));
    }
    spacings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spacings[spacings.len() / 2]
}

/// Perform a quick planar projection test to reject clearly mismatched faces.
fn orthogonal_precheck(
    face_a: &Face,
    face_b: &Face,
    block_a: &Block,
    block_b: &Block,
    delta: f64,
    tol: f64,
    axis: &str,
    min_shared_frac: f64,
    min_shared_abs: usize,
) -> bool {
    let mut pts_a = face_a.grid_points(block_a, 1, 1);
    let pts_b = face_b.grid_points(block_b, 1, 1);
    if pts_a.is_empty() || pts_b.is_empty() {
        return false;
    }
    match axis {
        "x" => pts_a.iter_mut().for_each(|p| p[0] += delta),
        "y" => pts_a.iter_mut().for_each(|p| p[1] += delta),
        _ => pts_a.iter_mut().for_each(|p| p[2] += delta),
    }

    let proj_a = project_plane(&pts_a, axis);
    let proj_b = project_plane(&pts_b, axis);

    let key_a: HashSet<(i64, i64)> = proj_a
        .iter()
        .map(|p| ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64))
        .collect();
    let key_b: HashSet<(i64, i64)> = proj_b
        .iter()
        .map(|p| ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64))
        .collect();

    let shared = key_a.intersection(&key_b).count();
    shared >= min_shared_abs
        && (shared as f64) >= min_shared_frac * (key_a.len().min(key_b.len()) as f64)
}

/// Project 3D points onto the plane orthogonal to `axis`.
fn project_plane(points: &[[f64; 3]], axis: &str) -> Vec<[f64; 2]> {
    points
        .iter()
        .map(|p| match axis {
            "x" => [p[1], p[2]],
            "y" => [p[0], p[2]],
            _ => [p[0], p[1]],
        })
        .collect()
}

/// Derive the I/J/K orientation mapping between two matched faces.
fn mapping_minmax(face_a: &Face, face_b: &Face) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for axis in ["I", "J", "K"] {
        let amin = axis_min(face_a, axis);
        let amax = axis_max(face_a, axis);
        let bmin = axis_min(face_b, axis);
        let bmax = axis_max(face_b, axis);
        let value = if amin == bmin && amax == bmax {
            "min->min"
        } else if amin == bmax && amax == bmin {
            "min->max"
        } else {
            let mm_cost =
                (amin as isize - bmin as isize).abs() + (amax as isize - bmax as isize).abs();
            let mm_flip_cost =
                (amin as isize - bmax as isize).abs() + (amax as isize - bmin as isize).abs();
            if mm_cost <= mm_flip_cost {
                "min->min"
            } else {
                "min->max"
            }
        };
        map.insert(axis.to_string(), value.to_string());
    }
    map
}

/// Convenience accessor for the minimum index along `axis`.
fn axis_min(face: &Face, axis: &str) -> usize {
    match axis {
        "I" => face.imin(),
        "J" => face.jmin(),
        _ => face.kmin(),
    }
}

/// Convenience accessor for the maximum index along `axis`.
fn axis_max(face: &Face, axis: &str) -> usize {
    match axis {
        "I" => face.imax(),
        "J" => face.jmax(),
        _ => face.kmax(),
    }
}

/// Remove duplicate faces while preserving the first occurrence.
fn dedup_faces(mut faces: Vec<Face>) -> Vec<Face> {
    let mut seen = HashSet::new();
    faces.retain(|f| seen.insert(face_key(f)));
    faces
}

type FaceKey = (usize, usize, usize, usize, usize, usize, usize);

/// Build a unique key for an exportable face record.
fn face_record_key(record: &FaceRecord) -> FaceKey {
    (
        record.block_index,
        record.imin,
        record.jmin,
        record.kmin,
        record.imax,
        record.jmax,
        record.kmax,
    )
}

/// Build a unique key directly from a `Face`.
fn face_key(face: &Face) -> FaceKey {
    (
        face.block_index().unwrap_or(usize::MAX),
        face.imin(),
        face.jmin(),
        face.kmin(),
        face.imax(),
        face.jmax(),
        face.kmax(),
    )
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
