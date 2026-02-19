//! Translational periodicity utilities that mirror the original Python implementation.
//!
//! While there is not yet a dedicated Rust integration test, the `tests/test_rotational_periodicity.rs`
//! fixture demonstrates the expected data flow for the periodicity modules and should be referenced
//! when extending this module. Run `cargo doc --open` to view these notes alongside the generated API
//! documentation.

use std::collections::HashSet;

use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    block::Block,
    block_face_functions::{find_bounding_faces, full_face_match_transformed, outer_face_records_to_list, Face},
    connectivity::{FaceMatch, FaceRecord},
    utils::{compute_min_gcd, FaceKey},
    Float,
};

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
    delta: Option<Float>,
    translational_direction: &str,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let axis = translational_direction.trim().to_ascii_lowercase();
    assert!(matches!(axis.as_str(), "x" | "y" | "z"));

    let (lower_faces_records, upper_faces_records, _, _) =
        find_bounding_faces(blocks, outer_faces, &axis, "both", 1e-6, 1e-6);

    let gcd_to_use = compute_min_gcd(blocks);

    let blocks_reduced = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    let lower_faces = outer_face_records_to_list(&blocks_reduced, &lower_faces_records, gcd_to_use);
    let upper_faces = outer_face_records_to_list(&blocks_reduced, &upper_faces_records, gcd_to_use);

    let delta_axis = match axis.as_str() {
        "x" => {
            let min_x = blocks_reduced
                .iter()
                .map(|b| b.x_slice().iter().cloned().fold(Float::INFINITY, Float::min))
                .fold(Float::INFINITY, Float::min);
            let max_x = blocks_reduced
                .iter()
                .map(|b| {
                    b.x_slice()
                        .iter()
                        .cloned()
                        .fold(Float::NEG_INFINITY, Float::max)
                })
                .fold(Float::NEG_INFINITY, Float::max);
            delta.unwrap_or(max_x - min_x)
        }
        "y" => {
            let min_y = blocks_reduced
                .iter()
                .map(|b| b.y_slice().iter().cloned().fold(Float::INFINITY, Float::min))
                .fold(Float::INFINITY, Float::min);
            let max_y = blocks_reduced
                .iter()
                .map(|b| {
                    b.y_slice()
                        .iter()
                        .cloned()
                        .fold(Float::NEG_INFINITY, Float::max)
                })
                .fold(Float::NEG_INFINITY, Float::max);
            delta.unwrap_or(max_y - min_y)
        }
        _ => {
            let min_z = blocks_reduced
                .iter()
                .map(|b| b.z_slice().iter().cloned().fold(Float::INFINITY, Float::min))
                .fold(Float::INFINITY, Float::min);
            let max_z = blocks_reduced
                .iter()
                .map(|b| {
                    b.z_slice()
                        .iter()
                        .cloned()
                        .fold(Float::NEG_INFINITY, Float::max)
                })
                .fold(Float::NEG_INFINITY, Float::max);
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

    let mut periodic_matches = Vec::new();

    let lower_pool = dedup_faces(lower_faces);
    let upper_pool = dedup_faces(upper_faces);

    // ── Phase 1: Fast full-face matching via 4-corner comparison ──
    let corner_tol = node_tol_xyz.unwrap_or(1e-6);
    let axis_char = axis.chars().next().unwrap();
    let mut consumed_lower = HashSet::<FaceKey>::new();
    let mut consumed_upper = HashSet::<FaceKey>::new();

    let pb1 = ProgressBar::new(lower_pool.len() as u64);
    pb1.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} faces ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb1.set_message("Translational Phase 1 (corners)");

    for face_l in &lower_pool {
        pb1.inc(1);
        if consumed_lower.contains(&face_key(face_l)) {
            continue;
        }
        // Build forward translation transform: shift face_l by +delta along axis
        let matched = upper_pool.iter().find_map(|face_u| {
            if consumed_upper.contains(&face_key(face_u)) {
                return None;
            }
            // Try lower shifted up vs upper original
            let orient = full_face_match_transformed(
                face_l,
                face_u,
                |mut p| {
                    match axis_char {
                        'x' => p[0] += delta_axis,
                        'y' => p[1] += delta_axis,
                        _ => p[2] += delta_axis,
                    }
                    p
                },
                corner_tol,
            );
            if orient.is_some() {
                return Some((face_u.clone(), orient.unwrap()));
            }
            // Try lower original vs upper shifted down
            let orient = full_face_match_transformed(
                face_u,
                face_l,
                |mut p| {
                    match axis_char {
                        'x' => p[0] -= delta_axis,
                        'y' => p[1] -= delta_axis,
                        _ => p[2] -= delta_axis,
                    }
                    p
                },
                corner_tol,
            );
            if orient.is_some() {
                return Some((face_u.clone(), orient.unwrap()));
            }
            None
        });
        if let Some((face_u, orient)) = matched {
            consumed_lower.insert(face_key(face_l));
            consumed_upper.insert(face_key(&face_u));
            periodic_matches.push(FaceMatch {
                block1: FaceRecord::from_face(face_l),
                block2: FaceRecord::from_face(&face_u),
                points: Vec::new(),
                orientation: Some(orient),
            });
        }
    }
    pb1.finish_with_message("Translational Phase 1 done");

    // Build remainder pools for Phase 2
    let lower_remainder: Vec<Face> = lower_pool
        .iter()
        .filter(|f| !consumed_lower.contains(&face_key(f)))
        .cloned()
        .collect();
    let mut upper_remainder: Vec<Face> = upper_pool
        .iter()
        .filter(|f| !consumed_upper.contains(&face_key(f)))
        .cloned()
        .collect();

    // ── Phase 2: Slow node-by-node matching on remainder ──
    let pb2 = ProgressBar::new(lower_remainder.len() as u64);
    pb2.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} faces ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb2.set_message("Translational Phase 2 (nodes)");

    for face_l in &lower_remainder {
        pb2.inc(1);
        let candidate = upper_remainder.iter().enumerate().find_map(|(idx, f)| {
            faces_translational_match(
                face_l,
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
        if let Some((pos, _mode)) = candidate {
            let face_u = upper_remainder.remove(pos);
            periodic_matches.push(FaceMatch {
                block1: FaceRecord::from_face(face_l),
                block2: FaceRecord::from_face(&face_u),
                points: Vec::new(),
                orientation: None,
            });
        }
    }
    pb2.finish_with_message("Translational Phase 2 done");

    // Free shifted block copies now that matching is complete
    drop(blocks_up);
    drop(blocks_dn);

    let mut periodic_keys = HashSet::new();
    for rec in &periodic_matches {
        periodic_keys.insert(rec.block1.index_key());
        periodic_keys.insert(rec.block2.index_key());
    }

    let mut remaining = Vec::new();
    for record in outer_faces {
        if !periodic_keys.contains(&record.index_key()) {
            remaining.push(record.clone());
        }
    }

    if gcd_to_use > 1 {
        for rec in &mut periodic_matches {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }
        for record in &mut remaining {
            record.scale_indices(gcd_to_use);
        }
    }

    (periodic_matches, remaining)
}

/// Assess one lower/upper face combo and return the match mode when successful.
fn faces_translational_match(
    face_l: &Face,
    face_u: &Face,
    blocks: &[Block],
    blocks_up: &[Block],
    blocks_dn: &[Block],
    axis: &str,
    delta_axis: Float,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
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
    override_tol: Option<Float>,
    axis: &str,
) -> Float {
    if let Some(tol) = override_tol {
        return tol;
    }
    let spacing_a = median_inplane_spacing(face_a, &blocks[face_a.block_index().unwrap()], axis);
    let spacing_b = median_inplane_spacing(face_b, &blocks[face_b.block_index().unwrap()], axis);
    (0.03 * spacing_a.max(spacing_b)).max(1e-4)
}

/// Compute a median edge length for the face in the non-periodic directions.
fn median_inplane_spacing(face: &Face, block: &Block, axis: &str) -> Float {
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
    delta: Float,
    tol: Float,
    axis: &str,
    min_shared_frac: Float,
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
        && (shared as Float) >= min_shared_frac * (key_a.len().min(key_b.len()) as Float)
}

/// Project 3D points onto the plane orthogonal to `axis`.
fn project_plane(points: &[[Float; 3]], axis: &str) -> Vec<[Float; 2]> {
    points
        .iter()
        .map(|p| match axis {
            "x" => [p[1], p[2]],
            "y" => [p[0], p[2]],
            _ => [p[0], p[1]],
        })
        .collect()
}

/// Remove duplicate faces while preserving the first occurrence.
fn dedup_faces(mut faces: Vec<Face>) -> Vec<Face> {
    let mut seen = HashSet::new();
    faces.retain(|f| seen.insert(face_key(f)));
    faces
}

/// Build a unique key directly from a `Face`.
#[inline]
fn face_key(face: &Face) -> FaceKey {
    face.index_key()
}
