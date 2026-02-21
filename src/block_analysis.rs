//! Block-level analysis: connectivity graphs, orientation standardization,
//! bounding face detection, and outward normal computation.
//!
//! These functions operate on collections of blocks and face records to
//! answer questions about block relationships and spatial extents.

use std::collections::{HashMap, HashSet, VecDeque};

use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    block::Block,
    block_face_functions::{
        get_outer_faces, outer_face_records_to_list, reduce_blocks, Face, FaceAxis,
    },
    face_record::{FaceMatch, FaceRecord},
    geometry::{distance, to_array},
    utils::{compute_min_gcd, cross3, sub3},
    Float,
};

/// Compute the global bounds across all blocks.
///
/// # Arguments
/// * `blocks` - Collection of blocks to inspect.
///
/// # Returns
/// `(x_bounds, y_bounds, z_bounds)` or `None` when the list is empty.
pub fn get_outer_bounds(
    blocks: &[Block],
) -> Option<((Float, Float), (Float, Float), (Float, Float))> {
    if blocks.is_empty() {
        return None;
    }
    let mut xmin = Float::INFINITY;
    let mut xmax = Float::NEG_INFINITY;
    let mut ymin = Float::INFINITY;
    let mut ymax = Float::NEG_INFINITY;
    let mut zmin = Float::INFINITY;
    let mut zmax = Float::NEG_INFINITY;
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
    pub node_tol_xyz: Float,
    pub min_shared_frac: Float,
    pub min_shared_abs: usize,
    pub stride_u: usize,
    pub stride_v: usize,
    pub use_area_fallback: bool,
    pub area_min_overlap_frac: Float,
}

impl Default for BlockConnectionOptions {
    fn default() -> Self {
        Self {
            node_tol_xyz: 1e-7,
            min_shared_frac: 0.02,
            min_shared_abs: 4,
            stride_u: 1,
            stride_v: 1,
            use_area_fallback: true,
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
    tol: Float,
    options: BlockConnectionOptions,
) -> (Vec<Vec<i8>>, Vec<Vec<i8>>, Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let gcd = compute_min_gcd(blocks);
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

    use rayon::prelude::*;

    let all_pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let pb = ProgressBar::new(all_pairs.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} pairs ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("Connection matrix");

    let results: Vec<(usize, usize, bool, Option<FaceAxis>)> = all_pairs
        .par_iter()
        .map(|&(i, j)| {
            pb.inc(1);
            let mut matched_axis: Option<FaceAxis> = None;
            let mut connected = false;
            for face_i in &faces_by_block[i] {
                for face_j in &faces_by_block[j] {
                    let node_match = face_i.touches_by_nodes(
                        face_j,
                        &reduced[i],
                        &reduced[j],
                        options.node_tol_xyz,
                        options.min_shared_frac,
                        options.min_shared_abs,
                        options.stride_u,
                        options.stride_v,
                    );

                    let area_match = if !node_match && options.use_area_fallback {
                        face_i.touches(face_j, 10.0, 1e-6, options.area_min_overlap_frac)
                    } else {
                        false
                    };

                    if node_match || area_match {
                        if face_i.const_axis() == face_j.const_axis() {
                            matched_axis = face_i.const_axis();
                        }
                        connected = true;
                        break;
                    }
                }
                if connected {
                    break;
                }
            }
            (i, j, connected, matched_axis)
        })
        .collect();

    for (i, j, connected, axis) in results {
        if connected {
            connectivity[i][j] = 1;
            connectivity[j][i] = 1;
            match axis {
                Some(FaceAxis::I) => {
                    conn_i[i][j] = 1;
                    conn_i[j][i] = 1;
                }
                Some(FaceAxis::J) => {
                    conn_j[i][j] = 1;
                    conn_j[j][i] = 1;
                }
                Some(FaceAxis::K) => {
                    conn_k[i][j] = 1;
                    conn_k[j][i] = 1;
                }
                None => {}
            }
        } else {
            connectivity[i][j] = -1;
            connectivity[j][i] = -1;
        }
    }
    pb.finish_with_message("Connection matrix done");

    if tol.is_finite() {
        let _ = tol;
    }

    (connectivity, conn_i, conn_j, conn_k)
}

/// Standardise block orientation so that indices increase with coordinate values.
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
    x: &mut [Float],
    y: &mut [Float],
    z: &mut [Float],
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
pub fn check_collinearity(v1: [Float; 3], v2: [Float; 3]) -> bool {
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    cross.iter().all(|c| c.abs() <= Float::EPSILON)
}

/// Compute outward normals for the six faces of a block.
///
/// Returns normals for `(Imin, Jmin, Kmin, Imax, Jmax, Kmax)`.
pub fn calculate_outward_normals(
    block: &Block,
) -> (
    [Float; 3],
    [Float; 3],
    [Float; 3],
    [Float; 3],
    [Float; 3],
    [Float; 3],
) {
    let i0 = to_array(block.xyz(0, 0, 0));
    let ij = to_array(block.xyz(0, block.jmax - 1, 0));
    let ik = to_array(block.xyz(0, 0, block.kmax - 1));
    let ni_min = cross3(sub3(ij, i0), sub3(ik, i0));

    let i1 = to_array(block.xyz(block.imax - 1, 0, 0));
    let i1j = to_array(block.xyz(block.imax - 1, block.jmax - 1, 0));
    let i1k = to_array(block.xyz(block.imax - 1, 0, block.kmax - 1));
    let ni_max = cross3(sub3(i1j, i1), sub3(i1k, i1));

    let j0 = to_array(block.xyz(0, 0, 0));
    let ji = to_array(block.xyz(block.imax - 1, 0, 0));
    let jk = to_array(block.xyz(0, 0, block.kmax - 1));
    let nj_min = cross3(sub3(ji, j0), sub3(jk, j0));

    let j1 = to_array(block.xyz(0, block.jmax - 1, 0));
    let j1i = to_array(block.xyz(block.imax - 1, block.jmax - 1, 0));
    let j1k = to_array(block.xyz(0, block.jmax - 1, block.kmax - 1));
    let nj_max = cross3(sub3(j1i, j1), sub3(j1k, j1));

    let k0 = to_array(block.xyz(0, 0, 0));
    let ki = to_array(block.xyz(block.imax - 1, 0, 0));
    let kj = to_array(block.xyz(0, block.jmax - 1, 0));
    let nk_min = cross3(sub3(ki, k0), sub3(kj, k0));

    let k1 = to_array(block.xyz(0, 0, block.kmax - 1));
    let k1i = to_array(block.xyz(block.imax - 1, 0, block.kmax - 1));
    let k1j = to_array(block.xyz(0, block.jmax - 1, block.kmax - 1));
    let nk_max = cross3(sub3(k1i, k1), sub3(k1j, k1));

    (ni_min, nj_min, nk_min, ni_max, nj_max, nk_max)
}

/// Identify outer faces on the extreme of the requested axis using BFS.
///
/// # Arguments
/// * `blocks` - All blocks in the system.
/// * `outer_faces_records` - Optional pre-computed outer faces in record form.
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
    tol_rel: Float,
    node_tol_xyz: Float,
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

    let gcd = compute_min_gcd(blocks);
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

fn global_axis_bounds(blocks: &[Block], axis: FaceAxis) -> Option<(Float, Float)> {
    let mut min_val = Float::INFINITY;
    let mut max_val = Float::NEG_INFINITY;
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
    tol_abs: Float,
    node_tol_xyz: Float,
) -> Vec<Face> {
    if faces.is_empty() {
        return Vec::new();
    }

    let mut plane_value = if is_min {
        Float::INFINITY
    } else {
        Float::NEG_INFINITY
    };
    for face in faces {
        for v in face.vertices() {
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
        let mut fmin = Float::INFINITY;
        let mut fmax = Float::NEG_INFINITY;
        for v in face.vertices() {
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
    centroid: [Float; 3],
    direction: &str,
    minvalue: bool,
) -> Option<(usize, Float, Float, Float)> {
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
    let mut best_dist = Float::INFINITY;
    for (idx, block) in blocks.iter().enumerate() {
        let cx = block.x_slice().iter().sum::<Float>() / block.x_slice().len() as Float;
        let cy = block.y_slice().iter().sum::<Float>() / block.y_slice().len() as Float;
        let cz = block.z_slice().iter().sum::<Float>() / block.z_slice().len() as Float;
        let dist = distance([cx, cy, cz], [target_x, target_y, target_z]);
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx.map(|idx| (idx, target_x, target_y, target_z))
}

/// Graph helper: find a neighbour connected to both `a` and `b`.
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

/// Convert face matches into an undirected adjacency map between block indices.
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
