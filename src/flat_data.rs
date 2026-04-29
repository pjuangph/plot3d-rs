//! Flat SoA (Structure of Arrays) mesh representation for GPU finite-volume solvers.
//!
//! Multi-block structured grids store data in per-block (i,j,k) arrays with
//! pointer-based block lookups. This is efficient for CPU codes but hostile to
//! GPU architectures that need coalesced memory access across thousands of
//! threads.
//!
//! This module converts a multi-block Plot3D grid into a flat, unstructured-like
//! representation where:
//!
//! - Every cell has a single global integer ID (no block/i/j/k tuple).
//! - Every face has an owner cell and a neighbor cell (or -1 for boundaries).
//! - All geometric data (volumes, area vectors, centers) are contiguous arrays.
//! - No pointer dereferences are needed -- every access is `array[id]`.
//!
//! # Face Convention
//!
//! Faces are categorized into three types:
//!
//! 1. **Interior faces** (within a block): between adjacent cells along the
//!    i, j, or k axis. These have both `owner` and `neighbor` as valid cell IDs.
//!
//! 2. **Cross-block faces**: between cells in different blocks that share an
//!    interface (from [`FaceMatch`] data). Both `owner` and `neighbor` are valid.
//!
//! 3. **Boundary (outer) faces**: block boundary faces that are NOT matched to
//!    another block. These have `neighbor = -1` and carry a `surface_id` for
//!    boundary condition assignment.
//!
//! The face area vector points from the owner cell toward the neighbor cell
//! (or outward for boundary faces), consistent with finite-volume flux
//! conventions.

use crate::block::Block;
use crate::dual_graph::{build_cell_graph, cell_index, CellGraph};
use crate::face_record::{FaceMatch, FaceRecord};
use crate::metrics::{compute_cell_centers, compute_cell_volumes, compute_face_metrics};
use crate::Float;

/// Flat mesh representation for finite-volume GPU solvers.
///
/// All arrays are contiguous and indexed by a single integer ID.
/// Cell arrays have length `n_cells`; face arrays have length `n_faces`.
///
/// This layout eliminates all structured (i,j,k) indexing and multi-block
/// pointer chains. The data is designed for coalesced GPU memory access:
/// a warp of threads processing consecutive face IDs will read consecutive
/// memory locations.
#[derive(Clone, Debug)]
pub struct FlatMesh {
    // -- Cell data (length = n_cells) --

    /// Total number of cells across all blocks.
    pub n_cells: usize,
    /// Cell volume computed via the divergence-theorem method.
    /// Units: length^3. Must be positive for valid meshes.
    pub cell_volume: Vec<Float>,
    /// X-coordinate of cell centroid (arithmetic mean of 8 corner nodes).
    pub cell_center_x: Vec<Float>,
    /// Y-coordinate of cell centroid.
    pub cell_center_y: Vec<Float>,
    /// Z-coordinate of cell centroid.
    pub cell_center_z: Vec<Float>,

    // -- Face data (length = n_faces) --

    /// Total number of faces (interior + cross-block + boundary).
    pub n_faces: usize,
    /// Owner cell for each face. The face area vector points away from
    /// the owner toward the neighbor. Always a valid cell index in `[0, n_cells)`.
    pub face_owner: Vec<u32>,
    /// Neighbor cell for each face. For interior and cross-block faces this is
    /// a valid cell index. For boundary faces this is `-1`.
    pub face_neighbor: Vec<i32>,
    /// X-component of face area vector (outward normal * area magnitude).
    pub face_area_x: Vec<Float>,
    /// Y-component of face area vector.
    pub face_area_y: Vec<Float>,
    /// Z-component of face area vector.
    pub face_area_z: Vec<Float>,
    /// X-coordinate of face centroid (arithmetic mean of the 4 corner nodes).
    /// Matches Fortran's ccCoord face-centroid formula
    /// (M_ccMBMesh.F:2528-2530), used by the solver to place ghost cell
    /// centers via exact plane reflection rather than a cuboid V/|A|
    /// approximation.
    pub face_centroid_x: Vec<Float>,
    /// Y-coordinate of face centroid.
    pub face_centroid_y: Vec<Float>,
    /// Z-coordinate of face centroid.
    pub face_centroid_z: Vec<Float>,

    // -- Boundary face metadata --

    /// Surface ID for boundary faces (used for BC assignment).
    /// `-1` for interior and cross-block faces.
    pub face_surface_id: Vec<i32>,

    // -- Reverse mapping (for post-processing / writing results back to Plot3D) --

    /// Which original block each cell came from. Length = `n_cells`.
    pub cell_block_id: Vec<u32>,
    /// Local cell index within the original block. Length = `n_cells`.
    /// To recover (i,j,k): use the block's cell dimensions from the CellGraph.
    pub cell_local_id: Vec<u32>,
}

impl FlatMesh {
    /// Produce a human-readable summary of the mesh statistics.
    ///
    /// Reports cell count, face count, boundary face count, and volume extremes.
    /// Useful for quick sanity checks after mesh conversion.
    pub fn stats(&self) -> String {
        let n_boundary = self.face_neighbor.iter().filter(|&&n| n < 0).count();
        let n_interior = self.n_faces - n_boundary;

        let (min_vol, max_vol) = if self.cell_volume.is_empty() {
            (0.0 as Float, 0.0 as Float)
        } else {
            let min_v = self.cell_volume.iter().cloned().fold(Float::INFINITY, Float::min);
            let max_v = self.cell_volume.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
            (min_v, max_v)
        };

        let total_vol: Float = self.cell_volume.iter().sum();

        format!(
            "FlatMesh statistics:\n\
             \x20 Cells:          {}\n\
             \x20 Faces (total):  {}\n\
             \x20   Interior:     {}\n\
             \x20   Boundary:     {}\n\
             \x20 Volume (total): {:.6e}\n\
             \x20 Volume (min):   {:.6e}\n\
             \x20 Volume (max):   {:.6e}",
            self.n_cells, self.n_faces, n_interior, n_boundary,
            total_vol, min_vol, max_vol,
        )
    }
}

/// Build a flat mesh from multi-block Plot3D data.
///
/// This is the central conversion function: structured blocks + connectivity
/// data are transformed into flat, GPU-friendly arrays.
///
/// # Pipeline
///
/// 1. **Build dual graph**: compute global cell numbering via [`build_cell_graph`].
/// 2. **Compute cell metrics**: volumes and centroids for each block using the
///    `metrics` module, then scatter into global arrays.
/// 3. **Build interior faces**: within each block, iterate over the three face
///    families (I-faces, J-faces, K-faces). Each interior face connects two
///    adjacent cells.
/// 4. **Build cross-block faces**: from the `face_matches`, each interface cell
///    pair becomes a face. Area vectors are taken from the metrics of whichever
///    block "owns" the face (block1 side).
/// 5. **Build boundary faces**: the `outer_faces` parameter lists all unmatched
///    boundary surfaces. Each boundary face has `neighbor = -1` and carries the
///    surface's ID for boundary condition dispatch.
///
/// # Arguments
///
/// * `blocks` - All blocks in the multi-block grid.
/// * `face_matches` - Block-block interface connectivity.
/// * `outer_faces` - Unmatched boundary faces with surface IDs. Each `FaceRecord`
///   should have its `id` field set to the desired surface ID.
///
/// # Returns
///
/// A [`FlatMesh`] ready for GPU upload.
pub fn build_flat_mesh(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    outer_faces: &[FaceRecord],
) -> FlatMesh {
    // --- Step 1: build the dual graph for global cell numbering ---
    let graph = build_cell_graph(blocks, face_matches);
    let n_cells = graph.n_cells;

    // --- Step 2: compute and flatten cell metrics ---
    let mut cell_volume = vec![0.0 as Float; n_cells];
    let mut cell_center_x = vec![0.0 as Float; n_cells];
    let mut cell_center_y = vec![0.0 as Float; n_cells];
    let mut cell_center_z = vec![0.0 as Float; n_cells];
    let mut cell_block_id = vec![0u32; n_cells];
    let mut cell_local_id = vec![0u32; n_cells];

    for (b, blk) in blocks.iter().enumerate() {
        let vols = compute_cell_volumes(blk);
        let (xc, yc, zc) = compute_cell_centers(blk);
        let offset = graph.block_offset[b];
        let n_local = vols.len();

        for local_id in 0..n_local {
            let gid = offset + local_id;
            cell_volume[gid] = vols[local_id];
            cell_center_x[gid] = xc[local_id];
            cell_center_y[gid] = yc[local_id];
            cell_center_z[gid] = zc[local_id];
            cell_block_id[gid] = b as u32;
            cell_local_id[gid] = local_id as u32;
        }
    }

    // --- Step 3: build face lists ---
    // Pre-compute face metrics for all blocks (we need area vectors).
    let all_face_metrics: Vec<_> = blocks.iter().map(|blk| compute_face_metrics(blk)).collect();

    // We will accumulate face data into these vectors.
    let mut face_owner: Vec<u32> = Vec::new();
    let mut face_neighbor: Vec<i32> = Vec::new();
    let mut face_area_x: Vec<Float> = Vec::new();
    let mut face_area_y: Vec<Float> = Vec::new();
    let mut face_area_z: Vec<Float> = Vec::new();
    let mut face_centroid_x: Vec<Float> = Vec::new();
    let mut face_centroid_y: Vec<Float> = Vec::new();
    let mut face_centroid_z: Vec<Float> = Vec::new();
    let mut face_surface_id: Vec<i32> = Vec::new();

    // --- 3a: Interior faces within each block ---
    //
    // For a block with node dimensions (ni, nj, nk) and cell dims (nci, ncj, nck):
    //
    // I-faces (between cells differing in i):
    //   For i in 1..nci, j in 0..ncj, k in 0..nck:
    //     owner  = cell(i-1, j, k)
    //     neighbor = cell(i, j, k)
    //     area vector from I-face metrics at (i, j, k)
    //
    // Similarly for J-faces and K-faces.

    for (b, blk) in blocks.iter().enumerate() {
        let ni = blk.imax;
        let nj = blk.jmax;
        let nk = blk.kmax;
        let nci = ni - 1;
        let ncj = nj - 1;
        let nck = nk - 1;
        let offset = graph.block_offset[b];
        let fm = &all_face_metrics[b];

        // --- I-faces (interior only: i = 1..nci-1 as node index, which is
        //     between cell i-1 and cell i) ---
        // I-face at node-i has face metric index: i + ni * j + ni * (nj-1) * k
        // Interior I-faces go from node-i = 1 to node-i = nci - 1
        // (node-i = 0 and node-i = nci = ni-1 are block boundary faces)
        for k in 0..nck {
            for j in 0..ncj {
                for i in 1..nci {
                    // This I-face separates cell (i-1,j,k) from cell (i,j,k).
                    let owner_local = cell_index(i - 1, j, k, nci, ncj);
                    let neighbor_local = cell_index(i, j, k, nci, ncj);

                    // I-face metric index: i + ni * j + ni * (nj-1) * k
                    let fid = i + ni * j + ni * (nj - 1) * k;

                    face_owner.push((offset + owner_local) as u32);
                    face_neighbor.push((offset + neighbor_local) as i32);
                    face_area_x.push(fm.si_x[fid]);
                    face_area_y.push(fm.si_y[fid]);
                    face_area_z.push(fm.si_z[fid]);
                    face_centroid_x.push(fm.ci_x[fid]);
                    face_centroid_y.push(fm.ci_y[fid]);
                    face_centroid_z.push(fm.ci_z[fid]);
                    face_surface_id.push(-1);
                }
            }
        }

        // --- J-faces (interior: node-j = 1..ncj-1) ---
        // J-face metric index: i + (ni-1) * j + (ni-1) * nj * k
        for k in 0..nck {
            for j in 1..ncj {
                for i in 0..nci {
                    let owner_local = cell_index(i, j - 1, k, nci, ncj);
                    let neighbor_local = cell_index(i, j, k, nci, ncj);

                    let fid = i + (ni - 1) * j + (ni - 1) * nj * k;

                    face_owner.push((offset + owner_local) as u32);
                    face_neighbor.push((offset + neighbor_local) as i32);
                    face_area_x.push(fm.sj_x[fid]);
                    face_area_y.push(fm.sj_y[fid]);
                    face_area_z.push(fm.sj_z[fid]);
                    face_centroid_x.push(fm.cj_x[fid]);
                    face_centroid_y.push(fm.cj_y[fid]);
                    face_centroid_z.push(fm.cj_z[fid]);
                    face_surface_id.push(-1);
                }
            }
        }

        // --- K-faces (interior: node-k = 1..nck-1) ---
        // K-face metric index: i + (ni-1) * j + (ni-1) * (nj-1) * k
        for k in 1..nck {
            for j in 0..ncj {
                for i in 0..nci {
                    let owner_local = cell_index(i, j, k - 1, nci, ncj);
                    let neighbor_local = cell_index(i, j, k, nci, ncj);

                    let fid = i + (ni - 1) * j + (ni - 1) * (nj - 1) * k;

                    face_owner.push((offset + owner_local) as u32);
                    face_neighbor.push((offset + neighbor_local) as i32);
                    face_area_x.push(fm.sk_x[fid]);
                    face_area_y.push(fm.sk_y[fid]);
                    face_area_z.push(fm.sk_z[fid]);
                    face_centroid_x.push(fm.ck_x[fid]);
                    face_centroid_y.push(fm.ck_y[fid]);
                    face_centroid_z.push(fm.ck_z[fid]);
                    face_surface_id.push(-1);
                }
            }
        }
    }

    // --- 3b: Cross-block faces from face_matches ---
    //
    // For each FaceMatch, the boundary cells on block1's face connect to
    // corresponding cells on block2's face. We use block1's face metric
    // as the area vector (it points from block1 toward block2).

    for fm_match in face_matches {
        let b1 = fm_match.block1.block_index;
        let b2 = fm_match.block2.block_index;

        let edges = cross_block_face_data(
            b1,
            &fm_match.block1,
            &blocks[b1],
            b2,
            &fm_match.block2,
            &blocks[b2],
            &graph,
            &all_face_metrics[b1],
            fm_match.orientation.as_ref(),
        );

        for (owner, neighbor, ax, ay, az, cx, cy, cz) in edges {
            face_owner.push(owner);
            face_neighbor.push(neighbor as i32);
            face_area_x.push(ax);
            face_area_y.push(ay);
            face_area_z.push(az);
            face_centroid_x.push(cx);
            face_centroid_y.push(cy);
            face_centroid_z.push(cz);
            face_surface_id.push(-1);
        }
    }

    // --- 3c: Boundary (outer) faces ---
    //
    // Each outer face is a block boundary face that is NOT matched to another
    // block. The neighbor is -1, and the surface_id comes from the FaceRecord.

    for oface in outer_faces {
        let b = oface.block_index;
        let blk = &blocks[b];
        let ni = blk.imax;
        let nj = blk.jmax;
        let nk = blk.kmax;
        let nci = ni - 1;
        let ncj = nj - 1;
        let _nck = nk - 1;
        let offset = graph.block_offset[b];
        let fm = &all_face_metrics[b];

        let surface_id = oface.id.map(|id| id as i32).unwrap_or(0);

        // Determine which axis is constant and whether it is at the low or high end
        let const_axis = oface.constant_axis();
        if const_axis.is_none() {
            continue; // Skip degenerate faces
        }
        let axis = const_axis.unwrap();
        let const_vals = [oface.i_lo(), oface.j_lo(), oface.k_lo()];
        let const_v = const_vals[axis];

        let n_nodes = [ni, nj, nk];
        let is_high = const_v == n_nodes[axis] - 1;

        // Iterate over the 2D cell grid on this boundary face
        let var_axes: Vec<usize> = (0..3).filter(|&a| a != axis).collect();

        let lo = [oface.i_lo(), oface.j_lo(), oface.k_lo()];
        let hi = [oface.i_hi(), oface.j_hi(), oface.k_hi()];

        let n_u = hi[var_axes[0]] - lo[var_axes[0]];
        let n_v = hi[var_axes[1]] - lo[var_axes[1]];

        if n_u == 0 || n_v == 0 {
            continue; // Edge or point, not a face
        }

        // The cell adjacent to this boundary face:
        // If the face is at the low end (const_v == 0), the cell is at cell index 0 along that axis.
        // If at the high end, the cell is at cell index n_cells_along_axis - 1.
        let cell_const = if is_high {
            n_nodes[axis] - 2
        } else {
            0
        };

        for v in 0..n_v {
            for u in 0..n_u {
                let mut ijk = [0usize; 3];
                ijk[axis] = cell_const;
                ijk[var_axes[0]] = lo[var_axes[0]] + u;
                ijk[var_axes[1]] = lo[var_axes[1]] + v;

                let gid = offset + cell_index(ijk[0], ijk[1], ijk[2], nci, ncj);

                // Retrieve the face area vector from the appropriate face metric.
                // The face metric index depends on which face family this is.
                let (ax, ay, az) = boundary_face_area(
                    axis, const_v, ijk, blk, fm,
                );
                // Face centroid (a position — unaffected by the low-end
                // sign flip applied to the area vector below).
                let (cx, cy, cz) = boundary_face_centroid(
                    axis, const_v, ijk, blk, fm,
                );

                face_owner.push(gid as u32);
                face_neighbor.push(-1);
                // For boundary faces at the low end, the outward normal points
                // in the -axis direction, so we negate the area vector (which
                // by convention points in the +axis direction).
                if !is_high {
                    face_area_x.push(-ax);
                    face_area_y.push(-ay);
                    face_area_z.push(-az);
                } else {
                    face_area_x.push(ax);
                    face_area_y.push(ay);
                    face_area_z.push(az);
                }
                face_centroid_x.push(cx);
                face_centroid_y.push(cy);
                face_centroid_z.push(cz);
                face_surface_id.push(surface_id);
            }
        }
    }

    let n_faces = face_owner.len();

    FlatMesh {
        n_cells,
        cell_volume,
        cell_center_x,
        cell_center_y,
        cell_center_z,
        n_faces,
        face_owner,
        face_neighbor,
        face_area_x,
        face_area_y,
        face_area_z,
        face_centroid_x,
        face_centroid_y,
        face_centroid_z,
        face_surface_id,
        cell_block_id,
        cell_local_id,
    }
}

/// Retrieve the face area vector for a boundary face from the pre-computed
/// face metrics.
///
/// `axis`: 0 = I-face, 1 = J-face, 2 = K-face.
/// `const_v`: the node index value on the constant axis.
/// `ijk`: the cell indices (not node indices).
fn boundary_face_area(
    axis: usize,
    const_v: usize,
    ijk: [usize; 3],
    blk: &Block,
    fm: &crate::metrics::FaceMetrics,
) -> (Float, Float, Float) {
    let ni = blk.imax;
    let nj = blk.jmax;

    match axis {
        0 => {
            // I-face at node-i = const_v.
            // I-face metric index: i + ni * j + ni * (nj-1) * k
            // Here i = const_v (the node index), j = cell-j, k = cell-k.
            let fid = const_v + ni * ijk[1] + ni * (nj - 1) * ijk[2];
            (fm.si_x[fid], fm.si_y[fid], fm.si_z[fid])
        }
        1 => {
            // J-face at node-j = const_v.
            // J-face metric index: i + (ni-1) * j + (ni-1) * nj * k
            // Here i = cell-i, j = const_v, k = cell-k.
            let fid = ijk[0] + (ni - 1) * const_v + (ni - 1) * nj * ijk[2];
            (fm.sj_x[fid], fm.sj_y[fid], fm.sj_z[fid])
        }
        2 => {
            // K-face at node-k = const_v.
            // K-face metric index: i + (ni-1) * j + (ni-1) * (nj-1) * k
            // Here i = cell-i, j = cell-j, k = const_v.
            let fid = ijk[0] + (ni - 1) * ijk[1] + (ni - 1) * (nj - 1) * const_v;
            (fm.sk_x[fid], fm.sk_y[fid], fm.sk_z[fid])
        }
        _ => unreachable!("axis must be 0, 1, or 2"),
    }
}

/// Retrieve the face centroid for a boundary face from the pre-computed
/// face metrics. Parameters match [`boundary_face_area`].
///
/// Unlike the area vector, the centroid is a position, so the sign-flip
/// applied to low-end boundary area vectors does NOT apply here — the
/// face centroid is the same regardless of which side owns the face.
fn boundary_face_centroid(
    axis: usize,
    const_v: usize,
    ijk: [usize; 3],
    blk: &Block,
    fm: &crate::metrics::FaceMetrics,
) -> (Float, Float, Float) {
    let ni = blk.imax;
    let nj = blk.jmax;

    match axis {
        0 => {
            let fid = const_v + ni * ijk[1] + ni * (nj - 1) * ijk[2];
            (fm.ci_x[fid], fm.ci_y[fid], fm.ci_z[fid])
        }
        1 => {
            let fid = ijk[0] + (ni - 1) * const_v + (ni - 1) * nj * ijk[2];
            (fm.cj_x[fid], fm.cj_y[fid], fm.cj_z[fid])
        }
        2 => {
            let fid = ijk[0] + (ni - 1) * ijk[1] + (ni - 1) * (nj - 1) * const_v;
            (fm.ck_x[fid], fm.ck_y[fid], fm.ck_z[fid])
        }
        _ => unreachable!("axis must be 0, 1, or 2"),
    }
}

/// Build cross-block face data for a single FaceMatch.
///
/// Returns a list of
/// `(owner_global, neighbor_global, area_x, area_y, area_z, cx, cy, cz)`
/// for each cell pair at the interface — the face centroid is taken from
/// block1's metrics (same physical point regardless of which side is the
/// owner).
///
/// The area vector is taken from block1's face metrics at the interface,
/// pointing from block1 (owner) toward block2 (neighbor).
fn cross_block_face_data(
    b1: usize,
    face1: &FaceRecord,
    blk1: &Block,
    b2: usize,
    face2: &FaceRecord,
    blk2: &Block,
    graph: &CellGraph,
    fm1: &crate::metrics::FaceMetrics,
    orientation: Option<&crate::face_record::Orientation>,
) -> Vec<(u32, u32, Float, Float, Float, Float, Float, Float)> {
    let mut result = Vec::new();

    let axis1 = match face1.constant_axis() {
        Some(a) => a,
        None => return result,
    };
    let axis2 = match face2.constant_axis() {
        Some(a) => a,
        None => return result,
    };

    let f1_bounds = face1.bounds();
    let f2_bounds = face2.bounds();
    let f1_const_val = f1_bounds.0[axis1];
    let f2_const_val = f2_bounds.0[axis2];

    let n_nodes1 = [blk1.imax, blk1.jmax, blk1.kmax];
    let n_nodes2 = [blk2.imax, blk2.jmax, blk2.kmax];

    let cell1_const = if f1_const_val == 0 {
        0
    } else if f1_const_val == n_nodes1[axis1] - 1 {
        n_nodes1[axis1] - 2
    } else {
        return result;
    };
    let cell2_const = if f2_const_val == 0 {
        0
    } else if f2_const_val == n_nodes2[axis2] - 1 {
        n_nodes2[axis2] - 2
    } else {
        return result;
    };

    let is_high1 = f1_const_val == n_nodes1[axis1] - 1;

    let var_axes1: Vec<usize> = (0..3).filter(|&a| a != axis1).collect();
    let var_axes2: Vec<usize> = (0..3).filter(|&a| a != axis2).collect();

    let f1_lo = [face1.i_lo(), face1.j_lo(), face1.k_lo()];
    let f1_hi = [face1.i_hi(), face1.j_hi(), face1.k_hi()];
    let f2_lo = [face2.i_lo(), face2.j_lo(), face2.k_lo()];
    let f2_hi = [face2.i_hi(), face2.j_hi(), face2.k_hi()];

    let n_u1 = f1_hi[var_axes1[0]] - f1_lo[var_axes1[0]];
    let n_v1 = f1_hi[var_axes1[1]] - f1_lo[var_axes1[1]];
    let n_u2 = f2_hi[var_axes2[0]] - f2_lo[var_axes2[0]];
    let n_v2 = f2_hi[var_axes2[1]] - f2_lo[var_axes2[1]];

    if n_u1 == 0 || n_v1 == 0 {
        return result;
    }

    // Decode the cell-pair mapping flags. Prefer the cascade-verified
    // `permutation_index` when present — it is the only reliable
    // source of truth for cross-axis 32×32 matches where the
    // extent-shape heuristic below is structurally indeterminate.
    //
    // `permutation_index` bit encoding (matches `apply_permutation` in
    // `verification.rs:117-143`):
    //   bit 0 → u_reversed
    //   bit 1 → v_reversed
    //   bit 2 → swapped (transpose u ↔ v)
    //
    // Legacy heuristic fallback (`orientation = None`):
    //   * `swapped` is inferred from extent-shape mismatch — works for
    //     non-square faces, but always returns `false` for square N×N.
    //   * `f2_u_reversed`/`f2_v_reversed` come from the lb>ub flip in
    //     each axis of FaceRecord.
    let f2_raw = [
        [face2.il, face2.ih],
        [face2.jl, face2.jh],
        [face2.kl, face2.kh],
    ];
    let (swapped, f2_u_reversed, f2_v_reversed) = match orientation {
        Some(o) => {
            let pi = o.permutation_index;
            (
                (pi & 0b100) != 0, // bit 2: swap
                (pi & 0b001) != 0, // bit 0: u_reversed
                (pi & 0b010) != 0, // bit 1: v_reversed
            )
        }
        None => {
            // Legacy heuristic — kept for callers that haven't run
            // the cascade verifier (e.g. unit tests with no orientation).
            let swap = (n_u1 == n_v2)
                && (n_v1 == n_u2)
                && !((n_u1 == n_u2) && (n_v1 == n_v2));
            let u_rev = f2_raw[var_axes2[0]][0] > f2_raw[var_axes2[0]][1];
            let v_rev = f2_raw[var_axes2[1]][0] > f2_raw[var_axes2[1]][1];
            (swap, u_rev, v_rev)
        }
    };

    let (nci1, ncj1, _) = graph.block_cell_dims[b1];
    let (nci2, ncj2, _) = graph.block_cell_dims[b2];

    for v in 0..n_v1 {
        for u in 0..n_u1 {
            // Block1 cell
            let mut ijk1 = [0usize; 3];
            ijk1[axis1] = cell1_const;
            ijk1[var_axes1[0]] = f1_lo[var_axes1[0]] + u;
            ijk1[var_axes1[1]] = f1_lo[var_axes1[1]] + v;

            // Block2 cell (with orientation mapping)
            let (u2, v2) = if swapped { (v, u) } else { (u, v) };
            let u2_mapped = if f2_u_reversed { n_u2 - 1 - u2 } else { u2 };
            let v2_mapped = if f2_v_reversed { n_v2 - 1 - v2 } else { v2 };

            let mut ijk2 = [0usize; 3];
            ijk2[axis2] = cell2_const;
            ijk2[var_axes2[0]] = f2_lo[var_axes2[0]] + u2_mapped;
            ijk2[var_axes2[1]] = f2_lo[var_axes2[1]] + v2_mapped;

            let gid1 = graph.block_offset[b1]
                + cell_index(ijk1[0], ijk1[1], ijk1[2], nci1, ncj1);
            let gid2 = graph.block_offset[b2]
                + cell_index(ijk2[0], ijk2[1], ijk2[2], nci2, ncj2);

            // Face area vector from block1's metrics at the interface face.
            // For the high-side face, the area vector already points in the +axis
            // direction (toward block2). For the low-side face, we negate.
            let (mut ax, mut ay, mut az) = boundary_face_area(
                axis1, f1_const_val, ijk1, blk1, fm1,
            );
            if !is_high1 {
                // Face at low end of block1: outward from block1 is the -axis
                // direction, which means toward block2. The raw area vector points
                // in +axis, so we negate it to get the outward direction.
                ax = -ax;
                ay = -ay;
                az = -az;
            }

            // Face centroid (a position — no sign flip for low-end faces).
            let (cx, cy, cz) = boundary_face_centroid(
                axis1, f1_const_val, ijk1, blk1, fm1,
            );

            result.push((gid1 as u32, gid2 as u32, ax, ay, az, cx, cy, cz));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::face_record::{FaceMatch, FaceRecord};

    /// Build a uniform block spanning [x0, x1] x [y0, y1] x [z0, z1].
    fn uniform_block(
        ni: usize, nj: usize, nk: usize,
        x0: f64, x1: f64, y0: f64, y1: f64, z0: f64, z1: f64,
    ) -> Block {
        let n = ni * nj * nk;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let dx = if ni > 1 { (x1 - x0) / (ni as f64 - 1.0) } else { 0.0 };
        let dy = if nj > 1 { (y1 - y0) / (nj as f64 - 1.0) } else { 0.0 };
        let dz = if nk > 1 { (z1 - z0) / (nk as f64 - 1.0) } else { 0.0 };
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    x.push(x0 + i as f64 * dx);
                    y.push(y0 + j as f64 * dy);
                    z.push(z0 + k as f64 * dz);
                }
            }
        }
        Block::new(ni, nj, nk, x, y, z)
    }

    #[test]
    fn test_single_block_flat_mesh() {
        // 3x3x3 nodes = 2x2x2 = 8 cells
        let blk = uniform_block(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        // All 6 block faces are outer boundaries
        let outer_faces = vec![
            // imin face
            FaceRecord { block_index: 0, il: 0, jl: 0, kl: 0, ih: 0, jh: 2, kh: 2, id: Some(1), u_physical: None, v_physical: None },
            // imax face
            FaceRecord { block_index: 0, il: 2, jl: 0, kl: 0, ih: 2, jh: 2, kh: 2, id: Some(2), u_physical: None, v_physical: None },
            // jmin face
            FaceRecord { block_index: 0, il: 0, jl: 0, kl: 0, ih: 2, jh: 0, kh: 2, id: Some(3), u_physical: None, v_physical: None },
            // jmax face
            FaceRecord { block_index: 0, il: 0, jl: 2, kl: 0, ih: 2, jh: 2, kh: 2, id: Some(4), u_physical: None, v_physical: None },
            // kmin face
            FaceRecord { block_index: 0, il: 0, jl: 0, kl: 0, ih: 2, jh: 2, kh: 0, id: Some(5), u_physical: None, v_physical: None },
            // kmax face
            FaceRecord { block_index: 0, il: 0, jl: 0, kl: 2, ih: 2, jh: 2, kh: 2, id: Some(6), u_physical: None, v_physical: None },
        ];

        let mesh = build_flat_mesh(&[blk], &[], &outer_faces);

        assert_eq!(mesh.n_cells, 8);
        assert_eq!(mesh.cell_volume.len(), 8);

        // Each cell volume should be 0.125 (unit cube divided into 8)
        for v in &mesh.cell_volume {
            assert!((v - 0.125).abs() < 1e-10, "Expected 0.125, got {}", v);
        }

        // Total volume
        let total_vol: f64 = mesh.cell_volume.iter().sum();
        assert!((total_vol - 1.0).abs() < 1e-10);

        // Interior faces: 1 interior I-face per j,k pair (2x2=4) + similarly
        // for J and K = 4 + 4 + 4 = 12 interior faces
        // Boundary faces: 4 per outer face x 6 faces = 24
        // Total: 12 + 24 = 36
        let n_boundary = mesh.face_neighbor.iter().filter(|&&n| n < 0).count();
        assert_eq!(n_boundary, 24, "Expected 24 boundary faces, got {}", n_boundary);
        let n_interior = mesh.n_faces - n_boundary;
        assert_eq!(n_interior, 12, "Expected 12 interior faces, got {}", n_interior);

        // Stats should not panic
        let stats = mesh.stats();
        assert!(stats.contains("Cells:"));
    }

    #[test]
    fn test_two_block_flat_mesh() {
        // Two blocks abutting in x: block0 [0,1]^3, block1 [1,2] x [0,1]^2
        let blk0 = uniform_block(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let blk1 = uniform_block(3, 3, 3, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0);

        let fm = FaceMatch {
            block1: FaceRecord {
                block_index: 0, il: 2, jl: 0, kl: 0, ih: 2, jh: 2, kh: 2,
                id: None, u_physical: None, v_physical: None,
            },
            block2: FaceRecord {
                block_index: 1, il: 0, jl: 0, kl: 0, ih: 0, jh: 2, kh: 2,
                id: None, u_physical: None, v_physical: None,
            },
            points: vec![],
            orientation: None,
        };

        let mesh = build_flat_mesh(&[blk0, blk1], &[fm], &[]);

        assert_eq!(mesh.n_cells, 16);
        // Cross-block faces: 2x2 = 4
        // Verify that some faces have neighbors in the other block
        let cross_faces: Vec<_> = (0..mesh.n_faces)
            .filter(|&f| {
                let o = mesh.face_owner[f] as usize;
                let n = mesh.face_neighbor[f];
                if n < 0 { return false; }
                let n = n as usize;
                mesh.cell_block_id[o] != mesh.cell_block_id[n]
            })
            .collect();
        assert_eq!(cross_faces.len(), 4, "Expected 4 cross-block faces");
    }
}
