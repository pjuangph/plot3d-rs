//! Cell-level dual graph for finite-volume discretization on multi-block
//! structured grids.
//!
//! In a finite-volume CFD solver, the unknowns live at cell centers and fluxes
//! cross cell faces. A multi-block structured grid stores cells in per-block
//! (i,j,k) arrays, but solvers (especially GPU solvers) need a single global
//! numbering. This module builds that mapping.
//!
//! # Global Cell Numbering
//!
//! All cells across all blocks are flattened into a contiguous global ID space:
//!
//! ```text
//!   global_id = block_offset[b] + local_cell_index(i, j, k)
//! ```
//!
//! where the local cell index within a block is:
//!
//! ```text
//!   local = i + nci * j + nci * ncj * k
//! ```
//!
//! with `nci = imax - 1`, `ncj = jmax - 1`, `nck = kmax - 1` being the cell
//! counts along each axis (one fewer than node counts).
//!
//! # Dual Graph Edges
//!
//! Each shared face between two cells creates a graph edge. There are two kinds:
//!
//! - **Interior edges**: within a single block, cell `(i,j,k)` shares an
//!   I-face with cell `(i+1,j,k)`, a J-face with `(i,j+1,k)`, and a K-face
//!   with `(i,j,k+1)`. These are implicit from the structured topology and do
//!   not need explicit storage.
//!
//! - **Cross-block edges**: where two blocks abut at a shared interface
//!   (described by a [`FaceMatch`]). The boundary cells on block1 connect to
//!   the corresponding boundary cells on block2. These edges must be stored
//!   explicitly because they break the structured pattern.

use crate::block::Block;
use crate::face_record::FaceMatch;

/// Cell-level dual graph for finite-volume discretization.
///
/// Provides the global numbering scheme and cross-block connectivity needed
/// to flatten a multi-block structured grid into a single unstructured-like
/// cell graph suitable for GPU solvers.
#[derive(Clone, Debug)]
pub struct CellGraph {
    /// Total number of cells across all blocks.
    pub n_cells: usize,
    /// Starting global cell ID for each block. Length = number of blocks.
    /// `block_offset[b]` is the global ID of cell (0,0,0) in block `b`.
    pub block_offset: Vec<usize>,
    /// Cell dimensions per block: `(nci, ncj, nck)` where `nci = imax - 1`, etc.
    /// Length = number of blocks.
    pub block_cell_dims: Vec<(usize, usize, usize)>,
    /// Cross-block face connections as `(global_cell_a, global_cell_b)` pairs.
    /// Each pair represents two cells in different blocks that share a face
    /// at a block-block interface. Interior (within-block) adjacency is implicit.
    pub cross_block_edges: Vec<(usize, usize)>,
}

/// Compute the local (within-block) flat cell index from structured indices.
///
/// The indexing convention is i-fastest (row-major in Fortran terms):
///
/// ```text
///   cell_index = i + nci * j + nci * ncj * k
/// ```
///
/// where `nci` and `ncj` are the number of cells (not nodes) in the i and j
/// directions respectively.
///
/// # Panics
///
/// Debug-mode assertions check that `i < nci` and `j < ncj`.
#[inline]
pub fn cell_index(i: usize, j: usize, k: usize, nci: usize, ncj: usize) -> usize {
    debug_assert!(i < nci, "cell i={} out of range [0, {})", i, nci);
    debug_assert!(j < ncj, "cell j={} out of range [0, {})", j, ncj);
    (k * ncj + j) * nci + i
}

/// Convert a block-local cell `(i, j, k)` to a global cell ID using the graph's
/// offset table.
///
/// This is the central mapping used throughout the flat data conversion:
///
/// ```text
///   global_id = block_offset[block] + i + nci * j + nci * ncj * k
/// ```
#[inline]
pub fn global_cell_id(block: usize, i: usize, j: usize, k: usize, graph: &CellGraph) -> usize {
    let (nci, ncj, _) = graph.block_cell_dims[block];
    graph.block_offset[block] + cell_index(i, j, k, nci, ncj)
}

/// Build the cell-level dual graph from a set of structured blocks and their
/// face connectivity.
///
/// # Algorithm
///
/// 1. **Compute block offsets**: scan through blocks, accumulating cell counts
///    `(imax-1) * (jmax-1) * (kmax-1)` to assign each block a starting global
///    cell ID.
///
/// 2. **Build cross-block edges**: for each [`FaceMatch`], identify the constant
///    axis on each side (the axis where `il == ih`), then iterate over the
///    boundary cells adjacent to that face. Each boundary cell on block1 is
///    paired with the corresponding cell on block2.
///
///    The FaceMatch stores index ranges `[il..ih, jl..jh, kl..kh]` on each
///    block's face. The constant axis determines which cells are "boundary"
///    (adjacent to the interface). For example, if block1's face is at `i = 0`,
///    the boundary cells are at cell-i = 0; if at `i = imax-1`, the cells are
///    at cell-i = imax-2.
///
/// # Arguments
///
/// * `blocks` - Slice of all blocks in the multi-block grid.
/// * `face_matches` - Slice of face matches describing block-block interfaces.
///
/// # Returns
///
/// A [`CellGraph`] containing the global numbering and cross-block edges.
pub fn build_cell_graph(blocks: &[Block], face_matches: &[FaceMatch]) -> CellGraph {
    let n_blocks = blocks.len();

    // --- Step 1: compute block cell dimensions and cumulative offsets ---
    let mut block_cell_dims = Vec::with_capacity(n_blocks);
    let mut block_offset = Vec::with_capacity(n_blocks);
    let mut cumulative = 0usize;

    for blk in blocks {
        let nci = blk.imax - 1;
        let ncj = blk.jmax - 1;
        let nck = blk.kmax - 1;
        block_cell_dims.push((nci, ncj, nck));
        block_offset.push(cumulative);
        cumulative += nci * ncj * nck;
    }
    let n_cells = cumulative;

    // --- Step 2: build cross-block edges from face matches ---
    let mut cross_block_edges = Vec::new();

    for fm in face_matches {
        let b1 = fm.block1.block_index;
        let b2 = fm.block2.block_index;

        // Determine the constant axis and cell ranges for each side.
        // A face is on a boundary plane where one index is constant
        // (il == ih for that axis in the normalized sense).
        let edges = build_cross_block_face_edges(
            b1,
            &fm.block1,
            &blocks[b1],
            b2,
            &fm.block2,
            &blocks[b2],
            &block_cell_dims,
            &block_offset,
        );
        cross_block_edges.extend(edges);
    }

    CellGraph {
        n_cells,
        block_offset,
        block_cell_dims,
        cross_block_edges,
    }
}

/// Determine which cell is adjacent to a block boundary face.
///
/// Given a face on a constant-axis plane at node index `face_val` with block
/// node count `n_nodes` along that axis:
/// - If `face_val == 0`, the adjacent cell is at cell index 0 along that axis.
/// - If `face_val == n_nodes - 1`, the adjacent cell is at cell index `n_nodes - 2`.
///
/// Returns `None` if the face is not on a boundary plane (internal face).
fn boundary_cell_index(face_val: usize, n_nodes: usize) -> Option<usize> {
    if face_val == 0 {
        Some(0)
    } else if face_val == n_nodes - 1 {
        Some(n_nodes - 2)
    } else {
        // Not a boundary face -- the face_val is interior
        None
    }
}

/// Build cross-block edges for a single FaceMatch.
///
/// This function identifies the constant axis on each block's face, then
/// iterates over the 2D grid of boundary cells on the interface. For each
/// (u,v) position on the face, it computes the global cell IDs on both sides
/// and emits an edge.
///
/// The face ranges from FaceRecord use node indices. The boundary cell along
/// the constant axis is determined by whether the face is at the low or high
/// end of that axis.
fn build_cross_block_face_edges(
    b1: usize,
    face1: &crate::face_record::FaceRecord,
    blk1: &Block,
    b2: usize,
    face2: &crate::face_record::FaceRecord,
    blk2: &Block,
    block_cell_dims: &[(usize, usize, usize)],
    block_offset: &[usize],
) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    // Determine the constant axis for face1 (the axis where lo == hi)
    let f1_const_axis = face1.constant_axis();
    let f2_const_axis = face2.constant_axis();

    // Both faces must have a constant axis for a valid face match
    let (axis1, axis2) = match (f1_const_axis, f2_const_axis) {
        (Some(a1), Some(a2)) => (a1, a2),
        _ => return edges, // Degenerate match, skip
    };

    // Get the constant-axis node value and determine which cell layer is adjacent
    let f1_bounds = face1.bounds();
    let f2_bounds = face2.bounds();
    let f1_const_val = f1_bounds.0[axis1]; // lo == hi for the constant axis
    let f2_const_val = f2_bounds.0[axis2];

    let n_nodes1 = [blk1.imax, blk1.jmax, blk1.kmax];
    let n_nodes2 = [blk2.imax, blk2.jmax, blk2.kmax];

    let cell1_const = match boundary_cell_index(f1_const_val, n_nodes1[axis1]) {
        Some(c) => c,
        None => return edges,
    };
    let cell2_const = match boundary_cell_index(f2_const_val, n_nodes2[axis2]) {
        Some(c) => c,
        None => return edges,
    };

    // The two varying axes on face1 define the 2D iteration.
    // For each cell on the face, we need to map from face1's varying axes
    // to face2's varying axes, accounting for orientation.
    //
    // The cell ranges along the varying axes come from the node ranges:
    // if node range is [lo..hi], cell range is [lo..hi-1] (hi-1 cells).
    let var_axes1: Vec<usize> = (0..3).filter(|&a| a != axis1).collect();
    let var_axes2: Vec<usize> = (0..3).filter(|&a| a != axis2).collect();

    // Cell ranges on face1's varying axes
    let f1_lo = [face1.i_lo(), face1.j_lo(), face1.k_lo()];
    let f1_hi = [face1.i_hi(), face1.j_hi(), face1.k_hi()];

    // Number of cells along each varying axis on face1
    let n_u1 = f1_hi[var_axes1[0]] - f1_lo[var_axes1[0]];
    let n_v1 = f1_hi[var_axes1[1]] - f1_lo[var_axes1[1]];

    // Cell ranges on face2's varying axes
    let f2_lo = [face2.i_lo(), face2.j_lo(), face2.k_lo()];
    let f2_hi = [face2.i_hi(), face2.j_hi(), face2.k_hi()];

    let n_u2 = f2_hi[var_axes2[0]] - f2_lo[var_axes2[0]];
    let n_v2 = f2_hi[var_axes2[1]] - f2_lo[var_axes2[1]];

    if n_u1 == 0 || n_v1 == 0 {
        return edges; // Degenerate face (edge or point match)
    }

    // Check dimension compatibility: the product of varying dimensions must match.
    // With orientation, axes may be swapped, so (n_u1, n_v1) might match
    // (n_v2, n_u2) instead of (n_u2, n_v2).
    let swapped = (n_u1 == n_v2) && (n_v1 == n_u2) && !((n_u1 == n_u2) && (n_v1 == n_v2));

    // Determine if face2's varying axes are reversed relative to face1.
    // We use the raw (non-normalized) il/ih etc. from face2 to detect reversal.
    let f2_raw = [
        [face2.il, face2.ih],
        [face2.jl, face2.jh],
        [face2.kl, face2.kh],
    ];
    let f2_u_reversed = f2_raw[var_axes2[0]][0] > f2_raw[var_axes2[0]][1];
    let f2_v_reversed = f2_raw[var_axes2[1]][0] > f2_raw[var_axes2[1]][1];

    let (nci1, ncj1, _) = block_cell_dims[b1];
    let (nci2, ncj2, _) = block_cell_dims[b2];

    for v in 0..n_v1 {
        for u in 0..n_u1 {
            // Cell indices on block1
            let mut ijk1 = [0usize; 3];
            ijk1[axis1] = cell1_const;
            ijk1[var_axes1[0]] = f1_lo[var_axes1[0]] + u;
            ijk1[var_axes1[1]] = f1_lo[var_axes1[1]] + v;

            // Map (u, v) to block2's coordinate system, accounting for
            // possible axis swap and reversal
            let (u2, v2) = if swapped { (v, u) } else { (u, v) };

            let mut ijk2 = [0usize; 3];
            ijk2[axis2] = cell2_const;

            // Apply reversal if needed
            let u2_mapped = if f2_u_reversed {
                n_u2 - 1 - u2
            } else {
                u2
            };
            let v2_mapped = if f2_v_reversed {
                n_v2 - 1 - v2
            } else {
                v2
            };

            ijk2[var_axes2[0]] = f2_lo[var_axes2[0]] + u2_mapped;
            ijk2[var_axes2[1]] = f2_lo[var_axes2[1]] + v2_mapped;

            let gid1 = block_offset[b1] + cell_index(ijk1[0], ijk1[1], ijk1[2], nci1, ncj1);
            let gid2 = block_offset[b2] + cell_index(ijk2[0], ijk2[1], ijk2[2], nci2, ncj2);

            edges.push((gid1, gid2));
        }
    }

    edges
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::face_record::{FaceMatch, FaceRecord};

    /// Build a uniform block spanning [x0, x1] x [y0, y1] x [z0, z1]
    /// with `ni x nj x nk` nodes.
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
    fn test_single_block_graph() {
        let block = uniform_block(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let graph = build_cell_graph(&[block], &[]);
        // 2x2x2 = 8 cells
        assert_eq!(graph.n_cells, 8);
        assert_eq!(graph.block_offset, vec![0]);
        assert_eq!(graph.block_cell_dims, vec![(2, 2, 2)]);
        assert!(graph.cross_block_edges.is_empty());
    }

    #[test]
    fn test_two_blocks_with_interface() {
        // Two 3x3x3 blocks abutting at the i-max/i-min interface.
        // Block 0: x in [0,1], Block 1: x in [1,2]
        let blk0 = uniform_block(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let blk1 = uniform_block(3, 3, 3, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0);

        // Face match: block0 imax face (i=2) matches block1 imin face (i=0)
        let fm = FaceMatch {
            block1: FaceRecord {
                block_index: 0,
                il: 2, jl: 0, kl: 0,
                ih: 2, jh: 2, kh: 2,
                id: None,
                u_physical: None,
                v_physical: None,
            },
            block2: FaceRecord {
                block_index: 1,
                il: 0, jl: 0, kl: 0,
                ih: 0, jh: 2, kh: 2,
                id: None,
                u_physical: None,
                v_physical: None,
            },
            points: vec![],
            orientation: None,
        };

        let graph = build_cell_graph(&[blk0, blk1], &[fm]);
        assert_eq!(graph.n_cells, 16); // 8 + 8
        assert_eq!(graph.block_offset, vec![0, 8]);
        // Cross-block edges: 2x2 = 4 cell pairs at the interface
        assert_eq!(graph.cross_block_edges.len(), 4);
    }

    #[test]
    fn test_cell_index_consistency() {
        assert_eq!(cell_index(0, 0, 0, 3, 4), 0);
        assert_eq!(cell_index(1, 0, 0, 3, 4), 1);
        assert_eq!(cell_index(0, 1, 0, 3, 4), 3);
        assert_eq!(cell_index(0, 0, 1, 3, 4), 12);
    }

    #[test]
    fn test_global_cell_id() {
        let blk0 = uniform_block(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let blk1 = uniform_block(4, 3, 3, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0);
        let graph = build_cell_graph(&[blk0, blk1], &[]);
        // Block 0 has 2*2*2 = 8 cells
        assert_eq!(graph.block_offset[1], 8);
        // Cell (0,0,0) in block 1 should have global ID 8
        assert_eq!(global_cell_id(1, 0, 0, 0, &graph), 8);
        // Cell (1,1,1) in block 0
        assert_eq!(global_cell_id(0, 1, 1, 1, &graph), 1 + 2 + 4);
    }
}
