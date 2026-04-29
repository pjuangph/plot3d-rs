//! Plot3D utilities for mesh connectivity, periodicity detection, and I/O.
//!
//! The crate deliberately mirrors the structure of the legacy Python tooling. For a walkthrough of
//! the rotational periodicity workflow refer to the integration test
//! `tests/test_rotational_periodicity.rs::rotational_periodicity_test`, which doubles as a usage
//! example in the generated documentation (`cargo doc --open`).
//!
//! # Diagonal Convention (FaceRecord)
//!
//! [`FaceRecord`] uses `il/jl/kl` and `ih/jh/kh` to describe the two diagonal
//! corners of a face on a block. These are **not** guaranteed to satisfy
//! `il <= ih`; the ordering encodes **orientation**. When `il > ih`, the
//! I-axis is reversed on that face relative to the matching face on the
//! other block.
//!
//! This matches the GridPro/GlennHT connectivity convention where
//! `IMIN,JMIN,KMIN → IMAX,JMAX,KMAX` are diagonal corners and reversed
//! indices encode face orientation.
//!
//! Use the normalized accessors `i_lo()/i_hi()` when you need min/max values
//! for range iteration or face reconstruction.
//!
//! # Orientation & Permutation System
//!
//! When two block faces meet at an interface, their parametric (u, v)
//! coordinate systems may be flipped, transposed, or both. The crate
//! encodes all 8 valid orientations as a 3-bit index:
//!
//! ```text
//! permutation_index = u_reversed | (v_reversed << 1) | (swapped << 2)
//! ```
//!
//! The constant [`PERMUTATION_MATRICES`] holds the corresponding 2×2
//! matrices (one per index, 0 through 7). Each matrix transforms face2's
//! parametric coordinates to align with face1's.
//!
//! | Index | Binary | u_rev | v_rev | swap | Matrix | Effect |
//! |:-----:|:------:|:-----:|:-----:|:----:|:------:|:------:|
//! | 0 | `000` | no | no | no | `[[ 1, 0],[ 0, 1]]` | identity |
//! | 1 | `001` | yes | no | no | `[[-1, 0],[ 0, 1]]` | flip u |
//! | 2 | `010` | no | yes | no | `[[ 1, 0],[ 0,-1]]` | flip v |
//! | 3 | `011` | yes | yes | no | `[[-1, 0],[ 0,-1]]` | flip both |
//! | 4 | `100` | no | no | yes | `[[ 0, 1],[ 1, 0]]` | transpose |
//! | 5 | `101` | yes | no | yes | `[[ 0,-1],[ 1, 0]]` | transpose + flip u |
//! | 6 | `110` | no | yes | yes | `[[ 0, 1],[-1, 0]]` | transpose + flip v |
//! | 7 | `111` | yes | yes | yes | `[[ 0,-1],[-1, 0]]` | transpose + both |
//!
//! The `u` and `v` names are abstract parametric axes that map to concrete
//! i/j/k axes depending on which axis is constant on the face:
//!
//! | Constant axis | u (outer loop) | v (inner loop) |
//! |:-------------:|:--------------:|:--------------:|
//! | I-constant | j | k |
//! | J-constant | i | k |
//! | K-constant | i | j |
//!
//! [`Orientation`] stores the index together with an [`OrientationPlane`]
//! tag indicating whether the match is **in-plane** (same constant axis)
//! or **cross-plane** (different constant axes, requiring a swap). The
//! connectivity pipeline populates `FaceMatch::orientation` automatically
//! so downstream code can reconstruct the exact node-to-node mapping
//! without re-sampling block coordinates.
//!
//! # Verification Pipeline
//!
//! After computing connectivity or periodicity, use the verification
//! functions in the [`verification`] module:
//!
//! 1. [`verify_connectivity`] — extracts a canonical 2D grid from each
//!    face pair, tries all 8 permutation matrices, and picks the one
//!    that aligns nodes point-by-point within tolerance. Sets
//!    [`Orientation`] on each verified match.
//!
//! 2. [`verify_periodicity`] — same approach but rotates block1's face
//!    by the periodicity angle before comparing grids.
//!
//! 3. [`align_face_orientations`] — for same-dimension in-plane matches,
//!    walks all 8 diagonal orientations to find the one where directed
//!    I→J→K traversal matches node-by-node.
//!
//! The recommended pipeline (as used by the `connectivity_finder` binary):
//!
//! ```text
//! connectivity_fast → face_matches_to_dict → verify_connectivity
//!   → align_face_orientations → rotated_periodicity → verify_periodicity
//! ```
//!
//! # JSON Output Format
//!
//! The [`serialization`] module exports directed `lb`/`ub` corners
//! (raw `il/jl/kl` and `ih/jh/kh` — no ascending sort). After
//! [`face_matches_to_dict`], block1's `lb` physically matches block2's
//! `lb` in xyz space, and `ub` likewise.
//!
//! ```json
//! {
//!   "block1": { "block_index": 0, "lb": [0,0,0], "ub": [24,408,0] },
//!   "block2": { "block_index": 1, "lb": [408,0,0], "ub": [0,0,24] },
//!   "permutation_index": 5
//! }
//! ```
//!
//! The `permutation_index` (0-7) is relative to ascending canonical grids
//! (as computed by [`extract_canonical_grid`]). It is included as metadata;
//! the directed corners already encode the corner-to-corner mapping.
//!
//! [`permutation_matrices_json`] embeds the full 8-matrix array in the
//! JSON output header so consumers can reconstruct orientations without
//! hard-coding the table.

/// Floating-point precision type used throughout the crate.
/// Defaults to `f64`; enable the `f32` Cargo feature for single precision.
#[cfg(not(feature = "f32"))]
pub type Float = f64;
#[cfg(feature = "f32")]
pub type Float = f32;

#[cfg(feature = "f32")]
pub use std::f32::consts::PI;
/// Pi constant matching the active [`Float`] precision.
#[cfg(not(feature = "f32"))]
pub use std::f64::consts::PI;

pub mod block;
pub mod block_analysis;
pub mod block_face_functions;
pub mod connectivity;
pub mod cylindrical;
pub mod differencing;
pub mod dual_graph;
pub mod face_pool;
pub mod face_record;
pub mod flat_data;
pub(crate) mod geometry;
pub mod graph;
pub mod merge_blocks;
pub mod metrics;
pub mod point_match;
pub mod read;
pub mod rotational_periodicity;
pub mod split_block;
pub mod translational_periodicity;
pub mod utils;
pub mod verification;
pub mod write;

pub use block::{Block, FaceData};
pub use block_analysis::{
    block_connection_matrix, build_connectivity_graph, calculate_outward_normals,
    check_collinearity, find_bounding_faces, find_closest_block, get_outer_bounds,
    standardize_block_orientation, BlockConnectionOptions,
};
pub use block_face_functions::{
    create_face_from_diagonals, full_face_match, full_face_match_transformed, get_outer_faces,
    reduce_blocks, rotate_block, Face,
};
pub use connectivity::{
    align_face_orientations, connectivity, connectivity_fast, face_matches_to_dict,
    get_face_intersection,
};
pub use cylindrical::{find_angular_bounding_faces, to_radius, to_theta};
pub use differencing::{find_edges, find_face_edges, BlockDiff, FaceDiff};
pub use face_record::{
    FaceKey, FaceMatch, FaceMatchPrinter, FaceRecord, FaceRecordTraits, MatchPoint, Orientation,
    OrientationPlane, PeriodicPair, PERMUTATION_MATRICES,
};
pub use graph::{build_weighted_graph_from_face_matches, write_ddcmp, BlockGraph, WeightAggregate};
pub use metrics::{compute_cell_centers, compute_cell_volumes, compute_face_metrics, FaceMetrics};
pub use merge_blocks::{
    combine_2_blocks_mixed_pairing, combine_blocks_mixed_pairs, combine_nxnxn_cubes_mixed_pairs,
};
pub use point_match::point_match;
pub use read::{read_ap_nasa, read_plot3d_ascii, read_plot3d_binary, BinaryFormat, FloatPrecision};
pub use rotational_periodicity::{
    create_rotation_matrix, rotate_block_with_matrix, rotated_periodicity, rotational_periodicity,
};
pub mod serialization;
pub use serialization::{
    face_match_from_json, face_match_to_json, face_record_from_json, face_record_to_json,
    permutation_matrices_json,
};
pub use split_block::{split_blocks, SplitDirection};
pub use translational_periodicity::translational_periodicity;
pub use utils::{apply_rotation, compute_min_gcd, Endian};
pub use verification::{
    apply_permutation, determine_plane, extract_canonical_grid, try_all_permutations,
    verify_connectivity, verify_match, verify_partial_match, verify_periodicity,
    verify_translational_periodicity,
};
pub use write::write_plot3d;
pub use dual_graph::{build_cell_graph, cell_index, global_cell_id, CellGraph};
pub use flat_data::{build_flat_mesh, FlatMesh};
