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
//! When two block faces meet at an interface, their (i, j, k) index
//! systems may be flipped, transposed, or both. There are **8 valid
//! orientations** for a face pair, arising from 3 independent binary
//! choices:
//!
//! - **u_reversed**: traversal along the first varying axis is flipped
//! - **v_reversed**: traversal along the second varying axis is flipped
//! - **swapped**: the two varying axes trade places (cross-plane match)
//!
//! ## The 8 Orientations (2×2 Representation)
//!
//! The constant [`PERMUTATION_MATRICES`] holds 8 canonical 2×2 matrices
//! that operate on the abstract parametric (u, v) coordinates. The index
//! is computed as:
//!
//! ```text
//! permutation_index = u_reversed | (v_reversed << 1) | (swapped << 2)
//! ```
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
//! ## 3×3 Permutation Matrix (Primary Representation)
//!
//! The [`Orientation`] struct stores a **3×3 signed permutation matrix**
//! `M` (entries in {-1, 0, +1}) that directly maps block1's (i, j, k)
//! indices to block2's:
//!
//! ```text
//! [i2]         [i1 - lb1_i]
//! [j2] = lb2 + M * [j1 - lb1_j]
//! [k2]         [k1 - lb1_k]
//! ```
//!
//! The 3×3 matrix is computed from the 2×2 matrix by embedding it in
//! the varying-axis subspace, using the known constant axes of both
//! faces. This eliminates the need to separately identify the constant
//! axis, extract (u, v), apply a 2×2 matrix, then reconstruct (i, j, k).
//!
//! ### Example 3×3 matrices
//!
//! **In-plane** (both faces K-constant, u=i, v=j):
//!
//! | Perm | 2×2 | 3×3 |
//! |:----:|:---:|:---:|
//! | 0: identity | `[[1,0],[0,1]]` | `[[1,0,0],[0,1,0],[0,0,1]]` |
//! | 1: flip u | `[[-1,0],[0,1]]` | `[[-1,0,0],[0,1,0],[0,0,1]]` |
//! | 3: flip both | `[[-1,0],[0,-1]]` | `[[-1,0,0],[0,-1,0],[0,0,1]]` |
//!
//! **Cross-plane** (face1 K-constant, face2 J-constant):
//!
//! | Perm | 2×2 | 3×3 |
//! |:----:|:---:|:---:|
//! | 4: swap | `[[0,1],[1,0]]` | `[[1,0,0],[0,0,1],[0,1,0]]` |
//!
//! ## In-plane vs Cross-plane
//!
//! - **In-plane** (perm 0-3): both faces have the same constant axis.
//!   The 3×3 matrix has a +1 on the constant-axis diagonal.
//! - **Cross-plane** (perm 4-7): faces have different constant axes.
//!   The 3×3 matrix has off-diagonal entries mapping one constant axis
//!   to another.
//!
//! [`OrientationPlane`] is derived from the matrix via
//! [`Orientation::plane()`]. The legacy `permutation_index` (0-7) is
//! available via [`Orientation::permutation_index()`] for serialization.
//!
//! # Verification Pipeline
//!
//! After computing connectivity or periodicity, use the verification
//! functions in the [`verification`] module:
//!
//! 1. [`verify_connectivity`] — extracts a canonical 2D grid from each
//!    face pair, tries all 8 permutation matrices, and picks the one
//!    that aligns nodes point-by-point within tolerance. Sets the 3×3
//!    [`Orientation`] matrix on each verified match.
//!
//! 2. [`verify_periodicity`] — same approach but rotates block1's face
//!    by the periodicity angle before comparing grids.
//!
//! 3. [`align_face_orientations`] — for same-dimension matches,
//!    walks all 8 orientations to find the one where node-by-node
//!    traversal matches.
//!
//! The recommended pipeline (as used by the `connectivity_finder` binary):
//!
//! ```text
//! connectivity_fast → face_matches_to_dict → verify_connectivity
//!   → align_face_orientations → rotated_periodicity → verify_periodicity
//! ```
//!
//! # JSON Output Formats
//!
//! The [`serialization`] module provides two JSON output formats:
//!
//! **Default (`lo`/`hi`)** — ascending bounds with `permutation_index` (0-7):
//!
//! ```json
//! {
//!   "block1": { "block_index": 0, "lo": [0,0,0], "hi": [0,101,33] },
//!   "block2": { "block_index": 30, "lo": [0,0,0], "hi": [0,101,33] },
//!   "permutation_index": 3
//! }
//! ```
//!
//! **Diagonal (`lb`/`ub`)** — GlennHT-compatible format. In-plane matches
//! (perm 0-3) encode direction in block2's `lb`/`ub` with
//! `permutation_index: -1`. Cross-plane matches (perm 4-7) use ascending
//! `lb`/`ub` with the actual `permutation_index`.
//!
//! ```json
//! {
//!   "block1": { "block_index": 0, "lb": [0,0,0], "ub": [0,101,33] },
//!   "block2": { "block_index": 30, "lb": [0,101,33], "ub": [0,0,0] },
//!   "permutation_index": -1
//! }
//! ```
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
pub mod face_pool;
pub mod face_record;
pub(crate) mod geometry;
pub mod graph;
pub mod merge_blocks;
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
    face_match_to_diagonal_json, face_match_to_json, face_record_to_diagonal_json,
    face_record_to_json, orientation_matrix_json, permutation_matrices_json,
};
pub use split_block::{split_blocks, SplitDirection};
pub use translational_periodicity::translational_periodicity;
pub use utils::{apply_rotation, compute_min_gcd, Endian};
pub use verification::{
    apply_permutation, determine_plane, extract_canonical_grid, try_all_permutations,
    verify_connectivity, verify_match, verify_partial_match, verify_periodicity,
};
pub use write::write_plot3d;
