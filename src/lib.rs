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
//! The constant [`PERMUTATION_MATRICES`] holds the corresponding 2x2
//! matrices (one per index, 0 through 7). Each matrix transforms face2's
//! parametric coordinates to align with face1's.
//!
//! [`Orientation`] stores the index together with an [`OrientationPlane`]
//! tag indicating whether the match is **in-plane** (same constant axis)
//! or **cross-plane** (different constant axes, requiring a swap). The
//! connectivity pipeline populates `FaceMatch::orientation` automatically
//! so downstream code can reconstruct the exact node-to-node mapping
//! without re-sampling block coordinates.
//!
//! See the `face_record` module documentation for the full table and
//! usage examples.

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
    OrientationPlane, PERMUTATION_MATRICES, PeriodicPair,
};
pub use graph::{build_weighted_graph_from_face_matches, write_ddcmp, BlockGraph, WeightAggregate};
pub use merge_blocks::{
    combine_2_blocks_mixed_pairing, combine_blocks_mixed_pairs, combine_nxnxn_cubes_mixed_pairs,
};
pub use point_match::point_match;
pub use read::{read_ap_nasa, read_plot3d_ascii, read_plot3d_binary, BinaryFormat, FloatPrecision};
pub use rotational_periodicity::{
    count_rotated_corners_on_face, create_rotation_matrix, faces_support_any,
    faces_support_direction, linear_real_transform, periodicity_check_with_points,
    rotate_block_with_matrix, rotated_periodicity, rotational_periodicity,
};
pub use verification::{verify_connectivity, verify_periodicity};
pub use split_block::{split_blocks, SplitDirection};
pub use translational_periodicity::translational_periodicity;
pub use utils::{apply_rotation, compute_min_gcd, Endian};
pub use write::write_plot3d;
