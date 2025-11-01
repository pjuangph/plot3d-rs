//! Plot3D utilities for mesh connectivity, periodicity detection, and I/O.
//!
//! The crate deliberately mirrors the structure of the legacy Python tooling. For a walkthrough of
//! the rotational periodicity workflow refer to the integration test
//! `tests/test_rotational_periodicity.rs::rotational_periodicity_test`, which doubles as a usage
//! example in the generated documentation (`cargo doc --open`).

pub mod block;
pub mod block_face_functions;
pub mod connectivity;
pub mod merge_blocks;
pub mod read;
pub mod rotational_periodicity;
pub mod translational_periodicity;
pub mod utils;
pub mod write;

pub use block::Block;
pub use connectivity::{
    connectivity, connectivity_fast, FaceMatch, FaceMatchPrinter, FaceRecord, FaceRecordTraits,
    MatchPoint,
};
pub use merge_blocks::{
    combine_2_blocks_mixed_pairing, combine_blocks_mixed_pairs, combine_nxnxn_cubes_mixed_pairs,
};
pub use read::{read_plot3d_ascii, read_plot3d_binary, BinaryFormat, FloatPrecision};
pub use rotational_periodicity::{
    create_rotation_matrix, rotate_block_with_matrix, rotated_periodicity, rotational_periodicity,
    rotational_periodicity_fast, PeriodicPair,
};
pub use translational_periodicity::translational_periodicity;
pub use utils::Endian; // <- works now because Endian is pub
pub use write::write_plot3d;
