pub mod block;
pub mod block_face_functions;
pub mod connectivity;
pub mod read;
pub mod rotational_periodicity;
pub mod translational_periodicity;
pub mod merge_blocks;
pub mod utils;
pub mod write;

pub use block::Block;
pub use connectivity::{
    connectivity, connectivity_fast, FaceMatch, FaceMatchPrinter, FaceRecord, FaceRecordTraits,
    MatchPoint,
};
pub use read::{read_plot3d_ascii, read_plot3d_binary, BinaryFormat, FloatPrecision};
pub use rotational_periodicity::{
    create_rotation_matrix, periodicity, periodicity_fast, rotate_block_with_matrix,
    rotated_periodicity, PeriodicPair, PeriodicPairExport,
};
pub use translational_periodicity::{
    translational_periodicity, TranslationalPair, TranslationalPairExport,
};
pub use merge_blocks::{
    combine_2_blocks_mixed_pairing, combine_blocks_mixed_pairs, combine_nxnxn_cubes_mixed_pairs,
};
pub use utils::Endian; // <- works now because Endian is pub
pub use write::write_plot3d;
