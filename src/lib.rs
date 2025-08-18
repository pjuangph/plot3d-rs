pub mod block;
pub mod utils;
pub mod read;
pub mod write;

pub use block::Block;
pub use utils::Endian; // <- works now because Endian is pub
pub use read::{read_plot3d_ascii, read_plot3d_binary, BinaryFormat, FloatPrecision};
pub use write::write_plot3d;
