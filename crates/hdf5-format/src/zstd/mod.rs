#![allow(dead_code, clippy::needless_range_loop, clippy::len_without_is_empty)]
//! Pure Rust Zstandard codec — compress + decompress, zero external dependencies.

pub mod bitstream;
pub mod constants;
pub mod fse;
pub mod huf;
pub mod compress;
pub mod decode;

pub use compress::{compress, compress_to_vec};
pub use decode::decompress;
