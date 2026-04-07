#![allow(dead_code, clippy::needless_range_loop, clippy::len_without_is_empty)]
//! Pure Rust Zstandard compressor.
//!
//! Produces valid zstd frames compatible with any standard decoder (ruzstd, libzstd).
//! Decompression is handled by the `ruzstd` crate (pure Rust).

pub mod bitstream;
pub mod constants;
pub mod fse;
pub mod huf;
pub mod compress;
#[cfg(feature = "zstandard")]
pub mod levels;

pub use compress::{compress, compress_to_vec};
