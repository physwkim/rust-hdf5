//! Pure Rust HDF5 library with full read/write support.
//!
//! This crate provides a high-level API for creating and reading HDF5 files
//! without any C dependencies. It supports contiguous and chunked datasets,
//! deflate compression (with optional shuffle), SWMR streaming, and hyperslab
//! (slice) I/O.
//!
//! # Quick start
//!
//! ## Writing a dataset
//!
//! ```no_run
//! use hdf5::H5File;
//!
//! let file = H5File::create("output.h5").unwrap();
//! let ds = file.new_dataset::<f64>()
//!     .shape(&[100, 200])
//!     .create("matrix")
//!     .unwrap();
//! let data: Vec<f64> = (0..20_000).map(|i| i as f64).collect();
//! ds.write_raw(&data).unwrap();
//! file.close().unwrap();
//! ```
//!
//! ## Reading a dataset
//!
//! ```no_run
//! use hdf5::H5File;
//!
//! let file = H5File::open("output.h5").unwrap();
//! let ds = file.dataset("matrix").unwrap();
//! let data = ds.read_raw::<f64>().unwrap();
//! assert_eq!(data.len(), 20_000);
//! ```
//!
//! ## Chunked + compressed streaming
//!
//! ```no_run
//! use hdf5::H5File;
//!
//! let file = H5File::create("stream.h5").unwrap();
//! let ds = file.new_dataset::<f32>()
//!     .shape(&[0usize, 64])
//!     .chunk(&[1, 64])
//!     .max_shape(&[None, Some(64)])
//!     .deflate(6)
//!     .create("sensor")
//!     .unwrap();
//!
//! for frame in 0..1000u64 {
//!     let vals: Vec<f32> = (0..64).map(|i| (frame * 64 + i) as f32).collect();
//!     let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
//!     ds.write_chunk(frame as usize, &raw).unwrap();
//! }
//! ds.extend(&[1000, 64]).unwrap();
//! file.close().unwrap();
//! ```
//!
//! ## Hyperslab (slice) I/O
//!
//! ```no_run
//! use hdf5::H5File;
//!
//! // Write a sub-region
//! let file = H5File::create("slice.h5").unwrap();
//! let ds = file.new_dataset::<i32>()
//!     .shape(&[10usize, 10])
//!     .create("grid")
//!     .unwrap();
//! ds.write_raw(&[0i32; 100]).unwrap();
//! ds.write_slice(&[2, 3], &[3, 4], &[1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
//! file.close().unwrap();
//!
//! // Read a sub-region
//! let file = H5File::open("slice.h5").unwrap();
//! let ds = file.dataset("grid").unwrap();
//! let region = ds.read_slice::<i32>(&[2, 3], &[3, 4]).unwrap();
//! assert_eq!(region.len(), 12);
//! ```

pub mod error;
pub mod types;
pub mod file;
pub mod group;
pub mod dataset;
pub mod attribute;
pub mod swmr;

pub use error::{Hdf5Error, Result};
pub use file::H5File;
pub use group::H5Group;
pub use dataset::H5Dataset;
pub use attribute::H5Attribute;
pub use types::{CompoundType, Complex32, Complex64, H5Type, HBool, VarLenUnicode};
