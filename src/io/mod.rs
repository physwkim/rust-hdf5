#![allow(dead_code)]
//! I/O engine for the pure Rust HDF5 library.
//!
//! Provides buffered file I/O, append-only allocation, dataset reading/writing,
//! and SWMR (Single Writer Multiple Reader) protocol support.

pub mod allocator;
pub(crate) mod chunk_grid;
pub mod file_handle;
pub(crate) mod hyperslab;
pub mod locking;
pub(crate) mod object_header_io;
pub mod reader;
pub mod swmr;
pub mod writer;

pub use reader::Hdf5Reader;
pub use swmr::SwmrWriter;
pub use writer::Hdf5Writer;

#[derive(Debug)]
pub enum IoError {
    Io(std::io::Error),
    Format(crate::format::FormatError),
    NotFound(String),
    InvalidState(String),
    /// The object exists in the file but uses a feature this reader cannot
    /// decode. The string names the feature.
    Unsupported(String),
    /// A soft link whose target does not exist — `H5Dopen`/`H5Gopen` on a
    /// path through it fails, which is not the same as the name being absent.
    DanglingLink {
        link: String,
        target: String,
    },
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<crate::format::FormatError> for IoError {
    fn from(e: crate::format::FormatError) -> Self {
        Self::Format(e)
    }
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Format(e) => write!(f, "format error: {}", e),
            Self::NotFound(s) => write!(f, "not found: {}", s),
            Self::InvalidState(s) => write!(f, "invalid state: {}", s),
            Self::Unsupported(s) => write!(f, "unsupported: {}", s),
            Self::DanglingLink { link, target } => write!(
                f,
                "soft link '{}' points to '{}', which does not exist",
                link, target
            ),
        }
    }
}

impl std::error::Error for IoError {}

pub type IoResult<T> = Result<T, IoError>;
