//! I/O engine for the pure Rust HDF5 library.
//!
//! Provides buffered file I/O, append-only allocation, dataset reading/writing,
//! and SWMR (Single Writer Multiple Reader) protocol support.

pub mod file_handle;
pub mod allocator;
pub mod writer;
pub mod reader;
pub mod swmr;

pub use writer::Hdf5Writer;
pub use reader::Hdf5Reader;
pub use swmr::SwmrWriter;

#[derive(Debug)]
pub enum IoError {
    Io(std::io::Error),
    Format(hdf5_format::FormatError),
    NotFound(String),
    InvalidState(String),
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<hdf5_format::FormatError> for IoError {
    fn from(e: hdf5_format::FormatError) -> Self {
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
        }
    }
}

impl std::error::Error for IoError {}

pub type IoResult<T> = Result<T, IoError>;
