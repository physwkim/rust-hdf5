#![allow(dead_code)]
//! I/O engine for the pure Rust HDF5 library.
//!
//! Provides buffered file I/O, append-only allocation, dataset reading/writing,
//! and SWMR (Single Writer Multiple Reader) protocol support.

pub mod allocator;
pub(crate) mod chunk_grid;
pub mod file_handle;
pub(crate) mod free_space_io;
pub(crate) mod hyperslab;
pub mod locking;
pub(crate) mod object_header_io;
pub mod reader;
pub mod swmr;
pub(crate) mod symbol_table_io;
pub mod writer;

pub use reader::Hdf5Reader;
pub use swmr::SwmrWriter;
pub use writer::Hdf5Writer;

/// The decode parameters that belong to a *file* rather than to any one
/// object in it.
///
/// Address/length widths come from the superblock, the v1-B-tree "K" ranks
/// from the superblock extension when it carries them, and the shared-message
/// master table says which fractal heap a shared message ID belongs to. They
/// travel together because reading any object header can need all three: a
/// symbol-table group needs the ranks, a shared message needs the table, and
/// everything needs the widths. Threading them as one value is what stops a
/// call site from reaching for a library default the file overrode.
#[derive(Debug, Clone)]
pub(crate) struct FileMeta {
    pub(crate) ctx: crate::format::FormatContext,
    pub(crate) btree: crate::format::btree_v1::BTreeV1Config,
    /// The SOHM master table, when the superblock extension names one. A
    /// message whose `H5O_MSG_FLAG_SHARED` bit is set stores a heap ID, and
    /// only this table says which fractal heap that ID belongs to.
    pub(crate) sohm: Option<crate::format::sohm::SohmMasterTable>,
}

#[derive(Debug)]
pub enum IoError {
    Io(std::io::Error),
    Format(crate::format::FormatError),
    NotFound(String),
    InvalidState(String),
    /// The object exists in the file but uses a feature this reader cannot
    /// decode. The string names the feature.
    Unsupported(String),
    /// A soft or external link whose target does not exist — `H5Dopen`/
    /// `H5Gopen` on a path through it fails, which is not the same as the name
    /// being absent. `target` is the link value: a path for a soft link,
    /// `file::path` for an external one.
    DanglingLink {
        link: String,
        target: String,
    },
    /// An external link whose target *file* could not be opened. `searched`
    /// lists the candidate paths tried, in the order
    /// `H5F_prefix_open_file` tries them, so a link that resolved on one
    /// machine and not another says where it looked.
    ExternalFileNotFound {
        link: String,
        file: String,
        searched: Vec<String>,
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
                "link '{}' points to '{}', which does not exist",
                link, target
            ),
            Self::ExternalFileNotFound {
                link,
                file,
                searched,
            } => write!(
                f,
                "external link '{}' names the file '{}', which could not be opened (tried: {})",
                link,
                file,
                searched.join(", ")
            ),
        }
    }
}

impl std::error::Error for IoError {}

pub type IoResult<T> = Result<T, IoError>;
