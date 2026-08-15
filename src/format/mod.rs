//! Pure Rust HDF5 on-disk format codec.
//!
//! This crate handles encoding and decoding of HDF5 binary structures
//! (superblock, object headers, messages, chunk indices) without performing
//! any file I/O. It is used by `hdf5-io` and `hdf5` crates.

pub mod btree_v1;
pub(crate) mod bytes;
pub mod checksum;
pub mod chunk_index;
pub mod creation_order;
pub mod dense_attr;
pub mod dense_link;
pub mod fractal_heap;
pub mod fractal_heap_write;
pub mod global_heap;
pub mod local_heap;
pub mod messages;
pub mod nbit_scaleoffset;
pub mod object_header;
pub mod reference;
pub mod selection;
pub mod sohm;
pub mod sohm_write;
pub mod superblock;
pub mod symbol_table;
pub mod szip;

/// Format context carrying file-level encoding parameters
#[derive(Debug, Clone, Copy)]
pub struct FormatContext {
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
}

impl FormatContext {
    pub fn default_v3() -> Self {
        Self {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }
}

/// The low half of libhdf5's `H5Pset_libver_bounds` — the oldest library
/// release a file must stay readable by.
///
/// It is the file-wide switch that picks between on-disk message versions:
/// libhdf5 keeps one table per message type (`H5O_dtype_ver_bounds`,
/// `H5O_layout_ver_bounds`, ...) mapping the bound to the version it stamps.
/// Raising the bound lets the library use newer, tighter encodings; lowering
/// it keeps older readers able to open the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum LibverBound {
    /// `H5F_LIBVER_EARLIEST`, the default: every message at its oldest
    /// version that can express what the file holds.
    #[default]
    Earliest,
    /// `H5F_LIBVER_V18`.
    V18,
    /// `H5F_LIBVER_V110`.
    V110,
    /// `H5F_LIBVER_V112`.
    V112,
    /// `H5F_LIBVER_V114`.
    V114,
    /// `H5F_LIBVER_V200`, which is `H5F_LIBVER_LATEST` for libhdf5 2.0.
    V200,
}

impl LibverBound {
    /// The datatype message version this bound calls for — libhdf5's
    /// `H5O_dtype_ver_bounds` (H5T.c), the floor `H5T_set_version` raises a
    /// datatype to.
    pub fn dtype_version(self) -> u8 {
        match self {
            Self::Earliest => 1,
            Self::V18 | Self::V110 => 3,
            Self::V112 | Self::V114 => 4,
            Self::V200 => 5,
        }
    }

    /// The superblock version this bound calls for — libhdf5's
    /// `HDF5_superblock_ver_bounds` (H5Fsuper.c:68), the floor
    /// `H5F__super_init` raises the content-derived version to.
    ///
    /// Version 0 is the `H5F_LIBVER_EARLIEST` entry
    /// (`HDF5_SUPERBLOCK_VERSION_DEF`); a writer whose own structures need
    /// more takes the higher of the two.
    pub fn superblock_version(self) -> u8 {
        match self {
            Self::Earliest => 0,
            Self::V18 => 2,
            Self::V110 | Self::V112 | Self::V114 | Self::V200 => 3,
        }
    }
}

/// Which generation of the on-disk object format one file's objects are
/// written in.
///
/// Not a second [`LibverBound`]: the bound picks the datatype version a
/// *caller* asked for, while this says which superblock generation the file
/// already is. The two never combine freely — libhdf5 derives both from the
/// same low bound, so a version-0/1 superblock always carries version-1 object
/// headers and the `H5F_LIBVER_EARLIEST` row of every message-version table,
/// and a version-2/3 superblock always carries version-2 headers and the
/// `H5F_LIBVER_V18` row. Choosing per message is what would let this writer
/// emit a combination libhdf5 never writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObjectFormat {
    /// A version-0/1 superblock file: version-1 object headers, symbol-table
    /// groups, and the oldest message versions that can express the content.
    Legacy,
    /// A version-2/3 superblock file: version-2 object headers and the
    /// message versions `H5F_LIBVER_V18` calls for.
    #[default]
    Modern,
}

impl ObjectFormat {
    /// The object header version (`H5O_obj_ver_bounds`, H5Oint.c:125).
    pub fn object_header_version(self) -> u8 {
        match self {
            Self::Legacy => 1,
            Self::Modern => 2,
        }
    }

    /// The floor for a dataspace message's version
    /// (`H5O_sdspace_ver_bounds`, H5S.c:61). A null dataspace cannot be
    /// expressed at version 1 and raises itself.
    pub fn dataspace_version(self) -> u8 {
        match self {
            Self::Legacy => 1,
            Self::Modern => 2,
        }
    }

    /// The fill-value message version (`H5O_fill_ver_bounds`, H5Ofill.c:150).
    /// The earliest bound writes version 2 — version 1 is the separate
    /// "fill value (old)" message type, which this writer never emits.
    pub fn fill_value_version(self) -> u8 {
        match self {
            Self::Legacy => 2,
            Self::Modern => 3,
        }
    }

    /// The attribute message version (`H5O_attr_ver_bounds`, H5Aint.c:95).
    pub fn attribute_version(self) -> u8 {
        match self {
            Self::Legacy => 1,
            Self::Modern => 3,
        }
    }
}

/// UNDEF address constant
pub const UNDEF_ADDR: u64 = u64::MAX;

/// Fetches arbitrary file regions for the structure walkers that cannot hold a
/// file handle themselves (fractal heap, v2 B-tree, dense attribute storage).
pub trait BlockReader {
    /// Read up to `len` bytes starting at `offset`. Returning fewer bytes is
    /// only permitted at end of file; every caller re-checks the length it
    /// actually needs, so a short read surfaces as `BufferTooShort` rather
    /// than a misparse.
    fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>>;
}

/// Encode/decode error
#[derive(Debug)]
pub enum FormatError {
    InvalidSignature,
    InvalidVersion(u8),
    BufferTooShort { needed: usize, available: usize },
    ChecksumMismatch { expected: u32, computed: u32 },
    UnsupportedFeature(String),
    InvalidData(String),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSignature => write!(f, "invalid HDF5 signature"),
            Self::InvalidVersion(v) => write!(f, "unsupported version: {}", v),
            Self::BufferTooShort { needed, available } => {
                write!(
                    f,
                    "buffer too short: need {} bytes, have {}",
                    needed, available
                )
            }
            Self::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "checksum mismatch: expected 0x{:08x}, computed 0x{:08x}",
                    expected, computed
                )
            }
            Self::UnsupportedFeature(s) => write!(f, "unsupported feature: {}", s),
            Self::InvalidData(s) => write!(f, "invalid data: {}", s),
        }
    }
}

impl std::error::Error for FormatError {}

pub type FormatResult<T> = Result<T, FormatError>;
