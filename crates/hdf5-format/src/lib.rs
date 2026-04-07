//! Pure Rust HDF5 on-disk format codec.
//!
//! This crate handles encoding and decoding of HDF5 binary structures
//! (superblock, object headers, messages, chunk indices) without performing
//! any file I/O. It is used by `hdf5-io` and `hdf5` crates.

pub mod btree_v1;
pub mod checksum;
pub mod chunk_index;
pub mod global_heap;
pub mod local_heap;
pub mod messages;
pub mod object_header;
pub mod superblock;
pub mod symbol_table;

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

/// UNDEF address constant
pub const UNDEF_ADDR: u64 = u64::MAX;

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
