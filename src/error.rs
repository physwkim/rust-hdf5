//! Error types for the hdf5 public API crate.

/// Errors that can occur when using the HDF5 public API.
#[derive(Debug)]
pub enum Hdf5Error {
    /// An I/O error from the operating system.
    Io(std::io::Error),
    /// A low-level format encoding/decoding error.
    Format(crate::format::FormatError),
    /// An I/O-layer error from hdf5-io.
    IoLayer(crate::io::IoError),
    /// A requested object (dataset, group, attribute) was not found.
    ///
    /// The string contains the name of the missing object (e.g., dataset name).
    NotFound(String),
    /// The file or object is in an invalid state for the requested operation.
    InvalidState(String),
    /// A type mismatch between the Rust type and the HDF5 datatype.
    TypeMismatch(String),
    /// The object exists in the file but uses a feature this crate cannot
    /// decode. The string names the feature. Distinct from
    /// [`NotFound`](Self::NotFound): the name is in the listing, the content
    /// is out of reach.
    Unsupported(String),
    /// A soft or external link whose target does not exist. `H5Dopen` on a
    /// path through such a link fails; the name itself is present in the
    /// listing. `target` is the link value: a path for a soft link,
    /// `file::path` for an external one.
    DanglingLink { link: String, target: String },
    /// A zero-copy view of a dataset was asked for and the dataset cannot be
    /// viewed. The reason names why; a copying read still works.
    #[cfg(feature = "mmap")]
    NotViewable(crate::mapped::ViewRefusal),
    /// An external link whose target *file* could not be opened, listing the
    /// candidate paths that were tried.
    ExternalFileNotFound {
        link: String,
        file: String,
        searched: Vec<String>,
    },
}

impl From<std::io::Error> for Hdf5Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<crate::format::FormatError> for Hdf5Error {
    fn from(e: crate::format::FormatError) -> Self {
        Self::Format(e)
    }
}

/// A caller's hyperslab the extent does not admit is a bad request, not a
/// malformed file.
impl From<crate::format::selection::HyperslabError> for Hdf5Error {
    fn from(e: crate::format::selection::HyperslabError) -> Self {
        Self::InvalidState(e.to_string())
    }
}

impl From<crate::io::IoError> for Hdf5Error {
    fn from(e: crate::io::IoError) -> Self {
        // Every I/O-layer outcome carries through as its public twin, so a
        // caller can match on why a call failed — a torn SWMR read is
        // `Format(ChecksumMismatch)` — without the crate-private `IoError`.
        // No catch-all: a new `IoError` variant must name its public form
        // here rather than fall into an opaque `IoLayer`.
        use crate::io::IoError;
        match e {
            IoError::Io(e) => Self::Io(e),
            IoError::Format(e) => Self::Format(e),
            IoError::NotFound(s) => Self::NotFound(s),
            IoError::InvalidState(s) => Self::InvalidState(s),
            IoError::Unsupported(s) => Self::Unsupported(s),
            IoError::DanglingLink { link, target } => Self::DanglingLink { link, target },
            IoError::ExternalFileNotFound {
                link,
                file,
                searched,
            } => Self::ExternalFileNotFound {
                link,
                file,
                searched,
            },
        }
    }
}

impl std::fmt::Display for Hdf5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Format(e) => write!(f, "format error: {}", e),
            Self::IoLayer(e) => write!(f, "hdf5-io error: {}", e),
            Self::NotFound(s) => write!(f, "dataset '{}' not found", s),
            Self::InvalidState(s) => write!(f, "invalid state: {}", s),
            Self::TypeMismatch(s) => write!(f, "type mismatch: {}", s),
            Self::Unsupported(s) => write!(f, "unsupported: {}", s),
            Self::DanglingLink { link, target } => write!(
                f,
                "link '{}' points to '{}', which does not exist",
                link, target
            ),
            #[cfg(feature = "mmap")]
            Self::NotViewable(reason) => {
                write!(f, "the dataset cannot be viewed in place: {reason}")
            }
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

impl std::error::Error for Hdf5Error {}

/// A specialized `Result` type for HDF5 operations.
pub type Result<T> = std::result::Result<T, Hdf5Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_not_found() {
        let err = Hdf5Error::NotFound("my_dataset".into());
        assert!(format!("{}", err).contains("my_dataset"));
    }

    #[test]
    fn display_invalid_state() {
        let err = Hdf5Error::InvalidState("file already closed".into());
        assert!(format!("{}", err).contains("file already closed"));
    }

    #[test]
    fn display_type_mismatch() {
        let err = Hdf5Error::TypeMismatch("expected f64, got u8".into());
        assert!(format!("{}", err).contains("expected f64, got u8"));
    }

    /// The I/O-layer outcomes that used to fall into `IoLayer` reach the
    /// caller as the public variant of the same name.
    #[test]
    fn io_layer_errors_carry_through_as_their_public_variant() {
        use crate::format::FormatError;
        use crate::io::IoError;

        let mismatch = IoError::Format(FormatError::ChecksumMismatch {
            expected: 1,
            computed: 2,
        });
        assert!(matches!(
            Hdf5Error::from(mismatch),
            Hdf5Error::Format(FormatError::ChecksumMismatch {
                expected: 1,
                computed: 2
            })
        ));
        let eof = IoError::Io(std::io::ErrorKind::UnexpectedEof.into());
        assert!(matches!(
            Hdf5Error::from(eof),
            Hdf5Error::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof
        ));
        let state = IoError::InvalidState("short buffer".into());
        assert!(matches!(
            Hdf5Error::from(state),
            Hdf5Error::InvalidState(s) if s == "short buffer"
        ));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: Hdf5Error = io_err.into();
        match err {
            Hdf5Error::Io(_) => {}
            other => panic!("expected Io variant, got: {:?}", other),
        }
    }
}
