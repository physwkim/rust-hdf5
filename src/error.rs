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

impl From<crate::io::IoError> for Hdf5Error {
    fn from(e: crate::io::IoError) -> Self {
        // Every outcome that names *why* a lookup failed carries through as
        // itself so a caller can match on it; everything else keeps its
        // existing shape. `NotFound` is among them: a `?` on an I/O-layer
        // lookup must not turn a plain absence into an opaque `IoLayer`,
        // which is what forced callers to re-map it by hand.
        match e {
            crate::io::IoError::NotFound(s) => Self::NotFound(s),
            crate::io::IoError::Unsupported(s) => Self::Unsupported(s),
            crate::io::IoError::DanglingLink { link, target } => {
                Self::DanglingLink { link, target }
            }
            crate::io::IoError::ExternalFileNotFound {
                link,
                file,
                searched,
            } => Self::ExternalFileNotFound {
                link,
                file,
                searched,
            },
            other => Self::IoLayer(other),
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
