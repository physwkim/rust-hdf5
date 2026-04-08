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
        Self::IoLayer(e)
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
