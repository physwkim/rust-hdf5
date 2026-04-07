//! Mapping from Rust types to HDF5 datatype messages.

use hdf5_format::messages::datatype::DatatypeMessage;

/// Trait for Rust types that have a corresponding HDF5 datatype.
///
/// Implement this for any `Copy + 'static` type that can be stored in an HDF5
/// dataset. The library provides implementations for all standard numeric types.
pub trait H5Type: Sized + Copy + 'static {
    /// Return the HDF5 datatype message describing this type.
    fn hdf5_type() -> DatatypeMessage;

    /// Return the size of a single element in bytes.
    fn element_size() -> usize;
}

impl H5Type for u8 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::u8_type()
    }
    fn element_size() -> usize {
        1
    }
}

impl H5Type for i8 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::i8_type()
    }
    fn element_size() -> usize {
        1
    }
}

impl H5Type for u16 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::u16_type()
    }
    fn element_size() -> usize {
        2
    }
}

impl H5Type for i16 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::i16_type()
    }
    fn element_size() -> usize {
        2
    }
}

impl H5Type for u32 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::u32_type()
    }
    fn element_size() -> usize {
        4
    }
}

impl H5Type for i32 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::i32_type()
    }
    fn element_size() -> usize {
        4
    }
}

impl H5Type for u64 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::u64_type()
    }
    fn element_size() -> usize {
        8
    }
}

impl H5Type for i64 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::i64_type()
    }
    fn element_size() -> usize {
        8
    }
}

impl H5Type for f32 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::f32_type()
    }
    fn element_size() -> usize {
        4
    }
}

impl H5Type for f64 {
    fn hdf5_type() -> DatatypeMessage {
        DatatypeMessage::f64_type()
    }
    fn element_size() -> usize {
        8
    }
}

/// HDF5 boolean type, stored as u8 (0=false, 1=true).
///
/// This is a `Copy` type that can be used with `H5Type`:
/// ```
/// use hdf5::types::HBool;
/// let t: HBool = true.into();
/// let b: bool = t.into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct HBool(pub u8);

impl From<bool> for HBool {
    fn from(b: bool) -> Self { Self(b as u8) }
}

impl From<HBool> for bool {
    fn from(h: HBool) -> Self { h.0 != 0 }
}

impl H5Type for HBool {
    fn hdf5_type() -> DatatypeMessage { DatatypeMessage::bool_type() }
    fn element_size() -> usize { 1 }
}

/// Complex number with f32 real and imaginary parts (8 bytes total).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl H5Type for Complex32 {
    fn hdf5_type() -> DatatypeMessage {
        use hdf5_format::messages::datatype::CompoundMember;
        DatatypeMessage::Compound {
            size: 8,
            members: vec![
                CompoundMember { name: "r".to_string(), offset: 0, datatype: DatatypeMessage::f32_type() },
                CompoundMember { name: "i".to_string(), offset: 4, datatype: DatatypeMessage::f32_type() },
            ],
        }
    }
    fn element_size() -> usize { 8 }
}

/// Complex number with f64 real and imaginary parts (16 bytes total).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl H5Type for Complex64 {
    fn hdf5_type() -> DatatypeMessage {
        use hdf5_format::messages::datatype::CompoundMember;
        DatatypeMessage::Compound {
            size: 16,
            members: vec![
                CompoundMember { name: "r".to_string(), offset: 0, datatype: DatatypeMessage::f64_type() },
                CompoundMember { name: "i".to_string(), offset: 8, datatype: DatatypeMessage::f64_type() },
            ],
        }
    }
    fn element_size() -> usize { 16 }
}

/// A description of a compound (struct) type for use with HDF5 datasets.
///
/// Users can build compound types manually to describe structured data.
///
/// # Example
///
/// ```
/// use hdf5::types::CompoundType;
/// use hdf5_format::messages::datatype::{DatatypeMessage, CompoundMember};
///
/// let ct = CompoundType {
///     members: vec![
///         ("x".to_string(), DatatypeMessage::f32_type(), 0),
///         ("y".to_string(), DatatypeMessage::f32_type(), 4),
///     ],
///     total_size: 8,
/// };
/// let dt = ct.to_datatype();
/// assert_eq!(dt.element_size(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct CompoundType {
    /// Members: (name, datatype, byte_offset).
    pub members: Vec<(String, DatatypeMessage, u32)>,
    /// Total size of the compound element in bytes.
    pub total_size: u32,
}

impl CompoundType {
    /// Convert this compound type description to a `DatatypeMessage`.
    pub fn to_datatype(&self) -> DatatypeMessage {
        use hdf5_format::messages::datatype::CompoundMember;

        let members = self
            .members
            .iter()
            .map(|(name, dt, offset)| CompoundMember {
                name: name.clone(),
                offset: *offset,
                datatype: dt.clone(),
            })
            .collect();

        DatatypeMessage::Compound {
            size: self.total_size,
            members,
        }
    }
}

/// A variable-length Unicode string type.
///
/// This type is used as a marker for the `new_attr` builder API.
/// Internally, attributes with this type use fixed-length string encoding
/// (the string value determines the size), avoiding the complexity of the
/// global heap while maintaining API compatibility with hdf5-metno.
///
/// # Example
///
/// ```no_run
/// use hdf5::types::VarLenUnicode;
///
/// let s: VarLenUnicode = "hello".parse().unwrap();
/// assert_eq!(s.0, "hello");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct VarLenUnicode(pub String);

impl std::str::FromStr for VarLenUnicode {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(s.to_string()))
    }
}

impl From<String> for VarLenUnicode {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for VarLenUnicode {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<VarLenUnicode> for String {
    fn from(v: VarLenUnicode) -> Self {
        v.0
    }
}

impl std::fmt::Display for VarLenUnicode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_type_matches() {
        assert_eq!(u8::element_size(), 1);
        assert_eq!(u8::hdf5_type(), DatatypeMessage::u8_type());
    }

    #[test]
    fn i8_type_matches() {
        assert_eq!(i8::element_size(), 1);
        assert_eq!(i8::hdf5_type(), DatatypeMessage::i8_type());
    }

    #[test]
    fn u16_type_matches() {
        assert_eq!(u16::element_size(), 2);
        assert_eq!(u16::hdf5_type(), DatatypeMessage::u16_type());
    }

    #[test]
    fn i16_type_matches() {
        assert_eq!(i16::element_size(), 2);
        assert_eq!(i16::hdf5_type(), DatatypeMessage::i16_type());
    }

    #[test]
    fn u32_type_matches() {
        assert_eq!(u32::element_size(), 4);
        assert_eq!(u32::hdf5_type(), DatatypeMessage::u32_type());
    }

    #[test]
    fn i32_type_matches() {
        assert_eq!(i32::element_size(), 4);
        assert_eq!(i32::hdf5_type(), DatatypeMessage::i32_type());
    }

    #[test]
    fn u64_type_matches() {
        assert_eq!(u64::element_size(), 8);
        assert_eq!(u64::hdf5_type(), DatatypeMessage::u64_type());
    }

    #[test]
    fn i64_type_matches() {
        assert_eq!(i64::element_size(), 8);
        assert_eq!(i64::hdf5_type(), DatatypeMessage::i64_type());
    }

    #[test]
    fn f32_type_matches() {
        assert_eq!(f32::element_size(), 4);
        assert_eq!(f32::hdf5_type(), DatatypeMessage::f32_type());
    }

    #[test]
    fn f64_type_matches() {
        assert_eq!(f64::element_size(), 8);
        assert_eq!(f64::hdf5_type(), DatatypeMessage::f64_type());
    }

    #[test]
    fn element_size_matches_std_mem() {
        assert_eq!(u8::element_size(), std::mem::size_of::<u8>());
        assert_eq!(i16::element_size(), std::mem::size_of::<i16>());
        assert_eq!(u32::element_size(), std::mem::size_of::<u32>());
        assert_eq!(f64::element_size(), std::mem::size_of::<f64>());
    }
}
