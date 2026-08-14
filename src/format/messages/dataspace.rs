//! Dataspace message (type 0x01) — describes dataset dimensionality.
//!
//! Binary layout (version 2):
//!   Byte 0: version = 2
//!   Byte 1: dimensionality (ndims, 0–32)
//!   Byte 2: flags (bit 0 = max dims present)
//!   Byte 3: type (0 = scalar, 1 = simple, 2 = null)
//!   Then ndims * sizeof_size bytes for current dimensions
//!   Then (if flag bit 0) ndims * sizeof_size bytes for max dimensions

use crate::format::bytes::read_le_uint as read_size;
use crate::format::{FormatContext, FormatError, FormatResult};

const VERSION: u8 = 2;
const FLAG_MAX_DIMS: u8 = 0x01;

/// Dataspace type field values.
const DS_TYPE_SCALAR: u8 = 0;
const DS_TYPE_SIMPLE: u8 = 1;
const DS_TYPE_NULL: u8 = 2;

/// Which of the three dataspace classes a message describes (`H5Osdspace.c`'s
/// type byte / `H5S_class_t`).
///
/// `Scalar` (rank 0, exactly one element) and `Null` (no elements at all) both
/// carry an empty `dims`, so the class cannot be inferred from `dims` alone —
/// it must be tracked explicitly, matching upstream's dedicated type field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataspaceClass {
    /// Rank 0: a single element, no dimensions.
    Scalar,
    /// Rank >= 1: an ordinary N-dimensional dataspace.
    Simple,
    /// No elements at all (`H5Screate(H5S_NULL)`) — distinct from a scalar
    /// or from a `Simple` dataspace with a zero-length dimension.
    Null,
}

/// Dataspace message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct DataspaceMessage {
    /// Which dataspace class this message describes.
    pub class: DataspaceClass,
    /// Current dimension sizes. Empty for `Scalar` and `Null`.
    pub dims: Vec<u64>,
    /// Optional maximum dimension sizes.  An entry of `u64::MAX` means unlimited.
    pub max_dims: Option<Vec<u64>>,
}

impl DataspaceMessage {
    // ------------------------------------------------------------------ factories

    /// A scalar dataspace (rank 0, no max dims).
    pub fn scalar() -> Self {
        Self {
            class: DataspaceClass::Scalar,
            dims: Vec::new(),
            max_dims: None,
        }
    }

    /// The NULL dataspace: no elements at all. Unlike [`scalar`](Self::scalar),
    /// a dataset with this dataspace holds zero bytes of data.
    pub fn null() -> Self {
        Self {
            class: DataspaceClass::Null,
            dims: Vec::new(),
            max_dims: None,
        }
    }

    /// A simple dataspace with fixed dimensions (max == current). An empty
    /// `dims` yields a scalar dataspace, matching how upstream never emits a
    /// `Simple`-class dataspace at rank 0.
    pub fn simple(dims: &[u64]) -> Self {
        Self {
            class: Self::class_for_rank(dims.len()),
            dims: dims.to_vec(),
            max_dims: None,
        }
    }

    /// A simple dataspace where every dimension is unlimited.
    pub fn unlimited(current: &[u64]) -> Self {
        Self {
            class: Self::class_for_rank(current.len()),
            dims: current.to_vec(),
            max_dims: Some(vec![u64::MAX; current.len()]),
        }
    }

    /// `Scalar` at rank 0 (matching upstream, which never writes a
    /// `Simple`-class dataspace with zero dimensions), `Simple` otherwise.
    fn class_for_rank(ndims: usize) -> DataspaceClass {
        if ndims == 0 {
            DataspaceClass::Scalar
        } else {
            DataspaceClass::Simple
        }
    }

    /// Whether this is the NULL dataspace (no elements at all).
    pub fn is_null(&self) -> bool {
        self.class == DataspaceClass::Null
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let ndims = self.dims.len();
        let ss = ctx.sizeof_size as usize;
        let has_max = self.max_dims.is_some();
        let flags: u8 = if has_max { FLAG_MAX_DIMS } else { 0 };

        let ds_type = match self.class {
            DataspaceClass::Scalar => DS_TYPE_SCALAR,
            DataspaceClass::Simple => DS_TYPE_SIMPLE,
            DataspaceClass::Null => DS_TYPE_NULL,
        };

        let body_len = 4 + ndims * ss + if has_max { ndims * ss } else { 0 };
        let mut buf = Vec::with_capacity(body_len);

        buf.push(VERSION);
        buf.push(ndims as u8);
        buf.push(flags);
        buf.push(ds_type); // type byte required for version 2

        // current dimensions
        for &d in &self.dims {
            buf.extend_from_slice(&d.to_le_bytes()[..ss]);
        }

        // max dimensions
        if let Some(ref maxes) = self.max_dims {
            for &m in maxes {
                buf.extend_from_slice(&m.to_le_bytes()[..ss]);
            }
        }

        buf
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        if buf.len() < 4 {
            return Err(FormatError::BufferTooShort {
                needed: 4,
                available: buf.len(),
            });
        }

        let version = buf[0];
        match version {
            1 => Self::decode_v1(buf, ctx),
            VERSION => Self::decode_v2(buf, ctx),
            _ => Err(FormatError::InvalidVersion(version)),
        }
    }

    /// Decode version 2 dataspace message.
    fn decode_v2(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        let ndims = buf[1] as usize;
        let flags = buf[2];
        let class = match buf[3] {
            DS_TYPE_SCALAR => DataspaceClass::Scalar,
            DS_TYPE_SIMPLE => DataspaceClass::Simple,
            DS_TYPE_NULL => DataspaceClass::Null,
            other => {
                return Err(FormatError::InvalidData(format!(
                    "dataspace type byte {other} is not scalar(0)/simple(1)/null(2)"
                )))
            }
        };
        let has_max = (flags & FLAG_MAX_DIMS) != 0;
        let ss = ctx.sizeof_size as usize;

        let needed = 4 + ndims * ss + if has_max { ndims * ss } else { 0 };
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }

        let mut pos = 4;

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(read_size(&buf[pos..], ss));
            pos += ss;
        }

        let max_dims = if has_max {
            let mut v = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                v.push(read_size(&buf[pos..], ss));
                pos += ss;
            }
            Some(v)
        } else {
            None
        };

        Ok((
            Self {
                class,
                dims,
                max_dims,
            },
            pos,
        ))
    }

    /// Decode version 1 dataspace message.
    ///
    /// Version 1 layout:
    /// ```text
    /// Byte 0: version = 1
    /// Byte 1: ndims
    /// Byte 2: flags (bit 0 = max dims present, bit 1 = permutation indices present)
    /// Byte 3: reserved
    /// Bytes 4-7: reserved (4 bytes)
    /// Then ndims * sizeof_size bytes for current dimensions
    /// Then (if flag bit 0) ndims * sizeof_size bytes for max dimensions
    /// Then (if flag bit 1) ndims * sizeof_size bytes for permutation indices
    /// ```
    fn decode_v1(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        if buf.len() < 8 {
            return Err(FormatError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        let ndims = buf[1] as usize;
        let flags = buf[2];
        let has_max = (flags & FLAG_MAX_DIMS) != 0;
        let has_perm = (flags & 0x02) != 0;
        let ss = ctx.sizeof_size as usize;

        // Header is 8 bytes for v1 (4 fixed + 4 reserved)
        let mut needed = 8 + ndims * ss;
        if has_max {
            needed += ndims * ss;
        }
        if has_perm {
            needed += ndims * ss;
        }
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }

        let mut pos = 8; // skip version(1) + ndims(1) + flags(1) + reserved(1) + reserved(4)

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(read_size(&buf[pos..], ss));
            pos += ss;
        }

        let max_dims = if has_max {
            let mut v = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                v.push(read_size(&buf[pos..], ss));
                pos += ss;
            }
            Some(v)
        } else {
            None
        };

        // Skip permutation indices if present
        if has_perm {
            pos += ndims * ss;
        }

        // Version 1 predates the NULL dataspace (added with the version-2
        // message, `H5Osdspace.c`): it carries no type byte, so the class is
        // always inferred from rank, same as `simple()`/`unlimited()`.
        Ok((
            Self {
                class: Self::class_for_rank(ndims),
                dims,
                max_dims,
            },
            pos,
        ))
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx8() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    fn ctx4() -> FormatContext {
        FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        }
    }

    #[test]
    fn roundtrip_scalar() {
        let msg = DataspaceMessage::scalar();
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded.len(), 4); // version + ndims + flags + type
        let (decoded, consumed) = DataspaceMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_simple_1d() {
        let msg = DataspaceMessage::simple(&[100]);
        let encoded = msg.encode(&ctx8());
        // 4 header + 1*8 dims = 12
        assert_eq!(encoded.len(), 12);
        let (decoded, consumed) = DataspaceMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_simple_3d_ctx4() {
        let msg = DataspaceMessage::simple(&[10, 20, 30]);
        let encoded = msg.encode(&ctx4());
        // 4 + 3*4 = 16
        assert_eq!(encoded.len(), 16);
        let (decoded, consumed) = DataspaceMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, 16);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_unlimited() {
        let msg = DataspaceMessage::unlimited(&[5, 10]);
        let encoded = msg.encode(&ctx8());
        // 4 + 2*8 dims + 2*8 max = 36
        assert_eq!(encoded.len(), 36);
        let (decoded, consumed) = DataspaceMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 36);
        assert_eq!(decoded, msg);
        assert_eq!(decoded.max_dims.as_ref().unwrap(), &vec![u64::MAX; 2]);
    }

    #[test]
    fn roundtrip_partial_max() {
        let msg = DataspaceMessage {
            class: DataspaceClass::Simple,
            dims: vec![3, 4],
            max_dims: Some(vec![100, u64::MAX]),
        };
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = DataspaceMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [99u8, 0, 0, 0]; // version 99 — unsupported
        let err = DataspaceMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidVersion(99) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_v1_simple_1d() {
        // Build a version 1 dataspace: 1D, dims=[100], no max
        let mut buf = vec![
            1, // version 1
            1, // ndims = 1
            0, // flags (no max dims)
            0, // reserved
        ];
        buf.extend_from_slice(&[0u8; 4]); // reserved (4 bytes)
        buf.extend_from_slice(&100u64.to_le_bytes()); // dims[0] = 100

        let (msg, consumed) = DataspaceMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, 16); // 8 header + 8 dim
        assert_eq!(msg.dims, vec![100]);
        assert_eq!(msg.max_dims, None);
    }

    #[test]
    fn decode_v1_with_max_dims() {
        let mut buf = vec![
            1, // version 1
            2, // ndims = 2
            1, // flags = has max dims
            0, // reserved
        ];
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&10u64.to_le_bytes()); // dims[0] = 10
        buf.extend_from_slice(&20u64.to_le_bytes()); // dims[1] = 20
        buf.extend_from_slice(&u64::MAX.to_le_bytes()); // max_dims[0] = unlimited
        buf.extend_from_slice(&100u64.to_le_bytes()); // max_dims[1] = 100

        let (msg, consumed) = DataspaceMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, 40); // 8 + 2*8 + 2*8
        assert_eq!(msg.dims, vec![10, 20]);
        assert_eq!(msg.max_dims, Some(vec![u64::MAX, 100]));
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [2u8, 1, 0]; // version ok, but too short (need 4 header bytes)
        let err = DataspaceMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short_for_dims() {
        // ndims=1, no max, sizeof_size=8 => need 4 + 8 = 12 bytes, give 6
        let buf = [2u8, 1, 0, 1, 0, 0];
        let err = DataspaceMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort {
                needed: 12,
                available: 6,
            } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_byte_is_two() {
        let msg = DataspaceMessage::simple(&[42]);
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded[0], 2);
    }

    #[test]
    fn roundtrip_null() {
        let msg = DataspaceMessage::null();
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded.len(), 4); // version + ndims + flags + type, no dims
        assert_eq!(encoded[3], 2); // type byte: null
        let (decoded, consumed) = DataspaceMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(decoded, msg);
        assert!(decoded.is_null());
    }

    /// A NULL dataspace and a scalar dataspace both have an empty `dims`,
    /// but must not decode to the same message: only the type byte tells
    /// them apart, and it must round-trip distinctly.
    #[test]
    fn null_and_scalar_are_distinct() {
        let null = DataspaceMessage::null();
        let scalar = DataspaceMessage::scalar();
        assert_ne!(null, scalar);
        assert!(null.is_null());
        assert!(!scalar.is_null());

        let null_encoded = null.encode(&ctx8());
        let scalar_encoded = scalar.encode(&ctx8());
        assert_ne!(null_encoded[3], scalar_encoded[3]);

        let (null_decoded, _) = DataspaceMessage::decode(&null_encoded, &ctx8()).unwrap();
        let (scalar_decoded, _) = DataspaceMessage::decode(&scalar_encoded, &ctx8()).unwrap();
        assert!(null_decoded.is_null());
        assert!(!scalar_decoded.is_null());
    }

    #[test]
    fn decode_v2_bad_type_byte() {
        // version 2, ndims=0, flags=0, type=3 (not scalar/simple/null)
        let buf = [2u8, 0, 0, 3];
        let err = DataspaceMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidData(_) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_v1_is_never_null() {
        // Version 1 has no type byte; rank 0 must decode as Scalar, not Null.
        let buf = [1u8, 0, 0, 0, 0, 0, 0, 0];
        let (msg, _) = DataspaceMessage::decode(&buf, &ctx8()).unwrap();
        assert!(!msg.is_null());
        assert_eq!(msg.class, DataspaceClass::Scalar);
    }
}
