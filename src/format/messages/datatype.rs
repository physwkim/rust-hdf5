//! Datatype message (type 0x03) — describes element data type.
//!
//! Binary layout:
//!   Byte 0:    (class & 0x0F) | (version << 4)     version = 1
//!   Bytes 1-3: class bit-field flags (24 bits, little-endian)
//!   Bytes 4-7: element size (u32 LE)
//!   Bytes 8+:  class-specific properties

use crate::format::{FormatContext, FormatError, FormatResult};

const DT_VERSION: u8 = 1;

/// libhdf5 `H5O_DTYPE_VERSION_LATEST`: the highest datatype message version
/// the format defines (5, the HDF5 2.0 encoding).
const DT_VERSION_LATEST: u8 = 5;

/// The highest datatype message version this crate decodes. Versions 1-4 share
/// the property layouts below; version 5 came with the 2.0 complex-number
/// encoding and is reported as unsupported rather than misread.
const DT_VERSION_MAX_SUPPORTED: u8 = 4;

/// libhdf5 `H5VM_limit_enc_size`: the number of bytes needed to encode any
/// value in `0..=limit`. Used for version-3 compound member offsets, which
/// are stored in `limit_enc_size(compound_size)` little-endian bytes.
fn limit_enc_size(limit: u64) -> usize {
    let log2 = if limit == 0 {
        0
    } else {
        63 - limit.leading_zeros()
    };
    (log2 / 8 + 1) as usize
}

/// Read a little-endian unsigned integer of `n` (`<= 4`) bytes.
fn read_uint_le(buf: &[u8], n: usize) -> u32 {
    let mut tmp = [0u8; 4];
    tmp[..n].copy_from_slice(&buf[..n]);
    u32::from_le_bytes(tmp)
}

/// Decode a compound- or enum-member name field starting at `pos`, returning
/// the name and the position just past the field.
///
/// The name is null-terminated. Message versions 1 and 2 pad the field — the
/// name plus its terminator — to a multiple of 8 bytes; version 3 dropped the
/// padding and advances by exactly `strlen + 1`. Both the compound and the
/// enum branch of libhdf5's `H5O__dtype_decode_helper` use this one rule
/// (`H5Odtype.c`, "Version 3 of the datatype message eliminated the padding to
/// multiple of 8 bytes"), so it lives here once rather than per class.
fn decode_name_field(
    buf: &[u8],
    pos: usize,
    version: u8,
    what: &str,
) -> FormatResult<(String, usize)> {
    let mut end = pos;
    while end < buf.len() && buf[end] != 0 {
        end += 1;
    }
    if end >= buf.len() {
        return Err(FormatError::InvalidData(format!(
            "unterminated {what} member name"
        )));
    }
    let name = String::from_utf8_lossy(&buf[pos..end]).to_string();
    let field_len = end + 1 - pos; // name bytes plus the null terminator
    let advance = if version >= 3 {
        field_len
    } else {
        field_len.div_ceil(8) * 8
    };
    let next = pos + advance;
    if next > buf.len() {
        // The 8-byte padding of a v1/v2 name can run past a truncated buffer.
        return Err(FormatError::BufferTooShort {
            needed: next,
            available: buf.len(),
        });
    }
    Ok((name, next))
}

/// The bytes of one fixed-length string element that carry its value, under
/// the padding rule the datatype declares.
///
/// libhdf5's `H5T__conv_s_s` (`H5Tconv_string.c`) is the rule and the only
/// place a fixed string's padding is interpreted: a null-terminated (0) or
/// null-padded (1) element ends at its first NUL — both branches stop on
/// `!s[nchars]` — and a space-padded (2) one ends after its last non-space
/// byte, so an embedded NUL survives there. Every other code is reserved;
/// libhdf5 fails the conversion with "source string padding method not
/// supported", and this returns `None` rather than guess a rule.
///
/// This is the single owner of that rule: dataset elements and attribute
/// values are found differently, but both end here.
pub fn fixed_string_content(elem: &[u8], padding: u8) -> Option<&[u8]> {
    let end = match padding {
        0 | 1 => elem.iter().position(|&b| b == 0).unwrap_or(elem.len()),
        2 => elem.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1),
        _ => return None,
    };
    Some(&elem[..end])
}

// Datatype class codes
const CLASS_FIXED_POINT: u8 = 0;
const CLASS_FLOATING_POINT: u8 = 1;
const CLASS_STRING: u8 = 3;
const CLASS_BITFIELD: u8 = 4;
const CLASS_OPAQUE: u8 = 5;
const CLASS_COMPOUND: u8 = 6;
const CLASS_ENUM: u8 = 8;
const CLASS_VLEN: u8 = 9;
const CLASS_ARRAY: u8 = 10;

/// libhdf5 `H5T_OPAQUE_TAG_MAX`: the opaque tag field is stored in
/// `(strlen + 7) & (H5T_OPAQUE_TAG_MAX - 8)` bytes, i.e. rounded up to a
/// multiple of 8 and capped at 248.
const OPAQUE_TAG_MAX: usize = 256;

/// Byte order for numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// A member within a compound datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset within the compound element.
    pub offset: u32,
    /// Datatype of this member.
    pub datatype: DatatypeMessage,
}

/// A member within an enum datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumMember {
    /// Enum member name.
    pub name: String,
    /// Raw value bytes (length matches base type size).
    pub value: Vec<u8>,
}

/// HDF5 datatype descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum DatatypeMessage {
    FixedPoint {
        size: u32,
        byte_order: ByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    },
    FloatingPoint {
        size: u32,
        byte_order: ByteOrder,
        sign_location: u8,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
    /// Bit field (class 4): `bit_precision` bits starting at `bit_offset`
    /// within a `size`-byte element, stored in `byte_order`.
    ///
    /// libhdf5 has no bit-level Rust analogue; a full-width bit field is read
    /// as an unsigned integer of the stored width (`H5T_STD_B8LE` → `u8`).
    BitField {
        /// Element size in bytes.
        size: u32,
        /// Byte order of the element.
        byte_order: ByteOrder,
        /// Bit offset of the first significant bit.
        bit_offset: u16,
        /// Number of significant bits.
        bit_precision: u16,
    },
    /// Opaque datatype (class 5): `size` uninterpreted bytes plus an ASCII tag
    /// naming the format they are in (`H5Tset_tag`).
    Opaque {
        /// Element size in bytes.
        size: u32,
        /// The tag, without the null padding it carries on disk.
        tag: String,
    },
    /// Fixed-length string type (class 3).
    FixedString {
        /// String size in bytes (including null terminator if null-terminated).
        size: u32,
        /// Padding type: 0 = null terminate, 1 = null pad, 2 = space pad.
        padding: u8,
        /// Character set: 0 = ASCII, 1 = UTF-8.
        charset: u8,
    },
    /// Compound datatype (class 6).
    Compound {
        /// Total size of the compound element in bytes.
        size: u32,
        /// Members of the compound type.
        members: Vec<CompoundMember>,
    },
    /// Enumeration datatype (class 8).
    Enum {
        /// Base integer type.
        base: Box<DatatypeMessage>,
        /// Enumeration members (name + value pairs).
        members: Vec<EnumMember>,
    },
    /// Variable-length string datatype (class 9, vlen type 1).
    VarLenString {
        /// Padding type: 0 = null terminate, 1 = null pad, 2 = space pad.
        ///
        /// A variable-length string carries the rule in its own bit field
        /// (`H5Odtype.c`: `vlen.pad = (flags >> 4) & 0x0f`), not in the parent
        /// type, so it survives a round trip through this message.
        padding: u8,
        /// Character set: 0 = ASCII, 1 = UTF-8.
        charset: u8,
    },
    /// Variable-length sequence datatype (class 9, vlen type 0): each item is a
    /// variable number of `base` elements stored in the global heap. With
    /// `base` = `u8` this is a variable-length byte array.
    ///
    /// Mirrors libhdf5 `H5T_VLEN`/`H5T_VLEN_SEQUENCE` (see `H5Odtype.c`): the
    /// bit field's low nibble is the vlen type (0 = sequence), and the parent
    /// (base) datatype message is embedded in the properties.
    VarLenSequence {
        /// Element type of each item's sequence.
        base: Box<DatatypeMessage>,
    },
    /// Array datatype (class 10).
    Array {
        /// Dimension sizes of the array.
        dims: Vec<u32>,
        /// Base element type.
        base: Box<DatatypeMessage>,
    },
}

// ========================================================================= factory methods

impl DatatypeMessage {
    pub fn u8_type() -> Self {
        Self::FixedPoint {
            size: 1,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    pub fn i8_type() -> Self {
        Self::FixedPoint {
            size: 1,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    pub fn u16_type() -> Self {
        Self::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    pub fn i16_type() -> Self {
        Self::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    pub fn u32_type() -> Self {
        Self::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    pub fn i32_type() -> Self {
        Self::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    pub fn u64_type() -> Self {
        Self::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 64,
        }
    }

    pub fn i64_type() -> Self {
        Self::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 64,
        }
    }

    pub fn f32_type() -> Self {
        Self::FloatingPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            sign_location: 31,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    pub fn f64_type() -> Self {
        Self::FloatingPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            sign_location: 63,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    /// Boolean type (stored as 1-byte enum: 0=FALSE, 1=TRUE).
    ///
    /// HDF5 represents booleans as an enumerated type over u8.
    pub fn bool_type() -> Self {
        Self::Enum {
            base: Box::new(Self::u8_type()),
            members: vec![
                EnumMember {
                    name: "FALSE".to_string(),
                    value: vec![0],
                },
                EnumMember {
                    name: "TRUE".to_string(),
                    value: vec![1],
                },
            ],
        }
    }

    /// Null-terminated ASCII fixed-length string.
    pub fn fixed_string(size: u32) -> Self {
        Self::FixedString {
            size,
            padding: 0, // null terminate
            charset: 0, // ASCII
        }
    }

    /// Null-terminated UTF-8 fixed-length string.
    pub fn fixed_string_utf8(size: u32) -> Self {
        Self::FixedString {
            size,
            padding: 0, // null terminate
            charset: 1, // UTF-8
        }
    }

    /// Null-terminated variable-length UTF-8 string type.
    ///
    /// Note: `element_size()` for this type requires a `FormatContext` to
    /// compute. Use `element_size_ctx()` or `vlen_ref_size()` instead.
    pub fn vlen_string_utf8() -> Self {
        Self::VarLenString {
            padding: 0, // null terminate, as h5py's `string_dtype` declares
            charset: 1, // UTF-8
        }
    }

    /// Null-terminated variable-length ASCII string type.
    pub fn vlen_string_ascii() -> Self {
        Self::VarLenString {
            padding: 0, // null terminate
            charset: 0, // ASCII
        }
    }

    /// Variable-length byte-array type: a vlen sequence of `u8`.
    ///
    /// Note: like `VarLenString`, `element_size()` for this type requires a
    /// `FormatContext`. Use `element_size_ctx()` or `vlen_ref_size()`.
    pub fn vlen_bytes() -> Self {
        Self::VarLenSequence {
            base: Box::new(Self::u8_type()),
        }
    }

    /// Compound datatype.
    pub fn compound(size: u32, members: Vec<CompoundMember>) -> Self {
        Self::Compound { size, members }
    }

    /// Enumeration datatype.
    pub fn enumeration(base: DatatypeMessage, members: Vec<EnumMember>) -> Self {
        Self::Enum {
            base: Box::new(base),
            members,
        }
    }

    /// Array datatype.
    pub fn array(dims: Vec<u32>, base: DatatypeMessage) -> Self {
        Self::Array {
            dims,
            base: Box::new(base),
        }
    }
}

/// The IEEE 754 binary interchange formats this crate reinterprets.
///
/// A floating-point datatype message describes an arbitrary bit layout; only
/// these three can be handed to a caller as a Rust float, so they are named
/// once here rather than re-tested per call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IeeeFormat {
    /// binary16 (half): 1 sign, 5 exponent, 10 mantissa bits, bias 15.
    Binary16,
    /// binary32 (`f32`).
    Binary32,
    /// binary64 (`f64`).
    Binary64,
}

// ========================================================================= queries

impl DatatypeMessage {
    /// The IEEE 754 interchange format of a floating-point datatype, or `None`
    /// for a float with any other bit layout (and for every other class).
    ///
    /// The whole layout is checked — width, bit offset and precision, sign,
    /// exponent and mantissa positions, and the exponent bias — because a
    /// float that differs anywhere cannot be reinterpreted, only converted.
    pub fn ieee_format(&self) -> Option<IeeeFormat> {
        let Self::FloatingPoint {
            size,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
            ..
        } = self
        else {
            return None;
        };
        if *bit_offset != 0 || u32::from(*bit_precision) != size * 8 {
            return None;
        }
        let layout = (
            *sign_location,
            *exponent_location,
            *exponent_size,
            *mantissa_location,
            *mantissa_size,
            *exponent_bias,
        );
        match (size, layout) {
            (2, (15, 10, 5, 0, 10, 15)) => Some(IeeeFormat::Binary16),
            (4, (31, 23, 8, 0, 23, 127)) => Some(IeeeFormat::Binary32),
            (8, (63, 52, 11, 0, 52, 1023)) => Some(IeeeFormat::Binary64),
            _ => None,
        }
    }

    /// IEEE 754 half precision, as h5py writes `numpy.float16`.
    ///
    /// Rust has no stable `f16`, so a dataset of these is read through
    /// [`read_numeric_as`](crate::dataset::H5Dataset::read_numeric_as) as
    /// `f32`, which represents every half exactly.
    pub fn f16_type() -> Self {
        Self::FloatingPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            sign_location: 15,
            bit_offset: 0,
            bit_precision: 16,
            exponent_location: 10,
            exponent_size: 5,
            mantissa_location: 0,
            mantissa_size: 10,
            exponent_bias: 15,
        }
    }

    /// The byte order of this type when its whole element image is one
    /// scalar, so reversing the element's bytes converts it between orders.
    ///
    /// `None` for a composite (compound, array, vlen) whose members each
    /// carry their own order, and for a type that has no byte order at all
    /// (string, opaque). Use [`contains_byte_order`](Self::contains_byte_order)
    /// to ask what a composite stores.
    pub fn scalar_byte_order(&self) -> Option<ByteOrder> {
        match self {
            Self::FixedPoint { byte_order, .. }
            | Self::FloatingPoint { byte_order, .. }
            | Self::BitField { byte_order, .. } => Some(*byte_order),
            // An enum's values are its base type's, stored in its order.
            Self::Enum { base, .. } => base.scalar_byte_order(),
            _ => None,
        }
    }

    /// True when any scalar anywhere in this type tree stores `order`.
    ///
    /// The question a caller reinterpreting a raw element image asks of a
    /// composite type: a compound with one big-endian member cannot be handed
    /// to a host that reads little-endian, however its other members are
    /// stored.
    pub fn contains_byte_order(&self, order: ByteOrder) -> bool {
        match self {
            Self::FixedPoint { byte_order, .. }
            | Self::FloatingPoint { byte_order, .. }
            | Self::BitField { byte_order, .. } => *byte_order == order,
            Self::Enum { base, .. } | Self::Array { base, .. } | Self::VarLenSequence { base } => {
                base.contains_byte_order(order)
            }
            Self::Compound { members, .. } => members
                .iter()
                .any(|m| m.datatype.contains_byte_order(order)),
            Self::Opaque { .. } | Self::FixedString { .. } | Self::VarLenString { .. } => false,
        }
    }

    /// Returns the element size in bytes.
    ///
    /// For `VarLenString`, this returns the on-disk reference size assuming
    /// 8-byte addresses (sizeof_addr=8): 8 + 4 = 12.
    /// Use `element_size_ctx()` for an exact answer with a specific context.
    pub fn element_size(&self) -> u32 {
        match self {
            Self::FixedPoint { size, .. } => *size,
            Self::FloatingPoint { size, .. } => *size,
            Self::BitField { size, .. } => *size,
            Self::Opaque { size, .. } => *size,
            Self::FixedString { size, .. } => *size,
            Self::Compound { size, .. } => *size,
            Self::Enum { base, .. } => base.element_size(),
            Self::VarLenString { .. } | Self::VarLenSequence { .. } => {
                // Default assumption: sizeof_addr = 8
                // vlen ref = 4 (seq_len) + sizeof_addr + 4 (index) = 16
                16
            }
            Self::Array { dims, base } => {
                // `dims` are file-derived; saturate so a crafted array type
                // yields a too-large size (rejected by buffer checks)
                // rather than overflowing.
                let product = dims.iter().fold(1u32, |acc, &d| acc.saturating_mul(d));
                product.saturating_mul(base.element_size())
            }
        }
    }

    /// Returns the element size using an explicit format context.
    ///
    /// This is needed for `VarLenString` where the size depends on
    /// `sizeof_addr`.
    pub fn element_size_ctx(&self, ctx: &FormatContext) -> u32 {
        match self {
            Self::VarLenString { .. } | Self::VarLenSequence { .. } => ctx.sizeof_addr as u32 + 8,
            _ => self.element_size(),
        }
    }

    /// Returns the size of a variable-length reference for a given context.
    pub fn vlen_ref_size(ctx: &FormatContext) -> u32 {
        ctx.sizeof_addr as u32 + 8
    }
}

// ========================================================================= encode / decode

impl DatatypeMessage {
    /// Encode into a byte vector.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        match self {
            Self::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                // Total: 8 header + 4 properties = 12 bytes
                let mut buf = Vec::with_capacity(12);

                // byte 0: class | version<<4
                buf.push(CLASS_FIXED_POINT | (DT_VERSION << 4));

                // bytes 1-3: class bit-field (24 bits LE)
                let mut flags0: u8 = 0;
                if *byte_order == ByteOrder::BigEndian {
                    flags0 |= 0x01; // bit 0
                }
                if *signed {
                    flags0 |= 0x08; // bit 3
                }
                buf.push(flags0);
                buf.push(0); // flags byte 1
                buf.push(0); // flags byte 2

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                // properties: bit_offset(u16) + bit_precision(u16)
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());

                buf
            }
            Self::FloatingPoint {
                size,
                byte_order,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            } => {
                // Total: 8 header + 12 properties = 20 bytes
                let mut buf = Vec::with_capacity(20);

                // byte 0: class | version<<4
                buf.push(CLASS_FLOATING_POINT | (DT_VERSION << 4));

                // bytes 1-3: class bit-field
                let mut flags0: u8 = 0;
                if *byte_order == ByteOrder::BigEndian {
                    flags0 |= 0x01; // bit 0 of byte order
                }
                // bits 4-5: mantissa normalization = 2 (implied leading 1 for IEEE)
                flags0 |= 0x02 << 4; // IMPLIED = 2
                buf.push(flags0);

                // flags byte 1: sign bit position
                buf.push(*sign_location);

                // flags byte 2: unused
                buf.push(0);

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                // properties (12 bytes)
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf.push(*exponent_location);
                buf.push(*exponent_size);
                buf.push(*mantissa_location);
                buf.push(*mantissa_size);
                buf.extend_from_slice(&exponent_bias.to_le_bytes());

                buf
            }
            Self::BitField {
                size,
                byte_order,
                bit_offset,
                bit_precision,
            } => {
                // Same shape as a fixed-point type: 8 header + 4 properties.
                let mut buf = Vec::with_capacity(12);
                buf.push(CLASS_BITFIELD | (DT_VERSION << 4));
                let mut flags0: u8 = 0;
                if *byte_order == ByteOrder::BigEndian {
                    flags0 |= 0x01; // bit 0
                }
                buf.push(flags0);
                buf.push(0);
                buf.push(0);
                buf.extend_from_slice(&size.to_le_bytes());
                buf.extend_from_slice(&bit_offset.to_le_bytes());
                buf.extend_from_slice(&bit_precision.to_le_bytes());
                buf
            }
            Self::Opaque { size, tag } => {
                // The tag occupies a field that is a multiple of 8 bytes, null
                // padded and not necessarily null terminated; its length lives
                // in the low byte of the class bit field (`H5Odtype.c`).
                let tag_len = tag.len().min(OPAQUE_TAG_MAX - 8);
                let aligned = tag_len.div_ceil(8) * 8;

                let mut buf = Vec::with_capacity(8 + aligned);
                buf.push(CLASS_OPAQUE | (DT_VERSION << 4));
                buf.push(aligned as u8);
                buf.push(0);
                buf.push(0);
                buf.extend_from_slice(&size.to_le_bytes());
                buf.extend_from_slice(&tag.as_bytes()[..tag_len]);
                buf.resize(8 + aligned, 0);
                buf
            }
            Self::FixedString {
                size,
                padding,
                charset,
            } => {
                // Total: 8 header bytes, no additional properties
                let mut buf = Vec::with_capacity(8);

                // byte 0: class | version<<4
                buf.push(CLASS_STRING | (DT_VERSION << 4));

                // byte 1: (padding & 0x0f) | ((charset & 0x0f) << 4)
                buf.push((padding & 0x0F) | ((charset & 0x0F) << 4));

                // bytes 2-3: rest of class bit fields (zero)
                buf.push(0);
                buf.push(0);

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                buf
            }
            Self::Compound { size, members } => {
                // Version 3 compound type
                let version: u8 = 3;
                let num_members = members.len() as u16;

                let mut buf = vec![
                    // byte 0: class | version<<4
                    CLASS_COMPOUND | (version << 4),
                    // bytes 1-3: num_members as 16-bit LE in bytes 1-2, byte 3 = 0
                    num_members as u8,
                    (num_members >> 8) as u8,
                    0,
                ];

                // bytes 4-7: element size
                buf.extend_from_slice(&size.to_le_bytes());

                // Version-3 member offsets are encoded in the minimum
                // number of bytes that can represent the compound's size
                // (H5VM_limit_enc_size / UINT32ENCODE_VAR).
                let offset_nbytes = limit_enc_size(*size as u64);

                // Properties: for each member
                for member in members {
                    // Name (null-terminated, no padding in version 3)
                    buf.extend_from_slice(member.name.as_bytes());
                    buf.push(0);

                    // Byte offset, variable width.
                    buf.extend_from_slice(&member.offset.to_le_bytes()[..offset_nbytes]);

                    // Member datatype (recursive)
                    let dt_encoded = member.datatype.encode(ctx);
                    buf.extend_from_slice(&dt_encoded);
                }

                buf
            }
            Self::Enum { base, members } => {
                let num_members = members.len() as u16;
                let base_size = base.element_size();

                let mut buf = vec![
                    // byte 0: class | version<<4
                    CLASS_ENUM | (DT_VERSION << 4),
                    // bytes 1-3: num_members as 16-bit LE
                    num_members as u8,
                    (num_members >> 8) as u8,
                    0,
                ];

                // bytes 4-7: element size (= base type size)
                buf.extend_from_slice(&base_size.to_le_bytes());

                // Properties: base datatype message
                let base_encoded = base.encode(ctx);
                buf.extend_from_slice(&base_encoded);

                // Then each member name (null-terminated, padded to 8-byte boundary)
                for member in members {
                    let name_start = buf.len();
                    buf.extend_from_slice(member.name.as_bytes());
                    buf.push(0);
                    // Pad name field (including null) to 8-byte boundary
                    let name_field_len = buf.len() - name_start;
                    let padded = (name_field_len + 7) & !7;
                    let pad = padded - name_field_len;
                    if pad > 0 {
                        buf.extend_from_slice(&vec![0u8; pad]);
                    }
                }
                // Then all values contiguously
                for member in members {
                    buf.extend_from_slice(&member.value);
                }

                buf
            }
            Self::VarLenString { padding, charset } => {
                // Variable-length string: class 9, version 1
                //
                // On-disk element size = sizeof_addr + 4 (the vlen reference).
                // The flags encode that this is a string-type vlen.
                // Properties: the base type (1-byte char, class 3 string).
                let vlen_size = Self::vlen_ref_size(ctx);

                let mut buf = vec![
                    // byte 0: class 9 | version<<4
                    CLASS_VLEN | (DT_VERSION << 4),
                    // bytes 1-3: flags
                    // byte 1 bits 0-3: type = 1 (string)
                    //         bits 4-7: padding type
                    0x01 | ((*padding & 0x0F) << 4),
                    // byte 2 bits 0-3: charset (0=ASCII, 1=UTF-8)
                    *charset & 0x0F, // charset
                    0,
                ];

                // bytes 4-7: element size
                buf.extend_from_slice(&vlen_size.to_le_bytes());

                // Properties: base type -- 1 byte char (class 3 string, size 1)
                // This is a minimal fixed-string type with size=1.
                let base_type = Self::FixedString {
                    size: 1,
                    padding: 0,
                    charset: *charset,
                };
                let base_encoded = base_type.encode(ctx);
                buf.extend_from_slice(&base_encoded);

                buf
            }
            Self::VarLenSequence { base } => {
                // Variable-length sequence: class 9, version 1.
                //
                // On-disk element size = sizeof_addr + 4 (the vlen reference),
                // identical to a vlen string. The bit field's low nibble is the
                // vlen type (0 = sequence); unlike a vlen string there are no
                // pad/charset bits. The properties embed the parent (base)
                // datatype message recursively (libhdf5 `H5Odtype.c`).
                let vlen_size = Self::vlen_ref_size(ctx);

                let mut buf = vec![
                    // byte 0: class 9 | version<<4
                    CLASS_VLEN | (DT_VERSION << 4),
                    // byte 1 bits 0-3: type = 0 (sequence)
                    0x00,
                    // bytes 2-3: reserved (no charset/pad for sequences)
                    0,
                    0,
                ];

                // bytes 4-7: element size
                buf.extend_from_slice(&vlen_size.to_le_bytes());

                // Properties: base (parent) datatype message, recursive.
                let base_encoded = base.encode(ctx);
                buf.extend_from_slice(&base_encoded);

                buf
            }
            Self::Array { dims, base } => {
                // Array: class 10, version 3
                let version: u8 = 3;
                let base_size = base.element_size();
                let product = dims.iter().fold(1u32, |acc, &d| acc.saturating_mul(d));
                let total_size = product.saturating_mul(base_size);

                let mut buf = vec![
                    // byte 0: class | version<<4
                    CLASS_ARRAY | (version << 4),
                    // bytes 1-3: flags = 0
                    0,
                    0,
                    0,
                ];

                // bytes 4-7: element size (total array size)
                buf.extend_from_slice(&total_size.to_le_bytes());

                // Properties:
                // ndims: u8
                buf.push(dims.len() as u8);

                // dims: ndims * u32 LE
                for &d in dims {
                    buf.extend_from_slice(&d.to_le_bytes());
                }

                // base datatype message (recursive)
                let base_encoded = base.encode(ctx);
                buf.extend_from_slice(&base_encoded);

                buf
            }
        }
    }

    /// Decode from a byte buffer.  Returns `(message, bytes_consumed)`.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        Self::decode_inner(buf, ctx, 0)
    }

    /// Recursive worker for [`decode`]. `depth` bounds datatype nesting:
    /// compound/enum/vlen/array types embed a base datatype recursively, and
    /// a crafted message can nest these deeply enough to exhaust the stack.
    /// libhdf5-written types nest only a handful of levels.
    #[allow(clippy::only_used_in_recursion)]
    fn decode_inner(buf: &[u8], ctx: &FormatContext, depth: usize) -> FormatResult<(Self, usize)> {
        const MAX_DATATYPE_DEPTH: usize = 256;
        if depth > MAX_DATATYPE_DEPTH {
            return Err(FormatError::InvalidData(
                "datatype nesting exceeds maximum depth".into(),
            ));
        }
        if buf.len() < 8 {
            return Err(FormatError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        let class = buf[0] & 0x0F;
        let version = buf[0] >> 4;

        let flags0 = buf[1];
        let flags1 = buf[2];
        // flags2 = buf[3]; // reserved / unused for classes 0 and 1

        let size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);

        // libhdf5 validates the message version once, before dispatching on
        // the class (`H5O__dtype_decode_helper`): anything outside
        // 1..=H5O_DTYPE_VERSION_LATEST is a bad message. Versions 1-4 share
        // every property layout decoded below — the version only selects the
        // name padding, the compound offset width and the array reserved
        // fields — so no class repeats the range check. Version 5 is the
        // HDF5 2.0 encoding introduced for complex numbers; a v5 message is
        // well-formed but not one this crate reads yet.
        if !(1..=DT_VERSION_LATEST).contains(&version) {
            return Err(FormatError::InvalidVersion(version));
        }
        if version > DT_VERSION_MAX_SUPPORTED {
            return Err(FormatError::UnsupportedFeature(format!(
                "datatype message version {version}"
            )));
        }

        match class {
            CLASS_FIXED_POINT => {
                if buf.len() < 12 {
                    return Err(FormatError::BufferTooShort {
                        needed: 12,
                        available: buf.len(),
                    });
                }
                let byte_order = if (flags0 & 0x01) != 0 {
                    ByteOrder::BigEndian
                } else {
                    ByteOrder::LittleEndian
                };
                let signed = (flags0 & 0x08) != 0;

                let bit_offset = u16::from_le_bytes([buf[8], buf[9]]);
                let bit_precision = u16::from_le_bytes([buf[10], buf[11]]);

                Ok((
                    Self::FixedPoint {
                        size,
                        byte_order,
                        signed,
                        bit_offset,
                        bit_precision,
                    },
                    12,
                ))
            }
            CLASS_FLOATING_POINT => {
                if buf.len() < 20 {
                    return Err(FormatError::BufferTooShort {
                        needed: 20,
                        available: buf.len(),
                    });
                }
                let byte_order = if (flags0 & 0x01) != 0 {
                    ByteOrder::BigEndian
                } else {
                    ByteOrder::LittleEndian
                };
                let sign_location = flags1;

                let bit_offset = u16::from_le_bytes([buf[8], buf[9]]);
                let bit_precision = u16::from_le_bytes([buf[10], buf[11]]);
                let exponent_location = buf[12];
                let exponent_size = buf[13];
                let mantissa_location = buf[14];
                let mantissa_size = buf[15];
                let exponent_bias = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);

                Ok((
                    Self::FloatingPoint {
                        size,
                        byte_order,
                        sign_location,
                        bit_offset,
                        bit_precision,
                        exponent_location,
                        exponent_size,
                        mantissa_location,
                        mantissa_size,
                        exponent_bias,
                    },
                    20,
                ))
            }
            CLASS_BITFIELD => {
                // Bit fields carry the same 4 property bytes as a fixed-point
                // type: bit offset and bit precision (`H5Odtype.c`).
                if buf.len() < 12 {
                    return Err(FormatError::BufferTooShort {
                        needed: 12,
                        available: buf.len(),
                    });
                }
                let byte_order = if (flags0 & 0x01) != 0 {
                    ByteOrder::BigEndian
                } else {
                    ByteOrder::LittleEndian
                };
                Ok((
                    Self::BitField {
                        size,
                        byte_order,
                        bit_offset: u16::from_le_bytes([buf[8], buf[9]]),
                        bit_precision: u16::from_le_bytes([buf[10], buf[11]]),
                    },
                    12,
                ))
            }
            CLASS_OPAQUE => {
                // The low byte of the class bit field holds the length of the
                // tag field, which must be a multiple of 8.
                let tag_len = (flags0 as usize) & (OPAQUE_TAG_MAX - 1);
                if !tag_len.is_multiple_of(8) {
                    return Err(FormatError::InvalidData(format!(
                        "opaque tag field length {tag_len} is not a multiple of 8"
                    )));
                }
                if buf.len() < 8 + tag_len {
                    return Err(FormatError::BufferTooShort {
                        needed: 8 + tag_len,
                        available: buf.len(),
                    });
                }
                // The tag is null padded, not necessarily null terminated.
                let raw = &buf[8..8 + tag_len];
                let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                let tag = String::from_utf8_lossy(&raw[..end]).to_string();
                Ok((Self::Opaque { size, tag }, 8 + tag_len))
            }
            CLASS_STRING => {
                // String class: 8-byte header, no additional properties.
                let padding = flags0 & 0x0F;
                let charset = (flags0 >> 4) & 0x0F;

                Ok((
                    Self::FixedString {
                        size,
                        padding,
                        charset,
                    },
                    8,
                ))
            }
            CLASS_COMPOUND => {
                // num_members from flags bytes 1-2 (16-bit LE)
                let num_members = u16::from_le_bytes([flags0, flags1]) as usize;

                let mut pos = 8; // past the 8-byte header

                let mut members = Vec::with_capacity(num_members);
                for _ in 0..num_members {
                    let (name, next) = decode_name_field(buf, pos, version, "compound")?;
                    pos = next;

                    // Byte offset. Versions 1 and 2 use a fixed 4-byte
                    // offset; version 3 uses limit_enc_size(size) bytes
                    // (H5VM_limit_enc_size / UINT32DECODE_VAR).
                    let offset_nbytes = if version >= 3 {
                        limit_enc_size(size as u64)
                    } else {
                        4
                    };
                    if pos + offset_nbytes > buf.len() {
                        return Err(FormatError::BufferTooShort {
                            needed: pos + offset_nbytes,
                            available: buf.len(),
                        });
                    }
                    let offset = read_uint_le(&buf[pos..], offset_nbytes);
                    pos += offset_nbytes;

                    if version == 1 {
                        // Version 1 also carries: dimensionality(1),
                        // reserved(3), dim_perm(4), reserved(4),
                        // dim_sizes(4*4) = 28 bytes.
                        if pos + 28 > buf.len() {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + 28,
                                available: buf.len(),
                            });
                        }
                        pos += 28;
                    }

                    // Member datatype (recursive)
                    let (member_dt, dt_consumed) = Self::decode_inner(&buf[pos..], ctx, depth + 1)?;
                    pos += dt_consumed;

                    members.push(CompoundMember {
                        name,
                        offset,
                        datatype: member_dt,
                    });
                }

                Ok((Self::Compound { size, members }, pos))
            }
            CLASS_ENUM => {
                let num_members = u16::from_le_bytes([flags0, flags1]) as usize;
                let base_size = size;

                let mut pos = 8;

                // Base datatype
                let (base_dt, base_consumed) = Self::decode_inner(&buf[pos..], ctx, depth + 1)?;
                pos += base_consumed;

                // Member names (null-terminated, padded to 8-byte boundary for v1)
                let mut names = Vec::with_capacity(num_members);
                for _ in 0..num_members {
                    let (name, next) = decode_name_field(buf, pos, version, "enum")?;
                    pos = next;
                    names.push(name);
                }

                // Member values (base_size bytes each)
                let mut members = Vec::with_capacity(num_members);
                for name in names {
                    if pos + base_size as usize > buf.len() {
                        return Err(FormatError::BufferTooShort {
                            needed: pos + base_size as usize,
                            available: buf.len(),
                        });
                    }
                    let value = buf[pos..pos + base_size as usize].to_vec();
                    pos += base_size as usize;
                    members.push(EnumMember { name, value });
                }

                Ok((
                    Self::Enum {
                        base: Box::new(base_dt),
                        members,
                    },
                    pos,
                ))
            }
            CLASS_VLEN => {
                // Variable-length types carry no version-dependent property
                // layout at all: libhdf5 only checks that the parent's version
                // does not exceed this one.
                let vlen_type = flags0 & 0x0F;
                // Only a string-type vlen carries these: `H5Odtype.c` reads
                // `pad` from bits 4-7 of the first flag byte and `cset` from
                // the low nibble of the second.
                let padding = (flags0 >> 4) & 0x0F;
                let charset = flags1 & 0x0F;

                let mut pos = 8;

                // Properties: base (parent) datatype
                let (base_dt, base_consumed) = Self::decode_inner(&buf[pos..], ctx, depth + 1)?;
                pos += base_consumed;

                if vlen_type == 1 {
                    // String type
                    Ok((Self::VarLenString { padding, charset }, pos))
                } else {
                    // Sequence type: variable-length array of `base_dt`.
                    Ok((
                        Self::VarLenSequence {
                            base: Box::new(base_dt),
                        },
                        pos,
                    ))
                }
            }
            CLASS_ARRAY => {
                // "There should be no array datatypes with version < 2"
                // (`H5Odtype.c`); the separate array class replaced the
                // intrinsic arrayness of v1 compound members.
                if version < 2 {
                    return Err(FormatError::InvalidVersion(version));
                }
                let mut pos = 8;

                // ndims: u8
                if pos >= buf.len() {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 1,
                        available: buf.len(),
                    });
                }
                let ndims = buf[pos] as usize;
                pos += 1;

                // Versions below 3 have 3 reserved bytes after ndims.
                if version < 3 {
                    pos += 3;
                }

                // dims: ndims * u32 LE
                if pos + ndims * 4 > buf.len() {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + ndims * 4,
                        available: buf.len(),
                    });
                }
                let mut dims = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    let d =
                        u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
                    pos += 4;
                    dims.push(d);
                }

                // Versions below 3 also carry dimension permutation indices.
                if version < 3 {
                    pos += ndims * 4;
                }

                // Base datatype
                let (base_dt, base_consumed) = Self::decode_inner(&buf[pos..], ctx, depth + 1)?;
                pos += base_consumed;

                Ok((
                    Self::Array {
                        dims,
                        base: Box::new(base_dt),
                    },
                    pos,
                ))
            }
            _ => Err(FormatError::UnsupportedFeature(format!(
                "datatype class {}",
                class
            ))),
        }
    }
}

// ========================================================================= Display

impl std::fmt::Display for DatatypeMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixedPoint { size, signed, .. } => {
                let prefix = if *signed { "i" } else { "u" };
                write!(f, "{}{}", prefix, size * 8)
            }
            Self::FloatingPoint { size, .. } => write!(f, "f{}", size * 8),
            Self::BitField {
                size,
                bit_offset,
                bit_precision,
                ..
            } => write!(
                f,
                "bitfield[{}; {}+{}]",
                size * 8,
                bit_offset,
                bit_precision
            ),
            Self::Opaque { size, tag } => write!(f, "opaque[{size}; {tag}]"),
            Self::FixedString { size, charset, .. } => {
                let cs = if *charset == 1 { "UTF-8" } else { "ASCII" };
                write!(f, "string[{}; {}]", size, cs)
            }
            Self::Compound { size, members } => {
                write!(f, "compound({} bytes, {} members)", size, members.len())
            }
            Self::Enum { base, members } => {
                write!(f, "enum<{}; {} members>", base, members.len())
            }
            Self::VarLenString { charset, .. } => {
                let cs = if *charset == 1 { "UTF-8" } else { "ASCII" };
                write!(f, "vlen_string({})", cs)
            }
            Self::VarLenSequence { base } => {
                write!(f, "vlen_sequence<{}>", base)
            }
            Self::Array { dims, base } => {
                let dim_str: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                write!(f, "array[{}; {}]", dim_str.join("x"), base)
            }
        }
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
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

    // ---- recursion / overflow hardening ----

    #[test]
    fn decode_rejects_deeply_nested_datatype() {
        // A crafted chain of vlen-of-vlen-of-... datatypes (8 bytes per
        // level) must be rejected by the depth guard, not recurse until the
        // stack overflows. Each level: byte0 = class 9 (vlen) | version 1.
        let levels = 4096;
        let mut buf = Vec::new();
        for _ in 0..levels {
            buf.extend_from_slice(&[(CLASS_VLEN | (1 << 4)), 1, 0, 0, 0, 0, 0, 0]);
        }
        let result = DatatypeMessage::decode(&buf, &ctx());
        assert!(
            matches!(result, Err(FormatError::InvalidData(_))),
            "expected depth-limit error, got {result:?}"
        );
    }

    #[test]
    fn array_element_size_saturates_on_absurd_dims() {
        // A crafted array datatype with huge dims must not overflow when its
        // element size is computed; it saturates instead.
        let msg = DatatypeMessage::Array {
            dims: vec![u32::MAX, u32::MAX, u32::MAX],
            base: Box::new(DatatypeMessage::u64_type()),
        };
        assert_eq!(msg.element_size(), u32::MAX);
    }

    // ---- fixed point roundtrips ----

    #[test]
    fn roundtrip_u8() {
        let msg = DatatypeMessage::u8_type();
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 12);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i8() {
        let msg = DatatypeMessage::i8_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_u16() {
        let msg = DatatypeMessage::u16_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i16() {
        let msg = DatatypeMessage::i16_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_u32() {
        let msg = DatatypeMessage::u32_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i32() {
        let msg = DatatypeMessage::i32_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_u64() {
        let msg = DatatypeMessage::u64_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_i64() {
        let msg = DatatypeMessage::i64_type();
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    // ---- floating point roundtrips ----

    #[test]
    fn roundtrip_f32() {
        let msg = DatatypeMessage::f32_type();
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 20);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_f64() {
        let msg = DatatypeMessage::f64_type();
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 20);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(decoded, msg);
    }

    // ---- edge / error cases ----

    #[test]
    fn fixed_point_big_endian() {
        let msg = DatatypeMessage::FixedPoint {
            size: 4,
            byte_order: ByteOrder::BigEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn floating_point_big_endian() {
        let msg = DatatypeMessage::FloatingPoint {
            size: 8,
            byte_order: ByteOrder::BigEndian,
            sign_location: 63,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        let (decoded, _) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8; 4];
        let err = DatatypeMessage::decode(&buf, &ctx()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_unsupported_class() {
        // class 7 (reference), version 1
        let mut buf = [0u8; 12];
        buf[0] = 7 | (1 << 4);
        buf[4] = 1; // size = 1
        let err = DatatypeMessage::decode(&buf, &ctx()).unwrap_err();
        match err {
            FormatError::UnsupportedFeature(_) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_encoding() {
        let encoded = DatatypeMessage::u32_type().encode(&ctx());
        assert_eq!(encoded[0] >> 4, DT_VERSION);
        assert_eq!(encoded[0] & 0x0F, CLASS_FIXED_POINT);
    }

    #[test]
    fn signed_flag_encoding() {
        let unsigned = DatatypeMessage::u32_type().encode(&ctx());
        let signed = DatatypeMessage::i32_type().encode(&ctx());
        assert_eq!(unsigned[1] & 0x08, 0);
        assert_eq!(signed[1] & 0x08, 0x08);
    }

    // ---- fixed string roundtrips ----

    #[test]
    fn roundtrip_fixed_string_ascii() {
        let msg = DatatypeMessage::fixed_string(10);
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 8); // 8-byte header, no properties
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_fixed_string_utf8() {
        let msg = DatatypeMessage::fixed_string_utf8(20);
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 8);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn fixed_string_element_size() {
        let msg = DatatypeMessage::fixed_string(42);
        assert_eq!(msg.element_size(), 42);
    }

    #[test]
    fn fixed_string_class_encoding() {
        let encoded = DatatypeMessage::fixed_string(5).encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, 3); // class = 3
        assert_eq!(encoded[0] >> 4, DT_VERSION); // version = 1
    }

    #[test]
    fn fixed_string_charset_encoding() {
        let ascii = DatatypeMessage::fixed_string(5).encode(&ctx());
        assert_eq!(ascii[1] & 0x0F, 0); // padding = null terminate
        assert_eq!((ascii[1] >> 4) & 0x0F, 0); // charset = ASCII

        let utf8 = DatatypeMessage::fixed_string_utf8(5).encode(&ctx());
        assert_eq!(utf8[1] & 0x0F, 0); // padding = null terminate
        assert_eq!((utf8[1] >> 4) & 0x0F, 1); // charset = UTF-8
    }

    // ---- vlen string roundtrips ----

    #[test]
    fn roundtrip_vlen_string_utf8() {
        let msg = DatatypeMessage::vlen_string_utf8();
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_vlen_string_ascii() {
        let msg = DatatypeMessage::vlen_string_ascii();
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn vlen_string_element_size() {
        let msg = DatatypeMessage::vlen_string_utf8();
        // Default: sizeof_addr=8, so 4+8+4 = 16
        assert_eq!(msg.element_size(), 16);
        assert_eq!(msg.element_size_ctx(&ctx()), 16);
        assert_eq!(msg.element_size_ctx(&ctx4()), 12);
    }

    #[test]
    fn vlen_string_class_encoding() {
        let encoded = DatatypeMessage::vlen_string_utf8().encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, CLASS_VLEN); // class = 9
        assert_eq!(encoded[0] >> 4, DT_VERSION); // version = 1
        assert_eq!(encoded[1] & 0x0F, 1); // type = string
        assert_eq!(encoded[2] & 0x0F, 1); // charset = UTF-8
    }

    #[test]
    fn vlen_string_4byte_ctx() {
        let c = ctx4();
        let msg = DatatypeMessage::vlen_string_utf8();
        let encoded = msg.encode(&c);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &c).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
        // Size field in the encoded bytes should be 4+4+4=12
        let sz = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
        assert_eq!(sz, 12);
    }

    /// The vlen-string datatype message h5py's `string_dtype("ascii")` puts in
    /// the file. The pad is the high nibble of the first flag byte and the
    /// character set the low nibble of the second (`H5Odtype.c`), and the
    /// parent libhdf5 stores is an unsigned 8-bit integer, not a 1-byte
    /// string — which is why only the decode is asserted against these bytes.
    #[test]
    fn decode_vlen_string_pad_from_libhdf5() {
        let nullterm: [u8; 20] = [
            0x19, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, // vlen: string, pad 0, ASCII
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
            0x00, // parent u8
        ];
        let (decoded, consumed) = DatatypeMessage::decode(&nullterm, &ctx()).unwrap();
        assert_eq!(consumed, nullterm.len());
        assert_eq!(
            decoded,
            DatatypeMessage::VarLenString {
                padding: 0,
                charset: 0,
            }
        );

        // The same type with H5T_STR_NULLPAD and H5T_CSET_UTF8.
        let mut nullpad = nullterm;
        nullpad[1] = 0x11;
        nullpad[2] = 0x01;
        let (decoded, _) = DatatypeMessage::decode(&nullpad, &ctx()).unwrap();
        assert_eq!(
            decoded,
            DatatypeMessage::VarLenString {
                padding: 1,
                charset: 1,
            }
        );
    }

    /// The pad survives an encode/decode round trip, which is what lets a
    /// space-padded vlen string be reported rather than silently normalized.
    #[test]
    fn roundtrip_vlen_string_pad_and_charset() {
        for padding in 0..=2u8 {
            for charset in 0..=1u8 {
                let msg = DatatypeMessage::VarLenString { padding, charset };
                let encoded = msg.encode(&ctx());
                assert_eq!(encoded[1] & 0x0F, 1, "type is still string");
                assert_eq!((encoded[1] >> 4) & 0x0F, padding);
                assert_eq!(encoded[2] & 0x0F, charset);
                let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
                assert_eq!(consumed, encoded.len());
                assert_eq!(decoded, msg);
            }
        }
    }

    // ---- fixed string padding ----

    /// `H5T__conv_s_s`: null-terminated and null-padded both end at the first
    /// NUL, space-padded ends after the last non-space byte.
    #[test]
    fn fixed_string_content_follows_the_declared_pad() {
        assert_eq!(fixed_string_content(b"alpha\0\0\0", 0).unwrap(), b"alpha");
        assert_eq!(fixed_string_content(b"alpha\0\0\0", 1).unwrap(), b"alpha");
        assert_eq!(fixed_string_content(b"alpha   ", 2).unwrap(), b"alpha");

        // Trailing spaces are content unless the pad says otherwise, and an
        // embedded NUL survives a space-padded element.
        assert_eq!(fixed_string_content(b"ab  \0\0\0\0", 0).unwrap(), b"ab  ");
        assert_eq!(fixed_string_content(b"a\0b     ", 2).unwrap(), b"a\0b");
        assert_eq!(fixed_string_content(b"a\0b\0\0\0\0\0", 1).unwrap(), b"a");

        // An element that is all padding is empty, not the whole field.
        assert_eq!(fixed_string_content(b"        ", 2).unwrap(), b"");
        assert_eq!(fixed_string_content(b"\0\0\0\0", 0).unwrap(), b"");

        // Nothing to strip.
        assert_eq!(fixed_string_content(b"exact", 0).unwrap(), b"exact");
        assert_eq!(fixed_string_content(b"exact", 2).unwrap(), b"exact");

        // Padding rules 3-15 are reserved; libhdf5 fails the conversion.
        assert!(fixed_string_content(b"xxxx", 3).is_none());
        assert!(fixed_string_content(b"xxxx", 15).is_none());
    }

    // ---- vlen sequence (byte array) roundtrips ----

    #[test]
    fn roundtrip_vlen_bytes() {
        let msg = DatatypeMessage::vlen_bytes();
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
        // Base type is a u8 fixed-point.
        match decoded {
            DatatypeMessage::VarLenSequence { base } => {
                assert_eq!(*base, DatatypeMessage::u8_type());
            }
            other => panic!("expected VarLenSequence, got {:?}", other),
        }
    }

    #[test]
    fn vlen_bytes_class_encoding() {
        let encoded = DatatypeMessage::vlen_bytes().encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, CLASS_VLEN); // class = 9
        assert_eq!(encoded[0] >> 4, DT_VERSION); // version = 1
        assert_eq!(encoded[1] & 0x0F, 0); // type = sequence (0)
    }

    #[test]
    fn vlen_bytes_element_size() {
        let msg = DatatypeMessage::vlen_bytes();
        // Same on-disk vlen reference size as a vlen string.
        assert_eq!(msg.element_size(), 16);
        assert_eq!(msg.element_size_ctx(&ctx()), 16);
        assert_eq!(msg.element_size_ctx(&ctx4()), 12);
    }

    #[test]
    fn vlen_bytes_4byte_ctx() {
        let c = ctx4();
        let msg = DatatypeMessage::vlen_bytes();
        let encoded = msg.encode(&c);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &c).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
        let sz = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
        assert_eq!(sz, 12);
    }

    // ---- message versions ----

    /// Retag an encoded message with a different version nibble.
    fn with_version(mut msg: Vec<u8>, version: u8) -> Vec<u8> {
        msg[0] = (msg[0] & 0x0F) | (version << 4);
        msg
    }

    /// Versions 1-4 share the property layout of every class this crate
    /// decodes, so a v4-tagged message — what libhdf5 emits under a
    /// `H5F_LIBVER_V112` or later low bound — must decode, not be rejected.
    #[test]
    fn decode_accepts_version_4() {
        for msg in [
            DatatypeMessage::u32_type(),
            DatatypeMessage::f64_type(),
            DatatypeMessage::fixed_string(8),
            DatatypeMessage::BitField {
                size: 1,
                byte_order: ByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 8,
            },
            DatatypeMessage::Opaque {
                size: 4,
                tag: "raw4".to_string(),
            },
            DatatypeMessage::vlen_string_utf8(),
            DatatypeMessage::vlen_bytes(),
        ] {
            let v4 = with_version(msg.encode(&ctx()), 4);
            let (decoded, _) = DatatypeMessage::decode(&v4, &ctx())
                .unwrap_or_else(|e| panic!("v4 {msg} rejected: {e:?}"));
            assert_eq!(decoded, msg, "v4 {msg} decoded differently");
        }
    }

    /// A v4 compound follows the v3 rules: no name padding, minimum-width
    /// member offsets.
    #[test]
    fn decode_accepts_version_4_compound() {
        let msg = DatatypeMessage::compound(
            12,
            vec![
                CompoundMember {
                    name: "x".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::i32_type(),
                },
                CompoundMember {
                    name: "y".to_string(),
                    offset: 4,
                    datatype: DatatypeMessage::f64_type(),
                },
            ],
        );
        let v4 = with_version(msg.encode(&ctx()), 4);
        let (decoded, consumed) = DatatypeMessage::decode(&v4, &ctx()).unwrap();
        assert_eq!(consumed, v4.len());
        assert_eq!(decoded, msg);
    }

    /// Version 5 is the HDF5 2.0 encoding; it is legal on the wire, so the
    /// error must name it as unsupported rather than claim a bad version.
    #[test]
    fn decode_reports_version_5_as_unsupported() {
        let v5 = with_version(DatatypeMessage::u32_type().encode(&ctx()), 5);
        let err = DatatypeMessage::decode(&v5, &ctx()).unwrap_err();
        assert!(
            matches!(&err, FormatError::UnsupportedFeature(m)
                     if m == "datatype message version 5"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn decode_rejects_versions_outside_the_format() {
        for version in [0u8, 6, 15] {
            let msg = with_version(DatatypeMessage::u32_type().encode(&ctx()), version);
            let err = DatatypeMessage::decode(&msg, &ctx()).unwrap_err();
            assert!(
                matches!(err, FormatError::InvalidVersion(v) if v == version),
                "version {version}: unexpected error: {err:?}"
            );
        }
    }

    /// An array datatype below version 2 does not exist in the format.
    #[test]
    fn decode_rejects_array_version_1() {
        let msg = DatatypeMessage::array(vec![2], DatatypeMessage::f32_type());
        let v1 = with_version(msg.encode(&ctx()), 1);
        let err = DatatypeMessage::decode(&v1, &ctx()).unwrap_err();
        assert!(
            matches!(err, FormatError::InvalidVersion(1)),
            "unexpected error: {err:?}"
        );
    }

    // ---- opaque / bit field ----

    /// The opaque datatype message libhdf5 writes for `H5Tcreate(H5T_OPAQUE, 4)`
    /// with tag "raw4" (from `H5Tencode`, minus its 2-byte prefix): the tag
    /// field is padded to 8 bytes and is not null terminated at its end.
    #[test]
    fn decode_opaque_from_libhdf5() {
        let msg: [u8; 16] = [
            0x15, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, b'r', b'a', b'w', b'4', 0, 0, 0, 0,
        ];
        let (decoded, consumed) = DatatypeMessage::decode(&msg, &ctx()).unwrap();
        assert_eq!(consumed, msg.len());
        assert_eq!(
            decoded,
            DatatypeMessage::Opaque {
                size: 4,
                tag: "raw4".to_string(),
            }
        );
        assert_eq!(decoded.element_size(), 4);
        assert_eq!(decoded.encode(&ctx()), msg);
    }

    #[test]
    fn decode_opaque_rejects_unaligned_tag_field() {
        let msg: [u8; 16] = [
            0x15, 0x05, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, b'r', b'a', b'w', b'4', 0, 0, 0, 0,
        ];
        let err = DatatypeMessage::decode(&msg, &ctx()).unwrap_err();
        assert!(
            matches!(&err, FormatError::InvalidData(m) if m.contains("multiple of 8")),
            "unexpected error: {err:?}"
        );
    }

    /// `H5T_STD_B8LE` as libhdf5 encodes it.
    #[test]
    fn decode_bitfield_from_libhdf5() {
        let msg: [u8; 12] = [
            0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00,
        ];
        let (decoded, consumed) = DatatypeMessage::decode(&msg, &ctx()).unwrap();
        assert_eq!(consumed, msg.len());
        assert_eq!(
            decoded,
            DatatypeMessage::BitField {
                size: 1,
                byte_order: ByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 8,
            }
        );
        assert_eq!(decoded.element_size(), 1);
        assert_eq!(decoded.encode(&ctx()), msg);
    }

    #[test]
    fn roundtrip_bitfield_be_narrow() {
        let msg = DatatypeMessage::BitField {
            size: 4,
            byte_order: ByteOrder::BigEndian,
            bit_offset: 3,
            bit_precision: 12,
        };
        let (decoded, consumed) = DatatypeMessage::decode(&msg.encode(&ctx()), &ctx()).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_opaque_tag_padding() {
        // 9-byte tag: the field grows to 16 bytes, the tag is truncated by
        // neither side.
        let msg = DatatypeMessage::Opaque {
            size: 24,
            tag: "nine-char".to_string(),
        };
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded.len(), 8 + 16);
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    // ---- compound roundtrips ----

    #[test]
    fn roundtrip_compound_simple() {
        let msg = DatatypeMessage::compound(
            12, // i32 + f64 = 4 + 8 = 12
            vec![
                CompoundMember {
                    name: "x".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::i32_type(),
                },
                CompoundMember {
                    name: "y".to_string(),
                    offset: 4,
                    datatype: DatatypeMessage::f64_type(),
                },
            ],
        );
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn compound_element_size() {
        let msg = DatatypeMessage::compound(
            16,
            vec![
                CompoundMember {
                    name: "a".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::u64_type(),
                },
                CompoundMember {
                    name: "b".to_string(),
                    offset: 8,
                    datatype: DatatypeMessage::u64_type(),
                },
            ],
        );
        assert_eq!(msg.element_size(), 16);
    }

    /// A version-2 compound message as h5py writes it (default libver, one
    /// array-typed member forces version 2): member names are padded to a
    /// multiple of 8 bytes, exactly as in version 1. Decoding it with the
    /// version-3 rule misreads the offset of every member.
    ///
    /// Byte image lifted from an h5py 3.15 file holding
    /// `dtype([('alpha','<i4'), ('beta',('<f4',(2,)))])`.
    #[test]
    fn decode_v2_compound_pads_member_names() {
        #[rustfmt::skip]
        let msg: [u8; 84] = [
            0x26, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, // v2 compound, 2 members, 12 bytes
            b'a', b'l', b'p', b'h', b'a', 0, 0, 0,          // "alpha" padded to 8
            0x00, 0x00, 0x00, 0x00,                         // offset 0 (4 bytes in v1/v2)
            0x10, 0x08, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, // i32
            0x00, 0x00, 0x20, 0x00,
            b'b', b'e', b't', b'a', 0, 0, 0, 0,             // "beta" padded to 8
            0x04, 0x00, 0x00, 0x00,                         // offset 4
            0x2a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, // v2 array, 8 bytes
            0x01, 0x00, 0x00, 0x00,                         // ndims 1 + 3 reserved
            0x02, 0x00, 0x00, 0x00,                         // dim 2
            0x00, 0x00, 0x00, 0x00,                         // permutation
            0x11, 0x20, 0x1f, 0x00, 0x04, 0x00, 0x00, 0x00, // f32
            0x00, 0x00, 0x20, 0x00, 0x17, 0x08, 0x00, 0x17,
            0x7f, 0x00, 0x00, 0x00,
        ];
        let (decoded, consumed) = DatatypeMessage::decode(&msg, &ctx()).unwrap();
        assert_eq!(consumed, msg.len());
        assert_eq!(
            decoded,
            DatatypeMessage::compound(
                12,
                vec![
                    CompoundMember {
                        name: "alpha".to_string(),
                        offset: 0,
                        datatype: DatatypeMessage::i32_type(),
                    },
                    CompoundMember {
                        name: "beta".to_string(),
                        offset: 4,
                        datatype: DatatypeMessage::array(vec![2], DatatypeMessage::f32_type()),
                    },
                ],
            )
        );
    }

    /// The v1/v2 name padding may not read past a truncated message.
    #[test]
    fn decode_v2_compound_name_padding_past_end_errors() {
        let mut msg = vec![
            0x26, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, // v2 compound, 1 member, 4 bytes
        ];
        msg.extend_from_slice(b"ab\0"); // name field would pad to 8 bytes
        let err = DatatypeMessage::decode(&msg, &ctx()).unwrap_err();
        assert!(
            matches!(err, FormatError::BufferTooShort { .. }),
            "expected BufferTooShort, got {err:?}"
        );
    }

    #[test]
    fn compound_class_encoding() {
        let msg = DatatypeMessage::compound(
            4,
            vec![CompoundMember {
                name: "val".to_string(),
                offset: 0,
                datatype: DatatypeMessage::i32_type(),
            }],
        );
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, CLASS_COMPOUND); // class = 6
        assert_eq!(encoded[0] >> 4, 3); // version = 3
    }

    #[test]
    fn roundtrip_compound_nested() {
        let inner = DatatypeMessage::compound(
            8,
            vec![
                CompoundMember {
                    name: "re".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::f32_type(),
                },
                CompoundMember {
                    name: "im".to_string(),
                    offset: 4,
                    datatype: DatatypeMessage::f32_type(),
                },
            ],
        );
        let msg = DatatypeMessage::compound(
            12,
            vec![
                CompoundMember {
                    name: "id".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::u32_type(),
                },
                CompoundMember {
                    name: "value".to_string(),
                    offset: 4,
                    datatype: inner,
                },
            ],
        );
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    // ---- enum roundtrips ----

    #[test]
    fn roundtrip_enum_simple() {
        let msg = DatatypeMessage::enumeration(
            DatatypeMessage::u8_type(),
            vec![
                EnumMember {
                    name: "RED".to_string(),
                    value: vec![0],
                },
                EnumMember {
                    name: "GREEN".to_string(),
                    value: vec![1],
                },
                EnumMember {
                    name: "BLUE".to_string(),
                    value: vec![2],
                },
            ],
        );
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn enum_element_size() {
        let msg = DatatypeMessage::enumeration(
            DatatypeMessage::i32_type(),
            vec![
                EnumMember {
                    name: "A".to_string(),
                    value: vec![0, 0, 0, 0],
                },
                EnumMember {
                    name: "B".to_string(),
                    value: vec![1, 0, 0, 0],
                },
            ],
        );
        assert_eq!(msg.element_size(), 4);
    }

    #[test]
    fn enum_class_encoding() {
        let msg = DatatypeMessage::enumeration(
            DatatypeMessage::u8_type(),
            vec![EnumMember {
                name: "X".to_string(),
                value: vec![0],
            }],
        );
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, CLASS_ENUM);
        assert_eq!(encoded[0] >> 4, DT_VERSION);
    }

    // ---- array roundtrips ----

    #[test]
    fn roundtrip_array_1d() {
        let msg = DatatypeMessage::array(vec![10], DatatypeMessage::f64_type());
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_array_2d() {
        let msg = DatatypeMessage::array(vec![3, 4], DatatypeMessage::i32_type());
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn array_element_size() {
        let msg = DatatypeMessage::array(vec![3, 4], DatatypeMessage::i32_type());
        assert_eq!(msg.element_size(), 3 * 4 * 4); // 48
    }

    #[test]
    fn array_class_encoding() {
        let msg = DatatypeMessage::array(vec![5], DatatypeMessage::u8_type());
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded[0] & 0x0F, CLASS_ARRAY); // class = 10
        assert_eq!(encoded[0] >> 4, 3); // version = 3
    }

    #[test]
    fn roundtrip_array_of_compound() {
        let compound = DatatypeMessage::compound(
            8,
            vec![
                CompoundMember {
                    name: "x".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::f32_type(),
                },
                CompoundMember {
                    name: "y".to_string(),
                    offset: 4,
                    datatype: DatatypeMessage::f32_type(),
                },
            ],
        );
        let msg = DatatypeMessage::array(vec![10], compound);
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
        assert_eq!(msg.element_size(), 80); // 10 * 8
    }
}
