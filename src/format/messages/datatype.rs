//! Datatype message (type 0x03) — describes element data type.
//!
//! Binary layout:
//!   Byte 0:    (class & 0x0F) | (version << 4)     version = 1
//!   Bytes 1-3: class bit-field flags (24 bits, little-endian)
//!   Bytes 4-7: element size (u32 LE)
//!   Bytes 8+:  class-specific properties

use crate::format::{FormatContext, FormatError, FormatResult, LibverBound};

/// The version `H5T__alloc` stamps on every fresh datatype (H5T.c:4030), and
/// the one an atomic class never leaves: `H5T__upgrade_version_cb`
/// (H5T.c:6509-6546) raises compound, array, enum and vlen and nothing else.
const DT_VERSION: u8 = 1;

/// `H5O_DTYPE_VERSION_2`, the floor `H5T__array_create` gives every array
/// (H5Tarray.c:169). It is the version that introduced the array class, so no
/// array message may claim less (H5Odtype.c:823-824, :1316-1317).
const ARRAY_MIN_VERSION: u8 = 2;

/// `H5O_DTYPE_VERSION_3`, the version that packed the member encodings:
/// compound and enum member names lost their padding to a multiple of 8, the
/// compound member offset shrank to `H5VM_limit_enc_size` bytes, and the array
/// dropped its reserved bytes and dimension permutations (H5Tpkg.h:85-91).
const DT_VERSION_PACKED: u8 = 3;

/// The bytes a version-1 compound member spends on the intrinsic 'arrayness'
/// that the array class replaced: dimensionality (1), reserved (3), dimension
/// permutation (4), reserved (4) and four dimensions (16), which
/// `H5O__dtype_encode_helper` writes as zeros (H5Odtype.c:1227-1247) and
/// `H5O__dtype_size` accounts for the same way (H5Odtype.c:1606-1611).
const V1_COMPOUND_MEMBER_ARRAY_BYTES: usize = 28;

/// libhdf5 `H5O_DTYPE_VERSION_LATEST`: the highest datatype message version
/// the format defines (5, the HDF5 2.0 encoding).
const DT_VERSION_LATEST: u8 = 5;

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
    let advance = if version >= DT_VERSION_PACKED {
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

/// The byte order an atomic class declares in its bit field.
///
/// `H5O__dtype_decode_helper` reads bit 0 for every atomic class — set is
/// big-endian, clear little-endian. A floating-point message of version 3 or
/// later also gives bit 6 a meaning: with bit 0 it is `H5T_ORDER_VAX`, and
/// without it libhdf5 fails the message with "bad byte order for datatype
/// message". No other class reads bit 6, and neither does a v1/v2 float, so
/// `reads_vax_bit` says which rule applies.
///
/// VAX is a distinct middle-endian layout, not a permutation of the two
/// orders this crate stores: `H5T_VAX_F8` carries its own exponent bias
/// (0x401) over otherwise IEEE-shaped fields. Decoding one as little- or
/// big-endian would hand back numbers the file does not hold, so it is named
/// and refused instead.
fn atomic_byte_order(flags0: u8, reads_vax_bit: bool) -> FormatResult<ByteOrder> {
    let big_endian = (flags0 & 0x01) != 0;
    if reads_vax_bit && (flags0 & 0x40) != 0 {
        if !big_endian {
            return Err(FormatError::InvalidData(
                "bad byte order for datatype message: bit 6 is set without bit 0".into(),
            ));
        }
        return Err(FormatError::UnsupportedFeature(
            "VAX byte order (H5T_ORDER_VAX)".into(),
        ));
    }
    Ok(if big_endian {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    })
}

// Datatype class codes
const CLASS_FIXED_POINT: u8 = 0;
const CLASS_FLOATING_POINT: u8 = 1;
const CLASS_STRING: u8 = 3;
const CLASS_BITFIELD: u8 = 4;
const CLASS_OPAQUE: u8 = 5;
const CLASS_COMPOUND: u8 = 6;
const CLASS_REFERENCE: u8 = 7;
const CLASS_ENUM: u8 = 8;
const CLASS_VLEN: u8 = 9;
const CLASS_ARRAY: u8 = 10;

/// The name `H5O__dtype_debug` prints for a class (H5Odtype.c:1959-2010),
/// spelled the way this crate's canon field spells it: one word, so the class
/// and its version fit a `class:version` pair.
fn class_name(class: u8) -> &'static str {
    match class {
        CLASS_FIXED_POINT => "integer",
        CLASS_FLOATING_POINT => "float",
        2 => "time",
        CLASS_STRING => "string",
        CLASS_BITFIELD => "bitfield",
        CLASS_OPAQUE => "opaque",
        CLASS_COMPOUND => "compound",
        CLASS_REFERENCE => "reference",
        CLASS_ENUM => "enum",
        CLASS_VLEN => "vlen",
        CLASS_ARRAY => "array",
        _ => "unknown",
    }
}

/// One datatype message in an encoded tree: how deep it sits under the
/// outermost message, its class and the version its own header byte claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatatypeNodeVersion {
    /// 0 for the outermost message, one more for each nesting level.
    pub depth: usize,
    /// The class name `class_name` gives, e.g. `compound`.
    pub class: &'static str,
    /// The version nibble of the message's first byte.
    pub version: u8,
}

/// libhdf5 `H5R_ENCODE_VERSION` (`H5Rprivate.h`): the only encoding version
/// the 1.12 reference kinds accept, stored in the bit field's second nibble.
const REFERENCE_ENCODE_VERSION: u8 = 1;

/// `H5T_STD_REF` is born at datatype message version 4 (`H5T_INIT_TYPE_REF_CORE`
/// in H5T.c), so a message holding one of the 1.12 reference kinds carries at
/// least that version whatever the file's libver bound is.
const REVISED_REFERENCE_MESSAGE_VERSION: u8 = 4;

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
    /// Reference datatype (class 7): an element that names another object, or
    /// a region of one, in this file.
    ///
    /// The message carries no properties — `H5O__dtype_decode_helper` reads
    /// only the class bit field, whose low nibble is the `H5R_type_t` — so
    /// `size` is where the element width lives: 8 (one address) for
    /// [`ReferenceKind::Object1`] and 12 (a global-heap id) for
    /// [`ReferenceKind::DatasetRegion1`] in a file with 8-byte addresses.
    Reference {
        /// Element size in bytes.
        size: u32,
        /// Which flavor of reference the elements are.
        kind: ReferenceKind,
    },
}

/// The flavors of reference an element can be (`H5R_type_t`, `H5Rpublic.h`).
///
/// The first two are the pre-1.12 encodings — what h5py 3.x writes today —
/// and store a file address directly. The last three are the 1.12 revised
/// encodings, whose elements are opaque tokens carrying an encoding version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// `H5R_OBJECT1`: the object header address of the target.
    Object1,
    /// `H5R_DATASET_REGION1`: a global-heap id whose heap object holds the
    /// target's object header address followed by a serialized dataspace
    /// selection.
    DatasetRegion1,
    /// `H5R_OBJECT2`: the 1.12 object reference.
    Object2,
    /// `H5R_DATASET_REGION2`: the 1.12 region reference.
    DatasetRegion2,
    /// `H5R_ATTR`: a reference to an attribute (1.12).
    Attr,
}

impl ReferenceKind {
    /// The `H5R_type_t` value this kind is stored as.
    pub fn code(self) -> u8 {
        match self {
            Self::Object1 => 0,
            Self::DatasetRegion1 => 1,
            Self::Object2 => 2,
            Self::DatasetRegion2 => 3,
            Self::Attr => 4,
        }
    }

    /// The kind a stored `H5R_type_t` names, or `None` for a value the format
    /// does not define (`>= H5R_MAXTYPE`, which libhdf5 rejects as an invalid
    /// reference type).
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Object1,
            1 => Self::DatasetRegion1,
            2 => Self::Object2,
            3 => Self::DatasetRegion2,
            4 => Self::Attr,
            _ => return None,
        })
    }

    /// Whether this kind's elements are 1.12 encoded references, which carry
    /// the encoding version in the bit field's second nibble.
    pub fn is_revised(self) -> bool {
        matches!(self.encoding(), ReferenceEncoding::Revised)
    }

    /// How elements of this kind are laid out.
    ///
    /// The split is what element decoding dispatches on: a pre-1.12 element is
    /// whatever the datatype message says it is, while every 1.12 element
    /// repeats its own reference type in its first byte — `H5T__ref_disk_read`
    /// reads that byte rather than consulting the datatype, because
    /// `H5T_STD_REF` is stored as `H5R_OBJECT2` no matter which of the three
    /// revised kinds an element turns out to hold.
    pub fn encoding(self) -> ReferenceEncoding {
        match self {
            Self::Object1 => ReferenceEncoding::Old(OldReferenceKind::Object),
            Self::DatasetRegion1 => ReferenceEncoding::Old(OldReferenceKind::DatasetRegion),
            Self::Object2 | Self::DatasetRegion2 | Self::Attr => ReferenceEncoding::Revised,
        }
    }
}

/// The element layout a [`ReferenceKind`] implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceEncoding {
    /// A pre-1.12 element: a bare file address, or a global-heap id.
    Old(OldReferenceKind),
    /// A 1.12 encoded reference, which names its own kind.
    Revised,
}

/// The two pre-1.12 reference kinds — the ones whose elements name a file
/// address this crate can follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OldReferenceKind {
    /// `H5R_OBJECT1`.
    Object,
    /// `H5R_DATASET_REGION1`.
    DatasetRegion,
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

    /// Object reference type (`H5T_STD_REF_OBJ`): one object header address
    /// per element, so the width follows the file's address size.
    pub fn object_reference(ctx: &FormatContext) -> Self {
        Self::Reference {
            size: ctx.sizeof_addr as u32,
            kind: ReferenceKind::Object1,
        }
    }

    /// Revised object reference type (`H5T_STD_REF`, the 1.12 form): the
    /// two-byte header every 1.12 element leads with, over the wider of the
    /// two payloads `H5T__ref_disk_getsize` sizes an element for.
    ///
    /// Those two are the encoded reference stored inline — a token length byte
    /// and the token — and the global-heap blob id everything else stores, a
    /// 4-byte size then the collection address then a 4-byte object index. The
    /// blob id is the wider, so it is what sets the width: 18 bytes over
    /// 8-byte addresses, which is what libhdf5 writes for an `H5T_STD_REF`
    /// dataset whatever its elements turn out to name.
    pub fn std_object_reference(ctx: &FormatContext) -> Self {
        Self::Reference {
            size: 2 + 4 + ctx.sizeof_addr as u32 + 4,
            kind: ReferenceKind::Object2,
        }
    }

    /// Dataset region reference type (`H5T_STD_REF_DSETREG`): one global-heap
    /// id per element, which is an address plus a 4-byte object index.
    pub fn region_reference(ctx: &FormatContext) -> Self {
        Self::Reference {
            size: ctx.sizeof_addr as u32 + 4,
            kind: ReferenceKind::DatasetRegion1,
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
            Self::Opaque { .. }
            | Self::FixedString { .. }
            | Self::VarLenString { .. }
            | Self::Reference { .. } => false,
        }
    }

    /// Whether this is the definition of one of libhdf5's predefined library
    /// types — the ones `H5T_INIT_TYPE` stamps `H5T_STATE_IMMUTABLE`
    /// (H5T.c:461), which is what `H5T_is_immutable` reports (H5T.c:6699-6712)
    /// and what `H5O__dtype_can_share` refuses to share (H5Odtype.c:1893-1896).
    ///
    /// The question is about the definition, not about provenance. libhdf5
    /// tells `H5T_STD_I32LE` apart from `H5Tcopy(H5T_STD_I32LE)` — which
    /// encode to the same message — by the state of the `H5T_t` the caller
    /// handed it, a copy being `H5T_STATE_RDONLY` (H5T.c:4461-4462). This
    /// crate has no `H5Tcopy`: a datatype it can name at all is the predefined
    /// type itself, so the definition is the whole answer here.
    ///
    /// The one predefined type below the crate's horizon is
    /// `H5T_NATIVE_LDOUBLE`, the 80-bit extended float: nothing produces or
    /// reads one, since [`ieee_format`](Self::ieee_format) is what decides
    /// whether a float is a Rust float at all.
    pub fn is_predefined(&self) -> bool {
        /// The widths the `H5T_STD_*` families come in, at full precision with
        /// no offset — anything narrower is an `H5Tset_precision` away from a
        /// predefined type and therefore a mutable copy of one.
        fn whole_word(size: u32, bit_offset: u16, bit_precision: u16) -> bool {
            matches!(size, 1 | 2 | 4 | 8) && bit_offset == 0 && u32::from(bit_precision) == size * 8
        }
        match self {
            // H5T_STD_{I,U}{8,16,32,64}{BE,LE}, and every H5T_NATIVE integer,
            // which `H5T__init_native` builds by copying one of them.
            Self::FixedPoint {
                size,
                bit_offset,
                bit_precision,
                ..
            }
            // H5T_STD_B{8,16,32,64}{BE,LE} and H5T_NATIVE_B{8,16,32,64}.
            | Self::BitField {
                size,
                bit_offset,
                bit_precision,
                ..
            } => whole_word(*size, *bit_offset, *bit_precision),
            // H5T_IEEE_F{16,32,64}{BE,LE} and H5T_NATIVE_{FLOAT16,FLOAT,DOUBLE}.
            Self::FloatingPoint { .. } => self.ieee_format().is_some(),
            // H5T_C_S1, one ASCII byte padded `H5T_STR_NULLTERM`, and
            // H5T_FORTRAN_S1, the same padded `H5T_STR_SPACEPAD`
            // (H5T.c:370-387). Any other width is an `H5Tset_size` away.
            Self::FixedString {
                size: 1,
                padding,
                charset: 0,
            } => matches!(padding, 0 | 2),
            // H5T_STD_REF_OBJ, H5T_STD_REF_DSETREG and H5T_STD_REF.
            Self::Reference { .. } => true,
            _ => false,
        }
    }

    /// Whether this type changes between memory and disk —
    /// `H5T_is_relocatable` (H5T.c:7072-7087), which is true for a
    /// variable-length or reference type anywhere in the tree.
    ///
    /// The term that stops a predefined *reference* type from taking the
    /// immutable path: `H5D__init_type` copies a relocatable type whatever its
    /// state (H5Dint.c:572).
    pub fn is_relocatable(&self) -> bool {
        match self {
            Self::VarLenString { .. } | Self::VarLenSequence { .. } | Self::Reference { .. } => {
                true
            }
            Self::Enum { base, .. } | Self::Array { base, .. } => base.is_relocatable(),
            Self::Compound { members, .. } => members.iter().any(|m| m.datatype.is_relocatable()),
            Self::FixedPoint { .. }
            | Self::FloatingPoint { .. }
            | Self::BitField { .. }
            | Self::Opaque { .. }
            | Self::FixedString { .. } => false,
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
            Self::Reference { size, .. } => *size,
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
    /// The message version libhdf5 would stamp on this datatype in a file
    /// whose low libver bound is `libver`: the version the type is born with,
    /// raised by `H5T_set_version` to `H5O_dtype_ver_bounds[H5F_LOW_BOUND(f)]`
    /// when that is higher (H5T.c:6584-6591, table H5T.c:605-612).
    ///
    /// The bound reaches only the classes that gain from a newer encoding:
    /// `H5T__upgrade_version_cb` (H5T.c:6509-6546) raises compound, array and
    /// enum, gives a vlen its base's version, and leaves every atomic class
    /// alone — so an integer or a string stays at version 1 in every file.
    /// Each class then has a floor of its own: `H5T__alloc` starts everything
    /// at 1 (H5T.c:4030), `H5T__array_create` raises an array to 2
    /// (H5Tarray.c:169), `H5T__insert` raises a compound to its newest member
    /// (H5Tcompound.c:458-465), and `H5T_INIT_TYPE_REF_CORE` is born at 4
    /// (H5T.c:327). The two remaining floors libhdf5 carries — VAX floats at 3
    /// (H5T.c:205, :220) and complex numbers at 5 (H5Tcomplex.c:142, 2.0 only)
    /// — need classes this encoder cannot build, so they have nothing to raise
    /// here.
    pub fn message_version(&self, libver: LibverBound) -> u8 {
        let bound = libver.dtype_version();
        match self {
            Self::Compound { members, .. } => members
                .iter()
                .map(|m| m.datatype.message_version(libver))
                .fold(bound.max(DT_VERSION), u8::max),
            Self::Enum { base, .. } => bound.max(base.message_version(libver)),
            Self::Array { base, .. } => bound
                .max(ARRAY_MIN_VERSION)
                .max(base.message_version(libver)),
            Self::VarLenSequence { base } => base.message_version(libver),
            Self::Reference { kind, .. } => {
                if kind.is_revised() {
                    REVISED_REFERENCE_MESSAGE_VERSION
                } else {
                    DT_VERSION
                }
            }
            _ => DT_VERSION,
        }
    }

    /// Encode into a byte vector at the default libver bound
    /// ([`LibverBound::Earliest`], libhdf5's own default).
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        self.encode_at(ctx, LibverBound::Earliest)
    }

    /// Encode into a byte vector for a file whose low libver bound is
    /// `libver`, which decides the message version through
    /// [`Self::message_version`].
    pub fn encode_at(&self, ctx: &FormatContext, libver: LibverBound) -> Vec<u8> {
        let version = self.message_version(libver);
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
                buf.push(CLASS_FIXED_POINT | (version << 4));

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
                buf.push(CLASS_FLOATING_POINT | (version << 4));

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
                buf.push(CLASS_BITFIELD | (version << 4));
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
                buf.push(CLASS_OPAQUE | (version << 4));
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
                buf.push(CLASS_STRING | (version << 4));

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

                // From version 3 a member offset takes the fewest bytes that
                // can represent the compound's size (`H5VM_limit_enc_size` /
                // `UINT32ENCODE_VAR`); versions 1 and 2 spend a full four
                // (H5Odtype.c:1216-1221).
                let offset_nbytes = if version >= DT_VERSION_PACKED {
                    limit_enc_size(*size as u64)
                } else {
                    4
                };

                // Properties: for each member
                for member in members {
                    // Name, null-terminated. Versions 1 and 2 pad the field to
                    // a multiple of 8 bytes; version 3 dropped the padding
                    // (H5Odtype.c:1205-1214).
                    let name_start = buf.len();
                    buf.extend_from_slice(member.name.as_bytes());
                    buf.push(0);
                    if version < DT_VERSION_PACKED {
                        let padded = (buf.len() - name_start).div_ceil(8) * 8;
                        buf.resize(name_start + padded, 0);
                    }

                    // Byte offset, variable width.
                    buf.extend_from_slice(&member.offset.to_le_bytes()[..offset_nbytes]);

                    // A version-1 member then carries the intrinsic
                    // 'arrayness' the array class replaced, written as zeros.
                    if version == DT_VERSION {
                        buf.resize(buf.len() + V1_COMPOUND_MEMBER_ARRAY_BYTES, 0);
                    }

                    // Member datatype (recursive)
                    let dt_encoded = member.datatype.encode_at(ctx, libver);
                    buf.extend_from_slice(&dt_encoded);
                }

                buf
            }
            Self::Enum { base, members } => {
                let num_members = members.len() as u16;
                let base_size = base.element_size();

                let mut buf = vec![
                    // byte 0: class | version<<4
                    CLASS_ENUM | (version << 4),
                    // bytes 1-3: num_members as 16-bit LE
                    num_members as u8,
                    (num_members >> 8) as u8,
                    0,
                ];

                // bytes 4-7: element size (= base type size)
                buf.extend_from_slice(&base_size.to_le_bytes());

                // Properties: base datatype message
                let base_encoded = base.encode_at(ctx, libver);
                buf.extend_from_slice(&base_encoded);

                // Then each member name, null-terminated. Versions 1 and 2 pad
                // the field to a multiple of 8 bytes; version 3 dropped the
                // padding (H5Odtype.c:1279-1288), so the version the file's
                // bound picked decides the layout here.
                for member in members {
                    let name_start = buf.len();
                    buf.extend_from_slice(member.name.as_bytes());
                    buf.push(0);
                    if version < DT_VERSION_PACKED {
                        let padded = (buf.len() - name_start).div_ceil(8) * 8;
                        buf.resize(name_start + padded, 0);
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
                    CLASS_VLEN | (version << 4),
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
                let base_encoded = base_type.encode_at(ctx, libver);
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
                    CLASS_VLEN | (version << 4),
                    // byte 1 bits 0-3: type = 0 (sequence)
                    0x00,
                    // bytes 2-3: reserved (no charset/pad for sequences)
                    0,
                    0,
                ];

                // bytes 4-7: element size
                buf.extend_from_slice(&vlen_size.to_le_bytes());

                // Properties: base (parent) datatype message, recursive.
                let base_encoded = base.encode_at(ctx, libver);
                buf.extend_from_slice(&base_encoded);

                buf
            }
            Self::Array { dims, base } => {
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

                // Versions below 3 follow it with three reserved bytes
                // (H5Odtype.c:1326-1332).
                if version < DT_VERSION_PACKED {
                    buf.extend_from_slice(&[0, 0, 0]);
                }

                // dims: ndims * u32 LE
                for &d in dims {
                    buf.extend_from_slice(&d.to_le_bytes());
                }

                // ...and with the 'fake' dimension permutations version 3
                // dropped, which libhdf5 writes as 0..ndims
                // (H5Odtype.c:1338-1343).
                if version < DT_VERSION_PACKED {
                    for i in 0..dims.len() as u32 {
                        buf.extend_from_slice(&i.to_le_bytes());
                    }
                }

                // base datatype message (recursive)
                let base_encoded = base.encode_at(ctx, libver);
                buf.extend_from_slice(&base_encoded);

                buf
            }
            Self::Reference { size, kind } => {
                // Class 7, no properties: the whole message is the 8-byte
                // header, with the reference type in the bit field's low
                // nibble and — for the 1.12 kinds — the encoding version in
                // the next one (`H5O__dtype_encode_helper`). The message
                // version is 1 for the old kinds and 4 for the 1.12 ones,
                // which `message_version` already settled.
                let mut flags0 = kind.code();
                if kind.is_revised() {
                    flags0 |= REFERENCE_ENCODE_VERSION << 4;
                }
                let mut buf = vec![CLASS_REFERENCE | (version << 4), flags0, 0, 0];
                buf.extend_from_slice(&size.to_le_bytes());
                buf
            }
        }
    }

    /// Decode from a byte buffer.  Returns `(message, bytes_consumed)`.
    ///
    /// `_ctx` is accepted for signature symmetry with the other message
    /// types' `decode`, several of which do need it; a datatype message is
    /// fully self-describing on disk, so decoding never reads it.
    pub fn decode(buf: &[u8], _ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        Self::decode_inner(buf, 0, &mut Vec::new())
    }

    /// The class and version of every datatype message in the tree at `buf`,
    /// outermost first and then depth-first in encode order — the same walk
    /// `H5O__dtype_debug` prints (H5Odtype.c:1984-2027).
    ///
    /// The version is the one thing a decode drops: [`Self::decode`] returns
    /// the type, not the encoding it arrived in, so this is what answers
    /// "which version does the message in this file claim".
    pub fn decode_versions(buf: &[u8]) -> FormatResult<Vec<DatatypeNodeVersion>> {
        let mut versions = Vec::new();
        Self::decode_inner(buf, 0, &mut versions)?;
        Ok(versions)
    }

    /// Recursive worker for [`decode`]. `depth` bounds datatype nesting:
    /// compound/enum/vlen/array types embed a base datatype recursively, and
    /// a crafted message can nest these deeply enough to exhaust the stack.
    /// libhdf5-written types nest only a handful of levels.
    fn decode_inner(
        buf: &[u8],
        depth: usize,
        versions: &mut Vec<DatatypeNodeVersion>,
    ) -> FormatResult<(Self, usize)> {
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
        // 1..=H5O_DTYPE_VERSION_LATEST is a bad message. Every version shares
        // the property layouts decoded below — the version only selects the
        // name padding, the compound offset width and the array reserved
        // fields. Version 5 is the HDF5 2.0 encoding, and all it added was
        // the complex-number class, which the class dispatch below refuses on
        // its own; a v5 compound or array is byte-identical to a v4 one.
        if !(1..=DT_VERSION_LATEST).contains(&version) {
            return Err(FormatError::InvalidVersion(version));
        }

        versions.push(DatatypeNodeVersion {
            depth,
            class: class_name(class),
            version,
        });

        match class {
            CLASS_FIXED_POINT => {
                if buf.len() < 12 {
                    return Err(FormatError::BufferTooShort {
                        needed: 12,
                        available: buf.len(),
                    });
                }
                let byte_order = atomic_byte_order(flags0, false)?;
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
                // Only a v3-or-later float gives bit 6 the VAX meaning.
                let byte_order = atomic_byte_order(flags0, version >= 3)?;
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
                let byte_order = atomic_byte_order(flags0, false)?;
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
                    let offset_nbytes = if version >= DT_VERSION_PACKED {
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

                    if version == DT_VERSION {
                        // Version 1 also carries the intrinsic 'arrayness' of
                        // the member, which this encoder writes as zeros.
                        if pos + V1_COMPOUND_MEMBER_ARRAY_BYTES > buf.len() {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + V1_COMPOUND_MEMBER_ARRAY_BYTES,
                                available: buf.len(),
                            });
                        }
                        pos += V1_COMPOUND_MEMBER_ARRAY_BYTES;
                    }

                    // Member datatype (recursive)
                    let (member_dt, dt_consumed) =
                        Self::decode_inner(&buf[pos..], depth + 1, versions)?;
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
                let (base_dt, base_consumed) =
                    Self::decode_inner(&buf[pos..], depth + 1, versions)?;
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
                let (base_dt, base_consumed) =
                    Self::decode_inner(&buf[pos..], depth + 1, versions)?;
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
                if version < ARRAY_MIN_VERSION {
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
                if version < DT_VERSION_PACKED {
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
                if version < DT_VERSION_PACKED {
                    pos += ndims * 4;
                }

                // Base datatype
                let (base_dt, base_consumed) =
                    Self::decode_inner(&buf[pos..], depth + 1, versions)?;
                pos += base_consumed;

                Ok((
                    Self::Array {
                        dims,
                        base: Box::new(base_dt),
                    },
                    pos,
                ))
            }
            CLASS_REFERENCE => {
                // No properties: `H5O__dtype_decode_helper` reads the class
                // bit field and nothing else. The low nibble is the
                // `H5R_type_t`; anything at or above `H5R_MAXTYPE` is an
                // invalid reference type.
                let kind = ReferenceKind::from_code(flags0 & 0x0F).ok_or_else(|| {
                    FormatError::InvalidData(format!("invalid reference type {}", flags0 & 0x0F))
                })?;
                // The 1.12 kinds carry their encoding version in the next
                // nibble, and libhdf5 fails a message whose version it does
                // not know rather than guess the token layout.
                if kind.is_revised() {
                    let encode_version = (flags0 >> 4) & 0x0F;
                    if encode_version != REFERENCE_ENCODE_VERSION {
                        return Err(FormatError::InvalidData(format!(
                            "reference version {encode_version} does not match"
                        )));
                    }
                }
                Ok((Self::Reference { size, kind }, 8))
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
            Self::Reference { size, kind } => write!(f, "reference[{size}; {kind:?}]"),
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
        // class 2 (H5T_TIME), version 1 — a class libhdf5 defines and this
        // crate does not decode.
        let mut buf = [0u8; 12];
        buf[0] = 2 | (1 << 4);
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

    /// Every version shares the property layout of every class this crate
    /// decodes, so a v4- or v5-tagged message — what libhdf5 emits under a
    /// `H5F_LIBVER_V112` or `H5F_LIBVER_V200` low bound — must decode, not be
    /// rejected.
    #[test]
    fn decode_accepts_versions_4_and_5() {
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
            for version in [4u8, 5] {
                let tagged = with_version(msg.encode(&ctx()), version);
                let (decoded, _) = DatatypeMessage::decode(&tagged, &ctx())
                    .unwrap_or_else(|e| panic!("v{version} {msg} rejected: {e:?}"));
                assert_eq!(decoded, msg, "v{version} {msg} decoded differently");
            }
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
        let v4 = msg.encode_at(&ctx(), LibverBound::V112);
        assert_eq!(v4[0] >> 4, 4);
        let (decoded, consumed) = DatatypeMessage::decode(&v4, &ctx()).unwrap();
        assert_eq!(consumed, v4.len());
        assert_eq!(decoded, msg);
    }

    /// All version 5 — the HDF5 2.0 encoding — added is the complex-number
    /// class, so what a v5 message can carry that a v4 one cannot is refused
    /// by class, naming class 11 rather than the version.
    #[test]
    fn decode_reports_the_complex_class_as_unsupported() {
        // A v5 rectangular complex over an 8-byte float: class 11
        // (`H5T_COMPLEX`), the homogeneous bit set, base type in the
        // properties.
        let mut msg = vec![11 | (5 << 4), 0x01, 0, 0];
        msg.extend_from_slice(&16u32.to_le_bytes());
        msg.extend_from_slice(&DatatypeMessage::f64_type().encode(&ctx()));
        let err = DatatypeMessage::decode(&msg, &ctx()).unwrap_err();
        assert!(
            matches!(&err, FormatError::UnsupportedFeature(m)
                     if m == "datatype class 11"),
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

    // ---- byte order ----

    /// The message libhdf5 writes for `H5T_VAX_F8`: version 3, class 1, both
    /// bit 0 and bit 6 set, and the VAX exponent bias 0x401 (`H5T.c`,
    /// `H5T_INIT_TYPE_DOUBLEVAX_CORE`). `H5Tdecode` in 1.14.6 reads these 20
    /// bytes back as `H5T_ORDER_VAX`, size 8, ebias 1025, fields
    /// (63, 52, 11, 0, 52), and re-encodes them byte for byte.
    const VAX_F8_MESSAGE: [u8; 20] = [
        0x31, 0x61, 0x3F, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x34, 0x0B, 0x00,
        0x34, 0x01, 0x04, 0x00, 0x00,
    ];

    /// VAX is a middle-endian layout with its own bias, not a permutation of
    /// the two orders this crate stores, so it is named and refused instead of
    /// decoded as the big-endian its bit 0 alone would say.
    #[test]
    fn decode_rejects_vax_byte_order() {
        let err = DatatypeMessage::decode(&VAX_F8_MESSAGE, &ctx()).unwrap_err();
        assert!(
            matches!(&err, FormatError::UnsupportedFeature(m) if m.contains("VAX")),
            "unexpected error: {err:?}"
        );
    }

    /// Bit 6 is the VAX flag only for a floating-point message of version 3 or
    /// later — `H5O__dtype_decode_helper` reads it nowhere else. A v1 float
    /// carrying the same bits is big-endian, which is exactly what libhdf5
    /// makes of the file h5py writes from `NATIVE_DOUBLE.set_order(ORDER_VAX)`:
    /// h5dump 1.14.6 prints `H5T_IEEE_F64BE` for it.
    #[test]
    fn bit_6_is_vax_only_for_a_version_3_float() {
        let mut v1 = VAX_F8_MESSAGE;
        v1[0] = 0x11;
        let (decoded, _) = DatatypeMessage::decode(&v1, &ctx()).unwrap();
        assert_eq!(decoded.scalar_byte_order(), Some(ByteOrder::BigEndian));

        // An integer and a bit field give bit 6 no meaning at any version.
        for msg in [
            DatatypeMessage::i32_type(),
            DatatypeMessage::BitField {
                size: 1,
                byte_order: ByteOrder::BigEndian,
                bit_offset: 0,
                bit_precision: 8,
            },
        ] {
            let mut bytes = msg.encode(&ctx());
            bytes[1] |= 0x41;
            let (decoded, _) = DatatypeMessage::decode(&bytes, &ctx()).unwrap();
            assert_eq!(decoded.scalar_byte_order(), Some(ByteOrder::BigEndian));
        }
    }

    /// Bit 6 without bit 0 is not a byte order at all; libhdf5 fails the
    /// message with "bad byte order for datatype message" rather than pick one.
    #[test]
    fn decode_rejects_the_vax_bit_without_the_big_endian_bit() {
        let mut msg = VAX_F8_MESSAGE;
        msg[1] &= !0x01;
        let err = DatatypeMessage::decode(&msg, &ctx()).unwrap_err();
        assert!(
            matches!(&err, FormatError::InvalidData(m) if m.contains("bad byte order")),
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
        assert_eq!(encoded[0] >> 4, 1); // version = 1 at the earliest bound
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
        assert_eq!(encoded[0] >> 4, 2); // version = 2, the array floor
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

    // ---- libver bounds pick the message version ----

    fn xy_compound() -> DatatypeMessage {
        DatatypeMessage::compound(
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
        )
    }

    /// The version each class carries at each low libver bound.
    ///
    /// Measured from libhdf5 1.14.6 (`h5debug` on files h5py wrote with
    /// `libver=` earliest / v108 / v112 / latest):
    ///
    /// ```text
    ///             compound  enum  array  int  vlen(i32)
    ///  earliest      1        1     2      1      1
    ///  v108          3        3     3      1      1
    ///  v112          4        4     4      1      1
    ///  latest        4        4     4      1      1
    /// ```
    ///
    /// Every cell is what this encoder writes, the two that matter most
    /// included: an integer never moves off version 1, and a vlen inherits its
    /// parent instead of the bound.
    #[test]
    fn message_version_follows_the_libver_bound() {
        let cases: [(LibverBound, [u8; 5]); 6] = [
            (LibverBound::Earliest, [1, 1, 2, 1, 1]),
            (LibverBound::V18, [3, 3, 3, 1, 1]),
            (LibverBound::V110, [3, 3, 3, 1, 1]),
            (LibverBound::V112, [4, 4, 4, 1, 1]),
            (LibverBound::V114, [4, 4, 4, 1, 1]),
            (LibverBound::V200, [5, 5, 5, 1, 1]),
        ];
        for (bound, expected) in cases {
            let got = [
                xy_compound().message_version(bound),
                DatatypeMessage::bool_type().message_version(bound),
                DatatypeMessage::array(vec![3], DatatypeMessage::i32_type()).message_version(bound),
                DatatypeMessage::i32_type().message_version(bound),
                DatatypeMessage::VarLenSequence {
                    base: Box::new(DatatypeMessage::i32_type()),
                }
                .message_version(bound),
            ];
            assert_eq!(got, expected, "{bound:?}");
        }
    }

    /// A compound is at least as new as its newest member (H5Tcompound.c),
    /// and an array at least as new as its base — even at the earliest bound,
    /// where a revised reference drags the whole message to version 4.
    #[test]
    fn a_composite_is_at_least_as_new_as_what_it_holds() {
        let stdref = DatatypeMessage::Reference {
            size: 8,
            kind: ReferenceKind::Object2,
        };
        let compound = DatatypeMessage::compound(
            12,
            vec![
                CompoundMember {
                    name: "id".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::i32_type(),
                },
                CompoundMember {
                    name: "to".to_string(),
                    offset: 4,
                    datatype: stdref.clone(),
                },
            ],
        );
        assert_eq!(compound.message_version(LibverBound::Earliest), 4);
        let array = DatatypeMessage::array(vec![2], stdref.clone());
        assert_eq!(array.message_version(LibverBound::Earliest), 4);
        // A vlen takes its parent's version rather than the bound's, so it
        // reaches 4 through the reference and stays at 1 over an integer.
        let vlen = DatatypeMessage::VarLenSequence {
            base: Box::new(stdref),
        };
        assert_eq!(vlen.message_version(LibverBound::V112), 4);
    }

    /// `H5T_STD_REF` is born at version 4 and `H5T__upgrade_version_cb`
    /// leaves references alone, so the 1.12 kinds sit at 4 under every bound
    /// while the old kinds sit at 1.
    #[test]
    fn reference_versions_ignore_the_bound() {
        for bound in [LibverBound::Earliest, LibverBound::V112, LibverBound::V200] {
            for kind in [ReferenceKind::Object1, ReferenceKind::DatasetRegion1] {
                let msg = DatatypeMessage::Reference { size: 8, kind };
                assert_eq!(msg.message_version(bound), 1, "{kind:?} at {bound:?}");
                assert_eq!(msg.encode_at(&ctx(), bound)[0] >> 4, 1);
            }
            for kind in [
                ReferenceKind::Object2,
                ReferenceKind::DatasetRegion2,
                ReferenceKind::Attr,
            ] {
                let msg = DatatypeMessage::Reference { size: 64, kind };
                assert_eq!(msg.message_version(bound), 4, "{kind:?} at {bound:?}");
                assert_eq!(msg.encode_at(&ctx(), bound)[0] >> 4, 4);
            }
        }
    }

    /// Version 3 dropped the padding that rounds an enum member name field up
    /// to a multiple of 8 bytes (`H5O__dtype_encode_helper`), so the bound
    /// that picks the version picks the layout with it.
    #[test]
    fn enum_name_padding_follows_the_message_version() {
        let msg = DatatypeMessage::bool_type();

        let v1 = msg.encode_at(&ctx(), LibverBound::Earliest);
        assert_eq!(v1[0] >> 4, 1);
        // 8 header + 12 base + "FALSE\0" padded to 8 + "TRUE\0" padded to 8
        // + two 1-byte values.
        assert_eq!(v1.len(), 8 + 12 + 8 + 8 + 2);
        assert_eq!(&v1[20..28], b"FALSE\0\0\0");
        assert_eq!(&v1[28..36], b"TRUE\0\0\0\0");

        let v4 = msg.encode_at(&ctx(), LibverBound::V112);
        assert_eq!(v4[0] >> 4, 4);
        assert_eq!(v4.len(), 8 + 12 + 6 + 5 + 2);
        assert_eq!(&v4[20..26], b"FALSE\0");
        assert_eq!(&v4[26..31], b"TRUE\0");

        // Both layouts survive a round trip, which is what the reader's
        // version-dependent padding rule is for.
        for encoded in [v1, v4] {
            let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
            assert_eq!(consumed, encoded.len());
            assert_eq!(decoded, msg);
        }
    }

    /// The earliest bound is where the layouts differ: a compound is version 1
    /// and spends 8-byte-padded names, a 4-byte offset and the 28 zero bytes
    /// of the intrinsic 'arrayness' on every member (H5Odtype.c:1205-1247).
    #[test]
    fn a_compound_at_the_earliest_bound_is_a_version_1_message() {
        let msg = xy_compound();
        let v1 = msg.encode_at(&ctx(), LibverBound::Earliest);

        assert_eq!(v1[0], CLASS_COMPOUND | (1 << 4));
        // 8 header + two members of 8 name + 4 offset + 28 arrayness + 20 f32.
        assert_eq!(
            v1.len(),
            8 + 2 * (8 + 4 + V1_COMPOUND_MEMBER_ARRAY_BYTES + 20)
        );
        assert_eq!(&v1[8..16], b"x\0\0\0\0\0\0\0");
        assert_eq!(&v1[16..20], &0u32.to_le_bytes());
        assert_eq!(&v1[20..48], &[0u8; V1_COMPOUND_MEMBER_ARRAY_BYTES]);
        assert_eq!(&v1[68..76], b"y\0\0\0\0\0\0\0");
        assert_eq!(&v1[76..80], &4u32.to_le_bytes());

        let (decoded, consumed) = DatatypeMessage::decode(&v1, &ctx()).unwrap();
        assert_eq!(consumed, v1.len());
        assert_eq!(decoded, msg);
    }

    /// An array is born at version 2 (H5Tarray.c:169) and never drops to 1, so
    /// the earliest bound still gets the three reserved bytes and the
    /// `0..ndims` dimension permutations version 3 dropped
    /// (H5Odtype.c:1326-1343).
    #[test]
    fn an_array_at_the_earliest_bound_is_a_version_2_message() {
        let msg = DatatypeMessage::array(vec![2, 3], DatatypeMessage::i32_type());
        let v2 = msg.encode_at(&ctx(), LibverBound::Earliest);

        assert_eq!(v2[0], CLASS_ARRAY | (ARRAY_MIN_VERSION << 4));
        // 8 header + 1 ndims + 3 reserved + 2 dims + 2 permutations + 12 i32.
        assert_eq!(v2.len(), 8 + 1 + 3 + 8 + 8 + 12);
        assert_eq!(&v2[8..12], &[2, 0, 0, 0]);
        assert_eq!(
            &v2[12..20],
            [2u32.to_le_bytes(), 3u32.to_le_bytes()].concat()
        );
        assert_eq!(
            &v2[20..28],
            [0u32.to_le_bytes(), 1u32.to_le_bytes()].concat()
        );

        let (decoded, consumed) = DatatypeMessage::decode(&v2, &ctx()).unwrap();
        assert_eq!(consumed, v2.len());
        assert_eq!(decoded, msg);
    }

    /// The array floor reaches the compound that holds it: `H5T__insert` lifts
    /// the parent to its newest member (H5Tcompound.c:458-465), which puts the
    /// compound on version 2 — padded names and a 4-byte offset, but none of
    /// the version-1 arrayness bytes.
    #[test]
    fn an_array_member_lifts_its_compound_to_version_2() {
        let msg = DatatypeMessage::compound(
            12,
            vec![
                CompoundMember {
                    name: "id".to_string(),
                    offset: 0,
                    datatype: DatatypeMessage::i32_type(),
                },
                CompoundMember {
                    name: "pts".to_string(),
                    offset: 4,
                    datatype: DatatypeMessage::array(vec![2], DatatypeMessage::i32_type()),
                },
            ],
        );
        assert_eq!(msg.message_version(LibverBound::Earliest), 2);

        let v2 = msg.encode_at(&ctx(), LibverBound::Earliest);
        assert_eq!(v2[0], CLASS_COMPOUND | (2 << 4));
        // 8 header + (8 name + 4 offset + 12 i32) + (8 name + 4 offset + 32 array).
        assert_eq!(v2.len(), 8 + (8 + 4 + 12) + (8 + 4 + 32));
        assert_eq!(&v2[8..16], b"id\0\0\0\0\0\0");
        assert_eq!(&v2[32..40], b"pts\0\0\0\0\0");
        assert_eq!(&v2[40..44], &4u32.to_le_bytes());

        let (decoded, consumed) = DatatypeMessage::decode(&v2, &ctx()).unwrap();
        assert_eq!(consumed, v2.len());
        assert_eq!(decoded, msg);
    }

    /// Writing at the natural version, reading back and rewriting unchanged is
    /// the round trip the found-format rule depends on: a reopened object must
    /// re-emit the version its file already carries.
    #[test]
    fn every_bound_round_trips_and_rewrites_to_the_same_bytes() {
        for bound in [
            LibverBound::Earliest,
            LibverBound::V18,
            LibverBound::V110,
            LibverBound::V112,
            LibverBound::V114,
            LibverBound::V200,
        ] {
            for msg in [
                xy_compound(),
                DatatypeMessage::bool_type(),
                DatatypeMessage::array(vec![2, 3], DatatypeMessage::i32_type()),
                DatatypeMessage::array(vec![4], xy_compound()),
                DatatypeMessage::i32_type(),
                DatatypeMessage::f64_type(),
                DatatypeMessage::fixed_string(8),
                DatatypeMessage::vlen_string_utf8(),
                DatatypeMessage::vlen_bytes(),
                DatatypeMessage::VarLenSequence {
                    base: Box::new(xy_compound()),
                },
            ] {
                let encoded = msg.encode_at(&ctx(), bound);
                assert_eq!(
                    encoded[0] >> 4,
                    msg.message_version(bound),
                    "{msg} at {bound:?}"
                );
                let (decoded, consumed) = DatatypeMessage::decode(&encoded, &ctx()).unwrap();
                assert_eq!(consumed, encoded.len(), "{msg} at {bound:?}");
                assert_eq!(decoded, msg, "{msg} at {bound:?}");
                assert_eq!(
                    decoded.encode_at(&ctx(), bound),
                    encoded,
                    "{msg} at {bound:?}"
                );
            }
        }
    }

    /// Version 3 is the last one that touched the compound and array member
    /// layout (H5Tpkg.h:85-91), so from `V18` up only the version nibble moves.
    #[test]
    fn a_higher_bound_moves_only_the_version_nibble() {
        for msg in [
            xy_compound(),
            DatatypeMessage::array(vec![2, 3], DatatypeMessage::i32_type()),
        ] {
            let base = msg.encode_at(&ctx(), LibverBound::V18);
            for (bound, version) in [(LibverBound::V112, 4u8), (LibverBound::V200, 5)] {
                let raised = msg.encode_at(&ctx(), bound);
                assert_eq!(raised[0] >> 4, version);
                assert_eq!(raised[1..], base[1..], "{msg} at {bound:?}");
                let (decoded, consumed) = DatatypeMessage::decode(&raised, &ctx()).unwrap();
                assert_eq!(consumed, raised.len());
                assert_eq!(decoded, msg);
            }
        }
    }

    /// The default bound is libhdf5's own default, and `encode` is exactly
    /// `encode_at` at that bound, version nibble included.
    #[test]
    fn the_default_bound_is_the_earliest_one() {
        assert_eq!(LibverBound::default(), LibverBound::Earliest);
        for msg in [
            xy_compound(),
            DatatypeMessage::bool_type(),
            DatatypeMessage::array(vec![4], DatatypeMessage::f64_type()),
            DatatypeMessage::i32_type(),
            DatatypeMessage::vlen_string_utf8(),
            DatatypeMessage::vlen_bytes(),
            DatatypeMessage::fixed_string(8),
        ] {
            assert_eq!(
                msg.encode(&ctx()),
                msg.encode_at(&ctx(), Default::default())
            );
            assert_eq!(
                msg.encode(&ctx())[0] >> 4,
                msg.message_version(LibverBound::Earliest)
            );
        }
    }
}
