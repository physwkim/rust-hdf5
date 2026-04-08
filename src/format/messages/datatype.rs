//! Datatype message (type 0x03) — describes element data type.
//!
//! Binary layout:
//!   Byte 0:    (class & 0x0F) | (version << 4)     version = 1
//!   Bytes 1-3: class bit-field flags (24 bits, little-endian)
//!   Bytes 4-7: element size (u32 LE)
//!   Bytes 8+:  class-specific properties

use crate::format::{FormatContext, FormatError, FormatResult};

const DT_VERSION: u8 = 1;

// Datatype class codes
const CLASS_FIXED_POINT: u8 = 0;
const CLASS_FLOATING_POINT: u8 = 1;
const CLASS_STRING: u8 = 3;
const CLASS_COMPOUND: u8 = 6;
const CLASS_ENUM: u8 = 8;
const CLASS_VLEN: u8 = 9;
const CLASS_ARRAY: u8 = 10;

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
    /// Variable-length datatype (class 9) -- currently only vlen strings.
    VarLenString {
        /// Character set: 0 = ASCII, 1 = UTF-8.
        charset: u8,
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

    /// Variable-length UTF-8 string type.
    ///
    /// Note: `element_size()` for this type requires a `FormatContext` to
    /// compute. Use `element_size_ctx()` or `vlen_ref_size()` instead.
    pub fn vlen_string_utf8() -> Self {
        Self::VarLenString { charset: 1 }
    }

    /// Variable-length ASCII string type.
    pub fn vlen_string_ascii() -> Self {
        Self::VarLenString { charset: 0 }
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

// ========================================================================= queries

impl DatatypeMessage {
    /// Returns the element size in bytes.
    ///
    /// For `VarLenString`, this returns the on-disk reference size assuming
    /// 8-byte addresses (sizeof_addr=8): 8 + 4 = 12.
    /// Use `element_size_ctx()` for an exact answer with a specific context.
    pub fn element_size(&self) -> u32 {
        match self {
            Self::FixedPoint { size, .. } => *size,
            Self::FloatingPoint { size, .. } => *size,
            Self::FixedString { size, .. } => *size,
            Self::Compound { size, .. } => *size,
            Self::Enum { base, .. } => base.element_size(),
            Self::VarLenString { .. } => {
                // Default assumption: sizeof_addr = 8
                // vlen ref = sizeof_addr + 4 = 12
                12
            }
            Self::Array { dims, base } => {
                let product: u32 = dims.iter().product();
                product * base.element_size()
            }
        }
    }

    /// Returns the element size using an explicit format context.
    ///
    /// This is needed for `VarLenString` where the size depends on
    /// `sizeof_addr`.
    pub fn element_size_ctx(&self, ctx: &FormatContext) -> u32 {
        match self {
            Self::VarLenString { .. } => ctx.sizeof_addr as u32 + 4,
            _ => self.element_size(),
        }
    }

    /// Returns the size of a variable-length reference for a given context.
    pub fn vlen_ref_size(ctx: &FormatContext) -> u32 {
        ctx.sizeof_addr as u32 + 4
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

                // Properties: for each member
                for member in members {
                    // Name (null-terminated)
                    buf.extend_from_slice(member.name.as_bytes());
                    buf.push(0);

                    // Byte offset (u32 LE for version 3)
                    buf.extend_from_slice(&member.offset.to_le_bytes());

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
            Self::VarLenString { charset } => {
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
                    //         bits 4-7: padding type (0 = null pad)
                    0x01, // type = string (1)
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
            Self::Array { dims, base } => {
                // Array: class 10, version 3
                let version: u8 = 3;
                let base_size = base.element_size();
                let product: u32 = dims.iter().product();
                let total_size = product * base_size;

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
    #[allow(clippy::only_used_in_recursion)]
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
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

        match class {
            CLASS_FIXED_POINT => {
                if version != DT_VERSION {
                    return Err(FormatError::InvalidVersion(version));
                }
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
                if version != DT_VERSION {
                    return Err(FormatError::InvalidVersion(version));
                }
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
            CLASS_STRING => {
                // String class: 8-byte header, no additional properties
                // Accept version 1 or 3 (HDF5 uses both)
                if version != DT_VERSION && version != 3 {
                    return Err(FormatError::InvalidVersion(version));
                }
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
                // Compound: version 1 or 3
                if version != 1 && version != 3 {
                    return Err(FormatError::UnsupportedFeature(format!(
                        "compound datatype version {}",
                        version
                    )));
                }

                // num_members from flags bytes 1-2 (16-bit LE)
                let num_members = u16::from_le_bytes([flags0, flags1]) as usize;

                let mut pos = 8; // past the 8-byte header

                let mut members = Vec::with_capacity(num_members);
                for _ in 0..num_members {
                    // Name: null-terminated string
                    let name_start = pos;
                    while pos < buf.len() && buf[pos] != 0 {
                        pos += 1;
                    }
                    if pos >= buf.len() {
                        return Err(FormatError::InvalidData(
                            "unterminated compound member name".into(),
                        ));
                    }
                    let name = String::from_utf8_lossy(&buf[name_start..pos]).to_string();
                    pos += 1; // skip null terminator

                    // Version 1: names are padded to 8-byte boundary
                    // (from the start of the name, including the null)
                    if version == 1 {
                        let name_field_len = pos - name_start;
                        let padded = (name_field_len + 7) & !7;
                        pos = name_start + padded;
                    }

                    // Byte offset
                    let offset = if version == 1 {
                        // Version 1: 4-byte offset
                        if pos + 4 > buf.len() {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + 4,
                                available: buf.len(),
                            });
                        }
                        let o = u32::from_le_bytes([
                            buf[pos],
                            buf[pos + 1],
                            buf[pos + 2],
                            buf[pos + 3],
                        ]);
                        pos += 4;

                        // Version 1 has additional fields: dimensionality(1),
                        // reserved(3), dim_perm(4), reserved(4), dim_sizes(4*4)
                        // = 1 + 3 + 4 + 4 + 16 = 28 bytes
                        if pos + 28 > buf.len() {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + 28,
                                available: buf.len(),
                            });
                        }
                        pos += 28; // skip v1 dimension info

                        o
                    } else {
                        // Version 3: 4-byte offset, no padding or dimension info
                        if pos + 4 > buf.len() {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + 4,
                                available: buf.len(),
                            });
                        }
                        let o = u32::from_le_bytes([
                            buf[pos],
                            buf[pos + 1],
                            buf[pos + 2],
                            buf[pos + 3],
                        ]);
                        pos += 4;
                        o
                    };

                    // Member datatype (recursive)
                    let (member_dt, dt_consumed) = Self::decode(&buf[pos..], ctx)?;
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
                // Enum: version 1
                if version != DT_VERSION {
                    return Err(FormatError::InvalidVersion(version));
                }
                let num_members = u16::from_le_bytes([flags0, flags1]) as usize;
                let base_size = size;

                let mut pos = 8;

                // Base datatype
                let (base_dt, base_consumed) = Self::decode(&buf[pos..], ctx)?;
                pos += base_consumed;

                // Member names (null-terminated, padded to 8-byte boundary for v1)
                let mut names = Vec::with_capacity(num_members);
                for _ in 0..num_members {
                    let name_start = pos;
                    while pos < buf.len() && buf[pos] != 0 {
                        pos += 1;
                    }
                    if pos >= buf.len() {
                        return Err(FormatError::InvalidData(
                            "unterminated enum member name".into(),
                        ));
                    }
                    let name = String::from_utf8_lossy(&buf[name_start..pos]).to_string();
                    pos += 1; // skip null
                              // Version 1: pad name (including null) to 8-byte boundary
                    let name_field_len = pos - name_start;
                    let padded = (name_field_len + 7) & !7;
                    pos = name_start + padded;
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
                // Variable-length: version 1
                if version != DT_VERSION {
                    return Err(FormatError::InvalidVersion(version));
                }
                let vlen_type = flags0 & 0x0F;
                let charset = flags1 & 0x0F;

                let mut pos = 8;

                // Properties: base datatype
                let (_base_dt, base_consumed) = Self::decode(&buf[pos..], ctx)?;
                pos += base_consumed;

                if vlen_type == 1 {
                    // String type
                    Ok((Self::VarLenString { charset }, pos))
                } else {
                    // Sequence type -- treat as unsupported for now
                    Err(FormatError::UnsupportedFeature(format!(
                        "vlen sequence type {}",
                        vlen_type
                    )))
                }
            }
            CLASS_ARRAY => {
                // Array: version 3
                if version != 3 && version != 2 {
                    return Err(FormatError::UnsupportedFeature(format!(
                        "array datatype version {}",
                        version
                    )));
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

                // Version 2 has 3 bytes of padding after ndims
                if version == 2 {
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

                // Version 2 has permutation indices after dims (ndims * u32), skip them
                if version == 2 {
                    pos += ndims * 4;
                }

                // Base datatype
                let (base_dt, base_consumed) = Self::decode(&buf[pos..], ctx)?;
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
            Self::VarLenString { charset } => {
                let cs = if *charset == 1 { "UTF-8" } else { "ASCII" };
                write!(f, "vlen_string({})", cs)
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
        // class 5, version 1
        let mut buf = [0u8; 12];
        buf[0] = 5 | (1 << 4);
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
        // Default: sizeof_addr=8, so 8+4 = 12
        assert_eq!(msg.element_size(), 12);
        assert_eq!(msg.element_size_ctx(&ctx()), 12);
        assert_eq!(msg.element_size_ctx(&ctx4()), 8);
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
        // Size field in the encoded bytes should be 4+4=8
        let sz = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
        assert_eq!(sz, 8);
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
