//! Attribute message (type 0x0C) -- describes an attribute attached to an object.
//!
//! Binary layout (version 3, no shared datatypes):
//!   Byte 0:    version = 3
//!   Byte 1:    flags (0 for non-shared)
//!   Bytes 2-3: name_size (u16 LE, including null terminator)
//!   Bytes 4-5: datatype_size (u16 LE)
//!   Bytes 6-7: dataspace_size (u16 LE)
//!   Byte 8:    name character set encoding (0=ASCII, 1=UTF-8)
//!   <name: name_size bytes, null-terminated>
//!   <encoded datatype message: datatype_size bytes>
//!   <encoded dataspace message: dataspace_size bytes>
//!   <raw attribute data>

use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::{FormatContext, FormatError, FormatResult, LibverBound};

const ATTR_VERSION: u8 = 3;

/// `H5O_ATTR_FLAG_TYPE_SHARED` (H5Oattr.c:88): the datatype field holds a
/// shared-message pointer rather than the datatype message.
pub const ATTR_FLAG_TYPE_SHARED: u8 = 0x01;
/// `H5O_ATTR_FLAG_SPACE_SHARED` (H5Oattr.c:89), the same for the dataspace.
pub const ATTR_FLAG_SPACE_SHARED: u8 = 0x02;

/// One attribute message body, and where its datatype and dataspace fields
/// sit inside it.
///
/// The offsets are what a caller that put a shared-message pointer in either
/// field needs in order to fill the pointer's heap ID in later, once the heap
/// it points into has been laid out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedAttribute {
    /// The message payload.
    pub body: Vec<u8>,
    /// Offset of the datatype field in `body`.
    pub datatype_at: usize,
    /// Offset of the dataspace field in `body`.
    pub dataspace_at: usize,
}

/// An HDF5 attribute message.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeMessage {
    /// Attribute name.
    pub name: String,
    /// Datatype of the attribute value.
    pub datatype: DatatypeMessage,
    /// Dataspace (scalar or simple).
    pub dataspace: DataspaceMessage,
    /// Raw attribute value data.
    pub data: Vec<u8>,
}

impl AttributeMessage {
    /// Create a scalar string attribute with the given name and value.
    ///
    /// Uses a null-terminated UTF-8 fixed-length string datatype with
    /// size = value.len() + 1 (for the null terminator), and a scalar
    /// dataspace.
    pub fn scalar_string(name: &str, value: &str) -> Self {
        let str_size = (value.len() + 1) as u32; // +1 for null terminator
        let datatype = DatatypeMessage::fixed_string_utf8(str_size);
        let dataspace = DataspaceMessage::scalar();

        // Data: string bytes + null terminator
        let mut data = Vec::with_capacity(str_size as usize);
        data.extend_from_slice(value.as_bytes());
        data.push(0); // null terminator

        Self {
            name: name.to_string(),
            datatype,
            dataspace,
            data,
        }
    }

    /// Create a scalar numeric attribute with raw bytes as value.
    pub fn scalar_numeric(name: &str, datatype: DatatypeMessage, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            datatype,
            dataspace: DataspaceMessage::scalar(),
            data,
        }
    }

    /// Create a numeric array attribute with a simple dataspace.
    ///
    /// `dims` are the dimension sizes (e.g. `&[3]` for the 1-D array
    /// attributes AreaDetector writes). `data` is the row-major raw bytes and
    /// must hold `product(dims) * datatype.element_size()` bytes — the caller
    /// owns that invariant. An empty `dims` yields a scalar dataspace; prefer
    /// [`Self::scalar_numeric`] for that case.
    pub fn array_numeric(
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        data: Vec<u8>,
    ) -> Self {
        debug_assert_eq!(
            data.len() as u64,
            dims.iter().product::<u64>() * datatype.element_size() as u64,
            "array_numeric data length must equal product(dims) * element_size"
        );
        Self {
            name: name.to_string(),
            datatype,
            dataspace: DataspaceMessage::simple(dims),
            data,
        }
    }

    /// Encode the attribute message into a byte vector.
    ///
    /// The result is the raw payload for an object header message of type
    /// 0x0C (MSG_ATTRIBUTE). It does NOT include the object header message
    /// envelope (type, size, flags bytes); that is handled by the caller.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        self.encode_at(ctx, LibverBound::Earliest)
    }

    /// Encode the attribute message for a file whose low libver bound is
    /// `libver`, which the datatype message inside it follows.
    pub fn encode_at(&self, ctx: &FormatContext, libver: LibverBound) -> Vec<u8> {
        self.encode_for(ctx, libver, crate::format::ObjectFormat::Modern)
    }

    /// Encode a version-1 attribute message (`H5O__attr_encode`, H5Oattr.c).
    ///
    /// Version 1 has no flags byte and no name character set: byte 1 is
    /// reserved, and the three size fields are followed by the name, the
    /// datatype and the dataspace each padded out to a multiple of eight
    /// bytes. The size fields record the *unpadded* lengths, so a decoder that
    /// forgets the padding walks into the middle of the next field — which is
    /// why the version is not something a writer may pick freely.
    fn encode_v1(&self, ctx: &FormatContext, libver: LibverBound) -> Vec<u8> {
        /// `H5O_ALIGN_OLD`, which version 1 of this message applies to each of
        /// its three variable-length fields.
        fn pad_to_8(buf: &mut Vec<u8>) {
            let padded = (buf.len() + 7) & !7;
            buf.resize(padded, 0);
        }

        let encoded_dt = self.datatype.encode_at(ctx, libver);
        let encoded_ds = self
            .dataspace
            .encode_for(ctx, crate::format::ObjectFormat::Legacy);
        let name_bytes = self.name.as_bytes();
        let name_size = name_bytes.len() + 1;

        let mut buf = Vec::with_capacity(8 + name_size + encoded_dt.len() + encoded_ds.len() + 24);
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(name_size as u16).to_le_bytes());
        buf.extend_from_slice(&(encoded_dt.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(encoded_ds.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(0);
        pad_to_8(&mut buf);
        buf.extend_from_slice(&encoded_dt);
        pad_to_8(&mut buf);
        buf.extend_from_slice(&encoded_ds);
        pad_to_8(&mut buf);
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Encode the attribute message at the version a file of this `format`
    /// calls for, with the datatype inside it at `libver`.
    pub fn encode_for(
        &self,
        ctx: &FormatContext,
        libver: LibverBound,
        format: crate::format::ObjectFormat,
    ) -> Vec<u8> {
        if format.attribute_version() == 1 {
            return self.encode_v1(ctx, libver);
        }
        let encoded_dt = self.datatype.encode_at(ctx, libver);
        let encoded_ds = self.dataspace.encode_for(ctx, format);
        self.encode_with_fields(0x00, &encoded_dt, &encoded_ds).body
    }

    /// The version-3 body with its datatype and dataspace fields supplied.
    ///
    /// `H5O__attr_encode` writes each of the two through its message class's
    /// encoder, which is the *shared* encoder when that piece is a shared
    /// message — the field then holds a `H5O_shared_t` and the attribute's own
    /// flags byte says so (`H5O_ATTR_FLAG_TYPE_SHARED` /
    /// `H5O_ATTR_FLAG_SPACE_SHARED`, H5Oattr.c:358-359). Whichever it is, the
    /// size fields record what is actually stored, so the caller supplies the
    /// bytes and the matching flag bits and this lays the message out around
    /// them.
    pub fn encode_with_fields(
        &self,
        flags: u8,
        datatype: &[u8],
        dataspace: &[u8],
    ) -> EncodedAttribute {
        // Name with null terminator
        let name_bytes = self.name.as_bytes();
        let name_size = name_bytes.len() + 1; // +1 for null terminator

        // Total: 9 (header) + name_size + datatype_size + dataspace_size + data_size
        let total = 9 + name_size + datatype.len() + dataspace.len() + self.data.len();
        let mut buf = Vec::with_capacity(total);

        // Byte 0: version
        buf.push(ATTR_VERSION);

        // Byte 1: flags — which of the two fields below is a shared pointer.
        buf.push(flags);

        // Bytes 2-3: name size (u16 LE)
        buf.extend_from_slice(&(name_size as u16).to_le_bytes());

        // Bytes 4-5: datatype size (u16 LE)
        buf.extend_from_slice(&(datatype.len() as u16).to_le_bytes());

        // Bytes 6-7: dataspace size (u16 LE)
        buf.extend_from_slice(&(dataspace.len() as u16).to_le_bytes());

        // Byte 8: name character set encoding (1 = UTF-8)
        buf.push(0x01);

        // Name (null-terminated)
        buf.extend_from_slice(name_bytes);
        buf.push(0x00);

        let datatype_at = buf.len();
        buf.extend_from_slice(datatype);

        let dataspace_at = buf.len();
        buf.extend_from_slice(dataspace);

        // Raw data
        buf.extend_from_slice(&self.data);

        debug_assert_eq!(buf.len(), total);
        EncodedAttribute {
            body: buf,
            datatype_at,
            dataspace_at,
        }
    }

    /// Decode an attribute message from a byte buffer.
    ///
    /// Supports versions 1, 2, and 3:
    /// - v1: 8-byte header, each field padded to 8-byte alignment
    /// - v2: 8-byte header, no alignment padding
    /// - v3: 9-byte header (adds charset byte), no alignment padding
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        let AttributeHeader {
            name,
            datatype_size,
            dataspace_size,
            align,
            mut pos,
        } = AttributeHeader::decode(buf)?;

        // Datatype
        let needed = pos + datatype_size;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }
        let (datatype, _) = DatatypeMessage::decode(&buf[pos..pos + datatype_size], ctx)?;
        pos += datatype_size;
        if align > 1 {
            pos = (pos + align - 1) & !(align - 1);
        }

        // Dataspace
        let needed = pos + dataspace_size;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }
        let (dataspace, _) = DataspaceMessage::decode(&buf[pos..pos + dataspace_size], ctx)?;
        pos += dataspace_size;
        if align > 1 {
            pos = (pos + align - 1) & !(align - 1);
        }

        // Data: remaining bytes = datatype.element_size() * number_of_elements
        let num_elements: u64 = if dataspace.dims.is_empty() {
            1 // scalar
        } else {
            // dims are file-derived; saturate so a crafted attribute with
            // absurd dimensions is rejected by the buffer check below
            // instead of overflowing.
            dataspace
                .dims
                .iter()
                .fold(1u64, |acc, &d| acc.saturating_mul(d))
        };
        let data_size = num_elements
            .saturating_mul(datatype.element_size() as u64)
            .min(usize::MAX as u64) as usize;
        let needed = pos.saturating_add(data_size);
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }
        let data = buf[pos..pos + data_size].to_vec();
        pos += data_size;

        Ok((
            Self {
                name,
                datatype,
                dataspace,
                data,
            },
            pos,
        ))
    }
}

/// The part of an attribute message that identifies it: the envelope and the
/// name, both of which sit ahead of the datatype.
///
/// Split out because that ordering is what makes an undecodable attribute
/// nameable — see [`AttributeEntry::parse`].
struct AttributeHeader {
    name: String,
    datatype_size: usize,
    dataspace_size: usize,
    /// Field alignment: 8 for version 1, 1 for versions 2 and 3.
    align: usize,
    /// Offset just past the (aligned) name, where the datatype begins.
    pos: usize,
}

impl AttributeHeader {
    fn decode(buf: &[u8]) -> FormatResult<Self> {
        if buf.len() < 8 {
            return Err(FormatError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if !(1..=ATTR_VERSION).contains(&version) {
            return Err(FormatError::InvalidVersion(version));
        }

        // Byte 1 says whether the datatype and dataspace that follow are
        // bodies or references (`H5O_ATTR_FLAG_TYPE_SHARED` /
        // `H5O_ATTR_FLAG_SPACE_SHARED`). A reference decoded as a body reads
        // its version byte as the body's, which invents a type rather than
        // failing, so an attribute that carries one is named here instead.
        // Resolving it needs the file the reference points into, which a
        // message decoder does not have — an attribute read out of an object
        // header has been resolved before it gets here, one read out of dense
        // storage has not.
        let flags = buf[1];
        if flags & (ATTR_FLAG_TYPE_SHARED | ATTR_FLAG_SPACE_SHARED) != 0 {
            let what = if flags & ATTR_FLAG_TYPE_SHARED != 0 {
                "datatype"
            } else {
                "dataspace"
            };
            return Err(FormatError::UnsupportedFeature(format!(
                "attribute whose {what} is a shared-message reference"
            )));
        }
        let name_size = u16::from_le_bytes([buf[2], buf[3]]) as usize;
        let datatype_size = u16::from_le_bytes([buf[4], buf[5]]) as usize;
        let dataspace_size = u16::from_le_bytes([buf[6], buf[7]]) as usize;

        let mut pos = if version >= 3 {
            // v3 has charset byte at offset 8
            9
        } else {
            // v1, v2: no charset byte
            8
        };

        // v1 pads each field to 8-byte alignment
        let align = if version == 1 { 8 } else { 1 };

        // Name
        let needed = pos + name_size;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }
        // Strip trailing null
        let name_end = if name_size > 0 && buf[pos + name_size - 1] == 0 {
            pos + name_size - 1
        } else {
            pos + name_size
        };
        let name = String::from_utf8_lossy(&buf[pos..name_end]).to_string();
        pos += name_size;
        // v1 alignment
        if align > 1 {
            pos = (pos + align - 1) & !(align - 1);
        }

        Ok(Self {
            name,
            datatype_size,
            dataspace_size,
            align,
            pos,
        })
    }
}

/// One attribute as an object header holds it: the message, plus the creation
/// index the file records for it.
///
/// [`AttributeMessage::decode`] fails on a payload this crate cannot model —
/// an object-reference datatype, say — but the name sits ahead of the datatype
/// in the message, so such an attribute is still identifiable. Carrying the
/// unreadable case in the same list is what lets a listing answer "this object
/// has an attribute named X that I cannot read" instead of answering as though
/// X were not there.
///
/// The creation index is a property of the attribute, exactly as its name is —
/// `H5A_shared_t::crt_idx`, stored in the object header message envelope when
/// the set is compact and in the index records when it is dense. Keeping it
/// here is what stops a rewrite from re-deriving it from the position an
/// attribute happens to occupy in a list: a dense set is read back in name-hash
/// order, so a position-derived index re-stamps the whole set with the order
/// the hash walk took.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeEntry {
    body: AttributeBody,
    /// The index this attribute was created with, when its object tracks
    /// creation order. `None` when the object does not, which is what
    /// `H5O_MAX_CRT_ORDER_IDX` says on disk.
    creation_index: Option<u16>,
}

/// The message an [`AttributeEntry`] carries, decoded or not.
#[derive(Debug, Clone, PartialEq)]
enum AttributeBody {
    /// Decoded, and usable through the typed accessors.
    Readable(AttributeMessage),
    /// Named, with the reason it could not be decoded and the message payload
    /// verbatim — so a header rewrite puts back exactly what it read rather
    /// than dropping what it could not model.
    Unreadable {
        name: String,
        raw: Vec<u8>,
        reason: String,
    },
}

impl AttributeEntry {
    /// Parse one attribute message. The entry carries no creation index —
    /// only the envelope or index record it came out of knows one, so the
    /// caller that has it attaches it with
    /// [`with_creation_index`](Self::with_creation_index).
    ///
    /// Total over every message whose envelope and name parse: a payload this
    /// crate cannot decode is named, never an absence. Only a message too
    /// damaged to yield a name at all is an error, because there is then no
    /// name to report.
    pub fn parse(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let body = match AttributeMessage::decode(buf, ctx) {
            Ok((attr, _)) => AttributeBody::Readable(attr),
            Err(payload_err) => {
                let header = AttributeHeader::decode(buf)?;
                AttributeBody::Unreadable {
                    name: header.name,
                    raw: buf.to_vec(),
                    reason: payload_err.to_string(),
                }
            }
        };
        Ok(Self {
            body,
            creation_index: None,
        })
    }

    /// This entry with `creation_index` attached.
    pub fn with_creation_index(mut self, creation_index: Option<u16>) -> Self {
        self.creation_index = creation_index;
        self
    }

    /// Attach `creation_index` in place.
    pub fn set_creation_index(&mut self, creation_index: Option<u16>) {
        self.creation_index = creation_index;
    }

    /// The index this attribute was created with, or `None` when its object
    /// does not track creation order.
    pub fn creation_index(&self) -> Option<u16> {
        self.creation_index
    }

    /// The attribute's name, whether or not its payload decoded.
    pub fn name(&self) -> &str {
        match &self.body {
            AttributeBody::Readable(attr) => &attr.name,
            AttributeBody::Unreadable { name, .. } => name,
        }
    }

    /// The decoded message, or the reason there is none — exactly one of the
    /// two, so a caller reporting the failure never needs a branch for an
    /// attribute that is neither.
    pub fn decoded(&self) -> Result<&AttributeMessage, &str> {
        match &self.body {
            AttributeBody::Readable(attr) => Ok(attr),
            AttributeBody::Unreadable { reason, .. } => Err(reason),
        }
    }

    /// The decoded message, or `None` when only the name is known.
    pub fn readable(&self) -> Option<&AttributeMessage> {
        self.decoded().ok()
    }

    /// Why this attribute cannot be read, or `None` when it can be.
    pub fn unreadable_reason(&self) -> Option<&str> {
        self.decoded().err()
    }

    /// The message payload to write back into an object header.
    ///
    /// An unreadable attribute returns the bytes it was read from: re-encoding
    /// is impossible without a decoded form, and dropping it would delete an
    /// attribute the caller never asked to change.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        self.encode_at(ctx, LibverBound::Earliest)
    }

    /// The same, for a file whose low libver bound is `libver`: the datatype
    /// message inside a readable attribute follows it. An unreadable one is
    /// bytes, and bytes have no version to choose.
    pub fn encode_at(&self, ctx: &FormatContext, libver: LibverBound) -> Vec<u8> {
        self.encode_for(ctx, libver, crate::format::ObjectFormat::Modern)
    }

    /// The same, at the message version `format` calls for.
    pub fn encode_for(
        &self,
        ctx: &FormatContext,
        libver: LibverBound,
        format: crate::format::ObjectFormat,
    ) -> Vec<u8> {
        match &self.body {
            AttributeBody::Readable(attr) => attr.encode_for(ctx, libver, format),
            AttributeBody::Unreadable { raw, .. } => raw.clone(),
        }
    }
}

impl From<AttributeMessage> for AttributeEntry {
    fn from(attr: AttributeMessage) -> Self {
        Self {
            body: AttributeBody::Readable(attr),
            creation_index: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    #[test]
    fn scalar_string_roundtrip() {
        let msg = AttributeMessage::scalar_string("my_attr", "hello");
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = AttributeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.name, "my_attr");
        assert_eq!(decoded.data, b"hello\0");
        assert_eq!(decoded, msg);
    }

    #[test]
    fn scalar_string_empty() {
        let msg = AttributeMessage::scalar_string("empty", "");
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = AttributeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.name, "empty");
        assert_eq!(decoded.data, b"\0");
        assert_eq!(decoded, msg);
    }

    #[test]
    fn version_is_three() {
        let msg = AttributeMessage::scalar_string("test", "val");
        let encoded = msg.encode(&ctx());
        assert_eq!(encoded[0], 3);
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8; 4];
        let err = AttributeMessage::decode(&buf, &ctx()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_bad_version() {
        let msg = AttributeMessage::scalar_string("x", "y");
        let mut encoded = msg.encode(&ctx());
        encoded[0] = 0; // invalid version
        let err = AttributeMessage::decode(&encoded, &ctx()).unwrap_err();
        match err {
            FormatError::InvalidVersion(0) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn array_numeric_1d_roundtrip() {
        use crate::format::messages::datatype::DatatypeMessage;
        // Three int32 values, 1-D array attribute (NDArrayDimOffset-style).
        let vals: [i32; 3] = [10, -20, 30];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let msg = AttributeMessage::array_numeric(
            "dim_offset",
            DatatypeMessage::i32_type(),
            &[3],
            data.clone(),
        );
        assert_eq!(msg.dataspace.dims, vec![3]);
        let encoded = msg.encode(&ctx());
        let (decoded, consumed) = AttributeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.name, "dim_offset");
        assert_eq!(decoded.dataspace.dims, vec![3]);
        assert_eq!(decoded.data, data);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn scalar_string_utf8_content() {
        let msg = AttributeMessage::scalar_string("desc", "caf\u{00e9}");
        let encoded = msg.encode(&ctx());
        let (decoded, _) = AttributeMessage::decode(&encoded, &ctx()).unwrap();
        assert_eq!(decoded.name, "desc");
        // "caf\u{e9}" is 5 bytes in UTF-8 + null = 6
        assert_eq!(decoded.data.len(), 6);
        assert_eq!(&decoded.data[..5], "caf\u{00e9}".as_bytes());
        assert_eq!(decoded.data[5], 0);
    }

    /// The 48 bytes libhdf5 1.14.6 wrote for `f.attrs["ra"] = 42` on the root
    /// group of a default (superblock-0) file. Version 1 pads the name, the
    /// datatype and the dataspace each out to eight bytes while the size
    /// fields keep the unpadded lengths, so the byte comparison is the only
    /// thing that catches a padding rule applied in the wrong place.
    #[test]
    fn a_legacy_attribute_matches_the_bytes_libhdf5_wrote() {
        let ctx = FormatContext::default_v3();
        let attr = AttributeMessage::scalar_numeric(
            "ra",
            DatatypeMessage::i64_type(),
            42i64.to_le_bytes().to_vec(),
        );
        let buf = attr.encode_for(
            &ctx,
            LibverBound::Earliest,
            crate::format::ObjectFormat::Legacy,
        );
        assert_eq!(
            buf,
            vec![
                0x01, 0x00, 0x03, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x10, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]
        );
        let (back, consumed) = AttributeMessage::decode(&buf, &ctx).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(back.name, "ra");
        assert_eq!(back.data, 42i64.to_le_bytes().to_vec());
    }

    /// A name whose padded length differs from the datatype's, so a decoder
    /// that pads one field and not the other lands mid-value.
    #[test]
    fn a_legacy_attribute_round_trips_at_every_field_padding() {
        let ctx = FormatContext::default_v3();
        for name in ["a", "ab", "abcdefg", "abcdefgh", "abcdefghi"] {
            let attr = AttributeMessage::array_numeric(
                name,
                DatatypeMessage::i32_type(),
                &[3],
                vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            );
            let buf = attr.encode_for(
                &ctx,
                LibverBound::Earliest,
                crate::format::ObjectFormat::Legacy,
            );
            assert_eq!(buf[0], 1, "{name}");
            let (back, consumed) = AttributeMessage::decode(&buf, &ctx).unwrap();
            assert_eq!(consumed, buf.len(), "{name}");
            assert_eq!(back.name, name);
            assert_eq!(back.data, attr.data, "{name}");
            assert_eq!(back.dataspace.dims, vec![3], "{name}");
        }
    }
}
