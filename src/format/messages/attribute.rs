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
use crate::format::{FormatContext, FormatError, FormatResult};

const ATTR_VERSION: u8 = 3;

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

    /// Encode the attribute message into a byte vector.
    ///
    /// The result is the raw payload for an object header message of type
    /// 0x0C (MSG_ATTRIBUTE). It does NOT include the object header message
    /// envelope (type, size, flags bytes); that is handled by the caller.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let encoded_dt = self.datatype.encode(ctx);
        let encoded_ds = self.dataspace.encode(ctx);

        // Name with null terminator
        let name_bytes = self.name.as_bytes();
        let name_size = name_bytes.len() + 1; // +1 for null terminator

        // Total: 9 (header) + name_size + datatype_size + dataspace_size + data_size
        let total = 9 + name_size + encoded_dt.len() + encoded_ds.len() + self.data.len();
        let mut buf = Vec::with_capacity(total);

        // Byte 0: version
        buf.push(ATTR_VERSION);

        // Byte 1: flags (0 = non-shared)
        buf.push(0x00);

        // Bytes 2-3: name size (u16 LE)
        buf.extend_from_slice(&(name_size as u16).to_le_bytes());

        // Bytes 4-5: datatype size (u16 LE)
        buf.extend_from_slice(&(encoded_dt.len() as u16).to_le_bytes());

        // Bytes 6-7: dataspace size (u16 LE)
        buf.extend_from_slice(&(encoded_ds.len() as u16).to_le_bytes());

        // Byte 8: name character set encoding (1 = UTF-8)
        buf.push(0x01);

        // Name (null-terminated)
        buf.extend_from_slice(name_bytes);
        buf.push(0x00);

        // Encoded datatype
        buf.extend_from_slice(&encoded_dt);

        // Encoded dataspace
        buf.extend_from_slice(&encoded_ds);

        // Raw data
        buf.extend_from_slice(&self.data);

        debug_assert_eq!(buf.len(), total);
        buf
    }

    /// Decode an attribute message from a byte buffer.
    ///
    /// Supports versions 1, 2, and 3:
    /// - v1: 8-byte header, each field padded to 8-byte alignment
    /// - v2: 8-byte header, no alignment padding
    /// - v3: 9-byte header (adds charset byte), no alignment padding
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
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

        // flags at buf[1]
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
            dataspace.dims.iter().product()
        };
        let data_size = (num_elements * datatype.element_size() as u64) as usize;
        let needed = pos + data_size;
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
}
