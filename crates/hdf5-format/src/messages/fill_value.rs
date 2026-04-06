//! Fill value message (type 0x05) — specifies default fill value for unwritten elements.
//!
//! Binary layout (version 3):
//!   Byte 0: version = 3
//!   Byte 1: space allocation time (1=early, 2=late, 3=incremental)
//!   Byte 2: fill value write time (0=on_alloc, 1=never, 2=if_set)
//!   Byte 3: fill value defined (0=undefined, 1=default, 2=user-defined)
//!   [if defined == 2]: u32 LE size + size bytes of fill data

use crate::{FormatError, FormatResult};

const VERSION: u8 = 3;

/// Fill value message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct FillValueMessage {
    /// Space allocation time: 1=early, 2=late, 3=incremental.
    pub alloc_time: u8,
    /// Fill value write time: 0=on alloc, 1=never, 2=if set.
    pub fill_write_time: u8,
    /// Fill value defined: 0=undefined, 1=default (zeros), 2=user-defined.
    pub fill_defined: u8,
    /// User-defined fill value data.  Present only when `fill_defined == 2`.
    pub fill_value: Option<Vec<u8>>,
}

impl Default for FillValueMessage {
    fn default() -> Self {
        Self {
            alloc_time: 2,      // late
            fill_write_time: 0, // on alloc
            fill_defined: 1,    // default value (zeros)
            fill_value: None,
        }
    }
}

impl FillValueMessage {

    /// A user-defined fill value.
    pub fn with_value(data: Vec<u8>) -> Self {
        Self {
            alloc_time: 2,
            fill_write_time: 0,
            fill_defined: 2,
            fill_value: Some(data),
        }
    }

    /// An undefined fill value (no fill is performed).
    pub fn undefined() -> Self {
        Self {
            alloc_time: 2,
            fill_write_time: 1, // never
            fill_defined: 0,
            fill_value: None,
        }
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8);
        buf.push(VERSION);
        buf.push(self.alloc_time);
        buf.push(self.fill_write_time);
        buf.push(self.fill_defined);

        if self.fill_defined == 2 {
            if let Some(ref data) = self.fill_value {
                buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                buf.extend_from_slice(data);
            } else {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        buf
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 4 {
            return Err(FormatError::BufferTooShort {
                needed: 4,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        let alloc_time = buf[1];
        let fill_write_time = buf[2];
        let fill_defined = buf[3];

        let mut pos = 4;
        let fill_value = if fill_defined == 2 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let size =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
                    as usize;
            pos += 4;
            if buf.len() < pos + size {
                return Err(FormatError::BufferTooShort {
                    needed: pos + size,
                    available: buf.len(),
                });
            }
            let data = buf[pos..pos + size].to_vec();
            pos += size;
            Some(data)
        } else {
            None
        };

        Ok((
            Self {
                alloc_time,
                fill_write_time,
                fill_defined,
                fill_value,
            },
            pos,
        ))
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_default() {
        let msg = FillValueMessage::default();
        let encoded = msg.encode();
        assert_eq!(encoded.len(), 4);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_user_defined() {
        let msg = FillValueMessage::with_value(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let encoded = msg.encode();
        // 4 header + 4 size + 4 data = 12
        assert_eq!(encoded.len(), 12);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded, msg);
        assert_eq!(
            decoded.fill_value.as_ref().unwrap(),
            &vec![0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    fn roundtrip_undefined() {
        let msg = FillValueMessage::undefined();
        let encoded = msg.encode();
        assert_eq!(encoded.len(), 4);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_empty_user_data() {
        let msg = FillValueMessage {
            alloc_time: 1,
            fill_write_time: 2,
            fill_defined: 2,
            fill_value: Some(vec![]),
        };
        let encoded = msg.encode();
        // 4 header + 4 size + 0 data = 8
        assert_eq!(encoded.len(), 8);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [1u8, 2, 0, 1]; // version 1
        let err = FillValueMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::InvalidVersion(1) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [3u8, 2];
        let err = FillValueMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_user_defined_truncated_size() {
        // fill_defined=2 but not enough bytes for the u32 size field
        let buf = [3u8, 2, 0, 2, 0xFF];
        let err = FillValueMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_user_defined_truncated_data() {
        // fill_defined=2, size=4, but only 2 bytes of data
        let buf = [3u8, 2, 0, 2, 4, 0, 0, 0, 0xAA, 0xBB];
        let err = FillValueMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::BufferTooShort { needed: 12, available: 10 } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_byte() {
        let encoded = FillValueMessage::default().encode();
        assert_eq!(encoded[0], 3);
    }
}
