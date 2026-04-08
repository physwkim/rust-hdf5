/// Object Header v2 encode/decode.
///
/// The Object Header is the primary metadata container in HDF5. Every named
/// object (group, dataset, committed datatype) has one. Version 2 headers use
/// the "OHDR" signature and end with a Jenkins checksum.
///
/// Layout of the header prefix (before messages):
/// ```text
/// "OHDR" (4 bytes)
/// Version: 2 (1 byte)
/// Flags (1 byte):
///   bits 0-1: chunk#0 data-size encoding (0=1B, 1=2B, 2=4B, 3=8B)
///   bit 2:    attribute creation order tracked
///   bit 3:    attribute creation order indexed
///   bit 4:    non-default attribute storage phase-change thresholds
///   bit 5:    store access/modify/change/birth timestamps
/// [if bit 5 set: 4x uint32 timestamps (16 bytes)]
/// [if bit 4 set: max_compact(u16) + min_dense(u16) (4 bytes)]
/// chunk0_data_size: 1/2/4/8 bytes depending on bits 0-1
/// <messages>
/// Checksum (4 bytes)
/// ```
///
/// Each message (v2 format):
/// ```text
/// msg_type:       u8
/// msg_data_size:  u16 LE
/// msg_flags:      u8
/// [if obj header flags bit 2: creation_order: u16 LE]
/// msg_data:       [u8; msg_data_size]
/// ```
use crate::format::checksum::checksum_metadata;
use crate::format::{FormatError, FormatResult};

/// The 4-byte object header v2 signature.
pub const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// Object header version 2.
pub const OHDR_VERSION: u8 = 2;

// Flag bit masks
const FLAG_SIZE_MASK: u8 = 0x03;
const FLAG_ATTR_CREATION_ORDER_TRACKED: u8 = 0x04;
// const FLAG_ATTR_CREATION_ORDER_INDEXED: u8 = 0x08; // bit 3
const FLAG_NON_DEFAULT_ATTR_THRESHOLDS: u8 = 0x10;
const FLAG_STORE_TIMESTAMPS: u8 = 0x20;

/// A single message within an object header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeaderMessage {
    /// Message type ID (e.g., 0x01 = Dataspace, 0x03 = Datatype, etc.)
    pub msg_type: u8,
    /// Per-message flags (bit 0 = constant, bit 1 = shared, etc.)
    pub flags: u8,
    /// Raw message payload.
    pub data: Vec<u8>,
}

/// Object Header v2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeader {
    /// Header flags byte. Bits 0-1 control chunk0 size encoding. Other bits
    /// control optional fields (timestamps, attr thresholds, creation order).
    pub flags: u8,
    /// The ordered list of header messages.
    pub messages: Vec<ObjectHeaderMessage>,
}

impl ObjectHeader {
    /// Create a new, empty object header with default flags.
    ///
    /// Defaults: bits 0-1 = 2 (4-byte chunk size encoding), no timestamps,
    /// no attribute creation order, no non-default thresholds.
    pub fn new() -> Self {
        Self {
            flags: 0x02, // bits 0-1 = 2 => 4-byte chunk0 size
            messages: Vec::new(),
        }
    }

    /// Append a message to the object header.
    pub fn add_message(&mut self, msg_type: u8, flags: u8, data: Vec<u8>) {
        self.messages.push(ObjectHeaderMessage {
            msg_type,
            flags,
            data,
        });
    }

    /// Returns the number of bytes used to encode chunk0's data size, based on
    /// flags bits 0-1.
    fn chunk0_size_bytes(&self) -> usize {
        match self.flags & FLAG_SIZE_MASK {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        }
    }

    /// Whether attribute creation order tracking is enabled (flags bit 2).
    fn has_creation_order(&self) -> bool {
        self.flags & FLAG_ATTR_CREATION_ORDER_TRACKED != 0
    }

    /// Compute the byte size of the messages region (chunk0 data).
    fn messages_data_size(&self) -> usize {
        let per_msg_overhead = if self.has_creation_order() {
            1 + 2 + 1 + 2 // type + size + flags + creation_order
        } else {
            1 + 2 + 1 // type + size + flags
        };
        self.messages
            .iter()
            .map(|m| per_msg_overhead + m.data.len())
            .sum()
    }

    /// Encode the object header to a byte vector, including "OHDR" signature
    /// and trailing checksum.
    pub fn encode(&self) -> Vec<u8> {
        let messages_size = self.messages_data_size();

        // Estimate total size for pre-allocation
        let mut prefix_size: usize = 4 + 1 + 1; // OHDR + version + flags
        if self.flags & FLAG_STORE_TIMESTAMPS != 0 {
            prefix_size += 16; // 4 x u32
        }
        if self.flags & FLAG_NON_DEFAULT_ATTR_THRESHOLDS != 0 {
            prefix_size += 4; // max_compact(u16) + min_dense(u16)
        }
        prefix_size += self.chunk0_size_bytes(); // chunk0 data size field
        let total = prefix_size + messages_size + 4; // + checksum

        let mut buf = Vec::with_capacity(total);

        // Signature
        buf.extend_from_slice(&OHDR_SIGNATURE);
        // Version
        buf.push(OHDR_VERSION);
        // Flags
        buf.push(self.flags);

        // Optional timestamps (bit 5) -- for MVP we write zeros if enabled
        if self.flags & FLAG_STORE_TIMESTAMPS != 0 {
            buf.extend_from_slice(&[0u8; 16]);
        }

        // Optional attr storage thresholds (bit 4) -- write defaults if enabled
        if self.flags & FLAG_NON_DEFAULT_ATTR_THRESHOLDS != 0 {
            // max_compact = 8, min_dense = 6 (HDF5 defaults)
            buf.extend_from_slice(&8u16.to_le_bytes());
            buf.extend_from_slice(&6u16.to_le_bytes());
        }

        // Chunk0 data size
        let chunk0_data_size = messages_size as u64;
        let csb = self.chunk0_size_bytes();
        buf.extend_from_slice(&chunk0_data_size.to_le_bytes()[..csb]);

        // Messages
        for msg in &self.messages {
            buf.push(msg.msg_type);
            buf.extend_from_slice(&(msg.data.len() as u16).to_le_bytes());
            buf.push(msg.flags);
            if self.has_creation_order() {
                // We don't track actual creation order values in the MVP --
                // write 0.
                buf.extend_from_slice(&0u16.to_le_bytes());
            }
            buf.extend_from_slice(&msg.data);
        }

        // Checksum over everything before the checksum
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), total);
        buf
    }

    /// Decode an object header from a byte buffer. Returns the parsed header
    /// and the number of bytes consumed from the buffer.
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        // Minimum: OHDR(4) + version(1) + flags(1) + chunk0_size(1) + checksum(4) = 11
        if buf.len() < 11 {
            return Err(FormatError::BufferTooShort {
                needed: 11,
                available: buf.len(),
            });
        }

        // Signature
        if buf[0..4] != OHDR_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        // Version
        let version = buf[4];
        if version != OHDR_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        let flags = buf[5];
        let mut pos: usize = 6;

        // Optional timestamps (bit 5)
        if flags & FLAG_STORE_TIMESTAMPS != 0 {
            if buf.len() < pos + 16 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 16,
                    available: buf.len(),
                });
            }
            // Skip timestamps for now (MVP doesn't use them)
            pos += 16;
        }

        // Optional attr storage thresholds (bit 4)
        if flags & FLAG_NON_DEFAULT_ATTR_THRESHOLDS != 0 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            // Skip thresholds for now
            pos += 4;
        }

        // Chunk0 data size
        let chunk0_size_bytes = match flags & FLAG_SIZE_MASK {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };

        if buf.len() < pos + chunk0_size_bytes {
            return Err(FormatError::BufferTooShort {
                needed: pos + chunk0_size_bytes,
                available: buf.len(),
            });
        }

        let mut size_bytes = [0u8; 8];
        size_bytes[..chunk0_size_bytes].copy_from_slice(&buf[pos..pos + chunk0_size_bytes]);
        let chunk0_data_size = u64::from_le_bytes(size_bytes) as usize;
        pos += chunk0_size_bytes;

        // We need chunk0_data_size bytes of messages + 4 bytes of checksum
        let total_consumed = pos + chunk0_data_size + 4;
        if buf.len() < total_consumed {
            return Err(FormatError::BufferTooShort {
                needed: total_consumed,
                available: buf.len(),
            });
        }

        // Verify checksum: covers everything from start up to (but not
        // including) the 4-byte checksum.
        let data_end = total_consumed - 4;
        let stored_cksum = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        // Parse messages
        let has_creation_order = flags & FLAG_ATTR_CREATION_ORDER_TRACKED != 0;
        let messages_end = pos + chunk0_data_size;
        let mut messages = Vec::new();

        while pos < messages_end {
            // Each message: type(1) + size(2) + flags(1) [+ creation_order(2)]
            let msg_header_size = if has_creation_order { 6 } else { 4 };
            if pos + msg_header_size > messages_end {
                return Err(FormatError::InvalidData(
                    "truncated message header in object header".into(),
                ));
            }

            let msg_type = buf[pos];
            let msg_data_size = u16::from_le_bytes([buf[pos + 1], buf[pos + 2]]) as usize;
            let msg_flags = buf[pos + 3];
            pos += 4;

            if has_creation_order {
                // Skip creation_order for now
                pos += 2;
            }

            if pos + msg_data_size > messages_end {
                return Err(FormatError::InvalidData(format!(
                    "message data ({} bytes) extends past chunk0 boundary",
                    msg_data_size
                )));
            }

            let data = buf[pos..pos + msg_data_size].to_vec();
            pos += msg_data_size;

            messages.push(ObjectHeaderMessage {
                msg_type,
                flags: msg_flags,
                data,
            });
        }

        Ok((ObjectHeader { flags, messages }, total_consumed))
    }
}

impl Default for ObjectHeader {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Object Header v1 — decode only (for reading legacy HDF5 files)
// =========================================================================

impl ObjectHeader {
    /// Decode a v1 object header from a byte buffer.
    ///
    /// v1 headers do NOT have the "OHDR" signature or a checksum. The layout is:
    /// ```text
    /// Byte 0: version = 1
    /// Byte 1: reserved
    /// Bytes 2-3: num_messages (u16 LE)
    /// Bytes 4-7: obj_ref_count (u32 LE)
    /// Bytes 8-11: header_data_size (u32 LE) — size of message data in first chunk
    /// Messages follow, each:
    ///   type: u16 LE
    ///   data_size: u16 LE
    ///   flags: u8
    ///   reserved: 3 bytes
    ///   data: data_size bytes (padded to 8-byte alignment)
    /// ```
    pub fn decode_v1(buf: &[u8]) -> FormatResult<(Self, usize)> {
        // V1 header prefix is 16 bytes: version(1) + reserved(1) + num_msg(2)
        // + ref_count(4) + chunk0_data_size(4) + reserved_padding(4)
        if buf.len() < 16 {
            return Err(FormatError::BufferTooShort {
                needed: 16,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != 1 {
            return Err(FormatError::InvalidVersion(version));
        }

        // buf[1] = reserved
        let num_messages = u16::from_le_bytes([buf[2], buf[3]]) as usize;
        let _obj_ref_count = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let header_data_size = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        // buf[12..16] = reserved alignment padding

        let total_consumed = 16 + header_data_size;
        if buf.len() < total_consumed {
            return Err(FormatError::BufferTooShort {
                needed: total_consumed,
                available: buf.len(),
            });
        }

        let msg_data_start = 16; // offset where message data begins (after 16-byte prefix)
        let mut pos = msg_data_start;
        let messages_end = msg_data_start + header_data_size;
        let mut messages = Vec::with_capacity(num_messages);

        for _ in 0..num_messages {
            if pos + 8 > messages_end {
                break; // no more room for a message header
            }

            let msg_type = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            let data_size = u16::from_le_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
            let msg_flags = buf[pos + 4];
            // bytes pos+5..pos+8 are reserved
            pos += 8;

            if pos + data_size > messages_end {
                return Err(FormatError::InvalidData(format!(
                    "v1 message data ({} bytes) extends past header boundary",
                    data_size
                )));
            }

            let data = buf[pos..pos + data_size].to_vec();
            pos += data_size;

            // In v1, messages are padded to 8-byte alignment relative to
            // the start of the message data region.
            let rel = pos - msg_data_start;
            let aligned_rel = (rel + 7) & !7;
            let aligned_pos = msg_data_start + aligned_rel;
            if aligned_pos <= messages_end {
                pos = aligned_pos;
            }

            // Skip null/padding messages (type 0)
            if msg_type == 0 {
                continue;
            }

            messages.push(ObjectHeaderMessage {
                msg_type: msg_type as u8,
                flags: msg_flags,
                data,
            });
        }

        Ok((
            ObjectHeader {
                flags: 0x02, // default flags (not meaningful for v1)
                messages,
            },
            total_consumed,
        ))
    }

    /// Auto-detect and decode either v1 or v2 object header.
    ///
    /// Checks for the "OHDR" signature to decide v2; otherwise tries v1.
    pub fn decode_any(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() >= 4 && buf[0..4] == OHDR_SIGNATURE {
            Self::decode(buf)
        } else if !buf.is_empty() && buf[0] == 1 {
            Self::decode_v1(buf)
        } else {
            // Try v2 first (will fail with proper error)
            Self::decode(buf)
        }
    }
}

#[cfg(test)]
mod tests_v1 {
    use super::*;

    /// Build a minimal v1 object header with given messages.
    fn build_v1_header(messages: &[(u16, u8, &[u8])]) -> Vec<u8> {
        let mut msg_data = Vec::new();
        for (msg_type, flags, data) in messages {
            msg_data.extend_from_slice(&msg_type.to_le_bytes());
            msg_data.extend_from_slice(&(data.len() as u16).to_le_bytes());
            msg_data.push(*flags);
            msg_data.extend_from_slice(&[0u8; 3]); // reserved
            msg_data.extend_from_slice(data);
            // Pad to 8-byte alignment
            let aligned = (msg_data.len() + 7) & !7;
            msg_data.resize(aligned, 0);
        }

        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(messages.len() as u16).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ref count
        buf.extend_from_slice(&(msg_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // reserved padding (align to 16 bytes)
        buf.extend_from_slice(&msg_data);
        buf
    }

    #[test]
    fn test_decode_v1_empty() {
        let buf = build_v1_header(&[]);
        let (hdr, consumed) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(consumed, 16); // 16-byte prefix, no messages
        assert!(hdr.messages.is_empty());
    }

    #[test]
    fn test_decode_v1_single_message() {
        let data = vec![0xAA, 0xBB, 0xCC];
        let buf = build_v1_header(&[(0x03, 0x00, &data)]);
        let (hdr, _consumed) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, 0x03);
        assert_eq!(hdr.messages[0].data, data);
    }

    #[test]
    fn test_decode_v1_multiple_messages() {
        let buf = build_v1_header(&[
            (0x01, 0x00, &[1, 2, 3, 4]),
            (0x03, 0x01, &[10, 20]),
            (0x08, 0x00, &[0xFF; 16]),
        ]);
        let (hdr, _) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(hdr.messages.len(), 3);
        assert_eq!(hdr.messages[0].msg_type, 0x01);
        assert_eq!(hdr.messages[1].msg_type, 0x03);
        assert_eq!(hdr.messages[2].msg_type, 0x08);
        assert_eq!(hdr.messages[2].data, vec![0xFF; 16]);
    }

    #[test]
    fn test_decode_v1_skips_null_messages() {
        let buf = build_v1_header(&[
            (0x00, 0x00, &[0; 8]), // null message (type 0)
            (0x03, 0x00, &[1, 2]),
        ]);
        let (hdr, _) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, 0x03);
    }

    #[test]
    fn test_decode_any_v2() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);
        let encoded = hdr.encode();
        let (decoded, _) = ObjectHeader::decode_any(&encoded).unwrap();
        assert_eq!(decoded.messages.len(), 1);
    }

    #[test]
    fn test_decode_any_v1() {
        let buf = build_v1_header(&[(0x03, 0x00, &[1, 2])]);
        let (decoded, _) = ObjectHeader::decode_any(&buf).unwrap();
        assert_eq!(decoded.messages.len(), 1);
        assert_eq!(decoded.messages[0].msg_type, 0x03);
    }

    #[test]
    fn test_decode_v1_bad_version() {
        let mut buf = build_v1_header(&[]);
        buf[0] = 5;
        assert!(matches!(
            ObjectHeader::decode_v1(&buf).unwrap_err(),
            FormatError::InvalidVersion(5)
        ));
    }

    #[test]
    fn test_decode_v1_buffer_too_short() {
        assert!(matches!(
            ObjectHeader::decode_v1(&[1, 0, 0]).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_header_roundtrip() {
        let hdr = ObjectHeader::new();
        let encoded = hdr.encode();

        // OHDR(4) + version(1) + flags(1) + chunk0_size(4) + checksum(4) = 14
        assert_eq!(encoded.len(), 14);
        assert_eq!(&encoded[..4], b"OHDR");
        assert_eq!(encoded[4], 2); // version

        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_single_message_roundtrip() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![0xAA, 0xBB, 0xCC]);

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages.len(), 1);
        assert_eq!(decoded.messages[0].msg_type, 0x01);
        assert_eq!(decoded.messages[0].flags, 0x00);
        assert_eq!(decoded.messages[0].data, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_multiple_messages_roundtrip() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3, 4]);
        hdr.add_message(0x03, 0x01, vec![10, 20]);
        hdr.add_message(0x0C, 0x00, vec![]);

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages.len(), 3);
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_with_creation_order() {
        let mut hdr = ObjectHeader {
            flags: 0x02 | FLAG_ATTR_CREATION_ORDER_TRACKED,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![0xFF; 8]);
        hdr.add_message(0x03, 0x00, vec![0xEE; 4]);

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages.len(), 2);
        assert_eq!(decoded.messages[0].data, vec![0xFF; 8]);
        assert_eq!(decoded.messages[1].data, vec![0xEE; 4]);
    }

    #[test]
    fn test_chunk0_size_1byte() {
        // flags bits 0-1 = 0 => 1-byte chunk0 size
        let mut hdr = ObjectHeader {
            flags: 0x00,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![42]);

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data, vec![42]);
    }

    #[test]
    fn test_chunk0_size_2byte() {
        // flags bits 0-1 = 1 => 2-byte chunk0 size
        let mut hdr = ObjectHeader {
            flags: 0x01,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn test_chunk0_size_8byte() {
        // flags bits 0-1 = 3 => 8-byte chunk0 size
        let mut hdr = ObjectHeader {
            flags: 0x03,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![0xDE, 0xAD]);

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_decode_bad_signature() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"XHDR");
        let err = ObjectHeader::decode(&data).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn test_decode_bad_version() {
        let hdr = ObjectHeader::new();
        let mut encoded = hdr.encode();
        encoded[4] = 99; // corrupt version
        let err = ObjectHeader::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(99)));
    }

    #[test]
    fn test_decode_checksum_mismatch() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);
        let mut encoded = hdr.encode();
        // Corrupt a message byte
        let last_data = encoded.len() - 5;
        encoded[last_data] ^= 0xFF;
        let err = ObjectHeader::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let err = ObjectHeader::decode(&[0u8; 5]).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn test_decode_with_trailing_data() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![7, 8, 9]);
        let mut encoded = hdr.encode();
        let original_len = encoded.len();
        encoded.extend_from_slice(&[0xBB; 50]); // trailing garbage

        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, original_len);
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_large_message_payload() {
        let mut hdr = ObjectHeader::new();
        let big_data = vec![0x42; 1000];
        hdr.add_message(0x0C, 0x00, big_data.clone());

        let encoded = hdr.encode();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data.len(), 1000);
        assert_eq!(decoded.messages[0].data, big_data);
    }

    #[test]
    fn test_default() {
        let hdr = ObjectHeader::default();
        assert_eq!(hdr.flags, 0x02);
        assert!(hdr.messages.is_empty());
    }
}
