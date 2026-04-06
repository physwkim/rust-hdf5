//! Group info message (type 0x0A) — optional group metadata.
//!
//! Binary layout (version 0):
//!   Byte 0: version = 0
//!   Byte 1: flags
//!     bit 0: link phase change values stored (max_compact, min_dense)
//!     bit 1: estimated entries / name length stored
//!   [if bit 0]: max_compact u16 LE, min_dense u16 LE
//!   [if bit 1]: est_num_entries u16 LE, est_name_len u16 LE

use crate::{FormatError, FormatResult};

const VERSION: u8 = 0;
const FLAG_PHASE_CHANGE: u8 = 0x01;
const FLAG_ESTIMATED: u8 = 0x02;

/// Group info message payload.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GroupInfoMessage {
    /// Max links before switching to dense storage.
    pub max_compact: Option<u16>,
    /// Min links before switching back to compact storage.
    pub min_dense: Option<u16>,
    /// Estimated number of entries (for pre-allocation).
    pub est_num_entries: Option<u16>,
    /// Estimated average link name length.
    pub est_name_len: Option<u16>,
}

impl GroupInfoMessage {

    /// Group info with phase-change thresholds.
    pub fn with_phase_change(max_compact: u16, min_dense: u16) -> Self {
        Self {
            max_compact: Some(max_compact),
            min_dense: Some(min_dense),
            est_num_entries: None,
            est_name_len: None,
        }
    }

    /// Group info with estimated entries and name length.
    pub fn with_estimates(est_num_entries: u16, est_name_len: u16) -> Self {
        Self {
            max_compact: None,
            min_dense: None,
            est_num_entries: Some(est_num_entries),
            est_name_len: Some(est_name_len),
        }
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self) -> Vec<u8> {
        let mut flags: u8 = 0;
        if self.max_compact.is_some() && self.min_dense.is_some() {
            flags |= FLAG_PHASE_CHANGE;
        }
        if self.est_num_entries.is_some() && self.est_name_len.is_some() {
            flags |= FLAG_ESTIMATED;
        }

        let mut buf = Vec::with_capacity(10);
        buf.push(VERSION);
        buf.push(flags);

        if (flags & FLAG_PHASE_CHANGE) != 0 {
            buf.extend_from_slice(&self.max_compact.unwrap().to_le_bytes());
            buf.extend_from_slice(&self.min_dense.unwrap().to_le_bytes());
        }

        if (flags & FLAG_ESTIMATED) != 0 {
            buf.extend_from_slice(&self.est_num_entries.unwrap().to_le_bytes());
            buf.extend_from_slice(&self.est_name_len.unwrap().to_le_bytes());
        }

        buf
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        let flags = buf[1];
        let mut pos = 2;

        let (max_compact, min_dense) = if (flags & FLAG_PHASE_CHANGE) != 0 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let mc = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            let md = u16::from_le_bytes([buf[pos + 2], buf[pos + 3]]);
            pos += 4;
            (Some(mc), Some(md))
        } else {
            (None, None)
        };

        let (est_num_entries, est_name_len) = if (flags & FLAG_ESTIMATED) != 0 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let ene = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            let enl = u16::from_le_bytes([buf[pos + 2], buf[pos + 3]]);
            pos += 4;
            (Some(ene), Some(enl))
        } else {
            (None, None)
        };

        Ok((
            Self {
                max_compact,
                min_dense,
                est_num_entries,
                est_name_len,
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
        let msg = GroupInfoMessage::default();
        let encoded = msg.encode();
        assert_eq!(encoded.len(), 2);
        let (decoded, consumed) = GroupInfoMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_phase_change() {
        let msg = GroupInfoMessage::with_phase_change(8, 6);
        let encoded = msg.encode();
        // 2 + 4 = 6
        assert_eq!(encoded.len(), 6);
        let (decoded, consumed) = GroupInfoMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 6);
        assert_eq!(decoded, msg);
        assert_eq!(decoded.max_compact, Some(8));
        assert_eq!(decoded.min_dense, Some(6));
    }

    #[test]
    fn roundtrip_estimates() {
        let msg = GroupInfoMessage::with_estimates(4, 16);
        let encoded = msg.encode();
        // 2 + 4 = 6
        assert_eq!(encoded.len(), 6);
        let (decoded, consumed) = GroupInfoMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 6);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_all_fields() {
        let msg = GroupInfoMessage {
            max_compact: Some(16),
            min_dense: Some(8),
            est_num_entries: Some(10),
            est_name_len: Some(32),
        };
        let encoded = msg.encode();
        // 2 + 4 + 4 = 10
        assert_eq!(encoded.len(), 10);
        let (decoded, consumed) = GroupInfoMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [1u8, 0];
        let err = GroupInfoMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::InvalidVersion(1) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8];
        let err = GroupInfoMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_truncated_phase_change() {
        // flags say phase change present but not enough bytes
        let buf = [0u8, 0x01, 0x08, 0x00];
        let err = GroupInfoMessage::decode(&buf).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_byte() {
        let encoded = GroupInfoMessage::default().encode();
        assert_eq!(encoded[0], 0);
    }
}
