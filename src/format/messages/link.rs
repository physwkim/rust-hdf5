//! Link message (type 0x06) — encodes a single link within a group.
//!
//! Binary layout (version 1):
//!   Byte 0: version = 1
//!   Byte 1: flags
//!     bits 0-1: size of name-length field (0=1B, 1=2B, 2=4B, 3=8B)
//!     bit 2:    creation order present
//!     bit 3:    link type present
//!     bit 4:    charset field present
//!   [if bit 3]: link_type u8 (0=hard, 1=soft, 64+=external)
//!   [if bit 2]: creation_order u32 LE
//!   [if bit 4]: charset u8 (0=ASCII, 1=UTF-8)
//!   name_length: 1/2/4/8 bytes per bits 0-1
//!   name:        name_length bytes (UTF-8)
//!   [hard link]:  address (sizeof_addr bytes)
//!   [soft link]:  target_length u16 LE + target string

use crate::format::{FormatContext, FormatError, FormatResult};

const VERSION: u8 = 1;

const FLAG_NAME_LEN_MASK: u8 = 0x03;
const FLAG_CREATION_ORDER: u8 = 0x04;
const FLAG_LINK_TYPE: u8 = 0x08;
const FLAG_CHARSET: u8 = 0x10;

const LINK_TYPE_HARD: u8 = 0;
const LINK_TYPE_SOFT: u8 = 1;

/// Link target discriminant.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkTarget {
    /// Hard link — points to an object header at `address`.
    Hard { address: u64 },
    /// Soft link — points to a path string.
    Soft { target: String },
}

/// Link message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkMessage {
    pub name: String,
    pub target: LinkTarget,
}

impl LinkMessage {
    /// Create a hard link.
    pub fn hard(name: &str, address: u64) -> Self {
        Self {
            name: name.to_string(),
            target: LinkTarget::Hard { address },
        }
    }

    /// Create a soft link.
    pub fn soft(name: &str, target: &str) -> Self {
        Self {
            name: name.to_string(),
            target: LinkTarget::Soft {
                target: target.to_string(),
            },
        }
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let name_len = name_bytes.len();
        let name_len_size = min_bytes_for_value(name_len as u64);
        let name_len_code = match name_len_size {
            1 => 0u8,
            2 => 1,
            4 => 2,
            _ => 3, // 8
        };

        let link_type = match &self.target {
            LinkTarget::Hard { .. } => LINK_TYPE_HARD,
            LinkTarget::Soft { .. } => LINK_TYPE_SOFT,
        };

        // Always store link type so that soft links are correctly identified.
        let mut flags: u8 = name_len_code & FLAG_NAME_LEN_MASK;
        flags |= FLAG_LINK_TYPE; // always include link type for clarity
        flags |= FLAG_CHARSET; // always include charset (UTF-8)

        let mut buf = Vec::with_capacity(32);
        buf.push(VERSION);
        buf.push(flags);

        // link type
        buf.push(link_type);

        // charset: 1 = UTF-8
        buf.push(1u8);

        // name length
        match name_len_size {
            1 => buf.push(name_len as u8),
            2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&(name_len as u64).to_le_bytes()),
        }

        // name
        buf.extend_from_slice(name_bytes);

        // link info
        match &self.target {
            LinkTarget::Hard { address } => {
                let sa = ctx.sizeof_addr as usize;
                buf.extend_from_slice(&address.to_le_bytes()[..sa]);
            }
            LinkTarget::Soft { target } => {
                let tbytes = target.as_bytes();
                buf.extend_from_slice(&(tbytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(tbytes);
            }
        }

        buf
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
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
        let name_len_code = flags & FLAG_NAME_LEN_MASK;
        let has_creation_order = (flags & FLAG_CREATION_ORDER) != 0;
        let has_link_type = (flags & FLAG_LINK_TYPE) != 0;
        let has_charset = (flags & FLAG_CHARSET) != 0;

        let mut pos = 2;

        // link type
        let link_type = if has_link_type {
            check_len(buf, pos, 1)?;
            let lt = buf[pos];
            pos += 1;
            lt
        } else {
            LINK_TYPE_HARD // default
        };

        // creation order
        if has_creation_order {
            check_len(buf, pos, 4)?;
            // skip creation order (u32) — we don't store it in the struct
            pos += 4;
        }

        // charset
        if has_charset {
            check_len(buf, pos, 1)?;
            // skip charset byte
            pos += 1;
        }

        // name length
        let name_len_size: usize = match name_len_code {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => 8,
        };
        check_len(buf, pos, name_len_size)?;
        let name_len = read_uint(&buf[pos..], name_len_size) as usize;
        pos += name_len_size;

        // name
        check_len(buf, pos, name_len)?;
        let name = std::str::from_utf8(&buf[pos..pos + name_len])
            .map_err(|e| FormatError::InvalidData(format!("invalid UTF-8 link name: {}", e)))?
            .to_string();
        pos += name_len;

        // target
        let target = match link_type {
            LINK_TYPE_HARD => {
                let sa = ctx.sizeof_addr as usize;
                check_len(buf, pos, sa)?;
                let address = read_uint(&buf[pos..], sa);
                pos += sa;
                LinkTarget::Hard { address }
            }
            LINK_TYPE_SOFT => {
                check_len(buf, pos, 2)?;
                let tlen = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                check_len(buf, pos, tlen)?;
                let target = std::str::from_utf8(&buf[pos..pos + tlen])
                    .map_err(|e| {
                        FormatError::InvalidData(format!("invalid UTF-8 soft link target: {}", e))
                    })?
                    .to_string();
                pos += tlen;
                LinkTarget::Soft { target }
            }
            other => {
                return Err(FormatError::UnsupportedFeature(format!(
                    "link type {}",
                    other
                )));
            }
        };

        Ok((Self { name, target }, pos))
    }
}

// ========================================================================= helpers

fn check_len(buf: &[u8], pos: usize, need: usize) -> FormatResult<()> {
    if buf.len() < pos + need {
        Err(FormatError::BufferTooShort {
            needed: pos + need,
            available: buf.len(),
        })
    } else {
        Ok(())
    }
}

/// Read a little-endian unsigned integer of `n` bytes (1..=8) into a `u64`.
fn read_uint(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

/// Minimum number of bytes (1, 2, 4, or 8) to represent `v`.
fn min_bytes_for_value(v: u64) -> usize {
    if v <= u8::MAX as u64 {
        1
    } else if v <= u16::MAX as u64 {
        2
    } else if v <= u32::MAX as u64 {
        4
    } else {
        8
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
    fn roundtrip_hard_link() {
        let msg = LinkMessage::hard("dataset1", 0x1000);
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_hard_link_ctx4() {
        let msg = LinkMessage::hard("grp", 0x2000);
        let encoded = msg.encode(&ctx4());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_soft_link() {
        let msg = LinkMessage::soft("alias", "/group/dataset");
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_empty_name() {
        // edge case: empty name
        let msg = LinkMessage::hard("", 0x100);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_long_name() {
        // name longer than 255 bytes triggers 2-byte name length
        let long_name: String = "a".repeat(300);
        let msg = LinkMessage::hard(&long_name, 0xABCD);
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_unicode_name() {
        let msg = LinkMessage::hard("日本語データ", 0x4000);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = LinkMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [2u8, 0]; // version 2 unsupported
        let err = LinkMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidVersion(2) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [1u8];
        let err = LinkMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_byte() {
        let encoded = LinkMessage::hard("x", 0).encode(&ctx8());
        assert_eq!(encoded[0], 1);
    }

    #[test]
    fn min_bytes_for_value_checks() {
        assert_eq!(min_bytes_for_value(0), 1);
        assert_eq!(min_bytes_for_value(255), 1);
        assert_eq!(min_bytes_for_value(256), 2);
        assert_eq!(min_bytes_for_value(65535), 2);
        assert_eq!(min_bytes_for_value(65536), 4);
        assert_eq!(min_bytes_for_value(u32::MAX as u64), 4);
        assert_eq!(min_bytes_for_value(u32::MAX as u64 + 1), 8);
    }
}
