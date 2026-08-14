//! Attribute info message (type 0x15) — metadata about attribute storage on
//! an object.
//!
//! Binary layout (version 0), per `H5O__ainfo_decode`:
//!   Byte 0: version = 0
//!   Byte 1: flags
//!     bit 0: max creation order tracked
//!     bit 1: creation order indexed
//!   [if bit 0]: max_creation_order u16 LE
//!   fractal_heap_address: sizeof_addr bytes (`UNDEF_ADDR` while compact)
//!   name_btree_address:   sizeof_addr bytes (`UNDEF_ADDR` while compact)
//!   [if bit 1]: creation_order_btree_address: sizeof_addr bytes
//!
//! The message says where the object's attributes are, not what they are:
//! while they are compact each one is its own `Attribute` message, and once
//! there are enough of them libhdf5 moves them into the fractal heap this
//! message names and writes no `Attribute` messages at all. The heap address
//! is therefore the difference between "these messages are all the
//! attributes" and "the attributes are somewhere this crate has not read".

use crate::format::bytes::read_le_addr as read_addr;
use crate::format::{FormatContext, FormatError, FormatResult};

const VERSION: u8 = 0;
const FLAG_MAX_CREATION_ORDER: u8 = 0x01;
const FLAG_CREATION_ORDER_INDEXED: u8 = 0x02;

/// Attribute info message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrInfoMessage {
    /// Maximum creation order value (present if tracking creation order).
    pub max_creation_order: Option<u16>,
    /// Fractal heap address of dense attribute storage. `UNDEF_ADDR` while
    /// the object's attributes are compact.
    pub fractal_heap_address: u64,
    /// Name-index B-tree v2 address. `UNDEF_ADDR` while compact.
    pub name_btree_address: u64,
    /// Creation-order B-tree v2 address (only when creation order is indexed).
    pub creation_order_btree_address: Option<u64>,
}

impl AttrInfoMessage {
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
        if flags & !(FLAG_MAX_CREATION_ORDER | FLAG_CREATION_ORDER_INDEXED) != 0 {
            return Err(FormatError::InvalidData(format!(
                "attribute info message has unknown flags {flags:#04x}"
            )));
        }

        let sa = ctx.sizeof_addr as usize;
        let mut pos = 2;

        let max_creation_order = if flags & FLAG_MAX_CREATION_ORDER != 0 {
            check_len(buf, pos, 2)?;
            let v = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            Some(v)
        } else {
            None
        };

        check_len(buf, pos, sa)?;
        let fractal_heap_address = read_addr(&buf[pos..], sa);
        pos += sa;

        check_len(buf, pos, sa)?;
        let name_btree_address = read_addr(&buf[pos..], sa);
        pos += sa;

        let creation_order_btree_address = if flags & FLAG_CREATION_ORDER_INDEXED != 0 {
            check_len(buf, pos, sa)?;
            let v = read_addr(&buf[pos..], sa);
            pos += sa;
            Some(v)
        } else {
            None
        };

        Ok((
            Self {
                max_creation_order,
                fractal_heap_address,
                name_btree_address,
                creation_order_btree_address,
            },
            pos,
        ))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::UNDEF_ADDR;

    fn ctx8() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    /// Creation order tracked, attributes still compact: both addresses
    /// undefined, and the creation-order counter is two bytes, not eight.
    #[test]
    fn tracked_but_compact_has_no_heap() {
        let mut buf = vec![VERSION, FLAG_MAX_CREATION_ORDER];
        buf.extend_from_slice(&3u16.to_le_bytes());
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        let (msg, consumed) = AttrInfoMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, 20);
        assert_eq!(msg.max_creation_order, Some(3));
        assert_eq!(msg.fractal_heap_address, UNDEF_ADDR);
        assert_eq!(msg.creation_order_btree_address, None);
    }

    #[test]
    fn dense_storage_names_its_heap_and_index() {
        let mut buf = vec![VERSION, 0];
        buf.extend_from_slice(&1024u64.to_le_bytes());
        buf.extend_from_slice(&2048u64.to_le_bytes());
        let (msg, consumed) = AttrInfoMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, 18);
        assert_eq!(msg.max_creation_order, None);
        assert_eq!(msg.fractal_heap_address, 1024);
        assert_eq!(msg.name_btree_address, 2048);
    }

    #[test]
    fn indexed_creation_order_carries_a_third_address() {
        let mut buf = vec![
            VERSION,
            FLAG_MAX_CREATION_ORDER | FLAG_CREATION_ORDER_INDEXED,
        ];
        buf.extend_from_slice(&7u16.to_le_bytes());
        buf.extend_from_slice(&1024u64.to_le_bytes());
        buf.extend_from_slice(&2048u64.to_le_bytes());
        buf.extend_from_slice(&4096u64.to_le_bytes());
        let (msg, consumed) = AttrInfoMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, 28);
        assert_eq!(msg.creation_order_btree_address, Some(4096));
    }

    #[test]
    fn an_unknown_version_is_refused_rather_than_guessed() {
        let buf = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            AttrInfoMessage::decode(&buf, &ctx8()),
            Err(FormatError::InvalidVersion(1))
        ));
    }

    #[test]
    fn a_truncated_message_is_short_not_zero() {
        let buf = [VERSION, 0, 0, 0, 0];
        assert!(matches!(
            AttrInfoMessage::decode(&buf, &ctx8()),
            Err(FormatError::BufferTooShort { needed: 10, .. })
        ));
    }
}
