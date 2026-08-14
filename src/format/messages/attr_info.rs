//! Attribute info message (type 0x15) — metadata about attribute storage on
//! an object.
//!
//! Binary layout (version 0, `H5Oainfo.c`):
//!   Byte 0: version = 0
//!   Byte 1: flags
//!     bit 0: creation order tracked
//!     bit 1: creation order indexed
//!   [if bit 0]: max_creation_index u16 LE
//!   fractal_heap_address: sizeof_addr bytes (UNDEF when storage is compact)
//!   name_btree_address:   sizeof_addr bytes (UNDEF when storage is compact)
//!   [if bit 1]: creation_order_btree_address: sizeof_addr bytes
//!
//! The attribute *count* is deliberately absent from the encoding:
//! `H5A__get_ainfo` derives it from the name index's record count when the
//! storage is dense, and from the object header's attribute-message count when
//! it is compact. The message's mere presence is therefore what makes
//! `H5Oget_info().num_attrs` non-zero on a version-2 object header — see
//! `H5O__attr_count_real`, which reports 0 for a v2 header that has no
//! attribute info message no matter how many attribute messages it carries.

use crate::format::bytes::read_le_addr as read_addr;
use crate::format::creation_order::CreationOrder;
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

const VERSION: u8 = 0;
const FLAG_TRACK_CREATION_ORDER: u8 = 0x01;
const FLAG_CREATION_ORDER_INDEXED: u8 = 0x02;

/// Value libhdf5 uses for "no creation index assigned"
/// (`H5O_MAX_CRT_ORDER_IDX`).
pub const MAX_CREATION_ORDER_INDEX: u16 = 65535;

/// Attribute info message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeInfoMessage {
    /// Maximum creation order index (present when creation order is tracked).
    pub max_creation_index: Option<u16>,
    /// Fractal heap holding the attribute messages. `UNDEF_ADDR` when the
    /// object's attributes are stored compactly in the object header.
    pub fractal_heap_address: u64,
    /// Name-index B-tree v2 address. `UNDEF_ADDR` for compact storage.
    pub name_btree_address: u64,
    /// Creation-order B-tree v2 address (only when creation order is indexed).
    pub creation_order_btree_address: Option<u64>,
}

impl AttributeInfoMessage {
    /// An attribute info message for compact storage: the attributes live in
    /// the object header, so neither the heap nor the name index exists.
    pub fn compact() -> Self {
        Self {
            max_creation_index: None,
            fractal_heap_address: UNDEF_ADDR,
            name_btree_address: UNDEF_ADDR,
            creation_order_btree_address: None,
        }
    }

    /// Whether the attributes live in dense (fractal-heap) storage.
    pub fn is_dense(&self) -> bool {
        self.fractal_heap_address != UNDEF_ADDR
    }

    /// The attribute creation-order policy this message declares.
    ///
    /// `H5Pget_attr_creation_order` reads the object header's flags instead,
    /// and `H5O__attr_create` asserts the two agree; this accessor exists so
    /// the writer can keep that agreement rather than assume it.
    pub fn creation_order(&self) -> CreationOrder {
        CreationOrder::from_flags(
            self.max_creation_index.is_some(),
            self.creation_order_btree_address.is_some(),
        )
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;

        let mut flags: u8 = 0;
        if self.max_creation_index.is_some() {
            flags |= FLAG_TRACK_CREATION_ORDER;
        }
        if self.creation_order_btree_address.is_some() {
            flags |= FLAG_CREATION_ORDER_INDEXED;
        }

        let mut buf = Vec::with_capacity(2 + 2 + 3 * sa);
        buf.push(VERSION);
        buf.push(flags);

        if let Some(max_idx) = self.max_creation_index {
            buf.extend_from_slice(&max_idx.to_le_bytes());
        }

        buf.extend_from_slice(&self.fractal_heap_address.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.name_btree_address.to_le_bytes()[..sa]);

        if let Some(co_addr) = self.creation_order_btree_address {
            buf.extend_from_slice(&co_addr.to_le_bytes()[..sa]);
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
        let track_corder = (flags & FLAG_TRACK_CREATION_ORDER) != 0;
        let index_corder = (flags & FLAG_CREATION_ORDER_INDEXED) != 0;

        let sa = ctx.sizeof_addr as usize;
        let mut pos = 2;

        let max_creation_index = if track_corder {
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

        let creation_order_btree_address = if index_corder {
            check_len(buf, pos, sa)?;
            let v = read_addr(&buf[pos..], sa);
            pos += sa;
            Some(v)
        } else {
            None
        };

        Ok((
            Self {
                max_creation_index,
                fractal_heap_address,
                name_btree_address,
                creation_order_btree_address,
            },
            pos,
        ))
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
    fn roundtrip_compact() {
        let msg = AttributeInfoMessage::compact();
        let encoded = msg.encode(&ctx8());
        // 2 header + 8 + 8 = 18 — the size h5debug reports for the `ainfo`
        // message libhdf5 writes with default (untracked) creation order.
        assert_eq!(encoded.len(), 18);
        let (decoded, consumed) = AttributeInfoMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 18);
        assert_eq!(decoded, msg);
        assert!(!decoded.is_dense());
    }

    #[test]
    fn roundtrip_compact_ctx4() {
        let msg = AttributeInfoMessage::compact();
        let encoded = msg.encode(&ctx4());
        assert_eq!(encoded.len(), 10);
        let (decoded, consumed) = AttributeInfoMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_dense() {
        let msg = AttributeInfoMessage {
            max_creation_index: None,
            fractal_heap_address: 577,
            name_btree_address: 723,
            creation_order_btree_address: None,
        };
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = AttributeInfoMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
        assert!(decoded.is_dense());
    }

    #[test]
    fn roundtrip_with_creation_order() {
        let msg = AttributeInfoMessage {
            max_creation_index: Some(7),
            fractal_heap_address: 0x1000,
            name_btree_address: 0x2000,
            creation_order_btree_address: Some(0x3000),
        };
        let encoded = msg.encode(&ctx8());
        // 2 + 2(max idx) + 8 + 8 + 8 = 28
        assert_eq!(encoded.len(), 28);
        let (decoded, consumed) = AttributeInfoMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 28);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = AttributeInfoMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidVersion(1) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8];
        let err = AttributeInfoMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_truncated_after_flags() {
        // Flags claim a creation-order index but the addresses are missing.
        let buf = [0u8, 0x03, 0, 0];
        let err = AttributeInfoMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }
}
