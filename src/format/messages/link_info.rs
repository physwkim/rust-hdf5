//! Link info message (type 0x02) — metadata about link storage in a group.
//!
//! Binary layout (version 0):
//!   Byte 0: version = 0
//!   Byte 1: flags
//!     bit 0: max creation order tracked
//!     bit 1: creation order indexed
//!   [if bit 0]: max_creation_order u64 LE
//!   fractal_heap_address: sizeof_addr bytes (UNDEF if compact storage)
//!   name_btree_address:   sizeof_addr bytes (UNDEF if compact storage)
//!   [if bit 1]: creation_order_btree_address: sizeof_addr bytes

use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

const VERSION: u8 = 0;
const FLAG_MAX_CREATION_ORDER: u8 = 0x01;
const FLAG_CREATION_ORDER_INDEXED: u8 = 0x02;

/// Link info message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkInfoMessage {
    /// Maximum creation order value (present if tracking creation order).
    pub max_creation_order: Option<u64>,
    /// Fractal heap address for link name storage.  `UNDEF_ADDR` for compact groups.
    pub fractal_heap_address: u64,
    /// Name-index B-tree v2 address.  `UNDEF_ADDR` for compact groups.
    pub name_btree_address: u64,
    /// Creation-order B-tree v2 address (present only when creation order is indexed).
    pub creation_order_btree_address: Option<u64>,
}

impl LinkInfoMessage {
    /// A compact link info (no fractal heap, no B-trees).
    pub fn compact() -> Self {
        Self {
            max_creation_order: None,
            fractal_heap_address: UNDEF_ADDR,
            name_btree_address: UNDEF_ADDR,
            creation_order_btree_address: None,
        }
    }

    /// A compact link info that tracks creation order.
    pub fn compact_with_creation_order() -> Self {
        Self {
            max_creation_order: Some(0),
            fractal_heap_address: UNDEF_ADDR,
            name_btree_address: UNDEF_ADDR,
            creation_order_btree_address: None,
        }
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;

        let mut flags: u8 = 0;
        if self.max_creation_order.is_some() {
            flags |= FLAG_MAX_CREATION_ORDER;
        }
        if self.creation_order_btree_address.is_some() {
            flags |= FLAG_CREATION_ORDER_INDEXED;
        }

        let mut buf = Vec::with_capacity(2 + 8 + 3 * sa);
        buf.push(VERSION);
        buf.push(flags);

        if let Some(max_co) = self.max_creation_order {
            buf.extend_from_slice(&max_co.to_le_bytes());
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
        let has_max_co = (flags & FLAG_MAX_CREATION_ORDER) != 0;
        let has_co_btree = (flags & FLAG_CREATION_ORDER_INDEXED) != 0;

        let sa = ctx.sizeof_addr as usize;
        let mut pos = 2;

        let max_creation_order = if has_max_co {
            check_len(buf, pos, 8)?;
            let v = u64::from_le_bytes([
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ]);
            pos += 8;
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

        let creation_order_btree_address = if has_co_btree {
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

fn read_addr(buf: &[u8], n: usize) -> u64 {
    // An "undefined address" is all-bits-one in sizeof_addr bytes.
    // A 4-byte all-ones is UNDEF; a 4-byte non-all-ones is zero-extended.
    if buf[..n].iter().all(|&b| b == 0xFF) {
        UNDEF_ADDR
    } else {
        let mut tmp = [0u8; 8];
        tmp[..n].copy_from_slice(&buf[..n]);
        u64::from_le_bytes(tmp)
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
        let msg = LinkInfoMessage::compact();
        let encoded = msg.encode(&ctx8());
        // 2 header + 8 + 8 = 18
        assert_eq!(encoded.len(), 18);
        let (decoded, consumed) = LinkInfoMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 18);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_compact_ctx4() {
        let msg = LinkInfoMessage::compact();
        let encoded = msg.encode(&ctx4());
        // 2 header + 4 + 4 = 10
        assert_eq!(encoded.len(), 10);
        let (decoded, consumed) = LinkInfoMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_with_creation_order() {
        let msg = LinkInfoMessage::compact_with_creation_order();
        let encoded = msg.encode(&ctx8());
        // 2 + 8(max_co) + 8 + 8 = 26
        assert_eq!(encoded.len(), 26);
        let (decoded, consumed) = LinkInfoMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 26);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_full() {
        let msg = LinkInfoMessage {
            max_creation_order: Some(42),
            fractal_heap_address: 0x1000,
            name_btree_address: 0x2000,
            creation_order_btree_address: Some(0x3000),
        };
        let encoded = msg.encode(&ctx8());
        // 2 + 8 + 8 + 8 + 8 = 34
        assert_eq!(encoded.len(), 34);
        let (decoded, consumed) = LinkInfoMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 34);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_undef_addresses() {
        let msg = LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: UNDEF_ADDR,
            name_btree_address: UNDEF_ADDR,
            creation_order_btree_address: None,
        };
        let encoded = msg.encode(&ctx4());
        let (decoded, _) = LinkInfoMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(decoded.fractal_heap_address, UNDEF_ADDR);
        assert_eq!(decoded.name_btree_address, UNDEF_ADDR);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = LinkInfoMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidVersion(1) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8];
        let err = LinkInfoMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_byte() {
        let encoded = LinkInfoMessage::compact().encode(&ctx8());
        assert_eq!(encoded[0], 0);
    }
}
