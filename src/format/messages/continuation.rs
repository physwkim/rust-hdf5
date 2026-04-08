//! Object header continuation message (type 0x10).
//!
//! Binary layout:
//!   offset: sizeof_addr bytes (LE) — address of continuation block
//!   length: sizeof_size bytes (LE) — length of continuation block

use crate::format::{FormatContext, FormatError, FormatResult};

/// Object header continuation message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationMessage {
    /// File offset of the continuation block.
    pub offset: u64,
    /// Length of the continuation block in bytes.
    pub length: u64,
}

impl ContinuationMessage {
    pub fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let mut buf = Vec::with_capacity(sa + ss);
        buf.extend_from_slice(&self.offset.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.length.to_le_bytes()[..ss]);
        buf
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let needed = sa + ss;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }

        let offset = read_uint(&buf[0..], sa);
        let length = read_uint(&buf[sa..], ss);

        Ok((Self { offset, length }, needed))
    }
}

/// Read a little-endian unsigned integer of `n` bytes (1..=8) into a `u64`.
fn read_uint(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
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
    fn roundtrip_ctx8() {
        let msg = ContinuationMessage::new(0x1000, 256);
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded.len(), 16);
        let (decoded, consumed) = ContinuationMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 16);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_ctx4() {
        let msg = ContinuationMessage::new(0x800, 128);
        let encoded = msg.encode(&ctx4());
        assert_eq!(encoded.len(), 8);
        let (decoded, consumed) = ContinuationMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_zero() {
        let msg = ContinuationMessage::new(0, 0);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = ContinuationMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_large_values() {
        let msg = ContinuationMessage::new(0xDEAD_BEEF_CAFE_0000, 0x0000_FFFF_0000_1234);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = ContinuationMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8; 4];
        let err = ContinuationMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort {
                needed: 16,
                available: 4,
            } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short_ctx4() {
        let buf = [0u8; 6];
        let err = ContinuationMessage::decode(&buf, &ctx4()).unwrap_err();
        match err {
            FormatError::BufferTooShort {
                needed: 8,
                available: 6,
            } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn encode_size() {
        let msg = ContinuationMessage::new(1, 2);
        assert_eq!(msg.encode(&ctx8()).len(), 16);
        assert_eq!(msg.encode(&ctx4()).len(), 8);
    }
}
