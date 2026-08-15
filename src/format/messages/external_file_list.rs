//! External Data Files message (type 0x0007) — H5O_EFL_ID.
//!
//! A dataset with this message stores its raw data outside the HDF5 file:
//! the Data Layout message (type 0x0008) still declares `Contiguous`
//! storage, but with its address left undefined (H5Dlayout.c switches the
//! dataset's layout ops to `H5D_LOPS_EFL` — H5Defl.c — whenever this message
//! is present, overriding the layout message's own storage). The dataset's
//! logical byte range is the concatenation of every slot's reserved region,
//! in slot order.
//!
//! Binary layout (version 1, `H5O__efl_decode` in H5Oefl.c):
//! ```text
//! version: 1 byte (= 1)
//! reserved: 3 bytes
//! nalloc: u16 LE (allocated slot count, > 0)
//! nused:  u16 LE (in-use slot count, <= nalloc)
//! heap_addr: sizeof_addr bytes (local heap holding the slot names)
//! nused * {
//!     name_offset: sizeof_size bytes (offset into the local heap)
//!     offset:      sizeof_size bytes (byte offset within the named file)
//!     size:        sizeof_size bytes (bytes reserved for this slot; the
//!                  all-ones sentinel marks an unlimited/growable slot)
//! }
//! ```
//!
//! A slot's name is stored as an offset into the local heap at `heap_addr`,
//! not inline — resolving it needs a second on-disk read (the heap header,
//! then its data block), which decode does not perform; the reader does
//! that once at dataset-discovery time; see
//! [`crate::format::local_heap`].

use crate::format::bytes::{read_le_addr as read_addr, read_le_uint as read_uint};
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

const VERSION: u8 = 1;

/// The declared-size sentinel marking a slot as unlimited/growable
/// (`H5O_EFL_UNLIMITED` in H5Oprivate.h, numerically `HSIZE_UNDEF` — the same
/// all-ones pattern as [`UNDEF_ADDR`]). Only the last slot may carry it, and
/// only for a dataset with an unlimited dataspace (`H5D__efl_construct`).
pub const UNLIMITED: u64 = u64::MAX;

/// One external-file slot, before its name is resolved through the local
/// heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalFileSlot {
    /// Offset of the slot's name within the local heap at
    /// [`ExternalFileListMessage::heap_addr`].
    pub name_offset: u64,
    /// Byte offset within the named file where this slot's reserved
    /// region begins.
    pub offset: u64,
    /// Bytes reserved for this slot. The all-ones sentinel (`u64::MAX`,
    /// `H5O_EFL_UNLIMITED` in H5Oprivate.h) marks the last slot as
    /// unlimited/growable.
    pub size: u64,
}

/// Decoded External Data Files message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFileListMessage {
    /// Address of the local heap holding every slot's name.
    pub heap_addr: u64,
    /// In-use slots, in the order the dataset's logical byte range
    /// concatenates them.
    pub slots: Vec<ExternalFileSlot>,
}

impl ExternalFileListMessage {
    /// Encode the message (`H5O__efl_encode`).
    ///
    /// The allocated-slot count is written as the in-use one — upstream
    /// encodes `nused` into both fields ("yes, twice"), so a message it wrote
    /// never declares spare slots however many the property list reserved.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let nused = self.slots.len() as u16;
        let mut buf = Vec::with_capacity(8 + sa + self.slots.len() * 3 * ss);
        buf.push(VERSION);
        buf.extend_from_slice(&[0u8; 3]); // reserved
        buf.extend_from_slice(&nused.to_le_bytes()); // nalloc
        buf.extend_from_slice(&nused.to_le_bytes()); // nused
        buf.extend_from_slice(&self.heap_addr.to_le_bytes()[..sa]);
        for slot in &self.slots {
            buf.extend_from_slice(&slot.name_offset.to_le_bytes()[..ss]);
            buf.extend_from_slice(&slot.offset.to_le_bytes()[..ss]);
            buf.extend_from_slice(&slot.size.to_le_bytes()[..ss]);
        }
        buf
    }

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        if buf.len() < 4 + 4 {
            return Err(FormatError::BufferTooShort {
                needed: 4 + 4,
                available: buf.len(),
            });
        }
        let version = buf[0];
        if version != VERSION {
            return Err(FormatError::InvalidVersion(version));
        }
        // buf[1..4] reserved
        let nalloc = u16::from_le_bytes([buf[4], buf[5]]);
        let nused = u16::from_le_bytes([buf[6], buf[7]]);
        if nalloc == 0 {
            return Err(FormatError::InvalidData(
                "external file list message declares zero allocated slots".into(),
            ));
        }
        if nused > nalloc {
            return Err(FormatError::InvalidData(format!(
                "external file list message has {nused} in-use slots but only {nalloc} allocated"
            )));
        }

        let mut pos = 8;
        if buf.len() < pos + sa {
            return Err(FormatError::BufferTooShort {
                needed: pos + sa,
                available: buf.len(),
            });
        }
        let heap_addr = read_addr(&buf[pos..], sa);
        pos += sa;
        if heap_addr == UNDEF_ADDR {
            return Err(FormatError::InvalidData(
                "external file list message has an undefined local heap address".into(),
            ));
        }

        let slot_len = ss * 3;
        let mut slots = Vec::with_capacity(nused as usize);
        for _ in 0..nused {
            if buf.len() < pos + slot_len {
                return Err(FormatError::BufferTooShort {
                    needed: pos + slot_len,
                    available: buf.len(),
                });
            }
            let name_offset = read_uint(&buf[pos..], ss);
            pos += ss;
            let offset = read_uint(&buf[pos..], ss);
            pos += ss;
            let size = read_uint(&buf[pos..], ss);
            pos += ss;
            slots.push(ExternalFileSlot {
                name_offset,
                offset,
                size,
            });
        }

        Ok((Self { heap_addr, slots }, pos))
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

    /// Byte-for-byte the message `h5debug` reported for an h5py-written
    /// single-slot external dataset: heap at 1072, one slot named at heap
    /// offset 8, file offset 0, 64 bytes reserved.
    fn single_slot_buf() -> Vec<u8> {
        let mut buf = vec![VERSION, 0, 0, 0];
        buf.extend_from_slice(&1u16.to_le_bytes()); // nalloc
        buf.extend_from_slice(&1u16.to_le_bytes()); // nused
        buf.extend_from_slice(&1072u64.to_le_bytes()); // heap_addr
        buf.extend_from_slice(&8u64.to_le_bytes()); // name_offset
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        buf.extend_from_slice(&64u64.to_le_bytes()); // size
        buf
    }

    #[test]
    fn decode_single_slot() {
        let buf = single_slot_buf();
        let (msg, consumed) = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(msg.heap_addr, 1072);
        assert_eq!(
            msg.slots,
            vec![ExternalFileSlot {
                name_offset: 8,
                offset: 0,
                size: 64,
            }]
        );
    }

    #[test]
    fn decode_multi_slot() {
        let mut buf = vec![VERSION, 0, 0, 0];
        buf.extend_from_slice(&2u16.to_le_bytes()); // nalloc
        buf.extend_from_slice(&2u16.to_le_bytes()); // nused
        buf.extend_from_slice(&2000u64.to_le_bytes()); // heap_addr
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&20u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        let (msg, consumed) = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(msg.slots.len(), 2);
        assert_eq!(msg.slots[0].size, 32);
        assert_eq!(msg.slots[1].name_offset, 20);
    }

    #[test]
    fn decode_rejects_bad_version() {
        let mut buf = single_slot_buf();
        buf[0] = 2;
        let err = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(2)));
    }

    #[test]
    fn decode_rejects_zero_nalloc() {
        let mut buf = single_slot_buf();
        buf[4] = 0;
        buf[5] = 0;
        let err = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_rejects_nused_over_nalloc() {
        let mut buf = single_slot_buf();
        buf[6] = 2; // nused = 2 > nalloc = 1
        let err = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_rejects_undefined_heap_addr() {
        let mut buf = single_slot_buf();
        buf[8..16].copy_from_slice(&[0xFFu8; 8]);
        let err = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_truncated() {
        let buf = single_slot_buf();
        let err = ExternalFileListMessage::decode(&buf[..20], &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn decode_ctx4() {
        let ctx = FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        };
        let mut buf = vec![VERSION, 0, 0, 0];
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&0x800u32.to_le_bytes()); // heap_addr
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&64u32.to_le_bytes());
        let (msg, consumed) = ExternalFileListMessage::decode(&buf, &ctx).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(msg.heap_addr, 0x800);
        assert_eq!(msg.slots[0].size, 64);
    }

    /// The all-ones size sentinel (`H5O_EFL_UNLIMITED`) decodes as a plain
    /// `u64::MAX` slot size rather than being special-cased at this layer —
    /// callers that cannot support a growable slot detect and reject it
    /// explicitly instead of this message type silently reinterpreting it.
    #[test]
    fn decode_preserves_unlimited_sentinel() {
        let mut buf = single_slot_buf();
        let last = buf.len() - 8;
        buf[last..].copy_from_slice(&[0xFFu8; 8]);
        let (msg, _) = ExternalFileListMessage::decode(&buf, &ctx8()).unwrap();
        assert_eq!(msg.slots[0].size, UNLIMITED);
    }

    /// The encoder reproduces the h5py-written message this module's fixture
    /// was captured from, byte for byte.
    #[test]
    fn encode_matches_the_captured_single_slot_message() {
        let msg = ExternalFileListMessage {
            heap_addr: 1072,
            slots: vec![ExternalFileSlot {
                name_offset: 8,
                offset: 0,
                size: 64,
            }],
        };
        assert_eq!(msg.encode(&ctx8()), single_slot_buf());
    }

    #[test]
    fn encode_roundtrips_multi_slot_at_ctx4() {
        let ctx = FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        };
        let msg = ExternalFileListMessage {
            heap_addr: 0x800,
            slots: vec![
                ExternalFileSlot {
                    name_offset: 8,
                    offset: 0,
                    size: 32,
                },
                ExternalFileSlot {
                    name_offset: 20,
                    offset: 16,
                    size: 32,
                },
            ],
        };
        let buf = msg.encode(&ctx);
        let (back, consumed) = ExternalFileListMessage::decode(&buf, &ctx).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(back, msg);
    }
}
