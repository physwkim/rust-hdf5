//! Shared object header messages (`H5O_shared_t`).
//!
//! A message whose header flags carry [`MSG_FLAG_SHARED`] does not hold its
//! own body. What follows the message header is a *reference* to where the
//! body really lives — another object header, or the file's shared-message
//! heap — and decoding those bytes as if they were the body is not a decode
//! failure but a wrong answer: the first byte of a v2 shared message is the
//! version `2`, which a datatype decoder reads as version 0, class 2.
//!
//! This is how a dataset built on a committed (named) datatype stores its
//! type: one `H5O_SHARE_TYPE_COMMITTED` reference to the object header of the
//! datatype it shares.

use crate::format::{FormatContext, FormatError, FormatResult};

/// `H5O_MSG_FLAG_SHARED` — the message body is a reference, not the message.
pub const MSG_FLAG_SHARED: u8 = 0x02;

/// `H5O_SHARE_TYPE_SOHM`: the body is in the shared-message heap.
const SHARE_TYPE_SOHM: u8 = 1;

/// Length of the fractal-heap ID a SOHM reference carries
/// (`H5O_FHEAP_ID_LEN`).
const FHEAP_ID_LEN: usize = 8;

/// Where the body of a shared object-header message actually lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedMessage {
    /// `H5O_SHARE_TYPE_COMMITTED`: in another object header. This is the
    /// committed-datatype mechanism — the address is the object header of the
    /// named datatype, and the body is that header's message of the same
    /// type.
    Committed { object_header: u64 },
    /// `H5O_SHARE_TYPE_SOHM`: in the file's shared object header message
    /// heap, addressed by a fractal-heap ID.
    Sohm { heap_id: [u8; FHEAP_ID_LEN] },
}

impl SharedMessage {
    /// Decode the body of a message whose header flags carry
    /// [`MSG_FLAG_SHARED`], per `H5O__shared_decode`.
    ///
    /// Versions 1 and 2 predate the type byte's meaning, so both name a
    /// committed datatype whatever that byte says — v1 additionally carries
    /// six reserved bytes and the local-heap address of a symbol-table entry
    /// ahead of the object header address.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let short = |needed: usize| FormatError::BufferTooShort {
            needed,
            available: buf.len(),
        };
        if buf.len() < 2 {
            return Err(short(2));
        }
        let version = buf[0];
        if !(1..=3).contains(&version) {
            return Err(FormatError::InvalidVersion(version));
        }
        let share_type = buf[1];
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        // Only version 3 lets the type byte mean anything.
        if version == 3 && share_type == SHARE_TYPE_SOHM {
            let end = 2 + FHEAP_ID_LEN;
            if buf.len() < end {
                return Err(short(end));
            }
            let mut heap_id = [0u8; FHEAP_ID_LEN];
            heap_id.copy_from_slice(&buf[2..end]);
            return Ok(Self::Sohm { heap_id });
        }

        // The address sits after the six reserved bytes and the unused
        // local-heap address that only version 1 stores.
        let start = if version == 1 { 2 + 6 + ss } else { 2 };
        let end = start + sa;
        if buf.len() < end {
            return Err(short(end));
        }
        Ok(Self::Committed {
            object_header: crate::format::bytes::read_le_uint(&buf[start..], sa),
        })
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

    /// The bytes libhdf5 1.14.6 writes for a dataset sharing `/t` at
    /// address 800, taken verbatim from an h5py-written file.
    #[test]
    fn version_2_names_the_object_header_it_shares() {
        let buf = [0x02, 0x02, 0x20, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            SharedMessage::decode(&buf, &ctx()).unwrap(),
            SharedMessage::Committed { object_header: 800 }
        );
    }

    /// Version 1 puts six reserved bytes and a local-heap address ahead of
    /// the object header address, and its type byte is not read at all.
    #[test]
    fn version_1_skips_the_reserved_bytes_and_the_heap_address() {
        let mut buf = vec![0x01, 0x00];
        buf.extend_from_slice(&[0u8; 6]);
        buf.extend_from_slice(&123u64.to_le_bytes());
        buf.extend_from_slice(&800u64.to_le_bytes());
        assert_eq!(
            SharedMessage::decode(&buf, &ctx()).unwrap(),
            SharedMessage::Committed { object_header: 800 }
        );
    }

    /// A version-2 message whose type byte says SOHM is still committed: the
    /// type byte only means something from version 3 on.
    #[test]
    fn version_2_ignores_the_type_byte() {
        let mut buf = vec![0x02, SHARE_TYPE_SOHM];
        buf.extend_from_slice(&800u64.to_le_bytes());
        assert_eq!(
            SharedMessage::decode(&buf, &ctx()).unwrap(),
            SharedMessage::Committed { object_header: 800 }
        );
    }

    #[test]
    fn version_3_sohm_carries_a_heap_id() {
        let mut buf = vec![0x03, SHARE_TYPE_SOHM];
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            SharedMessage::decode(&buf, &ctx()).unwrap(),
            SharedMessage::Sohm {
                heap_id: [1, 2, 3, 4, 5, 6, 7, 8]
            }
        );
    }

    #[test]
    fn an_unknown_version_is_refused_rather_than_guessed() {
        let buf = [0x04, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            SharedMessage::decode(&buf, &ctx()),
            Err(FormatError::InvalidVersion(4))
        ));
    }

    #[test]
    fn a_truncated_reference_is_short_not_zero() {
        let buf = [0x02, 0x02, 0x20];
        assert!(matches!(
            SharedMessage::decode(&buf, &ctx()),
            Err(FormatError::BufferTooShort { needed: 10, .. })
        ));
    }
}
