//! Shared object header messages (SOHM).
//!
//! A message that several objects would encode identically — most often a
//! datatype, dataspace or attribute — can be stored once and referenced from
//! every object header that uses it. The referencing header then holds a
//! *pointer*, not the message: either a fractal-heap ID into the SOHM heap, or
//! the address of the object header that owns the message (a committed
//! datatype). The `H5O_MSG_FLAG_SHARED` bit on the message says which of the
//! two shapes the body has.
//!
//! Resolving a pointer needs the SOHM master table, whose address comes from
//! the superblock extension's shared-message-table message: it maps a message
//! type to the fractal heap holding that type's shared messages.
//!
//! Upstream: `H5Oshared.c` (`H5O__shared_decode`, `H5O__shared_read`),
//! `H5SM.c` (`H5SM_get_fheap_addr`, `H5SM__type_to_flag`), `H5SMcache.c`
//! (`H5SM__cache_table_deserialize`).

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::checksum::checksum_metadata;
use crate::format::messages::{
    MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE, MSG_FILL_VALUE, MSG_FILL_VALUE_OLD,
    MSG_FILTER_PIPELINE,
};
use crate::format::{FormatContext, FormatError, FormatResult};

/// SOHM master table signature.
pub const SMTB_SIGNATURE: [u8; 4] = *b"SMTB";
/// SOHM list index signature.
pub const SMLI_SIGNATURE: [u8; 4] = *b"SMLI";

/// Length of a SOHM fractal-heap ID (`H5O_FHEAP_ID_LEN`). Unlike other heaps,
/// the SOHM heap's ID length is fixed by the format, not read from the heap.
pub const SOHM_HEAP_ID_LEN: usize = 8;

/// Where a shared message's body actually lives (`H5O_SHARE_TYPE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedLocation {
    /// Not shared: the body is the message itself.
    Unshared,
    /// In the SOHM fractal heap, addressed by a heap ID.
    Sohm,
    /// In another object header (a committed datatype).
    Committed,
    /// Shared, but stored in this same object header.
    Here,
}

impl SharedLocation {
    fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::Sohm,
            2 => Self::Committed,
            3 => Self::Here,
            _ => Self::Unshared,
        }
    }
}

/// A decoded shared-message pointer: the body of any object-header message
/// whose `H5O_MSG_FLAG_SHARED` bit is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessagePointer {
    /// Encoding version: 1, 2 or 3.
    pub version: u8,
    /// Which storage the pointer names.
    pub location: SharedLocation,
    /// Fractal-heap ID, for `Sohm`.
    pub heap_id: [u8; SOHM_HEAP_ID_LEN],
    /// Object header address, for `Committed` (and for the version-1 form,
    /// which is always committed).
    pub oh_addr: u64,
}

impl SharedMessagePointer {
    /// Decode the pointer that stands in for the message body.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        need(buf, 2)?;
        let version = buf[0];
        if version == 0 || version > 3 {
            return Err(FormatError::InvalidVersion(version));
        }

        // The type byte is unused before version 2: those messages are always
        // committed datatypes.
        let mut location = if version >= 2 {
            SharedLocation::from_byte(buf[1])
        } else {
            SharedLocation::Committed
        };
        let mut pos = 2;

        let mut heap_id = [0u8; SOHM_HEAP_ID_LEN];
        let mut oh_addr = 0u64;
        if version == 1 {
            // 6 reserved bytes, then a stripped-down symbol table entry: a
            // local-heap address that is skipped, then the object header's.
            pos += 6;
            need(buf, pos + ss + sa)?;
            pos += ss;
            oh_addr = read_uint(&buf[pos..], sa);
        } else if location == SharedLocation::Sohm {
            if version < 3 {
                return Err(FormatError::InvalidData(
                    "heap-shared message pointer requires version 3".into(),
                ));
            }
            need(buf, pos + SOHM_HEAP_ID_LEN)?;
            heap_id.copy_from_slice(&buf[pos..pos + SOHM_HEAP_ID_LEN]);
        } else {
            // Before version 3 the committed flag did not exist, so anything
            // that is not heap-shared is a committed datatype.
            if version < 3 {
                location = SharedLocation::Committed;
            }
            need(buf, pos + sa)?;
            oh_addr = read_uint(&buf[pos..], sa);
        }

        Ok(Self {
            version,
            location,
            heap_id,
            oh_addr,
        })
    }
}

/// One SOHM index header, as stored in the master table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmIndexHeader {
    /// 0 = list, 1 = v2 B-tree. The index is a write-side lookup structure;
    /// reading a shared message needs only `heap_addr`.
    pub index_type: u8,
    /// Bit mask of the message types this index covers (`H5SM__type_to_flag`).
    pub mesg_types: u16,
    /// Smallest message this index will share.
    pub min_mesg_size: u32,
    /// Message count at which a list index becomes a B-tree.
    pub list_max: u16,
    /// Message count at which a B-tree index becomes a list again.
    pub btree_min: u16,
    /// Messages currently in the index.
    pub num_messages: u16,
    /// Address of the list or B-tree.
    pub index_addr: u64,
    /// Address of the fractal heap holding this index's message bodies.
    pub heap_addr: u64,
}

/// The SOHM master table (`SMTB`): one index header per shared-message index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SohmMasterTable {
    /// Index headers, in table order.
    pub indexes: Vec<SohmIndexHeader>,
}

/// Version of a SOHM index header (`H5SM_LIST_VERSION`).
const SM_LIST_VERSION: u8 = 0;

impl SohmMasterTable {
    /// On-disk size of a table with `nindexes` index headers
    /// (`H5SM_TABLE_SIZE` / `H5SM_INDEX_HEADER_SIZE`).
    pub fn encoded_size(ctx: &FormatContext, nindexes: u8) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let per_index = 1 + 1 + 2 + 4 + 2 + 2 + 2 + sa + sa;
        4 + nindexes as usize * per_index + 4
    }

    /// Decode the master table. `nindexes` comes from the shared-message-table
    /// message in the superblock extension — the table itself does not store
    /// it.
    pub fn decode(buf: &[u8], ctx: &FormatContext, nindexes: u8) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let size = Self::encoded_size(ctx, nindexes);
        need(buf, size)?;
        if buf[0..4] != SMTB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let stored =
            u32::from_le_bytes([buf[size - 4], buf[size - 3], buf[size - 2], buf[size - 1]]);
        let computed = checksum_metadata(&buf[..size - 4]);
        if stored != computed {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }

        let mut pos = 4;
        let mut indexes = Vec::with_capacity(nindexes as usize);
        for _ in 0..nindexes {
            let version = buf[pos];
            if version != SM_LIST_VERSION {
                return Err(FormatError::InvalidVersion(version));
            }
            pos += 1;
            let index_type = buf[pos];
            pos += 1;
            let mesg_types = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let min_mesg_size =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
            pos += 4;
            let list_max = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let btree_min = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let num_messages = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let index_addr = read_uint(&buf[pos..], sa);
            pos += sa;
            let heap_addr = read_uint(&buf[pos..], sa);
            pos += sa;
            indexes.push(SohmIndexHeader {
                index_type,
                mesg_types,
                min_mesg_size,
                list_max,
                btree_min,
                num_messages,
                index_addr,
                heap_addr,
            });
        }

        Ok(Self { indexes })
    }

    /// Address of the fractal heap holding shared messages of `msg_type`, as
    /// `H5SM_get_fheap_addr` resolves it.
    pub fn heap_addr(&self, msg_type: u8) -> Option<u64> {
        let flag = type_flag(msg_type)?;
        self.indexes
            .iter()
            .find(|i| i.mesg_types & flag != 0)
            .map(|i| i.heap_addr)
    }
}

/// The index bit for a message type (`H5SM__type_to_flag`). Only the five
/// shareable types have one; the old fill-value message shares the new one's
/// bit, matching upstream.
pub fn type_flag(msg_type: u8) -> Option<u16> {
    let id = match msg_type {
        MSG_DATASPACE | MSG_DATATYPE | MSG_FILL_VALUE | MSG_FILTER_PIPELINE | MSG_ATTRIBUTE => {
            msg_type
        }
        MSG_FILL_VALUE_OLD => MSG_FILL_VALUE,
        _ => return None,
    };
    Some(1u16 << id)
}

fn need(buf: &[u8], n: usize) -> FormatResult<()> {
    if buf.len() < n {
        Err(FormatError::BufferTooShort {
            needed: n,
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

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    fn index_header(mesg_types: u16, heap_addr: u64) -> Vec<u8> {
        let mut b = vec![SM_LIST_VERSION, 0];
        b.extend_from_slice(&mesg_types.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // min_mesg_size
        b.extend_from_slice(&50u16.to_le_bytes()); // list_max
        b.extend_from_slice(&40u16.to_le_bytes()); // btree_min
        b.extend_from_slice(&7u16.to_le_bytes()); // num_messages
        b.extend_from_slice(&0x1234u64.to_le_bytes()); // index_addr
        b.extend_from_slice(&heap_addr.to_le_bytes());
        b
    }

    fn master_table(headers: &[Vec<u8>]) -> Vec<u8> {
        let mut b = SMTB_SIGNATURE.to_vec();
        for h in headers {
            b.extend_from_slice(h);
        }
        let sum = checksum_metadata(&b);
        b.extend_from_slice(&sum.to_le_bytes());
        b
    }

    #[test]
    fn master_table_roundtrip() {
        let buf = master_table(&[index_header(1 << MSG_DATATYPE, 0x8000)]);
        assert_eq!(buf.len(), SohmMasterTable::encoded_size(&ctx(), 1));
        let t = SohmMasterTable::decode(&buf, &ctx(), 1).unwrap();
        assert_eq!(t.indexes.len(), 1);
        assert_eq!(t.indexes[0].num_messages, 7);
        assert_eq!(t.indexes[0].heap_addr, 0x8000);
    }

    #[test]
    fn master_table_rejects_a_corrupt_checksum() {
        let mut buf = master_table(&[index_header(1 << MSG_DATATYPE, 0x8000)]);
        let n = buf.len();
        buf[n - 1] ^= 0xff;
        assert!(matches!(
            SohmMasterTable::decode(&buf, &ctx(), 1).unwrap_err(),
            FormatError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn master_table_rejects_a_bad_index_version() {
        let mut hdr = index_header(1 << MSG_DATATYPE, 0x8000);
        hdr[0] = 1;
        let buf = master_table(&[hdr]);
        assert!(matches!(
            SohmMasterTable::decode(&buf, &ctx(), 1).unwrap_err(),
            FormatError::InvalidVersion(1)
        ));
    }

    /// The heap is picked by the index whose type mask covers the message,
    /// not by index order.
    #[test]
    fn heap_address_is_selected_by_message_type() {
        let buf = master_table(&[
            index_header(1 << MSG_ATTRIBUTE, 0x1000),
            index_header((1 << MSG_DATATYPE) | (1 << MSG_DATASPACE), 0x2000),
        ]);
        let t = SohmMasterTable::decode(&buf, &ctx(), 2).unwrap();
        assert_eq!(t.heap_addr(MSG_ATTRIBUTE), Some(0x1000));
        assert_eq!(t.heap_addr(MSG_DATATYPE), Some(0x2000));
        assert_eq!(t.heap_addr(MSG_DATASPACE), Some(0x2000));
        // Not a shareable type at all.
        assert_eq!(t.heap_addr(crate::format::messages::MSG_DATA_LAYOUT), None);
    }

    /// `H5SM__type_to_flag` maps the old fill-value message onto the new
    /// one's bit, so a file sharing fill values finds one index either way.
    #[test]
    fn old_fill_value_shares_the_new_fill_value_bit() {
        assert_eq!(type_flag(MSG_FILL_VALUE_OLD), type_flag(MSG_FILL_VALUE));
        assert_eq!(type_flag(MSG_FILL_VALUE), Some(1 << MSG_FILL_VALUE));
        assert_eq!(type_flag(crate::format::messages::MSG_LINK), None);
    }

    #[test]
    fn version_three_heap_pointer_carries_a_heap_id() {
        let mut buf = vec![3u8, 1u8];
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = SharedMessagePointer::decode(&buf, &ctx()).unwrap();
        assert_eq!(p.location, SharedLocation::Sohm);
        assert_eq!(p.heap_id, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn version_three_committed_pointer_carries_an_address() {
        let mut buf = vec![3u8, 2u8];
        buf.extend_from_slice(&0x4321u64.to_le_bytes());
        let p = SharedMessagePointer::decode(&buf, &ctx()).unwrap();
        assert_eq!(p.location, SharedLocation::Committed);
        assert_eq!(p.oh_addr, 0x4321);
    }

    /// Version 1 has six reserved bytes and a local-heap address before the
    /// object header address, and no type byte at all.
    #[test]
    fn version_one_pointer_skips_the_symbol_table_entry_prefix() {
        let mut buf = vec![1u8, 0u8];
        buf.extend_from_slice(&[0u8; 6]);
        buf.extend_from_slice(&0xdeadu64.to_le_bytes()); // local heap address
        buf.extend_from_slice(&0x9999u64.to_le_bytes());
        let p = SharedMessagePointer::decode(&buf, &ctx()).unwrap();
        assert_eq!(p.location, SharedLocation::Committed);
        assert_eq!(p.oh_addr, 0x9999);
    }

    /// Version 2 predates the committed flag: a non-heap pointer is a
    /// committed datatype whatever the type byte says.
    #[test]
    fn version_two_non_heap_pointer_is_committed() {
        let mut buf = vec![2u8, 0u8];
        buf.extend_from_slice(&0x77u64.to_le_bytes());
        let p = SharedMessagePointer::decode(&buf, &ctx()).unwrap();
        assert_eq!(p.location, SharedLocation::Committed);
        assert_eq!(p.oh_addr, 0x77);
    }

    #[test]
    fn heap_pointer_before_version_three_is_rejected() {
        let mut buf = vec![2u8, 1u8];
        buf.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            SharedMessagePointer::decode(&buf, &ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    #[test]
    fn pointer_rejects_unknown_versions() {
        assert!(matches!(
            SharedMessagePointer::decode(&[0u8, 1u8], &ctx()).unwrap_err(),
            FormatError::InvalidVersion(0)
        ));
        assert!(matches!(
            SharedMessagePointer::decode(&[4u8, 1u8], &ctx()).unwrap_err(),
            FormatError::InvalidVersion(4)
        ));
    }

    #[test]
    fn pointer_rejects_a_truncated_body() {
        assert!(matches!(
            SharedMessagePointer::decode(&[3u8, 1u8, 0, 0], &ctx()).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }
}
