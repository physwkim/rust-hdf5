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
use crate::format::checksum::{checksum_metadata, jenkins_lookup3};
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

/// Offset of the heap ID inside a heap-shared pointer: past the version and
/// type bytes [`SharedMessagePointer::encode_sohm`] writes.
pub const SOHM_POINTER_HEAP_ID_AT: usize = 2;

/// `H5SM_IN_HEAP`: the record's message body is in the index's fractal heap.
pub const SOHM_IN_HEAP: u8 = 0;
/// `H5SM_IN_OH`: the record's message body is a message of an object header.
pub const SOHM_IN_OH: u8 = 1;

/// Index form stored in an index header's `index_type` byte (`H5SM_index_type_t`).
pub const SOHM_INDEX_LIST: u8 = 0;
/// The B-tree form of the same field.
pub const SOHM_INDEX_BTREE: u8 = 1;

/// v2 B-tree record type of a SOHM index (`H5B2_SOHM_INDEX_ID`).
pub const BT2_TYPE_SOHM_INDEX: u8 = 7;

/// Node size of a SOHM index B-tree (`H5SM_B2_NODE_SIZE`).
pub const SOHM_B2_NODE_SIZE: u32 = 512;

/// Most indexes a file may declare (`H5O_SHMESG_MAX_NINDEXES`).
pub const MAX_SOHM_INDEXES: usize = 8;

/// The hash an index records a message under (`H5SM__write_mesg`): Jenkins
/// lookup3 over the *encoded* message body, seeded with the message type id.
/// Two message classes that happen to encode identically therefore land on
/// different records, which is what lets one index cover several classes.
pub fn message_hash(body: &[u8], msg_type: u8) -> u32 {
    jenkins_lookup3(body, u32::from(msg_type))
}

/// On-disk size of one index record (`H5SM_SOHM_ENTRY_SIZE`): a location byte
/// and a hash, then whichever of the two location-specific bodies is larger.
pub fn record_size(ctx: &FormatContext) -> usize {
    let sa = ctx.sizeof_addr as usize;
    // Heap form: reference count + heap id. Object-header form: reserved byte,
    // message type, message index, header address.
    1 + 4 + (4 + SOHM_HEAP_ID_LEN).max(1 + 1 + 2 + sa)
}

/// One index record: the hash every record carries, and whichever of the two
/// bodies `H5SM_sohm_t` holds (`H5SM__message_encode`, H5SMmessage.c:265-292).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SohmRecord {
    /// [`message_hash`] of the body this record names.
    pub hash: u32,
    /// Where that body is.
    pub location: SohmRecordLocation,
}

/// The two forms a record's body takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SohmRecordLocation {
    /// `H5SM_IN_HEAP`: the body is a heap object of the index's fractal heap.
    InHeap {
        /// Messages referring to this body — pointers plus, when a first copy
        /// was left literal, that copy (`H5SM__write_mesg` opens the count at
        /// 2 when it moves an in-header body to the heap, H5SM.c:1298-1306).
        ref_count: u32,
        /// Fractal-heap ID of the body.
        heap_id: [u8; SOHM_HEAP_ID_LEN],
    },
    /// `H5SM_IN_OH`: the body is still a literal message of the object header
    /// that first wrote it, which carries [`MSG_FLAG_SHAREABLE`] and no
    /// pointer (H5SM.c:1400-1417).
    ///
    /// [`MSG_FLAG_SHAREABLE`]: crate::format::messages::MSG_FLAG_SHAREABLE
    InObjectHeader {
        /// Class of the message that holds the body.
        msg_type: u8,
        /// Its creation index within that header, which
        /// `H5O_msg_get_crt_index` reports as 0 for every class carrying
        /// `H5O_SHARE_IN_OHDR` — only the attribute class has a
        /// `get_crt_index` callback (H5Oattr.c:82).
        index: u16,
        /// Address of the object header holding it.
        oh_addr: u64,
    },
}

impl SohmRecord {
    /// Encode the record (`H5SM__message_encode`).
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let size = record_size(ctx);
        let mut buf = Vec::with_capacity(size);
        match self.location {
            SohmRecordLocation::InHeap { ref_count, heap_id } => {
                buf.push(SOHM_IN_HEAP);
                buf.extend_from_slice(&self.hash.to_le_bytes());
                buf.extend_from_slice(&ref_count.to_le_bytes());
                buf.extend_from_slice(&heap_id);
            }
            SohmRecordLocation::InObjectHeader {
                msg_type,
                index,
                oh_addr,
            } => {
                buf.push(SOHM_IN_OH);
                buf.extend_from_slice(&self.hash.to_le_bytes());
                // A reserved byte libhdf5 writes zero and never reads.
                buf.push(0);
                buf.push(msg_type);
                buf.extend_from_slice(&index.to_le_bytes());
                buf.extend_from_slice(&oh_addr.to_le_bytes()[..ctx.sizeof_addr as usize]);
            }
        }
        buf.resize(size, 0);
        buf
    }
}

/// Space a list index occupies (`H5SM_LIST_SIZE`), which is sized for
/// `list_max` records however few are in use.
pub fn list_size(ctx: &FormatContext, list_max: u16) -> usize {
    4 + record_size(ctx) * list_max as usize + 4
}

/// Encode a list index (`SMLI`, `H5SM__cache_list_serialize`).
///
/// The image covers only the records in use: the checksum follows the last
/// one, and the rest of the [`list_size`] block the index occupies is left
/// untouched. A reader sizes the buffer from `list_max` but checksums exactly
/// this prefix, so trailing bytes are never part of the sum.
pub fn encode_list(records: &[SohmRecord], ctx: &FormatContext) -> Vec<u8> {
    let mut buf = SMLI_SIGNATURE.to_vec();
    for record in records {
        buf.extend_from_slice(&record.encode(ctx));
    }
    let sum = checksum_metadata(&buf);
    buf.extend_from_slice(&sum.to_le_bytes());
    buf
}

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

    /// The pointer a dataset (or attribute) built on a committed datatype
    /// stores in place of the message body.
    ///
    /// `H5O__shared_encode` picks version 2 for `H5O_SHARE_TYPE_COMMITTED` —
    /// version 3 exists for the heap form, which needs a flag byte version 2
    /// has no room for — so a committed pointer libhdf5 wrote and one written
    /// here agree byte for byte.
    pub fn committed(oh_addr: u64) -> Self {
        Self {
            version: 2,
            location: SharedLocation::Committed,
            heap_id: [0u8; SOHM_HEAP_ID_LEN],
            oh_addr,
        }
    }

    /// Encode the committed form — the inverse of [`decode`](Self::decode)
    /// for the one shape this crate writes.
    ///
    /// Only the committed form is produced, so this takes the address rather
    /// than a pointer and cannot fail: nothing here shares through the SOHM
    /// heap, and the version-1 form libhdf5 no longer writes has a symbol
    /// table entry in it that nothing here can fill.
    pub fn encode_committed(oh_addr: u64, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let mut buf = Vec::with_capacity(2 + sa);
        buf.push(2); // H5O_SHARED_VERSION_2
        buf.push(2); // H5O_SHARE_TYPE_COMMITTED
        buf.extend_from_slice(&oh_addr.to_le_bytes()[..sa]);
        buf
    }

    /// The pointer a header stores in place of a message whose body was moved
    /// to the shared-message fractal heap.
    ///
    /// Version 3, where the committed form stays at version 2: only version 3
    /// has the type byte free to mean `H5O_SHARE_TYPE_SOHM`, and its body is
    /// the fixed-width heap ID rather than an address, so it does not follow
    /// the file's address size.
    pub fn encode_sohm(heap_id: [u8; SOHM_HEAP_ID_LEN]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 + SOHM_HEAP_ID_LEN);
        buf.push(3); // H5O_SHARED_VERSION_3
        buf.push(1); // H5O_SHARE_TYPE_SOHM
        buf.extend_from_slice(&heap_id);
        buf
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

    /// Encode the master table (`H5SM__cache_table_serialize`). The index
    /// count is not stored here — the shared-message-table message in the
    /// superblock extension is the only place it is written.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let mut buf = SMTB_SIGNATURE.to_vec();
        for index in &self.indexes {
            buf.push(SM_LIST_VERSION);
            buf.push(index.index_type);
            buf.extend_from_slice(&index.mesg_types.to_le_bytes());
            buf.extend_from_slice(&index.min_mesg_size.to_le_bytes());
            buf.extend_from_slice(&index.list_max.to_le_bytes());
            buf.extend_from_slice(&index.btree_min.to_le_bytes());
            buf.extend_from_slice(&index.num_messages.to_le_bytes());
            buf.extend_from_slice(&index.index_addr.to_le_bytes()[..sa]);
            buf.extend_from_slice(&index.heap_addr.to_le_bytes()[..sa]);
        }
        let sum = checksum_metadata(&buf);
        buf.extend_from_slice(&sum.to_le_bytes());
        buf
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

    /// The bytes h5py's `h5d.create` from a committed TypeID left in the
    /// dataset's datatype message: version 2, `H5O_SHARE_TYPE_COMMITTED`,
    /// then the datatype object header's address.
    #[test]
    fn committed_pointer_encodes_the_bytes_h5o_shared_encode_writes() {
        let buf = SharedMessagePointer::encode_committed(0x320, &ctx());
        let mut want = vec![2u8, 2u8];
        want.extend_from_slice(&0x320u64.to_le_bytes());
        assert_eq!(buf, want);
        assert_eq!(
            SharedMessagePointer::decode(&buf, &ctx()).unwrap(),
            SharedMessagePointer::committed(0x320)
        );
    }

    /// A four-byte address file narrows the pointer to match.
    #[test]
    fn committed_pointer_follows_the_address_width() {
        let ctx4 = FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        };
        let buf = SharedMessagePointer::encode_committed(0x1234, &ctx4);
        assert_eq!(buf, vec![2u8, 2, 0x34, 0x12, 0, 0]);
        assert_eq!(
            SharedMessagePointer::decode(&buf, &ctx4).unwrap().oh_addr,
            0x1234
        );
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

    /// The bytes libhdf5 1.14.6 writes for a dataset sharing `/t` at address
    /// 800, taken verbatim from an h5py-written file.
    #[test]
    fn a_libhdf5_committed_datatype_reference_names_its_object_header() {
        let buf = [0x02, 0x02, 0x20, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let p = SharedMessagePointer::decode(&buf, &ctx()).unwrap();
        assert_eq!(p.location, SharedLocation::Committed);
        assert_eq!(p.oh_addr, 800);
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

    /// The dataspace and datatype messages `sohm_list.h5` shares, with the
    /// hash libhdf5 filed each under. Seeding lookup3 with the message type
    /// id is what the seed is *for*: the same bytes under another class must
    /// not collide with these.
    #[test]
    fn message_hash_matches_the_fixture_records() {
        // A simple dataspace of [8] with max dims, as libhdf5 encoded it.
        let sdspace = [
            1u8, 1, 1, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(message_hash(&sdspace, MSG_DATASPACE), 701521455);
        // H5T_IEEE_F64LE.
        let dtype = [
            0x11u8, 0x20, 0x3f, 0x00, 8, 0, 0, 0, 0, 0, 0x40, 0x00, 0x34, 0x0b, 0x00, 0x34, 0xff,
            0x03, 0x00, 0x00,
        ];
        assert_eq!(message_hash(&dtype, MSG_DATATYPE), 3573483313);
        assert_ne!(
            message_hash(&sdspace, MSG_DATASPACE),
            message_hash(&sdspace, MSG_DATATYPE)
        );
    }

    #[test]
    fn record_is_seventeen_bytes_for_eight_byte_addresses() {
        assert_eq!(record_size(&ctx()), 17);
        assert_eq!(
            record_size(&FormatContext {
                sizeof_addr: 4,
                sizeof_size: 4
            }),
            17
        );
    }

    /// The four records `sohm_list.h5` holds, byte for byte, including the
    /// checksum that follows the last one rather than the end of the block.
    #[test]
    fn list_index_encodes_the_fixture_image() {
        let heaped = |hash, ref_count, heap_id| SohmRecord {
            hash,
            location: SohmRecordLocation::InHeap { ref_count, heap_id },
        };
        let records = [
            heaped(701521455, 5, [0x00, 0x42, 0, 0, 0, 0, 0x18, 0x00]),
            heaped(3573483313, 1, [0x00, 0x16, 0, 0, 0, 0, 0x14, 0x00]),
            heaped(826238635, 1, [0x00, 0x2a, 0, 0, 0, 0, 0x18, 0x00]),
            heaped(2575530442, 4, [0x00, 0x7a, 0, 0, 0, 0, 0x38, 0x00]),
        ];
        let image = encode_list(&records, &ctx());
        let want = concat!(
            "534d4c49",
            "002f5ed029050000000042000000001800",
            "003107ffd4010000000016000000001400",
            "00ab663f3101000000002a000000001800",
            "00ca79839904000000007a000000003800",
            "cbfec07c",
        );
        assert_eq!(hex(&image), want);
        // The block the index occupies is sized for `list_max` records; the
        // image stops after the ones in use.
        assert_eq!(list_size(&ctx(), 50), 858);
        assert!(image.len() < list_size(&ctx(), 50));
    }

    /// The other record form (`H5SM__message_encode`'s `else` branch): a
    /// location byte, the hash, a reserved zero, the message class, a
    /// creation index and the address of the header holding the body.
    #[test]
    fn an_object_header_record_names_the_header_holding_the_body() {
        let record = SohmRecord {
            hash: 701521455,
            location: SohmRecordLocation::InObjectHeader {
                msg_type: MSG_DATASPACE,
                index: 0,
                oh_addr: 0x0349,
            },
        };
        assert_eq!(
            hex(&record.encode(&ctx())),
            concat!(
                "01",               // H5SM_IN_OH
                "2f5ed029",         // hash, little-endian
                "00",               // reserved
                "01",               // message type: dataspace
                "0000",             // creation index
                "4903000000000000", // object header address
            )
        );
        assert_eq!(record.encode(&ctx()).len(), record_size(&ctx()));
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// `sohm_list.h5`'s master table: one list index over dataspace, datatype
    /// and attribute messages.
    #[test]
    fn master_table_encodes_what_decode_reads_back() {
        let table = SohmMasterTable {
            indexes: vec![SohmIndexHeader {
                index_type: SOHM_INDEX_LIST,
                mesg_types: (1 << MSG_DATASPACE) | (1 << MSG_DATATYPE) | (1 << MSG_ATTRIBUTE),
                min_mesg_size: 0,
                list_max: 50,
                btree_min: 40,
                num_messages: 4,
                index_addr: 1125,
                heap_addr: 1983,
            }],
        };
        let image = table.encode(&ctx());
        assert_eq!(image.len(), SohmMasterTable::encoded_size(&ctx(), 1));
        assert_eq!(&image[..4], &SMTB_SIGNATURE);
        assert_eq!(
            u16::from_le_bytes([image[6], image[7]]),
            0x100a,
            "the type mask libhdf5 wrote for DTYPE|SDSPACE|ATTR"
        );
        assert_eq!(SohmMasterTable::decode(&image, &ctx(), 1).unwrap(), table);
    }

    #[test]
    fn heap_pointer_is_a_version_three_message() {
        let id = [0x00, 0x7a, 0, 0, 0, 0, 0x38, 0x00];
        let buf = SharedMessagePointer::encode_sohm(id);
        assert_eq!(hex(&buf), "0301007a000000003800");
        let p = SharedMessagePointer::decode(&buf, &ctx()).unwrap();
        assert_eq!(p.location, SharedLocation::Sohm);
        assert_eq!(p.heap_id, id);
    }

    #[test]
    fn pointer_rejects_a_truncated_body() {
        assert!(matches!(
            SharedMessagePointer::decode(&[3u8, 1u8, 0, 0], &ctx()).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }
}
