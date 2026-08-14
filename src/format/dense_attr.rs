//! Dense attribute storage (read).
//!
//! Once an object accumulates more attributes than the object header's
//! `max_compact` threshold — or one attribute too large to encode as a header
//! message — libhdf5 moves *all* of its attributes out of the object header
//! (`H5Oattribute.c::H5O__attr_create`). The `Attribute Info` message then
//! points at a fractal heap holding each attribute as an encoded attribute
//! message, plus a v2 B-tree indexing them by name hash
//! (`H5Adense.c::H5A__dense_create`).
//!
//! The name index is the authority on which attributes exist: the heap alone
//! cannot be enumerated reliably (an attribute at or above the heap's
//! `max_man_size` is a "huge" object living outside the managed blocks). So
//! this module walks the B-tree, pulls each record's heap ID, and resolves it
//! through the heap.
//!
//! Reference: `H5Adense.c` (`H5A__dense_iterate`), `H5Abtree2.c`
//! (`H5A__dense_btree2_name_encode`).

use crate::format::chunk_index::btree_v2::{
    collect_btree_v2_records, Bt2Header, BT2_TYPE_ATTR_NAME,
};
use crate::format::fractal_heap::{
    collect_managed_blocks, read_heap_object, FractalHeapHeader, HeapId,
};
use crate::format::messages::attr_info::AttributeInfoMessage;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::{BlockReader, FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Length of the fractal-heap ID embedded in a dense-attribute name record
/// (`H5O_FHEAP_ID_LEN`).
const FHEAP_ID_LEN: usize = 8;

/// A name-index record: heap ID, message flags, creation order, name hash.
/// 17 bytes on disk (`H5A__dense_btree2_name_encode`).
const NAME_RECORD_LEN: usize = FHEAP_ID_LEN + 1 + 4 + 4;

/// Object-header message flag marking a shared (SOHM) attribute message.
const MSG_FLAG_SHARED: u8 = 0x02;

/// Read every attribute an object keeps in dense storage.
///
/// Returns them in name-index (hash) order, the order `H5Aiterate2` walks with
/// `H5_INDEX_NAME`. An `ainfo` describing compact storage yields an empty
/// vector; a record the reader cannot resolve is an error, not a silent
/// omission, so a partially-read dense object never masquerades as a complete
/// one.
pub fn read_dense_attributes<R: BlockReader>(
    ainfo: &AttributeInfoMessage,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<AttributeMessage>> {
    if !ainfo.is_dense() {
        return Ok(Vec::new());
    }
    if ainfo.name_btree_address == UNDEF_ADDR {
        return Err(FormatError::InvalidData(
            "dense attribute storage without a name index B-tree".into(),
        ));
    }

    // The heap header's on-disk size depends only on the address/length
    // widths, so a generous prefix read covers it.
    let heap_buf = reader.read_block(ainfo.fractal_heap_address, 512)?;
    let heap = FractalHeapHeader::decode(&heap_buf, ctx)?;
    let blocks = collect_managed_blocks(&heap, ctx, reader)?;

    let bt2_buf = reader.read_block(ainfo.name_btree_address, 256)?;
    let bt2 = Bt2Header::decode(&bt2_buf, ctx)?;
    if bt2.record_type != BT2_TYPE_ATTR_NAME {
        return Err(FormatError::InvalidData(format!(
            "attribute name index has B-tree record type {}, expected {}",
            bt2.record_type, BT2_TYPE_ATTR_NAME
        )));
    }
    if (bt2.record_size as usize) < NAME_RECORD_LEN {
        return Err(FormatError::InvalidData(format!(
            "attribute name index record is {} bytes, expected at least {}",
            bt2.record_size, NAME_RECORD_LEN
        )));
    }

    let records = collect_btree_v2_records(&bt2, ctx, reader)?;
    let rec_size = bt2.record_size as usize;
    let mut attrs = Vec::with_capacity(records.len() / rec_size);
    for rec in records.chunks_exact(rec_size) {
        // A shared attribute message lives in the file's shared-message heap
        // rather than this object's; decoding its heap ID against this heap
        // would read unrelated bytes.
        if rec[FHEAP_ID_LEN] & MSG_FLAG_SHARED != 0 {
            return Err(FormatError::UnsupportedFeature(
                "shared (SOHM) dense attribute".into(),
            ));
        }
        let id = HeapId::parse(&rec[..FHEAP_ID_LEN], &heap, ctx)?;
        let bytes = read_heap_object(&id, &heap, ctx, &blocks, reader)?;
        let (attr, _) = AttributeMessage::decode(&bytes, ctx)?;
        attrs.push(attr);
    }
    Ok(attrs)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SliceReader<'a>(&'a [u8]);

    impl BlockReader for SliceReader<'_> {
        fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>> {
            let start = offset as usize;
            if start > self.0.len() {
                return Err(FormatError::BufferTooShort {
                    needed: start,
                    available: self.0.len(),
                });
            }
            let end = (start + len).min(self.0.len());
            Ok(self.0[start..end].to_vec())
        }
    }

    fn ctx() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    #[test]
    fn compact_ainfo_reads_no_dense_attributes() {
        let ainfo = AttributeInfoMessage::compact();
        let mut reader = SliceReader(&[]);
        assert!(read_dense_attributes(&ainfo, &ctx(), &mut reader)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn dense_ainfo_without_name_index_is_an_error() {
        let ainfo = AttributeInfoMessage {
            max_creation_index: None,
            fractal_heap_address: 512,
            name_btree_address: UNDEF_ADDR,
            creation_order_btree_address: None,
        };
        let mut reader = SliceReader(&[]);
        let err = read_dense_attributes(&ainfo, &ctx(), &mut reader).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }
}
