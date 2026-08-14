//! Dense attribute storage.
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

use crate::format::checksum::checksum_metadata;
use crate::format::chunk_index::btree_v2::{
    collect_btree_v2_records, Bt2Header, Bt2Tree, BT2_TYPE_ATTR_NAME,
};
use crate::format::fractal_heap::{
    collect_managed_blocks, read_heap_object, FractalHeapHeader, HeapId, HeapParams,
};
use crate::format::fractal_heap_write::{build_heap, HeapBlock};
use crate::format::messages::attr_info::{AttributeInfoMessage, MAX_CREATION_ORDER_INDEX};
use crate::format::messages::attribute::AttributeEntry;
use crate::format::messages::MSG_FLAG_SHARED;
use crate::format::{BlockReader, FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Length of the fractal-heap ID embedded in a dense-attribute name record
/// (`H5O_FHEAP_ID_LEN`).
const FHEAP_ID_LEN: usize = 8;

/// A name-index record: heap ID, message flags, creation order, name hash.
/// 17 bytes on disk (`H5A__dense_btree2_name_encode`).
const NAME_RECORD_LEN: usize = FHEAP_ID_LEN + 1 + 4 + 4;

/// Node size of the name index (`H5A_NAME_BT2_NODE_SIZE`).
const NAME_BT2_NODE_SIZE: u32 = 512;

/// The hash a name is indexed under (`H5A__dense_insert`).
pub fn name_hash(name: &str) -> u32 {
    checksum_metadata(name.as_bytes())
}

/// Read every attribute an object keeps in dense storage.
///
/// Returns them in name-index (hash) order, the order `H5Aiterate2` walks with
/// `H5_INDEX_NAME`. An `ainfo` describing compact storage yields an empty
/// vector; a record the reader cannot resolve to a heap object is an error,
/// not a silent omission, so a partially-read dense object never masquerades
/// as a complete one. A heap object that resolves but whose payload this crate
/// cannot model is named rather than dropped — see [`AttributeEntry::parse`].
pub fn read_dense_attributes<R: BlockReader>(
    ainfo: &AttributeInfoMessage,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<AttributeEntry>> {
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
        attrs.push(AttributeEntry::parse(&bytes, ctx)?);
    }
    Ok(attrs)
}

/// Dense storage laid out for an object: what its header must say, and what
/// must be written for that to be true.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseAttributeStorage {
    /// The `Attribute Info` message naming the heap and the name index.
    pub ainfo: AttributeInfoMessage,
    /// Heap header, heap blocks, huge objects and both indices' nodes.
    pub blocks: Vec<HeapBlock>,
}

/// Lay `attrs` out as dense storage: a fractal heap holding one encoded
/// attribute message each, plus a v2 B-tree indexing them by name hash.
///
/// `alloc` allocates file space and returns the address; every allocation it
/// hands out is reported back through [`DenseAttributeStorage::blocks`], so a
/// caller that abandons the result can free exactly what it took.
///
/// Mirrors `H5A__dense_create` followed by one `H5A__dense_insert` per
/// attribute, except that the whole set is known up front, so the index is
/// bulk-loaded rather than grown by insertion.
pub fn build_dense_attributes(
    attrs: &[AttributeEntry],
    ctx: &FormatContext,
    alloc: &mut dyn FnMut(u64) -> u64,
) -> FormatResult<DenseAttributeStorage> {
    let objects: Vec<Vec<u8>> = attrs.iter().map(|a| a.encode(ctx)).collect();
    let heap = build_heap(&HeapParams::object_header(), ctx, &objects, alloc)?;

    // `H5A__dense_btree2_name_compare` orders on the hash and breaks ties by
    // strcmp of the name pulled back out of the heap, so a bulk load has to
    // sort the same way or a lookup walking the tree misses records.
    let mut order: Vec<usize> = (0..attrs.len()).collect();
    order.sort_by(|&a, &b| {
        name_hash(attrs[a].name())
            .cmp(&name_hash(attrs[b].name()))
            .then_with(|| attrs[a].name().cmp(attrs[b].name()))
    });

    let mut records = Vec::with_capacity(order.len() * NAME_RECORD_LEN);
    for &i in &order {
        records.extend_from_slice(&heap.ids[i]);
        // Nothing here is a shared (SOHM) message, and creation order is not
        // tracked, so every record carries the "no creation index" sentinel
        // `H5O_MAX_CRT_ORDER_IDX` the library writes in that case.
        records.push(0);
        records.extend_from_slice(&u32::from(MAX_CREATION_ORDER_INDEX).to_le_bytes());
        records.extend_from_slice(&name_hash(attrs[i].name()).to_le_bytes());
    }

    let tree = Bt2Tree::build(
        BT2_TYPE_ATTR_NAME,
        NAME_RECORD_LEN as u16,
        NAME_BT2_NODE_SIZE,
        ctx.sizeof_addr,
        &records,
    );
    let bt2_addr = alloc(tree.header(UNDEF_ADDR).encoded_size(ctx) as u64);
    let node_addrs: Vec<u64> = tree
        .nodes
        .iter()
        .map(|_| alloc(tree.node_size as u64))
        .collect();

    let mut blocks = heap.blocks;
    for (image, &addr) in tree.encode(ctx, &node_addrs).into_iter().zip(&node_addrs) {
        blocks.push(HeapBlock {
            addr,
            len: tree.node_size as u64,
            image,
        });
    }
    let root_addr = node_addrs.last().copied().unwrap_or(UNDEF_ADDR);
    let image = tree.header(root_addr).encode(ctx);
    blocks.push(HeapBlock {
        addr: bt2_addr,
        len: image.len() as u64,
        image,
    });

    Ok(DenseAttributeStorage {
        ainfo: AttributeInfoMessage {
            max_creation_index: None,
            fractal_heap_address: heap.header_addr,
            name_btree_address: bt2_addr,
            creation_order_btree_address: None,
        },
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::messages::attribute::AttributeMessage;
    use crate::format::messages::datatype::DatatypeMessage;

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

    /// A file image the builder's blocks are written into, so the dense
    /// reader can be pointed straight back at what the writer produced.
    struct MemFile {
        bytes: Vec<u8>,
    }

    impl MemFile {
        fn new() -> Self {
            // Leave the first block unused so address 0 never means "unset".
            Self { bytes: vec![0; 16] }
        }
        fn alloc(&mut self, len: u64) -> u64 {
            let addr = self.bytes.len() as u64;
            self.bytes.resize(self.bytes.len() + len as usize, 0);
            addr
        }
    }

    impl BlockReader for MemFile {
        fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>> {
            let start = offset as usize;
            if start > self.bytes.len() {
                return Err(FormatError::BufferTooShort {
                    needed: start,
                    available: self.bytes.len(),
                });
            }
            let end = (start + len).min(self.bytes.len());
            Ok(self.bytes[start..end].to_vec())
        }
    }

    /// Lay `attrs` out, write the result into a fresh image, and read them
    /// back through the dense reader.
    fn round_trip(attrs: &[AttributeEntry]) -> (MemFile, Vec<AttributeEntry>) {
        let mut file = MemFile::new();
        let dense = build_dense_attributes(attrs, &ctx(), &mut |len| file.alloc(len)).unwrap();
        for block in &dense.blocks {
            assert_eq!(block.len as usize, block.image.len(), "block len vs image");
            let at = block.addr as usize;
            file.bytes[at..at + block.image.len()].copy_from_slice(&block.image);
        }
        let read = read_dense_attributes(&dense.ainfo, &ctx(), &mut file).unwrap();
        (file, read)
    }

    fn numeric(name: &str, value: i32) -> AttributeEntry {
        AttributeMessage::scalar_numeric(
            name,
            DatatypeMessage::i32_type(),
            value.to_le_bytes().to_vec(),
        )
        .into()
    }

    #[test]
    fn a_dozen_attributes_round_trip_through_dense_storage() {
        let attrs: Vec<AttributeEntry> = (0..12).map(|i| numeric(&format!("attr{i}"), i)).collect();
        let (_file, read) = round_trip(&attrs);

        assert_eq!(read.len(), attrs.len());
        // The reader returns them in name-index (hash) order, so compare as
        // sets keyed by name.
        for want in &attrs {
            let got = read
                .iter()
                .find(|a| a.name() == want.name())
                .unwrap_or_else(|| panic!("'{}' missing from dense storage", want.name()));
            assert_eq!(got, want);
        }
    }

    #[test]
    fn an_attribute_past_the_managed_size_round_trips_as_a_huge_object() {
        // 25600 i32 elements is the `attr_large` oracle case: 100 KiB of data,
        // far past both the heap's 4 KiB `max_man_size` and the 65535-byte
        // ceiling an object header message can express.
        let data: Vec<u8> = (0..25600i32).flat_map(|v| v.to_le_bytes()).collect();
        let big = AttributeEntry::from(AttributeMessage::array_numeric(
            "big",
            DatatypeMessage::i32_type(),
            &[25600],
            data,
        ));
        assert!(big.encode(&ctx()).len() > 65535);

        let attrs = vec![numeric("small", 7), big];
        let (_file, read) = round_trip(&attrs);

        assert_eq!(read.len(), 2);
        for want in &attrs {
            let got = read.iter().find(|a| a.name() == want.name()).unwrap();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn an_object_with_no_attributes_yields_an_empty_index() {
        let (_file, read) = round_trip(&[]);
        assert!(read.is_empty());
    }

    #[test]
    fn name_records_are_ordered_by_hash() {
        // Enough attributes that the index is more than one leaf, so a
        // misordered bulk load would put a record under the wrong subtree.
        let attrs: Vec<AttributeEntry> = (0..64).map(|i| numeric(&format!("a{i}"), i)).collect();
        let mut file = MemFile::new();
        let dense = build_dense_attributes(&attrs, &ctx(), &mut |len| file.alloc(len)).unwrap();
        for block in &dense.blocks {
            let at = block.addr as usize;
            file.bytes[at..at + block.image.len()].copy_from_slice(&block.image);
        }

        let bt2_buf = file
            .read_block(dense.ainfo.name_btree_address, 256)
            .unwrap();
        let bt2 = Bt2Header::decode(&bt2_buf, &ctx()).unwrap();
        assert!(bt2.depth > 0, "expected a multi-level index, got one leaf");
        let records = collect_btree_v2_records(&bt2, &ctx(), &mut file).unwrap();

        let hashes: Vec<u32> = records
            .chunks_exact(NAME_RECORD_LEN)
            .map(|r| u32::from_le_bytes(r[13..17].try_into().unwrap()))
            .collect();
        assert_eq!(hashes.len(), attrs.len());
        assert!(
            hashes.windows(2).all(|w| w[0] <= w[1]),
            "name index is not hash-ordered: {hashes:?}"
        );
        // Every record carries the "no creation index" sentinel.
        for rec in records.chunks_exact(NAME_RECORD_LEN) {
            assert_eq!(rec[FHEAP_ID_LEN], 0, "no record is shared");
            assert_eq!(
                u32::from_le_bytes(rec[9..13].try_into().unwrap()),
                u32::from(MAX_CREATION_ORDER_INDEX)
            );
        }
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
