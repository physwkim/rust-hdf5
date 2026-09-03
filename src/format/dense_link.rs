//! Dense link storage.
//!
//! Once a group accumulates more links than the Group Info message's
//! `max_compact` threshold — or one link too large to encode as a header
//! message — libhdf5 moves *all* of its links out of the object header
//! (`H5Gobj.c::H5G_obj_insert`). The `Link Info` message then points at a
//! fractal heap holding each link as an encoded link message, plus a v2 B-tree
//! indexing them by name hash (`H5Gdense.c::H5G__dense_create`).
//!
//! The shape mirrors dense attribute storage, but none of the constants carry
//! over: the heap uses `H5G_FHEAP_*` (512-byte first row, a 32-bit heap address
//! space, and so a 7-byte heap ID rather than 8), and the name index is
//! record type 5 with the hash *before* the heap ID — the reverse of the
//! attribute record's field order.
//!
//! Reference: `H5Gdense.c` (`H5G__dense_create`, `H5G__dense_insert`),
//! `H5Gbtree2.c` (`H5G__dense_btree2_name_encode`).

use crate::format::checksum::checksum_metadata;
use crate::format::chunk_index::btree_v2::{
    collect_btree_v2_records, Bt2Header, Bt2Tree, BT2_TYPE_GRP_CORDER, BT2_TYPE_GRP_NAME,
};
use crate::format::creation_order::CreationOrder;
use crate::format::fractal_heap::{
    collect_managed_blocks, read_heap_object, FractalHeapHeader, HeapId, HeapParams,
};
use crate::format::fractal_heap_write::{build_heap, HeapBlock};
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::{BlockReader, FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Length of the fractal-heap ID embedded in a dense-link record
/// (`H5G_DENSE_FHEAP_ID_LEN`).
const FHEAP_ID_LEN: usize = 7;

/// A name-index record: name hash, then heap ID. 11 bytes on disk
/// (`H5G__dense_btree2_name_encode`).
const NAME_RECORD_LEN: usize = 4 + FHEAP_ID_LEN;

/// A creation-order-index record: creation order, then heap ID. 15 bytes on
/// disk (`H5G__dense_btree2_corder_encode`).
const CORDER_RECORD_LEN: usize = 8 + FHEAP_ID_LEN;

/// Node size of either index (`H5G_NAME_BT2_NODE_SIZE`,
/// `H5G_CORDER_BT2_NODE_SIZE`).
const NAME_BT2_NODE_SIZE: u32 = 512;

/// The hash a link name is indexed under (`H5G__dense_insert`).
pub fn name_hash(name: &str) -> u32 {
    checksum_metadata(name.as_bytes())
}

/// Read every link a group keeps in dense storage.
///
/// Returns them in name-index (hash) order, the order `H5Literate2` walks with
/// `H5_INDEX_NAME`. A `linfo` describing compact storage yields an empty
/// vector; a record the reader cannot resolve to a heap object is an error,
/// not a silent omission, so a partially-read group never masquerades as a
/// complete one.
///
/// The index is what is walked, not the heap: a link at or above the heap's
/// `max_man_size` is a "huge" object living outside the managed blocks, and
/// the free space trailing each direct block is not distinguishable from a
/// link message by inspection.
pub fn read_dense_links<R: BlockReader>(
    linfo: &LinkInfoMessage,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<LinkMessage>> {
    if linfo.fractal_heap_address == UNDEF_ADDR {
        return Ok(Vec::new());
    }
    if linfo.name_btree_address == UNDEF_ADDR {
        return Err(FormatError::InvalidData(
            "dense link storage without a name index B-tree".into(),
        ));
    }

    // The heap header's on-disk size depends only on the address/length
    // widths, so a generous prefix read covers it.
    let heap_buf = reader.read_block(linfo.fractal_heap_address, 512)?;
    let heap = FractalHeapHeader::decode(&heap_buf, ctx)?;
    let blocks = collect_managed_blocks(&heap, ctx, reader)?;

    let bt2_buf = reader.read_block(linfo.name_btree_address, 256)?;
    let bt2 = Bt2Header::decode(&bt2_buf, ctx)?;
    if bt2.record_type != BT2_TYPE_GRP_NAME {
        return Err(FormatError::InvalidData(format!(
            "link name index has B-tree record type {}, expected {}",
            bt2.record_type, BT2_TYPE_GRP_NAME
        )));
    }
    if (bt2.record_size as usize) < NAME_RECORD_LEN {
        return Err(FormatError::InvalidData(format!(
            "link name index record is {} bytes, expected at least {}",
            bt2.record_size, NAME_RECORD_LEN
        )));
    }

    let records = collect_btree_v2_records(&bt2, ctx, reader)?;
    let rec_size = bt2.record_size as usize;
    let mut links = Vec::with_capacity(records.len() / rec_size);
    for rec in records.chunks_exact(rec_size) {
        let id = HeapId::parse(&rec[4..4 + FHEAP_ID_LEN], &heap, ctx)?;
        let bytes = read_heap_object(&id, &heap, ctx, &blocks, reader)?;
        links.push(LinkMessage::decode(&bytes, ctx)?.0);
    }
    Ok(links)
}

/// Dense storage laid out for a group: what its header must say, and what must
/// be written for that to be true.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLinkStorage {
    /// The `Link Info` message naming the heap and the name index.
    pub linfo: LinkInfoMessage,
    /// Heap header, heap blocks, huge objects and the index's nodes.
    pub blocks: Vec<HeapBlock>,
}

/// Lay `links` out as dense storage: a fractal heap holding one encoded link
/// message each, plus a v2 B-tree indexing them by name hash.
///
/// `alloc` allocates file space and returns the address; every allocation it
/// hands out is reported back through [`DenseLinkStorage::blocks`], so a caller
/// that abandons the result can free exactly what it took.
///
/// `order` is the group's link creation-order policy: `Tracked` puts the
/// running maximum in the `Link Info` message, and `Indexed` additionally
/// bulk-loads the creation-order B-tree.
///
/// Mirrors `H5G__dense_create` followed by one `H5G__dense_insert` per link,
/// except that the whole set is known up front, so the index is bulk-loaded
/// rather than grown by insertion.
pub fn build_dense_links(
    links: &[LinkMessage],
    ctx: &FormatContext,
    order: CreationOrder,
    alloc: &mut dyn FnMut(u64) -> u64,
) -> FormatResult<DenseLinkStorage> {
    let objects: Vec<Vec<u8>> = links.iter().map(|l| l.encode(ctx)).collect();
    let heap = build_heap(&HeapParams::group_links(), ctx, &objects, alloc)?;

    // `H5G__dense_btree2_name_compare` orders on the hash and breaks ties by
    // strcmp of the name pulled back out of the heap, so a bulk load has to
    // sort the same way or a lookup walking the tree misses records.
    let mut by_name: Vec<usize> = (0..links.len()).collect();
    by_name.sort_by(|&a, &b| {
        name_hash(&links[a].name)
            .cmp(&name_hash(&links[b].name))
            .then_with(|| links[a].name.cmp(&links[b].name))
    });

    let mut records = Vec::with_capacity(by_name.len() * NAME_RECORD_LEN);
    for &i in &by_name {
        records.extend_from_slice(&name_hash(&links[i].name).to_le_bytes());
        records.extend_from_slice(&heap.ids[i]);
    }

    let mut blocks = heap.blocks;
    let bt2_addr = build_index(
        BT2_TYPE_GRP_NAME,
        NAME_RECORD_LEN as u16,
        &records,
        ctx,
        alloc,
        &mut blocks,
    );

    // Tracking is what stamps a creation order onto each link message, so a
    // tracked group whose links carry none is a caller bug, not a file the
    // index could be built without.
    let corders: Option<Vec<i64>> = order
        .is_tracked()
        .then(|| links.iter().map(|l| l.creation_order).collect())
        .flatten();
    if order.is_tracked() && corders.is_none() {
        return Err(FormatError::InvalidData(
            "a group tracking link creation order has a link with no creation order".into(),
        ));
    }

    let corder_bt2_addr = order.is_indexed().then(|| {
        let corders = corders.as_ref().expect("indexed implies tracked");
        let mut by_corder: Vec<usize> = (0..links.len()).collect();
        by_corder.sort_by_key(|&i| corders[i]);
        let mut records = Vec::with_capacity(by_corder.len() * CORDER_RECORD_LEN);
        for &i in &by_corder {
            records.extend_from_slice(&corders[i].to_le_bytes());
            records.extend_from_slice(&heap.ids[i]);
        }
        build_index(
            BT2_TYPE_GRP_CORDER,
            CORDER_RECORD_LEN as u16,
            &records,
            ctx,
            alloc,
            &mut blocks,
        )
    });

    Ok(DenseLinkStorage {
        linfo: LinkInfoMessage {
            // `H5G_obj_insert` post-increments, so after n links the running
            // maximum is n.
            max_creation_order: corders.map(|c| c.len() as u64),
            fractal_heap_address: heap.header_addr,
            name_btree_address: bt2_addr,
            creation_order_btree_address: corder_bt2_addr,
        },
        blocks,
    })
}

/// Bulk-load one v2 B-tree, allocate its header and nodes, and append their
/// images to `blocks`. Returns the header address.
fn build_index(
    record_type: u8,
    record_size: u16,
    records: &[u8],
    ctx: &FormatContext,
    alloc: &mut dyn FnMut(u64) -> u64,
    blocks: &mut Vec<HeapBlock>,
) -> u64 {
    let tree = Bt2Tree::build(
        record_type,
        record_size,
        NAME_BT2_NODE_SIZE,
        ctx.sizeof_addr,
        records,
    );
    let bt2_addr = alloc(tree.header(UNDEF_ADDR).encoded_size(ctx) as u64);
    let node_addrs: Vec<u64> = tree
        .nodes
        .iter()
        .map(|_| alloc(tree.node_size as u64))
        .collect();
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
    bt2_addr
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

    /// A file image the builder's blocks are written into, so the dense reader
    /// can be pointed straight back at what the writer produced.
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

    /// Lay `links` out untracked, write the result into a fresh image, and
    /// read them back through the dense reader.
    fn round_trip(links: &[LinkMessage]) -> (MemFile, DenseLinkStorage, Vec<LinkMessage>) {
        round_trip_ordered(links, CreationOrder::Untracked)
    }

    /// [`round_trip`] under an explicit creation-order policy.
    fn round_trip_ordered(
        links: &[LinkMessage],
        order: CreationOrder,
    ) -> (MemFile, DenseLinkStorage, Vec<LinkMessage>) {
        let mut file = MemFile::new();
        let dense = build_dense_links(links, &ctx(), order, &mut |len| file.alloc(len)).unwrap();
        for block in &dense.blocks {
            assert_eq!(block.len as usize, block.image.len(), "block len vs image");
            let at = block.addr as usize;
            file.bytes[at..at + block.image.len()].copy_from_slice(&block.image);
        }
        let read = read_dense_links(&dense.linfo, &ctx(), &mut file).unwrap();
        (file, dense, read)
    }

    #[test]
    fn compact_linfo_reads_no_dense_links() {
        let mut file = MemFile::new();
        assert!(
            read_dense_links(&LinkInfoMessage::compact(), &ctx(), &mut file)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_dozen_links_round_trip_through_dense_storage() {
        let links: Vec<LinkMessage> = (0..12)
            .map(|i| LinkMessage::hard(&format!("d{i:02}"), 0x400 + i as u64 * 8))
            .collect();
        let (_file, _dense, read) = round_trip(&links);

        assert_eq!(read.len(), links.len());
        // The reader returns them in name-index (hash) order, so compare as
        // sets keyed by name.
        for want in &links {
            let got = read
                .iter()
                .find(|l| l.name == want.name)
                .unwrap_or_else(|| panic!("'{}' missing from dense storage", want.name));
            assert_eq!(got, want);
        }
    }

    #[test]
    fn a_soft_link_round_trips_beside_hard_ones() {
        let links = vec![
            LinkMessage::hard("orig", 0x800),
            LinkMessage::soft("alias", "/orig"),
        ];
        let (_file, _dense, read) = round_trip(&links);
        assert_eq!(read.len(), 2);
        for want in &links {
            assert_eq!(read.iter().find(|l| l.name == want.name).unwrap(), want);
        }
    }

    #[test]
    fn a_group_with_no_links_yields_an_empty_index() {
        let (_file, _dense, read) = round_trip(&[]);
        assert!(read.is_empty());
    }

    /// The heap parameters are the group ones, not the attribute ones: a
    /// 7-byte heap ID over a 32-bit address space, first row 512 bytes.
    #[test]
    fn the_heap_uses_the_group_parameters() {
        let links: Vec<LinkMessage> = (0..12)
            .map(|i| LinkMessage::hard(&format!("d{i:02}"), 0x400 + i as u64 * 8))
            .collect();
        let (mut file, dense, _read) = round_trip(&links);
        let heap_buf = file
            .read_block(dense.linfo.fractal_heap_address, 512)
            .unwrap();
        let heap = FractalHeapHeader::decode(&heap_buf, &ctx()).unwrap();
        assert_eq!(heap.id_len, 7);
        assert_eq!(heap.heap_off_size, 4);
        assert_eq!(heap.heap_len_size, 2);
        assert_eq!(heap.start_block_size, 512);
        assert_eq!(heap.man_nobjs, 12);
    }

    #[test]
    fn name_records_are_ordered_by_hash() {
        // Enough links that the index is more than one leaf, so a misordered
        // bulk load would put a record under the wrong subtree.
        let links: Vec<LinkMessage> = (0..128)
            .map(|i| LinkMessage::hard(&format!("d{i:03}"), 0x400 + i as u64 * 8))
            .collect();
        let (mut file, dense, read) = round_trip(&links);
        assert_eq!(read.len(), links.len());

        let bt2_buf = file
            .read_block(dense.linfo.name_btree_address, 256)
            .unwrap();
        let bt2 = Bt2Header::decode(&bt2_buf, &ctx()).unwrap();
        assert_eq!(bt2.record_type, BT2_TYPE_GRP_NAME);
        assert_eq!(bt2.record_size as usize, NAME_RECORD_LEN);
        assert!(bt2.depth > 0, "expected a multi-level index, got one leaf");

        let records = collect_btree_v2_records(&bt2, &ctx(), &mut file).unwrap();
        let hashes: Vec<u32> = records
            .as_chunks::<NAME_RECORD_LEN>()
            .0
            .iter()
            .map(|r| u32::from_le_bytes(r[0..4].try_into().unwrap()))
            .collect();
        assert_eq!(hashes.len(), links.len());
        assert!(
            hashes.windows(2).all(|w| w[0] <= w[1]),
            "name index is not hash-ordered: {hashes:?}"
        );
    }

    /// Tracking on: the corder index is a second v2 B-tree of type 6, its
    /// records are ordered by creation order (not by name hash), and the
    /// `Link Info` message announces both the index and the post-incremented
    /// maximum.
    #[test]
    fn a_tracked_group_gets_a_creation_order_index() {
        // Names deliberately reverse the creation order, so an index built
        // from the name ordering would show up here.
        let links: Vec<LinkMessage> = (0..12u32)
            .map(|i| {
                LinkMessage::hard(&format!("d{:02}", 11 - i), 0x400 + i as u64 * 8)
                    .with_creation_order(i as i64)
            })
            .collect();
        let (mut file, dense, read) = round_trip_ordered(&links, CreationOrder::Indexed);
        assert_eq!(read.len(), links.len());
        assert_eq!(dense.linfo.max_creation_order, Some(12));

        let addr = dense
            .linfo
            .creation_order_btree_address
            .expect("tracked links must carry a creation-order index");
        let bt2 = Bt2Header::decode(&file.read_block(addr, 256).unwrap(), &ctx()).unwrap();
        assert_eq!(bt2.record_type, BT2_TYPE_GRP_CORDER);
        assert_eq!(bt2.record_size as usize, CORDER_RECORD_LEN);

        let records = collect_btree_v2_records(&bt2, &ctx(), &mut file).unwrap();
        let corders: Vec<i64> = records
            .as_chunks::<CORDER_RECORD_LEN>()
            .0
            .iter()
            .map(|r| i64::from_le_bytes(r[0..8].try_into().unwrap()))
            .collect();
        assert_eq!(corders, (0..12i64).collect::<Vec<_>>());
    }

    /// Tracking off: no second index, no maximum.
    #[test]
    fn an_untracked_group_gets_no_creation_order_index() {
        let links: Vec<LinkMessage> = (0..12)
            .map(|i| LinkMessage::hard(&format!("d{i:02}"), 0x400 + i as u64 * 8))
            .collect();
        let (_file, dense, _read) = round_trip(&links);
        assert_eq!(dense.linfo.creation_order_btree_address, None);
        assert_eq!(dense.linfo.max_creation_order, None);
    }

    /// `H5Pset_link_creation_order(H5P_CRT_ORDER_TRACKED)` without `INDEXED`
    /// is a state libhdf5 accepts: the running maximum is recorded and every
    /// link keeps its creation order, but no second B-tree is built.
    #[test]
    fn a_tracked_but_unindexed_group_records_the_maximum_and_no_index() {
        let links: Vec<LinkMessage> = (0..12u32)
            .map(|i| {
                LinkMessage::hard(&format!("d{i:02}"), 0x400 + i as u64 * 8)
                    .with_creation_order(i as i64)
            })
            .collect();
        let (_file, dense, read) = round_trip_ordered(&links, CreationOrder::Tracked);
        assert_eq!(read.len(), links.len());
        assert_eq!(dense.linfo.max_creation_order, Some(12));
        assert_eq!(dense.linfo.creation_order_btree_address, None);
    }

    /// A group that declares tracking but hands over links with no creation
    /// order is a caller bug; building the storage anyway would write an
    /// index whose records claim an order the heap objects do not carry.
    #[test]
    fn tracking_links_that_carry_no_creation_order_is_refused() {
        let links: Vec<LinkMessage> = (0..12)
            .map(|i| LinkMessage::hard(&format!("d{i:02}"), 0x400 + i as u64 * 8))
            .collect();
        let mut file = MemFile::new();
        let err = build_dense_links(&links, &ctx(), CreationOrder::Tracked, &mut |len| {
            file.alloc(len)
        })
        .unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)), "{err:?}");
    }

    #[test]
    fn dense_linfo_without_name_index_is_an_error() {
        let linfo = LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: 512,
            name_btree_address: UNDEF_ADDR,
            creation_order_btree_address: None,
        };
        let mut file = MemFile::new();
        let err = read_dense_links(&linfo, &ctx(), &mut file).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }
}
