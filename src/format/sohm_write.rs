//! Laying out a file's shared object header messages.
//!
//! The read side ([`crate::format::sohm`]) resolves a pointer; this is what
//! produces one. Given the message bodies a file wants to share, grouped by
//! the index that covers them, it lays out per index a fractal heap holding
//! the bodies and either a list or a v2 B-tree recording them, then the master
//! table naming both. The caller writes the returned blocks and puts the
//! table's address in the superblock extension.
//!
//! Reference: `H5SM.c` (`H5SM_init`, `H5SM__create_index`, `H5SM__write_mesg`),
//! `H5SMcache.c` (the table and list images).
//!
//! # Where this departs from libhdf5
//!
//! `H5SM__write_mesg` has two ways to record a message. A class carrying
//! `H5O_SHARE_IN_OHDR` — datatype, dataspace, fill value, filter pipeline —
//! leaves the *first* copy literal in the header that wrote it and files an
//! object-header record for it; only when a second object wants the same body
//! does the body move to the heap. Every other shareable class, attributes in
//! particular, goes to the heap on first use. This module takes the second
//! path for all of them, which is the branch libhdf5 itself takes whenever the
//! owning header is not open (`open_oh == NULL`). Every record is therefore
//! `H5SM_IN_HEAP`, and a body used once is a heap object with a reference
//! count of one — the shape the reference files already show for a dataspace
//! that only one attribute uses.

use std::collections::HashMap;

use crate::format::chunk_index::btree_v2::build_index;
use crate::format::fractal_heap::HeapParams;
use crate::format::fractal_heap_write::{plan_heap, HeapBlock};
use crate::format::sohm::{
    encode_list, list_size, message_hash, record_size, SohmIndexHeader, SohmMasterTable,
    SohmRecord, BT2_TYPE_SOHM_INDEX, SOHM_B2_NODE_SIZE, SOHM_HEAP_ID_LEN, SOHM_INDEX_BTREE,
    SOHM_INDEX_LIST,
};
use crate::format::{FormatContext, FormatError, FormatResult};

/// One index as the file was created with it (`H5P_shared_mesg_index_t` plus
/// the phase-change pair, which libhdf5 keeps file-wide and copies into every
/// index header).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SohmIndexSpec {
    /// Bit mask of the message types this index covers; see
    /// [`crate::format::sohm::type_flag`].
    pub mesg_types: u16,
    /// Smallest message the index will take.
    pub min_mesg_size: u32,
    /// Message count above which the index is a B-tree rather than a list.
    pub list_max: u16,
    /// Message count below which a B-tree index reverts to a list.
    pub btree_min: u16,
}

/// A message body as an index identifies it: the type it belongs to and the
/// bytes, which together seed the record's hash.
pub type SharedKey = (u8, Vec<u8>);

/// One place a message body holds another shared message's heap ID.
///
/// The attribute message is the one class that does this: `H5A__create`
/// shares the attribute's datatype and dataspace before `H5O__attr_create`
/// shares the attribute (H5Aint.c:375-377), so the body that reaches the heap
/// carries their heap IDs and says so in its own flags byte. Those IDs are
/// zero in [`SharedMessage::body`] — the heap has not been laid out when the
/// body is offered — and [`build_shared_messages`] fills them in.
///
/// The target is named by its own key rather than by position: an index picks
/// messages by class, so an attribute's datatype can belong to a different
/// index than the attribute, and a reader resolves the pointer through the
/// master table by class as well (`H5SM_get_fheap_addr`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NestedShare {
    /// Offset in the body of the eight-byte heap ID, i.e. two bytes past the
    /// start of the `H5O_shared_t` that holds it.
    pub heap_id_at: usize,
    /// The message this pointer names. It holds no nesting of its own —
    /// datatype and dataspace messages have nothing to nest.
    pub target: SharedKey,
}

/// One message body an index holds, and how many object header messages were
/// replaced by a pointer to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessage {
    /// The message type the body belongs to; it seeds the record's hash.
    pub msg_type: u8,
    /// The encoded message body, exactly as an unshared header would hold it,
    /// except that any heap ID [`nested`](Self::nested) names is still zero.
    pub body: Vec<u8>,
    /// Where this body points at other shared messages; empty for every class
    /// but the attribute.
    pub nested: Vec<NestedShare>,
    /// References to this body (`H5SM_sohm_t::ref_count`).
    pub ref_count: u32,
}

/// One index and the messages it was asked to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SohmIndexContent {
    /// The index's creation properties.
    pub spec: SohmIndexSpec,
    /// Its messages, in the order the caller wants heap IDs back in.
    pub messages: Vec<SharedMessage>,
}

/// A laid-out shared-message table: everything to write, and how to point at
/// what was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltSharedMessages {
    /// Address of the master table (`SMTB`), for the superblock extension's
    /// shared-message-table message.
    pub table_addr: u64,
    /// Heaps, indexes and the table, in write order.
    pub blocks: Vec<HeapBlock>,
    /// The heap ID of every message, keyed by `(msg_type, body)` — the pair
    /// that identifies a record, since the hash is seeded with the type.
    pub heap_ids: HashMap<(u8, Vec<u8>), [u8; SOHM_HEAP_ID_LEN]>,
}

/// Lay out every index of a file's shared-message table.
///
/// `alloc` allocates file space and returns the address; every allocation it
/// hands out comes back in [`BuiltSharedMessages::blocks`], so a caller that
/// abandons the result can free exactly what it took.
pub fn build_shared_messages(
    indexes: &[SohmIndexContent],
    ctx: &FormatContext,
    alloc: &mut dyn FnMut(u64) -> u64,
) -> FormatResult<BuiltSharedMessages> {
    let mut blocks = Vec::new();
    let mut heap_ids: HashMap<SharedKey, [u8; SOHM_HEAP_ID_LEN]> = HashMap::new();
    let mut headers = Vec::with_capacity(indexes.len());

    // Every heap is placed before any of them is written: an attribute body in
    // one index names its datatype and dataspace by heap ID, and an index
    // takes messages by class, so the two may belong to a different index than
    // the attribute. Only once every index is placed is every heap ID a body
    // could name known.
    let mut plans = Vec::with_capacity(indexes.len());
    for index in indexes {
        let lengths: Vec<usize> = index.messages.iter().map(|m| m.body.len()).collect();
        plans.push(plan_heap(
            &HeapParams::object_header(),
            ctx,
            &lengths,
            alloc,
        )?);
    }

    // What a nested pointer may name, keyed the way the message was collected.
    // A body that is itself pointed at nests nothing — datatype and dataspace
    // messages have nothing to nest — so for those two keys the collected body
    // and the stored one are the same bytes.
    let mut collected: HashMap<SharedKey, [u8; SOHM_HEAP_ID_LEN]> = HashMap::new();
    for (index, plan) in indexes.iter().zip(&plans) {
        for (message, id) in index.messages.iter().zip(plan.ids()) {
            let heap_id: [u8; SOHM_HEAP_ID_LEN] = id.as_slice().try_into().map_err(|_| {
                FormatError::InvalidData(format!(
                    "shared-message heap returned a {}-byte id, expected {SOHM_HEAP_ID_LEN}",
                    id.len()
                ))
            })?;
            collected.insert((message.msg_type, message.body.clone()), heap_id);
        }
    }

    for (index, plan) in indexes.iter().zip(plans) {
        let num_messages = u16::try_from(index.messages.len()).map_err(|_| {
            FormatError::InvalidData(format!(
                "shared-message index holds {} messages, more than the count field takes",
                index.messages.len()
            ))
        })?;

        let mut bodies = Vec::with_capacity(index.messages.len());
        for message in &index.messages {
            bodies.push(resolve_nested(message, &collected)?);
        }
        let heap_addr = plan.header_addr();
        let ids = plan.ids().to_vec();
        blocks.extend(plan.finish(&bodies)?.blocks);

        let mut records = Vec::with_capacity(index.messages.len());
        for ((message, body), id) in index.messages.iter().zip(bodies).zip(ids) {
            let heap_id: [u8; SOHM_HEAP_ID_LEN] = id
                .as_slice()
                .try_into()
                .expect("the plan's ids were checked above");
            // Keyed by what a header will hold: for a nesting body that is
            // this resolved one, not the one the collect pass offered.
            heap_ids.insert((message.msg_type, body.clone()), heap_id);
            records.push((
                SohmRecord {
                    hash: message_hash(&body, message.msg_type),
                    ref_count: message.ref_count,
                    heap_id,
                },
                body,
            ));
        }

        let (index_type, index_addr) = if is_btree(&index.spec, num_messages) {
            (
                SOHM_INDEX_BTREE,
                build_btree(&mut records, ctx, alloc, &mut blocks),
            )
        } else {
            (
                SOHM_INDEX_LIST,
                build_list(&records, &index.spec, ctx, alloc, &mut blocks),
            )
        };

        headers.push(SohmIndexHeader {
            index_type,
            mesg_types: index.spec.mesg_types,
            min_mesg_size: index.spec.min_mesg_size,
            list_max: index.spec.list_max,
            btree_min: index.spec.btree_min,
            num_messages,
            index_addr,
            heap_addr,
        });
    }

    let table = SohmMasterTable { indexes: headers };
    let nindexes = u8::try_from(indexes.len()).map_err(|_| {
        FormatError::InvalidData(format!("{} shared-message indexes", indexes.len()))
    })?;
    let table_addr = alloc(SohmMasterTable::encoded_size(ctx, nindexes) as u64);
    let image = table.encode(ctx);
    blocks.push(HeapBlock {
        addr: table_addr,
        len: image.len() as u64,
        image,
    });

    Ok(BuiltSharedMessages {
        table_addr,
        blocks,
        heap_ids,
    })
}

/// Which form `H5SM__create_index` gives an index holding `num_messages`.
///
/// A fresh index starts as a list whenever `list_max` leaves room for one and
/// becomes a B-tree the moment an insert takes it past that count
/// (`H5SM__write_mesg`); `list_max == 0` means it was a B-tree from the first
/// insert. Nothing converts a B-tree back on the way up, so `btree_min` — the
/// count a *deletion* would convert below — has no say here.
fn is_btree(spec: &SohmIndexSpec, num_messages: u16) -> bool {
    spec.list_max == 0 || num_messages > spec.list_max
}

/// The body a header holds for `message`: its collected bytes with every heap
/// ID it nests filled in from `placed`.
///
/// A body that nests nothing is returned unchanged, which is every class but
/// the attribute.
fn resolve_nested(
    message: &SharedMessage,
    placed: &HashMap<SharedKey, [u8; SOHM_HEAP_ID_LEN]>,
) -> FormatResult<Vec<u8>> {
    if message.nested.is_empty() {
        return Ok(message.body.clone());
    }
    let mut body = message.body.clone();
    for nested in &message.nested {
        let Some(heap_id) = placed.get(&nested.target) else {
            return Err(FormatError::InvalidData(
                "a shared message points at a body no index holds".into(),
            ));
        };
        let at = nested.heap_id_at;
        if body.len() < at + SOHM_HEAP_ID_LEN {
            return Err(FormatError::InvalidData(
                "a shared message's nested pointer runs past the body holding it".into(),
            ));
        }
        body[at..at + SOHM_HEAP_ID_LEN].copy_from_slice(heap_id);
    }
    Ok(body)
}

/// Lay out a list index and return its address.
fn build_list(
    records: &[(SohmRecord, Vec<u8>)],
    spec: &SohmIndexSpec,
    ctx: &FormatContext,
    alloc: &mut dyn FnMut(u64) -> u64,
    blocks: &mut Vec<HeapBlock>,
) -> u64 {
    // The block is sized for `list_max` records however few are in use; the
    // image covers only the ones written, and the rest stays as allocated.
    let len = list_size(ctx, spec.list_max) as u64;
    let addr = alloc(len);
    let entries: Vec<SohmRecord> = records.iter().map(|(r, _)| *r).collect();
    blocks.push(HeapBlock {
        addr,
        len,
        image: encode_list(&entries, ctx),
    });
    addr
}

/// Bulk-load a B-tree index and return its header address.
fn build_btree(
    records: &mut [(SohmRecord, Vec<u8>)],
    ctx: &FormatContext,
    alloc: &mut dyn FnMut(u64) -> u64,
    blocks: &mut Vec<HeapBlock>,
) -> u64 {
    // `H5SM__message_compare` orders on the hash and breaks ties on the
    // message body itself, so a bulk load has to sort the same way or a
    // lookup walking the tree misses records.
    records.sort_by(|a, b| a.0.hash.cmp(&b.0.hash).then_with(|| a.1.cmp(&b.1)));
    let mut image = Vec::with_capacity(records.len() * record_size(ctx));
    for (record, _) in records.iter() {
        image.extend_from_slice(&record.encode(ctx));
    }
    let (addr, nodes) = build_index(
        BT2_TYPE_SOHM_INDEX,
        record_size(ctx) as u16,
        SOHM_B2_NODE_SIZE,
        &image,
        ctx,
        alloc,
    );
    blocks.extend(nodes.into_iter().map(|(addr, image)| HeapBlock {
        addr,
        len: image.len() as u64,
        image,
    }));
    addr
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::chunk_index::btree_v2::{collect_btree_v2_records, Bt2Header};
    use crate::format::fractal_heap::{
        collect_managed_blocks, read_heap_object, FractalHeapHeader, HeapId,
    };
    use crate::format::messages::{MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE};
    use crate::format::sohm::{SharedLocation, SharedMessagePointer, SOHM_IN_HEAP};
    use crate::format::{BlockReader, UNDEF_ADDR};

    /// A file image the layout's blocks are written into, so the SOHM reader
    /// can be pointed straight back at what was produced.
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

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    fn spec(list_max: u16) -> SohmIndexSpec {
        SohmIndexSpec {
            mesg_types: (1 << MSG_DATASPACE) | (1 << MSG_DATATYPE) | (1 << MSG_ATTRIBUTE),
            min_mesg_size: 0,
            list_max,
            btree_min: 40,
        }
    }

    fn message(msg_type: u8, seed: u8, len: usize, ref_count: u32) -> SharedMessage {
        SharedMessage {
            msg_type,
            body: (0..len).map(|i| seed.wrapping_add(i as u8)).collect(),
            nested: Vec::new(),
            ref_count,
        }
    }

    /// Lay `indexes` out into a fresh image and hand back the image plus the
    /// result, so a test can read the structures the way libhdf5 would.
    fn lay_out(indexes: &[SohmIndexContent]) -> (MemFile, BuiltSharedMessages) {
        let mut file = MemFile::new();
        let built = build_shared_messages(indexes, &ctx(), &mut |len| file.alloc(len)).unwrap();
        for block in &built.blocks {
            assert!(
                block.image.len() as u64 <= block.len,
                "block image overruns its allocation"
            );
            let at = block.addr as usize;
            file.bytes[at..at + block.image.len()].copy_from_slice(&block.image);
        }
        (file, built)
    }

    /// Read one index's records back through the decoders the reader uses.
    fn read_records(file: &mut MemFile, header: &SohmIndexHeader) -> Vec<SohmRecord> {
        let size = record_size(&ctx());
        let raw = if header.index_type == SOHM_INDEX_LIST {
            let buf = file
                .read_block(header.index_addr, 4 + size * header.num_messages as usize)
                .unwrap();
            assert_eq!(&buf[..4], b"SMLI");
            buf[4..].to_vec()
        } else {
            let bt2 = Bt2Header::decode(&file.read_block(header.index_addr, 256).unwrap(), &ctx())
                .unwrap();
            assert_eq!(bt2.record_type, BT2_TYPE_SOHM_INDEX);
            assert_eq!(bt2.node_size, SOHM_B2_NODE_SIZE);
            assert_eq!(bt2.record_size as usize, size);
            collect_btree_v2_records(&bt2, &ctx(), file).unwrap()
        };
        raw.chunks_exact(size)
            .map(|r| {
                assert_eq!(r[0], SOHM_IN_HEAP);
                SohmRecord {
                    hash: u32::from_le_bytes(r[1..5].try_into().unwrap()),
                    ref_count: u32::from_le_bytes(r[5..9].try_into().unwrap()),
                    heap_id: r[9..17].try_into().unwrap(),
                }
            })
            .collect()
    }

    /// Pull a message body back out of an index's heap by its record.
    fn read_body(file: &mut MemFile, heap_addr: u64, record: &SohmRecord) -> Vec<u8> {
        let heap =
            FractalHeapHeader::decode(&file.read_block(heap_addr, 512).unwrap(), &ctx()).unwrap();
        let blocks = collect_managed_blocks(&heap, &ctx(), file).unwrap();
        let id = HeapId::parse(&record.heap_id, &heap, &ctx()).unwrap();
        read_heap_object(&id, &heap, &ctx(), &blocks, file).unwrap()
    }

    /// Four messages under a list index: the records name heap objects that
    /// still hold the bodies that went in, with the reference counts asked for.
    #[test]
    fn a_list_index_round_trips_every_body() {
        let messages = vec![
            message(MSG_DATASPACE, 1, 24, 5),
            message(MSG_DATATYPE, 40, 20, 1),
            message(MSG_DATASPACE, 90, 24, 1),
            message(MSG_ATTRIBUTE, 7, 56, 4),
        ];
        let (mut file, built) = lay_out(&[SohmIndexContent {
            spec: spec(50),
            messages: messages.clone(),
        }]);

        let table = SohmMasterTable::decode(
            &file
                .read_block(built.table_addr, SohmMasterTable::encoded_size(&ctx(), 1))
                .unwrap(),
            &ctx(),
            1,
        )
        .unwrap();
        let header = &table.indexes[0];
        assert_eq!(header.index_type, SOHM_INDEX_LIST);
        assert_eq!(header.num_messages, 4);
        assert_ne!(header.index_addr, UNDEF_ADDR);

        let records = read_records(&mut file, header);
        assert_eq!(records.len(), 4);
        for (message, record) in messages.iter().zip(&records) {
            assert_eq!(record.hash, message_hash(&message.body, message.msg_type));
            assert_eq!(record.ref_count, message.ref_count);
            assert_eq!(read_body(&mut file, header.heap_addr, record), message.body);
        }

        // The pointer the caller substitutes resolves to the same heap ID.
        for message in &messages {
            let id = built.heap_ids[&(message.msg_type, message.body.clone())];
            let pointer =
                SharedMessagePointer::decode(&SharedMessagePointer::encode_sohm(id), &ctx())
                    .unwrap();
            assert_eq!(pointer.location, SharedLocation::Sohm);
            assert_eq!(pointer.heap_id, id);
        }
    }

    /// `H5Pset_shared_mesg_phase_change(fcpl, 0, 0)` — the `sohm_btree`
    /// fixture's setting — makes the index a B-tree from the first insert.
    #[test]
    fn a_zero_list_max_index_is_a_btree() {
        let messages = vec![
            message(MSG_DATASPACE, 1, 24, 5),
            message(MSG_DATATYPE, 40, 20, 1),
            message(MSG_ATTRIBUTE, 7, 56, 4),
        ];
        let (mut file, built) = lay_out(&[SohmIndexContent {
            spec: SohmIndexSpec {
                list_max: 0,
                btree_min: 0,
                ..spec(0)
            },
            messages: messages.clone(),
        }]);
        let table = SohmMasterTable::decode(
            &file
                .read_block(built.table_addr, SohmMasterTable::encoded_size(&ctx(), 1))
                .unwrap(),
            &ctx(),
            1,
        )
        .unwrap();
        let header = &table.indexes[0];
        assert_eq!(header.index_type, SOHM_INDEX_BTREE);

        let records = read_records(&mut file, header);
        assert_eq!(records.len(), 3);
        // `H5SM__message_compare` walks the tree on the hash, so the load has
        // to be in that order.
        assert!(
            records.windows(2).all(|w| w[0].hash <= w[1].hash),
            "records are not hash-ordered: {records:?}"
        );
        for message in &messages {
            let hash = message_hash(&message.body, message.msg_type);
            let record = records.iter().find(|r| r.hash == hash).unwrap();
            assert_eq!(record.ref_count, message.ref_count);
            assert_eq!(read_body(&mut file, header.heap_addr, record), message.body);
        }
    }

    /// Past `list_max` the index libhdf5 would have converted is written as a
    /// B-tree outright, and one deep enough to have interior nodes still reads
    /// back whole.
    #[test]
    fn an_index_past_its_list_maximum_is_a_btree() {
        let messages: Vec<SharedMessage> = (0..200u32)
            .map(|i| SharedMessage {
                msg_type: MSG_DATASPACE,
                body: i.to_le_bytes().repeat(6),
                nested: Vec::new(),
                ref_count: i + 1,
            })
            .collect();
        let (mut file, built) = lay_out(&[SohmIndexContent {
            spec: spec(50),
            messages: messages.clone(),
        }]);
        let table = SohmMasterTable::decode(
            &file
                .read_block(built.table_addr, SohmMasterTable::encoded_size(&ctx(), 1))
                .unwrap(),
            &ctx(),
            1,
        )
        .unwrap();
        let header = &table.indexes[0];
        assert_eq!(header.index_type, SOHM_INDEX_BTREE);
        assert_eq!(header.num_messages, 200);

        let bt2 =
            Bt2Header::decode(&file.read_block(header.index_addr, 256).unwrap(), &ctx()).unwrap();
        assert!(bt2.depth > 0, "expected a multi-level index, got one leaf");

        let records = read_records(&mut file, header);
        assert_eq!(records.len(), 200);
        for message in &messages {
            let hash = message_hash(&message.body, message.msg_type);
            let record = records.iter().find(|r| r.hash == hash).unwrap();
            assert_eq!(read_body(&mut file, header.heap_addr, record), message.body);
        }
    }

    /// Two indexes: each gets its own heap, and the table sends a message type
    /// to the heap of the index whose mask covers it.
    #[test]
    fn each_index_gets_its_own_heap() {
        let (mut file, built) = lay_out(&[
            SohmIndexContent {
                spec: SohmIndexSpec {
                    mesg_types: 1 << MSG_ATTRIBUTE,
                    ..spec(50)
                },
                messages: vec![message(MSG_ATTRIBUTE, 3, 40, 2)],
            },
            SohmIndexContent {
                spec: SohmIndexSpec {
                    mesg_types: (1 << MSG_DATATYPE) | (1 << MSG_DATASPACE),
                    ..spec(50)
                },
                messages: vec![message(MSG_DATATYPE, 9, 20, 3)],
            },
        ]);
        let table = SohmMasterTable::decode(
            &file
                .read_block(built.table_addr, SohmMasterTable::encoded_size(&ctx(), 2))
                .unwrap(),
            &ctx(),
            2,
        )
        .unwrap();
        assert_eq!(table.indexes.len(), 2);
        assert_ne!(table.indexes[0].heap_addr, table.indexes[1].heap_addr);
        assert_eq!(
            table.heap_addr(MSG_ATTRIBUTE),
            Some(table.indexes[0].heap_addr)
        );
        assert_eq!(
            table.heap_addr(MSG_DATASPACE),
            Some(table.indexes[1].heap_addr)
        );
    }

    /// An index covering a type nothing used still has a heap and an index of
    /// its own — `H5SM__create_index` makes both when the file is created.
    #[test]
    fn an_empty_index_is_still_laid_out() {
        let (mut file, built) = lay_out(&[SohmIndexContent {
            spec: spec(50),
            messages: Vec::new(),
        }]);
        let table = SohmMasterTable::decode(
            &file
                .read_block(built.table_addr, SohmMasterTable::encoded_size(&ctx(), 1))
                .unwrap(),
            &ctx(),
            1,
        )
        .unwrap();
        assert_eq!(table.indexes[0].num_messages, 0);
        assert_ne!(table.indexes[0].heap_addr, UNDEF_ADDR);
        assert!(built.heap_ids.is_empty());
    }
}
