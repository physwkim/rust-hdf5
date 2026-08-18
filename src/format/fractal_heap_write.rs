//! Fractal heap writer.
//!
//! Lays a set of objects out as a fractal heap and hands back the blocks to
//! write plus one heap ID per object. The heap is built whole, in one pass,
//! rather than grown insert by insert the way `H5HF_insert` does — every
//! caller here rewrites its object header from scratch anyway, so there is
//! nothing to grow into.
//!
//! Objects land in one of two places, exactly as `H5HF_insert` chooses:
//!
//! * below the header's `max_man_size`, packed into *managed* direct blocks
//!   whose sizes come from the doubling table;
//! * at or above it, written as a *huge* object in its own file allocation and
//!   indexed by the heap's huge-object v2 B-tree.
//!
//! ("Tiny" objects — those that fit inside the heap ID itself — are a third
//! case libhdf5 uses; this writer does not produce them, and none of its
//! readers require it to. A tiny object is a space optimisation, not a
//! different heap.)
//!
//! References: `H5HFcache.c` (`H5HF__cache_hdr_serialize`,
//! `H5HF__cache_dblock_*`, `H5HF__cache_iblock_serialize`), `H5HFdblock.c`
//! (`H5HF__man_dblock_create`), `H5HFhuge.c` (`H5HF__huge_insert`),
//! `H5HFbtree2.c` (`H5HF__huge_bt2_indir_encode`).

use crate::format::checksum::checksum_metadata;
use crate::format::chunk_index::btree_v2::{Bt2Tree, BT2_TYPE_FHEAP_HUGE_INDIR};
use std::collections::HashMap;

use crate::format::fractal_heap::{
    indirect_nrows, FractalHeapHeader, HeapParams, FHDB_SIGNATURE, FHIB_SIGNATURE,
};
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Node size of the huge-object index (`H5HF_HUGE_BT2_NODE_SIZE`).
const HUGE_BT2_NODE_SIZE: u32 = 512;

/// Heap ID flag byte: current version, "managed" object
/// (`H5HF_ID_VERS_CURR | H5HF_ID_TYPE_MAN`).
const ID_FLAGS_MANAGED: u8 = 0x00;
/// Heap ID flag byte for a "huge" object.
const ID_FLAGS_HUGE: u8 = 0x10;

/// A block of bytes the caller must write at `addr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeapBlock {
    /// File address the block was allocated at.
    pub addr: u64,
    /// Length of the allocation. Equal to `image.len()` for every block this
    /// writer emits; kept explicitly so a caller freeing the block later does
    /// not have to re-derive it.
    pub len: u64,
    /// The bytes to write there.
    pub image: Vec<u8>,
}

/// A laid-out heap: everything to write, and how to find what was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltHeap {
    /// Address of the heap header — what an `Attribute Info` message names.
    pub header_addr: u64,
    /// Header, blocks, huge objects and index nodes, in write order.
    pub blocks: Vec<HeapBlock>,
    /// One heap ID per input object, in the order they were given.
    pub ids: Vec<Vec<u8>>,
}

/// A heap whose layout is settled but whose objects have not been written.
///
/// Every block has its address and every object its heap ID, and none of that
/// depends on what the objects contain — only on how long they are. The split
/// exists because a shared-message heap holds objects that name other objects
/// in the same heap by heap ID: `H5A__create` shares an attribute's datatype
/// and dataspace before `H5O__attr_create` shares the attribute itself
/// (H5Aint.c:375-377), so the attribute body that goes into the heap already
/// carries their IDs. A caller reads [`ids`](Self::ids), finishes its objects
/// against them, and hands the result to [`finish`](Self::finish).
pub struct PlannedHeap {
    header: FractalHeapHeader,
    ctx: FormatContext,
    header_addr: u64,
    /// The lengths the layout was computed from; [`finish`](Self::finish)
    /// refuses objects that no longer match them.
    lengths: Vec<usize>,
    ids: Vec<Vec<u8>>,
    /// Indirect blocks: their content is addresses, not objects.
    managed_meta: Vec<HeapBlock>,
    /// The huge-object index, likewise.
    huge_meta: Vec<HeapBlock>,
    managed: ManagedPlan,
    huge: Vec<HugeSlot>,
}

/// Where the managed objects go, once their direct blocks are placed.
struct ManagedPlan {
    built: Vec<DirectBlock>,
    addrs: Vec<u64>,
    /// Object indices, in placement order, each with the direct block it went
    /// into and the offset of its bytes in that block's object area.
    slots: Vec<(usize, usize, usize)>,
    overhead: usize,
}

/// One huge object's own allocation.
struct HugeSlot {
    addr: u64,
    len: u64,
    object: usize,
}

impl PlannedHeap {
    /// One heap ID per object, in the order the lengths were given.
    pub fn ids(&self) -> &[Vec<u8>] {
        &self.ids
    }

    /// Write the objects into the layout this planned.
    ///
    /// Every object must still be the length it was planned at — the heap IDs
    /// already handed out record each object's length, so a different one
    /// would describe bytes that are not there.
    pub fn finish(self, objects: &[Vec<u8>]) -> FormatResult<BuiltHeap> {
        if objects.len() != self.lengths.len()
            || objects
                .iter()
                .zip(&self.lengths)
                .any(|(o, &n)| o.len() != n)
        {
            return Err(FormatError::InvalidData(
                "fractal heap objects do not match the lengths their layout was planned from"
                    .into(),
            ));
        }
        let Self {
            header,
            ctx,
            header_addr,
            ids,
            managed_meta,
            huge_meta,
            managed,
            huge,
            ..
        } = self;

        let mut blocks = managed_meta;
        let mut images: Vec<Vec<u8>> = managed
            .built
            .iter()
            .map(|b| direct_prefix(&header, &ctx, header_addr, b.block_off, b.size))
            .collect();
        for &(object, bi, off) in &managed.slots {
            let start = managed.overhead + off;
            images[bi][start..start + objects[object].len()].copy_from_slice(&objects[object]);
        }
        for (image, b) in images.iter_mut().zip(&managed.built) {
            finish_direct_block(&header, &ctx, image, b.size as usize);
        }
        for (image, (&addr, b)) in images
            .into_iter()
            .zip(managed.addrs.iter().zip(&managed.built))
        {
            blocks.push(HeapBlock {
                addr,
                len: b.size,
                image,
            });
        }
        for slot in huge {
            blocks.push(HeapBlock {
                addr: slot.addr,
                len: slot.len,
                image: objects[slot.object].clone(),
            });
        }
        blocks.extend(huge_meta);

        let image = header.encode(&ctx);
        blocks.insert(
            0,
            HeapBlock {
                addr: header_addr,
                len: image.len() as u64,
                image,
            },
        );

        Ok(BuiltHeap {
            header_addr,
            blocks,
            ids,
        })
    }
}

/// Lay `objects` out as a fractal heap.
///
/// `alloc` allocates `len` bytes of file space and returns the address; it is
/// called once per block, header included, and the returned [`BuiltHeap`]
/// carries every allocation back so a caller that fails afterwards can free
/// them.
pub fn build_heap(
    params: &HeapParams,
    ctx: &FormatContext,
    objects: &[Vec<u8>],
    alloc: &mut dyn FnMut(u64) -> u64,
) -> FormatResult<BuiltHeap> {
    let lengths: Vec<usize> = objects.iter().map(Vec::len).collect();
    plan_heap(params, ctx, &lengths, alloc)?.finish(objects)
}

/// Decide where objects of these `lengths` would go, without needing them.
///
/// The half of [`build_heap`] that takes the file space; see [`PlannedHeap`]
/// for why a caller ever wants the two halves apart.
pub fn plan_heap(
    params: &HeapParams,
    ctx: &FormatContext,
    lengths: &[usize],
    alloc: &mut dyn FnMut(u64) -> u64,
) -> FormatResult<PlannedHeap> {
    let mut header = FractalHeapHeader::new(params, ctx);

    // The header address comes first: every direct and indirect block names it
    // in its own prefix, so nothing else can be encoded until it is known. Its
    // size does not depend on any of them.
    let header_addr = alloc(FractalHeapHeader::encoded_size(ctx) as u64);

    let mut ids: Vec<Vec<u8>> = vec![Vec::new(); lengths.len()];

    // Partition by the same rule as `H5HF_insert`: the size decides, not the
    // caller.
    let managed: Vec<usize> = (0..lengths.len())
        .filter(|&i| (lengths[i] as u64) < params.max_man_size as u64)
        .collect();
    let huge: Vec<usize> = (0..lengths.len())
        .filter(|&i| (lengths[i] as u64) >= params.max_man_size as u64)
        .collect();

    let mut managed_meta = Vec::new();
    let managed = plan_managed(
        &mut header,
        ctx,
        lengths,
        &managed,
        header_addr,
        alloc,
        &mut managed_meta,
        &mut ids,
    )?;
    let mut huge_meta = Vec::new();
    let huge = plan_huge(
        &mut header,
        ctx,
        lengths,
        &huge,
        alloc,
        &mut huge_meta,
        &mut ids,
    );

    Ok(PlannedHeap {
        header,
        ctx: *ctx,
        header_addr,
        lengths: lengths.to_vec(),
        ids,
        managed_meta,
        huge_meta,
        managed,
        huge,
    })
}

/// Bytes a direct block spends before its object area
/// (`H5HF_MAN_ABS_DIRECT_OVERHEAD`).
fn direct_overhead(header: &FractalHeapHeader, ctx: &FormatContext) -> usize {
    4 + 1
        + ctx.sizeof_addr as usize
        + header.heap_off_size as usize
        + if header.checksum_dblocks { 4 } else { 0 }
}

/// One managed direct block under construction.
struct DirectBlock {
    /// Position in the row-major block sequence.
    seq: usize,
    /// Block size from the doubling table.
    size: u64,
    /// Heap-space offset of the block image.
    block_off: u64,
    /// Object bytes packed so far.
    used: usize,
}

/// Direct blocks reachable from an indirect block of `n` rows, for every `n`
/// the doubling table can produce.
///
/// A row past `max_direct_rows` holds child indirect blocks rather than direct
/// ones, and each child covers that row's block size with rows of its own
/// (`H5HF__dtable_size_to_rows`), so the count is recursive. Every child has
/// strictly fewer rows than the row that names it, which is what lets this be
/// filled in one ascending pass.
fn direct_block_counts(header: &FractalHeapHeader) -> Vec<usize> {
    let width = header.table_width as usize;
    let mut counts = Vec::with_capacity(header.row_block_size.len() + 1);
    counts.push(0);
    for row in 0..header.row_block_size.len() {
        let this_row = if row < header.max_direct_rows as usize {
            width
        } else {
            let child = indirect_nrows(header, header.row_block_size[row]) as usize;
            width * counts[child]
        };
        counts.push(counts[row] + this_row);
    }
    counts
}

/// Heap-space offset one past `nrows` rows of an indirect block, i.e. the span
/// such a block covers.
fn row_span(header: &FractalHeapHeader, nrows: usize) -> u64 {
    match header.row_block_off.get(nrows) {
        Some(&off) => off,
        None => {
            let last = header.row_block_off.len() - 1;
            header.row_block_off[last] + header.row_block_size[last] * header.table_width as u64
        }
    }
}

/// Size and heap offset of the `n`-th managed direct block, walking the rows
/// of the indirect block at `base_off` in heap-offset order and descending
/// into the indirect rows the way `H5HF__man_iter_next` does. `None` once the
/// sequence runs past what this block's rows address.
fn nth_direct_block(
    header: &FractalHeapHeader,
    counts: &[usize],
    base_off: u64,
    nrows: usize,
    mut n: usize,
) -> Option<(u64, u64)> {
    let width = header.table_width as usize;
    for row in 0..nrows.min(header.row_block_size.len()) {
        let size = header.row_block_size[row];
        let row_base = base_off + header.row_block_off[row];
        if row < header.max_direct_rows as usize {
            if n < width {
                return Some((size, row_base + size * n as u64));
            }
            n -= width;
        } else {
            let child_nrows = indirect_nrows(header, size) as usize;
            let per_child = counts[child_nrows];
            for col in 0..width {
                if n < per_child {
                    return nth_direct_block(
                        header,
                        counts,
                        row_base + size * col as u64,
                        child_nrows,
                        n,
                    );
                }
                n -= per_child;
            }
        }
    }
    None
}

/// Pack the managed objects into direct blocks and record their heap IDs.
///
/// Only the lengths are needed: a managed heap ID is the object's heap-space
/// offset and its length, and a direct block's address comes from `alloc`, so
/// nothing here reads an object's bytes.
#[allow(clippy::too_many_arguments)]
fn plan_managed(
    header: &mut FractalHeapHeader,
    ctx: &FormatContext,
    lengths: &[usize],
    managed: &[usize],
    header_addr: u64,
    alloc: &mut dyn FnMut(u64) -> u64,
    blocks: &mut Vec<HeapBlock>,
    ids: &mut [Vec<u8>],
) -> FormatResult<ManagedPlan> {
    if managed.is_empty() {
        return Ok(ManagedPlan {
            built: Vec::new(),
            addrs: Vec::new(),
            slots: Vec::new(),
            overhead: 0,
        });
    }
    let overhead = direct_overhead(header, ctx);
    let counts = direct_block_counts(header);
    let root_rows = header.row_block_size.len();
    let mut built: Vec<DirectBlock> = Vec::new();
    // Which block each managed object went into, and where inside its object
    // area — the heap ID cannot be encoded until the block's own offset is
    // known, which it is from the start, so this is recorded as it goes.
    let mut placement: Vec<(usize, usize)> = Vec::with_capacity(managed.len());
    let mut cursor = 0usize;

    for &i in managed {
        let len = lengths[i];
        loop {
            let Some((size, block_off)) = nth_direct_block(header, &counts, 0, root_rows, cursor)
            else {
                return Err(FormatError::UnsupportedFeature(format!(
                    "fractal heap needs more than the {} bytes its {}-bit address space \
                     addresses",
                    row_span(header, root_rows),
                    header.max_heap_size_bits
                )));
            };
            let capacity = size as usize - overhead;
            // A row whose blocks are too small for this object can never take
            // it; skip the row rather than allocating a block it will not fit.
            if len > capacity {
                cursor += 1;
                continue;
            }
            let last = built.len().wrapping_sub(1);
            match built.last().map(|b| (b.seq, b.used)) {
                Some((seq, used)) if seq == cursor && used + len <= capacity => {
                    placement.push((last, used));
                    built[last].used += len;
                    break;
                }
                // The current block is full: move on to the next one, leaving
                // its tail unused. `H5HF__man_insert` reaches the same layout
                // whenever its free-space search comes up empty.
                Some((seq, _)) if seq == cursor => {
                    cursor += 1;
                    continue;
                }
                _ => built.push(DirectBlock {
                    seq: cursor,
                    size,
                    block_off,
                    used: 0,
                }),
            }
        }
    }

    // Assign addresses in block order, then the heap ID each object's slot
    // gives it. The bytes go in once the caller has finished them
    // ([`PlannedHeap::finish`]).
    let addrs: Vec<u64> = built.iter().map(|b| alloc(b.size)).collect();
    let mut slots = Vec::with_capacity(managed.len());
    for (&i, &(bi, off)) in managed.iter().zip(&placement) {
        let start = overhead + off;
        slots.push((i, bi, off));
        ids[i] = managed_id(
            header,
            built[bi].block_off + start as u64,
            lengths[i] as u64,
        );
    }

    let last = built.last().expect("a managed object built a block");
    let root_is_direct = built.len() == 1 && last.block_off == 0;
    if root_is_direct {
        header.table_addr = addrs[0];
        header.curr_root_rows = 0;
        header.man_size = header.start_block_size;
        // libhdf5 leaves the iterator at zero while the root is a lone direct
        // block (`H5HF__hdr_reset_iter`), and only starts advancing it once a
        // root indirect block exists.
        header.man_iter_off = 0;
    } else {
        // The root needs whatever rows it takes to reach the last block used.
        // libhdf5 grows the root by doubling and can end up with more; either
        // is legal, because `curr_root_rows` is what a reader goes by.
        let end = last.block_off + last.size;
        let nrows = (1..=root_rows)
            .find(|&r| row_span(header, r) >= end)
            .expect("the placement loop never runs past the addressable rows");
        let block_addrs: HashMap<u64, u64> = built
            .iter()
            .zip(&addrs)
            .map(|(b, &addr)| (b.block_off, addr))
            .collect();
        let iblock_addr = encode_indirect(
            header,
            ctx,
            header_addr,
            0,
            nrows,
            &block_addrs,
            alloc,
            blocks,
        );
        header.table_addr = iblock_addr;
        header.curr_root_rows = nrows as u16;
        header.man_size = row_span(header, nrows);
        header.man_iter_off = end;
    }
    header.man_alloc_size = built.iter().map(|b| b.size).sum();
    header.man_nobjs = managed.len() as u64;
    // Every unused byte past the last object stays unused: this writer keeps
    // no free-space manager, so it claims no free space either rather than
    // advertising bytes nothing can hand out.
    header.total_man_free = 0;
    header.fs_addr = UNDEF_ADDR;

    Ok(ManagedPlan {
        built,
        addrs,
        slots,
        overhead,
    })
}

/// The fixed prefix of a direct block, zero-padded to its full size.
fn direct_prefix(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    header_addr: u64,
    block_off: u64,
    size: u64,
) -> Vec<u8> {
    let sa = ctx.sizeof_addr as usize;
    let mut image = vec![0u8; size as usize];
    image[0..4].copy_from_slice(&FHDB_SIGNATURE);
    image[4] = 0; // version
    image[5..5 + sa].copy_from_slice(&header_addr.to_le_bytes()[..sa]);
    let off_at = 5 + sa;
    let off_size = header.heap_off_size as usize;
    image[off_at..off_at + off_size].copy_from_slice(&block_off.to_le_bytes()[..off_size]);
    image
}

/// Stamp a direct block's checksum, once its objects are in place.
///
/// `H5HF__cache_dblock_verify_chksum` sums the *whole* block image with the
/// checksum field zeroed, so it has to be the last thing written.
fn finish_direct_block(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    image: &mut [u8],
    _size: usize,
) {
    if !header.checksum_dblocks {
        return;
    }
    let at = 4 + 1 + ctx.sizeof_addr as usize + header.heap_off_size as usize;
    let cksum = checksum_metadata(image);
    image[at..at + 4].copy_from_slice(&cksum.to_le_bytes());
}

/// Encode the indirect block covering `base_off` with `nrows` rows, appending
/// its image — and those of every child indirect block under it — to `blocks`.
/// Returns its address.
///
/// A row below `max_direct_rows` names direct blocks, which `block_addrs`
/// already places by heap offset; a row at or above it names child indirect
/// blocks, each covering that row's block size, which this builds on the way
/// past. An entry with nothing under it is `UNDEF_ADDR`, exactly as
/// `H5HF__cache_iblock_serialize` writes an unallocated child.
#[allow(clippy::too_many_arguments)]
fn encode_indirect(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    header_addr: u64,
    base_off: u64,
    nrows: usize,
    block_addrs: &HashMap<u64, u64>,
    alloc: &mut dyn FnMut(u64) -> u64,
    blocks: &mut Vec<HeapBlock>,
) -> u64 {
    let sa = ctx.sizeof_addr as usize;
    let width = header.table_width as usize;
    let entries = nrows * width;
    let mut image =
        Vec::with_capacity(4 + 1 + sa + header.heap_off_size as usize + entries * sa + 4);
    image.extend_from_slice(&FHIB_SIGNATURE);
    image.push(0); // version
    image.extend_from_slice(&header_addr.to_le_bytes()[..sa]);
    image.extend_from_slice(&base_off.to_le_bytes()[..header.heap_off_size as usize]);
    for row in 0..nrows {
        let size = header.row_block_size[row];
        for col in 0..width {
            let off = base_off + header.row_block_off[row] + size * col as u64;
            let addr = if row < header.max_direct_rows as usize {
                block_addrs.get(&off).copied().unwrap_or(UNDEF_ADDR)
            } else if block_addrs
                .keys()
                .any(|&b| b >= off && b < off.saturating_add(size))
            {
                encode_indirect(
                    header,
                    ctx,
                    header_addr,
                    off,
                    indirect_nrows(header, size) as usize,
                    block_addrs,
                    alloc,
                    blocks,
                )
            } else {
                UNDEF_ADDR
            };
            image.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }
    }
    let cksum = checksum_metadata(&image);
    image.extend_from_slice(&cksum.to_le_bytes());

    let len = image.len() as u64;
    let addr = alloc(len);
    blocks.push(HeapBlock { addr, len, image });
    addr
}

/// A managed heap ID: flags, the object's heap-space offset, its length.
fn managed_id(header: &FractalHeapHeader, offset: u64, length: u64) -> Vec<u8> {
    let mut id = Vec::with_capacity(header.id_len as usize);
    id.push(ID_FLAGS_MANAGED);
    id.extend_from_slice(&offset.to_le_bytes()[..header.heap_off_size as usize]);
    id.extend_from_slice(&length.to_le_bytes()[..header.heap_len_size as usize]);
    id.resize(header.id_len as usize, 0);
    id
}

/// Give each huge object its own allocation and index them by ID.
fn plan_huge(
    header: &mut FractalHeapHeader,
    ctx: &FormatContext,
    lengths: &[usize],
    huge: &[usize],
    alloc: &mut dyn FnMut(u64) -> u64,
    blocks: &mut Vec<HeapBlock>,
    ids: &mut [Vec<u8>],
) -> Vec<HugeSlot> {
    if huge.is_empty() {
        return Vec::new();
    }
    let sa = ctx.sizeof_addr as usize;
    let ss = ctx.sizeof_size as usize;
    let record_size = (sa + ss + ss) as u16;

    // `H5HF__huge_new_id` pre-increments, so IDs start at 1 and 0 never
    // appears; the B-tree orders records by that ID, which insertion order
    // already gives.
    let mut slots = Vec::with_capacity(huge.len());
    let mut records = Vec::with_capacity(huge.len() * record_size as usize);
    for (n, &i) in huge.iter().enumerate() {
        let len = lengths[i] as u64;
        let addr = alloc(len);
        let huge_id = n as u64 + 1;
        slots.push(HugeSlot {
            addr,
            len,
            object: i,
        });
        records.extend_from_slice(&addr.to_le_bytes()[..sa]);
        records.extend_from_slice(&len.to_le_bytes()[..ss]);
        records.extend_from_slice(&huge_id.to_le_bytes()[..ss]);

        let mut id = Vec::with_capacity(header.id_len as usize);
        id.push(ID_FLAGS_HUGE);
        id.extend_from_slice(&huge_id.to_le_bytes()[..header.huge_id_size as usize]);
        id.resize(header.id_len as usize, 0);
        ids[i] = id;

        header.huge_size += len;
    }
    header.huge_nobjs = huge.len() as u64;
    header.huge_next_id = huge.len() as u64;

    let tree = Bt2Tree::build(
        BT2_TYPE_FHEAP_HUGE_INDIR,
        record_size,
        HUGE_BT2_NODE_SIZE,
        ctx.sizeof_addr,
        &records,
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
    header.huge_bt2_addr = bt2_addr;
    slots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::fractal_heap::{collect_managed_blocks, read_heap_object, HeapId};
    use crate::format::BlockReader;

    /// A file image the builder's blocks are written into, so the reader can
    /// be pointed straight back at what the writer produced.
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
        fn put(&mut self, block: &HeapBlock) {
            let at = block.addr as usize;
            self.bytes[at..at + block.image.len()].copy_from_slice(&block.image);
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
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    /// Build a heap of `objects`, then read every one of them back through
    /// the reader that parses libhdf5's own heaps.
    fn round_trip(objects: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let ctx = ctx();
        let params = HeapParams::object_header();
        let mut file = MemFile::new();
        let built = {
            let mut alloc = |len: u64| file.alloc(len);
            build_heap(&params, &ctx, objects, &mut alloc).unwrap()
        };
        for block in &built.blocks {
            file.put(block);
        }

        let heap_buf = file.read_block(built.header_addr, 512).unwrap();
        let header = FractalHeapHeader::decode(&heap_buf, &ctx).unwrap();
        let blocks = collect_managed_blocks(&header, &ctx, &mut file).unwrap();
        built
            .ids
            .iter()
            .map(|id| {
                let parsed = HeapId::parse(id, &header, &ctx).unwrap();
                read_heap_object(&parsed, &header, &ctx, &blocks, &mut file).unwrap()
            })
            .collect()
    }

    fn obj(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    #[test]
    fn a_single_object_round_trips_through_a_root_direct_block() {
        let objects = vec![obj(1, 40)];
        assert_eq!(round_trip(&objects), objects);
    }

    #[test]
    fn the_root_stays_a_direct_block_while_one_block_holds_everything() {
        let ctx = ctx();
        let params = HeapParams::object_header();
        let objects: Vec<Vec<u8>> = (0..10).map(|i| obj(i, 33)).collect();
        let mut file = MemFile::new();
        let built = {
            let mut alloc = |len: u64| file.alloc(len);
            build_heap(&params, &ctx, &objects, &mut alloc).unwrap()
        };
        for block in &built.blocks {
            file.put(block);
        }
        let heap_buf = file.read_block(built.header_addr, 512).unwrap();
        let header = FractalHeapHeader::decode(&heap_buf, &ctx).unwrap();
        assert_eq!(header.curr_root_rows, 0);
        assert_eq!(header.man_size, 1024);
        assert_eq!(header.man_alloc_size, 1024);
        assert_eq!(header.man_nobjs, 10);
        assert_eq!(round_trip(&objects), objects);
    }

    #[test]
    fn objects_past_one_block_grow_a_root_indirect_block() {
        // 1002 usable bytes per row-0 block, so 60 objects of 100 bytes need
        // seven of them — two rows of the doubling table.
        let objects: Vec<Vec<u8>> = (0..60).map(|i| obj(i, 100)).collect();
        let ctx = ctx();
        let params = HeapParams::object_header();
        let mut file = MemFile::new();
        let built = {
            let mut alloc = |len: u64| file.alloc(len);
            build_heap(&params, &ctx, &objects, &mut alloc).unwrap()
        };
        for block in &built.blocks {
            file.put(block);
        }
        let heap_buf = file.read_block(built.header_addr, 512).unwrap();
        let header = FractalHeapHeader::decode(&heap_buf, &ctx).unwrap();
        assert!(header.curr_root_rows >= 2, "{}", header.curr_root_rows);
        assert_eq!(round_trip(&objects), objects);
    }

    /// Past the doubling table's direct rows the root's next row holds child
    /// *indirect* blocks, each with rows of its own. 130 objects of 4000
    /// bytes overrun the 504 KiB the direct rows address, so the writer must
    /// build that second level for them.
    #[test]
    fn objects_past_the_direct_rows_grow_child_indirect_blocks() {
        let objects: Vec<Vec<u8>> = (0..130).map(|i| obj(i as u8, 4000)).collect();
        let ctx = ctx();
        let params = HeapParams::object_header();
        let mut file = MemFile::new();
        let built = {
            let mut alloc = |len: u64| file.alloc(len);
            build_heap(&params, &ctx, &objects, &mut alloc).unwrap()
        };
        for block in &built.blocks {
            file.put(block);
        }
        let heap_buf = file.read_block(built.header_addr, 512).unwrap();
        let header = FractalHeapHeader::decode(&heap_buf, &ctx).unwrap();
        assert!(
            header.curr_root_rows as u32 > header.max_direct_rows,
            "{} rows does not reach the indirect ones ({} direct)",
            header.curr_root_rows,
            header.max_direct_rows
        );
        assert_eq!(header.man_nobjs, 130);
        assert_eq!(round_trip(&objects), objects);
    }

    /// Deep enough that a child indirect block has indirect rows of its own:
    /// the root's row 11 covers 1 MiB, which needs nine rows, one more than
    /// the eight the table's direct rows fill.
    #[test]
    fn a_heap_deep_enough_nests_indirect_blocks_two_levels() {
        let objects: Vec<Vec<u8>> = (0..1000).map(|i| obj(i as u8, 4000)).collect();
        let ctx = ctx();
        let params = HeapParams::object_header();
        let mut file = MemFile::new();
        let built = {
            let mut alloc = |len: u64| file.alloc(len);
            build_heap(&params, &ctx, &objects, &mut alloc).unwrap()
        };
        for block in &built.blocks {
            file.put(block);
        }
        let heap_buf = file.read_block(built.header_addr, 512).unwrap();
        let header = FractalHeapHeader::decode(&heap_buf, &ctx).unwrap();
        assert!(header.curr_root_rows >= 12, "{}", header.curr_root_rows);
        assert_eq!(round_trip(&objects), objects);
    }

    #[test]
    fn an_object_too_big_for_a_managed_block_goes_huge() {
        // At and above `max_man_size` the object leaves the managed blocks.
        let objects = vec![obj(7, 4096), obj(9, 20), obj(3, 100_000)];
        let ctx = ctx();
        let params = HeapParams::object_header();
        let mut file = MemFile::new();
        let built = {
            let mut alloc = |len: u64| file.alloc(len);
            build_heap(&params, &ctx, &objects, &mut alloc).unwrap()
        };
        for block in &built.blocks {
            file.put(block);
        }
        let heap_buf = file.read_block(built.header_addr, 512).unwrap();
        let header = FractalHeapHeader::decode(&heap_buf, &ctx).unwrap();
        assert_eq!(header.huge_nobjs, 2);
        assert_eq!(header.huge_size, 4096 + 100_000);
        assert_eq!(header.man_nobjs, 1);
        assert_ne!(header.huge_bt2_addr, UNDEF_ADDR);
        assert!(matches!(
            HeapId::parse(&built.ids[0], &header, &ctx).unwrap(),
            HeapId::HugeIndirect { .. }
        ));
        assert!(matches!(
            HeapId::parse(&built.ids[1], &header, &ctx).unwrap(),
            HeapId::Managed { .. }
        ));
        assert_eq!(round_trip(&objects), objects);
    }

    /// A heap of nothing but huge objects has no managed block at all, so the
    /// doubling table's root stays undefined — the state the reader must not
    /// mistake for a missing block.
    #[test]
    fn a_heap_of_only_huge_objects_has_no_managed_blocks() {
        let objects: Vec<Vec<u8>> = (0..3).map(|i| obj(i, 5000)).collect();
        assert_eq!(round_trip(&objects), objects);
    }

    #[test]
    fn an_object_larger_than_a_row_skips_to_a_row_that_fits() {
        // 3000 bytes cannot go in a 1024-byte row-0 block; it must land in a
        // row whose blocks are big enough, and the small object beside it
        // must still be findable.
        let objects = vec![obj(1, 50), obj(2, 3000), obj(3, 50)];
        assert_eq!(round_trip(&objects), objects);
    }

    /// The header's derived fields must survive the round trip unchanged:
    /// heap IDs are only parseable against them.
    #[test]
    fn header_round_trips_with_its_derived_widths() {
        let ctx = ctx();
        let params = HeapParams::object_header();
        let built = FractalHeapHeader::new(&params, &ctx);
        let decoded = FractalHeapHeader::decode(&built.encode(&ctx), &ctx).unwrap();
        assert_eq!(decoded.heap_off_size, 5);
        assert_eq!(decoded.heap_len_size, 2);
        assert_eq!(decoded.huge_id_size, 7);
        assert!(!decoded.huge_ids_direct);
        assert_eq!(decoded.max_direct_rows, 8);
        assert_eq!(decoded.row_block_size[..4], [1024, 1024, 2048, 4096]);
        assert_eq!(decoded.row_block_off[..4], [0, 4096, 8192, 16384]);
        assert_eq!(decoded.start_root_rows, params.start_root_rows);
    }
}
