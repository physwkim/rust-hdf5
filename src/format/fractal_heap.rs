//! Fractal heap reader (read-only).
//!
//! A fractal heap stores many small variable-length objects. In HDF5, dense
//! link storage for groups (used once a group exceeds the link phase-change
//! threshold) keeps each link as an encoded `Link` message inside a fractal
//! heap, indexed by a v2 B-tree referenced from the `Link Info` message.
//!
//! This module decodes the heap header (`FRHP`) and walks the managed
//! direct/indirect blocks (`FHDB` / `FHIB`) to recover the raw bytes of
//! every managed object. Callers decode those bytes themselves (e.g. as
//! `LinkMessage`).
//!
//! Layout references (libhdf5 2.x): `H5HFcache.c` (`H5HF__cache_hdr_deserialize`,
//! `H5HF__cache_dblock_deserialize`, `H5HF__cache_iblock_deserialize`,
//! `H5HF__dtable_decode`), `H5HFhdr.c`, `H5HFdtable.c`.

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::checksum::checksum_metadata;
use crate::format::{FormatContext, FormatError, FormatResult};

/// Fractal heap header signature.
pub const FRHP_SIGNATURE: [u8; 4] = *b"FRHP";
/// Fractal heap indirect block signature.
pub const FHIB_SIGNATURE: [u8; 4] = *b"FHIB";
/// Fractal heap direct block signature.
pub const FHDB_SIGNATURE: [u8; 4] = *b"FHDB";

const HDR_FLAG_CHECKSUM_DBLOCKS: u8 = 0x02;

/// Upper bound on the number of blocks visited when walking one heap, to
/// bound work on a corrupt or hostile file.
const MAX_BLOCKS: usize = 65_536;

/// Decoded fractal heap header — the fields needed to walk managed blocks.
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    /// Length of a heap ID in bytes.
    pub id_len: u16,
    /// Encoded length of the I/O filter pipeline (0 = no filters).
    pub filter_len: u16,
    /// Whether direct blocks carry a trailing checksum.
    pub checksum_dblocks: bool,
    /// Managed space currently allocated to the heap, in bytes.
    pub man_size: u64,
    /// Number of managed objects currently stored in the heap.
    pub man_nobjs: u64,
    /// Doubling-table: number of columns.
    pub table_width: u16,
    /// Doubling-table: starting (row 0) direct-block size in bytes.
    pub start_block_size: u64,
    /// Doubling-table: maximum direct-block size in bytes.
    pub max_direct_size: u64,
    /// Doubling-table: maximum heap size expressed as a count of bits.
    pub max_heap_size_bits: u16,
    /// Doubling-table: file address of the root block.
    pub table_addr: u64,
    /// Doubling-table: current number of rows in the root indirect block
    /// (0 means the root block is a single direct block).
    pub curr_root_rows: u16,
    /// Bytes used to encode a block offset within the heap address space.
    pub heap_off_size: u8,
    /// Bytes used to encode a managed object's length in a heap ID.
    pub heap_len_size: u8,
    /// Largest object the managed blocks may hold; anything bigger is a
    /// "huge" object stored outside the heap.
    pub max_man_size: u64,
    /// Number of rows whose blocks are direct blocks.
    pub max_direct_rows: u32,
    /// Per-row direct-block sizes (length == `max_root_rows`).
    pub row_block_size: Vec<u64>,
}

/// `log2` of a power-of-two value. Returns 0 for inputs that are not a
/// positive power of two (defensive — real heaps always use powers of two).
fn log2_of2(n: u64) -> u32 {
    if n == 0 || (n & (n - 1)) != 0 {
        return 0;
    }
    n.trailing_zeros()
}

/// `floor(log2(n))` for any positive `n` (`H5VM_log2_gen`); 0 for zero.
fn log2_gen(n: u64) -> u32 {
    if n == 0 {
        0
    } else {
        63 - n.leading_zeros()
    }
}

/// Number of bytes needed to store a value spanning `bits` bits.
fn size_of_offset_bits(bits: u16) -> u8 {
    bits.div_ceil(8) as u8
}

fn need(buf: &[u8], pos: usize, n: usize) -> FormatResult<()> {
    if buf.len() < pos + n {
        Err(FormatError::BufferTooShort {
            needed: pos + n,
            available: buf.len(),
        })
    } else {
        Ok(())
    }
}

impl FractalHeapHeader {
    /// Total fixed (filter-free) on-disk size of the heap header, used to
    /// validate the checksum span.
    fn base_size(ctx: &FormatContext) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        // prefix: signature(4) + version(1)
        // general: id_len(2) + filter_len(2) + flags(1)
        // huge: max_man_size(4) + huge_next_id(ss) + huge_bt2_addr(sa)
        // free: total_man_free(ss) + fs_addr(sa)
        // stats: 8 * ss
        // dtable: width(2) + start_block_size(ss) + max_direct_size(ss)
        //         + max_index(2) + start_root_rows(2) + table_addr(sa)
        //         + curr_root_rows(2)
        // checksum(4)
        4 + 1 + 2 + 2 + 1 + 4 + ss + sa + ss + sa + 8 * ss + 2 + ss + ss + 2 + 2 + sa + 2 + 4
    }

    /// Decode a fractal heap header from the bytes at its file address.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        let base = Self::base_size(ctx);
        need(buf, 0, base)?;

        if buf[0..4] != FRHP_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }
        let version = buf[4];
        if version != 0 {
            return Err(FormatError::InvalidVersion(version));
        }

        let mut pos = 5;

        let id_len = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let filter_len = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let heap_flags = buf[pos];
        pos += 1;
        let checksum_dblocks = heap_flags & HDR_FLAG_CHECKSUM_DBLOCKS != 0;

        // "Huge" object info.
        let max_man_size =
            u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as u64;
        pos += 4;
        pos += ss; // huge_next_id
        pos += sa; // huge_bt2_addr

        // "Managed" free-space info.
        pos += ss; // total_man_free
        pos += sa; // fs_addr

        // Statistics: man_size, man_alloc_size, man_iter_off, man_nobjs,
        // huge_size, huge_nobjs, tiny_size, tiny_nobjs.
        let man_size = read_uint(&buf[pos..], ss);
        pos += ss;
        pos += ss; // man_alloc_size
        pos += ss; // man_iter_off
        let man_nobjs = read_uint(&buf[pos..], ss);
        pos += ss;
        pos += ss; // huge_size
        pos += ss; // huge_nobjs
        pos += ss; // tiny_size
        pos += ss; // tiny_nobjs

        // Doubling-table info.
        let table_width = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let start_block_size = read_uint(&buf[pos..], ss);
        pos += ss;
        let max_direct_size = read_uint(&buf[pos..], ss);
        pos += ss;
        let max_heap_size_bits = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        pos += 2; // start_root_rows
        let table_addr = read_uint(&buf[pos..], sa);
        pos += sa;
        let curr_root_rows = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;

        debug_assert_eq!(pos, base - 4);

        // Verify the header checksum (covers everything before the 4-byte sum).
        let stored = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        let computed = checksum_metadata(&buf[..pos]);
        if stored != computed {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }

        if table_width == 0 || start_block_size == 0 {
            return Err(FormatError::InvalidData(
                "fractal heap doubling-table has zero width or block size".into(),
            ));
        }

        // Doubling-table derived values — see H5HFdtable.c::H5HF__dtable_init
        // and H5HFhdr.c::H5HF__hdr_finish_init_phase1.
        let start_bits = log2_of2(start_block_size);
        let first_row_bits = start_bits + log2_of2(table_width as u64);
        let max_root_rows = (max_heap_size_bits as u32)
            .saturating_sub(first_row_bits)
            .saturating_add(1);
        let max_direct_bits = log2_of2(max_direct_size);
        let max_direct_rows = max_direct_bits.saturating_sub(start_bits).saturating_add(2);
        let heap_off_size = size_of_offset_bits(max_heap_size_bits);
        // H5HFhdr.c::H5HF__hdr_finish_init_phase1 — the length field of a
        // managed heap ID is sized by whichever bound is tighter: the largest
        // direct block, or the largest object the managed blocks may hold.
        let heap_len_size = std::cmp::min(
            size_of_offset_bits(log2_of2(max_direct_size) as u16),
            (log2_gen(max_man_size) / 8 + 1) as u8,
        );

        // Per-row direct-block sizes: row 0 == start, row 1 == start,
        // doubling from row 2 onward (H5HF__dtable_init).
        let mut row_block_size = Vec::with_capacity(max_root_rows as usize);
        if max_root_rows > 0 {
            row_block_size.push(start_block_size);
            let mut tmp = start_block_size;
            for _ in 1..max_root_rows {
                row_block_size.push(tmp);
                tmp = tmp.saturating_mul(2);
            }
        }

        Ok(Self {
            id_len,
            filter_len,
            checksum_dblocks,
            man_size,
            man_nobjs,
            table_width,
            start_block_size,
            max_direct_size,
            max_heap_size_bits,
            table_addr,
            curr_root_rows,
            heap_off_size,
            heap_len_size,
            max_man_size,
            max_direct_rows,
            row_block_size,
        })
    }
}

/// Fetches arbitrary file regions for the heap walker.
pub trait BlockReader {
    /// Read exactly `len` bytes starting at `offset`.
    fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>>;
}

/// Walk a fractal heap and return the raw payload bytes of every managed
/// direct block.
///
/// Each returned `Vec<u8>` is the object area of one direct block: a region
/// holding one or more managed objects packed contiguously. The caller
/// decodes objects from each payload using its own message decoder.
pub fn collect_managed_objects<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<Vec<u8>>> {
    let mut objects = Vec::new();
    if header.table_addr == u64::MAX || header.man_nobjs == 0 {
        return Ok(objects);
    }

    let mut block_budget = MAX_BLOCKS;

    if header.curr_root_rows == 0 {
        // Root block is a single direct block of `start_block_size`.
        read_direct_block(
            header,
            ctx,
            reader,
            header.table_addr,
            header.start_block_size as usize,
            &mut objects,
            &mut block_budget,
        )?;
    } else {
        walk_indirect_block(
            header,
            ctx,
            reader,
            header.table_addr,
            header.curr_root_rows as u32,
            &mut objects,
            &mut block_budget,
            0,
        )?;
    }

    Ok(objects)
}

/// Recursively walk an indirect block, descending into child direct and
/// indirect blocks.
#[allow(clippy::too_many_arguments)]
fn walk_indirect_block<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
    addr: u64,
    nrows: u32,
    objects: &mut Vec<Vec<u8>>,
    budget: &mut usize,
    depth: usize,
) -> FormatResult<()> {
    // The block budget bounds total blocks visited, not recursion depth: a
    // crafted heap that is a deep linear chain of indirect blocks would
    // recurse far enough to exhaust the stack before the budget runs out.
    const MAX_INDIRECT_DEPTH: usize = 256;
    if depth > MAX_INDIRECT_DEPTH {
        return Err(FormatError::InvalidData(
            "fractal heap indirect-block nesting exceeds maximum depth".into(),
        ));
    }
    if addr == u64::MAX || nrows == 0 {
        return Ok(());
    }
    if *budget == 0 {
        return Err(FormatError::InvalidData(
            "fractal heap block budget exhausted".into(),
        ));
    }
    *budget -= 1;

    let width = header.table_width as usize;
    let block = read_indirect_block(header, ctx, reader, addr, nrows)?;

    for (entry, &child_addr) in block.entries.iter().enumerate() {
        let row = entry / width;
        if child_addr == u64::MAX || child_addr == 0 {
            continue;
        }

        if row < header.max_direct_rows as usize {
            // Direct-block child.
            let size = header
                .row_block_size
                .get(row)
                .copied()
                .unwrap_or(header.start_block_size) as usize;
            read_direct_block(header, ctx, reader, child_addr, size, objects, budget)?;
        } else {
            // Indirect-block child. Its row count is derived from the row's
            // block size (see H5HFhdr.c / H5HF__dtable_size_to_rows).
            let block_size = header
                .row_block_size
                .get(row)
                .copied()
                .unwrap_or(header.start_block_size);
            let child_nrows = indirect_nrows(header, block_size);
            walk_indirect_block(
                header,
                ctx,
                reader,
                child_addr,
                child_nrows,
                objects,
                budget,
                depth + 1,
            )?;
        }
    }

    Ok(())
}

/// An indirect block's decoded fields: where it starts in the heap's address
/// space, and its child block addresses in entry order.
struct IndirectBlock {
    block_off: u64,
    entries: Vec<u64>,
}

/// Read and verify one indirect block (`FHIB`).
fn read_indirect_block<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
    addr: u64,
    nrows: u32,
) -> FormatResult<IndirectBlock> {
    let sa = ctx.sizeof_addr as usize;
    let width = header.table_width as usize;
    let n_entries = nrows as usize * width;

    // Indirect-block size: prefix(sig+ver) + heap_addr + block_off
    //   + per-entry child addresses (+ filter info on direct rows if filtered)
    //   + checksum.
    let dir_rows = nrows.min(header.max_direct_rows) as usize;
    let dir_entries = dir_rows * width;
    let per_dir_entry = if header.filter_len > 0 {
        sa + ctx.sizeof_size as usize + 4
    } else {
        sa
    };
    let indir_entries = n_entries - dir_entries;
    let block_len = 4
        + 1
        + sa
        + header.heap_off_size as usize
        + dir_entries * per_dir_entry
        + indir_entries * sa
        + 4;

    let buf = reader.read_block(addr, block_len)?;
    need(&buf, 0, block_len)?;

    if buf[0..4] != FHIB_SIGNATURE {
        return Err(FormatError::InvalidSignature);
    }
    if buf[4] != 0 {
        return Err(FormatError::InvalidVersion(buf[4]));
    }

    // Verify checksum.
    let csum_off = block_len - 4;
    let stored = u32::from_le_bytes([
        buf[csum_off],
        buf[csum_off + 1],
        buf[csum_off + 2],
        buf[csum_off + 3],
    ]);
    let computed = checksum_metadata(&buf[..csum_off]);
    if stored != computed {
        return Err(FormatError::ChecksumMismatch {
            expected: stored,
            computed,
        });
    }

    // prefix: signature(4) + version(1) + heap header address(sa)
    //         + block offset(heap_off_size).
    let mut pos = 4 + 1 + sa;
    let block_off = read_uint(&buf[pos..], header.heap_off_size as usize);
    pos += header.heap_off_size as usize;

    let mut entries = Vec::with_capacity(n_entries);
    for entry in 0..n_entries {
        let row = entry / width;
        entries.push(read_uint(&buf[pos..], sa));
        pos += sa;
        if header.filter_len > 0 && row < header.max_direct_rows as usize {
            // Filtered direct-block entries carry size + filter mask.
            pos += ctx.sizeof_size as usize + 4;
        }
    }

    Ok(IndirectBlock { block_off, entries })
}

/// Row and column of `off` in the doubling table (`H5HF__dtable_lookup`).
fn dtable_lookup(header: &FractalHeapHeader, off: u64) -> (u32, u32) {
    let width = header.table_width as u64;
    let num_id_first_row = header.start_block_size.saturating_mul(width);
    if off < num_id_first_row {
        (0, (off / header.start_block_size) as u32)
    } else {
        let first_row_bits = log2_of2(header.start_block_size) + log2_of2(width);
        let high_bit = log2_gen(off);
        let off_mask = 1u64 << high_bit;
        let row = high_bit.saturating_sub(first_row_bits) + 1;
        let row_size = header
            .row_block_size
            .get(row as usize)
            .copied()
            .unwrap_or(header.start_block_size)
            .max(1);
        (row, ((off - off_mask) / row_size) as u32)
    }
}

/// A managed-object heap ID: an offset into the heap's address space and the
/// object's length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManagedHeapId {
    /// Offset of the object in the heap's address space.
    pub obj_off: u64,
    /// Length of the object in bytes.
    pub obj_len: usize,
}

/// Heap-ID flag bits (`H5HFpkg.h`): version in the top two bits, storage type
/// in the next two.
const ID_VERSION_MASK: u8 = 0xC0;
const ID_TYPE_MASK: u8 = 0x30;
const ID_TYPE_MANAGED: u8 = 0x00;

impl ManagedHeapId {
    /// Decode a managed heap ID. Huge and tiny objects are rejected: this
    /// reader resolves only managed ones, and silently treating one kind as
    /// another would read unrelated bytes.
    pub fn decode(header: &FractalHeapHeader, id: &[u8]) -> FormatResult<Self> {
        let off_size = header.heap_off_size as usize;
        let len_size = header.heap_len_size as usize;
        need(id, 0, 1 + off_size + len_size)?;

        let flags = id[0];
        if flags & ID_VERSION_MASK != 0 {
            return Err(FormatError::InvalidData(format!(
                "unsupported fractal heap ID version in flags {flags:#04x}"
            )));
        }
        if flags & ID_TYPE_MASK != ID_TYPE_MANAGED {
            return Err(FormatError::UnsupportedFeature(
                "fractal heap ID refers to a huge or tiny object".into(),
            ));
        }

        let obj_off = read_uint(&id[1..], off_size);
        let obj_len = read_uint(&id[1 + off_size..], len_size);

        // H5HF__man_op_real's range checks, in the same order.
        if obj_off == 0 || obj_off > header.man_size {
            return Err(FormatError::InvalidData(format!(
                "fractal heap object offset {obj_off} outside the managed space"
            )));
        }
        if obj_len == 0 || obj_len > header.max_direct_size || obj_len > header.max_man_size {
            return Err(FormatError::InvalidData(format!(
                "fractal heap object length {obj_len} is not storable in a direct block"
            )));
        }

        Ok(Self {
            obj_off,
            obj_len: obj_len as usize,
        })
    }
}

/// Read one managed object out of the heap, given its heap ID.
///
/// This is the read half of `H5HF_read`: decode the ID, walk the doubling
/// table down to the direct block that owns the offset, and return the bytes
/// of the object itself.
pub fn read_managed_object<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
    id: &[u8],
) -> FormatResult<Vec<u8>> {
    let id = ManagedHeapId::decode(header, id)?;

    // Locate the direct block holding the offset.
    let (dblock_addr, dblock_size) = if header.curr_root_rows == 0 {
        (header.table_addr, header.start_block_size as usize)
    } else {
        locate_direct_block(header, ctx, reader, id.obj_off)?
    };
    if dblock_addr == u64::MAX || dblock_addr == 0 {
        return Err(FormatError::InvalidData(
            "fractal heap ID is not in an allocated direct block".into(),
        ));
    }

    let buf = reader.read_block(dblock_addr, dblock_size)?;
    need(&buf, 0, dblock_size)?;
    if buf[0..4] != FHDB_SIGNATURE {
        return Err(FormatError::InvalidSignature);
    }
    if buf[4] != 0 {
        return Err(FormatError::InvalidVersion(buf[4]));
    }

    let sa = ctx.sizeof_addr as usize;
    let mut overhead = 4 + 1 + sa + header.heap_off_size as usize;
    let block_off = read_uint(&buf[4 + 1 + sa..], header.heap_off_size as usize);
    if header.checksum_dblocks {
        overhead += 4;
    }

    if id.obj_off < block_off {
        return Err(FormatError::InvalidData(
            "fractal heap object offset precedes its direct block".into(),
        ));
    }
    let blk_off = (id.obj_off - block_off) as usize;
    if blk_off < overhead {
        return Err(FormatError::InvalidData(
            "fractal heap object located in the prefix of its direct block".into(),
        ));
    }
    if blk_off + id.obj_len > dblock_size {
        return Err(FormatError::InvalidData(
            "fractal heap object overruns the end of its direct block".into(),
        ));
    }

    Ok(buf[blk_off..blk_off + id.obj_len].to_vec())
}

/// Walk the doubling table from the root indirect block down to the direct
/// block that owns `obj_off` (`H5HF__man_dblock_locate`), returning its
/// address and size.
fn locate_direct_block<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
    obj_off: u64,
) -> FormatResult<(u64, usize)> {
    // Each descent moves to a strictly smaller indirect block, so the row
    // count bounds the walk; cap it anyway for a corrupt heap.
    const MAX_DESCENT: usize = 256;

    let width = header.table_width as usize;
    let mut addr = header.table_addr;
    let mut nrows = header.curr_root_rows as u32;

    for _ in 0..MAX_DESCENT {
        let block = read_indirect_block(header, ctx, reader, addr, nrows)?;
        if obj_off < block.block_off {
            return Err(FormatError::InvalidData(
                "fractal heap object offset precedes its indirect block".into(),
            ));
        }
        let (row, col) = dtable_lookup(header, obj_off - block.block_off);
        let entry = row as usize * width + col as usize;
        let child = block.entries.get(entry).copied().ok_or_else(|| {
            FormatError::InvalidData("fractal heap object offset outside its indirect block".into())
        })?;
        let block_size = header
            .row_block_size
            .get(row as usize)
            .copied()
            .unwrap_or(header.start_block_size);

        if row < header.max_direct_rows {
            return Ok((child, block_size as usize));
        }
        if child == u64::MAX || child == 0 {
            return Err(FormatError::InvalidData(
                "fractal heap ID is not in an allocated indirect block".into(),
            ));
        }
        addr = child;
        nrows = indirect_nrows(header, block_size);
    }

    Err(FormatError::InvalidData(
        "fractal heap doubling-table descent exceeds maximum depth".into(),
    ))
}

/// Number of rows in a child indirect block whose row-block size is
/// `block_size` (H5HF__dtable_size_to_rows).
fn indirect_nrows(header: &FractalHeapHeader, block_size: u64) -> u32 {
    let start_bits = log2_of2(header.start_block_size);
    let first_row_bits = start_bits + log2_of2(header.table_width as u64);
    let size_log2 = log2_of2(block_size);
    size_log2.saturating_sub(first_row_bits).saturating_add(1)
}

/// Read a direct block and append its object payload region.
fn read_direct_block<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
    addr: u64,
    size: usize,
    objects: &mut Vec<Vec<u8>>,
    budget: &mut usize,
) -> FormatResult<()> {
    if addr == u64::MAX || size == 0 {
        return Ok(());
    }
    if *budget == 0 {
        return Err(FormatError::InvalidData(
            "fractal heap block budget exhausted".into(),
        ));
    }
    *budget -= 1;

    let sa = ctx.sizeof_addr as usize;
    let buf = reader.read_block(addr, size)?;

    let prefix_min = 4 + 1 + sa + header.heap_off_size as usize;
    if buf.len() < prefix_min {
        return Ok(());
    }
    if buf[0..4] != FHDB_SIGNATURE {
        return Err(FormatError::InvalidSignature);
    }
    if buf[4] != 0 {
        return Err(FormatError::InvalidVersion(buf[4]));
    }

    // prefix: signature(4) + version(1) + heap header address(sa)
    //         + block offset(heap_off_size) + optional checksum(4)
    let mut payload_start = prefix_min;
    if header.checksum_dblocks {
        // Verify the direct-block checksum.
        //
        // libhdf5 (`H5HF__cache_dblock_verify_chksum` / `_pre_serialize` in
        // H5HFcache.c) computes the Jenkins `H5_checksum_metadata` over the
        // *entire* direct-block image (`dblock->size` bytes) with the 4-byte
        // checksum field cleared to zero. The checksum field sits at
        // `H5HF_MAN_ABS_DIRECT_OVERHEAD(hdr) - H5HF_SIZEOF_CHKSUM`, i.e.
        // immediately after signature(4) + version(1) + heap-header
        // address(sizeof_addr) + block offset(heap_off_size) = `prefix_min`.
        //
        // Filtered heaps store the checksum over the *decompressed* image;
        // since this reader does not run the direct-block filter pipeline,
        // verification is only performed for unfiltered heaps.
        let chk_off = prefix_min;
        if header.filter_len == 0 && buf.len() >= chk_off + 4 {
            let stored = u32::from_le_bytes([
                buf[chk_off],
                buf[chk_off + 1],
                buf[chk_off + 2],
                buf[chk_off + 3],
            ]);
            let mut image = buf.clone();
            image[chk_off..chk_off + 4].fill(0);
            let computed = checksum_metadata(&image);
            if stored != computed {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }
        payload_start += 4;
    }
    if payload_start >= buf.len() {
        return Ok(());
    }

    // Hand back the whole object area; the caller decodes packed objects.
    objects.push(buf[payload_start..].to_vec());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_of2_basic() {
        assert_eq!(log2_of2(1), 0);
        assert_eq!(log2_of2(2), 1);
        assert_eq!(log2_of2(512), 9);
        assert_eq!(log2_of2(4096), 12);
        assert_eq!(log2_of2(3), 0);
        assert_eq!(log2_of2(0), 0);
    }

    #[test]
    fn size_of_offset_bits_basic() {
        assert_eq!(size_of_offset_bits(0), 0);
        assert_eq!(size_of_offset_bits(8), 1);
        assert_eq!(size_of_offset_bits(9), 2);
        assert_eq!(size_of_offset_bits(16), 2);
        assert_eq!(size_of_offset_bits(17), 3);
    }

    #[test]
    fn bad_signature_rejected() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let buf = vec![0u8; FractalHeapHeader::base_size(&ctx)];
        let err = FractalHeapHeader::decode(&buf, &ctx).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn too_short_rejected() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let buf = vec![0u8; 8];
        let err = FractalHeapHeader::decode(&buf, &ctx).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }
}
