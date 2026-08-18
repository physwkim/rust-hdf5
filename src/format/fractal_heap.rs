//! Fractal heap reader (read-only).
//!
//! A fractal heap stores many small variable-length objects. In HDF5, dense
//! link storage for groups (used once a group exceeds the link phase-change
//! threshold) keeps each link as an encoded `Link` message inside a fractal
//! heap, indexed by a v2 B-tree referenced from the `Link Info` message.
//! Dense *attribute* storage works the same way, with the heap and name index
//! reached through the `Attribute Info` message.
//!
//! This module decodes the heap header (`FRHP`) and walks the managed
//! direct/indirect blocks (`FHDB` / `FHIB`) to recover the raw bytes of
//! every managed object. Callers decode those bytes themselves (e.g. as
//! `LinkMessage`). Objects too large for a managed block ("huge" objects,
//! at or above the header's `max_man_size`) live in their own file allocation
//! and are located through the heap's huge-object v2 B-tree.
//!
//! Layout references (libhdf5 2.x): `H5HFcache.c` (`H5HF__cache_hdr_deserialize`,
//! `H5HF__cache_dblock_deserialize`, `H5HF__cache_iblock_deserialize`,
//! `H5HF__dtable_decode`), `H5HFhdr.c`, `H5HFdtable.c`, `H5HFhuge.c`,
//! `H5HFman.c`.

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::checksum::checksum_metadata;
use crate::format::{BlockReader, FormatContext, FormatError, FormatResult, UNDEF_ADDR};

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

/// A fractal heap header, decoded from a file or built for one.
///
/// Every field libhdf5 stores is here, because this type is both ends of the
/// round trip: [`decode`](Self::decode) fills it from a file and
/// [`encode`](Self::encode) writes one back. The derived fields below the
/// stored ones are recomputed by [`derive`](Self::derive) either way, so a
/// heap this crate writes and a heap libhdf5 wrote parse through identical
/// arithmetic.
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    /// Length of a heap ID in bytes.
    pub id_len: u16,
    /// Encoded length of the I/O filter pipeline (0 = no filters).
    pub filter_len: u16,
    /// Whether direct blocks carry a trailing checksum.
    pub checksum_dblocks: bool,
    /// Largest object size the managed blocks accept; anything at or above it
    /// is stored as a "huge" object instead.
    pub max_man_size: u32,
    /// Last "huge" object ID handed out; the next one is this plus one
    /// (`H5HF__huge_new_id` pre-increments, so ID 0 is never used).
    pub huge_next_id: u64,
    /// v2 B-tree tracking the heap's "huge" objects (`UNDEF_ADDR` if none).
    pub huge_bt2_addr: u64,
    /// Free bytes inside allocated managed direct blocks.
    pub total_man_free: u64,
    /// Free-space manager tracking those bytes (`UNDEF_ADDR` if none).
    pub fs_addr: u64,
    /// Managed heap space the root block spans.
    pub man_size: u64,
    /// Bytes of managed direct blocks actually allocated in the file.
    pub man_alloc_size: u64,
    /// Heap-space offset the "next block" iterator sits at.
    pub man_iter_off: u64,
    /// Number of managed objects currently stored in the heap.
    pub man_nobjs: u64,
    /// Total size of the heap's "huge" objects.
    pub huge_size: u64,
    /// Number of "huge" objects in the heap.
    pub huge_nobjs: u64,
    /// Total size of the heap's "tiny" objects.
    pub tiny_size: u64,
    /// Number of "tiny" objects in the heap.
    pub tiny_nobjs: u64,
    /// Doubling-table: number of columns.
    pub table_width: u16,
    /// Doubling-table: starting (row 0) direct-block size in bytes.
    pub start_block_size: u64,
    /// Doubling-table: maximum direct-block size in bytes.
    pub max_direct_size: u64,
    /// Doubling-table: maximum heap size expressed as a count of bits.
    pub max_heap_size_bits: u16,
    /// Doubling-table: rows a root indirect block starts with.
    pub start_root_rows: u16,
    /// Doubling-table: file address of the root block.
    pub table_addr: u64,
    /// Doubling-table: current number of rows in the root indirect block
    /// (0 means the root block is a single direct block).
    pub curr_root_rows: u16,
    /// Bytes used to encode a block offset within the heap address space.
    pub heap_off_size: u8,
    /// Number of rows whose blocks are direct blocks.
    pub max_direct_rows: u32,
    /// Per-row direct-block sizes (length == `max_root_rows`).
    pub row_block_size: Vec<u64>,
    /// Per-row heap-space offset at which the row's first block starts
    /// (`H5HF_dtable_t::row_block_off`).
    pub row_block_off: Vec<u64>,
    /// Bytes used to encode a managed object's length inside a heap ID
    /// (`H5HF_hdr_t::heap_len_size`).
    pub heap_len_size: u8,
    /// Bytes used to encode a "huge" object's ID inside a heap ID
    /// (`H5HF_hdr_t::huge_id_size`).
    pub huge_id_size: u8,
    /// Whether a "huge" heap ID carries the object's address and length
    /// directly instead of a key into the huge-object B-tree
    /// (`H5HF_hdr_t::huge_ids_direct`).
    pub huge_ids_direct: bool,
}

/// `log2` of a power-of-two value. Returns 0 for inputs that are not a
/// positive power of two (defensive — real heaps always use powers of two).
fn log2_of2(n: u64) -> u32 {
    if n == 0 || (n & (n - 1)) != 0 {
        return 0;
    }
    n.trailing_zeros()
}

/// Number of bytes needed to store a value spanning `bits` bits.
fn size_of_offset_bits(bits: u16) -> u8 {
    bits.div_ceil(8) as u8
}

/// Bytes needed to encode any value up to `limit` (`H5VM_limit_enc_size`).
fn limit_enc_size(limit: u64) -> u8 {
    // H5VM_log2_gen is the floor log2 of the value (0 for 0).
    let log2 = if limit == 0 {
        0
    } else {
        63 - limit.leading_zeros()
    };
    (log2 / 8 + 1) as u8
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

    /// On-disk size of the header this crate writes (no filter pipeline).
    pub fn encoded_size(ctx: &FormatContext) -> usize {
        Self::base_size(ctx)
    }

    /// An empty heap with the given creation parameters.
    ///
    /// The caller fills in `table_addr`, `curr_root_rows` and the statistics
    /// once it has laid the blocks out — see
    /// [`build_heap`](crate::format::fractal_heap_write::build_heap).
    pub fn new(params: &HeapParams, ctx: &FormatContext) -> Self {
        let mut hdr = Self {
            id_len: params.id_len,
            filter_len: 0,
            checksum_dblocks: params.checksum_dblocks,
            max_man_size: params.max_man_size,
            huge_next_id: 0,
            huge_bt2_addr: UNDEF_ADDR,
            total_man_free: 0,
            fs_addr: UNDEF_ADDR,
            man_size: 0,
            man_alloc_size: 0,
            man_iter_off: 0,
            man_nobjs: 0,
            huge_size: 0,
            huge_nobjs: 0,
            tiny_size: 0,
            tiny_nobjs: 0,
            table_width: params.table_width,
            start_block_size: params.start_block_size,
            max_direct_size: params.max_direct_size,
            max_heap_size_bits: params.max_heap_size_bits,
            start_root_rows: params.start_root_rows,
            table_addr: UNDEF_ADDR,
            curr_root_rows: 0,
            // Filled in by `derive` below.
            heap_off_size: 0,
            max_direct_rows: 0,
            row_block_size: Vec::new(),
            row_block_off: Vec::new(),
            heap_len_size: 0,
            huge_id_size: 0,
            huge_ids_direct: false,
        };
        hdr.derive(ctx);
        hdr
    }

    /// Recompute every field the doubling-table parameters imply.
    ///
    /// None of these are stored: `H5HF__dtable_init` and
    /// `H5HF__hdr_finish_init_phase1/2` rebuild them from the parameters on
    /// every open, which is why a heap ID is only parseable alongside the
    /// header that produced it.
    fn derive(&mut self, ctx: &FormatContext) {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        let start_bits = log2_of2(self.start_block_size);
        let first_row_bits = start_bits + log2_of2(self.table_width as u64);
        let max_root_rows = (self.max_heap_size_bits as u32)
            .saturating_sub(first_row_bits)
            .saturating_add(1);
        let max_direct_bits = log2_of2(self.max_direct_size);
        self.max_direct_rows = max_direct_bits.saturating_sub(start_bits).saturating_add(2);
        self.heap_off_size = size_of_offset_bits(self.max_heap_size_bits);

        // Per-row direct-block sizes: row 0 == start, row 1 == start,
        // doubling from row 2 onward (H5HF__dtable_init).
        self.row_block_size = Vec::with_capacity(max_root_rows as usize);
        self.row_block_off = Vec::with_capacity(max_root_rows as usize);
        let mut tmp = self.start_block_size;
        let mut off = 0u64;
        for row in 0..max_root_rows {
            let size = if row == 0 { self.start_block_size } else { tmp };
            self.row_block_size.push(size);
            self.row_block_off.push(off);
            off = off.saturating_add(size.saturating_mul(self.table_width as u64));
            if row > 0 {
                tmp = tmp.saturating_mul(2);
            }
        }

        // Heap-ID field widths — H5HFhdr.c::H5HF__hdr_finish_init_phase1/2 and
        // H5HFhuge.c::H5HF__huge_init.
        let max_dir_blk_off_size = size_of_offset_bits(log2_of2(self.max_direct_size) as u16);
        self.heap_len_size = max_dir_blk_off_size.min(limit_enc_size(self.max_man_size as u64));
        let direct_huge_id_size = if self.filter_len > 0 {
            sa + ss + 4 + ss
        } else {
            sa + ss
        };
        self.huge_ids_direct =
            self.id_len >= 1 && direct_huge_id_size <= (self.id_len as usize - 1);
        self.huge_id_size = if self.huge_ids_direct {
            if self.filter_len > 0 {
                (sa + ss + ss) as u8
            } else {
                (sa + ss) as u8
            }
        } else if self.id_len as usize - 1 < 8 {
            (self.id_len - 1) as u8
        } else {
            8
        };
    }

    /// Serialize the header, mirroring `H5HF__cache_hdr_serialize`.
    ///
    /// Only unfiltered heaps are written, so the filter block that would
    /// follow the doubling table is absent and `filter_len` stays 0.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let mut buf = Vec::with_capacity(Self::base_size(ctx));

        buf.extend_from_slice(&FRHP_SIGNATURE);
        buf.push(0); // version
        buf.extend_from_slice(&self.id_len.to_le_bytes());
        buf.extend_from_slice(&self.filter_len.to_le_bytes());
        buf.push(if self.checksum_dblocks {
            HDR_FLAG_CHECKSUM_DBLOCKS
        } else {
            0
        });

        buf.extend_from_slice(&self.max_man_size.to_le_bytes());
        buf.extend_from_slice(&self.huge_next_id.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.huge_bt2_addr.to_le_bytes()[..sa]);

        buf.extend_from_slice(&self.total_man_free.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.fs_addr.to_le_bytes()[..sa]);

        for stat in [
            self.man_size,
            self.man_alloc_size,
            self.man_iter_off,
            self.man_nobjs,
            self.huge_size,
            self.huge_nobjs,
            self.tiny_size,
            self.tiny_nobjs,
        ] {
            buf.extend_from_slice(&stat.to_le_bytes()[..ss]);
        }

        buf.extend_from_slice(&self.table_width.to_le_bytes());
        buf.extend_from_slice(&self.start_block_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.max_direct_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.max_heap_size_bits.to_le_bytes());
        buf.extend_from_slice(&self.start_root_rows.to_le_bytes());
        buf.extend_from_slice(&self.table_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.curr_root_rows.to_le_bytes());

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        debug_assert_eq!(buf.len(), Self::base_size(ctx));
        buf
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
        let max_man_size = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        pos += 4;
        let huge_next_id = read_uint(&buf[pos..], ss);
        pos += ss;
        let huge_bt2_addr = read_uint(&buf[pos..], sa);
        pos += sa;

        // "Managed" free-space info.
        let total_man_free = read_uint(&buf[pos..], ss);
        pos += ss;
        let fs_addr = read_uint(&buf[pos..], sa);
        pos += sa;

        // Statistics: man_size, man_alloc_size, man_iter_off, man_nobjs,
        // huge_size, huge_nobjs, tiny_size, tiny_nobjs.
        let mut stats = [0u64; 8];
        for stat in &mut stats {
            *stat = read_uint(&buf[pos..], ss);
            pos += ss;
        }
        let [man_size, man_alloc_size, man_iter_off, man_nobjs, huge_size, huge_nobjs, tiny_size, tiny_nobjs] =
            stats;

        // Doubling-table info.
        let table_width = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let start_block_size = read_uint(&buf[pos..], ss);
        pos += ss;
        let max_direct_size = read_uint(&buf[pos..], ss);
        pos += ss;
        let max_heap_size_bits = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let start_root_rows = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
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

        let mut hdr = Self {
            id_len,
            filter_len,
            checksum_dblocks,
            max_man_size,
            huge_next_id,
            huge_bt2_addr,
            total_man_free,
            fs_addr,
            man_size,
            man_alloc_size,
            man_iter_off,
            man_nobjs,
            huge_size,
            huge_nobjs,
            tiny_size,
            tiny_nobjs,
            table_width,
            start_block_size,
            max_direct_size,
            max_heap_size_bits,
            start_root_rows,
            table_addr,
            curr_root_rows,
            heap_off_size: 0,
            max_direct_rows: 0,
            row_block_size: Vec::new(),
            row_block_off: Vec::new(),
            heap_len_size: 0,
            huge_id_size: 0,
            huge_ids_direct: false,
        };
        hdr.derive(ctx);
        Ok(hdr)
    }
}

/// Creation parameters for a new fractal heap (`H5HF_create_t`).
///
/// The defaults are the ones every object-header client uses
/// (`H5O_FHEAP_*` in H5Oprivate.h): dense attribute storage, dense link
/// storage and the shared-message heap all create their heaps with these, so
/// there is exactly one shape of heap to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeapParams {
    /// Heap ID width (`H5O_FHEAP_ID_LEN`).
    pub id_len: u16,
    /// Doubling-table columns.
    pub table_width: u16,
    /// Row-0 direct-block size.
    pub start_block_size: u64,
    /// Largest direct block the table reaches before rows become indirect.
    pub max_direct_size: u64,
    /// Heap address space, in bits.
    pub max_heap_size_bits: u16,
    /// Rows a root indirect block starts with.
    pub start_root_rows: u16,
    /// Whether direct blocks carry a checksum.
    pub checksum_dblocks: bool,
    /// Objects this size or larger bypass the managed blocks.
    pub max_man_size: u32,
}

impl HeapParams {
    /// The parameters `H5A__dense_create` uses for an object's attributes.
    pub fn object_header() -> Self {
        Self {
            id_len: 8,
            table_width: 4,
            start_block_size: 1024,
            max_direct_size: 64 * 1024,
            max_heap_size_bits: 40,
            start_root_rows: 1,
            checksum_dblocks: true,
            max_man_size: 4 * 1024,
        }
    }

    /// The parameters `H5G__dense_create` uses for a group's links
    /// (`H5G_FHEAP_*`, `H5Gdense.c`).
    ///
    /// A 32-bit heap address space and a 4 KiB `max_man_size` make the heap ID
    /// 1 + 4 + 2 = 7 bytes, which is what `H5G_DENSE_FHEAP_ID_LEN` asserts and
    /// what the name index's records carry.
    pub fn group_links() -> Self {
        Self {
            id_len: 7,
            table_width: 4,
            start_block_size: 512,
            max_direct_size: 64 * 1024,
            max_heap_size_bits: 32,
            start_root_rows: 1,
            checksum_dblocks: true,
            max_man_size: 4 * 1024,
        }
    }
}

/// One managed direct block, kept whole so that a heap offset can be resolved
/// against it.
#[derive(Debug, Clone)]
pub struct ManagedBlock {
    /// Heap-address-space offset of the block *image* (its signature byte),
    /// read from the block's own prefix. Managed heap IDs are offsets in this
    /// same space, so `id_offset - heap_offset` indexes straight into `image`
    /// (`H5HFman.c::H5HF__man_op_real`).
    pub heap_offset: u64,
    /// Offset within `image` at which the object area begins.
    pub payload_start: usize,
    /// The block image as read from the file.
    pub image: Vec<u8>,
}

impl ManagedBlock {
    /// The block's object area: one or more managed objects packed
    /// contiguously, followed by free space.
    pub fn payload(&self) -> &[u8] {
        &self.image[self.payload_start..]
    }

    /// The `len` bytes of the object stored at heap offset `offset`, or `None`
    /// if that span is not inside this block.
    fn object_at(&self, offset: u64, len: usize) -> Option<&[u8]> {
        let start = offset.checked_sub(self.heap_offset)? as usize;
        if start < self.payload_start {
            // Inside the block's prefix — not an object (H5HF__man_op_real
            // rejects the same case).
            return None;
        }
        let end = start.checked_add(len)?;
        self.image.get(start..end)
    }
}

/// One walk of a heap's doubling table.
#[derive(Debug, Default)]
struct HeapWalk {
    /// Every managed direct block reached, in walk order.
    blocks: Vec<ManagedBlock>,
    /// File extent of every block the walk descended through — indirect and
    /// direct alike — at the size the doubling table allocated it, which is
    /// what a caller freeing the heap must return. Taken from the table
    /// rather than from the bytes read back, so a block that reads short at
    /// end of file is still freed whole.
    extents: Vec<(u64, u64)>,
}

/// Walk a fractal heap and return every managed direct block.
pub fn collect_managed_blocks<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<ManagedBlock>> {
    if header.man_nobjs == 0 {
        return Ok(Vec::new());
    }
    Ok(walk_doubling_table(header, ctx, reader)?.blocks)
}

/// Every file extent a fractal heap occupies: its header block, the indirect
/// and direct blocks of its doubling table, each "huge" object, and the v2
/// B-tree that indexes those.
///
/// The single place that knows a heap's on-disk footprint, so that freeing one
/// enumerates the same blocks its writer allocated rather than re-deriving
/// them. A heap whose objects this crate can read is one whose extents it can
/// name; the cases it cannot (a filtered heap, or huge IDs that carry the
/// address inside the ID so the heap alone cannot list them) are refused here
/// rather than silently under-reported.
pub fn collect_heap_extents<R: BlockReader>(
    heap_addr: u64,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<(u64, u64)>> {
    use crate::format::chunk_index::btree_v2::collect_btree_v2_extents;

    // The header's on-disk size depends only on the address/length widths, so
    // a generous prefix read covers it. `decode` verifies the checksum over
    // exactly `encoded_size` bytes, so a header that decodes is one whose
    // block is that long — a filtered heap's longer header fails there.
    let buf = reader.read_block(heap_addr, 512)?;
    let header = FractalHeapHeader::decode(&buf, ctx)?;

    let mut extents = vec![(heap_addr, FractalHeapHeader::encoded_size(ctx) as u64)];
    extents.extend(walk_doubling_table(&header, ctx, reader)?.extents);

    if header.huge_nobjs > 0 {
        if header.huge_ids_direct {
            // The addresses live only inside the heap IDs, which are held by
            // whichever index references this heap — not by the heap. No
            // object-header client creates such a heap (their 8-byte IDs are
            // too narrow), so this is unreachable rather than a gap.
            return Err(FormatError::UnsupportedFeature(
                "freeing a fractal heap whose huge-object IDs are direct".into(),
            ));
        }
        for (addr, len, _id) in huge_object_records(&header, ctx, reader)? {
            extents.push((addr, len));
        }
        extents.extend(collect_btree_v2_extents(header.huge_bt2_addr, ctx, reader)?);
    }

    Ok(extents)
}

/// Walk the doubling table from its root block, collecting the managed direct
/// blocks and the file extent of every block visited.
fn walk_doubling_table<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<HeapWalk> {
    if header.table_addr == UNDEF_ADDR {
        return Ok(HeapWalk::default());
    }

    let mut walker = DoublingTableWalker {
        header,
        ctx,
        reader,
        walk: HeapWalk::default(),
        budget: MAX_BLOCKS,
    };

    if header.curr_root_rows == 0 {
        // Root block is a single direct block of `start_block_size`.
        walker.read_direct_block(header.table_addr, header.start_block_size as usize)?;
    } else {
        walker.walk_indirect_block(header.table_addr, header.curr_root_rows as u32, 0)?;
    }

    Ok(walker.walk)
}

/// The header, file context, reader and shared block budget a doubling-table
/// walk reads through, and the accumulator it fills. Held together so
/// [`walk_indirect_block`](Self::walk_indirect_block) and
/// [`read_direct_block`](Self::read_direct_block) take only what changes per
/// level: the block's address, its row count (indirect) or size (direct),
/// and — for the recursive indirect walk — how deep it has gone.
struct DoublingTableWalker<'a, R: BlockReader> {
    header: &'a FractalHeapHeader,
    ctx: &'a FormatContext,
    reader: &'a mut R,
    walk: HeapWalk,
    budget: usize,
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
    Ok(collect_managed_blocks(header, ctx, reader)?
        .into_iter()
        .map(|b| b.image[b.payload_start..].to_vec())
        .collect())
}

/// How a heap ID locates its object (`H5HFpkg.h`, `H5HF_ID_TYPE_*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeapId {
    /// Stored in a managed direct block at `offset`, `length` bytes long.
    Managed { offset: u64, length: u64 },
    /// Stored in its own file allocation, keyed by the huge-object B-tree.
    HugeIndirect { id: u64 },
    /// Stored in its own file allocation, whose address and length the ID
    /// carries directly.
    HugeDirect { address: u64, length: u64 },
    /// Stored inside the heap ID itself; `data` is the object.
    Tiny { data: Vec<u8> },
}

impl HeapId {
    /// Parse a heap ID against the header that defines its field widths.
    ///
    /// `id` is the fixed-width ID as it appears in an index record; only its
    /// first `header.id_len` bytes are meaningful.
    pub fn parse(id: &[u8], header: &FractalHeapHeader, ctx: &FormatContext) -> FormatResult<Self> {
        let id_len = header.id_len as usize;
        if id.len() < id_len || id_len == 0 {
            return Err(FormatError::BufferTooShort {
                needed: id_len,
                available: id.len(),
            });
        }
        let flags = id[0];
        let body = &id[1..id_len];
        match flags & ID_TYPE_MASK {
            ID_TYPE_MAN => {
                let off_size = header.heap_off_size as usize;
                let len_size = header.heap_len_size as usize;
                if body.len() < off_size + len_size {
                    return Err(FormatError::InvalidData(
                        "fractal heap managed ID shorter than its offset+length fields".into(),
                    ));
                }
                Ok(Self::Managed {
                    offset: read_uint(body, off_size),
                    length: read_uint(&body[off_size..], len_size),
                })
            }
            ID_TYPE_HUGE => {
                let sa = ctx.sizeof_addr as usize;
                let ss = ctx.sizeof_size as usize;
                if header.huge_ids_direct {
                    if body.len() < sa + ss {
                        return Err(FormatError::InvalidData(
                            "fractal heap direct huge ID shorter than address+length".into(),
                        ));
                    }
                    Ok(Self::HugeDirect {
                        address: read_uint(body, sa),
                        length: read_uint(&body[sa..], ss),
                    })
                } else {
                    let n = (header.huge_id_size as usize).min(body.len());
                    Ok(Self::HugeIndirect {
                        id: read_uint(body, n),
                    })
                }
            }
            ID_TYPE_TINY => {
                // Length lives in the low nibble of the flags byte, or in the
                // next byte for an "extended" tiny object
                // (H5HFtiny.c::H5HF__tiny_op_real).
                let (len, data) = if header.id_len <= 16 {
                    ((flags & ID_TINY_LEN_MASK) as usize + 1, body)
                } else {
                    if body.is_empty() {
                        return Err(FormatError::InvalidData(
                            "fractal heap extended tiny ID has no length byte".into(),
                        ));
                    }
                    (
                        (((flags & ID_TINY_LEN_MASK) as usize) << 8 | body[0] as usize) + 1,
                        &body[1..],
                    )
                };
                if len > data.len() {
                    return Err(FormatError::InvalidData(
                        "fractal heap tiny ID length exceeds the ID itself".into(),
                    ));
                }
                Ok(Self::Tiny {
                    data: data[..len].to_vec(),
                })
            }
            _ => Err(FormatError::UnsupportedFeature(
                "reserved fractal heap object type".into(),
            )),
        }
    }
}

const ID_TYPE_MASK: u8 = 0x30;
const ID_TYPE_MAN: u8 = 0x00;
const ID_TYPE_HUGE: u8 = 0x10;
const ID_TYPE_TINY: u8 = 0x20;
const ID_TINY_LEN_MASK: u8 = 0x0F;

/// Read the object a heap ID points at.
///
/// `blocks` are the heap's managed direct blocks as returned by
/// [`collect_managed_blocks`]; huge objects are located through the heap's
/// huge-object B-tree and read straight from the file.
pub fn read_heap_object<R: BlockReader>(
    id: &HeapId,
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    blocks: &[ManagedBlock],
    reader: &mut R,
) -> FormatResult<Vec<u8>> {
    match *id {
        HeapId::Managed { offset, length } => {
            let length = usize::try_from(length).map_err(|_| {
                FormatError::InvalidData("fractal heap object length overflows usize".into())
            })?;
            blocks
                .iter()
                .find_map(|b| b.object_at(offset, length))
                .map(|s| s.to_vec())
                .ok_or_else(|| {
                    FormatError::InvalidData(format!(
                        "fractal heap offset {offset} is outside every managed block"
                    ))
                })
        }
        HeapId::Tiny { ref data } => Ok(data.clone()),
        HeapId::HugeDirect { address, length } => read_span(reader, address, length),
        HeapId::HugeIndirect { id } => {
            let (address, length) = lookup_huge_object(id, header, ctx, reader)?;
            read_span(reader, address, length)
        }
    }
}

fn read_span<R: BlockReader>(reader: &mut R, address: u64, length: u64) -> FormatResult<Vec<u8>> {
    let length = usize::try_from(length).map_err(|_| {
        FormatError::InvalidData("fractal heap huge object length overflows usize".into())
    })?;
    let buf = reader.read_block(address, length)?;
    need(&buf, 0, length)?;
    Ok(buf)
}

/// Resolve a "huge" object ID to its (address, length) through the heap's
/// huge-object v2 B-tree (`H5HFbtree2.c`, record class
/// `H5B2_FHEAP_HUGE_INDIR_ID`: address, length, ID).
fn lookup_huge_object<R: BlockReader>(
    target_id: u64,
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<(u64, u64)> {
    for (addr, len, id) in huge_object_records(header, ctx, reader)? {
        if id == target_id {
            return Ok((addr, len));
        }
    }
    Err(FormatError::InvalidData(format!(
        "huge object ID {target_id} is not in the heap's huge-object B-tree"
    )))
}

/// Every "huge" object the heap's huge-object B-tree records, as
/// `(address, length, id)`.
///
/// One decode of that index serves both readers: resolving a single ID to its
/// object, and listing every object so the heap's file space can be freed.
fn huge_object_records<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<(u64, u64, u64)>> {
    use crate::format::chunk_index::btree_v2::{collect_btree_v2_records, Bt2Header};

    if header.huge_bt2_addr == UNDEF_ADDR {
        return Err(FormatError::InvalidData(
            "fractal heap has a huge object ID but no huge-object B-tree".into(),
        ));
    }
    let hdr_buf = reader.read_block(header.huge_bt2_addr, 256)?;
    let bt2 = Bt2Header::decode(&hdr_buf, ctx)?;
    let records = collect_btree_v2_records(&bt2, ctx, reader)?;

    let sa = ctx.sizeof_addr as usize;
    let ss = ctx.sizeof_size as usize;
    // Filtered huge records carry a filter mask and a de-filtered size between
    // the length and the ID; this reader does not run the direct-block filter
    // pipeline, so it declines them rather than misreading the ID field.
    if header.filter_len > 0 {
        return Err(FormatError::UnsupportedFeature(
            "filtered fractal heap huge objects".into(),
        ));
    }
    let rec_size = bt2.record_size as usize;
    if rec_size < sa + ss + ss {
        return Err(FormatError::InvalidData(
            "huge-object B-tree record is too small for address+length+ID".into(),
        ));
    }
    Ok(records
        .chunks_exact(rec_size)
        .map(|rec| {
            (
                read_uint(rec, sa),
                read_uint(&rec[sa..], ss),
                read_uint(&rec[sa + ss..], ss),
            )
        })
        .collect())
}

impl<R: BlockReader> DoublingTableWalker<'_, R> {
    /// Recursively walk an indirect block, descending into child direct and
    /// indirect blocks.
    ///
    /// `header`, `ctx`, `reader`, `walk` and `budget` are constant threads for
    /// the whole walk, so only `addr`, `nrows` and `depth` — what changes per
    /// level — are parameters; the rest live on `self`.
    fn walk_indirect_block(&mut self, addr: u64, nrows: u32, depth: usize) -> FormatResult<()> {
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
        if self.budget == 0 {
            return Err(FormatError::InvalidData(
                "fractal heap block budget exhausted".into(),
            ));
        }
        self.budget -= 1;

        let header = self.header;
        let ctx = self.ctx;
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
        self.walk.extents.push((addr, block_len as u64));

        let buf = self.reader.read_block(addr, block_len)?;
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

        // Skip prefix: signature(4) + version(1) + heap header address(sa)
        //              + block offset(heap_off_size).
        let mut pos = 4 + 1 + sa + header.heap_off_size as usize;

        for entry in 0..n_entries {
            let row = entry / width;
            let child_addr = read_uint(&buf[pos..], sa);
            pos += sa;
            if header.filter_len > 0 && row < header.max_direct_rows as usize {
                // Filtered direct-block entries carry size + filter mask.
                pos += ctx.sizeof_size as usize + 4;
            }

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
                self.read_direct_block(child_addr, size)?;
            } else {
                // Indirect-block child. Its row count is derived from the row's
                // block size (see H5HFhdr.c / H5HF__dtable_size_to_rows).
                let block_size = header
                    .row_block_size
                    .get(row)
                    .copied()
                    .unwrap_or(header.start_block_size);
                let child_nrows = indirect_nrows(header, block_size);
                self.walk_indirect_block(child_addr, child_nrows, depth + 1)?;
            }
        }

        Ok(())
    }
}

/// Number of rows in a child indirect block whose row-block size is
/// `block_size` (H5HF__dtable_size_to_rows).
pub(crate) fn indirect_nrows(header: &FractalHeapHeader, block_size: u64) -> u32 {
    let start_bits = log2_of2(header.start_block_size);
    let first_row_bits = start_bits + log2_of2(header.table_width as u64);
    let size_log2 = log2_of2(block_size);
    size_log2.saturating_sub(first_row_bits).saturating_add(1)
}

impl<R: BlockReader> DoublingTableWalker<'_, R> {
    /// Read a direct block and append it to `self.walk.blocks`.
    fn read_direct_block(&mut self, addr: u64, size: usize) -> FormatResult<()> {
        if addr == UNDEF_ADDR || size == 0 {
            return Ok(());
        }
        if self.budget == 0 {
            return Err(FormatError::InvalidData(
                "fractal heap block budget exhausted".into(),
            ));
        }
        self.budget -= 1;
        self.walk.extents.push((addr, size as u64));

        let header = self.header;
        let ctx = self.ctx;
        let sa = ctx.sizeof_addr as usize;
        let buf = self.reader.read_block(addr, size)?;

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

        // The block's own offset in the heap address space, from its prefix: this
        // is what a managed heap ID is relative to.
        let heap_offset = read_uint(&buf[4 + 1 + sa..], header.heap_off_size as usize);

        self.walk.blocks.push(ManagedBlock {
            heap_offset,
            payload_start,
            image: buf,
        });
        Ok(())
    }
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
