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

/// Number of bytes needed to store a value spanning `bits` bits.
fn size_of_offset_bits(bits: u16) -> u8 {
    bits.div_ceil(8) as u8
}

/// Read a little-endian unsigned integer of `n` bytes (1..=8) into a `u64`.
fn read_uint(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    let n = n.min(8).min(buf.len());
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
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
        pos += 4; // max_man_size (u32)
        pos += ss; // huge_next_id
        pos += sa; // huge_bt2_addr

        // "Managed" free-space info.
        pos += ss; // total_man_free
        pos += sa; // fs_addr

        // Statistics: man_size, man_alloc_size, man_iter_off, man_nobjs,
        // huge_size, huge_nobjs, tiny_size, tiny_nobjs.
        pos += ss; // man_size
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
            man_nobjs,
            table_width,
            start_block_size,
            max_direct_size,
            max_heap_size_bits,
            table_addr,
            curr_root_rows,
            heap_off_size,
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
        )?;
    }

    Ok(objects)
}

/// Recursively walk an indirect block, descending into child direct and
/// indirect blocks.
fn walk_indirect_block<R: BlockReader>(
    header: &FractalHeapHeader,
    ctx: &FormatContext,
    reader: &mut R,
    addr: u64,
    nrows: u32,
    objects: &mut Vec<Vec<u8>>,
    budget: &mut usize,
) -> FormatResult<()> {
    if addr == u64::MAX || nrows == 0 {
        return Ok(());
    }
    if *budget == 0 {
        return Err(FormatError::InvalidData(
            "fractal heap block budget exhausted".into(),
        ));
    }
    *budget -= 1;

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
            )?;
        }
    }

    Ok(())
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
