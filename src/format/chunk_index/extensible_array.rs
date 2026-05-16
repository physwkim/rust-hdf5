//! Extensible Array (EA) chunk index structures for HDF5.
//!
//! Implements the on-disk format for the extensible array used to index
//! chunked datasets with one unlimited dimension (the typical SWMR use case).
//!
//! Structures:
//!   - Header (EAHD): metadata and statistics about the extensible array
//!   - Index Block (EAIB): holds direct chunk addresses and pointers to data/super blocks
//!   - Data Block (EADB): holds additional chunk addresses when the index block is full

use crate::format::checksum::checksum_metadata;
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Signature for the extensible array header.
pub const EAHD_SIGNATURE: [u8; 4] = *b"EAHD";
/// Signature for the extensible array index block.
pub const EAIB_SIGNATURE: [u8; 4] = *b"EAIB";
/// Signature for the extensible array data block.
pub const EADB_SIGNATURE: [u8; 4] = *b"EADB";
/// Signature for the extensible array super block.
pub const EASB_SIGNATURE: [u8; 4] = *b"EASB";

/// Extensible array version.
pub const EA_VERSION: u8 = 0;

/// Class ID for unfiltered chunks (H5EA_CLS_CHUNK).
pub const EA_CLS_CHUNK: u8 = 0;
/// Class ID for filtered chunks (H5EA_CLS_FILT_CHUNK).
pub const EA_CLS_FILT_CHUNK: u8 = 1;

/// A filtered chunk element stored in the extensible array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilteredChunkEntry {
    /// Address of the compressed chunk data in the file.
    pub addr: u64,
    /// Size of the compressed chunk in bytes.
    pub nbytes: u64,
    /// Filter mask — bit N set means filter N was NOT applied.
    pub filter_mask: u32,
}

impl FilteredChunkEntry {
    pub fn undef() -> Self {
        Self {
            addr: UNDEF_ADDR,
            nbytes: 0,
            filter_mask: 0,
        }
    }

    pub fn is_undef(&self) -> bool {
        self.addr == UNDEF_ADDR
    }

    /// Compute raw element size on disk: sizeof_addr + chunk_size_len + 4.
    pub fn raw_size(sizeof_addr: u8, chunk_size_len: u8) -> u8 {
        sizeof_addr + chunk_size_len + 4
    }

    /// Encode a single filtered entry.
    pub fn encode(&self, sizeof_addr: usize, chunk_size_len: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(sizeof_addr + chunk_size_len + 4);
        buf.extend_from_slice(&self.addr.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&self.nbytes.to_le_bytes()[..chunk_size_len]);
        buf.extend_from_slice(&self.filter_mask.to_le_bytes());
        buf
    }

    /// Decode a single filtered entry.
    pub fn decode(buf: &[u8], sizeof_addr: usize, chunk_size_len: usize) -> Self {
        let addr = read_addr(buf, sizeof_addr);
        let nbytes = read_size(&buf[sizeof_addr..], chunk_size_len);
        let off = sizeof_addr + chunk_size_len;
        let filter_mask = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        Self {
            addr,
            nbytes,
            filter_mask,
        }
    }
}

/// Compute chunk_size_len: bytes needed to encode the uncompressed chunk size.
/// Formula from HDF5 C: 1 + (log2(chunk_size) + 8) / 8, capped at 8.
pub fn compute_chunk_size_len(uncompressed_chunk_bytes: u64) -> u8 {
    if uncompressed_chunk_bytes == 0 {
        return 1;
    }
    let log2 = 63 - uncompressed_chunk_bytes.leading_zeros();
    let len = 1 + (log2 + 8) / 8;
    std::cmp::min(len, 8) as u8
}

/// Extensible array header.
///
/// On-disk layout:
/// ```text
/// "EAHD"(4) + version=0(1) + class_id(1)
/// + raw_elmt_size(1) + max_nelmts_bits(1) + idx_blk_elmts(1)
/// + data_blk_min_elmts(1) + sup_blk_min_data_ptrs(1)
/// + max_dblk_page_nelmts_bits(1)
/// + 6 statistics (each sizeof_size bytes)
/// + idx_blk_addr (sizeof_addr)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensibleArrayHeader {
    pub class_id: u8,
    pub raw_elmt_size: u8,
    pub max_nelmts_bits: u8,
    pub idx_blk_elmts: u8,
    pub data_blk_min_elmts: u8,
    pub sup_blk_min_data_ptrs: u8,
    pub max_dblk_page_nelmts_bits: u8,
    // statistics
    pub num_sblks_created: u64,
    pub size_sblks_created: u64,
    pub num_dblks_created: u64,
    pub size_dblks_created: u64,
    pub max_idx_set: u64,
    pub num_elmts_realized: u64,
    pub idx_blk_addr: u64,
}

impl ExtensibleArrayHeader {
    /// Create a new header for unfiltered chunk indexing.
    pub fn new_for_chunks(ctx: &FormatContext) -> Self {
        Self {
            class_id: EA_CLS_CHUNK,
            raw_elmt_size: ctx.sizeof_addr,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            data_blk_min_elmts: 16,
            sup_blk_min_data_ptrs: 4,
            max_dblk_page_nelmts_bits: 10,
            num_sblks_created: 0,
            size_sblks_created: 0,
            num_dblks_created: 0,
            size_dblks_created: 0,
            max_idx_set: 0,
            num_elmts_realized: 0,
            idx_blk_addr: UNDEF_ADDR,
        }
    }

    /// Create a new header for filtered (compressed) chunk indexing.
    pub fn new_for_filtered_chunks(ctx: &FormatContext, chunk_size_len: u8) -> Self {
        Self {
            class_id: EA_CLS_FILT_CHUNK,
            raw_elmt_size: FilteredChunkEntry::raw_size(ctx.sizeof_addr, chunk_size_len),
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            data_blk_min_elmts: 16,
            sup_blk_min_data_ptrs: 4,
            max_dblk_page_nelmts_bits: 10,
            num_sblks_created: 0,
            size_sblks_created: 0,
            num_dblks_created: 0,
            size_dblks_created: 0,
            max_idx_set: 0,
            num_elmts_realized: 0,
            idx_blk_addr: UNDEF_ADDR,
        }
    }

    /// Compute the encoded size (for pre-allocation).
    pub fn encoded_size(&self, ctx: &FormatContext) -> usize {
        let ss = ctx.sizeof_size as usize;
        let sa = ctx.sizeof_addr as usize;
        // signature(4) + version(1) + class_id(1)
        // + raw_elmt_size(1) + max_nelmts_bits(1) + idx_blk_elmts(1)
        // + data_blk_min_elmts(1) + sup_blk_min_data_ptrs(1)
        // + max_dblk_page_nelmts_bits(1)
        // + 6 * sizeof_size (statistics)
        // + sizeof_addr (idx_blk_addr)
        // + checksum(4)
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * ss + sa + 4
    }

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let ss = ctx.sizeof_size as usize;
        let sa = ctx.sizeof_addr as usize;
        let size = self.encoded_size(ctx);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&EAHD_SIGNATURE);
        buf.push(EA_VERSION);
        buf.push(self.class_id);
        buf.push(self.raw_elmt_size);
        buf.push(self.max_nelmts_bits);
        buf.push(self.idx_blk_elmts);
        buf.push(self.data_blk_min_elmts);
        buf.push(self.sup_blk_min_data_ptrs);
        buf.push(self.max_dblk_page_nelmts_bits);

        // Statistics
        buf.extend_from_slice(&self.num_sblks_created.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.size_sblks_created.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.num_dblks_created.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.size_dblks_created.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.max_idx_set.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.num_elmts_realized.to_le_bytes()[..ss]);

        // Index block address
        buf.extend_from_slice(&self.idx_blk_addr.to_le_bytes()[..sa]);

        // Checksum
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let ss = ctx.sizeof_size as usize;
        let sa = ctx.sizeof_addr as usize;
        let min_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * ss + sa + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != EAHD_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != EA_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        // Verify checksum
        let data_end = min_size - 4;
        let stored_cksum = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        let mut pos = 5;
        let class_id = buf[pos];
        pos += 1;
        let raw_elmt_size = buf[pos];
        pos += 1;
        let max_nelmts_bits = buf[pos];
        pos += 1;
        let idx_blk_elmts = buf[pos];
        pos += 1;
        let data_blk_min_elmts = buf[pos];
        pos += 1;
        let sup_blk_min_data_ptrs = buf[pos];
        pos += 1;
        let max_dblk_page_nelmts_bits = buf[pos];
        pos += 1;

        let num_sblks_created = read_size(&buf[pos..], ss);
        pos += ss;
        let size_sblks_created = read_size(&buf[pos..], ss);
        pos += ss;
        let num_dblks_created = read_size(&buf[pos..], ss);
        pos += ss;
        let size_dblks_created = read_size(&buf[pos..], ss);
        pos += ss;
        let max_idx_set = read_size(&buf[pos..], ss);
        pos += ss;
        let num_elmts_realized = read_size(&buf[pos..], ss);
        pos += ss;

        let idx_blk_addr = read_addr(&buf[pos..], sa);

        Ok(Self {
            class_id,
            raw_elmt_size,
            max_nelmts_bits,
            idx_blk_elmts,
            data_blk_min_elmts,
            sup_blk_min_data_ptrs,
            max_dblk_page_nelmts_bits,
            num_sblks_created,
            size_sblks_created,
            num_dblks_created,
            size_dblks_created,
            max_idx_set,
            num_elmts_realized,
            idx_blk_addr,
        })
    }
}

/// Extensible array index block.
///
/// On-disk layout:
/// ```text
/// "EAIB"(4) + version=0(1) + class_id(1)
/// + header_addr(sizeof_addr)
/// + elements (idx_blk_elmts * raw_elmt_size bytes)
/// + data_block_addresses (ndblk_addrs * sizeof_addr)
/// + super_block_addresses (nsblk_addrs * sizeof_addr)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensibleArrayIndexBlock {
    pub class_id: u8,
    pub header_addr: u64,
    /// Direct chunk addresses in the index block.
    pub elements: Vec<u64>,
    /// Data block addresses.
    pub dblk_addrs: Vec<u64>,
    /// Super block addresses.
    pub sblk_addrs: Vec<u64>,
}

/// Filtered variant of the extensible array index block.
///
/// Stores `FilteredChunkEntry` elements instead of raw addresses.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredIndexBlock {
    pub class_id: u8,
    pub header_addr: u64,
    pub elements: Vec<FilteredChunkEntry>,
    pub dblk_addrs: Vec<u64>,
    pub sblk_addrs: Vec<u64>,
}

impl FilteredIndexBlock {
    pub fn new(
        header_addr: u64,
        idx_blk_elmts: u8,
        ndblk_addrs: usize,
        nsblk_addrs: usize,
    ) -> Self {
        Self {
            class_id: EA_CLS_FILT_CHUNK,
            header_addr,
            elements: vec![FilteredChunkEntry::undef(); idx_blk_elmts as usize],
            dblk_addrs: vec![UNDEF_ADDR; ndblk_addrs],
            sblk_addrs: vec![UNDEF_ADDR; nsblk_addrs],
        }
    }

    pub fn encode(&self, ctx: &FormatContext, chunk_size_len: u8) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let elmt_size = FilteredChunkEntry::raw_size(ctx.sizeof_addr, chunk_size_len) as usize;
        let size = 4
            + 1
            + 1
            + sa
            + self.elements.len() * elmt_size
            + self.dblk_addrs.len() * sa
            + self.sblk_addrs.len() * sa
            + 4;
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&EAIB_SIGNATURE);
        buf.push(EA_VERSION);
        buf.push(self.class_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);

        for elem in &self.elements {
            buf.extend_from_slice(&elem.encode(sa, chunk_size_len as usize));
        }
        for &addr in &self.dblk_addrs {
            buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }
        for &addr in &self.sblk_addrs {
            buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        idx_blk_elmts: usize,
        ndblk_addrs: usize,
        nsblk_addrs: usize,
        chunk_size_len: u8,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let elmt_size = FilteredChunkEntry::raw_size(ctx.sizeof_addr, chunk_size_len) as usize;
        let min_size =
            4 + 1 + 1 + sa + idx_blk_elmts * elmt_size + ndblk_addrs * sa + nsblk_addrs * sa + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }
        if buf[0..4] != EAIB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }
        if buf[4] != EA_VERSION {
            return Err(FormatError::InvalidVersion(buf[4]));
        }

        let data_end = min_size - 4;
        let stored = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed = checksum_metadata(&buf[..data_end]);
        if stored != computed {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }

        let class_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;

        let mut elements = Vec::with_capacity(idx_blk_elmts);
        for _ in 0..idx_blk_elmts {
            elements.push(FilteredChunkEntry::decode(
                &buf[pos..],
                sa,
                chunk_size_len as usize,
            ));
            pos += elmt_size;
        }
        let mut dblk_addrs = Vec::with_capacity(ndblk_addrs);
        for _ in 0..ndblk_addrs {
            dblk_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }
        let mut sblk_addrs = Vec::with_capacity(nsblk_addrs);
        for _ in 0..nsblk_addrs {
            sblk_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        Ok(Self {
            class_id,
            header_addr,
            elements,
            dblk_addrs,
            sblk_addrs,
        })
    }
}

/// Filtered variant of the extensible array data block.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredDataBlock {
    pub class_id: u8,
    pub header_addr: u64,
    pub block_offset: u64,
    pub elements: Vec<FilteredChunkEntry>,
}

impl FilteredDataBlock {
    pub fn new(header_addr: u64, block_offset: u64, nelmts: usize) -> Self {
        Self {
            class_id: EA_CLS_FILT_CHUNK,
            header_addr,
            block_offset,
            elements: vec![FilteredChunkEntry::undef(); nelmts],
        }
    }

    pub fn encode(&self, ctx: &FormatContext, max_nelmts_bits: u8, chunk_size_len: u8) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let bo_size = ExtensibleArrayDataBlock::block_offset_size(max_nelmts_bits);
        let elmt_size = FilteredChunkEntry::raw_size(ctx.sizeof_addr, chunk_size_len) as usize;
        let size = 4 + 1 + 1 + sa + bo_size + self.elements.len() * elmt_size + 4;
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&EADB_SIGNATURE);
        buf.push(EA_VERSION);
        buf.push(self.class_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.block_offset.to_le_bytes()[..bo_size]);

        for elem in &self.elements {
            buf.extend_from_slice(&elem.encode(sa, chunk_size_len as usize));
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        max_nelmts_bits: u8,
        nelmts: usize,
        chunk_size_len: u8,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let bo_size = ExtensibleArrayDataBlock::block_offset_size(max_nelmts_bits);
        let elmt_size = FilteredChunkEntry::raw_size(ctx.sizeof_addr, chunk_size_len) as usize;
        let min_size = 4 + 1 + 1 + sa + bo_size + nelmts * elmt_size + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }
        if buf[0..4] != EADB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }
        if buf[4] != EA_VERSION {
            return Err(FormatError::InvalidVersion(buf[4]));
        }

        let data_end = min_size - 4;
        let stored = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed = checksum_metadata(&buf[..data_end]);
        if stored != computed {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }

        let class_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;
        let block_offset = read_size(&buf[pos..], bo_size);
        pos += bo_size;

        let mut elements = Vec::with_capacity(nelmts);
        for _ in 0..nelmts {
            elements.push(FilteredChunkEntry::decode(
                &buf[pos..],
                sa,
                chunk_size_len as usize,
            ));
            pos += elmt_size;
        }

        Ok(Self {
            class_id,
            header_addr,
            block_offset,
            elements,
        })
    }
}

impl ExtensibleArrayIndexBlock {
    /// Create a new empty index block.
    pub fn new(
        header_addr: u64,
        idx_blk_elmts: u8,
        ndblk_addrs: usize,
        nsblk_addrs: usize,
    ) -> Self {
        Self {
            class_id: EA_CLS_CHUNK,
            header_addr,
            elements: vec![UNDEF_ADDR; idx_blk_elmts as usize],
            dblk_addrs: vec![UNDEF_ADDR; ndblk_addrs],
            sblk_addrs: vec![UNDEF_ADDR; nsblk_addrs],
        }
    }

    /// Compute the encoded size.
    pub fn encoded_size(&self, ctx: &FormatContext) -> usize {
        let sa = ctx.sizeof_addr as usize;
        // signature(4) + version(1) + class_id(1)
        // + header_addr(sa)
        // + elements(n * sa)
        // + dblk_addrs(n * sa)
        // + sblk_addrs(n * sa)
        // + checksum(4)
        4 + 1
            + 1
            + sa
            + self.elements.len() * sa
            + self.dblk_addrs.len() * sa
            + self.sblk_addrs.len() * sa
            + 4
    }

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let size = self.encoded_size(ctx);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&EAIB_SIGNATURE);
        buf.push(EA_VERSION);
        buf.push(self.class_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);

        for &elem in &self.elements {
            buf.extend_from_slice(&elem.to_le_bytes()[..sa]);
        }

        for &addr in &self.dblk_addrs {
            buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }

        for &addr in &self.sblk_addrs {
            buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        idx_blk_elmts: usize,
        ndblk_addrs: usize,
        nsblk_addrs: usize,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let min_size =
            4 + 1 + 1 + sa + idx_blk_elmts * sa + ndblk_addrs * sa + nsblk_addrs * sa + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != EAIB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != EA_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        // Verify checksum
        let data_end = min_size - 4;
        let stored_cksum = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        let class_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;

        let mut elements = Vec::with_capacity(idx_blk_elmts);
        for _ in 0..idx_blk_elmts {
            elements.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        let mut dblk_addrs = Vec::with_capacity(ndblk_addrs);
        for _ in 0..ndblk_addrs {
            dblk_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        let mut sblk_addrs = Vec::with_capacity(nsblk_addrs);
        for _ in 0..nsblk_addrs {
            sblk_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        Ok(Self {
            class_id,
            header_addr,
            elements,
            dblk_addrs,
            sblk_addrs,
        })
    }
}

/// Extensible array data block.
///
/// On-disk layout:
/// ```text
/// "EADB"(4) + version=0(1) + class_id(1)
/// + header_addr(sizeof_addr)
/// + block_offset (variable length)
/// + elements(nelmts * raw_elmt_size)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensibleArrayDataBlock {
    pub class_id: u8,
    pub header_addr: u64,
    pub block_offset: u64,
    /// Chunk addresses.
    pub elements: Vec<u64>,
}

impl ExtensibleArrayDataBlock {
    /// Create a new empty data block.
    pub fn new(header_addr: u64, block_offset: u64, nelmts: usize) -> Self {
        Self {
            class_id: EA_CLS_CHUNK,
            header_addr,
            block_offset,
            elements: vec![UNDEF_ADDR; nelmts],
        }
    }

    /// Number of bytes needed for the block_offset field.
    pub fn block_offset_size(max_nelmts_bits: u8) -> usize {
        std::cmp::max(1, (max_nelmts_bits as usize).div_ceil(8))
    }

    /// Compute the encoded size.
    pub fn encoded_size(&self, ctx: &FormatContext, max_nelmts_bits: u8) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let bo_size = Self::block_offset_size(max_nelmts_bits);
        // signature(4) + version(1) + class_id(1)
        // + header_addr(sa) + block_offset(bo_size)
        // + elements(n * sa) + checksum(4)
        4 + 1 + 1 + sa + bo_size + self.elements.len() * sa + 4
    }

    pub fn encode(&self, ctx: &FormatContext, max_nelmts_bits: u8) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let bo_size = Self::block_offset_size(max_nelmts_bits);
        let size = self.encoded_size(ctx, max_nelmts_bits);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&EADB_SIGNATURE);
        buf.push(EA_VERSION);
        buf.push(self.class_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.block_offset.to_le_bytes()[..bo_size]);

        for &elem in &self.elements {
            buf.extend_from_slice(&elem.to_le_bytes()[..sa]);
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        max_nelmts_bits: u8,
        nelmts: usize,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let bo_size = Self::block_offset_size(max_nelmts_bits);
        let min_size = 4 + 1 + 1 + sa + bo_size + nelmts * sa + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != EADB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != EA_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        // Verify checksum
        let data_end = min_size - 4;
        let stored_cksum = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        let class_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;
        let block_offset = read_size(&buf[pos..], bo_size);
        pos += bo_size;

        let mut elements = Vec::with_capacity(nelmts);
        for _ in 0..nelmts {
            elements.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        Ok(Self {
            class_id,
            header_addr,
            block_offset,
            elements,
        })
    }
}

// ========================================================================= helpers

fn read_addr(buf: &[u8], n: usize) -> u64 {
    if buf[..n].iter().all(|&b| b == 0xFF) {
        UNDEF_ADDR
    } else {
        let mut tmp = [0u8; 8];
        tmp[..n].copy_from_slice(&buf[..n]);
        u64::from_le_bytes(tmp)
    }
}

fn read_size(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

/// Compute ndblk_addrs for the index block given the default params.
///
/// For sup_blk_min_data_ptrs = K:
///   ndblk_addrs = 2 * (K - 1)
pub fn compute_ndblk_addrs(sup_blk_min_data_ptrs: u8) -> usize {
    2 * (sup_blk_min_data_ptrs as usize - 1)
}

/// `log2` of a power of two.
fn log2_pow2(n: u64) -> u32 {
    debug_assert!(n.is_power_of_two());
    n.trailing_zeros()
}

/// Floor of `log2(n)` for `n >= 1`.
fn log2_floor(n: u64) -> u32 {
    debug_assert!(n >= 1);
    63 - n.leading_zeros()
}

/// Layout of one extensible-array super block (`H5EA_sblk_info_t`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EaSblkInfo {
    /// Number of data blocks in this super block.
    pub ndblks: u64,
    /// Number of elements in each data block of this super block.
    pub dblk_nelmts: u64,
    /// Index of the first element in this super block (excludes `idx_blk_elmts`).
    pub start_idx: u64,
    /// Global index of the first data block in this super block.
    pub start_dblk: u64,
}

/// Where a chunk lives within the extensible array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EaLoc {
    /// Stored directly in the index block at `elem`.
    Index { elem: usize },
    /// Stored in a data block.
    Dblk(EaChunkLoc),
}

/// Location of a chunk that lives in an EA data block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EaChunkLoc {
    /// Owning super-block index.
    pub sblk_idx: usize,
    /// Elements per data block in that super block.
    pub dblk_nelmts: u64,
    /// Element offset of the chunk within its data block.
    pub offset_in_dblk: u64,
    /// `block_offset` value to stamp into the data block header.
    pub dblk_block_offset: u64,
    /// Whether the data block exceeds the page size (paged — unsupported).
    pub paged: bool,
    /// How the data block address is reached.
    pub path: EaDblkPath,
}

/// How an EA data block's address is reached from the index block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EaDblkPath {
    /// Address is `index_block.dblk_addrs[idx]`.
    Direct { idx: usize },
    /// Address is `super_block(index_block.sblk_addrs[sblk_off]).dblk_addrs[local_dblk]`.
    ViaSblk {
        sblk_off: usize,
        local_dblk: usize,
        ndblks_in_sblk: usize,
        /// `block_offset` value for the super block header.
        sblk_block_offset: u64,
    },
}

/// Extensible-array geometry derived from the creation parameters, matching
/// the libhdf5 on-disk layout (`H5EA__hdr_init`, `H5EAiblock.c`).
#[derive(Debug, Clone)]
pub struct EaGeometry {
    pub idx_blk_elmts: u64,
    pub data_blk_min_elmts: u64,
    /// Elements per data block page (paging threshold).
    pub dblk_page_nelmts: u64,
    /// Super blocks whose data-block addresses live in the index block.
    pub iblock_nsblks: usize,
    /// Data-block address slots in the index block.
    pub ndblk_addrs: usize,
    /// Super-block address slots in the index block.
    pub nsblk_addrs: usize,
    /// Per-super-block layout, length = total super-block count.
    pub sblk: Vec<EaSblkInfo>,
}

impl EaGeometry {
    /// Derive the geometry from the EA creation parameters.
    pub fn new(
        idx_blk_elmts: u8,
        data_blk_min_elmts: u8,
        sup_blk_min_data_ptrs: u8,
        max_nelmts_bits: u8,
        max_dblk_page_nelmts_bits: u8,
    ) -> Self {
        let min = data_blk_min_elmts as u64;
        let nsblks = 1 + (max_nelmts_bits as usize - log2_pow2(min) as usize);
        let mut sblk = Vec::with_capacity(nsblks);
        let mut start_idx = 0u64;
        let mut start_dblk = 0u64;
        for u in 0..nsblks {
            let ndblks = 1u64 << (u / 2);
            let dblk_nelmts = (1u64 << (u as u64).div_ceil(2)) * min;
            sblk.push(EaSblkInfo {
                ndblks,
                dblk_nelmts,
                start_idx,
                start_dblk,
            });
            start_idx += ndblks * dblk_nelmts;
            start_dblk += ndblks;
        }
        let iblock_nsblks = 2 * log2_pow2(sup_blk_min_data_ptrs as u64) as usize;
        Self {
            idx_blk_elmts: idx_blk_elmts as u64,
            data_blk_min_elmts: min,
            dblk_page_nelmts: 1u64 << max_dblk_page_nelmts_bits,
            iblock_nsblks,
            ndblk_addrs: 2 * (sup_blk_min_data_ptrs as usize - 1),
            nsblk_addrs: nsblks - iblock_nsblks,
            sblk,
        }
    }

    /// Super-block index containing chunk `idx` (`idx >= idx_blk_elmts`).
    pub fn sblk_index(&self, idx: u64) -> usize {
        let e = idx - self.idx_blk_elmts;
        log2_floor(e / self.data_blk_min_elmts + 1) as usize
    }

    /// Locate chunk `idx` within the array.
    pub fn locate(&self, idx: u64) -> EaLoc {
        if idx < self.idx_blk_elmts {
            return EaLoc::Index { elem: idx as usize };
        }
        let sblk_idx = self.sblk_index(idx);
        let s = self.sblk[sblk_idx];
        let elmt = (idx - self.idx_blk_elmts) - s.start_idx;
        let local_dblk = elmt / s.dblk_nelmts;
        let offset_in_dblk = elmt % s.dblk_nelmts;
        let paged = s.dblk_nelmts > self.dblk_page_nelmts;
        let path = if sblk_idx < self.iblock_nsblks {
            let global_dblk = s.start_dblk + local_dblk;
            EaDblkPath::Direct {
                idx: global_dblk as usize,
            }
        } else {
            EaDblkPath::ViaSblk {
                sblk_off: sblk_idx - self.iblock_nsblks,
                local_dblk: local_dblk as usize,
                ndblks_in_sblk: s.ndblks as usize,
                sblk_block_offset: s.start_idx,
            }
        };
        // libhdf5 stamps the data block's block_offset using the *global*
        // data-block index for index-block data blocks, and the local index
        // for super-block data blocks (H5EA__lookup_elmt).
        let dblk_block_offset = match path {
            EaDblkPath::Direct { idx } => s.start_idx + (idx as u64) * s.dblk_nelmts,
            EaDblkPath::ViaSblk { .. } => s.start_idx + local_dblk * s.dblk_nelmts,
        };
        EaLoc::Dblk(EaChunkLoc {
            sblk_idx,
            dblk_nelmts: s.dblk_nelmts,
            offset_in_dblk,
            dblk_block_offset,
            paged,
            path,
        })
    }
}

/// Extensible array super block (EASB).
///
/// Holds the data-block addresses for one super block. Used for super blocks
/// whose data-block addresses do not fit in the index block. Paged data
/// blocks (very large arrays) are not supported.
///
/// On-disk layout:
/// ```text
/// "EASB"(4) + version(1) + class_id(1)
/// + header_addr(sizeof_addr)
/// + block_offset(arr_off_size)
/// + data_block_addresses(ndblks * sizeof_addr)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensibleArraySuperBlock {
    pub class_id: u8,
    pub header_addr: u64,
    pub block_offset: u64,
    pub dblk_addrs: Vec<u64>,
}

impl ExtensibleArraySuperBlock {
    /// Create an empty super block with `ndblks` undefined data-block slots.
    pub fn new(class_id: u8, header_addr: u64, block_offset: u64, ndblks: usize) -> Self {
        Self {
            class_id,
            header_addr,
            block_offset,
            dblk_addrs: vec![UNDEF_ADDR; ndblks],
        }
    }

    /// Encoded size for `ndblks` data-block slots.
    pub fn encoded_size(ctx: &FormatContext, max_nelmts_bits: u8, ndblks: usize) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let bo = ExtensibleArrayDataBlock::block_offset_size(max_nelmts_bits);
        4 + 1 + 1 + sa + bo + ndblks * sa + 4
    }

    pub fn encode(&self, ctx: &FormatContext, max_nelmts_bits: u8) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let bo = ExtensibleArrayDataBlock::block_offset_size(max_nelmts_bits);
        let size = Self::encoded_size(ctx, max_nelmts_bits, self.dblk_addrs.len());
        let mut buf = Vec::with_capacity(size);
        buf.extend_from_slice(&EASB_SIGNATURE);
        buf.push(EA_VERSION);
        buf.push(self.class_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.block_offset.to_le_bytes()[..bo]);
        for &a in &self.dblk_addrs {
            buf.extend_from_slice(&a.to_le_bytes()[..sa]);
        }
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        max_nelmts_bits: u8,
        ndblks: usize,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let bo = ExtensibleArrayDataBlock::block_offset_size(max_nelmts_bits);
        let min_size = 4 + 1 + 1 + sa + bo + ndblks * sa + 4;
        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }
        if buf[0..4] != EASB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }
        if buf[4] != EA_VERSION {
            return Err(FormatError::InvalidVersion(buf[4]));
        }
        let data_end = min_size - 4;
        let stored = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed = checksum_metadata(&buf[..data_end]);
        if stored != computed {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }
        let class_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;
        let block_offset = read_size(&buf[pos..], bo);
        pos += bo;
        let mut dblk_addrs = Vec::with_capacity(ndblks);
        for _ in 0..ndblks {
            dblk_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }
        Ok(Self {
            class_id,
            header_addr,
            block_offset,
            dblk_addrs,
        })
    }
}

/// Compute nsblk_addrs for the index block: the number of super block
/// address slots stored in the EAIB.
pub fn compute_nsblk_addrs(
    idx_blk_elmts: u8,
    data_blk_min_elmts: u8,
    sup_blk_min_data_ptrs: u8,
    max_nelmts_bits: u8,
) -> usize {
    EaGeometry::new(
        idx_blk_elmts,
        data_blk_min_elmts,
        sup_blk_min_data_ptrs,
        max_nelmts_bits,
        10,
    )
    .nsblk_addrs
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx8() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    fn ctx4() -> FormatContext {
        FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        }
    }

    #[test]
    fn header_roundtrip() {
        let mut hdr = ExtensibleArrayHeader::new_for_chunks(&ctx8());
        hdr.idx_blk_addr = 0x1000;
        hdr.max_idx_set = 3;
        hdr.num_elmts_realized = 4;

        let encoded = hdr.encode(&ctx8());
        assert_eq!(encoded.len(), hdr.encoded_size(&ctx8()));
        assert_eq!(&encoded[..4], b"EAHD");

        let decoded = ExtensibleArrayHeader::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_roundtrip_ctx4() {
        let mut hdr = ExtensibleArrayHeader::new_for_chunks(&ctx4());
        hdr.raw_elmt_size = 4;
        hdr.idx_blk_addr = 0x800;

        let encoded = hdr.encode(&ctx4());
        let decoded = ExtensibleArrayHeader::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_bad_signature() {
        let mut hdr = ExtensibleArrayHeader::new_for_chunks(&ctx8());
        hdr.idx_blk_addr = 0x1000;
        let mut encoded = hdr.encode(&ctx8());
        encoded[0] = b'X';
        let err = ExtensibleArrayHeader::decode(&encoded, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn header_checksum_mismatch() {
        let mut hdr = ExtensibleArrayHeader::new_for_chunks(&ctx8());
        hdr.idx_blk_addr = 0x1000;
        let mut encoded = hdr.encode(&ctx8());
        encoded[6] ^= 0xFF; // corrupt a byte
        let err = ExtensibleArrayHeader::decode(&encoded, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn index_block_roundtrip() {
        let ndblk = compute_ndblk_addrs(4);
        assert_eq!(ndblk, 6);

        let mut iblk = ExtensibleArrayIndexBlock::new(0x500, 4, ndblk, 0);
        iblk.elements[0] = 0x1000;
        iblk.elements[1] = 0x2000;
        iblk.dblk_addrs[0] = 0x3000;

        let encoded = iblk.encode(&ctx8());
        assert_eq!(encoded.len(), iblk.encoded_size(&ctx8()));
        assert_eq!(&encoded[..4], b"EAIB");

        let decoded = ExtensibleArrayIndexBlock::decode(&encoded, &ctx8(), 4, ndblk, 0).unwrap();
        assert_eq!(decoded, iblk);
    }

    #[test]
    fn index_block_roundtrip_ctx4() {
        let iblk = ExtensibleArrayIndexBlock::new(0x300, 4, 6, 0);
        let encoded = iblk.encode(&ctx4());
        let decoded = ExtensibleArrayIndexBlock::decode(&encoded, &ctx4(), 4, 6, 0).unwrap();
        assert_eq!(decoded, iblk);
    }

    #[test]
    fn index_block_bad_checksum() {
        let iblk = ExtensibleArrayIndexBlock::new(0x500, 4, 6, 0);
        let mut encoded = iblk.encode(&ctx8());
        encoded[8] ^= 0xFF;
        let err = ExtensibleArrayIndexBlock::decode(&encoded, &ctx8(), 4, 6, 0).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn data_block_roundtrip() {
        let mut dblk = ExtensibleArrayDataBlock::new(0x500, 4, 16);
        dblk.elements[0] = 0xA000;
        dblk.elements[5] = 0xB000;

        let encoded = dblk.encode(&ctx8(), 32);
        assert_eq!(encoded.len(), dblk.encoded_size(&ctx8(), 32));
        assert_eq!(&encoded[..4], b"EADB");

        let decoded = ExtensibleArrayDataBlock::decode(&encoded, &ctx8(), 32, 16).unwrap();
        assert_eq!(decoded, dblk);
    }

    #[test]
    fn data_block_offset_size() {
        assert_eq!(ExtensibleArrayDataBlock::block_offset_size(8), 1);
        assert_eq!(ExtensibleArrayDataBlock::block_offset_size(16), 2);
        assert_eq!(ExtensibleArrayDataBlock::block_offset_size(32), 4);
        assert_eq!(ExtensibleArrayDataBlock::block_offset_size(0), 1);
    }

    #[test]
    fn compute_ndblk_addrs_default() {
        // sup_blk_min_data_ptrs=4 => ndblk=6
        assert_eq!(compute_ndblk_addrs(4), 6);
        assert_eq!(compute_ndblk_addrs(2), 2);
    }

    #[test]
    fn compute_nsblk_addrs_default() {
        // Default params: idx_blk_elmts=4, data_blk_min_elmts=16,
        // sup_blk_min_data_ptrs=4, max_nelmts_bits=32
        // Should give nsblk_addrs=25 (matching HDF5 library)
        assert_eq!(compute_nsblk_addrs(4, 16, 4, 32), 25);
    }

    #[test]
    fn ea_geometry_matches_libhdf5() {
        // Verified against an h5py/libhdf5-produced EA file (/tmp/ea_ref.h5).
        let g = EaGeometry::new(4, 16, 4, 32, 10);
        assert_eq!(g.sblk.len(), 29, "nsblks");
        assert_eq!(g.iblock_nsblks, 4);
        assert_eq!(g.ndblk_addrs, 6);
        assert_eq!(g.nsblk_addrs, 25);
        // (ndblks, dblk_nelmts, start_idx, start_dblk) for the first 5 sblks.
        let expect = [
            (1u64, 16u64, 0u64, 0u64),
            (1, 32, 16, 1),
            (2, 32, 48, 2),
            (2, 64, 112, 4),
            (4, 64, 240, 6),
        ];
        for (u, &(nd, dn, si, sd)) in expect.iter().enumerate() {
            let s = g.sblk[u];
            assert_eq!(
                (s.ndblks, s.dblk_nelmts, s.start_idx, s.start_dblk),
                (nd, dn, si, sd),
                "super block {}",
                u
            );
        }
        // chunk 4 (e=0): super block 0, direct data block 0, block_offset 0.
        match g.locate(4) {
            EaLoc::Dblk(l) => {
                assert_eq!(l.sblk_idx, 0);
                assert!(matches!(l.path, EaDblkPath::Direct { idx: 0 }));
                assert_eq!(l.dblk_block_offset, 0);
            }
            _ => panic!("expected data block"),
        }
        // chunk 20 (e=16): super block 1, direct data block 1, block_offset 48.
        match g.locate(20) {
            EaLoc::Dblk(l) => {
                assert!(matches!(l.path, EaDblkPath::Direct { idx: 1 }));
                assert_eq!(l.dblk_block_offset, 48);
            }
            _ => panic!("expected data block"),
        }
        // chunk 244 (e=240): super block 4 -> reached via the index block's
        // super-block address array.
        match g.locate(244) {
            EaLoc::Dblk(l) => {
                assert_eq!(l.sblk_idx, 4);
                match l.path {
                    EaDblkPath::ViaSblk {
                        sblk_off,
                        local_dblk,
                        sblk_block_offset,
                        ..
                    } => assert_eq!((sblk_off, local_dblk, sblk_block_offset), (0, 0, 240)),
                    _ => panic!("expected super-block path"),
                }
            }
            _ => panic!("expected data block"),
        }
    }
}
