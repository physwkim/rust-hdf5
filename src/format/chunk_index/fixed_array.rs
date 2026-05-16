//! Fixed Array (FA) chunk index structures for HDF5.
//!
//! Implements the on-disk format for the fixed array used to index chunked
//! datasets where no dimension is unlimited (all dimensions are fixed-size).
//!
//! Structures:
//!   - Header (FAHD): metadata about the fixed array
//!   - Data Block (FADB): holds chunk addresses (or filtered chunk entries)

use crate::format::checksum::checksum_metadata;
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Signature for the fixed array header.
pub const FAHD_SIGNATURE: [u8; 4] = *b"FAHD";
/// Signature for the fixed array data block.
pub const FADB_SIGNATURE: [u8; 4] = *b"FADB";

/// Fixed array version.
pub const FA_VERSION: u8 = 0;

/// Client ID for unfiltered chunks.
pub const FA_CLIENT_CHUNK: u8 = 0;
/// Client ID for filtered chunks.
pub const FA_CLIENT_FILT_CHUNK: u8 = 1;

/// Default `log2(max elements per data block page)`.
///
/// Matches libhdf5's `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS` (`H5Dpkg.h`),
/// i.e. 1024 elements per page. libhdf5 asserts this value is `> 0`
/// (`H5Dfarray.c`), so 0 is never a valid on-disk value.
pub const FA_MAX_DBLK_PAGE_NELMTS_BITS: u8 = 10;

/// Fixed array header.
///
/// On-disk layout:
/// ```text
/// "FAHD"(4) + version=0(1) + client_id(1)
/// + element_size(1) + max_dblk_page_nelmts_bits(1)
/// + num_elmts(sizeof_size)
/// + data_blk_addr(sizeof_addr)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FixedArrayHeader {
    pub client_id: u8,
    pub element_size: u8,
    pub max_dblk_page_nelmts_bits: u8,
    pub num_elmts: u64,
    pub data_blk_addr: u64,
}

impl FixedArrayHeader {
    /// Create a new header for unfiltered chunk indexing.
    pub fn new_for_chunks(ctx: &FormatContext, num_elmts: u64) -> Self {
        Self {
            client_id: FA_CLIENT_CHUNK,
            element_size: ctx.sizeof_addr,
            max_dblk_page_nelmts_bits: FA_MAX_DBLK_PAGE_NELMTS_BITS,
            num_elmts,
            data_blk_addr: UNDEF_ADDR,
        }
    }

    /// Create a new header for filtered chunk indexing.
    ///
    /// `chunk_size_len` is the number of bytes needed to encode the chunk size
    /// (typically computed from the maximum possible compressed chunk size).
    pub fn new_for_filtered_chunks(
        ctx: &FormatContext,
        num_elmts: u64,
        chunk_size_len: u8,
    ) -> Self {
        // element_size = sizeof_addr + chunk_size_len + 4 (filter_mask)
        let element_size = ctx.sizeof_addr + chunk_size_len + 4;
        Self {
            client_id: FA_CLIENT_FILT_CHUNK,
            element_size,
            max_dblk_page_nelmts_bits: FA_MAX_DBLK_PAGE_NELMTS_BITS,
            num_elmts,
            data_blk_addr: UNDEF_ADDR,
        }
    }

    /// Number of elements per data block page (`1 << max_dblk_page_nelmts_bits`).
    ///
    /// Mirrors `dblk_page_nelmts` in libhdf5 (`H5FAcache.c`).
    pub fn dblk_page_nelmts(&self) -> u64 {
        1u64 << (self.max_dblk_page_nelmts_bits as u64)
    }

    /// Whether the data block for this array is paged.
    ///
    /// A data block is paged when `num_elmts > dblk_page_nelmts`
    /// (`H5FAcache.c`: `if (hdr->cparam.nelmts > dblk_page_nelmts)`).
    pub fn is_paged(&self) -> bool {
        self.num_elmts > self.dblk_page_nelmts()
    }

    /// Number of pages in the (paged) data block.
    ///
    /// `npages = ceil(num_elmts / dblk_page_nelmts)` — encoded in C as
    /// `((nelmts + dblk_page_nelmts) - 1) / dblk_page_nelmts`.
    /// Returns 0 when the data block is not paged.
    pub fn npages(&self) -> u64 {
        if self.is_paged() {
            self.num_elmts.div_ceil(self.dblk_page_nelmts())
        } else {
            0
        }
    }

    /// Compute the encoded size (for pre-allocation).
    pub fn encoded_size(&self, ctx: &FormatContext) -> usize {
        let ss = ctx.sizeof_size as usize;
        let sa = ctx.sizeof_addr as usize;
        // signature(4) + version(1) + client_id(1)
        // + element_size(1) + max_dblk_page_nelmts_bits(1)
        // + num_elmts(sizeof_size) + data_blk_addr(sizeof_addr)
        // + checksum(4)
        4 + 1 + 1 + 1 + 1 + ss + sa + 4
    }

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let ss = ctx.sizeof_size as usize;
        let sa = ctx.sizeof_addr as usize;
        let size = self.encoded_size(ctx);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&FAHD_SIGNATURE);
        buf.push(FA_VERSION);
        buf.push(self.client_id);
        buf.push(self.element_size);
        buf.push(self.max_dblk_page_nelmts_bits);

        buf.extend_from_slice(&self.num_elmts.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.data_blk_addr.to_le_bytes()[..sa]);

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let ss = ctx.sizeof_size as usize;
        let sa = ctx.sizeof_addr as usize;
        let min_size = 4 + 1 + 1 + 1 + 1 + ss + sa + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != FAHD_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != FA_VERSION {
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

        let client_id = buf[5];
        let element_size = buf[6];
        let max_dblk_page_nelmts_bits = buf[7];

        let mut pos = 8;
        let num_elmts = read_size(&buf[pos..], ss);
        pos += ss;
        let data_blk_addr = read_addr(&buf[pos..], sa);

        Ok(Self {
            client_id,
            element_size,
            max_dblk_page_nelmts_bits,
            num_elmts,
            data_blk_addr,
        })
    }
}

/// A single element in a fixed array for unfiltered chunks.
/// Each element is simply a chunk address (sizeof_addr bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedArrayChunkElement {
    pub address: u64,
}

/// A single element in a fixed array for filtered chunks.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedArrayFilteredChunkElement {
    pub address: u64,
    pub chunk_size: u32,
    pub filter_mask: u32,
}

/// Fixed array data block.
///
/// On-disk layout:
/// ```text
/// "FADB"(4) + version=0(1) + client_id(1)
/// + header_addr(sizeof_addr)
/// + [if paged: page_init_bitmap]
/// + elements(num_elmts * element_size)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FixedArrayDataBlock {
    pub client_id: u8,
    pub header_addr: u64,
    /// Chunk addresses (for unfiltered chunks).
    pub elements: Vec<u64>,
    /// Filtered chunk entries (for filtered chunks; only used when client_id == 1).
    pub filtered_elements: Vec<FixedArrayFilteredChunkElement>,
}

impl FixedArrayDataBlock {
    /// Create a new empty data block for unfiltered chunks.
    pub fn new_unfiltered(header_addr: u64, num_elmts: usize) -> Self {
        Self {
            client_id: FA_CLIENT_CHUNK,
            header_addr,
            elements: vec![UNDEF_ADDR; num_elmts],
            filtered_elements: Vec::new(),
        }
    }

    /// Create a new empty data block for filtered chunks.
    pub fn new_filtered(header_addr: u64, num_elmts: usize) -> Self {
        let default_entry = FixedArrayFilteredChunkElement {
            address: UNDEF_ADDR,
            chunk_size: 0,
            filter_mask: 0,
        };
        Self {
            client_id: FA_CLIENT_FILT_CHUNK,
            header_addr,
            elements: Vec::new(),
            filtered_elements: vec![default_entry; num_elmts],
        }
    }

    /// Compute the encoded size for unfiltered chunks.
    pub fn encoded_size_unfiltered(&self, ctx: &FormatContext) -> usize {
        let sa = ctx.sizeof_addr as usize;
        // signature(4) + version(1) + client_id(1)
        // + header_addr(sa)
        // + elements(n * sa)
        // + checksum(4)
        4 + 1 + 1 + sa + self.elements.len() * sa + 4
    }

    /// Compute the encoded size for filtered chunks.
    pub fn encoded_size_filtered(&self, ctx: &FormatContext, chunk_size_len: usize) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let elem_size = sa + chunk_size_len + 4; // addr + chunk_size + filter_mask
                                                 // signature(4) + version(1) + client_id(1)
                                                 // + header_addr(sa)
                                                 // + elements(n * elem_size)
                                                 // + checksum(4)
        4 + 1 + 1 + sa + self.filtered_elements.len() * elem_size + 4
    }

    /// Encode for unfiltered chunks.
    pub fn encode_unfiltered(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let size = self.encoded_size_unfiltered(ctx);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&FADB_SIGNATURE);
        buf.push(FA_VERSION);
        buf.push(self.client_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);

        for &addr in &self.elements {
            buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    /// Encode for filtered chunks.
    pub fn encode_filtered(&self, ctx: &FormatContext, chunk_size_len: usize) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let size = self.encoded_size_filtered(ctx, chunk_size_len);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&FADB_SIGNATURE);
        buf.push(FA_VERSION);
        buf.push(self.client_id);
        buf.extend_from_slice(&self.header_addr.to_le_bytes()[..sa]);

        for elem in &self.filtered_elements {
            buf.extend_from_slice(&elem.address.to_le_bytes()[..sa]);
            buf.extend_from_slice(&elem.chunk_size.to_le_bytes()[..chunk_size_len]);
            buf.extend_from_slice(&elem.filter_mask.to_le_bytes());
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    /// Decode for unfiltered chunks.
    pub fn decode_unfiltered(
        buf: &[u8],
        ctx: &FormatContext,
        num_elmts: usize,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let min_size = 4 + 1 + 1 + sa + num_elmts * sa + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != FADB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != FA_VERSION {
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

        let client_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;

        let mut elements = Vec::with_capacity(num_elmts);
        for _ in 0..num_elmts {
            elements.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        Ok(Self {
            client_id,
            header_addr,
            elements,
            filtered_elements: Vec::new(),
        })
    }

    /// Decode for filtered chunks.
    pub fn decode_filtered(
        buf: &[u8],
        ctx: &FormatContext,
        num_elmts: usize,
        chunk_size_len: usize,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let elem_size = sa + chunk_size_len + 4;
        let min_size = 4 + 1 + 1 + sa + num_elmts * elem_size + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != FADB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != FA_VERSION {
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

        let client_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;

        let mut filtered_elements = Vec::with_capacity(num_elmts);
        for _ in 0..num_elmts {
            let address = read_addr(&buf[pos..], sa);
            pos += sa;
            let chunk_size = read_size(&buf[pos..], chunk_size_len) as u32;
            pos += chunk_size_len;
            let filter_mask =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
            pos += 4;
            filtered_elements.push(FixedArrayFilteredChunkElement {
                address,
                chunk_size,
                filter_mask,
            });
        }

        Ok(Self {
            client_id,
            header_addr,
            elements: Vec::new(),
            filtered_elements,
        })
    }
}

/// Parsed prefix of a *paged* fixed array data block.
///
/// On-disk layout (`H5FA_DBLOCK_PREFIX_SIZE` + checksum):
/// ```text
/// "FADB"(4) + version=0(1) + client_id(1)
/// + header_addr(sizeof_addr)
/// + page_init_bitmap(ceil(npages/8) bytes)
/// + checksum(4)
/// ```
/// The `npages` pages follow at `dblk_addr + prefix_size`, where
/// `prefix_size` is the full size including the checksum.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedArrayPagedPrefix {
    pub client_id: u8,
    pub header_addr: u64,
    /// Page-init bitmap, MSB-first: page `p` initialized iff
    /// `bitmap[p / 8] & (0x80 >> (p % 8))` is non-zero.
    pub page_init_bitmap: Vec<u8>,
    /// Total size of the prefix on disk, including the 4-byte checksum.
    /// Pages start at `dblk_addr + prefix_size`.
    pub prefix_size: usize,
}

impl FixedArrayPagedPrefix {
    /// Decode the prefix of a paged data block.
    ///
    /// `npages` comes from the header (`FixedArrayHeader::npages`).
    pub fn decode(buf: &[u8], ctx: &FormatContext, npages: u64) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let bitmap_size = (npages as usize).div_ceil(8);
        // signature(4) + version(1) + client_id(1) + header_addr(sa)
        // + bitmap(bitmap_size) + checksum(4)
        let prefix_size = 4 + 1 + 1 + sa + bitmap_size + 4;

        if buf.len() < prefix_size {
            return Err(FormatError::BufferTooShort {
                needed: prefix_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != FADB_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != FA_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        // Verify checksum over everything before the trailing 4 bytes.
        let data_end = prefix_size - 4;
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

        let client_id = buf[5];
        let mut pos = 6;
        let header_addr = read_addr(&buf[pos..], sa);
        pos += sa;
        let page_init_bitmap = buf[pos..pos + bitmap_size].to_vec();

        Ok(Self {
            client_id,
            header_addr,
            page_init_bitmap,
            prefix_size,
        })
    }

    /// Whether page `p` has been initialized (MSB-first bit test).
    ///
    /// Mirrors `H5VM_bit_get`: `bitmap[p / 8] & (0x80 >> (p % 8))`.
    pub fn page_initialized(&self, p: usize) -> bool {
        let byte = p / 8;
        if byte >= self.page_init_bitmap.len() {
            return false;
        }
        (self.page_init_bitmap[byte] & (0x80u8 >> (p % 8))) != 0
    }
}

/// Decode the unfiltered chunk addresses contained in a single data block page.
///
/// `page_buf` must start at the page address and contain at least
/// `nelmts * sizeof_addr + 4` bytes. The trailing 4 bytes are a Jenkins
/// checksum over the page elements.
pub fn decode_unfiltered_page(
    page_buf: &[u8],
    ctx: &FormatContext,
    nelmts: usize,
) -> FormatResult<Vec<u64>> {
    let sa = ctx.sizeof_addr as usize;
    let page_size = nelmts * sa + 4;
    if page_buf.len() < page_size {
        return Err(FormatError::BufferTooShort {
            needed: page_size,
            available: page_buf.len(),
        });
    }

    let data_end = page_size - 4;
    let stored_cksum = u32::from_le_bytes([
        page_buf[data_end],
        page_buf[data_end + 1],
        page_buf[data_end + 2],
        page_buf[data_end + 3],
    ]);
    let computed_cksum = checksum_metadata(&page_buf[..data_end]);
    if stored_cksum != computed_cksum {
        return Err(FormatError::ChecksumMismatch {
            expected: stored_cksum,
            computed: computed_cksum,
        });
    }

    let mut elements = Vec::with_capacity(nelmts);
    let mut pos = 0;
    for _ in 0..nelmts {
        elements.push(read_addr(&page_buf[pos..], sa));
        pos += sa;
    }
    Ok(elements)
}

/// Decode the filtered chunk entries contained in a single data block page.
///
/// `page_buf` must start at the page address and contain at least
/// `nelmts * (sizeof_addr + chunk_size_len + 4) + 4` bytes. The trailing 4
/// bytes are a Jenkins checksum over the page elements.
pub fn decode_filtered_page(
    page_buf: &[u8],
    ctx: &FormatContext,
    nelmts: usize,
    chunk_size_len: usize,
) -> FormatResult<Vec<FixedArrayFilteredChunkElement>> {
    let sa = ctx.sizeof_addr as usize;
    let elem_size = sa + chunk_size_len + 4;
    let page_size = nelmts * elem_size + 4;
    if page_buf.len() < page_size {
        return Err(FormatError::BufferTooShort {
            needed: page_size,
            available: page_buf.len(),
        });
    }

    let data_end = page_size - 4;
    let stored_cksum = u32::from_le_bytes([
        page_buf[data_end],
        page_buf[data_end + 1],
        page_buf[data_end + 2],
        page_buf[data_end + 3],
    ]);
    let computed_cksum = checksum_metadata(&page_buf[..data_end]);
    if stored_cksum != computed_cksum {
        return Err(FormatError::ChecksumMismatch {
            expected: stored_cksum,
            computed: computed_cksum,
        });
    }

    let mut elements = Vec::with_capacity(nelmts);
    let mut pos = 0;
    for _ in 0..nelmts {
        let address = read_addr(&page_buf[pos..], sa);
        pos += sa;
        let chunk_size = read_size(&page_buf[pos..], chunk_size_len) as u32;
        pos += chunk_size_len;
        let filter_mask = u32::from_le_bytes([
            page_buf[pos],
            page_buf[pos + 1],
            page_buf[pos + 2],
            page_buf[pos + 3],
        ]);
        pos += 4;
        elements.push(FixedArrayFilteredChunkElement {
            address,
            chunk_size,
            filter_mask,
        });
    }
    Ok(elements)
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
        let mut hdr = FixedArrayHeader::new_for_chunks(&ctx8(), 10);
        hdr.data_blk_addr = 0x2000;

        let encoded = hdr.encode(&ctx8());
        assert_eq!(encoded.len(), hdr.encoded_size(&ctx8()));
        assert_eq!(&encoded[..4], b"FAHD");

        let decoded = FixedArrayHeader::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_roundtrip_ctx4() {
        let mut hdr = FixedArrayHeader::new_for_chunks(&ctx4(), 5);
        hdr.data_blk_addr = 0x800;

        let encoded = hdr.encode(&ctx4());
        let decoded = FixedArrayHeader::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_bad_signature() {
        let hdr = FixedArrayHeader::new_for_chunks(&ctx8(), 10);
        let mut encoded = hdr.encode(&ctx8());
        encoded[0] = b'X';
        let err = FixedArrayHeader::decode(&encoded, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn header_checksum_mismatch() {
        let hdr = FixedArrayHeader::new_for_chunks(&ctx8(), 10);
        let mut encoded = hdr.encode(&ctx8());
        encoded[6] ^= 0xFF;
        let err = FixedArrayHeader::decode(&encoded, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn data_block_unfiltered_roundtrip() {
        let mut dblk = FixedArrayDataBlock::new_unfiltered(0x1000, 4);
        dblk.elements[0] = 0x3000;
        dblk.elements[1] = 0x4000;
        dblk.elements[2] = UNDEF_ADDR;
        dblk.elements[3] = 0x5000;

        let encoded = dblk.encode_unfiltered(&ctx8());
        assert_eq!(encoded.len(), dblk.encoded_size_unfiltered(&ctx8()));
        assert_eq!(&encoded[..4], b"FADB");

        let decoded = FixedArrayDataBlock::decode_unfiltered(&encoded, &ctx8(), 4).unwrap();
        assert_eq!(decoded.elements, dblk.elements);
        assert_eq!(decoded.header_addr, 0x1000);
    }

    #[test]
    fn data_block_unfiltered_roundtrip_ctx4() {
        let mut dblk = FixedArrayDataBlock::new_unfiltered(0x500, 3);
        dblk.elements[0] = 0x100;
        dblk.elements[1] = 0x200;
        dblk.elements[2] = 0x300;

        let encoded = dblk.encode_unfiltered(&ctx4());
        let decoded = FixedArrayDataBlock::decode_unfiltered(&encoded, &ctx4(), 3).unwrap();
        assert_eq!(decoded.elements, dblk.elements);
    }

    #[test]
    fn data_block_unfiltered_bad_checksum() {
        let dblk = FixedArrayDataBlock::new_unfiltered(0x1000, 2);
        let mut encoded = dblk.encode_unfiltered(&ctx8());
        encoded[8] ^= 0xFF;
        let err = FixedArrayDataBlock::decode_unfiltered(&encoded, &ctx8(), 2).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn data_block_filtered_roundtrip() {
        let mut dblk = FixedArrayDataBlock::new_filtered(0x1000, 2);
        dblk.filtered_elements[0] = FixedArrayFilteredChunkElement {
            address: 0x2000,
            chunk_size: 512,
            filter_mask: 0,
        };
        dblk.filtered_elements[1] = FixedArrayFilteredChunkElement {
            address: 0x3000,
            chunk_size: 400,
            filter_mask: 1,
        };

        let chunk_size_len = 4; // 4 bytes for chunk_size
        let encoded = dblk.encode_filtered(&ctx8(), chunk_size_len);
        assert_eq!(
            encoded.len(),
            dblk.encoded_size_filtered(&ctx8(), chunk_size_len)
        );

        let decoded =
            FixedArrayDataBlock::decode_filtered(&encoded, &ctx8(), 2, chunk_size_len).unwrap();
        assert_eq!(decoded.filtered_elements, dblk.filtered_elements);
    }

    #[test]
    fn header_filtered_roundtrip() {
        let hdr = FixedArrayHeader::new_for_filtered_chunks(&ctx8(), 6, 4);
        assert_eq!(hdr.element_size, 8 + 4 + 4); // addr + chunk_size_len + filter_mask
        assert_eq!(hdr.client_id, FA_CLIENT_FILT_CHUNK);

        let encoded = hdr.encode(&ctx8());
        let decoded = FixedArrayHeader::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn empty_data_block() {
        let dblk = FixedArrayDataBlock::new_unfiltered(0x500, 0);
        let encoded = dblk.encode_unfiltered(&ctx8());
        let decoded = FixedArrayDataBlock::decode_unfiltered(&encoded, &ctx8(), 0).unwrap();
        assert!(decoded.elements.is_empty());
    }

    // ----------------------------------------------------------------- paging

    #[test]
    fn header_paging_geometry() {
        // max_dblk_page_nelmts_bits = 2 => 4 elements per page.
        let mut hdr = FixedArrayHeader::new_for_chunks(&ctx8(), 10);
        hdr.max_dblk_page_nelmts_bits = 2;
        assert_eq!(hdr.dblk_page_nelmts(), 4);
        assert!(hdr.is_paged()); // 10 > 4
        assert_eq!(hdr.npages(), 3); // ceil(10 / 4)

        // num_elmts == dblk_page_nelmts => NOT paged (strict greater-than).
        hdr.num_elmts = 4;
        assert!(!hdr.is_paged());
        assert_eq!(hdr.npages(), 0);

        // One past the page size => paged with 2 pages.
        hdr.num_elmts = 5;
        assert!(hdr.is_paged());
        assert_eq!(hdr.npages(), 2);
    }

    /// Build a paged FADB prefix buffer for `npages` pages.
    fn build_paged_prefix(
        ctx: &FormatContext,
        client_id: u8,
        header_addr: u64,
        npages: usize,
        init_bits: &[bool],
    ) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let bitmap_size = npages.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_size];
        for (p, &on) in init_bits.iter().enumerate() {
            if on {
                bitmap[p / 8] |= 0x80u8 >> (p % 8);
            }
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(&FADB_SIGNATURE);
        buf.push(FA_VERSION);
        buf.push(client_id);
        buf.extend_from_slice(&header_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&bitmap);
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        buf
    }

    /// Build a single unfiltered data block page.
    fn build_unfiltered_page(ctx: &FormatContext, addrs: &[u64]) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let mut buf = Vec::new();
        for &a in addrs {
            buf.extend_from_slice(&a.to_le_bytes()[..sa]);
        }
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        buf
    }

    /// Build a single filtered data block page.
    fn build_filtered_page(
        ctx: &FormatContext,
        chunk_size_len: usize,
        elems: &[FixedArrayFilteredChunkElement],
    ) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let mut buf = Vec::new();
        for e in elems {
            buf.extend_from_slice(&e.address.to_le_bytes()[..sa]);
            buf.extend_from_slice(&(e.chunk_size as u64).to_le_bytes()[..chunk_size_len]);
            buf.extend_from_slice(&e.filter_mask.to_le_bytes());
        }
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        buf
    }

    #[test]
    fn paged_prefix_roundtrip_and_bitmap() {
        let ctx = ctx8();
        // 11 pages => 2-byte bitmap. Initialize pages 0, 7, 8, 10.
        let mut init = vec![false; 11];
        for &p in &[0usize, 7, 8, 10] {
            init[p] = true;
        }
        let buf = build_paged_prefix(&ctx, FA_CLIENT_CHUNK, 0xABCD, 11, &init);

        let prefix = FixedArrayPagedPrefix::decode(&buf, &ctx, 11).unwrap();
        assert_eq!(prefix.client_id, FA_CLIENT_CHUNK);
        assert_eq!(prefix.header_addr, 0xABCD);
        assert_eq!(prefix.prefix_size, buf.len());
        // signature(4)+version(1)+client(1)+addr(8)+bitmap(2)+cksum(4) = 20
        assert_eq!(prefix.prefix_size, 20);
        for p in 0..11 {
            assert_eq!(prefix.page_initialized(p), init[p], "page {p}");
        }
    }

    #[test]
    fn paged_prefix_bad_checksum() {
        let ctx = ctx8();
        let mut buf = build_paged_prefix(&ctx, FA_CLIENT_CHUNK, 0x1000, 3, &[true, true, true]);
        buf[6] ^= 0xFF;
        let err = FixedArrayPagedPrefix::decode(&buf, &ctx, 3).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn unfiltered_page_roundtrip() {
        let ctx = ctx8();
        let addrs = [0x100u64, UNDEF_ADDR, 0x300, 0x400];
        let page = build_unfiltered_page(&ctx, &addrs);
        let decoded = decode_unfiltered_page(&page, &ctx, 4).unwrap();
        assert_eq!(decoded, addrs);
    }

    #[test]
    fn unfiltered_page_bad_checksum() {
        let ctx = ctx8();
        let mut page = build_unfiltered_page(&ctx, &[0x100u64, 0x200]);
        page[0] ^= 0xFF;
        let err = decode_unfiltered_page(&page, &ctx, 2).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn filtered_page_roundtrip() {
        let ctx = ctx8();
        let csl = 4;
        let elems = vec![
            FixedArrayFilteredChunkElement {
                address: 0x2000,
                chunk_size: 321,
                filter_mask: 0,
            },
            FixedArrayFilteredChunkElement {
                address: 0x3000,
                chunk_size: 654,
                filter_mask: 2,
            },
        ];
        let page = build_filtered_page(&ctx, csl, &elems);
        let decoded = decode_filtered_page(&page, &ctx, 2, csl).unwrap();
        assert_eq!(decoded, elems);
    }

    #[test]
    fn page_too_short_errors() {
        let ctx = ctx8();
        let page = build_unfiltered_page(&ctx, &[0x100u64, 0x200]);
        // Ask for more elements than the buffer holds.
        let err = decode_unfiltered_page(&page, &ctx, 4).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }
}
