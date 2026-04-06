//! Extensible Array (EA) chunk index structures for HDF5.
//!
//! Implements the on-disk format for the extensible array used to index
//! chunked datasets with one unlimited dimension (the typical SWMR use case).
//!
//! Structures:
//!   - Header (EAHD): metadata and statistics about the extensible array
//!   - Index Block (EAIB): holds direct chunk addresses and pointers to data/super blocks
//!   - Data Block (EADB): holds additional chunk addresses when the index block is full

use crate::checksum::checksum_metadata;
use crate::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Signature for the extensible array header.
pub const EAHD_SIGNATURE: [u8; 4] = *b"EAHD";
/// Signature for the extensible array index block.
pub const EAIB_SIGNATURE: [u8; 4] = *b"EAIB";
/// Signature for the extensible array data block.
pub const EADB_SIGNATURE: [u8; 4] = *b"EADB";

/// Extensible array version.
pub const EA_VERSION: u8 = 0;

/// Class ID for unfiltered chunks (H5EA_CLS_CHUNK).
pub const EA_CLS_CHUNK: u8 = 0;

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
            buf[data_end], buf[data_end + 1], buf[data_end + 2], buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        let mut pos = 5;
        let class_id = buf[pos]; pos += 1;
        let raw_elmt_size = buf[pos]; pos += 1;
        let max_nelmts_bits = buf[pos]; pos += 1;
        let idx_blk_elmts = buf[pos]; pos += 1;
        let data_blk_min_elmts = buf[pos]; pos += 1;
        let sup_blk_min_data_ptrs = buf[pos]; pos += 1;
        let max_dblk_page_nelmts_bits = buf[pos]; pos += 1;

        let num_sblks_created = read_size(&buf[pos..], ss); pos += ss;
        let size_sblks_created = read_size(&buf[pos..], ss); pos += ss;
        let num_dblks_created = read_size(&buf[pos..], ss); pos += ss;
        let size_dblks_created = read_size(&buf[pos..], ss); pos += ss;
        let max_idx_set = read_size(&buf[pos..], ss); pos += ss;
        let num_elmts_realized = read_size(&buf[pos..], ss); pos += ss;

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

impl ExtensibleArrayIndexBlock {
    /// Create a new empty index block.
    pub fn new(header_addr: u64, idx_blk_elmts: u8, ndblk_addrs: usize, nsblk_addrs: usize) -> Self {
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
        4 + 1 + 1 + sa
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
        let min_size = 4 + 1 + 1 + sa
            + idx_blk_elmts * sa
            + ndblk_addrs * sa
            + nsblk_addrs * sa
            + 4;

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
            buf[data_end], buf[data_end + 1], buf[data_end + 2], buf[data_end + 3],
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
        let header_addr = read_addr(&buf[pos..], sa); pos += sa;

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
            buf[data_end], buf[data_end + 1], buf[data_end + 2], buf[data_end + 3],
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
        let header_addr = read_addr(&buf[pos..], sa); pos += sa;
        let block_offset = read_size(&buf[pos..], bo_size); pos += bo_size;

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

/// Compute the total number of super blocks (nsblks) for the given parameters.
fn compute_nsblks(
    idx_blk_elmts: u8,
    data_blk_min_elmts: u8,
    max_nelmts_bits: u8,
) -> usize {
    let max_nelmts: u64 = 1u64 << (max_nelmts_bits as u64);
    let nelmts_remaining = max_nelmts - idx_blk_elmts as u64;

    let mut nsblks = 0usize;
    let mut acc = 0u64;
    while acc < nelmts_remaining {
        let (ndblks_in_sblk, dblk_size) = if nsblks < 2 {
            (1u64, data_blk_min_elmts as u64)
        } else {
            let half = (nsblks - 2) / 2;
            (1u64 << (half + 1), (data_blk_min_elmts as u64) << (half + 1))
        };
        acc = acc.saturating_add(ndblks_in_sblk.saturating_mul(dblk_size));
        nsblks += 1;
    }
    nsblks
}

/// Compute sblk_idx_start: the first super block whose data block addresses
/// are NOT stored in the index block's dblk_addrs array.
fn compute_sblk_idx_start(
    sup_blk_min_data_ptrs: u8,
    nsblks: usize,
) -> usize {
    let ndblk_addrs = compute_ndblk_addrs(sup_blk_min_data_ptrs);
    let mut dblks_counted = 0usize;
    let mut sblk_idx_start = 0usize;

    for s in 0..nsblks {
        let ndblks_in_sblk = if s < 2 {
            1
        } else {
            let half = (s - 2) / 2;
            1 << (half + 1)
        };

        if dblks_counted + ndblks_in_sblk > ndblk_addrs {
            break;
        }
        dblks_counted += ndblks_in_sblk;
        sblk_idx_start = s + 1;
    }
    sblk_idx_start
}

/// Compute nsblk_addrs for the index block: the number of super block
/// address slots stored in the EAIB.
pub fn compute_nsblk_addrs(
    idx_blk_elmts: u8,
    data_blk_min_elmts: u8,
    sup_blk_min_data_ptrs: u8,
    max_nelmts_bits: u8,
) -> usize {
    let nsblks = compute_nsblks(idx_blk_elmts, data_blk_min_elmts, max_nelmts_bits);
    let sblk_idx_start = compute_sblk_idx_start(sup_blk_min_data_ptrs, nsblks);
    nsblks - sblk_idx_start
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx8() -> FormatContext {
        FormatContext { sizeof_addr: 8, sizeof_size: 8 }
    }

    fn ctx4() -> FormatContext {
        FormatContext { sizeof_addr: 4, sizeof_size: 4 }
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

        let decoded = ExtensibleArrayIndexBlock::decode(
            &encoded, &ctx8(), 4, ndblk, 0,
        ).unwrap();
        assert_eq!(decoded, iblk);
    }

    #[test]
    fn index_block_roundtrip_ctx4() {
        let iblk = ExtensibleArrayIndexBlock::new(0x300, 4, 6, 0);
        let encoded = iblk.encode(&ctx4());
        let decoded = ExtensibleArrayIndexBlock::decode(
            &encoded, &ctx4(), 4, 6, 0,
        ).unwrap();
        assert_eq!(decoded, iblk);
    }

    #[test]
    fn index_block_bad_checksum() {
        let iblk = ExtensibleArrayIndexBlock::new(0x500, 4, 6, 0);
        let mut encoded = iblk.encode(&ctx8());
        encoded[8] ^= 0xFF;
        let err = ExtensibleArrayIndexBlock::decode(
            &encoded, &ctx8(), 4, 6, 0,
        ).unwrap_err();
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
}
