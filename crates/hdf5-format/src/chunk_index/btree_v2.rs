//! B-tree v2 (BT2) chunk index structures for HDF5.
//!
//! Implements the on-disk format for B-tree version 2 used to index chunked
//! datasets with multiple unlimited dimensions.
//!
//! Structures:
//!   - Header (BTHD): metadata about the B-tree
//!   - Internal Node (BTIN): non-leaf nodes with records, child pointers
//!   - Leaf Node (BTLF): leaf nodes containing records
//!
//! Record types:
//!   - Type 10: unfiltered chunks (scaled offsets + chunk address)
//!   - Type 11: filtered chunks (scaled offsets + chunk address + chunk_size + filter_mask)

use crate::checksum::checksum_metadata;
use crate::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Signature for the B-tree v2 header.
pub const BTHD_SIGNATURE: [u8; 4] = *b"BTHD";
/// Signature for the B-tree v2 internal node.
pub const BTIN_SIGNATURE: [u8; 4] = *b"BTIN";
/// Signature for the B-tree v2 leaf node.
pub const BTLF_SIGNATURE: [u8; 4] = *b"BTLF";

/// B-tree v2 version.
pub const BT2_VERSION: u8 = 0;

/// Record type: unfiltered chunks (non-filtered chunked datasets).
pub const BT2_TYPE_CHUNK_UNFILT: u8 = 10;
/// Record type: filtered chunks (filtered chunked datasets).
pub const BT2_TYPE_CHUNK_FILT: u8 = 11;

/// A chunk record for BT2 type 10 (unfiltered).
///
/// Contains the scaled chunk coordinates and the file address.
#[derive(Debug, Clone, PartialEq)]
pub struct Bt2ChunkRecord {
    /// Scaled chunk coordinates (one per dataset dimension).
    pub scaled_offsets: Vec<u64>,
    /// File address of the chunk data.
    pub chunk_address: u64,
}

/// A filtered chunk record for BT2 type 11.
///
/// Contains scaled chunk coordinates, file address, chunk size, and filter mask.
#[derive(Debug, Clone, PartialEq)]
pub struct Bt2FilteredChunkRecord {
    /// Scaled chunk coordinates (one per dataset dimension).
    pub scaled_offsets: Vec<u64>,
    /// File address of the chunk data.
    pub chunk_address: u64,
    /// Size of the chunk after filtering (compressed size).
    pub chunk_size: u32,
    /// Filter mask (bit i set = skip filter i).
    pub filter_mask: u32,
}

/// B-tree v2 header.
///
/// On-disk layout:
/// ```text
/// "BTHD"(4) + version(1) + type(1)
/// + node_size(u32 LE) + record_size(u16 LE) + depth(u16 LE)
/// + split_percent(u8) + merge_percent(u8)
/// + root_node_addr(sizeof_addr) + num_records_in_root(u16 LE)
/// + total_num_records(sizeof_size)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Bt2Header {
    /// Record type (10=unfilt chunks, 11=filt chunks).
    pub record_type: u8,
    /// Size of each node in bytes.
    pub node_size: u32,
    /// Size of each record in bytes.
    pub record_size: u16,
    /// Depth of the B-tree (0 = root is a leaf).
    pub depth: u16,
    /// Percentage full a node must be before splitting.
    pub split_percent: u8,
    /// Percentage below which a node is merged.
    pub merge_percent: u8,
    /// Address of the root node.
    pub root_node_addr: u64,
    /// Number of records in the root node.
    pub num_records_in_root: u16,
    /// Total number of records in the entire B-tree.
    pub total_num_records: u64,
}

impl Bt2Header {
    /// Create a new B-tree v2 header for unfiltered chunk indexing.
    ///
    /// `ndims` is the number of dataset dimensions.
    pub fn new_for_chunks(ctx: &FormatContext, ndims: usize) -> Self {
        // record_size = ndims * 8 (scaled offsets) + sizeof_addr (chunk address)
        let record_size = (ndims * 8 + ctx.sizeof_addr as usize) as u16;
        Self {
            record_type: BT2_TYPE_CHUNK_UNFILT,
            node_size: 4096,
            record_size,
            depth: 0,
            split_percent: 100,
            merge_percent: 40,
            root_node_addr: UNDEF_ADDR,
            num_records_in_root: 0,
            total_num_records: 0,
        }
    }

    /// Create a new B-tree v2 header for filtered chunk indexing.
    ///
    /// `ndims` is the number of dataset dimensions.
    pub fn new_for_filtered_chunks(ctx: &FormatContext, ndims: usize) -> Self {
        // record_size = ndims * 8 + sizeof_addr + 4 (chunk_size) + 4 (filter_mask)
        let record_size = (ndims * 8 + ctx.sizeof_addr as usize + 4 + 4) as u16;
        Self {
            record_type: BT2_TYPE_CHUNK_FILT,
            node_size: 4096,
            record_size,
            depth: 0,
            split_percent: 100,
            merge_percent: 40,
            root_node_addr: UNDEF_ADDR,
            num_records_in_root: 0,
            total_num_records: 0,
        }
    }

    /// Compute the encoded size (for pre-allocation).
    pub fn encoded_size(&self, ctx: &FormatContext) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        // signature(4) + version(1) + type(1) + node_size(4) + record_size(2)
        // + depth(2) + split_percent(1) + merge_percent(1)
        // + root_node_addr(sa) + num_records_in_root(2) + total_num_records(ss)
        // + checksum(4)
        4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + sa + 2 + ss + 4
    }

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let size = self.encoded_size(ctx);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&BTHD_SIGNATURE);
        buf.push(BT2_VERSION);
        buf.push(self.record_type);
        buf.extend_from_slice(&self.node_size.to_le_bytes());
        buf.extend_from_slice(&self.record_size.to_le_bytes());
        buf.extend_from_slice(&self.depth.to_le_bytes());
        buf.push(self.split_percent);
        buf.push(self.merge_percent);
        buf.extend_from_slice(&self.root_node_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.num_records_in_root.to_le_bytes());
        buf.extend_from_slice(&self.total_num_records.to_le_bytes()[..ss]);

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let min_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + sa + 2 + ss + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != BTHD_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != BT2_VERSION {
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
        let record_type = buf[pos]; pos += 1;
        let node_size = u32::from_le_bytes([buf[pos], buf[pos+1], buf[pos+2], buf[pos+3]]);
        pos += 4;
        let record_size = u16::from_le_bytes([buf[pos], buf[pos+1]]);
        pos += 2;
        let depth = u16::from_le_bytes([buf[pos], buf[pos+1]]);
        pos += 2;
        let split_percent = buf[pos]; pos += 1;
        let merge_percent = buf[pos]; pos += 1;
        let root_node_addr = read_addr(&buf[pos..], sa); pos += sa;
        let num_records_in_root = u16::from_le_bytes([buf[pos], buf[pos+1]]);
        pos += 2;
        let total_num_records = read_size(&buf[pos..], ss);

        Ok(Self {
            record_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_node_addr,
            num_records_in_root,
            total_num_records,
        })
    }
}

/// B-tree v2 leaf node.
///
/// On-disk layout:
/// ```text
/// "BTLF"(4) + version(1) + type(1)
/// + records(num_records * record_size)
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Bt2LeafNode {
    pub record_type: u8,
    /// Raw record data. Each record is `record_size` bytes.
    pub record_data: Vec<u8>,
    /// Number of records in this node.
    pub num_records: u16,
    /// Size of each record.
    pub record_size: u16,
}

impl Bt2LeafNode {
    /// Create a new empty leaf node.
    pub fn new(record_type: u8, record_size: u16) -> Self {
        Self {
            record_type,
            record_data: Vec::new(),
            num_records: 0,
            record_size,
        }
    }

    /// Compute the encoded size.
    pub fn encoded_size(&self) -> usize {
        // signature(4) + version(1) + type(1) + records + checksum(4)
        4 + 1 + 1 + self.record_data.len() + 4
    }

    pub fn encode(&self) -> Vec<u8> {
        let size = self.encoded_size();
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&BTLF_SIGNATURE);
        buf.push(BT2_VERSION);
        buf.push(self.record_type);
        buf.extend_from_slice(&self.record_data);

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(buf: &[u8], num_records: u16, record_size: u16) -> FormatResult<Self> {
        let records_len = num_records as usize * record_size as usize;
        let min_size = 4 + 1 + 1 + records_len + 4;

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != BTLF_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != BT2_VERSION {
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

        let record_type = buf[5];
        let record_data = buf[6..6 + records_len].to_vec();

        Ok(Self {
            record_type,
            record_data,
            num_records,
            record_size,
        })
    }
}

/// B-tree v2 internal node.
///
/// On-disk layout:
/// ```text
/// "BTIN"(4) + version(1) + type(1)
/// + records(num_records * record_size)
/// + child_node_addrs((num_records+1) * sizeof_addr)
/// + child_nrecords((num_records+1) * nrec_size_bits, packed bytes)
/// + [if depth > 1: child_total_nrecords((num_records+1) * sizeof_size)]
/// + checksum(4)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Bt2InternalNode {
    pub record_type: u8,
    /// Raw record data.
    pub record_data: Vec<u8>,
    /// Number of records.
    pub num_records: u16,
    /// Size of each record.
    pub record_size: u16,
    /// Child node addresses.
    pub child_addrs: Vec<u64>,
    /// Number of records in each child.
    pub child_nrecords: Vec<u16>,
    /// Total number of records in each child's subtree (only for depth > 1).
    pub child_total_nrecords: Vec<u64>,
}

impl Bt2InternalNode {
    /// Create a new empty internal node.
    pub fn new(record_type: u8, record_size: u16) -> Self {
        Self {
            record_type,
            record_data: Vec::new(),
            num_records: 0,
            record_size,
            child_addrs: Vec::new(),
            child_nrecords: Vec::new(),
            child_total_nrecords: Vec::new(),
        }
    }

    /// Compute the number of bytes needed for the child_nrecords packed field.
    fn nrec_bytes(num_children: usize, max_nrec: u32) -> usize {
        let bits_per_child = std::cmp::max(1, (32 - max_nrec.leading_zeros()) as usize);
        (num_children * bits_per_child).div_ceil(8)
    }

    /// Compute the encoded size.
    pub fn encoded_size(
        &self,
        ctx: &FormatContext,
        depth: u16,
        max_nrec_per_node: u32,
    ) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let num_children = self.num_records as usize + 1;
        let nrec_bytes = Self::nrec_bytes(num_children, max_nrec_per_node);

        let mut size = 4 + 1 + 1; // signature + version + type
        size += self.record_data.len(); // records
        size += num_children * sa; // child addrs
        size += nrec_bytes; // child nrecords
        if depth > 1 {
            size += num_children * ss; // child total nrecords
        }
        size += 4; // checksum
        size
    }

    pub fn encode(
        &self,
        ctx: &FormatContext,
        depth: u16,
        max_nrec_per_node: u32,
    ) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let num_children = self.num_records as usize + 1;
        let size = self.encoded_size(ctx, depth, max_nrec_per_node);
        let mut buf = Vec::with_capacity(size);

        buf.extend_from_slice(&BTIN_SIGNATURE);
        buf.push(BT2_VERSION);
        buf.push(self.record_type);
        buf.extend_from_slice(&self.record_data);

        // Child addresses
        for &addr in &self.child_addrs {
            buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
        }

        // Child nrecords - packed bits
        let bits_per_child = std::cmp::max(1, (32 - max_nrec_per_node.leading_zeros()) as usize);
        let nrec_bytes = Self::nrec_bytes(num_children, max_nrec_per_node);
        let mut packed = vec![0u8; nrec_bytes];
        for (i, &nrec) in self.child_nrecords.iter().enumerate() {
            let bit_offset = i * bits_per_child;
            let byte_offset = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            let value = nrec as u32;
            // Write the value across potentially multiple bytes
            let combined = (value as u64) << bit_shift;
            let bytes = combined.to_le_bytes();
            let bytes_needed = (bits_per_child + bit_shift).div_ceil(8);
            for j in 0..bytes_needed {
                if byte_offset + j < nrec_bytes {
                    packed[byte_offset + j] |= bytes[j];
                }
            }
        }
        buf.extend_from_slice(&packed);

        // Child total nrecords (only if depth > 1)
        if depth > 1 {
            for &total in &self.child_total_nrecords {
                buf.extend_from_slice(&total.to_le_bytes()[..ss]);
            }
        }

        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        num_records: u16,
        record_size: u16,
        depth: u16,
        max_nrec_per_node: u32,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let num_children = num_records as usize + 1;
        let records_len = num_records as usize * record_size as usize;
        let bits_per_child = std::cmp::max(1, (32 - max_nrec_per_node.leading_zeros()) as usize);
        let nrec_bytes = Self::nrec_bytes(num_children, max_nrec_per_node);

        let mut min_size = 4 + 1 + 1 + records_len + num_children * sa + nrec_bytes;
        if depth > 1 {
            min_size += num_children * ss;
        }
        min_size += 4; // checksum

        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != BTIN_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != BT2_VERSION {
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

        let record_type = buf[5];
        let mut pos = 6;

        let record_data = buf[pos..pos + records_len].to_vec();
        pos += records_len;

        let mut child_addrs = Vec::with_capacity(num_children);
        for _ in 0..num_children {
            child_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
        }

        // Decode packed child nrecords
        let packed = &buf[pos..pos + nrec_bytes];
        pos += nrec_bytes;
        let mask = (1u32 << bits_per_child) - 1;
        let mut child_nrecords = Vec::with_capacity(num_children);
        for i in 0..num_children {
            let bit_offset = i * bits_per_child;
            let byte_offset = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            // Read up to 8 bytes starting at byte_offset
            let mut tmp = [0u8; 8];
            let avail = std::cmp::min(8, nrec_bytes - byte_offset);
            tmp[..avail].copy_from_slice(&packed[byte_offset..byte_offset + avail]);
            let value = u64::from_le_bytes(tmp);
            let nrec = ((value >> bit_shift) as u32 & mask) as u16;
            child_nrecords.push(nrec);
        }

        let child_total_nrecords = if depth > 1 {
            let mut totals = Vec::with_capacity(num_children);
            for _ in 0..num_children {
                totals.push(read_size(&buf[pos..], ss));
                pos += ss;
            }
            totals
        } else {
            Vec::new()
        };

        Ok(Self {
            record_type,
            record_data,
            num_records,
            record_size,
            child_addrs,
            child_nrecords,
            child_total_nrecords,
        })
    }
}

// ==========================================================================
// In-memory BT2 chunk index (flat approach)
// ==========================================================================

/// In-memory B-tree v2 chunk index.
///
/// Keeps all records in memory. Serializes as a header + leaf node(s) for
/// small trees. For larger trees, internal nodes would be needed but we use
/// the flat approach for simplicity.
#[derive(Debug, Clone)]
pub struct Bt2ChunkIndex {
    /// Number of dataset dimensions.
    pub ndims: usize,
    /// Whether chunks are filtered.
    pub filtered: bool,
    /// Unfiltered chunk records (used when filtered == false).
    pub records: Vec<Bt2ChunkRecord>,
    /// Filtered chunk records (used when filtered == true).
    pub filtered_records: Vec<Bt2FilteredChunkRecord>,
}

impl Bt2ChunkIndex {
    /// Create a new empty B-tree v2 chunk index for unfiltered chunks.
    pub fn new_unfiltered(ndims: usize) -> Self {
        Self {
            ndims,
            filtered: false,
            records: Vec::new(),
            filtered_records: Vec::new(),
        }
    }

    /// Create a new empty B-tree v2 chunk index for filtered chunks.
    pub fn new_filtered(ndims: usize) -> Self {
        Self {
            ndims,
            filtered: true,
            records: Vec::new(),
            filtered_records: Vec::new(),
        }
    }

    /// Insert an unfiltered chunk record.
    pub fn insert(&mut self, scaled_offsets: Vec<u64>, chunk_address: u64) {
        // Check if a record with the same coordinates already exists
        if let Some(existing) = self.records.iter_mut().find(|r| r.scaled_offsets == scaled_offsets) {
            existing.chunk_address = chunk_address;
        } else {
            self.records.push(Bt2ChunkRecord {
                scaled_offsets,
                chunk_address,
            });
        }
    }

    /// Insert a filtered chunk record.
    pub fn insert_filtered(
        &mut self,
        scaled_offsets: Vec<u64>,
        chunk_address: u64,
        chunk_size: u32,
        filter_mask: u32,
    ) {
        if let Some(existing) = self.filtered_records.iter_mut().find(|r| r.scaled_offsets == scaled_offsets) {
            existing.chunk_address = chunk_address;
            existing.chunk_size = chunk_size;
            existing.filter_mask = filter_mask;
        } else {
            self.filtered_records.push(Bt2FilteredChunkRecord {
                scaled_offsets,
                chunk_address,
                chunk_size,
                filter_mask,
            });
        }
    }

    /// Look up a chunk by its scaled coordinates. Returns the record if found.
    pub fn lookup(&self, scaled_offsets: &[u64]) -> Option<&Bt2ChunkRecord> {
        self.records.iter().find(|r| r.scaled_offsets == scaled_offsets)
    }

    /// Look up a filtered chunk by its scaled coordinates.
    pub fn lookup_filtered(&self, scaled_offsets: &[u64]) -> Option<&Bt2FilteredChunkRecord> {
        self.filtered_records.iter().find(|r| r.scaled_offsets == scaled_offsets)
    }

    /// Iterate all unfiltered records.
    pub fn iter(&self) -> impl Iterator<Item = &Bt2ChunkRecord> {
        self.records.iter()
    }

    /// Iterate all filtered records.
    pub fn iter_filtered(&self) -> impl Iterator<Item = &Bt2FilteredChunkRecord> {
        self.filtered_records.iter()
    }

    /// Total number of records.
    pub fn num_records(&self) -> usize {
        if self.filtered {
            self.filtered_records.len()
        } else {
            self.records.len()
        }
    }

    /// Compute the record size in bytes.
    pub fn record_size(&self, ctx: &FormatContext) -> u16 {
        let sa = ctx.sizeof_addr as usize;
        if self.filtered {
            (self.ndims * 8 + sa + 4 + 4) as u16
        } else {
            (self.ndims * 8 + sa) as u16
        }
    }

    /// Encode all records as raw bytes.
    fn encode_records(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let rec_size = self.record_size(ctx) as usize;
        let num = self.num_records();
        let mut buf = Vec::with_capacity(num * rec_size);

        if self.filtered {
            for rec in &self.filtered_records {
                for &offset in &rec.scaled_offsets {
                    buf.extend_from_slice(&offset.to_le_bytes());
                }
                buf.extend_from_slice(&rec.chunk_address.to_le_bytes()[..sa]);
                buf.extend_from_slice(&rec.chunk_size.to_le_bytes());
                buf.extend_from_slice(&rec.filter_mask.to_le_bytes());
            }
        } else {
            for rec in &self.records {
                for &offset in &rec.scaled_offsets {
                    buf.extend_from_slice(&offset.to_le_bytes());
                }
                buf.extend_from_slice(&rec.chunk_address.to_le_bytes()[..sa]);
            }
        }

        buf
    }

    /// Encode the B-tree as a header + leaf node.
    ///
    /// Returns `(header_bytes, leaf_bytes)`.
    pub fn encode(&self, ctx: &FormatContext) -> (Vec<u8>, Vec<u8>) {
        let rec_size = self.record_size(ctx);
        let num = self.num_records() as u16;

        let record_data = self.encode_records(ctx);

        let leaf = Bt2LeafNode {
            record_type: if self.filtered { BT2_TYPE_CHUNK_FILT } else { BT2_TYPE_CHUNK_UNFILT },
            record_data,
            num_records: num,
            record_size: rec_size,
        };
        let leaf_encoded = leaf.encode();

        // We'll set root_node_addr to UNDEF_ADDR; the caller sets it to the
        // actual leaf address after allocating.
        let header = Bt2Header {
            record_type: if self.filtered { BT2_TYPE_CHUNK_FILT } else { BT2_TYPE_CHUNK_UNFILT },
            node_size: leaf_encoded.len() as u32,
            record_size: rec_size,
            depth: 0,
            split_percent: 100,
            merge_percent: 40,
            root_node_addr: UNDEF_ADDR,
            num_records_in_root: num,
            total_num_records: num as u64,
        };
        let header_encoded = header.encode(ctx);

        (header_encoded, leaf_encoded)
    }

    /// Decode unfiltered records from a leaf node's raw record data.
    pub fn decode_unfiltered_records(
        record_data: &[u8],
        num_records: usize,
        ndims: usize,
        ctx: &FormatContext,
    ) -> FormatResult<Vec<Bt2ChunkRecord>> {
        let sa = ctx.sizeof_addr as usize;
        let rec_size = ndims * 8 + sa;
        if record_data.len() < num_records * rec_size {
            return Err(FormatError::BufferTooShort {
                needed: num_records * rec_size,
                available: record_data.len(),
            });
        }

        let mut records = Vec::with_capacity(num_records);
        let mut pos = 0;
        for _ in 0..num_records {
            let mut scaled_offsets = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                let offset = u64::from_le_bytes([
                    record_data[pos], record_data[pos+1], record_data[pos+2], record_data[pos+3],
                    record_data[pos+4], record_data[pos+5], record_data[pos+6], record_data[pos+7],
                ]);
                scaled_offsets.push(offset);
                pos += 8;
            }
            let chunk_address = read_addr(&record_data[pos..], sa);
            pos += sa;
            records.push(Bt2ChunkRecord {
                scaled_offsets,
                chunk_address,
            });
        }

        Ok(records)
    }

    /// Decode filtered records from a leaf node's raw record data.
    pub fn decode_filtered_records(
        record_data: &[u8],
        num_records: usize,
        ndims: usize,
        ctx: &FormatContext,
    ) -> FormatResult<Vec<Bt2FilteredChunkRecord>> {
        let sa = ctx.sizeof_addr as usize;
        let rec_size = ndims * 8 + sa + 4 + 4;
        if record_data.len() < num_records * rec_size {
            return Err(FormatError::BufferTooShort {
                needed: num_records * rec_size,
                available: record_data.len(),
            });
        }

        let mut records = Vec::with_capacity(num_records);
        let mut pos = 0;
        for _ in 0..num_records {
            let mut scaled_offsets = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                let offset = u64::from_le_bytes([
                    record_data[pos], record_data[pos+1], record_data[pos+2], record_data[pos+3],
                    record_data[pos+4], record_data[pos+5], record_data[pos+6], record_data[pos+7],
                ]);
                scaled_offsets.push(offset);
                pos += 8;
            }
            let chunk_address = read_addr(&record_data[pos..], sa);
            pos += sa;
            let chunk_size = u32::from_le_bytes([
                record_data[pos], record_data[pos+1], record_data[pos+2], record_data[pos+3],
            ]);
            pos += 4;
            let filter_mask = u32::from_le_bytes([
                record_data[pos], record_data[pos+1], record_data[pos+2], record_data[pos+3],
            ]);
            pos += 4;
            records.push(Bt2FilteredChunkRecord {
                scaled_offsets,
                chunk_address,
                chunk_size,
                filter_mask,
            });
        }

        Ok(records)
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

    // ---- Header tests ----

    #[test]
    fn header_roundtrip() {
        let mut hdr = Bt2Header::new_for_chunks(&ctx8(), 2);
        hdr.root_node_addr = 0x3000;
        hdr.num_records_in_root = 5;
        hdr.total_num_records = 5;

        let encoded = hdr.encode(&ctx8());
        assert_eq!(encoded.len(), hdr.encoded_size(&ctx8()));
        assert_eq!(&encoded[..4], b"BTHD");

        let decoded = Bt2Header::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_roundtrip_ctx4() {
        let hdr = Bt2Header::new_for_chunks(&ctx4(), 3);
        let encoded = hdr.encode(&ctx4());
        let decoded = Bt2Header::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_filtered_roundtrip() {
        let hdr = Bt2Header::new_for_filtered_chunks(&ctx8(), 2);
        assert_eq!(hdr.record_type, BT2_TYPE_CHUNK_FILT);
        // record_size = 2*8 + 8 + 4 + 4 = 32
        assert_eq!(hdr.record_size, 32);

        let encoded = hdr.encode(&ctx8());
        let decoded = Bt2Header::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn header_bad_signature() {
        let hdr = Bt2Header::new_for_chunks(&ctx8(), 2);
        let mut encoded = hdr.encode(&ctx8());
        encoded[0] = b'X';
        let err = Bt2Header::decode(&encoded, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn header_checksum_mismatch() {
        let hdr = Bt2Header::new_for_chunks(&ctx8(), 2);
        let mut encoded = hdr.encode(&ctx8());
        encoded[6] ^= 0xFF;
        let err = Bt2Header::decode(&encoded, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    // ---- Leaf node tests ----

    #[test]
    fn leaf_node_roundtrip() {
        let mut leaf = Bt2LeafNode::new(BT2_TYPE_CHUNK_UNFILT, 24);
        // 2 records, each 24 bytes (2 dims * 8 + 8 addr)
        let rec1 = [0u8; 24];
        let mut rec2 = [0u8; 24];
        rec2[0] = 1; // scaled_offset[0] = 1
        leaf.record_data.extend_from_slice(&rec1);
        leaf.record_data.extend_from_slice(&rec2);
        leaf.num_records = 2;

        let encoded = leaf.encode();
        assert_eq!(&encoded[..4], b"BTLF");

        let decoded = Bt2LeafNode::decode(&encoded, 2, 24).unwrap();
        assert_eq!(decoded.record_data, leaf.record_data);
        assert_eq!(decoded.record_type, BT2_TYPE_CHUNK_UNFILT);
    }

    #[test]
    fn leaf_node_empty() {
        let leaf = Bt2LeafNode::new(BT2_TYPE_CHUNK_UNFILT, 24);
        let encoded = leaf.encode();
        let decoded = Bt2LeafNode::decode(&encoded, 0, 24).unwrap();
        assert!(decoded.record_data.is_empty());
    }

    #[test]
    fn leaf_node_bad_checksum() {
        let mut leaf = Bt2LeafNode::new(BT2_TYPE_CHUNK_UNFILT, 8);
        leaf.record_data = vec![0u8; 8];
        leaf.num_records = 1;
        let mut encoded = leaf.encode();
        encoded[6] ^= 0xFF;
        let err = Bt2LeafNode::decode(&encoded, 1, 8).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    // ---- Internal node tests ----

    #[test]
    fn internal_node_roundtrip() {
        let rec_size = 24u16;
        let mut node = Bt2InternalNode::new(BT2_TYPE_CHUNK_UNFILT, rec_size);
        node.record_data = vec![0xAA; rec_size as usize]; // 1 record
        node.num_records = 1;
        node.child_addrs = vec![0x1000, 0x2000]; // 2 children
        node.child_nrecords = vec![3, 5];

        let max_nrec = 10u32;
        let encoded = node.encode(&ctx8(), 1, max_nrec);
        assert_eq!(&encoded[..4], b"BTIN");

        let decoded = Bt2InternalNode::decode(
            &encoded, &ctx8(), 1, rec_size, 1, max_nrec,
        ).unwrap();
        assert_eq!(decoded.record_data, node.record_data);
        assert_eq!(decoded.child_addrs, node.child_addrs);
        assert_eq!(decoded.child_nrecords, node.child_nrecords);
    }

    #[test]
    fn internal_node_depth2_roundtrip() {
        let rec_size = 16u16;
        let mut node = Bt2InternalNode::new(BT2_TYPE_CHUNK_UNFILT, rec_size);
        node.record_data = vec![0xBB; rec_size as usize * 2]; // 2 records
        node.num_records = 2;
        node.child_addrs = vec![0x1000, 0x2000, 0x3000]; // 3 children
        node.child_nrecords = vec![4, 6, 2];
        node.child_total_nrecords = vec![100, 200, 50];

        let max_nrec = 16u32;
        let encoded = node.encode(&ctx8(), 2, max_nrec);

        let decoded = Bt2InternalNode::decode(
            &encoded, &ctx8(), 2, rec_size, 2, max_nrec,
        ).unwrap();
        assert_eq!(decoded.child_total_nrecords, vec![100, 200, 50]);
        assert_eq!(decoded.child_nrecords, vec![4, 6, 2]);
    }

    // ---- In-memory index tests ----

    #[test]
    fn chunk_index_insert_and_lookup() {
        let mut idx = Bt2ChunkIndex::new_unfiltered(2);
        idx.insert(vec![0, 0], 0x1000);
        idx.insert(vec![0, 1], 0x2000);
        idx.insert(vec![1, 0], 0x3000);

        assert_eq!(idx.num_records(), 3);

        let r = idx.lookup(&[0, 1]).unwrap();
        assert_eq!(r.chunk_address, 0x2000);

        assert!(idx.lookup(&[2, 2]).is_none());
    }

    #[test]
    fn chunk_index_insert_replaces() {
        let mut idx = Bt2ChunkIndex::new_unfiltered(2);
        idx.insert(vec![0, 0], 0x1000);
        idx.insert(vec![0, 0], 0x2000); // replace
        assert_eq!(idx.num_records(), 1);
        assert_eq!(idx.lookup(&[0, 0]).unwrap().chunk_address, 0x2000);
    }

    #[test]
    fn chunk_index_iterate() {
        let mut idx = Bt2ChunkIndex::new_unfiltered(1);
        for i in 0..5 {
            idx.insert(vec![i], 0x1000 + i * 0x100);
        }
        let addrs: Vec<u64> = idx.iter().map(|r| r.chunk_address).collect();
        assert_eq!(addrs.len(), 5);
    }

    #[test]
    fn chunk_index_encode_decode_roundtrip() {
        let ctx = ctx8();
        let mut idx = Bt2ChunkIndex::new_unfiltered(2);
        idx.insert(vec![0, 0], 0x1000);
        idx.insert(vec![0, 1], 0x2000);
        idx.insert(vec![1, 0], 0x3000);

        let (hdr_bytes, leaf_bytes) = idx.encode(&ctx);

        // Decode header
        let hdr = Bt2Header::decode(&hdr_bytes, &ctx).unwrap();
        assert_eq!(hdr.record_type, BT2_TYPE_CHUNK_UNFILT);
        assert_eq!(hdr.depth, 0);
        assert_eq!(hdr.total_num_records, 3);
        assert_eq!(hdr.num_records_in_root, 3);
        // record_size = 2*8 + 8 = 24
        assert_eq!(hdr.record_size, 24);

        // Decode leaf
        let leaf = Bt2LeafNode::decode(&leaf_bytes, 3, hdr.record_size).unwrap();
        let records = Bt2ChunkIndex::decode_unfiltered_records(
            &leaf.record_data, 3, 2, &ctx,
        ).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].scaled_offsets, vec![0, 0]);
        assert_eq!(records[0].chunk_address, 0x1000);
        assert_eq!(records[1].scaled_offsets, vec![0, 1]);
        assert_eq!(records[1].chunk_address, 0x2000);
        assert_eq!(records[2].scaled_offsets, vec![1, 0]);
        assert_eq!(records[2].chunk_address, 0x3000);
    }

    #[test]
    fn filtered_chunk_index_encode_decode_roundtrip() {
        let ctx = ctx8();
        let mut idx = Bt2ChunkIndex::new_filtered(2);
        idx.insert_filtered(vec![0, 0], 0x1000, 512, 0);
        idx.insert_filtered(vec![1, 0], 0x2000, 300, 1);

        let (hdr_bytes, leaf_bytes) = idx.encode(&ctx);

        let hdr = Bt2Header::decode(&hdr_bytes, &ctx).unwrap();
        assert_eq!(hdr.record_type, BT2_TYPE_CHUNK_FILT);
        assert_eq!(hdr.total_num_records, 2);
        // record_size = 2*8 + 8 + 4 + 4 = 32
        assert_eq!(hdr.record_size, 32);

        let leaf = Bt2LeafNode::decode(&leaf_bytes, 2, hdr.record_size).unwrap();
        let records = Bt2ChunkIndex::decode_filtered_records(
            &leaf.record_data, 2, 2, &ctx,
        ).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].chunk_address, 0x1000);
        assert_eq!(records[0].chunk_size, 512);
        assert_eq!(records[0].filter_mask, 0);
        assert_eq!(records[1].chunk_address, 0x2000);
        assert_eq!(records[1].chunk_size, 300);
        assert_eq!(records[1].filter_mask, 1);
    }

    #[test]
    fn chunk_index_ctx4_roundtrip() {
        let ctx = ctx4();
        let mut idx = Bt2ChunkIndex::new_unfiltered(1);
        idx.insert(vec![0], 0x100);
        idx.insert(vec![1], 0x200);

        let (hdr_bytes, leaf_bytes) = idx.encode(&ctx);
        let hdr = Bt2Header::decode(&hdr_bytes, &ctx).unwrap();
        // record_size = 1*8 + 4 = 12
        assert_eq!(hdr.record_size, 12);

        let leaf = Bt2LeafNode::decode(&leaf_bytes, 2, hdr.record_size).unwrap();
        let records = Bt2ChunkIndex::decode_unfiltered_records(
            &leaf.record_data, 2, 1, &ctx,
        ).unwrap();
        assert_eq!(records[0].chunk_address, 0x100);
        assert_eq!(records[1].chunk_address, 0x200);
    }

    #[test]
    fn empty_chunk_index() {
        let ctx = ctx8();
        let idx = Bt2ChunkIndex::new_unfiltered(3);
        assert_eq!(idx.num_records(), 0);

        let (hdr_bytes, leaf_bytes) = idx.encode(&ctx);
        let hdr = Bt2Header::decode(&hdr_bytes, &ctx).unwrap();
        assert_eq!(hdr.total_num_records, 0);

        let leaf = Bt2LeafNode::decode(&leaf_bytes, 0, hdr.record_size).unwrap();
        assert!(leaf.record_data.is_empty());
    }
}
