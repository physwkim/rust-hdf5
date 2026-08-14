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

use crate::format::bytes::{read_le_addr as read_addr, read_le_uint as read_size};
use crate::format::checksum::checksum_metadata;
use crate::format::{BlockReader, FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// Signature for the B-tree v2 header.
pub const BTHD_SIGNATURE: [u8; 4] = *b"BTHD";
/// Signature for the B-tree v2 internal node.
pub const BTIN_SIGNATURE: [u8; 4] = *b"BTIN";
/// Signature for the B-tree v2 leaf node.
pub const BTLF_SIGNATURE: [u8; 4] = *b"BTLF";

/// B-tree v2 version.
pub const BT2_VERSION: u8 = 0;

/// Node size for a chunk-index v2 B-tree, matching libhdf5's
/// `H5D_BT2_NODE_SIZE` (`H5Dpkg.h`).
///
/// Every node — leaf or internal — occupies exactly this many bytes on disk.
/// That is what makes a node block reusable: re-serializing a tree overwrites
/// its nodes in place rather than relocating them, so a flush cannot orphan the
/// block it replaced. libhdf5 reads a whole node-size block and checksums only
/// the used prefix, so the tail is padding that must nevertheless exist in the
/// file.
pub const BT2_NODE_SIZE: u32 = 2048;

/// Percentage full a node must be before it splits (`H5D_BT2_SPLIT_PERC`).
pub const BT2_SPLIT_PERCENT: u8 = 100;

/// Percentage below which a node is merged (`H5D_BT2_MERGE_PERC`).
pub const BT2_MERGE_PERCENT: u8 = 40;

/// Record type: indirectly accessed, unfiltered "huge" fractal heap objects
/// (`H5B2_FHEAP_HUGE_INDIR_ID`).
pub const BT2_TYPE_FHEAP_HUGE_INDIR: u8 = 1;
/// Record type: name index for dense link storage in groups
/// (`H5B2_GRP_DENSE_NAME_ID`).
pub const BT2_TYPE_GRP_NAME: u8 = 5;
/// Record type: creation-order index for dense link storage in groups
/// (`H5B2_GRP_DENSE_CORDER_ID`).
pub const BT2_TYPE_GRP_CORDER: u8 = 6;
/// Record type: name index for dense attribute storage
/// (`H5B2_ATTR_DENSE_NAME_ID`).
pub const BT2_TYPE_ATTR_NAME: u8 = 8;
/// Record type: creation-order index for dense attribute storage
/// (`H5B2_ATTR_DENSE_CORDER_ID`).
pub const BT2_TYPE_ATTR_CORDER: u8 = 9;
/// Record type: unfiltered chunks (non-filtered chunked datasets).
pub const BT2_TYPE_CHUNK_UNFILT: u8 = 10;
/// Record type: filtered chunks (filtered chunked datasets).
pub const BT2_TYPE_CHUNK_FILT: u8 = 11;

/// Bytes used to encode a filtered chunk's compressed size in a v2 B-tree
/// record.
///
/// libhdf5 does not read this width off the record; it recomputes it
/// (`H5D_BT2_COMPUTE_CHUNK_SIZE_LEN`, `H5Dbtree2.c`) from the layout message
/// version: `sizeof_size` for version 5, and otherwise the same
/// magnitude-derived width the extensible and fixed arrays use. Our writer
/// emits version-4 layout messages, so a file it writes must use the latter —
/// hence [`compute_chunk_size_len`], shared with those two indexes.
///
/// Decoding stays width-agnostic: [`Bt2ChunkIndex::decode_filtered_records`]
/// derives the width from the header's `record_size`, so a version-5 file
/// written by libhdf5 reads back just as well.
pub use super::extensible_array::compute_chunk_size_len;

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
    /// Size of the chunk after filtering (compressed size). `u64` because the
    /// encoded field is `chunk_size_len` bytes wide — up to 8 under a
    /// version-5 layout (or a v4 layout whose uncompressed chunk derives an
    /// 8-byte width).
    pub chunk_size: u64,
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
            node_size: BT2_NODE_SIZE,
            record_size,
            depth: 0,
            split_percent: BT2_SPLIT_PERCENT,
            merge_percent: BT2_MERGE_PERCENT,
            root_node_addr: UNDEF_ADDR,
            num_records_in_root: 0,
            total_num_records: 0,
        }
    }

    /// Create a new B-tree v2 header for filtered chunk indexing.
    ///
    /// `ndims` is the number of dataset dimensions and `chunk_size_len` the
    /// width of the compressed-size field (see [`compute_chunk_size_len`]).
    pub fn new_for_filtered_chunks(ctx: &FormatContext, ndims: usize, chunk_size_len: u8) -> Self {
        // record_size = address + chunk_size + filter_mask(4) + scaled offsets
        let record_size =
            (ctx.sizeof_addr as usize + chunk_size_len as usize + 4 + ndims * 8) as u16;
        Self {
            record_type: BT2_TYPE_CHUNK_FILT,
            node_size: BT2_NODE_SIZE,
            record_size,
            depth: 0,
            split_percent: BT2_SPLIT_PERCENT,
            merge_percent: BT2_MERGE_PERCENT,
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
        let record_type = buf[pos];
        pos += 1;
        let node_size = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        pos += 4;
        let record_size = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let depth = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;

        // Validate the geometry-driving fields, which come straight off disk.
        // node_size must hold at least the metadata prefix
        // (H5B2_METADATA_PREFIX_SIZE = 10); record_size must be non-zero; and
        // depth is capped so the node-info geometry cannot overflow and the
        // recursive node walk cannot exhaust the stack. A v2 B-tree whose
        // every node holds the minimum two records still reaches u64 capacity
        // well before depth 64, so any larger depth signals corruption.
        if (node_size as u64) < 10 {
            return Err(FormatError::InvalidData(format!(
                "v2 B-tree node_size {node_size} is smaller than the metadata prefix"
            )));
        }
        if record_size == 0 {
            return Err(FormatError::InvalidData(
                "v2 B-tree record_size must be non-zero".into(),
            ));
        }
        if depth > 64 {
            return Err(FormatError::InvalidData(format!(
                "v2 B-tree depth {depth} is implausibly large"
            )));
        }

        let split_percent = buf[pos];
        pos += 1;
        let merge_percent = buf[pos];
        pos += 1;
        let root_node_addr = read_addr(&buf[pos..], sa);
        pos += sa;
        let num_records_in_root = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
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
        let records_len = (num_records as usize).saturating_mul(record_size as usize);
        let min_size = records_len.saturating_add(10);

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

    /// Encoded size of an internal node at `depth` with `nrec` records.
    pub fn encoded_size(
        ctx: &FormatContext,
        depth: u16,
        nrec: u16,
        rrec_size: u16,
        max_nrec_size: u8,
        child_total_size: u8,
    ) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let nchild = nrec as usize + 1;
        let ptr = sa
            + max_nrec_size as usize
            + if depth > 1 {
                child_total_size as usize
            } else {
                0
            };
        4 + 1 + 1 + nrec as usize * rrec_size as usize + nchild * ptr + 4
    }

    /// Encode an internal node (libhdf5 `H5B2__cache_int_serialize` layout):
    /// each child pointer is `address + node_nrec + [total_nrec if depth>1]`,
    /// where the widths come from the B-tree geometry.
    pub fn encode(
        &self,
        ctx: &FormatContext,
        depth: u16,
        max_nrec_size: u8,
        child_total_size: u8,
    ) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let nchild = self.num_records as usize + 1;
        let size = Self::encoded_size(
            ctx,
            depth,
            self.num_records,
            self.record_size,
            max_nrec_size,
            child_total_size,
        );
        debug_assert_eq!(self.child_addrs.len(), nchild);
        debug_assert_eq!(self.child_nrecords.len(), nchild);
        debug_assert!(depth <= 1 || self.child_total_nrecords.len() == nchild);
        let mut buf = Vec::with_capacity(size);
        buf.extend_from_slice(&BTIN_SIGNATURE);
        buf.push(BT2_VERSION);
        buf.push(self.record_type);
        buf.extend_from_slice(&self.record_data);
        for i in 0..nchild {
            buf.extend_from_slice(&self.child_addrs[i].to_le_bytes()[..sa]);
            buf.extend_from_slice(
                &(self.child_nrecords[i] as u64).to_le_bytes()[..max_nrec_size as usize],
            );
            if depth > 1 {
                buf.extend_from_slice(
                    &self.child_total_nrecords[i].to_le_bytes()[..child_total_size as usize],
                );
            }
        }
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        debug_assert_eq!(buf.len(), size);
        buf
    }

    /// Decode an internal node. `depth` is this node's depth (children are at
    /// `depth - 1`); `nrec` its record count; the size widths come from the
    /// B-tree geometry.
    pub fn decode(
        buf: &[u8],
        ctx: &FormatContext,
        depth: u16,
        nrec: u16,
        rrec_size: u16,
        max_nrec_size: u8,
        child_total_size: u8,
    ) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let nchild = nrec as usize + 1;
        let records_len = (nrec as usize).saturating_mul(rrec_size as usize);
        let ptr = sa
            + max_nrec_size as usize
            + if depth > 1 {
                child_total_size as usize
            } else {
                0
            };
        let min_size = records_len
            .saturating_add(nchild.saturating_mul(ptr))
            .saturating_add(10);
        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }
        if buf[0..4] != BTIN_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }
        if buf[4] != BT2_VERSION {
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
        let record_type = buf[5];
        let mut pos = 6;
        let record_data = buf[pos..pos + records_len].to_vec();
        pos += records_len;
        let mut child_addrs = Vec::with_capacity(nchild);
        let mut child_nrecords = Vec::with_capacity(nchild);
        let mut child_total_nrecords = Vec::with_capacity(nchild);
        for _ in 0..nchild {
            child_addrs.push(read_addr(&buf[pos..], sa));
            pos += sa;
            child_nrecords.push(read_size(&buf[pos..], max_nrec_size as usize) as u16);
            pos += max_nrec_size as usize;
            if depth > 1 {
                child_total_nrecords.push(read_size(&buf[pos..], child_total_size as usize));
                pos += child_total_size as usize;
            }
        }
        Ok(Self {
            record_type,
            record_data,
            num_records: nrec,
            record_size: rrec_size,
            child_addrs,
            child_nrecords,
            child_total_nrecords,
        })
    }
}

/// Per-depth v2 B-tree node geometry (`H5B2_node_info_t`).
#[derive(Debug, Clone, Copy)]
pub struct Bt2NodeInfo {
    /// Maximum records a node at this depth holds.
    pub max_nrec: u64,
    /// Maximum records in this node and all nodes below it.
    pub cum_max_nrec: u64,
    /// Bytes needed to encode `cum_max_nrec`.
    pub cum_max_nrec_size: u8,
}

/// v2 B-tree node geometry derived from the header parameters, matching
/// libhdf5 `H5B2__hdr_init`.
#[derive(Debug, Clone)]
pub struct Bt2Geometry {
    /// Bytes used to encode a child node's record count.
    pub max_nrec_size: u8,
    /// Node info indexed by depth (`0..=depth`).
    pub node_info: Vec<Bt2NodeInfo>,
}

/// libhdf5 `H5VM_limit_enc_size`: bytes to encode values in `0..=limit`.
fn limit_enc_size(limit: u64) -> u8 {
    let log2 = if limit == 0 {
        0
    } else {
        63 - limit.leading_zeros()
    };
    (log2 / 8 + 1) as u8
}

impl Bt2Geometry {
    /// v2 B-tree prefix size (`H5B2_METADATA_PREFIX_SIZE`): magic + version
    /// + type + checksum.
    const PREFIX: u64 = 10;

    pub fn new(node_size: u32, rrec_size: u16, depth: u16, sizeof_addr: u8) -> Self {
        // Saturating arithmetic throughout: `node_size` and `depth` are
        // validated by `Bt2Header::decode`, but this stays panic-free even
        // if constructed directly with degenerate parameters.
        let rrec = rrec_size.max(1) as u64;
        let leaf_max = (node_size as u64).saturating_sub(Self::PREFIX) / rrec;
        let max_nrec_size = limit_enc_size(leaf_max);
        let mut node_info = vec![Bt2NodeInfo {
            max_nrec: leaf_max,
            cum_max_nrec: leaf_max,
            cum_max_nrec_size: 0,
        }];
        for d in 1..=depth as usize {
            let ptr = sizeof_addr as u64
                + max_nrec_size as u64
                + node_info[d - 1].cum_max_nrec_size as u64;
            let max_nrec = if (node_size as u64) > Self::PREFIX + ptr {
                (node_size as u64 - Self::PREFIX - ptr) / (rrec + ptr)
            } else {
                0
            };
            let cum = max_nrec
                .saturating_add(1)
                .saturating_mul(node_info[d - 1].cum_max_nrec)
                .saturating_add(max_nrec);
            node_info.push(Bt2NodeInfo {
                max_nrec,
                cum_max_nrec: cum,
                cum_max_nrec_size: limit_enc_size(cum),
            });
        }
        Self {
            max_nrec_size,
            node_info,
        }
    }

    /// Width of a child's "total records" field for a node at `depth`
    /// (0 unless `depth > 1`).
    pub fn child_total_size(&self, depth: u16) -> u8 {
        if depth > 1 {
            self.node_info[depth as usize - 1].cum_max_nrec_size
        } else {
            0
        }
    }
}

// ==========================================================================
// Whole-tree record walk
// ==========================================================================

/// One walk of a v2 B-tree: every record it read, and every node block it
/// read them from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bt2Walk {
    /// Records in tree order, packed at `header.record_size` bytes each.
    pub records: Vec<u8>,
    /// Address of every node visited, in visit order. All are `node_size`
    /// bytes long, so this is the tree's node footprint as well as its shape.
    pub node_addrs: Vec<u64>,
}

/// Read every record in a v2 B-tree, in tree order, as one packed byte run of
/// `header.record_size`-byte records.
///
/// Record-type agnostic: the caller decodes the bytes according to
/// `header.record_type`. Both v2 B-tree users — the chunk index and the dense
/// link/attribute name index — go through this one walker so the node geometry
/// is derived in a single place.
pub fn collect_btree_v2_records<R: BlockReader>(
    header: &Bt2Header,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<u8>> {
    Ok(walk_btree_v2(header, ctx, reader)?.records)
}

/// [`collect_btree_v2_records`] plus the address of every node the walk read.
///
/// One traversal serves both: a caller that only wants the records goes
/// through the wrapper above, and a caller freeing the tree's file space needs
/// the node addresses the same walk already visited.
pub fn walk_btree_v2<R: BlockReader>(
    header: &Bt2Header,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Bt2Walk> {
    let mut out = Bt2Walk::default();
    if header.root_node_addr == UNDEF_ADDR || header.total_num_records == 0 {
        return Ok(out);
    }
    let geo = Bt2Geometry::new(
        header.node_size,
        header.record_size,
        header.depth,
        ctx.sizeof_addr,
    );
    collect_node(
        reader,
        ctx,
        &geo,
        header.root_node_addr,
        header.depth,
        header.num_records_in_root,
        header.record_size,
        header.node_size,
        &mut out,
    )?;
    Ok(out)
}

/// Every file extent the v2 B-tree headed at `header_addr` occupies: the
/// header block and every node block.
///
/// The single place that knows a v2 B-tree's on-disk footprint, so freeing one
/// does not re-derive the sizes its writer allocated
/// (`Bt2Header::encoded_size` for the header, `node_size` for each node).
pub fn collect_btree_v2_extents<R: BlockReader>(
    header_addr: u64,
    ctx: &FormatContext,
    reader: &mut R,
) -> FormatResult<Vec<(u64, u64)>> {
    // The header's on-disk size depends only on the address/length widths, so
    // a generous prefix read covers it.
    let buf = reader.read_block(header_addr, 256)?;
    let header = Bt2Header::decode(&buf, ctx)?;
    let walk = walk_btree_v2(&header, ctx, reader)?;
    let mut extents = Vec::with_capacity(walk.node_addrs.len() + 1);
    extents.push((header_addr, header.encoded_size(ctx) as u64));
    extents.extend(
        walk.node_addrs
            .into_iter()
            .map(|addr| (addr, header.node_size as u64)),
    );
    Ok(extents)
}

#[allow(clippy::too_many_arguments)]
fn collect_node<R: BlockReader>(
    reader: &mut R,
    ctx: &FormatContext,
    geo: &Bt2Geometry,
    addr: u64,
    depth: u16,
    nrec: u16,
    record_size: u16,
    node_size: u32,
    out: &mut Bt2Walk,
) -> FormatResult<()> {
    out.node_addrs.push(addr);
    let buf = reader.read_block(addr, node_size as usize)?;
    if depth == 0 {
        let leaf = Bt2LeafNode::decode(&buf, nrec, record_size)?;
        out.records.extend_from_slice(&leaf.record_data);
        return Ok(());
    }
    let node = Bt2InternalNode::decode(
        &buf,
        ctx,
        depth,
        nrec,
        record_size,
        geo.max_nrec_size,
        geo.child_total_size(depth),
    )?;
    // A node's own records separate its children, so key order is
    // child[0], record[0], child[1], record[1], ... record[n-1], child[n] --
    // an in-order walk. Emitting the node's records ahead of its children
    // would hand the caller a sequence that is sorted only within each node.
    let children: Vec<(u64, u16)> = node
        .child_addrs
        .iter()
        .zip(node.child_nrecords.iter())
        .map(|(&a, &n)| (a, n))
        .collect();
    let rec = record_size as usize;
    for (i, (child_addr, child_nrec)) in children.into_iter().enumerate() {
        collect_node(
            reader,
            ctx,
            geo,
            child_addr,
            depth - 1,
            child_nrec,
            record_size,
            node_size,
            out,
        )?;
        if let Some(record) = node.record_data.get(i * rec..(i + 1) * rec) {
            out.records.extend_from_slice(record);
        }
    }
    Ok(())
}

// ==========================================================================
// Bulk-loaded tree
// ==========================================================================

/// One node of a bulk-loaded v2 B-tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Bt2TreeNode {
    /// Height above the leaves; 0 for a leaf.
    pub depth: u16,
    /// The records this node holds directly, already encoded.
    pub record_data: Vec<u8>,
    /// How many records that is.
    pub num_records: u16,
    /// Indices into [`Bt2Tree::nodes`] of this node's children. A node holding
    /// *m* records has *m + 1* children; a leaf has none.
    pub children: Vec<usize>,
    /// Records in this node and every node beneath it.
    pub total_records: u64,
}

/// A v2 B-tree bulk-loaded from an ordered record list.
///
/// Nodes are laid out children-before-parents, so [`nodes`](Self::nodes)'s last
/// entry is the root and every node's children have smaller indices — which is
/// what lets [`encode`](Self::encode) resolve child addresses in one pass.
///
/// Every node is exactly [`node_size`](Self::node_size) bytes, so re-loading a
/// grown index overwrites the nodes already on disk and only ever *appends*
/// blocks. Sizing the root to its contents instead (one leaf that grows with
/// the record count) would force a relocation on every flush, orphaning the
/// block it replaced, and would cap the index at the 65535 records a node's
/// record count can express.
#[derive(Debug, Clone)]
pub struct Bt2Tree {
    /// Every node, children before parents; the last entry is the root.
    pub nodes: Vec<Bt2TreeNode>,
    /// Record type (10 = unfiltered chunks, 11 = filtered chunks).
    pub record_type: u8,
    /// Size of one record in bytes.
    pub record_size: u16,
    /// Size of every node in bytes.
    pub node_size: u32,
    /// Split percentage the header declares (pass-through; see
    /// [`Bt2ChunkIndex::split_percent`]).
    pub split_percent: u8,
    /// Merge percentage the header declares (pass-through).
    pub merge_percent: u8,
    /// Node geometry for depths `0..=depth()`.
    pub geometry: Bt2Geometry,
}

impl Bt2Tree {
    /// Bulk-load a tree from `records` — the encoded records in key order.
    ///
    /// Each level's records are spread evenly across that level's nodes, with
    /// one record promoted to the parent between adjacent siblings. That is the
    /// shape libhdf5 searches: `H5B2__locate_record` bisects a node's records
    /// and, on a miss, descends into the child the record would fall between.
    ///
    /// Node capacities come from [`Bt2Geometry`], so no node exceeds what a
    /// reader computing the same geometry will deserialize.
    pub fn build(
        record_type: u8,
        record_size: u16,
        node_size: u32,
        sizeof_addr: u8,
        records: &[u8],
    ) -> Self {
        let rec = record_size.max(1) as usize;
        let total = records.len() / rec;
        let mut nodes: Vec<Bt2TreeNode> = Vec::new();
        // Records still to be placed at the current level: the chunk records
        // at depth 0, and the separators promoted from below at each depth
        // above it.
        let mut pending: Vec<u8> = records[..total * rec].to_vec();
        // Nodes of the level below, in key order.
        let mut children: Vec<usize> = Vec::new();
        let mut depth: u16 = 0;

        // An empty index has no nodes at all: the header's root address stays
        // undefined, the state libhdf5 leaves a freshly created B-tree in.
        if total > 0 {
            loop {
                let geo = Bt2Geometry::new(node_size, record_size, depth, sizeof_addr);
                let cap = geo.node_info[depth as usize].max_nrec.max(1) as usize;
                let n = pending.len() / rec;
                // Fewest nodes that hold n records with one separator between
                // each adjacent pair: k * cap + (k - 1) >= n.
                let k = (n + 1).div_ceil(cap + 1);
                let own = n - (k - 1);
                let (base, extra) = (own / k, own % k);
                debug_assert!(base >= 1, "a bulk-loaded node must hold a record");
                debug_assert!(base + usize::from(extra > 0) <= cap);

                let mut next_pending: Vec<u8> = Vec::new();
                let mut next_children: Vec<usize> = Vec::with_capacity(k);
                let (mut r, mut c) = (0usize, 0usize);
                for i in 0..k {
                    let m = base + usize::from(i < extra);
                    let record_data = pending[r * rec..(r + m) * rec].to_vec();
                    r += m;
                    let kids: Vec<usize> = if depth == 0 {
                        Vec::new()
                    } else {
                        let kids = children[c..c + m + 1].to_vec();
                        c += m + 1;
                        kids
                    };
                    let total_records =
                        m as u64 + kids.iter().map(|&j| nodes[j].total_records).sum::<u64>();
                    nodes.push(Bt2TreeNode {
                        depth,
                        record_data,
                        num_records: m as u16,
                        children: kids,
                        total_records,
                    });
                    next_children.push(nodes.len() - 1);
                    // The record between this node and the next is the
                    // separator their parent holds.
                    if i + 1 < k {
                        next_pending.extend_from_slice(&pending[r * rec..(r + 1) * rec]);
                        r += 1;
                    }
                }
                if k == 1 {
                    break;
                }
                pending = next_pending;
                children = next_children;
                depth += 1;
            }
        }

        Self {
            geometry: Bt2Geometry::new(node_size, record_size, depth, sizeof_addr),
            nodes,
            record_type,
            record_size,
            node_size,
            split_percent: BT2_SPLIT_PERCENT,
            merge_percent: BT2_MERGE_PERCENT,
        }
    }

    /// Depth of the root (0 = the root is a leaf).
    pub fn depth(&self) -> u16 {
        self.nodes.last().map_or(0, |n| n.depth)
    }

    /// Records held directly by the root.
    pub fn root_num_records(&self) -> u16 {
        self.nodes.last().map_or(0, |n| n.num_records)
    }

    /// Records in the whole tree.
    pub fn total_records(&self) -> u64 {
        self.nodes.last().map_or(0, |n| n.total_records)
    }

    /// Serialize every node to a [`node_size`](Self::node_size)-byte image, in
    /// [`nodes`](Self::nodes) order.
    ///
    /// `addrs[i]` is the file address assigned to `nodes[i]`; entries past the
    /// node count are ignored, so a caller may pass a longer block pool.
    pub fn encode(&self, ctx: &FormatContext, addrs: &[u64]) -> Vec<Vec<u8>> {
        self.nodes
            .iter()
            .map(|n| {
                let mut image = if n.depth == 0 {
                    Bt2LeafNode {
                        record_type: self.record_type,
                        record_data: n.record_data.clone(),
                        num_records: n.num_records,
                        record_size: self.record_size,
                    }
                    .encode()
                } else {
                    Bt2InternalNode {
                        record_type: self.record_type,
                        record_data: n.record_data.clone(),
                        num_records: n.num_records,
                        record_size: self.record_size,
                        child_addrs: n.children.iter().map(|&c| addrs[c]).collect(),
                        child_nrecords: n
                            .children
                            .iter()
                            .map(|&c| self.nodes[c].num_records)
                            .collect(),
                        child_total_nrecords: n
                            .children
                            .iter()
                            .map(|&c| self.nodes[c].total_records)
                            .collect(),
                    }
                    .encode(
                        ctx,
                        n.depth,
                        self.geometry.max_nrec_size,
                        self.geometry.child_total_size(n.depth),
                    )
                };
                debug_assert!(image.len() <= self.node_size as usize);
                // Only the used prefix is checksummed, but the image is padded
                // to the whole block so re-serializing overwrites the block
                // rather than a prefix of it. A node's record count falls as
                // well as rises — a full 84-record leaf becomes two 42-record
                // leaves when the tree splits — so a short write would leave
                // the tail of the previous, larger image behind. A conforming
                // reader stops at the record count its parent gives it and
                // never sees those bytes, but they are stale records in a live
                // node block, and anything scanning the file raw reads them as
                // real.
                image.resize(self.node_size as usize, 0);
                image
            })
            .collect()
    }

    /// The header describing this tree. `root_addr` is the address given to the
    /// last node, and is ignored for an empty tree (whose root is undefined,
    /// the state libhdf5 leaves a freshly created B-tree in).
    pub fn header(&self, root_addr: u64) -> Bt2Header {
        Bt2Header {
            record_type: self.record_type,
            node_size: self.node_size,
            record_size: self.record_size,
            depth: self.depth(),
            split_percent: self.split_percent,
            merge_percent: self.merge_percent,
            root_node_addr: if self.nodes.is_empty() {
                UNDEF_ADDR
            } else {
                root_addr
            },
            num_records_in_root: self.root_num_records(),
            total_num_records: self.total_records(),
        }
    }
}

// ==========================================================================
// In-memory BT2 chunk index (flat approach)
// ==========================================================================

/// In-memory B-tree v2 chunk index.
///
/// Keeps every record in memory as one ordered list and bulk-loads it into a
/// tree of fixed-size nodes on demand — see [`build_tree`](Self::build_tree)
/// and [`Bt2Tree`].
///
/// Records are held sorted by scaled offsets and are inserted in place, so the
/// encoded leaf is always ordered. A B-tree node is searched by bisection —
/// libhdf5 compares records with `H5VM_vector_cmp_u` (`H5Dbtree2.c`
/// `H5D__bt2_compare`), which orders the scaled-offset vectors
/// lexicographically — so an unordered leaf makes libhdf5 miss chunks that are
/// present, reading them back as fill. Insertion order is *not* a safe order:
/// `write_chunk_at` lets a caller address the grid in any sequence.
#[derive(Debug, Clone)]
pub struct Bt2ChunkIndex {
    /// Number of dataset dimensions.
    pub ndims: usize,
    /// Whether chunks are filtered.
    pub filtered: bool,
    /// Unfiltered chunk records (used when filtered == false), sorted by
    /// scaled offsets.
    pub records: Vec<Bt2ChunkRecord>,
    /// Filtered chunk records (used when filtered == true), sorted by scaled
    /// offsets.
    pub filtered_records: Vec<Bt2FilteredChunkRecord>,
    /// Width in bytes of a filtered record's compressed-size field. Meaningful
    /// only when `filtered`; see [`compute_chunk_size_len`].
    pub chunk_size_len: u8,
    /// Node size every (re-)serialization of this index uses: [`BT2_NODE_SIZE`]
    /// for a tree this writer creates, the on-disk header's value for a
    /// reopened tree. libhdf5 sizes every node from `hdr->node_size`
    /// (`H5B2leaf.c`, `H5B2internal.c`), never from a compile-time constant.
    pub node_size: u32,
    /// Split percentage carried into the header. Advisory for this index —
    /// the bulk loader rebuilds whole trees instead of splitting nodes — but
    /// a reopened tree must hand back the value its creator declared.
    pub split_percent: u8,
    /// Merge percentage carried into the header (advisory, as above).
    pub merge_percent: u8,
}

impl Bt2ChunkIndex {
    /// Create a new empty B-tree v2 chunk index for unfiltered chunks.
    pub fn new_unfiltered(ndims: usize) -> Self {
        Self {
            ndims,
            filtered: false,
            records: Vec::new(),
            filtered_records: Vec::new(),
            chunk_size_len: 0,
            node_size: BT2_NODE_SIZE,
            split_percent: BT2_SPLIT_PERCENT,
            merge_percent: BT2_MERGE_PERCENT,
        }
    }

    /// Create a new empty B-tree v2 chunk index for filtered chunks.
    ///
    /// `chunk_size_len` must be the width libhdf5 will recompute for this
    /// dataset — [`compute_chunk_size_len`] of the uncompressed chunk size.
    pub fn new_filtered(ndims: usize, chunk_size_len: u8) -> Self {
        Self {
            ndims,
            filtered: true,
            records: Vec::new(),
            filtered_records: Vec::new(),
            chunk_size_len,
            node_size: BT2_NODE_SIZE,
            split_percent: BT2_SPLIT_PERCENT,
            merge_percent: BT2_MERGE_PERCENT,
        }
    }

    /// Insert an unfiltered chunk record, keeping the records sorted.
    pub fn insert(&mut self, scaled_offsets: Vec<u64>, chunk_address: u64) {
        match self
            .records
            .binary_search_by(|r| r.scaled_offsets.as_slice().cmp(&scaled_offsets))
        {
            Ok(i) => self.records[i].chunk_address = chunk_address,
            Err(i) => self.records.insert(
                i,
                Bt2ChunkRecord {
                    scaled_offsets,
                    chunk_address,
                },
            ),
        }
    }

    /// Insert a filtered chunk record, keeping the records sorted.
    pub fn insert_filtered(
        &mut self,
        scaled_offsets: Vec<u64>,
        chunk_address: u64,
        chunk_size: u64,
        filter_mask: u32,
    ) {
        match self
            .filtered_records
            .binary_search_by(|r| r.scaled_offsets.as_slice().cmp(&scaled_offsets))
        {
            Ok(i) => {
                let rec = &mut self.filtered_records[i];
                rec.chunk_address = chunk_address;
                rec.chunk_size = chunk_size;
                rec.filter_mask = filter_mask;
            }
            Err(i) => self.filtered_records.insert(
                i,
                Bt2FilteredChunkRecord {
                    scaled_offsets,
                    chunk_address,
                    chunk_size,
                    filter_mask,
                },
            ),
        }
    }

    /// Look up a chunk by its scaled coordinates. Returns the record if found.
    pub fn lookup(&self, scaled_offsets: &[u64]) -> Option<&Bt2ChunkRecord> {
        self.records
            .binary_search_by(|r| r.scaled_offsets.as_slice().cmp(scaled_offsets))
            .ok()
            .map(|i| &self.records[i])
    }

    /// Look up a filtered chunk by its scaled coordinates.
    pub fn lookup_filtered(&self, scaled_offsets: &[u64]) -> Option<&Bt2FilteredChunkRecord> {
        self.filtered_records
            .binary_search_by(|r| r.scaled_offsets.as_slice().cmp(scaled_offsets))
            .ok()
            .map(|i| &self.filtered_records[i])
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
    ///
    /// A filtered record adds the compressed size — in the
    /// [`chunk_size_len`](Self::chunk_size_len)-byte field libhdf5 will
    /// recompute — and a 4-byte filter mask.
    pub fn record_size(&self, ctx: &FormatContext) -> u16 {
        let sa = ctx.sizeof_addr as usize;
        if self.filtered {
            (sa + self.chunk_size_len as usize + 4 + self.ndims * 8) as u16
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
                // libhdf5 H5D__bt2_filt_encode: address, chunk size,
                // filter mask, then the scaled offsets.
                buf.extend_from_slice(&rec.chunk_address.to_le_bytes()[..sa]);
                buf.extend_from_slice(
                    &rec.chunk_size.to_le_bytes()[..self.chunk_size_len as usize],
                );
                buf.extend_from_slice(&rec.filter_mask.to_le_bytes());
                for &offset in &rec.scaled_offsets {
                    buf.extend_from_slice(&offset.to_le_bytes());
                }
            }
        } else {
            for rec in &self.records {
                // libhdf5 layout: chunk address first, then scaled offsets.
                buf.extend_from_slice(&rec.chunk_address.to_le_bytes()[..sa]);
                for &offset in &rec.scaled_offsets {
                    buf.extend_from_slice(&offset.to_le_bytes());
                }
            }
        }

        buf
    }

    /// The record type these records serialize as.
    pub fn record_type(&self) -> u8 {
        if self.filtered {
            BT2_TYPE_CHUNK_FILT
        } else {
            BT2_TYPE_CHUNK_UNFILT
        }
    }

    /// Bulk-load these records into a v2 B-tree of
    /// [`node_size`](Self::node_size)-byte nodes.
    ///
    /// The records are already in key order (see the type docs), which is
    /// exactly what [`Bt2Tree::build`] needs.
    pub fn build_tree(&self, ctx: &FormatContext) -> Bt2Tree {
        let mut tree = Bt2Tree::build(
            self.record_type(),
            self.record_size(ctx),
            self.node_size,
            ctx.sizeof_addr,
            &self.encode_records(ctx),
        );
        tree.split_percent = self.split_percent;
        tree.merge_percent = self.merge_percent;
        tree
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
            // libhdf5 record layout: chunk address first, then scaled offsets.
            let chunk_address = read_addr(&record_data[pos..], sa);
            pos += sa;
            let mut scaled_offsets = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                let offset = u64::from_le_bytes([
                    record_data[pos],
                    record_data[pos + 1],
                    record_data[pos + 2],
                    record_data[pos + 3],
                    record_data[pos + 4],
                    record_data[pos + 5],
                    record_data[pos + 6],
                    record_data[pos + 7],
                ]);
                scaled_offsets.push(offset);
                pos += 8;
            }
            records.push(Bt2ChunkRecord {
                scaled_offsets,
                chunk_address,
            });
        }

        Ok(records)
    }

    /// Decode filtered records from a node's raw record data.
    ///
    /// `record_size` is the on-disk record width from the B-tree header; the
    /// compressed-size field width is derived from it (libhdf5 encodes the
    /// chunk size in `record_size - sizeof_addr - 4 - ndims*8` bytes).
    pub fn decode_filtered_records(
        record_data: &[u8],
        num_records: usize,
        ndims: usize,
        record_size: u16,
        ctx: &FormatContext,
    ) -> FormatResult<Vec<Bt2FilteredChunkRecord>> {
        let sa = ctx.sizeof_addr as usize;
        let rec_size = record_size as usize;
        // chunk-size field width = record - address - filter_mask - offsets.
        let chunk_size_len = rec_size.checked_sub(sa + 4 + ndims * 8).ok_or_else(|| {
            FormatError::InvalidData(format!(
                "filtered v2 B-tree record size {} too small",
                rec_size
            ))
        })?;
        if record_data.len() < num_records * rec_size {
            return Err(FormatError::BufferTooShort {
                needed: num_records * rec_size,
                available: record_data.len(),
            });
        }

        let mut records = Vec::with_capacity(num_records);
        let mut pos = 0;
        for _ in 0..num_records {
            // libhdf5 H5D__bt2_filt_encode: address, chunk size, filter
            // mask, then the scaled offsets.
            let chunk_address = read_addr(&record_data[pos..], sa);
            pos += sa;
            let chunk_size = read_size(&record_data[pos..], chunk_size_len);
            pos += chunk_size_len;
            let filter_mask = u32::from_le_bytes([
                record_data[pos],
                record_data[pos + 1],
                record_data[pos + 2],
                record_data[pos + 3],
            ]);
            pos += 4;
            let mut scaled_offsets = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                scaled_offsets.push(read_size(&record_data[pos..], 8));
                pos += 8;
            }
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

    /// Re-encode a header buffer's checksum after patching a field, so the
    /// decoder reaches field validation instead of failing on the checksum.
    fn rechecksum(buf: &mut [u8]) {
        let data_end = buf.len() - 4;
        let cksum = checksum_metadata(&buf[..data_end]);
        buf[data_end..].copy_from_slice(&cksum.to_le_bytes());
    }

    #[test]
    fn header_decode_rejects_malformed_geometry_fields() {
        // node_size at offset 6..10, record_size at 10..12, depth at 12..14.
        let make = || {
            let mut hdr = Bt2Header::new_for_chunks(&ctx8(), 2);
            hdr.root_node_addr = 0x3000;
            hdr.encode(&ctx8())
        };

        // node_size smaller than the metadata prefix.
        let mut buf = make();
        buf[6..10].copy_from_slice(&4u32.to_le_bytes());
        rechecksum(&mut buf);
        assert!(matches!(
            Bt2Header::decode(&buf, &ctx8()),
            Err(FormatError::InvalidData(_))
        ));

        // record_size zero.
        let mut buf = make();
        buf[10..12].copy_from_slice(&0u16.to_le_bytes());
        rechecksum(&mut buf);
        assert!(matches!(
            Bt2Header::decode(&buf, &ctx8()),
            Err(FormatError::InvalidData(_))
        ));

        // Implausibly large depth.
        let mut buf = make();
        buf[12..14].copy_from_slice(&5000u16.to_le_bytes());
        rechecksum(&mut buf);
        assert!(matches!(
            Bt2Header::decode(&buf, &ctx8()),
            Err(FormatError::InvalidData(_))
        ));

        // A well-formed header still decodes.
        assert!(Bt2Header::decode(&make(), &ctx8()).is_ok());
    }

    #[test]
    fn bt2_geometry_is_panic_free_on_degenerate_params() {
        // node_size below the prefix, zero record size, oversized depth.
        let _ = Bt2Geometry::new(4, 24, 0, 8);
        let _ = Bt2Geometry::new(0, 0, 0, 8);
        let _ = Bt2Geometry::new(4096, 24, 600, 8);
        let _ = Bt2Geometry::new(u32::MAX, 1, 64, 8);
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
        let hdr = Bt2Header::new_for_filtered_chunks(&ctx8(), 2, 8);
        assert_eq!(hdr.record_type, BT2_TYPE_CHUNK_FILT);
        // address(8) + chunk_size(8) + filter_mask(4) + 2 offsets(16) = 36,
        // the same rule Bt2ChunkIndex::record_size applies.
        assert_eq!(hdr.record_size, 36);
        assert_eq!(
            hdr.record_size,
            Bt2ChunkIndex::new_filtered(2, 8).record_size(&ctx8()),
            "header and index must agree on the record size"
        );
        // A narrower size field shrinks the record by exactly that much.
        assert_eq!(
            Bt2Header::new_for_filtered_chunks(&ctx8(), 2, 2).record_size,
            30
        );

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

        // depth 1: children are leaves, no total-nrec field.
        let encoded = node.encode(&ctx8(), 1, 1, 0);
        assert_eq!(&encoded[..4], b"BTIN");

        let decoded = Bt2InternalNode::decode(&encoded, &ctx8(), 1, 1, rec_size, 1, 0).unwrap();
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

        // depth 2: children carry a 2-byte total-nrec field.
        let encoded = node.encode(&ctx8(), 2, 1, 2);
        let decoded = Bt2InternalNode::decode(&encoded, &ctx8(), 2, 2, rec_size, 1, 2).unwrap();
        assert_eq!(decoded.child_total_nrecords, vec![100, 200, 50]);
        assert_eq!(decoded.child_nrecords, vec![4, 6, 2]);
    }

    #[test]
    fn bt2_geometry_matches_libhdf5() {
        // node_size 2048, record_size 24, depth 1: leaf holds (2048-10)/24 = 84.
        let g = Bt2Geometry::new(2048, 24, 1, 8);
        assert_eq!(g.node_info[0].max_nrec, 84);
        assert_eq!(g.max_nrec_size, 1);
        assert_eq!(g.child_total_size(1), 0);
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

    // libhdf5 bisects a B-tree node, so the encoded leaf must be ordered by
    // scaled offsets no matter what order the caller writes chunks in.
    #[test]
    fn records_are_ordered_however_they_are_inserted() {
        let mut idx = Bt2ChunkIndex::new_unfiltered(2);
        for coords in [[1, 1], [0, 2], [1, 0], [0, 0], [0, 1]] {
            idx.insert(coords.to_vec(), 0x1000);
        }
        let order: Vec<Vec<u64>> = idx.iter().map(|r| r.scaled_offsets.clone()).collect();
        assert_eq!(
            order,
            vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 0], vec![1, 1]]
        );
        // Every record is still reachable after the reordering.
        for coords in [[1, 1], [0, 2], [1, 0], [0, 0], [0, 1]] {
            assert!(idx.lookup(&coords).is_some(), "lost {coords:?}");
        }
    }

    #[test]
    fn filtered_records_are_ordered_however_they_are_inserted() {
        let mut idx = Bt2ChunkIndex::new_filtered(2, 8);
        for (i, coords) in [[2, 0], [0, 1], [1, 3], [0, 0]].iter().enumerate() {
            idx.insert_filtered(coords.to_vec(), 0x1000 + i as u64, 7, 0);
        }
        let order: Vec<Vec<u64>> = idx
            .iter_filtered()
            .map(|r| r.scaled_offsets.clone())
            .collect();
        assert_eq!(order, vec![vec![0, 0], vec![0, 1], vec![1, 3], vec![2, 0]]);
        assert_eq!(idx.lookup_filtered(&[1, 3]).unwrap().chunk_address, 0x1002);
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

        let (hdr, record_bytes) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.record_type, BT2_TYPE_CHUNK_UNFILT);
        assert_eq!(hdr.depth, 0);
        assert_eq!(hdr.total_num_records, 3);
        assert_eq!(hdr.num_records_in_root, 3);
        // record_size = 2*8 + 8 = 24
        assert_eq!(hdr.record_size, 24);

        let records = Bt2ChunkIndex::decode_unfiltered_records(&record_bytes, 3, 2, &ctx).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].scaled_offsets, vec![0, 0]);
        assert_eq!(records[0].chunk_address, 0x1000);
        assert_eq!(records[1].scaled_offsets, vec![0, 1]);
        assert_eq!(records[1].chunk_address, 0x2000);
        assert_eq!(records[2].scaled_offsets, vec![1, 0]);
        assert_eq!(records[2].chunk_address, 0x3000);
    }

    #[test]
    fn unfiltered_record_is_address_first() {
        // libhdf5 (H5D__bt2_unfilt_encode) writes the chunk address before
        // the scaled offsets. Lock that byte order in.
        let ctx = ctx8();
        let mut idx = Bt2ChunkIndex::new_unfiltered(2);
        idx.insert(vec![3, 7], 0xABCD);
        let bytes = idx.encode_records(&ctx);
        // First 8 bytes = address, then two 8-byte scaled offsets.
        assert_eq!(&bytes[0..8], &0xABCDu64.to_le_bytes());
        assert_eq!(&bytes[8..16], &3u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &7u64.to_le_bytes());
    }

    #[test]
    fn filtered_chunk_index_encode_decode_roundtrip() {
        let ctx = ctx8();
        let mut idx = Bt2ChunkIndex::new_filtered(2, 8);
        idx.insert_filtered(vec![0, 0], 0x1000, 512, 0);
        idx.insert_filtered(vec![1, 0], 0x2000, 300, 1);

        let (hdr, record_bytes) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.record_type, BT2_TYPE_CHUNK_FILT);
        assert_eq!(hdr.total_num_records, 2);
        // record_size = sizeof_addr(8) + chunk_size_len(8) + filter_mask(4)
        //             + ndims*8(16) = 36
        assert_eq!(hdr.record_size, 36);

        let records =
            Bt2ChunkIndex::decode_filtered_records(&record_bytes, 2, 2, hdr.record_size, &ctx)
                .unwrap();

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

        let (hdr, record_bytes) = serialize_and_walk(&idx, &ctx);
        // record_size = 1*8 + 4 = 12
        assert_eq!(hdr.record_size, 12);

        let records = Bt2ChunkIndex::decode_unfiltered_records(&record_bytes, 2, 1, &ctx).unwrap();
        assert_eq!(records[0].chunk_address, 0x100);
        assert_eq!(records[1].chunk_address, 0x200);
    }

    #[test]
    fn empty_chunk_index() {
        let ctx = ctx8();
        let idx = Bt2ChunkIndex::new_unfiltered(3);
        assert_eq!(idx.num_records(), 0);

        // No records means no nodes at all: the header names an undefined root,
        // the state libhdf5 leaves a freshly created B-tree in.
        let tree = idx.build_tree(&ctx);
        assert!(tree.nodes.is_empty());
        let (hdr, record_bytes) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.total_num_records, 0);
        assert_eq!(hdr.root_node_addr, UNDEF_ADDR);
        assert!(record_bytes.is_empty());
    }

    // ---- Bulk-loaded tree tests ----

    /// Serialize an index the way the writer's flush does — a distinct block
    /// per node — then walk the result the way a reader does, in key order:
    /// child 0, record 0, child 1, record 1, ..., child m. Returns the header
    /// and the records the walk recovered, so a caller can compare them
    /// against the flat ordered list the index holds.
    fn serialize_and_walk(idx: &Bt2ChunkIndex, ctx: &FormatContext) -> (Bt2Header, Vec<u8>) {
        let tree = idx.build_tree(ctx);
        let addrs: Vec<u64> = (0..tree.nodes.len())
            .map(|i| 0x1000 + i as u64 * tree.node_size as u64)
            .collect();
        let blocks: Vec<(u64, Vec<u8>)> = addrs
            .iter()
            .copied()
            .zip(tree.encode(ctx, &addrs))
            .collect();
        for (_, image) in &blocks {
            assert_eq!(
                image.len(),
                tree.node_size as usize,
                "every node occupies a full block"
            );
        }
        let hdr = tree.header(addrs.last().copied().unwrap_or(UNDEF_ADDR));

        let mut out = Vec::new();
        if hdr.root_node_addr != UNDEF_ADDR {
            let geo = Bt2Geometry::new(hdr.node_size, hdr.record_size, hdr.depth, ctx.sizeof_addr);
            walk_node(
                &blocks,
                hdr.root_node_addr,
                hdr.depth,
                hdr.num_records_in_root,
                &hdr,
                &geo,
                ctx,
                &mut out,
            );
        }
        (hdr, out)
    }

    /// A serialized tree laid out contiguously, so the production record walk
    /// can be pointed at it the way it is pointed at a file.
    struct NodePool {
        base: u64,
        image: Vec<u8>,
    }

    impl BlockReader for NodePool {
        fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>> {
            let start = (offset - self.base) as usize;
            let end = (start + len).min(self.image.len());
            Ok(self.image[start..end].to_vec())
        }
    }

    /// Serialize `idx` into one contiguous pool and return it with its header.
    fn serialize_into_pool(idx: &Bt2ChunkIndex, ctx: &FormatContext) -> (Bt2Header, NodePool) {
        let tree = idx.build_tree(ctx);
        let base = 0x1000u64;
        let addrs: Vec<u64> = (0..tree.nodes.len() as u64)
            .map(|i| base + i * tree.node_size as u64)
            .collect();
        let mut image = vec![0u8; tree.nodes.len() * tree.node_size as usize];
        for (i, node) in tree.encode(ctx, &addrs).into_iter().enumerate() {
            let at = i * tree.node_size as usize;
            image[at..at + node.len()].copy_from_slice(&node);
        }
        let hdr = tree.header(addrs.last().copied().unwrap_or(UNDEF_ADDR));
        (hdr, NodePool { base, image })
    }

    /// `collect_btree_v2_records` is what the dense-attribute reader and
    /// `open_append`'s index rebuild both walk with. An internal node's
    /// records separate its children, so emitting them ahead of the subtrees
    /// yields a sequence sorted only within each node — enough to fool any
    /// caller that re-keys the records, and wrong for one that does not.
    #[test]
    fn the_record_walk_returns_key_order_across_levels() {
        let ctx = ctx8();
        for n in [84u64, 85, 5270] {
            let idx = index_with(n);
            let (hdr, mut pool) = serialize_into_pool(&idx, &ctx);
            let walked = collect_btree_v2_records(&hdr, &ctx, &mut pool).unwrap();
            assert_eq!(
                walked,
                idx.encode_records(&ctx),
                "{n} records at depth {} came back out of key order",
                hdr.depth
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_node(
        blocks: &[(u64, Vec<u8>)],
        addr: u64,
        depth: u16,
        nrec: u16,
        hdr: &Bt2Header,
        geo: &Bt2Geometry,
        ctx: &FormatContext,
        out: &mut Vec<u8>,
    ) {
        let buf = &blocks
            .iter()
            .find(|(a, _)| *a == addr)
            .unwrap_or_else(|| panic!("no node at {addr:#x}"))
            .1;
        let rec = hdr.record_size as usize;
        assert!(
            10 + nrec as usize * rec <= hdr.node_size as usize,
            "node at depth {depth} holds {nrec} records, more than its block fits"
        );
        if depth == 0 {
            let leaf = Bt2LeafNode::decode(buf, nrec, hdr.record_size).unwrap();
            out.extend_from_slice(&leaf.record_data);
            return;
        }
        let node = Bt2InternalNode::decode(
            buf,
            ctx,
            depth,
            nrec,
            hdr.record_size,
            geo.max_nrec_size,
            geo.child_total_size(depth),
        )
        .unwrap();
        for i in 0..=nrec as usize {
            walk_node(
                blocks,
                node.child_addrs[i],
                depth - 1,
                node.child_nrecords[i],
                hdr,
                geo,
                ctx,
                out,
            );
            if i < nrec as usize {
                out.extend_from_slice(&node.record_data[i * rec..(i + 1) * rec]);
            }
        }
    }

    /// Build an unfiltered 2-D index with `n` records at (i, 0).
    fn index_with(n: u64) -> Bt2ChunkIndex {
        let mut idx = Bt2ChunkIndex::new_unfiltered(2);
        for i in 0..n {
            idx.insert(vec![i, 0], 0x10_000 + i * 0x100);
        }
        idx
    }

    #[test]
    fn a_tree_that_fits_one_node_stays_a_single_leaf() {
        let ctx = ctx8();
        // record_size 24, node 2048: a leaf holds (2048 - 10) / 24 = 84.
        let idx = index_with(84);
        let tree = idx.build_tree(&ctx);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.total_records(), 84);

        let (hdr, walked) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.depth, 0);
        assert_eq!(walked, idx.encode_records(&ctx));
    }

    #[test]
    fn one_record_past_a_leaf_grows_the_tree_a_level() {
        let ctx = ctx8();
        let idx = index_with(85);
        let tree = idx.build_tree(&ctx);
        assert_eq!(tree.depth(), 1, "85 records no longer fit one leaf");
        assert_eq!(tree.total_records(), 85);
        // Root separator plus two leaves.
        assert_eq!(tree.nodes.len(), 3);

        let (hdr, walked) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.depth, 1);
        assert_eq!(hdr.total_num_records, 85);
        assert_eq!(
            walked,
            idx.encode_records(&ctx),
            "an in-order walk must recover every record in key order"
        );
    }

    #[test]
    fn a_tree_grows_a_second_level() {
        let ctx = ctx8();
        // Depth 1 tops out at 61 root records over 62 leaves of 84:
        // 61 + 62 * 84 = 5269.
        let idx = index_with(5270);
        let tree = idx.build_tree(&ctx);
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.total_records(), 5270);

        let (hdr, walked) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.depth, 2);
        assert_eq!(hdr.total_num_records, 5270);
        assert_eq!(walked, idx.encode_records(&ctx));
    }

    /// The bulk load sizes nodes from the index's `node_size`, not the
    /// compile-time default — a reopened foreign tree re-serializes at the
    /// size its header declares, the way libhdf5 allocates every node at
    /// `hdr->node_size`. Split/merge pass through to the header the same way.
    #[test]
    fn bulk_load_honors_a_foreign_node_size() {
        let ctx = ctx8();
        // record_size 24, node 512: a leaf holds (512 - 10) / 24 = 20, so
        // 200 records force at least one internal level.
        let mut idx = index_with(200);
        idx.node_size = 512;
        idx.split_percent = 90;
        idx.merge_percent = 30;

        let tree = idx.build_tree(&ctx);
        assert_eq!(tree.node_size, 512);
        assert!(
            tree.depth() >= 1,
            "200 records must not fit one 512-byte leaf"
        );
        let addrs: Vec<u64> = (0..tree.nodes.len() as u64)
            .map(|i| 0x1000 + i * 512)
            .collect();
        for image in tree.encode(&ctx, &addrs) {
            assert_eq!(image.len(), 512, "every node image fills its block");
        }

        let (hdr, walked) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.node_size, 512);
        assert_eq!(hdr.split_percent, 90);
        assert_eq!(hdr.merge_percent, 30);
        assert_eq!(walked, idx.encode_records(&ctx));
    }

    /// The record count a node reports is a u16; a tree that kept every record
    /// in one node would silently truncate past 65535. Splitting keeps every
    /// node small, so the only count that has to be wide is the header's
    /// `total_num_records`.
    #[test]
    fn a_tree_far_past_the_node_record_count_limit_stays_intact() {
        let ctx = ctx8();
        let idx = index_with(70_000);
        let tree = idx.build_tree(&ctx);
        assert_eq!(tree.total_records(), 70_000);
        assert!(tree.depth() >= 2);
        for node in &tree.nodes {
            let cap = tree.geometry.node_info[node.depth as usize].max_nrec;
            assert!(
                u64::from(node.num_records) <= cap && node.num_records > 0,
                "node at depth {} holds {} records (cap {cap})",
                node.depth,
                node.num_records
            );
            assert_eq!(
                node.children.len(),
                if node.depth == 0 {
                    0
                } else {
                    node.num_records as usize + 1
                }
            );
        }
        let (hdr, walked) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.total_num_records, 70_000);
        assert_eq!(walked, idx.encode_records(&ctx));
    }

    #[test]
    fn filtered_records_survive_a_split_tree() {
        let ctx = ctx8();
        // record_size = 8 + 2 + 4 + 16 = 30, so a leaf holds (2048-10)/30 = 67.
        let mut idx = Bt2ChunkIndex::new_filtered(2, 2);
        for i in 0..500u64 {
            idx.insert_filtered(vec![i, 0], 0x10_000 + i * 0x100, (i % 400) + 1, 0);
        }
        let tree = idx.build_tree(&ctx);
        assert!(tree.depth() >= 1);

        let (hdr, walked) = serialize_and_walk(&idx, &ctx);
        assert_eq!(hdr.record_size, 30);
        assert_eq!(hdr.total_num_records, 500);
        let records =
            Bt2ChunkIndex::decode_filtered_records(&walked, 500, 2, hdr.record_size, &ctx).unwrap();
        for (i, r) in records.iter().enumerate() {
            assert_eq!(r.scaled_offsets, vec![i as u64, 0]);
            assert_eq!(r.chunk_address, 0x10_000 + i as u64 * 0x100);
            assert_eq!(r.chunk_size, (i as u64 % 400) + 1);
        }
    }

    /// Growing the index must not change the addresses already handed out —
    /// that is what lets the writer overwrite node blocks instead of
    /// relocating (and orphaning) them.
    #[test]
    fn every_node_occupies_the_same_size_block_at_any_depth() {
        let ctx = ctx8();
        for n in [1u64, 84, 85, 1000, 5270] {
            let tree = index_with(n).build_tree(&ctx);
            let addrs: Vec<u64> = (0..tree.nodes.len() as u64).collect();
            for image in tree.encode(&ctx, &addrs) {
                assert_eq!(image.len(), BT2_NODE_SIZE as usize, "n = {n}");
            }
        }
    }
}
