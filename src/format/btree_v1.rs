//! B-tree v1 decode (for reading legacy HDF5 files).
//!
//! The B-tree v1 is used in v0/v1 groups to index symbol table entries.
//! For group B-trees (type 0), each key is a name offset into the local
//! heap, and each child pointer is an address of either a SNOD (leaf
//! level) or another TREE node (internal level).
//!
//! Layout:
//! ```text
//! "TREE" (4 bytes)
//! type: 1 byte (0 = group)
//! level: 1 byte (0 = leaf)
//! entries_used: u16 LE
//! left_sibling: sizeof_addr bytes LE
//! right_sibling: sizeof_addr bytes LE
//! Then interleaved keys and children:
//!   key[0], child[0], key[1], child[1], ..., key[entries_used]
//! ```
//!
//! For type-0 (group) B-trees:
//! - Each key is sizeof_size bytes (name offset into local heap)
//! - Each child is sizeof_addr bytes (address of SNOD or sub-TREE)
//!
//! For type-1 (raw data chunk) B-trees:
//! - Each key is `4 + 4 + (rank+1)*8` bytes: chunk_size(4), filter_mask(4),
//!   then (rank+1) 8-byte element offsets. The last offset is the
//!   element-size dimension and is always 0.
//! - Each child is sizeof_addr bytes: a chunk-data address at a leaf
//!   node (level 0), or a sub-TREE address at an internal node.

use crate::format::bytes::{read_le_addr as read_addr, read_le_uint as read_uint};
use crate::format::{FormatError, FormatResult};

/// The 4-byte B-tree v1 signature.
pub const BTREE_V1_SIGNATURE: [u8; 4] = *b"TREE";

/// The v1 B-tree split ranks ("K" values) in force for one file.
///
/// A v1 B-tree node is a *fixed-size* on-disk record whose length is derived
/// entirely from these ranks (`H5B.c:1676` `sizeof_rnode`), so a reader cannot
/// know how many bytes a node occupies — nor reject a node claiming more
/// entries than can fit — without them. They come from the v0/v1 superblock,
/// from the superblock extension's v1-B-tree-"K" message (0x0013) when one is
/// present, and otherwise from the library defaults below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BTreeV1Config {
    /// Symbol-table leaf (SNOD) 1/2 rank: a node holds up to `2 * k` entries.
    /// `H5F_CRT_SYM_LEAF_DEF`.
    pub sym_leaf_k: u16,
    /// Internal-node 1/2 rank for symbol-table (type 0) B-trees.
    /// `HDF5_BTREE_SNODE_IK_DEF`.
    pub snode_internal_k: u16,
    /// Internal-node 1/2 rank for chunked-storage (type 1) B-trees.
    /// `HDF5_BTREE_CHUNK_IK_DEF`.
    pub chunk_internal_k: u16,
}

impl Default for BTreeV1Config {
    fn default() -> Self {
        Self {
            sym_leaf_k: 4,
            snode_internal_k: 16,
            chunk_internal_k: 32,
        }
    }
}

/// A symbol-table entry's on-disk size: name offset, object header address,
/// cache type, reserved word and the 16-byte scratch pad.
fn symbol_table_entry_size(sizeof_addr: usize, sizeof_size: usize) -> usize {
    sizeof_size + sizeof_addr + 4 + 4 + 16
}

impl BTreeV1Config {
    /// Maximum entries a symbol table node (SNOD) may declare.
    pub fn sym_leaf_max_entries(&self) -> u16 {
        self.sym_leaf_k.saturating_mul(2)
    }

    /// Maximum entries a symbol-table (type 0) B-tree node may declare.
    pub fn snode_max_entries(&self) -> u16 {
        self.snode_internal_k.saturating_mul(2)
    }

    /// Maximum entries a chunked-storage (type 1) B-tree node may declare.
    pub fn chunk_max_entries(&self) -> u16 {
        self.chunk_internal_k.saturating_mul(2)
    }

    /// On-disk size of a symbol-table (type 0) B-tree node.
    pub fn snode_btree_node_size(&self, sizeof_addr: usize, sizeof_size: usize) -> usize {
        btree_node_size(self.snode_max_entries(), sizeof_addr, sizeof_size)
    }

    /// On-disk size of a chunked-storage (type 1) B-tree node for a dataset of
    /// the given `rank` (excluding the trailing element-size dimension).
    pub fn chunk_btree_node_size(&self, sizeof_addr: usize, rank: usize) -> usize {
        btree_node_size(
            self.chunk_max_entries(),
            sizeof_addr,
            4 + 4 + (rank + 1) * 8,
        )
    }

    /// On-disk size of a symbol table node (SNOD): its 8-byte prefix plus
    /// `2 * sym_leaf_k` entries (`H5Gpkg.h` `H5G_NODE_SIZE`).
    pub fn symbol_table_node_size(&self, sizeof_addr: usize, sizeof_size: usize) -> usize {
        8 + (self.sym_leaf_k as usize) * 2 * symbol_table_entry_size(sizeof_addr, sizeof_size)
    }
}

/// `H5B.c:1676`: header + `2K` child pointers + `2K + 1` keys.
fn btree_node_size(two_k: u16, sizeof_addr: usize, key_size: usize) -> usize {
    let two_k = two_k as usize;
    8 + 2 * sizeof_addr + two_k * sizeof_addr + (two_k + 1) * key_size
}

/// A decoded B-tree v1 node.
#[derive(Debug, Clone)]
pub struct BTreeV1Node {
    /// Node type: 0 = group, 1 = raw data chunk.
    pub node_type: u8,
    /// Node level: 0 = leaf (children are SNODs), >0 = internal (children are sub-TREE).
    pub level: u8,
    /// Number of entries used in this node.
    pub entries_used: u16,
    /// Address of left sibling, or UNDEF_ADDR if none.
    pub left_sibling: u64,
    /// Address of right sibling, or UNDEF_ADDR if none.
    pub right_sibling: u64,
    /// Keys (entries_used + 1 entries for type-0 group trees).
    pub keys: Vec<u64>,
    /// Child addresses (entries_used entries).
    pub children: Vec<u64>,
}

impl BTreeV1Node {
    /// Decode a B-tree v1 node from `buf`.
    ///
    /// `sizeof_addr` and `sizeof_size` come from the superblock; `max_entries`
    /// is `2 * K` for symbol-table B-trees ([`BTreeV1Config::snode_max_entries`]),
    /// which upstream guarantees no node exceeds.
    pub fn decode(
        buf: &[u8],
        sizeof_addr: usize,
        sizeof_size: usize,
        max_entries: u16,
    ) -> FormatResult<Self> {
        let header_size = 4 + 1 + 1 + 2 + sizeof_addr * 2;
        if buf.len() < header_size {
            return Err(FormatError::BufferTooShort {
                needed: header_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != BTREE_V1_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let node_type = buf[4];
        let level = buf[5];
        let entries_used = u16::from_le_bytes([buf[6], buf[7]]);
        if entries_used > max_entries {
            return Err(FormatError::InvalidData(format!(
                "B-tree v1 node declares {entries_used} entries, more than the \
                 {max_entries} its 'K' value allows"
            )));
        }

        let mut pos = 8;
        let left_sibling = read_addr(&buf[pos..], sizeof_addr);
        pos += sizeof_addr;
        let right_sibling = read_addr(&buf[pos..], sizeof_addr);
        pos += sizeof_addr;

        // For group B-trees (type 0):
        // Interleaved: key[0], child[0], key[1], child[1], ..., key[n]
        // That's (entries_used + 1) keys and entries_used children.
        let n = entries_used as usize;

        if node_type == 0 {
            // Group B-tree
            let key_size = sizeof_size;
            let child_size = sizeof_addr;
            // Total data: (n+1) keys interleaved with n children
            let data_size = (n + 1) * key_size + n * child_size;
            let needed = pos + data_size;
            if buf.len() < needed {
                return Err(FormatError::BufferTooShort {
                    needed,
                    available: buf.len(),
                });
            }

            let mut keys = Vec::with_capacity(n + 1);
            let mut children = Vec::with_capacity(n);

            for _i in 0..n {
                // key[i]
                keys.push(read_uint(&buf[pos..], key_size));
                pos += key_size;
                // child[i]
                children.push(read_uint(&buf[pos..], child_size));
                pos += child_size;
            }
            // final key[n]
            keys.push(read_uint(&buf[pos..], key_size));

            Ok(BTreeV1Node {
                node_type,
                level,
                entries_used,
                left_sibling,
                right_sibling,
                keys,
                children,
            })
        } else {
            // Raw data chunk B-tree (type 1) is decoded via
            // `ChunkBTreeV1Node::decode`, which understands the chunk-key
            // structure. `BTreeV1Node` only models type-0 group trees.
            Err(FormatError::UnsupportedFeature(format!(
                "B-tree v1 type {} not supported by BTreeV1Node (use ChunkBTreeV1Node)",
                node_type
            )))
        }
    }
}

/// A decoded chunk key from a raw-data-chunk (type-1) B-tree v1 node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkKey {
    /// Size in bytes of the stored chunk (compressed size when filtered).
    pub chunk_size: u32,
    /// Filter mask: bit `i` set means filter `i` was skipped for this chunk.
    pub filter_mask: u32,
    /// Per-dimension element offsets of the chunk's first element. The
    /// trailing entry is the element-size dimension and is always 0, so
    /// this has `rank + 1` entries.
    pub offsets: Vec<u64>,
}

/// A decoded raw-data-chunk (type-1) B-tree v1 node.
#[derive(Debug, Clone)]
pub struct ChunkBTreeV1Node {
    /// Node level: 0 = leaf (children point at chunk data), >0 = internal
    /// (children point at sub-TREE nodes).
    pub level: u8,
    /// Number of entries (children) used in this node.
    pub entries_used: u16,
    /// Keys, `entries_used + 1` of them. `keys[i]` describes `children[i]`;
    /// the final key is the right-boundary key.
    pub keys: Vec<ChunkKey>,
    /// Child addresses, `entries_used` of them.
    pub children: Vec<u64>,
}

impl ChunkBTreeV1Node {
    /// Decode a type-1 (raw data chunk) B-tree v1 node from `buf`.
    ///
    /// `rank` is the chunk rank *excluding* the trailing element-size
    /// dimension, so each key carries `rank + 1` 8-byte offsets — matching
    /// libhdf5's `H5O_layout_chunk_t::ndims` (which includes the element
    /// dimension). `max_entries` is `2 * K` for chunk B-trees
    /// ([`BTreeV1Config::chunk_max_entries`]).
    pub fn decode(
        buf: &[u8],
        sizeof_addr: usize,
        rank: usize,
        max_entries: u16,
    ) -> FormatResult<Self> {
        let header_size = 4 + 1 + 1 + 2 + sizeof_addr * 2;
        if buf.len() < header_size {
            return Err(FormatError::BufferTooShort {
                needed: header_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != BTREE_V1_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let node_type = buf[4];
        if node_type != 1 {
            return Err(FormatError::UnsupportedFeature(format!(
                "expected B-tree v1 chunk node (type 1), found type {node_type}"
            )));
        }
        let level = buf[5];
        let entries_used = u16::from_le_bytes([buf[6], buf[7]]);
        if entries_used > max_entries {
            return Err(FormatError::InvalidData(format!(
                "chunk B-tree v1 node declares {entries_used} entries, more than \
                 the {max_entries} its 'K' value allows"
            )));
        }

        // Skip the signature/type/level/entries header plus the two
        // sibling pointers.
        let mut pos = 8 + sizeof_addr * 2;

        let n = entries_used as usize;
        // Each chunk key: chunk_size(4) + filter_mask(4) + (rank+1)*8.
        let key_size = 4 + 4 + (rank + 1) * 8;
        // Interleaved: key[0] child[0] ... key[n-1] child[n-1] key[n].
        let data_size = (n + 1) * key_size + n * sizeof_addr;
        let needed = pos + data_size;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }

        let decode_key = |slice: &[u8]| -> ChunkKey {
            let chunk_size = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
            let filter_mask = u32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]);
            let mut offsets = Vec::with_capacity(rank + 1);
            let mut o = 8;
            for _ in 0..(rank + 1) {
                offsets.push(read_uint(&slice[o..], 8));
                o += 8;
            }
            ChunkKey {
                chunk_size,
                filter_mask,
                offsets,
            }
        };

        let mut keys = Vec::with_capacity(n + 1);
        let mut children = Vec::with_capacity(n);
        for _ in 0..n {
            keys.push(decode_key(&buf[pos..pos + key_size]));
            pos += key_size;
            children.push(read_addr(&buf[pos..], sizeof_addr));
            pos += sizeof_addr;
        }
        // Final right-boundary key.
        keys.push(decode_key(&buf[pos..pos + key_size]));

        Ok(ChunkBTreeV1Node {
            level,
            entries_used,
            keys,
            children,
        })
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::UNDEF_ADDR;

    /// Build a group B-tree v1 node for testing.
    fn build_group_btree(
        level: u8,
        keys: &[u64],
        children: &[u64],
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Vec<u8> {
        assert_eq!(keys.len(), children.len() + 1);
        let entries_used = children.len() as u16;

        let mut buf = Vec::new();
        buf.extend_from_slice(&BTREE_V1_SIGNATURE);
        buf.push(0); // type = group
        buf.push(level);
        buf.extend_from_slice(&entries_used.to_le_bytes());
        // left sibling = UNDEF
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);
        // right sibling = UNDEF
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);

        // Interleaved keys and children
        for i in 0..children.len() {
            buf.extend_from_slice(&keys[i].to_le_bytes()[..sizeof_size]);
            buf.extend_from_slice(&children[i].to_le_bytes()[..sizeof_addr]);
        }
        // Final key
        buf.extend_from_slice(&keys[children.len()].to_le_bytes()[..sizeof_size]);

        buf
    }

    #[test]
    fn decode_leaf_node() {
        let buf = build_group_btree(
            0,               // leaf
            &[0, 8, 16],     // 3 keys
            &[0x100, 0x200], // 2 children (SNOD addresses)
            8,
            8,
        );
        let node = BTreeV1Node::decode(&buf, 8, 8, 32).unwrap();
        assert_eq!(node.node_type, 0);
        assert_eq!(node.level, 0);
        assert_eq!(node.entries_used, 2);
        assert_eq!(node.keys, vec![0, 8, 16]);
        assert_eq!(node.children, vec![0x100, 0x200]);
        assert_eq!(node.left_sibling, UNDEF_ADDR);
        assert_eq!(node.right_sibling, UNDEF_ADDR);
    }

    #[test]
    fn decode_internal_node() {
        let buf = build_group_btree(
            1,         // internal
            &[0, 100], // 2 keys
            &[0x500],  // 1 child (sub-TREE address)
            8,
            8,
        );
        let node = BTreeV1Node::decode(&buf, 8, 8, 32).unwrap();
        assert_eq!(node.level, 1);
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![0x500]);
    }

    #[test]
    fn decode_single_entry() {
        let buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        let node = BTreeV1Node::decode(&buf, 8, 8, 32).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn decode_4byte() {
        let buf = build_group_btree(0, &[0, 4], &[0x80], 4, 4);
        let node = BTreeV1Node::decode(&buf, 4, 4, 32).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![0x80]);
    }

    #[test]
    fn decode_bad_sig() {
        let mut buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        buf[0] = b'X';
        assert!(matches!(
            BTreeV1Node::decode(&buf, 8, 8, 32).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn decode_too_short() {
        assert!(matches!(
            BTreeV1Node::decode(&[0u8; 4], 8, 8, 32).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    #[test]
    fn decode_unsupported_type() {
        let mut buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        buf[4] = 1; // type = raw data chunks
        assert!(matches!(
            BTreeV1Node::decode(&buf, 8, 8, 32).unwrap_err(),
            FormatError::UnsupportedFeature(_)
        ));
    }

    /// Build a type-1 (raw data chunk) B-tree v1 node for testing.
    /// `rank` excludes the trailing element-size dimension.
    fn build_chunk_btree(
        level: u8,
        keys: &[ChunkKey],
        children: &[u64],
        sizeof_addr: usize,
    ) -> Vec<u8> {
        assert_eq!(keys.len(), children.len() + 1);
        let entries_used = children.len() as u16;

        let mut buf = Vec::new();
        buf.extend_from_slice(&BTREE_V1_SIGNATURE);
        buf.push(1); // type = raw data chunk
        buf.push(level);
        buf.extend_from_slice(&entries_used.to_le_bytes());
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]); // left
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]); // right

        let encode_key = |buf: &mut Vec<u8>, k: &ChunkKey| {
            buf.extend_from_slice(&k.chunk_size.to_le_bytes());
            buf.extend_from_slice(&k.filter_mask.to_le_bytes());
            for &o in &k.offsets {
                buf.extend_from_slice(&o.to_le_bytes());
            }
        };

        for i in 0..children.len() {
            encode_key(&mut buf, &keys[i]);
            buf.extend_from_slice(&children[i].to_le_bytes()[..sizeof_addr]);
        }
        encode_key(&mut buf, &keys[children.len()]);
        buf
    }

    fn chunk_key(size: u32, mask: u32, offsets: &[u64]) -> ChunkKey {
        ChunkKey {
            chunk_size: size,
            filter_mask: mask,
            offsets: offsets.to_vec(),
        }
    }

    #[test]
    fn decode_chunk_leaf_1d() {
        // 1-D dataset (rank 1): each key has rank+1 = 2 offsets.
        let keys = [
            chunk_key(32, 0, &[0, 0]),
            chunk_key(32, 0, &[8, 0]),
            chunk_key(0, 0, &[16, 0]),
        ];
        let buf = build_chunk_btree(0, &keys, &[0x400, 0x800], 8);
        let node = ChunkBTreeV1Node::decode(&buf, 8, 1, 64).unwrap();
        assert_eq!(node.level, 0);
        assert_eq!(node.entries_used, 2);
        assert_eq!(node.children, vec![0x400, 0x800]);
        assert_eq!(node.keys.len(), 3);
        assert_eq!(node.keys[0].chunk_size, 32);
        assert_eq!(node.keys[1].offsets, vec![8, 0]);
    }

    #[test]
    fn decode_chunk_internal_2d() {
        // 2-D dataset (rank 2): each key has rank+1 = 3 offsets.
        let keys = [chunk_key(64, 0, &[0, 0, 0]), chunk_key(64, 0, &[4, 4, 0])];
        let buf = build_chunk_btree(1, &keys, &[0x1000], 8);
        let node = ChunkBTreeV1Node::decode(&buf, 8, 2, 64).unwrap();
        assert_eq!(node.level, 1);
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![0x1000]);
        assert_eq!(node.keys[0].offsets, vec![0, 0, 0]);
    }

    #[test]
    fn decode_chunk_filtered_key() {
        let keys = [chunk_key(17, 0x1, &[0, 0]), chunk_key(0, 0, &[8, 0])];
        let buf = build_chunk_btree(0, &keys, &[0x200], 8);
        let node = ChunkBTreeV1Node::decode(&buf, 8, 1, 64).unwrap();
        assert_eq!(node.keys[0].chunk_size, 17);
        assert_eq!(node.keys[0].filter_mask, 0x1);
    }

    #[test]
    fn decode_chunk_rejects_group_node() {
        let buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        assert!(matches!(
            ChunkBTreeV1Node::decode(&buf, 8, 1, 64).unwrap_err(),
            FormatError::UnsupportedFeature(_)
        ));
    }

    #[test]
    fn decode_chunk_too_short() {
        assert!(matches!(
            ChunkBTreeV1Node::decode(&[0u8; 4], 8, 1, 64).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    #[test]
    fn decode_chunk_bad_sig() {
        let keys = [chunk_key(8, 0, &[0, 0]), chunk_key(0, 0, &[8, 0])];
        let mut buf = build_chunk_btree(0, &keys, &[0x100], 8);
        buf[0] = b'X';
        assert!(matches!(
            ChunkBTreeV1Node::decode(&buf, 8, 1, 64).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn node_sizes_match_upstream_formula() {
        let cfg = BTreeV1Config::default();
        // H5B.c: 8 + 2*addr + 2K*addr + (2K+1)*key.
        assert_eq!(cfg.snode_btree_node_size(8, 8), 8 + 16 + 32 * 8 + 33 * 8);
        // rank 1 => key is 4 + 4 + 2*8 = 24 bytes, 2K = 64.
        assert_eq!(cfg.chunk_btree_node_size(8, 1), 8 + 16 + 64 * 8 + 65 * 24);
        // SNOD: 8-byte prefix + 2*sym_leaf_k entries of 40 bytes.
        assert_eq!(cfg.symbol_table_node_size(8, 8), 8 + 8 * 40);
    }

    #[test]
    fn non_default_k_scales_every_node_size() {
        let cfg = BTreeV1Config {
            sym_leaf_k: 128,
            snode_internal_k: 512,
            chunk_internal_k: 256,
        };
        assert_eq!(cfg.snode_max_entries(), 1024);
        assert_eq!(cfg.chunk_max_entries(), 512);
        // Every one of these exceeds the fixed 8 KiB window the reader used
        // before the 'K' values were wired in.
        assert!(cfg.snode_btree_node_size(8, 8) > 8192);
        assert!(cfg.chunk_btree_node_size(8, 1) > 8192);
        assert!(cfg.symbol_table_node_size(8, 8) > 8192);
    }

    #[test]
    fn decode_rejects_entries_beyond_two_k() {
        let buf = build_group_btree(0, &[0, 8, 16], &[0x100, 0x200], 8, 8);
        // The node really holds 2 entries; a K of 0 makes even that illegal.
        assert!(matches!(
            BTreeV1Node::decode(&buf, 8, 8, 0).unwrap_err(),
            FormatError::InvalidData(_)
        ));
        // Exactly 2K entries is legal.
        assert!(BTreeV1Node::decode(&buf, 8, 8, 2).is_ok());
    }

    #[test]
    fn decode_chunk_rejects_entries_beyond_two_k() {
        let keys = [
            chunk_key(32, 0, &[0, 0]),
            chunk_key(32, 0, &[8, 0]),
            chunk_key(0, 0, &[16, 0]),
        ];
        let buf = build_chunk_btree(0, &keys, &[0x400, 0x800], 8);
        assert!(matches!(
            ChunkBTreeV1Node::decode(&buf, 8, 1, 1).unwrap_err(),
            FormatError::InvalidData(_)
        ));
        assert!(ChunkBTreeV1Node::decode(&buf, 8, 1, 2).is_ok());
    }

    #[test]
    fn decode_chunk_4byte_addr() {
        let keys = [chunk_key(16, 0, &[0, 0]), chunk_key(0, 0, &[4, 0])];
        let buf = build_chunk_btree(0, &keys, &[0x80], 4);
        let node = ChunkBTreeV1Node::decode(&buf, 4, 1, 64).unwrap();
        assert_eq!(node.children, vec![0x80]);
    }
}
