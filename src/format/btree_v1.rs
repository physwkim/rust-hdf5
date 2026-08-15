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

use crate::format::superblock::symbol_table_entry_size;

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
    /// Encode this type-0 (symbol-table) node into exactly `node_size` bytes.
    ///
    /// Like a SNOD, a v1 B-tree node is a fixed-size record derived from the
    /// file's "K" values ([`BTreeV1Config::snode_btree_node_size`]) however
    /// few entries it uses, and the slots past `entries_used` are zeroed.
    /// `keys` must hold exactly one more entry than `children`: a v1 B-tree
    /// stores both the left and the right bound of every child, so a node with
    /// n children has n+1 keys.
    pub fn encode(
        &self,
        node_size: usize,
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> FormatResult<Vec<u8>> {
        if self.node_type != 0 {
            return Err(FormatError::UnsupportedFeature(format!(
                "B-tree v1 type {} is not encoded by BTreeV1Node (only type 0, \
                 symbol-table nodes)",
                self.node_type
            )));
        }
        if self.keys.len() != self.children.len() + 1 {
            return Err(FormatError::InvalidData(format!(
                "B-tree v1 node has {} keys for {} children; a v1 node stores both \
                 bounds of every child, so it needs exactly one more key than children",
                self.keys.len(),
                self.children.len()
            )));
        }
        if self.entries_used as usize != self.children.len() {
            return Err(FormatError::InvalidData(format!(
                "B-tree v1 node declares {} entries but carries {} children",
                self.entries_used,
                self.children.len()
            )));
        }
        let needed =
            8 + 2 * sizeof_addr + self.children.len() * (sizeof_size + sizeof_addr) + sizeof_size;
        if needed > node_size {
            return Err(FormatError::InvalidData(format!(
                "B-tree v1 node needs {needed} bytes for {} children, more than the \
                 {node_size}-byte record the file's 'K' value allows",
                self.children.len()
            )));
        }

        let mut buf = Vec::with_capacity(node_size);
        buf.extend_from_slice(&BTREE_V1_SIGNATURE);
        buf.push(self.node_type);
        buf.push(self.level);
        buf.extend_from_slice(&self.entries_used.to_le_bytes());
        buf.extend_from_slice(&self.left_sibling.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&self.right_sibling.to_le_bytes()[..sizeof_addr]);
        for (i, &child) in self.children.iter().enumerate() {
            buf.extend_from_slice(&self.keys[i].to_le_bytes()[..sizeof_size]);
            buf.extend_from_slice(&child.to_le_bytes()[..sizeof_addr]);
        }
        buf.extend_from_slice(&self.keys[self.children.len()].to_le_bytes()[..sizeof_size]);
        buf.resize(node_size, 0);
        Ok(buf)
    }

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

impl ChunkKey {
    /// The key describing the stored chunk whose grid position is `scaled`.
    ///
    /// `dims` is the *layout message's* chunk shape — `rank + 1` entries, the
    /// last being the element size — because the key stores element offsets,
    /// `scaled[u] * dims[u]` (`H5D__btree_encode_key`, H5Dbtree.c). A stored
    /// chunk begins at element 0 of its own footprint, so the element
    /// dimension of `scaled` is 0 and this appends it rather than asking the
    /// caller for it.
    pub fn for_chunk(scaled: &[u64], dims: &[u64], chunk_size: u32, filter_mask: u32) -> Self {
        let mut offsets: Vec<u64> = scaled
            .iter()
            .zip(dims)
            .map(|(&s, &d)| s.saturating_mul(d))
            .collect();
        offsets.push(0);
        Self {
            chunk_size,
            filter_mask,
            offsets,
        }
    }

    /// The right-boundary key closing a tree whose greatest chunk sits at
    /// `scaled`: a zero-width chunk one element-size past it.
    ///
    /// A v1 B-tree stores both bounds of every child, and a search descends
    /// only where `lt_key <= target < rt_key` under the *lexicographic* order
    /// of `H5VM_vector_cmp_u` (H5VMprivate.h). Moving the element dimension —
    /// the least significant one, and 0 for every stored chunk — on by one is
    /// what makes the greatest chunk fall inside its own node instead of past
    /// its right bound; libhdf5 reaches the same key from the other side, by
    /// setting `scaled + 1` when it opens a node (`H5D__btree_new_node`) and
    /// leaving it alone for every insert that lands within the bound.
    pub fn right_bound(scaled: &[u64], dims: &[u64]) -> Self {
        let mut offsets: Vec<u64> = scaled
            .iter()
            .zip(dims)
            .map(|(&s, &d)| s.saturating_mul(d))
            .collect();
        offsets.push(*dims.last().unwrap_or(&0));
        Self {
            chunk_size: 0,
            filter_mask: 0,
            offsets,
        }
    }

    /// Encoded width of one key for a chunk of the given `rank` (excluding
    /// the trailing element-size dimension).
    fn encoded_size(rank: usize) -> usize {
        4 + 4 + (rank + 1) * 8
    }

    fn encode_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.chunk_size.to_le_bytes());
        buf.extend_from_slice(&self.filter_mask.to_le_bytes());
        for &o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
    }
}

/// A decoded raw-data-chunk (type-1) B-tree v1 node.
#[derive(Debug, Clone)]
pub struct ChunkBTreeV1Node {
    /// Node level: 0 = leaf (children point at chunk data), >0 = internal
    /// (children point at sub-TREE nodes).
    pub level: u8,
    /// Number of entries (children) used in this node.
    pub entries_used: u16,
    /// Address of the left sibling at the same level, or `UNDEF_ADDR`.
    pub left_sibling: u64,
    /// Address of the right sibling at the same level, or `UNDEF_ADDR`.
    pub right_sibling: u64,
    /// Keys, `entries_used + 1` of them. `keys[i]` describes `children[i]`;
    /// the final key is the right-boundary key.
    pub keys: Vec<ChunkKey>,
    /// Child addresses, `entries_used` of them.
    pub children: Vec<u64>,
}

impl ChunkBTreeV1Node {
    /// Encode this type-1 (raw data chunk) node into exactly `node_size`
    /// bytes.
    ///
    /// Like every v1 B-tree node this is a fixed-size record whose width comes
    /// from the file's "K" value ([`BTreeV1Config::chunk_btree_node_size`])
    /// however few entries it uses; the slots past `entries_used` are zeroed.
    /// The chunk rank comes from the keys themselves, which all describe the
    /// same dataset and so are all the same width.
    pub fn encode(&self, node_size: usize, sizeof_addr: usize) -> FormatResult<Vec<u8>> {
        if self.keys.len() != self.children.len() + 1 {
            return Err(FormatError::InvalidData(format!(
                "chunk B-tree v1 node has {} keys for {} children; a v1 node stores \
                 both bounds of every child, so it needs exactly one more key than \
                 children",
                self.keys.len(),
                self.children.len()
            )));
        }
        if self.entries_used as usize != self.children.len() {
            return Err(FormatError::InvalidData(format!(
                "chunk B-tree v1 node declares {} entries but carries {} children",
                self.entries_used,
                self.children.len()
            )));
        }
        // `keys[0]` always exists: a node with no children still carries its
        // right boundary.
        let key_size = self.keys[0].offsets.len() * 8 + 8;
        if let Some(k) = self
            .keys
            .iter()
            .find(|k| k.offsets.len() * 8 + 8 != key_size)
        {
            return Err(FormatError::InvalidData(format!(
                "chunk B-tree v1 node mixes keys of {} and {} offsets; every key in \
                 one tree describes the same dataset",
                self.keys[0].offsets.len(),
                k.offsets.len()
            )));
        }
        let needed =
            8 + 2 * sizeof_addr + self.children.len() * sizeof_addr + self.keys.len() * key_size;
        if needed > node_size {
            return Err(FormatError::InvalidData(format!(
                "chunk B-tree v1 node needs {needed} bytes for {} children, more than \
                 the {node_size}-byte record the file's 'K' value allows",
                self.children.len()
            )));
        }

        let mut buf = Vec::with_capacity(node_size);
        buf.extend_from_slice(&BTREE_V1_SIGNATURE);
        buf.push(1);
        buf.push(self.level);
        buf.extend_from_slice(&self.entries_used.to_le_bytes());
        buf.extend_from_slice(&self.left_sibling.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&self.right_sibling.to_le_bytes()[..sizeof_addr]);
        for (i, &child) in self.children.iter().enumerate() {
            self.keys[i].encode_into(&mut buf);
            buf.extend_from_slice(&child.to_le_bytes()[..sizeof_addr]);
        }
        self.keys[self.children.len()].encode_into(&mut buf);
        buf.resize(node_size, 0);
        Ok(buf)
    }

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

        let mut pos = 8;
        let left_sibling = read_addr(&buf[pos..], sizeof_addr);
        pos += sizeof_addr;
        let right_sibling = read_addr(&buf[pos..], sizeof_addr);
        pos += sizeof_addr;

        let n = entries_used as usize;
        // Each chunk key: chunk_size(4) + filter_mask(4) + (rank+1)*8.
        let key_size = ChunkKey::encoded_size(rank);
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
            left_sibling,
            right_sibling,
            keys,
            children,
        })
    }
}

/// A bulk-loaded version-1 chunk B-tree: every node laid out in the order it
/// will be written, with the child pointers of the internal levels still node
/// *indices* — the file addresses are only known once the caller hands over a
/// pool of blocks.
///
/// A bulk load rather than a sequence of inserts, the same choice
/// `write_stab` makes for symbol tables: the writer holds every chunk record
/// in memory anyway, and a tree built from all of them at once needs neither
/// the split machinery of `H5B__insert_helper` nor the node-level bookkeeping
/// that goes with it. The result is a tree of uniform depth whose keys mean
/// exactly what libhdf5's mean, which is what its search requires; the *fill*
/// of the nodes differs from what a series of inserts would leave, and
/// nothing reads that.
pub struct ChunkBTreeV1Tree {
    /// Level 0 left to right, then level 1, and so on; the root is last.
    nodes: Vec<TreeNode>,
    /// Byte width of every node, from the file's "K" value and the rank.
    node_size: usize,
    sizeof_addr: usize,
}

/// One node of a [`ChunkBTreeV1Tree`] before its children have addresses.
struct TreeNode {
    level: u8,
    /// `children + 1` keys: the left bound of every child, then the right
    /// bound of the last.
    keys: Vec<ChunkKey>,
    children: TreeChildren,
    /// Same-level neighbours, as indices into [`ChunkBTreeV1Tree::nodes`].
    left: Option<usize>,
    right: Option<usize>,
}

/// What a node's child pointers name, which depends on its level rather than
/// on the value in the slot — so the two are separate variants and no address
/// can be read as an index.
enum TreeChildren {
    /// Level 0: the chunk data addresses themselves.
    Chunks(Vec<u64>),
    /// Above level 0: indices into [`ChunkBTreeV1Tree::nodes`].
    Nodes(Vec<usize>),
}

impl TreeChildren {
    fn len(&self) -> usize {
        match self {
            Self::Chunks(v) => v.len(),
            Self::Nodes(v) => v.len(),
        }
    }
}

impl ChunkBTreeV1Tree {
    /// Bulk-load a tree over `entries` — one `(key, chunk address)` pair per
    /// stored chunk, in ascending key order — closed on the right by
    /// `end_key` ([`ChunkKey::right_bound`]).
    ///
    /// Each level's children are spread evenly over that level's nodes, none
    /// holding more than the `2 * K` entries the file's "K" value allows. A
    /// node's keys are the first key of each child's subtree plus the first
    /// key of whatever follows the node — `end_key` for the rightmost node of
    /// every level — which is the invariant `H5B__find` bisects on.
    ///
    /// An empty index builds no nodes at all: libhdf5 leaves the layout
    /// message's address undefined until the first chunk is inserted, and
    /// [`root_address`](Self::root_address) says the same.
    pub fn build(
        entries: &[(ChunkKey, u64)],
        end_key: ChunkKey,
        config: &BTreeV1Config,
        sizeof_addr: usize,
    ) -> Self {
        let rank = end_key.offsets.len().saturating_sub(1);
        let node_size = config.chunk_btree_node_size(sizeof_addr, rank);
        let cap = (config.chunk_max_entries() as usize).max(1);
        let mut nodes: Vec<TreeNode> = Vec::new();

        if !entries.is_empty() {
            // Level 0: the chunks themselves.
            let mut level_range = spread(entries.len(), cap)
                .into_iter()
                .scan(0usize, |start, m| {
                    let range = *start..*start + m;
                    *start += m;
                    Some(range)
                })
                .map(|r| {
                    let mut keys: Vec<ChunkKey> =
                        entries[r.clone()].iter().map(|(k, _)| k.clone()).collect();
                    keys.push(match entries.get(r.end) {
                        Some((k, _)) => k.clone(),
                        None => end_key.clone(),
                    });
                    TreeNode {
                        level: 0,
                        keys,
                        children: TreeChildren::Chunks(
                            entries[r].iter().map(|&(_, a)| a).collect(),
                        ),
                        left: None,
                        right: None,
                    }
                })
                .collect::<Vec<_>>();
            let mut level: u8 = 0;
            loop {
                let base = nodes.len();
                let count = level_range.len();
                for (i, mut node) in level_range.into_iter().enumerate() {
                    node.left = (i > 0).then(|| base + i - 1);
                    node.right = (i + 1 < count).then(|| base + i + 1);
                    nodes.push(node);
                }
                if count == 1 {
                    break;
                }
                // The level above indexes the one just pushed: each parent
                // takes a run of children and repeats their first keys.
                let children: Vec<usize> = (base..base + count).collect();
                level += 1;
                let mut start = 0usize;
                level_range = spread(count, cap)
                    .into_iter()
                    .map(|m| {
                        let run = &children[start..start + m];
                        start += m;
                        let mut keys: Vec<ChunkKey> =
                            run.iter().map(|&c| nodes[c].keys[0].clone()).collect();
                        keys.push(match children.get(start) {
                            Some(&next) => nodes[next].keys[0].clone(),
                            None => end_key.clone(),
                        });
                        TreeNode {
                            level,
                            keys,
                            children: TreeChildren::Nodes(run.to_vec()),
                            left: None,
                            right: None,
                        }
                    })
                    .collect();
            }
        }

        Self {
            nodes,
            node_size,
            sizeof_addr,
        }
    }

    /// How many node-size blocks this tree needs.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Byte width of every one of those blocks.
    pub fn node_size(&self) -> usize {
        self.node_size
    }

    /// The root's address given the block pool — the address the version-3
    /// data layout message carries — or `UNDEF_ADDR` for an empty index.
    pub fn root_address(&self, addrs: &[u64]) -> u64 {
        match self.nodes.len() {
            0 => crate::format::UNDEF_ADDR,
            n => addrs[n - 1],
        }
    }

    /// Serialize every node to a [`node_size`](Self::node_size)-byte image,
    /// in [`build`](Self::build) order. `addrs[i]` is the address assigned to
    /// node `i`; entries past the node count are ignored, so a caller may
    /// pass a longer pool.
    pub fn encode(&self, addrs: &[u64]) -> FormatResult<Vec<Vec<u8>>> {
        let sibling = |i: Option<usize>| i.map_or(crate::format::UNDEF_ADDR, |j| addrs[j]);
        self.nodes
            .iter()
            .map(|n| {
                let children = match &n.children {
                    TreeChildren::Chunks(v) => v.clone(),
                    TreeChildren::Nodes(v) => v.iter().map(|&j| addrs[j]).collect(),
                };
                ChunkBTreeV1Node {
                    level: n.level,
                    entries_used: n.children.len() as u16,
                    left_sibling: sibling(n.left),
                    right_sibling: sibling(n.right),
                    keys: n.keys.clone(),
                    children,
                }
                .encode(self.node_size, self.sizeof_addr)
            })
            .collect()
    }
}

/// Split `n` items over the fewest nodes of capacity `cap`, as evenly as the
/// count allows: the fewest nodes first, then the remainder one item at a
/// time to the leftmost nodes.
fn spread(n: usize, cap: usize) -> Vec<usize> {
    let k = n.div_ceil(cap);
    if k == 0 {
        return Vec::new();
    }
    let (base, extra) = (n / k, n % k);
    (0..k).map(|i| base + usize::from(i < extra)).collect()
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

    /// The root group's B-tree in a file h5py wrote with no `libver` argument:
    /// one leaf, keys `[0, 24]` bounding the names `alpha`..`gamma`, and 496
    /// zero bytes of unused capacity.
    #[test]
    fn an_encoded_group_btree_node_matches_the_bytes_libhdf5_wrote() {
        let node = BTreeV1Node {
            node_type: 0,
            level: 0,
            entries_used: 1,
            left_sibling: UNDEF_ADDR,
            right_sibling: UNDEF_ADDR,
            keys: vec![0, 24],
            children: vec![0x430],
        };
        let node_size = BTreeV1Config::default().snode_btree_node_size(8, 8);
        assert_eq!(node_size, 544);
        let encoded = node.encode(node_size, 8, 8).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(b"TREE");
        expected.extend_from_slice(&[0, 0, 1, 0]); // type 0, level 0, 1 entry
        expected.extend_from_slice(&[0xff; 16]); // both siblings undefined
        expected.extend_from_slice(&0u64.to_le_bytes()); // key[0]: the empty name
        expected.extend_from_slice(&0x430u64.to_le_bytes()); // child[0]
        expected.extend_from_slice(&24u64.to_le_bytes()); // key[1]: "gamma"
        assert_eq!(&encoded[..expected.len()], &expected[..]);
        assert!(encoded[expected.len()..].iter().all(|&b| b == 0));
        assert_eq!(encoded.len(), node_size);
    }

    /// The two-level shape of a 200-link root group: the interior node's keys
    /// are the right bounds of its children, and its children point at each
    /// other.
    #[test]
    fn an_encoded_group_btree_node_round_trips_an_interior_level() {
        let cfg = BTreeV1Config::default();
        let node_size = cfg.snode_btree_node_size(8, 8);
        let node = BTreeV1Node {
            node_type: 0,
            level: 1,
            entries_used: 2,
            left_sibling: UNDEF_ADDR,
            right_sibling: UNDEF_ADDR,
            keys: vec![0, 896, 1600],
            children: vec![0x1a2a8, 0x1a088],
        };
        let encoded = node.encode(node_size, 8, 8).unwrap();
        let decoded = BTreeV1Node::decode(&encoded, 8, 8, cfg.snode_max_entries()).unwrap();
        assert_eq!(decoded.level, 1);
        assert_eq!(decoded.entries_used, 2);
        assert_eq!(decoded.keys, node.keys);
        assert_eq!(decoded.children, node.children);
        assert_eq!(decoded.left_sibling, UNDEF_ADDR);
    }

    /// The B-tree a freshly created group gets (`H5B_create`): a root node with
    /// no children at all, and the single key that bounds nothing.
    #[test]
    fn an_encoded_group_btree_node_round_trips_an_empty_root() {
        let cfg = BTreeV1Config::default();
        let node = BTreeV1Node {
            node_type: 0,
            level: 0,
            entries_used: 0,
            left_sibling: UNDEF_ADDR,
            right_sibling: UNDEF_ADDR,
            keys: vec![0],
            children: vec![],
        };
        let encoded = node.encode(cfg.snode_btree_node_size(8, 8), 8, 8).unwrap();
        let decoded = BTreeV1Node::decode(&encoded, 8, 8, cfg.snode_max_entries()).unwrap();
        assert_eq!(decoded.entries_used, 0);
        assert!(decoded.children.is_empty());
        assert_eq!(decoded.keys, vec![0]);
    }

    /// A node carrying one key per child, rather than one more, would put every
    /// later key and child at the wrong offset — the encoder refuses it instead
    /// of writing a node that decodes as something else.
    #[test]
    fn a_group_btree_node_refuses_a_key_count_that_does_not_bound_its_children() {
        let cfg = BTreeV1Config::default();
        let node = BTreeV1Node {
            node_type: 0,
            level: 0,
            entries_used: 2,
            left_sibling: UNDEF_ADDR,
            right_sibling: UNDEF_ADDR,
            keys: vec![0, 8],
            children: vec![0x400, 0x800],
        };
        assert!(matches!(
            node.encode(cfg.snode_btree_node_size(8, 8), 8, 8)
                .unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    /// Bulk-load the tree of `nchunks` chunks of a 1-D dataset chunked at
    /// `chunk` elements of `elem` bytes, the shape `gen_chunkidx_btree1`
    /// writes: chunk *i* lives at `0x1000 + i * chunk * elem`.
    fn dense_1d_tree(nchunks: u64, chunk: u64, elem: u64, cfg: &BTreeV1Config) -> ChunkBTreeV1Tree {
        let dims = [chunk, elem];
        let nbytes = (chunk * elem) as u32;
        let entries: Vec<(ChunkKey, u64)> = (0..nchunks)
            .map(|i| {
                (
                    ChunkKey::for_chunk(&[i], &dims, nbytes, 0),
                    0x1000 + i * chunk * elem,
                )
            })
            .collect();
        let end = ChunkKey::right_bound(&[nchunks - 1], &dims);
        ChunkBTreeV1Tree::build(&entries, end, cfg, 8)
    }

    /// The tree libhdf5 writes for a two-chunk 1-D dataset, key for key: the
    /// chunk keys carry element offsets (`scaled * chunk_dim`) and a stored
    /// size, and the right boundary is a zero-width chunk one element-size
    /// past the last one. Taken from a file h5py wrote at `libver='earliest'`.
    #[test]
    fn a_bulk_loaded_chunk_tree_keys_the_way_libhdf5_does() {
        let cfg = BTreeV1Config::default();
        let tree = dense_1d_tree(2, 4, 4, &cfg);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.node_size(), cfg.chunk_btree_node_size(8, 1));

        let addrs = [0x578u64];
        let images = tree.encode(&addrs).unwrap();
        let node = ChunkBTreeV1Node::decode(&images[0], 8, 1, cfg.chunk_max_entries()).unwrap();
        assert_eq!(images[0].len(), tree.node_size());
        assert_eq!(node.level, 0);
        assert_eq!(node.entries_used, 2);
        assert_eq!(node.children, vec![0x1000, 0x1010]);
        assert_eq!(node.left_sibling, UNDEF_ADDR);
        assert_eq!(node.right_sibling, UNDEF_ADDR);
        let offsets: Vec<&[u64]> = node.keys.iter().map(|k| k.offsets.as_slice()).collect();
        assert_eq!(offsets, vec![&[0, 0], &[4, 0], &[4, 4]]);
        assert_eq!(
            node.keys.iter().map(|k| k.chunk_size).collect::<Vec<_>>(),
            vec![16, 16, 0]
        );
        assert_eq!(tree.root_address(&addrs), 0x578);
    }

    /// Past `2 * K` chunks the tree grows a level: the leaves hold every
    /// chunk, the root repeats each leaf's first key, and the key that closes
    /// one leaf is the key that opens the next — the invariant `H5B__find`
    /// bisects on.
    #[test]
    fn a_bulk_loaded_chunk_tree_grows_a_level_past_2k() {
        let cfg = BTreeV1Config::default();
        let two_k = cfg.chunk_max_entries() as u64;
        let nchunks = two_k * 3 + 1;
        let tree = dense_1d_tree(nchunks, 4, 4, &cfg);
        // Four leaves (the fourth holds the one chunk past three full ones)
        // and one root above them.
        assert_eq!(tree.node_count(), 5);

        let addrs: Vec<u64> = (0..tree.node_count() as u64)
            .map(|i| 0x1_0000 + i * 4096)
            .collect();
        let images = tree.encode(&addrs).unwrap();
        let nodes: Vec<ChunkBTreeV1Node> = images
            .iter()
            .map(|img| ChunkBTreeV1Node::decode(img, 8, 1, cfg.chunk_max_entries()).unwrap())
            .collect();

        let (leaves, root) = nodes.split_at(4);
        let root = &root[0];
        assert!(leaves.iter().all(|n| n.level == 0));
        assert_eq!(root.level, 1);
        assert_eq!(root.entries_used, 4);
        assert_eq!(root.children, addrs[..4]);
        assert_eq!(root.left_sibling, UNDEF_ADDR);
        assert_eq!(root.right_sibling, UNDEF_ADDR);

        // Every chunk is in a leaf, and no leaf exceeds the file's rank.
        assert_eq!(
            leaves.iter().map(|n| n.entries_used as u64).sum::<u64>(),
            nchunks
        );
        assert!(leaves.iter().all(|n| n.entries_used as u64 <= two_k));

        // Leaves are doubly linked in key order, each one's closing key opens
        // the next, and the root's keys are the leaves' opening keys.
        for (i, leaf) in leaves.iter().enumerate() {
            assert_eq!(
                leaf.left_sibling,
                if i == 0 { UNDEF_ADDR } else { addrs[i - 1] }
            );
            assert_eq!(
                leaf.right_sibling,
                if i + 1 == leaves.len() {
                    UNDEF_ADDR
                } else {
                    addrs[i + 1]
                }
            );
            assert_eq!(root.keys[i], leaf.keys[0]);
            if let Some(next) = leaves.get(i + 1) {
                assert_eq!(leaf.keys[leaf.keys.len() - 1], next.keys[0]);
            }
        }
        // The right boundary of the whole tree closes both the last leaf and
        // the root: one element-size past the last chunk, zero bytes wide.
        let end = ChunkKey::right_bound(&[nchunks - 1], &[4, 4]);
        assert_eq!(*leaves[3].keys.last().unwrap(), end);
        assert_eq!(*root.keys.last().unwrap(), end);
        assert_eq!(end.offsets, vec![(nchunks - 1) * 4, 4]);
    }

    /// An index with no chunks in it has no nodes at all, and says so with an
    /// undefined root address — the state libhdf5 leaves a chunked dataset in
    /// until its first chunk is written.
    #[test]
    fn an_empty_chunk_tree_has_no_root() {
        let cfg = BTreeV1Config::default();
        let tree = ChunkBTreeV1Tree::build(&[], ChunkKey::right_bound(&[0], &[4, 4]), &cfg, 8);
        assert_eq!(tree.node_count(), 0);
        assert!(tree.encode(&[]).unwrap().is_empty());
        assert_eq!(tree.root_address(&[]), UNDEF_ADDR);
    }

    /// A node carrying more children than the file's "K" value allows would
    /// overrun its own record, which the decoder on the other side refuses
    /// outright — so the encoder refuses first.
    #[test]
    fn a_chunk_node_wider_than_its_record_is_refused() {
        let node = ChunkBTreeV1Node {
            level: 0,
            entries_used: 2,
            left_sibling: UNDEF_ADDR,
            right_sibling: UNDEF_ADDR,
            keys: (0..3)
                .map(|i| ChunkKey::for_chunk(&[i], &[4, 4], 16, 0))
                .collect(),
            children: vec![0x100, 0x200],
        };
        assert!(matches!(
            node.encode(64, 8).unwrap_err(),
            FormatError::InvalidData(_)
        ));
        // A node whose key count does not match its children is refused for
        // the same reason: a v1 node stores both bounds of every child.
        let mut broken = node.clone();
        broken.keys.pop();
        assert!(matches!(
            broken.encode(4096, 8).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    /// Chunk-index (type 1) trees have a different key structure entirely, so
    /// this encoder must not be reached for them.
    #[test]
    fn a_chunk_btree_node_is_not_encoded_by_the_group_encoder() {
        let node = BTreeV1Node {
            node_type: 1,
            level: 0,
            entries_used: 0,
            left_sibling: UNDEF_ADDR,
            right_sibling: UNDEF_ADDR,
            keys: vec![0],
            children: vec![],
        };
        assert!(matches!(
            node.encode(4096, 8, 8).unwrap_err(),
            FormatError::UnsupportedFeature(_)
        ));
    }
}
