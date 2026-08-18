//! Reading and rewriting the link storage of a version-0/1 (symbol-table)
//! group.
//!
//! A classic group keeps no link messages in its object header at all: the
//! header carries a Symbol Table message naming a version-1 B-tree and a local
//! heap, the B-tree indexes fixed-size symbol table nodes (SNODs) holding one
//! entry per link, and the heap holds the names those entries point at. This
//! module is the single owner of that triple on the write side.
//!
//! It rebuilds rather than inserts. `H5G__stab_insert` adds one entry to a
//! live tree and splits nodes as they fill (`H5B__insert_helper`,
//! `H5G__node_insert`); this writer already re-derives a group's whole link
//! set at close, so it lays a fresh heap and a fresh tree over that set and
//! frees the old ones. The result is a structure with the same invariants —
//! entries sorted by name inside each node, a node's key range `(left,
//! right]`, the empty string at heap offset 0 — reached by bulk load instead
//! of by splitting. That is also what removes the node-split boundary: a
//! rebuild fills as many SNODs and as many tree levels as the link count
//! needs, where incremental insertion would have to split to get there.

use crate::format::btree_v1::BTreeV1Node;
use crate::format::free_space::FreeSpaceClass;
use crate::format::local_heap::{
    local_heap_get_string, local_heap_header_size, LocalHeapHeader, LocalHeapImage,
    LOCAL_HEAP_FREE_NULL,
};
use crate::format::superblock::{SymbolTableCache, SymbolTableEntry};
use crate::format::symbol_table::SymbolTableNode;
use crate::format::{FormatContext, FormatError, UNDEF_ADDR};
use crate::io::allocator::FileAllocator;
use crate::io::file_handle::FileHandle;
use crate::io::{FileMeta, IoError, IoResult};

/// The two structures one symbol-table group's links live in, as its Symbol
/// Table message (and, for the root group, the superblock's symbol table
/// entry) names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stab {
    pub btree_addr: u64,
    pub heap_addr: u64,
}

impl Stab {
    /// The Symbol Table message body (`H5O__stab_encode`): the two addresses,
    /// nothing else.
    pub(crate) fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let mut buf = Vec::with_capacity(2 * sa);
        buf.extend_from_slice(&self.btree_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.heap_addr.to_le_bytes()[..sa]);
        buf
    }

    /// Decode a Symbol Table message body.
    pub(crate) fn decode(data: &[u8], ctx: &FormatContext) -> Option<Self> {
        let sa = ctx.sizeof_addr as usize;
        if data.len() < 2 * sa {
            return None;
        }
        Some(Self {
            btree_addr: crate::format::bytes::read_le_addr(data, sa),
            heap_addr: crate::format::bytes::read_le_addr(&data[sa..], sa),
        })
    }
}

/// What one symbol table entry names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StabTarget {
    /// A hard link. `cached` is the target's own symbol table when it has one:
    /// `H5G__link_to_ent` reads the target header's Symbol Table message and
    /// caches it in the scratch pad, and caches nothing when there is none.
    Hard { addr: u64, cached: Option<Stab> },
    /// A soft link (`H5G_CACHED_SLINK`), whose value string lives in this
    /// group's own local heap rather than in the entry.
    Soft { value: String },
}

/// One link of a symbol-table group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StabLink {
    pub name: String,
    pub target: StabTarget,
}

/// The file blocks one symbol table occupies, so a rewrite can free exactly
/// what it superseded.
///
/// Collected while reading rather than re-derived afterwards: the SNOD and
/// tree-node counts are properties of the tree that was there, and a rewrite
/// that guessed them would either strand blocks or free blocks it still uses.
#[derive(Debug, Default, Clone)]
pub(crate) struct StabExtents {
    pub blocks: Vec<(u64, u64)>,
}

/// One symbol-table group's links, and the storage they came out of.
#[derive(Debug, Default)]
pub(crate) struct StabContents {
    pub links: Vec<StabLink>,
    pub extents: StabExtents,
}

/// Bound on B-tree descent depth, matching the reader's walk: a well-formed
/// tree's level strictly decreases, so this only ever stops a corrupt one.
const MAX_BTREE_DEPTH: usize = 256;

/// Read every link a symbol-table group holds, and the blocks its storage
/// occupies.
///
/// A scratch pad's cached symbol table comes back attached to its link and
/// goes back out unchanged by [`write_stab`], which is correct for a child the
/// writer leaves alone. A caller that rewrites a child group's own storage owns
/// updating that cache — a stale pair there sends `H5G__stab_lookup` at freed
/// blocks.
pub(crate) fn read_stab(
    handle: &FileHandle,
    meta: &FileMeta,
    stab: Stab,
) -> IoResult<StabContents> {
    let sa = meta.ctx.sizeof_addr as usize;
    let ss = meta.ctx.sizeof_size as usize;
    let mut out = StabContents::default();

    let heap_hdr_buf = handle.read_at_most(stab.heap_addr, 64)?;
    let heap_hdr = LocalHeapHeader::decode(&heap_hdr_buf, sa, ss)?;
    let heap_data = handle.read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)?;
    out.extents
        .blocks
        .push((stab.heap_addr, local_heap_header_size(sa, ss) as u64));
    out.extents
        .blocks
        .push((heap_hdr.data_addr, heap_hdr.data_size));

    let snod_size = meta.btree.symbol_table_node_size(sa, ss);
    let mut visited = std::collections::HashSet::new();
    let mut snod_addrs = Vec::new();
    collect_snods(
        handle,
        meta,
        stab.btree_addr,
        0,
        &mut visited,
        &mut snod_addrs,
        &mut out.extents,
    )?;

    for snod_addr in snod_addrs {
        out.extents.blocks.push((snod_addr, snod_size as u64));
        let buf = handle.read_at_most(snod_addr, snod_size)?;
        let snod = SymbolTableNode::decode(&buf, sa, ss, meta.btree.sym_leaf_max_entries())?;
        for entry in &snod.entries {
            let name = local_heap_get_string(&heap_data, entry.name_offset)?;
            // The empty string at heap offset 0 is the B-tree's leftmost key,
            // not a link (`H5G__stab_create_components`).
            if name.is_empty() {
                continue;
            }
            let target = match entry.cache {
                SymbolTableCache::SoftLink { value_offset } => StabTarget::Soft {
                    value: local_heap_get_string(&heap_data, value_offset as u64)?,
                },
                SymbolTableCache::SymbolTable {
                    btree_addr,
                    heap_addr,
                } => StabTarget::Hard {
                    addr: entry.obj_header_addr,
                    cached: Some(Stab {
                        btree_addr,
                        heap_addr,
                    }),
                },
                SymbolTableCache::Nothing => StabTarget::Hard {
                    addr: entry.obj_header_addr,
                    cached: None,
                },
            };
            out.links.push(StabLink { name, target });
        }
    }
    Ok(out)
}

/// Walk the B-tree, collecting leaf (SNOD) addresses and the tree's own nodes.
fn collect_snods(
    handle: &FileHandle,
    meta: &FileMeta,
    tree_addr: u64,
    depth: usize,
    visited: &mut std::collections::HashSet<u64>,
    out: &mut Vec<u64>,
    extents: &mut StabExtents,
) -> IoResult<()> {
    if depth > MAX_BTREE_DEPTH || !visited.insert(tree_addr) {
        return Ok(());
    }
    let sa = meta.ctx.sizeof_addr as usize;
    let ss = meta.ctx.sizeof_size as usize;
    let node_size = meta.btree.snode_btree_node_size(sa, ss);
    let buf = handle.read_at_most(tree_addr, node_size)?;
    let node = BTreeV1Node::decode(&buf, sa, ss, meta.btree.snode_max_entries())?;
    extents.blocks.push((tree_addr, node_size as u64));
    if node.level == 0 {
        out.extend_from_slice(&node.children);
        return Ok(());
    }
    for &child in &node.children {
        collect_snods(handle, meta, child, depth + 1, visited, out, extents)?;
    }
    Ok(())
}

/// Lay out and write a fresh symbol table holding exactly `links`.
///
/// The order of `links` is the order their names go into the heap — creation
/// order, as libhdf5's incremental inserts leave it — while the entries
/// themselves are sorted by name, which is the order the B-tree search
/// requires (`H5G__node_insert`'s binary search, `strcmp` on the heap
/// strings). Duplicate names are refused: two entries of one name is a state
/// `H5G__node_insert` explicitly errors on, and it would make one of the two
/// links unreachable.
pub(crate) fn write_stab(
    handle: &FileHandle,
    allocator: &FileAllocator,
    meta: &FileMeta,
    links: &[StabLink],
) -> IoResult<Stab> {
    let sa = meta.ctx.sizeof_addr as usize;
    let ss = meta.ctx.sizeof_size as usize;
    let cfg = &meta.btree;

    let mut heap = LocalHeapImage::with_empty_string();
    let mut entries: Vec<SymbolTableEntry> = Vec::with_capacity(links.len());
    let mut names: Vec<&str> = Vec::with_capacity(links.len());
    for link in links {
        let name_offset = heap.insert_str(&link.name);
        let (obj_header_addr, cache) = match &link.target {
            StabTarget::Hard { addr, cached } => (
                *addr,
                match cached {
                    Some(s) => SymbolTableCache::SymbolTable {
                        btree_addr: s.btree_addr,
                        heap_addr: s.heap_addr,
                    },
                    None => SymbolTableCache::Nothing,
                },
            ),
            StabTarget::Soft { value } => {
                let value_offset = heap.insert_str(value);
                let Ok(value_offset) = u32::try_from(value_offset) else {
                    return Err(IoError::Format(FormatError::InvalidData(format!(
                        "the soft link '{}' lands at heap offset {value_offset}, past the \
                         4-byte offset a symbol table entry's scratch pad can hold",
                        link.name
                    ))));
                };
                (UNDEF_ADDR, SymbolTableCache::SoftLink { value_offset })
            }
        };
        names.push(&link.name);
        entries.push(SymbolTableEntry {
            name_offset,
            obj_header_addr,
            cache,
        });
    }

    // Sort by name, carrying each entry with its name. `sort_by` is stable, so
    // a duplicate pair stays adjacent and the check below sees it.
    let mut ordered: Vec<(&str, SymbolTableEntry)> = names.into_iter().zip(entries).collect();
    ordered.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    if let Some(w) = ordered.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(IoError::InvalidState(format!(
            "a symbol-table group cannot hold two links named '{}'",
            w[0].0
        )));
    }

    // The heap comes first so the tree's keys can name offsets in it.
    let heap_bytes = heap.as_bytes().to_vec();
    let heap_hdr_size = local_heap_header_size(sa, ss) as u64;
    let heap_addr = allocator.allocate(heap_hdr_size, FreeSpaceClass::Metadata);
    let heap_data_addr = allocator.allocate(heap_bytes.len() as u64, FreeSpaceClass::Metadata);
    let heap_hdr = LocalHeapHeader {
        data_size: heap_bytes.len() as u64,
        // A rebuilt heap holds exactly its objects, so there is no free block
        // to record; `H5HL_FREE_NULL` is how libhdf5 spells that.
        free_list_offset: LOCAL_HEAP_FREE_NULL,
        data_addr: heap_data_addr,
    };

    let snod_size = cfg.symbol_table_node_size(sa, ss);
    let node_size = cfg.snode_btree_node_size(sa, ss);
    let leaf_capacity = cfg.sym_leaf_max_entries() as usize;
    let node_capacity = cfg.snode_max_entries() as usize;

    // Leaves: entries packed into SNODs, each carrying the heap offset of its
    // last (greatest) name as the right bound of its key range.
    let mut pending: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut level: Vec<(u64, u64)> = Vec::new(); // (address, right-bound key)
    for chunk in ordered.chunks(leaf_capacity.max(1)) {
        let node = SymbolTableNode {
            entries: chunk.iter().map(|(_, e)| e.clone()).collect(),
        };
        let addr = allocator.allocate(snod_size as u64, FreeSpaceClass::Metadata);
        pending.push((addr, node.encode(snod_size, sa, ss)?));
        level.push((addr, chunk[chunk.len() - 1].1.name_offset));
    }

    // Tree levels, bottom up. An empty group still gets a root node with no
    // children: `H5B_create` makes one when the group is created, before any
    // link exists, so the Symbol Table message always names a real tree.
    let mut tree_level: u8 = 0;
    let root_addr = loop {
        let groups: Vec<Vec<(u64, u64)>> = if level.is_empty() {
            vec![Vec::new()]
        } else {
            level
                .chunks(node_capacity.max(1))
                .map(<[(u64, u64)]>::to_vec)
                .collect()
        };
        // Every address on this level has to exist before any node on it is
        // encoded: each node stores both its siblings' addresses.
        let addrs: Vec<u64> = groups
            .iter()
            .map(|_| allocator.allocate(node_size as u64, FreeSpaceClass::Metadata))
            .collect();
        let mut next: Vec<(u64, u64)> = Vec::with_capacity(groups.len());
        for (i, group) in groups.iter().enumerate() {
            // Heap offset 0 is the empty string, which sorts below every link
            // name, so the leftmost node's range is open at the bottom; every
            // other node starts where its left sibling stopped.
            let left_bound = if i == 0 {
                0
            } else {
                groups[i - 1].last().map_or(0, |&(_, r)| r)
            };
            let mut keys = Vec::with_capacity(group.len() + 1);
            keys.push(left_bound);
            keys.extend(group.iter().map(|&(_, right)| right));
            let node = BTreeV1Node {
                node_type: 0,
                level: tree_level,
                entries_used: group.len() as u16,
                left_sibling: if i == 0 { UNDEF_ADDR } else { addrs[i - 1] },
                right_sibling: if i + 1 == groups.len() {
                    UNDEF_ADDR
                } else {
                    addrs[i + 1]
                },
                keys,
                children: group.iter().map(|&(a, _)| a).collect(),
            };
            pending.push((addrs[i], node.encode(node_size, sa, ss)?));
            next.push((addrs[i], group.last().map_or(0, |&(_, right)| right)));
        }
        if next.len() == 1 {
            break addrs[0];
        }
        level = next;
        tree_level += 1;
    };

    handle.write_at(heap_addr, &heap_hdr.encode(sa, ss))?;
    handle.write_at(heap_data_addr, &heap_bytes)?;
    for (addr, bytes) in pending {
        handle.write_at(addr, &bytes)?;
    }

    Ok(Stab {
        btree_addr: root_addr,
        heap_addr,
    })
}

/// Return the blocks one superseded symbol table occupied to the allocator.
///
/// Metadata throughout: the version-1 B-tree is `H5FD_MEM_BTREE` and the name
/// heap `H5FD_MEM_LHEAP` (H5B.c, H5HL.c:123), both of which
/// `H5FD_FLMAP_DICHOTOMY` sends to `H5FD_MEM_SUPER`.
pub(crate) fn free_stab(allocator: &FileAllocator, extents: &StabExtents) {
    for &(addr, len) in &extents.blocks {
        if addr != 0 && addr != UNDEF_ADDR && len > 0 {
            allocator.free(addr, len, FreeSpaceClass::Metadata);
        }
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::btree_v1::BTreeV1Config;

    fn meta() -> FileMeta {
        FileMeta {
            ctx: FormatContext::default_v3(),
            btree: BTreeV1Config::default(),
            sohm: None,
        }
    }

    struct Scratch {
        dir: std::path::PathBuf,
        handle: FileHandle,
        allocator: FileAllocator,
    }

    impl Scratch {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "rust_hdf5_stab_io_{}_{}_{label}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let handle = FileHandle::create(&dir.join("stab.bin")).unwrap();
            Self {
                dir,
                handle,
                // Nothing lives below 96 in a real v0 file either: that is
                // where the superblock ends.
                allocator: FileAllocator::new(96),
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn hard(name: &str, addr: u64) -> StabLink {
        StabLink {
            name: name.to_string(),
            target: StabTarget::Hard { addr, cached: None },
        }
    }

    /// The three names of a default-h5py file's root group, laid out again:
    /// one leaf, and the same keys libhdf5 wrote — `[0, 24]`, the empty string
    /// and `gamma`.
    #[test]
    fn a_rebuilt_symbol_table_matches_the_shape_libhdf5_wrote() {
        let s = Scratch::new("shape");
        let meta = meta();
        let links = vec![
            hard("alpha", 0x320),
            hard("beta", 0x578),
            hard("gamma", 0x688),
        ];
        let stab = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();

        let node_size = meta.btree.snode_btree_node_size(8, 8);
        let tree = BTreeV1Node::decode(
            &s.handle.read_at(stab.btree_addr, node_size).unwrap(),
            8,
            8,
            meta.btree.snode_max_entries(),
        )
        .unwrap();
        assert_eq!(tree.level, 0);
        assert_eq!(tree.entries_used, 1);
        assert_eq!(tree.keys, vec![0, 24]);
        assert_eq!(tree.left_sibling, UNDEF_ADDR);
        assert_eq!(tree.right_sibling, UNDEF_ADDR);

        let snod_size = meta.btree.symbol_table_node_size(8, 8);
        let snod = SymbolTableNode::decode(
            &s.handle.read_at(tree.children[0], snod_size).unwrap(),
            8,
            8,
            meta.btree.sym_leaf_max_entries(),
        )
        .unwrap();
        assert_eq!(
            snod.entries
                .iter()
                .map(|e| e.name_offset)
                .collect::<Vec<_>>(),
            vec![8, 16, 24]
        );
        assert_eq!(
            snod.entries
                .iter()
                .map(|e| e.obj_header_addr)
                .collect::<Vec<_>>(),
            vec![0x320, 0x578, 0x688]
        );

        let heap_hdr =
            LocalHeapHeader::decode(&s.handle.read_at(stab.heap_addr, 32).unwrap(), 8, 8).unwrap();
        let heap = s
            .handle
            .read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)
            .unwrap();
        assert_eq!(local_heap_get_string(&heap, 0).unwrap(), "");
        assert_eq!(local_heap_get_string(&heap, 8).unwrap(), "alpha");
        assert_eq!(heap_hdr.free_list_offset, LOCAL_HEAP_FREE_NULL);
    }

    /// Entries go into the node sorted by name whatever order they arrive in;
    /// the heap keeps the arrival order, which is where their offsets come
    /// from.
    #[test]
    fn a_rebuilt_symbol_table_sorts_its_entries_and_not_its_heap() {
        let s = Scratch::new("sort");
        let meta = meta();
        let links = vec![
            hard("zulu", 0x100),
            hard("alpha", 0x200),
            hard("mike", 0x300),
        ];
        let stab = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();
        let read = read_stab(&s.handle, &meta, stab).unwrap();
        assert_eq!(
            read.links
                .iter()
                .map(|l| l.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "mike", "zulu"]
        );
        // Arrival order in the heap: zulu at 8, alpha at 16, mike at 24.
        let heap_hdr =
            LocalHeapHeader::decode(&s.handle.read_at(stab.heap_addr, 32).unwrap(), 8, 8).unwrap();
        let heap = s
            .handle
            .read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)
            .unwrap();
        assert_eq!(local_heap_get_string(&heap, 8).unwrap(), "zulu");
        assert_eq!(local_heap_get_string(&heap, 16).unwrap(), "alpha");
    }

    /// Past one SNOD's `2 * sym_leaf_k` entries the load fills more leaves, and
    /// past one tree node's `2 * snode_internal_k` children it grows a level —
    /// the two boundaries an incremental insert would have to reach by
    /// splitting.
    #[test]
    fn a_rebuilt_symbol_table_grows_leaves_then_levels() {
        let meta = meta();
        let leaf_capacity = u64::from(meta.btree.sym_leaf_max_entries());
        let node_capacity = u64::from(meta.btree.snode_max_entries());
        // One more than a single level-0 node can index.
        let count = leaf_capacity * node_capacity + 1;

        let s = Scratch::new("levels");
        let links: Vec<StabLink> = (0..count)
            .map(|i| hard(&format!("obj{i:05}"), 0x1000 + i * 8))
            .collect();
        let stab = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();

        let node_size = meta.btree.snode_btree_node_size(8, 8);
        let root = BTreeV1Node::decode(
            &s.handle.read_at(stab.btree_addr, node_size).unwrap(),
            8,
            8,
            meta.btree.snode_max_entries(),
        )
        .unwrap();
        assert_eq!(root.level, 1, "{count} links must not fit one tree level");

        let read = read_stab(&s.handle, &meta, stab).unwrap();
        assert_eq!(read.links.len(), count as usize);
        let mut names: Vec<&str> = read.links.iter().map(|l| l.name.as_str()).collect();
        let mut expected: Vec<&str> = links.iter().map(|l| l.name.as_str()).collect();
        names.sort_unstable();
        expected.sort_unstable();
        assert_eq!(names, expected);
    }

    /// Siblings at every level are linked both ways, and the leftmost node's
    /// left sibling is undefined: `H5G__node_iterate` walks a level by those
    /// pointers, so a broken chain silently truncates a listing.
    #[test]
    fn a_rebuilt_symbol_table_links_its_siblings_both_ways() {
        let meta = meta();
        // Enough links that the level below the root holds several nodes.
        let count = u64::from(meta.btree.sym_leaf_max_entries())
            * u64::from(meta.btree.snode_max_entries())
            * 2;
        let s = Scratch::new("siblings");
        let links: Vec<StabLink> = (0..count)
            .map(|i| hard(&format!("obj{i:05}"), 0x1000 + i * 8))
            .collect();
        let stab = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();

        let node_size = meta.btree.snode_btree_node_size(8, 8);
        let decode = |addr: u64| {
            BTreeV1Node::decode(
                &s.handle.read_at(addr, node_size).unwrap(),
                8,
                8,
                meta.btree.snode_max_entries(),
            )
            .unwrap()
        };
        let root = decode(stab.btree_addr);
        assert_eq!(root.level, 1);
        assert!(root.children.len() >= 2);
        let mut addr = root.children[0];
        let mut node = decode(addr);
        assert_eq!(node.left_sibling, UNDEF_ADDR);
        let mut seen = 1;
        while node.right_sibling != UNDEF_ADDR {
            let next = decode(node.right_sibling);
            assert_eq!(next.left_sibling, addr);
            addr = node.right_sibling;
            node = next;
            seen += 1;
        }
        assert_eq!(seen, root.children.len());
    }

    /// A soft link's value string goes into the group's own heap and its offset
    /// into the entry's scratch pad; the entry names no object header.
    #[test]
    fn a_rebuilt_symbol_table_carries_a_soft_link_in_its_own_heap() {
        let s = Scratch::new("soft");
        let meta = meta();
        let links = vec![
            hard("real", 0x320),
            StabLink {
                name: "link".to_string(),
                target: StabTarget::Soft {
                    value: "/real".to_string(),
                },
            },
        ];
        let stab = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();
        let read = read_stab(&s.handle, &meta, stab).unwrap();
        assert_eq!(read.links.len(), 2);
        let soft = read.links.iter().find(|l| l.name == "link").unwrap();
        assert_eq!(
            soft.target,
            StabTarget::Soft {
                value: "/real".to_string()
            }
        );

        let snod_size = meta.btree.symbol_table_node_size(8, 8);
        let node_size = meta.btree.snode_btree_node_size(8, 8);
        let tree = BTreeV1Node::decode(
            &s.handle.read_at(stab.btree_addr, node_size).unwrap(),
            8,
            8,
            meta.btree.snode_max_entries(),
        )
        .unwrap();
        let snod = SymbolTableNode::decode(
            &s.handle.read_at(tree.children[0], snod_size).unwrap(),
            8,
            8,
            meta.btree.sym_leaf_max_entries(),
        )
        .unwrap();
        // Sorted: "link" before "real".
        assert_eq!(snod.entries[0].obj_header_addr, UNDEF_ADDR);
        assert!(matches!(
            snod.entries[0].cache,
            SymbolTableCache::SoftLink { .. }
        ));
    }

    /// A child group's cached B-tree/heap pair rides through the rebuild — that
    /// scratch pad is what lets `H5G__stab_lookup` skip reading the child's
    /// header.
    #[test]
    fn a_rebuilt_symbol_table_keeps_a_child_groups_cached_pair() {
        let s = Scratch::new("cached");
        let meta = meta();
        let cached = Stab {
            btree_addr: 0x348,
            heap_addr: 0x568,
        };
        let links = vec![StabLink {
            name: "sub".to_string(),
            target: StabTarget::Hard {
                addr: 0x320,
                cached: Some(cached),
            },
        }];
        let stab = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();
        let read = read_stab(&s.handle, &meta, stab).unwrap();
        assert_eq!(
            read.links[0].target,
            StabTarget::Hard {
                addr: 0x320,
                cached: Some(cached)
            }
        );
    }

    /// A group with no links still gets a heap holding the empty string and a
    /// B-tree root with no children, exactly what `H5G__stab_create_components`
    /// leaves behind.
    #[test]
    fn an_empty_group_still_gets_a_heap_and_a_btree_root() {
        let s = Scratch::new("empty");
        let meta = meta();
        let stab = write_stab(&s.handle, &s.allocator, &meta, &[]).unwrap();
        let node_size = meta.btree.snode_btree_node_size(8, 8);
        let root = BTreeV1Node::decode(
            &s.handle.read_at(stab.btree_addr, node_size).unwrap(),
            8,
            8,
            meta.btree.snode_max_entries(),
        )
        .unwrap();
        assert_eq!(root.level, 0);
        assert_eq!(root.entries_used, 0);
        assert!(root.children.is_empty());
        let read = read_stab(&s.handle, &meta, stab).unwrap();
        assert!(read.links.is_empty());
    }

    /// Two links of one name is a state `H5G__node_insert` errors on, and it
    /// would leave one of them unreachable through the B-tree search.
    #[test]
    fn a_group_cannot_hold_two_links_of_one_name() {
        let s = Scratch::new("dup");
        let meta = meta();
        let links = vec![hard("same", 0x100), hard("same", 0x200)];
        let err = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap_err();
        assert!(
            matches!(&err, IoError::InvalidState(m) if m.contains("same")),
            "{err:?}"
        );
    }

    /// A rewrite hands back every block the old table held, so a file that is
    /// appended to repeatedly reuses that space instead of growing by a whole
    /// tree each time.
    #[test]
    fn a_superseded_symbol_table_returns_its_blocks() {
        let s = Scratch::new("free");
        let meta = meta();
        // Two links first, so the rewrite that follows needs blocks of the same
        // widths and can land in the freed ones.
        let links = [hard("a", 0x100), hard("b", 0x200)];
        let first = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();
        let before = s.allocator.eof();
        let contents = read_stab(&s.handle, &meta, first).unwrap();
        // heap header, heap data, one tree node, one SNOD.
        assert_eq!(contents.extents.blocks.len(), 4);
        free_stab(&s.allocator, &contents.extents);
        let second = write_stab(&s.handle, &s.allocator, &meta, &links).unwrap();
        assert_eq!(
            s.allocator.eof(),
            before,
            "the rewrite reused the freed blocks"
        );
        assert_eq!(read_stab(&s.handle, &meta, second).unwrap().links.len(), 2);
    }

    /// The Symbol Table message body is just the two addresses, and it decodes
    /// back to the same pair.
    #[test]
    fn a_symbol_table_message_round_trips_its_pair() {
        let ctx = FormatContext::default_v3();
        let stab = Stab {
            btree_addr: 0x88,
            heap_addr: 0x2a8,
        };
        let body = stab.encode(&ctx);
        assert_eq!(body.len(), 16);
        assert_eq!(&body[..8], &0x88u64.to_le_bytes());
        assert_eq!(&body[8..], &0x2a8u64.to_le_bytes());
        assert_eq!(Stab::decode(&body, &ctx), Some(stab));
        assert_eq!(Stab::decode(&body[..15], &ctx), None);
    }
}
