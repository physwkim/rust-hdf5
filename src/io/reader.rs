//! HDF5 file reader.
//!
//! Opens an HDF5 file, parses the superblock and root group, and provides
//! access to dataset metadata and raw data.
//!
//! Supports both legacy (v0/v1 superblock, v1 object headers, symbol tables)
//! and modern (v2/v3 superblock, v2 object headers, link messages) formats.

use std::path::Path;

use crate::format::btree_v1::{BTreeV1Node, ChunkBTreeV1Node};
use crate::format::bytes::read_le_uint as read_uint;
use crate::format::fractal_heap::{self, BlockReader, FractalHeapHeader};
use crate::format::global_heap::{
    decode_vlen_reference, vlen_reference_size, GlobalHeapCollection,
};
use crate::format::local_heap::{local_heap_get_string, LocalHeapHeader};
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::data_layout::{self, DataLayoutMessage};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::fill_value::{tiled_fill, FillValueMessage};
use crate::format::messages::filter::{self, FilterPipeline};
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link::LinkTarget;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::*;
use crate::format::object_header::ObjectHeader;
use crate::format::superblock::{detect_superblock_version, SuperblockV0V1, SuperblockV2V3};
use crate::format::symbol_table::SymbolTableNode;
use crate::format::{FormatContext, UNDEF_ADDR};

use crate::io::file_handle::FileHandle;
#[cfg(feature = "mmap")]
use crate::io::file_handle::MmapFileHandle;
use crate::io::IoResult;

/// Read-side metadata for a single dataset.
pub struct DatasetReadInfo {
    /// Dataset name (the link name in the root group).
    pub name: String,
    /// Element datatype.
    pub datatype: DatatypeMessage,
    /// Dataspace (dimensionality).
    pub dataspace: DataspaceMessage,
    /// Data layout (contiguous or compact).
    pub layout: DataLayoutMessage,
    /// Filter pipeline for compressed chunks (None = uncompressed).
    pub filter_pipeline: Option<FilterPipeline>,
    /// Attributes attached to this dataset.
    pub attributes: Vec<AttributeMessage>,
    /// User-defined fill value bytes (one element wide), decoded from the
    /// fill-value message when `fill_defined == 2`. `None` => default
    /// zero-fill. Applied to unallocated chunks and unwritten regions.
    pub fill_value: Option<Vec<u8>>,
}

/// Internal enum to represent what we know about the root group from the
/// superblock. For v2/v3 we have the root group object header address; for
/// v0/v1 we have a B-tree and local heap that index the root group's children.
/// These are stored for potential future use (e.g., SWMR refresh).
#[allow(dead_code)]
enum RootGroupInfo {
    V2V3 {
        root_group_object_header_address: u64,
    },
    V0V1 {
        root_obj_header_addr: u64,
        btree_addr: u64,
        heap_addr: u64,
    },
}

/// HDF5 file reader.
pub struct Hdf5Reader {
    handle: FileHandle,
    ctx: FormatContext,
    /// End-of-file address from the superblock.
    _eof: u64,
    #[allow(dead_code)]
    root_group_info: RootGroupInfo,
    datasets: Vec<DatasetReadInfo>,
    /// Attributes on the root group (file-level attributes).
    root_attributes: Vec<AttributeMessage>,
    /// Attributes on non-root groups, keyed by group path (no leading `/`).
    group_attributes: std::collections::HashMap<String, Vec<AttributeMessage>>,
}

impl Hdf5Reader {
    /// Open an existing HDF5 file using memory-mapped I/O for zero-copy reads.
    ///
    /// Available when the `mmap` feature is enabled. The entire file is
    /// mapped into memory, avoiding read syscalls. This can be significantly
    /// faster for random-access patterns on large files.
    #[cfg(feature = "mmap")]
    pub fn open_mmap(path: &Path) -> IoResult<(Self, MmapFileHandle)> {
        // Open normally first to parse metadata
        let reader = Self::open(path)?;
        // Also open an mmap handle for zero-copy data access
        let mmap = MmapFileHandle::open(path)?;
        Ok((reader, mmap))
    }

    /// Open an existing HDF5 file in SWMR read mode using the env-var-derived
    /// locking policy.
    ///
    /// Currently identical to `open()`, but indicates intent to use
    /// `refresh()` for re-reading metadata written by a concurrent SWMR writer.
    pub fn open_swmr(path: &Path) -> IoResult<Self> {
        Self::open(path)
    }

    /// Open an existing HDF5 file in SWMR read mode with an explicit locking
    /// policy.
    pub fn open_swmr_with_locking(
        path: &Path,
        locking: crate::io::locking::FileLocking,
    ) -> IoResult<Self> {
        Self::open_with_locking(path, locking)
    }

    /// Open an existing HDF5 file for reading using the env-var-derived
    /// locking policy.
    ///
    /// Auto-detects the superblock version and uses the appropriate code path:
    /// - v0/v1: legacy format with symbol tables and B-tree v1
    /// - v2/v3: modern format with link messages
    pub fn open(path: &Path) -> IoResult<Self> {
        Self::open_with_locking(
            path,
            crate::io::locking::FileLocking::from_env_or(Default::default()),
        )
    }

    /// Open an existing HDF5 file for reading with an explicit locking policy.
    pub fn open_with_locking(
        path: &Path,
        locking: crate::io::locking::FileLocking,
    ) -> IoResult<Self> {
        let mut handle = FileHandle::open_read_with_locking(path, locking)?;

        // Read enough bytes to detect the superblock version and parse it.
        let sb_buf = handle.read_at_most(0, 1024)?;
        let version = detect_superblock_version(&sb_buf)?;

        match version {
            0 | 1 => Self::open_v0v1(handle, &sb_buf),
            2 | 3 => Self::open_v2v3(handle, &sb_buf),
            v => Err(crate::io::IoError::Format(
                crate::format::FormatError::InvalidVersion(v),
            )),
        }
    }

    /// Open a file with v2/v3 superblock (existing code path).
    fn open_v2v3(mut handle: FileHandle, sb_buf: &[u8]) -> IoResult<Self> {
        let sb = SuperblockV2V3::decode(sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Read root group object header, following continuation blocks.
        let root_header =
            Self::read_object_header_full(&mut handle, &ctx, sb.root_group_object_header_address)?;

        // Walk link messages to discover datasets and group attributes.
        let (datasets, group_attributes) =
            Self::discover_datasets_from_links(&mut handle, &root_header, &ctx)?;

        // Collect root group attributes
        let mut root_attributes = Vec::new();
        for msg in &root_header.messages {
            if msg.msg_type == MSG_ATTRIBUTE {
                if let Ok((attr, _)) = AttributeMessage::decode(&msg.data, &ctx) {
                    root_attributes.push(attr);
                }
            }
        }

        Ok(Self {
            handle,
            ctx,
            _eof: sb.end_of_file_address,
            root_group_info: RootGroupInfo::V2V3 {
                root_group_object_header_address: sb.root_group_object_header_address,
            },
            datasets,
            root_attributes,
            group_attributes,
        })
    }

    /// Open a file with v0/v1 superblock (legacy format).
    fn open_v0v1(mut handle: FileHandle, sb_buf: &[u8]) -> IoResult<Self> {
        let sb = SuperblockV0V1::decode(sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        let ste = &sb.root_symbol_table_entry;
        let btree_addr = ste.btree_addr;
        let heap_addr = ste.heap_addr;

        // If the root group STE has cache_type == 1 (group with B-tree),
        // walk the B-tree + local heap to discover datasets.
        // Otherwise, try reading the root group object header for a symbol
        // table message.
        let (btree_addr, heap_addr) = if ste.cache_type == 1 {
            (btree_addr, heap_addr)
        } else {
            // Read the root group's object header to find a symbol table msg
            Self::find_stab_in_object_header(&mut handle, &ctx, ste.obj_header_addr)?
        };

        let datasets = if btree_addr != UNDEF_ADDR && heap_addr != UNDEF_ADDR {
            Self::discover_datasets_from_btree(&mut handle, &ctx, btree_addr, heap_addr)?
        } else {
            Vec::new()
        };

        Ok(Self {
            handle,
            ctx,
            _eof: sb.end_of_file_address,
            root_group_info: RootGroupInfo::V0V1 {
                root_obj_header_addr: ste.obj_header_addr,
                btree_addr,
                heap_addr,
            },
            datasets,
            root_attributes: Vec::new(),
            group_attributes: std::collections::HashMap::new(),
        })
    }

    /// Find symbol table message (btree_addr, heap_addr) in an object header.
    fn find_stab_in_object_header(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        obj_header_addr: u64,
    ) -> IoResult<(u64, u64)> {
        let buf = handle.read_at_most(obj_header_addr, 4096)?;
        let (header, _) = ObjectHeader::decode_any(&buf)?;

        for msg in &header.messages {
            if msg.msg_type == MSG_SYMBOL_TABLE {
                // Symbol table message: btree_addr(O) + heap_addr(O)
                let sa = ctx.sizeof_addr as usize;
                if msg.data.len() >= 2 * sa {
                    let btree = read_uint(&msg.data, sa);
                    let heap = read_uint(&msg.data[sa..], sa);
                    return Ok((btree, heap));
                }
            }
        }

        Ok((UNDEF_ADDR, UNDEF_ADDR))
    }

    /// Discover datasets by walking link messages in a v2 object header.
    /// Recursively descends into groups, prefixing dataset names with the group path.
    #[allow(clippy::type_complexity)]
    fn discover_datasets_from_links(
        handle: &mut FileHandle,
        root_header: &ObjectHeader,
        ctx: &FormatContext,
    ) -> IoResult<(
        Vec<DatasetReadInfo>,
        std::collections::HashMap<String, Vec<AttributeMessage>>,
    )> {
        let mut group_attrs = std::collections::HashMap::new();
        let mut visited = std::collections::HashSet::new();
        let datasets = Self::discover_datasets_recursive(
            handle,
            root_header,
            ctx,
            "",
            &mut group_attrs,
            &mut visited,
        )?;
        Ok((datasets, group_attrs))
    }

    fn discover_datasets_recursive(
        handle: &mut FileHandle,
        header: &ObjectHeader,
        ctx: &FormatContext,
        prefix: &str,
        group_attrs: &mut std::collections::HashMap<String, Vec<AttributeMessage>>,
        visited: &mut std::collections::HashSet<u64>,
    ) -> IoResult<Vec<DatasetReadInfo>> {
        // Bound recursion depth on a hostile/corrupt file.
        const MAX_GROUP_DEPTH: usize = 256;
        let depth = if prefix.is_empty() {
            0
        } else {
            prefix.matches('/').count() + 1
        };
        if depth > MAX_GROUP_DEPTH {
            return Ok(Vec::new());
        }

        let mut datasets = Vec::new();

        // Collect every link in this group: inline `Link` messages plus, for
        // groups using dense storage, links held in a fractal heap referenced
        // by the `Link Info` message.
        let mut links: Vec<LinkMessage> = Vec::new();
        for msg in &header.messages {
            if msg.msg_type == MSG_LINK {
                if let Ok((link, _)) = LinkMessage::decode(&msg.data, ctx) {
                    links.push(link);
                }
            } else if msg.msg_type == MSG_LINK_INFO {
                if let Ok((info, _)) = LinkInfoMessage::decode(&msg.data, ctx) {
                    if info.fractal_heap_address != UNDEF_ADDR {
                        let dense = Self::read_dense_links(handle, ctx, info.fractal_heap_address)?;
                        links.extend(dense);
                    }
                }
            }
        }

        for link in &links {
            if let LinkTarget::Hard { address } = &link.target {
                let full_name = if prefix.is_empty() {
                    link.name.clone()
                } else {
                    format!("{}/{}", prefix, link.name)
                };

                // Try to read as a dataset. A target whose object header
                // fails to decode (e.g. a stale link left by a deletion) is
                // skipped rather than aborting the whole file open.
                match Self::read_dataset_from_object_header(handle, ctx, *address, &full_name) {
                    Ok(Some(info)) => {
                        datasets.push(info);
                        continue;
                    }
                    Err(_) => continue,
                    Ok(None) => {}
                }
                // Not a dataset — treat as a group. Break hard-link cycles:
                // only descend into an object-header address once.
                if !visited.insert(*address) {
                    continue;
                }
                {
                    // Read its full object header (with continuations).
                    if let Ok(child_header) = Self::read_object_header_full(handle, ctx, *address) {
                        // Capture group attributes (e.g. the NeXus `NX_class`
                        // marker), keyed by path, whether or not it has
                        // children.
                        let mut attrs = Vec::new();
                        for m in &child_header.messages {
                            if m.msg_type == MSG_ATTRIBUTE {
                                if let Ok((a, _)) = AttributeMessage::decode(&m.data, ctx) {
                                    attrs.push(a);
                                }
                            }
                        }
                        if !attrs.is_empty() {
                            group_attrs.insert(full_name.clone(), attrs);
                        }
                        let child_ds = Self::discover_datasets_recursive(
                            handle,
                            &child_header,
                            ctx,
                            &full_name,
                            group_attrs,
                            visited,
                        )?;
                        datasets.extend(child_ds);
                    }
                }
            }
        }
        Ok(datasets)
    }

    /// Read every link stored in a group's dense (fractal-heap) link storage.
    ///
    /// The `Link Info` message gives the fractal-heap address; each managed
    /// object in the heap is an encoded `Link` message. Returns the decoded
    /// links (hard and soft).
    fn read_dense_links(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        fractal_heap_addr: u64,
    ) -> IoResult<Vec<LinkMessage>> {
        // Read the fractal heap header. Its on-disk size depends only on the
        // address/length widths, so a generous prefix read covers it.
        let hdr_buf = handle.read_at_most(fractal_heap_addr, 512)?;
        let fh_header = match FractalHeapHeader::decode(&hdr_buf, ctx) {
            Ok(h) => h,
            Err(_) => return Ok(Vec::new()),
        };

        // Walk the heap's managed blocks; each block hands back a payload
        // region holding one or more packed encoded `Link` messages.
        let mut br = HandleBlockReader { handle };
        let payloads = match fractal_heap::collect_managed_objects(&fh_header, ctx, &mut br) {
            Ok(p) => p,
            Err(_) => return Ok(Vec::new()),
        };

        let mut links = Vec::new();
        for payload in payloads {
            // Decode packed `Link` messages sequentially. Each decode reports
            // its consumed length; stop at the first byte that is not a valid
            // link (trailing free space or an unrelated managed object).
            let mut pos = 0;
            while pos < payload.len() {
                // A v1 link message starts with version byte 1.
                if payload[pos] != 1 {
                    break;
                }
                match LinkMessage::decode(&payload[pos..], ctx) {
                    Ok((link, consumed)) if consumed > 0 => {
                        links.push(link);
                        pos += consumed;
                    }
                    _ => break,
                }
            }
        }

        Ok(links)
    }

    /// Discover datasets by walking the B-tree v1 + local heap (legacy format).
    ///
    /// Recurses into subgroups: a symbol-table entry whose `cache_type == 1`
    /// carries scratch-pad `btree_addr`/`heap_addr` for the subgroup; for
    /// other entries the child object header is read for a symbol-table
    /// message. Discovered dataset names are prefixed with the group path.
    fn discover_datasets_from_btree(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        btree_addr: u64,
        heap_addr: u64,
    ) -> IoResult<Vec<DatasetReadInfo>> {
        let mut datasets = Vec::new();
        let mut visited = std::collections::HashSet::new();
        Self::discover_datasets_from_btree_recursive(
            handle,
            ctx,
            btree_addr,
            heap_addr,
            "",
            0,
            &mut datasets,
            &mut visited,
        )?;
        Ok(datasets)
    }

    /// Recursive worker for `discover_datasets_from_btree`. `prefix` is the
    /// path of the group being scanned; `depth` bounds recursion.
    #[allow(clippy::too_many_arguments)]
    fn discover_datasets_from_btree_recursive(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        btree_addr: u64,
        heap_addr: u64,
        prefix: &str,
        depth: usize,
        datasets: &mut Vec<DatasetReadInfo>,
        visited: &mut std::collections::HashSet<u64>,
    ) -> IoResult<()> {
        // Bound legacy-group nesting depth on a hostile/corrupt file.
        const MAX_GROUP_DEPTH: usize = 256;
        if depth > MAX_GROUP_DEPTH {
            return Ok(());
        }
        if btree_addr == UNDEF_ADDR || heap_addr == UNDEF_ADDR {
            return Ok(());
        }

        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        // Read the local heap header + data for this group.
        let heap_hdr_buf = handle.read_at_most(heap_addr, 64)?;
        let heap_hdr = LocalHeapHeader::decode(&heap_hdr_buf, sa, ss)?;
        let heap_data = handle.read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)?;

        // Collect all SNOD addresses by walking the B-tree.
        let snod_addrs = Self::collect_snod_addresses(handle, btree_addr, sa, ss, 0)?;

        for snod_addr in snod_addrs {
            let snod_buf = handle.read_at_most(snod_addr, 8192)?;
            let snod = SymbolTableNode::decode(&snod_buf, sa, ss)?;

            for entry in &snod.entries {
                let name = local_heap_get_string(&heap_data, entry.name_offset)?;
                // Skip empty names (root group self-reference).
                if name.is_empty() {
                    continue;
                }
                let full_name = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{}/{}", prefix, name)
                };

                // Try to read this entry as a dataset. A target whose
                // object header fails to decode is skipped, not fatal.
                match Self::read_dataset_from_object_header(
                    handle,
                    ctx,
                    entry.obj_header_addr,
                    &full_name,
                ) {
                    Ok(Some(info)) => {
                        datasets.push(info);
                        continue;
                    }
                    Err(_) => continue,
                    Ok(None) => {}
                }

                // Not a dataset — it is a subgroup. Break cycles: descend
                // into each group object header at most once.
                if !visited.insert(entry.obj_header_addr) {
                    continue;
                }

                // Find its B-tree + local heap and recurse, prefixing names
                // with the group path.
                let (sub_btree, sub_heap) = if entry.cache_type == 1 {
                    // Scratch-pad caches the subgroup's symbol-table info.
                    (entry.btree_addr, entry.heap_addr)
                } else {
                    // No scratch pad — read the child object header for a
                    // symbol-table message.
                    Self::find_stab_in_object_header(handle, ctx, entry.obj_header_addr)?
                };

                if sub_btree != UNDEF_ADDR && sub_heap != UNDEF_ADDR {
                    Self::discover_datasets_from_btree_recursive(
                        handle,
                        ctx,
                        sub_btree,
                        sub_heap,
                        &full_name,
                        depth + 1,
                        datasets,
                        visited,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Recursively walk a B-tree v1 to collect leaf-level SNOD addresses.
    fn collect_snod_addresses(
        handle: &mut FileHandle,
        tree_addr: u64,
        sizeof_addr: usize,
        sizeof_size: usize,
        depth: usize,
    ) -> IoResult<Vec<u64>> {
        // A well-formed v1 B-tree's level strictly decreases with depth;
        // bound the descent so a corrupt/cyclic tree cannot recurse forever.
        if depth > 256 {
            return Ok(Vec::new());
        }
        let buf = handle.read_at_most(tree_addr, 8192)?;
        let node = BTreeV1Node::decode(&buf, sizeof_addr, sizeof_size)?;

        if node.level == 0 {
            // Leaf level: children are SNOD addresses
            Ok(node.children.clone())
        } else {
            // Internal level: children are sub-TREE addresses
            let mut addrs = Vec::new();
            for &child_addr in &node.children {
                let child_addrs = Self::collect_snod_addresses(
                    handle,
                    child_addr,
                    sizeof_addr,
                    sizeof_size,
                    depth + 1,
                )?;
                addrs.extend(child_addrs);
            }
            Ok(addrs)
        }
    }

    /// Read an object header at `addr` and return it with the messages from
    /// every object-header continuation block flattened in.
    ///
    /// Handles both wire formats:
    /// - v1 headers: continuation blocks are bare v1 messages (type:u16,
    ///   size:u16, flags:u8, reserved:3, data, padded to 8-byte alignment).
    /// - v2 headers: continuation blocks are `"OCHK"(4) + messages +
    ///   checksum(4)` with v2 message headers (type:u8, size:u16, flags:u8,
    ///   and a 2-byte creation-order field when the header tracks creation
    ///   order).
    ///
    /// Nested continuations are followed; the total block count is bounded.
    fn read_object_header_full(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        addr: u64,
    ) -> IoResult<ObjectHeader> {
        /// Bound on the number of continuation blocks followed per header.
        const MAX_CONT_BLOCKS: usize = 4096;

        let buf = handle.read_at_most(addr, 8192)?;
        let (mut header, _) = ObjectHeader::decode_any(&buf)?;

        // A v1 header has no "OHDR" signature; detect by it.
        let is_v2 = buf.len() >= 4 && buf[0..4] == crate::format::object_header::OHDR_SIGNATURE;
        // v2 creation-order tracking is recorded in object-header flag bit 2.
        let track_creation_order = is_v2 && (header.flags & 0x04) != 0;

        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        // Collect continuation references from a slice of messages.
        let collect = |msgs: &[crate::format::object_header::ObjectHeaderMessage],
                       out: &mut Vec<(u64, u64)>| {
            for msg in msgs {
                if msg.msg_type == MSG_OBJ_HEADER_CONTINUATION && msg.data.len() >= sa + ss {
                    let cont_addr = read_uint(&msg.data, sa);
                    let cont_len = read_uint(&msg.data[sa..], ss);
                    out.push((cont_addr, cont_len));
                }
            }
        };

        let mut pending: Vec<(u64, u64)> = Vec::new();
        collect(&header.messages, &mut pending);

        let mut visited = std::collections::HashSet::new();
        let mut blocks_read = 0usize;

        while let Some((cont_addr, cont_len)) = pending.pop() {
            if cont_addr == UNDEF_ADDR || cont_addr == 0 || cont_len == 0 {
                continue;
            }
            if !visited.insert(cont_addr) {
                continue; // already followed — guard against cycles
            }
            blocks_read += 1;
            if blocks_read > MAX_CONT_BLOCKS {
                break;
            }

            let cont_buf = handle.read_at_most(cont_addr, cont_len as usize)?;
            let mut new_msgs = Vec::new();
            Self::parse_continuation_block(&cont_buf, is_v2, track_creation_order, &mut new_msgs);
            collect(&new_msgs, &mut pending);
            header.messages.extend(new_msgs);
        }

        Ok(header)
    }

    /// Parse the messages out of a single object-header continuation block.
    ///
    /// For v2 (`is_v2`) the block is `"OCHK"(4) + messages + checksum(4)`;
    /// for v1 it is bare messages. Null/padding messages (type 0) are skipped.
    fn parse_continuation_block(
        cont_buf: &[u8],
        is_v2: bool,
        track_creation_order: bool,
        out: &mut Vec<crate::format::object_header::ObjectHeaderMessage>,
    ) {
        if is_v2 {
            // "OCHK"(4) signature + messages + checksum(4).
            if cont_buf.len() < 8 || cont_buf[0..4] != *b"OCHK" {
                return;
            }
            let msgs_end = cont_buf.len() - 4; // strip trailing checksum
            let mut pos = 4; // skip "OCHK" signature
                             // v2 message header: type(1) + size(2) + flags(1) [+ crt_order(2)]
            let hdr_size = if track_creation_order { 6 } else { 4 };
            while pos + hdr_size <= msgs_end {
                let msg_type = cont_buf[pos];
                let data_size = u16::from_le_bytes([cont_buf[pos + 1], cont_buf[pos + 2]]) as usize;
                let msg_flags = cont_buf[pos + 3];
                pos += hdr_size;
                if pos + data_size > msgs_end {
                    break;
                }
                if msg_type != 0 {
                    out.push(crate::format::object_header::ObjectHeaderMessage {
                        msg_type,
                        flags: msg_flags,
                        data: cont_buf[pos..pos + data_size].to_vec(),
                    });
                }
                pos += data_size;
            }
        } else {
            // v1 continuation: bare messages, 8-byte aligned, no prefix.
            let mut pos = 0;
            while pos + 8 <= cont_buf.len() {
                let msg_type = u16::from_le_bytes([cont_buf[pos], cont_buf[pos + 1]]);
                let data_size = u16::from_le_bytes([cont_buf[pos + 2], cont_buf[pos + 3]]) as usize;
                let msg_flags = cont_buf[pos + 4];
                pos += 8; // type(2) + size(2) + flags(1) + reserved(3)
                if pos + data_size > cont_buf.len() {
                    break;
                }
                if msg_type != 0 {
                    out.push(crate::format::object_header::ObjectHeaderMessage {
                        msg_type: msg_type as u8,
                        flags: msg_flags,
                        data: cont_buf[pos..pos + data_size].to_vec(),
                    });
                }
                pos += data_size;
                pos = (pos + 7) & !7; // v1 8-byte alignment
            }
        }
    }

    /// Read a dataset's object header and extract metadata. Returns None if
    /// the object is not a dataset (e.g., it's a group).
    fn read_dataset_from_object_header(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        addr: u64,
        name: &str,
    ) -> IoResult<Option<DatasetReadInfo>> {
        // Read the object header, following continuation blocks (v1 and v2).
        let header = Self::read_object_header_full(handle, ctx, addr)?;

        let mut datatype = None;
        let mut dataspace = None;
        let mut layout = None;
        let mut filter_pipeline = None;
        let mut fill_value = None;
        let mut attributes = Vec::new();

        for msg in &header.messages {
            match msg.msg_type {
                MSG_DATATYPE => {
                    if let Ok((dt, _)) = DatatypeMessage::decode(&msg.data, ctx) {
                        datatype = Some(dt);
                    }
                }
                MSG_DATASPACE => {
                    if let Ok((ds, _)) = DataspaceMessage::decode(&msg.data, ctx) {
                        dataspace = Some(ds);
                    }
                }
                MSG_DATA_LAYOUT => {
                    if let Ok((dl, _)) = DataLayoutMessage::decode(&msg.data, ctx) {
                        layout = Some(dl);
                    }
                }
                MSG_FILTER_PIPELINE => {
                    if let Ok((fp, _)) = FilterPipeline::decode(&msg.data) {
                        if !fp.filters.is_empty() {
                            filter_pipeline = Some(fp);
                        }
                    }
                }
                MSG_FILL_VALUE => {
                    if let Ok((fv, _)) = FillValueMessage::decode(&msg.data) {
                        if fv.fill_defined == 2 {
                            fill_value = fv.fill_value;
                        }
                    }
                }
                MSG_ATTRIBUTE => {
                    if let Ok((attr, _)) = AttributeMessage::decode(&msg.data, ctx) {
                        attributes.push(attr);
                    }
                }
                _ => {}
            }
        }

        if let (Some(dt), Some(ds), Some(dl)) = (datatype, dataspace, layout) {
            Ok(Some(DatasetReadInfo {
                name: name.to_string(),
                datatype: dt,
                dataspace: ds,
                layout: dl,
                filter_pipeline,
                attributes,
                fill_value,
            }))
        } else {
            Ok(None)
        }
    }

    /// Return the names of all datasets in the root group.
    pub fn dataset_names(&self) -> Vec<&str> {
        self.datasets.iter().map(|d| d.name.as_str()).collect()
    }

    /// Return metadata for a dataset by name.
    pub fn dataset_info(&self, name: &str) -> Option<&DatasetReadInfo> {
        self.datasets.iter().find(|d| d.name == name)
    }

    /// Return the attribute names of a dataset.
    pub fn dataset_attr_names(&self, name: &str) -> IoResult<Vec<String>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        Ok(info.attributes.iter().map(|a| a.name.clone()).collect())
    }

    /// Return a specific attribute by dataset name and attribute name.
    pub fn dataset_attr(&self, ds_name: &str, attr_name: &str) -> IoResult<&AttributeMessage> {
        let info = self
            .dataset_info(ds_name)
            .ok_or_else(|| crate::io::IoError::NotFound(ds_name.to_string()))?;
        info.attributes
            .iter()
            .find(|a| a.name == attr_name)
            .ok_or_else(|| crate::io::IoError::NotFound(format!("{}:{}", ds_name, attr_name)))
    }

    /// Return the names of root-level (file) attributes.
    pub fn root_attr_names(&self) -> Vec<String> {
        self.root_attributes
            .iter()
            .map(|a| a.name.clone())
            .collect()
    }

    /// Return a root-level attribute by name.
    pub fn root_attr(&self, name: &str) -> Option<&AttributeMessage> {
        self.root_attributes.iter().find(|a| a.name == name)
    }

    /// Return the attribute names of a non-root group (path without a
    /// leading `/`, e.g. `"detector"` or `"entry/instrument"`).
    pub fn group_attr_names(&self, group_path: &str) -> Vec<String> {
        self.group_attributes
            .get(group_path)
            .map(|v| v.iter().map(|a| a.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Return a non-root group's attribute by name.
    pub fn group_attr(&self, group_path: &str, name: &str) -> Option<&AttributeMessage> {
        self.group_attributes
            .get(group_path)?
            .iter()
            .find(|a| a.name == name)
    }

    /// Return the dimensions of a dataset.
    pub fn dataset_shape(&self, name: &str) -> IoResult<Vec<u64>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        Ok(info.dataspace.dims.clone())
    }

    /// Read the raw bytes of a dataset.
    pub fn read_dataset_raw(&mut self, name: &str) -> IoResult<Vec<u8>> {
        let datatype = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?
            .datatype
            .clone();
        let mut data = self.read_dataset_raw_unconverted(name)?;
        Self::apply_post_filter_conversion(&mut data, &datatype)?;
        Ok(data)
    }

    /// Read a full dataset producing the raw filter-pipeline output, before
    /// the post-filter datatype conversion. `read_dataset_raw` wraps this
    /// and applies the conversion exactly once; internal callers that go on
    /// to convert themselves (e.g. the `read_slice` fallback) must use this
    /// to avoid converting twice.
    fn read_dataset_raw_unconverted(&mut self, name: &str) -> IoResult<Vec<u8>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;

        // Clone layout to avoid borrow conflict with &mut self in read methods.
        let layout = info.layout.clone();

        // Clone filter pipeline to avoid borrow conflict.
        let pipeline = info.filter_pipeline.clone();

        let data = match &layout {
            DataLayoutMessage::Contiguous { address, size } => {
                if *address == UNDEF_ADDR {
                    return Ok(vec![]);
                }
                self.handle.read_at(*address, *size as usize)?
            }
            DataLayoutMessage::Compact { data } => data.clone(),
            DataLayoutMessage::ChunkedV3 {
                chunk_dims,
                b_tree_address,
            } => {
                // The layout's chunk_dims include the element size as
                // the trailing dimension. Strip it for chunk indexing.
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_btree_v1(
                    name,
                    real_chunk_dims,
                    *b_tree_address,
                    pipeline.as_ref(),
                )?
            }
            DataLayoutMessage::ChunkedV4 {
                chunk_dims,
                index_address,
                index_type,
                earray_params,
                ..
            } => {
                // The layout's chunk_dims include the element size as
                // the trailing dimension. Strip it for chunk indexing.
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_v4(
                    name,
                    real_chunk_dims,
                    *index_address,
                    *index_type,
                    earray_params.as_ref(),
                    pipeline.as_ref(),
                )?
            }
        };

        Ok(data)
    }

    /// Apply the post-filter datatype conversion (libhdf5's `H5T_convert`
    /// step) to a fully-decoded output buffer.
    ///
    /// For N-bit / reduced-precision `FixedPoint` datatypes the filter
    /// pipeline leaves the significant value occupying `bit_precision` bits
    /// at `bit_offset` within each element, zero-filled and not
    /// sign-extended. This rewrites every element so the value occupies the
    /// whole element at bit offset 0, sign-extended when signed. It is a
    /// no-op for ordinary full-width datatypes.
    fn apply_post_filter_conversion(buffer: &mut [u8], datatype: &DatatypeMessage) -> IoResult<()> {
        use crate::format::nbit_scaleoffset::{
            apply_datatype_conversion, datatype_needs_bit_conversion,
        };
        if datatype_needs_bit_conversion(datatype) {
            apply_datatype_conversion(buffer, datatype)?;
        }
        Ok(())
    }

    /// Re-read the superblock and dataset metadata for SWMR.
    ///
    /// Call this periodically to pick up new data written by a concurrent
    /// SWMR writer. The superblock is re-read to get the latest EOF, then
    /// the root group is re-scanned for updated dataset headers (which may
    /// contain updated dataspace dimensions and chunk index addresses).
    pub fn refresh(&mut self) -> IoResult<()> {
        // Re-read superblock to get latest EOF and root group address.
        let sb_buf = self.handle.read_at_most(0, 256)?;

        // Only v2/v3 superblocks support SWMR refresh
        let sb = SuperblockV2V3::decode(&sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Re-read root group object header, following continuation blocks.
        let root_header = Self::read_object_header_full(
            &mut self.handle,
            &ctx,
            sb.root_group_object_header_address,
        )?;

        // Re-scan datasets and group attributes from link messages.
        let (datasets, group_attributes) =
            Self::discover_datasets_from_links(&mut self.handle, &root_header, &ctx)?;

        self._eof = sb.end_of_file_address;
        self.ctx = ctx;
        self.datasets = datasets;
        self.group_attributes = group_attributes;

        Ok(())
    }

    /// Read chunked dataset data by walking the chunk index.
    fn read_chunked_v4(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        index_type: data_layout::ChunkIndexType,
        earray_params: Option<&data_layout::EarrayParams>,
        pipeline: Option<&FilterPipeline>,
    ) -> IoResult<Vec<u8>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let fill_value = info.fill_value.clone();

        match index_type {
            data_layout::ChunkIndexType::SingleChunk => {
                // Single chunk: the index_address IS the chunk address
                let total_size: u64 = dims.iter().product::<u64>() * element_size;
                if index_address == UNDEF_ADDR || total_size == 0 {
                    // An unallocated single chunk reads back entirely as the
                    // fill value (when one is defined); otherwise empty.
                    if let Some(ref fv) = fill_value {
                        if total_size > 0 {
                            return Ok(tiled_fill(total_size as usize, Some(fv)));
                        }
                    }
                    return Ok(vec![]);
                }
                let data = if let Some(pipeline) = pipeline {
                    let raw = self
                        .handle
                        .read_at_most(index_address, total_size as usize * 2)?;
                    filter::reverse_filters(pipeline, &raw)?
                } else {
                    self.handle.read_at(index_address, total_size as usize)?
                };
                Ok(data)
            }
            data_layout::ChunkIndexType::FixedArray => {
                self.read_chunked_fixed_array(name, chunk_dims, index_address, pipeline)
            }
            data_layout::ChunkIndexType::BTreeV2 => {
                self.read_chunked_btree_v2(name, chunk_dims, index_address, pipeline)
            }
            data_layout::ChunkIndexType::ExtensibleArray => {
                let params = earray_params.ok_or_else(|| {
                    crate::io::IoError::InvalidState("missing earray params".into())
                })?;

                if index_address == UNDEF_ADDR {
                    return Ok(vec![]);
                }

                // Total chunk count across all dimensions.
                let chunks_dim0: u64 = (0..dims.len())
                    .map(|d| {
                        if chunk_dims[d] > 0 {
                            dims[d].div_ceil(chunk_dims[d])
                        } else {
                            0
                        }
                    })
                    .product();

                let chunk_entries = self.collect_ea_chunk_entries(
                    index_address,
                    params,
                    &dims,
                    chunk_dims,
                    element_size,
                )?;

                let total_size: u64 = dims.iter().product::<u64>() * element_size;
                let mut output = tiled_fill(total_size as usize, fill_value.as_deref());

                let n_chunks = std::cmp::min(chunks_dim0 as usize, chunk_entries.len());

                // Chunks are placed N-dimensionally: the linear chunk index
                // maps (row-major) to chunk-grid coordinates, so sub-frame
                // chunks (a chunk smaller than a full frame) land correctly.
                let ndims = dims.len();
                let chunks_per_dim: Vec<u64> = (0..ndims)
                    .map(|d| {
                        if chunk_dims[d] > 0 {
                            dims[d].div_ceil(chunk_dims[d])
                        } else {
                            0
                        }
                    })
                    .collect();
                let chunk_coords = |mut i: u64| -> Vec<u64> {
                    let mut c = vec![0u64; ndims];
                    for d in (0..ndims).rev() {
                        if chunks_per_dim[d] > 0 {
                            c[d] = i % chunks_per_dim[d];
                            i /= chunks_per_dim[d];
                        }
                    }
                    c
                };

                if let Some(pl) = pipeline {
                    // Read all raw chunks first, then decompress (optionally in parallel)
                    let file_size = self.handle.file_size()?;
                    let mut raw_chunks: Vec<Option<Vec<u8>>> = Vec::with_capacity(n_chunks);
                    for &(addr, nbytes) in &chunk_entries[..n_chunks] {
                        if addr == UNDEF_ADDR
                            || nbytes == 0
                            || addr >= file_size
                            || nbytes > file_size
                        {
                            raw_chunks.push(None);
                        } else {
                            raw_chunks.push(Some(self.handle.read_at(addr, nbytes as usize)?));
                        }
                    }

                    let reverse = |raw: Option<Vec<u8>>| -> IoResult<Option<Vec<u8>>> {
                        match raw {
                            Some(r) => Ok(Some(filter::reverse_filters(pl, &r)?)),
                            None => Ok(None),
                        }
                    };
                    #[cfg(feature = "parallel")]
                    let decompressed: Vec<Option<Vec<u8>>> = {
                        use rayon::prelude::*;
                        raw_chunks
                            .into_par_iter()
                            .map(reverse)
                            .collect::<IoResult<Vec<_>>>()?
                    };
                    #[cfg(not(feature = "parallel"))]
                    let decompressed: Vec<Option<Vec<u8>>> = raw_chunks
                        .into_iter()
                        .map(reverse)
                        .collect::<IoResult<Vec<_>>>()?;

                    for (i, chunk_data) in decompressed.iter().enumerate() {
                        if let Some(data) = chunk_data {
                            let coords = chunk_coords(i as u64);
                            self.copy_chunk_to_output(
                                data,
                                &mut output,
                                &dims,
                                chunk_dims,
                                &coords,
                                element_size,
                            );
                        }
                    }
                } else {
                    for (i, &(addr, nbytes)) in chunk_entries[..n_chunks].iter().enumerate() {
                        if addr == UNDEF_ADDR {
                            continue;
                        }
                        let chunk_data = self.handle.read_at_most(addr, nbytes as usize)?;
                        let coords = chunk_coords(i as u64);
                        self.copy_chunk_to_output(
                            &chunk_data,
                            &mut output,
                            &dims,
                            chunk_dims,
                            &coords,
                            element_size,
                        );
                    }
                }

                Ok(output)
            }
            _ => Err(crate::io::IoError::InvalidState(format!(
                "unsupported chunk index type: {:?}",
                index_type
            ))),
        }
    }

    /// Read a dataset indexed by a fixed array.
    fn read_chunked_fixed_array(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        pipeline: Option<&FilterPipeline>,
    ) -> IoResult<Vec<u8>> {
        use crate::format::chunk_index::fixed_array::*;

        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let fill_value = info.fill_value.clone();
        let ndims = dims.len();

        if index_address == UNDEF_ADDR {
            return Ok(vec![]);
        }

        // Read FA header
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let fa_hdr = FixedArrayHeader::decode(&hdr_buf, &self.ctx)?;

        if fa_hdr.data_blk_addr == UNDEF_ADDR {
            return Ok(vec![]);
        }

        // The chunk shape (from the layout message) must match the
        // dataspace rank; otherwise the chunk-grid indexing panics.
        if chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "fixed-array dataset rank {} does not match chunk rank {}",
                ndims,
                chunk_dims.len()
            )));
        }

        let is_filtered = fa_hdr.client_id == FA_CLIENT_FILT_CHUNK;
        let sizeof_addr = self.ctx.sizeof_addr as usize;
        // chunk_size_len = element_size - sizeof_addr - filter_mask(4)
        let chunk_size_len = if is_filtered {
            (fa_hdr.element_size as usize)
                .checked_sub(sizeof_addr + 4)
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState(
                        "fixed array filtered element_size too small".into(),
                    )
                })?
        } else {
            0
        };
        // The compressed-size field is read into a u64; reject a width that
        // would overflow the read_size helper.
        if chunk_size_len > 8 {
            return Err(crate::io::IoError::InvalidState(format!(
                "fixed array filtered chunk-size width {chunk_size_len} exceeds 8 bytes"
            )));
        }

        // Compute chunk byte size
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

        // Collect per-chunk (address, compressed_size). compressed_size is the
        // exact on-disk byte count for filtered chunks, or chunk_bytes when
        // unfiltered.
        let num_elmts = fa_hdr.num_elmts as usize;
        let mut chunk_entries: Vec<(u64, u64)> = Vec::with_capacity(num_elmts);

        if fa_hdr.is_paged() {
            // Paged data block: prefix (with page-init bitmap) followed by pages.
            let npages = fa_hdr.npages();
            let dblk_page_nelmts = fa_hdr.dblk_page_nelmts();
            let prefix_len = 4 + 1 + 1 + sizeof_addr + (npages as usize).div_ceil(8) + 4;
            let prefix_buf = self.handle.read_at_most(fa_hdr.data_blk_addr, prefix_len)?;
            let prefix = FixedArrayPagedPrefix::decode(&prefix_buf, &self.ctx, npages)?;

            let elem_size = if is_filtered {
                sizeof_addr + chunk_size_len + 4
            } else {
                sizeof_addr
            };
            // All pages have the same on-disk stride; only the last page holds
            // fewer elements (libhdf5: dblk_page_size is constant).
            let page_stride = dblk_page_nelmts as usize * elem_size + 4;
            let pages_base = fa_hdr.data_blk_addr + prefix.prefix_size as u64;

            for p in 0..npages as usize {
                // Elements on this page (last page may be short).
                let page_nelmts = if p + 1 == npages as usize {
                    let rem = fa_hdr.num_elmts % dblk_page_nelmts;
                    if rem == 0 {
                        dblk_page_nelmts
                    } else {
                        rem
                    }
                } else {
                    dblk_page_nelmts
                } as usize;

                if !prefix.page_initialized(p) {
                    // Uninitialized page: all chunk entries are undefined.
                    chunk_entries.extend(std::iter::repeat_n((UNDEF_ADDR, 0u64), page_nelmts));
                    continue;
                }

                let page_addr = pages_base + (p as u64) * page_stride as u64;
                let page_size = page_nelmts * elem_size + 4;
                let page_buf = self.handle.read_at_most(page_addr, page_size)?;

                if is_filtered {
                    let elems =
                        decode_filtered_page(&page_buf, &self.ctx, page_nelmts, chunk_size_len)?;
                    for e in elems {
                        chunk_entries.push((e.address, e.chunk_size as u64));
                    }
                } else {
                    let addrs = decode_unfiltered_page(&page_buf, &self.ctx, page_nelmts)?;
                    for addr in addrs {
                        chunk_entries.push((addr, chunk_bytes));
                    }
                }
            }
        } else {
            // Non-paged data block: all elements live inline in the data block.
            let elem_size = if is_filtered {
                sizeof_addr + chunk_size_len + 4
            } else {
                sizeof_addr
            };
            let dblk_size = 4 + 1 + 1 + sizeof_addr + num_elmts * elem_size + 4;
            let dblk_buf = self.handle.read_at_most(fa_hdr.data_blk_addr, dblk_size)?;

            if is_filtered {
                let fa_dblk = FixedArrayDataBlock::decode_filtered(
                    &dblk_buf,
                    &self.ctx,
                    num_elmts,
                    chunk_size_len,
                )?;
                for e in &fa_dblk.filtered_elements {
                    chunk_entries.push((e.address, e.chunk_size as u64));
                }
            } else {
                let fa_dblk =
                    FixedArrayDataBlock::decode_unfiltered(&dblk_buf, &self.ctx, num_elmts)?;
                for &addr in &fa_dblk.elements {
                    chunk_entries.push((addr, chunk_bytes));
                }
            }
        }

        // Total output size
        let total_size: u64 = dims.iter().product::<u64>() * element_size;
        let mut output = tiled_fill(total_size as usize, fill_value.as_deref());

        // Compute number of chunks per dimension. A zero chunk dimension
        // from a malformed layout message would divide by zero.
        if chunk_dims.contains(&0) {
            return Err(crate::io::IoError::InvalidState(
                "fixed-array layout has a zero chunk dimension".into(),
            ));
        }
        let chunks_per_dim: Vec<u64> = (0..ndims)
            .map(|d| dims[d].div_ceil(chunk_dims[d]))
            .collect();

        // Read all chunk raw data sequentially (I/O must be serial)
        let n_chunks = chunk_entries.len();
        let mut raw_chunks: Vec<(usize, Option<Vec<u8>>)> = Vec::with_capacity(n_chunks);
        for (linear_idx, &(addr, comp_size)) in chunk_entries.iter().enumerate() {
            if addr == UNDEF_ADDR {
                raw_chunks.push((linear_idx, None));
            } else if pipeline.is_some() {
                // For filtered chunks the entry carries the exact compressed
                // size; fall back to a generous estimate otherwise.
                let read_len = if comp_size > 0 {
                    comp_size as usize
                } else {
                    chunk_bytes as usize * 2
                };
                raw_chunks.push((linear_idx, Some(self.handle.read_at_most(addr, read_len)?)));
            } else {
                raw_chunks.push((
                    linear_idx,
                    Some(self.handle.read_at(addr, chunk_bytes as usize)?),
                ));
            }
        }

        // Decompress if a filter pipeline is set. A filter failure is
        // propagated rather than swallowed into garbage output.
        let decompressed: Vec<(usize, Option<Vec<u8>>)> = if let Some(pl) = pipeline {
            let reverse =
                |(idx, raw): (usize, Option<Vec<u8>>)| -> IoResult<(usize, Option<Vec<u8>>)> {
                    match raw {
                        Some(r) => Ok((idx, Some(filter::reverse_filters(pl, &r)?))),
                        None => Ok((idx, None)),
                    }
                };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                raw_chunks
                    .into_par_iter()
                    .map(reverse)
                    .collect::<IoResult<Vec<_>>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                raw_chunks
                    .into_iter()
                    .map(reverse)
                    .collect::<IoResult<Vec<_>>>()?
            }
        } else {
            raw_chunks
        };

        // Place chunks into output
        for (linear_idx, chunk_data) in &decompressed {
            let Some(data) = chunk_data else { continue };
            let mut remaining = *linear_idx as u64;
            let mut coords = vec![0u64; ndims];
            for d in (0..ndims).rev() {
                coords[d] = remaining % chunks_per_dim[d];
                remaining /= chunks_per_dim[d];
            }
            self.copy_chunk_to_output(data, &mut output, &dims, chunk_dims, &coords, element_size);
        }

        Ok(output)
    }

    /// Read a dataset indexed by a B-tree v2.
    fn read_chunked_btree_v2(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        pipeline: Option<&FilterPipeline>,
    ) -> IoResult<Vec<u8>> {
        use crate::format::chunk_index::btree_v2::*;

        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let fill_value = info.fill_value.clone();
        let ndims = dims.len();

        if index_address == UNDEF_ADDR {
            return Ok(vec![]);
        }

        // Read BT2 header
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let bt2_hdr = Bt2Header::decode(&hdr_buf, &self.ctx)?;

        if bt2_hdr.root_node_addr == UNDEF_ADDR || bt2_hdr.total_num_records == 0 {
            return Ok(vec![]);
        }

        // Walk the B-tree to any depth, collecting every record's raw bytes
        // from the internal nodes and leaves.
        let geo = Bt2Geometry::new(
            bt2_hdr.node_size,
            bt2_hdr.record_size,
            bt2_hdr.depth,
            self.ctx.sizeof_addr,
        );
        let mut record_bytes: Vec<u8> = Vec::new();
        self.collect_bt2_records(
            bt2_hdr.root_node_addr,
            bt2_hdr.depth,
            bt2_hdr.num_records_in_root,
            bt2_hdr.record_size,
            bt2_hdr.node_size,
            &geo,
            &mut record_bytes,
        )?;
        let total_records = if bt2_hdr.record_size > 0 {
            record_bytes.len() / bt2_hdr.record_size as usize
        } else {
            0
        };

        // Decode records
        // Compute chunk byte size
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

        // Unify filtered and unfiltered records into (address, read_size,
        // scaled offsets). read_size is the compressed size for filtered
        // chunks, the full chunk size otherwise.
        let entries: Vec<(u64, usize, Vec<u64>)> = if bt2_hdr.record_type == BT2_TYPE_CHUNK_UNFILT {
            Bt2ChunkIndex::decode_unfiltered_records(
                &record_bytes,
                total_records,
                ndims,
                &self.ctx,
            )?
            .into_iter()
            .map(|r| (r.chunk_address, chunk_bytes as usize, r.scaled_offsets))
            .collect()
        } else {
            Bt2ChunkIndex::decode_filtered_records(
                &record_bytes,
                total_records,
                ndims,
                bt2_hdr.record_size,
                &self.ctx,
            )?
            .into_iter()
            .map(|r| (r.chunk_address, r.chunk_size as usize, r.scaled_offsets))
            .collect()
        };

        let total_size: u64 = dims.iter().product::<u64>() * element_size;
        let mut output = tiled_fill(total_size as usize, fill_value.as_deref());

        // Read each chunk's raw bytes.
        let mut raw_chunks: Vec<Option<(Vec<u8>, Vec<u64>)>> = Vec::with_capacity(entries.len());
        for (addr, read_size, scaled) in &entries {
            if *addr == UNDEF_ADDR || *read_size == 0 {
                raw_chunks.push(None);
            } else {
                let data = self.handle.read_at(*addr, *read_size)?;
                raw_chunks.push(Some((data, scaled.clone())));
            }
        }

        // Decompress filtered chunks (optionally in parallel). A filter
        // failure is propagated, not swallowed into garbage output.
        let placed: Vec<Option<(Vec<u8>, Vec<u64>)>> = if let Some(pl) = pipeline {
            let reverse =
                |c: Option<(Vec<u8>, Vec<u64>)>| -> IoResult<Option<(Vec<u8>, Vec<u64>)>> {
                    match c {
                        Some((r, s)) => Ok(Some((filter::reverse_filters(pl, &r)?, s))),
                        None => Ok(None),
                    }
                };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                raw_chunks
                    .into_par_iter()
                    .map(reverse)
                    .collect::<IoResult<Vec<_>>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                raw_chunks
                    .into_iter()
                    .map(reverse)
                    .collect::<IoResult<Vec<_>>>()?
            }
        } else {
            raw_chunks
        };

        // Place each chunk N-dimensionally by its scaled (chunk-grid) offsets.
        for chunk in placed.iter().flatten() {
            let (data, scaled) = chunk;
            self.copy_chunk_to_output(data, &mut output, &dims, chunk_dims, scaled, element_size);
        }

        Ok(output)
    }

    /// Recursively walk a v2 B-tree, appending every node's raw record bytes
    /// (internal nodes and leaves alike) to `out`.
    #[allow(clippy::too_many_arguments)]
    fn collect_bt2_records(
        &mut self,
        addr: u64,
        depth: u16,
        nrec: u16,
        record_size: u16,
        node_size: u32,
        geo: &crate::format::chunk_index::btree_v2::Bt2Geometry,
        out: &mut Vec<u8>,
    ) -> IoResult<()> {
        use crate::format::chunk_index::btree_v2::{Bt2InternalNode, Bt2LeafNode};

        let buf = self.handle.read_at_most(addr, node_size as usize)?;
        if depth == 0 {
            let leaf = Bt2LeafNode::decode(&buf, nrec, record_size)?;
            out.extend_from_slice(&leaf.record_data);
        } else {
            let node = Bt2InternalNode::decode(
                &buf,
                &self.ctx,
                depth,
                nrec,
                record_size,
                geo.max_nrec_size,
                geo.child_total_size(depth),
            )?;
            out.extend_from_slice(&node.record_data);
            // Collect (addr, nrec) up front so the node borrow is released
            // before recursing.
            let children: Vec<(u64, u16)> = node
                .child_addrs
                .iter()
                .zip(node.child_nrecords.iter())
                .map(|(&a, &n)| (a, n))
                .collect();
            for (child_addr, child_nrec) in children {
                self.collect_bt2_records(
                    child_addr,
                    depth - 1,
                    child_nrec,
                    record_size,
                    node_size,
                    geo,
                    out,
                )?;
            }
        }
        Ok(())
    }

    /// Read a chunked dataset indexed by a version-1 B-tree (layout
    /// message version 3, class 2 "chunked").
    ///
    /// `chunk_dims` excludes the trailing element-size dimension.
    fn read_chunked_btree_v1(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        b_tree_address: u64,
        pipeline: Option<&FilterPipeline>,
    ) -> IoResult<Vec<u8>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let fill_value = info.fill_value.clone();
        let ndims = dims.len();

        // The chunk shape must match the dataspace rank or the chunk-grid
        // indexing below panics.
        if chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "B-tree-v1 dataset rank {} does not match chunk rank {}",
                ndims,
                chunk_dims.len()
            )));
        }

        let total_size: u64 = dims.iter().product::<u64>() * element_size;
        let mut output = tiled_fill(total_size as usize, fill_value.as_deref());

        if b_tree_address == UNDEF_ADDR || total_size == 0 {
            return Ok(output);
        }

        // Walk the B-tree, collecting every leaf entry as
        // (element_offsets, chunk_address, chunk_size, filter_mask).
        // `chunk_dims.len()` is the chunk rank; the B-tree keys carry
        // rank + 1 offsets (the extra one is the element-size dimension).
        let mut entries: Vec<(Vec<u64>, u64, u32, u32)> = Vec::new();
        let file_size = self.handle.file_size()?;
        self.collect_btree_v1_chunks(b_tree_address, ndims, file_size, 0, &mut entries)?;

        // The uncompressed byte size of a full chunk.
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

        // Read every chunk's raw bytes.
        let mut raw_chunks: Vec<Option<(Vec<u8>, Vec<u64>)>> = Vec::with_capacity(entries.len());
        for (offsets, addr, chunk_size, _mask) in &entries {
            if *addr == UNDEF_ADDR
                || *chunk_size == 0
                || *addr >= file_size
                || *chunk_size as u64 > file_size
            {
                raw_chunks.push(None);
                continue;
            }
            // Convert element offsets to chunk-grid (scaled) coordinates.
            // The trailing element-size dimension offset is always 0 and
            // is dropped here.
            let mut scaled = Vec::with_capacity(ndims);
            for d in 0..ndims {
                let cd = chunk_dims[d];
                scaled.push(if cd > 0 { offsets[d] / cd } else { 0 });
            }
            let data = self.handle.read_at(*addr, *chunk_size as usize)?;
            raw_chunks.push(Some((data, scaled)));
        }

        // Decompress filtered chunks (optionally in parallel). A filter
        // failure is propagated, not swallowed into garbage output.
        let placed: Vec<Option<(Vec<u8>, Vec<u64>)>> = if let Some(pl) = pipeline {
            let reverse =
                |c: Option<(Vec<u8>, Vec<u64>)>| -> IoResult<Option<(Vec<u8>, Vec<u64>)>> {
                    match c {
                        Some((r, s)) => Ok(Some((filter::reverse_filters(pl, &r)?, s))),
                        None => Ok(None),
                    }
                };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                raw_chunks
                    .into_par_iter()
                    .map(reverse)
                    .collect::<IoResult<Vec<_>>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                raw_chunks
                    .into_iter()
                    .map(reverse)
                    .collect::<IoResult<Vec<_>>>()?
            }
        } else {
            raw_chunks
        };

        // Place each chunk N-dimensionally by its scaled offsets.
        for chunk in placed.iter().flatten() {
            let (data, scaled) = chunk;
            self.copy_chunk_to_output(data, &mut output, &dims, chunk_dims, scaled, element_size);
        }

        // libhdf5 stores raw byte sizes; verify the uncompressed chunk
        // size is consistent for unfiltered datasets so a corrupt index
        // surfaces instead of silently producing garbage.
        if pipeline.is_none() {
            for (_, addr, chunk_size, _) in &entries {
                if *addr != UNDEF_ADDR && *chunk_size as u64 != chunk_bytes && *chunk_size != 0 {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "chunk B-tree v1: unfiltered chunk size {} != expected {}",
                        chunk_size, chunk_bytes
                    )));
                }
            }
        }

        Ok(output)
    }

    /// Recursively walk a version-1 raw-data-chunk B-tree, collecting every
    /// leaf entry as `(element_offsets, chunk_address, chunk_size,
    /// filter_mask)`.
    ///
    /// `rank` is the chunk rank excluding the trailing element-size
    /// dimension. Recursion is bounded by the node level read from disk:
    /// each recursive step descends to a strictly lower level, and the
    /// `depth` counter caps the descent at the 1-byte level field's range.
    fn collect_btree_v1_chunks(
        &mut self,
        addr: u64,
        rank: usize,
        file_size: u64,
        depth: u32,
        out: &mut Vec<(Vec<u64>, u64, u32, u32)>,
    ) -> IoResult<()> {
        // A v1 B-tree node level fits in one byte, so the tree can be at
        // most 256 levels deep; this also stops cyclic/corrupt indices.
        if depth > 256 {
            return Err(crate::io::IoError::InvalidState(
                "chunk B-tree v1 exceeds maximum depth".into(),
            ));
        }
        if addr == UNDEF_ADDR || addr >= file_size {
            return Ok(());
        }

        // A node is the header (8 + 2*sizeof_addr) plus interleaved
        // keys/children. Read a generous slice; ChunkBTreeV1Node::decode
        // validates the exact length it needs.
        let sa = self.ctx.sizeof_addr as usize;
        let key_size = 4 + 4 + (rank + 1) * 8;
        // u16 entries_used: at most 65535 children.
        let max_node = 8 + sa * 2 + 65536 * (key_size + sa);
        let buf = self.handle.read_at_most(addr, max_node)?;
        let node = ChunkBTreeV1Node::decode(&buf, sa, rank)?;

        if node.level == 0 {
            // Leaf node: each child points at chunk data.
            for (i, &child_addr) in node.children.iter().enumerate() {
                let key = &node.keys[i];
                out.push((
                    key.offsets[..rank].to_vec(),
                    child_addr,
                    key.chunk_size,
                    key.filter_mask,
                ));
            }
        } else {
            // Internal node: each child points at a sub-TREE node one
            // level below. `node.level` is read from disk and decreases on
            // every descent, so it also bounds the recursion.
            let children: Vec<u64> = node.children.clone();
            for child_addr in children {
                if child_addr == UNDEF_ADDR || child_addr >= file_size {
                    continue;
                }
                self.collect_btree_v1_chunks(child_addr, rank, file_size, depth + 1, out)?;
            }
        }
        Ok(())
    }

    /// Copy chunk data into the correct position in a multi-dimensional output buffer.
    fn copy_chunk_to_output(
        &self,
        chunk_data: &[u8],
        output: &mut [u8],
        dims: &[u64],
        chunk_dims: &[u64],
        chunk_coords: &[u64],
        element_size: u64,
    ) {
        let ndims = dims.len();
        if ndims == 0 {
            return;
        }

        // For 1D case, direct memcpy
        if ndims == 1 {
            // A corrupt index could place the chunk past the dataset extent;
            // saturating math keeps that from underflowing, and a zero span
            // simply copies nothing.
            let chunk_start = chunk_coords[0].saturating_mul(chunk_dims[0]);
            let start = chunk_start.saturating_mul(element_size);
            let actual_elems = std::cmp::min(chunk_dims[0], dims[0].saturating_sub(chunk_start));
            let copy_bytes = (actual_elems * element_size) as usize;
            let start = start as usize;
            if start + copy_bytes <= output.len() && copy_bytes <= chunk_data.len() {
                output[start..start + copy_bytes].copy_from_slice(&chunk_data[..copy_bytes]);
            }
            return;
        }

        // For multi-dimensional: compute row-major layout
        // The chunk occupies a sub-region of the output array.
        // We iterate over all elements in the chunk and compute their position.
        let chunk_elems: u64 = chunk_dims.iter().product();
        let mut chunk_coord_iter = vec![0u64; ndims];

        for elem_idx in 0..chunk_elems {
            // Compute multi-dimensional index within the chunk
            let mut remaining = elem_idx;
            for d in (0..ndims).rev() {
                chunk_coord_iter[d] = remaining % chunk_dims[d];
                remaining /= chunk_dims[d];
            }

            // Compute global position
            let mut valid = true;
            let mut global_linear = 0u64;
            let mut stride = 1u64;
            for d in (0..ndims).rev() {
                let global_d = chunk_coords[d] * chunk_dims[d] + chunk_coord_iter[d];
                if global_d >= dims[d] {
                    valid = false;
                    break;
                }
                global_linear += global_d * stride;
                stride *= dims[d];
            }

            if !valid {
                continue;
            }

            let src_offset = (elem_idx * element_size) as usize;
            let dst_offset = (global_linear * element_size) as usize;
            let es = element_size as usize;
            if src_offset + es <= chunk_data.len() && dst_offset + es <= output.len() {
                output[dst_offset..dst_offset + es]
                    .copy_from_slice(&chunk_data[src_offset..src_offset + es]);
            }
        }
    }

    /// Read variable-length string data from a dataset.
    ///
    /// h5py stores vlen strings as global heap references. Each element
    /// in the raw data is a (collection_address, object_index) pair that
    /// points to a string blob in a global heap collection.
    ///
    /// Returns a Vec<String> with one entry per element.
    pub fn read_vlen_strings(&mut self, name: &str) -> IoResult<Vec<String>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let layout = info.layout.clone();
        let total_elements: u64 = dims.iter().product();

        let raw = match &layout {
            DataLayoutMessage::Contiguous { address, size } => {
                if *address == UNDEF_ADDR {
                    return Ok(vec![]);
                }
                self.handle.read_at(*address, *size as usize)?
            }
            DataLayoutMessage::Compact { data } => data.clone(),
            _ => {
                // For chunked, read the full dataset first
                self.read_dataset_raw(name)?
            }
        };

        let ref_size = vlen_reference_size(&self.ctx);
        let mut strings = Vec::with_capacity(total_elements as usize);

        // Cache global heap collections to avoid re-reading.
        // Store as (collection, index→offset lookup) for O(1) object access.
        let mut heap_cache: std::collections::HashMap<
            u64,
            (GlobalHeapCollection, std::collections::HashMap<u16, usize>),
        > = std::collections::HashMap::new();

        for i in 0..total_elements as usize {
            let offset = i * ref_size;
            if offset + ref_size > raw.len() {
                break;
            }

            let (_seq_len, collection_addr, obj_index) =
                decode_vlen_reference(&raw[offset..], &self.ctx)?;

            if collection_addr == UNDEF_ADDR || collection_addr == 0 {
                strings.push(String::new());
                continue;
            }

            // Read or get cached global heap collection
            #[allow(clippy::map_entry)]
            if !heap_cache.contains_key(&collection_addr) {
                let ss = self.ctx.sizeof_size as usize;
                let header_len = 4 + 1 + 3 + ss;
                let header_buf = self.handle.read_at_most(collection_addr, header_len)?;
                if header_buf.len() < header_len || &header_buf[0..4] != b"GCOL" {
                    strings.push(String::new());
                    continue;
                }
                let collection_size = read_uint(&header_buf[8..], ss) as usize;

                if collection_size == 0 || collection_size > 64 * 1024 * 1024 {
                    strings.push(String::new());
                    continue;
                }

                let heap_buf = self.handle.read_at(collection_addr, collection_size)?;
                let (coll, _) = GlobalHeapCollection::decode(&heap_buf, &self.ctx)?;
                let lookup: std::collections::HashMap<u16, usize> = coll
                    .objects
                    .iter()
                    .enumerate()
                    .map(|(i, o)| (o.index, i))
                    .collect();
                heap_cache.insert(collection_addr, (coll, lookup));
            }

            let (coll, lookup) = &heap_cache[&collection_addr];
            if let Some(&idx) = lookup.get(&(obj_index as u16)) {
                let s = String::from_utf8_lossy(&coll.objects[idx].data).to_string();
                strings.push(s);
            } else {
                strings.push(String::new());
            }
        }

        Ok(strings)
    }

    /// Collect chunk (address, size) entries from an EA index.
    /// Returns a vector indexed by chunk linear index.
    fn collect_ea_chunk_entries(
        &mut self,
        index_address: u64,
        params: &data_layout::EarrayParams,
        dims: &[u64],
        chunk_dims: &[u64],
        element_size: u64,
    ) -> IoResult<Vec<(u64, u64)>> {
        use crate::format::chunk_index::extensible_array::{self as ea, *};

        if index_address == UNDEF_ADDR {
            return Ok(vec![]);
        }
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let ea_hdr = ExtensibleArrayHeader::decode(&hdr_buf, &self.ctx)?;
        if ea_hdr.idx_blk_addr == UNDEF_ADDR {
            return Ok(vec![]);
        }

        // Total chunk count across every dimension (sub-frame chunks make
        // this larger than the dim-0 chunk count alone).
        let chunks_dim0: usize = (0..dims.len())
            .map(|d| {
                if chunk_dims[d] > 0 {
                    dims[d].div_ceil(chunk_dims[d]) as usize
                } else {
                    0
                }
            })
            .product();
        let geo = EaGeometry::new(
            params.idx_blk_elmts,
            params.data_blk_min_elmts,
            params.sup_blk_min_data_ptrs,
            params.max_nelmts_bits,
            params.max_dblk_page_nelmts_bits,
        )?;
        let chunk_bytes = chunk_dims.iter().product::<u64>() * element_size;
        let is_filtered = ea_hdr.class_id == ea::EA_CLS_FILT_CHUNK;
        let chunk_size_len = if is_filtered {
            ea_hdr.raw_elmt_size - self.ctx.sizeof_addr - 4
        } else {
            0
        };
        let max_nelmts_bits = params.max_nelmts_bits;
        let mut entries: Vec<(u64, u64)> = Vec::new();

        // Read the index block: direct elements + the data-block / super-block
        // address arrays (the address arrays are filter-agnostic).
        let (dblk_addrs, sblk_addrs): (Vec<u64>, Vec<u64>) = if is_filtered {
            let buf = self.handle.read_at_most(ea_hdr.idx_blk_addr, 65536)?;
            let fiblk = ea::FilteredIndexBlock::decode(
                &buf,
                &self.ctx,
                params.idx_blk_elmts as usize,
                geo.ndblk_addrs,
                geo.nsblk_addrs,
                chunk_size_len,
            )?;
            for e in &fiblk.elements {
                entries.push((e.addr, e.nbytes));
            }
            (fiblk.dblk_addrs, fiblk.sblk_addrs)
        } else {
            let buf = self.handle.read_at_most(ea_hdr.idx_blk_addr, 65536)?;
            let iblk = ExtensibleArrayIndexBlock::decode(
                &buf,
                &self.ctx,
                params.idx_blk_elmts as usize,
                geo.ndblk_addrs,
                geo.nsblk_addrs,
            )?;
            for &addr in &iblk.elements {
                entries.push((addr, chunk_bytes));
            }
            (iblk.dblk_addrs, iblk.sblk_addrs)
        };

        // Walk super blocks in order, collecting each data block's entries.
        let sa = self.ctx.sizeof_addr as usize;
        let raw_elmt_size = if is_filtered {
            ea::FilteredChunkEntry::raw_size(self.ctx.sizeof_addr, chunk_size_len) as usize
        } else {
            sa
        };
        'outer: for (u, s) in geo.sblk.iter().enumerate() {
            if entries.len() >= chunks_dim0 {
                break;
            }
            let dblk_nelmts = s.dblk_nelmts as usize;
            let paged = geo.is_sblk_paged(u);

            // This super block's data-block addresses, plus its page-init
            // bitmap region (empty unless the super block is paged).
            let (this_dblk_addrs, page_init): (Vec<u64>, Vec<u8>) = if u < geo.iblock_nsblks {
                let start = s.start_dblk as usize;
                (
                    (0..s.ndblks as usize)
                        .map(|d| dblk_addrs.get(start + d).copied().unwrap_or(UNDEF_ADDR))
                        .collect(),
                    Vec::new(),
                )
            } else {
                let sblk_addr = sblk_addrs
                    .get(u - geo.iblock_nsblks)
                    .copied()
                    .unwrap_or(UNDEF_ADDR);
                if sblk_addr == UNDEF_ADDR {
                    (vec![UNDEF_ADDR; s.ndblks as usize], Vec::new())
                } else {
                    let page_init_total = if paged {
                        s.ndblks as usize * geo.dblk_page_init_size(u)
                    } else {
                        0
                    };
                    // Size the read from the super block's geometry rather
                    // than a fixed cap: signature+version+class+header_addr
                    // + block_offset(<=8) + page-init bitmaps
                    // + ndblks data-block addresses + checksum.
                    let sblk_size =
                        4 + 1 + 1 + sa + 8 + page_init_total + s.ndblks as usize * sa + 4;
                    let buf = self.handle.read_at_most(sblk_addr, sblk_size)?;
                    let sb = ExtensibleArraySuperBlock::decode(
                        &buf,
                        &self.ctx,
                        max_nelmts_bits,
                        s.ndblks as usize,
                        page_init_total,
                    )?;
                    (sb.dblk_addrs, sb.page_init)
                }
            };

            let npages = geo.npages(u) as usize;
            let page_size = geo.dblk_page_size(raw_elmt_size);
            let prefix = geo.dblk_prefix_size(self.ctx.sizeof_addr, max_nelmts_bits);

            for (d, &dblk_addr) in this_dblk_addrs.iter().enumerate() {
                if dblk_addr == UNDEF_ADDR {
                    entries.extend(std::iter::repeat_n((UNDEF_ADDR, 0), dblk_nelmts));
                } else if paged {
                    // Paged data block: only a prefix lives at `dblk_addr`;
                    // the elements live in `npages` page structures that
                    // follow it on disk. The super block's page-init bitmap
                    // is one flat MSB-first bitmap (H5VM bit ops) indexed by
                    // `dblk_idx * npages + page_idx` (H5EA.c), not a series
                    // of per-data-block sub-bitmaps.
                    for p in 0..npages {
                        let bit = d * npages + p;
                        let initialized = page_init[bit / 8] & (0x80u8 >> (bit % 8)) != 0;
                        if !initialized {
                            entries.extend(std::iter::repeat_n(
                                (UNDEF_ADDR, 0),
                                geo.dblk_page_nelmts as usize,
                            ));
                            continue;
                        }
                        let page_addr = dblk_addr + prefix as u64 + (p as u64) * page_size as u64;
                        let page = self.handle.read_at(page_addr, page_size)?;
                        for k in 0..geo.dblk_page_nelmts as usize {
                            let off = k * raw_elmt_size;
                            if is_filtered {
                                let e = ea::FilteredChunkEntry::decode(
                                    &page[off..],
                                    sa,
                                    chunk_size_len as usize,
                                );
                                entries.push((e.addr, e.nbytes));
                            } else {
                                entries.push((read_addr(&page[off..], sa), chunk_bytes));
                            }
                        }
                    }
                } else if is_filtered {
                    let dblk_size = prefix + dblk_nelmts * raw_elmt_size;
                    let buf = self.handle.read_at_most(dblk_addr, dblk_size)?;
                    let dblk = ea::FilteredDataBlock::decode(
                        &buf,
                        &self.ctx,
                        max_nelmts_bits,
                        dblk_nelmts,
                        chunk_size_len,
                    )?;
                    for e in &dblk.elements {
                        entries.push((e.addr, e.nbytes));
                    }
                } else {
                    let dblk_size = prefix + dblk_nelmts * raw_elmt_size;
                    let buf = self.handle.read_at_most(dblk_addr, dblk_size)?;
                    let dblk = ExtensibleArrayDataBlock::decode(
                        &buf,
                        &self.ctx,
                        max_nelmts_bits,
                        dblk_nelmts,
                    )?;
                    for &addr in &dblk.elements {
                        entries.push((addr, chunk_bytes));
                    }
                }
                if entries.len() >= chunks_dim0 {
                    break 'outer;
                }
            }
        }
        Ok(entries)
    }

    /// Read a slice (hyperslab) of a contiguous dataset.
    ///
    /// `starts` and `counts` define the N-dimensional selection:
    /// starts[d] is the first index along dim d, counts[d] is how many.
    /// Returns the selected data in row-major order.
    pub fn read_slice(&mut self, name: &str, starts: &[u64], counts: &[u64]) -> IoResult<Vec<u8>> {
        let datatype = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?
            .datatype
            .clone();
        let mut data = self.read_slice_inner(name, starts, counts)?;
        Self::apply_post_filter_conversion(&mut data, &datatype)?;
        Ok(data)
    }

    /// Slice read producing the raw filter-pipeline output, before the
    /// post-filter datatype conversion. `read_slice` wraps this and applies
    /// the conversion.
    fn read_slice_inner(
        &mut self,
        name: &str,
        starts: &[u64],
        counts: &[u64],
    ) -> IoResult<Vec<u8>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let layout = info.layout.clone();
        let pipeline = info.filter_pipeline.clone();
        let fill_value = info.fill_value.clone();
        let ndims = dims.len();

        if starts.len() != ndims || counts.len() != ndims {
            return Err(crate::io::IoError::InvalidState(
                "starts/counts length must match dataset rank".into(),
            ));
        }
        if ndims == 0 {
            return Err(crate::io::IoError::InvalidState(
                "read_slice does not support scalar datasets; use read_dataset_raw".into(),
            ));
        }
        for d in 0..ndims {
            if starts[d] + counts[d] > dims[d] {
                return Err(crate::io::IoError::InvalidState(format!(
                    "slice out of bounds: dim {} start {} + count {} > {}",
                    d, starts[d], counts[d], dims[d]
                )));
            }
        }

        let out_elems: u64 = counts.iter().product();
        let out_bytes = (out_elems * element_size) as usize;

        match &layout {
            DataLayoutMessage::Contiguous { address, .. } => {
                if *address == UNDEF_ADDR {
                    return Ok(tiled_fill(out_bytes, fill_value.as_deref()));
                }

                let strides = compute_strides(&dims, element_size);

                // For 1D, simple contiguous read
                if ndims == 1 {
                    let offset = *address + starts[0] * element_size;
                    return self
                        .handle
                        .read_at(offset, (counts[0] * element_size) as usize)
                        .map_err(Into::into);
                }

                // Multi-dimensional: read row by row along the last dimension
                let mut output = vec![0u8; out_bytes];
                let row_bytes = (counts[ndims - 1] * element_size) as usize;

                // Iterate over all rows in the slice
                let mut coords = vec![0u64; ndims - 1];
                let n_rows: u64 = counts[..ndims - 1].iter().product();

                for row in 0..n_rows {
                    // Compute file offset
                    let mut file_offset = *address + starts[ndims - 1] * element_size;
                    for d in 0..ndims - 1 {
                        file_offset += (starts[d] + coords[d]) * strides[d];
                    }

                    let out_offset = row as usize * row_bytes;
                    let data = self.handle.read_at(file_offset, row_bytes)?;
                    output[out_offset..out_offset + row_bytes].copy_from_slice(&data);

                    // Increment coords (carry)
                    for d in (0..ndims - 1).rev() {
                        coords[d] += 1;
                        if coords[d] < counts[d] {
                            break;
                        }
                        coords[d] = 0;
                    }
                }

                Ok(output)
            }
            DataLayoutMessage::Compact { data } => {
                // Same logic but from in-memory data
                let strides = compute_strides(&dims, element_size);

                let mut output = vec![0u8; out_bytes];
                let row_bytes = (counts[ndims - 1] * element_size) as usize;
                let n_rows: u64 = if ndims > 1 {
                    counts[..ndims - 1].iter().product()
                } else {
                    1
                };

                let mut coords = vec![0u64; ndims.saturating_sub(1)];
                for row in 0..n_rows {
                    let mut src_offset = (starts[ndims - 1] * element_size) as usize;
                    for d in 0..ndims.saturating_sub(1) {
                        src_offset += ((starts[d] + coords[d]) * strides[d]) as usize;
                    }
                    let out_offset = row as usize * row_bytes;
                    output[out_offset..out_offset + row_bytes]
                        .copy_from_slice(&data[src_offset..src_offset + row_bytes]);

                    for d in (0..ndims.saturating_sub(1)).rev() {
                        coords[d] += 1;
                        if coords[d] < counts[d] {
                            break;
                        }
                        coords[d] = 0;
                    }
                }
                Ok(output)
            }
            DataLayoutMessage::ChunkedV3 { .. } => {
                // Read the full dataset via the v1 B-tree path, then
                // extract the requested slice. Use the unconverted read so
                // the post-filter datatype conversion runs exactly once, in
                // the `read_slice` wrapper.
                let full = self.read_dataset_raw_unconverted(name)?;
                let mut output = vec![0u8; out_bytes];
                let src_strides = compute_strides(&dims, element_size);
                let row_bytes = (counts[ndims - 1] * element_size) as usize;
                let n_rows: u64 = if ndims > 1 {
                    counts[..ndims - 1].iter().product()
                } else {
                    1
                };
                if ndims == 1 {
                    let src_off = (starts[0] * element_size) as usize;
                    output[..row_bytes].copy_from_slice(&full[src_off..src_off + row_bytes]);
                } else {
                    let out_strides = compute_strides(counts, element_size);
                    let mut coords = vec![0u64; ndims - 1];
                    for _row in 0..n_rows {
                        let mut src_off = (starts[ndims - 1] * element_size) as usize;
                        let mut out_off = 0usize;
                        for d in 0..ndims - 1 {
                            src_off += ((starts[d] + coords[d]) * src_strides[d]) as usize;
                            out_off += (coords[d] * out_strides[d]) as usize;
                        }
                        output[out_off..out_off + row_bytes]
                            .copy_from_slice(&full[src_off..src_off + row_bytes]);
                        for d in (0..ndims - 1).rev() {
                            coords[d] += 1;
                            if coords[d] < counts[d] {
                                break;
                            }
                            coords[d] = 0;
                        }
                    }
                }
                Ok(output)
            }
            DataLayoutMessage::ChunkedV4 {
                chunk_dims: layout_chunk_dims,
                index_address,
                index_type,
                earray_params,
                ..
            } => {
                let real_chunk_dims = &layout_chunk_dims[..layout_chunk_dims.len() - 1];
                let fp = pipeline.clone();

                // Optimized path for 1D-chunked streaming (chunk_dim0 == 1, EA index):
                // Read only the chunks that overlap [starts[0]..starts[0]+counts[0]).
                let can_optimize = ndims >= 2
                    && real_chunk_dims[0] == 1
                    && *index_type == data_layout::ChunkIndexType::ExtensibleArray;

                if can_optimize {
                    let all_entries = self.collect_ea_chunk_entries(
                        *index_address,
                        earray_params.as_ref().unwrap(),
                        &dims,
                        real_chunk_dims,
                        element_size,
                    )?;
                    let mut output = tiled_fill(out_bytes, fill_value.as_deref());
                    let out_strides = compute_strides(counts, element_size);
                    let chunk_inner_dims = &real_chunk_dims[1..];
                    let chunk_strides = compute_strides(chunk_inner_dims, element_size);
                    let inner_starts = &starts[1..];
                    let inner_counts = &counts[1..];
                    let inner_ndims = inner_starts.len();
                    let row_bytes = (inner_counts[inner_ndims - 1] * element_size) as usize;
                    let n_inner_rows: u64 = if inner_ndims > 1 {
                        inner_counts[..inner_ndims - 1].iter().product()
                    } else {
                        1
                    };

                    for fi in 0..counts[0] {
                        let gi = starts[0] + fi;
                        if (gi as usize) >= all_entries.len() {
                            break;
                        }
                        let (addr, nbytes) = all_entries[gi as usize];
                        if addr == UNDEF_ADDR {
                            continue;
                        }

                        let chunk_data = if let Some(ref pl) = fp {
                            let raw = self.handle.read_at(addr, nbytes as usize)?;
                            filter::reverse_filters(pl, &raw)?
                        } else {
                            self.handle.read_at(addr, nbytes as usize)?
                        };

                        let mut ic = vec![0u64; inner_ndims.saturating_sub(1)];
                        for _irow in 0..n_inner_rows {
                            let mut src_off =
                                (inner_starts[inner_ndims - 1] * element_size) as usize;
                            let mut dst_off = (fi * out_strides[0]) as usize;
                            for d in 0..inner_ndims.saturating_sub(1) {
                                src_off += ((inner_starts[d] + ic[d]) * chunk_strides[d]) as usize;
                                dst_off += (ic[d] * out_strides[d + 1]) as usize;
                            }
                            if src_off + row_bytes <= chunk_data.len()
                                && dst_off + row_bytes <= output.len()
                            {
                                output[dst_off..dst_off + row_bytes]
                                    .copy_from_slice(&chunk_data[src_off..src_off + row_bytes]);
                            }
                            for d in (0..inner_ndims.saturating_sub(1)).rev() {
                                ic[d] += 1;
                                if ic[d] < inner_counts[d] {
                                    break;
                                }
                                ic[d] = 0;
                            }
                        }
                    }
                    Ok(output)
                } else {
                    // Fallback: read full dataset and extract slice. Use the
                    // unconverted read so the post-filter datatype conversion
                    // is applied exactly once, by the `read_slice` wrapper.
                    let full = self.read_dataset_raw_unconverted(name)?;
                    let mut output = vec![0u8; out_bytes];
                    let src_strides = compute_strides(&dims, element_size);
                    let row_bytes = (counts[ndims - 1] * element_size) as usize;
                    let n_rows: u64 = if ndims > 1 {
                        counts[..ndims - 1].iter().product()
                    } else {
                        1
                    };
                    if ndims == 1 {
                        let src_off = (starts[0] * element_size) as usize;
                        output[..row_bytes].copy_from_slice(&full[src_off..src_off + row_bytes]);
                    } else {
                        let out_strides = compute_strides(counts, element_size);
                        let mut coords = vec![0u64; ndims - 1];
                        for _row in 0..n_rows {
                            let mut src_off = (starts[ndims - 1] * element_size) as usize;
                            let mut out_off = 0usize;
                            for d in 0..ndims - 1 {
                                src_off += ((starts[d] + coords[d]) * src_strides[d]) as usize;
                                out_off += (coords[d] * out_strides[d]) as usize;
                            }
                            output[out_off..out_off + row_bytes]
                                .copy_from_slice(&full[src_off..src_off + row_bytes]);
                            for d in (0..ndims - 1).rev() {
                                coords[d] += 1;
                                if coords[d] < counts[d] {
                                    break;
                                }
                                coords[d] = 0;
                            }
                        }
                    }
                    Ok(output)
                }
            }
        }
    }
}

/// Compute row-major strides for an N-dimensional array.
fn compute_strides(dims: &[u64], element_size: u64) -> Vec<u64> {
    let ndims = dims.len();
    if ndims == 0 {
        return vec![];
    }
    let mut strides = vec![0u64; ndims];
    strides[ndims - 1] = element_size;
    for d in (0..ndims - 1).rev() {
        strides[d] = strides[d + 1] * dims[d + 1];
    }
    strides
}

/// Adapts a `FileHandle` to the `BlockReader` trait used by the fractal-heap
/// walker, so heap blocks can be fetched from the open file.
struct HandleBlockReader<'a> {
    handle: &'a mut FileHandle,
}

impl BlockReader for HandleBlockReader<'_> {
    fn read_block(&mut self, offset: u64, len: usize) -> crate::format::FormatResult<Vec<u8>> {
        self.handle.read_at(offset, len).map_err(|e| {
            crate::format::FormatError::InvalidData(format!(
                "fractal heap block read failed at {:#x}: {}",
                offset, e
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Per-call unique temp path. PID + atomic counter avoids
    /// path collisions across concurrent cargo invocations and
    /// kernel-side flock release races.
    fn temp_path(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rust_hdf5_reader_test_{}_{}_{}.h5",
            name,
            std::process::id(),
            n
        ))
    }

    /// Helper: write a little-endian u64 truncated to `n` bytes.
    fn write_le(buf: &mut Vec<u8>, value: u64, n: usize) {
        buf.extend_from_slice(&value.to_le_bytes()[..n]);
    }

    /// Build a minimal v0 HDF5 file in memory with one dataset containing
    /// `dataset_data`. Returns the complete file bytes.
    ///
    /// The file structure is:
    /// - Superblock v0 with root group STE
    /// - Root group object header (v1) with symbol table message
    /// - Local heap (header + data) with dataset name
    /// - B-tree v1 (group, leaf) pointing to one SNOD
    /// - SNOD with one entry for the dataset
    /// - Dataset object header (v1) with dataspace, datatype, layout messages
    /// - Raw dataset data (contiguous)
    fn build_v0_file(dataset_name: &str, dims: &[u64], data: &[u8]) -> Vec<u8> {
        let sa: usize = 8; // sizeof_addr
        let ss: usize = 8; // sizeof_size
        let ndims = dims.len();
        let element_size = data.len() as u64 / dims.iter().product::<u64>();

        // We'll lay out the file regions in order, computing offsets as we go.
        let mut file = Vec::new();

        // ---- Plan layout offsets ----
        // We need to know the addresses before writing, so let's compute them.
        // Superblock: starts at 0
        let sb_size = 8 + 8 + 4 + 4 * sa + (ss + sa + 4 + 4 + 16); // sig + header + flags + 4 addrs + STE
                                                                   // Pad to 8-byte alignment
        let sb_size_aligned = (sb_size + 7) & !7;

        // Root group object header (v1): after superblock
        let root_ohdr_addr = sb_size_aligned as u64;
        // The root ohdr contains a symbol table message (type 0x11):
        //   btree_addr(8) + heap_addr(8) = 16 bytes
        // v1 message wire format: type(2) + size(2) + flags(1) + reserved(3) + data
        let stab_msg_data_size = 2 * sa; // btree + heap addr
        let stab_msg_wire = 8 + stab_msg_data_size;
        let stab_msg_wire_aligned = (stab_msg_wire + 7) & !7;
        let root_ohdr_data_size = stab_msg_wire_aligned;
        let root_ohdr_total = 16 + root_ohdr_data_size; // v1 16-byte prefix + messages
        let root_ohdr_total_aligned = (root_ohdr_total + 7) & !7;

        // Local heap header: after root ohdr
        let heap_hdr_addr = root_ohdr_addr + root_ohdr_total_aligned as u64;
        let heap_hdr_size = 4 + 1 + 3 + ss + ss + sa;
        let heap_hdr_size_aligned = (heap_hdr_size + 7) & !7;

        // Local heap data: after heap header
        let heap_data_addr = heap_hdr_addr + heap_hdr_size_aligned as u64;
        // Data: empty string at offset 0 (for root), then dataset_name at offset 1
        let name_bytes = dataset_name.as_bytes();
        let heap_data_content_size = 1 + name_bytes.len() + 1; // \0 + name + \0
        let heap_data_size = (heap_data_content_size + 7) & !7;

        // B-tree v1 node: after heap data
        let btree_addr = heap_data_addr + heap_data_size as u64;
        // B-tree header: TREE(4) + type(1) + level(1) + entries_used(2) + left(sa) + right(sa)
        // Plus interleaved keys/children: key[0](ss), child[0](sa), key[1](ss)
        let btree_size = 4 + 1 + 1 + 2 + 2 * sa + 2 * ss + sa;
        let btree_size_aligned = (btree_size + 7) & !7;

        // SNOD: after B-tree
        let snod_addr = btree_addr + btree_size_aligned as u64;
        // SNOD header: SNOD(4) + version(1) + reserved(1) + num_symbols(2)
        // + 1 entry: name_offset(ss) + obj_header_addr(sa) + cache_type(4) + reserved(4) + scratch(16)
        let entry_size = ss + sa + 4 + 4 + 16;
        let snod_size = 8 + entry_size;
        let snod_size_aligned = (snod_size + 7) & !7;

        // Dataset object header (v1): after SNOD
        let ds_ohdr_addr = snod_addr + snod_size_aligned as u64;
        // Messages: dataspace(0x01), datatype(0x03), data_layout(0x08)

        // Dataspace v1: version(1) + ndims(1) + flags(1) + reserved(1) + reserved(4) + ndims*ss
        let ds_msg_data_size = 8 + ndims * ss;
        let ds_msg_wire = 8 + ds_msg_data_size;
        let ds_msg_wire_aligned = (ds_msg_wire + 7) & !7;

        // Datatype: for integer types, 12 bytes
        // Use i32: class=0, version=1, size=4, bit_offset=0, bit_precision=32, signed
        let dt_msg_data_size = 12;
        let dt_msg_wire = 8 + dt_msg_data_size;
        let dt_msg_wire_aligned = (dt_msg_wire + 7) & !7;

        // Data layout v3 contiguous: version(1) + class(1) + addr(sa) + size(ss)
        let dl_msg_data_size = 2 + sa + ss;
        let dl_msg_wire = 8 + dl_msg_data_size;
        let dl_msg_wire_aligned = (dl_msg_wire + 7) & !7;

        let ds_ohdr_data_size = ds_msg_wire_aligned + dt_msg_wire_aligned + dl_msg_wire_aligned;
        let ds_ohdr_total = 16 + ds_ohdr_data_size; // v1 16-byte prefix
        let ds_ohdr_total_aligned = (ds_ohdr_total + 7) & !7;

        // Raw data: after dataset object header
        let raw_data_addr = ds_ohdr_addr + ds_ohdr_total_aligned as u64;
        let raw_data_size = data.len();

        let eof = raw_data_addr + raw_data_size as u64;

        // ---- Write the file ----

        // 1. Superblock v0
        let sig: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
        file.extend_from_slice(&sig);
        file.push(0); // version 0
        file.push(0); // free-space version
        file.push(0); // root group STE version
        file.push(0); // reserved
        file.push(0); // shared header version
        file.push(sa as u8); // sizeof_addr
        file.push(ss as u8); // sizeof_size
        file.push(0); // reserved
        file.extend_from_slice(&4u16.to_le_bytes()); // sym_leaf_k
        file.extend_from_slice(&32u16.to_le_bytes()); // btree_internal_k
        file.extend_from_slice(&0u32.to_le_bytes()); // file_consistency_flags
                                                     // base_addr
        write_le(&mut file, 0, sa);
        // extension_addr = UNDEF
        write_le(&mut file, UNDEF_ADDR, sa);
        // eof_addr
        write_le(&mut file, eof, sa);
        // driver_info_addr = UNDEF
        write_le(&mut file, UNDEF_ADDR, sa);
        // Root group STE:
        write_le(&mut file, 0, ss); // name_offset
        write_le(&mut file, root_ohdr_addr, sa); // obj_header_addr
        file.extend_from_slice(&1u32.to_le_bytes()); // cache_type = 1 (stab)
        file.extend_from_slice(&0u32.to_le_bytes()); // reserved
                                                     // scratch pad: btree_addr + heap_addr
        write_le(&mut file, btree_addr, sa);
        write_le(&mut file, heap_hdr_addr, sa);
        // Pad superblock
        while file.len() < sb_size_aligned {
            file.push(0);
        }

        // 2. Root group object header (v1, 16-byte prefix)
        assert_eq!(file.len(), root_ohdr_addr as usize);
        file.push(1); // version
        file.push(0); // reserved
        file.extend_from_slice(&1u16.to_le_bytes()); // num_messages = 1
        file.extend_from_slice(&1u32.to_le_bytes()); // obj_ref_count
        file.extend_from_slice(&(root_ohdr_data_size as u32).to_le_bytes());
        file.extend_from_slice(&[0u8; 4]); // reserved padding (v1 alignment)
                                           // Symbol table message (type 0x0011)
        file.extend_from_slice(&0x0011u16.to_le_bytes()); // type
        file.extend_from_slice(&(stab_msg_data_size as u16).to_le_bytes()); // size
        file.push(0); // flags
        file.extend_from_slice(&[0u8; 3]); // reserved
        write_le(&mut file, btree_addr, sa);
        write_le(&mut file, heap_hdr_addr, sa);
        // Pad
        while file.len() < (root_ohdr_addr as usize + root_ohdr_total_aligned) {
            file.push(0);
        }

        // 3. Local heap header
        assert_eq!(file.len(), heap_hdr_addr as usize);
        file.extend_from_slice(b"HEAP");
        file.push(0); // version
        file.extend_from_slice(&[0u8; 3]); // reserved
        write_le(&mut file, heap_data_size as u64, ss); // data_size
        write_le(&mut file, u64::MAX, ss); // free_list_offset (none)
        write_le(&mut file, heap_data_addr, sa); // data_addr
        while file.len() < (heap_hdr_addr as usize + heap_hdr_size_aligned) {
            file.push(0);
        }

        // 4. Local heap data
        assert_eq!(file.len(), heap_data_addr as usize);
        file.push(0); // offset 0: empty string (root self-reference)
        file.extend_from_slice(name_bytes); // offset 1: dataset name
        file.push(0); // null terminator
        while file.len() < (heap_data_addr as usize + heap_data_size) {
            file.push(0);
        }

        // 5. B-tree v1 (leaf, 1 entry)
        assert_eq!(file.len(), btree_addr as usize);
        file.extend_from_slice(b"TREE");
        file.push(0); // type = group
        file.push(0); // level = leaf
        file.extend_from_slice(&1u16.to_le_bytes()); // entries_used = 1
        write_le(&mut file, UNDEF_ADDR, sa); // left sibling
        write_le(&mut file, UNDEF_ADDR, sa); // right sibling
                                             // key[0] = 0 (first name offset)
        write_le(&mut file, 0, ss);
        // child[0] = snod_addr
        write_le(&mut file, snod_addr, sa);
        // key[1] = dataset name offset (after root)
        write_le(&mut file, 1, ss);
        while file.len() < (btree_addr as usize + btree_size_aligned) {
            file.push(0);
        }

        // 6. SNOD with 1 entry
        assert_eq!(file.len(), snod_addr as usize);
        file.extend_from_slice(b"SNOD");
        file.push(1); // version
        file.push(0); // reserved
        file.extend_from_slice(&1u16.to_le_bytes()); // num_symbols = 1
                                                     // Entry: dataset
        write_le(&mut file, 1, ss); // name_offset = 1 (index into local heap)
        write_le(&mut file, ds_ohdr_addr, sa); // obj_header_addr
        file.extend_from_slice(&0u32.to_le_bytes()); // cache_type = 0 (not a group)
        file.extend_from_slice(&0u32.to_le_bytes()); // reserved
        file.extend_from_slice(&[0u8; 16]); // scratch pad (unused)
        while file.len() < (snod_addr as usize + snod_size_aligned) {
            file.push(0);
        }

        // 7. Dataset object header (v1, 16-byte prefix)
        assert_eq!(file.len(), ds_ohdr_addr as usize);
        file.push(1); // version
        file.push(0); // reserved
        file.extend_from_slice(&3u16.to_le_bytes()); // num_messages = 3
        file.extend_from_slice(&1u32.to_le_bytes()); // obj_ref_count
        file.extend_from_slice(&(ds_ohdr_data_size as u32).to_le_bytes());
        file.extend_from_slice(&[0u8; 4]); // reserved padding (v1 alignment)

        // Message 1: Dataspace (type 0x01) - version 1
        file.extend_from_slice(&0x0001u16.to_le_bytes());
        file.extend_from_slice(&(ds_msg_data_size as u16).to_le_bytes());
        file.push(0); // flags
        file.extend_from_slice(&[0u8; 3]); // reserved
                                           // Dataspace v1 payload:
        file.push(1); // version = 1
        file.push(ndims as u8);
        file.push(0); // flags (no max dims)
        file.push(0); // reserved
        file.extend_from_slice(&[0u8; 4]); // reserved (4 bytes)
        for &d in dims {
            write_le(&mut file, d, ss);
        }
        // Pad message
        let target = ds_ohdr_addr as usize + 16 + ds_msg_wire_aligned;
        while file.len() < target {
            file.push(0);
        }

        // Message 2: Datatype (type 0x03) - i32
        file.extend_from_slice(&0x0003u16.to_le_bytes());
        file.extend_from_slice(&(dt_msg_data_size as u16).to_le_bytes());
        file.push(0); // flags
        file.extend_from_slice(&[0u8; 3]); // reserved
                                           // Datatype payload: class=0 (fixed point), version=1
        file.push(0x10); // class(0) | version(1)<<4
        file.push(0x08); // byte_order=LE, signed=true (bit 3)
        file.push(0); // flags byte 1
        file.push(0); // flags byte 2
        file.extend_from_slice(&(element_size as u32).to_le_bytes()); // element size
        file.extend_from_slice(&0u16.to_le_bytes()); // bit_offset
        file.extend_from_slice(&((element_size * 8) as u16).to_le_bytes()); // bit_precision
        let target = ds_ohdr_addr as usize + 16 + ds_msg_wire_aligned + dt_msg_wire_aligned;
        while file.len() < target {
            file.push(0);
        }

        // Message 3: Data Layout (type 0x08) - contiguous v3
        file.extend_from_slice(&0x0008u16.to_le_bytes());
        file.extend_from_slice(&(dl_msg_data_size as u16).to_le_bytes());
        file.push(0); // flags
        file.extend_from_slice(&[0u8; 3]); // reserved
                                           // Data layout payload:
        file.push(3); // version = 3
        file.push(1); // class = contiguous
        write_le(&mut file, raw_data_addr, sa); // address
        write_le(&mut file, raw_data_size as u64, ss); // size
        let target = ds_ohdr_addr as usize + ds_ohdr_total_aligned;
        while file.len() < target {
            file.push(0);
        }

        // 8. Raw data
        assert_eq!(file.len(), raw_data_addr as usize);
        file.extend_from_slice(data);

        assert_eq!(file.len(), eof as usize);
        file
    }

    #[test]
    fn test_read_v0_file_with_one_dataset() {
        let dims = [3u64, 4];
        let values: Vec<i32> = (0..12).collect();
        let raw_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let file_bytes = build_v0_file("my_dataset", &dims, &raw_data);

        // Write to a temp file
        let path = temp_path("v0_reader");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&file_bytes).unwrap();
            f.sync_all().unwrap();
        }

        // Read it back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        let names = reader.dataset_names();
        assert_eq!(names, vec!["my_dataset"]);

        let shape = reader.dataset_shape("my_dataset").unwrap();
        assert_eq!(shape, vec![3, 4]);

        let data = reader.read_dataset_raw("my_dataset").unwrap();
        assert_eq!(data, raw_data);

        // Verify the values
        let read_values: Vec<i32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(read_values, values);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_v0_file_1d_dataset() {
        let dims = [5u64];
        let values: Vec<i32> = vec![100, 200, 300, 400, 500];
        let raw_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let file_bytes = build_v0_file("data_1d", &dims, &raw_data);

        let path = temp_path("v0_1d");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&file_bytes).unwrap();
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_names(), vec!["data_1d"]);
        assert_eq!(reader.dataset_shape("data_1d").unwrap(), vec![5]);

        let data = reader.read_dataset_raw("data_1d").unwrap();
        let read_values: Vec<i32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(read_values, values);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_v2v3_still_works() {
        // Verify that opening a v3 file written by our writer still works
        let path = temp_path("detect_v3");
        {
            use crate::io::writer::Hdf5Writer;
            let mut writer = Hdf5Writer::create(&path).unwrap();
            let datatype = crate::format::messages::datatype::DatatypeMessage::i32_type();
            let idx = writer.create_dataset("test", datatype, &[4]).unwrap();
            let data = [1i32, 2, 3, 4];
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_dataset_raw(idx, &raw).unwrap();
            writer.close().unwrap();
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_names(), vec!["test"]);
        let shape = reader.dataset_shape("test").unwrap();
        assert_eq!(shape, vec![4]);

        let data = reader.read_dataset_raw("test").unwrap();
        let vals: Vec<i32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);

        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod h5py_debug_tests {
    use super::*;

    #[test]
    fn debug_read_h5py() {
        let path = std::path::Path::new("/tmp/test_h5py_default.h5");
        if !path.exists() {
            return;
        }

        let mut handle = FileHandle::open_read(path).unwrap();
        let sb_buf = handle.read_at_most(0, 1024).unwrap();
        let version = detect_superblock_version(&sb_buf).unwrap();
        eprintln!("Superblock version: {}", version);

        let sb = SuperblockV0V1::decode(&sb_buf).unwrap();
        eprintln!(
            "sizeof_addr={}, sizeof_size={}",
            sb.sizeof_offsets, sb.sizeof_lengths
        );
        eprintln!(
            "STE: obj_header={}, cache_type={}, btree={}, heap={}",
            sb.root_symbol_table_entry.obj_header_addr,
            sb.root_symbol_table_entry.cache_type,
            sb.root_symbol_table_entry.btree_addr,
            sb.root_symbol_table_entry.heap_addr
        );

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Read local heap
        let heap_buf = handle
            .read_at_most(sb.root_symbol_table_entry.heap_addr, 128)
            .unwrap();
        let heap_hdr = LocalHeapHeader::decode(
            &heap_buf,
            ctx.sizeof_addr as usize,
            ctx.sizeof_size as usize,
        )
        .unwrap();
        eprintln!(
            "Heap data_addr={}, data_size={}",
            heap_hdr.data_addr, heap_hdr.data_size
        );

        let heap_data = handle
            .read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)
            .unwrap();
        eprintln!(
            "Heap data bytes: {:?}",
            &heap_data[..std::cmp::min(64, heap_data.len())]
        );

        // Read btree
        let btree_buf = handle
            .read_at_most(sb.root_symbol_table_entry.btree_addr, 8192)
            .unwrap();
        let btree = BTreeV1Node::decode(
            &btree_buf,
            ctx.sizeof_addr as usize,
            ctx.sizeof_size as usize,
        )
        .unwrap();
        eprintln!(
            "BTree: type={}, level={}, entries={}, children={:?}",
            btree.node_type, btree.level, btree.entries_used, btree.children
        );

        // Read SNOD
        for &child in &btree.children {
            let snod_buf = handle.read_at_most(child, 8192).unwrap();
            let snod = SymbolTableNode::decode(
                &snod_buf,
                ctx.sizeof_addr as usize,
                ctx.sizeof_size as usize,
            )
            .unwrap();
            eprintln!("SNOD at {}: {} entries", child, snod.entries.len());
            for entry in &snod.entries {
                let name = local_heap_get_string(&heap_data, entry.name_offset).unwrap();
                eprintln!(
                    "  entry: name='{}' (offset={}), obj_header={}, cache_type={}",
                    name, entry.name_offset, entry.obj_header_addr, entry.cache_type
                );
            }
        }

        // Try full open
        let reader = Hdf5Reader::open(path).unwrap();
        eprintln!("Datasets found: {:?}", reader.dataset_names());
    }

    // ====================================================================
    // Group/link discovery: continuation blocks, dense links, v0/v1 groups.
    //
    // These tests generate HDF5 fixtures with h5py (HDF5 2.0.0). If the
    // pinned Python interpreter is not present, the test skips so the suite
    // still runs in environments without it.
    // ====================================================================

    const TEST_PYTHON: &str = "/Users/stevek/mamba/envs/bs2026.1/bin/python";

    /// Per-call unique temp path (PID + atomic counter) to avoid collisions
    /// across concurrent test runs.
    fn temp_path(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rust_hdf5_gap_test_{}_{}_{}.h5",
            name,
            std::process::id(),
            n
        ))
    }

    /// Run a Python snippet to generate a fixture; returns false if Python
    /// is unavailable so the caller can skip the test.
    fn gen_fixture(script: &str) -> bool {
        if !std::path::Path::new(TEST_PYTHON).exists() {
            return false;
        }
        let status = std::process::Command::new(TEST_PYTHON)
            .arg("-c")
            .arg(script)
            .status();
        matches!(status, Ok(s) if s.success())
    }

    #[test]
    fn gap1_v2_root_continuation_block() {
        let path = temp_path("gap1_cont");
        let p = path.display().to_string();
        // ~6 datasets in a v2 root group forces an object-header
        // continuation block.
        let script = format!(
            "import h5py,numpy as np\n\
             f=h5py.File(r'{p}','w',libver='latest')\n\
             [f.create_dataset('ds_%d'%i,data=np.arange(i*10,i*10+10,dtype='int32')) for i in range(6)]\n\
             f.close()"
        );
        if !gen_fixture(&script) {
            eprintln!("skipping gap1: python unavailable");
            return;
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();
        let mut names = reader.dataset_names();
        names.sort();
        assert_eq!(
            names,
            vec!["ds_0", "ds_1", "ds_2", "ds_3", "ds_4", "ds_5"],
            "all 6 datasets must be found across the continuation block"
        );
        // Element-exact read of one dataset.
        let raw = reader.read_dataset_raw("ds_3").unwrap();
        let vals: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, (30..40).collect::<Vec<i32>>());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gap2_v2_dense_fractal_heap_links() {
        let path = temp_path("gap2_dense");
        let p = path.display().to_string();
        // 14 datasets in one v2 group forces dense (fractal-heap) link
        // storage.
        let script = format!(
            "import h5py,numpy as np\n\
             f=h5py.File(r'{p}','w',libver='latest')\n\
             g=f.create_group('dense')\n\
             [g.create_dataset('d%02d'%i,data=np.full(4,i,dtype='float64')) for i in range(14)]\n\
             f.close()"
        );
        if !gen_fixture(&script) {
            eprintln!("skipping gap2: python unavailable");
            return;
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();
        let mut names = reader.dataset_names();
        names.sort();
        let expected: Vec<String> = (0..14).map(|i| format!("dense/d{:02}", i)).collect();
        assert_eq!(
            names, expected,
            "all 14 dense-stored links must be recovered from the fractal heap"
        );
        // Element-exact read of one dense-stored dataset.
        let raw = reader.read_dataset_raw("dense/d07").unwrap();
        let vals: Vec<f64> = raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        assert_eq!(vals, vec![7.0; 4]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gap3_v0v1_legacy_subgroups() {
        let path = temp_path("gap3_legacy");
        let p = path.display().to_string();
        // libver='earliest' => v0 superblock, symbol-table groups; datasets
        // nested inside subgroups.
        let script = format!(
            "import h5py,numpy as np\n\
             f=h5py.File(r'{p}','w',libver='earliest')\n\
             g1=f.create_group('grp1')\n\
             g1.create_dataset('a',data=np.arange(5,dtype='int16'))\n\
             g2=g1.create_group('sub')\n\
             g2.create_dataset('b',data=np.arange(7,dtype='int64'))\n\
             f.create_dataset('top',data=np.arange(3,dtype='int32'))\n\
             f.close()"
        );
        if !gen_fixture(&script) {
            eprintln!("skipping gap3: python unavailable");
            return;
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();
        let mut names = reader.dataset_names();
        names.sort();
        assert_eq!(
            names,
            vec!["grp1/a", "grp1/sub/b", "top"],
            "datasets nested in legacy symbol-table subgroups must be found"
        );
        // Element-exact read of a doubly-nested dataset.
        let raw = reader.read_dataset_raw("grp1/sub/b").unwrap();
        let vals: Vec<i64> = raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        assert_eq!(vals, (0..7).collect::<Vec<i64>>());
        let _ = std::fs::remove_file(&path);
    }

    /// N-bit chunked datasets with non-zero bit offset and signed types with
    /// negative values must read back element-exact through the crate's
    /// chunked readers. The post-filter datatype conversion shifts/masks/
    /// sign-extends each element after the filter pipeline.
    #[test]
    fn nbit_chunked_post_filter_conversion() {
        let path = temp_path("nbit_conv");
        let p = path.display().to_string();
        // Build N-bit datasets with reduced precision + non-zero offset via
        // h5py's low-level filter API (h5py has no high-level N-bit knob).
        let script = format!(
            "import h5py,numpy as np\n\
             from h5py import h5t,h5p,h5s,h5d,h5f,h5z\n\
             fid=h5f.create(r'{p}'.encode())\n\
             def mk(name,bt,prec,off,npd,vals,chunk):\n\
            \x20dt=bt.copy();dt.set_precision(prec);dt.set_offset(off)\n\
            \x20arr=np.ascontiguousarray(np.asarray(vals,dtype=npd))\n\
            \x20sp=h5s.create_simple(arr.shape)\n\
            \x20dc=h5p.create(h5p.DATASET_CREATE);dc.set_chunk(chunk)\n\
            \x20dc.set_filter(h5z.FILTER_NBIT,h5z.FLAG_OPTIONAL,())\n\
            \x20ds=h5d.create(fid,name.encode(),dt,sp,dc)\n\
            \x20ds.write(h5s.ALL,h5s.ALL,arr);ds.close()\n\
             mk('u4_p17_o3',h5t.STD_U32LE,17,3,'u4',[0,1,1000,65535,131071,70000,42,99999],(4,))\n\
             mk('i4_p13_o5',h5t.STD_I32LE,13,5,'i4',[-5,-1,0,1,7,-4096,4095,-77,42,100,-100,3],(4,))\n\
             mk('i2_p9_o4',h5t.STD_I16LE,9,4,'i2',[-256,-1,0,1,255,-7,7,-200],(3,))\n\
             mk('i4_2d_p11_o6',h5t.STD_I32LE,11,6,'i4',np.array([[-1024,-1,0,5],[1023,-77,88,-3]],dtype='i4'),(1,4))\n\
             fid.close()"
        );
        if !gen_fixture(&script) {
            eprintln!("skipping nbit_chunked_post_filter_conversion: python unavailable");
            return;
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();

        // Unsigned u4, precision 17, bit offset 3.
        let raw = reader.read_dataset_raw("u4_p17_o3").unwrap();
        let got: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got,
            vec![0u32, 1, 1000, 65535, 131071, 70000, 42, 99999],
            "u4 N-bit dataset must decode to exact unsigned values"
        );

        // Signed i4 with negatives, precision 13, bit offset 5.
        let raw = reader.read_dataset_raw("i4_p13_o5").unwrap();
        let got: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got,
            vec![-5i32, -1, 0, 1, 7, -4096, 4095, -77, 42, 100, -100, 3],
            "i4 N-bit dataset must sign-extend negative values"
        );

        // Signed i2 with negatives, precision 9, bit offset 4.
        let raw = reader.read_dataset_raw("i2_p9_o4").unwrap();
        let got: Vec<i16> = raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(
            got,
            vec![-256i16, -1, 0, 1, 255, -7, 7, -200],
            "i2 N-bit dataset must sign-extend negative values"
        );

        // 2D signed i4, precision 11, bit offset 6 (1-row chunks).
        let raw = reader.read_dataset_raw("i4_2d_p11_o6").unwrap();
        let got: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got,
            vec![-1024i32, -1, 0, 5, 1023, -77, 88, -3],
            "2D i4 N-bit dataset must decode element-exact"
        );

        // read_slice path must also apply the conversion exactly once.
        let raw = reader.read_slice("i4_p13_o5", &[4], &[3]).unwrap();
        let got: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![7i32, -4096, 4095], "read_slice must convert too");

        // 2D slice: second row, all columns.
        let raw = reader.read_slice("i4_2d_p11_o6", &[1, 0], &[1, 4]).unwrap();
        let got: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got,
            vec![1023i32, -77, 88, -3],
            "2D read_slice must convert"
        );

        let _ = std::fs::remove_file(&path);
    }
}
