//! HDF5 file reader.
//!
//! Opens an HDF5 file, parses the superblock and root group, and provides
//! access to dataset metadata and raw data.
//!
//! Supports both legacy (v0/v1 superblock, v1 object headers, symbol tables)
//! and modern (v2/v3 superblock, v2 object headers, link messages) formats.

use std::path::Path;

use hdf5_format::superblock::{SuperblockV2V3, SuperblockV0V1, detect_superblock_version};
use hdf5_format::object_header::ObjectHeader;
use hdf5_format::messages::*;
use hdf5_format::messages::dataspace::DataspaceMessage;
use hdf5_format::messages::datatype::DatatypeMessage;
use hdf5_format::messages::link::LinkMessage;
use hdf5_format::messages::link::LinkTarget;
use hdf5_format::messages::data_layout::{self, DataLayoutMessage};
use hdf5_format::local_heap::{LocalHeapHeader, local_heap_get_string};
use hdf5_format::symbol_table::SymbolTableNode;
use hdf5_format::btree_v1::BTreeV1Node;
use hdf5_format::{FormatContext, UNDEF_ADDR};

use crate::file_handle::FileHandle;
use crate::IoResult;

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
}

impl Hdf5Reader {
    /// Open an existing HDF5 file in SWMR read mode.
    ///
    /// Currently identical to `open()`, but indicates intent to use
    /// `refresh()` for re-reading metadata written by a concurrent SWMR writer.
    pub fn open_swmr(path: &Path) -> IoResult<Self> {
        Self::open(path)
    }

    /// Open an existing HDF5 file for reading.
    ///
    /// Auto-detects the superblock version and uses the appropriate code path:
    /// - v0/v1: legacy format with symbol tables and B-tree v1
    /// - v2/v3: modern format with link messages
    pub fn open(path: &Path) -> IoResult<Self> {
        let mut handle = FileHandle::open_read(path)?;

        // Read enough bytes to detect the superblock version and parse it.
        let sb_buf = handle.read_at_most(0, 1024)?;
        let version = detect_superblock_version(&sb_buf)?;

        match version {
            0 | 1 => Self::open_v0v1(handle, &sb_buf),
            2 | 3 => Self::open_v2v3(handle, &sb_buf),
            v => Err(crate::IoError::Format(
                hdf5_format::FormatError::InvalidVersion(v),
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

        // Read root group object header.
        let root_buf = handle.read_at_most(sb.root_group_object_header_address, 4096)?;
        let (root_header, _) = ObjectHeader::decode(&root_buf)?;

        // Walk link messages to discover datasets.
        let datasets = Self::discover_datasets_from_links(&mut handle, &root_header, &ctx)?;

        Ok(Self {
            handle,
            ctx,
            _eof: sb.end_of_file_address,
            root_group_info: RootGroupInfo::V2V3 {
                root_group_object_header_address: sb.root_group_object_header_address,
            },
            datasets,
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
    fn discover_datasets_from_links(
        handle: &mut FileHandle,
        root_header: &ObjectHeader,
        ctx: &FormatContext,
    ) -> IoResult<Vec<DatasetReadInfo>> {
        let mut datasets = Vec::new();
        for msg in &root_header.messages {
            if msg.msg_type == MSG_LINK {
                let (link, _) = LinkMessage::decode(&msg.data, ctx)?;
                if let LinkTarget::Hard { address } = &link.target {
                    if let Some(info) = Self::read_dataset_from_object_header(
                        handle, ctx, *address, &link.name,
                    )? {
                        datasets.push(info);
                    }
                }
            }
        }
        Ok(datasets)
    }

    /// Discover datasets by walking the B-tree v1 + local heap (legacy format).
    fn discover_datasets_from_btree(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        btree_addr: u64,
        heap_addr: u64,
    ) -> IoResult<Vec<DatasetReadInfo>> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;

        // Read the local heap header
        let heap_hdr_buf = handle.read_at_most(heap_addr, 64)?;
        let heap_hdr = LocalHeapHeader::decode(&heap_hdr_buf, sa, ss)?;

        // Read the local heap data
        let heap_data = handle.read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)?;

        // Collect all SNOD addresses by walking the B-tree
        let snod_addrs = Self::collect_snod_addresses(handle, btree_addr, sa, ss)?;

        let mut datasets = Vec::new();

        for snod_addr in snod_addrs {
            // Read SNOD
            let snod_buf = handle.read_at_most(snod_addr, 8192)?;
            let snod = SymbolTableNode::decode(&snod_buf, sa, ss)?;

            for entry in &snod.entries {
                // Get the name from the local heap
                let name = local_heap_get_string(&heap_data, entry.name_offset)?;

                // Skip empty names (root group self-reference)
                if name.is_empty() {
                    continue;
                }

                // Try to read this as a dataset
                if let Some(info) = Self::read_dataset_from_object_header(
                    handle, ctx, entry.obj_header_addr, &name,
                )? {
                    datasets.push(info);
                }
            }
        }

        Ok(datasets)
    }

    /// Recursively walk a B-tree v1 to collect leaf-level SNOD addresses.
    fn collect_snod_addresses(
        handle: &mut FileHandle,
        tree_addr: u64,
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> IoResult<Vec<u64>> {
        let buf = handle.read_at_most(tree_addr, 8192)?;
        let node = BTreeV1Node::decode(&buf, sizeof_addr, sizeof_size)?;

        if node.level == 0 {
            // Leaf level: children are SNOD addresses
            Ok(node.children.clone())
        } else {
            // Internal level: children are sub-TREE addresses
            let mut addrs = Vec::new();
            for &child_addr in &node.children {
                let child_addrs =
                    Self::collect_snod_addresses(handle, child_addr, sizeof_addr, sizeof_size)?;
                addrs.extend(child_addrs);
            }
            Ok(addrs)
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
        // Read the primary object header chunk
        let buf = handle.read_at_most(addr, 4096)?;
        let (mut header, _) = ObjectHeader::decode_any(&buf)?;

        // Follow continuation messages (type 0x10) to read additional chunks.
        // This is essential for v1 headers that span multiple chunks.
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let mut continuations: Vec<(u64, u64)> = Vec::new();
        for msg in &header.messages {
            if msg.msg_type == MSG_OBJ_HEADER_CONTINUATION && msg.data.len() >= sa + ss {
                let cont_addr = read_uint(&msg.data, sa);
                let cont_len = read_uint(&msg.data[sa..], ss);
                continuations.push((cont_addr, cont_len));
            }
        }

        // Parse messages from continuation chunks
        for (cont_addr, cont_len) in continuations {
            if cont_addr == UNDEF_ADDR || cont_len == 0 {
                continue;
            }
            let cont_buf = handle.read_at_most(cont_addr, cont_len as usize)?;
            // Parse messages from the continuation chunk (v1 format: no header prefix)
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
                    header.messages.push(hdf5_format::object_header::ObjectHeaderMessage {
                        msg_type: msg_type as u8,
                        flags: msg_flags,
                        data: cont_buf[pos..pos + data_size].to_vec(),
                    });
                }
                pos += data_size;
                // v1 alignment to 8 bytes
                pos = (pos + 7) & !7;
            }
        }

        let mut datatype = None;
        let mut dataspace = None;
        let mut layout = None;

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
                _ => {}
            }
        }

        if let (Some(dt), Some(ds), Some(dl)) = (datatype, dataspace, layout) {
            Ok(Some(DatasetReadInfo {
                name: name.to_string(),
                datatype: dt,
                dataspace: ds,
                layout: dl,
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

    /// Return the dimensions of a dataset.
    pub fn dataset_shape(&self, name: &str) -> IoResult<Vec<u64>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;
        Ok(info.dataspace.dims.clone())
    }

    /// Read the raw bytes of a dataset.
    pub fn read_dataset_raw(&mut self, name: &str) -> IoResult<Vec<u8>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;

        // Clone layout to avoid borrow conflict with &mut self in read methods.
        let layout = info.layout.clone();

        match &layout {
            DataLayoutMessage::Contiguous { address, size } => {
                if *address == UNDEF_ADDR {
                    return Ok(vec![]);
                }
                let data = self.handle.read_at(*address, *size as usize)?;
                Ok(data)
            }
            DataLayoutMessage::Compact { data } => Ok(data.clone()),
            DataLayoutMessage::ChunkedV4 { chunk_dims, index_address, index_type, earray_params, .. } => {
                // The layout's chunk_dims include the element size as
                // the trailing dimension. Strip it for chunk indexing.
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_v4(name, real_chunk_dims, *index_address, *index_type, earray_params.as_ref())
            }
        }
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

        // Re-read root group object header.
        let root_buf = self.handle.read_at_most(sb.root_group_object_header_address, 4096)?;
        let (root_header, _) = ObjectHeader::decode(&root_buf)?;

        // Re-scan datasets from link messages.
        let datasets = Self::discover_datasets_from_links(&mut self.handle, &root_header, &ctx)?;

        self._eof = sb.end_of_file_address;
        self.ctx = ctx;
        self.datasets = datasets;

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
    ) -> IoResult<Vec<u8>> {
        use hdf5_format::chunk_index::extensible_array::*;

        let info = self.dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;

        match index_type {
            data_layout::ChunkIndexType::SingleChunk => {
                // Single chunk: the index_address IS the chunk address
                let total_size: u64 = dims.iter().product::<u64>() * element_size;
                if index_address == UNDEF_ADDR || total_size == 0 {
                    return Ok(vec![]);
                }
                let data = self.handle.read_at(index_address, total_size as usize)?;
                Ok(data)
            }
            data_layout::ChunkIndexType::FixedArray => {
                self.read_chunked_fixed_array(name, chunk_dims, index_address)
            }
            data_layout::ChunkIndexType::BTreeV2 => {
                self.read_chunked_btree_v2(name, chunk_dims, index_address)
            }
            data_layout::ChunkIndexType::ExtensibleArray => {
                let params = earray_params.ok_or_else(|| {
                    crate::IoError::InvalidState("missing earray params".into())
                })?;

                if index_address == UNDEF_ADDR {
                    return Ok(vec![]);
                }

                // Read the EA header
                let hdr_buf = self.handle.read_at_most(index_address, 256)?;
                let ea_hdr = ExtensibleArrayHeader::decode(&hdr_buf, &self.ctx)?;

                if ea_hdr.idx_blk_addr == UNDEF_ADDR {
                    return Ok(vec![]);
                }

                // Compute number of chunks along the unlimited dimension (dim 0)
                let chunks_dim0 = if chunk_dims[0] > 0 {
                    dims[0].div_ceil(chunk_dims[0])
                } else {
                    0
                };

                // Read index block
                let ndblk_addrs = compute_ndblk_addrs(params.sup_blk_min_data_ptrs);
                let nsblk_addrs = compute_nsblk_addrs(
                    params.idx_blk_elmts,
                    params.data_blk_min_elmts,
                    params.sup_blk_min_data_ptrs,
                    params.max_nelmts_bits,
                );
                let iblk_buf = self.handle.read_at_most(ea_hdr.idx_blk_addr, 8192)?;
                let iblk = ExtensibleArrayIndexBlock::decode(
                    &iblk_buf, &self.ctx,
                    params.idx_blk_elmts as usize,
                    ndblk_addrs, nsblk_addrs,
                )?;

                // Collect chunk addresses
                let mut chunk_addrs: Vec<u64> = Vec::new();
                for &addr in &iblk.elements {
                    chunk_addrs.push(addr);
                }

                // Read data blocks if needed
                let mut dblk_nelmts = params.data_blk_min_elmts as usize;
                for &dblk_addr in &iblk.dblk_addrs {
                    if dblk_addr == UNDEF_ADDR {
                        // Add UNDEF entries for the unallocated block
                        for _ in 0..dblk_nelmts {
                            chunk_addrs.push(UNDEF_ADDR);
                        }
                    } else {
                        let dblk_buf = self.handle.read_at_most(dblk_addr, 4096)?;
                        let dblk = ExtensibleArrayDataBlock::decode(
                            &dblk_buf, &self.ctx,
                            params.max_nelmts_bits, dblk_nelmts,
                        )?;
                        for &addr in &dblk.elements {
                            chunk_addrs.push(addr);
                        }
                    }
                    // Data blocks grow: first 2 are min size, then double
                    // For simplicity, keep at min for now
                    if chunk_addrs.len() >= chunks_dim0 as usize {
                        break;
                    }
                    dblk_nelmts *= 2;
                }

                // Compute chunk byte size
                let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

                // Total output size
                let total_size: u64 = dims.iter().product::<u64>() * element_size;
                let mut output = vec![0u8; total_size as usize];

                // Read each chunk
                for i in 0..chunks_dim0 as usize {
                    if i >= chunk_addrs.len() {
                        break;
                    }
                    let addr = chunk_addrs[i];
                    if addr == UNDEF_ADDR {
                        continue;
                    }
                    let chunk_data = self.handle.read_at(addr, chunk_bytes as usize)?;
                    let offset = i as u64 * chunk_bytes;
                    let end = std::cmp::min(offset + chunk_bytes, total_size);
                    let copy_len = (end - offset) as usize;
                    output[offset as usize..offset as usize + copy_len]
                        .copy_from_slice(&chunk_data[..copy_len]);
                }

                Ok(output)
            }
            _ => Err(crate::IoError::InvalidState(format!(
                "unsupported chunk index type: {:?}", index_type
            ))),
        }
    }

    /// Read a dataset indexed by a fixed array.
    fn read_chunked_fixed_array(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
    ) -> IoResult<Vec<u8>> {
        use hdf5_format::chunk_index::fixed_array::*;

        let info = self.dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
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

        // Read FA data block
        let dblk_buf = self.handle.read_at_most(fa_hdr.data_blk_addr, 65536)?;
        let fa_dblk = FixedArrayDataBlock::decode_unfiltered(
            &dblk_buf, &self.ctx, fa_hdr.num_elmts as usize,
        )?;

        // Compute chunk byte size
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

        // Total output size
        let total_size: u64 = dims.iter().product::<u64>() * element_size;
        let mut output = vec![0u8; total_size as usize];

        // Compute number of chunks per dimension
        let chunks_per_dim: Vec<u64> = (0..ndims)
            .map(|d| dims[d].div_ceil(chunk_dims[d]))
            .collect();

        // Read each chunk and place it in the correct position
        for linear_idx in 0..fa_hdr.num_elmts as usize {
            if linear_idx >= fa_dblk.elements.len() {
                break;
            }
            let addr = fa_dblk.elements[linear_idx];
            if addr == UNDEF_ADDR {
                continue;
            }

            // Convert linear index to multidimensional chunk coordinates
            let mut remaining = linear_idx as u64;
            let mut coords = vec![0u64; ndims];
            for d in (0..ndims).rev() {
                coords[d] = remaining % chunks_per_dim[d];
                remaining /= chunks_per_dim[d];
            }

            let chunk_data = self.handle.read_at(addr, chunk_bytes as usize)?;

            // Copy chunk data into output at the correct position
            // For each element in the chunk, compute its position in the output
            self.copy_chunk_to_output(
                &chunk_data, &mut output, &dims, chunk_dims, &coords, element_size,
            );
        }

        Ok(output)
    }

    /// Read a dataset indexed by a B-tree v2.
    fn read_chunked_btree_v2(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
    ) -> IoResult<Vec<u8>> {
        use hdf5_format::chunk_index::btree_v2::*;

        let info = self.dataset_info(name)
            .ok_or_else(|| crate::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
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

        // For depth=0, root is a leaf node
        if bt2_hdr.depth != 0 {
            return Err(crate::IoError::InvalidState(
                "B-tree v2 depth > 0 not yet supported for reading".into(),
            ));
        }

        // Read leaf node
        let leaf_buf = self.handle.read_at_most(bt2_hdr.root_node_addr, 65536)?;
        let leaf = Bt2LeafNode::decode(
            &leaf_buf,
            bt2_hdr.num_records_in_root,
            bt2_hdr.record_size,
        )?;

        // Decode records
        let records = if bt2_hdr.record_type == BT2_TYPE_CHUNK_UNFILT {
            Bt2ChunkIndex::decode_unfiltered_records(
                &leaf.record_data,
                bt2_hdr.num_records_in_root as usize,
                ndims,
                &self.ctx,
            )?
        } else {
            return Err(crate::IoError::InvalidState(
                "filtered B-tree v2 chunk reading not yet supported".into(),
            ));
        };

        // Compute chunk byte size
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;

        // Total output size
        let total_size: u64 = dims.iter().product::<u64>() * element_size;
        let mut output = vec![0u8; total_size as usize];

        // Read each chunk and place it in the correct position
        for rec in &records {
            if rec.chunk_address == UNDEF_ADDR {
                continue;
            }
            let chunk_data = self.handle.read_at(rec.chunk_address, chunk_bytes as usize)?;

            // The scaled_offsets are the chunk coordinates
            self.copy_chunk_to_output(
                &chunk_data, &mut output, &dims, chunk_dims,
                &rec.scaled_offsets, element_size,
            );
        }

        Ok(output)
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
            let start = chunk_coords[0] * chunk_dims[0] * element_size;
            let actual_elems = std::cmp::min(
                chunk_dims[0],
                dims[0] - chunk_coords[0] * chunk_dims[0],
            );
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
}

/// Read a little-endian unsigned integer of `n` bytes into a u64.
fn read_uint(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
        let path = std::env::temp_dir().join("hdf5_test_v0_reader.h5");
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

        let path = std::env::temp_dir().join("hdf5_test_v0_1d.h5");
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
        let path = std::env::temp_dir().join("hdf5_test_detect_v3.h5");
        {
            use crate::writer::Hdf5Writer;
            let mut writer = Hdf5Writer::create(&path).unwrap();
            let datatype = hdf5_format::messages::datatype::DatatypeMessage::i32_type();
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
        let vals: Vec<i32> = data.chunks_exact(4)
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
        if !path.exists() { return; }
        
        let mut handle = FileHandle::open_read(path).unwrap();
        let sb_buf = handle.read_at_most(0, 1024).unwrap();
        let version = detect_superblock_version(&sb_buf).unwrap();
        eprintln!("Superblock version: {}", version);
        
        let sb = SuperblockV0V1::decode(&sb_buf).unwrap();
        eprintln!("sizeof_addr={}, sizeof_size={}", sb.sizeof_offsets, sb.sizeof_lengths);
        eprintln!("STE: obj_header={}, cache_type={}, btree={}, heap={}",
            sb.root_symbol_table_entry.obj_header_addr,
            sb.root_symbol_table_entry.cache_type,
            sb.root_symbol_table_entry.btree_addr,
            sb.root_symbol_table_entry.heap_addr);
        
        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };
        
        // Read local heap
        let heap_buf = handle.read_at_most(sb.root_symbol_table_entry.heap_addr, 128).unwrap();
        let heap_hdr = LocalHeapHeader::decode(&heap_buf, ctx.sizeof_addr as usize, ctx.sizeof_size as usize).unwrap();
        eprintln!("Heap data_addr={}, data_size={}", heap_hdr.data_addr, heap_hdr.data_size);
        
        let heap_data = handle.read_at(heap_hdr.data_addr, heap_hdr.data_size as usize).unwrap();
        eprintln!("Heap data bytes: {:?}", &heap_data[..std::cmp::min(64, heap_data.len())]);
        
        // Read btree
        let btree_buf = handle.read_at_most(sb.root_symbol_table_entry.btree_addr, 8192).unwrap();
        let btree = BTreeV1Node::decode(&btree_buf, ctx.sizeof_addr as usize, ctx.sizeof_size as usize).unwrap();
        eprintln!("BTree: type={}, level={}, entries={}, children={:?}", 
            btree.node_type, btree.level, btree.entries_used, btree.children);
        
        // Read SNOD
        for &child in &btree.children {
            let snod_buf = handle.read_at_most(child, 8192).unwrap();
            let snod = SymbolTableNode::decode(&snod_buf, ctx.sizeof_addr as usize, ctx.sizeof_size as usize).unwrap();
            eprintln!("SNOD at {}: {} entries", child, snod.entries.len());
            for entry in &snod.entries {
                let name = local_heap_get_string(&heap_data, entry.name_offset).unwrap();
                eprintln!("  entry: name='{}' (offset={}), obj_header={}, cache_type={}",
                    name, entry.name_offset, entry.obj_header_addr, entry.cache_type);
            }
        }
        
        // Try full open
        let reader = Hdf5Reader::open(path).unwrap();
        eprintln!("Datasets found: {:?}", reader.dataset_names());
    }
}
