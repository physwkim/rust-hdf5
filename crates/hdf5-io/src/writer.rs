//! HDF5 file writer.
//!
//! Produces a valid HDF5 file with superblock v3, a root group object header,
//! and datasets with contiguous or chunked storage. The output is readable by `h5dump`.

use std::path::Path;

use hdf5_format::superblock::*;
use hdf5_format::object_header::ObjectHeader;
use hdf5_format::messages::*;
use hdf5_format::messages::dataspace::DataspaceMessage;
use hdf5_format::messages::datatype::DatatypeMessage;
use hdf5_format::messages::fill_value::FillValueMessage;
use hdf5_format::messages::link::LinkMessage;
use hdf5_format::messages::link_info::LinkInfoMessage;
use hdf5_format::messages::group_info::GroupInfoMessage;
use hdf5_format::messages::attribute::AttributeMessage;
use hdf5_format::messages::data_layout::{DataLayoutMessage, EarrayParams, FixedArrayParams};
use hdf5_format::messages::filter::{FilterPipeline, self};
use hdf5_format::chunk_index::extensible_array::{
    ExtensibleArrayHeader, ExtensibleArrayIndexBlock, ExtensibleArrayDataBlock,
    compute_ndblk_addrs, compute_nsblk_addrs,
};
use hdf5_format::chunk_index::fixed_array::{
    FixedArrayHeader, FixedArrayDataBlock,
};
use hdf5_format::chunk_index::btree_v2::Bt2ChunkIndex;
use hdf5_format::{FormatContext, UNDEF_ADDR};

use crate::file_handle::FileHandle;
use crate::allocator::FileAllocator;
use crate::IoResult;

/// Metadata for a dataset being written.
pub struct DatasetInfo {
    /// Link name within the root group.
    pub name: String,
    /// Element datatype.
    pub datatype: DatatypeMessage,
    /// Dataspace (dimensionality).
    pub dataspace: DataspaceMessage,
    /// File offset of the dataset's object header (set during finalize).
    pub obj_header_addr: u64,
    /// File offset of the raw data block (contiguous only).
    pub data_addr: u64,
    /// Size of the raw data in bytes (contiguous only).
    pub data_size: u64,
    /// Chunked storage info (None for contiguous).
    pub chunked: Option<ChunkedDatasetInfo>,
    /// Fixed array chunked storage info.
    pub fixed_array: Option<FixedArrayDatasetInfo>,
    /// B-tree v2 chunked storage info.
    pub btree_v2: Option<Bt2DatasetInfo>,
    /// Attributes attached to this dataset.
    pub attributes: Vec<AttributeMessage>,
    /// File offset where the dataset object header was written (for SWMR in-place rewrites).
    pub obj_header_written_addr: Option<u64>,
    /// Encoded size of the dataset object header (for verifying in-place rewrites fit).
    pub obj_header_encoded_size: usize,
    /// Filter pipeline for compressed chunks.
    pub filter_pipeline: Option<FilterPipeline>,
}

/// Runtime metadata for a chunked dataset.
pub struct ChunkedDatasetInfo {
    /// Chunk dimension sizes.
    pub chunk_dims: Vec<u64>,
    /// Maximum dimensions (u64::MAX = unlimited).
    pub max_dims: Vec<u64>,
    /// Extensible array parameters.
    pub earray_params: EarrayParams,
    /// File offset of the EA header.
    pub ea_header_addr: u64,
    /// File offset of the EA index block.
    pub ea_iblk_addr: u64,
    /// Number of data block address slots in the index block.
    pub ndblk_addrs: usize,
    /// In-memory copy of the EA header (for updating statistics).
    pub ea_header: ExtensibleArrayHeader,
    /// In-memory copy of the EA index block (for updating element addresses).
    pub ea_iblk: ExtensibleArrayIndexBlock,
    /// Data blocks that have been allocated. Each entry: (file_addr, data_block).
    pub data_blocks: Vec<(u64, ExtensibleArrayDataBlock)>,
    /// Number of chunks written so far.
    pub chunks_written: u64,
}

/// Runtime metadata for a fixed-array-indexed chunked dataset.
pub struct FixedArrayDatasetInfo {
    /// Chunk dimension sizes.
    pub chunk_dims: Vec<u64>,
    /// File offset of the FA header.
    pub fa_header_addr: u64,
    /// File offset of the FA data block.
    pub fa_dblk_addr: u64,
    /// In-memory copy of the FA header.
    pub fa_header: FixedArrayHeader,
    /// In-memory copy of the FA data block.
    pub fa_dblk: FixedArrayDataBlock,
    /// Number of chunks written so far.
    pub chunks_written: u64,
}

/// Runtime metadata for a B-tree v2 indexed chunked dataset.
pub struct Bt2DatasetInfo {
    /// Chunk dimension sizes.
    pub chunk_dims: Vec<u64>,
    /// Maximum dimensions (u64::MAX = unlimited).
    pub max_dims: Vec<u64>,
    /// File offset of the BT2 header.
    pub bt2_header_addr: u64,
    /// File offset of the BT2 leaf node.
    pub bt2_leaf_addr: u64,
    /// In-memory chunk index.
    pub index: Bt2ChunkIndex,
    /// Number of chunks written so far.
    pub chunks_written: u64,
}

/// HDF5 file writer.
///
/// Usage:
/// 1. `Hdf5Writer::create(path)` to create a new file.
/// 2. `create_dataset(name, datatype, dims)` to define datasets.
/// 3. `write_dataset_raw(index, data)` to write raw data.
/// 4. `close()` to finalize the file (writes superblock, headers, etc.).
pub struct Hdf5Writer {
    handle: FileHandle,
    allocator: FileAllocator,
    ctx: FormatContext,
    pub(crate) datasets: Vec<DatasetInfo>,
    closed: bool,
    /// Address of the root group object header (set after first finalize).
    root_group_addr: Option<u64>,
    /// Size of the encoded root group object header (for in-place rewrites).
    root_group_encoded_size: usize,
}

impl Hdf5Writer {
    /// Create a new HDF5 file at `path`.
    ///
    /// The superblock (48 bytes for v3 with 8-byte offsets) is reserved at
    /// offset 0 and written during `close()`.
    pub fn create(path: &Path) -> IoResult<Self> {
        let handle = FileHandle::create(path)?;
        let ctx = FormatContext::default_v3();

        // Reserve space for the superblock. We compute the size from a dummy
        // instance so that we stay in sync with the encoder.
        let sb_size = (SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: ctx.sizeof_addr,
            sizeof_lengths: ctx.sizeof_size,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 0,
            root_group_object_header_address: 0,
        })
        .encoded_size() as u64;

        let allocator = FileAllocator::new(sb_size);

        Ok(Self {
            handle,
            allocator,
            ctx,
            datasets: Vec::new(),
            closed: false,
            root_group_addr: None,
            root_group_encoded_size: 0,
        })
    }

    /// Provide public access to the format context.
    pub fn ctx(&self) -> &FormatContext {
        &self.ctx
    }

    /// Define a new contiguous dataset. Returns the dataset index (used with
    /// `write_dataset_raw`).
    ///
    /// The raw-data region is allocated immediately so that
    /// `write_dataset_raw` can be called at any time before `close()`.
    pub fn create_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
    ) -> IoResult<usize> {
        let total_elements: u64 = dims.iter().product();
        let element_size = datatype.element_size() as u64;
        let data_size = total_elements * element_size;

        // Allocate space for the raw data.
        let data_addr = if data_size > 0 {
            self.allocator.allocate(data_size)
        } else {
            UNDEF_ADDR
        };

        let dataspace = DataspaceMessage::simple(dims);

        let idx = self.datasets.len();
        self.datasets.push(DatasetInfo {
            name: name.to_string(),
            datatype,
            dataspace,
            obj_header_addr: 0, // set during finalize
            data_addr,
            data_size,
            chunked: None,
            fixed_array: None,
            btree_v2: None,
            attributes: Vec::new(),
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            filter_pipeline: None,
        });

        Ok(idx)
    }

    /// Define a new chunked dataset with an extensible array index.
    ///
    /// Returns the dataset index. The dataset starts empty (dims[0] = 0 if
    /// the first dimension is unlimited). Use `write_chunk` and
    /// `extend_dataset` to add data.
    pub fn create_chunked_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        let earray_params = EarrayParams::default_params();
        let ndblk_addrs = compute_ndblk_addrs(earray_params.sup_blk_min_data_ptrs);
        let nsblk_addrs = compute_nsblk_addrs(
            earray_params.idx_blk_elmts,
            earray_params.data_blk_min_elmts,
            earray_params.sup_blk_min_data_ptrs,
            earray_params.max_nelmts_bits,
        );

        // Create EA header
        let mut ea_header = ExtensibleArrayHeader::new_for_chunks(&self.ctx);
        ea_header.max_nelmts_bits = earray_params.max_nelmts_bits;
        ea_header.idx_blk_elmts = earray_params.idx_blk_elmts;
        ea_header.data_blk_min_elmts = earray_params.data_blk_min_elmts;
        ea_header.sup_blk_min_data_ptrs = earray_params.sup_blk_min_data_ptrs;
        ea_header.max_dblk_page_nelmts_bits = earray_params.max_dblk_page_nelmts_bits;

        // Allocate and write EA header (placeholder, will be updated)
        let hdr_encoded = ea_header.encode(&self.ctx);
        let ea_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create EA index block with pre-allocated super block address slots
        let ea_iblk = ExtensibleArrayIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );

        // Allocate and write EA index block
        let iblk_encoded = ea_iblk.encode(&self.ctx);
        let ea_iblk_addr = self.allocator.allocate(iblk_encoded.len() as u64);

        // Update header with index block address
        ea_header.idx_blk_addr = ea_iblk_addr;

        // Write both to disk
        let hdr_encoded = ea_header.encode(&self.ctx);
        self.handle.write_at(ea_header_addr, &hdr_encoded)?;
        self.handle.write_at(ea_iblk_addr, &iblk_encoded)?;

        // Build dataspace with max dims
        let dataspace = DataspaceMessage {
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let idx = self.datasets.len();
        self.datasets.push(DatasetInfo {
            name: name.to_string(),
            datatype,
            dataspace,
            obj_header_addr: 0,
            data_addr: UNDEF_ADDR,
            data_size: 0,
            attributes: Vec::new(),
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            filter_pipeline: None,
            fixed_array: None,
            btree_v2: None,
            chunked: Some(ChunkedDatasetInfo {
                chunk_dims: chunk_dims.to_vec(),
                max_dims: max_dims.to_vec(),
                earray_params,
                ea_header_addr,
                ea_iblk_addr,
                ndblk_addrs,
                ea_header,
                ea_iblk,
                data_blocks: Vec::new(),
                chunks_written: 0,
            }),
        });

        Ok(idx)
    }

    /// Write raw bytes to a contiguous dataset identified by `index`.
    ///
    /// The caller is responsible for providing data in the correct byte order
    /// and layout. The length must match the total data size declared at
    /// creation time.
    pub fn write_dataset_raw(&mut self, index: usize, data: &[u8]) -> IoResult<()> {
        let ds = &self.datasets[index];
        if ds.chunked.is_some() {
            return Err(crate::IoError::InvalidState(
                "use write_chunk for chunked datasets".into(),
            ));
        }
        if ds.data_addr == UNDEF_ADDR {
            return Err(crate::IoError::InvalidState(
                "dataset has no data allocated".into(),
            ));
        }
        if data.len() as u64 != ds.data_size {
            return Err(crate::IoError::InvalidState(format!(
                "data size mismatch: expected {} bytes, got {}",
                ds.data_size,
                data.len()
            )));
        }
        self.handle.write_at(ds.data_addr, data)?;
        Ok(())
    }

    /// Write a chunk of data to a chunked dataset.
    ///
    /// `chunk_offset` is the chunk coordinates (e.g., [frame_idx] for a 1D-chunked
    /// streaming dataset where chunk_dims = [1, H, W]).
    /// Only the first (unlimited) dimension index is used for EA indexing.
    ///
    /// `data` must be exactly chunk_size bytes (product of chunk_dims * element_size).
    pub fn write_chunk(
        &mut self,
        index: usize,
        chunk_idx: u64,
        data: &[u8],
    ) -> IoResult<()> {
        let ds = &self.datasets[index];
        let element_size = ds.datatype.element_size() as u64;
        let chunked = ds.chunked.as_ref().ok_or_else(|| {
            crate::IoError::InvalidState("not a chunked dataset".into())
        })?;
        let chunk_bytes: u64 = chunked.chunk_dims.iter().product::<u64>() * element_size;

        if data.len() as u64 != chunk_bytes {
            return Err(crate::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes, data.len()
            )));
        }

        // Allocate space for the chunk data
        let chunk_addr = self.allocator.allocate(chunk_bytes);
        self.handle.write_at(chunk_addr, data)?;

        // Update the extensible array to record this chunk address
        let idx_blk_elmts = {
            let c = self.datasets[index].chunked.as_ref().unwrap();
            c.earray_params.idx_blk_elmts as u64
        };

        if chunk_idx < idx_blk_elmts {
            // Fits in index block elements directly
            let chunked = self.datasets[index].chunked.as_mut().unwrap();
            chunked.ea_iblk.elements[chunk_idx as usize] = chunk_addr;
            chunked.chunks_written += 1;

            // Update stats
            if chunk_idx + 1 > chunked.ea_header.max_idx_set {
                chunked.ea_header.max_idx_set = chunk_idx + 1;
            }
            // num_elmts_realized = total slots allocated (idx_blk_elmts always realized)
            if chunked.ea_header.num_elmts_realized < idx_blk_elmts {
                chunked.ea_header.num_elmts_realized = idx_blk_elmts;
            }
        } else {
            // Need to use data blocks
            let offset_in_dblks = chunk_idx - idx_blk_elmts;

            // Figure out which data block this goes into.
            // Data blocks are sized: min_elmts, min_elmts, 2*min_elmts, 2*min_elmts,
            // 4*min_elmts, 4*min_elmts, ...
            // But for the index block's dblk_addrs, we have ndblk_addrs slots.
            // Each slot corresponds to a data block.
            let chunked = self.datasets[index].chunked.as_mut().unwrap();
            let min_elmts = chunked.earray_params.data_blk_min_elmts as u64;

            // Walk through data block sizes to find which one contains this offset
            let mut cumulative = 0u64;
            let mut dblk_idx = 0usize;
            let mut current_size = min_elmts;

            // Data block sizing pattern:
            // First 2 blocks: min_elmts each (corresponding to sblk 0)
            // Next 2 blocks: 2*min_elmts each (sblk 1)
            // etc.
            // But we only have ndblk_addrs slots in the index block.
            let mut pair_count = 0;
            loop {
                if offset_in_dblks < cumulative + current_size {
                    break;
                }
                cumulative += current_size;
                dblk_idx += 1;
                pair_count += 1;
                if pair_count >= 2 {
                    pair_count = 0;
                    current_size *= 2;
                }
                if dblk_idx >= chunked.ndblk_addrs {
                    return Err(crate::IoError::InvalidState(
                        "chunk index exceeds extensible array capacity".into(),
                    ));
                }
            }

            let offset_in_block = (offset_in_dblks - cumulative) as usize;
            let block_nelmts = current_size as usize;

            // Check if this data block exists already
            if chunked.ea_iblk.dblk_addrs[dblk_idx] == UNDEF_ADDR {
                // Allocate a new data block
                let mut dblk = ExtensibleArrayDataBlock::new(
                    chunked.ea_header_addr,
                    cumulative,
                    block_nelmts,
                );
                dblk.elements[offset_in_block] = chunk_addr;

                let dblk_encoded = dblk.encode(&self.ctx, chunked.earray_params.max_nelmts_bits);
                let dblk_addr = self.allocator.allocate(dblk_encoded.len() as u64);
                self.handle.write_at(dblk_addr, &dblk_encoded)?;

                chunked.ea_iblk.dblk_addrs[dblk_idx] = dblk_addr;
                chunked.data_blocks.push((dblk_addr, dblk));

                // Update header stats
                chunked.ea_header.num_dblks_created += 1;
                chunked.ea_header.size_dblks_created += dblk_encoded.len() as u64;
            } else {
                // Update existing data block
                let dblk_addr = chunked.ea_iblk.dblk_addrs[dblk_idx];
                let dblk_entry = chunked.data_blocks.iter_mut()
                    .find(|(addr, _)| *addr == dblk_addr);
                if let Some((_, ref mut dblk)) = dblk_entry {
                    dblk.elements[offset_in_block] = chunk_addr;
                    // Re-encode and write
                    let dblk_encoded = dblk.encode(&self.ctx, chunked.earray_params.max_nelmts_bits);
                    self.handle.write_at(dblk_addr, &dblk_encoded)?;
                }
            }

            chunked.chunks_written += 1;
            if chunk_idx + 1 > chunked.ea_header.max_idx_set {
                chunked.ea_header.max_idx_set = chunk_idx + 1;
            }
            // When creating a data block of size N, N more elements are realized
            let total_realized = idx_blk_elmts
                + chunked.data_blocks.iter()
                    .map(|(_, db)| db.elements.len() as u64)
                    .sum::<u64>();
            chunked.ea_header.num_elmts_realized = total_realized;
        }

        Ok(())
    }

    /// Add an attribute to a dataset.
    ///
    /// The attribute will be written as a message in the dataset's object
    /// header when the file is finalized.
    pub fn add_dataset_attribute(
        &mut self,
        ds_index: usize,
        attr: AttributeMessage,
    ) -> IoResult<()> {
        if ds_index >= self.datasets.len() {
            return Err(crate::IoError::InvalidState(format!(
                "dataset index {} out of range (have {})",
                ds_index,
                self.datasets.len()
            )));
        }
        self.datasets[ds_index].attributes.push(attr);
        Ok(())
    }

    /// Define a chunked dataset indexed by a fixed array (no unlimited dimensions).
    ///
    /// `dims` and `max_dims` should be the same (all fixed). `chunk_dims` defines the
    /// chunk shape. Returns the dataset index.
    pub fn create_fixed_array_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        // Compute total number of chunks
        let ndims = dims.len();
        let mut num_chunks: u64 = 1;
        for d in 0..ndims {
            num_chunks *= dims[d].div_ceil(chunk_dims[d]);
        }

        // Create FA header
        let mut fa_header = FixedArrayHeader::new_for_chunks(&self.ctx, num_chunks);
        let hdr_encoded = fa_header.encode(&self.ctx);
        let fa_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create FA data block
        let fa_dblk = FixedArrayDataBlock::new_unfiltered(fa_header_addr, num_chunks as usize);
        let dblk_encoded = fa_dblk.encode_unfiltered(&self.ctx);
        let fa_dblk_addr = self.allocator.allocate(dblk_encoded.len() as u64);

        // Update header with data block address
        fa_header.data_blk_addr = fa_dblk_addr;

        // Write both
        let hdr_encoded = fa_header.encode(&self.ctx);
        self.handle.write_at(fa_header_addr, &hdr_encoded)?;
        self.handle.write_at(fa_dblk_addr, &dblk_encoded)?;

        let dataspace = DataspaceMessage::simple(dims);

        let idx = self.datasets.len();
        self.datasets.push(DatasetInfo {
            name: name.to_string(),
            datatype,
            dataspace,
            obj_header_addr: 0,
            data_addr: UNDEF_ADDR,
            data_size: 0,
            chunked: None,
            btree_v2: None,
            attributes: Vec::new(),
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            filter_pipeline: None,
            fixed_array: Some(FixedArrayDatasetInfo {
                chunk_dims: chunk_dims.to_vec(),
                fa_header_addr,
                fa_dblk_addr,
                fa_header,
                fa_dblk,
                chunks_written: 0,
            }),
        });

        Ok(idx)
    }

    /// Define a chunked dataset indexed by a B-tree v2 (multiple unlimited dimensions).
    ///
    /// Returns the dataset index.
    pub fn create_btree_v2_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        let ndims = dims.len();
        let bt2_index = Bt2ChunkIndex::new_unfiltered(ndims);

        // We'll allocate space for header and leaf node; they'll be written
        // during flush_dataset_bt2.
        let hdr = hdf5_format::chunk_index::btree_v2::Bt2Header::new_for_chunks(&self.ctx, ndims);
        let hdr_encoded = hdr.encode(&self.ctx);
        let bt2_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);
        self.handle.write_at(bt2_header_addr, &hdr_encoded)?;

        // Allocate a placeholder leaf node (empty for now)
        let leaf = hdf5_format::chunk_index::btree_v2::Bt2LeafNode::new(
            hdf5_format::chunk_index::btree_v2::BT2_TYPE_CHUNK_UNFILT,
            bt2_index.record_size(&self.ctx),
        );
        let leaf_encoded = leaf.encode();
        let bt2_leaf_addr = self.allocator.allocate(leaf_encoded.len() as u64);
        self.handle.write_at(bt2_leaf_addr, &leaf_encoded)?;

        let dataspace = DataspaceMessage {
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let idx = self.datasets.len();
        self.datasets.push(DatasetInfo {
            name: name.to_string(),
            datatype,
            dataspace,
            obj_header_addr: 0,
            data_addr: UNDEF_ADDR,
            data_size: 0,
            chunked: None,
            fixed_array: None,
            attributes: Vec::new(),
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            filter_pipeline: None,
            btree_v2: Some(Bt2DatasetInfo {
                chunk_dims: chunk_dims.to_vec(),
                max_dims: max_dims.to_vec(),
                bt2_header_addr,
                bt2_leaf_addr,
                index: bt2_index,
                chunks_written: 0,
            }),
        });

        Ok(idx)
    }

    /// Create a chunked dataset with compression using the given filter pipeline.
    ///
    /// This is similar to `create_chunked_dataset` but attaches a filter pipeline
    /// (e.g., deflate compression). The pipeline is applied when writing chunks.
    pub fn create_chunked_dataset_compressed(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        compression_level: u32,
    ) -> IoResult<usize> {
        let idx = self.create_chunked_dataset(name, datatype, dims, max_dims, chunk_dims)?;
        self.datasets[idx].filter_pipeline = Some(FilterPipeline::deflate(compression_level));
        Ok(idx)
    }

    /// Write a chunk to a fixed-array-indexed dataset.
    ///
    /// `chunk_coords` is the multidimensional chunk index (e.g., [row_chunk, col_chunk]).
    pub fn write_chunk_fixed_array(
        &mut self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = &self.datasets[index];
        let element_size = ds.datatype.element_size() as u64;
        let fa = ds.fixed_array.as_ref().ok_or_else(|| {
            crate::IoError::InvalidState("not a fixed-array dataset".into())
        })?;
        let chunk_bytes: u64 = fa.chunk_dims.iter().product::<u64>() * element_size;

        // Possibly compress the data
        let write_data;
        let data_to_write = if let Some(ref pipeline) = ds.filter_pipeline {
            write_data = filter::apply_filters(pipeline, data)?;
            &write_data
        } else {
            if data.len() as u64 != chunk_bytes {
                return Err(crate::IoError::InvalidState(format!(
                    "chunk data size mismatch: expected {} bytes, got {}",
                    chunk_bytes,
                    data.len()
                )));
            }
            data
        };

        // Compute linear chunk index from multidimensional coordinates
        let dims = &ds.dataspace.dims;
        let chunk_dims = &fa.chunk_dims;
        let ndims = dims.len();
        let mut linear_idx: u64 = 0;
        let mut stride: u64 = 1;
        for d in (0..ndims).rev() {
            let n_chunks_in_dim = dims[d].div_ceil(chunk_dims[d]);
            linear_idx += chunk_coords[d] * stride;
            stride *= n_chunks_in_dim;
        }

        // Allocate space for the chunk data
        let chunk_addr = self.allocator.allocate(data_to_write.len() as u64);
        self.handle.write_at(chunk_addr, data_to_write)?;

        // Update the fixed array data block
        let fa = self.datasets[index].fixed_array.as_mut().unwrap();
        if (linear_idx as usize) < fa.fa_dblk.elements.len() {
            fa.fa_dblk.elements[linear_idx as usize] = chunk_addr;
            fa.chunks_written += 1;
        } else {
            return Err(crate::IoError::InvalidState(format!(
                "chunk index {} out of range (max {})",
                linear_idx,
                fa.fa_dblk.elements.len()
            )));
        }

        Ok(())
    }

    /// Write a chunk to a B-tree v2 indexed dataset.
    ///
    /// `chunk_coords` is the scaled chunk coordinates (one per dimension).
    pub fn write_chunk_btree_v2(
        &mut self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = &self.datasets[index];
        let element_size = ds.datatype.element_size() as u64;
        let bt2 = ds.btree_v2.as_ref().ok_or_else(|| {
            crate::IoError::InvalidState("not a B-tree v2 dataset".into())
        })?;
        let chunk_bytes: u64 = bt2.chunk_dims.iter().product::<u64>() * element_size;

        if data.len() as u64 != chunk_bytes {
            return Err(crate::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes,
                data.len()
            )));
        }

        // Allocate space for the chunk data
        let chunk_addr = self.allocator.allocate(chunk_bytes);
        self.handle.write_at(chunk_addr, data)?;

        // Insert into the in-memory BT2 index
        let bt2 = self.datasets[index].btree_v2.as_mut().unwrap();
        bt2.index.insert(chunk_coords.to_vec(), chunk_addr);
        bt2.chunks_written += 1;

        Ok(())
    }

    /// Write multiple chunks in a batch, optionally compressing in parallel.
    ///
    /// `chunks` is a list of (chunk_idx, data) pairs for an EA-indexed dataset.
    pub fn write_chunks_batch(
        &mut self,
        ds_index: usize,
        chunks: &[(u64, &[u8])],
    ) -> IoResult<()> {
        #[cfg(feature = "parallel")]
        {
            // If filter pipeline is set, compress all chunks in parallel
            if let Some(ref pipeline) = self.datasets[ds_index].filter_pipeline {
                let chunk_data: Vec<Vec<u8>> = chunks.iter().map(|(_, d)| d.to_vec()).collect();
                let compressed = filter::apply_filters_parallel(pipeline, &chunk_data);
                for ((idx, _), compressed_data) in chunks.iter().zip(compressed.iter()) {
                    self.write_compressed_chunk(ds_index, *idx, compressed_data)?;
                }
                return Ok(());
            }
        }
        // Fallback: sequential
        for (idx, data) in chunks {
            self.write_chunk(ds_index, *idx, data)?;
        }
        Ok(())
    }

    /// Write a pre-compressed chunk to a chunked dataset.
    ///
    /// The chunk data is already compressed; this method just writes it and updates
    /// the chunk index. The recorded chunk size in the index will reflect the
    /// compressed size.
    pub fn write_compressed_chunk(
        &mut self,
        index: usize,
        chunk_idx: u64,
        compressed_data: &[u8],
    ) -> IoResult<()> {
        // Allocate space for the compressed chunk data
        let chunk_addr = self.allocator.allocate(compressed_data.len() as u64);
        self.handle.write_at(chunk_addr, compressed_data)?;

        // Update the extensible array (same logic as write_chunk but with the
        // compressed data address)
        let idx_blk_elmts = {
            let c = self.datasets[index].chunked.as_ref().ok_or_else(|| {
                crate::IoError::InvalidState("not a chunked dataset".into())
            })?;
            c.earray_params.idx_blk_elmts as u64
        };

        if chunk_idx < idx_blk_elmts {
            let chunked = self.datasets[index].chunked.as_mut().unwrap();
            chunked.ea_iblk.elements[chunk_idx as usize] = chunk_addr;
            chunked.chunks_written += 1;
            if chunk_idx + 1 > chunked.ea_header.max_idx_set {
                chunked.ea_header.max_idx_set = chunk_idx + 1;
            }
            if chunked.ea_header.num_elmts_realized < idx_blk_elmts {
                chunked.ea_header.num_elmts_realized = idx_blk_elmts;
            }
        } else {
            // For simplicity, delegate to the same data-block allocation logic
            // We need to compute which data block this goes into
            let chunked = self.datasets[index].chunked.as_mut().unwrap();
            let min_elmts = chunked.earray_params.data_blk_min_elmts as u64;
            let offset_in_dblks = chunk_idx - idx_blk_elmts;

            let mut cumulative = 0u64;
            let mut dblk_idx = 0usize;
            let mut current_size = min_elmts;
            let mut pair_count = 0;
            loop {
                if offset_in_dblks < cumulative + current_size {
                    break;
                }
                cumulative += current_size;
                dblk_idx += 1;
                pair_count += 1;
                if pair_count >= 2 {
                    pair_count = 0;
                    current_size *= 2;
                }
                if dblk_idx >= chunked.ndblk_addrs {
                    return Err(crate::IoError::InvalidState(
                        "chunk index exceeds extensible array capacity".into(),
                    ));
                }
            }

            let offset_in_block = (offset_in_dblks - cumulative) as usize;
            let block_nelmts = current_size as usize;

            if chunked.ea_iblk.dblk_addrs[dblk_idx] == UNDEF_ADDR {
                let mut dblk = ExtensibleArrayDataBlock::new(
                    chunked.ea_header_addr,
                    cumulative,
                    block_nelmts,
                );
                dblk.elements[offset_in_block] = chunk_addr;
                let dblk_encoded = dblk.encode(&self.ctx, chunked.earray_params.max_nelmts_bits);
                let dblk_addr = self.allocator.allocate(dblk_encoded.len() as u64);
                self.handle.write_at(dblk_addr, &dblk_encoded)?;
                chunked.ea_iblk.dblk_addrs[dblk_idx] = dblk_addr;
                chunked.data_blocks.push((dblk_addr, dblk));
                chunked.ea_header.num_dblks_created += 1;
                chunked.ea_header.size_dblks_created += dblk_encoded.len() as u64;
            } else {
                let dblk_addr = chunked.ea_iblk.dblk_addrs[dblk_idx];
                let dblk_entry = chunked.data_blocks.iter_mut()
                    .find(|(addr, _)| *addr == dblk_addr);
                if let Some((_, ref mut dblk)) = dblk_entry {
                    dblk.elements[offset_in_block] = chunk_addr;
                    let dblk_encoded = dblk.encode(&self.ctx, chunked.earray_params.max_nelmts_bits);
                    self.handle.write_at(dblk_addr, &dblk_encoded)?;
                }
            }

            chunked.chunks_written += 1;
            if chunk_idx + 1 > chunked.ea_header.max_idx_set {
                chunked.ea_header.max_idx_set = chunk_idx + 1;
            }
            let total_realized = idx_blk_elmts
                + chunked.data_blocks.iter()
                    .map(|(_, db)| db.elements.len() as u64)
                    .sum::<u64>();
            chunked.ea_header.num_elmts_realized = total_realized;
        }

        Ok(())
    }

    /// Extend the dimensions of a chunked dataset.
    pub fn extend_dataset(&mut self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = &mut self.datasets[index];
        if ds.chunked.is_none() && ds.fixed_array.is_none() && ds.btree_v2.is_none() {
            return Err(crate::IoError::InvalidState(
                "can only extend chunked datasets".into(),
            ));
        }
        ds.dataspace.dims = new_dims.to_vec();
        Ok(())
    }

    /// Flush a chunked dataset's index structures to disk.
    pub fn flush_dataset(&mut self, index: usize) -> IoResult<()> {
        let ds = &self.datasets[index];

        // EA-indexed dataset
        if let Some(ref chunked) = ds.chunked {
            let iblk_encoded = chunked.ea_iblk.encode(&self.ctx);
            self.handle.write_at(chunked.ea_iblk_addr, &iblk_encoded)?;
            let hdr_encoded = chunked.ea_header.encode(&self.ctx);
            self.handle.write_at(chunked.ea_header_addr, &hdr_encoded)?;
            self.handle.sync_data()?;
            return Ok(());
        }

        // Fixed-array-indexed dataset
        if let Some(ref fa) = ds.fixed_array {
            let dblk_encoded = fa.fa_dblk.encode_unfiltered(&self.ctx);
            self.handle.write_at(fa.fa_dblk_addr, &dblk_encoded)?;
            let hdr_encoded = fa.fa_header.encode(&self.ctx);
            self.handle.write_at(fa.fa_header_addr, &hdr_encoded)?;
            self.handle.sync_data()?;
            return Ok(());
        }

        // BT2-indexed dataset
        if let Some(ref bt2) = ds.btree_v2 {
            // Re-encode the leaf node and header
            let (hdr_bytes, leaf_bytes) = bt2.index.encode(&self.ctx);

            // The leaf may have grown -- reallocate if needed
            let leaf_addr = self.allocator.allocate(leaf_bytes.len() as u64);
            self.handle.write_at(leaf_addr, &leaf_bytes)?;

            // Update header with new root node address
            let mut hdr = hdf5_format::chunk_index::btree_v2::Bt2Header::decode(&hdr_bytes, &self.ctx)?;
            hdr.root_node_addr = leaf_addr;
            let hdr_encoded = hdr.encode(&self.ctx);
            self.handle.write_at(bt2.bt2_header_addr, &hdr_encoded)?;

            // Update our in-memory copy's leaf addr
            let bt2_mut = self.datasets[index].btree_v2.as_mut().unwrap();
            bt2_mut.bt2_leaf_addr = leaf_addr;

            self.handle.sync_data()?;
            return Ok(());
        }

        Ok(())
    }

    /// Finalize and close the file.
    ///
    /// Writes the dataset object headers, root group object header, and
    /// superblock. After this call the file is a valid HDF5 file.
    pub fn close(mut self) -> IoResult<()> {
        self.finalize()?;
        self.closed = true;
        Ok(())
    }

    /// Provide mutable access to the underlying file handle.
    pub fn handle(&mut self) -> &mut FileHandle {
        &mut self.handle
    }

    /// Return the current end-of-file offset.
    pub fn eof(&self) -> u64 {
        self.allocator.eof()
    }

    /// Write the superblock at offset 0 with the given flags.
    ///
    /// Requires that the root group has already been written (via `finalize`
    /// or `finalize_for_swmr`).
    pub fn write_superblock(&mut self, flags: u8) -> IoResult<()> {
        let root_addr = self.root_group_addr.ok_or_else(|| {
            crate::IoError::InvalidState("root group not yet written".into())
        })?;
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: self.ctx.sizeof_addr,
            sizeof_lengths: self.ctx.sizeof_size,
            file_consistency_flags: flags,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: self.allocator.eof(),
            root_group_object_header_address: root_addr,
        };
        let sb_encoded = sb.encode();
        self.handle.write_at(0, &sb_encoded)?;
        Ok(())
    }

    /// Re-write a dataset's object header in place (SWMR update).
    ///
    /// The header must have been previously written via `finalize_for_swmr`.
    /// Only the dataspace dimensions change; the encoded size must not exceed
    /// the originally allocated space.
    pub fn write_dataset_header_inplace(&mut self, index: usize) -> IoResult<()> {
        let addr = self.datasets[index].obj_header_written_addr.ok_or_else(|| {
            crate::IoError::InvalidState("dataset header not yet written".into())
        })?;
        let original_size = self.datasets[index].obj_header_encoded_size;

        let header = self.build_dataset_header(index);
        let encoded = header.encode();

        if encoded.len() > original_size {
            return Err(crate::IoError::InvalidState(format!(
                "dataset header grew from {} to {} bytes; cannot rewrite in place",
                original_size, encoded.len()
            )));
        }

        // Pad to original size with zeros (the trailing zeros after the
        // checksum won't be parsed by readers since chunk0_data_size is fixed).
        let mut padded = encoded;
        padded.resize(original_size, 0);

        self.handle.write_at(addr, &padded)?;
        Ok(())
    }

    /// Perform a full finalize for SWMR mode.
    ///
    /// This writes all dataset object headers, the root group header, and the
    /// superblock with SWMR flags. After this call, the file is valid for
    /// SWMR readers. Subsequent writes use in-place updates.
    pub fn finalize_for_swmr(&mut self) -> IoResult<()> {
        // 0. Flush all chunked dataset index structures.
        for i in 0..self.datasets.len() {
            if self.datasets[i].chunked.is_some()
                || self.datasets[i].fixed_array.is_some()
                || self.datasets[i].btree_v2.is_some()
            {
                self.flush_dataset(i)?;
            }
        }

        // 1. Write each dataset's object header.
        for i in 0..self.datasets.len() {
            let ds_header = self.build_dataset_header(i);
            let encoded = ds_header.encode();
            let encoded_size = encoded.len();
            let addr = self.allocator.allocate(encoded_size as u64);
            self.handle.write_at(addr, &encoded)?;
            self.datasets[i].obj_header_addr = addr;
            self.datasets[i].obj_header_written_addr = Some(addr);
            self.datasets[i].obj_header_encoded_size = encoded_size;
        }

        // 2. Write root group object header.
        let root_header = self.build_root_group_header();
        let root_encoded = root_header.encode();
        let root_encoded_size = root_encoded.len();
        let root_addr = self.allocator.allocate(root_encoded_size as u64);
        self.handle.write_at(root_addr, &root_encoded)?;
        self.root_group_addr = Some(root_addr);
        self.root_group_encoded_size = root_encoded_size;

        // 3. Write superblock with SWMR flags.
        self.write_superblock(FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE)?;

        self.handle.sync_all()?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn finalize(&mut self) -> IoResult<()> {
        // If SWMR finalize was already done, re-write headers in place and
        // update the superblock with clean-close flags.
        if self.root_group_addr.is_some() {
            // Flush index structures for all chunked datasets.
            for i in 0..self.datasets.len() {
                if self.datasets[i].chunked.is_some()
                    || self.datasets[i].fixed_array.is_some()
                    || self.datasets[i].btree_v2.is_some()
                {
                    self.flush_dataset(i)?;
                }
            }
            // Re-write dataset headers in place with final dims.
            for i in 0..self.datasets.len() {
                if self.datasets[i].obj_header_written_addr.is_some() {
                    self.write_dataset_header_inplace(i)?;
                }
            }
            // Write superblock with clean-close flags (no SWMR).
            self.write_superblock(0)?;
            self.handle.sync_all()?;
            return Ok(());
        }

        // 0. Flush all chunked dataset index structures.
        for i in 0..self.datasets.len() {
            if self.datasets[i].chunked.is_some()
                || self.datasets[i].fixed_array.is_some()
                || self.datasets[i].btree_v2.is_some()
            {
                self.flush_dataset(i)?;
            }
        }

        // 1. Write each dataset's object header.
        for i in 0..self.datasets.len() {
            let ds_header = self.build_dataset_header(i);
            let encoded = ds_header.encode();
            let addr = self.allocator.allocate(encoded.len() as u64);
            self.handle.write_at(addr, &encoded)?;
            self.datasets[i].obj_header_addr = addr;
        }

        // 2. Write root group object header.
        let root_header = self.build_root_group_header();
        let root_encoded = root_header.encode();
        let root_addr = self.allocator.allocate(root_encoded.len() as u64);
        self.handle.write_at(root_addr, &root_encoded)?;
        self.root_group_addr = Some(root_addr);

        // 3. Write superblock at offset 0.
        self.write_superblock(0)?;

        self.handle.sync_all()?;
        Ok(())
    }

    fn build_dataset_header(&self, index: usize) -> ObjectHeader {
        let ds = &self.datasets[index];
        let mut header = ObjectHeader::new();

        // Dataspace message (type 0x01)
        let ds_msg = ds.dataspace.encode(&self.ctx);
        header.add_message(MSG_DATASPACE, 0x00, ds_msg);

        // Datatype message (type 0x03), flag 0x01 = constant
        let dt_msg = ds.datatype.encode(&self.ctx);
        header.add_message(MSG_DATATYPE, 0x01, dt_msg);

        // Fill Value message (type 0x05)
        let is_chunked = ds.chunked.is_some() || ds.fixed_array.is_some() || ds.btree_v2.is_some();
        let fv = if is_chunked {
            FillValueMessage {
                alloc_time: 3,      // incremental
                fill_write_time: 0, // on alloc
                fill_defined: 1,    // default value (zeros)
                fill_value: None,
            }
        } else {
            FillValueMessage::default()
        };
        let fv_msg = fv.encode();
        header.add_message(MSG_FILL_VALUE, 0x00, fv_msg);

        // Data Layout message (type 0x08)
        let layout = if let Some(ref chunked) = ds.chunked {
            let mut layout_dims = chunked.chunk_dims.clone();
            layout_dims.push(ds.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_earray(
                layout_dims,
                chunked.earray_params.clone(),
                chunked.ea_header_addr,
            )
        } else if let Some(ref fa) = ds.fixed_array {
            let mut layout_dims = fa.chunk_dims.clone();
            layout_dims.push(ds.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_farray(
                layout_dims,
                FixedArrayParams::default_params(),
                fa.fa_header_addr,
            )
        } else if let Some(ref bt2) = ds.btree_v2 {
            let mut layout_dims = bt2.chunk_dims.clone();
            layout_dims.push(ds.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_btree_v2(
                layout_dims,
                bt2.bt2_header_addr,
            )
        } else {
            DataLayoutMessage::contiguous(ds.data_addr, ds.data_size)
        };
        let layout_msg = layout.encode(&self.ctx);
        header.add_message(MSG_DATA_LAYOUT, 0x00, layout_msg);

        // Filter Pipeline message (type 0x0B) -- only if filters are configured
        if let Some(ref pipeline) = ds.filter_pipeline {
            if !pipeline.filters.is_empty() {
                let filter_msg = pipeline.encode();
                header.add_message(MSG_FILTER_PIPELINE, 0x00, filter_msg);
            }
        }

        // Attribute messages (type 0x0C)
        for attr in &ds.attributes {
            let attr_msg = attr.encode(&self.ctx);
            header.add_message(MSG_ATTRIBUTE, 0x00, attr_msg);
        }

        header
    }

    fn build_root_group_header(&self) -> ObjectHeader {
        let mut header = ObjectHeader::new();

        // Link Info message (type 0x02) — compact storage
        let link_info = LinkInfoMessage::compact();
        let li_msg = link_info.encode(&self.ctx);
        header.add_message(MSG_LINK_INFO, 0x00, li_msg);

        // Group Info message (type 0x0A) — defaults
        let group_info = GroupInfoMessage::default();
        let gi_msg = group_info.encode();
        header.add_message(MSG_GROUP_INFO, 0x00, gi_msg);

        // Link messages (type 0x06) — one per dataset
        for ds in &self.datasets {
            let link = LinkMessage::hard(&ds.name, ds.obj_header_addr);
            let link_msg = link.encode(&self.ctx);
            header.add_message(MSG_LINK, 0x00, link_msg);
        }

        header
    }
}

impl Drop for Hdf5Writer {
    fn drop(&mut self) {
        if !self.closed {
            // Best-effort finalize on drop.
            let _ = self.finalize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Hdf5Reader;
    use hdf5_format::messages::datatype::DatatypeMessage;

    #[test]
    fn create_empty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_empty.h5");

        let writer = Hdf5Writer::create(&path).unwrap();
        writer.close().unwrap();

        // Verify we can read it back
        let reader = Hdf5Reader::open(&path).unwrap();
        assert!(reader.dataset_names().is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_single_dataset() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_single.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_dataset("data", DatatypeMessage::f64_type(), &[4])
            .unwrap();
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_dataset_raw(idx, &raw).unwrap();
        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_names(), vec!["data"]);
        assert_eq!(reader.dataset_shape("data").unwrap(), vec![4]);
        let readback = reader.read_dataset_raw("data").unwrap();
        assert_eq!(readback, raw);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_multiple_datasets() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_multi.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();

        let idx0 = writer
            .create_dataset("ints", DatatypeMessage::i32_type(), &[3])
            .unwrap();
        let i_data: Vec<u8> = [10i32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_dataset_raw(idx0, &i_data).unwrap();

        let idx1 = writer
            .create_dataset("floats", DatatypeMessage::f32_type(), &[2, 2])
            .unwrap();
        let f_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_dataset_raw(idx1, &f_data).unwrap();

        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        let names = reader.dataset_names();
        assert!(names.contains(&"ints"));
        assert!(names.contains(&"floats"));
        assert_eq!(reader.dataset_shape("ints").unwrap(), vec![3]);
        assert_eq!(reader.dataset_shape("floats").unwrap(), vec![2, 2]);
        assert_eq!(reader.read_dataset_raw("ints").unwrap(), i_data);
        assert_eq!(reader.read_dataset_raw("floats").unwrap(), f_data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn data_size_mismatch() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mismatch.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_dataset("x", DatatypeMessage::u8_type(), &[4])
            .unwrap();
        let err = writer.write_dataset_raw(idx, &[1, 2, 3]); // 3 bytes instead of 4
        assert!(err.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_chunked_dataset_simple() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_chunked_simple.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_chunked_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 4],       // start empty
                &[u64::MAX, 4], // unlimited first dim
                &[1, 4],       // chunk = [1, 4]
            )
            .unwrap();

        // Write 3 frames (chunks)
        for frame in 0..3u64 {
            let values: Vec<f64> = (0..4).map(|i| (frame * 4 + i) as f64).collect();
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_chunk(idx, frame, &raw).unwrap();
        }

        // Extend dimensions
        writer.extend_dataset(idx, &[3, 4]).unwrap();

        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_names(), vec!["data"]);
        assert_eq!(reader.dataset_shape("data").unwrap(), vec![3, 4]);

        let raw = reader.read_dataset_raw("data").unwrap();
        let values: Vec<f64> = raw.chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 12);
        for i in 0..12 {
            assert_eq!(values[i], i as f64);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn chunked_dataset_many_frames() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_chunked_many.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_chunked_dataset(
                "frames",
                DatatypeMessage::i32_type(),
                &[0, 2],
                &[u64::MAX, 2],
                &[1, 2],
            )
            .unwrap();

        let n_frames = 10u64;
        for frame in 0..n_frames {
            let values = [(frame * 2) as i32, (frame * 2 + 1) as i32];
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_chunk(idx, frame, &raw).unwrap();
        }

        writer.extend_dataset(idx, &[n_frames, 2]).unwrap();
        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("frames").unwrap(), vec![10, 2]);

        let raw = reader.read_dataset_raw("frames").unwrap();
        let values: Vec<i32> = raw.chunks(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 20);
        for i in 0..20 {
            assert_eq!(values[i], i as i32);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_fixed_array_dataset_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_fixed_array.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_fixed_array_dataset(
                "grid",
                DatatypeMessage::i32_type(),
                &[4, 6],       // 4x6 grid
                &[2, 3],       // chunk = 2x3
            )
            .unwrap();

        // Write all chunks: 2x2 = 4 chunks
        // chunk (0,0): rows 0-1, cols 0-2
        let c00: Vec<u8> = [0i32, 1, 2, 6, 7, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_fixed_array(idx, &[0, 0], &c00).unwrap();

        // chunk (0,1): rows 0-1, cols 3-5
        let c01: Vec<u8> = [3i32, 4, 5, 9, 10, 11].iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_fixed_array(idx, &[0, 1], &c01).unwrap();

        // chunk (1,0): rows 2-3, cols 0-2
        let c10: Vec<u8> = [12i32, 13, 14, 18, 19, 20].iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_fixed_array(idx, &[1, 0], &c10).unwrap();

        // chunk (1,1): rows 2-3, cols 3-5
        let c11: Vec<u8> = [15i32, 16, 17, 21, 22, 23].iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_fixed_array(idx, &[1, 1], &c11).unwrap();

        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_names(), vec!["grid"]);
        assert_eq!(reader.dataset_shape("grid").unwrap(), vec![4, 6]);

        let raw = reader.read_dataset_raw("grid").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 24);
        for i in 0..24 {
            assert_eq!(values[i], i as i32);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_btree_v2_dataset_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_btree_v2.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_btree_v2_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 0],            // start empty
                &[u64::MAX, u64::MAX], // both dims unlimited
                &[2, 3],            // chunk = 2x3
            )
            .unwrap();

        // Write chunks for a 4x6 dataset
        // chunk (0,0)
        let c00: Vec<u8> = [0.0f64, 1.0, 2.0, 6.0, 7.0, 8.0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_btree_v2(idx, &[0, 0], &c00).unwrap();

        // chunk (0,1)
        let c01: Vec<u8> = [3.0f64, 4.0, 5.0, 9.0, 10.0, 11.0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_btree_v2(idx, &[0, 1], &c01).unwrap();

        // chunk (1,0)
        let c10: Vec<u8> = [12.0f64, 13.0, 14.0, 18.0, 19.0, 20.0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_btree_v2(idx, &[1, 0], &c10).unwrap();

        // chunk (1,1)
        let c11: Vec<u8> = [15.0f64, 16.0, 17.0, 21.0, 22.0, 23.0]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_chunk_btree_v2(idx, &[1, 1], &c11).unwrap();

        writer.extend_dataset(idx, &[4, 6]).unwrap();
        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_names(), vec!["data"]);
        assert_eq!(reader.dataset_shape("data").unwrap(), vec![4, 6]);

        let raw = reader.read_dataset_raw("data").unwrap();
        let values: Vec<f64> = raw
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 24);
        for i in 0..24 {
            assert_eq!(values[i], i as f64);
        }

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_batch_write_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_parallel_batch.h5");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_chunked_dataset(
                "data",
                DatatypeMessage::i32_type(),
                &[0, 4],
                &[u64::MAX, 4],
                &[1, 4],
            )
            .unwrap();

        // Prepare chunks
        let chunks_data: Vec<(u64, Vec<u8>)> = (0..8u64)
            .map(|frame| {
                let values: Vec<i32> = (0..4).map(|i| (frame * 4 + i) as i32).collect();
                let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                (frame, raw)
            })
            .collect();

        let batch: Vec<(u64, &[u8])> = chunks_data
            .iter()
            .map(|(idx, data)| (*idx, data.as_slice()))
            .collect();

        writer.write_chunks_batch(idx, &batch).unwrap();
        writer.extend_dataset(idx, &[8, 4]).unwrap();
        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("data").unwrap(), vec![8, 4]);
        let raw = reader.read_dataset_raw("data").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 32);
        for i in 0..32 {
            assert_eq!(values[i], i as i32);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn swmr_writer_append_frames() {
        use crate::swmr::SwmrWriter;

        let dir = std::env::temp_dir();
        let path = dir.join("test_swmr_append.h5");

        let mut swmr = SwmrWriter::create(&path).unwrap();
        let idx = swmr.create_streaming_dataset(
            "detector",
            DatatypeMessage::u16_type(),
            &[4, 4],
        ).unwrap();

        swmr.start_swmr().unwrap();

        // Append 5 frames
        for frame in 0..5u16 {
            let data: Vec<u16> = (0..16).map(|i| frame * 16 + i).collect();
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            swmr.append_frame(idx, &raw).unwrap();
        }

        swmr.flush().unwrap();
        swmr.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("detector").unwrap(), vec![5, 4, 4]);

        let raw = reader.read_dataset_raw("detector").unwrap();
        let values: Vec<u16> = raw.chunks(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 80); // 5 * 4 * 4
        // Verify first frame
        for i in 0..16 {
            assert_eq!(values[i], i as u16);
        }
        // Verify last frame
        for i in 0..16 {
            assert_eq!(values[64 + i], 4 * 16 + i as u16);
        }

        std::fs::remove_file(&path).ok();
    }
}
