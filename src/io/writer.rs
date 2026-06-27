//! HDF5 file writer.
//!
//! Produces a valid HDF5 file with superblock v3, a root group object header,
//! and datasets with contiguous or chunked storage. The output is readable by `h5dump`.

use std::path::Path;

use crate::format::chunk_index::btree_v2::Bt2ChunkIndex;
use crate::format::chunk_index::extensible_array::{
    compute_chunk_size_len, compute_ndblk_addrs, compute_nsblk_addrs, EaDblkPath, EaGeometry,
    EaLoc, ExtensibleArrayDataBlock, ExtensibleArrayHeader, ExtensibleArrayIndexBlock,
    ExtensibleArraySuperBlock, FilteredChunkEntry, FilteredDataBlock, FilteredIndexBlock,
    EA_CLS_CHUNK, EA_CLS_FILT_CHUNK,
};
use crate::format::chunk_index::fixed_array::{
    encode_filtered_page, encode_unfiltered_page, FixedArrayDataBlock,
    FixedArrayFilteredChunkElement, FixedArrayHeader, FixedArrayPagedPrefix, FA_CLIENT_FILT_CHUNK,
};
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::data_layout::{DataLayoutMessage, EarrayParams, FixedArrayParams};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::fill_value::FillValueMessage;
use crate::format::messages::filter::{self, FilterPipeline};
use crate::format::messages::group_info::GroupInfoMessage;
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::*;
use crate::format::object_header::ObjectHeader;
use crate::format::superblock::*;
use crate::format::{FormatContext, UNDEF_ADDR};

use crate::io::allocator::FileAllocator;
use crate::io::file_handle::FileHandle;
use crate::io::hyperslab::for_each_contiguous_run;
use crate::io::IoResult;

/// On-disk size in bytes of a fixed-array data block, for the layout (paged or
/// flat) implied by `hdr`.
///
/// Mirrors `H5FA_DBLOCK_SIZE` (`H5FApkg.h`):
///   - non-paged: `prefix + nelmts * raw_elmt_size + checksum`
///   - paged: `prefix + page_init_bitmap + nelmts * raw_elmt_size
///     + npages * checksum`, where the prefix checksum covers the bitmap.
///
/// `raw_elmt_size` is `sizeof_addr` for an unfiltered array, and
/// `sizeof_addr + chunk_size_len + 4` (the filtered element: address +
/// compressed size + filter mask) for a filtered array. libhdf5 carries this
/// value as `hdr->cparam.raw_elmt_size`, i.e. exactly `hdr.element_size`.
fn fixed_array_dblk_disk_size(ctx: &FormatContext, hdr: &FixedArrayHeader) -> u64 {
    let elem_size = hdr.element_size as u64;
    let sa = ctx.sizeof_addr as u64;
    let nelmts = hdr.num_elmts;
    // Common metadata prefix: signature(4) + version(1) + client_id(1) + header_addr(sa).
    let meta_prefix = 4 + 1 + 1 + sa;
    if hdr.is_paged() {
        let npages = hdr.npages();
        let bitmap_size = npages.div_ceil(8);
        // prefix (incl. its own 4-byte checksum) + elements + per-page checksums.
        (meta_prefix + bitmap_size + 4) + nelmts * elem_size + npages * 4
    } else {
        // prefix + elements + single 4-byte checksum.
        meta_prefix + nelmts * elem_size + 4
    }
}

/// Encode a fixed-array data block for the layout implied by `hdr`, using the
/// chunk addresses held in `dblk.elements` (unfiltered) or the filtered chunk
/// entries in `dblk.filtered_elements` (filtered, `client_id == 1`).
///
/// For the paged layout (`hdr.is_paged()`), emits the `FADB` prefix with a
/// page-init bitmap followed by `npages` checksummed element pages. A page is
/// marked initialized iff at least one of its chunk addresses is defined,
/// mirroring libhdf5's lazy `H5FA__dblk_page_create`. Uninitialized pages are
/// still written (all `UNDEF_ADDR`, valid checksum) so the file contains no
/// uninitialized bytes; the reader skips them via the bitmap.
fn encode_fixed_array_dblk(
    ctx: &FormatContext,
    hdr: &FixedArrayHeader,
    dblk: &FixedArrayDataBlock,
) -> Vec<u8> {
    let is_filtered = hdr.client_id == FA_CLIENT_FILT_CHUNK;
    let sa = ctx.sizeof_addr as usize;
    // chunk_size_len for filtered entries = element_size - sizeof_addr - 4.
    // libhdf5 carries element_size = sizeof_addr + chunk_size_len + 4.
    let chunk_size_len = (hdr.element_size as usize).saturating_sub(sa + 4);

    if !hdr.is_paged() {
        return if is_filtered {
            dblk.encode_filtered(ctx, chunk_size_len)
        } else {
            dblk.encode_unfiltered(ctx)
        };
    }

    let npages = hdr.npages() as usize;
    let dblk_page_nelmts = hdr.dblk_page_nelmts() as usize;

    // Build the page-init bitmap (MSB-first): a page is initialized iff any of
    // its elements points at a defined address.
    let mut bitmap = vec![0u8; npages.div_ceil(8)];
    let nelmts = if is_filtered {
        dblk.filtered_elements.len()
    } else {
        dblk.elements.len()
    };
    for p in 0..npages {
        let start = p * dblk_page_nelmts;
        let end = ((p + 1) * dblk_page_nelmts).min(nelmts);
        let initialized = if is_filtered {
            dblk.filtered_elements[start..end]
                .iter()
                .any(|e| e.address != UNDEF_ADDR)
        } else {
            dblk.elements[start..end].iter().any(|&a| a != UNDEF_ADDR)
        };
        if initialized {
            bitmap[p / 8] |= 0x80u8 >> (p % 8);
        }
    }

    let prefix = FixedArrayPagedPrefix {
        client_id: hdr.client_id,
        header_addr: dblk.header_addr,
        page_init_bitmap: bitmap,
        prefix_size: 4 + 1 + 1 + sa + npages.div_ceil(8) + 4,
    };

    let mut buf = prefix.encode(ctx);
    debug_assert_eq!(buf.len(), prefix.prefix_size);

    // Append each page: all pages use the full `dblk_page_nelmts` stride;
    // only the last page holds fewer elements (libhdf5 H5FA.c).
    for p in 0..npages {
        let start = p * dblk_page_nelmts;
        let end = ((p + 1) * dblk_page_nelmts).min(nelmts);
        if is_filtered {
            buf.extend_from_slice(&encode_filtered_page(
                &dblk.filtered_elements[start..end],
                ctx,
                chunk_size_len,
            ));
        } else {
            buf.extend_from_slice(&encode_unfiltered_page(&dblk.elements[start..end], ctx));
        }
    }
    buf
}

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
    /// Buffer for partially filled chunks during append.
    pub append_buffer: Vec<u8>,
    /// Number of frames accumulated in `append_buffer`.
    pub append_buffered_frames: u64,
    /// Soft-deleted: excluded from finalize output.
    pub deleted: bool,
    /// User-defined fill value bytes (exactly one element wide). `None`
    /// means default zero-fill; `Some` is emitted as a `fill_defined = 2`
    /// fill-value message in the dataset object header.
    pub fill_value: Option<Vec<u8>>,
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
    /// In-memory copy of the EA index block (for unfiltered datasets).
    pub ea_iblk: ExtensibleArrayIndexBlock,
    /// Number of chunks written so far.
    pub chunks_written: u64,
    /// Filtered index block (for compressed datasets).
    pub filt_iblk: Option<FilteredIndexBlock>,
    /// chunk_size_len for filtered entries.
    pub chunk_size_len: u8,
}

/// Where a newly-created EA data block's address must be recorded.
enum DblkParent {
    /// Slot `index_block.dblk_addrs[idx]`.
    IndexBlock(usize),
    /// Slot `super_block.dblk_addrs[local_dblk]` of the super block at `sblk_addr`.
    SuperBlock {
        sblk_addr: u64,
        ndblks_in_sblk: usize,
        local_dblk: usize,
    },
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

/// Metadata for a group being written.
pub struct GroupInfo {
    /// Full path of this group (e.g. "/detector" or "/detector/raw").
    pub name: String,
    /// Index of the parent group in the groups vec, or None for root-level groups.
    pub parent: Option<usize>,
    /// Indices of child datasets (into `datasets` vec).
    pub child_datasets: Vec<usize>,
    /// Indices of child groups (into `groups` vec).
    pub child_groups: Vec<usize>,
    /// File offset of this group's object header (set during finalize).
    pub obj_header_addr: u64,
    /// Soft-deleted: excluded from finalize output.
    pub deleted: bool,
    /// Attributes attached to this group (e.g. NeXus `NX_class`).
    pub attributes: Vec<AttributeMessage>,
}

/// The object a [`HardLink`] resolves to.
#[derive(Clone, Copy)]
pub enum HardLinkTarget {
    /// Index into the writer's `datasets` vec.
    Dataset(usize),
    /// Index into the writer's `groups` vec.
    Group(usize),
}

/// A user-created hard link: an additional name, in some group, for an
/// object that already exists under its own name.
///
/// The HDF5 file format makes every group entry a `name -> object header
/// address` mapping, so a hard link is just a second such entry pointing at
/// an already-written object. No data is copied.
pub struct HardLink {
    /// Parent group index (`None` = the root group).
    pub parent: Option<usize>,
    /// Leaf name of the link within the parent group.
    pub name: String,
    /// Object this link resolves to.
    pub target: HardLinkTarget,
}

/// Encode an Object Reference Count message (type 0x16) body: a version
/// byte (`H5O_REFCOUNT_VERSION` = 0) followed by the little-endian u32
/// count. Emitted on objects reached by more than one hard link.
fn encode_refcount(refcount: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(5);
    v.push(0u8);
    v.extend_from_slice(&refcount.to_le_bytes());
    v
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
    pub(crate) groups: Vec<GroupInfo>,
    /// User-created hard links (additional names for existing objects),
    /// resolved and emitted during finalize.
    pub(crate) hard_links: Vec<HardLink>,
    /// Attributes attached to the root group (file-level attributes).
    pub(crate) root_attributes: Vec<crate::format::messages::attribute::AttributeMessage>,
    closed: bool,
    /// Address of the root group object header (set after first finalize).
    root_group_addr: Option<u64>,
    /// Size of the encoded root group object header (for in-place rewrites).
    root_group_encoded_size: usize,
}

impl Hdf5Writer {
    /// Create a new HDF5 file at `path` using the env-var-derived locking
    /// policy (controlled by `HDF5_USE_FILE_LOCKING`).
    ///
    /// The superblock (48 bytes for v3 with 8-byte offsets) is reserved at
    /// offset 0 and written during `close()`.
    pub fn create(path: &Path) -> IoResult<Self> {
        Self::create_with_locking(
            path,
            crate::io::locking::FileLocking::from_env_or(Default::default()),
        )
    }

    /// Create a new HDF5 file at `path` with an explicit locking policy.
    pub fn create_with_locking(
        path: &Path,
        locking: crate::io::locking::FileLocking,
    ) -> IoResult<Self> {
        let handle = FileHandle::create_with_locking(path, locking)?;
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
            groups: Vec::new(),
            hard_links: Vec::new(),
            root_attributes: Vec::new(),
            closed: false,
            root_group_addr: None,
            root_group_encoded_size: 0,
        })
    }

    /// Provide public access to the format context.
    pub fn ctx(&self) -> &FormatContext {
        &self.ctx
    }

    /// Open an existing HDF5 file for appending new datasets, using the
    /// env-var-derived locking policy.
    ///
    /// Reads existing dataset object headers fully, reconstructing metadata
    /// for chunked datasets so that `write_chunk` and `extend_dataset` work
    /// on reopened datasets.
    pub fn open_append(path: &Path) -> IoResult<Self> {
        Self::open_append_with_locking(
            path,
            crate::io::locking::FileLocking::from_env_or(Default::default()),
        )
    }

    /// Open an existing HDF5 file for appending with an explicit locking
    /// policy.
    pub fn open_append_with_locking(
        path: &Path,
        locking: crate::io::locking::FileLocking,
    ) -> IoResult<Self> {
        use crate::format::messages::attribute::AttributeMessage;
        use crate::format::messages::data_layout::DataLayoutMessage;
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::datatype::DatatypeMessage;

        let mut handle = FileHandle::open_readwrite_with_locking(path, locking)?;
        let file_size = handle.file_size()?;

        let sb_buf = handle.read_at_most(0, 256)?;
        // open_append reconstructs writer state from the file's link/chunk
        // structures, which this crate only writes in the version-2/3
        // (v18+) format. A classic v0/v1-superblock file (e.g. h5py's
        // default `libver`) uses symbol-table groups and v1-B-tree chunk
        // indexes that the append path cannot rebuild — reject it with a
        // clear message rather than the cryptic version error, and without
        // touching the file.
        if matches!(
            crate::format::superblock::detect_superblock_version(&sb_buf),
            Ok(0) | Ok(1)
        ) {
            return Err(crate::io::IoError::InvalidState(
                "cannot open this file for appending: it uses the classic \
                 (version-0/1 superblock) HDF5 format; re-create it with a \
                 newer library-version bound to append to it"
                    .into(),
            ));
        }
        let sb = SuperblockV2V3::decode(&sb_buf)?;
        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Discover links from root group (and subgroups recursively).
        // Read to end-of-file so a large object header (many attributes) is
        // not truncated, which would silently drop datasets on reopen.
        let root_addr = sb.root_group_object_header_address;
        let root_buf =
            handle.read_at_most(root_addr, file_size.saturating_sub(root_addr) as usize)?;
        let (root_header, _) = crate::format::object_header::ObjectHeader::decode(&root_buf)?;

        // Collect existing root-level attributes
        let mut root_attributes = Vec::new();
        for msg in &root_header.messages {
            if msg.msg_type == crate::format::messages::MSG_ATTRIBUTE {
                if let Ok((a, _)) =
                    crate::format::messages::attribute::AttributeMessage::decode(&msg.data, &ctx)
                {
                    root_attributes.push(a);
                }
            }
        }

        let mut link_entries: Vec<(String, u64)> = Vec::new();
        let mut visited_groups = std::collections::HashSet::new();
        Self::collect_links_recursive(
            &mut handle,
            &root_header,
            &ctx,
            "",
            &mut link_entries,
            &mut visited_groups,
            0,
        )?;

        let mut existing_datasets = Vec::new();
        for (name, obj_addr) in &link_entries {
            // Read the dataset's full object header (to EOF — see above).
            let ds_buf =
                handle.read_at_most(*obj_addr, file_size.saturating_sub(*obj_addr) as usize)?;
            let (ds_header, _) =
                match crate::format::object_header::ObjectHeader::decode_any(&ds_buf) {
                    Ok(h) => h,
                    Err(_) => continue,
                };

            let mut datatype = None;
            let mut dataspace = None;
            let mut layout = None;
            let mut fp = None;
            let mut fill_value = None;
            let mut attrs = Vec::new();

            for msg in &ds_header.messages {
                match msg.msg_type {
                    crate::format::messages::MSG_DATATYPE => {
                        if let Ok((dt, _)) = DatatypeMessage::decode(&msg.data, &ctx) {
                            datatype = Some(dt);
                        }
                    }
                    crate::format::messages::MSG_DATASPACE => {
                        if let Ok((ds, _)) = DataspaceMessage::decode(&msg.data, &ctx) {
                            dataspace = Some(ds);
                        }
                    }
                    crate::format::messages::MSG_DATA_LAYOUT => {
                        if let Ok((dl, _)) = DataLayoutMessage::decode(&msg.data, &ctx) {
                            layout = Some(dl);
                        }
                    }
                    crate::format::messages::MSG_FILTER_PIPELINE => {
                        if let Ok((p, _)) = FilterPipeline::decode(&msg.data) {
                            if !p.filters.is_empty() {
                                fp = Some(p);
                            }
                        }
                    }
                    crate::format::messages::MSG_FILL_VALUE => {
                        if let Ok((fv, _)) = FillValueMessage::decode(&msg.data) {
                            if fv.fill_defined == 2 {
                                fill_value = fv.fill_value;
                            }
                        }
                    }
                    crate::format::messages::MSG_ATTRIBUTE => {
                        if let Ok((a, _)) = AttributeMessage::decode(&msg.data, &ctx) {
                            attrs.push(a);
                        }
                    }
                    _ => {}
                }
            }

            let (dt, ds, dl) = match (datatype, dataspace, layout) {
                (Some(dt), Some(ds), Some(dl)) => (dt, ds, dl),
                _ => continue, // Not a dataset (probably a group)
            };

            let mut info = DatasetInfo {
                name: name.clone(),
                datatype: dt,
                dataspace: ds,
                obj_header_addr: *obj_addr,
                data_addr: UNDEF_ADDR,
                data_size: 0,
                chunked: None,
                fixed_array: None,
                btree_v2: None,
                attributes: attrs,
                obj_header_written_addr: Some(*obj_addr),
                obj_header_encoded_size: 0,
                filter_pipeline: fp,
                append_buffer: Vec::new(),
                append_buffered_frames: 0,
                deleted: false,
                fill_value,
            };

            // Reconstruct storage-specific metadata
            match &dl {
                DataLayoutMessage::Contiguous { address, size } => {
                    info.data_addr = *address;
                    info.data_size = *size;
                }
                DataLayoutMessage::ChunkedV4 {
                    chunk_dims,
                    index_address,
                    index_type,
                    earray_params,
                    ..
                } => {
                    let real_chunk_dims: Vec<u64> = chunk_dims[..chunk_dims.len() - 1].to_vec();

                    if *index_type
                        == crate::format::messages::data_layout::ChunkIndexType::ExtensibleArray
                    {
                        if let Some(params) = earray_params {
                            let ep = EarrayParams {
                                max_nelmts_bits: params.max_nelmts_bits,
                                idx_blk_elmts: params.idx_blk_elmts,
                                sup_blk_min_data_ptrs: params.sup_blk_min_data_ptrs,
                                data_blk_min_elmts: params.data_blk_min_elmts,
                                max_dblk_page_nelmts_bits: params.max_dblk_page_nelmts_bits,
                            };
                            let ndblk_addrs = compute_ndblk_addrs(ep.sup_blk_min_data_ptrs)?;
                            let nsblk_addrs = compute_nsblk_addrs(
                                ep.idx_blk_elmts,
                                ep.data_blk_min_elmts,
                                ep.sup_blk_min_data_ptrs,
                                ep.max_nelmts_bits,
                            )?;

                            // Read EA header
                            let hdr_buf = handle.read_at_most(*index_address, 256)?;
                            let ea_header = ExtensibleArrayHeader::decode(&hdr_buf, &ctx)?;

                            let is_filtered = ea_header.class_id
                                == crate::format::chunk_index::extensible_array::EA_CLS_FILT_CHUNK;
                            let chunk_size_len = if is_filtered {
                                ea_header.raw_elmt_size - ctx.sizeof_addr - 4
                            } else {
                                0
                            };

                            // Read the EA index block. Filtered datasets
                            // store a `FilteredIndexBlock`; unfiltered ones a
                            // plain `ExtensibleArrayIndexBlock`. Both must be
                            // reconstructed so a reopened dataset can append
                            // (write_chunk consults whichever applies).
                            let ea_iblk_addr = ea_header.idx_blk_addr;
                            let (ea_iblk, filt_iblk) = if is_filtered {
                                let placeholder = ExtensibleArrayIndexBlock::new(
                                    *index_address,
                                    ep.idx_blk_elmts,
                                    ndblk_addrs,
                                    nsblk_addrs,
                                );
                                let fib = if ea_iblk_addr != UNDEF_ADDR {
                                    let iblk_buf = handle.read_at_most(ea_iblk_addr, 65536)?;
                                    FilteredIndexBlock::decode(
                                        &iblk_buf,
                                        &ctx,
                                        ep.idx_blk_elmts as usize,
                                        ndblk_addrs,
                                        nsblk_addrs,
                                        chunk_size_len,
                                    )
                                    .unwrap_or_else(|_| {
                                        FilteredIndexBlock::new(
                                            *index_address,
                                            ep.idx_blk_elmts,
                                            ndblk_addrs,
                                            nsblk_addrs,
                                        )
                                    })
                                } else {
                                    FilteredIndexBlock::new(
                                        *index_address,
                                        ep.idx_blk_elmts,
                                        ndblk_addrs,
                                        nsblk_addrs,
                                    )
                                };
                                (placeholder, Some(fib))
                            } else {
                                let eib = if ea_iblk_addr != UNDEF_ADDR {
                                    let iblk_buf = handle.read_at_most(ea_iblk_addr, 65536)?;
                                    ExtensibleArrayIndexBlock::decode(
                                        &iblk_buf,
                                        &ctx,
                                        ep.idx_blk_elmts as usize,
                                        ndblk_addrs,
                                        nsblk_addrs,
                                    )
                                    .unwrap_or_else(|_| {
                                        ExtensibleArrayIndexBlock::new(
                                            *index_address,
                                            ep.idx_blk_elmts,
                                            ndblk_addrs,
                                            nsblk_addrs,
                                        )
                                    })
                                } else {
                                    ExtensibleArrayIndexBlock::new(
                                        *index_address,
                                        ep.idx_blk_elmts,
                                        ndblk_addrs,
                                        nsblk_addrs,
                                    )
                                };
                                (eib, None)
                            };

                            let max_dims = info
                                .dataspace
                                .max_dims
                                .clone()
                                .unwrap_or_else(|| info.dataspace.dims.clone());

                            info.chunked = Some(ChunkedDatasetInfo {
                                chunk_dims: real_chunk_dims,
                                max_dims,
                                earray_params: ep,
                                ea_header_addr: *index_address,
                                ea_iblk_addr,
                                ndblk_addrs,
                                ea_header,
                                ea_iblk,
                                chunks_written: 0,
                                filt_iblk,
                                chunk_size_len,
                            });
                        }
                    }
                    // FA/BT2 datasets remain as placeholder (re-link only)
                }
                _ => {}
            }

            existing_datasets.push(info);
        }

        // Reconstruct group structure from dataset paths.
        // e.g. dataset "nodes/id" implies group "/nodes" exists.
        let mut groups: Vec<GroupInfo> = Vec::new();
        let mut group_index_map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for (di, ds) in existing_datasets.iter().enumerate() {
            let parts: Vec<&str> = ds.name.split('/').collect();
            if parts.len() <= 1 {
                continue; // root-level dataset, no group
            }
            // Build group hierarchy: e.g. "a/b/c" → groups "/a", "/a/b"
            let mut path = String::new();
            for part in &parts[..parts.len() - 1] {
                let parent_path = if path.is_empty() {
                    "/".to_string()
                } else {
                    path.clone()
                };
                if path.is_empty() {
                    path = format!("/{}", part);
                } else {
                    path = format!("{}/{}", path, part);
                }
                if group_index_map.contains_key(&path) {
                    continue;
                }
                let parent = if parent_path == "/" {
                    None
                } else {
                    group_index_map.get(&parent_path).copied()
                };
                let gidx = groups.len();
                groups.push(GroupInfo {
                    name: path.clone(),
                    parent,
                    child_datasets: Vec::new(),
                    child_groups: Vec::new(),
                    obj_header_addr: 0,
                    deleted: false,
                    attributes: Vec::new(),
                });
                if let Some(pidx) = parent {
                    groups[pidx].child_groups.push(gidx);
                }
                group_index_map.insert(path.clone(), gidx);
            }
            // Assign dataset to its immediate parent group
            let parent_path = if parts.len() == 2 {
                format!("/{}", parts[0])
            } else {
                format!("/{}", parts[..parts.len() - 1].join("/"))
            };
            if let Some(&gidx) = group_index_map.get(&parent_path) {
                groups[gidx].child_datasets.push(di);
            }
        }

        let allocator = FileAllocator::new(file_size);

        Ok(Self {
            handle,
            allocator,
            ctx,
            datasets: existing_datasets,
            groups,
            hard_links: Vec::new(),
            root_attributes,
            closed: false,
            root_group_addr: None,
            root_group_encoded_size: 0,
        })
    }

    /// Recursively collect (name, obj_header_addr) pairs from link messages.
    fn collect_links_recursive(
        handle: &mut FileHandle,
        header: &crate::format::object_header::ObjectHeader,
        ctx: &FormatContext,
        prefix: &str,
        out: &mut Vec<(String, u64)>,
        visited: &mut std::collections::HashSet<u64>,
        depth: usize,
    ) -> IoResult<()> {
        // Bound nesting depth so a pathologically deep group chain cannot
        // overflow the stack (the `visited` set bounds total work but not
        // recursion depth).
        if depth > 256 {
            return Ok(());
        }
        use crate::format::messages::link::{LinkMessage, LinkTarget};
        for msg in &header.messages {
            if msg.msg_type == crate::format::messages::MSG_LINK {
                if let Ok((link, _)) = LinkMessage::decode(&msg.data, ctx) {
                    if let LinkTarget::Hard { address } = &link.target {
                        let full_name = if prefix.is_empty() {
                            link.name.clone()
                        } else {
                            format!("{}/{}", prefix, link.name)
                        };
                        out.push((full_name.clone(), *address));

                        // Try to recurse into groups (read to EOF so a large
                        // child object header is not truncated).
                        let child_len = handle
                            .file_size()
                            .map(|fs| fs.saturating_sub(*address) as usize)
                            .unwrap_or(8192);
                        if let Ok(child_buf) = handle.read_at_most(*address, child_len) {
                            if let Ok((child_header, _)) =
                                crate::format::object_header::ObjectHeader::decode_any(&child_buf)
                            {
                                let has_links = child_header
                                    .messages
                                    .iter()
                                    .any(|m| m.msg_type == crate::format::messages::MSG_LINK);
                                // Recurse only into a group's header we have
                                // not entered before — breaks hard-link cycles.
                                if has_links && visited.insert(*address) {
                                    let _ = Self::collect_links_recursive(
                                        handle,
                                        &child_header,
                                        ctx,
                                        &full_name,
                                        out,
                                        visited,
                                        depth + 1,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Return the names of all datasets created so far.
    pub fn dataset_names(&self) -> Vec<&str> {
        self.datasets
            .iter()
            .filter(|d| !d.deleted)
            .map(|d| d.name.as_str())
            .collect()
    }

    /// Find a dataset index by name.
    pub fn dataset_index(&self, name: &str) -> Option<usize> {
        self.datasets
            .iter()
            .position(|d| d.name == name && !d.deleted)
    }

    /// Reconstruct the fields a writer-mode `H5Dataset` handle needs for the
    /// dataset at `index`: `(shape, element_size, chunked, btree2,
    /// fixed_array)`. Single owner of this mapping so `H5File::dataset_writer`,
    /// `H5Group::dataset_writer`, and the vlen-string helpers all agree.
    pub(crate) fn dataset_handle_parts(
        &self,
        index: usize,
    ) -> (Vec<usize>, usize, bool, bool, bool) {
        let ds = &self.datasets[index];
        let shape: Vec<usize> = ds.dataspace.dims.iter().map(|&d| d as usize).collect();
        let element_size = ds.datatype.element_size() as usize;
        let fixed_array = ds.fixed_array.is_some();
        let btree2 = ds.btree_v2.is_some();
        let chunked = ds.chunked.is_some() || fixed_array || btree2;
        (shape, element_size, chunked, btree2, fixed_array)
    }

    /// Reject a dataset name already used by a live dataset. Dataset names
    /// here are full paths, so they must be unique across the file (HDF5
    /// requires link names to be unique within their group).
    fn ensure_unique_dataset_name(&self, name: &str) -> IoResult<()> {
        if self.datasets.iter().any(|d| !d.deleted && d.name == name) {
            return Err(crate::io::IoError::InvalidState(format!(
                "a dataset named '{name}' already exists"
            )));
        }
        if self
            .hard_links
            .iter()
            .any(|l| self.hard_link_emitted(l) && self.hard_link_full_path(l) == name)
        {
            return Err(crate::io::IoError::InvalidState(format!(
                "a hard link named '{name}' already exists"
            )));
        }
        Ok(())
    }

    /// Soft-delete a dataset by name. The dataset is excluded from the file
    /// on close. File space is not reclaimed.
    pub fn delete_dataset(&mut self, name: &str) -> IoResult<()> {
        let idx = self
            .datasets
            .iter()
            .position(|d| d.name == name && !d.deleted)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        self.datasets[idx].deleted = true;
        // Remove from parent group's child_datasets
        for grp in &mut self.groups {
            grp.child_datasets.retain(|&di| di != idx);
        }
        Ok(())
    }

    /// Soft-delete a group and all its child datasets and sub-groups.
    /// File space is not reclaimed.
    pub fn delete_group(&mut self, name: &str) -> IoResult<()> {
        let name = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{}", name)
        };
        let gidx = self
            .groups
            .iter()
            .position(|g| g.name == name && !g.deleted)
            .ok_or_else(|| crate::io::IoError::NotFound(name.clone()))?;
        self.delete_group_recursive(gidx);
        // Remove from parent's child_groups
        if let Some(pidx) = self.groups[gidx].parent {
            self.groups[pidx].child_groups.retain(|&gi| gi != gidx);
        }
        Ok(())
    }

    fn delete_group_recursive(&mut self, gidx: usize) {
        self.groups[gidx].deleted = true;
        // Delete child datasets
        let child_ds: Vec<usize> = self.groups[gidx].child_datasets.clone();
        for di in child_ds {
            self.datasets[di].deleted = true;
        }
        // Recurse into child groups
        let child_gs: Vec<usize> = self.groups[gidx].child_groups.clone();
        for gi in child_gs {
            self.delete_group_recursive(gi);
        }
    }

    /// Return the chunk dimensions for a dataset, if chunked.
    pub fn dataset_chunk_dims(&self, index: usize) -> Option<&[u64]> {
        let ds = &self.datasets[index];
        if let Some(ref c) = ds.chunked {
            Some(&c.chunk_dims)
        } else if let Some(ref f) = ds.fixed_array {
            Some(&f.chunk_dims)
        } else if let Some(ref b) = ds.btree_v2 {
            Some(&b.chunk_dims)
        } else {
            None
        }
    }

    /// Return the current dimensions of a dataset.
    pub fn dataset_dims(&self, index: usize) -> &[u64] {
        &self.datasets[index].dataspace.dims
    }

    /// Return the names of all groups created so far.
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.iter().map(|g| g.name.as_str()).collect()
    }

    /// Create a group in the file hierarchy.
    ///
    /// `parent_path` is the full path of the parent group (e.g., "/" for root).
    /// `name` is the name of the new group (e.g., "detector").
    ///
    /// Returns the group index in the writer's group list.
    pub fn create_group(&mut self, parent_path: &str, name: &str) -> IoResult<usize> {
        let full_name = if parent_path == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", parent_path, name)
        };

        // Check for duplicates (ignore deleted groups)
        if self
            .groups
            .iter()
            .any(|g| g.name == full_name && !g.deleted)
        {
            return Err(crate::io::IoError::InvalidState(format!(
                "group '{}' already exists",
                full_name
            )));
        }
        // A hard link must not already occupy this name in its parent.
        let full_rel = full_name.trim_start_matches('/');
        if self
            .hard_links
            .iter()
            .any(|l| self.hard_link_emitted(l) && self.hard_link_full_path(l) == full_rel)
        {
            return Err(crate::io::IoError::InvalidState(format!(
                "a hard link named '{full_name}' already exists"
            )));
        }

        // Find parent group index (None means it's a root-level group)
        let parent_idx = if parent_path == "/" {
            None
        } else {
            let idx = self
                .groups
                .iter()
                .position(|g| g.name == parent_path)
                .ok_or_else(|| {
                    crate::io::IoError::NotFound(format!(
                        "parent group '{}' not found",
                        parent_path
                    ))
                })?;
            Some(idx)
        };

        let group_idx = self.groups.len();
        self.groups.push(GroupInfo {
            name: full_name,
            parent: parent_idx,
            child_datasets: Vec::new(),
            child_groups: Vec::new(),
            obj_header_addr: 0,
            deleted: false,
            attributes: Vec::new(),
        });

        // Register this group as a child of its parent
        if let Some(pidx) = parent_idx {
            self.groups[pidx].child_groups.push(group_idx);
        }

        Ok(group_idx)
    }

    /// Register a dataset as belonging to a group.
    ///
    /// `group_path` is the full path of the group (e.g., "/detector").
    /// `ds_index` is the dataset index returned by `create_dataset`.
    pub fn assign_dataset_to_group(&mut self, group_path: &str, ds_index: usize) -> IoResult<()> {
        let group_idx = self
            .groups
            .iter()
            .position(|g| g.name == group_path)
            .ok_or_else(|| {
                crate::io::IoError::NotFound(format!("group '{}' not found", group_path))
            })?;
        self.groups[group_idx].child_datasets.push(ds_index);
        Ok(())
    }

    /// Create a hard link: an additional name for an object that already
    /// exists in the file.
    ///
    /// No data is copied — the link and its target share one object header,
    /// exactly as `h5py` / libhdf5 hard links do.
    ///
    /// * `parent_group_path` — full path of the group that will hold the
    ///   link (`"/"` for the root group).
    /// * `link_name` — leaf name of the new link within that group.
    /// * `target_path` — full path of an existing dataset or group, with or
    ///   without a leading `/`.
    pub fn create_hard_link(
        &mut self,
        parent_group_path: &str,
        link_name: &str,
        target_path: &str,
    ) -> IoResult<()> {
        if link_name.is_empty() || link_name.contains('/') {
            return Err(crate::io::IoError::InvalidState(format!(
                "hard link name '{link_name}' must be a non-empty leaf name"
            )));
        }

        // Resolve the parent group (None == root).
        let parent = if parent_group_path == "/" {
            None
        } else {
            Some(
                self.groups
                    .iter()
                    .position(|g| g.name == parent_group_path && !g.deleted)
                    .ok_or_else(|| {
                        crate::io::IoError::NotFound(format!(
                            "parent group '{parent_group_path}' not found"
                        ))
                    })?,
            )
        };

        // Resolve the target. Dataset names are stored without a leading
        // '/', group names with one — compare on the trimmed form. A
        // trailing '/' is tolerated too.
        let target_rel = target_path.trim_matches('/');
        if target_rel.is_empty() {
            return Err(crate::io::IoError::InvalidState(
                "cannot hard-link the root group".into(),
            ));
        }
        let target = if let Some(idx) = self
            .datasets
            .iter()
            .position(|d| !d.deleted && d.name.trim_start_matches('/') == target_rel)
        {
            HardLinkTarget::Dataset(idx)
        } else if let Some(idx) = self
            .groups
            .iter()
            .position(|g| !g.deleted && g.name.trim_start_matches('/') == target_rel)
        {
            HardLinkTarget::Group(idx)
        } else {
            return Err(crate::io::IoError::NotFound(format!(
                "hard link target '{target_path}' not found"
            )));
        };

        // Reject a name already taken in the parent group.
        let parent_prefix = match parent {
            None => String::new(),
            Some(pi) => format!("{}/", self.groups[pi].name.trim_start_matches('/')),
        };
        let full = format!("{parent_prefix}{link_name}");
        let collides = self
            .datasets
            .iter()
            .any(|d| !d.deleted && d.name.trim_start_matches('/') == full)
            || self
                .groups
                .iter()
                .any(|g| !g.deleted && g.name.trim_start_matches('/') == full)
            || self
                .hard_links
                .iter()
                .any(|l| l.parent == parent && l.name == link_name);
        if collides {
            return Err(crate::io::IoError::InvalidState(format!(
                "'{full}' already exists in the file"
            )));
        }

        self.hard_links.push(HardLink {
            parent,
            name: link_name.to_string(),
            target,
        });
        Ok(())
    }

    /// Whether a hard link will actually be emitted: both its parent group
    /// and its target object must still be present (not soft-deleted).
    fn hard_link_emitted(&self, link: &HardLink) -> bool {
        let parent_ok = match link.parent {
            None => true,
            Some(pi) => !self.groups[pi].deleted,
        };
        let target_ok = match link.target {
            HardLinkTarget::Dataset(i) => !self.datasets[i].deleted,
            HardLinkTarget::Group(i) => !self.groups[i].deleted,
        };
        parent_ok && target_ok
    }

    /// The full path a hard link occupies, with no leading `/` — the same
    /// form dataset names are stored in. Used for name-collision checks.
    fn hard_link_full_path(&self, link: &HardLink) -> String {
        match link.parent {
            None => link.name.clone(),
            Some(pi) => format!(
                "{}/{}",
                self.groups[pi].name.trim_start_matches('/'),
                link.name
            ),
        }
    }

    /// Total number of hard links resolving to an object: its own tree link
    /// plus every emitted user-created hard link pointing at it.
    fn object_link_count(&self, target: HardLinkTarget) -> u32 {
        let same = |a: HardLinkTarget, b: HardLinkTarget| -> bool {
            matches!(
                (a, b),
                (HardLinkTarget::Dataset(x), HardLinkTarget::Dataset(y))
                    | (HardLinkTarget::Group(x), HardLinkTarget::Group(y))
                if x == y
            )
        };
        1 + self
            .hard_links
            .iter()
            .filter(|l| self.hard_link_emitted(l) && same(l.target, target))
            .count() as u32
    }

    /// Append a `MSG_LINK` message for every user-created hard link whose
    /// parent group is `parent` (`None` == the root group). Called while
    /// building group object headers, once every object's header address
    /// has been assigned.
    fn emit_hard_links(&self, header: &mut ObjectHeader, parent: Option<usize>) {
        for link in &self.hard_links {
            if link.parent != parent || !self.hard_link_emitted(link) {
                continue;
            }
            let addr = match link.target {
                HardLinkTarget::Dataset(i) => self.datasets[i].obj_header_addr,
                HardLinkTarget::Group(i) => self.groups[i].obj_header_addr,
            };
            let msg = LinkMessage::hard(&link.name, addr);
            header.add_message(MSG_LINK, 0x00, msg.encode(&self.ctx));
        }
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
        self.ensure_unique_dataset_name(name)?;
        let total_elements: u64 = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };
        let element_size = datatype.element_size() as u64;
        let data_size = total_elements * element_size;

        // Allocate space for the raw data.
        let data_addr = if data_size > 0 {
            self.allocator.allocate(data_size)
        } else {
            UNDEF_ADDR
        };

        let dataspace = if dims.is_empty() {
            DataspaceMessage::scalar()
        } else {
            DataspaceMessage::simple(dims)
        };

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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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
        self.ensure_unique_dataset_name(name)?;
        let earray_params = EarrayParams::default_params();
        let ndblk_addrs = compute_ndblk_addrs(earray_params.sup_blk_min_data_ptrs)?;
        let nsblk_addrs = compute_nsblk_addrs(
            earray_params.idx_blk_elmts,
            earray_params.data_blk_min_elmts,
            earray_params.sup_blk_min_data_ptrs,
            earray_params.max_nelmts_bits,
        )?;

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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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
                chunks_written: 0,
                filt_iblk: None,
                chunk_size_len: 0,
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
            return Err(crate::io::IoError::InvalidState(
                "use write_chunk for chunked datasets".into(),
            ));
        }
        if ds.data_addr == UNDEF_ADDR {
            return Err(crate::io::IoError::InvalidState(
                "dataset has no data allocated".into(),
            ));
        }
        if data.len() as u64 != ds.data_size {
            return Err(crate::io::IoError::InvalidState(format!(
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
    pub fn write_chunk(&mut self, index: usize, chunk_idx: u64, data: &[u8]) -> IoResult<()> {
        let ds = &self.datasets[index];
        let element_size = ds.datatype.element_size() as u64;
        let chunked = ds
            .chunked
            .as_ref()
            .ok_or_else(|| crate::io::IoError::InvalidState("not a chunked dataset".into()))?;
        let chunk_bytes: u64 = chunked.chunk_dims.iter().product::<u64>() * element_size;

        if data.len() as u64 != chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes,
                data.len()
            )));
        }

        // Apply compression if filter pipeline is set
        let compressed;
        let write_data = if let Some(ref pipeline) = ds.filter_pipeline {
            compressed = filter::apply_filters(pipeline, data)?;
            &compressed
        } else {
            data
        };
        let compressed_size = write_data.len() as u64;

        // Allocate space for the chunk data
        let chunk_addr = self.allocator.allocate(compressed_size);
        self.handle.write_at(chunk_addr, write_data)?;

        // filter_mask = 0: this path runs the whole pipeline, so no filter is
        // skipped for the chunk.
        self.record_ea_chunk(index, chunk_idx, chunk_addr, compressed_size, 0)
    }

    /// Record a written chunk in the extensible-array index, placing
    /// its address (and compressed size, for filtered datasets) into
    /// the index block, a data block, or a super block per the EA
    /// geometry. Shared by write_chunk and write_compressed_chunk.
    fn record_ea_chunk(
        &mut self,
        index: usize,
        chunk_idx: u64,
        chunk_addr: u64,
        compressed_size: u64,
        filter_mask: u32,
    ) -> IoResult<()> {
        let is_filtered = self.datasets[index].filter_pipeline.is_some();
        // For a filtered dataset the chunk's stored size is encoded in the
        // `chunk_size_len`-byte field of each filtered EA entry
        // (`FilteredChunkEntry::encode` writes `nbytes[..chunk_size_len]`,
        // which truncates silently). Reject a size that would not fit, the way
        // libhdf5's H5D_CHUNK_ENCODE_SIZE_CHECK does, instead of corrupting the
        // index. The compress path never exceeds this (chunk_size_len holds the
        // uncompressed chunk size); a direct/raw write with caller-supplied
        // bytes can.
        if is_filtered {
            let chunk_size_len = self.datasets[index]
                .chunked
                .as_ref()
                .unwrap()
                .chunk_size_len as usize;
            if chunk_size_len < 8 && compressed_size >= (1u64 << (chunk_size_len * 8)) {
                return Err(crate::io::IoError::InvalidState(format!(
                    "filtered chunk size {compressed_size} does not fit in the \
                     {chunk_size_len}-byte extensible-array chunk-size field"
                )));
            }
        }
        let idx_blk_elmts = {
            let c = self.datasets[index].chunked.as_ref().unwrap();
            c.earray_params.idx_blk_elmts as u64
        };

        if chunk_idx < idx_blk_elmts {
            let chunked = self.datasets[index].chunked.as_mut().unwrap();
            if is_filtered {
                if let Some(ref mut fiblk) = chunked.filt_iblk {
                    fiblk.elements[chunk_idx as usize] = FilteredChunkEntry {
                        addr: chunk_addr,
                        nbytes: compressed_size,
                        filter_mask,
                    };
                }
            } else {
                chunked.ea_iblk.elements[chunk_idx as usize] = chunk_addr;
            }
            chunked.chunks_written += 1;
            if chunk_idx + 1 > chunked.ea_header.max_idx_set {
                chunked.ea_header.max_idx_set = chunk_idx + 1;
            }
            if chunked.ea_header.num_elmts_realized < idx_blk_elmts {
                chunked.ea_header.num_elmts_realized = idx_blk_elmts;
            }
        } else {
            // chunk_idx >= idx_blk_elmts: place the chunk through the EA
            // data-block / super-block hierarchy (libhdf5-compatible geometry).
            let (geo, max_nelmts_bits, chunk_size_len, ea_header_addr) = {
                let c = self.datasets[index].chunked.as_ref().unwrap();
                let p = &c.earray_params;
                (
                    EaGeometry::new(
                        p.idx_blk_elmts,
                        p.data_blk_min_elmts,
                        p.sup_blk_min_data_ptrs,
                        p.max_nelmts_bits,
                        p.max_dblk_page_nelmts_bits,
                    )?,
                    p.max_nelmts_bits,
                    c.chunk_size_len,
                    c.ea_header_addr,
                )
            };
            let loc = match geo.locate(chunk_idx)? {
                EaLoc::Dblk(l) => l,
                EaLoc::Index { .. } => unreachable!("chunk_idx >= idx_blk_elmts"),
            };
            if loc.paged {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk index {} needs a paged extensible-array data block, \
                     which is not yet supported",
                    chunk_idx
                )));
            }
            let class_id = if is_filtered {
                EA_CLS_FILT_CHUNK
            } else {
                EA_CLS_CHUNK
            };
            let dblk_nelmts = loc.dblk_nelmts as usize;

            // Resolve the data block's current address and its parent slot,
            // creating the owning super block on demand.
            let parent: DblkParent;
            let mut dblk_addr: u64;
            match loc.path {
                EaDblkPath::Direct { idx: di } => {
                    let c = self.datasets[index].chunked.as_ref().unwrap();
                    dblk_addr = if is_filtered {
                        c.filt_iblk.as_ref().unwrap().dblk_addrs[di]
                    } else {
                        c.ea_iblk.dblk_addrs[di]
                    };
                    parent = DblkParent::IndexBlock(di);
                }
                EaDblkPath::ViaSblk {
                    sblk_off,
                    local_dblk,
                    ndblks_in_sblk,
                    sblk_block_offset,
                } => {
                    let mut sblk_addr = {
                        let c = self.datasets[index].chunked.as_ref().unwrap();
                        if is_filtered {
                            c.filt_iblk.as_ref().unwrap().sblk_addrs[sblk_off]
                        } else {
                            c.ea_iblk.sblk_addrs[sblk_off]
                        }
                    };
                    if sblk_addr == UNDEF_ADDR {
                        let sb = ExtensibleArraySuperBlock::new(
                            class_id,
                            ea_header_addr,
                            sblk_block_offset,
                            ndblks_in_sblk,
                        );
                        let enc = sb.encode(&self.ctx, max_nelmts_bits);
                        sblk_addr = self.allocator.allocate(enc.len() as u64);
                        self.handle.write_at(sblk_addr, &enc)?;
                        let c = self.datasets[index].chunked.as_mut().unwrap();
                        if is_filtered {
                            c.filt_iblk.as_mut().unwrap().sblk_addrs[sblk_off] = sblk_addr;
                        } else {
                            c.ea_iblk.sblk_addrs[sblk_off] = sblk_addr;
                        }
                        c.ea_header.num_sblks_created += 1;
                        c.ea_header.size_sblks_created += enc.len() as u64;
                    }
                    let sb_buf = self.handle.read_at_most(sblk_addr, 65536)?;
                    // The writer never creates paged super blocks (it errors
                    // before the paging threshold), so page_init_total is 0.
                    let sb = ExtensibleArraySuperBlock::decode(
                        &sb_buf,
                        &self.ctx,
                        max_nelmts_bits,
                        ndblks_in_sblk,
                        0,
                    )?;
                    dblk_addr = sb.dblk_addrs[local_dblk];
                    parent = DblkParent::SuperBlock {
                        sblk_addr,
                        ndblks_in_sblk,
                        local_dblk,
                    };
                }
            }

            // Create or update the data block holding this chunk's entry.
            let created = dblk_addr == UNDEF_ADDR;
            if is_filtered {
                let entry = FilteredChunkEntry {
                    addr: chunk_addr,
                    nbytes: compressed_size,
                    filter_mask,
                };
                let mut dblk = if created {
                    FilteredDataBlock::new(ea_header_addr, loc.dblk_block_offset, dblk_nelmts)
                } else {
                    let buf = self.handle.read_at_most(dblk_addr, 65536)?;
                    FilteredDataBlock::decode(
                        &buf,
                        &self.ctx,
                        max_nelmts_bits,
                        dblk_nelmts,
                        chunk_size_len,
                    )?
                };
                dblk.elements[loc.offset_in_dblk as usize] = entry;
                let enc = dblk.encode(&self.ctx, max_nelmts_bits, chunk_size_len);
                if created {
                    dblk_addr = self.allocator.allocate(enc.len() as u64);
                }
                self.handle.write_at(dblk_addr, &enc)?;
                if created {
                    let c = self.datasets[index].chunked.as_mut().unwrap();
                    c.ea_header.num_dblks_created += 1;
                    c.ea_header.size_dblks_created += enc.len() as u64;
                }
            } else {
                let mut dblk = if created {
                    ExtensibleArrayDataBlock::new(
                        ea_header_addr,
                        loc.dblk_block_offset,
                        dblk_nelmts,
                    )
                } else {
                    let buf = self.handle.read_at_most(dblk_addr, 65536)?;
                    ExtensibleArrayDataBlock::decode(&buf, &self.ctx, max_nelmts_bits, dblk_nelmts)?
                };
                dblk.elements[loc.offset_in_dblk as usize] = chunk_addr;
                let enc = dblk.encode(&self.ctx, max_nelmts_bits);
                if created {
                    dblk_addr = self.allocator.allocate(enc.len() as u64);
                }
                self.handle.write_at(dblk_addr, &enc)?;
                if created {
                    let c = self.datasets[index].chunked.as_mut().unwrap();
                    c.ea_header.num_dblks_created += 1;
                    c.ea_header.size_dblks_created += enc.len() as u64;
                }
            }

            // Record a newly-created data block's address in its parent.
            if created {
                match parent {
                    DblkParent::IndexBlock(di) => {
                        let c = self.datasets[index].chunked.as_mut().unwrap();
                        if is_filtered {
                            c.filt_iblk.as_mut().unwrap().dblk_addrs[di] = dblk_addr;
                        } else {
                            c.ea_iblk.dblk_addrs[di] = dblk_addr;
                        }
                    }
                    DblkParent::SuperBlock {
                        sblk_addr,
                        ndblks_in_sblk,
                        local_dblk,
                    } => {
                        let buf = self.handle.read_at_most(sblk_addr, 65536)?;
                        let mut sb = ExtensibleArraySuperBlock::decode(
                            &buf,
                            &self.ctx,
                            max_nelmts_bits,
                            ndblks_in_sblk,
                            0,
                        )?;
                        sb.dblk_addrs[local_dblk] = dblk_addr;
                        let enc = sb.encode(&self.ctx, max_nelmts_bits);
                        self.handle.write_at(sblk_addr, &enc)?;
                    }
                }
            }

            // Statistics.
            let c = self.datasets[index].chunked.as_mut().unwrap();
            c.chunks_written += 1;
            if chunk_idx + 1 > c.ea_header.max_idx_set {
                c.ea_header.max_idx_set = chunk_idx + 1;
            }
            if created {
                c.ea_header.num_elmts_realized += loc.dblk_nelmts;
            }
        }
        Ok(())
    }

    /// Write a slice (hyperslab) of data to a contiguous dataset.
    ///
    /// `starts` and `counts` define the N-dimensional selection.
    /// `data` must be exactly `product(counts) * element_size` bytes.
    pub fn write_slice(
        &mut self,
        index: usize,
        starts: &[u64],
        counts: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = &self.datasets[index];
        if ds.chunked.is_some() || ds.fixed_array.is_some() || ds.btree_v2.is_some() {
            return Err(crate::io::IoError::InvalidState(
                "write_slice is only for contiguous datasets".into(),
            ));
        }
        if ds.data_addr == UNDEF_ADDR {
            return Err(crate::io::IoError::InvalidState(
                "dataset has no data allocated".into(),
            ));
        }

        let dims = &ds.dataspace.dims;
        let element_size = ds.datatype.element_size() as u64;
        let ndims = dims.len();

        if starts.len() != ndims || counts.len() != ndims {
            return Err(crate::io::IoError::InvalidState(
                "starts/counts length must match dataset rank".into(),
            ));
        }
        if ndims == 0 {
            return Err(crate::io::IoError::InvalidState(
                "write_slice does not support scalar datasets; use write_dataset_raw".into(),
            ));
        }

        // Every hyperslab edge must stay inside the dataset; without this an
        // out-of-bounds selection writes raw bytes over neighbouring data.
        for d in 0..ndims {
            let end = starts[d]
                .checked_add(counts[d])
                .ok_or_else(|| crate::io::IoError::InvalidState("slice extent overflow".into()))?;
            if end > dims[d] {
                return Err(crate::io::IoError::InvalidState(format!(
                    "slice out of bounds in dimension {}: start {} + count {} exceeds extent {}",
                    d, starts[d], counts[d], dims[d]
                )));
            }
        }

        let out_elems: u64 = counts.iter().product();
        if data.len() as u64 != out_elems * element_size {
            return Err(crate::io::IoError::InvalidState(format!(
                "data size mismatch: expected {} bytes, got {}",
                out_elems * element_size,
                data.len()
            )));
        }

        let base_addr = ds.data_addr;
        // `dims` borrows self.datasets; collect what the run iterator needs so
        // the closure can borrow self.handle (a disjoint field) for the write.
        let dims = dims.clone();

        // Write each maximal contiguous run in one `write_at`. Trailing
        // full-selected dimensions coalesce, mirroring the read path: a slice
        // with a full last axis becomes one write per outer index instead of
        // one write per last-axis row.
        for_each_contiguous_run(
            &dims,
            starts,
            counts,
            element_size,
            |dst_off, src_off, len| {
                self.handle
                    .write_at(base_addr + dst_off, &data[src_off..src_off + len])
                    .map_err(Into::into)
            },
        )?;

        Ok(())
    }

    /// Add an attribute to the root group (file-level attribute).
    pub fn add_root_attribute(
        &mut self,
        attr: crate::format::messages::attribute::AttributeMessage,
    ) {
        // Replace existing attribute with the same name, or append new one.
        if let Some(pos) = self
            .root_attributes
            .iter()
            .position(|a| a.name == attr.name)
        {
            self.root_attributes[pos] = attr;
        } else {
            self.root_attributes.push(attr);
        }
    }

    /// Create a variable-length string dataset and write string data.
    ///
    /// Stores strings in a global heap collection. The dataset raw data
    /// consists of vlen references (collection_addr + object_index pairs).
    pub fn create_vlen_string_dataset(&mut self, name: &str, strings: &[&str]) -> IoResult<usize> {
        use crate::format::global_heap::{encode_vlen_reference, GlobalHeapCollection};
        use crate::format::messages::datatype::DatatypeMessage;

        let num_strings = strings.len() as u64;

        // Build a global heap collection with all strings
        let mut gcol = GlobalHeapCollection::new();
        let mut obj_indices = Vec::with_capacity(strings.len());
        for s in strings {
            let idx = gcol.add_object(s.as_bytes().to_vec())?;
            obj_indices.push(idx);
        }

        // Encode and write the global heap collection
        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

        // Build raw data: vlen references
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let data_size = (num_strings as usize) * ref_size;
        let mut raw_data = Vec::with_capacity(data_size);
        for (i, &obj_idx) in obj_indices.iter().enumerate() {
            let seq_len = strings[i].len() as u32;
            raw_data.extend_from_slice(&encode_vlen_reference(
                seq_len,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        // Allocate and write raw data
        let data_addr = self.allocator.allocate(data_size as u64);
        self.handle.write_at(data_addr, &raw_data)?;

        // Create the dataset with vlen string datatype
        let datatype = DatatypeMessage::vlen_string_utf8();
        let dataspace =
            crate::format::messages::dataspace::DataspaceMessage::simple(&[num_strings]);

        let idx = self.datasets.len();
        self.datasets.push(DatasetInfo {
            name: name.to_string(),
            datatype,
            dataspace,
            obj_header_addr: 0,
            data_addr,
            data_size: data_size as u64,
            chunked: None,
            fixed_array: None,
            btree_v2: None,
            attributes: Vec::new(),
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            filter_pipeline: None,
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
        });

        Ok(idx)
    }

    /// Create a chunked, compressed variable-length string dataset.
    ///
    /// Strings are stored in the global heap (same as `create_vlen_string_dataset`),
    /// but the vlen references are stored in chunked layout with the given filter
    /// pipeline (e.g., deflate, zstd). `chunk_size` is the number of strings per chunk.
    pub fn create_vlen_string_dataset_compressed(
        &mut self,
        name: &str,
        strings: &[&str],
        chunk_size: usize,
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        use crate::format::global_heap::{encode_vlen_reference, GlobalHeapCollection};
        use crate::format::messages::datatype::DatatypeMessage;

        let num_strings = strings.len() as u64;

        // Build a global heap collection with all strings
        let mut gcol = GlobalHeapCollection::new();
        let mut obj_indices = Vec::with_capacity(strings.len());
        for s in strings {
            let idx = gcol.add_object(s.as_bytes().to_vec())?;
            obj_indices.push(idx);
        }

        // Encode and write the global heap collection
        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

        // Build raw data: vlen references
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let data_size = (num_strings as usize) * ref_size;
        let mut raw_data = Vec::with_capacity(data_size);
        for (i, &obj_idx) in obj_indices.iter().enumerate() {
            let seq_len = strings[i].len() as u32;
            raw_data.extend_from_slice(&encode_vlen_reference(
                seq_len,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        // Set up chunked compressed layout
        let datatype = DatatypeMessage::vlen_string_utf8();
        let element_size = datatype.element_size_ctx(&self.ctx) as u64;
        let chunk_dims: Vec<u64> = vec![chunk_size as u64];
        let dims: Vec<u64> = vec![num_strings];
        let max_dims: Vec<u64> = vec![num_strings];
        let chunk_bytes = chunk_size as u64 * element_size;
        let chunk_size_len = compute_chunk_size_len(chunk_bytes);

        let earray_params = EarrayParams::default_params();
        let ndblk_addrs = compute_ndblk_addrs(earray_params.sup_blk_min_data_ptrs)?;
        let nsblk_addrs = compute_nsblk_addrs(
            earray_params.idx_blk_elmts,
            earray_params.data_blk_min_elmts,
            earray_params.sup_blk_min_data_ptrs,
            earray_params.max_nelmts_bits,
        )?;

        // Create filtered EA header
        let mut ea_header =
            ExtensibleArrayHeader::new_for_filtered_chunks(&self.ctx, chunk_size_len);
        ea_header.max_nelmts_bits = earray_params.max_nelmts_bits;
        ea_header.idx_blk_elmts = earray_params.idx_blk_elmts;
        ea_header.data_blk_min_elmts = earray_params.data_blk_min_elmts;
        ea_header.sup_blk_min_data_ptrs = earray_params.sup_blk_min_data_ptrs;
        ea_header.max_dblk_page_nelmts_bits = earray_params.max_dblk_page_nelmts_bits;

        let hdr_encoded = ea_header.encode(&self.ctx);
        let ea_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create filtered index block
        let filt_iblk = FilteredIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );
        let iblk_encoded = filt_iblk.encode(&self.ctx, chunk_size_len);
        let ea_iblk_addr = self.allocator.allocate(iblk_encoded.len() as u64);

        ea_header.idx_blk_addr = ea_iblk_addr;

        let hdr_encoded = ea_header.encode(&self.ctx);
        self.handle.write_at(ea_header_addr, &hdr_encoded)?;
        self.handle.write_at(ea_iblk_addr, &iblk_encoded)?;

        let dataspace = DataspaceMessage {
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let ea_iblk = ExtensibleArrayIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );

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
            filter_pipeline: Some(pipeline),
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
            fixed_array: None,
            btree_v2: None,
            chunked: Some(ChunkedDatasetInfo {
                chunk_dims: chunk_dims.clone(),
                max_dims: max_dims.clone(),
                earray_params,
                ea_header_addr,
                ea_iblk_addr,
                ndblk_addrs,
                ea_header,
                ea_iblk,
                chunks_written: 0,
                filt_iblk: Some(filt_iblk),
                chunk_size_len,
            }),
        });

        // Write chunks of vlen references with compression
        let chunk_byte_size = chunk_bytes as usize;
        let num_chunks = raw_data.len().div_ceil(chunk_byte_size);
        for chunk_i in 0..num_chunks {
            let start = chunk_i * chunk_byte_size;
            let end = (start + chunk_byte_size).min(raw_data.len());
            let chunk_data = if end - start < chunk_byte_size {
                // Pad last chunk to full size (vlen datasets carry no user
                // fill value, so this resolves to zero = null vlen reference).
                let mut padded = self.new_chunk_buffer(idx, chunk_byte_size);
                padded[..end - start].copy_from_slice(&raw_data[start..end]);
                padded
            } else {
                raw_data[start..end].to_vec()
            };
            self.write_chunk(idx, chunk_i as u64, &chunk_data)?;
        }

        Ok(idx)
    }

    /// Create an empty chunked vlen string dataset ready for incremental appends.
    ///
    /// The dataset starts with `dims = [0]` and `max_dims = [unlimited]`.
    /// Use `append_vlen_strings` to add data.
    pub fn create_appendable_vlen_string_dataset(
        &mut self,
        name: &str,
        chunk_size: usize,
        pipeline: Option<FilterPipeline>,
    ) -> IoResult<usize> {
        let datatype = DatatypeMessage::vlen_string_utf8();
        let chunk_dims: Vec<u64> = vec![chunk_size as u64];
        let dims: Vec<u64> = vec![0];
        let max_dims: Vec<u64> = vec![u64::MAX];

        if let Some(ref pl) = pipeline {
            self.create_chunked_dataset_with_pipeline(
                name,
                datatype,
                &dims,
                &max_dims,
                &chunk_dims,
                pl.clone(),
            )
        } else {
            self.create_chunked_dataset(name, datatype, &dims, &max_dims, &chunk_dims)
        }
    }

    /// Append variable-length strings to an existing chunked vlen string dataset.
    ///
    /// Creates a new global heap collection for the strings, builds vlen
    /// references, and appends them as new chunks to the dataset.
    pub fn append_vlen_strings(&mut self, ds_index: usize, strings: &[&str]) -> IoResult<()> {
        use crate::format::global_heap::{encode_vlen_reference, GlobalHeapCollection};

        if strings.is_empty() {
            return Ok(());
        }

        // Build a new global heap collection for this batch
        let mut gcol = GlobalHeapCollection::new();
        let mut obj_indices = Vec::with_capacity(strings.len());
        for s in strings {
            let idx = gcol.add_object(s.as_bytes().to_vec())?;
            obj_indices.push(idx);
        }
        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

        // Build raw vlen reference bytes
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let mut raw = Vec::with_capacity(strings.len() * ref_size);
        for (i, &obj_idx) in obj_indices.iter().enumerate() {
            let seq_len = strings[i].len() as u32;
            raw.extend_from_slice(&encode_vlen_reference(
                seq_len,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        // Use the same chunked-append logic as append<T>
        let chunk_dims = self
            .dataset_chunk_dims(ds_index)
            .ok_or_else(|| crate::io::IoError::InvalidState("not a chunked dataset".into()))?
            .to_vec();
        let dims = self.dataset_dims(ds_index).to_vec();

        let n_new_frames = strings.len();
        let current_dim0 = dims[0] as usize;
        let chunk_dim0 = chunk_dims[0] as usize;
        let frame_bytes = ref_size;

        // Merge buffered data with new data
        let ds = &mut self.datasets[ds_index];
        let buffered_frames = ds.append_buffered_frames as usize;
        let mut combined = std::mem::take(&mut ds.append_buffer);
        combined.extend_from_slice(&raw);
        ds.append_buffered_frames = 0;

        let total_frames = buffered_frames + n_new_frames;
        let total_bytes = combined.len();
        let base_dim0 = current_dim0 - buffered_frames;
        let mut byte_pos = 0usize;
        let mut frame_pos = 0usize;

        while frame_pos < total_frames {
            let abs_frame = base_dim0 + frame_pos;
            let chunk_idx = abs_frame / chunk_dim0;
            let remaining_frames = total_frames - frame_pos;
            let frames_to_fill = chunk_dim0 - (abs_frame % chunk_dim0);

            if remaining_frames >= frames_to_fill {
                let end = byte_pos + frames_to_fill * frame_bytes;
                if frames_to_fill == chunk_dim0 {
                    self.write_chunk(ds_index, chunk_idx as u64, &combined[byte_pos..end])?;
                } else {
                    // Partial-chunk write: this branch only runs with
                    // offset_in_chunk > 0, meaning the chunk already holds
                    // earlier frames on disk. Read-modify-write so those
                    // frames survive — a fresh fill buffer would erase them.
                    let offset_in_chunk = (abs_frame % chunk_dim0) * frame_bytes;
                    let mut chunk_buf =
                        match self.read_chunk_if_present(ds_index, chunk_idx as u64)? {
                            Some(existing) => existing,
                            None => {
                                return Err(crate::io::IoError::InvalidState(format!(
                                    "cannot append into partially-written chunk {}: its \
                                     existing content was not found in the chunk index \
                                     (the file may be inconsistent)",
                                    chunk_idx
                                )));
                            }
                        };
                    chunk_buf[offset_in_chunk..offset_in_chunk + frames_to_fill * frame_bytes]
                        .copy_from_slice(&combined[byte_pos..end]);
                    self.write_chunk(ds_index, chunk_idx as u64, &chunk_buf)?;
                }
                byte_pos = end;
                frame_pos += frames_to_fill;
            } else {
                let ds = &mut self.datasets[ds_index];
                ds.append_buffer = combined[byte_pos..total_bytes].to_vec();
                ds.append_buffered_frames = remaining_frames as u64;
                frame_pos = total_frames;
            }
        }

        // Extend dims
        let logical_dim0 = base_dim0 + total_frames;
        let mut new_dims = dims;
        new_dims[0] = logical_dim0 as u64;
        self.extend_dataset(ds_index, &new_dims)?;

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
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset index {} out of range (have {})",
                ds_index,
                self.datasets.len()
            )));
        }
        self.datasets[ds_index].attributes.push(attr);
        Ok(())
    }

    /// Add (or replace) an attribute on a group identified by its full path.
    ///
    /// The attribute is written into the group's object header when the
    /// file is finalized. An existing attribute with the same name is
    /// replaced, matching [`add_root_attribute`](Self::add_root_attribute).
    pub fn add_group_attribute(
        &mut self,
        group_path: &str,
        attr: AttributeMessage,
    ) -> IoResult<()> {
        let gi = self
            .groups
            .iter()
            .position(|g| g.name == group_path && !g.deleted)
            .ok_or_else(|| {
                crate::io::IoError::NotFound(format!("group '{}' not found", group_path))
            })?;
        let attrs = &mut self.groups[gi].attributes;
        if let Some(pos) = attrs.iter().position(|a| a.name == attr.name) {
            attrs[pos] = attr;
        } else {
            attrs.push(attr);
        }
        Ok(())
    }

    /// Set a user-defined fill value for a dataset.
    ///
    /// `bytes` must be exactly one element wide (matching the dataset's
    /// datatype). The value is emitted as a `fill_defined = 2` fill-value
    /// message in the dataset object header when the file is finalized.
    ///
    /// IMPORTANT: for a *contiguous* dataset this also immediately writes
    /// the tiled fill value across the whole data block, so it must be
    /// called BEFORE any `write_dataset_raw` / `write_slice` — otherwise the
    /// fill write clobbers data already written. (The high-level builder
    /// always calls this right after creating the dataset.)
    pub fn set_dataset_fill_value(&mut self, ds_index: usize, bytes: Vec<u8>) -> IoResult<()> {
        let ds = self.datasets.get_mut(ds_index).ok_or_else(|| {
            crate::io::IoError::InvalidState(format!("dataset index {} out of range", ds_index))
        })?;
        let es = ds.datatype.element_size() as usize;
        if bytes.len() != es {
            return Err(crate::io::IoError::InvalidState(format!(
                "fill value is {} bytes but dataset element size is {}",
                bytes.len(),
                es
            )));
        }
        ds.fill_value = Some(bytes);

        // For a contiguous dataset the fill-value message declares
        // fill-on-allocation, but contiguous storage has no per-chunk
        // fill path — write the tiled fill value across the data block now
        // so unwritten elements read back as the fill value. (The high-level
        // builder calls this immediately after create, before any data is
        // written; a subsequent write_raw/write_slice overwrites its region.)
        let is_chunked = ds.chunked.is_some() || ds.fixed_array.is_some() || ds.btree_v2.is_some();
        if !is_chunked && ds.data_addr != UNDEF_ADDR && ds.data_size > 0 {
            let data_addr = ds.data_addr;
            let data_size = ds.data_size as usize;
            let fv = ds.fill_value.as_deref();
            let filled = crate::format::messages::fill_value::tiled_fill(data_size, fv);
            self.handle.write_at(data_addr, &filled)?;
        }
        Ok(())
    }

    /// Allocate a `chunk_bytes`-sized buffer pre-filled with dataset
    /// `ds_index`'s fill value (tiled one element wide), or zeros when no
    /// user-defined fill value exists.
    ///
    /// Every partial chunk the writer emits must be built on top of a
    /// buffer from this method, so that the unwritten element region of an
    /// allocated chunk reads back as the fill value rather than zero.
    pub(crate) fn new_chunk_buffer(&self, ds_index: usize, chunk_bytes: usize) -> Vec<u8> {
        let fv = self.datasets[ds_index].fill_value.as_deref();
        crate::format::messages::fill_value::tiled_fill(chunk_bytes, fv)
    }

    /// Read an already-written chunk's *decompressed* bytes when the chunk
    /// is allocated and resolvable from the in-memory extensible-array
    /// index. Handles index-block and data-block chunks, filtered and
    /// unfiltered.
    ///
    /// Returns `Ok(None)` only when the chunk has never been written
    /// (address `UNDEF`) or the index genuinely does not reach it. A
    /// caller doing a read-modify-write of a partial chunk treats `None`
    /// as an error rather than silently overwriting the chunk.
    pub(crate) fn read_chunk_if_present(
        &mut self,
        ds_index: usize,
        chunk_idx: u64,
    ) -> IoResult<Option<Vec<u8>>> {
        // Phase 1: resolve the chunk's location from the in-memory index.
        let ds = &self.datasets[ds_index];
        let element_size = ds.datatype.element_size() as u64;
        let pipeline = ds.filter_pipeline.clone();
        let Some(chunked) = ds.chunked.as_ref() else {
            return Ok(None);
        };
        let chunk_bytes = chunked.chunk_dims.iter().product::<u64>() * element_size;
        let max_nelmts_bits = chunked.earray_params.max_nelmts_bits;
        let chunk_size_len = chunked.chunk_size_len;
        let is_filtered = chunked.filt_iblk.is_some();

        // The chunk entry is either read straight from an index block, or
        // located via a data block that must itself be read from disk.
        enum Loc {
            Direct(u64, u64, u32),
            DataBlock {
                dblk_addr: u64,
                offset: usize,
                nelmts: usize,
            },
        }

        // Resolve the chunk's location with the libhdf5-compatible EA
        // geometry (super-block-grouped data blocks), matching `record_ea_chunk`.
        let ea_loc = {
            let p = &chunked.earray_params;
            EaGeometry::new(
                p.idx_blk_elmts,
                p.data_blk_min_elmts,
                p.sup_blk_min_data_ptrs,
                p.max_nelmts_bits,
                p.max_dblk_page_nelmts_bits,
            )?
            .locate(chunk_idx)?
        };
        let loc = match ea_loc {
            EaLoc::Index { elem } => {
                if is_filtered {
                    let e = &chunked.filt_iblk.as_ref().unwrap().elements[elem];
                    Loc::Direct(e.addr, e.nbytes, e.filter_mask)
                } else {
                    Loc::Direct(chunked.ea_iblk.elements[elem], chunk_bytes, 0)
                }
            }
            EaLoc::Dblk(l) => {
                if l.paged {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "chunk index {} lives in a paged extensible-array data \
                         block, which is not yet supported for read-modify-write",
                        chunk_idx
                    )));
                }
                let dblk_addr = match l.path {
                    EaDblkPath::Direct { idx } => {
                        if is_filtered {
                            chunked.filt_iblk.as_ref().unwrap().dblk_addrs[idx]
                        } else {
                            chunked.ea_iblk.dblk_addrs[idx]
                        }
                    }
                    EaDblkPath::ViaSblk {
                        sblk_off,
                        local_dblk,
                        ndblks_in_sblk,
                        ..
                    } => {
                        let sblk_addr = if is_filtered {
                            chunked.filt_iblk.as_ref().unwrap().sblk_addrs[sblk_off]
                        } else {
                            chunked.ea_iblk.sblk_addrs[sblk_off]
                        };
                        if sblk_addr == UNDEF_ADDR {
                            return Ok(None);
                        }
                        let sb_buf = self.handle.read_at_most(sblk_addr, 65536)?;
                        let sb = ExtensibleArraySuperBlock::decode(
                            &sb_buf,
                            &self.ctx,
                            max_nelmts_bits,
                            ndblks_in_sblk,
                            0,
                        )?;
                        sb.dblk_addrs[local_dblk]
                    }
                };
                if dblk_addr == UNDEF_ADDR {
                    return Ok(None);
                }
                Loc::DataBlock {
                    dblk_addr,
                    offset: l.offset_in_dblk as usize,
                    nelmts: l.dblk_nelmts as usize,
                }
            }
        };

        // Phase 2: resolve through the data block (if needed) and read. The
        // mask is the chunk's filter mask (0 for unfiltered), so a chunk
        // written via a direct chunk write with a skipped filter is reversed
        // correctly during read-modify-write.
        let (addr, nbytes, mask) = match loc {
            Loc::Direct(a, n, m) => (a, n, m),
            Loc::DataBlock {
                dblk_addr,
                offset,
                nelmts,
            } => {
                let buf = self.handle.read_at_most(dblk_addr, 65536)?;
                if is_filtered {
                    let dblk = FilteredDataBlock::decode(
                        &buf,
                        &self.ctx,
                        max_nelmts_bits,
                        nelmts,
                        chunk_size_len,
                    )?;
                    let e = &dblk.elements[offset];
                    (e.addr, e.nbytes, e.filter_mask)
                } else {
                    let dblk =
                        ExtensibleArrayDataBlock::decode(&buf, &self.ctx, max_nelmts_bits, nelmts)?;
                    (dblk.elements[offset], chunk_bytes, 0)
                }
            }
        };
        if addr == UNDEF_ADDR || nbytes == 0 {
            return Ok(None);
        }

        let raw = self.handle.read_at(addr, nbytes as usize)?;
        if is_filtered {
            let Some(pl) = pipeline.as_ref() else {
                return Ok(None);
            };
            Ok(Some(filter::reverse_filters_masked(pl, &raw, mask)?))
        } else {
            Ok(Some(raw))
        }
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
        self.ensure_unique_dataset_name(name)?;
        // Compute total number of chunks. `chunk_dims` is caller-supplied;
        // validate it before any indexing or division.
        let ndims = dims.len();
        if chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk shape has {} dimensions but the dataspace has {}",
                chunk_dims.len(),
                ndims
            )));
        }
        let mut num_chunks: u64 = 1;
        for d in 0..ndims {
            if chunk_dims[d] == 0 {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk dimension {d} is zero"
                )));
            }
            num_chunks = num_chunks
                .checked_mul(dims[d].div_ceil(chunk_dims[d]))
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState("chunk count overflows u64".into())
                })?;
        }

        // Create FA header
        let mut fa_header = FixedArrayHeader::new_for_chunks(&self.ctx, num_chunks);
        let hdr_encoded = fa_header.encode(&self.ctx);
        let fa_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create FA data block. libhdf5 switches to a paged layout once
        // num_elmts exceeds dblk_page_nelmts; both layouts allocate space
        // for `num_chunks` chunk addresses up front, but the paged layout
        // also reserves the page-init bitmap and a per-page checksum.
        let fa_dblk = FixedArrayDataBlock::new_unfiltered(fa_header_addr, num_chunks as usize);
        let dblk_size = fixed_array_dblk_disk_size(&self.ctx, &fa_header);
        let fa_dblk_addr = self.allocator.allocate(dblk_size);

        // Update header with data block address
        fa_header.data_blk_addr = fa_dblk_addr;

        // Write both. The data block content is finalized in `flush_dataset`
        // once all chunk addresses are known; here we just reserve space and
        // write the header so the file is structurally consistent.
        let hdr_encoded = fa_header.encode(&self.ctx);
        self.handle.write_at(fa_header_addr, &hdr_encoded)?;
        let dblk_encoded = encode_fixed_array_dblk(&self.ctx, &fa_header, &fa_dblk);
        debug_assert_eq!(dblk_encoded.len() as u64, dblk_size);
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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

    /// Define a fixed-shape (no unlimited dimension) compressed chunked dataset
    /// indexed by a *filtered* Fixed Array.
    ///
    /// Like `create_fixed_array_dataset`, but the FA header carries the filtered
    /// client id and a `chunk_size_len`-wide compressed-size field per chunk
    /// (`FixedArrayFilteredChunkElement`), and the dataset gets a filter
    /// pipeline. Chunks written via `write_chunk_fixed_array` are compressed and
    /// their compressed size + filter mask are recorded in the data block.
    pub fn create_fixed_array_dataset_with_pipeline(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk_dims: &[u64],
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        self.ensure_unique_dataset_name(name)?;
        let ndims = dims.len();
        if chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk shape has {} dimensions but the dataspace has {}",
                chunk_dims.len(),
                ndims
            )));
        }
        let mut num_chunks: u64 = 1;
        for d in 0..ndims {
            if chunk_dims[d] == 0 {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk dimension {d} is zero"
                )));
            }
            num_chunks = num_chunks
                .checked_mul(dims[d].div_ceil(chunk_dims[d]))
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState("chunk count overflows u64".into())
                })?;
        }

        // chunk_size_len is sized from the uncompressed chunk byte count, the
        // same way the filtered Extensible Array path computes it: the
        // compressed size never exceeds the uncompressed size meaningfully, so
        // this width always holds the stored value.
        let element_size = datatype.element_size() as u64;
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;
        let chunk_size_len = compute_chunk_size_len(chunk_bytes);

        // Create the filtered FA header.
        let mut fa_header =
            FixedArrayHeader::new_for_filtered_chunks(&self.ctx, num_chunks, chunk_size_len);
        let hdr_encoded = fa_header.encode(&self.ctx);
        let fa_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create the filtered FA data block; both flat and paged layouts
        // reserve space for `num_chunks` filtered entries up front.
        let fa_dblk = FixedArrayDataBlock::new_filtered(fa_header_addr, num_chunks as usize);
        let dblk_size = fixed_array_dblk_disk_size(&self.ctx, &fa_header);
        let fa_dblk_addr = self.allocator.allocate(dblk_size);

        fa_header.data_blk_addr = fa_dblk_addr;

        let hdr_encoded = fa_header.encode(&self.ctx);
        self.handle.write_at(fa_header_addr, &hdr_encoded)?;
        let dblk_encoded = encode_fixed_array_dblk(&self.ctx, &fa_header, &fa_dblk);
        debug_assert_eq!(dblk_encoded.len() as u64, dblk_size);
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
            filter_pipeline: Some(pipeline),
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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
        self.ensure_unique_dataset_name(name)?;
        let ndims = dims.len();
        let bt2_index = Bt2ChunkIndex::new_unfiltered(ndims);

        // We'll allocate space for header and leaf node; they'll be written
        // during flush_dataset_bt2.
        let hdr = crate::format::chunk_index::btree_v2::Bt2Header::new_for_chunks(&self.ctx, ndims);
        let hdr_encoded = hdr.encode(&self.ctx);
        let bt2_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);
        self.handle.write_at(bt2_header_addr, &hdr_encoded)?;

        // Allocate a placeholder leaf node (empty for now)
        let leaf = crate::format::chunk_index::btree_v2::Bt2LeafNode::new(
            crate::format::chunk_index::btree_v2::BT2_TYPE_CHUNK_UNFILT,
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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
        let element_size = datatype.element_size() as u64;
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;
        let chunk_size_len = compute_chunk_size_len(chunk_bytes);

        let earray_params = EarrayParams::default_params();
        let ndblk_addrs = compute_ndblk_addrs(earray_params.sup_blk_min_data_ptrs)?;
        let nsblk_addrs = compute_nsblk_addrs(
            earray_params.idx_blk_elmts,
            earray_params.data_blk_min_elmts,
            earray_params.sup_blk_min_data_ptrs,
            earray_params.max_nelmts_bits,
        )?;

        // Create filtered EA header
        let mut ea_header =
            ExtensibleArrayHeader::new_for_filtered_chunks(&self.ctx, chunk_size_len);
        ea_header.max_nelmts_bits = earray_params.max_nelmts_bits;
        ea_header.idx_blk_elmts = earray_params.idx_blk_elmts;
        ea_header.data_blk_min_elmts = earray_params.data_blk_min_elmts;
        ea_header.sup_blk_min_data_ptrs = earray_params.sup_blk_min_data_ptrs;
        ea_header.max_dblk_page_nelmts_bits = earray_params.max_dblk_page_nelmts_bits;

        let hdr_encoded = ea_header.encode(&self.ctx);
        let ea_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create filtered index block
        let filt_iblk = FilteredIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );
        let iblk_encoded = filt_iblk.encode(&self.ctx, chunk_size_len);
        let ea_iblk_addr = self.allocator.allocate(iblk_encoded.len() as u64);

        ea_header.idx_blk_addr = ea_iblk_addr;

        let hdr_encoded = ea_header.encode(&self.ctx);
        self.handle.write_at(ea_header_addr, &hdr_encoded)?;
        self.handle.write_at(ea_iblk_addr, &iblk_encoded)?;

        let dataspace = DataspaceMessage {
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        // Also create a dummy unfiltered iblk (not used for compressed, but needed for struct)
        let ea_iblk = ExtensibleArrayIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );

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
            filter_pipeline: Some(FilterPipeline::deflate(compression_level)),
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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
                chunks_written: 0,
                filt_iblk: Some(filt_iblk),
                chunk_size_len,
            }),
        });

        Ok(idx)
    }

    /// Create a chunked dataset with a custom filter pipeline.
    pub fn create_chunked_dataset_with_pipeline(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        self.ensure_unique_dataset_name(name)?;
        let element_size = datatype.element_size() as u64;
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;
        let chunk_size_len = compute_chunk_size_len(chunk_bytes);

        let earray_params = EarrayParams::default_params();
        let ndblk_addrs = compute_ndblk_addrs(earray_params.sup_blk_min_data_ptrs)?;
        let nsblk_addrs = compute_nsblk_addrs(
            earray_params.idx_blk_elmts,
            earray_params.data_blk_min_elmts,
            earray_params.sup_blk_min_data_ptrs,
            earray_params.max_nelmts_bits,
        )?;

        let mut ea_header =
            ExtensibleArrayHeader::new_for_filtered_chunks(&self.ctx, chunk_size_len);
        ea_header.max_nelmts_bits = earray_params.max_nelmts_bits;
        ea_header.idx_blk_elmts = earray_params.idx_blk_elmts;
        ea_header.data_blk_min_elmts = earray_params.data_blk_min_elmts;
        ea_header.sup_blk_min_data_ptrs = earray_params.sup_blk_min_data_ptrs;
        ea_header.max_dblk_page_nelmts_bits = earray_params.max_dblk_page_nelmts_bits;

        let hdr_encoded = ea_header.encode(&self.ctx);
        let ea_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        let filt_iblk = FilteredIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );
        let iblk_encoded = filt_iblk.encode(&self.ctx, chunk_size_len);
        let ea_iblk_addr = self.allocator.allocate(iblk_encoded.len() as u64);

        ea_header.idx_blk_addr = ea_iblk_addr;
        let hdr_encoded = ea_header.encode(&self.ctx);
        self.handle.write_at(ea_header_addr, &hdr_encoded)?;
        self.handle.write_at(ea_iblk_addr, &iblk_encoded)?;

        let dataspace = DataspaceMessage {
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };
        let ea_iblk = ExtensibleArrayIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );

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
            filter_pipeline: Some(pipeline),
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            deleted: false,
            fill_value: None,
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
                chunks_written: 0,
                filt_iblk: Some(filt_iblk),
                chunk_size_len,
            }),
        });
        Ok(idx)
    }

    /// Write a chunk to a fixed-array-indexed dataset.
    ///
    /// `chunk_coords` is the multidimensional chunk index (e.g., [row_chunk, col_chunk]).
    /// The uncompressed `data` must be exactly one chunk wide; the filter
    /// pipeline (if any) runs here before the bytes reach the index.
    pub fn write_chunk_fixed_array(
        &mut self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = &self.datasets[index];
        let element_size = ds.datatype.element_size() as u64;
        let fa = ds
            .fixed_array
            .as_ref()
            .ok_or_else(|| crate::io::IoError::InvalidState("not a fixed-array dataset".into()))?;
        let chunk_bytes: u64 = fa.chunk_dims.iter().product::<u64>() * element_size;

        if data.len() as u64 != chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes,
                data.len()
            )));
        }
        let write_data;
        let data_to_write = if let Some(ref pipeline) = ds.filter_pipeline {
            write_data = filter::apply_filters(pipeline, data)?;
            &write_data[..]
        } else {
            data
        };
        // filter_mask = 0: the whole pipeline ran (or the dataset is
        // unfiltered), so no filter is skipped for this chunk.
        self.record_fixed_array_chunk(index, chunk_coords, data_to_write, 0)
    }

    /// Write a pre-filtered chunk verbatim to a fixed-array dataset, recording
    /// the caller-supplied `filter_mask`.
    ///
    /// The bytes are stored exactly as given (no filter pipeline is run); this
    /// is the fixed-array half of the HDF5 "direct chunk write"
    /// (`H5Dwrite_chunk`) operation. `filter_mask` is a bitfield: bit *i* set
    /// means filter *i* of the pipeline was **not** applied to this chunk and
    /// must be skipped on read; pass 0 when the full pipeline was applied
    /// upstream.
    ///
    /// Requires a filtered dataset — only the filtered FA element carries the
    /// size+mask slot.
    pub fn write_compressed_chunk_fixed_array(
        &mut self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        if self.datasets[index].filter_pipeline.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "write_compressed_chunk_fixed_array requires a filtered dataset \
                 (no slot for a compressed size or filter mask on an unfiltered \
                 chunk index)"
                    .into(),
            ));
        }
        self.record_fixed_array_chunk(index, chunk_coords, data, filter_mask)
    }

    /// Place an already-final chunk (`final_bytes` is whatever goes to disk —
    /// filtered if the dataset is filtered, raw otherwise) into a fixed-array
    /// dataset's data block, recording the caller-supplied `filter_mask`.
    /// Shared by [`write_chunk_fixed_array`](Self::write_chunk_fixed_array)
    /// and [`write_compressed_chunk_fixed_array`](Self::write_compressed_chunk_fixed_array).
    fn record_fixed_array_chunk(
        &mut self,
        index: usize,
        chunk_coords: &[u64],
        final_bytes: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        let ds = &self.datasets[index];
        let is_filtered = ds.filter_pipeline.is_some();
        let fa = ds
            .fixed_array
            .as_ref()
            .ok_or_else(|| crate::io::IoError::InvalidState("not a fixed-array dataset".into()))?;

        // Compute linear chunk index from multidimensional coordinates.
        let dims = &ds.dataspace.dims;
        let chunk_dims = &fa.chunk_dims;
        let ndims = dims.len();
        if chunk_coords.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk_coords has {} entries but the dataset has {} dimensions",
                chunk_coords.len(),
                ndims
            )));
        }
        let mut linear_idx: u64 = 0;
        let mut stride: u64 = 1;
        for d in (0..ndims).rev() {
            let n_chunks_in_dim = dims[d].div_ceil(chunk_dims[d]);
            // Reject an out-of-grid coordinate: without this an inner
            // dimension's overflow silently aliases a different chunk slot.
            if chunk_coords[d] >= n_chunks_in_dim {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk coordinate {} in dimension {} is outside the chunk grid (0..{})",
                    chunk_coords[d], d, n_chunks_in_dim
                )));
            }
            linear_idx += chunk_coords[d] * stride;
            stride *= n_chunks_in_dim;
        }

        // Allocate space for the chunk data
        let chunk_addr = self.allocator.allocate(final_bytes.len() as u64);
        self.handle.write_at(chunk_addr, final_bytes)?;

        // Update the fixed array data block.
        let fa = self.datasets[index].fixed_array.as_mut().unwrap();
        let lidx = linear_idx as usize;
        if is_filtered {
            // Filtered FA: store address + stored size + filter mask. A
            // non-zero mask bit means "filter i was skipped for this chunk".
            let stored_size = final_bytes.len();
            if stored_size > u32::MAX as usize {
                return Err(crate::io::IoError::InvalidState(format!(
                    "compressed chunk size {stored_size} exceeds u32::MAX"
                )));
            }
            // The stored size is encoded in the FA header's `chunk_size_len`-byte
            // field; libhdf5 errors if it does not fit (H5D_CHUNK_ENCODE_SIZE_CHECK)
            // rather than truncating silently. element_size = sizeof_addr +
            // chunk_size_len + 4 by construction.
            let chunk_size_len = (fa.fa_header.element_size as usize)
                .checked_sub(self.ctx.sizeof_addr as usize + 4)
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState(
                        "filtered fixed-array element size is too small".into(),
                    )
                })?;
            if chunk_size_len < 8 && stored_size >= (1usize << (chunk_size_len * 8)) {
                return Err(crate::io::IoError::InvalidState(format!(
                    "compressed chunk size {stored_size} does not fit in the \
                     {chunk_size_len}-byte fixed-array chunk-size field"
                )));
            }
            if lidx < fa.fa_dblk.filtered_elements.len() {
                fa.fa_dblk.filtered_elements[lidx] = FixedArrayFilteredChunkElement {
                    address: chunk_addr,
                    chunk_size: stored_size as u32,
                    filter_mask,
                };
                fa.chunks_written += 1;
            } else {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk index {} out of range (max {})",
                    linear_idx,
                    fa.fa_dblk.filtered_elements.len()
                )));
            }
        } else {
            // An unfiltered fixed array stores only addresses — there is no
            // slot for a filter mask, so a non-zero mask cannot be honored.
            if filter_mask != 0 {
                return Err(crate::io::IoError::InvalidState(
                    "filter_mask is non-zero but the dataset is unfiltered".into(),
                ));
            }
            if lidx < fa.fa_dblk.elements.len() {
                fa.fa_dblk.elements[lidx] = chunk_addr;
                fa.chunks_written += 1;
            } else {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk index {} out of range (max {})",
                    linear_idx,
                    fa.fa_dblk.elements.len()
                )));
            }
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
        let bt2 = ds
            .btree_v2
            .as_ref()
            .ok_or_else(|| crate::io::IoError::InvalidState("not a B-tree v2 dataset".into()))?;
        let chunk_bytes: u64 = bt2.chunk_dims.iter().product::<u64>() * element_size;

        if data.len() as u64 != chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
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
    pub fn write_chunks_batch(&mut self, ds_index: usize, chunks: &[(u64, &[u8])]) -> IoResult<()> {
        #[cfg(feature = "parallel")]
        {
            // If filter pipeline is set, compress all chunks in parallel
            if let Some(ref pipeline) = self.datasets[ds_index].filter_pipeline {
                let chunk_data: Vec<Vec<u8>> = chunks.iter().map(|(_, d)| d.to_vec()).collect();
                let compressed = filter::apply_filters_parallel(pipeline, &chunk_data);
                for ((idx, _), compressed_data) in chunks.iter().zip(compressed.iter()) {
                    // filter_mask = 0: the parallel compressor ran the whole
                    // pipeline, so no filter is skipped for these chunks.
                    self.write_compressed_chunk(ds_index, *idx, compressed_data, 0)?;
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

    /// Write a pre-filtered chunk verbatim to an EA-indexed dataset, recording
    /// the caller-supplied `filter_mask`.
    ///
    /// The bytes are stored exactly as given (no filter pipeline is run); this
    /// is the extensible-array half of the HDF5 "direct chunk write"
    /// (`H5Dwrite_chunk`) operation. `filter_mask` is a bitfield: bit *i* set
    /// means filter *i* of the pipeline was **not** applied to this chunk and
    /// must be skipped on read; pass 0 when the full pipeline was applied
    /// upstream.
    ///
    /// Requires a filtered dataset — only the filtered EA entry carries the
    /// size+mask slot. An unfiltered dataset has nowhere to record either.
    pub fn write_compressed_chunk(
        &mut self,
        index: usize,
        chunk_idx: u64,
        compressed_data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        if self.datasets[index].filter_pipeline.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "write_compressed_chunk requires a filtered dataset (no slot for \
                 a compressed size or filter mask on an unfiltered chunk index)"
                    .into(),
            ));
        }
        let compressed_size = compressed_data.len() as u64;
        let chunk_addr = self.allocator.allocate(compressed_size);
        self.handle.write_at(chunk_addr, compressed_data)?;

        self.record_ea_chunk(index, chunk_idx, chunk_addr, compressed_size, filter_mask)
    }

    /// Extend the dimensions of a chunked dataset.
    pub fn extend_dataset(&mut self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = &mut self.datasets[index];
        if ds.chunked.is_none() && ds.fixed_array.is_none() && ds.btree_v2.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "can only extend chunked datasets".into(),
            ));
        }
        if new_dims.len() != ds.dataspace.dims.len() {
            return Err(crate::io::IoError::InvalidState(format!(
                "extend_dataset rank mismatch: dataset has {} dimensions, got {}",
                ds.dataspace.dims.len(),
                new_dims.len()
            )));
        }
        // The chunk index and append buffers assume the logical size only
        // grows; shrinking below already-written data desynchronizes them.
        for (d, (&new, &cur)) in new_dims.iter().zip(&ds.dataspace.dims).enumerate() {
            if new < cur {
                return Err(crate::io::IoError::InvalidState(format!(
                    "extend_dataset cannot shrink dimension {d} from {cur} to {new}"
                )));
            }
            if let Some(ref max) = ds.dataspace.max_dims {
                if new > max[d] {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "extend_dataset dimension {d} ({new}) exceeds the maximum {}",
                        max[d]
                    )));
                }
            }
        }
        ds.dataspace.dims = new_dims.to_vec();
        Ok(())
    }

    /// Set the logical extent of a chunked dataset, growing **or shrinking**
    /// any dimension (unlike [`extend_dataset`](Self::extend_dataset), which
    /// only grows).
    ///
    /// Shrinking sets the logical dataspace only: chunks (or parts of
    /// chunks) beyond the new extent stay in the file but are no longer
    /// visible on read, exactly as libhdf5's `H5Dset_extent` behaves. This
    /// is how a partial multi-frame chunk's over-extended frame count is
    /// corrected back to the true number of frames written.
    pub fn set_dataset_extent(&mut self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = &mut self.datasets[index];
        if ds.chunked.is_none() && ds.fixed_array.is_none() && ds.btree_v2.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "can only set the extent of chunked datasets".into(),
            ));
        }
        if new_dims.len() != ds.dataspace.dims.len() {
            return Err(crate::io::IoError::InvalidState(format!(
                "set_extent rank mismatch: dataset has {} dimensions, got {}",
                ds.dataspace.dims.len(),
                new_dims.len()
            )));
        }
        // A pending append buffer is positioned relative to the current
        // logical size; changing the extent underneath it would make
        // `flush_append_buffers` write the chunk at the wrong index.
        if ds.append_buffered_frames > 0 {
            return Err(crate::io::IoError::InvalidState(
                "set_extent cannot run while the dataset has buffered appends; \
                 flush them first"
                    .into(),
            ));
        }
        if let Some(ref max) = ds.dataspace.max_dims {
            for (d, (&new, &m)) in new_dims.iter().zip(max).enumerate() {
                if new > m {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "set_extent dimension {d} ({new}) exceeds the maximum {m}"
                    )));
                }
            }
        }
        ds.dataspace.dims = new_dims.to_vec();
        Ok(())
    }

    /// Flush a chunked dataset's index structures to disk.
    pub fn flush_dataset(&mut self, index: usize) -> IoResult<()> {
        let ds = &self.datasets[index];

        // EA-indexed dataset
        if let Some(ref chunked) = ds.chunked {
            if let Some(ref fiblk) = chunked.filt_iblk {
                // Filtered EA
                let iblk_encoded = fiblk.encode(&self.ctx, chunked.chunk_size_len);
                self.handle.write_at(chunked.ea_iblk_addr, &iblk_encoded)?;
            } else {
                // Unfiltered EA
                let iblk_encoded = chunked.ea_iblk.encode(&self.ctx);
                self.handle.write_at(chunked.ea_iblk_addr, &iblk_encoded)?;
            }
            let hdr_encoded = chunked.ea_header.encode(&self.ctx);
            self.handle.write_at(chunked.ea_header_addr, &hdr_encoded)?;
            self.handle.sync_data()?;
            return Ok(());
        }

        // Fixed-array-indexed dataset
        if let Some(ref fa) = ds.fixed_array {
            let dblk_encoded = encode_fixed_array_dblk(&self.ctx, &fa.fa_header, &fa.fa_dblk);
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
            let mut hdr =
                crate::format::chunk_index::btree_v2::Bt2Header::decode(&hdr_bytes, &self.ctx)?;
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
        let root_addr = self
            .root_group_addr
            .ok_or_else(|| crate::io::IoError::InvalidState("root group not yet written".into()))?;
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
        let addr = self.datasets[index]
            .obj_header_written_addr
            .ok_or_else(|| {
                crate::io::IoError::InvalidState("dataset header not yet written".into())
            })?;
        let original_size = self.datasets[index].obj_header_encoded_size;

        let header = self.build_dataset_header(index);
        let encoded = header.encode();

        if encoded.len() > original_size {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset header grew from {} to {} bytes; cannot rewrite in place",
                original_size,
                encoded.len()
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

        // 1b. Group object headers. A hard link can point to a group whose
        // header is written later, so addresses are assigned in a first
        // pass (a header's encoded size is independent of the address
        // values it carries) and the content is written in a second.
        for gi in 0..self.groups.len() {
            let size = self.build_group_header(gi).encode().len() as u64;
            self.groups[gi].obj_header_addr = self.allocator.allocate(size);
        }
        for gi in 0..self.groups.len() {
            let encoded = self.build_group_header(gi).encode();
            self.handle
                .write_at(self.groups[gi].obj_header_addr, &encoded)?;
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

    /// Flush any partial append buffers, padding each chunk's unwritten
    /// tail with the dataset's fill value (zeros when none is defined).
    fn flush_append_buffers(&mut self) -> IoResult<()> {
        for i in 0..self.datasets.len() {
            if self.datasets[i].append_buffer.is_empty() {
                continue;
            }
            let chunk_dims = if let Some(ref c) = self.datasets[i].chunked {
                c.chunk_dims.clone()
            } else if let Some(ref f) = self.datasets[i].fixed_array {
                f.chunk_dims.clone()
            } else if let Some(ref b) = self.datasets[i].btree_v2 {
                b.chunk_dims.clone()
            } else {
                continue;
            };
            let es = self.datasets[i].datatype.element_size() as usize;
            let chunk_bytes: usize = chunk_dims.iter().map(|&d| d as usize).product::<usize>() * es;
            let chunk_dim0 = chunk_dims[0] as usize;
            let buffered_frames = self.datasets[i].append_buffered_frames as usize;
            let current_dim0 = self.datasets[i].dataspace.dims[0] as usize;
            let base_frame = current_dim0 - buffered_frames;
            let chunk_idx = base_frame / chunk_dim0;

            let buf = std::mem::take(&mut self.datasets[i].append_buffer);
            self.datasets[i].append_buffered_frames = 0;

            let mut chunk_buf = self.new_chunk_buffer(i, chunk_bytes);
            let frame_bytes = if self.datasets[i].dataspace.dims.len() > 1 {
                self.datasets[i].dataspace.dims[1..]
                    .iter()
                    .map(|&d| d as usize)
                    .product::<usize>()
                    * es
            } else {
                es
            };
            let offset_in_chunk = (base_frame % chunk_dim0) * frame_bytes;
            chunk_buf[offset_in_chunk..offset_in_chunk + buf.len()].copy_from_slice(&buf);
            self.write_chunk(i, chunk_idx as u64, &chunk_buf)?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> IoResult<()> {
        // Flush any partial append buffers before finalizing
        self.flush_append_buffers()?;

        // A SWMR session (`finalize_for_swmr` already ran, so
        // `root_group_addr` is `Some`) is closed by the same full finalize as
        // a fresh write: every object header is rebuilt at a fresh address and
        // the superblock is written with clean-close flags. A full rebuild —
        // rather than the in-place header rewrite used by the live
        // `SwmrWriter::flush` path — is required so any structural change made
        // after `start_swmr` is committed to the final file. A hard link, in
        // particular, both grows its target's header with an object
        // reference-count message and adds a `MSG_LINK` record to a group
        // header; an in-place rewrite cannot accommodate the grown header and
        // never re-emits group/root headers. The fall-through below already
        // handles datasets whose header was written by `finalize_for_swmr`
        // (`obj_header_written_addr.is_some()`).

        // 0. Flush chunked dataset index structures (only modified datasets).
        for i in 0..self.datasets.len() {
            if self.datasets[i].obj_header_written_addr.is_some() {
                let modified = self.datasets[i]
                    .chunked
                    .as_ref()
                    .is_some_and(|c| c.chunks_written > 0);
                if !modified {
                    continue;
                }
            }
            if self.datasets[i].chunked.is_some()
                || self.datasets[i].fixed_array.is_some()
                || self.datasets[i].btree_v2.is_some()
            {
                self.flush_dataset(i)?;
            }
        }

        // 1. Write each dataset's object header.
        for i in 0..self.datasets.len() {
            if self.datasets[i].obj_header_written_addr.is_some() {
                // Existing dataset from append mode.
                // If it has chunked info with chunks_written > 0, it was modified
                // and needs a new object header.
                let modified = self.datasets[i]
                    .chunked
                    .as_ref()
                    .is_some_and(|c| c.chunks_written > 0);
                if !modified {
                    // Keep the original object header address for the root group link.
                    self.datasets[i].obj_header_addr =
                        self.datasets[i].obj_header_written_addr.unwrap();
                    continue;
                }
            }
            let ds_header = self.build_dataset_header(i);
            let encoded = ds_header.encode();
            let addr = self.allocator.allocate(encoded.len() as u64);
            self.handle.write_at(addr, &encoded)?;
            self.datasets[i].obj_header_addr = addr;
        }

        // 1b. Group object headers. A hard link can point to a group whose
        // header is written later, so addresses are assigned in a first
        // pass (a header's encoded size is independent of the address
        // values it carries) and the content is written in a second.
        for gi in 0..self.groups.len() {
            let size = self.build_group_header(gi).encode().len() as u64;
            self.groups[gi].obj_header_addr = self.allocator.allocate(size);
        }
        for gi in 0..self.groups.len() {
            let encoded = self.build_group_header(gi).encode();
            self.handle
                .write_at(self.groups[gi].obj_header_addr, &encoded)?;
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
        let alloc_time = if is_chunked { 3 } else { 2 }; // 3 = incremental, 2 = late
        let fv = if let Some(ref bytes) = ds.fill_value {
            // User-defined fill value (fill_defined = 2).
            FillValueMessage {
                alloc_time,
                fill_write_time: 0, // on alloc
                fill_defined: 2,
                fill_value: Some(bytes.clone()),
            }
        } else if is_chunked {
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
            DataLayoutMessage::chunked_v4_btree_v2(layout_dims, bt2.bt2_header_addr)
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

        // Object Reference Count message (type 0x16): emitted only when
        // more than one hard link resolves to this dataset.
        let rc = self.object_link_count(HardLinkTarget::Dataset(index));
        if rc > 1 {
            header.add_message(MSG_OBJ_REF_COUNT, 0x00, encode_refcount(rc));
        }

        header
    }

    /// Build the object header for a subgroup.
    fn build_group_header(&self, group_idx: usize) -> ObjectHeader {
        let mut header = ObjectHeader::new();

        // Link Info message (type 0x02) -- compact storage
        let link_info = LinkInfoMessage::compact();
        let li_msg = link_info.encode(&self.ctx);
        header.add_message(MSG_LINK_INFO, 0x00, li_msg);

        // Group Info message (type 0x0A) -- defaults
        let group_info = GroupInfoMessage::default();
        let gi_msg = group_info.encode();
        header.add_message(MSG_GROUP_INFO, 0x00, gi_msg);

        let grp = &self.groups[group_idx];

        // Link messages for child datasets (skip deleted)
        for &ds_idx in &grp.child_datasets {
            let ds = &self.datasets[ds_idx];
            if ds.deleted {
                continue;
            }
            let leaf_name = ds.name.rsplit('/').next().unwrap_or(&ds.name);
            let link = LinkMessage::hard(leaf_name, ds.obj_header_addr);
            let link_msg = link.encode(&self.ctx);
            header.add_message(MSG_LINK, 0x00, link_msg);
        }

        // Link messages for child groups (skip deleted)
        for &child_idx in &grp.child_groups {
            let child_grp = &self.groups[child_idx];
            if child_grp.deleted {
                continue;
            }
            let leaf_name = child_grp.name.rsplit('/').next().unwrap_or(&child_grp.name);
            let link = LinkMessage::hard(leaf_name, child_grp.obj_header_addr);
            let link_msg = link.encode(&self.ctx);
            header.add_message(MSG_LINK, 0x00, link_msg);
        }

        // User-created hard links whose parent is this group.
        self.emit_hard_links(&mut header, Some(group_idx));

        // Attribute messages (type 0x0C) -- e.g. NeXus `NX_class`.
        for attr in &grp.attributes {
            let attr_msg = attr.encode(&self.ctx);
            header.add_message(MSG_ATTRIBUTE, 0x00, attr_msg);
        }

        // Object Reference Count message: emitted only when this group is
        // itself a hard-link target reached by more than one link.
        let rc = self.object_link_count(HardLinkTarget::Group(group_idx));
        if rc > 1 {
            header.add_message(MSG_OBJ_REF_COUNT, 0x00, encode_refcount(rc));
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

        // Collect dataset indices that belong to the root group (not assigned to any subgroup)
        let datasets_in_subgroups: std::collections::HashSet<usize> = self
            .groups
            .iter()
            .filter(|g| !g.deleted)
            .flat_map(|g| g.child_datasets.iter().copied())
            .collect();

        // Link messages for root-level datasets
        for (i, ds) in self.datasets.iter().enumerate() {
            if ds.deleted {
                continue;
            }
            if !datasets_in_subgroups.contains(&i) {
                let link = LinkMessage::hard(&ds.name, ds.obj_header_addr);
                let link_msg = link.encode(&self.ctx);
                header.add_message(MSG_LINK, 0x00, link_msg);
            }
        }

        // Link messages for root-level groups (those with no parent)
        for grp in &self.groups {
            if grp.deleted {
                continue;
            }
            if grp.parent.is_none() {
                let leaf_name = grp.name.rsplit('/').next().unwrap_or(&grp.name);
                let link = LinkMessage::hard(leaf_name, grp.obj_header_addr);
                let link_msg = link.encode(&self.ctx);
                header.add_message(MSG_LINK, 0x00, link_msg);
            }
        }

        // User-created hard links in the root group.
        self.emit_hard_links(&mut header, None);

        // Root-level attributes
        for attr in &self.root_attributes {
            let attr_msg = attr.encode(&self.ctx);
            header.add_message(MSG_ATTRIBUTE, 0x00, attr_msg);
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
    use crate::format::messages::datatype::DatatypeMessage;
    use crate::io::reader::Hdf5Reader;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rust_hdf5_w_{}_{}_{}.h5",
            std::process::id(),
            tag,
            n
        ))
    }

    #[test]
    fn create_empty_file() {
        let path = temp_path("empty");

        let writer = Hdf5Writer::create(&path).unwrap();
        writer.close().unwrap();

        // Verify we can read it back
        let reader = Hdf5Reader::open(&path).unwrap();
        assert!(reader.dataset_names().is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_single_dataset() {
        let path = temp_path("single");

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
        let path = temp_path("multi");

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
        let path = temp_path("mismatch");

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
        let path = temp_path("chunked_simple");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_chunked_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 4],        // start empty
                &[u64::MAX, 4], // unlimited first dim
                &[1, 4],        // chunk = [1, 4]
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
        let values: Vec<f64> = raw
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 12);
        for (i, val) in values.iter().enumerate() {
            assert_eq!(*val, i as f64);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn chunked_dataset_many_frames() {
        let path = temp_path("chunked_many");

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
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 20);
        for (i, val) in values.iter().enumerate() {
            assert_eq!(*val, i as i32);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_fixed_array_dataset_roundtrip() {
        let path = temp_path("fixed_array");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_fixed_array_dataset(
                "grid",
                DatatypeMessage::i32_type(),
                &[4, 6], // 4x6 grid
                &[2, 3], // chunk = 2x3
            )
            .unwrap();

        // Write all chunks: 2x2 = 4 chunks
        // chunk (0,0): rows 0-1, cols 0-2
        let c00: Vec<u8> = [0i32, 1, 2, 6, 7, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[0, 0], &c00).unwrap();

        // chunk (0,1): rows 0-1, cols 3-5
        let c01: Vec<u8> = [3i32, 4, 5, 9, 10, 11]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[0, 1], &c01).unwrap();

        // chunk (1,0): rows 2-3, cols 0-2
        let c10: Vec<u8> = [12i32, 13, 14, 18, 19, 20]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[1, 0], &c10).unwrap();

        // chunk (1,1): rows 2-3, cols 3-5
        let c11: Vec<u8> = [15i32, 16, 17, 21, 22, 23]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
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
        for (i, val) in values.iter().enumerate() {
            assert_eq!(*val, i as i32);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fixed_array_paged_dblk_disk_size() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        // 1024 elements per page (bits=10). 3000 chunks => 3 pages.
        let hdr = FixedArrayHeader::new_for_chunks(&ctx, 3000);
        assert!(hdr.is_paged());
        assert_eq!(hdr.npages(), 3);
        // prefix: 4+1+1+8 + bitmap(1) + cksum(4) = 19
        // elements: 3000 * 8 = 24000 ; per-page cksum: 3 * 4 = 12
        assert_eq!(fixed_array_dblk_disk_size(&ctx, &hdr), 19 + 24000 + 12);

        // Non-paged: 1000 elements. prefix(14) + 1000*8 + cksum(4).
        let small = FixedArrayHeader::new_for_chunks(&ctx, 1000);
        assert!(!small.is_paged());
        assert_eq!(fixed_array_dblk_disk_size(&ctx, &small), 14 + 8000 + 4);
    }

    #[test]
    fn fixed_array_paged_encode_matches_reader_layout() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let mut hdr = FixedArrayHeader::new_for_chunks(&ctx, 2500);
        hdr.data_blk_addr = 0x9000;
        let npages = hdr.npages() as usize; // ceil(2500/1024) = 3

        let mut dblk = FixedArrayDataBlock::new_unfiltered(0x1000, 2500);
        for (i, e) in dblk.elements.iter_mut().enumerate() {
            *e = 0x10000 + (i as u64) * 0x100;
        }

        let encoded = encode_fixed_array_dblk(&ctx, &hdr, &dblk);
        assert_eq!(encoded.len() as u64, fixed_array_dblk_disk_size(&ctx, &hdr));

        // Decode the prefix and pages exactly as the reader does.
        let prefix = FixedArrayPagedPrefix::decode(&encoded, &ctx, npages as u64).unwrap();
        assert_eq!(prefix.header_addr, 0x1000);
        for p in 0..npages {
            assert!(prefix.page_initialized(p), "page {p} should be initialized");
        }

        let dblk_page_nelmts = hdr.dblk_page_nelmts() as usize;
        let page_stride = dblk_page_nelmts * 8 + 4;
        let mut recovered = Vec::new();
        for p in 0..npages {
            let page_nelmts = if p + 1 == npages {
                2500 - p * dblk_page_nelmts
            } else {
                dblk_page_nelmts
            };
            let off = prefix.prefix_size + p * page_stride;
            let page_buf = &encoded[off..];
            let addrs = crate::format::chunk_index::fixed_array::decode_unfiltered_page(
                page_buf,
                &ctx,
                page_nelmts,
            )
            .unwrap();
            recovered.extend(addrs);
        }
        assert_eq!(recovered, dblk.elements);
    }

    #[test]
    fn create_fixed_array_paged_dataset_roundtrip() {
        let path = temp_path("fixed_array_paged");

        // 1D dataset of 3000 elements, chunk size 1 => 3000 chunks.
        // 3000 > 1024 (one page) => the FA data block must be paged.
        let n: usize = 3000;
        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_fixed_array_dataset("paged", DatatypeMessage::i32_type(), &[n as u64], &[1])
            .unwrap();

        for i in 0..n {
            let v = (i as i32).to_le_bytes();
            writer
                .write_chunk_fixed_array(idx, &[i as u64], &v)
                .unwrap();
        }
        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("paged").unwrap(), vec![n as u64]);
        let raw = reader.read_dataset_raw("paged").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), n);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as i32, "element {i}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn create_filtered_fixed_array_dataset_roundtrip() {
        // Small compressed fixed-shape chunked dataset: flat filtered FA.
        let path = temp_path("fixed_array_filt");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_fixed_array_dataset_with_pipeline(
                "grid",
                DatatypeMessage::i32_type(),
                &[4, 6], // 4x6 grid
                &[2, 3], // chunk = 2x3 => 2x2 = 4 chunks
                FilterPipeline::deflate(6),
            )
            .unwrap();

        let c00: Vec<u8> = [0i32, 1, 2, 6, 7, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[0, 0], &c00).unwrap();
        let c01: Vec<u8> = [3i32, 4, 5, 9, 10, 11]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[0, 1], &c01).unwrap();
        let c10: Vec<u8> = [12i32, 13, 14, 18, 19, 20]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[1, 0], &c10).unwrap();
        let c11: Vec<u8> = [15i32, 16, 17, 21, 22, 23]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_fixed_array(idx, &[1, 1], &c11).unwrap();

        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("grid").unwrap(), vec![4, 6]);
        let raw = reader.read_dataset_raw("grid").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 24);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as i32, "element {i}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn create_filtered_fixed_array_paged_dataset_roundtrip() {
        // Large compressed fixed-shape chunked dataset (>1024 chunks): the
        // filtered FA data block must be paged.
        let path = temp_path("fixed_array_filt_paged");

        let n: usize = 3000;
        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_fixed_array_dataset_with_pipeline(
                "paged",
                DatatypeMessage::i32_type(),
                &[n as u64],
                &[1],
                FilterPipeline::deflate(6),
            )
            .unwrap();

        for i in 0..n {
            let v = (i as i32).to_le_bytes();
            writer
                .write_chunk_fixed_array(idx, &[i as u64], &v)
                .unwrap();
        }
        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("paged").unwrap(), vec![n as u64]);
        let raw = reader.read_dataset_raw("paged").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), n);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as i32, "element {i}");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn filtered_fixed_array_dblk_disk_size_and_encode() {
        // Cross-check filtered FA data-block sizing against the encoded length,
        // for both flat and paged layouts.
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let csl = 3u8; // chunk_size_len
        let elem_size = 8 + csl as usize + 4; // addr + size + filter_mask

        // Flat: 100 chunks. prefix(14) + 100*elem_size + cksum(4).
        let mut flat = FixedArrayHeader::new_for_filtered_chunks(&ctx, 100, csl);
        flat.data_blk_addr = 0x4000;
        assert!(!flat.is_paged());
        assert_eq!(
            fixed_array_dblk_disk_size(&ctx, &flat),
            (14 + 100 * elem_size + 4) as u64
        );
        let flat_dblk = FixedArrayDataBlock::new_filtered(0x1000, 100);
        assert_eq!(
            encode_fixed_array_dblk(&ctx, &flat, &flat_dblk).len() as u64,
            fixed_array_dblk_disk_size(&ctx, &flat)
        );

        // Paged: 2500 chunks => 3 pages. prefix(4+1+1+8+1+4=19)
        // + 2500*elem_size + 3*cksum(4).
        let mut paged = FixedArrayHeader::new_for_filtered_chunks(&ctx, 2500, csl);
        paged.data_blk_addr = 0x9000;
        assert!(paged.is_paged());
        assert_eq!(paged.npages(), 3);
        assert_eq!(
            fixed_array_dblk_disk_size(&ctx, &paged),
            (19 + 2500 * elem_size + 12) as u64
        );
        let mut paged_dblk = FixedArrayDataBlock::new_filtered(0x1000, 2500);
        for (i, e) in paged_dblk.filtered_elements.iter_mut().enumerate() {
            e.address = 0x10000 + (i as u64) * 0x100;
            e.chunk_size = (i % 200) as u32;
        }
        let encoded = encode_fixed_array_dblk(&ctx, &paged, &paged_dblk);
        assert_eq!(
            encoded.len() as u64,
            fixed_array_dblk_disk_size(&ctx, &paged)
        );

        // Decode the paged prefix + pages as the reader does.
        let npages = paged.npages() as usize;
        let prefix = FixedArrayPagedPrefix::decode(&encoded, &ctx, npages as u64).unwrap();
        for p in 0..npages {
            assert!(prefix.page_initialized(p), "page {p}");
        }
        let dblk_page_nelmts = paged.dblk_page_nelmts() as usize;
        let page_stride = dblk_page_nelmts * elem_size + 4;
        let mut recovered = Vec::new();
        for p in 0..npages {
            let page_nelmts = if p + 1 == npages {
                2500 - p * dblk_page_nelmts
            } else {
                dblk_page_nelmts
            };
            let off = prefix.prefix_size + p * page_stride;
            let elems = crate::format::chunk_index::fixed_array::decode_filtered_page(
                &encoded[off..],
                &ctx,
                page_nelmts,
                csl as usize,
            )
            .unwrap();
            recovered.extend(elems);
        }
        assert_eq!(recovered, paged_dblk.filtered_elements);
    }

    #[test]
    fn create_btree_v2_dataset_roundtrip() {
        let path = temp_path("btree_v2");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_btree_v2_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 0],               // start empty
                &[u64::MAX, u64::MAX], // both dims unlimited
                &[2, 3],               // chunk = 2x3
            )
            .unwrap();

        // Write chunks for a 4x6 dataset
        // chunk (0,0)
        let c00: Vec<u8> = [0.0f64, 1.0, 2.0, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_btree_v2(idx, &[0, 0], &c00).unwrap();

        // chunk (0,1)
        let c01: Vec<u8> = [3.0f64, 4.0, 5.0, 9.0, 10.0, 11.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_btree_v2(idx, &[0, 1], &c01).unwrap();

        // chunk (1,0)
        let c10: Vec<u8> = [12.0f64, 13.0, 14.0, 18.0, 19.0, 20.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_chunk_btree_v2(idx, &[1, 0], &c10).unwrap();

        // chunk (1,1)
        let c11: Vec<u8> = [15.0f64, 16.0, 17.0, 21.0, 22.0, 23.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
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
        for (i, val) in values.iter().enumerate() {
            assert_eq!(*val, i as f64);
        }

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_batch_write_roundtrip() {
        let path = temp_path("parallel_batch");

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
        for (i, val) in values.iter().enumerate() {
            assert_eq!(*val, i as i32);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn swmr_writer_append_frames() {
        use crate::io::swmr::SwmrWriter;

        // Per-call unique path so concurrent cargo invocations and
        // kernel-side flock release races cannot collide.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_append_{}_{}.h5",
            std::process::id(),
            n
        ));

        let mut swmr = SwmrWriter::create(&path).unwrap();
        let idx = swmr
            .create_streaming_dataset("detector", DatatypeMessage::u16_type(), &[4, 4])
            .unwrap();

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
        let values: Vec<u16> = raw
            .chunks(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 80); // 5 * 4 * 4
                                      // Verify first frame
        for (i, val) in values.iter().enumerate().take(16) {
            assert_eq!(*val, i as u16);
        }
        // Verify last frame
        for (i, val) in values[64..80].iter().enumerate() {
            assert_eq!(*val, 4 * 16 + i as u16);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn swmr_writer_tiled_frames() {
        use crate::io::swmr::SwmrWriter;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_tiled_{}_{}.h5",
            std::process::id(),
            n
        ));

        let mut swmr = SwmrWriter::create(&path).unwrap();
        // 4x4 frames, tiled into 2x2 chunks -> 4 chunks per frame.
        let idx = swmr
            .create_streaming_dataset_tiled("det", DatatypeMessage::u16_type(), &[4, 4], &[2, 2])
            .unwrap();
        swmr.start_swmr().unwrap();

        for frame in 0..3u16 {
            let data: Vec<u16> = (0..16).map(|i| frame * 100 + i).collect();
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            swmr.append_frame(idx, &raw).unwrap();
        }
        swmr.flush().unwrap();
        swmr.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("det").unwrap(), vec![3, 4, 4]);
        let raw = reader.read_dataset_raw("det").unwrap();
        let values: Vec<u16> = raw
            .chunks(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 48);
        // Every element must survive the frame -> tile split and the
        // tile -> frame reassembly on read.
        for frame in 0..3u16 {
            for i in 0..16usize {
                assert_eq!(values[frame as usize * 16 + i], frame * 100 + i as u16);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn swmr_writer_tiled_chunk_larger_than_frame() {
        use crate::io::swmr::SwmrWriter;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_bigchunk_{}_{}.h5",
            std::process::id(),
            n
        ));

        // Chunk tile larger than the frame: a 1x1 chunk grid, but the frame
        // must still be zero-padded up to the full chunk size.
        let mut swmr = SwmrWriter::create(&path).unwrap();
        let idx = swmr
            .create_streaming_dataset_tiled("det", DatatypeMessage::u16_type(), &[3, 3], &[8, 8])
            .unwrap();
        swmr.start_swmr().unwrap();
        for frame in 0..2u16 {
            let data: Vec<u16> = (0..9).map(|i| frame * 10 + i).collect();
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            swmr.append_frame(idx, &raw).unwrap();
        }
        swmr.flush().unwrap();
        swmr.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("det").unwrap(), vec![2, 3, 3]);
        let raw = reader.read_dataset_raw("det").unwrap();
        let values: Vec<u16> = raw
            .chunks(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 18);
        for frame in 0..2u16 {
            for i in 0..9usize {
                assert_eq!(values[frame as usize * 9 + i], frame * 10 + i as u16);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn swmr_writer_multi_frame_chunks() {
        use crate::io::swmr::SwmrWriter;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_mfc_{}_{}.h5",
            std::process::id(),
            n
        ));

        // 3x3 frames, chunk = 4 frames x full frame. 10 frames -> 3 bands
        // of 4, 4, 2 (the last band partial).
        let mut swmr = SwmrWriter::create(&path).unwrap();
        let idx = swmr
            .create_streaming_dataset_chunked(
                "det",
                DatatypeMessage::u16_type(),
                &[3, 3],
                &[4, 3, 3],
            )
            .unwrap();
        swmr.start_swmr().unwrap();
        for frame in 0..10u16 {
            let data: Vec<u16> = (0..9).map(|i| frame * 100 + i).collect();
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            swmr.append_frame(idx, &raw).unwrap();
        }
        swmr.flush().unwrap();
        swmr.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        // The partial last band must not over-extend the frame count.
        assert_eq!(reader.dataset_shape("det").unwrap(), vec![10, 3, 3]);
        let raw = reader.read_dataset_raw("det").unwrap();
        let values: Vec<u16> = raw
            .chunks(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 90);
        for frame in 0..10u16 {
            for i in 0..9usize {
                assert_eq!(values[frame as usize * 9 + i], frame * 100 + i as u16);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn swmr_writer_multi_frame_tiled_chunks() {
        use crate::io::swmr::SwmrWriter;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_mftc_{}_{}.h5",
            std::process::id(),
            n
        ));

        // 4x4 frames, chunk = 2 frames x 2x2 tiles. 5 frames -> bands of
        // 2, 2, 1; every frame is also split into a 2x2 tile grid.
        let mut swmr = SwmrWriter::create(&path).unwrap();
        let idx = swmr
            .create_streaming_dataset_chunked(
                "det",
                DatatypeMessage::u16_type(),
                &[4, 4],
                &[2, 2, 2],
            )
            .unwrap();
        swmr.start_swmr().unwrap();
        for frame in 0..5u16 {
            let data: Vec<u16> = (0..16).map(|i| frame * 100 + i).collect();
            let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            swmr.append_frame(idx, &raw).unwrap();
        }
        swmr.flush().unwrap();
        swmr.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("det").unwrap(), vec![5, 4, 4]);
        let raw = reader.read_dataset_raw("det").unwrap();
        let values: Vec<u16> = raw
            .chunks(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 80);
        for frame in 0..5u16 {
            for i in 0..16usize {
                assert_eq!(values[frame as usize * 16 + i], frame * 100 + i as u16);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn swmr_writer_compressed_frames() {
        use crate::io::swmr::SwmrWriter;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_comp_{}_{}.h5",
            std::process::id(),
            n
        ));

        let mut swmr = SwmrWriter::create(&path).unwrap();
        let pipeline = crate::format::messages::filter::FilterPipeline::deflate(4);
        let idx = swmr
            .create_streaming_dataset_compressed(
                "detector",
                DatatypeMessage::i32_type(),
                &[8],
                pipeline,
            )
            .unwrap();
        swmr.start_swmr().unwrap();

        for frame in 0..40i32 {
            let raw: Vec<u8> = (0..8).flat_map(|i| (frame * 8 + i).to_le_bytes()).collect();
            swmr.append_frame(idx, &raw).unwrap();
            if frame % 7 == 0 {
                swmr.flush().unwrap();
            }
        }
        swmr.flush().unwrap();
        swmr.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("detector").unwrap(), vec![40, 8]);
        let raw = reader.read_dataset_raw("detector").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, (0..320).collect::<Vec<i32>>());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn group_hierarchy_writer_reader() {
        let path = temp_path("group_hierarchy");

        let mut writer = Hdf5Writer::create(&path).unwrap();

        // Create groups
        let g0 = writer.create_group("/", "group1").unwrap();
        let g1 = writer.create_group("/group1", "sub").unwrap();
        assert_eq!(g0, 0);
        assert_eq!(g1, 1);

        // Create datasets
        let ds_root = writer
            .create_dataset("root_data", DatatypeMessage::f64_type(), &[2])
            .unwrap();
        let raw_root: Vec<u8> = [1.0f64, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_dataset_raw(ds_root, &raw_root).unwrap();

        let ds_g0 = writer
            .create_dataset("group1/data", DatatypeMessage::i32_type(), &[3])
            .unwrap();
        writer.assign_dataset_to_group("/group1", ds_g0).unwrap();
        let raw_g0: Vec<u8> = [10i32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_dataset_raw(ds_g0, &raw_g0).unwrap();

        let ds_g1 = writer
            .create_dataset("group1/sub/values", DatatypeMessage::u8_type(), &[4])
            .unwrap();
        writer
            .assign_dataset_to_group("/group1/sub", ds_g1)
            .unwrap();
        writer.write_dataset_raw(ds_g1, &[1u8, 2, 3, 4]).unwrap();

        writer.close().unwrap();

        // Read back
        let mut reader = Hdf5Reader::open(&path).unwrap();
        let names = reader.dataset_names();
        assert!(names.contains(&"root_data"), "names: {:?}", names);
        assert!(names.contains(&"group1/data"), "names: {:?}", names);
        assert!(names.contains(&"group1/sub/values"), "names: {:?}", names);

        let raw = reader.read_dataset_raw("root_data").unwrap();
        let vals: Vec<f64> = raw
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0]);

        let raw = reader.read_dataset_raw("group1/data").unwrap();
        let vals: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![10, 20, 30]);

        let raw = reader.read_dataset_raw("group1/sub/values").unwrap();
        assert_eq!(raw, vec![1, 2, 3, 4]);

        std::fs::remove_file(&path).ok();
    }
}
