//! HDF5 file writer.
//!
//! Produces a valid HDF5 file with superblock v3, a root group object header,
//! and datasets with contiguous or chunked storage. The output is readable by `h5dump`.

use std::collections::HashMap;
use std::path::Path;

use crate::format::chunk_index::btree_v2::Bt2ChunkIndex;
use crate::format::chunk_index::extensible_array::{
    compute_chunk_size_len, compute_ndblk_addrs, compute_nsblk_addrs, EaDblkPath, EaGeometry,
    EaLoc, ExtensibleArrayDataBlock, ExtensibleArrayHeader, ExtensibleArrayIndexBlock,
    ExtensibleArraySuperBlock, FilteredChunkEntry, FilteredDataBlock, FilteredIndexBlock,
    EA_CLS_CHUNK, EA_CLS_FILT_CHUNK,
};
use crate::format::chunk_index::fixed_array::{
    decode_filtered_page, decode_unfiltered_page, encode_filtered_page, encode_unfiltered_page,
    FixedArrayDataBlock, FixedArrayFilteredChunkElement, FixedArrayHeader, FixedArrayPagedPrefix,
    FA_CLIENT_FILT_CHUNK,
};
use crate::format::creation_order::CreationOrder;
use crate::format::dense_attr::build_dense_attributes;
use crate::format::dense_link::build_dense_links;
use crate::format::messages::attr_info::{next_creation_index, AttributeInfoMessage};
use crate::format::messages::attribute::{AttributeEntry, AttributeMessage};
use crate::format::messages::data_layout::{DataLayoutMessage, EarrayParams, FixedArrayParams};
use crate::format::messages::dataspace::{DataspaceClass, DataspaceMessage};
use crate::format::messages::datatype::{DatatypeMessage, ReferenceKind};
use crate::format::messages::fill_value::FillValueMessage;
use crate::format::messages::filter::{self, FilterPipeline};
use crate::format::messages::group_info::GroupInfoMessage;
use crate::format::messages::link::{LinkMessage, LinkTarget};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::*;
use crate::format::object_header::{ObjectHeader, ObjectTimes, MAX_MESSAGE_SIZE};
use crate::format::superblock::*;
use crate::format::{FormatContext, LibverBound, ObjectFormat, UNDEF_ADDR};

use crate::io::allocator::FileAllocator;
use crate::io::file_handle::FileHandle;
use crate::io::hyperslab::{for_each_contiguous_run, for_each_dual_run};
use crate::io::symbol_table_io::{free_stab, write_stab, Stab, StabExtents, StabLink, StabTarget};
use crate::io::{FileMeta, IoResult};

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

/// Walk a v2 B-tree from `addr`, collecting every node's raw record bytes
/// and every node block's address — the reader's record walk plus the
/// addresses, which `open_append` needs so the reconstructed
/// [`Bt2DatasetInfo::node_addrs`] pool owns the on-disk nodes (the next
/// flush re-serializes the tree over them, and a delete frees them).
#[allow(clippy::too_many_arguments)]
fn collect_bt2_nodes(
    handle: &FileHandle,
    ctx: &FormatContext,
    addr: u64,
    depth: u16,
    nrec: u16,
    record_size: u16,
    node_size: u32,
    geo: &crate::format::chunk_index::btree_v2::Bt2Geometry,
    records: &mut Vec<u8>,
    node_addrs: &mut Vec<u64>,
) -> IoResult<()> {
    use crate::format::chunk_index::btree_v2::{Bt2InternalNode, Bt2LeafNode};

    node_addrs.push(addr);
    let buf = handle.read_at_most(addr, node_size as usize)?;
    if depth == 0 {
        let leaf = Bt2LeafNode::decode(&buf, nrec, record_size)?;
        records.extend_from_slice(&leaf.record_data);
    } else {
        let node = Bt2InternalNode::decode(
            &buf,
            ctx,
            depth,
            nrec,
            record_size,
            geo.max_nrec_size,
            geo.child_total_size(depth),
        )?;
        // In-order: an internal node's records separate its children, so each
        // one belongs between the subtrees on either side of it.
        let children: Vec<(u64, u16)> = node
            .child_addrs
            .iter()
            .zip(node.child_nrecords.iter())
            .map(|(&a, &n)| (a, n))
            .collect();
        let rec = record_size as usize;
        for (i, (child_addr, child_nrec)) in children.into_iter().enumerate() {
            collect_bt2_nodes(
                handle,
                ctx,
                child_addr,
                depth - 1,
                child_nrec,
                record_size,
                node_size,
                geo,
                records,
                node_addrs,
            )?;
            if let Some(record) = node.record_data.get(i * rec..(i + 1) * rec) {
                records.extend_from_slice(record);
            }
        }
    }
    Ok(())
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

/// Decode a fixed-array data block for the layout implied by `hdr` — the
/// inverse of [`encode_fixed_array_dblk`], and the single decode dispatch
/// over non-paged/paged × unfiltered/filtered.
///
/// For the paged layout, pages whose bitmap bit is clear are skipped, not
/// decoded: libhdf5 never writes an uninitialized page, so its bytes are
/// arbitrary and carry no valid checksum. Their elements stay at the
/// undefined-address defaults, which is exactly what the bitmap means.
fn decode_fixed_array_dblk(
    ctx: &FormatContext,
    hdr: &FixedArrayHeader,
    buf: &[u8],
    chunk_size_len: usize,
) -> crate::format::FormatResult<FixedArrayDataBlock> {
    let is_filtered = hdr.client_id == FA_CLIENT_FILT_CHUNK;
    let num_elmts = hdr.num_elmts as usize;

    if !hdr.is_paged() {
        return if is_filtered {
            FixedArrayDataBlock::decode_filtered(buf, ctx, num_elmts, chunk_size_len)
        } else {
            FixedArrayDataBlock::decode_unfiltered(buf, ctx, num_elmts)
        };
    }

    let npages = hdr.npages() as usize;
    let dblk_page_nelmts = hdr.dblk_page_nelmts() as usize;
    let prefix = FixedArrayPagedPrefix::decode(buf, ctx, npages as u64)?;

    let mut dblk = if is_filtered {
        FixedArrayDataBlock::new_filtered(prefix.header_addr, num_elmts)
    } else {
        FixedArrayDataBlock::new_unfiltered(prefix.header_addr, num_elmts)
    };
    dblk.client_id = hdr.client_id;

    // Pages follow the prefix back to back; every page spans the full
    // `dblk_page_nelmts` stride except the last, which holds the remainder.
    let mut pos = prefix.prefix_size;
    for p in 0..npages {
        let start = p * dblk_page_nelmts;
        let end = ((p + 1) * dblk_page_nelmts).min(num_elmts);
        let nelmts = end - start;
        if prefix.page_initialized(p) {
            let page_buf = buf.get(pos..).unwrap_or(&[]);
            if is_filtered {
                let elems = decode_filtered_page(page_buf, ctx, nelmts, chunk_size_len)?;
                dblk.filtered_elements[start..end].clone_from_slice(&elems);
            } else {
                let addrs = decode_unfiltered_page(page_buf, ctx, nelmts)?;
                dblk.elements[start..end].copy_from_slice(&addrs);
            }
        }
        pos += nelmts * hdr.element_size as usize + 4;
    }
    Ok(dblk)
}

/// Interior-mutability cell for per-dataset write state, selected by feature.
///
/// This is the §5-B "cfg-selected interior types" from
/// `docs/threadsafe-fine-grained-locking.md`: the single-threaded build uses a
/// `RefCell` (zero overhead, no atomics), while the `threadsafe` build uses a
/// `Mutex` so two threads can write *different* datasets concurrently while the
/// same dataset's writes serialize. Call sites are identical across both via
/// [`Slot::lock`].
#[cfg(not(feature = "threadsafe"))]
pub(crate) struct Slot<T>(std::cell::RefCell<T>);

#[cfg(not(feature = "threadsafe"))]
impl<T> Slot<T> {
    pub(crate) fn new(value: T) -> Self {
        Slot(std::cell::RefCell::new(value))
    }
    /// Borrow the contents mutably (an uncontended `RefCell` borrow).
    pub(crate) fn lock(&self) -> std::cell::RefMut<'_, T> {
        self.0.borrow_mut()
    }
}

#[cfg(feature = "threadsafe")]
pub(crate) struct Slot<T>(std::sync::Mutex<T>);

#[cfg(feature = "threadsafe")]
impl<T> Slot<T> {
    pub(crate) fn new(value: T) -> Self {
        Slot(std::sync::Mutex::new(value))
    }
    /// Lock the contents. Different datasets hold different slots, so this
    /// only contends when two threads write the *same* dataset.
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.0.lock().unwrap()
    }
}

/// Proof that the create gate (`create_lock`) is held and the new dataset's
/// name passed the uniqueness check. Only [`Hdf5Writer::begin_create`]
/// constructs one and [`Hdf5Writer::push_dataset`] demands one, so a creator
/// cannot reach the dataset registry while skipping either step. Carries
/// the canonical (link-resolved) name the creator must store, so the
/// registry only ever holds tree paths.
pub(crate) struct CreateGuard<'a> {
    #[cfg(not(feature = "threadsafe"))]
    _gate: std::cell::RefMut<'a, ()>,
    #[cfg(feature = "threadsafe")]
    _gate: std::sync::MutexGuard<'a, ()>,
    /// The dataset name with every group hard link in it resolved.
    pub(crate) name: String,
    /// The group that will hold the new dataset's link, resolved from the
    /// path components of `name`; `None` is the root group. Carried here so
    /// [`Hdf5Writer::push_dataset`] registers the child itself and no creator
    /// can leave a dataset whose name says one thing and whose parent group
    /// says another.
    pub(crate) parent: Option<usize>,
}

/// Reference-counted shared pointer, feature-selected. The single-thread
/// build uses `Rc` (no atomics); the `threadsafe` build uses `Arc` so a
/// dataset/group slot can be cloned out of the registry and locked on its
/// own — letting writes to *different* datasets proceed concurrently without
/// holding the registry lock. See `docs/threadsafe-fine-grained-locking.md`
/// (Stage 3).
#[cfg(not(feature = "threadsafe"))]
pub(crate) type Shared<T> = std::rc::Rc<T>;
#[cfg(feature = "threadsafe")]
pub(crate) type Shared<T> = std::sync::Arc<T>;

/// One dataset's cell in the registry: its metadata slot plus the operation
/// lock that serializes whole logical operations on it. Both live in one
/// allocation so they cannot fall out of step — every dataset has its op
/// lock by construction.
pub(crate) struct DatasetCell {
    /// Serializes one *whole* logical operation on this dataset.
    ///
    /// The metadata slot below serializes each individual acquisition, but a
    /// multi-acquisition operation — take the append buffer → write chunks →
    /// re-buffer the tail → extend, or flush-then-overwrite in a slice write
    /// — would interleave with a concurrent same-dataset operation *between*
    /// its acquisitions under `threadsafe`. Public write entries take this
    /// lock and delegate to `_inner` variants; `_inner` variants and the
    /// `pub(crate)` write helpers require the caller to hold it (or to hold
    /// the writer exclusively via `&mut`, as close and the SWMR wrapper do).
    ///
    /// Not reentrant: the single-thread build's `RefCell` panics instantly
    /// on a nested acquisition, so a missed entry/inner split fails loudly
    /// in every test run rather than deadlocking only under `threadsafe`.
    ///
    /// Lock order: `create_lock → op → registry spine → metadata slot`. An
    /// op lock is never held across another dataset's op lock, and no
    /// op-lock holder takes `create_lock`, so the order is acyclic.
    pub(crate) op: Slot<()>,
    info: Slot<DatasetInfo>,
}

impl DatasetCell {
    pub(crate) fn new(info: DatasetInfo) -> Self {
        DatasetCell {
            op: Slot::new(()),
            info: Slot::new(info),
        }
    }

    /// Borrow the metadata slot (a single acquisition; see [`Self::op`] for
    /// whole-operation serialization).
    #[cfg(not(feature = "threadsafe"))]
    pub(crate) fn lock(&self) -> std::cell::RefMut<'_, DatasetInfo> {
        self.info.lock()
    }

    /// Lock the metadata slot (a single acquisition; see [`Self::op`] for
    /// whole-operation serialization).
    #[cfg(feature = "threadsafe")]
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, DatasetInfo> {
        self.info.lock()
    }
}

/// A single dataset's [`DatasetCell`], reference-counted so a writer can
/// clone it out of the registry (releasing the registry lock) and then lock
/// just this one dataset. Two threads writing different datasets take
/// different `DatasetRef` locks and never contend; the same dataset's writes
/// serialize, which is required because one chunk index is not concurrently
/// mutable.
pub(crate) type DatasetRef = Shared<DatasetCell>;

/// A single group's metadata behind its own [`Slot`], reference-counted like
/// [`DatasetRef`].
pub(crate) type GroupRef = Shared<Slot<GroupInfo>>;

/// Appended frames held back until they complete a chunk.
///
/// The buffer is the sole authority for rows `base .. base + frames`: the
/// file's chunks do not hold them yet, and any operation that writes those
/// rows must go through [`Hdf5Writer::flush_append_buffer`] first. `base` is
/// recorded when the frames are buffered — never derived from the current
/// extent, which an `extend_dataset` can move independently.
pub struct AppendBuffer {
    /// Absolute row of the first buffered frame.
    pub base: u64,
    /// Number of buffered frames.
    pub frames: u64,
    /// The frames' bytes, `frames` whole rows, row-major.
    pub bytes: Vec<u8>,
}

/// Metadata for a dataset being written.
///
/// The whole struct lives behind a per-dataset [`Slot`] (via [`DatasetRef`]).
/// The streaming write path locks it only briefly — compression runs *outside*
/// the lock — so writes to different datasets do not contend, and a structural
/// op (create/delete) that scans names only momentarily touches a sibling
/// slot.
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
    /// Appended frames not yet written to chunks, `None` when empty.
    pub append: Option<AppendBuffer>,
    /// Attributes attached to this dataset.
    pub attributes: Vec<AttributeEntry>,
    /// File offset where the dataset object header was written (for SWMR in-place rewrites).
    pub obj_header_written_addr: Option<u64>,
    /// Encoded size of the dataset object header (for verifying in-place rewrites fit).
    pub obj_header_encoded_size: usize,
    /// Filter pipeline for compressed chunks.
    pub filter_pipeline: Option<FilterPipeline>,
    /// Soft-deleted: excluded from finalize output.
    pub deleted: bool,
    /// The dataspace extent changed this session (`extend_dataset` /
    /// `set_dataset_extent`). On a reopened dataset the finalize gate
    /// otherwise infers "modified" from `chunks_written` alone, and a
    /// session that only changed the extent would keep the old on-disk
    /// header — silently dropping the new shape.
    pub extent_dirty: bool,
    /// Something the object header encodes changed this session without
    /// touching the dataset's storage — an attribute set or removed, a fill
    /// value defined. See [`header_stale`](DatasetInfo::header_stale).
    pub header_dirty: bool,
    /// The hard link count the on-disk header was written with, so finalize
    /// can tell that this session changed it.
    ///
    /// A count, not a flag, because the count is what the header records and
    /// the ways to change it are many: creating a link, unlinking one,
    /// deleting a link's parent group, promoting a link to a primary name.
    /// Comparing the value closes all of them at once, where a dirty flag
    /// would have to be set at each and would be forgotten at the next one
    /// added.
    pub nlink_written: u32,
    /// When the link naming this dataset was created; see
    /// [`GroupInfo::creation_seq`].
    pub creation_seq: u64,
    /// How this dataset records creation order for its attributes — the
    /// file's creation-order policy captured when the dataset was created,
    /// the way libhdf5 captures the DCPL. A dataset holds no links, so only
    /// the attribute half of [`TrackOrder`] applies to it.
    pub track_attr_order: CreationOrder,
    /// User-defined fill value bytes (exactly one element wide). `None`
    /// means default zero-fill; `Some` is emitted as a `fill_defined = 2`
    /// fill-value message in the dataset object header.
    pub fill_value: Option<Vec<u8>>,
    /// Layout message version for chunked storage: 4, or 5 when the chunk
    /// index encodes stored chunk sizes in a fixed `sizeof_size` field
    /// (libhdf5 2.0). Chosen at create by `Hdf5Writer::chunk_layout_version`,
    /// preserved from the file on reopen, and emitted verbatim at finalize.
    /// Contiguous datasets ignore it.
    pub layout_version: u8,
    /// The four times this object's header stores, when it was created with
    /// `H5Pset_obj_track_times(true)`. `None` for a dataset this writer
    /// created — it offers no way to ask for times — and whatever the file
    /// held for a reopened one, so a rewrite hands them back rather than
    /// dropping the flag that announces them.
    pub times: Option<ObjectTimes>,
}

impl DatasetInfo {
    /// Whether this session wrote chunk data or changed the extent, so the
    /// dataset's index structures have to be re-flushed.
    fn storage_dirty(&self) -> bool {
        self.chunked.as_ref().is_some_and(|c| c.chunks_written > 0)
            || self
                .fixed_array
                .as_ref()
                .is_some_and(|f| f.chunks_written > 0)
            || self.btree_v2.as_ref().is_some_and(|b| b.chunks_written > 0)
            || self.extent_dirty
    }

    /// Whether a reopened dataset's on-disk object header no longer describes
    /// it.
    ///
    /// INVARIANT: every mutation of something `build_dataset_header` encodes
    /// must show up here. Finalize keeps the original header when this is
    /// false, so a change this misses is not deferred — it is discarded, with
    /// no error to say so. Attributes were the case that proved it: they are
    /// invisible to the chunk-write counters, so an attribute set on a
    /// reopened dataset vanished at close.
    fn header_stale(&self) -> bool {
        self.storage_dirty() || self.header_dirty
    }

    /// The same question for the one thing the dataset itself cannot see: how
    /// many hard links resolve to it. That count lives in the header — an
    /// Object Reference Count message in a version-2 header, the `nlink`
    /// prefix field of a version-1 one — but it is a property of the file's
    /// link graph, so the caller supplies today's value.
    fn header_stale_with(&self, nlink: u32) -> bool {
        self.header_stale() || nlink != self.nlink_written
    }
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

/// Which attribute list an attribute operation targets: the root group's,
/// a group's (by full path), or a dataset's (by writer index).
#[derive(Clone, Copy)]
pub enum AttrTarget<'a> {
    /// The root group's (file-level) attributes.
    Root,
    /// A group's attributes, by full path.
    Group(&'a str),
    /// A dataset's attributes, by writer index.
    Dataset(usize),
}

/// Which chunk index a dataset uses. libhdf5 picks it from the dataspace: a
/// v2 B-tree for two or more unlimited dimensions, an extensible array for
/// exactly one, a fixed array for none.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChunkIndexKind {
    ExtensibleArray,
    FixedArray,
    BtreeV2,
}

/// A chunked dataset's grid geometry, snapshotted out of its slot.
///
/// The single owner of chunk-grid arithmetic: how many chunks span each
/// dimension, where a coordinate sits in the row-major order the array
/// indices record, and how many bytes one chunk holds.
struct ChunkGeometry {
    kind: ChunkIndexKind,
    dims: Vec<u64>,
    max_dims: Option<Vec<u64>>,
    chunk_dims: Vec<u64>,
    element_size: u64,
}

impl ChunkGeometry {
    /// Unfiltered byte size of one whole chunk.
    fn chunk_bytes(&self) -> u64 {
        self.chunk_dims.iter().product::<u64>() * self.element_size
    }

    /// Row-major position of `coords` in the chunk grid — the linear index an
    /// extensible or fixed array records the chunk under, computed against
    /// the maximum-extent grid by [`crate::io::chunk_grid::linear_index`].
    fn linear_index(&self, coords: &[u64]) -> IoResult<u64> {
        crate::io::chunk_grid::linear_index(
            &self.dims,
            self.max_dims.as_deref(),
            &self.chunk_dims,
            coords,
        )
    }
}

/// The refusal every attribute mutation gets while SWMR streaming is
/// active, from the two owners of attribute-list change
/// ([`Hdf5Writer::set_attribute`] and `evict_attr`).
fn swmr_attr_error(name: &str) -> crate::io::IoError {
    crate::io::IoError::InvalidState(format!(
        "cannot add or modify attribute '{name}' during SWMR streaming: object \
         headers are frozen while readers stream, and a superseded variable-length \
         value's heap storage could never be reclaimed; set attributes before \
         start_swmr (libhdf5 forbids attribute changes during SWMR writes too)"
    ))
}

/// Where an attribute arriving at [`Hdf5Writer::insert_attribute`] came from.
///
/// The variable-length setters have to evict before they allocate — the
/// free-before-alloc order — so by the time the replacement is inserted the
/// list no longer holds the entry it replaces, and the ordinary "already
/// present, so keep its index" test cannot see it. `H5A__attr_write` does not
/// create the attribute again, so the index travels with the eviction rather
/// than being stamped afresh; without it a rewritten attribute takes the set's
/// running maximum and moves to the end of the creation order.
#[derive(Debug, Clone, Copy)]
enum AttrOrigin {
    /// A new attribute, which takes the set's next creation index.
    Created,
    /// A value written over an attribute this writer has just evicted, which
    /// keeps that attribute's creation index — `None` when the object tracks
    /// no order, and so records none. An eviction that found nothing to remove
    /// answers `Created`: what follows it is a create like any other.
    Rewritten(Option<u16>),
}
use AttrOrigin::{Created, Rewritten};

/// Take an object's attributes into the append session, or refuse the reopen.
///
/// Append mode rebuilds every object header it touches out of the attributes
/// read from it, so what this returns is what the object will still have when
/// the session finalizes. An attribute set that could not be read whole —
/// `ObjectAttributes::into_complete` refuses it — would come back as the part
/// that did read, silently deleting the rest.
///
/// Left to surface at `finalize`, that failure would land after this session's
/// chunk data and indices had already been written past the allocation point
/// the superblock still records, leaving a file libhdf5 reads as truncated.
/// Refusing the open leaves it untouched.
///
/// Size is no longer a reason to refuse: an attribute too large for a header
/// message goes back out through dense storage, the form libhdf5 read it from.
///
/// The set comes back in creation-index order, which is the order the registry
/// holds attributes in for an object made in this session too. A dense set is
/// read through the name index, so the order it arrives in is the order a hash
/// walk took; sorting here is what makes "the list is in creation order" true
/// of a reopened object as well, without any later stage having to know which
/// storage form the attributes came out of. Attributes of an untracked object
/// carry no index and keep the order they were read in.
fn take_reopened_attributes(
    attrs: crate::io::reader::ObjectAttributes,
    owner: &str,
) -> IoResult<Vec<AttributeEntry>> {
    let mut attrs = attrs.into_complete(owner)?;
    attrs.sort_by_key(|a| a.creation_index());
    Ok(attrs)
}

/// The creation-order policy an on-disk object header declares — the single
/// owner of the recovery rule, used for the root group, every reopened group
/// and (through its attribute half) every reopened dataset.
///
/// The two halves come from two different places, and reading one for both is
/// how a file that sets only one of them came back with both or neither:
///
///   * links — the `Link Info` message's flag bits, which is what
///     `H5Pget_link_creation_order` reads (`H5G__get_create_plist`). A group
///     with no such message (or one this crate cannot decode) tracks nothing;
///     so does every dataset, which has no links to order.
///   * attributes — the object header's own flag bits, which is what
///     `H5Pget_attr_creation_order` reads (`H5Pocpl.c`). The `Attribute Info`
///     message carries the same two bits, but the header is the authority
///     libhdf5 consults, and it is present even when the object has no
///     attributes yet.
fn recover_track_order(
    header: &crate::format::object_header::ObjectHeader,
    ctx: &FormatContext,
) -> TrackOrder {
    let links = header
        .messages
        .iter()
        .find(|m| m.msg_type == crate::format::messages::MSG_LINK_INFO)
        .and_then(|m| LinkInfoMessage::decode(&m.data, ctx).ok())
        .map(|(info, _)| info.creation_order())
        .unwrap_or_default();
    TrackOrder {
        links,
        attrs: header.attribute_creation_order(),
    }
}

/// The times a header being (re)written carries, given what the object had.
///
/// Every object header this writer emits is one it is writing *now*, which is
/// what `H5O_touch_oh` is called for: an object that stores times gets its
/// access and change time moved to now, and one that does not store them stays
/// that way — the flag belongs to the object's creation property list, and a
/// rewrite is not a creation.
fn touched_times(times: Option<ObjectTimes>) -> Option<ObjectTimes> {
    times.map(|t| t.touched(now_seconds()))
}

/// Seconds since the epoch, as an object header stores them (`H5_now`).
///
/// Saturates rather than wrapping: the field is a 32-bit count, and a clock
/// past 2106 is better reported as the largest time the format can express
/// than as a time in 1970. A clock before the epoch yields 0, which is what
/// libhdf5 writes for "no time recorded".
fn now_seconds() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX))
}

/// The dense storage an on-disk object header names: the fractal heap and the
/// indices its `Attribute Info` and `Link Info` messages point at.
///
/// A rewrite of that header lays fresh storage out and stops naming this, so
/// what this returns is exactly what the rewrite supersedes and must free.
/// Compact storage names no heap and yields `None` — there is nothing to free
/// and nothing that could be freed twice.
fn superseded_dense(
    header: &crate::format::object_header::ObjectHeader,
    ctx: &FormatContext,
) -> (Option<AttributeInfoMessage>, Option<LinkInfoMessage>) {
    let decode = |msg_type: u8| {
        header
            .messages
            .iter()
            .find(|m| m.msg_type == msg_type)
            .map(|m| m.data.as_slice())
    };
    let attrs = decode(crate::format::messages::MSG_ATTR_INFO)
        .and_then(|d| AttributeInfoMessage::decode(d, ctx).ok())
        .map(|(info, _)| info)
        .filter(|info| info.is_dense());
    let links = decode(crate::format::messages::MSG_LINK_INFO)
        .and_then(|d| LinkInfoMessage::decode(d, ctx).ok())
        .map(|(info, _)| info)
        .filter(|info| info.is_dense());
    (attrs, links)
}

/// One collection block with free space that a later vlen insert may
/// fill — an entry in the writer's CWFS list (libhdf5 `f->shared->cwfs`).
struct CwfsEntry {
    /// Block address of the collection.
    addr: u64,
    /// Declared block size; never changes after allocation.
    size: usize,
    /// Bytes its free-space marker owns, per
    /// [`GlobalHeapCollection::free_space_at`](crate::format::global_heap::GlobalHeapCollection::free_space_at).
    free: usize,
}

/// Maximum CWFS entries tracked — libhdf5's `H5HG_NCWFS` (H5HGpkg.h).
const H5HG_NCWFS: usize = 16;

/// Record a collection with `free` bytes in the CWFS list: update its
/// entry if present, append while the list is short, and otherwise
/// replace the entry with the least free space when this one has more —
/// the retention rule of libhdf5's `H5HG_insert`.
fn cwfs_note(cwfs: &mut Vec<CwfsEntry>, addr: u64, size: usize, free: usize) {
    if let Some(p) = cwfs.iter().position(|e| e.addr == addr) {
        cwfs[p].free = free;
        return;
    }
    if cwfs.len() < H5HG_NCWFS {
        cwfs.insert(0, CwfsEntry { addr, size, free });
        return;
    }
    if let Some(p) = (0..cwfs.len()).min_by_key(|&p| cwfs[p].free) {
        if free > cwfs[p].free {
            cwfs[p] = CwfsEntry { addr, size, free };
        }
    }
}

/// The uniform rejection for `delete_dataset` / `delete_group` while SWMR
/// streaming is active: deleting frees the object's blocks, and a live
/// reader may hold any of their addresses.
fn swmr_delete_error(name: &str) -> crate::io::IoError {
    crate::io::IoError::InvalidState(format!(
        "cannot delete '{name}' during SWMR streaming: a reader may hold the \
         object's header and storage addresses (libhdf5 forbids link deletion \
         during SWMR writes too)"
    ))
}

/// Whether the chunk at grid `coords` lies entirely at or beyond `extent` in
/// some dimension — no element of it would survive a shrink to that extent.
fn chunk_outside_extent(coords: &[u64], chunk_dims: &[u64], extent: &[u64]) -> bool {
    coords
        .iter()
        .zip(chunk_dims)
        .zip(extent)
        .any(|((&c, &cd), &e)| c.saturating_mul(cd) >= e)
}

/// Whether the chunk at grid `coords` keeps elements under `extent` but
/// extends past it in some dimension — a shrink must refill its
/// out-of-extent region with the fill value.
fn chunk_straddles_extent(coords: &[u64], chunk_dims: &[u64], extent: &[u64]) -> bool {
    !chunk_outside_extent(coords, chunk_dims, extent)
        && coords
            .iter()
            .zip(chunk_dims)
            .zip(extent)
            .any(|((&c, &cd), &e)| (c + 1).saturating_mul(cd) > e)
}

/// Overwrite, in `data` (one whole chunk, unfiltered, row-major), every
/// element at or beyond `extent` with the matching bytes of `fill` — a
/// same-sized buffer tiled with the fill value. The caller guarantees the
/// chunk at `coords` straddles `extent`, so every dimension keeps at least
/// one element. Returns the replaced bytes, so a vlen dataset's dead
/// heap references can be released rather than stranded.
fn refill_chunk_beyond_extent(
    data: &mut [u8],
    fill: &[u8],
    coords: &[u64],
    chunk_dims: &[u64],
    extent: &[u64],
    element_size: usize,
) -> Vec<u8> {
    let ndims = chunk_dims.len();
    let keep: Vec<usize> = (0..ndims)
        .map(|d| {
            let origin = coords[d] * chunk_dims[d];
            chunk_dims[d].min(extent[d].saturating_sub(origin)) as usize
        })
        .collect();
    // Row-major walk: for every row (all dimensions but the last),
    // overwrite the whole row when its prefix is outside the keep box,
    // else only the row's out-of-extent tail.
    let row_elems = chunk_dims[ndims - 1] as usize;
    let keep_last = keep[ndims - 1];
    let nrows: u64 = chunk_dims[..ndims - 1].iter().product();
    let mut replaced = Vec::new();
    for r in 0..nrows {
        let mut rem = r;
        let mut in_keep = true;
        for d in (0..ndims - 1).rev() {
            let c = rem % chunk_dims[d];
            rem /= chunk_dims[d];
            if c as usize >= keep[d] {
                in_keep = false;
            }
        }
        let start = if in_keep { keep_last } else { 0 };
        if start == row_elems {
            continue;
        }
        let a = (r as usize * row_elems + start) * element_size;
        let b = (r as usize + 1) * row_elems * element_size;
        replaced.extend_from_slice(&data[a..b]);
        data[a..b].copy_from_slice(&fill[a..b]);
    }
    replaced
}

/// Validate caller-supplied chunk geometry at dataset definition, the rule
/// libhdf5 applies in `H5D__chunk_construct` (H5Dchunk.c): the chunk rank
/// must match the dataspace rank, no chunk dimension may be zero, and a
/// chunk dimension may not exceed a fixed maximum dimension — except in a
/// dimension whose current size is zero, which libhdf5 exempts.
fn validate_chunk_geometry(dims: &[u64], max_dims: &[u64], chunk_dims: &[u64]) -> IoResult<()> {
    let ndims = dims.len();
    if chunk_dims.len() != ndims {
        return Err(crate::io::IoError::InvalidState(format!(
            "chunk shape has {} dimensions but the dataspace has {}",
            chunk_dims.len(),
            ndims
        )));
    }
    if max_dims.len() != ndims {
        return Err(crate::io::IoError::InvalidState(format!(
            "maximum shape has {} dimensions but the dataspace has {}",
            max_dims.len(),
            ndims
        )));
    }
    for d in 0..ndims {
        if chunk_dims[d] == 0 {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk dimension {d} is zero"
            )));
        }
        if dims[d] != 0 && max_dims[d] != u64::MAX && max_dims[d] < chunk_dims[d] {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk dimension {} is {} but the maximum dimension size is {}",
                d, chunk_dims[d], max_dims[d]
            )));
        }
    }
    Ok(())
}

/// An extensible-array index requires at most one unlimited dimension —
/// `H5D__chunk_construct` (H5Dchunk.c) only selects this index for exactly
/// one — at any position: `chunk_grid::linear_index` seeds the unlimited
/// dimension into the slot no down-chunks multiplier touches, the same
/// address libhdf5 reaches by swizzling it to the slowest position
/// (`H5VM_swizzle_coords`, H5Dearray.c). Two or more unlimited dimensions
/// have no finite grid at all; that shape needs a v2 B-tree index instead.
fn ensure_at_most_one_unlimited(max_dims: &[u64]) -> IoResult<()> {
    let unlimited: Vec<usize> = max_dims
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m == u64::MAX)
        .map(|(d, _)| d)
        .collect();
    if unlimited.len() > 1 {
        return Err(crate::io::IoError::InvalidState(format!(
            "an extensible-array index supports at most one unlimited dimension, \
             but dimensions {unlimited:?} are all unlimited; a v2 B-tree index \
             handles two or more"
        )));
    }
    Ok(())
}

/// Reject strings the dataset's declared character set cannot label.
///
/// A Rust `&str` is always UTF-8, so only an ASCII declaration (charset 0)
/// can be violated. libhdf5 stores the bytes unvalidated — its vlen write
/// path has no cset check anywhere — which mislabels them for every reader
/// that trusts the declaration (h5py raises on the same mismatch).
fn ensure_vlen_charset(charset: u8, strings: &[&str]) -> IoResult<()> {
    if charset == 0 {
        if let Some((i, s)) = strings.iter().enumerate().find(|(_, s)| !s.is_ascii()) {
            return Err(crate::io::IoError::InvalidState(format!(
                "string {i} ({s:?}) is not ASCII, but the dataset's character set is"
            )));
        }
    }
    Ok(())
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
    /// Pool of node-size blocks (the index's
    /// [`node_size`](Bt2ChunkIndex::node_size) bytes each) holding the tree's
    /// nodes, in the order [`Bt2Tree::encode`] emits them.
    ///
    /// The single owner of the tree's node addresses: a flush re-serializes the
    /// whole tree over these blocks and allocates only the shortfall, so no
    /// flush can orphan a block it replaced. Every node is the same size, so a
    /// block stays usable however the tree reshapes.
    ///
    /// The pool holds exactly one block per node after every flush, in both
    /// directions: a taller tree allocates the shortfall, a smaller one frees
    /// the surplus. Nothing here depends on the record count only ever rising,
    /// so a record-removal path can be added to [`Bt2ChunkIndex`] without the
    /// blocks it drops going unreachable.
    pub node_addrs: Vec<u64>,
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
    /// File offset of the on-disk header a reopen found for this group, so
    /// finalize can free the block it supersedes.
    pub obj_header_written_addr: Option<u64>,
    /// Encoded size of that on-disk header (first block).
    pub obj_header_encoded_size: usize,
    /// Soft-deleted: excluded from finalize output.
    pub deleted: bool,
    /// Attributes attached to this group (e.g. NeXus `NX_class`).
    pub attributes: Vec<AttributeEntry>,
    /// When the link naming this group was created, on the writer's single
    /// monotonic sequence. Groups, datasets and hard links share it, so a
    /// parent can order its links the way they were actually made.
    pub creation_seq: u64,
    /// How this group records creation order for its links and, separately,
    /// for its attributes. Creation-order tracking is a property of the
    /// object's creation property list in libhdf5, so it is captured here
    /// when the group is created rather than read from the writer at
    /// finalize: a later change of policy must not rewrite an object already
    /// made.
    pub track_order: TrackOrder,
    /// This group's stored times, on the same terms as
    /// [`DatasetInfo::times`].
    pub times: Option<ObjectTimes>,
}

/// One object's creation-order policy, with the two subsystems libhdf5 keeps
/// apart kept apart here too.
///
/// `H5Pset_link_creation_order` and `H5Pset_attr_creation_order` are separate
/// calls reading back out of separate places on disk — the Link Info message
/// and the object header's own flag bits — and a file may set either alone.
/// Carrying them as one flag made a reopen give a one-of-two file both or
/// neither.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackOrder {
    /// Creation order of the links this group holds. Meaningless for a
    /// dataset, which is why `DatasetInfo` keeps only the attribute half.
    pub links: CreationOrder,
    /// Creation order of the attributes attached to this object.
    pub attrs: CreationOrder,
}

impl TrackOrder {
    /// The policy the crate's single `track_order` knob selects: both
    /// subsystems tracked *and* indexed, or neither — the pair h5py's
    /// `File(track_order=True)` writes.
    pub fn uniform(track: bool) -> Self {
        let order = if track {
            CreationOrder::Indexed
        } else {
            CreationOrder::Untracked
        };
        Self {
            links: order,
            attrs: order,
        }
    }
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
#[derive(Clone)]
pub struct HardLink {
    /// Parent group index (`None` = the root group).
    pub parent: Option<usize>,
    /// Leaf name of the link within the parent group.
    pub name: String,
    /// Object this link resolves to.
    pub target: HardLinkTarget,
    /// When this link was created; see [`GroupInfo::creation_seq`].
    pub creation_seq: u64,
}

/// A link a reopened file already held that this writer cannot express.
///
/// Soft, external and user-defined links have no creation, retarget or delete
/// operation here — only hard links do — so a header rewrite that emits what
/// the registry models would erase them. Their encoded `Link` message rides
/// along instead and is written back byte for byte, which preserves every
/// field (name character set, creation order, the link value) without this
/// writer having to model any of them.
///
/// A *hard* link is preserved the same way when the object it names is one
/// the reopen could not model: writing the link back unchanged leaves that
/// object's header exactly where it is, which is the only way the rewrite can
/// keep what it cannot rebuild.
#[derive(Clone)]
pub struct PreservedLink {
    /// Parent group index (`None` = the root group).
    pub parent: Option<usize>,
    /// Leaf name of the link within the parent group.
    pub name: String,
    /// The link's class, decoded once at collection so listings can report
    /// it. Never the source of what gets written — `encoded` is.
    pub class: crate::io::reader::LinkClass,
    /// The encoded `Link` message body, exactly as read from the file.
    pub encoded: Vec<u8>,
    /// Why the object this link names could not be modelled, for the callers
    /// that ask for it by name. `None` when the link's own class — not its
    /// target — is what this writer cannot express.
    pub reason: Option<String>,
}

/// Every link a reopen walk met, split by what the writer can do with it.
/// A header rewrite emits both halves, so a link in neither half is a link
/// the close would destroy.
#[derive(Default)]
struct CollectedLinks {
    /// Hard links whose target the reopen modelled, with the plan that says
    /// how to rebuild it.
    hard: Vec<(HardEntry, CollectedObject)>,
    /// Links written back unchanged: the class this writer cannot express,
    /// and the hard links whose object it cannot model.
    preserved: Vec<PreservedEntry>,
}

/// One hard link the reopen walk met: what it names, and the exact message
/// that names it.
#[derive(Clone)]
struct HardEntry {
    /// Full link path, in the no-leading-`/` form the registry uses.
    path: String,
    /// Object header address the link names.
    address: u64,
    /// The encoded `Link` message body, exactly as read from the file.
    encoded: Vec<u8>,
}

/// A link the rewrite writes back exactly as it read it.
struct PreservedEntry {
    path: String,
    class: crate::io::reader::LinkClass,
    encoded: Vec<u8>,
    /// Why the object it names could not be modelled; `None` when the link's
    /// own class is what this writer cannot express.
    reason: Option<String>,
}

/// What a reopen can do with one object it reached.
///
/// A header rewrite emits a modelled object out of the registry, so the
/// registry may hold an object only when *every* message the model consumes
/// decoded. A partial read is not a smaller object, it is a different one:
/// before this rule a dataset whose datatype message did not decode was
/// registered as a group, and the close rewrote its header as one.
enum ObjectPlan {
    /// A dataset the rewrite can rebuild.
    Dataset(Box<DatasetParts>),
    /// A group the rewrite can rebuild, and the links it holds.
    Group(GroupParts),
    /// An object this writer cannot model, and why. Its header is never
    /// rewritten and never freed; the link naming it is written back byte for
    /// byte, so the object stays exactly as the file already had it — what
    /// libhdf5 does with the parts of a file it does not understand.
    Preserve(String),
}

/// The messages a dataset's rewrite is built from, all decoded.
struct DatasetParts {
    /// Chunk-0 size: the block the rewrite supersedes and frees.
    header_size: usize,
    datatype: DatatypeMessage,
    dataspace: crate::format::messages::dataspace::DataspaceMessage,
    layout: crate::format::messages::data_layout::DataLayoutMessage,
    filter_pipeline: Option<FilterPipeline>,
    fill_value: Option<Vec<u8>>,
    attributes: Vec<AttributeEntry>,
    /// The creation-order policy the on-disk header declares; a rewrite that
    /// read it from the writer instead would stamp this session's policy onto
    /// an object libhdf5 created under another.
    track_order: TrackOrder,
    /// The times the on-disk header stores, for the same reason: whether an
    /// object tracks them is settled when it is created, not when it is
    /// rewritten.
    times: Option<ObjectTimes>,
    /// The dense storage the rewrite supersedes and must free.
    dense: DenseCarry,
}

/// The same for a group, plus the links it holds — decoded once, with the
/// bytes they came from, so the walk and the rewrite agree on its contents.
struct GroupParts {
    header_size: usize,
    attributes: Vec<AttributeEntry>,
    links: Vec<(crate::format::messages::link::LinkMessage, Vec<u8>)>,
    track_order: TrackOrder,
    times: Option<ObjectTimes>,
    dense: DenseCarry,
    /// The symbol-table storage a classic group's header names — the blocks
    /// the rewrite supersedes. `None` for a link-message group, which has
    /// none. Its links are already in `links`: the walk turns each symbol
    /// table entry into the link message it stands for, so nothing downstream
    /// has to know which of the two forms the group was in.
    stab: Option<StabExtents>,
}

/// The dense storage one reopened object's header names, which the rewrite of
/// that header stops naming and therefore has to free. Both halves are read
/// back before this is built — a heap that could not be read makes the object
/// [`ObjectPlan::Preserve`], so nothing here describes storage whose contents
/// were lost.
#[derive(Default)]
struct DenseCarry {
    attrs: Option<AttributeInfoMessage>,
    links: Option<LinkInfoMessage>,
}

/// The raw-data storage a creator is about to give a dataset.
///
/// Passed into [`Hdf5Writer::begin_create`] rather than checked inside each
/// creator, so that a creator added later has to say which it is and cannot
/// silently skip the format check that goes with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NewStorage {
    /// One contiguous run of bytes, or none at all (a null dataspace).
    Contiguous,
    /// Chunks reached through an index.
    Chunked,
}

/// A modelled object, as the walk hands it to the registry rebuild. A group's
/// links are not here: the walk followed them, and each child is an entry of
/// its own.
enum CollectedObject {
    Dataset(Box<DatasetParts>),
    Group {
        header_size: usize,
        attributes: Vec<AttributeEntry>,
        track_order: TrackOrder,
        times: Option<ObjectTimes>,
        dense: DenseCarry,
        stab: Option<StabExtents>,
    },
}

/// The reopen's discovery pass: one walk that classifies every object it
/// reaches and descends into the groups among them.
///
/// Every object the close will touch is decided here and nowhere else, so
/// "modelled or preserved" is a property of the walk rather than of whatever
/// each later stage happened to be able to decode.
struct ReopenWalk<'a> {
    handle: &'a mut FileHandle,
    meta: &'a crate::io::FileMeta,
    /// End of the file, bounding each object-header read.
    file_size: u64,
    out: CollectedLinks,
    /// Object headers already descended into, so hard-link cycles end.
    visited: std::collections::HashSet<u64>,
}

impl<'a> ReopenWalk<'a> {
    fn new(handle: &'a mut FileHandle, meta: &'a crate::io::FileMeta, file_size: u64) -> Self {
        Self {
            handle,
            meta,
            file_size,
            out: CollectedLinks::default(),
            visited: std::collections::HashSet::new(),
        }
    }

    /// Everything the walk found.
    fn finish(self) -> CollectedLinks {
        self.out
    }

    /// Decide what the reopen can do with the object at `addr`.
    ///
    /// The single gate: every object the rewrite touches is classified here,
    /// and an object is modelled only when each message the model consumes
    /// decoded. See [`ObjectPlan`] for why anything else must keep its bytes.
    fn plan(&mut self, addr: u64) -> IoResult<ObjectPlan> {
        let (handle, meta, file_size) = (&mut *self.handle, self.meta, self.file_size);
        let ctx = &meta.ctx;
        use crate::format::messages::data_layout::DataLayoutMessage;
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::link::LinkMessage;
        use crate::format::messages::link_info::LinkInfoMessage;
        use crate::format::messages::shared::MSG_FLAG_SHARED;
        use crate::format::messages::{
            MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE, MSG_DATA_LAYOUT, MSG_FILL_VALUE,
            MSG_FILTER_PIPELINE, MSG_LINK, MSG_LINK_INFO, MSG_SYMBOL_TABLE,
        };

        // Chunk 0 only, for its encoded size: that is the block the rewrite
        // supersedes and frees.
        let buf = handle.read_at_most(addr, file_size.saturating_sub(addr) as usize)?;
        let header_size = match crate::format::object_header::ObjectHeader::decode_any(&buf) {
            Ok((_, size)) => size,
            Err(e) => {
                return Ok(ObjectPlan::Preserve(format!(
                    "its object header does not decode: {e}"
                )))
            }
        };
        // The messages, on the other hand, must come from the whole chain: a
        // filter pipeline or an attribute that spilled into a continuation is
        // one the rewrite would otherwise drop.
        let header = match crate::io::object_header_io::read_object_header_full(handle, meta, addr)
        {
            Ok(h) => h,
            Err(e) => {
                return Ok(ObjectPlan::Preserve(format!(
                    "its object header chain does not read: {e}"
                )))
            }
        };

        // The policy, the times and the storage the header declares, read once
        // from the whole chain: all three are properties of the object, not of
        // any one message the loop below happens to reach.
        let track_order = recover_track_order(&header, ctx);
        let times = header.times;
        let (dense_attrs, dense_links) = superseded_dense(&header, ctx);

        // Attributes come from the reader's collector rather than from the
        // loop below, so compact, dense and shared attributes all reach the
        // rewrite by the one path that knows how to read each of them. An
        // object whose set did not read whole is preserved: a short set here
        // would be a rewrite deleting the attributes it could not read.
        let attributes = match take_reopened_attributes(
            crate::io::reader::collect_object_attributes(handle, ctx, &header),
            &format!("the object at {addr:#x}"),
        ) {
            Ok(a) => a,
            Err(e) => {
                return Ok(ObjectPlan::Preserve(format!(
                    "its attributes do not read back whole: {e}"
                )))
            }
        };

        let mut datatype = None;
        let mut dataspace = None;
        let mut layout = None;
        let mut filter_pipeline = None;
        let mut fill_value = None;
        let mut links = Vec::new();
        let mut stab = None;
        // A datatype, dataspace or layout message says the object is not a
        // group, whether or not the three a dataset needs are all there.
        let mut dataset_shaped = false;

        for msg in &header.messages {
            let consumed = matches!(
                msg.msg_type,
                MSG_DATATYPE
                    | MSG_DATASPACE
                    | MSG_DATA_LAYOUT
                    | MSG_FILTER_PIPELINE
                    | MSG_FILL_VALUE
                    | MSG_ATTRIBUTE
                    | MSG_LINK
                    | MSG_LINK_INFO
                    | MSG_SYMBOL_TABLE
            );
            // A shared message holds a reference to where its body lives, not
            // the body. Decoding those bytes as one does not fail loudly — the
            // reference's version byte reads as a version and a class of its
            // own — so the guard is the only thing between a shared datatype
            // and a rewrite that invents a type for it.
            if consumed && msg.flags & MSG_FLAG_SHARED != 0 {
                return Ok(ObjectPlan::Preserve(format!(
                    "its message of type {:#04x} is a shared-message reference, which this \
                     writer does not resolve",
                    msg.msg_type
                )));
            }
            macro_rules! consume {
                ($decode:expr, $what:literal) => {
                    match $decode {
                        Ok(v) => v,
                        Err(e) => {
                            return Ok(ObjectPlan::Preserve(format!(
                                "its {} message does not decode: {e}",
                                $what
                            )))
                        }
                    }
                };
            }
            match msg.msg_type {
                MSG_DATATYPE => {
                    dataset_shaped = true;
                    let (dt, _) = consume!(DatatypeMessage::decode(&msg.data, ctx), "datatype");
                    datatype = Some(dt);
                }
                MSG_DATASPACE => {
                    dataset_shaped = true;
                    let (ds, _) = consume!(DataspaceMessage::decode(&msg.data, ctx), "dataspace");
                    dataspace = Some(ds);
                }
                MSG_DATA_LAYOUT => {
                    dataset_shaped = true;
                    let (dl, _) =
                        consume!(DataLayoutMessage::decode(&msg.data, ctx), "data layout");
                    layout = Some(dl);
                }
                MSG_FILTER_PIPELINE => {
                    let (p, _) = consume!(FilterPipeline::decode(&msg.data), "filter pipeline");
                    if !p.filters.is_empty() {
                        filter_pipeline = Some(p);
                    }
                }
                MSG_FILL_VALUE => {
                    let (fv, _) = consume!(FillValueMessage::decode(&msg.data), "fill value");
                    if fv.fill_defined == 2 {
                        fill_value = fv.fill_value;
                    }
                }
                MSG_LINK => {
                    let (l, _) = consume!(LinkMessage::decode(&msg.data, ctx), "link");
                    links.push((l, msg.data.clone()));
                }
                MSG_LINK_INFO => {
                    let (li, _) = consume!(LinkInfoMessage::decode(&msg.data, ctx), "link info");
                    // Once a group holds enough links libhdf5 moves them into
                    // the fractal heap this message names and writes no `Link`
                    // messages at all. Reading them back is what makes the
                    // rewrite emit the group with its children; a rewrite from
                    // the header messages alone emitted it empty, orphaning
                    // every object below it.
                    if li.fractal_heap_address != UNDEF_ADDR {
                        let dense = match crate::io::reader::Hdf5Reader::read_dense_links(
                            handle,
                            ctx,
                            li.fractal_heap_address,
                        ) {
                            Ok(l) => l,
                            Err(e) => {
                                return Ok(ObjectPlan::Preserve(format!(
                                    "its dense link storage does not read: {e}"
                                )))
                            }
                        };
                        // Re-encoded rather than carried as bytes: a heap
                        // object is not a header message, so there are no
                        // message bytes to carry. The encoding round-trips
                        // through the same decoder that just read it.
                        links.extend(dense.into_iter().map(|l| {
                            let bytes = l.encode(ctx);
                            (l, bytes)
                        }));
                    }
                }
                MSG_SYMBOL_TABLE => {
                    // A classic group keeps no link message at all: its links
                    // are symbol table entries in the B-tree this message
                    // names. Turning each into the link message it stands for
                    // is what lets the rest of the reopen — the walk, the
                    // registry, the preserve path — work on one link model
                    // whichever form the group is in.
                    let Some(s) = Stab::decode(&msg.data, ctx) else {
                        return Ok(ObjectPlan::Preserve(
                            "its symbol table message is shorter than the two addresses it \
                             must carry"
                                .into(),
                        ));
                    };
                    let contents = match crate::io::symbol_table_io::read_stab(handle, meta, s) {
                        Ok(c) => c,
                        Err(e) => {
                            return Ok(ObjectPlan::Preserve(format!(
                                "its symbol table does not read: {e}"
                            )))
                        }
                    };
                    stab = Some(contents.extents);
                    links.extend(contents.links.into_iter().map(|l| {
                        let msg = match l.target {
                            StabTarget::Hard { addr, .. } => LinkMessage::hard(&l.name, addr),
                            StabTarget::Soft { value } => LinkMessage::soft(&l.name, &value),
                        };
                        let bytes = msg.encode(ctx);
                        (msg, bytes)
                    }));
                }
                _ => {}
            }
        }

        match (datatype, dataspace, layout) {
            // A layout `rebuild_dataset` has no arm for leaves the registry
            // entry with an undefined data address, and the close then rewrites
            // the header as a contiguous, unallocated dataset — every element
            // gone, silently. Only the two layouts that rebuild are modelled;
            // the rest keep their bytes, as an undecodable message already
            // does. Version-3 chunked (a version-1 B-tree index, which h5py
            // writes at `libver='v108'` and in every classic file), compact and
            // virtual layouts are all this.
            (Some(_), Some(_), Some(layout))
                if !matches!(
                    layout,
                    DataLayoutMessage::Contiguous { .. } | DataLayoutMessage::ChunkedV4 { .. }
                ) =>
            {
                Ok(ObjectPlan::Preserve(format!(
                    "its data layout is {}, which this writer reads but does not build",
                    layout.describe()
                )))
            }
            (Some(datatype), Some(dataspace), Some(layout)) => {
                Ok(ObjectPlan::Dataset(Box::new(DatasetParts {
                    header_size,
                    datatype,
                    dataspace,
                    layout,
                    filter_pipeline,
                    fill_value,
                    attributes,
                    track_order,
                    times,
                    dense: DenseCarry {
                        attrs: dense_attrs,
                        links: dense_links,
                    },
                })))
            }
            // A committed (named) datatype has a datatype message and neither
            // of the other two; so does a dataset whose header this crate only
            // half understands. Neither is a group, and modelling either as
            // one is what rewrote them into empty groups.
            _ if dataset_shaped => Ok(ObjectPlan::Preserve(
                "it carries a datatype, dataspace or layout message but not the three a \
                 dataset is built from; this writer models only groups and datasets"
                    .into(),
            )),
            _ => Ok(ObjectPlan::Group(GroupParts {
                header_size,
                attributes,
                links,
                track_order,
                times,
                dense: DenseCarry {
                    attrs: dense_attrs,
                    links: dense_links,
                },
                stab,
            })),
        }
    }

    /// Walk `links` (one group's, already decoded), classifying every object
    /// they name and descending into the groups among them.
    fn group(
        &mut self,
        links: &[(crate::format::messages::link::LinkMessage, Vec<u8>)],
        prefix: &str,
        depth: usize,
    ) -> IoResult<()> {
        // Bound nesting depth so a pathologically deep group chain cannot
        // overflow the stack (the `visited` set bounds total work but not
        // recursion depth).
        if depth > 256 {
            return Ok(());
        }
        use crate::format::messages::link::LinkTarget;
        for (link, encoded) in links {
            let full_name = if prefix.is_empty() {
                link.name.clone()
            } else {
                format!("{}/{}", prefix, link.name)
            };

            // Only a hard link names an object this writer can rebuild. Every
            // other class is kept by its bytes, because a close that emitted
            // only what the registry models would drop it from the file.
            let LinkTarget::Hard { address } = &link.target else {
                self.out.preserved.push(PreservedEntry {
                    path: full_name,
                    class: crate::io::reader::LinkClass::from_target(&link.target),
                    encoded: encoded.clone(),
                    reason: None,
                });
                continue;
            };
            let entry = HardEntry {
                path: full_name.clone(),
                address: *address,
                encoded: encoded.clone(),
            };

            match self.plan(*address)? {
                // Kept by its bytes, exactly as a link class this writer
                // cannot express is: writing the link back unchanged is what
                // leaves the object's header where the file already has it.
                ObjectPlan::Preserve(reason) => self.out.preserved.push(PreservedEntry {
                    path: full_name,
                    class: crate::io::reader::LinkClass::Hard,
                    encoded: entry.encoded,
                    reason: Some(reason),
                }),
                ObjectPlan::Dataset(parts) => {
                    self.out.hard.push((entry, CollectedObject::Dataset(parts)));
                }
                ObjectPlan::Group(parts) => {
                    self.out.hard.push((
                        entry,
                        CollectedObject::Group {
                            header_size: parts.header_size,
                            attributes: parts.attributes,
                            track_order: parts.track_order,
                            times: parts.times,
                            dense: parts.dense,
                            stab: parts.stab,
                        },
                    ));
                    // Recurse only into a group's header we have not entered
                    // before — breaks hard-link cycles.
                    if self.visited.insert(*address) {
                        self.group(&parts.links, &full_name, depth + 1)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Rebuild one reopened dataset's in-memory registry entry, storage and
/// all, from the header messages the walk decoded.
///
/// Fails when the chunk index the file names does not read back. The
/// caller answers that by preserving the object rather than registering
/// a dataset whose index has forgotten where its chunks are: the close
/// rewrites what the registry holds, so an index rebuilt from the part of
/// it that decoded would strand every chunk it could not read.
fn rebuild_dataset(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    name: String,
    obj_addr: u64,
    parts: DatasetParts,
) -> IoResult<DatasetInfo> {
    let DatasetParts {
        header_size: ds_header_size,
        datatype: dt,
        dataspace: ds,
        layout: dl,
        filter_pipeline: fp,
        fill_value,
        attributes: attrs,
        track_order,
        times,
        dense: _,
    } = parts;

    let mut info = DatasetInfo {
        name,
        datatype: dt,
        dataspace: ds,
        obj_header_addr: obj_addr,
        data_addr: UNDEF_ADDR,
        data_size: 0,
        chunked: None,
        fixed_array: None,
        btree_v2: None,
        append: None,
        attributes: attrs,
        obj_header_written_addr: Some(obj_addr),
        obj_header_encoded_size: ds_header_size,
        filter_pipeline: fp,
        deleted: false,
        extent_dirty: false,
        header_dirty: false,
        // Stamped by the caller once the whole link graph is registered: it
        // is the count of links reaching this object, which one dataset's
        // parts cannot see.
        nlink_written: 1,
        // Stamped by the caller, which knows the order the walk met each
        // object; the rebuild sees one dataset at a time.
        creation_seq: 0,
        track_attr_order: track_order.attrs,
        fill_value,
        // Preserve the on-disk layout version so finalize re-encodes
        // what it read: a v5 file reopened and appended to must not be
        // silently downgraded to v4 (the filtered indexes keep their
        // 8-byte size fields, which v4 readers would mis-derive).
        layout_version: match &dl {
            DataLayoutMessage::ChunkedV4 { version, .. } => *version,
            _ => 4,
        },
        times,
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

            if *index_type == crate::format::messages::data_layout::ChunkIndexType::ExtensibleArray
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
                    let ea_header = ExtensibleArrayHeader::decode(&hdr_buf, ctx)?;

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
                                ctx,
                                ep.idx_blk_elmts as usize,
                                ndblk_addrs,
                                nsblk_addrs,
                                chunk_size_len,
                            )?
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
                                ctx,
                                ep.idx_blk_elmts as usize,
                                ndblk_addrs,
                                nsblk_addrs,
                            )?
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
            } else if *index_type
                == crate::format::messages::data_layout::ChunkIndexType::FixedArray
            {
                // Read the FA header and data block back so a
                // reopened dataset is writable and deletable, not
                // re-link only — a placeholder made a delete free
                // just the header and leak every chunk plus the
                // index. Paged data blocks (any FA with more than
                // dblk_page_nelmts chunks, libhdf5 default 1024)
                // reconstruct through the same decode owner; only
                // pages the bitmap marks initialized are decoded.
                let hdr_buf = handle.read_at_most(*index_address, 256)?;
                let fa_header = FixedArrayHeader::decode(&hdr_buf, ctx)?;
                let is_filtered = fa_header.client_id == FA_CLIENT_FILT_CHUNK;
                let chunk_size_len = if is_filtered {
                    (fa_header.element_size as usize)
                        .checked_sub(ctx.sizeof_addr as usize + 4)
                        .ok_or_else(|| {
                            crate::io::IoError::InvalidState(
                                "fixed array filtered element_size too small".into(),
                            )
                        })?
                } else {
                    0
                };
                if fa_header.data_blk_addr != UNDEF_ADDR && chunk_size_len <= 8 {
                    let dblk_size = fixed_array_dblk_disk_size(ctx, &fa_header) as usize;
                    let dblk_buf = handle.read_at_most(fa_header.data_blk_addr, dblk_size)?;
                    let fa_dblk =
                        decode_fixed_array_dblk(ctx, &fa_header, &dblk_buf, chunk_size_len)?;
                    info.fixed_array = Some(FixedArrayDatasetInfo {
                        chunk_dims: real_chunk_dims,
                        fa_header_addr: *index_address,
                        fa_dblk_addr: fa_header.data_blk_addr,
                        fa_header,
                        fa_dblk,
                        // Chunks written this session, matching the
                        // EA reconstruction above.
                        chunks_written: 0,
                    });
                }
            } else if *index_type == crate::format::messages::data_layout::ChunkIndexType::BTreeV2 {
                use crate::format::chunk_index::btree_v2::{
                    Bt2Geometry, Bt2Header, BT2_TYPE_CHUNK_FILT, BT2_TYPE_CHUNK_UNFILT,
                };

                // Walk the tree back into the in-memory index and
                // adopt its node blocks as the flush pool. The pool
                // re-serializes at the header's node_size, whatever
                // it is — libhdf5 sizes every node from
                // hdr->node_size (H5B2leaf.c, H5B2internal.c) — so
                // a foreign size reopens too. Only a record type
                // that is not a chunk record, or a node size below
                // the bulk loader's few-records-per-node floor
                // (the same bound creation enforces), stays
                // re-link only.
                let hdr_buf = handle.read_at_most(*index_address, 256)?;
                let bt2_hdr = Bt2Header::decode(&hdr_buf, ctx)?;
                let ndims = real_chunk_dims.len();
                let is_filt = match bt2_hdr.record_type {
                    BT2_TYPE_CHUNK_UNFILT => Some(false),
                    BT2_TYPE_CHUNK_FILT => Some(true),
                    _ => None,
                };
                if let (Some(is_filt), true) = (
                    is_filt,
                    bt2_hdr.node_size as usize >= 10 + 3 * bt2_hdr.record_size as usize,
                ) {
                    let mut index = if is_filt {
                        let csl = (bt2_hdr.record_size as usize)
                            .checked_sub(ctx.sizeof_addr as usize + 4 + ndims * 8)
                            .filter(|&c| c <= 8)
                            .ok_or_else(|| {
                                crate::io::IoError::InvalidState(
                                    "v2 B-tree filtered record size does not fit \
                                     its rank and address width"
                                        .into(),
                                )
                            })?;
                        Bt2ChunkIndex::new_filtered(ndims, csl as u8)
                    } else {
                        Bt2ChunkIndex::new_unfiltered(ndims)
                    };
                    // Re-serialize with the creator's parameters:
                    // node blocks keep their size and the rewritten
                    // header keeps its declared split/merge.
                    index.node_size = bt2_hdr.node_size;
                    index.split_percent = bt2_hdr.split_percent;
                    index.merge_percent = bt2_hdr.merge_percent;
                    let mut node_addrs = Vec::new();
                    if bt2_hdr.root_node_addr != UNDEF_ADDR && bt2_hdr.total_num_records > 0 {
                        let geo = Bt2Geometry::new(
                            bt2_hdr.node_size,
                            bt2_hdr.record_size,
                            bt2_hdr.depth,
                            ctx.sizeof_addr,
                        );
                        let mut record_bytes = Vec::new();
                        collect_bt2_nodes(
                            handle,
                            ctx,
                            bt2_hdr.root_node_addr,
                            bt2_hdr.depth,
                            bt2_hdr.num_records_in_root,
                            bt2_hdr.record_size,
                            bt2_hdr.node_size,
                            &geo,
                            &mut record_bytes,
                            &mut node_addrs,
                        )?;
                        let total = if bt2_hdr.record_size > 0 {
                            record_bytes.len() / bt2_hdr.record_size as usize
                        } else {
                            0
                        };
                        if is_filt {
                            for r in Bt2ChunkIndex::decode_filtered_records(
                                &record_bytes,
                                total,
                                ndims,
                                bt2_hdr.record_size,
                                ctx,
                            )? {
                                index.insert_filtered(
                                    r.scaled_offsets,
                                    r.chunk_address,
                                    r.chunk_size,
                                    r.filter_mask,
                                );
                            }
                        } else {
                            for r in Bt2ChunkIndex::decode_unfiltered_records(
                                &record_bytes,
                                total,
                                ndims,
                                ctx,
                            )? {
                                index.insert(r.scaled_offsets, r.chunk_address);
                            }
                        }
                    }
                    let max_dims = info
                        .dataspace
                        .max_dims
                        .clone()
                        .unwrap_or_else(|| info.dataspace.dims.clone());
                    info.btree_v2 = Some(Bt2DatasetInfo {
                        chunk_dims: real_chunk_dims,
                        max_dims,
                        bt2_header_addr: *index_address,
                        node_addrs,
                        index,
                        chunks_written: 0,
                    });
                }
            }
        }
        _ => {}
    }

    Ok(info)
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

/// Everything a version-0/1 (symbol-table) file carries that a version-2/3 one
/// does not.
///
/// Its presence *is* the format switch — [`Hdf5Writer::object_format`] reads
/// nothing else — because the three differences travel together: libhdf5 at
/// default library bounds writes a version-0/1 superblock over version-1
/// object headers over symbol-table groups, and writes no other combination of
/// the three. A file this crate creates never has one.
struct LegacyFile {
    /// The superblock as it was read. The close re-emits it with only the end
    /// of file and the root symbol table entry recomputed: the "K" ranks in
    /// particular are recorded nowhere else, and every node width in the file
    /// is derived from them.
    superblock: SuperblockV0V1,
    /// The file-level parameters every node width in a symbol table is derived
    /// from — the address/length widths and the B-tree "K" ranks, after the
    /// superblock extension has had its say. Carried rather than re-derived
    /// because a rewrite that used the library defaults on a file that
    /// overrode them would write nodes of the wrong width.
    meta: FileMeta,
    /// The symbol-table storage each group's header already names, by the
    /// scope whose rewrite supersedes it.
    ///
    /// INVARIANT: every entry is freed exactly once, by
    /// [`Hdf5Writer::prepare_symbol_tables`], which removes it as it frees.
    superseded: Slot<HashMap<LinkScope, StabExtents>>,
    /// The storage that same pass laid out, read by the header builders.
    ///
    /// INVARIANT: an entry exists here only after every block of that group's
    /// heap and B-tree is on disk. `build_group_header` reads it and never
    /// builds — a header is sized and then written by two separate calls, so a
    /// build that allocated would allocate twice.
    written: Slot<HashMap<LinkScope, Stab>>,
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
    /// Dataset registry. The outer [`Slot`] guards the spine (push on create,
    /// index/clone on access) and is held only briefly; each [`DatasetRef`]
    /// carries one dataset's metadata behind its own lock. A writer clones
    /// the `DatasetRef` out (releasing this lock) before doing the long
    /// per-dataset work, so a create never blocks an in-flight write.
    pub(crate) datasets: Slot<Vec<DatasetRef>>,
    /// Group registry, same shape as [`Self::datasets`].
    pub(crate) groups: Slot<Vec<GroupRef>>,
    /// User-created hard links (additional names for existing objects),
    /// resolved and emitted during finalize.
    pub(crate) hard_links: Slot<Vec<HardLink>>,
    /// Links a reopened file held that this writer cannot express, carried
    /// through every header rewrite by their encoded bytes. Always empty for
    /// a freshly created file; see [`PreservedLink`].
    pub(crate) preserved_links: Slot<Vec<PreservedLink>>,
    /// Attributes attached to the root group (file-level attributes).
    pub(crate) root_attributes: Slot<Vec<crate::format::messages::attribute::AttributeEntry>>,
    /// Serializes object creation so name-uniqueness check and registry insert
    /// happen atomically.
    ///
    /// INVARIANT: no two emitted links share a full-path name. Under
    /// `threadsafe`, create methods run on the shared read guard, so without
    /// this gate two threads could both pass the duplicate-name check (which
    /// snapshots a registry and drops its lock) and both push, writing an
    /// invalid HDF5 file with two same-named links. A create holds this lock
    /// across its check *and* its push; the streaming write path never takes
    /// it, so writes to existing datasets stay fully concurrent. It is the
    /// outermost lock a create acquires (create_lock → spine → slot), and no
    /// write path takes it, so it cannot deadlock with the registry locks.
    pub(crate) create_lock: Slot<()>,
    /// The file's low `H5Pset_libver_bounds` bound: the oldest libhdf5 a
    /// file this writer creates must stay readable by. It is the one switch
    /// the version-bearing messages read — the datatype message version
    /// (`H5O_dtype_ver_bounds`) and, at `V200`, layout version 5 for filtered
    /// chunked datasets, whose chunk indexes encode stored chunk sizes in a
    /// fixed `sizeof_size` field so an expanding filter cannot overflow it.
    /// `Earliest` by default — version-5 files are rejected by libhdf5 before
    /// 2.0, including the 1.14-based h5py wheels.
    libver: LibverBound,
    closed: bool,
    /// Set once `finalize_for_swmr` has published a readable file.
    ///
    /// A SWMR reader may hold a chunk index that still points at a block this
    /// writer has since replaced, so from that point on a relocated chunk's
    /// old block is kept rather than released for reuse — the same rule as
    /// libhdf5's `H5D__chunk_file_alloc`, which skips `H5MF_xfree` under
    /// `H5F_ACC_SWMR_WRITE`.
    swmr_active: bool,
    /// Collections with free space — libhdf5's `f->shared->cwfs` list. A
    /// vlen insert fills these partially-filled collection blocks before
    /// creating a new one, so many small writes share 4096-byte blocks
    /// instead of each taking their own. Entries hold `(addr, block size,
    /// free bytes)` hints; the block on disk stays the single truth for
    /// contents, and only the two functions that rewrite collection blocks
    /// ([`insert_vlen_objects`](Self::insert_vlen_objects) and
    /// [`release_vlen_references`](Self::release_vlen_references)) may
    /// update this list. In-memory only, like the allocator's free list:
    /// a reopened file's free space is rediscovered as releases touch its
    /// collections. Capped at [`H5HG_NCWFS`] entries.
    cwfs: Slot<Vec<CwfsEntry>>,
    /// Address of the root group object header (set after first finalize).
    root_group_addr: Option<u64>,
    /// Size of the encoded root group object header (for in-place rewrites).
    root_group_encoded_size: usize,
    /// The on-disk root header block a reopen found, `(addr, len)`, so
    /// finalize can free the block its rewrite supersedes.
    superseded_root_header: Option<(u64, u64)>,
    /// Superblock version this file starts from — the lowest one it may be
    /// written with. [`superblock_version_for`](Self::superblock_version_for)
    /// raises it to what the content needs.
    superblock_version_base: u8,
    /// Objects whose attributes this finalize spilled to dense storage, and
    /// the `Attribute Info` message naming what was written for each.
    ///
    /// INVARIANT: an entry exists here only after every block of that
    /// object's heap and name index is on disk, and only
    /// [`prepare_dense_attributes`](Self::prepare_dense_attributes) may add
    /// one. `emit_attributes` reads it and never builds — a header is sized
    /// and then written by two separate `build_*_header` calls, so a build
    /// that allocated would allocate twice and leave the sized-for blocks
    /// stranded.
    dense_attributes: Slot<HashMap<AttrScope, AttributeInfoMessage>>,
    /// Groups whose links this finalize spilled to dense storage, and the
    /// `Link Info` message naming what was written for each.
    ///
    /// INVARIANT: an entry exists here only after every block of that group's
    /// heap and name index is on disk, and only
    /// [`prepare_dense_links`](Self::prepare_dense_links) may add one.
    dense_links: Slot<HashMap<LinkScope, LinkInfoMessage>>,
    /// The dense storage the reopened object headers already name — the heaps
    /// and indices this session's rewrites and deletes supersede.
    ///
    /// `None` for a file this session created: every block such a file will
    /// hold was allocated here, so there is nothing on disk to supersede and
    /// nothing to allocate for the bookkeeping either.
    ///
    /// INVARIANT: every entry is freed exactly once, by
    /// [`release_superseded_dense_attrs`](Self::release_superseded_dense_attrs)
    /// or [`release_superseded_dense_links`](Self::release_superseded_dense_links),
    /// which remove it as they free. Nothing else may remove one: an entry
    /// that leaves without reaching the allocator is a leaked heap, and one
    /// that reaches it twice hands the same blocks to two objects.
    superseded_dense: Slot<Option<Box<SupersededDense>>>,
    /// The creation-order policy in force: whether an object created from
    /// now on records creation order for its links and its attributes. The
    /// h5py `track_order` analogue; see
    /// [`set_track_order`](Self::set_track_order). Each object captures this
    /// at creation, so changing it never rewrites an object already made.
    track_order: TrackOrder,
    /// The root group's own captured policy. The root is created with the
    /// file, so its value comes from
    /// [`create_with_options`](Self::create_with_options) — or, on reopen,
    /// from the header already on disk.
    root_track_order: TrackOrder,
    /// The root group's stored times, on the same terms as
    /// [`GroupInfo::times`]: whatever a reopened file's root header had, and
    /// `None` for a file this writer created.
    root_times: Option<ObjectTimes>,
    /// Hands out the creation sequence numbers that order a group's links.
    next_creation_seq: Slot<u64>,
    /// Object-reference elements waiting for their target's object header
    /// address, which only exists once finalize has placed every header.
    pending_object_references: Slot<Vec<PendingObjectReference>>,
    /// Set when this writer reopened a version-0/1 file, and never otherwise.
    /// See [`LegacyFile`]; [`object_format`](Self::object_format) is the only
    /// reader of whether it is there.
    legacy: Option<Box<LegacyFile>>,
}

/// One object-reference element written before its value could be known.
///
/// An `H5R_OBJECT1` element is the target's object header address, and
/// addresses are assigned during finalize, so a write records the target by
/// path here and [`Hdf5Writer::apply_object_reference_fixups`] stamps the
/// address in once every header has one.
pub(crate) struct PendingObjectReference {
    /// Dataset holding the element.
    dataset: usize,
    /// Element index within that dataset.
    element: u64,
    /// Path of the object the element names; `/` is the root group.
    target: String,
}

/// What a reopen found already on disk in dense form, by the scope whose
/// header names it.
///
/// Both halves together because they are found together — one walk of the
/// reopened headers fills both — and released together only in the delete
/// path; a finalize supersedes attribute storage before it lays object
/// headers out and link storage after, so each half has its own owner.
#[derive(Debug, Default)]
struct SupersededDense {
    attrs: HashMap<AttrScope, AttributeInfoMessage>,
    links: HashMap<LinkScope, LinkInfoMessage>,
}

/// Which object's attribute list a prepared dense layout belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AttrScope {
    Root,
    Group(usize),
    Dataset(usize),
}

/// Which group's link list a prepared dense layout belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LinkScope {
    Root,
    Group(usize),
}

/// Attributes an object header keeps before libhdf5 spills the whole set to
/// dense storage (`H5O_CRT_ATTR_MAX_COMPACT_DEF`).
const MAX_COMPACT_ATTRS: usize = 8;

/// Links a group header keeps before libhdf5 spills the whole set to dense
/// storage (`H5G_CRT_GINFO_MAX_COMPACT`). This writer emits no phase-change
/// values in the Group Info message, so the default is what applies.
const MAX_COMPACT_LINKS: usize = 8;

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
        Self::create_with_options(path, locking, false)
    }

    /// Create a new HDF5 file at `path` with an explicit locking policy and
    /// creation-order policy.
    ///
    /// `track_order` is the policy the root group is created under and the
    /// default for every object created afterwards; see
    /// [`set_track_order`](Self::set_track_order).
    pub fn create_with_options(
        path: &Path,
        locking: crate::io::locking::FileLocking,
        track_order: bool,
    ) -> IoResult<Self> {
        let handle = FileHandle::create_with_locking(path, locking)?;
        let ctx = FormatContext::default_v3();

        // Reserve the superblock at offset 0. Which version it gets is only
        // known once the file's content is (see `superblock_version_for`),
        // but versions 2 and 3 encode to the same size, so the reservation
        // does not have to wait for that choice.
        let allocator = FileAllocator::new(SuperblockV2V3::size_for(ctx.sizeof_addr) as u64);

        Ok(Self {
            handle,
            allocator,
            ctx,
            datasets: Slot::new(Vec::new()),
            groups: Slot::new(Vec::new()),
            hard_links: Slot::new(Vec::new()),
            preserved_links: Slot::new(Vec::new()),
            root_attributes: Slot::new(Vec::new()),
            create_lock: Slot::new(()),
            libver: LibverBound::Earliest,
            closed: false,
            swmr_active: false,
            cwfs: Slot::new(Vec::new()),
            root_group_addr: None,
            root_group_encoded_size: 0,
            superseded_root_header: None,
            // A new file starts at the oldest superblock this writer's own
            // structures allow, and finalize raises it if the content needs
            // a newer one.
            superblock_version_base: SUPERBLOCK_V2,
            dense_attributes: Slot::new(HashMap::new()),
            dense_links: Slot::new(HashMap::new()),
            superseded_dense: Slot::new(None),
            track_order: TrackOrder::uniform(track_order),
            root_track_order: TrackOrder::uniform(track_order),
            root_times: None,
            next_creation_seq: Slot::new(0),
            pending_object_references: Slot::new(Vec::new()),
            // A file this crate creates is always the modern generation.
            legacy: None,
        })
    }

    /// Target the libhdf5 2.0 file format for datasets created after this
    /// call: filtered chunked datasets get layout message version 5, whose
    /// chunk indexes store chunk sizes in a fixed `sizeof_size`-byte field
    /// with no overflow limit (see [`Self::chunk_layout_version`]). Off by
    /// default, because readers older than libhdf5 2.0 — including the
    /// 1.14-based h5py wheels — reject version 5.
    pub fn set_libver_latest(&mut self, latest: bool) -> IoResult<()> {
        self.set_libver_bound(if latest {
            LibverBound::V200
        } else {
            LibverBound::Earliest
        })
    }

    /// Set the file's low libver bound, the equivalent of
    /// `H5Pset_libver_bounds`'s `low` argument. Objects created after this
    /// call encode their messages at the versions that bound calls for.
    pub fn set_libver_bound(&mut self, libver: LibverBound) -> IoResult<()> {
        // A classic file cannot honour a newer bound: every encoder in it
        // reads `H5F_LOW_BOUND`, and raising that is what makes libhdf5 write
        // the version-2/3 superblock this file does not have. Refused rather
        // than pinned silently, so the caller learns the bound did not take.
        if libver != LibverBound::Earliest && self.is_legacy() {
            return Err(crate::io::IoError::Unsupported(format!(
                "cannot set the library-version bound to {libver:?} on this file: it is                  in the classic (version-0/1 superblock) format, which libhdf5 writes                  only at H5F_LIBVER_EARLIEST"
            )));
        }
        self.libver = libver;
        Ok(())
    }

    /// The file's low libver bound.
    pub fn libver(&self) -> LibverBound {
        self.libver
    }

    /// The generation the *message* encoders follow — dataspace, datatype,
    /// fill value, attribute.
    ///
    /// A property of the file, not of the object: `H5S__set_version`,
    /// `H5O__fill_set_version`, `H5A__set_version` and `H5T_set_version` all
    /// read `H5F_LOW_BOUND(f)` and nothing about the object they are encoding
    /// for. So a creation-order-tracking group in a classic file still gets
    /// version-1 dataspaces and version-1 attribute messages, even though its
    /// own header is version 2.
    fn message_format(&self) -> ObjectFormat {
        match self.legacy {
            Some(_) => ObjectFormat::Legacy,
            None => ObjectFormat::Modern,
        }
    }

    /// The object header version an object with this creation-order policy
    /// gets — `H5O__set_version` (H5Oint.c:251).
    ///
    /// Version 1 is the floor a classic file's low bound sets, but tracking
    /// creation order of *either* kind raises the object past it: the link
    /// creation index lives in the message envelope and the attribute tracking
    /// flags live in the header prefix, and version 1 has neither. This is a
    /// per-object question in a classic file, which is why the format is not
    /// one switch for the whole file — libhdf5 writes version-2 headers inside
    /// a version-0 superblock whenever the creation property list asks for
    /// creation order.
    fn header_format(&self, track: TrackOrder) -> ObjectFormat {
        if self.legacy.is_some() && !track.links.is_tracked() && !track.attrs.is_tracked() {
            ObjectFormat::Legacy
        } else {
            ObjectFormat::Modern
        }
    }

    /// Whether a group whose links have this creation-order policy stores them
    /// in a symbol table — `H5G__obj_create_real` (H5Gobj.c:129).
    ///
    /// The new group format is used unconditionally from `H5F_LIBVER_V18` up,
    /// and below it only when the group tracks link creation order: a symbol
    /// table entry has no room for a creation index. The two axes are
    /// independent — a group that tracks only *attribute* creation order gets
    /// a version-2 header over a symbol table, which is what libhdf5 writes
    /// for it.
    fn uses_symbol_table(&self, links: CreationOrder) -> bool {
        self.legacy.is_some() && !links.is_tracked()
    }

    /// The header format of the registered dataset at `index`.
    ///
    /// A dataset has no links, so only the attribute half of the policy can
    /// raise it past version 1.
    fn dataset_header_format(&self, index: usize) -> ObjectFormat {
        let ds = self.ds(index);
        let attrs = ds.lock().track_attr_order;
        self.header_format(TrackOrder {
            links: CreationOrder::default(),
            attrs,
        })
    }

    /// The header format of the registered group at `index`.
    fn group_header_format(&self, index: usize) -> ObjectFormat {
        let grp = self.grp(index);
        let track = grp.lock().track_order;
        self.header_format(track)
    }

    /// The bound the message encoders see.
    ///
    /// Pinned to `Earliest` in a classic file: `H5F_LIBVER_EARLIEST` is the
    /// only low bound under which libhdf5 writes a version-0/1 superblock at
    /// all, so a newer message version inside one is a combination no libhdf5
    /// produces. Raising the bound on such a file is refused where the caller
    /// asks for it (`H5File::set_libver_bound`) rather than silently dropped
    /// here; this keeps the encoders honest if a path ever misses that gate.
    fn encoding_libver(&self) -> LibverBound {
        match self.message_format() {
            ObjectFormat::Legacy => LibverBound::Earliest,
            ObjectFormat::Modern => self.libver,
        }
    }

    /// Whether this writer is appending to a file whose groups store their
    /// links in symbol tables.
    pub(crate) fn is_legacy(&self) -> bool {
        self.legacy.is_some()
    }

    /// Track and index creation order for the links and the attributes of
    /// every object created after this call — the equivalent of setting
    /// `H5Pset_link_creation_order` and `H5Pset_attr_creation_order` to
    /// `H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED` on the creation
    /// property lists those objects are made with.
    ///
    /// Objects already created keep the policy they were made under, exactly
    /// as libhdf5 keeps what their creation property list said. The root
    /// group is created with the file, so its policy comes from
    /// [`create_with_options`](Self::create_with_options) instead.
    pub fn set_track_order(&mut self, track: bool) {
        self.track_order = TrackOrder::uniform(track);
    }

    /// Layout message version for a new chunked dataset — the
    /// `H5D__chunk_set_info` rule (H5Dchunk.c): version 5 is *required* for
    /// a chunk over 4 GiB (pre-2.0 readers cannot handle one even though the
    /// v4 wire format could express it) and *preferred* for filtered chunks
    /// when the file targets the 2.0 format; everything else stays at
    /// version 4, which every 1.10+ reader accepts.
    fn chunk_layout_version(&self, filtered: bool, chunk_bytes: u64) -> u8 {
        if chunk_bytes > u32::MAX as u64 || (filtered && self.libver == LibverBound::V200) {
            5
        } else {
            4
        }
    }

    /// Width of the stored-chunk-size field in a filtered chunk index:
    /// version 5 uses the fixed `sizeof_size`; version 4 derives it from the
    /// uncompressed chunk byte count (one spare byte included), the
    /// `H5D_*_COMPUTE_CHUNK_SIZE_LEN` rule shared by the extensible-array,
    /// fixed-array and v2-B-tree indexes.
    fn chunk_size_len_for(&self, layout_version: u8, chunk_bytes: u64) -> u8 {
        if layout_version >= 5 {
            self.ctx.sizeof_size
        } else {
            compute_chunk_size_len(chunk_bytes)
        }
    }

    /// Provide public access to the format context.
    pub fn ctx(&self) -> &FormatContext {
        &self.ctx
    }

    /// Number of dataset slots in the registry (including soft-deleted ones).
    pub(crate) fn dataset_count(&self) -> usize {
        self.datasets.lock().len()
    }

    /// Clone out the [`DatasetRef`] for `index`, releasing the registry lock
    /// immediately. Lock the returned ref to read or mutate that one dataset.
    ///
    /// Panics on an out-of-range index, exactly like the `Vec` indexing it
    /// replaces; bounds-checking callers consult [`Self::dataset_count`] first.
    ///
    /// MUST NOT be called while the registry [`Slot`] is already locked (it
    /// would deadlock the `threadsafe` mutex / panic the single-thread
    /// `RefCell`): collect the refs you need, drop the registry guard, then work.
    pub(crate) fn ds(&self, index: usize) -> DatasetRef {
        Shared::clone(&self.datasets.lock()[index])
    }

    /// Number of group slots in the registry (including soft-deleted ones).
    pub(crate) fn group_count(&self) -> usize {
        self.groups.lock().len()
    }

    /// Clone out the [`GroupRef`] for `index`. Same contract as [`Self::ds`].
    pub(crate) fn grp(&self, index: usize) -> GroupRef {
        Shared::clone(&self.groups.lock()[index])
    }

    /// Enter the create gate: take `create_lock` and check that `name` is not
    /// already taken. The returned witness is what [`Self::push_dataset`]
    /// requires, so the uniqueness check and the registry push are atomic
    /// (see `create_lock`) at every creator by construction.
    /// Refuse a storage form this file's format cannot hold.
    ///
    /// A chunked dataset in a classic file is a version-3 data layout message
    /// over a version-1 B-tree chunk index — the shape libhdf5 writes at
    /// default bounds, and the one this crate reads but does not build. The
    /// version-4 layout it does build would drag the superblock to version 3
    /// (`H5O_layout_ver_bounds`, H5Dlayout.c:44), silently converting the file
    /// the append was supposed to leave classic.
    fn reject_unwritable_storage(&self, storage: NewStorage, name: &str) -> IoResult<()> {
        if storage == NewStorage::Chunked && self.is_legacy() {
            return Err(crate::io::IoError::Unsupported(format!(
                "cannot create the chunked dataset '{name}' in this file: it is in the                  classic (version-0/1 superblock) format, whose chunk index is a                  version-1 B-tree that this crate reads but does not write. Create it                  contiguously, or re-create the file at a newer library-version bound"
            )));
        }
        Ok(())
    }

    pub(crate) fn begin_create(
        &self,
        name: &str,
        storage: NewStorage,
    ) -> IoResult<CreateGuard<'_>> {
        let gate = self.create_lock.lock();
        // A creation path through hard links lands in the link's target
        // group, as HDF5 traversal does. Canonicalizing here — the one
        // entry every creator passes — keeps alias forms out of the
        // registry.
        let name = self.canonical_dataset_path(name);
        // A path that leaves this file, or that runs into an object the
        // reopen kept verbatim, is refused here rather than at each creator:
        // this is the one gate every creation passes, so a creator added
        // later cannot forget the check. Both run before the parent lookup,
        // which would otherwise report the group such a path names as absent
        // instead of naming what stops the path. Uniqueness comes first among
        // them: a name already in the file is taken whatever holds it.
        self.reject_external_traversal(&name)?;
        self.ensure_unique_dataset_name(&name)?;
        self.reject_preserved_object(&name)?;
        self.reject_unwritable_storage(storage, &name)?;
        let (parent, _leaf) = self.split_parent(&name)?;
        Ok(CreateGuard {
            _gate: gate,
            name,
            parent,
        })
    }

    /// Split an object path into the group that will hold its link and the
    /// leaf link name, resolving every component through the group registry.
    ///
    /// `path` is the registry form — no leading `/`, e.g. `"grp/sub/late"`.
    /// This is what keeps a `/` out of a link name: HDF5 link names are
    /// single path components (`H5G_traverse` splits on `/` before it ever
    /// reaches `H5L_link`), so a name that carries a path must name a group
    /// that exists, or be refused.
    ///
    /// A missing component is an error rather than an implicit group: the
    /// default link creation property list has `H5Pset_create_intermediate_group`
    /// off, and this writer exposes no property list to turn it on with.
    fn split_parent(&self, path: &str) -> IoResult<(Option<usize>, String)> {
        let (parent_path, leaf) = path.rsplit_once('/').unwrap_or(("", path));
        if leaf.is_empty() {
            return Err(crate::io::IoError::InvalidState(format!(
                "'{path}' does not end in a link name"
            )));
        }
        if parent_path.is_empty() {
            return Ok((None, leaf.to_string()));
        }
        let abs = format!("/{parent_path}");
        let groups = self.group_refs();
        let idx = groups
            .iter()
            .position(|g| {
                let gg = g.lock();
                gg.name == abs && !gg.deleted
            })
            .ok_or_else(|| {
                crate::io::IoError::NotFound(format!(
                    "cannot create '{path}': group '{abs}' does not exist"
                ))
            })?;
        Ok((Some(idx), leaf.to_string()))
    }

    /// Push a freshly-built dataset into the registry and return its index.
    /// Takes the registry lock only for the push, so it does not block an
    /// in-flight write that already cloned its own [`DatasetRef`] out.
    /// The [`CreateGuard`] proves the caller entered through
    /// [`Self::begin_create`] and still holds the gate.
    pub(crate) fn push_dataset(&self, create: &CreateGuard<'_>, info: DatasetInfo) -> usize {
        let idx = {
            let mut reg = self.datasets.lock();
            let idx = reg.len();
            reg.push(Shared::new(DatasetCell::new(info)));
            idx
        };
        // The spine guard is dropped before the group slot is taken: the lock
        // order is spine -> slot and never the reverse.
        if let Some(pidx) = create.parent {
            self.grp(pidx).lock().child_datasets.push(idx);
        }
        idx
    }

    /// Push a freshly-built group into the registry and return its index.
    pub(crate) fn push_group(&self, info: GroupInfo) -> usize {
        let mut reg = self.groups.lock();
        let idx = reg.len();
        reg.push(Shared::new(Slot::new(info)));
        idx
    }

    /// Snapshot every [`DatasetRef`] (spine lock held only for the clone).
    /// Iterate the snapshot to lock each dataset one at a time — this keeps
    /// the lock order *spine → slot* and never reacquires the spine while a
    /// slot is held, which is what makes the registry deadlock-free.
    pub(crate) fn dataset_refs(&self) -> Vec<DatasetRef> {
        self.datasets.lock().iter().map(Shared::clone).collect()
    }

    /// Snapshot every [`GroupRef`]; see [`Self::dataset_refs`].
    pub(crate) fn group_refs(&self) -> Vec<GroupRef> {
        self.groups.lock().iter().map(Shared::clone).collect()
    }

    /// Snapshot the hard-link list (the lock is held only for the clone), so
    /// callers can resolve each link's target/parent — which locks dataset and
    /// group slots — without holding the hard-link lock.
    /// The next creation sequence number.
    ///
    /// One monotonic counter for datasets, groups and hard links alike: a
    /// group orders its links by it, so an interleaved run of `create_group`
    /// and `create_dataset` comes back out in the order it was made rather
    /// than grouped by kind.
    fn take_creation_seq(&self) -> u64 {
        let mut next = self.next_creation_seq.lock();
        let seq = *next;
        *next += 1;
        seq
    }

    pub(crate) fn hard_links_vec(&self) -> Vec<HardLink> {
        self.hard_links.lock().clone()
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
        let mut handle = FileHandle::open_readwrite_with_locking(path, locking)?;
        // The same `H5FD_locate_signature` search the read path makes, through
        // the same handle mechanism: the offset it finds is the file's base
        // address, so the allocator's end-of-file, every write and the
        // superblock rewrite all work in the HDF5 address space, and the
        // userblock in `[0, base)` is not addressable from this writer at all.
        let super_addr = handle
            .locate_signature()?
            .ok_or(crate::format::FormatError::InvalidSignature)?;
        handle.set_base(super_addr);
        let file_size = handle.file_size()?;

        let sb_buf = handle.read_at_most(0, 256)?;
        // Which generation the file is decides everything the close then
        // writes back: version-1 object headers and symbol-table groups over a
        // version-0/1 superblock, or version-2 headers and link-message groups
        // over a version-2/3 one. libhdf5 writes those two combinations and no
        // mixture of them, so the branch is taken once, here, and carried as
        // `legacy`.
        let version = crate::format::superblock::detect_superblock_version(&sb_buf)?;
        let (ctx, sb_btree, root_addr, ext_addr, legacy) = if version <= 1 {
            let sb = SuperblockV0V1::decode(&sb_buf)?;
            let ctx = FormatContext {
                sizeof_addr: sb.sizeof_offsets,
                sizeof_size: sb.sizeof_lengths,
            };
            // Unlike a v2/v3 superblock, a classic one carries the "K" ranks
            // itself; every v1-B-tree and symbol-table node width in the file
            // comes from them.
            let btree = crate::format::btree_v1::BTreeV1Config {
                sym_leaf_k: sb.sym_leaf_k,
                snode_internal_k: sb.btree_internal_k,
                chunk_internal_k: sb.indexed_storage_k.unwrap_or(32),
            };
            let root = sb.root_symbol_table_entry.obj_header_addr;
            let ext = sb.superblock_extension_address;
            (ctx, btree, root, ext, Some(sb))
        } else {
            let sb = SuperblockV2V3::decode(&sb_buf)?;
            let ctx = FormatContext {
                sizeof_addr: sb.sizeof_offsets,
                sizeof_size: sb.sizeof_lengths,
            };
            (
                ctx,
                crate::format::btree_v1::BTreeV1Config::default(),
                sb.root_group_object_header_address,
                sb.superblock_extension_address,
                None,
            )
        };

        // The reopen reads object headers exactly as the reader does, so it
        // needs the same file-level parameters: a v2/v3 superblock carries no
        // B-tree K values, and only the extension can override the defaults.
        let (meta, ext) = crate::io::reader::Hdf5Reader::read_extension_and_meta(
            &mut handle,
            ctx,
            sb_btree,
            ext_addr,
        )?;

        // A file with shared object header messages keeps datatypes,
        // dataspaces and attributes in a SOHM fractal heap, and each object
        // header holds only a heap ID pointing at them. This crate reads that
        // indirection but never writes it, so finalize would rewrite every
        // header it touches with the shared messages dropped and the master
        // table left claiming they are still referenced — a file libhdf5 then
        // reads as missing its objects. Refuse before anything is written.
        let nindexes = ext
            .shared_message_table
            .as_ref()
            .map_or(0, |smt| smt.nindexes);
        if nindexes > 0 {
            return Err(crate::io::IoError::InvalidState(format!(
                "cannot open this file for appending: its superblock extension \
                 declares {nindexes} shared object header message (SOHM) \
                 index(es), which this crate can read but not write"
            )));
        }
        // Discover links from root group (and subgroups recursively). Every
        // object is classified before it is registered, and the root is the
        // one object with no alternative: its header must be rewritten to
        // hold anything new, so an unmodellable root is refused here rather
        // than rewritten into whatever this writer could read of it.
        let mut walk = ReopenWalk::new(&mut handle, &meta, file_size);
        let root = match walk.plan(root_addr)? {
            ObjectPlan::Group(parts) => parts,
            ObjectPlan::Dataset(_) => {
                return Err(crate::io::IoError::InvalidState(
                    "cannot open this file for appending: its root object is a dataset, \
                     not a group"
                        .into(),
                ))
            }
            ObjectPlan::Preserve(why) => {
                return Err(crate::io::IoError::Unsupported(format!(
                    "cannot open this file for appending: {why}. Every append rewrites the \
                     root group's header, and this writer will not rewrite it from the part \
                     of it that it can read"
                )));
            }
        };
        // Chunk 0 is the block the rewrite supersedes.
        let root_header_size = root.header_size;
        let root_attributes = root.attributes;
        let root_track_order = root.track_order;
        let root_times = root.times;
        let root_dense = root.dense;
        let root_stab = root.stab;

        walk.group(&root.links, "", 0)?;
        let collected = walk.finish();
        let mut link_entries = collected.hard;
        let mut preserved = collected.preserved;
        // Objects the loop below could not rebuild, by header address, so the
        // other links to one are preserved with it rather than left pointing
        // at a registry entry that is no longer there.
        let mut unrebuilt: std::collections::HashMap<u64, String> = Default::default();

        // Two link entries can share one object header — hard links. Only
        // the first-walked path becomes the object; the rest are rebuilt
        // as hard-link registry entries further down. Without this split
        // every alias came back as its own DatasetInfo carrying the same
        // storage addresses, so deleting (or finalizing) one freed blocks
        // the others still referenced.
        let mut seen_header_addrs = std::collections::HashSet::new();
        let mut alias_entries: Vec<HardEntry> = Vec::new();
        link_entries.retain(|(entry, _)| {
            if seen_header_addrs.insert(entry.address) {
                true
            } else {
                alias_entries.push(entry.clone());
                false
            }
        });

        // The order the walk met each object, kept before the loop below
        // consumes the entries: `ensure_groups_for` needs parents to precede
        // children.
        let walk_order: Vec<String> = link_entries.iter().map(|(e, _)| e.path.clone()).collect();

        let mut existing_datasets = Vec::new();
        // Non-dataset link targets (groups): header block `(addr, len)` by
        // link path — so finalize can free the block its rewrite supersedes —
        // plus the attributes the header carries, which the group registry
        // below must keep or finalize rewrites the group without them.
        type GroupHeaderInfo = (
            u64,
            usize,
            Vec<AttributeEntry>,
            TrackOrder,
            Option<ObjectTimes>,
        );
        let mut group_headers: std::collections::HashMap<String, GroupHeaderInfo> =
            Default::default();
        // The dense storage each rebuilt dataset's header named, by registry
        // index, so finalize frees exactly what its rewrite supersedes. Keyed
        // after the rebuild succeeded: a preserved dataset keeps its header,
        // and freeing the heap that header still names would strand it.
        let mut dataset_dense: Vec<(usize, AttributeInfoMessage)> = Vec::new();
        let mut group_dense: Vec<(String, DenseCarry)> = Vec::new();
        // The same, for the symbol-table storage a classic group's header
        // names: keyed by path here, by registry index once every group has
        // one.
        let mut group_stabs: Vec<(String, StabExtents)> = Vec::new();
        for (entry, object) in link_entries {
            let HardEntry {
                path: name,
                address: obj_addr,
                encoded,
            } = entry;
            let parts = match object {
                CollectedObject::Group {
                    header_size,
                    attributes,
                    track_order,
                    times,
                    dense,
                    stab,
                } => {
                    group_dense.push((name.clone(), dense));
                    if let Some(stab) = stab {
                        group_stabs.push((name.clone(), stab));
                    }
                    group_headers.insert(
                        name,
                        (obj_addr, header_size, attributes, track_order, times),
                    );
                    continue;
                }
                CollectedObject::Dataset(parts) => *parts,
            };
            let dense_attrs = parts.dense.attrs.clone();
            match rebuild_dataset(&mut handle, &ctx, name.clone(), obj_addr, parts) {
                Ok(info) => {
                    if let Some(ainfo) = dense_attrs {
                        dataset_dense.push((existing_datasets.len(), ainfo));
                    }
                    existing_datasets.push(info);
                }
                // Kept by its bytes for the same reason a header this walk
                // could not decode is: the rewrite would otherwise emit an
                // object whose chunk index no longer names its chunks.
                Err(e) => {
                    let why = format!("this writer could not rebuild its chunk index: {e}");
                    unrebuilt.insert(obj_addr, why.clone());
                    preserved.push(PreservedEntry {
                        path: name,
                        class: crate::io::reader::LinkClass::Hard,
                        encoded,
                        reason: Some(why),
                    });
                }
            }
        }

        // Reconstruct the group registry. Every group is a link entry of its
        // own, whether or not a dataset lives under it, so the registry is
        // built from the discovered links — rebuilding it from dataset paths
        // alone made attribute-only and empty groups vanish at close, and
        // dropped the attributes of the groups that survived.
        let mut groups: Vec<GroupInfo> = Vec::new();
        let mut group_index_map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        // Register the chain of groups "/a", "/a/b", … for the link-style
        // path `link_path` ("a/b"), taking each one's on-disk header block
        // and attributes out of `group_headers` when the link walk saw it.
        fn ensure_groups_for(
            link_path: &str,
            groups: &mut Vec<GroupInfo>,
            group_index_map: &mut std::collections::HashMap<String, usize>,
            group_headers: &mut std::collections::HashMap<String, GroupHeaderInfo>,
        ) {
            let mut path = String::new();
            for part in link_path.split('/') {
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
                let (
                    obj_header_written_addr,
                    obj_header_encoded_size,
                    attributes,
                    track_order,
                    times,
                ) = group_headers.remove(path.trim_start_matches('/')).map_or(
                    (None, 0, Vec::new(), TrackOrder::default(), None),
                    |(addr, len, attrs, track, times)| (Some(addr), len, attrs, track, times),
                );
                groups.push(GroupInfo {
                    name: path.clone(),
                    parent,
                    creation_seq: 0,
                    track_order,
                    times,
                    child_datasets: Vec::new(),
                    child_groups: Vec::new(),
                    obj_header_addr: 0,
                    obj_header_written_addr,
                    obj_header_encoded_size,
                    deleted: false,
                    attributes,
                });
                if let Some(pidx) = parent {
                    groups[pidx].child_groups.push(gidx);
                }
                group_index_map.insert(path.clone(), gidx);
            }
        }

        // Every linked group, in link-walk order (parents precede children).
        for name in &walk_order {
            if group_headers.contains_key(name.as_str()) {
                ensure_groups_for(name, &mut groups, &mut group_index_map, &mut group_headers);
            }
        }

        // Assign each dataset to its immediate parent group, creating any
        // group the link walk could not decode (its chain stays placeholder).
        for (di, ds) in existing_datasets.iter().enumerate() {
            let parts: Vec<&str> = ds.name.split('/').collect();
            if parts.len() <= 1 {
                continue; // root-level dataset, no group
            }
            let parent_link_path = parts[..parts.len() - 1].join("/");
            ensure_groups_for(
                &parent_link_path,
                &mut groups,
                &mut group_index_map,
                &mut group_headers,
            );
            let gidx = group_index_map[&format!("/{}", parent_link_path)];
            groups[gidx].child_datasets.push(di);
        }

        // An object the rebuild above gave up on is preserved by its bytes,
        // so the other links to it are preserved too: there is no registry
        // entry for them to name.
        alias_entries.retain(|entry| match unrebuilt.get(&entry.address) {
            None => true,
            Some(why) => {
                preserved.push(PreservedEntry {
                    path: entry.path.clone(),
                    class: crate::io::reader::LinkClass::Hard,
                    encoded: entry.encoded.clone(),
                    reason: Some(why.clone()),
                });
                false
            }
        });

        // Rebuild the hard-link registry from the alias entries set aside
        // above, so the H5Ldelete semantics survive a reopen. An alias whose
        // target the walk could not model is not here at all: it was
        // preserved by its own bytes, exactly as the first link to that
        // object was.
        let mut hard_links: Vec<HardLink> = Vec::new();
        for HardEntry {
            path,
            address: addr,
            ..
        } in alias_entries
        {
            let target = if let Some(di) = existing_datasets
                .iter()
                .position(|d| d.obj_header_addr == addr)
            {
                HardLinkTarget::Dataset(di)
            } else if let Some(gi) = groups
                .iter()
                .position(|g| g.obj_header_written_addr == Some(addr))
            {
                HardLinkTarget::Group(gi)
            } else {
                continue;
            };
            let (parent, link_name) = match path.rsplit_once('/') {
                None => (None, path),
                Some((dir, leaf)) => {
                    ensure_groups_for(dir, &mut groups, &mut group_index_map, &mut group_headers);
                    (
                        group_index_map.get(&format!("/{dir}")).copied(),
                        leaf.to_string(),
                    )
                }
            };
            hard_links.push(HardLink {
                parent,
                name: link_name,
                target,
                creation_seq: 0,
            });
        }

        // Attach every link the writer cannot express to the group that
        // holds it, so the rewrite of that group's header emits it again.
        // `ensure_groups_for` registers the parent chain, which matters for
        // a group whose only content is such a link: nothing else would put
        // it in the registry, and the close would drop group and link alike.
        let mut preserved_links: Vec<PreservedLink> = Vec::new();
        for PreservedEntry {
            path,
            class,
            encoded,
            reason,
        } in preserved
        {
            let (parent, link_name) = match path.rsplit_once('/') {
                None => (None, path),
                Some((dir, leaf)) => {
                    ensure_groups_for(dir, &mut groups, &mut group_index_map, &mut group_headers);
                    (
                        group_index_map.get(&format!("/{dir}")).copied(),
                        leaf.to_string(),
                    )
                }
            };
            preserved_links.push(PreservedLink {
                parent,
                name: link_name,
                class,
                encoded,
                reason,
            });
        }

        // Stamp the creation sequence a reopened file cannot supply. Nothing
        // on disk says which link was made first unless the group tracked
        // creation order, and this reader does not carry that back out, so
        // discovery order is what there is: datasets, then groups, then the
        // hard links found beside them — the order the writer emitted links
        // in before it ordered them at all.
        let mut creation_seq = 0u64;
        for d in &mut existing_datasets {
            d.creation_seq = creation_seq;
            creation_seq += 1;
        }
        for g in &mut groups {
            g.creation_seq = creation_seq;
            creation_seq += 1;
        }
        for l in &mut hard_links {
            l.creation_seq = creation_seq;
            creation_seq += 1;
        }

        let allocator = FileAllocator::new(file_size);

        // Now that every object has its registry index, key the dense storage
        // found on disk by the scope that will supersede it. A group the link
        // walk saw but never registered is not rewritten either, so leaving it
        // out is what keeps its storage referenced.
        let mut superseded = SupersededDense {
            attrs: dataset_dense
                .into_iter()
                .map(|(di, ainfo)| (AttrScope::Dataset(di), ainfo))
                .collect(),
            links: HashMap::new(),
        };
        superseded
            .attrs
            .extend(root_dense.attrs.map(|a| (AttrScope::Root, a)));
        superseded
            .links
            .extend(root_dense.links.map(|l| (LinkScope::Root, l)));
        for (name, dense) in group_dense {
            let Some(&gidx) = group_index_map.get(&format!("/{name}")) else {
                continue;
            };
            superseded
                .attrs
                .extend(dense.attrs.map(|a| (AttrScope::Group(gidx), a)));
            superseded
                .links
                .extend(dense.links.map(|l| (LinkScope::Group(gidx), l)));
        }
        let superseded = (!superseded.attrs.is_empty() || !superseded.links.is_empty())
            .then(|| Box::new(superseded));

        // The same keying for the symbol-table storage, and the superblock the
        // close re-emits. Built from the superblock alone: a group whose header
        // carried no Symbol Table message contributes nothing, which is exactly
        // what happens to a group libhdf5 wrote at a newer bound inside an
        // otherwise classic file.
        let legacy = legacy.map(|superblock| {
            let mut stabs: HashMap<LinkScope, StabExtents> = HashMap::new();
            stabs.extend(root_stab.map(|s| (LinkScope::Root, s)));
            for (name, extents) in group_stabs {
                if let Some(&gidx) = group_index_map.get(&format!("/{name}")) {
                    stabs.insert(LinkScope::Group(gidx), extents);
                }
            }
            Box::new(LegacyFile {
                superblock,
                meta: meta.clone(),
                superseded: Slot::new(stabs),
                written: Slot::new(HashMap::new()),
            })
        });

        // Wrap the reconstructed plain vecs into the per-slot registry. The
        // reconstruction logic above runs single-threaded on local `Vec`s;
        // only the final hand-off needs the `Shared<Slot<_>>` shape.
        let datasets = existing_datasets
            .into_iter()
            .map(|i| Shared::new(DatasetCell::new(i)))
            .collect();
        let groups = groups
            .into_iter()
            .map(|g| Shared::new(Slot::new(g)))
            .collect();

        let writer = Self {
            handle,
            allocator,
            ctx,
            datasets: Slot::new(datasets),
            groups: Slot::new(groups),
            hard_links: Slot::new(hard_links),
            preserved_links: Slot::new(preserved_links),
            root_attributes: Slot::new(root_attributes),
            create_lock: Slot::new(()),
            libver: LibverBound::Earliest,
            closed: false,
            swmr_active: false,
            cwfs: Slot::new(Vec::new()),
            root_group_addr: None,
            root_group_encoded_size: 0,
            superseded_root_header: Some((root_addr, root_header_size as u64)),
            // Start from the version the file already has, so an append that
            // adds nothing needing a newer one hands the file back as it
            // found it.
            superblock_version_base: version,
            // The reopened file's own policy, so objects added in this
            // session are made the way the file already declares.
            root_track_order,
            root_times,
            dense_attributes: Slot::new(HashMap::new()),
            dense_links: Slot::new(HashMap::new()),
            superseded_dense: Slot::new(superseded),
            track_order: root_track_order,
            next_creation_seq: Slot::new(creation_seq),
            pending_object_references: Slot::new(Vec::new()),
            legacy,
        };
        // The link graph is complete only now, so this is the first point the
        // count each on-disk header was written with can be read off it: in a
        // well-formed file the links the walk found reaching an object *are*
        // that count, so nothing has to be decoded out of the headers.
        for i in 0..writer.dataset_count() {
            let nlink = writer.object_link_count(HardLinkTarget::Dataset(i));
            writer.ds(i).lock().nlink_written = nlink;
        }
        Ok(writer)
    }

    /// Return the names of all datasets created so far.
    pub fn dataset_names(&self) -> Vec<String> {
        self.dataset_refs()
            .iter()
            .filter_map(|d| {
                let g = d.lock();
                (!g.deleted).then(|| g.name.clone())
            })
            .collect()
    }

    /// Find a dataset index by name. Like `H5Dopen`, the name may be any
    /// link path to the dataset: a user hard link's path — or a path
    /// whose group components pass through such links — resolves to its
    /// target.
    pub fn dataset_index(&self, name: &str) -> Option<usize> {
        let name = self.canonical_dataset_path(name);
        self.dataset_refs()
            .iter()
            .position(|d| {
                let g = d.lock();
                g.name == name && !g.deleted
            })
            .or_else(|| {
                self.hard_links_vec().iter().find_map(|l| match l.target {
                    HardLinkTarget::Dataset(i)
                        if self.hard_link_emitted(l) && self.hard_link_full_path(l) == name =>
                    {
                        Some(i)
                    }
                    _ => None,
                })
            })
    }

    /// Reconstruct the fields a writer-mode `H5Dataset` handle needs for the
    /// dataset at `index`: `(shape, element_size, chunked, btree2,
    /// fixed_array)`. Single owner of this mapping so `H5File::dataset_writer`,
    /// `H5Group::dataset_writer`, and the vlen-string helpers all agree.
    pub(crate) fn dataset_handle_parts(
        &self,
        index: usize,
    ) -> (Vec<usize>, usize, bool, bool, bool) {
        let ds = self.ds(index);
        let g = ds.lock();
        let shape: Vec<usize> = g.dataspace.dims.iter().map(|&d| d as usize).collect();
        let element_size = g.datatype.element_size() as usize;
        let (fixed_array, btree2, has_chunked) = (
            g.fixed_array.is_some(),
            g.btree_v2.is_some(),
            g.chunked.is_some(),
        );
        let chunked = has_chunked || fixed_array || btree2;
        (shape, element_size, chunked, btree2, fixed_array)
    }

    /// Reject a dataset name already used by a live dataset. Dataset names
    /// here are full paths, so they must be unique across the file (HDF5
    /// requires link names to be unique within their group).
    fn ensure_unique_dataset_name(&self, name: &str) -> IoResult<()> {
        // A path *through* a carried external link belongs to the other file:
        // creating it here would put a second link of that name in the group.
        self.reject_external_traversal(name)?;
        let exists = self.dataset_refs().iter().any(|d| {
            let g = d.lock();
            !g.deleted && g.name == name
        });
        if exists {
            return Err(crate::io::IoError::InvalidState(format!(
                "a dataset named '{name}' already exists"
            )));
        }
        if self
            .hard_links_vec()
            .iter()
            .any(|l| self.hard_link_emitted(l) && self.hard_link_full_path(l) == name)
        {
            return Err(crate::io::IoError::InvalidState(format!(
                "a hard link named '{name}' already exists"
            )));
        }
        // A preserved link occupies its name in the group just as a modelled
        // one does; both are emitted, and two link messages of one name in a
        // group is an invalid file.
        if self.preserved_link_paths().iter().any(|(p, _)| *p == name) {
            return Err(crate::io::IoError::InvalidState(format!(
                "a link named '{name}' already exists"
            )));
        }
        Ok(())
    }

    /// Delete a dataset name, with libhdf5's `H5Ldelete` semantics: a name
    /// is only a link. If `name` is a user hard link's path, just that
    /// link is removed and the object is untouched. If it is the tree name
    /// and a user hard link still names the object, the object survives
    /// under it — the link becomes the primary name and nothing is freed.
    /// Only deleting the *last* name soft-deletes the object and frees the
    /// file space it owned: its chunk blocks and chunk-index structures
    /// (or contiguous data block), the global-heap objects of its
    /// variable-length data and attributes, and — on a reopened file — the
    /// on-disk object header block. The freed space is reused by later
    /// allocations in this session; the file does not shrink.
    ///
    /// Refused while SWMR streaming is active: a live reader may hold any
    /// of those addresses (libhdf5 forbids link deletion during SWMR
    /// writes too).
    pub fn delete_dataset(&self, name: &str) -> IoResult<()> {
        if self.swmr_active {
            return Err(swmr_delete_error(name));
        }
        self.reject_external_traversal(name)?;
        // The gate keeps the link list and child lists still while this
        // delete reads and rewrites them (create_lock → op → slot order,
        // the same as every creator).
        let _create = self.create_lock.lock();
        // `H5Ldelete` resolves the path through links only *up to* the
        // leaf — the leaf is what gets deleted, so a leaf naming a user
        // link must stay literal and be unlinked, not its target.
        let name = match name.rsplit_once('/') {
            None => name.to_string(),
            Some((dir, leaf)) => format!(
                "{}/{leaf}",
                self.canonical_group_path(&format!("/{dir}"))
                    .trim_start_matches('/')
            ),
        };
        let refs = self.dataset_refs();
        let idx = match refs.iter().position(|d| {
            let g = d.lock();
            g.name == name && !g.deleted
        }) {
            Some(i) => i,
            None => {
                // Not a tree name — the path may name a user hard link,
                // and deleting a link path unlinks just that link (the
                // creation collision checks keep the two namespaces
                // disjoint, so the order of the lookups cannot matter).
                let link = self.hard_links_vec().iter().position(|l| {
                    self.hard_link_emitted(l)
                        && matches!(l.target, HardLinkTarget::Dataset(_))
                        && self.hard_link_full_path(l) == name
                });
                let Some(pos) = link else {
                    return Err(crate::io::IoError::NotFound(name));
                };
                self.hard_links.lock().remove(pos);
                return Ok(());
            }
        };
        // A surviving hard link keeps the object: promote the first one to
        // the primary name and delete nothing.
        let promote = self.hard_links_vec().iter().position(|l| {
            self.hard_link_emitted(l) && matches!(l.target, HardLinkTarget::Dataset(i) if i == idx)
        });
        if let Some(pos) = promote {
            self.promote_dataset_to_link(idx, pos);
            return Ok(());
        }
        refs[idx].lock().deleted = true;
        // Remove from parent group's child_datasets
        for grp in self.group_refs() {
            grp.lock().child_datasets.retain(|&di| di != idx);
        }
        self.purge_dead_hard_links();
        let ds = self.ds(idx);
        let _op = ds.op.lock();
        self.release_dataset_storage(idx)
    }

    /// Soft-delete a group and all its child datasets and sub-groups,
    /// freeing every deleted object's file space the way
    /// [`delete_dataset`](Self::delete_dataset) does — with the same
    /// `H5Ldelete` semantics: a `name` that is a user hard link's path
    /// unlinks just that link, and hard links from *outside* the subtree
    /// keep their targets. A dataset or group such a link names survives,
    /// re-homed under the link (a group brings its whole subtree with
    /// it); a link naming the deleted group itself turns the call into a
    /// pure rename and nothing is freed. Refused while SWMR streaming is
    /// active, same rule as `delete_dataset`.
    pub fn delete_group(&self, name: &str) -> IoResult<()> {
        if self.swmr_active {
            return Err(swmr_delete_error(name));
        }
        self.reject_external_traversal(name)?;
        // Same gate as `delete_dataset`: the pre-scan below and the
        // promotions must see a still link list and child lists.
        let _create = self.create_lock.lock();
        let name = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{}", name)
        };
        // Leaf stays literal, directory resolves through links — the
        // same `H5Ldelete` rule as `delete_dataset`.
        let name = match name.rsplit_once('/') {
            Some((dir, leaf)) if !dir.is_empty() => {
                format!("{}/{leaf}", self.canonical_group_path(dir))
            }
            _ => name,
        };
        let groups = self.group_refs();
        let gidx = match groups.iter().position(|g| {
            let gg = g.lock();
            gg.name == name && !gg.deleted
        }) {
            Some(i) => i,
            None => {
                // Same `H5Ldelete` rule as `delete_dataset`: a path naming
                // a user hard link to a group unlinks just that link.
                let trimmed = name.trim_start_matches('/');
                let link = self.hard_links_vec().iter().position(|l| {
                    self.hard_link_emitted(l)
                        && matches!(l.target, HardLinkTarget::Group(_))
                        && self.hard_link_full_path(l) == trimmed
                });
                let Some(pos) = link else {
                    return Err(crate::io::IoError::NotFound(name.clone()));
                };
                self.hard_links.lock().remove(pos);
                return Ok(());
            }
        };

        // A link is "outside" when its parent group does not die with the
        // subtree; only outside links can keep their targets alive.
        fn outside(parent: Option<usize>, doomed_gs: &[usize]) -> bool {
            match parent {
                None => true,
                Some(pi) => !doomed_gs.contains(&pi),
            }
        }
        // A group an outside link names survives, re-homed with its whole
        // subtree under the link. Each promotion moves that subtree out of
        // the doomed set — and can turn a link inside it into an outside
        // one — so rescan from scratch until no promotable group is left.
        // Promoting `gidx` itself makes the delete a pure rename: return.
        let mut doomed_ds = Vec::new();
        let mut doomed_gs = Vec::new();
        loop {
            doomed_ds.clear();
            doomed_gs.clear();
            self.collect_live_subtree(gidx, &mut doomed_ds, &mut doomed_gs);
            let promote = self
                .hard_links_vec()
                .iter()
                .enumerate()
                .find_map(|(pos, l)| match l.target {
                    HardLinkTarget::Group(gi)
                        if self.hard_link_emitted(l)
                            && outside(l.parent, &doomed_gs)
                            && doomed_gs.contains(&gi) =>
                    {
                        Some((pos, gi))
                    }
                    _ => None,
                });
            let Some((pos, gi)) = promote else { break };
            self.promote_group_to_link(gi, pos);
            if gi == gidx {
                return Ok(());
            }
        }
        // A dataset an outside link names survives its container: re-home
        // it under the link now, so the marking pass below never sees it.
        for di in doomed_ds {
            let promote = self.hard_links_vec().iter().position(|l| {
                self.hard_link_emitted(l)
                    && outside(l.parent, &doomed_gs)
                    && matches!(l.target, HardLinkTarget::Dataset(i) if i == di)
            });
            if let Some(pos) = promote {
                self.promote_dataset_to_link(di, pos);
            }
        }

        let mut ds_deleted = Vec::new();
        let mut gs_deleted = Vec::new();
        self.delete_group_recursive(gidx, &mut ds_deleted, &mut gs_deleted);
        // Remove from parent's child_groups
        let parent = groups[gidx].lock().parent;
        if let Some(pidx) = parent {
            groups[pidx].lock().child_groups.retain(|&gi| gi != gidx);
        }
        self.purge_dead_hard_links();
        // Free storage only after the whole subtree is marked: the lists
        // hold each object exactly once (the marking pass skips anything
        // already deleted), so nothing is freed twice.
        for di in ds_deleted {
            let ds = self.ds(di);
            let _op = ds.op.lock();
            self.release_dataset_storage(di)?;
        }
        for gi in gs_deleted {
            self.release_group_storage(gi)?;
        }
        Ok(())
    }

    /// Collect the live (not soft-deleted) members of `gidx`'s subtree,
    /// each exactly once, without changing anything — the read-only twin
    /// of [`delete_group_recursive`](Self::delete_group_recursive), for
    /// the pre-scan that must run before any marking.
    fn collect_live_subtree(&self, gidx: usize, ds_out: &mut Vec<usize>, gs_out: &mut Vec<usize>) {
        if gs_out.contains(&gidx) {
            return;
        }
        let (child_ds, child_gs) = {
            let grp = self.grp(gidx);
            let g = grp.lock();
            if g.deleted {
                return;
            }
            (g.child_datasets.clone(), g.child_groups.clone())
        };
        gs_out.push(gidx);
        for di in child_ds {
            if !self.ds(di).lock().deleted && !ds_out.contains(&di) {
                ds_out.push(di);
            }
        }
        for gi in child_gs {
            self.collect_live_subtree(gi, ds_out, gs_out);
        }
    }

    /// Re-home dataset `idx` under the hard link at `pos` in the link
    /// list — the surviving half of `H5Ldelete`: the link leaves the user
    /// list and becomes the dataset's primary (tree) name, in the link's
    /// parent group. Storage is untouched; any further links to the
    /// dataset stay in the list and keep resolving.
    fn promote_dataset_to_link(&self, idx: usize, pos: usize) {
        let link = self.hard_links.lock().remove(pos);
        let new_name = self.hard_link_full_path(&link);
        for grp in self.group_refs() {
            grp.lock().child_datasets.retain(|&di| di != idx);
        }
        if let Some(pi) = link.parent {
            self.grp(pi).lock().child_datasets.push(idx);
        }
        self.ds(idx).lock().name = new_name;
    }

    /// The group counterpart of
    /// [`promote_dataset_to_link`](Self::promote_dataset_to_link): re-home
    /// group `gidx` under the hard link at `pos`, bringing its whole
    /// subtree with it. Names are stored as full paths, so every live
    /// descendant is renamed by prefix.
    fn promote_group_to_link(&self, gidx: usize, pos: usize) {
        let link = self.hard_links.lock().remove(pos);
        let new_name = format!("/{}", self.hard_link_full_path(&link));
        let old_name = self.grp(gidx).lock().name.clone();
        for grp in self.group_refs() {
            grp.lock().child_groups.retain(|&g| g != gidx);
        }
        {
            let grp = self.grp(gidx);
            let mut g = grp.lock();
            g.parent = link.parent;
            g.name = new_name.clone();
        }
        if let Some(pi) = link.parent {
            self.grp(pi).lock().child_groups.push(gidx);
        }

        let mut ds_in = Vec::new();
        let mut gs_in = Vec::new();
        self.collect_live_subtree(gidx, &mut ds_in, &mut gs_in);
        // Group names carry a leading '/' ("/a/b"), dataset names none
        // ("a/b/ds") — two prefix forms of the same rename.
        let old_grp_prefix = format!("{old_name}/");
        let new_grp_prefix = format!("{new_name}/");
        let old_ds_prefix = old_grp_prefix.trim_start_matches('/').to_string();
        let new_ds_prefix = new_grp_prefix.trim_start_matches('/').to_string();
        for gi in gs_in {
            if gi == gidx {
                continue;
            }
            let grp = self.grp(gi);
            let mut g = grp.lock();
            let renamed = g
                .name
                .strip_prefix(&old_grp_prefix)
                .map(|rest| format!("{new_grp_prefix}{rest}"));
            if let Some(n) = renamed {
                g.name = n;
            }
        }
        for di in ds_in {
            let ds = self.ds(di);
            let mut d = ds.lock();
            let renamed = d
                .name
                .strip_prefix(&old_ds_prefix)
                .map(|rest| format!("{new_ds_prefix}{rest}"));
            if let Some(n) = renamed {
                d.name = n;
            }
        }
    }

    /// Drop hard-link entries that can no longer be emitted — their parent
    /// group or target object was just deleted — so the list mirrors what
    /// the file will hold instead of carrying suppressed zombies.
    fn purge_dead_hard_links(&self) {
        let dead: Vec<usize> = self
            .hard_links_vec()
            .iter()
            .enumerate()
            .filter(|(_, l)| !self.hard_link_emitted(l))
            .map(|(p, _)| p)
            .collect();
        let mut links = self.hard_links.lock();
        for p in dead.into_iter().rev() {
            links.remove(p);
        }
    }

    /// Mark `gidx` and its subtree deleted, appending each newly-deleted
    /// object's index to `ds_out` / `gs_out` exactly once — the caller
    /// frees their storage, and an object reachable twice (or a subtree
    /// already deleted) must not be freed twice.
    fn delete_group_recursive(
        &self,
        gidx: usize,
        ds_out: &mut Vec<usize>,
        gs_out: &mut Vec<usize>,
    ) {
        // Mark deleted and snapshot the child lists, releasing the group lock
        // before locking any dataset/child-group slot (spine → slot order).
        let (child_ds, child_gs) = {
            let grp = self.grp(gidx);
            let mut g = grp.lock();
            if g.deleted {
                return;
            }
            g.deleted = true;
            (g.child_datasets.clone(), g.child_groups.clone())
        };
        gs_out.push(gidx);
        for di in child_ds {
            let ds = self.ds(di);
            let mut d = ds.lock();
            if !d.deleted {
                d.deleted = true;
                ds_out.push(di);
            }
        }
        for gi in child_gs {
            self.delete_group_recursive(gi, ds_out, gs_out);
        }
    }

    /// Free everything a soft-deleted dataset owned. The single owner of
    /// delete-time reclamation, called only from the two delete paths with
    /// the dataset already marked deleted and its op lock held.
    ///
    /// A deleted dataset contributes nothing to finalize (the header,
    /// index-flush and append-flush loops all skip it), so nothing in the
    /// finalized file can reference the blocks freed here. Never runs under
    /// SWMR — the delete entry points refuse first.
    fn release_dataset_storage(&self, index: usize) -> IoResult<()> {
        use crate::format::messages::datatype::DatatypeMessage;
        let (indexed, ndims, contiguous, is_vlen, attrs, header_block) = {
            let ds = self.ds(index);
            let mut m = ds.lock();
            // Buffered rows were never written to a chunk; they die with
            // the dataset instead of being flushed at close.
            m.append = None;
            let indexed = m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some();
            let contiguous = (!indexed && m.data_addr != UNDEF_ADDR && m.data_size > 0)
                .then_some((m.data_addr, m.data_size));
            m.data_addr = UNDEF_ADDR;
            m.data_size = 0;
            let is_vlen = matches!(
                m.datatype,
                DatatypeMessage::VarLenString { .. } | DatatypeMessage::VarLenSequence { .. }
            );
            let attrs = std::mem::take(&mut m.attributes);
            let header_block = m
                .obj_header_written_addr
                .take()
                .filter(|_| m.obj_header_encoded_size > 0)
                .map(|a| (a, m.obj_header_encoded_size as u64));
            m.obj_header_encoded_size = 0;
            (
                indexed,
                m.dataspace.dims.len(),
                contiguous,
                is_vlen,
                attrs,
                header_block,
            )
        };
        if indexed {
            // Prune to a zero extent: every stored chunk is entirely beyond
            // it, so the walk frees each chunk block and collects the vlen
            // references its bytes held (released inside).
            self.prune_chunks_beyond(index, &vec![0; ndims])?;
            self.free_chunk_index(index)?;
        } else if let Some((addr, size)) = contiguous {
            if is_vlen {
                let data = self.handle.read_at(addr, size as usize)?;
                self.release_vlen_references(&data)?;
            }
            self.allocator.free(addr, size);
        }
        for attr in &attrs {
            self.release_attr_vlen(attr)?;
        }
        self.release_superseded_dense_attrs(AttrScope::Dataset(index))?;
        if let Some((addr, size)) = header_block {
            self.allocator.free(addr, size);
        }
        Ok(())
    }

    /// Free a deleted group's file space: its attributes' global-heap
    /// objects and, on a reopened file, the on-disk header block. The
    /// group counterpart of
    /// [`release_dataset_storage`](Self::release_dataset_storage).
    fn release_group_storage(&self, gidx: usize) -> IoResult<()> {
        let (attrs, header_block) = {
            let grp = self.grp(gidx);
            let mut g = grp.lock();
            let attrs = std::mem::take(&mut g.attributes);
            let header_block = g
                .obj_header_written_addr
                .take()
                .filter(|_| g.obj_header_encoded_size > 0)
                .map(|a| (a, g.obj_header_encoded_size as u64));
            g.obj_header_encoded_size = 0;
            (attrs, header_block)
        };
        for attr in &attrs {
            self.release_attr_vlen(attr)?;
        }
        self.release_superseded_dense_attrs(AttrScope::Group(gidx))?;
        self.release_superseded_dense_links(LinkScope::Group(gidx))?;
        if let Some((addr, size)) = header_block {
            self.allocator.free(addr, size);
        }
        Ok(())
    }

    /// Free the dense attribute storage a reopened header names, once, when
    /// this session stops naming it — because the header is being rewritten
    /// around fresh storage, or because the object was deleted.
    ///
    /// The single owner of that transition: nothing else removes an attribute
    /// entry from [`superseded_dense`](Self::superseded_dense), and this
    /// removes it as it frees, so no heap is freed twice or left half freed.
    /// An object whose storage was compact, or whose header this session
    /// keeps, has no entry and nothing happens.
    ///
    /// Never under SWMR: a live reader may still be walking the storage the
    /// published headers name, the same rule the superseded-header and
    /// relocated-chunk paths follow. The entry stays in place, unfreed.
    fn release_superseded_dense_attrs(&self, scope: AttrScope) -> IoResult<()> {
        if self.swmr_active {
            return Ok(());
        }
        let taken = self
            .superseded_dense
            .lock()
            .as_mut()
            .and_then(|s| s.attrs.remove(&scope));
        let Some(ainfo) = taken else {
            return Ok(());
        };
        self.release_dense_storage(
            ainfo.fractal_heap_address,
            ainfo.name_btree_address,
            ainfo.creation_order_btree_address,
        )
    }

    /// The link counterpart of
    /// [`release_superseded_dense_attrs`](Self::release_superseded_dense_attrs),
    /// under the same invariant and the same SWMR rule. Split from it because
    /// the two are superseded at different points of a finalize: attribute
    /// storage before the object headers are laid out, link storage after
    /// every one of them has an address.
    fn release_superseded_dense_links(&self, scope: LinkScope) -> IoResult<()> {
        if self.swmr_active {
            return Ok(());
        }
        let taken = self
            .superseded_dense
            .lock()
            .as_mut()
            .and_then(|s| s.links.remove(&scope));
        let Some(linfo) = taken else {
            return Ok(());
        };
        self.release_dense_storage(
            linfo.fractal_heap_address,
            linfo.name_btree_address,
            linfo.creation_order_btree_address,
        )
    }

    /// Return one dense storage's file space to the allocator: the fractal
    /// heap in full, its name index, and the creation-order index when the
    /// object had one.
    ///
    /// The extents come from walking the structures themselves rather than
    /// from re-deriving what a writer would have allocated, so storage
    /// libhdf5 laid out is freed as accurately as storage this crate wrote.
    /// Every walk here already ran once this session — the reopen read every
    /// attribute out of this heap through the same index — so a failure means
    /// the file changed underneath us, and surfacing it beats freeing a
    /// partial extent list.
    fn release_dense_storage(
        &self,
        heap_addr: u64,
        name_bt2_addr: u64,
        corder_bt2_addr: Option<u64>,
    ) -> IoResult<()> {
        use crate::format::chunk_index::btree_v2::collect_btree_v2_extents;
        use crate::format::fractal_heap::collect_heap_extents;

        let mut reader = crate::io::reader::HandleBlockReader {
            handle: &self.handle,
        };
        let mut extents = Vec::new();
        if heap_addr != UNDEF_ADDR {
            extents.extend(collect_heap_extents(heap_addr, &self.ctx, &mut reader)?);
        }
        for addr in [Some(name_bt2_addr), corder_bt2_addr]
            .into_iter()
            .flatten()
            .filter(|&a| a != UNDEF_ADDR)
        {
            extents.extend(collect_btree_v2_extents(addr, &self.ctx, &mut reader)?);
        }
        for (addr, len) in extents {
            self.allocator.free(addr, len);
        }
        Ok(())
    }

    /// Free a deleted dataset's chunk-index structures, after the chunks
    /// themselves were freed by a zero-extent prune. Takes the index info
    /// out of the slot, so the dataset no longer claims chunked storage.
    ///
    /// Every block's size is recovered the way its allocation computed it:
    /// re-encoding the in-memory copy (EA header and index block, FA
    /// header and data block, BT2 header) or sizing a same-shape dummy
    /// from the array geometry (EA data blocks, whose element counts come
    /// from [`EaGeometry`]; BT2 nodes are all `node_size`).
    fn free_chunk_index(&self, index: usize) -> IoResult<()> {
        let ds = self.ds(index);
        let mut m = ds.lock();
        let is_filtered = m.filter_pipeline.is_some();
        if let Some(c) = m.chunked.take() {
            let p = &c.earray_params;
            let bits = p.max_nelmts_bits;
            let csl = c.chunk_size_len;
            let geo = EaGeometry::new(
                p.idx_blk_elmts,
                p.data_blk_min_elmts,
                p.sup_blk_min_data_ptrs,
                bits,
                p.max_dblk_page_nelmts_bits,
            )?;
            let dblk_size = |nelmts: u64| -> u64 {
                if is_filtered {
                    FilteredDataBlock::new(c.ea_header_addr, 0, nelmts as usize)
                        .encode(&self.ctx, bits, csl)
                        .len() as u64
                } else {
                    ExtensibleArrayDataBlock::new(c.ea_header_addr, 0, nelmts as usize)
                        .encoded_size(&self.ctx, bits) as u64
                }
            };
            let (dblk_addrs, sblk_addrs, iblk_size) = if is_filtered {
                let f = c.filt_iblk.as_ref().unwrap();
                (
                    f.dblk_addrs.clone(),
                    f.sblk_addrs.clone(),
                    f.encode(&self.ctx, csl).len() as u64,
                )
            } else {
                (
                    c.ea_iblk.dblk_addrs.clone(),
                    c.ea_iblk.sblk_addrs.clone(),
                    c.ea_iblk.encoded_size(&self.ctx) as u64,
                )
            };
            // Data blocks addressed from the index block belong to the
            // first `iblock_nsblks` super blocks; each of those defines the
            // element count (and so the disk size) of its data blocks.
            let mut g = 0usize;
            'direct: for s in geo.sblk.iter().take(geo.iblock_nsblks) {
                for _ in 0..s.ndblks {
                    let Some(&a) = dblk_addrs.get(g) else {
                        break 'direct;
                    };
                    g += 1;
                    if a == UNDEF_ADDR {
                        continue;
                    }
                    if s.dblk_nelmts > geo.dblk_page_nelmts {
                        return Err(crate::io::IoError::InvalidState(
                            "cannot free a paged extensible-array data block, \
                             which is not yet supported"
                                .into(),
                        ));
                    }
                    self.allocator.free(a, dblk_size(s.dblk_nelmts));
                }
            }
            for (off, &sa) in sblk_addrs.iter().enumerate() {
                if sa == UNDEF_ADDR {
                    continue;
                }
                let s = geo.sblk[geo.iblock_nsblks + off];
                if s.dblk_nelmts > geo.dblk_page_nelmts {
                    return Err(crate::io::IoError::InvalidState(
                        "cannot free a paged extensible-array data block, \
                         which is not yet supported"
                            .into(),
                    ));
                }
                let buf = self.handle.read_at_most(sa, 65536)?;
                let sb =
                    ExtensibleArraySuperBlock::decode(&buf, &self.ctx, bits, s.ndblks as usize, 0)?;
                for &da in &sb.dblk_addrs {
                    if da != UNDEF_ADDR {
                        self.allocator.free(da, dblk_size(s.dblk_nelmts));
                    }
                }
                self.allocator
                    .free(sa, sb.encode(&self.ctx, bits).len() as u64);
            }
            self.allocator.free(c.ea_iblk_addr, iblk_size);
            self.allocator
                .free(c.ea_header_addr, c.ea_header.encoded_size(&self.ctx) as u64);
            return Ok(());
        }
        if let Some(fa) = m.fixed_array.take() {
            self.allocator.free(
                fa.fa_dblk_addr,
                fixed_array_dblk_disk_size(&self.ctx, &fa.fa_header),
            );
            self.allocator.free(
                fa.fa_header_addr,
                fa.fa_header.encode(&self.ctx).len() as u64,
            );
            return Ok(());
        }
        if let Some(bt2) = m.btree_v2.take() {
            let tree = bt2.index.build_tree(&self.ctx);
            for &a in &bt2.node_addrs {
                self.allocator.free(a, tree.node_size as u64);
            }
            self.allocator.free(
                bt2.bt2_header_addr,
                tree.header(UNDEF_ADDR).encode(&self.ctx).len() as u64,
            );
        }
        Ok(())
    }

    /// Return the chunk dimensions for a dataset, if chunked.
    ///
    /// Returns an owned `Vec` because the chunk geometry now lives behind the
    /// per-dataset [`Slot`]; it cannot be borrowed past the guard.
    pub fn dataset_chunk_dims(&self, index: usize) -> Option<Vec<u64>> {
        let ds = self.ds(index);
        let m = ds.lock();
        if let Some(ref c) = m.chunked {
            Some(c.chunk_dims.clone())
        } else if let Some(ref f) = m.fixed_array {
            Some(f.chunk_dims.clone())
        } else {
            m.btree_v2.as_ref().map(|b| b.chunk_dims.clone())
        }
    }

    /// Return the current dimensions of a dataset.
    ///
    /// Returns an owned `Vec` because the dataspace now lives behind the
    /// per-dataset [`Slot`]; it cannot be borrowed past the guard.
    pub fn dataset_dims(&self, index: usize) -> Vec<u64> {
        self.ds(index).lock().dataspace.dims.clone()
    }

    /// Return the datatype a dataset declares on disk.
    ///
    /// The typed write paths need it to store bytes in the declared byte
    /// order; a reopened dataset handle has no copy of its own, and a cached
    /// one could disagree with what the header will say.
    pub fn dataset_datatype(&self, index: usize) -> DatatypeMessage {
        self.ds(index).lock().datatype.clone()
    }

    /// Return the names of all groups created so far.
    pub fn group_names(&self) -> Vec<String> {
        self.group_refs()
            .iter()
            .map(|g| g.lock().name.clone())
            .collect()
    }

    /// Create a group in the file hierarchy.
    ///
    /// `parent_path` is the full path of the parent group (e.g., "/" for root).
    /// `name` is the name of the new group (e.g., "detector").
    ///
    /// Returns the group index in the writer's group list.
    pub fn create_group(&self, parent_path: &str, name: &str) -> IoResult<usize> {
        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
        // A parent path through hard links creates in the link's target,
        // as HDF5 traversal does.
        let parent_path = self.canonical_group_path(parent_path);
        let parent_path = parent_path.as_str();
        let full_name = if parent_path == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", parent_path, name)
        };
        // Same rule as dataset creation: a path through a carried external
        // link names a group in the other file, which this writer cannot make.
        self.reject_external_traversal(&full_name)?;
        // `name` may itself carry path components; resolving the whole thing
        // is what keeps a '/' out of the link this group will be reached by.
        let (parent_idx, _leaf) = self.split_parent(full_name.trim_start_matches('/'))?;

        let groups = self.group_refs();
        // Check for duplicates (ignore deleted groups)
        let dup = groups.iter().any(|g| {
            let gg = g.lock();
            gg.name == full_name && !gg.deleted
        });
        if dup {
            return Err(crate::io::IoError::InvalidState(format!(
                "group '{}' already exists",
                full_name
            )));
        }
        // A hard link must not already occupy this name in its parent.
        let full_rel = full_name.trim_start_matches('/');
        if self
            .hard_links_vec()
            .iter()
            .any(|l| self.hard_link_emitted(l) && self.hard_link_full_path(l) == full_rel)
        {
            return Err(crate::io::IoError::InvalidState(format!(
                "a hard link named '{full_name}' already exists"
            )));
        }

        let group_idx = self.push_group(GroupInfo {
            name: full_name,
            parent: parent_idx,
            creation_seq: self.take_creation_seq(),
            track_order: self.track_order,
            times: None,
            child_datasets: Vec::new(),
            child_groups: Vec::new(),
            obj_header_addr: 0,
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            deleted: false,
            attributes: Vec::new(),
        });

        // Register this group as a child of its parent
        if let Some(pidx) = parent_idx {
            self.grp(pidx).lock().child_groups.push(group_idx);
        }

        Ok(group_idx)
    }

    /// Register a dataset as belonging to a group.
    ///
    /// `group_path` is the full path of the group (e.g., "/detector").
    /// `ds_index` is the dataset index returned by `create_dataset`.
    pub fn assign_dataset_to_group(&self, group_path: &str, ds_index: usize) -> IoResult<()> {
        let group_path = self.canonical_group_path(group_path);
        let group_path = group_path.as_str();
        let groups = self.group_refs();
        let group_idx = groups
            .iter()
            .position(|g| {
                let gg = g.lock();
                gg.name == group_path && !gg.deleted
            })
            .ok_or_else(|| {
                crate::io::IoError::NotFound(format!("group '{}' not found", group_path))
            })?;
        // A move, not an addition: the create gate has already placed every
        // dataset from the path components of its name, so appending here
        // would leave one dataset linked from two groups at once.
        for g in &groups {
            g.lock().child_datasets.retain(|&d| d != ds_index);
        }
        groups[group_idx].lock().child_datasets.push(ds_index);
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
        &self,
        parent_group_path: &str,
        link_name: &str,
        target_path: &str,
    ) -> IoResult<()> {
        if link_name.is_empty() || link_name.contains('/') {
            return Err(crate::io::IoError::InvalidState(format!(
                "hard link name '{link_name}' must be a non-empty leaf name"
            )));
        }

        // Neither end may sit across a carried external link: the target
        // would be an object in the other file, and the link itself would be
        // a name in a group this writer does not own.
        self.reject_external_traversal(target_path)?;
        self.reject_external_traversal(&format!(
            "{}/{link_name}",
            parent_group_path.trim_end_matches('/')
        ))?;

        // Hold the create gate across the collision check and the hard-link
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
        // Both paths resolve through hard links, as HDF5 traversal does.
        let parent_group_path = self.canonical_group_path(parent_group_path);
        let parent_group_path = parent_group_path.as_str();

        let datasets = self.dataset_refs();
        let groups = self.group_refs();

        // Resolve the parent group (None == root).
        let parent = if parent_group_path == "/" {
            None
        } else {
            Some(
                groups
                    .iter()
                    .position(|g| {
                        let gg = g.lock();
                        gg.name == parent_group_path && !gg.deleted
                    })
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
        let target_rel = self.canonical_dataset_path(target_path.trim_matches('/'));
        let target_rel = target_rel.as_str();
        if target_rel.is_empty() {
            return Err(crate::io::IoError::InvalidState(
                "cannot hard-link the root group".into(),
            ));
        }
        let target = self.resolve_object(target_rel).ok_or_else(|| {
            crate::io::IoError::NotFound(format!("hard link target '{target_path}' not found"))
        })?;

        // Reject a name already taken in the parent group.
        let parent_prefix = match parent {
            None => String::new(),
            Some(pi) => format!("{}/", groups[pi].lock().name.trim_start_matches('/')),
        };
        let full = format!("{parent_prefix}{link_name}");
        let collides = datasets.iter().any(|d| {
            let g = d.lock();
            !g.deleted && g.name.trim_start_matches('/') == full
        }) || groups.iter().any(|g| {
            let gg = g.lock();
            !gg.deleted && gg.name.trim_start_matches('/') == full
        }) || self
            .hard_links_vec()
            .iter()
            .any(|l| l.parent == parent && l.name == link_name);
        if collides {
            return Err(crate::io::IoError::InvalidState(format!(
                "'{full}' already exists in the file"
            )));
        }

        self.hard_links.lock().push(HardLink {
            parent,
            name: link_name.to_string(),
            target,
            creation_seq: self.take_creation_seq(),
        });
        Ok(())
    }

    /// Whether a hard link will actually be emitted: both its parent group
    /// and its target object must still be present (not soft-deleted).
    fn hard_link_emitted(&self, link: &HardLink) -> bool {
        let parent_ok = match link.parent {
            None => true,
            Some(pi) => !self.grp(pi).lock().deleted,
        };
        let target_ok = match link.target {
            HardLinkTarget::Dataset(i) => !self.ds(i).lock().deleted,
            HardLinkTarget::Group(i) => !self.grp(i).lock().deleted,
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
                self.grp(pi).lock().name.trim_start_matches('/'),
                link.name
            ),
        }
    }

    /// Rewrite a group path that passes through hard links into the tree
    /// path of the group it reaches — HDF5 traversal, where any link in a
    /// path component resolves to its target. Group-name form (leading
    /// `/`). Repeats because a substituted target's subtree can hold
    /// further links; bounded like libhdf5's link-traversal limit, so a
    /// link cycle cannot loop forever. A path with no link components
    /// (including one naming nothing at all) comes back unchanged.
    pub(crate) fn canonical_group_path(&self, path: &str) -> String {
        let mut path = path.to_string();
        for _ in 0..64 {
            // The longest emitted group-link path that is the whole of
            // `path` or a '/'-boundary prefix of it.
            let mut best: Option<(usize, usize)> = None; // (prefix len, target)
            for l in self.hard_links_vec() {
                let HardLinkTarget::Group(gi) = l.target else {
                    continue;
                };
                if !self.hard_link_emitted(&l) {
                    continue;
                }
                let lp = format!("/{}", self.hard_link_full_path(&l));
                let covers = path == lp || path.starts_with(&format!("{lp}/"));
                if covers && best.is_none_or(|(len, _)| lp.len() > len) {
                    best = Some((lp.len(), gi));
                }
            }
            let Some((len, gi)) = best else { break };
            let target_name = self.grp(gi).lock().name.clone();
            path = format!("{}{}", target_name, &path[len..]);
        }
        path
    }

    /// [`canonical_group_path`](Self::canonical_group_path) in the
    /// dataset-name form (no leading `/`): the leaf is a dataset, so only
    /// group links can appear as components and the whole path can go
    /// through the group rewrite unchanged.
    fn canonical_dataset_path(&self, name: &str) -> String {
        self.canonical_group_path(&format!("/{name}"))
            .trim_start_matches('/')
            .to_string()
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
            .hard_links_vec()
            .iter()
            .filter(|l| self.hard_link_emitted(l) && same(l.target, target))
            .count() as u32
    }

    /// The object a path names, or `None` when nothing in the file does.
    ///
    /// `path` is the trimmed, hard-link-canonical form (no leading or
    /// trailing `/`) that dataset and group names compare against. The single
    /// owner of path→object resolution on the write side: hard links and
    /// object references must agree on what a path means, including that a
    /// path may itself be a user hard link — links have no chain (each points
    /// straight at the object header, as in libhdf5), so the existing link's
    /// target is the answer.
    pub(crate) fn resolve_object(&self, path: &str) -> Option<HardLinkTarget> {
        if let Some(idx) = self.dataset_refs().iter().position(|d| {
            let g = d.lock();
            !g.deleted && g.name.trim_start_matches('/') == path
        }) {
            return Some(HardLinkTarget::Dataset(idx));
        }
        if let Some(idx) = self.group_refs().iter().position(|g| {
            let gg = g.lock();
            !gg.deleted && gg.name.trim_start_matches('/') == path
        }) {
            return Some(HardLinkTarget::Group(idx));
        }
        self.hard_links_vec().iter().find_map(|l| {
            (self.hard_link_emitted(l) && self.hard_link_full_path(l) == path).then_some(l.target)
        })
    }

    /// Store object references naming `paths` into the elements of dataset
    /// `index` starting at `start`.
    ///
    /// The value of an `H5R_OBJECT1` element is its target's object header
    /// address, which finalize assigns, so what lands here is the target path;
    /// [`Self::apply_object_reference_fixups`] writes the addresses. Elements
    /// never written keep the zero image libhdf5 reads back as a null
    /// reference.
    pub fn write_object_references(
        &self,
        index: usize,
        start: u64,
        paths: &[&str],
    ) -> IoResult<()> {
        let (elements, data_addr, indexed) = {
            let ds = self.ds(index);
            let m = ds.lock();
            match &m.datatype {
                DatatypeMessage::Reference {
                    kind: ReferenceKind::Object1,
                    ..
                } => {}
                other => {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "dataset '{}' has datatype {other}, not an object reference",
                        m.name
                    )))
                }
            }
            let elements = m
                .dataspace
                .dims
                .iter()
                .fold(1u64, |a, &d| a.saturating_mul(d));
            let indexed = m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some();
            (elements, m.data_addr, indexed)
        };
        if indexed || data_addr == UNDEF_ADDR {
            return Err(crate::io::IoError::InvalidState(
                "object references are stamped into contiguous storage; \
                 create the dataset without chunking"
                    .into(),
            ));
        }
        let end = start.saturating_add(paths.len() as u64);
        if end > elements {
            return Err(crate::io::IoError::InvalidState(format!(
                "elements {start}..{end} are outside the dataset's {elements}"
            )));
        }
        // Resolve now as well as at fixup time, so a path that names nothing
        // is reported at the call that got it wrong.
        for path in paths {
            self.object_reference_target(path)?;
        }
        let mut pending = self.pending_object_references.lock();
        for (i, path) in paths.iter().enumerate() {
            pending.push(PendingObjectReference {
                dataset: index,
                element: start + i as u64,
                target: (*path).to_string(),
            });
        }
        Ok(())
    }

    /// Record a hard link count of `rc` in `header`, if this file's format
    /// needs a message to carry it.
    ///
    /// A version-2 header carries the count in an Object Reference Count
    /// message, and only when more than one link reaches the object. A
    /// version-1 header carries it in its prefix and gets no message at all —
    /// `H5O_link_oh` gates every refcount-message operation on
    /// `oh->version > H5O_VERSION_1` (H5Oint.c:876), so a version-1 header
    /// holding one is a shape libhdf5 never writes.
    fn emit_refcount(&self, header: &mut ObjectHeader, rc: u32, format: ObjectFormat) {
        if rc > 1 && format == ObjectFormat::Modern {
            header.add_message(MSG_OBJ_REF_COUNT, 0x00, encode_refcount(rc));
        }
    }

    /// Encode an object header at the version this file's format calls for,
    /// with `rc` as the object's hard link count.
    ///
    /// The count is passed rather than read off the header because the two
    /// versions carry it in different places — the version-1 prefix's `nlink`
    /// field, the version-2 Reference Count message
    /// [`emit_refcount`](Self::emit_refcount) already added — and only the
    /// caller knows it.
    fn encode_header(
        &self,
        header: &ObjectHeader,
        rc: u32,
        format: ObjectFormat,
    ) -> IoResult<Vec<u8>> {
        Ok(header.encode_for(format, rc)?)
    }

    /// The object an object reference's path names, as a hard-link target;
    /// `None` for the root group, which has no registry slot.
    fn object_reference_target(&self, path: &str) -> IoResult<Option<HardLinkTarget>> {
        let rel = self.canonical_dataset_path(path.trim_matches('/'));
        if rel.is_empty() {
            return Ok(None);
        }
        self.resolve_object(&rel)
            .map(Some)
            .ok_or_else(|| crate::io::IoError::NotFound(format!("reference target '{path}'")))
    }

    /// Stamp every pending object reference with its target's object header
    /// address.
    ///
    /// INVARIANT: a reference element on disk holds its target's header
    /// address. Both finalize paths call this once every dataset, group and
    /// root header has an address and before the superblock is written, so no
    /// file is closed with a placeholder element in it; a target that no
    /// longer resolves fails the finalize rather than leaving one behind.
    fn apply_object_reference_fixups(&mut self) -> IoResult<()> {
        // Snapshot rather than drain: a SWMR session finalizes twice, and the
        // close-time finalize rebuilds every header at a fresh address, so the
        // elements must be stamped again with the addresses that survive.
        let pending: Vec<(usize, u64, String)> = self
            .pending_object_references
            .lock()
            .iter()
            .map(|p| (p.dataset, p.element, p.target.clone()))
            .collect();
        let width = self.ctx.sizeof_addr as usize;
        for (dataset, element, target) in &pending {
            let addr = match self.object_reference_target(target)? {
                Some(HardLinkTarget::Dataset(i)) => self.ds(i).lock().obj_header_addr,
                Some(HardLinkTarget::Group(i)) => self.grp(i).lock().obj_header_addr,
                None => self.root_group_addr.ok_or_else(|| {
                    crate::io::IoError::InvalidState(
                        "root group header address is not assigned yet".into(),
                    )
                })?,
            };
            let data_addr = self.ds(*dataset).lock().data_addr;
            let at = data_addr + element * width as u64;
            self.handle.write_at(at, &addr.to_le_bytes()[..width])?;
        }
        Ok(())
    }

    /// Append every user-created hard link whose parent group is `parent`
    /// (`None` == the root group). Called while collecting a group's links,
    /// once every object's header address has been assigned.
    fn push_hard_links(&self, links: &mut Vec<(u64, LinkMessage)>, parent: Option<usize>) {
        for link in self.hard_links_vec() {
            if link.parent != parent || !self.hard_link_emitted(&link) {
                continue;
            }
            let addr = match link.target {
                HardLinkTarget::Dataset(i) => self.ds(i).lock().obj_header_addr,
                HardLinkTarget::Group(i) => self.grp(i).lock().obj_header_addr,
            };
            links.push((link.creation_seq, LinkMessage::hard(&link.name, addr)));
        }
    }

    /// Refuse a caller path that would have to leave this file through one of
    /// the external links a reopened file brought in.
    ///
    /// The reader follows such a path into the file the link names; the writer
    /// cannot, because it models one file and would have to write into
    /// another. Saying which link stops the path — rather than reporting the
    /// name as absent, or worse, creating a second link of that name beside
    /// it — is the whole of what write mode does here.
    pub(crate) fn reject_external_traversal(&self, path: &str) -> IoResult<()> {
        let path = path.trim_start_matches('/');
        let crossing = self.preserved_link_paths().into_iter().find(|(p, class)| {
            matches!(class, crate::io::reader::LinkClass::External { .. })
                && (path == p || path.starts_with(&format!("{p}/")))
        });
        match crossing {
            None => Ok(()),
            Some((link, crate::io::reader::LinkClass::External { file, path: target })) => {
                Err(crate::io::IoError::Unsupported(format!(
                    "'{path}' resolves through the external link '{link}' to '{target}' in \
                     '{file}'; this writer carries external links through a rewrite but does \
                     not open the file they name"
                )))
            }
            // `find` matched on the External arm, so no other class reaches here.
            Some(_) => Ok(()),
        }
    }

    /// Resolve `name` to a live dataset index, reporting *why* it does not
    /// resolve rather than collapsing every cause into absence.
    ///
    /// The write-mode counterpart of [`Hdf5Reader::open_dataset`]: the single
    /// gate every by-name dataset lookup in write mode goes through.
    ///
    /// [`Hdf5Reader::open_dataset`]: crate::io::reader::Hdf5Reader::open_dataset
    pub(crate) fn open_dataset_index(&self, name: &str) -> IoResult<usize> {
        self.reject_external_traversal(name)?;
        self.reject_preserved_object(name)?;
        self.dataset_index(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))
    }

    /// Refuse a caller path that names an object the reopen kept by its bytes
    /// rather than modelling.
    ///
    /// Such an object is in the file and stays in it, but this writer holds
    /// none of what it would need to read or rewrite it. Saying so — with the
    /// reason the classification recorded — is the difference between an
    /// object the writer will not touch and a name the file does not have.
    pub(crate) fn reject_preserved_object(&self, path: &str) -> IoResult<()> {
        let path = path.trim_start_matches('/');
        let objects: Vec<(String, String)> = {
            let preserved = self.preserved_links.lock();
            preserved
                .iter()
                .filter_map(|l| {
                    l.reason
                        .as_ref()
                        .map(|why| (self.preserved_link_full_path(l), why.clone()))
                })
                .collect()
        };
        match objects
            .into_iter()
            .find(|(full, _)| path == full || path.starts_with(&format!("{full}/")))
        {
            None => Ok(()),
            Some((link, why)) => Err(crate::io::IoError::Unsupported(format!(
                "'{path}' is, or is inside, the object '{link}', which this file's reopen \
                 kept exactly as it found it because {why}"
            ))),
        }
    }

    /// Every link this writer is carrying but cannot express, by full path.
    pub(crate) fn preserved_link_paths(&self) -> Vec<(String, crate::io::reader::LinkClass)> {
        self.preserved_links
            .lock()
            .iter()
            .map(|l| (self.preserved_link_full_path(l), l.class.clone()))
            .collect()
    }

    /// The full path of a preserved link: its parent group's path plus its
    /// leaf name, in the no-leading-`/` form the registry uses.
    fn preserved_link_full_path(&self, link: &PreservedLink) -> String {
        match link.parent {
            None => link.name.clone(),
            Some(gi) => {
                let group = self.grp(gi).lock().name.clone();
                format!("{}/{}", group.trim_start_matches('/'), link.name)
            }
        }
    }

    /// The single owner of "which links does this group hold", in the order
    /// they were created and, when the file tracks creation order, stamped
    /// with it.
    ///
    /// Both the compact form (one `MSG_LINK` per link) and the dense form (the
    /// same messages inside a fractal heap) are built from this one list, so
    /// the phase-change decision, the storage it selects and the creation
    /// order recorded in either can never disagree about what the group
    /// contains.
    fn group_links(&self, scope: LinkScope, order: CreationOrder) -> Vec<LinkMessage> {
        let mut links: Vec<(u64, LinkMessage)> = Vec::new();
        match scope {
            LinkScope::Root => {
                // Datasets that belong to a subgroup are that group's links,
                // not the root's. Each group slot is locked one at a time.
                let mut datasets_in_subgroups: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for grp in self.group_refs() {
                    let g = grp.lock();
                    if g.deleted {
                        continue;
                    }
                    datasets_in_subgroups.extend(g.child_datasets.iter().copied());
                }
                // `dataset_refs` preserves registry order, so `enumerate`
                // yields each dataset's true index.
                for (i, ds) in self.dataset_refs().into_iter().enumerate() {
                    let m = ds.lock();
                    if m.deleted || datasets_in_subgroups.contains(&i) {
                        continue;
                    }
                    // The leaf, never the registry path: a link name is one
                    // path component, and `H5G_traverse` would split a '/'
                    // in it before `H5L_link` ever saw the name.
                    let leaf_name = m.name.rsplit('/').next().unwrap_or(&m.name);
                    links.push((
                        m.creation_seq,
                        LinkMessage::hard(leaf_name, m.obj_header_addr),
                    ));
                }
                for grp in self.group_refs() {
                    let g = grp.lock();
                    if g.deleted || g.parent.is_some() {
                        continue;
                    }
                    let leaf_name = g.name.rsplit('/').next().unwrap_or(&g.name);
                    links.push((
                        g.creation_seq,
                        LinkMessage::hard(leaf_name, g.obj_header_addr),
                    ));
                }
                self.push_hard_links(&mut links, None);
            }
            LinkScope::Group(group_idx) => {
                // Snapshot the child lists, then drop the slot guard: the
                // per-child reads below re-lock dataset and group slots
                // (including this one).
                let (child_datasets, child_groups) = {
                    let grp = self.grp(group_idx);
                    let g = grp.lock();
                    (g.child_datasets.clone(), g.child_groups.clone())
                };
                for ds_idx in child_datasets {
                    let ds = self.ds(ds_idx);
                    let m = ds.lock();
                    if m.deleted {
                        continue;
                    }
                    let leaf_name = m.name.rsplit('/').next().unwrap_or(&m.name);
                    links.push((
                        m.creation_seq,
                        LinkMessage::hard(leaf_name, m.obj_header_addr),
                    ));
                }
                for child_idx in child_groups {
                    let child_grp = self.grp(child_idx);
                    let g = child_grp.lock();
                    if g.deleted {
                        continue;
                    }
                    let leaf_name = g.name.rsplit('/').next().unwrap_or(&g.name);
                    links.push((
                        g.creation_seq,
                        LinkMessage::hard(leaf_name, g.obj_header_addr),
                    ));
                }
                self.push_hard_links(&mut links, Some(group_idx));
            }
        }
        // Creation order, not order by kind: a run of create_group and
        // create_dataset draws from one counter, so this is the order the
        // caller made them in. `H5G_obj_insert` numbers from zero within the
        // group, so the rank here is the link's creation order.
        links.sort_by_key(|(seq, _)| *seq);
        links
            .into_iter()
            .enumerate()
            .map(|(rank, (_, link))| {
                if order.is_tracked() {
                    link.with_creation_order(rank as i64)
                } else {
                    link
                }
            })
            .collect()
    }

    /// Whether `links` must live in dense storage rather than in the group's
    /// object header — the `H5G_obj_insert` phase-change rule, applied to the
    /// whole set at once because this writer builds each header from scratch
    /// rather than inserting one link at a time.
    ///
    /// libhdf5 converts when the count *reaches* `max_compact` and another
    /// link arrives, so a set of exactly `max_compact` is still compact; and
    /// separately when one message would not fit the 16-bit size field an
    /// object header message has.
    ///
    /// The answer depends only on the link names and kinds, never on the
    /// addresses they point at, which is what lets a group header be sized
    /// before [`prepare_dense_links`](Self::prepare_dense_links) has run.
    fn links_need_dense(&self, links: &[LinkMessage]) -> bool {
        links.len() > MAX_COMPACT_LINKS
            || links
                .iter()
                .any(|l| l.encode(&self.ctx).len() > MAX_MESSAGE_SIZE)
    }

    /// The single owner of link emission into a group object header: the Link
    /// Info and Group Info messages, and then either one `MSG_LINK` per link
    /// or nothing at all when the set has spilled to dense storage.
    ///
    /// The two storage forms are exclusive (`H5G_obj_insert` moves the whole
    /// set at once), and a header carrying both would report every link twice.
    ///
    /// A group whose links are dense but not yet laid out gets a compact Link
    /// Info message here. That is deliberate: the message encodes to the same
    /// length either way — two addresses, defined or not — so the sizing pass
    /// that runs before `prepare_dense_links` still reserves the right number
    /// of bytes, and the write pass that runs after it emits the real heap and
    /// index addresses. It is the same two-pass rule the child link addresses
    /// already follow.
    fn emit_links(
        &self,
        header: &mut ObjectHeader,
        scope: LinkScope,
        links: &[LinkMessage],
        order: CreationOrder,
    ) {
        // A classic group holds no link messages at all: its links are the
        // entries of the symbol table `prepare_symbol_tables` laid out, and
        // the header carries only the two addresses naming it. Link Info and
        // Group Info are version-1.8 messages and have no business in a
        // version-1 header — `H5G__stab_valid` reads the Symbol Table message
        // and nothing else.
        if self.uses_symbol_table(order) {
            let legacy = self
                .legacy
                .as_deref()
                .expect("uses_symbol_table is only true in a classic file");
            // Sizing runs before the tables are laid out; the message is the
            // same two addresses wide either way, so the placeholder reserves
            // exactly what the real one needs. Same two-pass rule the child
            // link addresses already follow.
            let stab = legacy.written.lock().get(&scope).copied().unwrap_or(Stab {
                btree_addr: UNDEF_ADDR,
                heap_addr: UNDEF_ADDR,
            });
            header.add_message(MSG_SYMBOL_TABLE, 0x00, stab.encode(&self.ctx));
            return;
        }
        // Links a reopen carried through verbatim because this writer cannot
        // express them. They are emitted here rather than by a second caller
        // so that no header-rewrite path can drop them, and their presence
        // pins the group to compact storage: dense storage would have to
        // re-encode each link into the heap, which is exactly the byte
        // fidelity preserving them is for.
        let preserved = self.preserved_links_for(scope);
        let dense = preserved.is_empty() && self.links_need_dense(links);
        let link_info = self.dense_links.lock().get(&scope).cloned();
        let link_info = link_info.unwrap_or_else(|| {
            let mut info = LinkInfoMessage::compact();
            if order.is_tracked() {
                // `H5G__obj_insert` post-increments `max_corder`, so a group
                // holding n links reports n.
                info.max_creation_order = Some(links.len() as u64);
            }
            if order.is_indexed() {
                // The index address stays undefined while the links live in
                // the header, but the message must still carry the field:
                // `H5Pget_link_creation_order` reads INDEXED off this flag,
                // not off the address.
                info.creation_order_btree_address = Some(UNDEF_ADDR);
            }
            info
        });
        header.add_message(MSG_LINK_INFO, 0x00, link_info.encode(&self.ctx));
        header.add_message(MSG_GROUP_INFO, 0x00, GroupInfoMessage::default().encode());
        if dense {
            return;
        }
        for link in links {
            header.add_message(MSG_LINK, 0x00, link.encode(&self.ctx));
        }
        for encoded in preserved {
            header.add_message(MSG_LINK, 0x00, encoded);
        }
    }

    /// The verbatim link bodies a reopen carried into `scope`.
    fn preserved_links_for(&self, scope: LinkScope) -> Vec<Vec<u8>> {
        let parent = match scope {
            LinkScope::Root => None,
            LinkScope::Group(i) => Some(i),
        };
        self.preserved_links
            .lock()
            .iter()
            .filter(|l| l.parent == parent)
            .map(|l| l.encoded.clone())
            .collect()
    }

    /// Lay out and write dense link storage for every group that needs it,
    /// recording the resulting `Link Info` message per group.
    ///
    /// The sole owner of that transition. It must run after every object
    /// header address is assigned — the heap holds encoded link messages, and
    /// those name their targets — and before any group header is written.
    ///
    /// Every group whose header this finalize rewrites passes through here,
    /// dense or not: the storage a reopened header named is superseded by the
    /// rewrite whichever form the new link set takes, and freeing it first is
    /// what lets the replacement reuse those blocks.
    fn prepare_dense_links(&self) -> IoResult<()> {
        let mut scopes: Vec<(LinkScope, Vec<LinkMessage>, CreationOrder)> = Vec::new();
        for gi in 0..self.group_count() {
            let (deleted, order) = {
                let grp = self.grp(gi);
                let g = grp.lock();
                (g.deleted, g.track_order.links)
            };
            // A symbol-table group is `prepare_symbol_tables`' business; it
            // has no Link Info message to hold a fractal heap address, and it
            // never had dense storage to release.
            if deleted || self.uses_symbol_table(order) {
                continue;
            }
            self.release_superseded_dense_links(LinkScope::Group(gi))?;
            let links = self.group_links(LinkScope::Group(gi), order);
            if self.links_need_dense(&links) {
                scopes.push((LinkScope::Group(gi), links, order));
            }
        }
        let root_order = self.root_track_order.links;
        if !self.uses_symbol_table(root_order) {
            self.release_superseded_dense_links(LinkScope::Root)?;
            let root_links = self.group_links(LinkScope::Root, root_order);
            if self.links_need_dense(&root_links) {
                scopes.push((LinkScope::Root, root_links, root_order));
            }
        }

        for (scope, links, order) in scopes {
            // `close` after `start_swmr` finalizes a second time over the same
            // groups, so rebuilding here would allocate a whole second heap
            // and strand the one the published headers already name.
            if self.dense_links.lock().contains_key(&scope) {
                continue;
            }
            let dense = build_dense_links(&links, &self.ctx, order, &mut |len| {
                self.allocator.allocate(len)
            })?;
            for block in &dense.blocks {
                self.handle.write_at(block.addr, &block.image)?;
            }
            self.dense_links.lock().insert(scope, dense.linfo);
        }
        Ok(())
    }

    /// Lay out whichever of the two forms of link storage this file uses,
    /// before any group header is written.
    ///
    /// The two are exclusive because the formats are: a classic group has no
    /// Link Info message to put a fractal heap address in, and a link-message
    /// group has no symbol table.
    fn prepare_link_storage(&self) -> IoResult<()> {
        self.prepare_dense_links()?;
        self.prepare_symbol_tables()
    }

    /// Lay out and write the symbol table of every classic group, and free the
    /// storage each rewrite supersedes. A no-op on a link-message file.
    ///
    /// The classic counterpart of [`prepare_dense_links`](Self::prepare_dense_links),
    /// and the sole owner of that transition. The same two placement rules
    /// apply for the same two reasons: it runs after every object header has
    /// an address, because a symbol table entry names its target's header, and
    /// before any group header is written, because the header carries the
    /// Symbol Table message naming what this laid out.
    ///
    /// Deepest group first, root last. A hard link to a group caches that
    /// group's own B-tree and heap in the entry's scratch pad
    /// (`H5G__link_to_ent`), so the child's table must exist before the
    /// parent's is built; `H5G__stab_valid` checks the root entry's cache
    /// against the root header's Symbol Table message, so a stale pair there
    /// is not a slow lookup but a file `H5Fopen` rejects.
    ///
    /// Every classic group is rebuilt on every pass — there is no "already
    /// done" short-circuit like the dense one, because the only way this runs
    /// twice is a `Drop` retry after a failed `close`, and the entries of the
    /// first pass name header addresses the second pass has moved. (A SWMR
    /// session, the other double-finalize, cannot reach here: SWMR needs a
    /// version-3 superblock, so `start_swmr` refuses a classic file.)
    fn prepare_symbol_tables(&self) -> IoResult<()> {
        let Some(legacy) = self.legacy.as_deref() else {
            return Ok(());
        };
        // Depth by parent chain, not by counting separators in the registry
        // path: the chain is what actually says which table has to exist first.
        let mut scopes: Vec<(usize, LinkScope, CreationOrder)> = Vec::new();
        for gi in 0..self.group_count() {
            let (deleted, order, mut parent) = {
                let grp = self.grp(gi);
                let g = grp.lock();
                (g.deleted, g.track_order.links, g.parent)
            };
            if deleted || !self.uses_symbol_table(order) {
                continue;
            }
            let mut depth = 1usize;
            while let Some(p) = parent {
                depth += 1;
                parent = self.grp(p).lock().parent;
            }
            scopes.push((depth, LinkScope::Group(gi), order));
        }
        scopes.sort_by_key(|&(depth, ..)| std::cmp::Reverse(depth));
        let root_order = self.root_track_order.links;
        if self.uses_symbol_table(root_order) {
            scopes.push((0, LinkScope::Root, root_order));
        }

        for (_, scope, order) in scopes {
            // Freed before the replacement is laid out, so a rewrite reuses
            // the same blocks instead of growing the file on every open/close
            // cycle — the rule `prepare_dense_links` and the header rewrite
            // already follow. Removed as it is freed, so no second pass can
            // free it twice.
            let superseded = legacy.superseded.lock().remove(&scope);
            if let Some(extents) = superseded {
                free_stab(&self.allocator, &extents);
            }
            let links = self.stab_links_for(scope, order)?;
            let stab = write_stab(&self.handle, &self.allocator, &legacy.meta, &links)?;
            legacy.written.lock().insert(scope, stab);
        }
        Ok(())
    }

    /// `scope`'s links as symbol table entries.
    ///
    /// A link a reopen carried through verbatim is decoded back out of its
    /// encoded Link message here, because a classic group has no link message
    /// to preserve it into. Nothing is lost in the round trip: the walk built
    /// that message from a symbol table entry in the first place, and the two
    /// forms carry the same three facts.
    fn stab_links_for(&self, scope: LinkScope, order: CreationOrder) -> IoResult<Vec<StabLink>> {
        let groups = self.group_header_scopes();
        let mut out = Vec::new();
        for link in self.group_links(scope, order) {
            out.push(self.stab_link(&link, &groups)?);
        }
        for encoded in self.preserved_links_for(scope) {
            let (link, _) = LinkMessage::decode(&encoded, &self.ctx)?;
            out.push(self.stab_link(&link, &groups)?);
        }
        Ok(out)
    }

    /// Where each group's object header now sits, so a hard link that lands on
    /// one can cache that group's symbol table in its scratch pad.
    fn group_header_scopes(&self) -> HashMap<u64, LinkScope> {
        let mut map = HashMap::new();
        for gi in 0..self.group_count() {
            let grp = self.grp(gi);
            let g = grp.lock();
            if !g.deleted {
                map.insert(g.obj_header_addr, LinkScope::Group(gi));
            }
        }
        map
    }

    /// One link as a symbol table entry.
    ///
    /// The scratch pad caches the target group's B-tree and heap when the
    /// target is a group this pass has already laid out — what
    /// `H5G__link_to_ent` does, and what lets `H5G__stab_lookup` walk a path
    /// without opening each header on the way. For anything else the pad stays
    /// `H5G_NOTHING_CACHED`, the value libhdf5 itself writes whenever the
    /// target has no Symbol Table message to read.
    fn stab_link(
        &self,
        link: &LinkMessage,
        groups: &HashMap<u64, LinkScope>,
    ) -> IoResult<StabLink> {
        let target = match &link.target {
            LinkTarget::Hard { address } => {
                let cached = groups
                    .get(address)
                    .and_then(|scope| self.legacy.as_deref()?.written.lock().get(scope).copied());
                StabTarget::Hard {
                    addr: *address,
                    cached,
                }
            }
            LinkTarget::Soft { target } => StabTarget::Soft {
                value: target.clone(),
            },
            // A symbol table entry has three cache types and no room for a
            // fourth: `H5L_TYPE_EXTERNAL` and the user-defined classes exist
            // only as link messages, which is why libhdf5 converts a group to
            // the new format before inserting one (`H5G_obj_insert`). This
            // writer does not convert, so it refuses.
            LinkTarget::External { .. } | LinkTarget::UserDefined { .. } => {
                return Err(crate::io::IoError::InvalidState(format!(
                    "cannot store the link {:?} in this file: its groups use \
                     symbol tables, which hold only hard and soft links",
                    link.name
                )))
            }
        };
        Ok(StabLink {
            name: link.name.clone(),
            target,
        })
    }

    /// The single owner of attribute emission into an object header: appends
    /// the Attribute Info message and then one `MSG_ATTRIBUTE` per attribute.
    ///
    /// On a version-2 object header the two are inseparable.
    /// `H5O__attr_count_real` derives `H5Oget_info().num_attrs` from the
    /// Attribute Info message alone — with no such message the count reads as
    /// zero however many attribute messages follow, which is what made every
    /// rust-written file report `num_attrs == 0` to libhdf5 while
    /// `H5Aiterate2` still yielded the attributes. The message carries no
    /// count of its own: `H5A__get_ainfo` fills `nattrs` from the attribute
    /// messages the header loader actually saw, so compact storage needs
    /// nothing but the message's presence.
    ///
    /// When [`prepare_dense_attributes`](Self::prepare_dense_attributes) has
    /// spilled `scope`'s attributes to a fractal heap, the same message names
    /// that heap instead and *no* attribute message follows: the two storage
    /// forms are exclusive (`H5O__attr_create` moves the whole set at once),
    /// and a header carrying both would report every attribute twice.
    fn emit_attributes(
        &self,
        header: &mut ObjectHeader,
        scope: AttrScope,
        attributes: &[AttributeEntry],
        order: CreationOrder,
        format: ObjectFormat,
    ) {
        // `H5Pget_attr_creation_order` reads the object header's own flags,
        // not the Attribute Info message, so this is what makes the object
        // report creation-ordered attributes — and tracking widens every
        // message envelope by the creation index below.
        header.set_attribute_creation_order(order);
        if attributes.is_empty() {
            return;
        }
        // A version-1 object header gets the attribute messages alone.
        // `H5O__attr_create` gates every mention of the Attribute Info message
        // on `oh->version > H5O_VERSION_1` (H5Oattribute.c:218), and so does
        // `H5O__attr_count_real`, which is why the count still reads correctly
        // without it: on a version-1 header libhdf5 counts the messages.
        if format == ObjectFormat::Legacy {
            for attr in attributes {
                header.add_message(MSG_ATTRIBUTE, 0x00, self.encode_attribute(attr));
            }
            return;
        }
        if let Some(ainfo) = self.dense_attributes.lock().get(&scope) {
            header.add_message(MSG_ATTR_INFO, MSG_FLAG_DONTSHARE, ainfo.encode(&self.ctx));
            return;
        }
        let mut ainfo = AttributeInfoMessage::compact();
        if order.is_tracked() {
            ainfo.max_creation_index = Some(next_creation_index(attributes));
        }
        if order.is_indexed() {
            // Compact storage has no index B-tree, but the message still
            // announces one so that its flags match the header's
            // (`H5O__attr_create` asserts they agree).
            ainfo.creation_order_btree_address = Some(UNDEF_ADDR);
        }
        header.add_message(MSG_ATTR_INFO, MSG_FLAG_DONTSHARE, ainfo.encode(&self.ctx));
        // Each attribute states its own creation index — the one it was
        // created with here, or the one the file it was read from records. An
        // attribute with none belongs to an object that tracks no order, where
        // the field is not encoded at all.
        for attr in attributes {
            header.add_message_indexed(
                MSG_ATTRIBUTE,
                0x00,
                self.encode_attribute(attr),
                attr.creation_index().unwrap_or(0),
            );
        }
    }

    /// One attribute message body, at the version this file's low library
    /// bound calls for (`H5A__set_version`, which reads the bound and nothing
    /// about the object the attribute hangs on).
    fn encode_attribute(&self, attr: &AttributeEntry) -> Vec<u8> {
        attr.encode_for(&self.ctx, self.encoding_libver(), self.message_format())
    }

    /// Whether `attributes` must live in dense storage rather than in the
    /// object header — the `H5O__attr_create` phase-change rule, applied to
    /// the whole set at once because this writer builds each header from
    /// scratch rather than inserting one attribute at a time.
    ///
    /// libhdf5 converts when the count *reaches* `max_compact` and another
    /// attribute arrives, so a set of exactly `max_compact` is still compact;
    /// and separately when one message would not fit the 16-bit size field an
    /// object header message has.
    ///
    /// Never in a classic file. Dense attribute storage is a fractal heap
    /// reached through an Attribute Info message, both introduced in the 1.8
    /// format; at `H5F_LIBVER_EARLIEST` libhdf5 keeps every attribute in the
    /// header however many there are (`H5O__attr_create` reaches the phase
    /// change only when the object header version allows it). An attribute
    /// too large for the 16-bit size field is then an error, which
    /// `ObjectHeader::encode_v1` raises, rather than a reason to spill.
    fn attributes_need_dense(&self, attributes: &[AttributeEntry], format: ObjectFormat) -> bool {
        if format == ObjectFormat::Legacy {
            return false;
        }
        attributes.len() > MAX_COMPACT_ATTRS
            || attributes
                .iter()
                .any(|a| self.encode_attribute(a).len() > MAX_MESSAGE_SIZE)
    }

    /// Lay out and write dense attribute storage for every object that needs
    /// it, recording the resulting `Attribute Info` message per object.
    ///
    /// The sole owner of that transition: it runs before any object header is
    /// built, so `emit_attributes` never allocates. Every block is on disk
    /// before the map naming it is populated, so a header written from that
    /// map can only point at bytes that exist.
    ///
    /// `datasets` lists the datasets whose headers this finalize will
    /// actually write. A reopened dataset that took no writes keeps its
    /// original header — and with it whatever storage that header already
    /// names — so building a heap for it would strand every block of it.
    ///
    /// That list is also what makes freeing safe: every object named here has
    /// its header rewritten, so the dense storage a reopen found on it is
    /// superseded whether or not the new attribute set is dense again, and
    /// freeing it before the replacement is laid out is what lets the
    /// replacement reuse the same blocks.
    fn prepare_dense_attributes(&self, datasets: &[usize]) -> IoResult<()> {
        let mut scopes: Vec<(AttrScope, Vec<AttributeEntry>, CreationOrder)> = Vec::new();
        {
            self.release_superseded_dense_attrs(AttrScope::Root)?;
            let root = self.root_attributes.lock();
            if self.attributes_need_dense(&root, self.header_format(self.root_track_order)) {
                scopes.push((AttrScope::Root, root.clone(), self.root_track_order.attrs));
            }
        }
        for gi in 0..self.group_count() {
            if self.grp(gi).lock().deleted {
                continue;
            }
            self.release_superseded_dense_attrs(AttrScope::Group(gi))?;
            let grp = self.grp(gi);
            let g = grp.lock();
            if self.attributes_need_dense(&g.attributes, self.header_format(g.track_order)) {
                scopes.push((
                    AttrScope::Group(gi),
                    g.attributes.clone(),
                    g.track_order.attrs,
                ));
            }
        }
        for &i in datasets {
            self.release_superseded_dense_attrs(AttrScope::Dataset(i))?;
            let ds = self.ds(i);
            let m = ds.lock();
            let format = self.header_format(TrackOrder {
                links: CreationOrder::default(),
                attrs: m.track_attr_order,
            });
            if self.attributes_need_dense(&m.attributes, format) {
                scopes.push((
                    AttrScope::Dataset(i),
                    m.attributes.clone(),
                    m.track_attr_order,
                ));
            }
        }

        for (scope, attributes, order) in scopes {
            // `close` after `start_swmr` finalizes a second time over the same
            // attribute sets — SWMR refuses every attribute mutation — so
            // rebuilding here would allocate a whole second heap and strand
            // the one the published headers already name.
            if self.dense_attributes.lock().contains_key(&scope) {
                continue;
            }
            let dense = build_dense_attributes(&attributes, &self.ctx, order, &mut |len| {
                self.allocator.allocate(len)
            })?;
            for block in &dense.blocks {
                self.handle.write_at(block.addr, &block.image)?;
            }
            self.dense_attributes.lock().insert(scope, dense.ainfo);
        }
        Ok(())
    }

    /// Define a new contiguous dataset. Returns the dataset index (used with
    /// `write_dataset_raw`).
    ///
    /// The raw-data region is allocated immediately so that
    /// `write_dataset_raw` can be called at any time before `close()`.
    pub fn create_dataset(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
    ) -> IoResult<usize> {
        let create = self.begin_create(name, NewStorage::Contiguous)?;
        let name = create.name.as_str();
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

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
                name: name.to_string(),
                datatype,
                dataspace,
                obj_header_addr: 0, // set during finalize
                data_addr,
                data_size,
                chunked: None,
                fixed_array: None,
                btree_v2: None,
                append: None,
                attributes: Vec::new(),
                obj_header_written_addr: None,
                obj_header_encoded_size: 0,
                filter_pipeline: None,
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version: 4,
                times: None,
            },
        );

        Ok(idx)
    }

    /// Define a new dataset with the NULL dataspace: no elements at all.
    ///
    /// Distinct from a scalar dataset (`create_dataset` with `dims == []`),
    /// which holds exactly one element — a NULL dataspace holds zero, so
    /// there is no raw image to allocate: `data_addr` stays `UNDEF_ADDR` and
    /// `data_size` stays 0 permanently, the same terminal state
    /// `create_dataset` already reaches for a zero-length dimension.
    pub fn create_null_dataset(&self, name: &str, datatype: DatatypeMessage) -> IoResult<usize> {
        let create = self.begin_create(name, NewStorage::Contiguous)?;
        let name = create.name.as_str();

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
                name: name.to_string(),
                datatype,
                dataspace: DataspaceMessage::null(),
                obj_header_addr: 0, // set during finalize
                data_addr: UNDEF_ADDR,
                data_size: 0,
                chunked: None,
                fixed_array: None,
                btree_v2: None,
                append: None,
                attributes: Vec::new(),
                obj_header_written_addr: None,
                obj_header_encoded_size: 0,
                filter_pipeline: None,
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version: 4,
                times: None,
            },
        );

        Ok(idx)
    }

    /// Define a new chunked dataset with an extensible array index.
    ///
    /// Returns the dataset index. The dataset starts empty (dims[0] = 0 if
    /// the first dimension is unlimited). Use `write_chunk` and
    /// `extend_dataset` to add data.
    pub fn create_chunked_dataset(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        let create = self.begin_create(name, NewStorage::Chunked)?;
        let name = create.name.as_str();
        validate_chunk_geometry(dims, max_dims, chunk_dims)?;
        ensure_at_most_one_unlimited(max_dims)?;
        let chunk_bytes = chunk_dims.iter().product::<u64>() * datatype.element_size() as u64;
        let layout_version = self.chunk_layout_version(false, chunk_bytes);
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
            // Chunked storage always requires at least one dimension, so
            // this is never Scalar or Null.
            class: DataspaceClass::Simple,
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
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
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version,
                times: None,
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
                append: None,
            },
        );

        Ok(idx)
    }

    /// Write raw bytes to a contiguous dataset identified by `index`.
    ///
    /// The caller is responsible for providing data in the correct byte order
    /// and layout. The length must match the total data size declared at
    /// creation time.
    pub fn write_dataset_raw(&self, index: usize, data: &[u8]) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        let data_addr = {
            let g = ds.lock();
            if g.chunked.is_some() {
                return Err(crate::io::IoError::InvalidState(
                    "use write_chunk for chunked datasets".into(),
                ));
            }
            if g.data_addr == UNDEF_ADDR {
                return Err(crate::io::IoError::InvalidState(
                    "dataset has no data allocated".into(),
                ));
            }
            if data.len() as u64 != g.data_size {
                return Err(crate::io::IoError::InvalidState(format!(
                    "data size mismatch: expected {} bytes, got {}",
                    g.data_size,
                    data.len()
                )));
            }
            g.data_addr
        };
        self.handle.write_at(data_addr, data)?;
        Ok(())
    }

    /// Write a chunk of data to a chunked dataset.
    ///
    /// `chunk_offset` is the chunk coordinates (e.g., [frame_idx] for a 1D-chunked
    /// streaming dataset where chunk_dims = [1, H, W]).
    /// Only the first (unlimited) dimension index is used for EA indexing.
    ///
    /// `data` must be exactly chunk_size bytes (product of chunk_dims * element_size).
    pub fn write_chunk(&self, index: usize, chunk_idx: u64, data: &[u8]) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_chunk_inner(index, chunk_idx, data)
    }

    /// [`Self::write_chunk`] body; the caller holds the dataset's op lock or
    /// the writer exclusively.
    pub(crate) fn write_chunk_inner(
        &self,
        index: usize,
        chunk_idx: u64,
        data: &[u8],
    ) -> IoResult<()> {
        let ds = self.ds(index);
        // Read the chunk geometry and filter pipeline under one brief lock,
        // then drop it: compression runs *outside* the lock, and
        // `record_ea_chunk` re-locks the same slot, so the guard must not be
        // held across either.
        let (chunk_bytes, pipeline) = {
            let g = ds.lock();
            let element_size = g.datatype.element_size() as u64;
            let chunked = g
                .chunked
                .as_ref()
                .ok_or_else(|| crate::io::IoError::InvalidState("not a chunked dataset".into()))?;
            (
                chunked.chunk_dims.iter().product::<u64>() * element_size,
                g.filter_pipeline.clone(),
            )
        };

        if data.len() as u64 != chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes,
                data.len()
            )));
        }

        // Apply compression if filter pipeline is set
        let compressed;
        let write_data = if let Some(ref pipeline) = pipeline {
            compressed = filter::apply_filters(pipeline, data)?;
            &compressed
        } else {
            data
        };
        // filter_mask = 0: this path runs the whole pipeline, so no filter is
        // skipped for the chunk.
        self.record_ea_chunk(index, chunk_idx, write_data, 0)
    }

    /// Decide where a chunk's bytes belong and put them there, returning the
    /// address to record in the index.
    ///
    /// `old` is the chunk's current `(address, stored length)` if the index
    /// already holds an entry for it. This is the single owner of the
    /// rewrite-placement rule, mirroring libhdf5's `H5D__chunk_file_alloc`
    /// (`H5Dchunk.c`): a chunk whose stored size is unchanged is overwritten
    /// where it already lives, and only a chunk that no longer fits moves,
    /// releasing its old block. Without this every rewrite would abandon the
    /// old block and grow the file.
    fn place_chunk(&self, old: Option<(u64, u64)>, new_len: u64) -> u64 {
        match old {
            // Same stored size: overwrite in place. This is every unfiltered
            // rewrite (the stored size is fixed by the chunk shape) and every
            // filtered rewrite that compressed to the same length.
            Some((addr, len)) if addr != UNDEF_ADDR && len == new_len => addr,
            Some((addr, len)) if addr != UNDEF_ADDR => {
                // The chunk has to move. Under SWMR a reader may still hold an
                // index that points at the old block, so libhdf5 keeps it
                // (H5D__chunk_file_alloc skips H5MF_xfree when the file is
                // open for SWMR writing); do the same.
                if !self.swmr_active {
                    self.allocator.free(addr, len);
                }
                self.allocator.allocate(new_len)
            }
            _ => self.allocator.allocate(new_len),
        }
    }

    /// Place a chunk's already-final bytes (filtered if the dataset is
    /// filtered) in the file and record them in the extensible-array index —
    /// in the index block, a data block, or a super block per the EA geometry.
    /// Shared by write_chunk and write_compressed_chunk.
    ///
    /// The index lookup happens *before* the bytes are placed, because the
    /// entry it finds is what tells [`place_chunk`](Self::place_chunk) whether
    /// this is a rewrite that can stay put.
    fn record_ea_chunk(
        &self,
        index: usize,
        chunk_idx: u64,
        final_bytes: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        let compressed_size = final_bytes.len() as u64;
        let ds = self.ds(index);
        // Hold one slot guard for the whole method: every dataset-state access
        // below goes through `m`, while `self.handle`/`self.allocator`/`self.ctx`
        // are disjoint fields safe to touch with the guard held.
        let mut m = ds.lock();
        let is_filtered = m.filter_pipeline.is_some();
        // For a filtered dataset the chunk's stored size is encoded in the
        // `chunk_size_len`-byte field of each filtered EA entry
        // (`FilteredChunkEntry::encode` writes `nbytes[..chunk_size_len]`,
        // which truncates silently). Reject a size that would not fit, the way
        // libhdf5's H5D_CHUNK_ENCODE_SIZE_CHECK does, instead of corrupting the
        // index. The compress path never exceeds this (chunk_size_len holds the
        // uncompressed chunk size); a direct/raw write with caller-supplied
        // bytes can.
        if is_filtered {
            let chunk_size_len = m.chunked.as_ref().unwrap().chunk_size_len as usize;
            if chunk_size_len < 8 && compressed_size >= (1u64 << (chunk_size_len * 8)) {
                return Err(crate::io::IoError::InvalidState(format!(
                    "filtered chunk size {compressed_size} does not fit in the \
                     {chunk_size_len}-byte extensible-array chunk-size field"
                )));
            }
        }
        let idx_blk_elmts = {
            let c = m.chunked.as_ref().unwrap();
            c.earray_params.idx_blk_elmts as u64
        };

        if chunk_idx < idx_blk_elmts {
            let chunked = m.chunked.as_mut().unwrap();
            if is_filtered {
                if let Some(ref mut fiblk) = chunked.filt_iblk {
                    let old = fiblk.elements[chunk_idx as usize];
                    let chunk_addr =
                        self.place_chunk(Some((old.addr, old.nbytes)), compressed_size);
                    self.handle.write_at(chunk_addr, final_bytes)?;
                    fiblk.elements[chunk_idx as usize] = FilteredChunkEntry {
                        addr: chunk_addr,
                        nbytes: compressed_size,
                        filter_mask,
                    };
                }
            } else {
                // An unfiltered chunk's stored size is fixed by the chunk
                // shape, so a rewrite always fits where it already is.
                let old = chunked.ea_iblk.elements[chunk_idx as usize];
                let chunk_addr = self.place_chunk(Some((old, compressed_size)), compressed_size);
                self.handle.write_at(chunk_addr, final_bytes)?;
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
                let c = m.chunked.as_ref().unwrap();
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
                    let c = m.chunked.as_ref().unwrap();
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
                        let c = m.chunked.as_ref().unwrap();
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
                        let c = m.chunked.as_mut().unwrap();
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
                // A freshly created data block holds only undefined addresses,
                // so this reads as "no previous chunk" without a special case.
                let old = dblk.elements[loc.offset_in_dblk as usize];
                let chunk_addr = self.place_chunk(Some((old.addr, old.nbytes)), compressed_size);
                self.handle.write_at(chunk_addr, final_bytes)?;
                let entry = FilteredChunkEntry {
                    addr: chunk_addr,
                    nbytes: compressed_size,
                    filter_mask,
                };
                dblk.elements[loc.offset_in_dblk as usize] = entry;
                let enc = dblk.encode(&self.ctx, max_nelmts_bits, chunk_size_len);
                if created {
                    dblk_addr = self.allocator.allocate(enc.len() as u64);
                }
                self.handle.write_at(dblk_addr, &enc)?;
                if created {
                    let c = m.chunked.as_mut().unwrap();
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
                // Unfiltered: the stored size is fixed by the chunk shape, so
                // a rewrite always fits its old block. A freshly created data
                // block holds undefined addresses and falls through to a new
                // allocation.
                let old = dblk.elements[loc.offset_in_dblk as usize];
                let chunk_addr = self.place_chunk(Some((old, compressed_size)), compressed_size);
                self.handle.write_at(chunk_addr, final_bytes)?;
                dblk.elements[loc.offset_in_dblk as usize] = chunk_addr;
                let enc = dblk.encode(&self.ctx, max_nelmts_bits);
                if created {
                    dblk_addr = self.allocator.allocate(enc.len() as u64);
                }
                self.handle.write_at(dblk_addr, &enc)?;
                if created {
                    let c = m.chunked.as_mut().unwrap();
                    c.ea_header.num_dblks_created += 1;
                    c.ea_header.size_dblks_created += enc.len() as u64;
                }
            }

            // Record a newly-created data block's address in its parent.
            if created {
                match parent {
                    DblkParent::IndexBlock(di) => {
                        let c = m.chunked.as_mut().unwrap();
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
            let c = m.chunked.as_mut().unwrap();
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

    /// Write a slice (hyperslab) of data to a dataset, contiguous or chunked.
    ///
    /// `starts` and `counts` define the N-dimensional selection.
    /// `data` must be exactly `product(counts) * element_size` bytes.
    ///
    /// The selection is validated once here and then handed to the layout's
    /// own writer, so a caller never has to know which storage the dataset
    /// uses.
    pub fn write_slice(
        &self,
        index: usize,
        starts: &[u64],
        counts: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_slice_inner(index, starts, counts, data)
    }

    /// [`Self::write_slice`] body; the caller holds the dataset's op lock or
    /// the writer exclusively.
    pub(crate) fn write_slice_inner(
        &self,
        index: usize,
        starts: &[u64],
        counts: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds_ref = self.ds(index);
        let ds = ds_ref.lock();
        let is_chunked = ds.chunked.is_some() || ds.fixed_array.is_some() || ds.btree_v2.is_some();

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

        // `dims` borrows the dataset slot; collect what the writers below need
        // so the guard can be dropped before they re-lock it.
        let dims = dims.clone();
        let base_addr = ds.data_addr;
        drop(ds);

        if is_chunked {
            // Rows the append buffer holds are not in the chunks yet; writing
            // them there anyway would be undone when the buffer flushes at
            // close. Hand them to the chunks first.
            self.flush_append_buffer_if_intersecting(index, starts[0], starts[0] + counts[0])?;
            return self.write_slice_chunked(index, starts, counts, data);
        }
        if base_addr == UNDEF_ADDR {
            return Err(crate::io::IoError::InvalidState(
                "dataset has no data allocated".into(),
            ));
        }

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

    /// Write a hyperslab into a chunked dataset, one chunk at a time.
    ///
    /// The selection is already validated by [`write_slice`](Self::write_slice).
    /// For each chunk the selection touches, the chunk's share of `data` is
    /// scattered into a whole-chunk buffer and the chunk is rewritten:
    ///
    /// - a chunk the selection covers completely is built from `data` alone —
    ///   nothing needs reading back (libhdf5 takes the same shortcut with the
    ///   `relax` flag of `H5D__chunk_lock`);
    /// - a chunk covered only in part starts from what is already stored, or
    ///   from a fill-value buffer when the chunk has never been written, so
    ///   neighbouring elements survive and untouched ones read as fill.
    ///
    /// An edge chunk that hangs past the dataset extent is always the partial
    /// case, so the region beyond the extent keeps its fill value.
    fn write_slice_chunked(
        &self,
        index: usize,
        starts: &[u64],
        counts: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        if counts.contains(&0) {
            return Ok(());
        }
        let geo = self.chunk_geometry(index)?;
        let ndims = geo.dims.len();
        if geo.chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset chunk shape has {} dimensions but the dataspace has {}",
                geo.chunk_dims.len(),
                ndims
            )));
        }
        if geo.chunk_dims.contains(&0) {
            return Err(crate::io::IoError::InvalidState(
                "chunk shape has a zero-length dimension".into(),
            ));
        }
        let chunk_bytes = geo.chunk_bytes() as usize;

        // Grid range the selection touches, inclusive on both ends.
        let first: Vec<u64> = (0..ndims).map(|d| starts[d] / geo.chunk_dims[d]).collect();
        let last: Vec<u64> = (0..ndims)
            .map(|d| (starts[d] + counts[d] - 1) / geo.chunk_dims[d])
            .collect();

        let mut coords = first.clone();
        loop {
            // Intersect the selection with this chunk. `in_chunk` is the
            // region's origin inside the chunk, `in_data` its origin inside
            // the caller's counts-shaped buffer, `extent` its size.
            let mut in_chunk = vec![0u64; ndims];
            let mut in_data = vec![0u64; ndims];
            let mut extent = vec![0u64; ndims];
            let mut covers_whole_chunk = true;
            for d in 0..ndims {
                let chunk_origin = coords[d] * geo.chunk_dims[d];
                let lo = starts[d].max(chunk_origin);
                let hi = (starts[d] + counts[d]).min(chunk_origin + geo.chunk_dims[d]);
                in_chunk[d] = lo - chunk_origin;
                in_data[d] = lo - starts[d];
                extent[d] = hi - lo;
                if in_chunk[d] != 0 || extent[d] != geo.chunk_dims[d] {
                    covers_whole_chunk = false;
                }
            }

            let mut buf = if covers_whole_chunk {
                // Every byte is overwritten below.
                vec![0u8; chunk_bytes]
            } else {
                match self.read_chunk_at_coords(index, &coords)? {
                    Some(existing) => {
                        if existing.len() != chunk_bytes {
                            return Err(crate::io::IoError::InvalidState(format!(
                                "stored chunk at {coords:?} is {} bytes but the chunk shape \
                                 needs {chunk_bytes}",
                                existing.len()
                            )));
                        }
                        existing
                    }
                    None => self.new_chunk_buffer(index, chunk_bytes),
                }
            };

            for_each_dual_run(
                &geo.chunk_dims,
                &in_chunk,
                counts,
                &in_data,
                &extent,
                geo.element_size,
                |dst_off, src_off, len| {
                    let dst = dst_off as usize;
                    let src = src_off as usize;
                    buf[dst..dst + len].copy_from_slice(&data[src..src + len]);
                    Ok(())
                },
            )?;
            self.write_chunk_at_coords(index, &coords, &buf)?;

            // Odometer over the touched grid range.
            let mut d = ndims;
            loop {
                if d == 0 {
                    return Ok(());
                }
                d -= 1;
                if coords[d] < last[d] {
                    coords[d] += 1;
                    break;
                }
                coords[d] = first[d];
            }
        }
    }

    /// Add an attribute to the root group (file-level attribute), replacing
    /// a same-name attribute. See [`set_attribute`](Self::set_attribute).
    pub fn add_root_attribute(&self, attr: AttributeMessage) -> IoResult<()> {
        self.set_attribute(AttrTarget::Root, attr)
    }

    /// Insert `attr` into the attribute list `target` names, replacing a
    /// same-name attribute.
    ///
    /// The single owner of attribute-list mutation: an `AttributeMessage`
    /// that leaves a list here has its vlen global-heap objects released, so
    /// no replacement — vlen over vlen, numeric over vlen — can strand heap
    /// space (the attribute counterpart of issue #10's dataset fix).
    ///
    /// Under SWMR every attribute mutation is refused, matching libhdf5's
    /// rule for SWMR writes. Object headers are frozen once streaming
    /// starts — a change was committed at close only when the header
    /// happened to be rebuilt (group attrs always, dataset attrs only if
    /// the dataset also got chunk writes) and silently dropped otherwise —
    /// and a replacement's superseded vlen value could never be reclaimed,
    /// since a streaming reader may hold its heap references.
    pub fn set_attribute(&self, target: AttrTarget<'_>, attr: AttributeMessage) -> IoResult<()> {
        self.insert_attribute(target, attr, Created)
    }

    /// The body of [`set_attribute`](Self::set_attribute), told whether the
    /// attribute it is inserting is genuinely new — see [`AttrOrigin`].
    fn insert_attribute(
        &self,
        target: AttrTarget<'_>,
        attr: AttributeMessage,
        origin: AttrOrigin,
    ) -> IoResult<()> {
        if self.swmr_active {
            return Err(swmr_attr_error(&attr.name));
        }
        // No size gate: an attribute whose message is too large for the
        // 16-bit size field an object header message has spills the object's
        // whole attribute set to dense storage at finalize, exactly as
        // `H5O__attr_create` does. See `attributes_need_dense`.
        let mut entry = AttributeEntry::from(attr);
        let old = self.with_attr_list(target, |attrs| {
            if let Some(pos) = attrs.iter().position(|a| a.name() == entry.name()) {
                // `H5O__attr_write` replaces an existing attribute's value and
                // leaves its `crt_idx` alone: the attribute was not created
                // again, so its creation index does not move.
                entry.set_creation_index(attrs[pos].creation_index());
                Some(std::mem::replace(&mut attrs[pos], entry))
            } else {
                // `H5O__attr_create` stamps the set's running maximum onto the
                // new attribute and post-increments it — but only a create
                // reaches for it.
                entry.set_creation_index(match origin {
                    Created => Some(next_creation_index(attrs)),
                    Rewritten(kept) => kept,
                });
                attrs.push(entry);
                None
            }
        })?;
        match old {
            Some(old) => self.release_attr_vlen(&old),
            None => Ok(()),
        }
    }

    /// Set a variable-length string attribute on `target`, replacing any
    /// same-name attribute.
    ///
    /// Owns the whole replacement sequence: the superseded attribute is
    /// removed and its heap objects released *before* the new value's
    /// collection is allocated — the free-before-alloc order (issue #10)
    /// that lets a reopen-replace loop land in the block it just freed
    /// instead of growing the file every session. The cost, as on the
    /// dataset path: a failure between the eviction and the insert below
    /// loses the attribute rather than leaking its heap space.
    pub fn set_vlen_string_attribute(
        &self,
        target: AttrTarget<'_>,
        name: &str,
        value: &str,
    ) -> IoResult<()> {
        let origin = self.evict_attr(target, name)?;
        let attr = self.vlen_string_attribute(name, value)?;
        self.insert_attribute(target, attr, origin)
    }

    /// The array counterpart of
    /// [`set_vlen_string_attribute`](Self::set_vlen_string_attribute).
    pub fn set_vlen_string_array_attribute(
        &self,
        target: AttrTarget<'_>,
        name: &str,
        values: &[&str],
        dims: &[u64],
    ) -> IoResult<()> {
        let origin = self.evict_attr(target, name)?;
        let attr = self.vlen_string_array_attribute(name, values, dims)?;
        self.insert_attribute(target, attr, origin)
    }

    /// Take the attribute `name` off `target`'s list, releasing its heap
    /// objects. No-op when absent. Refused under SWMR — see
    /// [`set_attribute`](Self::set_attribute).
    ///
    /// What it answers is what the insert that follows it must be told: an
    /// attribute that was there is being rewritten and keeps its creation
    /// index, and one that was not is created.
    fn evict_attr(&self, target: AttrTarget<'_>, name: &str) -> IoResult<AttrOrigin> {
        if self.swmr_active {
            return Err(swmr_attr_error(name));
        }
        let old = self.with_attr_list(target, |attrs| {
            attrs
                .iter()
                .position(|a| a.name() == name)
                .map(|pos| attrs.remove(pos))
        })?;
        match old {
            Some(old) => {
                let origin = Rewritten(old.creation_index());
                self.release_attr_vlen(&old)?;
                Ok(origin)
            }
            None => Ok(Created),
        }
    }

    /// Release the global-heap objects a superseded attribute owned.
    /// Recognizes top-level vlen datatypes only: a *compound* attribute
    /// with vlen members — which this crate cannot write, only a foreign
    /// file can carry — keeps its members' heap objects when replaced or
    /// deleted, the storage cost the foreign writer accepted. Every other
    /// class stores its value inline in the message. Per-object removal
    /// keeps collections shared with other refs (libhdf5-written files)
    /// intact.
    fn release_attr_vlen(&self, old: &AttributeEntry) -> IoResult<()> {
        use crate::format::messages::datatype::DatatypeMessage;
        // An attribute whose message this crate could not decode keeps
        // whatever heap space it references: releasing objects named by bytes
        // we cannot interpret would free storage that is still live.
        let Some(old) = old.readable() else {
            return Ok(());
        };
        if matches!(
            old.datatype,
            DatatypeMessage::VarLenString { .. } | DatatypeMessage::VarLenSequence { .. }
        ) {
            self.release_vlen_references(&old.data)?;
        }
        Ok(())
    }

    /// Run `f` on the attribute list `target` names — the accessor every
    /// attribute mutation shares.
    fn with_attr_list<R>(
        &self,
        target: AttrTarget<'_>,
        f: impl FnOnce(&mut Vec<AttributeEntry>) -> R,
    ) -> IoResult<R> {
        match target {
            AttrTarget::Root => Ok(f(&mut self.root_attributes.lock())),
            AttrTarget::Group(path) => {
                let path = self.canonical_group_path(path);
                for grp in self.group_refs() {
                    let mut g = grp.lock();
                    if g.name == path && !g.deleted {
                        return Ok(f(&mut g.attributes));
                    }
                }
                Err(crate::io::IoError::NotFound(format!(
                    "group '{path}' not found"
                )))
            }
            AttrTarget::Dataset(index) => {
                let count = self.dataset_count();
                if index >= count {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "dataset index {index} out of range (have {count})"
                    )));
                }
                let ds = self.ds(index);
                let mut m = ds.lock();
                // Every caller of this mutates the list, and a reopened
                // dataset's header is rewritten only when it is marked stale.
                m.header_dirty = true;
                Ok(f(&mut m.attributes))
            }
        }
    }

    /// Store each of `items` as a global heap object and return its
    /// placement `(collection address, object index)`, in input order —
    /// the writer side of libhdf5's `H5HG_insert`.
    ///
    /// Placement follows libhdf5: a collection from the CWFS list takes an
    /// item when its free space holds the object *and* a residual
    /// free-space marker header (`encode_at_size` always emits the
    /// marker); what no listed collection can take goes into a fresh
    /// collection, spilling into another at the 65535-object index cap.
    /// One batch may therefore span several collections — invisible to
    /// readers, which resolve each reference's own collection address. An
    /// empty batch allocates nothing: an empty collection still encodes
    /// to the 4096-byte `H5HG_MINALLOC` minimum, a block nothing would
    /// reference. libhdf5 additionally tries to extend a nearly-full
    /// collection's block in place (`H5MF_try_extend`); this writer does
    /// not — an oversized item always starts a fresh collection.
    ///
    /// The `cwfs` lock is held across every read-modify-rewrite of a
    /// listed collection block: it serializes concurrent inserts (two
    /// datasets' writers can pack the same block) and inserts against
    /// [`release_vlen_references`](Self::release_vlen_references), which
    /// rewrites the same blocks when objects are freed.
    ///
    /// Under SWMR the CWFS list is neither consulted nor updated and every
    /// batch gets fresh collections: packing rewrites a block a streaming
    /// reader may be mid-walk on — the same reason `place_chunk` keeps a
    /// relocated chunk's old block.
    fn insert_vlen_objects(&self, items: &[&[u8]]) -> IoResult<Vec<(u64, u16)>> {
        use crate::format::global_heap::{GlobalHeapCollection, GlobalHeapObject};

        if items.is_empty() {
            return Ok(Vec::new());
        }
        let objhdr = GlobalHeapCollection::object_disk_size(&self.ctx, 0);
        let mut placements = Vec::with_capacity(items.len());
        let mut i = 0;

        // Pack into listed collections while one can take the next item.
        if !self.swmr_active {
            let mut cwfs = self.cwfs.lock();
            while i < items.len() {
                let need = GlobalHeapCollection::object_disk_size(&self.ctx, items[i].len());
                let Some(pos) = cwfs.iter().position(|e| e.free >= need + objhdr) else {
                    // Second pass of libhdf5's H5F_cwfs_find_free_heap: no
                    // listed collection has room, so try to grow one in
                    // place before falling back to a fresh collection.
                    if self.extend_listed_collection(&mut cwfs, need + objhdr)? {
                        continue;
                    }
                    break;
                };
                let (addr, size) = (cwfs[pos].addr, cwfs[pos].size);
                let image = self.handle.read_at(addr, size)?;
                let (mut gcol, _) = GlobalHeapCollection::decode(&image[..size], &self.ctx)?;
                // The disk is the truth for free space; the entry is a hint.
                let Some(mut free) = gcol.free_space_at(&self.ctx, size) else {
                    cwfs.remove(pos);
                    continue;
                };
                let mut next_idx = gcol.max_index();
                let mut took = false;
                while i < items.len() && next_idx < u16::MAX {
                    let need = GlobalHeapCollection::object_disk_size(&self.ctx, items[i].len());
                    if free < need + objhdr {
                        break;
                    }
                    next_idx += 1;
                    gcol.objects.push(GlobalHeapObject {
                        index: next_idx,
                        ref_count: 0,
                        data: items[i].to_vec(),
                    });
                    placements.push((addr, next_idx));
                    free -= need;
                    took = true;
                    i += 1;
                }
                if took {
                    let rewritten = gcol.encode_at_size(&self.ctx, size)?;
                    self.handle.write_at(addr, &rewritten)?;
                    // Correct the entry to the measured free space and move
                    // it to the front — libhdf5 keeps `cwfs` in
                    // most-recently-used order.
                    let mut e = cwfs.remove(pos);
                    e.free = free;
                    cwfs.insert(0, e);
                } else if next_idx == u16::MAX {
                    // At the index cap nothing can be inserted no matter the
                    // free space; drop the entry or the scan re-picks it
                    // forever. (A removal can lower the top index again, and
                    // the release side re-lists the collection then.)
                    cwfs.remove(pos);
                } else {
                    // The hint overstated the block's free space — shrink it
                    // to the measured value so the scan moves on.
                    cwfs[pos].free = free;
                }
            }
        }

        // What remains goes into fresh collections.
        while i < items.len() {
            let mut gcol = GlobalHeapCollection::new();
            // Objects are pushed with a running index: `add_object` rescans
            // for the max index per call, O(n²) across a spill-sized batch.
            let mut next_idx: u16 = 0;
            while i < items.len() && next_idx < u16::MAX {
                next_idx += 1;
                gcol.objects.push(GlobalHeapObject {
                    index: next_idx,
                    ref_count: 0,
                    data: items[i].to_vec(),
                });
                i += 1;
            }
            let encoded = gcol.encode(&self.ctx);
            let addr = self.allocator.allocate(encoded.len() as u64);
            self.handle.write_at(addr, &encoded)?;
            for idx in 1..=next_idx {
                placements.push((addr, idx));
            }
            // List the block's leftover free space for later inserts — the
            // minimum-size padding of a small batch is most of 4096 bytes.
            // Below two object headers not even an empty object fits.
            if !self.swmr_active {
                if let Some(free) = gcol.free_space_at(&self.ctx, encoded.len()) {
                    if free >= 2 * objhdr {
                        cwfs_note(&mut self.cwfs.lock(), addr, encoded.len(), free);
                    }
                }
            }
        }
        Ok(placements)
    }

    /// Try to extend one listed collection in place so it can take an
    /// object needing `want` bytes of free space — the second pass of
    /// libhdf5's `H5F_cwfs_find_free_heap`: grow the file allocation
    /// ([`FileAllocator::try_extend`], mirroring `H5MF_try_extend`) and then
    /// the collection itself (`H5HG_extend`: a larger declared size and a
    /// free-space marker covering the new tail — here by re-encoding at the
    /// grown size, which writes exactly those two things).
    ///
    /// Extension size is `max(collection_size, shortfall)` — at least a
    /// doubling — capped so the result stays within [`GCOL_MAX_SIZE`], both
    /// as upstream computes them. On success the grown entry moves to the
    /// front of the list and the caller's scan re-picks it; the free-space
    /// measurement is taken from the block on disk, not the list's hint, so
    /// the rewrite and the entry agree.
    ///
    /// Caller holds the `cwfs` lock (it passes the guarded list), which is
    /// what serializes this read-modify-rewrite against concurrent inserts
    /// and releases.
    fn extend_listed_collection(&self, cwfs: &mut Vec<CwfsEntry>, want: usize) -> IoResult<bool> {
        use crate::format::global_heap::{GlobalHeapCollection, GCOL_MAX_SIZE};

        let mut pos = 0;
        while pos < cwfs.len() {
            let (addr, size) = (cwfs[pos].addr, cwfs[pos].size);
            let image = self.handle.read_at(addr, size)?;
            let (gcol, _) = GlobalHeapCollection::decode(&image[..size], &self.ctx)?;
            // The disk is the truth for free space; the entry is a hint.
            let Some(free) = gcol.free_space_at(&self.ctx, size) else {
                cwfs.remove(pos);
                continue;
            };
            // A hint can understate the block (upstream's FREE_SIZE is its
            // in-memory truth and cannot): if the block already has room,
            // correct the hint instead of doubling the collection.
            if free >= want {
                cwfs[pos].free = free;
                return Ok(true);
            }
            let new_need = size.max(want.saturating_sub(free));
            if size + new_need > GCOL_MAX_SIZE
                || !self
                    .allocator
                    .try_extend(addr, size as u64, new_need as u64)
            {
                pos += 1;
                continue;
            }
            let new_size = size + new_need;
            let rewritten = gcol.encode_at_size(&self.ctx, new_size)?;
            self.handle.write_at(addr, &rewritten)?;
            let mut e = cwfs.remove(pos);
            e.size = new_size;
            e.free = free + new_need;
            cwfs.insert(0, e);
            return Ok(true);
        }
        Ok(false)
    }

    /// Create a variable-length string dataset and write string data.
    ///
    /// Stores strings in the global heap. The dataset raw data consists of
    /// vlen references (collection_addr + object_index pairs).
    ///
    /// `charset` is the datatype's declared character set (0 = ASCII,
    /// 1 = UTF-8); the strings are checked against it before anything is
    /// written, so the type never misdescribes the bytes under it.
    pub fn create_vlen_string_dataset(
        &self,
        name: &str,
        strings: &[&str],
        charset: u8,
    ) -> IoResult<usize> {
        use crate::format::global_heap::encode_vlen_reference;
        use crate::format::messages::datatype::DatatypeMessage;

        ensure_vlen_charset(charset, strings)?;

        let create = self.begin_create(name, NewStorage::Contiguous)?;
        let name = create.name.as_str();
        let num_strings = strings.len() as u64;

        // Store the strings as heap objects; a batch that fits an earlier
        // collection's free space shares its block.
        let items: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let placements = self.insert_vlen_objects(&items)?;

        // Build raw data: vlen references
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let data_size = (num_strings as usize) * ref_size;
        let mut raw_data = Vec::with_capacity(data_size);
        for (i, &(gcol_addr, obj_idx)) in placements.iter().enumerate() {
            let seq_len = crate::format::global_heap::vlen_seq_len(strings[i].len())?;
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
        let datatype = DatatypeMessage::VarLenString {
            padding: 0,
            charset,
        };
        let dataspace =
            crate::format::messages::dataspace::DataspaceMessage::simple(&[num_strings]);

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
                name: name.to_string(),
                datatype,
                dataspace,
                obj_header_addr: 0,
                data_addr,
                data_size: data_size as u64,
                attributes: Vec::new(),
                obj_header_written_addr: None,
                obj_header_encoded_size: 0,
                filter_pipeline: None,
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version: 4,
                times: None,
                chunked: None,
                fixed_array: None,
                btree_v2: None,
                append: None,
            },
        );

        Ok(idx)
    }

    /// Create a 1-D variable-length byte-array dataset.
    ///
    /// The `u8` case of [`create_vlen_sequence_dataset`], where an item's
    /// byte image and its element count are the same number.
    ///
    /// [`create_vlen_sequence_dataset`]: Self::create_vlen_sequence_dataset
    pub fn create_vlen_bytes_dataset(&self, name: &str, items: &[&[u8]]) -> IoResult<usize> {
        use crate::format::messages::datatype::DatatypeMessage;

        self.create_vlen_sequence_dataset(name, DatatypeMessage::u8_type(), items)
    }

    /// Create a 1-D variable-length sequence dataset over `base`.
    ///
    /// Each item is the encoded image of one sequence — `n * base.element_size()`
    /// bytes in the base type's own byte order — and is stored as a global-heap
    /// object; the dataset holds one vlen reference per item, the same on-disk
    /// shape a vlen string dataset has. The `H5T_VLEN` length field counts base
    /// elements rather than bytes, so an image whose length is not a whole
    /// number of elements is refused here rather than stored under a length
    /// that misreads it.
    pub fn create_vlen_sequence_dataset(
        &self,
        name: &str,
        base: DatatypeMessage,
        items: &[&[u8]],
    ) -> IoResult<usize> {
        use crate::format::global_heap::encode_vlen_reference;
        use crate::format::messages::datatype::DatatypeMessage;

        let elem_size = base.element_size() as usize;
        if elem_size == 0 {
            return Err(crate::io::IoError::InvalidState(format!(
                "vlen base datatype {base} has no element size"
            )));
        }
        for (i, item) in items.iter().enumerate() {
            if !item.len().is_multiple_of(elem_size) {
                return Err(crate::io::IoError::InvalidState(format!(
                    "sequence {i} is {} bytes, not a whole number of {elem_size}-byte elements",
                    item.len()
                )));
            }
        }

        let create = self.begin_create(name, NewStorage::Contiguous)?;
        let name = create.name.as_str();
        let num_items = items.len() as u64;

        // Store the sequence images as heap objects, sharing collection
        // blocks as `create_vlen_string_dataset` does.
        let placements = self.insert_vlen_objects(items)?;

        // Build raw data: one vlen reference per item.
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let data_size = (num_items as usize) * ref_size;
        let mut raw_data = Vec::with_capacity(data_size);
        for (i, &(gcol_addr, obj_idx)) in placements.iter().enumerate() {
            let seq_len = crate::format::global_heap::vlen_seq_len(items[i].len() / elem_size)?;
            raw_data.extend_from_slice(&encode_vlen_reference(
                seq_len,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        // Allocate and write raw data.
        let data_addr = self.allocator.allocate(data_size as u64);
        self.handle.write_at(data_addr, &raw_data)?;

        let datatype = DatatypeMessage::VarLenSequence {
            base: Box::new(base),
        };
        let dataspace = crate::format::messages::dataspace::DataspaceMessage::simple(&[num_items]);

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
                name: name.to_string(),
                datatype,
                dataspace,
                obj_header_addr: 0,
                data_addr,
                data_size: data_size as u64,
                attributes: Vec::new(),
                obj_header_written_addr: None,
                obj_header_encoded_size: 0,
                filter_pipeline: None,
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version: 4,
                times: None,
                chunked: None,
                fixed_array: None,
                btree_v2: None,
                append: None,
            },
        );

        Ok(idx)
    }

    /// Create a chunked, compressed variable-length string dataset.
    ///
    /// Strings are stored in the global heap (same as `create_vlen_string_dataset`),
    /// but the vlen references are stored in chunked layout with the given filter
    /// pipeline (e.g., deflate, zstd). `chunk_size` is the number of strings per chunk.
    pub fn create_vlen_string_dataset_compressed(
        &self,
        name: &str,
        strings: &[&str],
        chunk_size: usize,
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        use crate::format::global_heap::encode_vlen_reference;
        use crate::format::messages::datatype::DatatypeMessage;

        let create = self.begin_create(name, NewStorage::Chunked)?;
        let name = create.name.as_str();
        let num_strings = strings.len() as u64;
        validate_chunk_geometry(&[num_strings], &[num_strings], &[chunk_size as u64])?;

        // Store the strings as heap objects; the geometry validation above
        // must precede this so a refused call allocates nothing.
        let items: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let placements = self.insert_vlen_objects(&items)?;

        // Build raw data: vlen references
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let data_size = (num_strings as usize) * ref_size;
        let mut raw_data = Vec::with_capacity(data_size);
        for (i, &(gcol_addr, obj_idx)) in placements.iter().enumerate() {
            let seq_len = crate::format::global_heap::vlen_seq_len(strings[i].len())?;
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
        let layout_version = self.chunk_layout_version(true, chunk_bytes);
        let chunk_size_len = self.chunk_size_len_for(layout_version, chunk_bytes);

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
            // Chunked storage always requires at least one dimension, so
            // this is never Scalar or Null.
            class: DataspaceClass::Simple,
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let ea_iblk = ExtensibleArrayIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
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
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version,
                times: None,
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
                append: None,
            },
        );

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
        &self,
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
    pub fn append_vlen_strings(&self, ds_index: usize, strings: &[&str]) -> IoResult<()> {
        use crate::format::global_heap::encode_vlen_reference;
        use crate::format::messages::datatype::DatatypeMessage;

        if strings.is_empty() {
            return Ok(());
        }

        // Whole-operation guard: buffer take, frame writes, re-buffer and
        // extend below are separate slot acquisitions that a concurrent
        // same-dataset append must not interleave with.
        let cell = self.ds(ds_index);
        let _op = cell.op.lock();

        // The elements about to be written are vlen references; any other
        // element type would be overwritten with them as raw bytes.
        let charset = {
            let ds = self.ds(ds_index);
            let m = ds.lock();
            match m.datatype {
                DatatypeMessage::VarLenString { charset, .. } => charset,
                _ => {
                    return Err(crate::io::IoError::InvalidState(
                        "append_vlen_strings is only for variable-length string datasets".into(),
                    ))
                }
            }
        };
        ensure_vlen_charset(charset, strings)?;

        // Every deterministic rejection must precede the heap write below:
        // a collection written for a batch the append then refuses (a
        // contiguous dataset, or a reopened dataset whose chunk index was
        // not reconstructed) is a 4096-byte orphan nothing references.
        let chunk_dims = self
            .dataset_chunk_dims(ds_index)
            .ok_or_else(|| crate::io::IoError::InvalidState("not a chunked dataset".into()))?
            .to_vec();
        let dims = self.dataset_dims(ds_index).to_vec();

        // Store the batch's strings as heap objects; a batch that fits an
        // earlier collection's free space shares its block.
        let items: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let placements = self.insert_vlen_objects(&items)?;

        // Build raw vlen reference bytes
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let mut raw = Vec::with_capacity(strings.len() * ref_size);
        for (i, &(gcol_addr, obj_idx)) in placements.iter().enumerate() {
            let seq_len = crate::format::global_heap::vlen_seq_len(strings[i].len())?;
            raw.extend_from_slice(&encode_vlen_reference(
                seq_len,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        let n_new_frames = strings.len();
        let current_dim0 = dims[0] as usize;
        let chunk_dim0 = chunk_dims[0] as usize;
        let frame_bytes = ref_size;

        // Merge the buffer with the new frames when it is the dataset's tail;
        // a buffer left mid-extent (the extent moved past it) keeps its
        // recorded place — flush it and start fresh at the current end.
        let taken = { self.ds(ds_index).lock().append.take() };
        let (base_dim0, buffered_frames, mut combined) = match taken {
            Some(b) if b.base + b.frames == current_dim0 as u64 => {
                (b.base as usize, b.frames as usize, b.bytes)
            }
            Some(b) => {
                self.write_append_frames(ds_index, b.base, b.frames, &b.bytes)?;
                (current_dim0, 0, Vec::new())
            }
            None => (current_dim0, 0, Vec::new()),
        };
        combined.extend_from_slice(&raw);

        let total_frames = buffered_frames + n_new_frames;

        // Rows up to the last chunk boundary are written now; the tail that
        // does not complete a chunk goes back in the buffer for the next
        // append (or the flush at close). The boundary can precede
        // `base_dim0` — a reopened file's flushed partial chunk leaves the
        // base mid-chunk — in which case everything is tail.
        let last_boundary = ((base_dim0 + total_frames) / chunk_dim0) * chunk_dim0;
        let write_frames = last_boundary.saturating_sub(base_dim0);
        let tail_frames = total_frames - write_frames;
        if write_frames > 0 {
            self.write_append_frames(
                ds_index,
                base_dim0 as u64,
                write_frames as u64,
                &combined[..write_frames * frame_bytes],
            )?;
        }
        if tail_frames > 0 {
            let ds = self.ds(ds_index);
            let mut m = ds.lock();
            m.append = Some(AppendBuffer {
                base: (base_dim0 + write_frames) as u64,
                frames: tail_frames as u64,
                bytes: combined[write_frames * frame_bytes..].to_vec(),
            });
        }

        // Extend dims
        let logical_dim0 = base_dim0 + total_frames;
        let mut new_dims = dims;
        new_dims[0] = logical_dim0 as u64;
        self.extend_dataset_inner(ds_index, &new_dims)?;

        Ok(())
    }

    /// Replace elements `start .. start + strings.len()` of a 1-D
    /// variable-length string dataset, leaving its extent and every other
    /// element alone.
    ///
    /// The replacements go into the global heap and only the vlen
    /// references of the named elements are rewritten, so the cost is the
    /// new strings plus the chunks those references live in — not the column.
    /// The objects the old references pointed at are freed *before* the
    /// replacement is allocated, so repeated updates reuse space instead of
    /// growing the file — including across close/reopen cycles, where the
    /// in-memory free list starts empty and only this free-first order lets
    /// the session reuse the block it just released. This is what libhdf5
    /// does: `H5T__vlen_disk_write` deletes the reference it read into the
    /// conversion background buffer before storing the new one.
    ///
    /// Elements the append buffer still holds are flushed to their chunks
    /// first, so the whole range is on disk and one write path covers it.
    pub fn write_vlen_strings_slice(
        &self,
        ds_index: usize,
        start: u64,
        strings: &[&str],
    ) -> IoResult<()> {
        use crate::format::global_heap::{encode_vlen_reference, vlen_reference_size};
        use crate::format::messages::datatype::DatatypeMessage;

        // An empty batch is a no-op: nothing to replace, nothing to free.
        if strings.is_empty() {
            return Ok(());
        }

        // Whole-operation guard: the flush, the old-reference reads and the
        // slice write below must not interleave with a concurrent
        // same-dataset operation.
        let cell = self.ds(ds_index);
        let _op = cell.op.lock();

        // Snapshot what the write needs, then drop the guard: `write_slice`
        // below re-locks the same slot.
        let (charset, dims, writable) = {
            let ds = self.ds(ds_index);
            let m = ds.lock();
            let charset = match m.datatype {
                DatatypeMessage::VarLenString { charset, .. } => charset,
                _ => {
                    return Err(crate::io::IoError::InvalidState(
                        "write_vlen_strings_slice is only for variable-length string datasets"
                            .into(),
                    ))
                }
            };
            let writable = m.chunked.is_some()
                || m.fixed_array.is_some()
                || m.btree_v2.is_some()
                || m.data_addr != UNDEF_ADDR;
            (charset, m.dataspace.dims.clone(), writable)
        };

        // `write_slice_inner` rejects a dataset with neither chunk machinery
        // nor allocated data (a reopened dataset whose index was not
        // reconstructed) — that rejection must come before the heap write
        // below, or every failed call orphans a 4096-byte collection.
        if !writable {
            return Err(crate::io::IoError::InvalidState(
                "dataset has no data allocated".into(),
            ));
        }

        if dims.len() != 1 {
            return Err(crate::io::IoError::InvalidState(format!(
                "write_vlen_strings_slice is only for 1-dimension datasets, this one has {}",
                dims.len()
            )));
        }
        let end = start + strings.len() as u64;
        if end > dims[0] {
            return Err(crate::io::IoError::InvalidState(format!(
                "elements {start}..{end} are outside the dataset's {} elements",
                dims[0]
            )));
        }
        ensure_vlen_charset(charset, strings)?;

        let ref_size = vlen_reference_size(&self.ctx);

        // Elements the append buffer holds are not in the chunks yet: hand
        // them to the chunks first so the whole range is on disk and the one
        // write path below covers it.
        self.flush_append_buffer_if_intersecting(ds_index, start, end)?;

        // The on-disk references about to be overwritten, read before anything
        // moves. libhdf5 reads the same bytes into the conversion background
        // buffer (`H5D__scatgath_write` gathers the file's current elements
        // when `need_bkg` is set) and hands them to `H5T__vlen_disk_write`,
        // which deletes them before storing the new reference.
        let superseded = self.current_element_bytes(ds_index, start, end - start, ref_size)?;

        // Free the superseded objects *before* allocating the replacement,
        // the order `H5T__vlen_disk_write` uses. The freed block satisfies
        // the allocation below within this same session, so a reopen-and-
        // replace loop keeps the file flat — no persisted free-space
        // information exists to carry it across sessions (issue #10). The
        // cost, shared with libhdf5: a failure between here and the ref
        // write below leaves the dataset's old references dangling.
        self.release_vlen_references(&superseded)?;

        // The insert comes after the release above so the space the release
        // recovered — a freed block, or in-collection bytes the release just
        // listed in `cwfs` — can satisfy this batch.
        let items: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let placements = self.insert_vlen_objects(&items)?;

        let mut refs = Vec::with_capacity(strings.len() * ref_size);
        for (i, &(gcol_addr, obj_idx)) in placements.iter().enumerate() {
            refs.extend_from_slice(&encode_vlen_reference(
                crate::format::global_heap::vlen_seq_len(strings[i].len())?,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        self.write_slice_inner(ds_index, &[start], &[strings.len() as u64], &refs)?;

        Ok(())
    }

    /// The bytes elements `start .. start + count` of a 1-D dataset currently
    /// hold, whichever layout stores them.
    ///
    /// Elements no write has reached yet read as zeros — for a vlen dataset
    /// that is the nil reference, which names no heap object.
    fn current_element_bytes(
        &self,
        ds_index: usize,
        start: u64,
        count: u64,
        element_size: usize,
    ) -> IoResult<Vec<u8>> {
        let mut out = vec![0u8; count as usize * element_size];
        if count == 0 {
            return Ok(out);
        }

        let (is_chunked, data_addr) = {
            let ds = self.ds(ds_index);
            let m = ds.lock();
            (
                m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some(),
                m.data_addr,
            )
        };

        if !is_chunked {
            if data_addr != UNDEF_ADDR {
                // `read_at_most`, not `read_at`: a contiguous dataset's block is
                // reserved when it is created, so the file can still be shorter
                // than the block until something writes it. What is missing has
                // never been written, which is the zeros above.
                let at = data_addr + start * element_size as u64;
                let got = self.handle.read_at_most(at, out.len())?;
                out[..got.len()].copy_from_slice(&got);
            }
            return Ok(out);
        }

        let geo = self.chunk_geometry(ds_index)?;
        let per_chunk = geo.chunk_dims[0];
        // Only a corrupt or crafted file declares a zero-length chunk
        // dimension; the divisions below must reject it the way
        // `write_slice` does, not panic.
        if per_chunk == 0 {
            return Err(crate::io::IoError::InvalidState(
                "chunk shape has a zero-length dimension".into(),
            ));
        }
        let end = start + count;
        for c in (start / per_chunk)..=((end - 1) / per_chunk) {
            let origin = c * per_chunk;
            let lo = start.max(origin);
            let hi = end.min(origin + per_chunk);
            // A chunk with no block yet leaves this span as the zeros above.
            let Some(chunk) = self.read_chunk_at_coords(ds_index, &[c])? else {
                continue;
            };
            let src = ((lo - origin) as usize) * element_size;
            let dst = ((lo - start) as usize) * element_size;
            let len = ((hi - lo) as usize) * element_size;
            if src + len > chunk.len() {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk {c} is {} bytes, too short for elements {lo}..{hi}",
                    chunk.len()
                )));
            }
            out[dst..dst + len].copy_from_slice(&chunk[src..src + len]);
        }
        Ok(out)
    }

    /// Free the global heap objects `refs` names, so replacing a vlen element
    /// does not strand what it used to point at.
    ///
    /// Callers pass refs only for *top-level* vlen datatypes (the
    /// `collect_refs` / `is_vlen` decisions at the prune, delete and
    /// attribute-release sites all match `VarLenString`/`VarLenSequence`).
    /// A compound datatype with vlen members — writable only by a foreign
    /// library, never by this crate — keeps its members' heap objects when
    /// its storage is pruned, deleted or replaced.
    ///
    /// This is libhdf5's `H5HG_remove` reached through `H5T__vlen_disk_delete`:
    /// the object leaves its collection, the collection is rewritten at its
    /// existing size with the recovered bytes given to the free-space marker,
    /// and a collection that ends up empty returns its block to the allocator.
    /// A rewritten collection's recovered space is listed in `cwfs` for
    /// [`insert_vlen_objects`](Self::insert_vlen_objects) to pack into; a
    /// freed block leaves the list.
    /// A nil reference (address 0 or `UNDEF_ADDR`) names no object. The
    /// address decides, not the sequence length: this crate's writers store
    /// even the empty string as a real heap object, so a zero-length reference
    /// with a defined address still holds one that must be released. libhdf5
    /// diverges here against itself — `H5T__vlen_disk_delete` returns before
    /// `H5HG_remove` when the sequence length is zero, yet its write path
    /// (`H5VL__native_blob_put`) inserts a heap object even for an empty
    /// sequence, stranding it forever. The address rule frees those objects.
    ///
    /// Heap objects carry no reference count on this path, matching libhdf5:
    /// its vlen code never calls `H5HG_link` (only the virtual-dataset layer
    /// does). Releasing the same reference twice is absorbed by the
    /// missing-index check below, but a crafted file in which two elements
    /// share one heap object would lose it for the survivor when either is
    /// replaced — the same exposure the file has under libhdf5. This crate's
    /// writers never share: each element write inserts its own object.
    ///
    /// Under SWMR nothing is freed and no collection is rewritten: a reader may
    /// be following those references, the same reason `place_chunk` keeps a
    /// relocated chunk's old block.
    fn release_vlen_references(&self, refs: &[u8]) -> IoResult<()> {
        use crate::format::global_heap::{
            decode_vlen_reference, vlen_reference_size, GlobalHeapCollection,
        };

        if self.swmr_active {
            return Ok(());
        }
        let ref_size = vlen_reference_size(&self.ctx);
        if ref_size == 0 || refs.len() < ref_size {
            return Ok(());
        }

        // Group by collection so one holding several replaced objects is read,
        // rewritten and judged empty exactly once.
        let mut per_collection: std::collections::BTreeMap<u64, Vec<u16>> = Default::default();
        for r in refs.chunks_exact(ref_size) {
            let (_seq_len, addr, obj_idx) = decode_vlen_reference(r, &self.ctx)?;
            if addr == 0 || addr == UNDEF_ADDR {
                continue;
            }
            let Ok(idx) = u16::try_from(obj_idx) else {
                return Err(crate::io::IoError::InvalidState(format!(
                    "global heap object index {obj_idx} does not fit the 16-bit on-disk field"
                )));
            };
            per_collection.entry(addr).or_default().push(idx);
        }

        // The `cwfs` lock is held across the sweep: it serializes these
        // collection-block rewrites (and frees) against
        // `insert_vlen_objects`, which may be packing new objects into the
        // same blocks.
        let objhdr = GlobalHeapCollection::object_disk_size(&self.ctx, 0);
        let mut cwfs = self.cwfs.lock();
        for (addr, indices) in per_collection {
            // A collection is at least 4096 bytes (H5HG_MINALLOC) and most are
            // exactly that, so one read usually covers the whole image; only
            // an oversized collection needs a second read at its declared size.
            let mut image = self.handle.read_at_most(addr, 4096)?;
            let declared = GlobalHeapCollection::decode_size(&image, &self.ctx)?;
            if declared > image.len() {
                image = self.handle.read_at(addr, declared)?;
            }
            let (mut gcol, _) = GlobalHeapCollection::decode(&image[..declared], &self.ctx)?;
            let mut removed_any = false;
            for idx in indices {
                removed_any |= gcol.remove_object(idx);
            }
            // Every index already gone (a stale or duplicate reference):
            // leave the image alone. Rewriting is not just wasted I/O — a
            // 100%-full collection written by libhdf5 has no free-space
            // marker, so re-encoding it at its declared size cannot fit one
            // and the whole element update would fail.
            if !removed_any {
                continue;
            }
            if gcol.is_empty() {
                self.allocator.free(addr, declared as u64);
                // The block is gone; a lingering entry would let an insert
                // pack into space the allocator can hand to anything.
                cwfs.retain(|e| e.addr != addr);
            } else {
                let rewritten = gcol.encode_at_size(&self.ctx, declared)?;
                self.handle.write_at(addr, &rewritten)?;
                // The recovered bytes are packable now — list them, the way
                // libhdf5's `H5HG_remove` adds the heap to `cwfs`.
                if let Some(free) = gcol.free_space_at(&self.ctx, declared) {
                    if free >= 2 * objhdr {
                        cwfs_note(&mut cwfs, addr, declared, free);
                    }
                }
            }
        }
        Ok(())
    }

    /// Add an attribute to a dataset.
    ///
    /// The attribute will be written as a message in the dataset's object
    /// header when the file is finalized.
    pub fn add_dataset_attribute(&self, ds_index: usize, attr: AttributeMessage) -> IoResult<()> {
        self.set_attribute(AttrTarget::Dataset(ds_index), attr)
    }

    /// Add (or replace) an attribute on a group identified by its full path.
    ///
    /// The attribute is written into the group's object header when the
    /// file is finalized. An existing attribute with the same name is
    /// replaced, matching [`add_root_attribute`](Self::add_root_attribute).
    pub fn add_group_attribute(&self, group_path: &str, attr: AttributeMessage) -> IoResult<()> {
        self.set_attribute(AttrTarget::Group(group_path), attr)
    }

    /// Build a variable-length UTF-8 string attribute message.
    ///
    /// The string is stored as one object in a global heap collection and the
    /// returned [`AttributeMessage`] carries the vlen reference as its data,
    /// with a vlen-string datatype and scalar dataspace. h5py reads the value
    /// back as a Python `str` (not `bytes`).
    ///
    /// This is the single owner of vlen-string-attribute construction: every
    /// public string-attribute setter (dataset, group, root, and the SWMR
    /// equivalents) routes through it, so a `VarLenUnicode` /
    /// `set_attr_string` value is always stored as a true variable-length
    /// string rather than the fixed-length string it used to be.
    ///
    /// The string's heap object is placed by
    /// [`insert_vlen_objects`](Self::insert_vlen_objects), so consecutive
    /// attributes pack into a shared collection instead of each paying the
    /// 4096-byte `H5HG_MINALLOC` minimum for a block that holds one string.
    fn vlen_string_attribute(&self, name: &str, value: &str) -> IoResult<AttributeMessage> {
        use crate::format::global_heap::encode_vlen_reference;
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::datatype::DatatypeMessage;

        let (gcol_addr, obj_idx) = self.insert_vlen_objects(&[value.as_bytes()])?[0];
        let seq_len = crate::format::global_heap::vlen_seq_len(value.len())?;
        let data = encode_vlen_reference(seq_len, gcol_addr, obj_idx as u32, &self.ctx);
        Ok(AttributeMessage {
            name: name.to_string(),
            datatype: DatatypeMessage::vlen_string_utf8(),
            dataspace: DataspaceMessage::scalar(),
            data,
        })
    }

    /// Build a variable-length UTF-8 string **array** attribute message.
    ///
    /// The N-dimensional counterpart of
    /// [`vlen_string_attribute`](Self::vlen_string_attribute): every element
    /// string is stored as one object in a single global heap collection, and
    /// the attribute data is the row-major concatenation of one vlen reference
    /// per element. The datatype is the same vlen-string datatype; the dataspace
    /// is the simple dataspace described by `shape` (an empty `shape` is a
    /// scalar). h5py reads the value back as a numpy array of Python `str` with
    /// that shape.
    ///
    /// The caller owns the invariant that `values.len()` equals the product of
    /// `shape` (the public setters validate it before calling). The element
    /// objects are placed by
    /// [`insert_vlen_objects`](Self::insert_vlen_objects) — a zero-element
    /// array allocates nothing, and each reference carries its element's
    /// own collection address.
    fn vlen_string_array_attribute(
        &self,
        name: &str,
        values: &[&str],
        shape: &[u64],
    ) -> IoResult<AttributeMessage> {
        use crate::format::global_heap::encode_vlen_reference;
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::datatype::DatatypeMessage;

        debug_assert_eq!(
            values.len() as u64,
            shape.iter().product::<u64>(),
            "vlen_string_array_attribute values.len() must equal product(shape)"
        );

        let items: Vec<&[u8]> = values.iter().map(|v| v.as_bytes()).collect();
        let placements = self.insert_vlen_objects(&items)?;

        let mut data = Vec::with_capacity(values.len() * 16);
        for (i, &(gcol_addr, obj_idx)) in placements.iter().enumerate() {
            data.extend_from_slice(&encode_vlen_reference(
                crate::format::global_heap::vlen_seq_len(values[i].len())?,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }
        Ok(AttributeMessage {
            name: name.to_string(),
            datatype: DatatypeMessage::vlen_string_utf8(),
            dataspace: DataspaceMessage::simple(shape),
            data,
        })
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
    pub fn set_dataset_fill_value(&self, ds_index: usize, bytes: Vec<u8>) -> IoResult<()> {
        let count = self.dataset_count();
        if ds_index >= count {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset index {} out of range",
                ds_index
            )));
        }
        let ds_ref = self.ds(ds_index);
        let mut ds = ds_ref.lock();
        let es = ds.datatype.element_size() as usize;
        if bytes.len() != es {
            return Err(crate::io::IoError::InvalidState(format!(
                "fill value is {} bytes but dataset element size is {}",
                bytes.len(),
                es
            )));
        }
        ds.fill_value = Some(bytes);
        ds.header_dirty = true;

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
        let ds = self.ds(ds_index);
        let m = ds.lock();
        let fv = m.fill_value.as_deref();
        crate::format::messages::fill_value::tiled_fill(chunk_bytes, fv)
    }

    /// Write `n_frames` whole frames whose first row is `base_frame`, for
    /// whichever chunk index the dataset uses and whatever its chunk shape.
    ///
    /// The single owner of an append's chunk writes. The frames are one
    /// hyperslab — rows `base_frame .. base_frame + n_frames` over the full
    /// row shape — so the write goes through
    /// [`write_slice_chunked`](Self::write_slice_chunked), the same engine
    /// `write_slice` uses: a chunk the span covers completely is written
    /// straight through, a partial one is read-modify-write on top of what
    /// is stored (or the fill value), and a chunk row narrower or wider
    /// than the frame row is scattered at the chunk stride. The previous
    /// owner required the extensible-array index and packed rows at the
    /// frame stride, so appends to a fixed-array or v2 B-tree dataset
    /// failed at close and lost the buffered rows.
    ///
    /// The caller holds the dataset's op lock or the writer exclusively.
    pub(crate) fn write_append_frames(
        &self,
        ds_index: usize,
        base_frame: u64,
        n_frames: u64,
        frames: &[u8],
    ) -> IoResult<()> {
        if n_frames == 0 {
            return Ok(());
        }
        let geo = self.chunk_geometry(ds_index)?;
        let mut starts = vec![0u64; geo.dims.len()];
        starts[0] = base_frame;
        let mut counts = geo.dims.clone();
        counts[0] = n_frames;
        let expected = counts.iter().product::<u64>() * geo.element_size;
        if frames.len() as u64 != expected {
            return Err(crate::io::IoError::InvalidState(format!(
                "{n_frames} frames at rows {base_frame}.. need {expected} bytes, got {}",
                frames.len()
            )));
        }
        self.write_slice_chunked(ds_index, &starts, &counts, frames)
    }

    /// Write the dataset's append buffer (if any) into its chunks and clear
    /// it. The single owner of the buffer-to-chunks transition: the flush at
    /// close, an append meeting a non-contiguous buffer, and any operation
    /// about to write rows the buffer holds all come through here.
    ///
    /// The caller holds the dataset's op lock or the writer exclusively —
    /// the take and the frame writes are separate acquisitions.
    pub(crate) fn flush_append_buffer(&self, ds_index: usize) -> IoResult<()> {
        let taken = { self.ds(ds_index).lock().append.take() };
        match taken {
            Some(b) => self.write_append_frames(ds_index, b.base, b.frames, &b.bytes),
            None => Ok(()),
        }
    }

    /// Flush the append buffer when rows `start_row .. end_row` intersect
    /// the buffered range — those rows' current content is the buffer, and
    /// writing them on disk while the buffer still holds them would be
    /// undone by the flush at close.
    ///
    /// The caller holds the dataset's op lock or the writer exclusively.
    pub(crate) fn flush_append_buffer_if_intersecting(
        &self,
        ds_index: usize,
        start_row: u64,
        end_row: u64,
    ) -> IoResult<()> {
        let intersects = {
            let ds = self.ds(ds_index);
            let m = ds.lock();
            m.append
                .as_ref()
                .is_some_and(|b| start_row < b.base + b.frames && end_row > b.base)
        };
        if intersects {
            self.flush_append_buffer(ds_index)
        } else {
            Ok(())
        }
    }

    /// Read an already-written chunk's *decompressed* bytes when the chunk
    /// is allocated and resolvable from the in-memory extensible-array
    /// index. Handles index-block and data-block chunks, filtered and
    /// unfiltered.
    ///
    /// Returns `Ok(None)` only when the chunk has never been written
    /// (address `UNDEF`) or the index genuinely does not reach it, which for
    /// a read-modify-write means the chunk's content is the fill value.
    pub(crate) fn read_chunk_if_present(
        &self,
        ds_index: usize,
        chunk_idx: u64,
    ) -> IoResult<Option<Vec<u8>>> {
        // Phase 1: resolve the chunk's location from the in-memory index.
        // Hold the slot guard through Phase 1: `chunked` borrows it, while the
        // `self.handle`/`self.ctx` reads below touch disjoint fields.
        let ds = self.ds(ds_index);
        let m = ds.lock();
        let element_size = m.datatype.element_size() as u64;
        let pipeline = m.filter_pipeline.clone();
        let Some(chunked) = m.chunked.as_ref() else {
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
        self.read_chunk_block(pipeline.as_ref(), addr, nbytes, mask)
    }

    /// Read one stored chunk block and undo its filters.
    ///
    /// `nbytes` is the *stored* length and `mask` the chunk's filter mask, so
    /// a chunk written by a direct chunk write with a skipped filter is
    /// reversed correctly. `Ok(None)` means the chunk has no block yet — the
    /// single place that judgement is made, shared by every chunk index.
    fn read_chunk_block(
        &self,
        pipeline: Option<&FilterPipeline>,
        addr: u64,
        nbytes: u64,
        mask: u32,
    ) -> IoResult<Option<Vec<u8>>> {
        if addr == UNDEF_ADDR || nbytes == 0 {
            return Ok(None);
        }
        let raw = self.handle.read_at(addr, nbytes as usize)?;
        match pipeline {
            Some(pl) => Ok(Some(filter::reverse_filters_masked(pl, &raw, mask)?)),
            None => Ok(Some(raw)),
        }
    }

    /// Read the *decompressed* bytes of the chunk at `chunk_coords`, whichever
    /// chunk index the dataset uses, or `Ok(None)` when that chunk has never
    /// been written.
    ///
    /// This is the read half of a partial-chunk read-modify-write: a hyperslab
    /// write that covers only part of a chunk must start from what is already
    /// there. Keeping one entry point for all three index types is what lets
    /// [`write_slice`](Self::write_slice) stay index-agnostic.
    pub(crate) fn read_chunk_at_coords(
        &self,
        ds_index: usize,
        chunk_coords: &[u64],
    ) -> IoResult<Option<Vec<u8>>> {
        let geo = self.chunk_geometry(ds_index)?;
        // Only the linearly-addressed indexes compute a slot; a v2 B-tree is
        // keyed by the coordinates themselves (and may hold unlimited inner
        // dimensions, which have no linear slot).
        match geo.kind {
            ChunkIndexKind::ExtensibleArray => {
                let linear = geo.linear_index(chunk_coords)?;
                self.read_chunk_if_present(ds_index, linear)
            }
            ChunkIndexKind::FixedArray => {
                let linear = geo.linear_index(chunk_coords)?;
                let ds = self.ds(ds_index);
                let m = ds.lock();
                let pipeline = m.filter_pipeline.clone();
                let fa = m.fixed_array.as_ref().unwrap();
                let lidx = linear as usize;
                let (addr, nbytes, mask) = if pipeline.is_some() {
                    match fa.fa_dblk.filtered_elements.get(lidx) {
                        Some(e) => (e.address, e.chunk_size, e.filter_mask),
                        None => return Ok(None),
                    }
                } else {
                    match fa.fa_dblk.elements.get(lidx) {
                        Some(&a) => (a, geo.chunk_bytes(), 0),
                        None => return Ok(None),
                    }
                };
                drop(m);
                self.read_chunk_block(pipeline.as_ref(), addr, nbytes, mask)
            }
            ChunkIndexKind::BtreeV2 => {
                let ds = self.ds(ds_index);
                let m = ds.lock();
                let pipeline = m.filter_pipeline.clone();
                let bt2 = m.btree_v2.as_ref().unwrap();
                // A filtered index records the stored size and mask per chunk;
                // an unfiltered one stores whole chunks, so their size is the
                // chunk shape and no filter ran.
                let found = if bt2.index.filtered {
                    bt2.index
                        .lookup_filtered(chunk_coords)
                        .map(|r| (r.chunk_address, r.chunk_size, r.filter_mask))
                } else {
                    bt2.index
                        .lookup(chunk_coords)
                        .map(|r| (r.chunk_address, geo.chunk_bytes(), 0))
                };
                drop(m);
                match found {
                    Some((addr, nbytes, mask)) => {
                        self.read_chunk_block(pipeline.as_ref(), addr, nbytes, mask)
                    }
                    None => Ok(None),
                }
            }
        }
    }

    /// Write one whole chunk addressed by its grid coordinates, whichever
    /// chunk index the dataset uses. `data` is the chunk's unfiltered bytes;
    /// the dataset's filter pipeline (if any) runs here.
    ///
    /// The write half of the pair with
    /// [`read_chunk_at_coords`](Self::read_chunk_at_coords). Unlike the
    /// dataset-level `write_chunk_at`, this never grows the dataspace — a
    /// hyperslab write is bounded by the current extent by definition.
    ///
    /// The caller holds the dataset's op lock or the writer exclusively.
    pub(crate) fn write_chunk_at_coords(
        &self,
        ds_index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let geo = self.chunk_geometry(ds_index)?;
        match geo.kind {
            ChunkIndexKind::ExtensibleArray => {
                let linear = geo.linear_index(chunk_coords)?;
                self.write_chunk_inner(ds_index, linear, data)
            }
            ChunkIndexKind::FixedArray => {
                self.write_chunk_fixed_array_inner(ds_index, chunk_coords, data)
            }
            ChunkIndexKind::BtreeV2 => {
                self.write_chunk_btree_v2_inner(ds_index, chunk_coords, data)
            }
        }
    }

    /// Snapshot the geometry needed to address a chunked dataset's grid.
    ///
    /// Taken under one brief slot guard so the callers below — which re-lock
    /// the slot through `write_chunk`/`read_chunk_*` — never hold it across
    /// compression or I/O.
    fn chunk_geometry(&self, ds_index: usize) -> IoResult<ChunkGeometry> {
        let ds = self.ds(ds_index);
        let m = ds.lock();
        let (kind, chunk_dims) = if let Some(ref c) = m.chunked {
            (ChunkIndexKind::ExtensibleArray, c.chunk_dims.clone())
        } else if let Some(ref f) = m.fixed_array {
            (ChunkIndexKind::FixedArray, f.chunk_dims.clone())
        } else if let Some(ref b) = m.btree_v2 {
            (ChunkIndexKind::BtreeV2, b.chunk_dims.clone())
        } else {
            return Err(crate::io::IoError::InvalidState(
                "not a chunked dataset".into(),
            ));
        };
        Ok(ChunkGeometry {
            kind,
            dims: m.dataspace.dims.clone(),
            max_dims: m.dataspace.max_dims.clone(),
            chunk_dims,
            element_size: m.datatype.element_size() as u64,
        })
    }

    /// Index-grid slot of the chunk at grid `coords` (see
    /// [`crate::io::chunk_grid`]).
    pub(crate) fn chunk_slot(&self, ds_index: usize, coords: &[u64]) -> IoResult<u64> {
        self.chunk_geometry(ds_index)?.linear_index(coords)
    }

    /// Grid coordinates of the chunk recorded under index-grid slot `linear`
    /// — the inverse of [`Self::chunk_slot`].
    pub(crate) fn chunk_coords_from_slot(
        &self,
        ds_index: usize,
        linear: u64,
    ) -> IoResult<Vec<u64>> {
        let geo = self.chunk_geometry(ds_index)?;
        crate::io::chunk_grid::coords_of(
            &geo.dims,
            geo.max_dims.as_deref(),
            &geo.chunk_dims,
            linear,
        )
    }

    /// Define a chunked dataset indexed by a fixed array, fixed at its
    /// current shape (`max_dims == dims`). `chunk_dims` defines the chunk
    /// shape. Returns the dataset index.
    pub fn create_fixed_array_dataset(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        self.create_fixed_array_dataset_with_max(name, datatype, dims, dims, chunk_dims, None)
    }

    /// Define a fixed-shape compressed chunked dataset indexed by a
    /// *filtered* Fixed Array (`max_dims == dims`).
    ///
    /// Like `create_fixed_array_dataset`, but the FA header carries the filtered
    /// client id and a `chunk_size_len`-wide compressed-size field per chunk
    /// (`FixedArrayFilteredChunkElement`), and the dataset gets a filter
    /// pipeline. Chunks written via `write_chunk_fixed_array` are compressed and
    /// their compressed size + filter mask are recorded in the data block.
    pub fn create_fixed_array_dataset_with_pipeline(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk_dims: &[u64],
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        self.create_fixed_array_dataset_with_max(
            name,
            datatype,
            dims,
            dims,
            chunk_dims,
            Some(pipeline),
        )
    }

    /// Define a chunked dataset indexed by a fixed array, growable up to
    /// `max_dims` (every maximum finite — libhdf5 picks this index exactly
    /// when no dimension is unlimited).
    ///
    /// The array is sized for the chunk grid of the *maximum* extent, the
    /// libhdf5 rule (`H5D__farray_idx_create` uses `max_nchunks`), so the
    /// dataset can be extended to `max_dims` without re-indexing chunks.
    pub fn create_fixed_array_dataset_with_max(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        pipeline: Option<FilterPipeline>,
    ) -> IoResult<usize> {
        let create = self.begin_create(name, NewStorage::Chunked)?;
        let name = create.name.as_str();
        validate_chunk_geometry(dims, max_dims, chunk_dims)?;
        if max_dims.contains(&u64::MAX) {
            return Err(crate::io::IoError::InvalidState(
                "a fixed-array index requires a fixed maximum shape (no unlimited dimension)"
                    .into(),
            ));
        }
        let mut num_chunks: u64 = 1;
        for g in crate::io::chunk_grid::index_grid(dims, Some(max_dims), chunk_dims)? {
            num_chunks = num_chunks.checked_mul(g).ok_or_else(|| {
                crate::io::IoError::InvalidState("chunk count overflows u64".into())
            })?;
        }

        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * datatype.element_size() as u64;
        let layout_version = self.chunk_layout_version(pipeline.is_some(), chunk_bytes);

        // Create the FA header. For a filtered FA, chunk_size_len is sized
        // the same way the filtered Extensible Array path computes it:
        // derived from the uncompressed chunk byte count under layout v4,
        // the fixed `sizeof_size` under layout v5.
        let mut fa_header = if pipeline.is_some() {
            let chunk_size_len = self.chunk_size_len_for(layout_version, chunk_bytes);
            FixedArrayHeader::new_for_filtered_chunks(&self.ctx, num_chunks, chunk_size_len)
        } else {
            FixedArrayHeader::new_for_chunks(&self.ctx, num_chunks)
        };
        let hdr_encoded = fa_header.encode(&self.ctx);
        let fa_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);

        // Create the FA data block. libhdf5 switches to a paged layout once
        // num_elmts exceeds dblk_page_nelmts; both layouts allocate space
        // for `num_chunks` entries up front, but the paged layout also
        // reserves the page-init bitmap and a per-page checksum.
        let fa_dblk = if pipeline.is_some() {
            FixedArrayDataBlock::new_filtered(fa_header_addr, num_chunks as usize)
        } else {
            FixedArrayDataBlock::new_unfiltered(fa_header_addr, num_chunks as usize)
        };
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

        // The maximum is stored even when it equals the dims: it is what
        // `extend_dataset` checks growth against, and the FA capacity above
        // is exactly its chunk grid.
        let dataspace = DataspaceMessage {
            // Chunked storage always requires at least one dimension, so
            // this is never Scalar or Null.
            class: DataspaceClass::Simple,
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
                name: name.to_string(),
                datatype,
                dataspace,
                obj_header_addr: 0,
                data_addr: UNDEF_ADDR,
                data_size: 0,
                attributes: Vec::new(),
                obj_header_written_addr: None,
                obj_header_encoded_size: 0,
                filter_pipeline: pipeline,
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version,
                times: None,
                chunked: None,
                btree_v2: None,
                fixed_array: Some(FixedArrayDatasetInfo {
                    chunk_dims: chunk_dims.to_vec(),
                    fa_header_addr,
                    fa_dblk_addr,
                    fa_header,
                    fa_dblk,
                    chunks_written: 0,
                }),
                append: None,
            },
        );

        Ok(idx)
    }

    /// Define a chunked dataset indexed by a B-tree v2 (multiple unlimited dimensions).
    ///
    /// Returns the dataset index.
    pub fn create_btree_v2_dataset(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        self.create_btree_v2_dataset_inner(name, datatype, dims, max_dims, chunk_dims, None)
    }

    /// Define a *filtered* chunked dataset indexed by a B-tree v2.
    ///
    /// The v2 B-tree counterpart of
    /// [`create_chunked_dataset_with_pipeline`](Self::create_chunked_dataset_with_pipeline):
    /// chunks are compressed on write and the index records each chunk's
    /// stored size and filter mask (record type 11), the same shape libhdf5
    /// builds when a multi-unlimited-dimension dataset has a filter pipeline
    /// (`H5Dbtree2.c`, `H5D_BT2_FILT`).
    pub fn create_btree_v2_dataset_with_pipeline(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        self.create_btree_v2_dataset_inner(
            name,
            datatype,
            dims,
            max_dims,
            chunk_dims,
            Some(pipeline),
        )
    }

    fn create_btree_v2_dataset_inner(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        pipeline: Option<FilterPipeline>,
    ) -> IoResult<usize> {
        use crate::format::chunk_index::btree_v2::Bt2Header;

        let create = self.begin_create(name, NewStorage::Chunked)?;
        let name = create.name.as_str();
        validate_chunk_geometry(dims, max_dims, chunk_dims)?;
        let ndims = dims.len();
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * datatype.element_size() as u64;
        let layout_version = self.chunk_layout_version(pipeline.is_some(), chunk_bytes);

        // The filtered record's size field is as wide as libhdf5 will
        // recompute it — from the uncompressed chunk size under layout v4,
        // the fixed `sizeof_size` under layout v5 — exactly as the
        // extensible- and fixed-array filtered paths size theirs.
        let bt2_index = match pipeline {
            Some(_) => {
                let len = self.chunk_size_len_for(layout_version, chunk_bytes);
                Bt2ChunkIndex::new_filtered(ndims, len)
            }
            None => Bt2ChunkIndex::new_unfiltered(ndims),
        };

        // The bulk loader spreads a level's records evenly over its nodes, one
        // separator between adjacent siblings, which needs room for a few
        // records per node. HDF5's rank limit of 32 leaves room for seven; a
        // wider rank than that has no valid geometry, so reject it here rather
        // than emit a tree no reader can walk.
        let record_size = bt2_index.record_size(&self.ctx) as usize;
        let node_size = bt2_index.node_size as usize;
        if node_size < 10 + 3 * record_size {
            return Err(crate::io::IoError::InvalidState(format!(
                "a {ndims}-dimension v2 B-tree record is {record_size} bytes, too wide \
                 for a {node_size}-byte node"
            )));
        }

        // Only the header gets a home now: it names an empty tree, whose root
        // is undefined until the first flush bulk-loads the index into nodes.
        let hdr = if bt2_index.filtered {
            Bt2Header::new_for_filtered_chunks(&self.ctx, ndims, bt2_index.chunk_size_len)
        } else {
            Bt2Header::new_for_chunks(&self.ctx, ndims)
        };
        let hdr_encoded = hdr.encode(&self.ctx);
        let bt2_header_addr = self.allocator.allocate(hdr_encoded.len() as u64);
        self.handle.write_at(bt2_header_addr, &hdr_encoded)?;

        let dataspace = DataspaceMessage {
            // Chunked storage always requires at least one dimension, so
            // this is never Scalar or Null.
            class: DataspaceClass::Simple,
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
                name: name.to_string(),
                datatype,
                dataspace,
                obj_header_addr: 0,
                data_addr: UNDEF_ADDR,
                data_size: 0,
                attributes: Vec::new(),
                obj_header_written_addr: None,
                obj_header_encoded_size: 0,
                filter_pipeline: pipeline,
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version,
                times: None,
                chunked: None,
                fixed_array: None,
                btree_v2: Some(Bt2DatasetInfo {
                    chunk_dims: chunk_dims.to_vec(),
                    max_dims: max_dims.to_vec(),
                    bt2_header_addr,
                    node_addrs: Vec::new(),
                    index: bt2_index,
                    chunks_written: 0,
                }),
                append: None,
            },
        );

        Ok(idx)
    }

    /// Create a chunked dataset with compression using the given filter pipeline.
    ///
    /// This is similar to `create_chunked_dataset` but attaches a filter pipeline
    /// (e.g., deflate compression). The pipeline is applied when writing chunks.
    pub fn create_chunked_dataset_compressed(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        compression_level: u32,
    ) -> IoResult<usize> {
        let create = self.begin_create(name, NewStorage::Chunked)?;
        let name = create.name.as_str();
        validate_chunk_geometry(dims, max_dims, chunk_dims)?;
        ensure_at_most_one_unlimited(max_dims)?;
        let element_size = datatype.element_size() as u64;
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;
        let layout_version = self.chunk_layout_version(true, chunk_bytes);
        let chunk_size_len = self.chunk_size_len_for(layout_version, chunk_bytes);

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
            // Chunked storage always requires at least one dimension, so
            // this is never Scalar or Null.
            class: DataspaceClass::Simple,
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

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
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
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version,
                times: None,
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
                append: None,
            },
        );

        Ok(idx)
    }

    /// Create a chunked dataset with a custom filter pipeline.
    pub fn create_chunked_dataset_with_pipeline(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        let create = self.begin_create(name, NewStorage::Chunked)?;
        let name = create.name.as_str();
        validate_chunk_geometry(dims, max_dims, chunk_dims)?;
        ensure_at_most_one_unlimited(max_dims)?;
        let element_size = datatype.element_size() as u64;
        let chunk_bytes: u64 = chunk_dims.iter().product::<u64>() * element_size;
        let layout_version = self.chunk_layout_version(true, chunk_bytes);
        let chunk_size_len = self.chunk_size_len_for(layout_version, chunk_bytes);

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
            // Chunked storage always requires at least one dimension, so
            // this is never Scalar or Null.
            class: DataspaceClass::Simple,
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };
        let ea_iblk = ExtensibleArrayIndexBlock::new(
            ea_header_addr,
            earray_params.idx_blk_elmts,
            ndblk_addrs,
            nsblk_addrs,
        );

        let idx = self.push_dataset(
            &create,
            DatasetInfo {
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
                deleted: false,
                extent_dirty: false,
                header_dirty: false,
                nlink_written: 1,
                creation_seq: self.take_creation_seq(),
                track_attr_order: self.track_order.attrs,
                fill_value: None,
                layout_version,
                times: None,
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
                append: None,
            },
        );
        Ok(idx)
    }

    /// Write a chunk to a fixed-array-indexed dataset.
    ///
    /// `chunk_coords` is the multidimensional chunk index (e.g., [row_chunk, col_chunk]).
    /// The uncompressed `data` must be exactly one chunk wide; the filter
    /// pipeline (if any) runs here before the bytes reach the index.
    pub fn write_chunk_fixed_array(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_chunk_fixed_array_inner(index, chunk_coords, data)
    }

    /// [`Self::write_chunk_fixed_array`] body; the caller holds the dataset's
    /// op lock or the writer exclusively.
    pub(crate) fn write_chunk_fixed_array_inner(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        // Read what we need under one brief slot guard, then compress
        // OUTSIDE the lock: `record_fixed_array_chunk` re-locks the same slot,
        // so the guard must be dropped before it (and before apply_filters).
        let ds = self.ds(index);
        let (chunk_bytes, pipeline) = {
            let m = ds.lock();
            let element_size = m.datatype.element_size() as u64;
            let fa = m.fixed_array.as_ref().ok_or_else(|| {
                crate::io::IoError::InvalidState("not a fixed-array dataset".into())
            })?;
            (
                fa.chunk_dims.iter().product::<u64>() * element_size,
                m.filter_pipeline.clone(),
            )
        };

        if data.len() as u64 != chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes,
                data.len()
            )));
        }
        let write_data;
        let data_to_write = if let Some(ref pipeline) = pipeline {
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
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_compressed_chunk_fixed_array_inner(index, chunk_coords, data, filter_mask)
    }

    /// [`Self::write_compressed_chunk_fixed_array`] body; the caller holds
    /// the dataset's op lock or the writer exclusively.
    pub(crate) fn write_compressed_chunk_fixed_array_inner(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        if self.ds(index).lock().filter_pipeline.is_none() {
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
        &self,
        index: usize,
        chunk_coords: &[u64],
        final_bytes: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        // Hold one slot guard for the whole method; `self.allocator`/`self.handle`/
        // `self.ctx` below touch disjoint fields safe to use with the guard held.
        let ds = self.ds(index);
        let mut m = ds.lock();
        let is_filtered = m.filter_pipeline.is_some();
        let fa = m
            .fixed_array
            .as_ref()
            .ok_or_else(|| crate::io::IoError::InvalidState("not a fixed-array dataset".into()))?;

        // Linear chunk index in the maximum-extent grid — the slot the fixed
        // array (sized from that grid at create) records the chunk under.
        let linear_idx = crate::io::chunk_grid::linear_index(
            &m.dataspace.dims,
            m.dataspace.max_dims.as_deref(),
            &fa.chunk_dims,
            chunk_coords,
        )?;

        // Update the fixed array data block. The slot is read before the bytes
        // are placed so a rewrite can stay where it is (see `place_chunk`).
        let fa = m.fixed_array.as_mut().unwrap();
        let lidx = linear_idx as usize;
        if is_filtered {
            // Filtered FA: store address + stored size + filter mask. A
            // non-zero mask bit means "filter i was skipped for this chunk".
            let stored_size = final_bytes.len();
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
                let old = &fa.fa_dblk.filtered_elements[lidx];
                let chunk_addr =
                    self.place_chunk(Some((old.address, old.chunk_size)), stored_size as u64);
                self.handle.write_at(chunk_addr, final_bytes)?;
                fa.fa_dblk.filtered_elements[lidx] = FixedArrayFilteredChunkElement {
                    address: chunk_addr,
                    chunk_size: stored_size as u64,
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
                // Unfiltered: the stored size is fixed by the chunk shape, so
                // a rewrite always fits its old block.
                let old = fa.fa_dblk.elements[lidx];
                let len = final_bytes.len() as u64;
                let chunk_addr = self.place_chunk(Some((old, len)), len);
                self.handle.write_at(chunk_addr, final_bytes)?;
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
    /// `data` is the chunk's unfiltered bytes; if the dataset has a filter
    /// pipeline it runs here and the index records the stored size and mask.
    pub fn write_chunk_btree_v2(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_chunk_btree_v2_inner(index, chunk_coords, data)
    }

    /// [`Self::write_chunk_btree_v2`] body; the caller holds the dataset's op
    /// lock or the writer exclusively.
    pub(crate) fn write_chunk_btree_v2_inner(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        // Read what the write needs under a brief guard, then compress OUTSIDE
        // the lock — filtering a chunk must not hold the dataset slot.
        let ds = self.ds(index);
        let (chunk_bytes, pipeline) = {
            let m = ds.lock();
            let element_size = m.datatype.element_size() as u64;
            let bt2 = m.btree_v2.as_ref().ok_or_else(|| {
                crate::io::IoError::InvalidState("not a B-tree v2 dataset".into())
            })?;
            (
                bt2.chunk_dims.iter().product::<u64>() * element_size,
                m.filter_pipeline.clone(),
            )
        };

        if data.len() as u64 != chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk data size mismatch: expected {} bytes, got {}",
                chunk_bytes,
                data.len()
            )));
        }

        let filtered;
        let stored = match pipeline {
            Some(ref pl) => {
                filtered = filter::apply_filters(pl, data)?;
                &filtered[..]
            }
            None => data,
        };

        // filter_mask = 0: the whole pipeline ran (or the dataset is
        // unfiltered), so no filter is skipped.
        self.record_btree_v2_chunk(index, chunk_coords, stored, 0)
    }

    /// Write a pre-filtered chunk verbatim to a BT2-indexed dataset, recording
    /// the caller-supplied `filter_mask`.
    ///
    /// The v2-B-tree half of the HDF5 "direct chunk write" (`H5Dwrite_chunk`).
    /// The bytes are stored exactly as given; `filter_mask` bit *i* set means
    /// filter *i* of the pipeline was **not** applied and must be skipped on
    /// read. Requires a filtered dataset — only a type-11 record has a slot for
    /// a stored size and mask.
    pub fn write_compressed_chunk_btree_v2(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_compressed_chunk_btree_v2_inner(index, chunk_coords, data, filter_mask)
    }

    /// [`Self::write_compressed_chunk_btree_v2`] body; the caller holds the
    /// dataset's op lock or the writer exclusively.
    pub(crate) fn write_compressed_chunk_btree_v2_inner(
        &self,
        index: usize,
        chunk_coords: &[u64],
        data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        if self.ds(index).lock().filter_pipeline.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "write_compressed_chunk_btree_v2 requires a filtered dataset (no \
                 slot for a compressed size or filter mask on an unfiltered chunk \
                 index)"
                    .into(),
            ));
        }
        self.record_btree_v2_chunk(index, chunk_coords, data, filter_mask)
    }

    /// Place a chunk's already-final bytes (filtered if the dataset is
    /// filtered, raw otherwise) in the file and record them in the v2 B-tree,
    /// under the caller-supplied `filter_mask`.
    ///
    /// Shared by [`write_chunk_btree_v2`](Self::write_chunk_btree_v2) and
    /// [`write_compressed_chunk_btree_v2`](Self::write_compressed_chunk_btree_v2),
    /// so both reach the index through one placement rule.
    fn record_btree_v2_chunk(
        &self,
        index: usize,
        chunk_coords: &[u64],
        final_bytes: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        let stored_len = final_bytes.len() as u64;
        let ds = self.ds(index);
        let mut m = ds.lock();
        let element_size = m.datatype.element_size() as u64;
        let bt2 = m
            .btree_v2
            .as_ref()
            .ok_or_else(|| crate::io::IoError::InvalidState("not a B-tree v2 dataset".into()))?;
        let chunk_bytes = bt2.chunk_dims.iter().product::<u64>() * element_size;
        // A filtered record encodes the stored size in a `chunk_size_len`-byte
        // field that truncates silently. Reject a size that would not fit, as
        // the extensible-array path does — the compress path never exceeds it,
        // but a direct write with caller-supplied bytes can.
        if bt2.index.filtered {
            let chunk_size_len = bt2.index.chunk_size_len as usize;
            if chunk_size_len < 8 && stored_len >= (1u64 << (chunk_size_len * 8)) {
                return Err(crate::io::IoError::InvalidState(format!(
                    "filtered chunk size {stored_len} does not fit in the \
                     {chunk_size_len}-byte v2 B-tree chunk-size field"
                )));
            }
        }
        // Place the bytes: a rewrite whose stored size is unchanged stays
        // where it is (always so when unfiltered — the size is fixed by the
        // chunk shape), and one that no longer fits moves, releasing its old
        // block. See `place_chunk`.
        let old = if bt2.index.filtered {
            bt2.index
                .lookup_filtered(chunk_coords)
                .map(|r| (r.chunk_address, r.chunk_size))
        } else {
            bt2.index
                .lookup(chunk_coords)
                .map(|r| (r.chunk_address, chunk_bytes))
        };
        let chunk_addr = self.place_chunk(old, stored_len);
        self.handle.write_at(chunk_addr, final_bytes)?;

        let bt2 = m.btree_v2.as_mut().unwrap();
        if bt2.index.filtered {
            bt2.index
                .insert_filtered(chunk_coords.to_vec(), chunk_addr, stored_len, filter_mask);
        } else {
            bt2.index.insert(chunk_coords.to_vec(), chunk_addr);
        }
        bt2.chunks_written += 1;

        Ok(())
    }

    /// Write multiple chunks in a batch, optionally compressing in parallel.
    ///
    /// `chunks` is a list of (chunk_idx, data) pairs for an EA-indexed dataset.
    pub fn write_chunks_batch(&self, ds_index: usize, chunks: &[(u64, &[u8])]) -> IoResult<()> {
        let ds = self.ds(ds_index);
        let _op = ds.op.lock();
        self.write_chunks_batch_inner(ds_index, chunks)
    }

    /// [`Self::write_chunks_batch`] body; the caller holds the dataset's op
    /// lock or the writer exclusively.
    pub(crate) fn write_chunks_batch_inner(
        &self,
        ds_index: usize,
        chunks: &[(u64, &[u8])],
    ) -> IoResult<()> {
        #[cfg(feature = "parallel")]
        {
            // If filter pipeline is set, compress all chunks in parallel.
            // Clone the pipeline out under a brief slot guard so the parallel
            // compression below runs off the lock.
            let pipeline = self.ds(ds_index).lock().filter_pipeline.clone();
            if let Some(ref pipeline) = pipeline {
                let chunk_data: Vec<Vec<u8>> = chunks.iter().map(|(_, d)| d.to_vec()).collect();
                // Propagate a filter error rather than storing raw bytes under a
                // filter_mask that claims the pipeline ran (see
                // apply_filters_parallel). Ok reaching here means every chunk
                // compressed fully, so filter_mask = 0 is truthful.
                let compressed = filter::apply_filters_parallel(pipeline, &chunk_data)?;
                for ((idx, _), compressed_data) in chunks.iter().zip(compressed.iter()) {
                    self.write_compressed_chunk_inner(ds_index, *idx, compressed_data, 0)?;
                }
                return Ok(());
            }
        }
        // Fallback: sequential
        for (idx, data) in chunks {
            self.write_chunk_inner(ds_index, *idx, data)?;
        }
        Ok(())
    }

    /// Write multiple fixed-array chunks in a batch, compressing them in
    /// parallel when a filter pipeline is set and the `parallel` feature is on.
    ///
    /// The fixed-array analogue of [`write_chunks_batch`](Self::write_chunks_batch):
    /// chunks are addressed by grid coordinates rather than a linear index.
    /// `record_fixed_array_chunk` writes already-compressed bytes verbatim, so
    /// the parallel compressor is the only place a filter runs. Falls back to
    /// per-chunk [`write_chunk_fixed_array`](Self::write_chunk_fixed_array) when
    /// unfiltered or when `parallel` is off.
    pub fn write_chunks_fixed_array_batch(
        &self,
        ds_index: usize,
        chunks: &[(&[u64], &[u8])],
    ) -> IoResult<()> {
        let ds = self.ds(ds_index);
        let _op = ds.op.lock();
        self.write_chunks_fixed_array_batch_inner(ds_index, chunks)
    }

    /// [`Self::write_chunks_fixed_array_batch`] body; the caller holds the
    /// dataset's op lock or the writer exclusively.
    pub(crate) fn write_chunks_fixed_array_batch_inner(
        &self,
        ds_index: usize,
        chunks: &[(&[u64], &[u8])],
    ) -> IoResult<()> {
        #[cfg(feature = "parallel")]
        {
            // Clone the pipeline out under a brief slot guard so the parallel
            // compression below runs off the lock.
            let pipeline = self.ds(ds_index).lock().filter_pipeline.clone();
            if let Some(ref pipeline) = pipeline {
                let chunk_data: Vec<Vec<u8>> = chunks.iter().map(|(_, d)| d.to_vec()).collect();
                // Same single owner as the EA batch: apply_filters_parallel
                // propagates a filter error instead of storing raw bytes under a
                // filter_mask that claims the pipeline ran. Ok here means every
                // chunk compressed fully, so filter_mask = 0 is truthful.
                let compressed = filter::apply_filters_parallel(pipeline, &chunk_data)?;
                for ((coords, _), compressed_data) in chunks.iter().zip(compressed.iter()) {
                    self.record_fixed_array_chunk(ds_index, coords, compressed_data, 0)?;
                }
                return Ok(());
            }
        }
        // Fallback: sequential (write_chunk_fixed_array_inner compresses per
        // chunk).
        for (coords, data) in chunks {
            self.write_chunk_fixed_array_inner(ds_index, coords, data)?;
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
        &self,
        index: usize,
        chunk_idx: u64,
        compressed_data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.write_compressed_chunk_inner(index, chunk_idx, compressed_data, filter_mask)
    }

    /// [`Self::write_compressed_chunk`] body; the caller holds the dataset's
    /// op lock or the writer exclusively.
    pub(crate) fn write_compressed_chunk_inner(
        &self,
        index: usize,
        chunk_idx: u64,
        compressed_data: &[u8],
        filter_mask: u32,
    ) -> IoResult<()> {
        if self.ds(index).lock().filter_pipeline.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "write_compressed_chunk requires a filtered dataset (no slot for \
                 a compressed size or filter mask on an unfiltered chunk index)"
                    .into(),
            ));
        }
        self.record_ea_chunk(index, chunk_idx, compressed_data, filter_mask)
    }

    /// Extend the dimensions of a chunked dataset.
    pub fn extend_dataset(&self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.extend_dataset_inner(index, new_dims)
    }

    /// [`Self::extend_dataset`] body; the caller holds the dataset's op lock
    /// or the writer exclusively.
    pub(crate) fn extend_dataset_inner(&self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = self.ds(index);
        let mut m = ds.lock();
        let is_unindexed = m.chunked.is_none() && m.fixed_array.is_none() && m.btree_v2.is_none();
        if is_unindexed {
            return Err(crate::io::IoError::InvalidState(
                "can only extend chunked datasets".into(),
            ));
        }
        if new_dims.len() != m.dataspace.dims.len() {
            return Err(crate::io::IoError::InvalidState(format!(
                "extend_dataset rank mismatch: dataset has {} dimensions, got {}",
                m.dataspace.dims.len(),
                new_dims.len()
            )));
        }
        // The chunk index and append buffers assume the logical size only
        // grows; shrinking below already-written data desynchronizes them.
        for (d, (&new, &cur)) in new_dims.iter().zip(&m.dataspace.dims).enumerate() {
            if new < cur {
                return Err(crate::io::IoError::InvalidState(format!(
                    "extend_dataset cannot shrink dimension {d} from {cur} to {new}"
                )));
            }
            // An absent maximum shape means the shape is fixed (libhdf5
            // defaults maxdims to dims at creation), so any growth exceeds it.
            match m.dataspace.max_dims {
                Some(ref max) if new > max[d] => {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "extend_dataset dimension {d} ({new}) exceeds the maximum {}",
                        max[d]
                    )));
                }
                None if new > cur => {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "extend_dataset dimension {d} ({new}) exceeds the maximum {cur}: \
                         a dataset without a stored maximum shape is fixed at its extent"
                    )));
                }
                _ => {}
            }
        }
        if m.dataspace.dims != new_dims {
            m.dataspace.dims = new_dims.to_vec();
            m.extent_dirty = true;
        }
        Ok(())
    }

    /// Set the logical extent of a chunked dataset, growing **or shrinking**
    /// any dimension (unlike [`extend_dataset`](Self::extend_dataset), which
    /// only grows).
    ///
    /// A shrink prunes the stored chunks the way libhdf5's
    /// `H5D__chunk_prune_by_extent` (H5Dchunk.c) does: a chunk entirely
    /// beyond the new extent leaves the chunk index and its block is freed
    /// for reuse (kept under SWMR, where a live reader may still hold its
    /// address — the rule `H5Dearray.c` applies in `idx_remove`), and a
    /// chunk the new extent cuts through has its out-of-extent region
    /// overwritten with the fill value, so growing the extent back exposes
    /// fill values rather than the stale data.
    pub fn set_dataset_extent(&self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        let old_dims = {
            let m = ds.lock();
            let is_unindexed =
                m.chunked.is_none() && m.fixed_array.is_none() && m.btree_v2.is_none();
            if is_unindexed {
                return Err(crate::io::IoError::InvalidState(
                    "can only set the extent of chunked datasets".into(),
                ));
            }
            if new_dims.len() != m.dataspace.dims.len() {
                return Err(crate::io::IoError::InvalidState(format!(
                    "set_extent rank mismatch: dataset has {} dimensions, got {}",
                    m.dataspace.dims.len(),
                    new_dims.len()
                )));
            }
            // A shrink can cut into buffered rows, whose recorded base would
            // then point past the extent; refuse rather than reconcile.
            if m.append.is_some() {
                return Err(crate::io::IoError::InvalidState(
                    "set_extent cannot run while the dataset has buffered appends; \
                     flush them first"
                        .into(),
                ));
            }
            // An absent maximum shape means the shape is fixed (libhdf5
            // defaults maxdims to dims at creation), so growth is bounded by
            // the extent.
            match m.dataspace.max_dims {
                Some(ref max) => {
                    for (d, (&new, &mx)) in new_dims.iter().zip(max).enumerate() {
                        if new > mx {
                            return Err(crate::io::IoError::InvalidState(format!(
                                "set_extent dimension {d} ({new}) exceeds the maximum {mx}"
                            )));
                        }
                    }
                }
                None => {
                    for (d, (&new, &cur)) in new_dims.iter().zip(&m.dataspace.dims).enumerate() {
                        if new > cur {
                            return Err(crate::io::IoError::InvalidState(format!(
                                "set_extent dimension {d} ({new}) exceeds the maximum {cur}: \
                                 a dataset without a stored maximum shape is fixed at its extent"
                            )));
                        }
                    }
                }
            }
            m.dataspace.dims.clone()
        };
        // A shrink strands chunks; prune them (and refill the straddlers)
        // *before* the dims update — chunk addressing uses the
        // maximum-extent grid, which the update does not change, and the
        // helpers re-lock the slot themselves.
        if new_dims.iter().zip(&old_dims).any(|(&n, &o)| n < o) {
            self.prune_chunks_beyond(index, new_dims)?;
        }
        let mut m = ds.lock();
        if m.dataspace.dims != new_dims {
            m.dataspace.dims = new_dims.to_vec();
            m.extent_dirty = true;
        }
        Ok(())
    }

    /// Remove and refill the chunks a shrink to `new_dims` strands — the
    /// libhdf5 `H5D__chunk_prune_by_extent` behavior. A chunk entirely
    /// beyond the new extent leaves the index and its block is freed (kept
    /// under SWMR, where a live reader may still hold its address); a chunk
    /// the extent cuts through gets its out-of-extent region refilled with
    /// the fill value, so a later regrow reads fill, not stale elements.
    ///
    /// Runs *before* the dims update: the index grid chunks are addressed in
    /// comes from the maximum extent, which a shrink never changes, so every
    /// stored entry still resolves. The caller holds the dataset's op lock.
    fn prune_chunks_beyond(&self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let geo = self.chunk_geometry(index)?;
        // A vlen dataset's elements are global-heap IDs: the pruned chunks
        // still reference live heap objects, so the walkers read each dead
        // chunk's bytes before freeing its block and the heap objects are
        // released here — otherwise every shrink strands its strings in the
        // file. `release_vlen_references` is a SWMR no-op, so the reads are
        // skipped under SWMR too.
        let collect_refs = !self.swmr_active && {
            let ds = self.ds(index);
            let m = ds.lock();
            matches!(
                m.datatype,
                DatatypeMessage::VarLenString { .. } | DatatypeMessage::VarLenSequence { .. }
            )
        };
        let (straddlers, dead_refs) = match geo.kind {
            ChunkIndexKind::ExtensibleArray => {
                self.prune_ea_chunks(index, &geo, new_dims, collect_refs)?
            }
            ChunkIndexKind::FixedArray => {
                self.prune_fa_chunks(index, &geo, new_dims, collect_refs)?
            }
            ChunkIndexKind::BtreeV2 => {
                self.prune_bt2_chunks(index, &geo, new_dims, collect_refs)?
            }
        };
        if !dead_refs.is_empty() {
            self.release_vlen_references(&dead_refs)?;
        }
        // Whole-chunk read-modify-write per straddler: an unfiltered chunk
        // rewrites in place, a filtered one re-places through `place_chunk`.
        let chunk_bytes = geo.chunk_bytes() as usize;
        for coords in straddlers {
            let Some(mut data) = self.read_chunk_at_coords(index, &coords)? else {
                continue;
            };
            let fill = self.new_chunk_buffer(index, chunk_bytes);
            let replaced = refill_chunk_beyond_extent(
                &mut data,
                &fill,
                &coords,
                &geo.chunk_dims,
                new_dims,
                geo.element_size as usize,
            );
            // Release before the write-back: a filtered straddler re-places
            // its block, and freed heap space must be visible to that
            // allocation (free-before-alloc, as everywhere else).
            if collect_refs && !replaced.is_empty() {
                self.release_vlen_references(&replaced)?;
            }
            self.write_chunk_at_coords(index, &coords, &data)?;
        }
        Ok(())
    }

    /// Extensible-array half of [`prune_chunks_beyond`](Self::prune_chunks_beyond):
    /// walk every slot the array has ever set, free and clear the entries of
    /// chunks entirely beyond `new_dims`, and return the grid coordinates of
    /// the chunks that straddle it, plus — when `collect_refs` — the dead
    /// chunks' element bytes so the caller can release their heap objects.
    fn prune_ea_chunks(
        &self,
        index: usize,
        geo: &ChunkGeometry,
        new_dims: &[u64],
        collect_refs: bool,
    ) -> IoResult<(Vec<Vec<u64>>, Vec<u8>)> {
        let ds = self.ds(index);
        // One slot guard for the whole walk, the `record_ea_chunk` pattern:
        // `self.handle`/`self.allocator`/`self.ctx` are disjoint fields.
        let mut m = ds.lock();
        let is_filtered = m.filter_pipeline.is_some();
        let pipeline = m.filter_pipeline.clone();
        let chunk_bytes = geo.chunk_bytes();
        let (ea_geo, max_nelmts_bits, chunk_size_len, max_idx) = {
            let c = m.chunked.as_ref().unwrap();
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
                c.ea_header.max_idx_set,
            )
        };

        let mut straddlers = Vec::new();
        let mut dead_refs = Vec::new();

        // The decoded data block the walk is currently inside, written back
        // when the walk leaves it (or ends) having cleared an entry.
        enum Dblk {
            Unfiltered(ExtensibleArrayDataBlock),
            Filtered(FilteredDataBlock),
        }
        let mut cache: Option<(u64, Dblk, bool)> = None;
        let flush = |cache: &mut Option<(u64, Dblk, bool)>| -> IoResult<()> {
            if let Some((addr, blk, dirty)) = cache.take() {
                if dirty {
                    let enc = match &blk {
                        Dblk::Unfiltered(d) => d.encode(&self.ctx, max_nelmts_bits),
                        Dblk::Filtered(d) => d.encode(&self.ctx, max_nelmts_bits, chunk_size_len),
                    };
                    self.handle.write_at(addr, &enc)?;
                }
            }
            Ok(())
        };
        // Consecutive slots resolve through the same super block, so keep
        // the last decode. Super blocks are only read here — clearing a
        // data-block element never moves the block — so it never dirties.
        let mut sblk_cache: Option<(usize, ExtensibleArraySuperBlock)> = None;

        let mut slot = 0u64;
        while slot < max_idx {
            let coords = crate::io::chunk_grid::coords_of(
                &geo.dims,
                geo.max_dims.as_deref(),
                &geo.chunk_dims,
                slot,
            )?;
            if !chunk_outside_extent(&coords, &geo.chunk_dims, new_dims) {
                if chunk_straddles_extent(&coords, &geo.chunk_dims, new_dims) {
                    straddlers.push(coords);
                }
                slot += 1;
                continue;
            }
            match ea_geo.locate(slot)? {
                EaLoc::Index { elem } => {
                    let c = m.chunked.as_mut().unwrap();
                    if is_filtered {
                        let fiblk = c.filt_iblk.as_mut().unwrap();
                        let e = fiblk.elements[elem];
                        if e.addr != UNDEF_ADDR {
                            if collect_refs {
                                if let Some(bytes) = self.read_chunk_block(
                                    pipeline.as_ref(),
                                    e.addr,
                                    e.nbytes,
                                    e.filter_mask,
                                )? {
                                    dead_refs.extend_from_slice(&bytes);
                                }
                            }
                            if !self.swmr_active {
                                self.allocator.free(e.addr, e.nbytes);
                            }
                            fiblk.elements[elem] = FilteredChunkEntry {
                                addr: UNDEF_ADDR,
                                nbytes: 0,
                                filter_mask: 0,
                            };
                        }
                    } else {
                        let a = c.ea_iblk.elements[elem];
                        if a != UNDEF_ADDR {
                            if collect_refs {
                                if let Some(bytes) =
                                    self.read_chunk_block(pipeline.as_ref(), a, chunk_bytes, 0)?
                                {
                                    dead_refs.extend_from_slice(&bytes);
                                }
                            }
                            if !self.swmr_active {
                                self.allocator.free(a, chunk_bytes);
                            }
                            c.ea_iblk.elements[elem] = UNDEF_ADDR;
                        }
                    }
                    slot += 1;
                }
                EaLoc::Dblk(l) => {
                    if l.paged {
                        return Err(crate::io::IoError::InvalidState(format!(
                            "chunk index {slot} lives in a paged extensible-array \
                             data block, which is not yet supported"
                        )));
                    }
                    let dblk_start = slot - l.offset_in_dblk;
                    let dblk_end = dblk_start + l.dblk_nelmts;
                    // Resolve the data block's address; an undefined super or
                    // data block means nothing in its whole element range was
                    // ever written, so the walk skips the range.
                    let dblk_addr = {
                        let c = m.chunked.as_ref().unwrap();
                        match l.path {
                            EaDblkPath::Direct { idx } => {
                                if is_filtered {
                                    c.filt_iblk.as_ref().unwrap().dblk_addrs[idx]
                                } else {
                                    c.ea_iblk.dblk_addrs[idx]
                                }
                            }
                            EaDblkPath::ViaSblk {
                                sblk_off,
                                local_dblk,
                                ndblks_in_sblk,
                                ..
                            } => {
                                let sblk_addr = if is_filtered {
                                    c.filt_iblk.as_ref().unwrap().sblk_addrs[sblk_off]
                                } else {
                                    c.ea_iblk.sblk_addrs[sblk_off]
                                };
                                if sblk_addr == UNDEF_ADDR {
                                    UNDEF_ADDR
                                } else {
                                    if sblk_cache.as_ref().map(|&(o, _)| o) != Some(sblk_off) {
                                        let buf = self.handle.read_at_most(sblk_addr, 65536)?;
                                        let sb = ExtensibleArraySuperBlock::decode(
                                            &buf,
                                            &self.ctx,
                                            max_nelmts_bits,
                                            ndblks_in_sblk,
                                            0,
                                        )?;
                                        sblk_cache = Some((sblk_off, sb));
                                    }
                                    sblk_cache.as_ref().unwrap().1.dblk_addrs[local_dblk]
                                }
                            }
                        }
                    };
                    if dblk_addr == UNDEF_ADDR {
                        slot = dblk_end;
                        continue;
                    }
                    if cache.as_ref().map(|&(a, _, _)| a) != Some(dblk_addr) {
                        flush(&mut cache)?;
                        let buf = self.handle.read_at_most(dblk_addr, 65536)?;
                        let blk = if is_filtered {
                            Dblk::Filtered(FilteredDataBlock::decode(
                                &buf,
                                &self.ctx,
                                max_nelmts_bits,
                                l.dblk_nelmts as usize,
                                chunk_size_len,
                            )?)
                        } else {
                            Dblk::Unfiltered(ExtensibleArrayDataBlock::decode(
                                &buf,
                                &self.ctx,
                                max_nelmts_bits,
                                l.dblk_nelmts as usize,
                            )?)
                        };
                        cache = Some((dblk_addr, blk, false));
                    }
                    let (_, blk, dirty) = cache.as_mut().unwrap();
                    match blk {
                        Dblk::Filtered(d) => {
                            let e = d.elements[l.offset_in_dblk as usize];
                            if e.addr != UNDEF_ADDR {
                                if collect_refs {
                                    if let Some(bytes) = self.read_chunk_block(
                                        pipeline.as_ref(),
                                        e.addr,
                                        e.nbytes,
                                        e.filter_mask,
                                    )? {
                                        dead_refs.extend_from_slice(&bytes);
                                    }
                                }
                                if !self.swmr_active {
                                    self.allocator.free(e.addr, e.nbytes);
                                }
                                d.elements[l.offset_in_dblk as usize] = FilteredChunkEntry {
                                    addr: UNDEF_ADDR,
                                    nbytes: 0,
                                    filter_mask: 0,
                                };
                                *dirty = true;
                            }
                        }
                        Dblk::Unfiltered(d) => {
                            let a = d.elements[l.offset_in_dblk as usize];
                            if a != UNDEF_ADDR {
                                if collect_refs {
                                    if let Some(bytes) =
                                        self.read_chunk_block(pipeline.as_ref(), a, chunk_bytes, 0)?
                                    {
                                        dead_refs.extend_from_slice(&bytes);
                                    }
                                }
                                if !self.swmr_active {
                                    self.allocator.free(a, chunk_bytes);
                                }
                                d.elements[l.offset_in_dblk as usize] = UNDEF_ADDR;
                                *dirty = true;
                            }
                        }
                    }
                    slot += 1;
                }
            }
        }
        flush(&mut cache)?;
        Ok((straddlers, dead_refs))
    }

    /// Fixed-array half of [`prune_chunks_beyond`](Self::prune_chunks_beyond):
    /// the whole element array is in memory and flushed at close, so
    /// clearing an entry is pure bookkeeping.
    fn prune_fa_chunks(
        &self,
        index: usize,
        geo: &ChunkGeometry,
        new_dims: &[u64],
        collect_refs: bool,
    ) -> IoResult<(Vec<Vec<u64>>, Vec<u8>)> {
        let ds = self.ds(index);
        let mut m = ds.lock();
        let is_filtered = m.filter_pipeline.is_some();
        let pipeline = m.filter_pipeline.clone();
        let chunk_bytes = geo.chunk_bytes();
        let mut straddlers = Vec::new();
        let mut dead_refs = Vec::new();
        let fa = m.fixed_array.as_mut().unwrap();
        let nslots = if is_filtered {
            fa.fa_dblk.filtered_elements.len()
        } else {
            fa.fa_dblk.elements.len()
        };
        for lidx in 0..nslots {
            let (addr, stored, mask) = if is_filtered {
                let e = &fa.fa_dblk.filtered_elements[lidx];
                (e.address, e.chunk_size, e.filter_mask)
            } else {
                (fa.fa_dblk.elements[lidx], chunk_bytes, 0)
            };
            if addr == UNDEF_ADDR {
                continue;
            }
            let coords = crate::io::chunk_grid::coords_of(
                &geo.dims,
                geo.max_dims.as_deref(),
                &geo.chunk_dims,
                lidx as u64,
            )?;
            if chunk_outside_extent(&coords, &geo.chunk_dims, new_dims) {
                if collect_refs {
                    if let Some(bytes) =
                        self.read_chunk_block(pipeline.as_ref(), addr, stored, mask)?
                    {
                        dead_refs.extend_from_slice(&bytes);
                    }
                }
                if !self.swmr_active {
                    self.allocator.free(addr, stored);
                }
                if is_filtered {
                    fa.fa_dblk.filtered_elements[lidx] = FixedArrayFilteredChunkElement {
                        address: UNDEF_ADDR,
                        chunk_size: 0,
                        filter_mask: 0,
                    };
                } else {
                    fa.fa_dblk.elements[lidx] = UNDEF_ADDR;
                }
            } else if chunk_straddles_extent(&coords, &geo.chunk_dims, new_dims) {
                straddlers.push(coords);
            }
        }
        Ok((straddlers, dead_refs))
    }

    /// V2-B-tree half of [`prune_chunks_beyond`](Self::prune_chunks_beyond):
    /// drop the records of chunks beyond the extent — the next flush
    /// re-serializes the smaller tree over the node pool and releases the
    /// surplus node blocks.
    fn prune_bt2_chunks(
        &self,
        index: usize,
        geo: &ChunkGeometry,
        new_dims: &[u64],
        collect_refs: bool,
    ) -> IoResult<(Vec<Vec<u64>>, Vec<u8>)> {
        let ds = self.ds(index);
        let mut m = ds.lock();
        let pipeline = m.filter_pipeline.clone();
        let chunk_bytes = geo.chunk_bytes();
        let swmr = self.swmr_active;
        let mut straddlers = Vec::new();
        let mut dead_refs = Vec::new();
        let bt2 = m.btree_v2.as_mut().unwrap();
        if bt2.index.filtered {
            let records = std::mem::take(&mut bt2.index.filtered_records);
            let mut kept = Vec::with_capacity(records.len());
            for r in records {
                if chunk_outside_extent(&r.scaled_offsets, &geo.chunk_dims, new_dims) {
                    if collect_refs {
                        if let Some(bytes) = self.read_chunk_block(
                            pipeline.as_ref(),
                            r.chunk_address,
                            r.chunk_size,
                            r.filter_mask,
                        )? {
                            dead_refs.extend_from_slice(&bytes);
                        }
                    }
                    if !swmr {
                        self.allocator.free(r.chunk_address, r.chunk_size);
                    }
                } else {
                    if chunk_straddles_extent(&r.scaled_offsets, &geo.chunk_dims, new_dims) {
                        straddlers.push(r.scaled_offsets.clone());
                    }
                    kept.push(r);
                }
            }
            bt2.index.filtered_records = kept;
        } else {
            let records = std::mem::take(&mut bt2.index.records);
            let mut kept = Vec::with_capacity(records.len());
            for r in records {
                if chunk_outside_extent(&r.scaled_offsets, &geo.chunk_dims, new_dims) {
                    if collect_refs {
                        if let Some(bytes) = self.read_chunk_block(
                            pipeline.as_ref(),
                            r.chunk_address,
                            chunk_bytes,
                            0,
                        )? {
                            dead_refs.extend_from_slice(&bytes);
                        }
                    }
                    if !swmr {
                        self.allocator.free(r.chunk_address, chunk_bytes);
                    }
                } else {
                    if chunk_straddles_extent(&r.scaled_offsets, &geo.chunk_dims, new_dims) {
                        straddlers.push(r.scaled_offsets.clone());
                    }
                    kept.push(r);
                }
            }
            bt2.index.records = kept;
        }
        Ok((straddlers, dead_refs))
    }

    /// Flush a chunked dataset's index structures to disk (durable).
    ///
    /// Writes the index blocks and issues an `fdatasync` so the data is
    /// durable — the guarantee SWMR readers and standalone callers rely on.
    pub fn flush_dataset(&self, index: usize) -> IoResult<()> {
        let ds = self.ds(index);
        let _op = ds.op.lock();
        self.flush_dataset_synced(index, true)
    }

    /// Flush a chunked dataset's index structures, syncing only if `sync`.
    ///
    /// `finalize` threads its own durability choice here so that a
    /// [`close_no_sync`](Self::close_no_sync) skips this per-dataset
    /// `sync_data` too — otherwise gating only the final `sync_all` would
    /// leave one `fdatasync` per indexed dataset and defeat the fast close.
    fn flush_dataset_synced(&self, index: usize, sync: bool) -> IoResult<()> {
        // Hold one slot guard for the whole method; `self.handle`/`self.ctx`/
        // `self.allocator` below touch disjoint fields.
        let ds = self.ds(index);
        let mut m = ds.lock();

        // EA-indexed dataset
        if let Some(ref chunked) = m.chunked {
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
            if sync {
                self.handle.sync_data()?;
            }
            return Ok(());
        }

        // Fixed-array-indexed dataset
        if let Some(ref fa) = m.fixed_array {
            let dblk_encoded = encode_fixed_array_dblk(&self.ctx, &fa.fa_header, &fa.fa_dblk);
            self.handle.write_at(fa.fa_dblk_addr, &dblk_encoded)?;
            let hdr_encoded = fa.fa_header.encode(&self.ctx);
            self.handle.write_at(fa.fa_header_addr, &hdr_encoded)?;
            if sync {
                self.handle.sync_data()?;
            }
            return Ok(());
        }

        // BT2-indexed dataset
        if let Some(ref bt2) = m.btree_v2 {
            // Bulk-load the index into fixed-size nodes and lay them over the
            // dataset's block pool. Because every node is the same size, the
            // blocks already on disk are reused in place and only the shortfall
            // is allocated — the pool is the single owner of these addresses,
            // so no flush leaves a block behind. The addresses a reader already
            // holds stay valid, which is also what SWMR needs.
            let tree = bt2.index.build_tree(&self.ctx);
            let mut node_addrs = bt2.node_addrs.clone();
            while node_addrs.len() < tree.nodes.len() {
                node_addrs.push(self.allocator.allocate(tree.node_size as u64));
            }
            // A tree with fewer nodes than last flush releases the surplus
            // rather than leaving it recorded and unreachable, so the pool is
            // exactly one block per node whichever way the count moved. Under
            // SWMR a reader may still hold a header naming those blocks, so
            // keep them out of the free list — the same rule `place_chunk`
            // applies to a relocated chunk.
            for addr in node_addrs.split_off(tree.nodes.len()) {
                if !self.swmr_active {
                    self.allocator.free(addr, tree.node_size as u64);
                }
            }

            for (image, &addr) in tree.encode(&self.ctx, &node_addrs).iter().zip(&node_addrs) {
                self.handle.write_at(addr, image)?;
            }

            // The root is the last node the bulk load emits.
            let root_addr = match tree.nodes.len() {
                0 => UNDEF_ADDR,
                n => node_addrs[n - 1],
            };
            let hdr_encoded = tree.header(root_addr).encode(&self.ctx);
            self.handle.write_at(bt2.bt2_header_addr, &hdr_encoded)?;

            m.btree_v2.as_mut().unwrap().node_addrs = node_addrs;

            if sync {
                self.handle.sync_data()?;
            }
            return Ok(());
        }

        Ok(())
    }

    /// Finalize and close the file.
    ///
    /// Writes the dataset object headers, root group object header, and
    /// superblock. After this call the file is a valid HDF5 file.
    pub fn close(mut self) -> IoResult<()> {
        // Mark closed BEFORE finalizing: finalize writes external truth
        // (object headers + superblock) and must run exactly once. If we
        // finalized first and it failed, the `?` would return with `closed`
        // still false, and dropping `self` would re-run `finalize` a second
        // time over a half-written file (and print the "call close()" notice
        // the caller already heeded). Committing to the close path first makes
        // `Drop` (the only other finalize site) a no-op regardless of outcome,
        // so the error is reported exactly once via this `Result`.
        self.closed = true;
        self.finalize(true)
    }

    /// Finalize and close the file without a final `fsync`.
    ///
    /// Identical to [`close`](Self::close) — the same object headers and
    /// superblock are written, so on return the file is a complete, valid HDF5
    /// file readable by any process — except that the trailing `sync_all`
    /// (fsync) is skipped. The bytes are handed to the OS but are not
    /// guaranteed durable against power loss or an OS crash until the OS
    /// flushes its page cache; a normal process exit or a same-machine reader
    /// sees the full file regardless.
    ///
    /// This trades durability for speed: `sync_all` typically dominates close
    /// latency, so bulk writers that do not need crash durability (the file can
    /// be regenerated) can use this to avoid that cost. Use [`close`](Self::close)
    /// when durability matters. `Drop` always finalizes durably, so a writer
    /// finalized this way must reach `close_no_sync` explicitly.
    pub fn close_no_sync(mut self) -> IoResult<()> {
        // Same close-once discipline as `close`: commit to the close path
        // before finalizing so `Drop` cannot re-run `finalize` on failure.
        self.closed = true;
        self.finalize(false)
    }

    /// Provide mutable access to the underlying file handle.
    pub fn handle(&mut self) -> &mut FileHandle {
        &mut self.handle
    }

    /// Return the current end-of-file offset.
    pub fn eof(&self) -> u64 {
        self.allocator.eof()
    }

    /// The superblock version this file will be written with.
    ///
    /// `H5F__super_init` takes the oldest version that can describe the file
    /// and raises it to the one the file's library-version low bound implies:
    /// `super_vers = MAX(super_vers, HDF5_superblock_ver_bounds[low_bound])`,
    /// with the bounds table reading 0, 2, 3, 3, 3, 3, 3 for EARLIEST, V18,
    /// V110, V112, V114, V200, LATEST (H5Fsuper.c:68, :1128-1154). This crate
    /// has no version-bound property list, so the bound is read back from
    /// what it writes:
    ///
    /// * The floor is `H5F_LIBVER_V18`, hence version 2. Every group this
    ///   writer emits is a link-message group, which libhdf5 only writes at a
    ///   low bound of V18 or newer (`use_at_least_v18`, H5Gobj.c:179), and
    ///   every object header it emits is version 2, which `H5O_obj_ver_bounds`
    ///   likewise puts at V18 (H5Oint.c:125). A version-0 superblock over
    ///   this content would claim a file libhdf5 1.6 can read, and no libhdf5
    ///   writes that combination.
    /// * A chunked dataset — extensible array, fixed array or version-2
    ///   B-tree, all reached through a version-4 or -5 data layout message —
    ///   raises the bound to V110 (`H5O_layout_ver_bounds`, H5Dlayout.c:44),
    ///   hence version 3.
    /// * SWMR writes version 3 outright (H5Fsuper.c:1129), and so does the
    ///   2.0 format [`set_libver_latest`](Self::set_libver_latest) asks for.
    ///
    /// A reopened file starts from the version it already carries, so an
    /// append that adds nothing newer leaves that version alone.
    fn superblock_version_for(&self, flags: u8) -> u8 {
        if self.is_legacy() {
            // A classic file keeps the version it came with. Nothing this
            // session can add to it reaches past that version: the objects it
            // creates get version-1 headers and symbol-table links, and every
            // feature that would raise the bound — chunked storage, SWMR, the
            // 2.0 format — is refused where the caller asks for it.
            return self.superblock_version_base;
        }
        let mut version = self.superblock_version_base.max(SUPERBLOCK_V2);
        if self.libver == LibverBound::V200
            || self.swmr_active
            || flags & FLAG_SWMR_WRITE != 0
            || self.has_chunked_dataset()
        {
            version = version.max(SUPERBLOCK_V3);
        }
        version
    }

    /// Whether any dataset still in the file is chunked — the three index
    /// markers `build_dataset_header` turns into a version-4/5 data layout
    /// message, and nothing else it can emit reaches that version.
    fn has_chunked_dataset(&self) -> bool {
        self.dataset_refs().iter().any(|d| {
            let m = d.lock();
            !m.deleted && (m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some())
        })
    }

    /// Write the superblock at offset 0 with the given flags.
    ///
    /// Requires that the root group has already been written (via `finalize`
    /// or `finalize_for_swmr`).
    pub fn write_superblock(&mut self, flags: u8) -> IoResult<()> {
        let root_addr = self
            .root_group_addr
            .ok_or_else(|| crate::io::IoError::InvalidState("root group not yet written".into()))?;
        // The userblock this file was opened with. `H5F__super_read` prefers
        // the located address over this field, but `H5Pget_userblock` reports
        // it, so a rewrite that zeroed it would hide the block from every
        // reader that asks for its size.
        let base = self.handle.base();
        // The end of file is the one address in the superblock measured from
        // the start of the *file* rather than from the base: `H5F__super_read`
        // sets the EOA to `stored_eof - base_addr` (H5Fsuper.c:635) and calls
        // the file truncated when `eof + base_addr < stored_eof` (:573). The
        // allocator counts in the based space, so the userblock is added back.
        let eof = self.allocator.eof() + base;
        if let Some(legacy) = self.legacy.as_deref() {
            // Re-emitted, not rebuilt: the "K" ranks, the userblock size and
            // the driver info address are recorded nowhere else in the file,
            // and every node width in it is derived from the ranks. Only the
            // three things this session can have changed are recomputed.
            let root_stab = legacy.written.lock().get(&LinkScope::Root).copied();
            let mut sb = legacy.superblock.clone();
            sb.version = self.superblock_version_for(flags);
            sb.file_consistency_flags = flags as u32;
            sb.end_of_file_address = eof;
            sb.root_symbol_table_entry.obj_header_addr = root_addr;
            // `H5G__stab_valid` (H5Groot.c) reads this pair back and compares
            // it against the root header's Symbol Table message, repairing the
            // superblock when they disagree. Writing the pair that message now
            // names is what keeps the file from needing that repair. A root
            // that keeps its links in messages has no such pair and no entry
            // in `written`, and gets `H5G_NOTHING_CACHED` — what libhdf5
            // writes for the same root.
            sb.root_symbol_table_entry.cache = match root_stab {
                Some(s) => SymbolTableCache::SymbolTable {
                    btree_addr: s.btree_addr,
                    heap_addr: s.heap_addr,
                },
                None => SymbolTableCache::Nothing,
            };
            self.handle.write_at(0, &sb.encode())?;
            return Ok(());
        }
        let sb = SuperblockV2V3 {
            version: self.superblock_version_for(flags),
            sizeof_offsets: self.ctx.sizeof_addr,
            sizeof_lengths: self.ctx.sizeof_size,
            file_consistency_flags: flags,
            base_address: base,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: eof,
            root_group_object_header_address: root_addr,
        };
        self.handle.write_at(0, &sb.encode())?;
        Ok(())
    }

    /// Re-write a dataset's object header in place (SWMR update).
    ///
    /// The header must have been previously written via `finalize_for_swmr`.
    /// Only the dataspace dimensions change; the encoded size must not exceed
    /// the originally allocated space.
    pub fn write_dataset_header_inplace(&mut self, index: usize) -> IoResult<()> {
        // Scope the slot guard: `build_dataset_header` re-locks the same slot.
        let (addr, original_size) = {
            let ds = self.ds(index);
            let m = ds.lock();
            let addr = m.obj_header_written_addr.ok_or_else(|| {
                crate::io::IoError::InvalidState("dataset header not yet written".into())
            })?;
            (addr, m.obj_header_encoded_size)
        };

        let header = self.build_dataset_header(index);
        let encoded = self.encode_header(
            &header,
            self.object_link_count(HardLinkTarget::Dataset(index)),
            self.dataset_header_format(index),
        )?;

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
        // SWMR writes a version-3 superblock outright (H5Fsuper.c:1129), and
        // the status flags that mark a file SWMR-open live only in that
        // version. A classic file cannot carry either, so the session is
        // refused before its first header is written rather than closed as a
        // file that says nothing about the writer still attached to it.
        if self.is_legacy() {
            return Err(crate::io::IoError::Unsupported(
                "cannot start an SWMR session on this file: it is in the classic                  (version-0/1 superblock) format, and SWMR needs a version-3                  superblock to record that a writer is attached"
                    .into(),
            ));
        }
        // 0. Flush all chunked dataset index structures.
        for i in 0..self.dataset_count() {
            let is_indexed = {
                let ds = self.ds(i);
                let m = ds.lock();
                !m.deleted
                    && (m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some())
            };
            if is_indexed {
                self.flush_dataset(i)?;
            }
        }

        // 1. Write each dataset's object header (none for a dataset deleted
        // before start_swmr — its storage was freed at delete time).
        let live: Vec<usize> = (0..self.dataset_count())
            .filter(|&i| !self.ds(i).lock().deleted)
            .collect();
        self.prepare_dense_attributes(&live)?;
        for i in live {
            let nlink = self.object_link_count(HardLinkTarget::Dataset(i));
            let ds_header = self.build_dataset_header(i);
            let encoded = self.encode_header(&ds_header, nlink, self.dataset_header_format(i))?;
            let encoded_size = encoded.len();
            let addr = self.allocator.allocate(encoded_size as u64);
            self.handle.write_at(addr, &encoded)?;
            let ds = self.ds(i);
            let mut m = ds.lock();
            m.obj_header_addr = addr;
            m.obj_header_written_addr = Some(addr);
            m.obj_header_encoded_size = encoded_size;
            m.nlink_written = nlink;
        }

        // 1b. Group object headers. A hard link can point to a group whose
        // header is written later, so addresses are assigned in a first
        // pass (a header's encoded size is independent of the address
        // values it carries) and the content is written in a second.
        for gi in 0..self.group_count() {
            if self.grp(gi).lock().deleted {
                continue;
            }
            let rc = self.object_link_count(HardLinkTarget::Group(gi));
            let size = self
                .encode_header(
                    &self.build_group_header(gi),
                    rc,
                    self.group_header_format(gi),
                )?
                .len() as u64;
            self.grp(gi).lock().obj_header_addr = self.allocator.allocate(size);
        }
        // Every object header now has an address, so a link message names its
        // real target; and no group header has been written yet, so the Link
        // Info messages this lays out are the ones that reach the file.
        self.prepare_link_storage()?;
        for gi in 0..self.group_count() {
            if self.grp(gi).lock().deleted {
                continue;
            }
            let rc = self.object_link_count(HardLinkTarget::Group(gi));
            let encoded = self.encode_header(
                &self.build_group_header(gi),
                rc,
                self.group_header_format(gi),
            )?;
            let addr = self.grp(gi).lock().obj_header_addr;
            self.handle.write_at(addr, &encoded)?;
        }

        // 2. Write root group object header.
        let root_header = self.build_root_group_header();
        let root_encoded =
            self.encode_header(&root_header, 1, self.header_format(self.root_track_order))?;
        let root_encoded_size = root_encoded.len();
        let root_addr = self.allocator.allocate(root_encoded_size as u64);
        self.handle.write_at(root_addr, &root_encoded)?;
        self.root_group_addr = Some(root_addr);
        self.root_group_encoded_size = root_encoded_size;

        // 2b. Stamp object references now that every header has an address.
        self.apply_object_reference_fixups()?;

        // 3. Write superblock with SWMR flags.
        self.write_superblock(FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE)?;

        self.handle.sync_all()?;
        // Readers can now be following this file, so a chunk that moves must
        // leave its old block intact for whoever is still holding the previous
        // index (see `swmr_active`).
        self.swmr_active = true;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Flush every dataset's append buffer into the chunks it belongs to,
    /// through [`flush_append_buffer`](Self::flush_append_buffer): frames
    /// already in the chunk survive, and the rest of it reads back as the
    /// dataset's fill value (zeros when none is defined).
    fn flush_append_buffers(&mut self) -> IoResult<()> {
        for i in 0..self.dataset_count() {
            if self.ds(i).lock().deleted {
                continue;
            }
            self.flush_append_buffer(i)?;
        }
        Ok(())
    }

    /// Write all object headers and the superblock, producing a complete,
    /// valid HDF5 file.
    ///
    /// `sync == true` issues a final `sync_all` (fsync) so the bytes are
    /// durable against power loss / OS crash before returning. `sync == false`
    /// skips that fsync: the file is still fully written to the OS and readable
    /// by any process, but durability is left to the OS page-cache flush. This
    /// is the only difference between [`close`](Self::close) (durable) and
    /// [`close_no_sync`](Self::close_no_sync) (fast).
    fn finalize(&mut self, sync: bool) -> IoResult<()> {
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
        for i in 0..self.dataset_count() {
            let ds = self.ds(i);
            {
                let m = ds.lock();
                if m.deleted {
                    continue;
                }
                if m.obj_header_written_addr.is_some() && !m.storage_dirty() {
                    continue;
                }
                let is_indexed =
                    m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some();
                if !is_indexed {
                    continue;
                }
            }
            self.flush_dataset_synced(i, sync)?;
        }

        // Every header block this finalize supersedes — a reopened root or
        // group header, a modified dataset's reopened header — is freed
        // before its replacement is allocated, so the rewrite reuses the
        // block instead of growing the file on every open/close cycle.
        // Never under SWMR: a live reader may be walking the old headers,
        // the same rule `release_vlen_references` and `place_chunk` follow.
        // Hard links can alias one header under several names; the set keeps
        // an aliased block from entering the free list twice.
        let mut freed_headers = std::collections::HashSet::new();

        // 1. Write each dataset's object header (deleted datasets get none —
        // their storage was already freed at delete time). Which datasets get
        // one is settled first, because dense attribute storage is laid out
        // only for headers this finalize actually rewrites.
        let mut rewritten: Vec<usize> = Vec::new();
        for i in 0..self.dataset_count() {
            // Before the slot guard: `object_link_count` re-locks every
            // dataset and group slot, this one included.
            let nlink = self.object_link_count(HardLinkTarget::Dataset(i));
            let ds = self.ds(i);
            let mut m = ds.lock();
            if m.deleted {
                continue;
            }
            if m.obj_header_written_addr.is_some() {
                // An existing dataset from append mode keeps its header — and
                // everything that header names — unless this session changed
                // what the header says.
                if !m.header_stale_with(nlink) {
                    // Keep the original object header address for the root group link.
                    m.obj_header_addr = m.obj_header_written_addr.unwrap();
                    continue;
                }
                if !self.swmr_active && m.obj_header_encoded_size > 0 {
                    let old = m.obj_header_written_addr.take().unwrap();
                    if freed_headers.insert(old) {
                        self.allocator.free(old, m.obj_header_encoded_size as u64);
                    }
                    m.obj_header_encoded_size = 0;
                }
            }
            rewritten.push(i);
        }
        self.prepare_dense_attributes(&rewritten)?;
        for i in rewritten {
            let nlink = self.object_link_count(HardLinkTarget::Dataset(i));
            let ds_header = self.build_dataset_header(i);
            let encoded = self.encode_header(&ds_header, nlink, self.dataset_header_format(i))?;
            let addr = self.allocator.allocate(encoded.len() as u64);
            self.handle.write_at(addr, &encoded)?;
            let ds = self.ds(i);
            let mut m = ds.lock();
            m.obj_header_addr = addr;
            m.nlink_written = nlink;
        }

        // 1b. Group object headers. A hard link can point to a group whose
        // header is written later, so addresses are assigned in a first
        // pass (a header's encoded size is independent of the address
        // values it carries) and the content is written in a second.
        if !self.swmr_active {
            for gi in 0..self.group_count() {
                let grp = self.grp(gi);
                let mut g = grp.lock();
                if g.obj_header_encoded_size > 0 {
                    if let Some(old) = g.obj_header_written_addr.take() {
                        if freed_headers.insert(old) {
                            self.allocator.free(old, g.obj_header_encoded_size as u64);
                        }
                        g.obj_header_encoded_size = 0;
                    }
                }
            }
        }
        for gi in 0..self.group_count() {
            if self.grp(gi).lock().deleted {
                continue;
            }
            let rc = self.object_link_count(HardLinkTarget::Group(gi));
            let size = self
                .encode_header(
                    &self.build_group_header(gi),
                    rc,
                    self.group_header_format(gi),
                )?
                .len() as u64;
            self.grp(gi).lock().obj_header_addr = self.allocator.allocate(size);
        }
        // Every object header now has an address, so a link message names its
        // real target; and no group header has been written yet, so the Link
        // Info messages this lays out are the ones that reach the file.
        self.prepare_link_storage()?;
        for gi in 0..self.group_count() {
            if self.grp(gi).lock().deleted {
                continue;
            }
            let rc = self.object_link_count(HardLinkTarget::Group(gi));
            let encoded = self.encode_header(
                &self.build_group_header(gi),
                rc,
                self.group_header_format(gi),
            )?;
            let addr = self.grp(gi).lock().obj_header_addr;
            self.handle.write_at(addr, &encoded)?;
        }

        // 2. Write root group object header.
        if !self.swmr_active {
            if let Some((addr, len)) = self.superseded_root_header.take() {
                if freed_headers.insert(addr) {
                    self.allocator.free(addr, len);
                }
            }
        }
        let root_header = self.build_root_group_header();
        let root_encoded =
            self.encode_header(&root_header, 1, self.header_format(self.root_track_order))?;
        let root_addr = self.allocator.allocate(root_encoded.len() as u64);
        self.handle.write_at(root_addr, &root_encoded)?;
        self.root_group_addr = Some(root_addr);

        // 2b. Every object header now has an address, so the object
        // references waiting on one can be stamped into their datasets.
        self.apply_object_reference_fixups()?;

        // 3. Write superblock at offset 0.
        self.write_superblock(0)?;

        // Durability is opt-in per call: `close` passes `true`, `close_no_sync`
        // passes `false`, and `Drop` passes `true` so an un-`close`d writer is
        // still finalized durably by default.
        if sync {
            self.handle.sync_all()?;
        }
        Ok(())
    }

    fn build_dataset_header(&self, index: usize) -> ObjectHeader {
        // Compute the link count first: object_link_count re-locks dataset and
        // group slots (including this one), so it must run before we take this
        // dataset's slot guard — otherwise it would deadlock on the same slot.
        let rc = self.object_link_count(HardLinkTarget::Dataset(index));

        // Hold one slot guard for the whole header build.
        let ds = self.ds(index);
        let m = ds.lock();
        let mut header = ObjectHeader::new();
        header.times = touched_times(m.times);

        // Dataspace message (type 0x01)
        let ds_msg = m.dataspace.encode_for(&self.ctx, self.message_format());
        header.add_message(MSG_DATASPACE, 0x00, ds_msg);

        // Datatype message (type 0x03), flag 0x01 = constant
        let dt_msg = m.datatype.encode_at(&self.ctx, self.encoding_libver());
        header.add_message(MSG_DATATYPE, 0x01, dt_msg);

        // Fill Value message (type 0x05)
        let is_chunked = m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some();
        let alloc_time = if is_chunked { 3 } else { 2 }; // 3 = incremental, 2 = late
        let fv = if let Some(ref bytes) = m.fill_value {
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
        let fv_msg = fv.encode_for(self.message_format());
        header.add_message(MSG_FILL_VALUE, 0x00, fv_msg);

        // Data Layout message (type 0x08)
        let layout = if let Some(ref chunked) = m.chunked {
            let mut layout_dims = chunked.chunk_dims.clone();
            layout_dims.push(m.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_earray(
                m.layout_version,
                layout_dims,
                chunked.earray_params.clone(),
                chunked.ea_header_addr,
            )
        } else if let Some(ref fa) = m.fixed_array {
            let mut layout_dims = fa.chunk_dims.clone();
            layout_dims.push(m.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_farray(
                m.layout_version,
                layout_dims,
                FixedArrayParams::default_params(),
                fa.fa_header_addr,
            )
        } else if let Some(ref bt2) = m.btree_v2 {
            let mut layout_dims = bt2.chunk_dims.clone();
            layout_dims.push(m.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_btree_v2(
                m.layout_version,
                layout_dims,
                crate::format::messages::data_layout::Bt2Params {
                    node_size: bt2.index.node_size,
                    split_percent: bt2.index.split_percent,
                    merge_percent: bt2.index.merge_percent,
                },
                bt2.bt2_header_addr,
            )
        } else {
            DataLayoutMessage::contiguous(m.data_addr, m.data_size)
        };
        let layout_msg = layout.encode(&self.ctx);
        header.add_message(MSG_DATA_LAYOUT, 0x00, layout_msg);

        // Filter Pipeline message (type 0x0B) -- only if filters are configured
        if let Some(ref pipeline) = m.filter_pipeline {
            if !pipeline.filters.is_empty() {
                let filter_msg = pipeline.encode();
                header.add_message(MSG_FILTER_PIPELINE, 0x00, filter_msg);
            }
        }

        // Attribute Info (type 0x15) + attribute messages (type 0x0C)
        // A dataset has no links, so only attribute creation order can raise
        // its header past version 1 (`H5O__set_version`).
        let format = self.header_format(TrackOrder {
            links: CreationOrder::default(),
            attrs: m.track_attr_order,
        });
        self.emit_attributes(
            &mut header,
            AttrScope::Dataset(index),
            &m.attributes,
            m.track_attr_order,
            format,
        );

        self.emit_refcount(&mut header, rc, format);

        header
    }

    /// Build the object header for a subgroup.
    fn build_group_header(&self, group_idx: usize) -> ObjectHeader {
        let mut header = ObjectHeader::new();

        // Link Info (type 0x02) + Group Info (type 0x0A) + the links
        // themselves, compact or dense.
        // Snapshot what the header needs, then drop the slot guard: the calls
        // below re-lock group slots (including this one).
        let (attributes, track_order, times) = {
            let grp = self.grp(group_idx);
            let g = grp.lock();
            (g.attributes.clone(), g.track_order, g.times)
        };
        header.times = touched_times(times);

        let links = self.group_links(LinkScope::Group(group_idx), track_order.links);
        self.emit_links(
            &mut header,
            LinkScope::Group(group_idx),
            &links,
            track_order.links,
        );

        // Attribute Info (type 0x15) + attributes (type 0x0C) -- e.g. NeXus
        // `NX_class`.
        let format = self.header_format(track_order);
        self.emit_attributes(
            &mut header,
            AttrScope::Group(group_idx),
            &attributes,
            track_order.attrs,
            format,
        );

        self.emit_refcount(
            &mut header,
            self.object_link_count(HardLinkTarget::Group(group_idx)),
            format,
        );

        header
    }

    fn build_root_group_header(&self) -> ObjectHeader {
        let mut header = ObjectHeader::new();
        header.times = touched_times(self.root_times);

        // Link Info (type 0x02) + Group Info (type 0x0A) + the links
        // themselves, compact or dense.
        let links = self.group_links(LinkScope::Root, self.root_track_order.links);
        self.emit_links(
            &mut header,
            LinkScope::Root,
            &links,
            self.root_track_order.links,
        );

        // Root-level attributes
        let root_attributes = self.root_attributes.lock();
        self.emit_attributes(
            &mut header,
            AttrScope::Root,
            &root_attributes,
            self.root_track_order.attrs,
            self.header_format(self.root_track_order),
        );

        header
    }
}

impl Drop for Hdf5Writer {
    fn drop(&mut self) {
        if !self.closed {
            // Best-effort finalize on drop. Drop cannot return a Result, so a
            // failure here is otherwise invisible: it would leave a truncated
            // or unflushed file on disk while the caller believes the write
            // succeeded. Surface it on stderr instead of swallowing it.
            // Callers that need to handle the error must call
            // `H5File::close()` explicitly, which returns the Result.
            if let Err(e) = self.finalize(true) {
                eprintln!(
                    "rust-hdf5: failed to finalize HDF5 file on drop: {e}. \
                     The file may be incomplete or corrupt; call \
                     H5File::close() to handle this error explicitly."
                );
            }
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

    /// A group past the link phase change keeps its links in a fractal heap
    /// with a v2 B-tree name index. The reopen that rewrites that group's
    /// header lays a fresh pair out, so both blocks the old header named must
    /// come back to the allocator — every block of the heap, and the index
    /// header with its nodes.
    ///
    /// Asserted on the free list rather than on the file size: a reopen does
    /// not yet carry dense links forward, so the rewritten group's links (and
    /// the datasets they name) are dropped, and the file size that follows
    /// says more about that than about this.
    #[test]
    fn a_reopen_frees_the_dense_link_storage_its_rewrite_supersedes() {
        let path = temp_path("dense_link_reclaim");

        let writer = Hdf5Writer::create(&path).unwrap();
        writer.create_group("/", "run").unwrap();
        for i in 0..12 {
            writer
                .create_dataset(&format!("run/d{i:02}"), DatatypeMessage::i32_type(), &[2])
                .unwrap();
        }
        writer.close().unwrap();

        let writer = Hdf5Writer::open_append(&path).unwrap();
        let gidx = (0..writer.group_count())
            .find(|&g| writer.grp(g).lock().name == "/run")
            .expect("the reopen registered the group");
        let linfo = writer
            .superseded_dense
            .lock()
            .as_ref()
            .and_then(|s| s.links.get(&LinkScope::Group(gidx)).cloned())
            .expect("the reopen recorded the group's dense link storage");
        assert_ne!(linfo.fractal_heap_address, UNDEF_ADDR);
        assert_ne!(linfo.name_btree_address, UNDEF_ADDR);

        writer
            .release_superseded_dense_links(LinkScope::Group(gidx))
            .unwrap();
        let freed = writer.allocator.free_blocks();
        let covers = |addr: u64| {
            freed
                .iter()
                .any(|&(a, len)| addr >= a && addr < a.saturating_add(len))
        };
        assert!(covers(linfo.fractal_heap_address), "heap header: {freed:?}");
        assert!(covers(linfo.name_btree_address), "name index: {freed:?}");

        // And exactly once: the entry is gone, so the finalize that follows
        // cannot hand the same blocks back a second time.
        assert!(writer
            .superseded_dense
            .lock()
            .as_ref()
            .is_none_or(|s| s.links.is_empty()));
        writer
            .release_superseded_dense_links(LinkScope::Group(gidx))
            .unwrap();
        assert_eq!(writer.allocator.free_blocks(), freed);

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// The rewrite frees what it supersedes even when the replacement is not
    /// dense at all. An attribute set that drops back under `max_compact`
    /// goes into the object header, so nothing names the old heap any more —
    /// and a free driven by "the new set needs dense storage" would never
    /// reach this one.
    #[test]
    fn a_rewrite_that_drops_out_of_dense_storage_still_frees_it() {
        let path = temp_path("dense_attr_to_compact");
        let numeric = |name: &str| {
            AttributeMessage::scalar_numeric(
                name,
                DatatypeMessage::i32_type(),
                7i32.to_le_bytes().to_vec(),
            )
        };

        let writer = Hdf5Writer::create(&path).unwrap();
        for i in 0..12 {
            writer
                .add_root_attribute(numeric(&format!("a{i:02}")))
                .unwrap();
        }
        writer.close().unwrap();

        let writer = Hdf5Writer::open_append(&path).unwrap();
        let ainfo = writer
            .superseded_dense
            .lock()
            .as_ref()
            .and_then(|s| s.attrs.get(&AttrScope::Root).cloned())
            .expect("the reopen recorded the root's dense attribute storage");
        for i in 0..10 {
            writer
                .evict_attr(AttrTarget::Root, &format!("a{i:02}"))
                .unwrap();
        }
        assert!(!writer.attributes_need_dense(&writer.root_attributes.lock(), ObjectFormat::Modern));

        writer.prepare_dense_attributes(&[]).unwrap();
        let freed = writer.allocator.free_blocks();
        let covers = |addr: u64| {
            freed
                .iter()
                .any(|&(a, len)| addr >= a && addr < a.saturating_add(len))
        };
        assert!(covers(ainfo.fractal_heap_address), "heap header: {freed:?}");
        assert!(covers(ainfo.name_btree_address), "name index: {freed:?}");
        assert!(writer
            .superseded_dense
            .lock()
            .as_ref()
            .is_none_or(|s| s.attrs.is_empty()));

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Deleting a reopened object supersedes its dense storage as surely as
    /// rewriting one does: nothing in the finalized file names the heap, so
    /// the delete owner frees it through the same entry.
    #[test]
    fn deleting_a_reopened_group_frees_its_dense_attribute_storage() {
        let path = temp_path("dense_attr_delete");
        let numeric = |name: &str| {
            AttributeMessage::scalar_numeric(
                name,
                DatatypeMessage::i32_type(),
                7i32.to_le_bytes().to_vec(),
            )
        };

        let writer = Hdf5Writer::create(&path).unwrap();
        writer.create_group("/", "run").unwrap();
        for i in 0..12 {
            writer
                .set_attribute(AttrTarget::Group("/run"), numeric(&format!("a{i:02}")))
                .unwrap();
        }
        writer.close().unwrap();

        let writer = Hdf5Writer::open_append(&path).unwrap();
        let gidx = (0..writer.group_count())
            .find(|&g| writer.grp(g).lock().name == "/run")
            .expect("the reopen registered the group");
        let ainfo = writer
            .superseded_dense
            .lock()
            .as_ref()
            .and_then(|s| s.attrs.get(&AttrScope::Group(gidx)).cloned())
            .expect("the reopen recorded the group's dense attribute storage");

        writer.delete_group("/run").unwrap();
        let freed = writer.allocator.free_blocks();
        let covers = |addr: u64| {
            freed
                .iter()
                .any(|&(a, len)| addr >= a && addr < a.saturating_add(len))
        };
        assert!(covers(ainfo.fractal_heap_address), "heap header: {freed:?}");
        assert!(covers(ainfo.name_btree_address), "name index: {freed:?}");
        assert!(writer
            .superseded_dense
            .lock()
            .as_ref()
            .is_none_or(|s| s.attrs.is_empty()));

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// The charset rule is one owner shared by every vlen string writer:
    /// appends into an ASCII-declared dataset reject non-ASCII strings the
    /// same way the slice writer does, and a dataset whose elements are not
    /// vlen references at all is refused instead of overwritten with them.
    #[test]
    fn append_vlen_strings_checks_the_datatype_and_charset() {
        let path = temp_path("append_vlen_charset");

        let writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_appendable_vlen_string_dataset("d", 4, None)
            .unwrap();
        writer.ds(idx).lock().datatype = DatatypeMessage::vlen_string_ascii();
        let err = writer
            .append_vlen_strings(idx, &["ok", "안녕"])
            .unwrap_err();
        assert!(
            err.to_string().contains("is not ASCII"),
            "unexpected error: {err}"
        );
        writer.append_vlen_strings(idx, &["ok", "fine"]).unwrap();

        let nums = writer
            .create_chunked_dataset("n", DatatypeMessage::i32_type(), &[0], &[u64::MAX], &[4])
            .unwrap();
        let err = writer.append_vlen_strings(nums, &["x"]).unwrap_err();
        assert!(
            err.to_string()
                .contains("only for variable-length string datasets"),
            "unexpected error: {err}"
        );

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// `create_chunked_dataset` builds an extensible-array index unconditionally
    /// (the caller — the high-level dataset API — is the one that decides when
    /// two-or-more unlimited dimensions should go to a v2 B-tree instead), so
    /// its own guard is the last line of defense against a shape that index
    /// can't represent at all.
    #[test]
    fn create_chunked_dataset_rejects_two_unlimited_dimensions() {
        let path = temp_path("earray_two_unlimited");
        let writer = Hdf5Writer::create(&path).unwrap();
        let err = writer
            .create_chunked_dataset(
                "d",
                DatatypeMessage::i32_type(),
                &[4, 4],
                &[u64::MAX, u64::MAX],
                &[2, 2],
            )
            .unwrap_err();
        assert!(err.to_string().contains("at most one unlimited"), "{err}");
        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Every creator must enter through `begin_create`; the four that used
    /// to bypass it could push a second dataset under an existing name and
    /// emit an invalid file with two same-named links.
    #[test]
    fn every_creator_rejects_an_existing_dataset_name() {
        let path = temp_path("create_gate");

        let writer = Hdf5Writer::create(&path).unwrap();
        writer
            .create_dataset("d", DatatypeMessage::i32_type(), &[2])
            .unwrap();

        let attempts: [(&str, IoResult<usize>); 4] = [
            (
                "vlen_string",
                writer.create_vlen_string_dataset("d", &["x"], 1),
            ),
            ("vlen_bytes", writer.create_vlen_bytes_dataset("d", &[b"x"])),
            (
                "vlen_string_compressed",
                writer.create_vlen_string_dataset_compressed(
                    "d",
                    &["x"],
                    1,
                    FilterPipeline::deflate(6),
                ),
            ),
            (
                "chunked_compressed",
                writer.create_chunked_dataset_compressed(
                    "d",
                    DatatypeMessage::i32_type(),
                    &[0],
                    &[u64::MAX],
                    &[4],
                    6,
                ),
            ),
        ];
        for (which, res) in attempts {
            match res {
                Ok(_) => panic!("{which} accepted a duplicate name"),
                Err(e) => assert!(
                    e.to_string().contains("already exists"),
                    "{which}: unexpected error: {e}"
                ),
            }
        }

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// The `H5T_VLEN` length field counts base elements, so an image that is
    /// not a whole number of them has no length that reads back as what was
    /// handed over; it is refused at the call rather than stored truncated.
    #[test]
    fn vlen_sequence_refuses_a_partial_element() {
        let path = temp_path("vlen_partial_element");

        let writer = Hdf5Writer::create(&path).unwrap();
        let err = writer
            .create_vlen_sequence_dataset("d", DatatypeMessage::i32_type(), &[&[1u8, 2, 3, 4, 5]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("5 bytes"), "unexpected error: {err}");
        assert!(err.contains("4-byte elements"), "unexpected error: {err}");

        // The refusal is the length rule alone: the same base takes a whole
        // number of elements, and an empty sequence is a legal one.
        writer
            .create_vlen_sequence_dataset(
                "d",
                DatatypeMessage::i32_type(),
                &[&[1u8, 2, 3, 4], &[][..]],
            )
            .unwrap();

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A corrupt file can declare a zero-length chunk dimension; the
    /// superseded-reference read must reject it the way `write_slice` does,
    /// not divide by it.
    #[test]
    fn vlen_slice_rejects_a_zero_chunk_dimension() {
        let path = temp_path("vlen_slice_zero_chunk");

        let writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_appendable_vlen_string_dataset("d", 2, None)
            .unwrap();
        writer.append_vlen_strings(idx, &["a", "b"]).unwrap();
        writer.ds(idx).lock().chunked.as_mut().unwrap().chunk_dims[0] = 0;
        let err = writer.write_vlen_strings_slice(idx, 0, &["x"]).unwrap_err();
        assert!(
            err.to_string().contains("zero-length dimension"),
            "unexpected error: {err}"
        );

        writer.ds(idx).lock().chunked.as_mut().unwrap().chunk_dims[0] = 2;
        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A libhdf5-written collection can be 100% full — no free-space marker,
    /// content exactly the declared size. When a stale reference names an
    /// index that is not there, nothing is removed, and the collection must
    /// be left alone: re-encoding it at its declared size cannot fit the
    /// free-space marker and would fail the whole update.
    #[test]
    fn release_leaves_a_full_collection_it_removed_nothing_from() {
        use crate::format::global_heap::encode_vlen_reference;

        let path = temp_path("release_full_gcol");
        let writer = Hdf5Writer::create(&path).unwrap();

        // Hand-built full collection: 16-byte header + one 16+8-byte object,
        // declared size exactly 40, no free-space marker.
        let mut img = Vec::new();
        img.extend_from_slice(b"GCOL");
        img.push(1);
        img.extend_from_slice(&[0u8; 3]);
        img.extend_from_slice(&40u64.to_le_bytes());
        img.extend_from_slice(&1u16.to_le_bytes()); // object index 1
        img.extend_from_slice(&1u16.to_le_bytes()); // ref_count
        img.extend_from_slice(&0u32.to_le_bytes()); // reserved
        img.extend_from_slice(&8u64.to_le_bytes()); // data size
        img.extend_from_slice(b"deadbeef");
        assert_eq!(img.len(), 40);
        let addr = writer.allocator.allocate(img.len() as u64);
        writer.handle.write_at(addr, &img).unwrap();

        // The superseded reference names index 2, which the collection does
        // not hold — a no-op removal.
        let refs = encode_vlen_reference(3, addr, 2, &writer.ctx);
        writer.release_vlen_references(&refs).unwrap();
        assert_eq!(writer.handle.read_at(addr, 40).unwrap(), img);

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// The CWFS second pass (`H5F_cwfs_find_free_heap`): an object too big
    /// for the listed collection's remaining free space extends the
    /// collection in place — the file allocation grows off the end of the
    /// file (`H5MF_try_extend`) and the collection's declared size and
    /// free-space marker grow with it (`H5HG_extend`) — instead of opening
    /// a second collection.
    #[test]
    fn an_oversized_vlen_insert_extends_the_listed_collection() {
        use crate::format::global_heap::GlobalHeapCollection;

        let path = temp_path("cwfs_extend_tail");
        let writer = Hdf5Writer::create(&path).unwrap();
        // A small object opens a minimum-size (4096) listed collection —
        // the file's last allocation, so the extension grows the file end.
        let p1 = writer.insert_vlen_objects(&[b"hello".as_slice()]).unwrap();
        let big = vec![0x41u8; 5000]; // more than the ~4 KiB remaining
        let p2 = writer.insert_vlen_objects(&[big.as_slice()]).unwrap();
        assert_eq!(
            p2[0].0, p1[0].0,
            "the big object opened a second collection"
        );

        // The block on disk is one grown collection holding both objects.
        let img = writer.handle.read_at_most(p1[0].0, 65536).unwrap();
        let (gcol, csize) = GlobalHeapCollection::decode(&img, &writer.ctx).unwrap();
        assert!(csize > 4096, "declared size did not grow: {csize}");
        assert_eq!(gcol.objects.len(), 2);
        assert_eq!(gcol.objects[1].data, big);

        writer.close().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            bytes.windows(4).filter(|w| *w == b"GCOL").count(),
            1,
            "a second collection signature is in the file"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The non-tail counterpart: the collection is pinned away from the end
    /// of the file, but a released block starts right after it, so the
    /// extension consumes the front of that block (`H5MF_try_extend`'s
    /// free-section path) and the remainder stays reusable.
    #[test]
    fn extension_consumes_a_freed_block_after_the_collection() {
        use crate::format::global_heap::GlobalHeapCollection;

        let path = temp_path("cwfs_extend_freed");
        let writer = Hdf5Writer::create(&path).unwrap();
        let p1 = writer.insert_vlen_objects(&[b"hello".as_slice()]).unwrap();
        let addr = p1[0].0;
        // Land a block right after the collection, pin the file end past
        // it, then release it: extension must use the released space.
        let spacer = writer.allocator.allocate(8192);
        assert_eq!(spacer, addr + 4096, "spacer not adjacent; layout changed");
        writer.allocator.allocate(8);
        writer.allocator.free(spacer, 8192);

        let big = vec![0x42u8; 5000];
        let p2 = writer.insert_vlen_objects(&[big.as_slice()]).unwrap();
        assert_eq!(p2[0].0, addr, "the big object opened a second collection");

        let img = writer.handle.read_at_most(addr, 65536).unwrap();
        let (gcol, csize) = GlobalHeapCollection::decode(&img, &writer.ctx).unwrap();
        assert_eq!(csize, 8192, "grew by max(size, shortfall) = 4096");
        assert_eq!(gcol.objects.len(), 2);

        // The remainder of the released block is still allocatable.
        assert_eq!(
            writer.allocator.allocate(4096),
            addr + 8192,
            "the freed block's tail was lost"
        );
        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Issue #10: a reopen-and-replace loop on a vlen string must not grow
    /// the file. The superseded heap objects are freed *before* the
    /// replacement is allocated, so each session reuses the block it just
    /// released even though the free list starts empty on reopen. The old
    /// free-after-alloc order failed this by one collection per session.
    #[test]
    fn vlen_replace_across_reopen_keeps_the_file_flat() {
        let path = temp_path("vlen_reopen_flat");
        let payload_a = "a".repeat(64 * 1024);
        let payload_b = "b".repeat(64 * 1024);

        let writer = Hdf5Writer::create(&path).unwrap();
        writer
            .create_vlen_string_dataset("notes", &["initial"], 1)
            .unwrap();
        writer.close().unwrap();

        let mut sizes = Vec::new();
        for i in 0..8 {
            let writer = Hdf5Writer::open_append(&path).unwrap();
            let payload = if i % 2 == 0 { &payload_a } else { &payload_b };
            writer
                .write_vlen_strings_slice(0, 0, &[payload.as_str()])
                .unwrap();
            writer.close().unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        // The first replacement grows the file once (the initial collection
        // cannot hold 64 KiB); every later equal-size replacement must land
        // in the block its own session just freed.
        assert_eq!(&sizes[1..], &vec![sizes[0]; 7][..], "sizes: {sizes:?}");

        // The reused blocks still form a valid file holding the last value.
        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(
            reader.read_vlen_strings("notes").unwrap(),
            vec![payload_b.clone()]
        );

        std::fs::remove_file(&path).ok();
    }

    /// Replacing a vlen string attribute must release the superseded
    /// global-heap collection *before* the replacement's collection is
    /// allocated, so a reopen-replace loop lands each new value in the block
    /// it just freed instead of growing the file by one collection per
    /// session — the attribute counterpart of
    /// [`vlen_replace_across_reopen_keeps_the_file_flat`].
    #[test]
    fn vlen_attr_replace_across_reopen_keeps_the_file_flat() {
        let path = temp_path("vlen_attr_reopen_flat");
        let payload_a = "a".repeat(8 * 1024);
        let payload_b = "b".repeat(8 * 1024);

        let writer = Hdf5Writer::create(&path).unwrap();
        writer
            .set_vlen_string_attribute(AttrTarget::Root, "note", &payload_a)
            .unwrap();
        writer.close().unwrap();

        let mut sizes = Vec::new();
        for i in 0..8 {
            let writer = Hdf5Writer::open_append(&path).unwrap();
            let payload = if i % 2 == 0 { &payload_b } else { &payload_a };
            writer
                .set_vlen_string_attribute(AttrTarget::Root, "note", payload)
                .unwrap();
            writer.close().unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        assert_eq!(&sizes[1..], &vec![sizes[0]; 7][..], "sizes: {sizes:?}");

        // The reused blocks still hold the last value.
        let reader = Hdf5Reader::open(&path).unwrap();
        let attr = reader.root_attr("note").unwrap().clone();
        let mut reader = reader;
        assert_eq!(reader.attr_string_value(&attr).unwrap(), payload_a);

        std::fs::remove_file(&path).ok();
    }

    /// A numeric attribute replacing a vlen one goes through the same list
    /// owner, so the superseded collection is released even though the new
    /// value holds no heap reference: a later same-size vlen attribute must
    /// land in the freed block, making the file exactly as large as one that
    /// never stored the replaced value.
    #[test]
    fn numeric_replacing_a_vlen_attr_releases_its_collection() {
        let payload = "x".repeat(8 * 1024);
        let numeric = || {
            AttributeMessage::scalar_numeric(
                "x",
                DatatypeMessage::i32_type(),
                7i32.to_le_bytes().to_vec(),
            )
        };

        let path_a = temp_path("vlen_attr_cross_a");
        let writer = Hdf5Writer::create(&path_a).unwrap();
        writer
            .set_vlen_string_attribute(AttrTarget::Root, "x", &payload)
            .unwrap();
        writer.add_root_attribute(numeric()).unwrap();
        writer
            .set_vlen_string_attribute(AttrTarget::Root, "y", &payload)
            .unwrap();
        writer.close().unwrap();

        // The same end state written without the replaced vlen value.
        let path_b = temp_path("vlen_attr_cross_b");
        let writer = Hdf5Writer::create(&path_b).unwrap();
        writer.add_root_attribute(numeric()).unwrap();
        writer
            .set_vlen_string_attribute(AttrTarget::Root, "y", &payload)
            .unwrap();
        writer.close().unwrap();

        assert_eq!(
            std::fs::metadata(&path_a).unwrap().len(),
            std::fs::metadata(&path_b).unwrap().len()
        );

        let reader = Hdf5Reader::open(&path_a).unwrap();
        let y = reader.root_attr("y").unwrap().clone();
        let mut reader = reader;
        assert_eq!(reader.attr_string_value(&y).unwrap(), payload);

        std::fs::remove_file(&path_a).ok();
        std::fs::remove_file(&path_b).ok();
    }

    /// Reopen/write/close cycles must not leak the object-header blocks
    /// finalize rewrites: the reopened root header, the reopened group
    /// header, and the modified chunked dataset's header are each freed
    /// before their replacements are allocated. The chunk rewrite itself is
    /// in place (unfiltered chunks never move), so a leak of any header
    /// block shows up as monotonic growth here.
    #[test]
    fn reopen_cycles_reuse_superseded_header_blocks() {
        let path = temp_path("header_reuse");
        {
            let writer = Hdf5Writer::create(&path).unwrap();
            writer.create_group("/", "g").unwrap();
            let idx = writer
                .create_chunked_dataset(
                    "g/data",
                    DatatypeMessage::i32_type(),
                    &[4],
                    &[u64::MAX],
                    &[4],
                )
                .unwrap();
            let seed: Vec<u8> = [1i32, 2, 3, 4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            writer.write_chunk(idx, 0, &seed).unwrap();
            writer.close().unwrap();
        }

        let mut sizes = Vec::new();
        for i in 0..6i32 {
            let writer = Hdf5Writer::open_append(&path).unwrap();
            let data: Vec<u8> = [i; 4].iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_chunk(0, 0, &data).unwrap();
            writer.close().unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        assert_eq!(&sizes[1..], &vec![sizes[0]; 5][..], "sizes: {sizes:?}");

        // The reused header blocks still form a valid file.
        let mut reader = Hdf5Reader::open(&path).unwrap();
        let raw = reader.read_dataset_raw("g/data").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![5, 5, 5, 5]);

        std::fs::remove_file(&path).ok();
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

        let writer = Hdf5Writer::create(&path).unwrap();
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

        let writer = Hdf5Writer::create(&path).unwrap();

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

        let writer = Hdf5Writer::create(&path).unwrap();
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

        let writer = Hdf5Writer::create(&path).unwrap();
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

        let writer = Hdf5Writer::create(&path).unwrap();
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

        let writer = Hdf5Writer::create(&path).unwrap();
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
    fn fixed_array_paged_decode_roundtrip_with_uninitialized_page() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let hdr = FixedArrayHeader::new_for_chunks(&ctx, 2500);
        let npages = hdr.npages() as usize; // 3
        let page = hdr.dblk_page_nelmts() as usize; // 1024

        // Populate pages 0 and 2; leave page 1 entirely undefined so its
        // bitmap bit stays clear on encode.
        let mut dblk = FixedArrayDataBlock::new_unfiltered(0x1000, 2500);
        for i in (0..page).chain(2 * page..2500) {
            dblk.elements[i] = 0x10000 + (i as u64) * 0x100;
        }

        let mut encoded = encode_fixed_array_dblk(&ctx, &hdr, &dblk);
        let prefix = FixedArrayPagedPrefix::decode(&encoded, &ctx, npages as u64).unwrap();
        assert!(prefix.page_initialized(0));
        assert!(!prefix.page_initialized(1));
        assert!(prefix.page_initialized(2));

        // Corrupt the uninitialized page's bytes the way libhdf5 leaves
        // them: arbitrary, no valid checksum. Decode must not look at it.
        let page_stride = page * 8 + 4;
        let p1 = prefix.prefix_size + page_stride;
        for b in &mut encoded[p1..p1 + page_stride] {
            *b = 0x5A;
        }

        let decoded = decode_fixed_array_dblk(&ctx, &hdr, &encoded, 0).unwrap();
        assert_eq!(decoded.elements, dblk.elements);
        assert_eq!(decoded.header_addr, 0x1000);
    }

    #[test]
    fn fixed_array_paged_decode_filtered_roundtrip() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let chunk_size_len = 4usize;
        let hdr = FixedArrayHeader::new_for_filtered_chunks(&ctx, 1500, chunk_size_len as u8);
        assert!(hdr.is_paged());

        let mut dblk = FixedArrayDataBlock::new_filtered(0x2000, 1500);
        for (i, e) in dblk.filtered_elements.iter_mut().enumerate() {
            e.address = 0x8000 + (i as u64) * 0x40;
            e.chunk_size = 100 + i as u64;
            e.filter_mask = (i % 3) as u32;
        }

        let encoded = encode_fixed_array_dblk(&ctx, &hdr, &dblk);
        assert_eq!(encoded.len() as u64, fixed_array_dblk_disk_size(&ctx, &hdr));
        let decoded = decode_fixed_array_dblk(&ctx, &hdr, &encoded, chunk_size_len).unwrap();
        assert_eq!(decoded.filtered_elements, dblk.filtered_elements);
        assert_eq!(decoded.client_id, FA_CLIENT_FILT_CHUNK);
    }

    #[test]
    fn create_fixed_array_paged_dataset_roundtrip() {
        let path = temp_path("fixed_array_paged");

        // 1D dataset of 3000 elements, chunk size 1 => 3000 chunks.
        // 3000 > 1024 (one page) => the FA data block must be paged.
        let n: usize = 3000;
        let writer = Hdf5Writer::create(&path).unwrap();
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

        let writer = Hdf5Writer::create(&path).unwrap();
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
        let writer = Hdf5Writer::create(&path).unwrap();
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
            e.chunk_size = (i % 200) as u64;
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

        let writer = Hdf5Writer::create(&path).unwrap();
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

    /// Bytes one chunk of [`btree_v2_flush_probe`]'s dataset occupies — an
    /// f64 element, so the allocator's alignment neither pads nor merges it and
    /// the file's growth is exactly the bytes asked for.
    const BT2_PROBE_CHUNK: u64 = 8;

    /// Write chunks of a 1x1-chunked 2-D BT2 dataset, flushing at each batch
    /// boundary, and report `(node addresses, file length)` after every flush.
    /// Chunks are addressed down column 0 so the record count — and hence the
    /// tree's shape — grows one record at a time.
    fn btree_v2_flush_probe(path: &std::path::Path, batches: &[u64]) -> Vec<(Vec<u64>, u64)> {
        let writer = Hdf5Writer::create(path).unwrap();
        let idx = writer
            .create_btree_v2_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 0],
                &[u64::MAX, u64::MAX],
                &[1, 1],
            )
            .unwrap();
        let mut written = 0u64;
        let mut out = Vec::new();
        for &upto in batches {
            while written < upto {
                writer
                    .write_chunk_btree_v2(idx, &[written, 0], &(written as f64).to_le_bytes())
                    .unwrap();
                written += 1;
            }
            writer.flush_dataset(idx).unwrap();
            let addrs = writer
                .ds(idx)
                .lock()
                .btree_v2
                .as_ref()
                .unwrap()
                .node_addrs
                .clone();
            out.push((addrs, std::fs::metadata(path).unwrap().len()));
        }
        writer.extend_dataset(idx, &[written.max(1), 1]).unwrap();
        writer.close().unwrap();
        out
    }

    /// The node pool tracks the tree in both directions. Dropping records is
    /// what a removal path would do — [`Bt2ChunkIndex`] has none today, so the
    /// test drops them itself — and the flush that follows must hand the blocks
    /// its smaller tree no longer needs back to the allocator instead of
    /// leaving them recorded and unreachable.
    #[test]
    fn a_btree_v2_flush_frees_the_node_blocks_its_tree_gave_up() {
        use crate::format::chunk_index::btree_v2::BT2_NODE_SIZE;

        let path = temp_path("bt2_node_shrink");
        let writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_btree_v2_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 0],
                &[u64::MAX, u64::MAX],
                &[1, 1],
            )
            .unwrap();
        // 85 records is one past a leaf, so the tree is two leaves and a root.
        for i in 0..85u64 {
            writer
                .write_chunk_btree_v2(idx, &[i, 0], &(i as f64).to_le_bytes())
                .unwrap();
        }
        writer.flush_dataset(idx).unwrap();
        let grown = writer
            .ds(idx)
            .lock()
            .btree_v2
            .as_ref()
            .unwrap()
            .node_addrs
            .clone();
        assert_eq!(grown.len(), 3, "expected two leaves and a root");

        // Back to 84 records: one leaf, so two of the three blocks are surplus.
        writer
            .ds(idx)
            .lock()
            .btree_v2
            .as_mut()
            .unwrap()
            .index
            .records
            .truncate(84);
        writer.flush_dataset(idx).unwrap();
        let shrunk = writer
            .ds(idx)
            .lock()
            .btree_v2
            .as_ref()
            .unwrap()
            .node_addrs
            .clone();
        assert_eq!(
            shrunk,
            grown[..1],
            "the pool still records the surplus blocks"
        );

        // The surplus went back to the allocator, not on the floor: the next
        // node-sized allocation lands inside the region the two blocks covered.
        let reused = writer.allocator.allocate(BT2_NODE_SIZE as u64);
        assert!(
            (grown[1]..grown[1] + 2 * BT2_NODE_SIZE as u64).contains(&reused),
            "a node block allocated at {reused:#x}, outside the freed \
             [{:#x}, {:#x}) the flush gave up",
            grown[1],
            grown[1] + 2 * BT2_NODE_SIZE as u64
        );

        writer.extend_dataset(idx, &[85, 1]).unwrap();
        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A v2 B-tree whose header declares a non-default node size — libhdf5
    /// built with a different `H5D_BT2_NODE_SIZE`, or any other writer —
    /// reopens for append: the reconstruction adopts the header's node_size,
    /// split and merge instead of refusing everything but 2048, and the next
    /// flush re-serializes at that size (upstream allocates every node at
    /// `hdr->node_size`, H5B2leaf.c / H5B2internal.c).
    #[test]
    fn a_btree_v2_with_a_foreign_node_size_reopens_and_grows() {
        let path = temp_path("bt2_foreign_node_size");
        {
            let writer = Hdf5Writer::create(&path).unwrap();
            let idx = writer
                .create_btree_v2_dataset(
                    "data",
                    DatatypeMessage::f64_type(),
                    &[0, 0],
                    &[u64::MAX, u64::MAX],
                    &[1, 1],
                )
                .unwrap();
            // Act as a foreign writer: 512-byte nodes, non-default tuning.
            // record_size 24 => a 512-byte leaf holds 20 records, so 85
            // records make a depth-1 tree of 512-byte blocks.
            {
                let ds = writer.ds(idx);
                let mut m = ds.lock();
                let index = &mut m.btree_v2.as_mut().unwrap().index;
                index.node_size = 512;
                index.split_percent = 90;
                index.merge_percent = 30;
            }
            for i in 0..85u64 {
                writer
                    .write_chunk_btree_v2(idx, &[i, 0], &(i as f64).to_le_bytes())
                    .unwrap();
            }
            writer.extend_dataset(idx, &[85, 1]).unwrap();
            writer.close().unwrap();
        }
        {
            let writer = Hdf5Writer::open_append(&path).unwrap();
            let idx = writer.dataset_index("data").unwrap();
            {
                let ds = writer.ds(idx);
                let m = ds.lock();
                let index = &m.btree_v2.as_ref().unwrap().index;
                assert_eq!(index.node_size, 512, "header node_size not adopted");
                assert_eq!(index.split_percent, 90);
                assert_eq!(index.merge_percent, 30);
                assert_eq!(index.records.len(), 85, "records not walked back");
            }
            for i in 85..115u64 {
                writer
                    .write_chunk_btree_v2(idx, &[i, 0], &(i as f64).to_le_bytes())
                    .unwrap();
            }
            writer.extend_dataset(idx, &[115, 1]).unwrap();
            writer.close().unwrap();
        }

        let mut reader = Hdf5Reader::open(&path).unwrap();
        let raw = reader.read_dataset_raw("data").unwrap();
        let values: Vec<f64> = raw
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 115);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as f64, "element {i}");
        }
        std::fs::remove_file(&path).ok();
    }

    /// A node's record count falls as well as rises: the tree's first leaf goes
    /// from a full 84 records to 42 when 85 records force it to split. The node
    /// image is padded to the whole block so re-serializing overwrites the
    /// block, not a prefix of it — otherwise that leaf keeps the tail of its
    /// 84-record self, stale records sitting in a live node block.
    #[test]
    fn a_shrinking_btree_v2_node_leaves_no_stale_records_behind() {
        use crate::format::chunk_index::btree_v2::{Bt2ChunkIndex, BT2_NODE_SIZE};

        let path = temp_path("bt2_node_blocks");
        let probe = btree_v2_flush_probe(&path, &[84, 85]);
        let node0 = probe.last().unwrap().0[0];

        // What the first leaf holds once the tree has split.
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let mut index = Bt2ChunkIndex::new_unfiltered(2);
        for i in 0..85u64 {
            index.insert(vec![i, 0], 0);
        }
        let tree = index.build_tree(&ctx);
        assert!(
            tree.nodes[0].num_records < 84,
            "this test needs the first leaf to shrink, got {}",
            tree.nodes[0].num_records
        );
        // signature(4) + version(1) + type(1) + records + checksum(4)
        let used = 10 + tree.nodes[0].num_records as usize * tree.record_size as usize;

        let bytes = std::fs::read(&path).unwrap();
        let block = &bytes[node0 as usize..node0 as usize + BT2_NODE_SIZE as usize];
        assert!(
            block[used..].iter().all(|&b| b == 0),
            "leaf block at {node0:#x} still holds {} bytes of its previous, larger image",
            block[used..].iter().rposition(|&b| b != 0).unwrap_or(0) + 1
        );
        std::fs::remove_file(&path).ok();
    }

    /// The node pool is the single owner of the tree's block addresses: a flush
    /// reuses every block already in it and allocates only the shortfall. So
    /// re-flushing an unchanged index must cost nothing, and a flush that grows
    /// the tree must cost exactly the blocks it added — anything more means a
    /// block was stranded.
    #[test]
    fn a_btree_v2_flush_allocates_only_the_node_blocks_it_adds() {
        use crate::format::chunk_index::btree_v2::BT2_NODE_SIZE;

        let path = temp_path("bt2_pool_growth");
        // Re-flush at 84 (still one leaf), then cross into a three-node depth-1
        // tree, then keep growing.
        let batches = [84u64, 84, 85, 200, 200];
        let probe = btree_v2_flush_probe(&path, &batches);
        for i in 1..probe.len() {
            let (prev_addrs, prev_len) = &probe[i - 1];
            let (addrs, len) = &probe[i];
            assert!(
                addrs.starts_with(prev_addrs),
                "flush {i} moved a node block instead of reusing it"
            );
            let new_blocks = (addrs.len() - prev_addrs.len()) as u64 * BT2_NODE_SIZE as u64;
            let new_chunks = (batches[i] - batches[i - 1]) * BT2_PROBE_CHUNK;
            assert_eq!(
                len - prev_len,
                new_blocks + new_chunks,
                "flush {i} grew the file by more than the blocks it added"
            );
        }
        // The unchanged re-flushes must be free.
        assert_eq!(probe[1].1, probe[0].1);
        assert_eq!(probe[4].1, probe[3].1);
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_batch_write_roundtrip() {
        let path = temp_path("parallel_batch");

        let writer = Hdf5Writer::create(&path).unwrap();
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

    /// A chunk tile larger than the frame is geometry libhdf5 refuses to
    /// create (`H5D__chunk_construct`: chunk must not exceed a fixed maximum
    /// dimension), so no libhdf5-based writer — including the NDFileHDF5
    /// tiling controls this API mirrors — can produce such a file. Until
    /// 0.4.1 we accepted it and zero-padded the frame up to the tile; now
    /// the create is rejected like every other creator's.
    #[test]
    fn swmr_writer_tiled_chunk_larger_than_frame_is_rejected() {
        use crate::io::swmr::SwmrWriter;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_swmr_bigchunk_{}_{}.h5",
            std::process::id(),
            n
        ));

        let mut swmr = SwmrWriter::create(&path).unwrap();
        let err = swmr
            .create_streaming_dataset_tiled("det", DatatypeMessage::u16_type(), &[3, 3], &[8, 8])
            .unwrap_err();
        assert!(
            err.to_string().contains("maximum dimension size"),
            "unexpected error: {err}"
        );
        swmr.close().unwrap();
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

        let writer = Hdf5Writer::create(&path).unwrap();

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
        let raw_g0: Vec<u8> = [10i32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_dataset_raw(ds_g0, &raw_g0).unwrap();

        let ds_g1 = writer
            .create_dataset("group1/sub/values", DatatypeMessage::u8_type(), &[4])
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

    /// libhdf5 (`H5D__chunk_construct`) rejects a chunk dimension that
    /// exceeds a fixed maximum dimension. Before this check, such a dataset
    /// was created and appends landed rows at the chunk stride instead of
    /// the row stride, reading back [1, 2, 0, 0] for [1, 2, 3, 4].
    #[test]
    fn create_rejects_a_chunk_wider_than_a_fixed_max_dimension() {
        let path = temp_path("chunk_wider_than_max");

        let writer = Hdf5Writer::create(&path).unwrap();
        let err = writer
            .create_chunked_dataset(
                "data",
                DatatypeMessage::f64_type(),
                &[0, 2],
                &[u64::MAX, 2],
                &[2, 4],
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("maximum dimension size"),
            "unexpected error: {err}"
        );

        // The fixed-array creators derive the maximum from the fixed dims.
        let err = writer
            .create_fixed_array_dataset("fa", DatatypeMessage::f64_type(), &[3], &[5])
            .unwrap_err();
        assert!(
            err.to_string().contains("maximum dimension size"),
            "unexpected error: {err}"
        );

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// libhdf5 exempts a dimension whose *current* size is zero from the
    /// chunk-vs-maximum check (`curr_dims[u] &&` in `H5D__chunk_construct`),
    /// and rejects a zero chunk dimension on every path.
    #[test]
    fn create_mirrors_the_libhdf5_chunk_geometry_exemptions() {
        let path = temp_path("chunk_geometry_exemptions");

        let writer = Hdf5Writer::create(&path).unwrap();
        // dims[1] == 0: chunk 4 > max 2 is allowed, as libhdf5 allows it.
        writer
            .create_chunked_dataset(
                "exempt",
                DatatypeMessage::f64_type(),
                &[0, 0],
                &[u64::MAX, 2],
                &[2, 4],
            )
            .unwrap();

        let err = writer
            .create_chunked_dataset("zero", DatatypeMessage::f64_type(), &[0], &[u64::MAX], &[0])
            .unwrap_err();
        assert!(
            err.to_string().contains("chunk dimension 0 is zero"),
            "unexpected error: {err}"
        );

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A file written by 0.4.0 can carry a chunk row wider than the frame
    /// row — create now rejects that geometry, but reopened files keep it.
    /// Appends must scatter frames at the chunk stride, not pack them at
    /// the frame stride (which read back `[1, 2, 0, 0]` for `[1, 2, 3, 4]`).
    /// The wide shape is simulated by widening the registered chunk dims
    /// after create, which also lands in the layout message at close.
    #[test]
    fn append_scatters_into_a_legacy_wider_than_row_chunk() {
        let path = temp_path("legacy_wide_chunk_append");

        let writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer
            .create_chunked_dataset(
                "data",
                DatatypeMessage::i32_type(),
                &[0, 2],
                &[u64::MAX, 2],
                &[2, 2],
            )
            .unwrap();
        writer.ds(idx).lock().chunked.as_mut().unwrap().chunk_dims = vec![2, 4];

        let frames: Vec<u8> = [1i32, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_append_frames(idx, 0, 2, &frames).unwrap();
        writer.extend_dataset(idx, &[2, 2]).unwrap();
        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        assert_eq!(reader.dataset_shape("data").unwrap(), vec![2, 2]);
        let raw = reader.read_dataset_raw("data").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1, 2, 3, 4]);
        std::fs::remove_file(&path).ok();
    }

    /// The compressed vlen creator sizes its chunked layout from a
    /// caller-supplied chunk size; it goes through the same geometry
    /// validation as every other creator (empty inputs are exempt because
    /// their current size is zero).
    #[test]
    #[cfg(feature = "deflate")]
    fn compressed_vlen_create_validates_its_chunk_size() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("vlen_compressed_chunk");

        let writer = Hdf5Writer::create(&path).unwrap();
        let err = writer
            .create_vlen_string_dataset_compressed(
                "texts",
                &["a", "b", "c"],
                100,
                FilterPipeline::deflate(6),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("maximum dimension size"),
            "unexpected error: {err}"
        );

        writer
            .create_vlen_string_dataset_compressed("empty", &[], 16, FilterPipeline::deflate(6))
            .unwrap();

        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// `set_libver_latest` moves *filtered* chunked datasets to layout v5 with
    /// fixed 8-byte chunk-size fields; unfiltered chunked and pre-opt-in
    /// datasets keep v4 with the derived width, matching libhdf5's
    /// `version_perf` rule (only the filtered index arms bump to 5).
    #[cfg(feature = "deflate")]
    #[test]
    fn libver_latest_selects_v5_for_filtered_chunks_only() {
        let path = temp_path("libver_v5_select");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        let before = writer
            .create_chunked_dataset_compressed(
                "d4",
                DatatypeMessage::i32_type(),
                &[0],
                &[u64::MAX],
                &[16],
                4,
            )
            .unwrap();
        writer.set_libver_latest(true).unwrap();
        let ea5 = writer
            .create_chunked_dataset_compressed(
                "ea5",
                DatatypeMessage::i32_type(),
                &[0],
                &[u64::MAX],
                &[16],
                4,
            )
            .unwrap();
        let plain = writer
            .create_chunked_dataset(
                "plain",
                DatatypeMessage::i32_type(),
                &[0],
                &[u64::MAX],
                &[16],
            )
            .unwrap();
        let fa5 = writer
            .create_fixed_array_dataset_with_pipeline(
                "fa5",
                DatatypeMessage::i32_type(),
                &[4, 6],
                &[2, 3],
                FilterPipeline::deflate(6),
            )
            .unwrap();
        let bt5 = writer
            .create_btree_v2_dataset_with_pipeline(
                "bt5",
                DatatypeMessage::i32_type(),
                &[0, 0],
                &[u64::MAX, u64::MAX],
                &[2, 3],
                FilterPipeline::deflate(6),
            )
            .unwrap();

        {
            let d4 = writer.ds(before);
            let d4 = d4.lock();
            assert_eq!(d4.layout_version, 4);
            assert_eq!(
                d4.chunked.as_ref().unwrap().chunk_size_len,
                compute_chunk_size_len(16 * 4)
            );
            let e5 = writer.ds(ea5);
            let e5 = e5.lock();
            assert_eq!(e5.layout_version, 5);
            assert_eq!(e5.chunked.as_ref().unwrap().chunk_size_len, 8);
            assert_eq!(writer.ds(plain).lock().layout_version, 4);
            assert_eq!(writer.ds(fa5).lock().layout_version, 5);
            assert_eq!(writer.ds(bt5).lock().layout_version, 5);
        }

        // Write through the FA and BT2 v5 indexes so their 8-byte chunk-size
        // fields are exercised end to end, not just selected.
        for (coords, vals) in [
            ([0u64, 0], [0i32, 1, 2, 6, 7, 8]),
            ([0, 1], [3, 4, 5, 9, 10, 11]),
            ([1, 0], [12, 13, 14, 18, 19, 20]),
            ([1, 1], [15, 16, 17, 21, 22, 23]),
        ] {
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer
                .write_chunk_fixed_array(fa5, &coords, &bytes)
                .unwrap();
            writer.write_chunk_btree_v2(bt5, &coords, &bytes).unwrap();
        }
        writer.extend_dataset(bt5, &[4, 6]).unwrap();
        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        for name in ["fa5", "bt5"] {
            let raw = reader.read_dataset_raw(name).unwrap();
            let values: Vec<i32> = raw
                .chunks(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(values, (0..24).collect::<Vec<i32>>(), "dataset {name}");
        }

        std::fs::remove_file(&path).ok();
    }

    /// A v5 file reopened for append must stay v5: the decode → `DatasetInfo`
    /// → finalize path carries the version through, so the re-encoded layout
    /// message matches the 8-byte size fields the filtered index was built
    /// with. A silent v4 downgrade here would make libhdf5 derive a narrower
    /// field width than the index uses.
    #[cfg(feature = "deflate")]
    #[test]
    fn v5_layout_survives_reopen_and_append() {
        let path = temp_path("libver_v5_reopen");
        let chunk: usize = 8;

        let mut writer = Hdf5Writer::create(&path).unwrap();
        writer.set_libver_latest(true).unwrap();
        let idx = writer
            .create_chunked_dataset_compressed(
                "d",
                DatatypeMessage::i32_type(),
                &[0],
                &[u64::MAX],
                &[chunk as u64],
                4,
            )
            .unwrap();
        for c in 0..2u64 {
            let data: Vec<u8> = (0..chunk as i32)
                .flat_map(|i| (c as i32 * chunk as i32 + i).to_le_bytes())
                .collect();
            writer.write_chunk(idx, c, &data).unwrap();
        }
        writer.extend_dataset(idx, &[2 * chunk as u64]).unwrap();
        writer.close().unwrap();

        // Reopen: the decoded layout version must be preserved, and appends
        // must keep working against the 8-byte-size-field index.
        let writer = Hdf5Writer::open_append(&path).unwrap();
        assert_eq!(writer.ds(0).lock().layout_version, 5);
        for c in 2..4u64 {
            let data: Vec<u8> = (0..chunk as i32)
                .flat_map(|i| (c as i32 * chunk as i32 + i).to_le_bytes())
                .collect();
            writer.write_chunk(0, c, &data).unwrap();
        }
        writer.extend_dataset(0, &[4 * chunk as u64]).unwrap();
        writer.close().unwrap();

        // Still v5 after the second finalize, and fully readable.
        let writer = Hdf5Writer::open_append(&path).unwrap();
        assert_eq!(writer.ds(0).lock().layout_version, 5);
        writer.close().unwrap();

        let mut reader = Hdf5Reader::open(&path).unwrap();
        let raw = reader.read_dataset_raw("d").unwrap();
        let values: Vec<i32> = raw
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, (0..4 * chunk as i32).collect::<Vec<i32>>());

        std::fs::remove_file(&path).ok();
    }

    /// A chunk strictly larger than `u32::MAX` bytes forces layout v5 with no
    /// opt-in — v4's size field cannot represent it — while a chunk of exactly
    /// `u32::MAX` bytes stays v4, matching libhdf5's `version_req` boundary
    /// (`> 0xffffffff`, filtered or not).
    #[test]
    fn oversized_chunk_forces_v5_without_opt_in() {
        let path = temp_path("libver_4gib_force");

        let writer = Hdf5Writer::create(&path).unwrap();
        let at_limit = writer
            .create_chunked_dataset_compressed(
                "at_limit",
                DatatypeMessage::u8_type(),
                &[0],
                &[u64::MAX],
                &[u32::MAX as u64],
                4,
            )
            .unwrap();
        let over = writer
            .create_chunked_dataset_compressed(
                "over",
                DatatypeMessage::u8_type(),
                &[0],
                &[u64::MAX],
                &[u32::MAX as u64 + 1],
                4,
            )
            .unwrap();
        let over_unfiltered = writer
            .create_chunked_dataset(
                "over_plain",
                DatatypeMessage::u8_type(),
                &[0],
                &[u64::MAX],
                &[u32::MAX as u64 + 1],
            )
            .unwrap();

        assert_eq!(writer.ds(at_limit).lock().layout_version, 4);
        {
            let ds = writer.ds(over);
            let ds = ds.lock();
            assert_eq!(ds.layout_version, 5);
            assert_eq!(ds.chunked.as_ref().unwrap().chunk_size_len, 8);
        }
        assert_eq!(writer.ds(over_unfiltered).lock().layout_version, 5);
        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// SWMR reaches version 3 on its own, without a chunked dataset to raise
    /// the bound — through the flags `finalize_for_swmr` passes, and then
    /// through `swmr_active` for every superblock written after it. Only a
    /// file with nothing else newer in it can tell the two arms apart, and
    /// the public SWMR API always creates a chunked streaming dataset.
    #[test]
    fn swmr_reaches_version_3_with_no_chunked_dataset_in_the_file() {
        let path = temp_path("swmr_superblock");

        let mut writer = Hdf5Writer::create(&path).unwrap();
        writer
            .create_dataset("d", DatatypeMessage::i32_type(), &[2])
            .unwrap();
        assert_eq!(writer.superblock_version_for(0), SUPERBLOCK_V2);

        writer.finalize_for_swmr().unwrap();
        assert_eq!(std::fs::read(&path).unwrap()[8], SUPERBLOCK_V3);

        // The close-time finalize carries no SWMR flag; the file is still an
        // SWMR file and must not be handed back a version older than the one
        // its readers attached to.
        writer.close().unwrap();
        assert_eq!(std::fs::read(&path).unwrap()[8], SUPERBLOCK_V3);
        std::fs::remove_file(&path).ok();
    }
}
