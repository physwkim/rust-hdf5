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
use crate::io::hyperslab::{for_each_contiguous_run, for_each_dual_run};
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

/// A single dataset's metadata behind its own [`Slot`], reference-counted so
/// a writer can clone it out of the registry (releasing the registry lock)
/// and then lock just this one dataset. Two threads writing different
/// datasets take different `DatasetRef` locks and never contend; the same
/// dataset's writes serialize, which is required because one chunk index is
/// not concurrently mutable.
pub(crate) type DatasetRef = Shared<Slot<DatasetInfo>>;

/// A single group's metadata behind its own [`Slot`], reference-counted like
/// [`DatasetRef`].
pub(crate) type GroupRef = Shared<Slot<GroupInfo>>;

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
    /// Buffer for partially filled chunks during append.
    pub append_buffer: Vec<u8>,
    /// Number of frames accumulated in `append_buffer`.
    pub append_buffered_frames: u64,
    /// Attributes attached to this dataset.
    pub attributes: Vec<AttributeMessage>,
    /// File offset where the dataset object header was written (for SWMR in-place rewrites).
    pub obj_header_written_addr: Option<u64>,
    /// Encoded size of the dataset object header (for verifying in-place rewrites fit).
    pub obj_header_encoded_size: usize,
    /// Filter pipeline for compressed chunks.
    pub filter_pipeline: Option<FilterPipeline>,
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

    /// Number of chunks spanning each dimension at the current extent.
    fn grid(&self) -> Vec<u64> {
        self.dims
            .iter()
            .zip(&self.chunk_dims)
            .map(|(&d, &c)| if c > 0 { d.div_ceil(c) } else { 0 })
            .collect()
    }

    /// Row-major position of `coords` in the chunk grid — the linear index an
    /// extensible or fixed array records the chunk under.
    ///
    /// The grid extents multiplied here come from the *current* dims, which is
    /// how every chunk written so far was indexed. The bound a coordinate is
    /// checked against comes from `max_dims`, so an unlimited dimension — the
    /// one an extensible array exists to grow — is unbounded, while a fixed
    /// dimension still rejects an out-of-grid coordinate that would otherwise
    /// silently alias another chunk's slot.
    fn linear_index(&self, coords: &[u64]) -> IoResult<u64> {
        let ndims = self.dims.len();
        if coords.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk_coords has {} entries but the dataset has {} dimensions",
                coords.len(),
                ndims
            )));
        }
        if self.chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset chunk shape has {} dimensions but the dataspace has {}",
                self.chunk_dims.len(),
                ndims
            )));
        }
        let grid = self.grid();
        let mut linear = 0u64;
        for d in 0..ndims {
            if self.chunk_dims[d] == 0 {
                return Err(crate::io::IoError::InvalidState(format!(
                    "chunk dimension {d} is zero"
                )));
            }
            let extent = self.max_dims.as_ref().map_or(self.dims[d], |m| m[d]);
            if extent != u64::MAX {
                let bound = extent.div_ceil(self.chunk_dims[d]);
                if coords[d] >= bound {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "chunk coordinate {} in dimension {} is outside the chunk grid (0..{})",
                        coords[d], d, bound
                    )));
                }
            }
            linear = linear
                .checked_mul(grid[d])
                .and_then(|l| l.checked_add(coords[d]))
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState(
                        "chunk coordinates overflow the array index".into(),
                    )
                })?;
        }
        Ok(linear)
    }
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
    /// Pool of `BT2_NODE_SIZE`-byte blocks holding the tree's nodes, in the
    /// order [`Bt2Tree::encode`] emits them.
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
#[derive(Clone)]
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
    /// Attributes attached to the root group (file-level attributes).
    pub(crate) root_attributes: Slot<Vec<crate::format::messages::attribute::AttributeMessage>>,
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
    closed: bool,
    /// Set once `finalize_for_swmr` has published a readable file.
    ///
    /// A SWMR reader may hold a chunk index that still points at a block this
    /// writer has since replaced, so from that point on a relocated chunk's
    /// old block is kept rather than released for reuse — the same rule as
    /// libhdf5's `H5D__chunk_file_alloc`, which skips `H5MF_xfree` under
    /// `H5F_ACC_SWMR_WRITE`.
    swmr_active: bool,
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
            datasets: Slot::new(Vec::new()),
            groups: Slot::new(Vec::new()),
            hard_links: Slot::new(Vec::new()),
            root_attributes: Slot::new(Vec::new()),
            create_lock: Slot::new(()),
            closed: false,
            swmr_active: false,
            root_group_addr: None,
            root_group_encoded_size: 0,
        })
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

    /// Push a freshly-built dataset into the registry and return its index.
    /// Takes the registry lock only for the push, so it does not block an
    /// in-flight write that already cloned its own [`DatasetRef`] out.
    pub(crate) fn push_dataset(&self, info: DatasetInfo) -> usize {
        let mut reg = self.datasets.lock();
        let idx = reg.len();
        reg.push(Shared::new(Slot::new(info)));
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
                append_buffer: Vec::new(),
                append_buffered_frames: 0,
                attributes: attrs,
                obj_header_written_addr: Some(*obj_addr),
                obj_header_encoded_size: 0,
                filter_pipeline: fp,
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

        // Wrap the reconstructed plain vecs into the per-slot registry. The
        // reconstruction logic above runs single-threaded on local `Vec`s;
        // only the final hand-off needs the `Shared<Slot<_>>` shape.
        let datasets = existing_datasets
            .into_iter()
            .map(|i| Shared::new(Slot::new(i)))
            .collect();
        let groups = groups
            .into_iter()
            .map(|g| Shared::new(Slot::new(g)))
            .collect();

        Ok(Self {
            handle,
            allocator,
            ctx,
            datasets: Slot::new(datasets),
            groups: Slot::new(groups),
            hard_links: Slot::new(Vec::new()),
            root_attributes: Slot::new(root_attributes),
            create_lock: Slot::new(()),
            closed: false,
            swmr_active: false,
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
    pub fn dataset_names(&self) -> Vec<String> {
        self.dataset_refs()
            .iter()
            .filter_map(|d| {
                let g = d.lock();
                (!g.deleted).then(|| g.name.clone())
            })
            .collect()
    }

    /// Find a dataset index by name.
    pub fn dataset_index(&self, name: &str) -> Option<usize> {
        self.dataset_refs().iter().position(|d| {
            let g = d.lock();
            g.name == name && !g.deleted
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
        Ok(())
    }

    /// Soft-delete a dataset by name. The dataset is excluded from the file
    /// on close. File space is not reclaimed.
    pub fn delete_dataset(&self, name: &str) -> IoResult<()> {
        let refs = self.dataset_refs();
        let idx = refs
            .iter()
            .position(|d| {
                let g = d.lock();
                g.name == name && !g.deleted
            })
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        refs[idx].lock().deleted = true;
        // Remove from parent group's child_datasets
        for grp in self.group_refs() {
            grp.lock().child_datasets.retain(|&di| di != idx);
        }
        Ok(())
    }

    /// Soft-delete a group and all its child datasets and sub-groups.
    /// File space is not reclaimed.
    pub fn delete_group(&self, name: &str) -> IoResult<()> {
        let name = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{}", name)
        };
        let groups = self.group_refs();
        let gidx = groups
            .iter()
            .position(|g| {
                let gg = g.lock();
                gg.name == name && !gg.deleted
            })
            .ok_or_else(|| crate::io::IoError::NotFound(name.clone()))?;
        self.delete_group_recursive(gidx);
        // Remove from parent's child_groups
        let parent = groups[gidx].lock().parent;
        if let Some(pidx) = parent {
            groups[pidx].lock().child_groups.retain(|&gi| gi != gidx);
        }
        Ok(())
    }

    fn delete_group_recursive(&self, gidx: usize) {
        // Mark deleted and snapshot the child lists, releasing the group lock
        // before locking any dataset/child-group slot (spine → slot order).
        let (child_ds, child_gs) = {
            let grp = self.grp(gidx);
            let mut g = grp.lock();
            g.deleted = true;
            (g.child_datasets.clone(), g.child_groups.clone())
        };
        for di in child_ds {
            self.ds(di).lock().deleted = true;
        }
        for gi in child_gs {
            self.delete_group_recursive(gi);
        }
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
        let full_name = if parent_path == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", parent_path, name)
        };

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

        // Find parent group index (None means it's a root-level group). Indices
        // are append-only, so a parent found in the snapshot stays valid even
        // if another thread pushes a new group concurrently.
        let parent_idx = if parent_path == "/" {
            None
        } else {
            let idx = groups
                .iter()
                .position(|g| g.lock().name == parent_path)
                .ok_or_else(|| {
                    crate::io::IoError::NotFound(format!(
                        "parent group '{}' not found",
                        parent_path
                    ))
                })?;
            Some(idx)
        };

        let group_idx = self.push_group(GroupInfo {
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
            self.grp(pidx).lock().child_groups.push(group_idx);
        }

        Ok(group_idx)
    }

    /// Register a dataset as belonging to a group.
    ///
    /// `group_path` is the full path of the group (e.g., "/detector").
    /// `ds_index` is the dataset index returned by `create_dataset`.
    pub fn assign_dataset_to_group(&self, group_path: &str, ds_index: usize) -> IoResult<()> {
        let groups = self.group_refs();
        let group_idx = groups
            .iter()
            .position(|g| g.lock().name == group_path)
            .ok_or_else(|| {
                crate::io::IoError::NotFound(format!("group '{}' not found", group_path))
            })?;
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

        // Hold the create gate across the collision check and the hard-link
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();

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
        let target_rel = target_path.trim_matches('/');
        if target_rel.is_empty() {
            return Err(crate::io::IoError::InvalidState(
                "cannot hard-link the root group".into(),
            ));
        }
        let target = if let Some(idx) = datasets.iter().position(|d| {
            let g = d.lock();
            !g.deleted && g.name.trim_start_matches('/') == target_rel
        }) {
            HardLinkTarget::Dataset(idx)
        } else if let Some(idx) = groups.iter().position(|g| {
            let gg = g.lock();
            !gg.deleted && gg.name.trim_start_matches('/') == target_rel
        }) {
            HardLinkTarget::Group(idx)
        } else {
            return Err(crate::io::IoError::NotFound(format!(
                "hard link target '{target_path}' not found"
            )));
        };

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

    /// Append a `MSG_LINK` message for every user-created hard link whose
    /// parent group is `parent` (`None` == the root group). Called while
    /// building group object headers, once every object's header address
    /// has been assigned.
    fn emit_hard_links(&self, header: &mut ObjectHeader, parent: Option<usize>) {
        for link in self.hard_links_vec() {
            if link.parent != parent || !self.hard_link_emitted(&link) {
                continue;
            }
            let addr = match link.target {
                HardLinkTarget::Dataset(i) => self.ds(i).lock().obj_header_addr,
                HardLinkTarget::Group(i) => self.grp(i).lock().obj_header_addr,
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
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
    ) -> IoResult<usize> {
        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
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

        let idx = self.push_dataset(DatasetInfo {
            name: name.to_string(),
            datatype,
            dataspace,
            obj_header_addr: 0, // set during finalize
            data_addr,
            data_size,
            chunked: None,
            fixed_array: None,
            btree_v2: None,
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
            attributes: Vec::new(),
            obj_header_written_addr: None,
            obj_header_encoded_size: 0,
            filter_pipeline: None,
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
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        max_dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
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

        let idx = self.push_dataset(DatasetInfo {
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });

        Ok(idx)
    }

    /// Write raw bytes to a contiguous dataset identified by `index`.
    ///
    /// The caller is responsible for providing data in the correct byte order
    /// and layout. The length must match the total data size declared at
    /// creation time.
    pub fn write_dataset_raw(&self, index: usize, data: &[u8]) -> IoResult<()> {
        let ds = self.ds(index);
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

    /// Add an attribute to the root group (file-level attribute).
    pub fn add_root_attribute(&self, attr: crate::format::messages::attribute::AttributeMessage) {
        // Replace existing attribute with the same name, or append new one.
        let mut attrs = self.root_attributes.lock();
        if let Some(pos) = attrs.iter().position(|a| a.name == attr.name) {
            attrs[pos] = attr;
        } else {
            attrs.push(attr);
        }
    }

    /// Create a variable-length string dataset and write string data.
    ///
    /// Stores strings in a global heap collection. The dataset raw data
    /// consists of vlen references (collection_addr + object_index pairs).
    pub fn create_vlen_string_dataset(&self, name: &str, strings: &[&str]) -> IoResult<usize> {
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
        let datatype = DatatypeMessage::vlen_string_utf8();
        let dataspace =
            crate::format::messages::dataspace::DataspaceMessage::simple(&[num_strings]);

        let idx = self.push_dataset(DatasetInfo {
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
            fill_value: None,
            chunked: None,
            fixed_array: None,
            btree_v2: None,
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });

        Ok(idx)
    }

    /// Create a 1-D variable-length byte-array dataset.
    ///
    /// Each item's bytes are stored as a global-heap object and the dataset
    /// holds one vlen reference per item (same on-disk shape as a vlen string
    /// dataset). The datatype is a vlen sequence of `u8`, so h5py reads it back
    /// as an array of variable-length `uint8` arrays. `seq_len` is the number
    /// of base (`u8`) elements, i.e. the byte length; no null terminator is
    /// appended.
    pub fn create_vlen_bytes_dataset(&self, name: &str, items: &[&[u8]]) -> IoResult<usize> {
        use crate::format::global_heap::{encode_vlen_reference, GlobalHeapCollection};
        use crate::format::messages::datatype::DatatypeMessage;

        let num_items = items.len() as u64;

        // Build a global heap collection with all byte arrays.
        let mut gcol = GlobalHeapCollection::new();
        let mut obj_indices = Vec::with_capacity(items.len());
        for item in items {
            let idx = gcol.add_object(item.to_vec())?;
            obj_indices.push(idx);
        }

        // Encode and write the global heap collection.
        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

        // Build raw data: one vlen reference per item.
        let ref_size = crate::format::global_heap::vlen_reference_size(&self.ctx);
        let data_size = (num_items as usize) * ref_size;
        let mut raw_data = Vec::with_capacity(data_size);
        for (i, &obj_idx) in obj_indices.iter().enumerate() {
            // base is u8, so element count == byte count.
            let seq_len = crate::format::global_heap::vlen_seq_len(items[i].len())?;
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

        // Create the dataset with a vlen byte-array datatype.
        let datatype = DatatypeMessage::vlen_bytes();
        let dataspace = crate::format::messages::dataspace::DataspaceMessage::simple(&[num_items]);

        let idx = self.push_dataset(DatasetInfo {
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
            fill_value: None,
            chunked: None,
            fixed_array: None,
            btree_v2: None,
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });

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

        let idx = self.push_dataset(DatasetInfo {
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
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
            let seq_len = crate::format::global_heap::vlen_seq_len(strings[i].len())?;
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

        // Merge buffered data with new data. Scope the slot guard: the loop
        // below calls `write_chunk`, which re-locks the same slot.
        let (buffered_frames, mut combined) = {
            let ds = self.ds(ds_index);
            let mut m = ds.lock();
            let buffered_frames = m.append_buffered_frames as usize;
            let combined = std::mem::take(&mut m.append_buffer);
            m.append_buffered_frames = 0;
            (buffered_frames, combined)
        };
        combined.extend_from_slice(&raw);

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
                let offset_in_chunk = (abs_frame % chunk_dim0) * frame_bytes;
                self.append_frames_into_chunk(
                    ds_index,
                    chunk_idx as u64,
                    offset_in_chunk,
                    &combined[byte_pos..end],
                )?;
                byte_pos = end;
                frame_pos += frames_to_fill;
            } else {
                let ds = self.ds(ds_index);
                let mut m = ds.lock();
                m.append_buffer = combined[byte_pos..total_bytes].to_vec();
                m.append_buffered_frames = remaining_frames as u64;
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

    /// Replace elements `start .. start + strings.len()` of a 1-D
    /// variable-length string dataset, leaving its extent and every other
    /// element alone.
    ///
    /// The replacements go into a fresh global heap collection and only the
    /// vlen references of the named elements are rewritten, so the cost is the
    /// new strings plus the chunks those references live in — not the column.
    /// The objects the old references pointed at are freed, so repeated
    /// updates reuse space instead of growing the file. This is what libhdf5
    /// does: `H5T__vlen_disk_write` deletes the reference it read into the
    /// conversion background buffer before storing the new one.
    ///
    /// Elements the append buffer still holds are patched in the buffer rather
    /// than on disk, because that buffer — not the file — is their current
    /// content until it is flushed.
    pub fn write_vlen_strings_slice(
        &self,
        ds_index: usize,
        start: u64,
        strings: &[&str],
    ) -> IoResult<()> {
        use crate::format::global_heap::{
            encode_vlen_reference, vlen_reference_size, GlobalHeapCollection,
        };
        use crate::format::messages::datatype::DatatypeMessage;

        // An empty batch must not reach the heap write below: an empty
        // collection still encodes to the 4096-byte `H5HG_MINALLOC` minimum,
        // so the file would grow by a block nothing references.
        if strings.is_empty() {
            return Ok(());
        }

        // Snapshot what the write needs, then drop the guard: `write_slice`
        // below re-locks the same slot.
        let (charset, dims, buffered_frames) = {
            let ds = self.ds(ds_index);
            let m = ds.lock();
            let charset = match m.datatype {
                DatatypeMessage::VarLenString { charset } => charset,
                _ => {
                    return Err(crate::io::IoError::InvalidState(
                        "write_vlen_strings_slice is only for variable-length string datasets"
                            .into(),
                    ))
                }
            };
            (
                charset,
                m.dataspace.dims.clone(),
                m.append_buffered_frames as usize,
            )
        };

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
        // charset 0 is ASCII. A Rust `&str` is UTF-8, so anything non-ASCII
        // would be stored as UTF-8 under a datatype that declares otherwise.
        if charset == 0 {
            if let Some((i, s)) = strings.iter().enumerate().find(|(_, s)| !s.is_ascii()) {
                return Err(crate::io::IoError::InvalidState(format!(
                    "string {i} ({s:?}) is not ASCII, but the dataset's character set is"
                )));
            }
        }

        // One collection for the batch, as `append_vlen_strings` does.
        let mut gcol = GlobalHeapCollection::new();
        let mut obj_indices = Vec::with_capacity(strings.len());
        for s in strings {
            obj_indices.push(gcol.add_object(s.as_bytes().to_vec())?);
        }
        let ref_size = vlen_reference_size(&self.ctx);

        // The append buffer holds the tail of the dataset, so the range splits
        // into an on-disk prefix and a buffered suffix. An append that is
        // between publishing its buffered count and extending the dims (two
        // separate lock acquisitions on its side) can make the count exceed
        // the extent; report that instead of wrapping.
        let buffer_base = dims[0].checked_sub(buffered_frames as u64).ok_or_else(|| {
            crate::io::IoError::InvalidState(format!(
                "the append buffer holds {buffered_frames} frames but the dataset has {} elements",
                dims[0]
            ))
        })?;
        let split = end.min(buffer_base).max(start);

        // The on-disk references about to be overwritten, read before anything
        // moves. libhdf5 reads the same bytes into the conversion background
        // buffer (`H5D__scatgath_write` gathers the file's current elements
        // when `need_bkg` is set) and hands them to `H5T__vlen_disk_write`,
        // which deletes them before storing the new reference. The buffered
        // suffix is not on disk yet; its references are read further down,
        // under the same lock that patches them.
        let mut superseded =
            self.current_element_bytes(ds_index, start, split - start, ref_size)?;

        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

        let mut refs = Vec::with_capacity(strings.len() * ref_size);
        for (i, &obj_idx) in obj_indices.iter().enumerate() {
            refs.extend_from_slice(&encode_vlen_reference(
                crate::format::global_heap::vlen_seq_len(strings[i].len())?,
                gcol_addr,
                obj_idx as u32,
                &self.ctx,
            ));
        }

        if split > start {
            let n = (split - start) as usize;
            self.write_slice(ds_index, &[start], &[n as u64], &refs[..n * ref_size])?;
        }
        if end > split {
            // One lock acquisition owns the buffered suffix: the offset is
            // derived, the superseded references read and the new ones patched
            // against the same buffer. The snapshot above can be stale under
            // the `threadsafe` feature — a concurrent append may have flushed
            // the buffer and moved its base between the two acquisitions,
            // putting the suffix on disk where this patch no longer reaches
            // it — so a moved base is reported, not applied. A base that is
            // unchanged means nothing flushed (appends only push it up), and
            // then the buffer can only have grown past our range.
            let ds = self.ds(ds_index);
            let mut m = ds.lock();
            m.dataspace.dims[0]
                .checked_sub(m.append_buffered_frames)
                .filter(|&b| b == buffer_base)
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState(format!(
                        "the append buffer moved while elements {split}..{end} were being replaced"
                    ))
                })?;
            let off = ((split - buffer_base) as usize) * ref_size;
            let tail = &refs[(split - start) as usize * ref_size..];
            if off + tail.len() > m.append_buffer.len() {
                return Err(crate::io::IoError::InvalidState(format!(
                    "buffered elements {split}..{end} run past the {}-byte append buffer",
                    m.append_buffer.len()
                )));
            }
            superseded.extend_from_slice(&m.append_buffer[off..off + tail.len()]);
            m.append_buffer[off..off + tail.len()].copy_from_slice(tail);
        }

        // Only now that the new references are in place: a failure above must
        // not leave the file naming objects this already freed.
        self.release_vlen_references(&superseded)?;

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
    /// This is libhdf5's `H5HG_remove` reached through `H5T__vlen_disk_delete`:
    /// the object leaves its collection, the collection is rewritten at its
    /// existing size with the recovered bytes given to the free-space marker,
    /// and a collection that ends up empty returns its block to the allocator.
    /// A nil reference (address 0 or `UNDEF_ADDR`) names no object. The
    /// sequence length does not decide: this crate's writers store even the
    /// empty string as a real heap object, so a zero-length reference with a
    /// defined address still holds one that must be released.
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

        for (addr, indices) in per_collection {
            let head = self.handle.read_at_most(addr, 64)?;
            let declared = GlobalHeapCollection::decode_size(&head, &self.ctx)?;
            let image = self.handle.read_at(addr, declared)?;
            let (mut gcol, _) = GlobalHeapCollection::decode(&image, &self.ctx)?;
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
            } else {
                let rewritten = gcol.encode_at_size(&self.ctx, declared)?;
                self.handle.write_at(addr, &rewritten)?;
            }
        }
        Ok(())
    }

    /// Add an attribute to a dataset.
    ///
    /// The attribute will be written as a message in the dataset's object
    /// header when the file is finalized.
    pub fn add_dataset_attribute(&self, ds_index: usize, attr: AttributeMessage) -> IoResult<()> {
        let count = self.dataset_count();
        if ds_index >= count {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset index {} out of range (have {})",
                ds_index, count
            )));
        }
        self.ds(ds_index).lock().attributes.push(attr);
        Ok(())
    }

    /// Add (or replace) an attribute on a group identified by its full path.
    ///
    /// The attribute is written into the group's object header when the
    /// file is finalized. An existing attribute with the same name is
    /// replaced, matching [`add_root_attribute`](Self::add_root_attribute).
    pub fn add_group_attribute(&self, group_path: &str, attr: AttributeMessage) -> IoResult<()> {
        for grp in self.group_refs() {
            let mut g = grp.lock();
            if g.name == group_path && !g.deleted {
                let attrs = &mut g.attributes;
                if let Some(pos) = attrs.iter().position(|a| a.name == attr.name) {
                    attrs[pos] = attr;
                } else {
                    attrs.push(attr);
                }
                return Ok(());
            }
        }
        Err(crate::io::IoError::NotFound(format!(
            "group '{}' not found",
            group_path
        )))
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
    /// One global heap collection is allocated per attribute, matching the
    /// per-call collection of [`create_vlen_string_dataset`](Self::create_vlen_string_dataset).
    /// A single shared attribute heap would avoid the per-attribute padding
    /// (`H5HG_MINALLOC` = 4096 bytes) but is a heap-management change that
    /// would also need to cover the dataset path.
    pub fn vlen_string_attribute(&self, name: &str, value: &str) -> IoResult<AttributeMessage> {
        use crate::format::global_heap::{encode_vlen_reference, GlobalHeapCollection};
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::datatype::DatatypeMessage;

        let mut gcol = GlobalHeapCollection::new();
        let obj_idx = gcol.add_object(value.as_bytes().to_vec())?;
        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

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
    /// `shape` (the public setters validate it before calling). One global heap
    /// collection is allocated for the whole array (all elements share it),
    /// matching the per-attribute collection of the scalar path.
    pub fn vlen_string_array_attribute(
        &mut self,
        name: &str,
        values: &[&str],
        shape: &[u64],
    ) -> IoResult<AttributeMessage> {
        use crate::format::global_heap::{encode_vlen_reference, GlobalHeapCollection};
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::datatype::DatatypeMessage;

        debug_assert_eq!(
            values.len() as u64,
            shape.iter().product::<u64>(),
            "vlen_string_array_attribute values.len() must equal product(shape)"
        );

        let mut gcol = GlobalHeapCollection::new();
        // (byte length, heap object index) per element, in order.
        let mut entries: Vec<(u32, u16)> = Vec::with_capacity(values.len());
        for v in values {
            let obj_idx = gcol.add_object(v.as_bytes().to_vec())?;
            entries.push((crate::format::global_heap::vlen_seq_len(v.len())?, obj_idx));
        }
        let gcol_encoded = gcol.encode(&self.ctx);
        let gcol_addr = self.allocator.allocate(gcol_encoded.len() as u64);
        self.handle.write_at(gcol_addr, &gcol_encoded)?;

        let mut data = Vec::with_capacity(values.len() * 16);
        for (len, obj_idx) in &entries {
            data.extend_from_slice(&encode_vlen_reference(
                *len,
                gcol_addr,
                *obj_idx as u32,
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

    /// Place `frames` in chunk `chunk_idx`, `offset_in_chunk` bytes from its
    /// start, keeping every byte outside that span.
    ///
    /// The single owner of an append's chunk write. A span covering the whole
    /// chunk is written straight through; a narrower one is read-modify-write,
    /// because the bytes around it belong to frames appended earlier. Building
    /// a fresh fill-value buffer for a chunk that already holds frames is what
    /// erased them. A chunk the index does not reach has had no write land in
    /// it, so the fill value *is* its content — the rule
    /// [`write_slice`](Self::write_slice) already applies to a partially
    /// covered chunk.
    pub(crate) fn append_frames_into_chunk(
        &self,
        ds_index: usize,
        chunk_idx: u64,
        offset_in_chunk: usize,
        frames: &[u8],
    ) -> IoResult<()> {
        let chunk_bytes = {
            let ds = self.ds(ds_index);
            let m = ds.lock();
            let chunked = m
                .chunked
                .as_ref()
                .ok_or_else(|| crate::io::IoError::InvalidState("not a chunked dataset".into()))?;
            chunked.chunk_dims.iter().product::<u64>() as usize * m.datatype.element_size() as usize
        };
        if offset_in_chunk + frames.len() > chunk_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "{} bytes at offset {offset_in_chunk} do not fit chunk {chunk_idx}, \
                 which is {chunk_bytes} bytes",
                frames.len()
            )));
        }
        if offset_in_chunk == 0 && frames.len() == chunk_bytes {
            return self.write_chunk(ds_index, chunk_idx, frames);
        }
        let mut chunk_buf = match self.read_chunk_if_present(ds_index, chunk_idx)? {
            Some(existing) if existing.len() == chunk_bytes => existing,
            Some(existing) => {
                return Err(crate::io::IoError::InvalidState(format!(
                    "stored chunk {chunk_idx} is {} bytes but the chunk shape needs \
                     {chunk_bytes}",
                    existing.len()
                )))
            }
            None => self.new_chunk_buffer(ds_index, chunk_bytes),
        };
        chunk_buf[offset_in_chunk..offset_in_chunk + frames.len()].copy_from_slice(frames);
        self.write_chunk(ds_index, chunk_idx, &chunk_buf)
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
        let linear = geo.linear_index(chunk_coords)?;
        match geo.kind {
            ChunkIndexKind::ExtensibleArray => self.read_chunk_if_present(ds_index, linear),
            ChunkIndexKind::FixedArray => {
                let ds = self.ds(ds_index);
                let m = ds.lock();
                let pipeline = m.filter_pipeline.clone();
                let fa = m.fixed_array.as_ref().unwrap();
                let lidx = linear as usize;
                let (addr, nbytes, mask) = if pipeline.is_some() {
                    match fa.fa_dblk.filtered_elements.get(lidx) {
                        Some(e) => (e.address, e.chunk_size as u64, e.filter_mask),
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
                        .map(|r| (r.chunk_address, r.chunk_size as u64, r.filter_mask))
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
                self.write_chunk(ds_index, linear, data)
            }
            ChunkIndexKind::FixedArray => {
                self.write_chunk_fixed_array(ds_index, chunk_coords, data)
            }
            ChunkIndexKind::BtreeV2 => self.write_chunk_btree_v2(ds_index, chunk_coords, data),
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

    /// Define a chunked dataset indexed by a fixed array (no unlimited dimensions).
    ///
    /// `dims` and `max_dims` should be the same (all fixed). `chunk_dims` defines the
    /// chunk shape. Returns the dataset index.
    pub fn create_fixed_array_dataset(
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk_dims: &[u64],
    ) -> IoResult<usize> {
        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
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

        let idx = self.push_dataset(DatasetInfo {
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
            fill_value: None,
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
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
        &self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk_dims: &[u64],
        pipeline: FilterPipeline,
    ) -> IoResult<usize> {
        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
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

        let idx = self.push_dataset(DatasetInfo {
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
            fill_value: None,
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });

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
        use crate::format::chunk_index::btree_v2::{
            compute_chunk_size_len, Bt2Header, BT2_NODE_SIZE,
        };

        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
        self.ensure_unique_dataset_name(name)?;
        let ndims = dims.len();
        if chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk shape has {} dimensions but the dataspace has {}",
                chunk_dims.len(),
                ndims
            )));
        }

        // The filtered record's size field is as wide as libhdf5 will
        // recompute it from the uncompressed chunk size, exactly as the
        // extensible- and fixed-array filtered paths size theirs.
        let bt2_index = match pipeline {
            Some(_) => {
                let chunk_bytes: u64 =
                    chunk_dims.iter().product::<u64>() * datatype.element_size() as u64;
                let len = compute_chunk_size_len(chunk_bytes);
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
        if (BT2_NODE_SIZE as usize) < 10 + 3 * record_size {
            return Err(crate::io::IoError::InvalidState(format!(
                "a {ndims}-dimension v2 B-tree record is {record_size} bytes, too wide \
                 for a {BT2_NODE_SIZE}-byte node"
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
            dims: dims.to_vec(),
            max_dims: Some(max_dims.to_vec()),
        };

        let idx = self.push_dataset(DatasetInfo {
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
            fill_value: None,
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });

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

        let idx = self.push_dataset(DatasetInfo {
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });

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
        // Hold the create gate across the uniqueness check and the registry
        // push so the two are atomic (see `create_lock`).
        let _create = self.create_lock.lock();
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

        let idx = self.push_dataset(DatasetInfo {
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
            append_buffer: Vec::new(),
            append_buffered_frames: 0,
        });
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

        // Compute linear chunk index from multidimensional coordinates.
        let dims = &m.dataspace.dims;
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

        // Update the fixed array data block. The slot is read before the bytes
        // are placed so a rewrite can stay where it is (see `place_chunk`).
        let fa = m.fixed_array.as_mut().unwrap();
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
                let old = &fa.fa_dblk.filtered_elements[lidx];
                let chunk_addr = self.place_chunk(
                    Some((old.address, old.chunk_size as u64)),
                    stored_size as u64,
                );
                self.handle.write_at(chunk_addr, final_bytes)?;
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
                .map(|r| (r.chunk_address, r.chunk_size as u64))
        } else {
            bt2.index
                .lookup(chunk_coords)
                .map(|r| (r.chunk_address, chunk_bytes))
        };
        let chunk_addr = self.place_chunk(old, stored_len);
        self.handle.write_at(chunk_addr, final_bytes)?;

        let bt2 = m.btree_v2.as_mut().unwrap();
        if bt2.index.filtered {
            bt2.index.insert_filtered(
                chunk_coords.to_vec(),
                chunk_addr,
                stored_len as u32,
                filter_mask,
            );
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
        // Fallback: sequential (write_chunk_fixed_array compresses per chunk).
        for (coords, data) in chunks {
            self.write_chunk_fixed_array(ds_index, coords, data)?;
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
            if let Some(ref max) = m.dataspace.max_dims {
                if new > max[d] {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "extend_dataset dimension {d} ({new}) exceeds the maximum {}",
                        max[d]
                    )));
                }
            }
        }
        m.dataspace.dims = new_dims.to_vec();
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
    pub fn set_dataset_extent(&self, index: usize, new_dims: &[u64]) -> IoResult<()> {
        let ds = self.ds(index);
        let mut m = ds.lock();
        let is_unindexed = m.chunked.is_none() && m.fixed_array.is_none() && m.btree_v2.is_none();
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
        // A pending append buffer is positioned relative to the current
        // logical size; changing the extent underneath it would make
        // `flush_append_buffers` write the chunk at the wrong index.
        if m.append_buffered_frames > 0 {
            return Err(crate::io::IoError::InvalidState(
                "set_extent cannot run while the dataset has buffered appends; \
                 flush them first"
                    .into(),
            ));
        }
        if let Some(ref max) = m.dataspace.max_dims {
            for (d, (&new, &mx)) in new_dims.iter().zip(max).enumerate() {
                if new > mx {
                    return Err(crate::io::IoError::InvalidState(format!(
                        "set_extent dimension {d} ({new}) exceeds the maximum {mx}"
                    )));
                }
            }
        }
        m.dataspace.dims = new_dims.to_vec();
        Ok(())
    }

    /// Flush a chunked dataset's index structures to disk (durable).
    ///
    /// Writes the index blocks and issues an `fdatasync` so the data is
    /// durable — the guarantee SWMR readers and standalone callers rely on.
    pub fn flush_dataset(&self, index: usize) -> IoResult<()> {
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
        for i in 0..self.dataset_count() {
            let is_indexed = {
                let ds = self.ds(i);
                let m = ds.lock();
                m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some()
            };
            if is_indexed {
                self.flush_dataset(i)?;
            }
        }

        // 1. Write each dataset's object header.
        for i in 0..self.dataset_count() {
            let ds_header = self.build_dataset_header(i);
            let encoded = ds_header.encode();
            let encoded_size = encoded.len();
            let addr = self.allocator.allocate(encoded_size as u64);
            self.handle.write_at(addr, &encoded)?;
            let ds = self.ds(i);
            let mut m = ds.lock();
            m.obj_header_addr = addr;
            m.obj_header_written_addr = Some(addr);
            m.obj_header_encoded_size = encoded_size;
        }

        // 1b. Group object headers. A hard link can point to a group whose
        // header is written later, so addresses are assigned in a first
        // pass (a header's encoded size is independent of the address
        // values it carries) and the content is written in a second.
        for gi in 0..self.group_count() {
            let size = self.build_group_header(gi).encode().len() as u64;
            self.grp(gi).lock().obj_header_addr = self.allocator.allocate(size);
        }
        for gi in 0..self.group_count() {
            let encoded = self.build_group_header(gi).encode();
            let addr = self.grp(gi).lock().obj_header_addr;
            self.handle.write_at(addr, &encoded)?;
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
        // Readers can now be following this file, so a chunk that moves must
        // leave its old block intact for whoever is still holding the previous
        // index (see `swmr_active`).
        self.swmr_active = true;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Flush any partial append buffers into the chunks they belong to,
    /// through [`append_frames_into_chunk`](Self::append_frames_into_chunk):
    /// frames already in the chunk survive, and the rest of it reads back as
    /// the dataset's fill value (zeros when none is defined).
    fn flush_append_buffers(&mut self) -> IoResult<()> {
        for i in 0..self.dataset_count() {
            // Snapshot everything needed under one brief slot guard, then drop
            // it: `new_chunk_buffer` and `write_chunk` below re-lock the slot.
            let (buf, buffered_frames, chunk_dims, es, dims) = {
                let ds = self.ds(i);
                let mut m = ds.lock();
                if m.append_buffer.is_empty() {
                    continue;
                }
                let chunk_dims = if let Some(ref c) = m.chunked {
                    c.chunk_dims.clone()
                } else if let Some(ref f) = m.fixed_array {
                    f.chunk_dims.clone()
                } else if let Some(ref b) = m.btree_v2 {
                    b.chunk_dims.clone()
                } else {
                    continue;
                };
                let es = m.datatype.element_size() as usize;
                let buffered_frames = m.append_buffered_frames as usize;
                let dims = m.dataspace.dims.clone();
                let buf = std::mem::take(&mut m.append_buffer);
                m.append_buffered_frames = 0;
                (buf, buffered_frames, chunk_dims, es, dims)
            };

            let chunk_dim0 = chunk_dims[0] as usize;
            let current_dim0 = dims[0] as usize;
            let base_frame = current_dim0 - buffered_frames;
            let chunk_idx = base_frame / chunk_dim0;

            let frame_bytes = if dims.len() > 1 {
                dims[1..].iter().map(|&d| d as usize).product::<usize>() * es
            } else {
                es
            };
            let offset_in_chunk = (base_frame % chunk_dim0) * frame_bytes;
            self.append_frames_into_chunk(i, chunk_idx as u64, offset_in_chunk, &buf)?;
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
                if m.obj_header_written_addr.is_some() {
                    let modified = m.chunked.as_ref().is_some_and(|c| c.chunks_written > 0);
                    if !modified {
                        continue;
                    }
                }
                let is_indexed =
                    m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some();
                if !is_indexed {
                    continue;
                }
            }
            self.flush_dataset_synced(i, sync)?;
        }

        // 1. Write each dataset's object header.
        for i in 0..self.dataset_count() {
            let ds = self.ds(i);
            {
                let mut m = ds.lock();
                if m.obj_header_written_addr.is_some() {
                    // Existing dataset from append mode.
                    // If it has chunked info with chunks_written > 0, it was modified
                    // and needs a new object header.
                    let modified = m.chunked.as_ref().is_some_and(|c| c.chunks_written > 0);
                    if !modified {
                        // Keep the original object header address for the root group link.
                        m.obj_header_addr = m.obj_header_written_addr.unwrap();
                        continue;
                    }
                }
            }
            let ds_header = self.build_dataset_header(i);
            let encoded = ds_header.encode();
            let addr = self.allocator.allocate(encoded.len() as u64);
            self.handle.write_at(addr, &encoded)?;
            ds.lock().obj_header_addr = addr;
        }

        // 1b. Group object headers. A hard link can point to a group whose
        // header is written later, so addresses are assigned in a first
        // pass (a header's encoded size is independent of the address
        // values it carries) and the content is written in a second.
        for gi in 0..self.group_count() {
            let size = self.build_group_header(gi).encode().len() as u64;
            self.grp(gi).lock().obj_header_addr = self.allocator.allocate(size);
        }
        for gi in 0..self.group_count() {
            let encoded = self.build_group_header(gi).encode();
            let addr = self.grp(gi).lock().obj_header_addr;
            self.handle.write_at(addr, &encoded)?;
        }

        // 2. Write root group object header.
        let root_header = self.build_root_group_header();
        let root_encoded = root_header.encode();
        let root_addr = self.allocator.allocate(root_encoded.len() as u64);
        self.handle.write_at(root_addr, &root_encoded)?;
        self.root_group_addr = Some(root_addr);

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

        // Dataspace message (type 0x01)
        let ds_msg = m.dataspace.encode(&self.ctx);
        header.add_message(MSG_DATASPACE, 0x00, ds_msg);

        // Datatype message (type 0x03), flag 0x01 = constant
        let dt_msg = m.datatype.encode(&self.ctx);
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
        let fv_msg = fv.encode();
        header.add_message(MSG_FILL_VALUE, 0x00, fv_msg);

        // Data Layout message (type 0x08)
        let layout = if let Some(ref chunked) = m.chunked {
            let mut layout_dims = chunked.chunk_dims.clone();
            layout_dims.push(m.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_earray(
                layout_dims,
                chunked.earray_params.clone(),
                chunked.ea_header_addr,
            )
        } else if let Some(ref fa) = m.fixed_array {
            let mut layout_dims = fa.chunk_dims.clone();
            layout_dims.push(m.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_farray(
                layout_dims,
                FixedArrayParams::default_params(),
                fa.fa_header_addr,
            )
        } else if let Some(ref bt2) = m.btree_v2 {
            let mut layout_dims = bt2.chunk_dims.clone();
            layout_dims.push(m.datatype.element_size() as u64);
            DataLayoutMessage::chunked_v4_btree_v2(layout_dims, bt2.bt2_header_addr)
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

        // Attribute messages (type 0x0C)
        for attr in &m.attributes {
            let attr_msg = attr.encode(&self.ctx);
            header.add_message(MSG_ATTRIBUTE, 0x00, attr_msg);
        }

        // Object Reference Count message (type 0x16): emitted only when
        // more than one hard link resolves to this dataset (computed above).
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

        // Snapshot the group's child lists and attributes, then drop the slot
        // guard: the per-child reads and emit_hard_links/object_link_count
        // below re-lock dataset and group slots (including this one).
        let (child_datasets, child_groups, attributes) = {
            let grp = self.grp(group_idx);
            let g = grp.lock();
            (
                g.child_datasets.clone(),
                g.child_groups.clone(),
                g.attributes.clone(),
            )
        };

        // Link messages for child datasets (skip deleted)
        for ds_idx in child_datasets {
            let ds = self.ds(ds_idx);
            let m = ds.lock();
            if m.deleted {
                continue;
            }
            let leaf_name = m.name.rsplit('/').next().unwrap_or(&m.name);
            let link = LinkMessage::hard(leaf_name, m.obj_header_addr);
            let link_msg = link.encode(&self.ctx);
            header.add_message(MSG_LINK, 0x00, link_msg);
        }

        // Link messages for child groups (skip deleted)
        for child_idx in child_groups {
            let child_grp = self.grp(child_idx);
            let g = child_grp.lock();
            if g.deleted {
                continue;
            }
            let leaf_name = g.name.rsplit('/').next().unwrap_or(&g.name);
            let link = LinkMessage::hard(leaf_name, g.obj_header_addr);
            let link_msg = link.encode(&self.ctx);
            header.add_message(MSG_LINK, 0x00, link_msg);
        }

        // User-created hard links whose parent is this group.
        self.emit_hard_links(&mut header, Some(group_idx));

        // Attribute messages (type 0x0C) -- e.g. NeXus `NX_class`.
        for attr in &attributes {
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

        // Collect dataset indices that belong to a subgroup (not the root
        // group). Each group slot is locked one at a time, then released.
        let mut datasets_in_subgroups: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for grp in self.group_refs() {
            let g = grp.lock();
            if g.deleted {
                continue;
            }
            datasets_in_subgroups.extend(g.child_datasets.iter().copied());
        }

        // Link messages for root-level datasets. `dataset_refs` preserves
        // registry order, so `enumerate` yields each dataset's true index.
        for (i, ds) in self.dataset_refs().into_iter().enumerate() {
            let m = ds.lock();
            if m.deleted {
                continue;
            }
            if !datasets_in_subgroups.contains(&i) {
                let link = LinkMessage::hard(&m.name, m.obj_header_addr);
                let link_msg = link.encode(&self.ctx);
                header.add_message(MSG_LINK, 0x00, link_msg);
            }
        }

        // Link messages for root-level groups (those with no parent)
        for grp in self.group_refs() {
            let g = grp.lock();
            if g.deleted {
                continue;
            }
            if g.parent.is_none() {
                let leaf_name = g.name.rsplit('/').next().unwrap_or(&g.name);
                let link = LinkMessage::hard(leaf_name, g.obj_header_addr);
                let link_msg = link.encode(&self.ctx);
                header.add_message(MSG_LINK, 0x00, link_msg);
            }
        }

        // User-created hard links in the root group.
        self.emit_hard_links(&mut header, None);

        // Root-level attributes
        for attr in self.root_attributes.lock().iter() {
            let attr_msg = attr.encode(&self.ctx);
            header.add_message(MSG_ATTRIBUTE, 0x00, attr_msg);
        }

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

    /// A concurrent append publishes its buffered count and extends the dims
    /// under two different lock acquisitions, so a slice update can observe a
    /// count that exceeds the extent. That half-published state must come back
    /// as an error, not wrap `dims[0] - buffered` around zero.
    #[test]
    fn vlen_slice_reports_a_half_published_append_count() {
        let path = temp_path("vlen_slice_half_published");

        let writer = Hdf5Writer::create(&path).unwrap();
        let idx = writer.create_vlen_string_dataset("d", &["a", "b"]).unwrap();
        writer.ds(idx).lock().append_buffered_frames = 3; // dims[0] == 2
        let err = writer.write_vlen_strings_slice(idx, 0, &["x"]).unwrap_err();
        assert!(
            err.to_string().contains("append buffer holds 3 frames"),
            "unexpected error: {err}"
        );

        writer.ds(idx).lock().append_buffered_frames = 0;
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
