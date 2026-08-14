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
use crate::format::messages::fill_value::{try_tiled_fill, FillValueMessage};
use crate::format::messages::filter::{self, FilterPipeline};
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link::LinkTarget;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::*;
use crate::format::object_header::ObjectHeader;
use crate::format::superblock::{
    detect_superblock_version, SuperblockV0V1, SuperblockV2V3, SymbolTableCache,
};
use crate::format::symbol_table::SymbolTableNode;
use crate::format::{FormatContext, UNDEF_ADDR};

use crate::io::file_handle::FileHandle;
#[cfg(feature = "mmap")]
use crate::io::file_handle::MmapFileHandle;
use crate::io::hyperslab::{compute_strides, for_each_contiguous_run};
use crate::io::IoResult;

/// The version-4 chunk-index descriptor pulled from a data-layout message:
/// the index kind, its address, and the per-kind parameters the reader needs
/// to walk it. Bundled so the chunked read entry point takes one descriptor
/// instead of a long parameter list.
struct ChunkIndexDesc<'a> {
    /// The kind of chunk index (single chunk, fixed/extensible array, …).
    index_type: data_layout::ChunkIndexType,
    /// Address of the chunk index structure (or the chunk itself, for a
    /// single chunk). `UNDEF_ADDR` when unallocated.
    index_address: u64,
    /// Extensible-array parameters (present iff `index_type == ExtensibleArray`).
    earray_params: Option<&'a data_layout::EarrayParams>,
    /// Filtered single-chunk parameters (present iff `index_type ==
    /// SingleChunk` and the layout's filtered flag is set): the chunk's exact
    /// on-disk size and per-chunk filter mask.
    single_chunk_filter: Option<data_layout::SingleChunkFilter>,
}

/// What a chunked read should produce: the whole dataset, or one hyperslab.
///
/// Threaded through every chunked reader so the index walk, raw read, and
/// filter pipeline are shared between full reads and slice reads. For a
/// `Slice`, the reader allocates a `counts`-shaped buffer, skips reading any
/// chunk that does not overlap the selection (the I/O win), and scatters only
/// the chunk∩selection intersection. `Full` reads and places every chunk.
#[derive(Clone, Copy)]
enum ChunkTarget<'a> {
    Full,
    Slice {
        starts: &'a [u64],
        counts: &'a [u64],
    },
}

impl<'a> ChunkTarget<'a> {
    /// Dimensions of the produced output buffer: the dataset dims for `Full`,
    /// the selection extent for `Slice`.
    fn out_dims(&self, dims: &'a [u64]) -> &'a [u64] {
        match self {
            ChunkTarget::Full => dims,
            ChunkTarget::Slice { counts, .. } => counts,
        }
    }

    /// Whether a chunk at chunk-grid `coords` (extent `chunk_dims`) intersects
    /// the target. `Full` always intersects; a `Slice` intersects iff every
    /// dimension's chunk span `[origin, origin+chunk_dims)` overlaps the
    /// selection span `[start, start+count)`.
    fn overlaps(&self, coords: &[u64], chunk_dims: &[u64]) -> bool {
        match self {
            ChunkTarget::Full => true,
            ChunkTarget::Slice { starts, counts } => coords.iter().enumerate().all(|(d, &c)| {
                let origin = c.saturating_mul(chunk_dims[d]);
                let chunk_end = origin.saturating_add(chunk_dims[d]);
                let sel_end = starts[d].saturating_add(counts[d]);
                origin < sel_end && starts[d] < chunk_end
            }),
        }
    }
}

/// Read-side metadata for a single dataset.
pub struct DatasetReadInfo {
    /// Dataset name (the link name in the root group).
    pub name: String,
    /// Element datatype.
    pub datatype: DatatypeMessage,
    /// Dataspace (dimensionality).
    pub dataspace: DataspaceMessage,
    /// Data layout (contiguous, compact, or chunked).
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

/// The class of one link record in a group: what `H5Lget_info` reports,
/// carrying the value `H5Lget_val` returns for the classes that have one.
///
/// Every link a group holds gets one of these, whether or not the object it
/// names can be opened — a listing is a listing of links, not of objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkClass {
    /// Another name for an object in this file.
    Hard,
    /// A path inside this file, resolved when the link is traversed.
    Soft { path: String },
    /// A path inside another file. Listed and reported, but not followed.
    External { file: String, path: String },
    /// A user-defined link class this reader has no interpreter for; libhdf5
    /// needs a registered link class for these too.
    UserDefined { link_type: u8 },
}

impl LinkClass {
    pub(crate) fn from_target(target: &LinkTarget) -> Self {
        match target {
            LinkTarget::Hard { .. } => Self::Hard,
            LinkTarget::Soft { target } => Self::Soft {
                path: target.clone(),
            },
            LinkTarget::External { file, path } => Self::External {
                file: file.clone(),
                path: path.clone(),
            },
            LinkTarget::UserDefined { link_type, .. } => Self::UserDefined {
                link_type: *link_type,
            },
        }
    }
}

/// The soft link a traversal crossed, kept so a lookup that finds nothing can
/// say the link dangles instead of reporting a bare absence.
struct SoftLinkRef {
    link: String,
    target: String,
}

/// What path traversal produced.
enum Traversal {
    /// A path in this file, after every group hard-link alias and soft link
    /// on it was followed. `via` names the last soft link crossed, if any.
    Path {
        path: String,
        via: Option<SoftLinkRef>,
    },
    /// A component of the path is an external link, so the path leaves this
    /// file. `path` is the remainder inside `file`.
    External {
        link: String,
        file: String,
        path: String,
    },
}

/// One rewrite a traversal step can apply to the path prefix it matched.
#[derive(Clone, Copy)]
enum Rewrite<'a> {
    /// A group hard link: continue from the group's first-walked path.
    Alias(&'a str),
    /// A soft link: continue from its value.
    Soft(&'a str),
    /// An external link: stop, the rest of the path is in another file.
    External { file: &'a str, path: &'a str },
}

/// Resolve a soft link's value against the group the link lives in, the way
/// `H5G_traverse` does: a value starting with `/` is absolute, anything else
/// is relative to that group. `.` and `..` components fold. The result has no
/// leading `/`.
fn resolve_link_value(link_path: &str, value: &str) -> String {
    let mut components: Vec<&str> = Vec::new();
    if !value.starts_with('/') {
        // The link's own parent group is everything before its last component.
        if let Some(parent) = link_path.rsplit_once('/').map(|(p, _)| p) {
            components.extend(parent.split('/').filter(|c| !c.is_empty()));
        }
    }
    for component in value.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            c => components.push(c),
        }
    }
    components.join("/")
}

/// Everything one discovery walk found: the objects, the link records that
/// name them, and the group metadata the lookup paths need. Carried as one
/// value so the walk has a single owner rather than a widening tuple, and so
/// every walk (link-message and symbol-table alike) fills the same fields.
#[derive(Default)]
struct Catalog {
    datasets: Vec<DatasetReadInfo>,
    /// Dataset-shaped objects this crate cannot read, keyed by path; the
    /// value names what stopped it. They are listed exactly like readable
    /// datasets — the name is in the file either way — and refuse typed
    /// access with that reason.
    unreadable: std::collections::BTreeMap<String, String>,
    /// Attributes on non-root groups, keyed by group path.
    group_attributes: std::collections::HashMap<String, Vec<AttributeMessage>>,
    /// Every non-root group path the walk traversed into.
    group_paths: std::collections::BTreeSet<String>,
    /// Group hard-link aliases: alias path → first-walked path.
    group_aliases: std::collections::HashMap<String, String>,
    /// Every link record seen, keyed by its full path (no leading `/`).
    links: std::collections::BTreeMap<String, LinkClass>,
}

/// What an object header describes.
///
/// libhdf5 decides an object's class from *which* messages the header holds
/// (`H5O_obj_class`) and only then reads their contents; the two questions are
/// separate, and answering them with one `Option<DatasetReadInfo>` is what let
/// a dataset whose datatype this crate cannot decode leave the catalog as if
/// the file did not contain it. Each outcome now has its own name, so no
/// caller can turn "unreadable" back into "absent".
enum ObjectKind {
    /// A dataset: it carries a datatype, a dataspace and a data layout, and
    /// every message the payload depends on decoded.
    Dataset(Box<DatasetReadInfo>),
    /// A dataset whose payload depends on a message this crate cannot decode.
    /// The string names what stopped it and reaches the caller of any typed
    /// access to the name.
    UnreadableDataset(String),
    /// A group: it carries link, link-info, symbol-table or group-info
    /// storage, or none of the messages that identify anything else.
    Group,
    /// A committed (named) datatype object: a datatype message with neither
    /// group storage nor the dataspace/layout pair a dataset needs.
    CommittedDatatype,
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
    /// Dataset-shaped objects this crate cannot read, keyed by path (no
    /// leading `/`), the value naming what stopped it. They are listed with
    /// the readable datasets and refuse typed access with that reason: an
    /// object the file contains is never reported as one it does not.
    unreadable: std::collections::BTreeMap<String, String>,
    /// Attributes on the root group (file-level attributes).
    root_attributes: Vec<AttributeMessage>,
    /// Attributes on non-root groups, keyed by group path (no leading `/`).
    group_attributes: std::collections::HashMap<String, Vec<AttributeMessage>>,
    /// Every non-root group path the discovery walk traversed into (no
    /// leading `/`), regardless of whether the group has datasets or
    /// attributes. Built from actual link records, so empty groups,
    /// attribute-only groups, and subgroup-only groups are all included.
    group_paths: std::collections::BTreeSet<String>,
    /// Group hard links: alias path → the first-walked path of the same
    /// group object header (both without a leading `/`). The walk
    /// descends each header once, so objects under the alias are stored
    /// under the first path; lookups resolve alias prefixes through this
    /// map, as HDF5 path traversal does.
    group_aliases: std::collections::HashMap<String, String>,
    /// Every link record in the file, keyed by full path (no leading `/`).
    /// A listing is a listing of links, so this holds soft and external
    /// links as well as the hard links that name the objects above.
    links: std::collections::BTreeMap<String, LinkClass>,
}

/// Total byte length of `dims.product() * element_size`, computed with
/// saturating arithmetic. `dims` and `element_size` are file-derived; a
/// crafted file with huge dimensions thus yields a saturated (too-large)
/// value — rejected downstream by the file-size/buffer checks — rather
/// than panicking in a debug build or wrapping in release.
fn saturating_byte_len(dims: &[u64], element_size: u64) -> u64 {
    dims.iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(d))
        .saturating_mul(element_size)
}

/// Materialize a `total`-byte fill buffer, mapping allocation failure to a
/// clean error. `total` on a read path comes from untrusted file fields, so
/// a crafted file declaring an absurd dataset size would otherwise abort the
/// process when `vec![0u8; total]` fails to allocate.
fn alloc_tiled_fill(total: usize, fill_value: Option<&[u8]>) -> IoResult<Vec<u8>> {
    try_tiled_fill(total, fill_value).map_err(|_| {
        crate::io::IoError::InvalidState(format!(
            "cannot allocate {total} bytes for dataset buffer (file may be corrupt)"
        ))
    })
}

/// Fill an existing buffer in place with the dataset's tiled fill value (or
/// zero when no fill value is set), matching [`try_tiled_fill`]'s tiling.
///
/// Used to initialize a read destination — both an internally-allocated `Vec`
/// and a caller-provided read-into buffer (whose prior contents are arbitrary)
/// — so any region a chunked read leaves untouched reads back as the fill
/// value rather than stale bytes.
fn fill_tiled_into(out: &mut [u8], fill_value: Option<&[u8]>) {
    out.fill(0);
    if let Some(fv) = fill_value {
        if !fv.is_empty() && !out.is_empty() {
            for slot in out.chunks_mut(fv.len()) {
                let n = slot.len().min(fv.len());
                slot[..n].copy_from_slice(&fv[..n]);
            }
        }
    }
}

/// One chunk's on-disk read request, built by a read path before any I/O.
struct ChunkReadJob {
    /// Byte offset of the chunk's stored bytes.
    addr: u64,
    /// Number of bytes to read.
    len: usize,
    /// `true` → [`FileHandle::read_at_most`] (short reads near EOF are fine);
    /// `false` → [`FileHandle::read_at`] (exact, errors on a short read).
    at_most: bool,
    /// Per-chunk filter mask (ignored when the pipeline is `None`).
    mask: u32,
}

/// Read one chunk's raw bytes according to its job.
fn read_chunk_raw(handle: &FileHandle, j: &ChunkReadJob) -> IoResult<Vec<u8>> {
    if j.at_most {
        Ok(handle.read_at_most(j.addr, j.len)?)
    } else {
        Ok(handle.read_at(j.addr, j.len)?)
    }
}

/// Run the reverse filter pipeline (if any) over one chunk's raw bytes.
fn decompress_chunk(
    pipeline: Option<&FilterPipeline>,
    raw: Vec<u8>,
    mask: u32,
) -> IoResult<Vec<u8>> {
    match pipeline {
        Some(pl) => Ok(filter::reverse_filters_masked(pl, &raw, mask)?),
        None => Ok(raw),
    }
}

/// Read and decompress a batch of chunk jobs, preserving job order.
///
/// `jobs[i] == None` yields `Ok(None)` — a chunk skipped as out-of-selection
/// or unallocated. Otherwise the chunk's raw bytes are read and, when
/// `pipeline` is `Some`, run through the reverse filter pipeline with the
/// job's mask. This is the single owner of the read-then-decompress step for
/// every chunk index type; each read path only builds the jobs and scatters
/// the results.
///
/// On Unix and Windows, positioned reads at distinct offsets on a shared
/// `&File` each carry their own explicit offset and never consult a shared file
/// cursor (on Windows the cursor may move as a side effect, but nothing reads
/// it), so read + decompress run fused in one parallel pass — overlapping chunk
/// I/O across cores, which the
/// C library's default (non-MPI) path does not do. On targets with neither
/// positioned API the seek-based fallback shares the file cursor, so reads
/// stay serial there while decompression still parallelizes.
fn read_and_decompress_chunks(
    handle: &FileHandle,
    pipeline: Option<&FilterPipeline>,
    jobs: Vec<Option<ChunkReadJob>>,
) -> IoResult<Vec<Option<Vec<u8>>>> {
    #[cfg(all(feature = "parallel", any(unix, windows)))]
    {
        use rayon::prelude::*;
        // Fused read + decompress for one job.
        let decode = |job: Option<ChunkReadJob>| -> IoResult<Option<Vec<u8>>> {
            match job {
                Some(j) => Ok(Some(decompress_chunk(
                    pipeline,
                    read_chunk_raw(handle, &j)?,
                    j.mask,
                )?)),
                None => Ok(None),
            }
        };
        // Run on rust-hdf5's private half-cores pool, not rayon's global pool;
        // fall back to serial if the pool could not be built.
        match crate::parallel::io_pool() {
            Some(pool) => pool.install(|| {
                jobs.into_par_iter()
                    .map(&decode)
                    .collect::<IoResult<Vec<_>>>()
            }),
            None => jobs.into_iter().map(decode).collect::<IoResult<Vec<_>>>(),
        }
    }
    #[cfg(all(feature = "parallel", not(any(unix, windows))))]
    {
        use rayon::prelude::*;
        // No positioned read API here: concurrent reads would race the shared
        // file cursor, so read serially, then parallelize decompression on
        // rust-hdf5's private half-cores pool (not rayon's global pool).
        let raws: Vec<Option<(Vec<u8>, u32)>> = jobs
            .into_iter()
            .map(|job| match job {
                Some(j) => Ok(Some((read_chunk_raw(handle, &j)?, j.mask))),
                None => Ok(None),
            })
            .collect::<IoResult<Vec<_>>>()?;
        let decode = |r: Option<(Vec<u8>, u32)>| -> IoResult<Option<Vec<u8>>> {
            match r {
                Some((raw, mask)) => Ok(Some(decompress_chunk(pipeline, raw, mask)?)),
                None => Ok(None),
            }
        };
        // Fall back to serial if the private pool could not be built.
        match crate::parallel::io_pool() {
            Some(pool) => pool.install(|| {
                raws.into_par_iter()
                    .map(&decode)
                    .collect::<IoResult<Vec<_>>>()
            }),
            None => raws.into_iter().map(decode).collect::<IoResult<Vec<_>>>(),
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        jobs.into_iter()
            .map(|job| match job {
                Some(j) => Ok(Some(decompress_chunk(
                    pipeline,
                    read_chunk_raw(handle, &j)?,
                    j.mask,
                )?)),
                None => Ok(None),
            })
            .collect()
    }
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
        let handle = FileHandle::open_read_with_locking(path, locking)?;

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

        // Walk link messages to discover datasets, group attributes, and
        // every group path that exists.
        let catalog = Self::build_catalog_from_links(
            &mut handle,
            &root_header,
            sb.root_group_object_header_address,
            &ctx,
        )?;

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
            datasets: catalog.datasets,
            unreadable: catalog.unreadable,
            root_attributes,
            group_attributes: catalog.group_attributes,
            group_paths: catalog.group_paths,
            group_aliases: catalog.group_aliases,
            links: catalog.links,
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
        let root_obj_addr = ste.obj_header_addr;
        let ste_stab = ste.cached_symbol_table();
        let (ste_btree_addr, ste_heap_addr) = ste_stab.unwrap_or((UNDEF_ADDR, UNDEF_ADDR));

        // Read the root group's object header (following continuations).
        let root_hdr = Self::read_object_header_full(&mut handle, &ctx, root_obj_addr).ok();

        // Collect the root group's own attributes.
        let mut root_attributes = Vec::new();
        if let Some(ref h) = root_hdr {
            for m in &h.messages {
                if m.msg_type == MSG_ATTRIBUTE {
                    if let Ok((a, _)) = AttributeMessage::decode(&m.data, &ctx) {
                        root_attributes.push(a);
                    }
                }
            }
        }

        // A v0/v1-superblock file whose root group has migrated to link
        // storage (more than ~8 objects) carries `Link` / `Link Info`
        // messages in its object header; the superblock symbol-table
        // scratch-pad is then stale. Prefer the link-based walk when those
        // messages are present, and fall back to the symbol-table B-tree.
        let has_links = root_hdr.as_ref().is_some_and(|h| {
            h.messages
                .iter()
                .any(|m| m.msg_type == MSG_LINK || m.msg_type == MSG_LINK_INFO)
        });

        let catalog = if has_links {
            Self::build_catalog_from_links(
                &mut handle,
                root_hdr.as_ref().unwrap(),
                root_obj_addr,
                &ctx,
            )?
        } else {
            // Symbol-table storage: the STE scratch-pad caches the B-tree
            // and local heap; otherwise read them from the object header.
            let (btree_addr, heap_addr) = match ste_stab {
                Some(pair) => pair,
                None => {
                    // Read the symbol-table message from the already-loaded
                    // full root header (which followed continuation blocks).
                    root_hdr
                        .as_ref()
                        .map(|h| Self::stab_from_header(h, &ctx))
                        .unwrap_or((UNDEF_ADDR, UNDEF_ADDR))
                }
            };
            if btree_addr != UNDEF_ADDR && heap_addr != UNDEF_ADDR {
                Self::build_catalog_from_btree(
                    &mut handle,
                    &ctx,
                    btree_addr,
                    heap_addr,
                    root_obj_addr,
                )?
            } else {
                Catalog::default()
            }
        };

        Ok(Self {
            handle,
            ctx,
            _eof: sb.end_of_file_address,
            root_group_info: RootGroupInfo::V0V1 {
                root_obj_header_addr: root_obj_addr,
                btree_addr: ste_btree_addr,
                heap_addr: ste_heap_addr,
            },
            datasets: catalog.datasets,
            unreadable: catalog.unreadable,
            root_attributes,
            group_attributes: catalog.group_attributes,
            group_paths: catalog.group_paths,
            group_aliases: catalog.group_aliases,
            links: catalog.links,
        })
    }

    /// Extract the symbol-table message (btree_addr, heap_addr) from an
    /// already-decoded object header.
    fn stab_from_header(header: &ObjectHeader, ctx: &FormatContext) -> (u64, u64) {
        for msg in &header.messages {
            if msg.msg_type == MSG_SYMBOL_TABLE {
                let sa = ctx.sizeof_addr as usize;
                if msg.data.len() >= 2 * sa {
                    return (read_uint(&msg.data, sa), read_uint(&msg.data[sa..], sa));
                }
            }
        }
        (UNDEF_ADDR, UNDEF_ADDR)
    }

    /// Build the file catalog by walking link messages in a v2 object header.
    /// Recursively descends into groups, prefixing names with the group path.
    fn build_catalog_from_links(
        handle: &mut FileHandle,
        root_header: &ObjectHeader,
        root_addr: u64,
        ctx: &FormatContext,
    ) -> IoResult<Catalog> {
        let mut catalog = Catalog::default();
        // Object headers already descended into, keyed to the first path
        // that reached them: a later path to the same header is a group
        // hard link, recorded in `aliases` so lookups can resolve through
        // it instead of walking (and cycling) a second time.
        let mut visited = std::collections::HashMap::new();
        // Seed the root object header so a hard link cycling back to the
        // root is not descended into a second time.
        visited.insert(root_addr, String::new());
        Self::walk_links_recursive(handle, root_header, ctx, "", &mut catalog, &mut visited)?;
        Ok(catalog)
    }

    fn walk_links_recursive(
        handle: &mut FileHandle,
        header: &ObjectHeader,
        ctx: &FormatContext,
        prefix: &str,
        catalog: &mut Catalog,
        visited: &mut std::collections::HashMap<u64, String>,
    ) -> IoResult<()> {
        // Bound recursion depth on a hostile/corrupt file.
        const MAX_GROUP_DEPTH: usize = 256;
        let depth = if prefix.is_empty() {
            0
        } else {
            prefix.matches('/').count() + 1
        };
        if depth > MAX_GROUP_DEPTH {
            return Ok(());
        }

        // Collect every link in this group: inline `Link` messages plus, for
        // groups using dense storage, links held in a fractal heap referenced
        // by the `Link Info` message.
        //
        // A link message that does not decode has no name to report the
        // failure against, and a `Link Info` message that does not decode
        // hides a whole group's dense storage. Either way the listing would
        // come back silently short, so both are errors: a listing this
        // reader cannot complete must not present itself as complete.
        let mut links: Vec<LinkMessage> = Vec::new();
        for msg in &header.messages {
            if msg.msg_type == MSG_LINK {
                let (link, _) = LinkMessage::decode(&msg.data, ctx)?;
                links.push(link);
            } else if msg.msg_type == MSG_LINK_INFO {
                let (info, _) = LinkInfoMessage::decode(&msg.data, ctx)?;
                if info.fractal_heap_address != UNDEF_ADDR {
                    let dense = Self::read_dense_links(handle, ctx, info.fractal_heap_address)?;
                    links.extend(dense);
                }
            }
        }

        for link in &links {
            let full_name = if prefix.is_empty() {
                link.name.clone()
            } else {
                format!("{}/{}", prefix, link.name)
            };

            // Every link is a listing entry whatever it points at; only a
            // hard link names an object in this file to descend into.
            catalog
                .links
                .insert(full_name.clone(), LinkClass::from_target(&link.target));
            let LinkTarget::Hard { address } = &link.target else {
                continue;
            };

            // The link names an object, so the object is in the listing
            // whatever comes of reading it: a header that does not decode
            // (a stale link left by a deletion, say) is reported against
            // this name, never dropped from it.
            let child_header = match Self::read_object_header_full(handle, ctx, *address) {
                Ok(h) => h,
                Err(e) => {
                    catalog
                        .unreadable
                        .insert(full_name, format!("its object header does not decode: {e}"));
                    continue;
                }
            };
            match Self::classify_object(&child_header, ctx, &full_name) {
                ObjectKind::Dataset(info) => {
                    catalog.datasets.push(*info);
                    continue;
                }
                ObjectKind::UnreadableDataset(why) => {
                    catalog.unreadable.insert(full_name, why);
                    continue;
                }
                // A committed (named) datatype is neither a group nor a
                // dataset, so it must not be recorded as either; the link
                // record above already carries its name.
                ObjectKind::CommittedDatatype => continue,
                ObjectKind::Group => {}
            }
            // It is a group. Record its path from the actual link record —
            // before the cycle check, so a hard-link alias of an
            // already-visited group still appears — whether or not it
            // contains datasets or attributes.
            catalog.group_paths.insert(full_name.clone());
            // Capture group attributes (e.g. the NeXus `NX_class` marker),
            // keyed by path.
            let mut attrs = Vec::new();
            for m in &child_header.messages {
                if m.msg_type == MSG_ATTRIBUTE {
                    if let Ok((a, _)) = AttributeMessage::decode(&m.data, ctx) {
                        attrs.push(a);
                    }
                }
            }
            if !attrs.is_empty() {
                catalog.group_attributes.insert(full_name.clone(), attrs);
            }
            // Descend at most once per object header (cycle guard); a second
            // path to it is a group hard link — record the alias for lookups
            // instead.
            if let Some(first) = visited.get(address) {
                catalog.group_aliases.insert(full_name, first.clone());
                continue;
            }
            visited.insert(*address, full_name.clone());
            Self::walk_links_recursive(handle, &child_header, ctx, &full_name, catalog, visited)?;
        }
        Ok(())
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
        let fh_header = FractalHeapHeader::decode(&hdr_buf, ctx)?;

        // Walk the heap's managed blocks; each block hands back a payload
        // region holding one or more packed encoded `Link` messages.
        let mut br = HandleBlockReader { handle };
        let payloads = fractal_heap::collect_managed_objects(&fh_header, ctx, &mut br)?;

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

        // That scan stops at the first byte that does not begin a link, which
        // is how trailing free space in a direct block ends it — and would
        // equally swallow a link the scan could not read. The heap header
        // counts its managed objects, so a short scan is detectable, and a
        // group listing that is short is the loss this guards against.
        if links.len() < fh_header.man_nobjs as usize {
            return Err(crate::io::IoError::InvalidState(format!(
                "dense link storage at address {fractal_heap_addr:#x} holds {} managed \
                 objects but only {} decoded as links",
                fh_header.man_nobjs,
                links.len()
            )));
        }

        Ok(links)
    }

    /// Build the file catalog by walking the B-tree v1 + local heap (legacy
    /// format).
    ///
    /// Recurses into subgroups: a symbol-table entry that caches its child
    /// group's B-tree and local heap uses those; for other entries the child
    /// object header is read for a symbol-table message. Discovered names are
    /// prefixed with the group path.
    fn build_catalog_from_btree(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        btree_addr: u64,
        heap_addr: u64,
        root_obj_addr: u64,
    ) -> IoResult<Catalog> {
        let mut catalog = Catalog::default();
        // First path per descended object header + the group-hard-link
        // aliases met later, as in `build_catalog_from_links`.
        let mut visited = std::collections::HashMap::new();
        // Seed the root object header so a hard link cycling back to the
        // root group is not descended into a second time.
        visited.insert(root_obj_addr, String::new());
        Self::walk_btree_recursive(
            handle,
            ctx,
            btree_addr,
            heap_addr,
            "",
            0,
            &mut catalog,
            &mut visited,
        )?;
        Ok(catalog)
    }

    /// Recursive worker for `build_catalog_from_btree`. `prefix` is the
    /// path of the group being scanned; `depth` bounds recursion.
    #[allow(clippy::too_many_arguments)]
    fn walk_btree_recursive(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        btree_addr: u64,
        heap_addr: u64,
        prefix: &str,
        depth: usize,
        catalog: &mut Catalog,
        visited: &mut std::collections::HashMap<u64, String>,
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
        let mut snod_tree_visited = std::collections::HashSet::new();
        let snod_addrs =
            Self::collect_snod_addresses(handle, btree_addr, sa, ss, 0, &mut snod_tree_visited)?;

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

                // A `H5G_CACHED_SLINK` entry is a soft link: it names no
                // object at all, and its value string lives in this group's
                // local heap. Record the link and move on — reading its
                // undefined object-header address is what used to drop it.
                if let SymbolTableCache::SoftLink { value_offset } = entry.cache {
                    let target = local_heap_get_string(&heap_data, value_offset as u64)?;
                    catalog
                        .links
                        .insert(full_name, LinkClass::Soft { path: target });
                    continue;
                }
                catalog.links.insert(full_name.clone(), LinkClass::Hard);

                // The entry names an object, so the object is in the
                // listing whatever comes of reading it — the same rule the
                // link-message walk applies.
                let hdr = match Self::read_object_header_full(handle, ctx, entry.obj_header_addr) {
                    Ok(h) => h,
                    Err(e) => {
                        catalog
                            .unreadable
                            .insert(full_name, format!("its object header does not decode: {e}"));
                        continue;
                    }
                };
                match Self::classify_object(&hdr, ctx, &full_name) {
                    ObjectKind::Dataset(info) => {
                        catalog.datasets.push(*info);
                        continue;
                    }
                    ObjectKind::UnreadableDataset(why) => {
                        catalog.unreadable.insert(full_name, why);
                        continue;
                    }
                    ObjectKind::CommittedDatatype => continue,
                    ObjectKind::Group => {}
                }

                // It is a subgroup. Record its path from the actual
                // symbol-table entry, whether or not it has datasets or
                // attributes.
                catalog.group_paths.insert(full_name.clone());

                // Break cycles: descend into each group object header at
                // most once. A second path to it is a group hard link —
                // record the alias for lookups instead.
                if let Some(first) = visited.get(&entry.obj_header_addr) {
                    catalog.group_aliases.insert(full_name, first.clone());
                    continue;
                }
                visited.insert(entry.obj_header_addr, full_name.clone());

                // Collect this subgroup's attributes (e.g. NeXus NX_class).
                {
                    let mut attrs = Vec::new();
                    for m in &hdr.messages {
                        if m.msg_type == MSG_ATTRIBUTE {
                            if let Ok((a, _)) = AttributeMessage::decode(&m.data, ctx) {
                                attrs.push(a);
                            }
                        }
                    }
                    if !attrs.is_empty() {
                        catalog.group_attributes.insert(full_name.clone(), attrs);
                    }
                }

                // Find its B-tree + local heap and recurse, prefixing names
                // with the group path.
                let (sub_btree, sub_heap) = match entry.cached_symbol_table() {
                    // Scratch-pad caches the subgroup's symbol-table info.
                    Some(pair) => pair,
                    // No scratch pad — take the symbol-table message from
                    // the child object header already read above.
                    None => Self::stab_from_header(&hdr, ctx),
                };

                if sub_btree != UNDEF_ADDR && sub_heap != UNDEF_ADDR {
                    Self::walk_btree_recursive(
                        handle,
                        ctx,
                        sub_btree,
                        sub_heap,
                        &full_name,
                        depth + 1,
                        catalog,
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
        visited: &mut std::collections::HashSet<u64>,
    ) -> IoResult<Vec<u64>> {
        // A well-formed v1 B-tree's level strictly decreases with depth;
        // bound the descent so a corrupt/cyclic tree cannot recurse forever.
        // The `visited` set additionally stops a corrupt tree whose child
        // points back at an ancestor node from fanning out exponentially.
        if depth > 256 || !visited.insert(tree_addr) {
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
                    visited,
                )?;
                addrs.extend(child_addrs);
            }
            Ok(addrs)
        }
    }

    /// Read the object header at `addr` with every continuation block
    /// flattened in. One owner for both halves of the crate — see
    /// [`crate::io::object_header_io`].
    fn read_object_header_full(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        addr: u64,
    ) -> IoResult<ObjectHeader> {
        crate::io::object_header_io::read_object_header_full(handle, ctx, addr)
    }

    /// Classify one object from its (already read) header, and decode the
    /// dataset metadata while doing so.
    ///
    /// The class comes from which messages are present, never from whether
    /// they decode: an object holding a datatype, a dataspace and a data
    /// layout is a dataset even when this crate cannot decode one of them,
    /// and it says so as [`ObjectKind::UnreadableDataset`] rather than
    /// vanishing.
    ///
    /// Only the messages the payload depends on can make a dataset
    /// unreadable. A failed *attribute* decode leaves the dataset itself
    /// readable, so it does not.
    fn classify_object(header: &ObjectHeader, ctx: &FormatContext, name: &str) -> ObjectKind {
        let present = |t: u8| header.messages.iter().any(|m| m.msg_type == t);
        let is_group = present(MSG_LINK)
            || present(MSG_LINK_INFO)
            || present(MSG_SYMBOL_TABLE)
            || present(MSG_GROUP_INFO);
        if is_group {
            return ObjectKind::Group;
        }
        let is_dataset =
            present(MSG_DATATYPE) && present(MSG_DATASPACE) && present(MSG_DATA_LAYOUT);
        if !is_dataset {
            return if present(MSG_DATATYPE) {
                ObjectKind::CommittedDatatype
            } else {
                ObjectKind::Group
            };
        }

        let mut datatype = None;
        let mut dataspace = None;
        let mut layout = None;
        let mut filter_pipeline = None;
        let mut fill_value = None;
        let mut attributes = Vec::new();
        // The first message that did not decode, kept verbatim: it is the
        // answer a caller gets when it asks for this dataset.
        let mut blocked: Option<String> = None;
        let mut block = |what: &str, e: crate::format::FormatError| {
            if blocked.is_none() {
                blocked = Some(format!("its {what} message does not decode: {e}"));
            }
        };

        for msg in &header.messages {
            match msg.msg_type {
                MSG_DATATYPE => match DatatypeMessage::decode(&msg.data, ctx) {
                    Ok((dt, _)) => datatype = Some(dt),
                    Err(e) => block("datatype", e),
                },
                MSG_DATASPACE => match DataspaceMessage::decode(&msg.data, ctx) {
                    Ok((ds, _)) => dataspace = Some(ds),
                    Err(e) => block("dataspace", e),
                },
                MSG_DATA_LAYOUT => match DataLayoutMessage::decode(&msg.data, ctx) {
                    Ok((dl, _)) => layout = Some(dl),
                    Err(e) => block("data layout", e),
                },
                // A filter pipeline that does not decode would leave the raw
                // chunk bytes to be handed back as if they were never
                // filtered, and an undecodable fill value would leave
                // unwritten regions reading as zeros. Both change the data a
                // read returns, so both block the dataset.
                MSG_FILTER_PIPELINE => match FilterPipeline::decode(&msg.data) {
                    Ok((fp, _)) => {
                        if !fp.filters.is_empty() {
                            filter_pipeline = Some(fp);
                        }
                    }
                    Err(e) => block("filter pipeline", e),
                },
                MSG_FILL_VALUE => match FillValueMessage::decode(&msg.data) {
                    Ok((fv, _)) => {
                        if fv.fill_defined == 2 {
                            fill_value = fv.fill_value;
                        }
                    }
                    Err(e) => block("fill value", e),
                },
                MSG_ATTRIBUTE => {
                    if let Ok((attr, _)) = AttributeMessage::decode(&msg.data, ctx) {
                        attributes.push(attr);
                    }
                }
                _ => {}
            }
        }

        if let Some(why) = blocked {
            return ObjectKind::UnreadableDataset(why);
        }
        match (datatype, dataspace, layout) {
            (Some(dt), Some(ds), Some(dl)) => ObjectKind::Dataset(Box::new(DatasetReadInfo {
                name: name.to_string(),
                datatype: dt,
                dataspace: ds,
                layout: dl,
                filter_pipeline,
                attributes,
                fill_value,
            })),
            // The three messages are present and none of them reported an
            // error, so this is unreachable; report it as unreadable rather
            // than dropping the name on an invariant this function owns.
            _ => ObjectKind::UnreadableDataset(
                "its datatype, dataspace and data layout messages decoded but did not all \
                 produce a value"
                    .into(),
            ),
        }
    }

    /// Every dataset in the file, by path (no leading `/`).
    ///
    /// A dataset this crate cannot read is still a dataset the file
    /// contains, so it is listed here alongside the readable ones and
    /// answers [`Self::unreadable_reason`]; opening it reports that reason.
    pub fn dataset_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.datasets.iter().map(|d| d.name.as_str()).collect();
        names.extend(self.unreadable.keys().map(String::as_str));
        names
    }

    /// Why the dataset at `path` (no leading `/`) cannot be read, or `None`
    /// when it can be — or does not exist.
    pub fn unreadable_reason(&self, path: &str) -> Option<&str> {
        self.unreadable
            .get(&self.canonical_path(path))
            .map(String::as_str)
    }

    /// Every link record in the file, keyed by full path (no leading `/`).
    pub fn links(&self) -> &std::collections::BTreeMap<String, LinkClass> {
        &self.links
    }

    /// The class of the link at `path` (no leading `/`), or `None` when no
    /// link of that name exists. The path is traversed first, so a link
    /// reached through a group hard link or a soft link resolves.
    pub fn link_class(&self, path: &str) -> Option<&LinkClass> {
        if let Some(class) = self.links.get(path) {
            return Some(class);
        }
        match self.traverse(path) {
            Traversal::Path { path, .. } => self.links.get(&path),
            Traversal::External { .. } => None,
        }
    }

    /// Follow a path (no leading `/`) the way `H5Dopen` / `H5Gopen` do:
    /// rewrite each component that is a group hard-link alias or a soft link
    /// until nothing changes, bounded so a link cycle cannot loop forever.
    ///
    /// This is the single owner of link traversal — every lookup that takes a
    /// caller-supplied path goes through it rather than comparing the path to
    /// a catalog key directly.
    fn traverse(&self, name: &str) -> Traversal {
        // libhdf5 bounds soft-link traversal at `H5L_NLINKS_DEF`; this covers
        // that and the hard-link alias rewrites interleaved with it.
        const MAX_TRAVERSALS: usize = 64;
        let mut name = name.trim_start_matches('/').to_string();
        let mut via = None;
        for _ in 0..MAX_TRAVERSALS {
            // Take the longest matching prefix so a nested alias wins over a
            // shorter one that also covers the path.
            let covers = |prefix: &str| name == prefix || name.starts_with(&format!("{prefix}/"));
            let mut best: Option<(&str, Rewrite<'_>)> = None;
            for (alias, first) in &self.group_aliases {
                if covers(alias) && best.is_none_or(|(p, _)| alias.len() > p.len()) {
                    best = Some((alias, Rewrite::Alias(first)));
                }
            }
            for (link, class) in &self.links {
                let rewrite = match class {
                    LinkClass::Soft { path } => Rewrite::Soft(path),
                    LinkClass::External { file, path } => Rewrite::External { file, path },
                    _ => continue,
                };
                if covers(link) && best.is_none_or(|(p, _)| link.len() > p.len()) {
                    best = Some((link, rewrite));
                }
            }
            let Some((prefix, rewrite)) = best else { break };
            let rest = name[prefix.len()..].to_string();
            match rewrite {
                Rewrite::Alias(first) => {
                    // `first` is empty for an alias of the root group;
                    // trimming keeps the no-leading-'/' form either way.
                    name = format!("{first}{rest}").trim_start_matches('/').to_string();
                }
                Rewrite::Soft(target) => {
                    let resolved = resolve_link_value(prefix, target);
                    via = Some(SoftLinkRef {
                        link: prefix.to_string(),
                        target: target.to_string(),
                    });
                    name = format!("{resolved}{rest}")
                        .trim_start_matches('/')
                        .to_string();
                }
                Rewrite::External { file, path } => {
                    return Traversal::External {
                        link: prefix.to_string(),
                        file: file.to_string(),
                        path: format!("{path}{rest}"),
                    };
                }
            }
        }
        Traversal::Path { path: name, via }
    }

    /// Rewrite a path (no leading `/`) into the path of the object it reaches
    /// after link traversal. A path that leaves the file through an external
    /// link comes back unchanged — the callers that must report that case use
    /// [`Self::traverse`] directly.
    pub fn canonical_path(&self, name: &str) -> String {
        match self.traverse(name) {
            Traversal::Path { path, .. } => path,
            Traversal::External { .. } => name.trim_start_matches('/').to_string(),
        }
    }

    /// Return metadata for a dataset by name. Like `H5Dopen`, the name
    /// may pass through group hard links and soft links.
    pub fn dataset_info(&self, name: &str) -> Option<&DatasetReadInfo> {
        let name = self.canonical_path(name);
        self.datasets.iter().find(|d| d.name == name)
    }

    /// Open `name` as a dataset the way `H5Dopen` does, reporting *why* it
    /// cannot be opened instead of collapsing every cause into absence: a
    /// soft link whose target does not exist is a dangling link, and a path
    /// through an external link leaves this file.
    ///
    /// This is the gate every typed dataset access goes through.
    pub fn open_dataset(&self, name: &str) -> IoResult<&DatasetReadInfo> {
        let (path, via) = match self.traverse(name) {
            Traversal::Path { path, via } => (path, via),
            Traversal::External { link, file, path } => {
                return Err(crate::io::IoError::Unsupported(format!(
                    "'{name}' resolves through the external link '{link}' to '{path}' in \
                     '{file}'; this reader lists external links but does not follow them"
                )));
            }
        };
        if let Some(info) = self.datasets.iter().find(|d| d.name == path) {
            return Ok(info);
        }
        if let Some(why) = self.unreadable.get(&path) {
            return Err(crate::io::IoError::Unsupported(format!(
                "'{name}' is a dataset this crate cannot read: {why}"
            )));
        }
        if let Some(SoftLinkRef { link, target }) = via {
            return Err(crate::io::IoError::DanglingLink { link, target });
        }
        Err(crate::io::IoError::NotFound(name.to_string()))
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
    /// leading `/`, e.g. `"detector"` or `"entry/instrument"`; may pass
    /// through group hard links).
    pub fn group_attr_names(&self, group_path: &str) -> Vec<String> {
        self.group_attributes
            .get(&self.canonical_path(group_path))
            .map(|v| v.iter().map(|a| a.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Return a non-root group's attribute by name.
    pub fn group_attr(&self, group_path: &str, name: &str) -> Option<&AttributeMessage> {
        self.group_attributes
            .get(&self.canonical_path(group_path))?
            .iter()
            .find(|a| a.name == name)
    }

    /// Return every non-root group path the discovery walk traversed into
    /// (no leading `/`). Built from actual link records, so empty groups,
    /// attribute-only groups, and subgroup-only groups are all included.
    pub fn group_paths(&self) -> &std::collections::BTreeSet<String> {
        &self.group_paths
    }

    /// Report whether a group exists at `group_path` (no leading `/`;
    /// may pass through group hard links). The empty string denotes the
    /// root group, which always exists.
    pub fn has_group(&self, group_path: &str) -> bool {
        if group_path.is_empty() || self.group_paths.contains(group_path) {
            return true;
        }
        let canon = self.canonical_path(group_path);
        canon.is_empty() || self.group_paths.contains(&canon)
    }

    /// Read and decode the global-heap collection at `addr`, applying the
    /// validation of libhdf5's `H5HG__cache_heap_deserialize`: the `GCOL`
    /// signature must be present and the declared size at least
    /// `H5HG_MINSIZE` (4096 bytes). There is no upper size cap — libhdf5
    /// has none, and this crate's writers put a whole write call's strings
    /// into one collection, which a cap would turn into silent data loss.
    fn read_heap_collection(&mut self, addr: u64) -> IoResult<GlobalHeapCollection> {
        let ss = self.ctx.sizeof_size as usize;
        let header_len = 4 + 1 + 3 + ss;
        let header_buf = self.handle.read_at_most(addr, header_len)?;
        if header_buf.len() < header_len || header_buf[0..4] != *b"GCOL" {
            return Err(crate::io::IoError::InvalidState(format!(
                "bad global heap collection signature at address {addr:#x}"
            )));
        }
        let declared = read_uint(&header_buf[8..], ss) as usize;
        if declared < 4096 {
            return Err(crate::io::IoError::InvalidState(format!(
                "global heap collection at address {addr:#x} declares size {declared}, \
                 below the 4096-byte minimum"
            )));
        }
        let heap_buf = self.handle.read_at(addr, declared)?;
        let (coll, _) = GlobalHeapCollection::decode(&heap_buf, &self.ctx)?;
        Ok(coll)
    }

    /// Decode an attribute's value as a string, resolving a variable-length
    /// string attribute through the global heap (h5py writes string
    /// attributes as variable-length by default).
    pub fn attr_string_value(&mut self, attr: &AttributeMessage) -> IoResult<String> {
        use crate::format::messages::datatype::DatatypeMessage;
        if !matches!(attr.datatype, DatatypeMessage::VarLenString { .. }) {
            // Fixed-length string: raw bytes, truncated at the first NUL.
            let end = attr
                .data
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(attr.data.len());
            return Ok(String::from_utf8_lossy(&attr.data[..end]).to_string());
        }
        // Variable-length string: the attribute value is a global-heap
        // reference (sequence length + collection address + object index).
        if attr.data.len() < vlen_reference_size(&self.ctx) {
            return Ok(String::new());
        }
        let (_seq, coll_addr, obj_index) = decode_vlen_reference(&attr.data, &self.ctx)?;
        if coll_addr == UNDEF_ADDR || coll_addr == 0 {
            return Ok(String::new());
        }
        let coll = self.read_heap_collection(coll_addr)?;
        let idx = u16::try_from(obj_index).map_err(|_| {
            crate::io::IoError::InvalidState(format!(
                "global heap object index {obj_index} does not fit the 16-bit on-disk field"
            ))
        })?;
        let obj = coll.get_object(idx).ok_or_else(|| {
            crate::io::IoError::InvalidState(format!(
                "global heap object {idx} not found in the collection at address {coll_addr:#x}"
            ))
        })?;
        Ok(String::from_utf8_lossy(obj).to_string())
    }

    /// Return the dimensions of a dataset.
    pub fn dataset_shape(&self, name: &str) -> IoResult<Vec<u64>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        Ok(info.dataspace.dims.clone())
    }

    /// Logical byte size of a dataset's full image (`product(dims) *
    /// element_size`), with the datatype needed for the post-filter conversion.
    fn raw_size_and_datatype(&self, name: &str) -> IoResult<(DatatypeMessage, u64)> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let total = saturating_byte_len(&info.dataspace.dims, info.datatype.element_size() as u64);
        Ok((info.datatype.clone(), total))
    }

    /// Read the raw bytes of a dataset.
    pub fn read_dataset_raw(&mut self, name: &str) -> IoResult<Vec<u8>> {
        let (datatype, total) = self.raw_size_and_datatype(name)?;
        let mut data = alloc_tiled_fill(total as usize, None)?;
        self.read_dataset_raw_into_unconverted(name, &mut data)?;
        Self::apply_post_filter_conversion(&mut data, &datatype)?;
        Ok(data)
    }

    /// Read the full raw dataset image into a caller-provided buffer.
    ///
    /// `out.len()` must equal the dataset's logical byte size
    /// (`product(dims) * element_size`); otherwise an error is returned. This
    /// is the no-allocation counterpart of [`read_dataset_raw`](Self::read_dataset_raw):
    /// the bytes are read straight into `out`, making it the zero-copy entry
    /// point for reading directly into a pinned/registered host buffer for an
    /// H2D transfer.
    pub fn read_dataset_raw_into(&mut self, name: &str, out: &mut [u8]) -> IoResult<()> {
        let (datatype, total) = self.raw_size_and_datatype(name)?;
        if out.len() as u64 != total {
            return Err(crate::io::IoError::InvalidState(format!(
                "read_dataset_raw_into: buffer is {} bytes but dataset needs {}",
                out.len(),
                total
            )));
        }
        self.read_dataset_raw_into_unconverted(name, out)?;
        Self::apply_post_filter_conversion(out, &datatype)?;
        Ok(())
    }

    /// Fill `out` with the full raw dataset image, before the post-filter
    /// datatype conversion. The single owner of read-destination semantics for
    /// full reads: it fully defines every byte of `out` (reading allocated data
    /// straight in, pre-filling chunked or never-written regions with the tiled
    /// fill value), so callers supply only a correctly-sized buffer. Both the
    /// allocating `read_dataset_raw` and the zero-copy `read_dataset_raw_into`
    /// wrap it and apply the conversion exactly once.
    ///
    /// `out.len()` must equal `product(dims) * element_size`.
    fn read_dataset_raw_into_unconverted(&mut self, name: &str, out: &mut [u8]) -> IoResult<()> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;

        // Clone to avoid borrow conflict with &mut self in read methods.
        let layout = info.layout.clone();
        let pipeline = info.filter_pipeline.clone();
        let fill_value = info.fill_value.clone();

        match &layout {
            DataLayoutMessage::Contiguous { address, .. } => {
                if *address == UNDEF_ADDR {
                    // Never-written contiguous data reads back as the fill value.
                    fill_tiled_into(out, fill_value.as_deref());
                } else {
                    // Read exactly the logical image straight into `out`.
                    self.handle.read_exact_at_into(*address, out)?;
                }
            }
            DataLayoutMessage::Compact { data } => {
                let n = out.len().min(data.len());
                out[..n].copy_from_slice(&data[..n]);
                if n < out.len() {
                    fill_tiled_into(&mut out[n..], fill_value.as_deref());
                }
            }
            DataLayoutMessage::ChunkedV3 {
                chunk_dims,
                b_tree_address,
            } => {
                // Pre-fill so any chunk gap reads back as fill, then scatter.
                fill_tiled_into(out, fill_value.as_deref());
                // The layout's chunk_dims include the element size as the
                // trailing dimension. Strip it for chunk indexing.
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_btree_v1(
                    name,
                    real_chunk_dims,
                    *b_tree_address,
                    pipeline.as_ref(),
                    ChunkTarget::Full,
                    out,
                )?;
            }
            DataLayoutMessage::ChunkedV4 {
                chunk_dims,
                index_address,
                index_type,
                earray_params,
                single_chunk_filter,
                ..
            } => {
                fill_tiled_into(out, fill_value.as_deref());
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_v4(
                    name,
                    real_chunk_dims,
                    ChunkIndexDesc {
                        index_type: *index_type,
                        index_address: *index_address,
                        earray_params: earray_params.as_ref(),
                        single_chunk_filter: *single_chunk_filter,
                    },
                    pipeline.as_ref(),
                    ChunkTarget::Full,
                    out,
                )?;
            }
        }
        Ok(())
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

        // Re-scan datasets, group attributes, group paths, and link records
        // from link messages.
        let catalog = Self::build_catalog_from_links(
            &mut self.handle,
            &root_header,
            sb.root_group_object_header_address,
            &ctx,
        )?;

        self._eof = sb.end_of_file_address;
        self.ctx = ctx;
        self.datasets = catalog.datasets;
        self.unreadable = catalog.unreadable;
        self.group_attributes = catalog.group_attributes;
        self.group_paths = catalog.group_paths;
        self.group_aliases = catalog.group_aliases;
        self.links = catalog.links;

        Ok(())
    }

    /// Read chunked dataset data by walking the chunk index.
    ///
    /// `desc` bundles the version-4 chunk-index descriptor extracted from the
    /// data-layout message (kind, address, and per-kind parameters), so this
    /// entry point takes one descriptor rather than a long parameter list.
    ///
    /// Scatters only; `output` must already be sized to the target extent and
    /// pre-filled with the tiled fill value by the caller (so unallocated or
    /// non-overlapping regions read back as fill).
    fn read_chunked_v4(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        desc: ChunkIndexDesc<'_>,
        pipeline: Option<&FilterPipeline>,
        target: ChunkTarget,
        output: &mut [u8],
    ) -> IoResult<()> {
        let ChunkIndexDesc {
            index_type,
            index_address,
            earray_params,
            single_chunk_filter,
        } = desc;
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;

        match index_type {
            data_layout::ChunkIndexType::SingleChunk => {
                // Single chunk: the index_address IS the chunk address
                let total_size: u64 = saturating_byte_len(&dims, element_size);
                if index_address == UNDEF_ADDR || total_size == 0 {
                    // Unallocated single chunk: `output` is already the
                    // pre-filled fill/zero buffer, so there is nothing to read
                    // or scatter.
                    return Ok(());
                }
                let data = if let Some(pipeline) = pipeline {
                    // A filtered single chunk records its exact on-disk size
                    // and per-chunk filter mask in the layout message. Use
                    // them to read precisely the stored bytes and to skip the
                    // filters the mask marks as not applied. Without those
                    // params (older/edge layouts) fall back to the
                    // read-extra-and-inflate heuristic with the full pipeline.
                    let (raw, mask) = match single_chunk_filter {
                        Some(scf) => (
                            self.handle.read_at(index_address, scf.nbytes as usize)?,
                            scf.filter_mask,
                        ),
                        None => (
                            self.handle.read_at_most(
                                index_address,
                                total_size.saturating_mul(2) as usize,
                            )?,
                            0,
                        ),
                    };
                    filter::reverse_filters_masked(pipeline, &raw, mask)?
                } else {
                    self.handle.read_at(index_address, total_size as usize)?
                };
                // The lone chunk spans the whole dataset; place it respecting
                // the dataset extent (Full) or the selection (Slice) into the
                // caller's pre-filled buffer.
                let coords = vec![0u64; dims.len()];
                match target {
                    ChunkTarget::Full => self.copy_chunk_to_output(
                        &data,
                        output,
                        &dims,
                        chunk_dims,
                        &coords,
                        element_size,
                    ),
                    ChunkTarget::Slice { starts, counts } => self.copy_chunk_to_slice(
                        &data,
                        output,
                        &dims,
                        chunk_dims,
                        &coords,
                        element_size,
                        starts,
                        counts,
                    ),
                }
                Ok(())
            }
            data_layout::ChunkIndexType::FixedArray => self.read_chunked_fixed_array(
                name,
                chunk_dims,
                index_address,
                pipeline,
                target,
                output,
            ),
            data_layout::ChunkIndexType::BTreeV2 => self.read_chunked_btree_v2(
                name,
                chunk_dims,
                index_address,
                pipeline,
                target,
                output,
            ),
            data_layout::ChunkIndexType::ExtensibleArray => {
                let params = earray_params.ok_or_else(|| {
                    crate::io::IoError::InvalidState("missing earray params".into())
                })?;

                if index_address == UNDEF_ADDR {
                    // Unallocated: `output` is already the pre-filled buffer.
                    return Ok(());
                }

                // Total slot count of the index grid. The maximum extent
                // decides the multipliers (libhdf5 max_down_chunks); an
                // unlimited dimension 0 is bounded by the current extent for
                // this read — a slot beyond it (written before a shrink) is
                // not visible.
                let max_dims = info.dataspace.max_dims.clone();
                let grid =
                    crate::io::chunk_grid::index_grid(&dims, max_dims.as_deref(), chunk_dims)?;
                let chunks_total: u64 = grid.iter().fold(1u64, |acc, &n| acc.saturating_mul(n));

                let chunk_entries = self.collect_ea_chunk_entries(
                    index_address,
                    params,
                    &dims,
                    max_dims.as_deref(),
                    chunk_dims,
                    element_size,
                )?;

                let n_chunks = std::cmp::min(chunks_total as usize, chunk_entries.len());

                // Chunks are placed N-dimensionally: each slot decodes
                // (row-major, against the index grid) to chunk-grid
                // coordinates, so sub-frame chunks (a chunk smaller than a
                // full frame) land correctly.
                let mut slot_coords = Vec::with_capacity(n_chunks);
                for i in 0..n_chunks as u64 {
                    slot_coords.push(crate::io::chunk_grid::coords_of(
                        &dims,
                        max_dims.as_deref(),
                        chunk_dims,
                        i,
                    )?);
                }
                let chunk_coords = |i: u64| -> &[u64] { &slot_coords[i as usize] };

                // Build one read job per chunk (no I/O yet), then read +
                // decompress them together (in parallel where positioned reads
                // are race-free), then scatter serially. Filtered chunks record
                // their exact on-disk size, so read exactly that; unfiltered
                // chunks read at-most since the entry size can exceed the file
                // tail. Skip conditions differ between the two, so build jobs
                // per branch.
                let jobs: Vec<Option<ChunkReadJob>> = if pipeline.is_some() {
                    let file_size = self.handle.file_size()?;
                    chunk_entries[..n_chunks]
                        .iter()
                        .enumerate()
                        .map(|(i, &(addr, nbytes, mask))| {
                            if addr == UNDEF_ADDR
                                || nbytes == 0
                                || addr >= file_size
                                || nbytes > file_size
                                || !target.overlaps(chunk_coords(i as u64), chunk_dims)
                            {
                                None
                            } else {
                                Some(ChunkReadJob {
                                    addr,
                                    len: nbytes as usize,
                                    at_most: false,
                                    mask,
                                })
                            }
                        })
                        .collect()
                } else {
                    chunk_entries[..n_chunks]
                        .iter()
                        .enumerate()
                        .map(|(i, &(addr, nbytes, _))| {
                            if addr == UNDEF_ADDR
                                || !target.overlaps(chunk_coords(i as u64), chunk_dims)
                            {
                                None
                            } else {
                                Some(ChunkReadJob {
                                    addr,
                                    len: nbytes as usize,
                                    at_most: true,
                                    mask: 0,
                                })
                            }
                        })
                        .collect()
                };

                let decompressed = read_and_decompress_chunks(&self.handle, pipeline, jobs)?;

                for (i, chunk_data) in decompressed.iter().enumerate() {
                    if let Some(data) = chunk_data {
                        let coords = chunk_coords(i as u64);
                        self.scatter_chunk(
                            target,
                            data,
                            output,
                            &dims,
                            chunk_dims,
                            coords,
                            element_size,
                        );
                    }
                }

                Ok(())
            }
            _ => Err(crate::io::IoError::InvalidState(format!(
                "unsupported chunk index type: {:?}",
                index_type
            ))),
        }
    }

    /// Read a dataset indexed by a fixed array.
    ///
    /// Scatters only; `output` must already be sized to the target extent and
    /// pre-filled with the tiled fill value by the caller.
    fn read_chunked_fixed_array(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        pipeline: Option<&FilterPipeline>,
        target: ChunkTarget,
        output: &mut [u8],
    ) -> IoResult<()> {
        use crate::format::chunk_index::fixed_array::*;

        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let ndims = dims.len();

        if index_address == UNDEF_ADDR {
            // Unallocated: `output` is already the pre-filled buffer.
            return Ok(());
        }

        // Read FA header
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let fa_hdr = FixedArrayHeader::decode(&hdr_buf, &self.ctx)?;

        if fa_hdr.data_blk_addr == UNDEF_ADDR {
            // Unallocated data block: `output` is already pre-filled.
            return Ok(());
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
        let chunk_bytes: u64 = saturating_byte_len(chunk_dims, element_size);

        // Collect per-chunk (address, compressed_size). compressed_size is the
        // exact on-disk byte count for filtered chunks, or chunk_bytes when
        // unfiltered.
        let num_elmts = fa_hdr.num_elmts as usize;
        // (chunk address, on-disk byte count, filter mask). The mask is the
        // per-chunk filter mask for filtered chunks, 0 when unfiltered.
        let mut chunk_entries: Vec<(u64, u64, u32)> = Vec::with_capacity(num_elmts);

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
                    chunk_entries
                        .extend(std::iter::repeat_n((UNDEF_ADDR, 0u64, 0u32), page_nelmts));
                    continue;
                }

                let page_addr = pages_base + (p as u64) * page_stride as u64;
                let page_size = page_nelmts * elem_size + 4;
                let page_buf = self.handle.read_at_most(page_addr, page_size)?;

                if is_filtered {
                    let elems =
                        decode_filtered_page(&page_buf, &self.ctx, page_nelmts, chunk_size_len)?;
                    for e in elems {
                        chunk_entries.push((e.address, e.chunk_size, e.filter_mask));
                    }
                } else {
                    let addrs = decode_unfiltered_page(&page_buf, &self.ctx, page_nelmts)?;
                    for addr in addrs {
                        chunk_entries.push((addr, chunk_bytes, 0));
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
                    chunk_entries.push((e.address, e.chunk_size, e.filter_mask));
                }
            } else {
                let fa_dblk =
                    FixedArrayDataBlock::decode_unfiltered(&dblk_buf, &self.ctx, num_elmts)?;
                for &addr in &fa_dblk.elements {
                    chunk_entries.push((addr, chunk_bytes, 0));
                }
            }
        }

        // Index-grid slot -> chunk-grid coordinates (row-major, against the
        // maximum extent — the array was sized from its chunk grid, so a slot
        // beyond the current extent still decodes to its true position and
        // then simply falls outside the read target). A zero chunk dimension
        // from a malformed layout message is rejected inside.
        let max_dims = info.dataspace.max_dims.clone();
        let mut slot_coords = Vec::with_capacity(chunk_entries.len());
        for i in 0..chunk_entries.len() as u64 {
            slot_coords.push(crate::io::chunk_grid::coords_of(
                &dims,
                max_dims.as_deref(),
                chunk_dims,
                i,
            )?);
        }
        let chunk_coords = |i: u64| -> &[u64] { &slot_coords[i as usize] };

        // Build one read job per chunk (no I/O yet). Filtered chunks carry
        // their exact compressed size (read at-most, since a zero size means
        // "unknown" and falls back to a generous estimate); unfiltered chunks
        // read the exact chunk byte count. For a slice, chunks outside the
        // selection become None and are never read.
        let jobs: Vec<Option<ChunkReadJob>> = chunk_entries
            .iter()
            .enumerate()
            .map(|(linear_idx, &(addr, comp_size, mask))| {
                if addr == UNDEF_ADDR
                    || !target.overlaps(chunk_coords(linear_idx as u64), chunk_dims)
                {
                    None
                } else if pipeline.is_some() {
                    let read_len = if comp_size > 0 {
                        comp_size as usize
                    } else {
                        chunk_bytes as usize * 2
                    };
                    Some(ChunkReadJob {
                        addr,
                        len: read_len,
                        at_most: true,
                        mask,
                    })
                } else {
                    Some(ChunkReadJob {
                        addr,
                        len: chunk_bytes as usize,
                        at_most: false,
                        mask,
                    })
                }
            })
            .collect();

        // Read + decompress (in parallel where positioned reads are race-free),
        // then place each chunk into output.
        let decompressed = read_and_decompress_chunks(&self.handle, pipeline, jobs)?;
        for (linear_idx, chunk_data) in decompressed.iter().enumerate() {
            let Some(data) = chunk_data else { continue };
            let coords = chunk_coords(linear_idx as u64);
            self.scatter_chunk(
                target,
                data,
                output,
                &dims,
                chunk_dims,
                coords,
                element_size,
            );
        }

        Ok(())
    }

    /// Read a dataset indexed by a B-tree v2.
    ///
    /// Scatters only; `output` must already be sized to the target extent and
    /// pre-filled with the tiled fill value by the caller.
    fn read_chunked_btree_v2(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        pipeline: Option<&FilterPipeline>,
        target: ChunkTarget,
        output: &mut [u8],
    ) -> IoResult<()> {
        use crate::format::chunk_index::btree_v2::*;

        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let ndims = dims.len();

        if index_address == UNDEF_ADDR {
            // Unallocated: `output` is already the pre-filled buffer.
            return Ok(());
        }

        // Read BT2 header
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let bt2_hdr = Bt2Header::decode(&hdr_buf, &self.ctx)?;

        if bt2_hdr.root_node_addr == UNDEF_ADDR || bt2_hdr.total_num_records == 0 {
            // No records: `output` is already pre-filled.
            return Ok(());
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
        let chunk_bytes: u64 = saturating_byte_len(chunk_dims, element_size);

        // Unify filtered and unfiltered records into (address, read_size,
        // scaled offsets, filter mask). read_size is the compressed size for
        // filtered chunks, the full chunk size otherwise; the mask is 0 for
        // unfiltered records.
        let entries: Vec<(u64, usize, Vec<u64>, u32)> =
            if bt2_hdr.record_type == BT2_TYPE_CHUNK_UNFILT {
                Bt2ChunkIndex::decode_unfiltered_records(
                    &record_bytes,
                    total_records,
                    ndims,
                    &self.ctx,
                )?
                .into_iter()
                .map(|r| (r.chunk_address, chunk_bytes as usize, r.scaled_offsets, 0))
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
                .map(|r| {
                    (
                        r.chunk_address,
                        r.chunk_size as usize,
                        r.scaled_offsets,
                        r.filter_mask,
                    )
                })
                .collect()
            };

        // Build one read job per chunk (no I/O yet), keeping each chunk's
        // scaled (chunk-grid) offsets alongside for the scatter. For a slice,
        // chunks outside the selection become None and are never read.
        let mut jobs: Vec<Option<ChunkReadJob>> = Vec::with_capacity(entries.len());
        let coords: Vec<&Vec<u64>> = entries.iter().map(|(_, _, scaled, _)| scaled).collect();
        for (addr, read_size, scaled, mask) in &entries {
            if *addr == UNDEF_ADDR || *read_size == 0 || !target.overlaps(scaled, chunk_dims) {
                jobs.push(None);
            } else {
                jobs.push(Some(ChunkReadJob {
                    addr: *addr,
                    len: *read_size,
                    at_most: false,
                    mask: *mask,
                }));
            }
        }

        // Read + decompress (in parallel where positioned reads are race-free),
        // then place each chunk N-dimensionally by its scaled offsets.
        let placed = read_and_decompress_chunks(&self.handle, pipeline, jobs)?;
        for (i, chunk_data) in placed.iter().enumerate() {
            if let Some(data) = chunk_data {
                self.scatter_chunk(
                    target,
                    data,
                    output,
                    &dims,
                    chunk_dims,
                    coords[i],
                    element_size,
                );
            }
        }

        Ok(())
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
    ///
    /// Scatters only; `output` must already be sized to the target extent and
    /// pre-filled with the tiled fill value by the caller.
    fn read_chunked_btree_v1(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        b_tree_address: u64,
        pipeline: Option<&FilterPipeline>,
        target: ChunkTarget,
        output: &mut [u8],
    ) -> IoResult<()> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
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

        let total_size: u64 = saturating_byte_len(&dims, element_size);
        if b_tree_address == UNDEF_ADDR || total_size == 0 {
            // Unallocated: `output` is already the pre-filled buffer.
            return Ok(());
        }

        // Walk the B-tree, collecting every leaf entry as
        // (element_offsets, chunk_address, chunk_size, filter_mask).
        // `chunk_dims.len()` is the chunk rank; the B-tree keys carry
        // rank + 1 offsets (the extra one is the element-size dimension).
        let mut entries: Vec<(Vec<u64>, u64, u32, u32)> = Vec::new();
        let file_size = self.handle.file_size()?;
        self.collect_btree_v1_chunks(b_tree_address, ndims, file_size, 0, &mut entries)?;

        // The uncompressed byte size of a full chunk.
        let chunk_bytes: u64 = saturating_byte_len(chunk_dims, element_size);

        // Build one read job per chunk (no I/O yet), keeping each chunk's
        // scaled (chunk-grid) coordinates alongside for the scatter. The
        // trailing element-size dimension offset is always 0 and is dropped.
        let mut jobs: Vec<Option<ChunkReadJob>> = Vec::with_capacity(entries.len());
        let mut coords: Vec<Vec<u64>> = Vec::with_capacity(entries.len());
        for (offsets, addr, chunk_size, mask) in &entries {
            let mut scaled = Vec::with_capacity(ndims);
            for d in 0..ndims {
                let cd = chunk_dims[d];
                scaled.push(offsets[d].checked_div(cd).unwrap_or(0));
            }
            let skip = *addr == UNDEF_ADDR
                || *chunk_size == 0
                || *addr >= file_size
                || *chunk_size as u64 > file_size
                || !target.overlaps(&scaled, chunk_dims);
            jobs.push(if skip {
                None
            } else {
                Some(ChunkReadJob {
                    addr: *addr,
                    len: *chunk_size as usize,
                    at_most: false,
                    mask: *mask,
                })
            });
            coords.push(scaled);
        }

        // Read + decompress (in parallel where positioned reads are race-free),
        // then place each chunk N-dimensionally by its scaled offsets.
        let placed = read_and_decompress_chunks(&self.handle, pipeline, jobs)?;
        for (i, chunk_data) in placed.iter().enumerate() {
            if let Some(data) = chunk_data {
                self.scatter_chunk(
                    target,
                    data,
                    output,
                    &dims,
                    chunk_dims,
                    &coords[i],
                    element_size,
                );
            }
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

        Ok(())
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

        // Multi-dimensional: the last axis is innermost in both the chunk and
        // the output, so each fixed setting of the outer dimensions copies one
        // contiguous last-axis run — no per-element loop. This mirrors
        // copy_chunk_to_slice, but writes into the full dataset extent.
        let last = ndims - 1;

        // Per dimension: the chunk's global origin and the valid extent within
        // the dataset (a chunk may hang off the high edge, so clamp).
        let mut origin = vec![0u64; ndims];
        let mut extent = vec![0u64; ndims];
        for d in 0..ndims {
            origin[d] = chunk_coords[d].saturating_mul(chunk_dims[d]);
            if origin[d] >= dims[d] {
                return; // chunk lies entirely past the extent
            }
            extent[d] = chunk_dims[d].min(dims[d] - origin[d]);
        }

        let chunk_strides = compute_strides(chunk_dims, element_size);
        let out_strides = compute_strides(dims, element_size);
        let run_bytes = (extent[last] * element_size) as usize;

        // Iterate the outer box [0, last); each position copies one contiguous
        // last-axis run. The last-axis source starts at the chunk row origin.
        let outer_extent: Vec<u64> = (0..last).map(|d| extent[d]).collect();
        let n_outer: u64 = outer_extent.iter().product(); // empty product == 1
        let mut oc = vec![0u64; last];
        for _ in 0..n_outer {
            let mut src_off = 0u64;
            let mut dst_off = 0u64;
            for d in 0..ndims {
                let local = if d < last { oc[d] } else { 0 };
                src_off += local * chunk_strides[d];
                dst_off += (origin[d] + local) * out_strides[d];
            }
            let s = src_off as usize;
            let dst = dst_off as usize;
            if s + run_bytes <= chunk_data.len() && dst + run_bytes <= output.len() {
                output[dst..dst + run_bytes].copy_from_slice(&chunk_data[s..s + run_bytes]);
            }
            for d in (0..last).rev() {
                oc[d] += 1;
                if oc[d] < outer_extent[d] {
                    break;
                }
                oc[d] = 0;
            }
        }
    }

    /// Copy the intersection of one chunk with a hyperslab selection into a
    /// `counts`-shaped slice output buffer.
    ///
    /// `chunk_coords` is the chunk-grid position (global origin =
    /// `chunk_coords[d] * chunk_dims[d]`); `output` is row-major over `counts`.
    /// The global intersection box is `[max(origin, start), min(origin +
    /// chunk_dims, start + count, dims))` per dimension. Because the last axis
    /// is innermost in both the chunk and the output, each fixed setting of the
    /// outer box dimensions copies one contiguous run of
    /// `(hi[last]-lo[last])` elements — no per-element loop.
    #[allow(clippy::too_many_arguments)]
    fn copy_chunk_to_slice(
        &self,
        chunk_data: &[u8],
        output: &mut [u8],
        dims: &[u64],
        chunk_dims: &[u64],
        chunk_coords: &[u64],
        element_size: u64,
        starts: &[u64],
        counts: &[u64],
    ) {
        let ndims = dims.len();
        if ndims == 0 {
            return;
        }
        // Global intersection box [lo, hi) of chunk ∩ selection ∩ dataset.
        let mut lo = vec![0u64; ndims];
        let mut hi = vec![0u64; ndims];
        for d in 0..ndims {
            let origin = chunk_coords[d].saturating_mul(chunk_dims[d]);
            let chunk_end = origin.saturating_add(chunk_dims[d]).min(dims[d]);
            let sel_end = starts[d].saturating_add(counts[d]);
            lo[d] = origin.max(starts[d]);
            hi[d] = chunk_end.min(sel_end);
            if lo[d] >= hi[d] {
                return; // no overlap in this dimension
            }
        }

        let chunk_strides = compute_strides(chunk_dims, element_size);
        let out_strides = compute_strides(counts, element_size);
        let last = ndims - 1;
        let run_bytes = ((hi[last] - lo[last]) * element_size) as usize;

        // Iterate the outer box dimensions [0, last); each position copies one
        // contiguous last-axis run.
        let outer_extent: Vec<u64> = (0..last).map(|d| hi[d] - lo[d]).collect();
        let n_outer: u64 = outer_extent.iter().product(); // empty product == 1
        let mut oc = vec![0u64; last];
        for _ in 0..n_outer {
            let mut src_off = 0u64;
            let mut dst_off = 0u64;
            for d in 0..ndims {
                let g = if d < last { lo[d] + oc[d] } else { lo[last] };
                let origin = chunk_coords[d].saturating_mul(chunk_dims[d]);
                src_off += (g - origin) * chunk_strides[d];
                dst_off += (g - starts[d]) * out_strides[d];
            }
            let s = src_off as usize;
            let dst = dst_off as usize;
            if s + run_bytes <= chunk_data.len() && dst + run_bytes <= output.len() {
                output[dst..dst + run_bytes].copy_from_slice(&chunk_data[s..s + run_bytes]);
            }
            for d in (0..last).rev() {
                oc[d] += 1;
                if oc[d] < outer_extent[d] {
                    break;
                }
                oc[d] = 0;
            }
        }
    }

    /// Scatter one decoded chunk into the output for the given target: the
    /// whole dataset (`Full`) or a hyperslab (`Slice`).
    #[allow(clippy::too_many_arguments)]
    fn scatter_chunk(
        &self,
        target: ChunkTarget,
        chunk_data: &[u8],
        output: &mut [u8],
        dims: &[u64],
        chunk_dims: &[u64],
        chunk_coords: &[u64],
        element_size: u64,
    ) {
        match target {
            ChunkTarget::Full => self.copy_chunk_to_output(
                chunk_data,
                output,
                dims,
                chunk_dims,
                chunk_coords,
                element_size,
            ),
            ChunkTarget::Slice { starts, counts } => self.copy_chunk_to_slice(
                chunk_data,
                output,
                dims,
                chunk_dims,
                chunk_coords,
                element_size,
                starts,
                counts,
            ),
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
        // A vlen string is a vlen sequence of `u8` reinterpreted as UTF-8.
        // The global-heap walk is identical; decode the raw object bytes.
        Ok(self
            .read_vlen_objects(name)?
            .into_iter()
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .collect())
    }

    /// Read a 1-D variable-length byte-array dataset (vlen sequence of `u8`).
    ///
    /// Returns a `Vec<Vec<u8>>` with one byte array per element. Missing or
    /// undefined references yield an empty `Vec`.
    pub fn read_vlen_bytes(&mut self, name: &str) -> IoResult<Vec<Vec<u8>>> {
        self.read_vlen_objects(name)
    }

    /// Shared owner of the global-heap walk for variable-length datasets.
    ///
    /// Each element of the raw data is a vlen reference (collection address +
    /// object index) into a global heap collection. Returns the raw object
    /// bytes for each element, with an empty `Vec` for undefined/missing
    /// references. Both `read_vlen_strings` (UTF-8 view) and `read_vlen_bytes`
    /// (raw view) layer on top of this.
    fn read_vlen_objects(&mut self, name: &str) -> IoResult<Vec<Vec<u8>>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let layout = info.layout.clone();
        let total_elements: u64 = dims.iter().fold(1u64, |acc, &d| acc.saturating_mul(d));

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
        let mut items: Vec<Vec<u8>> = Vec::with_capacity(total_elements as usize);

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
                items.push(Vec::new());
                continue;
            }

            // Read or get cached global heap collection
            #[allow(clippy::map_entry)]
            if !heap_cache.contains_key(&collection_addr) {
                let coll = self.read_heap_collection(collection_addr)?;
                let lookup: std::collections::HashMap<u16, usize> = coll
                    .objects
                    .iter()
                    .enumerate()
                    .map(|(i, o)| (o.index, i))
                    .collect();
                heap_cache.insert(collection_addr, (coll, lookup));
            }

            let idx = u16::try_from(obj_index).map_err(|_| {
                crate::io::IoError::InvalidState(format!(
                    "global heap object index {obj_index} does not fit the 16-bit on-disk field \
                     (element {i} of \"{name}\")"
                ))
            })?;
            let (coll, lookup) = &heap_cache[&collection_addr];
            let &oi = lookup.get(&idx).ok_or_else(|| {
                crate::io::IoError::InvalidState(format!(
                    "global heap object {idx} not found in the collection at address \
                     {collection_addr:#x} (element {i} of \"{name}\")"
                ))
            })?;
            items.push(coll.objects[oi].data.clone());
        }

        Ok(items)
    }

    /// Collect chunk (address, size) entries from an EA index.
    /// Returns a vector indexed by chunk linear index.
    fn collect_ea_chunk_entries(
        &mut self,
        index_address: u64,
        params: &data_layout::EarrayParams,
        dims: &[u64],
        max_dims: Option<&[u64]>,
        chunk_dims: &[u64],
        element_size: u64,
    ) -> IoResult<Vec<(u64, u64, u32)>> {
        use crate::format::chunk_index::extensible_array::{self as ea, *};

        if index_address == UNDEF_ADDR {
            return Ok(vec![]);
        }
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let ea_hdr = ExtensibleArrayHeader::decode(&hdr_buf, &self.ctx)?;
        if ea_hdr.idx_blk_addr == UNDEF_ADDR {
            return Ok(vec![]);
        }

        // Slot count of the index grid, bounding the collection walk. The
        // maximum extent decides the multipliers (sub-frame chunks make this
        // larger than the dim-0 chunk count alone); the unlimited dimension 0
        // is bounded by the current extent.
        let chunks_dim0: usize = crate::io::chunk_grid::index_grid(dims, max_dims, chunk_dims)?
            .iter()
            .fold(1usize, |acc, &n| acc.saturating_mul(n as usize));
        let geo = EaGeometry::new(
            params.idx_blk_elmts,
            params.data_blk_min_elmts,
            params.sup_blk_min_data_ptrs,
            params.max_nelmts_bits,
            params.max_dblk_page_nelmts_bits,
        )?;
        let chunk_bytes = saturating_byte_len(chunk_dims, element_size);
        let is_filtered = ea_hdr.class_id == ea::EA_CLS_FILT_CHUNK;
        let chunk_size_len = if is_filtered {
            ea_hdr.raw_elmt_size - self.ctx.sizeof_addr - 4
        } else {
            0
        };
        let max_nelmts_bits = params.max_nelmts_bits;
        // Each entry is (chunk address, on-disk byte count, filter mask). The
        // mask is the per-chunk filter mask for filtered datasets (0 for an
        // unfiltered index, where it is meaningless).
        let mut entries: Vec<(u64, u64, u32)> = Vec::new();

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
                entries.push((e.addr, e.nbytes, e.filter_mask));
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
                entries.push((addr, chunk_bytes, 0));
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
                    entries.extend(std::iter::repeat_n((UNDEF_ADDR, 0, 0), dblk_nelmts));
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
                                (UNDEF_ADDR, 0, 0),
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
                                entries.push((e.addr, e.nbytes, e.filter_mask));
                            } else {
                                entries.push((read_addr(&page[off..], sa), chunk_bytes, 0));
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
                        entries.push((e.addr, e.nbytes, e.filter_mask));
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
                        entries.push((addr, chunk_bytes, 0));
                    }
                }
                if entries.len() >= chunks_dim0 {
                    break 'outer;
                }
            }
        }
        Ok(entries)
    }

    /// Read a slice (hyperslab) of a dataset, whatever its layout.
    ///
    /// Contiguous and compact datasets are read run by run; a chunked one
    /// reads only the chunks the selection overlaps, and any gap the writer
    /// never filled comes back as the fill value.
    ///
    /// `starts` and `counts` define the N-dimensional selection:
    /// starts[d] is the first index along dim d, counts[d] is how many.
    /// Returns the selected data in row-major order.
    pub fn read_slice(&mut self, name: &str, starts: &[u64], counts: &[u64]) -> IoResult<Vec<u8>> {
        let (datatype, out_bytes) = self.slice_size_and_datatype(name, counts)?;
        let mut data = alloc_tiled_fill(out_bytes as usize, None)?;
        self.read_slice_into_unconverted(name, starts, counts, &mut data)?;
        Self::apply_post_filter_conversion(&mut data, &datatype)?;
        Ok(data)
    }

    /// Read a hyperslab straight into a caller-provided buffer (no allocation).
    ///
    /// `out.len()` must equal `product(counts) * element_size`; otherwise an
    /// error is returned. The no-allocation counterpart of
    /// [`read_slice`](Self::read_slice) and the zero-copy entry point for
    /// reading a selection directly into a pinned/registered host buffer for an
    /// H2D transfer.
    pub fn read_slice_into(
        &mut self,
        name: &str,
        starts: &[u64],
        counts: &[u64],
        out: &mut [u8],
    ) -> IoResult<()> {
        let (datatype, out_bytes) = self.slice_size_and_datatype(name, counts)?;
        if out.len() as u64 != out_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "read_slice_into: buffer is {} bytes but selection needs {}",
                out.len(),
                out_bytes
            )));
        }
        self.read_slice_into_unconverted(name, starts, counts, out)?;
        Self::apply_post_filter_conversion(out, &datatype)?;
        Ok(())
    }

    /// Logical byte size of a hyperslab (`product(counts) * element_size`) with
    /// the datatype needed for the post-filter conversion.
    fn slice_size_and_datatype(
        &self,
        name: &str,
        counts: &[u64],
    ) -> IoResult<(DatatypeMessage, u64)> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let out_bytes = saturating_byte_len(counts, info.datatype.element_size() as u64);
        Ok((info.datatype.clone(), out_bytes))
    }

    /// Fill `out` with a hyperslab selection, before the post-filter datatype
    /// conversion. The single owner of read-destination semantics for slice
    /// reads (mirrors [`read_dataset_raw_into_unconverted`](Self::read_dataset_raw_into_unconverted)):
    /// it validates the selection, then fully defines every byte of `out`
    /// (contiguous/compact runs cover the whole selection; chunked layouts
    /// pre-fill with the tiled fill value and scatter only overlapping chunks).
    /// Both [`read_slice`](Self::read_slice) and [`read_slice_into`](Self::read_slice_into)
    /// wrap it and apply the conversion exactly once.
    ///
    /// `out.len()` must equal `product(counts) * element_size`.
    fn read_slice_into_unconverted(
        &mut self,
        name: &str,
        starts: &[u64],
        counts: &[u64],
        out: &mut [u8],
    ) -> IoResult<()> {
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

        match &layout {
            DataLayoutMessage::Contiguous { address, .. } => {
                if *address == UNDEF_ADDR {
                    // Never-written: the selection reads back as the fill value.
                    fill_tiled_into(out, fill_value.as_deref());
                } else {
                    // Read each maximal contiguous run straight into `out`.
                    // Trailing full-selected dimensions coalesce, so a slice
                    // like `[:, r0:r1, :]` of `[nproj, nz, nx]` becomes `nproj`
                    // reads of `(r1-r0)*nx` elements instead of `nproj*(r1-r0)`
                    // per-`nx`-row reads. The 1-D case folds to a single run.
                    // The runs cover the whole selection, so every byte of `out`
                    // is written.
                    let base = *address;
                    for_each_contiguous_run(
                        &dims,
                        starts,
                        counts,
                        element_size,
                        |src_off, out_off, len| {
                            self.handle
                                .read_exact_at_into(
                                    base + src_off,
                                    &mut out[out_off..out_off + len],
                                )
                                .map_err(Into::into)
                        },
                    )?;
                }
            }
            DataLayoutMessage::Compact { data } => {
                // Same coalesced geometry, copying from the in-memory full
                // dataset instead of reading from the file.
                for_each_contiguous_run(
                    &dims,
                    starts,
                    counts,
                    element_size,
                    |src_off, out_off, len| {
                        let src = src_off as usize;
                        out[out_off..out_off + len].copy_from_slice(&data[src..src + len]);
                        Ok(())
                    },
                )?;
            }
            DataLayoutMessage::ChunkedV3 {
                chunk_dims,
                b_tree_address,
            } => {
                // Walk the v1 B-tree index reading only chunks that overlap the
                // selection, scattering each chunk∩selection into the slice
                // buffer. Pre-fill so any unwritten gap reads back as fill. The
                // unconverted read keeps the post-filter conversion to exactly
                // once, in the wrappers.
                fill_tiled_into(out, fill_value.as_deref());
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_btree_v1(
                    name,
                    real_chunk_dims,
                    *b_tree_address,
                    pipeline.as_ref(),
                    ChunkTarget::Slice { starts, counts },
                    out,
                )?;
            }
            DataLayoutMessage::ChunkedV4 {
                chunk_dims,
                index_address,
                index_type,
                earray_params,
                single_chunk_filter,
                ..
            } => {
                // Same selection-aware chunk read for every v4 index kind
                // (single chunk, fixed/extensible array, B-tree v2): only
                // overlapping chunks are read and only their intersection with
                // the selection is scattered into the slice output.
                fill_tiled_into(out, fill_value.as_deref());
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                self.read_chunked_v4(
                    name,
                    real_chunk_dims,
                    ChunkIndexDesc {
                        index_type: *index_type,
                        index_address: *index_address,
                        earray_params: earray_params.as_ref(),
                        single_chunk_filter: *single_chunk_filter,
                    },
                    pipeline.as_ref(),
                    ChunkTarget::Slice { starts, counts },
                    out,
                )?;
            }
        }
        Ok(())
    }
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
            let writer = Hdf5Writer::create(&path).unwrap();
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

    /// Collect the (src_off, out_off, len) runs the coalescer emits.
    fn collect_runs(
        dims: &[u64],
        starts: &[u64],
        counts: &[u64],
        es: u64,
    ) -> Vec<(u64, usize, usize)> {
        let mut v = Vec::new();
        for_each_contiguous_run(dims, starts, counts, es, |s, o, l| {
            v.push((s, o, l));
            Ok(())
        })
        .unwrap();
        v
    }

    #[test]
    fn coalesce_1d_is_single_run() {
        // 1-D selection is always one contiguous run.
        assert_eq!(collect_runs(&[10], &[2], &[3], 1), vec![(2, 0, 3)]);
        // element_size scales offsets and length.
        assert_eq!(collect_runs(&[10], &[2], &[3], 4), vec![(8, 0, 12)]);
    }

    #[test]
    fn coalesce_full_last_dim_merges_into_one_run() {
        // 2-D, full last dim => the whole [r0:r1, :] block is one run.
        // dims=[4,5], select rows 1..3, all 5 columns.
        assert_eq!(collect_runs(&[4, 5], &[1, 0], &[2, 5], 1), vec![(5, 0, 10)]);
    }

    #[test]
    fn coalesce_partial_last_dim_keeps_one_run_per_row() {
        // 2-D, partial last dim => no merge; one run per selected row.
        // dims=[4,5] strides=[5,1]; select rows 1..3, cols 1..4.
        assert_eq!(
            collect_runs(&[4, 5], &[1, 1], &[2, 3], 1),
            vec![(6, 0, 3), (11, 3, 3)]
        );
        // Same shape, element_size=4: strides=[20,4], inner_base=4.
        assert_eq!(
            collect_runs(&[4, 5], &[1, 1], &[2, 3], 4),
            vec![(24, 0, 12), (44, 12, 12)]
        );
    }

    #[test]
    fn coalesce_3d_reported_case_one_run_per_outer_index() {
        // The reported workload: [:, r0:r1, :] of [nproj, nz, nx].
        // dims=[3,4,5] strides=[20,5,1]; select all of dim0, rows 1..3 of
        // dim1, all of dim2. Last dim full => merge dim1+dim2; dim1 partial
        // => one run per dim0 index (3 runs, not 3*2=6 rows).
        assert_eq!(
            collect_runs(&[3, 4, 5], &[0, 1, 0], &[3, 2, 5], 1),
            vec![(5, 0, 10), (25, 10, 10), (45, 20, 10)]
        );
    }

    #[test]
    fn coalesce_3d_full_inner_dims_is_single_run() {
        // [r0:r1, :, :] => both inner dims full => one contiguous run.
        // dims=[3,4,5] strides=[20,5,1]; select rows 1..3 of dim0.
        assert_eq!(
            collect_runs(&[3, 4, 5], &[1, 0, 0], &[2, 4, 5], 1),
            vec![(20, 0, 40)]
        );
    }

    /// Build a contiguous i32 dataset and verify `read_slice` returns the
    /// correct bytes for both coalesced and non-coalesced selections.
    #[test]
    fn read_slice_contiguous_3d_matches_naive_extraction() {
        let dims = [3u64, 4, 5];
        let total: usize = (dims[0] * dims[1] * dims[2]) as usize;
        let values: Vec<i32> = (0..total as i32).collect();
        let raw_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let file_bytes = build_v0_file("vol", &dims, &raw_data);

        let path = temp_path("slice_3d_contig");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&file_bytes).unwrap();
            f.sync_all().unwrap();
        }
        let mut reader = Hdf5Reader::open(&path).unwrap();

        // Naive row-major extraction for an arbitrary [starts, counts).
        let expect = |starts: [u64; 3], counts: [u64; 3]| -> Vec<i32> {
            let mut out = Vec::new();
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    for k in 0..counts[2] {
                        let gi = starts[0] + i;
                        let gj = starts[1] + j;
                        let gk = starts[2] + k;
                        out.push(values[(gi * dims[1] * dims[2] + gj * dims[2] + gk) as usize]);
                    }
                }
            }
            out
        };
        let decode = |raw: Vec<u8>| -> Vec<i32> {
            raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        // Mix of coalesced and non-coalesced selections.
        let cases: &[([u64; 3], [u64; 3])] = &[
            ([0, 1, 0], [3, 2, 5]), // [:, 1:3, :]  -> coalesced (3 runs)
            ([1, 0, 0], [2, 4, 5]), // [1:3, :, :]  -> single run
            ([0, 0, 1], [3, 4, 3]), // [:, :, 1:4]  -> partial last dim, no merge
            ([1, 2, 1], [2, 2, 4]), // interior block, partial all dims
            ([0, 0, 0], [3, 4, 5]), // whole dataset -> single run
            ([2, 3, 4], [1, 1, 1]), // single element
        ];
        for &(starts, counts) in cases {
            let got = decode(reader.read_slice("vol", &starts, &counts).unwrap());
            assert_eq!(
                got,
                expect(starts, counts),
                "slice starts={starts:?} counts={counts:?}"
            );
        }

        let _ = std::fs::remove_file(&path);
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

        let handle = FileHandle::open_read(path).unwrap();
        let sb_buf = handle.read_at_most(0, 1024).unwrap();
        let version = detect_superblock_version(&sb_buf).unwrap();
        eprintln!("Superblock version: {}", version);

        let sb = SuperblockV0V1::decode(&sb_buf).unwrap();
        eprintln!(
            "sizeof_addr={}, sizeof_size={}",
            sb.sizeof_offsets, sb.sizeof_lengths
        );
        let (ste_btree, ste_heap) = sb
            .root_symbol_table_entry
            .cached_symbol_table()
            .unwrap_or((UNDEF_ADDR, UNDEF_ADDR));
        eprintln!(
            "STE: obj_header={}, cache={:?}, btree={}, heap={}",
            sb.root_symbol_table_entry.obj_header_addr,
            sb.root_symbol_table_entry.cache,
            ste_btree,
            ste_heap
        );

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // Read local heap
        let heap_buf = handle.read_at_most(ste_heap, 128).unwrap();
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
        let btree_buf = handle.read_at_most(ste_btree, 8192).unwrap();
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
                    "  entry: name='{}' (offset={}), obj_header={}, cache={:?}",
                    name, entry.name_offset, entry.obj_header_addr, entry.cache
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

    /// Partial-slice reads of chunked datasets must skip non-overlapping
    /// chunks yet return exactly the selected region, for every chunk index
    /// type: v1 B-tree (libver=earliest), single chunk, fixed array,
    /// extensible array (one unlimited dim), and v2 B-tree (>1 unlimited dim).
    ///
    /// The dataset is a 5×4×6 int32 `arange`, chunked 2×2×2 so the chunk grid
    /// is ragged (edge chunks) and most selections touch a strict subset of
    /// chunks. Each slice is checked against the row-major `arange` value so a
    /// dropped/misplaced chunk or a mis-sized output buffer is caught.
    #[test]
    fn read_slice_chunked_all_index_types() {
        let latest = temp_path("slice_chunk_latest");
        let earliest = temp_path("slice_chunk_earliest");
        let pl = latest.display().to_string();
        let pe = earliest.display().to_string();
        // libver=latest selects modern indices by maxshape: fixed -> Fixed
        // Array, one unlimited dim -> Extensible Array, >1 unlimited -> v2
        // B-tree, single chunk -> Single Chunk index. libver=earliest always
        // uses the v1 B-tree chunk index.
        let script = format!(
            "import h5py,numpy as np\n\
             a=np.arange(5*4*6,dtype='int32').reshape(5,4,6)\n\
             f=h5py.File(r'{pl}','w',libver='latest')\n\
             f.create_dataset('single',data=a,chunks=(5,4,6))\n\
             f.create_dataset('fa',data=a,chunks=(2,2,2))\n\
             f.create_dataset('ea',data=a,chunks=(2,2,2),maxshape=(None,4,6))\n\
             f.create_dataset('btv2',data=a,chunks=(2,2,2),maxshape=(None,None,6))\n\
             f.close()\n\
             g=h5py.File(r'{pe}','w',libver='earliest')\n\
             g.create_dataset('btv1',data=a,chunks=(2,2,2))\n\
             g.close()"
        );
        if !gen_fixture(&script) {
            eprintln!("skipping read_slice_chunked_all_index_types: python unavailable");
            return;
        }

        let dims = [5u64, 4, 6];
        // Row-major value of element (i,j,k) in the arange dataset.
        let val = |i: u64, j: u64, k: u64| (i * dims[1] * dims[2] + j * dims[2] + k) as i32;
        let expect = |starts: [u64; 3], counts: [u64; 3]| -> Vec<i32> {
            let mut out = Vec::new();
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    for k in 0..counts[2] {
                        out.push(val(starts[0] + i, starts[1] + j, starts[2] + k));
                    }
                }
            }
            out
        };
        let decode = |raw: Vec<u8>| -> Vec<i32> {
            raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let cases: &[([u64; 3], [u64; 3])] = &[
            ([0, 1, 0], [5, 2, 6]), // full last dim, partial mid -> coalesced runs
            ([1, 0, 0], [3, 4, 6]), // partial dim0, inner dims full -> one run
            ([0, 0, 2], [5, 4, 3]), // partial last dim -> one run per (i,j) row
            ([2, 1, 3], [1, 2, 2]), // interior block spanning few chunks
            ([0, 0, 0], [5, 4, 6]), // whole dataset via read_slice
            ([4, 3, 5], [1, 1, 1]), // single element at the far edge chunk
            ([1, 1, 1], [3, 3, 4]), // straddles chunk boundaries on all axes
        ];

        let mut reader_l = Hdf5Reader::open(&latest).unwrap();
        for name in ["single", "fa", "ea", "btv2"] {
            // Full read sanity first, then every slice.
            let full = decode(reader_l.read_dataset_raw(name).unwrap());
            assert_eq!(full, expect([0, 0, 0], [5, 4, 6]), "{name} full read");
            for &(starts, counts) in cases {
                let got = decode(reader_l.read_slice(name, &starts, &counts).unwrap());
                assert_eq!(
                    got,
                    expect(starts, counts),
                    "{name} slice starts={starts:?} counts={counts:?}"
                );
            }
        }

        let mut reader_e = Hdf5Reader::open(&earliest).unwrap();
        let full = decode(reader_e.read_dataset_raw("btv1").unwrap());
        assert_eq!(full, expect([0, 0, 0], [5, 4, 6]), "btv1 full read");
        for &(starts, counts) in cases {
            let got = decode(reader_e.read_slice("btv1", &starts, &counts).unwrap());
            assert_eq!(
                got,
                expect(starts, counts),
                "btv1 slice starts={starts:?} counts={counts:?}"
            );
        }

        let _ = std::fs::remove_file(&latest);
        let _ = std::fs::remove_file(&earliest);
    }
}
