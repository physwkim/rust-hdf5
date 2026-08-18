//! HDF5 file reader.
//!
//! Opens an HDF5 file, parses the superblock and root group, and provides
//! access to dataset metadata and raw data.
//!
//! Supports both legacy (v0/v1 superblock, v1 object headers, symbol tables)
//! and modern (v2/v3 superblock, v2 object headers, link messages) formats.

use std::path::{Path, PathBuf};

use crate::dataset::{DatasetAccess, VirtualView};
use crate::format::btree_v1::{BTreeV1Config, BTreeV1Node, ChunkBTreeV1Node};
use crate::format::bytes::read_le_uint as read_uint;
use crate::format::creation_order::CreationOrder;
use crate::format::fractal_heap::{self, FractalHeapHeader};
use crate::format::global_heap::{
    decode_vlen_reference, vlen_reference_size, GlobalHeapCollection,
};
use crate::format::local_heap::{local_heap_get_string, LocalHeapHeader};
use crate::format::messages::attr_info::AttributeInfoMessage;
use crate::format::messages::attribute::{AttributeEntry, AttributeMessage};
use crate::format::messages::data_layout::{self, DataLayoutMessage};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::{DatatypeMessage, OldReferenceKind, ReferenceEncoding};
use crate::format::messages::external_file_list::ExternalFileListMessage;
use crate::format::messages::fill_value::{
    try_tiled_fill, FillValueMessage, ALLOC_TIME_LATE, FILL_TIME_IFSET,
};
use crate::format::messages::filter::{self, FilterPipeline};
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link::LinkTarget;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::shared::MSG_FLAG_SHARED;
use crate::format::messages::superblock_ext::{
    BtreeKMessage, DriverInfoMessage, FileSpaceInfoMessage, SharedMessageTableMessage,
};
use crate::format::messages::virtual_mapping::{
    parse_source_name, VirtualMapping, VirtualMappingList,
};
use crate::format::messages::*;
use crate::format::object_header::ObjectHeader;
use crate::format::reference::{
    decode_object_element, decode_region_element, decode_region_heap_object, decode_revised_body,
    decode_revised_element, DecodedReference, Reference, ReferenceTarget, RevisedElement,
};
use crate::format::selection::{Hyperslab, PointSelection, RegularHyperslab, Selection};
use crate::format::sohm::SohmMasterTable;
use crate::format::storage_kind::{AttributeStorage, LinkStorage};
use crate::format::superblock::{
    detect_superblock_version, SuperblockV0V1, SuperblockV2V3, SymbolTableCache,
};
use crate::format::symbol_table::SymbolTableNode;
use crate::format::{BlockReader, FormatContext, UNDEF_ADDR};

use crate::io::file_handle::FileHandle;
#[cfg(feature = "mmap")]
use crate::io::file_handle::MmapFileHandle;
use crate::io::hyperslab::{compute_strides, for_each_contiguous_run, for_each_dual_run};
use crate::io::locking::FileLocking;
use crate::io::{FileMeta, IoResult};

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

/// One v2-B-tree chunk-index record, resolved to `(chunk address, on-disk
/// read size, scaled chunk-grid offsets, filter mask)` — see
/// [`Hdf5Reader::collect_bt2_chunk_entries`].
type Bt2ChunkEntry = (u64, usize, Vec<u64>, u32);

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

/// The dataset extent, chunk shape and element size a chunk-placement call
/// reads through — constant across every chunk of one read, so
/// [`Hdf5Reader::copy_chunk_to_output`], [`Hdf5Reader::copy_chunk_to_slice`]
/// and [`Hdf5Reader::scatter_chunk`] take this once instead of the three
/// fields separately, leaving only what actually varies per chunk (its data
/// and grid coordinates) as their own parameters.
#[derive(Clone, Copy)]
struct ChunkOutputGeometry<'a> {
    dims: &'a [u64],
    chunk_dims: &'a [u64],
    element_size: u64,
}

/// One resolved external-file slot (H5O_EFL_ID): the on-disk message
/// stores each slot's name as an offset into a local heap, so this is that
/// slot after the heap lookup, in the order the dataset's logical byte
/// range concatenates them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFileSegment {
    /// The external file's name, exactly as stored — relative names are
    /// resolved against `HDF5_EXTFILE_PREFIX` at read time, not here.
    pub name: String,
    /// Byte offset within the named file where this slot's reserved
    /// region begins.
    pub offset: u64,
    /// Bytes reserved for this slot. `u64::MAX` (`H5O_EFL_UNLIMITED`)
    /// marks the last slot as unlimited/growable.
    pub size: u64,
}

/// Read-side metadata for a single dataset.
pub struct DatasetReadInfo {
    /// Dataset name (the link name in the root group).
    pub name: String,
    /// Address of the dataset's object header — what an object reference to
    /// this dataset stores.
    pub object_header_address: u64,
    /// Element datatype.
    pub datatype: DatatypeMessage,
    /// Dataspace (dimensionality).
    pub dataspace: DataspaceMessage,
    /// Data layout (contiguous, compact, or chunked).
    pub layout: DataLayoutMessage,
    /// Filter pipeline for compressed chunks (None = uncompressed).
    pub filter_pipeline: Option<FilterPipeline>,
    /// Attributes attached to this dataset.
    pub attributes: ObjectAttributes,
    /// User-defined fill value bytes (one element wide), decoded from the
    /// fill-value message when `fill_defined == 2`. `None` => default
    /// zero-fill. Applied to unallocated chunks and unwritten regions.
    pub fill_value: Option<Vec<u8>>,
    /// The fill-value message's own definedness byte
    /// (`H5D_fill_value_t`/`H5Pfill_value_defined`): 0 = explicitly
    /// undefined (no fill is ever performed), 1 = default (zero-fill, no
    /// value stored), 2 = user-defined (`fill_value` carries the bytes). A
    /// dataset with no fill-value message at all reads as 1, matching a
    /// fresh dataset creation property list (`FillValueMessage::default`).
    pub fill_defined: u8,
    /// The fill-value message's write-time byte (`H5D_fill_time_t`): 0 =
    /// `H5D_FILL_TIME_ALLOC`, 1 = `H5D_FILL_TIME_NEVER`, 2 =
    /// `H5D_FILL_TIME_IFSET`. A dataset with no fill-value message at all
    /// reads as 2, `H5D_CRT_FILL_TIME_DEF` — the same "no message" default
    /// [`fill_defined`](Self::fill_defined) uses.
    pub fill_write_time: u8,
    /// The fill-value message's space-allocation-time byte
    /// (`H5D_alloc_time_t`): 1 = `H5D_ALLOC_TIME_EARLY`, 2 =
    /// `H5D_ALLOC_TIME_LATE`, 3 = `H5D_ALLOC_TIME_INCR`. A dataset with no
    /// fill-value message at all reads as `ALLOC_TIME_LATE`, matching
    /// [`FillValueMessage::default`]'s "no message" convention.
    pub alloc_time: u8,
    /// External raw-data segments (H5O_EFL_ID). Non-empty only when this
    /// dataset's storage is an External Data Files list instead of a
    /// normal contiguous block — `layout` still reports `Contiguous` with
    /// an undefined address in that case (H5Dlayout.c overrides the
    /// layout's storage ops whenever this message is present).
    pub external_files: Vec<ExternalFileSegment>,
    /// Virtual dataset source/virtual mappings (H5D_VIRTUAL), resolved from
    /// the global heap object `layout`'s `Virtual` variant points at.
    /// `Some` only when `layout` is `DataLayoutMessage::Virtual` and it
    /// names a mapping list (`heap_index != 0`); `None` for every other
    /// layout, and for a virtual dataset that has no mappings yet.
    pub virtual_mappings: Option<VirtualMappingList>,
    /// What each mapping in [`virtual_mappings`](Self::virtual_mappings)
    /// resolved to when the file was opened, in the same order — see
    /// [`MappingResolution`] and `Hdf5Reader::resolve_virtual_extents`.
    /// `None` for every non-virtual dataset and for a virtual one with no
    /// mapping list.
    pub virtual_resolution: Option<Vec<MappingResolution>>,
    /// The extent this dataset's dataspace message stores, kept because
    /// `dataspace.dims` holds the extent the *sources* gave it once
    /// `resolve_virtual_extents` has run. `H5D__virtual_set_extent_unlim`
    /// resolves from the space freshly loaded from the object header on
    /// every `H5Dopen` (H5Dvirtual.c:1386), so a later open under different
    /// [`DatasetAccess`] must resolve from this, not from its own last
    /// answer.
    ///
    /// `Some` exactly for a virtual dataset whose extent has been resolved;
    /// `None` for every dataset whose extent is simply its stored one.
    pub virtual_stored_dims: Option<Vec<u64>>,
}

/// What one virtual-dataset mapping resolved to at open time —
/// `H5D__virtual_set_extent_unlim` (H5Dvirtual.c), which libhdf5 runs when
/// the dataset is opened and which both the dataset's reported extent and
/// every read of it depend on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappingResolution {
    /// Neither selection grows: the mapping is already concrete, and the
    /// dataset's stored extent is its extent.
    Bounded,
    /// Both selections are unlimited (`unlim_dim_virtual >= 0` and
    /// `unlim_dim_source >= 0`). `virtual_clip` is how far this mapping
    /// reaches in its unlimited virtual dimension, `source_clip` the source
    /// dataset's own extent in its unlimited source dimension — the two
    /// values `H5S_hyper_clip_unlim` clips the mapping's selections to.
    Unlimited { virtual_clip: u64, source_clip: u64 },
    /// A printf mapping (unlimited virtual selection, limited source
    /// selection, `%b` in a source name).
    ///
    /// `blocks` is upstream's `first_missing`: the scan stops at the first
    /// block whose source is absent and then looks
    /// [`DatasetAccess::virtual_printf_gap`] blocks further, so blocks
    /// `0..blocks` are the ones the extent covers. `present` lists which of
    /// them actually have a source — with a non-zero gap the others are
    /// inside the extent but read as the fill value.
    Printf { blocks: u64, present: Vec<u64> },
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

/// Where a path leaves this file: the external link it crosses, the file that
/// link names, and the remainder of the path inside that file.
pub(crate) struct ExternalEdge {
    pub link: String,
    pub file: String,
    pub path: String,
}

impl ExternalEdge {
    /// The error for "the link resolved to a file, but the object it names is
    /// not in it" — the external counterpart of a dangling soft link.
    fn dangling(&self) -> crate::io::IoError {
        crate::io::IoError::DanglingLink {
            link: self.link.clone(),
            target: format!("{}::{}", self.file, self.path),
        }
    }
}

/// How a reader was opened, carried from the entry point to the per-superblock
/// constructors: both facts an external link needs later (where to resolve a
/// relative target from, and under which locking policy to open it) are fixed
/// at open time and belong together.
struct Origin {
    path: PathBuf,
    locking: crate::io::locking::FileLocking,
}

/// Bound on how many external links one path resolution may cross, matching
/// libhdf5's `H5L_NUM_LINKS` (the `H5Pset_nlinks` default). Two files that
/// link to each other form a cycle whose every hop opens a fresh target, so it
/// is this count, not target identity, that terminates the walk.
const MAX_EXTERNAL_HOPS: usize = 16;

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
    group_attributes: std::collections::HashMap<String, ObjectAttributes>,
    /// Link storage kind and link creation-order policy of non-root groups,
    /// keyed by group path.
    group_link_storage: std::collections::HashMap<String, (LinkStorage, CreationOrder)>,
    /// Every non-root group path the walk traversed into.
    group_paths: std::collections::BTreeSet<String>,
    /// Group object header address → the first path that reached it (no
    /// leading `/`), taken from the walk's cycle guard. What turns the
    /// address an object reference stores back into a name.
    group_object_paths: std::collections::HashMap<u64, String>,
    /// Group hard-link aliases: alias path → first-walked path.
    group_aliases: std::collections::HashMap<String, String>,
    /// Every link record seen, keyed by its full path (no leading `/`).
    links: std::collections::BTreeMap<String, LinkClass>,
    /// Committed (named) datatype objects, keyed by path.
    datatypes: std::collections::BTreeMap<String, CommittedDatatypeInfo>,
}

impl Catalog {
    /// The address→absolute-path catalog this walk implies, with `root_addr`
    /// named `/` whether or not the walk itself reached it.
    ///
    /// The single owner of the catalog: file open and the SWMR
    /// [`Hdf5Reader::refresh`] rescan both build it here, so a dataset that
    /// appears after open is resolvable exactly as one present at open is.
    fn object_paths(&self, root_addr: u64) -> std::collections::HashMap<u64, String> {
        let mut paths = std::collections::HashMap::new();
        paths.insert(root_addr, "/".to_string());
        for (addr, path) in &self.group_object_paths {
            paths.insert(*addr, absolute_path(path));
        }
        for ds in &self.datasets {
            paths.insert(ds.object_header_address, absolute_path(&ds.name));
        }
        paths
    }
}

/// The state one catalog walk threads through every group it visits.
///
/// A group stores its children either as `Link` messages in its object header
/// (with dense overflow in a fractal heap) or in the legacy symbol-table
/// B-tree plus local heap — and *which* it uses is a property of that group
/// alone. One file mixes the two freely: writing a single link that the old
/// format cannot express (an external link, a creation-order-tracked group)
/// migrates just that group, leaving its parent and its children where they
/// were. Walking a group in the format its *parent* used therefore finds no
/// children at all and reports an empty group, which is why the two storages
/// share one walker here: [`CatalogWalk::group`] asks each object header what
/// it declares, and [`CatalogWalk::child`] is the single place a child is
/// classified, recorded and descended into.
struct CatalogWalk<'a> {
    handle: &'a mut FileHandle,
    meta: &'a FileMeta,
    catalog: Catalog,
    /// Object headers already descended into, keyed to the first path that
    /// reached them: a later path to the same header is a group hard link,
    /// recorded in `group_aliases` so lookups resolve through it instead of
    /// walking (and cycling) a second time.
    visited: std::collections::HashMap<u64, String>,
}

impl<'a> CatalogWalk<'a> {
    /// Bound group nesting on a hostile or corrupt file.
    const MAX_DEPTH: usize = 256;

    /// Start a walk at the root group's object header address, seeded so a
    /// hard link cycling back to the root is not descended into again.
    fn new(handle: &'a mut FileHandle, meta: &'a FileMeta, root_addr: u64) -> Self {
        let mut visited = std::collections::HashMap::new();
        visited.insert(root_addr, String::new());
        Self {
            handle,
            meta,
            catalog: Catalog::default(),
            visited,
        }
    }

    /// The address/length widths, which most of the walk needs and `meta`
    /// carries alongside the file's B-tree ranks and shared-message table.
    fn ctx(&self) -> &FormatContext {
        &self.meta.ctx
    }

    fn finish(mut self) -> Catalog {
        self.catalog.group_object_paths = self.visited;
        self.catalog
    }

    /// Enumerate one group's children, choosing the storage from what this
    /// group's own header declares.
    ///
    /// `stab` is the symbol-table scratch-pad copy from the entry that named
    /// this group (the superblock's root entry, or the parent's symbol-table
    /// entry), which is the only source of those addresses when the header
    /// itself did not decode. `header` is `None` in exactly that case.
    fn group(
        &mut self,
        header: Option<&ObjectHeader>,
        prefix: &str,
        depth: usize,
        stab: Option<(u64, u64)>,
    ) -> IoResult<()> {
        if depth > Self::MAX_DEPTH {
            return Ok(());
        }
        let link_storage = header.filter(|h| header_declares_link_storage(h));
        if let Some(h) = link_storage {
            return self.links(h, prefix, depth);
        }
        // Symbol-table storage: the scratch-pad copy wins when it is set,
        // otherwise the addresses come from the group's own `stab` message.
        let (btree_addr, heap_addr) = match stab {
            Some(pair) if pair.0 != UNDEF_ADDR && pair.1 != UNDEF_ADDR => pair,
            _ => header.map_or((UNDEF_ADDR, UNDEF_ADDR), |h| {
                Hdf5Reader::stab_from_header(h, self.ctx())
            }),
        };
        if btree_addr != UNDEF_ADDR && heap_addr != UNDEF_ADDR {
            self.btree(btree_addr, heap_addr, prefix, depth)?;
        }
        Ok(())
    }

    /// Enumerate a group that stores its children as `Link` messages.
    fn links(&mut self, header: &ObjectHeader, prefix: &str, depth: usize) -> IoResult<()> {
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
                let (link, _) = LinkMessage::decode(&msg.data, self.ctx())?;
                links.push(link);
            } else if msg.msg_type == MSG_LINK_INFO {
                let (info, _) = LinkInfoMessage::decode(&msg.data, self.ctx())?;
                if info.fractal_heap_address != UNDEF_ADDR {
                    let ctx = self.meta.ctx;
                    let dense =
                        Hdf5Reader::read_dense_links(self.handle, &ctx, info.fractal_heap_address)?;
                    links.extend(dense);
                }
            }
        }

        for link in &links {
            let full_name = join_path(prefix, &link.name);
            // Every link is a listing entry whatever it points at; only a
            // hard link names an object in this file to descend into.
            self.catalog
                .links
                .insert(full_name.clone(), LinkClass::from_target(&link.target));
            let LinkTarget::Hard { address } = &link.target else {
                continue;
            };
            self.child(full_name, *address, depth, None)?;
        }
        Ok(())
    }

    /// Enumerate a group that stores its children in a symbol-table B-tree
    /// plus local heap.
    fn btree(
        &mut self,
        btree_addr: u64,
        heap_addr: u64,
        prefix: &str,
        depth: usize,
    ) -> IoResult<()> {
        let sa = self.ctx().sizeof_addr as usize;
        let ss = self.ctx().sizeof_size as usize;

        // Read the local heap header + data for this group.
        let heap_hdr_buf = self.handle.read_at_most(heap_addr, 64)?;
        let heap_hdr = LocalHeapHeader::decode(&heap_hdr_buf, sa, ss)?;
        let heap_data = self
            .handle
            .read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)?;

        // Collect all SNOD addresses by walking the B-tree.
        let mut snod_tree_visited = std::collections::HashSet::new();
        let snod_addrs = Hdf5Reader::collect_snod_addresses(
            self.handle,
            self.meta,
            btree_addr,
            0,
            &mut snod_tree_visited,
        )?;

        // A symbol-table node is a fixed-size record sized by `sym_leaf_k`,
        // which the superblock extension may have overridden.
        let snod_size = self.meta.btree.symbol_table_node_size(sa, ss);
        for snod_addr in snod_addrs {
            let snod_buf = self.handle.read_at_most(snod_addr, snod_size)?;
            let snod =
                SymbolTableNode::decode(&snod_buf, sa, ss, self.meta.btree.sym_leaf_max_entries())?;

            for entry in &snod.entries {
                let name = local_heap_get_string(&heap_data, entry.name_offset)?;
                // Skip empty names (root group self-reference).
                if name.is_empty() {
                    continue;
                }
                let full_name = join_path(prefix, &name);

                // A `H5G_CACHED_SLINK` entry is a soft link: it names no
                // object at all, and its value string lives in this group's
                // local heap. Record the link and move on — reading its
                // undefined object-header address is what used to drop it.
                if let SymbolTableCache::SoftLink { value_offset } = entry.cache {
                    let target = local_heap_get_string(&heap_data, value_offset as u64)?;
                    self.catalog
                        .links
                        .insert(full_name, LinkClass::Soft { path: target });
                    continue;
                }
                self.catalog
                    .links
                    .insert(full_name.clone(), LinkClass::Hard);
                self.child(
                    full_name,
                    entry.obj_header_addr,
                    depth,
                    entry.cached_symbol_table(),
                )?;
            }
        }

        Ok(())
    }

    /// Record one child of a group and, when it is itself a group, descend.
    ///
    /// Both storages end here, so a child is classified, catalogued and
    /// cycle-guarded the same way whichever way its name was found.
    fn child(
        &mut self,
        full_name: String,
        addr: u64,
        depth: usize,
        stab: Option<(u64, u64)>,
    ) -> IoResult<()> {
        // The entry names an object, so the object is in the listing whatever
        // comes of reading it: a header that does not decode (a stale link
        // left by a deletion, say) is reported against this name, never
        // dropped from it.
        let header = match Hdf5Reader::read_object_header_full(self.handle, self.meta, addr) {
            Ok(h) => h,
            Err(e) => {
                self.catalog
                    .unreadable
                    .insert(full_name, format!("its object header does not decode: {e}"));
                return Ok(());
            }
        };
        match Hdf5Reader::classify_object(self.handle, &header, self.meta, &full_name, addr) {
            ObjectKind::Dataset(info) => {
                self.catalog.datasets.push(*info);
                return Ok(());
            }
            ObjectKind::UnreadableDataset(why) => {
                self.catalog.unreadable.insert(full_name, why);
                return Ok(());
            }
            // A committed (named) datatype is neither a group nor a dataset,
            // so it must not be recorded as either; the link record above
            // already carries its name.
            ObjectKind::CommittedDatatype(info) => {
                self.catalog.datatypes.insert(full_name, *info);
                return Ok(());
            }
            ObjectKind::Group => {}
        }

        // It is a group. Record its path from the actual link record — before
        // the cycle check, so a hard-link alias of an already-visited group
        // still appears — whether or not it holds datasets or attributes.
        self.catalog.group_paths.insert(full_name.clone());
        // Capture group attributes (e.g. the NeXus `NX_class` marker), keyed
        // by path. Through the shared collector, which carries a per-attribute
        // failure as an entry naming it: a group whose attribute did not
        // decode must not come back as a group with one fewer attribute.
        // Recorded unconditionally, even for a group with none: the entry
        // also carries the header's own creation-order and storage facts,
        // which exist whether or not the group currently has any attributes.
        let ctx = self.meta.ctx;
        let attrs = collect_object_attributes(self.handle, &ctx, &header);
        self.catalog
            .group_attributes
            .insert(full_name.clone(), attrs);
        // Same unconditional recording for link storage and link
        // creation-order: this group's own header answers both whether or
        // not it descends any further.
        self.catalog.group_link_storage.insert(
            full_name.clone(),
            describe_link_storage(Some(&header), &ctx, stab),
        );

        // Descend at most once per object header (cycle guard); a second path
        // to it is a group hard link — record the alias for lookups instead.
        if let Some(first) = self.visited.get(&addr) {
            let first = first.clone();
            self.catalog.group_aliases.insert(full_name, first);
            return Ok(());
        }
        self.visited.insert(addr, full_name.clone());
        self.group(Some(&header), &full_name, depth + 1, stab)
    }
}

/// Join a group path prefix and a child name, with no leading `/` on a
/// root-level name.
fn join_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", prefix, name)
    }
}

/// Whether an object header declares link-message storage (compact or
/// dense) rather than the legacy symbol-table format — `Walk::group`'s own
/// dispatch predicate, factored out so [`describe_link_storage`] answers the
/// same question by construction rather than by keeping two checks in sync.
fn header_declares_link_storage(header: &ObjectHeader) -> bool {
    header
        .messages
        .iter()
        .any(|m| m.msg_type == MSG_LINK || m.msg_type == MSG_LINK_INFO)
}

/// A group's own link storage kind and link creation-order policy — h5py's
/// `link_storage_str` and `get_link_creation_order()`, computed together
/// because both read the same `Link Info` message.
///
/// `stab` is the symbol-table scratch-pad copy from the entry that named
/// this group, exactly as [`CatalogWalk::group`] takes it; `header` is
/// `None` only when the object header itself did not decode.
fn describe_link_storage(
    header: Option<&ObjectHeader>,
    ctx: &FormatContext,
    stab: Option<(u64, u64)>,
) -> (LinkStorage, CreationOrder) {
    if let Some(h) = header.filter(|h| header_declares_link_storage(h)) {
        // Link-message storage: the `Link Info` message, when present, gives
        // both facts at once. Its absence means compact and untracked — no
        // message exists to carry a creation index in.
        return h
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_LINK_INFO)
            .and_then(|m| LinkInfoMessage::decode(&m.data, ctx).ok())
            .map(|(info, _)| {
                let storage = if info.is_dense() {
                    LinkStorage::Dense
                } else {
                    LinkStorage::Compact
                };
                (storage, info.creation_order())
            })
            .unwrap_or((LinkStorage::Compact, CreationOrder::Untracked));
    }
    // No link-message storage in the header (or no header to check at all):
    // symbol-table format when the scratch-pad or the header's own `Symbol
    // Table` message resolves an address pair. Creation order is always
    // untracked here — the pre-1.8 format predates the feature, and 1.8+
    // never tracks creation order without also converting to link storage.
    let (btree_addr, heap_addr) = match stab {
        Some(pair) if pair.0 != UNDEF_ADDR && pair.1 != UNDEF_ADDR => pair,
        _ => header.map_or((UNDEF_ADDR, UNDEF_ADDR), |h| {
            Hdf5Reader::stab_from_header(h, ctx)
        }),
    };
    let storage = if btree_addr != UNDEF_ADDR && heap_addr != UNDEF_ADDR {
        LinkStorage::SymbolTable
    } else {
        // Neither link-message nor symbol-table storage is declared — an
        // object header this crate could not fully account for. Nothing in
        // the 92-case oracle suite reaches this path; it exists so an
        // unreadable root group still answers rather than panicking.
        LinkStorage::Compact
    };
    (storage, CreationOrder::Untracked)
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
    CommittedDatatype(Box<CommittedDatatypeInfo>),
}

/// Whether a header's messages say it is a committed (named) datatype: a
/// datatype message, no group storage, and not the dataspace/layout pair a
/// dataset needs.
///
/// The one authority for that question. [`Hdf5Reader::classify_object`] asks it
/// of a file being read and `ReopenWalk::plan` of one being appended to, and a
/// header that is a named datatype to one and an unclassifiable object to the
/// other is how `named_datatype_names` came to answer differently in the two
/// modes for the same file.
pub(crate) fn header_is_committed_datatype(header: &ObjectHeader) -> bool {
    let present = |t: u8| header.messages.iter().any(|m| m.msg_type == t);
    let is_group = present(MSG_LINK)
        || present(MSG_LINK_INFO)
        || present(MSG_SYMBOL_TABLE)
        || present(MSG_GROUP_INFO);
    !is_group && present(MSG_DATATYPE) && !(present(MSG_DATASPACE) && present(MSG_DATA_LAYOUT))
}

/// A committed (named) datatype as read from its own object header.
///
/// `H5Tcommit` gives a type a name and a place in the file; every dataset and
/// attribute built on it then stores a reference to this object rather than a
/// copy of the type. It is a third kind of object beside groups and datasets,
/// and classifying it as neither is what left its name in the file with
/// nothing behind it.
#[derive(Debug, Clone)]
pub struct CommittedDatatypeInfo {
    /// The type this object commits, or what stopped it from decoding. The
    /// object is in the listing either way, exactly as an unreadable dataset
    /// is: the name is in the file whether or not this crate can read what it
    /// names.
    datatype: Result<DatatypeMessage, String>,
    /// Attributes attached to the committed datatype itself.
    attributes: Vec<AttributeMessage>,
}

impl CommittedDatatypeInfo {
    /// The committed type, or the reason it cannot be read.
    pub fn datatype(&self) -> Result<&DatatypeMessage, &str> {
        self.datatype.as_ref().map_err(String::as_str)
    }

    /// The attributes attached to the committed datatype.
    pub fn attributes(&self) -> &[AttributeMessage] {
        &self.attributes
    }
}

/// Everything the superblock extension object header contributes to the
/// file-level view.
///
/// `H5Fsuper.c::H5F__super_read` opens this header immediately after decoding
/// the superblock, before any user object is reachable, so every message here
/// is in force for the first metadata decode that follows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SuperblockExtension {
    /// Shared Message Table message (0x000F): where the SOHM master table is.
    pub shared_message_table: Option<SharedMessageTableMessage>,
    /// v1 B-tree "K" values message (0x0013): non-default split ranks.
    pub btree_k: Option<BtreeKMessage>,
    /// Driver info message (0x0014).
    pub driver_info: Option<DriverInfoMessage>,
    /// File space info message (0x0017): allocation strategy, page size, and
    /// the persisted free-space manager addresses.
    pub file_space_info: Option<FileSpaceInfoMessage>,
}

/// HDF5 file reader.
pub struct Hdf5Reader {
    handle: FileHandle,
    meta: FileMeta,
    /// Messages read from the superblock extension object header, empty when
    /// the file has no extension.
    ext: SuperblockExtension,
    /// End-of-file address from the superblock.
    _eof: u64,
    /// Superblock format version (0-3), decoded once at open time by
    /// `detect_superblock_version` and never re-derived: 0/1 is the legacy
    /// symbol-table root, 2/3 the link-message root.
    superblock_version: u8,
    datasets: Vec<DatasetReadInfo>,
    /// Dataset-shaped objects this crate cannot read, keyed by path (no
    /// leading `/`), the value naming what stopped it. They are listed with
    /// the readable datasets and refuse typed access with that reason: an
    /// object the file contains is never reported as one it does not.
    unreadable: std::collections::BTreeMap<String, String>,
    /// Attributes on the root group (file-level attributes).
    root_attributes: ObjectAttributes,
    /// Link storage kind and link creation-order policy of the root group.
    root_link_storage: (LinkStorage, CreationOrder),
    /// Attributes on non-root groups, keyed by group path (no leading `/`).
    group_attributes: std::collections::HashMap<String, ObjectAttributes>,
    /// Link storage kind and link creation-order policy of non-root groups,
    /// keyed by group path (no leading `/`).
    group_link_storage: std::collections::HashMap<String, (LinkStorage, CreationOrder)>,
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
    /// The path this file was opened with. An external link resolves its
    /// target relative to the directory holding it (libhdf5 keeps the same
    /// thing as `H5F_EXTPATH`), so the reader has to remember where it came
    /// from.
    path: PathBuf,
    /// The directory holding this HDF5 file, resolved once at open time.
    /// External raw-data files (H5O_EFL_ID) are named relative to it when
    /// `HDF5_EXTFILE_PREFIX` contains `${ORIGIN}` (`H5D__build_file_prefix`,
    /// H5Dint.c) — captured at open time rather than re-derived from the
    /// process's current directory at read time, matching libhdf5's own
    /// one-time capture in `H5F_t::extpath`.
    source_dir: PathBuf,
    /// The locking policy this file was opened under, reused verbatim for
    /// every external target: libhdf5 hands `H5F_prefix_open_file` the
    /// parent's file-access property list, so one `HDF5_USE_FILE_LOCKING`
    /// setting (or one `H5FileOptions::locking` call) governs every file a
    /// path touches, not just the first.
    locking: crate::io::locking::FileLocking,
    /// External-link target files, keyed by the resolved path that opened
    /// them, so N links naming one file share one open handle.
    ///
    /// An entry is created the first time a path actually crosses the link and
    /// lives until this reader is dropped — that is, for the life of the
    /// `H5File` that owns it. A target's own external links are cached in that
    /// target's map, so the first reader in a chain transitively holds the
    /// whole chain open.
    external: std::collections::BTreeMap<PathBuf, Box<Hdf5Reader>>,
    /// What each external link's stored file name resolved to, keyed by that
    /// name exactly as the link holds it.
    ///
    /// The search runs once per distinct name per reader and its answer is
    /// then fixed: re-probing the filesystem on a later crossing would let one
    /// link answer differently mid-session, and would fail to find the handle
    /// it already holds once the target has been renamed or unlinked.
    external_resolved: std::collections::BTreeMap<String, PathBuf>,
    /// Committed (named) datatype objects, keyed by path (no leading `/`).
    datatypes: std::collections::BTreeMap<String, CommittedDatatypeInfo>,
    /// Object header address → absolute path, for every group and dataset the
    /// discovery walk reached plus the root group. This is what turns the
    /// address an object reference stores back into a name.
    object_paths: std::collections::HashMap<u64, String>,
    /// The dataset-access properties in force for each virtual dataset,
    /// keyed by canonical path (no leading `/`); an absent entry means
    /// [`DatasetAccess::default`].
    ///
    /// libhdf5 keeps this per open handle — `H5D__virtual_init` copies the
    /// dapl into the dataset's own layout storage (H5Dvirtual.c:2224-2226) —
    /// but this reader keeps one resolved extent per dataset, so the entry
    /// here is what the most recent [`open_dataset_with`](Self::open_dataset_with)
    /// asked for. It is the single owner of that answer: the resolution
    /// reads it and nothing else writes it, so a SWMR
    /// [`refresh`](Self::refresh) re-resolves under the same properties
    /// rather than reverting to the defaults.
    virtual_access: std::collections::BTreeMap<String, DatasetAccess>,
}

/// The absolute form of a discovery-walk path (which carries no leading `/`):
/// the root group's empty path becomes `/`, `entry/data` becomes
/// `/entry/data`.
fn absolute_path(path: &str) -> String {
    format!("/{}", path.trim_start_matches('/'))
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

/// A fixed-length string attribute's value, under the padding rule its
/// datatype declares.
///
/// The single owner for the attribute side of that rule: both
/// [`H5Reader::attr_string_value`] and the writer-mode fallback in
/// `H5Attribute::read_string` end here, so a space-padded attribute does not
/// read back with its padding attached on one path and not the other. Bytes
/// that are not valid UTF-8 become U+FFFD, as they always have on this path.
///
/// A datatype that is not a string at all is read as null-terminated, which is
/// what asking for the string value of, say, an integer attribute has always
/// meant here.
pub(crate) fn fixed_string_attr_value(attr: &AttributeMessage) -> IoResult<String> {
    use crate::format::messages::datatype::{fixed_string_content, DatatypeMessage};
    let padding = match attr.datatype {
        DatatypeMessage::FixedString { padding, .. } => padding,
        _ => 0,
    };
    let content = fixed_string_content(&attr.data, padding).ok_or_else(|| {
        crate::io::IoError::InvalidState(format!(
            "attribute {:?} uses string padding rule {padding}, which the format reserves",
            attr.name
        ))
    })?;
    Ok(String::from_utf8_lossy(content).to_string())
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

/// Resolve the directory a raw-data file name is joined against, matching
/// libhdf5's `H5D__build_file_prefix` (H5Dint.c) for the given environment
/// variable — `HDF5_EXTFILE_PREFIX` for External Data Files
/// ([`resolve_extfile_prefix`]), `HDF5_VDS_PREFIX` for Virtual Dataset
/// sources ([`resolve_vdsfile_prefix`]); both features route through the
/// same C function, just keyed on a different variable. There is no dataset
/// access property list here to fall back to, so unset behaves exactly as
/// an unset/empty DAPL property would.
///
/// `${ORIGIN}` expands to `source_dir` (the directory holding the open
/// HDF5 file); any other value is used as a literal prefix; unset, empty,
/// or `"."` means "no prefix" (`H5_combine_path`'s own default), so a
/// relative name resolves against the process's current directory instead.
fn resolve_file_prefix(env_var: &str, source_dir: &Path) -> Option<PathBuf> {
    let prefix = std::env::var(env_var).ok()?;
    if prefix.is_empty() || prefix == "." {
        return None;
    }
    Some(match prefix.strip_prefix("${ORIGIN}") {
        Some(rest) => {
            let rest = rest.trim_start_matches(['/', '\\']);
            if rest.is_empty() {
                source_dir.to_path_buf()
            } else {
                source_dir.join(rest)
            }
        }
        None => PathBuf::from(prefix),
    })
}

pub(crate) fn resolve_extfile_prefix(source_dir: &Path) -> Option<PathBuf> {
    resolve_file_prefix("HDF5_EXTFILE_PREFIX", source_dir)
}

/// Resolve the directory Virtual Dataset source file names are joined
/// against (`HDF5_VDS_PREFIX`; see [`resolve_file_prefix`]).
fn resolve_vdsfile_prefix(source_dir: &Path) -> Option<PathBuf> {
    resolve_file_prefix("HDF5_VDS_PREFIX", source_dir)
}

/// Join a raw-data file `name` against a resolved prefix, matching
/// libhdf5's `H5_combine_path` (H5system.c): an absolute `name` is used
/// as-is regardless of the prefix, and no prefix means "relative to the
/// process's current directory" — both of which `Path::join` already
/// implements for an absolute joinee. Shared by External Data Files and
/// Virtual Dataset source resolution — both call the same C function.
pub(crate) fn combine_prefixed_path(prefix: Option<&Path>, name: &str) -> PathBuf {
    match prefix {
        Some(p) => p.join(name),
        None => PathBuf::from(name),
    }
}

/// Read `len` bytes starting at *dataset-relative* offset `skip` from an
/// external file list into `out`, walking slots by cumulative declared
/// size exactly like libhdf5's `H5D__efl_read` (H5Defl.c). A read past an
/// individual slot's actual on-disk length reads back as zero — the file
/// backing a slot may be shorter than the space the layout reserved in
/// it — but a read past the *total* declared size of the file list is
/// still an error, matching `H5D__efl_read`'s own "read past logical end
/// of file" check.
///
/// An `H5O_EFL_UNLIMITED` last slot needs no special case: the walk below
/// never steps past it (`skip >= u64::MAX` is never true), which is upstream's
/// `H5O_EFL_UNLIMITED == size || addr < cur + size`, and the read it then
/// takes is the whole remainder, bounded by whatever the file physically
/// holds.
fn read_external_file_bytes(
    external_files: &[ExternalFileSegment],
    extfile_prefix: Option<&Path>,
    mut skip: u64,
    out: &mut [u8],
) -> IoResult<()> {
    let mut slot_idx = 0usize;
    while slot_idx < external_files.len() && skip >= external_files[slot_idx].size {
        skip -= external_files[slot_idx].size;
        slot_idx += 1;
    }

    let mut written = 0usize;
    while written < out.len() {
        let Some(slot) = external_files.get(slot_idx) else {
            return Err(crate::io::IoError::InvalidState(
                "read past the logical end of the external file list".into(),
            ));
        };
        let full_path = combine_prefixed_path(extfile_prefix, &slot.name);
        let ext_handle = FileHandle::open_read_with_locking(&full_path, FileLocking::Disabled)
            .map_err(|e| {
                crate::io::IoError::InvalidState(format!(
                    "unable to open external raw data file {}: {e}",
                    full_path.display()
                ))
            })?;
        let avail_in_slot = slot.size.saturating_sub(skip);
        let want = (out.len() - written) as u64;
        let this_read = avail_in_slot.min(want) as usize;
        let dst = &mut out[written..written + this_read];
        // A short physical file — the reserved slot size exceeds what was
        // ever actually written to it — reads back as zero for the
        // remainder, exactly like `H5D__efl_read`.
        let got = ext_handle.read_at_most(slot.offset + skip, this_read)?;
        dst[..got.len()].copy_from_slice(&got);
        dst[got.len()..].fill(0);

        written += this_read;
        skip = 0;
        slot_idx += 1;
    }
    Ok(())
}

/// Recursion ceiling for virtual dataset nesting (a VDS whose source is
/// itself a VDS — possibly in another file). Bounded so a crafted cyclic
/// mapping chain fails cleanly instead of recursing until the stack
/// overflows; real VDS chains do not nest anywhere near this deep.
const MAX_VIRTUAL_DEPTH: usize = 16;

/// Replace every unlimited mapping with the concrete mapping its open-time
/// resolution makes it — the clipped selections `H5D__virtual_set_extent_unlim`
/// leaves in `clipped_virtual_select` / `clipped_source_select` for a read to
/// use (`H5D__virtual_read` never sees the unclipped ones).
///
/// A mapping with no resolution recorded is passed through unchanged, which is
/// what a bounded mapping needs and what a mapping list written before this
/// pass existed reduces to.
fn concrete_virtual_mappings(
    list: &VirtualMappingList,
    resolution: &[MappingResolution],
) -> IoResult<Vec<VirtualMapping>> {
    let mut out = Vec::with_capacity(list.mappings.len());
    for (i, m) in list.mappings.iter().enumerate() {
        match resolution.get(i) {
            Some(MappingResolution::Unlimited {
                virtual_clip,
                source_clip,
            }) => out.push(VirtualMapping {
                virtual_selection: m.virtual_selection.clip_unlimited(*virtual_clip)?,
                source_selection: m.source_selection.clip_unlimited(*source_clip)?,
                ..built_names(m, 0)?
            }),
            // One printf mapping is a whole family: block `j` of the virtual
            // selection (`H5S_hyper_get_unlim_block`) is filled by the source
            // dataset whose name substitutes `j`, taking the mapping's whole
            // (limited) source selection. Only the blocks that have a source
            // become mappings — a non-zero printf gap leaves the others
            // inside the extent, reading as the fill value.
            Some(MappingResolution::Printf { present, .. }) => {
                let Some(r) = regular_hyperslab(&m.virtual_selection) else {
                    continue;
                };
                let rank = r.start.len();
                for &j in present {
                    out.push(VirtualMapping {
                        virtual_selection: Selection::Hyperslab {
                            rank,
                            form: Hyperslab::Regular(r.unlim_block(j)),
                        },
                        ..built_names(m, j)?
                    });
                }
            }
            _ => out.push(built_names(m, 0)?),
        }
    }
    Ok(out)
}

/// One mapping with both source names built for block `blockno` —
/// `H5D__virtual_build_source_name`. A mapping with no substitutions still
/// goes through this, because that is where `%%` is unescaped: upstream uses
/// the parsed name rather than the stored one for an ordinary mapping too
/// (`H5D__virtual_load_layout`).
fn built_names(m: &VirtualMapping, blockno: u64) -> IoResult<VirtualMapping> {
    Ok(VirtualMapping {
        source_file_name: parse_source_name(&m.source_file_name)?.build(blockno),
        source_dset_name: parse_source_name(&m.source_dset_name)?.build(blockno),
        ..m.clone()
    })
}

/// Recursion bound for open-time virtual-extent resolution.
///
/// Resolving a virtual dataset's extent opens the source datasets its
/// unlimited mappings name, and a source may itself be a virtual dataset in
/// another file whose own open resolves its own extent. A crafted cyclic
/// chain would otherwise recurse until the stack overflows, so the nesting is
/// counted per thread and the resolution is skipped once it reaches
/// [`MAX_VIRTUAL_DEPTH`] — a dataset that deep keeps its stored extent
/// instead of taking one from a cycle.
struct VirtualResolveDepth;

thread_local! {
    static VIRTUAL_RESOLVE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl VirtualResolveDepth {
    fn enter<F: FnOnce() -> IoResult<()>>(f: F) -> IoResult<()> {
        let depth = VIRTUAL_RESOLVE_DEPTH.with(std::cell::Cell::get);
        if depth >= MAX_VIRTUAL_DEPTH {
            return Ok(());
        }
        VIRTUAL_RESOLVE_DEPTH.with(|d| d.set(depth + 1));
        let out = f();
        VIRTUAL_RESOLVE_DEPTH.with(|d| d.set(depth));
        out
    }
}

/// The regular (start, stride, count, block) form behind a selection, or
/// `None` — the only form `H5S_UNLIMITED` can appear in, so every unlimited
/// computation goes through it.
fn regular_hyperslab(sel: &Selection) -> Option<&RegularHyperslab> {
    match sel {
        Selection::Hyperslab {
            form: Hyperslab::Regular(r),
            ..
        } => Some(r),
        _ => None,
    }
}

/// Read each of `source_boxes` (via `read_source_box`) and scatter it into
/// `out` (shaped `virtual_dims`) at the same-indexed box in
/// `virtual_boxes`.
///
/// The two box lists must have equal length, and each pair of boxes must be
/// the same shape under `H5S_select_shape_same` (H5Sselect.c) — which
/// compares the two from the fast end and requires the extra *leading*
/// dimensions of the higher-rank side to be flat, so a rank-1 source box
/// legitimately fills a rank-2 virtual box of the same trailing shape (the
/// usual printf-mapping arrangement: one 1-D dataset per row of a 2-D
/// virtual dataset). A mapping whose selections diverge beyond that — same
/// point count but a different box decomposition on each side — would need
/// the general element-by-element linear-order match H5S's own iterator
/// does; rather than risk a silently wrong scatter, that case is rejected
/// here.
/// Whether two boxes select the same elements in the same order —
/// `H5S_select_shape_same`'s rule (H5Sselect.c), which aligns the two shapes
/// at their fast end and requires every extra leading dimension of the
/// higher-rank side to be flat.
fn shape_same(a: &[u64], b: &[u64]) -> bool {
    let common = a.len().min(b.len());
    a[a.len() - common..] == b[b.len() - common..]
        && a[..a.len() - common].iter().all(|&d| d == 1)
        && b[..b.len() - common].iter().all(|&d| d == 1)
}

fn copy_matched_boxes(
    mut read_source_box: impl FnMut(&[u64], &[u64], &mut [u8]) -> IoResult<()>,
    source_boxes: &[(Vec<u64>, Vec<u64>)],
    virtual_boxes: &[(Vec<u64>, Vec<u64>)],
    virtual_dims: &[u64],
    element_size: u64,
    out: &mut [u8],
) -> IoResult<()> {
    if source_boxes.len() != virtual_boxes.len() {
        return Err(crate::io::IoError::InvalidState(format!(
            "virtual dataset mapping's source and virtual selections decompose into a \
             different number of boxes ({} vs {}), which is not supported",
            source_boxes.len(),
            virtual_boxes.len()
        )));
    }
    for ((src_start, src_count), (dst_start, dst_count)) in source_boxes.iter().zip(virtual_boxes) {
        if !shape_same(src_count, dst_count) {
            return Err(crate::io::IoError::InvalidState(format!(
                "virtual dataset mapping's source box shape {src_count:?} does not match its \
                 virtual box shape {dst_count:?}, which is not supported"
            )));
        }
        let nbytes = saturating_byte_len(src_count, element_size) as usize;
        let mut buf = alloc_tiled_fill(nbytes, None)?;
        read_source_box(src_start, src_count, &mut buf)?;

        // `buf` holds the box in source linear order, and shape-same boxes
        // differ only by flat leading dimensions, so it can be walked at the
        // virtual box's own rank.
        let src_origin = vec![0u64; dst_count.len()];
        for_each_dual_run(
            virtual_dims,
            dst_start,
            dst_count,
            &src_origin,
            dst_count,
            element_size,
            |dst_off, src_off, len| {
                let dst_off = dst_off as usize;
                let src_off = src_off as usize;
                out[dst_off..dst_off + len].copy_from_slice(&buf[src_off..src_off + len]);
                Ok(())
            },
        )?;
    }
    Ok(())
}

/// Read and decode the global-heap collection at `addr`, applying the
/// validation of libhdf5's `H5HG__cache_heap_deserialize`: the `GCOL`
/// signature must be present and the declared size at least `H5HG_MINSIZE`
/// (4096 bytes). There is no upper size cap — libhdf5 has none, and this
/// crate's writers put a whole write call's strings into one collection,
/// which a cap would turn into silent data loss.
///
/// A free function (not a method) so both [`Hdf5Reader::read_heap_collection`]
/// and the static dataset-open path (which only has a `&mut FileHandle`, not
/// a full `&mut Hdf5Reader`) share the one implementation.
fn read_heap_collection_from(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    addr: u64,
) -> IoResult<GlobalHeapCollection> {
    let ss = ctx.sizeof_size as usize;
    let header_len = 4 + 1 + 3 + ss;
    let header_buf = handle.read_at_most(addr, header_len)?;
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
    let heap_buf = handle.read_at(addr, declared)?;
    let (coll, _) = GlobalHeapCollection::decode(&heap_buf, ctx)?;
    Ok(coll)
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
        // Also open an mmap handle for zero-copy data access, in the same
        // address space the reader located the superblock in — otherwise every
        // address read through the mmap would be short by the userblock.
        let mut mmap = MmapFileHandle::open(path)?;
        mmap.set_base(reader.userblock_size());
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

        // The superblock is not necessarily at the start of the file: a
        // userblock precedes it, and `H5FD_locate_signature` finds it by
        // probing offset 0 and then every power of two from 512 up. The offset
        // it is found at is where HDF5 addresses are measured from, so it
        // becomes the handle's base address and every later offset — including
        // the superblock read just below — is relative to it.
        let super_addr = handle
            .locate_signature()?
            .ok_or(crate::format::FormatError::InvalidSignature)?;
        handle.set_base(super_addr);

        // Read enough bytes to detect the superblock version and parse it.
        let sb_buf = handle.read_at_most(0, 1024)?;
        let version = detect_superblock_version(&sb_buf)?;

        let origin = Origin {
            path: path.to_path_buf(),
            locking,
        };
        let mut reader = match version {
            0 | 1 => Self::open_v0v1(handle, &sb_buf, origin)?,
            2 | 3 => Self::open_v2v3(handle, &sb_buf, origin)?,
            v => {
                return Err(crate::io::IoError::Format(
                    crate::format::FormatError::InvalidVersion(v),
                ))
            }
        };
        // Resolved from the path this file was opened with (not the
        // process's current directory at read time) — see `source_dir`.
        let canonical = std::fs::canonicalize(path)?;
        reader.source_dir = canonical
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        // Only now, with `source_dir` set, can a source name be resolved —
        // and a virtual dataset's extent is not final until they are.
        VirtualResolveDepth::enter(|| reader.resolve_virtual_extents())?;
        Ok(reader)
    }

    /// Open a file with v2/v3 superblock (existing code path).
    fn open_v2v3(mut handle: FileHandle, sb_buf: &[u8], origin: Origin) -> IoResult<Self> {
        let sb = SuperblockV2V3::decode(sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // A v2/v3 superblock has no room for the B-tree K values, so they are
        // the library defaults unless the extension carries the K message.
        let (meta, ext) = Self::read_extension_and_meta(
            &mut handle,
            ctx,
            BTreeV1Config::default(),
            sb.superblock_extension_address,
        )?;

        // Read root group object header, following continuation blocks.
        let root_header =
            Self::read_object_header_full(&mut handle, &meta, sb.root_group_object_header_address)?;

        // Walk the root group to discover datasets, group attributes, and
        // every group path that exists, from whichever storage each group's
        // own header declares.
        let catalog = Self::build_catalog(
            &mut handle,
            &meta,
            Some(&root_header),
            sb.root_group_object_header_address,
            None,
        )?;

        // Collect root group attributes
        let root_attributes = collect_object_attributes(&mut handle, &ctx, &root_header);
        // A v2/v3 root group is always addressed directly by the superblock,
        // never through a symbol-table scratch-pad.
        let root_link_storage = describe_link_storage(Some(&root_header), &ctx, None);

        Ok(Self {
            handle,
            meta,
            ext,
            _eof: sb.end_of_file_address,
            superblock_version: sb.version,
            object_paths: catalog.object_paths(sb.root_group_object_header_address),
            datasets: catalog.datasets,
            unreadable: catalog.unreadable,
            root_attributes,
            root_link_storage,
            group_attributes: catalog.group_attributes,
            group_link_storage: catalog.group_link_storage,
            group_paths: catalog.group_paths,
            group_aliases: catalog.group_aliases,
            links: catalog.links,
            datatypes: catalog.datatypes,
            path: origin.path,
            locking: origin.locking,
            external: Default::default(),
            external_resolved: Default::default(),
            virtual_access: Default::default(),
            // Overwritten by `open_with_locking` once this returns.
            source_dir: PathBuf::new(),
        })
    }

    /// Open a file with v0/v1 superblock (legacy format).
    fn open_v0v1(mut handle: FileHandle, sb_buf: &[u8], origin: Origin) -> IoResult<Self> {
        let sb = SuperblockV0V1::decode(sb_buf)?;

        let ctx = FormatContext {
            sizeof_addr: sb.sizeof_offsets,
            sizeof_size: sb.sizeof_lengths,
        };

        // A v0/v1 superblock carries the K values itself; a v0 superblock has
        // no chunk-tree field, so that one keeps the library default. The
        // extension's K message, when present, overrides all three.
        let sb_btree = BTreeV1Config {
            sym_leaf_k: sb.sym_leaf_k,
            snode_internal_k: sb.btree_internal_k,
            chunk_internal_k: sb
                .indexed_storage_k
                .unwrap_or(BTreeV1Config::default().chunk_internal_k),
        };
        let (meta, ext) = Self::read_extension_and_meta(
            &mut handle,
            ctx,
            sb_btree,
            sb.superblock_extension_address,
        )?;

        let ste = &sb.root_symbol_table_entry;
        let root_obj_addr = ste.obj_header_addr;
        let ste_stab = ste.cached_symbol_table();

        // Read the root group's object header (following continuations).
        let root_hdr = Self::read_object_header_full(&mut handle, &meta, root_obj_addr).ok();

        // Collect the root group's own attributes.
        let root_attributes = match root_hdr {
            Some(ref h) => collect_object_attributes(&mut handle, &ctx, h),
            None => ObjectAttributes::default(),
        };
        // The root group's own link storage and link creation-order, from
        // the same header-first/scratch-pad-fallback rule the walk below
        // uses to choose how to enumerate it.
        let root_link_storage = describe_link_storage(root_hdr.as_ref(), &ctx, ste_stab);

        // A v0/v1-superblock file whose root group has migrated to link
        // storage (more than ~8 objects, or one link the old format cannot
        // express) carries `Link` / `Link Info` messages in its object
        // header, and the superblock symbol-table scratch-pad is then stale.
        // The walk picks the storage from the header for that reason, taking
        // the scratch-pad only as the symbol-table addresses — and only for a
        // symbol-table root group (`H5G_CACHED_STAB`), which is the one cache
        // type those two addresses mean anything for.
        let catalog = Self::build_catalog(
            &mut handle,
            &meta,
            root_hdr.as_ref(),
            root_obj_addr,
            ste_stab,
        )?;

        Ok(Self {
            handle,
            meta,
            ext,
            _eof: sb.end_of_file_address,
            superblock_version: sb.version,
            object_paths: catalog.object_paths(root_obj_addr),
            datasets: catalog.datasets,
            unreadable: catalog.unreadable,
            root_attributes,
            root_link_storage,
            group_attributes: catalog.group_attributes,
            group_link_storage: catalog.group_link_storage,
            group_paths: catalog.group_paths,
            group_aliases: catalog.group_aliases,
            links: catalog.links,
            datatypes: catalog.datatypes,
            path: origin.path,
            locking: origin.locking,
            external: Default::default(),
            external_resolved: Default::default(),
            virtual_access: Default::default(),
            // Overwritten by `open_with_locking` once this returns.
            source_dir: PathBuf::new(),
        })
    }

    /// Read the superblock extension object header at `ext_addr` (if any) and
    /// fold what it says into the file-level decode parameters.
    ///
    /// `sb_btree` is what the superblock alone implies; the extension's
    /// v1-B-tree-"K" message replaces all three ranks when present, exactly as
    /// `H5F__super_read` does after `H5O_msg_read(&ext_loc, H5O_BTREEK_ID)`.
    pub(crate) fn read_extension_and_meta(
        handle: &mut FileHandle,
        ctx: FormatContext,
        sb_btree: BTreeV1Config,
        ext_addr: u64,
    ) -> IoResult<(FileMeta, SuperblockExtension)> {
        let mut meta = FileMeta {
            ctx,
            btree: sb_btree,
            sohm: None,
        };
        let ext = Self::superblock_extension_at(handle, ctx, sb_btree, ext_addr)?;
        if let Some(k) = ext.btree_k {
            meta.btree = BTreeV1Config {
                sym_leaf_k: k.sym_leaf_k,
                snode_internal_k: k.snode_internal_k,
                chunk_internal_k: k.chunk_internal_k,
            };
        }
        // A zero rank would make every v1 B-tree node zero-sized and every
        // symbol-table node hold no entries; libhdf5 rejects it at creation
        // (`H5Pset_sym_k`, `H5Pset_istore_k`), so a file carrying one is
        // corrupt rather than merely unusual.
        let b = &meta.btree;
        if b.sym_leaf_k == 0 || b.snode_internal_k == 0 || b.chunk_internal_k == 0 {
            return Err(crate::io::IoError::Format(
                crate::format::FormatError::InvalidData(format!(
                    "v1 B-tree K values must be non-zero (sym_leaf={}, snode={}, chunk={})",
                    b.sym_leaf_k, b.snode_internal_k, b.chunk_internal_k
                )),
            ));
        }
        // `H5F__super_read` calls `H5SM_get_info` here, so the shared-message
        // table is in place before the root group — the first object header
        // that can hold a shared message — is opened.
        if let Some(smt) = &ext.shared_message_table {
            meta.sohm = Some(Self::read_sohm_table(handle, &meta.ctx, smt)?);
        }
        Ok((meta, ext))
    }

    /// The superblock extension's messages, for the address the superblock
    /// names. Yields the default (every field `None`) when there is no
    /// extension.
    ///
    /// The extension header is read with the pre-extension parameters: its own
    /// messages are never shared and never in a v1 B-tree, so nothing it
    /// contains is needed to decode it. This is also how the writer's append
    /// path learns what the file declares before it rewrites anything.
    pub(crate) fn superblock_extension_at(
        handle: &mut FileHandle,
        ctx: FormatContext,
        btree: BTreeV1Config,
        ext_addr: u64,
    ) -> IoResult<SuperblockExtension> {
        if ext_addr == UNDEF_ADDR || ext_addr == 0 {
            return Ok(SuperblockExtension::default());
        }
        let meta = FileMeta {
            ctx,
            btree,
            sohm: None,
        };
        Self::read_superblock_extension(handle, &meta, ext_addr)
    }

    /// Decode the messages of the superblock extension object header.
    ///
    /// Upstream reads each of these with `H5O_msg_exists` + `H5O_msg_read` and
    /// fails the open when one is present but undecodable; a message this
    /// crate does not model is skipped, as an unknown non-critical message is
    /// elsewhere.
    fn read_superblock_extension(
        handle: &mut FileHandle,
        meta: &FileMeta,
        addr: u64,
    ) -> IoResult<SuperblockExtension> {
        let header = Self::read_object_header_full(handle, meta, addr)?;
        let ctx = &meta.ctx;
        let mut ext = SuperblockExtension::default();
        for msg in &header.messages {
            match msg.msg_type {
                MSG_SHARED_MESSAGE_TABLE => {
                    ext.shared_message_table =
                        Some(SharedMessageTableMessage::decode(&msg.data, ctx)?);
                }
                MSG_BTREE_K => ext.btree_k = Some(BtreeKMessage::decode(&msg.data)?),
                MSG_DRIVER_INFO => ext.driver_info = Some(DriverInfoMessage::decode(&msg.data)?),
                MSG_FILE_SPACE_INFO => {
                    ext.file_space_info = Some(FileSpaceInfoMessage::decode(&msg.data, ctx)?);
                }
                _ => {}
            }
        }
        Ok(ext)
    }

    /// Read the SOHM master table named by the extension's shared-message
    /// table message.
    ///
    /// The table's length is not stored with it: the index count comes from
    /// the message, exactly as `H5SM__cache_table_get_final_load_size` takes it
    /// from `H5F_SOHM_NINDEXES`.
    fn read_sohm_table(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        smt: &SharedMessageTableMessage,
    ) -> IoResult<SohmMasterTable> {
        if smt.table_address == UNDEF_ADDR || smt.nindexes == 0 {
            return Ok(SohmMasterTable::default());
        }
        let size = SohmMasterTable::encoded_size(ctx, smt.nindexes);
        let buf = handle.read_at(smt.table_address, size)?;
        Ok(SohmMasterTable::decode(&buf, ctx, smt.nindexes)?)
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

    /// Build the file catalog for a whole file, starting at its root group.
    ///
    /// `root_header` is `None` only when the root object header did not
    /// decode; `root_stab` then carries the superblock symbol-table entry's
    /// cached B-tree and local heap, which is enough to list a legacy file.
    fn build_catalog(
        handle: &mut FileHandle,
        meta: &FileMeta,
        root_header: Option<&ObjectHeader>,
        root_addr: u64,
        root_stab: Option<(u64, u64)>,
    ) -> IoResult<Catalog> {
        let mut walk = CatalogWalk::new(handle, meta, root_addr);
        walk.group(root_header, "", 0, root_stab)?;
        Ok(walk.finish())
    }

    /// Read every link stored in a group's dense (fractal-heap) link storage.
    ///
    /// The `Link Info` message gives the fractal-heap address; each managed
    /// object in the heap is an encoded `Link` message. Returns the decoded
    /// links (hard and soft).
    pub(crate) fn read_dense_links(
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

    /// Recursively walk a B-tree v1 to collect leaf-level SNOD addresses.
    fn collect_snod_addresses(
        handle: &mut FileHandle,
        meta: &FileMeta,
        tree_addr: u64,
        depth: usize,
        visited: &mut std::collections::HashSet<u64>,
    ) -> IoResult<Vec<u64>> {
        let sizeof_addr = meta.ctx.sizeof_addr as usize;
        let sizeof_size = meta.ctx.sizeof_size as usize;
        // A well-formed v1 B-tree's level strictly decreases with depth;
        // bound the descent so a corrupt/cyclic tree cannot recurse forever.
        // The `visited` set additionally stops a corrupt tree whose child
        // points back at an ancestor node from fanning out exponentially.
        if depth > 256 || !visited.insert(tree_addr) {
            return Ok(Vec::new());
        }
        // A v1 B-tree node is a fixed-size record whose length follows from
        // the file's K values; reading exactly that much also bounds what a
        // corrupt address can pull in.
        let node_size = meta.btree.snode_btree_node_size(sizeof_addr, sizeof_size);
        let buf = handle.read_at_most(tree_addr, node_size)?;
        let node = BTreeV1Node::decode(
            &buf,
            sizeof_addr,
            sizeof_size,
            meta.btree.snode_max_entries(),
        )?;

        if node.level == 0 {
            // Leaf level: children are SNOD addresses
            Ok(node.children.clone())
        } else {
            // Internal level: children are sub-TREE addresses
            let mut addrs = Vec::new();
            for &child_addr in &node.children {
                let child_addrs =
                    Self::collect_snod_addresses(handle, meta, child_addr, depth + 1, visited)?;
                addrs.extend(child_addrs);
            }
            Ok(addrs)
        }
    }

    /// Read the object header at `addr` with every continuation block
    /// flattened in and every stored-shared message resolved to its literal
    /// body. One owner for both halves of the crate — see
    /// [`crate::io::object_header_io`].
    fn read_object_header_full(
        handle: &mut FileHandle,
        meta: &FileMeta,
        addr: u64,
    ) -> IoResult<ObjectHeader> {
        crate::io::object_header_io::read_object_header_full(handle, meta, addr)
    }

    /// Read a committed datatype object's type and attributes from its header.
    ///
    /// A committed datatype's own message holds the type itself, but the
    /// format does not forbid it being a reference in turn, so it goes
    /// through the same resolver every other datatype message does.
    fn committed_datatype(
        handle: &mut FileHandle,
        header: &ObjectHeader,
        meta: &FileMeta,
    ) -> CommittedDatatypeInfo {
        let datatype = header
            .messages
            .iter()
            .find(|m| m.msg_type == MSG_DATATYPE)
            .cloned()
            .ok_or_else(|| "it holds no datatype message".to_string())
            .and_then(|m| {
                crate::io::object_header_io::read_datatype_message(handle, meta, &m).map_err(|e| {
                    match e {
                        crate::io::IoError::Unsupported(why) => why,
                        other => format!("its datatype message does not decode: {other}"),
                    }
                })
            });
        let attributes = header
            .messages
            .iter()
            .filter(|m| m.msg_type == MSG_ATTRIBUTE && m.flags & MSG_FLAG_SHARED == 0)
            .filter_map(|m| {
                AttributeMessage::decode(&m.data, &meta.ctx)
                    .ok()
                    .map(|(a, _)| a)
            })
            .collect();
        CommittedDatatypeInfo {
            datatype,
            attributes,
        }
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
    fn classify_object(
        handle: &mut FileHandle,
        header: &ObjectHeader,
        meta: &FileMeta,
        name: &str,
        addr: u64,
    ) -> ObjectKind {
        let ctx = &meta.ctx;
        let present = |t: u8| header.messages.iter().any(|m| m.msg_type == t);
        let is_group = present(MSG_LINK)
            || present(MSG_LINK_INFO)
            || present(MSG_SYMBOL_TABLE)
            || present(MSG_GROUP_INFO);
        if is_group {
            return ObjectKind::Group;
        }
        if header_is_committed_datatype(header) {
            return ObjectKind::CommittedDatatype(Box::new(Self::committed_datatype(
                handle, header, meta,
            )));
        }
        let is_dataset =
            present(MSG_DATATYPE) && present(MSG_DATASPACE) && present(MSG_DATA_LAYOUT);
        if !is_dataset {
            return ObjectKind::Group;
        }

        let mut datatype = None;
        let mut dataspace = None;
        let mut layout = None;
        let mut filter_pipeline = None;
        let mut fill_value = None;
        // No message at all is the library default: a fresh dataset
        // creation property list starts fill_defined = 1
        // (`FillValueMessage::default`), so a dataset that never got one
        // written reads back exactly as if it had.
        let mut fill_defined: u8 = 1;
        let mut fill_write_time: u8 = FILL_TIME_IFSET;
        let mut alloc_time: u8 = ALLOC_TIME_LATE;
        // The first message that did not decode, kept verbatim: it is the
        // answer a caller gets when it asks for this dataset.
        let mut blocked: Option<String> = None;
        let mut block = |why: String| {
            if blocked.is_none() {
                blocked = Some(why);
            }
        };
        let mut external_file_list = None;

        for msg in &header.messages {
            // A shared message holds a reference to where its body lives, not
            // the body. Decoding one as a body does not fail loudly — it
            // reads the reference's version byte as the body's — so anything
            // this crate does not follow is named here instead. A datatype
            // reference is followed; an attribute reference is skipped, since
            // an attribute never blocks the dataset it hangs on.
            let shared = msg.flags & MSG_FLAG_SHARED != 0;
            if shared && !matches!(msg.msg_type, MSG_DATATYPE | MSG_ATTRIBUTE) {
                block(format!(
                    "its message of type {:#04x} is a shared-message reference, which this \
                     crate follows only for datatypes",
                    msg.msg_type
                ));
                continue;
            }
            match msg.msg_type {
                // The resolver already says whether the type failed to decode
                // or sits somewhere this crate does not follow, so its wording
                // is the reason rather than something to wrap.
                MSG_DATATYPE => {
                    match crate::io::object_header_io::read_datatype_message(handle, meta, msg) {
                        Ok(dt) => datatype = Some(dt),
                        Err(crate::io::IoError::Unsupported(why)) => block(why),
                        Err(e) => block(format!("its datatype message does not decode: {e}")),
                    }
                }
                MSG_DATASPACE => match DataspaceMessage::decode(&msg.data, ctx) {
                    Ok((ds, _)) => dataspace = Some(ds),
                    Err(e) => block(format!("its dataspace message does not decode: {e}")),
                },
                MSG_DATA_LAYOUT => match DataLayoutMessage::decode(&msg.data, ctx) {
                    Ok((dl, _)) => layout = Some(dl),
                    Err(e) => block(format!("its data layout message does not decode: {e}")),
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
                    Err(e) => block(format!("its filter pipeline message does not decode: {e}")),
                },
                MSG_FILL_VALUE => match FillValueMessage::decode(&msg.data) {
                    Ok((fv, _)) => {
                        fill_defined = fv.fill_defined;
                        fill_write_time = fv.fill_write_time;
                        alloc_time = fv.alloc_time;
                        if fv.fill_defined == 2 {
                            fill_value = fv.fill_value;
                        }
                    }
                    Err(e) => block(format!("its fill value message does not decode: {e}")),
                },
                MSG_EXTERNAL_FILE_LIST => {
                    // Unlike a layout message, this one *is* the storage: a
                    // dataset with an external file list has no data address
                    // of its own (H5Dlayout.c routes storage through this
                    // message instead), so a list that does not decode must
                    // block the dataset rather than read back as zero bytes.
                    match ExternalFileListMessage::decode(&msg.data, ctx) {
                        Ok((efl, _)) => external_file_list = Some(efl),
                        Err(e) => block(format!(
                            "its external file list message does not decode: {e}"
                        )),
                    }
                }
                _ => {}
            }
        }

        if let Some(why) = blocked {
            return ObjectKind::UnreadableDataset(why);
        }
        // The storage a dataset names outside its layout message. Both are
        // resolved before the dataset is registered and both block it when
        // they do not resolve, for the same reason the decode above does: a
        // `Virtual` or external-file layout carries no address of its own, so
        // a dropped mapping reads back as fill with no error at all.
        let external_files = match external_file_list {
            Some(efl) => match Self::resolve_external_file_slots(handle, ctx, &efl) {
                Ok(slots) => slots,
                Err(e) => {
                    return ObjectKind::UnreadableDataset(format!(
                        "its external file list does not resolve: {e}"
                    ))
                }
            },
            None => Vec::new(),
        };
        let virtual_mappings = match &layout {
            Some(DataLayoutMessage::Virtual {
                heap_address,
                heap_index,
                ..
            }) if *heap_index != 0 => {
                match Self::resolve_virtual_mappings(handle, ctx, *heap_address, *heap_index, name)
                {
                    Ok(list) => Some(list),
                    Err(e) => {
                        return ObjectKind::UnreadableDataset(format!(
                            "its virtual dataset mapping list does not resolve: {e}"
                        ))
                    }
                }
            }
            _ => None,
        };
        // The attribute set is collected whole, or the object says it could
        // not be: a short list here would be a dataset reporting attributes
        // the file does not agree it has.
        let attributes = collect_object_attributes(handle, ctx, header);
        match (datatype, dataspace, layout) {
            (Some(dt), Some(ds), Some(dl)) => ObjectKind::Dataset(Box::new(DatasetReadInfo {
                name: name.to_string(),
                object_header_address: addr,
                datatype: dt,
                dataspace: ds,
                layout: dl,
                filter_pipeline,
                attributes,
                fill_value,
                fill_defined,
                fill_write_time,
                alloc_time,
                external_files,
                virtual_mappings,
                // Both filled in by `resolve_virtual_extents` once the
                // reader has the directory source names resolve against; a
                // catalog on its own cannot open another file.
                virtual_resolution: None,
                virtual_stored_dims: None,
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
    /// Resolve a virtual dataset's mapping list from the global heap object
    /// its layout message points at (`H5D__virtual_load_layout`,
    /// H5Dvirtual.c). Like the external-file-list decode above, a failure
    /// here must not fall back to silently treating the dataset as having
    /// no data: a `Virtual` layout carries no data address of its own, so a
    /// dropped mapping list would read back as all-fill with no error.
    fn resolve_virtual_mappings(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        heap_address: u64,
        heap_index: u32,
        name: &str,
    ) -> IoResult<VirtualMappingList> {
        let coll = read_heap_collection_from(handle, ctx, heap_address)?;
        let idx = u16::try_from(heap_index).map_err(|_| {
            crate::io::IoError::InvalidState(format!(
                "dataset {name:?} virtual mapping heap index {heap_index} does not fit \
                 the 16-bit on-disk field"
            ))
        })?;
        let obj = coll.get_object(idx).ok_or_else(|| {
            crate::io::IoError::InvalidState(format!(
                "dataset {name:?} virtual mapping list object {idx} not found in the \
                 global heap collection at address {heap_address:#x}"
            ))
        })?;
        VirtualMappingList::decode(obj, ctx).map_err(|e| {
            crate::io::IoError::InvalidState(format!(
                "dataset {name:?} has a malformed virtual dataset mapping list: {e}"
            ))
        })
    }

    /// Resolve every external-file slot's name through the local heap the
    /// EFL message points at (H5Oefl.c decodes only the byte offset; the
    /// string itself lives in a separate on-disk local heap, exactly like a
    /// v0/v1 group's link names — see [`local_heap_get_string`]).
    pub(crate) fn resolve_external_file_slots(
        handle: &mut FileHandle,
        ctx: &FormatContext,
        efl: &ExternalFileListMessage,
    ) -> IoResult<Vec<ExternalFileSegment>> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let heap_hdr_buf = handle.read_at_most(efl.heap_addr, 64)?;
        let heap_hdr = LocalHeapHeader::decode(&heap_hdr_buf, sa, ss)?;
        let heap_data = handle.read_at(heap_hdr.data_addr, heap_hdr.data_size as usize)?;

        efl.slots
            .iter()
            .map(|slot| {
                let name = local_heap_get_string(&heap_data, slot.name_offset)?;
                Ok(ExternalFileSegment {
                    name,
                    offset: slot.offset,
                    size: slot.size,
                })
            })
            .collect()
    }

    /// Return the names of all datasets in the root group.
    pub fn dataset_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.datasets.iter().map(|d| d.name.as_str()).collect();
        names.extend(self.unreadable.keys().map(String::as_str));
        names
    }

    /// Why the dataset at `path` (no leading `/`) cannot be read, or `None`
    /// when it can be — or does not exist.
    pub fn unreadable_reason(&mut self, path: &str) -> Option<&str> {
        if self.external_edge(path).is_some() {
            let (owner, local, _) = self.external_owner(path, MAX_EXTERNAL_HOPS).ok()?;
            let local = owner.canonical_path(&local);
            return owner.unreadable.get(&local).map(String::as_str);
        }
        let path = self.canonical_path(path);
        self.unreadable.get(&path).map(String::as_str)
    }

    /// Every link record in the file, keyed by full path (no leading `/`).
    pub fn links(&self) -> &std::collections::BTreeMap<String, LinkClass> {
        &self.links
    }

    /// The paths of every committed (named) datatype object in this file.
    ///
    /// A committed datatype is in neither [`dataset_names`](Self::dataset_names)
    /// nor the group listing — it is a third kind of object, and this is its
    /// listing.
    pub fn named_datatype_names(&self) -> Vec<&str> {
        self.datatypes.keys().map(String::as_str).collect()
    }

    /// The committed datatype at `path` (no leading `/`), following group hard
    /// links, soft links and external links the way `H5Topen` does.
    ///
    /// `NotFound` means no committed datatype of that name; a name that *is*
    /// one but whose type this crate cannot decode answers `Unsupported` with
    /// the reason, never an absence.
    pub fn named_datatype(&mut self, path: &str) -> IoResult<&DatatypeMessage> {
        self.named_datatype_info(path)?
            .datatype()
            .map_err(|why| crate::io::IoError::Unsupported(why.to_string()))
    }

    /// The attribute names of the committed datatype at `path`, in name
    /// order — matching h5py's default iteration for the (usual) case where
    /// the committed datatype does not track attribute creation order.
    /// Unlike [`Self::dataset_attr_names`] and its group/root counterparts,
    /// this path does not carry a per-attribute creation index to prefer
    /// when the object does track it: committed-datatype attributes are
    /// collected straight from compact header messages
    /// ([`Self::committed_datatype`]), without the envelope's creation index
    /// or dense-storage support the shared `AttributeEntry` collector has.
    pub fn named_datatype_attr_names(&mut self, path: &str) -> IoResult<Vec<String>> {
        let mut names: Vec<String> = self
            .named_datatype_info(path)?
            .attributes()
            .iter()
            .map(|a| a.name.clone())
            .collect();
        names.sort();
        Ok(names)
    }

    /// The committed datatype at `path`'s own object-header attribute count.
    ///
    /// Committed-datatype attributes are collected only from compact header
    /// messages ([`Self::committed_datatype`]) — this crate does not model
    /// dense attribute storage on a named datatype — so unlike
    /// [`ObjectAttributes::header_count`] this is simply the count of what
    /// [`Self::named_datatype_attr_names`] already lists, with no separate
    /// dense-index path to fall back to.
    pub fn named_datatype_header_attr_count(&mut self, path: &str) -> IoResult<u64> {
        Ok(self.named_datatype_info(path)?.attributes().len() as u64)
    }

    /// One attribute of the committed datatype at `path`, by name.
    pub fn named_datatype_attr(
        &mut self,
        path: &str,
        attr_name: &str,
    ) -> IoResult<&AttributeMessage> {
        let owned = attr_name.to_string();
        self.named_datatype_info(path)?
            .attributes()
            .iter()
            .find(|a| a.name == owned)
            .ok_or_else(|| crate::io::IoError::NotFound(format!("{path}:{attr_name}")))
    }

    /// The committed datatype object at `path`, after link traversal.
    ///
    /// The object answers here whether or not its type decodes; the reason it
    /// does not is on [`CommittedDatatypeInfo::datatype`].
    pub fn named_datatype_info(&mut self, path: &str) -> IoResult<&CommittedDatatypeInfo> {
        if self.external_edge(path).is_some() {
            let (owner, local, _) = self.external_owner(path, MAX_EXTERNAL_HOPS)?;
            let local = owner.canonical_path(&local);
            return owner
                .datatypes
                .get(&local)
                .ok_or(crate::io::IoError::NotFound(local));
        }
        let local = self.canonical_path(path);
        self.datatypes
            .get(&local)
            .ok_or(crate::io::IoError::NotFound(local))
    }

    /// The class of the link at `path` (no leading `/`), or `None` when no
    /// link of that name exists. The path is traversed first, so a link
    /// reached through a group hard link, a soft link or an external link
    /// resolves — an external link's own record is found before the
    /// traversal crosses it, since a name matches its own link exactly.
    pub fn link_class(&mut self, path: &str) -> Option<&LinkClass> {
        let path = path.trim_start_matches('/');
        if self.links.contains_key(path) {
            return self.links.get(path);
        }
        if self.external_edge(path).is_some() {
            let (owner, local, _) = self.external_owner(path, MAX_EXTERNAL_HOPS).ok()?;
            let local = owner.canonical_path(&local);
            return owner.links.get(&local);
        }
        let path = self.canonical_path(path);
        self.links.get(&path)
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

    /// The messages read from the superblock extension object header. All
    /// fields are `None` for a file without an extension.
    pub fn superblock_extension(&self) -> &SuperblockExtension {
        &self.ext
    }

    /// Bytes the file's on-disk free-space managers record as free —
    /// `H5Fget_freespace`, the number `h5stat -S` prints as "Amount of tracked
    /// free space".
    ///
    /// Zero for a file whose file-space info message names no manager, which
    /// includes every file written without `persist`. The strategy is not
    /// consulted: a manager's header and section-info blocks have one layout
    /// whichever strategy allocated the space they describe.
    pub fn tracked_free_space(&mut self) -> IoResult<u64> {
        let Some(info) = self.ext.file_space_info.clone() else {
            return Ok(0);
        };
        crate::io::free_space_io::tracked_free_space(&mut self.handle, &self.meta.ctx, &info)
    }

    /// Size in bytes of the userblock preceding the superblock: the offset the
    /// signature was found at, which is also the file's base address. Zero for
    /// a file without a userblock.
    pub fn userblock_size(&self) -> u64 {
        self.handle.base()
    }

    /// The superblock format version (0-3), decoded once at open time and
    /// immutable for the life of an open file — a live SWMR refresh rescans
    /// the file's contents but never its own format version.
    pub fn superblock_version(&self) -> u8 {
        self.superblock_version
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

    /// Where `name` leaves this file, or `None` when it resolves inside it.
    ///
    /// This is the one question every path-taking entry point asks before it
    /// looks anything up: a name that crosses an external link is not this
    /// file's to answer, and answering it from this file's catalog anyway is
    /// how such a name came back as a plain absence.
    pub(crate) fn external_edge(&self, name: &str) -> Option<ExternalEdge> {
        match self.traverse(name) {
            Traversal::Path { .. } => None,
            Traversal::External { link, file, path } => Some(ExternalEdge { link, file, path }),
        }
    }

    /// Candidate filesystem paths for an external link's target file, in the
    /// order `H5F_prefix_open_file` tries them: an absolute name as given,
    /// then each `HDF5_EXT_PREFIX` component, then the directory holding this
    /// file, then the working directory. An absolute name that does not exist
    /// falls back to its last component for the later attempts, as the C does.
    ///
    /// The one step of that order this does not implement is the link-access
    /// property list's `H5Pset_elink_prefix`, which has no equivalent in this
    /// crate's API yet; it would sit between the environment variable and this
    /// file's directory.
    fn external_candidates(&self, file: &str) -> Vec<PathBuf> {
        let raw = Path::new(file);
        let mut candidates = Vec::new();
        if raw.is_absolute() {
            candidates.push(raw.to_path_buf());
        }
        // Every attempt after an absolute miss uses the bare file name.
        let base: &Path = if raw.is_absolute() {
            Path::new(raw.file_name().unwrap_or(raw.as_os_str()))
        } else {
            raw
        };
        if let Ok(prefixes) = std::env::var("HDF5_EXT_PREFIX") {
            candidates.extend(
                prefixes
                    .split(':')
                    .filter(|p| !p.is_empty())
                    .map(|p| Path::new(p).join(base)),
            );
        }
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            candidates.push(dir.join(base));
        }
        candidates.push(base.to_path_buf());
        candidates
    }

    /// Open one external link's target file, or hand back the handle a
    /// previous link to the same resolved path already opened.
    fn external_target(&mut self, link: &str, file: &str) -> IoResult<&mut Hdf5Reader> {
        // The search runs once per link value; after that the answer is what
        // this reader resolved it to, whatever the filesystem does next.
        let resolved = match self.external_resolved.get(file) {
            Some(resolved) => resolved.clone(),
            None => {
                let candidates = self.external_candidates(file);
                let resolved = candidates
                    .iter()
                    .find(|p| p.is_file())
                    .cloned()
                    .ok_or_else(|| crate::io::IoError::ExternalFileNotFound {
                        link: link.to_string(),
                        file: file.to_string(),
                        searched: candidates.iter().map(|p| p.display().to_string()).collect(),
                    })?;
                self.external_resolved
                    .insert(file.to_string(), resolved.clone());
                resolved
            }
        };
        self.cross_file(resolved)
    }

    /// Open `resolved`, or hand back the handle a previous crossing to the
    /// same file already opened.
    ///
    /// The single owner of every file this reader opens on another file's
    /// behalf, so a path named by any number of external links and external
    /// references is opened once and read through one handle. What resolved
    /// the name to this path is the caller's business, and differs by kind:
    /// an external link runs `H5F_prefix_open_file`'s search order, a
    /// reference has no search order at all.
    fn cross_file(&mut self, resolved: PathBuf) -> IoResult<&mut Hdf5Reader> {
        let locking = self.locking;
        match self.external.entry(resolved) {
            std::collections::btree_map::Entry::Occupied(e) => Ok(&mut **e.into_mut()),
            std::collections::btree_map::Entry::Vacant(e) => {
                let reader = Hdf5Reader::open_with_locking(e.key(), locking)?;
                Ok(&mut **e.insert(Box::new(reader)))
            }
        }
    }

    /// The reader that owns `name`, the path of `name` inside it, and the last
    /// external link crossed to get there.
    ///
    /// This is the single owner of cross-file resolution: it follows external
    /// links until the remaining path resolves inside the reader it returns,
    /// so callers do exactly one delegation and never have to re-check.
    fn external_owner(
        &mut self,
        name: &str,
        hops: usize,
    ) -> IoResult<(&mut Self, String, Option<ExternalEdge>)> {
        let path = name.trim_start_matches('/').to_string();
        let Some(edge) = self.external_edge(&path) else {
            return Ok((self, path, None));
        };
        if hops == 0 {
            return Err(crate::io::IoError::InvalidState(format!(
                "resolving '{name}' crossed more than {MAX_EXTERNAL_HOPS} external links \
                 (libhdf5 stops at the same H5L_NUM_LINKS); the links may form a cycle"
            )));
        }
        let target = self.external_target(&edge.link, &edge.file)?;
        let (owner, path, deeper) = target.external_owner(&edge.path, hops - 1)?;
        Ok((owner, path, deeper.or(Some(edge))))
    }

    /// Return metadata for a dataset by name. Like `H5Dopen`, the name may
    /// pass through group hard links, soft links and external links.
    pub fn dataset_info(&mut self, name: &str) -> Option<&DatasetReadInfo> {
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS).ok()?;
            return owner.dataset_info_local(&path);
        }
        self.dataset_info_local(name)
    }

    /// [`dataset_info`](Self::dataset_info) restricted to this file: soft
    /// links and group hard links resolve, an external link does not. Every
    /// read path uses this, because by then the owning reader has already been
    /// selected and the path is local to it.
    fn dataset_info_local(&self, name: &str) -> Option<&DatasetReadInfo> {
        let name = self.canonical_path(name);
        self.datasets.iter().find(|d| d.name == name)
    }

    /// Open `name` as a dataset the way `H5Dopen2` does, reporting *why* it
    /// cannot be opened instead of collapsing every cause into absence: a
    /// soft link whose target does not exist is a dangling link, and a path
    /// through an external link is resolved in the file that link names.
    ///
    /// This is the gate every typed dataset access goes through. `access` is
    /// the dapl the open names; its properties are put in force for the
    /// dataset first, so the extent this returns is the one they resolve it
    /// to.
    pub fn open_dataset_with(
        &mut self,
        name: &str,
        access: DatasetAccess,
    ) -> IoResult<&DatasetReadInfo> {
        if self.external_edge(name).is_some() {
            let (owner, path, edge) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            owner.apply_dataset_access(&path, access)?;
            return match owner.open_dataset_local(&path) {
                // The name is absent in the target file, which makes the link
                // that pointed there dangling — not the caller's path absent.
                Err(crate::io::IoError::NotFound(_)) => {
                    Err(edge.map_or_else(|| crate::io::IoError::NotFound(path), |e| e.dangling()))
                }
                other => other,
            };
        }
        self.apply_dataset_access(name, access)?;
        self.open_dataset_local(name)
    }

    /// [`open_dataset`](Self::open_dataset) restricted to this file.
    fn open_dataset_local(&self, name: &str) -> IoResult<&DatasetReadInfo> {
        let Traversal::Path { path, via } = self.traverse(name) else {
            // `external_owner` only ever returns a reader in which the
            // remaining path resolves locally, so no caller can land here.
            return Err(crate::io::IoError::NotFound(name.to_string()));
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

    /// Resolve one entry of an attribute list.
    ///
    /// The cases an attribute list can answer are kept apart here rather than
    /// at each call site: decoded, present but undecodable, absent from a set
    /// known to be whole, and absent from a set that was never read whole. A
    /// caller that collapsed any of the middle cases into the last would
    /// report an attribute the file contains as one it does not.
    fn resolve_attr<'a>(
        attrs: &'a ObjectAttributes,
        owner: &str,
        name: &str,
    ) -> IoResult<&'a AttributeMessage> {
        match attrs.entries.iter().find(|a| a.name() == name) {
            Some(entry) => entry.decoded().map_err(|reason| {
                crate::io::IoError::Unsupported(format!(
                    "attribute '{name}' on '{owner}' cannot be decoded: {reason}"
                ))
            }),
            // Not among what was read — but the part that was not read could
            // hold it, so an incomplete set cannot answer "absent".
            None => match attrs.unreadable_reason() {
                Some(reason) => Err(incomplete_error(owner, reason)),
                None => Err(crate::io::IoError::NotFound(format!("{owner}:{name}"))),
            },
        }
    }

    /// Why the attribute `name` in `attrs` cannot be read, or `None` when it
    /// can be — or is not there at all, which the accessors above report as
    /// `NotFound`.
    fn attr_reason<'a>(attrs: &'a ObjectAttributes, name: &str) -> Option<&'a str> {
        attrs
            .entries
            .iter()
            .find(|a| a.name() == name)?
            .unreadable_reason()
    }

    /// Why a dataset's attribute cannot be read, or `None` when it can be.
    pub fn dataset_attr_unreadable_reason(
        &mut self,
        ds_name: &str,
        attr_name: &str,
    ) -> Option<&str> {
        Self::attr_reason(&self.dataset_info(ds_name)?.attributes, attr_name)
    }

    /// Why a root-level attribute cannot be read, or `None` when it can be.
    pub fn root_attr_unreadable_reason(&self, name: &str) -> Option<&str> {
        Self::attr_reason(&self.root_attributes, name)
    }

    /// Why a non-root group's attribute cannot be read, or `None` when it can
    /// be.
    pub fn group_attr_unreadable_reason(&self, group_path: &str, name: &str) -> Option<&str> {
        Self::attr_reason(
            self.group_attributes
                .get(&self.canonical_path(group_path))?,
            name,
        )
    }

    /// Why a dataset's attributes cannot be listed at all, or `None` when the
    /// set is whole. Object scope, unlike
    /// [`Self::dataset_attr_unreadable_reason`]: the failure belongs to no
    /// single name.
    pub fn dataset_attrs_unreadable_reason(&mut self, ds_name: &str) -> Option<&str> {
        self.dataset_info(ds_name)?.attributes.unreadable_reason()
    }

    /// A dataset's own compact-vs-dense attribute storage.
    pub fn dataset_attr_storage(&mut self, ds_name: &str) -> IoResult<AttributeStorage> {
        Ok(self
            .dataset_info(ds_name)
            .ok_or_else(|| crate::io::IoError::NotFound(ds_name.to_string()))?
            .attributes
            .storage())
    }

    /// A dataset's own object-header attribute count.
    pub fn dataset_header_attr_count(&mut self, ds_name: &str) -> IoResult<u64> {
        let info = self
            .dataset_info(ds_name)
            .ok_or_else(|| crate::io::IoError::NotFound(ds_name.to_string()))?;
        info.attributes.header_count(ds_name)
    }

    /// Why the root group's attributes cannot be listed at all, or `None` when
    /// the set is whole.
    pub fn root_attrs_unreadable_reason(&self) -> Option<&str> {
        self.root_attributes.unreadable_reason()
    }

    /// Why a non-root group's attributes cannot be listed at all, or `None`
    /// when the set is whole.
    pub fn group_attrs_unreadable_reason(&self, group_path: &str) -> Option<&str> {
        self.group_attributes
            .get(&self.canonical_path(group_path))?
            .unreadable_reason()
    }

    /// Return the attribute names of a dataset.
    ///
    /// Includes attributes this crate cannot decode: the object header carries
    /// them, so the listing does too. [`Self::dataset_attr`] says why one of
    /// those cannot be read. An object whose attribute set could not be read
    /// whole has no listing to give and returns the reason instead — see
    /// [`Self::dataset_attrs_unreadable_reason`].
    pub fn dataset_attr_names(&mut self, name: &str) -> IoResult<Vec<String>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        info.attributes.ordered_names(name)
    }

    /// Return a specific attribute by dataset name and attribute name.
    pub fn dataset_attr(&mut self, ds_name: &str, attr_name: &str) -> IoResult<&AttributeMessage> {
        let info = self
            .dataset_info(ds_name)
            .ok_or_else(|| crate::io::IoError::NotFound(ds_name.to_string()))?;
        Self::resolve_attr(&info.attributes, ds_name, attr_name)
    }

    /// Return the names of root-level (file) attributes, undecodable ones
    /// included — see [`Self::dataset_attr_names`].
    pub fn root_attr_names(&self) -> IoResult<Vec<String>> {
        self.root_attributes.ordered_names("/")
    }

    /// Return a root-level attribute by name.
    pub fn root_attr(&self, name: &str) -> IoResult<&AttributeMessage> {
        Self::resolve_attr(&self.root_attributes, "/", name)
    }

    /// The root group's own attribute creation-order policy.
    pub fn root_attr_creation_order(&self) -> CreationOrder {
        self.root_attributes.creation_order()
    }

    /// The root group's own compact-vs-dense attribute storage.
    pub fn root_attr_storage(&self) -> AttributeStorage {
        self.root_attributes.storage()
    }

    /// The root group's own object-header attribute count.
    pub fn root_header_attr_count(&self) -> IoResult<u64> {
        self.root_attributes.header_count("/")
    }

    /// The root group's own link creation-order policy.
    pub fn root_link_creation_order(&self) -> CreationOrder {
        self.root_link_storage.1
    }

    /// The root group's own link storage kind: symbol-table (legacy),
    /// compact link messages, or dense (fractal heap plus name index).
    pub fn root_link_storage(&self) -> LinkStorage {
        self.root_link_storage.0
    }

    /// Return the attribute names of a non-root group (path without a
    /// leading `/`, e.g. `"detector"` or `"entry/instrument"`; may pass
    /// through group hard links). Undecodable attributes included — see
    /// [`Self::dataset_attr_names`].
    pub fn group_attr_names(&mut self, group_path: &str) -> IoResult<Vec<String>> {
        if self.external_edge(group_path).is_some() {
            let (owner, local, _) = self.external_owner(group_path, MAX_EXTERNAL_HOPS)?;
            // The empty remainder is the target file's root group, whose
            // attributes are not in the per-group map.
            if local.is_empty() {
                return owner.root_attr_names();
            }
            return owner.group_attr_names_local(&local);
        }
        self.group_attr_names_local(group_path)
    }

    fn group_attr_names_local(&self, group_path: &str) -> IoResult<Vec<String>> {
        let Some(attrs) = self.group_attributes.get(&self.canonical_path(group_path)) else {
            return Ok(Vec::new());
        };
        attrs.ordered_names(group_path)
    }

    /// A non-root group's own attribute creation-order policy. `Untracked`
    /// for a path the walk never reached, the same silent default
    /// [`group_attr_names_local`](Self::group_attr_names_local) gives an
    /// unknown group's attribute listing.
    pub fn group_attr_creation_order(&self, group_path: &str) -> CreationOrder {
        self.group_attributes
            .get(&self.canonical_path(group_path))
            .map(ObjectAttributes::creation_order)
            .unwrap_or_default()
    }

    /// A non-root group's own compact-vs-dense attribute storage. `Compact`
    /// — the same silent default as an empty attribute set — for a path the
    /// walk never reached.
    pub fn group_attr_storage(&self, group_path: &str) -> AttributeStorage {
        self.group_attributes
            .get(&self.canonical_path(group_path))
            .map(ObjectAttributes::storage)
            .unwrap_or_default()
    }

    /// A non-root group's own object-header attribute count. `0` for a path
    /// the walk never reached, the same silent default
    /// [`group_attr_names_local`](Self::group_attr_names_local) gives an
    /// unknown group's attribute listing.
    pub fn group_header_attr_count(&self, group_path: &str) -> IoResult<u64> {
        let Some(attrs) = self.group_attributes.get(&self.canonical_path(group_path)) else {
            return Ok(0);
        };
        attrs.header_count(group_path)
    }

    /// A non-root group's own link creation-order policy. `Untracked` for a
    /// path the walk never reached, the same silent default
    /// [`group_attr_creation_order`](Self::group_attr_creation_order) gives.
    pub fn group_link_creation_order(&self, group_path: &str) -> CreationOrder {
        self.group_link_storage
            .get(&self.canonical_path(group_path))
            .map_or(CreationOrder::Untracked, |(_, order)| *order)
    }

    /// A non-root group's own link storage kind. `Compact` — the same
    /// silent default as an empty link set — for a path the walk never
    /// reached.
    pub fn group_link_storage(&self, group_path: &str) -> LinkStorage {
        self.group_link_storage
            .get(&self.canonical_path(group_path))
            .map_or(LinkStorage::Compact, |(storage, _)| *storage)
    }

    /// Return a non-root group's attribute by name.
    pub fn group_attr(&mut self, group_path: &str, name: &str) -> IoResult<&AttributeMessage> {
        if self.external_edge(group_path).is_some() {
            let (owner, local, _) = self.external_owner(group_path, MAX_EXTERNAL_HOPS)?;
            if local.is_empty() {
                return owner.root_attr(name);
            }
            return owner.group_attr_local(&local, name);
        }
        self.group_attr_local(group_path, name)
    }

    fn group_attr_local(&self, group_path: &str, name: &str) -> IoResult<&AttributeMessage> {
        match self.group_attributes.get(&self.canonical_path(group_path)) {
            Some(attrs) => Self::resolve_attr(attrs, group_path, name),
            // No entry at all: the walk found nothing to record on this group.
            None => Err(crate::io::IoError::NotFound(format!("{group_path}:{name}"))),
        }
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
        read_heap_collection_from(&mut self.handle, &self.meta.ctx, addr)
    }

    /// Decode an attribute's value as a string, resolving a variable-length
    /// string attribute through the global heap (h5py writes string
    /// attributes as variable-length by default).
    pub fn attr_string_value(&mut self, attr: &AttributeMessage) -> IoResult<String> {
        use crate::format::messages::datatype::DatatypeMessage;
        if !matches!(attr.datatype, DatatypeMessage::VarLenString { .. }) {
            return fixed_string_attr_value(attr);
        }
        // Variable-length string: the attribute value is a global-heap
        // reference (sequence length + collection address + object index).
        if attr.data.len() < vlen_reference_size(&self.meta.ctx) {
            return Ok(String::new());
        }
        let (_seq, coll_addr, obj_index) = decode_vlen_reference(&attr.data, &self.meta.ctx)?;
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

    /// The absolute path of the object whose header sits at `addr` — what an
    /// object reference to it names — or `None` when no group or dataset the
    /// discovery walk reached lives there (a reference into a file region the
    /// walk never traversed, or a stale one).
    pub fn path_for_object(&self, addr: u64) -> Option<&str> {
        self.object_paths.get(&addr).map(String::as_str)
    }

    /// Read a reference dataset's elements, resolved to the objects they name.
    pub fn read_references(&mut self, name: &str) -> IoResult<Vec<Reference>> {
        let datatype = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?
            .datatype
            .clone();
        let raw = self.read_dataset_raw(name)?;
        self.decode_references(&datatype, &raw)
    }

    /// Read an attribute's value as reference elements.
    pub fn attr_references(&mut self, attr: &AttributeMessage) -> IoResult<Vec<Reference>> {
        self.decode_references(&attr.datatype, &attr.data)
    }

    /// The single owner of reference decoding for both carriers of reference
    /// elements — dataset payloads and attribute values.
    fn decode_references(
        &mut self,
        datatype: &DatatypeMessage,
        bytes: &[u8],
    ) -> IoResult<Vec<Reference>> {
        let DatatypeMessage::Reference { size, kind } = datatype else {
            return Err(crate::io::IoError::InvalidState(format!(
                "datatype {datatype} is not a reference"
            )));
        };
        let (size, kind) = (*size as usize, *kind);
        if size == 0 {
            // A corrupt file can declare it; `chunks_exact(0)` panics.
            return Err(crate::io::IoError::InvalidState(
                "reference datatype has zero width".into(),
            ));
        }
        let encoding = kind.encoding();

        // References written by one call share one heap collection, so read
        // each collection once rather than per element.
        let mut heaps = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(bytes.len() / size);
        for elem in bytes.chunks_exact(size) {
            out.push(self.decode_reference_element(elem, encoding, &mut heaps)?);
        }
        Ok(out)
    }

    /// One reference element, resolved against the file.
    ///
    /// `heaps` caches the global-heap collections region references point
    /// into, keyed by collection address.
    fn decode_reference_element(
        &mut self,
        elem: &[u8],
        encoding: ReferenceEncoding,
        heaps: &mut std::collections::HashMap<u64, GlobalHeapCollection>,
    ) -> IoResult<Reference> {
        match encoding {
            ReferenceEncoding::Old(OldReferenceKind::Object) => {
                match decode_object_element(elem, &self.meta.ctx)? {
                    None => Ok(Reference::Null),
                    Some(address) => Ok(self.resolve_reference(DecodedReference {
                        address,
                        file: None,
                        target: ReferenceTarget::Object,
                    })),
                }
            }
            ReferenceEncoding::Old(OldReferenceKind::DatasetRegion) => {
                let Some((coll_addr, obj_index)) = decode_region_element(elem, &self.meta.ctx)?
                else {
                    return Ok(Reference::Null);
                };
                let obj = self.heap_object(coll_addr, obj_index, heaps)?;
                let (address, selection) = decode_region_heap_object(obj, &self.meta.ctx)?;
                Ok(self.resolve_reference(DecodedReference {
                    address,
                    file: None,
                    target: ReferenceTarget::Region(selection),
                }))
            }
            ReferenceEncoding::Revised => {
                let (kind, external, body) = match decode_revised_element(elem, &self.meta.ctx)? {
                    RevisedElement::Null => return Ok(Reference::Null),
                    RevisedElement::Inline { kind, body } => (kind, false, body.to_vec()),
                    RevisedElement::Heap {
                        kind,
                        external,
                        collection,
                        index,
                    } => (
                        kind,
                        external,
                        self.heap_object(collection, index, heaps)?.to_vec(),
                    ),
                };
                match decode_revised_body(kind, external, &body, &self.meta.ctx)? {
                    None => Ok(Reference::Null),
                    Some(decoded) => Ok(self.resolve_reference(decoded)),
                }
            }
        }
    }

    /// Attach the target's path to a decoded reference — the one place an
    /// address becomes a [`Reference`], so every kind resolves the same way.
    ///
    /// A reference naming another file is looked up in that file, which this
    /// opens by the name the reference carries and nothing else:
    /// `H5R__reopen_file` hands the name straight to `H5VL_file_open` with no
    /// prefix search, so it is read against the process working directory the
    /// way `H5Ropen_object` would read it (H5Rint.c:466, :487). A file that is
    /// not there leaves the path unresolved while the reference still names
    /// it, which is `H5Rget_file_name` answering from the reference alone
    /// while `H5Ropen_object` fails (H5R.c:1036-1039).
    fn resolve_reference(&mut self, decoded: DecodedReference) -> Reference {
        let DecodedReference {
            address,
            file,
            target,
        } = decoded;
        let path = match &file {
            None => self.path_for_object(address).map(str::to_string),
            Some(name) => self
                .cross_file(PathBuf::from(name))
                .ok()
                .and_then(|target| target.path_for_object(address).map(str::to_string)),
        };
        match target {
            ReferenceTarget::Object => Reference::Object {
                address,
                file,
                path,
            },
            ReferenceTarget::Region(selection) => Reference::Region {
                address,
                file,
                path,
                selection,
            },
            ReferenceTarget::Attribute(name) => Reference::Attr {
                address,
                file,
                path,
                name,
            },
        }
    }

    /// One global-heap object, reading its collection at most once.
    fn heap_object<'h>(
        &mut self,
        collection: u64,
        index: u32,
        heaps: &'h mut std::collections::HashMap<u64, GlobalHeapCollection>,
    ) -> IoResult<&'h [u8]> {
        if let std::collections::hash_map::Entry::Vacant(slot) = heaps.entry(collection) {
            slot.insert(self.read_heap_collection(collection)?);
        }
        let idx = u16::try_from(index).map_err(|_| {
            crate::io::IoError::InvalidState(format!(
                "global heap object index {index} does not fit the 16-bit on-disk field"
            ))
        })?;
        heaps[&collection].get_object(idx).ok_or_else(|| {
            crate::io::IoError::InvalidState(format!(
                "global heap object {idx} not found in the collection at address {collection:#x}"
            ))
        })
    }

    /// Return the dimensions of a dataset.
    pub fn dataset_shape(&mut self, name: &str) -> IoResult<Vec<u64>> {
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        Ok(info.dataspace.dims.clone())
    }

    /// Logical byte size of a dataset's full image (`product(dims) *
    /// element_size`), with the datatype needed for the post-filter conversion.
    ///
    /// The NULL dataspace (`dataspace.is_null()`) holds zero elements — not
    /// one, the way an empty `dims` would suggest by the same product-of-dims
    /// arithmetic a scalar dataspace uses (`dims` is empty for both).
    fn raw_size_and_datatype(&self, name: &str) -> IoResult<(DatatypeMessage, u64)> {
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let total = if info.dataspace.is_null() {
            0
        } else {
            saturating_byte_len(&info.dataspace.dims, info.datatype.element_size() as u64)
        };
        Ok((info.datatype.clone(), total))
    }

    /// Read the raw bytes of a dataset.
    pub fn read_dataset_raw(&mut self, name: &str) -> IoResult<Vec<u8>> {
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_dataset_raw(&path);
        }
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
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_dataset_raw_into(&path, out);
        }
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
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;

        // Clone to avoid borrow conflict with &mut self in read methods.
        let layout = info.layout.clone();
        let pipeline = info.filter_pipeline.clone();
        let fill_value = info.fill_value.clone();
        let external_files = info.external_files.clone();

        match &layout {
            DataLayoutMessage::Contiguous { .. } if !external_files.is_empty() => {
                let prefix = resolve_extfile_prefix(&self.source_dir);
                read_external_file_bytes(&external_files, prefix.as_deref(), 0, out)?;
            }
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
            DataLayoutMessage::Virtual { .. } => {
                fill_tiled_into(out, fill_value.as_deref());
                self.read_virtual_into(name, out, 0)?;
            }
        }
        Ok(())
    }

    /// Set every virtual dataset's extent from the sources its unlimited
    /// mappings can reach — `H5D__virtual_set_extent_unlim` (H5Dvirtual.c),
    /// which libhdf5 runs when it opens such a dataset.
    ///
    /// INVARIANT: a virtual dataset's `dataspace.dims` are the extent its
    /// available sources give it. This is the single owner of that
    /// resolution: the dims a VDS with an unlimited mapping reports are not
    /// the ones its dataspace message stores, and every shape query, read and
    /// slice bound must see the same value, so the resolved extent is stamped
    /// in once — here, immediately after a catalog is built — rather than
    /// recomputed per call. The per-mapping clip sizes the extent came from
    /// are kept beside it in
    /// [`virtual_resolution`](DatasetReadInfo::virtual_resolution) so a read
    /// walks exactly the sources the extent was derived from.
    ///
    /// A source that cannot be opened contributes a clip size of 0, never an
    /// error: a virtual dataset whose sources are not written yet is legal,
    /// and reads back as the fill value (upstream's "clip_size = 0" arm when
    /// `H5D__virtual_open_source_dset` leaves the dataset closed).
    ///
    /// The default view is assumed throughout — `H5D_VDS_LAST_AVAILABLE` is
    /// `H5D_ACS_VDS_VIEW_DEF`, and `H5Pset_virtual_view` sets a *dataset
    /// access* property that is never stored in the file, so a reader opening
    /// a file it did not create always sees the default.
    fn resolve_virtual_extents(&mut self) -> IoResult<()> {
        let targets: Vec<usize> = (0..self.datasets.len())
            .filter(|&i| self.datasets[i].virtual_mappings.is_some())
            .collect();
        for i in targets {
            // The catalog is freshly built, so `dataspace.dims` is still the
            // extent the dataspace message stores. Record it before the
            // resolution replaces it: that is what a later open under other
            // access properties has to resolve from.
            let stored = self.datasets[i].dataspace.dims.clone();
            self.datasets[i].virtual_stored_dims = Some(stored.clone());
            self.resolve_virtual_extent_of(i, &stored)?;
        }
        Ok(())
    }

    /// Resolve one virtual dataset's extent from its *stored* dims under the
    /// [`DatasetAccess`] in force for it, and stamp both the extent and the
    /// per-mapping resolutions in.
    fn resolve_virtual_extent_of(&mut self, i: usize, stored: &[u64]) -> IoResult<()> {
        let Some(mappings) = self.datasets[i].virtual_mappings.clone() else {
            return Ok(());
        };
        let access = self.access_in_force(&self.datasets[i].name.clone());
        let (resolution, dims) = self.resolve_one_virtual_extent(&mappings, stored, access)?;
        self.datasets[i].dataspace.dims = dims;
        self.datasets[i].virtual_resolution = Some(resolution);
        Ok(())
    }

    /// The dataset-access properties in force for `name` (already canonical),
    /// libhdf5's defaults when no open has named others.
    fn access_in_force(&self, canonical: &str) -> DatasetAccess {
        self.virtual_access
            .get(canonical)
            .copied()
            .unwrap_or_default()
    }

    /// Put `access` in force for `name` and re-resolve its extent under it —
    /// what `H5Dopen` with a non-default dapl does, since `H5D__virtual_init`
    /// stores the dapl on the dataset and `H5D__virtual_set_extent_unlim`
    /// then runs against it (H5Dvirtual.c:2178-2188, :1386).
    ///
    /// A name that is not a resolved virtual dataset takes nothing: the two
    /// properties this models are the two that only a virtual dataset reads.
    fn apply_dataset_access(&mut self, name: &str, access: DatasetAccess) -> IoResult<()> {
        let canonical = self.canonical_path(name);
        let Some(i) = self.datasets.iter().position(|d| d.name == canonical) else {
            return Ok(());
        };
        let Some(stored) = self.datasets[i].virtual_stored_dims.clone() else {
            return Ok(());
        };
        if self.access_in_force(&canonical) == access {
            return Ok(());
        }
        self.virtual_access.insert(canonical, access);
        // A source may be a virtual dataset in this same file, and the
        // access propagates to it (H5Dvirtual.c:2224-2226), so this can
        // re-enter; the depth counter is the same cycle guard the open-time
        // resolution uses.
        VirtualResolveDepth::enter(|| self.resolve_virtual_extent_of(i, &stored))
    }

    /// [`resolve_virtual_extents`](Self::resolve_virtual_extents) for one
    /// dataset: the per-mapping resolutions and the extent they imply.
    fn resolve_one_virtual_extent(
        &mut self,
        mappings: &VirtualMappingList,
        curr_dims: &[u64],
        access: DatasetAccess,
    ) -> IoResult<(Vec<MappingResolution>, Vec<u64>)> {
        let rank = curr_dims.len();
        let mut resolution = Vec::with_capacity(mappings.mappings.len());
        let mut new_dims: Vec<Option<u64>> = vec![None; rank];
        // `H5D_virtual_update_min_dims`: whatever the unlimited dimension
        // resolves to, the extent must still hold every bounded mapping.
        let mut min_dims = vec![0u64; rank];
        // `H5S_hyper_get_clip_extent_match`'s `incl_trail`: a
        // `H5D_VDS_FIRST_MISSING` view stops where the trailing partial
        // block would begin (H5Dvirtual.c:1447-1451).
        let incl_trail = access.view() == VirtualView::FirstMissing;
        // Where two mappings disagree about the unlimited dimension,
        // `H5D_VDS_FIRST_MISSING` takes the smallest clip and
        // `H5D_VDS_LAST_AVAILABLE` the largest (H5Dvirtual.c:1662-1667).
        let take_clip = |slot: &mut Option<u64>, clip: u64| {
            if slot.is_none_or(|d| if incl_trail { clip < d } else { clip > d }) {
                *slot = Some(clip);
            }
        };

        for m in &mappings.mappings {
            let unlim_virtual = m.virtual_selection.unlim_dim();
            let res = match (unlim_virtual, m.source_selection.unlim_dim()) {
                (Some(vd), Some(sd)) => {
                    let source_clip = self
                        .virtual_source_dims(m, access)
                        .ok()
                        .flatten()
                        .and_then(|d| d.get(sd).copied())
                        .unwrap_or(0);
                    let virtual_clip = match (
                        regular_hyperslab(&m.virtual_selection),
                        regular_hyperslab(&m.source_selection),
                    ) {
                        // `H5S_hyper_get_clip_extent_match`: how many slices
                        // the source supplies, then the virtual extent that
                        // covers exactly that many. Its `incl_trail`
                        // argument is `view == H5D_VDS_FIRST_MISSING`
                        // (H5Dvirtual.c:1447-1451).
                        (Some(v), Some(sr)) => {
                            v.clip_extent(sr.num_slices(source_clip), incl_trail)
                        }
                        _ => 0,
                    };
                    take_clip(&mut new_dims[vd], virtual_clip);
                    MappingResolution::Unlimited {
                        virtual_clip,
                        source_clip,
                    }
                }
                // Unlimited virtual selection, limited source selection:
                // the printf shape, where the successive blocks of the
                // virtual selection come from successively-named source
                // datasets.
                (Some(vd), None) => {
                    let (blocks, present) = self.printf_blocks_present(m, access);
                    let virtual_clip = match (blocks, regular_hyperslab(&m.virtual_selection)) {
                        // `H5D__virtual_set_extent_unlim`'s "check for no
                        // datasets" arm, which is 0 under either view
                        // (H5Dvirtual.c:1623-1626).
                        (0, _) | (_, None) => 0,
                        // The extent ends just past the last block that has
                        // a source under `H5D_VDS_LAST_AVAILABLE`, and where
                        // the first missing block starts under
                        // `H5D_VDS_FIRST_MISSING` (H5Dvirtual.c:1630-1653).
                        (n, Some(r)) => match access.view() {
                            VirtualView::LastAvailable => {
                                let last = r.unlim_block(n - 1);
                                last.start[vd] + last.block[vd]
                            }
                            VirtualView::FirstMissing => r.unlim_block(n).start[vd],
                        },
                    };
                    take_clip(&mut new_dims[vd], virtual_clip);
                    MappingResolution::Printf { blocks, present }
                }
                _ => MappingResolution::Bounded,
            };
            if let Some((_, hi)) = m.virtual_selection.bounds() {
                for (d, &e) in hi.iter().enumerate().take(rank) {
                    if Some(d) != unlim_virtual && e + 1 > min_dims[d] {
                        min_dims[d] = e + 1;
                    }
                }
            }
            resolution.push(res);
        }

        let dims = (0..rank)
            .map(|d| match new_dims[d] {
                Some(v) => v.max(min_dims[d]),
                None => curr_dims[d],
            })
            .collect();
        Ok((resolution, dims))
    }

    /// The extent of the source dataset one mapping names, or `None` when it
    /// cannot be reached — `H5D__virtual_open_source_dset` leaving the source
    /// closed, which upstream reads as "no data there yet" rather than an
    /// error.
    fn virtual_source_dims(
        &mut self,
        m: &VirtualMapping,
        access: DatasetAccess,
    ) -> IoResult<Option<Vec<u64>>> {
        let m = built_names(m, 0)?;
        Ok(self.source_dims(&m.source_file_name, &m.source_dset_name, access))
    }

    /// A printf mapping's `first_missing` and the blocks below it that
    /// actually have a source — upstream's search loop in
    /// `H5D__virtual_set_extent_unlim` (H5Dvirtual.c:1519-1614), which stops
    /// at the first block whose source cannot be opened and looks
    /// [`DatasetAccess::virtual_printf_gap`] blocks past it before giving up.
    ///
    /// The loop bound is upstream's `j <= printf_gap + first_missing`
    /// rearranged so a large gap cannot overflow the sum: `first_missing` is
    /// never above `j` when the test runs, because it only ever becomes the
    /// *previous* `j` plus one.
    fn printf_blocks_present(
        &mut self,
        m: &VirtualMapping,
        access: DatasetAccess,
    ) -> (u64, Vec<u64>) {
        let gap = access.effective_printf_gap();
        let mut first_missing = 0u64;
        let mut present = Vec::new();
        let mut j = 0u64;
        while j - first_missing <= gap {
            let Ok(built) = built_names(m, j) else {
                break;
            };
            if self
                .source_dims(&built.source_file_name, &built.source_dset_name, access)
                .is_some()
            {
                first_missing = j + 1;
                present.push(j);
            }
            j += 1;
        }
        (first_missing, present)
    }

    /// The extent of one named source dataset, or `None` when the file or
    /// the dataset in it cannot be opened.
    ///
    /// `access` is the virtual dataset's own: `H5D__virtual_init` copies the
    /// dapl into the layout as `source_dapl` (H5Dvirtual.c:2224-2226) and
    /// every source is opened with it (H5Dvirtual.c:901-902), so a source
    /// that is itself a virtual dataset resolves under the same view and
    /// printf gap.
    fn source_dims(
        &mut self,
        file_name: &str,
        dset_name: &str,
        access: DatasetAccess,
    ) -> Option<Vec<u64>> {
        let dset_name = dset_name.trim_start_matches('/');
        if file_name == "." {
            self.apply_dataset_access(dset_name, access).ok()?;
            return self
                .dataset_info_local(dset_name)
                .map(|i| i.dataspace.dims.clone());
        }
        let prefix = resolve_vdsfile_prefix(&self.source_dir);
        let full_path = combine_prefixed_path(prefix.as_deref(), file_name);
        let mut reader = Hdf5Reader::open_with_locking(&full_path, FileLocking::Disabled).ok()?;
        reader.apply_dataset_access(dset_name, access).ok()?;
        reader
            .dataset_info(dset_name)
            .map(|i| i.dataspace.dims.clone())
    }

    /// Fill `out` (shaped like the virtual dataset's own extent) by
    /// stitching each mapping's source bytes in order (`H5D__virtual_read`,
    /// H5Dvirtual.c). `out` must already be pre-filled with the tiled fill
    /// value — every element no mapping covers is left exactly as the
    /// caller filled it. Mappings apply in list order, so a later mapping's
    /// bytes win over an earlier one's on overlap, exactly like the C
    /// reader; an unlimited or printf mapping has already been replaced by
    /// the concrete mappings its open-time resolution made it
    /// (`H5D_VDS_LAST_AVAILABLE`, the default view — see
    /// [`Hdf5Reader::resolve_virtual_extents`]).
    ///
    /// A mapping whose source cannot be opened — the file is absent, or the
    /// dataset is not in it — contributes nothing and leaves its virtual
    /// region at the fill value, rather than failing the read.
    /// `H5D__virtual_open_source_dset` treats both as "no data there yet":
    /// it asks `H5F_prefix_open_file` to *try* the file and accepts a null
    /// one, and clears the error stack when the dataset is missing
    /// (H5Dvirtual.c:877-909); `H5D__virtual_read_one` then performs I/O
    /// "only ... if there is a projected memory space, otherwise there were
    /// no elements in the projection or the source dataset could not be
    /// opened" (H5Dvirtual.c:2661-2665).
    ///
    /// `depth` counts virtual-dataset nesting — a mapping whose source is
    /// itself a virtual dataset, possibly in another file — so a crafted
    /// cyclic mapping chain fails cleanly instead of recursing until the
    /// stack overflows.
    fn read_virtual_into(&mut self, name: &str, out: &mut [u8], depth: usize) -> IoResult<()> {
        if depth >= MAX_VIRTUAL_DEPTH {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset {name:?}: virtual dataset mapping nests {MAX_VIRTUAL_DEPTH} levels \
                 deep, aborting (possible cyclic mapping)"
            )));
        }
        let info = self
            .dataset_info(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let Some(mappings) = info.virtual_mappings.clone() else {
            // No mapping list written yet: every element is unmapped, and
            // `out` is already the fill value the caller pre-filled it with.
            return Ok(());
        };
        // Every unlimited mapping is replaced by the concrete one its
        // open-time resolution makes it, so the walk below only ever sees
        // bounded selections.
        let resolution = info.virtual_resolution.clone().unwrap_or_default();
        let mappings = concrete_virtual_mappings(&mappings, &resolution)?;
        let source_dir = self.source_dir.clone();
        // The same properties the extent resolved under reach every source
        // (H5Dvirtual.c:2224-2226, :901-902), so a source that is itself a
        // virtual dataset is read the same way this one is.
        let access = self.access_in_force(&self.canonical_path(name));

        // Cross-file source readers, opened at most once per distinct
        // resolved path for the duration of this call.
        let mut cross_file_cache: std::collections::HashMap<PathBuf, Hdf5Reader> =
            std::collections::HashMap::new();

        for mapping in &mappings {
            let virtual_boxes = mapping.virtual_selection.to_boxes(&dims).map_err(|e| {
                crate::io::IoError::InvalidState(format!(
                    "dataset {name:?}: virtual mapping's virtual selection is not \
                     supported: {e}"
                ))
            })?;
            if virtual_boxes.is_empty() {
                continue;
            }

            let source_name = mapping.source_dset_name.trim_start_matches('/');

            if mapping.source_file_name == "." {
                self.apply_dataset_access(source_name, access)?;
                let Some(src_dims) = self
                    .dataset_info(source_name)
                    .map(|i| i.dataspace.dims.clone())
                else {
                    continue;
                };
                let source_boxes = mapping.source_selection.to_boxes(&src_dims).map_err(|e| {
                    crate::io::IoError::InvalidState(format!(
                        "dataset {name:?}: virtual mapping's source selection is not \
                         supported: {e}"
                    ))
                })?;
                copy_matched_boxes(
                    |s, c, buf| self.read_slice_into_unconverted(source_name, s, c, buf, depth + 1),
                    &source_boxes,
                    &virtual_boxes,
                    &dims,
                    element_size,
                    out,
                )?;
            } else {
                let prefix = resolve_vdsfile_prefix(&source_dir);
                let full_path = combine_prefixed_path(prefix.as_deref(), &mapping.source_file_name);
                let cache_key =
                    std::fs::canonicalize(&full_path).unwrap_or_else(|_| full_path.clone());
                if !cross_file_cache.contains_key(&cache_key) {
                    let Ok(reader) =
                        Hdf5Reader::open_with_locking(&full_path, FileLocking::Disabled)
                    else {
                        continue;
                    };
                    cross_file_cache.insert(cache_key.clone(), reader);
                }
                let src_reader = cross_file_cache.get_mut(&cache_key).unwrap();
                src_reader.apply_dataset_access(source_name, access)?;
                let Some(src_dims) = src_reader
                    .dataset_info(source_name)
                    .map(|i| i.dataspace.dims.clone())
                else {
                    continue;
                };
                let source_boxes = mapping.source_selection.to_boxes(&src_dims).map_err(|e| {
                    crate::io::IoError::InvalidState(format!(
                        "dataset {name:?}: virtual mapping's source selection is not \
                         supported: {e}"
                    ))
                })?;
                copy_matched_boxes(
                    |s, c, buf| {
                        src_reader.read_slice_into_unconverted(source_name, s, c, buf, depth + 1)
                    },
                    &source_boxes,
                    &virtual_boxes,
                    &dims,
                    element_size,
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

        // The superblock extension can also have changed under SWMR (a new
        // free-space or shared-message table), so re-read it before the walk.
        let (meta, ext) = Self::read_extension_and_meta(
            &mut self.handle,
            ctx,
            self.meta.btree,
            sb.superblock_extension_address,
        )?;

        // Re-read root group object header, following continuation blocks.
        let root_header = Self::read_object_header_full(
            &mut self.handle,
            &meta,
            sb.root_group_object_header_address,
        )?;

        // Re-scan datasets, group attributes, group paths, and link records.
        let catalog = Self::build_catalog(
            &mut self.handle,
            &meta,
            Some(&root_header),
            sb.root_group_object_header_address,
            None,
        )?;

        // Root link storage from the freshly re-read header, the same way
        // `open_v2v3` derives it at open time — SWMR refresh is v2/v3-only,
        // so there is no symbol-table scratch-pad to fall back to here either.
        let root_link_storage = describe_link_storage(Some(&root_header), &meta.ctx, None);

        self._eof = sb.end_of_file_address;
        self.meta = meta;
        self.ext = ext;
        self.object_paths = catalog.object_paths(sb.root_group_object_header_address);
        self.datasets = catalog.datasets;
        self.unreadable = catalog.unreadable;
        self.root_link_storage = root_link_storage;
        self.group_attributes = catalog.group_attributes;
        self.group_link_storage = catalog.group_link_storage;
        self.group_paths = catalog.group_paths;
        self.group_aliases = catalog.group_aliases;
        self.links = catalog.links;
        self.datatypes = catalog.datatypes;
        // The catalog is freshly built, so every virtual dataset's resolved
        // extent went with the old one — a SWMR refresh is exactly when a
        // source may have grown.
        self.resolve_virtual_extents()?;

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
            .dataset_info_local(name)
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
                let geo = ChunkOutputGeometry {
                    dims: &dims,
                    chunk_dims,
                    element_size,
                };
                match target {
                    ChunkTarget::Full => self.copy_chunk_to_output(&data, output, &geo, &coords),
                    ChunkTarget::Slice { starts, counts } => {
                        self.copy_chunk_to_slice(&data, output, &geo, &coords, starts, counts)
                    }
                }
                Ok(())
            }
            data_layout::ChunkIndexType::Implicit => {
                self.read_chunked_implicit(name, chunk_dims, index_address, target, output)
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

                let geo = ChunkOutputGeometry {
                    dims: &dims,
                    chunk_dims,
                    element_size,
                };
                for (i, chunk_data) in decompressed.iter().enumerate() {
                    if let Some(data) = chunk_data {
                        let coords = chunk_coords(i as u64);
                        self.scatter_chunk(target, data, output, &geo, coords);
                    }
                }

                Ok(())
            }
        }
    }

    /// Collect a fixed-array dataset's per-chunk `(address, on-disk byte
    /// count, filter mask)` entries, indexed by index-grid linear slot
    /// ([`crate::io::chunk_grid`]). Empty when the index or its data block is
    /// unallocated.
    ///
    /// Shared by the full/slice chunked reader
    /// ([`read_chunked_fixed_array`](Self::read_chunked_fixed_array)) and the
    /// direct single-chunk read
    /// ([`read_chunk_raw_at`](Self::read_chunk_raw_at)), so the fixed-array
    /// wire format has one decoder.
    fn collect_fa_chunk_entries(
        &mut self,
        chunk_dims: &[u64],
        ndims: usize,
        element_size: u64,
        index_address: u64,
    ) -> IoResult<Vec<(u64, u64, u32)>> {
        use crate::format::chunk_index::fixed_array::*;

        if index_address == UNDEF_ADDR {
            // Unallocated: no chunks recorded.
            return Ok(Vec::new());
        }

        // Read FA header
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let fa_hdr = FixedArrayHeader::decode(&hdr_buf, &self.meta.ctx)?;

        if fa_hdr.data_blk_addr == UNDEF_ADDR {
            // Unallocated data block: no chunks recorded.
            return Ok(Vec::new());
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
        let sizeof_addr = self.meta.ctx.sizeof_addr as usize;
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
            let prefix = FixedArrayPagedPrefix::decode(&prefix_buf, &self.meta.ctx, npages)?;

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
                    let elems = decode_filtered_page(
                        &page_buf,
                        &self.meta.ctx,
                        page_nelmts,
                        chunk_size_len,
                    )?;
                    for e in elems {
                        chunk_entries.push((e.address, e.chunk_size, e.filter_mask));
                    }
                } else {
                    let addrs = decode_unfiltered_page(&page_buf, &self.meta.ctx, page_nelmts)?;
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
                    &self.meta.ctx,
                    num_elmts,
                    chunk_size_len,
                )?;
                for e in &fa_dblk.filtered_elements {
                    chunk_entries.push((e.address, e.chunk_size, e.filter_mask));
                }
            } else {
                let fa_dblk =
                    FixedArrayDataBlock::decode_unfiltered(&dblk_buf, &self.meta.ctx, num_elmts)?;
                for &addr in &fa_dblk.elements {
                    chunk_entries.push((addr, chunk_bytes, 0));
                }
            }
        }

        Ok(chunk_entries)
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
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let max_dims = info.dataspace.max_dims.clone();
        let ndims = dims.len();

        let chunk_entries =
            self.collect_fa_chunk_entries(chunk_dims, ndims, element_size, index_address)?;
        if chunk_entries.is_empty() {
            // Unallocated index/data block: `output` is already pre-filled.
            return Ok(());
        }
        let chunk_bytes: u64 = saturating_byte_len(chunk_dims, element_size);

        // Index-grid slot -> chunk-grid coordinates (row-major, against the
        // maximum extent — the array was sized from its chunk grid, so a slot
        // beyond the current extent still decodes to its true position and
        // then simply falls outside the read target). A zero chunk dimension
        // from a malformed layout message is rejected inside.
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
        let geo = ChunkOutputGeometry {
            dims: &dims,
            chunk_dims,
            element_size,
        };
        for (linear_idx, chunk_data) in decompressed.iter().enumerate() {
            let Some(data) = chunk_data else { continue };
            let coords = chunk_coords(linear_idx as u64);
            self.scatter_chunk(target, data, output, &geo, coords);
        }

        Ok(())
    }

    /// Read a dataset indexed by the implicit ("none") chunk index.
    ///
    /// There is no on-disk index structure at all (`H5Dnone.c`): every chunk
    /// slot in the maximum-extent grid is allocated in one block at dataset
    /// creation, so a chunk's address is purely arithmetic — `index_address +
    /// slot * chunk_bytes`, where `slot` is its row-major position in the
    /// same maximum-extent grid the fixed/extensible-array/v2-B-tree indexes
    /// use (`H5D__chunk_set_info_real`'s `max_down_chunks`). This index type
    /// is only ever selected for a fixed (non-unlimited) chunked dataset with
    /// early allocation and no filters, so there is no per-chunk allocation
    /// flag, compressed size, or filter mask to track.
    ///
    /// Scatters only; `output` must already be sized to the target extent and
    /// pre-filled with the tiled fill value by the caller.
    fn read_chunked_implicit(
        &mut self,
        name: &str,
        chunk_dims: &[u64],
        index_address: u64,
        target: ChunkTarget,
        output: &mut [u8],
    ) -> IoResult<()> {
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

        if chunk_dims.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "implicit-index dataset rank {} does not match chunk rank {}",
                ndims,
                chunk_dims.len()
            )));
        }

        let max_dims = info.dataspace.max_dims.clone();
        let chunk_bytes: u64 = saturating_byte_len(chunk_dims, element_size);
        let grid = crate::io::chunk_grid::index_grid(&dims, max_dims.as_deref(), chunk_dims)?;
        let chunks_total: u64 = grid.iter().fold(1u64, |acc, &n| acc.saturating_mul(n));

        // Every slot's coordinates and address are computed directly, not
        // read from disk, so there is no "unallocated chunk" case here the
        // way a sparse index has one: `at_most: false` because
        // `H5D__none_idx_create` guarantees the whole block is present.
        let mut slot_coords = Vec::with_capacity(chunks_total as usize);
        for i in 0..chunks_total {
            slot_coords.push(crate::io::chunk_grid::coords_of(
                &dims,
                max_dims.as_deref(),
                chunk_dims,
                i,
            )?);
        }

        let jobs: Vec<Option<ChunkReadJob>> = (0..chunks_total)
            .map(|i| {
                let coords = &slot_coords[i as usize];
                if !target.overlaps(coords, chunk_dims) {
                    None
                } else {
                    Some(ChunkReadJob {
                        addr: index_address + i * chunk_bytes,
                        len: chunk_bytes as usize,
                        at_most: false,
                        mask: 0,
                    })
                }
            })
            .collect();

        let decompressed = read_and_decompress_chunks(&self.handle, None, jobs)?;
        let geo = ChunkOutputGeometry {
            dims: &dims,
            chunk_dims,
            element_size,
        };
        for (i, chunk_data) in decompressed.iter().enumerate() {
            if let Some(data) = chunk_data {
                self.scatter_chunk(target, data, output, &geo, &slot_coords[i]);
            }
        }

        Ok(())
    }

    /// Collect a v2-B-tree-indexed dataset's per-chunk `(address, read
    /// size, scaled chunk-grid offsets, filter mask)` entries, walking the
    /// tree once. `read_size` is the compressed size for a filtered chunk,
    /// the full chunk size otherwise. Empty when the index has no records.
    ///
    /// Shared by the full/slice chunked reader
    /// ([`read_chunked_btree_v2`](Self::read_chunked_btree_v2)) and the
    /// direct single-chunk read
    /// ([`read_chunk_raw_at`](Self::read_chunk_raw_at)), so the v2 B-tree
    /// record format has one decoder.
    fn collect_bt2_chunk_entries(
        &mut self,
        chunk_dims: &[u64],
        ndims: usize,
        element_size: u64,
        index_address: u64,
    ) -> IoResult<Vec<Bt2ChunkEntry>> {
        use crate::format::chunk_index::btree_v2::*;

        if index_address == UNDEF_ADDR {
            // Unallocated: no chunks recorded.
            return Ok(Vec::new());
        }

        // Read BT2 header
        let hdr_buf = self.handle.read_at_most(index_address, 256)?;
        let bt2_hdr = Bt2Header::decode(&hdr_buf, &self.meta.ctx)?;

        if bt2_hdr.root_node_addr == UNDEF_ADDR || bt2_hdr.total_num_records == 0 {
            // No records.
            return Ok(Vec::new());
        }

        // Walk the B-tree to any depth, collecting every record's raw bytes
        // from the internal nodes and leaves.
        let ctx = self.meta.ctx;
        let record_bytes = collect_btree_v2_records(
            &bt2_hdr,
            &ctx,
            &mut HandleBlockReader {
                handle: &mut self.handle,
            },
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
        let entries: Vec<Bt2ChunkEntry> = if bt2_hdr.record_type == BT2_TYPE_CHUNK_UNFILT {
            Bt2ChunkIndex::decode_unfiltered_records(
                &record_bytes,
                total_records,
                ndims,
                &self.meta.ctx,
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
                &self.meta.ctx,
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

        Ok(entries)
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
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let ndims = dims.len();

        let entries =
            self.collect_bt2_chunk_entries(chunk_dims, ndims, element_size, index_address)?;
        if entries.is_empty() {
            // Unallocated index or no records: `output` is already pre-filled.
            return Ok(());
        }

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
        let geo = ChunkOutputGeometry {
            dims: &dims,
            chunk_dims,
            element_size,
        };
        for (i, chunk_data) in placed.iter().enumerate() {
            if let Some(data) = chunk_data {
                self.scatter_chunk(target, data, output, &geo, coords[i]);
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
            .dataset_info_local(name)
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
        let geo = ChunkOutputGeometry {
            dims: &dims,
            chunk_dims,
            element_size,
        };
        for (i, chunk_data) in placed.iter().enumerate() {
            if let Some(data) = chunk_data {
                self.scatter_chunk(target, data, output, &geo, &coords[i]);
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

        // A node is a fixed-size record: the header (8 + 2*sizeof_addr) plus
        // `2 * chunk_internal_k` interleaved keys/children.
        let sa = self.meta.ctx.sizeof_addr as usize;
        let node_size = self.meta.btree.chunk_btree_node_size(sa, rank);
        let buf = self.handle.read_at_most(addr, node_size)?;
        let node = ChunkBTreeV1Node::decode(&buf, sa, rank, self.meta.btree.chunk_max_entries())?;

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
        geo: &ChunkOutputGeometry,
        chunk_coords: &[u64],
    ) {
        let ChunkOutputGeometry {
            dims,
            chunk_dims,
            element_size,
        } = *geo;
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
    fn copy_chunk_to_slice(
        &self,
        chunk_data: &[u8],
        output: &mut [u8],
        geo: &ChunkOutputGeometry,
        chunk_coords: &[u64],
        starts: &[u64],
        counts: &[u64],
    ) {
        let ChunkOutputGeometry {
            dims,
            chunk_dims,
            element_size,
        } = *geo;
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
    fn scatter_chunk(
        &self,
        target: ChunkTarget,
        chunk_data: &[u8],
        output: &mut [u8],
        geo: &ChunkOutputGeometry,
        chunk_coords: &[u64],
    ) {
        match target {
            ChunkTarget::Full => self.copy_chunk_to_output(chunk_data, output, geo, chunk_coords),
            ChunkTarget::Slice { starts, counts } => {
                self.copy_chunk_to_slice(chunk_data, output, geo, chunk_coords, starts, counts)
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
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_vlen_objects(&path);
        }
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let layout = info.layout.clone();
        let external_files = info.external_files.clone();
        let total_elements: u64 = dims.iter().fold(1u64, |acc, &d| acc.saturating_mul(d));

        let raw = match &layout {
            DataLayoutMessage::Contiguous { size, .. } if !external_files.is_empty() => {
                let prefix = resolve_extfile_prefix(&self.source_dir);
                let mut buf = vec![0u8; *size as usize];
                read_external_file_bytes(&external_files, prefix.as_deref(), 0, &mut buf)?;
                buf
            }
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

        let ref_size = vlen_reference_size(&self.meta.ctx);
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
                decode_vlen_reference(&raw[offset..], &self.meta.ctx)?;

            if collection_addr == UNDEF_ADDR || collection_addr == 0 {
                items.push(Vec::new());
                continue;
            }

            // Read or get cached global heap collection
            if let std::collections::hash_map::Entry::Vacant(e) = heap_cache.entry(collection_addr)
            {
                let coll = self.read_heap_collection(collection_addr)?;
                let lookup: std::collections::HashMap<u16, usize> = coll
                    .objects
                    .iter()
                    .enumerate()
                    .map(|(i, o)| (o.index, i))
                    .collect();
                e.insert((coll, lookup));
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
        let ea_hdr = ExtensibleArrayHeader::decode(&hdr_buf, &self.meta.ctx)?;
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
            ea_hdr.raw_elmt_size - self.meta.ctx.sizeof_addr - 4
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
                &self.meta.ctx,
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
                &self.meta.ctx,
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
        let sa = self.meta.ctx.sizeof_addr as usize;
        let raw_elmt_size = if is_filtered {
            ea::FilteredChunkEntry::raw_size(self.meta.ctx.sizeof_addr, chunk_size_len) as usize
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
                        &self.meta.ctx,
                        max_nelmts_bits,
                        s.ndblks as usize,
                        page_init_total,
                    )?;
                    (sb.dblk_addrs, sb.page_init)
                }
            };

            let npages = geo.npages(u) as usize;
            let page_size = geo.dblk_page_size(raw_elmt_size);
            let prefix = geo.dblk_prefix_size(self.meta.ctx.sizeof_addr, max_nelmts_bits);

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
                        &self.meta.ctx,
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
                        &self.meta.ctx,
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
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_slice(&path, starts, counts);
        }
        let (datatype, out_bytes) = self.slice_size_and_datatype(name, counts)?;
        let mut data = alloc_tiled_fill(out_bytes as usize, None)?;
        self.read_slice_into_unconverted(name, starts, counts, &mut data, 0)?;
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
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_slice_into(&path, starts, counts, out);
        }
        let (datatype, out_bytes) = self.slice_size_and_datatype(name, counts)?;
        if out.len() as u64 != out_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "read_slice_into: buffer is {} bytes but selection needs {}",
                out.len(),
                out_bytes
            )));
        }
        self.read_slice_into_unconverted(name, starts, counts, out, 0)?;
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
            .dataset_info_local(name)
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
    /// `out.len()` must equal `product(counts) * element_size`. `depth`
    /// counts virtual-dataset nesting for a caller reached through
    /// [`read_virtual_into`](Self::read_virtual_into); pass `0` for a
    /// top-level call.
    fn read_slice_into_unconverted(
        &mut self,
        name: &str,
        starts: &[u64],
        counts: &[u64],
        out: &mut [u8],
        depth: usize,
    ) -> IoResult<()> {
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let layout = info.layout.clone();
        let pipeline = info.filter_pipeline.clone();
        let fill_value = info.fill_value.clone();
        let external_files = info.external_files.clone();
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
            DataLayoutMessage::Contiguous { .. } if !external_files.is_empty() => {
                // Same coalesced run geometry as the normal contiguous case
                // below, but each run is read through the external file
                // list instead of straight from this file (H5D__efl_read,
                // H5Defl.c) — `src_off` is already dataset-relative, which
                // is exactly what `read_external_file_bytes` walks slots by.
                let prefix = resolve_extfile_prefix(&self.source_dir);
                for_each_contiguous_run(
                    &dims,
                    starts,
                    counts,
                    element_size,
                    |src_off, out_off, len| {
                        read_external_file_bytes(
                            &external_files,
                            prefix.as_deref(),
                            src_off,
                            &mut out[out_off..out_off + len],
                        )
                    },
                )?;
            }
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
            DataLayoutMessage::Virtual { .. } => {
                // Stitch the full virtual image, then extract the
                // requested region from it. This reads more than the
                // selection strictly needs (no per-mapping intersection
                // against the caller's box), but a virtual dataset's data
                // is composed from other datasets rather than stored
                // contiguously, so there is no cheaper selective path
                // without duplicating `read_virtual_into`'s mapping walk
                // for a bounded region — correctness, not I/O pruning, is
                // what a VDS slice read needs here.
                fill_tiled_into(out, fill_value.as_deref());
                let total = saturating_byte_len(&dims, element_size) as usize;
                let mut full = alloc_tiled_fill(total, fill_value.as_deref())?;
                self.read_virtual_into(name, &mut full, depth)?;
                for_each_contiguous_run(
                    &dims,
                    starts,
                    counts,
                    element_size,
                    |src_off, out_off, len| {
                        let src_off = src_off as usize;
                        out[out_off..out_off + len].copy_from_slice(&full[src_off..src_off + len]);
                        Ok(())
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Read a strided hyperslab — h5py's stepped slicing (`ds[a:b:s]`) or
    /// the general `start`/`stride`/`count`/`block` form of
    /// `H5Sselect_hyperslab` — into a returned buffer.
    ///
    /// One tuple per dimension: `start[d]` is the first index, `stride[d]`
    /// the spacing between selected blocks (`1` = the classic contiguous
    /// selection [`read_slice`](Self::read_slice) reads), `count[d]` how
    /// many blocks, and `block[d]` how many contiguous elements each block
    /// covers. The returned buffer is row-major over `count[d] * block[d]`
    /// per dimension — exactly the shape h5py's stepped slicing produces.
    ///
    /// Built on the same selection-decomposition primitives the virtual
    /// dataset reader uses for its per-mapping scatter
    /// ([`Selection::to_boxes`], [`copy_matched_boxes`]) rather than a
    /// second box walker: the requested selection and a densely-packed
    /// "output" selection sharing the same `count` decompose into the same
    /// number of same-shaped boxes in the same row-major order, so each
    /// source box is read with the ordinary per-layout selective read
    /// ([`read_slice_into_unconverted`](Self::read_slice_into_unconverted))
    /// and scattered straight into its matching output box.
    pub fn read_hyperslab(
        &mut self,
        name: &str,
        start: &[u64],
        stride: &[u64],
        count: &[u64],
        block: &[u64],
    ) -> IoResult<Vec<u8>> {
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_hyperslab(&path, start, stride, count, block);
        }
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let datatype = info.datatype.clone();
        let element_size = datatype.element_size() as u64;
        let rank = dims.len();

        if start.len() != rank || stride.len() != rank || count.len() != rank || block.len() != rank
        {
            return Err(crate::io::IoError::InvalidState(
                "start/stride/count/block length must match dataset rank".into(),
            ));
        }
        if stride.contains(&0) {
            return Err(crate::io::IoError::InvalidState(
                "hyperslab stride must be nonzero in every dimension".into(),
            ));
        }

        let src_sel = Selection::Hyperslab {
            rank,
            form: Hyperslab::Regular(RegularHyperslab {
                start: start.to_vec(),
                stride: stride.to_vec(),
                count: count.to_vec(),
                block: block.to_vec(),
            }),
        };
        let out_dims: Vec<u64> = (0..rank)
            .map(|d| count[d].saturating_mul(block[d]))
            .collect();
        // A densely-packed selection sharing the same `count`: its boxes
        // decompose in the same row-major order as `src_sel`'s (the nested
        // loop `Selection::to_boxes` drives is indexed purely by `count`),
        // so `src_boxes[i]` and `dst_boxes[i]` are always the same logical
        // block, letting `copy_matched_boxes` pair them positionally.
        let dst_sel = Selection::Hyperslab {
            rank,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0u64; rank],
                stride: block.to_vec(),
                count: count.to_vec(),
                block: block.to_vec(),
            }),
        };
        let src_boxes = src_sel.to_boxes(&dims)?;
        let dst_boxes = dst_sel.to_boxes(&out_dims)?;

        let total = saturating_byte_len(&out_dims, element_size) as usize;
        let mut out = alloc_tiled_fill(total, None)?;
        copy_matched_boxes(
            |bstart, bcount, buf| self.read_slice_into_unconverted(name, bstart, bcount, buf, 0),
            &src_boxes,
            &dst_boxes,
            &out_dims,
            element_size,
            &mut out,
        )?;
        Self::apply_post_filter_conversion(&mut out, &datatype)?;
        Ok(out)
    }

    /// Read a list of coordinates in one call — h5py fancy indexing with a
    /// coordinate list (`H5S_SEL_POINTS`) — into a returned buffer.
    ///
    /// `points[i]` is a `rank`-length coordinate; the returned buffer holds
    /// one element per point, `element_size` bytes each, in the same order
    /// as `points` (point selection order is significant, see
    /// [`PointSelection`]). Backed by [`Selection::Points`] and
    /// [`Selection::to_boxes`] — the same decomposition
    /// [`read_hyperslab`](Self::read_hyperslab) and the virtual dataset
    /// reader use — each point's 1-element box is read with the ordinary
    /// per-layout selective read
    /// ([`read_slice_into_unconverted`](Self::read_slice_into_unconverted)).
    /// A 1-element box is already a flat `element_size`-byte run, so
    /// placing it needs no further run-decomposition
    /// ([`for_each_dual_run`] would degenerate to exactly this copy).
    pub fn read_points(&mut self, name: &str, points: &[Vec<u64>]) -> IoResult<Vec<u8>> {
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_points(&path, points);
        }
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let datatype = info.datatype.clone();
        let element_size = datatype.element_size() as u64;
        let rank = dims.len();

        for p in points {
            if p.len() != rank {
                return Err(crate::io::IoError::InvalidState(format!(
                    "point coordinate has {} entries but the dataset has {} dimensions",
                    p.len(),
                    rank
                )));
            }
        }

        let sel = Selection::Points(PointSelection {
            rank,
            points: points.to_vec(),
        });
        let boxes = sel.to_boxes(&dims)?;

        let es = element_size as usize;
        let mut out = alloc_tiled_fill(points.len() * es, None)?;
        for (i, (bstart, bcount)) in boxes.iter().enumerate() {
            self.read_slice_into_unconverted(
                name,
                bstart,
                bcount,
                &mut out[i * es..(i + 1) * es],
                0,
            )?;
        }
        Self::apply_post_filter_conversion(&mut out, &datatype)?;
        Ok(out)
    }

    /// Read one chunk's raw (still-filtered) bytes and its filter mask —
    /// the read half of `H5Dread_chunk` (h5py:
    /// `Dataset.id.read_direct_chunk`).
    ///
    /// `chunk_coords` is the chunk's position in the chunk grid, one
    /// coordinate per dimension counted in chunks (not elements) — the same
    /// addressing [`write_chunk_raw_at`](crate::Dataset::write_chunk_raw_at)
    /// uses on the write side. The bytes returned are exactly what is
    /// stored on disk: filtered/compressed if the dataset has a filter
    /// pipeline, with no decompression applied — the caller runs the
    /// pipeline itself (honoring the returned mask, which marks any filter
    /// this particular chunk skipped) if it wants decoded data.
    ///
    /// Resolved through whichever chunk index the dataset uses, reusing the
    /// same per-index decoders the full/slice chunked reader walks
    /// ([`collect_fa_chunk_entries`](Self::collect_fa_chunk_entries),
    /// [`collect_ea_chunk_entries`](Self::collect_ea_chunk_entries),
    /// [`collect_bt2_chunk_entries`](Self::collect_bt2_chunk_entries),
    /// [`collect_btree_v1_chunks`](Self::collect_btree_v1_chunks)) rather
    /// than a new index walker.
    ///
    /// `Err` when the dataset is not chunked, `chunk_coords` has the wrong
    /// rank, or the chunk at those coordinates has never been written.
    pub fn read_chunk_raw_at(
        &mut self,
        name: &str,
        chunk_coords: &[u64],
    ) -> IoResult<(Vec<u8>, u32)> {
        if self.external_edge(name).is_some() {
            let (owner, path, _) = self.external_owner(name, MAX_EXTERNAL_HOPS)?;
            return owner.read_chunk_raw_at(&path, chunk_coords);
        }
        let info = self
            .dataset_info_local(name)
            .ok_or_else(|| crate::io::IoError::NotFound(name.to_string()))?;
        let dims = info.dataspace.dims.clone();
        let max_dims = info.dataspace.max_dims.clone();
        let element_size = info.datatype.element_size() as u64;
        let layout = info.layout.clone();
        let ndims = dims.len();

        if chunk_coords.len() != ndims {
            return Err(crate::io::IoError::InvalidState(format!(
                "chunk_coords has {} entries but the dataset has {} dimensions",
                chunk_coords.len(),
                ndims
            )));
        }

        let not_written = || {
            crate::io::IoError::InvalidState(format!(
                "chunk at coordinates {chunk_coords:?} has not been written"
            ))
        };

        match &layout {
            DataLayoutMessage::ChunkedV3 {
                chunk_dims,
                b_tree_address,
            } => {
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                if *b_tree_address == UNDEF_ADDR {
                    return Err(not_written());
                }
                let file_size = self.handle.file_size()?;
                let mut entries = Vec::new();
                self.collect_btree_v1_chunks(*b_tree_address, ndims, file_size, 0, &mut entries)?;
                for (offsets, addr, chunk_size, mask) in &entries {
                    if *addr == UNDEF_ADDR {
                        continue;
                    }
                    let mut scaled = Vec::with_capacity(ndims);
                    for d in 0..ndims {
                        scaled.push(offsets[d].checked_div(real_chunk_dims[d]).unwrap_or(0));
                    }
                    if scaled.as_slice() == chunk_coords {
                        return Ok((self.handle.read_at(*addr, *chunk_size as usize)?, *mask));
                    }
                }
                Err(not_written())
            }
            DataLayoutMessage::ChunkedV4 {
                chunk_dims,
                index_type,
                index_address,
                earray_params,
                single_chunk_filter,
                ..
            } => {
                let real_chunk_dims = &chunk_dims[..chunk_dims.len() - 1];
                match index_type {
                    data_layout::ChunkIndexType::SingleChunk => {
                        if chunk_coords.iter().any(|&c| c != 0) {
                            return Err(crate::io::IoError::InvalidState(format!(
                                "chunk coordinates {chunk_coords:?} are outside the chunk \
                                 grid (0..1): this dataset has a single-chunk index"
                            )));
                        }
                        if *index_address == UNDEF_ADDR {
                            return Err(not_written());
                        }
                        match single_chunk_filter {
                            Some(scf) => Ok((
                                self.handle.read_at(*index_address, scf.nbytes as usize)?,
                                scf.filter_mask,
                            )),
                            None => {
                                let total = saturating_byte_len(&dims, element_size);
                                Ok((self.handle.read_at(*index_address, total as usize)?, 0))
                            }
                        }
                    }
                    data_layout::ChunkIndexType::Implicit => {
                        if *index_address == UNDEF_ADDR {
                            return Err(not_written());
                        }
                        let linear = crate::io::chunk_grid::linear_index(
                            &dims,
                            max_dims.as_deref(),
                            real_chunk_dims,
                            chunk_coords,
                        )?;
                        let chunk_bytes = saturating_byte_len(real_chunk_dims, element_size);
                        let addr = index_address.saturating_add(linear.saturating_mul(chunk_bytes));
                        Ok((self.handle.read_at(addr, chunk_bytes as usize)?, 0))
                    }
                    data_layout::ChunkIndexType::FixedArray => {
                        let linear = crate::io::chunk_grid::linear_index(
                            &dims,
                            max_dims.as_deref(),
                            real_chunk_dims,
                            chunk_coords,
                        )?;
                        let entries = self.collect_fa_chunk_entries(
                            real_chunk_dims,
                            ndims,
                            element_size,
                            *index_address,
                        )?;
                        match entries.get(linear as usize) {
                            Some(&(addr, size, mask)) if addr != UNDEF_ADDR => {
                                Ok((self.handle.read_at(addr, size as usize)?, mask))
                            }
                            _ => Err(not_written()),
                        }
                    }
                    data_layout::ChunkIndexType::ExtensibleArray => {
                        let params = earray_params.as_ref().ok_or_else(|| {
                            crate::io::IoError::InvalidState("missing earray params".into())
                        })?;
                        let linear = crate::io::chunk_grid::linear_index(
                            &dims,
                            max_dims.as_deref(),
                            real_chunk_dims,
                            chunk_coords,
                        )?;
                        let entries = self.collect_ea_chunk_entries(
                            *index_address,
                            params,
                            &dims,
                            max_dims.as_deref(),
                            real_chunk_dims,
                            element_size,
                        )?;
                        match entries.get(linear as usize) {
                            Some(&(addr, size, mask)) if addr != UNDEF_ADDR => {
                                Ok((self.handle.read_at(addr, size as usize)?, mask))
                            }
                            _ => Err(not_written()),
                        }
                    }
                    data_layout::ChunkIndexType::BTreeV2 => {
                        let entries = self.collect_bt2_chunk_entries(
                            real_chunk_dims,
                            ndims,
                            element_size,
                            *index_address,
                        )?;
                        match entries
                            .iter()
                            .find(|(_, _, scaled, _)| scaled.as_slice() == chunk_coords)
                        {
                            Some(&(addr, size, _, mask)) if addr != UNDEF_ADDR => {
                                Ok((self.handle.read_at(addr, size)?, mask))
                            }
                            _ => Err(not_written()),
                        }
                    }
                }
            }
            _ => Err(crate::io::IoError::InvalidState(
                "read_chunk_raw_at is only for chunked datasets".into(),
            )),
        }
    }
}

/// Adapts a `FileHandle` to the `BlockReader` trait used by the fractal-heap
/// walker, so heap blocks can be fetched from the open file.
///
/// The handle is shared, not exclusive: every read goes through
/// `FileHandle::read_at_most`, which takes `&self`, so the writer can walk a
/// structure it is about to free while holding only `&self` itself.
pub(crate) struct HandleBlockReader<'a> {
    pub(crate) handle: &'a FileHandle,
}

/// One object's attribute set, or the reason it could not be read whole.
///
/// [`AttributeEntry`] carries a per-attribute failure, which needs a name to
/// hang on. Two failures have none. A dense set is indexed by name *hash*, so
/// a heap or index that will not read yields no names at all; and an attribute
/// message too damaged to yield its own name cannot be listed under one
/// either. The only listing that can report those honestly is the object's, so
/// this type carries the object-scope reason beside the entries, and every
/// accessor that would present the set as whole returns the reason instead.
///
/// The entries are deliberately unreachable while the set is incomplete: the
/// only way out is [`Self::into_complete`], which refuses. That is what keeps
/// the writer from rebuilding an object header out of a partial set and
/// deleting the attributes it never saw.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectAttributes {
    entries: Vec<AttributeEntry>,
    incomplete: Option<String>,
    /// This object's own attribute creation-order policy, from the header
    /// flag bits `H5Pget_attr_creation_order` reads back
    /// (`attribute_creation_order`) — a structural fact about the header,
    /// known even when the entries above are `incomplete`.
    creation_order: CreationOrder,
    /// This object's own compact-vs-dense attribute storage, from the
    /// attribute info message's heap address (or its absence) — likewise a
    /// structural fact known even when the entries are `incomplete`.
    storage: AttributeStorage,
}

impl ObjectAttributes {
    /// Record an attribute this collector could name.
    fn push(&mut self, entry: AttributeEntry) {
        self.entries.push(entry);
    }

    /// Record that part of the set could not be read. The first reason stands:
    /// it is the one that explains the earliest missing attributes.
    fn mark_incomplete(&mut self, reason: String) {
        if self.incomplete.is_none() {
            self.incomplete = Some(reason);
        }
    }

    /// Why this object's attributes cannot be listed, or `None` when the set
    /// is whole. Individual entries in a whole set may still be undecodable —
    /// [`AttributeEntry::unreadable_reason`] answers for those.
    pub fn unreadable_reason(&self) -> Option<&str> {
        self.incomplete.as_deref()
    }

    /// This object's own attribute creation-order policy —
    /// `H5Pget_attr_creation_order`'s answer, read off the object header's
    /// own flag bits rather than derived from the entries. Available even
    /// when the set is [`incomplete`](Self::unreadable_reason): it names
    /// nothing that failed to decode.
    pub fn creation_order(&self) -> CreationOrder {
        self.creation_order
    }

    /// This object's own compact-vs-dense attribute storage — h5py's
    /// `h5o.get_info(...).meta_size.attr.index_size` check, read off the
    /// attribute info message's heap address rather than derived from the
    /// entries. Available even when the set is
    /// [`incomplete`](Self::unreadable_reason).
    pub fn storage(&self) -> AttributeStorage {
        self.storage
    }

    /// This object's own attribute count as `H5Oget_info().num_attrs`
    /// reports it — the object header's count, not necessarily the same
    /// enumeration path as [`ordered_names`](Self::ordered_names).
    ///
    /// `H5O__attr_count_real` derives this from the attribute info message
    /// when the header carries one (the dense name-index record count, or
    /// the compact message count `H5O__attr_open_by_idx` already counted
    /// while building it) and from the raw attribute-message envelope count
    /// otherwise. Both reduce to the number of attributes this collector
    /// successfully names: a conformant writer creates the info message
    /// exactly when it has attributes to report through it, so a whole set's
    /// length already equals what libhdf5's header-count algorithm answers,
    /// without replaying its v1-header/v2-header branch here.
    pub fn header_count(&self, owner: &str) -> IoResult<u64> {
        Ok(self.complete(owner)?.len() as u64)
    }

    /// The entries, once the set is known to be whole.
    ///
    /// The sole route from an `ObjectAttributes` to an owned entry list. A
    /// caller that rewrites the object header — the append path — must take
    /// this route, so an unread set stops the rewrite instead of erasing the
    /// attributes behind it.
    pub(crate) fn into_complete(self, owner: &str) -> IoResult<Vec<AttributeEntry>> {
        match self.incomplete {
            Some(reason) => Err(incomplete_error(owner, &reason)),
            None => Ok(self.entries),
        }
    }

    /// The entries, once the set is known to be whole, borrowed.
    fn complete(&self, owner: &str) -> IoResult<&[AttributeEntry]> {
        match &self.incomplete {
            Some(reason) => Err(incomplete_error(owner, reason)),
            None => Ok(&self.entries),
        }
    }

    /// This object's attribute names, once the set is known to be whole, in
    /// the order h5py's default iteration produces them: creation order when
    /// the object tracks it, name order otherwise
    /// (`H5A__compact_cmp_corder`/`H5A__compact_cmp_name` for compact
    /// storage, the matching v2 B-tree index for dense) — never the physical
    /// order the entries happen to sit in, which `entries` otherwise
    /// preserves for the writer's rewrite path.
    pub(crate) fn ordered_names(&self, owner: &str) -> IoResult<Vec<String>> {
        let mut ordered: Vec<&AttributeEntry> = self.complete(owner)?.iter().collect();
        if !ordered.is_empty() && ordered.iter().all(|e| e.creation_index().is_some()) {
            ordered.sort_by_key(|e| e.creation_index());
        } else {
            ordered.sort_by(|a, b| a.name().cmp(b.name()));
        }
        Ok(ordered.into_iter().map(|e| e.name().to_string()).collect())
    }
}

/// The one wording for "this object's attributes are not all here".
///
/// `Unsupported`, the same variant an undecodable dataset message raises: the
/// name is in the listing and the content is out of reach, which is what the
/// variant is for. Wrapping a `FormatError` here instead would put the same
/// condition behind two different public variants.
fn incomplete_error(owner: &str, reason: &str) -> crate::io::IoError {
    crate::io::IoError::Unsupported(format!(
        "attributes of '{owner}' cannot be read whole: {reason}"
    ))
}

/// Every attribute attached to an object, whichever storage it uses.
///
/// This is the only place attributes are pulled off an object header — reader
/// and writer alike. Compact storage keeps them as `Attribute` messages in the
/// header itself; once an object crosses the phase-change threshold libhdf5
/// moves *all* of them into a fractal heap named by the `Attribute Info`
/// message and leaves no attribute message behind
/// (`H5Oattribute.c::H5O__attr_create`). Scanning only the messages therefore
/// reports zero attributes for a dense object: a silent loss on read, and a
/// silent deletion when the writer rebuilds that object's header from what it
/// collected.
///
/// An attribute this crate cannot decode is kept, named, with the reason
/// attached: a listing that omitted it would report a file that does not
/// contain it. What cannot be named at all — a damaged attribute message, an
/// attribute info message that will not decode, a dense set whose heap or name
/// index will not read — marks the whole set incomplete, so the object reports
/// the failure rather than a short list.
pub(crate) fn collect_object_attributes(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    header: &ObjectHeader,
) -> ObjectAttributes {
    let mut attrs = ObjectAttributes {
        creation_order: header.attribute_creation_order(),
        ..ObjectAttributes::default()
    };
    // The message envelope carries a creation index only when the header says
    // the object tracks one; the field is not even encoded otherwise
    // (`H5O_SIZEOF_MSGHDR_OH`), so reading it as an index would report zero
    // for every attribute of an untracked object.
    let tracked = header.has_creation_order();
    for msg in &header.messages {
        match msg.msg_type {
            MSG_ATTRIBUTE => match AttributeEntry::parse(&msg.data, ctx) {
                Ok(entry) => {
                    attrs.push(entry.with_creation_index(tracked.then_some(msg.creation_index)))
                }
                Err(e) => attrs.mark_incomplete(format!("an attribute message is unreadable: {e}")),
            },
            MSG_ATTR_INFO => match AttributeInfoMessage::decode(&msg.data, ctx) {
                Ok((info, _)) => {
                    attrs.storage = if info.is_dense() {
                        AttributeStorage::Dense
                    } else {
                        AttributeStorage::Compact
                    };
                    let mut br = HandleBlockReader { handle };
                    match crate::format::dense_attr::read_dense_attributes(&info, ctx, &mut br) {
                        Ok(dense) => attrs.entries.extend(dense),
                        Err(e) => attrs
                            .mark_incomplete(format!("dense attribute storage is unreadable: {e}")),
                    }
                }
                Err(e) => {
                    attrs.mark_incomplete(format!("the attribute info message is unreadable: {e}"))
                }
            },
            _ => {}
        }
    }
    attrs
}

impl BlockReader for HandleBlockReader<'_> {
    fn read_block(&mut self, offset: u64, len: usize) -> crate::format::FormatResult<Vec<u8>> {
        // `read_at_most`, not `read_at`: a metadata block allocated at the end
        // of the file can be shorter on disk than its nominal size, and every
        // decoder re-checks the length it needs.
        self.handle.read_at_most(offset, len).map_err(|e| {
            crate::format::FormatError::InvalidData(format!(
                "metadata block read failed at {:#x}: {}",
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

    /// Run the collector against a header built by hand. The handle is only
    /// touched when a message sends the collector to the heap, which none of
    /// these do, so an empty file is enough of a file.
    fn collect_from(
        messages: Vec<crate::format::object_header::ObjectHeaderMessage>,
    ) -> Result<Vec<String>, String> {
        let path = temp_path("collect");
        std::fs::File::create(&path).unwrap();
        let mut handle = FileHandle::open_read(&path).unwrap();
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let header = ObjectHeader {
            flags: 0x02,
            times: None,
            messages,
        };
        let attrs = collect_object_attributes(&mut handle, &ctx, &header);
        drop(handle);
        let _ = std::fs::remove_file(&path);
        attrs
            .complete("obj")
            .map(|e| e.iter().map(|a| a.name().to_string()).collect())
            .map_err(|e| e.to_string())
    }

    fn msg(msg_type: u8, data: Vec<u8>) -> crate::format::object_header::ObjectHeaderMessage {
        crate::format::object_header::ObjectHeaderMessage {
            msg_type,
            flags: 0,
            data,
            creation_index: 0,
        }
    }

    /// An attribute info message that will not decode takes the object's whole
    /// attribute set with it — the dense storage it names is where the
    /// attributes are. The listing must say so rather than come back short.
    #[test]
    fn an_undecodable_attribute_info_message_fails_the_listing() {
        // Version 9: `H5O_AINFO_VERSION_0` is the only one that exists.
        let err = collect_from(vec![msg(MSG_ATTR_INFO, vec![9, 0])]).unwrap_err();
        assert!(
            err.contains("attributes of 'obj' cannot be read whole")
                && err.contains("attribute info message"),
            "{err}"
        );
    }

    /// An attribute message damaged past its own name has no name to be listed
    /// under, so it too is an object-level failure — the one case
    /// [`AttributeEntry::parse`] cannot name.
    #[test]
    fn an_unnameable_attribute_message_fails_the_listing() {
        let err = collect_from(vec![msg(MSG_ATTRIBUTE, vec![1, 0, 0])]).unwrap_err();
        assert!(
            err.contains("attributes of 'obj' cannot be read whole")
                && err.contains("attribute message"),
            "{err}"
        );
    }

    /// The failure is the object's, not one name's: a set that reads whole
    /// still lists, and a compact attribute info message is not dense storage.
    #[test]
    fn a_whole_attribute_set_still_lists() {
        let ctx = FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let ainfo = crate::format::messages::attr_info::AttributeInfoMessage::compact();
        let names = collect_from(vec![msg(MSG_ATTR_INFO, ainfo.encode(&ctx))]).unwrap();
        assert!(names.is_empty(), "{names:?}");
    }

    /// A space-padded fixed-length string attribute reads back without its
    /// padding: `H5T__conv_s_s` ends the value after the last non-space byte,
    /// and nothing else in the element marks where it stops.
    #[test]
    fn fixed_string_attr_value_honors_the_declared_pad() {
        use crate::format::messages::dataspace::DataspaceMessage;
        use crate::format::messages::datatype::DatatypeMessage;

        let attr = |padding: u8, data: &[u8]| AttributeMessage {
            name: "units".to_string(),
            datatype: DatatypeMessage::FixedString {
                size: data.len() as u32,
                padding,
                charset: 0,
            },
            dataspace: DataspaceMessage::scalar(),
            data: data.to_vec(),
        };

        // Space padded: no NUL anywhere, so truncating at the first NUL kept
        // the padding.
        assert_eq!(
            fixed_string_attr_value(&attr(2, b"volt    ")).unwrap(),
            "volt"
        );
        // Null terminated and null padded end at the first NUL, so trailing
        // spaces before it are content.
        assert_eq!(
            fixed_string_attr_value(&attr(0, b"volt  \0\0")).unwrap(),
            "volt  "
        );
        assert_eq!(
            fixed_string_attr_value(&attr(1, b"volt\0\0\0\0")).unwrap(),
            "volt"
        );
        // A reserved rule is named rather than guessed at.
        let err = fixed_string_attr_value(&attr(7, b"volt    ")).unwrap_err();
        assert!(
            err.to_string().contains("padding rule 7"),
            "unexpected error: {err}"
        );
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
            BTreeV1Config::default().snode_max_entries(),
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
                BTreeV1Config::default().sym_leaf_max_entries(),
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
