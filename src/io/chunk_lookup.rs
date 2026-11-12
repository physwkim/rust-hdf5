//! Point lookup of one chunk in a dataset's chunk index.
//!
//! Every chunked read asks one question per chunk it touches — where is the
//! chunk at these grid coordinates stored, at what size, under which filter
//! mask — and [`find`] is the only place that answers it, for every index type
//! libhdf5 writes. A lookup decodes only the index blocks on the path to its
//! chunk, the way `H5D__chunk_lookup` does, so reading one frame of a growing
//! dataset costs a few block reads whatever the number of chunks already
//! written. Blocks a lookup decodes stay in a [`BlockCache`] that lives as long
//! as the catalog entry it was read against, so a run of lookups through one
//! data block decodes it once.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use crate::format::btree_v1::{BTreeV1Config, ChunkBTreeV1Node};
use crate::format::chunk_index::btree_v2::{
    Bt2ChunkIndex, Bt2Geometry, Bt2Header, Bt2InternalNode, Bt2LeafNode, BT2_TYPE_CHUNK_UNFILT,
};
use crate::format::chunk_index::extensible_array::{
    self as ea, EaDblkPath, EaGeometry, EaLoc, ExtensibleArrayDataBlock, ExtensibleArrayHeader,
    ExtensibleArrayIndexBlock, ExtensibleArraySuperBlock,
};
use crate::format::chunk_index::fixed_array::{
    self as fa, FixedArrayDataBlock, FixedArrayHeader, FixedArrayPagedPrefix,
};
use crate::format::messages::data_layout::{
    ChunkIndexType, DataLayoutMessage, EarrayParams, SingleChunkFilter,
};
use crate::format::{FormatContext, FormatResult, UNDEF_ADDR};
use crate::io::chunk_grid;
use crate::io::file_handle::FileHandle;
use crate::io::{FileMeta, IoError, IoResult};

/// Where one chunk is stored, as its index records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkRecord {
    pub(crate) addr: u64,
    /// The stored byte count: the filtered size the index records for a
    /// filtered chunk, the full chunk size for an unfiltered one. A filtered
    /// fixed-array entry may record 0, meaning "not recorded".
    pub(crate) nbytes: u64,
    /// Filters this chunk skipped (bit `i` = filter `i`); 0 when unfiltered.
    pub(crate) mask: u32,
}

/// Which index structure a chunked dataset uses, with what the layout message
/// says about it beyond its address.
#[derive(Debug, Clone)]
pub(crate) enum IndexKind {
    /// The index address is the chunk itself; the filter parameters, when the
    /// layout records them, are its stored size and mask.
    SingleChunk(Option<SingleChunkFilter>),
    Implicit,
    FixedArray,
    ExtensibleArray(EarrayParams),
    BTreeV2,
    BTreeV1,
}

/// One chunked dataset's index, as the catalog entry describes it: what a
/// lookup needs besides the file and the blocks already decoded.
#[derive(Debug, Clone)]
pub(crate) struct ChunkIndexView {
    kind: IndexKind,
    addr: u64,
    dims: Vec<u64>,
    max_dims: Option<Vec<u64>>,
    chunk_dims: Vec<u64>,
    grid: Vec<u64>,
    chunk_bytes: u64,
    whole_bytes: u64,
    ctx: FormatContext,
    btree: BTreeV1Config,
    file_size: u64,
}

impl ChunkIndexView {
    /// The index of a dataset with this layout and extent, or `None` when the
    /// layout is not chunked. Rejects a chunk shape the extent has no grid
    /// for (rank mismatch, zero chunk dimension).
    pub(crate) fn new(
        layout: &DataLayoutMessage,
        dims: &[u64],
        max_dims: Option<&[u64]>,
        element_size: u64,
        meta: &FileMeta,
        file_size: u64,
    ) -> IoResult<Option<Self>> {
        // Both chunked layouts carry the element size as a trailing chunk
        // dimension; the grid is over the rest.
        let (kind, addr, layout_chunk) = match layout {
            DataLayoutMessage::ChunkedV3 {
                chunk_dims,
                b_tree_address,
            } => (IndexKind::BTreeV1, *b_tree_address, chunk_dims),
            DataLayoutMessage::ChunkedV4 {
                chunk_dims,
                index_type,
                earray_params,
                single_chunk_filter,
                index_address,
                ..
            } => {
                let kind = match index_type {
                    ChunkIndexType::SingleChunk => IndexKind::SingleChunk(*single_chunk_filter),
                    ChunkIndexType::Implicit => IndexKind::Implicit,
                    ChunkIndexType::FixedArray => IndexKind::FixedArray,
                    ChunkIndexType::ExtensibleArray => IndexKind::ExtensibleArray(
                        earray_params
                            .clone()
                            .ok_or_else(|| IoError::InvalidState("missing earray params".into()))?,
                    ),
                    ChunkIndexType::BTreeV2 => IndexKind::BTreeV2,
                };
                (kind, *index_address, chunk_dims)
            }
            _ => return Ok(None),
        };
        let chunk_dims = layout_chunk[..layout_chunk.len().saturating_sub(1)].to_vec();
        let grid = chunk_grid::index_grid(dims, max_dims, &chunk_dims)?;
        Ok(Some(Self {
            kind,
            addr,
            dims: dims.to_vec(),
            max_dims: max_dims.map(<[u64]>::to_vec),
            chunk_bytes: byte_len(&chunk_dims, element_size),
            whole_bytes: byte_len(dims, element_size),
            chunk_dims,
            grid,
            ctx: meta.ctx,
            btree: meta.btree,
            file_size,
        }))
    }

    pub(crate) fn kind(&self) -> &IndexKind {
        &self.kind
    }

    /// The index structure's address, which is what a cache of its decoded
    /// blocks is keyed by. `UNDEF_ADDR` when no chunk was ever allocated.
    pub(crate) fn index_address(&self) -> u64 {
        self.addr
    }

    /// The chunk shape, without the trailing element-size dimension.
    pub(crate) fn chunk_dims(&self) -> &[u64] {
        &self.chunk_dims
    }

    /// Bytes of one whole chunk, unfiltered.
    pub(crate) fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    /// Bytes of the whole dataset at its current extent, unfiltered.
    pub(crate) fn whole_bytes(&self) -> u64 {
        self.whole_bytes
    }

    pub(crate) fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Whether the chunk at `coords` starts inside the current extent. A
    /// chunk past it is not part of the dataset this view describes, however
    /// the index came to hold it (written before a shrink, or by a SWMR
    /// writer ahead of the extent this reader last refreshed).
    fn within_extent(&self, coords: &[u64]) -> bool {
        coords
            .iter()
            .zip(&self.chunk_dims)
            .zip(&self.dims)
            .all(|((&c, &cd), &d)| c.checked_mul(cd).is_some_and(|start| start < d))
    }

    fn linear(&self, coords: &[u64]) -> IoResult<u64> {
        chunk_grid::grid_linear_index(&self.grid, self.max_dims.as_deref(), coords)
    }
}

/// `count(dims) * element_size`, saturating.
fn byte_len(dims: &[u64], element_size: u64) -> u64 {
    dims.iter()
        .fold(element_size, |acc, &d| acc.saturating_mul(d))
}

/// The chunk at grid `coords`, or `None` when the index holds none there
/// that the current extent can see.
///
/// `Err` for coordinates of the wrong rank or outside the bounded part of the
/// chunk grid, and for an index that does not decode.
pub(crate) fn find(
    handle: &FileHandle,
    view: &ChunkIndexView,
    blocks: &mut BlockCache,
    coords: &[u64],
) -> IoResult<Option<ChunkRecord>> {
    if let IndexKind::SingleChunk(_) = view.kind {
        chunk_grid::check_rank(view.grid.len(), coords)?;
        if coords.iter().any(|&c| c != 0) {
            return Err(IoError::InvalidState(format!(
                "chunk coordinates {coords:?} are outside the chunk grid (0..1): this \
                 dataset has a single-chunk index"
            )));
        }
    } else {
        chunk_grid::check_coords(&view.grid, view.max_dims.as_deref(), coords)?;
    }
    if view.addr == UNDEF_ADDR || !view.within_extent(coords) {
        return Ok(None);
    }
    let record = handle.load_metadata(|| {
        let found = match &view.kind {
            IndexKind::SingleChunk(filter) => Ok(Some(ChunkRecord {
                addr: view.addr,
                nbytes: filter.map_or(view.whole_bytes, |f| f.nbytes),
                mask: filter.map_or(0, |f| f.filter_mask),
            })),
            IndexKind::Implicit => {
                // `H5D__none_idx_create` allocates every slot of the maximum
                // grid in one block, so the address is arithmetic.
                let linear = view.linear(coords)?;
                Ok(Some(ChunkRecord {
                    addr: view
                        .addr
                        .saturating_add(linear.saturating_mul(view.chunk_bytes)),
                    nbytes: view.chunk_bytes,
                    mask: 0,
                }))
            }
            IndexKind::FixedArray => find_fixed_array(handle, view, blocks, view.linear(coords)?),
            IndexKind::ExtensibleArray(params) => {
                find_extensible_array(handle, view, params, blocks, view.linear(coords)?)
            }
            IndexKind::BTreeV2 => find_btree_v2(handle, view, blocks, coords),
            IndexKind::BTreeV1 => find_btree_v1(handle, view, blocks, coords),
        };
        // A failed walk leaves nothing memoized: a block's checksum is placed
        // by its parent's image (a B-tree v2 child's record count), so a
        // retry that kept the parent would check the same wrong range again.
        if found.is_err() {
            *blocks = BlockCache::default();
        }
        found
    })?;
    Ok(record.filter(|r| r.addr != UNDEF_ADDR))
}

/// Index blocks one dataset's lookups have decoded, kept for the next lookup
/// through the same block.
///
/// MUST NOT outlive the catalog entry whose index it was read from: it hangs
/// under that entry beside the index address it was decoded from, and goes
/// when the entry changes or a SWMR refresh rebuilds the catalog. Within that
/// lifetime a block is decoded once; a SWMR writer only ever adds chunks past
/// the extent the reader last refreshed to, so a block decoded after the
/// refresh answers for every chunk that extent covers.
#[derive(Default)]
pub(crate) struct BlockCache {
    fa_header: Option<Arc<FaInfo>>,
    fa_prefix: Option<Arc<FixedArrayPagedPrefix>>,
    ea_header: Option<Arc<EaInfo>>,
    ea_index: Option<Arc<EaIndexBlock>>,
    ea_super: HashMap<u64, Arc<ExtensibleArraySuperBlock>>,
    /// Fixed-array data blocks and pages, extensible-array data blocks and
    /// pages: by address and element count.
    records: HashMap<(u64, usize), Arc<Vec<ChunkRecord>>>,
    bt2_header: Option<Arc<Bt2Info>>,
    /// By address, record count and depth: a node's own bytes do not say how
    /// many records it holds or how deep it sits, its parent does.
    bt2_nodes: HashMap<(u64, u16, u16), Arc<Bt2Node>>,
    v1_nodes: HashMap<u64, Arc<V1Node>>,
}

impl BlockCache {
    /// How many index blocks are held.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        usize::from(self.fa_header.is_some())
            + usize::from(self.fa_prefix.is_some())
            + usize::from(self.ea_header.is_some())
            + usize::from(self.ea_index.is_some())
            + self.ea_super.len()
            + self.records.len()
            + usize::from(self.bt2_header.is_some())
            + self.bt2_nodes.len()
            + self.v1_nodes.len()
    }
}

/// The value cached under `key`, decoding it with `load` the first time.
fn memo<K: Eq + Hash, V>(
    map: &mut HashMap<K, Arc<V>>,
    key: K,
    load: impl FnOnce() -> IoResult<V>,
) -> IoResult<Arc<V>> {
    if let Some(hit) = map.get(&key) {
        return Ok(Arc::clone(hit));
    }
    let value = Arc::new(load()?);
    map.insert(key, Arc::clone(&value));
    Ok(value)
}

/// [`memo`] for a block an index has exactly one of.
fn memo_slot<V>(slot: &mut Option<Arc<V>>, load: impl FnOnce() -> IoResult<V>) -> IoResult<Arc<V>> {
    if let Some(hit) = slot {
        return Ok(Arc::clone(hit));
    }
    let value = Arc::new(load()?);
    *slot = Some(Arc::clone(&value));
    Ok(value)
}

/// Read up to `len` bytes at `addr` and decode them as one index block.
fn read_block<T>(
    handle: &FileHandle,
    addr: u64,
    len: usize,
    decode: impl Fn(&[u8]) -> FormatResult<T>,
) -> IoResult<T> {
    let buf = handle.read_at_most(addr, len)?;
    Ok(decode(&buf)?)
}

/// Entry `i` of a decoded block, which must hold it.
fn entry<T: Copy>(entries: &[T], i: u64, what: &str) -> IoResult<T> {
    usize::try_from(i)
        .ok()
        .and_then(|i| entries.get(i))
        .copied()
        .ok_or_else(|| {
            IoError::InvalidState(format!(
                "{what} has {} entries; entry {i} is past its end",
                entries.len()
            ))
        })
}

/// Width of the stored-size field of a filtered array element, from the
/// element's raw size: what is left after the address and the 4-byte mask.
fn filtered_size_len(raw_elmt_size: u8, sizeof_addr: u8, what: &str) -> IoResult<u8> {
    let len = raw_elmt_size
        .checked_sub(sizeof_addr)
        .and_then(|n| n.checked_sub(4))
        .ok_or_else(|| IoError::InvalidState(format!("{what} filtered element_size too small")))?;
    // The stored size is read into a u64.
    if len > 8 {
        return Err(IoError::InvalidState(format!(
            "{what} filtered chunk-size width {len} exceeds 8 bytes"
        )));
    }
    Ok(len)
}

/// A fixed-array header, with what its elements' encoding follows from it.
struct FaInfo {
    hdr: FixedArrayHeader,
    filtered: bool,
    size_len: usize,
    /// Bytes one element takes in the data block or a page of it.
    elem_size: usize,
}

fn find_fixed_array(
    handle: &FileHandle,
    view: &ChunkIndexView,
    blocks: &mut BlockCache,
    linear: u64,
) -> IoResult<Option<ChunkRecord>> {
    let ctx = view.ctx;
    let info = memo_slot(&mut blocks.fa_header, || {
        let hdr = read_block(handle, view.addr, 256, |b| {
            FixedArrayHeader::decode(b, &ctx)
        })?;
        let filtered = hdr.client_id == fa::FA_CLIENT_FILT_CHUNK;
        let size_len = if filtered {
            filtered_size_len(hdr.element_size, ctx.sizeof_addr, "fixed array")? as usize
        } else {
            0
        };
        let sa = ctx.sizeof_addr as usize;
        let elem_size = if filtered { sa + size_len + 4 } else { sa };
        // Everything below is sized from the element count: refuse one whose
        // elements could not all be in the file before it sizes a read.
        if hdr.data_blk_addr != UNDEF_ADDR
            && (hdr.num_elmts)
                .checked_mul(elem_size as u64)
                .is_none_or(|bytes| bytes > view.file_size)
        {
            return Err(IoError::InvalidState(format!(
                "fixed array declares {} elements of {elem_size} bytes, more than the \
                 {}-byte file holds",
                hdr.num_elmts, view.file_size
            )));
        }
        Ok(FaInfo {
            hdr,
            filtered,
            size_len,
            elem_size,
        })
    })?;
    let hdr = &info.hdr;
    if hdr.data_blk_addr == UNDEF_ADDR || linear >= hdr.num_elmts {
        return Ok(None);
    }
    let decode_page = |addr: u64, nelmts: usize, chunk_bytes: u64| {
        read_block(
            handle,
            addr,
            nelmts * info.elem_size + 4,
            |b| -> FormatResult<Vec<ChunkRecord>> {
                Ok(if info.filtered {
                    fa::decode_filtered_page(b, &ctx, nelmts, info.size_len)?
                        .into_iter()
                        .map(|e| ChunkRecord {
                            addr: e.address,
                            nbytes: e.chunk_size,
                            mask: e.filter_mask,
                        })
                        .collect()
                } else {
                    unfiltered_records(fa::decode_unfiltered_page(b, &ctx, nelmts)?, chunk_bytes)
                })
            },
        )
    };
    if !hdr.is_paged() {
        let n = hdr.num_elmts as usize;
        let records = memo(&mut blocks.records, (hdr.data_blk_addr, n), || {
            let sa = ctx.sizeof_addr as usize;
            read_block(
                handle,
                hdr.data_blk_addr,
                4 + 1 + 1 + sa + n * info.elem_size + 4,
                |b| -> FormatResult<Vec<ChunkRecord>> {
                    Ok(if info.filtered {
                        FixedArrayDataBlock::decode_filtered(b, &ctx, n, info.size_len)?
                            .filtered_elements
                            .into_iter()
                            .map(|e| ChunkRecord {
                                addr: e.address,
                                nbytes: e.chunk_size,
                                mask: e.filter_mask,
                            })
                            .collect()
                    } else {
                        unfiltered_records(
                            FixedArrayDataBlock::decode_unfiltered(b, &ctx, n)?.elements,
                            view.chunk_bytes,
                        )
                    })
                },
            )
        })?;
        return entry(&records, linear, "fixed array data block").map(Some);
    }

    // Paged: a prefix carrying the page-init bitmap, then equal-stride pages,
    // only the last of which may hold fewer elements.
    let npages = hdr.npages();
    let page_nelmts = hdr.dblk_page_nelmts();
    let prefix = memo_slot(&mut blocks.fa_prefix, || {
        let sa = ctx.sizeof_addr as usize;
        let len = 4 + 1 + 1 + sa + (npages as usize).div_ceil(8) + 4;
        read_block(handle, hdr.data_blk_addr, len, |b| {
            FixedArrayPagedPrefix::decode(b, &ctx, npages)
        })
    })?;
    let page = linear / page_nelmts;
    if !prefix.page_initialized(page as usize) {
        return Ok(None);
    }
    let nelmts = if page + 1 == npages && hdr.num_elmts % page_nelmts != 0 {
        hdr.num_elmts % page_nelmts
    } else {
        page_nelmts
    } as usize;
    let stride = page_nelmts * info.elem_size as u64 + 4;
    let page_addr = hdr
        .data_blk_addr
        .saturating_add(prefix.prefix_size as u64)
        .saturating_add(page * stride);
    let records = memo(&mut blocks.records, (page_addr, nelmts), || {
        decode_page(page_addr, nelmts, view.chunk_bytes)
    })?;
    entry(&records, linear % page_nelmts, "fixed array page").map(Some)
}

/// Unfiltered index elements are bare addresses of whole chunks.
fn unfiltered_records(addrs: Vec<u64>, chunk_bytes: u64) -> Vec<ChunkRecord> {
    addrs
        .into_iter()
        .map(|addr| ChunkRecord {
            addr,
            nbytes: chunk_bytes,
            mask: 0,
        })
        .collect()
}

/// An extensible-array header, with the geometry its creation parameters give
/// and what its elements' encoding follows from it.
struct EaInfo {
    geo: EaGeometry,
    idx_blk_addr: u64,
    filtered: bool,
    size_len: u8,
    raw_elmt_size: usize,
}

/// An extensible-array index block: the elements it holds directly, and the
/// data-block and super-block addresses that reach the rest.
struct EaIndexBlock {
    elements: Vec<ChunkRecord>,
    dblk_addrs: Vec<u64>,
    sblk_addrs: Vec<u64>,
}

fn find_extensible_array(
    handle: &FileHandle,
    view: &ChunkIndexView,
    params: &EarrayParams,
    blocks: &mut BlockCache,
    linear: u64,
) -> IoResult<Option<ChunkRecord>> {
    let ctx = view.ctx;
    let sa = ctx.sizeof_addr as usize;
    let max_nelmts_bits = params.max_nelmts_bits;
    let info = memo_slot(&mut blocks.ea_header, || {
        let hdr = read_block(handle, view.addr, 256, |b| {
            ExtensibleArrayHeader::decode(b, &ctx)
        })?;
        let geo = EaGeometry::new(
            params.idx_blk_elmts,
            params.data_blk_min_elmts,
            params.sup_blk_min_data_ptrs,
            params.max_nelmts_bits,
            params.max_dblk_page_nelmts_bits,
        )?;
        let filtered = hdr.class_id == ea::EA_CLS_FILT_CHUNK;
        let size_len = if filtered {
            filtered_size_len(hdr.raw_elmt_size, ctx.sizeof_addr, "extensible array")?
        } else {
            0
        };
        let raw_elmt_size = if filtered {
            ea::FilteredChunkEntry::raw_size(ctx.sizeof_addr, size_len) as usize
        } else {
            sa
        };
        Ok(EaInfo {
            geo,
            idx_blk_addr: hdr.idx_blk_addr,
            filtered,
            size_len,
            raw_elmt_size,
        })
    })?;
    if info.idx_blk_addr == UNDEF_ADDR {
        return Ok(None);
    }
    let geo = &info.geo;
    let filtered_records = |entries: Vec<ea::FilteredChunkEntry>| -> Vec<ChunkRecord> {
        entries
            .into_iter()
            .map(|e| ChunkRecord {
                addr: e.addr,
                nbytes: e.nbytes,
                mask: e.filter_mask,
            })
            .collect()
    };
    let index = memo_slot(&mut blocks.ea_index, || {
        let n = geo.idx_blk_elmts as usize;
        let len = 4
            + 1
            + 1
            + sa
            + n * info.raw_elmt_size
            + geo.ndblk_addrs * sa
            + geo.nsblk_addrs * sa
            + 4;
        let buf = handle.read_at_most(info.idx_blk_addr, len)?;
        Ok(if info.filtered {
            let b = ea::FilteredIndexBlock::decode(
                &buf,
                &ctx,
                n,
                geo.ndblk_addrs,
                geo.nsblk_addrs,
                info.size_len,
            )?;
            EaIndexBlock {
                elements: filtered_records(b.elements),
                dblk_addrs: b.dblk_addrs,
                sblk_addrs: b.sblk_addrs,
            }
        } else {
            let b =
                ExtensibleArrayIndexBlock::decode(&buf, &ctx, n, geo.ndblk_addrs, geo.nsblk_addrs)?;
            EaIndexBlock {
                elements: unfiltered_records(b.elements, view.chunk_bytes),
                dblk_addrs: b.dblk_addrs,
                sblk_addrs: b.sblk_addrs,
            }
        })
    })?;

    let loc = match geo.locate(linear)? {
        EaLoc::Index { elem } => {
            return entry(&index.elements, elem as u64, "extensible array index block").map(Some)
        }
        EaLoc::Dblk(loc) => loc,
    };
    // The data block's address, and for a paged one the super block whose
    // page-init bitmap says which of its pages exist.
    let (dblk_addr, sblk) = match loc.path {
        EaDblkPath::Direct { idx } => {
            if loc.paged {
                return Err(IoError::Unsupported(
                    "extensible array with paged data blocks addressed from the index \
                     block"
                        .into(),
                ));
            }
            (
                entry(
                    &index.dblk_addrs,
                    idx as u64,
                    "extensible array index block",
                )?,
                None,
            )
        }
        EaDblkPath::ViaSblk {
            sblk_off,
            local_dblk,
            ndblks_in_sblk,
            ..
        } => {
            let sblk_addr = entry(
                &index.sblk_addrs,
                sblk_off as u64,
                "extensible array index block",
            )?;
            if sblk_addr == UNDEF_ADDR {
                return Ok(None);
            }
            let page_init_total = if loc.paged {
                ndblks_in_sblk * geo.dblk_page_init_size(loc.sblk_idx)
            } else {
                0
            };
            let sb = memo(&mut blocks.ea_super, sblk_addr, || {
                let len = 4
                    + 1
                    + 1
                    + sa
                    + ExtensibleArrayDataBlock::block_offset_size(max_nelmts_bits)
                    + page_init_total
                    + ndblks_in_sblk * sa
                    + 4;
                read_block(handle, sblk_addr, len, |b| {
                    ExtensibleArraySuperBlock::decode(
                        b,
                        &ctx,
                        max_nelmts_bits,
                        ndblks_in_sblk,
                        page_init_total,
                    )
                })
            })?;
            let addr = entry(
                &sb.dblk_addrs,
                local_dblk as u64,
                "extensible array super block",
            )?;
            (addr, Some((sb, local_dblk)))
        }
    };
    if dblk_addr == UNDEF_ADDR {
        return Ok(None);
    }
    let prefix = geo.dblk_prefix_size(ctx.sizeof_addr, max_nelmts_bits);

    if !loc.paged {
        let n = loc.dblk_nelmts as usize;
        let records = memo(&mut blocks.records, (dblk_addr, n), || {
            read_block(
                handle,
                dblk_addr,
                prefix + n * info.raw_elmt_size,
                |b| -> FormatResult<Vec<ChunkRecord>> {
                    Ok(if info.filtered {
                        filtered_records(
                            ea::FilteredDataBlock::decode(
                                b,
                                &ctx,
                                max_nelmts_bits,
                                n,
                                info.size_len,
                            )?
                            .elements,
                        )
                    } else {
                        unfiltered_records(
                            ExtensibleArrayDataBlock::decode(b, &ctx, max_nelmts_bits, n)?.elements,
                            view.chunk_bytes,
                        )
                    })
                },
            )
        })?;
        return entry(&records, loc.offset_in_dblk, "extensible array data block").map(Some);
    }

    // Paged: only a prefix lives at the data block's address, its pages follow
    // it. The super block's page-init bitmap is one flat MSB-first bitmap
    // (H5VM bit ops) indexed by `local_dblk * npages + page` (H5EA.c). A page
    // is its elements and a checksum, the same encoding a fixed-array page has.
    let Some((sb, local_dblk)) = sblk else {
        unreachable!("a paged data block addressed from the index block returned above");
    };
    let npages = geo.npages(loc.sblk_idx);
    let page_nelmts = geo.dblk_page_nelmts;
    let page = loc.offset_in_dblk / page_nelmts;
    let bit = local_dblk as u64 * npages + page;
    let byte = entry(&sb.page_init, bit / 8, "extensible array page-init bitmap")?;
    if byte & (0x80u8 >> (bit % 8)) == 0 {
        return Ok(None);
    }
    let page_size = geo.dblk_page_size(info.raw_elmt_size);
    let page_addr = dblk_addr
        .saturating_add(prefix as u64)
        .saturating_add(page.saturating_mul(page_size as u64));
    let n = page_nelmts as usize;
    let records = memo(&mut blocks.records, (page_addr, n), || {
        read_block(
            handle,
            page_addr,
            page_size,
            |b| -> FormatResult<Vec<ChunkRecord>> {
                Ok(if info.filtered {
                    fa::decode_filtered_page(b, &ctx, n, info.size_len as usize)?
                        .into_iter()
                        .map(|e| ChunkRecord {
                            addr: e.address,
                            nbytes: e.chunk_size,
                            mask: e.filter_mask,
                        })
                        .collect()
                } else {
                    unfiltered_records(fa::decode_unfiltered_page(b, &ctx, n)?, view.chunk_bytes)
                })
            },
        )
    })?;
    entry(
        &records,
        loc.offset_in_dblk % page_nelmts,
        "extensible array page",
    )
    .map(Some)
}

/// A v2-B-tree header, with the node geometry it gives.
struct Bt2Info {
    hdr: Bt2Header,
    geo: Bt2Geometry,
    filtered: bool,
}

/// One v2-B-tree node's records, in key order: their scaled chunk offsets
/// (`rank` per record, flattened), where each chunk is, and for an internal
/// node each child's address and record count.
struct Bt2Node {
    keys: Vec<u64>,
    records: Vec<ChunkRecord>,
    children: Vec<(u64, u16)>,
}

fn find_btree_v2(
    handle: &FileHandle,
    view: &ChunkIndexView,
    blocks: &mut BlockCache,
    coords: &[u64],
) -> IoResult<Option<ChunkRecord>> {
    let ctx = view.ctx;
    let rank = coords.len();
    let info = memo_slot(&mut blocks.bt2_header, || {
        let hdr = read_block(handle, view.addr, 256, |b| Bt2Header::decode(b, &ctx))?;
        let filtered = hdr.record_type != BT2_TYPE_CHUNK_UNFILT;
        // An unfiltered record is the scaled offsets and the address, nothing
        // else; any other width would misalign every record after the first.
        let unfiltered_size = rank * 8 + ctx.sizeof_addr as usize;
        if !filtered && hdr.record_size as usize != unfiltered_size {
            return Err(IoError::InvalidState(format!(
                "v2 B-tree chunk records are {} bytes, but an unfiltered record of a \
                 {rank}-dimensional dataset is {unfiltered_size}",
                hdr.record_size
            )));
        }
        let geo = Bt2Geometry::new(hdr.node_size, hdr.record_size, hdr.depth, ctx.sizeof_addr);
        Ok(Bt2Info { hdr, geo, filtered })
    })?;
    let hdr = &info.hdr;
    if hdr.root_node_addr == UNDEF_ADDR || hdr.total_num_records == 0 {
        return Ok(None);
    }
    let (mut addr, mut nrec, mut depth) = (hdr.root_node_addr, hdr.num_records_in_root, hdr.depth);
    loop {
        let node = memo(&mut blocks.bt2_nodes, (addr, nrec, depth), || {
            read_block(handle, addr, hdr.node_size as usize, |b| {
                let (record_data, children) = if depth == 0 {
                    (
                        Bt2LeafNode::decode(b, nrec, hdr.record_size)?.record_data,
                        Vec::new(),
                    )
                } else {
                    let node = Bt2InternalNode::decode(
                        b,
                        &ctx,
                        depth,
                        nrec,
                        hdr.record_size,
                        info.geo.max_nrec_size,
                        info.geo.child_total_size(depth),
                    )?;
                    let children = node
                        .child_addrs
                        .iter()
                        .copied()
                        .zip(node.child_nrecords.iter().copied())
                        .collect();
                    (node.record_data, children)
                };
                let n = nrec as usize;
                let mut keys = Vec::with_capacity(n * rank);
                let mut records = Vec::with_capacity(n);
                if info.filtered {
                    let decoded = Bt2ChunkIndex::decode_filtered_records(
                        &record_data,
                        n,
                        rank,
                        hdr.record_size,
                        &ctx,
                    )?;
                    for r in decoded {
                        keys.extend_from_slice(&r.scaled_offsets);
                        records.push(ChunkRecord {
                            addr: r.chunk_address,
                            nbytes: r.chunk_size,
                            mask: r.filter_mask,
                        });
                    }
                } else {
                    let decoded =
                        Bt2ChunkIndex::decode_unfiltered_records(&record_data, n, rank, &ctx)?;
                    for r in decoded {
                        keys.extend_from_slice(&r.scaled_offsets);
                        records.push(ChunkRecord {
                            addr: r.chunk_address,
                            nbytes: view.chunk_bytes,
                            mask: 0,
                        });
                    }
                }
                Ok(Bt2Node {
                    keys,
                    records,
                    children,
                })
            })
        })?;
        // Records are sorted by scaled offsets; in an internal node record i
        // separates child i from child i + 1.
        match search(
            node.records.len(),
            |i| &node.keys[i * rank..(i + 1) * rank],
            coords,
        ) {
            Ok(i) => return Ok(Some(node.records[i])),
            Err(_) if depth == 0 => return Ok(None),
            Err(i) => {
                (addr, nrec) = entry(&node.children, i as u64, "v2 B-tree internal node")?;
                depth -= 1;
            }
        }
    }
}

/// Binary search of `n` keys sorted ascending: `Ok(i)` where `key(i)` equals
/// `target`, else `Err` with the count of keys below it.
fn search<'a>(n: usize, key: impl Fn(usize) -> &'a [u64], target: &[u64]) -> Result<usize, usize> {
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match key(mid).cmp(target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Ok(mid),
        }
    }
    Err(lo)
}

/// One v1-B-tree chunk node: its level, each child's left key as scaled chunk
/// offsets (`rank` per child, flattened), the children, and at a leaf what
/// each child chunk's key records about it.
struct V1Node {
    level: u8,
    keys: Vec<u64>,
    children: Vec<u64>,
    records: Vec<ChunkRecord>,
}

/// A v1 node level fits in one byte, so a well-formed tree is at most 256
/// levels deep; this also stops a cyclic index.
const V1_MAX_DEPTH: usize = 256;

fn find_btree_v1(
    handle: &FileHandle,
    view: &ChunkIndexView,
    blocks: &mut BlockCache,
    coords: &[u64],
) -> IoResult<Option<ChunkRecord>> {
    let ctx = view.ctx;
    let rank = coords.len();
    let sa = ctx.sizeof_addr as usize;
    let mut addr = view.addr;
    for _ in 0..=V1_MAX_DEPTH {
        if addr == UNDEF_ADDR || addr >= view.file_size {
            return Ok(None);
        }
        let node = memo(&mut blocks.v1_nodes, addr, || {
            let len = view.btree.chunk_btree_node_size(sa, rank);
            read_block(handle, addr, len, |b| {
                let node = ChunkBTreeV1Node::decode(b, sa, rank, view.btree.chunk_max_entries())?;
                let n = node.children.len();
                let mut keys = Vec::with_capacity(n * rank);
                let mut records = Vec::with_capacity(n);
                for (key, &child) in node.keys.iter().zip(&node.children) {
                    // Keys hold element offsets; the grid is in chunks.
                    keys.extend(
                        key.offsets
                            .iter()
                            .zip(&view.chunk_dims)
                            .map(|(&o, &cd)| o.checked_div(cd).unwrap_or(0)),
                    );
                    records.push(ChunkRecord {
                        addr: child,
                        nbytes: key.chunk_size as u64,
                        mask: key.filter_mask,
                    });
                }
                Ok(V1Node {
                    level: node.level,
                    keys,
                    children: node.children,
                    records,
                })
            })
        })?;
        // The child to follow is the last whose left key is at or before the
        // target (`H5B_find`); at a leaf that key must be the target itself.
        let key = |i: usize| &node.keys[i * rank..(i + 1) * rank];
        let below = match search(node.children.len(), key, coords) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        let Some(i) = below.checked_sub(1) else {
            return Ok(None);
        };
        if node.level == 0 {
            return Ok((key(i) == coords).then_some(node.records[i]));
        }
        addr = node.children[i];
    }
    Err(IoError::InvalidState(
        "chunk B-tree v1 exceeds maximum depth".into(),
    ))
}
