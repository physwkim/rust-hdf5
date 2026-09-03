//! Serialized H5S dataspace-selection decoder.
//!
//! `H5S_select_deserialize` (H5Sselect.c) and its per-type callbacks
//! (H5Sall.c, H5Snone.c, H5Shyper.c) define a small self-describing wire
//! format for "which elements of a dataspace are selected". It shows up
//! embedded inside other structures — a Virtual Dataset mapping entry
//! (H5Dvirtual.c `H5D__virtual_load_layout`) carries two of them per
//! mapping, and a region reference embeds one too — so this module decodes
//! only the selection bytes themselves and knows nothing about either
//! caller.
//!
//! Binary layout, common header:
//! ```text
//! sel_type: u32 LE (0 = none, 1 = points, 2 = hyperslabs, 3 = all)
//! ```
//! followed by a type-specific body.
//!
//! All / None body (version is always 1):
//! ```text
//! version:  u32 LE (= 1)
//! reserved: 8 bytes
//! ```
//!
//! Point body (`H5S__point_deserialize`, H5Spoint.c):
//! ```text
//! version: u32 LE (1 or 2)
//! if version >= 2: enc_size: 1 byte (2, 4, or 8 bytes)
//! else (version == 1): padding: 4 bytes, length: 4 bytes, enc_size = 4
//! rank: u32 LE
//! num_points: enc_size bytes LE
//! num_points * rank * { coordinate }, each enc_size bytes LE (point-major,
//! coordinate-minor — point 0's rank coordinates, then point 1's, ...)
//! ```
//! The version-1 `length` field records the byte count from `rank` to the
//! end of the point list (`8 + num_points * rank * 4`); this module does
//! not validate it on decode (H5S__point_deserialize doesn't either), but
//! [`Selection::encode`] reproduces the exact value libhdf5 writes there,
//! since a byte-for-byte comparison against a captured image needs it.
//!
//! Hyperslab body:
//! ```text
//! version: u32 LE (1, 2, or 3)
//! if version >= 2: flags: 1 byte (bit 0 = REGULAR)
//! if version >= 3: enc_size: 1 byte (tag 0x02/0x04/0x08 => 2/4/8 bytes)
//! else if version == 2: reserved: 4 bytes, enc_size = 8
//! else (version == 1): reserved: 8 bytes, enc_size = 4
//! rank: u32 LE
//! if the REGULAR flag is set (only possible for version >= 2):
//!   rank * { start, stride, count, block }, each enc_size bytes LE
//!   (an all-ones count or block of that width means H5S_UNLIMITED)
//! else (block list, any version):
//!   num_blocks: enc_size bytes LE
//!   num_blocks * { rank * start_coord, rank * end_coord }, each
//!   enc_size bytes LE (absolute element coordinates, end inclusive,
//!   blocks combined by union)
//! ```
//!
//! Empirically confirmed against libhdf5 1.14.6 (h5py's `VirtualLayout`
//! writes hyperslab selections in the version-1 block-list form — with
//! exactly one block — even for a selection that is mathematically a
//! single regular block; see [`h5py_single_block_selection_is_version_one`]).

use crate::format::{FormatError, FormatResult};

/// Sentinel marking a hyperslab `count` or `block` value as
/// unlimited/growable (`H5S_UNLIMITED`, numerically `HSIZE_UNDEF` —
/// H5Spublic.h / H5public.h).
pub const UNLIMITED: u64 = u64::MAX;

const SEL_NONE: u32 = 0;
const SEL_POINTS: u32 = 1;
const SEL_HYPERSLABS: u32 = 2;
const SEL_ALL: u32 = 3;

const ALL_NONE_VERSION: u32 = 1;

const HYPER_VERSION_1: u32 = 1;
const HYPER_VERSION_2: u32 = 2;
const HYPER_VERSION_3: u32 = 3;

const HYPER_REGULAR_FLAG: u8 = 0x01;

const POINT_VERSION_1: u32 = 1;
const POINT_VERSION_2: u32 = 2;

/// The dataspace rank ceiling libhdf5 enforces (`H5S_MAX_RANK`,
/// H5Spublic.h). Bounds `rank`-sized allocations before any element count
/// derived from the file is trusted.
const MAX_RANK: usize = 32;

/// One block of a hyperslab block-list selection: inclusive
/// `start..=end` element coordinates, one pair per dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperslabBlock {
    pub start: Vec<u64>,
    pub end: Vec<u64>,
}

/// A regular (start, stride, count, block) hyperslab: one tuple per
/// dimension. `count`/`block` may hold [`UNLIMITED`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegularHyperslab {
    pub start: Vec<u64>,
    pub stride: Vec<u64>,
    pub count: Vec<u64>,
    pub block: Vec<u64>,
}

impl RegularHyperslab {
    /// The one dimension whose `count` or `block` is [`UNLIMITED`], or `None`
    /// when the selection is bounded — `H5S_get_select_unlim_dim`, which
    /// returns a single dimension because `H5Sselect_hyperslab` refuses a
    /// second unlimited one.
    pub fn unlim_dim(&self) -> Option<usize> {
        (0..self.count.len())
            .find(|&d| self.count[d] == UNLIMITED || self.block.get(d) == Some(&UNLIMITED))
    }

    /// The `(count, block)` this dimension takes once its extent is known —
    /// `H5S__hyper_get_clip_diminfo`. A zero in either field means the
    /// selection ends up empty.
    fn clip_diminfo(start: u64, stride: u64, count: u64, block: u64, clip_size: u64) -> (u64, u64) {
        if start >= clip_size {
            if block == UNLIMITED {
                (count, 0)
            } else {
                (0, block)
            }
        } else if block == UNLIMITED || block == stride {
            (1, clip_size - start)
        } else {
            // The other unlimited form: an unbounded count of fixed blocks.
            let stride = stride.max(1);
            ((clip_size - start).div_ceil(stride), block)
        }
    }

    /// How many slices of the unlimited dimension this selection covers when
    /// its dataspace extent is `clip_size` — the slice count
    /// `H5S_hyper_get_clip_extent_match` computes from the *match* space
    /// before handing it to [`Self::clip_extent`].
    pub fn num_slices(&self, clip_size: u64) -> u64 {
        let Some(d) = self.unlim_dim() else {
            return 0;
        };
        let (start, stride) = (self.start[d], self.stride[d]);
        let (count, block) =
            Self::clip_diminfo(start, stride, self.count[d], self.block[d], clip_size);
        if block == 0 || count == 0 {
            return 0;
        }
        if count == 1 {
            return block;
        }
        let span = stride * (count - 1) + block;
        let avail = clip_size - start;
        if span > avail {
            block * count - (span - avail)
        } else {
            block * count
        }
    }

    /// The virtual extent that makes this selection cover exactly `num_slices`
    /// slices of its unlimited dimension — `H5S__hyper_get_clip_extent_real`.
    /// `incl_trail` is the `H5D_VDS_FIRST_MISSING` view; the default
    /// `H5D_VDS_LAST_AVAILABLE` passes `false`.
    pub fn clip_extent(&self, num_slices: u64, incl_trail: bool) -> u64 {
        let Some(d) = self.unlim_dim() else {
            return 0;
        };
        let (start, stride, block) = (self.start[d], self.stride[d], self.block[d]);
        if num_slices == 0 {
            return if incl_trail { start } else { 0 };
        }
        if block == UNLIMITED || block == stride {
            return start + num_slices;
        }
        let block = block.max(1);
        let count = num_slices / block;
        let rem = num_slices - count * block;
        if rem > 0 {
            start + count * stride + rem
        } else if incl_trail {
            start + count * stride
        } else {
            start + (count - 1) * stride + block
        }
    }

    /// Elements in one slice through the dimensions that are *not* the
    /// unlimited one — `H5S_get_select_num_elem_non_unlim`, the quantity two
    /// unlimited selections in one mapping must agree on.
    pub fn num_elem_non_unlim(&self) -> Option<u64> {
        let d = self.unlim_dim()?;
        (0..self.count.len())
            .filter(|&i| i != d)
            .try_fold(1u64, |acc, i| {
                acc.checked_mul(self.count[i])?.checked_mul(self.block[i])
            })
    }

    /// Block `index` of the unlimited dimension on its own, every other
    /// dimension left as it is — `H5S_hyper_get_unlim_block`, the selection a
    /// printf mapping's `index`-th source dataset fills.
    pub fn unlim_block(&self, index: u64) -> Self {
        let mut out = self.clone();
        if let Some(d) = self.unlim_dim() {
            out.start[d] = self.start[d] + index * self.stride[d];
            out.count[d] = 1;
        }
        out
    }
}

/// The two wire forms a hyperslab selection can take
/// (`H5S__hyper_deserialize`, H5Shyper.c): a single compact
/// (start, stride, count, block) tuple per dimension, or an explicit list
/// of blocks combined by union. libhdf5 always writes the block-list form
/// for a selection made of exactly one block — including h5py's
/// `VirtualLayout` — so both forms are real on-disk data, not just
/// alternates on paper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hyperslab {
    Regular(RegularHyperslab),
    Blocks(Vec<HyperslabBlock>),
}

/// An explicit list of selected element coordinates (`H5S_SEL_POINTS`):
/// one `rank`-length coordinate vector per point, in selection order.
/// Selection order is significant for a pointwise iterator (it is the
/// linear order in which elements are visited), so callers must not
/// reorder `points`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointSelection {
    pub rank: usize,
    pub points: Vec<Vec<u64>>,
}

/// One maximal run of consecutive selected elements, as it sits both in the
/// box it came from and in the full extent that box was resolved against.
///
/// A run is contiguous in the fastest-varying dimension, so it is one
/// unbroken stretch of elements in a row-major buffer of either shape — the
/// unit a transfer can move with a single copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SelectionRun {
    /// Index into [`ResolvedSelection::boxes`] of the box this run lies in.
    pub box_index: usize,
    /// Element offset of the run's first element within a row-major buffer
    /// shaped like that box.
    pub offset_in_box: u64,
    /// Element offset of the run's first element within a row-major buffer
    /// shaped like the whole extent.
    pub offset_in_extent: u64,
    /// How many consecutive elements the run covers.
    pub len: u64,
}

/// A selection bound to a concrete extent — [`Selection::resolve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedSelection {
    /// The disjoint `(start, count)` boxes covering the selected elements,
    /// in [`Selection::to_boxes`] order.
    pub boxes: Vec<(Vec<u64>, Vec<u64>)>,
    /// Every selected element exactly once, grouped into runs and ordered
    /// the way H5S's own selection iterator visits them.
    pub runs: Vec<SelectionRun>,
}

impl ResolvedSelection {
    /// How many elements the selection holds — `H5S_GET_SELECT_NPOINTS`.
    pub(crate) fn n_elements(&self) -> u64 {
        self.runs.iter().map(|r| r.len).sum()
    }
}

/// Row-major element strides of an extent: `strides[d]` is how far one step
/// in dimension `d` moves within a densely-packed buffer of shape `dims`.
fn row_major_strides(dims: &[u64]) -> FormatResult<Vec<u64>> {
    let mut strides = vec![1u64; dims.len()];
    for d in (0..dims.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1].checked_mul(dims[d + 1]).ok_or_else(|| {
            FormatError::InvalidData(format!(
                "extent {dims:?} holds more elements than u64 counts"
            ))
        })?;
    }
    Ok(strides)
}

/// A hyperslab `[starts, counts)` that an extent does not admit.
///
/// One rule for every strideless selection, whoever supplies it — a slice
/// read or written through the public API, a point read, the boxes of a
/// region reference — owned by [`check_hyperslab`] alone. Each layer that
/// meets it converts it to its own class: a file's selection is invalid
/// data, a caller's slice an invalid request. The text is the same either
/// way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HyperslabError {
    /// `starts` or `counts` has a different length than the extent.
    Rank { got: usize, rank: usize },
    /// `start + count` wraps, or lands past the extent, in `dim`.
    OutOfBounds {
        dim: usize,
        start: u64,
        count: u64,
        extent: u64,
    },
}

impl std::fmt::Display for HyperslabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rank { got, rank } => {
                write!(f, "selection rank {got} does not match dataset rank {rank}")
            }
            Self::OutOfBounds {
                dim,
                start,
                count,
                extent,
            } => write!(
                f,
                "slice out of bounds in dimension {dim}: start {start} + count {count} exceeds extent {extent}"
            ),
        }
    }
}

impl From<HyperslabError> for FormatError {
    fn from(e: HyperslabError) -> Self {
        FormatError::InvalidData(e.to_string())
    }
}

/// Refuse a hyperslab `[starts, counts)` that `dims` does not admit: the
/// rank must match and every edge `starts[d] + counts[d]` must stay inside
/// the extent. The edge is computed checked, so a start near `u64::MAX`
/// (caller input, or a box a file supplied) is refused here rather than
/// wrapping into an offset that lands inside the extent and reads
/// unrelated bytes.
pub(crate) fn check_hyperslab(
    dims: &[u64],
    starts: &[u64],
    counts: &[u64],
) -> Result<(), HyperslabError> {
    let rank = dims.len();
    if starts.len() != rank {
        return Err(HyperslabError::Rank {
            got: starts.len(),
            rank,
        });
    }
    if counts.len() != rank {
        return Err(HyperslabError::Rank {
            got: counts.len(),
            rank,
        });
    }
    for (dim, &extent) in dims.iter().enumerate() {
        if starts[dim]
            .checked_add(counts[dim])
            .is_none_or(|end| end > extent)
        {
            return Err(HyperslabError::OutOfBounds {
                dim,
                start: starts[dim],
                count: counts[dim],
                extent,
            });
        }
    }
    Ok(())
}

/// Append the runs one box contributes, in the box's own row-major order.
///
/// Consecutive runs that are adjacent in *both* the box and the extent are
/// merged into one, which is the "trailing dimension fully selected on both
/// sides" coalescing a dual-array walk performs: a box covering a whole
/// trailing region comes out as a single run rather than one run per row.
fn push_box_runs(
    box_index: usize,
    start: &[u64],
    count: &[u64],
    dims: &[u64],
    strides: &[u64],
    runs: &mut Vec<SelectionRun>,
) -> FormatResult<()> {
    let rank = dims.len();
    check_hyperslab(dims, start, count)?;
    if count.contains(&0) {
        return Ok(());
    }
    if rank == 0 {
        // A scalar dataspace holds exactly one element and admits only the
        // all/none selections, so its one box is one run of one element.
        runs.push(SelectionRun {
            box_index,
            offset_in_box: 0,
            offset_in_extent: 0,
            len: 1,
        });
        return Ok(());
    }
    let box_strides = row_major_strides(count)?;
    let run_len = count[rank - 1];
    let n_outer = count[..rank - 1]
        .iter()
        .try_fold(1u64, |acc, &c| acc.checked_mul(c))
        .ok_or_else(|| FormatError::InvalidData("selection box element count overflows".into()))?;
    let mut coords = vec![0u64; rank - 1];
    for _ in 0..n_outer {
        let mut offset_in_extent = start[rank - 1];
        let mut offset_in_box = 0u64;
        for d in 0..rank - 1 {
            offset_in_extent += (start[d] + coords[d]) * strides[d];
            offset_in_box += coords[d] * box_strides[d];
        }
        match runs.last_mut() {
            Some(prev)
                if prev.box_index == box_index
                    && prev.offset_in_extent + prev.len == offset_in_extent
                    && prev.offset_in_box + prev.len == offset_in_box =>
            {
                prev.len += run_len;
            }
            _ => runs.push(SelectionRun {
                box_index,
                offset_in_box,
                offset_in_extent,
                len: run_len,
            }),
        }
        for d in (0..rank - 1).rev() {
            coords[d] += 1;
            if coords[d] < count[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    Ok(())
}

/// A decoded H5S dataspace selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// `H5S_SEL_NONE`: no elements selected.
    None,
    /// `H5S_SEL_ALL`: every element of whatever dataspace this selection
    /// is later bound against.
    All,
    /// `H5S_SEL_HYPERSLABS`.
    Hyperslab { rank: usize, form: Hyperslab },
    /// `H5S_SEL_POINTS`.
    Points(PointSelection),
}

impl Selection {
    /// Decode one serialized selection from the front of `buf`.
    ///
    /// Returns the selection and the number of bytes consumed; `buf` may
    /// have trailing bytes belonging to whatever follows (a VDS mapping
    /// entry decodes a source selection immediately followed by a virtual
    /// selection out of the same buffer, for instance).
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 4 {
            return Err(FormatError::BufferTooShort {
                needed: 4,
                available: buf.len(),
            });
        }
        let sel_type = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let body = &buf[4..];
        match sel_type {
            SEL_NONE => {
                let consumed = decode_all_none_body(body)?;
                Ok((Self::None, 4 + consumed))
            }
            SEL_ALL => {
                let consumed = decode_all_none_body(body)?;
                Ok((Self::All, 4 + consumed))
            }
            SEL_HYPERSLABS => {
                let (rank, form, consumed) = decode_hyperslab_body(body)?;
                Ok((Self::Hyperslab { rank, form }, 4 + consumed))
            }
            SEL_POINTS => {
                let (points, consumed) = decode_points_body(body)?;
                Ok((Self::Points(points), 4 + consumed))
            }
            other => Err(FormatError::InvalidData(format!(
                "unknown dataspace selection type {other}"
            ))),
        }
    }

    /// Decompose this selection into `(start, count)` boxes — each
    /// `dims.len()` elements wide — whose union is exactly the selected
    /// elements, with no overlap between boxes. `dims` supplies the full
    /// extent for [`Selection::All`] and must match a hyperslab's `rank`;
    /// it is not otherwise consulted (coordinates are absolute, so this
    /// does not clip an out-of-range block against `dims` — a caller that
    /// needs that validates it itself).
    ///
    /// `Err` for a regular hyperslab whose `count` or `block` holds
    /// [`UNLIMITED`]: an unbounded dimension cannot become a finite box
    /// without a growable-dataset extent this call does not have.
    pub fn to_boxes(&self, dims: &[u64]) -> FormatResult<Vec<(Vec<u64>, Vec<u64>)>> {
        match self {
            Self::None => Ok(Vec::new()),
            Self::All => Ok(vec![(vec![0u64; dims.len()], dims.to_vec())]),
            Self::Points(ps) => {
                if ps.rank != dims.len() {
                    return Err(FormatError::InvalidData(format!(
                        "point selection rank {} does not match the {}-dimensional extent",
                        ps.rank,
                        dims.len()
                    )));
                }
                // Each point is its own 1-element box in every dimension —
                // the only axis-aligned decomposition that holds in
                // general for an arbitrary scatter of coordinates.
                Ok(ps
                    .points
                    .iter()
                    .map(|p| (p.clone(), vec![1u64; ps.rank]))
                    .collect())
            }
            Self::Hyperslab { rank, form } => {
                if *rank != dims.len() {
                    return Err(FormatError::InvalidData(format!(
                        "hyperslab selection rank {rank} does not match the \
                         {}-dimensional extent",
                        dims.len()
                    )));
                }
                match form {
                    Hyperslab::Blocks(blocks) => blocks
                        .iter()
                        .map(|b| {
                            let count = b
                                .start
                                .iter()
                                .zip(&b.end)
                                .map(|(&s, &e)| {
                                    e.checked_sub(s).and_then(|d| d.checked_add(1)).ok_or_else(
                                        || {
                                            FormatError::InvalidData(
                                                "hyperslab block end precedes its start".into(),
                                            )
                                        },
                                    )
                                })
                                .collect::<FormatResult<Vec<u64>>>()?;
                            Ok((b.start.clone(), count))
                        })
                        .collect(),
                    Hyperslab::Regular(r) => regular_hyperslab_to_boxes(r),
                }
            }
        }
    }

    /// Bind this selection to a concrete extent: the boxes it covers plus
    /// the element runs those boxes contribute, in H5S selection order.
    ///
    /// Where [`to_boxes`](Self::to_boxes) answers "which regions", this
    /// answers "which elements, in which order" — the order
    /// `H5S_select_iter_next` walks a selection in, which is what pairs one
    /// selection with another. `H5S_select_project_intersection`
    /// (H5Sselect.c:2402) matches a virtual dataset mapping's two selections
    /// off one element against one element in exactly this order, and asks
    /// of them only `H5S_GET_SELECT_NPOINTS(src) == NPOINTS(dst)` — the same
    /// single condition `H5D_virtual_check_mapping_pre` enforces when the
    /// mapping is created (H5Dvirtual.c:254-257). Neither the two ranks nor
    /// the two box decompositions need agree.
    ///
    /// Unlike [`to_boxes`](Self::to_boxes) this *is* the caller that
    /// validates against `dims`: a run's offset within the extent only means
    /// anything if its box lies inside it, so a box reaching past `dims` is
    /// [`FormatError::InvalidData`] rather than an offset pointing at some
    /// other element.
    pub(crate) fn resolve(&self, dims: &[u64]) -> FormatResult<ResolvedSelection> {
        let boxes = self.to_boxes(dims)?;
        let strides = row_major_strides(dims)?;
        let mut runs = Vec::new();
        for (i, (start, count)) in boxes.iter().enumerate() {
            push_box_runs(i, start, count, dims, &strides, &mut runs)?;
        }
        // Every selection but a point list is walked in increasing row-major
        // coordinate order, and the runs of one box come out in that order
        // already; ordering the boxes against each other is what the sort
        // adds. Disjoint boxes hold disjoint elements, so their runs hold
        // disjoint offset intervals and sorting by the first offset is a
        // total order. A point list keeps the order its coordinates were
        // given in (`H5S__point_iter_next`, H5Spoint.c) and must not be
        // sorted — selection order is the point order.
        if !matches!(self, Self::Points(_)) {
            runs.sort_unstable_by_key(|r| r.offset_in_extent);
        }
        Ok(ResolvedSelection { boxes, runs })
    }

    /// Encode this selection into its `H5S_select_serialize` wire form
    /// (H5Sselect.c and the per-type callbacks in H5Sall.c/H5Snone.c/
    /// H5Spoint.c/H5Shyper.c).
    ///
    /// This module has no file context (no libver bounds), so it always
    /// targets the version libhdf5 itself picks for the *default* low
    /// format-version bound (`H5F_LIBVER_V18`, `H5F_ACS_LIBVER_LOW_BOUND_DEF`
    /// in H5Pfapl.c) with no huge counts or coordinates: version 1 for
    /// points, and version 1 (the block-list wire form) for hyperslabs —
    /// `H5S__hyper_get_version_enc_size` picks version 1 for *any*
    /// hyperslab (regular or not) whenever the low bound is below
    /// `H5F_LIBVER_V112`, so [`Hyperslab::Regular`] is decomposed into
    /// blocks first ([`Selection::to_boxes`]'s own expansion) rather than
    /// written with the version-2/3 REGULAR flag. Confirmed byte-for-byte
    /// against libhdf5-captured `H5Sencode2` images — see
    /// `selection_matches_libhdf5_image` in this module's tests.
    ///
    /// A selection that cannot be expressed in that version-1, 4-byte-wide
    /// form — more than `u32::MAX` points/blocks, or a coordinate that
    /// does not fit `u32` — is [`FormatError::UnsupportedFeature`], not
    /// silently truncated.
    pub fn encode(&self) -> FormatResult<Vec<u8>> {
        match self {
            Self::None => Ok(encode_all_none(SEL_NONE)),
            Self::All => Ok(encode_all_none(SEL_ALL)),
            Self::Points(ps) => encode_points(ps),
            Self::Hyperslab { rank, form } => encode_hyperslab(*rank, form),
        }
    }

    /// The one dimension this selection grows in, or `None` when it is
    /// bounded — `H5S_get_select_unlim_dim`. Only a regular hyperslab can
    /// carry `H5S_UNLIMITED` at all (`H5Sselect_hyperslab` refuses it for
    /// every other form), so every other selection answers `None`.
    pub fn unlim_dim(&self) -> Option<usize> {
        match self {
            Self::Hyperslab {
                form: Hyperslab::Regular(r),
                ..
            } => r.unlim_dim(),
            _ => None,
        }
    }

    /// This selection with its unlimited dimension clipped to an extent of
    /// `clip_size` — `H5S_hyper_clip_unlim` (H5Shyper.c), which is how a
    /// virtual dataset's unlimited mapping becomes the finite one a read
    /// walks.
    ///
    /// The result is a block list rather than a regular hyperslab because the
    /// last block may be cut in half by the extent: upstream builds a span
    /// tree and intersects it with `clip_size` for exactly that case, and an
    /// explicit block list is the same set of elements. A selection that
    /// clips away to nothing becomes [`Selection::None`]; a bounded selection
    /// is returned unchanged.
    pub fn clip_unlimited(&self, clip_size: u64) -> FormatResult<Self> {
        let Self::Hyperslab {
            rank,
            form: Hyperslab::Regular(r),
        } = self
        else {
            return Ok(self.clone());
        };
        let Some(d) = r.unlim_dim() else {
            return Ok(self.clone());
        };
        let (count, block) = RegularHyperslab::clip_diminfo(
            r.start[d],
            r.stride[d],
            r.count[d],
            r.block[d],
            clip_size,
        );
        if count == 0 || block == 0 {
            return Ok(Self::None);
        }
        let mut clipped = r.clone();
        clipped.count[d] = count;
        clipped.block[d] = block;

        let mut blocks = Vec::new();
        for (start, count) in regular_hyperslab_to_boxes(&clipped)? {
            // Trim the block the extent cuts through, and drop one that
            // starts past it — the intersection upstream's span tree takes
            // against `block[unlim_dim] = clip_size`.
            if start[d] >= clip_size {
                continue;
            }
            let mut count = count;
            count[d] = count[d].min(clip_size - start[d]);
            let end = start
                .iter()
                .zip(&count)
                .map(|(&s, &c)| s + c - 1)
                .collect::<Vec<u64>>();
            blocks.push(HyperslabBlock { start, end });
        }
        if blocks.is_empty() {
            return Ok(Self::None);
        }
        Ok(Self::Hyperslab {
            rank: *rank,
            form: Hyperslab::Blocks(blocks),
        })
    }

    /// The inclusive bounding box of the selection — `H5Sget_select_bounds`
    /// — or `None` when the selection has no bounds of its own:
    /// [`Selection::All`] takes the extent it is bound against,
    /// [`Selection::None`] selects nothing, and a regular hyperslab with an
    /// [`UNLIMITED`] count or block has no upper bound until a growable
    /// extent supplies one.
    pub fn bounds(&self) -> Option<(Vec<u64>, Vec<u64>)> {
        match self {
            Self::All | Self::None => None,
            Self::Hyperslab { rank, form } => {
                let blocks = hyperslab_to_block_list(*rank, form).ok()?;
                let first = blocks.first()?;
                let mut lo = first.start.clone();
                let mut hi = first.end.clone();
                for b in &blocks[1..] {
                    for (l, s) in lo.iter_mut().zip(&b.start) {
                        *l = (*l).min(*s);
                    }
                    for (h, e) in hi.iter_mut().zip(&b.end) {
                        *h = (*h).max(*e);
                    }
                }
                Some((lo, hi))
            }
            Self::Points(ps) => {
                let first = ps.points.first()?;
                let mut lo = first.clone();
                let mut hi = first.clone();
                for p in &ps.points[1..] {
                    for (l, c) in lo.iter_mut().zip(p) {
                        *l = (*l).min(*c);
                    }
                    for (h, c) in hi.iter_mut().zip(p) {
                        *h = (*h).max(*c);
                    }
                }
                Some((lo, hi))
            }
        }
    }
}

/// Cast a decoded/caller-supplied coordinate down to the 4-byte width
/// this module's version-1-only [`Selection::encode`] writes.
fn to_u32(v: u64) -> FormatResult<u32> {
    u32::try_from(v).map_err(|_| {
        FormatError::UnsupportedFeature(format!(
            "value {v} exceeds the 4-byte encoding Selection::encode targets (version 1, \
             matching libhdf5's default H5F_LIBVER_V18 low format-version bound)"
        ))
    })
}

fn encode_all_none(sel_type: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&sel_type.to_le_bytes());
    buf.extend_from_slice(&ALL_NONE_VERSION.to_le_bytes());
    buf.extend_from_slice(&[0u8; 8]);
    buf
}

fn encode_points(ps: &PointSelection) -> FormatResult<Vec<u8>> {
    if ps.rank == 0 || ps.rank > MAX_RANK {
        return Err(FormatError::InvalidData(format!(
            "invalid point selection rank {}",
            ps.rank
        )));
    }
    for p in &ps.points {
        if p.len() != ps.rank {
            return Err(FormatError::InvalidData(format!(
                "point selection coordinate length {} does not match rank {}",
                p.len(),
                ps.rank
            )));
        }
    }
    let num_points: u32 = ps.points.len().try_into().map_err(|_| {
        FormatError::UnsupportedFeature(format!(
            "point selection has {} points, too many for version-1 encode (u32 count)",
            ps.points.len()
        ))
    })?;
    let rank_u32 = ps.rank as u32;
    let payload_bytes: u32 = num_points
        .checked_mul(4)
        .and_then(|v| v.checked_mul(rank_u32))
        .ok_or_else(|| {
            FormatError::UnsupportedFeature(
                "point selection payload too large for version-1 encode".into(),
            )
        })?;
    let len = 8u32.checked_add(payload_bytes).ok_or_else(|| {
        FormatError::UnsupportedFeature(
            "point selection payload too large for version-1 encode".into(),
        )
    })?;

    let mut buf = Vec::with_capacity(24 + payload_bytes as usize);
    buf.extend_from_slice(&SEL_POINTS.to_le_bytes());
    buf.extend_from_slice(&POINT_VERSION_1.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // padding
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&rank_u32.to_le_bytes());
    buf.extend_from_slice(&num_points.to_le_bytes());
    for p in &ps.points {
        for &c in p {
            buf.extend_from_slice(&to_u32(c)?.to_le_bytes());
        }
    }
    Ok(buf)
}

/// Normalize either wire form of a hyperslab selection into an explicit
/// block list — the form [`encode_hyperslab`] writes regardless of which
/// form `form` holds (see [`Selection::encode`]'s doc comment for why).
fn hyperslab_to_block_list(rank: usize, form: &Hyperslab) -> FormatResult<Vec<HyperslabBlock>> {
    match form {
        Hyperslab::Blocks(blocks) => {
            for b in blocks {
                if b.start.len() != rank || b.end.len() != rank {
                    return Err(FormatError::InvalidData(format!(
                        "hyperslab block coordinate length does not match rank {rank}"
                    )));
                }
            }
            Ok(blocks.clone())
        }
        Hyperslab::Regular(r) => {
            if r.start.len() != rank
                || r.stride.len() != rank
                || r.count.len() != rank
                || r.block.len() != rank
            {
                return Err(FormatError::InvalidData(format!(
                    "regular hyperslab field length does not match rank {rank}"
                )));
            }
            regular_hyperslab_to_boxes(r)?
                .into_iter()
                .map(|(start, count)| {
                    let end = start
                        .iter()
                        .zip(&count)
                        .map(|(&s, &c)| {
                            s.checked_add(c - 1).ok_or_else(|| {
                                FormatError::InvalidData(
                                    "hyperslab box coordinate overflows".into(),
                                )
                            })
                        })
                        .collect::<FormatResult<Vec<u64>>>()?;
                    Ok(HyperslabBlock { start, end })
                })
                .collect()
        }
    }
}

/// The version-2 REGULAR wire form: `(start, stride, count, block)` per
/// dimension at 8 bytes each, which is the only encoding that can carry
/// `H5S_UNLIMITED` at the default low format-version bound
/// (`H5S__hyper_serialize`'s `version == 2` arm).
fn encode_regular_hyperslab_v2(rank: usize, r: &RegularHyperslab) -> FormatResult<Vec<u8>> {
    if r.start.len() != rank
        || r.stride.len() != rank
        || r.count.len() != rank
        || r.block.len() != rank
    {
        return Err(FormatError::InvalidData(format!(
            "regular hyperslab field length does not match rank {rank}"
        )));
    }
    let mut buf = Vec::with_capacity(17 + rank * 32);
    buf.extend_from_slice(&SEL_HYPERSLABS.to_le_bytes());
    buf.extend_from_slice(&HYPER_VERSION_2.to_le_bytes());
    buf.push(HYPER_REGULAR_FLAG);
    // The length field version 2 keeps where version 3 keeps its encoded-size
    // tag: the rank field plus the four 8-byte fields per dimension, exactly
    // what `H5S__hyper_serialize` accumulates into `len`. `rank <= MAX_RANK`,
    // so this cannot overflow.
    let len = 4u32 + 32 * rank as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&(rank as u32).to_le_bytes());
    for d in 0..rank {
        buf.extend_from_slice(&r.start[d].to_le_bytes());
        buf.extend_from_slice(&r.stride[d].to_le_bytes());
        buf.extend_from_slice(&r.count[d].to_le_bytes());
        buf.extend_from_slice(&r.block[d].to_le_bytes());
    }
    Ok(buf)
}

fn encode_hyperslab(rank: usize, form: &Hyperslab) -> FormatResult<Vec<u8>> {
    if rank == 0 || rank > MAX_RANK {
        return Err(FormatError::InvalidData(format!(
            "invalid hyperslab selection rank {rank}"
        )));
    }
    // `H5S__hyper_get_version_enc_size` takes `MAX(H5S_HYPER_VERSION_2, ...)`
    // the moment the selection has an unlimited dimension, whatever the low
    // format-version bound: version 1 is a block list, and a list of blocks
    // cannot spell an unbounded one. Version 2 fixes the encoded width at 8
    // bytes, so `H5S_UNLIMITED` goes out as the all-ones field `read_dim`
    // reads back.
    if let Hyperslab::Regular(r) = form {
        if r.unlim_dim().is_some() {
            return encode_regular_hyperslab_v2(rank, r);
        }
    }
    let blocks = hyperslab_to_block_list(rank, form)?;
    let num_blocks: u32 = blocks.len().try_into().map_err(|_| {
        FormatError::UnsupportedFeature(format!(
            "hyperslab selection has {} blocks, too many for version-1 encode (u32 count)",
            blocks.len()
        ))
    })?;
    let rank_u32 = rank as u32;

    let mut coords: Vec<u32> = Vec::with_capacity(blocks.len() * rank * 2);
    for b in &blocks {
        for &s in &b.start {
            coords.push(to_u32(s)?);
        }
        for &e in &b.end {
            coords.push(to_u32(e)?);
        }
    }

    let block_payload = 8u32
        .checked_mul(rank_u32)
        .and_then(|v| v.checked_mul(num_blocks))
        .ok_or_else(|| {
            FormatError::UnsupportedFeature(
                "hyperslab selection too large for version-1 encode".into(),
            )
        })?;
    let len = 8u32.checked_add(block_payload).ok_or_else(|| {
        FormatError::UnsupportedFeature("hyperslab selection too large for version-1 encode".into())
    })?;

    let mut buf = Vec::with_capacity(24 + coords.len() * 4);
    buf.extend_from_slice(&SEL_HYPERSLABS.to_le_bytes());
    buf.extend_from_slice(&HYPER_VERSION_1.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // padding
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&rank_u32.to_le_bytes());
    buf.extend_from_slice(&num_blocks.to_le_bytes());
    for v in coords {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Ok(buf)
}

/// Cap on the number of boxes [`Selection::to_boxes`] will materialize for
/// a REGULAR hyperslab (`count[0] * count[1] * ...`). `count` comes
/// straight off the wire with no relation to buffer size the way a
/// block-list's `num_blocks` is implicitly bounded by (each block consumes
/// buffer bytes; a `(start, stride, count, block)` tuple does not), so an
/// unchecked expansion could turn a few dozen bytes of crafted input into
/// an unbounded allocation.
const MAX_REGULAR_BOXES: u64 = 1 << 20;

fn regular_hyperslab_to_boxes(r: &RegularHyperslab) -> FormatResult<Vec<(Vec<u64>, Vec<u64>)>> {
    let rank = r.start.len();
    if r.count.contains(&UNLIMITED) || r.block.contains(&UNLIMITED) {
        return Err(FormatError::UnsupportedFeature(
            "unlimited (H5S_UNLIMITED) regular hyperslab dimension".into(),
        ));
    }
    if r.count.contains(&0) || r.block.contains(&0) {
        return Ok(Vec::new());
    }
    let total_boxes = r
        .count
        .iter()
        .try_fold(1u64, |acc, &c| acc.checked_mul(c))
        .ok_or_else(|| FormatError::InvalidData("regular hyperslab box count overflows".into()))?;
    if total_boxes > MAX_REGULAR_BOXES {
        return Err(FormatError::UnsupportedFeature(format!(
            "regular hyperslab selection expands to {total_boxes} boxes, over the \
             {MAX_REGULAR_BOXES} cap"
        )));
    }

    let mut boxes = Vec::with_capacity(total_boxes as usize);
    let mut idx = vec![0u64; rank];
    for _ in 0..total_boxes {
        let start: Vec<u64> = (0..rank)
            .map(|d| r.start[d] + idx[d] * r.stride[d])
            .collect();
        boxes.push((start, r.block.clone()));
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < r.count[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(boxes)
}

fn decode_all_none_body(buf: &[u8]) -> FormatResult<usize> {
    if buf.len() < 4 + 8 {
        return Err(FormatError::BufferTooShort {
            needed: 4 + 8,
            available: buf.len(),
        });
    }
    let version = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if version != ALL_NONE_VERSION {
        return Err(FormatError::InvalidData(format!(
            "bad version {version} for all/none dataspace selection"
        )));
    }
    // 8 reserved bytes, unconditionally skipped (H5Sall.c / H5Snone.c).
    Ok(4 + 8)
}

fn decode_hyperslab_body(buf: &[u8]) -> FormatResult<(usize, Hyperslab, usize)> {
    let mut pos = 0usize;
    if buf.len() < 4 {
        return Err(FormatError::BufferTooShort {
            needed: 4,
            available: buf.len(),
        });
    }
    let version = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    pos += 4;
    if !(HYPER_VERSION_1..=HYPER_VERSION_3).contains(&version) {
        return Err(FormatError::InvalidData(format!(
            "bad version {version} for hyperslab dataspace selection"
        )));
    }

    let mut flags = 0u8;
    let enc_size: usize;
    if version >= HYPER_VERSION_2 {
        if buf.len() < pos + 1 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 1,
                available: buf.len(),
            });
        }
        flags = buf[pos];
        pos += 1;
        if flags & !HYPER_REGULAR_FLAG != 0 {
            return Err(FormatError::InvalidData(format!(
                "unknown hyperslab selection flag bits in {flags:#x}"
            )));
        }

        if version >= HYPER_VERSION_3 {
            if buf.len() < pos + 1 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 1,
                    available: buf.len(),
                });
            }
            enc_size = match buf[pos] {
                0x02 => 2,
                0x04 => 4,
                0x08 => 8,
                other => {
                    return Err(FormatError::InvalidData(format!(
                        "unknown hyperslab selection encoding size tag {other:#x}"
                    )))
                }
            };
            pos += 1;
        } else {
            // Version 2: 4 reserved bytes, encoding size fixed at 8.
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            pos += 4;
            enc_size = 8;
        }
    } else {
        // Version 1: 8 reserved bytes, encoding size fixed at 4.
        if buf.len() < pos + 8 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 8,
                available: buf.len(),
            });
        }
        pos += 8;
        enc_size = 4;
    }

    if buf.len() < pos + 4 {
        return Err(FormatError::BufferTooShort {
            needed: pos + 4,
            available: buf.len(),
        });
    }
    let rank = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
    pos += 4;
    if rank == 0 || rank > MAX_RANK {
        return Err(FormatError::InvalidData(format!(
            "invalid hyperslab selection rank {rank}"
        )));
    }

    if flags & HYPER_REGULAR_FLAG != 0 {
        let mut start = Vec::with_capacity(rank);
        let mut stride = Vec::with_capacity(rank);
        let mut count = Vec::with_capacity(rank);
        let mut block = Vec::with_capacity(rank);
        for _ in 0..rank {
            if buf.len() < pos + 4 * enc_size {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4 * enc_size,
                    available: buf.len(),
                });
            }
            start.push(read_plain(&buf[pos..], enc_size));
            pos += enc_size;
            stride.push(read_plain(&buf[pos..], enc_size));
            pos += enc_size;
            count.push(read_dim(&buf[pos..], enc_size));
            pos += enc_size;
            block.push(read_dim(&buf[pos..], enc_size));
            pos += enc_size;
        }
        Ok((
            rank,
            Hyperslab::Regular(RegularHyperslab {
                start,
                stride,
                count,
                block,
            }),
            pos,
        ))
    } else {
        if buf.len() < pos + enc_size {
            return Err(FormatError::BufferTooShort {
                needed: pos + enc_size,
                available: buf.len(),
            });
        }
        let num_blocks = read_plain(&buf[pos..], enc_size) as usize;
        pos += enc_size;

        // No `Vec::with_capacity(num_blocks)`: `num_blocks` is untrusted
        // file data up to a 64-bit field, and the per-iteration buffer
        // check below already bounds the loop to at most
        // `buf.len() / (rank * 2 * enc_size)` iterations, so a corrupt
        // huge count fails with `BufferTooShort` instead of an
        // allocation blowup.
        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            let mut start = Vec::with_capacity(rank);
            let mut end = Vec::with_capacity(rank);
            if buf.len() < pos + 2 * rank * enc_size {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 2 * rank * enc_size,
                    available: buf.len(),
                });
            }
            for _ in 0..rank {
                start.push(read_plain(&buf[pos..], enc_size));
                pos += enc_size;
            }
            for _ in 0..rank {
                end.push(read_plain(&buf[pos..], enc_size));
                pos += enc_size;
            }
            blocks.push(HyperslabBlock { start, end });
        }
        Ok((rank, Hyperslab::Blocks(blocks), pos))
    }
}

fn decode_points_body(buf: &[u8]) -> FormatResult<(PointSelection, usize)> {
    let mut pos = 0usize;
    if buf.len() < 4 {
        return Err(FormatError::BufferTooShort {
            needed: 4,
            available: buf.len(),
        });
    }
    let version = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    pos += 4;
    if version != POINT_VERSION_1 && version != POINT_VERSION_2 {
        return Err(FormatError::InvalidData(format!(
            "bad version {version} for point dataspace selection"
        )));
    }

    let enc_size: usize;
    if version >= POINT_VERSION_2 {
        if buf.len() < pos + 1 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 1,
                available: buf.len(),
            });
        }
        enc_size = match buf[pos] {
            0x02 => 2,
            0x04 => 4,
            0x08 => 8,
            other => {
                return Err(FormatError::InvalidData(format!(
                    "unknown point selection encoding size tag {other:#x}"
                )))
            }
        };
        pos += 1;
    } else {
        // Version 1: 4 padding bytes + a 4-byte length field, neither of
        // which this decoder validates (H5S__point_deserialize doesn't
        // either — the length is a serialize-side convenience, not a
        // decode-side check). Encoding size is fixed at 4.
        if buf.len() < pos + 8 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 8,
                available: buf.len(),
            });
        }
        pos += 8;
        enc_size = 4;
    }

    if buf.len() < pos + 4 {
        return Err(FormatError::BufferTooShort {
            needed: pos + 4,
            available: buf.len(),
        });
    }
    let rank = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
    pos += 4;
    if rank == 0 || rank > MAX_RANK {
        return Err(FormatError::InvalidData(format!(
            "invalid point selection rank {rank}"
        )));
    }

    if buf.len() < pos + enc_size {
        return Err(FormatError::BufferTooShort {
            needed: pos + enc_size,
            available: buf.len(),
        });
    }
    let num_points = read_plain(&buf[pos..], enc_size);
    pos += enc_size;

    // Mirrors H5S__point_deserialize's own overflow guard: `rank *
    // enc_size * num_points` is computed in checked 64-bit arithmetic
    // before it is trusted as a buffer offset, so a crafted huge
    // `num_points` fails cleanly here instead of via an under-allocated
    // `Vec::with_capacity`.
    let point_bytes = (rank as u64)
        .checked_mul(enc_size as u64)
        .and_then(|per_point| per_point.checked_mul(num_points))
        .ok_or_else(|| {
            FormatError::InvalidData("point selection coordinate buffer size overflows".into())
        })?;
    if (buf.len() as u64) < pos as u64 + point_bytes {
        let needed = usize::try_from(point_bytes)
            .ok()
            .and_then(|b| pos.checked_add(b))
            .unwrap_or(usize::MAX);
        return Err(FormatError::BufferTooShort {
            needed,
            available: buf.len(),
        });
    }

    let mut points = Vec::with_capacity(num_points as usize);
    for _ in 0..num_points {
        let mut coord = Vec::with_capacity(rank);
        for _ in 0..rank {
            coord.push(read_plain(&buf[pos..], enc_size));
            pos += enc_size;
        }
        points.push(coord);
    }

    Ok((PointSelection { rank, points }, pos))
}

/// Decode an `n`-byte little-endian field with no sentinel substitution
/// (`start`, `stride`, and block-list coordinates).
fn read_plain(buf: &[u8], n: usize) -> u64 {
    crate::format::bytes::read_le_uint(buf, n)
}

/// Decode an `n`-byte little-endian `count` or `block` field, mapping an
/// all-ones field of that width to [`UNLIMITED`] — `H5S_UINT16_MAX` /
/// `H5S_UINT32_MAX` / `H5S_UINT64_MAX` in H5Shyper.c, one sentinel per
/// encoding width since a narrower field cannot spell `HSIZE_UNDEF`
/// itself.
fn read_dim(buf: &[u8], n: usize) -> u64 {
    let v = read_plain(buf, n);
    let all_ones = if n >= 8 {
        u64::MAX
    } else {
        (1u64 << (n * 8)) - 1
    };
    if v == all_ones {
        UNLIMITED
    } else {
        v
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Strip an `H5Sencode2` envelope (`tests/fixtures/gen_selection_images.c`
    /// captures the whole blob) down to the selection-only bytes this
    /// module decodes: `type(1) + version(1) + sizeof_size(1) +
    /// extent_size(4 LE)` followed by `extent_size` bytes of serialized
    /// extent (H5S.c `H5S_encode`), then the selection.
    fn strip_h5sencode_envelope(blob: &[u8]) -> &[u8] {
        let extent_size = u32::from_le_bytes([blob[3], blob[4], blob[5], blob[6]]) as usize;
        &blob[7 + extent_size..]
    }

    /// Byte-for-byte the source selection h5debug reported inside a real
    /// h5py-written VDS mapping entry (`layout[...] = VirtualSource(...)`,
    /// full-extent mapping): H5S_SEL_ALL, version 1, 8 reserved bytes.
    #[test]
    fn decode_all_selection() {
        let buf = [
            0x03, 0x00, 0x00, 0x00, // type = SEL_ALL
            0x01, 0x00, 0x00, 0x00, // version = 1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
        ];
        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(sel, Selection::All);
    }

    #[test]
    fn decode_none_selection() {
        let buf = [
            0x00, 0x00, 0x00, 0x00, // type = SEL_NONE
            0x01, 0x00, 0x00, 0x00, // version = 1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
        ];
        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(sel, Selection::None);
    }

    /// A trailing selection right after this one must be left untouched —
    /// a VDS mapping entry decodes a source selection immediately
    /// followed by a virtual selection out of one shared buffer.
    #[test]
    fn decode_all_selection_leaves_trailer_untouched() {
        let mut buf = vec![0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        buf.extend_from_slice(&[0xAA; 4]);
        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(sel, Selection::All);
        assert_eq!(consumed, 16);
        assert_eq!(&buf[consumed..], &[0xAA; 4]);
    }

    /// Byte-for-byte a real h5py-written VDS mapping's virtual selection
    /// for `layout[4:12] = VirtualSource(..., shape=(8,))`: hyperslab,
    /// version 1 (so no flags byte — always the block-list form), rank 1,
    /// one block spanning element 4 through 11 inclusive.
    #[test]
    fn h5py_single_block_selection_is_version_one() {
        let mut buf = vec![0x02, 0, 0, 0]; // type = SEL_HYPERSLABS
        buf.extend_from_slice(&1u32.to_le_bytes()); // version = 1
        buf.extend_from_slice(&[0u8; 8]); // reserved
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        buf.extend_from_slice(&4u32.to_le_bytes()); // block 0 start
        buf.extend_from_slice(&11u32.to_le_bytes()); // block 0 end (inclusive)

        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        match sel {
            Selection::Hyperslab {
                rank,
                form: Hyperslab::Blocks(blocks),
            } => {
                assert_eq!(rank, 1);
                assert_eq!(
                    blocks,
                    vec![HyperslabBlock {
                        start: vec![4],
                        end: vec![11],
                    }]
                );
            }
            other => panic!("expected a version-1 block list, got {other:?}"),
        }
    }

    #[test]
    fn decode_hyperslab_block_list_multi_block_2d() {
        let mut buf = vec![0x02, 0, 0, 0]; // SEL_HYPERSLABS
        buf.extend_from_slice(&1u32.to_le_bytes()); // version 1
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&2u32.to_le_bytes()); // rank = 2
        buf.extend_from_slice(&2u32.to_le_bytes()); // num_blocks = 2
                                                    // block 0: (0,0)..(1,1)
        for v in [0u32, 0, 1, 1] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // block 1: (2,2)..(3,3)
        for v in [2u32, 2, 3, 3] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        match sel {
            Selection::Hyperslab {
                rank,
                form: Hyperslab::Blocks(blocks),
            } => {
                assert_eq!(rank, 2);
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].start, vec![0, 0]);
                assert_eq!(blocks[0].end, vec![1, 1]);
                assert_eq!(blocks[1].start, vec![2, 2]);
                assert_eq!(blocks[1].end, vec![3, 3]);
            }
            other => panic!("expected block list, got {other:?}"),
        }
    }

    /// A version-2, REGULAR-flagged hyperslab: start/stride/count/block per
    /// dimension, 8-byte fields (the version-2 default when no explicit
    /// encoding-size byte is present).
    #[test]
    fn decode_regular_hyperslab_v2() {
        let mut buf = vec![0x02, 0, 0, 0]; // SEL_HYPERSLABS
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.push(0x01); // flags = REGULAR
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&2u64.to_le_bytes()); // start
        buf.extend_from_slice(&4u64.to_le_bytes()); // stride
        buf.extend_from_slice(&3u64.to_le_bytes()); // count
        buf.extend_from_slice(&2u64.to_le_bytes()); // block

        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        match sel {
            Selection::Hyperslab {
                rank,
                form: Hyperslab::Regular(r),
            } => {
                assert_eq!(rank, 1);
                assert_eq!(r.start, vec![2]);
                assert_eq!(r.stride, vec![4]);
                assert_eq!(r.count, vec![3]);
                assert_eq!(r.block, vec![2]);
            }
            other => panic!("expected a regular hyperslab, got {other:?}"),
        }
    }

    /// A version-3, REGULAR-flagged hyperslab with an explicit 2-byte
    /// encoding size and an unlimited count — the all-ones sentinel for
    /// that width must decode to [`UNLIMITED`], not `0xFFFF` taken
    /// literally.
    #[test]
    fn decode_regular_hyperslab_v3_unlimited_count() {
        let mut buf = vec![0x02, 0, 0, 0]; // SEL_HYPERSLABS
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.push(0x01); // flags = REGULAR
        buf.push(0x02); // enc_size tag = 2 bytes
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&0u16.to_le_bytes()); // start = 0
        buf.extend_from_slice(&5u16.to_le_bytes()); // stride = 5
        buf.extend_from_slice(&0xFFFFu16.to_le_bytes()); // count = UNLIMITED
        buf.extend_from_slice(&3u16.to_le_bytes()); // block = 3

        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        match sel {
            Selection::Hyperslab {
                form: Hyperslab::Regular(r),
                ..
            } => {
                assert_eq!(r.count, vec![UNLIMITED]);
                assert_eq!(r.block, vec![3]);
            }
            other => panic!("expected a regular hyperslab, got {other:?}"),
        }
    }

    #[test]
    fn decode_points_truncated_header_is_buffer_too_short() {
        let buf = [0x01, 0, 0, 0]; // type = SEL_POINTS, no body at all
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    /// Byte-for-byte a libhdf5-captured `H5Sencode2` image
    /// (`tests/fixtures/gen_selection_images.c`, `points4_v1.bin`): a
    /// version-1 point selection, rank 1, points at 1/3/7/15.
    #[test]
    fn decode_points_matches_libhdf5_image() {
        let blob = include_bytes!("../../tests/fixtures/points4_v1.bin");
        let sel_bytes = strip_h5sencode_envelope(blob);
        let (sel, consumed) = Selection::decode(sel_bytes).unwrap();
        assert_eq!(consumed, sel_bytes.len());
        assert_eq!(
            sel,
            Selection::Points(PointSelection {
                rank: 1,
                points: vec![vec![1], vec![3], vec![7], vec![15]],
            })
        );
    }

    #[test]
    fn decode_points_version_2_small_enc_size() {
        let mut buf = vec![0x01, 0, 0, 0]; // SEL_POINTS
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.push(0x02); // enc_size tag = 2 bytes
        buf.extend_from_slice(&2u32.to_le_bytes()); // rank = 2
        buf.extend_from_slice(&2u16.to_le_bytes()); // num_points = 2
        buf.extend_from_slice(&1u16.to_le_bytes()); // point 0 = (1, 5)
        buf.extend_from_slice(&5u16.to_le_bytes());
        buf.extend_from_slice(&9u16.to_le_bytes()); // point 1 = (9, 0)
        buf.extend_from_slice(&0u16.to_le_bytes());

        let (sel, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(
            sel,
            Selection::Points(PointSelection {
                rank: 2,
                points: vec![vec![1, 5], vec![9, 0]],
            })
        );
    }

    #[test]
    fn decode_points_rejects_bad_version() {
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3 does not exist
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_points_rejects_unknown_enc_size_tag() {
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.push(0x03); // not one of 2/4/8
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_points_rejects_zero_rank() {
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&0u32.to_le_bytes()); // rank = 0
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_points_rejects_rank_over_max() {
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&33u32.to_le_bytes()); // rank = 33 > 32
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_points_truncated_coordinates() {
        // Claims 5 points of rank 1 but the buffer only has room for one.
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&1u32.to_le_bytes()); // version 1
        buf.extend_from_slice(&[0u8; 8]); // padding + length placeholder
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&5u32.to_le_bytes()); // num_points = 5 (lie)
        buf.extend_from_slice(&0u32.to_le_bytes());
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    /// A `num_points` claim that comfortably fits the `rank * enc_size *
    /// num_points` multiplication (so it doesn't hit the overflow guard)
    /// but wildly exceeds the actual buffer must still fail on the
    /// bounds check rather than attempting to allocate.
    #[test]
    fn decode_points_huge_num_points_claim_does_not_allocate() {
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.push(0x08); // enc_size tag = 8 bytes
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&(1u64 << 40).to_le_bytes()); // num_points (lie)
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    /// A `num_points` claim that overflows the `rank * enc_size *
    /// num_points` multiplication itself must fail cleanly rather than
    /// wrapping into a small, incorrect buffer requirement.
    #[test]
    fn decode_points_num_points_overflow_does_not_allocate() {
        let mut buf = vec![0x01, 0, 0, 0];
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.push(0x08); // enc_size tag = 8 bytes
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&(u64::MAX - 1).to_le_bytes()); // num_points (lie)
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn to_boxes_points_each_point_is_its_own_unit_box() {
        let sel = Selection::Points(PointSelection {
            rank: 2,
            points: vec![vec![1, 2], vec![5, 5]],
        });
        let boxes = sel.to_boxes(&[8, 8]).unwrap();
        assert_eq!(
            boxes,
            vec![(vec![1, 2], vec![1, 1]), (vec![5, 5], vec![1, 1])]
        );
    }

    #[test]
    fn to_boxes_points_rejects_rank_mismatch() {
        let sel = Selection::Points(PointSelection {
            rank: 2,
            points: vec![],
        });
        let err = sel.to_boxes(&[8]).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_unknown_type_is_invalid() {
        let buf = [0x09, 0, 0, 0];
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_bad_all_version() {
        let buf = [0x03, 0, 0, 0, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_bad_hyperslab_version() {
        let buf = [0x02, 0, 0, 0, 0x04, 0, 0, 0];
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_hyperslab_rejects_unknown_flag_bits() {
        let mut buf = vec![0x02, 0, 0, 0];
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.push(0x02); // bit 1 is not a known flag
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&1u32.to_le_bytes());
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_hyperslab_rejects_zero_rank() {
        let mut buf = vec![0x02, 0, 0, 0];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&0u32.to_le_bytes()); // rank = 0
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_hyperslab_rejects_rank_over_max() {
        let mut buf = vec![0x02, 0, 0, 0];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&33u32.to_le_bytes()); // rank = 33 > 32
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_truncated_header() {
        let buf = [0x03, 0, 0];
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn decode_truncated_hyperslab_blocks() {
        // Claims 5 blocks of rank 1 but the buffer only has room for one.
        let mut buf = vec![0x02, 0, 0, 0];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&5u32.to_le_bytes()); // num_blocks = 5 (lie)
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    /// A `num_blocks` claim near `u64::MAX` (the widest on-disk field, via
    /// an 8-byte encoding size) must fail on the first bounds check rather
    /// than attempting to allocate — this is the case
    /// `Vec::with_capacity(num_blocks)` would have been unsafe for.
    #[test]
    fn decode_huge_num_blocks_claim_does_not_allocate() {
        let mut buf = vec![0x02, 0, 0, 0];
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.push(0x00); // flags = 0 (block list)
        buf.push(0x08); // enc_size tag = 8 bytes
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        buf.extend_from_slice(&(u64::MAX - 1).to_le_bytes()); // num_blocks (lie)
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    // --------------------------------------------------------------- to_boxes

    /// A multi-block selection bounds to the box that covers every block, the
    /// same answer `H5Sget_select_bounds` gives. Moved here with the decoder
    /// it belongs to, from `format::reference`'s own copy.
    #[test]
    fn bounds_cover_every_block_and_point() {
        let hyper = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Blocks(vec![
                HyperslabBlock {
                    start: vec![4, 1],
                    end: vec![5, 3],
                },
                HyperslabBlock {
                    start: vec![0, 6],
                    end: vec![1, 7],
                },
            ]),
        };
        assert_eq!(hyper.bounds(), Some((vec![0, 1], vec![5, 7])));
        let points = Selection::Points(PointSelection {
            rank: 2,
            points: vec![vec![3, 9], vec![7, 2]],
        });
        assert_eq!(points.bounds(), Some((vec![3, 2], vec![7, 9])));
        assert_eq!(Selection::All.bounds(), None);
        assert_eq!(Selection::None.bounds(), None);
    }

    /// A regular hyperslab lists no blocks; it stores the
    /// start/stride/count/block pattern, and its bounds cover the blocks that
    /// pattern expands to.
    #[test]
    fn bounds_of_a_regular_hyperslab_cover_its_expansion() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SEL_HYPERSLABS.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.push(HYPER_REGULAR_FLAG);
        buf.push(4); // coordinate width
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank
        for v in [2u32, 5, 3, 2] {
            // start 2, stride 5, count 3, block 2
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let (selection, consumed) = Selection::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        // Blocks [2,3], [7,8], [12,13].
        assert_eq!(selection.bounds(), Some((vec![2], vec![13])));
    }

    /// A regular hyperslab with no upper bound has no bounding box either,
    /// rather than one built from the `UNLIMITED` sentinel.
    #[test]
    fn bounds_of_an_unlimited_regular_hyperslab_are_absent() {
        let selection = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![UNLIMITED],
                block: vec![1],
            }),
        };
        assert_eq!(selection.bounds(), None);
    }

    #[test]
    fn to_boxes_all_covers_the_full_extent() {
        let boxes = Selection::All.to_boxes(&[3, 5]).unwrap();
        assert_eq!(boxes, vec![(vec![0, 0], vec![3, 5])]);
    }

    #[test]
    fn to_boxes_none_is_empty() {
        assert_eq!(Selection::None.to_boxes(&[3, 5]).unwrap(), vec![]);
    }

    #[test]
    fn to_boxes_single_block_matches_h5py_fixture() {
        // layout[4:12] = VirtualSource(..., shape=(8,)): one block, [4, 11].
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Blocks(vec![HyperslabBlock {
                start: vec![4],
                end: vec![11],
            }]),
        };
        let boxes = sel.to_boxes(&[20]).unwrap();
        assert_eq!(boxes, vec![(vec![4], vec![8])]);
    }

    #[test]
    fn to_boxes_multi_block_2d() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Blocks(vec![
                HyperslabBlock {
                    start: vec![0, 0],
                    end: vec![1, 1],
                },
                HyperslabBlock {
                    start: vec![2, 2],
                    end: vec![3, 3],
                },
            ]),
        };
        let boxes = sel.to_boxes(&[4, 4]).unwrap();
        assert_eq!(
            boxes,
            vec![(vec![0, 0], vec![2, 2]), (vec![2, 2], vec![2, 2])]
        );
    }

    /// Runs come out in H5S element order, which for a multi-block
    /// hyperslab is *not* the order the boxes come out in: the 2x2-block
    /// grid over a 4x4 extent decomposes into boxes
    /// `(0,0) (0,2) (2,0) (2,2)`, but the elements are visited row by row,
    /// so the run list is `(0,0) (0,2) (1,0) (1,2) (2,0) ...`.
    #[test]
    fn resolve_orders_runs_by_element_not_by_box() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0, 0],
                stride: vec![2, 2],
                count: vec![2, 2],
                block: vec![2, 2],
            }),
        };
        let resolved = sel.resolve(&[4, 4]).unwrap();
        assert_eq!(resolved.n_elements(), 16);
        // Every element of the 4x4 extent, once, in row-major order.
        let mut covered = Vec::new();
        for r in &resolved.runs {
            for k in 0..r.len {
                covered.push(r.offset_in_extent + k);
            }
        }
        assert_eq!(covered, (0..16).collect::<Vec<u64>>());
        // The first two runs are the two halves of row 0, taken from
        // different boxes.
        assert_eq!(resolved.runs[0].box_index, 0);
        assert_eq!(resolved.runs[1].box_index, 1);
        assert_eq!(resolved.runs[0].len, 2);
        // Row 1 of box 0 sits two elements into that box's own buffer.
        assert_eq!(resolved.runs[2].box_index, 0);
        assert_eq!(resolved.runs[2].offset_in_box, 2);
    }

    /// A box that covers a whole trailing region is contiguous in both the
    /// box and the extent, so it collapses to one run rather than one per
    /// row — the same merge a dual-array walk performs.
    #[test]
    fn resolve_coalesces_a_box_that_fills_its_extent() {
        let resolved = Selection::All.resolve(&[3, 4]).unwrap();
        assert_eq!(resolved.runs.len(), 1);
        assert_eq!(resolved.runs[0].len, 12);
        assert_eq!(resolved.n_elements(), 12);
    }

    /// A partial box cannot coalesce across its second dimension: rows are
    /// separated by the extent's stride.
    #[test]
    fn resolve_keeps_one_run_per_row_of_a_partial_box() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Blocks(vec![HyperslabBlock {
                start: vec![1, 1],
                end: vec![2, 2],
            }]),
        };
        let resolved = sel.resolve(&[4, 4]).unwrap();
        assert_eq!(resolved.runs.len(), 2);
        assert_eq!(resolved.runs[0].offset_in_extent, 5);
        assert_eq!(resolved.runs[0].offset_in_box, 0);
        assert_eq!(resolved.runs[1].offset_in_extent, 9);
        assert_eq!(resolved.runs[1].offset_in_box, 2);
    }

    /// Selection order for a point list is the order the coordinates were
    /// given in (`H5S__point_iter_next`, H5Spoint.c), not coordinate order,
    /// so the runs must not be sorted.
    #[test]
    fn resolve_keeps_point_selection_order() {
        let sel = Selection::Points(PointSelection {
            rank: 2,
            points: vec![vec![2, 3], vec![0, 1], vec![1, 0]],
        });
        let resolved = sel.resolve(&[4, 4]).unwrap();
        assert_eq!(
            resolved
                .runs
                .iter()
                .map(|r| r.offset_in_extent)
                .collect::<Vec<u64>>(),
            vec![11, 1, 4]
        );
        assert!(resolved.runs.iter().all(|r| r.len == 1));
    }

    /// `to_boxes` deliberately does not clip a block against the extent, so
    /// `resolve` — which needs every box's offset within that extent to mean
    /// something — is the caller that rejects one reaching past it.
    /// The one bounds rule, at each of its boundaries: an edge on the
    /// extent is in, one past it is out, a sum that wraps is out, and a
    /// rank that differs is out before any edge is looked at.
    #[test]
    fn check_hyperslab_by_boundary() {
        let dims = [4u64, 6];
        assert_eq!(check_hyperslab(&dims, &[1, 2], &[3, 4]), Ok(()));
        assert_eq!(check_hyperslab(&dims, &[4, 0], &[0, 6]), Ok(()));
        assert_eq!(
            check_hyperslab(&dims, &[1, 2], &[3, 5]),
            Err(HyperslabError::OutOfBounds {
                dim: 1,
                start: 2,
                count: 5,
                extent: 6
            })
        );
        assert_eq!(
            check_hyperslab(&dims, &[u64::MAX, 0], &[1, 1]),
            Err(HyperslabError::OutOfBounds {
                dim: 0,
                start: u64::MAX,
                count: 1,
                extent: 4
            })
        );
        assert_eq!(
            check_hyperslab(&dims, &[1, 0], &[u64::MAX, 1]),
            Err(HyperslabError::OutOfBounds {
                dim: 0,
                start: 1,
                count: u64::MAX,
                extent: 4
            })
        );
        assert_eq!(check_hyperslab(&[], &[], &[]), Ok(()));
        assert_eq!(
            check_hyperslab(&dims, &[0], &[1, 1]),
            Err(HyperslabError::Rank { got: 1, rank: 2 })
        );
        assert_eq!(
            check_hyperslab(&dims, &[0, 0, 0], &[1, 1, 1]),
            Err(HyperslabError::Rank { got: 3, rank: 2 })
        );
        assert_eq!(
            check_hyperslab(&dims, &[0, 0], &[1]),
            Err(HyperslabError::Rank { got: 1, rank: 2 })
        );
        let text = HyperslabError::OutOfBounds {
            dim: 0,
            start: 3,
            count: 2,
            extent: 4,
        }
        .to_string();
        assert!(text.contains("out of bounds"), "{text}");
    }

    #[test]
    fn resolve_rejects_a_box_past_the_extent() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Blocks(vec![HyperslabBlock {
                start: vec![2],
                end: vec![5],
            }]),
        };
        let err = sel.resolve(&[4]).unwrap_err();
        assert!(
            format!("{err}")
                .contains("out of bounds in dimension 0: start 2 + count 4 exceeds extent 4"),
            "unexpected error: {err}"
        );
    }

    /// A scalar dataspace holds one element and takes only all/none.
    #[test]
    fn resolve_of_a_scalar_extent_holds_one_element() {
        let all = Selection::All.resolve(&[]).unwrap();
        assert_eq!(all.n_elements(), 1);
        assert_eq!(all.runs[0].len, 1);
        assert_eq!(Selection::None.resolve(&[]).unwrap().n_elements(), 0);
    }

    #[test]
    fn to_boxes_rejects_rank_mismatch() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Blocks(vec![]),
        };
        let err = sel.to_boxes(&[4]).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn to_boxes_rejects_inverted_block() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Blocks(vec![HyperslabBlock {
                start: vec![5],
                end: vec![2],
            }]),
        };
        let err = sel.to_boxes(&[10]).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    /// A regular hyperslab expands into one box per `count` position, block
    /// sized `block`, spaced `stride` apart — the odometer must walk the
    /// last dimension fastest, matching row-major box order.
    #[test]
    fn to_boxes_regular_expands_2d_grid() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0, 0],
                stride: vec![4, 4],
                count: vec![2, 2],
                block: vec![2, 2],
            }),
        };
        let boxes = sel.to_boxes(&[8, 8]).unwrap();
        assert_eq!(
            boxes,
            vec![
                (vec![0, 0], vec![2, 2]),
                (vec![0, 4], vec![2, 2]),
                (vec![4, 0], vec![2, 2]),
                (vec![4, 4], vec![2, 2]),
            ]
        );
    }

    #[test]
    fn to_boxes_regular_single_block_matches_block_list_shape() {
        // A count=1 regular hyperslab is the same box a block-list form of
        // the same selection would produce.
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![4],
                stride: vec![1],
                count: vec![1],
                block: vec![8],
            }),
        };
        assert_eq!(sel.to_boxes(&[20]).unwrap(), vec![(vec![4], vec![8])]);
    }

    #[test]
    fn to_boxes_regular_zero_count_is_empty() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![0],
                block: vec![1],
            }),
        };
        assert_eq!(sel.to_boxes(&[10]).unwrap(), vec![]);
    }

    #[test]
    fn to_boxes_regular_rejects_unlimited_count() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![UNLIMITED],
                block: vec![1],
            }),
        };
        let err = sel.to_boxes(&[10]).unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
    }

    #[test]
    fn to_boxes_regular_rejects_unlimited_block() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![1],
                block: vec![UNLIMITED],
            }),
        };
        let err = sel.to_boxes(&[10]).unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
    }

    /// A crafted `count` product over the box cap must fail cleanly instead
    /// of attempting a huge allocation.
    #[test]
    fn to_boxes_regular_rejects_huge_box_count() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0, 0],
                stride: vec![1, 1],
                count: vec![1 << 30, 1 << 30],
                block: vec![1, 1],
            }),
        };
        let err = sel.to_boxes(&[u64::MAX, u64::MAX]).unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
    }

    // ------------------------------------------------------------------ encode

    /// Every fixture in `tests/fixtures/gen_selection_images.c`, matched
    /// byte-for-byte against `Selection::encode()` of the equivalent
    /// value — not just decode(encode(x)) == x, but the exact bytes
    /// libhdf5 1.14.6 itself wrote.
    ///
    /// A [`Hyperslab::Regular`] case's `decoded` value is its own
    /// Blocks-normalized equivalent, not the original `Regular` selection:
    /// regularity does not survive a real version-1 round trip either —
    /// libhdf5's own decode never reconstructs a REGULAR pattern from a
    /// block list, since the version-1 wire form has no flags byte to
    /// carry that information at all.
    #[test]
    fn selection_matches_libhdf5_image() {
        let cases: Vec<(&[u8], Selection, Selection)> = vec![
            (
                include_bytes!("../../tests/fixtures/all_v1.bin"),
                Selection::All,
                Selection::All,
            ),
            (
                include_bytes!("../../tests/fixtures/none_v1.bin"),
                Selection::None,
                Selection::None,
            ),
            (
                include_bytes!("../../tests/fixtures/hyperslab_single_block_v1.bin"),
                Selection::Hyperslab {
                    rank: 1,
                    form: Hyperslab::Blocks(vec![HyperslabBlock {
                        start: vec![4],
                        end: vec![11],
                    }]),
                },
                Selection::Hyperslab {
                    rank: 1,
                    form: Hyperslab::Blocks(vec![HyperslabBlock {
                        start: vec![4],
                        end: vec![11],
                    }]),
                },
            ),
            (
                include_bytes!("../../tests/fixtures/hyperslab_regular_3blocks_v1.bin"),
                // start=0, stride=5, count=3, block=2 — the exact Regular
                // form the C generator built via H5Sselect_hyperslab; the
                // captured image is nonetheless the version-1 block list
                // (see Selection::encode's doc comment), so encoding this
                // Regular value must reproduce those blocks byte-for-byte.
                Selection::Hyperslab {
                    rank: 1,
                    form: Hyperslab::Regular(RegularHyperslab {
                        start: vec![0],
                        stride: vec![5],
                        count: vec![3],
                        block: vec![2],
                    }),
                },
                Selection::Hyperslab {
                    rank: 1,
                    form: Hyperslab::Blocks(vec![
                        HyperslabBlock {
                            start: vec![0],
                            end: vec![1],
                        },
                        HyperslabBlock {
                            start: vec![5],
                            end: vec![6],
                        },
                        HyperslabBlock {
                            start: vec![10],
                            end: vec![11],
                        },
                    ]),
                },
            ),
            (
                include_bytes!("../../tests/fixtures/hyperslab_2d_regular_v1.bin"),
                Selection::Hyperslab {
                    rank: 2,
                    form: Hyperslab::Regular(RegularHyperslab {
                        start: vec![0, 0],
                        stride: vec![4, 4],
                        count: vec![2, 2],
                        block: vec![2, 2],
                    }),
                },
                Selection::Hyperslab {
                    rank: 2,
                    form: Hyperslab::Blocks(vec![
                        HyperslabBlock {
                            start: vec![0, 0],
                            end: vec![1, 1],
                        },
                        HyperslabBlock {
                            start: vec![0, 4],
                            end: vec![1, 5],
                        },
                        HyperslabBlock {
                            start: vec![4, 0],
                            end: vec![5, 1],
                        },
                        HyperslabBlock {
                            start: vec![4, 4],
                            end: vec![5, 5],
                        },
                    ]),
                },
            ),
            (
                include_bytes!("../../tests/fixtures/points4_v1.bin"),
                Selection::Points(PointSelection {
                    rank: 1,
                    points: vec![vec![1], vec![3], vec![7], vec![15]],
                }),
                Selection::Points(PointSelection {
                    rank: 1,
                    points: vec![vec![1], vec![3], vec![7], vec![15]],
                }),
            ),
        ];

        for (blob, sel, want_decoded) in cases {
            let expected = strip_h5sencode_envelope(blob);
            let encoded = sel.encode().unwrap();
            assert_eq!(
                encoded, expected,
                "encode() mismatch for {sel:?}: got {encoded:02x?}, want {expected:02x?}"
            );

            // And the image itself decodes back to the expected value, so
            // decode/encode agree on both ends of the wire, not just this
            // module's own encode output.
            let (decoded, consumed) = Selection::decode(expected).unwrap();
            assert_eq!(consumed, expected.len());
            assert_eq!(decoded, want_decoded);
        }
    }

    /// decode(encode(x)) == x for [`Selection`] variants whose wire form
    /// is lossless (everything but [`Hyperslab::Regular`], which the
    /// version-1 wire form always normalizes into a block list — see
    /// `selection_matches_libhdf5_image`).
    #[test]
    fn encode_decode_round_trips() {
        let values = vec![
            Selection::None,
            Selection::All,
            Selection::Points(PointSelection {
                rank: 2,
                points: vec![vec![0, 0], vec![3, 5], vec![9, 1]],
            }),
            Selection::Hyperslab {
                rank: 2,
                form: Hyperslab::Blocks(vec![
                    HyperslabBlock {
                        start: vec![0, 0],
                        end: vec![1, 1],
                    },
                    HyperslabBlock {
                        start: vec![4, 4],
                        end: vec![5, 6],
                    },
                ]),
            },
        ];
        for sel in values {
            let encoded = sel.encode().unwrap();
            let (decoded, consumed) = Selection::decode(&encoded).unwrap();
            assert_eq!(consumed, encoded.len());
            assert_eq!(decoded, sel);
        }
    }

    /// A [`Hyperslab::Regular`] value's round trip lands on its
    /// Blocks-normalized equivalent, not the original value — this is the
    /// same lossy-regularity behavior real libhdf5 has under the version-1
    /// wire form (no flags byte to carry a REGULAR marker at all).
    #[test]
    fn encode_decode_round_trips_regular_hyperslab_to_its_block_list() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![1, 2],
                stride: vec![3, 3],
                count: vec![2, 3],
                block: vec![1, 2],
            }),
        };
        let encoded = sel.encode().unwrap();
        let (decoded, consumed) = Selection::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        let Selection::Hyperslab {
            form: Hyperslab::Blocks(blocks),
            ..
        } = decoded
        else {
            panic!("expected a decoded block list");
        };
        // 2 * 3 = 6 blocks, one per (start[d] + idx[d]*stride[d]) grid
        // point, each block[0]=1 x block[1]=2 wide.
        assert_eq!(blocks.len(), 6);
        assert_eq!(blocks[0].start, vec![1, 2]);
        assert_eq!(blocks[0].end, vec![1, 3]);
        assert_eq!(blocks[5].start, vec![4, 8]);
        assert_eq!(blocks[5].end, vec![4, 9]);
    }

    #[test]
    fn encode_points_rejects_zero_rank() {
        let sel = Selection::Points(PointSelection {
            rank: 0,
            points: vec![],
        });
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn encode_points_rejects_coordinate_length_mismatch() {
        let sel = Selection::Points(PointSelection {
            rank: 2,
            points: vec![vec![1, 2, 3]],
        });
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn encode_points_rejects_coordinate_over_u32() {
        let sel = Selection::Points(PointSelection {
            rank: 1,
            points: vec![vec![1u64 << 40]],
        });
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
    }

    #[test]
    fn encode_hyperslab_rejects_zero_rank() {
        let sel = Selection::Hyperslab {
            rank: 0,
            form: Hyperslab::Blocks(vec![]),
        };
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn encode_hyperslab_rejects_block_length_mismatch() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Blocks(vec![HyperslabBlock {
                start: vec![0],
                end: vec![1],
            }]),
        };
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn encode_hyperslab_rejects_regular_field_length_mismatch() {
        let sel = Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![1],
                block: vec![1],
            }),
        };
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn encode_hyperslab_rejects_block_coordinate_over_u32() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Blocks(vec![HyperslabBlock {
                start: vec![0],
                end: vec![1u64 << 40],
            }]),
        };
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
    }

    /// The libhdf5 image of an unlimited hyperslab, byte for byte.
    ///
    /// Captured from the virtual-dataset mapping list h5py 3.15.1 writes for
    /// `layout[:h5py.h5s.UNLIMITED, :] = vsrc[:h5py.h5s.UNLIMITED, :]` over a
    /// rank-2 `(1, 2)` layout — the two selections are identical, so this is
    /// the image of both. An unlimited dimension forces version 2
    /// (`H5S__hyper_get_version_enc_size`), whose REGULAR form spells the
    /// unlimited count as an all-ones 8-byte field.
    const LIBHDF5_UNLIMITED_HYPERSLAB: &[u8] = &[
        0x02, 0x00, 0x00, 0x00, // H5S_SEL_HYPERSLABS
        0x02, 0x00, 0x00, 0x00, // version 2
        0x01, // flags: H5S_HYPER_REGULAR
        0x44, 0x00, 0x00, 0x00, // length: 4 + 32 * rank
        0x02, 0x00, 0x00, 0x00, // rank
        // dim 0: start 0, stride 1, count H5S_UNLIMITED, block 1
        0, 0, 0, 0, 0, 0, 0, 0, //
        1, 0, 0, 0, 0, 0, 0, 0, //
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, //
        1, 0, 0, 0, 0, 0, 0, 0, //
        // dim 1: start 0, stride 1, count 1, block 2
        0, 0, 0, 0, 0, 0, 0, 0, //
        1, 0, 0, 0, 0, 0, 0, 0, //
        1, 0, 0, 0, 0, 0, 0, 0, //
        2, 0, 0, 0, 0, 0, 0, 0, //
    ];

    fn unlimited_rows() -> Selection {
        Selection::Hyperslab {
            rank: 2,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0, 0],
                stride: vec![1, 1],
                count: vec![UNLIMITED, 1],
                block: vec![1, 2],
            }),
        }
    }

    /// An unlimited count encodes to exactly what libhdf5 writes: the
    /// version-2 REGULAR form, not the version-1 block list every bounded
    /// hyperslab takes.
    #[test]
    fn unlimited_hyperslab_encodes_as_the_libhdf5_version_2_image() {
        assert_eq!(
            unlimited_rows().encode().unwrap(),
            LIBHDF5_UNLIMITED_HYPERSLAB
        );
    }

    /// And the decoder reads that image back as the same selection, so a
    /// mapping list survives a write/read round trip unchanged.
    #[test]
    fn unlimited_hyperslab_decodes_from_the_libhdf5_image() {
        let (sel, used) = Selection::decode(LIBHDF5_UNLIMITED_HYPERSLAB).unwrap();
        assert_eq!(used, LIBHDF5_UNLIMITED_HYPERSLAB.len());
        assert_eq!(sel, unlimited_rows());
        assert_eq!(sel.unlim_dim(), Some(0));
    }

    /// An unlimited selection has no bounds — `H5Sget_select_bounds` fails on
    /// one, which is why the oracle's canon renders it as `?`.
    #[test]
    fn an_unlimited_selection_has_no_bounds() {
        assert_eq!(unlimited_rows().bounds(), None);
    }

    /// `H5S_hyper_clip_unlim`: the unlimited dimension is cut to the extent,
    /// and the result covers exactly the elements inside it.
    #[test]
    fn clip_unlimited_cuts_the_unlimited_dimension_to_the_extent() {
        let clipped = unlimited_rows().clip_unlimited(3).unwrap();
        assert_eq!(
            // `H5S__hyper_get_clip_diminfo` collapses a contiguous run
            // (block == stride) into one block of the whole extent, so the
            // three unit rows come back as one 3-row box.
            clipped.to_boxes(&[3, 2]).unwrap(),
            vec![(vec![0, 0], vec![3, 2])]
        );
        // Nothing available: the mapping selects nothing at all, rather than
        // an empty block list.
        assert_eq!(unlimited_rows().clip_unlimited(0).unwrap(), Selection::None);
        // A bounded selection is its own clip.
        assert_eq!(Selection::All.clip_unlimited(7).unwrap(), Selection::All);
    }

    /// A stride wider than the block leaves a gap, and the extent can cut
    /// through the middle of a block — `H5S__hyper_get_clip_diminfo`'s
    /// unlimited-count arm, where the last block comes back short.
    #[test]
    fn clip_unlimited_truncates_the_block_the_extent_cuts_through() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![1],
                stride: vec![4],
                count: vec![UNLIMITED],
                block: vec![3],
            }),
        };
        // Blocks at [1,4) and [5,8); an extent of 7 cuts the second short.
        assert_eq!(
            sel.clip_unlimited(7).unwrap().to_boxes(&[7]).unwrap(),
            vec![(vec![1], vec![3]), (vec![5], vec![2])]
        );
    }

    /// `H5S_hyper_get_clip_extent_match`, both halves: how many slices a
    /// source extent supplies, and the virtual extent that covers exactly
    /// that many.
    #[test]
    fn clip_extent_matches_the_slices_the_source_supplies() {
        let Selection::Hyperslab {
            form: Hyperslab::Regular(rows),
            ..
        } = unlimited_rows()
        else {
            unreachable!()
        };
        // Contiguous unit blocks: slices and extent are the source extent.
        for extent in [0u64, 1, 6, 10] {
            assert_eq!(rows.num_slices(extent), extent);
            assert_eq!(rows.clip_extent(extent, false), extent);
        }

        // Strided blocks: 3 elements every 4, so an extent of 10 supplies
        // 3 + 3 + 2 = 8 slices, and covering 8 slices needs an extent of 10.
        let strided = RegularHyperslab {
            start: vec![0],
            stride: vec![4],
            count: vec![UNLIMITED],
            block: vec![3],
        };
        assert_eq!(strided.num_slices(10), 8);
        assert_eq!(strided.clip_extent(8, false), 10);
        // Two whole blocks: `incl_trail` is the difference between ending at
        // the last block's end and ending before the first missing block.
        assert_eq!(strided.clip_extent(6, false), 7);
        assert_eq!(strided.clip_extent(6, true), 8);
        assert_eq!(strided.clip_extent(0, false), 0);
    }

    /// `H5S_get_select_num_elem_non_unlim` and `H5S_hyper_get_unlim_block`:
    /// the per-slice element count two unlimited selections must agree on,
    /// and the single block a printf mapping's `index`-th source fills.
    #[test]
    fn unlimited_slice_shape_and_block_extraction() {
        let Selection::Hyperslab {
            form: Hyperslab::Regular(rows),
            ..
        } = unlimited_rows()
        else {
            unreachable!()
        };
        assert_eq!(rows.num_elem_non_unlim(), Some(2));
        let third = rows.unlim_block(3);
        assert_eq!(third.start, vec![3, 0]);
        assert_eq!(third.count, vec![1, 1]);
        assert_eq!(third.block, vec![1, 2]);
        assert_eq!(third.unlim_dim(), None);
    }
}
