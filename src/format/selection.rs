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

fn encode_hyperslab(rank: usize, form: &Hyperslab) -> FormatResult<Vec<u8>> {
    if rank == 0 || rank > MAX_RANK {
        return Err(FormatError::InvalidData(format!(
            "invalid hyperslab selection rank {rank}"
        )));
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

    /// A Regular form's UNLIMITED count cannot become a finite block list
    /// (the same reason [`Selection::to_boxes`] rejects it) — encode must
    /// surface that as a clean error, not silently drop the dimension.
    #[test]
    fn encode_hyperslab_rejects_regular_unlimited_count() {
        let sel = Selection::Hyperslab {
            rank: 1,
            form: Hyperslab::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![UNLIMITED],
                block: vec![1],
            }),
        };
        let err = sel.encode().unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
    }
}
