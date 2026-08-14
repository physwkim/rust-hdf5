//! Reference elements and the dataspace selections region references carry.
//!
//! Two element layouts exist for the pre-1.12 reference kinds, both written by
//! h5py 3.x today (`H5Tref.c`):
//!
//! ```text
//! H5R_OBJECT1          sizeof_addr bytes   the target's object header address
//! H5R_DATASET_REGION1  sizeof_addr + 4     a global-heap id: collection
//!                                          address then a u32 object index
//! ```
//!
//! The heap object a region reference points at is the target's object header
//! address followed by the serialized selection
//! (`H5R__encode_token_region_compat`), whose wire format lives in the `H5S`
//! serializers (`H5S__hyper_serialize`, `H5S__point_serialize`,
//! `H5S__all_serialize`, `H5S__none_serialize`).

use crate::format::bytes::{read_le_addr, read_le_uint};
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// `H5S_sel_type` codes as they appear in a serialized selection.
const SEL_NONE: u32 = 0;
const SEL_POINTS: u32 = 1;
const SEL_HYPERSLABS: u32 = 2;
const SEL_ALL: u32 = 3;

/// `H5S_HYPER_REGULAR`: set in a version-2-or-later hyperslab whose blocks are
/// a regular start/stride/count/block pattern rather than an explicit list.
const HYPER_FLAG_REGULAR: u8 = 0x01;

/// The address a reference element leads with, or `None` when it names
/// nothing.
///
/// Both element layouts start with a file address, and both spell "no target"
/// the same two ways: the all-ones undefined address `H5F_addr_decode`
/// produces, and 0 — the superblock's own address, so never an object header,
/// and what an unwritten (fill-value) element holds. `H5R__decode_heap`
/// rejects both together (`!H5_addr_defined(hobjid.addr) || hobjid.addr == 0`),
/// so this crate applies the one rule to both kinds rather than per element
/// layout.
fn target_address(elem: &[u8], sizeof_addr: usize) -> Option<u64> {
    match read_le_addr(elem, sizeof_addr) {
        0 | UNDEF_ADDR => None,
        addr => Some(addr),
    }
}

/// The address a `H5R_OBJECT1` element names, or `None` for a null reference.
pub fn decode_object_element(elem: &[u8], ctx: &FormatContext) -> FormatResult<Option<u64>> {
    let sa = ctx.sizeof_addr as usize;
    if elem.len() < sa {
        return Err(FormatError::BufferTooShort {
            needed: sa,
            available: elem.len(),
        });
    }
    Ok(target_address(elem, sa))
}

/// The `(collection address, object index)` a `H5R_DATASET_REGION1` element
/// names, or `None` when the element is a null reference.
pub fn decode_region_element(elem: &[u8], ctx: &FormatContext) -> FormatResult<Option<(u64, u32)>> {
    let sa = ctx.sizeof_addr as usize;
    if elem.len() < sa + 4 {
        return Err(FormatError::BufferTooShort {
            needed: sa + 4,
            available: elem.len(),
        });
    }
    let Some(addr) = target_address(elem, sa) else {
        return Ok(None);
    };
    let idx = u32::from_le_bytes([elem[sa], elem[sa + 1], elem[sa + 2], elem[sa + 3]]);
    Ok(Some((addr, idx)))
}

/// Split a region reference's heap object into the target's object header
/// address and the selection over it.
pub fn decode_region_heap_object(
    data: &[u8],
    ctx: &FormatContext,
) -> FormatResult<(u64, RegionSelection)> {
    let sa = ctx.sizeof_addr as usize;
    if data.len() < sa {
        return Err(FormatError::BufferTooShort {
            needed: sa,
            available: data.len(),
        });
    }
    let addr = read_le_addr(data, sa);
    let selection = RegionSelection::decode(&data[sa..])?;
    Ok((addr, selection))
}

/// One reference element, decoded and resolved against the file it came from.
///
/// `path` is the target's absolute path when the file's link structure names
/// it, and `None` when nothing in the traversed structure points at that
/// address — a reference into an untraversed part of the file, or a stale one
/// left by a deletion. The address is reported either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reference {
    /// An element naming no object: the undefined address libhdf5 writes for
    /// an unset object reference, or a zeroed region-reference heap id.
    Null,
    /// `H5R_OBJECT1`: a whole object.
    Object {
        /// Object header address of the target.
        address: u64,
        /// Absolute path of the target.
        path: Option<String>,
    },
    /// `H5R_DATASET_REGION1`: a dataset plus a selection over it.
    Region {
        /// Object header address of the target dataset.
        address: u64,
        /// Absolute path of the target dataset.
        path: Option<String>,
        /// The selection the reference carries.
        selection: RegionSelection,
    },
}

impl Reference {
    /// The target's absolute path, when the file names it.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Null => None,
            Self::Object { path, .. } | Self::Region { path, .. } => path.as_deref(),
        }
    }

    /// The target's object header address, or `None` for a null reference.
    pub fn address(&self) -> Option<u64> {
        match self {
            Self::Null => None,
            Self::Object { address, .. } | Self::Region { address, .. } => Some(*address),
        }
    }

    /// The selection a region reference carries; `None` for the other kinds.
    pub fn selection(&self) -> Option<&RegionSelection> {
        match self {
            Self::Region { selection, .. } => Some(selection),
            _ => None,
        }
    }

    /// The inclusive bounding box of a region reference's selection —
    /// `H5Sget_select_bounds` on the dereferenced region.
    pub fn bounds(&self) -> Option<(Vec<u64>, Vec<u64>)> {
        self.selection()?.bounds()
    }

    /// Whether this element names no object.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

/// A dataspace selection as a region reference stores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionSelection {
    /// Every element of the target's extent. The reference records only the
    /// fact, so the bounds come from the target's shape.
    All,
    /// No elements at all; libhdf5 reports no bounds for one.
    None,
    /// Hyperslab blocks, each an inclusive `(start, end)` coordinate pair.
    Hyperslab(Vec<(Vec<u64>, Vec<u64>)>),
    /// Individual element coordinates.
    Points(Vec<Vec<u64>>),
}

impl RegionSelection {
    /// The inclusive bounding box of the selection — libhdf5's
    /// `H5Sget_select_bounds` — or `None` when the selection has no bounds of
    /// its own ([`RegionSelection::All`] takes the target's extent,
    /// [`RegionSelection::None`] has none at all).
    pub fn bounds(&self) -> Option<(Vec<u64>, Vec<u64>)> {
        match self {
            Self::All | Self::None => None,
            Self::Hyperslab(blocks) => {
                let (first_start, first_end) = blocks.first()?;
                let mut lo = first_start.clone();
                let mut hi = first_end.clone();
                for (start, end) in &blocks[1..] {
                    for (l, s) in lo.iter_mut().zip(start) {
                        *l = (*l).min(*s);
                    }
                    for (h, e) in hi.iter_mut().zip(end) {
                        *h = (*h).max(*e);
                    }
                }
                Some((lo, hi))
            }
            Self::Points(points) => {
                let first = points.first()?;
                let mut lo = first.clone();
                let mut hi = first.clone();
                for p in &points[1..] {
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

    /// Decode one serialized selection.
    ///
    /// The four-word preamble — selection type, version, then either padding
    /// plus a length (version 1) or the version's own flag bytes — is shared
    /// by every selection class; what follows depends on the class.
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        let mut r = Cursor::new(buf);
        let sel_type = r.u32()?;
        let version = r.u32()?;
        match sel_type {
            SEL_ALL => Ok(Self::All),
            SEL_NONE => Ok(Self::None),
            SEL_POINTS => Self::decode_points(&mut r, version),
            SEL_HYPERSLABS => Self::decode_hyperslab(&mut r, version),
            other => Err(FormatError::UnsupportedFeature(format!(
                "dataspace selection type {other}"
            ))),
        }
    }

    fn decode_points(r: &mut Cursor<'_>, version: u32) -> FormatResult<Self> {
        // Version 1 stores 4 bytes of padding and a length; version 2 stores
        // the coordinate width instead (`H5S__point_serialize`).
        let width = match version {
            1 => {
                r.skip(8)?;
                4
            }
            2 => r.u8()? as usize,
            other => {
                return Err(FormatError::UnsupportedFeature(format!(
                    "point selection version {other}"
                )))
            }
        };
        let rank = r.u32()? as usize;
        let count = r.uint(width)? as usize;
        check_extent(rank, count)?;
        let mut points = Vec::with_capacity(count);
        for _ in 0..count {
            let mut coords = Vec::with_capacity(rank);
            for _ in 0..rank {
                coords.push(r.uint(width)?);
            }
            points.push(coords);
        }
        Ok(Self::Points(points))
    }

    fn decode_hyperslab(r: &mut Cursor<'_>, version: u32) -> FormatResult<Self> {
        // Version 1 has no flags and always lists blocks; versions 2 and 3
        // carry a flag saying the blocks are a regular pattern, and version 3
        // also chooses the coordinate width (`H5S__hyper_serialize`).
        let (flags, width) = match version {
            1 => {
                r.skip(8)?; // padding + length
                (0, 4)
            }
            2 => {
                let flags = r.u8()?;
                r.skip(4)?; // length
                (flags, 8)
            }
            3 => {
                let flags = r.u8()?;
                let width = r.u8()? as usize;
                (flags, width)
            }
            other => {
                return Err(FormatError::UnsupportedFeature(format!(
                    "hyperslab selection version {other}"
                )))
            }
        };
        let rank = r.u32()? as usize;

        if flags & HYPER_FLAG_REGULAR != 0 {
            // start/stride/count/block per dimension, expanded into the same
            // block list an irregular selection lists explicitly.
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                let start = r.uint(width)?;
                let stride = r.uint(width)?;
                let count = r.uint(width)?;
                let block = r.uint(width)?;
                dims.push((start, stride, count, block));
            }
            return regular_blocks(&dims);
        }

        let count = r.uint(width)? as usize;
        check_extent(rank, count)?;
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let mut start = Vec::with_capacity(rank);
            for _ in 0..rank {
                start.push(r.uint(width)?);
            }
            let mut end = Vec::with_capacity(rank);
            for _ in 0..rank {
                end.push(r.uint(width)?);
            }
            blocks.push((start, end));
        }
        Ok(Self::Hyperslab(blocks))
    }
}

/// Expand a regular `(start, stride, count, block)` pattern into its blocks,
/// in the same row-major order `H5S__hyper_serialize` lists them.
fn regular_blocks(dims: &[(u64, u64, u64, u64)]) -> FormatResult<RegionSelection> {
    let total = dims
        .iter()
        .try_fold(1usize, |acc, (_, _, count, _)| {
            usize::try_from(*count)
                .ok()
                .and_then(|c| acc.checked_mul(c))
        })
        .ok_or_else(|| {
            FormatError::InvalidData("regular hyperslab block count overflows".into())
        })?;
    check_extent(dims.len(), total)?;

    let mut blocks = Vec::with_capacity(total);
    let mut index = vec![0u64; dims.len()];
    for _ in 0..total {
        let mut start = Vec::with_capacity(dims.len());
        let mut end = Vec::with_capacity(dims.len());
        for (d, (s, stride, _, block)) in dims.iter().enumerate() {
            let origin = s.saturating_add(index[d].saturating_mul(*stride));
            start.push(origin);
            end.push(origin.saturating_add(block.saturating_sub(1)));
        }
        blocks.push((start, end));
        // Odometer over the block grid, fastest dimension last.
        for d in (0..dims.len()).rev() {
            index[d] += 1;
            if index[d] < dims[d].2 {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(RegionSelection::Hyperslab(blocks))
}

/// A selection naming more elements than any file could hold is a corrupt
/// one; refuse it before allocating for it.
fn check_extent(rank: usize, count: usize) -> FormatResult<()> {
    const MAX_SELECTION_ITEMS: usize = 1 << 24;
    if rank > 32 {
        return Err(FormatError::InvalidData(format!(
            "selection rank {rank} exceeds the format maximum of 32"
        )));
    }
    if count > MAX_SELECTION_ITEMS {
        return Err(FormatError::InvalidData(format!(
            "selection lists {count} items, beyond the {MAX_SELECTION_ITEMS} this reader accepts"
        )));
    }
    Ok(())
}

/// Little-endian cursor over a serialized selection.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> FormatResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(FormatError::BufferTooShort {
            needed: usize::MAX,
            available: self.buf.len(),
        })?;
        if end > self.buf.len() {
            return Err(FormatError::BufferTooShort {
                needed: end,
                available: self.buf.len(),
            });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn skip(&mut self, n: usize) -> FormatResult<()> {
        self.take(n).map(|_| ())
    }

    fn u8(&mut self) -> FormatResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> FormatResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// An unsigned little-endian integer of the selection's coordinate width,
    /// which `H5S_SELECT_INFO_ENC_SIZE_*` fixes at 2, 4 or 8 bytes.
    fn uint(&mut self, width: usize) -> FormatResult<u64> {
        if !matches!(width, 2 | 4 | 8) {
            return Err(FormatError::UnsupportedFeature(format!(
                "selection coordinate width {width}"
            )));
        }
        Ok(read_le_uint(self.take(width)?, width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    /// The heap object libhdf5 1.14.6 writes for `dset.regionref[0:3]` on a
    /// 1-D 8-element dataset: the target's object header address followed by a
    /// version-1 hyperslab naming one block, [0]-[2].
    const REGION_HEAP_OBJECT: [u8; 40] = [
        0x20, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // target address 0x320
        0x02, 0x00, 0x00, 0x00, // H5S_SEL_HYPERSLABS
        0x01, 0x00, 0x00, 0x00, // version 1
        0x00, 0x00, 0x00, 0x00, // padding
        0x10, 0x00, 0x00, 0x00, // length 16
        0x01, 0x00, 0x00, 0x00, // rank 1
        0x01, 0x00, 0x00, 0x00, // one block
        0x00, 0x00, 0x00, 0x00, // start [0]
        0x02, 0x00, 0x00, 0x00, // end [2]
    ];

    #[test]
    fn region_heap_object_from_libhdf5_decodes() {
        let (addr, selection) = decode_region_heap_object(&REGION_HEAP_OBJECT, &ctx()).unwrap();
        assert_eq!(addr, 0x320);
        assert_eq!(
            selection,
            RegionSelection::Hyperslab(vec![(vec![0], vec![2])])
        );
        assert_eq!(selection.bounds(), Some((vec![0], vec![2])));
    }

    #[test]
    fn object_and_region_elements_report_null() {
        assert_eq!(
            decode_object_element(&0x320u64.to_le_bytes(), &ctx()).unwrap(),
            Some(0x320)
        );
        assert_eq!(
            decode_object_element(&[0xFF; 8], &ctx()).unwrap(),
            None,
            "an undefined address is a null reference"
        );
        assert_eq!(
            decode_object_element(&[0; 8], &ctx()).unwrap(),
            None,
            "so is address 0, which h5py writes for an unset element"
        );
        let mut elem = [0u8; 12];
        elem[..8].copy_from_slice(&0x820u64.to_le_bytes());
        elem[8..].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode_region_element(&elem, &ctx()).unwrap(),
            Some((0x820, 2))
        );
        assert_eq!(
            decode_region_element(&[0u8; 12], &ctx()).unwrap(),
            None,
            "a zeroed element carries no heap id"
        );
    }

    /// A multi-block selection bounds to the box that covers every block, the
    /// same answer `H5Sget_select_bounds` gives.
    #[test]
    fn bounds_cover_every_block_and_point() {
        let hyper =
            RegionSelection::Hyperslab(vec![(vec![4, 1], vec![5, 3]), (vec![0, 6], vec![1, 7])]);
        assert_eq!(hyper.bounds(), Some((vec![0, 1], vec![5, 7])));
        let points = RegionSelection::Points(vec![vec![3, 9], vec![7, 2]]);
        assert_eq!(points.bounds(), Some((vec![3, 2], vec![7, 9])));
        assert_eq!(RegionSelection::All.bounds(), None);
        assert_eq!(RegionSelection::None.bounds(), None);
    }

    /// A version-3 regular hyperslab lists no blocks; it stores the
    /// start/stride/count/block pattern, which expands to the same blocks.
    #[test]
    fn regular_hyperslab_expands_to_its_blocks() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SEL_HYPERSLABS.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.push(HYPER_FLAG_REGULAR);
        buf.push(4); // coordinate width
        buf.extend_from_slice(&1u32.to_le_bytes()); // rank
        for v in [2u32, 5, 3, 2] {
            // start 2, stride 5, count 3, block 2
            buf.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            RegionSelection::decode(&buf).unwrap(),
            RegionSelection::Hyperslab(vec![
                (vec![2], vec![3]),
                (vec![7], vec![8]),
                (vec![12], vec![13]),
            ])
        );
    }

    #[test]
    fn all_and_none_selections_decode() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SEL_ALL.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(RegionSelection::decode(&buf).unwrap(), RegionSelection::All);
        buf[..4].copy_from_slice(&SEL_NONE.to_le_bytes());
        assert_eq!(
            RegionSelection::decode(&buf).unwrap(),
            RegionSelection::None
        );
    }

    /// A truncated selection is reported, not read past.
    #[test]
    fn a_truncated_selection_is_refused() {
        let err = RegionSelection::decode(&REGION_HEAP_OBJECT[8..20]).unwrap_err();
        assert!(
            matches!(err, FormatError::BufferTooShort { .. }),
            "unexpected error: {err:?}"
        );
    }
}
