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
//! followed by a type-specific body. `H5S_SEL_POINTS` bodies are not
//! decoded — see [`Selection::decode`].
//!
//! All / None body (version is always 1):
//! ```text
//! version:  u32 LE (= 1)
//! reserved: 8 bytes
//! ```
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
            SEL_POINTS => Err(FormatError::UnsupportedFeature(
                "point selection decode (H5S_SEL_POINTS)".into(),
            )),
            other => Err(FormatError::InvalidData(format!(
                "unknown dataspace selection type {other}"
            ))),
        }
    }
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
    fn decode_points_is_unsupported() {
        let buf = [0x01, 0, 0, 0]; // type = SEL_POINTS
        let err = Selection::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::UnsupportedFeature(_)));
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
}
