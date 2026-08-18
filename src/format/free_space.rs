//! Persisted free-space managers: the header block (`FSHD`) and the serialized
//! sections that hang off it (`FSSE`).
//!
//! A file created with `H5Pset_file_space_strategy(..., persist = true)` keeps
//! the space its own edits released across close, in one manager per allocation
//! type. The file-space info message in the superblock extension names the
//! managers by address; each manager is a fixed-size header naming a variable
//! block of sections, and each section is an `(address, length)` pair with a
//! class byte. Nothing else in the file records that space, so a library that
//! reopens the file without reading these blocks has no way to tell free space
//! from live.
//!
//! Upstream: `H5FScache.c` (`H5FS__cache_hdr_deserialize`,
//! `H5FS__cache_hdr_serialize`, `H5FS__cache_sinfo_deserialize`,
//! `H5FS__cache_sinfo_serialize`), `H5FSsection.c` (`H5FS__sinfo_new`,
//! `H5FS__sect_serialize_size`, `H5FS__sect_merge`), `H5MFsection.c` for the
//! section classes the file client uses.

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::checksum::checksum_metadata;
use crate::format::{FormatContext, FormatError, FormatResult};

/// Free-space header signature (`H5FS_HDR_MAGIC`).
pub const FSHD_SIGNATURE: [u8; 4] = *b"FSHD";
/// Serialized-sections signature (`H5FS_SINFO_MAGIC`).
pub const FSSE_SIGNATURE: [u8; 4] = *b"FSSE";

/// The only version either block has (`H5FS_HDR_VERSION`, `H5FS_SINFO_VERSION`).
const FS_VERSION: u8 = 0;

/// `H5FS_CLIENT_FILE_ID` — the client every manager the file itself owns
/// declares. The other value, 0, belongs to a fractal heap's own manager.
pub const CLIENT_FILE: u8 = 1;

/// `H5MF_FSPACE_SECT_SIMPLE`: the section class a non-paged file uses for
/// every section. The paged strategy's `SMALL` and `LARGE` classes are 1 and 2.
pub const SECT_CLASS_SIMPLE: u8 = 0;

/// Section classes `H5MF__create_fstype` registers, and so the count every
/// manager the file client owns declares.
pub const FILE_SECT_CLASSES: u16 = 3;

/// `H5MF_FSPACE_SHRINK` / `H5MF_FSPACE_EXPAND`, the thresholds at which the
/// serialized-section block is reallocated. This crate lays the block out
/// afresh on every close, so they are recorded rather than acted on.
pub const SHRINK_PERCENT: u16 = 80;
/// See [`SHRINK_PERCENT`].
pub const EXPAND_PERCENT: u16 = 120;

/// `H5FD_sec2`'s `MAXADDR`: the largest address the driver this crate
/// implements can address, and so the `max_sect_size` every manager in one of
/// its files declares.
pub const SEC2_MAXADDR: u64 = (1 << 63) - 1;

/// `1 + H5VM_log2_gen(maxaddr)` for [`SEC2_MAXADDR`] — the width of the
/// address space sections are encoded in.
pub const SEC2_MAX_SECT_ADDR: u16 = 63;

/// `H5VM_log2_gen`: the position of the highest set bit, and 0 for 0.
fn log2_gen(n: u64) -> u32 {
    63u32.saturating_sub(n.leading_zeros().min(63))
}

/// `H5VM_limit_enc_size`: bytes needed to encode any value up to `limit`.
fn limit_enc_size(limit: u64) -> usize {
    (log2_gen(limit) / 8) as usize + 1
}

/// Read `n` little-endian bytes as a `u64` (`UINT64DECODE_VAR`).
fn read_var(buf: &[u8], n: usize) -> u64 {
    read_uint(buf, n)
}

/// Append `n` little-endian bytes of `v` (`UINT64ENCODE_VAR`).
fn write_var(buf: &mut Vec<u8>, v: u64, n: usize) {
    buf.extend_from_slice(&v.to_le_bytes()[..n]);
}

/// One free region of the file, as a manager records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FreeSection {
    /// File address of the first free byte.
    pub addr: u64,
    /// How many bytes are free there.
    pub len: u64,
    /// Section class — [`SECT_CLASS_SIMPLE`] for a non-paged file.
    pub class: u8,
}

/// A free-space manager header (`FSHD`, `H5FS_t`'s serialized fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeSpaceHeader {
    /// Who owns the manager — [`CLIENT_FILE`] for the ones this crate writes.
    pub client: u8,
    /// Bytes covered by the sections, which is what `h5stat` reports as
    /// tracked free space.
    pub total_space: u64,
    /// Sections tracked, serializable and ghost together.
    pub total_sections: u64,
    /// Of those, the ones written to the sections block.
    pub serial_sections: u64,
    /// Of those, the ones that are not (the file client has none).
    pub ghost_sections: u64,
    /// Section classes the manager was created with.
    pub nclasses: u16,
    /// See [`SHRINK_PERCENT`].
    pub shrink_percent: u16,
    /// See [`EXPAND_PERCENT`].
    pub expand_percent: u16,
    /// Width, in bits, of the address space section addresses are encoded in.
    pub max_sect_addr: u16,
    /// Largest section the manager will track.
    pub max_sect_size: u64,
    /// Address of the sections block, or [`UNDEF_ADDR`] when there is none.
    ///
    /// [`UNDEF_ADDR`]: crate::format::UNDEF_ADDR
    pub sect_addr: u64,
    /// Bytes of that block the sections image uses.
    pub sect_size: u64,
    /// Bytes the block occupies. Never below `sect_size`; the difference is
    /// slack the manager keeps for the next section it gains.
    pub alloc_sect_size: u64,
}

impl FreeSpaceHeader {
    /// On-disk size of the header (`H5FS_HEADER_SIZE`).
    pub fn encoded_size(ctx: &FormatContext) -> usize {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        // Signature, version, checksum; then client, four counts, four 16-bit
        // fields, the max section size, and the sections block's address,
        // used size and allocated size.
        4 + 1 + 4 + 1 + 4 * ss + 4 * 2 + ss + sa + ss + ss
    }

    /// Decode the header (`H5FS__cache_hdr_deserialize`).
    ///
    /// The checksum covers everything before it, exactly as the block was
    /// written; a mismatch means the block is not a free-space header this
    /// file's manager wrote, which is a corrupt file rather than a version
    /// this crate cannot read.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let size = Self::encoded_size(ctx);
        need(buf, size)?;
        if buf[..4] != FSHD_SIGNATURE {
            return Err(FormatError::InvalidData(
                "free-space header does not start with FSHD".into(),
            ));
        }
        if buf[4] != FS_VERSION {
            return Err(FormatError::InvalidVersion(buf[4]));
        }
        verify_checksum(&buf[..size], "free-space header")?;

        let mut pos = 5;
        let client = buf[pos];
        pos += 1;
        let take_size = |pos: &mut usize| {
            let v = read_uint(&buf[*pos..], ss);
            *pos += ss;
            v
        };
        let total_space = take_size(&mut pos);
        let total_sections = take_size(&mut pos);
        let serial_sections = take_size(&mut pos);
        let ghost_sections = take_size(&mut pos);
        let take_u16 = |pos: &mut usize| {
            let v = u16::from_le_bytes([buf[*pos], buf[*pos + 1]]);
            *pos += 2;
            v
        };
        let nclasses = take_u16(&mut pos);
        let shrink_percent = take_u16(&mut pos);
        let expand_percent = take_u16(&mut pos);
        let max_sect_addr = take_u16(&mut pos);
        let max_sect_size = read_uint(&buf[pos..], ss);
        pos += ss;
        let sect_addr = read_uint(&buf[pos..], sa);
        pos += sa;
        let sect_size = read_uint(&buf[pos..], ss);
        pos += ss;
        let alloc_sect_size = read_uint(&buf[pos..], ss);

        Ok(Self {
            client,
            total_space,
            total_sections,
            serial_sections,
            ghost_sections,
            nclasses,
            shrink_percent,
            expand_percent,
            max_sect_addr,
            max_sect_size,
            sect_addr,
            sect_size,
            alloc_sect_size,
        })
    }

    /// Encode the header (`H5FS__cache_hdr_serialize`).
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let mut buf = Vec::with_capacity(Self::encoded_size(ctx));
        buf.extend_from_slice(&FSHD_SIGNATURE);
        buf.push(FS_VERSION);
        buf.push(self.client);
        for v in [
            self.total_space,
            self.total_sections,
            self.serial_sections,
            self.ghost_sections,
        ] {
            buf.extend_from_slice(&v.to_le_bytes()[..ss]);
        }
        for v in [
            self.nclasses,
            self.shrink_percent,
            self.expand_percent,
            self.max_sect_addr,
        ] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&self.max_sect_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.sect_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&self.sect_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.alloc_sect_size.to_le_bytes()[..ss]);
        let sum = checksum_metadata(&buf);
        buf.extend_from_slice(&sum.to_le_bytes());
        buf
    }

    /// Width of an encoded section address, in bytes (`sinfo->sect_off_size`).
    fn sect_off_size(&self) -> usize {
        (self.max_sect_addr as usize).div_ceil(8)
    }

    /// Width of an encoded section length, in bytes (`sinfo->sect_len_size`).
    fn sect_len_size(&self) -> usize {
        limit_enc_size(self.max_sect_size)
    }

    /// Width of an encoded per-size section count (`udata.sect_cnt_size`).
    fn sect_cnt_size(&self) -> usize {
        limit_enc_size(self.serial_sections)
    }
}

/// Sections in the order a manager serializes them: by size, then by address.
///
/// `H5FS__cache_sinfo_serialize` walks the size bins in ascending order, each
/// bin's size nodes in ascending size, and each node's sections in ascending
/// address. A bin is `floor(log2(size))`, which is monotone in size, so the
/// bins drop out of a plain sort by `(len, addr)`.
pub fn serialization_order(sections: &[FreeSection]) -> Vec<FreeSection> {
    let mut ordered = sections.to_vec();
    ordered.sort_unstable_by_key(|s| (s.len, s.addr));
    ordered
}

/// Bytes the sections image occupies (`H5FS__sect_serialize_size`).
///
/// `sections` must already be in [`serialization_order`]: the count of
/// distinct sizes is taken by walking it.
pub fn sinfo_encoded_size(
    hdr: &FreeSpaceHeader,
    sections: &[FreeSection],
    ctx: &FormatContext,
) -> u64 {
    let prefix = 4 + 1 + ctx.sizeof_addr as u64 + 4;
    if sections.is_empty() {
        return prefix;
    }
    let distinct = distinct_sizes(sections) as u64;
    let n = sections.len() as u64;
    prefix
        + distinct * (hdr.sect_cnt_size() + hdr.sect_len_size()) as u64
        + n * (hdr.sect_off_size() as u64 + 1)
}

/// Distinct section sizes in a list already in [`serialization_order`].
fn distinct_sizes(sections: &[FreeSection]) -> usize {
    let mut count = 0;
    let mut last = None;
    for s in sections {
        if last != Some(s.len) {
            count += 1;
            last = Some(s.len);
        }
    }
    count
}

/// Encode the sections block (`H5FS__cache_sinfo_serialize`).
///
/// `hdr` supplies the field widths and the header address the block names;
/// `image_len` is the block's allocated size, which may exceed the bytes the
/// sections need — upstream leaves the gap before the checksum zeroed and
/// records the used length in the header's `sect_size`.
pub fn encode_sections(
    hdr: &FreeSpaceHeader,
    hdr_addr: u64,
    sections: &[FreeSection],
    image_len: usize,
    ctx: &FormatContext,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(image_len);
    buf.extend_from_slice(&FSSE_SIGNATURE);
    buf.push(FS_VERSION);
    buf.extend_from_slice(&hdr_addr.to_le_bytes()[..ctx.sizeof_addr as usize]);

    let cnt_size = hdr.sect_cnt_size();
    let len_size = hdr.sect_len_size();
    let off_size = hdr.sect_off_size();
    let mut i = 0;
    while i < sections.len() {
        let size = sections[i].len;
        let run = sections[i..].iter().take_while(|s| s.len == size).count();
        write_var(&mut buf, run as u64, cnt_size);
        write_var(&mut buf, size, len_size);
        for s in &sections[i..i + run] {
            write_var(&mut buf, s.addr, off_size);
            buf.push(s.class);
        }
        i += run;
    }

    buf.resize(image_len - 4, 0);
    let sum = checksum_metadata(&buf);
    buf.extend_from_slice(&sum.to_le_bytes());
    buf
}

/// Decode the sections block (`H5FS__cache_sinfo_deserialize`).
///
/// `buf` must be exactly the header's `sect_size` bytes: the checksum lives at
/// its end, and the section records may stop short of it.
pub fn decode_sections(
    buf: &[u8],
    hdr: &FreeSpaceHeader,
    hdr_addr: u64,
    ctx: &FormatContext,
) -> FormatResult<Vec<FreeSection>> {
    let sa = ctx.sizeof_addr as usize;
    need(buf, 4 + 1 + sa + 4)?;
    if buf[..4] != FSSE_SIGNATURE {
        return Err(FormatError::InvalidData(
            "free-space sections block does not start with FSSE".into(),
        ));
    }
    if buf[4] != FS_VERSION {
        return Err(FormatError::InvalidVersion(buf[4]));
    }
    verify_checksum(buf, "free-space sections block")?;
    let named = read_uint(&buf[5..], sa);
    if named != hdr_addr {
        return Err(FormatError::InvalidData(format!(
            "free-space sections at name header {named:#x}, not {hdr_addr:#x}"
        )));
    }

    let mut sections = Vec::with_capacity(hdr.serial_sections as usize);
    if hdr.serial_sections == 0 {
        return Ok(sections);
    }
    let cnt_size = hdr.sect_cnt_size();
    let len_size = hdr.sect_len_size();
    let off_size = hdr.sect_off_size();
    let end = buf.len() - 4;
    let mut pos = 5 + sa;
    while sections.len() < hdr.serial_sections as usize {
        need(buf, pos + cnt_size + len_size)?;
        let run = read_var(&buf[pos..], cnt_size);
        pos += cnt_size;
        let size = read_var(&buf[pos..], len_size);
        pos += len_size;
        if run == 0 || size == 0 {
            return Err(FormatError::InvalidData(
                "free-space sections block declares an empty size node".into(),
            ));
        }
        for _ in 0..run {
            if pos + off_size + 1 > end {
                return Err(FormatError::InvalidData(
                    "free-space sections block ends inside a section record".into(),
                ));
            }
            let addr = read_var(&buf[pos..], off_size);
            pos += off_size;
            let class = buf[pos];
            pos += 1;
            if class >= hdr.nclasses as u8 && hdr.nclasses != 0 {
                return Err(FormatError::InvalidData(format!(
                    "free-space section class {class} is beyond the {} the manager declares",
                    hdr.nclasses
                )));
            }
            sections.push(FreeSection {
                addr,
                len: size,
                class,
            });
        }
    }
    Ok(sections)
}

/// Merge `blocks` into one address-ordered section list with adjacent regions
/// coalesced — `H5FS__sect_merge`'s effect on a set of `(addr, len)` pairs.
///
/// Overlap is a contradiction rather than something to merge over: two callers
/// each believing they own the same bytes is what the caller must not do, so
/// it is reported instead of being papered over.
pub fn merge_sections(blocks: &[(u64, u64)]) -> Result<Vec<FreeSection>, String> {
    let mut sorted: Vec<(u64, u64)> = blocks.iter().copied().filter(|b| b.1 > 0).collect();
    sorted.sort_unstable();
    let mut merged: Vec<FreeSection> = Vec::with_capacity(sorted.len());
    for (addr, len) in sorted {
        match merged.last_mut() {
            Some(last) if last.addr + last.len == addr => last.len += len,
            Some(last) if last.addr + last.len > addr => {
                return Err(format!(
                    "free block {addr:#x}+{len} overlaps {:#x}+{}",
                    last.addr, last.len
                ))
            }
            _ => merged.push(FreeSection {
                addr,
                len,
                class: SECT_CLASS_SIMPLE,
            }),
        }
    }
    Ok(merged)
}

fn verify_checksum(buf: &[u8], what: &str) -> FormatResult<()> {
    let split = buf.len() - 4;
    let stored = u32::from_le_bytes([buf[split], buf[split + 1], buf[split + 2], buf[split + 3]]);
    let computed = checksum_metadata(&buf[..split]);
    if stored != computed {
        return Err(FormatError::InvalidData(format!(
            "{what} checksum {stored:#010x} does not match the computed {computed:#010x}"
        )));
    }
    Ok(())
}

fn need(buf: &[u8], n: usize) -> FormatResult<()> {
    if buf.len() < n {
        Err(FormatError::BufferTooShort {
            needed: n,
            available: buf.len(),
        })
    } else {
        Ok(())
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    /// The metadata manager `libhdf5` 1.14.6 wrote for a `persist:true`,
    /// `FSM_AGGR` file after one dataset was deleted: three sections totalling
    /// 62 bytes, the sections block immediately after the 82-byte header.
    const FIXTURE_HDR: [u8; 82] = [
        0x46, 0x53, 0x48, 0x44, 0x00, 0x01, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x50, 0x00, 0x78, 0x00, 0x3f,
        0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x84, 0x06, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xe2, 0xcf, 0x1b, 0xf4,
    ];

    /// The sections block that header names, at address 1668.
    const FIXTURE_SINFO: [u8; 71] = [
        0x46, 0x53, 0x53, 0x45, 0x00, 0x32, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x0a,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb9, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xcb, 0x06,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5c, 0xf2, 0xb9, 0x34,
    ];

    fn fixture_header() -> FreeSpaceHeader {
        FreeSpaceHeader::decode(&FIXTURE_HDR, &ctx()).unwrap()
    }

    #[test]
    fn header_size_matches_upstream() {
        assert_eq!(FreeSpaceHeader::encoded_size(&ctx()), 82);
    }

    #[test]
    fn decodes_the_libhdf5_header() {
        let h = fixture_header();
        assert_eq!(h.client, CLIENT_FILE);
        assert_eq!(h.total_space, 62);
        assert_eq!(h.total_sections, 3);
        assert_eq!(h.serial_sections, 3);
        assert_eq!(h.ghost_sections, 0);
        assert_eq!(h.nclasses, FILE_SECT_CLASSES);
        assert_eq!(h.shrink_percent, SHRINK_PERCENT);
        assert_eq!(h.expand_percent, EXPAND_PERCENT);
        assert_eq!(h.max_sect_addr, SEC2_MAX_SECT_ADDR);
        assert_eq!(h.max_sect_size, SEC2_MAXADDR);
        assert_eq!(h.sect_addr, 1668);
        assert_eq!(h.sect_size, 71);
        assert_eq!(h.alloc_sect_size, 71);
    }

    #[test]
    fn header_reencodes_byte_for_byte() {
        assert_eq!(fixture_header().encode(&ctx()), FIXTURE_HDR);
    }

    #[test]
    fn decodes_the_libhdf5_sections() {
        let h = fixture_header();
        let s = decode_sections(&FIXTURE_SINFO, &h, 1586, &ctx()).unwrap();
        assert_eq!(
            s,
            vec![
                FreeSection {
                    addr: 2038,
                    len: 10,
                    class: SECT_CLASS_SIMPLE
                },
                FreeSection {
                    addr: 185,
                    len: 15,
                    class: SECT_CLASS_SIMPLE
                },
                FreeSection {
                    addr: 1739,
                    len: 37,
                    class: SECT_CLASS_SIMPLE
                },
            ]
        );
        assert_eq!(s.iter().map(|x| x.len).sum::<u64>(), h.total_space);
    }

    #[test]
    fn sections_reencode_byte_for_byte() {
        let h = fixture_header();
        let s = decode_sections(&FIXTURE_SINFO, &h, 1586, &ctx()).unwrap();
        let ordered = serialization_order(&s);
        assert_eq!(sinfo_encoded_size(&h, &ordered, &ctx()), h.sect_size);
        assert_eq!(
            encode_sections(&h, 1586, &ordered, h.sect_size as usize, &ctx()),
            FIXTURE_SINFO
        );
    }

    #[test]
    fn a_corrupt_header_checksum_is_rejected() {
        let mut buf = FIXTURE_HDR;
        buf[10] ^= 0xff;
        assert!(matches!(
            FreeSpaceHeader::decode(&buf, &ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    #[test]
    fn a_sections_block_naming_the_wrong_header_is_rejected() {
        let h = fixture_header();
        assert!(matches!(
            decode_sections(&FIXTURE_SINFO, &h, 99, &ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    #[test]
    fn empty_managers_serialize_to_the_prefix_alone() {
        let h = fixture_header();
        assert_eq!(sinfo_encoded_size(&h, &[], &ctx()), 17);
    }

    #[test]
    fn merge_coalesces_adjacent_blocks_and_rejects_overlap() {
        assert_eq!(
            merge_sections(&[(64, 16), (16, 16), (32, 32), (200, 8)]).unwrap(),
            vec![
                FreeSection {
                    addr: 16,
                    len: 64,
                    class: SECT_CLASS_SIMPLE
                },
                FreeSection {
                    addr: 200,
                    len: 8,
                    class: SECT_CLASS_SIMPLE
                },
            ]
        );
        assert!(merge_sections(&[(16, 32), (32, 8)]).is_err());
        assert!(merge_sections(&[(16, 0)]).unwrap().is_empty());
    }

    #[test]
    fn encoding_widths_follow_the_header() {
        let h = fixture_header();
        assert_eq!(h.sect_off_size(), 8);
        assert_eq!(h.sect_len_size(), 8);
        assert_eq!(h.sect_cnt_size(), 1);
        assert_eq!(limit_enc_size(0), 1);
        assert_eq!(limit_enc_size(255), 1);
        assert_eq!(limit_enc_size(256), 2);
        assert_eq!(log2_gen(0), 0);
        assert_eq!(log2_gen(1), 0);
        assert_eq!(log2_gen(SEC2_MAXADDR), 62);
    }

    /// An image larger than the sections need keeps the gap zeroed and the
    /// checksum at its end, which is what `sect_size < alloc_sect_size` means.
    #[test]
    fn a_padded_image_checksums_over_the_whole_block() {
        let h = fixture_header();
        let s = vec![FreeSection {
            addr: 4096,
            len: 24,
            class: SECT_CLASS_SIMPLE,
        }];
        let image = encode_sections(&h, 1586, &s, 64, &ctx());
        assert_eq!(image.len(), 64);
        let mut padded = h;
        padded.serial_sections = 1;
        padded.sect_size = 64;
        assert_eq!(decode_sections(&image, &padded, 1586, &ctx()).unwrap(), s);
    }
}
