//! Reading the on-disk free-space managers a file-space info message names.
//!
//! One owner for a walk both halves of the crate need. The writer reopens the
//! managers so a close can rewrite them; the reader reports how much space
//! they record, which is what `H5Fget_freespace` answers and what `h5stat -S`
//! prints as "Amount of tracked free space". The two differ only in policy —
//! the writer refuses the strategies whose allocation rules it does not model
//! — never in how the blocks are found, so the walk itself lives here.

use crate::format::free_space::{self, FreeSpaceClass, FreeSpaceHeader};
use crate::format::messages::superblock_ext::FileSpaceInfoMessage;
use crate::format::{FormatContext, UNDEF_ADDR};
use crate::io::allocator::FreeBlock;
use crate::io::file_handle::FileHandle;
use crate::io::{IoError, IoResult};

/// What one walk of an fsinfo message's managers found.
pub(crate) struct ManagerContents {
    /// The header and section-info blocks the managers themselves occupy.
    /// Space a rewrite supersedes, not space the file has free. Metadata
    /// whichever manager they belong to: `H5FD_MEM_FSPACE_HDR` and
    /// `H5FD_MEM_FSPACE_SINFO` are `H5FD_MEM_OHDR` and `H5FD_MEM_LHEAP`
    /// (H5FDdevelop.h:75-80), both of which the dichotomy sends to
    /// `H5FD_MEM_SUPER`.
    pub(crate) blocks: Vec<(u64, u64)>,
    /// Every section the managers record, each tagged with the manager it came
    /// out of, address-ordered and coalesced within its class.
    pub(crate) sections: Vec<FreeBlock>,
}

/// Walk every manager `info` names and return its sections.
///
/// Sections coalesce only within one manager: two that were adjacent in
/// different managers are two sections upstream as well, because
/// `H5FS__sect_merge` never sees more than the manager it was called for.
///
/// The class comes from the message slot, which under the sec2 driver's
/// dichotomy has exactly two meanings — slot 2 is `H5FD_MEM_DRAW`, everything
/// else is metadata (H5Fsuper.c:831-833). A paged file numbers its slots by
/// `H5F_mem_page_t` instead, so only the *total* is meaningful there; the
/// writer never reaches this with a paged file, `reopen_free_space` having
/// refused it.
pub(crate) fn read_managers(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    info: &FileSpaceInfoMessage,
) -> IoResult<ManagerContents> {
    let hdr_size = FreeSpaceHeader::encoded_size(ctx);
    let mut blocks = Vec::new();
    let mut found: Vec<(FreeSpaceClass, Vec<(u64, u64)>)> = FreeSpaceClass::ALL
        .iter()
        .map(|&class| (class, Vec::new()))
        .collect();
    for (slot, &addr) in info.fs_addr.iter().enumerate() {
        if addr == UNDEF_ADDR || addr == 0 {
            continue;
        }
        let class = if slot == FreeSpaceClass::RawData.message_slot() {
            FreeSpaceClass::RawData
        } else {
            FreeSpaceClass::Metadata
        };
        let hdr = FreeSpaceHeader::decode(&handle.read_at(addr, hdr_size)?, ctx)?;
        blocks.push((addr, hdr_size as u64));
        if hdr.sect_addr == UNDEF_ADDR || hdr.sect_size == 0 {
            continue;
        }
        let image = handle.read_at(hdr.sect_addr, hdr.sect_size as usize)?;
        let decoded = free_space::decode_sections(&image, &hdr, addr, ctx)?;
        blocks.push((hdr.sect_addr, hdr.alloc_sect_size.max(hdr.sect_size)));
        let bucket = &mut found.iter_mut().find(|(c, _)| *c == class).unwrap().1;
        bucket.extend(decoded.iter().map(|s| (s.addr, s.len)));
    }
    let mut sections = Vec::new();
    for (class, blocks) in &found {
        let merged = free_space::merge_sections(blocks).map_err(|why| {
            IoError::InvalidState(format!(
                "the free-space managers of this file overlap: {why}"
            ))
        })?;
        sections.extend(merged.iter().map(|s| FreeBlock {
            addr: s.addr,
            len: s.len,
            class: *class,
        }));
    }
    sections.sort_unstable_by_key(|b| b.addr);
    Ok(ManagerContents { blocks, sections })
}

/// Total bytes the managers `info` names record as free — `H5Fget_freespace`.
pub(crate) fn tracked_free_space(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    info: &FileSpaceInfoMessage,
) -> IoResult<u64> {
    Ok(read_managers(handle, ctx, info)?
        .sections
        .iter()
        .map(|s| s.len)
        .sum())
}
