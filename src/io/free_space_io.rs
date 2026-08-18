//! Reading the on-disk free-space managers a file-space info message names.
//!
//! One owner for a walk both halves of the crate need. The writer reopens the
//! managers so a close can rewrite them; the reader reports how much space
//! they record, which is what `H5Fget_freespace` answers and what `h5stat -S`
//! prints as "Amount of tracked free space". The two differ only in policy —
//! the writer refuses the strategies whose allocation rules it does not model
//! — never in how the blocks are found, so the walk itself lives here.

use crate::format::free_space::{self, FreeSection, FreeSpaceHeader};
use crate::format::messages::superblock_ext::FileSpaceInfoMessage;
use crate::format::{FormatContext, UNDEF_ADDR};
use crate::io::file_handle::FileHandle;
use crate::io::{IoError, IoResult};

/// What one walk of an fsinfo message's managers found.
pub(crate) struct ManagerContents {
    /// The header and section-info blocks the managers themselves occupy.
    /// Space a rewrite supersedes, not space the file has free.
    pub(crate) blocks: Vec<(u64, u64)>,
    /// Every section the managers record, coalesced into one address-ordered
    /// set across all of them.
    pub(crate) sections: Vec<FreeSection>,
}

/// Walk every manager `info` names and return its sections.
///
/// Sections that were adjacent in *different* managers coalesce here: this
/// crate's allocator is typeless, so the set it works from is one set. A
/// caller that needs the per-manager split must do the walk itself.
pub(crate) fn read_managers(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    info: &FileSpaceInfoMessage,
) -> IoResult<ManagerContents> {
    let hdr_size = FreeSpaceHeader::encoded_size(ctx);
    let mut blocks = Vec::new();
    let mut sections: Vec<(u64, u64)> = Vec::new();
    for &addr in info.fs_addr.iter().filter(|&&a| a != UNDEF_ADDR && a != 0) {
        let hdr = FreeSpaceHeader::decode(&handle.read_at(addr, hdr_size)?, ctx)?;
        blocks.push((addr, hdr_size as u64));
        if hdr.sect_addr == UNDEF_ADDR || hdr.sect_size == 0 {
            continue;
        }
        let image = handle.read_at(hdr.sect_addr, hdr.sect_size as usize)?;
        let decoded = free_space::decode_sections(&image, &hdr, addr, ctx)?;
        blocks.push((hdr.sect_addr, hdr.alloc_sect_size.max(hdr.sect_size)));
        sections.extend(decoded.iter().map(|s| (s.addr, s.len)));
    }
    let sections = free_space::merge_sections(&sections).map_err(|why| {
        IoError::InvalidState(format!(
            "the free-space managers of this file overlap: {why}"
        ))
    })?;
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
