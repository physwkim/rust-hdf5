use crate::format::free_space::{self, FreeSpaceHeader, FreeSpaceManager};
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
    /// out of, address-ordered.
    pub(crate) sections: Vec<FreeBlock>,
}

/// Walk every manager `info` names and return its sections.
///
/// Sections come back exactly as the file records them, tagged with the
/// manager they came out of; coalescing adjacent ones is the allocator's,
/// because which of them may merge depends on the file's strategy.
///
/// The manager comes from the message slot, which `H5F__super_read` fills as
/// `f->shared->fs_addr[u] = fsinfo.fs_addr[u - 1]` (H5Fsuper.c:831-833). Under
/// the sec2 driver only three of the twelve slots can ever be filled: an
/// unpaged file uses the dichotomy's two, and a paged one adds
/// `H5F_MEM_PAGE_GENERIC` in slot 6, because sec2 declares no
/// `H5FD_FEAT_PAGED_AGGR` and every large request therefore maps onto that one
/// manager. A file naming any other slot was written for a driver whose
/// address space this crate does not model, and is refused rather than read
/// with its sections silently dropped.
pub(crate) fn read_managers(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    info: &FileSpaceInfoMessage,
) -> IoResult<ManagerContents> {
    let policy = free_space::SpacePolicy::for_message(info);
    let hdr_size = FreeSpaceHeader::encoded_size(ctx);
    let mut blocks = Vec::new();
    let mut found: Vec<(FreeSpaceManager, Vec<(u64, u64)>)> = FreeSpaceManager::ALL
        .iter()
        .map(|&manager| (manager, Vec::new()))
        .collect();
    for (slot, &addr) in info.fs_addr.iter().enumerate() {
        if addr == UNDEF_ADDR || addr == 0 {
            continue;
        }
        let Some(manager) = FreeSpaceManager::from_message_slot(slot) else {
            return Err(IoError::InvalidState(format!(
                "the file names a free-space manager in message slot {slot}, which only a \
                 driver with a non-contiguous address space fills"
            )));
        };
        if manager == FreeSpaceManager::Large && policy.page().is_none() {
            return Err(IoError::InvalidState(
                "an unpaged file names the large free-space manager, which only paged \
                 aggregation has"
                    .into(),
            ));
        }
        let hdr = FreeSpaceHeader::decode(&handle.read_at(addr, hdr_size)?, ctx)?;
        blocks.push((addr, hdr_size as u64));
        if hdr.sect_addr == UNDEF_ADDR || hdr.sect_size == 0 {
            continue;
        }
        let image = handle.read_at(hdr.sect_addr, hdr.sect_size as usize)?;
        let decoded = free_space::decode_sections(&image, &hdr, addr, ctx)?;
        blocks.push((hdr.sect_addr, hdr.alloc_sect_size.max(hdr.sect_size)));
        let bucket = &mut found.iter_mut().find(|(m, _)| *m == manager).unwrap().1;
        bucket.extend(decoded.iter().map(|s| (s.addr, s.len)));
    }
    let mut sections: Vec<FreeBlock> = Vec::new();
    for (manager, blocks) in &found {
        sections.extend(blocks.iter().map(|&(addr, len)| FreeBlock {
            addr,
            len,
            manager: *manager,
        }));
    }
    free_space::check_disjoint(&sections.iter().map(|s| (s.addr, s.len)).collect::<Vec<_>>())
        .map_err(|why| {
            IoError::InvalidState(format!(
                "the free-space managers of this file overlap: {why}"
            ))
        })?;
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
