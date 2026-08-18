use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::format::free_space::{FreeSpaceClass, FreeSpaceManager, SpacePolicy};

/// One released, reusable region of the file, and the free-space manager that
/// records it.
///
/// The manager is decided where the block is released, from the class the
/// caller names and the block's own length — `H5MF_xfree` asks
/// `H5MF__alloc_to_fs_type` the same question with the same two arguments —
/// and read again on close, when the list is split across the managers the
/// file-space info message names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FreeBlock {
    pub(crate) addr: u64,
    pub(crate) len: u64,
    pub(crate) manager: FreeSpaceManager,
}

/// File space allocator: bump-the-end-of-file, with reuse of released blocks.
///
/// Hands out file offsets by bumping an end-of-file pointer. Every
/// allocation is aligned to the configured boundary (default 8 bytes).
///
/// The end-of-file pointer is an [`AtomicU64`], so `allocate` takes `&self`
/// and is safe to call concurrently: two threads allocating at once each get
/// a distinct, non-overlapping, aligned offset. This is the lock-free
/// foundation that lets the `threadsafe` writer hand out chunk space without
/// a global lock (see `docs/threadsafe-fine-grained-locking.md`). A writer
/// that never calls [`free`](Self::free) never touches the free list, and
/// [`allocate`](Self::allocate) skips its lock entirely while the list is
/// empty, so the streaming path keeps that lock-free fast path.
///
/// [`free`](Self::free) returns a block for reuse — the counterpart of
/// libhdf5's `H5MF_xfree`, called when a rewritten chunk no longer fits its
/// old location. Where the list comes from and where it goes both follow the
/// file. A file whose file-space info message says `persist` opens with the
/// sections its on-disk free-space managers recorded already in the list, so
/// this session allocates out of them the way `H5MF_alloc` does, and gets
/// what is left written back to the managers on close
/// (`Hdf5Writer::write_free_space_managers`). Every other file starts with an
/// empty list, and — like libhdf5's default, non-persistent strategy — a
/// block it released but did not reuse stays as slack.
///
/// Every released block carries the [`FreeSpaceManager`] its bytes belong to,
/// which is what lets the close split the list across the managers the file's
/// strategy defines, and what stops two adjacent blocks in different managers
/// from merging into one section no manager could hold. Reuse reads it too:
/// a request is served only out of its own manager's sections, the one
/// `H5MF_alloc` searches, so a block's manager is fixed for the life of the
/// file and the two accounts cannot drift apart.
pub struct FileAllocator {
    eof: AtomicU64,
    alignment: u64,
    /// How the file's strategy maps requests onto managers and pages. Fixed
    /// for the allocator's life: it is a file creation property.
    policy: SpacePolicy,
    /// Free regions of the file, sorted by address with adjacent regions
    /// merged: blocks this session released, and — for a persisting file —
    /// the sections it opened holding. Only the former are guaranteed to
    /// start on the alignment boundary.
    ///
    /// A plain `Mutex` regardless of the `threadsafe` feature: the allocator
    /// is shared across threads in both builds (see
    /// `concurrent_allocations_are_disjoint`), and this lock is only ever
    /// taken on the rare free/reuse path.
    free_list: Mutex<Vec<FreeBlock>>,
    /// `free_list.len()`, readable without taking the lock so the common
    /// never-freed case costs one relaxed load.
    free_count: AtomicU64,
}

impl FileAllocator {
    /// Create a new allocator whose free region starts at `initial_eof`, for
    /// a file with no page structure.
    #[cfg(test)]
    pub(crate) fn new(initial_eof: u64) -> Self {
        Self::with_policy(initial_eof, SpacePolicy::Aggr)
    }

    /// Create a new allocator whose free region starts at `initial_eof`,
    /// for a file whose strategy is known.
    ///
    /// A paged file aligns nothing to eight bytes: a small allocation is
    /// carved from a page and a large one is page-aligned, so the only
    /// boundary that exists is the page. An unpaged file keeps this crate's
    /// eight-byte alignment, which libhdf5 does not have (`H5Pset_alignment`
    /// defaults to a threshold and an alignment of one) but which every file
    /// this crate has written so far does.
    pub fn with_policy(initial_eof: u64, policy: SpacePolicy) -> Self {
        Self {
            eof: AtomicU64::new(initial_eof),
            alignment: match policy {
                SpacePolicy::Aggr => 8,
                SpacePolicy::Paged { .. } => 1,
            },
            policy,
            free_list: Mutex::new(Vec::new()),
            free_count: AtomicU64::new(0),
        }
    }

    /// How this file's strategy maps requests onto managers and pages.
    pub(crate) fn policy(&self) -> SpacePolicy {
        self.policy
    }

    /// Round `size` up to `alignment`, which must be a power of two.
    fn round_up(size: u64, alignment: u64) -> u64 {
        (size + alignment - 1) & !(alignment - 1)
    }

    /// The boundary an allocation drawn from `manager` must start on.
    ///
    /// `H5MF__open_fstype` (H5MF.c:325-330) gives a paged file's large manager
    /// an alignment of one page and every other manager `H5F_ALIGN_DEF`, which
    /// is one byte. Unpaged, this crate uses its own eight.
    fn draw_alignment(&self, manager: FreeSpaceManager) -> u64 {
        match (self.policy, manager) {
            (SpacePolicy::Paged { page }, FreeSpaceManager::Large) => page,
            (SpacePolicy::Paged { .. }, _) => 1,
            (SpacePolicy::Aggr, _) => self.alignment,
        }
    }

    /// Allocate `size` bytes for `class`, returning the starting offset.
    ///
    /// A released block in the manager the request maps to is reused before
    /// the file grows; otherwise the end-of-file pointer is bumped. The bump
    /// is lock-free: it is published with a compare-and-swap loop, so
    /// concurrent callers never overlap (alignment makes a plain `fetch_add`
    /// insufficient, hence the CAS).
    ///
    /// `class` is `H5MF_alloc`'s `alloc_type` reduced through the sec2
    /// driver's dichotomy. It decides which manager the request is served
    /// from and which one records whatever the request leaves over — the
    /// alignment fragment before it, the page remainder after it. Without it
    /// those bytes would be held by nothing and recorded by no manager, which
    /// is what `h5stat -S` counts as unaccounted space.
    pub fn allocate(&self, size: u64, class: FreeSpaceClass) -> u64 {
        match self.policy {
            SpacePolicy::Aggr => self.allocate_aggr(size, class),
            SpacePolicy::Paged { page } => self.allocate_paged(size, class, page),
        }
    }

    /// [`allocate`](Self::allocate) for a file with no page structure.
    ///
    /// Rounding the end-of-file pointer up skips the bytes between the old
    /// pointer and the aligned address; nothing else will ever name them, so
    /// they go on the free list under the class of the allocation that
    /// displaced them. That is `H5MF__aggr_alloc`'s own alignment fragment,
    /// which it hands to `H5MF_xfree(f, alloc_type, eoa_frag_addr,
    /// eoa_frag_size)` (H5MFaggr.c:339-341).
    fn allocate_aggr(&self, size: u64, class: FreeSpaceClass) -> u64 {
        if let Some(addr) = self.take_free(size, self.policy.manager(class, size)) {
            return addr;
        }
        let mut cur = self.eof.load(Ordering::Acquire);
        let aligned = loop {
            let aligned = Self::round_up(cur, self.alignment);
            match self.eof.compare_exchange_weak(
                cur,
                aligned + size,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break aligned,
                Err(actual) => cur = actual,
            }
        };
        if aligned > cur {
            self.free(cur, aligned - cur, class);
        }
        aligned
    }

    /// [`allocate`](Self::allocate) under paged aggregation —
    /// `H5MF__alloc_pagefs` (H5MF.c:858).
    ///
    /// A request of at least one page is served from the end of the file and
    /// the misaligned tail between it and the next page boundary
    /// (`H5MF_EOA_MISALIGN`) becomes a section of the large manager, which
    /// keeps the end-of-file pointer on a page boundary for the next one. A
    /// smaller request takes a whole page — itself a large request, so it
    /// comes from a free page if there is one — and the rest of that page
    /// becomes a section of the small manager for the request's own class,
    /// which is what keeps one page to one kind of data.
    fn allocate_paged(&self, size: u64, class: FreeSpaceClass, page: u64) -> u64 {
        if size == 0 {
            return self.eof.load(Ordering::Acquire);
        }
        let manager = self.policy.manager(class, size);
        if let Some(addr) = self.take_free(size, manager) {
            return addr;
        }
        if manager != FreeSpaceManager::Large {
            let new_page = self.allocate(page, class);
            self.record(new_page + size, page - size, manager);
            return new_page;
        }
        let mut cur = self.eof.load(Ordering::Acquire);
        let (start, addr, frag) = loop {
            // `H5MF__alloc_pagefs` asserts the end of the file is on a page
            // boundary and computes only the tail fragment
            // (`H5MF_EOA_MISALIGN`). The head below is what makes that true
            // rather than assumed: every allocation here leaves the end on a
            // boundary, so the only way `cur` is off one is a reopened file
            // whose end was, and rounding up puts the file back on its own
            // grid instead of laying a page-aligned block at an unaligned
            // address.
            let addr = Self::round_up(cur, page);
            let frag = (page - (addr + size) % page) % page;
            match self.eof.compare_exchange_weak(
                cur,
                addr + size + frag,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break (cur, addr, frag),
                Err(actual) => cur = actual,
            }
        };
        if addr > start {
            self.record(start, addr - start, FreeSpaceManager::Large);
        }
        if frag > 0 {
            self.record(addr + size, frag, FreeSpaceManager::Large);
        }
        addr
    }

    /// Release `len` bytes at `addr` for reuse by later allocations.
    ///
    /// The caller must have already dropped every reference to the block (for
    /// a chunk: the index entry must be about to point elsewhere). Which
    /// manager records it is `H5MF__alloc_to_fs_type`'s answer for the class
    /// the caller names and the length being freed, so a released block of at
    /// least one page goes to a paged file's large manager whatever it held.
    pub fn free(&self, addr: u64, len: u64, class: FreeSpaceClass) {
        if len == 0 {
            return;
        }
        self.record(addr, len, self.policy.manager(class, len));
    }

    /// Put one section into `manager`, merging it with what is already there.
    fn record(&self, addr: u64, len: u64, manager: FreeSpaceManager) {
        if len == 0 {
            return;
        }
        let mut list = self.free_list.lock().unwrap();
        self.insert_section(&mut list, FreeBlock { addr, len, manager });
        self.free_count.store(list.len() as u64, Ordering::Release);
    }

    /// Whether `a` and the section that follows it merge into one.
    ///
    /// Only within one manager: `H5FS__sect_merge` only ever sees the manager
    /// it was called for, so two adjacent sections in different managers are
    /// two sections upstream as well. A paged file's small sections carry the
    /// further rule that the merged section may not cross a page boundary
    /// (`H5MF__sect_small_can_merge`, H5MFsection.c:684-686); its large
    /// sections merge like simple ones.
    fn joins(&self, a: &FreeBlock, b: &FreeBlock) -> bool {
        if a.addr + a.len != b.addr || a.manager != b.manager {
            return false;
        }
        match self.policy {
            SpacePolicy::Paged { page } if a.manager != FreeSpaceManager::Large => {
                a.addr / page == (b.addr + b.len - 1) / page
            }
            _ => true,
        }
    }

    /// Insert `block` in address order and merge it with its neighbours.
    ///
    /// A small section that grows to exactly one page stops being a small
    /// section: `H5MF__sect_small_merge` (H5MFsection.c:728-733) hands the
    /// whole page back through `H5MF_xfree`, which re-maps it to the large
    /// manager, where it can then merge with the pages around it. The loop is
    /// that hand-back.
    fn insert_section(&self, list: &mut Vec<FreeBlock>, block: FreeBlock) {
        let mut block = block;
        loop {
            let mut pos = list.partition_point(|x| x.addr < block.addr);
            list.insert(pos, block);
            if pos + 1 < list.len() && self.joins(&list[pos], &list[pos + 1]) {
                list[pos].len += list[pos + 1].len;
                list.remove(pos + 1);
            }
            if pos > 0 && self.joins(&list[pos - 1], &list[pos]) {
                list[pos - 1].len += list[pos].len;
                list.remove(pos);
                pos -= 1;
            }
            match self.policy {
                SpacePolicy::Paged { page }
                    if list[pos].manager != FreeSpaceManager::Large && list[pos].len == page =>
                {
                    block = FreeBlock {
                        manager: FreeSpaceManager::Large,
                        ..list[pos]
                    };
                    list.remove(pos);
                }
                _ => {
                    self.shrink_to(list, pos);
                    return;
                }
            }
        }
    }

    /// Give the file back the section at `pos` if it is the end of the file.
    ///
    /// Space past the end of the file is not free space, it is space that was
    /// never allocated: recording it would have a manager name bytes the
    /// superblock's own end-of-file address says are not there. `H5MF_xfree`
    /// hands such a section to `H5MF__sect_simple_shrink`, which lowers the
    /// EOA by the whole of it (H5MFsection.c:428-460); a paged file's large
    /// section gives back only whole pages and keeps the part below the page
    /// boundary, so the end stays on the grid
    /// (`H5MF__sect_large_shrink`, H5MFsection.c:902-940), and a small section
    /// never reaches the end at all because it lives inside a page.
    ///
    /// The end of file is moved by compare-and-swap against the exact value
    /// this section ends at, so a concurrent [`allocate`](Self::allocate) that
    /// already moved it wins and the section simply stays on the list.
    fn shrink_to(&self, list: &mut Vec<FreeBlock>, pos: usize) {
        let block = list[pos];
        let keep = match self.policy {
            SpacePolicy::Aggr => 0,
            SpacePolicy::Paged { page } if block.manager == FreeSpaceManager::Large => {
                if block.len < page {
                    return;
                }
                (page - block.addr % page) % page
            }
            SpacePolicy::Paged { .. } => return,
        };
        if self
            .eof
            .compare_exchange(
                block.addr + block.len,
                block.addr + keep,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        if keep == 0 {
            list.remove(pos);
        } else {
            list[pos].len = keep;
        }
    }

    /// Take the smallest released block that fits `size`, splitting off the
    /// remainder. Returns `None` when nothing fits (or nothing was freed).
    ///
    /// Only `manager`'s own sections are searched, which is what `H5MF_alloc`
    /// does — it asks `fs_man[fs_type]` and nothing else. That a block belongs
    /// to one manager for its whole life is what makes the manager a property
    /// of the bytes: a request served out of another manager's section would
    /// hand those bytes back to its own manager when it released them, and the
    /// two accounts would drift a block apart on every reuse.
    fn take_free(&self, size: u64, manager: FreeSpaceManager) -> Option<u64> {
        if size == 0 || self.free_count.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut list = self.free_list.lock().unwrap();
        // What a block can actually hand out. A block this allocator released
        // itself starts aligned, but one read out of a file's free-space
        // manager starts wherever the file put it, and the bytes before the
        // first aligned address in it can hold no allocation.
        let usable = |b: &FreeBlock| {
            let align = self.draw_alignment(b.manager);
            b.len.saturating_sub(Self::round_up(b.addr, align) - b.addr)
        };
        // Best fit: the smallest sufficient block, so a large released region
        // stays available for a large chunk.
        let pos = list
            .iter()
            .enumerate()
            .filter(|(_, b)| b.manager == manager)
            .filter(|(_, b)| usable(b) >= size)
            .min_by_key(|(_, b)| usable(b))
            .map(|(i, _)| i)?;
        let block = list[pos];
        let addr = Self::round_up(block.addr, self.draw_alignment(block.manager));
        // Exactly `size`: what a caller is handed is what it will hand back,
        // so a block released later returns every byte drawn here. Rounding
        // the draw up to the alignment instead left the difference inside a
        // live block — bytes no structure holds and no free-space manager
        // records, which is what `h5stat -S` counts as unaccounted space.
        let used = size;
        let head = FreeBlock {
            addr: block.addr,
            len: addr - block.addr,
            manager: block.manager,
        };
        let tail = FreeBlock {
            addr: addr + used,
            len: block.addr + block.len - addr - used,
            manager: block.manager,
        };
        // Both remainders stay free, and stay in the manager the block was in
        // — `H5MF__find_sect` re-adds the remainder to the manager it carved
        // it from, so a large section carved below a page is still a large
        // section. The head is what alignment cost and the tail is what the
        // request did not use. A remainder too short to start an aligned
        // allocation is kept anyway: `usable` will pass it over, but it stays
        // recorded, and it merges the moment a neighbour is released.
        list.remove(pos);
        for b in [tail, head] {
            if b.len > 0 {
                list.insert(pos, b);
            }
        }
        self.free_count.store(list.len() as u64, Ordering::Release);
        Some(addr)
    }

    /// Try to grow the allocation `[addr, addr + len)` by `extra` bytes in
    /// place — libhdf5's `H5MF_try_extend`. Returns whether the block now
    /// extends to `addr + len + extra`.
    ///
    /// Two ways it can succeed, tried in `H5MF_try_extend`'s order: the
    /// block ends at the end of the file, so the end-of-file pointer moves
    /// (published by compare-and-swap, so a concurrent `allocate` cannot be
    /// handed the same region); or a released block starts exactly at
    /// `addr + len` and is large enough, so the front of it is consumed —
    /// exactly `extra` bytes of it, an extension being contiguous by
    /// definition, so there is no address to choose here.
    ///
    /// `class` names the manager the extension may draw from, the one
    /// `H5FS_sect_try_extend` is called on. On a paged file the end-of-file
    /// route also lays down the page fragment the growth leaves behind, so
    /// the file's end stays on a page boundary.
    pub fn try_extend(&self, addr: u64, len: u64, extra: u64, class: FreeSpaceClass) -> bool {
        if extra == 0 {
            return true;
        }
        let end = addr + len;
        // The manager is the one for the block as it stands, not as it will
        // stand: `H5MF_try_extend` maps `size`, not `size + extra_requested`
        // (H5MF.c:1304), so a small block grows within the small manager and
        // never becomes a large one by growing.
        let manager = self.policy.manager(class, len);
        if let SpacePolicy::Paged { page } = self.policy {
            // A small block lives inside one page, so it can only grow while
            // it stays there (H5MF.c:1285-1289); growing across the boundary
            // would make one block out of two pages of possibly different
            // kinds.
            if manager != FreeSpaceManager::Large && addr / page != (end + extra - 1) / page {
                return false;
            }
        }
        let mut cur = self.eof.load(Ordering::Acquire);
        while cur == end {
            // Only a large block reaches the end of a paged file — the end is
            // on a page boundary, so a small block that ended there could not
            // have grown without crossing it — which is what
            // `H5MF_try_extend` asserts before it lays the fragment down
            // (H5MF.c:1325-1327).
            let frag = match self.policy {
                SpacePolicy::Paged { page } if manager == FreeSpaceManager::Large => {
                    (page - (end + extra) % page) % page
                }
                _ => 0,
            };
            match self.eof.compare_exchange_weak(
                end,
                end + extra + frag,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if frag > 0 {
                        self.record(end + extra, frag, FreeSpaceManager::Large);
                    }
                    return true;
                }
                Err(actual) => cur = actual,
            }
        }

        if self.free_count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let mut list = self.free_list.lock().unwrap();
        // Exactly `extra`, for the reason `take_free` draws exactly `size`.
        let used = extra;
        let Some(pos) = list
            .iter()
            .position(|b| b.addr == end && b.len >= used && b.manager == manager)
        else {
            return false;
        };
        let block = list[pos];
        if block.len > used {
            list[pos] = FreeBlock {
                addr: block.addr + used,
                len: block.len - used,
                manager: block.manager,
            };
        } else {
            list.remove(pos);
        }
        self.free_count.store(list.len() as u64, Ordering::Release);
        true
    }

    /// Return the current end-of-file offset.
    pub fn eof(&self) -> u64 {
        self.eof.load(Ordering::Acquire)
    }

    /// Snapshot of the free list, as `(addr, len)` sorted by address with
    /// adjacent blocks already merged.
    ///
    /// The set a persisting file writes to its free-space manager on close,
    /// and the seam a reclamation test asserts on where the file size cannot
    /// show the reuse — a session that reuses the freed space immediately, or
    /// one whose file is dominated by something else.
    #[cfg(test)]
    pub(crate) fn free_blocks(&self) -> Vec<(u64, u64)> {
        self.free_list
            .lock()
            .unwrap()
            .iter()
            .map(|b| (b.addr, b.len))
            .collect()
    }

    /// Snapshot of the free list with each block's class, the form the close
    /// splits across the file's managers.
    pub(crate) fn free_extents(&self) -> Vec<FreeBlock> {
        self.free_list.lock().unwrap().clone()
    }

    /// Install `blocks` as the whole free list, replacing what is there.
    ///
    /// How a reopen puts the sections a file's free-space managers recorded
    /// back into circulation. They go in one at a time through the same
    /// [`insert_section`](Self::insert_section) every release uses, so the
    /// merge rules have one owner: sections that libhdf5 left separate because
    /// they sit in different managers, or in different pages of one, stay
    /// separate here too, and two that it would have merged merge. `blocks`
    /// must be non-overlapping, and need not be aligned — a file's sections
    /// sit at whatever addresses the file gave them.
    pub(crate) fn reset_free_list(&self, blocks: &[FreeBlock]) {
        let mut list = self.free_list.lock().unwrap();
        list.clear();
        for &block in blocks.iter().filter(|b| b.len > 0) {
            self.insert_section(&mut list, block);
        }
        self.free_count.store(list.len() as u64, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: FreeSpaceClass = FreeSpaceClass::Metadata;
    const RAW: FreeSpaceClass = FreeSpaceClass::RawData;

    /// A metadata free block, which is what every test that does not care
    /// about the class is exercising.
    fn block(addr: u64, len: u64) -> FreeBlock {
        FreeBlock {
            addr,
            len,
            manager: FreeSpaceManager::Metadata,
        }
    }

    /// Adjacent blocks merge only inside one manager: `H5FS__sect_merge` is
    /// called per manager, so two sections that touch across the metadata /
    /// raw-data line are two sections upstream too.
    #[test]
    fn adjacent_blocks_of_different_classes_do_not_merge() {
        let alloc = FileAllocator::new(1024);
        alloc.free(100, 50, META);
        alloc.free(150, 50, RAW);
        assert_eq!(
            alloc.free_extents(),
            vec![
                FreeBlock {
                    addr: 100,
                    len: 50,
                    manager: FreeSpaceManager::Metadata
                },
                FreeBlock {
                    addr: 150,
                    len: 50,
                    manager: FreeSpaceManager::RawData
                },
            ]
        );
    }

    #[test]
    fn adjacent_blocks_of_one_class_merge() {
        let alloc = FileAllocator::new(1024);
        alloc.free(100, 50, RAW);
        alloc.free(150, 50, RAW);
        assert_eq!(
            alloc.free_extents(),
            vec![FreeBlock {
                addr: 100,
                len: 100,
                manager: FreeSpaceManager::RawData
            }]
        );
    }

    /// The block that fills a gap merges with a same-class neighbour on each
    /// side and leaves an other-class neighbour alone.
    #[test]
    fn a_gap_filler_merges_only_with_its_own_class() {
        let alloc = FileAllocator::new(1024);
        alloc.free(100, 20, META);
        alloc.free(140, 20, RAW);
        alloc.free(120, 20, META);
        assert_eq!(
            alloc.free_extents(),
            vec![
                FreeBlock {
                    addr: 100,
                    len: 40,
                    manager: FreeSpaceManager::Metadata
                },
                FreeBlock {
                    addr: 140,
                    len: 20,
                    manager: FreeSpaceManager::RawData
                },
            ]
        );
    }

    /// The remainder of a carved block stays in the manager the block was in
    /// — `H5MF__find_sect` re-adds it to the manager it carved it from.
    #[test]
    fn the_remainder_of_a_reused_block_keeps_its_manager() {
        let alloc = FileAllocator::new(1024);
        alloc.free(200, 64, RAW);
        assert_eq!(alloc.allocate(16, RAW), 200);
        assert_eq!(
            alloc.free_extents(),
            vec![FreeBlock {
                addr: 216,
                len: 48,
                manager: FreeSpaceManager::RawData
            }]
        );
    }

    /// A request is served out of its own manager and no other, which is what
    /// `H5MF_alloc` does — it asks `fs_man[fs_type]` alone. Space the raw-data
    /// manager holds is not space a metadata request may take, however well it
    /// would fit: the byte would come back to the metadata manager when it was
    /// released, and the file's two accounts would drift a block apart.
    #[test]
    fn a_request_is_not_served_out_of_another_managers_section() {
        let alloc = FileAllocator::new(1024);
        alloc.free(200, 64, RAW);
        assert_eq!(alloc.allocate(16, META), 1024, "took the raw-data section");
        assert_eq!(free_blocks(&alloc), vec![(200, 64)]);
    }

    #[test]
    fn basic_allocation() {
        let alloc = FileAllocator::new(48);
        let a = alloc.allocate(100, META);
        assert_eq!(a, 48);
        assert_eq!(alloc.eof(), 148);
    }

    #[test]
    fn alignment() {
        let alloc = FileAllocator::new(50); // not 8-aligned
        let a = alloc.allocate(10, META);
        assert_eq!(a, 56); // aligned to 8
        assert_eq!(alloc.eof(), 66);
    }

    #[test]
    fn zero_size_allocation() {
        let alloc = FileAllocator::new(48);
        let a = alloc.allocate(0, META);
        assert_eq!(a, 48);
        assert_eq!(alloc.eof(), 48);
    }

    #[test]
    fn successive_allocations() {
        let alloc = FileAllocator::new(0);
        let a1 = alloc.allocate(10, META);
        let a2 = alloc.allocate(20, META);
        let a3 = alloc.allocate(5, META);
        assert_eq!(a1, 0);
        assert_eq!(a2, 16); // 10 -> aligned to 16
        assert_eq!(a3, 40); // 36 -> aligned to 40
    }

    // Concurrent allocation must hand out distinct, non-overlapping,
    // aligned ranges — the property the fine-grained-locking writer relies
    // on to let threads claim chunk space without a global lock.
    #[test]
    fn concurrent_allocations_are_disjoint() {
        use std::sync::Arc;
        use std::thread;

        let alloc = Arc::new(FileAllocator::new(0));
        let n_threads = 8;
        let per_thread = 1000;
        let size = 7u64; // unaligned size to exercise the alignment path

        let mut handles = Vec::new();
        for _ in 0..n_threads {
            let a = Arc::clone(&alloc);
            handles.push(thread::spawn(move || {
                let mut offs = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    offs.push(a.allocate(size, META));
                }
                offs
            }));
        }
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_unstable();
        // Every offset is 8-aligned and no two allocated ranges overlap.
        for w in all.windows(2) {
            assert_eq!(w[0] % 8, 0, "offset {} not 8-aligned", w[0]);
            assert!(
                w[1] >= w[0] + size,
                "ranges overlap: {} + {} > {}",
                w[0],
                size,
                w[1]
            );
        }
        assert_eq!(all.len(), n_threads * per_thread);
        // No duplicates.
        let unique = all.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(unique, all.len(), "duplicate offsets handed out");
    }

    /// Snapshot of the free list for assertions.
    fn free_blocks(alloc: &FileAllocator) -> Vec<(u64, u64)> {
        alloc.free_blocks()
    }

    #[test]
    fn freed_block_is_reused_before_the_file_grows() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64, META);
        alloc.allocate(64, META);
        let eof_before = alloc.eof();

        alloc.free(a, 64, META);
        assert_eq!(alloc.allocate(64, META), a, "exact-fit reuse");
        assert_eq!(alloc.eof(), eof_before, "file must not grow on reuse");
        assert!(free_blocks(&alloc).is_empty());
    }

    /// A partial reuse hands out exactly the bytes asked for, so the remainder
    /// starts wherever that ends — aligned or not. Rounding the draw up would
    /// bury the difference inside a live allocation, where no free-space
    /// manager can record it.
    #[test]
    fn reusing_part_of_a_block_leaves_exactly_the_undrawn_remainder() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64, META);
        alloc.allocate(8, META);
        let eof_before = alloc.eof();

        alloc.free(a, 64, META);
        assert_eq!(alloc.allocate(10, META), a);
        assert_eq!(free_blocks(&alloc), vec![(a + 10, 54)]);
        // The next draw still starts aligned; the six bytes that costs stay on
        // the list instead of disappearing into the allocation before them.
        assert_eq!(alloc.allocate(48, META), a + 16);
        assert_eq!(free_blocks(&alloc), vec![(a + 10, 6)]);
        assert_eq!(alloc.eof(), eof_before);
    }

    /// Growing an unaligned file leaves the bytes between its end and the
    /// next boundary held by nothing, so `allocate` records them rather than
    /// stepping over them — `H5MF__aggr_alloc` hands the same fragment to
    /// `H5MF_xfree` (H5MFaggr.c:339-341). That is what lets the managers'
    /// own blocks come out of this allocator at all: `settle_free_space_managers`
    /// re-reads the section set each round, so a fragment one round leaves is
    /// inside the set the next round records, and no byte below the end of
    /// the file is outside every account.
    #[test]
    fn allocating_at_the_end_leaves_no_alignment_gap() {
        let alloc = FileAllocator::new(0);
        alloc.allocate(5, META);
        assert_eq!(alloc.eof(), 5);

        // 5 is not a multiple of eight, so the draw starts at 8 — and 5..8
        // lands on the free list instead of becoming a byte no structure
        // holds and no manager names.
        assert_eq!(alloc.allocate(16, META), 8);
        assert_eq!(alloc.eof(), 24);
        assert_eq!(free_blocks(&alloc), vec![(5, 3)]);

        // Every byte below the end of the file is now either allocated or on
        // the list: 0..5 and 8..24 are held, 5..8 is recorded.
        let recorded: u64 = free_blocks(&alloc).iter().map(|(_, len)| len).sum();
        assert_eq!(5 + 16 + recorded, alloc.eof());
    }

    #[test]
    fn a_request_larger_than_every_free_block_grows_the_file() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(32, META);
        alloc.allocate(32, META);
        let eof_before = alloc.eof();

        alloc.free(a, 32, META);
        let big = alloc.allocate(33, META);
        assert_eq!(big, eof_before, "must come from the end of the file");
        assert_eq!(
            free_blocks(&alloc),
            vec![(a, 32)],
            "the block that did not fit stays available"
        );
    }

    #[test]
    fn best_fit_picks_the_smallest_sufficient_block() {
        let alloc = FileAllocator::new(0);
        // Separators keep the three blocks apart, so freeing them cannot
        // merge them into one and the choice between them is a real one.
        let small = alloc.allocate(16, META);
        alloc.allocate(8, META);
        let mid = alloc.allocate(32, META);
        alloc.allocate(8, META);
        let big = alloc.allocate(64, META);
        alloc.allocate(8, META);
        alloc.free(big, 64, META);
        alloc.free(small, 16, META);
        alloc.free(mid, 32, META);

        assert_eq!(
            alloc.allocate(20, META),
            mid,
            "20 fits 32 more tightly than 64"
        );
        assert_eq!(alloc.allocate(16, META), small);
        assert_eq!(alloc.allocate(64, META), big);
    }

    #[test]
    fn adjacent_freed_blocks_merge() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(32, META);
        let b = alloc.allocate(32, META);
        let c = alloc.allocate(32, META);
        alloc.allocate(8, META);

        // Free the outer two first: they are not adjacent, so they stay apart.
        alloc.free(a, 32, META);
        alloc.free(c, 32, META);
        assert_eq!(free_blocks(&alloc), vec![(a, 32), (c, 32)]);

        // Filling the hole between them collapses all three into one block,
        // which is then large enough for a 96-byte request.
        alloc.free(b, 32, META);
        assert_eq!(free_blocks(&alloc), vec![(a, 96)]);
        let eof_before = alloc.eof();
        assert_eq!(alloc.allocate(96, META), a);
        assert_eq!(alloc.eof(), eof_before);
    }

    #[test]
    fn try_extend_grows_the_file_when_the_block_ends_at_eof() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64, META);
        assert!(alloc.try_extend(a, 64, 32, META));
        assert_eq!(alloc.eof(), 96);
        // The extension owns [64, 96): the next allocation starts after it.
        assert_eq!(alloc.allocate(8, META), 96);
    }

    #[test]
    fn try_extend_consumes_the_front_of_an_adjacent_free_block() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64, META);
        let b = alloc.allocate(64, META);
        alloc.allocate(8, META); // pin: the freed block is not at EOF
        alloc.free(b, 64, META);

        assert!(alloc.try_extend(a, 64, 16, META));
        assert_eq!(free_blocks(&alloc), vec![(b + 16, 48)]);
        // Growing into the whole remainder empties the list.
        assert!(alloc.try_extend(a, 80, 48, META));
        assert!(free_blocks(&alloc).is_empty());
    }

    #[test]
    fn try_extend_fails_without_room_past_the_block() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64, META);
        alloc.allocate(64, META); // live block right after `a`
        let c = alloc.allocate(16, META);
        alloc.allocate(8, META); // keeps `c` off the end of the file
        alloc.free(c, 16, META); // free space exists, but not at a + 64

        assert!(!alloc.try_extend(a, 64, 8, META));
        assert_eq!(free_blocks(&alloc), vec![(c, 16)], "nothing consumed");
    }

    #[test]
    fn try_extend_fails_when_the_adjacent_block_is_too_small() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64, META);
        let b = alloc.allocate(32, META);
        alloc.allocate(8, META);
        alloc.free(b, 32, META);

        assert!(!alloc.try_extend(a, 64, 40, META), "32 free < 40 wanted");
        assert_eq!(free_blocks(&alloc), vec![(b, 32)], "nothing consumed");
    }

    /// A section read out of a file's free-space manager starts wherever the
    /// file put it. The allocation still comes back aligned, and the bytes
    /// alignment skipped stay free rather than becoming untracked slack.
    #[test]
    fn an_unaligned_block_is_carved_from_its_first_aligned_address() {
        let alloc = FileAllocator::new(4096);
        alloc.reset_free_list(&[block(185, 15)]);

        assert_eq!(alloc.allocate(8, META), 192);
        // The seven bytes before 192 stay on the list; the block ends exactly
        // where the allocation does, so there is no tail.
        assert_eq!(free_blocks(&alloc), vec![(185, 7)]);
        assert_eq!(
            alloc.eof(),
            4096,
            "the file must not grow while it has room"
        );
    }

    /// Both remainders at once: the head alignment skipped and the tail the
    /// allocation did not reach.
    #[test]
    fn carving_an_unaligned_block_keeps_the_head_and_the_tail() {
        let alloc = FileAllocator::new(4096);
        alloc.reset_free_list(&[block(2038, 100)]);

        assert_eq!(alloc.allocate(16, META), 2040);
        assert_eq!(free_blocks(&alloc), vec![(2038, 2), (2056, 82)]);
    }

    /// A draw that takes a block's usable part to its end leaves only the head
    /// alignment cost, and nothing past the block: the draw is `size`, never
    /// `size` rounded up.
    #[test]
    fn a_block_whose_usable_part_exactly_fits_leaves_only_the_head() {
        let alloc = FileAllocator::new(4096);
        alloc.reset_free_list(&[block(2038, 12)]);

        // Ten usable bytes from 2040, and ten are asked for.
        assert_eq!(alloc.allocate(10, META), 2040);
        assert_eq!(free_blocks(&alloc), vec![(2038, 2)]);
    }

    /// Best fit is by what a block can hand out, not by its length: the longer
    /// but badly aligned block loses to the shorter aligned one.
    #[test]
    fn best_fit_measures_the_usable_part_of_a_block() {
        let alloc = FileAllocator::new(4096);
        alloc.reset_free_list(&[block(1001, 9), block(2000, 8)]);

        assert_eq!(alloc.allocate(8, META), 2000);
        assert_eq!(free_blocks(&alloc), vec![(1001, 9)]);
        // Nothing left can hold eight aligned bytes, so the file grows.
        assert_eq!(alloc.allocate(8, META), 4096);
    }

    #[test]
    fn freeing_nothing_is_a_no_op() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(16, META);
        alloc.allocate(8, META); // keeps `a` off the end of the file
        alloc.free(a, 0, META);
        assert!(free_blocks(&alloc).is_empty());
        // A zero-size request never consumes a free block either.
        alloc.free(a, 16, META);
        assert_eq!(alloc.allocate(0, META), alloc.eof());
        assert_eq!(free_blocks(&alloc), vec![(a, 16)]);
    }

    /// Space at the end of the file is given back to the file rather than
    /// recorded: `H5MF__sect_simple_shrink` lowers the EOA by the whole
    /// section, because a manager that recorded it would be naming bytes the
    /// superblock's own end-of-file address says are not there.
    #[test]
    fn freeing_the_end_of_the_file_shrinks_it() {
        let alloc = FileAllocator::new(0);
        alloc.allocate(64, META);
        let b = alloc.allocate(32, META);
        alloc.free(b, 32, META);
        assert_eq!(alloc.eof(), 64);
        assert!(free_blocks(&alloc).is_empty());
    }

    /// The shrink takes the merged section, not the block just released:
    /// `H5FS__sect_merge` runs before the shrink callback, so a block released
    /// against a section that already reached the end gives back both.
    #[test]
    fn a_release_that_merges_into_the_end_of_the_file_shrinks_all_of_it() {
        let alloc = FileAllocator::new(0);
        alloc.allocate(64, META);
        let b = alloc.allocate(32, META);
        let c = alloc.allocate(16, META);
        alloc.allocate(8, RAW); // raw-data space, which cannot merge with `c`
        alloc.free(c, 16, META);
        assert_eq!(alloc.eof(), 120, "a metadata block behind live space");
        alloc.free(112, 8, RAW);
        assert_eq!(alloc.eof(), 112, "only the raw-data block came back");
        alloc.free(b, 32, META);
        assert_eq!(alloc.eof(), 64, "the merged metadata section came back");
        assert!(free_blocks(&alloc).is_empty());
    }

    /// A paged file gives back whole pages and keeps what is below the page
    /// boundary, so its end stays on the grid — `H5MF__sect_large_shrink`.
    #[test]
    fn a_paged_file_shrinks_by_whole_pages() {
        let page = 4096;
        let alloc = FileAllocator::with_policy(0, SpacePolicy::Paged { page });
        alloc.allocate(64, META); // takes page 0, records the rest of it
        let big = alloc.allocate(3 * page, META);
        assert_eq!(big, page);
        assert_eq!(alloc.eof(), 4 * page);
        // Giving back two of the three pages leaves the end on a boundary.
        alloc.free(2 * page, 2 * page, META);
        assert_eq!(alloc.eof(), 2 * page);
        assert_eq!(free_blocks(&alloc), vec![(64, page - 64)]);
    }
}
