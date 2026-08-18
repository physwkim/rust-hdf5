use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::format::free_space::FreeSpaceClass;

/// One released, reusable region of the file, and the free-space manager it
/// belongs to.
///
/// The class travels with the block because it is decided where the block is
/// released — the caller is the only place that knows what the bytes held —
/// and read again on close, when the sections are split across the managers
/// `H5MF_ALLOC_TO_FS_AGGR_TYPE` would have put them in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FreeBlock {
    pub(crate) addr: u64,
    pub(crate) len: u64,
    pub(crate) class: FreeSpaceClass,
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
/// Every released block carries the [`FreeSpaceClass`] its bytes belonged to,
/// which is what lets the close split the list across the managers
/// `H5MF_ALLOC_TO_FS_AGGR_TYPE` would have chosen, and what stops two adjacent
/// blocks in different managers from merging into one section no manager could
/// hold. Reuse does not read it: `allocate` takes the best fit from the whole
/// list where `H5MF_alloc` would search only the manager for its own type,
/// so a metadata allocation here can be handed raw-data space. What that
/// changes is which free block a request gets, never whether the block is
/// free or which manager records what is left of it.
pub struct FileAllocator {
    eof: AtomicU64,
    alignment: u64,
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
    /// Create a new allocator whose free region starts at `initial_eof`.
    pub fn new(initial_eof: u64) -> Self {
        Self {
            eof: AtomicU64::new(initial_eof),
            alignment: 8,
            free_list: Mutex::new(Vec::new()),
            free_count: AtomicU64::new(0),
        }
    }

    /// Round `size` up to the allocator's alignment.
    fn align_up(&self, size: u64) -> u64 {
        (size + self.alignment - 1) & !(self.alignment - 1)
    }

    /// Allocate `size` bytes, returning the aligned starting offset.
    ///
    /// A released block large enough to hold `size` is reused before the file
    /// grows; otherwise the end-of-file pointer is bumped. The bump is
    /// lock-free: it is published with a compare-and-swap loop, so concurrent
    /// callers never overlap (alignment makes a plain `fetch_add`
    /// insufficient, hence the CAS).
    pub fn allocate(&self, size: u64) -> u64 {
        if let Some(addr) = self.take_free(size) {
            return addr;
        }
        let mut cur = self.eof.load(Ordering::Acquire);
        loop {
            let aligned = (cur + self.alignment - 1) & !(self.alignment - 1);
            let next = aligned + size;
            match self
                .eof
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return aligned,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Release `len` bytes at `addr` for reuse by later allocations.
    ///
    /// The caller must have already dropped every reference to the block (for
    /// a chunk: the index entry must be about to point elsewhere). Adjacent
    /// blocks merge, so a repeatedly grown-and-shrunk chunk does not shred the
    /// list into unusable fragments.
    pub fn free(&self, addr: u64, len: u64, class: FreeSpaceClass) {
        if len == 0 {
            return;
        }
        let mut list = self.free_list.lock().unwrap();
        let pos = list.partition_point(|b| b.addr < addr);
        list.insert(pos, FreeBlock { addr, len, class });
        // Merge with the following block, then with the preceding one, so a
        // block that fills the gap between two free blocks yields one block.
        // Only within a class: two adjacent sections in different managers are
        // two sections upstream as well, because `H5FS__sect_merge` only ever
        // sees the one manager it was called for.
        let joins = |a: &FreeBlock, b: &FreeBlock| a.addr + a.len == b.addr && a.class == b.class;
        if pos + 1 < list.len() && joins(&list[pos], &list[pos + 1]) {
            list[pos].len += list[pos + 1].len;
            list.remove(pos + 1);
        }
        if pos > 0 && joins(&list[pos - 1], &list[pos]) {
            list[pos - 1].len += list[pos].len;
            list.remove(pos);
        }
        self.free_count.store(list.len() as u64, Ordering::Release);
    }

    /// Take the smallest released block that fits `size`, splitting off the
    /// remainder. Returns `None` when nothing fits (or nothing was freed).
    fn take_free(&self, size: u64) -> Option<u64> {
        if size == 0 || self.free_count.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut list = self.free_list.lock().unwrap();
        // What a block can actually hand out. A block this allocator released
        // itself starts aligned, but one read out of a file's free-space
        // manager starts wherever the file put it, and the bytes before the
        // first aligned address in it can hold no allocation.
        let usable = |b: &FreeBlock| b.len.saturating_sub(self.align_up(b.addr) - b.addr);
        // Best fit: the smallest sufficient block, so a large released region
        // stays available for a large chunk.
        let pos = list
            .iter()
            .enumerate()
            .filter(|(_, b)| usable(b) >= size)
            .min_by_key(|(_, b)| usable(b))
            .map(|(i, _)| i)?;
        let block = list[pos];
        let addr = self.align_up(block.addr);
        // Exactly `size`: what a caller is handed is what it will hand back,
        // so a block released later returns every byte drawn here. Rounding
        // the draw up to the alignment instead left the difference inside a
        // live block — bytes no structure holds and no free-space manager
        // records, which is what `h5stat -S` counts as unaccounted space.
        let used = size;
        let head = FreeBlock {
            addr: block.addr,
            len: addr - block.addr,
            class: block.class,
        };
        let tail = FreeBlock {
            addr: addr + used,
            len: block.addr + block.len - addr - used,
            class: block.class,
        };
        // Both remainders stay free: the head is what alignment cost and the
        // tail is what the request did not use. A remainder too short to
        // start an aligned allocation is kept anyway — `usable` will pass it
        // over, but it stays recorded, and it merges the moment a neighbour
        // is released.
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
    pub fn try_extend(&self, addr: u64, len: u64, extra: u64) -> bool {
        if extra == 0 {
            return true;
        }
        let end = addr + len;
        let mut cur = self.eof.load(Ordering::Acquire);
        while cur == end {
            match self.eof.compare_exchange_weak(
                end,
                end + extra,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }

        if self.free_count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let mut list = self.free_list.lock().unwrap();
        // Exactly `extra`, for the reason `take_free` draws exactly `size`.
        let used = extra;
        let Some(pos) = list.iter().position(|b| b.addr == end && b.len >= used) else {
            return false;
        };
        let block = list[pos];
        if block.len > used {
            list[pos] = FreeBlock {
                addr: block.addr + used,
                len: block.len - used,
                class: block.class,
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
    #[cfg(test)]
    pub(crate) fn free_extents(&self) -> Vec<FreeBlock> {
        self.free_list.lock().unwrap().clone()
    }

    /// Empty the free list, returning what was in it.
    ///
    /// Leaves [`allocate`](Self::allocate) with nothing to reuse, so every
    /// call after this one bumps the end of the file. That is what lets the
    /// close settle a free-space manager over the list: the blocks the manager
    /// itself needs must come out of a set nothing else can allocate from
    /// while the layout is being chosen.
    pub(crate) fn take_all_free(&self) -> Vec<FreeBlock> {
        let mut list = self.free_list.lock().unwrap();
        let taken = list.clone();
        list.clear();
        self.free_count.store(0, Ordering::Release);
        taken
    }

    /// Put `blocks` back as the whole free list, replacing what is there.
    ///
    /// The counterpart of [`take_all_free`](Self::take_all_free). The list is
    /// sorted here rather than by the caller: address order is what
    /// [`free`](Self::free) and [`try_extend`](Self::try_extend) search by, so
    /// a caller that handed the blocks over in some other order — a free-space
    /// manager's layout orders its sections by size — would leave the list
    /// unsearchable. `blocks` must still be non-overlapping, and need not be
    /// aligned: this is also how a reopen installs the sections a file's
    /// manager recorded, which sit at whatever addresses the file gave them.
    pub(crate) fn reset_free_list(&self, blocks: &[FreeBlock]) {
        let mut sorted: Vec<FreeBlock> = blocks.iter().copied().filter(|b| b.len > 0).collect();
        sorted.sort_unstable_by_key(|b| b.addr);
        let mut list = self.free_list.lock().unwrap();
        *list = sorted;
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
            class: META,
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
                    class: META
                },
                FreeBlock {
                    addr: 150,
                    len: 50,
                    class: RAW
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
                class: RAW
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
                    class: META
                },
                FreeBlock {
                    addr: 140,
                    len: 20,
                    class: RAW
                },
            ]
        );
    }

    /// The remainder of a carved block keeps the class of the block it came
    /// from, whichever class asked for the space.
    #[test]
    fn the_remainder_of_a_reused_block_keeps_its_class() {
        let alloc = FileAllocator::new(1024);
        alloc.free(200, 64, RAW);
        assert_eq!(alloc.allocate(16), 200);
        assert_eq!(
            alloc.free_extents(),
            vec![FreeBlock {
                addr: 216,
                len: 48,
                class: RAW
            }]
        );
    }

    #[test]
    fn basic_allocation() {
        let alloc = FileAllocator::new(48);
        let a = alloc.allocate(100);
        assert_eq!(a, 48);
        assert_eq!(alloc.eof(), 148);
    }

    #[test]
    fn alignment() {
        let alloc = FileAllocator::new(50); // not 8-aligned
        let a = alloc.allocate(10);
        assert_eq!(a, 56); // aligned to 8
        assert_eq!(alloc.eof(), 66);
    }

    #[test]
    fn zero_size_allocation() {
        let alloc = FileAllocator::new(48);
        let a = alloc.allocate(0);
        assert_eq!(a, 48);
        assert_eq!(alloc.eof(), 48);
    }

    #[test]
    fn successive_allocations() {
        let alloc = FileAllocator::new(0);
        let a1 = alloc.allocate(10);
        let a2 = alloc.allocate(20);
        let a3 = alloc.allocate(5);
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
                    offs.push(a.allocate(size));
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
        let a = alloc.allocate(64);
        alloc.allocate(64);
        let eof_before = alloc.eof();

        alloc.free(a, 64, META);
        assert_eq!(alloc.allocate(64), a, "exact-fit reuse");
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
        let a = alloc.allocate(64);
        alloc.allocate(8);
        let eof_before = alloc.eof();

        alloc.free(a, 64, META);
        assert_eq!(alloc.allocate(10), a);
        assert_eq!(free_blocks(&alloc), vec![(a + 10, 54)]);
        // The next draw still starts aligned; the six bytes that costs stay on
        // the list instead of disappearing into the allocation before them.
        assert_eq!(alloc.allocate(48), a + 16);
        assert_eq!(free_blocks(&alloc), vec![(a + 10, 6)]);
        assert_eq!(alloc.eof(), eof_before);
    }

    #[test]
    fn a_request_larger_than_every_free_block_grows_the_file() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(32);
        alloc.allocate(32);
        let eof_before = alloc.eof();

        alloc.free(a, 32, META);
        let big = alloc.allocate(33);
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
        let small = alloc.allocate(16);
        alloc.allocate(8);
        let mid = alloc.allocate(32);
        alloc.allocate(8);
        let big = alloc.allocate(64);
        alloc.allocate(8);
        alloc.free(big, 64, META);
        alloc.free(small, 16, META);
        alloc.free(mid, 32, META);

        assert_eq!(alloc.allocate(20), mid, "20 fits 32 more tightly than 64");
        assert_eq!(alloc.allocate(16), small);
        assert_eq!(alloc.allocate(64), big);
    }

    #[test]
    fn adjacent_freed_blocks_merge() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(32);
        let b = alloc.allocate(32);
        let c = alloc.allocate(32);
        alloc.allocate(8);

        // Free the outer two first: they are not adjacent, so they stay apart.
        alloc.free(a, 32, META);
        alloc.free(c, 32, META);
        assert_eq!(free_blocks(&alloc), vec![(a, 32), (c, 32)]);

        // Filling the hole between them collapses all three into one block,
        // which is then large enough for a 96-byte request.
        alloc.free(b, 32, META);
        assert_eq!(free_blocks(&alloc), vec![(a, 96)]);
        let eof_before = alloc.eof();
        assert_eq!(alloc.allocate(96), a);
        assert_eq!(alloc.eof(), eof_before);
    }

    #[test]
    fn try_extend_grows_the_file_when_the_block_ends_at_eof() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64);
        assert!(alloc.try_extend(a, 64, 32));
        assert_eq!(alloc.eof(), 96);
        // The extension owns [64, 96): the next allocation starts after it.
        assert_eq!(alloc.allocate(8), 96);
    }

    #[test]
    fn try_extend_consumes_the_front_of_an_adjacent_free_block() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64);
        let b = alloc.allocate(64);
        alloc.allocate(8); // pin: the freed block is not at EOF
        alloc.free(b, 64, META);

        assert!(alloc.try_extend(a, 64, 16));
        assert_eq!(free_blocks(&alloc), vec![(b + 16, 48)]);
        // Growing into the whole remainder empties the list.
        assert!(alloc.try_extend(a, 80, 48));
        assert!(free_blocks(&alloc).is_empty());
    }

    #[test]
    fn try_extend_fails_without_room_past_the_block() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64);
        alloc.allocate(64); // live block right after `a`
        let c = alloc.allocate(16);
        alloc.free(c, 16, META); // free space exists, but not at a + 64

        assert!(!alloc.try_extend(a, 64, 8));
        assert_eq!(free_blocks(&alloc), vec![(c, 16)], "nothing consumed");
    }

    #[test]
    fn try_extend_fails_when_the_adjacent_block_is_too_small() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64);
        let b = alloc.allocate(32);
        alloc.allocate(8);
        alloc.free(b, 32, META);

        assert!(!alloc.try_extend(a, 64, 40), "32 free < 40 wanted");
        assert_eq!(free_blocks(&alloc), vec![(b, 32)], "nothing consumed");
    }

    /// A section read out of a file's free-space manager starts wherever the
    /// file put it. The allocation still comes back aligned, and the bytes
    /// alignment skipped stay free rather than becoming untracked slack.
    #[test]
    fn an_unaligned_block_is_carved_from_its_first_aligned_address() {
        let alloc = FileAllocator::new(4096);
        alloc.reset_free_list(&[block(185, 15)]);

        assert_eq!(alloc.allocate(8), 192);
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

        assert_eq!(alloc.allocate(16), 2040);
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
        assert_eq!(alloc.allocate(10), 2040);
        assert_eq!(free_blocks(&alloc), vec![(2038, 2)]);
    }

    /// Best fit is by what a block can hand out, not by its length: the longer
    /// but badly aligned block loses to the shorter aligned one.
    #[test]
    fn best_fit_measures_the_usable_part_of_a_block() {
        let alloc = FileAllocator::new(4096);
        alloc.reset_free_list(&[block(1001, 9), block(2000, 8)]);

        assert_eq!(alloc.allocate(8), 2000);
        assert_eq!(free_blocks(&alloc), vec![(1001, 9)]);
        // Nothing left can hold eight aligned bytes, so the file grows.
        assert_eq!(alloc.allocate(8), 4096);
    }

    #[test]
    fn freeing_nothing_is_a_no_op() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(16);
        alloc.free(a, 0, META);
        assert!(free_blocks(&alloc).is_empty());
        // A zero-size request never consumes a free block either.
        alloc.free(a, 16, META);
        assert_eq!(alloc.allocate(0), alloc.eof());
        assert_eq!(free_blocks(&alloc), vec![(a, 16)]);
    }
}
