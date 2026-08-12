use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// One released, reusable region of the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeBlock {
    addr: u64,
    len: u64,
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
/// old location. Like libhdf5's default (non-persistent) free-space strategy,
/// the list lives only for the session: a block released but not reused
/// before `close` stays as slack in the file rather than being recorded in an
/// on-disk free-space manager.
pub struct FileAllocator {
    eof: AtomicU64,
    alignment: u64,
    /// Released blocks, sorted by address with adjacent blocks merged.
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
    pub fn free(&self, addr: u64, len: u64) {
        if len == 0 {
            return;
        }
        let mut list = self.free_list.lock().unwrap();
        let pos = list.partition_point(|b| b.addr < addr);
        list.insert(pos, FreeBlock { addr, len });
        // Merge with the following block, then with the preceding one, so a
        // block that fills the gap between two free blocks yields one block.
        if pos + 1 < list.len() && list[pos].addr + list[pos].len == list[pos + 1].addr {
            list[pos].len += list[pos + 1].len;
            list.remove(pos + 1);
        }
        if pos > 0 && list[pos - 1].addr + list[pos - 1].len == list[pos].addr {
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
        // Best fit: the smallest sufficient block, so a large released region
        // stays available for a large chunk.
        let pos = list
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len >= size)
            .min_by_key(|(_, b)| b.len)
            .map(|(i, _)| i)?;
        let block = list[pos];
        // `block.addr` is aligned (every allocation is) and `used` is a
        // multiple of the alignment, so the remainder stays aligned too.
        let used = self.align_up(size);
        if block.len > used {
            list[pos] = FreeBlock {
                addr: block.addr + used,
                len: block.len - used,
            };
        } else {
            list.remove(pos);
        }
        self.free_count.store(list.len() as u64, Ordering::Release);
        Some(block.addr)
    }

    /// Return the current end-of-file offset.
    pub fn eof(&self) -> u64 {
        self.eof.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        alloc
            .free_list
            .lock()
            .unwrap()
            .iter()
            .map(|b| (b.addr, b.len))
            .collect()
    }

    #[test]
    fn freed_block_is_reused_before_the_file_grows() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64);
        alloc.allocate(64);
        let eof_before = alloc.eof();

        alloc.free(a, 64);
        assert_eq!(alloc.allocate(64), a, "exact-fit reuse");
        assert_eq!(alloc.eof(), eof_before, "file must not grow on reuse");
        assert!(free_blocks(&alloc).is_empty());
    }

    #[test]
    fn reusing_part_of_a_block_leaves_an_aligned_remainder() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(64);
        alloc.allocate(8);
        let eof_before = alloc.eof();

        alloc.free(a, 64);
        // 10 bytes round up to 16, so 48 bytes remain at a + 16.
        assert_eq!(alloc.allocate(10), a);
        assert_eq!(free_blocks(&alloc), vec![(a + 16, 48)]);
        assert_eq!(alloc.allocate(48), a + 16);
        assert_eq!(alloc.eof(), eof_before);
    }

    #[test]
    fn a_request_larger_than_every_free_block_grows_the_file() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(32);
        alloc.allocate(32);
        let eof_before = alloc.eof();

        alloc.free(a, 32);
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
        alloc.free(big, 64);
        alloc.free(small, 16);
        alloc.free(mid, 32);

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
        alloc.free(a, 32);
        alloc.free(c, 32);
        assert_eq!(free_blocks(&alloc), vec![(a, 32), (c, 32)]);

        // Filling the hole between them collapses all three into one block,
        // which is then large enough for a 96-byte request.
        alloc.free(b, 32);
        assert_eq!(free_blocks(&alloc), vec![(a, 96)]);
        let eof_before = alloc.eof();
        assert_eq!(alloc.allocate(96), a);
        assert_eq!(alloc.eof(), eof_before);
    }

    #[test]
    fn freeing_nothing_is_a_no_op() {
        let alloc = FileAllocator::new(0);
        let a = alloc.allocate(16);
        alloc.free(a, 0);
        assert!(free_blocks(&alloc).is_empty());
        // A zero-size request never consumes a free block either.
        alloc.free(a, 16);
        assert_eq!(alloc.allocate(0), alloc.eof());
        assert_eq!(free_blocks(&alloc), vec![(a, 16)]);
    }
}
