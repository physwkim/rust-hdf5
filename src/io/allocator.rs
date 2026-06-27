use std::sync::atomic::{AtomicU64, Ordering};

/// Simple append-only file space allocator.
///
/// Hands out file offsets by bumping an end-of-file pointer. Every
/// allocation is aligned to the configured boundary (default 8 bytes).
///
/// The end-of-file pointer is an [`AtomicU64`], so `allocate` takes `&self`
/// and is safe to call concurrently: two threads allocating at once each get
/// a distinct, non-overlapping, aligned offset. This is the lock-free
/// foundation that lets the `threadsafe` writer hand out chunk space without
/// a global lock (see `docs/threadsafe-fine-grained-locking.md`).
pub struct FileAllocator {
    eof: AtomicU64,
    alignment: u64,
}

impl FileAllocator {
    /// Create a new allocator whose free region starts at `initial_eof`.
    pub fn new(initial_eof: u64) -> Self {
        Self {
            eof: AtomicU64::new(initial_eof),
            alignment: 8,
        }
    }

    /// Allocate `size` bytes, returning the aligned starting offset.
    ///
    /// Lock-free: the aligned bump is published with a compare-and-swap loop,
    /// so concurrent callers never overlap (alignment makes a plain
    /// `fetch_add` insufficient, hence the CAS).
    pub fn allocate(&self, size: u64) -> u64 {
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

    /// Return the current end-of-file offset.
    pub fn eof(&self) -> u64 {
        self.eof.load(Ordering::Acquire)
    }

    /// Manually set the end-of-file offset.
    pub fn set_eof(&self, eof: u64) {
        self.eof.store(eof, Ordering::Release);
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
}
