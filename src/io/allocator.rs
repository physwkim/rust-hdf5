/// Simple append-only file space allocator.
///
/// Hands out file offsets by bumping an end-of-file pointer. Every
/// allocation is aligned to the configured boundary (default 8 bytes).
pub struct FileAllocator {
    eof: u64,
    alignment: u64,
}

impl FileAllocator {
    /// Create a new allocator whose free region starts at `initial_eof`.
    pub fn new(initial_eof: u64) -> Self {
        Self {
            eof: initial_eof,
            alignment: 8,
        }
    }

    /// Allocate `size` bytes, returning the aligned starting offset.
    pub fn allocate(&mut self, size: u64) -> u64 {
        let aligned = (self.eof + self.alignment - 1) & !(self.alignment - 1);
        self.eof = aligned + size;
        aligned
    }

    /// Return the current end-of-file offset.
    pub fn eof(&self) -> u64 {
        self.eof
    }

    /// Manually set the end-of-file offset.
    pub fn set_eof(&mut self, eof: u64) {
        self.eof = eof;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_allocation() {
        let mut alloc = FileAllocator::new(48);
        let a = alloc.allocate(100);
        assert_eq!(a, 48);
        assert_eq!(alloc.eof(), 148);
    }

    #[test]
    fn alignment() {
        let mut alloc = FileAllocator::new(50); // not 8-aligned
        let a = alloc.allocate(10);
        assert_eq!(a, 56); // aligned to 8
        assert_eq!(alloc.eof(), 66);
    }

    #[test]
    fn zero_size_allocation() {
        let mut alloc = FileAllocator::new(48);
        let a = alloc.allocate(0);
        assert_eq!(a, 48);
        assert_eq!(alloc.eof(), 48);
    }

    #[test]
    fn successive_allocations() {
        let mut alloc = FileAllocator::new(0);
        let a1 = alloc.allocate(10);
        let a2 = alloc.allocate(20);
        let a3 = alloc.allocate(5);
        assert_eq!(a1, 0);
        assert_eq!(a2, 16); // 10 -> aligned to 16
        assert_eq!(a3, 40); // 36 -> aligned to 40
    }
}
