//! Loom models for the `threadsafe` writer's two lock-free / lock-ordered
//! primitives (Stage 3d of `docs/threadsafe-fine-grained-locking.md`).
//!
//! Loom exhaustively explores every thread interleaving and (for the weak
//! CAS) every spurious failure under the C11 memory model, which a wall-clock
//! stress test only samples. These models reimplement the *algorithm* of the
//! production code with loom primitives — they are not wired into the real
//! types, so production keeps plain `std` atomics/locks with zero loom
//! overhead. Keep them tiny: loom's state space is exponential in thread and
//! operation count.
//!
//! Not part of the normal build. Loom is a `cfg(loom)` dev-dependency, so run
//! these with:
//!
//! ```text
//! RUSTFLAGS="--cfg loom" cargo test --test loom_concurrency
//! ```

#![cfg(loom)]

use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::{Arc, Mutex};

/// Byte-for-byte the production `FileAllocator::allocate` CAS loop
/// (`src/io/allocator.rs`): align the current EOF up, bump by `size`, publish
/// with a weak compare-and-swap, retry on contention or spurious failure.
fn allocate(eof: &AtomicU64, alignment: u64, size: u64) -> u64 {
    let mut cur = eof.load(Ordering::Acquire);
    loop {
        let aligned = (cur + alignment - 1) & !(alignment - 1);
        let next = aligned + size;
        match eof.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return aligned,
            Err(actual) => cur = actual,
        }
    }
}

/// Two concurrent allocations must hand out distinct, non-overlapping,
/// aligned ranges under every interleaving — the invariant the writer relies
/// on to claim chunk space without a global lock. With `eof = 0`, `size = 8`,
/// `alignment = 8`, the only correct outcome set is `{0, 8}`.
#[test]
fn concurrent_allocate_two_threads_disjoint() {
    loom::model(|| {
        let eof = Arc::new(AtomicU64::new(0));

        let e1 = Arc::clone(&eof);
        let t1 = loom::thread::spawn(move || allocate(&e1, 8, 8));

        let a0 = allocate(&eof, 8, 8);
        let a1 = t1.join().unwrap();

        assert_ne!(a0, a1, "allocator handed out the same offset twice");
        let (lo, hi) = if a0 < a1 { (a0, a1) } else { (a1, a0) };
        assert_eq!(lo % 8, 0, "offset {lo} not 8-aligned");
        assert_eq!(hi % 8, 0, "offset {hi} not 8-aligned");
        assert!(hi >= lo + 8, "allocated ranges overlap: {lo}+8 > {hi}");
        // The two size-8 allocations from EOF 0 fully account for [0, 16).
        assert_eq!(eof.load(Ordering::Acquire), 16, "EOF must reach 16");
    });
}

/// Three concurrent allocations: still exhaustively checkable, and confirms
/// the retry loop never loses an update (no two threads see the same `cur`
/// win the CAS). The three size-8 allocations from EOF 0 must be exactly
/// `{0, 8, 16}`.
#[test]
fn concurrent_allocate_three_threads_disjoint() {
    loom::model(|| {
        let eof = Arc::new(AtomicU64::new(0));

        let e1 = Arc::clone(&eof);
        let e2 = Arc::clone(&eof);
        let t1 = loom::thread::spawn(move || allocate(&e1, 8, 8));
        let t2 = loom::thread::spawn(move || allocate(&e2, 8, 8));

        let a0 = allocate(&eof, 8, 8);
        let a1 = t1.join().unwrap();
        let a2 = t2.join().unwrap();

        let mut got = [a0, a1, a2];
        got.sort_unstable();
        assert_eq!(
            got,
            [0, 8, 16],
            "three allocations must tile [0, 24) exactly"
        );
        assert_eq!(eof.load(Ordering::Acquire), 24, "EOF must reach 24");
    });
}

/// Models the writer's registry lock order: a writer reaches a dataset slot
/// by cloning its `Arc` out of the spine *under the spine lock*, releasing the
/// spine lock, then locking only that one slot — while a concurrent creator
/// pushes a brand-new slot under the spine lock.
///
/// What this proves: under that production acquisition order (spine → slot,
/// spine released before the slot is locked), no interleaving deadlocks, and
/// the cloned `Arc` keeps slot 0 alive across the creator's `Vec` push (which
/// may reallocate the spine's backing store), so the write lands and the new
/// slot is independent. What it does NOT prove: that *arbitrary* code is
/// deadlock-free. It never holds a slot guard while taking the spine, so it
/// cannot catch a future slot → spine inversion (e.g. someone making a header
/// builder call `object_link_count` while holding the slot). That the
/// production code never inverts the order is enforced by the `ds`/`grp`
/// accessor contract (writer.rs) and the review audit, not by this model.
#[test]
fn spine_then_slot_no_deadlock_and_disjoint() {
    loom::model(|| {
        // Spine pre-seeded with one slot holding 0.
        let spine: Arc<Mutex<Vec<Arc<Mutex<u64>>>>> =
            Arc::new(Mutex::new(vec![Arc::new(Mutex::new(0))]));

        // Writer thread: clone slot 0 under the spine lock, drop the spine
        // lock, then lock just that slot and mutate it.
        let sw = Arc::clone(&spine);
        let writer = loom::thread::spawn(move || {
            let slot = {
                let guard = sw.lock().unwrap();
                Arc::clone(&guard[0])
            };
            *slot.lock().unwrap() = 10;
        });

        // Creator thread: push a fresh slot under the spine lock.
        let sc = Arc::clone(&spine);
        let creator = loom::thread::spawn(move || {
            let mut guard = sc.lock().unwrap();
            guard.push(Arc::new(Mutex::new(20)));
        });

        writer.join().unwrap();
        creator.join().unwrap();

        let guard = spine.lock().unwrap();
        assert_eq!(guard.len(), 2, "creator's slot must be present");
        assert_eq!(*guard[0].lock().unwrap(), 10, "writer's mutation must land");
        assert_eq!(*guard[1].lock().unwrap(), 20, "new slot keeps its value");
    });
}
