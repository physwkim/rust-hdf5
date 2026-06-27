# Fine-grained locking for the `threadsafe` writer (design)

Status: **proposed — awaiting sign-off before implementation.**

## 1. Problem

`H5File` shares its inner state through `SharedInner`:

```rust
#[cfg(not(feature = "threadsafe"))]
type SharedInner = Rc<RefCell<H5FileInner>>;
#[cfg(feature = "threadsafe")]
type SharedInner = Arc<Mutex<H5FileInner>>;        // file.rs:36-39
```

Every public operation goes through `borrow_inner_mut()`, which under
`threadsafe` takes the **single** `Mutex` guarding the entire
`H5FileInner` (and therefore the whole `Hdf5Writer`). So with the
`threadsafe` feature:

- `H5File`/`H5Dataset` become `Send + Sync` (which is what callers such as
  `mdfr`'s rayon `par_iter` over channels require), **but**
- every write — create, `write_raw`, `write_chunk`, compression, the
  positioned `write_at`, index updates — serializes on that one lock.

Net effect: the API is thread-*safe* but has **zero write concurrency**.
`mdfr` pays the `Arc<Mutex>` cost and the `Send + Sync` bound yet sees no
parallel speedup, because the expensive part (filter/compression + the
positioned write of each chunk) runs under the global lock.

Goal: under `threadsafe`, let independent operations — in particular
compressing and writing chunks of **different** datasets — proceed
concurrently, while keeping the single-threaded (`Rc<RefCell>`) path at
its current zero-overhead.

## 2. Where concurrency actually pays off

The `mdfr` export pattern is `par_iter` over channels; each task creates
one dataset and writes it. Two observations shape the design:

1. **Compression dominates.** `filter::apply_filters` (deflate/zstd/lz4,
   shuffle, bitshuffle) is CPU-bound and currently runs inside
   `write_chunk_fixed_array` **under the global lock**
   (`writer.rs` ~2989). Moving compression *outside* any lock is the
   single biggest win.
2. **Creation is inherently partly serial.** Creating a dataset mutates
   the shared `datasets: Vec` and the allocator. Even with per-dataset
   locks, the registry insert + the space bump must be serialized (but
   briefly). The parallel win is in the *write*, not the *create*.

So the target is: compress with no lock held; hold a lock only for the
short allocate + positioned write + index-record steps; and let those
short critical sections be per-dataset / per-allocator rather than one
global section.

## 3. Current serialization points

All of these sit behind the one `Mutex<H5FileInner>`:

| # | Component | Today | Concurrency obstacle |
|---|-----------|-------|----------------------|
| 1 | `FileAllocator` (`allocator.rs`) | `allocate(&mut self)` bumps a single `eof` | trivially atomicizable |
| 2 | `FileHandle` (`file_handle.rs`) | single `BufWriter<File>`; `write_at(&mut self,…)` seeks+writes through the buffer | **inherently sequential** — needs positioned `pwrite` (`FileExt::write_all_at(&File)`) and removal of the shared `BufWriter` |
| 3 | `datasets: Vec`, `groups: Vec`, link/registry state | mutated on create, read on write | needs an append-only / `RwLock` registry so writes don't block on create |
| 4 | Per-dataset state — chunk index (FA/EA/btree2), `append_buffer`, counters | mutated per chunk write | needs a per-dataset lock so two datasets don't contend |

## 4. Target architecture

### 4.1 `FileHandle` → positioned writes

The hard dependency. The current `BufWriter<File>` is a single cursor;
concurrent `write_at` is impossible through it. Replace the write path
with positioned writes:

- Keep the `File` (and the lock policy / `lock_held` bookkeeping).
- Replace `write_at(&mut self, off, data)` with a positioned write that
  takes `&self`: `std::os::unix::fs::FileExt::write_all_at` /
  `std::os::windows::fs::FileExt::seek_write`. `pwrite` at distinct,
  non-overlapping offsets is safe to call concurrently on a shared
  `&File`.
- **Buffering.** The current `BufWriter` coalesces small writes. With
  positioned writes there is no shared cursor to buffer. Options
  (decision needed — see §8):
  - (a) Drop buffering; rely on the OS page cache. Simplest; each
    `write_at` is one `pwrite`. Most chunk writes are already
    chunk-sized, so the loss is mainly for many tiny metadata writes at
    finalize (single-threaded anyway).
  - (b) Per-call buffering only where it matters (e.g. object-header
    assembly already builds a full `Vec` then writes once).
  - Recommended: (a), measure, add (b) only if finalize regresses.
- `sync_all`/flush become `&self` (fsync the `File`).

This makes `FileHandle` a `Sync` positioned-write sink.

### 4.2 `FileAllocator` → atomic bump

`allocate` is a single aligned bump of `eof`. Replace the field with an
`AtomicU64` and bump with a compare-and-swap loop (alignment makes a
plain `fetch_add` insufficient, so CAS):

```text
loop {
    let cur = eof.load(Acquire);
    let aligned = align_up(cur, alignment);
    let next = aligned + size;
    if eof.compare_exchange_weak(cur, next, AcqRel, Acquire).is_ok() {
        return aligned;
    }
}
```

`allocate` then takes `&self` and needs no lock. `set_eof`/`eof` become
atomic load/store. This removes the allocator from every critical
section.

### 4.3 `Hdf5Writer` interior mutability

Split the monolithic `&mut self` writer into fields with independent
interior mutability:

- `allocator: FileAllocator` — atomic, shared by `&self` (§4.2).
- `handle: FileHandle` — positioned writes by `&self` (§4.1).
- **Dataset registry**: replace `datasets: Vec<DatasetInfo>` with an
  append-only structure that hands out a stable index and lets a writer
  reach one dataset without locking the others. Concretely a
  `RwLock<Vec<Arc<DatasetSlot>>>` where:
  - the `RwLock` is taken in *write* mode only to push a new slot
    (create), in *read* mode to clone the `Arc` for an existing index;
  - each `DatasetSlot` holds the per-dataset metadata behind its own
    `Mutex` (chunk index, append buffer, counters), so two datasets
    never contend.
- **Group / link / root-attr registries**: same `RwLock<…>` treatment;
  these are touched on create/metadata, rarely on the hot path.
- **Compression moves out of the lock**: `write_chunk_fixed_array` etc.
  split into `(read dataset params) → compress (no lock) → lock slot,
  allocate, write_at, record index`.

The streaming write path (`write_chunk*`, `append`, `write_raw` on a
chunked dataset) becomes: lock the *one* target `DatasetSlot`, do the
short allocate+write+record, unlock. Different datasets run fully in
parallel; same-dataset writes serialize (correct — a single chunk index
is not concurrently mutable).

### 4.4 `SharedInner` — drop the outer `Mutex`

Under `threadsafe`, `H5FileInner` no longer needs an outer `Mutex` once
the writer is internally fine-grained:

- `threadsafe`: `SharedInner = Arc<H5FileInner>` where `H5FileInner`
  holds the fine-grained writer (all interior locks). `borrow_inner` /
  `borrow_inner_mut` return `&H5FileInner` (no guard); callers use
  `&self` writer methods.
- non-`threadsafe`: keep `Rc<RefCell<H5FileInner>>` exactly as today —
  zero overhead, single-thread. The writer's `&self` methods work under
  `RefCell` too (a `&self` method that takes its own interior locks is a
  no-op contention under single thread; for the `Rc` path the interior
  "locks" can be the same atomics/`Mutex` — uncontended and cheap — or a
  cfg-selected `RefCell`-friendly variant). **Reconciling these two
  paths is the main design tension** (see §8).

### 4.5 `close` / `Drop` finalize gate

`close()` and `Drop` both finalize. Today the outer `Mutex` + the
`Closed` sentinel serialize that. With the outer lock gone, finalize is
a one-time terminal transition and must run exactly once even if several
handles race to drop. Use a single `Once`/atomic `finalized` flag on the
writer; the first to set it runs `finalize()`, others no-op. This also
interacts with the [Strong state transitions] rule already applied to
`Drop` in commit 5c3fdb1.

### 4.6 Lock ordering (deadlock avoidance)

Establish and document a strict acquire order; never hold two in the
other order:

```
registry RwLock  →  one DatasetSlot Mutex   (allocator is lock-free)
```

The hot path takes at most **one** `DatasetSlot` lock at a time and the
lock-free allocator, so it cannot deadlock. Finalize takes the registry
read lock then each slot in index order. No path takes two slot locks.

## 5. Rc vs Arc reconciliation (the crux)

Two viable strategies; pick one in review:

- **A. One internally-concurrent writer for both features.** The writer
  always uses atomics + `RwLock`/`Mutex` internally. The `Rc<RefCell>`
  outer wrapper stays for the non-threadsafe API surface but the inner
  locks are uncontended (cheap). Pro: one code path, no `cfg` sprinkled
  through the writer. Con: the single-thread path pays uncontended-lock
  overhead (small, but non-zero — violates the "zero-overhead" promise
  in the `file.rs` doc comment).
- **B. `cfg`-selected interior types.** A thin abstraction
  (`type Shared<T> = RefCell<T>` vs `RwLock<T>`, `type Slot<T> =
  RefCell<T>` vs `Mutex<T>`, allocator = `Cell<u64>` vs `AtomicU64`) so
  the single-thread build keeps true zero overhead and the threadsafe
  build gets real locks. Pro: preserves the zero-overhead promise. Con:
  two behaviors to test; the abstraction must be carefully `Send/Sync`
  correct only under `threadsafe`.

Recommendation: **B**, to keep the single-threaded promise the README
and `file.rs` make, accepting the cost of a small interior-mutability
abstraction.

## 6. Staging

Each stage is independently committed and verified; the build and the
existing single-threaded behavior stay green throughout.

- **Stage 1 — I/O foundation.** Atomic `FileAllocator` (§4.2) +
  positioned-write `FileHandle` (§4.1). Public API unchanged; the writer
  still holds the outer lock. Verify: full suite + a concurrency stress
  test that hammers positioned writes at distinct offsets.
- **Stage 2 — concurrent chunk path.** Compression outside the lock +
  per-`DatasetSlot` locking for `write_chunk*` / `append` / chunked
  `write_raw` (§4.3). Outer lock still present for create/metadata.
  Verify: parallel multi-dataset write stress test (deflate/zstd/lz4) +
  h5py read-back for value correctness; this is the stage that delivers
  the `mdfr` speedup.
- **Stage 3 — remove the outer lock.** Fine-grained registries (§4.3),
  drop the outer `Mutex` (§4.4), the finalize gate (§4.5), convert the
  remaining `&mut self` API call sites. Verify: full concurrency stress
  + `loom` model of the create/write/close interleavings if feasible.

## 7. Test strategy

- **Existing suite** (556 tests) must stay green at every stage — it
  guards single-threaded correctness.
- **Concurrency stress** (new, `threadsafe`-gated): N threads each create
  + compress + write their own dataset; reopen and assert every
  dataset's bytes; run under repeated iterations to shake out races.
- **h5py cross-validation**: the parallel-written compressed datasets are
  read back by h5py for value + filter correctness (extends the existing
  `tests/h5py_cross_validation.rs` pattern).
- **`loom`** (Stage 3, if practical): model the allocator CAS, the
  finalize gate, and registry-vs-slot ordering.
- **TSan/Miri**: not generally usable here (real file I/O), but the
  lock-free allocator alone can be Miri-checked in isolation.

## 8. Open questions for sign-off

1. **Buffering (§4.1):** drop `BufWriter` entirely (option a) or keep a
   narrow buffered path for finalize metadata (option b)?
2. **Rc/Arc strategy (§5):** abstraction-based zero-overhead single-thread
   (B, recommended) vs one always-concurrent writer (A)?
3. **Scope of `&self` conversion:** convert the entire writer in Stage 3,
   or leave seldom-used cold paths (`open_append` resume, hard links,
   SWMR `flush`) under a coarse lock and only fine-grain the hot write
   path? The latter bounds risk; the former is the "full" refactor as
   literally stated.
4. **`loom` dependency:** acceptable to add as a `dev-dependency` for
   Stage 3 modelling?

Once 1–4 are decided, Stage 1 can begin.
