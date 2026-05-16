# Changelog

## 0.2.14

### Added

- `H5Group::link(link_name, target_path)` — create a hard link: an
  additional name for an existing dataset or group. No data is
  copied; the link and its target share one object header, and an
  Object Reference Count message records the shared count, exactly
  as h5py / libhdf5 hard links do. This is the NeXus-style way to
  expose a dataset at a second canonical location (such as
  `/entry/data/data`) without duplicating it.
- `SwmrFileWriter::create_streaming_dataset_tiled` and
  `create_streaming_dataset_tiled_compressed` — streaming datasets
  whose frames are split into fixed-size chunk tiles (an on-disk
  chunk shape of `[1, frame_chunk...]`), the equivalent of an
  area-detector writer's `nRowChunks` / `nColChunks` controls.
  `append_frame` accepts a whole frame and splits it into tiles
  automatically, zero-padding partial edge tiles. The previous
  streaming API always stored one chunk per frame.

### Fixed

- Gated six `deflate`-dependent tests behind the `deflate` feature
  so `--no-default-features` builds and test runs pass.
- Resolved two clippy lints surfaced by newer Rust toolchains:
  `collapsible_match` in the data-layout decoder and
  `manual_checked_ops` in the v1 B-tree chunk reader.

## 0.2.13

### Added

- Read chunked datasets stored under every libhdf5 chunk index:
  Extensible Array (including paged data blocks), Fixed Array
  (including paged and filtered data blocks), version-1 B-tree
  (version-3 data layout), and version-2 B-tree of any depth
  (including filtered records).
- Read dense group links stored in a fractal heap, with
  direct-block checksum verification.
- SZIP / AEC filter: an in-crate codec that is byte-compatible
  with libaec / libhdf5 for both compression and decompression.
- N-bit and Scale-offset filters, with element-exact reads and
  post-filter datatype conversion for chunked datasets.
- Decode version-1 (as well as version-2) filter pipeline
  messages.
- Fill-value API: `set_dataset_fill_value`, with the version-3
  on-disk fill-value message layout. Unwritten and unallocated
  regions read back as the declared fill value.
- Group attributes and a group/root attribute API; read group
  and root attributes from legacy (version-0/1 superblock)
  files, including variable-length string attributes.
- Sub-frame chunking (chunks smaller than one frame).
- Compressed SWMR streaming datasets.
- Route multi-unlimited-dimension datasets to the version-2
  B-tree chunk index; route fixed-shape chunked datasets to the
  Fixed Array index; write filtered Fixed Array chunk indexes.
- Enumerate groups from link records rather than dataset path
  prefixes, so attribute-only and subgroup-only groups are
  discovered.

### Fixed

- Fletcher-32 filter trailer endianness (`UINT32ENCODE` is
  little-endian).
- Extensible Array, Fixed Array, and version-2 B-tree on-disk
  byte layouts now match libhdf5; paged Extensible Array
  page-init bitmap indexing corrected.
- Version-3 fill-value message corrected to the real on-disk
  layout.
- Global-heap index exhaustion handled.
- Group discovery is cycle-safe and tolerant of stale links;
  `open_append` rejects unsupported version-0/1 superblocks
  with a clear error.

### Hardening

- Unified the little-endian integer/address decoders behind
  clamped helpers (`src/format/bytes.rs`) so a short or
  malformed buffer cannot panic.
- Bounded every recursive parser against corrupt or adversarial
  input: v1 B-tree group traversal, group-link recursion,
  datatype nesting, and fractal-heap indirect-block nesting all
  carry depth and/or visited-set guards.
- Dataset and chunk byte-length computation uses saturating
  arithmetic; buffers sized from untrusted file fields are
  allocated with `try_reserve` so a crafted file yields a clean
  error instead of aborting the process.
- Hardened datatype, global-heap, and link-message parsing,
  chunk-index readers, Extensible/Fixed Array and v2 B-tree
  geometry, the Fixed Array writer, and the N-bit decoder
  against panics on malformed input.
- Verified against h5py 3.16 / libhdf5 2.0.0.

## 0.2.12

### Reliability

- `try_acquire` now retries briefly (~100 ms total: 10 attempts × 10 ms)
  when `try_lock_*` returns `WouldBlock`. macOS in particular has been
  observed to surface a stale lock for a short window after the
  previous holder's `close(2)`; a brief retry distinguishes that
  release-pending race from a real long-lived conflict without
  meaningfully slowing the real-conflict path.

### Tests

- Centralized `unique_test_path` helper at `src/file.rs` module scope
  (used by `mod tests`, `mod integration_tests`, `mod h5py_compat_tests`).
  Equivalent helpers added to `src/io/reader.rs::tests` and
  `src/io/writer.rs::tests::swmr_writer_append_frames`. All
  unit/integration test paths now embed PID + atomic counter, so
  concurrent cargo invocations and kernel-side flock races cannot
  collide. Fixes intermittent CI failures of
  `dataset::tests::type_mismatch_element_size` and
  `file::integration_tests::append_mode`.

## 0.2.11

### Tests

- `dataset::tests` and `file::tests` `temp_path` helpers now produce
  per-call unique paths (PID + atomic counter). Fixes intermittent
  CI flakiness on macOS where a previous holder's `flock` release
  was not yet visible when the next opener tried to acquire its
  shared lock — surfaced by `dataset::tests::write_slice_2d`
  intermittently failing with
  `WouldBlock: unable to lock file: another process holds a
  conflicting lock`.

## 0.2.10

### Documentation

- Document Windows lock semantics: `LockFileEx` is mandatory, so
  `FileLocking::Disabled` and `FileLocking::BestEffort` only control
  whether *we* try to acquire a lock — they cannot bypass an
  exclusive lock another handle already holds (the HDF5 C library
  has the same limitation on Windows).

### Tests

- Two integration tests rely on advisory-lock semantics that don't
  exist on Windows. Gated with `#[cfg(unix)]`:
  `best_effort_does_not_error_on_conflict`,
  `options_locking_disabled_bypasses_real_lock` (split out from
  `options_locking_overrides_env`). The `Enabled`-policy half of
  the original test runs cross-platform as
  `options_locking_overrides_env_enabled_blocks`.

## 0.2.9

### Bug Fixes

- Windows: `SwmrFileWriter::start_swmr` no longer attempts to downgrade
  the writer's exclusive lock to shared. Windows' `LockFileEx` is a
  mandatory range lock, and a same-handle unlock-then-shared-relock
  left subsequent `WriteFile` calls failing with
  `ERROR_LOCK_VIOLATION` (33). The writer now releases its lock
  entirely when SWMR mode starts, matching the HDF5 C library — which
  also relies on the SWMR file-format sentinel rather than OS locks
  during streaming. Trade-off: a second writer attaching to a
  streaming SWMR file is no longer blocked by an OS lock; the SWMR
  protocol's single-writer guarantee is the caller's responsibility.

### Internal

- CI: `cargo fmt --all -- --check` now passes (0.2.8 introduced
  unformatted lines).

## 0.2.8

### Added

- OS-level advisory file locking, mirroring the HDF5 C library:
  - Read opens take a shared lock; write opens (`create` / `open_rw`)
    take an exclusive lock.
  - `SwmrFileWriter::start_swmr` downgrades the exclusive lock to shared
    so concurrent `SwmrFileReader`s can attach while still blocking
    other writers.
  - Honors the `HDF5_USE_FILE_LOCKING` environment variable
    (`TRUE` / `FALSE` / `BEST_EFFORT`).
  - New `H5File::options()` builder with `.locking()`, `.no_locking()`,
    and `.best_effort_locking()` for explicit per-open control.
  - `SwmrFileWriter::create_with_locking` and
    `SwmrFileReader::open_with_locking` for explicit SWMR control.
  - Cross-platform: Unix (`flock` / `fcntl`) and Windows (`LockFileEx`)
    via `std::fs::File::lock` (Rust 1.89+).
- `FileLocking` and `LockMode` types re-exported at the crate root.

### Changed

- MSRV raised to 1.89 (uses stable `File::lock` / `File::try_lock` /
  `File::unlock`).

### Internal correctness

- `H5File::create` opens the file without `O_TRUNC`, acquires the
  exclusive lock, and only then calls `set_len(0)`. A pre-release
  review caught that the previous order would destroy an existing
  file's contents when the lock attempt lost a race, even though
  `create()` returned an error.
- `MmapFileHandle` now retains the underlying `File` so its shared
  lock persists for the lifetime of the mmap.

## 0.2.7

### Added

- `create_appendable_vlen_dataset()` + `append_vlen_strings()` for
  incremental vlen string writes with chunked storage and optional
  compression. Each append creates a new GCOL; partial chunks are
  buffered automatically.
- `delete_dataset(name)` and `delete_group(name)` for soft-deleting
  datasets and groups (excluded from file on close, space not reclaimed).
- `open_rw` now reconstructs group hierarchy from existing dataset paths,
  enabling `delete_group` → `create_group` → write workflows.

## 0.2.6

### Performance

- Fix O(n²) vlen string read performance. Reading 24k+ strings previously
  took ~50s due to cloning the entire GlobalHeapCollection per element and
  using linear search for object lookup. Now uses cached reference with
  HashMap index for O(1) access — same workload completes in <1s.

### Bug Fixes

- Harden chunked reader against corrupt/truncated files:
  - Validate chunk addresses and sizes against file bounds before reading.
  - Skip chunks where decompression fails instead of using raw compressed
    bytes as data.
  - Validate GCOL signature and collection_size before reading global heap.
  - Add 64MB sanity limit on LZ4 decompressed size to prevent OOM.

## 0.2.5

### Bug Fixes

- Fix `open_rw` file corruption when modifying attributes without changing
  datasets. Three issues corrected:
  - Unmodified dataset links pointed to address 0 instead of preserving
    the original object header address.
  - `flush_dataset` was called on all chunked datasets including unmodified
    ones, overwriting valid EA index structures with incomplete in-memory
    copies.
  - Root group attributes were lost because `open_append` did not load
    existing attributes from the file.
- `set_attr_string` now replaces existing attributes with the same name
  instead of creating duplicates.

## 0.2.4

### Added

- Add `write_vlen_strings_compressed()` API for writing chunked, compressed
  variable-length string datasets. Accepts a `FilterPipeline` parameter
  supporting deflate, zstd, or any custom filter combination.
- Re-export `FilterPipeline` from crate root for ergonomic usage.

### Bug Fixes

- Remove 64KB hard limit on global heap collection reads for vlen strings.
  Previously, collections larger than 64KB were truncated, causing decode
  failures on files with many or large variable-length strings. Now reads
  the actual collection size from the GCOL header.

## 0.2.3

### Added

- Add `H5Dataset::append` for incrementally appending data along the first
  dimension of chunked datasets. Supports arbitrary `chunk_dims[0]` with
  internal buffering of partial chunks (flushed automatically on close).

## 0.2.2

### Bug Fixes

- Fix vlen string h5py/HDF5 C library incompatibility. Three issues corrected:
  - Vlen references were missing the 4-byte `sequence_length` prefix
    (wrote `addr+index` instead of `seq_len+addr+index`).
  - Global heap collection size was below the HDF5 minimum of 4096 bytes
    (`H5HG_MINALLOC`), causing "global heap size is too small" errors.
  - Free-space marker size was miscalculated (off by 16 bytes).
- Files written by rust-hdf5 are now fully readable by h5py and h5dump.

## 0.2.1

### Bug Fixes

- Fix `write_vlen_strings` not assigning datasets to their parent group when the
  name contains a path separator (e.g., `"nodes/id"`). Previously, such datasets
  were incorrectly linked at the root level instead of inside the target group.

### Added

- Add `H5Group::write_vlen_strings` method for writing variable-length string
  datasets directly within a group.

## 0.2.0

- Add Blosc sub-codec support (BloscLZ, LZ4HC, Snappy, Zlib, Zstd)
- Merge workspace into single `rust-hdf5` crate for crates.io publishing
- Add Zstandard (zstd) filter support via pure Rust
- Add pure Rust SZIP (AEC) compress/decompress
- Add custom filter pipeline support to DatasetBuilder

## 0.1.0

- Initial release
