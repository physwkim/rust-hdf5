# Changelog

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
