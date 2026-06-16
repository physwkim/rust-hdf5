# Changelog

## 0.2.22

### Added

- `DatasetBuilder::datatype(dt)` — override the stored on-disk element
  datatype independently of the in-memory type parameter `T`. Combined with
  the new N-bit constructor, this lets a dataset store a reduced-precision
  fixed-point type while writing from a wider Rust integer.
- `FilterPipeline::nbit(&dt, d_nelmts)` — construct the N-bit filter
  (`H5Z_FILTER_NBIT`, id 5) for an atomic numeric datatype. The `cd_values`
  tree (`[nparms, need_not_compress, d_nelmts, NBIT_ATOMIC, size, order,
  precision, offset]`) mirrors libhdf5's `H5Z__set_local_nbit`, so packed
  datasets are readable by h5py / libhdf5. Verified byte-compatible with
  h5py for 10- and 12-bit `u16` data, single- and multi-chunk.
- 1-D (and N-D) array attributes. `AttrBuilder::shape` now records the
  attribute dimensions instead of discarding them, accepting either `()`
  (scalar) or an array/slice/`Vec` of dimension sizes via the new
  `AttrShape` trait (re-exported at the crate root). `H5Attribute::write_array`
  writes the values (length must equal the product of the shape), and
  `AttributeMessage::array_numeric` builds the underlying simple-dataspace
  attribute. The scalar writers (`write_scalar`, `write_numeric`) are
  unchanged. Verified against h5py: a written `int32` array attribute reads
  back with the correct shape, dtype, and values.
- `H5File::dataset_writer(name)` — reopen an existing dataset by name in
  write mode (the write-mode counterpart of `dataset()`), reconstructing the
  same handle `new_dataset().create()` returns so attributes can be attached
  or chunks appended without keeping the original handle. Returns
  `Hdf5Error::NotFound` for an unknown name.
- `SwmrFileWriter::set_dataset_attr_array(ds_index, name, dims, values)` —
  set a numeric array attribute on a dataset (the SWMR counterpart of
  `H5Attribute::write_array`). As with the other SWMR dataset attributes, it
  must be called before `start_swmr`; resolve a dataset path to its index via
  `dataset_index(name)`.

## 0.2.21

### Fixed

- The bitshuffle filter (32008, including the LZ4 "BSLZ4" mode) is now
  byte-for-byte compatible with the canonical bitshuffle HDF5 filter
  (kiyo-masui/bitshuffle), so files written by this crate are readable by
  h5py / libhdf5 and vice versa. Two divergences were corrected:
  - **Bit order.** The bit transpose used an MSB-first convention; the
    canonical filter is LSB-first in both dimensions. Files written before
    this fix were unreadable by h5py/libhdf5, and canonical files could not
    be read back by this crate.
  - **Block framing.** The trailing elements of a chunk are now split into a
    final transposed block rounded down to a multiple of 8 elements plus a
    raw `n_elems % 8` leftover, matching `bshuf_blocked_wrap_fun`. The
    previous code copied the entire `n_elems % block_size` tail verbatim, so
    any chunk whose element count was not a multiple of the block size (the
    common case) was framed incompatibly. The non-canonical per-block
    "store uncompressed" fallback was removed (every block is LZ4; only the
    `n_elems % 8` leftover is raw), and no-compression decoding now honors
    the block size carried in `cd_values[3]` instead of always recomputing
    the default.

  Validated byte-for-byte against the upstream C reference and end-to-end
  with h5py + hdf5plugin in both directions.

### Added

- `FilterPipeline::bshuf(element_size)` and
  `FilterPipeline::bshuf_lz4(element_size)` — construct a bitshuffle filter
  pipeline (bit transpose only, or bit transpose + LZ4), mirroring the
  existing `lz4()` / `zstd()` constructors.

## 0.2.20

### Added

- `H5Attribute::datatype()` — return the parsed `DatatypeMessage` (class,
  signedness, byte order, bit precision) of a read-mode attribute, mirroring
  `H5Dataset::datatype()` from 0.2.18. A generic attribute→metadata mapper can
  now recover the exact stored type instead of guessing from the byte width
  (which cannot tell `u8` from `i8`, or `i32` from `f32`). Errors for a
  write-mode handle, which carries no decoded attribute message.

## 0.2.19

### Added

- `H5Dataset::write_chunk_raw(chunk_idx, data, filter_mask)` — the HDF5
  "direct chunk write" (`H5Dwrite_chunk` / `H5DOwrite_chunk`): store an
  already-filtered chunk verbatim without re-running the dataset's filter
  pipeline, recording a per-chunk `filter_mask`. Bit *i* set means filter
  *i* was not applied to this chunk and must be skipped on read. Unblocks
  handing already-compressed frames (e.g. from a codec plugin) straight to
  disk. Supported for extensible-array (one unlimited dimension) and
  fixed-array (all dimensions bounded) indexes; rejected for unfiltered and
  v2-B-tree datasets, which have no slot for a stored size or mask.
- `filter::reverse_filters_masked(pipeline, data, filter_mask)` and
  `DataLayoutMessage`'s new `SingleChunkFilter` — public format-layer
  building blocks for per-chunk masking.

### Fixed

- The reader now honors the per-chunk `filter_mask` on every chunked index
  path — extensible array, fixed array, version-1 and version-2 B-trees,
  and the single-chunk index — so a chunk written with a filter skipped
  round-trips through this crate's own reader, not only libhdf5/h5py. A
  single-chunk filtered layout's inline on-disk size and mask are now
  decoded (previously discarded), so such a chunk is read at its exact
  stored size instead of an over-read-and-inflate guess.
- A filtered chunk whose stored size does not fit the chunk-index size
  field now errors (matching libhdf5's `H5D_CHUNK_ENCODE_SIZE_CHECK`)
  instead of silently truncating the recorded size.

## 0.2.18

### Added

- `H5Dataset::datatype()` — return the element type a dataset was
  written with, as the parsed `DatatypeMessage` (class, signedness,
  byte order, and bit precision), in read mode. Callers that must
  reconstruct the exact stored type — for example to map it to a NumPy
  or Arrow dtype — no longer have to infer it from `element_size`,
  which cannot tell `u8` from `i8` (both 1 byte) or `i32` from `f32`
  (both 4 bytes).
- `DatatypeMessage` and `ByteOrder` are now re-exported at the crate
  root, alongside the existing `FilterPipeline` re-export.

## 0.2.17

### Added

- `SwmrFileWriter::create_group`, `set_group_attr_string`, and
  `set_group_attr_numeric` — build a nested group layout (e.g. the
  NeXus `/entry` → `/entry/data` tree) and tag groups, or the root
  group, with attributes such as `NX_class`. A group created before
  `start_swmr` is visible to readers for the whole streaming window;
  one created after is committed at `close`.
- `SwmrFileWriter::write_dataset` and `write_string_dataset` — write
  fixed-shape, scalar (`dims = &[]`), and variable-length-string
  datasets for the metadata that surrounds an image stream.
- `SwmrFileWriter::set_dataset_attr_string`, `set_dataset_attr_numeric`,
  `set_dataset_fill_value`, and `assign_dataset_to_group` — dataset
  attributes (`units`, `signal`, …), streaming fill values, and
  placement of a dataset inside a group.
- `SwmrFileWriter::open_append` / `open_append_with_locking` and
  `dataset_index` — reopen a cleanly-closed SWMR file and resume
  streaming into its existing datasets. Appending to a multi-frame-chunk
  dataset (`chunk[0] > 1`) after reopen is rejected with a clear error,
  because its final partial band was zero-padded at the original close.
- `SwmrFileReader::read_slice` and `read_slice_raw` — hyperslab reads.
  For a streaming dataset only the chunks the slice overlaps are read,
  so a live viewer can fetch the latest frame without re-reading the
  whole stream.
- `SwmrFileReader::read_vlen_strings`, `dataset_element_size`,
  `group_paths`, `has_group`, `dataset_attr_names`,
  `dataset_attr_string`, `group_attr_names`, and `group_attr_string` —
  inspect groups, datasets, and string attributes through a SWMR reader.

## 0.2.16

### Added

- `SwmrFileWriter::create_hard_link(parent_group_path, link_name,
  target_path)` — create a hard link in a SWMR file through the public
  API. A link created **before** `start_swmr` is committed by
  `start_swmr` and is visible to SWMR readers for the whole streaming
  window; a link created **after** `start_swmr` is committed by `close`
  and is not visible during the live SWMR window.

### Fixed

- Closing a SWMR file now commits structural changes made after
  `start_swmr` (such as a hard link) via a full re-finalize of all
  object headers. Previously the SWMR close path only rewrote dataset
  headers in place: creating a hard link after `start_swmr` grew its
  target's object header past the in-place slot, so `close` failed with
  `dataset header grew ... cannot rewrite in place` and left the file
  marked SWMR-dirty (the clean-close superblock was never written).

## 0.2.15

### Added

- `H5Dataset::set_extent(&[dims])` — set the logical extent of a chunked
  dataset, growing **or shrinking** any dimension. Unlike `extend`
  (grow-only), this can reduce a dimension — for example to correct an
  over-extended frame count after a partial multi-frame chunk. Shrinking
  changes the logical dataspace only: data in chunks beyond the new
  extent stays in the file but is no longer visible on read, as with
  libhdf5's `H5Dset_extent`.
- `SwmrFileWriter::create_streaming_dataset_chunked` and
  `create_streaming_dataset_chunked_compressed` — streaming datasets
  with full control over the chunk shape, including the frame axis.
  `chunk[0]` sets the number of frames per chunk (the NDFileHDF5
  `nFramesChunks` control); `chunk[1..]` sets the per-frame tile shape
  (`nRowChunks` / `nColChunks`). `append_frame` buffers whole frames
  until a chunk band fills and writes the final partial band
  zero-padded at `close`; the dataset's logical frame count always
  equals the exact number of frames appended, so a partial last chunk
  never over-extends it.

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
