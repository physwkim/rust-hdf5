# Changelog

## Unreleased

### Added

- `write_slice` now works on chunked datasets, including compressed ones
  (issue #2). It was previously rejected with "write_slice is only for
  contiguous datasets", so updating one row of an appendable dataset meant
  rebuilding and rewriting the whole thing — O(dataset) memory and I/O for an
  O(row) change.

  The selection is decomposed onto the chunk grid and only the intersecting
  chunks are touched. A chunk the selection covers entirely is written
  straight from the caller's buffer; a partially covered chunk is read,
  patched and written back (libhdf5 makes the same distinction in
  `H5D__chunk_lock`'s `relax` flag). Regions of a chunk that no write has
  reached hold the dataset's fill value. All three chunk index types
  (extensible array, fixed array, v2 B-tree) are supported.

### Fixed

- Rewriting a chunk no longer leaks its old file block. Every chunk write
  previously allocated fresh space and left the previous block stranded, so
  repeatedly rewriting the same chunk grew the file without bound — including
  through `append`. A chunk is now placed by consulting its index entry first:
  an unfiltered chunk (whose size never changes) is rewritten in place, and a
  filtered chunk that no longer fits its old block is relocated with the old
  block released to the allocator for reuse. This mirrors libhdf5's
  `H5D__chunk_file_alloc` / `H5MF_xfree`. Under SWMR the release is suppressed,
  since a reader may still be following the old address.

- `FileAllocator` gained a free list, so released blocks are reused before the
  file grows. Best fit with merging of adjacent blocks; like libhdf5's default
  strategy the list is not persisted to disk, so a block released but unused at
  close remains slack in the file.

## 0.3.2

### Fixed

- File creation no longer issues an `ftruncate`-to-0 on a brand-new empty
  file. On ext4 (with the default `auto_da_alloc`), that truncate arms
  replace-via-truncate protection on the inode, which turns the final
  `close(2)` into an implicit writeback of everything written since (~330 ms
  for a 512³ f32 dataset) — silently defeating `close_no_sync`, whose skipped
  `fsync`s were then paid inside `close(2)` anyway. The truncate now runs only
  when the file already has content, which is the only case it exists for
  (destroying prior contents after the create lock is validated). The
  lock-before-truncate ordering is unchanged, and creating over an existing
  non-empty file still truncates it. Verified on ext4: `close(2)` after a
  `close_no_sync` write went from ~326 ms to ~0 ms.

## 0.3.1

### Added

- `H5File::close_no_sync` (and the underlying `Hdf5Writer::close_no_sync`) —
  a close that finalizes the file (writes all object headers and the
  superblock, so on return it is a complete, valid HDF5 file readable by any
  process) but skips every `fsync` in the finalize path. The bytes are handed
  to the OS but are not guaranteed durable against power loss or an OS crash
  until the OS flushes its page cache. This trades durability for speed for
  bulk output that can be regenerated; the `fsync`s otherwise dominate close
  latency. `close` remains durable, and dropping a file without calling either
  finalizes durably.

  The finalize path has two durability points — the final `sync_all` and the
  per-indexed-dataset `sync_data` inside `flush_dataset` — and both are skipped
  by `close_no_sync`, so a file with many chunked datasets sees no `fsync` at
  all. The public `flush_dataset`, SWMR finalize/flush paths, and
  `H5Dataset::flush` are unchanged and still sync.

## 0.3.0

Parallel chunk I/O under the `parallel` feature, a private thread pool that
bounds the CPU footprint, and a correctness fix in the parallel filter path.

### Added

- Parallel chunk I/O when the crate is built with the `parallel` feature
  (rayon), doing intra-process chunk-level work that libhdf5's default
  (non-MPI) path does not:
  - **Reads.** A single owner, `read_and_decompress_chunks`, drives every
    chunk index type (extensible-array, fixed-array, B-tree v1, B-tree v2).
    On Unix and Windows — where positioned reads carry their own offset and
    never consult a shared file cursor — the per-chunk disk read and its
    decompression run fused in one parallel pass, overlapping I/O across
    cores. On targets without a positioned-read API, reads stay serial and
    only decompression parallelizes.
  - **Writes.** Full-image chunked writes (`write_raw` on a filtered dataset)
    compress their chunks in parallel for both the extensible-array and
    fixed-array layouts, in bounded windows. B-tree v2 datasets are always
    stored unfiltered, so they remain serial.
  - **Bounded CPU footprint.** All of the crate's rayon work runs on a private
    thread pool sized to **half** the logical cores by default, rather than
    rayon's global pool (which defaults to every core). This keeps a single
    HDF5 read or write from saturating the machine and starving co-running
    processes. Override the thread count with the `RUST_HDF5_IO_THREADS`
    environment variable. The pool never calls `build_global()`, so it does
    not fight the host application for the global pool, and if it cannot be
    built (the OS refuses worker threads) each parallel section falls back to
    serial execution instead of panicking.

### Changed

- **Breaking (`parallel` feature).** `apply_filters_parallel` and
  `reverse_filters_parallel` in `format::messages::filter` now return
  `FormatResult<Vec<Vec<u8>>>` instead of `Vec<Vec<u8>>`. They propagate a
  per-chunk filter error instead of silently substituting the raw bytes, so
  callers must handle the `Result`. This is the only public API change and is
  the reason for the minor version bump.

### Fixed

- Parallel chunk compression no longer swallows a filter error into raw bytes
  recorded under `filter_mask = 0`. Previously a compression failure (an
  szip/bzip2/unknown/scale-offset filter, or a codec whose feature was not
  compiled in) stored the uncompressed bytes while marking the chunk as fully
  filtered, so the reader would try to reverse-filter unfiltered data and
  corrupt the chunk. Both the extensible-array and fixed-array batch writers
  now route through one error-propagating compressor. deflate never triggered
  the bug, so no existing deflate file is affected.

### Performance

- Multi-dimensional whole-image reads scatter each chunk into the output by
  contiguous last-axis run (`copy_from_slice`) instead of element by element,
  matching the write path.
- Measured on a 12-core machine (throughput, `--features parallel` vs serial),
  1000-chunk datasets:
  - compressed read: +264% at half cores (617 MiB/s -> 2.18 GiB/s), ~5x at
    full cores;
  - compressed `write_raw`, extensible array: +297% at half cores
    (31 -> 121 MiB/s);
  - compressed `write_raw`, fixed array: +340% at half cores
    (32 -> 141 MiB/s).

## 0.2.28

### Added

- Zero-allocation reads into a caller-provided buffer, the Rust analogue of
  HDF5 C `H5Dread(..., buf)` and h5py `Dataset.read_direct`:
  - `H5Dataset::read_raw_into::<T>(out: &mut [T])` — read the whole dataset
    image into `out`.
  - `H5Dataset::read_slice_into::<T>(out: &mut [T], starts, counts)` — read a
    hyperslab into `out`.

  The data is decoded straight into `out` with no intermediate `Vec`, so a
  pinned / page-locked host buffer can be filled in one pass and DMA'd to a GPU
  without the staging copy a `read_raw` + copy-into-pinned would incur. Both
  validate that `T::element_size()` matches the dataset's on-disk element size
  (`TypeMismatch` otherwise) and that the buffer length is exactly the read
  size. They are the zero-copy counterparts of `read_raw` / `read_slice` and
  share the same buffer-filling core, so every layout and chunk index type
  (contiguous, compact, btree-v1, single-chunk, fixed-array, extensible-array,
  btree-v2) is handled identically. Verified against the allocating readers on
  contiguous, multi-chunk, single-chunk, and deflate-compressed layouts.

## 0.2.27

### Added

- N-dimensional array attributes, generalizing the 1-D array setters added in
  0.2.26 to an arbitrary shape for both numeric and variable-length string
  attributes:
  - `H5File::set_attr_array_numeric_nd` / `set_attr_string_array_nd` (root group)
  - `H5Group::set_attr_array_numeric_nd` / `set_attr_string_array_nd`
  - `H5Attribute::write_string_array` now honors the full `AttrBuilder::shape`
    of any rank (the previous 1-D-only restriction is removed), matching the
    already-N-D `write_array` for numeric attributes.

  `shape` gives the dataspace dimensions and `values` is the row-major data,
  whose length must equal the product of `shape` (an empty `shape` is a scalar);
  a mismatch returns an error instead of truncating. The 1-D
  `set_attr_array_numeric` / `set_attr_string_array` setters are now thin
  wrappers over the N-D forms with `shape = [values.len()]`, so existing callers
  are unaffected. Verified by h5py cross-validation: a 2x3 `i32` and 2x2x2 `f64`
  numeric attribute, and 2x3 / 2x2 / dataset-builder vlen-string attributes, all
  read back with the exact multi-dimensional shape and row-major values.

## 0.2.26

### Added

- 1-D array attributes for files, groups, and the attribute builder, extending
  the scalar attribute setters:
  - `H5File::set_attr_string_array` / `set_attr_array_numeric` (root group)
  - `H5Group::set_attr_string_array` / `set_attr_array_numeric`
  - `H5Attribute::write_string_array` (builder path; requires a 1-D `shape([n])`
    and validates `values.len() == n`)

  String arrays are stored as a `[n]` dataspace of variable-length UTF-8 string
  references backed by one shared global-heap collection, so h5py reads them back
  as a 1-D array of Python `str`; numeric arrays use a `[n]` dataspace of
  little-endian elements and read back as a NumPy array. Verified by h5py
  cross-validation (root + group string/numeric arrays, dataset-builder string
  array).

## 0.2.25

### Added

- The `threadsafe` feature now provides real write concurrency. Built with
  `--features threadsafe`, `H5File` is `Send + Sync` and writer operations —
  create, append, extend, chunk write, and metadata — run concurrently across
  *distinct* datasets on a shared `&H5File` (e.g. a Rayon `par_iter` writing
  one dataset per task). The single-threaded default build keeps its
  `Rc`/`RefCell` zero-overhead path unchanged. Internally this is a lock-free
  `AtomicU64` file-space allocator, positioned `pread`/`pwrite` with no shared
  cursor, a per-dataset-`Mutex` registry behind a read-mostly `RwLock`, and
  compression run outside the per-dataset lock. Verified with loom interleaving
  models and multi-threaded create/append/metadata stress tests plus h5py
  read-back. Concurrent writes to a *single* dataset remain unsupported;
  serialize those externally.
- Runtime compound (record) datatypes: `DatasetBuilder::datatype(...)` overrides
  the element type and `H5Dataset::write_raw_bytes`/`read_raw_bytes` move raw
  element bytes, so a `{id: i32, val: f64}` record round-trips and reads back as
  a structured array in h5py.
- Variable-length byte arrays — a vlen sequence of `u8` — via the
  `VarLenSequence` datatype, with create/append and `read_vlen_bytes`.
- `Group::dataset_writer(...)`, and the vlen-string dataset helpers now return an
  `H5Dataset` so attributes can be attached to those datasets.

### Changed

- Compression is now usable through the natural builder API. Requesting a filter
  (deflate/zstd/`filter_pipeline`) without explicit chunk dimensions auto-chunks
  the dataset as one whole-dataset chunk instead of silently dropping to the
  contiguous path and discarding the filter, and `write_raw`/`write_raw_bytes`
  now accept chunked datasets, scattering the row-major image across the chunk
  grid (edge chunks zero-padded).
- String attributes are written as variable-length UTF-8 (read back as Python
  `str` by h5py) instead of fixed-length strings (which h5py returned as
  `bytes`).

### Fixed

- `H5File::close()` (and `SwmrFileWriter::close()`) finalize exactly once: a
  finalize failure is now reported through the returned `Result` and is no
  longer re-attempted by the writer's `Drop`.
- Concurrent creation of two objects with the same name (under `threadsafe`) now
  reliably fails for all but one creator instead of racing past the
  duplicate-name check and writing an invalid file with two same-named links.
- The writer no longer silently swallows a finalize error on drop; it is now
  reported on stderr (use `close()` to handle it as a `Result`).
- The `threadsafe` feature no longer builds on targets that lack positioned file
  I/O (neither Unix nor Windows), where the seek-based fallback would race the
  shared file cursor; it now fails to compile there with an explanatory message.

## 0.2.24

### Fixed

- Hyperslab slice reads and writes no longer issue one I/O per last-axis
  row. `read_slice`/`write_slice` now coalesce trailing fully-selected
  dimensions into a single contiguous transfer: a `[:, r0:r1, :]` read of a
  `[nproj, nz, nx]` contiguous dataset becomes `nproj` reads instead of
  `nproj * (r1 - r0)` per-row reads, and `[r0:r1, :, :]` becomes a single
  read. The contiguous, compact, and all chunked layouts share one
  contiguous-run primitive (`io::hyperslab::for_each_contiguous_run`), so the
  read and write paths stay symmetric.
- Chunked `read_slice` now reads only the chunks that overlap the selection
  instead of materializing the whole dataset and extracting the sub-region.
  Each chunk∩selection is copied as contiguous last-axis runs, and this
  applies uniformly to single-chunk, fixed-array, extensible-array, and
  v1/v2 B-tree chunk indices (the previous extensible-array-only fast path is
  removed, subsumed by the general path).
- Contiguous slice reads no longer perform a per-call `fstat`; bytes are read
  straight into the caller's pre-sized output buffer.

## 0.2.23

### Added

- `SwmrFileWriter::create_grid_dataset(name, dims, chunk)` — create a
  fixed-shape multi-dimensional grid dataset whose chunks are filled at
  explicit positions, rather than appended along a single unlimited axis
  (`create_streaming_dataset`). This is the AreaDetector "extra dimensions"
  layout: a scan of known size (e.g. `[Na, Nb, H, W]`) filled in odometer
  order. Because the dataspace is fully bounded, it uses the fixed-array
  chunk index — the index libhdf5 requires when there is no unlimited
  dimension; an extensible array would make the file unreadable
  ("didn't find unlimited dimension"). Verified readable by h5py.
- `SwmrFileWriter::write_chunk_at(ds_index, chunk_coords, data)` — write one
  full chunk at explicit chunk-grid coordinates of a grid dataset. Positions
  may be written in any order; unwritten positions read back as fill. Call
  `flush()` to make writes visible to SWMR readers. Rejects wrong-rank or
  out-of-grid coordinates and non-grid (streaming) datasets.

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
