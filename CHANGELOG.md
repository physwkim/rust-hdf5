# Changelog

## 0.2.4

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
