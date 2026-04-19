# Changelog

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
