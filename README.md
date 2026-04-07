# hdf5-rs

Pure Rust HDF5 library — no C dependencies.

Read and write HDF5 files with contiguous, chunked, and compressed datasets, hierarchical groups, attributes, SWMR streaming, and hyperslab I/O.

## Why hdf5-rs?

- **Zero C dependencies** — no `libhdf5`, no `h5cc`, no system packages. Works anywhere Rust compiles. (Note: the optional `zstandard` feature links the C `libzstd` via `zstd-sys` and requires a C compiler.)
- **Memory safe** — Rust's type system prevents buffer overflows, use-after-free, and data races. Minimal `unsafe` only for type reinterpretation.
- **Simple API** — fluent builder pattern instead of C-style opaque handles (`h5d_*`, `h5g_*`, ...)
- **Batteries included** — compression codecs (deflate, LZ4, Zstandard) bundled as Rust crates. No plugin system needed.
- **Easy cross-compilation** — all dependencies are pure Rust. No cross-compile toolchain for C libraries required.

## Features

- **Read & write** — create new files, open existing files, append datasets
- **Chunked storage** — extensible array, fixed array, B-tree v2 indices
- **Compression** — deflate (gzip), shuffle, Fletcher-32, LZ4, Zstandard filters
- **Parallel compression** — per-chunk compression/decompression via rayon
- **Groups** — hierarchical group structure with nested object headers
- **Attributes** — string and numeric attributes on datasets and root
- **SWMR** — Single Writer / Multiple Reader streaming protocol
- **Hyperslab I/O** — `read_slice` / `write_slice` for partial N-dimensional access
- **Buffered I/O** — BufWriter/BufReader with automatic mode switching
- **Memory-mapped I/O** — optional zero-copy read-only access via `mmap` feature
- **Thread safety** — optional `threadsafe` feature (`Arc<Mutex>` instead of `Rc<RefCell>`)
- **Legacy format support** — reads v0/v1 superblock and v1 object header files (h5py, HDF5 C library)
- **Variable-length strings** — reads h5py-style vlen string datasets via global heap
- **Compound types** — user-defined struct types and complex numbers (`Complex32`, `Complex64`)

## Quick start

```toml
[dependencies]
hdf5 = { path = "crates/hdf5" }
```

### Write

```rust
use hdf5::H5File;

let file = H5File::create("output.h5")?;
let ds = file.new_dataset::<f64>()
    .shape(&[100, 200])
    .create("matrix")?;
ds.write_raw(&vec![0.0f64; 20_000])?;
file.close()?;
```

### Read

```rust
use hdf5::H5File;

let file = H5File::open("output.h5")?;
let ds = file.dataset("matrix")?;
let data = ds.read_raw::<f64>()?;
assert_eq!(data.len(), 20_000);
```

### Chunked + compressed streaming

```rust
use hdf5::H5File;

let file = H5File::create("stream.h5")?;
let ds = file.new_dataset::<f32>()
    .shape(&[0usize, 256])
    .chunk(&[1, 256])
    .max_shape(&[None, Some(256)])
    .deflate(6)
    .create("sensor")?;

for frame in 0..1000u64 {
    let vals: Vec<f32> = (0..256).map(|i| (frame * 256 + i) as f32).collect();
    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    ds.write_chunk(frame as usize, &raw)?;
}
ds.extend(&[1000, 256])?;
file.close()?;
```

### Groups

```rust
use hdf5::H5File;

let file = H5File::create("groups.h5")?;
let det = file.create_group("detector")?;
let raw = det.create_group("raw")?;

let ds = raw.new_dataset::<u16>()
    .shape(&[64usize, 64])
    .create("image")?;
ds.write_raw(&vec![0u16; 4096])?;
file.close()?;

// Read back
let file = H5File::open("groups.h5")?;
let ds = file.dataset("detector/raw/image")?;
assert_eq!(ds.shape(), vec![64, 64]);
```

### Hyperslab (slice) I/O

```rust
use hdf5::H5File;

let file = H5File::create("slice.h5")?;
let ds = file.new_dataset::<i32>()
    .shape(&[10usize, 10])
    .create("grid")?;
ds.write_raw(&[0i32; 100])?;

// Overwrite a 2x3 sub-region
ds.write_slice(&[2, 3], &[2, 3], &[1i32, 2, 3, 4, 5, 6])?;
file.close()?;

// Read a sub-region
let file = H5File::open("slice.h5")?;
let ds = file.dataset("grid")?;
let region = ds.read_slice::<i32>(&[2, 3], &[2, 3])?;
assert_eq!(region, vec![1, 2, 3, 4, 5, 6]);
```

### Attributes

```rust
use hdf5::{H5File, VarLenUnicode};

let file = H5File::create("attrs.h5")?;
let ds = file.new_dataset::<f32>().shape(&[10]).create("data")?;
ds.write_raw(&[0.0f32; 10])?;

let attr = ds.new_attr::<VarLenUnicode>().shape(()).create("units")?;
attr.write_string("kelvin")?;
file.close()?;

// Read back
let file = H5File::open("attrs.h5")?;
let ds = file.dataset("data")?;
let units = ds.attr("units")?;
assert_eq!(units.read_string()?, "kelvin");
```

### Append mode

```rust
use hdf5::H5File;

// Add datasets to an existing file
let file = H5File::open_rw("existing.h5")?;
let ds = file.new_dataset::<f64>().shape(&[100]).create("new_data")?;
ds.write_raw(&vec![0.0f64; 100])?;
file.close()?;
```

### SWMR streaming

```rust
use hdf5::swmr::{SwmrFileWriter, SwmrFileReader};

// Writer process
let mut writer = SwmrFileWriter::create("swmr.h5")?;
let ds = writer.create_streaming_dataset::<f32>("frames", &[256, 256])?;
writer.start_swmr()?;
writer.append_frame(ds, &vec![0u8; 256 * 256 * 4])?;
writer.flush()?;
writer.close()?;

// Reader process (concurrent)
let mut reader = SwmrFileReader::open("swmr.h5")?;
reader.refresh()?;
let data = reader.read_dataset::<f32>("frames")?;
```

## Crate structure

```
hdf5-rs/
├── crates/
│   ├── hdf5-format/   # On-disk format codec (encode/decode, no I/O)
│   ├── hdf5-io/       # I/O engine (file handles, reader, writer, SWMR)
│   └── hdf5/          # Public API (H5File, H5Dataset, H5Group, H5Attribute)
```

| Crate | Description |
|-------|-------------|
| `hdf5-format` | Pure format codec — superblock, object headers, messages, chunk indices |
| `hdf5-io` | I/O layer — buffered file handles, allocator, reader, writer, SWMR protocol |
| `hdf5` | High-level API — fluent builders, typed read/write, groups, attributes |

## Supported types

| Rust type | HDF5 type |
|-----------|-----------|
| `u8`, `i8` | 8-bit integer |
| `u16`, `i16` | 16-bit integer |
| `u32`, `i32` | 32-bit integer |
| `u64`, `i64` | 64-bit integer |
| `f32` | IEEE 754 single |
| `f64` | IEEE 754 double |
| `HBool` | Boolean (enum over u8) |
| `Complex32` | Compound {re: f32, im: f32} |
| `Complex64` | Compound {re: f64, im: f64} |
| `CompoundType` | User-defined compound |
| `VarLenUnicode` | Variable-length UTF-8 string (read) |

## Compression filters

| Filter | Feature flag | Crate |
|--------|-------------|-------|
| Deflate (gzip) | `deflate` (default) | `flate2` |
| Shuffle | built-in | — |
| Fletcher-32 | built-in | — |
| LZ4 | `lz4` | `lz4_flex` |
| Zstandard | `zstandard` | `zstd` |

```toml
# Enable LZ4 and Zstandard
[dependencies]
hdf5-format = { path = "crates/hdf5-format", features = ["lz4", "zstandard"] }
```

## Feature flags

| Feature | Crate | Description |
|---------|-------|-------------|
| `deflate` | `hdf5-format` | Deflate compression (default) |
| `lz4` | `hdf5-format` | LZ4 compression |
| `zstandard` | `hdf5-format` | Zstandard compression (requires C compiler; links `libzstd` via `zstd-sys`) |
| `parallel` | `hdf5-io` | Parallel chunk compression via rayon (default) |
| `threadsafe` | `hdf5` | `Send + Sync` file handles (`Arc<Mutex>`) |
| `mmap` | `hdf5-io` | Memory-mapped read-only file access |

## HDF5 format support

| Feature | Read | Write |
|---------|------|-------|
| Superblock v0/v1 (legacy) | Yes | — |
| Superblock v2/v3 | Yes | Yes |
| Object header v1 | Yes | — |
| Object header v2 | Yes | Yes |
| Contiguous storage | Yes | Yes |
| Chunked storage (EA) | Yes | Yes |
| Chunked storage (FA) | Yes | Yes |
| Chunked storage (BT2) | Yes | Yes |
| Compressed chunks | Yes | Yes |
| Hierarchical groups | Yes | Yes |
| Attributes | Yes | Yes |
| SWMR protocol | Yes | Yes |
| Hyperslab selection | Yes | Yes |
| Variable-length strings | Yes | — |
| Compound types | Yes | Yes |

## Benchmarks

```sh
cargo bench -p hdf5
```

Benchmarks cover contiguous read/write, chunked write, and compressed write throughput using [criterion](https://bheisler.github.io/criterion.rs/).

## Testing

```sh
cargo test --workspace
```

338 tests covering format codec, I/O roundtrips, compression, groups, attributes, SWMR, slice I/O, append mode with dataset resize, scalar datasets, and h5dump validation.

## License

MIT

This project is a Rust port inspired by the HDF5 C library, which is licensed
under the BSD-3-Clause license. See [LICENSE-HDF5](LICENSE-HDF5) for the
original HDF5 copyright notice and license terms.

The Zstandard implementation was developed with reference to the Zstandard C
library by Meta Platforms, Inc., licensed under BSD. See
[LICENSE-ZSTD](crates/hdf5-format/src/zstd/LICENSE-ZSTD) for details.
