# Where a contiguous dataset's raw data lands, and what it costs

`perf/run.py contig-write` — create a file, create one 16Mi-element `f64`
contiguous dataset, write it, close, delete — has sat at 1.04–1.07x of
libhdf5 1.14.6 while every other write workload is at or below 1.0x. This
records what that residual is, because it is not something the write path
can be made to stop doing.

All numbers below are minima of interleaved runs pinned with
`taskset -c 20`, on `/dev/shm` unless a line says ext4.

## The write path spends no measurable CPU

`Dataset::write_raw` hands the caller's `&[T]` to `to_stored_byte_order`,
which returns `Cow::Borrowed` for a native-order type, and the borrowed
slice goes straight to one `pwrite`. There is no staging buffer, no
conversion pass, and no second allocation — `strace` shows three writes per
rep (128 MiB of data at 48, a 157-byte object header at the end, a 48-byte
superblock at 0) against libhdf5's four, and in-process phase timing of the
best rep splits as:

| phase | time |
| --- | --- |
| `H5File::create` | 13.5 µs |
| `new_dataset(...).create()` | 3.3 µs |
| `write_raw` | 33.37 ms |
| `close` | 12.6 µs |
| (probe's own `remove_file`) | 5.44 ms |

The library's own CPU is 30 µs out of a 38.8 ms rep — 0.08%. The rep is one
`pwrite` and one `unlink`.

## The residual is the destination file offset

This crate allocates the raw-data block immediately after the 48-byte
version-3 superblock, so the 128 MiB `pwrite` lands at file offset 48.
libhdf5 carves metadata out of 2048-byte blocks (`H5F_ACC_DEF_META_BLOCK_SIZE`),
so its raw data lands at 2048. That offset, and nothing else about the two
call sequences, is the gap. Randomised 20-round comparison of the two
syscall shapes with no library under them:

| shape | min |
| --- | --- |
| libhdf5's four writes, data at 2048 | 37.81 ms |
| this crate's three writes, data at 48 | 38.90 ms |
| this crate's three writes, data at **2048** | 37.78 ms |

Pre-extending with `ftruncate`, writing the superblock first, writing the
tail first, splitting the big write at the page boundary, and dropping the
`fsync` all land at 38.82–38.92 ms: none of them matter.

libhdf5 pays the same penalty when its own data block is moved. Forcing the
offset with a userblock (`H5Pset_userblock`), same library, same data:

| userblock | data offset | /dev/shm | ext4 |
| --- | --- | --- | --- |
| none | 2048 | 37.26 ms | 50.94 ms |
| 512 | 2560 | 37.43 ms | 51.26 ms |
| 1024 | 3072 | 37.26 ms | 51.05 ms |
| 2048 | **4096** | **40.04 ms** | **53.89 ms** |
| 4096 | 6144 | 37.23 ms | 50.91 ms |

So this is a property of `copy_from_user` into the page cache on this
machine, not of either library's write path.

## Read and write want opposite offsets

The cost depends on the page-offset relationship between the caller's buffer
and the file offset, and it inverts between the two directions. Bare 128 MiB
`pwrite`/`pread`, user buffer at page offset 16 (what a large `malloc` gives),
each offset measured with a freshly written file, randomised, 12 rounds:

| data offset | write | read |
| --- | --- | --- |
| 48 | 33.25 ms | 66.08 ms |
| 256 | 33.39 ms | 65.43 ms |
| 512 | 33.46 ms | 72.03 ms |
| 1024 | 31.94 ms | 71.13 ms |
| 2048 | 31.51 ms | 71.35 ms |
| 3072 | 31.70 ms | 71.55 ms |
| 4048 | 32.73 ms | 71.60 ms |

Writing prefers an in-page offset of 1024 or more; reading prefers 256 or
less. Minor-fault counts are identical (32769) across all of them, so this is
copy cost on both sides, not fault accounting. Offset 48 is the read
optimum, 2048 is the write optimum, and there is no offset that is both.

That is exactly where the two libraries sit in the perf matrix: this crate is
1.05x on `contig-write` and 0.97x on `contig-read`; libhdf5 is the mirror
image. Summed over the pair the two are within 0.2% of each other.

## What aligning raw data would buy, and cost

Rounding raw-data allocations of at least 64 KiB up to 2048 puts this crate's
data block at 2048 and closes `contig-write` — measured as an interleaved
before/after of the same probe:

| workload | ratio after |
| --- | --- |
| contig-write | 0.956 |
| chunked-write | 0.943 |
| chunked-read | 0.974 |
| deflate-write | 0.987 |
| small-write | 1.042 |
| **contig-read** | **1.080** |

Against libhdf5 that moves `contig-write` from 1.05x to 0.99x and
`contig-read` from 0.97x to about 1.05x. It is a trade between the two
directions of the same 128 MiB, not a saving, and it changes the byte layout
of every file holding a large dataset.
