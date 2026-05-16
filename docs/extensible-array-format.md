# HDF5 Extensible Array (EA) chunk index — format reference

Reverse-engineered from the libhdf5 C source (`H5EApkg.h`, `H5EAhdr.c`,
`H5EAcache.c`, `H5EA.c`, `H5EAiblock.c`, `H5EAdblock.c`) and verified against
files produced by h5py 3.16 / libhdf5 2.0.0.

This document originally recorded why `rust-hdf5`'s Extensible Array was **not
byte-compatible with libhdf5**. Those defects have since been fixed; the
document is kept as the format reference. The two defects were:

1. **Data-block geometry.** rust-hdf5 sized data blocks `16, 16, 32, 32, 64,
   64, …` (doubling every two data blocks). libhdf5 sizes them `16, 32, 32,
   64, 64, 128, …`. They diverged at data-block index 1.
2. **Super blocks (`EASB`) were not implemented.** Even a 500-chunk dataset
   needs one super block with default parameters.

Both are now fixed (`EaGeometry` + `ExtensibleArraySuperBlock` in
`extensible_array.rs`, with matching writer/reader walks). EA round-trips
with h5py / libhdf5 in both directions, filtered and unfiltered, across
super blocks.

## Creation parameters

Stored in the data layout message and the EA header. Defaults shown are what
libhdf5 and rust-hdf5 both use.

| Parameter                   | Default | Meaning |
|------------------------------|---------|---------|
| `idx_blk_elmts`              | 4       | elements stored directly in the index block |
| `data_blk_min_elmts`         | 16      | smallest data block size (elements); power of two |
| `sup_blk_min_data_ptrs`      | 4       | min data-block pointers per super block; power of two |
| `max_nelmts_bits`            | 32      | `log2` of the maximum element count |
| `max_dblk_page_nelmts_bits`  | 10      | `log2` of the data-block page size (paging threshold) |

Derived:

```
arr_off_size      = (max_nelmts_bits + 7) / 8          # bytes for a block offset; 4 for 32
dblk_page_nelmts  = 1 << max_dblk_page_nelmts_bits      # 1024
```

## Geometry (`H5EA__hdr_init`)

The element space beyond the index block is partitioned into **super blocks**.
Super block `u` (`u = 0 .. nsblks-1`) holds `ndblks(u)` data blocks, each of
`dblk_nelmts(u)` elements:

```
nsblks         = 1 + (max_nelmts_bits - log2(data_blk_min_elmts))     # 29 for 32,16
ndblks(u)      = 2^(u / 2)                 # integer division
dblk_nelmts(u) = 2^((u + 1) / 2) * data_blk_min_elmts
```

`start_idx(u)` / `start_dblk(u)` accumulate over `u`:

```
start_idx(0)  = 0,  start_dblk(0) = 0
start_idx(u+1)  = start_idx(u)  + ndblks(u) * dblk_nelmts(u)
start_dblk(u+1) = start_dblk(u) + ndblks(u)
```

`start_idx(u)` is an element offset that **excludes** `idx_blk_elmts`.

With the default parameters (`data_blk_min_elmts = 16`):

| `u` | `ndblks` | `dblk_nelmts` | `start_idx` | `start_dblk` |
|-----|----------|---------------|-------------|--------------|
| 0   | 1        | 16            | 0           | 0            |
| 1   | 1        | 32            | 16          | 1            |
| 2   | 2        | 32            | 48          | 2            |
| 3   | 2        | 64            | 112         | 4            |
| 4   | 4        | 64            | 240         | 6            |
| 5   | 4        | 128           | 496         | 10           |
| 6   | 8        | 128           | 1008        | 14           |
| …   | …        | …             | …           | …            |
| 11  | 32       | 1024          | 32752       | —            |
| 12  | 64       | 1024          | 65520       | —            |
| 13  | 64       | 2048 (paged)  | 131056      | —            |

Data-block sizes in file order are therefore `16, 32, 32, 64, 64, 128, …`.

## Index block layout parameters (`H5EAiblock.c`)

The index block stores the data-block addresses of the first few super blocks
directly, and super-block addresses for the rest:

```
iblock_nsblks = 2 * log2(sup_blk_min_data_ptrs)   # 4 for sup_blk_min_data_ptrs = 4
ndblk_addrs   = 2 * (sup_blk_min_data_ptrs - 1)    # 6
nsblk_addrs   = nsblks - iblock_nsblks             # 25
```

Super blocks `0 .. iblock_nsblks-1` have their data-block addresses in the
index block's `dblk_addrs` array (`ndblk_addrs` slots — exactly the data-block
count of those super blocks). Super blocks `iblock_nsblks .. nsblks-1` are
reached through `EASB` structures whose addresses live in the index block's
`sblk_addrs` array (`nsblk_addrs` slots).

## Locating a chunk (`H5EA__lookup_elmt`, `H5EA__dblock_sblk_idx`)

Given a 0-based chunk index `idx`:

```
if idx < idx_blk_elmts:
    -> index block element[idx]

e        = idx - idx_blk_elmts
sblk_idx = floor(log2( e / data_blk_min_elmts + 1 ))     # H5VM_log2_gen
elmt     = e - start_idx(sblk_idx)

if sblk_idx < iblock_nsblks:
    global_dblk = start_dblk(sblk_idx) + elmt / dblk_nelmts(sblk_idx)
    data block  = index_block.dblk_addrs[global_dblk]
else:
    sblk_off    = sblk_idx - iblock_nsblks
    super block = index_block.sblk_addrs[sblk_off]
    local_dblk  = elmt / dblk_nelmts(sblk_idx)
    data block  = super_block.dblk_addrs[local_dblk]

offset_in_dblk = elmt % dblk_nelmts(sblk_idx)
```

## On-disk byte layouts (`H5EAcache.c`)

`sa` = `sizeof_addr`, `ss` = `sizeof_size`, `raw_elmt_size` = `sa` for
unfiltered chunks. Every structure ends with a 4-byte Jenkins lookup3
checksum (`H5_checksum_metadata`). Multi-byte integers are little-endian.

### EA header — magic `EAHD`

```
"EAHD"(4) version(1) client_id(1)
element_size(1) max_nelmts_bits(1) idx_blk_elmts(1)
data_blk_min_elmts(1) sup_blk_min_data_ptrs(1) max_dblk_page_nelmts_bits(1)
num_sblks_created(ss) size_sblks_created(ss)
num_dblks_created(ss) size_dblks_created(ss)
max_idx_set(ss) num_elmts_realized(ss)
index_block_address(sa)
checksum(4)
```

rust-hdf5 already matches this.

### EA index block — magic `EAIB`

```
"EAIB"(4) version(1) client_id(1)
header_address(sa)
elements(idx_blk_elmts * raw_elmt_size)
data_block_addresses(ndblk_addrs * sa)
super_block_addresses(nsblk_addrs * sa)
checksum(4)
```

rust-hdf5 already matches this.

### EA super block — magic `EASB`

```
"EASB"(4) version(1) client_id(1)
header_address(sa)
block_offset(arr_off_size)
[ page-init bitmaps: ndblks * dblk_page_init_size  -- only if data blocks paged ]
data_block_addresses(ndblks * sa)
checksum(4)
```

`ndblks` here is `ndblks(sblk_idx)` for the super block's own index.

### EA data block — magic `EADB`

```
"EADB"(4) version(1) client_id(1)
header_address(sa)
block_offset(arr_off_size)
[ elements: nelmts * raw_elmt_size  -- only if not paged ]
checksum(4)
```

rust-hdf5 already matches this.

### EA data block page (paged data blocks only)

```
elements(dblk_page_nelmts * raw_elmt_size)
checksum(4)
```

## `block_offset` field values

`block_offset` is a sanity field; libhdf5's reader decodes but does not verify
it. To produce byte-faithful files, libhdf5 writes:

- index-block data block: `start_idx(sblk) + (start_dblk(sblk) + local_dblk) * dblk_nelmts(sblk)`
- super-block data block:  `start_idx(sblk) + local_dblk * dblk_nelmts(sblk)`
- super block itself:      `start_idx(sblk)`

(The index-block formula uses the *global* data-block index, which inflates
the value; this is libhdf5's actual behaviour, confirmed against
`/tmp/ea_ref.h5`.)

## Paging

When `dblk_nelmts(u) > dblk_page_nelmts` (i.e. `u >= 13` for the default
parameters — `dblk_nelmts(12) = 1024` equals the page size and is *not*
paged, `dblk_nelmts(13) = 2048` is), data blocks in super block `u` are
split into pages of `dblk_page_nelmts` elements. The data block then stores
only its prefix; the elements live in `EA data block page` structures
appended after the prefix, and the owning super block carries a page-init
bitmap. With default parameters this only matters at chunk index `131,060`
and beyond; non-paged super blocks cover everything below that.

## Worked example — `/tmp/ea_ref.h5`

A 500-chunk `i4` dataset, `chunks=(1,)`, written by h5py with `libver=latest`:

- EA header at offset 447, index block at 519.
- Index block elements (chunks 0–3): chunk data addresses `0x800, 0x804,
  0x808, 0x80c`.
- 6 direct data-block addresses; their `block_offset` fields are
  `0, 48, 112, 144, 368, 432` — matching the index-block `block_offset`
  formula exactly.
- 1 super block (at 0x709) with 4 data-block addresses, covering chunks
  244–499.

## Verification

h5py / libhdf5 are available at `/Users/stevek/mamba/envs/bs2026.1`. The fix
must be validated **both** directions:

- rust-hdf5 writes an EA dataset → h5py reads it back correctly.
- h5py writes an EA dataset → rust-hdf5 reads it back correctly.

`/tmp/make_ea_ref.py` and `/tmp/ea_dump*.py` hold the diagnostic scripts used
to produce this document.

## Implementation status

Done:

1. `extensible_array.rs` — `EaGeometry` is the single source of truth for
   `nsblks`, `ndblks(u)`, `dblk_nelmts(u)`, `start_idx`, `start_dblk`,
   `iblock_nsblks`, `ndblk_addrs`, `nsblk_addrs`, and the chunk lookup
   (`EaGeometry::locate`).
2. `ExtensibleArraySuperBlock` with `encode` / `decode` (super blocks are
   filter-agnostic — they hold data-block addresses only).
3. `writer.rs::write_chunk` walks data blocks and super blocks via
   `EaGeometry`, creating super blocks on demand.
4. `reader.rs::collect_ea_chunk_entries` walks the corrected geometry,
   reading super blocks; `read_chunked_v4` delegates to it.
5. Filtered EA and `open_append` both covered.
6. Layout message version 5 is accepted (it is structurally identical to
   version 4; libhdf5 emits v5 for filtered datasets).

7. `reader.rs::collect_ea_chunk_entries` reads paged data blocks: when a
   super block's data blocks exceed the page size, it walks the per-super-
   block page-init bitmap (one flat MSB-first bitmap indexed by
   `dblk_idx * npages + page_idx`) and reads the element pages.

Verified round-trip with h5py 3.16 / libhdf5 2.0.0 in both directions,
filtered and unfiltered, across super blocks (1500–2000 chunks) and across
paged data blocks (140,000 chunks).

Not done — *writing* paged data blocks (chunk index `>= 131,060` with
default parameters): the writer errors clearly rather than emitting an
unpaged block where libhdf5 expects pages. The reader handles paged data
blocks (item 7).
