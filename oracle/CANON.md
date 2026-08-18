# The canonical dump format (`!canon 7`)

Both sides of the oracle — `oracle/canon.py` (h5py / libhdf5 1.14.6) and
`src/bin/oracle_probe.rs` (rust-hdf5 public API) — emit this format, so the
two dumps of the same file are comparable as text and, more usefully, as a
`key -> value` map.

## Grammar

    line   := header | record
    header := "!canon" TAB "7"
    record := key TAB value
    key    := path [ "@" attrname ] "#" field
    value  := <no TAB, no LF>

Records are emitted objects-first in ascending path order; within an object
the fields follow the fixed order listed below, and attributes follow sorted
by name. Both sides must agree on the ordering, but the runner compares the
parsed `key -> value` map, so a pure ordering difference is reported
separately from a value difference.

`path` is the absolute HDF5 path (`/`, `/grp`, `/grp/ds`). The file itself
uses the empty path, so its fields are keyed `#field`.

## Value markers

A value that begins with

* `UNSUPPORTED(` — the producing side has no public API to observe this
  field. This is a *capability gap*, not a divergence.
* `ERROR(` — the producing side has an API for the field but it failed.
  On the rust side this is a read defect; on the h5py side it means the
  oracle itself could not answer, and the field is excluded from the verdict.

Everything else is an observed value and must match byte for byte.

## File-level fields (key `#...`)

| field        | value                                                    |
|--------------|----------------------------------------------------------|
| `superblock` | superblock version integer (0, 2, 3)                      |
| `userblock`  | user block size in bytes; `0` when there is none          |
| `fspace`     | `<strategy>/<persist>/<threshold>/<page size>`            |
| `freespace`  | `tracked` or `none`                                       |

The user block displaces the superblock, so `userblock` is the offset at
which the HDF5 signature was found (0, or the first power of two >= 512).

`fspace` is the file-space strategy the file was created with: one of
`fsmaggr`, `page`, `aggr`, `none` (`H5F_fspace_strategy_t`), then `true` or
`false` for whether free-space manager state persists across close, then the
smallest section a manager tracks, then the file-space page size. Those four
are the properties `H5F__super_init` weighs against the library defaults when
deciding whether the file needs a file-space info message at all, so a file
differing only in its page size still carries one. A file with no message
reports the defaults, `fsmaggr/false/1/4096`. The h5py side reads the file
creation property list, which `H5F__super_read` fills from the on-disk
message; the rust side reads the message itself.

`freespace` says whether the file's on-disk free-space managers record any
space at all — `tracked` when at least one section is stored, `none`
otherwise. It is deliberately not a byte count: the two writers lay a file out
differently, so the same sequence of creates and appends frees different
numbers of bytes, and only the presence or absence of recorded space is a
property both must agree on. The h5py side parses `h5stat -S`'s
"Amount/Percent of tracked free space" line, `H5Fget_freespace` having no
h5py binding; the rust side sums the sections `H5File::tracked_free_space`
decodes.

## Object fields

`kind` is always first: `group`, `dataset`, `softlink`, `extlink`,
`committed-datatype`, `unknown`.

Group fields: `kind`, `linkorder`, `attrorder`, `nattrs`, then attributes.

Dataset fields, in order: `kind`, `dtype`, `strpad`, `shape`, `maxshape`,
`layout`, `chunk`, `chunkindex`, `external`, `virtual`, `filters`,
`fillvalue`, `filltime`, `alloctime`, `nattrs`, attributes, `data`.

Committed datatype fields: `kind`, `dtype`, `strpad`, `nattrs`, attributes.

Link fields: `kind`, `target`, and for an external link `resolved`.

| field        | value                                                                     |
|--------------|---------------------------------------------------------------------------|
| `dtype`      | canonical dtype string (below)                                            |
| `shape`      | `[2,3]`, `[]` for scalar, `null` for a NULL dataspace                     |
| `maxshape`   | `[2,U]` — `U` for an unlimited dimension; `[]`/`null` as for `shape`      |
| `layout`     | `compact` \| `contiguous` \| `chunked` \| `virtual`                        |
| `chunk`      | `[4,4]`, or `-` when the layout is not chunked                            |
| `chunkindex` | `btree1` \| `single` \| `implicit` \| `farray` \| `earray` \| `btree2` \| `-` |
| `external`   | `-`, or `["ext.raw"@0+64,...]` — `name@offset+size` per segment            |
| `virtual`    | `-`, or `["src.h5"::"/d" [0]-[7]->[0]-[7],...]` — source, then bounds      |
| `filters`    | `[]` or `[deflate(6)@0,shuffle(4)@0]` — `name(cd0\|cd1)@flags`             |
| `fillvalue`  | `default` \| `undefined` \| `0x<hex of the raw fill bytes>`                |
| `filltime`   | `alloc` \| `never` \| `ifset` — `H5D_fill_time_t`                          |
| `alloctime`  | `early` \| `late` \| `incr` — `H5D_alloc_time_t`                           |
| `nattrs`     | attribute count as iteration reports it                                   |
| `nattrs_hdr` | attribute count as the object header records it (what `H5Oget_info` uses) |
| `linkorder`  | `-` \| `tracked` \| `tracked+indexed` — link creation-order tracking       |
| `attrorder`  | as `linkorder`, for attributes                                            |
| `strpad`     | `-`, or `.=null;.member=spacepad` for each vlen string in the type tree   |
| `data`       | see below                                                                 |
| `target`     | soft: the link path; external: `<file>::<path>`                           |
| `resolved`   | external only: `dataset <shape> <data>` \| `group` \| `committed-datatype` \| `dangling` |

`target` is only the value stored in the link, so it is identical whether or
not the producing side can open the file it names. `resolved` is what crossing
the link lands on — the target's shape and payload for a dataset, `dangling`
when the target file or the target object is not there — which is what
separates a reader that follows external links from one that only lists them.

`chunkindex` is read on the h5py side from the on-disk layout message via
`h5debug`, the same route as `filters`. It was originally *derived* from the
DCPL and the dataspace following libhdf5's selection rules
(`H5D__layout_set_version` / `H5D__chunk_construct`), but those rules moved
under HDF5 2.0: `H5F__super_read` raises the file's low bound only for
SWMR-write opens there, where 1.14 raised it for every open, so a superblock-3
file can mix v1.10 indexes with v1 B-trees and no static property of the
dataset decides which — only the stored message does.

`filters` is read on the h5py side from the on-disk filter-pipeline message
via `h5debug`, not from `dcpl.get_filter()` (`H5Pget_filter2`). Opening a
scale-offset-filtered dataset re-invokes `H5Z__set_local_scaleoffset`, which
rebuilds the pipeline's cd_values in memory; for the reserved/unused tail
slots past the packed fill-value bytes, that rebuild leaves whatever was
there before instead of the zeros actually on disk. On a real single-filter
scale-offset case this was verified two ways, both independent of any Python
binding: a byte-exact manual parse of the filter-pipeline message (its
declared 88-byte size fully and exactly accounts for 20 zero-filled
cd_values, nothing left over), and `h5debug <file> <header_addr>` — which
calls libhdf5's own `H5O_pline_debug` directly — reporting `CD value 16`
through `19` as `0`. `dcpl.get_filter()` instead reported
`(1818321779, 1717989221, 7628147, 0)` for that same slot range, which
decodes as the ASCII bytes `"scaleoffset\0"`: the filter's own name string
bleeding through uncleared memory in the in-memory reconstruction. Routing
`filters` through `h5debug` for every case (filtered or not, not just
scale-offset) makes both oracle arms measure the same on-disk bytes; see
`canon.py`'s `filters_str`.

## `data`

* `empty` — NULL dataspace.
* `raw:<hex>` — the element bytes exactly as stored, C order, when the total
  is <= 1024 bytes.
* `sha256:<hex>` — SHA-256 of those same bytes when the total exceeds 1024.
* `vals:[v1,v2,...]` — for datatypes with no flat byte image (variable-length
  strings and sequences, references); each `v` is a canonical value.
* `valsha256:<hex>` — SHA-256 of the UTF-8 of the `vals:[...]` form when it
  exceeds 1024 characters.

## Attribute fields (key `path@name#...`)

`dtype`, `strpad`, `shape`, `value`. `value` uses the same encoding as `data`.

## Canonical values

* integers — decimal, signed when the type is signed.
* floats — `0x` followed by the IEEE bits in big-endian hex, `2*size` digits.
  Never a formatted decimal, so NaN payloads and signed zero survive.
* strings — `"..."`, with `"` and `\` backslash-escaped, printable ASCII
  verbatim, and everything else as `\xNN` / `\uNNNN` / `\UNNNNNNNN`.
* byte strings that are not valid UTF-8 — `b0x<hex>`.
* sequences — `[v1,v2,...]`; compound records — `{v1,v2,...}`.

## Canonical dtype strings

    i32le                       signed 32-bit little-endian
    u8le                        unsigned 8-bit
    i32le+off3p20               non-default bit offset / precision
    f64le                       IEEE double
    f16be+s15e10,5m0,10b15off0p16   non-IEEE float parameters
    str[8],pad=null,cset=ascii  fixed-length string
    vstr,cset=utf8              variable-length string
    compound[16]{x@0:f32le;y@4:f32le}
    enum(i8le){RED=0;GREEN=1}
    array[2,3](f64le)
    vlen(i32le)
    opaque[4],tag="raw"
    bits[1]le
    objref | regref
    time[4]le

Float parameters are appended only when they deviate from IEEE 754 for that
width (sign/exponent/mantissa positions and sizes, exponent bias, bit offset,
bit precision).

A fixed string carries its pad inline (`pad=null`). `vstr` does not: its pad
travels in the separate `strpad` field, so that a type tree holding several
variable-length strings names the pad of each. Both sides answer both forms.

`strpad` names each variable-length string by its position in the type tree:
`.` is the type itself, `.m` a compound member, `[]` an array element, `()` a
vlen element — so a compound is `.name=null;.tags[]=nullpad`. It is `-` when
the type tree holds no variable-length string, which both sides can answer.
