# The canonical dump format (`!canon 1`)

Both sides of the oracle — `oracle/canon.py` (h5py / libhdf5 1.14.6) and
`src/bin/oracle_probe.rs` (rust-hdf5 public API) — emit this format, so the
two dumps of the same file are comparable as text and, more usefully, as a
`key -> value` map.

## Grammar

    line   := header | record
    header := "!canon" TAB "1"
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

## Object fields

`kind` is always first: `group`, `dataset`, `softlink`, `extlink`,
`committed-datatype`, `unknown`.

Group fields: `kind`, `nattrs`, then attributes.

Dataset fields, in order: `kind`, `dtype`, `shape`, `maxshape`, `layout`,
`chunk`, `chunkindex`, `filters`, `fillvalue`, `nattrs`, attributes, `data`.

Link fields: `kind`, `target`.

| field        | value                                                                     |
|--------------|---------------------------------------------------------------------------|
| `dtype`      | canonical dtype string (below)                                            |
| `shape`      | `[2,3]`, `[]` for scalar, `null` for a NULL dataspace                     |
| `maxshape`   | `[2,U]` — `U` for an unlimited dimension; `[]`/`null` as for `shape`      |
| `layout`     | `compact` \| `contiguous` \| `chunked` \| `virtual`                        |
| `chunk`      | `[4,4]`, or `-` when the layout is not chunked                            |
| `chunkindex` | `btree1` \| `single` \| `implicit` \| `farray` \| `earray` \| `btree2` \| `-` |
| `filters`    | `[]` or `[deflate(6)@0,shuffle(4)@0]` — `name(cd0\|cd1)@flags`             |
| `fillvalue`  | `default` \| `undefined` \| `0x<hex of the raw fill bytes>`                |
| `nattrs`     | attribute count                                                           |
| `data`       | see below                                                                 |
| `target`     | soft: the link path; external: `<file>::<path>`                           |

`chunkindex` is *derived* on the h5py side from the DCPL and the dataspace
following libhdf5's selection rules (`H5D__layout_set_version` /
`H5D__chunk_construct`), because neither h5py nor the h5 CLI tools expose the
chosen index type. It is marked as derived in the report.

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

`dtype`, `shape`, `value`. `value` uses the same encoding as `data`.

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

`vstr` deliberately omits the string-pad field: `DatatypeMessage::VarLenString`
in rust-hdf5 models only the character set, so including the pad would report
a modelling gap as a per-case divergence in every variable-length string case.
The gap is recorded once in the report instead.
