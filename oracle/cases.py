#!/usr/bin/env python3
"""The oracle's feature matrix.

Each case names one HDF5 feature, carries an h5py generator that writes the
reference file, and — when rust-hdf5's *public* API can express the same file —
the id of the matching arm in `src/bin/oracle_probe.rs`'s `write` subcommand.

The data every case writes is deliberately formulaic (`arange`-like ramps,
short literal strings) so the rust writer can reproduce it byte for byte
without the two sides sharing a data file.

Only the standard library, numpy and h5py are used.
"""

import pathlib
import shutil

import numpy as np

import h5py
from h5py import h5d, h5p, h5s, h5t

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

N = 8  # default element count for the 1-D dtype ramps


def ramp(dtype, n=N):
    return np.arange(n, dtype=np.dtype(dtype))


def lowlevel_dataset(f, name, tid, sid, data=None, dcpl=None, mtype=None):
    """Create a dataset from a raw TypeID/SpaceID, which h5py cannot express.

    Returns the low-level DatasetID so the caller can write through it.
    """
    dsid = h5d.create(f.id, name.encode("utf-8"), tid, sid, dcpl=dcpl)
    if data is not None:
        dsid.write(h5s.ALL, h5s.ALL, np.ascontiguousarray(data), mtype=mtype)
    return dsid


def chunked_dcpl(chunk, alloc_time=None, layout=h5d.CHUNKED):
    dcpl = h5p.create(h5p.DATASET_CREATE)
    dcpl.set_layout(layout)
    if layout == h5d.CHUNKED:
        dcpl.set_chunk(tuple(chunk))
    if alloc_time is not None:
        dcpl.set_alloc_time(alloc_time)
    return dcpl


class Case:
    def __init__(self, name, group, gen, rust=None, note="", ext_files=()):
        self.name = name
        self.group = group
        self.gen = gen
        self.rust = rust
        self.note = note
        self.ext_files = ext_files

    def __repr__(self):
        return "Case(%s)" % self.name


# --------------------------------------------------------------------------
# integer widths / signedness / endianness
# --------------------------------------------------------------------------


def _int_case(name, npdtype, rust):
    def gen(path):
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=ramp(npdtype))

    return Case(name, "dtype-int", gen, rust, "1-D ramp of %s" % npdtype)


INT_CASES = [
    _int_case("int_i8", "i1", "int_i8"),
    _int_case("int_u8", "u1", "int_u8"),
    _int_case("int_i16le", "<i2", "int_i16le"),
    _int_case("int_u16le", "<u2", "int_u16le"),
    _int_case("int_i32le", "<i4", "int_i32le"),
    _int_case("int_u32le", "<u4", "int_u32le"),
    _int_case("int_i64le", "<i8", "int_i64le"),
    _int_case("int_u64le", "<u8", "int_u64le"),
    _int_case("int_i16be", ">i2", "int_i16be"),
    _int_case("int_i32be", ">i4", "int_i32be"),
    _int_case("int_u64be", ">u8", "int_u64be"),
]


# --------------------------------------------------------------------------
# floating point
# --------------------------------------------------------------------------


def _float_case(name, npdtype, rust):
    def gen(path):
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=ramp(npdtype))

    return Case(name, "dtype-float", gen, rust, "1-D ramp of %s" % npdtype)


def gen_float_specials(path):
    bits = np.array(
        [
            0x7FF8000000000001,  # quiet NaN with a payload
            0x7FF0000000000000,  # +inf
            0xFFF0000000000000,  # -inf
            0x8000000000000000,  # -0.0
            0x0000000000000001,  # smallest denormal
            0x3FF0000000000000,  # 1.0
            0xBFF0000000000000,  # -1.0
            0x0000000000000000,  # +0.0
        ],
        dtype="<u8",
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=bits.view("<f8"))


FLOAT_CASES = [
    _float_case("float_f16le", "<f2", "float_f16le"),
    _float_case("float_f32le", "<f4", "float_f32le"),
    _float_case("float_f64le", "<f8", "float_f64le"),
    _float_case("float_f64be", ">f8", "float_f64be"),
    Case(
        "float_specials",
        "dtype-float",
        gen_float_specials,
        "float_specials",
        "NaN payload, +/-inf, -0.0, denormal — bit patterns must survive",
    ),
]


# --------------------------------------------------------------------------
# strings
# --------------------------------------------------------------------------

STRINGS = ["alpha", "b", "", "delta12"]
UNISTR = ["été", "日本", "", "café"]


def gen_str_fixed_ascii(path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data",
            data=np.array([s.encode("ascii") for s in STRINGS], dtype="S8"),
            dtype=h5py.string_dtype("ascii", 8),
        )


def gen_str_fixed_utf8(path):
    # libhdf5 has no ASCII -> UTF-8 conversion path, so the memory type has to
    # be the UTF-8 fixed string itself and the bytes go through verbatim.
    with h5py.File(path, "w") as f:
        tid = h5t.C_S1.copy()
        tid.set_size(16)
        tid.set_cset(h5t.CSET_UTF8)
        tid.set_strpad(h5t.STR_NULLPAD)
        sid = h5s.create_simple((len(UNISTR),))
        lowlevel_dataset(
            f,
            "data",
            tid,
            sid,
            np.array([s.encode("utf-8") for s in UNISTR], dtype="S16"),
            mtype=tid,
        )


def _fixed_str_pad(strpad):
    # libhdf5 does not re-pad on a same-size same-cset copy, so the padding
    # bytes have to be written explicitly for the declared rule to be what is
    # actually on disk.
    padbyte = b" " if strpad == h5t.STR_SPACEPAD else b"\0"

    def gen(path):
        with h5py.File(path, "w") as f:
            tid = h5t.C_S1.copy()
            tid.set_size(8)
            tid.set_cset(h5t.CSET_ASCII)
            tid.set_strpad(strpad)
            sid = h5s.create_simple((len(STRINGS),))
            data = np.array(
                [s.encode("ascii").ljust(8, padbyte) for s in STRINGS], dtype="S8"
            )
            lowlevel_dataset(f, "data", tid, sid, data, mtype=tid)

    return gen


def gen_str_vlen_ascii(path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data",
            data=np.array(STRINGS, dtype=object),
            dtype=h5py.string_dtype("ascii"),
        )


def gen_str_vlen_utf8(path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data",
            data=np.array(UNISTR, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )


STRING_CASES = [
    Case("str_fixed_ascii", "dtype-string", gen_str_fixed_ascii, "str_fixed_ascii",
         "8-byte fixed ASCII strings"),
    Case("str_fixed_utf8", "dtype-string", gen_str_fixed_utf8, "str_fixed_utf8",
         "16-byte fixed UTF-8 strings"),
    Case("str_fixed_nullpad", "dtype-string", _fixed_str_pad(h5t.STR_NULLPAD), "str_fixed_nullpad",
         "fixed string with STR_NULLPAD"),
    Case("str_fixed_spacepad", "dtype-string", _fixed_str_pad(h5t.STR_SPACEPAD), "str_fixed_spacepad",
         "fixed string with STR_SPACEPAD"),
    Case("str_vlen_ascii", "dtype-string", gen_str_vlen_ascii, "str_vlen_ascii",
         "variable-length ASCII strings via the global heap"),
    Case("str_vlen_utf8", "dtype-string", gen_str_vlen_utf8, "str_vlen_utf8",
         "variable-length UTF-8 strings via the global heap"),
]


# --------------------------------------------------------------------------
# compound / array / enum / opaque / bitfield / references / vlen
# --------------------------------------------------------------------------

COMPOUND_SIMPLE = np.dtype([("x", "<f4"), ("y", "<f4")])
COMPOUND_NESTED = np.dtype([("a", "<i4"), ("inner", [("u", "<i2"), ("v", "<i2")])])
COMPOUND_STR = np.dtype([("id", "<i4"), ("name", "S8")])
COMPOUND_PAD = np.dtype(
    {"names": ["a", "b"], "formats": ["<i2", "<i4"], "offsets": [0, 4], "itemsize": 12}
)


def gen_compound_simple(path):
    arr = np.zeros(4, dtype=COMPOUND_SIMPLE)
    arr["x"] = np.arange(4, dtype="<f4")
    arr["y"] = np.arange(100, 104, dtype="<f4")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=arr)


def gen_compound_nested(path):
    arr = np.zeros(4, dtype=COMPOUND_NESTED)
    arr["a"] = np.arange(4, dtype="<i4")
    arr["inner"]["u"] = np.arange(10, 14, dtype="<i2")
    arr["inner"]["v"] = np.arange(20, 24, dtype="<i2")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=arr)


def gen_compound_with_string(path):
    arr = np.zeros(3, dtype=COMPOUND_STR)
    arr["id"] = np.arange(3, dtype="<i4")
    arr["name"] = [b"aa", b"bbb", b"cccc"]
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=arr)


def gen_compound_padded(path):
    arr = np.zeros(4, dtype=COMPOUND_PAD)
    arr["a"] = np.arange(4, dtype="<i2")
    arr["b"] = np.arange(1000, 1004, dtype="<i4")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=arr)


def gen_array_dtype(path):
    with h5py.File(path, "w") as f:
        tid = h5t.array_create(h5t.IEEE_F64LE, (2, 3))
        sid = h5s.create_simple((2,))
        data = np.arange(12, dtype="<f8").reshape(2, 2, 3)
        lowlevel_dataset(f, "data", tid, sid, data, mtype=tid)


def gen_enum_i8(path):
    dt = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 2}, basetype="i1")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.array([0, 1, 2, 1], dtype="i1"), dtype=dt)


def gen_enum_i32(path):
    dt = h5py.enum_dtype({"LOW": -1, "MID": 0, "HIGH": 1000}, basetype="<i4")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.array([-1, 0, 1000, 0], dtype="<i4"), dtype=dt)


def gen_compound_dtype_v4(path):
    # A v1.12 low bound makes libhdf5 tag the compound datatype message
    # version 4 (H5O_dtype_ver_bounds); its members stay version 1. Nothing
    # else in the matrix produces a datatype message above version 3.
    #
    # Chunked on purpose: a *contiguous* dataset at a v1.10+ bound is dropped
    # from the listing by an unrelated gap (`layout_contiguous_v110`), which
    # would mask what this case is about.
    arr = np.zeros(4, dtype=COMPOUND_SIMPLE)
    arr["x"] = np.arange(4, dtype="<f4")
    arr["y"] = np.arange(100, 104, dtype="<f4")
    with h5py.File(path, "w", libver=("v112", "v112")) as f:
        ds = f.create_dataset("data", (4,), chunks=(4,), dtype=COMPOUND_SIMPLE)
        ds[...] = arr


def gen_opaque(path):
    with h5py.File(path, "w") as f:
        tid = h5t.create(h5t.OPAQUE, 4)
        tid.set_tag(b"raw4")
        sid = h5s.create_simple((3,))
        data = np.frombuffer(bytes(range(12)), dtype="V4")
        lowlevel_dataset(f, "data", tid, sid, data, mtype=tid)


def gen_bitfield(path):
    with h5py.File(path, "w") as f:
        tid = h5t.STD_B8LE.copy()
        sid = h5s.create_simple((4,))
        lowlevel_dataset(
            f, "data", tid, sid, np.array([0x01, 0x80, 0xFF, 0x00], dtype="u1")
        )


def gen_ref_object(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("target", data=ramp("<i4"))
        g = f.create_group("grp")
        refs = np.array([f["target"].ref, g.ref], dtype=h5py.ref_dtype)
        f.create_dataset("refs", data=refs, dtype=h5py.ref_dtype)


def gen_ref_region(path):
    with h5py.File(path, "w") as f:
        t = f.create_dataset("target", data=ramp("<i4"))
        refs = np.array([t.regionref[0:3], t.regionref[4:8]],
                        dtype=h5py.regionref_dtype)
        f.create_dataset("refs", data=refs, dtype=h5py.regionref_dtype)


def gen_vlen_numeric(path):
    dt = h5py.vlen_dtype(np.dtype("<i4"))
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", (3,), dtype=dt)
        ds[0] = np.array([1, 2, 3], dtype="<i4")
        ds[1] = np.array([], dtype="<i4")
        ds[2] = np.array([-7], dtype="<i4")


def gen_vlen_bytes(path):
    dt = h5py.vlen_dtype(np.dtype("u1"))
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", (3,), dtype=dt)
        ds[0] = np.array([0, 1, 2], dtype="u1")
        ds[1] = np.array([], dtype="u1")
        ds[2] = np.array([255], dtype="u1")


def gen_named_datatype(path):
    with h5py.File(path, "w") as f:
        f["t"] = np.dtype("<i4")
        f.create_dataset("data", data=ramp("<i4"))
        # A dataset created from the committed TypeID stores a *shared*
        # datatype message pointing at /t rather than a datatype of its own.
        sid = h5s.create_simple((N,))
        lowlevel_dataset(f, "shared", f["t"].id, sid, ramp("<i4"))


COMPOSITE_CASES = [
    Case("compound_simple", "dtype-composite", gen_compound_simple, "compound_simple",
         "two f32 members, no padding"),
    Case("compound_nested", "dtype-composite", gen_compound_nested, "compound_nested",
         "compound member inside a compound"),
    Case("compound_with_string", "dtype-composite", gen_compound_with_string,
         "compound_with_string", "fixed string member"),
    Case("compound_padded", "dtype-composite", gen_compound_padded, "compound_padded",
         "member offsets with gaps and trailing padding"),
    Case("compound_dtype_v4", "dtype-composite", gen_compound_dtype_v4,
         "compound_dtype_v4", "version-4 datatype message (libver v1.12 bounds)"),
    Case("array_dtype", "dtype-composite", gen_array_dtype, "array_dtype",
         "H5T_ARRAY element type (2x3 f64)"),
    Case("enum_i8", "dtype-composite", gen_enum_i8, "enum_i8", "3-member i8 enum"),
    Case("enum_i32", "dtype-composite", gen_enum_i32, "enum_i32",
         "i32 enum with a negative member"),
    Case("opaque", "dtype-composite", gen_opaque, "opaque", "H5T_OPAQUE with a tag"),
    Case("bitfield", "dtype-composite", gen_bitfield, "bitfield",
         "H5T_BITFIELD (STD_B8LE)"),
    Case("ref_object", "dtype-composite", gen_ref_object, "ref_object",
         "object references to a dataset and a group"),
    Case("ref_region", "dtype-composite", gen_ref_region, "ref_region",
         "dataset region references"),
    Case("vlen_numeric", "dtype-composite", gen_vlen_numeric, "vlen_numeric",
         "variable-length i32 sequences"),
    Case("vlen_bytes", "dtype-composite", gen_vlen_bytes, "vlen_bytes",
         "variable-length u8 sequences"),
    Case("named_datatype", "dtype-composite", gen_named_datatype, None,
         "committed datatype object, and a dataset that shares it"),
]


# --------------------------------------------------------------------------
# layouts and chunk indexes
# --------------------------------------------------------------------------


def gen_layout_compact(path):
    with h5py.File(path, "w") as f:
        dcpl = h5p.create(h5p.DATASET_CREATE)
        dcpl.set_layout(h5d.COMPACT)
        sid = h5s.create_simple((16,))
        lowlevel_dataset(
            f, "data", h5t.STD_I32LE, sid, ramp("<i4", 16), dcpl=dcpl
        )


def gen_layout_contiguous(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=ramp("<i4", 16))


def gen_layout_contiguous_v110(path):
    with h5py.File(path, "w", libver=("v110", "v110")) as f:
        f.create_dataset("data", data=ramp("<i4", 16))


def gen_layout_chunked_v110(path):
    with h5py.File(path, "w", libver=("v110", "v110")) as f:
        ds = f.create_dataset("data", (16,), chunks=(16,), dtype="<i4")
        ds[...] = ramp("<i4", 16)


def gen_chunkidx_btree1(path):
    with h5py.File(path, "w", libver="earliest") as f:
        ds = f.create_dataset(
            "data", (8,), maxshape=(None,), chunks=(4,), dtype="<i4"
        )
        ds[...] = ramp("<i4")


def gen_chunkidx_single(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset("data", (8,), chunks=(8,), dtype="<i4")
        ds[...] = ramp("<i4")


def gen_chunkidx_implicit(path):
    with h5py.File(path, "w", libver="latest") as f:
        dcpl = chunked_dcpl((4,), alloc_time=h5d.ALLOC_TIME_EARLY)
        sid = h5s.create_simple((16,))
        lowlevel_dataset(f, "data", h5t.STD_I32LE, sid, ramp("<i4", 16), dcpl=dcpl)


def gen_chunkidx_farray(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset("data", (16,), chunks=(4,), dtype="<i4")
        ds[...] = ramp("<i4", 16)


def gen_chunkidx_earray(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", (16,), maxshape=(None,), chunks=(4,), dtype="<i4"
        )
        ds[...] = ramp("<i4", 16)


def gen_chunkidx_earray_unlim_inner(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", (4, 4), maxshape=(4, None), chunks=(2, 2), dtype="<i4"
        )
        ds[...] = ramp("<i4", 16).reshape(4, 4)


def gen_layout_contiguous_v108(path):
    with h5py.File(path, "w", libver=("v108", "v108")) as f:
        f.create_dataset("data", data=ramp("<i4", 16))


def gen_layout_chunked_v108(path):
    with h5py.File(path, "w", libver=("v108", "v108")) as f:
        ds = f.create_dataset("data", (16,), chunks=(16,), dtype="<i4")
        ds[...] = ramp("<i4", 16)


def gen_chunkidx_earray_dim1(path):
    # The extensible dimension is the fastest-changing one, so the chunk
    # coordinate the index is keyed on is not the first.
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", (4, 4), maxshape=(4, None), chunks=(2, 4), dtype="<i4"
        )
        ds[...] = ramp("<i4", 16).reshape(4, 4)


def gen_external_storage(path):
    raw = path.parent / (path.stem + "_ext.raw")
    raw.write_bytes(ramp("<i4", 16).tobytes())
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data", shape=(16,), dtype="<i4", external=[(raw.name, 0, 64)]
        )


def gen_vds(path):
    src = path.parent / (path.stem + "_src.h5")
    with h5py.File(src, "w") as g:
        g.create_dataset("src", data=ramp("<i4", 16))
    layout = h5py.VirtualLayout(shape=(16,), dtype="<i4")
    layout[...] = h5py.VirtualSource(src.name, "src", shape=(16,))
    with h5py.File(path, "w") as f:
        f.create_virtual_dataset("vds", layout)


def gen_chunkidx_btree2(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", (4, 4), maxshape=(None, None), chunks=(2, 2), dtype="<i4"
        )
        ds[...] = ramp("<i4", 16).reshape(4, 4)


LAYOUT_CASES = [
    Case("layout_compact", "layout", gen_layout_compact, None,
         "compact layout — data inside the object header"),
    Case("layout_contiguous", "layout", gen_layout_contiguous, "layout_contiguous",
         "contiguous layout"),
    Case("layout_contiguous_v110", "layout", gen_layout_contiguous_v110, None,
         "contiguous layout under v1.10 bounds — data layout message v4"),
    Case("layout_chunked_v110", "layout", gen_layout_chunked_v110, None,
         "chunked layout under v1.10 bounds — the control for the case above"),
    Case("chunkidx_btree1", "layout", gen_chunkidx_btree1, "chunkidx_btree1",
         "layout v3 + version-1 B-tree chunk index (libver earliest)"),
    Case("chunkidx_single", "layout", gen_chunkidx_single, "chunkidx_single",
         "single-chunk index"),
    Case("chunkidx_implicit", "layout", gen_chunkidx_implicit, None,
         "implicit index — fixed shape, early allocation, no filter"),
    Case("chunkidx_farray", "layout", gen_chunkidx_farray, "chunkidx_farray",
         "fixed-array index"),
    Case("chunkidx_earray", "layout", gen_chunkidx_earray, "chunkidx_earray",
         "extensible-array index — one unlimited dimension"),
    Case("chunkidx_earray_unlim_inner", "layout", gen_chunkidx_earray_unlim_inner,
         "chunkidx_earray_unlim_inner",
         "extensible-array index — the unlimited dimension is dim 1, not dim 0"),
    Case("chunkidx_btree2", "layout", gen_chunkidx_btree2, "chunkidx_btree2",
         "version-2 B-tree index — two unlimited dimensions"),
    Case("layout_contiguous_v108", "layout", gen_layout_contiguous_v108, None,
         "contiguous layout under v1.8 bounds — the v1.10 pair's control"),
    Case("layout_chunked_v108", "layout", gen_layout_chunked_v108, None,
         "chunked layout under v1.8 bounds"),
    Case("chunkidx_earray_dim1", "layout", gen_chunkidx_earray_dim1,
         "chunkidx_earray_dim1",
         "extensible-array index whose unlimited dimension is not the first"),
    Case("external_storage", "layout", gen_external_storage, None,
         "contiguous data held in an external raw file",
         ext_files=("_ext.raw",)),
    Case("vds", "layout", gen_vds, None,
         "virtual dataset mapped onto a dataset in a sibling file",
         ext_files=("_src.h5",)),
]


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------


def _filter_case(name, rust, note, **kw):
    def gen(path):
        with h5py.File(path, "w", libver="latest") as f:
            ds = f.create_dataset("data", (64,), chunks=(16,), dtype="<i4", **kw)
            ds[...] = ramp("<i4", 64)

    return Case(name, "filter", gen, rust, note)


FILTER_CASES = [
    _filter_case("filter_deflate", "filter_deflate", "deflate level 6",
                 compression="gzip", compression_opts=6),
    _filter_case("filter_shuffle", None, "shuffle only",
                 shuffle=True),
    _filter_case("filter_fletcher32", "filter_fletcher32", "fletcher32 checksum",
                 fletcher32=True),
    _filter_case("filter_deflate_shuffle", "filter_deflate_shuffle",
                 "shuffle then deflate",
                 compression="gzip", compression_opts=6, shuffle=True),
    _filter_case("filter_scaleoffset", "filter_scaleoffset",
                 "scale-offset, library-computed minimum bits",
                 scaleoffset=0),
]


# --------------------------------------------------------------------------
# fill values
# --------------------------------------------------------------------------


def gen_fill_default(path):
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("data", (16,), chunks=(4,), dtype="<i4")


def gen_fill_set_int(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset("data", (16,), chunks=(4,), dtype="<i4", fillvalue=-1)
        ds[0:4] = ramp("<i4", 4)


def gen_fill_set_float_nan(path):
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            "data", (16,), chunks=(4,), dtype="<f8", fillvalue=np.float64("nan")
        )


FILL_CASES = [
    Case("fill_default", "fillvalue", gen_fill_default, "fill_default",
         "default (zero) fill, nothing written"),
    Case("fill_set_int", "fillvalue", gen_fill_set_int, "fill_set_int",
         "user-defined integer fill, first chunk written"),
    Case("fill_set_float_nan", "fillvalue", gen_fill_set_float_nan,
         "fill_set_float_nan", "user-defined NaN fill"),
]


# --------------------------------------------------------------------------
# dataspaces
# --------------------------------------------------------------------------


def gen_space_scalar(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.int32(42))


def gen_space_null(path):
    with h5py.File(path, "w") as f:
        f["data"] = h5py.Empty("<i4")


def gen_space_zerosized(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", (0,), dtype="<i4")


def gen_space_unlimited_resized(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset("data", (4,), maxshape=(None,), chunks=(4,), dtype="<i4")
        ds[...] = ramp("<i4", 4)
        ds.resize((12,))
        ds[4:12] = ramp("<i4", 8) + 100


SPACE_CASES = [
    Case("space_scalar", "dataspace", gen_space_scalar, "space_scalar",
         "scalar (rank 0) dataspace"),
    Case("space_null", "dataspace", gen_space_null, "space_null",
         "NULL dataspace — no elements at all"),
    Case("space_zerosized", "dataspace", gen_space_zerosized, "space_zerosized",
         "simple dataspace with a zero-length dimension"),
    Case("space_unlimited_resized", "dataspace", gen_space_unlimited_resized,
         "space_unlimited_resized", "unlimited maxshape, grown after creation"),
]


# --------------------------------------------------------------------------
# groups, links
# --------------------------------------------------------------------------


def gen_groups_nested(path):
    with h5py.File(path, "w") as f:
        g = f.create_group("a")
        h = g.create_group("b")
        h.create_group("c")
        h.create_dataset("leaf", data=ramp("<i4"))
        f.create_dataset("top", data=ramp("<i4"))


def gen_link_hard(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("orig", data=ramp("<i4"))
        f["alias"] = f["orig"]


def gen_link_soft(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("orig", data=ramp("<i4"))
        f["alias"] = h5py.SoftLink("/orig")


def gen_link_external(path):
    target = path.parent / (path.stem + "_ext.h5")
    with h5py.File(target, "w") as g:
        g.create_dataset("payload", data=ramp("<i4"))
    with h5py.File(path, "w") as f:
        f.create_dataset("orig", data=ramp("<i4"))
        f["ext"] = h5py.ExternalLink(target.name, "/payload")


def gen_link_external_read(path):
    """A master file whose payload lives entirely in a sibling.

    Every dataset is reached only by crossing a link, so the `resolved` field
    is the whole content check: a reader that lists the links but never opens
    the other file matches on `target` and diverges here. The two dangling
    links pin the other half — a target file that is not there and a target
    object that is not there both have to say so rather than read something.
    """
    target = path.parent / (path.stem + "_data.h5")
    with h5py.File(target, "w") as g:
        g.create_dataset("top", data=ramp("<f8"))
        g.create_group("deep").create_dataset("inner", data=ramp("<i2"))
    with h5py.File(path, "w") as f:
        f["direct"] = h5py.ExternalLink(target.name, "/top")
        f["nested"] = h5py.ExternalLink(target.name, "/deep/inner")
        f["gone_object"] = h5py.ExternalLink(target.name, "/absent")
        f["gone_file"] = h5py.ExternalLink("no_such_file.h5", "/top")


def gen_links_dense(path):
    # v1.8 bounds, not "latest": dense link storage needs the v1.8 group
    # format, and stopping there keeps the v1.10 layout message out of the
    # file so this case isolates link storage.
    with h5py.File(path, "w", libver=("v108", "v108")) as f:
        g = f.create_group("g", track_order=True)
        for i in range(12):
            g.create_dataset("d%02d" % i, data=np.array([i], dtype="<i4"))


def gen_track_order(path):
    # Creation-order tracking adds a second index (a v2 B-tree keyed on
    # creation order) beside the name index, on both links and attributes.
    with h5py.File(path, "w", track_order=True) as f:
        for name in ("zebra", "apple", "mango"):
            f.create_group(name)
        for i, key in enumerate(("zeta", "alpha", "mu")):
            f.attrs.create(key, np.int32(i))
        g = f.create_group("g", track_order=True)
        g.create_dataset("data", data=ramp("<i4"))
        g.attrs.create("second", np.int32(2))
        g.attrs.create("first", np.int32(1))


def gen_group_storage_modern_root(path):
    """A symbol-table group, holding children, under a link-message root.

    `track_order` migrates the group that asks for it and nothing else, so one
    h5py call writes a root using link messages over a child still using the
    legacy symbol table. A reader that walks a group the way its parent is
    stored lists `legacy` and finds none of its children.
    """
    with h5py.File(path, "w", track_order=True) as f:
        legacy = f.create_group("legacy")
        legacy.create_dataset("a", data=ramp("<i4"))
        legacy.create_group("inner").create_dataset("c", data=ramp("<i2"))


def gen_group_storage_legacy_root(path):
    """The same mismatch the other way up: a link-message group, holding
    children, under a symbol-table root."""
    with h5py.File(path, "w") as f:
        f.create_group("legacy").create_dataset("a", data=ramp("<i4"))
        modern = f.create_group("modern", track_order=True)
        modern.create_dataset("b", data=ramp("<f8"))
        modern.create_group("inner").create_dataset("c", data=ramp("<i2"))


LINK_CASES = [
    Case("groups_nested", "group", gen_groups_nested, "groups_nested",
         "three levels of nested groups plus an empty leaf group"),
    Case("link_hard", "link", gen_link_hard, "link_hard",
         "two names for one object"),
    Case("link_soft", "link", gen_link_soft, "link_soft", "soft link to /orig"),
    Case("link_external", "link", gen_link_external, None,
         "external link into a sibling file",
         ext_files=("_ext.h5",)),
    Case("link_external_read", "link", gen_link_external_read, None,
         "datasets read through external links, plus a dangling object and a "
         "dangling file",
         ext_files=("_data.h5",)),
    Case("links_dense", "link", gen_links_dense, "links_dense",
         "12 links in one group — dense link storage (fractal heap + v2 B-tree)"),
    Case("track_order", "group", gen_track_order, "track_order",
         "creation-order indices on links and attributes"),
    Case("group_storage_modern_root", "group", gen_group_storage_modern_root, None,
         "symbol-table group with children under a link-message root"),
    Case("group_storage_legacy_root", "group", gen_group_storage_legacy_root, None,
         "link-message group with children under a symbol-table root"),
]


# --------------------------------------------------------------------------
# attributes
# --------------------------------------------------------------------------


def gen_attr_scalar_num(path):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=ramp("<i4"))
        ds.attrs.create("gain", np.float64(2.5))
        ds.attrs.create("count", np.int32(7))


def gen_attr_array_num(path):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=ramp("<i4"))
        ds.attrs.create("offsets", np.arange(4, dtype="<i4"))
        ds.attrs.create("matrix", np.arange(6, dtype="<f8").reshape(2, 3))


def gen_attr_string(path):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=ramp("<i4"))
        ds.attrs.create("units", "volt", dtype=h5py.string_dtype("utf-8"))
        f.create_group("g").attrs.create(
            "NX_class", "NXdetector", dtype=h5py.string_dtype("utf-8")
        )


def gen_attrs_dense(path):
    # v1.8 bounds for the same reason as links_dense: dense attribute storage
    # arrives with v1.8, and staying there isolates it from the v1.10 layout
    # message.
    with h5py.File(path, "w", libver=("v108", "v108")) as f:
        ds = f.create_dataset("data", data=ramp("<i4"))
        for i in range(12):
            ds.attrs.create("a%02d" % i, np.int32(i))


def gen_attrs_dense_group(path):
    # The same phase change on a group and on the root group, where the
    # attributes share their object header with the link messages rather than
    # with a dataset's layout.
    with h5py.File(path, "w", libver=("v108", "v108")) as f:
        g = f.create_group("g")
        for i in range(12):
            g.attrs.create("g%02d" % i, np.int32(i))
        for i in range(12):
            f.attrs.create("r%02d" % i, np.int32(i))
        f.create_dataset("data", data=ramp("<i4"))


def gen_attr_on_root(path):
    with h5py.File(path, "w") as f:
        f.attrs.create("title", "root", dtype=h5py.string_dtype("utf-8"))
        f.attrs.create("version", np.int64(3))
        f.create_dataset("data", data=ramp("<i4"))


def gen_attr_large(path):
    # One attribute past the 64 KiB object-header message limit: the value
    # spills to dense storage no matter how few attributes there are.
    with h5py.File(path, "w", libver=("v108", "v108")) as f:
        ds = f.create_dataset("data", data=ramp("<i4"))
        ds.attrs.create("big", np.arange(25600, dtype="<i4"))


ATTR_CASES = [
    Case("attr_scalar_num", "attribute", gen_attr_scalar_num, "attr_scalar_num",
         "scalar f64 and i32 attributes"),
    Case("attr_array_num", "attribute", gen_attr_array_num, "attr_array_num",
         "1-D and 2-D numeric attributes"),
    Case("attr_string", "attribute", gen_attr_string, "attr_string",
         "vlen UTF-8 string attributes on a dataset and a group"),
    Case("attrs_dense", "attribute", gen_attrs_dense, "attrs_dense",
         "12 attributes — dense attribute storage"),
    Case("attrs_dense_group", "attribute", gen_attrs_dense_group,
         "attrs_dense_group",
         "12 attributes on a group and on the root — dense storage"),
    Case("attr_on_root", "attribute", gen_attr_on_root, "attr_on_root",
         "attributes on the root group"),
    Case("attr_large", "attribute", gen_attr_large, "attr_large",
         "single 100 KiB attribute — dense storage forced by size, not count"),
]


# --------------------------------------------------------------------------
# library version bounds / superblock
# --------------------------------------------------------------------------


def _libver_case(name, libver, rust, note):
    def gen(path):
        with h5py.File(path, "w", libver=libver) as f:
            f.create_dataset("data", data=ramp("<i4"))
            f.create_group("g")

    return Case(name, "superblock", gen, rust, note)


def gen_userblock(path):
    """A 512-byte userblock in front of the superblock.

    The block is filled with text afterwards, as an application that keeps a
    script or a header there would: the reader has to find the superblock at
    512 rather than at 0, and must not mistake the block's bytes for metadata.
    """
    with h5py.File(path, "w", userblock_size=512) as f:
        f.create_dataset("data", data=ramp("<i4"))
        f.create_group("g")
    prefix = b"#!/bin/sh\n# userblock\n"
    with open(path, "r+b") as fh:
        fh.write(prefix + b"#" * (512 - len(prefix) - 1) + b"\n")


LIBVER_CASES = [
    _libver_case("libver_earliest", "earliest", "libver_earliest",
                 "libver earliest — superblock v0, symbol-table groups"),
    _libver_case("libver_v108", ("v108", "v108"), None, "libver v1.8 bounds"),
    _libver_case("libver_v110", ("v110", "v110"), None, "libver v1.10 bounds"),
    _libver_case("libver_latest", "latest", "libver_latest",
                 "libver latest — superblock v3, new-style groups"),
    # Superblock v1 is not reachable from h5py 3.15: it is produced only by a
    # non-default B-tree K value (H5Pset_sym_k / H5Pset_istore_k), and neither
    # is wrapped on PropFCID. v0, v2 and v3 are covered by the four cases
    # above; the user block below is the remaining v0 variant.
    # No rust writer arm: the public API cannot ask for a userblock.
    Case("userblock", "superblock", gen_userblock, None,
         "512-byte userblock — the superblock, and every address, is based at 512"),
]


# --------------------------------------------------------------------------
# SWMR and bulk
# --------------------------------------------------------------------------


def gen_swmr_created(path):
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset("stream", (0, 4), maxshape=(None, 4), chunks=(1, 4),
                              dtype="<f4")
        f.swmr_mode = True
        for i in range(8):
            ds.resize((i + 1, 4))
            ds[i, :] = np.arange(i * 4, i * 4 + 4, dtype="<f4")
            ds.flush()


def gen_large_multi_mb(path):
    with h5py.File(path, "w", libver="latest") as f:
        data = np.arange(512 * 512, dtype="<f8").reshape(512, 512)
        f.create_dataset("big", data=data, chunks=(64, 512))


MISC_CASES = [
    Case("swmr_created", "swmr", gen_swmr_created, "swmr_created",
         "file created through the SWMR writer path and appended frame by frame"),
    Case("large_multi_mb", "bulk", gen_large_multi_mb, "large_multi_mb",
         "2 MiB chunked f64 dataset — payload compared by SHA-256"),
]


# --------------------------------------------------------------------------
# checked-in fixtures
#
# Some file-level features have no h5py binding at all, so the reference file
# cannot be written from Python. Those come from a C generator run against the
# pinned libhdf5 (`tests/fixtures/gen_*.sh`), are checked in, and are copied
# into the run directory here. h5py still reads them, so direction A compares
# exactly as it does for a generated case.
# --------------------------------------------------------------------------

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _fixture_case(name, fixture, generator, group, note):
    def gen(path):
        src = FIXTURE_DIR / fixture
        if not src.exists():
            raise FileNotFoundError(
                "%s is missing; regenerate it with tests/fixtures/%s"
                % (src, generator)
            )
        shutil.copyfile(src, path)

    # No rust writer arm: the public API cannot ask for these files.
    return Case(name, group, gen, None, note)


FIXTURE_CASES = [
    _fixture_case(
        "sohm_list", "sohm_list.h5", "gen_sohm.sh", "sohm",
        "shared datatype/dataspace/attribute messages, list index "
        "(H5Pset_shared_mesg_index) + a committed datatype",
    ),
    _fixture_case(
        "sohm_btree", "sohm_btree.h5", "gen_sohm.sh", "sohm",
        "the same file with the shared-message index forced to a v2 B-tree",
    ),
    _fixture_case(
        "ochk_root", "ochk_root.h5", "gen_ochk.sh", "objectheader",
        "root group whose object header spills into two continuation chunks",
    ),
]


# --------------------------------------------------------------------------

ALL_CASES = (
    INT_CASES
    + FLOAT_CASES
    + STRING_CASES
    + COMPOSITE_CASES
    + LAYOUT_CASES
    + FILTER_CASES
    + FILL_CASES
    + SPACE_CASES
    + LINK_CASES
    + ATTR_CASES
    + LIBVER_CASES
    + MISC_CASES
    + FIXTURE_CASES
)


def by_name(name):
    for c in ALL_CASES:
        if c.name == name:
            return c
    raise KeyError(name)


if __name__ == "__main__":
    print("%d cases" % len(ALL_CASES))
    for c in ALL_CASES:
        print("  %-24s %-16s rust=%s" % (c.name, c.group, c.rust or "-"))
