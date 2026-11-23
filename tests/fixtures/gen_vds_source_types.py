#!/usr/bin/env python3
"""Generate `vds_source_types.h5` and its external source file
`vds_source_types_src.h5` for `tests/vds_source_types.rs`.

Each virtual dataset maps row `r` of its (rows, 5) extent from the whole of
a 1-D source of 5 elements whose datatype is not the virtual dataset's own:
another precision, bit offset, byte order, size or sign, some behind the
N-bit filter, one in another file, one itself a virtual dataset. A row with
no source reads back as the virtual dataset's fill value.

The oracle the Rust tests compare against is the plain dataset
`expected/<name>`: what libhdf5 reads from each virtual dataset into the
virtual dataset's own datatype, converted to its native type. Reading in
the dataset's own datatype matters: libhdf5 converts each source straight
to the memory type (H5Dvirtual.c:2683), so a wider memory type would see
source values the virtual datatype cannot hold. For the same reason a
nested virtual dataset's oracle is its source's own-type values converted
to its datatype.

Regenerate with (h5py over libhdf5):
    cd tests/fixtures && python3 gen_vds_source_types.py
"""
import ctypes

import h5py
import numpy as np

T = h5py.h5t


def libhdf5():
    """The libhdf5 h5py runs on, for the H5Tset_precision and
    H5Tset_offset h5py does not wrap."""
    with open("/proc/self/maps") as maps:
        path = next(line.split()[-1] for line in maps if "/libhdf5.so" in line)
    lib = ctypes.CDLL(path)
    for fn in (lib.H5Tset_precision, lib.H5Tset_offset):
        fn.argtypes = (ctypes.c_int64, ctypes.c_size_t)
        fn.restype = ctypes.c_int
    return lib


H5 = libhdf5()
N = 5


def reduced(base, precision, offset):
    tid = base.copy()
    assert H5.H5Tset_precision(tid.id, precision) >= 0
    assert H5.H5Tset_offset(tid.id, offset) >= 0
    return tid


def native(tid):
    size = tid.get_size()
    if tid.get_class() == T.FLOAT:
        return np.dtype(f"f{size}")
    signed = tid.get_class() == T.INTEGER and tid.get_sign() == T.SGN_2
    return np.dtype(f"{'i' if signed else 'u'}{size}")


# name -> (datatype, values, N-bit filtered)
SOURCES = {
    "s_i16le_12_3": (reduced(T.STD_I16LE, 12, 3), (-2048, -5, 0, 77, 2047), False),
    "s_i16be_11_5_nbit": (reduced(T.STD_I16BE, 11, 5), (-1024, -1, 1, 500, 1023), True),
    "s_i16be": (T.STD_I16BE, (-32768, -300, 0, 300, 32767), False),
    "s_i32le": (T.STD_I32LE, (-40000, -32768, 0, 32767, 40000), False),
    "s_u16le_12_4": (reduced(T.STD_U16LE, 12, 4), (0, 1, 2048, 4000, 4095), False),
    "s_u16be_10_0_nbit": (reduced(T.STD_U16BE, 10, 0), (0, 1, 512, 1000, 1023), True),
    "s_u16le": (T.STD_U16LE, (0, 4095, 4096, 50000, 65535), False),
    "s_f32be": (T.IEEE_F32BE, (-1.5, 0.0, 0.1, 1e30, -2.5e-3), False),
    "s_f64le": (T.IEEE_F64LE, (0.1, -1e39, 1e39, 3.4028235e38, 1e-50), False),
}

# Sources in `vds_source_types_src.h5`.
EXTERNAL_SOURCES = {
    "s_i8": (T.STD_I8LE, (-128, -1, 0, 1, 127), False),
}

SRC_FILE = "vds_source_types_src.h5"

# name -> (datatype, fill value, rows); a row is (file, source) or None.
VIRTUAL = {
    "v_i16": (
        T.STD_I16LE,
        -7,
        [
            (".", "s_i16le_12_3"),
            (".", "s_i16be_11_5_nbit"),
            (".", "s_i16be"),
            (".", "s_i32le"),
            (".", "s_u16le_12_4"),
            (SRC_FILE, "s_i8"),
            None,
        ],
    ),
    "v_u16_12_4": (
        reduced(T.STD_U16LE, 12, 4),
        100,
        [
            (".", "s_u16le_12_4"),
            (".", "s_u16be_10_0_nbit"),
            (".", "s_u16le"),
            None,
        ],
    ),
    "v_f32": (T.IEEE_F32LE, 0.5, [(".", "s_f32be"), (".", "s_f64le"), None]),
    "v_f64_of_i32": (T.IEEE_F64LE, 0.0, [(".", "s_i32le")]),
}

# A virtual dataset whose source is the whole of another one.
NESTED = {"v_i32be_of_v_i16": (T.STD_I32BE, 9, "v_i16")}

# Virtual datasets libhdf5 has no conversion path into (a bit field from an
# integer), so no read of them succeeds.
REFUSED = {"v_b16_of_i16": (T.STD_B16LE, 0, [(".", "s_i16be")])}


def create_source(h5, name, tid, values, nbit):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    if nbit:
        dcpl.set_chunk((N,))
        dcpl.set_filter(h5py.h5z.FILTER_NBIT, h5py.h5z.FLAG_OPTIONAL)
    space = h5py.h5s.create_simple((N,))
    dsid = h5py.h5d.create(h5.id, name.encode(), tid, space, dcpl=dcpl)
    data = np.array(values, dtype=native(tid))
    mtype = h5py.h5t.py_create(data.dtype)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, data, mtype=mtype)
    back = np.empty_like(data)
    dsid.read(h5py.h5s.ALL, h5py.h5s.ALL, back, mtype=mtype)
    assert (back == data).all(), name


def virtual_dcpl(tid, fill):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_fill_value(np.array(fill, dtype=native(tid)))
    return dcpl


def create_virtual(h5, name, tid, fill, rows):
    dcpl = virtual_dcpl(tid, fill)
    vspace = h5py.h5s.create_simple((len(rows), N))
    for r, row in enumerate(rows):
        if row is None:
            continue
        vspace.select_hyperslab((r, 0), (1, N))
        file, source = row
        dcpl.set_virtual(vspace, file.encode(), source.encode(), h5py.h5s.create_simple((N,)))
    h5py.h5d.create(h5.id, name.encode(), tid, h5py.h5s.create_simple((len(rows), N)), dcpl=dcpl)


def create_nested(h5, name, tid, fill, source):
    shape = h5[source].shape
    dcpl = virtual_dcpl(tid, fill)
    dcpl.set_virtual(
        h5py.h5s.create_simple(shape), b".", source.encode(), h5py.h5s.create_simple(shape)
    )
    h5py.h5d.create(h5.id, name.encode(), tid, h5py.h5s.create_simple(shape), dcpl=dcpl)


def own_type_values(h5, name):
    """The elements of virtual dataset `name` as libhdf5 reads them into
    its own datatype, and that datatype."""
    dsid = h5[name].id
    tid = dsid.get_type()
    n = int(np.prod(dsid.shape))
    if name in NESTED:
        inner, inner_tid = own_type_values(h5, NESTED[name][2])
        buf = np.zeros(n * max(tid.get_size(), inner_tid.get_size()), dtype=np.uint8)
        buf[: inner.size] = inner
        h5py.h5t.convert(inner_tid, tid, n, buf)
        return buf[: n * tid.get_size()], tid
    buf = np.empty(n * tid.get_size(), dtype=np.uint8)
    dsid.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=tid)
    return buf, tid


def record_expected(h5, name):
    raw, tid = own_type_values(h5, name)
    out = np.empty(h5[name].shape, dtype=native(tid))
    buf = np.zeros(max(raw.size, out.nbytes), dtype=np.uint8)
    buf[: raw.size] = raw
    h5py.h5t.convert(tid, h5py.h5t.py_create(out.dtype), out.size, buf)
    out.flat[:] = np.frombuffer(buf[: out.nbytes].tobytes(), dtype=out.dtype)
    h5.create_dataset(f"expected/{name}", data=out)
    print(name, out.tolist())


def main():
    with h5py.File(SRC_FILE, "w", libver="latest") as h5:
        for name, spec in EXTERNAL_SOURCES.items():
            create_source(h5, name, *spec)
    with h5py.File("vds_source_types.h5", "w", libver="latest") as h5:
        for name, spec in SOURCES.items():
            create_source(h5, name, *spec)
        for name, spec in VIRTUAL.items():
            create_virtual(h5, name, *spec)
        for name, spec in NESTED.items():
            create_nested(h5, name, *spec)
        for name, spec in REFUSED.items():
            create_virtual(h5, name, *spec)
    with h5py.File("vds_source_types.h5", "r+") as h5:
        for name in list(VIRTUAL) + list(NESTED):
            record_expected(h5, name)
        for name in REFUSED:
            try:
                own_type_values(h5, name)
            except Exception:
                continue
            raise AssertionError(f"libhdf5 read {name}")


if __name__ == "__main__":
    main()
