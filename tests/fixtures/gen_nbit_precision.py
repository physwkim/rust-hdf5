#!/usr/bin/env python3
"""Generate `nbit_precision.h5` for `tests/nbit_precision.rs`.

Each dataset stores integers whose datatype keeps fewer significant bits
than its width (`H5Tset_precision`, `H5Tset_offset`), as NDFileHDF5 writes
for its N-bit compression. Shape (3, 4, 5); the chunked ones are chunked
(1, 4, 5) along an unlimited first axis, so libver="latest" indexes them
with an extensible array, as an SWMR writer does.

Each value is the `values(i)` given in `DATASETS` (or `ATTRIBUTES`, shape
(5,) on the root group) for the element's row-major index `i`; libhdf5
converts it into the reduced type on write, and the script checks that
libhdf5 reads it back unchanged.

`ENUMS` are enums over such a reduced type (and `ENUM_ATTRIBUTES` likewise,
shape (5,)): element `i` is member `i % len(members)`, whose value libhdf5
stores in the reduced base's bits. The script checks that libhdf5 reads each
back both as the enum and as the base's native integer type
(`H5T__conv_enum_numeric`), the read the Rust tests compare against.

Regenerate with (h5py over libhdf5):
    cd tests/fixtures && python3 gen_nbit_precision.py
"""
import ctypes

import h5py
import numpy as np


def libhdf5():
    """The libhdf5 h5py runs on: h5py wraps neither H5Tset_precision nor
    H5Tset_offset for a bit field."""
    with open("/proc/self/maps") as maps:
        path = next(line.split()[-1] for line in maps if "/libhdf5.so" in line)
    lib = ctypes.CDLL(path)
    for fn in (lib.H5Tset_precision, lib.H5Tset_offset):
        fn.argtypes = (ctypes.c_int64, ctypes.c_size_t)
        fn.restype = ctypes.c_int
    return lib


H5 = libhdf5()

SHAPE = (3, 4, 5)
N = int(np.prod(SHAPE))

# name -> (base type, precision, offset, layout, values(i), memory type)
#   layout: "nbit" (chunked + N-bit filter), "chunked", "contig"
DATASETS = {
    "u12_o0_nbit": (h5py.h5t.STD_U16LE, 12, 0, "nbit", lambda i: (i * 37 + 5) % 4096, None),
    "u12_o4_nbit": (h5py.h5t.STD_U16LE, 12, 4, "nbit", lambda i: (i * 37 + 5) % 4096, None),
    "s12_o0_nbit": (h5py.h5t.STD_I16LE, 12, 0, "nbit", lambda i: (i * 37) % 4096 - 2048, None),
    "s12_o3_nbit": (h5py.h5t.STD_I16LE, 12, 3, "nbit", lambda i: (i * 37) % 4096 - 2048, None),
    "s27_o2_nbit": (
        h5py.h5t.STD_I32LE, 27, 2, "nbit",
        lambda i: (i * 1_000_003) % (1 << 27) - (1 << 26), None,
    ),
    "u12_o4_chunked": (h5py.h5t.STD_U16LE, 12, 4, "chunked", lambda i: (i * 37 + 5) % 4096, None),
    "s12_o3_chunked": (h5py.h5t.STD_I16LE, 12, 3, "chunked", lambda i: (i * 37) % 4096 - 2048, None),
    "u12_o4_contig": (h5py.h5t.STD_U16LE, 12, 4, "contig", lambda i: (i * 37 + 5) % 4096, None),
    "s11_o5_be_contig": (
        h5py.h5t.STD_I16BE, 11, 5, "contig", lambda i: (i * 37) % 2048 - 1024, None,
    ),
    "b12_o4_contig": (
        h5py.h5t.STD_B16LE, 12, 4, "contig", lambda i: (i * 41 + 3) % 4096, h5py.h5t.NATIVE_B16,
    ),
}

# name -> (base type, precision, offset, values(i), memory type)
ATTRIBUTES = {
    "s12_o3": (h5py.h5t.STD_I16LE, 12, 3, lambda i: i * 1000 - 2048, None),
    "b12_o4": (h5py.h5t.STD_B16LE, 12, 4, lambda i: i * 1000 + 7, h5py.h5t.NATIVE_B16),
}

# name -> (base type, precision, offset, layout, member values)
ENUMS = {
    "e_s12_o3_nbit": (
        h5py.h5t.STD_I16LE, 12, 3, "nbit", (-2048, -1000, -1, 0, 1, 7, 1500, 2047),
    ),
    "e_u12_o4_be_contig": (
        h5py.h5t.STD_U16BE, 12, 4, "contig", (0, 1, 9, 300, 2048, 4000, 4095),
    ),
}

# name -> (base type, precision, offset, member values)
ENUM_ATTRIBUTES = {
    "e_s12_o3": (h5py.h5t.STD_I16LE, 12, 3, (-2048, -1000, -1, 0, 2047)),
}


def reduced(base, precision, offset):
    tid = base.copy()
    assert H5.H5Tset_precision(tid.id, precision) >= 0
    assert H5.H5Tset_offset(tid.id, offset) >= 0
    return tid


def dtype_of(base):
    signed = base.get_class() == h5py.h5t.INTEGER and base.get_sign() == h5py.h5t.SGN_2
    return np.dtype(f"{'i' if signed else 'u'}{base.get_size()}")


def enum_of(base, members):
    tid = h5py.h5t.enum_create(base)
    for k, v in enumerate(members):
        tid.enum_insert(f"m{k}".encode(), v)
    return tid


def create(h5, name, base, precision, offset, layout, values, mtype):
    tid = reduced(base, precision, offset)
    data = np.array([values(i) for i in range(N)], dtype=dtype_of(base)).reshape(SHAPE)
    write(h5, name, tid, layout, data, mtype)


def create_enum(h5, name, base, precision, offset, layout, members):
    tid = enum_of(reduced(base, precision, offset), members)
    native = dtype_of(base)
    mtype = enum_of(h5py.h5t.py_create(native), members)
    data = np.array([members[i % len(members)] for i in range(N)], dtype=native).reshape(SHAPE)
    dsid = write(h5, name, tid, layout, data, mtype)
    back = np.empty_like(data)
    dsid.read(h5py.h5s.ALL, h5py.h5s.ALL, back, mtype=h5py.h5t.py_create(native))
    assert (back == data).all(), name


def write(h5, name, tid, layout, data, mtype):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    if layout == "contig":
        space = h5py.h5s.create_simple(SHAPE)
    else:
        space = h5py.h5s.create_simple(SHAPE, (h5py.h5s.UNLIMITED,) + SHAPE[1:])
        dcpl.set_chunk((1,) + SHAPE[1:])
        if layout == "nbit":
            # H5Pset_nbit, which h5py does not wrap.
            dcpl.set_filter(h5py.h5z.FILTER_NBIT, h5py.h5z.FLAG_OPTIONAL)
    dsid = h5py.h5d.create(h5.id, name.encode(), tid, space, dcpl=dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, data, mtype=mtype)
    back = np.empty_like(data)
    dsid.read(h5py.h5s.ALL, h5py.h5s.ALL, back, mtype=mtype)
    assert (back == data).all(), name
    if layout == "nbit":
        assert dsid.get_create_plist().get_filter(0)[0] == h5py.h5z.FILTER_NBIT, name
    return dsid


def create_attr(h5, name, base, precision, offset, values, mtype):
    tid = reduced(base, precision, offset)
    data = np.array([values(i) for i in range(5)], dtype=dtype_of(base))
    aid = h5py.h5a.create(h5.id, name.encode(), tid, h5py.h5s.create_simple((5,)))
    aid.write(data, mtype=mtype)
    back = np.empty_like(data)
    aid.read(back, mtype=mtype)
    assert (back == data).all(), name


def create_enum_attr(h5, name, base, precision, offset, members):
    tid = enum_of(reduced(base, precision, offset), members)
    native = dtype_of(base)
    data = np.array(members, dtype=native)
    aid = h5py.h5a.create(h5.id, name.encode(), tid, h5py.h5s.create_simple((len(members),)))
    aid.write(data, mtype=enum_of(h5py.h5t.py_create(native), members))
    back = np.empty_like(data)
    aid.read(back, mtype=h5py.h5t.py_create(native))
    assert (back == data).all(), name


def main():
    with h5py.File("nbit_precision.h5", "w", libver="latest") as h5:
        for name, spec in DATASETS.items():
            create(h5, name, *spec)
        for name, spec in ATTRIBUTES.items():
            create_attr(h5, name, *spec)
        for name, spec in ENUMS.items():
            create_enum(h5, name, *spec)
        for name, spec in ENUM_ATTRIBUTES.items():
            create_enum_attr(h5, name, *spec)
    print("generated nbit_precision.h5")


if __name__ == "__main__":
    main()
