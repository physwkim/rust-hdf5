#!/usr/bin/env python3
"""Generate `blosc_c1120.bin`, the c-blosc frames the blosc filter tests
decode, and check rust-hdf5's own frames against the same library.

Every frame is made by c-blosc 1.12.0 (the copy ADSupport builds) through
`blosc_compress_ctx` with a destination of `nbytes + BLOSC_MAX_OVERHEAD`
bytes, blocksize 0 (computed) and one thread. The inputs are not stored:
`make_input` is deterministic and `blosc_frames_from_c_blosc_1_12` in
src/format/messages/filter.rs builds the same bytes.

  python3 tests/fixtures/gen_blosc.py LIBBLOSC gen tests/fixtures/blosc_c1120.bin
  python3 tests/fixtures/gen_blosc.py LIBBLOSC check RS_FRAMES.bin

`check` reads a container in the same format, decodes every frame with
LIBBLOSC (any 1.x) and compares the result with `make_input`; it also
reports which frames are byte-identical to the C frame made from the same
parameters. The snappy 1.1.1 that c-blosc 1.12.0 bundles decodes some
streams wrong, its own included; a later libblosc decodes them.

Container: b"BLOSCFX1", then per record the little-endian fields
u8 compcode, u8 shuffle, u16 typesize, u8 clevel, u8 kind, u16 pad,
u32 nbytes, u32 frame length, frame bytes.
"""
import ctypes
import os
import struct
import sys

SMOOTH, RANDOM, MIXED = 0, 1, 2
NAMES = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
HCR = {2, 4, 5}  # lz4hc, zlib, zstd: compute_blocksize doubles their block
MAGIC = b"BLOSCFX1"
HDR = "<BBHBBHII"


def make_input(kind, nbytes, typesize):
    x = (0x9E3779B9 ^ nbytes ^ (typesize << 20) ^ (kind << 28)) & 0xFFFFFFFF

    def step():
        nonlocal x
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        return x >> 16

    out = bytearray(nbytes)
    if kind == RANDOM:
        for i in range(nbytes):
            out[i] = step() & 0xFF
        return bytes(out)
    w = min(typesize, 8)
    n = nbytes // w
    for i in range(n):
        v = ((i // 7) % 50) * 3 + (1 if step() & 1023 == 0 else 0)
        out[i * w:(i + 1) * w] = v.to_bytes(8, "little")[:w]
    for i in range(n * w, nbytes):
        out[i] = 0xA5
    if kind == MIXED:
        for i in range(nbytes - 1500, nbytes):
            out[i] = step() & 0xFF
    return bytes(out)


def records():
    for c in range(6):
        big_clevel = 3 if c in HCR else 5  # 32 KiB blocks for every codec
        for shuffle in (0, 1, 2):
            for ts in (1, 2, 4, 8):
                yield c, shuffle, ts, 5, SMOOTH, 2000
                yield c, shuffle, ts, big_clevel, SMOOTH, 34000
            yield c, shuffle, 4, big_clevel, MIXED, 34000
        yield c, 1, 4, 5, SMOOTH, 100        # below MIN_BUFFERSIZE: memcpy
        yield c, 1, 4, 5, RANDOM, 2000       # incompressible: memcpy
        yield c, 0, 4, 0, SMOOTH, 2000       # clevel 0: memcpy
        yield c, 2, 4, 5, SMOOTH, 2004       # 501 elements: no bit transpose
        yield c, 2, 4, big_clevel, SMOOTH, 34012  # leftover block likewise
        yield c, 1, 3, 5, SMOOTH, 34001      # odd typesize, ragged tail
        yield c, 1, 16, 5, SMOOTH, 34000     # MAX_SPLITS
        yield c, 1, 32, 5, SMOOTH, 34000     # past MAX_SPLITS: one stream
        yield c, 1, 300, 5, SMOOTH, 3000     # > BLOSC_MAX_TYPESIZE: 1
    for clevel in range(1, 10):
        yield 0, 1, 4, clevel, SMOOTH, 34000
    yield 0, 1, 4, 9, SMOOTH, 70000          # one block past 64 KiB
    # past BLOSC_POOL_MIN_DECODE: the `parallel` feature's block pool
    for c in range(6):
        yield c, 1, 4, 3 if c in HCR else 5, SMOOTH, 1050000
    yield 0, 2, 4, 5, SMOOTH, 1050000
    yield 0, 1, 4, 5, MIXED, 1050000


def load(path):
    L = ctypes.CDLL(path)
    L.blosc_compress_ctx.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_int]
    L.blosc_decompress_ctx.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    L.blosc_get_version_string.restype = ctypes.c_char_p
    return L


def c_compress(L, c, shuffle, ts, clevel, src):
    dst = ctypes.create_string_buffer(len(src) + 16)
    n = L.blosc_compress_ctx(clevel, shuffle, ts, len(src), src, dst,
                             len(src) + 16, NAMES[c].encode(), 0, 1)
    assert n > 0, (c, shuffle, ts, clevel, len(src), n)
    return dst.raw[:n]


def split_stats(frame):
    """(compressed, stored raw) split streams of a frame; memcpy frames
    count as one raw stream."""
    flags, ts = frame[2], frame[3]
    nbytes, bsize = struct.unpack_from("<II", frame, 4)
    if flags & 2:
        return 0, 1
    nblocks = -(-nbytes // bsize)
    comp = raw = 0
    for j in range(nblocks):
        pos = struct.unpack_from("<I", frame, 16 + 4 * j)[0]
        leftover = j == nblocks - 1 and nbytes % bsize
        b = nbytes % bsize if leftover else bsize
        split = (not flags & 0x10 and ts <= 16 and b // ts >= 128
                 and not leftover)
        ns = ts if split else 1
        for _ in range(ns):
            cs = struct.unpack_from("<i", frame, pos)[0]
            pos += 4 + cs
            if cs == b // ns:
                raw += 1
            else:
                comp += 1
    return comp, raw


def read(path):
    data = open(path, "rb").read()
    assert data[:8] == MAGIC
    pos, out = 8, []
    while pos < len(data):
        c, sh, ts, cl, kind, _, nb, fl = struct.unpack_from(HDR, data, pos)
        pos += struct.calcsize(HDR)
        out.append(((c, sh, ts, cl, kind, nb), data[pos:pos + fl]))
        pos += fl
    return out


def gen(L, path):
    assert L.blosc_get_version_string() == b"1.12.0"
    blob = bytearray(MAGIC)
    stats = {}
    for rec in records():
        c, sh, ts, cl, kind, nb = rec
        frame = c_compress(L, c, sh, ts, cl, make_input(kind, nb, ts))
        blob += struct.pack(HDR, c, sh, ts, cl, kind, 0, nb, len(frame))
        blob += frame
        s = stats.setdefault(NAMES[c], [0, 0, 0])
        comp, raw = split_stats(frame)
        s[0] += 1
        s[1] += comp
        s[2] += raw
    open(path, "wb").write(blob)
    print(f"{path}: {len(blob)} bytes")
    for name, (n, comp, raw) in stats.items():
        print(f"{name:8s} {n:3d} frames, {comp:4d} compressed streams, "
              f"{raw:3d} stored raw")


def check(L, path):
    here = os.path.dirname(os.path.abspath(__file__))
    want = dict(read(os.path.join(here, "blosc_c1120.bin")))
    rows = {}
    for rec, frame in read(path):
        c, sh, ts, cl, kind, nb = rec
        src = make_input(kind, nb, ts)
        out = ctypes.create_string_buffer(nb + 1)
        n = L.blosc_decompress_ctx(frame, out, nb + 1, 1)
        row = rows.setdefault(NAMES[c], [0, 0, 0, []])
        row[0] += 1
        if n == nb and out.raw[:nb] == src:
            row[1] += 1
        else:
            row[3].append((rec, n))
        if want.get(rec) == frame:
            row[2] += 1
    for name, (n, ok, same, bad) in rows.items():
        print(f"{name:8s} rs->C decoded {ok:3d}/{n:3d}  "
              f"byte-identical to C {same:3d}/{n:3d}")
        for rec, got in bad:
            print(f"    FAIL {rec}: blosc_decompress_ctx returned {got}")
    if any(r[3] for r in rows.values()):
        sys.exit(1)


if __name__ == "__main__":
    lib = load(sys.argv[1])
    {"gen": gen, "check": check}[sys.argv[2]](lib, sys.argv[3])
