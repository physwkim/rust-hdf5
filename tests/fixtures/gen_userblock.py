#!/usr/bin/env python3
"""Generate `userblock.h5` and `userblock_v0.h5`, used by `tests/userblock.rs`.

A userblock is an application-owned prefix of the file: the HDF5 superblock
starts after it, at a power-of-two offset that is at least 512, and every
address in the file is measured from there. The reader has to find the
signature the way `H5FD_locate_signature` does (probe 0, then 512, 1024, ...)
instead of assuming offset 0.

  userblock.h5     v1.8 bounds — superblock v2, which the append path accepts,
                   so this one covers reading *and* appending behind a
                   userblock.
  userblock_v0.h5  default bounds — superblock v0, so the signature search runs
                   before version detection on the classic path too.

Each block is filled with a shebang line and padding after h5py closes the
file, which is what makes a userblock worth having: the result is both a
runnable script and a valid HDF5 file. Nothing in it may look like a
superblock, so the padding is plain text.

Regenerate with:  python3 tests/fixtures/gen_userblock.py
Requires h5py (libhdf5); the pinned one lives in the `tomo` env:

    ~/micromamba/envs/tomo/bin/python tests/fixtures/gen_userblock.py
"""
import pathlib

import h5py
import numpy as np

USERBLOCK = 512
HERE = pathlib.Path(__file__).resolve().parent
PAYLOAD = np.arange(8, dtype=np.int32)
NOTE = "the superblock starts at 512"


def write(name, libver, want_superblock):
    path = HERE / name
    with h5py.File(path, "w", userblock_size=USERBLOCK, libver=libver) as f:
        f.create_dataset("data", data=PAYLOAD)
        f.attrs["note"] = NOTE

    prefix = b"#!/bin/sh\n# userblock: this file is also an HDF5 file\n"
    with open(path, "r+b") as fh:
        fh.write(prefix + b"#" * (USERBLOCK - len(prefix) - 1) + b"\n")

    raw = path.read_bytes()
    assert raw[USERBLOCK:USERBLOCK + 8] == b"\x89HDF\r\n\x1a\n", name
    assert raw[USERBLOCK + 8] == want_superblock, (name, raw[USERBLOCK + 8])
    with h5py.File(path) as f:
        assert f.userblock_size == USERBLOCK, f.userblock_size
        assert (f["data"][:] == PAYLOAD).all()
        assert f.attrs["note"] == NOTE
    print("generated %s (superblock v%d)" % (path, want_superblock))


# v1.8 bounds keep the data layout message at version 3 (version 4, which
# "latest" would select, is not read by this crate) while lifting the
# superblock to version 2, which is what the append path requires.
write("userblock.h5", ("v108", "v108"), 2)
write("userblock_v0.h5", "earliest", 0)
