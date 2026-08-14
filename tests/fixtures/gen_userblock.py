#!/usr/bin/env python3
"""Generate `userblock.h5`, used by `tests/userblock.rs`.

A userblock is an application-owned prefix of the file: the HDF5 superblock
starts after it, at a power-of-two offset that is at least 512, and every
address in the file is measured from there. The reader has to find the
signature the way `H5FD_locate_signature` does (probe 0, then 512, 1024, ...)
instead of assuming offset 0.

The block is filled with a shebang line and padding after h5py closes the
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
PATH = pathlib.Path(__file__).resolve().parent / "userblock.h5"
PAYLOAD = np.arange(8, dtype=np.int32)

with h5py.File(PATH, "w", userblock_size=USERBLOCK) as f:
    f.create_dataset("data", data=PAYLOAD)
    f.attrs["note"] = "the superblock starts at 512"

prefix = b"#!/bin/sh\n# userblock: this file is also an HDF5 file\n"
with open(PATH, "r+b") as fh:
    fh.write(prefix + b"#" * (USERBLOCK - len(prefix) - 1) + b"\n")

with h5py.File(PATH) as f:
    assert f.userblock_size == USERBLOCK, f.userblock_size
    assert (f["data"][:] == PAYLOAD).all()
    assert f.attrs["note"] == "the superblock starts at 512"
print("generated %s" % PATH)
