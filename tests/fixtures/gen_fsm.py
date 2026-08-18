#!/usr/bin/env python3
"""Generate `fsm_persist.h5` and `fsm_persist_page.h5`, used by the
free-space-manager tests.

A file created with `H5Pset_file_space_strategy(strategy, persist=True, ...)`
keeps the space its own edits release across close, in an on-disk free-space
manager per allocation type. The superblock extension's file-space info
message names them; each is a header block (`FSHD`) pointing at a serialized
sections block (`FSSE`).

  fsm_persist.h5       H5F_FSPACE_STRATEGY_FSM_AGGR, the strategy this crate
                       writes managers for. Two managers end up populated:
                       metadata (`fs_addr[0]`) and raw data (`fs_addr[2]`),
                       because the sec2 driver's free-list map is dichotomous.
  fsm_persist_page.h5  H5F_FSPACE_STRATEGY_PAGE. Under the sec2 driver only
                       three of the twelve page-typed managers can be filled —
                       `H5MF__alloc_to_fs_type` sends every request of a page
                       or more to `H5F_MEM_PAGE_GENERIC` unless the driver
                       declares `H5FD_FEAT_PAGED_AGGR`, which sec2 does not —
                       so the message names `fs_addr[0]`, `fs_addr[2]` and
                       `fs_addr[6]`. Sections never cross a page boundary and
                       each page holds one kind of data.

The free space is real: each file is written, closed, reopened and has one
dataset deleted, so the manager records what the delete released rather than
an empty section list.

Regenerate with the pinned h5py (libhdf5 1.14.6):

    ~/micromamba/envs/tomo/bin/python tests/fixtures/gen_fsm.py
"""
import pathlib

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def write(name, strategy):
    path = HERE / name
    with h5py.File(
        path, "w", fs_strategy=strategy, fs_persist=True, fs_threshold=1
    ) as f:
        f.create_dataset("keep", data=np.arange(40, dtype=np.int32))
        f.create_dataset("drop", data=np.arange(40, dtype=np.float64))
        f.create_group("grp").create_dataset(
            "inner", data=np.arange(10, dtype=np.int32)
        )
    with h5py.File(path, "a") as f:
        del f["drop"]
    with h5py.File(path) as f:
        assert (f["keep"][:] == np.arange(40, dtype=np.int32)).all()
        assert (f["grp/inner"][:] == np.arange(10, dtype=np.int32)).all()
        assert "drop" not in f
    print("generated %s (%s)" % (path, strategy))


write("fsm_persist.h5", "fsm")
write("fsm_persist_page.h5", "page")
