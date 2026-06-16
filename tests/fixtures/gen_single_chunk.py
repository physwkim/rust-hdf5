#!/usr/bin/env python3
"""Generate the single-chunk filtered HDF5 fixtures used by
`tests/single_chunk_filtered.rs`.

Both datasets are 1-D `u8[64]` holding the bytes 0..64, chunked as a single
whole-array chunk (`chunks == shape`, fixed dims), so libhdf5 selects the
**Single Chunk** index. With a filter pipeline this stores the chunk's
on-disk size and a per-chunk filter mask inline in the data-layout message.

  single_chunk_deflate_mask0.h5 : dataset "y" written normally (gzip applied,
                                  filter_mask == 0).
  single_chunk_deflate_mask1.h5 : dataset "x" written via write_direct_chunk
                                  with the raw bytes and filter_mask == 1, so
                                  the gzip filter is marked "not applied" for
                                  this chunk and must be skipped on read.

Regenerate with:  python3 tests/fixtures/gen_single_chunk.py
Requires h5py (libhdf5). Deterministic output (no timestamps in the payload).
"""
import h5py
import numpy as np

PAYLOAD = np.arange(64, dtype=np.uint8)

# mask == 1: write the single chunk RAW, telling libhdf5 the gzip filter was
# not applied to it (bit 0 set).
with h5py.File("single_chunk_deflate_mask1.h5", "w") as f:
    d = f.create_dataset(
        "x", shape=(64,), dtype="u1", chunks=(64,),
        compression="gzip", compression_opts=4,
    )
    d.id.write_direct_chunk((0,), PAYLOAD.tobytes(), filter_mask=1)

# mask == 0: normal filtered write (gzip applied to the chunk).
with h5py.File("single_chunk_deflate_mask0.h5", "w") as f:
    f.create_dataset(
        "y", data=PAYLOAD, chunks=(64,),
        compression="gzip", compression_opts=4,
    )

# Sanity check: libhdf5 reads both back as 0..64.
for fn, ds in [("single_chunk_deflate_mask1.h5", "x"),
               ("single_chunk_deflate_mask0.h5", "y")]:
    with h5py.File(fn) as f:
        assert (f[ds][:] == PAYLOAD).all(), fn
print("generated single_chunk_deflate_mask{0,1}.h5")
