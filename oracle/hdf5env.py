"""Process-wide libhdf5 settings the oracle depends on. Import before h5py.

External raw-data files and virtual-dataset sources are stored in the HDF5
file as the name the writer passed, and libhdf5 resolves a relative name
against the *current working directory* — not against the file that names it.
The oracle generates files into a work directory and reads them back from the
repository root, so a relative name would never resolve. Storing an absolute
name instead would make the reference dump machine-specific.

``${ORIGIN}`` is libhdf5's own token for "the directory holding the HDF5 file
that names this target", so setting it as the prefix keeps the stored name
relative and the dump reproducible. These variables are read by libhdf5 when
it resolves a target, so setting them any time before the read is enough, but
importing this module first keeps the ordering obvious.

They cover raw data only: external *links* are resolved through
HDF5_EXT_PREFIX, which is left alone.
"""

import os

os.environ.setdefault("HDF5_EXTFILE_PREFIX", "${ORIGIN}")
os.environ.setdefault("HDF5_VDS_PREFIX", "${ORIGIN}")
