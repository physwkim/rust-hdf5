#!/bin/sh
# Rebuild the checked-in SOHM fixtures (sohm_list.h5, sohm_btree.h5,
# sohm_paged.h5, sohm_named_attr.h5, sohm_nested.h5, sohm_group.h5).
#
# Needs the pinned libhdf5's h5cc; override with H5CC=/path/to/h5cc.
# The generated files are byte-stable for a given libhdf5 version, so they are
# checked in and the test suite does not run this script.
set -eu

H5CC=${H5CC:-/home/stevek/micromamba/envs/tomo/bin/h5cc}
# The pinned h5cc hardcodes the conda toolchain's compiler name, which is not
# installed; point it at the system compiler unless the caller says otherwise.
HDF5_CC=${HDF5_CC:-cc}
HDF5_CLINKER=${HDF5_CLINKER:-cc}
export HDF5_CC HDF5_CLINKER
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Compile inside the temp dir: h5cc leaves its intermediate object file in
# the working directory.
cd "$tmp"
"$H5CC" -O2 -o "$tmp/gen_sohm" "$here/gen_sohm.c"
"$tmp/gen_sohm" "$here"
