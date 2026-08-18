/* Generator for `vds_late_layout.h5`, the fixture behind tests/vds_late_layout.rs.
 *
 * `H5Pset_virtual` installs the virtual layout with `H5P_poke` on
 * `H5D_CRT_LAYOUT_NAME` (1.14.6 H5Pdcpl.c:2146), bypassing `H5P__set_layout`,
 * which is the only place the default allocation time is re-derived from the
 * storage class (1.14.6 H5Pdcpl.c:1758-1782: virtual and chunked get
 * `H5D_ALLOC_TIME_INCR`).  So a dcpl that never sees `H5Pset_layout` keeps the
 * contiguous default `H5D_ALLOC_TIME_LATE` while carrying a virtual layout,
 * and the dataset it creates stores that byte.  That is the whole on-disk
 * difference: writing the two datasets below into separate files gives files
 * that differ in exactly one byte, the fill value message's space allocation
 * time (0x03 -> 0x02).
 *
 * h5py's `VirtualLayout` calls `dcpl.set_layout(h5d.VIRTUAL)` before any
 * mapping (h5py 3.15.1 _hl/vds.py:174, :216), so the variant is only
 * reachable from a hand-built dcpl; this generator is that dcpl.
 *
 *     cc -o gen_vds_late_layout tests/fixtures/gen_vds_late_layout.c \
 *         -I$PREFIX/include -L$PREFIX/lib -lhdf5
 *     ./gen_vds_late_layout tests/fixtures/vds_late_layout.h5
 */
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>

#define CHK(x)                                                                                     \
    do {                                                                                           \
        if ((x) < 0) {                                                                             \
            fprintf(stderr, "failed at line %d\n", __LINE__);                                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* One 4x4 virtual dataset whose top two rows map to the whole of `src`; the
 * bottom two rows have no mapping and read as the fill value. */
static void make_vds(hid_t file, const char *name, int set_layout_first)
{
    hid_t   dcpl, vspace, sspace, dset;
    hsize_t vdims[2] = {4, 4}, sdims[2] = {2, 4};
    hsize_t start[2] = {0, 0}, count[2] = {2, 4};
    int     fill = -9;

    CHK(dcpl = H5Pcreate(H5P_DATASET_CREATE));
    /* The conventional order; omitting this is the "late" variant. */
    if (set_layout_first)
        CHK(H5Pset_layout(dcpl, H5D_VIRTUAL));
    CHK(H5Pset_fill_value(dcpl, H5T_NATIVE_INT, &fill));
    CHK(vspace = H5Screate_simple(2, vdims, NULL));
    CHK(H5Sselect_hyperslab(vspace, H5S_SELECT_SET, start, NULL, count, NULL));
    CHK(sspace = H5Screate_simple(2, sdims, NULL));
    CHK(H5Sselect_all(sspace));
    CHK(H5Pset_virtual(dcpl, vspace, ".", "/src", sspace));
    CHK(dset = H5Dcreate2(file, name, H5T_STD_I32LE, vspace, H5P_DEFAULT, dcpl, H5P_DEFAULT));
    CHK(H5Dclose(dset));
    CHK(H5Sclose(sspace));
    CHK(H5Sclose(vspace));
    CHK(H5Pclose(dcpl));
}

/* An unlimited-on-both-sides mapping, so the extent is not the stored one
 * but whatever `H5D__virtual_set_extent_unlim` clips it to from the source's
 * current 3 rows. */
static void make_unlim_vds(hid_t file, const char *name, int set_layout_first)
{
    hid_t   dcpl, vspace, sspace, dset;
    hsize_t vdims[2] = {1, 4}, vmax[2] = {H5S_UNLIMITED, 4};
    hsize_t sdims[2] = {1, 4}, smax[2] = {H5S_UNLIMITED, 4};
    hsize_t start[2] = {0, 0}, count[2] = {H5S_UNLIMITED, 1}, block[2] = {1, 4};
    int     fill = -8;

    CHK(dcpl = H5Pcreate(H5P_DATASET_CREATE));
    if (set_layout_first)
        CHK(H5Pset_layout(dcpl, H5D_VIRTUAL));
    CHK(H5Pset_fill_value(dcpl, H5T_NATIVE_INT, &fill));
    CHK(vspace = H5Screate_simple(2, vdims, vmax));
    CHK(H5Sselect_hyperslab(vspace, H5S_SELECT_SET, start, NULL, count, block));
    CHK(sspace = H5Screate_simple(2, sdims, smax));
    CHK(H5Sselect_hyperslab(sspace, H5S_SELECT_SET, start, NULL, count, block));
    CHK(H5Pset_virtual(dcpl, vspace, ".", "/src_unlim", sspace));
    CHK(dset = H5Dcreate2(file, name, H5T_STD_I32LE, vspace, H5P_DEFAULT, dcpl, H5P_DEFAULT));
    CHK(H5Dclose(dset));
    CHK(H5Sclose(sspace));
    CHK(H5Sclose(vspace));
    CHK(H5Pclose(dcpl));
}

int main(int argc, char **argv)
{
    hid_t   file, space, dset, chunked;
    hsize_t sdims[2] = {2, 4};
    hsize_t udims[2] = {3, 4}, umax[2] = {H5S_UNLIMITED, 4}, uchunk[2] = {1, 4};
    int     data[8], udata[12], i;

    if (argc != 2) {
        fprintf(stderr, "usage: %s OUT.h5\n", argv[0]);
        return 1;
    }
    for (i = 0; i < 8; i++)
        data[i] = i;

    CHK(file = H5Fcreate(argv[1], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    CHK(space = H5Screate_simple(2, sdims, NULL));
    CHK(dset = H5Dcreate2(file, "src", H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    CHK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    CHK(H5Dclose(dset));
    CHK(H5Sclose(space));

    for (i = 0; i < 12; i++)
        udata[i] = 100 + i;
    CHK(space = H5Screate_simple(2, udims, umax));
    CHK(chunked = H5Pcreate(H5P_DATASET_CREATE));
    CHK(H5Pset_chunk(chunked, 2, uchunk));
    CHK(dset = H5Dcreate2(file, "src_unlim", H5T_STD_I32LE, space, H5P_DEFAULT, chunked, H5P_DEFAULT));
    CHK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, udata));
    CHK(H5Dclose(dset));
    CHK(H5Pclose(chunked));
    CHK(H5Sclose(space));

    make_vds(file, "vds", 1);
    make_vds(file, "vds_late", 0);
    make_unlim_vds(file, "vds_unlim", 1);
    make_unlim_vds(file, "vds_late_unlim", 0);

    CHK(H5Fclose(file));
    return 0;
}
