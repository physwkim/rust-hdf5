/* Generator for `revised_refs.h5`, the fixture behind tests/revised_references.rs.
 *
 * The 1.12 reference kinds (H5R_OBJECT2, H5R_DATASET_REGION2, H5R_ATTR, all
 * stored as H5T_STD_REF) have no h5py binding — h5py 3.x raises "Unknown
 * reference type" even when *reading* them — so this fixture is written by
 * libhdf5 itself:
 *
 *     cc -o gen_revised_refs gen_revised_refs.c -lhdf5
 *     ./gen_revised_refs revised_refs.h5
 *
 * The file keeps H5F_LIBVER_EARLIEST as its low bound, which pins the
 * dataspace selections inside the references to their version-1 wire format;
 * pass "latest" as a second argument for the V112 low bound that produces the
 * version-2 point and version-3 regular-hyperslab selections whose bytes the
 * unit tests in src/format/reference.rs carry.
 */
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHK(x)                                                                                     \
    do {                                                                                           \
        if ((x) < 0) {                                                                             \
            fprintf(stderr, "failed at line %d\n", __LINE__);                                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "revised_refs.h5";
    int latest = argc > 2 && strcmp(argv[2], "latest") == 0;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHK(H5Pset_libver_bounds(fapl, latest ? H5F_LIBVER_V112 : H5F_LIBVER_EARLIEST,
                             H5F_LIBVER_LATEST));
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHK(f);

    /* The object every reference in the file points at. */
    hsize_t dims[2] = {4, 6};
    hid_t sp = H5Screate_simple(2, dims, NULL);
    int data[24];
    for (int i = 0; i < 24; i++)
        data[i] = i;
    hid_t ds = H5Dcreate2(f, "matrix", H5T_STD_I32LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));

    hsize_t adim = 3;
    hid_t asp = H5Screate_simple(1, &adim, NULL);
    int av[3] = {7, 8, 9};
    hid_t at = H5Acreate2(ds, "note", H5T_STD_I32LE, asp, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Awrite(at, H5T_NATIVE_INT, av));
    CHK(H5Aclose(at));
    CHK(H5Sclose(asp));

    hid_t g = H5Gcreate2(f, "grp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t two = 2;
    hid_t rsp = H5Screate_simple(1, &two, NULL);

    /* H5R_OBJECT2: a dataset and a group. */
    H5R_ref_t objrefs[2];
    CHK(H5Rcreate_object(f, "matrix", H5P_DEFAULT, &objrefs[0]));
    CHK(H5Rcreate_object(f, "grp", H5P_DEFAULT, &objrefs[1]));
    hid_t rds = H5Dcreate2(f, "objrefs", H5T_STD_REF, rsp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(rds, H5T_STD_REF, H5S_ALL, H5S_ALL, H5P_DEFAULT, objrefs));
    CHK(H5Dclose(rds));

    /* H5R_DATASET_REGION2: one hyperslab, one point list. */
    hsize_t start[2] = {1, 2}, count[2] = {2, 3};
    hid_t rsel = H5Scopy(sp);
    CHK(H5Sselect_hyperslab(rsel, H5S_SELECT_SET, start, NULL, count, NULL));
    hsize_t pts[4] = {0, 1, 3, 5};
    hid_t psel = H5Scopy(sp);
    CHK(H5Sselect_elements(psel, H5S_SELECT_SET, 2, pts));

    H5R_ref_t regrefs[2];
    CHK(H5Rcreate_region(f, "matrix", rsel, H5P_DEFAULT, &regrefs[0]));
    CHK(H5Rcreate_region(f, "matrix", psel, H5P_DEFAULT, &regrefs[1]));
    hid_t gds = H5Dcreate2(f, "regrefs", H5T_STD_REF, rsp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(gds, H5T_STD_REF, H5S_ALL, H5S_ALL, H5P_DEFAULT, regrefs));
    CHK(H5Dclose(gds));

    /* H5R_ATTR: the attribute written above. */
    H5R_ref_t attrref;
    CHK(H5Rcreate_attr(f, "matrix", "note", H5P_DEFAULT, &attrref));
    hsize_t one = 1;
    hid_t osp = H5Screate_simple(1, &one, NULL);
    hid_t ads = H5Dcreate2(f, "attrrefs", H5T_STD_REF, osp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(ads, H5T_STD_REF, H5S_ALL, H5S_ALL, H5P_DEFAULT, &attrref));
    CHK(H5Dclose(ads));

    for (int i = 0; i < 2; i++) {
        H5Rdestroy(&objrefs[i]);
        H5Rdestroy(&regrefs[i]);
    }
    H5Rdestroy(&attrref);
    CHK(H5Sclose(osp));
    CHK(H5Sclose(rsp));
    CHK(H5Sclose(rsel));
    CHK(H5Sclose(psel));
    CHK(H5Gclose(g));
    CHK(H5Dclose(ds));
    CHK(H5Sclose(sp));
    CHK(H5Pclose(fapl));
    CHK(H5Fclose(f));
    return 0;
}
