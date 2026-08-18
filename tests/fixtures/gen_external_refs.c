/* Generator for `ext_refs.h5` and `ext_ref_target.h5`, the fixture pair behind
 * the external-reference cases in tests/revised_references.rs.
 *
 * A 1.12 reference carries the H5R_IS_EXTERNAL flag and the target file's name
 * when the file it is written into is not the file its target lives in
 * (H5T__ref_mem_getsize, H5Tref.c:436), and the name recorded is the one that
 * file was opened under — H5F_get_name, not a canonical path. Dereferencing
 * uses it verbatim, against the process working directory: H5R__reopen_file
 * hands it straight to H5VL_file_open with no prefix search of any kind
 * (H5Rint.c:466 "TODO add search path", :487). So the target is created under
 * the very path the test reads it back by, and this generator must run from
 * the crate root:
 *
 *     cc -o gen_external_refs tests/fixtures/gen_external_refs.c \
 *         -I$PREFIX/include -L$PREFIX/lib -lhdf5
 *     ./gen_external_refs tests/fixtures/ext_refs.h5 tests/fixtures/ext_ref_target.h5
 *
 * h5py has no H5T_STD_REF binding at all, so libhdf5 writes this one itself,
 * as it does for `revised_refs.h5`.
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

int main(int argc, char **argv)
{
    const char *holder_path = argc > 1 ? argv[1] : "ext_refs.h5";
    const char *target_path = argc > 2 ? argv[2] : "ext_ref_target.h5";

    /* --- the file every reference in the holder points into ------------- */
    hid_t target = H5Fcreate(target_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHK(target);

    hsize_t dims[2] = {4, 6};
    hid_t sp = H5Screate_simple(2, dims, NULL);
    int data[24];
    for (int i = 0; i < 24; i++)
        data[i] = i;
    hid_t ds = H5Dcreate2(target, "matrix", H5T_STD_I32LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));

    hsize_t adim = 3;
    hid_t asp = H5Screate_simple(1, &adim, NULL);
    int av[3] = {7, 8, 9};
    hid_t at = H5Acreate2(ds, "note", H5T_STD_I32LE, asp, H5P_DEFAULT, H5P_DEFAULT);
    CHK(H5Awrite(at, H5T_NATIVE_INT, av));
    CHK(H5Aclose(at));
    CHK(H5Sclose(asp));

    hid_t g = H5Gcreate2(target, "grp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHK(g);

    /* --- the file holding the references -------------------------------- */
    hid_t holder = H5Fcreate(holder_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHK(holder);

    hsize_t two = 2;
    hid_t rsp = H5Screate_simple(1, &two, NULL);
    hsize_t one = 1;
    hid_t osp = H5Screate_simple(1, &one, NULL);

    /* H5R_OBJECT2 across files: a dataset and a group. */
    H5R_ref_t objrefs[2];
    CHK(H5Rcreate_object(target, "matrix", H5P_DEFAULT, &objrefs[0]));
    CHK(H5Rcreate_object(target, "grp", H5P_DEFAULT, &objrefs[1]));
    hid_t ods = H5Dcreate2(holder, "extobjrefs", H5T_STD_REF, rsp, H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT);
    CHK(H5Dwrite(ods, H5T_STD_REF, H5S_ALL, H5S_ALL, H5P_DEFAULT, objrefs));
    CHK(H5Dclose(ods));

    /* H5R_DATASET_REGION2 across files: the hyperslab (1,2)-(2,4). */
    hid_t rsel = H5Screate_simple(2, dims, NULL);
    hsize_t start[2] = {1, 2}, count[2] = {2, 3};
    CHK(H5Sselect_hyperslab(rsel, H5S_SELECT_SET, start, NULL, count, NULL));
    H5R_ref_t regref;
    CHK(H5Rcreate_region(target, "matrix", rsel, H5P_DEFAULT, &regref));
    hid_t gds = H5Dcreate2(holder, "extregrefs", H5T_STD_REF, osp, H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT);
    CHK(H5Dwrite(gds, H5T_STD_REF, H5S_ALL, H5S_ALL, H5P_DEFAULT, &regref));
    CHK(H5Dclose(gds));

    /* H5R_ATTR across files: `note` on the target's `matrix`. */
    H5R_ref_t attrref;
    CHK(H5Rcreate_attr(target, "matrix", "note", H5P_DEFAULT, &attrref));
    hid_t ads = H5Dcreate2(holder, "extattrrefs", H5T_STD_REF, osp, H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT);
    CHK(H5Dwrite(ads, H5T_STD_REF, H5S_ALL, H5S_ALL, H5P_DEFAULT, &attrref));
    CHK(H5Dclose(ads));

    H5Rdestroy(&objrefs[0]);
    H5Rdestroy(&objrefs[1]);
    H5Rdestroy(&regref);
    H5Rdestroy(&attrref);
    CHK(H5Sclose(rsel));
    CHK(H5Sclose(osp));
    CHK(H5Sclose(rsp));
    CHK(H5Fclose(holder));
    CHK(H5Gclose(g));
    CHK(H5Dclose(ds));
    CHK(H5Sclose(sp));
    CHK(H5Fclose(target));
    return 0;
}
