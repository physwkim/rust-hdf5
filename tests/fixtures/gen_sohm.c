/* Generate the shared-object-header-message (SOHM) fixtures.
 *
 * h5py exposes no binding for H5Pset_shared_mesg_nlevels /
 * H5Pset_shared_mesg_index, so these files cannot be produced from Python.
 * Build and run with the pinned libhdf5:
 *
 *     tests/fixtures/gen_sohm.sh
 *
 * Two files are written next to this source:
 *
 *   sohm_list.h5   one SOHM index holding datatype + dataspace + attribute
 *                  messages, left in its initial "list" form.
 *   sohm_btree.h5  the same content with the list->B-tree phase change forced
 *                  to zero, so the index is a v2 B-tree from the first insert.
 *
 * Both files put the shared-message table in the superblock extension, which
 * is what makes them exercise the extension walk as well as SOHM itself.
 */

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if ((expr) < 0) {                                                                          \
            fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #expr);                      \
            return -1;                                                                             \
        }                                                                                          \
    } while (0)

/* Write one file. `max_list`/`min_btree` drive the index form: the default
 * (50, 40) keeps a small index as a list, (0, 0) forces a B-tree. */
static int
write_file(const char *path, unsigned max_list, unsigned min_btree)
{
    hid_t   fcpl = H5Pcreate(H5P_FILE_CREATE);
    hid_t   file, space, dset, attr, aspace, atype;
    hsize_t dims[1]  = {8};
    hsize_t adims[1] = {3};
    int     data[8];
    double  adata[3] = {0.5, 1.5, 2.5};
    char    name[32];
    int     i, j;

    CHECK(fcpl);
    /* One index, covering the three message types libhdf5 shares most often.
     * min_mesg_size = 0 shares even the smallest messages, so the fixture
     * does not depend on the exact encoded size of a datatype message. */
    CHECK(H5Pset_shared_mesg_nindexes(fcpl, 1));
    CHECK(H5Pset_shared_mesg_index(
        fcpl, 0, H5O_SHMESG_DTYPE_FLAG | H5O_SHMESG_SDSPACE_FLAG | H5O_SHMESG_ATTR_FLAG, 0));
    CHECK(H5Pset_shared_mesg_phase_change(fcpl, max_list, min_btree));

    file = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
    CHECK(file);
    CHECK(H5Pclose(fcpl));

    space = H5Screate_simple(1, dims, NULL);
    CHECK(space);
    aspace = H5Screate_simple(1, adims, NULL);
    CHECK(aspace);
    atype = H5Tcopy(H5T_IEEE_F64LE);
    CHECK(atype);

    /* Four datasets with the identical datatype/dataspace: after the first,
     * every later header stores a shared-message pointer instead of the
     * literal message. */
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 8; j++)
            data[j] = i * 10 + j;
        snprintf(name, sizeof(name), "shared%d", i);
        dset = H5Dcreate2(file, name, H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(dset);
        CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));

        /* The same attribute on every dataset — shared through the same
         * index once the message type flag covers attributes. */
        attr = H5Acreate2(dset, "cal", atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(attr);
        CHECK(H5Awrite(attr, H5T_NATIVE_DOUBLE, adata));
        CHECK(H5Aclose(attr));
        CHECK(H5Dclose(dset));
    }

    /* A committed (named) datatype plus a dataset that uses it: that dataset's
     * datatype message is shared by object-header address, the other half of
     * the shared-message decode path. */
    {
        hid_t named = H5Tcopy(H5T_STD_I32LE);
        CHECK(named);
        CHECK(H5Tcommit2(file, "named_i32", named, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        dset = H5Dcreate2(file, "uses_named", named, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(dset);
        for (j = 0; j < 8; j++)
            data[j] = 100 + j;
        CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
        CHECK(H5Dclose(dset));
        CHECK(H5Tclose(named));
    }

    CHECK(H5Tclose(atype));
    CHECK(H5Sclose(aspace));
    CHECK(H5Sclose(space));
    CHECK(H5Fclose(file));
    return 0;
}

int
main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : ".";
    char        path[512];

    snprintf(path, sizeof(path), "%s/sohm_list.h5", dir);
    if (write_file(path, 50, 40) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sohm_btree.h5", dir);
    if (write_file(path, 0, 0) < 0)
        return 1;
    printf("wrote %s\n", path);

    return 0;
}
