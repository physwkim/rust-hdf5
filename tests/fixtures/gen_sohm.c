/* Generate the shared-object-header-message (SOHM) fixtures.
 *
 * h5py exposes no binding for H5Pset_shared_mesg_nlevels /
 * H5Pset_shared_mesg_index, so these files cannot be produced from Python.
 * Build and run with the pinned libhdf5:
 *
 *     tests/fixtures/gen_sohm.sh
 *
 * Five files are written next to this source:
 *
 *   sohm_list.h5   one SOHM index holding datatype + dataspace + attribute
 *                  messages, left in its initial "list" form.
 *   sohm_btree.h5  the same content with the list->B-tree phase change forced
 *                  to zero, so the index is a v2 B-tree from the first insert.
 *   sohm_paged.h5  the list form again, over paged file space, so the
 *                  superblock extension carries a file space info message
 *                  beside the shared-message table.
 *   sohm_named_attr.h5
 *                  the list form again, with the shared `cal' attribute also
 *                  on the committed datatype. A committed datatype is kept by
 *                  its bytes across a reopen, so this is a file where an
 *                  object that keeps its bytes holds a heap pointer.
 *   sohm_nested.h5 the list form again, with the datasets' datatype a copy of
 *                  H5T_STD_I32LE rather than the predefined type itself. A
 *                  predefined type is immutable, and H5O__dtype_can_share
 *                  refuses to share an immutable or committed type, so the
 *                  other files' datasets keep their datatype literal. The
 *                  copy is shareable, which makes this the file whose record
 *                  census a crate-written twin can be compared against
 *                  one-for-one (tests/sohm_write.rs).
 *   sohm_group.h5  the list form again, plus a subgroup ("g") whose only
 *                  content is one hard link (named GROUP_LINK_NAME below).
 *                  Neither H5Pset_shared_mesg_* nor H5Pset_libver_bounds is
 *                  called on this file, so "g" gets libhdf5's EARLIEST-format
 *                  default group storage: a classic Symbol Table message
 *                  (B-tree + local heap + one Symbol Table Node holding the
 *                  one link), not the newer Link-message form. Well-formed as
 *                  written; tests/sohm_append.rs corrupts a copy of it at
 *                  test time (flips a byte of that SNOD's 4-byte "SNOD"
 *                  signature) to reach `blocks_shared_message_rebuild`'s
 *                  "names objects of its own" clause: a *group* this writer
 *                  cannot classify on reopen, kept by its bytes, whose
 *                  unwalked subtree could hide anything.
 *
 * All five put the shared-message table in the superblock extension, which is
 * what makes them exercise the extension walk as well as SOHM itself.
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
 * (50, 40) keeps a small index as a list, (0, 0) forces a B-tree. `paged`
 * adds a file space info message to the superblock extension, which is the
 * only way a file gets an extension message the shared-message table does not
 * account for. `named_attr` puts the shared attribute on the committed
 * datatype as well, which is the only way an object no writer can re-encode
 * ends up holding a shared-message pointer. `copy_dtype` gives the datasets a
 * copy of H5T_STD_I32LE instead of the predefined type, which is what makes
 * their datatype shareable at all. */
static int
write_file(const char *path, unsigned max_list, unsigned min_btree, int paged, int named_attr,
           int copy_dtype)
{
    hid_t   fcpl = H5Pcreate(H5P_FILE_CREATE);
    hid_t   file, space, dset, attr, aspace, atype, dtype;
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
    if (paged)
        CHECK(H5Pset_file_space_strategy(fcpl, H5F_FSPACE_STRATEGY_PAGE, 0, (hsize_t)1));

    file = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
    CHECK(file);
    CHECK(H5Pclose(fcpl));

    space = H5Screate_simple(1, dims, NULL);
    CHECK(space);
    aspace = H5Screate_simple(1, adims, NULL);
    CHECK(aspace);
    atype = H5Tcopy(H5T_IEEE_F64LE);
    CHECK(atype);
    dtype = copy_dtype ? H5Tcopy(H5T_STD_I32LE) : H5T_STD_I32LE;
    CHECK(dtype);

    /* Four datasets with the identical datatype/dataspace: after the first,
     * every later header stores a shared-message pointer instead of the
     * literal message. */
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 8; j++)
            data[j] = i * 10 + j;
        snprintf(name, sizeof(name), "shared%d", i);
        dset = H5Dcreate2(file, name, dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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
        if (named_attr) {
            attr = H5Acreate2(named, "cal", atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
            CHECK(attr);
            CHECK(H5Awrite(attr, H5T_NATIVE_DOUBLE, adata));
            CHECK(H5Aclose(attr));
        }
        dset = H5Dcreate2(file, "uses_named", named, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(dset);
        for (j = 0; j < 8; j++)
            data[j] = 100 + j;
        CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
        CHECK(H5Dclose(dset));
        CHECK(H5Tclose(named));
    }

    if (copy_dtype)
        CHECK(H5Tclose(dtype));
    CHECK(H5Tclose(atype));
    CHECK(H5Sclose(aspace));
    CHECK(H5Sclose(space));
    CHECK(H5Fclose(file));
    return 0;
}

/* The link this file's one subgroup holds. Named long and distinctive so a
 * raw byte search for it in tests/sohm_append.rs cannot collide with
 * anything else the file happens to contain (used only to confirm which
 * local-heap string belongs to "g" while diagnosing the fixture's layout;
 * the corruption itself targets the Symbol Table Node's signature, not this
 * name). */
#define GROUP_LINK_NAME "corrupt_target_link"

/* One SOHM index (same shape as `write_file`'s list form) plus a subgroup
 * whose only content is one hard link. This file pins no libver bound, so
 * that subgroup gets libhdf5's default EARLIEST-format storage: a classic
 * Symbol Table (B-tree + local heap + one Symbol Table Node), never the
 * newer Link-message form regardless of how few links it holds. */
static int
write_group_file(const char *path)
{
    hid_t   fcpl = H5Pcreate(H5P_FILE_CREATE);
    hid_t   file, space, dset, attr, aspace, atype, group;
    hsize_t dims[1]  = {4};
    hsize_t adims[1] = {2};
    int     data[4]  = {1, 2, 3, 4};
    double  adata[2] = {0.5, 1.5};

    CHECK(fcpl);
    CHECK(H5Pset_shared_mesg_nindexes(fcpl, 1));
    CHECK(H5Pset_shared_mesg_index(
        fcpl, 0, H5O_SHMESG_DTYPE_FLAG | H5O_SHMESG_SDSPACE_FLAG | H5O_SHMESG_ATTR_FLAG, 0));
    CHECK(H5Pset_shared_mesg_phase_change(fcpl, 50, 40));

    file = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
    CHECK(file);
    CHECK(H5Pclose(fcpl));

    space = H5Screate_simple(1, dims, NULL);
    CHECK(space);
    aspace = H5Screate_simple(1, adims, NULL);
    CHECK(aspace);
    atype = H5Tcopy(H5T_IEEE_F64LE);
    CHECK(atype);

    dset = H5Dcreate2(file, "shared0", H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(dset);
    CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    attr = H5Acreate2(dset, "cal", atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(attr);
    CHECK(H5Awrite(attr, H5T_NATIVE_DOUBLE, adata));
    CHECK(H5Aclose(attr));
    CHECK(H5Dclose(dset));

    /* The subgroup this fixture exists for: one hard link, stored the
     * classic way (Symbol Table message + B-tree + local heap + SNOD) since
     * this file never raises the format's libver bound. */
    group = H5Gcreate2(file, "g", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(group);
    CHECK(H5Lcreate_hard(file, "/shared0", group, GROUP_LINK_NAME, H5P_DEFAULT, H5P_DEFAULT));
    CHECK(H5Gclose(group));

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
    if (write_file(path, 50, 40, 0, 0, 0) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sohm_btree.h5", dir);
    if (write_file(path, 0, 0, 0, 0, 0) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sohm_paged.h5", dir);
    if (write_file(path, 50, 40, 1, 0, 0) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sohm_named_attr.h5", dir);
    if (write_file(path, 50, 40, 0, 1, 0) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sohm_nested.h5", dir);
    if (write_file(path, 50, 40, 0, 0, 1) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sohm_group.h5", dir);
    if (write_group_file(path) < 0)
        return 1;
    printf("wrote %s\n", path);

    return 0;
}
