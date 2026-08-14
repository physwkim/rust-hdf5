/* Generate the superblock-extension fixtures.
 *
 * h5py exposes neither H5Pset_sym_k/H5Pset_istore_k nor
 * H5Pset_file_space_strategy, so these files cannot be produced from Python.
 * Build and run with the pinned libhdf5:
 *
 *     tests/fixtures/gen_sbext.sh
 *
 * Three files are written next to this source:
 *
 *   btreek_legacy.h5 non-default v1 B-tree "K" values only. All three fit in
 *                    the superblock itself, which H5F__super_init keeps at
 *                    version 1 (the chunk rank has no version-0 field) with no
 *                    extension.
 *   sbext_btreek.h5  the same content plus persisted free-space managers,
 *                    which moves the superblock to version 2: the K values now
 *                    have nowhere to go but a B-tree-K message in the
 *                    superblock extension.
 *   sbext_paged.h5   paged aggregation with persisted free-space managers, so
 *                    the file-space info message carries strategy PAGE, a
 *                    non-default page size and the twelve manager addresses.
 *
 * Both K files hold enough links in the root group that its single symbol
 * table node is larger than 8 KiB, and a chunked dataset whose v1 B-tree node
 * only fits its entries at the non-default rank.
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

/* Symbol-table leaf rank: a SNOD holds up to 2 * 110 = 220 entries, and its
 * on-disk size is 8 + 220 * 40 = 8808 bytes — past the fixed 8 KiB window a
 * reader that assumes the default rank would use. */
#define SYM_LEAF_K 110
/* Symbol-table B-tree internal rank (default 16). */
#define SNODE_IK 100
/* Chunked-storage B-tree internal rank (default 32): the chunk tree below
 * holds 100 entries, which fit in one node only at this rank. */
#define ISTORE_K 64
/* Padding links: a symbol-table node needs more than 204 entries before it
 * passes 8 KiB (8 + n * 40 bytes), and must stay under 2 * SYM_LEAF_K so the
 * B-tree keeps a single leaf. Soft links are used because they cost one
 * symbol-table entry and a heap string, with no object header behind them. */
#define NLINKS 200
/* Real datasets alongside the padding, so the walk has something to find. */
#define NDSETS 8

/* `with_extension` persists free-space managers, which is what pushes the
 * superblock to version 2 and the K values out into the extension. */
static int
write_btreek(const char *path, int with_extension)
{
    hid_t   fcpl = H5Pcreate(H5P_FILE_CREATE);
    hid_t   dcpl, file, space, dset;
    hsize_t dims[1]       = {1000};
    hsize_t chunk_dims[1] = {10};
    hsize_t small_dims[1] = {4};
    char    name[32];
    int    *data;
    int     i;

    CHECK(fcpl);
    CHECK(H5Pset_sym_k(fcpl, SNODE_IK, SYM_LEAF_K));
    CHECK(H5Pset_istore_k(fcpl, ISTORE_K));
    if (with_extension)
        CHECK(H5Pset_file_space_strategy(fcpl, H5F_FSPACE_STRATEGY_FSM_AGGR, 1, (hsize_t)1));

    file = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
    CHECK(file);
    CHECK(H5Pclose(fcpl));

    space = H5Screate_simple(1, small_dims, NULL);
    CHECK(space);
    for (i = 0; i < NDSETS; i++) {
        int small[4];
        int j;
        for (j = 0; j < 4; j++)
            small[j] = i * 100 + j;
        snprintf(name, sizeof(name), "d%d", i);
        dset = H5Dcreate2(file, name, H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(dset);
        CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, small));
        CHECK(H5Dclose(dset));
    }
    CHECK(H5Sclose(space));

    for (i = 0; i < NLINKS; i++) {
        snprintf(name, sizeof(name), "s%03d", i);
        CHECK(H5Lcreate_soft("/d0", file, name, H5P_DEFAULT, H5P_DEFAULT));
    }

    /* A chunked dataset with 100 chunks: its v1 B-tree root is one node of
     * 2 * ISTORE_K capacity, which a reader sizing by the default rank would
     * read short. */
    space = H5Screate_simple(1, dims, NULL);
    CHECK(space);
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK(dcpl);
    CHECK(H5Pset_chunk(dcpl, 1, chunk_dims));
    dset = H5Dcreate2(file, "chunked", H5T_STD_I32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    CHECK(dset);
    data = (int *)malloc(sizeof(int) * 1000);
    if (!data)
        return -1;
    for (i = 0; i < 1000; i++)
        data[i] = i;
    CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    free(data);
    CHECK(H5Dclose(dset));
    CHECK(H5Pclose(dcpl));
    CHECK(H5Sclose(space));
    CHECK(H5Fclose(file));
    return 0;
}

static int
write_paged(const char *path)
{
    hid_t   fcpl = H5Pcreate(H5P_FILE_CREATE);
    hid_t   file, space, dset;
    hsize_t dims[1] = {16};
    int     data[16];
    int     i;

    CHECK(fcpl);
    /* Paged aggregation with a non-default page size, free-space managers
     * persisted: the file-space info message then carries strategy PAGE and
     * the twelve page-type manager addresses. */
    CHECK(H5Pset_file_space_page_size(fcpl, (hsize_t)8192));
    CHECK(H5Pset_file_space_strategy(fcpl, H5F_FSPACE_STRATEGY_PAGE, 1, (hsize_t)1));

    file = H5Fcreate(path, H5F_ACC_TRUNC, fcpl, H5P_DEFAULT);
    CHECK(file);
    CHECK(H5Pclose(fcpl));

    space = H5Screate_simple(1, dims, NULL);
    CHECK(space);
    for (i = 0; i < 16; i++)
        data[i] = i * 3;
    dset = H5Dcreate2(file, "paged", H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(dset);
    CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    CHECK(H5Dclose(dset));
    CHECK(H5Sclose(space));
    CHECK(H5Fclose(file));
    return 0;
}

int
main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : ".";
    char        path[512];

    snprintf(path, sizeof(path), "%s/btreek_legacy.h5", dir);
    if (write_btreek(path, 0) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sbext_btreek.h5", dir);
    if (write_btreek(path, 1) < 0)
        return 1;
    printf("wrote %s\n", path);

    snprintf(path, sizeof(path), "%s/sbext_paged.h5", dir);
    if (write_paged(path) < 0)
        return 1;
    printf("wrote %s\n", path);

    return 0;
}
