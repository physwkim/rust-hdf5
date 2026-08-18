/* Generate a file whose ROOT GROUP object header spills into a continuation
 * chunk ("OCHK"), so a corrupt chunk is reached on the open path itself.
 *
 * Build and run with the pinned libhdf5:
 *
 *     tests/fixtures/gen_ochk.sh
 *
 * The attributes are few enough to stay in compact storage (the dense
 * threshold is 8) but large enough that they do not fit chunk 0, which is what
 * makes libhdf5 allocate the continuation chunk.
 */

#include <hdf5.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if ((expr) < 0) {                                                                          \
            fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #expr);                      \
            return -1;                                                                             \
        }                                                                                          \
    } while (0)

static int
write_file(const char *path)
{
    hid_t   fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t   file, space, dset, attr, atype, ascalar;
    hsize_t dims[1] = {8};
    int     data[8];
    char    text[256];
    char    name[32];
    int     i;

    CHECK(fapl);
    /* Version-2 object headers, which are the ones that carry a checksum.
     * The v1.8 bounds stop short of the version-4 data layout message, so the
     * file's only unusual feature is the continuation chunk. */
    CHECK(H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_V18));

    file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK(file);
    CHECK(H5Pclose(fapl));

    space = H5Screate_simple(1, dims, NULL);
    CHECK(space);
    dset = H5Dcreate2(file, "data", H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(dset);
    for (i = 0; i < 8; i++)
        data[i] = i;
    CHECK(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    CHECK(H5Dclose(dset));
    CHECK(H5Sclose(space));

    ascalar = H5Screate(H5S_SCALAR);
    CHECK(ascalar);
    atype = H5Tcopy(H5T_C_S1);
    CHECK(atype);
    CHECK(H5Tset_size(atype, sizeof(text)));

    memset(text, 'x', sizeof(text));
    text[sizeof(text) - 1] = '\0';
    for (i = 0; i < 6; i++) {
        snprintf(name, sizeof(name), "note%d", i);
        text[0] = (char)('0' + i);
        attr = H5Acreate2(file, name, atype, ascalar, H5P_DEFAULT, H5P_DEFAULT);
        CHECK(attr);
        CHECK(H5Awrite(attr, atype, text));
        CHECK(H5Aclose(attr));
    }

    CHECK(H5Tclose(atype));
    CHECK(H5Sclose(ascalar));
    CHECK(H5Fclose(file));
    return 0;
}

int
main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : ".";
    char        path[512];

    snprintf(path, sizeof(path), "%s/ochk_root.h5", dir);
    if (write_file(path) < 0)
        return 1;
    printf("wrote %s\n", path);
    return 0;
}
