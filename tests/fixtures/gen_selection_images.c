/* Generator for the H5Sencode2 selection-image fixtures behind
 * src/format/selection.rs's `selection_matches_libhdf5_image` tests.
 *
 * H5Sencode2's output is a fixed preamble (type/version/sizeof_size/
 * extent_size) followed by the serialized extent, followed immediately by
 * `H5S_SELECT_SERIALIZE` output — byte-identical to what a VDS mapping or
 * an old-style region reference embeds. Capturing via H5Sencode2 needs no
 * dataset or global-heap plumbing, so it is the cheaper of the two capture
 * routes the task offered.
 *
 *     cc -o gen_selection_images gen_selection_images.c -lhdf5
 *     ./gen_selection_images
 *
 * Keeps H5F_LIBVER_EARLIEST as the file's low bound (same rationale as
 * gen_revised_refs.c on the wp-dtype branch): that pins every selection
 * below to its version-1 wire form, which is what this module's `encode()`
 * targets — the version H5S__point_get_version_enc_size /
 * H5S__hyper_get_version_enc_size both choose whenever the low bound is at
 * or below H5F_LIBVER_V110 and no count/bound exceeds 2^32.
 *
 * Each case is written as its own raw `.bin` file (just the H5Sencode2
 * blob, envelope included) under this directory; the Rust test strips the
 * envelope itself so the fixture stays a faithful, unmodified capture.
 */
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>

#define CHK(x)                                                                                             \
    do {                                                                                                   \
        if ((x) < 0) {                                                                                     \
            fprintf(stderr, "failed at line %d\n", __LINE__);                                              \
            exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

static void
dump(hid_t sp, const char *path)
{
    size_t nalloc = 0;
    CHK(H5Sencode2(sp, NULL, &nalloc, H5P_DEFAULT));
    unsigned char *buf = malloc(nalloc);
    if (!buf) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    CHK(H5Sencode2(sp, buf, &nalloc, H5P_DEFAULT));

    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        exit(1);
    }
    if (fwrite(buf, 1, nalloc, f) != nalloc) {
        fprintf(stderr, "short write to %s\n", path);
        exit(1);
    }
    fclose(f);
    free(buf);
    printf("%s: %zu bytes\n", path, nalloc);
}

int
main(void)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHK(H5Pset_libver_bounds(fapl, H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST));
    /* H5Sencode2 needs no open file, but H5CX_get_libver_bounds (used by
     * H5S__point_get_version_enc_size / H5S__hyper_get_version_enc_size)
     * reads whatever bounds the last opened file pushed onto the API
     * context, so open one to make the EARLIEST bound active. */
    hid_t file = H5Fcreate("gen_selection_images.tmp.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHK(file);

    hsize_t dims1[1] = {20};
    hid_t sp;

    /* H5S_SEL_ALL: freshly created simple dataspace selects everything. */
    sp = H5Screate_simple(1, dims1, NULL);
    dump(sp, "all_v1.bin");
    CHK(H5Sclose(sp));

    /* H5S_SEL_NONE. */
    sp = H5Screate_simple(1, dims1, NULL);
    CHK(H5Sselect_none(sp));
    dump(sp, "none_v1.bin");
    CHK(H5Sclose(sp));

    /* One hyperslab block: elements [4, 11]. */
    sp = H5Screate_simple(1, dims1, NULL);
    hsize_t start1[1] = {4}, count1[1] = {8};
    CHK(H5Sselect_hyperslab(sp, H5S_SELECT_SET, start1, NULL, count1, NULL));
    dump(sp, "hyperslab_single_block_v1.bin");
    CHK(H5Sclose(sp));

    /* A REGULAR hyperslab pattern (start=0, stride=5, count=3, block=2):
     * still written as a version-1 block list under EARLIEST, exercising
     * this module's Regular -> block-list encode path against a selection
     * that was NOT built one block at a time. */
    sp = H5Screate_simple(1, dims1, NULL);
    hsize_t rstart[1] = {0}, rcount[1] = {3}, rstride[1] = {5}, rblock[1] = {2};
    CHK(H5Sselect_hyperslab(sp, H5S_SELECT_SET, rstart, rstride, rcount, rblock));
    dump(sp, "hyperslab_regular_3blocks_v1.bin");
    CHK(H5Sclose(sp));

    /* A 2-D regular hyperslab: start=(0,0) stride=(4,4) count=(2,2)
     * block=(2,2) -- exercises row-major (last-dim-fastest) block order. */
    hsize_t dims2[2] = {8, 8};
    sp = H5Screate_simple(2, dims2, NULL);
    hsize_t start2[2] = {0, 0}, count2[2] = {2, 2}, stride2[2] = {4, 4}, block2[2] = {2, 2};
    CHK(H5Sselect_hyperslab(sp, H5S_SELECT_SET, start2, stride2, count2, block2));
    dump(sp, "hyperslab_2d_regular_v1.bin");
    CHK(H5Sclose(sp));

    /* A point selection: 4 elements, rank 1. */
    sp = H5Screate_simple(1, dims1, NULL);
    hsize_t pts[4] = {1, 3, 7, 15};
    CHK(H5Sselect_elements(sp, H5S_SELECT_SET, 4, pts));
    dump(sp, "points4_v1.bin");
    CHK(H5Sclose(sp));

    CHK(H5Fclose(file));
    CHK(H5Pclose(fapl));
    remove("gen_selection_images.tmp.h5");
    return 0;
}
