/* C mirror of src/bin/perf_probe.rs against libhdf5. Same shapes, dtypes,
 * chunk sizes, deflate level, and access patterns; only the library under
 * the workload differs. Built and run by perf/run.py.
 *
 * Usage: probe_c <workdir> <workload> <reps>
 * Output: one "BENCH <workload> rep <i> ns <elapsed>" line per rep. */

#include <fcntl.h>
#include <hdf5.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CONTIG_N (16 * 1024 * 1024)
#define CHUNK_ELEMS (256 * 1024)
#define DEFLATE_N (8 * 1024 * 1024)
#define DEFLATE_LEVEL 6
#define SLICE_READS 1000
#define SLICE_ELEMS 8192
/* deflate-slice: chunks small enough for the default 1 MiB chunk cache,
 * so its reuse across consecutive slices is part of the workload. */
#define DSLICE_CHUNK (32 * 1024)
#define SMALL_DSETS 2000
#define SMALL_N 128
#define ATTRS 1000
#define APPEND_COLS 4096
#define APPEND_ROWS 1024
#define APPEND_BATCH 16

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double *ramp(size_t n) {
    double *d = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) d[i] = (double)i;
    return d;
}

static double *compressible(size_t n) {
    double *d = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) d[i] = (double)(i & 0xFF);
    return d;
}

/* Same LCG as the Rust probe. */
static uint64_t lcg_state;
static uint64_t lcg_next(void) {
    lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
    return lcg_state;
}

#define CHECK(x)                                                          \
    do {                                                                  \
        if ((x) < 0) {                                                    \
            fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x);   \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static void write_contig(const char *path, const double *data, hsize_t n) {
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(f);
    hid_t sp = H5Screate_simple(1, &n, NULL);
    hid_t d = H5Dcreate2(f, "data", H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
    CHECK(d);
    CHECK(H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    H5Dclose(d);
    H5Sclose(sp);
    H5Fclose(f);
}

static void write_chunked(const char *path, const double *data, hsize_t n,
                          int deflate, hsize_t chunk) {
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK(f);
    hid_t sp = H5Screate_simple(1, &n, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK(H5Pset_chunk(dcpl, 1, &chunk));
    if (deflate) CHECK(H5Pset_deflate(dcpl, DEFLATE_LEVEL));
    hid_t d = H5Dcreate2(f, "data", H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, dcpl,
                         H5P_DEFAULT);
    CHECK(d);
    CHECK(H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data));
    H5Dclose(d);
    H5Pclose(dcpl);
    H5Sclose(sp);
    H5Fclose(f);
}

static double *read_full(const char *path, hsize_t expect) {
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(f);
    hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
    CHECK(d);
    hid_t sp = H5Dget_space(d);
    hsize_t n;
    H5Sget_simple_extent_dims(sp, &n, NULL);
    if (n != expect) exit(1);
    double *buf = malloc(n * sizeof(double));
    CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
    H5Sclose(sp);
    H5Dclose(d);
    H5Fclose(f);
    return buf;
}

/* Keeps a summed-over-every-element workload from being optimized into
 * nothing; the Rust probe's black_box does the same job. */
static volatile double sink;

/* The libhdf5 idiom for reading a contiguous dataset without a copy:
 * H5Dget_offset says where its image starts in the file, and the caller maps
 * the file itself. H5Dread has no way to hand back the file's own bytes, so
 * this — not H5Dread — is what the Rust view is measured against. */
static double view_and_sum(const char *path, size_t n) {
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(f);
    hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
    CHECK(d);
    haddr_t off = H5Dget_offset(d);
    if (off == HADDR_UNDEF) {
        fprintf(stderr, "FAIL dataset has no contiguous offset\n");
        exit(1);
    }
    size_t span = (size_t)off + n * sizeof(double);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(1);
    }
    const char *m = mmap(NULL, span, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    const double *v = (const double *)(m + off);
    double total = 0;
    for (size_t i = 0; i < n; i++) total += v[i];
    munmap((void *)m, span);
    close(fd);
    H5Dclose(d);
    H5Fclose(f);
    return total;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: probe_c <workdir> <workload> <reps>\n");
        return 2;
    }
    const char *workdir = argv[1];
    const char *wl = argv[2];
    int reps = atoi(argv[3]);
    char path[4096];

/* Variadic: brace initializers in the body contain commas the
 * preprocessor would otherwise split on. */
#define TIMED(...)                                                          \
    for (int rep = 0; rep < reps; rep++) {                                  \
        uint64_t t0 = now_ns();                                             \
        __VA_ARGS__;                                                        \
        printf("BENCH %s rep %d ns %" PRIu64 "\n", wl, rep, now_ns() - t0); \
    }

    if (strcmp(wl, "contig-write") == 0) {
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-contig.h5", workdir);
        TIMED({
            write_contig(path, data, CONTIG_N);
            remove(path);
        });
    } else if (strcmp(wl, "contig-read") == 0) {
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-contig-in.h5", workdir);
        write_contig(path, data, CONTIG_N);
        TIMED({ free(read_full(path, CONTIG_N)); });
    } else if (strcmp(wl, "contig-view") == 0) {
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-contig-in.h5", workdir);
        write_contig(path, data, CONTIG_N);
        free(data);
        TIMED({ sink = view_and_sum(path, CONTIG_N); });
    } else if (strcmp(wl, "into-read") == 0) {
        /* Buffer reuse: open and the first read are setup, so every timed
         * read lands in an already-faulted buffer. */
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-intoread-in.h5", workdir);
        write_contig(path, data, CONTIG_N);
        free(data);
        double *buf = malloc(CONTIG_N * sizeof(double));
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
        CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        TIMED({
            CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
            if (buf[CONTIG_N - 1] != (double)(CONTIG_N - 1)) abort();
        });
        H5Dclose(d);
        H5Fclose(f);
        free(buf);
    } else if (strcmp(wl, "into-slice") == 0) {
        /* The same reuse per piece: sequential 128 KiB slices into one
         * buffer, covering the dataset once per rep. */
        const int PIECE = 16 * 1024;
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-intoslice-in.h5", workdir);
        write_contig(path, data, CONTIG_N);
        free(data);
        double *buf = malloc(PIECE * sizeof(double));
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
        hid_t fsp = H5Dget_space(d);
        hsize_t count = PIECE;
        hid_t msp = H5Screate_simple(1, &count, NULL);
        hsize_t off0 = 0;
        CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, &off0, NULL, &count, NULL));
        CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, msp, fsp, H5P_DEFAULT, buf));
        TIMED({
            for (int k = 0; k < CONTIG_N / PIECE; k++) {
                hsize_t off = (hsize_t)k * PIECE;
                CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, &off, NULL,
                                          &count, NULL));
                CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, msp, fsp, H5P_DEFAULT, buf));
            }
            if (buf[PIECE - 1] != (double)(CONTIG_N - 1)) abort();
        });
        H5Sclose(msp);
        H5Sclose(fsp);
        H5Dclose(d);
        H5Fclose(f);
        free(buf);
    } else if (strcmp(wl, "chunked-write") == 0) {
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-chunked.h5", workdir);
        TIMED({
            write_chunked(path, data, CONTIG_N, 0, CHUNK_ELEMS);
            remove(path);
        });
    } else if (strcmp(wl, "chunked-read") == 0) {
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-chunked-in.h5", workdir);
        write_chunked(path, data, CONTIG_N, 0, CHUNK_ELEMS);
        TIMED({ free(read_full(path, CONTIG_N)); });
    } else if (strcmp(wl, "chunked-into-read") == 0) {
        /* The buffer-reuse pair again, on an unfiltered chunked dataset. */
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-chunkintoread-in.h5", workdir);
        write_chunked(path, data, CONTIG_N, 0, CHUNK_ELEMS);
        free(data);
        double *buf = malloc(CONTIG_N * sizeof(double));
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
        CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
        TIMED({
            CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf));
            if (buf[CONTIG_N - 1] != (double)(CONTIG_N - 1)) abort();
        });
        H5Dclose(d);
        H5Fclose(f);
        free(buf);
    } else if (strcmp(wl, "chunked-into-slice") == 0) {
        /* slice-read's random 64 KiB selections, into a kept buffer. */
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-chunkintoslice-in.h5", workdir);
        write_chunked(path, data, CONTIG_N, 0, CHUNK_ELEMS);
        free(data);
        double *buf = malloc(SLICE_ELEMS * sizeof(double));
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
        hid_t fsp = H5Dget_space(d);
        hsize_t count = SLICE_ELEMS;
        hid_t msp = H5Screate_simple(1, &count, NULL);
        hsize_t off0 = 0;
        CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, &off0, NULL, &count, NULL));
        CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, msp, fsp, H5P_DEFAULT, buf));
        TIMED({
            lcg_state = 1;
            hsize_t last = 0;
            for (int k = 0; k < SLICE_READS; k++) {
                hsize_t off = lcg_next() % (CONTIG_N - SLICE_ELEMS);
                CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, &off, NULL,
                                          &count, NULL));
                CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, msp, fsp, H5P_DEFAULT,
                              buf));
                last = off;
            }
            if (buf[0] != (double)last) abort();
        });
        H5Sclose(msp);
        H5Sclose(fsp);
        H5Dclose(d);
        H5Fclose(f);
        free(buf);
    } else if (strcmp(wl, "deflate-write") == 0) {
        double *data = compressible(DEFLATE_N);
        snprintf(path, sizeof path, "%s/c-deflate.h5", workdir);
        TIMED({
            write_chunked(path, data, DEFLATE_N, 1, CHUNK_ELEMS);
            remove(path);
        });
    } else if (strcmp(wl, "deflate-read") == 0) {
        double *data = compressible(DEFLATE_N);
        snprintf(path, sizeof path, "%s/c-deflate-in.h5", workdir);
        write_chunked(path, data, DEFLATE_N, 1, CHUNK_ELEMS);
        TIMED({ free(read_full(path, DEFLATE_N)); });
    } else if (strcmp(wl, "deflate-slice") == 0) {
        double *data = compressible(DEFLATE_N);
        snprintf(path, sizeof path, "%s/c-deflate-slice-in.h5", workdir);
        write_chunked(path, data, DEFLATE_N, 1, DSLICE_CHUNK);
        double *sbuf = malloc(SLICE_ELEMS * sizeof(double));
        TIMED({
            hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
            hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
            hid_t fsp = H5Dget_space(d);
            hsize_t count = SLICE_ELEMS;
            hid_t msp = H5Screate_simple(1, &count, NULL);
            size_t total = 0;
            for (int k = 0; k < DEFLATE_N / SLICE_ELEMS; k++) {
                hsize_t off = (hsize_t)k * SLICE_ELEMS;
                CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, &off, NULL,
                                          &count, NULL));
                CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, msp, fsp, H5P_DEFAULT,
                              sbuf));
                total += SLICE_ELEMS;
            }
            if (total != (size_t)DEFLATE_N) exit(1);
            H5Sclose(msp);
            H5Sclose(fsp);
            H5Dclose(d);
            H5Fclose(f);
        });
    } else if (strcmp(wl, "slice-read") == 0) {
        double *data = ramp(CONTIG_N);
        snprintf(path, sizeof path, "%s/c-slice-in.h5", workdir);
        write_chunked(path, data, CONTIG_N, 0, CHUNK_ELEMS);
        double *buf = malloc(SLICE_ELEMS * sizeof(double));
        TIMED({
            hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
            hid_t d = H5Dopen2(f, "data", H5P_DEFAULT);
            hid_t fsp = H5Dget_space(d);
            hsize_t count = SLICE_ELEMS;
            hid_t msp = H5Screate_simple(1, &count, NULL);
            lcg_state = 1;
            size_t total = 0;
            for (int k = 0; k < SLICE_READS; k++) {
                hsize_t off = lcg_next() % (CONTIG_N - SLICE_ELEMS);
                CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, &off, NULL,
                                          &count, NULL));
                CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, msp, fsp, H5P_DEFAULT,
                              buf));
                total += SLICE_ELEMS;
            }
            if (total != (size_t)SLICE_READS * SLICE_ELEMS) exit(1);
            H5Sclose(msp);
            H5Sclose(fsp);
            H5Dclose(d);
            H5Fclose(f);
        });
    } else if (strcmp(wl, "small-write") == 0) {
        double *data = ramp(SMALL_N);
        snprintf(path, sizeof path, "%s/c-small.h5", workdir);
        TIMED({
            hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hsize_t n = SMALL_N;
            hid_t sp = H5Screate_simple(1, &n, NULL);
            for (int i = 0; i < SMALL_DSETS; i++) {
                char name[16];
                snprintf(name, sizeof name, "d%04d", i);
                hid_t d = H5Dcreate2(f, name, H5T_NATIVE_DOUBLE, sp,
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                CHECK(d);
                CHECK(H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, data));
                H5Dclose(d);
            }
            H5Sclose(sp);
            H5Fclose(f);
            remove(path);
        });
    } else if (strcmp(wl, "small-read") == 0) {
        double *data = ramp(SMALL_N);
        snprintf(path, sizeof path, "%s/c-small-in.h5", workdir);
        hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t n = SMALL_N;
        hid_t sp = H5Screate_simple(1, &n, NULL);
        for (int i = 0; i < SMALL_DSETS; i++) {
            char name[16];
            snprintf(name, sizeof name, "d%04d", i);
            hid_t d = H5Dcreate2(f, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
            CHECK(H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           data));
            H5Dclose(d);
        }
        H5Sclose(sp);
        H5Fclose(f);
        double *buf = malloc(SMALL_N * sizeof(double));
        TIMED({
            hid_t fr = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
            size_t total = 0;
            for (int i = 0; i < SMALL_DSETS; i++) {
                char name[16];
                snprintf(name, sizeof name, "d%04d", i);
                hid_t d = H5Dopen2(fr, name, H5P_DEFAULT);
                CHECK(d);
                CHECK(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, buf));
                total += SMALL_N;
                H5Dclose(d);
            }
            if (total != (size_t)SMALL_DSETS * SMALL_N) exit(1);
            H5Fclose(fr);
        });
    } else if (strcmp(wl, "attr-write") == 0) {
        snprintf(path, sizeof path, "%s/c-attr.h5", workdir);
        TIMED({
            hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hid_t root = H5Gopen2(f, "/", H5P_DEFAULT);
            hid_t sp = H5Screate(H5S_SCALAR);
            for (int i = 0; i < ATTRS; i++) {
                char name[16];
                snprintf(name, sizeof name, "a%04d", i);
                double v = (double)i;
                hid_t a = H5Acreate2(root, name, H5T_NATIVE_DOUBLE, sp,
                                     H5P_DEFAULT, H5P_DEFAULT);
                CHECK(a);
                CHECK(H5Awrite(a, H5T_NATIVE_DOUBLE, &v));
                H5Aclose(a);
            }
            H5Sclose(sp);
            H5Gclose(root);
            H5Fclose(f);
            remove(path);
        });
    } else if (strcmp(wl, "append") == 0) {
        float *batch = malloc((size_t)APPEND_BATCH * APPEND_COLS * sizeof(float));
        for (size_t i = 0; i < (size_t)APPEND_BATCH * APPEND_COLS; i++)
            batch[i] = (float)i;
        snprintf(path, sizeof path, "%s/c-append.h5", workdir);
        TIMED({
            hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hsize_t dims[2] = {0, APPEND_COLS};
            hsize_t maxdims[2] = {H5S_UNLIMITED, APPEND_COLS};
            hsize_t chunk[2] = {APPEND_BATCH, APPEND_COLS};
            hid_t sp = H5Screate_simple(2, dims, maxdims);
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            CHECK(H5Pset_chunk(dcpl, 2, chunk));
            hid_t d = H5Dcreate2(f, "data", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT,
                                 dcpl, H5P_DEFAULT);
            CHECK(d);
            hsize_t mcount[2] = {APPEND_BATCH, APPEND_COLS};
            hid_t msp = H5Screate_simple(2, mcount, NULL);
            for (int b = 0; b < APPEND_ROWS / APPEND_BATCH; b++) {
                hsize_t newdims[2] = {(hsize_t)(b + 1) * APPEND_BATCH,
                                      APPEND_COLS};
                CHECK(H5Dset_extent(d, newdims));
                hid_t fsp = H5Dget_space(d);
                hsize_t start[2] = {(hsize_t)b * APPEND_BATCH, 0};
                CHECK(H5Sselect_hyperslab(fsp, H5S_SELECT_SET, start, NULL,
                                          mcount, NULL));
                CHECK(H5Dwrite(d, H5T_NATIVE_FLOAT, msp, fsp, H5P_DEFAULT,
                               batch));
                H5Sclose(fsp);
            }
            H5Sclose(msp);
            H5Dclose(d);
            H5Pclose(dcpl);
            H5Sclose(sp);
            H5Fclose(f);
            remove(path);
        });
    } else {
        fprintf(stderr, "unknown workload %s\n", wl);
        return 2;
    }
    return 0;
}
