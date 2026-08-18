//! Performance probe. `perf/probe.c` is the line-for-line C mirror of these
//! workloads against libhdf5; `perf/run.py` builds both, runs them on the
//! same matrix, and prints the comparison. Every workload here must keep the
//! same shapes, dtypes, chunk sizes, deflate level, and access pattern as the
//! C side — the two probes differ only in which library does the work.
//!
//! Usage: `perf_probe <workdir> <workload> <reps>`
//! Output: one `BENCH <workload> rep <i> ns <elapsed>` line per rep;
//! setup (input-file creation, data generation) is untimed.

use rust_hdf5::H5File;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CONTIG_N: usize = 16 * 1024 * 1024; // f64 -> 128 MiB
const CHUNK_ELEMS: usize = 256 * 1024; // 2 MiB chunks
const DEFLATE_N: usize = 8 * 1024 * 1024; // f64 -> 64 MiB
const DEFLATE_LEVEL: u32 = 6;
const SLICE_READS: usize = 1000;
const SLICE_ELEMS: usize = 8192;
// deflate-slice: chunks small enough for libhdf5's default 1 MiB chunk
// cache, so its reuse across consecutive slices is part of the workload.
const DSLICE_CHUNK: usize = 32 * 1024; // 256 KiB chunks
const SMALL_DSETS: usize = 2000;
const SMALL_N: usize = 128;
const ATTRS: usize = 1000;
const APPEND_COLS: usize = 4096;
const APPEND_ROWS: usize = 1024;
const APPEND_BATCH: usize = 16;

fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

fn compressible(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i & 0xFF) as f64).collect()
}

/// Same LCG as the C probe so both read the same slice offsets.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn timed<F: FnMut()>(name: &str, reps: usize, mut f: F) {
    for i in 0..reps {
        let t = Instant::now();
        f();
        println!("BENCH {name} rep {i} ns {}", t.elapsed().as_nanos());
    }
}

fn write_contig(path: &Path, data: &[f64]) {
    let file = H5File::create(path).unwrap();
    let ds = file
        .new_dataset::<f64>()
        .shape([data.len()])
        .create("data")
        .unwrap();
    ds.write_raw(data).unwrap();
    file.close().unwrap();
}

fn write_chunked(path: &Path, data: &[f64], deflate: bool, chunk_elems: usize) {
    let file = H5File::create(path).unwrap();
    let mut b = file
        .new_dataset::<f64>()
        .shape([data.len()])
        .chunk(&[chunk_elems]);
    if deflate {
        b = b.deflate(DEFLATE_LEVEL);
    }
    let ds = b.create("data").unwrap();
    ds.write_raw(data).unwrap();
    file.close().unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let workdir = PathBuf::from(args.next().expect("workdir"));
    let workload = args.next().expect("workload");
    let reps: usize = args.next().expect("reps").parse().unwrap();
    let p = |name: &str| workdir.join(name);

    match workload.as_str() {
        "contig-write" => {
            let data = ramp(CONTIG_N);
            let path = p("rs-contig.h5");
            timed(&workload, reps, || {
                write_contig(&path, &data);
                std::fs::remove_file(&path).unwrap();
            });
        }
        "contig-read" => {
            let path = p("rs-contig-in.h5");
            write_contig(&path, &ramp(CONTIG_N));
            timed(&workload, reps, || {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                let v = ds.read_raw::<f64>().unwrap();
                assert_eq!(v.len(), CONTIG_N);
            });
        }
        "chunked-write" => {
            let data = ramp(CONTIG_N);
            let path = p("rs-chunked.h5");
            timed(&workload, reps, || {
                write_chunked(&path, &data, false, CHUNK_ELEMS);
                std::fs::remove_file(&path).unwrap();
            });
        }
        "chunked-read" => {
            let path = p("rs-chunked-in.h5");
            write_chunked(&path, &ramp(CONTIG_N), false, CHUNK_ELEMS);
            timed(&workload, reps, || {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                let v = ds.read_raw::<f64>().unwrap();
                assert_eq!(v.len(), CONTIG_N);
            });
        }
        "deflate-write" => {
            let data = compressible(DEFLATE_N);
            let path = p("rs-deflate.h5");
            timed(&workload, reps, || {
                write_chunked(&path, &data, true, CHUNK_ELEMS);
                std::fs::remove_file(&path).unwrap();
            });
        }
        "deflate-read" => {
            let path = p("rs-deflate-in.h5");
            write_chunked(&path, &compressible(DEFLATE_N), true, CHUNK_ELEMS);
            timed(&workload, reps, || {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                let v = ds.read_raw::<f64>().unwrap();
                assert_eq!(v.len(), DEFLATE_N);
            });
        }
        "slice-read" => {
            let path = p("rs-slice-in.h5");
            write_chunked(&path, &ramp(CONTIG_N), false, CHUNK_ELEMS);
            timed(&workload, reps, || {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                let mut rng = Lcg(1);
                let mut total = 0usize;
                for _ in 0..SLICE_READS {
                    let off = (rng.next() % (CONTIG_N - SLICE_ELEMS) as u64) as usize;
                    let v = ds.read_slice::<f64>(&[off], &[SLICE_ELEMS]).unwrap();
                    total += v.len();
                }
                assert_eq!(total, SLICE_READS * SLICE_ELEMS);
            });
        }
        "deflate-slice" => {
            let path = p("rs-deflate-slice-in.h5");
            write_chunked(&path, &compressible(DEFLATE_N), true, DSLICE_CHUNK);
            timed(&workload, reps, || {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                let mut total = 0usize;
                for k in 0..DEFLATE_N / SLICE_ELEMS {
                    let v = ds
                        .read_slice::<f64>(&[k * SLICE_ELEMS], &[SLICE_ELEMS])
                        .unwrap();
                    total += v.len();
                }
                assert_eq!(total, DEFLATE_N);
            });
        }
        "small-write" => {
            let data = ramp(SMALL_N);
            let path = p("rs-small.h5");
            timed(&workload, reps, || {
                let file = H5File::create(&path).unwrap();
                for i in 0..SMALL_DSETS {
                    let ds = file
                        .new_dataset::<f64>()
                        .shape([SMALL_N])
                        .create(&format!("d{i:04}"))
                        .unwrap();
                    ds.write_raw(&data).unwrap();
                }
                file.close().unwrap();
                std::fs::remove_file(&path).unwrap();
            });
        }
        "small-read" => {
            let path = p("rs-small-in.h5");
            let data = ramp(SMALL_N);
            let file = H5File::create(&path).unwrap();
            for i in 0..SMALL_DSETS {
                let ds = file
                    .new_dataset::<f64>()
                    .shape([SMALL_N])
                    .create(&format!("d{i:04}"))
                    .unwrap();
                ds.write_raw(&data).unwrap();
            }
            file.close().unwrap();
            timed(&workload, reps, || {
                let file = H5File::open(&path).unwrap();
                let mut total = 0usize;
                for i in 0..SMALL_DSETS {
                    let ds = file.dataset(&format!("d{i:04}")).unwrap();
                    total += ds.read_raw::<f64>().unwrap().len();
                }
                assert_eq!(total, SMALL_DSETS * SMALL_N);
            });
        }
        "attr-write" => {
            let path = p("rs-attr.h5");
            timed(&workload, reps, || {
                let file = H5File::create(&path).unwrap();
                for i in 0..ATTRS {
                    file.set_attr_numeric(&format!("a{i:04}"), &(i as f64))
                        .unwrap();
                }
                file.close().unwrap();
                std::fs::remove_file(&path).unwrap();
            });
        }
        "append" => {
            let batch: Vec<f32> = (0..APPEND_BATCH * APPEND_COLS).map(|i| i as f32).collect();
            let path = p("rs-append.h5");
            timed(&workload, reps, || {
                let file = H5File::create(&path).unwrap();
                let ds = file
                    .new_dataset::<f32>()
                    .shape([0, APPEND_COLS])
                    .chunk(&[APPEND_BATCH, APPEND_COLS])
                    .max_shape(&[None, Some(APPEND_COLS)])
                    .create("data")
                    .unwrap();
                for _ in 0..APPEND_ROWS / APPEND_BATCH {
                    ds.append(&batch).unwrap();
                }
                file.close().unwrap();
                std::fs::remove_file(&path).unwrap();
            });
        }
        other => {
            eprintln!("unknown workload {other}");
            std::process::exit(2);
        }
    }
}
