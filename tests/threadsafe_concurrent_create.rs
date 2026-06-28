//! Concurrent create + append + metadata stress test for the `threadsafe`
//! writer (Stage 3d of `docs/threadsafe-fine-grained-locking.md`).
//!
//! Where `threadsafe_parallel_write.rs` exercises concurrent *writes* to
//! datasets created up front, this test drives the paths Stage 3c moved onto
//! the shared read guard: every thread, holding only `&H5File`, **creates**
//! its own unlimited chunked dataset, **appends** to it in several batches
//! (create → extend → append → write_chunk), and writes a per-dataset
//! **string attribute** (a global-heap metadata write). All of this runs
//! concurrently across threads on distinct datasets, so the only
//! serialization is the per-slot mutex (never contended here — distinct
//! datasets), the brief spine mutex (registry push/clone), the atomic
//! allocator, and positioned pwrite.
//!
//! Every iteration reopens the file and asserts each dataset's full contents
//! and its attribute; the run repeats so a data race surfaces as a wrong or
//! short read rather than passing silently. h5py reads the result back on the
//! first iteration for standard-tool correctness.
//!
//! Gated on `threadsafe` (the concurrency under test). Present under the
//! default feature set plus `--features threadsafe`.

#![cfg(feature = "threadsafe")]

use rust_hdf5::types::VarLenUnicode;
use rust_hdf5::H5File;

const TEST_PYTHON: &str = "/Users/stevek/mamba/envs/bs2026.1/bin/python";

fn python() -> Option<&'static str> {
    if std::path::Path::new(TEST_PYTHON).exists() {
        Some(TEST_PYTHON)
    } else {
        eprintln!("skipping h5py cross-check: {TEST_PYTHON} not present");
        None
    }
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_cc_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// Value at element `i` of dataset `k`. Distinct per (k, i) so a cross-dataset
/// or cross-chunk mix-up shows up as a wrong value on read-back.
fn value(k: usize, i: usize) -> i32 {
    (k * 1_000_000 + i) as i32
}

#[test]
fn concurrent_create_append_metadata_distinct_datasets() {
    const N: usize = 8; // threads / datasets
    const BATCHES: usize = 8; // appends per dataset
    const BATCH: usize = 32; // elements per append
    const LEN: usize = BATCHES * BATCH; // 256 elements per dataset
    const CHUNK: usize = 16; // -> several chunks, partial-chunk buffering exercised
    const ITERS: usize = 16; // repeat to shake out races

    for iter in 0..ITERS {
        let path = tmp(&format!("iter{iter}"));

        {
            let file = H5File::create(&path).unwrap();

            // Each thread owns one dataset end to end: create, append in
            // batches, then attach a string attribute. Threads share only
            // `&file` (so creation itself runs under the read guard).
            std::thread::scope(|s| {
                for k in 0..N {
                    let file = &file;
                    s.spawn(move || {
                        let ds = file
                            .new_dataset::<i32>()
                            .shape([0])
                            .chunk(&[CHUNK])
                            .max_shape(&[None])
                            .create(&format!("ds{k}"))
                            .unwrap_or_else(|e| panic!("create ds{k}: {e}"));

                        for b in 0..BATCHES {
                            let batch: Vec<i32> =
                                (0..BATCH).map(|i| value(k, b * BATCH + i)).collect();
                            ds.append(&batch)
                                .unwrap_or_else(|e| panic!("append ds{k} batch {b}: {e}"));
                        }

                        let attr = ds
                            .new_attr::<VarLenUnicode>()
                            .shape(())
                            .create("origin")
                            .unwrap_or_else(|e| panic!("create attr ds{k}: {e}"));
                        attr.write_string(&format!("thread-{k}"))
                            .unwrap_or_else(|e| panic!("write attr ds{k}: {e}"));
                    });
                }
            });

            file.close().unwrap();
        }

        // Reopen and assert every dataset's full contents and its attribute.
        // A race in the allocator, the positioned writes, the per-slot index,
        // or the registry push would surface here.
        {
            let file = H5File::open(&path).unwrap();
            for k in 0..N {
                let ds = file.dataset(&format!("ds{k}")).unwrap();
                let got = ds.read_raw::<i32>().unwrap();
                let want: Vec<i32> = (0..LEN).map(|i| value(k, i)).collect();
                assert_eq!(
                    got.len(),
                    LEN,
                    "dataset ds{k} length mismatch on iter {iter}"
                );
                assert_eq!(got, want, "dataset ds{k} content mismatch on iter {iter}");

                let origin = ds
                    .attr("origin")
                    .and_then(|a| a.read_string())
                    .unwrap_or_else(|e| panic!("read attr ds{k} iter {iter}: {e}"));
                assert_eq!(
                    origin,
                    format!("thread-{k}"),
                    "dataset ds{k} attribute mismatch on iter {iter}"
                );
            }
        }

        // h5py reads the concurrently-created datasets + attributes back on
        // the first iteration: confirms standard-tool readability, not merely
        // round-trippability by our own reader.
        if iter == 0 {
            if let Some(py) = python() {
                let script = format!(
                    "import h5py, numpy as np\n\
                     f = h5py.File(r'{}', 'r')\n\
                     for k in range({N}):\n\
                     \x20   d = f['ds%d' % k]\n\
                     \x20   v = d[...]\n\
                     \x20   exp = np.arange({LEN}, dtype='i8') + k * 1000000\n\
                     \x20   assert v.shape == ({LEN},), (k, v.shape)\n\
                     \x20   assert (v.astype('i8') == exp).all(), (k, v[:8], exp[:8])\n\
                     \x20   assert d.attrs['origin'] == ('thread-%d' % k), (k, d.attrs['origin'])\n\
                     f.close()\n",
                    path.display()
                );
                let status = std::process::Command::new(py)
                    .arg("-c")
                    .arg(&script)
                    .status()
                    .expect("failed to spawn python");
                assert!(status.success(), "h5py cross-check failed for {path:?}");
            }
        }

        std::fs::remove_file(&path).ok();
    }
}
