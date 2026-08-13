//! Parallel-write stress test for the `threadsafe` writer.
//!
//! Stage 2b-ii of the fine-grained-locking plan
//! (docs/threadsafe-fine-grained-locking.md). N threads each compress and
//! write their *own* chunked+deflate dataset concurrently. This exercises the
//! whole concurrency story:
//!
//! - the outer `RwLock` is held in *read* mode by every writing thread at once,
//! - the lock-free atomic allocator hands each thread disjoint file space,
//! - positioned `pwrite` lets the disjoint writes proceed without a shared
//!   cursor,
//! - each dataset's per-`Slot` mutex serializes only same-dataset access.
//!
//! The design's target win — compression of *different* datasets overlapping
//! instead of serializing on one global lock — is what runs here. Every
//! iteration reopens the file and asserts each dataset's full contents, and the
//! run repeats so a data race shows up as a corrupted read rather than passing
//! silently. h5py reads the compressed result back for standard-tool +
//! filter correctness on the first iteration.
//!
//! Gated on `threadsafe` (the concurrency under test) and `deflate` (the
//! compression that must run off the lock). Both are present under the default
//! feature set plus `--features threadsafe`.

#![cfg(all(feature = "threadsafe", feature = "deflate"))]

use rust_hdf5::H5File;

const TEST_PYTHON: &str = "/Users/stevek/mamba/envs/bs2026.1/bin/python";

fn python() -> Option<&'static str> {
    static PY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PY.get_or_init(|| {
        let candidate =
            std::env::var("RUST_HDF5_TEST_PYTHON").unwrap_or_else(|_| TEST_PYTHON.to_string());
        if std::path::Path::new(&candidate).exists() {
            Some(candidate)
        } else {
            eprintln!("skipping h5py cross-check: {candidate} not present");
            None
        }
    })
    .as_deref()
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_par_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// Value written at element `i` of dataset `k`. Distinct per (k, i) so a cross-
/// dataset or cross-chunk mix-up surfaces as a wrong value on read-back. Stays
/// within `i32` for `k < 64`, `i < 1_000_000`.
fn value(k: usize, i: usize) -> i32 {
    (k * 1_000_000 + i) as i32
}

#[test]
fn parallel_multi_dataset_compressed_write_roundtrips() {
    const N: usize = 8; // datasets / threads
    const LEN: usize = 4096; // elements per dataset
    const CHUNK: usize = 256; // -> 16 chunks per dataset
    const ITERS: usize = 20; // repeat to shake out races

    for iter in 0..ITERS {
        let path = tmp(&format!("iter{iter}"));

        {
            let file = H5File::create(&path).unwrap();
            // Create every dataset up front. Creation mutates the shared
            // registry + allocator, so it takes the write guard and is serial
            // by design — the parallel win is in the write below.
            let mut handles = Vec::with_capacity(N);
            for k in 0..N {
                let ds = file
                    .new_dataset::<i32>()
                    .shape([LEN])
                    .chunk(&[CHUNK])
                    .deflate(4)
                    .create(&format!("ds{k}"))
                    .unwrap();
                handles.push((k, ds));
            }

            // Concurrent writes: each thread compresses and writes its own
            // dataset under a shared read guard. Different datasets => no slot
            // contention, so the deflate work genuinely overlaps.
            std::thread::scope(|s| {
                for (k, ds) in handles {
                    s.spawn(move || {
                        let data: Vec<i32> = (0..LEN).map(|i| value(k, i)).collect();
                        ds.write_raw(&data)
                            .unwrap_or_else(|e| panic!("write_raw ds{k}: {e}"));
                    });
                }
            });

            file.close().unwrap();
        }

        // Reopen and assert every dataset's full contents. A race in the
        // allocator, the positioned writes, or the per-slot index would show
        // up here as a wrong or short read.
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
            }
        }

        // h5py reads the parallel-written, deflate-compressed datasets back on
        // the first iteration: confirms the bytes + filter pipeline are
        // standard-tool readable, not merely round-trippable by our own reader.
        if iter == 0 {
            if let Some(py) = python() {
                let script = format!(
                    "import h5py, numpy as np\n\
                     f = h5py.File(r'{}', 'r')\n\
                     for k in range({N}):\n\
                     \x20   d = f['ds%d' % k][...]\n\
                     \x20   exp = np.arange({LEN}, dtype='i8') + k * 1000000\n\
                     \x20   assert d.shape == ({LEN},), (k, d.shape)\n\
                     \x20   assert (d.astype('i8') == exp).all(), (k, d[:8], exp[:8])\n\
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
