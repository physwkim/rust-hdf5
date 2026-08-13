//! Concurrent same-dataset appends under the `threadsafe` writer.
//!
//! An append is several dataset-slot acquisitions: take the buffered tail,
//! write the chunk-aligned rows, re-buffer the new tail, extend the extent.
//! The per-slot mutex serializes each acquisition but not the operation, so
//! without the per-dataset op lock two appends interleave between
//! acquisitions — two calls take the same base row, rows are lost or
//! overwritten. These tests drive that exact schedule: N threads append to
//! *one* dataset, with a frame count that never divides the chunk so the
//! buffered tail is always in play.
//!
//! Every append call must land as a contiguous, in-order block of rows
//! (atomicity), every appended frame must appear exactly once (no lost or
//! doubled rows), and no frame may be torn. Order *between* calls is the
//! scheduler's.

#![cfg(feature = "threadsafe")]

use rust_hdf5::H5File;

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_same_ds_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// Tag of frame `f` of append call `j` by thread `k`. Distinct per
/// (k, j, f) so a lost, doubled, or split append shows up as a wrong
/// multiset or a broken run on read-back.
fn tag(k: usize, j: usize, f: usize) -> i32 {
    (k * 1_000_000 + j * 100 + f) as i32
}

#[test]
fn concurrent_same_dataset_appends_serialize_wholly() {
    const N: usize = 4; // threads
    const M: usize = 40; // append calls per thread
    const FRAMES: usize = 3; // frames per call — never a multiple of CHUNK0
    const W: usize = 8; // row width
    const CHUNK0: usize = 4; // chunk rows: every call leaves a buffered tail
    const ITERS: usize = 10; // repeat to shake out schedules

    for iter in 0..ITERS {
        let path = tmp(&format!("iter{iter}"));
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, W])
                .chunk(&[CHUNK0, W])
                .max_shape(&[None, Some(W)])
                .create("d")
                .unwrap();

            std::thread::scope(|s| {
                for k in 0..N {
                    let ds = &ds;
                    s.spawn(move || {
                        for j in 0..M {
                            let data: Vec<i32> = (0..FRAMES)
                                .flat_map(|f| std::iter::repeat_n(tag(k, j, f), W))
                                .collect();
                            ds.append(&data)
                                .unwrap_or_else(|e| panic!("append k={k} j={j}: {e}"));
                        }
                    });
                }
            });
            file.close().unwrap();
        }

        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape(), vec![N * M * FRAMES, W], "iter {iter}");
        let got = ds.read_raw::<i32>().unwrap();

        // No torn frames: each row is uniform.
        let rows: Vec<i32> = got
            .chunks_exact(W)
            .enumerate()
            .map(|(r, row)| {
                assert!(
                    row.iter().all(|&v| v == row[0]),
                    "iter {iter}: torn frame at row {r}: {row:?}"
                );
                row[0]
            })
            .collect();

        // Atomicity: each call's frames form one contiguous in-order run.
        // Walking runs also proves every frame appears exactly once — the
        // walk consumes exactly FRAMES rows per (k, j) and the total length
        // already matched.
        let mut seen = std::collections::HashSet::new();
        let mut r = 0;
        while r < rows.len() {
            let first = rows[r];
            let (k, rem) = ((first / 1_000_000) as usize, first % 1_000_000);
            let (j, f) = ((rem / 100) as usize, (rem % 100) as usize);
            assert_eq!(f, 0, "iter {iter}: run at row {r} starts mid-call: {first}");
            for f in 0..FRAMES {
                assert_eq!(
                    rows[r + f],
                    tag(k, j, f),
                    "iter {iter}: call (k={k}, j={j}) split at row {}",
                    r + f
                );
            }
            assert!(
                seen.insert((k, j)),
                "iter {iter}: call (k={k}, j={j}) appended twice"
            );
            r += FRAMES;
        }
        assert_eq!(seen.len(), N * M, "iter {iter}");

        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn concurrent_same_dataset_vlen_appends_serialize_wholly() {
    const N: usize = 4; // threads
    const M: usize = 30; // append calls per thread
    const STRINGS: usize = 3; // strings per call
    const CHUNK: usize = 4; // never a multiple of STRINGS: tail always buffered
    const ITERS: usize = 10;

    for iter in 0..ITERS {
        let path = tmp(&format!("vlen{iter}"));
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("strs", CHUNK, None)
                .unwrap();

            std::thread::scope(|s| {
                for k in 0..N {
                    let file = &file;
                    s.spawn(move || {
                        for j in 0..M {
                            let batch: Vec<String> =
                                (0..STRINGS).map(|f| format!("{k}:{j}:{f}")).collect();
                            let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
                            file.append_vlen_strings("strs", &refs)
                                .unwrap_or_else(|e| panic!("append k={k} j={j}: {e}"));
                        }
                    });
                }
            });
            file.close().unwrap();
        }

        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("strs").unwrap();
        let got = ds.read_vlen_strings().unwrap();
        assert_eq!(got.len(), N * M * STRINGS, "iter {iter}");

        // Each call's strings must form one contiguous in-order run, and the
        // runs must cover every (k, j) exactly once.
        let mut seen = std::collections::HashSet::new();
        let mut r = 0;
        while r < got.len() {
            let parts: Vec<usize> = got[r].split(':').map(|p| p.parse().unwrap()).collect();
            let (k, j, f) = (parts[0], parts[1], parts[2]);
            assert_eq!(
                f, 0,
                "iter {iter}: run at element {r} starts mid-call: {}",
                got[r]
            );
            for f in 0..STRINGS {
                assert_eq!(
                    got[r + f],
                    format!("{k}:{j}:{f}"),
                    "iter {iter}: call (k={k}, j={j}) split at element {}",
                    r + f
                );
            }
            assert!(
                seen.insert((k, j)),
                "iter {iter}: call (k={k}, j={j}) appended twice"
            );
            r += STRINGS;
        }
        assert_eq!(seen.len(), N * M, "iter {iter}");

        std::fs::remove_file(&path).ok();
    }
}
