//! Reading libhdf5-written **single-chunk filtered** datasets, honoring the
//! per-chunk filter mask stored inline in the data-layout message.
//!
//! When a chunked dataset has exactly one whole-array chunk and fixed dims,
//! libhdf5 uses the Single Chunk index and records the chunk's on-disk size
//! and per-chunk filter mask inline in the layout message (H5Olayout.c).
//! A set mask bit means the corresponding filter was *not* applied to the
//! chunk and must be skipped on read — reversing the full pipeline would
//! corrupt the data (or error trying to inflate raw bytes).
//!
//! Fixtures are libhdf5/h5py output (see `tests/fixtures/gen_single_chunk.py`);
//! both hold `u8[64] == 0..64`.
//!
//! Both fixtures are deflate-filtered datasets, so the whole module requires
//! the `deflate` feature (mask 0 inflates the chunk; even mask 1 needs the
//! pipeline present). Without it this compiles to an empty test binary.
#![cfg(feature = "deflate")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// `mask == 1`: the single chunk was written raw (gzip marked not-applied).
const MASK1: &[u8] = include_bytes!("fixtures/single_chunk_deflate_mask1.h5");
/// `mask == 0`: the single chunk was gzip-compressed normally.
const MASK0: &[u8] = include_bytes!("fixtures/single_chunk_deflate_mask0.h5");

fn write_temp(label: &str, bytes: &[u8]) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_single_chunk_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.h5"));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Regression: a single-chunk dataset whose one chunk carries `filter_mask=1`
/// (gzip skipped) must read back the raw bytes verbatim, NOT be run through
/// inflate. Before the reader honored the mask this returned an inflate error
/// or garbage.
#[test]
fn single_chunk_filter_mask1_skips_the_filter() {
    let path = write_temp("mask1", MASK1);
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("x").unwrap();
    let got = ds.read_raw::<u8>().unwrap();
    assert_eq!(
        got,
        (0u8..64).collect::<Vec<u8>>(),
        "single chunk with filter_mask=1 must be read as raw stored bytes"
    );
    drop(file);
    cleanup(&path);
}

/// Control: a single-chunk dataset whose one chunk was gzip-compressed
/// normally (`filter_mask=0`) still round-trips — the mask-aware reverse with
/// mask 0 reverses the full pipeline, matching the prior behavior.
#[test]
fn single_chunk_filter_mask0_reverses_the_filter() {
    let path = write_temp("mask0", MASK0);
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("y").unwrap();
    let got = ds.read_raw::<u8>().unwrap();
    assert_eq!(
        got,
        (0u8..64).collect::<Vec<u8>>(),
        "single chunk with filter_mask=0 must inflate back to the original"
    );
    drop(file);
    cleanup(&path);
}
