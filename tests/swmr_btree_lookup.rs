//! A SWMR reader keeps the B-tree nodes its chunk lookups decode, while the
//! writer keeps inserting. The writer lays every node a flush changes out in a
//! fresh block, so the memoized tree stays intact but lacks the records written
//! since; a lookup that tree misses must still find its chunk.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};
use rust_hdf5::{FileLocking, H5Type};

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_swmr_btree_{label}_{}_{n}.h5",
        std::process::id()
    ))
}

/// A 400 x 1 grid of one-byte chunks with two unlimited axes, so a v2 B-tree
/// index, holding chunk `[i, 0] = i` for `i` in `0..200`: 24-byte records, 84
/// to a leaf, so a root over leaves of 84, 84 and 30.
fn grid(path: &Path) -> (SwmrFileWriter, usize) {
    let mut w = SwmrFileWriter::create_with_locking(path, FileLocking::Disabled).unwrap();
    let ds = w
        .create_grid_dataset_with_max(
            "grid",
            u8::hdf5_type(),
            &[400, 1],
            &[None, None],
            &[1, 1],
            None,
        )
        .unwrap();
    w.start_swmr().unwrap();
    for i in 0..200u8 {
        w.write_chunk_at(ds, &[u64::from(i), 0], &[i]).unwrap();
    }
    w.flush().unwrap();
    (w, ds)
}

#[test]
fn a_record_written_after_the_memoized_tree_is_found() {
    let path = tmp("appended");
    let (mut w, ds) = grid(&path);
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    assert_eq!(
        r.read_chunk_raw("grid", &[0, 0]).unwrap(),
        Some((vec![0], 0))
    );

    // 201 records: chunk 200 goes into a copy of the last leaf, and the
    // reader's memoized root still sends it to the 30-record original.
    w.write_chunk_at(ds, &[200, 0], &[200]).unwrap();
    w.flush().unwrap();
    assert_eq!(
        r.read_chunk_raw("grid", &[200, 0]).unwrap(),
        Some((vec![200], 0))
    );
    assert_eq!(
        r.read_chunk_raw("grid", &[67, 0]).unwrap(),
        Some((vec![67], 0))
    );
    w.close().unwrap();
    let _ = std::fs::remove_file(&path);
}
