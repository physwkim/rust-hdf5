//! Integration tests for `H5Dataset::set_extent`.
//!
//! Unlike `extend`, `set_extent` can shrink a chunked dataset's logical
//! dimensions — the way to correct an over-extended frame count after a
//! partial multi-frame chunk. Data in chunks beyond the new extent stays in
//! the file but is no longer visible on read.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_set_extent_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// `set_extent` shrinks the logical frame count; the reader returns exactly
/// the retained frames, and the written data within them is intact.
#[test]
fn set_extent_shrinks_logical_extent() {
    let path = unique_tmp("shrink");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0usize, 4])
            .chunk(&[1, 4])
            .max_shape(&[None, Some(4)])
            .create("data")
            .unwrap();

        for f in 0..5i32 {
            let row = [f * 10, f * 10 + 1, f * 10 + 2, f * 10 + 3];
            let raw: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            ds.write_chunk(f as usize, &raw).unwrap();
        }
        ds.extend(&[5, 4]).unwrap();

        // Correct the extent down to 3 frames.
        ds.set_extent(&[3, 4]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.shape(), vec![3, 4]);
        let v = ds.read_raw::<i32>().unwrap();
        assert_eq!(v.len(), 12);
        for f in 0..3i32 {
            for c in 0..4usize {
                assert_eq!(v[f as usize * 4 + c], f * 10 + c as i32);
            }
        }
    }

    cleanup(&path);
}

/// `set_extent` rejects an extent above the dataset's maximum dimensions.
#[test]
fn set_extent_rejects_exceeding_max() {
    let path = unique_tmp("overmax");
    let file = H5File::create(&path).unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shape([0usize, 4])
        .chunk(&[1, 4])
        .max_shape(&[None, Some(4)])
        .create("data")
        .unwrap();

    // Dimension 1 is capped at 4; growing it to 9 must be rejected.
    let err = ds.set_extent(&[3, 9]).expect_err("should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("maximum"), "unexpected error: {msg}");

    drop(file);
    cleanup(&path);
}
