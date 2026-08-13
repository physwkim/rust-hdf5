//! Reopened fixed-array and v2-B-tree chunk indexes.
//!
//! `open_append` used to rebuild only the extensible-array index and leave
//! FA/BT2 datasets as re-link placeholders: writes to them were refused,
//! and — worse — deleting one freed just the attributes and object header
//! while every chunk block and the index structures leaked. The index is
//! now read back like the EA one, so reopened FA/BT2 datasets take writes
//! and deletes reclaim their storage.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_reopen_index_{}_{}_{}",
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

fn make_fa(file: &H5File, vals: &[i32]) {
    let ds = file
        .new_dataset::<i32>()
        .shape([4usize, 4])
        .chunk(&[1, 4])
        .max_shape(&[Some(8), Some(4)])
        .create("grid")
        .unwrap();
    ds.write_slice(&[0, 0], &[4, 4], vals).unwrap();
}

fn make_bt2(file: &H5File, vals: &[i32]) {
    let ds = file
        .new_dataset::<i32>()
        .shape([4usize, 4])
        .chunk(&[2, 2])
        .max_shape(&[None, None])
        .create("tiles")
        .unwrap();
    ds.write_slice(&[0, 0], &[4, 4], vals).unwrap();
}

/// Deleting a reopened FA dataset must free the chunks it read back off
/// the disk, plus the FA header and data block — the settled-size oracle
/// of `delete_reclamation.rs`, across sessions.
#[test]
fn reopen_session_delete_frees_fixed_array_storage() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("fa_del_{cycles}"));
        let vals: Vec<i32> = (0..16).collect();
        {
            let file = H5File::create(&path).unwrap();
            make_fa(&file, &vals);
            file.close().unwrap();
        }
        for _ in 0..cycles {
            let file = H5File::options().no_locking().open_rw(&path).unwrap();
            file.delete_dataset("grid").unwrap();
            make_fa(&file, &vals);
            file.close().unwrap();
        }
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("grid").unwrap().read_raw::<i32>().unwrap(),
            vals
        );
        drop(read);
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(10), size_after(2), "10 reopen cycles against 2");
}

/// Deleting a reopened BT2 dataset must free the chunks its tree names,
/// the node blocks, and the header.
#[test]
fn reopen_session_delete_frees_btree_v2_storage() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("bt2_del_{cycles}"));
        let vals: Vec<i32> = (100..116).collect();
        {
            let file = H5File::create(&path).unwrap();
            make_bt2(&file, &vals);
            file.close().unwrap();
        }
        for _ in 0..cycles {
            let file = H5File::options().no_locking().open_rw(&path).unwrap();
            file.delete_dataset("tiles").unwrap();
            make_bt2(&file, &vals);
            file.close().unwrap();
        }
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("tiles").unwrap().read_raw::<i32>().unwrap(),
            vals
        );
        drop(read);
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(10), size_after(2), "10 reopen cycles against 2");
}

/// A reopened FA dataset takes writes into chunk slots the first session
/// never allocated, and the second session's index flush makes them
/// readable — the flush used to be gated on the EA index alone.
#[test]
fn reopened_fixed_array_dataset_takes_new_chunks() {
    let path = unique_tmp("fa_write");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([8usize, 4])
            .chunk(&[1, 4])
            .max_shape(&[Some(8), Some(4)])
            .create("grid")
            .unwrap();
        // Rows 0..4 only; rows 4..8 stay unallocated (incremental).
        let top: Vec<i32> = (0..16).collect();
        ds.write_slice(&[0, 0], &[4, 4], &top).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        let ds = file.dataset_writer("grid").unwrap();
        let bottom: Vec<i32> = (16..32).collect();
        ds.write_slice(&[4, 0], &[4, 4], &bottom).unwrap();
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let all: Vec<i32> = (0..32).collect();
    assert_eq!(
        file.dataset("grid").unwrap().read_raw::<i32>().unwrap(),
        all
    );
    drop(file);
    cleanup(&path);
}

/// The filtered-BT2 counterpart: compressed records decode back into the
/// index (address, stored size, filter mask), a reopened write patches
/// and adds tiles, and the whole grid reads back.
#[cfg(feature = "deflate")]
#[test]
fn reopened_filtered_btree_v2_dataset_takes_new_chunks() {
    let path = unique_tmp("bt2_write");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4usize, 4])
            .chunk(&[2, 2])
            .max_shape(&[None, None])
            .deflate(4)
            .create("tiles")
            .unwrap();
        // Top half only: tiles (0,0) and (0,1).
        let top: Vec<i32> = (0..8).collect();
        ds.write_slice(&[0, 0], &[2, 4], &top).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        let ds = file.dataset_writer("tiles").unwrap();
        // Bottom half is new tiles; patching (0,0) recompresses an
        // existing record to a new size and address.
        let bottom: Vec<i32> = (8..16).collect();
        ds.write_slice(&[2, 0], &[2, 4], &bottom).unwrap();
        ds.write_slice(&[0, 0], &[1, 1], &[99i32]).unwrap();
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let mut expect: Vec<i32> = (0..16).collect();
    expect[0] = 99;
    assert_eq!(
        file.dataset("tiles").unwrap().read_raw::<i32>().unwrap(),
        expect
    );
    drop(file);
    cleanup(&path);
}
