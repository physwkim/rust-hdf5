//! A selection the extent does not admit is refused at the API boundary,
//! whatever its shape: a start near `usize::MAX`, whose sum with the count
//! wraps back inside the extent, must not turn into an offset that reads
//! (or writes) unrelated bytes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_read_bounds_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("test.h5")
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// A 4x6 u8 dataset holding 0..24 in row-major order.
fn write_grid(path: &PathBuf) {
    let file = H5File::create(path).unwrap();
    let ds = file
        .new_dataset::<u8>()
        .shape([4usize, 6])
        .create("g")
        .unwrap();
    ds.write_raw_bytes(&(0..24u8).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

fn is_refused<T>(r: rust_hdf5::Result<T>) -> bool {
    match r {
        Ok(_) => false,
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("out of bounds"), "unexpected error: {msg}");
            true
        }
    }
}

/// Every reader entry point that takes a start refuses a start whose edge
/// wraps: `read_slice`, `read_slice_into` and `read_points`.
#[test]
fn a_wrapping_start_is_refused_on_every_read_path() {
    let path = unique_tmp("read");
    write_grid(&path);

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("g").unwrap();
    // The in-bounds neighbours of each refused case still read.
    assert_eq!(ds.read_slice::<u8>(&[3, 5], &[1, 1]).unwrap(), vec![23]);
    assert_eq!(ds.read_points::<u8>(&[vec![3, 5]]).unwrap(), vec![23]);

    assert!(is_refused(ds.read_slice::<u8>(&[usize::MAX, 0], &[1, 1])));
    assert!(is_refused(ds.read_slice::<u8>(&[0, usize::MAX], &[1, 1])));
    assert!(is_refused(ds.read_slice::<u8>(&[1, 0], &[usize::MAX, 1])));
    assert!(is_refused(ds.read_slice::<u8>(&[3, 5], &[2, 1])));

    let mut one = [0u8; 1];
    assert!(is_refused(ds.read_slice_into::<u8>(
        &mut one,
        &[usize::MAX, 0],
        &[1, 1]
    )));
    assert!(is_refused(ds.read_slice_into::<u8>(
        &mut one,
        &[0, 6],
        &[1, 1]
    )));

    assert!(is_refused(ds.read_points::<u8>(&[vec![usize::MAX, 0]])));
    assert!(is_refused(ds.read_points::<u8>(&[vec![0, 6]])));
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}

/// An oversized count is reported as a selection problem before its byte
/// size is computed, not as a failed allocation.
#[test]
fn an_oversized_count_is_refused_before_allocation() {
    let path = unique_tmp("count");
    write_grid(&path);

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("g").unwrap();
    assert!(is_refused(ds.read_slice::<u8>(&[0, 0], &[usize::MAX, 6])));
    assert!(is_refused(
        ds.read_slice::<u8>(&[0, 0], &[1 << 40, 1 << 20])
    ));
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}

/// The writer's `write_slice` shares the rule, so a wrapping start cannot
/// land bytes over a neighbour.
#[test]
fn a_wrapping_start_is_refused_on_write() {
    let path = unique_tmp("write");
    let file = H5File::create(&path).unwrap();
    let ds = file
        .new_dataset::<u8>()
        .shape([4usize, 6])
        .create("g")
        .unwrap();
    ds.write_raw_bytes(&(0..24u8).collect::<Vec<_>>()).unwrap();
    assert!(is_refused(ds.write_slice::<u8>(
        &[usize::MAX, 0],
        &[1, 1],
        &[99]
    )));
    assert!(is_refused(ds.write_slice::<u8>(
        &[3, 5],
        &[1, 2],
        &[99, 99]
    )));
    assert!(ds
        .write_slice::<u8>(&[0, 0], &[usize::MAX, usize::MAX], &[99])
        .is_err());
    ds.write_slice::<u8>(&[3, 5], &[1, 1], &[99]).unwrap();
    drop(ds);
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let mut want: Vec<u8> = (0..24).collect();
    want[23] = 99;
    assert_eq!(
        file.dataset("g")
            .unwrap()
            .read_slice::<u8>(&[0, 0], &[4, 6])
            .unwrap(),
        want
    );

    cleanup(&path);
}
