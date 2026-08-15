//! `H5File::superblock_version()` and `H5File::libver_bound()` — the raw
//! version byte straight after the signature (`oracle/canon.py`'s
//! `read_superblock`; libhdf5 has no public getter for it either, only
//! `H5Fget_info2`'s `super_version` field, which h5py does not bind) and the
//! lowest [`LibverBound`] consistent with it.
//!
//! Three superblock generations, not two: [`LibverBound::Earliest`] writes
//! version 0 (legacy, symbol-table root); a file with no bound named writes
//! version 2 (`superblock_version_for`'s floor for a non-legacy file); and
//! [`LibverBound::V110`] (or higher, up to `V200`) writes version 3. Every
//! case below pairs against a contrasting generation, so an accessor that
//! just returned a constant would fail every pair but one.

use std::path::PathBuf;

use rust_hdf5::{H5File, LibverBound};

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_superblock_version_accessor_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

#[test]
fn earliest_bound_writes_superblock_version_0() {
    let path = unique_tmp("v0");
    let file = H5File::options()
        .libver(LibverBound::Earliest)
        .create(&path)
        .unwrap();
    file.create_group("g").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.superblock_version().unwrap(), 0);
    assert_eq!(file.libver_bound().unwrap(), LibverBound::Earliest);
    std::fs::remove_file(&path).ok();
}

#[test]
fn no_bound_named_writes_superblock_version_2() {
    let path = unique_tmp("v2");
    let file = H5File::create(&path).unwrap();
    file.create_group("g").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.superblock_version().unwrap(), 2);
    assert_eq!(file.libver_bound().unwrap(), LibverBound::V18);
    std::fs::remove_file(&path).ok();
}

#[test]
fn v110_bound_writes_superblock_version_3() {
    let path = unique_tmp("v3");
    let file = H5File::options()
        .libver(LibverBound::V110)
        .create(&path)
        .unwrap();
    file.create_group("g").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.superblock_version().unwrap(), 3);
    assert_eq!(file.libver_bound().unwrap(), LibverBound::V110);
    std::fs::remove_file(&path).ok();
}

/// `libver_bound` is a view over the on-disk version, not the exact bound a
/// writer named: `V200` and `V110` both write superblock version 3, so a
/// version-3 file reads back at the lowest of the four bounds that produce
/// it, not the one that actually wrote it.
#[test]
fn libver_bound_reports_the_lowest_bound_sharing_a_version() {
    let path = unique_tmp("v3_from_v200");
    let file = H5File::options()
        .libver(LibverBound::V200)
        .create(&path)
        .unwrap();
    file.create_group("g").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.superblock_version().unwrap(), 3);
    assert_eq!(file.libver_bound().unwrap(), LibverBound::V110);
    std::fs::remove_file(&path).ok();
}

#[test]
fn superblock_version_errors_in_write_mode() {
    let path = unique_tmp("write_mode");
    let file = H5File::create(&path).unwrap();
    assert!(file.superblock_version().is_err());
    assert!(file.libver_bound().is_err());
    file.close().unwrap();
    std::fs::remove_file(&path).ok();
}
