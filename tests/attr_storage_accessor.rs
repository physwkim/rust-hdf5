//! `attr_storage()` — h5py's `h5o.get_info(...).meta_size.attr.index_size`
//! being nonzero — on `H5Group`, `H5Dataset`, and the root group.
//!
//! `H5O__attr_create`'s phase change fires once an object's attribute count
//! passes `MAX_COMPACT_ATTRS` (8): a set of exactly 8 stays compact, and a
//! 9th attribute pushes it to dense fractal-heap storage. Every case below
//! pairs a small set against a set that has crossed that line, so an
//! accessor that just returned a constant would fail half of every pair.

use std::path::PathBuf;

use rust_hdf5::{AttributeStorage, H5File};

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_attr_storage_accessor_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

const COMPACT_COUNT: usize = 2;
const DENSE_COUNT: usize = 9;

#[test]
fn group_attr_storage_is_compact_under_the_phase_change_threshold() {
    let path = unique_tmp("group_compact");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..COMPACT_COUNT {
        grp.set_attr_numeric(&format!("a{i}"), &(i as i32)).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.attr_storage().unwrap(), AttributeStorage::Compact);
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_attr_storage_is_dense_past_the_phase_change_threshold() {
    let path = unique_tmp("group_dense");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..DENSE_COUNT {
        grp.set_attr_numeric(&format!("a{i}"), &(i as i32)).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.attr_storage().unwrap(), AttributeStorage::Dense);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dataset_attr_storage_is_compact_under_the_phase_change_threshold() {
    let path = unique_tmp("dataset_compact");
    let file = H5File::create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
    for i in 0..COMPACT_COUNT {
        ds.new_attr::<i32>()
            .shape(())
            .create(&format!("a{i}"))
            .unwrap()
            .write_numeric(&(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.attr_storage().unwrap(), AttributeStorage::Compact);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dataset_attr_storage_is_dense_past_the_phase_change_threshold() {
    let path = unique_tmp("dataset_dense");
    let file = H5File::create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
    for i in 0..DENSE_COUNT {
        ds.new_attr::<i32>()
            .shape(())
            .create(&format!("a{i}"))
            .unwrap()
            .write_numeric(&(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.attr_storage().unwrap(), AttributeStorage::Dense);
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_attr_storage_is_dense_past_the_phase_change_threshold() {
    let path = unique_tmp("root_dense");
    let file = H5File::create(&path).unwrap();
    for i in 0..DENSE_COUNT {
        file.set_attr_numeric(&format!("a{i}"), &(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().attr_storage().unwrap(),
        AttributeStorage::Dense
    );
    std::fs::remove_file(&path).ok();
}
