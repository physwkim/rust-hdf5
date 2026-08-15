//! `attr_names()`'s listing order matches h5py's default attribute
//! iteration (`H5Aiterate2` with no explicit index/order requested):
//! creation order when the object tracks it, name order otherwise — never
//! the physical order the attribute messages happen to sit in the header.
//!
//! Every case here writes its attributes in an order that is neither
//! alphabetical nor already ascending, so a listing that merely preserved
//! insertion order — the defect this rule replaces — would show up as a
//! mismatch against both expectations at once.

use std::path::PathBuf;

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_attr_order_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

/// Attributes created in this order on any object below: neither
/// alphabetical nor sorted any other obvious way.
const CREATION_ORDER: [&str; 4] = ["zeta", "alpha", "delta", "beta"];
const NAME_ORDER: [&str; 4] = ["alpha", "beta", "delta", "zeta"];

#[test]
fn dataset_attrs_list_in_name_order_when_untracked() {
    let path = unique_tmp("dataset_untracked");
    let file = H5File::create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
    for (i, name) in CREATION_ORDER.iter().enumerate() {
        ds.new_attr::<i32>()
            .shape(())
            .create(name)
            .unwrap()
            .write_numeric(&(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let names = file.dataset("data").unwrap().attr_names().unwrap();
    assert_eq!(names, NAME_ORDER);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dataset_attrs_list_in_creation_order_when_tracked() {
    let path = unique_tmp("dataset_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
    for (i, name) in CREATION_ORDER.iter().enumerate() {
        ds.new_attr::<i32>()
            .shape(())
            .create(name)
            .unwrap()
            .write_numeric(&(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let names = file.dataset("data").unwrap().attr_names().unwrap();
    assert_eq!(names, CREATION_ORDER);
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_attrs_list_in_creation_order_when_tracked() {
    let path = unique_tmp("group_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for (i, name) in CREATION_ORDER.iter().enumerate() {
        grp.set_attr_numeric(name, &(i as i32)).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let names = file.root_group().group("g").unwrap().attr_names().unwrap();
    assert_eq!(names, CREATION_ORDER);
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_attrs_list_in_creation_order_when_tracked() {
    let path = unique_tmp("root_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    for (i, name) in CREATION_ORDER.iter().enumerate() {
        file.set_attr_numeric(name, &(i as i32)).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let names = file.attr_names().unwrap();
    assert_eq!(names, CREATION_ORDER);
    std::fs::remove_file(&path).ok();
}
