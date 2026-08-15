//! `H5Group::attr_creation_order()` — the equivalent of h5py's
//! `gid.get_create_plist().get_attr_creation_order()` — reads the object
//! header's own creation-order flag bits, independent of whether the group
//! currently holds any attributes at all.
//!
//! Every case pairs a tracked group against an untracked one from the same
//! file, so an accessor that just returned a constant would fail half of
//! every pair.

use std::path::PathBuf;

use rust_hdf5::{CreationOrder, H5File};

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_attr_creation_order_accessor_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

#[test]
fn group_attr_creation_order_is_untracked_by_default() {
    let path = unique_tmp("group_untracked");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    grp.set_attr_numeric("a", &1i32).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.attr_creation_order().unwrap(), CreationOrder::Untracked);
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_attr_creation_order_is_indexed_when_track_order_is_set() {
    let path = unique_tmp("group_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    grp.set_attr_numeric("a", &1i32).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.attr_creation_order().unwrap(), CreationOrder::Indexed);
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_attr_creation_order_is_untracked_by_default() {
    let path = unique_tmp("root_untracked");
    let file = H5File::create(&path).unwrap();
    file.set_attr_numeric("a", &1i32).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().attr_creation_order().unwrap(),
        CreationOrder::Untracked
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_attr_creation_order_is_indexed_when_track_order_is_set() {
    let path = unique_tmp("root_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    file.set_attr_numeric("a", &1i32).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().attr_creation_order().unwrap(),
        CreationOrder::Indexed
    );
    std::fs::remove_file(&path).ok();
}

/// Regression case for a bug caught while wiring this accessor: the walk
/// only recorded a group's `ObjectAttributes` (the sole carrier of the
/// header's creation-order flags) when the group held at least one
/// attribute. A group that tracks creation order but currently has zero
/// attributes would have silently reported back `Untracked` — the shared
/// collector's default — instead of the header's real answer.
#[test]
fn empty_tracked_group_still_reports_its_creation_order() {
    let path = unique_tmp("group_tracked_empty");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    file.root_group().create_group("empty").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("empty").unwrap();
    assert_eq!(grp.attr_creation_order().unwrap(), CreationOrder::Indexed);
    std::fs::remove_file(&path).ok();
}
