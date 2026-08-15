//! `H5Group::link_creation_order()` — the equivalent of h5py's
//! `gid.get_create_plist().get_link_creation_order()` — reads the group's
//! own `Link Info` message flag bits, independent of
//! [`attr_creation_order`](rust_hdf5::H5Group::attr_creation_order): the two
//! subsystems track creation order separately, and independent of a symbol
//! table group, which predates the whole feature and is always untracked.
//!
//! Every case pairs a tracked group against an untracked one, so an accessor
//! that just returned a constant would fail half of every pair.

use std::path::PathBuf;

use rust_hdf5::{CreationOrder, H5File, LibverBound};

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_link_creation_order_accessor_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

#[test]
fn group_link_creation_order_is_untracked_by_default() {
    let path = unique_tmp("group_untracked");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    grp.create_group("child").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.link_creation_order().unwrap(), CreationOrder::Untracked);
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_link_creation_order_is_indexed_when_track_order_is_set() {
    let path = unique_tmp("group_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    grp.create_group("child").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.link_creation_order().unwrap(), CreationOrder::Indexed);
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_link_creation_order_is_untracked_by_default() {
    let path = unique_tmp("root_untracked");
    let file = H5File::create(&path).unwrap();
    file.create_group("child").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().link_creation_order().unwrap(),
        CreationOrder::Untracked
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_link_creation_order_is_indexed_when_track_order_is_set() {
    let path = unique_tmp("root_tracked");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    file.create_group("child").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().link_creation_order().unwrap(),
        CreationOrder::Indexed
    );
    std::fs::remove_file(&path).ok();
}

/// A group past HDF5 1.8's symbol-table format predates creation-order
/// tracking entirely: `H5Gcreate` with `H5P_CRT_ORDER_TRACKED` always
/// produces the "new" link-message format, so a symbol-table group can never
/// carry the flag. The contrasting shape for this accessor is not another
/// tracked/untracked pair but a storage kind that has no tracked case at all.
#[test]
fn symbol_table_group_link_creation_order_is_always_untracked() {
    let path = unique_tmp("group_symtab");
    let file = H5File::options()
        .libver(LibverBound::Earliest)
        .create(&path)
        .unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    grp.create_group("child").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.link_creation_order().unwrap(), CreationOrder::Untracked);
    let root = file.root_group();
    assert_eq!(
        root.link_creation_order().unwrap(),
        CreationOrder::Untracked
    );
    std::fs::remove_file(&path).ok();
}

/// Regression case for the same shape of bug `attr_creation_order` caught: a
/// group with zero children still has its own header to answer from, so a
/// tracked-but-empty group must not fall back to the collector's untracked
/// default.
#[test]
fn empty_tracked_group_still_reports_its_link_creation_order() {
    let path = unique_tmp("group_tracked_empty");
    let file = H5File::options().track_order(true).create(&path).unwrap();
    file.root_group().create_group("empty").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("empty").unwrap();
    assert_eq!(grp.link_creation_order().unwrap(), CreationOrder::Indexed);
    std::fs::remove_file(&path).ok();
}
