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

use rust_hdf5::format::creation_order::CreationOrder;
use rust_hdf5::format::messages::{MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::sohm::type_flag;
use rust_hdf5::format::superblock::SuperblockV2V3;
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

/// `track_order` is not the only way an object comes to track creation
/// order. A file that shares attribute bodies must record a creation index on
/// every header (`H5O_SHMESG_ATTR_FLAG` forces `store_msg_crt_idx`), because
/// a shared attribute's order cannot be recovered from a body it no longer
/// owns. So a caller who asked for shared messages and nothing else still
/// gets creation-order headers — and this listing, which reads the flag off
/// the header rather than off the options, must follow them there.
#[test]
fn shared_attribute_bodies_put_the_listing_in_creation_order() {
    let path = unique_tmp("sohm_floor");
    let types = type_flag(MSG_ATTRIBUTE).unwrap()
        | type_flag(MSG_DATATYPE).unwrap()
        | type_flag(MSG_DATASPACE).unwrap();
    let file = H5File::options()
        .shared_messages(&[(types, 0)], 50, 40)
        .create(&path)
        .unwrap();
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

    // The floor really did reach this header — otherwise the assertion below
    // would be the untracked name-order case passing under another name.
    let bytes = std::fs::read(&path).unwrap();
    let sb = SuperblockV2V3::decode(&bytes).unwrap();
    let at = (sb.base_address + sb.root_group_object_header_address) as usize;
    let (root, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    assert_eq!(root.attribute_creation_order(), CreationOrder::Tracked);

    let file = H5File::open(&path).unwrap();
    let names = file.dataset("data").unwrap().attr_names().unwrap();
    assert_eq!(names, CREATION_ORDER);
    std::fs::remove_file(&path).ok();
}
