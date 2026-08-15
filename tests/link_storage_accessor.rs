//! `H5Group::link_storage()` — the equivalent of libhdf5's
//! `H5Gget_info(gid).storage_type` (h5py has no binding for it, so the
//! oracle reconstructs it from `h5o.get_info`; see `oracle/canon.py`'s
//! `link_storage_str`) — on `H5Group` and the root group.
//!
//! Three storage kinds, not two: a pre-1.8 group keeps its links in a
//! **symbol table** (a v1 B-tree plus local heap) regardless of how many it
//! holds, since `H5G_obj_insert`'s compact/dense phase change belongs to the
//! newer link-message format only. A modern group's links stay **compact**
//! header messages up to `MAX_COMPACT_LINKS` (8), then move to a **dense**
//! fractal heap plus name index past it. Every case below pairs against a
//! contrasting shape, so an accessor that just returned a constant would
//! fail every pair but one.

use std::path::PathBuf;

use rust_hdf5::{H5File, LibverBound, LinkStorage};

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_link_storage_accessor_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

const COMPACT_COUNT: usize = 2;
const DENSE_COUNT: usize = 9;

#[test]
fn group_link_storage_is_compact_under_the_phase_change_threshold() {
    let path = unique_tmp("group_compact");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..COMPACT_COUNT {
        grp.create_group(&format!("c{i}")).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.link_storage().unwrap(), LinkStorage::Compact);
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_link_storage_is_dense_past_the_phase_change_threshold() {
    let path = unique_tmp("group_dense");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..DENSE_COUNT {
        grp.create_group(&format!("c{i}")).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.link_storage().unwrap(), LinkStorage::Dense);
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_link_storage_is_compact_under_the_phase_change_threshold() {
    let path = unique_tmp("root_compact");
    let file = H5File::create(&path).unwrap();
    for i in 0..COMPACT_COUNT {
        file.create_group(&format!("c{i}")).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().link_storage().unwrap(),
        LinkStorage::Compact
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_link_storage_is_dense_past_the_phase_change_threshold() {
    let path = unique_tmp("root_dense");
    let file = H5File::create(&path).unwrap();
    for i in 0..DENSE_COUNT {
        file.create_group(&format!("c{i}")).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().link_storage().unwrap(),
        LinkStorage::Dense
    );
    std::fs::remove_file(&path).ok();
}

/// A file created at the earliest libver bound keeps every group's links in
/// a symbol table no matter how many it holds — the compact/dense phase
/// change belongs to the link-message format a legacy file never adopts.
#[test]
fn group_link_storage_is_symbol_table_at_the_earliest_libver_bound() {
    let path = unique_tmp("group_symtab");
    let file = H5File::options()
        .libver(LibverBound::Earliest)
        .create(&path)
        .unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..DENSE_COUNT {
        grp.create_group(&format!("c{i}")).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.link_storage().unwrap(), LinkStorage::SymbolTable);
    let root = file.root_group();
    assert_eq!(root.link_storage().unwrap(), LinkStorage::SymbolTable);
    std::fs::remove_file(&path).ok();
}
