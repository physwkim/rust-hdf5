//! Integration tests for hard-link creation (`H5Group::link`).
//!
//! A hard link gives an existing object a second name without copying its
//! data — the NeXus-style way to expose a dataset at `/entry/data/data`
//! while it physically lives elsewhere. Both names must resolve to
//! byte-identical data, and the reader must enumerate the aliased path.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// Per-test unique temp path so parallel cargo runs cannot collide.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_hard_links_{}_{}_{}",
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

/// A hard link in a subgroup must resolve to the same data as the target,
/// and the reader must enumerate the aliased path.
#[test]
fn hard_link_to_dataset_shares_data() {
    let path = unique_tmp("hl_dataset");
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();

        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<f32>()
            .shape([12])
            .create("detector")
            .unwrap();
        ds.write_raw(&data).unwrap();

        // NeXus-style alias: /data/detector -> /instrument/detector.
        let data_grp = root.create_group("data").unwrap();
        data_grp.link("detector", "/instrument/detector").unwrap();

        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();

        let original = file
            .dataset("instrument/detector")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();
        let aliased = file
            .dataset("data/detector")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();

        assert_eq!(original, data, "target reads back the written data");
        assert_eq!(aliased, data, "hard link resolves to the same data");
    }

    cleanup(&path);
}

/// A hard link can live in the root group and point at a nested dataset.
#[test]
fn hard_link_in_root_group() {
    let path = unique_tmp("hl_root");
    let data: Vec<i32> = vec![7, 8, 9];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([3])
            .create("counts")
            .unwrap();
        ds.write_raw(&data).unwrap();

        root.link("counts_alias", "instrument/counts").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let aliased = file
            .dataset("counts_alias")
            .unwrap()
            .read_raw::<i32>()
            .unwrap();
        assert_eq!(aliased, data);
    }

    cleanup(&path);
}

/// Linking to a non-existent target is rejected.
#[test]
fn hard_link_rejects_unknown_target() {
    let path = unique_tmp("hl_unknown");
    let file = H5File::create(&path).unwrap();
    let err = file
        .root_group()
        .link("alias", "/does/not/exist")
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("not found"), "unexpected error: {msg}");
    drop(file);
    cleanup(&path);
}

/// A link name that already exists in the parent group is rejected.
#[test]
fn hard_link_rejects_duplicate_name() {
    let path = unique_tmp("hl_dup");
    let file = H5File::create(&path).unwrap();
    let root = file.root_group();
    let inst = root.create_group("instrument").unwrap();
    inst.new_dataset::<f32>()
        .shape([4])
        .create("detector")
        .unwrap();

    // "detector" already names a dataset in /instrument.
    let err = inst.link("detector", "/instrument/detector").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("already exists"), "unexpected error: {msg}");

    drop(file);
    cleanup(&path);
}
