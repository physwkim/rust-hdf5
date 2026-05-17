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

/// Creating a dataset whose name a hard link already occupies is rejected
/// (the reverse order of `hard_link_rejects_duplicate_name`): otherwise the
/// parent group would carry two link records with the same name.
#[test]
fn dataset_rejects_name_taken_by_hard_link() {
    let path = unique_tmp("hl_reverse");
    let file = H5File::create(&path).unwrap();
    let root = file.root_group();
    let inst = root.create_group("instrument").unwrap();
    inst.new_dataset::<f32>()
        .shape([4])
        .create("detector")
        .unwrap();

    let data = root.create_group("data").unwrap();
    data.link("detector", "/instrument/detector").unwrap();

    // /data/detector is already a hard link; a dataset there must fail.
    let ds_result = data.new_dataset::<f32>().shape([4]).create("detector");
    let err = ds_result
        .err()
        .expect("dataset creation should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("already exists"), "unexpected error: {msg}");

    // ...and so must a group of the same name.
    let grp_result = data.create_group("detector");
    let err = grp_result.err().expect("group creation should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("already exists"), "unexpected error: {msg}");

    drop(file);
    cleanup(&path);
}

/// A hard link can be created through the public SWMR writer API. Created
/// before `start_swmr`, it is committed with the streaming layout and
/// resolves to the target's data after close.
#[test]
fn swmr_writer_creates_hard_link() {
    use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};

    let path = unique_tmp("hl_swmr");
    {
        let mut w = SwmrFileWriter::create(&path).unwrap();
        let ds = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        // Layout alias created before start_swmr -> visible for the whole run.
        w.create_hard_link("/", "alias", "frames").unwrap();
        w.start_swmr().unwrap();
        w.append_frame(ds, &[1u8, 2, 3, 4]).unwrap();
        w.close().unwrap();
    }

    let mut r = SwmrFileReader::open(&path).unwrap();
    let names = r.dataset_names();
    assert!(
        names.iter().any(|n| n == "alias"),
        "hard link 'alias' missing: {names:?}"
    );
    assert_eq!(r.read_dataset_raw("frames").unwrap(), vec![1u8, 2, 3, 4]);
    assert_eq!(r.read_dataset_raw("alias").unwrap(), vec![1u8, 2, 3, 4]);

    cleanup(&path);
}

/// The public SWMR writer API can build a nested NeXus-style layout: groups
/// tagged with `NX_class` attributes plus a hard link aliasing a streaming
/// dataset into that layout. All structure is created before `start_swmr`.
#[test]
fn swmr_writer_builds_nexus_layout() {
    use rust_hdf5::swmr::SwmrFileWriter;

    let path = unique_tmp("swmr_nexus");
    {
        let mut w = SwmrFileWriter::create(&path).unwrap();
        let ds = w
            .create_streaming_dataset::<u16>("frames", &[2, 2])
            .unwrap();

        // NeXus group tree: /entry (NXentry) -> /entry/data (NXdata).
        w.create_group("/", "entry").unwrap();
        w.create_group("/entry", "data").unwrap();
        w.set_group_attr_string("/entry", "NX_class", "NXentry")
            .unwrap();
        w.set_group_attr_string("/entry/data", "NX_class", "NXdata")
            .unwrap();
        // Alias the streaming dataset at the NeXus canonical location.
        w.create_hard_link("/entry/data", "data", "frames").unwrap();

        w.start_swmr().unwrap();
        // One frame of 4 u16 values, little-endian: 1, 2, 3, 4.
        w.append_frame(ds, &[1u8, 0, 2, 0, 3, 0, 4, 0]).unwrap();
        w.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let root = file.root_group();

    let entry = root.group("entry").unwrap();
    assert_eq!(entry.attr_string("NX_class").unwrap(), "NXentry");

    let data = entry.group("data").unwrap();
    assert_eq!(data.attr_string("NX_class").unwrap(), "NXdata");

    // The hard link resolves to the streaming dataset's data.
    let aliased = file
        .dataset("entry/data/data")
        .unwrap()
        .read_raw::<u16>()
        .unwrap();
    assert_eq!(aliased, vec![1u16, 2, 3, 4]);

    cleanup(&path);
}

/// A target path given with a trailing slash still resolves.
#[test]
fn hard_link_tolerates_trailing_slash() {
    let path = unique_tmp("hl_trailing");
    let data: Vec<i32> = vec![3, 1, 4, 1, 5];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([5])
            .create("counts")
            .unwrap();
        ds.write_raw(&data).unwrap();

        // Leading and trailing slashes both tolerated.
        root.link("alias", "/instrument/counts/").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let aliased = file.dataset("alias").unwrap().read_raw::<i32>().unwrap();
        assert_eq!(aliased, data);
    }

    cleanup(&path);
}
