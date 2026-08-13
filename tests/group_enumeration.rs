//! Integration tests for group enumeration based on actual link records.
//!
//! `H5Group::group()` and `H5Group::group_names()` must discover groups
//! from real link records, not from inferred dataset-path prefixes. A group
//! that contains only attributes, only a subgroup, or nothing at all has no
//! dataset beneath it on a direct path, yet must still be openable and
//! listable. This is common in NeXus files, where a group is created and
//! given an `NX_class` attribute before any leaf dataset exists.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// Per-test unique temp path. Avoids collisions when cargo runs tests in
/// parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_group_enum_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

/// Remove the temp file and its directory.
fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// (a) empty group, (b) subgroup-only group, (c) attribute-only group,
/// (d) normal group with a dataset — all must be openable and listed.
#[test]
fn group_enumeration_covers_groups_without_datasets() {
    let path = unique_tmp("mixed_groups");

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();

        // (a) an empty group: no datasets, no subgroups, no attributes.
        root.create_group("empty").unwrap();

        // (b) a group containing only a subgroup.
        let subgroup_only = root.create_group("subgroup_only").unwrap();
        subgroup_only.create_group("child").unwrap();

        // (c) a group with only an attribute and no dataset (NeXus style).
        let attr_only = root.create_group("attr_only").unwrap();
        attr_only.set_attr_string("NX_class", "NXdetector").unwrap();

        // (d) a normal group with a dataset.
        let with_data = root.create_group("with_data").unwrap();
        with_data
            .new_dataset::<f32>()
            .shape([3])
            .create("temperature")
            .unwrap();

        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let root = file.root_group();

        // group_names() must list all four immediate child groups.
        let mut names = root.group_names().unwrap();
        names.sort();
        assert_eq!(
            names,
            vec!["attr_only", "empty", "subgroup_only", "with_data"],
            "group_names() must list groups discovered from link records"
        );

        // group() must open each of them.
        let empty = root.group("empty").unwrap();
        assert_eq!(empty.name(), "/empty");
        assert!(empty.group_names().unwrap().is_empty());
        assert!(empty.dataset_names().unwrap().is_empty());

        let subgroup_only = root.group("subgroup_only").unwrap();
        assert_eq!(subgroup_only.name(), "/subgroup_only");
        assert_eq!(
            subgroup_only.group_names().unwrap(),
            vec!["child".to_string()]
        );
        // The nested subgroup must itself be openable.
        let child = subgroup_only.group("child").unwrap();
        assert_eq!(child.name(), "/subgroup_only/child");

        let attr_only = root.group("attr_only").unwrap();
        assert_eq!(attr_only.name(), "/attr_only");
        assert_eq!(
            attr_only.attr_string("NX_class").unwrap(),
            "NXdetector",
            "attribute-only group must keep its attribute"
        );

        let with_data = root.group("with_data").unwrap();
        assert_eq!(
            with_data.dataset_names().unwrap(),
            vec!["temperature".to_string()]
        );

        // A non-existent group must still be reported as NotFound.
        assert!(root.group("does_not_exist").is_err());
    }

    cleanup(&path);
}

/// Deeply nested empty groups must each be openable and listed at their
/// own level.
#[test]
fn group_enumeration_handles_deep_empty_nesting() {
    let path = unique_tmp("deep_empty");

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let a = root.create_group("a").unwrap();
        let b = a.create_group("b").unwrap();
        let _c = b.create_group("c").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let root = file.root_group();

        assert_eq!(root.group_names().unwrap(), vec!["a".to_string()]);

        let a = root.group("a").unwrap();
        assert_eq!(a.group_names().unwrap(), vec!["b".to_string()]);

        let b = a.group("b").unwrap();
        assert_eq!(b.group_names().unwrap(), vec!["c".to_string()]);

        let c = b.group("c").unwrap();
        assert_eq!(c.name(), "/a/b/c");
        assert!(c.group_names().unwrap().is_empty());
    }

    cleanup(&path);
}

/// `group_names()` on a parent must only list immediate children, not
/// grandchildren.
#[test]
fn group_names_lists_only_immediate_children() {
    let path = unique_tmp("immediate_children");

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let entry = root.create_group("entry").unwrap();
        let instrument = entry.create_group("instrument").unwrap();
        instrument.create_group("detector").unwrap();
        entry.create_group("sample").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let root = file.root_group();

        assert_eq!(root.group_names().unwrap(), vec!["entry".to_string()]);

        let entry = root.group("entry").unwrap();
        let mut entry_children = entry.group_names().unwrap();
        entry_children.sort();
        assert_eq!(
            entry_children,
            vec!["instrument".to_string(), "sample".to_string()],
            "must list only immediate children, not /entry/instrument/detector"
        );
    }

    cleanup(&path);
}

/// A bare `open_rw` + `close` must not change the group structure. The
/// reopen used to rebuild the group registry from dataset paths alone, so
/// a group with no dataset beneath it (empty, subgroup-only, or NeXus
/// attribute-only) vanished at close — and even surviving groups lost
/// their attributes, which the reopen had parsed and then discarded.
#[test]
fn reopen_session_keeps_groups_and_their_attributes() {
    let path = unique_tmp("reopen_keeps");

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        root.create_group("empty").unwrap();
        let subgroup_only = root.create_group("subgroup_only").unwrap();
        subgroup_only.create_group("child").unwrap();
        let attr_only = root.create_group("attr_only").unwrap();
        attr_only.set_attr_string("NX_class", "NXdetector").unwrap();
        let with_data = root.create_group("with_data").unwrap();
        with_data.set_attr_string("NX_class", "NXentry").unwrap();
        file.write_vlen_strings("with_data/notes", &["n0"]).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let root = file.root_group();
    let mut names = root.group_names().unwrap();
    names.sort();
    assert_eq!(names, ["attr_only", "empty", "subgroup_only", "with_data"]);
    assert_eq!(
        root.group("subgroup_only").unwrap().group_names().unwrap(),
        vec!["child".to_string()]
    );
    assert_eq!(
        root.group("attr_only")
            .unwrap()
            .attr_string("NX_class")
            .unwrap(),
        "NXdetector"
    );
    assert_eq!(
        root.group("with_data")
            .unwrap()
            .attr_string("NX_class")
            .unwrap(),
        "NXentry"
    );
    assert_eq!(
        file.dataset("with_data/notes")
            .unwrap()
            .read_vlen_strings()
            .unwrap(),
        vec!["n0"]
    );
    drop(file);
    cleanup(&path);
}

/// A reopen session must be able to put an attribute on a group that has
/// no dataset beneath it — the group has to be in the writer's registry
/// for `set_attr_string` to find it.
#[test]
fn reopen_session_can_attribute_a_datasetless_group() {
    let path = unique_tmp("reopen_attr_empty");

    {
        let file = H5File::create(&path).unwrap();
        file.root_group().create_group("empty").unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        file.root_group()
            .group("empty")
            .unwrap()
            .set_attr_string("NX_class", "NXcollection")
            .unwrap();
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group()
            .group("empty")
            .unwrap()
            .attr_string("NX_class")
            .unwrap(),
        "NXcollection"
    );
    drop(file);
    cleanup(&path);
}
