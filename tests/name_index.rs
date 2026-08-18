//! Name-uniqueness across the paths that change a name without creating one.
//!
//! Creation refuses a name the file already holds, and the writer answers
//! that question from an index rather than by walking every registry. The
//! index is a filter: it may keep a name whose object is gone, and the
//! predicates behind it sort that out. What it must never do is miss a name
//! that is taken — so the cases here are the ones where a name arrives or
//! leaves *without* a create: a delete, the two promotions that re-home an
//! object under a surviving hard link, and a reopen, which fills the
//! registries before anything has registered a thing.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// The message of a creation that had to be refused. `unwrap_err` is out of
/// reach here: the handles these calls return on success are not `Debug`.
fn refusal<T>(result: rust_hdf5::Result<T>) -> String {
    match result {
        Ok(_) => panic!("creation should have been refused"),
        Err(e) => format!("{e}"),
    }
}

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_name_index_{}_{}_{}",
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

/// The owner path: a name a create registered is refused to the next create,
/// and the refusal names what holds it.
#[test]
fn a_created_name_is_refused_to_the_next_creator() {
    let path = unique_tmp("created");
    let file = H5File::create(&path).unwrap();

    file.new_dataset::<f32>().shape([4]).create("ds").unwrap();
    let err = refusal(file.new_dataset::<f32>().shape([4]).create("ds"));
    assert!(
        err.contains("a dataset named 'ds' already exists"),
        "unexpected error: {err}"
    );

    file.create_group("grp").unwrap();
    let err = refusal(file.create_group("grp"));
    assert!(
        err.contains("a group named 'grp' already exists"),
        "unexpected error: {err}"
    );

    file.create_soft_link("soft", "/ds").unwrap();
    let err = refusal(file.new_dataset::<f32>().shape([4]).create("soft"));
    assert!(
        err.contains("a link named 'soft' already exists"),
        "unexpected error: {err}"
    );

    drop(file);
    cleanup(&path);
}

/// Deleting a dataset gives its name back. The index keeps the entry it made
/// at creation, so this is the case where a stale hit must not be mistaken
/// for a live one.
#[test]
fn a_name_a_delete_freed_can_be_created_again() {
    let path = unique_tmp("freed");
    let file = H5File::create(&path).unwrap();

    let ds = file.new_dataset::<f32>().shape([4]).create("ds").unwrap();
    ds.write_raw(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    drop(ds);
    file.delete_dataset("ds").unwrap();

    let ds = file.new_dataset::<f32>().shape([2]).create("ds").unwrap();
    ds.write_raw(&[7.0f32, 8.0]).unwrap();
    drop(ds);
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let got = file.dataset("ds").unwrap().read_raw::<f32>().unwrap();
    assert_eq!(got, vec![7.0f32, 8.0]);
    drop(file);
    cleanup(&path);
}

/// Deleting the tree name of a dataset a hard link still names re-homes the
/// dataset under the link. Nothing was created, yet the link's path now
/// holds a dataset and the old path holds nothing.
#[test]
fn a_dataset_promoted_onto_its_link_holds_the_link_s_name() {
    let path = unique_tmp("promote_ds");
    let file = H5File::create(&path).unwrap();
    let root = file.root_group();

    let src = root.create_group("src").unwrap();
    let ds = src.new_dataset::<f32>().shape([4]).create("ds").unwrap();
    ds.write_raw(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    drop(ds);
    let alias = root.create_group("alias").unwrap();
    alias.link("ds", "/src/ds").unwrap();

    // The link survives the delete and becomes the dataset's only name.
    file.delete_dataset("src/ds").unwrap();

    let err = refusal(file.new_dataset::<f32>().shape([4]).create("alias/ds"));
    assert!(
        err.contains("a dataset named 'alias/ds' already exists"),
        "unexpected error: {err}"
    );

    // ...and the name it left is free.
    let ds = file
        .new_dataset::<f32>()
        .shape([2])
        .create("src/ds")
        .unwrap();
    ds.write_raw(&[9.0f32, 9.0]).unwrap();
    drop(ds);
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("alias/ds").unwrap().read_raw::<f32>().unwrap(),
        vec![1.0f32, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        file.dataset("src/ds").unwrap().read_raw::<f32>().unwrap(),
        vec![9.0f32, 9.0]
    );
    drop(file);
    cleanup(&path);
}

/// The group counterpart: promoting a group onto its link renames the group
/// and every live object beneath it, so a whole subtree of names moves at
/// once and the names under the new prefix must be taken.
#[test]
fn a_group_promoted_onto_its_link_holds_its_subtree_s_names() {
    let path = unique_tmp("promote_grp");
    let file = H5File::create(&path).unwrap();
    let root = file.root_group();

    let src = root.create_group("src").unwrap();
    let sub = src.create_group("sub").unwrap();
    let ds = sub.new_dataset::<f32>().shape([3]).create("x").unwrap();
    ds.write_raw(&[1.0f32, 2.0, 3.0]).unwrap();
    drop(ds);
    let alias = root.create_group("alias").unwrap();
    alias.link("moved", "/src").unwrap();

    // A link naming the group itself makes this a pure rename.
    file.delete_group("src").unwrap();

    let err = refusal(file.create_group("alias/moved/sub"));
    assert!(
        err.contains("a group named 'alias/moved/sub' already exists"),
        "unexpected error: {err}"
    );
    let err = refusal(
        file.new_dataset::<f32>()
            .shape([3])
            .create("alias/moved/sub/x"),
    );
    assert!(
        err.contains("a dataset named 'alias/moved/sub/x' already exists"),
        "unexpected error: {err}"
    );

    file.close().unwrap();
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("alias/moved/sub/x")
            .unwrap()
            .read_raw::<f32>()
            .unwrap(),
        vec![1.0f32, 2.0, 3.0]
    );
    drop(file);
    cleanup(&path);
}

/// A reopened file's names were never registered by a create — they came
/// off the disk into the registries — and creating over one must still be
/// refused.
#[test]
fn names_a_reopened_file_already_holds_are_refused() {
    let path = unique_tmp("reopen");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<f32>().shape([4]).create("ds").unwrap();
        ds.write_raw(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        drop(ds);
        file.create_group("grp").unwrap();
        file.close().unwrap();
    }

    let file = H5File::open_rw(&path).unwrap();
    let err = refusal(file.new_dataset::<f32>().shape([4]).create("ds"));
    assert!(err.contains("already exists"), "unexpected error: {err}");
    let err = refusal(file.create_group("grp"));
    assert!(err.contains("already exists"), "unexpected error: {err}");
    // A name the reopened file does not hold is still free.
    file.new_dataset::<f32>()
        .shape([2])
        .create("later")
        .unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("ds").unwrap().read_raw::<f32>().unwrap(),
        vec![1.0f32, 2.0, 3.0, 4.0]
    );
    drop(file);
    cleanup(&path);
}
