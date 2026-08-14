//! Shared object header messages (SOHM), write side.
//!
//! Reading a file whose messages live in the SOHM fractal heap works (see
//! `tests/sohm.rs`), but writing that indirection does not exist here: a
//! finalize would rewrite the headers it touches with the shared messages
//! dropped, while the master table still claims they are referenced, and
//! libhdf5 then reads the result as a file missing its objects. The append
//! path therefore refuses such a file, before it writes anything.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_sohm_guard_{}_{}_{}",
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

/// The message of the error `open_rw` fails with. `H5File` is not `Debug`, so
/// `expect_err` is not available.
fn open_rw_error(path: &PathBuf) -> String {
    match H5File::open_rw(path) {
        Ok(_) => panic!("a file with shared messages must not open for writing"),
        Err(e) => e.to_string(),
    }
}

/// Copy a fixture, refuse to open it for appending, and require the bytes to
/// be exactly what they were.
fn refuses_and_leaves_untouched(name: &str) {
    let bytes = std::fs::read(fixture(name)).unwrap();
    let path = unique_tmp(name.trim_end_matches(".h5"));
    std::fs::write(&path, &bytes).unwrap();

    let text = open_rw_error(&path);
    assert!(
        text.contains("shared object header message (SOHM)"),
        "{text}"
    );
    assert!(
        text.contains("cannot open this file for appending"),
        "{text}"
    );

    assert_eq!(std::fs::read(&path).unwrap(), bytes, "{name} was modified");
    cleanup(&path);
}

#[test]
fn a_file_with_a_list_index_is_refused_for_appending() {
    refuses_and_leaves_untouched("sohm_list.h5");
}

#[test]
fn a_file_with_a_btree_index_is_refused_for_appending() {
    refuses_and_leaves_untouched("sohm_btree.h5");
}

/// The options builder is the other public way in, and it must refuse too.
#[test]
fn the_options_builder_refuses_the_same_file() {
    let bytes = std::fs::read(fixture("sohm_list.h5")).unwrap();
    let path = unique_tmp("options");
    std::fs::write(&path, &bytes).unwrap();

    let text = match H5File::options().open_rw(&path) {
        Ok(_) => panic!("a file with shared messages must not open for writing"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("shared object header message (SOHM)"),
        "{text}"
    );

    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    cleanup(&path);
}

/// The guard reads the superblock extension, which most files do not have and
/// none of this crate's own files fill with a shared-message table: appending
/// to them keeps working.
#[test]
fn a_file_without_shared_messages_still_opens_for_appending() {
    let path = unique_tmp("plain");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("a")
            .unwrap()
            .write_raw(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("b")
            .unwrap()
            .write_raw(&[4i32, 5, 6])
            .unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, vec!["a", "b"]);
    drop(file);
    cleanup(&path);
}
