//! Files whose superblock is preceded by a userblock.
//!
//! The superblock does not have to start at offset 0: with a userblock it
//! starts at a power-of-two offset of at least 512, and `H5FD_locate_signature`
//! finds it by probing 0, 512, 1024, ... . That offset is the file's base
//! address — every address in the file is measured from it — so a reader that
//! assumes offset 0 does not merely mis-read the userblock, it cannot open the
//! file at all.
//!
//! `userblock.h5` (from `tests/fixtures/gen_userblock.py`) has a 512-byte
//! userblock holding a shell shebang, one dataset and one root attribute.

use std::path::PathBuf;

use rust_hdf5::H5File;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn a_file_behind_a_userblock_opens_and_reads() {
    let file = H5File::open(fixture("userblock.h5")).unwrap();

    assert_eq!(file.userblock_size(), 512);
    assert_eq!(file.dataset_names(), vec!["data"]);

    let data = file.dataset("data").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let note = file.attr_string("note").unwrap();
    assert_eq!(note, "the superblock starts at 512");
}

/// The userblock is the application's, and reading the file must not touch it:
/// its bytes are not addressable through the HDF5 address space.
#[test]
fn the_userblock_bytes_stay_the_applications_own() {
    let bytes = std::fs::read(fixture("userblock.h5")).unwrap();
    assert!(bytes.starts_with(b"#!/bin/sh\n"), "the fixture's userblock");
    assert_eq!(&bytes[512..520], b"\x89HDF\r\n\x1a\n");

    let file = H5File::open(fixture("userblock.h5")).unwrap();
    drop(file);

    assert_eq!(std::fs::read(fixture("userblock.h5")).unwrap(), bytes);
}

/// A file without a userblock keeps reporting zero, so the accessor cannot be
/// read as "the search found something".
#[test]
fn a_file_without_a_userblock_reports_zero() {
    let file = H5File::open(fixture("ochk_root.h5")).unwrap();
    assert_eq!(file.userblock_size(), 0);
}

/// The signature search must not turn a file that has no superblock at all
/// into a slow scan or a spurious success: it fails the same way it did before.
#[test]
fn a_file_with_no_signature_anywhere_is_still_rejected() {
    let dir = std::env::temp_dir().join(format!("rust_hdf5_userblock_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("not_hdf5.bin");
    std::fs::write(&path, vec![0x5au8; 4096]).unwrap();

    let text = match H5File::open(&path) {
        Ok(_) => panic!("a file with no superblock must not open"),
        Err(e) => e.to_string(),
    };
    assert!(text.contains("invalid HDF5 signature"), "{text}");

    let _ = std::fs::remove_dir_all(&dir);
}
