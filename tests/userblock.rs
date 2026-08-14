//! Files whose superblock is preceded by a userblock.
//!
//! The superblock does not have to start at offset 0: with a userblock it
//! starts at a power-of-two offset of at least 512, and `H5FD_locate_signature`
//! finds it by probing 0, 512, 1024, ... . That offset is the file's base
//! address — every address in the file is measured from it — so a reader that
//! assumes offset 0 does not merely mis-read the userblock, it cannot open the
//! file at all.
//!
//! `userblock.h5` and `userblock_v0.h5` (from
//! `tests/fixtures/gen_userblock.py`) both have a 512-byte userblock holding a
//! shell shebang, one dataset and one root attribute; they differ only in
//! superblock version, and only the version-2 one can be appended to. The
//! tests below them create the same shape of file from this crate, through
//! `H5FileOptions::userblock`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

const USERBLOCK: usize = 512;

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
        "rust_hdf5_userblock_{}_{}_{}",
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

/// The search runs before the superblock version is looked at, so the classic
/// (version-0 superblock) format is found behind a userblock too.
#[test]
fn a_classic_superblock_behind_a_userblock_opens_and_reads() {
    let file = H5File::open(fixture("userblock_v0.h5")).unwrap();

    assert_eq!(file.userblock_size(), 512);
    assert_eq!(file.dataset_names(), vec!["data"]);
    let data = file.dataset("data").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// A file without a userblock keeps reporting zero, so the accessor cannot be
/// read as "the search found something".
#[test]
fn a_file_without_a_userblock_reports_zero() {
    let file = H5File::open(fixture("ochk_root.h5")).unwrap();
    assert_eq!(file.userblock_size(), 0);
}

/// Little-endian `u64` at `field` in the version-2/3 superblock that starts
/// at `base`.
fn sb_u64(bytes: &[u8], base: usize, field: usize) -> u64 {
    // signature(8) version(1) sizeof_offsets(1) sizeof_lengths(1) flags(1),
    // then the four 8-byte addresses.
    let at = base + 12 + field * 8;
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Appending has to work in the same based address space: the new blocks go
/// after the file's end, the superblock is rewritten at 512 (not at 0), and the
/// userblock is left exactly as the application wrote it.
#[test]
fn appending_behind_a_userblock_keeps_the_block_and_both_datasets() {
    let original = std::fs::read(fixture("userblock.h5")).unwrap();
    let path = unique_tmp("append");
    std::fs::write(&path, &original).unwrap();

    {
        let file = H5File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .create("added")
            .unwrap()
            .write_raw(&[100i32, 101, 102, 103])
            .unwrap();
        file.close().unwrap();
    }

    // A second round reopens the superblock this crate wrote, so the base
    // address survives its own round trip and the allocator picks up from the
    // grown end of file.
    {
        let file = H5File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([2])
            .create("again")
            .unwrap()
            .write_raw(&[200i32, 201])
            .unwrap();
        file.close().unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(
        &after[..USERBLOCK],
        &original[..USERBLOCK],
        "the writer must never touch [0, base)"
    );
    assert_eq!(&after[USERBLOCK..USERBLOCK + 8], b"\x89HDF\r\n\x1a\n");
    // h5py wrote this fixture at version 2 and the appended datasets are
    // contiguous, so nothing in it asks for a newer superblock.
    assert_eq!(after[USERBLOCK + 8], 2, "superblock version");
    // The rewritten superblock keeps the userblock size (H5Pget_userblock
    // reports this field) and records the end of file measured from the start
    // of the file: libhdf5 takes the allocated end as `stored_eof -
    // base_addr`, so a value short by the userblock puts every appended object
    // "past end of allocation".
    assert_eq!(
        sb_u64(&after, USERBLOCK, 0),
        USERBLOCK as u64,
        "base_address"
    );
    assert_eq!(
        sb_u64(&after, USERBLOCK, 2),
        after.len() as u64,
        "end_of_file_address"
    );

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.userblock_size(), 512);
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, vec!["added", "again", "data"]);
    assert_eq!(
        file.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        vec![100, 101, 102, 103]
    );
    assert_eq!(
        file.dataset("again").unwrap().read_raw::<i32>().unwrap(),
        vec![200, 201]
    );
    // sbext measured the append path dropping root-group attributes on any
    // file, userblock or not; the root attribute lives in chunk 0 of the root
    // header, so this is the exact repro re-run.
    assert_eq!(
        file.attr_string("note").unwrap(),
        "the superblock starts at 512"
    );
    drop(file);
    cleanup(&path);
}

/// The classic-format refusal is about the superblock version, and it must
/// still be that refusal — not a signature error — behind a userblock.
#[test]
fn appending_to_a_classic_file_behind_a_userblock_is_refused_as_classic() {
    let original = std::fs::read(fixture("userblock_v0.h5")).unwrap();
    let path = unique_tmp("append_v0");
    std::fs::write(&path, &original).unwrap();

    let text = match H5File::open_rw(&path) {
        Ok(_) => panic!("the append path does not support the classic format"),
        Err(e) => e.to_string(),
    };
    assert!(text.contains("version-0/1 superblock"), "{text}");
    assert_eq!(std::fs::read(&path).unwrap(), original);
    cleanup(&path);
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

/// Creating with a userblock puts the whole HDF5 address space behind it: the
/// signature lands at the block's end, the superblock records the block as the
/// base address and the file's own length as the end of file, and the block
/// itself is left zeroed for the application to fill.
#[test]
fn creating_with_a_userblock_bases_the_file_at_it() {
    let path = unique_tmp("create_512");
    {
        let file = H5File::options().userblock(512).create(&path).unwrap();
        assert_eq!(file.userblock_size(), 512);
        file.new_dataset::<i32>()
            .shape([8])
            .create("data")
            .unwrap()
            .write_raw(&(0..8i32).collect::<Vec<_>>())
            .unwrap();
        file.root_group().create_group("g").unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..USERBLOCK], &vec![0u8; USERBLOCK][..]);
    assert_eq!(&bytes[USERBLOCK..USERBLOCK + 8], b"\x89HDF\r\n\x1a\n");
    assert_eq!(
        sb_u64(&bytes, USERBLOCK, 0),
        USERBLOCK as u64,
        "base_address"
    );
    assert_eq!(
        sb_u64(&bytes, USERBLOCK, 2),
        bytes.len() as u64,
        "end_of_file_address"
    );

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.userblock_size(), 512);
    assert_eq!(
        file.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        (0..8i32).collect::<Vec<_>>()
    );
    assert!(file.root_group().group("g").is_ok());
    drop(file);
    cleanup(&path);
}

/// The block is the application's to write, and doing so must leave the file
/// readable — the search finds the signature at the same offset whatever the
/// block holds. Appending afterwards goes on working in the based space.
#[test]
fn a_filled_userblock_survives_a_read_and_an_append() {
    let path = unique_tmp("create_filled");
    {
        let file = H5File::options().userblock(512).create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .create("data")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        file.close().unwrap();
    }

    let mut block = b"#!/bin/sh\n# userblock\n".to_vec();
    block.resize(USERBLOCK, b'#');
    {
        use std::io::Write;
        let mut fh = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        fh.write_all(&block).unwrap();
    }

    {
        let file = H5File::open_rw(&path).unwrap();
        assert_eq!(file.userblock_size(), 512);
        file.new_dataset::<i32>()
            .shape([2])
            .create("added")
            .unwrap()
            .write_raw(&[5i32, 6])
            .unwrap();
        file.close().unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(&after[..USERBLOCK], &block[..], "the block is untouched");
    assert_eq!(
        sb_u64(&after, USERBLOCK, 2),
        after.len() as u64,
        "end_of_file_address"
    );

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        vec![5, 6]
    );
    drop(file);
    cleanup(&path);
}

/// A userblock larger than the minimum is found by the same doubling search,
/// so every power of two above 512 works and bases the file just as far in.
#[test]
fn a_larger_power_of_two_userblock_works_the_same() {
    let path = unique_tmp("create_4096");
    {
        let file = H5File::options().userblock(4096).create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .create("data")
            .unwrap()
            .write_raw(&[7i32, 8, 9, 10])
            .unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4096], &vec![0u8; 4096][..]);
    assert_eq!(&bytes[4096..4096 + 8], b"\x89HDF\r\n\x1a\n");
    assert_eq!(sb_u64(&bytes, 4096, 0), 4096, "base_address");
    assert_eq!(
        sb_u64(&bytes, 4096, 2),
        bytes.len() as u64,
        "end_of_file_address"
    );

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.userblock_size(), 4096);
    assert_eq!(
        file.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9, 10]
    );
    drop(file);
    cleanup(&path);
}

/// A size the search could never land on is refused before the file is
/// created, rather than rounded up into a file no reader can open.
#[test]
fn a_size_the_signature_search_cannot_reach_is_refused() {
    for size in [1u64, 256, 511, 513, 600, 1536] {
        let path = unique_tmp(&format!("bad_{size}"));
        let text = match H5File::options().userblock(size).create(&path) {
            Ok(_) => panic!("a {size}-byte userblock must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(text.contains(&size.to_string()), "{text}");
        assert!(!path.exists(), "the file must not be created: {size}");
        cleanup(&path);
    }
}

/// Zero is the default and means no block at all: the superblock is at offset
/// 0 and the file is byte-for-byte the one the plain constructor writes.
#[test]
fn a_zero_userblock_is_a_file_with_no_block() {
    let path = unique_tmp("create_zero");
    {
        let file = H5File::options().userblock(0).create(&path).unwrap();
        assert_eq!(file.userblock_size(), 0);
        file.new_dataset::<i32>()
            .shape([4])
            .create("data")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        file.close().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
    assert_eq!(sb_u64(&bytes, 0, 0), 0, "base_address");

    let file = H5File::open(&path).unwrap();
    assert_eq!(file.userblock_size(), 0);
    drop(file);
    cleanup(&path);
}
