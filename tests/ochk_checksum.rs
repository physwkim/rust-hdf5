//! Object-header continuation chunk ("OCHK") integrity.
//!
//! A version-2 continuation chunk ends with a Jenkins checksum over its whole
//! image, and `H5O__cache_chk_verify_chksum` refuses the chunk when it does not
//! match; `H5O__chunk_deserialize` refuses it when the signature is not
//! "OCHK". Taking the messages without either check turns a corrupt chunk into
//! plausible-looking metadata: attributes that are not in the file, or a
//! dataset with the wrong datatype.
//!
//! `ochk_root.h5` (from `tests/fixtures/gen_ochk.sh`) puts the chunk on the
//! root group, which every open reads.

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
        "rust_hdf5_ochk_{}_{}_{}",
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

/// Byte offset of the first continuation chunk in the file.
fn first_ochk_offset(bytes: &[u8]) -> usize {
    bytes
        .windows(4)
        .position(|w| w == b"OCHK")
        .expect("the fixture holds a continuation chunk")
}

/// Write `bytes` with the byte at `at` flipped, and return the path.
fn corrupted(label: &str, bytes: &[u8], at: usize) -> PathBuf {
    let mut broken = bytes.to_vec();
    broken[at] ^= 0xff;
    let path = unique_tmp(label);
    std::fs::write(&path, &broken).unwrap();
    path
}

/// The message of the error `H5File::open` fails with. `H5File` is not
/// `Debug`, so `expect_err` is not available.
fn open_error(path: &PathBuf) -> String {
    match H5File::open(path) {
        Ok(_) => panic!("a corrupt chunk must not open"),
        Err(e) => e.to_string(),
    }
}

/// Positive control for the corruption tests below: the untouched fixture has
/// a chunk that passes verification, so a failure there is the corruption and
/// not the check itself.
#[test]
fn an_intact_continuation_chunk_is_accepted() {
    let bytes = std::fs::read(fixture("ochk_root.h5")).unwrap();
    first_ochk_offset(&bytes);

    let path = unique_tmp("intact");
    std::fs::write(&path, &bytes).unwrap();
    let file = H5File::open(&path).unwrap();
    let mut attrs = file.attr_names().unwrap();
    attrs.sort();
    assert_eq!(
        attrs,
        ["note0", "note1", "note2", "note3", "note4", "note5"]
    );
    assert_eq!(file.dataset_names(), vec!["data"]);
    drop(file);
    cleanup(&path);
}

#[test]
fn a_flipped_byte_inside_a_continuation_chunk_is_rejected() {
    let bytes = std::fs::read(fixture("ochk_root.h5")).unwrap();
    // The first message header inside the chunk: past the 4-byte signature,
    // and well before the trailing checksum.
    let path = corrupted("flipped_message", &bytes, first_ochk_offset(&bytes) + 6);

    let text = open_error(&path);
    assert!(text.contains("checksum mismatch"), "{text}");
    cleanup(&path);
}

#[test]
fn a_continuation_chunk_with_the_wrong_signature_is_rejected() {
    let bytes = std::fs::read(fixture("ochk_root.h5")).unwrap();
    // The continuation message still points here, so a chunk that does not
    // start with "OCHK" means the address or the chunk is corrupt.
    let path = corrupted("wrong_signature", &bytes, first_ochk_offset(&bytes) + 1);

    let text = open_error(&path);
    assert!(text.contains("invalid HDF5 signature"), "{text}");
    cleanup(&path);
}
