//! Global-heap resolution hardening for variable-length data.
//!
//! A reference whose collection cannot be resolved must be a hard error,
//! the way libhdf5's `H5HG__cache_heap_deserialize` fails on a bad
//! signature or an undersized collection — not a silent run of empty
//! strings. And a collection larger than 64 MiB is not an error at all:
//! libhdf5 has no upper cap, and this crate's writers put a whole write
//! call's strings into one collection, so a cap silently blanks every
//! string of a large batch.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_vlen_heap_{}_{}_{}",
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

/// Find the file's first global-heap collection and hand its bytes to
/// `mutate`. The tests below write exactly one collection per file, so
/// "first" is "the" collection.
fn patch_gcol(path: &PathBuf, mutate: impl FnOnce(&mut [u8])) {
    let mut bytes = std::fs::read(path).unwrap();
    let off = bytes
        .windows(4)
        .position(|w| w == b"GCOL")
        .expect("no GCOL collection in file");
    mutate(&mut bytes[off..]);
    std::fs::write(path, &bytes).unwrap();
}

/// One write call's strings all go into a single collection, so 66 x 1 MiB
/// puts the collection past the reader's former 64 MiB cap — which turned
/// every one of them into an empty string on read-back.
#[test]
fn vlen_collection_above_64_mib_roundtrips() {
    let path = unique_tmp("big_collection");
    let strings: Vec<String> = (0..66)
        .map(|i| {
            let c = char::from(b'a' + (i % 26) as u8);
            std::iter::repeat_n(c, 1 << 20).collect()
        })
        .collect();
    let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("notes", &refs).unwrap();
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let read = file.dataset("notes").unwrap().read_vlen_strings().unwrap();
    assert_eq!(read, strings);
    drop(file);
    cleanup(&path);
}

/// A collection whose signature is gone is corruption, and reading a
/// string through it must fail loudly instead of yielding "".
#[test]
fn corrupt_gcol_signature_is_a_dataset_read_error() {
    let path = unique_tmp("bad_signature");
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("notes", &["alpha", "beta"])
            .unwrap();
        file.close().unwrap();
    }
    patch_gcol(&path, |g| g[..4].copy_from_slice(b"XXXX"));

    let file = H5File::open(&path).unwrap();
    let err = file
        .dataset("notes")
        .unwrap()
        .read_vlen_strings()
        .expect_err("corrupt collection must not read as empty strings");
    assert!(format!("{err}").contains("signature"), "got: {err}");
    drop(file);
    cleanup(&path);
}

/// A declared collection size below `H5HG_MINSIZE` (4096) is rejected the
/// way libhdf5 rejects it.
#[test]
fn undersized_gcol_declared_size_is_a_read_error() {
    let path = unique_tmp("undersized");
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("notes", &["alpha", "beta"])
            .unwrap();
        file.close().unwrap();
    }
    // Collection size is the 8-byte field after signature+version+reserved.
    patch_gcol(&path, |g| g[8..16].copy_from_slice(&100u64.to_le_bytes()));

    let file = H5File::open(&path).unwrap();
    let err = file
        .dataset("notes")
        .unwrap()
        .read_vlen_strings()
        .expect_err("undersized collection must not read as empty strings");
    assert!(format!("{err}").contains("4096"), "got: {err}");
    drop(file);
    cleanup(&path);
}

/// The attribute path resolves through the same collection loader, so a
/// corrupt collection fails an attribute string read too.
#[test]
fn corrupt_gcol_signature_is_an_attr_read_error() {
    let path = unique_tmp("bad_signature_attr");
    {
        let file = H5File::create(&path).unwrap();
        file.set_attr_string("conventions", "NeXus").unwrap();
        file.close().unwrap();
    }
    patch_gcol(&path, |g| g[..4].copy_from_slice(b"XXXX"));

    let file = H5File::open(&path).unwrap();
    let err = file
        .attr_string("conventions")
        .expect_err("corrupt collection must not read as an empty attribute");
    assert!(format!("{err}").contains("signature"), "got: {err}");
    drop(file);
    cleanup(&path);
}
