//! A chunked layout whose stored dimensionality disagrees with the dataspace
//! rank (CVE-2026-19025).
//!
//! The chunk layout message stores the chunk rank plus one trailing
//! element-size dimension, so its dimensionality must be exactly one more
//! than the dataspace rank. libhdf5 checked that only at dataset creation
//! (`H5D__chunk_construct`) until HDFGroup/hdf5#6508 moved the check into
//! `H5O__layout_decode`, which now refuses the message against its sibling
//! dataspace as it decodes. Before that, the disagreeing ranks reached chunk
//! I/O, where the memory selection (dataspace rank) and the file selection
//! (chunk rank) produced a zero stride and a divide-by-zero in
//! `H5S__hyper_iter_get_seq_list`.
//!
//! `bad_chunk_ndims.h5` is upstream's `test/testfiles/bad_chunk_ndims.h5`
//! (written by `test/gen_bad_chunk.c`): a 3x4x5 `int` dataset with 2x2x4
//! chunks under a version-3 layout message whose dimensionality byte was
//! patched from 4 to 3, so the layout claims rank-2 chunks of 4-byte
//! elements over a rank-3 dataspace.

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
        "rust_hdf5_bad_chunk_{}_{}_{}",
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

/// The reason the read side gives for the dataset, or a panic naming what it
/// did instead.
fn refusal(file: &H5File) -> String {
    let names = file.dataset_names();
    assert!(
        names.iter().any(|n| n.trim_start_matches('/') == "dset"),
        "dataset missing from the listing: {names:?}"
    );
    match file.dataset("dset") {
        Ok(_) => panic!("opening a dataset with mismatched chunk/dataspace rank should fail"),
        Err(e) => e.to_string(),
    }
}

/// The dataset is a name the file contains, so it stays in the listing, and
/// opening it reports the rank disagreement the way `H5Dopen2` now fails
/// (`test_chunk_dims_mismatch`, test/dsets.c) instead of handing back a
/// handle whose chunk grid cannot be indexed.
#[test]
fn mismatched_chunk_rank_is_listed_but_refused_at_open() {
    let file = H5File::open(fixture("bad_chunk_ndims.h5")).unwrap();
    let err = refusal(&file);
    assert!(
        err.contains("dimensionality of its chunks"),
        "open failed for a reason other than the chunk rank: {err}"
    );
}

/// A read-write open walks every object to rebuild its registry, and the
/// close rewrites each group out of it. The dataset it cannot model keeps its
/// bytes through that round trip: afterwards it is still listed and still
/// refused for the same reason, rather than rebuilt at the wrong rank or
/// dropped from its group.
#[test]
fn mismatched_chunk_rank_survives_a_read_write_reopen_by_its_bytes() {
    let path = unique_tmp("reopen");
    std::fs::copy(fixture("bad_chunk_ndims.h5"), &path).unwrap();

    let file = H5File::open_rw(&path).unwrap();
    assert!(
        file.dataset("dset").is_err(),
        "a write-mode open of the mismatched dataset should fail"
    );
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let err = refusal(&file);
    assert!(
        err.contains("dimensionality of its chunks"),
        "after the reopen the dataset is refused for a different reason: {err}"
    );
    drop(file);
    cleanup(&path);
}
