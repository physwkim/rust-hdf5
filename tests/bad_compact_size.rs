//! A compact layout whose stored size disagrees with the dataset's extent.
//!
//! `H5D__compact_init` (H5Dcompact.c) refuses to open a compact dataset
//! whose payload is not exactly the extent's element count times the
//! element size. Read as if it were, a short payload is run past its end by
//! any selection that reaches the missing elements, and a long one carries
//! bytes no element owns.
//!
//! The files here are written by this crate and then patched: the
//! dataspace's first dimension is bumped so the payload comes up short, or
//! cut so it comes up long, and the object header's checksum refreshed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_bad_compact_{}_{}_{}",
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

/// A 4x6 u8 compact dataset holding 0..24 in row-major order.
fn write_compact(path: &PathBuf) {
    let file = H5File::create(path).unwrap();
    let ds = file
        .new_dataset::<u8>()
        .shape([4usize, 6])
        .compact()
        .create("dset")
        .unwrap();
    ds.write_raw_bytes(&(0..24u8).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

/// Rewrite the dataspace's first dimension from `from` to `to` wherever the
/// `[from, 6]` pair is stored (the current dims, and the maximum dims when
/// the message carries them), then refresh the checksum of the version-2
/// object header holding them (its first chunk; these headers are one chunk).
fn patch_dim0(path: &PathBuf, from: u64, to: u64) {
    let mut bytes = std::fs::read(path).unwrap();
    let mut needle = from.to_le_bytes().to_vec();
    needle.extend_from_slice(&6u64.to_le_bytes());
    let hits: Vec<usize> = bytes
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle.as_slice())
        .map(|(i, _)| i)
        .collect();
    assert!(
        !hits.is_empty() && hits.len() <= 2 && hits.last().unwrap() - hits[0] <= 16,
        "dims pair found at {hits:?}"
    );
    for &at in &hits {
        bytes[at..at + 8].copy_from_slice(&to.to_le_bytes());
    }

    // OHDR v2: signature, version, flags, optional times (flag 0x20),
    // optional attribute phase-change (0x10), chunk-0 size in 1 << (flags & 3)
    // bytes, the messages, then a checksum over everything before it.
    let at = hits[0];
    let ohdr = bytes[..at]
        .windows(4)
        .rposition(|w| w == b"OHDR")
        .expect("an OHDR before the patched bytes");
    let flags = bytes[ohdr + 5];
    let mut p = ohdr + 6;
    if flags & 0x20 != 0 {
        p += 16;
    }
    if flags & 0x10 != 0 {
        p += 4;
    }
    let width = 1usize << (flags & 3);
    let mut chunk0 = 0u64;
    for i in 0..width {
        chunk0 |= (bytes[p + i] as u64) << (8 * i);
    }
    p += width;
    let end = p + chunk0 as usize;
    assert!(
        hits.last().unwrap() + 16 <= end,
        "patched bytes are not in the header's first chunk"
    );
    let sum = rust_hdf5::format::checksum::checksum_metadata(&bytes[ohdr..end]);
    bytes[end..end + 4].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(path, &bytes).unwrap();
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
        Ok(_) => panic!("opening a compact dataset whose size mismatches should fail"),
        Err(e) => e.to_string(),
    }
}

/// The untouched file reads, so the refusals below are about the patch.
#[test]
fn a_compact_dataset_of_the_right_size_reads() {
    let path = unique_tmp("intact");
    write_compact(&path);
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("dset").unwrap();
    assert_eq!(ds.read_slice::<u8>(&[3, 5], &[1, 1]).unwrap(), vec![23]);
    drop(ds);
    drop(file);
    cleanup(&path);
}

/// A payload one row short of the extent: listed, refused at open with the
/// size disagreement, never sliced past its end.
#[test]
fn a_short_compact_payload_is_listed_but_refused_at_open() {
    let path = unique_tmp("short");
    write_compact(&path);
    patch_dim0(&path, 4, 5);

    let file = H5File::open(&path).unwrap();
    let err = refusal(&file);
    assert!(
        err.contains("compact storage holds 24 bytes") && err.contains("need 30"),
        "open failed for a reason other than the compact size: {err}"
    );
    drop(file);
    cleanup(&path);
}

/// A payload one row longer than the extent is just as much a mismatch.
#[test]
fn a_long_compact_payload_is_listed_but_refused_at_open() {
    let path = unique_tmp("long");
    write_compact(&path);
    patch_dim0(&path, 4, 3);

    let file = H5File::open(&path).unwrap();
    let err = refusal(&file);
    assert!(
        err.contains("compact storage holds 24 bytes") && err.contains("need 18"),
        "open failed for a reason other than the compact size: {err}"
    );
    drop(file);
    cleanup(&path);
}

/// A read-write open walks every object to rebuild its registry, and the
/// close rewrites each group out of it. The dataset it cannot model keeps its
/// bytes through that round trip: afterwards it is still listed and still
/// refused for the same reason.
#[test]
fn a_short_compact_payload_survives_a_read_write_reopen_by_its_bytes() {
    let path = unique_tmp("reopen");
    write_compact(&path);
    patch_dim0(&path, 4, 5);

    let file = H5File::open_rw(&path).unwrap();
    assert!(
        file.dataset("dset").is_err(),
        "a write-mode open of the mismatched dataset should fail"
    );
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let err = refusal(&file);
    assert!(
        err.contains("compact storage holds 24 bytes"),
        "after the reopen the dataset is refused for a different reason: {err}"
    );
    drop(file);
    cleanup(&path);
}
