//! A selection the extent does not admit is refused at the API boundary,
//! whatever its shape: a start near `usize::MAX`, whose sum with the count
//! wraps back inside the extent, must not turn into an offset that reads
//! (or writes) unrelated bytes. The same holds for the file's own claims: a
//! contiguous address whose sum with a run offset wraps is an error, not a
//! read of whatever lies at the wrapped offset.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_read_bounds_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("test.h5")
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// A 4x6 u8 dataset holding 0..24 in row-major order.
fn write_grid(path: &PathBuf) {
    let file = H5File::create(path).unwrap();
    let ds = file
        .new_dataset::<u8>()
        .shape([4usize, 6])
        .create("g")
        .unwrap();
    ds.write_raw_bytes(&(0..24u8).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

fn is_refused<T>(r: rust_hdf5::Result<T>) -> bool {
    match r {
        Ok(_) => false,
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("out of bounds"), "unexpected error: {msg}");
            true
        }
    }
}

/// Every reader entry point that takes a start refuses a start whose edge
/// wraps: `read_slice`, `read_slice_into` and `read_points`.
#[test]
fn a_wrapping_start_is_refused_on_every_read_path() {
    let path = unique_tmp("read");
    write_grid(&path);

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("g").unwrap();
    // The in-bounds neighbours of each refused case still read.
    assert_eq!(ds.read_slice::<u8>(&[3, 5], &[1, 1]).unwrap(), vec![23]);
    assert_eq!(ds.read_points::<u8>(&[vec![3, 5]]).unwrap(), vec![23]);

    assert!(is_refused(ds.read_slice::<u8>(&[usize::MAX, 0], &[1, 1])));
    assert!(is_refused(ds.read_slice::<u8>(&[0, usize::MAX], &[1, 1])));
    assert!(is_refused(ds.read_slice::<u8>(&[1, 0], &[usize::MAX, 1])));
    assert!(is_refused(ds.read_slice::<u8>(&[3, 5], &[2, 1])));

    let mut one = [0u8; 1];
    assert!(is_refused(ds.read_slice_into::<u8>(
        &mut one,
        &[usize::MAX, 0],
        &[1, 1]
    )));
    assert!(is_refused(ds.read_slice_into::<u8>(
        &mut one,
        &[0, 6],
        &[1, 1]
    )));

    assert!(is_refused(ds.read_points::<u8>(&[vec![usize::MAX, 0]])));
    assert!(is_refused(ds.read_points::<u8>(&[vec![0, 6]])));
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}

/// An oversized count is reported as a selection problem before its byte
/// size is computed, not as a failed allocation.
#[test]
fn an_oversized_count_is_refused_before_allocation() {
    let path = unique_tmp("count");
    write_grid(&path);

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("g").unwrap();
    assert!(is_refused(ds.read_slice::<u8>(&[0, 0], &[usize::MAX, 6])));
    assert!(is_refused(
        ds.read_slice::<u8>(&[0, 0], &[1 << 40, 1 << 20])
    ));
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}

/// The writer's `write_slice` shares the rule, so a wrapping start cannot
/// land bytes over a neighbour.
#[test]
fn a_wrapping_start_is_refused_on_write() {
    let path = unique_tmp("write");
    let file = H5File::create(&path).unwrap();
    let ds = file
        .new_dataset::<u8>()
        .shape([4usize, 6])
        .create("g")
        .unwrap();
    ds.write_raw_bytes(&(0..24u8).collect::<Vec<_>>()).unwrap();
    assert!(is_refused(ds.write_slice::<u8>(
        &[usize::MAX, 0],
        &[1, 1],
        &[99]
    )));
    assert!(is_refused(ds.write_slice::<u8>(
        &[3, 5],
        &[1, 2],
        &[99, 99]
    )));
    assert!(ds
        .write_slice::<u8>(&[0, 0], &[usize::MAX, usize::MAX], &[99])
        .is_err());
    ds.write_slice::<u8>(&[3, 5], &[1, 1], &[99]).unwrap();
    drop(ds);
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let mut want: Vec<u8> = (0..24).collect();
    want[23] = 99;
    assert_eq!(
        file.dataset("g")
            .unwrap()
            .read_slice::<u8>(&[0, 0], &[4, 6])
            .unwrap(),
        want
    );

    cleanup(&path);
}

/// Find the one occurrence of `needle` in the file, hand the bytes from it
/// to `mutate`, and refresh the checksum of the version-2 object header the
/// patched bytes sit in (its first chunk; the headers here are one chunk).
fn patch_unique(path: &PathBuf, needle: &[u8], mutate: impl FnOnce(&mut [u8])) {
    let mut bytes = std::fs::read(path).unwrap();
    let hits: Vec<usize> = bytes
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "pattern occurs {} times", hits.len());
    let at = hits[0];
    mutate(&mut bytes[at..]);

    // OHDR v2: signature, version, flags, optional times (flag 0x20),
    // optional attribute phase-change (0x10), chunk-0 size in 1 << (flags & 3)
    // bytes, the messages, then a checksum over everything before it.
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
        at < end,
        "patched bytes are not in the header's first chunk"
    );
    let sum = rust_hdf5::format::checksum::checksum_metadata(&bytes[ohdr..end]);
    bytes[end..end + 4].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(path, &bytes).unwrap();
}

/// A contiguous layout whose address is a few bytes short of `u64::MAX`:
/// a run offset past that gap wraps, and the read refuses rather than
/// serving the bytes at the wrapped offset.
#[test]
fn a_contiguous_address_that_wraps_is_refused() {
    let path = unique_tmp("wrap_addr");
    write_grid(&path);

    // The version-3 contiguous layout message: version, class 1, address,
    // size. The data address is where the 0..24 payload sits.
    let bytes = std::fs::read(&path).unwrap();
    let payload: Vec<u8> = (0..24).collect();
    let data_addr = bytes
        .windows(24)
        .position(|w| w == payload.as_slice())
        .expect("payload in file") as u64;
    let mut needle = vec![3u8, 1];
    needle.extend_from_slice(&data_addr.to_le_bytes());
    needle.extend_from_slice(&24u64.to_le_bytes());
    patch_unique(&path, &needle, |m| {
        m[2..10].copy_from_slice(&(u64::MAX - 4).to_le_bytes());
    });

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("g").unwrap();
    let err = ds
        .read_slice::<u8>(&[3, 5], &[1, 1])
        .expect_err("a wrapping run offset must not read");
    let msg = format!("{err}");
    assert!(msg.contains("overflows"), "unexpected error: {msg}");
    // A run that does not wrap fails on the read itself, never silently.
    assert!(ds.read_slice::<u8>(&[0, 0], &[1, 1]).is_err());
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}
