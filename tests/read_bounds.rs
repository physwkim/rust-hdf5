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
    assert!(is_refused(ds.write_slice::<u8>(
        &[0, 0],
        &[usize::MAX, usize::MAX],
        &[99]
    )));
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
    refresh_ohdr_checksum(&mut bytes, at);
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

/// Patch the one `OHDR` the byte offset `at` falls in: refresh the checksum
/// of its first chunk (these headers are one chunk).
fn refresh_ohdr_checksum(bytes: &mut [u8], at: usize) {
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
}

/// A variable-length dataset whose extent claims far more elements than its
/// reference image holds: the read is bounded by the image, not sized by
/// the claim.
#[test]
fn a_vlen_extent_past_its_image_does_not_size_the_read() {
    let path = unique_tmp("vlen_extent");
    let strings: Vec<String> = (0..37).map(|i| format!("s{i}")).collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("notes", &refs).unwrap();
        file.close().unwrap();
    }

    // The dataspace stores the extent and then its maximum as adjacent
    // 8-byte counts; the pair is what tells them from any other 37.
    let mut bytes = std::fs::read(&path).unwrap();
    let mut needle = 37u64.to_le_bytes().to_vec();
    needle.extend_from_slice(&37u64.to_le_bytes());
    let hits: Vec<usize> = bytes
        .windows(16)
        .enumerate()
        .filter(|(_, w)| *w == needle.as_slice())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "extent pair found at {hits:?}");
    let at = hits[0];
    bytes[at..at + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    bytes[at + 8..at + 16].copy_from_slice(&(1u64 << 40).to_le_bytes());
    refresh_ohdr_checksum(&mut bytes, at);
    std::fs::write(&path, &bytes).unwrap();

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("notes").unwrap();
    assert_eq!(ds.shape(), vec![1usize << 40]);
    assert_eq!(ds.read_vlen_strings().unwrap(), strings);
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}

/// A fixed-array chunk index whose header claims more elements than the file
/// could hold is refused before the claim sizes anything.
#[test]
fn a_fixed_array_element_count_past_the_file_is_refused() {
    let path = unique_tmp("fa_count");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([8usize, 4])
            .chunk(&[1, 4])
            .max_shape(&[Some(8), Some(4)])
            .create("grid")
            .unwrap();
        ds.write_slice(&[0, 0], &[8, 4], &(0..32).collect::<Vec<i32>>())
            .unwrap();
        file.close().unwrap();
    }

    // FAHD: signature, version, client id, element size, page-size bits,
    // the element count (8 bytes here), data block address, checksum.
    let mut bytes = std::fs::read(&path).unwrap();
    let fahd = bytes
        .windows(4)
        .position(|w| w == b"FAHD")
        .expect("a fixed array header");
    let count = u64::from_le_bytes(bytes[fahd + 8..fahd + 16].try_into().unwrap());
    assert_eq!(count, 8);
    bytes[fahd + 8..fahd + 16].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let end = fahd + 8 + 8 + 8;
    let sum = rust_hdf5::format::checksum::checksum_metadata(&bytes[fahd..end]);
    bytes[end..end + 4].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("grid").unwrap();
    let err = ds
        .read_slice::<i32>(&[0, 0], &[1, 4])
        .expect_err("a count past the file must not size the read");
    let msg = format!("{err}");
    assert!(
        msg.contains("more than the") && msg.contains("file holds"),
        "unexpected error: {msg}"
    );
    drop(ds);
    file.close().unwrap();

    cleanup(&path);
}

/// An external file list slot whose file offset is a few bytes short of
/// `u64::MAX`: a selection that starts past that gap wraps, and the read
/// refuses rather than serving the slot from the wrapped offset.
#[test]
fn an_external_slot_offset_that_wraps_is_refused() {
    let path = unique_tmp("wrap_efl");
    let dir = path.parent().unwrap().to_path_buf();
    // The raw file holds 0..24 at slot offset 12345. The slot records that
    // offset next to its size, the one place the two numbers sit together.
    let mut raw = vec![0u8; 12345];
    raw.extend(0..24u8);
    std::fs::write(dir.join("raw.bin"), &raw).unwrap();
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<u8>()
            .shape([24usize])
            .external(&[("raw.bin", 12345, 24)])
            .efile_prefix(dir.display().to_string())
            .create("e")
            .unwrap();
        file.close().unwrap();
    }
    let access = || rust_hdf5::DatasetAccess::new().efile_prefix(dir.display().to_string());
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset_with("e", access()).unwrap();
        assert_eq!(
            ds.read_slice::<u8>(&[8], &[8]).unwrap(),
            (8..16).collect::<Vec<_>>()
        );
        drop(ds);
        file.close().unwrap();
    }

    let mut needle = 12345u64.to_le_bytes().to_vec();
    needle.extend_from_slice(&24u64.to_le_bytes());
    patch_unique(&path, &needle, |m| {
        m[..8].copy_from_slice(&(u64::MAX - 4).to_le_bytes());
    });

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset_with("e", access()).unwrap();
    let err = ds
        .read_slice::<u8>(&[8], &[8])
        .expect_err("a wrapping slot offset must not read");
    let msg = format!("{err}");
    assert!(msg.contains("overflows"), "unexpected error: {msg}");
    drop(ds);
    file.close().unwrap();

    // The writer walks the same slots, and must not land bytes at the
    // wrapped offset either.
    let file = H5File::open_rw(&path).unwrap();
    let ds = file.dataset_writer_with("e", access()).unwrap();
    let err = ds
        .write_slice(&[8], &[8], &[0u8; 8])
        .expect_err("a wrapping slot offset must not write");
    let msg = format!("{err}");
    assert!(msg.contains("overflows"), "unexpected error: {msg}");
    drop(ds);
    file.close().unwrap();
    assert_eq!(std::fs::read(dir.join("raw.bin")).unwrap(), raw);

    cleanup(&path);
}
