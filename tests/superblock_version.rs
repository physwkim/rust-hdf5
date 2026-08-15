//! Which superblock version this crate writes.
//!
//! libhdf5 does not pick it from a knob: `H5F__super_init` takes the oldest
//! version that can describe the file and raises it to the one the file's
//! library-version low bound implies (`HDF5_superblock_ver_bounds`,
//! H5Fsuper.c:68). Both halves are in play here: `H5FileOptions::libver`
//! contributes the bound's own entry, and what the file holds contributes the
//! rest — link-message groups and version-2 object headers put the floor at
//! v1.8 (version 2), a chunked dataset's version-4/5 layout puts it at v1.10
//! (version 3), and SWMR is version 3 outright. A reopened file keeps the
//! version it already has unless what is appended needs a newer one.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{H5File, LibverBound};

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_superblock_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The version byte that follows the signature.
fn superblock_version(path: &Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n", "{}", path.display());
    bytes[8]
}

fn write_contiguous(file: &H5File, name: &str) {
    file.new_dataset::<i32>()
        .shape([8usize])
        .create(name)
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
}

fn write_chunked(file: &H5File, name: &str) {
    file.new_dataset::<i32>()
        .shape([8usize])
        .chunk(&[4])
        .create(name)
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
}

/// Contiguous data, compact links and attributes are all expressible in the
/// v1.8 format, which is where this crate's floor is: it writes no
/// symbol-table group and no version-1 object header, so it never has cause
/// to claim the version-0 superblock libhdf5 writes under earliest bounds.
#[test]
fn a_file_of_contiguous_datasets_is_written_at_version_2() {
    let path = unique_tmp("contiguous");
    let file = H5File::create(&path).unwrap();
    write_contiguous(&file, "data");
    file.root_group().create_group("g").unwrap();
    file.set_attr_string("note", "root attribute").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 2);
    cleanup(&path);
}

/// A chunked dataset is indexed by an extensible array, a fixed array or a
/// version-2 B-tree, all of them 1.10 structures reached through a version-4
/// data layout message — that is a V110 low bound, hence version 3.
#[test]
fn a_file_with_a_chunked_dataset_is_written_at_version_3() {
    let path = unique_tmp("chunked");
    let file = H5File::create(&path).unwrap();
    write_chunked(&file, "data");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 3);
    cleanup(&path);
}

/// One case per entry of `HDF5_superblock_ver_bounds`, against a file whose
/// content asks for nothing above the floor. The bound's own entry is 0 for
/// EARLIEST and 2 for V18, both below this crate's version-2 floor, so the two
/// oldest bounds land on the same version from opposite directions; V110 and
/// everything after it are 3.
#[test]
fn each_libver_bound_selects_its_superblock_version() {
    for (bound, expected) in [
        (LibverBound::Earliest, 2),
        (LibverBound::V18, 2),
        (LibverBound::V110, 3),
        (LibverBound::V112, 3),
        (LibverBound::V114, 3),
        (LibverBound::V200, 3),
    ] {
        let path = unique_tmp(&format!("bound_{bound:?}"));
        let file = H5File::options().libver(bound).create(&path).unwrap();
        write_contiguous(&file, "data");
        file.root_group().create_group("g").unwrap();
        file.close().unwrap();

        assert_eq!(superblock_version(&path), expected, "bound {bound:?}");
        cleanup(&path);
    }
}

/// The bound is a floor, not an override: a v1.8 file holding a chunked
/// dataset still needs the version-3 superblock its version-4 layout message
/// implies.
#[test]
fn a_chunked_dataset_raises_a_v18_file_to_version_3() {
    let path = unique_tmp("v18_chunked");
    let file = H5File::options()
        .libver(LibverBound::V18)
        .create(&path)
        .unwrap();
    write_chunked(&file, "data");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 3);
    cleanup(&path);
}

/// `set_libver_latest` asks for the 2.0 format, whose bound is LATEST.
#[test]
fn libver_latest_raises_a_contiguous_file_to_version_3() {
    let path = unique_tmp("latest");
    let file = H5File::create(&path).unwrap();
    file.set_libver_latest(true).unwrap();
    write_contiguous(&file, "data");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 3);
    cleanup(&path);
}

/// SWMR is version 3 in `H5F__super_init` before any bound is consulted.
#[test]
fn an_swmr_file_is_written_at_version_3() {
    use rust_hdf5::swmr::SwmrFileWriter;

    let path = unique_tmp("swmr");
    let mut writer = SwmrFileWriter::create(&path).unwrap();
    let idx = writer
        .create_streaming_dataset::<f32>("stream", &[4u64])
        .unwrap();
    writer.start_swmr().unwrap();
    assert_eq!(superblock_version(&path), 3, "at start_swmr");
    let frame: Vec<u8> = (0..4u32).flat_map(|j| (j as f32).to_le_bytes()).collect();
    writer.append_frame(idx, &frame).unwrap();
    writer.close().unwrap();

    assert_eq!(superblock_version(&path), 3);
    cleanup(&path);
}

/// The version follows the file's content, not the writing session: an
/// append that adds nothing newer hands the file back at the version it was
/// opened with.
#[test]
fn appending_contiguous_data_leaves_a_version_2_file_at_version_2() {
    let path = unique_tmp("append_contig");
    let file = H5File::create(&path).unwrap();
    write_contiguous(&file, "data");
    file.close().unwrap();
    assert_eq!(superblock_version(&path), 2);

    let file = H5File::open_rw(&path).unwrap();
    write_contiguous(&file, "added");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 2);
    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, vec!["added", "data"]);
    drop(file);
    cleanup(&path);
}

/// ... and one that does need a newer one raises it, because the file now
/// holds a 1.10 chunk index.
#[test]
fn appending_a_chunked_dataset_raises_a_version_2_file_to_version_3() {
    let path = unique_tmp("append_chunked");
    let file = H5File::create(&path).unwrap();
    write_contiguous(&file, "data");
    file.close().unwrap();
    assert_eq!(superblock_version(&path), 2);

    let file = H5File::open_rw(&path).unwrap();
    write_chunked(&file, "added");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 3);
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..8i32).collect::<Vec<_>>()
    );
    drop(file);
    cleanup(&path);
}

/// The version is a floor, never a target: deleting what forced version 3
/// does not walk the file back to version 2, which would strand any reader
/// that recorded the older bound.
#[test]
fn a_version_3_file_stays_at_version_3_when_its_chunked_dataset_goes_away() {
    let path = unique_tmp("no_downgrade");
    let file = H5File::create(&path).unwrap();
    write_contiguous(&file, "data");
    write_chunked(&file, "chunky");
    file.close().unwrap();
    assert_eq!(superblock_version(&path), 3);

    let file = H5File::open_rw(&path).unwrap();
    file.delete_dataset("chunky").unwrap();
    write_contiguous(&file, "added");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 3);
    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, vec!["added", "data"]);
    drop(file);
    cleanup(&path);
}
