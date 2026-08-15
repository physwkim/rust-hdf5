//! The 1.12 reference kinds — `H5R_OBJECT2`, `H5R_DATASET_REGION2` and
//! `H5R_ATTR`, all stored as `H5T_STD_REF`. Reading covers all three; writing
//! covers `H5R_OBJECT2`, the one kind whose element carries its target inline
//! rather than through the global heap.
//!
//! The fixture is libhdf5 1.14.6 output built by `tests/fixtures/gen_revised_refs.c`;
//! h5py 3.x cannot stand in for it, as it raises "Unknown reference type" on
//! an `H5T_STD_REF` dataset even for reading. It holds a 4x6 `matrix` with an
//! attribute `note`, a group `grp`, and one dataset per reference kind:
//! `objrefs` names `/matrix` and `/grp`, `regrefs` selects the hyperslab
//! (1,2)-(2,4) and the points (0,1) and (3,5) of `/matrix`, and `attrrefs`
//! names `/matrix`'s `note`.
//!
//! That h5py blind spot is also why the write case is judged by `h5dump`,
//! which dereferences an `H5T_STD_REF` element and prints what it names.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{H5File, LibverBound, PointSelection, Reference, Selection};

const REVISED_REFS: &[u8] = include_bytes!("fixtures/revised_refs.h5");

fn write_temp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_revised_refs_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.h5"));
    std::fs::write(&path, REVISED_REFS).unwrap();
    path
}

/// A path for a file this crate writes, in a directory of its own.
fn write_path(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_revised_write_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`. Only the directory matters here: `h5dump` is taken
/// from beside the interpreter, so it is the tool of the libhdf5 the rest of
/// the suite is judged against.
const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn h5dump() -> Option<PathBuf> {
    let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
        Ok(p) => vec![p],
        Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
    };
    let found = candidates
        .iter()
        .map(|c| PathBuf::from(c).parent().unwrap().join("h5dump"))
        .find(|t| t.exists());
    if found.is_none() {
        eprintln!("skipping the h5dump cross-check: none of {candidates:?} ships one");
    }
    found
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// An `H5R_OBJECT2` element carries its token inline; both a dataset and a
/// group resolve to their paths.
#[test]
fn object2_references_resolve_to_paths() {
    let path = write_temp("object2");
    let file = H5File::open(&path).unwrap();
    let refs = file.dataset("objrefs").unwrap().read_references().unwrap();
    let paths: Vec<Option<&str>> = refs.iter().map(Reference::path).collect();
    assert_eq!(paths, vec![Some("/matrix"), Some("/grp")]);
    assert!(refs.iter().all(|r| matches!(r, Reference::Object { .. })));
    drop(file);
    cleanup(&path);
}

/// An `H5R_DATASET_REGION2` keeps its selection in a global-heap blob; both
/// selection classes decode to the bounds `H5Sget_select_bounds` reports.
#[test]
fn region2_references_report_their_selections() {
    let path = write_temp("region2");
    let file = H5File::open(&path).unwrap();
    let refs = file.dataset("regrefs").unwrap().read_references().unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].path(), Some("/matrix"));
    assert_eq!(refs[0].bounds(), Some((vec![1, 2], vec![2, 4])));
    assert!(matches!(
        refs[0].selection(),
        Some(Selection::Hyperslab { .. })
    ));
    assert_eq!(refs[1].path(), Some("/matrix"));
    assert_eq!(refs[1].bounds(), Some((vec![0, 1], vec![3, 5])));
    assert_eq!(
        refs[1].selection(),
        Some(&Selection::Points(PointSelection {
            rank: 2,
            points: vec![vec![0, 1], vec![3, 5]],
        }))
    );
    drop(file);
    cleanup(&path);
}

/// An `H5T_STD_REF` dataset written here holds `H5R_OBJECT2` elements that
/// both this crate and libhdf5 follow to their targets.
///
/// The addresses come from the same finalize pass that writes the headers they
/// name, so this is also the case that says a reference element is stamped
/// after every object header has an address.
#[test]
fn object2_references_written_here_resolve_in_libhdf5() {
    let path = write_path("object2");
    let file = H5File::options()
        .libver(LibverBound::V112)
        .create(&path)
        .unwrap();
    let matrix = file
        .new_dataset::<i32>()
        .shape([4])
        .create("matrix")
        .unwrap();
    matrix.write_raw(&[10i32, 20, 30, 40]).unwrap();
    file.create_group("grp").unwrap();
    let refs = file
        .new_dataset::<u64>()
        .std_object_references()
        .shape([2])
        .create("objrefs")
        .unwrap();
    refs.write_object_references(&["/matrix", "/grp"]).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let read = file.dataset("objrefs").unwrap().read_references().unwrap();
    let paths: Vec<Option<&str>> = read.iter().map(Reference::path).collect();
    assert_eq!(paths, vec![Some("/matrix"), Some("/grp")]);
    assert!(read.iter().all(|r| matches!(r, Reference::Object { .. })));
    drop(file);

    if let Some(h5dump) = h5dump() {
        let out = std::process::Command::new(&h5dump)
            .args(["-d", "/objrefs", path.to_str().unwrap()])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "h5dump failed:\n{text}");
        assert!(text.contains("H5T_REFERENCE { H5T_STD_REF }"), "{text}");
        // h5dump prints a dereferenced element as the object it names, and a
        // dataset with its data, so the values prove it followed the address.
        assert!(text.contains("DATASET \""), "{text}");
        assert!(text.contains("10, 20, 30, 40"), "{text}");
        assert!(text.contains("GROUP \""), "{text}");
    }
    cleanup(&path);
}

/// An `H5R_ATTR` names an object and one of its attributes.
#[test]
fn attribute_references_name_the_attribute() {
    let path = write_temp("attr");
    let file = H5File::open(&path).unwrap();
    let refs = file.dataset("attrrefs").unwrap().read_references().unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].path(), Some("/matrix"));
    assert_eq!(refs[0].attribute_name(), Some("note"));
    // The path and the name together reach the attribute the reference means.
    // `H5File::dataset` names datasets relative to the root, while a reference
    // reports the absolute path, so the leading separator comes off here.
    let target = file
        .dataset(refs[0].path().unwrap().trim_start_matches('/'))
        .unwrap();
    let note = target.attr(refs[0].attribute_name().unwrap()).unwrap();
    assert_eq!(note.read_numeric_as::<i32>().unwrap(), vec![7, 8, 9]);
    drop(file);
    cleanup(&path);
}
