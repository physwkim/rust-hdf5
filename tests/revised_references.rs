//! Reading the 1.12 reference kinds — `H5R_OBJECT2`, `H5R_DATASET_REGION2`
//! and `H5R_ATTR`, all stored as `H5T_STD_REF`.
//!
//! The fixture is libhdf5 1.14.6 output built by `tests/fixtures/gen_revised_refs.c`;
//! h5py 3.x cannot stand in for it, as it raises "Unknown reference type" on
//! an `H5T_STD_REF` dataset even for reading. It holds a 4x6 `matrix` with an
//! attribute `note`, a group `grp`, and one dataset per reference kind:
//! `objrefs` names `/matrix` and `/grp`, `regrefs` selects the hyperslab
//! (1,2)-(2,4) and the points (0,1) and (3,5) of `/matrix`, and `attrrefs`
//! names `/matrix`'s `note`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{H5File, PointSelection, Reference, Selection};

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
