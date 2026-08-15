//! `header_attr_count()` — h5py's `h5o.get_info(...).num_attrs` — on
//! `H5Group`, `H5Dataset`, the root group, and `H5NamedDatatype`.
//!
//! Every rust-writable case below carries a known, non-uniform attribute
//! count and is checked in both compact and dense storage, since
//! `header_count`'s whole premise is that the object-header count equals the
//! plain listing's length regardless of which one the object is in. The
//! named-datatype case is read-only in this crate's writer, so it goes
//! through an h5py fixture like the crate's other read-parity tests.

use std::path::PathBuf;

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_header_attr_count_accessor_{}_{}_{}.h5",
        label,
        std::process::id(),
        n
    ))
}

const COMPACT_COUNT: usize = 3;
const DENSE_COUNT: usize = 9;

#[test]
fn group_header_attr_count_matches_the_listing_when_compact() {
    let path = unique_tmp("group_compact");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..COMPACT_COUNT {
        grp.set_attr_numeric(&format!("a{i}"), &(i as i32)).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.header_attr_count().unwrap(), COMPACT_COUNT as u64);
    assert_eq!(grp.attr_names().unwrap().len(), COMPACT_COUNT);
    std::fs::remove_file(&path).ok();
}

#[test]
fn group_header_attr_count_matches_the_listing_when_dense() {
    let path = unique_tmp("group_dense");
    let file = H5File::create(&path).unwrap();
    let grp = file.root_group().create_group("g").unwrap();
    for i in 0..DENSE_COUNT {
        grp.set_attr_numeric(&format!("a{i}"), &(i as i32)).unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("g").unwrap();
    assert_eq!(grp.header_attr_count().unwrap(), DENSE_COUNT as u64);
    assert_eq!(grp.attr_names().unwrap().len(), DENSE_COUNT);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dataset_header_attr_count_matches_the_listing_when_compact() {
    let path = unique_tmp("dataset_compact");
    let file = H5File::create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
    for i in 0..COMPACT_COUNT {
        ds.new_attr::<i32>()
            .shape(())
            .create(&format!("a{i}"))
            .unwrap()
            .write_numeric(&(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.header_attr_count().unwrap(), COMPACT_COUNT as u64);
    assert_eq!(ds.attr_names().unwrap().len(), COMPACT_COUNT);
    std::fs::remove_file(&path).ok();
}

#[test]
fn dataset_header_attr_count_matches_the_listing_when_dense() {
    let path = unique_tmp("dataset_dense");
    let file = H5File::create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([1]).create("data").unwrap();
    for i in 0..DENSE_COUNT {
        ds.new_attr::<i32>()
            .shape(())
            .create(&format!("a{i}"))
            .unwrap()
            .write_numeric(&(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.header_attr_count().unwrap(), DENSE_COUNT as u64);
    assert_eq!(ds.attr_names().unwrap().len(), DENSE_COUNT);
    std::fs::remove_file(&path).ok();
}

#[test]
fn root_header_attr_count_matches_the_listing() {
    let path = unique_tmp("root");
    let file = H5File::create(&path).unwrap();
    for i in 0..COMPACT_COUNT {
        file.set_attr_numeric(&format!("a{i}"), &(i as i32))
            .unwrap();
    }
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.root_group().header_attr_count().unwrap(),
        COMPACT_COUNT as u64
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn empty_group_header_attr_count_is_zero() {
    let path = unique_tmp("empty_group");
    let file = H5File::create(&path).unwrap();
    file.root_group().create_group("empty").unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let grp = file.root_group().group("empty").unwrap();
    assert_eq!(grp.header_attr_count().unwrap(), 0);
    std::fs::remove_file(&path).ok();
}

// -- named-datatype: read-only in this crate's writer, so h5py builds the
// fixture, matching the idiom `tests/catalog_read_parity.rs` uses for the
// same reason.

const PINNED_PYTHON: &str = "/home/stevek/micromamba/envs/tomo/bin/python";

fn python() -> Option<&'static str> {
    static PY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PY.get_or_init(|| {
        let candidate = std::env::var("RUST_HDF5_TEST_PYTHON")
            .or_else(|_| std::env::var("RUST_HDF5_ORACLE_PYTHON"))
            .unwrap_or_else(|_| PINNED_PYTHON.to_string());
        if std::path::Path::new(&candidate).exists() {
            Some(candidate)
        } else {
            eprintln!("skipping h5py fixture test: {candidate} not present");
            None
        }
    })
    .as_deref()
}

fn h5py_write(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nPATH = r'{}'\n{}\n",
        path.display(),
        body
    );
    let status = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python");
    assert!(
        status.success(),
        "h5py fixture generation failed for {}",
        path.display()
    );
}

#[test]
fn named_datatype_header_attr_count_matches_the_listing() {
    let Some(py) = python() else { return };
    let path = unique_tmp("named_datatype");
    h5py_write(
        py,
        &path,
        "with h5py.File(PATH, 'w') as f:\n\
         \x20   f['t'] = np.dtype('<f8')\n\
         \x20   f['t'].attrs['units'] = 'mm'\n\
         \x20   f['t'].attrs['scale'] = np.int32(7)\n\
         \x20   f['t'].attrs['offset'] = np.int32(3)\n",
    );

    let file = H5File::open(&path).unwrap();
    let t = file.named_datatype("t").unwrap();
    assert_eq!(t.header_attr_count().unwrap(), 3);
    assert_eq!(t.attr_names().unwrap().len(), 3);
    std::fs::remove_file(&path).ok();
}
