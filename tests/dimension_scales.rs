//! Dimension scales — `H5DSset_scale` / `H5DSattach_scale` through
//! `H5Dataset::set_scale` and `H5Dataset::attach_scale`.
//!
//! What the attributes hold is cross-checked against h5py in
//! `h5py_cross_validation.rs`; these tests cover the preconditions upstream
//! enforces and what this crate's own reader sees.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{DatatypeMessage, H5File};

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_dimension_scales_{}_{}_{}",
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

/// `set_scale` writes `CLASS` 16 bytes wide and null-terminated — the only
/// form `H5DSis_scale` accepts for a fixed-length string — and `NAME` as
/// `strlen + 1` bytes; `attach_scale` writes the two lists with the types
/// `H5DSattach_scale` gives them.
#[test]
fn scale_attributes_have_the_h5ds_types() {
    let path = unique_tmp("types");
    {
        let file = H5File::create(&path).unwrap();
        let data = file
            .new_dataset::<u16>()
            .shape([2, 3])
            .create("data")
            .unwrap();
        data.write_raw(&[0u16; 6]).unwrap();
        let x = file.new_dataset::<f32>().shape([3]).create("x").unwrap();
        x.write_raw(&[0.0f32; 3]).unwrap();
        x.set_scale(Some("x axis")).unwrap();
        data.attach_scale(1, &x).unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    let x = file.dataset("x").unwrap();
    let class = x.attr("CLASS").unwrap();
    assert_eq!(class.datatype().unwrap(), DatatypeMessage::fixed_string(16));
    assert_eq!(class.read_string().unwrap(), "DIMENSION_SCALE");
    assert_eq!(class.read_raw().unwrap().len(), 16);
    let name = x.attr("NAME").unwrap();
    assert_eq!(name.datatype().unwrap(), DatatypeMessage::fixed_string(7));
    assert_eq!(name.read_string().unwrap(), "x axis");
    let reflist = x.attr("REFERENCE_LIST").unwrap();
    let DatatypeMessage::Compound { size, members } = reflist.datatype().unwrap() else {
        panic!("REFERENCE_LIST is not a compound");
    };
    assert_eq!(size, 16);
    assert_eq!(
        members
            .iter()
            .map(|m| (m.name.as_str(), m.offset))
            .collect::<Vec<_>>(),
        vec![("dataset", 0), ("dimension", 8)]
    );
    assert_eq!(members[1].datatype, DatatypeMessage::u32_type());
    let raw = reflist.read_raw().unwrap();
    assert_eq!(raw.len(), 16);
    assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), 1);

    let data = file.dataset("data").unwrap();
    let dimlist = data.attr("DIMENSION_LIST").unwrap();
    let DatatypeMessage::VarLenSequence { base } = dimlist.datatype().unwrap() else {
        panic!("DIMENSION_LIST is not a vlen sequence");
    };
    assert!(
        matches!(*base, DatatypeMessage::Reference { size: 8, .. }),
        "{base}"
    );
    // One 16-byte vlen reference per axis.
    assert_eq!(dimlist.read_raw().unwrap().len(), 32);
    assert!(x.attr("DIMENSION_LIST").is_err());
    assert!(data.attr("CLASS").is_err());
    cleanup(&path);
}

/// The preconditions `H5DSattach_scale` and `H5DSset_scale` enforce.
#[test]
fn attach_scale_refuses_what_upstream_refuses() {
    let path = unique_tmp("refuse");
    let file = H5File::create(&path).unwrap();
    let data = file
        .new_dataset::<u16>()
        .shape([2, 3])
        .create("data")
        .unwrap();
    let x = file.new_dataset::<f32>().shape([3]).create("x").unwrap();
    let y = file.new_dataset::<f32>().shape([2]).create("y").unwrap();
    let scalar = file
        .new_dataset::<f32>()
        .shape([])
        .create("scalar")
        .unwrap();
    let image = file
        .new_dataset::<u8>()
        .shape([2, 2])
        .create("image")
        .unwrap();

    // Itself.
    data.attach_scale(0, &data).unwrap_err();
    // Beyond the rank; a scalar counts as rank 1.
    data.attach_scale(2, &x).unwrap_err();
    scalar.attach_scale(1, &x).unwrap_err();
    scalar.attach_scale(0, &x).unwrap();
    // A dataset with scales cannot be a scale, and a scale cannot have scales.
    data.attach_scale(1, &x).unwrap();
    y.attach_scale(0, &data).unwrap_err();
    data.set_scale(None).unwrap_err();
    x.attach_scale(0, &y).unwrap_err();
    // A reserved CLASS.
    let class = image
        .new_attr::<rust_hdf5::types::VarLenUnicode>()
        .shape(())
        .create("CLASS")
        .unwrap();
    class.write_string("IMAGE").unwrap();
    image.attach_scale(0, &y).unwrap_err();
    // Another file's dataset.
    let other_path = unique_tmp("refuse_other");
    let other = H5File::create(&other_path).unwrap();
    let z = other.new_dataset::<f32>().shape([2]).create("z").unwrap();
    data.attach_scale(0, &z).unwrap_err();
    other.close().unwrap();
    cleanup(&other_path);

    file.close().unwrap();
    // Read-mode handles.
    let file = H5File::open(&path).unwrap();
    let data = file.dataset("data").unwrap();
    let y = file.dataset("y").unwrap();
    data.attach_scale(0, &y).unwrap_err();
    y.set_scale(None).unwrap_err();
    cleanup(&path);
}

/// Attaching a scale already on the axis changes neither list, and the
/// attributes survive a reopen that rewrites both headers.
#[test]
fn repeated_attach_is_a_no_op() {
    let path = unique_tmp("idempotent");
    {
        let file = H5File::create(&path).unwrap();
        let data = file
            .new_dataset::<u16>()
            .shape([2, 3])
            .create("data")
            .unwrap();
        let x = file.new_dataset::<f32>().shape([3]).create("x").unwrap();
        data.attach_scale(1, &x).unwrap();
        data.attach_scale(1, &x).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        let data = file.dataset_writer("data").unwrap();
        let x = file.dataset_writer("x").unwrap();
        data.attach_scale(1, &x).unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    let raw = file
        .dataset("x")
        .unwrap()
        .attr("REFERENCE_LIST")
        .unwrap()
        .read_raw()
        .unwrap();
    assert_eq!(raw.len(), 16, "one (dataset, axis) entry");
    cleanup(&path);
}
