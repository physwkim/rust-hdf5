//! A chunked dataset indexed by a version-1 B-tree survives a reopen as a
//! *dataset*, not as a block of bytes.
//!
//! The version-3 data layout message is the only chunked layout a classic
//! file can hold and the one the v1.8 row of `H5O_layout_ver_bounds` gives, so
//! it is what h5py writes at `libver='v108'` and what every chunked dataset in
//! a libhdf5 1.6-era file is. `rebuild_dataset` had no arm for it: the reopen
//! walk saw a layout it could not reconstruct and preserved the object by its
//! bytes, which kept the data safe but left the dataset out of
//! `dataset_names`, unopenable for writing and unextendable.
//!
//! It rebuilds now, into the same `BtreeV1DatasetInfo` a chunked dataset
//! *created* in this format gets — so one set of machinery drives both, and a
//! reopened tree grows through the same bulk load and block pool as a fresh
//! one. What these cases check is that the dataset is writable through the
//! reopen and that libhdf5 still reads every element afterwards.

use rust_hdf5::{ChunkIndex, H5File, LibverBound};

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
            eprintln!("skipping h5py cross-check: {candidate} not present");
            None
        }
    })
    .as_deref()
}

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_btree_v1_reopen_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &Path) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The version byte after the 8-byte signature, in either superblock image.
fn superblock_version(path: &Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let at = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("no HDF5 signature");
    bytes[at + 8]
}

/// Read `path` back through libhdf5 and run `body`, with `f` bound to the open
/// file. A failed assert inside the script fails the test.
fn read_with_h5py(py: &str, path: &Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nwith h5py.File(r'{}', 'r') as f:\n{}",
        path.display(),
        body
    );
    let status = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python");
    assert!(status.success(), "python readback failed");
}

/// One fixed-shape chunked dataset and one unlimited one, created under
/// `libver` — the two shapes whose reopened behaviour differs, a rewrite of an
/// existing chunk versus a new key past the tree's right bound.
fn create_chunked(path: &Path, libver: LibverBound) {
    let file = H5File::options().libver(libver).create(path).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize, 4])
        .chunk(&[2, 2])
        .create("grid")
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .max_shape(&[None])
        .chunk(&[4])
        .create("growing")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// Reopen, and put both datasets through the operations a preserved object
/// could not take: a listing, a write into an existing chunk, and a growth
/// past the extent.
fn reopen_write_and_extend(path: &Path) {
    let file = H5File::open_rw(path).unwrap();
    let names = file.dataset_names();
    for want in ["grid", "growing"] {
        assert!(
            names.contains(&want.to_string()),
            "a rebuilt dataset is in the registry, so it is listed: {names:?}"
        );
    }

    // Overwriting a chunk the tree already keys replaces its record rather
    // than adding one beside it — the path that needs the reopened records to
    // be in key order.
    let grid = file.dataset_writer("grid").unwrap();
    grid.write_raw(&(100..116i32).collect::<Vec<_>>()).unwrap();

    // A key past the tree's right bound, which is where a bulk load that
    // dropped the reopened records would leave the old chunks unreachable.
    let growing = file.dataset_writer("growing").unwrap();
    growing.append(&(8..16i32).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

/// What both files must say afterwards, through this crate and through
/// libhdf5: the overwritten grid, the grown vector, and the superblock version
/// the file started at.
fn assert_survived(path: &Path, expect_sb: u8) {
    assert_eq!(
        superblock_version(path),
        expect_sb,
        "the reopen must not change the superblock version"
    );
    {
        let file = H5File::open(path).unwrap();
        assert_eq!(
            file.dataset("grid").unwrap().read_raw::<i32>().unwrap(),
            (100..116).collect::<Vec<i32>>()
        );
        assert_eq!(
            file.dataset("growing").unwrap().read_raw::<i32>().unwrap(),
            (0..16).collect::<Vec<i32>>()
        );
    }
    let Some(py) = python() else { return };
    read_with_h5py(
        py,
        path,
        "    assert f['grid'].chunks == (2, 2), f['grid'].chunks\n\
     \x20   assert f['grid'][...].tolist() == \
     [[100, 101, 102, 103], [104, 105, 106, 107], \
     [108, 109, 110, 111], [112, 113, 114, 115]], f['grid'][...].tolist()\n\
     \x20   assert f['growing'].shape == (16,), f['growing'].shape\n\
     \x20   assert f['growing'].maxshape == (None,), f['growing'].maxshape\n\
     \x20   assert list(f['growing'][...]) == list(range(16)), list(f['growing'][...])\n",
    );
}

/// The v1.8 generation: a version-2 superblock whose chunked datasets are all
/// version-3 layouts, which is the shape h5py writes at `libver='v108'`.
#[test]
fn a_v18_chunked_dataset_is_writable_through_a_reopen() {
    let path = tmp("v18");
    create_chunked(&path, LibverBound::V18);
    assert_eq!(superblock_version(&path), 2);
    reopen_write_and_extend(&path);
    assert_survived(&path, 2);
    cleanup(&path);
}

/// The classic generation, reached through the symbol-table root group rather
/// than through link messages — the same index either way, since a version-0
/// superblock has no other.
#[test]
fn a_classic_chunked_dataset_is_writable_through_a_reopen() {
    let path = tmp("classic");
    create_chunked(&path, LibverBound::Earliest);
    assert_eq!(superblock_version(&path), 0);
    reopen_write_and_extend(&path);
    assert_survived(&path, 0);
    cleanup(&path);
}

/// The second-generation reopen: a dataset that was itself *appended* on a
/// reopen, reopened again.
///
/// This case exists only because the superblock version now floors the append
/// — before that, appending a chunked dataset to a version-2 file produced a
/// v1.10 index, so the file this reopens could not be written in the first
/// place. Getting the version-3 layout right on the first reopen is what makes
/// a second one meet it.
#[test]
fn a_chunked_dataset_appended_on_a_reopen_survives_a_second_reopen() {
    let path = tmp("second_gen");
    {
        let file = H5File::options()
            .libver(LibverBound::V18)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([8usize])
            .create("data")
            .unwrap()
            .write_raw(&(0..8i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }
    // First reopen: the append whose index the file's own generation decides.
    {
        let file = H5File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([8usize])
            .max_shape(&[None])
            .chunk(&[4])
            .create("appended")
            .unwrap()
            .write_raw(&(0..8i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }
    assert_eq!(superblock_version(&path), 2);
    {
        // The index the append picked, through the public accessor: a v1.8
        // file's chunked dataset is a version-1 B-tree, and it is the rebuild
        // of exactly this that the second reopen below exercises.
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("appended").unwrap().chunk_index().unwrap(),
            Some(ChunkIndex::BtreeV1)
        );
    }

    // Second reopen: that dataset is now one the walk has to rebuild.
    {
        let file = H5File::open_rw(&path).unwrap();
        let names = file.dataset_names();
        assert!(
            names.contains(&"appended".to_string()),
            "the dataset the first reopen appended must come back as a dataset: {names:?}"
        );
        file.dataset_writer("appended")
            .unwrap()
            .append(&(8..16i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }

    assert_eq!(
        superblock_version(&path),
        2,
        "neither reopen re-decides the superblock version"
    );
    {
        let file = H5File::open(&path).unwrap();
        let appended = file.dataset("appended").unwrap();
        assert_eq!(
            appended.read_raw::<i32>().unwrap(),
            (0..16).collect::<Vec<i32>>()
        );
        // The rebuild put the tree back as a version-1 B-tree rather than
        // re-indexing it under this session's own defaults: the accessor reads
        // the index out of the re-serialized layout message, so a rebuild that
        // silently promoted the index would show here.
        assert_eq!(
            appended.chunk_index().unwrap(),
            Some(ChunkIndex::BtreeV1),
            "a second-generation reopen must keep the version-1 B-tree index"
        );
        assert_eq!(
            file.dataset("data").unwrap().read_raw::<i32>().unwrap(),
            (0..8).collect::<Vec<i32>>()
        );
    }
    if let Some(py) = python() {
        read_with_h5py(
            py,
            &path,
            "    assert f['appended'].chunks == (4,), f['appended'].chunks\n\
         \x20   assert list(f['appended'][...]) == list(range(16)), \
         list(f['appended'][...])\n\
         \x20   assert list(f['data'][...]) == list(range(8))\n",
        );
    }
    cleanup(&path);
}
