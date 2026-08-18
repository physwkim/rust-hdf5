//! `H5F_prefix_open_file`'s search order (H5Fint.c:826-1025), one test per
//! step that can decide it, for a virtual dataset's source file names.
//!
//! Every expected value below was measured first, against libhdf5 1.14.6
//! (h5py 3.15.1) and 2.0.0 (h5py 3.16.0), by running the same arrangement
//! through `h5d.open(..., dapl)` with `H5Pset_virtual_prefix`; both
//! libraries answered identically. The fixtures here are written by this crate rather than by
//! h5py because what is under test is the *search*, not the bytes — the file
//! contents these read back are already h5py-validated elsewhere.
//!
//! The layout is the same in every test, with the target file named by the
//! bare relative name `src.h5` from a file in `home/`:
//!
//! ```text
//! far/src.h5     [1, 1, 1, 1]   named by a prefix
//! other/src.h5   [7, 7, 7, 7]   named by the other prefix
//! empty/         nothing        a prefix that resolves nowhere
//! home/src.h5    [9, 9, 9, 9]   the neighbouring-file step, when present
//! ```

use std::path::{Path, PathBuf};

use rust_hdf5::{DatasetAccess, H5File, Selection};

/// Per-test unique root; nextest gives each test its own process, so the
/// environment variables set below cannot reach another test.
fn root(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_prefix_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    for sub in ["home", "far", "other", "empty"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    dir
}

/// `src.h5` in `dir`, four elements all `value`.
fn source(dir: &Path, value: i32) {
    let file = H5File::create(dir.join("src.h5")).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize])
        .create("data")
        .unwrap()
        .write_raw(&[value; 4])
        .unwrap();
    file.close().unwrap();
}

/// The three sources every test starts from; the neighbour in `home/` is
/// added per-case by [`neighbour`].
fn sources(root: &Path) {
    source(&root.join("far"), 1);
    source(&root.join("other"), 7);
}

fn neighbour(root: &Path) {
    source(&root.join("home"), 9);
}

/// `home/vds.h5` holding a virtual dataset `v` over the bare name `src.h5`,
/// filled with -1 where no source is found.
fn virtual_file(root: &Path) -> PathBuf {
    let path = root.join("home").join("vds.h5");
    let file = H5File::create(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize])
        .fill_value(-1i32)
        .virtual_mapping(Selection::All, "src.h5", "data", Selection::All)
        .create("v")
        .unwrap();
    file.close().unwrap();
    path
}

/// What the virtual dataset reads back under `access`; every element is the
/// same, so one value names which file won the search.
fn vds_reads(path: &Path, access: DatasetAccess) -> i32 {
    let file = H5File::open(path).unwrap();
    let ds = file.dataset_with("v", access).unwrap();
    let data = ds.read_raw::<i32>().unwrap();
    assert_eq!(data.len(), 4);
    assert!(data.iter().all(|&v| v == data[0]), "{data:?}");
    data[0]
}

fn dir_str(root: &Path, sub: &str) -> String {
    root.join(sub).display().to_string()
}

/// `H5Pset_virtual_prefix` is the third step of the search — after the
/// environment variable and before the virtual file's own directory
/// (H5Fint.c:889-950).
///
/// Measured (both references): no prefix reads the fill value; the property
/// naming `far` reads 1; the property still wins with the neighbouring
/// `home/src.h5` present, and without a property that neighbour is what
/// answers.
#[test]
fn a_virtual_prefix_is_searched_before_the_virtual_file_s_own_directory() {
    let root = root("vds_prop");
    sources(&root);
    let vds = virtual_file(&root);

    assert_eq!(
        vds_reads(&vds, DatasetAccess::new()),
        -1,
        "nothing names the source, so every element is the fill value"
    );
    assert_eq!(
        vds_reads(
            &vds,
            DatasetAccess::new().virtual_prefix(dir_str(&root, "far"))
        ),
        1,
        "the property should resolve the source"
    );

    neighbour(&root);
    assert_eq!(
        vds_reads(
            &vds,
            DatasetAccess::new().virtual_prefix(dir_str(&root, "far"))
        ),
        1,
        "the property is searched before the virtual file's own directory"
    );
    assert_eq!(
        vds_reads(&vds, DatasetAccess::new()),
        9,
        "with no property the neighbouring file answers"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// `HDF5_VDS_PREFIX` does not merely *precede* `H5Pset_virtual_prefix` — it
/// replaces it. `H5D__build_file_prefix` reads the environment variable first
/// and only falls back to the property when it is unset or empty
/// (H5Dint.c:1077-1082), so the property never reaches
/// `H5F_prefix_open_file` at all while the variable is set.
///
/// Measured (both references): with `HDF5_VDS_PREFIX` naming a directory that
/// holds no source and the property naming one that does, the read is the
/// fill value — the property is not tried. With the variable naming the
/// directory that has the source, it answers whatever the property says.
#[test]
fn the_virtual_prefix_environment_variable_shadows_the_property() {
    let root = root("vds_shadow");
    sources(&root);
    let vds = virtual_file(&root);
    let far = DatasetAccess::new().virtual_prefix(dir_str(&root, "far"));

    std::env::set_var("HDF5_VDS_PREFIX", dir_str(&root, "empty"));
    assert_eq!(
        vds_reads(&vds, far.clone()),
        -1,
        "an environment prefix that resolves nowhere still shadows the property"
    );

    std::env::set_var("HDF5_VDS_PREFIX", dir_str(&root, "far"));
    assert_eq!(
        vds_reads(
            &vds,
            DatasetAccess::new().virtual_prefix(dir_str(&root, "other"))
        ),
        1,
        "the environment variable answers, not the property"
    );

    std::env::remove_var("HDF5_VDS_PREFIX");
    assert_eq!(vds_reads(&vds, far), 1, "unset, the property is reached");
    std::fs::remove_dir_all(&root).ok();
}

/// A `${ORIGIN}` at the front of the virtual prefix stands for the directory
/// holding the virtual dataset's own file (H5Dint.c:1105-1113), and a `"."`
/// or an empty prefix means no prefix at all (:1096-1100) — the property gets
/// the same treatment the environment variable does, because both arrive
/// through `H5D__build_file_prefix`.
///
/// Measured (both references): `${ORIGIN}/../far` reads 1 with the
/// neighbouring `home/src.h5` present, so the token was expanded rather than
/// taken as a directory name; `"."` reads that neighbour's 9, because the
/// step is skipped and the virtual file's own directory answers.
#[test]
fn a_virtual_prefix_expands_origin_and_treats_dot_as_no_prefix() {
    let root = root("vds_origin");
    sources(&root);
    neighbour(&root);
    let vds = virtual_file(&root);

    assert_eq!(
        vds_reads(
            &vds,
            DatasetAccess::new().virtual_prefix("${ORIGIN}/../far")
        ),
        1,
        "${{ORIGIN}} should expand to the virtual file's directory"
    );
    assert_eq!(
        vds_reads(&vds, DatasetAccess::new().virtual_prefix("${ORIGIN}")),
        9,
        "${{ORIGIN}} alone is that directory"
    );
    assert_eq!(
        vds_reads(&vds, DatasetAccess::new().virtual_prefix(".")),
        9,
        "\".\" is no prefix, so the neighbouring-file step answers"
    );
    std::fs::remove_dir_all(&root).ok();
}
