//! `H5F_prefix_open_file`'s search order (H5Fint.c:826-1025), one test per
//! step that can decide it, for both kinds of cross-file name.
//!
//! Every expected value below was measured first, against libhdf5 1.14.6
//! (h5py 3.15.1) and 2.0.0 (h5py 3.16.0), by running the same arrangement
//! through `h5d.open(..., dapl)` with `H5Pset_virtual_prefix` and
//! `h5o.open(..., lapl)` with `H5Pset_elink_prefix`; both libraries answered
//! identically. The fixtures here are written by this crate rather than by
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

/// `home/master.h5` whose external link `ext` names the bare `src.h5`.
fn linking_file(root: &Path) -> PathBuf {
    let path = root.join("home").join("master.h5");
    let file = H5File::create(&path).unwrap();
    file.create_external_link("ext", "src.h5", "/data").unwrap();
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

/// What the external link reads back under `prefix`, or `None` when the
/// target file is not found at all.
fn link_reads(path: &Path, prefix: Option<&str>) -> Option<i32> {
    let mut options = H5File::options();
    if let Some(p) = prefix {
        options = options.elink_prefix(p);
    }
    let file = options.open(path).unwrap();
    let data = file.dataset("ext").ok()?.read_raw::<i32>().unwrap();
    assert!(data.iter().all(|&v| v == data[0]), "{data:?}");
    Some(data[0])
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

/// `H5Pset_elink_prefix` occupies the same third step for an external link
/// (H5Lexternal.c:210-215 into H5Fint.c:938-950): after `HDF5_EXT_PREFIX`,
/// before the linking file's own directory.
///
/// Measured (both references): without a prefix the open of the linked
/// object fails outright — an external link is `H5F_prefix_open_file`'s
/// `try = false` caller (H5Lexternal.c:215) — and with the property it reads
/// 1, still 1 with a neighbouring `home/src.h5` present.
#[test]
fn an_elink_prefix_is_searched_before_the_linking_file_s_own_directory() {
    let root = root("elink_prop");
    sources(&root);
    let master = linking_file(&root);

    assert_eq!(
        link_reads(&master, None),
        None,
        "nothing names the target, so the link does not resolve"
    );
    assert_eq!(link_reads(&master, Some(&dir_str(&root, "far"))), Some(1));

    neighbour(&root);
    assert_eq!(
        link_reads(&master, Some(&dir_str(&root, "far"))),
        Some(1),
        "the property is searched before the linking file's own directory"
    );
    assert_eq!(
        link_reads(&master, None),
        Some(9),
        "with no property the neighbouring file answers"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// The external-link prefix is the half of this pair that its environment
/// variable does *not* shadow: `H5L__extern_traverse` peeks
/// `H5L_ACS_ELINK_PREFIX_NAME` and passes it straight to the search
/// (H5Lexternal.c:210-215) with no `H5D__build_file_prefix` in between, so
/// both steps run, in the order `H5F_prefix_open_file` lists them.
///
/// Measured (both references), against the identical arrangement that reads
/// the fill value for a virtual source: `HDF5_EXT_PREFIX` naming a directory
/// that holds no target and the property naming one that does reads 1.
#[test]
fn the_elink_prefix_environment_variable_does_not_shadow_the_property() {
    let root = root("elink_shadow");
    sources(&root);
    let master = linking_file(&root);
    let far = dir_str(&root, "far");

    std::env::set_var("HDF5_EXT_PREFIX", dir_str(&root, "empty"));
    assert_eq!(
        link_reads(&master, Some(&far)),
        Some(1),
        "an environment prefix that resolves nowhere leaves the property to answer"
    );

    std::env::set_var("HDF5_EXT_PREFIX", &far);
    assert_eq!(
        link_reads(&master, Some(&dir_str(&root, "other"))),
        Some(1),
        "the environment variable is searched first"
    );
    std::env::remove_var("HDF5_EXT_PREFIX");
    std::fs::remove_dir_all(&root).ok();
}

/// And it is the half that gets no `${ORIGIN}` expansion, for the same
/// reason: nothing rewrites it between `H5P_peek` and `H5F__build_name`.
///
/// Measured (both references): `${ORIGIN}/../far` on an external link reads
/// the neighbouring 9, not `far`'s 1 — the token stayed a directory name,
/// the step missed, and the next one answered. The virtual-source twin of
/// this case reads 1.
#[test]
fn an_elink_prefix_leaves_origin_alone() {
    let root = root("elink_origin");
    sources(&root);
    neighbour(&root);
    let master = linking_file(&root);

    assert_eq!(
        link_reads(&master, Some("${ORIGIN}/../far")),
        Some(9),
        "${{ORIGIN}} is not expanded for an external link"
    );
    assert_eq!(
        vds_reads(
            &virtual_file(&root),
            DatasetAccess::new().virtual_prefix("${ORIGIN}/../far")
        ),
        1,
        "the same prefix on a virtual source is expanded"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// The prefix reaches every hop of a chain of external links, the way the
/// link-access property list does upstream: `H5L__extern_traverse` opens the
/// linked object through the same API context, so the next link it crosses
/// peeks the same property.
///
/// Measured against libhdf5 1.14.6: with `home/b.h5` reachable as a
/// neighbour and `b.h5`'s own link to `c.h5` resolvable only under the
/// prefix, the two-hop read succeeds.
#[test]
fn an_elink_prefix_reaches_the_second_hop_of_a_chain() {
    let root = root("elink_chain");
    source(&root.join("far"), 1);
    // home/b.h5 -> "src.h5", which only `far/` holds.
    let b = H5File::create(root.join("home").join("b.h5")).unwrap();
    b.create_external_link("next", "src.h5", "/data").unwrap();
    b.close().unwrap();
    let a = H5File::create(root.join("home").join("a.h5")).unwrap();
    a.create_external_link("hop", "b.h5", "/next").unwrap();
    a.close().unwrap();

    let file = H5File::options()
        .elink_prefix(dir_str(&root, "far"))
        .open(root.join("home").join("a.h5"))
        .unwrap();
    assert_eq!(
        file.dataset("hop").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 4],
        "the prefix given at the first hop must reach the second"
    );
    drop(file);
    std::fs::remove_dir_all(&root).ok();
}

/// The prefix is a read-mode property: nothing on the write side traverses an
/// external link, so an open that produces a writer refuses it rather than
/// accepting a setting it would never consult.
#[test]
fn a_write_mode_open_refuses_an_elink_prefix() {
    let root = root("elink_write");
    let path = root.join("home").join("w.h5");
    assert!(H5File::options()
        .elink_prefix(dir_str(&root, "far"))
        .create(&path)
        .is_err());
    H5File::create(&path).unwrap().close().unwrap();
    assert!(H5File::options()
        .elink_prefix(dir_str(&root, "far"))
        .open_rw(&path)
        .is_err());
    assert!(H5File::options()
        .elink_prefix(dir_str(&root, "far"))
        .open(&path)
        .is_ok());
    std::fs::remove_dir_all(&root).ok();
}
