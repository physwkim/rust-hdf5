//! Where a file named from inside an HDF5 file is looked for, one test per
//! step that can decide it, for each of the three kinds of name: a virtual
//! dataset's source, an external link's target, and an external file list's
//! raw data file.
//!
//! The first two are searched — `H5F_prefix_open_file` walks six steps in
//! order (H5Fint.c:826-1025) and takes the first that opens. The third is
//! not: `H5D__efl_read` joins the prefix to the stored name and opens that
//! one path (H5Defl.c:315-317), so a raw data file that is not where the
//! prefix says has nowhere else to be found.
//!
//! Every expected value below was measured first, against libhdf5 1.14.6
//! (h5py 3.15.1) and 2.0.0 (h5py 3.16.0), by running the same arrangement
//! through `h5d.open(..., dapl)` with `H5Pset_virtual_prefix` or
//! `H5Pset_efile_prefix` and `h5o.open(..., lapl)` with
//! `H5Pset_elink_prefix`; both libraries answered identically. The fixtures
//! here are written by this crate rather than by h5py because what is under
//! test is where the name resolves, not the bytes — the file contents these
//! read back are already h5py-validated elsewhere.
//!
//! The layout is the same in every test, with the target named by a bare
//! relative name — `src.h5` for the two searched kinds, `raw.bin` for the
//! external file list — from a file in `home/`:
//!
//! ```text
//! far/     [1, 1, 1, 1]   named by a prefix
//! other/   [7, 7, 7, 7]   named by the other prefix
//! empty/   nothing        a prefix that resolves nowhere
//! home/    [9, 9, 9, 9]   next to the HDF5 file, when present
//! cwd/     [5, 5, 5, 5]   the process's current directory, when set there
//! ```

use std::path::{Path, PathBuf, MAIN_SEPARATOR as MAIN};

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
    for sub in ["home", "far", "other", "empty", "cwd"] {
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

// ---------------------------------------------------------------------------
// External file list raw data files
// ---------------------------------------------------------------------------

/// `raw.bin` in `dir`, four `i32` all `value`, laid out the way this crate
/// stores a native-order `i32` dataset.
fn raw(dir: &Path, value: i32) {
    let mut bytes = Vec::with_capacity(16);
    for _ in 0..4 {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    std::fs::write(dir.join("raw.bin"), bytes).unwrap();
}

/// `home/ext.h5` whose dataset `d` keeps its four elements in the bare name
/// `raw.bin` rather than in the HDF5 file itself.
fn external_file(root: &Path) -> PathBuf {
    let path = root.join("home").join("ext.h5");
    let file = H5File::create(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize])
        .external(&[("raw.bin", 0, 16)])
        .create("d")
        .unwrap();
    file.close().unwrap();
    path
}

/// What the external-file-list dataset reads back under `access`, or `None`
/// when its raw data file is not where the prefix in force says it is.
fn efl_reads(path: &Path, access: DatasetAccess) -> Option<i32> {
    let file = H5File::open(path).unwrap();
    let data = file
        .dataset_with("d", access)
        .unwrap()
        .read_raw::<i32>()
        .ok()?;
    assert_eq!(data.len(), 4);
    assert!(data.iter().all(|&v| v == data[0]), "{data:?}");
    Some(data[0])
}

/// `H5Pset_efile_prefix` is not a step of a search — it is the whole answer.
/// `H5D__efl_read` builds one path with `H5_combine_path` and opens it
/// (H5Defl.c:315-317), so the neighbouring-file step that saves a virtual
/// source or a link target does not exist here.
///
/// Measured (both references): with `home/raw.bin` sitting right next to the
/// HDF5 file and no prefix named, the read fails outright; the property
/// naming `far` reads 1, and still 1 with that neighbour present.
#[test]
fn an_efile_prefix_is_the_only_place_a_raw_data_file_is_looked_for() {
    let root = root("efile_prop");
    raw(&root.join("far"), 1);
    let ext = external_file(&root);
    raw(&root.join("home"), 9);

    assert_eq!(
        efl_reads(&ext, DatasetAccess::new()),
        None,
        "a raw data file next to the HDF5 file is not looked for there"
    );
    assert_eq!(
        efl_reads(
            &ext,
            DatasetAccess::new().efile_prefix(dir_str(&root, "far"))
        ),
        Some(1),
        "the property is where the raw data file is looked for"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// `HDF5_EXTFILE_PREFIX` shadows the property exactly as `HDF5_VDS_PREFIX`
/// shadows [`DatasetAccess::virtual_prefix`] — both prefixes are built by the
/// same `H5D__build_file_prefix`, which reads the environment first and falls
/// back to the property only when it is unset or empty (H5Dint.c:1084-1090).
///
/// Measured (both references): with the variable naming a directory holding
/// no raw data file and the property naming one that does, the read fails —
/// the property is not tried.
#[test]
fn the_efile_prefix_environment_variable_shadows_the_property() {
    let root = root("efile_shadow");
    raw(&root.join("far"), 1);
    raw(&root.join("other"), 7);
    let ext = external_file(&root);
    let far = DatasetAccess::new().efile_prefix(dir_str(&root, "far"));

    std::env::set_var("HDF5_EXTFILE_PREFIX", dir_str(&root, "empty"));
    assert_eq!(
        efl_reads(&ext, far.clone()),
        None,
        "an environment prefix that resolves nowhere still shadows the property"
    );

    std::env::set_var("HDF5_EXTFILE_PREFIX", dir_str(&root, "far"));
    assert_eq!(
        efl_reads(
            &ext,
            DatasetAccess::new().efile_prefix(dir_str(&root, "other"))
        ),
        Some(1),
        "the environment variable answers, not the property"
    );

    std::env::remove_var("HDF5_EXTFILE_PREFIX");
    assert_eq!(
        efl_reads(&ext, far),
        Some(1),
        "unset, the property is reached"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// The other half of what `H5D__build_file_prefix` gives both prefixes: a
/// leading `${ORIGIN}` becomes the directory holding the HDF5 file
/// (H5Dint.c:1105-1113), and `"."` or an empty prefix means no prefix at all
/// (:1098-1102) — which for an external file list leaves the stored name to
/// resolve against the process's current directory, not against the HDF5
/// file's own.
///
/// Measured (both references): `${ORIGIN}` reads the neighbour's 9,
/// `${ORIGIN}/../far` reads 1, and `"."` with the current directory moved to
/// `cwd/` reads 5 — as does naming no prefix at all.
#[test]
fn an_efile_prefix_expands_origin_and_treats_dot_as_no_prefix() {
    let root = root("efile_origin");
    raw(&root.join("far"), 1);
    raw(&root.join("cwd"), 5);
    let ext = external_file(&root);
    raw(&root.join("home"), 9);

    assert_eq!(
        efl_reads(&ext, DatasetAccess::new().efile_prefix("${ORIGIN}")),
        Some(9),
        "${{ORIGIN}} is the directory holding the HDF5 file"
    );
    assert_eq!(
        efl_reads(
            &ext,
            DatasetAccess::new().efile_prefix(format!("${{ORIGIN}}{}..{}far", MAIN, MAIN))
        ),
        Some(1),
        "the expansion keeps whatever follows ${{ORIGIN}}"
    );

    // nextest runs each test in its own process, so moving the current
    // directory here cannot reach another test.
    std::env::set_current_dir(root.join("cwd")).unwrap();
    assert_eq!(
        efl_reads(&ext, DatasetAccess::new().efile_prefix(".")),
        Some(5),
        "\".\" is no prefix, so the name resolves against the current directory"
    );
    assert_eq!(
        efl_reads(&ext, DatasetAccess::new()),
        Some(5),
        "naming no prefix resolves the same way"
    );
    std::env::set_current_dir("/").unwrap();
    std::fs::remove_dir_all(&root).ok();
}

/// The one property a joining open may not disagree about. Where a second
/// open of a virtual dataset silently inherits the first open's view, a
/// second open under a different *expanded* external file prefix is refused:
/// `H5D__open_name` compares it against the open dataset's and fails
/// (H5Dint.c:1533-1545).
///
/// Measured (both references): the second open raises "new external file
/// prefix does not match external file prefix of already open dataset" both
/// when it names a different directory and when it names none at all; naming
/// the same one joins normally; and once every handle is closed the next open
/// sets its own prefix and reads the other file.
#[test]
fn a_second_open_may_not_disagree_about_the_efile_prefix() {
    let root = root("efile_join");
    raw(&root.join("far"), 1);
    raw(&root.join("other"), 7);
    let ext = external_file(&root);
    let far = || DatasetAccess::new().efile_prefix(dir_str(&root, "far"));
    let other = || DatasetAccess::new().efile_prefix(dir_str(&root, "other"));

    let file = H5File::open(&ext).unwrap();
    let held = file.dataset_with("d", far()).unwrap();
    assert_eq!(held.read_raw::<i32>().unwrap(), vec![1; 4]);

    assert!(
        file.dataset_with("d", other()).is_err(),
        "a second open naming another directory is refused"
    );
    assert!(
        file.dataset_with("d", DatasetAccess::new()).is_err(),
        "a second open naming no prefix is refused too"
    );
    assert_eq!(
        file.dataset_with("d", far())
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![1; 4],
        "a second open naming the same directory joins the first"
    );

    drop(held);
    assert_eq!(
        file.dataset_with("d", other())
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![7; 4],
        "with every handle closed the next open sets its own prefix"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// An open that differs only in a property `HDF5_EXTFILE_PREFIX` shadows
/// still agrees, because what `H5D__open_name` compares is the *expanded*
/// prefix (H5Dint.c:1533-1545) — measured under both references: with the
/// variable set, an open naming no prefix joins one that named a directory,
/// which without it is refused.
#[test]
fn the_efile_prefix_join_check_compares_what_the_environment_left() {
    let root = root("efile_join_env");
    raw(&root.join("far"), 1);
    let ext = external_file(&root);

    std::env::set_var("HDF5_EXTFILE_PREFIX", dir_str(&root, "far"));
    let file = H5File::open(&ext).unwrap();
    let held = file
        .dataset_with(
            "d",
            DatasetAccess::new().efile_prefix(dir_str(&root, "other")),
        )
        .unwrap();
    assert_eq!(held.read_raw::<i32>().unwrap(), vec![1; 4]);
    assert_eq!(
        file.dataset_with("d", DatasetAccess::new())
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![1; 4],
        "both opens expand to the environment's prefix, so they agree"
    );
    std::env::remove_var("HDF5_EXTFILE_PREFIX");
    std::fs::remove_dir_all(&root).ok();
}

// ---------------------------------------------------------------------------
// External file list raw data files, on the way out
// ---------------------------------------------------------------------------

/// `home/ext.h5` with the same dataset [`external_file`] builds, created
/// through a dapl naming `prefix` — `H5Dcreate2`'s own dapl, which
/// `H5D__create` reads at H5Dint.c:1318.
fn external_file_written(root: &Path, prefix: Option<&str>, value: i32) -> PathBuf {
    let path = root.join("home").join("ext.h5");
    let file = H5File::create(&path).unwrap();
    let mut builder = file
        .new_dataset::<i32>()
        .shape([4usize])
        .external(&[("raw.bin", 0, 16)]);
    if let Some(prefix) = prefix {
        builder = builder.efile_prefix(prefix);
    }
    builder.create("d").unwrap().write_raw(&[value; 4]).unwrap();
    file.close().unwrap();
    path
}

/// What `dir/raw.bin` holds, or `None` when the write never created it.
fn written(dir: &Path) -> Option<i32> {
    let bytes = std::fs::read(dir.join("raw.bin")).ok()?;
    assert_eq!(bytes.len(), 16);
    let mut first = [0u8; 4];
    first.copy_from_slice(&bytes[..4]);
    Some(i32::from_ne_bytes(first))
}

/// The prefix decides where a write *creates* the raw data file, not only
/// where a read looks for one: `H5D__efl_write` joins the slot name against
/// `dset->shared->extfile_prefix` with the same `H5_combine_path` the read
/// side uses and opens that one path `O_CREAT` (H5Defl.c:429-433).
///
/// Measured (both references): a create through a dapl naming `far` puts the
/// bytes in `far/raw.bin` and leaves the HDF5 file's own directory and the
/// current directory empty, and a read under the same prefix gets them back.
#[test]
fn an_efile_prefix_decides_where_a_write_creates_its_raw_data_file() {
    let root = root("efile_write_prop");
    std::env::set_current_dir(root.join("cwd")).unwrap();
    let ext = external_file_written(&root, Some(&dir_str(&root, "far")), 1);

    assert_eq!(written(&root.join("far")), Some(1));
    assert_eq!(
        written(&root.join("home")),
        None,
        "the write does not fall back to the directory holding the HDF5 file"
    );
    assert_eq!(
        written(&root.join("cwd")),
        None,
        "nor to the current directory, which is where no prefix would put it"
    );
    assert_eq!(
        efl_reads(
            &ext,
            DatasetAccess::new().efile_prefix(dir_str(&root, "far"))
        ),
        Some(1),
        "a read naming the same prefix finds what the write left"
    );
    std::env::set_current_dir("/").unwrap();
    std::fs::remove_dir_all(&root).ok();
}

/// A prefix naming a directory that is not there fails the write rather than
/// creating it: `H5D__efl_write` opens `O_CREAT` and reports "external raw
/// data file does not exist" when that fails (H5Defl.c:432-437).
///
/// Measured (both references): the same arrangement raises "Can't
/// synchronously write data (external raw data file does not exist)".
#[test]
fn a_write_does_not_create_the_directory_an_efile_prefix_names() {
    let root = root("efile_write_missing");
    let file = H5File::create(root.join("home").join("ext.h5")).unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shape([4usize])
        .external(&[("raw.bin", 0, 16)])
        .efile_prefix(dir_str(&root, "absent"))
        .create("d")
        .unwrap();

    assert!(
        ds.write_raw(&[1i32; 4]).is_err(),
        "a prefix naming no directory fails the write"
    );
    assert!(!root.join("absent").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// `HDF5_EXTFILE_PREFIX` shadows the create-time property exactly as it
/// shadows the open-time one — one `H5D__build_file_prefix` serves both
/// (H5Dint.c:1084-1090, reached from :1318 for a create).
///
/// Measured (both references): with the variable naming `other`, a create
/// through a dapl naming `far` writes to `other`; with it empty the property
/// is reached again.
#[test]
fn the_efile_prefix_environment_variable_shadows_the_property_on_a_write() {
    let root = root("efile_write_shadow");

    std::env::set_var("HDF5_EXTFILE_PREFIX", dir_str(&root, "other"));
    external_file_written(&root, Some(&dir_str(&root, "far")), 7);
    assert_eq!(written(&root.join("other")), Some(7));
    assert_eq!(
        written(&root.join("far")),
        None,
        "the property named it, the environment overrode it"
    );

    std::env::set_var("HDF5_EXTFILE_PREFIX", "");
    external_file_written(&root, Some(&dir_str(&root, "far")), 1);
    assert_eq!(
        written(&root.join("far")),
        Some(1),
        "an empty variable is no variable, so the property answers"
    );

    std::env::remove_var("HDF5_EXTFILE_PREFIX");
    std::fs::remove_dir_all(&root).ok();
}

/// The expansion rules reach the write side unchanged, `H5D__build_file_prefix`
/// being run once per open whichever direction the I/O then goes:
/// `${ORIGIN}` is the directory holding the HDF5 file (H5Dint.c:1105-1113),
/// `"."` and `""` are no prefix at all (:1098-1102), which leaves the stored
/// name to be created relative to the process's current directory.
///
/// Measured (both references): `${ORIGIN}` writes next to the HDF5 file,
/// `${ORIGIN}/../far` writes to `far`, and `"."`, `""` and no property alike
/// write to the current directory.
#[test]
fn a_written_efile_prefix_expands_origin_and_treats_dot_as_no_prefix() {
    let root = root("efile_write_origin");
    // nextest runs each test in its own process, so moving the current
    // directory here cannot reach another test.
    std::env::set_current_dir(root.join("cwd")).unwrap();

    external_file_written(&root, Some("${ORIGIN}"), 9);
    assert_eq!(written(&root.join("home")), Some(9));

    external_file_written(&root, Some(&format!("${{ORIGIN}}{}..{}far", MAIN, MAIN)), 1);
    assert_eq!(written(&root.join("far")), Some(1));

    external_file_written(&root, Some("."), 5);
    assert_eq!(written(&root.join("cwd")), Some(5));

    std::fs::remove_file(root.join("cwd").join("raw.bin")).unwrap();
    external_file_written(&root, Some(""), 6);
    assert_eq!(written(&root.join("cwd")), Some(6), "\"\" is no prefix");

    std::fs::remove_file(root.join("cwd").join("raw.bin")).unwrap();
    external_file_written(&root, None, 7);
    assert_eq!(
        written(&root.join("cwd")),
        Some(7),
        "naming no prefix resolves the same way"
    );

    std::env::set_current_dir("/").unwrap();
    std::fs::remove_dir_all(&root).ok();
}

/// The join rule holds between two *write* opens, the create being one of
/// them: `H5D__create` puts its expanded prefix in the shared info
/// (H5Dint.c:1318) that `H5D__open_name` then refuses to disagree with
/// (:1533-1545).
///
/// Measured (both references), with the created dataset's handle still open:
/// a second open naming another directory is refused, one naming none is
/// refused, one naming the same directory joins, and once every handle is
/// dropped a disagreeing open is taken and settles its own prefix.
#[test]
fn a_second_write_open_may_not_disagree_about_the_efile_prefix() {
    let root = root("efile_write_join");
    let file = H5File::create(root.join("home").join("ext.h5")).unwrap();
    let far = || DatasetAccess::new().efile_prefix(dir_str(&root, "far"));
    let other = || DatasetAccess::new().efile_prefix(dir_str(&root, "other"));
    let held = file
        .new_dataset::<i32>()
        .shape([4usize])
        .external(&[("raw.bin", 0, 16)])
        .efile_prefix(dir_str(&root, "far"))
        .create("d")
        .unwrap();

    assert!(
        file.dataset_writer_with("d", other()).is_err(),
        "a second write open naming another directory is refused"
    );
    assert!(
        file.dataset_writer("d").is_err(),
        "a second write open naming no prefix is refused too"
    );
    let joined = file
        .dataset_writer_with("d", far())
        .expect("naming the same directory joins the create");
    joined.write_raw(&[1i32; 4]).unwrap();
    assert_eq!(written(&root.join("far")), Some(1));

    drop(held);
    drop(joined);
    file.dataset_writer_with("d", other())
        .expect("with every handle dropped the next open settles its own prefix")
        .write_raw(&[7i32; 4])
        .unwrap();
    assert_eq!(written(&root.join("other")), Some(7));
    std::fs::remove_dir_all(&root).ok();
}

/// A dataset reopened from an existing file has no prefix until an open
/// gives it one — `H5Fopen` opens no dataset, so the first `H5Dopen2` is what
/// runs `H5D__build_file_prefix` (H5Dint.c:1537) and the `H5Dwrite` after it
/// joins against that answer.
///
/// Measured (both references): creating the dataset without writing it,
/// closing the file, reopening it and opening the dataset through a dapl
/// naming `far` puts a subsequent write's bytes in `far/raw.bin`.
#[test]
fn a_reopened_external_dataset_is_written_under_the_prefix_its_open_names() {
    let root = root("efile_write_reopen");
    std::env::set_current_dir(root.join("cwd")).unwrap();
    let path = root.join("home").join("ext.h5");
    let file = H5File::create(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize])
        .external(&[("raw.bin", 0, 16)])
        .create("d")
        .unwrap();
    file.close().unwrap();
    assert_eq!(
        written(&root.join("cwd")),
        None,
        "creating the dataset writes none of its bytes"
    );

    let file = H5File::open_rw(&path).unwrap();
    file.dataset_writer_with(
        "d",
        DatasetAccess::new().efile_prefix(dir_str(&root, "far")),
    )
    .unwrap()
    .write_raw(&[1i32; 4])
    .unwrap();
    file.close().unwrap();

    assert_eq!(written(&root.join("far")), Some(1));
    assert_eq!(
        written(&root.join("cwd")),
        None,
        "the reopen's own prefix answers, not the default the create had"
    );
    std::env::set_current_dir("/").unwrap();
    std::fs::remove_dir_all(&root).ok();
}
