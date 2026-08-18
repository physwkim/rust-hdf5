//! Persisted free-space managers across a reopen.
//!
//! A file whose file-space info message says `persist` records every free
//! region in on-disk managers (`FSHD` header, `FSSE` sections). Nothing else
//! in the file says those regions are free, so a session that rewrites the
//! file without reading them leaks what it frees and grows a file that had
//! room. These cases take a file libhdf5 wrote with `fs_persist`, append to
//! it, and ask libhdf5 itself — `h5stat -S`, `h5clear -s`, h5py — whether the
//! account still adds up.

use rust_hdf5::{FileSpaceStrategy, H5File};

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`. `h5stat` and `h5clear` are taken from the same
/// directory: they are the tools of the libhdf5 that h5py is linked against.
const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn python() -> Option<&'static str> {
    static PY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PY.get_or_init(|| {
        let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
            Ok(p) => vec![p],
            Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
        };
        let found = candidates
            .iter()
            .find(|c| std::path::Path::new(c).exists())
            .cloned();
        if found.is_none() {
            eprintln!("skipping free-space cross-check: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
}

fn h5_tool(py: &str, name: &str) -> Option<std::path::PathBuf> {
    let tool = std::path::Path::new(py).parent()?.join(name);
    tool.exists().then_some(tool)
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_fsm_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run(program: impl AsRef<std::ffi::OsStr>, args: &[&str], what: &str) -> String {
    let out = std::process::Command::new(&program)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed ({}):\n{}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `h5stat -S`'s two numbers: what the free-space managers account for, and
/// what nothing in the file accounts for. The leak this file is about shows up
/// entirely in the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpaceAccount {
    tracked: u64,
    unaccounted: u64,
    total: u64,
}

fn h5stat_space(py: &str, path: &std::path::Path) -> SpaceAccount {
    let tool = h5_tool(py, "h5stat").expect("h5stat ships with this libhdf5");
    let out = run(tool, &["-S", path.to_str().unwrap()], "h5stat -S");
    let number = |label: &str| -> u64 {
        let line = out
            .lines()
            .find(|l| l.trim_start().starts_with(label))
            .unwrap_or_else(|| panic!("h5stat -S printed no {label:?} line:\n{out}"));
        line.split_whitespace()
            .find_map(|w| w.split('/').next().unwrap().parse::<u64>().ok())
            .unwrap_or_else(|| panic!("no number in {line:?}"))
    };
    SpaceAccount {
        tracked: number("Amount/Percent of tracked free space:"),
        unaccounted: number("Unaccounted space:"),
        total: number("Total space:"),
    }
}

fn h5clear_accepts(py: &str, path: &std::path::Path) {
    let tool = h5_tool(py, "h5clear").expect("h5clear ships with this libhdf5");
    run(tool, &["-s", path.to_str().unwrap()], "h5clear -s");
}

/// Write a file with the managers this is about: `fsm` strategy, persisted,
/// and a threshold of one byte so every freed block is recorded.
fn write_persisting_file(py: &str, path: &std::path::Path) {
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'w', fs_strategy='fsm', fs_persist=True, fs_threshold=1)\n\
         f.create_dataset('keep', data=np.arange(16, dtype='i4'))\n\
         f.create_group('grp').create_dataset('inner', data=np.arange(8, dtype='f8'))\n\
         f.create_dataset('drop', data=np.arange(64, dtype='i4'))\n\
         del f['drop']\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py write");
}

/// Read every dataset back and add one, all through libhdf5.
fn h5py_reads_and_appends(py: &str, path: &std::path::Path, name: &str) {
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'r+')\n\
         assert list(f['keep'][:]) == list(range(16)), list(f['keep'][:])\n\
         assert list(f['grp/inner'][:]) == list(range(8)), list(f['grp/inner'][:])\n\
         f.create_dataset('{name}', data=np.arange(4, dtype='i4'))\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back and append");
}

/// One small dataset through the public API, the smallest append that still
/// rewrites the root header, the superblock extension and the managers.
fn crate_appends(path: &std::path::Path, name: &str) {
    let file = H5File::open_rw(path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create(name)
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// A file libhdf5 wrote with persisted managers takes a crate append without
/// losing the account: the append comes out of the free space the managers
/// already tracked, so the file does not grow, and what the rewrite frees goes
/// back into managers `h5stat` still reads.
#[test]
fn a_crate_append_spends_and_rewrites_the_persisted_managers() {
    let Some(py) = python() else { return };
    let path = tmp("append");
    write_persisting_file(py, &path);

    let before = h5stat_space(py, &path);
    assert!(before.tracked > 0, "{before:?} has nothing to reuse");
    assert_eq!(
        before.unaccounted, 0,
        "libhdf5 left the file fully accounted"
    );

    crate_appends(&path, "added");
    let after = h5stat_space(py, &path);
    // The whole append fits in space the managers named, so the end of the
    // file does not move. Before this, every byte of it came from the end.
    assert_eq!(
        after.total, before.total,
        "the append grew a file that had {} bytes free",
        before.tracked
    );
    assert!(
        after.tracked > 0,
        "the append left the file with no tracked free space: {after:?}"
    );

    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, ["added", "by_libhdf5", "grp/inner", "keep"]);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        (0..16).collect::<Vec<i32>>()
    );
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// The second append reads managers this crate wrote rather than libhdf5's,
/// which is the only way to reach the encoding the write side produces. The
/// file stays one libhdf5 accepts and h5py can still add to.
#[test]
fn the_managers_the_crate_wrote_are_readable_by_both_libraries() {
    let Some(py) = python() else { return };
    let path = tmp("twice");
    write_persisting_file(py, &path);

    crate_appends(&path, "first");
    let once = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the file persists its free space");
    assert_ne!(once.fs_addr[0], u64::MAX, "no manager was written");

    crate_appends(&path, "second");
    let twice = h5stat_space(py, &path);
    assert!(twice.tracked > 0, "{twice:?}");
    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(
        names,
        ["by_libhdf5", "first", "grp/inner", "keep", "second"]
    );
    assert_eq!(
        file.dataset("grp/inner")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        (0..8).map(f64::from).collect::<Vec<f64>>()
    );
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// `sohm_paged.h5` is the paged fixture, and it persists nothing: `gen_sohm.c`
/// passes `persist = 0` to `H5Pset_file_space_strategy`, so its file-space
/// info message names no manager and `h5stat -S` reports no tracked free space
/// at all. There is no free space to record for it, and an append must not
/// invent one.
#[test]
fn the_paged_fixture_persists_no_manager_to_rewrite() {
    let Some(py) = python() else { return };
    let path = tmp("paged");
    std::fs::copy(fixture("sohm_paged.h5"), &path).unwrap();

    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the paged fixture declares a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert!(!info.persist, "the fixture would have managers to read");
    assert_eq!(h5stat_space(py, &path).tracked, 0);

    crate_appends(&path, "added");
    assert_eq!(
        h5stat_space(py, &path).tracked,
        0,
        "a file that persists nothing came back with a manager"
    );
    h5clear_accepts(py, &path);
    // The fixture's own datasets, not the ones the persisting file has.
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'r+')\n\
         assert list(f['shared0'][:]) == list(range(8)), list(f['shared0'][:])\n\
         assert list(f['shared1'][:]) == list(range(10, 18)), list(f['shared1'][:])\n\
         assert f['uses_named'].dtype == f['named_i32'].dtype\n\
         f.create_dataset('by_libhdf5', data=np.arange(4, dtype='i4'))\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back and append");
    let _ = std::fs::remove_file(&path);
}
