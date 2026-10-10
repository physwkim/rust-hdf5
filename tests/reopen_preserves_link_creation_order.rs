//! An append session must keep a group's link creation order.
//!
//! libhdf5 lists — and netcdf-c numbers variables — in creation order when a
//! group tracks it, which every NetCDF-4 group does. The reopen walk meets
//! links in header-message order for compact storage, so the rewrite
//! must take each link's stored creation order rather than its discovery
//! order, or a session that changes nothing still reorders the group.
//!
//! The file is written by h5py/libhdf5 (the case that matters: files this
//! writer did not create), so the test needs an interpreter with h5py, found as
//! in `h5py_cross_validation.rs`: `RUST_HDF5_TEST_PYTHON`, else the pinned
//! paths. Without one it skips (passes).

use rust_hdf5::H5File;

const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn python() -> Option<String> {
    let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
        Ok(p) => vec![p],
        Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
    };
    let found = candidates
        .iter()
        .find(|c| std::path::Path::new(c).exists())
        .cloned();
    if found.is_none() {
        eprintln!("skipping h5py cross-check: none of {candidates:?} present");
    }
    found
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "rust_hdf5_reopen_corder_{}_{}.h5",
        name,
        std::process::id()
    ))
}

fn run_python(py: &str, script: &str) {
    let out = std::process::Command::new(py)
        .arg("-c")
        .arg(script)
        .output()
        .expect("run python");
    assert!(
        out.status.success(),
        "python failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Names deliberately out of alphabetical order. With `grp` the root holds 7
/// links and `grp` 3, so both stay in compact (header-message) storage — the
/// case the reopen walk used to reorder; dense storage already kept its order.
const NAMES: [&str; 6] = ["y", "x", "time", "Hs", "Tp", "Dir"];

fn check_script(path: &std::path::Path, expected_root: &[&str], expected_sub: &[&str]) -> String {
    format!(
        r#"
import h5py
def order(g):
    names = []
    g.id.links.iterate(lambda n: names.append(n.decode()), idx_type=h5py.h5.INDEX_CRT_ORDER)
    return names
with h5py.File(r"{}", "r") as f:
    assert order(f) == {:?}, order(f)
    assert order(f["grp"]) == {:?}, order(f["grp"])
"#,
        path.display(),
        expected_root,
        expected_sub
    )
}

#[test]
fn noop_append_session_keeps_link_creation_order() {
    let Some(py) = python() else { return };
    let path = tmp("noop");
    let names: Vec<&str> = NAMES.iter().copied().chain(["grp"]).collect();
    run_python(
        &py,
        &format!(
            r#"
import h5py, numpy as np
with h5py.File(r"{}", "w", libver="latest", track_order=True) as f:
    for n in {:?}:
        f.create_dataset(n, data=np.arange(3, dtype="f4"))
    g = f.create_group("grp", track_order=True)
    for n in ["c", "a", "b"]:
        g.create_dataset(n, data=np.zeros(1, dtype="i4"))
"#,
            path.display(),
            NAMES
        ),
    );
    run_python(&py, &check_script(&path, &names, &["c", "a", "b"]));

    let file = H5File::open_rw(&path).unwrap();
    file.close().unwrap();

    run_python(&py, &check_script(&path, &names, &["c", "a", "b"]));
    std::fs::remove_file(&path).ok();
}

#[test]
fn links_added_in_an_append_session_follow_the_existing_ones() {
    let Some(py) = python() else { return };
    let path = tmp("add");
    run_python(
        &py,
        &format!(
            r#"
import h5py, numpy as np
with h5py.File(r"{}", "w", libver="latest", track_order=True) as f:
    for n in {:?}:
        f.create_dataset(n, data=np.arange(3, dtype="f4"))
    g = f.create_group("grp", track_order=True)
    for n in ["c", "a", "b"]:
        g.create_dataset(n, data=np.zeros(1, dtype="i4"))
"#,
            path.display(),
            NAMES
        ),
    );

    let file = H5File::open_rw(&path).unwrap();
    let ds = file
        .new_dataset::<f32>()
        .shape([2])
        .create("added")
        .unwrap();
    ds.write_raw(&[1.0f32, 2.0]).unwrap();
    file.close().unwrap();

    let names: Vec<&str> = NAMES.iter().copied().chain(["grp", "added"]).collect();
    run_python(&py, &check_script(&path, &names, &["c", "a", "b"]));
    std::fs::remove_file(&path).ok();
}
