//! Reopen-rewrite parity: an object this writer cannot model must survive a
//! reopen exactly as the file already had it.
//!
//! `open_rw` rebuilds its registry by reading the file, and `close` rewrites
//! the root group — and every group — out of that registry. Anything the read
//! only half understood used to be rebuilt from the half it got: a dataset
//! whose datatype message did not decode came back as an empty *group*, and a
//! group whose links or attributes were in a fractal heap came back empty.
//! Fixtures come from h5py, as in `catalog_read_parity`.

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

/// A fresh directory per case: each holds the fixture and the copy the writer
/// reopens, so the two can be compared byte for byte afterwards.
fn tmp_dir(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_reopen_{}_{}_{}",
        name,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).expect("failed to create fixture directory");
    dir
}

/// Run `body` with `ORIG` and `WORK` bound to the two file paths. A failed
/// `assert` inside the script fails the test, and python's traceback reaches
/// test stderr.
fn py_run(py: &str, orig: &std::path::Path, work: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nORIG = r'{}'\nWORK = r'{}'\n{}\n",
        orig.display(),
        work.display(),
        body
    );
    let status = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python");
    assert!(status.success(), "python step failed");
}

/// The helper the comparisons share: every object named must be at the same
/// header address in both files, with the same header bytes.
const HEADER_IDENTITY: &str = "\
def header_bytes(path, name):\n\
\x20   with h5py.File(path, 'r') as f:\n\
\x20       info = h5py.h5o.get_info(f[name].id)\n\
\x20       addr, size = info.addr, info.hdr.space.total\n\
\x20   with open(path, 'rb') as fh:\n\
\x20       fh.seek(addr)\n\
\x20       return addr, fh.read(size)\n\
def assert_untouched(name):\n\
\x20   a = header_bytes(ORIG, name)\n\
\x20   b = header_bytes(WORK, name)\n\
\x20   assert a[0] == b[0], (name, 'moved', a[0], b[0])\n\
\x20   assert a[1] == b[1], (name, 'header rewritten')\n";

/// Reopen `work`, add one dataset, close. This is the whole of what a caller
/// has to do to make the rewrite happen.
fn reopen_and_add(work: &std::path::Path) {
    let file = rust_hdf5::H5File::open_rw(work).expect("open_rw");
    file.new_dataset::<i32>()
        .shape([2])
        .create("added")
        .expect("create")
        .write_raw(&[7i32, 8])
        .expect("write");
    file.close().expect("close");
}

/// Three objects this writer models as nothing: a dataset with an opaque
/// datatype (the message does not decode), a dataset built on a committed
/// type (its datatype message is a *reference*, and decoding those bytes as a
/// type answers something else), and the committed type itself. All three
/// were rewritten as empty groups, which orphaned their data and destroyed
/// the type.
#[test]
fn an_object_the_writer_cannot_model_keeps_its_bytes_through_a_reopen() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("unmodelled");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    py_run(
        py,
        &orig,
        &work,
        "from h5py import h5d, h5s\n\
         with h5py.File(ORIG, 'w', libver='latest') as f:\n\
         \x20   f.create_dataset('plain', data=np.arange(4, dtype='<i4'))\n\
         \x20   f.create_dataset('opaque', data=np.arange(4, dtype='u1').view('V4'))\n\
         \x20   f['t'] = np.dtype('<i4')\n\
         \x20   sid = h5s.create_simple((4,))\n\
         \x20   dsid = h5d.create(f.id, b'shared', f['t'].id, sid)\n\
         \x20   dsid.write(h5s.ALL, h5s.ALL, np.arange(4, dtype='<i4'))\n",
    );
    std::fs::copy(&orig, &work).unwrap();
    reopen_and_add(&work);

    py_run(
        py,
        &orig,
        &work,
        &format!(
            "{HEADER_IDENTITY}\
             with h5py.File(WORK, 'r') as f:\n\
             \x20   assert sorted(f.keys()) == ['added', 'opaque', 'plain', 'shared', 't'], \
             sorted(f.keys())\n\
             \x20   assert isinstance(f['opaque'], h5py.Dataset), type(f['opaque'])\n\
             \x20   assert f['opaque'].dtype == np.dtype('V4'), f['opaque'].dtype\n\
             \x20   assert isinstance(f['t'], h5py.Datatype), type(f['t'])\n\
             \x20   assert f['t'].dtype == np.dtype('<i4'), f['t'].dtype\n\
             \x20   assert isinstance(f['shared'], h5py.Dataset), type(f['shared'])\n\
             \x20   assert list(f['shared'][...]) == [0, 1, 2, 3], list(f['shared'][...])\n\
             \x20   assert list(f['plain'][...]) == [0, 1, 2, 3]\n\
             \x20   assert list(f['added'][...]) == [7, 8]\n\
             for name in ('opaque', 't', 'shared'):\n\
             \x20   assert_untouched(name)\n"
        ),
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A group with enough links moves them into a fractal heap, which this
/// writer does not read: the rewrite emitted the group with the links it
/// found as messages — none — and twenty datasets left the file.
#[test]
fn a_group_whose_links_are_dense_keeps_its_children_through_a_reopen() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("dense_links");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(ORIG, 'w', libver='latest') as f:\n\
         \x20   g = f.create_group('dense')\n\
         \x20   for i in range(20):\n\
         \x20       g.create_dataset('d%02d' % i, data=np.arange(2, dtype='<i4'))\n\
         \x20   f.create_dataset('plain', data=np.arange(4, dtype='<i4'))\n",
    );
    std::fs::copy(&orig, &work).unwrap();
    reopen_and_add(&work);

    py_run(
        py,
        &orig,
        &work,
        &format!(
            "{HEADER_IDENTITY}\
             with h5py.File(WORK, 'r') as f:\n\
             \x20   names = sorted(f['dense'].keys())\n\
             \x20   assert names == ['d%02d' % i for i in range(20)], names\n\
             \x20   assert list(f['dense/d07'][...]) == [0, 1]\n\
             \x20   assert list(f['added'][...]) == [7, 8]\n\
             assert_untouched('dense')\n"
        ),
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The same for attributes: past the compact limit libhdf5 moves them into a
/// heap and writes no attribute messages at all, so a rewrite from the
/// messages alone emitted the group without them.
#[test]
fn a_group_whose_attributes_are_dense_keeps_them_through_a_reopen() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("dense_attrs");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(ORIG, 'w', libver='latest') as f:\n\
         \x20   g = f.create_group('g')\n\
         \x20   for i in range(12):\n\
         \x20       g.attrs['a%02d' % i] = np.int32(i)\n\
         \x20   g.create_dataset('inner', data=np.arange(3, dtype='<i4'))\n\
         \x20   f.create_dataset('plain', data=np.arange(4, dtype='<i4'))\n",
    );
    std::fs::copy(&orig, &work).unwrap();
    reopen_and_add(&work);

    py_run(
        py,
        &orig,
        &work,
        &format!(
            "{HEADER_IDENTITY}\
             with h5py.File(WORK, 'r') as f:\n\
             \x20   names = sorted(f['g'].attrs.keys())\n\
             \x20   assert names == ['a%02d' % i for i in range(12)], names\n\
             \x20   assert f['g'].attrs['a05'] == 5\n\
             \x20   assert list(f['g/inner'][...]) == [0, 1, 2]\n\
             \x20   assert list(f['added'][...]) == [7, 8]\n\
             assert_untouched('g')\n"
        ),
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Creation-order tracking writes the same attribute-info message while the
/// attributes are still messages. That is not dense storage, and treating it
/// as such would refuse a file this writer can rewrite in full.
#[test]
fn tracked_creation_order_is_not_dense_storage() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("track_order");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(ORIG, 'w', libver='latest', track_order=True) as f:\n\
         \x20   f.attrs['b'] = np.int32(1)\n\
         \x20   f.attrs['a'] = np.int32(2)\n\
         \x20   g = f.create_group('g')\n\
         \x20   g.attrs['z'] = np.int32(3)\n",
    );
    std::fs::copy(&orig, &work).unwrap();
    reopen_and_add(&work);

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(WORK, 'r') as f:\n\
         \x20   assert sorted(f.attrs.keys()) == ['a', 'b'], sorted(f.attrs.keys())\n\
         \x20   assert sorted(f['g'].attrs.keys()) == ['z']\n\
         \x20   assert list(f['added'][...]) == [7, 8]\n",
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The root group is the one object with no alternative: every append
/// rewrites its header, so an unreadable part of it cannot ride along. The
/// open says so and leaves the file alone, rather than rewriting the root
/// from the part it could read and dropping the rest.
#[test]
fn a_root_group_this_writer_cannot_rewrite_in_full_is_refused_untouched() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("dense_root_attrs");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(ORIG, 'w', libver='latest') as f:\n\
         \x20   for i in range(12):\n\
         \x20       f.attrs['r%02d' % i] = np.int32(i)\n\
         \x20   f.create_dataset('plain', data=np.arange(4, dtype='<i4'))\n",
    );
    std::fs::copy(&orig, &work).unwrap();

    let err = rust_hdf5::H5File::open_rw(&work)
        .err()
        .expect("must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("fractal heap") && msg.contains("root group"),
        "the refusal must name what it could not read and why the root forces it: {msg}"
    );

    assert_eq!(
        std::fs::read(&orig).unwrap(),
        std::fs::read(&work).unwrap(),
        "a refused open must not have written anything"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A preserved object still occupies its name. Asking for it by name says
/// what it is rather than reporting it absent, and creating something of that
/// name is refused — two link messages of one name in a group is an invalid
/// file, and the preserved one is written back on every rewrite.
#[test]
fn a_preserved_object_is_named_by_the_writer_rather_than_reported_absent() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("preserved_name");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(ORIG, 'w', libver='latest') as f:\n\
         \x20   f.create_dataset('opaque', data=np.arange(4, dtype='u1').view('V4'))\n",
    );
    std::fs::copy(&orig, &work).unwrap();

    let file = rust_hdf5::H5File::open_rw(&work).unwrap();
    let why = match file.dataset_writer("opaque") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the writer must not hand out a preserved object"),
    };
    assert!(
        why.contains("opaque") && why.contains("does not decode"),
        "the writer must say why it will not open the object: {why}"
    );
    let taken = match file.new_dataset::<i32>().shape([1]).create("opaque") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a preserved name must not be creatable"),
    };
    assert!(
        taken.contains("already exists"),
        "a preserved name is taken: {taken}"
    );
    // The listing carries it, so a caller can see the name at all.
    assert!(file
        .root_group()
        .link_names()
        .unwrap()
        .contains(&"opaque".to_string()));
    file.close().unwrap();

    std::fs::remove_dir_all(&dir).ok();
}
