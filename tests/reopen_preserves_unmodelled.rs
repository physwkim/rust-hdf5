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

/// A group with enough links moves them into a fractal heap. The rewrite once
/// emitted the group with the links it found as messages — none — and twenty
/// datasets left the file. The reopen now reads the heap back, so the group is
/// rewritten holding every child it had; what must never happen is a child
/// going missing, whichever of the two the writer does.
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
             \x20   for i in range(20):\n\
             \x20       assert list(f['dense/d%02d' % i][...]) == [0, 1], i\n\
             \x20   assert list(f['added'][...]) == [7, 8]\n\
             assert_untouched('plain')\n"
        ),
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The same for attributes: past the compact limit libhdf5 moves them into a
/// heap and writes no attribute messages at all, so a rewrite from the
/// messages alone emitted the group without them. The reopen reads the heap
/// back and writes the set out again — every name, every value.
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
             \x20   for i in range(12):\n\
             \x20       assert f['g'].attrs['a%02d' % i] == i, i\n\
             \x20   assert list(f['g/inner'][...]) == [0, 1, 2]\n\
             \x20   assert list(f['added'][...]) == [7, 8]\n\
             assert_untouched('plain')\n"
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
/// rewrites its header, so an unreadable part of it cannot ride along. Any
/// other object would be kept by its bytes; the root cannot be, so the open
/// says so and leaves the file alone, rather than rewriting the root from the
/// part it could read and dropping the rest.
///
/// Dense root attributes on their own no longer force this — the reopen reads
/// the heap back. What forces it is a heap that will not read, made here by
/// breaking the name index the attribute info message points at.
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
    // The v2 B-tree header the root's attribute name index lives in. It sits
    // outside the object header, so this breaks the attribute set without
    // disturbing the checksum that guards the root's links.
    let mut raw = std::fs::read(&orig).unwrap();
    let hits: Vec<usize> = (0..raw.len() - 4)
        .filter(|&i| &raw[i..i + 4] == b"BTHD")
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "fixture must hold exactly one v2 B-tree, the attribute name index"
    );
    raw[hits[0]..hits[0] + 4].copy_from_slice(b"XXXX");
    std::fs::write(&orig, &raw).unwrap();
    std::fs::copy(&orig, &work).unwrap();

    let err = rust_hdf5::H5File::open_rw(&work)
        .err()
        .expect("must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("dense attribute storage") && msg.contains("root group"),
        "the refusal must name what it could not read and why the root forces it: {msg}"
    );

    assert_eq!(
        std::fs::read(&orig).unwrap(),
        std::fs::read(&work).unwrap(),
        "a refused open must not have written anything"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The same fixture without the damage: dense root attributes are read back
/// out of the heap and written out again, so an append keeps every one of
/// them. This is the case the refusal above must *not* catch.
#[test]
fn dense_root_attributes_survive_a_reopen() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("dense_root_attrs_ok");
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
    reopen_and_add(&work);

    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(WORK, 'r') as f:\n\
         \x20   names = sorted(f.attrs.keys())\n\
         \x20   assert names == ['r%02d' % i for i in range(12)], names\n\
         \x20   for i in range(12):\n\
         \x20       assert f.attrs['r%02d' % i] == i, i\n\
         \x20   assert list(f['plain'][...]) == [0, 1, 2, 3]\n\
         \x20   assert list(f['added'][...]) == [7, 8]\n",
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Flip one byte of the first `sig` block in `path`, at `offset` bytes past
/// the signature. Every chunk-index block ends in a checksum over its own
/// bytes, so this is enough to make one unreadable while leaving the object
/// header — and the rest of the file — exactly as libhdf5 wrote it.
fn corrupt_block(path: &std::path::Path, sig: &[u8], offset: usize) {
    let mut data = std::fs::read(path).unwrap();
    let at = data
        .windows(sig.len())
        .position(|w| w == sig)
        .unwrap_or_else(|| panic!("{} not in fixture", String::from_utf8_lossy(sig)));
    data[at + offset] ^= 0xff;
    std::fs::write(path, data).unwrap();
}

/// A chunk index the reopen cannot read back is the same loss one message
/// down: the extensible-array index block was decoded with a fallback that
/// substituted an *empty* index, so the reopened dataset was registered
/// believing it had no chunks. Writing one chunk to it then rewrote the index
/// with that chunk alone and every chunk the file already held was stranded —
/// the blocks still in the file, nothing naming them.
#[test]
fn a_chunk_index_that_does_not_read_back_keeps_its_chunks_through_a_reopen() {
    let Some(py) = python() else { return };
    let dir = tmp_dir("bad_chunk_index");
    let (orig, work) = (dir.join("orig.h5"), dir.join("work.h5"));

    // maxshape with exactly one unlimited dimension is what makes libhdf5
    // index the chunks with an extensible array.
    py_run(
        py,
        &orig,
        &work,
        "with h5py.File(ORIG, 'w', libver='latest') as f:\n\
         \x20   f.create_dataset('ea', data=np.arange(8, dtype='<i4'), \
         maxshape=(None,), chunks=(2,))\n\
         \x20   f.create_dataset('plain', data=np.arange(4, dtype='<i4'))\n",
    );
    corrupt_block(&orig, b"EAIB", 6);
    std::fs::copy(&orig, &work).unwrap();

    let file = rust_hdf5::H5File::open_rw(&work).expect("the rest of the file still opens");
    let why = match file.dataset_writer("ea") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the writer must not hand out a dataset whose index it could not read"),
    };
    assert!(
        why.contains("ea") && why.contains("chunk index"),
        "the writer must say which object it kept and why: {why}"
    );
    file.new_dataset::<i32>()
        .shape([2])
        .create("added")
        .expect("create")
        .write_raw(&[7i32, 8])
        .expect("write");
    file.close().expect("close");

    py_run(
        py,
        &orig,
        &work,
        &format!(
            "{HEADER_IDENTITY}\
             def block(path, sig, n):\n\
             \x20   data = open(path, 'rb').read()\n\
             \x20   i = data.find(sig)\n\
             \x20   assert i > 0, (path, sig)\n\
             \x20   return i, data[i:i + n]\n\
             with h5py.File(WORK, 'r') as f:\n\
             \x20   assert sorted(f.keys()) == ['added', 'ea', 'plain'], sorted(f.keys())\n\
             \x20   assert f['ea'].shape == (8,), f['ea'].shape\n\
             \x20   assert list(f['plain'][...]) == [0, 1, 2, 3]\n\
             \x20   assert list(f['added'][...]) == [7, 8]\n\
             assert_untouched('ea')\n\
             # header, then signature through the four chunk addresses.\n\
             assert block(ORIG, b'EAHD', 44) == block(WORK, b'EAHD', 44), 'array header moved'\n\
             assert block(ORIG, b'EAIB', 46) == block(WORK, b'EAIB', 46), 'index block rewritten'\n"
        ),
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
         \x20   f.create_dataset('src', data=np.arange(4, dtype='<i4'))\n\
         \x20   layout = h5py.VirtualLayout(shape=(4,), dtype='<i4')\n\
         \x20   layout[...] = h5py.VirtualSource(f['src'])\n\
         \x20   f.create_virtual_dataset('opaque', layout)\n",
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
