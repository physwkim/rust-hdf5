//! Appending to a classic (version-0/1 superblock, symbol-table) file.
//!
//! h5py called without a `libver` argument writes superblock version 0, so
//! this is what a default h5py file looks like: version-1 object headers, and
//! groups whose links live in a local heap indexed by a version-1 B-tree of
//! symbol table nodes. `open_rw` used to refuse the whole shape.
//!
//! Every case here reopens a file libhdf5 itself wrote, adds to it, and hands
//! it back to libhdf5 — through h5py, through `h5dump`, and through `h5clear`,
//! which is the one tool that reads the superblock's own consistency flags.
//! The invariant they share: the file comes back in the format it went in, so
//! a reader that could open it before the append can open it after.

use rust_hdf5::{H5File, LibverBound};

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`. `h5dump` and `h5clear` are taken from the same
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
            eprintln!("skipping classic-append cross-check: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
}

/// A libhdf5 command-line tool from the same install as `py`, or `None` when
/// that install ships without it.
fn h5_tool(py: &str, name: &str) -> Option<std::path::PathBuf> {
    let tool = std::path::Path::new(py).parent()?.join(name);
    tool.exists().then_some(tool)
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_legacy_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// Write `path` with h5py at its default bounds — the whole point of these
/// tests, so the call takes no `libver` argument at all.
fn write_default_h5py(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nf = h5py.File(r'{}', 'w')\n{}\nf.close()\n",
        path.display(),
        body
    );
    run(py, &["-c", &script], "h5py write");
}

/// Read `path` back with h5py; `body` runs with the file open as `f`.
fn read_with_h5py(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nf = h5py.File(r'{}', 'r')\n{}\n",
        path.display(),
        body
    );
    run(py, &["-c", &script], "h5py read-back");
}

fn run(program: impl AsRef<std::ffi::OsStr>, args: &[&str], what: &str) {
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
}

/// The superblock version byte. The signature is at offset 0 in every file
/// here (none has a userblock — `tests/userblock.rs` covers that combination).
fn superblock_version(path: &std::path::Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
    bytes[8]
}

/// Hand the file to the two libhdf5 tools that judge a whole file rather than
/// one object: `h5dump` walks every object header and `h5clear -s` reads the
/// superblock's consistency flags, which is where a half-finished write shows.
fn libhdf5_tools_accept(py: &str, path: &std::path::Path) {
    let path = path.to_str().unwrap();
    if let Some(h5dump) = h5_tool(py, "h5dump") {
        run(h5dump, &["-pBH", path], "h5dump");
    }
    if let Some(h5clear) = h5_tool(py, "h5clear") {
        run(h5clear, &["-s", path], "h5clear -s");
    }
}

/// A default h5py file takes a new dataset and a new root attribute, and comes
/// back a version-0 file that h5py, `h5dump` and `h5clear` all accept.
///
/// This is the case the package exists for: before it, `open_rw` returned
/// "cannot open this file for appending: it uses the classic (version-0/1
/// superblock) HDF5 format" and wrote nothing.
#[test]
fn a_default_h5py_file_takes_a_dataset_and_a_root_attribute() {
    let Some(py) = python() else { return };
    let path = tmp("default");
    write_default_h5py(
        py,
        &path,
        "f['alpha'] = np.arange(6, dtype='<i4')\n\
         f.attrs['made_by'] = 'h5py'\n",
    );
    assert_eq!(superblock_version(&path), 0);

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([3])
        .create("added")
        .unwrap()
        .write_raw(&[7i32, 8, 9])
        .unwrap();
    file.set_attr_numeric("stamped", &42i32).unwrap();
    file.close().unwrap();

    assert_eq!(
        superblock_version(&path),
        0,
        "the append must leave the file in the format it found it"
    );
    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['added', 'alpha'], sorted(f.keys())\n\
         assert list(f['alpha'][...]) == [0, 1, 2, 3, 4, 5]\n\
         assert list(f['added'][...]) == [7, 8, 9]\n\
         assert f.attrs['made_by'] == 'h5py', dict(f.attrs)\n\
         assert f.attrs['stamped'] == 42, dict(f.attrs)\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// Every dataset the append does not touch keeps its bytes, at the address it
/// already had: a classic append rewrites the group headers and the symbol
/// tables under them, and nothing else. (Group headers are rewritten on every
/// finalize whatever the format, because a group's link set is re-derived from
/// the registry rather than edited in place.)
#[test]
fn untouched_datasets_keep_their_bytes_through_a_classic_append() {
    let Some(py) = python() else { return };
    let path = tmp("untouched");
    write_default_h5py(
        py,
        &path,
        "f['alpha'] = np.arange(6, dtype='<i4')\n\
         g = f.create_group('grp')\n\
         g['beta'] = np.arange(4, dtype='<f8')\n\
         g.attrs['units'] = 'mm'\n",
    );

    // The header bytes of each object, before.
    let before = tmp("untouched_copy");
    std::fs::copy(&path, &before).unwrap();

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([2])
        .create("added")
        .unwrap()
        .write_raw(&[1i32, 2])
        .unwrap();
    file.close().unwrap();

    let script = format!(
        "import h5py\n\
         def hdr(p, name):\n\
         \x20   with h5py.File(p, 'r') as f:\n\
         \x20       info = h5py.h5o.get_info(f[name].id)\n\
         \x20       addr, size = info.addr, info.hdr.space.total\n\
         \x20   with open(p, 'rb') as fh:\n\
         \x20       fh.seek(addr)\n\
         \x20       return addr, fh.read(size)\n\
         for name in ('alpha', 'grp/beta'):\n\
         \x20   a, b = hdr(r'{}', name), hdr(r'{}', name)\n\
         \x20   assert a[0] == b[0], (name, 'moved', a[0], b[0])\n\
         \x20   assert a[1] == b[1], (name, 'header rewritten')\n",
        before.display(),
        path.display()
    );
    run(py, &["-c", &script], "header identity");

    read_with_h5py(
        py,
        &path,
        "assert list(f['grp/beta'][...]) == [0.0, 1.0, 2.0, 3.0]\n\
         assert f['grp'].attrs['units'] == 'mm'\n\
         assert list(f['added'][...]) == [1, 2]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&before);
}

/// A dataset added inside an existing subgroup: the insertion is into that
/// group's own symbol table, and its entry in the root's table caches the
/// pair, so a stale cache there is what `H5G__stab_lookup` would follow.
#[test]
fn a_subgroup_of_a_classic_file_takes_a_new_dataset() {
    let Some(py) = python() else { return };
    let path = tmp("subgroup");
    write_default_h5py(
        py,
        &path,
        "g = f.create_group('outer/inner')\n\
         g['beta'] = np.arange(4, dtype='<i4')\n",
    );

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([2])
        .create("outer/inner/late")
        .unwrap()
        .write_raw(&[5i32, 6])
        .unwrap();
    file.create_group("outer/sibling").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert sorted(f['outer'].keys()) == ['inner', 'sibling'], sorted(f['outer'].keys())\n\
         assert sorted(f['outer/inner'].keys()) == ['beta', 'late']\n\
         assert list(f['outer/inner/late'][...]) == [5, 6]\n\
         assert list(f['outer/inner/beta'][...]) == [0, 1, 2, 3]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// Past the single-node boundary. A symbol table node holds `2 * sym_leaf_k`
/// entries and the default `sym_leaf_k` is 4, so the ninth link in a group is
/// the first that cannot fit one node; the default `btree_internal_k` is 16,
/// so the thirty-third node is the first that cannot fit one internal node.
/// 600 links is past both: 75 nodes over three internal nodes under a root,
/// i.e. a two-level tree.
#[test]
fn a_classic_group_grows_past_one_node_and_past_one_level() {
    let Some(py) = python() else { return };
    let path = tmp("split");
    write_default_h5py(py, &path, "f['seed'] = np.arange(2, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();
    for i in 0i32..600 {
        file.new_dataset::<i32>()
            .shape([1])
            .create(&format!("ds{i:04}"))
            .unwrap()
            .write_raw(&[i])
            .unwrap();
    }
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "names = sorted(f.keys())\n\
         assert len(names) == 601, len(names)\n\
         assert names[0] == 'ds0000' and names[-1] == 'seed', (names[0], names[-1])\n\
         for i in (0, 1, 8, 9, 255, 256, 599):\n\
         \x20   assert list(f['ds%04d' % i][...]) == [i], i\n\
         assert list(f['seed'][...]) == [0, 1]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A soft link in a classic group lives in the scratch pad as a heap offset
/// (`H5G_CACHED_SLINK`), not as a link message, and the value string goes in
/// the group's own local heap. It has to survive being read out of one table
/// and written back into the next.
#[test]
fn a_soft_link_survives_a_classic_append() {
    let Some(py) = python() else { return };
    let path = tmp("softlink");
    write_default_h5py(
        py,
        &path,
        "f['alpha'] = np.arange(3, dtype='<i4')\n\
         f['shortcut'] = h5py.SoftLink('/alpha')\n",
    );

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([1])
        .create("added")
        .unwrap()
        .write_raw(&[9i32])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "link = f.get('shortcut', getlink=True)\n\
         assert isinstance(link, h5py.SoftLink), type(link)\n\
         assert link.path == '/alpha', link.path\n\
         assert list(f['shortcut'][...]) == [0, 1, 2]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A version-1 object header records the hard link count in its prefix rather
/// than in an Object Reference Count message — that message does not exist
/// below version 2 (`H5O_link_oh` gates every use of it on the header
/// version). `H5Oget_info().rc` is that prefix field.
#[test]
fn a_hard_link_in_a_classic_file_counts_in_the_header_prefix() {
    let Some(py) = python() else { return };
    let path = tmp("hardlink");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(3, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();
    file.root_group().link("twin", "/alpha").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'twin'], sorted(f.keys())\n\
         assert f['alpha'].id.get_offset() == f['twin'].id.get_offset()\n\
         info = h5py.h5o.get_info(f['alpha'].id)\n\
         assert info.rc == 2, info.rc\n\
         assert info.hdr.version == 1, info.hdr.version\n\
         assert list(f['twin'][...]) == [0, 1, 2]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// Attributes on an object added to a classic file are version-1 attribute
/// messages in a version-1 header, with no Attribute Info message:
/// `H5O__attr_create` gates that message on `oh->version > H5O_VERSION_1`, and
/// so does the dense-storage phase change, so a classic object keeps every
/// attribute in its header however many there are.
#[test]
fn attributes_on_a_classic_object_stay_in_its_header() {
    let Some(py) = python() else { return };
    let path = tmp("attrs");
    write_default_h5py(py, &path, "f['seed'] = np.arange(2, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shape([1])
        .create("carrier")
        .unwrap();
    ds.write_raw(&[1i32]).unwrap();
    // Past `max_compact` (8), which in a version-2 header is the point the
    // attributes would move to a fractal heap.
    for i in 0..12 {
        ds.new_attr::<i32>()
            .shape(())
            .create(&format!("a{i:02}"))
            .unwrap()
            .write_numeric(&i)
            .unwrap();
    }
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "d = f['carrier']\n\
         assert len(d.attrs) == 12, len(d.attrs)\n\
         assert [d.attrs['a%02d' % i] for i in range(12)] == list(range(12))\n\
         info = h5py.h5o.get_info(d.id)\n\
         assert info.hdr.version == 1, info.hdr.version\n\
         assert info.num_attrs == 12, info.num_attrs\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A version-0 file may hold version-2 object headers over link-message
/// groups: `H5G__obj_create_real` uses the new group format when the group
/// tracks link creation order, and `H5O__set_version` raises the header to
/// match. Those groups must not be converted to symbol tables by an append —
/// libhdf5 never converts in that direction, and the conversion would drop the
/// creation order the file was written to keep.
#[test]
fn a_creation_order_group_in_a_classic_file_keeps_its_link_messages() {
    let Some(py) = python() else { return };
    let path = tmp("mixed");
    // `track_order` is a `File` argument, so this one does not go through the
    // default-bounds helper; the bounds are still h5py's defaults.
    let script = format!(
        "import h5py, numpy as np\n\
         with h5py.File(r'{}', 'w', track_order=True) as f:\n\
         \x20   f['zeta'] = np.arange(3, dtype='<i4')\n\
         \x20   f['alpha'] = np.arange(3, dtype='<i4')\n\
         \x20   g = f.create_group('ordered', track_order=True)\n\
         \x20   g['q'] = np.arange(2, dtype='<i4')\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py track_order write");
    assert_eq!(superblock_version(&path), 0);

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([1])
        .create("added")
        .unwrap()
        .write_raw(&[4i32])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert list(f.keys()) == ['zeta', 'alpha', 'ordered', 'added'], list(f.keys())\n\
         assert list(f['ordered'].keys()) == ['q']\n\
         assert list(f['ordered/q'][...]) == [0, 1]\n\
         assert h5py.h5o.get_info(f.id).hdr.version == 2\n\
         assert h5py.h5o.get_info(f['ordered'].id).hdr.version == 2\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A chunked dataset in a classic file is a version-3 data layout message over
/// a version-1 B-tree chunk index, and this file gets one of each: a 1-D grid
/// whose chunks divide the extent evenly and a 2-D grid whose last row and
/// column hang past it, which is where a key's offsets and the tree's right
/// bound are easiest to get wrong.
#[test]
fn a_classic_file_takes_a_chunked_dataset() {
    let Some(py) = python() else { return };
    let path = tmp("chunked");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(6, dtype='<i4')\n");
    let before = std::fs::read(&path).unwrap();

    let file = H5File::open_rw(&path).unwrap();
    let line: Vec<i32> = (0..64).collect();
    file.new_dataset::<i32>()
        .shape([64])
        .chunk(&[8])
        .create("chunky")
        .unwrap()
        .write_raw(&line)
        .unwrap();
    // 3 x 2 chunks over a 10 x 7 extent: both edges are partial.
    let plane: Vec<i32> = (0..70).collect();
    file.new_dataset::<i32>()
        .shape([10, 7])
        .chunk(&[4, 4])
        .create("plane")
        .unwrap()
        .write_raw(&plane)
        .unwrap();
    // A contiguous dataset in the same session still works.
    file.new_dataset::<i32>()
        .shape([2])
        .create("flat")
        .unwrap()
        .write_raw(&[1i32, 2])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    assert_eq!(&before[..8], &std::fs::read(&path).unwrap()[..8]);
    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'chunky', 'flat', 'plane'], sorted(f.keys())\n\
         assert f['chunky'].chunks == (8,), f['chunky'].chunks\n\
         assert list(f['chunky'][...]) == list(range(64)), list(f['chunky'][...])\n\
         assert f['plane'].chunks == (4, 4), f['plane'].chunks\n\
         assert (f['plane'][...] == np.arange(70).reshape(10, 7)).all(), f['plane'][...]\n\
         assert f['chunky'].id.get_create_plist().get_layout() == h5py.h5d.CHUNKED\n",
    );
    libhdf5_tools_accept(py, &path);

    // This crate's own reader walks the same tree it just wrote.
    let back = H5File::open(&path).unwrap();
    assert_eq!(
        back.dataset("chunky").unwrap().read_raw::<i32>().unwrap(),
        line
    );
    assert_eq!(
        back.dataset("plane").unwrap().read_raw::<i32>().unwrap(),
        plane
    );
    drop(back);

    // A second append reopens the tree the first wrote: the chunk data has to
    // survive a header rebuild that does not touch it.
    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([1])
        .create("later")
        .unwrap()
        .write_raw(&[7i32])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert list(f['chunky'][...]) == list(range(64)), list(f['chunky'][...])\n\
         assert (f['plane'][...] == np.arange(70).reshape(10, 7)).all(), f['plane'][...]\n\
         assert list(f['later'][...]) == [7]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A chunk tree past one node, and an unlimited extent grown after the fact.
///
/// The classic index holds `2 * indexed_storage_k` = 64 entries per node at
/// the default K the superblock carries, so 200 chunks force a second level
/// and the separator keys that come with it. libhdf5 reading every value back
/// is what says the keys are ordered and bounded the way `H5D__btree_cmp3`
/// expects.
#[test]
fn a_classic_chunk_tree_grows_past_one_node_and_past_its_extent() {
    let Some(py) = python() else { return };
    let path = tmp("chunk_tree");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(3, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();
    let wide: Vec<i32> = (0..200 * 4).collect();
    file.new_dataset::<i32>()
        .shape([800])
        .chunk(&[4])
        .create("wide")
        .unwrap()
        .write_raw(&wide)
        .unwrap();

    // An unlimited dimension in a classic file is the same index: the tree
    // takes whatever key it is given, so growth needs no reshape of it.
    let growing = file
        .new_dataset::<i32>()
        .shape([8])
        .max_shape(&[None])
        .chunk(&[4])
        .create("growing")
        .unwrap();
    growing
        .write_raw(&[0i32, 1, 2, 3, 100, 101, 102, 103])
        .unwrap();
    growing.append(&[8i32, 9, 10, 11, 12, 13, 14, 15]).unwrap();
    // Overwrite one already-stored chunk by its linear index: the record has
    // to be replaced in the tree rather than added beside itself.
    let second: Vec<u8> = [4i32, 5, 6, 7]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    growing.write_chunk(1, &second).unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert list(f['wide'][...]) == list(range(800)), f['wide'][...]\n\
         assert f['growing'].shape == (16,), f['growing'].shape\n\
         assert f['growing'].maxshape == (None,), f['growing'].maxshape\n\
         assert list(f['growing'][...]) == list(range(16)), list(f['growing'][...])\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// What a classic file accepts is decided per storage form, not per file.
///
/// Compact, contiguous and chunked layouts all encode as a version-3 data
/// layout message, which is exactly what libhdf5 writes at
/// `H5F_LIBVER_EARLIEST`, so all three belong in a version-0 file and are
/// written into one here. A filter does not: its pipeline message would have
/// to be version 1, and this crate encodes version 2 only, so every filtered
/// dataset is still refused where the caller asks for it.
#[test]
fn a_classic_file_takes_compact_and_contiguous_but_no_filter() {
    let Some(py) = python() else { return };
    let path = tmp("storage_matrix");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(6, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();

    // Writable: the image lives inside the layout message in the object
    // header, so a compact dataset asks the file for no structure a version-0
    // superblock lacks.
    file.new_dataset::<i32>()
        .shape([4])
        .compact()
        .create("packed")
        .unwrap()
        .write_raw(&[10i32, 20, 30, 40])
        .unwrap();

    // Writable: one run of bytes and an address to it.
    file.new_dataset::<i32>()
        .shape([3])
        .create("flat")
        .unwrap()
        .write_raw(&[1i32, 2, 3])
        .unwrap();

    // Writable: chunk offsets, sizes and filter masks in a version-1 B-tree.
    file.new_dataset::<i32>()
        .shape([16])
        .chunk(&[8])
        .create("chunky")
        .unwrap()
        .write_raw(&(0i32..16).collect::<Vec<_>>())
        .unwrap();

    // Refused, each for the version-1 filter pipeline message it would need.
    for (which, res) in [
        (
            "deflate",
            file.new_dataset::<i32>()
                .shape([64])
                .chunk(&[8])
                .deflate(6)
                .create("b"),
        ),
        (
            "shuffle",
            file.new_dataset::<i32>()
                .shape([64])
                .chunk(&[8])
                .shuffle()
                .create("c"),
        ),
        (
            "shuffle+deflate",
            file.new_dataset::<i32>()
                .shape([64])
                .chunk(&[8])
                .shuffle_deflate(6)
                .create("d"),
        ),
    ] {
        match res {
            Ok(_) => panic!("{which} needs a filter pipeline message this file cannot hold"),
            Err(e) => assert!(e.to_string().contains("filter pipeline"), "{which}: {e}"),
        }
    }

    file.close().unwrap();

    // Still the file it was, and libhdf5 reads all three new datasets —
    // including the compact one, whose bytes are only in its header.
    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'chunky', 'flat', 'packed'], sorted(f.keys())\n\
         assert list(f['packed'][...]) == [10, 20, 30, 40], list(f['packed'][...])\n\
         assert list(f['flat'][...]) == [1, 2, 3], list(f['flat'][...])\n\
         assert list(f['chunky'][...]) == list(range(16)), list(f['chunky'][...])\n\
         assert f['packed'].id.get_create_plist().get_layout() == h5py.h5d.COMPACT\n\
         assert f['chunky'].id.get_create_plist().get_layout() == h5py.h5d.CHUNKED\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// Raising the library-version bound on a classic file is refused rather than
/// ignored: `H5F_LIBVER_EARLIEST` is the only bound at which libhdf5 writes
/// this format, so a caller who asks for a newer one is asking for a file this
/// append cannot produce.
#[test]
fn raising_the_libver_bound_on_a_classic_file_is_refused() {
    let Some(py) = python() else { return };
    let path = tmp("libver");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(3, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();
    let err = file.set_libver_latest(true).unwrap_err().to_string();
    assert!(err.contains("H5F_LIBVER_EARLIEST"), "{err}");
    // The bound it already has is not a change, so it is not refused.
    file.set_libver_bound(LibverBound::Earliest).unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    let _ = std::fs::remove_file(&path);
}

/// Two append sessions in a row. The second reopens the symbol table the first
/// wrote, so anything the rebuild got wrong about heap layout or B-tree key
/// bounds fails here rather than staying latent.
#[test]
fn a_classic_file_survives_two_appends_in_a_row() {
    let Some(py) = python() else { return };
    let path = tmp("twice");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(3, dtype='<i4')\n");

    for round in 0i32..2 {
        let file = H5File::open_rw(&path).unwrap();
        for i in 0i32..6 {
            file.new_dataset::<i32>()
                .shape([1])
                .create(&format!("r{round}_{i}"))
                .unwrap()
                .write_raw(&[round * 10 + i])
                .unwrap();
        }
        file.close().unwrap();
        assert_eq!(superblock_version(&path), 0, "round {round}");
    }

    read_with_h5py(
        py,
        &path,
        "assert len(f.keys()) == 13, sorted(f.keys())\n\
         assert list(f['r0_5'][...]) == [5]\n\
         assert list(f['r1_5'][...]) == [15]\n\
         assert list(f['alpha'][...]) == [0, 1, 2]\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// The edge-write cap reaches the classic index too. A chunk write addressed
/// by coordinates grows the dataspace to cover the chunk, and the last chunk
/// of a dimension normally hangs past the extent — 10 elements in chunks of 4
/// end at 12 — so the growth is capped at the declared maximum. The version-1
/// B-tree takes any shape and so reaches that extend, unlike the fixed-array
/// and implicit indexes, which return before it.
#[test]
fn a_classic_edge_chunk_written_by_coordinates_keeps_the_extent() {
    let Some(py) = python() else { return };
    let path = tmp("edge_chunk");
    write_default_h5py(py, &path, "f['alpha'] = np.arange(3, dtype='<i4')\n");

    let file = H5File::open_rw(&path).unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shape([10, 7])
        .chunk(&[4, 4])
        .create("plane")
        .unwrap();
    // Chunk row 2 spans elements 8..12 of a dimension that stops at 10, and
    // chunk column 1 spans 4..8 of one that stops at 7.
    let chunk: Vec<u8> = (0i32..16).flat_map(|v| v.to_le_bytes()).collect();
    ds.write_chunk_at(&[2, 1], &chunk).unwrap();
    assert_eq!(ds.shape(), vec![10, 7]);
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "d = f['plane']\n\
         assert d.shape == (10, 7), d.shape\n\
         assert d.chunks == (4, 4), d.chunks\n\
         assert list(d[8, 4:7]) == [0, 1, 2], list(d[8, 4:7])\n\
         assert list(d[9, 4:7]) == [4, 5, 6], list(d[9, 4:7])\n\
         info = h5py.h5o.get_info(d.id)\n\
         assert info.hdr.version == 1, info.hdr.version\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}
