//! Creating a file at `H5F_LIBVER_EARLIEST`.
//!
//! `H5Pset_libver_bounds(fapl, H5F_LIBVER_EARLIEST, ...)` does not select one
//! structure; it selects a generation. `HDF5_superblock_ver_bounds` gives
//! version 0, `H5O_obj_ver_bounds` gives version-1 object headers,
//! `H5G__obj_create_real` puts a group's links in a symbol table, and
//! `H5O_layout_ver_bounds` keeps the data layout message at version 3, whose
//! only chunk index is the version-1 B-tree. The file that comes out is the
//! one libhdf5 1.6 could read.
//!
//! `tests/legacy_append.rs` covers the same format reached from the other
//! side — a file libhdf5 wrote, reopened and appended to. This file covers it
//! reached from creation, which before this had no way in: `H5File::create`
//! wrote the v1.8-shaped file and nothing asked for anything older.
//!
//! Two probes recur. The four-byte signatures a structure writes are checked
//! directly in the file — `HEAP`/`SNOD`/`TREE` for the symbol-table group and
//! the version-1 chunk B-tree, and the absence of `OHDR`, `FRHP`, `BTHD`,
//! `EAHD` and `FAHD` for everything newer that must not be there. Then
//! libhdf5 itself reads the file back, through h5py and through `h5dump` and
//! `h5clear`, which is the check that the structures are not merely present
//! but correct.

use rust_hdf5::{H5File, LibverBound};

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset,
/// matching `legacy_append`. `h5dump` and `h5clear` are taken from the same
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
            eprintln!("skipping earliest-bound cross-check: none of {candidates:?} present");
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
        "rust_hdf5_earliest_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

fn earliest(path: &std::path::Path) -> H5File {
    H5File::options()
        .libver(LibverBound::Earliest)
        .create(path)
        .unwrap()
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

/// Hand the file to the two libhdf5 tools that judge a whole file rather than
/// one object: `h5dump` walks every object header and `h5clear -s` reads the
/// superblock's consistency flags.
fn libhdf5_tools_accept(py: &str, path: &std::path::Path) {
    let path = path.to_str().unwrap();
    if let Some(h5dump) = h5_tool(py, "h5dump") {
        run(h5dump, &["-pBH", path], "h5dump");
    }
    if let Some(h5clear) = h5_tool(py, "h5clear") {
        run(h5clear, &["-s", path], "h5clear -s");
    }
}

/// The superblock version byte, at `offset` because a userblock moves it.
fn superblock_version_at(path: &std::path::Path, offset: usize) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[offset..offset + 8], b"\x89HDF\r\n\x1a\n");
    bytes[offset + 8]
}

fn superblock_version(path: &std::path::Path) -> u8 {
    superblock_version_at(path, 0)
}

fn contains(path: &std::path::Path, magic: &[u8; 4]) -> bool {
    std::fs::read(path)
        .unwrap()
        .windows(4)
        .any(|w| w == magic.as_slice())
}

/// Every structure that belongs to a generation newer than this one. None may
/// appear in a file created at the earliest bound: `OHDR` is the version-2
/// object header signature (a version-1 header has none), `FRHP` the fractal
/// heap behind dense links and attributes, and the last three the v1.10 chunk
/// indexes.
fn no_newer_structures(path: &std::path::Path) {
    for magic in [b"OHDR", b"FRHP", b"BTHD", b"EAHD", b"FAHD"] {
        assert!(
            !contains(path, magic),
            "{} appears in a file created at H5F_LIBVER_EARLIEST",
            String::from_utf8_lossy(magic)
        );
    }
}

/// The whole generation in one file: superblock version 0, a symbol-table
/// root group (its local heap and B-tree named in the superblock's own root
/// entry), version-1 object headers, and nothing newer anywhere.
#[test]
fn a_file_created_at_earliest_is_the_classic_generation() {
    let path = tmp("generation");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([6])
        .create("alpha")
        .unwrap()
        .write_raw(&(0..6i32).collect::<Vec<_>>())
        .unwrap();
    file.create_group("outer").unwrap();
    file.set_attr_string("made_by", "rust-hdf5").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    for magic in [b"HEAP", b"SNOD", b"TREE"] {
        assert!(
            contains(&path, magic),
            "{} is missing from a symbol-table file",
            String::from_utf8_lossy(magic)
        );
    }
    no_newer_structures(&path);

    // The superblock's root symbol table entry: the cache type is 1
    // (`H5G_CACHED_STAB`), so the pair of addresses after it is the root
    // group's B-tree and local heap. Its zero name offset is the empty name
    // every root entry carries.
    let bytes = std::fs::read(&path).unwrap();
    // 8 signature + 1 version + 3 sub-versions + 1 reserved + 4 K ranks
    // + 4 consistency flags + 4 * 8 addresses = 56.
    let root_entry = 56;
    assert_eq!(
        u64::from_le_bytes(bytes[root_entry..root_entry + 8].try_into().unwrap()),
        0,
        "root entry name offset"
    );
    assert_eq!(
        u32::from_le_bytes(bytes[root_entry + 16..root_entry + 20].try_into().unwrap()),
        1,
        "root entry cache type"
    );
    // The root object header's version byte, at the address the entry names.
    let root_addr =
        u64::from_le_bytes(bytes[root_entry + 8..root_entry + 16].try_into().unwrap()) as usize;
    assert_eq!(bytes[root_addr], 1, "root object header version");

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("alpha").unwrap().read_raw::<i32>().unwrap(),
        (0..6i32).collect::<Vec<_>>()
    );
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// libhdf5 reads the whole file: the datasets, the nested groups, the links
/// and the attributes. `H5Oget_info().hdr.version` is the object header
/// version libhdf5 itself decoded, which is the check the signature scan
/// cannot make — a version-1 header has no signature to look for.
#[test]
fn libhdf5_reads_a_file_created_at_earliest() {
    let Some(py) = python() else { return };
    let path = tmp("libhdf5_reads");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([6])
        .create("alpha")
        .unwrap()
        .write_raw(&(0..6i32).collect::<Vec<_>>())
        .unwrap();
    file.create_group("outer").unwrap();
    let inner = file.create_group("outer/inner").unwrap();
    inner
        .new_dataset::<f64>()
        .shape([3])
        .create("beta")
        .unwrap()
        .write_raw(&[1.5f64, 2.5, 3.5])
        .unwrap();
    inner.set_attr_string("who", "inner").unwrap();
    file.create_soft_link("shortcut", "/alpha").unwrap();
    file.set_attr_string("made_by", "rust-hdf5").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'outer', 'shortcut'], sorted(f.keys())\n\
         assert list(f['alpha'][...]) == list(range(6)), list(f['alpha'][...])\n\
         assert list(f['outer/inner/beta'][...]) == [1.5, 2.5, 3.5]\n\
         assert f['outer/inner'].attrs['who'] == 'inner'\n\
         assert f.attrs['made_by'] == 'rust-hdf5'\n\
         link = f.get('shortcut', getlink=True)\n\
         assert isinstance(link, h5py.SoftLink), type(link)\n\
         assert link.path == '/alpha', link.path\n\
         for path in ('/', '/outer', '/outer/inner', '/alpha', '/outer/inner/beta'):\n\
         \x20   v = h5py.h5o.get_info(f[path].id).hdr.version\n\
         \x20   assert v == 1, (path, v)\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// Every group in the file keeps its links in a symbol table, not only the
/// root: `H5G__obj_create_real` reads the same bound for each. Past
/// `2 * sym_leaf_k` = 8 links the table splits, so the wide group here is one
/// whose B-tree has more than a root node — the case where a wrong key bound
/// stops libhdf5 finding a name rather than stopping it opening the file.
#[test]
fn every_group_created_at_earliest_stores_its_links_in_a_symbol_table() {
    let Some(py) = python() else { return };
    let path = tmp("symtab_groups");
    let file = earliest(&path);
    let wide = file.create_group("wide").unwrap();
    for i in 0i32..40 {
        wide.new_dataset::<i32>()
            .shape([1])
            .create(&format!("ds{i:02}"))
            .unwrap()
            .write_raw(&[i])
            .unwrap();
    }
    file.create_group("wide/nested").unwrap();
    file.create_group("wide/nested/deeper").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    read_with_h5py(
        py,
        &path,
        "names = sorted(f['wide'].keys())\n\
         assert len(names) == 41, len(names)\n\
         for i in (0, 7, 8, 39):\n\
         \x20   assert list(f['wide/ds%02d' % i][...]) == [i], i\n\
         assert list(f['wide/nested'].keys()) == ['deeper']\n\
         for path in ('/', '/wide', '/wide/nested', '/wide/nested/deeper'):\n\
         \x20   v = h5py.h5o.get_info(f[path].id).hdr.version\n\
         \x20   assert v == 1, (path, v)\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A chunked dataset at this bound is indexed by the version-1 B-tree, the
/// only index a version-3 data layout message can name. The v1.10 indexes
/// this crate reaches for by default each write a signature of their own, and
/// none of them is here; the `TREE` that is here is read back by libhdf5,
/// including an edge chunk and a second dataset whose tree grows past one
/// node.
#[test]
fn a_chunked_dataset_created_at_earliest_uses_the_version_1_btree() {
    let Some(py) = python() else { return };
    let path = tmp("chunked");
    let file = earliest(&path);
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
    // Past 2 * indexed_storage_k = 64 entries, so past one B-tree node.
    let wide: Vec<i32> = (0..200 * 4).collect();
    file.new_dataset::<i32>()
        .shape([200 * 4])
        .chunk(&[4])
        .create("wide")
        .unwrap()
        .write_raw(&wide)
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    assert!(contains(&path, b"TREE"));
    no_newer_structures(&path);
    read_with_h5py(
        py,
        &path,
        "assert f['chunky'].chunks == (8,), f['chunky'].chunks\n\
         assert list(f['chunky'][...]) == list(range(64))\n\
         assert f['plane'].chunks == (4, 4), f['plane'].chunks\n\
         assert (f['plane'][...] == np.arange(70).reshape(10, 7)).all()\n\
         assert (f['wide'][...] == np.arange(800)).all()\n\
         for name in ('chunky', 'plane', 'wide'):\n\
         \x20   assert f[name].id.get_create_plist().get_layout() == h5py.h5d.CHUNKED\n",
    );
    libhdf5_tools_accept(py, &path);

    let back = H5File::open(&path).unwrap();
    assert_eq!(
        back.dataset("wide").unwrap().read_raw::<i32>().unwrap(),
        wide
    );
    drop(back);
    let _ = std::fs::remove_file(&path);
}

/// A filtered chunked dataset, whose pipeline message is version 1 at this
/// bound (`H5O_pline_ver_bounds`) — the version that names every filter,
/// libhdf5's own included, and pads the name and the client-data array. The
/// chunks go in the same version-1 B-tree an unfiltered dataset uses; its
/// keys already carry the stored size and the filter mask.
#[test]
fn a_filtered_dataset_created_at_earliest_gets_a_version_1_pipeline() {
    let Some(py) = python() else { return };
    let path = tmp("filtered");
    let file = earliest(&path);
    let ramp: Vec<i32> = (0..64).collect();
    for (name, builder) in [
        (
            "gz",
            file.new_dataset::<i32>().shape([64]).chunk(&[8]).deflate(6),
        ),
        (
            "sh",
            file.new_dataset::<i32>().shape([64]).chunk(&[8]).shuffle(),
        ),
        (
            "shgz",
            file.new_dataset::<i32>()
                .shape([64])
                .chunk(&[8])
                .shuffle_deflate(6),
        ),
    ] {
        builder.create(name).unwrap().write_raw(&ramp).unwrap();
    }
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    // The message as it lies in the file: version 1, one filter, six reserved
    // bytes, then deflate with a name length of 8, flags, one client-data
    // value, the padded name, the level and the pad that makes the value
    // count even. A version-2 message would name no filter below
    // `H5Z_FILTER_RESERVED` at all, so the name alone separates them — but
    // the whole message is checked, because the padding rules are the half
    // that is easy to get wrong.
    let mut deflate_v1 = vec![1u8, 1, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 1, 0];
    deflate_v1.extend_from_slice(b"deflate\0");
    deflate_v1.extend_from_slice(&[6, 0, 0, 0, 0, 0, 0, 0]);
    assert!(
        std::fs::read(&path)
            .unwrap()
            .windows(deflate_v1.len())
            .any(|w| w == deflate_v1),
        "no version-1 deflate pipeline in the file"
    );

    read_with_h5py(
        py,
        &path,
        "for name in ('gz', 'sh', 'shgz'):\n\
         \x20   assert list(f[name][...]) == list(range(64)), (name, list(f[name][...]))\n\
         \x20   assert f[name].chunks == (8,), (name, f[name].chunks)\n\
         assert f['gz'].compression == 'gzip' and f['gz'].compression_opts == 6\n\
         assert f['sh'].shuffle is True and f['sh'].compression is None\n\
         assert f['shgz'].compression == 'gzip' and f['shgz'].shuffle is True\n",
    );
    libhdf5_tools_accept(py, &path);

    let back = H5File::open(&path).unwrap();
    assert_eq!(
        back.dataset("shgz").unwrap().read_raw::<i32>().unwrap(),
        ramp
    );
    drop(back);
    let _ = std::fs::remove_file(&path);
}

/// An unlimited dimension does not move the index: at this bound the
/// extensible array is not available, and libhdf5 grows the same version-1
/// B-tree instead. The rows are appended rather than written at once, so the
/// tree takes keys past the extent it was created with.
#[test]
fn an_unlimited_dataset_created_at_earliest_stays_on_the_version_1_btree() {
    let Some(py) = python() else { return };
    let path = tmp("unlimited");
    let file = earliest(&path);
    let ds = file
        .new_dataset::<i32>()
        .shape([0usize])
        .chunk(&[4])
        .max_shape(&[None])
        .create("growing")
        .unwrap();
    for start in [0i32, 4, 8, 12] {
        ds.append(&(start..start + 4).collect::<Vec<_>>()).unwrap();
    }
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    read_with_h5py(
        py,
        &path,
        "assert f['growing'].shape == (16,), f['growing'].shape\n\
         assert f['growing'].maxshape == (None,), f['growing'].maxshape\n\
         assert list(f['growing'][...]) == list(range(16)), list(f['growing'][...])\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// The userblock is the application's, and the format the file behind it is
/// in is the caller's: the two options compose, and the superblock is the
/// version-0 one at the end of the block.
#[test]
fn a_userblock_holds_a_file_created_at_earliest() {
    let Some(py) = python() else { return };
    let path = tmp("userblock");
    let file = H5File::options()
        .libver(LibverBound::Earliest)
        .userblock(512)
        .create(&path)
        .unwrap();
    file.new_dataset::<i32>()
        .shape([4])
        .create("alpha")
        .unwrap()
        .write_raw(&[1i32, 2, 3, 4])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version_at(&path, 512), 0);
    assert_eq!(&std::fs::read(&path).unwrap()[..8], &[0u8; 8]);
    read_with_h5py(
        py,
        &path,
        "assert list(f['alpha'][...]) == [1, 2, 3, 4]\n\
         assert f.userblock_size == 512, f.userblock_size\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// A file created at this bound reopens as what it is — the append path reads
/// the format off the superblock, not off the options the creator passed —
/// and comes back a version-0 file after two more sessions.
#[test]
fn a_file_created_at_earliest_reopens_and_takes_appends() {
    let Some(py) = python() else { return };
    let path = tmp("reopen");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([3])
        .create("alpha")
        .unwrap()
        .write_raw(&[1i32, 2, 3])
        .unwrap();
    file.close().unwrap();

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([2])
        .create("beta")
        .unwrap()
        .write_raw(&[4i32, 5])
        .unwrap();
    file.create_group("later").unwrap();
    file.close().unwrap();

    let file = H5File::open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([8])
        .chunk(&[4])
        .create("later/gamma")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'beta', 'later'], sorted(f.keys())\n\
         assert list(f['alpha'][...]) == [1, 2, 3]\n\
         assert list(f['beta'][...]) == [4, 5]\n\
         assert list(f['later/gamma'][...]) == list(range(8))\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// Shared object header messages override the bound. Their master table lives
/// in a superblock extension, which only a version-2 superblock has, so
/// `H5F__super_init` raises such a file to version 2 whatever low bound it
/// asked for (H5Fsuper.c:1135) — and a version-2 superblock is this crate's
/// modern file, link-message groups and all.
#[test]
fn shared_messages_raise_a_file_asking_for_earliest_to_version_2() {
    use rust_hdf5::format::messages::{MSG_DATASPACE, MSG_DATATYPE};
    use rust_hdf5::format::sohm::type_flag;

    let path = tmp("sohm");
    let types = type_flag(MSG_DATATYPE).unwrap() | type_flag(MSG_DATASPACE).unwrap();
    let file = H5File::options()
        .libver(LibverBound::Earliest)
        .shared_messages(&[(types, 0)], 50, 40)
        .create(&path)
        .unwrap();
    file.new_dataset::<i32>()
        .shape([4])
        .create("alpha")
        .unwrap()
        .write_raw(&[1i32, 2, 3, 4])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 2);
    assert!(contains(&path, b"OHDR"), "version-2 object headers");
    let _ = std::fs::remove_file(&path);
}

/// A virtual dataset is refused by name for the same reason SWMR is: nothing
/// older than the version-4 data layout message can say "virtual" at all, and
/// the layout message at this bound is version 3.
#[test]
fn a_virtual_dataset_in_a_file_created_at_earliest_is_refused() {
    use rust_hdf5::Selection;

    let path = tmp("virtual");
    let file = earliest(&path);
    let err = match file
        .new_dataset::<i32>()
        .shape([4])
        .virtual_mapping(Selection::All, "src.h5", "src", Selection::All)
        .create("mapped")
    {
        Ok(_) => panic!("a virtual dataset needs a layout message this file cannot hold"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("virtual"), "{err}");
    assert!(err.contains("version-4 data layout"), "{err}");
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    let _ = std::fs::remove_file(&path);
}

/// SWMR needs a version-3 superblock to record that a writer is attached, and
/// this file has a version-0 one. libhdf5 answers the combination by raising
/// the low bound to V110 and writing the newer file; this refuses it by name,
/// because the bound arrived as a request for the classic format and a
/// version-3 file would answer a different request than the one made.
#[test]
fn an_swmr_session_on_a_file_created_at_earliest_is_refused() {
    use rust_hdf5::swmr::SwmrFileWriter;

    let path = tmp("swmr");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([3])
        .create("alpha")
        .unwrap()
        .write_raw(&[1i32, 2, 3])
        .unwrap();
    file.close().unwrap();

    let mut writer = SwmrFileWriter::open_append(&path).unwrap();
    let err = writer.start_swmr().unwrap_err().to_string();
    assert!(err.contains("classic"), "{err}");
    assert!(err.contains("version-3"), "{err}");
    drop(writer);

    // Refused, not half-done: the file is the version-0 one it was.
    assert_eq!(superblock_version(&path), 0);
    let _ = std::fs::remove_file(&path);
}

/// The bound is opt-in. A file created without one is the v1.8-shaped file
/// this crate has always written — version-2 superblock, version-2 object
/// headers, link-message groups — and asking for `V18` explicitly does not
/// change that either.
#[test]
fn a_file_created_without_the_bound_is_unchanged() {
    for (label, bound) in [("no bound", None), ("V18", Some(LibverBound::V18))] {
        let path = tmp("unchanged");
        let file = match bound {
            None => H5File::create(&path).unwrap(),
            Some(bound) => H5File::options().libver(bound).create(&path).unwrap(),
        };
        file.new_dataset::<i32>()
            .shape([4])
            .create("alpha")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        file.create_group("outer").unwrap();
        file.close().unwrap();

        assert_eq!(superblock_version(&path), 2, "{label}");
        assert!(
            contains(&path, b"OHDR"),
            "{label}: version-2 object headers"
        );
        assert!(
            !contains(&path, b"SNOD"),
            "{label}: no symbol table node belongs in a v1.8 file"
        );
        let _ = std::fs::remove_file(&path);
    }
}
