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

use rust_hdf5::format::btree_v1::{BTreeV1Config, BTreeV1Node};
use rust_hdf5::format::local_heap::{local_heap_get_string, LocalHeapHeader};
use rust_hdf5::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage};
use rust_hdf5::format::messages::link::{CharacterSet, LinkMessage, LinkTarget};
use rust_hdf5::format::messages::{
    MSG_DATA_LAYOUT, MSG_FILL_VALUE, MSG_FILL_VALUE_OLD, MSG_FLAG_CONSTANT, MSG_LINK,
    MSG_SYMBOL_TABLE,
};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::superblock::{SuperblockV0V1, SuperblockV2V3};
use rust_hdf5::format::symbol_table::SymbolTableNode;
use rust_hdf5::format::FormatContext;
use rust_hdf5::{FillValue, H5File, LibverBound};

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

/// The one group holding an external link converts out of its symbol table,
/// and nothing else in the file moves with it.
///
/// A symbol table entry has three cache types and no room for a fourth, so
/// `H5G_obj_insert` answers an `H5L_TYPE_EXTERNAL` insert by giving that group
/// a Link Info and a Group Info message, re-inserting its entries as link
/// messages and dropping its Symbol Table message (H5Gobj.c:512). The
/// superblock stays version 0 and the headers stay version 1 — the conversion
/// is per group, not per file, which is what the sibling `plain` group and the
/// symbol-table root here pin.
#[test]
fn a_group_holding_an_external_link_converts_to_link_messages() {
    let Some(py) = python() else { return };
    let path = tmp("external_link");
    let target = path.with_file_name("external_link_target.h5");
    let payload = H5File::create(&target).unwrap();
    payload
        .new_dataset::<i32>()
        .shape([4])
        .create("payload")
        .unwrap()
        .write_raw(&[10i32, 11, 12, 13])
        .unwrap();
    payload.close().unwrap();

    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([4])
        .create("orig")
        .unwrap()
        .write_raw(&[0i32, 1, 2, 3])
        .unwrap();
    let plain = file.create_group("plain").unwrap();
    plain
        .new_dataset::<i32>()
        .shape([2])
        .create("beta")
        .unwrap()
        .write_raw(&[7i32, 8])
        .unwrap();
    let crossing = file.create_group("crossing").unwrap();
    crossing
        .new_dataset::<i32>()
        .shape([2])
        .create("gamma")
        .unwrap()
        .write_raw(&[4i32, 5])
        .unwrap();
    crossing
        .create_external_link("ext", "external_link_target.h5", "/payload")
        .unwrap();
    crossing.create_soft_link("shortcut", "/orig").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    // The conversion does not reach for the version-2 generation: no version-2
    // object header, and the two groups that kept their symbol table still
    // write one.
    no_newer_structures(&path);
    assert!(contains(&path, b"SNOD"), "the unconverted groups' tables");

    // A Symbol Table message is the observable — `H5Oget_info().hdr.mesg`
    // reports which message types an object header holds, and it is the same
    // bit libhdf5's own `H5G_STORAGE_TYPE_SYMBOL_TABLE` is derived from.
    read_with_h5py(
        py,
        &path,
        "STAB = 1 << 0x11\n\
         def stab(p):\n\
         \x20   return bool(h5py.h5o.get_info(f[p].id).hdr.mesg.present & STAB)\n\
         assert stab('/'), 'the root holds only hard links'\n\
         assert stab('/plain'), 'plain holds only a hard link'\n\
         assert not stab('/crossing'), 'crossing holds an external link'\n\
         for path in ('/', '/plain', '/crossing'):\n\
         \x20   v = h5py.h5o.get_info(f[path].id).hdr.version\n\
         \x20   assert v == 1, (path, v)\n\
         assert sorted(f['crossing'].keys()) == ['ext', 'gamma', 'shortcut']\n\
         link = f['crossing'].get('ext', getlink=True)\n\
         assert isinstance(link, h5py.ExternalLink), type(link)\n\
         assert link.filename == 'external_link_target.h5', link.filename\n\
         assert link.path == '/payload', link.path\n\
         assert list(f['crossing/ext'][...]) == [10, 11, 12, 13]\n\
         assert list(f['crossing/gamma'][...]) == [4, 5]\n\
         assert list(f['crossing/shortcut'][...]) == [0, 1, 2, 3]\n\
         assert list(f['plain/beta'][...]) == [7, 8]\n",
    );
    libhdf5_tools_accept(py, &path);

    // And this crate reads its own file back the same way.
    let reopened = H5File::open(&path).unwrap();
    let mut names = reopened
        .root_group()
        .group("crossing")
        .unwrap()
        .link_names()
        .unwrap();
    names.sort();
    assert_eq!(names, ["ext", "gamma", "shortcut"]);
    drop(reopened);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&target);
}

/// The other half of `H5G_obj_insert`'s conversion test: a link whose name is
/// not ASCII takes its group out of the symbol table exactly as an external
/// link does.
///
/// `obj_lnk->cset != H5T_CSET_ASCII || obj_lnk->type > H5L_TYPE_BUILTIN_MAX`
/// is one condition with two halves (H5Gobj.c:514), and the character set is
/// what a symbol table entry has no field for. h5py reaches the same branch
/// because it encodes a `str` name to ASCII when it can and to UTF-8 when it
/// cannot, and puts the answer in the lcpl (`CommonStateObject._e`) — so a
/// Rust `&str` and an h5py `str` produce the same file.
///
/// The conversion is per group: the root converts over its two non-ASCII
/// names, its ASCII siblings come along as link messages, and a subgroup whose
/// own children are ASCII-named keeps its table. Character sets are checked in
/// the written bytes as well, because libhdf5 reads a name the same either way
/// and would not complain about a wrong one.
#[test]
fn a_group_holding_a_non_ascii_name_converts_to_link_messages() {
    let Some(py) = python() else { return };
    let path = tmp("nonascii_names");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([4])
        .create("데이터")
        .unwrap()
        .write_raw(&[0i32, 1, 2, 3])
        .unwrap();
    file.create_group("plain").unwrap();
    for parent in ["그룹", "ascii_only"] {
        file.create_group(parent)
            .unwrap()
            .new_dataset::<i32>()
            .shape([2])
            .create("inner")
            .unwrap()
            .write_raw(&[7i32, 8])
            .unwrap();
    }
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    assert!(contains(&path, b"SNOD"), "the unconverted groups' tables");

    // Each root link message, decoded from the header this writer emitted:
    // the two non-ASCII names carry UTF-8 and the two ASCII ones carry no
    // character set field at all.
    let ctx = FormatContext {
        sizeof_addr: 8,
        sizeof_size: 8,
    };
    let mut seen: Vec<(String, CharacterSet)> = root_link_messages(&path, &ctx)
        .into_iter()
        .map(|(link, encoded)| {
            let has_cset_field = encoded[1] & 0x10 != 0;
            assert_eq!(
                has_cset_field,
                link.cset != CharacterSet::Ascii,
                "{}: the character set field is written exactly when the set is \
                 not the default one `H5O__link_decode` assumes",
                link.name
            );
            (link.name, link.cset)
        })
        .collect();
    seen.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        seen,
        [
            ("ascii_only".to_string(), CharacterSet::Ascii),
            ("plain".to_string(), CharacterSet::Ascii),
            ("그룹".to_string(), CharacterSet::Utf8),
            ("데이터".to_string(), CharacterSet::Utf8),
        ]
    );

    read_with_h5py(
        py,
        &path,
        "STAB = 1 << 0x11\n\
         def stab(p):\n\
         \x20   return bool(h5py.h5o.get_info(f[p].id).hdr.mesg.present & STAB)\n\
         assert not stab('/'), 'the root holds two non-ASCII names'\n\
         assert stab('/plain'), 'plain is empty and ASCII-named throughout'\n\
         assert stab('/그룹'), 'its own child is ASCII-named'\n\
         assert stab('/ascii_only'), 'ASCII throughout'\n\
         for path in ('/', '/plain', '/그룹', '/ascii_only'):\n\
         \x20   v = h5py.h5o.get_info(f[path].id).hdr.version\n\
         \x20   assert v == 1, (path, v)\n\
         assert sorted(f.keys()) == ['ascii_only', 'plain', '그룹', '데이터']\n\
         assert list(f['데이터'][...]) == [0, 1, 2, 3]\n\
         assert list(f['그룹/inner'][...]) == [7, 8]\n\
         assert list(f['ascii_only/inner'][...]) == [7, 8]\n",
    );
    libhdf5_tools_accept(py, &path);

    let reopened = H5File::open(&path).unwrap();
    let mut names = reopened.root_group().link_names().unwrap();
    names.sort();
    assert_eq!(names, ["ascii_only", "plain", "그룹", "데이터"]);
    drop(reopened);
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
    // bytes, then deflate with a name length of 8, the `H5Z_FLAG_OPTIONAL`
    // flags `H5Pset_deflate` asks for, one client-data value, the padded
    // name, the level and the pad that makes the value count even. A
    // version-2 message would name no filter below `H5Z_FILTER_RESERVED` at
    // all, so the name alone separates them — but the whole message is
    // checked, because the padding rules are the half that is easy to get
    // wrong.
    let mut deflate_v1 = vec![1u8, 1, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 1, 0, 1, 0];
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

/// The data layout message of the root-level dataset `name` in a *classic*
/// file, reached the way libhdf5 reaches it: the superblock's own root symbol
/// table entry caches the root group's B-tree and local heap, the B-tree's
/// leaves name symbol table nodes, and an entry's name offset points into the
/// heap. Every group here fits in one leaf level, so the walk is the leaf's
/// children rather than a descent.
fn classic_layout_of(path: &std::path::Path, name: &str) -> DataLayoutMessage {
    let (msg, ctx) = classic_layout_message_of(path, name);
    DataLayoutMessage::decode(&msg, &ctx).unwrap().0
}

/// The same message undecoded, for the one claim the decoded form cannot
/// carry: the version byte of a class whose variant does not record it.
fn classic_layout_version_of(path: &std::path::Path, name: &str) -> u8 {
    classic_layout_message_of(path, name).0[0]
}

fn classic_layout_message_of(path: &std::path::Path, name: &str) -> (Vec<u8>, FormatContext) {
    let (messages, ctx) = classic_messages_of(path, name);
    let msg = messages
        .iter()
        .find(|(msg_type, _, _)| *msg_type == MSG_DATA_LAYOUT)
        .unwrap_or_else(|| panic!("'{name}' has no data layout message"));
    (msg.2.clone(), ctx)
}

/// Every header message of a classic file's dataset, as `(type, flags, body)`.
fn classic_messages_of(
    path: &std::path::Path,
    name: &str,
) -> (Vec<(u8, u8, Vec<u8>)>, FormatContext) {
    let bytes = std::fs::read(path).unwrap();
    let sb = SuperblockV0V1::decode(&bytes).unwrap();
    let (addr_size, size_size) = (sb.sizeof_offsets as usize, sb.sizeof_lengths as usize);
    let ctx = FormatContext {
        sizeof_addr: sb.sizeof_offsets,
        sizeof_size: sb.sizeof_lengths,
    };
    let cfg = BTreeV1Config {
        sym_leaf_k: sb.sym_leaf_k,
        snode_internal_k: sb.btree_internal_k,
        ..Default::default()
    };
    let at = |addr: u64| (sb.base_address + addr) as usize;

    let (btree_addr, heap_addr) = sb
        .root_symbol_table_entry
        .cached_symbol_table()
        .expect("a classic root group caches its symbol table");
    let heap = LocalHeapHeader::decode(&bytes[at(heap_addr)..], addr_size, size_size).unwrap();
    let heap_data = &bytes[at(heap.data_addr)..at(heap.data_addr) + heap.data_size as usize];

    let node = BTreeV1Node::decode(
        &bytes[at(btree_addr)..],
        addr_size,
        size_size,
        cfg.snode_max_entries(),
    )
    .unwrap();
    assert_eq!(node.level, 0, "these groups fit in a leaf");

    let obj_addr = node
        .children
        .iter()
        .flat_map(|&child| {
            SymbolTableNode::decode(
                &bytes[at(child)..],
                addr_size,
                size_size,
                cfg.sym_leaf_max_entries(),
            )
            .unwrap()
            .entries
        })
        .find(|e| local_heap_get_string(heap_data, e.name_offset).unwrap() == name)
        .unwrap_or_else(|| panic!("no '{name}' in the root group's symbol table"))
        .obj_header_addr;

    let at = at(obj_addr);
    // A version-1 header carries no "OHDR" signature, so this is the arm
    // `decode_any` picks by its absence.
    let (header, _) = ObjectHeader::decode_v1(&bytes[at..]).unwrap();
    let messages = header
        .messages
        .iter()
        .map(|m| (m.msg_type, m.flags, m.data.clone()))
        .collect();
    (messages, ctx)
}

/// Every link message of a version-0 superblock's root group, paired with the
/// raw bytes it was decoded from.
///
/// The root symbol table entry names the object header whether or not the
/// group still has a symbol table, so it is the way in to a classic file whose
/// root `H5G_obj_insert` converted to link messages.
fn root_link_messages(path: &std::path::Path, ctx: &FormatContext) -> Vec<(LinkMessage, Vec<u8>)> {
    let bytes = std::fs::read(path).unwrap();
    let sb = SuperblockV0V1::decode(&bytes).unwrap();
    let at = (sb.base_address + sb.root_symbol_table_entry.obj_header_addr) as usize;
    let (root, _) = ObjectHeader::decode_v1(&bytes[at..]).unwrap();
    root.messages
        .iter()
        .filter(|m| m.msg_type == MSG_LINK)
        .map(|m| (LinkMessage::decode(&m.data, ctx).unwrap().0, m.data.clone()))
        .collect()
}

/// The same, for a file whose root group holds link messages.
fn modern_messages_of(path: &std::path::Path, name: &str) -> Vec<(u8, u8, Vec<u8>)> {
    let bytes = std::fs::read(path).unwrap();
    let sb = SuperblockV2V3::decode(&bytes).unwrap();
    let ctx = FormatContext {
        sizeof_addr: sb.sizeof_offsets,
        sizeof_size: sb.sizeof_lengths,
    };
    let at = (sb.base_address + sb.root_group_object_header_address) as usize;
    let (root, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    let addr = root
        .messages
        .iter()
        .filter(|m| m.msg_type == MSG_LINK)
        .filter_map(|m| LinkMessage::decode(&m.data, &ctx).ok())
        .find_map(|(l, _)| match l.target {
            LinkTarget::Hard { address } if l.name == name => Some(address),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no link '{name}' in the root group"));
    let (header, _) = ObjectHeader::decode(&bytes[(sb.base_address + addr) as usize..]).unwrap();
    header
        .messages
        .iter()
        .map(|m| (m.msg_type, m.flags, m.data.clone()))
        .collect()
}

/// The fill-value messages of `name`, in header order.
fn fill_messages(messages: Vec<(u8, u8, Vec<u8>)>) -> Vec<(u8, u8, Vec<u8>)> {
    messages
        .into_iter()
        .filter(|(msg_type, _, _)| *msg_type == MSG_FILL_VALUE || *msg_type == MSG_FILL_VALUE_OLD)
        .collect()
}

/// A user-defined fill value in a classic file is written twice: once in the
/// fill-value message (0x05) every generation writes, and once more in the
/// "fill value (old)" message (0x04) that predates it.
///
/// `H5D__update_oh_info` (H5Dint.c:1024-1035) appends the old message whenever
/// `fill_prop->buf` is set and `use_at_least_v18` — `H5F_LOW_BOUND(file) >=
/// H5F_LIBVER_V18` — is false, so that a reader too old to know the new
/// message still finds the value. It carries the size and the bytes and
/// nothing else (`H5O__fill_old_encode`, H5Ofill.c:512): the allocation time,
/// the write time and the defined flag have no place in it.
///
/// The new message stays at version 2 at this bound, which the version table
/// alone does not say — `H5O_fill_ver_bounds[H5F_LIBVER_EARLIEST]` is
/// `H5O_FILL_VERSION_1`, but `H5O__fill_set_version` takes the maximum of that
/// and `fill->version`, which the default creation property list has already
/// set to `H5O_FILL_VERSION_2` (H5Pdcpl.c:163). Version 1 is unreachable.
///
/// Every byte asserted here was read out of an h5py `libver='earliest'` file
/// by walking its version-1 object headers.
#[test]
fn a_classic_user_defined_fill_value_carries_the_old_message_too() {
    let path = tmp("classic_fill_old");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([4])
        .create("plain")
        .unwrap();
    file.new_dataset::<i32>()
        .shape([4])
        .fill_value(7i32)
        .create("withfill")
        .unwrap();
    file.close().unwrap();

    // No fill value of the dataset's own, so `fill_prop->buf` is NULL and the
    // old message is not written — h5py's `/plain` has the one message too.
    let plain = fill_messages(classic_messages_of(&path, "plain").0);
    assert_eq!(
        plain.iter().map(|m| m.0).collect::<Vec<_>>(),
        vec![MSG_FILL_VALUE],
        "a dataset with no fill value of its own gets the new message alone"
    );
    assert_eq!(plain[0].2[0], 2, "the classic fill message is version 2");

    let withfill = fill_messages(classic_messages_of(&path, "withfill").0);
    assert_eq!(
        withfill.iter().map(|m| m.0).collect::<Vec<_>>(),
        vec![MSG_FILL_VALUE, MSG_FILL_VALUE_OLD],
        "the old message follows the new one, as `H5D__update_oh_info` appends it"
    );
    assert_eq!(
        withfill[0].2,
        // Version 2, allocation time late, write time `H5D_FILL_TIME_IFSET`,
        // the defined flag, the size and the value; the last four bytes are
        // the version-1 header's eight-byte message alignment. Byte for byte
        // h5py's.
        vec![0x02, 0x02, 0x02, 0x01, 0x04, 0, 0, 0, 0x07, 0, 0, 0, 0, 0, 0, 0],
    );
    assert_eq!(
        withfill[1].1, MSG_FLAG_CONSTANT,
        "H5O_MSG_FLAG_CONSTANT, the flag `H5D__update_oh_info` passes"
    );
    assert_eq!(
        withfill[1].2,
        // The four-byte size and the four value bytes, byte-for-byte h5py's.
        vec![0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00],
    );

    // The same dataset at the default bounds: `use_at_least_v18` is true, so
    // the new message stands alone and is version 3.
    let modern = tmp("modern_fill_old");
    let file = H5File::create(&modern).unwrap();
    file.new_dataset::<i32>()
        .shape([4])
        .fill_value(7i32)
        .create("withfill")
        .unwrap();
    file.close().unwrap();
    let at_v18 = fill_messages(modern_messages_of(&modern, "withfill"));
    assert_eq!(
        at_v18.iter().map(|m| m.0).collect::<Vec<_>>(),
        vec![MSG_FILL_VALUE],
        "above the v1.8 bound the old message is not written at all"
    );
    assert_eq!(at_v18[0].2[0], 3, "and the new message is version 3");
    // The same content in version 3's packed flag byte: allocation time late
    // in bits 0-1, `H5D_FILL_TIME_IFSET` in bits 2-3, value defined in bit 5.
    // Byte for byte h5py's 0x2a.
    assert_eq!(at_v18[0].2, vec![0x03, 0x2a, 0x04, 0, 0, 0, 0x07, 0, 0, 0]);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&modern);
}

/// The read side of the dual fill-value message: two messages on disk, one
/// answer out of [`H5Dataset::fill_value`].
///
/// `H5O_FILL_ID` (0x05) and `H5O_FILL_ID_OLD` (0x04) carry the same four bytes
/// in a classic file, and the reader takes the new one — the only one that
/// also carries the allocation time, the write time and the defined flag the
/// accessor's three-way answer needs. So the old message must not add a second
/// user fill value, and must not shadow the new one into `Default`.
///
/// The oracle's `fillvalue` field is the same question asked of the same file
/// from h5py: `dcpl.get_fill_value()` renders as `0x` plus the raw bytes, so
/// the string this test builds from the accessor is the string `canon.py`
/// builds from libhdf5 — asserted here directly against h5py rather than
/// waiting for a full oracle run.
#[test]
fn a_classic_dual_fill_message_dataset_reads_back_one_user_fill_value() {
    let path = tmp("classic_fill_read");
    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([4])
        .create("plain")
        .unwrap();
    file.new_dataset::<i32>()
        .shape([4])
        .fill_value(7i32)
        .create("withfill")
        .unwrap();
    file.close().unwrap();

    // The premise: this dataset really does carry both messages.
    assert_eq!(
        fill_messages(classic_messages_of(&path, "withfill").0)
            .iter()
            .map(|m| m.0)
            .collect::<Vec<_>>(),
        vec![MSG_FILL_VALUE, MSG_FILL_VALUE_OLD],
    );

    let file = H5File::open(&path).unwrap();
    let fill = file.dataset("withfill").unwrap().fill_value().unwrap();
    assert_eq!(
        fill,
        FillValue::UserDefined(7i32.to_le_bytes().to_vec()),
        "the new message's value, once, not the old message's copy beside it"
    );
    // The canonical rendering `oracle/canon.py` compares against.
    let FillValue::UserDefined(bytes) = &fill else {
        unreachable!()
    };
    let canon: String = bytes.iter().fold("0x".to_string(), |mut s, b| {
        s.push_str(&format!("{b:02x}"));
        s
    });
    assert_eq!(canon, "0x07000000");
    assert_eq!(
        file.dataset("plain").unwrap().fill_value().unwrap(),
        FillValue::Default,
        "a dataset with no fill value of its own has no old message to mistake"
    );
    file.close().unwrap();

    if let Some(py) = python() {
        read_with_h5py(
            py,
            &path,
            "fv = f['withfill'].fillvalue\n\
             assert fv.tobytes().hex() == '07000000', fv.tobytes().hex()\n\
             assert f['plain'].fillvalue.tobytes().hex() == '00000000'\n\
             f.close()",
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// A fixed shape covered by exactly one chunk is the shape
/// `H5D__layout_set_latest_indexing` gives the single-chunk index, ahead of
/// every other v1.10 index. That rule is reached only after the layout
/// message version is settled, and at this bound the version is 3
/// (`H5O_layout_ver_bounds`), whose sole index is the version-1 B-tree —
/// which is also the only one a version-0 superblock can carry. So the
/// format decides before the shape does, and this pins the crossing from
/// both sides: the same one-chunk shape gets the B-tree in a classic file
/// and the single-chunk index in the default one.
#[test]
fn a_one_chunk_shape_created_at_earliest_takes_the_btree_not_the_single_chunk_index() {
    let vals: Vec<i32> = (0..16).collect();

    let classic = tmp("one_chunk_classic");
    let file = earliest(&classic);
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[16])
        .create("one")
        .unwrap()
        .write_raw(&vals)
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&classic), 0);
    no_newer_structures(&classic);
    let layout = classic_layout_of(&classic, "one");
    assert!(
        matches!(layout, DataLayoutMessage::ChunkedV3 { .. }),
        "a classic file's one-chunk dataset must stay on the version-3 \
         layout message and its version-1 B-tree, got {layout:?}"
    );
    assert!(
        H5File::open(&classic)
            .unwrap()
            .dataset("one")
            .unwrap()
            .read_raw::<i32>()
            .unwrap()
            == vals
    );

    // The default bounds, same shape: here the shape does decide, and the
    // index is the single-chunk one.
    let modern = tmp("one_chunk_modern");
    let file = H5File::create(&modern).unwrap();
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[16])
        .create("one")
        .unwrap()
        .write_raw(&vals)
        .unwrap();
    file.close().unwrap();

    let bytes = std::fs::read(&modern).unwrap();
    let sb = SuperblockV2V3::decode(&bytes).unwrap();
    let ctx = FormatContext {
        sizeof_addr: sb.sizeof_offsets,
        sizeof_size: sb.sizeof_lengths,
    };
    let at = (sb.base_address + sb.root_group_object_header_address) as usize;
    let (root, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    let addr = root
        .messages
        .iter()
        .filter(|m| m.msg_type == MSG_LINK)
        .filter_map(|m| LinkMessage::decode(&m.data, &ctx).ok())
        .find_map(|(l, _)| match l.target {
            LinkTarget::Hard { address } if l.name == "one" => Some(address),
            _ => None,
        })
        .expect("no link 'one' in the root group");
    let (header, _) = ObjectHeader::decode(&bytes[(sb.base_address + addr) as usize..]).unwrap();
    let msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_DATA_LAYOUT)
        .expect("'one' has no data layout message");
    let layout = DataLayoutMessage::decode(&msg.data, &ctx).unwrap().0;
    let DataLayoutMessage::ChunkedV4 { index_type, .. } = layout else {
        panic!("the default bounds give a version-4 layout message, got {layout:?}");
    };
    assert_eq!(index_type, ChunkIndexType::SingleChunk);

    let _ = std::fs::remove_file(&classic);
    let _ = std::fs::remove_file(&modern);
}

/// The same crossing reached from the other side: the classic file is one
/// libhdf5 wrote, and the one-chunk dataset is appended to it. The bound was
/// never asked for here — `is_legacy` reads it off the file — so this is the
/// path `tests/legacy_append.rs` covers, carrying the shape that would
/// otherwise select the single-chunk index.
#[test]
fn a_one_chunk_shape_appended_to_a_classic_file_takes_the_btree() {
    let Some(py) = python() else { return };
    let path = tmp("one_chunk_append");
    run(
        py,
        &[
            "-c",
            &format!(
                "import h5py, numpy as np\n\
                 with h5py.File({:?}, 'w', libver='earliest') as f:\n\
                 \x20   f['seed'] = np.arange(4, dtype='<i4')\n",
                path.to_str().unwrap()
            ),
        ],
        "h5py seed file",
    );
    assert_eq!(superblock_version(&path), 0);

    let vals: Vec<i32> = (0..16).collect();
    let file = H5File::options().no_locking().open_rw(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[16])
        .create("one")
        .unwrap()
        .write_raw(&vals)
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    // `no_newer_structures` cannot see this one: the single-chunk index
    // writes no signature of its own, keeping its one address inline in the
    // layout message. The message itself is the only witness.
    let layout = classic_layout_of(&path, "one");
    assert!(
        matches!(layout, DataLayoutMessage::ChunkedV3 { .. }),
        "a dataset appended to a classic file must stay on the version-3 \
         layout message and its version-1 B-tree, got {layout:?}"
    );
    read_with_h5py(
        py,
        &path,
        "assert f['one'].chunks == (16,), f['one'].chunks\n\
         assert list(f['one'][...]) == list(range(16)), list(f['one'][...])\n\
         assert list(f['seed'][...]) == [0, 1, 2, 3]\n",
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

/// Shared object header messages raise the superblock and nothing else.
///
/// Their master table lives in a superblock extension, which only a version-2
/// superblock has, so `H5F__super_init` raises such a file to version 2
/// whatever low bound it asked for (H5Fsuper.c:1135). What it does not do is
/// touch `H5F_LOW_BOUND`, and that is what every other rule reads: the groups
/// are still the symbol tables `H5G__obj_create_real` gives the earliest bound
/// (H5Gobj.c:179) and the messages are still version 1. The headers are
/// version 2, but from a third rule — sharing attribute messages means finding
/// one again by its creation index, so `H5SM_init` sets `store_msg_crt_idx`
/// (H5SM.c:220) and `H5O__create_ohdr` turns that into a version-2 header for
/// every object in the file (H5Oint.c:364).
///
/// So superblock, header and group format disagree in the one file, each
/// answering to its own rule. `tests/fixtures/sohm_list.h5`, written by
/// libhdf5 itself with a default fapl, is that file; the oracle compares
/// against it and this pins the three axes from inside.
#[test]
fn shared_messages_raise_only_the_superblock_of_a_file_asking_for_earliest() {
    use rust_hdf5::format::messages::{MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE};
    use rust_hdf5::format::sohm::type_flag;

    let Some(py) = python() else { return };
    let path = tmp("sohm");
    let types = type_flag(MSG_DATATYPE).unwrap()
        | type_flag(MSG_DATASPACE).unwrap()
        | type_flag(MSG_ATTRIBUTE).unwrap();
    let file = H5File::options()
        .libver(LibverBound::Earliest)
        .shared_messages(&[(types, 0)], 50, 40)
        .create(&path)
        .unwrap();
    for name in ["alpha", "beta"] {
        let ds = file
            .new_dataset::<i32>()
            .shape([4])
            .chunk(&[2])
            .create(name)
            .unwrap();
        ds.write_raw(&[1i32, 2, 3, 4]).unwrap();
        ds.new_attr::<f64>()
            .shape(())
            .create("gain")
            .unwrap()
            .write_numeric(&2.5f64)
            .unwrap();
    }
    file.create_group("inner").unwrap();
    file.close().unwrap();

    assert_eq!(
        superblock_version(&path),
        2,
        "the extension needs version 2"
    );
    for magic in [b"HEAP", b"SNOD", b"TREE"] {
        assert!(
            contains(&path, magic),
            "{} is missing from a symbol-table file",
            String::from_utf8_lossy(magic)
        );
    }
    // `no_newer_structures` is not the probe here: the shared-message table
    // keeps its own fractal heap, so `FRHP` is in this file legitimately and
    // the version-2 headers put `OHDR` in it too. What each group stores is
    // read from the group instead, below and through h5py.
    let bytes = std::fs::read(&path).unwrap();
    let sb = SuperblockV2V3::decode(&bytes).unwrap();
    let root = sb.root_group_object_header_address as usize;
    assert_eq!(&bytes[root..root + 4], b"OHDR", "version-2 object header");
    let stab = sohm_symbol_table_of(&bytes, root);
    let (btree_addr, heap_addr) = stab.expect("the root keeps its links in a symbol table");

    // The dataset the root's symbol table names: its header is version 2 for
    // the same reason the root's is, but its data layout message — a class no
    // index here shares, so it is still in the header to read — is the version
    // 3 `H5O_layout_ver_bounds` gives the earliest bound, over the version-1
    // B-tree that version's only chunk index. A file that had followed its
    // superblock into the modern generation would have version 4 here and one
    // of the v1.10 indexes behind it.
    let obj = sohm_named(&bytes, btree_addr, heap_addr, "alpha");
    assert_eq!(&bytes[obj..obj + 4], b"OHDR", "version-2 object header");
    let (header, _) = ObjectHeader::decode_any(&bytes[obj..]).unwrap();
    let layout = header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_DATA_LAYOUT)
        .expect("a dataset carries a data layout message");
    assert_eq!(layout.data[0], 3, "version-3 data layout message");

    // The subgroup answers to the bound too, not to the superblock its
    // sibling's shared messages raised.
    let inner = sohm_named(&bytes, btree_addr, heap_addr, "inner");
    assert!(
        sohm_symbol_table_of(&bytes, inner).is_some(),
        "a group created at the earliest bound keeps a symbol table"
    );

    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'beta', 'inner'], sorted(f.keys())\n\
         assert list(f['alpha'][...]) == [1, 2, 3, 4]\n\
         assert f['beta'].attrs['gain'] == 2.5\n\
         import h5py\n\
         for p in ['/', '/inner']:\n\
         \x20   present = h5py.h5o.get_info(f[p].id).hdr.mesg.present\n\
         \x20   assert present & (1 << 0x11), p\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// The Symbol Table message of the object header at `addr`: the B-tree and
/// local heap it names, or `None` when the header keeps its links elsewhere.
fn sohm_symbol_table_of(bytes: &[u8], addr: usize) -> Option<(u64, u64)> {
    let (header, _) = ObjectHeader::decode_any(&bytes[addr..]).unwrap();
    let msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_SYMBOL_TABLE)?;
    let addr_at = |at: usize| u64::from_le_bytes(msg.data[at..at + 8].try_into().unwrap());
    Some((addr_at(0), addr_at(8)))
}

/// The object header address the symbol table `(btree_addr, heap_addr)` gives
/// `name`. A version-2 superblock records no "K" ranks, so the node widths are
/// the library defaults `H5G_CRT_BTREE_RANK` sets.
fn sohm_named(bytes: &[u8], btree_addr: u64, heap_addr: u64, name: &str) -> usize {
    let cfg = BTreeV1Config::default();
    let heap = LocalHeapHeader::decode(&bytes[heap_addr as usize..], 8, 8).unwrap();
    let data = heap.data_addr as usize;
    let heap_data = &bytes[data..data + heap.data_size as usize];
    let node =
        BTreeV1Node::decode(&bytes[btree_addr as usize..], 8, 8, cfg.snode_max_entries()).unwrap();
    assert_eq!(node.level, 0, "this group fits in a leaf");
    node.children
        .iter()
        .flat_map(|&child| {
            SymbolTableNode::decode(&bytes[child as usize..], 8, 8, cfg.sym_leaf_max_entries())
                .unwrap()
                .entries
        })
        .find(|e| local_heap_get_string(heap_data, e.name_offset).unwrap() == name)
        .unwrap_or_else(|| panic!("no '{name}' in the symbol table"))
        .obj_header_addr as usize
}

/// A virtual dataset raises its own layout message to version 4 and takes
/// nothing else in the file with it.
///
/// The bound sets a floor, not a ceiling. `H5O_layout_ver_bounds` gives the
/// earliest bound version 1, and every other dataset here settles at the
/// version 3 that reaches; `H5D__virtual_construct` then raises this one
/// message on its own, because "virtual datasets require layout version 4",
/// and checks only the *high* bound while doing it (H5Dvirtual.c:2679). So the
/// classic file keeps its version-0 superblock, its version-1 object headers
/// and its symbol-table root over a version-4 layout message — which is what
/// h5py writes for the same file.
#[test]
fn a_virtual_dataset_in_a_file_created_at_earliest_raises_only_its_layout_message() {
    use rust_hdf5::format::messages::data_layout::DataLayoutMessage;
    use rust_hdf5::Selection;

    let Some(py) = python() else { return };
    let path = tmp("virtual");
    let source = path.with_file_name("virtual_source.h5");
    let src = earliest(&source);
    src.new_dataset::<i32>()
        .shape([16])
        .create("src")
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    src.close().unwrap();

    let file = earliest(&path);
    file.new_dataset::<i32>()
        .shape([16])
        .virtual_mapping(
            Selection::All,
            source.file_name().unwrap().to_str().unwrap(),
            "src",
            Selection::All,
        )
        .create("mapped")
        .unwrap();
    // The control: a contiguous dataset in the same file stays at version 3.
    file.new_dataset::<i32>()
        .shape([4])
        .create("plain")
        .unwrap()
        .write_raw(&[1i32, 2, 3, 4])
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 0);
    no_newer_structures(&path);
    match classic_layout_of(&path, "mapped") {
        DataLayoutMessage::Virtual { version, .. } => assert_eq!(version, 4),
        other => panic!("expected a virtual layout, got {other:?}"),
    }
    let plain = classic_layout_of(&path, "plain");
    assert!(
        matches!(plain, DataLayoutMessage::Contiguous { .. }),
        "expected a contiguous layout, got {plain:?}"
    );
    assert_eq!(
        classic_layout_version_of(&path, "plain"),
        3,
        "the neighbour is untouched by the virtual dataset's raise"
    );

    read_with_h5py(
        py,
        &path,
        "assert f['mapped'].is_virtual\n\
         assert list(f['mapped'][...]) == list(range(16)), list(f['mapped'][...])\n\
         (vspace, fname, dname, sspace), = f['mapped'].virtual_sources()\n\
         assert fname == 'virtual_source.h5', fname\n\
         assert dname == 'src', dname\n\
         assert list(f['plain'][...]) == [1, 2, 3, 4]\n\
         STAB = 1 << 0x11\n\
         assert h5py.h5o.get_info(f['/'].id).hdr.mesg.present & STAB, 'symbol-table root'\n\
         for path in ('/', '/mapped', '/plain'):\n\
         \x20   v = h5py.h5o.get_info(f[path].id).hdr.version\n\
         \x20   assert v == 1, (path, v)\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&source);
}

/// SWMR needs a version-3 superblock to record that a writer is attached, and
/// this file has a version-0 one. Reopened here, so it is
/// `H5F__start_swmr_write`'s own first check that answers — refuse below
/// version 3 (H5Fint.c:3814) — and the refusal names the version it found.
/// libhdf5 refuses the same call the same way; the one place it upgrades
/// instead is SWMR asked for at *create* time, which is not this.
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
    assert!(err.contains("superblock is version 0"), "{err}");
    assert!(err.contains("version-3"), "{err}");
    drop(writer);

    // Refused, not half-done: the file is the version-0 one it was.
    assert_eq!(superblock_version(&path), 0);
    let _ = std::fs::remove_file(&path);
}

/// The bound is opt-in. A file created without one keeps the shape this
/// crate has always written — version-2 superblock, version-2 object headers,
/// link-message groups — which is also what asking for `V18` gives, these
/// being the rows the two agree on. Where they part is the chunk index, which
/// `tests/libver_v18.rs` covers; nothing here is chunked.
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
