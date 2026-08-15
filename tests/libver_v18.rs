//! Creating a file at `H5F_LIBVER_V18`.
//!
//! The v1.8 bound is not "the modern file, one notch older". It is one row of
//! every message-version table at once, and the rows disagree about which
//! generation they belong to: `HDF5_superblock_ver_bounds` gives version 2,
//! `H5O_obj_ver_bounds` version-2 object headers and `H5G__obj_create_real`
//! link-message groups — all 1.8 structures — while `H5O_layout_ver_bounds`
//! still gives the *version-3* data layout message, which has no index-type
//! field at all. So a chunked dataset in such a file goes on the version-1
//! B-tree, the same index a classic file uses, under a superblock and object
//! headers a classic file cannot carry.
//!
//! That is the crossing this file exists for. `H5D__chunk_set_info` reaches
//! the index-selection block only once the layout version is 4 or more
//! (H5Dchunk.c:936), so the bound decides before the dataspace does: a fixed
//! shape covered by exactly one chunk takes the single-chunk index at the
//! crate default and the version-1 B-tree here, from the same builder call.
//! `tests/libver_earliest.rs` pins the same crossing at the other bound,
//! where the whole file is classic.
//!
//! The referee is libhdf5 itself: h5py opening the file back with
//! `libver=('v108','v108')` refuses any superblock above version 2, so an
//! opened file *is* the proof that nothing in it asked for the v1.10
//! generation.

use rust_hdf5::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage};
use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
use rust_hdf5::format::messages::{MSG_DATA_LAYOUT, MSG_LINK};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::superblock::SuperblockV2V3;
use rust_hdf5::format::FormatContext;
use rust_hdf5::{H5File, LibverBound};

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset,
/// matching `libver_earliest`. `h5dump` and `h5clear` are taken from the same
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
            eprintln!("skipping v1.8-bound cross-check: none of {candidates:?} present");
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
        "rust_hdf5_v18_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

fn v18(path: &std::path::Path) -> H5File {
    H5File::options()
        .libver(LibverBound::V18)
        .create(path)
        .unwrap()
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

/// Read `path` back with h5py *under the same bounds it was written at*.
///
/// The bounds are the assertion, not decoration: libhdf5 refuses to open a
/// file whose superblock version is above the one `HDF5_superblock_ver_bounds`
/// gives the high bound, so `libver=('v108','v108')` rejects the version-3
/// superblock a v1.10 chunk index used to drag in ("superblock version exceeds
/// high bound", observed on libhdf5 1.14.6). A file that opens here is a file
/// nothing in which asked for the newer generation.
fn read_at_v108(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nf = h5py.File(r'{}', 'r', libver=('v108', 'v108'))\n{}\n",
        path.display(),
        body
    );
    run(py, &["-c", &script], "h5py read-back at libver v108");
}

/// `h5dump` walks every object header and `h5clear -s` reads the superblock's
/// consistency flags: between them they judge the whole file rather than one
/// object.
fn libhdf5_tools_accept(py: &str, path: &std::path::Path) {
    let path = path.to_str().unwrap();
    if let Some(h5dump) = h5_tool(py, "h5dump") {
        run(h5dump, &["-pBH", path], "h5dump");
    }
    if let Some(h5clear) = h5_tool(py, "h5clear") {
        run(h5clear, &["-s", path], "h5clear -s");
    }
}

fn superblock_version(path: &std::path::Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n", "{}", path.display());
    bytes[8]
}

fn contains(path: &std::path::Path, magic: &[u8; 4]) -> bool {
    std::fs::read(path)
        .unwrap()
        .windows(4)
        .any(|w| w == magic.as_slice())
}

/// The data layout message of the root-level dataset `name` in a file with a
/// version-2/3 superblock, reached the way libhdf5 reaches it: the root group
/// object header holds a Link message per child, naming the child's own
/// header address.
fn layout_of(path: &std::path::Path, name: &str) -> DataLayoutMessage {
    let bytes = std::fs::read(path).unwrap();
    let sb = SuperblockV2V3::decode(&bytes).unwrap();
    let ctx = FormatContext {
        sizeof_addr: sb.sizeof_offsets,
        sizeof_size: sb.sizeof_lengths,
    };
    let at = |addr: u64| (sb.base_address + addr) as usize;
    let (root, _) =
        ObjectHeader::decode(&bytes[at(sb.root_group_object_header_address)..]).unwrap();
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
    let (header, _) = ObjectHeader::decode(&bytes[at(addr)..]).unwrap();
    let msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_DATA_LAYOUT)
        .unwrap_or_else(|| panic!("'{name}' has no data layout message"));
    DataLayoutMessage::decode(&msg.data, &ctx).unwrap().0
}

/// The whole generation in one file: a version-2 superblock over version-2
/// object headers and link-message groups, with the version-1 chunk B-tree
/// under them and no v1.10 index anywhere.
#[test]
fn a_file_created_at_v18_is_the_v18_generation() {
    let path = tmp("generation");
    let file = v18(&path);
    file.new_dataset::<i32>()
        .shape([8usize])
        .chunk(&[4])
        .create("grid")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.create_group("outer").unwrap();
    file.set_attr_string("made_by", "rust-hdf5").unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 2);
    // The version-2 object header signature and the version-1 B-tree node
    // signature: the two generations this bound puts in one file.
    for magic in [b"OHDR", b"TREE"] {
        assert!(
            contains(&path, magic),
            "{} is missing from a file created at H5F_LIBVER_V18",
            String::from_utf8_lossy(magic)
        );
    }
    // The v1.10 chunk indexes, none of which a version-3 layout message can
    // name, and the symbol-table group this bound left behind.
    for magic in [b"EAHD", b"FAHD", b"BTHD", b"SNOD"] {
        assert!(
            !contains(&path, magic),
            "{} appears in a file created at H5F_LIBVER_V18",
            String::from_utf8_lossy(magic)
        );
    }
    assert!(matches!(
        layout_of(&path, "grid"),
        DataLayoutMessage::ChunkedV3 { .. }
    ));

    let vals = H5File::open(&path)
        .unwrap()
        .dataset("grid")
        .unwrap()
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(vals, (0..8i32).collect::<Vec<_>>());
    let _ = std::fs::remove_file(&path);
}

/// The crossing. A fixed shape covered by exactly one chunk is the shape
/// `H5D__chunk_set_info` gives the single-chunk index ahead of every other
/// v1.10 index — but only inside the block the layout version guards. At the
/// v1.8 bound the version is 3 and the block is never entered, so the same
/// builder call that yields the single-chunk index at the crate default
/// yields the version-1 B-tree here.
#[test]
fn a_one_chunk_shape_created_at_v18_takes_the_btree_not_the_single_chunk_index() {
    let vals: Vec<i32> = (0..16).collect();

    let bound = tmp("one_chunk_v18");
    let file = v18(&bound);
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[16])
        .create("one")
        .unwrap()
        .write_raw(&vals)
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&bound), 2);
    assert!(
        matches!(
            layout_of(&bound, "one"),
            DataLayoutMessage::ChunkedV3 { .. }
        ),
        "a v1.8 file's one-chunk dataset must stay on the version-3 layout \
         message and its version-1 B-tree, got {:?}",
        layout_of(&bound, "one")
    );
    // The B-tree is really there: the single-chunk index writes no signature
    // of its own, so its absence alone would not tell the two apart.
    assert!(contains(&bound, b"TREE"), "no version-1 B-tree node");

    // The default bounds, same shape: here the shape does decide.
    let default = tmp("one_chunk_default");
    let file = H5File::create(&default).unwrap();
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[16])
        .create("one")
        .unwrap()
        .write_raw(&vals)
        .unwrap();
    file.close().unwrap();

    let DataLayoutMessage::ChunkedV4 { index_type, .. } = layout_of(&default, "one") else {
        panic!("the default bounds give a version-4 layout message");
    };
    assert_eq!(index_type, ChunkIndexType::SingleChunk);

    for path in [&bound, &default] {
        assert_eq!(
            H5File::open(path)
                .unwrap()
                .dataset("one")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vals
        );
        let _ = std::fs::remove_file(path);
    }
}

/// Every index the shape could select is the version-1 B-tree at this bound:
/// the two unlimited dimensions that reach the v2 B-tree, the one that
/// reaches the extensible array, and the fixed shape wider than its chunk
/// that reaches the fixed array. One case per arm of the selection this bound
/// short-circuits.
#[test]
fn every_shape_takes_the_btree_at_v18() {
    for (label, shape, chunk, max) in [
        ("bt2", vec![4usize, 4], vec![2usize, 2], vec![None, None]),
        ("earray", vec![16usize], vec![4usize], vec![None]),
        ("farray", vec![16usize], vec![4usize], vec![Some(16usize)]),
        ("implicit", vec![16usize], vec![4usize], vec![Some(16usize)]),
    ] {
        let path = tmp(label);
        let file = v18(&path);
        let mut builder = file
            .new_dataset::<i32>()
            .shape(shape.clone())
            .chunk(&chunk)
            .max_shape(&max);
        if label == "implicit" {
            builder = builder.early_allocation();
        }
        let n: usize = shape.iter().product();
        builder
            .create("data")
            .unwrap()
            .write_raw(&(0..n as i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();

        assert_eq!(superblock_version(&path), 2, "{label}");
        assert!(
            matches!(
                layout_of(&path, "data"),
                DataLayoutMessage::ChunkedV3 { .. }
            ),
            "{label}: {:?}",
            layout_of(&path, "data")
        );
        let _ = std::fs::remove_file(&path);
    }
}

/// libhdf5 reads the file back under the bounds it was written at. This is
/// the referee for everything above: `libver=('v108','v108')` is a high bound
/// of v1.8, and libhdf5 refuses a superblock above that bound's row, so the
/// open succeeding is the on-disk statement that the file stayed inside the
/// generation it asked for.
#[test]
fn h5py_reads_a_v18_file_back_at_the_same_bound() {
    let Some(py) = python() else { return };
    let path = tmp("h5py");
    let file = v18(&path);
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[16])
        .create("one")
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    file.new_dataset::<i32>()
        .shape([16usize])
        .chunk(&[4])
        .create("many")
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    file.create_group("outer").unwrap();
    file.set_attr_string("made_by", "rust-hdf5").unwrap();
    file.close().unwrap();

    read_at_v108(
        py,
        &path,
        "expected = np.arange(16, dtype='<i4')\n\
         for name in ('one', 'many'):\n\
         \x20   ds = f[name]\n\
         \x20   assert np.array_equal(ds[...], expected), (name, ds[...])\n\
         \x20   assert ds.chunks is not None, name\n\
         assert 'outer' in f, list(f)\n\
         assert f.attrs['made_by'] == 'rust-hdf5', dict(f.attrs)\n",
    );
    libhdf5_tools_accept(py, &path);
    let _ = std::fs::remove_file(&path);
}
