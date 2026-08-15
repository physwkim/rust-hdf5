//! Every wave-3 finalize-time subsystem, in one file each.
//!
//! `finalize` runs in phases: plan, allocate every object header, build the
//! content those headers name, write the headers, write the superblock. Each
//! subsystem that produces header content has exactly one phase it belongs in,
//! and the phases are not independent — the shared-message table counts the
//! bodies the headers hold, and an attribute holding an object reference is
//! not that body until the allocation phase has run. `check_header_size`
//! refuses a header whose body does not fill the block that was reserved for
//! it, so a subsystem in the wrong phase fails the close rather than writing a
//! file that disagrees with itself.
//!
//! The two files here are the composition: everything a modern file can carry
//! at once, and the classic file's version-1 chunk B-tree beside it — which
//! cannot be the same file, because a shared-message table needs a version-2
//! superblock (H5Fsuper.c:1135) that a classic file does not have.

use rust_hdf5::format::messages::{MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE};
use rust_hdf5::format::object_header::{ObjectHeader, OCHK_SIGNATURE};
use rust_hdf5::format::sohm::type_flag;
use rust_hdf5::format::superblock::SuperblockV2V3;
use rust_hdf5::{DatatypeMessage, H5File};

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation` and `legacy_append`.
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
            eprintln!("skipping finalize-phase cross-check: none of {candidates:?} present");
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
        "rust_hdf5_phases_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

fn run(program: &std::path::Path, args: &[&str], what: &str) {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Read `path` back with h5py; `body` runs with the file open as `f`.
fn read_with_h5py(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np\nf = h5py.File(r'{}', 'r')\n{}\n",
        path.display(),
        body
    );
    run(std::path::Path::new(py), &["-c", &script], "h5py read-back");
}

/// `h5dump` walks every structure a header names — the continuation chunk, the
/// dense attribute heap, the shared-message heap, the chunk index — so it
/// fails on a file whose phases disagreed even where h5py would not look.
fn h5dump_accepts(py: &str, path: &std::path::Path) {
    let Some(tool) = h5_tool(py, "h5dump") else {
        eprintln!("skipping h5dump: not in the interpreter's bin/");
        return;
    };
    run(&tool, &[path.to_str().unwrap()], "h5dump");
}

/// A 256-byte fixed-length ASCII attribute, far past what a group header's
/// creation-time estimate covers.
fn note(i: u8) -> (DatatypeMessage, Vec<u8>) {
    let mut text = vec![b'x'; 256];
    text[0] = b'0' + i;
    text[255] = 0;
    (
        DatatypeMessage::FixedString {
            size: 256,
            padding: 0,
            charset: 0,
        },
        text,
    )
}

/// The root group's object header, and every continuation chunk it names.
fn root_header(path: &std::path::Path) -> (ObjectHeader, Vec<(u64, u64)>) {
    let bytes = std::fs::read(path).unwrap();
    let superblock = SuperblockV2V3::decode(&bytes).unwrap();
    let at = (superblock.base_address + superblock.root_group_object_header_address) as usize;
    let (header, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    let chunks = header
        .messages
        .iter()
        .filter(|m| m.msg_type == rust_hdf5::format::messages::MSG_OBJ_HEADER_CONTINUATION)
        .map(|m| {
            (
                u64::from_le_bytes(m.data[..8].try_into().unwrap()),
                u64::from_le_bytes(m.data[8..16].try_into().unwrap()),
            )
        })
        .collect();
    (header, chunks)
}

/// Shared messages, a root header that spills into a continuation chunk, dense
/// attribute storage and an attribute whose value is an object header address,
/// all in one finalize.
///
/// The last two are what make this more than the sum of its parts. A shared
/// attribute body is counted once and written once, so the pass that counts it
/// has to see the same bytes the header ends up holding — which for a
/// reference attribute means after every object has an address, and for a
/// spilled one means after the phase change that decides whether the header
/// holds the attribute at all. Get either wrong and the body counted is not
/// the body written, the header no longer fills the block reserved for it, and
/// `check_header_size` fails this close.
#[test]
fn shared_messages_dense_attributes_and_a_spilling_root_finalize_together() {
    let path = tmp("composed");
    let types = type_flag(MSG_DATATYPE).unwrap()
        | type_flag(MSG_DATASPACE).unwrap()
        | type_flag(MSG_ATTRIBUTE).unwrap();
    let file = H5File::options()
        .shared_messages(&[(types, 0)], 50, 40)
        .create(&path)
        .unwrap();

    // Six datasets of one shape and one type, each carrying the same
    // attribute: every one of those bodies is shared, so each reaches the file
    // once and each header holds a pointer to it.
    let datasets: Vec<_> = (0..6i32)
        .map(|i| {
            let ds = file
                .new_dataset::<i32>()
                .shape([8usize])
                .create(&format!("shared{i}"))
                .unwrap();
            ds.write_raw(&(0..8i32).map(|j| i * 10 + j).collect::<Vec<_>>())
                .unwrap();
            ds.new_attr::<f64>()
                .create("units")
                .unwrap()
                .write_numeric(&1.5f64)
                .unwrap();
            ds
        })
        .collect();

    // A group whose attributes pass the phase change: they leave the header
    // for a fractal heap, so they are not in any header for an index to share.
    let dense = file.create_group("dense").unwrap();
    for i in 0..12 {
        dense
            .set_attr_numeric(&format!("d{i}"), &(i as i64))
            .unwrap();
    }

    // An attribute whose value is an object header address, which exists only
    // once the allocation phase has run.
    let neighbours = datasets[0]
        .new_attr::<u64>()
        .shape([2usize])
        .create("neighbours")
        .unwrap();
    neighbours
        .write_object_references(&["/shared1", "/dense"])
        .unwrap();

    // Six 256-byte attributes on the root: past its header's creation-time
    // estimate, so its messages spill into a continuation chunk.
    for i in 0..6u8 {
        let (dt, text) = note(i);
        file.set_attr_typed(&format!("note{i}"), dt, text).unwrap();
    }

    drop(datasets);
    file.close().unwrap();

    // The root really did spill, and the spilled block really is an OCHK.
    let (_, chunks) = root_header(&path);
    assert_eq!(chunks.len(), 1, "the root group spills: {chunks:?}");
    let bytes = std::fs::read(&path).unwrap();
    let (addr, _) = chunks[0];
    assert_eq!(&bytes[addr as usize..addr as usize + 4], &OCHK_SIGNATURE);
    // The shared-message table is published in the superblock extension.
    let superblock = SuperblockV2V3::decode(&bytes).unwrap();
    assert_ne!(
        superblock.superblock_extension_address,
        rust_hdf5::format::UNDEF_ADDR,
        "a file with shared messages names its superblock extension"
    );

    // This crate reads back everything the four subsystems wrote.
    let back = H5File::open(&path).unwrap();
    for i in 0..6i32 {
        let data: Vec<i32> = back
            .dataset(&format!("shared{i}"))
            .unwrap()
            .read_raw()
            .unwrap();
        assert_eq!(data, (0..8i32).map(|j| i * 10 + j).collect::<Vec<_>>());
    }
    let mut notes = back.attr_names().unwrap();
    notes.sort();
    assert_eq!(
        notes,
        (0..6).map(|i| format!("note{i}")).collect::<Vec<_>>()
    );
    let dense_back = back.root_group().group("dense").unwrap();
    let mut dense_names = dense_back.attr_names().unwrap();
    dense_names.sort();
    assert_eq!(dense_names.len(), 12, "{dense_names:?}");
    drop(back);

    let Some(py) = python() else {
        let _ = std::fs::remove_file(&path);
        return;
    };
    read_with_h5py(
        py,
        &path,
        "names = sorted(f.keys())\n\
         assert names == ['dense'] + ['shared%d' % i for i in range(6)], names\n\
         assert sorted(f.attrs.keys()) == ['note%d' % i for i in range(6)], sorted(f.attrs.keys())\n\
         assert len(f['dense'].attrs) == 12, len(f['dense'].attrs)\n\
         assert f['shared3'].attrs['units'] == 1.5\n\
         refs = f['shared0'].attrs['neighbours']\n\
         assert f[refs[0]].name == '/shared1', f[refs[0]].name\n\
         assert f[refs[1]].name == '/dense', f[refs[1]].name\n\
         assert list(f['shared5'][...]) == [50, 51, 52, 53, 54, 55, 56, 57]\n",
    );
    h5dump_accepts(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// The classic sibling: the version-1 chunk B-tree is built in the flush phase
/// that runs before the plan, because the layout message in the header names
/// the tree's root node. A shared-message table cannot join it — that is what
/// makes this a second file rather than a second dataset.
#[test]
fn a_classic_sibling_carries_its_chunks_in_a_version_one_btree() {
    let Some(py) = python() else { return };
    let path = tmp("classic");
    // h5py with no libver argument writes a version-0 superblock.
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'w')\n\
         f['alpha'] = np.arange(6, dtype='<i4')\n\
         f.close()\n",
        path.display()
    );
    run(std::path::Path::new(py), &["-c", &script], "h5py write");

    let file = H5File::open_rw(&path).unwrap();
    // 3 x 2 chunks over a 10 x 7 extent: both edges hang past it, which is
    // where the tree's keys and right bound are easiest to get wrong.
    let plane: Vec<i32> = (0..70).collect();
    file.new_dataset::<i32>()
        .shape([10, 7])
        .chunk(&[4, 4])
        .create("plane")
        .unwrap()
        .write_raw(&plane)
        .unwrap();
    file.set_attr_numeric("stamp", &7i64).unwrap();
    file.close().unwrap();

    // Still classic: the append did not drag the file to a newer superblock.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes[8], 0, "superblock version");

    let back = H5File::open(&path).unwrap();
    assert_eq!(
        back.dataset("plane").unwrap().read_raw::<i32>().unwrap(),
        plane
    );
    drop(back);

    read_with_h5py(
        py,
        &path,
        "assert sorted(f.keys()) == ['alpha', 'plane'], sorted(f.keys())\n\
         assert f['plane'].chunks == (4, 4), f['plane'].chunks\n\
         assert (f['plane'][...] == np.arange(70).reshape(10, 7)).all()\n\
         assert f.attrs['stamp'] == 7\n",
    );
    h5dump_accepts(py, &path);
    let _ = std::fs::remove_file(&path);
}
