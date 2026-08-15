//! Reopening a file never changes its superblock version, and the version it
//! already has is what the appended structures are written at.
//!
//! `H5F__super_init` is the only place libhdf5 decides a superblock version
//! (H5Fsuper.c:1154). `H5F__super_read` validates the version it finds and
//! never recomputes one; what it does instead is raise the file's *low*
//! library-version bound to the row that version belongs to — version 2 to at
//! least `H5F_LIBVER_V18`, version 3 to at least `H5F_LIBVER_V110`
//! (hdf5_1.14.6 H5Fsuper.c:460-466). So a reopen with no bound named writes
//! the generation the file already is: the v1.8 row's version-3 data layout
//! message over the version-1 chunk B-tree for a version-2 superblock, and the
//! v1.10 row's version-4 message over one of the v1.10 indexes for a version-3
//! one.
//!
//! The two halves are one rule. A superblock version that is never re-decided
//! is only meaningful if nothing appended can need a newer one, and nothing
//! can because the bound the appends are written at is floored — not raised —
//! by that version.
//!
//! Checked here for all three generations a reopen can find: version 0
//! (classic), version 2 (v1.8) and version 3 (v1.10). Each case reads the
//! superblock version byte before and after the append and decodes the
//! appended dataset's own data layout message out of the file.

use rust_hdf5::format::btree_v1::{BTreeV1Config, BTreeV1Node};
use rust_hdf5::format::local_heap::{local_heap_get_string, LocalHeapHeader};
use rust_hdf5::format::messages::data_layout::DataLayoutMessage;
use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
use rust_hdf5::format::messages::{MSG_DATA_LAYOUT, MSG_LINK};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::superblock::{SuperblockV0V1, SuperblockV2V3};
use rust_hdf5::format::symbol_table::SymbolTableNode;
use rust_hdf5::format::FormatContext;
use rust_hdf5::{H5File, LibverBound};

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_reopen_bounds_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The version byte that follows the 8-byte signature, in either superblock
/// image — the one field both generations put in the same place.
fn superblock_version(path: &Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let at = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("no HDF5 signature");
    bytes[at + 8]
}

/// The encoded data layout message of the root-level dataset `name`, reached
/// the way libhdf5 reaches it: through the root group's symbol table in a
/// classic file, and through its Link messages in a version-2/3 one.
fn layout_message_of(path: &Path, name: &str) -> (Vec<u8>, FormatContext) {
    let bytes = std::fs::read(path).unwrap();
    if superblock_version(path) <= 1 {
        classic_layout_message_of(&bytes, name)
    } else {
        modern_layout_message_of(&bytes, name)
    }
}

/// The version byte of that message — the one claim the decoded form cannot
/// carry, a version-3 chunked layout and a version-1 one decoding to variants
/// that do not record it.
fn layout_version_of(path: &Path, name: &str) -> u8 {
    layout_message_of(path, name).0[0]
}

fn layout_of(path: &Path, name: &str) -> DataLayoutMessage {
    let (msg, ctx) = layout_message_of(path, name);
    DataLayoutMessage::decode(&msg, &ctx).unwrap().0
}

fn modern_layout_message_of(bytes: &[u8], name: &str) -> (Vec<u8>, FormatContext) {
    let sb = SuperblockV2V3::decode(bytes).unwrap();
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
    (layout_body(&header, name), ctx)
}

fn classic_layout_message_of(bytes: &[u8], name: &str) -> (Vec<u8>, FormatContext) {
    let sb = SuperblockV0V1::decode(bytes).unwrap();
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
    assert_eq!(node.level, 0, "this root group fits in a leaf");
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
    let (header, _) = ObjectHeader::decode_v1(&bytes[at(obj_addr)..]).unwrap();
    (layout_body(&header, name), ctx)
}

fn layout_body(header: &ObjectHeader, name: &str) -> Vec<u8> {
    header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_DATA_LAYOUT)
        .unwrap_or_else(|| panic!("'{name}' has no data layout message"))
        .data
        .clone()
}

/// Create a file of one generation, holding one contiguous dataset.
fn create_generation(path: &Path, libver: Option<LibverBound>) {
    let mut options = H5File::options();
    if let Some(libver) = libver {
        options = options.libver(libver);
    }
    let file = options.create(path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create("data")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// Reopen with no bound named and append one chunked dataset — the append
/// whose chunk index the file's own generation decides.
fn reopen_and_append_chunked(path: &Path, name: &str) {
    let file = H5File::open_rw(path).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize, 4])
        .chunk(&[2, 2])
        .create(name)
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// The whole invariant for one generation: the superblock version byte is the
/// same before and after, and the appended dataset's layout message is the one
/// that version's row of `H5O_layout_ver_bounds` names.
fn reopen_keeps_generation(label: &str, libver: Option<LibverBound>, expect_sb: u8) -> PathBuf {
    let path = tmp(label);
    create_generation(&path, libver);
    let before = superblock_version(&path);
    assert_eq!(
        before, expect_sb,
        "{label}: the file was not created in the generation this case is about"
    );

    reopen_and_append_chunked(&path, "appended");

    assert_eq!(
        superblock_version(&path),
        before,
        "{label}: reopening and appending changed the superblock version from \
         {before} to {}; H5F__super_read never re-decides it",
        superblock_version(&path)
    );
    path
}

/// A version-0 superblock: the classic generation, whose only chunk index is
/// the version-1 B-tree behind a version-3 layout message.
#[test]
fn reopening_a_classic_file_appends_classic_structures() {
    let path = reopen_keeps_generation("classic", Some(LibverBound::Earliest), 0);
    assert_eq!(
        layout_version_of(&path, "appended"),
        3,
        "a classic file's appended chunked dataset takes the version-3 layout \
         message (H5O_LAYOUT_VERSION_DEFAULT, the floor its bound's row of 1 \
         cannot lower)"
    );
    assert!(
        matches!(
            layout_of(&path, "appended"),
            DataLayoutMessage::ChunkedV3 { .. }
        ),
        "a version-3 layout message has no index-type field: the chunks are on \
         the version-1 B-tree"
    );
    cleanup(&path);
}

/// A version-2 superblock: the v1.8 row, whose layout version of 3 keeps the
/// append off every v1.10 index. This is the case the reopen used to get
/// wrong, taking the v1.10 index and lifting the superblock to 3.
#[test]
fn reopening_a_v18_file_appends_v18_structures() {
    let path = reopen_keeps_generation("v18", Some(LibverBound::V18), 2);
    assert_eq!(
        layout_version_of(&path, "appended"),
        3,
        "H5O_layout_ver_bounds[H5F_LIBVER_V18] is H5O_LAYOUT_VERSION_3, and a \
         reopen of a version-2 superblock is written at no newer bound"
    );
    assert!(
        matches!(
            layout_of(&path, "appended"),
            DataLayoutMessage::ChunkedV3 { .. }
        ),
        "the appended dataset must be on the version-1 B-tree, as libhdf5 \
         writes it for the same reopen"
    );
    cleanup(&path);
}

/// A version-3 superblock: the v1.10 row, where the append does take a v1.10
/// index — the same rule, read off a newer version.
#[test]
fn reopening_a_v110_file_appends_v110_structures() {
    let path = reopen_keeps_generation("v110", Some(LibverBound::V110), 3);
    assert_eq!(
        layout_version_of(&path, "appended"),
        4,
        "H5O_layout_ver_bounds[H5F_LIBVER_V110] is H5O_LAYOUT_VERSION_4"
    );
    assert!(
        matches!(
            layout_of(&path, "appended"),
            DataLayoutMessage::ChunkedV4 { .. }
        ),
        "a version-3 superblock's reopen reaches the v1.10 chunk indexes"
    );
    cleanup(&path);
}

/// The default-created file, which names no bound and is a version-2/3
/// superblock over the v1.10 chunk indexes. Reopening it must leave that alone
/// too: the invariant is about the version the file has, not about how it was
/// asked for.
#[test]
fn reopening_the_default_file_keeps_its_superblock_version() {
    let path = tmp("default");
    create_generation(&path, None);
    let before = superblock_version(&path);
    reopen_and_append_chunked(&path, "appended");
    assert_eq!(superblock_version(&path), before);
    cleanup(&path);
}

/// The bound a caller names on a reopened file still applies: libhdf5's
/// superblock-derived bound is a floor on `H5F_LOW_BOUND`, not a ceiling, and
/// `H5Fopen` with `low = V110` on a version-2 superblock writes a version-4
/// layout into it without touching the superblock. Verified against libhdf5
/// 1.14.6 directly.
#[test]
fn a_named_bound_above_the_floor_still_applies_on_reopen() {
    let path = tmp("named");
    create_generation(&path, Some(LibverBound::V18));
    assert_eq!(superblock_version(&path), 2);

    let file = H5File::open_rw(&path).unwrap();
    file.set_libver_bound(LibverBound::V110).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize, 4])
        .chunk(&[2, 2])
        .create("forced")
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();

    assert_eq!(
        superblock_version(&path),
        2,
        "naming a newer bound does not rewrite the superblock version"
    );
    assert_eq!(layout_version_of(&path, "forced"), 4);
    cleanup(&path);
}

/// The other side of the same floor. `H5F__super_read` writes
/// `low_bound = MAX(H5F_LIBVER_V18, low_bound)`, so a bound named *below* the
/// row the superblock version sits on is raised, not honoured: a version-2
/// superblock never gets the classic generation's version-1 structures poured
/// into it.
#[test]
fn a_named_bound_below_the_floor_is_raised_to_it() {
    let path = tmp("below");
    create_generation(&path, Some(LibverBound::V18));

    let file = H5File::open_rw(&path).unwrap();
    file.set_libver_bound(LibverBound::Earliest).unwrap();
    file.new_dataset::<i32>()
        .shape([4usize, 4])
        .chunk(&[2, 2])
        .create("appended")
        .unwrap()
        .write_raw(&(0..16i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();

    assert_eq!(superblock_version(&path), 2);
    assert_eq!(
        layout_version_of(&path, "appended"),
        3,
        "the V18 floor holds: H5O_layout_ver_bounds[EARLIEST] of 1 is below it"
    );
    cleanup(&path);
}
