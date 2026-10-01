//! Reopening a file never changes its superblock version, and that version
//! says nothing about the bound the appended structures are written at.
//!
//! `H5F__super_init` is the only place libhdf5 decides a superblock version
//! (H5Fsuper.c:1154). `H5F__super_read` validates the version it finds and
//! never recomputes one. libhdf5 1.14 also raised the file's *low*
//! library-version bound to the row that version belongs to; libhdf5 2.0
//! dropped that (HDFGroup/hdf5#4939, H5Fsuper.c:438-453 keeps only the
//! SWMR-write raise), so the bound a reopened file is appended at is the
//! fapl's — what the caller named, or the default — whatever its superblock
//! says. A version-3 superblock opened at `H5F_LIBVER_EARLIEST` gets a
//! version-3 layout message over the version-1 chunk B-tree appended, and a
//! version-2 superblock opened at `V110` gets a version-4 message over a v1.10
//! index, the superblock version untouched either way.
//!
//! Checked here for all three generations a reopen can find: version 0
//! (classic), version 2 (v1.8) and version 3 (v1.10). Each case reads the
//! superblock version byte before and after the append and decodes the
//! appended dataset's own data layout message out of the file.

use rust_hdf5::format::btree_v1::{BTreeV1Config, BTreeV1Node};
use rust_hdf5::format::local_heap::{local_heap_get_string, LocalHeapHeader};
use rust_hdf5::format::messages::data_layout::DataLayoutMessage;
use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
use rust_hdf5::format::messages::{MSG_DATA_LAYOUT, MSG_LINK, MSG_OBJ_HEADER_CONTINUATION};
use rust_hdf5::format::object_header::{ObjectHeader, ObjectHeaderMessage};
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
    let addr = root_messages(bytes, &sb, &ctx)
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

/// Every message of the root group's header: chunk 0's, then those of each
/// continuation chunk it names. A reopen writes the root header back over
/// its own chunk 0 and spills a link it no longer has room for into a
/// continuation chunk, which is where the appended dataset's link is found.
fn root_messages(
    bytes: &[u8],
    sb: &SuperblockV2V3,
    ctx: &FormatContext,
) -> Vec<ObjectHeaderMessage> {
    let at = |addr: u64| (sb.base_address + addr) as usize;
    let (root, _) =
        ObjectHeader::decode(&bytes[at(sb.root_group_object_header_address)..]).unwrap();
    let envelope = if root.has_creation_order() { 6 } else { 4 };
    let mut messages = root.messages;
    let mut i = 0;
    while i < messages.len() {
        if messages[i].msg_type == MSG_OBJ_HEADER_CONTINUATION {
            let sa = ctx.sizeof_addr as usize;
            let data = &messages[i].data;
            let cont_addr = u64::from_le_bytes(data[..sa].try_into().unwrap());
            let cont_len = u64::from_le_bytes(data[sa..sa + 8].try_into().unwrap()) as usize;
            let chunk = &bytes[at(cont_addr)..at(cont_addr) + cont_len];
            assert_eq!(&chunk[..4], b"OCHK");
            // Messages up to the checksum; a gap shorter than an envelope ends them.
            let (mut pos, end) = (4, cont_len - 4);
            while pos + envelope <= end {
                let msg_type = chunk[pos];
                let size = u16::from_le_bytes([chunk[pos + 1], chunk[pos + 2]]) as usize;
                let flags = chunk[pos + 3];
                let body = pos + envelope;
                messages.push(ObjectHeaderMessage {
                    msg_type,
                    flags,
                    creation_index: 0,
                    data: chunk[body..body + size].to_vec(),
                });
                pos = body + size;
            }
        }
        i += 1;
    }
    messages
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

/// A version-2 superblock, reopened with no bound named: the append is
/// written at the writer's default, whose layout row is `V110`'s, so it takes
/// a v1.10 index — and the superblock stays at version 2, which is the case
/// the reopen used to get wrong by lifting it to 3.
#[test]
fn reopening_a_v18_file_appends_at_the_default_bound() {
    let path = reopen_keeps_generation("v18", Some(LibverBound::V18), 2);
    assert_eq!(
        layout_version_of(&path, "appended"),
        4,
        "the default layout row is H5F_LIBVER_V110's, whatever the superblock"
    );
    assert!(
        matches!(
            layout_of(&path, "appended"),
            DataLayoutMessage::ChunkedV4 { .. }
        ),
        "a version-4 layout message inside a version-2 superblock, which \
         libhdf5 reads by the message's own version"
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

/// The bound a caller names on a reopened file applies above the row its
/// superblock sits on: `H5Fopen` with `low = V110` on a version-2 superblock
/// writes a version-4 layout into it without touching the superblock.
#[test]
fn a_named_bound_above_the_superblocks_row_applies_on_reopen() {
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

/// And below it. A bound named under the row the superblock version sits on
/// is honoured, as libhdf5 2.0 honours a fapl's `low` on `H5Fopen`
/// (HDFGroup/hdf5#4939): a version-3 superblock opened at `Earliest` gets the
/// classic generation's version-3 layout message and version-1 B-tree
/// appended — the file h5py on libhdf5 2.x leaves behind, since it pins
/// `H5F_LIBVER_EARLIEST` on every open — and keeps its version. libhdf5 1.14
/// raised the bound to the row instead.
#[test]
fn a_named_bound_below_the_superblocks_row_is_honoured() {
    let path = tmp("below");
    create_generation(&path, Some(LibverBound::V110));
    assert_eq!(superblock_version(&path), 3);

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

    assert_eq!(superblock_version(&path), 3);
    assert_eq!(layout_version_of(&path, "appended"), 3);
    assert!(
        matches!(
            layout_of(&path, "appended"),
            DataLayoutMessage::ChunkedV3 { .. }
        ),
        "a version-1 B-tree inside a version-3 superblock"
    );
    cleanup(&path);
}
