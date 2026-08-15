//! Shared object header messages (SOHM), write side.
//!
//! A file created with `H5FileOptions::shared_messages` writes each covered
//! message body once into a shared-message fractal heap and stores a pointer
//! to it in every object header that would have held the body. What has to
//! hold: the file still reads back as the same objects, the master table and
//! the index describe what was actually written, and a file created without
//! the option is unchanged.
//!
//! `tests/sohm.rs` is the read side, against libhdf5-written fixtures;
//! `tests/sohm_append.rs` covers reopening a file that already has a
//! shared-message table.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::creation_order::CreationOrder;
use rust_hdf5::format::messages::{MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::sohm::{
    type_flag, SohmMasterTable, BT2_TYPE_SOHM_INDEX, SMLI_SIGNATURE, SOHM_B2_NODE_SIZE,
    SOHM_INDEX_BTREE, SOHM_INDEX_LIST,
};
use rust_hdf5::format::superblock::{SuperblockV0V1, SuperblockV2V3};
use rust_hdf5::format::{FormatContext, UNDEF_ADDR};
use rust_hdf5::{DatatypeMessage, H5File};

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_sohm_write_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The mask `gen_sohm.c` uses: datatype, dataspace and attribute messages.
fn all_three() -> u16 {
    type_flag(MSG_DATATYPE).unwrap()
        | type_flag(MSG_DATASPACE).unwrap()
        | type_flag(MSG_ATTRIBUTE).unwrap()
}

/// Write the `gen_sohm.c` content into `path` under one index.
fn write_sohm_file(path: &PathBuf, types: u16, min_mesg_size: u32, list_max: u16, btree_min: u16) {
    let file = H5File::options()
        .shared_messages(&[(types, min_mesg_size)], list_max, btree_min)
        .create(path)
        .unwrap();
    for i in 0..4i32 {
        let ds = file
            .new_dataset::<i32>()
            .shape([8usize])
            .create(&format!("shared{i}"))
            .unwrap();
        ds.write_raw(&(0..8i32).map(|j| i * 10 + j).collect::<Vec<_>>())
            .unwrap();
        ds.new_attr::<f64>()
            .shape([3usize])
            .create("cal")
            .unwrap()
            .write_array(&[0.5f64, 1.5, 2.5])
            .unwrap();
    }
    file.commit_datatype("named_i32", DatatypeMessage::i32_type())
        .unwrap();
    file.new_dataset::<i32>()
        .committed_type("named_i32")
        .shape([8usize])
        .create("uses_named")
        .unwrap()
        .write_raw(&(100..108i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// Every object the file was asked for, read back through this crate's own
/// reader — which has to resolve each pointer to get any of it.
fn check_contents(path: &PathBuf) {
    let file = H5File::open(path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(
        names,
        vec!["shared0", "shared1", "shared2", "shared3", "uses_named"]
    );
    for i in 0..4i32 {
        let ds = file.dataset(&format!("shared{i}")).unwrap();
        assert_eq!(ds.shape(), vec![8]);
        let data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(data, (0..8i32).map(|j| i * 10 + j).collect::<Vec<_>>());
        assert_eq!(ds.attr_names().unwrap(), vec!["cal"]);
        let cal: Vec<f64> = ds.attr("cal").unwrap().read_numeric_as().unwrap();
        assert_eq!(cal, vec![0.5, 1.5, 2.5]);
    }
    let used: Vec<i32> = file.dataset("uses_named").unwrap().read_raw().unwrap();
    assert_eq!(used, (100..108).collect::<Vec<i32>>());
    assert_eq!(file.named_datatype_names(), vec!["named_i32"]);
}

/// The master table the file's superblock extension names, decoded from the
/// bytes on disk.
fn read_master_table(path: &PathBuf) -> SohmMasterTable {
    let ctx = FormatContext::default_v3();
    let table = {
        let file = H5File::open(path).unwrap();
        file.superblock_extension()
            .shared_message_table
            .expect("a file with shared messages names its table in the extension")
    };
    let bytes = std::fs::read(path).unwrap();
    let at = table.table_address as usize;
    let size = SohmMasterTable::encoded_size(&ctx, table.nindexes);
    SohmMasterTable::decode(&bytes[at..at + size], &ctx, table.nindexes).unwrap()
}

/// The root group's object header, decoded from the bytes on disk.
fn read_root_header(path: &PathBuf) -> ObjectHeader {
    let bytes = std::fs::read(path).unwrap();
    let superblock = SuperblockV2V3::decode(&bytes).unwrap();
    let at = (superblock.base_address + superblock.root_group_object_header_address) as usize;
    ObjectHeader::decode(&bytes[at..]).unwrap().0
}

/// One index over all three classes: the bodies five datasets share end up in
/// the heap once each, and the file still reads back whole.
#[test]
fn a_created_file_shares_the_bodies_its_index_covers() {
    let path = unique_tmp("list");
    write_sohm_file(&path, all_three(), 0, 50, 40);
    check_contents(&path);

    let table = read_master_table(&path);
    assert_eq!(table.indexes.len(), 1);
    let index = &table.indexes[0];
    assert_eq!(index.index_type, SOHM_INDEX_LIST);
    assert_eq!(index.mesg_types, all_three());
    assert_eq!(index.list_max, 50);
    assert_eq!(index.btree_min, 40);
    // The dataspace every dataset has, the datatype the four that describe
    // their own share, and the attribute body. `uses_named` reaches its type
    // through the committed datatype, which is shared by address instead.
    assert_eq!(index.num_messages, 3);

    let bytes = std::fs::read(&path).unwrap();
    let at = index.index_addr as usize;
    assert_eq!(&bytes[at..at + 4], &SMLI_SIGNATURE);
    cleanup(&path);
}

/// `H5Pset_shared_mesg_phase_change(fcpl, 0, 0)` — the `sohm_btree` fixture's
/// setting — puts the index in B-tree form from the first message.
#[test]
fn a_zero_list_maximum_writes_a_btree_index() {
    let path = unique_tmp("btree");
    write_sohm_file(&path, all_three(), 0, 0, 0);
    check_contents(&path);

    let table = read_master_table(&path);
    let index = &table.indexes[0];
    assert_eq!(index.index_type, SOHM_INDEX_BTREE);
    assert_eq!(index.num_messages, 3);

    let bytes = std::fs::read(&path).unwrap();
    let at = index.index_addr as usize;
    assert_eq!(&bytes[at..at + 4], b"BTHD");
    assert_eq!(bytes[at + 5], BT2_TYPE_SOHM_INDEX);
    assert_eq!(
        u32::from_le_bytes(bytes[at + 6..at + 10].try_into().unwrap()),
        SOHM_B2_NODE_SIZE
    );
    assert_eq!(
        u16::from_le_bytes(bytes[at + 10..at + 12].try_into().unwrap()),
        17,
        "H5SM_SOHM_ENTRY_SIZE for eight-byte addresses"
    );
    cleanup(&path);
}

/// An index takes only the classes its mask names. Covering just dataspaces
/// leaves the datatype and attribute messages in the headers, so the index
/// holds one body.
#[test]
fn an_index_takes_only_the_classes_its_mask_names() {
    let path = unique_tmp("mask");
    write_sohm_file(&path, type_flag(MSG_DATASPACE).unwrap(), 0, 50, 40);
    check_contents(&path);

    let table = read_master_table(&path);
    assert_eq!(table.indexes[0].num_messages, 1);
    cleanup(&path);
}

/// `H5Pset_shared_mesg_index`'s minimum size: a message under it stays in the
/// header. Every body this file would share is well under 4 KiB, so the index
/// ends up empty — and an empty index is still written, the way
/// `H5SM__create_index` makes one at file creation.
#[test]
fn a_minimum_size_above_every_body_leaves_an_empty_index() {
    let path = unique_tmp("minsize");
    write_sohm_file(&path, all_three(), 4096, 50, 40);
    check_contents(&path);

    let table = read_master_table(&path);
    assert_eq!(table.indexes[0].num_messages, 0);
    cleanup(&path);
}

/// The default is unchanged: a file created without the option has no
/// superblock extension at all.
#[test]
fn a_file_created_without_indexes_has_no_shared_message_table() {
    let path = unique_tmp("plain");
    let file = H5File::create(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create("data")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert!(file.superblock_extension().shared_message_table.is_none());
    cleanup(&path);
}

/// Two indexes, each over its own classes: every message goes to the heap of
/// the index whose mask covers it.
#[test]
fn two_indexes_split_the_message_classes_between_them() {
    let path = unique_tmp("split");
    let file = H5File::options()
        .shared_messages(
            &[
                (type_flag(MSG_ATTRIBUTE).unwrap(), 0),
                (
                    type_flag(MSG_DATATYPE).unwrap() | type_flag(MSG_DATASPACE).unwrap(),
                    0,
                ),
            ],
            50,
            40,
        )
        .create(&path)
        .unwrap();
    for i in 0..3i32 {
        let ds = file
            .new_dataset::<i32>()
            .shape([8usize])
            .create(&format!("d{i}"))
            .unwrap();
        ds.write_raw(&(0..8i32).collect::<Vec<_>>()).unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_numeric(&7i32)
            .unwrap();
    }
    file.close().unwrap();

    let table = read_master_table(&path);
    assert_eq!(table.indexes.len(), 2);
    // One attribute body in the first index; the shared dataspace and the
    // shared datatype in the second.
    assert_eq!(table.indexes[0].num_messages, 1);
    assert_eq!(table.indexes[1].num_messages, 2);
    assert_ne!(table.indexes[0].heap_addr, table.indexes[1].heap_addr);
    assert_eq!(
        table.heap_addr(MSG_ATTRIBUTE),
        Some(table.indexes[0].heap_addr)
    );
    assert_eq!(
        table.heap_addr(MSG_DATASPACE),
        Some(table.indexes[1].heap_addr)
    );

    let file = H5File::open(&path).unwrap();
    for i in 0..3i32 {
        let ds = file.dataset(&format!("d{i}")).unwrap();
        let data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(data, (0..8).collect::<Vec<i32>>());
        let tag: Vec<i32> = ds.attr("tag").unwrap().read_numeric_as().unwrap();
        assert_eq!(tag, vec![7]);
    }
    cleanup(&path);
}

/// A shared attribute is found again through its message creation index, so
/// `H5SM_init` sets `store_msg_crt_idx` on a file whose index covers
/// attributes and every object header created afterwards records creation
/// indices — whatever the object's creation property list asked for. An index
/// over the other two classes leaves the headers alone.
#[test]
fn sharing_attributes_makes_every_header_record_creation_indices() {
    let path = unique_tmp("crtidx");
    write_sohm_file(&path, all_three(), 0, 50, 40);
    assert_eq!(
        read_root_header(&path).attribute_creation_order(),
        CreationOrder::Tracked
    );
    cleanup(&path);

    let path = unique_tmp("nocrtidx");
    write_sohm_file(
        &path,
        type_flag(MSG_DATATYPE).unwrap() | type_flag(MSG_DATASPACE).unwrap(),
        0,
        50,
        40,
    );
    assert_eq!(
        read_root_header(&path).attribute_creation_order(),
        CreationOrder::Untracked
    );
    cleanup(&path);
}

/// The configurations `H5Pset_shared_mesg_nindexes` and
/// `H5Pset_shared_mesg_phase_change` reject are rejected here too, at the
/// point the file is created.
#[test]
fn file_creation_refuses_a_configuration_libhdf5_refuses() {
    let too_many: Vec<(u16, u32)> = (0..9).map(|_| (all_three(), 0)).collect();
    let path = unique_tmp("invalid");
    for (indexes, list_max, btree_min, expect) in [
        (too_many.as_slice(), 50u16, 40u16, "at most 8"),
        (&[(all_three(), 0)][..], 10, 40, "btree_min"),
        (&[(0, 0)][..], 50, 40, "covering no message type"),
    ] {
        let err = match H5File::options()
            .shared_messages(indexes, list_max, btree_min)
            .create(&path)
        {
            Ok(_) => panic!("an invalid shared-message configuration must not create a file"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(expect), "{err}");
    }
    cleanup(&path);
}

/// Every object header reachable from the root, by path, decoded from the
/// bytes. Reaching one at all is the version assertion: `ObjectHeader::decode`
/// requires the `OHDR` signature, which a version-1 header does not have.
fn every_header(bytes: &[u8]) -> Vec<(String, ObjectHeader)> {
    use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
    use rust_hdf5::format::messages::MSG_LINK;

    let sb = SuperblockV2V3::decode(bytes).unwrap();
    let ctx = FormatContext {
        sizeof_addr: sb.sizeof_offsets,
        sizeof_size: sb.sizeof_lengths,
    };
    let mut out = Vec::new();
    let mut queue = vec![(String::from("/"), sb.root_group_object_header_address)];
    while let Some((path, addr)) = queue.pop() {
        let at = (sb.base_address + addr) as usize;
        let (header, _) = ObjectHeader::decode(&bytes[at..])
            .unwrap_or_else(|e| panic!("the header of {path} is not a version-2 one: {e}"));
        for m in header.messages.iter().filter(|m| m.msg_type == MSG_LINK) {
            if let Ok((link, _)) = LinkMessage::decode(&m.data, &ctx) {
                if let LinkTarget::Hard { address } = link.target {
                    queue.push((format!("{}{}/", path, link.name), address));
                }
            }
        }
        out.push((path, header));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `store_msg_crt_idx` is a property of the *file*, so the floor it puts under
/// attribute creation order reaches every object header the file holds, not
/// just the root's: `H5O__create_ohdr` raises each one it makes to version 2
/// and ORs `H5O_HDR_ATTR_CRT_ORDER_TRACKED` in (H5Oint.c:364, H5Oint.c:442).
///
/// The other half is that such a file can never be a classic one. libhdf5
/// reaches that by raising the superblock to version 2 for `sohm_nindexes > 0`
/// and then refusing the file outright when the high bound cannot reach that
/// version (H5Fsuper.c:1135). Here the combination is unrepresentable instead:
/// `shared_messages` is read only by `create`, which never writes a superblock
/// below version 2, and `open_rw` — the one path that produces a classic
/// writer — refuses the option outright rather than reading it, so the indexes
/// a reopened file has are the ones it was created with and never a set this
/// session asked for. So a classic file keeps its version-1 headers with the
/// floor nowhere in sight.
#[test]
fn the_creation_index_floor_reaches_every_header_and_never_a_classic_file() {
    let path = unique_tmp("crtidx_every");
    {
        let file = H5File::options()
            .shared_messages(&[(all_three(), 0)], 50, 40)
            .create(&path)
            .unwrap();
        let group = file.root_group().create_group("g").unwrap();
        group.set_attr_numeric("scale", &2.0f64).unwrap();
        for (parent, name) in [(&file.root_group(), "outer"), (&group, "inner")] {
            let ds = parent.new_dataset::<i32>().shape([4]).create(name).unwrap();
            ds.write_raw(&[7i32; 4]).unwrap();
            // The same body in both headers, so the index actually shares it.
            ds.new_attr::<f64>()
                .shape(())
                .create("units")
                .unwrap()
                .write_numeric(&1.5f64)
                .unwrap();
        }
        file.close().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert!(SuperblockV2V3::decode(&bytes).unwrap().version >= 2);
    let headers = every_header(&bytes);
    assert_eq!(
        headers.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        ["/", "/g/", "/g/inner/", "/outer/"]
    );
    for (path, header) in &headers {
        assert_eq!(
            header.attribute_creation_order(),
            CreationOrder::Tracked,
            "{path}"
        );
    }
    cleanup(&path);

    // The classic side: the option is create-only, so open_rw of an
    // existing file refuses it outright rather than silently doing nothing.
    let path = unique_tmp("classic");
    std::fs::write(
        &path,
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/btreek_legacy.h5"),
        )
        .unwrap(),
    )
    .unwrap();
    let before = std::fs::read(&path).unwrap();
    let err = H5File::options()
        .shared_messages(&[(all_three(), 0)], 50, 40)
        .open_rw(&path)
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("shared_messages"),
        "error should name the offending option: {err}"
    );
    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after, "a refused open must not touch the file");
    let sb = SuperblockV0V1::decode(&after).unwrap();
    assert_eq!(sb.version, 1);
    assert_eq!(sb.superblock_extension_address, UNDEF_ADDR);
    let root = (sb.base_address + sb.root_symbol_table_entry.obj_header_addr) as usize;
    assert!(
        ObjectHeader::decode(&after[root..]).is_err(),
        "a classic file's root header must not carry the OHDR signature"
    );
    let (root_header, _) = ObjectHeader::decode_v1(&after[root..]).unwrap();
    assert_eq!(
        root_header.attribute_creation_order(),
        CreationOrder::Untracked
    );
    cleanup(&path);
}
