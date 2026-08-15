//! Object header continuation chunks, write side.
//!
//! A group's object header is sized when the group is created, from the link
//! info and group info messages plus an estimate of the links to come
//! (`H5G__obj_create_real`). Attributes are not in that estimate, so a group
//! carrying more than a few of them overflows chunk 0 and the rest of its
//! messages go into an `OCHK` continuation chunk. `tests/ochk.rs` is the read
//! side, against the libhdf5-written fixture.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::messages::MSG_OBJ_HEADER_CONTINUATION;
use rust_hdf5::format::object_header::{ObjectHeader, OCHK_SIGNATURE};
use rust_hdf5::format::superblock::SuperblockV2V3;
use rust_hdf5::{DatatypeMessage, H5File};

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_ochk_write_{}_{}_{}",
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

/// The `gen_ochk.c` attribute: a 256-byte fixed-length ASCII string whose
/// first character says which one it is.
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

/// Write the `gen_ochk.c` content: one dataset and `count` root attributes.
fn write_notes(path: &PathBuf, count: u8) {
    let file = H5File::create(path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create("data")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    for i in 0..count {
        let (dt, text) = note(i);
        file.set_attr_typed(&format!("note{i}"), dt, text).unwrap();
    }
    file.close().unwrap();
}

/// The root group's chunk 0, and every continuation chunk it names, as
/// `(address, length)`.
fn root_chunks(path: &PathBuf) -> (ObjectHeader, Vec<(u64, u64)>) {
    let bytes = std::fs::read(path).unwrap();
    let superblock = SuperblockV2V3::decode(&bytes).unwrap();
    let at = (superblock.base_address + superblock.root_group_object_header_address) as usize;
    let (header, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    let chunks = header
        .messages
        .iter()
        .filter(|m| m.msg_type == MSG_OBJ_HEADER_CONTINUATION)
        .map(|m| {
            (
                u64::from_le_bytes(m.data[..8].try_into().unwrap()),
                u64::from_le_bytes(m.data[8..16].try_into().unwrap()),
            )
        })
        .collect();
    (header, chunks)
}

/// Read every note back through this crate's own reader, which has to follow
/// the continuation chunk to find any of them.
fn check_notes(path: &PathBuf, count: u8) {
    let file = H5File::open(path).unwrap();
    let mut names = file.attr_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        (0..count).map(|i| format!("note{i}")).collect::<Vec<_>>()
    );
    for i in 0..count {
        let text = file.attr_string(&format!("note{i}")).unwrap();
        let expected = String::from_utf8(note(i).1[..255].to_vec()).unwrap();
        assert_eq!(text, expected, "note{i}");
    }
    let data: Vec<i32> = file.dataset("data").unwrap().read_raw().unwrap();
    assert_eq!(data, (0..8).collect::<Vec<i32>>());
}

/// Six 256-byte attributes are far past what the root group's header was
/// sized for, so its messages spill — and the spilled block is a well-formed
/// `OCHK` chunk that libhdf5 and this crate both read.
#[test]
fn a_large_root_attribute_set_spills_into_a_continuation_chunk() {
    let path = unique_tmp("notes");
    write_notes(&path, 6);

    let (header, chunks) = root_chunks(&path);
    assert_eq!(chunks.len(), 1, "one continuation chunk");
    let (addr, len) = chunks[0];
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[addr as usize..addr as usize + 4], &OCHK_SIGNATURE);
    // Chunk 0 keeps the messages the estimate covered; the attributes are what
    // did not fit.
    assert!(
        !header
            .messages
            .iter()
            .any(|m| m.msg_type == rust_hdf5::format::messages::MSG_ATTRIBUTE),
        "chunk 0 still holds an attribute message"
    );
    assert!(len > 6 * 256, "the continuation holds the attributes");

    check_notes(&path, 6);
    cleanup(&path);
}

/// The continuation chunk lives inside the block the object header address
/// and its encoded size describe: it follows chunk 0 immediately, so freeing
/// or relocating a header by address still takes the whole of it.
#[test]
fn a_continuation_chunk_follows_chunk_zero_in_the_same_block() {
    let path = unique_tmp("contiguous");
    write_notes(&path, 6);

    let bytes = std::fs::read(&path).unwrap();
    let superblock = SuperblockV2V3::decode(&bytes).unwrap();
    let root = superblock.base_address + superblock.root_group_object_header_address;
    let (_, chunks) = root_chunks(&path);
    let (addr, _) = chunks[0];
    let (_, chunk0_len) = ObjectHeader::decode(&bytes[root as usize..]).unwrap();
    assert_eq!(addr, root + chunk0_len as u64);
    cleanup(&path);
}

/// A header whose messages fit the estimate gets no continuation chunk at
/// all — the layout of every file this writer wrote before is unchanged.
#[test]
fn a_header_that_fits_its_estimate_stays_one_chunk() {
    let path = unique_tmp("fits");
    write_notes(&path, 0);

    let (header, chunks) = root_chunks(&path);
    assert!(chunks.is_empty(), "{:?}", chunks);
    assert!(header.messages.iter().any(|m| m.msg_type == 0x06));
    cleanup(&path);
}

/// Reopening such a file reads the whole chain and writes it back whole: the
/// attributes that lived in the continuation chunk are still there, and so is
/// the one added on top.
#[test]
fn a_reopened_file_keeps_the_attributes_its_continuation_chunk_held() {
    let path = unique_tmp("reopen");
    write_notes(&path, 6);
    {
        let file = H5File::open_rw(&path).unwrap();
        let (dt, text) = note(6);
        file.set_attr_typed("note6", dt, text).unwrap();
        file.close().unwrap();
    }
    let (_, chunks) = root_chunks(&path);
    assert_eq!(chunks.len(), 1);
    check_notes(&path, 7);
    cleanup(&path);
}
