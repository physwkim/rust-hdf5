//! An implicitly indexed dataset's chunk grid is one run of raw bytes, and
//! every write into it — the fill at create and each chunk afterwards — goes
//! to the file through the one owner of a raw-byte write, against the run the
//! dataset's layout message names.
//!
//! That owner also knows two destinations that are not this file: the files an
//! External File List names, and a virtual dataset's sources. Neither can be
//! reached from a chunk write, because neither can be a chunked dataset at
//! all — which is the other half of what this pins.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage};
use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
use rust_hdf5::format::messages::{MSG_DATA_LAYOUT, MSG_LINK};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::superblock::SuperblockV2V3;
use rust_hdf5::format::{FormatContext, UNDEF_ADDR};
use rust_hdf5::{H5File, Selection};

/// Per-test unique temp directory; cargo runs tests in parallel.
fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_implicit_own_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The data layout message of the root-level dataset `name`, decoded from the
/// file's bytes.
fn layout_of(bytes: &[u8], name: &str) -> DataLayoutMessage {
    let sb = SuperblockV2V3::decode(bytes).unwrap();
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
    let at = (sb.base_address + addr) as usize;
    let (header, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    let msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_DATA_LAYOUT)
        .unwrap_or_else(|| panic!("'{name}' has no data layout message"));
    DataLayoutMessage::decode(&msg.data, &ctx).unwrap().0
}

/// A chunk write into an implicitly indexed dataset lands in the slot its
/// coordinates name inside the grid the layout message points at, and changes
/// nothing outside it: `data_addr + linear_index * chunk_bytes` is the whole
/// of that index (`H5D__none_idx_get_addr`, H5Dnone.c), and the write goes
/// through the same owner the create-time fill does, against the same run.
#[test]
fn an_implicit_chunk_write_lands_in_its_grid_slot_and_nowhere_else() {
    let dir = tmp("grid");
    let path = dir.join("implicit.h5");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([16usize])
            .chunk(&[4])
            .fill_value(-9i32)
            .early_allocation()
            .create("grid")
            .unwrap();
        // Chunk 2 covers elements 8..12.
        let chunk: Vec<u8> = (100i32..104).flat_map(|v| v.to_le_bytes()).collect();
        ds.write_chunk_at(&[2], &chunk).unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let DataLayoutMessage::ChunkedV4 {
        index_type,
        index_address,
        ..
    } = layout_of(&bytes, "grid")
    else {
        panic!("an implicitly indexed dataset has a version-4 chunked layout");
    };
    assert_eq!(index_type, ChunkIndexType::Implicit);
    assert_ne!(index_address, UNDEF_ADDR);

    // The grid is 4 chunks of 4 i32s. Read it back off the disk as the writer
    // addressed it, not through the reader.
    let grid = index_address as usize;
    let grid_bytes = &bytes[grid..grid + 64];
    let values: Vec<i32> = grid_bytes
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let mut expected = vec![-9i32; 16];
    expected[8..12].copy_from_slice(&[100, 101, 102, 103]);
    assert_eq!(values, expected);

    // And the same through the reader, which resolves the slot by the same
    // arithmetic.
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("grid").unwrap().read_raw::<i32>().unwrap(),
        expected
    );
    drop(file);
    std::fs::remove_dir_all(&dir).ok();
}

/// The other two destinations the raw-byte owner knows are unreachable from a
/// chunk write: an externally stored dataset's layout is contiguous with an
/// undefined address (its bytes are in the files its External File List
/// names), and a virtual dataset's layout names a mapping list. Neither is a
/// chunk grid, and neither dataset can be chunked in the first place.
#[test]
fn external_and_virtual_storage_never_carry_a_chunk_grid() {
    let dir = tmp("targets");
    let path = dir.join("targets.h5");
    {
        let file = H5File::create(&path).unwrap();
        // Absolute, so the name resolves the same way wherever the test
        // process runs: with no `HDF5_EXTFILE_PREFIX` set, a relative name is
        // relative to the process's current directory (`H5_combine_path`).
        let payload = dir.join("payload.raw");
        let ext = file
            .new_dataset::<i32>()
            .shape([16usize])
            .external(&[(payload.to_str().unwrap(), 0, 64)])
            .create("outside")
            .unwrap();
        ext.write_raw(&(0..16i32).collect::<Vec<_>>()).unwrap();
        file.new_dataset::<i32>()
            .shape([16usize])
            .virtual_mapping(Selection::All, "src.h5", "src", Selection::All)
            .create("elsewhere")
            .unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    match layout_of(&bytes, "outside") {
        DataLayoutMessage::Contiguous { address, .. } => assert_eq!(address, UNDEF_ADDR),
        other => panic!("an externally stored dataset is contiguous, not {other:?}"),
    }
    assert!(matches!(
        layout_of(&bytes, "elsewhere"),
        DataLayoutMessage::Virtual { .. }
    ));

    // The external bytes went to the external file, not into the HDF5 one.
    let payload = std::fs::read(dir.join("payload.raw")).unwrap();
    assert_eq!(payload.len(), 64);
    assert_eq!(i32::from_le_bytes(payload[0..4].try_into().unwrap()), 0);
    assert_eq!(i32::from_le_bytes(payload[60..64].try_into().unwrap()), 15);

    std::fs::remove_dir_all(&dir).ok();
}
