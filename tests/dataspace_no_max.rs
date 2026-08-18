//! A dataspace message that stores no maximum dimensions.
//!
//! `H5S_VALID_MAX` clear is a state the libhdf5 API cannot reach:
//! `H5S_set_extent_simple` allocates `extent.max` for every simple extent and
//! fills it from the current dimensions when the caller passed none
//! (H5S.c:1293-1299), so every extent that reaches `H5O__sdspace_encode` has
//! one. Only a file whose bytes say otherwise produces it, so both the
//! measurement below and the fixture here are made by byte surgery — clearing
//! bit 0 of the message's flags byte — and the eight bytes that held the
//! maximum stay behind as message slack.
//!
//! What libhdf5 does with such a message when it rewrites it is the question
//! this file pins: it keeps it. `H5S_read` decodes the message into
//! `ds->extent` (H5S.c:1100) and `H5S_write` re-encodes that same extent
//! (H5S.c:1039), `H5O__sdspace_decode` leaves `sdim->max` null when the flag
//! is clear (H5Osdspace.c:188-194), and `H5O__sdspace_encode` raises the flag
//! only for a non-null array (H5Osdspace.c:271-272). Measured against libhdf5
//! 1.14.6 through h5py 3.15.1: clearing the flag on a chunked dataset written
//! at `libver='earliest'` and then calling `H5Dset_extent`, which reaches
//! `H5S_write`, rewrote the dimensions and left the flags byte at 0. The
//! extent grew past the maximum the cleared bytes still name, too — a null
//! `extent.max` bounds nothing (H5S.c:1777).
//!
//! The crate reaches the same re-encode whenever an appended dataset's header
//! goes stale and is emitted again, which a second hard link is enough to do.
//! Neither side of the difference is visible through the API —
//! `H5S_extent_get_dims` reports the current dimensions for a null maximum
//! (H5S.c:968-973), so h5py's `maxshape` reads the same either way — so the
//! witness has to be the message bytes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{H5File, LibverBound};

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_dataspace_no_max_{}_{}_{}",
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

/// Every offset at which a version-1 rank-1 dataspace message for `dim`
/// starts, whatever its flags byte: version 1, rank 1, flags, then the
/// reserved byte and reserved word version 1 carries in place of a type byte.
/// Each hit is preceded by the version-1 object header message prologue that
/// names its type and length, which is what pins the message down: a
/// coincidental byte sequence in raw data has no such prologue.
fn v1_rank1_messages(bytes: &[u8], dim: u64) -> Vec<usize> {
    (PROLOGUE..bytes.len().saturating_sub(16))
        .filter(|&i| {
            bytes[i] == 1
                && bytes[i + 1] == 1
                && bytes[i + 3..i + 8] == [0u8; 5]
                && bytes[i + 8..i + 16] == dim.to_le_bytes()
                && bytes[i - PROLOGUE..i - PROLOGUE + 2] == [1, 0]
        })
        .collect()
}

/// `type`, `size`, `flags`, three reserved bytes — the header a version-1
/// object header puts in front of every message.
const PROLOGUE: usize = 8;

/// The length the object header records for the message starting at `at`.
fn message_len(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at - PROLOGUE + 2], bytes[at - PROLOGUE + 3]])
}

/// The crate writes the maximum `H5S_set_extent_simple` fills in, so a
/// crate-written simple dataspace always has the flag set — this is the
/// starting point the surgery below works from.
#[test]
fn a_crate_written_simple_dataspace_names_its_maximum() {
    let path = unique_tmp("written");
    {
        let file = H5File::options()
            .libver(LibverBound::Earliest)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([4usize])
            .create("fixed")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let hits = v1_rank1_messages(&bytes, 4);
    assert_eq!(hits.len(), 1, "expected one rank-1 dataspace message");
    assert_eq!(bytes[hits[0] + 2], 0x01, "H5S_VALID_MAX must be set");
    assert_eq!(
        bytes[hits[0] + 16..hits[0] + 24],
        4u64.to_le_bytes(),
        "the maximum is the current dimension"
    );
    assert_eq!(message_len(&bytes, hits[0]), 24, "prefix, dims, maximum");

    cleanup(&path);
}

/// A dataspace the file stores without a maximum keeps none when the crate
/// rewrites the message, matching what libhdf5 does through
/// `H5S_read`/`H5S_write`. Before this, the re-encode invented one, so a
/// message libhdf5 preserves came back as a different message.
#[test]
fn a_stored_dataspace_with_no_maximum_keeps_none_through_a_rewrite() {
    let path = unique_tmp("nomax");
    {
        let file = H5File::options()
            .libver(LibverBound::Earliest)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([4usize])
            .create("fixed")
            .unwrap()
            .write_raw(&[11i32, 22, 33, 44])
            .unwrap();
        file.close().unwrap();
    }

    // Byte surgery: clear `H5S_VALID_MAX`. The eight bytes that held the
    // maximum stay behind as message slack, exactly as they do after
    // libhdf5's own rewrite of such a message.
    {
        let mut bytes = std::fs::read(&path).unwrap();
        let hits = v1_rank1_messages(&bytes, 4);
        assert_eq!(hits.len(), 1);
        assert_eq!(bytes[hits[0] + 2], 0x01);
        assert_eq!(message_len(&bytes, hits[0]), 24);
        bytes[hits[0] + 2] = 0x00;
        std::fs::write(&path, &bytes).unwrap();
    }

    // A second hard link makes the dataset's header stale, so the crate
    // re-emits it — and with it the dataspace message, encoded again from
    // what the reader decoded.
    {
        let file = H5File::open_rw(&path).unwrap();
        file.root_group().link("alias", "/fixed").unwrap();
        file.close().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let hits = v1_rank1_messages(&bytes, 4);
    assert_eq!(hits.len(), 1, "expected one rank-1 dataspace message");
    assert_eq!(
        bytes[hits[0] + 2],
        0x00,
        "a stored dataspace with no maximum must be rewritten with none"
    );
    // The message was re-encoded, not left alone: it is now the eight bytes
    // shorter that dropping the maximum makes it.
    assert_eq!(message_len(&bytes, hits[0]), 16, "prefix and dims only");

    {
        let file = H5File::open(&path).unwrap();
        for name in ["fixed", "alias"] {
            let ds = file.dataset(name).unwrap();
            assert_eq!(ds.shape(), vec![4]);
            assert_eq!(ds.read_raw::<i32>().unwrap(), vec![11, 22, 33, 44]);
        }
    }

    cleanup(&path);
}
