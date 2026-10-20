//! Integration tests for the extended SWMR public API: NeXus-layout
//! metadata (fixed/scalar/string datasets, dataset & group attributes),
//! hyperslab reads, dataset placement in groups, and stream resumption
//! via `open_append`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};
use rust_hdf5::FileLocking;

/// Per-test unique temp path so parallel cargo runs cannot collide.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_swmr_api_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &Path) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

const NO_LOCK: FileLocking = FileLocking::Disabled;

/// Fixed/scalar/string datasets and dataset/group/root attributes written
/// through the SWMR API round-trip through `SwmrFileReader`.
#[test]
fn metadata_datasets_and_attributes_round_trip() {
    let path = unique_tmp("metadata");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();

        // Scalar, 1-D numeric, and vlen-string metadata datasets.
        w.write_dataset::<f64>("distance", &[], &[0.55]).unwrap();
        let axis = w.write_dataset::<i32>("axis", &[3], &[10, 20, 30]).unwrap();
        w.write_string_dataset("start_time", &["2026-05-18T10:00:00"])
            .unwrap();

        // Dataset attributes (string + numeric).
        w.set_dataset_attr_string(axis, "units", "mm").unwrap();
        w.set_dataset_attr_numeric::<i32>(axis, "count", &3)
            .unwrap();

        // Group + group/root attributes.
        w.create_group("/", "entry").unwrap();
        w.set_group_attr_string("/entry", "NX_class", "NXentry")
            .unwrap();
        w.set_group_attr_numeric::<f64>("/entry", "version", &2.0)
            .unwrap();
        w.set_group_attr_string("/", "file_name", "metadata.h5")
            .unwrap();

        let frames = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        w.start_swmr().unwrap();
        w.append_frame(frames, &[1u8, 2, 3, 4]).unwrap();
        w.close().unwrap();
    }

    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();

    assert_eq!(r.read_dataset::<f64>("distance").unwrap(), vec![0.55]);
    assert_eq!(r.read_dataset::<i32>("axis").unwrap(), vec![10, 20, 30]);
    assert_eq!(
        r.read_vlen_strings("start_time").unwrap(),
        vec!["2026-05-18T10:00:00".to_string()]
    );
    assert_eq!(r.dataset_element_size("axis").unwrap(), 4);
    assert_eq!(r.dataset_element_size("frames").unwrap(), 1);

    let attr_names = r.dataset_attr_names("axis").unwrap();
    assert!(attr_names.iter().any(|n| n == "units"), "{attr_names:?}");
    assert!(attr_names.iter().any(|n| n == "count"), "{attr_names:?}");
    assert_eq!(r.dataset_attr_string("axis", "units").unwrap(), "mm");

    assert_eq!(
        r.group_attr_string("/entry", "NX_class").unwrap(),
        "NXentry"
    );
    assert!(r
        .group_attr_names("/entry")
        .unwrap()
        .iter()
        .any(|n| n == "version"));
    assert_eq!(
        r.group_attr_string("/", "file_name").unwrap(),
        "metadata.h5"
    );

    cleanup(&path);
}

/// The SWMR metadata writers declare the character set they are named for,
/// and the ASCII twin refuses a string that declaration would misdescribe.
#[test]
fn string_dataset_writers_declare_their_character_set() {
    use rust_hdf5::{DatatypeMessage, H5File};

    let path = unique_tmp("string_cset");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        w.write_string_dataset("utf8", &["\u{e9}t\u{e9}"]).unwrap();
        w.write_string_dataset_ascii("ascii", &["2026-05-18T10:00:00"])
            .unwrap();
        let err = w
            .write_string_dataset_ascii("rejected", &["\u{c548}\u{b155}"])
            .expect_err("a non-ASCII string was accepted under an ASCII datatype")
            .to_string();
        assert!(err.contains("is not ASCII"), "got: {err}");
        w.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("ascii").unwrap().datatype().unwrap(),
        DatatypeMessage::VarLenString {
            padding: 0,
            charset: 0,
        }
    );
    assert_eq!(
        file.dataset("utf8").unwrap().datatype().unwrap(),
        DatatypeMessage::VarLenString {
            padding: 0,
            charset: 1,
        }
    );
    assert_eq!(
        file.dataset("ascii").unwrap().read_strings().unwrap(),
        vec!["2026-05-18T10:00:00".to_string()]
    );
    file.close().unwrap();

    cleanup(&path);
}

/// The SWMR typed reads reinterpret the element image like `read_raw` does,
/// so they owe the same byte-order conversion: a big-endian dataset read as
/// `u32` must give the stored values, not their byte-reversed images.
#[test]
fn big_endian_dataset_reads_as_values() {
    use rust_hdf5::{ByteOrder, DatatypeMessage, H5File};

    let path = unique_tmp("big_endian");
    let values: [u32; 4] = [1, 0x0102_0304, 0xdead_beef, u32::MAX];
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<u32>()
            .datatype(DatatypeMessage::FixedPoint {
                size: 4,
                byte_order: ByteOrder::BigEndian,
                signed: false,
                bit_offset: 0,
                bit_precision: 32,
            })
            .shape([values.len()])
            .create("be")
            .unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
        ds.write_raw_bytes(&bytes).unwrap();
        file.close().unwrap();
    }

    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.read_dataset::<u32>("be").unwrap(), values);
    assert_eq!(r.read_slice::<u32>("be", &[1], &[2]).unwrap(), values[1..3]);

    cleanup(&path);
}

/// `read_slice` fetches one frame of a streaming dataset without reading
/// the whole stream.
#[test]
fn read_slice_reads_a_single_frame() {
    let path = unique_tmp("slice");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let ds = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        w.start_swmr().unwrap();
        w.append_frame(ds, &[1u8, 2, 3, 4]).unwrap();
        w.append_frame(ds, &[5u8, 6, 7, 8]).unwrap();
        w.append_frame(ds, &[9u8, 10, 11, 12]).unwrap();
        w.close().unwrap();
    }

    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![3, 2, 2]);

    // Middle frame only.
    let frame1 = r
        .read_slice::<u8>("frames", &[1, 0, 0], &[1, 2, 2])
        .unwrap();
    assert_eq!(frame1, vec![5, 6, 7, 8]);

    // Last frame, raw bytes.
    let frame2 = r.read_slice_raw("frames", &[2, 0, 0], &[1, 2, 2]).unwrap();
    assert_eq!(frame2, vec![9, 10, 11, 12]);

    cleanup(&path);
}

/// A streaming dataset can be placed inside a group; the reader then sees
/// it at the nested path and the group tree is enumerable.
#[test]
fn assign_dataset_to_group_places_the_stream() {
    let path = unique_tmp("assign");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        w.create_group("/", "entry").unwrap();
        w.create_group("/entry", "data").unwrap();
        let ds = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        w.assign_dataset_to_group("/entry/data", ds).unwrap();
        w.start_swmr().unwrap();
        w.append_frame(ds, &[1u8, 2, 3, 4]).unwrap();
        w.close().unwrap();
    }

    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert!(r.has_group("/entry"));
    assert!(r.has_group("/entry/data"));
    let groups = r.group_paths();
    assert!(
        groups
            .iter()
            .any(|g| g.trim_start_matches('/') == "entry/data"),
        "group paths: {groups:?}"
    );

    let names = r.dataset_names();
    assert!(
        names.iter().any(|n| n == "entry/data/frames"),
        "dataset names: {names:?}"
    );
    assert_eq!(
        r.read_dataset_raw("entry/data/frames").unwrap(),
        vec![1, 2, 3, 4]
    );

    cleanup(&path);
}

/// A cleanly-closed SWMR file can be reopened and its streaming dataset
/// extended with further frames.
#[test]
fn open_append_resumes_streaming() {
    let path = unique_tmp("resume");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let ds = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        w.start_swmr().unwrap();
        w.append_frame(ds, &[1u8, 2, 3, 4]).unwrap();
        w.append_frame(ds, &[5u8, 6, 7, 8]).unwrap();
        w.close().unwrap();
    }

    {
        let mut w = SwmrFileWriter::open_append_with_locking(&path, NO_LOCK).unwrap();
        let ds = w
            .dataset_index("frames")
            .expect("reopened dataset 'frames'");
        w.start_swmr().unwrap();
        w.append_frame(ds, &[9u8, 10, 11, 12]).unwrap();
        w.append_frame(ds, &[13u8, 14, 15, 16]).unwrap();
        w.close().unwrap();
    }

    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![4, 2, 2]);
    assert_eq!(
        r.read_dataset::<u8>("frames").unwrap(),
        (1u8..=16).collect::<Vec<_>>()
    );

    cleanup(&path);
}

/// A multi-frame-chunk writer dropped without `close` still writes the band
/// it was buffering: every appended frame is already in the extent, so the
/// frames of the partial band would otherwise read back as fill.
#[test]
fn a_dropped_writer_writes_its_partial_band() {
    let path = unique_tmp("drop_band");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let ds = w
            .create_streaming_dataset_chunked::<u8>("frames", &[2, 2], &[3, 2, 2])
            .unwrap();
        w.start_swmr().unwrap();
        // One full band (frames 0..3) and one frame of the next.
        for f in 0..4u8 {
            let v = 4 * f + 1;
            w.append_frame(ds, &[v, v + 1, v + 2, v + 3]).unwrap();
        }
        drop(w);
    }

    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![4, 2, 2]);
    assert_eq!(
        r.read_dataset::<u8>("frames").unwrap(),
        (1u8..=16).collect::<Vec<_>>()
    );

    cleanup(&path);
}

/// A flush publishes the frames of a band still filling, not just the
/// extent that counts them: a reader opened after each flush sees every
/// appended frame, through the band completing and the next one starting.
#[test]
fn a_flush_publishes_a_partial_band() {
    let path = unique_tmp("flush_band");
    let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
    let ds = w
        .create_streaming_dataset_chunked::<u8>("frames", &[2, 2], &[3, 2, 2])
        .unwrap();
    w.start_swmr().unwrap();
    let frame = |f: u8| {
        let v = 4 * f + 1;
        [v, v + 1, v + 2, v + 3]
    };
    let published = |n: u8| -> Vec<u8> { (0..n).flat_map(frame).collect() };

    // Two frames: a partial first band.
    w.append_frame(ds, &frame(0)).unwrap();
    w.append_frame(ds, &frame(1)).unwrap();
    w.flush().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![2, 2, 2]);
    assert_eq!(r.read_dataset::<u8>("frames").unwrap(), published(2));

    // Two more: the first band completes, the second starts.
    w.append_frame(ds, &frame(2)).unwrap();
    w.append_frame(ds, &frame(3)).unwrap();
    w.flush().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![4, 2, 2]);
    assert_eq!(r.read_dataset::<u8>("frames").unwrap(), published(4));

    // A flush with nothing new to say changes nothing.
    w.flush().unwrap();
    w.close().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.read_dataset::<u8>("frames").unwrap(), published(4));

    cleanup(&path);
}

/// `start_swmr` publishes the extent, which counts the frames of a band
/// still filling, so it writes that band first: a reader attaching right
/// after it sees the frames, not fill.
#[test]
fn start_swmr_publishes_a_partial_band() {
    let path = unique_tmp("start_band");
    let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
    let ds = w
        .create_streaming_dataset_chunked::<u8>("frames", &[2, 2], &[3, 2, 2])
        .unwrap();
    let frame = |f: u8| {
        let v = 4 * f + 1;
        [v, v + 1, v + 2, v + 3]
    };
    let published = |n: u8| -> Vec<u8> { (0..n).flat_map(frame).collect() };
    w.append_frame(ds, &frame(0)).unwrap();
    w.append_frame(ds, &frame(1)).unwrap();
    w.start_swmr().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![2, 2, 2]);
    assert_eq!(r.read_dataset::<u8>("frames").unwrap(), published(2));

    // The band completes after the switch and is written whole.
    w.append_frame(ds, &frame(2)).unwrap();
    w.flush().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.read_dataset::<u8>("frames").unwrap(), published(3));
    w.close().unwrap();

    cleanup(&path);
}

/// Resuming a multi-frame-chunk dataset (`chunk[0] > 1`) after `open_append`
/// is rejected with a clear error rather than corrupting the chunk grid.
#[test]
fn open_append_rejects_multi_frame_chunk_resume() {
    let path = unique_tmp("resume_mfc");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        // chunk[0] = 3 frames per chunk.
        let ds = w
            .create_streaming_dataset_chunked::<u8>("frames", &[2, 2], &[3, 2, 2])
            .unwrap();
        w.start_swmr().unwrap();
        for f in 0..3u8 {
            w.append_frame(ds, &[f, f, f, f]).unwrap();
        }
        w.close().unwrap();
    }

    let mut w = SwmrFileWriter::open_append_with_locking(&path, NO_LOCK).unwrap();
    let ds = w.dataset_index("frames").expect("reopened 'frames'");
    w.start_swmr().unwrap();
    let err = w
        .append_frame(ds, &[9u8, 9, 9, 9])
        .expect_err("multi-frame-chunk resume must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("multi-frame chunks"),
        "unexpected error: {msg}"
    );
    drop(w);

    cleanup(&path);
}

/// A 1-D int32 array dataset attribute set before `start_swmr` (the
/// AreaDetector NDArrayDimOffset/Binning/Reverse shape) round-trips: the
/// SWMR reader sees its name and the full read path recovers shape + values.
#[test]
fn dataset_array_attribute_round_trips() {
    use rust_hdf5::H5File;

    let path = unique_tmp("array_attr");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let axis = w.write_dataset::<i32>("axis", &[3], &[10, 20, 30]).unwrap();
        // Open-time 1-D int32 array dataset attribute.
        w.set_dataset_attr_array::<i32>(axis, "NDArrayDimOffset", &[3], &[0, 4, 8])
            .unwrap();
        // Wrong element count is rejected.
        assert!(w
            .set_dataset_attr_array::<i32>(axis, "bad", &[3], &[1, 2])
            .is_err());

        let frames = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        w.start_swmr().unwrap();
        w.append_frame(frames, &[1u8, 2, 3, 4]).unwrap();
        w.close().unwrap();
    }

    // SWMR reader sees the attribute name.
    {
        let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
        assert!(
            r.dataset_attr_names("axis")
                .unwrap()
                .iter()
                .any(|n| n == "NDArrayDimOffset"),
            "array attribute name not visible to SWMR reader"
        );
    }

    // Full read path verifies shape + values.
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("axis").unwrap();
        let raw = ds.attr("NDArrayDimOffset").unwrap().read_raw().unwrap();
        assert_eq!(raw.len(), 3 * 4);
        let got: Vec<i32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        assert_eq!(got, vec![0, 4, 8]);
    }

    cleanup(&path);
}

/// A fixed-shape multi-dimensional grid dataset (AreaDetector "extra
/// dimensions" layout) filled at explicit chunk positions out of order
/// round-trips: written positions recover their data, unwritten positions
/// read back as fill, and the SWMR reader sees the full bounded shape.
#[test]
fn grid_dataset_positioned_writes_round_trip() {
    use rust_hdf5::H5File;

    // Grid [Na=2, Nb=3, H=4, W=4], chunk [1,1,H,W]: one chunk == one frame.
    const NA: u64 = 2;
    const NB: u64 = 3;
    const H: u64 = 4;
    const W: u64 = 4;
    let fpp = (H * W) as usize; // elements per frame

    // Distinct, position-dependent pixel values.
    let frame = |a: u64, b: u64| -> Vec<u16> {
        (0..fpp as u16)
            .map(|p| (1000 * a + 100 * b) as u16 + p)
            .collect()
    };
    let bytes = |d: &[u16]| -> Vec<u8> { d.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let path = unique_tmp("grid");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let ds = w
            .create_grid_dataset::<u16>("grid", &[NA, NB, H, W], &[1, 1, H, W])
            .unwrap();
        w.start_swmr().unwrap();

        // Deliberately non-sequential placement; leave (a=1, b=2) unwritten.
        let order = [(0u64, 2u64), (1, 0), (0, 0), (1, 1), (0, 1)];
        for &(a, b) in &order {
            w.write_chunk_at(ds, &[a, b, 0, 0], &bytes(&frame(a, b)))
                .unwrap();
            w.flush().unwrap();
        }
        w.close().unwrap();
    }

    // SWMR reader sees the full bounded shape.
    {
        let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
        assert_eq!(r.dataset_shape("grid").unwrap(), vec![NA, NB, H, W]);
    }

    // Full read path: each written position matches, (1,2) is fill (zero).
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("grid").unwrap();
        assert_eq!(
            ds.shape(),
            vec![NA as usize, NB as usize, H as usize, W as usize]
        );
        let flat = ds.read_raw::<u16>().unwrap();
        let idx = |a: u64, b: u64| ((a * NB + b) as usize) * fpp;
        for a in 0..NA {
            for b in 0..NB {
                let got = &flat[idx(a, b)..idx(a, b) + fpp];
                if a == 1 && b == 2 {
                    assert!(
                        got.iter().all(|&v| v == 0),
                        "unwritten position (1,2) is not fill: {got:?}"
                    );
                } else {
                    assert_eq!(got, &frame(a, b)[..], "mismatch at ({a},{b})");
                }
            }
        }
    }

    cleanup(&path);
}

/// `write_chunk_at` rejects misuse: wrong-rank coordinates, out-of-grid
/// coordinates, and a non-grid (streaming) dataset.
#[test]
fn grid_dataset_positioned_write_rejects_misuse() {
    const H: u64 = 4;
    const W: u64 = 4;
    let chunk_bytes = vec![0u8; (H * W) as usize * 2]; // u16 full chunk

    let path = unique_tmp("grid_misuse");
    let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
    let grid = w
        .create_grid_dataset::<u16>("grid", &[2, 3, H, W], &[1, 1, H, W])
        .unwrap();
    // A streaming (extensible-array) dataset is not a grid.
    let stream = w
        .create_streaming_dataset::<u16>("stream", &[H, W])
        .unwrap();
    w.start_swmr().unwrap();

    // Wrong-rank coordinates (2 vs 4) are rejected.
    assert!(
        w.write_chunk_at(grid, &[0, 0], &chunk_bytes).is_err(),
        "wrong-rank coords accepted"
    );
    // Out-of-grid coordinate (a=2 with Na=2) is rejected.
    assert!(
        w.write_chunk_at(grid, &[2, 0, 0, 0], &chunk_bytes).is_err(),
        "out-of-grid coord accepted"
    );
    // Positioned write to a non-grid dataset is rejected.
    assert!(
        w.write_chunk_at(stream, &[0, 0], &chunk_bytes).is_err(),
        "positioned write to streaming dataset accepted"
    );

    w.close().unwrap();
    cleanup(&path);
}

/// Under SWMR a relocated chunk's old block must stay intact for readers still
/// holding the previous index, so the allocator cannot recycle it. An
/// unfiltered chunk therefore has only one way not to grow the file on
/// rewrite: overwrite the block it already occupies (libhdf5 does the same —
/// `H5D__chunk_flush_entry` leaves `must_alloc` false when the address is
/// already defined and no filter changes the stored size).
#[test]
fn swmr_chunk_rewrite_does_not_grow_the_file() {
    const H: u64 = 4;
    const W: u64 = 4;
    let chunk = |v: u16| -> Vec<u8> {
        (0..(H * W) as u16)
            .flat_map(|p| (v + p).to_le_bytes())
            .collect()
    };

    // Same file, written once vs rewritten eight times over.
    let write = |label: &str, rewrites: u16| -> (PathBuf, u64) {
        let path = unique_tmp(label);
        {
            let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
            let ds = w
                .create_grid_dataset::<u16>("grid", &[1, 1, H, W], &[1, 1, H, W])
                .unwrap();
            w.start_swmr().unwrap();
            for v in 1..=rewrites {
                w.write_chunk_at(ds, &[0, 0, 0, 0], &chunk(v * 100))
                    .unwrap();
                w.flush().unwrap();
            }
            w.close().unwrap();
        }
        let size = std::fs::metadata(&path).unwrap().len();
        (path, size)
    };

    let (once_path, once) = write("swmr_rewrite_1", 1);
    let (many_path, many) = write("swmr_rewrite_8", 8);
    assert_eq!(
        many, once,
        "8 SWMR rewrites of one chunk grew the file past a single write"
    );

    // The last write is the one that survives.
    let mut r = SwmrFileReader::open_with_locking(&many_path, NO_LOCK).unwrap();
    let data: Vec<u16> = r.read_dataset("grid").unwrap();
    assert_eq!(data[0], 800);
    drop(r);

    cleanup(&once_path);
    cleanup(&many_path);
}

/// Boundary: the typed reads check `T` against the stored element width.
/// An f64 dataset read as `i32` used to pass the divisibility check and
/// return twice as many garbage values.
#[test]
fn typed_reads_reject_a_mismatched_element_width() {
    let path = unique_tmp("typed_width");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        w.write_dataset::<f64>("d", &[2], &[1.5, -2.25]).unwrap();
        w.close().unwrap();
    }
    let mut r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
    assert_eq!(r.read_dataset::<f64>("d").unwrap(), vec![1.5, -2.25]);

    let err = r.read_dataset::<i32>("d").unwrap_err();
    assert!(
        err.to_string().contains("element size"),
        "unexpected error: {err}"
    );
    let err = r.read_slice::<i32>("d", &[0], &[1]).unwrap_err();
    assert!(
        err.to_string().contains("element size"),
        "unexpected error: {err}"
    );
    assert_eq!(r.read_slice::<f64>("d", &[1], &[1]).unwrap(), vec![-2.25]);
    cleanup(&path);
}

/// An attribute set or replaced after `start_swmr` is committed at close,
/// as libhdf5 1.14 commits `H5Acreate2`/`H5Awrite` under SWMR: on groups and
/// on datasets, with or without chunk writes, vlen or numeric.
#[test]
fn attribute_changes_during_swmr_are_committed_at_close() {
    let path = unique_tmp("swmr_attr_commit");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        w.create_group("/", "entry").unwrap();
        w.set_group_attr_string("/entry", "NX_class", "NXentry")
            .unwrap();
        let ds = w.create_streaming_dataset::<i32>("frames", &[4]).unwrap();
        w.set_dataset_attr_string(ds, "units", "mm").unwrap();
        w.set_dataset_attr_numeric(ds, "scale", &2i32).unwrap();
        let quiet = w.create_streaming_dataset::<i32>("quiet", &[4]).unwrap();
        w.start_swmr().unwrap();
        w.append_frame(ds, &[0u8; 16]).unwrap();

        w.set_group_attr_string("/entry", "NX_class", "NXdata")
            .unwrap();
        w.set_group_attr_numeric("/entry", "count", &5i64).unwrap();
        w.set_dataset_attr_string(ds, "units", "cm").unwrap();
        w.set_dataset_attr_string(ds, "long_name", "detector x")
            .unwrap();
        w.set_dataset_attr_numeric(ds, "scale", &3i32).unwrap();
        w.set_dataset_attr_numeric(quiet, "gain", &7i32).unwrap();
        // A flush publishes the new extent with the header as published.
        w.flush().unwrap();
        let mut mid = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
        assert_eq!(mid.dataset_attr_names("frames").unwrap().len(), 2);
        drop(mid);
        w.append_frame(ds, &[1u8; 16]).unwrap();
        w.flush().unwrap();
        w.close().unwrap();
    }

    let file = rust_hdf5::H5File::open(&path).unwrap();
    assert_eq!(file.dataset("frames").unwrap().shape(), vec![2, 4]);
    let entry = file.root_group().group("entry").unwrap();
    assert_eq!(entry.attr_string("NX_class").unwrap(), "NXdata");
    assert!(entry.attr_names().unwrap().contains(&"count".to_string()));
    let ds = file.dataset("frames").unwrap();
    assert_eq!(ds.attr("units").unwrap().read_string().unwrap(), "cm");
    assert_eq!(
        ds.attr("long_name").unwrap().read_string().unwrap(),
        "detector x"
    );
    assert_eq!(ds.attr("scale").unwrap().read_numeric::<i32>().unwrap(), 3);
    let quiet = file.dataset("quiet").unwrap();
    assert_eq!(
        quiet.attr("gain").unwrap().read_numeric::<i32>().unwrap(),
        7
    );
    drop(file);
    cleanup(&path);
}

/// A late attribute reaches a dataset whose attributes were already dense at
/// `start_swmr`, and one it pushes from compact to dense; a flush between
/// keeps publishing the header readers were given.
#[test]
fn late_attributes_reach_dense_storage() {
    let path = unique_tmp("swmr_attr_dense");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let dense = w.create_streaming_dataset::<i32>("dense", &[2]).unwrap();
        let edge = w.create_streaming_dataset::<i32>("edge", &[2]).unwrap();
        for i in 0..10i32 {
            w.set_dataset_attr_numeric(dense, &format!("a{i}"), &i)
                .unwrap();
        }
        for i in 0..8i32 {
            w.set_dataset_attr_numeric(edge, &format!("a{i}"), &i)
                .unwrap();
        }
        w.start_swmr().unwrap();
        w.append_frame(dense, &[0u8; 8]).unwrap();
        w.set_dataset_attr_numeric(dense, "late", &42i32).unwrap();
        w.set_dataset_attr_numeric(dense, "a3", &33i32).unwrap();
        for i in 8..10i32 {
            w.set_dataset_attr_numeric(edge, &format!("a{i}"), &i)
                .unwrap();
        }
        w.flush().unwrap();
        w.append_frame(dense, &[0u8; 8]).unwrap();
        w.close().unwrap();
    }

    let file = rust_hdf5::H5File::open(&path).unwrap();
    let dense = file.dataset("dense").unwrap();
    assert_eq!(dense.attr_names().unwrap().len(), 11);
    assert_eq!(
        dense.attr("late").unwrap().read_numeric::<i32>().unwrap(),
        42
    );
    assert_eq!(dense.attr("a3").unwrap().read_numeric::<i32>().unwrap(), 33);
    assert_eq!(dense.attr("a9").unwrap().read_numeric::<i32>().unwrap(), 9);
    let edge = file.dataset("edge").unwrap();
    assert_eq!(edge.attr_names().unwrap().len(), 10);
    assert_eq!(edge.attr("a9").unwrap().read_numeric::<i32>().unwrap(), 9);
    drop(file);
    cleanup(&path);
}

fn u16_bytes(v: impl IntoIterator<Item = u16>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One [2, 3] u16 frame per seed.
fn frame6(seed: u16) -> Vec<u8> {
    u16_bytes((0..6u16).map(|i| 10 * seed + i))
}

/// A stream bounded at `max_frames` stores that maximum on its frame axis —
/// NDFileHDF5's maxdims[0] = NumCapture — whether a chunk holds a whole
/// frame, a tile or a band of frames; an append past it is refused and the
/// file keeps the frames it held. `None` keeps the unlimited frame axis.
#[test]
fn bounded_streaming_dataset_keeps_its_max_frames() {
    use rust_hdf5::{H5File, H5Type};

    let path = unique_tmp("bounded");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let mut make = |name: &str, chunk: &[u64], max: Option<u64>| {
            w.create_streaming_dataset_with_max_frames(
                name,
                u16::hdf5_type(),
                &[2, 3],
                chunk,
                max,
                None,
            )
        };
        let whole = make("whole", &[1, 2, 3], Some(3)).unwrap();
        let tiled = make("tiled", &[1, 1, 3], Some(5)).unwrap();
        let banded = make("banded", &[2, 2, 3], Some(5)).unwrap();
        let open = make("open", &[1, 2, 3], None).unwrap();
        assert!(make("empty", &[1, 2, 3], Some(0)).is_err());
        w.start_swmr().unwrap();
        for s in 0..3 {
            for ds in [whole, tiled, banded, open] {
                w.append_frame(ds, &frame6(s)).unwrap();
            }
        }
        assert!(w.append_frame(whole, &frame6(9)).is_err());
        w.append_frame(banded, &frame6(3)).unwrap();
        w.append_frame(banded, &frame6(4)).unwrap();
        assert!(w.append_frame(banded, &frame6(9)).is_err());
        w.close().unwrap();
    }
    // A bounded stream resumes after `open_append` up to its maximum.
    {
        let mut w = SwmrFileWriter::open_append_with_locking(&path, NO_LOCK).unwrap();
        let tiled = w.dataset_index("tiled").unwrap();
        w.start_swmr().unwrap();
        w.append_frame(tiled, &frame6(3)).unwrap();
        w.append_frame(tiled, &frame6(4)).unwrap();
        assert!(w.append_frame(tiled, &frame6(9)).is_err());
        w.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    for (name, n, max) in [
        ("whole", 3, Some(3)),
        ("tiled", 5, Some(5)),
        ("banded", 5, Some(5)),
        ("open", 3, None),
    ] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.shape(), vec![n, 2, 3], "{name}");
        assert_eq!(
            ds.max_shape().unwrap(),
            vec![max, Some(2), Some(3)],
            "{name}"
        );
        let expect: Vec<u8> = (0..n as u16).flat_map(frame6).collect();
        assert_eq!(u16_bytes(ds.read_raw::<u16>().unwrap()), expect, "{name}");
    }
    drop(file);
    cleanup(&path);
}

/// `set_extent` moves a grid's extent inside its maximum, as `H5Dset_extent`
/// does for NDFileHDF5's extra dimensions: the file keeps the maximum and
/// shows the extent last set. Past the maximum it is refused, and on a
/// multi-frame-chunk stream, whose extent counts the frames its band buffer
/// holds, it is refused too.
#[test]
fn grid_set_extent_moves_within_the_maximum() {
    use rust_hdf5::H5File;

    let path = unique_tmp("grid_extent");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        let grid = w
            .create_grid_dataset::<u16>("grid", &[3, 2, 2, 3], &[1, 1, 2, 3])
            .unwrap();
        let banded = w
            .create_streaming_dataset_chunked::<u16>("banded", &[2, 3], &[2, 2, 3])
            .unwrap();
        w.set_extent(grid, &[1, 1, 2, 3]).unwrap();
        w.start_swmr().unwrap();
        // Each frame's position, and the extent that covers every position
        // written so far.
        let steps = [
            ([0, 0], [1, 1]),
            ([0, 1], [1, 2]),
            ([1, 0], [2, 2]),
            ([1, 1], [2, 2]),
        ];
        for (s, ([a, b], [na, nb])) in steps.into_iter().enumerate() {
            w.write_chunk_at(grid, &[a, b, 0, 0], &frame6(s as u16))
                .unwrap();
            w.set_extent(grid, &[na, nb, 2, 3]).unwrap();
            w.flush().unwrap();
        }
        assert!(w.set_extent(grid, &[4, 2, 2, 3]).is_err());
        w.append_frame(banded, &frame6(0)).unwrap();
        assert!(w.set_extent(banded, &[2, 2, 3]).is_err());
        w.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("grid").unwrap();
    assert_eq!(ds.shape(), vec![2, 2, 2, 3]);
    assert_eq!(
        ds.max_shape().unwrap(),
        vec![Some(3), Some(2), Some(2), Some(3)]
    );
    let expect: Vec<u8> = (0..4u16).flat_map(frame6).collect();
    assert_eq!(u16_bytes(ds.read_raw::<u16>().unwrap()), expect);
    assert_eq!(file.dataset("banded").unwrap().shape(), vec![1, 2, 3]);
    drop(file);
    cleanup(&path);
}
