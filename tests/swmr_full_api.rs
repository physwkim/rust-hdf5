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
    assert!(r.group_attr_names("/entry").iter().any(|n| n == "version"));
    assert_eq!(
        r.group_attr_string("/", "file_name").unwrap(),
        "metadata.h5"
    );

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
        let r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
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
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
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
        let r = SwmrFileReader::open_with_locking(&path, NO_LOCK).unwrap();
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

/// Every attribute mutation after `start_swmr` is refused, libhdf5's rule
/// for SWMR writes. Object headers are frozen once streaming starts, so a
/// change only reached the file when its header happened to be rebuilt at
/// close (group attrs always, dataset attrs only alongside chunk writes)
/// and was silently dropped otherwise — and replacing a vlen value
/// stranded its old 4096-byte collection forever, since a streaming
/// reader may still hold the references.
#[test]
fn attribute_changes_during_swmr_are_refused() {
    let path = unique_tmp("swmr_attr_freeze");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, NO_LOCK).unwrap();
        w.create_group("/", "entry").unwrap();
        w.set_group_attr_string("/entry", "NX_class", "NXentry")
            .unwrap();
        let ds = w.create_streaming_dataset::<i32>("frames", &[4]).unwrap();
        w.set_dataset_attr_string(ds, "units", "mm").unwrap();
        w.set_dataset_attr_numeric(ds, "scale", &2i32).unwrap();
        w.start_swmr().unwrap();

        let err = w
            .set_group_attr_string("/entry", "NX_class", "NXdata")
            .expect_err("vlen replace under SWMR must be refused");
        assert!(format!("{err}").contains("SWMR"), "got: {err}");
        w.set_dataset_attr_string(ds, "units", "cm")
            .expect_err("vlen replace under SWMR must be refused");
        w.set_dataset_attr_string(ds, "long_name", "detector x")
            .expect_err("a new vlen attribute under SWMR must be refused");
        w.set_dataset_attr_numeric(ds, "scale", &3i32)
            .expect_err("a numeric replace under SWMR must be refused");
        w.close().unwrap();
    }

    // No stranded collection: the refused new-attribute call must not have
    // written a heap block ("GCOL" appears once per pre-start vlen attr).
    let bytes = std::fs::read(&path).unwrap();
    let gcols = bytes.windows(4).filter(|w| *w == b"GCOL").count();
    assert_eq!(gcols, 2, "only NX_class and units may own a collection");

    // The pre-start values are what the file holds.
    let file = rust_hdf5::H5File::open(&path).unwrap();
    let entry = file.root_group().group("entry").unwrap();
    assert_eq!(entry.attr_string("NX_class").unwrap(), "NXentry");
    let ds = file.dataset("frames").unwrap();
    assert_eq!(ds.attr("units").unwrap().read_string().unwrap(), "mm");
    assert_eq!(ds.attr("scale").unwrap().read_numeric::<i32>().unwrap(), 2);
    drop(file);
    cleanup(&path);
}
