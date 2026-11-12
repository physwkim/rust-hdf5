//! `SwmrFileWriter::append_frame_raw` / `write_chunk_raw_at`: already-filtered
//! chunks are stored verbatim under the dataset's recorded pipeline, and
//! `SwmrFileReader::read_chunk_raw` reads them back as the writer goes.

#![cfg(feature = "deflate")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::messages::filter::{apply_filters, reverse_filters, reverse_filters_masked};
use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};
use rust_hdf5::{FileLocking, FilterPipeline, H5File};

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_swmr_raw_{label}_{}_{n}.h5",
        std::process::id()
    ))
}

fn frame(seed: u16) -> Vec<u8> {
    (0..32u16)
        .flat_map(|i| (i * 3 + seed).to_le_bytes())
        .collect()
}

#[test]
fn append_frame_raw_stores_bytes_verbatim() {
    let path = tmp("append");
    let pipeline = FilterPipeline::deflate(4);
    let packed: Vec<Vec<u8>> = (0..3)
        .map(|s| apply_filters(&pipeline, &frame(s)).unwrap())
        .collect();
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let ds = w
            .create_streaming_dataset_compressed::<u16>("frames", &[4, 8], pipeline.clone())
            .unwrap();
        w.start_swmr().unwrap();
        w.append_frame_raw(ds, &packed[0], 0).unwrap();
        w.append_frame_raw(ds, &packed[1], 0).unwrap();
        // Mask bit 0: deflate was not applied, so the stored bytes are the
        // frame itself.
        w.append_frame_raw(ds, &frame(2), 1).unwrap();
        w.flush().unwrap();
        w.close().unwrap();
    }
    let f = H5File::open(&path).unwrap();
    let ds = f.dataset("frames").unwrap();
    assert_eq!(ds.shape(), vec![3, 4, 8]);
    assert_eq!(
        ds.read_chunk_raw_at(&[0, 0, 0]).unwrap(),
        (packed[0].clone(), 0)
    );
    assert_eq!(ds.read_chunk_raw_at(&[2, 0, 0]).unwrap(), (frame(2), 1));
    let expect: Vec<u16> = (0..3u16)
        .flat_map(|s| (0..32u16).map(move |i| i * 3 + s))
        .collect();
    assert_eq!(ds.read_raw::<u16>().unwrap(), expect);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn append_frame_raw_refuses_layouts_without_one_frame_per_chunk() {
    let path = tmp("refuse");
    let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
    let plain = w.create_streaming_dataset::<u16>("plain", &[4, 8]).unwrap();
    let tiled = w
        .create_streaming_dataset_tiled_compressed::<u16>(
            "tiled",
            &[4, 8],
            &[2, 8],
            FilterPipeline::deflate(1),
        )
        .unwrap();
    let banded = w
        .create_streaming_dataset_chunked_compressed::<u16>(
            "banded",
            &[4, 8],
            &[2, 4, 8],
            FilterPipeline::deflate(1),
        )
        .unwrap();
    let grid = w
        .create_grid_dataset::<u16>("grid", &[2, 4, 8], &[1, 4, 8])
        .unwrap();
    for ds in [plain, tiled, banded, grid] {
        assert!(w.append_frame_raw(ds, &frame(0), 0).is_err(), "ds {ds}");
    }
    w.close().unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_chunk_raw_at_places_filtered_grid_chunks() {
    let path = tmp("grid");
    let pipeline = FilterPipeline::deflate(4);
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let grid = w
            .create_grid_dataset_compressed::<u16>("grid", &[3, 4, 8], &[1, 4, 8], pipeline.clone())
            .unwrap();
        let plain = w
            .create_grid_dataset::<u16>("plain", &[2, 4, 8], &[1, 4, 8])
            .unwrap();
        w.start_swmr().unwrap();
        assert!(w
            .write_chunk_raw_at(plain, &[0, 0, 0], &frame(0), 0)
            .is_err());
        let packed = apply_filters(&pipeline, &frame(7)).unwrap();
        w.write_chunk_raw_at(grid, &[1, 0, 0], &packed, 0).unwrap();
        w.write_chunk_raw_at(grid, &[0, 0, 0], &frame(5), 1)
            .unwrap();
        // The compressing path on the same dataset.
        w.write_chunk_at(grid, &[2, 0, 0], &frame(9)).unwrap();
        w.flush().unwrap();
        w.close().unwrap();
    }
    let f = H5File::open(&path).unwrap();
    let ds = f.dataset("grid").unwrap();
    assert_eq!(ds.read_chunk_raw_at(&[0, 0, 0]).unwrap(), (frame(5), 1));
    assert_eq!(ds.read_chunk_raw_at(&[1, 0, 0]).unwrap(), (packed_of(7), 0));
    let expect: Vec<u16> = [5u16, 7, 9]
        .iter()
        .flat_map(|&s| (0..32u16).map(move |i| i * 3 + s))
        .collect();
    assert_eq!(ds.read_raw::<u16>().unwrap(), expect);
    let _ = std::fs::remove_file(&path);
}

/// A bounded stream takes already-filtered frames up to its maximum and
/// keeps that maximum on its frame axis.
#[test]
fn append_frame_raw_fills_a_bounded_stream() {
    use rust_hdf5::H5Type;

    let path = tmp("bounded");
    let pipeline = FilterPipeline::deflate(4);
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let ds = w
            .create_streaming_dataset_with_max_frames(
                "frames",
                u16::hdf5_type(),
                &[4, 8],
                &[1, 4, 8],
                Some(2),
                Some(pipeline.clone()),
            )
            .unwrap();
        w.start_swmr().unwrap();
        w.append_frame_raw(ds, &packed_of(0), 0).unwrap();
        w.append_frame_raw(ds, &frame(1), 1).unwrap();
        assert!(w.append_frame_raw(ds, &packed_of(2), 0).is_err());
        w.close().unwrap();
    }
    let f = H5File::open(&path).unwrap();
    let ds = f.dataset("frames").unwrap();
    assert_eq!(ds.shape(), vec![2, 4, 8]);
    assert_eq!(ds.max_shape().unwrap(), vec![Some(2), Some(4), Some(8)]);
    assert_eq!(ds.read_chunk_raw_at(&[0, 0, 0]).unwrap(), (packed_of(0), 0));
    assert_eq!(ds.read_chunk_raw_at(&[1, 0, 0]).unwrap(), (frame(1), 1));
    let expect: Vec<u16> = (0..2u16)
        .flat_map(|s| (0..32u16).map(move |i| i * 3 + s))
        .collect();
    assert_eq!(ds.read_raw::<u16>().unwrap(), expect);
    let _ = std::fs::remove_file(&path);
}

fn packed_of(seed: u16) -> Vec<u8> {
    apply_filters(&FilterPipeline::deflate(4), &frame(seed)).unwrap()
}

/// A reader that stays open and refreshes reads each frame's stored bytes
/// once the writer has flushed it, and `None` for the frame after it.
#[test]
fn a_refreshing_reader_reads_each_flushed_frame_raw() {
    let path = tmp("follow");
    let pipeline = FilterPipeline::deflate(4);
    let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
    let ds = w
        .create_streaming_dataset_compressed::<u16>("frames", &[4, 8], pipeline.clone())
        .unwrap();
    w.start_swmr().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    assert_eq!(r.read_chunk_raw("frames", &[0, 0, 0]).unwrap(), None);
    for i in 0..40u16 {
        w.append_frame_raw(ds, &packed_of(i), 0).unwrap();
        w.flush().unwrap();
        // Published but not yet refreshed to: the reader's extent is the
        // one it read last.
        assert_eq!(
            r.read_chunk_raw("frames", &[u64::from(i), 0, 0]).unwrap(),
            None
        );
        r.refresh().unwrap();
        let (raw, mask) = r
            .read_chunk_raw("frames", &[u64::from(i), 0, 0])
            .unwrap()
            .unwrap_or_else(|| panic!("frame {i} not found after refresh"));
        assert_eq!((mask, &raw), (0, &packed_of(i)), "frame {i}");
        assert_eq!(reverse_filters(&pipeline, &raw).unwrap(), frame(i));
        assert_eq!(
            r.read_chunk_raw("frames", &[u64::from(i) + 1, 0, 0])
                .unwrap(),
            None
        );
        assert_eq!(
            r.read_chunk_raw("frames", &[0, 0, 0]).unwrap(),
            Some((packed_of(0), 0))
        );
    }
    w.close().unwrap();
    let _ = std::fs::remove_file(&path);
}

/// The reader reports the pipeline the writer recorded, and that pipeline
/// undoes a live raw chunk under the chunk's own mask: both filters applied,
/// and deflate skipped (mask bit 1). An unfiltered dataset reports none; an
/// unknown name is an error.
#[test]
fn a_live_reader_decodes_raw_chunks_with_the_reported_filters() {
    let path = tmp("filters");
    let pipeline = FilterPipeline::shuffle_deflate(2, 4);
    let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
    let ds = w
        .create_streaming_dataset_compressed::<u16>("frames", &[4, 8], pipeline.clone())
        .unwrap();
    w.create_streaming_dataset::<u16>("plain", &[4, 8]).unwrap();
    w.start_swmr().unwrap();
    let packed = apply_filters(&pipeline, &frame(0)).unwrap();
    let shuffled = apply_filters(&FilterPipeline::shuffle(2), &frame(1)).unwrap();
    w.append_frame_raw(ds, &packed, 0).unwrap();
    w.append_frame_raw(ds, &shuffled, 0b10).unwrap();
    w.flush().unwrap();

    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    let filters = r.dataset_filters("frames").unwrap();
    assert_eq!(filters, pipeline.filters);
    let reported = FilterPipeline { filters };
    for i in 0..2u16 {
        let (raw, mask) = r
            .read_chunk_raw("frames", &[u64::from(i), 0, 0])
            .unwrap()
            .unwrap();
        assert_eq!(
            reverse_filters_masked(&reported, &raw, mask).unwrap(),
            frame(i),
            "frame {i}"
        );
    }
    assert_eq!(r.dataset_filters("plain").unwrap(), Vec::new());
    assert!(r.dataset_filters("missing").is_err());
    w.close().unwrap();
    let _ = std::fs::remove_file(&path);
}

/// `None` is for a chunk not there yet; coordinates no chunk can ever have
/// are an error.
#[test]
fn read_chunk_raw_tells_not_yet_from_never() {
    use rust_hdf5::H5Type;

    let path = tmp("never");
    let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
    let ds = w
        .create_streaming_dataset_with_max_frames(
            "frames",
            u16::hdf5_type(),
            &[4, 8],
            &[1, 4, 8],
            Some(2),
            Some(FilterPipeline::deflate(4)),
        )
        .unwrap();
    let grid = w
        .create_grid_dataset_compressed::<u16>(
            "grid",
            &[3, 4, 8],
            &[1, 4, 8],
            FilterPipeline::deflate(4),
        )
        .unwrap();
    w.set_extent(grid, &[1, 4, 8]).unwrap();
    w.start_swmr().unwrap();
    w.append_frame_raw(ds, &packed_of(0), 0).unwrap();
    // Indexed, but beyond the extent the grid publishes.
    w.write_chunk_raw_at(grid, &[2, 0, 0], &packed_of(2), 0)
        .unwrap();
    w.flush().unwrap();

    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    assert_eq!(
        r.read_chunk_raw("frames", &[0, 0, 0]).unwrap(),
        Some((packed_of(0), 0))
    );
    assert_eq!(r.read_chunk_raw("frames", &[1, 0, 0]).unwrap(), None);
    for (coords, why) in [
        (&[2u64, 0, 0][..], "past the maximum frame count"),
        (&[0, 1, 0], "past a fixed dimension"),
        (&[0, 0], "wrong rank"),
        (&[0, 0, 0, 0], "wrong rank"),
    ] {
        assert!(r.read_chunk_raw("frames", coords).is_err(), "{why}");
    }
    assert_eq!(r.read_chunk_raw("grid", &[2, 0, 0]).unwrap(), None);
    assert_eq!(r.read_chunk_raw("grid", &[0, 0, 0]).unwrap(), None);

    w.set_extent(grid, &[3, 4, 8]).unwrap();
    w.flush().unwrap();
    r.refresh().unwrap();
    assert_eq!(
        r.read_chunk_raw("grid", &[2, 0, 0]).unwrap(),
        Some((packed_of(2), 0))
    );
    assert_eq!(r.read_chunk_raw("grid", &[1, 0, 0]).unwrap(), None);
    w.close().unwrap();
    let _ = std::fs::remove_file(&path);
}

/// The single-chunk index's only chunk is at the origin, and a dataset with
/// no chunks has no raw chunk to read.
#[test]
fn read_chunk_raw_on_single_chunk_and_contiguous_datasets() {
    let path = tmp("single");
    let vals: Vec<u16> = (0..32).collect();
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<u16>()
            .shape([4usize, 8])
            .chunk(&[4, 8])
            .create("one")
            .unwrap()
            .write_raw(&vals)
            .unwrap();
        file.new_dataset::<u16>()
            .shape([32usize])
            .create("flat")
            .unwrap()
            .write_raw(&vals)
            .unwrap();
        file.close().unwrap();
    }
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(r.read_chunk_raw("one", &[0, 0]).unwrap(), Some((bytes, 0)));
    let err = r.read_chunk_raw("one", &[1, 0]).unwrap_err().to_string();
    assert!(err.contains("single-chunk index"), "{err}");
    assert!(r.read_chunk_raw("one", &[0]).is_err());
    let err = r.read_chunk_raw("flat", &[0]).unwrap_err().to_string();
    assert!(err.contains("is not chunked"), "{err}");
    let _ = std::fs::remove_file(&path);
}
