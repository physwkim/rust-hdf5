//! `SwmrFileWriter::append_frame_raw`: already-filtered frames are stored verbatim under the dataset's recorded pipeline.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::messages::filter::apply_filters;
use rust_hdf5::swmr::SwmrFileWriter;
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
