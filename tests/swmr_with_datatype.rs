//! `SwmrFileWriter::*_with_datatype`: streaming and grid datasets built on a
//! stored datatype no carrier `T` describes (a 12-bit N-bit type).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::swmr::SwmrFileWriter;
use rust_hdf5::{ByteOrder, DatatypeMessage, FileLocking, FilterPipeline, H5File};

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_swmr_dt_{label}_{}_{n}.h5",
        std::process::id()
    ))
}

fn u12() -> DatatypeMessage {
    DatatypeMessage::FixedPoint {
        size: 2,
        byte_order: ByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 12,
    }
}

fn frame(seed: u16) -> Vec<u16> {
    (0..32u16).map(|i| (i * 97 + seed) & 0x0fff).collect()
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn with_datatype_constructors_store_the_given_type() {
    let path = tmp("all");
    let dt = u12();
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let plain = w
            .create_streaming_dataset_with_datatype("plain", dt.clone(), &[4, 8])
            .unwrap();
        let packed = w
            .create_streaming_dataset_compressed_with_datatype(
                "packed",
                dt.clone(),
                &[4, 8],
                FilterPipeline::nbit(&dt, 32),
            )
            .unwrap();
        let tiled = w
            .create_streaming_dataset_tiled_compressed_with_datatype(
                "tiled",
                dt.clone(),
                &[4, 8],
                &[2, 8],
                FilterPipeline::nbit(&dt, 16),
            )
            .unwrap();
        let banded = w
            .create_streaming_dataset_chunked_compressed_with_datatype(
                "banded",
                dt.clone(),
                &[4, 8],
                &[2, 4, 8],
                FilterPipeline::nbit(&dt, 64),
            )
            .unwrap();
        let grid = w
            .create_grid_dataset_with_datatype("grid", dt.clone(), &[2, 4, 8], &[1, 4, 8])
            .unwrap();
        w.start_swmr().unwrap();
        for s in 0..3 {
            for ds in [plain, packed, tiled, banded] {
                w.append_frame(ds, &bytes(&frame(s))).unwrap();
            }
        }
        w.write_chunk_at(grid, &[1, 0, 0], &bytes(&frame(9)))
            .unwrap();
        w.write_chunk_at(grid, &[0, 0, 0], &bytes(&frame(8)))
            .unwrap();
        w.close().unwrap();
    }
    let f = H5File::open(&path).unwrap();
    let expect: Vec<u16> = (0..3).flat_map(frame).collect();
    for name in ["plain", "packed", "tiled", "banded"] {
        let ds = f.dataset(name).unwrap();
        assert_eq!(ds.datatype().unwrap(), dt, "{name}");
        assert_eq!(ds.shape(), vec![3, 4, 8], "{name}");
        assert_eq!(ds.read_raw::<u16>().unwrap(), expect, "{name}");
    }
    let grid = f.dataset("grid").unwrap();
    assert_eq!(grid.datatype().unwrap(), dt);
    assert_eq!(
        grid.read_raw::<u16>().unwrap(),
        [frame(8), frame(9)].concat()
    );
    let _ = std::fs::remove_file(&path);
}
