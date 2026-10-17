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
        let packed_grid = w
            .create_grid_dataset_compressed_with_datatype(
                "packed_grid",
                dt.clone(),
                &[2, 4, 8],
                &[1, 4, 8],
                FilterPipeline::nbit(&dt, 32),
            )
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
        for (c, seed) in [(0u64, 8u16), (1, 9)] {
            w.write_chunk_at(packed_grid, &[c, 0, 0], &bytes(&frame(seed)))
                .unwrap();
        }
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
    for name in ["grid", "packed_grid"] {
        let grid = f.dataset(name).unwrap();
        assert_eq!(grid.datatype().unwrap(), dt, "{name}");
        assert_eq!(
            grid.read_raw::<u16>().unwrap(),
            [frame(8), frame(9)].concat(),
            "{name}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// `set_dataset_fill_value_bytes` takes the stored type's host image and puts
/// it in the declared order; the typed setter goes through the same step, so
/// a big-endian dataset gets a big-endian fill either way.
#[test]
fn fill_value_bytes_follow_the_declared_byte_order() {
    let path = tmp("fill");
    let be = DatatypeMessage::FixedPoint {
        size: 2,
        byte_order: ByteOrder::BigEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 12,
    };
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let by_bytes = w
            .create_grid_dataset_compressed_with_datatype(
                "by_bytes",
                be.clone(),
                &[2, 4],
                &[1, 4],
                FilterPipeline::nbit(&be, 4),
            )
            .unwrap();
        let typed = w
            .create_grid_dataset_with_datatype("typed", be.clone(), &[2, 4], &[1, 4])
            .unwrap();
        w.set_dataset_fill_value_bytes(by_bytes, &0x0abcu16.to_ne_bytes())
            .unwrap();
        w.set_dataset_fill_value(typed, &0x0123u16).unwrap();
        assert!(w.set_dataset_fill_value_bytes(typed, &[1, 2, 3]).is_err());
        assert!(w.set_dataset_fill_value_bytes(99, &[1, 2]).is_err());
        w.start_swmr().unwrap();
        let row: Vec<u8> = [1u16, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        w.write_chunk_at(by_bytes, &[0, 0], &row).unwrap();
        w.write_chunk_at(typed, &[0, 0], &row).unwrap();
        w.close().unwrap();
    }
    let f = H5File::open(&path).unwrap();
    for (name, fill) in [("by_bytes", 0x0abcu16), ("typed", 0x0123)] {
        let raw = f.dataset(name).unwrap().read_raw_bytes().unwrap();
        let expect: Vec<u8> = [1u16, 2, 3, 4, fill, fill, fill, fill]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        assert_eq!(raw, expect, "{name}");
    }
    let _ = std::fs::remove_file(&path);
}
