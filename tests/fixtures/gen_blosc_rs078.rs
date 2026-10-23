//! Generate `blosc_rs078.h5`: blosc chunks as rust-hdf5 0.7.8 wrote them,
//! for `tests/blosc_rs078.rs`. The main of a binary crate depending on
//! rust-hdf5 0.7.8 (tag v0.7.8) with `features = ["all_filters"]`:
//!
//!   cargo run --release -- tests/fixtures/blosc_rs078.h5
//!
//! Each `c{codec}_sh{shuffle}` dataset holds `value(i)` for i in 0..1024 as
//! u16, in two chunks, with the cd_values epics-rs 0.32.0 NDFileHDF5 set
//! (`[2, 2, 2, 0, 5, shuffle, codec]`); `default_codec` gives no codec, so
//! 0.7.8 compressed it with LZ4. `one_byte` is an LZ4 frame shorter than
//! the bstarts table would end; `lost` holds 8 bytes 0.7.8 stored as a
//! frame without a payload. `ts256` and `ts300` hold `byte(i)` for i in
//! 0..4096 as u8, shuffled by a typesize 0.7.8 kept only the low byte of
//! in the header.
use rust_hdf5::format::messages::filter::FILTER_BLOSC;
use rust_hdf5::{Filter, FilterPipeline, H5File};

fn value(i: usize) -> u16 {
    ((i / 4) * 3 + i % 3) as u16
}

fn byte(i: usize) -> u8 {
    (i % 251) as u8 ^ (i / 256) as u8
}

fn blosc(cd_values: Vec<u32>) -> FilterPipeline {
    FilterPipeline {
        filters: vec![Filter {
            id: FILTER_BLOSC,
            flags: 0,
            cd_values,
        }],
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("output path");
    let file = H5File::create(&path).unwrap();
    let data: Vec<u16> = (0..1024).map(value).collect();
    let mut sets: Vec<(String, Vec<u32>)> = Vec::new();
    for codec in 0..6 {
        for shuffle in 0..3 {
            sets.push((format!("c{codec}_sh{shuffle}"), vec![2, 2, 2, 0, 5, shuffle, codec]));
        }
    }
    sets.push(("default_codec".into(), vec![2, 2, 2, 0, 5, 1]));
    for (name, cd) in sets {
        file.new_dataset::<u16>()
            .shape([data.len()])
            .chunk(&[data.len() / 2])
            .filter_pipeline(blosc(cd))
            .create(name.as_str())
            .unwrap()
            .write_raw(&data)
            .unwrap();
    }
    // 0.7.8's LZ4 frame for one byte, 18 bytes long
    file.new_dataset::<u8>()
        .shape([1])
        .chunk(&[1])
        .filter_pipeline(blosc(vec![2, 2, 1, 0, 5, 0, 1]))
        .create("one_byte")
        .unwrap()
        .write_raw(&[0xA5u8])
        .unwrap();
    file.new_dataset::<u8>()
        .shape([8])
        .chunk(&[8])
        .filter_pipeline(blosc(vec![2, 2, 1, 0, 5, 0, 0]))
        .create("lost")
        .unwrap()
        .write_raw(&[1u8, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    let bytes: Vec<u8> = (0..4096).map(byte).collect();
    for typesize in [256, 300] {
        file.new_dataset::<u8>()
            .shape([bytes.len()])
            .chunk(&[bytes.len() / 2])
            .filter_pipeline(blosc(vec![2, 2, typesize, 0, 5, 1, 1]))
            .create(format!("ts{typesize}").as_str())
            .unwrap()
            .write_raw(&bytes)
            .unwrap();
    }
    file.close().unwrap();
}
