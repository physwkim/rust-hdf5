//! Blosc chunks as rust-hdf5 0.7.8 wrote them, before it wrote c-blosc
//! frames: `fixtures/blosc_rs078.h5`, from `fixtures/gen_blosc_rs078.rs`.
#![cfg(feature = "blosc")]

use rust_hdf5::format::messages::filter::reverse_filters;
use rust_hdf5::{FilterPipeline, H5File};

const FIXTURE: &str = "tests/fixtures/blosc_rs078.h5";

fn value(i: usize) -> u16 {
    ((i / 4) * 3 + i % 3) as u16
}

#[test]
fn reads_the_blosc_chunks_rust_hdf5_0_7_8_wrote() {
    let want: Vec<u16> = (0..1024).map(value).collect();
    let second: Vec<u8> = want[512..].iter().flat_map(|v| v.to_le_bytes()).collect();
    // zlib and zstd need their features
    let codecs = (0..6)
        .filter(|&c| (c != 4 || cfg!(feature = "deflate")) && (c != 5 || cfg!(feature = "zstd")));
    let mut names: Vec<String> = codecs
        .flat_map(|c| (0..3).map(move |sh| format!("c{c}_sh{sh}")))
        .collect();
    names.push("default_codec".into());
    let file = H5File::open(FIXTURE).unwrap();
    for name in names {
        let ds = file.dataset(&name).unwrap();
        assert_eq!(ds.read_raw::<u16>().unwrap(), want, "{name}");
        assert_eq!(
            ds.read_slice::<u16>(&[500], &[24]).unwrap(),
            want[500..524],
            "{name} across the chunks"
        );
        let pipeline = FilterPipeline {
            filters: ds.filters().unwrap(),
        };
        let (raw, mask) = ds.read_chunk_raw_at(&[1]).unwrap();
        assert_eq!(mask, 0);
        assert_eq!(reverse_filters(&pipeline, &raw).unwrap(), second, "{name}");
    }
}

/// An LZ4 frame ends before a c-blosc frame's bstarts table would.
#[test]
fn reads_a_0_7_8_frame_shorter_than_a_bstarts_table() {
    let file = H5File::open(FIXTURE).unwrap();
    let ds = file.dataset("one_byte").unwrap();
    assert_eq!(ds.read_chunk_raw_at(&[0]).unwrap().0.len(), 18);
    assert_eq!(ds.read_raw::<u8>().unwrap(), [0xA5]);
}

/// 0.7.8 stored a BloscLZ input under 16 bytes as a header alone.
#[test]
fn a_0_7_8_chunk_written_without_its_payload_is_refused() {
    let file = H5File::open(FIXTURE).unwrap();
    let ds = file.dataset("lost").unwrap();
    assert_eq!(ds.read_chunk_raw_at(&[0]).unwrap().0.len(), 16);
    assert!(ds.read_raw::<u8>().is_err());
    assert!(ds.read_slice::<u8>(&[2], &[3]).is_err());
}
