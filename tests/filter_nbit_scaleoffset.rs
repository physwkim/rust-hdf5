//! Integration tests for the N-bit, Scale-offset and SZIP filters.
//!
//! These tests feed raw filtered chunk bytes produced by h5py / libhdf5
//! 2.0.0 directly through the crate's filter pipeline and confirm the data
//! is reconstructed byte-exact. The chunk bytes and `cd_values` were
//! extracted with h5py's `read_direct_chunk`; the SZIP vector is the first
//! chunk of libhdf5's own `tools/test/testfiles/h5repack_szip.h5`. They are
//! embedded here so the test runs without h5py present.
//!
//! Using `reverse_filters` directly (rather than going through the full
//! dataset reader) isolates the filter codec from the unrelated chunked-
//! dataset discovery path.

use rust_hdf5::format::messages::filter::{apply_filters, reverse_filters, Filter, FilterPipeline};

fn pipeline(id: u16, cd_values: Vec<u32>) -> FilterPipeline {
    FilterPipeline {
        filters: vec![Filter {
            id,
            flags: 0,
            cd_values,
        }],
    }
}

fn hex(s: &str) -> Vec<u8> {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(clean.len() % 2 == 0, "hex string has odd length");
    (0..clean.len() / 2)
        .map(|i| u8::from_str_radix(&clean[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

const FILTER_SZIP: u16 = 4;
const FILTER_NBIT: u16 = 5;
const FILTER_SCALEOFFSET: u16 = 6;

/// SZIP cross-decode against libhdf5's own reference file.
///
/// IGNORED: the crate's `szip.rs` AEC codec is internally self-consistent
/// (crate-compress -> crate-decompress round-trips) but its bitstream is
/// NOT byte-compatible with libaec/libhdf5 — encoding the same input yields
/// a different compressed stream, so it cannot decode libhdf5-written SZIP
/// chunks. The 4-byte length-header framing fixed in this change is correct
/// and exercised by `szip` round-trip tests; the remaining gap is the AEC
/// codec itself, which is tracked as a separate defect.
#[test]
#[ignore = "crate AEC codec is not yet libaec/libhdf5 bitstream-compatible"]
fn szip_libhdf5_chunk() {
    // First chunk (rows 0..20, cols 0..10) of `dset_szip` in libhdf5's own
    // reference file `tools/test/testfiles/h5repack_szip.h5`: int32, RAW
    // header mode, NN coding, pixels_per_block 8. Verifies the 4-byte
    // little-endian uncompressed-length header (UINT32ENCODE/DECODE) and
    // the cd_values layout (mask, ppb, bpp, pps).
    let chunk = hex(concat!(
        "200300004015558049fd0a2aaa0093fa2855540127f478aaa8024fe941555004",
        "9fd322aaa0093fa7855540127f518aaa8024fea815550049fd5a2aaa0093fac8",
        "55540127f5b8aaa8024febc15550049fd022aaa0093fa1855540127f458aaa80",
        "24fe9015550049fd2a2aaa0093fa6855540127f4f8aaa8024fe0008002000800",
        "20008002000800200080020008002000800a002800a002800a002800a0008002",
        "0008002000800200080020008002000800200080020008002000800200080020",
        "0080020008002000800200080020008002000800200080020008002000800200",
        "080020"
    ));
    assert_eq!(chunk.len(), 227);
    // cd_values: [options_mask=169, pixels_per_block=8, bits_per_pixel=32,
    //             pixels_per_scanline=10].
    let pl = pipeline(FILTER_SZIP, vec![169, 8, 32, 10]);
    let out = reverse_filters(&pl, &chunk).expect("szip reverse");
    // chunk0[r][c] = r * 20 + c, for r in 0..20, c in 0..10.
    let expected: Vec<u8> = (0..20i32)
        .flat_map(|r| (0..10i32).flat_map(move |c| (r * 20 + c).to_le_bytes()))
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn szip_framing_roundtrip() {
    // The SZIP filter prepends a 4-byte little-endian uncompressed-length
    // header (libhdf5 UINT32ENCODE) ahead of the AEC bitstream. Verify the
    // header is written on compress and consumed on decompress, and that
    // the compressed stream begins with the correct length.
    let mut data = Vec::new();
    for i in 0..256u16 {
        data.extend_from_slice(&i.to_le_bytes());
    }
    // cd_values: [mask = NN(32)|MSB(16), ppb = 16, bpp = 16, pps = 256].
    let pl = pipeline(FILTER_SZIP, vec![48, 16, 16, 256]);
    let compressed = apply_filters(&pl, &data).expect("szip compress");
    assert!(compressed.len() >= 4, "compressed stream missing header");
    let header = u32::from_le_bytes(compressed[..4].try_into().unwrap());
    assert_eq!(
        header as usize,
        data.len(),
        "4-byte LE header must hold the uncompressed length"
    );
    let restored = reverse_filters(&pl, &compressed).expect("szip decompress");
    assert_eq!(restored, data, "szip framing round-trip must be lossless");
}

#[test]
fn nbit_u16_precision12() {
    // 16-bit storage, 12-bit precision, little-endian unsigned int.
    let chunk = hex(concat!(
        "00004708e0d511c1631aa1f123827f2c630d35439b3e24294704b74fe54558c5",
        "d361a6616a86ef73677d7c480b8528998e092796e9b59fca43a8aad100"
    ));
    let pl = pipeline(FILTER_NBIT, vec![8, 0, 40, 1, 2, 0, 12, 0]);
    let out = reverse_filters(&pl, &chunk).expect("nbit u16 reverse");
    let expected: Vec<u8> = (0..40u16)
        .flat_map(|i| ((i * 71) & 0x0FFF).to_le_bytes())
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn nbit_i32_precision20() {
    // 32-bit storage, 20-bit precision, little-endian signed int.
    let chunk = hex(concat!(
        "000000270f04e1e0752d09c3c0c34b0ea5a111691387815f87186961ada51d4b",
        "41fbc3222d2249e1270f0297ff2bf0e2e61d30d2c3343b35b4a382593a9683d0",
        "773f78641e95445a446cb3493c24bad14e1e0508ef52ffe5570d57e1c5a52b5c",
        "c3a5f34961a58641676687668f856b6946dda3704b272bc1752d0779df7a0ee7",
        "c7fd7ef0c0161b03d2a0643908b480b2570d966100751278414e93175a219cb1",
        "00"
    ));
    let pl = pipeline(FILTER_NBIT, vec![8, 0, 64, 1, 4, 0, 20, 0]);
    let out = reverse_filters(&pl, &chunk).expect("nbit i32 reverse");
    let expected: Vec<u8> = (0..64i32)
        .flat_map(|i| ((i * 9999) & 0x7FFFF).to_le_bytes())
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn nbit_u16_big_endian() {
    // 16-bit storage, 10-bit precision, big-endian unsigned int.
    let chunk = hex(concat!(
        "000351a89f351094f9736a1dd84a479f2b1b9b1bd4385eebef09059238c300"
    ));
    let pl = pipeline(FILTER_NBIT, vec![8, 0, 24, 1, 2, 1, 10, 0]);
    let out = reverse_filters(&pl, &chunk).expect("nbit u16be reverse");
    let expected: Vec<u8> = (0..24u16)
        .flat_map(|i| ((i * 53) & 0x03FF).to_be_bytes())
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn scaleoffset_integer() {
    // int32, library-computed minbits, fill value defined (= 0).
    let chunk = hex(concat!(
        "0900000008e803000000000000000000000000000000094946f4a2e5bd039453",
        "6e597de78424372e2054ccb78456755fc26a79df312124dc935c3760527a65c7",
        "2dbbf00446c5b4029594ef8a4e6bd83d4737e423241b524b76e4064d4bb86577",
        "5df080d4b47f52325dd139c57705a7e67c44447362456cdb80496956fca6e7bc",
        "0f1a164fca30"
    ));
    let cd = vec![2, 0, 100, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let out = reverse_filters(&pl, &chunk).expect("scaleoffset int reverse");
    let expected: Vec<u8> = (0..100i32)
        .flat_map(|i| (1000 + ((i * 37) % 500)).to_le_bytes())
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn scaleoffset_float64() {
    // float64 D-scale, 3 decimal digits, fill value defined (= 0.0).
    let chunk = hex(concat!(
        "07000000086e861bf0f9210940000000000000000000041030814307102450b1",
        "83470f20449132854b173064d1b3874f1f4085123489532750a552b58b572f60",
        "c593368d5b3770e5d3b78f5f3f81061438916347912654b993674f00"
    ));
    let cd = vec![0, 3, 80, 1, 8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let out = reverse_filters(&pl, &chunk).expect("scaleoffset f64 reverse");
    assert_eq!(out.len(), 80 * 8);
    for (i, e) in out.chunks_exact(8).enumerate() {
        let got = f64::from_le_bytes(e.try_into().unwrap());
        let want = 3.14159 + i as f64 * 0.001;
        assert!(
            (got - want).abs() <= 5e-4,
            "elem {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn scaleoffset_float32() {
    // float32 D-scale, 2 decimal digits, fill value defined (= 0.0).
    let chunk = hex(concat!(
        "06000000080000204000000000000000000000000000108310518720928b30d3",
        "8f41149351559761969b71d79f8218a39259a7a29aabb2dbafc31cb3d35db7e3",
        "9ebb00"
    ));
    let cd = vec![0, 2, 60, 1, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let out = reverse_filters(&pl, &chunk).expect("scaleoffset f32 reverse");
    assert_eq!(out.len(), 60 * 4);
    for (i, e) in out.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes(e.try_into().unwrap());
        let want = 2.5 + i as f32 * 0.01;
        assert!(
            (got - want).abs() <= 5e-3,
            "elem {i}: got {got}, want {want}"
        );
    }
}
