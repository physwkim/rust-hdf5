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
    assert!(clean.len().is_multiple_of(2), "hex string has odd length");
    (0..clean.len() / 2)
        .map(|i| u8::from_str_radix(&clean[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

const FILTER_SZIP: u16 = 4;
const FILTER_NBIT: u16 = 5;
const FILTER_SCALEOFFSET: u16 = 6;

/// SZIP cross-decode against libhdf5's own reference file: the crate's AEC
/// codec is byte-compatible with libaec/libhdf5, so it decodes a real
/// libhdf5-written SZIP chunk exactly.
#[test]
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
    let chunk = hex("000351a89f351094f9736a1dd84a479f2b1b9b1bd4385eebef09059238c300");
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
    let expected: Vec<u8> = (0..100i32)
        .flat_map(|i| (1000 + ((i * 37) % 500)).to_le_bytes())
        .collect();
    let out = reverse_filters(&pl, &chunk).expect("scaleoffset int reverse");
    assert_eq!(out, expected);
    // Compressing the same values must reproduce libhdf5's own chunk byte for
    // byte: the 21-byte parameter header (minbits 9, minval 1000) and the
    // 9-bit packing that follows it.
    let back = apply_filters(&pl, &expected).expect("scaleoffset int compress");
    assert_eq!(back, chunk);
}

/// Signed values that straddle zero, so `minval` is stored as a negative
/// `long long` and the offsets are taken against it.
#[test]
fn scaleoffset_integer_negative_minimum() {
    let cd = vec![2, 0, 64, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    // -1000 .. 1000 in steps of ~31, so the span is 2001 -> minbits 11.
    let data: Vec<u8> = (0..64i32)
        .flat_map(|i| (-1000 + i * 31).to_le_bytes())
        .collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    assert_eq!(
        u32::from_le_bytes(chunk[..4].try_into().unwrap()),
        11,
        "1953 distinct offsets plus a fill sentinel need 11 bits"
    );
    assert_eq!(
        i64::from_le_bytes(chunk[5..13].try_into().unwrap()),
        -1000,
        "minval is the sign-extended chunk minimum"
    );
    assert_eq!(chunk.len(), 21 + 64 * 4 * 11 / 32 + 1);
    assert_eq!(reverse_filters(&pl, &chunk).unwrap(), data);
}

/// An element equal to the declared fill value is replaced by the all-ones
/// sentinel rather than by an offset, and comes back as the fill value.
#[test]
fn scaleoffset_integer_fill_value_sentinel() {
    // uint16, fill value 65535 (cd_values[8], LSB-first).
    let cd = vec![
        2, 0, 8, 0, 2, 0, 0, 1, 65535, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let values: [u16; 8] = [40, 65535, 42, 44, 65535, 41, 47, 40];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    // Range 40..47 -> span 8, +1 for the sentinel -> log2(9) = 4 bits.
    assert_eq!(u32::from_le_bytes(chunk[..4].try_into().unwrap()), 4);
    assert_eq!(u64::from_le_bytes(chunk[5..13].try_into().unwrap()), 40);
    assert_eq!(reverse_filters(&pl, &chunk).unwrap(), data);
}

/// A chunk whose elements are all equal, with no fill value, compresses to
/// `minbits == 0`: nothing but the header and the one trailing byte the size
/// formula always leaves.
#[test]
fn scaleoffset_integer_constant_chunk_stores_no_bits() {
    let cd = vec![2, 0, 16, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let data: Vec<u8> = std::iter::repeat_n(777i32, 16)
        .flat_map(i32::to_le_bytes)
        .collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    assert_eq!(u32::from_le_bytes(chunk[..4].try_into().unwrap()), 0);
    assert_eq!(u64::from_le_bytes(chunk[5..13].try_into().unwrap()), 777);
    assert_eq!(chunk.len(), 21 + 1);
    assert_eq!(reverse_filters(&pl, &chunk).unwrap(), data);
}

/// A span too wide for a sentinel to fit above it falls back to full
/// precision: the values are stored raw behind the header, and `minval` stays
/// zero because the library gives up before recording it.
#[test]
fn scaleoffset_integer_full_span_stores_raw() {
    let cd = vec![2, 0, 4, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    // uint8 with 0 as the fill value: 1..255 spans 254 > 255 - 2 - 1.
    let data = vec![1u8, 255, 128, 254];
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    assert_eq!(u32::from_le_bytes(chunk[..4].try_into().unwrap()), 8);
    assert_eq!(u64::from_le_bytes(chunk[5..13].try_into().unwrap()), 0);
    assert_eq!(chunk.len(), 21 + 4, "no trailing spare byte in this case");
    assert_eq!(&chunk[21..], &data[..]);
    assert_eq!(reverse_filters(&pl, &chunk).unwrap(), data);
}

/// A user-set minimum bit count equal to full precision makes the filter a
/// no-op in the forward direction too: no header is written.
#[test]
fn scaleoffset_integer_full_precision_compress_is_a_noop() {
    // The three values in the tail are the ASCII of "scaleoffset\0": libhdf5
    // reports them in place of the trailing zeros the file actually stores.
    // Reproduced verbatim from the capture so the vector stays comparable; the
    // filter reads nothing past index 8 for a 4-byte type.
    let cd = vec![
        2, 32, 8, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1818321779, 1717989221, 7628147, 0,
    ];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let data: Vec<u8> = (1000..1008i32).flat_map(i32::to_le_bytes).collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    assert_eq!(chunk, data);
}

/// A user-set minimum bit count below full precision is used as-is: the
/// filter only computes the chunk minimum, it does not shrink the count.
#[test]
fn scaleoffset_integer_honours_a_user_set_minbits() {
    let cd = vec![2, 20, 32, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let data: Vec<u8> = (0..32i32)
        .flat_map(|i| (5000 + i * 3).to_le_bytes())
        .collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    assert_eq!(
        u32::from_le_bytes(chunk[..4].try_into().unwrap()),
        20,
        "the requested 20 bits are kept even though 7 would do"
    );
    assert_eq!(chunk.len(), 21 + 32 * 4 * 20 / 32 + 1);
    assert_eq!(reverse_filters(&pl, &chunk).unwrap(), data);
}

/// Big-endian elements pack from the opposite end of each value.
#[test]
fn scaleoffset_integer_big_endian() {
    let cd = vec![2, 0, 24, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let data: Vec<u8> = (0..24i16)
        .flat_map(|i| (-300 + i * 7).to_be_bytes())
        .collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset compress");
    assert_eq!(reverse_filters(&pl, &chunk).unwrap(), data);
}

/// A user-set minimum-bit count equal to the datatype's full precision
/// (h5py's `scaleoffset=32` on an `i4`) makes the filter a no-op:
/// `H5Z__filter_scaleoffset` returns before the forward/reverse split, so the
/// stored chunk is the raw element buffer with no parameter header. The
/// capture is libhdf5 1.14.6's own output for `np.arange(1000, 1008)`.
#[test]
fn scaleoffset_integer_full_precision_has_no_header() {
    let chunk = hex("e8030000e9030000ea030000eb030000ec030000ed030000ee030000ef030000");
    assert_eq!(chunk.len(), 8 * 4, "no 21-byte header is stored");
    let cd = vec![
        2, 32, 8, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1818321779, 1717989221, 7628147, 0,
    ];
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let out = reverse_filters(&pl, &chunk).expect("scaleoffset full-precision reverse");
    let expected: Vec<u8> = (1000..1008i32).flat_map(i32::to_le_bytes).collect();
    assert_eq!(out, expected);
}

// 3.14159 is the literal libhdf5 was fed when this reference chunk was
// captured, not a stand-in for PI: swapping in `std::f64::consts::PI` changes
// the scale-offset filter's stored minval bytes (verified — the `back ==
// chunk` byte comparison below fails with PI, since the filter's minval
// encoding is sensitive to the exact double, not just its rounded display),
// so the value is load-bearing and the digits must stay exactly as spelled.
#[test]
#[allow(clippy::approx_constant)]
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
    for (i, e) in out.as_chunks::<8>().0.iter().enumerate() {
        let got = f64::from_le_bytes(*e);
        let want = 3.14159 + i as f64 * 0.001;
        assert!(
            (got - want).abs() <= 5e-4,
            "elem {i}: got {got}, want {want}"
        );
    }
    // The original values are what libhdf5 was handed; D-scaling them again
    // has to land on the same chunk it wrote.
    let source: Vec<u8> = (0..80)
        .flat_map(|i| (3.14159f64 + i as f64 * 0.001).to_le_bytes())
        .collect();
    let back = apply_filters(&pl, &source).expect("scaleoffset f64 compress");
    assert_eq!(back, chunk);
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
    for (i, e) in out.as_chunks::<4>().0.iter().enumerate() {
        let got = f32::from_le_bytes(*e);
        let want = 2.5 + i as f32 * 0.01;
        assert!(
            (got - want).abs() <= 5e-3,
            "elem {i}: got {got}, want {want}"
        );
    }
    // Same values back through the filter must reproduce libhdf5's chunk. The
    // scaling here runs in `float`, as `H5Z_scaleoffset_precompress_3` does
    // for a 4-byte element; doing it in `double` shifts several roundings.
    let source: Vec<u8> = (0..60)
        .flat_map(|i| (2.5f32 + i as f32 * 0.01).to_le_bytes())
        .collect();
    let back = apply_filters(&pl, &source).expect("scaleoffset f32 compress");
    assert_eq!(back, chunk);
}

/// Floats equal to the declared fill value survive the scaling as the fill
/// value, not as a scaled offset.
#[test]
fn scaleoffset_float_fill_value_sentinel() {
    // float64, 2 decimal digits, fill value -9999.0 packed into cd_values[8..].
    let filval = (-9999.0f64).to_bits();
    let mut cd = vec![
        0u32, 2, 6, 1, 8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    cd[8] = filval as u32;
    cd[9] = (filval >> 32) as u32;
    let pl = pipeline(FILTER_SCALEOFFSET, cd);
    let values = [1.25f64, -9999.0, 1.5, 2.0, -9999.0, 1.75];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let chunk = apply_filters(&pl, &data).expect("scaleoffset f64 compress");
    let out = reverse_filters(&pl, &chunk).expect("scaleoffset f64 reverse");
    for (i, e) in out.as_chunks::<8>().0.iter().enumerate() {
        let got = f64::from_le_bytes(*e);
        assert!(
            (got - values[i]).abs() <= 5e-3,
            "elem {i}: got {got}, want {}",
            values[i]
        );
    }
}
