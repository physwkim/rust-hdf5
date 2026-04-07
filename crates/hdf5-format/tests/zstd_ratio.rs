#[cfg(feature = "zstandard")]
#[test]
fn compression_ratios_by_level() {
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_64K", vec![0u8; 65536]),
        ("text_50K", b"The quick brown fox jumps over the lazy dog. ".repeat(1100)),
        ("seq_f64_32K", (0..4096u64).flat_map(|i| (i as f64 * 0.5).to_le_bytes()).collect()),
        ("mixed_128K", (0..32768u32).flat_map(|i| {
            if i % 100 < 50 { [0u8; 4] } else { i.to_le_bytes() }
        }).collect()),
    ];

    for (name, data) in &patterns {
        eprintln!("\n{} ({} bytes):", name, data.len());
        for level in [1, 3, 7, 11] {
            let compressed = hdf5_format::zstd::compress(data, level);
            let ratio = data.len() as f64 / compressed.len() as f64;
            eprintln!("  level {:2}: {:>8} -> {:>8}  ({:.2}x, {:.1}%)",
                level, data.len(), compressed.len(), ratio,
                (1.0 - compressed.len() as f64 / data.len() as f64) * 100.0);

            // Verify roundtrip
            use std::io::Read;
            let mut dec = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice()).unwrap();
            let mut out = Vec::new();
            dec.read_to_end(&mut out).unwrap();
            assert_eq!(out.len(), data.len(), "{} level {} length mismatch", name, level);
            assert_eq!(&out, data, "{} level {} data mismatch", name, level);
        }
    }
}
