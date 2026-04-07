#[test]
fn compression_ratios_by_level() {
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_64K", vec![0u8; 65536]),
        ("text_50K", b"The quick brown fox jumps over the lazy dog. ".repeat(1100)),
        ("seq_f64_32K", (0..4096u64).flat_map(|i| (i as f64 * 0.5).to_le_bytes()).collect()),
    ];

    for (name, data) in &patterns {
        eprintln!("\n{} ({} bytes):", name, data.len());
        for level in [0, 1, 3, 7, 11] {
            let compressed = hdf5_format::zstd::compress(data, level);
            let decompressed = hdf5_format::zstd::decompress(&compressed)
                .unwrap_or_else(|e| panic!("{} level {} decompress: {}", name, level, e));
            assert_eq!(decompressed.len(), data.len(), "{} level {}", name, level);
            assert_eq!(&decompressed, data, "{} level {}", name, level);
            let ratio = data.len() as f64 / compressed.len() as f64;
            eprintln!("  level {:2}: {:>8} -> {:>8}  ({:.2}x)", level, data.len(), compressed.len(), ratio);
        }
    }
}
