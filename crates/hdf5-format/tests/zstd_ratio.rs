#[test]
fn compression_ratios() {
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_64K", vec![0u8; 65536]),
        ("text_50K", b"The quick brown fox jumps over the lazy dog. ".repeat(1100)),
        ("seq_f64_32K", (0..4096u64).flat_map(|i| (i as f64 * 0.5).to_le_bytes()).collect()),
    ];

    for (name, data) in &patterns {
        let compressed = hdf5_format::zstd::compress(data, 1);
        let decompressed = hdf5_format::zstd::decompress(&compressed)
            .unwrap_or_else(|e| panic!("{}: decompress failed: {}", name, e));
        assert_eq!(decompressed.len(), data.len(), "{}: length mismatch", name);
        assert_eq!(&decompressed, data, "{}: data mismatch", name);
    }
}
