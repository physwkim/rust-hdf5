use std::time::Instant;

fn bench_level(_name: &str, data: &[u8], level: i32, iterations: u32) -> (f64, f64, usize) {
    // Warmup
    let compressed = hdf5_format::zstd::compress(data, level);
    let _ = hdf5_format::zstd::decompress(&compressed);

    // Compress benchmark
    let start = Instant::now();
    let mut comp_out = Vec::new();
    for _ in 0..iterations {
        comp_out = hdf5_format::zstd::compress(data, level);
    }
    let comp_time = start.elapsed().as_secs_f64() / iterations as f64;

    // Decompress benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hdf5_format::zstd::decompress(&comp_out);
    }
    let decomp_time = start.elapsed().as_secs_f64() / iterations as f64;

    (comp_time, decomp_time, comp_out.len())
}

#[test]
fn zstd_speed_comparison() {
    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_1M", vec![0u8; 1_048_576]),
        ("text_1M", b"The quick brown fox jumps over the lazy dog. Hello world! ".repeat(18000)),
        ("f64_seq_1M", (0..131072u64).flat_map(|i| (i as f64).to_le_bytes()).collect()),
        ("mixed_1M", (0..262144u32).flat_map(|i| {
            if i % 4 == 0 { [0u8; 4] } else { i.to_le_bytes() }
        }).collect()),
    ];

    eprintln!("\n{:=<95}", "");
    eprintln!("{:<15} {:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
        "Dataset", "Level", "CompSize", "Ratio", "CompMB/s", "DecompSz", "DecTime", "DecMB/s");
    eprintln!("{:-<95}", "");

    for (name, data) in &datasets {
        let mb = data.len() as f64 / (1024.0 * 1024.0);
        let iters = if data.len() > 500_000 { 20 } else { 100 };

        for level in [0, 1, 3, 7, 11] {
            let (ct, dt, cs) = bench_level(name, data, level, iters);
            let ratio = data.len() as f64 / cs as f64;
            let comp_mbps = mb / ct;
            let dec_mbps = mb / dt;
            eprintln!("{:<15} {:>6} {:>10} {:>9.2}x {:>7.0} {:>10} {:>10.1}µs {:>7.0}",
                name, level, cs, ratio, comp_mbps, data.len(), dt * 1e6, dec_mbps);
        }
        eprintln!("{:-<95}", "");
    }
}
