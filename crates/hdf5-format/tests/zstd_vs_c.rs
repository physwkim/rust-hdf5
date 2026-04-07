use std::time::Instant;
use std::process::Command;

fn bench_rust(data: &[u8], level: i32, iters: u32) -> (f64, f64, usize) {
    let compressed = hdf5_format::zstd::compress(data, level);
    let _ = hdf5_format::zstd::decompress(&compressed);

    let start = Instant::now();
    let mut c = Vec::new();
    for _ in 0..iters { c = hdf5_format::zstd::compress(data, level); }
    let ct = start.elapsed().as_secs_f64() / iters as f64;

    let start = Instant::now();
    for _ in 0..iters { let _ = hdf5_format::zstd::decompress(&c); }
    let dt = start.elapsed().as_secs_f64() / iters as f64;

    (ct, dt, c.len())
}

fn bench_c_zstd(data: &[u8], level: i32, iters: u32) -> (f64, f64, usize) {
    let tmp_in = "/tmp/zstd_bench_input.bin";
    let tmp_out = "/tmp/zstd_bench_output.zst";
    let tmp_dec = "/tmp/zstd_bench_decoded.bin";

    std::fs::write(tmp_in, data).unwrap();

    // Warmup
    Command::new("zstd").args(["-f", &format!("-{}", level), tmp_in, "-o", tmp_out])
        .output().unwrap();

    // Compress
    let start = Instant::now();
    for _ in 0..iters {
        Command::new("zstd").args(["-f", "-q", &format!("-{}", level), tmp_in, "-o", tmp_out])
            .output().unwrap();
    }
    let ct = start.elapsed().as_secs_f64() / iters as f64;
    let comp_size = std::fs::metadata(tmp_out).unwrap().len() as usize;

    // Decompress
    let start = Instant::now();
    for _ in 0..iters {
        Command::new("zstd").args(["-d", "-f", "-q", tmp_out, "-o", tmp_dec])
            .output().unwrap();
    }
    let dt = start.elapsed().as_secs_f64() / iters as f64;

    std::fs::remove_file(tmp_in).ok();
    std::fs::remove_file(tmp_out).ok();
    std::fs::remove_file(tmp_dec).ok();

    (ct, dt, comp_size)
}

#[test]
fn rust_vs_c_zstd() {
    // Check zstd is available
    if Command::new("zstd").arg("--version").output().is_err() {
        eprintln!("skipping: zstd CLI not found");
        return;
    }

    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_1M", vec![0u8; 1_048_576]),
        ("text_1M", b"The quick brown fox jumps over the lazy dog. Hello world! ".repeat(18000)),
        ("f64_seq_1M", (0..131072u64).flat_map(|i| (i as f64).to_le_bytes()).collect()),
        ("mixed_1M", (0..262144u32).flat_map(|i| {
            if i % 4 == 0 { [0u8; 4] } else { i.to_le_bytes() }
        }).collect()),
    ];

    let iters = 5;

    eprintln!("\n{:=<110}", "");
    eprintln!("{:<12} {:>5} │ {:>8} {:>7} {:>7} │ {:>8} {:>7} {:>7} │ {:>6} {:>6}",
        "Dataset", "Level",
        "C_Size", "C_Comp", "C_Dec",
        "Rs_Size", "Rs_Comp", "Rs_Dec",
        "Ratio", "Speed");
    eprintln!("{:=<110}", "");

    for (name, data) in &datasets {
        let mb = data.len() as f64 / (1024.0 * 1024.0);

        for level in [1, 3, 7, 11] {
            let (c_ct, c_dt, c_sz) = bench_c_zstd(data, level, iters);
            let (r_ct, r_dt, r_sz) = bench_rust(data, level, iters);

            let c_comp = mb / c_ct;
            let c_dec = mb / c_dt;
            let r_comp = mb / r_ct;
            let r_dec = mb / r_dt;

            // Compression ratio comparison (Rust size / C size)
            let size_ratio = r_sz as f64 / c_sz as f64;
            // Speed comparison (Rust speed / C speed for compress)
            let speed_ratio = r_comp / c_comp;

            eprintln!("{:<12} {:>5} │ {:>8} {:>6.0}M {:>6.0}M │ {:>8} {:>6.0}M {:>6.0}M │ {:>5.2}x {:>5.1}%",
                name, level,
                c_sz, c_comp, c_dec,
                r_sz, r_comp, r_dec,
                size_ratio, speed_ratio * 100.0);
        }
        eprintln!("{:-<110}", "");
    }
}
