use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hdf5::H5File;

fn bench_contiguous_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_write");
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            b.iter(|| {
                let path = std::env::temp_dir().join("bench_write.h5");
                let file = H5File::create(&path).unwrap();
                let ds = file.new_dataset::<f64>().shape([n]).create("data").unwrap();
                ds.write_raw(&data).unwrap();
                file.close().unwrap();
                std::fs::remove_file(&path).ok();
            });
        });
    }
    group.finish();
}

fn bench_contiguous_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_read");
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        // Prepare file
        let path = std::env::temp_dir().join(format!("bench_read_{}.h5", n));
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f64>().shape([n]).create("data").unwrap();
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("data").unwrap();
                let _data = ds.read_raw::<f64>().unwrap();
            });
        });

        std::fs::remove_file(&path).ok();
    }
    group.finish();
}

fn bench_chunked_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_write");
    for &nframes in &[10, 100, 1000] {
        let ncols = 100;
        let total_bytes = nframes * ncols * 8;
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nframes), &nframes, |b, &nframes| {
            b.iter(|| {
                let path = std::env::temp_dir().join("bench_chunked.h5");
                let file = H5File::create(&path).unwrap();
                let ds = file.new_dataset::<f64>()
                    .shape([0, ncols])
                    .chunk(&[1, ncols])
                    .max_shape(&[None, Some(ncols)])
                    .create("stream")
                    .unwrap();
                for frame in 0..nframes {
                    let vals: Vec<f64> = (0..ncols).map(|i| (frame * ncols + i) as f64).collect();
                    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                    ds.write_chunk(frame, &raw).unwrap();
                }
                ds.extend(&[nframes, ncols]).unwrap();
                file.close().unwrap();
                std::fs::remove_file(&path).ok();
            });
        });
    }
    group.finish();
}

fn bench_compressed_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("compressed_write");
    for &nframes in &[10, 100] {
        let ncols = 100;
        let total_bytes = nframes * ncols * 8;
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nframes), &nframes, |b, &nframes| {
            b.iter(|| {
                let path = std::env::temp_dir().join("bench_compressed.h5");
                let file = H5File::create(&path).unwrap();
                let ds = file.new_dataset::<f64>()
                    .shape([0, ncols])
                    .chunk(&[1, ncols])
                    .max_shape(&[None, Some(ncols)])
                    .deflate(6)
                    .create("stream")
                    .unwrap();
                for frame in 0..nframes {
                    let vals: Vec<f64> = (0..ncols).map(|i| (frame * ncols + i) as f64).collect();
                    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                    ds.write_chunk(frame, &raw).unwrap();
                }
                ds.extend(&[nframes, ncols]).unwrap();
                file.close().unwrap();
                std::fs::remove_file(&path).ok();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_contiguous_write,
    bench_contiguous_read,
    bench_chunked_write,
    bench_compressed_write,
);
criterion_main!(benches);
