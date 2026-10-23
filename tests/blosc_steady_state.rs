//! A blosc frame compressed or decoded again at the same size allocates
//! nothing: block temporaries come from per-thread scratch kept across
//! calls. Two codecs allocate inside their own crates on every stream and
//! are held to that count: lz4_flex's `compress_into` builds a hash table,
//! rust-zstd's `Compressor::compress` collects its job list.
//!
//! The pool runs one worker, so every block lands on a thread a warm-up
//! frame already served. rayon allocates a block of its injector queue once
//! every 63 jobs a thread outside the pool hands it, so of two measured
//! calls the one with fewer allocations is the check. The tests share the
//! counter and the pool, so they run one at a time.
#![cfg(feature = "blosc")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use rust_hdf5::format::messages::filter::{
    apply_filters_into, reverse_filters_masked_into, Filter, FilterPipeline, FILTER_BLOSC,
};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// This test's hold on the counter, with the pool of one worker.
fn serial() -> MutexGuard<'static, ()> {
    let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("RUST_HDF5_IO_THREADS", "1");
    serial
}

/// `f`'s result and the allocations every thread made during it.
fn allocs<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCS.load(Ordering::Relaxed);
    let r = f();
    (r, ALLOCS.load(Ordering::Relaxed) - before)
}

/// Little-endian u16 samples a bit of noise off a slow ramp: every codec
/// shrinks it enough that a wave's spilled blocks are placed, so each block
/// is compressed once.
fn image(nbytes: usize) -> Vec<u8> {
    let mut x = 0x9E37_79B9u32;
    (0..nbytes / 2)
        .flat_map(|i| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((((i / 8) as u32 + (x >> 31)) & 0x0FFF) as u16).to_le_bytes()
        })
        .collect()
}

#[test]
fn a_second_same_size_frame_allocates_nothing() {
    let _serial = serial();
    // (compressor code, name, allocations its crate makes per stream)
    let mut codecs = vec![
        (0, "blosclz", 0),
        (1, "lz4", 1),
        (2, "lz4hc", 1),
        (3, "snappy", 0),
    ];
    if cfg!(feature = "deflate") {
        codecs.push((4, "zlib", 0));
    }
    if cfg!(feature = "zstd") {
        codecs.push((5, "zstd", 2));
    }
    let mut failed = Vec::new();
    // Under the pool's thresholds, and past both.
    for nbytes in [256 * 1024, 1088 * 1024] {
        let data = image(nbytes);
        let mut out = vec![0u8; nbytes + 16];
        let mut back = vec![0u8; nbytes];
        for &(code, name, per_stream) in &codecs {
            for shuffle in 0..3u32 {
                let p = FilterPipeline {
                    filters: vec![Filter {
                        id: FILTER_BLOSC,
                        flags: 0,
                        cd_values: vec![2, 2, 2, nbytes as u32, 5, shuffle, code],
                    }],
                };
                let (mut c, mut d) = (usize::MAX, usize::MAX);
                for round in 0..3 {
                    let (written, a) = allocs(|| apply_filters_into(&p, &data, &mut out));
                    let n = written.unwrap();
                    let (read, b) =
                        allocs(|| reverse_filters_masked_into(&p, &out[..n], 0, &mut back));
                    assert_eq!(read.unwrap(), nbytes, "{name} shuffle {shuffle}");
                    assert!(back == data, "{name} shuffle {shuffle}: round trip differs");
                    if round > 0 {
                        c = c.min(a);
                        d = d.min(b);
                    }
                }
                // Neither codec splits a block, so a block is a stream.
                let word = |at: usize| u32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                let streams = (word(4) as usize).div_ceil(word(8) as usize);
                if c > per_stream * streams || d > 0 {
                    failed.push(format!(
                        "{name} shuffle {shuffle}, {nbytes} bytes: compress {c} allocations \
                         ({} allowed), decompress {d}",
                        per_stream * streams
                    ));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A zstd stream that ends part-way fails its frame and leaves the
/// thread's decoder inside a zstd frame; the next frame still decodes.
#[cfg(feature = "zstd")]
#[test]
fn a_zstd_frame_decodes_after_one_cut_short() {
    let _serial = serial();
    let nbytes = 16 * 1024;
    let data = image(nbytes);
    let p = FilterPipeline {
        filters: vec![Filter {
            id: FILTER_BLOSC,
            flags: 0,
            cd_values: vec![2, 2, 2, nbytes as u32, 5, 0, 5],
        }],
    };
    let mut out = vec![0u8; nbytes + 16];
    let n = apply_filters_into(&p, &data, &mut out).unwrap();
    let frame = &out[..n];
    // One block: its stream follows its bstarts entry and length.
    let csize = u32::from_le_bytes(frame[20..24].try_into().unwrap()) as usize;
    assert_eq!(frame.len(), 24 + csize);
    let mut cut = frame[..24 + csize / 2].to_vec();
    cut[12..16].copy_from_slice(&(24 + csize as u32 / 2).to_le_bytes());
    cut[20..24].copy_from_slice(&(csize as u32 / 2).to_le_bytes());
    let mut back = vec![0u8; nbytes];
    assert!(reverse_filters_masked_into(&p, &cut, 0, &mut back).is_err());
    assert_eq!(
        reverse_filters_masked_into(&p, frame, 0, &mut back).unwrap(),
        nbytes
    );
    assert!(back == data);
}

/// A thread keeps the buffers of a block up to the largest blosc_blocksize
/// makes, 1 MiB, and none of a larger block only a foreign frame names.
#[test]
fn a_block_past_the_largest_blosc_makes_is_not_kept() {
    let _serial = serial();
    let mut kept = Vec::new();
    for blocksize in [1024 * 1024, 1024 * 1024 + 2] {
        // One block, shuffled and stored as it is, in one split.
        let mut frame = vec![2u8, 1, 0x11, 2];
        for word in [blocksize, blocksize, 24 + blocksize, 20, blocksize] {
            frame.extend((word as u32).to_le_bytes());
        }
        frame.extend(image(blocksize));
        let p = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 2, blocksize as u32, 5, 1, 0],
            }],
        };
        let mut back = vec![0u8; blocksize];
        let mut least = usize::MAX;
        for round in 0..3 {
            let (read, n) = allocs(|| reverse_filters_masked_into(&p, &frame, 0, &mut back));
            assert_eq!(read.unwrap(), blocksize);
            if round > 0 {
                least = least.min(n);
            }
        }
        kept.push(least == 0);
    }
    assert_eq!(kept, [true, false]);
}

/// A thread keeps a zstd context of up to the workspace of a level-15
/// stream of 512 KiB, whatever its level, and builds one for each larger
/// stream: a frame compresses a second time with the allocations of a kept
/// context at clevel 8 and 512 KiB and at clevel 9 and 256 KiB, and with
/// those of a new one at clevel 9 and 256 KiB + 2.
#[cfg(feature = "zstd")]
#[test]
fn a_zstd_workspace_past_the_kept_bound_is_not_kept() {
    let _serial = serial();
    let mut kept = Vec::new();
    // (blosc clevel, its zstd level, frame and stream length, the workspace)
    for (clevel, level, nbytes, workspace) in [
        (8, 15, 512 * 1024, 8_900_472),
        (9, 22, 256 * 1024, 5_499_894),
        (9, 22, 256 * 1024 + 2, 9_694_198),
    ] {
        let data = image(nbytes);
        let mut out = vec![0u8; nbytes + 16];
        let mut z = rust_zstd::Compressor::new(rust_zstd::CompressOptions {
            level,
            ..Default::default()
        });
        z.compress(&data, &mut Vec::new());
        assert_eq!(z.workspace_sizes(), [workspace], "clevel {clevel}");
        let p = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 2, nbytes as u32, clevel, 1, 5],
            }],
        };
        let mut least = usize::MAX;
        for round in 0..3 {
            let (written, n) = allocs(|| apply_filters_into(&p, &data, &mut out));
            written.unwrap();
            if round > 0 {
                least = least.min(n);
            }
        }
        // The steady state's allowance for one zstd stream.
        kept.push(least <= 2);
    }
    assert_eq!(kept, [true, true, false]);
}
