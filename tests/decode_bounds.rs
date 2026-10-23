//! Every reverse filter decodes under the chunk image's size: a small stored
//! chunk that claims, or expands to, far more than the image fails before
//! it allocates more than about the image, and an image of exactly the
//! bound still decodes.
//!
//! The bombs are 64 MiB against a 64 KiB image. The peak the allocator sees
//! across all threads during the decode is the check; the tests serialise on
//! one lock so no other test's allocations count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rust_hdf5::format::messages::filter::{
    apply_filters, reverse_filters_masked_bounded, reverse_filters_masked_into, Filter,
    FilterPipeline, FILTER_BLOSC, FILTER_BSHUF, FILTER_BZIP2, FILTER_DEFLATE, FILTER_FLETCHER32,
    FILTER_LZ4, FILTER_LZF, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP,
    FILTER_ZSTD,
};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, layout, new_size) };
        if !q.is_null() {
            if new_size > layout.size() {
                grew(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// `f`'s result and the most bytes live during it beyond those live before.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let r = f();
    (r, PEAK.load(Ordering::Relaxed).saturating_sub(base))
}

const IMAGE: usize = 64 * 1024;
const BOMB: usize = 64 * 1024 * 1024;
/// Several times the image, a sixteenth of the bomb: room for a decoder's
/// own state (bzip2 level 1 keeps 400 KB) and nothing the bomb sizes.
const PEAK_CAP: usize = 4 * 1024 * 1024;

fn pipeline(filters: &[(u16, &[u32])]) -> FilterPipeline {
    FilterPipeline {
        filters: filters
            .iter()
            .map(|(id, cd)| Filter {
                id: *id,
                flags: 0,
                cd_values: cd.to_vec(),
            })
            .collect(),
    }
}

/// Both reverse paths refuse `stored` for a 64 KiB image under the cap.
fn assert_refused(what: &str, p: &FilterPipeline, stored: &[u8]) {
    let (vec, vec_peak) = peak_during(|| reverse_filters_masked_bounded(p, stored, 0, IMAGE));
    let mut out = vec![0u8; IMAGE];
    let (into, into_peak) = peak_during(|| reverse_filters_masked_into(p, stored, 0, &mut out));
    assert!(
        vec.is_err() && vec_peak < PEAK_CAP && into.is_err() && into_peak < PEAK_CAP,
        "{what}: bounded decode gave {:?} at a peak of {vec_peak} bytes, \
         decode into the image {into:?} at {into_peak}",
        vec.map(|v| v.len()),
    );
}

/// A zstd frame of RLE blocks, 128 KiB of zeros per 4 bytes stored, that
/// declares no content size: `n` bytes out of about `n / 32768` in.
#[cfg(feature = "zstd")]
fn zstd_rle_bomb(n: usize) -> Vec<u8> {
    const BLOCK: usize = 128 * 1024;
    // Magic, a descriptor with no content size, a 128 KiB window.
    let mut z = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x38];
    let blocks = n / BLOCK;
    for b in 0..blocks {
        let header = (BLOCK << 3) | (1 << 1) | usize::from(b + 1 == blocks);
        z.extend_from_slice(&header.to_le_bytes()[..3]);
        z.push(0);
    }
    z
}

#[test]
#[cfg(feature = "zstd")]
fn zstd_bomb_is_refused() {
    let stored = zstd_rle_bomb(BOMB);
    assert!(stored.len() < 4096);
    assert_refused("zstd", &pipeline(&[(FILTER_ZSTD, &[])]), &stored);
}

#[test]
#[cfg(all(feature = "blosc", feature = "zstd"))]
fn blosc_zstd_stream_bomb_is_refused() {
    // One 64 KiB block, unsplit, whose zstd stream expands to the bomb.
    let z = zstd_rle_bomb(BOMB);
    let mut f = vec![2, 1, 0x10 | (4 << 5), 1];
    for word in [IMAGE, IMAGE, 20 + 4 + z.len(), 20] {
        f.extend_from_slice(&(word as u32).to_le_bytes());
    }
    f.extend_from_slice(&(z.len() as u32).to_le_bytes());
    f.extend_from_slice(&z);
    assert_refused("blosc zstd", &pipeline(&[(FILTER_BLOSC, &[])]), &f);
}

#[test]
#[cfg(feature = "blosc")]
fn blosc_nbytes_bomb_is_refused() {
    // A shuffled blosclz frame whose header claims one 64 MiB block.
    let mut f = vec![2, 1, 0x01, 4];
    for word in [BOMB, BOMB, 28, 20, 4] {
        f.extend_from_slice(&(word as u32).to_le_bytes());
    }
    f.extend_from_slice(&[0; 4]);
    assert_refused("blosc nbytes", &pipeline(&[(FILTER_BLOSC, &[])]), &f);
}

#[test]
#[cfg(feature = "bzip2")]
fn bzip2_bomb_is_refused() {
    let p = pipeline(&[(FILTER_BZIP2, &[1])]);
    let stored = apply_filters(&p, &vec![0u8; BOMB]).unwrap();
    assert!(stored.len() < IMAGE);
    assert_refused("bzip2", &p, &stored);
}

#[test]
#[cfg(feature = "deflate")]
fn deflate_bomb_is_refused() {
    let stored = apply_filters(&pipeline(&[(FILTER_DEFLATE, &[9])]), &vec![0u8; BOMB]).unwrap();
    assert_refused("deflate", &pipeline(&[(FILTER_DEFLATE, &[])]), &stored);
    // The stage above a shuffle is bounded by what the shuffle reverses to.
    assert_refused(
        "shuffle+deflate",
        &pipeline(&[(FILTER_SHUFFLE, &[2]), (FILTER_DEFLATE, &[])]),
        &stored,
    );
}

/// A chunk a write or a shrink only partly replaces is read back first, and
/// that read decodes under the chunk's image too.
#[test]
#[cfg(feature = "deflate")]
fn read_modify_write_of_a_bomb_chunk_is_refused() {
    let stored = apply_filters(&pipeline(&[(FILTER_DEFLATE, &[9])]), &vec![0u8; BOMB]).unwrap();
    let path = std::env::temp_dir().join(format!("rust_hdf5_rmw_bomb_{}.h5", std::process::id()));
    let file = rust_hdf5::H5File::create(&path).unwrap();
    let dataset = |name| {
        let ds = file
            .new_dataset::<u8>()
            .shape([2 * IMAGE])
            .chunk(&[IMAGE])
            .max_shape(&[None])
            .deflate(6)
            .create(name)
            .unwrap();
        ds.write_chunk_raw(0, &stored, 0).unwrap();
        ds
    };
    let (written, write_peak) = peak_during(|| dataset("w").write_slice(&[0], &[16], &[1u8; 16]));
    let (shrunk, shrink_peak) = peak_during(|| dataset("s").set_extent(&[IMAGE / 2]));
    drop(file);
    std::fs::remove_file(&path).ok();
    assert!(
        written.is_err() && write_peak < PEAK_CAP && shrunk.is_err() && shrink_peak < PEAK_CAP,
        "write_slice gave {written:?} at a peak of {write_peak} bytes, \
         set_extent {shrunk:?} at {shrink_peak}"
    );
}

#[test]
#[cfg(feature = "lz4")]
fn lz4_header_bomb_is_refused() {
    // orig_size and block_size 64 MiB, one stored 1-byte block.
    let mut stored = (BOMB as u64).to_be_bytes().to_vec();
    stored.extend_from_slice(&(BOMB as u32).to_be_bytes());
    stored.extend_from_slice(&1u32.to_be_bytes());
    stored.push(0);
    assert_refused("lz4", &pipeline(&[(FILTER_LZ4, &[])]), &stored);
}

#[test]
fn szip_header_bomb_is_refused() {
    let mut stored = (BOMB as u32).to_le_bytes().to_vec();
    stored.extend_from_slice(&[0; 16]);
    assert_refused(
        "szip",
        &pipeline(&[(FILTER_SZIP, &[0, 32, 8, 256])]),
        &stored,
    );
}

#[test]
fn nbit_parameter_bomb_is_refused() {
    // 16 Mi 4-byte elements of 8 bits each, from 16 bytes.
    let cd = [8, 0, (BOMB / 4) as u32, 1, 4, 0, 8, 0];
    assert_refused("nbit", &pipeline(&[(FILTER_NBIT, &cd)]), &[0; 16]);
}

#[test]
fn scaleoffset_parameter_bomb_is_refused() {
    // Integer, minbits 0: every one of 16 Mi elements is the minimum, and
    // the 21-byte header is all the chunk stores.
    let mut cd = [0u32; 20];
    cd[0] = 2;
    cd[2] = (BOMB / 4) as u32;
    cd[4] = 4;
    let mut stored = vec![0u8; 21];
    stored[4] = 4;
    assert_refused(
        "scaleoffset",
        &pipeline(&[(FILTER_SCALEOFFSET, &cd)]),
        &stored,
    );
}

#[test]
#[cfg(feature = "lz4")]
fn bshuf_block_bomb_is_refused() {
    // A 64 KiB chunk whose header claims 64 MiB blocks.
    let mut stored = (IMAGE as u64).to_be_bytes().to_vec();
    stored.extend_from_slice(&(BOMB as u32).to_be_bytes());
    assert_refused(
        "bshuf",
        &pipeline(&[(FILTER_BSHUF, &[0, 0, 1, 0, 2])]),
        &stored,
    );
}

#[test]
fn lzf_bomb_is_refused() {
    // One literal, then 264-byte back-references of 3 bytes each.
    let mut stored = vec![0x00, 0x00];
    for _ in 0..BOMB / 264 + 1 {
        stored.extend_from_slice(&[0xE0, 0xFF, 0x00]);
    }
    assert_refused("lzf", &pipeline(&[(FILTER_LZF, &[4, 0, 0])]), &stored);
}

/// The image every boundary case filters: 4-byte integers under 2^16.
fn image() -> Vec<u8> {
    (0..1024u32)
        .flat_map(|i| ((i * 7) % 3000).to_le_bytes())
        .collect()
}

/// Pipelines that each reverse to `image()`, with the mask of the filters
/// the write skipped, for the codecs this build has.
fn boundary_pipelines() -> Vec<(&'static str, FilterPipeline, u32)> {
    let n = image().len() as u32;
    let mut so = vec![0u32; 20];
    so[0] = 2;
    so[2] = n / 4;
    so[4] = 4;
    so[7] = 1;
    let shuffle_deflate = || pipeline(&[(FILTER_SHUFFLE, &[4]), (FILTER_DEFLATE, &[6])]);
    let blosc = |compcode| pipeline(&[(FILTER_BLOSC, &[2, 2, 4, n, 5, 1, compcode])]);
    let deflate = cfg!(feature = "deflate");
    let blosc_on = cfg!(feature = "blosc");
    [
        (
            "szip",
            true,
            pipeline(&[(FILTER_SZIP, &[4, 32, 32, 16])]),
            0,
        ),
        (
            "nbit",
            true,
            pipeline(&[(FILTER_NBIT, &[8, 0, n / 4, 1, 4, 0, 16, 0])]),
            0,
        ),
        (
            "scaleoffset",
            true,
            pipeline(&[(FILTER_SCALEOFFSET, &so)]),
            0,
        ),
        ("lzf", true, pipeline(&[(FILTER_LZF, &[4, 0, n])]), 0),
        (
            "bshuf",
            true,
            pipeline(&[(FILTER_BSHUF, &[0, 0, 4, 0, 0])]),
            0,
        ),
        ("deflate", deflate, pipeline(&[(FILTER_DEFLATE, &[6])]), 0),
        ("shuffle+deflate", deflate, shuffle_deflate(), 0),
        (
            "shuffle+deflate+fletcher32",
            deflate,
            pipeline(&[
                (FILTER_SHUFFLE, &[4]),
                (FILTER_DEFLATE, &[6]),
                (FILTER_FLETCHER32, &[]),
            ]),
            0,
        ),
        // The shuffle was skipped on write: the deflate decodes the image.
        ("masked shuffle+deflate", deflate, shuffle_deflate(), 1),
        // Both were: the stored bytes are the image.
        ("all-masked shuffle+deflate", true, shuffle_deflate(), 3),
        (
            "zstd",
            cfg!(feature = "zstd"),
            pipeline(&[(FILTER_ZSTD, &[3])]),
            0,
        ),
        (
            "bzip2",
            cfg!(feature = "bzip2"),
            pipeline(&[(FILTER_BZIP2, &[9])]),
            0,
        ),
        (
            "lz4",
            cfg!(feature = "lz4"),
            pipeline(&[(FILTER_LZ4, &[1024])]),
            0,
        ),
        (
            "bshuf lz4",
            cfg!(feature = "lz4"),
            pipeline(&[(FILTER_BSHUF, &[0, 0, 4, 0, 2])]),
            0,
        ),
        ("blosc blosclz", blosc_on, blosc(0), 0),
        ("blosc lz4", blosc_on, blosc(1), 0),
        ("blosc zlib", blosc_on && deflate, blosc(4), 0),
        (
            "blosc zstd",
            blosc_on && cfg!(feature = "zstd"),
            blosc(5),
            0,
        ),
    ]
    .into_iter()
    .filter(|case| case.1)
    .map(|(what, _, p, mask)| (what, p, mask))
    .collect()
}

/// Forward `image()` through `p`, skipping the filters `mask` names.
fn stored(p: &FilterPipeline, mask: u32) -> Vec<u8> {
    let applied = FilterPipeline {
        filters: p
            .filters
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) == 0)
            .map(|(_, f)| f.clone())
            .collect(),
    };
    apply_filters(&applied, &image()).unwrap()
}

#[test]
fn image_of_exactly_the_bound_decodes() {
    let want = image();
    for (what, p, mask) in boundary_pipelines() {
        let s = stored(&p, mask);
        let got = reverse_filters_masked_bounded(&p, &s, mask, want.len());
        assert!(
            got.as_ref().is_ok_and(|v| *v == want),
            "{what}: {:?}",
            got.map(|v| v.len())
        );
        let mut out = vec![0u8; want.len()];
        let n = reverse_filters_masked_into(&p, &s, mask, &mut out);
        assert_eq!(n.ok(), Some(want.len()), "{what}");
        assert_eq!(out, want, "{what}");
    }
}

#[test]
fn image_one_byte_past_the_bound_fails() {
    let n = image().len();
    let mut decoded = Vec::new();
    for (what, p, mask) in boundary_pipelines() {
        let s = stored(&p, mask);
        if let Ok(v) = reverse_filters_masked_bounded(&p, &s, mask, n - 1) {
            decoded.push(format!("{what}: bounded decode gave {} bytes", v.len()));
        }
        let mut out = vec![0u8; n - 1];
        if let Ok(len) = reverse_filters_masked_into(&p, &s, mask, &mut out) {
            decoded.push(format!("{what}: decode into the image gave {len}"));
        }
    }
    assert!(decoded.is_empty(), "{decoded:#?}");
}
