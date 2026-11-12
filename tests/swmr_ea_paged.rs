//! A SWMR stream past 131,060 frames. From there on the extensible array's
//! data blocks hold more entries than one 1024-entry page, so the writer
//! pages them the way H5EA does, and a live reader follows the pages to the
//! newest frame. h5py reads the same stream live and after close, and
//! libhdf5's own array for the same shape carries the same statistics.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};
use rust_hdf5::{FileLocking, FilterPipeline, H5File};

/// The first chunk index in a paged data block under the default parameters:
/// the index block's 4 entries and super blocks 0 to 12's 131,056.
const FIRST_PAGED: u64 = 131_060;
/// The first chunk index in super block 14, whose data blocks are 4 pages.
const SBLK_14: u64 = 262_132;
/// Frames streamed: into super block 14, past its first page.
const N: u64 = SBLK_14 + 1100;

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`.
const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn python() -> Option<&'static str> {
    static PY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PY.get_or_init(|| {
        let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
            Ok(p) => vec![p],
            Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
        };
        let found = candidates.iter().find(|c| Path::new(c).exists()).cloned();
        if found.is_none() {
            eprintln!("skipping the h5py checks: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
}

/// Reads frame `argv[2]` of `argv[1]` through a live SWMR handle: the extent
/// ends at it and the chunk is stored under mask `argv[3]`.
const LIVE: &str = r#"
import sys, h5py, numpy as np
path, i, mask = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
want = np.array([i], dtype='<u8').tobytes()
with h5py.File(path, 'r', swmr=True) as f:
    d = f['frames']
    assert d.shape == (i + 1, 8), d.shape
    assert d[i].tobytes() == want, d[i]
    got = d.id.read_direct_chunk((i, 0))
    assert got == (mask, want), got
"#;

/// Checks the closed stream `argv[1]` of `argv[2]` frames (whole read, and
/// `read_direct_chunk` under mask `argv[3]` at the chunks `argv[4]` lists),
/// then writes the same shape with libhdf5 into `argv[5]` and compares the
/// two extensible-array headers' statistics. The data block bytes depend on
/// the entry size, which libhdf5 2.x widens for filtered chunks, so that one
/// is compared only when the entry sizes agree.
const CLOSED: &str = r#"
import sys, struct, h5py, numpy as np
path, n, mask, ref = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[5]
at = [int(c) for c in sys.argv[4].split(',')]
want = np.arange(n, dtype='<u8').view('u1').reshape(n, 8)
bad = []
with h5py.File(path, 'r') as f:
    d = f['frames']
    if d.shape != (n, 8):
        bad.append(f'shape {d.shape}')
    else:
        got = d[...]
        if not np.array_equal(got, want):
            bad.append(f'first wrong frame {int(np.argmax((got != want).any(axis=1)))}')
    for i in at:
        got = d.id.read_direct_chunk((i, 0))
        if got != (mask, want[i].tobytes()):
            bad.append(f'chunk {i}: {got}')
with h5py.File(ref, 'w', libver='latest') as f:
    kw = dict(compression='gzip') if mask else {}
    f.create_dataset('frames', data=want, maxshape=(None, 8), chunks=(1, 8), **kw)

def eahd(p):
    data = open(p, 'rb').read()
    i = data.find(b'EAHD')
    assert i >= 0 and data.find(b'EAHD', i + 1) < 0, f'{p}: not one EAHD'
    names = ['nsblks', 'sblk_size', 'ndblks', 'dblk_size', 'max_idx_set', 'nelmts']
    return data[i + 6], dict(zip(names, struct.unpack_from('<6Q', data, i + 12)))

(raw, ours), (ref_raw, theirs) = eahd(path), eahd(ref)
for k in ours:
    if k == 'dblk_size' and raw != ref_raw:
        continue
    if ours[k] != theirs[k]:
        bad.append(f'{k}: ours {ours[k]}, libhdf5 {theirs[k]} (entry sizes {raw}, {ref_raw})')
assert not bad, '\n'.join(bad)
"#;

/// An 8-byte frame that names its own index.
fn frame(i: u64) -> Vec<u8> {
    i.to_le_bytes().to_vec()
}

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_swmr_ea_paged_{label}_{}_{n}.h5",
        std::process::id()
    ))
}

/// Chunk indices where a page, a data block or a super block starts or ends,
/// past the first paged one: pages are 1024 entries, super block 13's data
/// blocks 2048 and super block 14's 4096.
fn boundaries() -> Vec<u64> {
    vec![
        FIRST_PAGED - 1,
        FIRST_PAGED,
        FIRST_PAGED + 1,
        FIRST_PAGED + 1023,
        FIRST_PAGED + 1024,
        FIRST_PAGED + 2047,
        FIRST_PAGED + 2048,
        SBLK_14 - 1,
        SBLK_14,
        SBLK_14 + 1023,
        SBLK_14 + 1024,
        N - 1,
    ]
}

fn run_python(script: &str, args: &[String]) {
    let Some(py) = python() else { return };
    let out = std::process::Command::new(py)
        .arg("-c")
        .arg(script)
        .args(args)
        .output()
        .expect("spawn python");
    assert!(
        out.status.success(),
        "h5py check {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Streams `N` frames, writing each through `append`, and checks after each
/// boundary flush that the live reader and h5py find the newest frame and
/// nothing past it. `mask` is the filter mask every chunk is stored under.
fn stream_reads_back(
    label: &str,
    create: impl FnOnce(&mut SwmrFileWriter) -> usize,
    append: impl Fn(&mut SwmrFileWriter, usize, &[u8]),
    mask: u32,
) {
    let path = tmp(label);
    let p = path.display().to_string();
    let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
    let ds = create(&mut w);
    w.start_swmr().unwrap();
    for i in 0..FIRST_PAGED - 1 {
        append(&mut w, ds, &frame(i));
    }
    w.flush().unwrap();
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    let newest = |r: &mut SwmrFileReader, i: u64| r.read_chunk_raw("frames", &[i, 0]).unwrap();

    let checks = boundaries();
    let live = [FIRST_PAGED + 1024, N - 1];
    for i in FIRST_PAGED - 1..N {
        append(&mut w, ds, &frame(i));
        if checks.contains(&i) {
            w.flush().unwrap();
            r.refresh().unwrap();
            assert_eq!(newest(&mut r, i), Some((frame(i), mask)), "frame {i}");
            assert_eq!(newest(&mut r, i + 1), None, "frame {} ahead", i + 1);
            if live.contains(&i) {
                run_python(LIVE, &[p.clone(), i.to_string(), mask.to_string()]);
            }
        }
    }
    w.close().unwrap();

    r.refresh().unwrap();
    for &i in &checks {
        assert_eq!(newest(&mut r, i), Some((frame(i), mask)), "frame {i}");
    }
    drop(r);
    let all = H5File::open(&path)
        .unwrap()
        .dataset("frames")
        .unwrap()
        .read_raw::<u8>()
        .unwrap();
    let want: Vec<u8> = (0..N).flat_map(frame).collect();
    assert_eq!(all.len(), want.len());
    let first_bad = all.iter().zip(&want).position(|(a, b)| a != b);
    assert_eq!(first_bad, None, "first wrong byte");

    let reference = tmp(&format!("{label}_libhdf5"));
    let at: Vec<String> = checks.iter().map(u64::to_string).collect();
    run_python(
        CLOSED,
        &[
            p,
            N.to_string(),
            mask.to_string(),
            at.join(","),
            reference.display().to_string(),
        ],
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&reference);
}

#[test]
fn an_unfiltered_stream_reads_back_past_the_first_paged_data_block() {
    stream_reads_back(
        "unfiltered",
        |w| w.create_streaming_dataset::<u8>("frames", &[8]).unwrap(),
        |w, ds, f| w.append_frame(ds, f).unwrap(),
        0,
    );
}

/// Filtered entries are wider (address, stored size, mask), so the pages and
/// data blocks are too. Frames go in verbatim under mask 1, as a raw
/// `H5Dwrite_chunk` with the filter skipped stores them.
#[test]
fn a_filtered_stream_reads_back_past_the_first_paged_data_block() {
    stream_reads_back(
        "filtered",
        |w| {
            w.create_streaming_dataset_compressed::<u8>("frames", &[8], FilterPipeline::deflate(4))
                .unwrap()
        },
        |w, ds, f| w.append_frame_raw(ds, f, 1).unwrap(),
        1,
    );
}

/// Checks `argv[1]`'s `frames` against `argv[2]` (the expected bytes) and the
/// rows libhdf5 finds a stored chunk for against `argv[3]`.
const PRUNED: &str = r#"
import sys, h5py, numpy as np
path, want, stored = sys.argv[1:4]
want = np.fromfile(want, 'u1').reshape(-1, 8)
stored = [int(r) for r in open(stored).read().split()]
with h5py.File(path, 'r') as f:
    d = f['frames']
    got = d[...]
    assert got.shape == want.shape, got.shape
    wrong = np.flatnonzero((got != want).any(axis=1))
    assert wrong.size == 0, f'rows {wrong[:8]}: {got[wrong[:2]]}'
    found = []
    d.id.chunk_iter(lambda c: found.append(c.chunk_offset[0]))
    extra, missing = sorted(set(found) - set(stored)), sorted(set(stored) - set(found))
    assert not extra and not missing, (extra[:8], missing[:8])
"#;

/// Checks `argv[1]` after `frames` was deleted and `again` written in its
/// place with the first `argv[2]` frames.
const RECREATED: &str = r#"
import sys, h5py, numpy as np
path, n = sys.argv[1], int(sys.argv[2])
with h5py.File(path, 'r') as f:
    assert 'frames' not in f
    got = f['again'][...]
    assert np.array_equal(got, np.arange(n, dtype='<u8').view('u1').reshape(n, 8))
"#;

/// The other paths through a paged data block, without SWMR: a partial-chunk
/// write starts from the chunk in an existing page, and from fill in a page or
/// a data block not created yet; a shrink clears entries across pages and data
/// blocks; and deleting the dataset frees its data blocks with their pages, so
/// a dataset written again in its place reuses that space.
#[test]
fn a_file_rewrites_shrinks_and_frees_paged_data_blocks() {
    let path = tmp("plain");
    // Offsets into super block 13: its data blocks are 2048 entries, two
    // pages each.
    let at = |off: u64| FIRST_PAGED + off;
    let (head, later, grown, kept) = (at(1000), at(2048)..at(4000), at(5000), at(2100));
    // In page 0, in page 1 (absent until written), in data block 2 (absent).
    let rmw = [at(50), at(1500), at(4500)];
    let rows = |r: std::ops::Range<u64>| -> Vec<u8> { r.flat_map(frame).collect() };
    let create = |f: &H5File, name: &str| {
        let ds = f
            .new_dataset::<u8>()
            .shape([head as usize, 8])
            .chunk(&[1, 8])
            .max_shape(&[None, Some(8)])
            .create(name)
            .unwrap();
        ds.write_raw(&rows(0..head)).unwrap();
        ds
    };
    {
        let f = H5File::create(&path).unwrap();
        let ds = create(&f, "frames");
        ds.set_extent(&[grown as usize, 8]).unwrap();
        let n = (later.end - later.start) as usize;
        ds.write_slice(&[later.start as usize, 0], &[n, 8], &rows(later.clone()))
            .unwrap();
        for r in rmw {
            ds.write_slice(&[r as usize, 2], &[1, 3], &[0xAAu8; 3])
                .unwrap();
        }
        ds.set_extent(&[kept as usize, 8]).unwrap();
        ds.set_extent(&[grown as usize, 8]).unwrap();
        f.close().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert!(
        bytes.windows(4).any(|w| w == b"EAHD"),
        "not an extensible array"
    );

    let stored: Vec<u64> = (0..head).chain([rmw[1]]).chain(later.start..kept).collect();
    let mut want = vec![0u8; grown as usize * 8];
    for &r in &stored {
        if r != rmw[1] {
            want[r as usize * 8..][..8].copy_from_slice(&frame(r));
        }
    }
    for r in &rmw[..2] {
        want[*r as usize * 8 + 2..][..3].fill(0xAA);
    }
    let (want_path, stored_path) = (tmp("plain_want"), tmp("plain_stored"));
    std::fs::write(&want_path, &want).unwrap();
    let list: Vec<String> = stored.iter().map(u64::to_string).collect();
    std::fs::write(&stored_path, list.join("\n")).unwrap();
    run_python(
        PRUNED,
        &[&path, &want_path, &stored_path].map(|p| p.display().to_string()),
    );
    let _ = std::fs::remove_file(&want_path);
    let _ = std::fs::remove_file(&stored_path);

    let before = std::fs::metadata(&path).unwrap().len();
    {
        let f = H5File::open_rw(&path).unwrap();
        f.delete_dataset("frames").unwrap();
        create(&f, "again");
        f.close().unwrap();
    }
    let after = std::fs::metadata(&path).unwrap().len();
    // Written again without the freed space, the file would grow by the whole
    // dataset: 1 MB of chunks and as much again in data blocks and pages.
    assert!(
        after < before + (1 << 20),
        "file grew from {before} to {after} bytes"
    );
    run_python(RECREATED, &[path.display().to_string(), head.to_string()]);
    let _ = std::fs::remove_file(&path);
}
