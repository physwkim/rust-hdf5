//! A SWMR reader reads a metadata image again while its checksum fails, up to
//! `SwmrFileReader::metadata_read_attempts` times: the writer may be part way
//! through rewriting it. The tests tear an image on disk and restore it from
//! another thread while the reader is retrying.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};
use rust_hdf5::FileLocking;

fn tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_swmr_retry_{label}_{}_{n}.h5",
        std::process::id()
    ))
}

fn attempts(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}

/// A closed SWMR file holding three 4-byte frames.
fn stream(path: &Path) {
    let mut w = SwmrFileWriter::create_with_locking(path, FileLocking::Disabled).unwrap();
    let ds = w.create_streaming_dataset::<u8>("frames", &[4]).unwrap();
    w.start_swmr().unwrap();
    for i in 0..3u8 {
        w.append_frame(ds, &[i; 4]).unwrap();
    }
    w.flush().unwrap();
    w.close().unwrap();
}

/// Flip byte `at` of the file in place; flipping it again restores it.
fn flip(path: &Path, at: u64) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut b = [0u8];
    f.seek(SeekFrom::Start(at)).unwrap();
    f.read_exact(&mut b).unwrap();
    f.seek(SeekFrom::Start(at)).unwrap();
    f.write_all(&[!b[0]]).unwrap();
}

/// Restore byte `at` from another thread after `delay`.
fn restore_later(path: &Path, at: u64, delay: Duration) -> std::thread::JoinHandle<()> {
    let path = path.to_path_buf();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        flip(&path, at);
    })
}

/// The offset of the v2/v3 superblock's checksum: it closes the superblock,
/// whose four addresses are `sizeof_offsets` (byte 9) wide each.
fn superblock_checksum_at(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes[8] >= 2, "a v2/v3 superblock");
    12 + 4 * u64::from(bytes[9])
}

#[test]
fn a_torn_superblock_is_read_again_until_it_checks() {
    let path = tmp("superblock");
    stream(&path);
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    assert_eq!(r.metadata_read_attempts().get(), 100);
    let at = superblock_checksum_at(&path);
    flip(&path, at);

    r.set_metadata_read_attempts(attempts(3));
    let err = r.refresh().unwrap_err().to_string();
    assert!(err.contains("checksum mismatch"), "{err}");

    // 100 attempts sleep 1 ns, doubling up to 2^20 ns, after each of the
    // first 99: 80 * 2^20 ns in all.
    r.set_metadata_read_attempts(attempts(100));
    let t = Instant::now();
    let err = r.refresh().unwrap_err().to_string();
    assert!(err.contains("checksum mismatch"), "{err}");
    assert!(
        t.elapsed() >= Duration::from_nanos(80 << 20),
        "{:?}",
        t.elapsed()
    );

    // Enough attempts to outlast the restore by seconds.
    r.set_metadata_read_attempts(attempts(10_000));
    let fixer = restore_later(&path, at, Duration::from_millis(30));
    r.refresh().unwrap();
    fixer.join().unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![3, 4]);
    let _ = std::fs::remove_file(&path);
}

/// The offset of the checksum closing chunk 0 of the root group's v2 object
/// header: after the prefix (signature, version, flags, the optional times
/// and phase-change fields, the chunk-size field the flags size) and the
/// chunk the size field counts.
fn root_header_checksum_at(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    let w = usize::from(bytes[9]);
    let root_at = 12 + 3 * w;
    let mut root = [0u8; 8];
    root[..w].copy_from_slice(&bytes[root_at..root_at + w]);
    let root = u64::from_le_bytes(root) as usize;
    assert_eq!(&bytes[root..root + 4], b"OHDR");
    let flags = bytes[root + 5];
    let mut pos = root + 6;
    if flags & 0x20 != 0 {
        pos += 16;
    }
    if flags & 0x10 != 0 {
        pos += 4;
    }
    let size_len = 1usize << (flags & 0x03);
    let mut size = [0u8; 8];
    size[..size_len].copy_from_slice(&bytes[pos..pos + size_len]);
    (pos + size_len) as u64 + u64::from_le_bytes(size)
}

#[test]
fn a_torn_object_header_is_read_again_until_it_checks() {
    let path = tmp("ohdr");
    stream(&path);
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    let at = root_header_checksum_at(&path);
    flip(&path, at);

    r.set_metadata_read_attempts(attempts(2));
    let err = r.refresh().unwrap_err().to_string();
    assert!(err.contains("checksum mismatch"), "{err}");

    r.set_metadata_read_attempts(attempts(10_000));
    let fixer = restore_later(&path, at, Duration::from_millis(30));
    r.refresh().unwrap();
    fixer.join().unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![3, 4]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_reader_opened_on_a_torn_superblock_reads_it_again() {
    let path = tmp("open");
    stream(&path);
    let at = superblock_checksum_at(&path);
    flip(&path, at);
    let fixer = restore_later(&path, at, Duration::from_millis(30));
    // 100 attempts take ~84 ms, which the 30 ms restore lands inside of.
    let r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    fixer.join().unwrap();
    assert_eq!(r.dataset_names(), vec!["frames".to_string()]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_torn_chunk_index_block_is_read_again_until_it_checks() {
    let path = tmp("index");
    stream(&path);
    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    // One unlimited axis: an extensible array, whose index block holds the
    // first chunks' addresses right after its 14-byte prefix (signature,
    // version, client id, header address).
    let bytes = std::fs::read(&path).unwrap();
    let eaib = bytes.windows(4).position(|w| w == b"EAIB").unwrap() as u64;
    let at = eaib + 14;
    flip(&path, at);

    r.set_metadata_read_attempts(attempts(2));
    let err = r.read_chunk_raw("frames", &[0, 0]).unwrap_err().to_string();
    assert!(err.contains("checksum mismatch"), "{err}");

    r.set_metadata_read_attempts(attempts(10_000));
    let fixer = restore_later(&path, at, Duration::from_millis(30));
    assert_eq!(
        r.read_chunk_raw("frames", &[0, 0]).unwrap(),
        Some((vec![0u8; 4], 0))
    );
    fixer.join().unwrap();
    assert_eq!(
        r.read_chunk_raw("frames", &[2, 0]).unwrap(),
        Some((vec![2u8; 4], 0))
    );
    let _ = std::fs::remove_file(&path);
}

/// A SWMR writer reads metadata again while its checksum fails, as
/// `H5F__start_swmr_write` gives it libhdf5's SWMR read attempts: once it
/// has released its lock, another opener may be rewriting what it loads.
#[test]
fn a_swmr_writer_reads_a_torn_data_block_again() {
    let path = tmp("writer");
    {
        let mut w = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let ds = w.create_streaming_dataset::<u8>("frames", &[4]).unwrap();
        w.start_swmr().unwrap();
        for i in 0..40u8 {
            w.append_frame(ds, &[i; 4]).unwrap();
        }
        w.close().unwrap();
    }
    let mut w = SwmrFileWriter::open_append_with_locking(&path, FileLocking::Disabled).unwrap();
    let ds = w.dataset_index("frames").unwrap();
    w.start_swmr().unwrap();
    // Frame 40 goes in a data block already on disk, which the reopened
    // writer has to load. Each block's first entry follows its prefix:
    // signature, version, client id, header address, 4-byte block offset.
    let bytes = std::fs::read(&path).unwrap();
    let blocks: Vec<u64> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"EADB")
        .map(|(at, _)| at as u64 + 18)
        .collect();
    assert_eq!(blocks.len(), 2);
    for &at in &blocks {
        flip(&path, at);
    }
    let fixer = {
        let path = path.clone();
        let blocks = blocks.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            for at in blocks {
                flip(&path, at);
            }
        })
    };
    w.append_frame(ds, &[40; 4]).unwrap();
    fixer.join().unwrap();
    w.close().unwrap();

    let mut r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    assert_eq!(r.dataset_shape("frames").unwrap(), vec![41, 4]);
    for i in 0..41u64 {
        assert_eq!(
            r.read_chunk_raw("frames", &[i, 0]).unwrap(),
            Some((vec![i as u8; 4], 0))
        );
    }
    let _ = std::fs::remove_file(&path);
}
