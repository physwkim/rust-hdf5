//! Cross-process file-locking integration tests.
//!
//! These tests open the same HDF5 file twice from the same process to verify
//! that the OS-level advisory lock actually blocks conflicting opens. On Unix
//! `flock(2)` and `fcntl(F_OFD_SETLK)` (which is what Rust's std uses
//! internally on Linux/macOS) treat each open file description independently,
//! so two opens within the same process can still conflict — exactly what we
//! want to verify here.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};
use rust_hdf5::{FileLocking, H5File};

/// Force `FileLocking::Enabled` regardless of the shell's
/// `HDF5_USE_FILE_LOCKING` setting. Each integration test below uses
/// `enabled().{create,open,open_rw}` instead of the bare `H5File::create`
/// helpers so that the test outcome doesn't depend on the operator's env.
fn enabled() -> rust_hdf5::H5FileOptions {
    H5File::options().locking(FileLocking::Enabled)
}

/// Per-test unique temp path. Avoids collisions when cargo runs tests in
/// parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_lock_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

#[test]
fn second_writer_open_is_blocked() {
    let path = unique_tmp("two_writers");
    let _file1 = enabled().create(&path).unwrap();

    let err = enabled().open_rw(&path);
    assert!(
        err.is_err(),
        "expected second writer open to fail with lock conflict"
    );
}

#[test]
fn reader_blocks_writer() {
    let path = unique_tmp("reader_blocks_writer");
    enabled().create(&path).unwrap().close().unwrap();
    let _reader = enabled().open(&path).unwrap();
    let err = enabled().open_rw(&path);
    assert!(
        err.is_err(),
        "expected writer open to be blocked by reader's shared lock"
    );
}

#[test]
fn writer_blocks_reader() {
    let path = unique_tmp("writer_blocks_reader");
    let _writer = enabled().create(&path).unwrap();
    let err = enabled().open(&path);
    assert!(
        err.is_err(),
        "expected reader open to be blocked by writer's exclusive lock"
    );
}

#[test]
fn multiple_readers_coexist() {
    let path = unique_tmp("multi_readers");
    enabled().create(&path).unwrap().close().unwrap();
    let _r1 = enabled().open(&path).unwrap();
    let _r2 = enabled().open(&path).unwrap();
    let _r3 = enabled().open(&path).unwrap();
}

#[test]
fn disabled_locking_bypasses_conflict() {
    let path = unique_tmp("disabled");
    // Materialize a valid file first (close so the superblock is written).
    enabled().create(&path).unwrap().close().unwrap();

    let _w1 = H5File::options().no_locking().open_rw(&path).unwrap();
    // With locking disabled the second writer should succeed.
    let w2 = H5File::options().no_locking().open_rw(&path);
    assert!(
        w2.is_ok(),
        "expected disabled-locking second open to succeed: {:?}",
        w2.err()
    );
}

// Unix-only: this test relies on advisory-lock semantics. On Windows
// `LockFileEx` is mandatory, so a second opener cannot read the file
// while the first holder has an exclusive lock — even when the second
// opener uses BestEffort and skips its own lock acquisition. The
// HDF5 C library has the same limitation on Windows.
#[cfg(unix)]
#[test]
fn best_effort_does_not_error_on_conflict() {
    let path = unique_tmp("best_effort");
    enabled().create(&path).unwrap().close().unwrap();

    // First holder takes a real exclusive lock.
    let _w1 = enabled().open_rw(&path).unwrap();
    // Best-effort opener should not error even though the lock is contended.
    let w2 = H5File::options().best_effort_locking().open_rw(&path);
    assert!(
        w2.is_ok(),
        "expected best-effort open to succeed despite conflict: {:?}",
        w2.err()
    );
}

#[test]
fn lock_releases_on_drop() {
    let path = unique_tmp("release_on_drop");
    // create()+close() leaves a valid file with no lingering lock.
    enabled().create(&path).unwrap().close().unwrap();
    {
        let _w1 = enabled().open_rw(&path).unwrap();
    } // _w1 dropped — lock released
      // Now another writer should be able to open without error.
    let w2 = enabled().open_rw(&path);
    assert!(
        w2.is_ok(),
        "expected reopen after drop to succeed: {:?}",
        w2.err()
    );
}

#[test]
fn create_with_lock_conflict_does_not_truncate_existing_file() {
    // Regression: H5File::create used to call OpenOptions::truncate(true)
    // before acquiring the lock, so a lock conflict on an existing file
    // would destroy its contents while still returning an error.
    let path = unique_tmp("preserve_on_conflict");

    // Stage 1: create a real file with a known dataset and close it.
    {
        let f = enabled().create(&path).unwrap();
        let ds = f.new_dataset::<u8>().shape([4]).create("data").unwrap();
        ds.write_raw(&[1u8, 2, 3, 4]).unwrap();
        f.close().unwrap();
    }
    let original_size = std::fs::metadata(&path).unwrap().len();
    assert!(original_size > 0, "fixture file should be non-empty");

    // Stage 2: hold the file open with an exclusive lock.
    let _holder = enabled().open_rw(&path).unwrap();

    // Stage 3: a second create() must error AND must not have truncated
    // the file on disk.
    let conflict = enabled().create(&path);
    assert!(conflict.is_err(), "second create should fail under lock");
    let size_after = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        size_after, original_size,
        "file must not be truncated when create() loses the lock race"
    );
}

#[test]
fn lock_releases_on_close() {
    let path = unique_tmp("release_on_close");
    let w1 = enabled().create(&path).unwrap();
    w1.close().unwrap();
    // After explicit close(), another writer must succeed.
    let w2 = enabled().open_rw(&path);
    assert!(
        w2.is_ok(),
        "expected reopen after close to succeed: {:?}",
        w2.err()
    );
}

#[test]
fn options_locking_overrides_env_enabled_blocks() {
    // Cross-platform: with Enabled policy, a second open_rw must error.
    // We don't mutate HDF5_USE_FILE_LOCKING here because cargo runs tests
    // in parallel and that would race with other tests. Instead we use
    // `options().locking(Enabled)` to be explicit.
    let path = unique_tmp("options_override_enabled");
    enabled().create(&path).unwrap().close().unwrap();

    let _w1 = enabled().open_rw(&path).unwrap();
    let conflict = enabled().open_rw(&path);
    assert!(conflict.is_err(), "Enabled policy should block second open");
}

// Unix-only: a second opener with `Disabled` cannot read the file on
// Windows while the first holder owns the exclusive `LockFileEx` range
// — Windows locks are mandatory, not advisory.
#[cfg(unix)]
#[test]
fn options_locking_disabled_bypasses_real_lock() {
    let path = unique_tmp("options_override_disabled");
    enabled().create(&path).unwrap().close().unwrap();

    let _w1 = enabled().open_rw(&path).unwrap();
    let bypass = H5File::options()
        .locking(FileLocking::Disabled)
        .open_rw(&path);
    assert!(bypass.is_ok(), "Disabled policy should bypass the lock");
}

#[test]
fn swmr_reader_attaches_after_start_swmr() {
    let path = unique_tmp("swmr_attach");

    // Writer holds an exclusive lock.
    let mut writer = SwmrFileWriter::create_with_locking(&path, FileLocking::Enabled).unwrap();
    let _ds = writer
        .create_streaming_dataset::<f32>("frames", &[4, 4])
        .unwrap();

    // Before start_swmr, the writer's exclusive lock blocks readers.
    let too_early = SwmrFileReader::open_with_locking(&path, FileLocking::Enabled);
    assert!(
        too_early.is_err(),
        "SWMR reader should be blocked before start_swmr"
    );

    // After start_swmr the writer releases its exclusive lock so a SWMR
    // reader (which takes a shared lock) can attach. The SWMR protocol
    // itself assumes a single writer — we don't enforce that via OS
    // locks here because Windows `LockFileEx` semantics make the
    // exclusive→shared downgrade unsafe for subsequent writes through
    // the same handle.
    writer.start_swmr().unwrap();
    let reader = SwmrFileReader::open_with_locking(&path, FileLocking::Enabled);
    assert!(
        reader.is_ok(),
        "SWMR reader should attach after start_swmr: {:?}",
        reader.err()
    );
    drop(reader);
    writer.close().unwrap();
}

#[test]
fn swmr_disabled_locking_allows_concurrent_writer() {
    let path = unique_tmp("swmr_no_lock");
    let mut writer = SwmrFileWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
    let _ds = writer
        .create_streaming_dataset::<f32>("frames", &[2, 2])
        .unwrap();
    writer.start_swmr().unwrap();

    // With locking disabled, a second SWMR reader still attaches (it would
    // even with locking enabled because of the shared-lock downgrade), and
    // a second writer with locking disabled is also allowed (no enforcement).
    let _r = SwmrFileReader::open_with_locking(&path, FileLocking::Disabled).unwrap();
    let other_writer = H5File::options().no_locking().open_rw(&path);
    assert!(
        other_writer.is_ok(),
        "disabled-locking second writer should succeed: {:?}",
        other_writer.err()
    );
}

/// A virtual dataset's source file is opened under the *virtual* file's
/// locking policy, not under one of its own. `H5D__virtual_open_source_dset`
/// hands `H5F_prefix_open_file` `source_fapl` (H5Dvirtual.c:877-882), which
/// `H5D__virtual_init` made as a copy of the virtual file's own file-access
/// property list (H5Dvirtual.c:2193-2194) — and `H5F_get_access_plist` copies
/// `use_file_locking` into it verbatim (H5Fint.c:389). Measured against
/// libhdf5 1.14.6 by watching `flock` on the source across a VDS read: locked
/// under the default, unlocked under `HDF5_USE_FILE_LOCKING=FALSE`.
///
/// The source is opened lazily, on the read that needs it — libhdf5 opens it
/// in `H5D__virtual_read_one`'s path, not at `H5Dopen`, and the same
/// measurement shows the source's descriptor appearing only after the read.
#[test]
fn a_virtual_source_takes_the_virtual_file_s_lock() {
    use rust_hdf5::Selection;
    let src = unique_tmp("vds_source");
    let vds = src.parent().unwrap().join("vds_holder.h5");
    enabled()
        .create(&src)
        .unwrap()
        .new_dataset::<i32>()
        .shape([2usize, 4])
        .create("data")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    {
        let file = enabled().create(&vds).unwrap();
        file.new_dataset::<i32>()
            .shape([2usize, 4])
            .fill_value(-9i32)
            .virtual_mapping(
                Selection::All,
                src.to_str().unwrap(),
                "data",
                Selection::All,
            )
            .create("v")
            .unwrap();
        file.close().unwrap();
    }

    let want: Vec<i32> = (0..8).collect();
    {
        let reader = enabled().open(&vds).unwrap();
        let ds = reader.dataset("v").unwrap();
        assert!(
            enabled().open_rw(&src).is_ok(),
            "the source is not opened until the read that needs it"
        );
        assert_eq!(ds.read_raw::<i32>().unwrap(), want);
        assert!(
            enabled().open_rw(&src).is_err(),
            "expected the source to be locked by the virtual file's reader"
        );
    }
    assert!(
        enabled().open_rw(&src).is_ok(),
        "expected the source lock to go with the virtual file's reader"
    );

    // The same read under a virtual file opened with locking off leaves the
    // source unlocked, because the source inherits that policy too.
    {
        let reader = H5File::options().no_locking().open(&vds).unwrap();
        let ds = reader.dataset("v").unwrap();
        assert_eq!(ds.read_raw::<i32>().unwrap(), want);
        assert!(
            enabled().open_rw(&src).is_ok(),
            "expected no lock on the source when the virtual file disabled locking"
        );
    }
    let _ = std::fs::remove_dir_all(src.parent().unwrap());
}
