//! `H5Pset_fill_time`/`H5Pget_fill_time` — when a dataset's fill value is
//! written into allocated storage, as distinct from [`FillValue`] (what the
//! fill value is) and `alloc_time` (when storage itself is allocated).
//!
//! `H5D__chunk_lock`'s cache-miss path (H5Dchunk.c:4894) fills a newly locked
//! chunk buffer only for `ALLOC`, or for `IFSET` with a fill value defined;
//! `NEVER` leaves it as the zeros a fresh buffer already has. That is the one
//! place this crate's writer must honor the policy — everywhere else (a
//! shrink's straddler refill) fills unconditionally, matching
//! `H5D__chunk_prune_fill`.
//!
//! `H5D__update_oh_info` (H5Dint.c:927-943) carries one more special case:
//! a variable-length datatype with no explicit fill value forces `ALLOC`
//! regardless of the declared policy, because a heap-reference encoding has
//! no safe all-zero "no fill" representation.

use rust_hdf5::{FillTime, H5File};

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_fill_time_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// A dataset created with no `.fill_time(...)` call reads back `IfSet`,
/// `H5D_CRT_FILL_TIME_DEF` — the default every dataset creation property list
/// carries. This is the negative control: before this finding there was no
/// `fill_time()` accessor and no `filltime` oracle field, so nothing observed
/// this byte. Before wave-6 (`ec54601`) the writer hardcoded the fill
/// message's write-time field to `ALLOC` (0) regardless of policy; this
/// assertion is exactly the byte a regression back to that would flip.
#[test]
fn default_fill_time_is_ifset() {
    let path = tmp("default");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .fill_value(7)
            .create("d")
            .unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().fill_time().unwrap(),
        FillTime::IfSet
    );
    std::fs::remove_file(&path).ok();
}

/// The accessor reads back each of the three declared policies distinctly —
/// `H5Pset_fill_time` followed by close/reopen/`H5Pget_fill_time`.
#[test]
fn fill_time_accessor_reads_declared_policy() {
    let path = tmp("policies");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .fill_value(1)
            .fill_time(FillTime::Alloc)
            .create("alloc")
            .unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .fill_value(1)
            .fill_time(FillTime::Never)
            .create("never")
            .unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .fill_value(1)
            .fill_time(FillTime::IfSet)
            .create("ifset")
            .unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("alloc").unwrap().fill_time().unwrap(),
        FillTime::Alloc
    );
    assert_eq!(
        file.dataset("never").unwrap().fill_time().unwrap(),
        FillTime::Never
    );
    assert_eq!(
        file.dataset("ifset").unwrap().fill_time().unwrap(),
        FillTime::IfSet
    );
    std::fs::remove_file(&path).ok();
}

/// `fill_time()` is a read-mode accessor — same shape as `fill_value()`,
/// `storage_layout()`.
#[test]
fn fill_time_unavailable_in_write_mode() {
    let path = tmp("write_mode");
    let file = H5File::create(&path).unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shape([4])
        .fill_value(1)
        .create("d")
        .unwrap();
    assert!(ds.fill_time().is_err());
    std::fs::remove_file(&path).ok();
}

/// A NULL dataspace dataset rejects `.fill_time(...)` the same way it rejects
/// `.fill_value(...)`: there is no allocated storage for a write-time policy
/// to describe.
#[test]
fn fill_time_rejected_on_null_dataspace() {
    let path = tmp("null");
    let file = H5File::create(&path).unwrap();
    let result = file
        .new_dataset::<i32>()
        .null()
        .fill_time(FillTime::Never)
        .create("d");
    let msg = match result {
        Ok(_) => panic!("expected a NULL dataspace + fill_time to be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("fill time"), "unexpected error: {msg}");
    std::fs::remove_file(&path).ok();
}

/// The byte-level contrast: `NEVER` must skip this writer's own eager tiling
/// of the fill value into a chunk touched for the first time by a partial
/// write, while `ALLOC` fills it. Verified against a real h5py capture with
/// the same dcpl (chunk `[4]`, `fillvalue=99`, element 0 written): libhdf5
/// produces `[7, 0, 0, 0]` for `NEVER` and `[7, 99, 99, 99]` for `ALLOC`.
#[test]
fn never_skips_eager_tiling_alloc_fills_it() {
    let path = tmp("contrast");
    {
        let file = H5File::create(&path).unwrap();
        let alloc = file
            .new_dataset::<i32>()
            .shape([4])
            .chunk(&[4])
            .fill_value(99)
            .fill_time(FillTime::Alloc)
            .create("alloc")
            .unwrap();
        alloc.write_slice(&[0], &[1], &[7i32]).unwrap();

        let never = file
            .new_dataset::<i32>()
            .shape([4])
            .chunk(&[4])
            .fill_value(99)
            .fill_time(FillTime::Never)
            .create("never")
            .unwrap();
        never.write_slice(&[0], &[1], &[7i32]).unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("alloc").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 99, 99, 99]
    );
    assert_eq!(
        file.dataset("never").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 0, 0, 0]
    );
    std::fs::remove_file(&path).ok();
}

/// A variable-length-typed dataset with no explicit fill value reads back
/// `Alloc`, not the `IfSet` every other datatype defaults to — the oracle
/// caught this: `dcpl.get_fill_time()` on a rust-written `str_vlen_ascii`
/// dataset reported `alloc` while this crate reported `ifset` for the same
/// file, because the writer stamped every dataset's declared default
/// verbatim without applying `H5D__update_oh_info`'s VL special case.
#[test]
fn vlen_dataset_with_no_fill_value_forces_alloc() {
    let path = tmp("vlen");
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings_ascii("strings", &["a", "bb", "ccc"])
            .unwrap();
        file.write_vlen_numeric("seqs", &[&[1i32, 2, 3][..], &[4, 5][..]])
            .unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("strings").unwrap().fill_time().unwrap(),
        FillTime::Alloc
    );
    assert_eq!(
        file.dataset("seqs").unwrap().fill_time().unwrap(),
        FillTime::Alloc
    );
    std::fs::remove_file(&path).ok();
}

/// A dataset reopened for append preserves its declared fill-time policy
/// instead of resetting to `IFSET` — the same round-trip `fill_value` gets
/// through `DatasetParts`.
#[test]
fn fill_time_survives_reopen_and_append() {
    let path = tmp("reopen");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([2])
            .max_shape(&[None])
            .chunk(&[2])
            .fill_value(99)
            .fill_time(FillTime::Never)
            .create("d")
            .unwrap();
        ds.write_slice(&[0], &[1], &[7i32]).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        let ds = file.dataset_writer("d").unwrap();
        ds.extend(&[4]).unwrap();
        ds.write_slice(&[2], &[1], &[8i32]).unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_time().unwrap(), FillTime::Never);
    // The never-touched element (index 3) reads as raw zero, not the fill
    // value: NEVER was preserved across the reopen, not reset to IFSET.
    assert_eq!(ds.read_raw::<i32>().unwrap(), vec![7, 0, 8, 0]);
    std::fs::remove_file(&path).ok();
}
