//! `H5Pget_alloc_time` — when a dataset's raw-data storage is allocated, as
//! distinct from `H5D_fill_time_t` ([`FillTime`], when the fill value is
//! written into that storage).
//!
//! Not user-settable in this crate (no `DatasetBuilder::alloc_time`, matching
//! upstream: `H5P__set_layout`, H5Pdcpl.c:1864-1877, picks the default from
//! the storage class alone — compact is `EARLY`, chunked and virtual are
//! `INCR`, contiguous is `LATE` — and nothing later overrides it for a
//! dataset this crate creates). [`H5Dataset::alloc_time`] exists to read that
//! declared byte back.

use rust_hdf5::{AllocTime, H5File, Selection};

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_alloc_time_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// A compact dataset with no fill value of its own reads back `Early`. This
/// is the finding's regression case: `build_dataset_header`'s no-fill-value,
/// non-chunked branch used to fall through to `FillValueMessage::default()`,
/// which hardcodes `alloc_time = 2` (late) — wrong for compact, only
/// coincidentally right for contiguous.
#[test]
fn compact_dataset_with_no_fill_value_is_early() {
    let path = tmp("compact_no_fill");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .compact()
            .create("d")
            .unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().alloc_time().unwrap(),
        AllocTime::Early
    );
    std::fs::remove_file(&path).ok();
}

/// A compact dataset with an explicit fill value also reads back `Early` —
/// this path already went through the correctly computed `alloc_time`
/// (the branch keyed on `m.fill_value.is_some()`), so it is the byte-
/// unchanged control for the no-fill-value case above.
#[test]
fn compact_dataset_with_fill_value_is_early() {
    let path = tmp("compact_fill");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .compact()
            .fill_value(7)
            .create("d")
            .unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().alloc_time().unwrap(),
        AllocTime::Early
    );
    std::fs::remove_file(&path).ok();
}

/// A contiguous dataset with no fill value reads back `Late` — unchanged by
/// the fix, since `FillValueMessage::default()`'s hardcoded `alloc_time = 2`
/// happened to already agree with the real default for this one layout
/// class. Pins the "byte-unchanged" half of the fix.
#[test]
fn contiguous_dataset_with_no_fill_value_is_late() {
    let path = tmp("contiguous_no_fill");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>().shape([4]).create("d").unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().alloc_time().unwrap(),
        AllocTime::Late
    );
    std::fs::remove_file(&path).ok();
}

/// A chunked dataset with no fill value reads back `Incr` — already correct
/// before this fix (the `is_chunked` branch used the computed `alloc_time`),
/// kept as a regression guard now that the no-fill-value branch was merged
/// with it.
#[test]
fn chunked_dataset_with_no_fill_value_is_incr() {
    let path = tmp("chunked_no_fill");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .chunk(&[2])
            .create("d")
            .unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().alloc_time().unwrap(),
        AllocTime::Incr
    );
    std::fs::remove_file(&path).ok();
}

/// A virtual dataset with no fill value of its own reads back `Incr` — the
/// finding's other regression case: `is_chunked()` is false for a virtual
/// layout, so this also fell through to the hardcoded `Late` before the fix.
#[test]
fn virtual_dataset_with_no_fill_value_is_incr() {
    let path = tmp("virtual_no_fill");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .virtual_mapping(Selection::All, ".", "src", Selection::All)
            .create("vds")
            .unwrap();
    }
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("vds").unwrap().alloc_time().unwrap(),
        AllocTime::Incr
    );
    std::fs::remove_file(&path).ok();
}

/// `alloc_time()` is a read-mode accessor — same shape as `fill_time()`,
/// `fill_value()`, `storage_layout()`.
#[test]
fn alloc_time_unavailable_in_write_mode() {
    let path = tmp("write_mode");
    let file = H5File::create(&path).unwrap();
    let ds = file.new_dataset::<i32>().shape([4]).create("d").unwrap();
    assert!(ds.alloc_time().is_err());
    std::fs::remove_file(&path).ok();
}
