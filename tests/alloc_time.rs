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

/// `H5D__alloc_storage` allocates nothing for a dataset stored through an
/// external file list — "we assume that external storage is already allocated
/// by the caller, or at least will be before I/O is performed" — and the
/// `H5D__init_storage` that would tile the fill value into that storage sits
/// inside the same skipped block (H5Dint.c:2270-2274). So declaring a fill
/// value does not bring the raw data file into existence; only a write does.
///
/// Measured under libhdf5 1.14.6 and 2.0.0: with a user fill value,
/// `H5D_FILL_TIME_ALLOC` and `H5D_ALLOC_TIME_EARLY` all set, `H5Dcreate2`
/// leaves the raw data file uncreated and a read before any write fails with
/// "unable to open external raw data file" instead of reporting the fill.
#[test]
fn creating_an_external_dataset_with_a_fill_value_writes_no_raw_data_file() {
    let dir = tmp("external_fill").with_extension("");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ext.h5");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4])
            .external(&[("raw.bin", 0, 16)])
            .efile_prefix(dir.display().to_string())
            .fill_value(7i32)
            .create("d")
            .unwrap();
        file.close().unwrap();
    }
    assert!(
        !dir.join("raw.bin").exists(),
        "the fill value must not reach storage this writer never allocated"
    );

    let file = H5File::open(&path).unwrap();
    assert!(
        file.dataset_with(
            "d",
            rust_hdf5::DatasetAccess::new().efile_prefix(dir.display().to_string())
        )
        .unwrap()
        .read_raw::<i32>()
        .is_err(),
        "reading before any write fails on the missing file rather than reporting the fill"
    );
    std::fs::remove_dir_all(&dir).ok();
}
