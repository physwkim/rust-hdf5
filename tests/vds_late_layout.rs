//! Reading the VDS variant a dcpl builds when `H5Pset_virtual` is called
//! without `H5Pset_layout`.
//!
//! `H5Pset_virtual` installs the virtual layout with `H5P_poke` on
//! `H5D_CRT_LAYOUT_NAME` (H5Pdcpl.c:2146), which — unlike `H5Pset_layout`'s
//! `H5P__set_layout` — never re-derives the default allocation time from the
//! storage class (H5Pdcpl.c:1758-1782, where virtual would take
//! `H5D_ALLOC_TIME_INCR`). A dcpl that only ever sees `H5Pset_virtual`
//! therefore keeps the contiguous default `H5D_ALLOC_TIME_LATE` while
//! carrying a virtual layout.
//!
//! That is the entire on-disk difference, and it is one byte: writing the two
//! orders into separate files with libhdf5 1.14.6 gives byte-identical files
//! but for one offset, the space allocation time field of the version-2 fill
//! value message (`0x03` -> `0x02`) — measured on both a bounded mapping
//! (6176-byte files differing at offset 1497) and an unlimited one
//! (12208-byte files differing at offset 6161). The data layout message, the
//! mapping list, and the fill value itself are byte-identical.
//!
//! `tests/fixtures/vds_late_layout.h5` (from
//! `tests/fixtures/gen_vds_late_layout.sh`) holds both orders side by side
//! twice — a bounded mapping over `/src` and an unlimited one over the
//! extendible `/src_unlim` — so each pair is the assertion: same extent,
//! same data, same mappings, different allocation time. h5py's
//! `VirtualLayout` always calls `set_layout` first (h5py 3.15.1
//! `_hl/vds.py:174`, `:216`), so libhdf5 writes this fixture itself.
//!
//! Writing the variant is out of scope: this crate's builder has no
//! allocation-time control, so every VDS it writes takes the conventional
//! `Incremental`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{AllocTime, FillValue, H5File, Selection, StorageLayout};

const LATE_LAYOUT: &[u8] = include_bytes!("fixtures/vds_late_layout.h5");

/// The fixture's mappings name `.`, so it has to be read from a real path.
fn write_temp() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rust_hdf5_vds_late_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vds_late_layout.h5");
    std::fs::write(&path, LATE_LAYOUT).unwrap();
    path
}

fn cleanup(path: &Path) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The late variant resolves its extent, reads its mapped rows from the
/// source and its unmapped rows as the fill value, exactly as the
/// conventional one does. Measured against libhdf5 1.14.6: both datasets are
/// `(4, 4)` and read `[[0..4], [4..8], [-9; 4], [-9; 4]]`.
#[test]
fn the_late_order_reads_like_the_conventional_one() {
    let path = write_temp();
    let file = H5File::open(&path).unwrap();
    let want: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, -9, -9, -9, -9, -9, -9, -9, -9];
    for name in ["vds", "vds_late"] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.shape(), vec![4, 4], "{name}");
        assert_eq!(ds.read_raw::<i32>().unwrap(), want, "{name}");
        assert_eq!(
            ds.read_slice::<i32>(&[1, 1], &[2, 2]).unwrap(),
            vec![5, 6, -9, -9],
            "{name}"
        );
    }
    drop(file);
    cleanup(&path);
}

/// The late order resolves an unlimited extent the same way too: both
/// datasets store `(1, 4)` and both report the source's current three rows
/// after `H5D__virtual_set_extent_unlim` clips the mapping. Measured against
/// libhdf5 1.14.6: `(3, 4)` reading `100..112`.
#[test]
fn the_late_order_resolves_an_unlimited_extent_like_the_conventional_one() {
    let path = write_temp();
    let file = H5File::open(&path).unwrap();
    let want: Vec<i32> = (100..112).collect();
    for name in ["vds_unlim", "vds_late_unlim"] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.shape(), vec![3, 4], "{name}");
        assert_eq!(ds.read_raw::<i32>().unwrap(), want, "{name}");
    }
    drop(file);
    cleanup(&path);
}

/// The canon fields both orders share: virtual storage, one mapping over the
/// whole of `/src` in the same file, and the same fill value.
#[test]
fn the_late_order_carries_the_same_layout_and_mapping() {
    let path = write_temp();
    let file = H5File::open(&path).unwrap();
    for name in ["vds", "vds_late"] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(
            ds.storage_layout().unwrap(),
            StorageLayout::Virtual,
            "{name}"
        );
        let maps = ds.virtual_mappings().unwrap();
        assert_eq!(maps.len(), 1, "{name}");
        assert_eq!(maps[0].source_file_name, ".", "{name}");
        assert_eq!(maps[0].source_dset_name, "/src", "{name}");
        assert_eq!(maps[0].source_selection, Selection::All, "{name}");
        assert_eq!(
            ds.fill_value().unwrap(),
            FillValue::UserDefined((-9i32).to_le_bytes().to_vec()),
            "{name}"
        );
    }
    drop(file);
    cleanup(&path);
}

/// The one field that does differ. `H5Pset_virtual` alone leaves the
/// contiguous default in place, so the late variant declares `Late` where the
/// conventional one declares `Incremental`. Measured against libhdf5 1.14.6
/// through `H5Pget_alloc_time`: 3 (`H5D_ALLOC_TIME_INCR`) and 2
/// (`H5D_ALLOC_TIME_LATE`).
#[test]
fn the_late_order_declares_the_contiguous_default_allocation_time() {
    let path = write_temp();
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("vds").unwrap().alloc_time().unwrap(),
        AllocTime::Incr
    );
    assert_eq!(
        file.dataset("vds_late").unwrap().alloc_time().unwrap(),
        AllocTime::Late
    );
    assert_eq!(
        file.dataset("vds_unlim").unwrap().alloc_time().unwrap(),
        AllocTime::Incr
    );
    assert_eq!(
        file.dataset("vds_late_unlim")
            .unwrap()
            .alloc_time()
            .unwrap(),
        AllocTime::Late
    );
    drop(file);
    cleanup(&path);
}
