//! Writer-side global-heap packing (libhdf5's CWFS list, `H5HG_insert`).
//!
//! Every vlen write call used to allocate its own collection, so a file of
//! small strings paid the 4096-byte `H5HG_MINALLOC` minimum per call — and
//! a batch of more than 65535 strings was a hard error, because one call
//! meant one collection and a collection's object index is 16-bit. Now
//! `insert_vlen_objects` packs into collections with free space and spills
//! into a fresh collection at the index cap.
//!
//! The oracle for sharing is the number of `GCOL` signatures in the file:
//! these tests never free a whole collection block, so no stale signature
//! can inflate the count, and none of the written strings contain the
//! bytes `GCOL`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_vlen_packing_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

fn gcol_count(path: &PathBuf) -> usize {
    let bytes = std::fs::read(path).unwrap();
    bytes.windows(4).filter(|w| *w == b"GCOL").count()
}

/// Six string attributes used to mean six 4096-byte collections; they all
/// fit one.
#[test]
fn small_attributes_share_one_collection() {
    let path = unique_tmp("attrs");
    {
        let file = H5File::create(&path).unwrap();
        let g = file.root_group().create_group("entry").unwrap();
        for (name, value) in [
            ("NX_class", "NXentry"),
            ("definition", "NXtomo"),
            ("title", "sample scan"),
            ("operator", "kim"),
            ("start_time", "2026-08-13T09:00:00"),
            ("end_time", "2026-08-13T09:20:00"),
        ] {
            g.set_attr_string(name, value).unwrap();
        }
        file.close().unwrap();
    }

    assert_eq!(gcol_count(&path), 1, "six small attrs, one collection");
    let file = H5File::open(&path).unwrap();
    let g = file.root_group().group("entry").unwrap();
    assert_eq!(g.attr_string("NX_class").unwrap(), "NXentry");
    assert_eq!(g.attr_string("end_time").unwrap(), "2026-08-13T09:20:00");
    drop(file);
    cleanup(&path);
}

/// Separate write calls — two contiguous datasets and two append batches —
/// share one collection while it has room.
#[test]
fn small_vlen_datasets_share_one_collection() {
    let path = unique_tmp("datasets");
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("notes", &["alpha", "beta", "gamma"])
            .unwrap();
        file.write_vlen_strings("tags", &["red", "green"]).unwrap();
        file.create_appendable_vlen_dataset("log", 2, None).unwrap();
        file.append_vlen_strings("log", &["l0", "l1"]).unwrap();
        file.append_vlen_strings("log", &["l2", "l3"]).unwrap();
        file.close().unwrap();
    }

    assert_eq!(gcol_count(&path), 1, "four write calls, one collection");
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
        vec!["alpha", "beta", "gamma"]
    );
    assert_eq!(
        file.dataset("tags").unwrap().read_vlen_strings().unwrap(),
        vec!["red", "green"]
    );
    assert_eq!(
        file.dataset("log").unwrap().read_vlen_strings().unwrap(),
        vec!["l0", "l1", "l2", "l3"]
    );
    drop(file);
    cleanup(&path);
}

/// The exact fit rule: an object packs only when the collection's free
/// space holds it *plus* a residual free-space marker header (16 bytes
/// here). One byte of string past that boundary must start a fresh
/// collection, not corrupt the shared one.
///
/// Geometry (8-byte lengths): block 4096, header 16, marker header 16. A
/// 4000-byte string leaves free = 4096 - 16 - (16 + 4000) = 64. A 32-byte
/// string needs 16 + 32 = 48 ≤ 64 - 16 → fits; 33 bytes needs 16 + 40 =
/// 56 > 64 - 16 → fresh block.
#[test]
fn packing_respects_the_free_space_marker_boundary() {
    let filler = "x".repeat(4000);

    let fits = unique_tmp("fit");
    {
        let file = H5File::create(&fits).unwrap();
        file.write_vlen_strings("big", &[&filler]).unwrap();
        file.write_vlen_strings("small", &["y".repeat(32).as_str()])
            .unwrap();
        file.close().unwrap();
    }
    assert_eq!(gcol_count(&fits), 1, "32-byte string fits the 64-byte gap");

    let overflows = unique_tmp("overflow");
    {
        let file = H5File::create(&overflows).unwrap();
        file.write_vlen_strings("big", &[&filler]).unwrap();
        file.write_vlen_strings("small", &["y".repeat(33).as_str()])
            .unwrap();
        file.close().unwrap();
    }
    assert_eq!(gcol_count(&overflows), 2, "33-byte string must not fit");

    for path in [&fits, &overflows] {
        let file = H5File::open(path).unwrap();
        assert_eq!(
            file.dataset("big").unwrap().read_vlen_strings().unwrap(),
            vec![filler.clone()]
        );
        let small = &file.dataset("small").unwrap().read_vlen_strings().unwrap()[0];
        assert!(small.chars().all(|c| c == 'y'), "small got {small:?}");
        drop(file);
        cleanup(path);
    }
}

/// A batch past the 65535-object index cap spills into a second
/// collection instead of failing — the reporter's "collection is full"
/// error. Every element still reads back through its own reference.
#[test]
fn spilled_batch_past_the_index_cap_roundtrips() {
    let path = unique_tmp("spill");
    let n = 65537usize;
    let strings = vec!["x"; n];
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("bulk", &strings).unwrap();
        file.close().unwrap();
    }

    assert_eq!(gcol_count(&path), 2, "65537 objects need two collections");
    let file = H5File::open(&path).unwrap();
    let read = file.dataset("bulk").unwrap().read_vlen_strings().unwrap();
    assert_eq!(read.len(), n);
    assert!(read.iter().all(|s| s == "x"));
    drop(file);
    cleanup(&path);
}

/// Replacing elements releases the old objects into the collection's free
/// space and the replacements pack back into it: a replace loop settles
/// at one collection and one file size.
#[test]
fn replacing_elements_reuses_the_recovered_heap_space() {
    let size_after = |rounds: usize| {
        let path = unique_tmp(&format!("replace_{rounds}"));
        let file = H5File::create(&path).unwrap();
        let ds = file
            .write_vlen_strings("notes", &["one", "two", "three", "four"])
            .unwrap();
        for i in 0..rounds {
            ds.write_vlen_strings_slice(0, &[format!("round_{i}").as_str()])
                .unwrap();
        }
        file.close().unwrap();

        assert_eq!(gcol_count(&path), 1, "{rounds} rounds, one collection");
        let read = H5File::open(&path).unwrap();
        let got = read.dataset("notes").unwrap().read_vlen_strings().unwrap();
        assert_eq!(
            got,
            vec![
                format!("round_{}", rounds - 1),
                "two".into(),
                "three".into(),
                "four".into()
            ]
        );
        drop(read);
        let size = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        size
    };

    assert_eq!(size_after(20), size_after(2), "20 replace rounds against 2");
}

/// Deleting one dataset that shares a collection rewrites the block in
/// place — the survivor keeps its objects — and the recovered space takes
/// the next write.
#[test]
fn deleting_one_sharer_keeps_and_reuses_the_collection() {
    let path = unique_tmp("sharer");
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("doomed", &["d0", "d1", "d2"])
            .unwrap();
        file.write_vlen_strings("keep", &["k0", "k1", "k2"])
            .unwrap();
        file.delete_dataset("doomed").unwrap();
        file.write_vlen_strings("after", &["a0", "a1", "a2"])
            .unwrap();
        file.close().unwrap();
    }

    assert_eq!(
        gcol_count(&path),
        1,
        "delete rewrote in place, no new block"
    );
    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_vlen_strings().unwrap(),
        vec!["k0", "k1", "k2"]
    );
    assert_eq!(
        file.dataset("after").unwrap().read_vlen_strings().unwrap(),
        vec!["a0", "a1", "a2"]
    );
    assert!(file.dataset("doomed").is_err());
    drop(file);
    cleanup(&path);
}
