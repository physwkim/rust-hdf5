//! Deleting a dataset or group must free the file space it owned — chunk
//! blocks, chunk-index structures, contiguous data, the global-heap
//! objects of variable-length values, and (on reopened files) the on-disk
//! object header — the way libhdf5's `H5O_delete` does. It used to only
//! unlink: every create/delete cycle grew the file by the object's whole
//! footprint, and finalize even wrote a fresh orphan header for each
//! deleted object.
//!
//! The oracle is a settled file size: with reclamation, repeating a
//! create/delete cycle reuses the freed blocks, so many cycles end at the
//! same file size as few. (Freed space is reused, not returned — a byte
//! scan can still see stale block contents, so size is the observable.)

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_delete_reclaim_{}_{}_{}",
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

/// Contiguous vlen dataset: each delete must release the strings' heap
/// objects and the reference block, or every cycle leaks a collection
/// plus the data block.
#[test]
fn deleted_contiguous_vlen_dataset_frees_its_heap_and_data() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("contig_vlen_{cycles}"));
        let file = H5File::create(&path).unwrap();
        for _ in 0..cycles {
            file.write_vlen_strings("notes", &["alpha", "beta", "gamma"])
                .unwrap();
            file.delete_dataset("notes").unwrap();
        }
        // Recreate once more and keep it: the delete must not have broken
        // the name for reuse, and the survivor must read back.
        file.write_vlen_strings("notes", &["alpha", "beta", "gamma"])
            .unwrap();
        file.close().unwrap();
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["alpha", "beta", "gamma"]
        );
        drop(read);
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(20), size_after(2), "20 delete cycles against 2");
}

/// EA-chunked vlen dataset, plain and compressed: the delete walks every
/// chunk, releases its heap objects, frees the chunk blocks, and frees
/// the extensible-array index structures themselves.
#[test]
fn deleted_chunked_vlen_dataset_frees_chunks_heap_and_index() {
    for (tag, pipeline) in [
        ("plain", None),
        ("deflate", Some(rust_hdf5::FilterPipeline::deflate(4))),
    ] {
        let size_after = |cycles: usize| {
            let path = unique_tmp(&format!("chunked_vlen_{tag}_{cycles}"));
            let file = H5File::create(&path).unwrap();
            for c in 0..cycles {
                file.create_appendable_vlen_dataset("log", 2, pipeline.clone())
                    .unwrap();
                for i in 0..3 {
                    file.append_vlen_strings("log", &[&format!("a{c}_{i}"), &format!("b{c}_{i}")])
                        .unwrap();
                }
                file.delete_dataset("log").unwrap();
            }
            file.write_vlen_strings("keep", &["survivor"]).unwrap();
            file.close().unwrap();
            let read = H5File::open(&path).unwrap();
            assert_eq!(
                read.dataset("keep").unwrap().read_vlen_strings().unwrap(),
                vec!["survivor"]
            );
            drop(read);
            let n = std::fs::metadata(&path).unwrap().len();
            cleanup(&path);
            n
        };

        assert_eq!(
            size_after(20),
            size_after(2),
            "{tag}: 20 delete cycles against 2"
        );
    }
}

/// Fixed-array-indexed dataset (bounded max shape): the delete frees the
/// chunk blocks plus the FA header and data block.
#[test]
fn deleted_fixed_array_dataset_frees_chunks_and_index() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("fa_{cycles}"));
        let file = H5File::create(&path).unwrap();
        let row: Vec<u8> = (0..4i32).flat_map(|v| v.to_le_bytes()).collect();
        for _ in 0..cycles {
            let ds = file
                .new_dataset::<i32>()
                .shape([4usize, 4])
                .chunk(&[1, 4])
                .max_shape(&[Some(8), Some(4)])
                .create("grid")
                .unwrap();
            for f in 0..4 {
                ds.write_chunk(f, &row).unwrap();
            }
            file.delete_dataset("grid").unwrap();
        }
        file.close().unwrap();
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(20), size_after(2), "20 delete cycles against 2");
}

/// v2-B-tree-indexed dataset (two unlimited dimensions): the delete frees
/// the chunk blocks plus the BT2 header (and any flushed node blocks).
#[test]
fn deleted_btree_v2_dataset_frees_chunks_and_header() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("bt2_{cycles}"));
        let file = H5File::create(&path).unwrap();
        let tile: Vec<u8> = (0..4i32).flat_map(|v| v.to_le_bytes()).collect();
        for _ in 0..cycles {
            let ds = file
                .new_dataset::<i32>()
                .shape([4usize, 4])
                .chunk(&[2, 2])
                .max_shape(&[None, None])
                .create("tiles")
                .unwrap();
            for r in 0..2usize {
                for c in 0..2usize {
                    ds.write_chunk_at(&[r, c], &tile).unwrap();
                }
            }
            file.delete_dataset("tiles").unwrap();
        }
        file.close().unwrap();
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(20), size_after(2), "20 delete cycles against 2");
}

/// Deleting a group releases its attributes' heap objects and every child
/// object's storage.
#[test]
fn deleted_group_frees_attr_heap_and_child_storage() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("group_{cycles}"));
        let file = H5File::create(&path).unwrap();
        for _ in 0..cycles {
            let g = file.root_group().create_group("run").unwrap();
            g.set_attr_string("NX_class", "NXentry").unwrap();
            g.new_dataset::<f32>().shape([64]).create("temp").unwrap();
            file.delete_group("run").unwrap();
        }
        file.close().unwrap();
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(20), size_after(2), "20 delete cycles against 2");
}

/// Freeing the deleted dataset's storage must not touch its neighbors:
/// the survivors' strings and data still read back after a delete in the
/// same session.
#[test]
fn deleting_one_dataset_keeps_the_others_readable() {
    let path = unique_tmp("survivors");
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_strings("keep", &["k0", "k1"]).unwrap();
        file.write_vlen_strings("drop", &["d0", "d1"]).unwrap();
        file.create_appendable_vlen_dataset("log", 2, None).unwrap();
        file.append_vlen_strings("log", &["l0", "l1"]).unwrap();
        file.delete_dataset("drop").unwrap();
        assert!(
            file.delete_dataset("drop").is_err(),
            "a deleted name must not resolve again"
        );
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_vlen_strings().unwrap(),
        vec!["k0", "k1"]
    );
    assert_eq!(
        file.dataset("log").unwrap().read_vlen_strings().unwrap(),
        vec!["l0", "l1"]
    );
    assert!(file.dataset("drop").is_err());
    drop(file);
    cleanup(&path);
}

/// A delete in a reopen session frees storage that was parsed back off
/// the disk — the previous session's data blocks, heap objects, and
/// object header block — so delete/recreate sessions settle instead of
/// growing the file every time.
#[test]
fn reopen_session_delete_frees_the_previous_sessions_storage() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("reopen_{cycles}"));
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("notes", &["alpha", "beta"])
                .unwrap();
            file.close().unwrap();
        }
        for _ in 0..cycles {
            let file = H5File::options().no_locking().open_rw(&path).unwrap();
            file.delete_dataset("notes").unwrap();
            file.write_vlen_strings("notes", &["alpha", "beta"])
                .unwrap();
            file.close().unwrap();
        }
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["alpha", "beta"]
        );
        drop(read);
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(10), size_after(2), "10 reopen cycles against 2");
}
