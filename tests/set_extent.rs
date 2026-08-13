//! Integration tests for `H5Dataset::set_extent`.
//!
//! Unlike `extend`, `set_extent` can shrink a chunked dataset's logical
//! dimensions — the way to correct an over-extended frame count after a
//! partial multi-frame chunk. A shrink prunes stored chunks the way
//! libhdf5's `H5D__chunk_prune_by_extent` does: chunks entirely beyond the
//! new extent are de-indexed and their storage freed, and chunks the new
//! extent cuts through get their out-of-extent region refilled with the
//! fill value.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_set_extent_{}_{}_{}",
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

/// `set_extent` shrinks the logical frame count; the reader returns exactly
/// the retained frames, and the written data within them is intact.
#[test]
fn set_extent_shrinks_logical_extent() {
    let path = unique_tmp("shrink");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0usize, 4])
            .chunk(&[1, 4])
            .max_shape(&[None, Some(4)])
            .create("data")
            .unwrap();

        for f in 0..5i32 {
            let row = [f * 10, f * 10 + 1, f * 10 + 2, f * 10 + 3];
            let raw: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            ds.write_chunk(f as usize, &raw).unwrap();
        }
        ds.extend(&[5, 4]).unwrap();

        // Correct the extent down to 3 frames.
        ds.set_extent(&[3, 4]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.shape(), vec![3, 4]);
        let v = ds.read_raw::<i32>().unwrap();
        assert_eq!(v.len(), 12);
        for f in 0..3i32 {
            for c in 0..4usize {
                assert_eq!(v[f as usize * 4 + c], f * 10 + c as i32);
            }
        }
    }

    cleanup(&path);
}

/// An extent change made in a reopen session with no chunk write survives
/// the close: the finalize path must rebuild the dataset header for it,
/// not just when chunks were written.
#[test]
fn set_extent_alone_survives_a_reopen_session() {
    let path = unique_tmp("bare_extent");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0usize, 4])
            .chunk(&[1, 4])
            .max_shape(&[None, Some(4)])
            .create("data")
            .unwrap();
        for f in 0..2i32 {
            let row = [f, f + 1, f + 2, f + 3];
            let raw: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            ds.write_chunk(f as usize, &raw).unwrap();
        }
        ds.extend(&[2, 4]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        let ds = file.dataset_writer("data").unwrap();
        ds.set_extent(&[4, 4]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.shape(), vec![4, 4]);
    }

    cleanup(&path);
}

/// Shrinking an extensible-array dataset prunes the stranded chunks, so
/// growing the extent back reads fill values — not the stale data. Twelve
/// chunks with `idx_blk_elmts = 4` puts the pruned entries in both the EA
/// index block and on-disk data blocks.
#[test]
fn ea_shrink_then_regrow_reads_fill_not_stale() {
    let path = unique_tmp("ea_regrow");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([24usize, 4])
            .chunk(&[2, 4])
            .max_shape(&[None, Some(4)])
            .create("data")
            .unwrap();
        let vals: Vec<i32> = (0..24 * 4).collect();
        ds.write_slice(&[0, 0], &[24, 4], &vals).unwrap();
        // Chunk 1 (rows 2..4) straddles the new extent; chunks 2..12 are
        // entirely beyond it.
        ds.set_extent(&[3, 4]).unwrap();
        ds.set_extent(&[24, 4]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.shape(), vec![24, 4]);
        let v = ds.read_raw::<i32>().unwrap();
        for r in 0..24usize {
            for c in 0..4usize {
                let expect = if r < 3 { (r * 4 + c) as i32 } else { 0 };
                assert_eq!(v[r * 4 + c], expect, "row {r} col {c}");
            }
        }
    }

    cleanup(&path);
}

/// The same prune-and-refill through the filtered extensible-array path:
/// stored chunk sizes vary, so the freed lengths come from the index
/// entries and the refilled straddler is re-compressed.
#[test]
fn ea_filtered_shrink_then_regrow_reads_fill() {
    let path = unique_tmp("ea_filt_regrow");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([24usize, 4])
            .chunk(&[2, 4])
            .max_shape(&[None, Some(4)])
            .deflate(4)
            .create("data")
            .unwrap();
        let vals: Vec<i32> = (0..24 * 4).map(|i| i / 4).collect();
        ds.write_slice(&[0, 0], &[24, 4], &vals).unwrap();
        ds.set_extent(&[3, 4]).unwrap();
        ds.set_extent(&[24, 4]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        let v = ds.read_raw::<i32>().unwrap();
        for r in 0..24usize {
            for c in 0..4usize {
                let expect = if r < 3 { r as i32 } else { 0 };
                assert_eq!(v[r * 4 + c], expect, "row {r} col {c}");
            }
        }
    }

    cleanup(&path);
}

/// A shrink frees the pruned chunks' storage for reuse: writing a second
/// dataset's chunks after the shrink consumes exactly the freed blocks, so
/// the file ends up as large as one that never wrote the pruned chunks.
/// (Both datasets' index blocks are created before the shrink, so the
/// post-shrink chunk writes are the only allocations left — an exact-size
/// reuse with no fragmentation.)
#[test]
fn fa_shrink_frees_chunk_storage_for_reuse() {
    let path1 = unique_tmp("fa_reuse_shrunk");
    let path2 = unique_tmp("fa_reuse_reference");

    // Shrunk file: A's chunks 1..4 (rows 2..8) are freed, then B's three
    // chunks reuse those blocks.
    {
        let file = H5File::create(&path1).unwrap();
        let a = file
            .new_dataset::<i32>()
            .shape([8usize, 4])
            .chunk(&[2, 4])
            .create("a")
            .unwrap();
        let vals: Vec<i32> = (0..8 * 4).collect();
        a.write_slice(&[0, 0], &[8, 4], &vals).unwrap();
        let b = file
            .new_dataset::<i32>()
            .shape([6usize, 4])
            .chunk(&[2, 4])
            .create("b")
            .unwrap();
        a.set_extent(&[2, 4]).unwrap();
        let vals: Vec<i32> = (0..6 * 4).collect();
        b.write_slice(&[0, 0], &[6, 4], &vals).unwrap();
        file.close().unwrap();
    }

    // Reference file: identical structure, but A never writes the chunks
    // the other file pruned, and B's chunks are fresh allocations.
    {
        let file = H5File::create(&path2).unwrap();
        let a = file
            .new_dataset::<i32>()
            .shape([8usize, 4])
            .chunk(&[2, 4])
            .create("a")
            .unwrap();
        let vals: Vec<i32> = (0..2 * 4).collect();
        a.write_slice(&[0, 0], &[2, 4], &vals).unwrap();
        let b = file
            .new_dataset::<i32>()
            .shape([6usize, 4])
            .chunk(&[2, 4])
            .create("b")
            .unwrap();
        let vals: Vec<i32> = (0..6 * 4).collect();
        b.write_slice(&[0, 0], &[6, 4], &vals).unwrap();
        file.close().unwrap();
    }

    let s1 = std::fs::metadata(&path1).unwrap().len();
    let s2 = std::fs::metadata(&path2).unwrap().len();
    assert_eq!(
        s1, s2,
        "shrunk-then-reused file should be as large as one that never wrote \
         the pruned chunks"
    );

    cleanup(&path1);
    cleanup(&path2);
}

/// Fixed-array prune-and-refill read back: pruned region reads fill after a
/// regrow, retained region is intact, straddlers in both dimensions.
#[test]
fn fa_shrink_then_regrow_reads_fill() {
    let path = unique_tmp("fa_regrow");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([6usize, 6])
            .chunk(&[2, 3])
            .create("data")
            .unwrap();
        let vals: Vec<i32> = (0..36).collect();
        ds.write_slice(&[0, 0], &[6, 6], &vals).unwrap();
        // [3, 4] cuts through chunks in both dimensions.
        ds.set_extent(&[3, 4]).unwrap();
        ds.set_extent(&[6, 6]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        let v = ds.read_raw::<i32>().unwrap();
        for r in 0..6usize {
            for c in 0..6usize {
                let expect = if r < 3 && c < 4 {
                    (r * 6 + c) as i32
                } else {
                    0
                };
                assert_eq!(v[r * 6 + c], expect, "row {r} col {c}");
            }
        }
    }

    cleanup(&path);
}

/// V2-B-tree prune-and-refill read back: records beyond the extent leave
/// the tree (the node pool releases the surplus at flush), straddlers are
/// refilled in both dimensions.
#[test]
fn bt2_shrink_then_regrow_reads_fill() {
    let path = unique_tmp("bt2_regrow");

    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4usize, 6])
            .chunk(&[2, 3])
            .max_shape(&[None, None])
            .create("data")
            .unwrap();
        let vals: Vec<i32> = (0..24).collect();
        ds.write_slice(&[0, 0], &[4, 6], &vals).unwrap();
        ds.set_extent(&[3, 4]).unwrap();
        ds.set_extent(&[4, 6]).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        let v = ds.read_raw::<i32>().unwrap();
        for r in 0..4usize {
            for c in 0..6usize {
                let expect = if r < 3 && c < 4 {
                    (r * 6 + c) as i32
                } else {
                    0
                };
                assert_eq!(v[r * 6 + c], expect, "row {r} col {c}");
            }
        }
    }

    cleanup(&path);
}

/// `set_extent` rejects an extent above the dataset's maximum dimensions.
#[test]
fn set_extent_rejects_exceeding_max() {
    let path = unique_tmp("overmax");
    let file = H5File::create(&path).unwrap();
    let ds = file
        .new_dataset::<i32>()
        .shape([0usize, 4])
        .chunk(&[1, 4])
        .max_shape(&[None, Some(4)])
        .create("data")
        .unwrap();

    // Dimension 1 is capped at 4; growing it to 9 must be rejected.
    let err = ds.set_extent(&[3, 9]).expect_err("should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("maximum"), "unexpected error: {msg}");

    drop(file);
    cleanup(&path);
}

// ---- vlen datasets: a shrink must release the stranded heap objects ------

/// Chunk-aligned shrink of a vlen dataset: every pruned chunk's strings are
/// global-heap objects, and freeing the chunk block without releasing them
/// strands one `H5HG_MINALLOC` (4096-byte) collection per append/shrink
/// cycle. With the release the cycle reuses the same blocks and the file
/// size settles.
#[test]
fn vlen_shrink_releases_the_pruned_chunks_heap_objects() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("vlen_prune_release_{cycles}"));
        let file = H5File::create(&path).unwrap();
        file.create_appendable_vlen_dataset("notes", 2, None)
            .unwrap();
        file.append_vlen_strings("notes", &["keep0", "keep1"])
            .unwrap();
        for i in 0..cycles {
            // One full chunk per cycle, so nothing stays in the append
            // buffer (a buffered append blocks `set_extent`).
            file.append_vlen_strings("notes", &[&format!("x{i}"), &format!("y{i}")])
                .unwrap();
            file.dataset_writer("notes")
                .unwrap()
                .set_extent(&[2])
                .unwrap();
        }
        file.close().unwrap();
        let n = std::fs::metadata(&path).unwrap().len();
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["keep0", "keep1"]
        );
        drop(read);
        cleanup(&path);
        n
    };

    let settled = size_after(3);
    assert_eq!(size_after(20), settled, "20 cycles against 3");
    assert_eq!(size_after(50), settled, "50 cycles against 3");
}

/// A shrink that cuts through a chunk refills its out-of-extent tail with
/// the fill value; the strings the refill overwrites must be released too,
/// or every cycle strands the replaced element's collection.
#[test]
fn vlen_shrink_releases_the_refilled_straddlers_heap_objects() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("vlen_straddle_release_{cycles}"));
        let file = H5File::create(&path).unwrap();
        file.create_appendable_vlen_dataset("notes", 2, None)
            .unwrap();
        file.append_vlen_strings("notes", &["keep0", "stub"])
            .unwrap();
        let ds = file.dataset_writer("notes").unwrap();
        ds.set_extent(&[1]).unwrap();
        for i in 0..cycles {
            // Regrow, write row 1 directly into the chunk (a slice write
            // flushes, it never buffers), shrink it back out: the shrink's
            // straddler refill is the only thing that can release `x{i}`.
            ds.set_extent(&[2]).unwrap();
            ds.write_vlen_strings_slice(1, &[&format!("x{i}")]).unwrap();
            ds.set_extent(&[1]).unwrap();
        }
        file.close().unwrap();
        let n = std::fs::metadata(&path).unwrap().len();
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["keep0"]
        );
        drop(read);
        cleanup(&path);
        n
    };

    let settled = size_after(3);
    assert_eq!(size_after(20), settled, "20 cycles against 3");
    assert_eq!(size_after(50), settled, "50 cycles against 3");
}

/// Survivor strings stay intact through a shrink and a regrown row reads
/// as the fill value — the nil reference, i.e. the empty string — for both
/// an unfiltered and a deflate-compressed vlen dataset (the compressed one
/// must decompress the pruned chunk before parsing its references).
#[test]
fn vlen_shrink_then_regrow_keeps_survivors_and_reads_empty() {
    for (tag, pipeline) in [
        ("plain", None),
        ("deflate", Some(rust_hdf5::FilterPipeline::deflate(4))),
    ] {
        let path = unique_tmp(&format!("vlen_regrow_{tag}"));
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("notes", 2, pipeline)
                .unwrap();
            file.append_vlen_strings("notes", &["s0", "s1", "s2", "s3", "s4", "s5"])
                .unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            // Chunk 1 straddles (s3 refilled), chunk 2 is pruned (s4, s5).
            ds.set_extent(&[3]).unwrap();
            ds.set_extent(&[6]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["s0", "s1", "s2", "", "", ""],
            "{tag}"
        );
        drop(file);
        cleanup(&path);
    }
}
