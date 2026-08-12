//! Hyperslab writes into chunked datasets.
//!
//! One case per boundary of the chunk-decomposition rule: whole-chunk vs
//! partial coverage, first/last chunk of a span, edge chunks that hang past
//! the extent, never-written chunks, each of the three chunk index types, and
//! filtered storage where a partial write has to decompress and recompress.

use rust_hdf5::{H5Dataset, H5File};

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_chunk_slice_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// 4x6 i32 dataset in 2x3 chunks (a 2x2 chunk grid), created with `build`,
/// zero-filled by writing every chunk, then patched by `patch`. Returns what
/// the whole dataset reads back as.
fn round_trip(
    name: &str,
    build: impl Fn(&H5File) -> H5Dataset,
    patch: impl Fn(&H5Dataset),
) -> Vec<i32> {
    let path = tmp(name);
    {
        let file = H5File::create(&path).unwrap();
        let ds = build(&file);
        // Seed every element with its own index so any clobbering is visible.
        let seed: Vec<i32> = (0..24).collect();
        ds.write_slice(&[0, 0], &[4, 6], &seed).unwrap();
        patch(&ds);
        file.close().unwrap();
    }
    let out = {
        let file = H5File::open(&path).unwrap();
        file.dataset("d").unwrap().read_raw::<i32>().unwrap()
    };
    std::fs::remove_file(&path).ok();
    out
}

fn fixed_array(file: &H5File) -> H5Dataset {
    file.new_dataset::<i32>()
        .shape([4, 6])
        .chunk(&[2, 3])
        .create("d")
        .unwrap()
}

fn extensible_array(file: &H5File) -> H5Dataset {
    file.new_dataset::<i32>()
        .shape([4, 6])
        .chunk(&[2, 3])
        .max_shape(&[None, Some(6)])
        .create("d")
        .unwrap()
}

fn btree_v2(file: &H5File) -> H5Dataset {
    file.new_dataset::<i32>()
        .shape([4, 6])
        .chunk(&[2, 3])
        .max_shape(&[None, None])
        .create("d")
        .unwrap()
}

#[cfg(feature = "deflate")]
fn deflated(file: &H5File) -> H5Dataset {
    file.new_dataset::<i32>()
        .shape([4, 6])
        .chunk(&[2, 3])
        .max_shape(&[None, Some(6)])
        .deflate(6)
        .create("d")
        .unwrap()
}

/// A storage layout to run a case against: a label for assertion messages and
/// the constructor that produces a dataset with that layout.
type Layout = (&'static str, fn(&H5File) -> H5Dataset);

/// Every storage layout a 4x6/2x3 dataset can have, paired with a label for
/// assertion messages. Filtered storage only appears when the filter is built
/// in, so the suite still passes under `--no-default-features`.
fn layouts() -> Vec<Layout> {
    let mut cases: Vec<Layout> = vec![
        ("fa", fixed_array),
        ("ea", extensible_array),
        ("bt2", btree_v2),
    ];
    cases.extend(filtered_layout());
    cases
}

#[cfg(feature = "deflate")]
fn filtered_layout() -> Option<Layout> {
    Some(("deflate", deflated))
}

#[cfg(not(feature = "deflate"))]
fn filtered_layout() -> Option<Layout> {
    None
}

/// Expected contents of the seeded 4x6 grid after writing `values` into the
/// region at `starts` of size `counts`.
fn expected(starts: [usize; 2], counts: [usize; 2], values: &[i32]) -> Vec<i32> {
    let mut want: Vec<i32> = (0..24).collect();
    for r in 0..counts[0] {
        for c in 0..counts[1] {
            want[(starts[0] + r) * 6 + starts[1] + c] = values[r * counts[1] + c];
        }
    }
    want
}

/// A selection covering exactly one whole chunk needs no read-back at all.
#[test]
fn whole_chunk_selection_replaces_that_chunk_only() {
    let values: Vec<i32> = vec![-1, -2, -3, -4, -5, -6];
    for (label, build) in layouts() {
        // Chunk (1,1) is rows 2..4, cols 3..6.
        let got = round_trip(&format!("whole_{label}"), build, |ds| {
            ds.write_slice(&[2, 3], &[2, 3], &values).unwrap()
        });
        assert_eq!(got, expected([2, 3], [2, 3], &values), "index {label}");
    }
}

/// A selection inside one chunk must leave the rest of that chunk alone —
/// the read-modify-write path.
#[test]
fn partial_chunk_selection_preserves_the_rest_of_the_chunk() {
    let values: Vec<i32> = vec![-7, -8];
    for (label, build) in layouts() {
        // Row 1, cols 1..3: inside chunk (0,0), touching neither its first
        // row nor its first column.
        let got = round_trip(&format!("partial_{label}"), build, |ds| {
            ds.write_slice(&[1, 1], &[1, 2], &values).unwrap()
        });
        assert_eq!(got, expected([1, 1], [1, 2], &values), "index {label}");
    }
}

/// One row of the dataset crosses the chunk grid horizontally: partial in
/// every chunk it touches, untouched in the chunks it does not.
#[test]
fn row_update_spans_the_chunk_row_and_leaves_the_others() {
    let values: Vec<i32> = vec![100, 101, 102, 103, 104, 105];
    for (label, build) in layouts() {
        let got = round_trip(&format!("row_{label}"), build, |ds| {
            ds.write_slice(&[2, 0], &[1, 6], &values).unwrap()
        });
        assert_eq!(got, expected([2, 0], [1, 6], &values), "index {label}");
    }
}

/// A selection straddling all four chunks, partial in each of them.
#[test]
fn selection_straddling_every_chunk_boundary() {
    let values: Vec<i32> = vec![70, 71, 72, 73, 74, 75, 76, 77];
    for (label, build) in layouts() {
        // Rows 1..3, cols 2..6: crosses the row boundary at 2 and the column
        // boundary at 3, so all four chunks are partially covered.
        let got = round_trip(&format!("straddle_{label}"), build, |ds| {
            ds.write_slice(&[1, 2], &[2, 4], &values).unwrap()
        });
        assert_eq!(got, expected([1, 2], [2, 4], &values), "index {label}");
    }
}

/// Elements of a chunk that no write has ever touched read back as the
/// dataset's fill value, not as zeros left over from the buffer.
#[test]
fn untouched_elements_of_a_new_chunk_hold_the_fill_value() {
    let path = tmp("fill");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 6])
            .chunk(&[2, 3])
            .fill_value(-99)
            .create("d")
            .unwrap();
        // Only one element, in chunk (0,0). Nothing else is ever written.
        ds.write_slice(&[1, 1], &[1, 1], &[42i32]).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open(&path).unwrap();
        let got = file.dataset("d").unwrap().read_raw::<i32>().unwrap();
        let mut want = vec![-99i32; 24];
        want[6 + 1] = 42; // row 1, column 1
        assert_eq!(got, want);
    }
    std::fs::remove_file(&path).ok();
}

/// The chunk grid does not divide the extent evenly: the last chunk of each
/// axis hangs past the end of the dataset. Writing into it must not disturb
/// its in-extent neighbours, and the reader must still see the right shape.
#[test]
fn edge_chunks_that_hang_past_the_extent() {
    let path = tmp("edge");
    {
        let file = H5File::create(&path).unwrap();
        // 5x5 in 2x2 chunks: a 3x3 grid whose last row/column of chunks is
        // only half inside the dataset.
        let ds = file
            .new_dataset::<i32>()
            .shape([5, 5])
            .chunk(&[2, 2])
            .fill_value(0)
            .create("d")
            .unwrap();
        let seed: Vec<i32> = (0..25).collect();
        ds.write_slice(&[0, 0], &[5, 5], &seed).unwrap();
        // Bottom-right 1x1 corner: the only in-extent element of chunk (2,2).
        ds.write_slice(&[4, 4], &[1, 1], &[999i32]).unwrap();
        // Last row, all columns: crosses every chunk of the bottom grid row.
        ds.write_slice(&[4, 0], &[1, 4], &[900i32, 901, 902, 903])
            .unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape(), vec![5, 5]);
        let mut want: Vec<i32> = (0..25).collect();
        want[24] = 999;
        want[20..24].copy_from_slice(&[900, 901, 902, 903]);
        assert_eq!(ds.read_raw::<i32>().unwrap(), want);
    }
    std::fs::remove_file(&path).ok();
}

/// Three dimensions, with the middle axis crossing a chunk boundary.
#[test]
fn three_dimensional_selection() {
    let path = tmp("3d");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([2, 4, 3])
            .chunk(&[1, 2, 3])
            .create("d")
            .unwrap();
        let seed: Vec<i32> = (0..24).collect();
        ds.write_slice(&[0, 0, 0], &[2, 4, 3], &seed).unwrap();
        // Plane 1, rows 1..3 (crossing the chunk boundary at 2), cols 0..2.
        ds.write_slice(&[1, 1, 0], &[1, 2, 2], &[-1i32, -2, -3, -4])
            .unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open(&path).unwrap();
        let got = file.dataset("d").unwrap().read_raw::<i32>().unwrap();
        let mut want: Vec<i32> = (0..24).collect();
        // (1,1,0)=15 (1,1,1)=16 (1,2,0)=18 (1,2,1)=19
        want[15] = -1;
        want[16] = -2;
        want[18] = -3;
        want[19] = -4;
        assert_eq!(got, want);
    }
    std::fs::remove_file(&path).ok();
}

/// Repeating the same partial write must converge, not accumulate: the reread
/// of an already-patched chunk has to give back exactly what was stored.
#[test]
fn repeated_partial_writes_are_idempotent() {
    for (label, build) in layouts() {
        let got = round_trip(&format!("idem_{label}"), build, |ds| {
            for _ in 0..5 {
                ds.write_slice(&[1, 2], &[2, 2], &[-1i32, -2, -3, -4])
                    .unwrap();
            }
        });
        assert_eq!(
            got,
            expected([1, 2], [2, 2], &[-1, -2, -3, -4]),
            "index {label}"
        );
    }
}

/// Updating one row of a large chunked dataset must touch only the chunks
/// that row crosses — the whole point of the feature. With 1x16 chunks, a
/// one-row write to a 64x16 dataset should leave the other 63 chunks
/// unallocated, so the file stays far smaller than the full extent.
#[test]
fn a_row_update_allocates_only_the_chunks_it_touches() {
    let path = tmp("cost");
    let full_extent_bytes = 64 * 16 * 4;
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([64, 16])
            .chunk(&[1, 16])
            .max_shape(&[None, Some(16)])
            .create("d")
            .unwrap();
        ds.write_slice(&[7, 0], &[1, 16], &[5i32; 16]).unwrap();
        file.close().unwrap();
    }
    let size = std::fs::metadata(&path).unwrap().len();
    assert!(
        size < full_extent_bytes / 4,
        "one-row update wrote {size} bytes for a {full_extent_bytes}-byte extent"
    );
    {
        let file = H5File::open(&path).unwrap();
        let got = file.dataset("d").unwrap().read_raw::<i32>().unwrap();
        assert_eq!(&got[7 * 16..8 * 16], &[5i32; 16]);
        assert!(got[..7 * 16].iter().all(|&v| v == 0));
        assert!(got[8 * 16..].iter().all(|&v| v == 0));
    }
    std::fs::remove_file(&path).ok();
}

/// A selection outside the current extent is still rejected, for chunked
/// storage just as for contiguous.
#[test]
fn out_of_bounds_selection_is_rejected() {
    let path = tmp("oob");
    let file = H5File::create(&path).unwrap();
    let ds = fixed_array(&file);
    assert!(ds.write_slice(&[3, 0], &[2, 6], &[0i32; 12]).is_err());
    assert!(ds.write_slice(&[0, 4], &[4, 4], &[0i32; 16]).is_err());
    // Wrong data length for the selection.
    assert!(ds.write_slice(&[0, 0], &[2, 3], &[0i32; 5]).is_err());
    // An in-bounds selection still works.
    assert!(ds.write_slice(&[2, 3], &[2, 3], &[1i32; 6]).is_ok());
    std::fs::remove_file(&path).ok();
}
