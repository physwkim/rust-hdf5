//! A stored chunk that decodes to more or fewer bytes than its image is
//! refused, by a read and by a write that reads it back. libhdf5 drops the
//! surplus of a long chunk and keeps what a short one decoded; this crate
//! refuses both.
#![cfg(feature = "deflate")]

use rust_hdf5::format::messages::filter::apply_filters;
use rust_hdf5::{FilterPipeline, H5File};

const IMAGE: usize = 1024;

/// `n` bytes no two chunks of the dataset share.
fn bytes(n: usize, seed: u8) -> Vec<u8> {
    (0..n).map(|i| (i as u8).wrapping_mul(7) ^ seed).collect()
}

/// A chunk of `n` bytes as it is stored: deflated, or with the deflate
/// skipped (filter mask 1) and so stored as it is.
fn stored(n: usize, masked: bool) -> (Vec<u8>, u32) {
    if masked {
        (bytes(n, 1), 1)
    } else {
        (
            apply_filters(&FilterPipeline::deflate(6), &bytes(n, 1)).unwrap(),
            0,
        )
    }
}

/// Every way the crate decodes chunk 0 of a two-chunk deflate dataset whose
/// chunk 0 decodes to `n` bytes, with what each gave when it did not fail.
fn decodes(n: usize, masked: bool) -> Vec<String> {
    let path = std::env::temp_dir().join(format!(
        "rust_hdf5_chunk_image_length_{n}_{masked}_{}.h5",
        std::process::id()
    ));
    let (chunk, mask) = stored(n, masked);
    let good = bytes(IMAGE, 2);
    let mut got = Vec::new();
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<u8>()
            .shape([2 * IMAGE])
            .chunk(&[IMAGE])
            .deflate(6)
            .create("d")
            .unwrap();
        ds.write_chunk_raw(0, &chunk, mask).unwrap();
        ds.write_chunk(1, &good).unwrap();
        if ds.write_slice(&[0], &[16], &[9u8; 16]).is_ok() {
            got.push("write_slice over it".to_string());
        }
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    // The whole chunk decodes into the read's own output; part of it decodes
    // into an image of its own.
    if let Ok(v) = ds.read_slice::<u8>(&[0], &[IMAGE]) {
        got.push(format!("whole-chunk read: {:?}..", &v[..4]));
    }
    if let Ok(v) = ds.read_slice::<u8>(&[0], &[16]) {
        got.push(format!("partial read: {:?}..", &v[..4]));
    }
    assert_eq!(ds.read_slice::<u8>(&[IMAGE], &[IMAGE]).unwrap(), good);
    drop(ds);
    drop(file);
    std::fs::remove_file(&path).ok();
    got
}

#[test]
fn a_chunk_that_decodes_short_is_refused() {
    for masked in [false, true] {
        let got = decodes(IMAGE / 2, masked);
        assert!(got.is_empty(), "masked {masked}: {got:#?}");
    }
}

#[test]
fn a_chunk_that_decodes_long_is_refused() {
    for masked in [false, true] {
        let got = decodes(2 * IMAGE, masked);
        assert!(got.is_empty(), "masked {masked}: {got:#?}");
    }
}

#[test]
fn a_chunk_of_exactly_its_image_reads() {
    for masked in [false, true] {
        let got = decodes(IMAGE, masked);
        assert_eq!(got.len(), 3, "masked {masked}: {got:#?}");
    }
}

/// A shrink releases the references of the chunks it drops, read back
/// through the same decode: a dropped chunk decoding short of its image, or
/// stored with the deflate skipped and a surplus, fails the shrink instead
/// of releasing what its bytes happen to name.
#[test]
fn a_shrink_refuses_a_dropped_vlen_chunk_of_another_length() {
    use rust_hdf5::format::messages::filter::reverse_filters;
    for case in ["short", "long"] {
        let path = std::env::temp_dir().join(format!(
            "rust_hdf5_chunk_image_length_vlen_{case}_{}.h5",
            std::process::id()
        ));
        let pipeline = FilterPipeline::deflate(6);
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("notes", 2, Some(pipeline.clone()))
                .unwrap();
            file.append_vlen_strings("notes", &["a", "b", "c", "d"])
                .unwrap();
            file.close().unwrap();
        }
        let (raw, _) = H5File::open(&path)
            .unwrap()
            .dataset("notes")
            .unwrap()
            .read_chunk_raw_at(&[1])
            .unwrap();
        let image = reverse_filters(&pipeline, &raw).unwrap();
        let (chunk, mask) = if case == "short" {
            let half = &image[..image.len() / 2];
            (apply_filters(&pipeline, half).unwrap(), 0)
        } else {
            ([image.as_slice(), &image[..image.len() / 2]].concat(), 1)
        };
        let shrunk = {
            let file = H5File::open_rw(&path).unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            ds.write_chunk_raw(1, &chunk, mask).unwrap();
            let r = ds.set_extent(&[2]);
            drop(ds);
            file.close().unwrap();
            r
        };
        std::fs::remove_file(&path).ok();
        assert!(shrunk.is_err(), "{case}: {shrunk:?}");
    }
}
