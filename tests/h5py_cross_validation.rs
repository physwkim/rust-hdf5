//! Cross-validation against h5py / libhdf5.
//!
//! Each test writes a file with rust-hdf5's public API and reads it back with
//! h5py to confirm the bytes are standard-tool readable. The tests skip (pass)
//! when the pinned h5py interpreter is not present, so CI without h5py is green.

use rust_hdf5::types::VarLenUnicode;
use rust_hdf5::H5File;

const TEST_PYTHON: &str = "/Users/stevek/mamba/envs/bs2026.1/bin/python";

fn python() -> Option<&'static str> {
    if std::path::Path::new(TEST_PYTHON).exists() {
        Some(TEST_PYTHON)
    } else {
        eprintln!("skipping h5py cross-check: {TEST_PYTHON} not present");
        None
    }
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_xcheck_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

/// Run `body` (top-level python statements) with the file already opened as
/// `f` in read mode. A non-zero exit (e.g. a failed `assert`) fails the test;
/// python's traceback is inherited to the test's stderr.
fn read_back_with_h5py(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np, sys\nf = h5py.File(r'{}', 'r')\n{}",
        path.display(),
        body
    );
    let status = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python");
    assert!(
        status.success(),
        "h5py cross-check failed for {}",
        path.display()
    );
}

/// Run `body` (top-level python statements) with a fresh file opened as `f` in
/// write mode. The body should populate `f` and is followed by `f.close()`. A
/// non-zero exit fails the test.
fn write_with_h5py(py: &str, path: &std::path::Path, body: &str) {
    let script = format!(
        "import h5py, numpy as np, sys\nf = h5py.File(r'{}', 'w')\n{}\nf.close()\n",
        path.display(),
        body
    );
    let status = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python");
    assert!(status.success(), "h5py write failed for {}", path.display());
}

/// F1: a vlen-string dataset created via the group helper returns a handle, so
/// attributes attach to it; the dataset and its attributes are h5py-readable.
#[test]
fn f1_vlen_dataset_with_attrs_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("f1_vlen_attr");
    {
        let file = H5File::create(&path).unwrap();
        let grp = file.root_group().create_group("ch").unwrap();
        let ds = grp
            .write_vlen_strings("labels", &["a", "bb", "ccc"])
            .unwrap();
        ds.new_attr::<VarLenUnicode>()
            .shape(())
            .create("unit")
            .unwrap()
            .write_string("volt")
            .unwrap();
        let ds2 = grp.dataset_writer("labels").unwrap();
        ds2.new_attr::<VarLenUnicode>()
            .shape(())
            .create("desc")
            .unwrap()
            .write_string("channel labels")
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        // String attributes are stored as variable-length UTF-8 strings, so
        // h5py hands them back as Python `str` (not bytes). The dataset
        // elements of a vlen-string dataset still come back as bytes, so the
        // `dec` helper is kept only for those.
        "dec = lambda v: v.decode() if isinstance(v, bytes) else v\n\
         ds = f['ch/labels']\n\
         vals = [dec(x) for x in ds[...]]\n\
         assert vals == ['a', 'bb', 'ccc'], vals\n\
         assert isinstance(ds.attrs['unit'], str), type(ds.attrs['unit'])\n\
         assert ds.attrs['unit'] == 'volt', ds.attrs['unit']\n\
         assert isinstance(ds.attrs['desc'], str), type(ds.attrs['desc'])\n\
         assert ds.attrs['desc'] == 'channel labels', ds.attrs['desc']\n",
    );
    std::fs::remove_file(&path).ok();
}

/// F2: a runtime `CompoundType` of arbitrary size, written via the
/// `datatype()` override + `write_raw_bytes`, is read by h5py as a structured
/// array with the right field names and dtypes.
#[test]
fn f2_runtime_compound_readable_by_h5py() {
    use rust_hdf5::types::{CompoundType, H5Type};
    let Some(py) = python() else { return };
    let path = tmp("f2_compound");
    // 12-byte packed compound: {id: i32 @0, val: f64 @4}. No matching Rust
    // primitive carrier exists, so this exercises the carrier-agnostic path.
    let ct = CompoundType {
        members: vec![
            ("id".to_string(), i32::hdf5_type(), 0),
            ("val".to_string(), f64::hdf5_type(), 4),
        ],
        total_size: 12,
    };
    let recs: [(i32, f64); 3] = [(1, 2.5), (2, 3.5), (3, -4.0)];
    let mut bytes = Vec::new();
    for (id, val) in recs {
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<u8>()
            .datatype(ct.to_datatype())
            .shape([recs.len()])
            .create("records")
            .unwrap();
        ds.write_raw_bytes(&bytes).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['records']\n\
         assert ds.dtype.names == ('id', 'val'), ds.dtype\n\
         assert ds.dtype['id'] == np.dtype('<i4'), ds.dtype['id']\n\
         assert ds.dtype['val'] == np.dtype('<f8'), ds.dtype['val']\n\
         assert list(ds['id']) == [1, 2, 3], ds['id']\n\
         assert list(ds['val']) == [2.5, 3.5, -4.0], ds['val']\n",
    );
    std::fs::remove_file(&path).ok();
}

/// F3 (rust → h5py): a vlen byte-array dataset written by rust-hdf5 is read by
/// h5py as a vlen `uint8` dataset, element for element.
#[test]
fn f3_vlen_bytes_written_by_rust_read_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("f3_bytes_rw");
    let items: [&[u8]; 4] = [b"abc", b"", &[0u8, 1, 2, 255], b"hi"];
    {
        let file = H5File::create(&path).unwrap();
        file.write_vlen_bytes("blobs", &items).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['blobs']\n\
         assert h5py.check_vlen_dtype(ds.dtype) == np.uint8, ds.dtype\n\
         got = [list(int(b) for b in x) for x in ds[...]]\n\
         assert got == [[97, 98, 99], [], [0, 1, 2, 255], [104, 105]], got\n",
    );
    std::fs::remove_file(&path).ok();
}

/// F3 (h5py → rust): a vlen `uint8` dataset written by h5py is read back by
/// rust-hdf5's `read_vlen_bytes`, element for element.
#[test]
fn f3_vlen_bytes_written_by_h5py_read_by_rust() {
    let Some(py) = python() else { return };
    let path = tmp("f3_bytes_wr");
    write_with_h5py(
        py,
        &path,
        "dt = h5py.vlen_dtype(np.uint8)\n\
         ds = f.create_dataset('blobs', (3,), dtype=dt)\n\
         ds[0] = np.array([1, 2, 3], dtype=np.uint8)\n\
         ds[1] = np.array([], dtype=np.uint8)\n\
         ds[2] = np.array([255, 254], dtype=np.uint8)\n",
    );
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("blobs").unwrap();
    let got = ds.read_vlen_bytes().unwrap();
    let expected: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![], vec![255, 254]];
    assert_eq!(got, expected);
    drop(file);
    std::fs::remove_file(&path).ok();
}

/// C: string attributes written via the public setters are variable-length
/// UTF-8 strings. h5py must read them back as Python `str` (vlen dtype),
/// on a dataset, a group, and the root group.
#[test]
fn c_vlen_string_attrs_readable_as_str_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("c_vlen_attrs");
    {
        let file = H5File::create(&path).unwrap();
        // Root attribute via H5File::set_attr_string.
        file.set_attr_string("title", "experiment").unwrap();
        // Group attribute via Group::set_attr_string.
        let grp = file.root_group().create_group("entry").unwrap();
        grp.set_attr_string("NX_class", "NXentry").unwrap();
        // Dataset attribute via new_attr::<VarLenUnicode>().write_string.
        let ds = file.new_dataset::<i32>().shape([3]).create("data").unwrap();
        ds.write_raw(&[1, 2, 3]).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .shape(())
            .create("units")
            .unwrap()
            .write_string("volt")
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "assert isinstance(f.attrs['title'], str), type(f.attrs['title'])\n\
         assert f.attrs['title'] == 'experiment', f.attrs['title']\n\
         g = f['entry']\n\
         assert isinstance(g.attrs['NX_class'], str), type(g.attrs['NX_class'])\n\
         assert g.attrs['NX_class'] == 'NXentry', g.attrs['NX_class']\n\
         ds = f['data']\n\
         assert isinstance(ds.attrs['units'], str), type(ds.attrs['units'])\n\
         assert ds.attrs['units'] == 'volt', ds.attrs['units']\n\
         assert h5py.check_string_dtype(ds.attrs.get_id('units').dtype), 'units not vlen string'\n",
    );
    std::fs::remove_file(&path).ok();
}

/// Array attributes: variable-length string arrays and numeric arrays written
/// via the public setters must come back through h5py as 1-D arrays — `str`
/// elements for the vlen-string arrays and a numpy int array for the numeric
/// ones — on the root group, a sub-group, and a dataset.
#[test]
fn arr_string_and_numeric_array_attrs_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("arr_array_attrs");
    {
        let file = H5File::create(&path).unwrap();
        // Root: vlen-string array + numeric array.
        file.set_attr_string_array("names", &["alpha", "beta", "gamma"])
            .unwrap();
        file.set_attr_array_numeric("ids", &[10i32, 20, 30])
            .unwrap();
        // Group: vlen-string array + numeric array.
        let grp = file.root_group().create_group("grp").unwrap();
        grp.set_attr_string_array("labels", &["p", "qq"]).unwrap();
        grp.set_attr_array_numeric("counts", &[1i32, 2, 3, 4])
            .unwrap();
        // Dataset: vlen-string array attribute via the builder + shape([n]).
        let ds = file.new_dataset::<i32>().shape([3]).create("data").unwrap();
        ds.write_raw(&[1, 2, 3]).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .shape([2])
            .create("tags")
            .unwrap()
            .write_string_array(&["x", "y"])
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "names = f.attrs['names']\n\
         assert names.shape == (3,), names.shape\n\
         assert list(names) == ['alpha', 'beta', 'gamma'], list(names)\n\
         assert all(isinstance(x, str) for x in names), [type(x) for x in names]\n\
         assert h5py.check_string_dtype(f.attrs.get_id('names').dtype), 'names not vlen string'\n\
         ids = f.attrs['ids']\n\
         assert ids.dtype == np.dtype('<i4'), ids.dtype\n\
         assert ids.shape == (3,), ids.shape\n\
         assert list(ids) == [10, 20, 30], list(ids)\n\
         g = f['grp']\n\
         assert list(g.attrs['labels']) == ['p', 'qq'], list(g.attrs['labels'])\n\
         assert all(isinstance(x, str) for x in g.attrs['labels'])\n\
         assert g.attrs['counts'].dtype == np.dtype('<i4'), g.attrs['counts'].dtype\n\
         assert list(g.attrs['counts']) == [1, 2, 3, 4], list(g.attrs['counts'])\n\
         ds = f['data']\n\
         assert list(ds.attrs['tags']) == ['x', 'y'], list(ds.attrs['tags'])\n\
         assert all(isinstance(x, str) for x in ds.attrs['tags'])\n",
    );
    std::fs::remove_file(&path).ok();
}

/// N-D numeric array attributes (`set_attr_array_numeric_nd`): a 2x3 attribute
/// on the root and a 2x2x2 attribute on a group must come back from h5py as
/// numpy arrays with the exact multi-dimensional shape and row-major values.
#[test]
fn nd_numeric_array_attrs_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("nd_array_attrs");
    {
        let file = H5File::create(&path).unwrap();
        // Root: 2x3 row-major i32.
        file.set_attr_array_numeric_nd("mat", &[1i32, 2, 3, 4, 5, 6], &[2, 3])
            .unwrap();
        // Group: 2x2x2 row-major f64.
        let grp = file.root_group().create_group("grp").unwrap();
        grp.set_attr_array_numeric_nd(
            "cube",
            &[0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[2, 2, 2],
        )
        .unwrap();
        // Length/shape mismatch must error, not silently truncate.
        assert!(file
            .set_attr_array_numeric_nd("bad", &[1i32, 2, 3], &[2, 2])
            .is_err());
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "mat = f.attrs['mat']\n\
         assert mat.dtype == np.dtype('<i4'), mat.dtype\n\
         assert mat.shape == (2, 3), mat.shape\n\
         assert mat.tolist() == [[1, 2, 3], [4, 5, 6]], mat.tolist()\n\
         cube = f['grp'].attrs['cube']\n\
         assert cube.dtype == np.dtype('<f8'), cube.dtype\n\
         assert cube.shape == (2, 2, 2), cube.shape\n\
         assert cube.tolist() == [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], cube.tolist()\n",
    );
    std::fs::remove_file(&path).ok();
}

/// N-D variable-length string array attributes (`set_attr_string_array_nd` and
/// the `AttrBuilder::shape` + `write_string_array` path): a 2x3 attribute on the
/// root, a 2x2 attribute on a group, and a 2x2 attribute on a dataset must come
/// back from h5py as vlen-string numpy arrays with the exact multi-dimensional
/// shape, row-major `str` values, and a vlen-string dtype.
#[test]
fn nd_string_array_attrs_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("nd_string_array_attrs");
    {
        let file = H5File::create(&path).unwrap();
        // Root: 2x3 row-major vlen strings.
        file.set_attr_string_array_nd("grid", &["a", "b", "c", "d", "e", "f"], &[2, 3])
            .unwrap();
        // Group: 2x2 row-major vlen strings.
        let grp = file.root_group().create_group("grp").unwrap();
        grp.set_attr_string_array_nd("cell", &["p", "qq", "rrr", "s"], &[2, 2])
            .unwrap();
        // Dataset: 2x2 vlen-string attribute via the builder + shape([2, 2]).
        let ds = file.new_dataset::<i32>().shape([2]).create("data").unwrap();
        ds.write_raw(&[1, 2]).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .shape([2, 2])
            .create("tags")
            .unwrap()
            .write_string_array(&["w", "xx", "yyy", "z"])
            .unwrap();
        // Length/shape mismatch must error, not silently truncate.
        assert!(file
            .set_attr_string_array_nd("bad", &["a", "b", "c"], &[2, 2])
            .is_err());
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "grid = f.attrs['grid']\n\
         assert grid.shape == (2, 3), grid.shape\n\
         assert grid.tolist() == [['a', 'b', 'c'], ['d', 'e', 'f']], grid.tolist()\n\
         assert h5py.check_string_dtype(f.attrs.get_id('grid').dtype), 'grid not vlen string'\n\
         cell = f['grp'].attrs['cell']\n\
         assert cell.shape == (2, 2), cell.shape\n\
         assert cell.tolist() == [['p', 'qq'], ['rrr', 's']], cell.tolist()\n\
         tags = f['data'].attrs['tags']\n\
         assert tags.shape == (2, 2), tags.shape\n\
         assert tags.tolist() == [['w', 'xx'], ['yyy', 'z']], tags.tolist()\n\
         assert h5py.check_string_dtype(f['data'].attrs.get_id('tags').dtype), 'tags not vlen string'\n",
    );
    std::fs::remove_file(&path).ok();
}

/// A: a filter requested without explicit chunk dimensions auto-chunks (whole
/// dataset = one chunk) and `write_raw` populates it. h5py must see a gzip-
/// compressed, chunked dataset with the right values — proving the filter is
/// honored, not silently dropped on a contiguous layout.
#[cfg(feature = "deflate")]
#[test]
fn a_autochunk_deflate_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("a_autochunk");
    let data: Vec<i32> = (0..8).collect();
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .deflate(6)
            .shape([8])
            .create("seq")
            .unwrap();
        ds.write_raw(&data).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['seq']\n\
         assert ds.compression == 'gzip', ds.compression\n\
         assert ds.chunks == (8,), ds.chunks\n\
         assert list(ds[...]) == list(range(8)), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// B: `write_raw` on an explicitly chunked + compressed 2-D dataset scatters
/// the full row-major image across a multi-chunk grid with edge chunks
/// (7/3 and 5/2). h5py must reassemble the exact array and see gzip+chunks.
#[cfg(feature = "deflate")]
#[test]
fn b_multichunk_deflate_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("b_multichunk");
    let data: Vec<i32> = (0..35).collect(); // 7 x 5 row-major
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([7, 5])
            .chunk(&[3, 2])
            .deflate(4)
            .create("grid")
            .unwrap();
        ds.write_raw(&data).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.compression == 'gzip', ds.compression\n\
         assert ds.chunks == (3, 2), ds.chunks\n\
         assert ds.shape == (7, 5), ds.shape\n\
         assert np.array_equal(ds[...], np.arange(35).reshape(7, 5)), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// CS1: a hyperslab written into a chunked dataset must be readable by
/// libhdf5 — including the chunks the selection only partially covers and the
/// chunks it never touches, which must come back as the fill value.
#[test]
fn cs1_chunked_write_slice_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("cs1_chunk_slice");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([6, 8])
            .chunk(&[2, 3])
            .fill_value(-5)
            .create("grid")
            .unwrap();
        // Rows 1..4 x cols 2..7: crosses both chunk-row boundaries and all
        // three chunk columns, partial in every chunk it touches.
        let patch: Vec<i32> = (100..115).collect();
        ds.write_slice(&[1, 2], &[3, 5], &patch).unwrap();
        // One element in the far corner chunk, which nothing else touches.
        ds.write_slice(&[5, 7], &[1, 1], &[777i32]).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.chunks == (2, 3), ds.chunks\n\
         assert ds.shape == (6, 8), ds.shape\n\
         want = np.full((6, 8), -5, dtype='i4')\n\
         want[1:4, 2:7] = np.arange(100, 115).reshape(3, 5)\n\
         want[5, 7] = 777\n\
         assert np.array_equal(ds[...], want), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// CS2: patching part of a *filtered* chunk means decompressing it, editing
/// it, and recompressing. The rewritten chunk is deliberately made harder to
/// compress than the original, so it no longer fits its old file block and
/// has to be relocated — libhdf5 must still find and decode it.
#[cfg(feature = "deflate")]
#[test]
fn cs2_filtered_chunked_write_slice_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("cs2_filtered_slice");
    // Incompressible patch: a scrambled sequence, not a run of small ints.
    let noise: Vec<i32> = (0..4u32)
        .map(|i| (i.wrapping_mul(0x9e37_79b9) ^ 0x5bd1_e995) as i32)
        .collect();
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 6])
            .chunk(&[2, 3])
            .deflate(6)
            .create("grid")
            .unwrap();
        // Highly compressible seed: every chunk is a short deflate stream.
        ds.write_slice(&[0, 0], &[4, 6], &[7i32; 24]).unwrap();
        // 2x2 patch straddling the column boundary at 3, so both chunks of
        // the top chunk-row are read, modified and recompressed.
        ds.write_slice(&[0, 2], &[2, 2], &noise).unwrap();
        file.close().unwrap();
    }
    let noise_py = noise
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    read_back_with_h5py(
        py,
        &path,
        &format!(
            "ds = f['grid']\n\
             assert ds.compression == 'gzip', ds.compression\n\
             assert ds.chunks == (2, 3), ds.chunks\n\
             want = np.full((4, 6), 7, dtype='i4')\n\
             want[0:2, 2:4] = np.array([{noise_py}], dtype='i4').reshape(2, 2)\n\
             assert np.array_equal(ds[...], want), ds[...]\n"
        ),
    );
    std::fs::remove_file(&path).ok();
}
