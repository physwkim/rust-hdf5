//! Cross-validation against h5py / libhdf5.
//!
//! Each test writes a file with rust-hdf5's public API and reads it back with
//! h5py to confirm the bytes are standard-tool readable. The interpreter comes
//! from `RUST_HDF5_TEST_PYTHON`, falling back to the pinned path; the tests
//! skip (pass) when neither is present, so CI without h5py is green.

use rust_hdf5::types::VarLenUnicode;
use rust_hdf5::H5File;

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset. The
/// second is the same environment the parity oracle pins
/// (`oracle/run.py::DEFAULT_PYTHON`), so a checkout that can run the oracle can
/// run these cross-checks too.
const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn python() -> Option<&'static str> {
    static PY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PY.get_or_init(|| {
        let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
            Ok(p) => vec![p],
            Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
        };
        let found = candidates
            .iter()
            .find(|c| std::path::Path::new(c).exists())
            .cloned();
        if found.is_none() {
            eprintln!("skipping h5py cross-check: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
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
    write_with_h5py_libver(py, path, None, body);
}

/// `write_with_h5py`, pinning the library version bounds (e.g. `"v108"`).
///
/// Needed for anything that depends on a version-2 object header: dense
/// attribute storage only exists there, and h5py's default lower bound is
/// `earliest`, which pins the header to version 1.
fn write_with_h5py_libver(
    py: &str,
    path: &std::path::Path,
    libver: Option<&str>,
    body: &str,
) -> String {
    let libver_arg = match libver {
        Some(v) => format!(", libver=('{v}', '{v}')"),
        None => String::new(),
    };
    let script = format!(
        "import h5py, numpy as np, sys\nf = h5py.File(r'{}', 'w'{})\n{}\nf.close()\n",
        path.display(),
        libver_arg,
        body
    );
    let out = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .output()
        .expect("failed to spawn python");
    assert!(
        out.status.success(),
        "h5py write failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
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

/// BT2-1: a v2-B-tree index is searched by bisection, so its leaf records must
/// be ordered by scaled offsets. Write the chunk grid in reverse order — the
/// order a caller is free to use — and require libhdf5 to still find every
/// chunk. With records left in insertion order, h5py reads back mostly fill.
#[test]
fn bt2_1_out_of_order_chunk_writes_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("bt2_order");
    let chunk = |vals: [i32; 6]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 6])
            .chunk(&[2, 3])
            .max_shape(&[None, None])
            .create("grid")
            .unwrap();
        // Reverse grid order: (1,1), (1,0), (0,1), (0,0).
        ds.write_chunk_at(&[1, 1], &chunk([15, 16, 17, 21, 22, 23]))
            .unwrap();
        ds.write_chunk_at(&[1, 0], &chunk([12, 13, 14, 18, 19, 20]))
            .unwrap();
        ds.write_chunk_at(&[0, 1], &chunk([3, 4, 5, 9, 10, 11]))
            .unwrap();
        ds.write_chunk_at(&[0, 0], &chunk([0, 1, 2, 6, 7, 8]))
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.maxshape == (None, None), ds.maxshape\n\
         assert ds.chunks == (2, 3), ds.chunks\n\
         assert np.array_equal(ds[...], np.arange(24).reshape(4, 6)), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// BT2-2: a compressed dataset with two unlimited dimensions — a v2 B-tree
/// index whose records are type 11, carrying each chunk's stored size and
/// filter mask. libhdf5 recomputes that size field's width from the layout
/// version rather than reading it off the record, so this also pins that our
/// version-4 layout and our record width agree with what it expects.
#[cfg(feature = "deflate")]
#[test]
fn bt2_2_compressed_multi_unlimited_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("bt2_filtered");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([6, 8])
            .chunk(&[2, 4])
            .max_shape(&[None, None])
            .deflate(6)
            .create("grid")
            .unwrap();
        ds.write_slice(&[0, 0], &[6, 8], &[7i32; 48]).unwrap();
        // Patching part of one chunk recompresses it to a different size, so
        // the chunk moves and the index must record the new size and address.
        ds.write_slice(&[1, 1], &[2, 2], &[1i32, 2, 3, 4]).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.compression == 'gzip', ds.compression\n\
         assert ds.maxshape == (None, None), ds.maxshape\n\
         assert ds.chunks == (2, 4), ds.chunks\n\
         want = np.full((6, 8), 7, dtype='i4')\n\
         want[1, 1] = 1; want[1, 2] = 2; want[2, 1] = 3; want[2, 2] = 4\n\
         assert np.array_equal(ds[...], want), ds[...]\n\
         # Every chunk must be stored compressed, i.e. smaller than 2*4*4 = 32 B.\n\
         sizes = [ds.id.get_chunk_info(i).size for i in range(ds.id.get_num_chunks())]\n\
         assert len(sizes) == 6, sizes\n\
         assert all(s < 32 for s in sizes), sizes\n",
    );
    std::fs::remove_file(&path).ok();
}

/// BT2-5: a direct chunk write on a v2 B-tree — bytes stored verbatim with a
/// per-chunk filter mask. libhdf5 must skip exactly the filters the mask marks
/// as not applied, so one chunk arrives deflated (mask 0) and its neighbour
/// uncompressed (mask 1) and both must read back as the same values.
#[cfg(feature = "deflate")]
#[test]
fn bt2_5_direct_chunk_write_mask_honored_by_h5py() {
    use flate2::{write::ZlibEncoder, Compression};
    use std::io::Write;

    let Some(py) = python() else { return };
    let path = tmp("bt2_direct");
    let bytes = |vals: [i32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let deflate = |raw: &[u8]| -> Vec<u8> {
        let mut e = ZlibEncoder::new(Vec::new(), Compression::new(6));
        e.write_all(raw).unwrap();
        e.finish().unwrap()
    };
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0, 0])
            .chunk(&[2, 2])
            .max_shape(&[None, None])
            .deflate(6)
            .create("grid")
            .unwrap();
        ds.write_chunk_raw_at(&[0, 0], &deflate(&bytes([0, 1, 4, 5])), 0)
            .unwrap();
        // Same values, handed over uncompressed with filter 0 marked skipped.
        ds.write_chunk_raw_at(&[0, 1], &bytes([2, 3, 6, 7]), 1)
            .unwrap();
        ds.write_chunk_raw_at(&[1, 0], &deflate(&bytes([8, 9, 12, 13])), 0)
            .unwrap();
        ds.write_chunk_raw_at(&[1, 1], &bytes([10, 11, 14, 15]), 1)
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.compression == 'gzip', ds.compression\n\
         assert ds.chunks == (2, 2), ds.chunks\n\
         assert np.array_equal(ds[...], np.arange(16).reshape(4, 4)), ds[...]\n\
         # The mask-1 chunks are stored raw (16 B); the mask-0 ones went through\n\
         # deflate, so their filter_mask differs.\n\
         masks = sorted(ds.id.get_chunk_info(i).filter_mask\n\
                        for i in range(ds.id.get_num_chunks()))\n\
         assert masks == [0, 0, 1, 1], masks\n",
    );
    std::fs::remove_file(&path).ok();
}

/// BT2-3: more chunks than a 2048-byte node holds, so the index must be a real
/// tree — a root of separator records over several leaves — not one oversized
/// node. libhdf5 sizes every node from the header's `node_size` and descends
/// through the child pointers, so a flat index would either read past a node or
/// lose every record beyond the first.
#[test]
fn bt2_3_multi_node_tree_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("bt2_multinode");
    // 24-byte records, so a leaf holds (2048 - 10) / 24 = 84. A 20x20 extent in
    // 2x2 chunks is 100 of them.
    let data: Vec<i32> = (0..400).collect();
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([20, 20])
            .chunk(&[2, 2])
            .max_shape(&[None, None])
            .create("grid")
            .unwrap();
        ds.write_slice(&[0, 0], &[20, 20], &data).unwrap();
        file.close().unwrap();
    }
    // Our own reader must agree before h5py is consulted.
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("grid").unwrap();
        assert_eq!(ds.read_raw::<i32>().unwrap(), data);
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.chunks == (2, 2), ds.chunks\n\
         assert ds.id.get_num_chunks() == 100, ds.id.get_num_chunks()\n\
         assert np.array_equal(ds[...], np.arange(400).reshape(20, 20)), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// BT2-4: past 5269 records a depth-1 tree is full too, so the root becomes an
/// internal node over internal nodes. Depth 2 is the first shape where a child
/// pointer carries a subtree total (`child_total_nrecords`), whose width comes
/// from the geometry rather than the record — get it wrong and libhdf5
/// misparses every pointer after the first.
#[test]
fn bt2_4_depth_two_tree_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("bt2_depth2");
    // 1x1 chunks: 73 * 73 = 5329 records, past the 61 + 62 * 84 = 5269 a
    // depth-1 tree holds.
    let n = 73usize;
    let data: Vec<i32> = (0..(n * n) as i32).collect();
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([n, n])
            .chunk(&[1, 1])
            .max_shape(&[None, None])
            .create("grid")
            .unwrap();
        ds.write_slice(&[0, 0], &[n, n], &data).unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("grid").unwrap();
        assert_eq!(ds.read_raw::<i32>().unwrap(), data);
    }
    read_back_with_h5py(
        py,
        &path,
        &format!(
            "ds = f['grid']\n\
             assert ds.chunks == (1, 1), ds.chunks\n\
             assert ds.id.get_num_chunks() == {}, ds.id.get_num_chunks()\n\
             assert np.array_equal(ds[...], np.arange({}).reshape({n}, {n})), ds[...]\n",
            n * n,
            n * n
        ),
    );
    std::fs::remove_file(&path).ok();
}

/// A fixed-array dataset with a finite maximum above its shape: the array is
/// sized from the maximum's chunk grid and slots are row-major in that grid
/// (libhdf5 `max_nchunks` / `max_down_chunks`), including a maximum that
/// grows a non-leading dimension — the case where the index grid and the
/// current-extent grid disagree. h5py must see the maxshape and read every
/// value written before and after the extends.
#[test]
fn fa_growable_max_shape_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("fa_growable_max");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 3])
            .chunk(&[2, 3])
            .max_shape(&[Some(6), Some(9)])
            .create("grid")
            .unwrap();
        ds.write_raw(&(0..12).collect::<Vec<i32>>()).unwrap();
        ds.extend(&[6, 6]).unwrap();
        ds.write_slice(&[0, 3], &[6, 3], &(100..118).collect::<Vec<i32>>())
            .unwrap();
        ds.write_slice(&[4, 0], &[2, 3], &(200..206).collect::<Vec<i32>>())
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['grid']\n\
         assert ds.shape == (6, 6), ds.shape\n\
         assert ds.maxshape == (6, 9), ds.maxshape\n\
         assert ds.chunks == (2, 3), ds.chunks\n\
         expect = np.zeros((6, 6), dtype=np.int32)\n\
         expect[:4, :3] = np.arange(12).reshape(4, 3)\n\
         expect[:, 3:6] = np.arange(100, 118).reshape(6, 3)\n\
         expect[4:6, :3] = np.arange(200, 206).reshape(2, 3)\n\
         assert np.array_equal(ds[...], expect), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// A global-heap collection extended in place (CWFS second pass: bigger
/// declared size, free-space marker moved to the new tail) must stay
/// standard-readable: h5py reads both the object that fit the original
/// 4096 bytes and the one that forced the extension.
#[test]
fn extended_vlen_collection_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("extended_gcol");
    let big = "x".repeat(5000);
    {
        let file = H5File::create(&path).unwrap();
        let g = file.root_group().create_group("entry").unwrap();
        g.set_attr_string("small", "hello").unwrap();
        g.set_attr_string("big", &big).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "g = f['entry']\n\
         assert g.attrs['small'] == 'hello', g.attrs['small']\n\
         assert g.attrs['big'] == 'x' * 5000, len(g.attrs['big'])\n",
    );
    std::fs::remove_file(&path).ok();
}

/// A paged FA index (1200 chunks > 1024 per page) written across two rust
/// sessions — the second reconstructs the paged data block via
/// `open_append` — reads back exactly through h5py.
#[test]
fn fa_paged_reopened_write_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("fa_paged_reopen");
    let n = 1200usize;
    {
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([n])
            .chunk(&[1])
            .max_shape(&[Some(n)])
            .create("wide")
            .unwrap();
        // Page 0 only; page 1 (elements 1024..1200) stays uninitialized.
        ds.write_slice(&[0], &[600], &(0..600).collect::<Vec<i32>>())
            .unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        let ds = file.dataset_writer("wide").unwrap();
        ds.write_slice(&[600], &[600], &(600..n as i32).collect::<Vec<i32>>())
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['wide']\n\
         assert ds.shape == (1200,), ds.shape\n\
         assert ds.chunks == (1,), ds.chunks\n\
         assert np.array_equal(ds[...], np.arange(1200, dtype=np.int32)), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// h5py (libver='latest', fixed shape, 1200 unit chunks) writes a paged FA
/// data block and initializes only page 0; libhdf5 leaves page 1's file
/// space unwritten. Our reader must honor the page-init bitmap: page-0
/// values read back, page-1 elements read as fill.
#[test]
fn fa_paged_written_by_h5py_readable_by_rust() {
    let Some(py) = python() else { return };
    let path = tmp("fa_paged_from_h5py");
    write_with_h5py(
        py,
        &path,
        "name = f.filename; f.close()\n\
         f = h5py.File(name, 'w', libver='latest')\n\
         ds = f.create_dataset('wide', shape=(1200,), chunks=(1,), dtype='<i4')\n\
         ds[:600] = np.arange(600, dtype=np.int32)",
    );
    let file = H5File::open(&path).unwrap();
    let vals = file.dataset("wide").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(vals.len(), 1200);
    for (i, v) in vals.iter().enumerate() {
        let expect = if i < 600 { i as i32 } else { 0 };
        assert_eq!(*v, expect, "element {i}");
    }
    drop(file);
    std::fs::remove_file(&path).ok();
}

/// The full parity loop: h5py writes the paged FA (page 1 genuinely
/// unwritten by libhdf5), rust `open_append` reconstructs it and writes
/// the remaining chunks, and h5py reads the completed array back.
#[test]
fn fa_paged_written_by_h5py_completed_by_rust() {
    let Some(py) = python() else { return };
    let path = tmp("fa_paged_complete");
    write_with_h5py(
        py,
        &path,
        "name = f.filename; f.close()\n\
         f = h5py.File(name, 'w', libver='latest')\n\
         ds = f.create_dataset('wide', shape=(1200,), chunks=(1,), dtype='<i4')\n\
         ds[:600] = np.arange(600, dtype=np.int32)",
    );
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        let ds = file.dataset_writer("wide").unwrap();
        ds.write_slice(&[600], &[600], &(600..1200).collect::<Vec<i32>>())
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "ds = f['wide']\n\
         assert np.array_equal(ds[...], np.arange(1200, dtype=np.int32)), ds[...]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// Issue #8: `set_libver_latest(true)` writes a version-5 data layout message
/// for filtered chunked datasets. Two identical files differing only in the
/// knob prove both directions: the default file stays h5py-readable (v4), and
/// hdf5 < 2.0 rejects the opt-in file — the rejection is the on-disk proof
/// that a genuine v5 message was written, not a v4 one with wider index
/// fields. Under hdf5 >= 2.0 the v5 file must instead read back exactly.
/// Either way, rust-hdf5's own reader must read the v5 file.
#[cfg(feature = "deflate")]
#[test]
fn libver_latest_v5_layout_write_and_hdf5_1x_rejection() {
    let Some(py) = python() else { return };
    let data: Vec<i32> = (0..35).collect(); // 7 x 5 row-major
    let path_v4 = tmp("layout_default_v4");
    let path_v5 = tmp("layout_optin_v5");
    for (path, latest) in [(&path_v4, false), (&path_v5, true)] {
        let file = H5File::create(path).unwrap();
        file.set_libver_latest(latest).unwrap();
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

    // rust-hdf5's reader handles both versions.
    for path in [&path_v4, &path_v5] {
        let file = H5File::open(path).unwrap();
        let got = file.dataset("grid").unwrap().read_raw::<i32>().unwrap();
        assert_eq!(got, data, "rust read-back of {}", path.display());
    }

    // Positive control: the default file is plain v4 and h5py-readable.
    read_back_with_h5py(
        py,
        &path_v4,
        "ds = f['grid']\n\
         assert ds.compression == 'gzip', ds.compression\n\
         assert np.array_equal(ds[...], np.arange(35).reshape(7, 5)), ds[...]\n",
    );

    // The v5 file: rejected below hdf5 2.0, readable at 2.0+.
    let script = format!(
        "import h5py, numpy as np, sys\n\
         v2 = h5py.version.hdf5_version_tuple >= (2, 0, 0)\n\
         try:\n\
         \x20   f = h5py.File(r'{}', 'r')\n\
         \x20   v = f['grid'][...]\n\
         except Exception:\n\
         \x20   assert not v2, 'hdf5 >= 2.0 must read a v5 layout'\n\
         \x20   sys.exit(0)\n\
         assert v2, 'hdf5 < 2.0 read the opt-in file, so it is not v5 on disk'\n\
         assert np.array_equal(v, np.arange(35).reshape(7, 5)), v\n",
        path_v5.display()
    );
    let status = std::process::Command::new(py)
        .arg("-c")
        .arg(&script)
        .status()
        .expect("failed to spawn python");
    assert!(status.success(), "v5 layout check failed for {path_v5:?}");

    std::fs::remove_file(&path_v4).ok();
    std::fs::remove_file(&path_v5).ok();
}

/// Issue #11: datatype-aware conversion reads against an externally-written
/// file — h5py writes int16, big-endian int32, float32, and uint64 datasets;
/// rust-hdf5 converts on read, including the checked-overflow error path.
#[test]
fn numeric_conversion_reads_from_h5py_written_file() {
    let Some(py) = python() else { return };
    let path = tmp("numeric_conv");
    write_with_h5py(
        py,
        &path,
        "f.create_dataset('i2', data=np.array([-3, -1, 0, 7], dtype='int16'))\n\
         f.create_dataset('be_i4', data=np.array([-100000, 100000], dtype='>i4'))\n\
         f.create_dataset('f4', data=np.array([1.5, -2.25], dtype='float32'))\n\
         f.create_dataset('u8', data=np.array([1, 2**64 - 1], dtype='uint64'))\n",
    );

    let file = H5File::open(&path).unwrap();
    assert_eq!(
        file.dataset("i2")
            .unwrap()
            .read_numeric_as::<i64>()
            .unwrap(),
        vec![-3, -1, 0, 7]
    );
    assert_eq!(
        file.dataset("be_i4")
            .unwrap()
            .read_numeric_as::<i64>()
            .unwrap(),
        vec![-100_000, 100_000]
    );
    assert_eq!(
        file.dataset("f4")
            .unwrap()
            .read_numeric_as::<f64>()
            .unwrap(),
        vec![1.5, -2.25]
    );
    assert_eq!(
        file.dataset("u8")
            .unwrap()
            .read_numeric_as::<u128>()
            .unwrap(),
        vec![1, u128::from(u64::MAX)]
    );
    let err = file
        .dataset("u8")
        .unwrap()
        .read_numeric_as::<i64>()
        .unwrap_err();
    assert!(
        err.to_string().contains("does not fit in i64"),
        "unexpected error: {err}"
    );
    std::fs::remove_file(&path).ok();
}

/// `H5Attribute::read_numeric` must accept h5py's standard little-endian
/// numeric attribute datatypes (the strict datatype check cannot be *too*
/// strict), refuse a big-endian one, and `read_numeric_as` must convert it.
#[test]
fn attr_numeric_reads_from_h5py_written_file() {
    let Some(py) = python() else { return };
    let path = tmp("attr_numeric");
    write_with_h5py(
        py,
        &path,
        "d = f.create_dataset('d', data=np.zeros(2, dtype='float32'))\n\
         d.attrs.create('i4', np.int32(-7))\n\
         d.attrs.create('f8', np.float64(1.5))\n\
         d.attrs.create('be_i4', np.int32(100000), dtype='>i4')\n",
    );

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.attr("i4").unwrap().read_numeric::<i32>().unwrap(), -7);
    assert_eq!(ds.attr("f8").unwrap().read_numeric::<f64>().unwrap(), 1.5);
    let be = ds.attr("be_i4").unwrap();
    assert!(be.read_numeric::<i32>().is_err());
    assert_eq!(be.read_numeric_as::<i64>().unwrap(), vec![100_000]);
    std::fs::remove_file(&path).ok();
}

/// A dataset shrunk with `set_extent` (pruning chunks from the index) and
/// grown back is read by h5py/libhdf5 as retained data plus fill values —
/// the pruned entries must leave the extensible-array index in a state
/// libhdf5 accepts.
#[test]
fn shrunk_and_regrown_dataset_reads_fill_via_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("shrink_prune");
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
        ds.set_extent(&[3, 4]).unwrap();
        ds.set_extent(&[24, 4]).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "d = f['data']\n\
         assert d.shape == (24, 4), d.shape\n\
         v = d[...]\n\
         exp = np.arange(96, dtype='int32').reshape(24, 4)\n\
         assert (v[:3] == exp[:3]).all(), v[:3]\n\
         assert (v[3:] == 0).all(), v[3:]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// A vlen batch past the 65535-object collection index cap spills into a
/// second collection; libhdf5 must resolve references across both — the
/// per-element collection address is all it needs.
#[test]
fn spilled_vlen_batch_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("vlen_spill");
    {
        let file = H5File::create(&path).unwrap();
        let strings = vec!["x"; 65537];
        file.write_vlen_strings("bulk", &strings).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "d = f['bulk']\n\
         assert d.shape == (65537,), d.shape\n\
         v = d[...]\n\
         assert v[0] == b'x' and v[65535] == b'x' and v[65536] == b'x', v[:3]\n\
         assert (v == b'x').all()\n",
    );
    std::fs::remove_file(&path).ok();
}

/// Small vlen attributes and datasets packed into one shared collection
/// (the writer's CWFS path) stay readable: h5py resolves each reference
/// by its own (address, index) pair regardless of who else shares the
/// block.
#[test]
fn packed_shared_collection_readable_by_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("vlen_packed");
    {
        let file = H5File::create(&path).unwrap();
        let g = file.root_group().create_group("entry").unwrap();
        g.set_attr_string("NX_class", "NXentry").unwrap();
        g.set_attr_string("title", "packed heap").unwrap();
        file.write_vlen_strings("notes", &["alpha", "beta"])
            .unwrap();
        file.write_vlen_strings("tags", &["red", "green"]).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "assert f['entry'].attrs['NX_class'] == 'NXentry'\n\
         assert f['entry'].attrs['title'] == 'packed heap'\n\
         assert list(f['notes'][...]) == [b'alpha', b'beta']\n\
         assert list(f['tags'][...]) == [b'red', b'green']\n",
    );
    std::fs::remove_file(&path).ok();
}

/// An object with more attributes than the object header's compact threshold
/// keeps every one of them in a fractal heap named by the `Attribute Info`
/// message, with nothing left in the header itself. Reading only the header's
/// attribute messages therefore reports zero attributes, which is what the
/// dense-storage read path exists to prevent.
///
/// h5py's default lower libver bound pins the object header to version 1,
/// where dense attribute storage does not exist — hence the explicit v108
/// bounds.
#[test]
fn dense_attributes_written_by_h5py_read_by_rust() {
    let Some(py) = python() else { return };
    let path = tmp("attrs_dense");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "d = f.create_dataset('data', data=np.arange(8, dtype='<i4'))\n\
         g = f.create_group('grp')\n\
         for i in range(12):\n\
         \x20   d.attrs.create('a%02d' % i, np.int32(i))\n\
         for i in range(12):\n\
         \x20   g.attrs.create('b%02d' % i, np.int64(100 + i))\n\
         for i in range(12):\n\
         \x20   f.attrs.create('r%02d' % i, np.int32(1000 + i))\n",
    );

    let file = H5File::open(&path).unwrap();

    let ds = file.dataset("data").unwrap();
    let mut names = ds.attr_names().unwrap();
    names.sort();
    let expected: Vec<String> = (0..12).map(|i| format!("a{i:02}")).collect();
    assert_eq!(names, expected, "dense dataset attributes");
    for i in 0..12i32 {
        let a = ds.attr(&format!("a{i:02}")).unwrap();
        assert_eq!(a.read_numeric::<i32>().unwrap(), i);
    }

    // The same storage on a subgroup and on the root group.
    let root = file.root_group();
    let grp = root.group("grp").unwrap();
    let mut gnames = grp.attr_names().unwrap();
    gnames.sort();
    let gexpected: Vec<String> = (0..12).map(|i| format!("b{i:02}")).collect();
    assert_eq!(gnames, gexpected, "dense group attributes");

    let mut rnames = root.attr_names().unwrap();
    rnames.sort();
    let rexpected: Vec<String> = (0..12).map(|i| format!("r{i:02}")).collect();
    assert_eq!(rnames, rexpected, "dense root-group attributes");

    std::fs::remove_file(&path).ok();
}

/// An attribute at or above the heap's `max_man_size` (4 KiB) does not fit a
/// managed block, so libhdf5 stores it as a "huge" object addressed through
/// the heap's own v2 B-tree. Reading it exercises a different heap-ID branch
/// than the small dense attributes above.
#[test]
fn huge_dense_attribute_written_by_h5py_read_by_rust() {
    let Some(py) = python() else { return };
    let path = tmp("attr_huge");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "d = f.create_dataset('data', data=np.arange(8, dtype='<i4'))\n\
         d.attrs.create('big', np.arange(25600, dtype='<i4'))\n\
         d.attrs.create('small', np.int32(5))\n",
    );

    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let mut names = ds.attr_names().unwrap();
    names.sort();
    assert_eq!(names, vec!["big".to_string(), "small".to_string()]);
    let big = ds.attr("big").unwrap().read_numeric_as::<i32>().unwrap();
    assert_eq!(big.len(), 25600);
    assert_eq!(big[0], 0);
    assert_eq!(big[25599], 25599);
    assert_eq!(ds.attr("small").unwrap().read_numeric::<i32>().unwrap(), 5);
    std::fs::remove_file(&path).ok();
}

/// Opening an h5py-written file for append and touching a group rewrites that
/// group's object header from the attributes the writer collected. Collecting
/// only the header's attribute messages would rebuild a dense-storage group
/// without any of its attributes — a silent deletion of data the caller never
/// asked to change.
#[test]
fn rust_append_preserves_h5py_dense_attributes() {
    let Some(py) = python() else { return };
    let path = tmp("dense_append");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "g = f.create_group('grp')\n\
         for i in range(12):\n\
         \x20   g.attrs.create('b%02d' % i, np.int32(i))\n\
         for i in range(12):\n\
         \x20   f.attrs.create('r%02d' % i, np.int32(100 + i))\n\
         f.create_dataset('d', data=np.arange(4, dtype='<i4'))\n",
    );
    {
        let file = H5File::open_rw(&path).unwrap();
        file.root_group()
            .group("grp")
            .unwrap()
            .set_attr_string("added", "x")
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "names = sorted(f['grp'].attrs.keys())\n\
         assert names == ['added'] + ['b%02d' % i for i in range(12)], names\n\
         for i in range(12):\n\
         \x20   assert f['grp'].attrs['b%02d' % i] == i\n\
         rnames = sorted(f.attrs.keys())\n\
         assert rnames == ['r%02d' % i for i in range(12)], rnames\n\
         assert list(f['d'][...]) == [0, 1, 2, 3]\n",
    );
    std::fs::remove_file(&path).ok();
}

/// `H5Oget_info().num_attrs` on a version-2 object header is derived from the
/// Attribute Info message alone (`H5O__attr_count_real`), so a header that
/// carries attribute messages without it reports zero attributes to libhdf5
/// even though `H5Aiterate2` still yields every one. Every object the writer
/// emits — dataset, subgroup, root group — must agree with its own attribute
/// list; an object with no attributes must carry no Attribute Info message,
/// which libhdf5 asserts (`ainfo.nattrs > 0`) whenever the message is present.
#[test]
fn object_header_attribute_count_matches_h5py() {
    let Some(py) = python() else { return };
    let path = tmp("nattrs_hdr");
    {
        let file = H5File::create(&path).unwrap();
        file.set_attr_numeric("version", &3i64).unwrap();
        let grp = file.root_group().create_group("grp").unwrap();
        grp.set_attr_string("NX_class", "NXentry").unwrap();
        grp.set_attr_numeric("depth", &2i32).unwrap();
        let ds = file.new_dataset::<i32>().shape([4]).create("data").unwrap();
        ds.write_raw(&[1, 2, 3, 4]).unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("gain")
            .unwrap()
            .write_numeric(&7i32)
            .unwrap();
        let bare = file.new_dataset::<i32>().shape([2]).create("bare").unwrap();
        bare.write_raw(&[5, 6]).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "def n(o):\n\
         \x20   return h5py.h5o.get_info(o.id).num_attrs\n\
         assert n(f['/']) == 1, n(f['/'])\n\
         assert n(f['grp']) == 2, n(f['grp'])\n\
         assert n(f['data']) == 1, n(f['data'])\n\
         assert n(f['bare']) == 0, n(f['bare'])\n\
         assert sorted(f.attrs.keys()) == ['version'], sorted(f.attrs.keys())\n\
         assert sorted(f['grp'].attrs.keys()) == ['NX_class', 'depth']\n\
         assert f['data'].attrs['gain'] == 7\n",
    );
    std::fs::remove_file(&path).ok();
}

/// An attribute past the object header message limit spills the object's
/// whole attribute set to dense storage, the way `H5O__attr_create` does on
/// `raw_size >= H5O_MESG_MAX_SIZE`. Writing it as a header message instead
/// would truncate the length modulo 65536 — the size field is a `u16` — and
/// every following message would decode from the middle of this one's
/// payload, under a checksum that still matches.
///
/// `meta_size.attr` is libhdf5's own answer to "is this object dense": it
/// reports the size of the name index and the fractal heap, and is zero for
/// compact storage.
#[test]
fn an_oversized_attribute_spills_the_object_to_dense_storage() {
    let Some(py) = python() else { return };
    let path = tmp("attr_oversized");
    let big: Vec<i32> = (0..25600).collect();
    {
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<i32>().shape([4]).create("data").unwrap();
        ds.write_raw(&[1, 2, 3, 4]).unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("gain")
            .unwrap()
            .write_numeric(&7i32)
            .unwrap();
        ds.new_attr::<i32>()
            .shape([25600])
            .create("big")
            .unwrap()
            .write_array(&big)
            .unwrap();
        file.set_attr_array_numeric("rootbig", &big).unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "import numpy as np\n\
         d = f['data']\n\
         assert list(d[...]) == [1, 2, 3, 4], list(d[...])\n\
         assert sorted(d.attrs.keys()) == ['big', 'gain'], sorted(d.attrs.keys())\n\
         assert d.attrs['gain'] == 7\n\
         assert np.array_equal(d.attrs['big'], np.arange(25600, dtype='<i4'))\n\
         assert np.array_equal(f.attrs['rootbig'], np.arange(25600, dtype='<i4'))\n\
         for o in (d, f['/']):\n\
         \x20   i = h5py.h5o.get_info(o.id)\n\
         \x20   assert i.meta_size.attr.index_size > 0, i.meta_size.attr.index_size\n\
         \x20   assert i.meta_size.attr.heap_size > 0, i.meta_size.attr.heap_size\n\
         assert h5py.h5o.get_info(d.id).num_attrs == 2\n",
    );

    // And this crate reads its own dense storage back.
    let file = H5File::open(&path).unwrap();
    let ds = file.dataset("data").unwrap();
    let mut names = ds.attr_names().unwrap();
    names.sort();
    assert_eq!(names, vec!["big".to_string(), "gain".to_string()]);
    let read = ds.attr("big").unwrap().read_numeric_as::<i32>().unwrap();
    assert_eq!(read, big);
    std::fs::remove_file(&path).ok();
}

/// More links than `max_compact` (8) moves a group's whole link set out of
/// the object header and into a fractal heap plus a name index — the
/// `H5G_obj_insert` phase change. Nine is the first count that converts; the
/// eighth stays compact, so the sibling group next to it pins the boundary,
/// and the root group carries the same rule.
#[test]
fn past_max_compact_the_link_set_goes_dense() {
    let Some(py) = python() else { return };
    let path = tmp("links_dense_count");
    {
        let file = H5File::create(&path).unwrap();
        let many = file.create_group("many").unwrap();
        for i in 0..12i32 {
            many.new_dataset::<i32>()
                .shape([1])
                .create(&format!("d{i:02}"))
                .unwrap()
                .write_raw(&[i])
                .unwrap();
        }
        let few = file.create_group("few").unwrap();
        for i in 0..8i32 {
            few.new_dataset::<i32>()
                .shape([1])
                .create(&format!("e{i:02}"))
                .unwrap()
                .write_raw(&[i])
                .unwrap();
        }
        // The root group holds `many`, `few` and these ten datasets: twelve
        // links, past the same threshold.
        for i in 0..10i32 {
            file.new_dataset::<i32>()
                .shape([1])
                .create(&format!("r{i:02}"))
                .unwrap()
                .write_raw(&[i])
                .unwrap();
        }
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "many, few = f['many'], f['few']\n\
         assert sorted(many.keys()) == sorted('d%02d' % i for i in range(12))\n\
         assert all(many['d%02d' % i][0] == i for i in range(12))\n\
         assert sorted(few.keys()) == sorted('e%02d' % i for i in range(8))\n\
         assert all(few['e%02d' % i][0] == i for i in range(8))\n\
         assert all(f['r%02d' % i][0] == i for i in range(10))\n\
         for g in (many, f['/']):\n\
         \x20   i = h5py.h5o.get_info(g.id)\n\
         \x20   assert i.meta_size.obj.index_size > 0, 'expected dense links'\n\
         \x20   assert i.meta_size.obj.heap_size > 0, i.meta_size.obj.heap_size\n\
         i = h5py.h5o.get_info(few.id)\n\
         assert i.meta_size.obj.index_size == 0, 'eight links stay compact'\n",
    );

    // And this crate reads its own dense link storage back.
    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    let mut want: Vec<String> = (0..12).map(|i| format!("many/d{i:02}")).collect();
    want.extend((0..8).map(|i| format!("few/e{i:02}")));
    want.extend((0..10).map(|i| format!("r{i:02}")));
    want.sort();
    assert_eq!(names, want);
    assert_eq!(
        file.dataset("many/d07").unwrap().read_raw::<i32>().unwrap(),
        vec![7]
    );
    std::fs::remove_file(&path).ok();
}

/// More attributes than `max_compact` (8) spills the set the same way, with
/// no attribute large enough to force it on its own — the count rule of
/// `H5O__attr_create`. Nine is the first count that converts; the eighth
/// stays compact, so the object next to it pins the boundary.
#[test]
fn past_max_compact_the_attribute_set_goes_dense() {
    let Some(py) = python() else { return };
    let path = tmp("attr_dense_count");
    {
        let file = H5File::create(&path).unwrap();
        let many = file.new_dataset::<i32>().shape([2]).create("many").unwrap();
        many.write_raw(&[1, 2]).unwrap();
        for i in 0..12i32 {
            many.new_attr::<i32>()
                .shape(())
                .create(&format!("a{i}"))
                .unwrap()
                .write_numeric(&i)
                .unwrap();
        }
        let few = file.new_dataset::<i32>().shape([2]).create("few").unwrap();
        few.write_raw(&[3, 4]).unwrap();
        for i in 0..8i32 {
            few.new_attr::<i32>()
                .shape(())
                .create(&format!("b{i}"))
                .unwrap()
                .write_numeric(&i)
                .unwrap();
        }
        let grp = file.create_group("grp").unwrap();
        for i in 0..12i32 {
            grp.set_attr_numeric(&format!("g{i}"), &i).unwrap();
        }
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "many, few, grp = f['many'], f['few'], f['grp']\n\
         assert sorted(many.attrs.keys()) == sorted('a%d' % i for i in range(12))\n\
         assert all(many.attrs['a%d' % i] == i for i in range(12))\n\
         assert all(grp.attrs['g%d' % i] == i for i in range(12))\n\
         for o in (many, grp):\n\
         \x20   i = h5py.h5o.get_info(o.id)\n\
         \x20   assert i.num_attrs == 12, i.num_attrs\n\
         \x20   assert i.meta_size.attr.index_size > 0, 'expected dense storage'\n\
         i = h5py.h5o.get_info(few.id)\n\
         assert i.num_attrs == 8, i.num_attrs\n\
         assert i.meta_size.attr.index_size == 0, 'eight attributes stay compact'\n",
    );

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset("many").unwrap().attr_names().unwrap();
    names.sort();
    let mut want: Vec<String> = (0..12).map(|i| format!("a{i}")).collect();
    want.sort();
    assert_eq!(names, want);
    std::fs::remove_file(&path).ok();
}

/// The append path takes the same attribute in through the other door: it
/// rebuilds an object header from the attributes read out of it, so an
/// attribute libhdf5 put in dense storage has to go back out the same way.
/// The root group's header is rewritten by every finalize, so this is the
/// reopen that used to be refused outright — nothing else in the session
/// touches the file, and the whole 100 KiB attribute has to survive it.
#[test]
fn reopen_carries_a_libhdf5_dense_root_attribute_back_out_dense() {
    let Some(py) = python() else { return };
    let path = tmp("attr_oversized_reopen");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "f.create_dataset('data', data=np.arange(8, dtype='<i4'))\n\
         f.attrs.create('big', np.arange(25600, dtype='<i4'))\n\
         f.attrs['gain'] = np.int32(7)\n",
    );

    {
        let file = H5File::open_rw(&path).unwrap();
        file.set_attr_numeric("added", &5i32).unwrap();
        file.close().unwrap();
    }

    read_back_with_h5py(
        py,
        &path,
        "assert list(f['data'][...]) == list(range(8)), list(f['data'][...])\n\
         assert f.attrs['big'].shape == (25600,), f.attrs['big'].shape\n\
         assert f.attrs['big'][25599] == 25599\n\
         assert f.attrs['gain'] == 7\n\
         assert f.attrs['added'] == 5\n\
         i = h5py.h5o.get_info(f['/'].id)\n\
         assert i.num_attrs == 3, i.num_attrs\n\
         assert i.meta_size.attr.index_size > 0, 'expected dense storage'\n",
    );

    let file = H5File::open(&path).unwrap();
    let mut names = file.attr_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        vec!["added".to_string(), "big".to_string(), "gain".to_string()]
    );
    std::fs::remove_file(&path).ok();
}

/// Dense storage this crate wrote has to be readable by the crate's own
/// append path, not just by libhdf5: the reopen reads the set back out of the
/// heap and name index it wrote, adds to it, and lays a fresh one down.
#[test]
fn rust_written_dense_attributes_survive_a_rust_reopen() {
    let Some(py) = python() else { return };
    let path = tmp("attr_dense_rust_reopen");
    {
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<i32>().shape([2]).create("data").unwrap();
        ds.write_raw(&[1, 2]).unwrap();
        for i in 0..12i32 {
            ds.new_attr::<i32>()
                .shape(())
                .create(&format!("a{i:02}"))
                .unwrap()
                .write_numeric(&i)
                .unwrap();
        }
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        let ds = file.dataset_writer("data").unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("late")
            .unwrap()
            .write_numeric(&99i32)
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "d = f['data']\n\
         assert all(d.attrs['a%02d' % i] == i for i in range(12))\n\
         assert d.attrs['late'] == 99, d.attrs['late']\n\
         i = h5py.h5o.get_info(d.id)\n\
         assert i.num_attrs == 13, i.num_attrs\n\
         assert i.meta_size.attr.index_size > 0, 'expected dense storage'\n",
    );
    std::fs::remove_file(&path).ok();
}

/// A reopened *group*'s attributes go back out through a header this writer
/// rebuilds from scratch on every finalize, so a dense set has to survive a
/// session that never mentions the group at all.
#[test]
fn reopen_carries_a_libhdf5_dense_group_attribute_back_out_dense() {
    let Some(py) = python() else { return };
    let path = tmp("attr_dense_group_reopen");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "g = f.create_group('g')\n\
         for i in range(12):\n\
         \x20   g.attrs.create('g%02d' % i, np.int32(i))\n\
         g.attrs.create('big', np.arange(25600, dtype='<i4'))\n\
         f.create_dataset('data', data=np.arange(8, dtype='<i4'))\n",
    );

    {
        let file = H5File::open_rw(&path).unwrap();
        file.set_attr_numeric("touched", &1i32).unwrap();
        file.close().unwrap();
    }

    read_back_with_h5py(
        py,
        &path,
        "g = f['g']\n\
         assert all(g.attrs['g%02d' % i] == i for i in range(12))\n\
         assert g.attrs['big'].shape == (25600,), g.attrs['big'].shape\n\
         assert g.attrs['big'][25599] == 25599\n\
         assert list(f['data'][...]) == list(range(8))\n\
         i = h5py.h5o.get_info(g.id)\n\
         assert i.num_attrs == 13, i.num_attrs\n\
         assert i.meta_size.attr.index_size > 0, 'expected dense storage'\n",
    );

    let file = H5File::open(&path).unwrap();
    let names = file.root_group().group("g").unwrap().attr_names().unwrap();
    assert_eq!(names.len(), 13, "{names:?}");
    std::fs::remove_file(&path).ok();
}

/// An attribute set on a *reopened dataset* is a header change the chunk-write
/// counters cannot see, so finalize used to keep the dataset's original header
/// and discard the attribute — no error, no trace. Both the replacement of an
/// existing value and the addition of a new one have to reach the file, and
/// the spill to dense storage has to fire for the reopened set as well.
#[test]
fn attributes_set_on_a_reopened_dataset_reach_the_file() {
    let Some(py) = python() else { return };
    let path = tmp("attr_reopen_dataset");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "d = f.create_dataset('data', data=np.arange(8, dtype='<i4'))\n\
         d.attrs['gain'] = np.int32(7)\n",
    );

    {
        let file = H5File::open_rw(&path).unwrap();
        let ds = file.dataset_writer("data").unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("gain")
            .unwrap()
            .write_numeric(&42i32)
            .unwrap();
        // Past max_compact, so the reopened set spills to dense storage too.
        for i in 0..10i32 {
            ds.new_attr::<i32>()
                .shape(())
                .create(&format!("added{i}"))
                .unwrap()
                .write_numeric(&i)
                .unwrap();
        }
        file.close().unwrap();
    }

    read_back_with_h5py(
        py,
        &path,
        "d = f['data']\n\
         assert list(d[...]) == list(range(8)), list(d[...])\n\
         assert d.attrs['gain'] == 42, d.attrs['gain']\n\
         assert all(d.attrs['added%d' % i] == i for i in range(10))\n\
         i = h5py.h5o.get_info(d.id)\n\
         assert i.num_attrs == 11, i.num_attrs\n\
         assert i.meta_size.attr.index_size > 0, 'expected dense storage'\n",
    );
    std::fs::remove_file(&path).ok();
}

/// An attribute this crate cannot decode must be listed, not dropped.
///
/// An object-reference datatype (class 7) is one `DatatypeMessage::decode`
/// refuses, and the attribute message that carries it was silently discarded:
/// `attr_names()` then described a file that did not contain the attribute,
/// and a header rewrite deleted it for real. The name sits ahead of the
/// datatype in the message, so it is knowable either way — the listing keeps
/// it, `attr_unreadable_reason` says what stands in the way, and typed access
/// fails with the same text.
#[test]
fn undecodable_attribute_is_listed_with_a_reason() {
    let Some(py) = python() else { return };
    let path = tmp("attr_undecodable");
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "d = f.create_dataset('data', data=np.arange(8, dtype='<i4'))\n\
         g = f.create_group('g')\n\
         d.attrs['gain'] = np.int32(7)\n\
         d.attrs.create('ref', d.ref, dtype=h5py.ref_dtype)\n\
         g.attrs['label'] = 'ok'\n\
         g.attrs.create('gref', d.ref, dtype=h5py.ref_dtype)\n",
    );
    {
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        let mut names = ds.attr_names().unwrap();
        names.sort();
        assert_eq!(names, vec!["gain".to_string(), "ref".to_string()]);

        let why = ds
            .attr_unreadable_reason("ref")
            .unwrap()
            .expect("a reference attribute must report why it cannot be read");
        assert!(why.contains("datatype class 7"), "{why}");
        assert_eq!(ds.attr_unreadable_reason("gain").unwrap(), None);

        // Typed access refuses with the same reason, and the readable
        // attribute beside it is unaffected.
        let err = ds.attr("ref").err().expect("attr('ref') must fail");
        assert!(err.to_string().contains("datatype class 7"), "{err}");
        assert_eq!(ds.attr("gain").unwrap().read_numeric::<i32>().unwrap(), 7);

        let grp = file.root_group().group("g").unwrap();
        let mut gnames = grp.attr_names().unwrap();
        gnames.sort();
        assert_eq!(gnames, vec!["gref".to_string(), "label".to_string()]);
        assert!(grp
            .attr_unreadable_reason("gref")
            .unwrap()
            .is_some_and(|w| w.contains("datatype class 7")));
        assert_eq!(grp.attr_unreadable_reason("label").unwrap(), None);
        assert_eq!(grp.attr_string("label").unwrap(), "ok");

        // An attribute that is genuinely absent stays absent, not unreadable.
        assert_eq!(ds.attr_unreadable_reason("nope").unwrap(), None);
        assert!(ds.attr("nope").is_err());
    }

    // A header rewrite must put back what it could not decode: the writer
    // re-emits the message bytes verbatim rather than dropping the attribute.
    {
        let file = H5File::open_rw(&path).unwrap();
        file.root_group()
            .group("g")
            .unwrap()
            .set_attr_string("added", "x")
            .unwrap();
        file.close().unwrap();
    }
    read_back_with_h5py(
        py,
        &path,
        "g = f['g']\n\
         assert sorted(g.attrs.keys()) == ['added', 'gref', 'label'], sorted(g.attrs.keys())\n\
         assert f[g.attrs['gref']].name == '/data', f[g.attrs['gref']].name\n\
         assert g.attrs['label'] == 'ok'\n\
         d = f['data']\n\
         assert sorted(d.attrs.keys()) == ['gain', 'ref'], sorted(d.attrs.keys())\n\
         assert f[d.attrs['ref']].name == '/data'\n",
    );
    std::fs::remove_file(&path).ok();
}

/// A dense attribute set that will not read is the object's failure to report.
///
/// The name index stores hashes, not names, so a heap or index this crate
/// cannot walk leaves nothing to list and nothing to hang a per-attribute
/// reason on. The collector used to swallow that whole and hand back an empty
/// list: the object then looked like one with no attributes, and the append
/// path rebuilt its header from that emptiness — deleting on disk the
/// attributes it had never read. Now the listing carries the failure, and the
/// reopen refuses.
#[test]
fn an_unreadable_dense_attribute_set_is_reported_on_the_object() {
    let Some(py) = python() else { return };
    let path = tmp("attr_dense_corrupt");
    // Ten attributes on `g` cross `max_compact` (8), so libhdf5 moves them to
    // dense storage; the dataset and the root group stay compact.
    write_with_h5py_libver(
        py,
        &path,
        Some("v108"),
        "d = f.create_dataset('data', data=np.arange(4, dtype='<i4'))\n\
         d.attrs['gain'] = np.int32(7)\n\
         f.attrs['top'] = 'root'\n\
         g = f.create_group('g')\n\
         for i in range(9):\n\
        \x20    g.attrs['a%02d' % i] = np.int32(i)\n\
         g.attrs['keep'] = 'yes'\n",
    );

    // Break the name index the attribute info message points at. The B-tree
    // header lives outside the object header, so this corrupts the attribute
    // set without disturbing the header checksum that guards the group's
    // links — exactly the shape of damage that used to read back as "no
    // attributes".
    let mut raw = std::fs::read(&path).unwrap();
    let hits: Vec<usize> = (0..raw.len() - 4)
        .filter(|&i| &raw[i..i + 4] == b"BTHD")
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "fixture must hold exactly one v2 B-tree, the attribute name index"
    );
    raw[hits[0]..hits[0] + 4].copy_from_slice(b"XXXX");
    std::fs::write(&path, &raw).unwrap();

    {
        let file = H5File::open(&path).unwrap();
        let grp = file.root_group().group("g").unwrap();

        let err = grp
            .attr_names()
            .expect_err("an unreadable attribute set must not list as empty")
            .to_string();
        assert!(
            err.contains("attributes of 'g' cannot be read whole"),
            "{err}"
        );
        assert!(err.contains("dense attribute storage"), "{err}");

        let why = grp
            .attrs_unreadable_reason()
            .unwrap()
            .expect("the object must say why its attributes cannot be listed");
        assert!(why.contains("dense attribute storage"), "{why}");

        // A name lookup cannot answer "absent" out of a set that was never
        // read whole: `keep` is in there somewhere.
        let err = grp.attr_string("keep").expect_err("must not be NotFound");
        assert!(err.to_string().contains("cannot be read whole"), "{err}");

        // Objects whose attributes did read are unaffected.
        assert_eq!(file.attr_names().unwrap(), vec!["top".to_string()]);
        assert_eq!(file.attrs_unreadable_reason().unwrap(), None);
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.attr_names().unwrap(), vec!["gain".to_string()]);
        assert_eq!(ds.attrs_unreadable_reason().unwrap(), None);
        file.close().unwrap();
    }

    // The append path rebuilds object headers out of the attributes it read,
    // so a set it could not read whole must stop the open — before any of this
    // session's bytes land in the file.
    let before = std::fs::read(&path).unwrap();
    let err = match H5File::open_rw(&path) {
        Ok(_) => panic!("open_rw must refuse a file with an unreadable attribute set"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("cannot be read whole"), "{err}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused reopen must leave the file byte-identical"
    );
    std::fs::remove_file(&path).ok();
}
