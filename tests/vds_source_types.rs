//! A virtual dataset whose sources store their values in other datatypes:
//! another precision, bit offset, byte order, size or sign, behind the N-bit
//! filter, in another file, or itself virtual. Each source's elements read
//! back converted from that source's own datatype into the virtual
//! dataset's, as libhdf5's `H5T_convert` converts them.
//!
//! The fixture is libhdf5 output (`tests/fixtures/gen_vds_source_types.py`);
//! `expected/<name>` holds what libhdf5 reads from each virtual dataset in
//! its own datatype, as the native type. Row `r` of each (rows, 5) virtual
//! dataset is one whole source; a row no mapping covers is the fill value.

use std::fmt::Debug;
use std::path::PathBuf;

use rust_hdf5::format::FormatError;
use rust_hdf5::swmr::SwmrFileReader;
use rust_hdf5::types::H5Type;
use rust_hdf5::{H5File, Hdf5Error, ReadNumeric};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vds_source_types.h5")
}

fn oracle<T: H5Type + Copy + Default>(name: &str) -> Vec<T> {
    let file = H5File::open(fixture()).unwrap();
    let ds = file.dataset(&format!("expected/{name}")).unwrap();
    ds.read_raw::<T>().unwrap()
}

/// Every typed read path of the (rows, 5) virtual dataset `name`, against
/// `expected` in row-major order; `read_numeric_as` reads it as `W`.
fn check<T, W>(name: &str, expected: &[T])
where
    T: H5Type + Copy + Default + PartialEq + Debug + Into<W>,
    W: ReadNumeric + PartialEq + Debug,
{
    let rows = expected.len() / 5;
    assert!(rows >= 2, "{name}: {rows} rows");
    let wide: Vec<W> = expected.iter().map(|&v| v.into()).collect();
    // Rows 1.. and the first and last row.
    let tail = &expected[5..];
    let ends: Vec<T> = [&expected[..5], &expected[expected.len() - 5..]].concat();

    let file = H5File::open(fixture()).unwrap();
    let ds = file.dataset(name).unwrap();
    assert_eq!(ds.read_raw::<T>().unwrap(), expected, "{name}: read_raw");
    assert_eq!(
        ds.read_slice::<T>(&[1, 0], &[rows - 1, 5]).unwrap(),
        tail,
        "{name}: read_slice"
    );
    assert_eq!(
        ds.read_hyperslab::<T>(&[0, 0], &[rows - 1, 1], &[2, 1], &[1, 5])
            .unwrap(),
        ends,
        "{name}: read_hyperslab"
    );
    assert_eq!(
        ds.read_points::<T>(&[vec![rows - 1, 4], vec![1, 2]])
            .unwrap(),
        [expected[expected.len() - 1], expected[7]],
        "{name}: read_points"
    );
    let mut all = vec![T::default(); expected.len()];
    ds.read_raw_into(&mut all).unwrap();
    assert_eq!(all, expected, "{name}: read_raw_into");
    let mut part = vec![T::default(); tail.len()];
    ds.read_slice_into(&mut part, &[1, 0], &[rows - 1, 5])
        .unwrap();
    assert_eq!(part, tail, "{name}: read_slice_into");
    assert_eq!(
        ds.read_numeric_as::<W>().unwrap(),
        wide,
        "{name}: read_numeric_as"
    );

    let mut swmr = SwmrFileReader::open(fixture()).unwrap();
    assert_eq!(
        swmr.read_dataset::<T>(name).unwrap(),
        expected,
        "{name}: SwmrFileReader::read_dataset"
    );
    assert_eq!(
        swmr.read_slice::<T>(name, &[1, 0], &[rows as u64 - 1, 5])
            .unwrap(),
        tail,
        "{name}: SwmrFileReader::read_slice"
    );
}

/// Sources of reduced precision at a bit offset, of the other byte order,
/// N-bit filtered, wider (clamped), unsigned, and in another file.
#[test]
fn integer_sources_convert_into_the_virtual_datatype() {
    let expected = oracle::<i16>("v_i16");
    // The 4-byte source's -40000 and 40000 clamp to the 2-byte range.
    assert_eq!(expected[15..20], [-32768, -32768, 0, 32767, 32767]);
    check::<i16, i64>("v_i16", &expected);
}

/// A virtual dataset of reduced precision at a bit offset: a full-width
/// source's values past its 12 bits clamp to 4095.
#[test]
fn sources_convert_into_a_reduced_precision_virtual_datatype() {
    let expected = oracle::<u16>("v_u16_12_4");
    assert_eq!(expected[10..15], [0, 4095, 4095, 4095, 4095]);
    check::<u16, i64>("v_u16_12_4", &expected);
}

/// A big-endian binary32 source and a binary64 one narrowed to binary32.
#[test]
fn float_sources_convert_into_the_virtual_datatype() {
    let mut expected = oracle::<f32>("v_f32");
    // libhdf5 reads 3.4028235e38 (binary64), which rounds to f32::MAX, as
    // +inf: its hard double-to-float conversion tests `> FLT_MAX` before it
    // rounds (H5T_CONV_Ff_CORE, H5Tconv_macros.h:516), where its soft one
    // (`H5T__conv_f_f`) rounds to f32::MAX. This reads the rounded value.
    assert_eq!(expected[8], f32::INFINITY);
    expected[8] = f32::MAX;
    check::<f32, f64>("v_f32", &expected);
}

#[test]
fn an_integer_source_converts_into_a_float_virtual_datatype() {
    let expected = oracle::<f64>("v_f64_of_i32");
    assert_eq!(expected, [-40000.0, -32768.0, 0.0, 32767.0, 40000.0]);
    let file = H5File::open(fixture()).unwrap();
    let ds = file.dataset("v_f64_of_i32").unwrap();
    assert_eq!(ds.read_raw::<f64>().unwrap(), expected);
    assert_eq!(ds.read_numeric_as::<f64>().unwrap(), expected);
}

/// A virtual dataset whose source is a virtual dataset: the inner one's
/// values are those of its own datatype, then convert into the outer one's.
#[test]
fn a_virtual_source_converts_from_its_own_datatype() {
    let expected = oracle::<i32>("v_i32be_of_v_i16");
    assert_eq!(
        expected,
        oracle::<i16>("v_i16")
            .into_iter()
            .map(i32::from)
            .collect::<Vec<_>>()
    );
    check::<i32, i64>("v_i32be_of_v_i16", &expected);
}

/// libhdf5 converts no integer into a bit field, so neither does a read of
/// a bit-field virtual dataset over an integer source.
#[test]
fn a_source_with_no_conversion_into_the_virtual_datatype_is_refused() {
    let file = H5File::open(fixture()).unwrap();
    let ds = file.dataset("v_b16_of_i16").unwrap();
    let err = ds.read_raw::<u16>().unwrap_err();
    assert!(
        matches!(err, Hdf5Error::Format(FormatError::UnsupportedFeature(_))),
        "{err:?}"
    );
}
