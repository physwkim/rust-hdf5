//! Reading integers whose datatype keeps fewer significant bits than its
//! width (`H5Tset_precision`/`H5Tset_offset`), as NDFileHDF5 writes for its
//! N-bit compression. Every typed read returns what libhdf5's `H5Dread`
//! returns into the native type: the significant bits shifted down, masked
//! for an unsigned type or bit field, sign-extended from the precision for a
//! signed one.
//!
//! An enum over such a type reads as its base does: libhdf5 converts an
//! enum to a number through its base (`H5T__conv_enum_numeric`).
//!
//! The fixture is libhdf5 output (`tests/fixtures/gen_nbit_precision.py`):
//! datasets of shape (3, 4, 5) and root attributes of shape (5,), whose
//! values are the generator's formula of the row-major element index,
//! repeated here.

use std::fmt::Debug;
use std::path::PathBuf;

use rust_hdf5::swmr::SwmrFileReader;
use rust_hdf5::types::H5Type;
use rust_hdf5::{DatatypeMessage, H5File};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nbit_precision.h5")
}

const N: i64 = 60;

fn values<T: TryFrom<i64>>(f: impl Fn(i64) -> i64) -> Vec<T>
where
    T::Error: Debug,
{
    (0..N).map(|i| T::try_from(f(i)).unwrap()).collect()
}

fn u12(i: i64) -> i64 {
    (i * 37 + 5) % 4096
}

fn s12(i: i64) -> i64 {
    (i * 37) % 4096 - 2048
}

/// Every typed read path of `name`, against `expected` in row-major order.
fn check<T>(name: &str, expected: &[T])
where
    T: H5Type + Copy + Default + PartialEq + Debug + Into<i64>,
{
    let wide: Vec<i64> = expected.iter().map(|&v| v.into()).collect();
    // Rows 1..3, the second and third chunk of a chunked dataset.
    let tail = &expected[20..];
    // Rows 0 and 2.
    let strided: Vec<T> = [&expected[..20], &expected[40..]].concat();

    let file = H5File::open(fixture()).unwrap();
    let ds = file.dataset(name).unwrap();
    assert_eq!(ds.read_raw::<T>().unwrap(), expected, "{name}: read_raw");
    assert_eq!(
        ds.read_slice::<T>(&[1, 0, 0], &[2, 4, 5]).unwrap(),
        tail,
        "{name}: read_slice"
    );
    assert_eq!(
        ds.read_hyperslab::<T>(&[0, 0, 0], &[2, 1, 1], &[2, 1, 1], &[1, 4, 5])
            .unwrap(),
        strided,
        "{name}: read_hyperslab"
    );
    assert_eq!(
        ds.read_points::<T>(&[vec![2, 3, 4], vec![0, 1, 2]])
            .unwrap(),
        [expected[59], expected[7]],
        "{name}: read_points"
    );
    let mut all = vec![T::default(); expected.len()];
    ds.read_raw_into(&mut all).unwrap();
    assert_eq!(all, expected, "{name}: read_raw_into");
    let mut part = vec![T::default(); tail.len()];
    ds.read_slice_into(&mut part, &[1, 0, 0], &[2, 4, 5])
        .unwrap();
    assert_eq!(part, tail, "{name}: read_slice_into");
    assert_eq!(
        ds.read_numeric_as::<i64>().unwrap(),
        wide,
        "{name}: read_numeric_as"
    );
    assert_eq!(
        ds.read_numeric_slice_as::<i64>(&[1, 0, 0], &[2, 4, 5])
            .unwrap(),
        wide[20..],
        "{name}: read_numeric_slice_as"
    );

    let mut swmr = SwmrFileReader::open(fixture()).unwrap();
    assert_eq!(
        swmr.read_dataset::<T>(name).unwrap(),
        expected,
        "{name}: SwmrFileReader::read_dataset"
    );
    assert_eq!(
        swmr.read_slice::<T>(name, &[1, 0, 0], &[2, 4, 5]).unwrap(),
        tail,
        "{name}: SwmrFileReader::read_slice"
    );
}

#[test]
fn unsigned_offset_0_nbit_filter() {
    check::<u16>("u12_o0_nbit", &values(u12));
}

#[test]
fn unsigned_offset_4_nbit_filter() {
    check::<u16>("u12_o4_nbit", &values(u12));
}

#[test]
fn signed_offset_0_nbit_filter() {
    check::<i16>("s12_o0_nbit", &values(s12));
}

#[test]
fn signed_offset_3_nbit_filter() {
    check::<i16>("s12_o3_nbit", &values(s12));
}

#[test]
fn signed_4_byte_offset_2_nbit_filter() {
    check::<i32>(
        "s27_o2_nbit",
        &values(|i| (i * 1_000_003) % (1 << 27) - (1 << 26)),
    );
}

#[test]
fn unsigned_offset_4_chunked_without_filter() {
    check::<u16>("u12_o4_chunked", &values(u12));
}

#[test]
fn signed_offset_3_chunked_without_filter() {
    check::<i16>("s12_o3_chunked", &values(s12));
}

#[test]
fn unsigned_offset_4_contiguous() {
    check::<u16>("u12_o4_contig", &values(u12));
}

#[test]
fn signed_big_endian_offset_5_contiguous() {
    check::<i16>("s11_o5_be_contig", &values(|i| (i * 37) % 2048 - 1024));
}

#[test]
fn bit_field_offset_4_contiguous() {
    check::<u16>("b12_o4_contig", &values(|i| (i * 41 + 3) % 4096));
}

/// An attribute's bytes are its stored image, unpacked by its typed read as
/// a dataset's are by the reader.
#[test]
fn reduced_precision_attributes() {
    let file = H5File::open(fixture()).unwrap();
    let signed = file.attr("s12_o3").unwrap().read_numeric_as::<i64>();
    assert_eq!(signed.unwrap(), [-2048, -1048, -48, 952, 1952]);
    let bits = file.attr("b12_o4").unwrap().read_numeric_as::<u16>();
    assert_eq!(bits.unwrap(), [7, 1007, 2007, 3007, 4007]);
}

/// Element `i` of an enum dataset is member `i % members.len()`.
fn check_enum<T>(name: &str, members: &[i64])
where
    T: H5Type + Copy + Default + PartialEq + Debug + Into<i64> + TryFrom<i64>,
    <T as TryFrom<i64>>::Error: Debug,
{
    let ds = H5File::open(fixture()).unwrap().dataset(name).unwrap();
    assert!(
        matches!(ds.datatype().unwrap(), DatatypeMessage::Enum { .. }),
        "{name} is an enum"
    );
    check::<T>(name, &values(|i| members[(i as usize) % members.len()]));
}

#[test]
fn signed_enum_offset_3_nbit_filter() {
    check_enum::<i16>("e_s12_o3_nbit", &[-2048, -1000, -1, 0, 1, 7, 1500, 2047]);
}

#[test]
fn unsigned_big_endian_enum_offset_4_contiguous() {
    check_enum::<u16>("e_u12_o4_be_contig", &[0, 1, 9, 300, 2048, 4000, 4095]);
}

#[test]
fn reduced_precision_enum_attribute() {
    let file = H5File::open(fixture()).unwrap();
    let attr = file.attr("e_s12_o3").unwrap();
    assert!(matches!(
        attr.datatype().unwrap(),
        DatatypeMessage::Enum { .. }
    ));
    assert_eq!(
        attr.read_numeric_as::<i64>().unwrap(),
        [-2048, -1000, -1, 0, 2047]
    );
}
