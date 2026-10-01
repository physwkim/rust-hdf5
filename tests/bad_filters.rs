//! Filter pipelines whose stored parameters or chunk sizes describe more
//! than the chunk holds (CVE fixes in HDFGroup/hdf5#6497).
//!
//! The four fixtures are upstream's `test/testfiles/bad_*.h5` (written by
//! `test/gen_bad_filters.c`): each is a well-formed file with one filtered
//! chunk whose pipeline message or chunk record was patched afterwards.
//!
//! - `bad_nbit_params.h5`: the nbit record's stored parameter count is 0,
//!   so the filter runs with no parameters at all.
//! - `bad_nbit_decompress.h5`: the nbit element count (`cd_values[2]`) is
//!   inflated far past what the small packed chunk holds.
//! - `bad_nbit_parms_walk.h5`: the nbit parameter count is cut to 7, one
//!   short of the 8 that describe an atomic type, so the walk through the
//!   parameters runs one past the list.
//! - `bad_fletcher32.h5`: a fletcher32 chunk's stored size is 2, smaller
//!   than the 4-byte checksum it must end with.
//!
//! libhdf5 crashed on each until `H5Z__filter_nbit` validated the header
//! and bounded both the packed input and the parameter walk, and
//! `H5Z__filter_fletcher32` refused a chunk shorter than its checksum. The
//! read here must fail with an error and nothing else.

use std::path::PathBuf;

use rust_hdf5::H5File;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// The error the whole-dataset read gives, or a panic naming what happened
/// instead.
fn read_refusal(fixture_name: &str, dataset: &str) -> String {
    let file = H5File::open(fixture(fixture_name)).unwrap();
    let dset = file.dataset(dataset).unwrap();
    match dset.read_raw_bytes() {
        Ok(bytes) => panic!(
            "{fixture_name}: read {} bytes from a chunk it should refuse",
            bytes.len()
        ),
        Err(e) => e.to_string(),
    }
}

#[test]
fn an_nbit_record_with_no_parameters_is_refused() {
    let err = read_refusal("bad_nbit_params.h5", "Nbit_float_data_le");
    assert!(err.contains("cd_values too short"), "{err}");
}

#[test]
fn an_nbit_element_count_past_the_packed_chunk_is_refused() {
    let err = read_refusal("bad_nbit_decompress.h5", "Nbit_float_data_le");
    assert!(err.contains("nbit: buffer too short"), "{err}");
}

#[test]
fn an_nbit_parameter_list_shorter_than_its_type_is_refused() {
    let err = read_refusal("bad_nbit_parms_walk.h5", "Nbit_int_data_le");
    assert!(err.contains("parameter list truncated"), "{err}");
}

#[test]
fn a_fletcher32_chunk_shorter_than_its_checksum_is_refused() {
    let err = read_refusal("bad_fletcher32.h5", "Fletcher_float_data_be");
    assert!(
        err.contains("fletcher32: data too short for checksum"),
        "{err}"
    );
}
