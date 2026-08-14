//! Symbol-table group storage under a version-2 superblock.
//!
//! A file moves to a version-2 superblock as soon as it asks for a
//! shared-message table, a file-space strategy or non-default B-tree ranks —
//! none of which changes how its groups are stored. Unless the library-version
//! bounds ask for the newer format, the root group is still a symbol table, so
//! the open path must pick group storage from the root object header rather
//! than from the superblock version.
//!
//! Fixtures come from `tests/fixtures/gen_sbext.sh`.

use std::path::PathBuf;

use rust_hdf5::H5File;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Paged aggregation: version-2 superblock, symbol-table root group, one
/// dataset. Reading it as link storage finds nothing and reports an empty
/// file.
#[test]
fn v2_superblock_with_a_symbol_table_root_lists_its_datasets() {
    let file = H5File::open(fixture("sbext_paged.h5")).unwrap();
    assert_eq!(file.dataset_names(), vec!["paged"]);

    let data: Vec<i32> = file.dataset("paged").unwrap().read_raw().unwrap();
    assert_eq!(data, (0..16).map(|i| i * 3).collect::<Vec<i32>>());
}

/// The same storage with 209 entries in the root symbol table node and a
/// chunked dataset behind a v1 B-tree, so the walk covers the whole node and
/// the chunk index below it.
#[test]
fn v2_superblock_symbol_table_root_walks_every_entry() {
    let file = H5File::open(fixture("sbext_btreek.h5")).unwrap();

    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(
        names,
        vec!["chunked", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
    );

    let chunked: Vec<i32> = file.dataset("chunked").unwrap().read_raw().unwrap();
    assert_eq!(chunked.len(), 1000);
    assert_eq!(chunked[999], 999);
}
