//! The 1.12 reference kinds — `H5R_OBJECT2`, `H5R_DATASET_REGION2` and
//! `H5R_ATTR`, all stored as `H5T_STD_REF`. Reading and writing both cover all
//! three: the object form carries its target inline, the other two through a
//! global-heap blob whose id the element holds.
//!
//! The fixture is libhdf5 1.14.6 output built by `tests/fixtures/gen_revised_refs.c`;
//! h5py 3.x cannot stand in for it, as it raises "Unknown reference type" on
//! an `H5T_STD_REF` dataset even for reading. It holds a 4x6 `matrix` with an
//! attribute `note`, a group `grp`, and one dataset per reference kind:
//! `objrefs` names `/matrix` and `/grp`, `regrefs` selects the hyperslab
//! (1,2)-(2,4) and the points (0,1) and (3,5) of `/matrix`, and `attrrefs`
//! names `/matrix`'s `note`.
//!
//! That h5py blind spot is also why the write case is judged by `h5dump`,
//! which dereferences an `H5T_STD_REF` element and prints what it names.
//!
//! `ext_refs.h5` and `ext_ref_target.h5`, from
//! `tests/fixtures/gen_external_refs.c`, are the same three kinds written
//! across files, which is the only way to make libhdf5 set `H5R_IS_EXTERNAL`
//! and record a file name. `h5dump` is the witness there too: run from the
//! package root it resolves every one of them, run from anywhere else it
//! prints the file name and `UNKNOWN`, because the name is used verbatim
//! against the working directory.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::{
    H5File, Hyperslab, LibverBound, PointSelection, Reference, RegularHyperslab, Selection,
};

const REVISED_REFS: &[u8] = include_bytes!("fixtures/revised_refs.h5");

fn write_temp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_revised_refs_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.h5"));
    std::fs::write(&path, REVISED_REFS).unwrap();
    path
}

/// A path for a file this crate writes, in a directory of its own.
fn write_path(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_revised_write_{}_{}_{}",
        label,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{label}.h5"))
}

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`. Only the directory matters here: `h5dump` is taken
/// from beside the interpreter, so it is the tool of the libhdf5 the rest of
/// the suite is judged against.
const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn h5dump() -> Option<PathBuf> {
    let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
        Ok(p) => vec![p],
        Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
    };
    let found = candidates
        .iter()
        .map(|c| PathBuf::from(c).parent().unwrap().join("h5dump"))
        .find(|t| t.exists());
    if found.is_none() {
        eprintln!("skipping the h5dump cross-check: none of {candidates:?} ships one");
    }
    found
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// An `H5R_OBJECT2` element carries its token inline; both a dataset and a
/// group resolve to their paths.
#[test]
fn object2_references_resolve_to_paths() {
    let path = write_temp("object2");
    let file = H5File::open(&path).unwrap();
    let refs = file.dataset("objrefs").unwrap().read_references().unwrap();
    let paths: Vec<Option<&str>> = refs.iter().map(Reference::path).collect();
    assert_eq!(paths, vec![Some("/matrix"), Some("/grp")]);
    assert!(refs.iter().all(|r| matches!(r, Reference::Object { .. })));
    drop(file);
    cleanup(&path);
}

/// An `H5R_DATASET_REGION2` keeps its selection in a global-heap blob; both
/// selection classes decode to the bounds `H5Sget_select_bounds` reports.
#[test]
fn region2_references_report_their_selections() {
    let path = write_temp("region2");
    let file = H5File::open(&path).unwrap();
    let refs = file.dataset("regrefs").unwrap().read_references().unwrap();
    assert_eq!(refs.len(), 2);
    assert_eq!(refs[0].path(), Some("/matrix"));
    assert_eq!(refs[0].bounds(), Some((vec![1, 2], vec![2, 4])));
    assert!(matches!(
        refs[0].selection(),
        Some(Selection::Hyperslab { .. })
    ));
    assert_eq!(refs[1].path(), Some("/matrix"));
    assert_eq!(refs[1].bounds(), Some((vec![0, 1], vec![3, 5])));
    assert_eq!(
        refs[1].selection(),
        Some(&Selection::Points(PointSelection {
            rank: 2,
            points: vec![vec![0, 1], vec![3, 5]],
        }))
    );
    drop(file);
    cleanup(&path);
}

/// An `H5T_STD_REF` dataset written here holds `H5R_OBJECT2` elements that
/// both this crate and libhdf5 follow to their targets.
///
/// The addresses come from the same finalize pass that writes the headers they
/// name, so this is also the case that says a reference element is stamped
/// after every object header has an address.
#[test]
fn object2_references_written_here_resolve_in_libhdf5() {
    let path = write_path("object2");
    let file = H5File::options()
        .libver(LibverBound::V112)
        .create(&path)
        .unwrap();
    let matrix = file
        .new_dataset::<i32>()
        .shape([4])
        .create("matrix")
        .unwrap();
    matrix.write_raw(&[10i32, 20, 30, 40]).unwrap();
    file.create_group("grp").unwrap();
    let refs = file
        .new_dataset::<u64>()
        .std_object_references()
        .shape([2])
        .create("objrefs")
        .unwrap();
    refs.write_object_references(&["/matrix", "/grp"]).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let read = file.dataset("objrefs").unwrap().read_references().unwrap();
    let paths: Vec<Option<&str>> = read.iter().map(Reference::path).collect();
    assert_eq!(paths, vec![Some("/matrix"), Some("/grp")]);
    assert!(read.iter().all(|r| matches!(r, Reference::Object { .. })));
    drop(file);

    if let Some(h5dump) = h5dump() {
        let out = std::process::Command::new(&h5dump)
            .args(["-d", "/objrefs", path.to_str().unwrap()])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "h5dump failed:\n{text}");
        assert!(text.contains("H5T_REFERENCE { H5T_STD_REF }"), "{text}");
        // h5dump prints a dereferenced element as the object it names, and a
        // dataset with its data, so the values prove it followed the address.
        assert!(text.contains("DATASET \""), "{text}");
        assert!(text.contains("10, 20, 30, 40"), "{text}");
        assert!(text.contains("GROUP \""), "{text}");
    }
    cleanup(&path);
}

/// An `H5R_ATTR` names an object and one of its attributes.
#[test]
fn attribute_references_name_the_attribute() {
    let path = write_temp("attr");
    let file = H5File::open(&path).unwrap();
    let refs = file.dataset("attrrefs").unwrap().read_references().unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].path(), Some("/matrix"));
    assert_eq!(refs[0].attribute_name(), Some("note"));
    // The path and the name together reach the attribute the reference means.
    // `H5File::dataset` names datasets relative to the root, while a reference
    // reports the absolute path, so the leading separator comes off here.
    let target = file
        .dataset(refs[0].path().unwrap().trim_start_matches('/'))
        .unwrap();
    let note = target.attr(refs[0].attribute_name().unwrap()).unwrap();
    assert_eq!(note.read_numeric_as::<i32>().unwrap(), vec![7, 8, 9]);
    drop(file);
    cleanup(&path);
}

/// A `H5R_DATASET_REGION2` written here is a heap blob whose token this crate
/// stamps at finalize; both this crate and libhdf5 follow it to the region it
/// names.
#[test]
fn region2_references_written_here_resolve_in_libhdf5() {
    let path = write_path("region2");
    let file = H5File::options()
        .libver(LibverBound::V112)
        .create(&path)
        .unwrap();
    let matrix = file
        .new_dataset::<i32>()
        .shape([4, 6])
        .create("matrix")
        .unwrap();
    matrix.write_raw(&(0..24).collect::<Vec<i32>>()).unwrap();
    let refs = file
        .new_dataset::<u64>()
        .std_region_references()
        .shape([3])
        .create("regrefs")
        .unwrap();
    let rows = Selection::Hyperslab {
        rank: 2,
        form: Hyperslab::Regular(RegularHyperslab {
            start: vec![1, 2],
            stride: vec![1, 1],
            count: vec![2, 3],
            block: vec![1, 1],
        }),
    };
    let points = Selection::Points(PointSelection {
        rank: 2,
        points: vec![vec![0, 1], vec![3, 5]],
    });
    let whole = Selection::All;
    refs.write_std_region_references(&[("/matrix", rows), ("/matrix", points), ("/matrix", whole)])
        .unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let read = file.dataset("regrefs").unwrap().read_references().unwrap();
    assert_eq!(read.len(), 3);
    assert_eq!(read[0].path(), Some("/matrix"));
    assert_eq!(read[0].bounds(), Some((vec![1, 2], vec![2, 4])));
    assert_eq!(read[1].path(), Some("/matrix"));
    assert_eq!(read[1].bounds(), Some((vec![0, 1], vec![3, 5])));
    // `H5S_SEL_ALL` carries no rank of its own; the blob says the target's,
    // which is what libhdf5 rebuilds the dataspace from.
    assert_eq!(read[2].path(), Some("/matrix"));
    assert_eq!(read[2].selection(), Some(&Selection::All));
    drop(file);

    if let Some(h5dump) = h5dump() {
        let out = std::process::Command::new(&h5dump)
            .args(["-d", "/regrefs", path.to_str().unwrap()])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "h5dump failed:\n{text}");
        assert!(text.contains("H5T_REFERENCE { H5T_STD_REF }"), "{text}");
        // h5dump prints a dereferenced region reference as the target's name
        // followed by the selection it holds, so both lines prove libhdf5 read
        // the blob this crate wrote.
        assert!(text.contains("REGION_TYPE BLOCK"), "{text}");
        assert!(text.contains("(1,2)-(2,4)"), "{text}");
        assert!(text.contains("REGION_TYPE POINT"), "{text}");
        assert!(text.contains("(0,1)"), "{text}");
        assert!(text.contains("(3,5)"), "{text}");
        // All three elements dereference; the whole-extent one prints its
        // target and no region, which is what h5dump prints for libhdf5's own
        // `H5S_SEL_ALL` region references.
        assert_eq!(text.matches("/matrix\"").count(), 3, "{text}");
        assert_eq!(text.matches("REGION_TYPE").count(), 2, "{text}");
    }
    cleanup(&path);
}

/// An `H5R_ATTR` written here names the attribute libhdf5 dereferences it to.
#[test]
fn attribute_references_written_here_resolve_in_libhdf5() {
    let path = write_path("attr");
    let file = H5File::options()
        .libver(LibverBound::V112)
        .create(&path)
        .unwrap();
    let matrix = file
        .new_dataset::<i32>()
        .shape([4])
        .create("matrix")
        .unwrap();
    matrix.write_raw(&[10i32, 20, 30, 40]).unwrap();
    matrix
        .new_attr::<i32>()
        .shape([3])
        .create("note")
        .unwrap()
        .write_array(&[7i32, 8, 9])
        .unwrap();
    let grp = file.create_group("grp").unwrap();
    grp.set_attr_array_numeric("tag", &[1i32, 2]).unwrap();
    let refs = file
        .new_dataset::<u64>()
        .attribute_references()
        .shape([2])
        .create("attrrefs")
        .unwrap();
    refs.write_attribute_references(&[("/matrix", "note"), ("/grp", "tag")])
        .unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let read = file.dataset("attrrefs").unwrap().read_references().unwrap();
    assert_eq!(read.len(), 2);
    assert_eq!(read[0].path(), Some("/matrix"));
    assert_eq!(read[0].attribute_name(), Some("note"));
    assert_eq!(read[1].path(), Some("/grp"));
    assert_eq!(read[1].attribute_name(), Some("tag"));
    let note = file.dataset("matrix").unwrap().attr("note").unwrap();
    assert_eq!(note.read_numeric_as::<i32>().unwrap(), vec![7, 8, 9]);
    drop(file);

    if let Some(h5dump) = h5dump() {
        let out = std::process::Command::new(&h5dump)
            .args(["-d", "/attrrefs", path.to_str().unwrap()])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "h5dump failed:\n{text}");
        assert!(text.contains("H5T_REFERENCE { H5T_STD_REF }"), "{text}");
        // h5dump dereferences an attribute reference to the attribute itself:
        // the owner's path, the attribute's name, and the type and shape it
        // found there.
        assert!(text.contains("/matrix/note\""), "{text}");
        assert!(text.contains("/grp/tag\""), "{text}");
        assert_eq!(text.matches("ATTRIBUTE \"").count(), 2, "{text}");
        assert!(text.contains("SIMPLE { ( 3 ) / ( 3 ) }"), "{text}");
        assert!(text.contains("SIMPLE { ( 2 ) / ( 2 ) }"), "{text}");
    }
    cleanup(&path);
}

/// The three rules a revised reference write applies before anything reaches
/// the file: the datatype has to be `H5T_STD_REF`, a region target has to be a
/// dataset, and an attribute reference has to name an attribute that exists.
#[test]
fn revised_reference_writes_refuse_what_h5r_refuses() {
    let path = write_path("refused");
    let file = H5File::options()
        .libver(LibverBound::V112)
        .create(&path)
        .unwrap();
    file.new_dataset::<i32>().shape([4]).create("m").unwrap();
    file.create_group("g").unwrap();

    let whole = Selection::Hyperslab {
        rank: 1,
        form: Hyperslab::Regular(RegularHyperslab {
            start: vec![0],
            stride: vec![1],
            count: vec![4],
            block: vec![1],
        }),
    };

    // The pre-1.12 region datatype does not hold 1.12 elements.
    let legacy = file
        .new_dataset::<u64>()
        .region_references()
        .shape([1])
        .create("legacy")
        .unwrap();
    assert!(legacy
        .write_std_region_references(&[("/m", whole.clone())])
        .is_err());

    let refs = file
        .new_dataset::<u64>()
        .std_region_references()
        .shape([2])
        .create("refs")
        .unwrap();
    // A group is not a region.
    assert!(refs
        .write_std_region_references(&[("/g", whole.clone())])
        .is_err());
    // A selection its target's extent does not admit.
    let past_end = Selection::Hyperslab {
        rank: 1,
        form: Hyperslab::Regular(RegularHyperslab {
            start: vec![2],
            stride: vec![1],
            count: vec![4],
            block: vec![1],
        }),
    };
    assert!(refs
        .write_std_region_references(&[("/m", past_end)])
        .is_err());
    // An attribute that does not exist, and an object that does not exist.
    assert!(refs
        .write_attribute_references(&[("/m", "missing")])
        .is_err());
    assert!(refs
        .write_attribute_references(&[("/nope", "note")])
        .is_err());
    // What is left is written, and nothing the refusals touched is on disk.
    refs.write_std_region_references(&[("/m", whole)]).unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    let read = file.dataset("refs").unwrap().read_references().unwrap();
    assert_eq!(read[0].bounds(), Some((vec![0], vec![3])));
    assert_eq!(
        read[1],
        Reference::Null,
        "the second element was never written"
    );
    drop(file);
    cleanup(&path);
}

/// The name an external reference in `ext_refs.h5` carries, and the path the
/// fixture pair is read back by. `H5R__reopen_file` opens the name a reference
/// carries verbatim against the process working directory, with no prefix
/// search of any kind (H5Rint.c:466, :487), so the target has to be where the
/// generator created it: `tests/fixtures/`, relative to the package root Cargo
/// runs a test binary in.
const EXT_TARGET: &str = "tests/fixtures/ext_ref_target.h5";

/// The holder file, whose three datasets reference into `EXT_TARGET`.
const EXT_HOLDER: &str = "tests/fixtures/ext_refs.h5";

/// All three 1.12 kinds cross files, and each names the file it crosses into
/// as well as the path inside it.
#[test]
fn external_references_name_the_file_they_cross_into() {
    let file = H5File::open(EXT_HOLDER).unwrap();

    let objs = file
        .dataset("extobjrefs")
        .unwrap()
        .read_references()
        .unwrap();
    assert_eq!(
        objs.iter().map(Reference::file).collect::<Vec<_>>(),
        vec![Some(EXT_TARGET), Some(EXT_TARGET)]
    );
    assert_eq!(
        objs.iter().map(Reference::path).collect::<Vec<_>>(),
        vec![Some("/matrix"), Some("/grp")]
    );

    let regs = file
        .dataset("extregrefs")
        .unwrap()
        .read_references()
        .unwrap();
    assert_eq!(regs[0].file(), Some(EXT_TARGET));
    assert_eq!(regs[0].path(), Some("/matrix"));
    assert_eq!(regs[0].bounds(), Some((vec![1, 2], vec![2, 4])));

    let attrs = file
        .dataset("extattrrefs")
        .unwrap()
        .read_references()
        .unwrap();
    assert_eq!(attrs[0].file(), Some(EXT_TARGET));
    assert_eq!(attrs[0].path(), Some("/matrix"));
    assert_eq!(attrs[0].attribute_name(), Some("note"));

    // The three together reach the value: open the file the reference names,
    // then the path and attribute name inside it.
    let target = H5File::open(attrs[0].file().unwrap()).unwrap();
    let matrix = target
        .dataset(attrs[0].path().unwrap().trim_start_matches('/'))
        .unwrap();
    let note = matrix.attr(attrs[0].attribute_name().unwrap()).unwrap();
    assert_eq!(note.read_numeric_as::<i32>().unwrap(), vec![7, 8, 9]);
}

/// A reference whose file is not there still says which file it wanted.
///
/// `H5Rget_file_name` answers from the reference alone and needs no open,
/// while `H5Ropen_object` fails (H5R.c:1036-1039) — which is what `h5dump`
/// prints as `UNKNOWN "<the file name>"` for this same file.
#[test]
fn an_external_reference_to_an_absent_file_still_names_it() {
    // The fixture with its file name pointed at a name nothing is under. The
    // replacement is the same length, so nothing else in the file moves, and
    // the heap the name sits in carries no checksum to invalidate.
    let (was, now) = (b"ext_ref_target.h5", b"ext_ref_absent.h5");
    let mut bytes = std::fs::read(EXT_HOLDER).unwrap();
    let mut hits = 0;
    for at in 0..bytes.len() - was.len() {
        if &bytes[at..at + was.len()] == was {
            bytes[at..at + now.len()].copy_from_slice(now);
            hits += 1;
        }
    }
    assert_eq!(hits, 4, "one name per reference in the fixture");
    let path = write_path("ext_absent");
    std::fs::write(&path, &bytes).unwrap();

    let file = H5File::open(&path).unwrap();
    let refs = file
        .dataset("extobjrefs")
        .unwrap()
        .read_references()
        .unwrap();
    assert_eq!(refs[0].file(), Some("tests/fixtures/ext_ref_absent.h5"));
    assert_eq!(refs[0].path(), None, "nothing is under that name");
    assert_eq!(refs[0].address(), Some(0x320));
    drop(file);
    cleanup(&path);
}
