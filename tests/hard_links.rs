//! Integration tests for hard-link creation (`H5Group::link`) and for
//! deletion against hard links (`H5Ldelete` semantics).
//!
//! A hard link gives an existing object a second name without copying its
//! data — the NeXus-style way to expose a dataset at `/entry/data/data`
//! while it physically lives elsewhere. Both names must resolve to
//! byte-identical data, and the reader must enumerate the aliased path.
//! Deleting a name only unlinks it: the object survives under a
//! remaining hard link, and its storage is freed with the last name.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::H5File;

/// Per-test unique temp path so parallel cargo runs cannot collide.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_hard_links_{}_{}_{}",
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

/// A hard link in a subgroup must resolve to the same data as the target,
/// and the reader must enumerate the aliased path.
#[test]
fn hard_link_to_dataset_shares_data() {
    let path = unique_tmp("hl_dataset");
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();

        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<f32>()
            .shape([12])
            .create("detector")
            .unwrap();
        ds.write_raw(&data).unwrap();

        // NeXus-style alias: /data/detector -> /instrument/detector.
        let data_grp = root.create_group("data").unwrap();
        data_grp.link("detector", "/instrument/detector").unwrap();

        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();

        let original = file
            .dataset("instrument/detector")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();
        let aliased = file
            .dataset("data/detector")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();

        assert_eq!(original, data, "target reads back the written data");
        assert_eq!(aliased, data, "hard link resolves to the same data");
    }

    cleanup(&path);
}

/// A hard link can live in the root group and point at a nested dataset.
#[test]
fn hard_link_in_root_group() {
    let path = unique_tmp("hl_root");
    let data: Vec<i32> = vec![7, 8, 9];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([3])
            .create("counts")
            .unwrap();
        ds.write_raw(&data).unwrap();

        root.link("counts_alias", "instrument/counts").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let aliased = file
            .dataset("counts_alias")
            .unwrap()
            .read_raw::<i32>()
            .unwrap();
        assert_eq!(aliased, data);
    }

    cleanup(&path);
}

/// Linking to a non-existent target is rejected.
#[test]
fn hard_link_rejects_unknown_target() {
    let path = unique_tmp("hl_unknown");
    let file = H5File::create(&path).unwrap();
    let err = file
        .root_group()
        .link("alias", "/does/not/exist")
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("not found"), "unexpected error: {msg}");
    drop(file);
    cleanup(&path);
}

/// A link name that already exists in the parent group is rejected.
#[test]
fn hard_link_rejects_duplicate_name() {
    let path = unique_tmp("hl_dup");
    let file = H5File::create(&path).unwrap();
    let root = file.root_group();
    let inst = root.create_group("instrument").unwrap();
    inst.new_dataset::<f32>()
        .shape([4])
        .create("detector")
        .unwrap();

    // "detector" already names a dataset in /instrument.
    let err = inst.link("detector", "/instrument/detector").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("already exists"), "unexpected error: {msg}");

    drop(file);
    cleanup(&path);
}

/// Creating a dataset whose name a hard link already occupies is rejected
/// (the reverse order of `hard_link_rejects_duplicate_name`): otherwise the
/// parent group would carry two link records with the same name.
#[test]
fn dataset_rejects_name_taken_by_hard_link() {
    let path = unique_tmp("hl_reverse");
    let file = H5File::create(&path).unwrap();
    let root = file.root_group();
    let inst = root.create_group("instrument").unwrap();
    inst.new_dataset::<f32>()
        .shape([4])
        .create("detector")
        .unwrap();

    let data = root.create_group("data").unwrap();
    data.link("detector", "/instrument/detector").unwrap();

    // /data/detector is already a hard link; a dataset there must fail.
    let ds_result = data.new_dataset::<f32>().shape([4]).create("detector");
    let err = ds_result
        .err()
        .expect("dataset creation should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("already exists"), "unexpected error: {msg}");

    // ...and so must a group of the same name.
    let grp_result = data.create_group("detector");
    let err = grp_result.err().expect("group creation should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("already exists"), "unexpected error: {msg}");

    drop(file);
    cleanup(&path);
}

/// A hard link can be created through the public SWMR writer API. Created
/// before `start_swmr`, it is committed with the streaming layout and
/// resolves to the target's data after close.
#[test]
fn swmr_writer_creates_hard_link() {
    use rust_hdf5::swmr::{SwmrFileReader, SwmrFileWriter};

    let path = unique_tmp("hl_swmr");
    {
        let mut w = SwmrFileWriter::create(&path).unwrap();
        let ds = w.create_streaming_dataset::<u8>("frames", &[2, 2]).unwrap();
        // Layout alias created before start_swmr -> visible for the whole run.
        w.create_hard_link("/", "alias", "frames").unwrap();
        w.start_swmr().unwrap();
        w.append_frame(ds, &[1u8, 2, 3, 4]).unwrap();
        w.close().unwrap();
    }

    let mut r = SwmrFileReader::open(&path).unwrap();
    let names = r.dataset_names();
    assert!(
        names.iter().any(|n| n == "alias"),
        "hard link 'alias' missing: {names:?}"
    );
    assert_eq!(r.read_dataset_raw("frames").unwrap(), vec![1u8, 2, 3, 4]);
    assert_eq!(r.read_dataset_raw("alias").unwrap(), vec![1u8, 2, 3, 4]);

    cleanup(&path);
}

/// The public SWMR writer API can build a nested NeXus-style layout: groups
/// tagged with `NX_class` attributes plus a hard link aliasing a streaming
/// dataset into that layout. All structure is created before `start_swmr`.
#[test]
fn swmr_writer_builds_nexus_layout() {
    use rust_hdf5::swmr::SwmrFileWriter;

    let path = unique_tmp("swmr_nexus");
    {
        let mut w = SwmrFileWriter::create(&path).unwrap();
        let ds = w
            .create_streaming_dataset::<u16>("frames", &[2, 2])
            .unwrap();

        // NeXus group tree: /entry (NXentry) -> /entry/data (NXdata).
        w.create_group("/", "entry").unwrap();
        w.create_group("/entry", "data").unwrap();
        w.set_group_attr_string("/entry", "NX_class", "NXentry")
            .unwrap();
        w.set_group_attr_string("/entry/data", "NX_class", "NXdata")
            .unwrap();
        // Alias the streaming dataset at the NeXus canonical location.
        w.create_hard_link("/entry/data", "data", "frames").unwrap();

        w.start_swmr().unwrap();
        // One frame of 4 u16 values, little-endian: 1, 2, 3, 4.
        w.append_frame(ds, &[1u8, 0, 2, 0, 3, 0, 4, 0]).unwrap();
        w.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let root = file.root_group();

    let entry = root.group("entry").unwrap();
    assert_eq!(entry.attr_string("NX_class").unwrap(), "NXentry");

    let data = entry.group("data").unwrap();
    assert_eq!(data.attr_string("NX_class").unwrap(), "NXdata");

    // The hard link resolves to the streaming dataset's data.
    let aliased = file
        .dataset("entry/data/data")
        .unwrap()
        .read_raw::<u16>()
        .unwrap();
    assert_eq!(aliased, vec![1u16, 2, 3, 4]);

    cleanup(&path);
}

/// Deleting a hard-linked dataset's tree name must not destroy the
/// object: the link is promoted to the primary name, the data stays
/// readable there, and the old path is gone.
#[test]
fn deleting_primary_name_promotes_hard_link() {
    let path = unique_tmp("hl_del_promote");
    let data: Vec<i32> = (0..8).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([8])
            .create("detector")
            .unwrap();
        ds.write_raw(&data).unwrap();

        let data_grp = root.create_group("data").unwrap();
        data_grp.link("detector", "/instrument/detector").unwrap();

        file.delete_dataset("instrument/detector").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let survived = file
            .dataset("data/detector")
            .unwrap()
            .read_raw::<i32>()
            .unwrap();
        assert_eq!(survived, data, "object survives under the hard link");
        assert!(
            file.dataset("instrument/detector").is_err(),
            "the deleted name must not resolve"
        );
    }

    cleanup(&path);
}

/// Deleting the *last* name frees the storage: cycles of
/// create → link → delete-primary → delete-promoted-name settle to a
/// fixed file size (the oracle of `delete_reclamation.rs`).
#[test]
fn deleting_last_name_frees_storage() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("hl_del_free_{cycles}"));
        let vals: Vec<i32> = (0..1024).collect();
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let data_grp = root.create_group("data").unwrap();
        for _ in 0..cycles {
            let ds = inst
                .new_dataset::<i32>()
                .shape([1024])
                .create("detector")
                .unwrap();
            ds.write_raw(&vals).unwrap();
            data_grp.link("detector", "/instrument/detector").unwrap();
            // First delete only unlinks (the link survives as
            // /data/detector); the second removes the last name.
            file.delete_dataset("instrument/detector").unwrap();
            file.delete_dataset("data/detector").unwrap();
        }
        file.close().unwrap();
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(10), size_after(2), "10 cycles against 2");
}

/// A hard link from outside the subtree naming an inner *group* refuses
/// the whole `delete_group`, and the refusal leaves the file untouched.
#[test]
fn delete_group_refused_by_outside_link_to_inner_group() {
    let path = unique_tmp("hl_del_refuse");
    let data: Vec<i32> = vec![5, 6, 7];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let container = root.create_group("container").unwrap();
        let inner = container.create_group("inner").unwrap();
        let ds = inner.new_dataset::<i32>().shape([3]).create("ds").unwrap();
        ds.write_raw(&data).unwrap();

        root.link("inner_alias", "/container/inner").unwrap();

        let err = file.delete_group("container").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("delete the link's parent group first"),
            "unexpected error: {msg}"
        );
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("container/inner/ds")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            data,
            "refused delete must leave the subtree intact"
        );
    }

    cleanup(&path);
}

/// `delete_group` re-homes an inner dataset a hard link from outside
/// still names: the alias survives close, the rest of the subtree is
/// gone.
#[test]
fn delete_group_promotes_outside_linked_dataset() {
    let path = unique_tmp("hl_del_rehome");
    let keep: Vec<i32> = (0..6).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let doomed = root.create_group("doomed").unwrap();
        let ds = doomed
            .new_dataset::<i32>()
            .shape([6])
            .create("keep")
            .unwrap();
        ds.write_raw(&keep).unwrap();
        doomed
            .new_dataset::<i32>()
            .shape([4])
            .create("gone")
            .unwrap();

        root.link("survivor", "/doomed/keep").unwrap();

        file.delete_group("doomed").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("survivor").unwrap().read_raw::<i32>().unwrap(),
            keep,
            "outside-linked dataset survives its container"
        );
        assert!(
            file.dataset("doomed/keep").is_err(),
            "old path inside the deleted group must not resolve"
        );
        assert!(
            file.dataset("doomed/gone").is_err(),
            "unlinked sibling must be gone"
        );
    }

    cleanup(&path);
}

/// Deleting a path that names a hard link removes just that link: the
/// target dataset and its tree name are untouched.
#[test]
fn deleting_a_link_path_unlinks_only_the_link() {
    let path = unique_tmp("hl_del_link");
    let data: Vec<i32> = (0..5).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([5])
            .create("detector")
            .unwrap();
        ds.write_raw(&data).unwrap();
        let data_grp = root.create_group("data").unwrap();
        data_grp.link("detector", "/instrument/detector").unwrap();

        file.delete_dataset("data/detector").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("instrument/detector")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            data,
            "the tree name must survive its link's deletion"
        );
        assert!(
            file.dataset("data/detector").is_err(),
            "the deleted link path must not resolve"
        );
    }

    cleanup(&path);
}

/// Deleting a group-link path unlinks it too — clearing the outside link
/// that made `delete_group` refuse.
#[test]
fn deleting_a_group_link_path_clears_the_refusal() {
    let path = unique_tmp("hl_del_glink");

    let file = H5File::create(&path).unwrap();
    let root = file.root_group();
    let container = root.create_group("container").unwrap();
    let inner = container.create_group("inner").unwrap();
    inner.new_dataset::<i32>().shape([2]).create("ds").unwrap();
    root.link("inner_alias", "/container/inner").unwrap();

    assert!(
        file.delete_group("container").is_err(),
        "outside link must refuse the subtree delete"
    );
    file.delete_group("inner_alias").unwrap();
    file.delete_group("container").unwrap();

    drop(file);
    cleanup(&path);
}

/// `dataset_writer` resolves a hard-link path to its target, like
/// `H5Dopen`: a write through the alias lands in the one object.
#[test]
fn dataset_writer_resolves_a_hard_link_path() {
    let path = unique_tmp("hl_write_alias");
    let data: Vec<i32> = (0..4).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        inst.new_dataset::<i32>()
            .shape([4])
            .chunk(&[2])
            .max_shape(&[None])
            .create("detector")
            .unwrap();
        let data_grp = root.create_group("data").unwrap();
        data_grp.link("detector", "/instrument/detector").unwrap();

        let ds = file.dataset_writer("data/detector").unwrap();
        ds.write_slice(&[0], &[4], &data).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("instrument/detector")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            data,
            "a write through the alias must land in the target"
        );
    }

    cleanup(&path);
}

/// A hard link whose target path is itself a hard link points straight at
/// the object (links have no chain): it survives both other names.
#[test]
fn hard_link_to_a_link_path_targets_the_object() {
    let path = unique_tmp("hl_chain");
    let data: Vec<i32> = vec![41, 42];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let ds = root.new_dataset::<i32>().shape([2]).create("ds").unwrap();
        ds.write_raw(&data).unwrap();
        root.link("alias1", "ds").unwrap();
        root.link("alias2", "alias1").unwrap();

        file.delete_dataset("ds").unwrap();
        file.delete_dataset("alias1").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("alias2").unwrap().read_raw::<i32>().unwrap(),
            data,
            "the last name must keep the object"
        );
    }

    cleanup(&path);
}

/// A reopened file rebuilds hard-link identity: the alias and the tree
/// name are one object again, so deleting one name keeps the data alive
/// under the other. Reopen used to give every alias its own
/// `DatasetInfo` with the same storage addresses — deleting either path
/// freed blocks the other still referenced.
#[test]
fn reopened_file_keeps_hard_link_identity() {
    let path = unique_tmp("hl_reopen");
    let data: Vec<i32> = (0..10).collect();

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([10])
            .create("detector")
            .unwrap();
        ds.write_raw(&data).unwrap();
        let data_grp = root.create_group("data").unwrap();
        data_grp.link("detector", "/instrument/detector").unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        file.delete_dataset("instrument/detector").unwrap();
        // Anything the delete wrongly freed gets reused here with other
        // bytes — under the old per-alias DatasetInfo reopen, this filler
        // landed in the object's storage and the alias read it back.
        let filler: Vec<i32> = (100..110).collect();
        let f = file
            .root_group()
            .new_dataset::<i32>()
            .shape([10])
            .create("filler")
            .unwrap();
        f.write_raw(&filler).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("data/detector")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            data,
            "the other name must keep the object across sessions"
        );
        assert!(
            file.dataset("instrument/detector").is_err(),
            "the deleted name must not resolve"
        );
    }

    cleanup(&path);
}

/// Reopening and closing without changes keeps both names resolving.
#[test]
fn reopen_close_preserves_hard_links() {
    let path = unique_tmp("hl_reopen_noop");
    let data: Vec<i32> = vec![9, 8, 7];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let ds = root.new_dataset::<i32>().shape([3]).create("ds").unwrap();
        ds.write_raw(&data).unwrap();
        root.link("alias", "ds").unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert_eq!(file.dataset("ds").unwrap().read_raw::<i32>().unwrap(), data);
        assert_eq!(
            file.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
            data
        );
    }

    cleanup(&path);
}

/// Cross-session last-name deletes settle the file size: cycles of
/// reopen → delete both names → recreate dataset + link do not grow the
/// file, so the freed storage is really recovered.
#[test]
fn reopen_delete_both_names_settles_file_size() {
    let size_after = |cycles: usize| {
        let path = unique_tmp(&format!("hl_reopen_free_{cycles}"));
        let vals: Vec<i32> = (0..256).collect();
        let create = |file: &H5File| {
            let root = file.root_group();
            let ds = root.new_dataset::<i32>().shape([256]).create("ds").unwrap();
            ds.write_raw(&vals).unwrap();
            root.link("alias", "ds").unwrap();
        };
        {
            let file = H5File::create(&path).unwrap();
            create(&file);
            file.close().unwrap();
        }
        for _ in 0..cycles {
            let file = H5File::options().no_locking().open_rw(&path).unwrap();
            file.delete_dataset("ds").unwrap();
            file.delete_dataset("alias").unwrap();
            create(&file);
            file.close().unwrap();
        }
        let read = H5File::open(&path).unwrap();
        assert_eq!(
            read.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
            vals
        );
        drop(read);
        let n = std::fs::metadata(&path).unwrap().len();
        cleanup(&path);
        n
    };

    assert_eq!(size_after(10), size_after(2), "10 reopen cycles against 2");
}

/// A hard link to a *group* survives reopen too: it still refuses the
/// subtree delete, and unlinking it first clears the way.
#[test]
fn reopened_group_link_still_refuses_subtree_delete() {
    let path = unique_tmp("hl_reopen_group");

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let container = root.create_group("container").unwrap();
        let inner = container.create_group("inner").unwrap();
        inner.new_dataset::<i32>().shape([2]).create("ds").unwrap();
        root.link("inner_alias", "/container/inner").unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::options().no_locking().open_rw(&path).unwrap();
        assert!(
            file.delete_group("container").is_err(),
            "the reopened group link must still refuse the delete"
        );
        file.delete_group("inner_alias").unwrap();
        file.delete_group("container").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        assert!(
            file.dataset("container/inner/ds").is_err(),
            "the subtree must be gone after the cleared delete"
        );
    }

    cleanup(&path);
}

/// A target path given with a trailing slash still resolves.
#[test]
fn hard_link_tolerates_trailing_slash() {
    let path = unique_tmp("hl_trailing");
    let data: Vec<i32> = vec![3, 1, 4, 1, 5];

    {
        let file = H5File::create(&path).unwrap();
        let root = file.root_group();
        let inst = root.create_group("instrument").unwrap();
        let ds = inst
            .new_dataset::<i32>()
            .shape([5])
            .create("counts")
            .unwrap();
        ds.write_raw(&data).unwrap();

        // Leading and trailing slashes both tolerated.
        root.link("alias", "/instrument/counts/").unwrap();
        file.close().unwrap();
    }

    {
        let file = H5File::open(&path).unwrap();
        let aliased = file.dataset("alias").unwrap().read_raw::<i32>().unwrap();
        assert_eq!(aliased, data);
    }

    cleanup(&path);
}
