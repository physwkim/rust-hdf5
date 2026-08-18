//! Persisted free-space managers across a reopen.
//!
//! A file whose file-space info message says `persist` records every free
//! region in on-disk managers (`FSHD` header, `FSSE` sections). Nothing else
//! in the file says those regions are free, so a session that rewrites the
//! file without reading them leaks what it frees and grows a file that had
//! room. These cases take a file libhdf5 wrote with `fs_persist`, append to
//! it, and ask libhdf5 itself — `h5stat -S`, `h5clear -s`, h5py — whether the
//! account still adds up.

use rust_hdf5::{FileSpaceStrategy, H5File};

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`. `h5stat` and `h5clear` are taken from the same
/// directory: they are the tools of the libhdf5 that h5py is linked against.
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
            eprintln!("skipping free-space cross-check: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
}

fn h5_tool(py: &str, name: &str) -> Option<std::path::PathBuf> {
    let tool = std::path::Path::new(py).parent()?.join(name);
    tool.exists().then_some(tool)
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_fsm_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run(program: impl AsRef<std::ffi::OsStr>, args: &[&str], what: &str) -> String {
    let out = std::process::Command::new(&program)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed ({}):\n{}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `h5stat -S`'s two numbers: what the free-space managers account for, and
/// what nothing in the file accounts for. The leak this file is about shows up
/// entirely in the second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpaceAccount {
    tracked: u64,
    unaccounted: u64,
    total: u64,
}

fn h5stat_space(py: &str, path: &std::path::Path) -> SpaceAccount {
    let tool = h5_tool(py, "h5stat").expect("h5stat ships with this libhdf5");
    let out = run(tool, &["-S", path.to_str().unwrap()], "h5stat -S");
    let number = |label: &str| -> u64 {
        let line = out
            .lines()
            .find(|l| l.trim_start().starts_with(label))
            .unwrap_or_else(|| panic!("h5stat -S printed no {label:?} line:\n{out}"));
        line.split_whitespace()
            .find_map(|w| w.split('/').next().unwrap().parse::<u64>().ok())
            .unwrap_or_else(|| panic!("no number in {line:?}"))
    };
    SpaceAccount {
        tracked: number("Amount/Percent of tracked free space:"),
        unaccounted: number("Unaccounted space:"),
        total: number("Total space:"),
    }
}

fn h5clear_accepts(py: &str, path: &std::path::Path) {
    let tool = h5_tool(py, "h5clear").expect("h5clear ships with this libhdf5");
    run(tool, &["-s", path.to_str().unwrap()], "h5clear -s");
}

/// Write a file with the managers this is about: `fsm` strategy, persisted,
/// and a threshold of one byte so every freed block is recorded.
fn write_persisting_file(py: &str, path: &std::path::Path) {
    write_persisting_file_with(py, path, "fsm")
}

/// [`write_persisting_file`] under a named strategy — `fsm` or `page`.
fn write_persisting_file_with(py: &str, path: &std::path::Path, strategy: &str) {
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'w', fs_strategy='{strategy}', fs_persist=True, fs_threshold=1)\n\
         f.create_dataset('keep', data=np.arange(16, dtype='i4'))\n\
         f.create_group('grp').create_dataset('inner', data=np.arange(8, dtype='f8'))\n\
         f.create_dataset('drop', data=np.arange(64, dtype='i4'))\n\
         del f['drop']\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py write");
}

/// Read every dataset back and add one, all through libhdf5.
fn h5py_reads_and_appends(py: &str, path: &std::path::Path, name: &str) {
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'r+')\n\
         assert list(f['keep'][:]) == list(range(16)), list(f['keep'][:])\n\
         assert list(f['grp/inner'][:]) == list(range(8)), list(f['grp/inner'][:])\n\
         f.create_dataset('{name}', data=np.arange(4, dtype='i4'))\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back and append");
}

/// One small dataset through the public API, the smallest append that still
/// rewrites the root header, the superblock extension and the managers.
fn crate_appends(path: &std::path::Path, name: &str) {
    let file = H5File::open_rw(path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create(name)
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// A file libhdf5 wrote with persisted managers takes a crate append without
/// losing the account: the append comes out of the free space the managers
/// already tracked, so the file does not grow, and what the rewrite frees goes
/// back into managers `h5stat` still reads.
#[test]
fn a_crate_append_spends_and_rewrites_the_persisted_managers() {
    let Some(py) = python() else { return };
    let path = tmp("append");
    write_persisting_file(py, &path);

    let before = h5stat_space(py, &path);
    assert!(before.tracked > 0, "{before:?} has nothing to reuse");
    assert_eq!(
        before.unaccounted, 0,
        "libhdf5 left the file fully accounted"
    );

    crate_appends(&path, "added");
    let after = h5stat_space(py, &path);
    // Most of the append comes out of the space the managers named. What it
    // cannot take is the raw-data manager's sections for a metadata block:
    // `H5MF_alloc` asks `fs_man[fs_type]` and no other, so metadata that
    // outruns the metadata sections comes from the end of the file however
    // much raw-data space is free.
    assert!(
        after.total < before.total + before.tracked,
        "the append took nothing from the {} bytes free: {after:?}",
        before.tracked
    );
    assert!(
        after.tracked > 0,
        "the append left the file with no tracked free space: {after:?}"
    );
    assert_eq!(
        after.unaccounted, 0,
        "the append left space no manager records: {after:?}"
    );

    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, ["added", "by_libhdf5", "grp/inner", "keep"]);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        (0..16).collect::<Vec<i32>>()
    );
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// Every byte an append touches lands in something the file names or in a
/// section a manager records. `h5stat -S`'s "Unaccounted space" is libhdf5's
/// own count of the bytes in neither, and an append must leave it where
/// libhdf5 left it, at zero.
///
/// Two things put bytes outside the account before this. Rewriting an object
/// header freed its first chunk only, so the continuation chunk of a
/// multi-chunk header — the file libhdf5 writes here has one, in its
/// superblock extension — stayed allocated with nothing pointing at it. And
/// reusing part of a free block rounded the draw up to the allocator's
/// alignment, burying the difference inside the allocation, where no manager
/// can name it.
#[test]
fn an_append_leaves_no_byte_outside_the_account() {
    let Some(py) = python() else { return };
    let path = tmp("accounted");
    write_persisting_file(py, &path);
    assert_eq!(h5stat_space(py, &path).unaccounted, 0);

    crate_appends(&path, "added");
    let once = h5stat_space(py, &path);
    assert_eq!(once.unaccounted, 0, "{once:?}");

    // Again, so the second append reads back the managers the first wrote.
    crate_appends(&path, "again");
    let twice = h5stat_space(py, &path);
    assert_eq!(twice.unaccounted, 0, "{twice:?}");

    h5clear_accepts(py, &path);
    let _ = std::fs::remove_file(&path);
}

/// The second append reads managers this crate wrote rather than libhdf5's,
/// which is the only way to reach the encoding the write side produces. The
/// file stays one libhdf5 accepts and h5py can still add to.
#[test]
fn the_managers_the_crate_wrote_are_readable_by_both_libraries() {
    let Some(py) = python() else { return };
    let path = tmp("twice");
    write_persisting_file(py, &path);

    crate_appends(&path, "first");
    let once = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the file persists its free space");
    assert_ne!(once.fs_addr[0], u64::MAX, "no manager was written");

    crate_appends(&path, "second");
    let twice = h5stat_space(py, &path);
    assert!(twice.tracked > 0, "{twice:?}");
    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(
        names,
        ["by_libhdf5", "first", "grp/inner", "keep", "second"]
    );
    assert_eq!(
        file.dataset("grp/inner")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        (0..8).map(f64::from).collect::<Vec<f64>>()
    );
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// A file this crate creates with `fs_persist` carries the strategy, keeps
/// real managers once an append frees anything, and stays a file libhdf5
/// reads, checks and appends to.
#[test]
fn a_crate_created_persisting_file_is_one_libhdf5_accepts() {
    let Some(py) = python() else { return };
    let path = tmp("created");
    {
        let file = H5File::options()
            .file_space(FileSpaceStrategy::FsmAggr, true, 1)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([16usize])
            .create("keep")
            .unwrap()
            .write_raw(&(0..16i32).collect::<Vec<_>>())
            .unwrap();
        file.create_group("grp").unwrap();
        file.new_dataset::<f64>()
            .shape([8usize])
            .create("grp/inner")
            .unwrap()
            .write_raw(&(0..8).map(f64::from).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }

    let created = h5stat_space(py, &path);
    // What the creation freed is the alignment fragments before each block;
    // the managers record every one of them, so nothing in the file is
    // unaccounted for even before anything is deleted.
    assert_eq!(
        created.unaccounted, 0,
        "the created file leaks space: {created:?}"
    );
    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the created file declares its strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist);
    // `h5clear -s` opens the file read-write and closes it, and libhdf5's own
    // close drops the managers of a file it did not modify and shrinks the
    // file over their blocks. The sections they recorded that were not at the
    // end become libhdf5's own unaccounted space, so the append below is
    // measured against what libhdf5 left, not against zero.
    h5clear_accepts(py, &path);
    let cleared = h5stat_space(py, &path);

    // The append supersedes the root header and the extension; that is the
    // first free space this file has, and the managers are where it goes.
    crate_appends(&path, "added");
    let appended = h5stat_space(py, &path);
    assert!(
        appended.tracked > cleared.tracked,
        "the append recorded nothing: {appended:?}"
    );
    assert_eq!(
        appended.unaccounted, cleared.unaccounted,
        "the append leaked space on top of the {cleared:?} libhdf5 left"
    );
    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, ["added", "by_libhdf5", "grp/inner", "keep"]);
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// Freed metadata and freed raw data go to different managers, and both are
/// ones libhdf5 reads.
///
/// `H5MF_ALLOC_TO_FS_AGGR_TYPE` (H5MF.c:56) maps every allocation type through
/// the sec2 driver's `H5FD_FLMAP_DICHOTOMY` (H5FDsec2.c:157), which sends
/// `H5FD_MEM_DRAW` and `H5FD_MEM_GHEAP` to one manager and the other five types
/// to another. Their addresses are message slots 2 and 0, `H5F__super_read`
/// reading `fsinfo.fs_addr[u - 1]` into `f->shared->fs_addr[u]`
/// (H5Fsuper.c:831-833).
#[test]
fn a_delete_fills_both_of_the_managers_the_dichotomy_defines() {
    let Some(py) = python() else { return };
    let path = tmp("dichotomy");
    {
        let file = H5File::options()
            .file_space(FileSpaceStrategy::FsmAggr, true, 1)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([512usize])
            .create("bulk")
            .unwrap()
            .write_raw(&(0..512i32).collect::<Vec<_>>())
            .unwrap();
        file.new_dataset::<i32>()
            .shape([8usize])
            .create("keep")
            .unwrap()
            .write_raw(&(0..8i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        file.delete_dataset("bulk").unwrap();
        file.close().unwrap();
    }

    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the file declares its strategy");
    assert_ne!(info.fs_addr[0], u64::MAX, "no metadata manager: {info:?}");
    assert_ne!(info.fs_addr[2], u64::MAX, "no raw-data manager: {info:?}");
    for (slot, &addr) in info.fs_addr.iter().enumerate() {
        if slot != 0 && slot != 2 {
            assert_eq!(
                addr,
                u64::MAX,
                "slot {slot} names a manager libhdf5 would not"
            );
        }
    }

    // The two libraries agree on what those managers hold.
    let space = h5stat_space(py, &path);
    assert_eq!(
        H5File::open(&path).unwrap().tracked_free_space().unwrap(),
        space.tracked,
        "the crate and h5stat disagree about the tracked free space: {space:?}"
    );
    assert!(space.tracked > 0, "nothing was recorded: {space:?}");
    h5clear_accepts(py, &path);
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'r+')\n\
         assert list(f.keys()) == ['keep'], list(f.keys())\n\
         assert list(f['keep'][:]) == list(range(8)), list(f['keep'][:])\n\
         f.create_dataset('by_libhdf5', data=np.arange(4, dtype='i4'))\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back after the delete");
    let _ = std::fs::remove_file(&path);
}

/// `sohm_paged.h5` is the paged fixture, and it persists nothing: `gen_sohm.c`
/// passes `persist = 0` to `H5Pset_file_space_strategy`, so its file-space
/// info message names no manager and `h5stat -S` reports no tracked free space
/// at all. There is no free space to record for it, and an append must not
/// invent one.
#[test]
fn the_paged_fixture_persists_no_manager_to_rewrite() {
    let Some(py) = python() else { return };
    let path = tmp("paged");
    std::fs::copy(fixture("sohm_paged.h5"), &path).unwrap();

    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the paged fixture declares a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert!(!info.persist, "the fixture would have managers to read");
    let before = h5stat_space(py, &path);
    assert_eq!(before.tracked, 0);

    crate_appends(&path, "added");
    let after = h5stat_space(py, &path);
    assert_eq!(
        after.tracked, 0,
        "a file that persists nothing came back with a manager"
    );
    // Paged allocation is the file's whether or not it persists managers, so
    // the append still lands on the page grid and ends the file on it. It
    // grows by a page: with no managers on disk there is nothing that says
    // the space the fixture is not using is free, so this session cannot
    // reuse it any more than libhdf5 could.
    assert_eq!(after.total % info.page_size, 0, "{after:?}");
    assert_eq!(after.total, before.total + info.page_size, "{after:?}");
    h5clear_accepts(py, &path);
    // The fixture's own datasets, not the ones the persisting file has.
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'r+')\n\
         assert list(f['shared0'][:]) == list(range(8)), list(f['shared0'][:])\n\
         assert list(f['shared1'][:]) == list(range(10, 18)), list(f['shared1'][:])\n\
         assert f['uses_named'].dtype == f['named_i32'].dtype\n\
         f.create_dataset('by_libhdf5', data=np.arange(4, dtype='i4'))\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back and append");
    let _ = std::fs::remove_file(&path);
}

/// Rewrite a file's version-1 file-space info message as the deprecated
/// version-0 one, in place.
///
/// No writer produces a version-0 message: libhdf5 has encoded version 1 since
/// 1.10.1 (`H5O_fsinfo_set_version` starts at version 1 and only raises it,
/// and `H5O_fsinfo_ver_bounds` maps every `libver` bound to version 1 or to
/// "no such message"), so no `h5py.File(..., libver=...)` call can ask for the
/// older form. A 1.10.0-written file is the only source of one, and this is
/// the byte surgery that stands in for it.
///
/// The two bodies differ in length — 125 bytes against 58 — so the message the
/// version-0 one replaces is followed by a NIL message covering the rest of
/// the space it occupied, which is what libhdf5 itself leaves behind when it
/// shrinks a message (`H5O__release_mesg`). The chunk keeps its size and the
/// header's message count goes up by one.
fn downgrade_fsinfo_to_version_0(path: &std::path::Path) {
    const FSINFO: u16 = 0x0017;
    let mut bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n", "not an HDF5 file");
    assert!(
        bytes[8] >= 2,
        "a version-{} superblock has no extension",
        bytes[8]
    );
    let ext = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    assert_eq!(
        bytes[ext], 1,
        "the superblock extension is not a version-1 header"
    );
    let nmesgs = u16::from_le_bytes(bytes[ext + 2..ext + 4].try_into().unwrap());
    let chunk = u32::from_le_bytes(bytes[ext + 8..ext + 12].try_into().unwrap()) as usize;

    let mut pos = ext + 16;
    let end = pos + chunk;
    let (at, old_len, flags) = loop {
        assert!(
            pos + 8 <= end,
            "the extension carries no file-space info message"
        );
        let msg_type = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
        let len = u16::from_le_bytes(bytes[pos + 2..pos + 4].try_into().unwrap()) as usize;
        if msg_type == FSINFO {
            break (pos, len, bytes[pos + 4]);
        }
        pos += 8 + len;
    };

    // The version-1 body, by field: version, strategy, persist, threshold,
    // page size, page-end metadata threshold, EOA, then twelve addresses.
    let body = &bytes[at + 8..at + 8 + old_len];
    assert_eq!(body[0], 1, "the message is already at version {}", body[0]);
    assert_eq!(body[1], 0, "only the FSM_AGGR strategy predates version 1");
    assert_eq!(body[2], 1, "this fixture wants a persisting message");
    let threshold = &body[3..11];

    // The version-0 body: strategy 1 is H5F_FILE_SPACE_ALL_PERSIST, and its
    // six addresses are the first six of the twelve.
    let mut v0 = vec![0u8, 1u8];
    v0.extend_from_slice(threshold);
    v0.extend_from_slice(&body[29..29 + 6 * 8]);
    assert_eq!(v0.len(), 58);
    let v0_padded = 64;
    let nil_len = old_len - v0_padded - 8;

    let mut spliced = Vec::with_capacity(8 + old_len);
    spliced.extend_from_slice(&FSINFO.to_le_bytes());
    spliced.extend_from_slice(&(v0_padded as u16).to_le_bytes());
    spliced.push(flags);
    spliced.extend_from_slice(&[0u8; 3]);
    spliced.extend_from_slice(&v0);
    spliced.resize(8 + v0_padded, 0);
    spliced.extend_from_slice(&0u16.to_le_bytes()); // H5O_NULL_ID
    spliced.extend_from_slice(&(nil_len as u16).to_le_bytes());
    spliced.resize(8 + v0_padded + 8 + nil_len, 0);
    assert_eq!(spliced.len(), 8 + old_len);

    bytes[at..at + 8 + old_len].copy_from_slice(&spliced);
    bytes[ext + 2..ext + 4].copy_from_slice(&(nmesgs + 1).to_le_bytes());
    std::fs::write(path, &bytes).unwrap();
}

/// A file carrying the deprecated version-0 file-space info message takes an
/// append that leaves it a version-0 file.
///
/// libhdf5 would upgrade it: `H5O__fsinfo_decode` marks a version-0 message
/// `mapped`, and `H5F__super_read` then removes it and writes a version-1
/// replacement on read-write open (H5Fsuper.c:843-885). This crate re-emits
/// the form it read instead — a file's on-disk format is not this writer's to
/// change — and the case pins both halves of that: the crate leaves version 0
/// alone across two appends, and libhdf5 still reads, checks and appends to
/// what it wrote, upgrading it on its own terms when it does.
///
/// The file carries a shared-message table it does not otherwise need, to
/// steer around a libhdf5 1.14.6 crash that has nothing to do with this crate.
/// When the version-0 message is the *only* message in the extension,
/// `H5F__super_ext_remove_msg` finds the chunk all-NULL after taking it out
/// and calls `H5O_delete` on the extension itself (its `nchunks == 1` branch);
/// `H5F__super_ext_write_msg` then has to re-create the extension from inside
/// `H5F__super_read`, where `f->shared` is not yet set up, and segfaults. A
/// second message keeps the chunk non-empty and the upgrade on the ordinary
/// path.
#[test]
fn an_append_leaves_a_version_zero_file_space_message_at_version_zero() {
    let Some(py) = python() else { return };
    let path = tmp("fsinfo_v0");
    {
        // The shared-message table is here to keep libhdf5 alive, not
        // because the case is about it: see the note above on the
        // empty-extension crash. Everything asserted below is the file-space
        // message.
        use rust_hdf5::format::messages::{MSG_DATASPACE, MSG_DATATYPE};
        use rust_hdf5::format::sohm::type_flag;
        let types = type_flag(MSG_DATATYPE).unwrap() | type_flag(MSG_DATASPACE).unwrap();
        let file = H5File::options()
            .shared_messages(&[(types, 0)], 50, 40)
            .file_space(FileSpaceStrategy::FsmAggr, true, 1)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([64usize])
            .create("keep")
            .unwrap()
            .write_raw(&(0..64i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }
    // Managers first, so the version-0 message the surgery leaves behind names
    // real ones and the reopen below has something to read out of them.
    crate_appends(&path, "before_downgrade");
    let v1 = h5stat_space(py, &path);
    assert!(
        v1.tracked > 0,
        "nothing to carry into the older message: {v1:?}"
    );
    downgrade_fsinfo_to_version_0(&path);

    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the downgraded file still declares its strategy");
    assert_eq!(info.version, 0);
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist);
    assert_eq!(
        H5File::open(&path).unwrap().tracked_free_space().unwrap(),
        v1.tracked,
        "the older message lost sight of the managers it names"
    );

    // Read-only checks only until the appends are done: `h5clear -s` opens
    // read-write, which is the upgrade the last block below is for.
    for round in 0..2 {
        crate_appends(&path, &format!("appended{round}"));
        let info = H5File::open(&path)
            .unwrap()
            .superblock_extension()
            .file_space_info
            .expect("the append kept the message");
        assert_eq!(info.version, 0, "round {round} upgraded the message");
        assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
        assert!(info.persist);
        assert_ne!(
            info.fs_addr[0],
            u64::MAX,
            "round {round} left the file with no metadata manager"
        );
        let after = h5stat_space(py, &path);
        assert!(
            after.tracked > 0,
            "round {round} recorded nothing: {after:?}"
        );
    }

    // libhdf5's turn: it reads the version-0 message, and its own read-write
    // open rewrites it as version 1 — the upgrade this crate declines to make.
    h5clear_accepts(py, &path);
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File(r'{}', 'r+')\n\
         assert list(f['keep'][:]) == list(range(64)), list(f['keep'][:])\n\
         assert sorted(f.keys()) == ['appended0', 'appended1', 'before_downgrade', 'keep'], \
sorted(f.keys())\n\
         f.create_dataset('by_libhdf5', data=np.arange(4, dtype='i4'))\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back and append");
    let upgraded = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("libhdf5 kept the message");
    assert_eq!(
        upgraded.version, 1,
        "libhdf5 no longer upgrades a mapped file-space info message"
    );
    let _ = std::fs::remove_file(&path);
}

/// The same account on a file libhdf5 wrote with paged aggregation.
///
/// A paged file's free space is sorted into per-page managers by
/// `H5MF__alloc_to_fs_type` (H5MF.c:265) rather than the dichotomy's two, and
/// its allocations are page-shaped: everything below a page is packed into a
/// page of its own kind and everything at or above one is page-aligned with
/// the misaligned tail returned to the large manager
/// (`H5MF__alloc_pagefs`, H5MF.c:858). An append that got any of that wrong
/// leaves either a section libhdf5 will not take back or bytes nothing
/// records, and `h5stat -S` names both.
#[test]
fn a_crate_append_rewrites_a_paged_files_managers() {
    let Some(py) = python() else { return };
    let path = tmp("paged_append");
    write_persisting_file_with(py, &path, "page");

    let before = h5stat_space(py, &path);
    assert!(before.tracked > 0, "{before:?} has nothing to reuse");
    assert_eq!(before.unaccounted, 0, "libhdf5 left the file accounted for");
    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the paged file declares its strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::Page);

    crate_appends(&path, "added");
    let after = h5stat_space(py, &path);
    assert_eq!(
        after.unaccounted, 0,
        "the append left space no manager records: {after:?}"
    );
    assert!(
        after.tracked > 0,
        "the append left the file with no tracked free space: {after:?}"
    );
    assert_eq!(
        after.total % info.page_size,
        0,
        "the append left the file off its page grid: {after:?}"
    );

    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, ["added", "by_libhdf5", "grp/inner", "keep"]);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        (0..16).collect::<Vec<i32>>()
    );
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..8).collect::<Vec<i32>>()
    );
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// The page size the builder names is the one libhdf5 reads back off the file:
/// a non-default size lays the file out and comes back through the fcpl,
/// exactly as `H5Pget_file_space_page_size` reports what
/// `H5Pset_file_space_page_size` set.
#[test]
fn a_crate_created_paged_file_keeps_a_non_default_page_size() {
    let Some(py) = python() else { return };
    const PAGE: u64 = 8192;
    let path = tmp("paged_page_size");
    {
        let file = H5File::options()
            .file_space(FileSpaceStrategy::Page, true, 1)
            .file_space_page_size(PAGE)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([16usize])
            .create("keep")
            .unwrap()
            .write_raw(&(0..16i32).collect::<Vec<_>>())
            .unwrap();
        file.create_group("grp").unwrap();
        file.new_dataset::<f64>()
            .shape([8usize])
            .create("grp/inner")
            .unwrap()
            .write_raw(&(0..8).map(f64::from).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }

    let created = h5stat_space(py, &path);
    assert_eq!(
        created.unaccounted, 0,
        "the created file leaks space: {created:?}"
    );
    assert_eq!(
        created.total % PAGE,
        0,
        "the file ends on one of its {PAGE}-byte pages: {created:?}"
    );
    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the created file declares its strategy");
    assert_eq!(info.page_size, PAGE);
    h5clear_accepts(py, &path);

    let script = format!(
        "import h5py\n\
         f = h5py.File(r'{}', 'r')\n\
         size = f.id.get_create_plist().get_file_space_page_size()\n\
         assert size == {PAGE}, size\n\
         assert list(f['keep'][:]) == list(range(16)), list(f['keep'][:])\n\
         assert list(f['grp/inner'][:]) == list(range(8)), list(f['grp/inner'][:])\n\
         f.close()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py page-size read-back");

    h5py_reads_and_appends(py, &path, "by_libhdf5");
    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, ["by_libhdf5", "grp/inner", "keep"]);
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// `H5Pset_file_space_page_size` refuses a size below 512 outright
/// (H5Pfcpl.c:1389); this crate's builder cannot fail on the call itself, so
/// the same bound is enforced where the file is made.
#[test]
fn a_page_size_below_the_library_minimum_is_refused() {
    let path = tmp("paged_page_size_small");
    let Err(err) = H5File::options()
        .file_space(FileSpaceStrategy::Page, true, 1)
        .file_space_page_size(256)
        .create(&path)
    else {
        panic!("a 256-byte file-space page was accepted");
    };
    assert!(
        format!("{err}").contains("between 512 bytes and 1073741824"),
        "{err}"
    );
    assert!(
        !path.exists() || std::fs::metadata(&path).unwrap().len() == 0,
        "a refused create left a file behind"
    );
    let _ = std::fs::remove_file(&path);
}

/// A paged file this crate creates is one libhdf5 opens, appends to and finds
/// fully accounted for.
#[test]
fn a_crate_created_paged_file_is_one_libhdf5_accepts() {
    let Some(py) = python() else { return };
    let path = tmp("paged_created");
    {
        let file = H5File::options()
            .file_space(FileSpaceStrategy::Page, true, 1)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([16usize])
            .create("keep")
            .unwrap()
            .write_raw(&(0..16i32).collect::<Vec<_>>())
            .unwrap();
        file.create_group("grp").unwrap();
        file.new_dataset::<f64>()
            .shape([8usize])
            .create("grp/inner")
            .unwrap()
            .write_raw(&(0..8).map(f64::from).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }

    let created = h5stat_space(py, &path);
    assert_eq!(
        created.unaccounted, 0,
        "the created paged file leaks space: {created:?}"
    );
    let info = H5File::open(&path)
        .unwrap()
        .superblock_extension()
        .file_space_info
        .expect("the created file declares its strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert!(info.persist);
    assert_eq!(
        created.total % info.page_size,
        0,
        "a paged file ends on a page boundary: {created:?}"
    );
    // As in `a_crate_created_persisting_file_is_one_libhdf5_accepts`, libhdf5's
    // own close drops the managers of a file it did not modify, so the append
    // is measured against what `h5clear -s` left rather than against zero.
    h5clear_accepts(py, &path);
    let cleared = h5stat_space(py, &path);

    crate_appends(&path, "added");
    let appended = h5stat_space(py, &path);
    assert_eq!(
        appended.unaccounted, cleared.unaccounted,
        "the append leaked space on top of the {cleared:?} libhdf5 left"
    );
    h5clear_accepts(py, &path);
    h5py_reads_and_appends(py, &path, "by_libhdf5");

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, ["added", "by_libhdf5", "grp/inner", "keep"]);
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// `eoa_pre_fsm_fsalloc` is the end of the finished file, not the end it had
/// before the managers took their own two blocks.
///
/// The name says otherwise, and it is the older meaning: 1.10 kept both EOAs
/// and put the pre-allocation one in this field. 1.14 keeps one value, taken
/// after the allocation loop (H5MF.c:3234-3240), and calls it "the final eoa"
/// where it reads it back (H5Fsuper.c:826). So the check is not against a
/// number this crate chose — it is the identity libhdf5's own file satisfies,
/// asserted first on the twin h5py writes and then on the file this crate
/// writes for the same properties.
#[test]
fn the_recorded_eoa_is_the_end_of_the_finished_file() {
    let Some(py) = python() else { return };
    for strategy in ["fsm", "page"] {
        let reference = tmp(&format!("eoa_ref_{strategy}"));
        write_persisting_file_with(py, &reference, strategy);
        let twin = H5File::open(&reference)
            .unwrap()
            .superblock_extension()
            .file_space_info
            .expect("the h5py twin declares its strategy");
        // Non-vacuous only if the settle had something to allocate: a file
        // with no manager on disk never moves the EOA during the settle, so
        // both readings of the field would agree on it.
        assert_ne!(
            twin.fs_addr[0],
            u64::MAX,
            "the {strategy} twin persisted no manager: {twin:?}"
        );
        assert_eq!(
            twin.eoa_pre_fsm_fsalloc,
            std::fs::metadata(&reference).unwrap().len(),
            "libhdf5 did not record the end of the {strategy} file it wrote"
        );
        let _ = std::fs::remove_file(&reference);

        let path = tmp(&format!("eoa_{strategy}"));
        let file = H5File::options()
            .file_space(
                match strategy {
                    "fsm" => FileSpaceStrategy::FsmAggr,
                    _ => FileSpaceStrategy::Page,
                },
                true,
                1,
            )
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([16usize])
            .create("keep")
            .unwrap()
            .write_raw(&(0..16i32).collect::<Vec<_>>())
            .unwrap();
        file.create_group("grp").unwrap();
        file.new_dataset::<f64>()
            .shape([8usize])
            .create("grp/inner")
            .unwrap()
            .write_raw(&(0..8).map(f64::from).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();

        // Once for the file this crate creates, once for the one it reopens:
        // the second close settles managers it read off disk, and records the
        // end of a file whose blocks moved.
        for round in ["created", "appended"] {
            let info = H5File::open(&path)
                .unwrap()
                .superblock_extension()
                .file_space_info
                .expect("the created file declares its strategy");
            assert_ne!(
                info.fs_addr[0],
                u64::MAX,
                "the {round} {strategy} file persisted no manager: {info:?}"
            );
            assert_eq!(
                info.eoa_pre_fsm_fsalloc,
                std::fs::metadata(&path).unwrap().len(),
                "the {round} {strategy} file recorded an EOA from before its \
                 managers were placed"
            );
            let space = h5stat_space(py, &path);
            assert_eq!(
                space.unaccounted, 0,
                "the {round} {strategy} file leaks space: {space:?}"
            );
            if round == "created" {
                crate_appends(&path, "added");
            }
        }
        h5clear_accepts(py, &path);
        let _ = std::fs::remove_file(&path);
    }
}
