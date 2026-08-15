//! Shared object header messages (SOHM), append side.
//!
//! Creating a file with shared messages works (`tests/sohm_write.rs`) and so
//! does reading one (`tests/sohm.rs`). Appending is the third case: the table
//! and its indexes are laid out *whole* from the whole message set, never
//! grown insert by insert, so a reopen replaces the table rather than adding
//! to it. Every heap ID in the file is therefore reassigned, which is sound
//! only while no header that keeps its bytes still points into the old heap —
//! the reopen forces every modelled object into the rewrite and refuses the
//! file when a preserved one shares.
//!
//! What has to hold afterwards: libhdf5 reads the result, every record's
//! reference count is the number of pointers that actually name it, and a
//! second reopen works on the file the first one wrote.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::btree_v1::BTreeV1Config;
use rust_hdf5::format::chunk_index::btree_v2::{collect_btree_v2_records, Bt2Header};
use rust_hdf5::format::messages::shared::MSG_FLAG_SHARED;
use rust_hdf5::format::messages::{MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE};
use rust_hdf5::format::object_header::{ObjectHeader, ObjectHeaderMessage};
use rust_hdf5::format::sohm::{
    record_size, type_flag, SharedLocation, SharedMessagePointer, SohmIndexHeader, SohmMasterTable,
    SOHM_HEAP_ID_LEN, SOHM_INDEX_BTREE, SOHM_INDEX_LIST, SOHM_IN_HEAP,
};
use rust_hdf5::format::superblock::SuperblockV2V3;
use rust_hdf5::format::{BlockReader, FormatContext, FormatResult, LibverBound};
use rust_hdf5::H5File;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Per-test unique temp path; cargo runs tests in parallel.
fn unique_tmp(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust_hdf5_sohm_append_{}_{}_{}",
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

/// Copy a fixture into a temp path of its own.
fn copy_fixture(name: &str, label: &str) -> PathBuf {
    let path = unique_tmp(label);
    std::fs::write(&path, std::fs::read(fixture(name)).unwrap()).unwrap();
    path
}

/// Reopen `path` and add one contiguous dataset called `name`.
fn append_dataset(path: &PathBuf, name: &str, base: i32) {
    let file = H5File::open_rw(path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create(name)
        .unwrap()
        .write_raw(&(0..8i32).map(|j| base + j).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// Read `path` through this crate's own reader, which has to resolve every
/// shared pointer to answer any of it, and assert the `gen_sohm.c` content is
/// all still there beside `extra`.
fn check_fixture_contents(path: &PathBuf, extra: &[(&str, i32)]) {
    let file = H5File::open(path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    let mut expected: Vec<String> = ["shared0", "shared1", "shared2", "shared3", "uses_named"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    expected.extend(extra.iter().map(|(n, _)| n.to_string()));
    expected.sort();
    assert_eq!(names, expected);

    for i in 0..4i32 {
        let ds = file.dataset(&format!("shared{i}")).unwrap();
        assert_eq!(ds.shape(), vec![8]);
        let data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(data, (0..8i32).map(|j| i * 10 + j).collect::<Vec<_>>());
        assert_eq!(ds.attr_names().unwrap(), vec!["cal"]);
        let cal: Vec<f64> = ds.attr("cal").unwrap().read_numeric_as().unwrap();
        assert_eq!(cal, vec![0.5, 1.5, 2.5]);
    }
    let used: Vec<i32> = file.dataset("uses_named").unwrap().read_raw().unwrap();
    assert_eq!(used, (100..108).collect::<Vec<i32>>());
    assert_eq!(file.named_datatype_names(), vec!["named_i32"]);
    for (name, base) in extra {
        let data: Vec<i32> = file.dataset(name).unwrap().read_raw().unwrap();
        assert_eq!(data, (0..8i32).map(|j| base + j).collect::<Vec<_>>());
    }
}

// ---------------------------------------------------------------- the census

/// A `BlockReader` over a whole file already in memory, so the B-tree walker
/// can read index nodes out of the same bytes the headers come from.
struct Bytes<'a>(&'a [u8]);

impl BlockReader for Bytes<'_> {
    fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>> {
        let at = offset as usize;
        let end = (at + len).min(self.0.len());
        Ok(self.0[at..end].to_vec())
    }
}

/// The shared-message table the file's superblock extension names.
fn master_table(path: &PathBuf) -> SohmMasterTable {
    let smt = {
        let file = H5File::open(path).unwrap();
        file.superblock_extension()
            .shared_message_table
            .expect("a file with shared messages names its table in the extension")
    };
    let bytes = std::fs::read(path).unwrap();
    let ctx = FormatContext::default_v3();
    let at = smt.table_address as usize;
    let size = SohmMasterTable::encoded_size(&ctx, smt.nindexes);
    SohmMasterTable::decode(&bytes[at..at + size], &ctx, smt.nindexes).unwrap()
}

/// Every record of one index, whichever form it is in, as `(heap id, ref
/// count)`.
fn index_records(
    bytes: &[u8],
    ctx: &FormatContext,
    header: &SohmIndexHeader,
) -> Vec<([u8; SOHM_HEAP_ID_LEN], u32)> {
    let size = record_size(ctx);
    let at = header.index_addr as usize;
    let raw = if header.index_type == SOHM_INDEX_LIST {
        assert_eq!(&bytes[at..at + 4], b"SMLI");
        bytes[at + 4..at + 4 + size * header.num_messages as usize].to_vec()
    } else {
        let bt2 = Bt2Header::decode(&bytes[at..], ctx).unwrap();
        collect_btree_v2_records(&bt2, ctx, &mut Bytes(bytes)).unwrap()
    };
    raw.chunks_exact(size)
        .map(|r| {
            assert_eq!(r[0], SOHM_IN_HEAP, "a record this crate wrote is in a heap");
            (
                r[9..9 + SOHM_HEAP_ID_LEN].try_into().unwrap(),
                u32::from_le_bytes(r[5..9].try_into().unwrap()),
            )
        })
        .collect()
}

/// Every message of one version-2 object header, continuation blocks
/// included. A census that stopped at chunk 0 would miss exactly the links
/// and attributes a rewrite pushed out of it, and so undercount references.
fn all_messages(bytes: &[u8], at: usize) -> Vec<ObjectHeaderMessage> {
    use rust_hdf5::format::checksum::checksum_metadata;
    use rust_hdf5::format::messages::MSG_OBJ_HEADER_CONTINUATION;

    let (header, _) = ObjectHeader::decode(&bytes[at..]).unwrap();
    let hdr_size = if header.has_creation_order() { 6 } else { 4 };
    let mut out = header.messages.clone();
    let mut pending: Vec<_> = out
        .iter()
        .filter(|m| m.msg_type == MSG_OBJ_HEADER_CONTINUATION)
        .map(|m| {
            (
                u64::from_le_bytes(m.data[0..8].try_into().unwrap()) as usize,
                u64::from_le_bytes(m.data[8..16].try_into().unwrap()) as usize,
            )
        })
        .collect();
    while let Some((addr, len)) = pending.pop() {
        let block = &bytes[addr..addr + len];
        assert_eq!(&block[..4], b"OCHK");
        let end = len - 4;
        assert_eq!(
            u32::from_le_bytes(block[end..].try_into().unwrap()),
            checksum_metadata(&block[..end])
        );
        let mut pos = 4;
        while pos + hdr_size <= end {
            let msg_type = block[pos];
            let size = u16::from_le_bytes([block[pos + 1], block[pos + 2]]) as usize;
            let flags = block[pos + 3];
            pos += hdr_size;
            if pos + size > end {
                break;
            }
            if msg_type == MSG_OBJ_HEADER_CONTINUATION {
                pending.push((
                    u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap()) as usize,
                    u64::from_le_bytes(block[pos + 8..pos + 16].try_into().unwrap()) as usize,
                ));
            }
            if msg_type != 0 {
                out.push(ObjectHeaderMessage {
                    msg_type,
                    flags,
                    creation_index: 0,
                    data: block[pos..pos + size].to_vec(),
                });
            }
            pos += size;
        }
    }
    out
}

/// The object header addresses one Symbol Table message names, by walking its
/// version-1 B-tree down to the symbol table nodes.
///
/// A file with shared messages has a version-2 superblock over symbol-table
/// groups, so a census that followed only Link messages would stop at the root
/// and find no pointers at all.
fn symbol_table_targets(bytes: &[u8], ctx: &FormatContext, body: &[u8]) -> Vec<u64> {
    use rust_hdf5::format::btree_v1::{BTreeV1Config, BTreeV1Node};
    use rust_hdf5::format::symbol_table::SymbolTableNode;
    use rust_hdf5::format::UNDEF_ADDR;

    let sa = ctx.sizeof_addr as usize;
    let ss = ctx.sizeof_size as usize;
    let btree_addr = u64::from_le_bytes(body[..8].try_into().unwrap());
    // A version-2 superblock keeps its "K" ranks in a B-tree-K message, which
    // none of these fixtures carries: every node in them is the library-default
    // width. (`tests/superblock_extension.rs` covers a file that does carry
    // one.)
    let cfg = BTreeV1Config::default();

    let mut snods = Vec::new();
    let mut queue = vec![btree_addr];
    while let Some(at) = queue.pop() {
        if at == UNDEF_ADDR {
            continue;
        }
        let node =
            BTreeV1Node::decode(&bytes[at as usize..], sa, ss, cfg.snode_max_entries()).unwrap();
        if node.level == 0 {
            snods.extend(node.children);
        } else {
            queue.extend(node.children);
        }
    }

    let mut out = Vec::new();
    for at in snods {
        let snod =
            SymbolTableNode::decode(&bytes[at as usize..], sa, ss, cfg.sym_leaf_max_entries())
                .unwrap();
        out.extend(
            snod.entries
                .iter()
                .map(|e| e.obj_header_addr)
                .filter(|&a| a != UNDEF_ADDR && a != 0),
        );
    }
    out
}

/// Every object header reachable from the root of a version-2 file, by path.
fn every_header(bytes: &[u8], ctx: &FormatContext) -> Vec<(String, Vec<ObjectHeaderMessage>)> {
    use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
    use rust_hdf5::format::messages::{MSG_LINK, MSG_SYMBOL_TABLE};

    let sb = SuperblockV2V3::decode(bytes).unwrap();
    let mut out = Vec::new();
    let mut queue = vec![(String::from("/"), sb.root_group_object_header_address)];
    let mut seen = std::collections::HashSet::new();
    while let Some((path, addr)) = queue.pop() {
        if !seen.insert(addr) {
            continue;
        }
        let messages = all_messages(bytes, (sb.base_address + addr) as usize);
        for m in &messages {
            match m.msg_type {
                MSG_LINK => {
                    if let Ok((link, _)) = LinkMessage::decode(&m.data, ctx) {
                        if let LinkTarget::Hard { address } = link.target {
                            queue.push((format!("{}{}/", path, link.name), address));
                        }
                    }
                }
                MSG_SYMBOL_TABLE => {
                    for (i, target) in symbol_table_targets(bytes, ctx, &m.data)
                        .into_iter()
                        .enumerate()
                    {
                        queue.push((format!("{path}stab{i}/"), target));
                    }
                }
                _ => {}
            }
        }
        out.push((path, messages));
    }
    out
}

/// The heap-ID pointer of a shared datatype or dataspace stored *inside* an
/// attribute body, where the attribute's own flags byte says which is shared
/// (`H5O_ATTR_FLAG_TYPE_SHARED` / `H5O_ATTR_FLAG_SPACE_SHARED`).
fn attribute_shared_fields(body: &[u8]) -> Vec<&[u8]> {
    if body.len() < 8 || body[0] < 2 || body[1] & 0x03 == 0 {
        return Vec::new();
    }
    let flags = body[1];
    let name_size = u16::from_le_bytes([body[2], body[3]]) as usize;
    let dt_size = u16::from_le_bytes([body[4], body[5]]) as usize;
    let ds_size = u16::from_le_bytes([body[6], body[7]]) as usize;
    let name_end = if body[0] >= 3 { 9 } else { 8 } + name_size;
    let dt_end = name_end + dt_size;
    let ds_end = dt_end + ds_size;
    let mut out = Vec::new();
    if flags & 0x01 != 0 {
        out.push(&body[name_end..dt_end]);
    }
    if flags & 0x02 != 0 {
        out.push(&body[dt_end..ds_end]);
    }
    out
}

/// The invariant on `SohmState`, checked against the file the append wrote:
/// a record's reference count is the number of object header pointers that
/// name its heap object, counting both the whole-message form and the
/// datatype/dataspace an attribute body carries.
fn reference_counts_match_the_pointers(path: &PathBuf) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap();
    let ctx = FormatContext::default_v3();

    let mut pointers: std::collections::HashMap<[u8; SOHM_HEAP_ID_LEN], u32> = Default::default();
    let mut count = |raw: &[u8]| {
        if let Ok(p) = SharedMessagePointer::decode(raw, &ctx) {
            if p.location == SharedLocation::Sohm {
                *pointers.entry(p.heap_id).or_default() += 1;
            }
        }
    };
    for (_, messages) in every_header(&bytes, &ctx) {
        for msg in &messages {
            if msg.flags & MSG_FLAG_SHARED != 0 {
                count(&msg.data);
            } else if msg.msg_type == MSG_ATTRIBUTE {
                for raw in attribute_shared_fields(&msg.data) {
                    count(raw);
                }
            }
        }
    }

    let table = master_table(path);
    let mut counts = Vec::new();
    for header in &table.indexes {
        for (heap_id, ref_count) in index_records(&bytes, &ctx, header) {
            counts.push(ref_count);
            assert_eq!(
                pointers.remove(&heap_id),
                Some(ref_count),
                "record {heap_id:02x?} claims {ref_count} references"
            );
        }
    }
    assert_eq!(
        counts.len(),
        table.indexes.iter().map(|h| h.num_messages as usize).sum(),
        "the master table's message counts do not match the records"
    );
    assert!(
        pointers.is_empty(),
        "headers point at heap objects no record names: {pointers:02x?}"
    );
    counts
}

// ------------------------------------------------------------ libhdf5 tools

/// The same Python-install probe `tests/libver_earliest.rs` uses: the tools
/// beside it are the ones of the libhdf5 h5py is linked against.
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
            eprintln!("skipping SOHM append cross-check: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
}

fn run(program: impl AsRef<std::ffi::OsStr>, args: &[&str], what: &str) {
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
}

/// Hand the file to h5dump, h5clear and h5py: the first walks every object
/// header, the second reads the superblock's consistency flags, the third
/// answers what the objects are.
fn libhdf5_reads_back(path: &Path, body: &str) {
    let Some(py) = python() else { return };
    let dir = std::path::Path::new(py).parent().unwrap();
    let text = path.to_str().unwrap();
    for tool in ["h5dump", "h5clear"] {
        let exe = dir.join(tool);
        if !exe.exists() {
            continue;
        }
        let args = if tool == "h5dump" {
            ["-pBH", text]
        } else {
            ["-s", text]
        };
        run(&exe, &args, tool);
    }
    let script = format!(
        "import h5py, numpy as np\nf = h5py.File(r'{}', 'r')\n{}\n",
        path.display(),
        body
    );
    run(py, &["-c", &script], "h5py read-back");
}

/// What h5py must be able to say about an appended `gen_sohm.c` file. The
/// committed-datatype assertion is the one that fails when a rewrite inlines
/// the named type instead of re-emitting the pointer to its object header.
const H5PY_FIXTURE_CHECKS: &str = "\
assert sorted(f) == sorted(EXPECTED), sorted(f)
for i in range(4):
    d = f['shared%d' % i]
    assert list(d[()]) == [i * 10 + j for j in range(8)], list(d[()])
    assert list(d.attrs['cal']) == [0.5, 1.5, 2.5], list(d.attrs['cal'])
assert list(f['uses_named'][()]) == list(range(100, 108))
assert f['uses_named'].id.get_type().committed(), 'uses_named lost its named datatype'
assert isinstance(f['named_i32'], h5py.Datatype)
";

fn h5py_fixture_body(extra: &[(&str, i32)]) -> String {
    let mut names: Vec<String> = ["shared0", "shared1", "shared2", "shared3", "uses_named"]
        .iter()
        .map(|s| format!("'{s}'"))
        .collect();
    names.push("'named_i32'".into());
    names.extend(extra.iter().map(|(n, _)| format!("'{n}'")));
    let mut body = format!("EXPECTED = [{}]\n{}", names.join(", "), H5PY_FIXTURE_CHECKS);
    for (name, base) in extra {
        body.push_str(&format!(
            "assert list(f['{name}'][()]) == {:?}, list(f['{name}'][()])\n",
            (0..8i32).map(|j| base + j).collect::<Vec<_>>()
        ));
    }
    body
}

// ------------------------------------------------------------------- tests

/// A list index: the append replaces the table, and everything the file held
/// — including the dataset that reaches its type through `named_i32` — comes
/// back with the reference counts the new records claim.
#[test]
fn a_list_index_file_takes_a_new_dataset() {
    let path = copy_fixture("sohm_list.h5", "list");
    let before = master_table(&path);
    assert_eq!(before.indexes[0].num_messages, 4);
    append_dataset(&path, "appended", 200);

    let extra = [("appended", 200)];
    check_fixture_contents(&path, &extra);
    let counts = reference_counts_match_the_pointers(&path);
    libhdf5_reads_back(&path, &h5py_fixture_body(&extra));

    // Consistent counts alone would also describe a file that shares nothing:
    // the four identical datasets must still reach one body between them.
    assert!(
        counts.iter().any(|&c| c >= 4),
        "nothing in the rebuilt index is shared four ways: {counts:?}"
    );

    // The index specification is a file creation property: `H5SM_init` fixes
    // the mask, the minimum size and the phase-change pair when the file is
    // made, so a reopen must carry all of them forward unchanged.
    let after = master_table(&path);
    assert_eq!(after.indexes.len(), before.indexes.len());
    let (a, b) = (&after.indexes[0], &before.indexes[0]);
    assert_eq!(a.index_type, SOHM_INDEX_LIST);
    assert_eq!(
        (a.mesg_types, a.min_mesg_size, a.list_max, a.btree_min),
        (b.mesg_types, b.min_mesg_size, b.list_max, b.btree_min)
    );
    cleanup(&path);
}

/// A file with shared messages is a version-2 superblock over symbol-table
/// groups, and `H5F__super_read` reads that superblock as `H5F_LIBVER_V18`. The
/// bound decides what a *new* group is made as; it does not move a group that
/// already exists, so the root here stays a symbol table and gains its link as
/// an entry — the same file shape libhdf5's own reopen leaves behind.
#[test]
fn a_reopened_symbol_table_group_keeps_its_symbol_table() {
    use rust_hdf5::format::messages::{MSG_LINK, MSG_LINK_INFO, MSG_SYMBOL_TABLE};

    let path = copy_fixture("sohm_list.h5", "stabroot");
    append_dataset(&path, "appended", 200);

    let bytes = std::fs::read(&path).unwrap();
    let ctx = FormatContext::default_v3();
    let (_, root) = every_header(&bytes, &ctx)
        .into_iter()
        .find(|(p, _)| p == "/")
        .expect("the root group has a header");
    let kinds: Vec<u8> = root.iter().map(|m| m.msg_type).collect();
    assert!(
        kinds.contains(&MSG_SYMBOL_TABLE),
        "the root lost its symbol table: {kinds:?}"
    );
    assert!(
        !kinds.contains(&MSG_LINK) && !kinds.contains(&MSG_LINK_INFO),
        "the root was converted to link messages: {kinds:?}"
    );

    // The new link is in that symbol table, not somewhere beside it.
    let stab = root
        .iter()
        .find(|m| m.msg_type == MSG_SYMBOL_TABLE)
        .expect("a Symbol Table message");
    assert_eq!(symbol_table_targets(&bytes, &ctx, &stab.data).len(), 7);

    check_fixture_contents(&path, &[("appended", 200)]);
    libhdf5_reads_back(&path, &h5py_fixture_body(&[("appended", 200)]));
    cleanup(&path);
}

/// The same for the B-tree form, which the append must keep: `list_max` is 0
/// in that fixture, so a rebuilt index that fell back to a list would be a
/// file libhdf5 never writes.
#[test]
fn a_btree_index_file_takes_a_new_dataset() {
    let path = copy_fixture("sohm_btree.h5", "btree");
    append_dataset(&path, "appended", 200);

    let extra = [("appended", 200)];
    check_fixture_contents(&path, &extra);
    let counts = reference_counts_match_the_pointers(&path);
    libhdf5_reads_back(&path, &h5py_fixture_body(&extra));
    assert!(
        counts.iter().any(|&c| c >= 4),
        "nothing in the rebuilt index is shared four ways: {counts:?}"
    );

    let index = &master_table(&path).indexes[0];
    assert_eq!(index.index_type, SOHM_INDEX_BTREE);
    assert_eq!((index.list_max, index.btree_min), (0, 0));
    cleanup(&path);
}

/// The blocks of the table a reopen replaces go back to the allocator before
/// the replacement is laid out, so repeated open/close cycles reuse them
/// instead of growing the file without bound — and the file each round writes
/// is the input the next round reopens.
#[test]
fn a_third_open_still_appends() {
    let path = copy_fixture("sohm_list.h5", "rounds");
    let mut extra = Vec::new();
    let mut sizes = Vec::new();
    for round in 0..3i32 {
        let name = ["added0", "added1", "added2"][round as usize];
        append_dataset(&path, name, 300 + round * 10);
        extra.push((name, 300 + round * 10));
        check_fixture_contents(&path, &extra);
        reference_counts_match_the_pointers(&path);
        sizes.push(std::fs::metadata(&path).unwrap().len());
    }
    libhdf5_reads_back(&path, &h5py_fixture_body(&extra));
    // Each round adds one dataset, so the file grows; what it must not do is
    // strand a whole structure per round. Two are replaced wholesale — the
    // shared-message table with its fractal heap, and the root group's symbol
    // table — and the budget is sized so that leaking either one overruns it.
    let table = master_table(&path);
    let index = &table.indexes[0];
    let table_span = index.num_messages as u64 * record_size(&FormatContext::default_v3()) as u64;
    let cfg = BTreeV1Config::default();
    let stab_span = (cfg.snode_btree_node_size(8, 8) + cfg.symbol_table_node_size(8, 8)) as u64;
    let round_growth = sizes[2] - sizes[1];
    assert!(
        round_growth < table_span * 8 + stab_span,
        "a round grew the file by {round_growth} bytes, past the {} it may take: \
         a superseded structure is not being freed",
        table_span * 8 + stab_span
    );
    cleanup(&path);
}

/// The case the append path was written for: a file this crate created with
/// `shared_messages` at the earliest bound — a version-2 superblock over
/// version-1 headers and symbol-table groups — reopens and takes new objects.
#[test]
fn a_crate_written_earliest_file_with_shared_messages_reopens() {
    let path = unique_tmp("earliest");
    let types = type_flag(MSG_DATATYPE).unwrap()
        | type_flag(MSG_DATASPACE).unwrap()
        | type_flag(MSG_ATTRIBUTE).unwrap();
    {
        let file = H5File::options()
            .libver(LibverBound::Earliest)
            .shared_messages(&[(types, 0)], 50, 40)
            .create(&path)
            .unwrap();
        let group = file.root_group().create_group("g").unwrap();
        for (parent, name) in [(&file.root_group(), "outer"), (&group, "inner")] {
            let ds = parent
                .new_dataset::<i32>()
                .shape([4usize])
                .create(name)
                .unwrap();
            ds.write_raw(&[7i32; 4]).unwrap();
            ds.new_attr::<f64>()
                .shape([2usize])
                .create("cal")
                .unwrap()
                .write_array(&[1.5f64, 2.5])
                .unwrap();
        }
        file.close().unwrap();
    }
    // A symbol-table group under a version-2 superblock: the shared-message
    // table is what forces the superblock up, and nothing forces the groups.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes[8], 2, "shared messages need a version-2 superblock");
    assert!(
        bytes.windows(4).any(|w| w == b"SNOD"),
        "the earliest bound keeps symbol-table groups"
    );

    for round in 0..2i32 {
        let file = H5File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([4usize])
            .create(&format!("added{round}"))
            .unwrap()
            .write_raw(&[9i32; 4])
            .unwrap();
        file.close().unwrap();
    }

    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, vec!["added0", "added1", "g/inner", "outer"]);
    for name in ["outer", "g/inner"] {
        let ds = file.dataset(name).unwrap();
        assert_eq!(ds.read_raw::<i32>().unwrap(), vec![7i32; 4]);
        let cal: Vec<f64> = ds.attr("cal").unwrap().read_numeric_as().unwrap();
        assert_eq!(cal, vec![1.5, 2.5]);
    }
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap()[8], 2);
    libhdf5_reads_back(
        &path,
        "assert sorted(f) == ['added0', 'added1', 'g', 'outer'], sorted(f)\n\
         assert list(f['outer'][()]) == [7, 7, 7, 7]\n\
         assert list(f['g/inner'].attrs['cal']) == [1.5, 2.5]\n\
         assert list(f['added1'][()]) == [9, 9, 9, 9]\n",
    );
    cleanup(&path);
}

/// The shared-message table is the one extension message this path owns — its
/// storage is laid out afresh, so its address is recomputed. Everything else
/// the extension held is carried across unread, and a file that has both proves
/// the two do not displace each other.
#[test]
fn an_extension_message_beside_the_table_survives_the_append() {
    let path = copy_fixture("sohm_paged.h5", "paged");

    // The fixture really does have both: a shared-message table to relocate,
    // and a file space strategy that would go missing with it.
    let before = {
        let file = H5File::open(&path).unwrap();
        let ext = file.superblock_extension();
        assert!(ext.shared_message_table.is_some());
        assert!(ext.file_space_info.is_some());
        ext
    };

    append_dataset(&path, "appended", 200);

    let after = H5File::open(&path).unwrap().superblock_extension();
    assert_eq!(after.file_space_info, before.file_space_info);
    assert_eq!(after.driver_info, before.driver_info);
    assert_eq!(after.btree_k, before.btree_k);
    // The table moved with its heap, and is still named.
    let table = after
        .shared_message_table
        .expect("the appended file still declares its shared messages");
    assert_eq!(
        table.nindexes,
        before.shared_message_table.unwrap().nindexes
    );

    let extra = [("appended", 200)];
    check_fixture_contents(&path, &extra);
    reference_counts_match_the_pointers(&path);
    libhdf5_reads_back(&path, &h5py_fixture_body(&extra));
    cleanup(&path);
}

/// A committed datatype is carried by its bytes, so a shared pointer inside
/// one would still name the old heap after the table is laid out afresh. The
/// file is refused instead — without the gate the append writes a header whose
/// attribute points at a heap object that no longer exists.
#[test]
fn a_preserved_object_holding_a_heap_pointer_refuses_the_file() {
    let bytes = std::fs::read(fixture("sohm_named_attr.h5")).unwrap();
    let path = copy_fixture("sohm_named_attr.h5", "namedattr");

    // The fixture really is the awkward shape: `named_i32` is a committed
    // datatype (nothing re-encodes it) and it carries the same `cal` attribute
    // the four datasets share, so its header holds a heap pointer.
    {
        let file = H5File::open(&path).unwrap();
        assert!(file.superblock_extension().shared_message_table.is_some());
        let named = file.named_datatype("named_i32").unwrap();
        assert_eq!(named.attr_names().unwrap(), vec!["cal"]);
        let values: Vec<f64> = named.attr("cal").unwrap().read_numeric_as().unwrap();
        assert_eq!(values, vec![0.5, 1.5, 2.5]);
    }

    let text = match H5File::open_rw(&path) {
        Ok(_) => panic!("a file whose committed datatype shares must not open for writing"),
        Err(e) => e.to_string(),
    };
    assert!(text.contains("named_i32"), "{text}");
    assert!(
        text.contains("holds a shared object header message"),
        "{text}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes,
        "a refusal wrote bytes"
    );
    cleanup(&path);
}

/// The append path reads the superblock extension, which most files do not
/// have: they keep working.
#[test]
fn a_file_without_shared_messages_still_opens_for_appending() {
    let path = unique_tmp("plain");
    {
        let file = H5File::create(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("a")
            .unwrap()
            .write_raw(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    }
    {
        let file = H5File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("b")
            .unwrap()
            .write_raw(&[4i32, 5, 6])
            .unwrap();
        file.close().unwrap();
    }
    let file = H5File::open(&path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(names, vec!["a", "b"]);
    drop(file);
    cleanup(&path);
}
