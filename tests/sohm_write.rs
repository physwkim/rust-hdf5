//! Shared object header messages (SOHM), write side.
//!
//! A file created with `H5FileOptions::shared_messages` writes each covered
//! message body once into a shared-message fractal heap and stores a pointer
//! to it in every object header that would have held the body. What has to
//! hold: the file still reads back as the same objects, the master table and
//! the index describe what was actually written, and a file created without
//! the option is unchanged.
//!
//! `tests/sohm.rs` is the read side, against libhdf5-written fixtures;
//! `tests/sohm_append.rs` covers reopening a file that already has a
//! shared-message table.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rust_hdf5::format::chunk_index::btree_v2::{collect_btree_v2_records, Bt2Header};
use rust_hdf5::format::creation_order::CreationOrder;
use rust_hdf5::format::fractal_heap::{
    collect_managed_blocks, read_heap_object, FractalHeapHeader, HeapId,
};
use rust_hdf5::format::messages::attribute::{ATTR_FLAG_SPACE_SHARED, ATTR_FLAG_TYPE_SHARED};
use rust_hdf5::format::messages::dataspace::DataspaceMessage;
use rust_hdf5::format::messages::{
    MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE, MSG_FLAG_SHAREABLE, MSG_FLAG_SHARED,
};
use rust_hdf5::format::object_header::ObjectHeader;
use rust_hdf5::format::sohm::{
    record_size, type_flag, SharedLocation, SharedMessagePointer, SohmMasterTable,
    BT2_TYPE_SOHM_INDEX, SMLI_SIGNATURE, SOHM_B2_NODE_SIZE, SOHM_HEAP_ID_LEN, SOHM_INDEX_BTREE,
    SOHM_INDEX_LIST, SOHM_IN_HEAP, SOHM_IN_OH,
};
use rust_hdf5::format::superblock::{SuperblockV0V1, SuperblockV2V3};
use rust_hdf5::format::{BlockReader, FormatContext, FormatResult, UNDEF_ADDR};
use rust_hdf5::{DatatypeMessage, H5File};

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
        "rust_hdf5_sohm_write_{}_{}_{}",
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

/// The mask `gen_sohm.c` uses: datatype, dataspace and attribute messages.
fn all_three() -> u16 {
    type_flag(MSG_DATATYPE).unwrap()
        | type_flag(MSG_DATASPACE).unwrap()
        | type_flag(MSG_ATTRIBUTE).unwrap()
}

/// Write the `gen_sohm.c` content into `path` under one index.
fn write_sohm_file(path: &PathBuf, types: u16, min_mesg_size: u32, list_max: u16, btree_min: u16) {
    let file = H5File::options()
        .shared_messages(&[(types, min_mesg_size)], list_max, btree_min)
        .create(path)
        .unwrap();
    for i in 0..4i32 {
        let ds = file
            .new_dataset::<i32>()
            .shape([8usize])
            .create(&format!("shared{i}"))
            .unwrap();
        ds.write_raw(&(0..8i32).map(|j| i * 10 + j).collect::<Vec<_>>())
            .unwrap();
        ds.new_attr::<f64>()
            .shape([3usize])
            .create("cal")
            .unwrap()
            .write_array(&[0.5f64, 1.5, 2.5])
            .unwrap();
    }
    file.commit_datatype("named_i32", DatatypeMessage::i32_type())
        .unwrap();
    file.new_dataset::<i32>()
        .committed_type("named_i32")
        .shape([8usize])
        .create("uses_named")
        .unwrap()
        .write_raw(&(100..108i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();
}

/// Every object the file was asked for, read back through this crate's own
/// reader — which has to resolve each pointer to get any of it.
fn check_contents(path: &PathBuf) {
    let file = H5File::open(path).unwrap();
    let mut names = file.dataset_names();
    names.sort();
    assert_eq!(
        names,
        vec!["shared0", "shared1", "shared2", "shared3", "uses_named"]
    );
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
}

/// The master table the file's superblock extension names, decoded from the
/// bytes on disk.
fn read_master_table(path: &PathBuf) -> SohmMasterTable {
    let ctx = FormatContext::default_v3();
    let table = {
        let file = H5File::open(path).unwrap();
        file.superblock_extension()
            .shared_message_table
            .expect("a file with shared messages names its table in the extension")
    };
    let bytes = std::fs::read(path).unwrap();
    let at = table.table_address as usize;
    let size = SohmMasterTable::encoded_size(&ctx, table.nindexes);
    SohmMasterTable::decode(&bytes[at..at + size], &ctx, table.nindexes).unwrap()
}

/// The root group's object header, decoded from the bytes on disk.
fn read_root_header(path: &PathBuf) -> ObjectHeader {
    let bytes = std::fs::read(path).unwrap();
    let superblock = SuperblockV2V3::decode(&bytes).unwrap();
    let at = (superblock.base_address + superblock.root_group_object_header_address) as usize;
    ObjectHeader::decode(&bytes[at..]).unwrap().0
}

// -------------------------------------------------------------- the census

/// A `BlockReader` over a whole file already in memory.
struct Bytes<'a>(&'a [u8]);

impl BlockReader for Bytes<'_> {
    fn read_block(&mut self, offset: u64, len: usize) -> FormatResult<Vec<u8>> {
        let at = offset as usize;
        let end = (at + len).min(self.0.len());
        Ok(self.0[at..end].to_vec())
    }
}

/// What one shared-message record holds, said in terms two writers can agree
/// on: the decoded meaning of the body rather than the bytes, since the
/// message version a writer picks is its own business.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    /// The decoded datatype, printed. Both writers encode these identically
    /// here, so the whole value is compared.
    Datatype(String),
    /// Dataspace class, current dimensions and maximum dimensions. The
    /// maximum is part of the comparison because both writers store one:
    /// `H5S_set_extent_simple` fills `extent.max` in from `dims` when the
    /// caller named no maximum (H5S.c:1293-1299), and the crate's
    /// constructors do the same.
    Dataspace(String, Vec<u64>, Option<Vec<u64>>),
    /// An attribute body: its name, its flags byte, and — in field order —
    /// the role and reference count of each record its own datatype and
    /// dataspace fields name.
    Attribute {
        name: String,
        flags: u8,
        fields: Vec<(Role, u32)>,
    },
}

/// Every record of every index of `path`, as `(role, reference count)`,
/// sorted so two files can be compared whatever order their indexes grew in.
///
/// This is the census libhdf5's own `h5debug` prints for a SOHM list, read
/// out of the bytes instead: one entry per heap object, carrying the count
/// the record claims.
fn census(path: &PathBuf) -> Vec<(Role, u32)> {
    let ctx = FormatContext::default_v3();
    let table = read_master_table(path);
    let bytes = std::fs::read(path).unwrap();

    // Every record of every index first, so that an attribute body can name a
    // datatype that lives in a different index's heap.
    let mut records: Vec<(usize, [u8; SOHM_HEAP_ID_LEN], u32, Vec<u8>)> = Vec::new();
    for (i, header) in table.indexes.iter().enumerate() {
        let mut reader = Bytes(&bytes);
        let raw = reader.read_block(header.heap_addr, 512).unwrap();
        let heap = FractalHeapHeader::decode(&raw, &ctx).unwrap();
        let blocks = collect_managed_blocks(&heap, &ctx, &mut reader).unwrap();

        let size = record_size(&ctx);
        let at = header.index_addr as usize;
        let entries = if header.index_type == SOHM_INDEX_LIST {
            assert_eq!(&bytes[at..at + 4], &SMLI_SIGNATURE);
            bytes[at + 4..at + 4 + size * header.num_messages as usize].to_vec()
        } else {
            let bt2 = Bt2Header::decode(&bytes[at..], &ctx).unwrap();
            collect_btree_v2_records(&bt2, &ctx, &mut Bytes(&bytes)).unwrap()
        };
        for entry in entries.chunks_exact(size) {
            assert_eq!(
                entry[0], SOHM_IN_HEAP,
                "this census reads heap bodies; the record is in an object header"
            );
            let id: [u8; SOHM_HEAP_ID_LEN] = entry[9..9 + SOHM_HEAP_ID_LEN].try_into().unwrap();
            let parsed = HeapId::parse(&id, &heap, &ctx).unwrap();
            let body = read_heap_object(&parsed, &heap, &ctx, &blocks, &mut reader).unwrap();
            let ref_count = u32::from_le_bytes(entry[5..9].try_into().unwrap());
            records.push((i, id, ref_count, body));
        }
    }

    let mut out: Vec<(Role, u32)> = records
        .iter()
        .map(|(_, _, ref_count, body)| (role(&ctx, body, &table, &records), *ref_count))
        .collect();
    out.sort();
    out
}

/// The message bytes a shared pointer names, with the reference count of the
/// record it named.
fn target(
    ctx: &FormatContext,
    field: &[u8],
    msg_type: u8,
    table: &SohmMasterTable,
    records: &[(usize, [u8; SOHM_HEAP_ID_LEN], u32, Vec<u8>)],
) -> (Vec<u8>, u32) {
    let pointer = SharedMessagePointer::decode(field, ctx).unwrap();
    assert_eq!(pointer.location, SharedLocation::Sohm);
    // Which heap the ID belongs to is decided by the pointer's message class,
    // the way `H5SM_get_fheap_addr` resolves one.
    let heap = table.heap_addr(msg_type).unwrap();
    let index = table
        .indexes
        .iter()
        .position(|h| h.heap_addr == heap)
        .unwrap();
    let (_, _, ref_count, body) = records
        .iter()
        .find(|(i, id, _, _)| *i == index && *id == pointer.heap_id)
        .expect("a shared pointer inside a heap body names a record of that index");
    (body.clone(), *ref_count)
}

/// Classify one heap body by which message decoder accepts all of it.
fn role(
    ctx: &FormatContext,
    body: &[u8],
    table: &SohmMasterTable,
    records: &[(usize, [u8; SOHM_HEAP_ID_LEN], u32, Vec<u8>)],
) -> Role {
    // A record carries no message class, so the body has to say what it is.
    // An attribute is the only one of the three whose header accounts for its
    // own total length, so try that first and let the exact fit decide.
    if let Some(role) = attribute_role(ctx, body, table, records) {
        return role;
    }
    if let Ok((dt, n)) = DatatypeMessage::decode(body, ctx) {
        if n == body.len() {
            return Role::Datatype(format!("{dt:?}"));
        }
    }
    let (ds, n) = DataspaceMessage::decode(body, ctx).expect("a heap body is one of the three");
    assert_eq!(n, body.len());
    Role::Dataspace(format!("{:?}", ds.class), ds.dims, ds.max_dims)
}

/// `body` as an attribute, or `None` when it is not one: the header must
/// account for every byte, down to the elements its own datatype and
/// dataspace describe.
fn attribute_role(
    ctx: &FormatContext,
    body: &[u8],
    table: &SohmMasterTable,
    records: &[(usize, [u8; SOHM_HEAP_ID_LEN], u32, Vec<u8>)],
) -> Option<Role> {
    if body.len() < 9 || !(2..=3).contains(&body[0]) {
        return None;
    }
    let flags = body[1];
    let name_size = u16::from_le_bytes([body[2], body[3]]) as usize;
    let dt_size = u16::from_le_bytes([body[4], body[5]]) as usize;
    let ds_size = u16::from_le_bytes([body[6], body[7]]) as usize;
    let name_at: usize = if body[0] >= 3 { 9 } else { 8 };
    let name_end = name_at.checked_add(name_size)?;
    let dt_end = name_end.checked_add(dt_size)?;
    let ds_end = dt_end.checked_add(ds_size)?;
    if ds_end > body.len() || name_size == 0 {
        return None;
    }
    let name = String::from_utf8(body[name_at..name_end - 1].to_vec()).ok()?;

    // A shared field is a pointer to the record holding the message; a
    // literal one is the message. Either way the bytes decode the same, and
    // only a shared one carries a reference count of its own.
    let field = |range: std::ops::Range<usize>, msg_type: u8, mask: u8| {
        if flags & mask != 0 {
            target(ctx, &body[range], msg_type, table, records)
        } else {
            (body[range].to_vec(), 0)
        }
    };
    let (dt_bytes, dt_refs) = field(name_end..dt_end, MSG_DATATYPE, ATTR_FLAG_TYPE_SHARED);
    let (ds_bytes, ds_refs) = field(dt_end..ds_end, MSG_DATASPACE, ATTR_FLAG_SPACE_SHARED);
    let (datatype, _) = DatatypeMessage::decode(&dt_bytes, ctx).ok()?;
    let (dataspace, _) = DataspaceMessage::decode(&ds_bytes, ctx).ok()?;

    let elements: u64 = dataspace.dims.iter().copied().product::<u64>().max(1);
    let width = u64::from(datatype.element_size_ctx(ctx));
    if ds_end as u64 + elements * width != body.len() as u64 {
        return None;
    }
    Some(Role::Attribute {
        name,
        flags,
        fields: vec![
            (Role::Datatype(format!("{datatype:?}")), dt_refs),
            (
                Role::Dataspace(
                    format!("{:?}", dataspace.class),
                    dataspace.dims,
                    dataspace.max_dims,
                ),
                ds_refs,
            ),
        ],
    })
}

/// One index over all three classes: the bodies five datasets share end up in
/// the heap once each, and the file still reads back whole.
#[test]
fn a_created_file_shares_the_bodies_its_index_covers() {
    let path = unique_tmp("list");
    write_sohm_file(&path, all_three(), 0, 50, 40);
    check_contents(&path);

    let table = read_master_table(&path);
    assert_eq!(table.indexes.len(), 1);
    let index = &table.indexes[0];
    assert_eq!(index.index_type, SOHM_INDEX_LIST);
    assert_eq!(index.mesg_types, all_three());
    assert_eq!(index.list_max, 50);
    assert_eq!(index.btree_min, 40);
    // The dataspace every dataset has, the attribute body, and the datatype
    // and dataspace that body points at. The datasets' own datatype is the
    // predefined `H5T_STD_I32LE`, which `H5O__dtype_can_share` refuses
    // (H5Odtype.c:1893), and `uses_named` reaches its type through the
    // committed datatype, which is shared by address instead.
    assert_eq!(index.num_messages, 4);

    let bytes = std::fs::read(&path).unwrap();
    let at = index.index_addr as usize;
    assert_eq!(&bytes[at..at + 4], &SMLI_SIGNATURE);
    cleanup(&path);
}

/// A body only one object uses never reaches the heap: it stays literal in
/// the header that offered it, carrying `H5O_MSG_FLAG_SHAREABLE` instead of
/// becoming a pointer (H5SM.c:1112), and the index files an `H5SM_IN_OH`
/// record naming that header (H5SM.c:1400-1417).
///
/// libhdf5 writes the same file here, so the record is compared byte for
/// byte apart from the header address the two writers place differently.
#[test]
fn a_body_only_one_object_uses_stays_in_its_header() {
    let path = unique_tmp("in_ohdr");
    let dataspace = type_flag(MSG_DATASPACE).unwrap();
    {
        let file = H5File::options()
            .shared_messages(&[(dataspace, 0)], 50, 40)
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([8usize])
            .create("only")
            .unwrap()
            .write_raw(&(0..8i32).collect::<Vec<_>>())
            .unwrap();
        file.close().unwrap();
    }

    let ctx = FormatContext::default_v3();
    let bytes = std::fs::read(&path).unwrap();
    let table = read_master_table(&path);
    assert_eq!(table.indexes.len(), 1);
    let index = &table.indexes[0];
    assert_eq!(index.num_messages, 1);

    let at = index.index_addr as usize;
    assert_eq!(&bytes[at..at + 4], &SMLI_SIGNATURE);
    let record = &bytes[at + 4..at + 4 + record_size(&ctx)];
    assert_eq!(record[0], SOHM_IN_OH);
    assert_eq!(record[6], MSG_DATASPACE, "the class the record names");
    let oh_addr = u64::from_le_bytes(record[9..17].try_into().unwrap());

    // The header the record names holds the body itself, marked shareable
    // and not shared.
    let (header, _) = ObjectHeader::decode(&bytes[oh_addr as usize..]).unwrap();
    let dataspace_msg = header
        .messages
        .iter()
        .find(|m| m.msg_type == MSG_DATASPACE)
        .expect("the header the record names has a dataspace message");
    assert_eq!(dataspace_msg.flags & MSG_FLAG_SHARED, 0);
    assert_ne!(dataspace_msg.flags & MSG_FLAG_SHAREABLE, 0);
    DataspaceMessage::decode(&dataspace_msg.data, &ctx).expect("the body is a dataspace, literal");

    // And the file still reads back.
    let file = H5File::open(&path).unwrap();
    let data: Vec<i32> = file.dataset("only").unwrap().read_raw().unwrap();
    assert_eq!(data, (0..8i32).collect::<Vec<_>>());
    drop(file);

    libhdf5_writes_the_same_single_use_record(record);
    cleanup(&path);
}

/// The libhdf5 half of [`a_body_only_one_object_uses_stays_in_its_header`]:
/// the same one-dataset file written by libhdf5 itself, whose record must
/// take the same form, hash the same body and name the same class. Only the
/// header address differs, the two writers laying a file out differently.
///
/// h5py exposes no `H5Pset_shared_mesg_*`, so the creation properties are set
/// through `ctypes` on the libhdf5 h5py is linked against — the same library
/// `tests/fixtures/gen_sohm.c` was run against.
fn libhdf5_writes_the_same_single_use_record(record: &[u8]) {
    let Some(py) = python() else { return };
    let theirs = unique_tmp("in_ohdr_h5py");
    let script = format!(
        "import ctypes, glob, os, sys, h5py, numpy as np\n\
         so = sorted(glob.glob(os.path.join(sys.prefix, 'lib', 'libhdf5.so.*')))\n\
         if not so:\n\
         \x20   raise SystemExit(0)\n\
         h5 = ctypes.CDLL(so[0])\n\
         h5.H5Pcreate.restype = h5.H5Fcreate.restype = ctypes.c_int64\n\
         h5.H5Pcreate.argtypes = [ctypes.c_int64]\n\
         h5.H5Fcreate.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_int64, ctypes.c_int64]\n\
         h5.H5Pset_shared_mesg_nindexes.argtypes = [ctypes.c_int64, ctypes.c_uint]\n\
         h5.H5Pset_shared_mesg_index.argtypes = [ctypes.c_int64, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]\n\
         h5.H5Pset_shared_mesg_phase_change.argtypes = [ctypes.c_int64, ctypes.c_uint, ctypes.c_uint]\n\
         h5.H5Pclose.argtypes = h5.H5Fclose.argtypes = [ctypes.c_int64]\n\
         fcpl = h5.H5Pcreate(ctypes.c_int64.in_dll(h5, 'H5P_CLS_FILE_CREATE_ID_g').value)\n\
         assert fcpl >= 0\n\
         assert h5.H5Pset_shared_mesg_nindexes(fcpl, 1) >= 0\n\
         assert h5.H5Pset_shared_mesg_index(fcpl, 0, 0x0002, 0) >= 0\n\
         assert h5.H5Pset_shared_mesg_phase_change(fcpl, 50, 40) >= 0\n\
         fid = h5.H5Fcreate(r'{}'.encode(), 2, fcpl, 0)\n\
         assert fid >= 0\n\
         h5.H5Pclose(fcpl)\n\
         h5.H5Fclose(fid)\n\
         f = h5py.File(r'{}', 'r+')\n\
         f.create_dataset('only', data=np.arange(8, dtype='<i4'))\n\
         f.close()\n",
        theirs.display(),
        theirs.display(),
    );
    run(py, &["-c", &script], "libhdf5 single-use shared dataspace");
    if !theirs.exists() {
        return;
    }

    let ctx = FormatContext::default_v3();
    let bytes = std::fs::read(&theirs).unwrap();
    let table = read_master_table(&theirs);
    let index = &table.indexes[0];
    assert_eq!(index.num_messages, 1);
    let at = index.index_addr as usize + 4;
    let theirs_record = &bytes[at..at + record_size(&ctx)];
    // Location, hash, reserved byte, class and creation index — everything
    // but the address, which is where the two layouts part.
    assert_eq!(theirs_record[..9], record[..9]);
    cleanup(&theirs);
}

/// The shared attribute this crate writes holds pointers to its own datatype
/// and dataspace, and the file reads back through those pointers.
///
/// `H5A__create` shares an attribute's datatype and dataspace before the
/// attribute itself (H5Aint.c:375-378), so `H5O__attr_encode` writes each as
/// a version-3 shared pointer and records which in the attribute's flags byte
/// (H5Oattr.c:346-360). The components end at one reference each — the
/// pointer lives in the one heap object every dataset's header names, not in
/// each header.
/// The public accessor the oracle's `shared` field is read through, checked
/// against `h5debug` on the libhdf5-written fixture.
///
/// `h5debug tests/fixtures/sohm_list.h5 <addr>` prints, per object:
/// `/shared0` dataspace `<SA>` and attribute `<S>`/SOHM; `/shared1..3`
/// dataspace and attribute both `<S>`/SOHM; `/uses_named` dataspace
/// `<S>`/SOHM and datatype `<C, S>`/Obj Hdr; `/named_i32` and the root group
/// nothing at all. The first is the case `H5Oget_info`'s `hdr.mesg.shared`
/// mask cannot express: `H5SM__write_mesg` left the body literal in the
/// header that offered it (H5SM.c:1400-1417), so the flag is
/// `H5O_MSG_FLAG_SHAREABLE` and the mask, which counts only
/// `H5O_MSG_FLAG_SHARED`, reads zero (H5Oint.c:2072-2073).
#[test]
fn the_storage_accessor_answers_what_h5debug_prints_for_the_fixture() {
    use rust_hdf5::format::messages::shared::MessageStorage;
    use rust_hdf5::format::sohm::SharedLocation;

    let file = H5File::open(fixture("sohm_list.h5")).unwrap();
    let seen: Vec<(&str, Vec<(u8, MessageStorage)>)> =
        ["/", "/named_i32", "/shared0", "/shared1", "/uses_named"]
            .into_iter()
            .map(|p| (p, file.object_message_storage(p).unwrap()))
            .collect();
    assert_eq!(
        seen,
        vec![
            ("/", vec![]),
            ("/named_i32", vec![]),
            (
                "/shared0",
                vec![
                    (MSG_DATASPACE, MessageStorage::Shareable),
                    (MSG_ATTRIBUTE, MessageStorage::Shared(SharedLocation::Sohm)),
                ]
            ),
            (
                "/shared1",
                vec![
                    (MSG_DATASPACE, MessageStorage::Shared(SharedLocation::Sohm)),
                    (MSG_ATTRIBUTE, MessageStorage::Shared(SharedLocation::Sohm)),
                ]
            ),
            (
                "/uses_named",
                vec![
                    (MSG_DATASPACE, MessageStorage::Shared(SharedLocation::Sohm)),
                    (
                        MSG_DATATYPE,
                        MessageStorage::Shared(SharedLocation::Committed)
                    ),
                ]
            ),
        ]
    );
}

#[test]
fn a_shared_attribute_points_at_its_own_datatype_and_dataspace() {
    let path = unique_tmp("nested");
    write_sohm_file(&path, all_three(), 0, 50, 40);

    let census = census(&path);
    let (attribute, refs) = census
        .iter()
        .find(|(role, _)| matches!(role, Role::Attribute { .. }))
        .expect("the index holds the attribute body");
    let Role::Attribute {
        name,
        flags,
        fields,
    } = attribute
    else {
        unreachable!()
    };
    assert_eq!(name, "cal");
    assert_eq!(*flags, ATTR_FLAG_TYPE_SHARED | ATTR_FLAG_SPACE_SHARED);
    assert_eq!(*refs, 4, "one reference per dataset holding the attribute");
    assert!(
        matches!(fields[0], (Role::Datatype(_), 1)),
        "the attribute's datatype is a record of its own, named once: {fields:?}"
    );
    assert_eq!(
        fields[1],
        (Role::Dataspace("Simple".into(), vec![3], Some(vec![3])), 1),
        "and so is its dataspace"
    );

    // The reader resolves what the writer nested: each attribute comes back
    // with the datatype and the values it was written with.
    let file = H5File::open(&path).unwrap();
    for i in 0..4i32 {
        let attr = file
            .dataset(&format!("shared{i}"))
            .unwrap()
            .attr("cal")
            .unwrap();
        assert_eq!(attr.datatype().unwrap(), DatatypeMessage::f64_type());
        let values: Vec<f64> = attr.read_numeric_as().unwrap();
        assert_eq!(values, vec![0.5, 1.5, 2.5]);
    }
    cleanup(&path);
}

/// Census parity: for the construction `sohm_list.h5` was written from, this
/// crate's index holds the same records with the same reference counts
/// libhdf5's does.
///
/// That fixture gives its datasets the predefined `H5T_STD_I32LE` itself,
/// which is what `new_dataset::<i32>()` names too — and what
/// `H5O__dtype_can_share` refuses, being immutable (H5Odtype.c:1893). So the
/// datasets' datatype is in neither file's index, and the four records are the
/// datasets' dataspace, the attribute body, and the datatype and dataspace
/// that body points at.
#[test]
fn the_record_census_matches_libhdf5_for_the_same_construction() {
    let path = unique_tmp("parity");
    write_sohm_file(&path, all_three(), 0, 50, 40);

    let theirs = census(&fixture("sohm_list.h5"));
    let ours = census(&path);
    assert_eq!(theirs.len(), 4, "four records: {theirs:#?}");
    assert!(
        !theirs
            .iter()
            .any(|(role, _)| matches!(role, Role::Datatype(t) if t.contains("FixedPoint"))),
        "libhdf5 leaves the predefined datatype out of the index: {theirs:#?}"
    );
    assert_eq!(ours, theirs);
    cleanup(&path);
}

/// The other side of that rule: a datatype no predefined type matches is
/// shared, because the libhdf5 program writing it would have had to build it
/// with `H5Tcopy`/`H5Tset_size` and so hands `H5Dcreate2` a mutable type.
///
/// `sohm_nested.h5` is the fixture for this half — its datasets take
/// `H5Tcopy(H5T_STD_I32LE)`, whose message is byte-identical to the
/// predefined type's, so the difference is one of type state and not of
/// bytes. This crate has no `H5Tcopy`, so the shareable case it can express
/// is a type whose *definition* no predefined type has: an eight-byte fixed
/// string is `H5Tcopy(H5T_C_S1)` plus `H5Tset_size` in every writer.
#[test]
fn a_datatype_no_predefined_type_matches_is_still_shared() {
    let path = unique_tmp("mutable_dtype");
    let file = H5File::options()
        .shared_messages(&[(all_three(), 0)], 50, 40)
        .create(&path)
        .unwrap();
    for i in 0..3 {
        file.new_dataset::<u8>()
            .datatype(DatatypeMessage::fixed_string(8))
            .shape([2usize])
            .create(&format!("s{i}"))
            .unwrap();
    }
    file.close().unwrap();

    let census = census(&path);
    let types: Vec<&Role> = census.iter().map(|(role, _)| role).collect();
    assert!(
        census.iter().any(
            |(role, refs)| matches!(role, Role::Datatype(t) if t.contains("FixedString"))
                && *refs == 3
        ),
        "the fixed-string datatype is one record named by all three datasets: {types:#?}"
    );
    cleanup(&path);
}

// -------------------------------------------------------------- libhdf5

/// The same Python-install probe the other cross-check suites use: the tools
/// beside it belong to the libhdf5 h5py is linked against.
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
            eprintln!("skipping SOHM write cross-check: none of {candidates:?} present");
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

/// h5py reads the file this crate wrote, and h5diff finds no difference
/// between it and the libhdf5-written twin.
///
/// The census test says the two files hold the same records; this says
/// libhdf5 itself agrees about what those records mean.
#[test]
fn libhdf5_reads_the_file_and_diffs_it_clean_against_its_twin() {
    let Some(py) = python() else { return };
    let path = unique_tmp("h5py_readback");
    write_sohm_file(&path, all_three(), 0, 50, 40);

    let script = format!(
        "import h5py\n\
         f = h5py.File(r'{}', 'r')\n\
         assert sorted(f) == ['named_i32', 'shared0', 'shared1', 'shared2', 'shared3', \
         'uses_named'], sorted(f)\n\
         for i in range(4):\n\
         \x20   d = f['shared%d' % i]\n\
         \x20   assert list(d[()]) == [i * 10 + j for j in range(8)], list(d[()])\n\
         \x20   a = d.attrs['cal']\n\
         \x20   assert a.dtype == '<f8', a.dtype\n\
         \x20   assert a.shape == (3,), a.shape\n\
         \x20   assert list(a) == [0.5, 1.5, 2.5], list(a)\n\
         assert list(f['uses_named'][()]) == list(range(100, 108))\n\
         assert f['uses_named'].id.get_type().committed()\n",
        path.display()
    );
    run(py, &["-c", &script], "h5py read-back of the nested file");

    let tools = std::path::Path::new(py).parent().unwrap();
    let h5diff = tools.join("h5diff");
    if h5diff.exists() {
        // `-c` also reports an object that is in one file and not the
        // other, which a plain run passes over.
        run(
            &h5diff,
            &[
                "-c",
                path.to_str().unwrap(),
                fixture("sohm_list.h5").to_str().unwrap(),
            ],
            "h5diff against the libhdf5-written twin",
        );
    }
    cleanup(&path);
}

/// The eligibility rule as libhdf5 itself reports it: `H5Oget_info`'s
/// `hdr.mesg.shared` bitmask says which message classes a header holds as a
/// shared-message pointer, and for the predefined-datatype construction the
/// datatype bit must be clear in this crate's file exactly as it is in
/// libhdf5's.
///
/// The dataspace bit is asserted too, so a rule that simply stopped sharing
/// would fail this rather than pass it.
///
/// `shared0` is the object the two writers used to disagree about, so its
/// whole mask is compared rather than probed bit by bit: `H5SM__write_mesg`
/// leaves the first copy of a share-in-object-header class literal in the
/// header that offered it (H5SM.c:1400-1417), and `H5O__get_hdr_info` only
/// counts a message the header stores as a pointer (H5Oint.c:2072-2073), so
/// the dataspace bit is *clear* on the first of the four datasets and set on
/// the rest.
#[test]
fn h5py_sees_the_predefined_datatype_left_out_of_the_index() {
    let Some(py) = python() else { return };
    let path = unique_tmp("h5py_shared_mask");
    write_sohm_file(&path, all_three(), 0, 50, 40);

    let script = format!(
        "import h5py\n\
         DTYPE, SDSPACE, ATTR = 1 << 3, 1 << 1, 1 << 12\n\
         masks = []\n\
         for name in (r'{}', r'{}'):\n\
         \x20   f = h5py.File(name, 'r')\n\
         \x20   m = h5py.h5o.get_info(f['shared1'].id).hdr.mesg.shared\n\
         \x20   assert not m & DTYPE, (name, hex(m))\n\
         \x20   assert m & SDSPACE and m & ATTR, (name, hex(m))\n\
         \x20   first = h5py.h5o.get_info(f['shared0'].id).hdr.mesg.shared\n\
         \x20   assert not first & SDSPACE, (name, hex(first))\n\
         \x20   masks.append((first, m))\n\
         \x20   f.close()\n\
         assert masks[0] == masks[1], masks\n",
        path.display(),
        fixture("sohm_list.h5").display(),
    );
    run(py, &["-c", &script], "h5py shared-message mask check");
    cleanup(&path);
}

/// `H5Pset_shared_mesg_phase_change(fcpl, 0, 0)` — the `sohm_btree` fixture's
/// setting — puts the index in B-tree form from the first message.
#[test]
fn a_zero_list_maximum_writes_a_btree_index() {
    let path = unique_tmp("btree");
    write_sohm_file(&path, all_three(), 0, 0, 0);
    check_contents(&path);

    let table = read_master_table(&path);
    let index = &table.indexes[0];
    assert_eq!(index.index_type, SOHM_INDEX_BTREE);
    assert_eq!(index.num_messages, 4);

    let bytes = std::fs::read(&path).unwrap();
    let at = index.index_addr as usize;
    assert_eq!(&bytes[at..at + 4], b"BTHD");
    assert_eq!(bytes[at + 5], BT2_TYPE_SOHM_INDEX);
    assert_eq!(
        u32::from_le_bytes(bytes[at + 6..at + 10].try_into().unwrap()),
        SOHM_B2_NODE_SIZE
    );
    assert_eq!(
        u16::from_le_bytes(bytes[at + 10..at + 12].try_into().unwrap()),
        17,
        "H5SM_SOHM_ENTRY_SIZE for eight-byte addresses"
    );
    cleanup(&path);
}

/// An index takes only the classes its mask names. Covering just dataspaces
/// leaves the datatype and attribute messages in the headers, so the index
/// holds the two dataspaces this file has: the datasets' and the one the
/// attributes carry, which `H5A__create` offers whether or not the attribute
/// around it is itself shared (H5Aint.c:375-378).
#[test]
fn an_index_takes_only_the_classes_its_mask_names() {
    let path = unique_tmp("mask");
    write_sohm_file(&path, type_flag(MSG_DATASPACE).unwrap(), 0, 50, 40);
    check_contents(&path);

    let table = read_master_table(&path);
    assert_eq!(table.indexes[0].num_messages, 2);
    cleanup(&path);
}

/// `H5Pset_shared_mesg_index`'s minimum size: a message under it stays in the
/// header. Every body this file would share is well under 4 KiB, so the index
/// ends up empty — and an empty index is still written, the way
/// `H5SM__create_index` makes one at file creation.
#[test]
fn a_minimum_size_above_every_body_leaves_an_empty_index() {
    let path = unique_tmp("minsize");
    write_sohm_file(&path, all_three(), 4096, 50, 40);
    check_contents(&path);

    let table = read_master_table(&path);
    assert_eq!(table.indexes[0].num_messages, 0);
    cleanup(&path);
}

/// The default is unchanged: a file created without the option has no
/// superblock extension at all.
#[test]
fn a_file_created_without_indexes_has_no_shared_message_table() {
    let path = unique_tmp("plain");
    let file = H5File::create(&path).unwrap();
    file.new_dataset::<i32>()
        .shape([8usize])
        .create("data")
        .unwrap()
        .write_raw(&(0..8i32).collect::<Vec<_>>())
        .unwrap();
    file.close().unwrap();

    let file = H5File::open(&path).unwrap();
    assert!(file.superblock_extension().shared_message_table.is_none());
    cleanup(&path);
}

/// Two indexes, each over its own classes: every message goes to the heap of
/// the index whose mask covers it.
#[test]
fn two_indexes_split_the_message_classes_between_them() {
    let path = unique_tmp("split");
    let file = H5File::options()
        .shared_messages(
            &[
                (type_flag(MSG_ATTRIBUTE).unwrap(), 0),
                (
                    type_flag(MSG_DATATYPE).unwrap() | type_flag(MSG_DATASPACE).unwrap(),
                    0,
                ),
            ],
            50,
            40,
        )
        .create(&path)
        .unwrap();
    for i in 0..3i32 {
        let ds = file
            .new_dataset::<i32>()
            .shape([8usize])
            .create(&format!("d{i}"))
            .unwrap();
        ds.write_raw(&(0..8i32).collect::<Vec<_>>()).unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_numeric(&7i32)
            .unwrap();
    }
    file.close().unwrap();

    let table = read_master_table(&path);
    assert_eq!(table.indexes.len(), 2);
    // One attribute body in the first index; in the second, the datasets'
    // dataspace, the datatype the datasets and the attribute share, and the
    // scalar dataspace the attribute body points at across the index
    // boundary.
    assert_eq!(table.indexes[0].num_messages, 1);
    assert_eq!(table.indexes[1].num_messages, 3);
    assert_ne!(table.indexes[0].heap_addr, table.indexes[1].heap_addr);
    assert_eq!(
        table.heap_addr(MSG_ATTRIBUTE),
        Some(table.indexes[0].heap_addr)
    );
    assert_eq!(
        table.heap_addr(MSG_DATASPACE),
        Some(table.indexes[1].heap_addr)
    );

    let file = H5File::open(&path).unwrap();
    for i in 0..3i32 {
        let ds = file.dataset(&format!("d{i}")).unwrap();
        let data: Vec<i32> = ds.read_raw().unwrap();
        assert_eq!(data, (0..8).collect::<Vec<i32>>());
        let tag: Vec<i32> = ds.attr("tag").unwrap().read_numeric_as().unwrap();
        assert_eq!(tag, vec![7]);
    }
    cleanup(&path);
}

/// A shared attribute is found again through its message creation index, so
/// `H5SM_init` sets `store_msg_crt_idx` on a file whose index covers
/// attributes and every object header created afterwards records creation
/// indices — whatever the object's creation property list asked for. An index
/// over the other two classes leaves the headers alone.
#[test]
fn sharing_attributes_makes_every_header_record_creation_indices() {
    let path = unique_tmp("crtidx");
    write_sohm_file(&path, all_three(), 0, 50, 40);
    assert_eq!(
        read_root_header(&path).attribute_creation_order(),
        CreationOrder::Tracked
    );
    cleanup(&path);

    let path = unique_tmp("nocrtidx");
    write_sohm_file(
        &path,
        type_flag(MSG_DATATYPE).unwrap() | type_flag(MSG_DATASPACE).unwrap(),
        0,
        50,
        40,
    );
    assert_eq!(
        read_root_header(&path).attribute_creation_order(),
        CreationOrder::Untracked
    );
    cleanup(&path);
}

/// The configurations `H5Pset_shared_mesg_nindexes` and
/// `H5Pset_shared_mesg_phase_change` reject are rejected here too, at the
/// point the file is created.
#[test]
fn file_creation_refuses_a_configuration_libhdf5_refuses() {
    let too_many: Vec<(u16, u32)> = (0..9).map(|_| (all_three(), 0)).collect();
    let path = unique_tmp("invalid");
    for (indexes, list_max, btree_min, expect) in [
        (too_many.as_slice(), 50u16, 40u16, "at most 8"),
        (&[(all_three(), 0)][..], 10, 40, "btree_min"),
        (&[(0, 0)][..], 50, 40, "covering no message type"),
    ] {
        let err = match H5File::options()
            .shared_messages(indexes, list_max, btree_min)
            .create(&path)
        {
            Ok(_) => panic!("an invalid shared-message configuration must not create a file"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(expect), "{err}");
    }
    cleanup(&path);
}

/// Every object header reachable from the root, by path, decoded from the
/// bytes. Reaching one at all is the version assertion: `ObjectHeader::decode`
/// requires the `OHDR` signature, which a version-1 header does not have.
fn every_header(bytes: &[u8]) -> Vec<(String, ObjectHeader)> {
    use rust_hdf5::format::messages::link::{LinkMessage, LinkTarget};
    use rust_hdf5::format::messages::MSG_LINK;

    let sb = SuperblockV2V3::decode(bytes).unwrap();
    let ctx = FormatContext {
        sizeof_addr: sb.sizeof_offsets,
        sizeof_size: sb.sizeof_lengths,
    };
    let mut out = Vec::new();
    let mut queue = vec![(String::from("/"), sb.root_group_object_header_address)];
    while let Some((path, addr)) = queue.pop() {
        let at = (sb.base_address + addr) as usize;
        let (header, _) = ObjectHeader::decode(&bytes[at..])
            .unwrap_or_else(|e| panic!("the header of {path} is not a version-2 one: {e}"));
        for m in header.messages.iter().filter(|m| m.msg_type == MSG_LINK) {
            if let Ok((link, _)) = LinkMessage::decode(&m.data, &ctx) {
                if let LinkTarget::Hard { address } = link.target {
                    queue.push((format!("{}{}/", path, link.name), address));
                }
            }
        }
        out.push((path, header));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `store_msg_crt_idx` is a property of the *file*, so the floor it puts under
/// attribute creation order reaches every object header the file holds, not
/// just the root's: `H5O__create_ohdr` raises each one it makes to version 2
/// and ORs `H5O_HDR_ATTR_CRT_ORDER_TRACKED` in (H5Oint.c:364, H5Oint.c:442).
///
/// The other half is that such a file can never be a classic one. libhdf5
/// reaches that by raising the superblock to version 2 for `sohm_nindexes > 0`
/// and then refusing the file outright when the high bound cannot reach that
/// version (H5Fsuper.c:1135). Here the combination is unrepresentable instead:
/// `shared_messages` is read only by `create`, which never writes a superblock
/// below version 2, and `open_rw` — the one path that produces a classic
/// writer — refuses the option outright rather than reading it, so the indexes
/// a reopened file has are the ones it was created with and never a set this
/// session asked for. So a classic file keeps its version-1 headers with the
/// floor nowhere in sight.
#[test]
fn the_creation_index_floor_reaches_every_header_and_never_a_classic_file() {
    let path = unique_tmp("crtidx_every");
    {
        let file = H5File::options()
            .shared_messages(&[(all_three(), 0)], 50, 40)
            .create(&path)
            .unwrap();
        let group = file.root_group().create_group("g").unwrap();
        group.set_attr_numeric("scale", &2.0f64).unwrap();
        for (parent, name) in [(&file.root_group(), "outer"), (&group, "inner")] {
            let ds = parent.new_dataset::<i32>().shape([4]).create(name).unwrap();
            ds.write_raw(&[7i32; 4]).unwrap();
            // The same body in both headers, so the index actually shares it.
            ds.new_attr::<f64>()
                .shape(())
                .create("units")
                .unwrap()
                .write_numeric(&1.5f64)
                .unwrap();
        }
        file.close().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert!(SuperblockV2V3::decode(&bytes).unwrap().version >= 2);
    let headers = every_header(&bytes);
    assert_eq!(
        headers.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        ["/", "/g/", "/g/inner/", "/outer/"]
    );
    for (path, header) in &headers {
        assert_eq!(
            header.attribute_creation_order(),
            CreationOrder::Tracked,
            "{path}"
        );
    }
    cleanup(&path);

    // The classic side: the option is create-only, so open_rw of an
    // existing file refuses it outright rather than silently doing nothing.
    let path = unique_tmp("classic");
    std::fs::write(
        &path,
        std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/btreek_legacy.h5"),
        )
        .unwrap(),
    )
    .unwrap();
    let before = std::fs::read(&path).unwrap();
    let err = H5File::options()
        .shared_messages(&[(all_three(), 0)], 50, 40)
        .open_rw(&path)
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("shared_messages"),
        "error should name the offending option: {err}"
    );
    let after = std::fs::read(&path).unwrap();
    assert_eq!(before, after, "a refused open must not touch the file");
    let sb = SuperblockV0V1::decode(&after).unwrap();
    assert_eq!(sb.version, 1);
    assert_eq!(sb.superblock_extension_address, UNDEF_ADDR);
    let root = (sb.base_address + sb.root_symbol_table_entry.obj_header_addr) as usize;
    assert!(
        ObjectHeader::decode(&after[root..]).is_err(),
        "a classic file's root header must not carry the OHDR signature"
    );
    let (root_header, _) = ObjectHeader::decode_v1(&after[root..]).unwrap();
    assert_eq!(
        root_header.attribute_creation_order(),
        CreationOrder::Untracked
    );
    cleanup(&path);
}
