//! Reading a *complete* object header off disk.
//!
//! An object header is a chain: chunk 0 plus every block a continuation
//! message points at. Anything that walks a file's objects — the reader
//! building its catalog, the writer rebuilding its registry on reopen — must
//! see the whole chain, because a message in a continuation block is
//! indistinguishable from one in chunk 0 as far as the file's meaning goes.
//! A walker that reads only chunk 0 does not see a shortened object; it sees
//! a different one, and on the write side it then rewrites the object as the
//! shortened version. So this lives in one place that both sides call rather
//! than as a habit each side has to remember.
//!
//! The same argument applies to *stored-shared* messages. A message whose
//! `H5O_MSG_FLAG_SHARED` bit is set holds a pointer, not a body, and its
//! first byte is a version a message decoder happily reads as its own. The
//! substitution therefore happens here too, once, before the header is handed
//! out: after [`read_object_header_full`] returns, every message body is
//! literal and every shared flag is clear, so no downstream decoder can be
//! given a pointer to parse.

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::fractal_heap::{
    collect_managed_blocks, read_heap_object, FractalHeapHeader, HeapId,
};
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::shared::MSG_FLAG_SHARED;
use crate::format::messages::{
    MSG_ATTRIBUTE, MSG_DATASPACE, MSG_DATATYPE, MSG_LINK, MSG_LINK_INFO,
    MSG_OBJ_HEADER_CONTINUATION, MSG_SYMBOL_TABLE,
};
use crate::format::object_header::{ObjectHeader, ObjectHeaderMessage};
use crate::format::sohm::{SharedLocation, SharedMessagePointer};
use crate::format::BlockReader;
use crate::format::{FormatError, UNDEF_ADDR};
use crate::io::file_handle::FileHandle;
use crate::io::{FileMeta, IoResult};

/// Bound on the number of continuation blocks followed per header.
const MAX_CONT_BLOCKS: usize = 4096;

/// An object header's chunk-0 can hold more than 8 KiB of inline messages
/// (many/large attributes), but reading the whole file tail would allocate
/// gigabytes per object on a large valid file. Probe a bounded prefix; if the
/// header declares a larger chunk-0, `decode_any` reports the exact byte count
/// via `BufferTooShort` and we read precisely that much.
const HEADER_PROBE: usize = 8192;

/// Bound on committed-message indirection: a shared message read from another
/// object header can itself be shared. libhdf5 relies on the writer never
/// building a cycle; a crafted file must terminate.
const MAX_SHARED_DEPTH: usize = 8;

/// Read the object header at `addr` and return it with the messages from
/// every object-header continuation block flattened in.
///
/// Handles both wire formats:
/// - v1 headers: continuation blocks are bare v1 messages (type:u16,
///   size:u16, flags:u8, reserved:3, data, padded to 8-byte alignment).
/// - v2 headers: continuation blocks are `"OCHK"(4) + messages +
///   checksum(4)` with v2 message headers (type:u8, size:u16, flags:u8,
///   and a 2-byte creation-order field when the header tracks creation
///   order).
///
/// Nested continuations are followed; the total block count is bounded.
///
/// Every message of the returned header holds its literal body: a message
/// stored shared (`H5O_MSG_FLAG_SHARED`) has been resolved through the SOHM
/// heap or its committed object header and its flag cleared, so no caller can
/// hand a shared pointer to a message decoder.
pub(crate) fn read_object_header_full(
    handle: &mut FileHandle,
    meta: &FileMeta,
    addr: u64,
) -> IoResult<ObjectHeader> {
    read_object_header_at(handle, meta, addr, 0)
}

/// The size of the chunk-0 block the object header at `addr` occupies — what
/// a rewrite of that header supersedes and frees.
pub(crate) fn object_header_block_size(handle: &mut FileHandle, addr: u64) -> IoResult<usize> {
    let mut buf = handle.read_at_most(addr, HEADER_PROBE)?;
    if let Err(FormatError::BufferTooShort { needed, .. }) = ObjectHeader::decode_any(&buf) {
        if needed > buf.len() {
            buf = handle.read_at_most(addr, needed)?;
        }
    }
    Ok(ObjectHeader::decode_any(&buf)?.1)
}

/// [`read_object_header_full`], with the committed-message indirection depth
/// already reached.
fn read_object_header_at(
    handle: &mut FileHandle,
    meta: &FileMeta,
    addr: u64,
    depth: usize,
) -> IoResult<ObjectHeader> {
    let mut header = read_header_chain(handle, meta, addr)?;
    resolve_shared_messages(handle, meta, &mut header, addr, depth)?;
    Ok(header)
}

/// Chunk 0 and every continuation block it leads to, flattened into one
/// message list and nothing else done to them.
fn read_header_chain(
    handle: &mut FileHandle,
    meta: &FileMeta,
    addr: u64,
) -> IoResult<ObjectHeader> {
    let ctx = &meta.ctx;
    let mut buf = handle.read_at_most(addr, HEADER_PROBE)?;
    if let Err(FormatError::BufferTooShort { needed, .. }) = ObjectHeader::decode_any(&buf) {
        if needed > buf.len() {
            buf = handle.read_at_most(addr, needed)?;
        }
    }
    let (mut header, _) = ObjectHeader::decode_any(&buf)?;

    // A v1 header has no "OHDR" signature; detect by it.
    let is_v2 = buf.len() >= 4 && buf[0..4] == crate::format::object_header::OHDR_SIGNATURE;
    // v2 creation-order tracking is recorded in object-header flag bit 2.
    let track_creation_order = is_v2 && (header.flags & 0x04) != 0;

    let sa = ctx.sizeof_addr as usize;
    let ss = ctx.sizeof_size as usize;

    // Collect continuation references from a slice of messages.
    let collect = |msgs: &[ObjectHeaderMessage], out: &mut Vec<(u64, u64)>| {
        for msg in msgs {
            if msg.msg_type == MSG_OBJ_HEADER_CONTINUATION && msg.data.len() >= sa + ss {
                let cont_addr = read_uint(&msg.data, sa);
                let cont_len = read_uint(&msg.data[sa..], ss);
                out.push((cont_addr, cont_len));
            }
        }
    };

    let mut pending: Vec<(u64, u64)> = Vec::new();
    collect(&header.messages, &mut pending);

    let mut visited = std::collections::HashSet::new();
    let mut blocks_read = 0usize;

    while let Some((cont_addr, cont_len)) = pending.pop() {
        if cont_addr == UNDEF_ADDR || cont_addr == 0 || cont_len == 0 {
            continue;
        }
        if !visited.insert(cont_addr) {
            continue; // already followed — guard against cycles
        }
        blocks_read += 1;
        if blocks_read > MAX_CONT_BLOCKS {
            break;
        }

        // The continuation message states the chunk's exact length, and a v2
        // chunk's checksum covers exactly that image: a short read would check
        // a different buffer than the writer hashed.
        let cont_buf = handle.read_at(cont_addr, cont_len as usize)?;
        let mut new_msgs = Vec::new();
        parse_continuation_block(&cont_buf, is_v2, track_creation_order, &mut new_msgs)?;
        collect(&new_msgs, &mut pending);
        header.messages.extend(new_msgs);
    }

    Ok(header)
}

/// Why the object at `addr` cannot keep its bytes across a rewrite of the
/// file's shared-message table, or `None` when it can.
///
/// A rewritten table reassigns every heap ID in the file, so an object left
/// untouched must be one that names no heap object. Two things disqualify it:
///
/// * It resolves something through the heap. Both forms libhdf5 stores count —
///   a whole message replaced by a pointer (`H5O_MSG_FLAG_SHARED`), and the
///   datatype or dataspace of an attribute, whose pointer sits inside the
///   attribute body and is flagged there instead
///   (`H5O_ATTR_FLAG_TYPE_SHARED` / `H5O_ATTR_FLAG_SPACE_SHARED`). A pointer
///   to a *committed* object header is not one of these: it names an object,
///   not a heap object, and survives the heap being rebuilt.
/// * It names other objects. Everything below a preserved group keeps its
///   bytes with it, and this reopen never walked that subtree — the object is
///   preserved precisely because the walk could not model it — so nothing here
///   can say whether anything down there shares.
///
/// The reopen asks this of every object it will keep by its bytes, which is
/// the one thing a rebuilt shared-message table can break.
pub(crate) fn blocks_shared_message_rebuild(
    handle: &mut FileHandle,
    meta: &FileMeta,
    addr: u64,
) -> IoResult<Option<&'static str>> {
    let header = read_header_chain(handle, meta, addr)?;
    let in_heap = |bytes: &[u8]| {
        SharedMessagePointer::decode(bytes, &meta.ctx)
            .is_ok_and(|p| p.location == SharedLocation::Sohm)
    };
    for msg in &header.messages {
        if msg.flags & MSG_FLAG_SHARED != 0 && in_heap(&msg.data) {
            return Ok(Some("holds a shared object header message"));
        }
        if msg.msg_type == MSG_ATTRIBUTE
            && shared_attribute_fields(&msg.data)
                .into_iter()
                .flatten()
                .any(in_heap)
        {
            return Ok(Some(
                "holds an attribute whose datatype or dataspace is a shared object header \
                 message",
            ));
        }
        if matches!(msg.msg_type, MSG_LINK | MSG_LINK_INFO | MSG_SYMBOL_TABLE) {
            return Ok(Some(
                "names objects of its own, which would keep their bytes with it",
            ));
        }
    }
    Ok(None)
}

/// The object header address of the committed (named) datatype the object at
/// `addr` declares its datatype from, or `None` when it holds its own.
///
/// [`read_object_header_full`] substitutes the named type's literal datatype
/// message for the pointer, which is what every reader wants and what makes a
/// rewrite of that header silently detach the dataset from the name. A rewrite
/// therefore asks this of the *raw* chain and re-emits the pointer, so
/// `H5Tget_class`-level equality and `H5Tcommitted` both survive it.
pub(crate) fn committed_datatype_address(
    handle: &mut FileHandle,
    meta: &FileMeta,
    addr: u64,
) -> IoResult<Option<u64>> {
    let header = read_header_chain(handle, meta, addr)?;
    for msg in &header.messages {
        if msg.msg_type != MSG_DATATYPE || msg.flags & MSG_FLAG_SHARED == 0 {
            continue;
        }
        let ptr = SharedMessagePointer::decode(&msg.data, &meta.ctx)?;
        if ptr.location == SharedLocation::Committed {
            return Ok(Some(ptr.oh_addr));
        }
    }
    Ok(None)
}

/// The pointer bytes of whichever of an attribute's datatype and dataspace
/// its flags byte says are stored shared — datatype first.
///
/// Both slots are empty when the message is not an attribute this crate reads
/// that way at all, which is the same answer as an attribute that shares
/// neither field: a caller only ever asks what there is to follow. The version
/// and length checks are the ones [`resolve_shared_attribute_fields`] makes
/// before it splices.
fn shared_attribute_fields(body: &[u8]) -> [Option<&[u8]>; 2] {
    /// `H5O_ATTR_FLAG_TYPE_SHARED`.
    const ATTR_FLAG_TYPE_SHARED: u8 = 0x01;
    /// `H5O_ATTR_FLAG_SPACE_SHARED`.
    const ATTR_FLAG_SPACE_SHARED: u8 = 0x02;

    let fields = || {
        if body.len() < 8 || body[0] < 2 {
            return None;
        }
        let flags = body[1];
        if flags & (ATTR_FLAG_TYPE_SHARED | ATTR_FLAG_SPACE_SHARED) == 0 {
            return None;
        }
        let name_size = u16::from_le_bytes([body[2], body[3]]) as usize;
        let dt_size = u16::from_le_bytes([body[4], body[5]]) as usize;
        let ds_size = u16::from_le_bytes([body[6], body[7]]) as usize;
        let hdr_len: usize = if body[0] >= 3 { 9 } else { 8 };
        let name_end = hdr_len.checked_add(name_size)?;
        let dt_end = name_end.checked_add(dt_size)?;
        let ds_end = dt_end.checked_add(ds_size)?;
        if body.len() < ds_end {
            return None;
        }
        Some([
            (flags & ATTR_FLAG_TYPE_SHARED != 0).then(|| &body[name_end..dt_end]),
            (flags & ATTR_FLAG_SPACE_SHARED != 0).then(|| &body[dt_end..ds_end]),
        ])
    };
    fields().unwrap_or([None, None])
}

/// Replace every stored-shared message body in `header` with the literal
/// message it points at, and clear the shared flag.
///
/// This is `H5O__shared_decode`'s job in libhdf5, where it runs inside each
/// message class's decode. The port decodes message bodies lazily from raw
/// bytes at many call sites, so the substitution happens once here instead:
/// after this returns, `msg.data` is the message.
fn resolve_shared_messages(
    handle: &mut FileHandle,
    meta: &FileMeta,
    header: &mut ObjectHeader,
    self_addr: u64,
    depth: usize,
) -> IoResult<()> {
    // Whole-message sharing: the body is a shared pointer.
    let mut resolved: Vec<(usize, Vec<u8>)> = Vec::new();
    for (i, msg) in header.messages.iter().enumerate() {
        if msg.flags & MSG_FLAG_SHARED == 0 {
            continue;
        }
        let body = resolve_shared_body(
            handle,
            meta,
            msg.msg_type,
            &msg.data,
            &header.messages,
            self_addr,
            depth,
        )?;
        resolved.push((i, body));
    }
    for (i, body) in resolved {
        header.messages[i].data = body;
        header.messages[i].flags &= !MSG_FLAG_SHARED;
    }

    // Field-level sharing: an attribute whose embedded datatype or dataspace
    // is shared carries the pointer inside its own body, flagged in the
    // attribute's flags byte rather than the message's (`H5O__attr_decode`).
    let mut rewritten: Vec<(usize, Vec<u8>)> = Vec::new();
    for (i, msg) in header.messages.iter().enumerate() {
        if msg.msg_type != MSG_ATTRIBUTE {
            continue;
        }
        if let Some(body) = resolve_shared_attribute_fields(
            handle,
            meta,
            &msg.data,
            &header.messages,
            self_addr,
            depth,
        )? {
            rewritten.push((i, body));
        }
    }
    for (i, body) in rewritten {
        header.messages[i].data = body;
    }

    Ok(())
}

/// Resolve one shared pointer of message type `msg_type` into the literal
/// message body it names (`H5O__shared_read`).
///
/// `siblings` are the messages of the header being read, needed for a
/// committed pointer that names that same header — an attribute whose
/// datatype is the committed datatype it hangs off.
fn resolve_shared_body(
    handle: &mut FileHandle,
    meta: &FileMeta,
    msg_type: u8,
    ptr_bytes: &[u8],
    siblings: &[ObjectHeaderMessage],
    self_addr: u64,
    depth: usize,
) -> IoResult<Vec<u8>> {
    let invalid = |msg: String| crate::io::IoError::Format(FormatError::InvalidData(msg));

    if depth >= MAX_SHARED_DEPTH {
        return Err(invalid(format!(
            "shared message indirection deeper than {MAX_SHARED_DEPTH} levels"
        )));
    }
    let ptr = SharedMessagePointer::decode(ptr_bytes, &meta.ctx)?;
    match ptr.location {
        SharedLocation::Sohm => {
            let table = meta.sohm.as_ref().ok_or_else(|| {
                invalid(format!(
                    "message type {msg_type:#04x} is shared in the heap but the file has no \
                     shared message table"
                ))
            })?;
            let heap_addr = table.heap_addr(msg_type).ok_or_else(|| {
                invalid(format!(
                    "no shared-message index covers message type {msg_type:#04x}"
                ))
            })?;
            // The heap header's length depends only on the address/length
            // widths and the filter pipeline it may carry; a bounded prefix
            // covers it, as it does for dense link storage.
            let hdr_buf = handle.read_at_most(heap_addr, 512)?;
            let fh_header = FractalHeapHeader::decode(&hdr_buf, &meta.ctx)?;
            let mut br = HandleBlockReader { handle };
            // The same heap-ID reader dense links and dense attributes use: a
            // shared message can be managed, tiny or huge exactly as any other
            // heap object can, and the ID says which.
            let id = HeapId::parse(&ptr.heap_id, &fh_header, &meta.ctx)?;
            let blocks = collect_managed_blocks(&fh_header, &meta.ctx, &mut br)?;
            Ok(read_heap_object(
                &id, &fh_header, &meta.ctx, &blocks, &mut br,
            )?)
        }
        SharedLocation::Committed => {
            let pick = |msgs: &[ObjectHeaderMessage]| {
                msgs.iter()
                    .find(|m| m.msg_type == msg_type && m.flags & MSG_FLAG_SHARED == 0)
                    .map(|m| m.data.clone())
            };
            let found = if ptr.oh_addr == self_addr {
                pick(siblings)
            } else {
                let target = read_object_header_at(handle, meta, ptr.oh_addr, depth + 1)?;
                pick(&target.messages)
            };
            found.ok_or_else(|| {
                invalid(format!(
                    "object header at {:#x} holds no message of type {msg_type:#04x} for a \
                     committed shared message",
                    ptr.oh_addr
                ))
            })
        }
        // `H5O__shared_read` asserts the message is stored shared; a file that
        // flags a message shared and then says it is not is corrupt.
        SharedLocation::Unshared | SharedLocation::Here => Err(invalid(format!(
            "message type {msg_type:#04x} is flagged shared but its pointer says it is not \
             stored shared"
        ))),
    }
}

/// Splice the literal datatype/dataspace into an attribute message whose
/// flags byte says either is shared, returning the rewritten body.
///
/// Returns `None` when nothing is shared, so the common attribute costs only
/// the flags check. The attribute flags byte is defined from version 2 on
/// (`H5O__attr_decode` skips it as unused before that), and the 8-byte field
/// alignment applies only to version 1, so a rewritten attribute never needs
/// padding recomputed.
fn resolve_shared_attribute_fields(
    handle: &mut FileHandle,
    meta: &FileMeta,
    body: &[u8],
    siblings: &[ObjectHeaderMessage],
    self_addr: u64,
    depth: usize,
) -> IoResult<Option<Vec<u8>>> {
    /// `H5O_ATTR_FLAG_TYPE_SHARED`.
    const ATTR_FLAG_TYPE_SHARED: u8 = 0x01;
    /// `H5O_ATTR_FLAG_SPACE_SHARED`.
    const ATTR_FLAG_SPACE_SHARED: u8 = 0x02;

    if body.len() < 8 || body[0] < 2 {
        return Ok(None);
    }
    let flags = body[1];
    if flags & (ATTR_FLAG_TYPE_SHARED | ATTR_FLAG_SPACE_SHARED) == 0 {
        return Ok(None);
    }
    let name_size = u16::from_le_bytes([body[2], body[3]]) as usize;
    let dt_size = u16::from_le_bytes([body[4], body[5]]) as usize;
    let ds_size = u16::from_le_bytes([body[6], body[7]]) as usize;
    // Version 3 adds the name character-set byte.
    let hdr_len = if body[0] >= 3 { 9 } else { 8 };
    let name_end = hdr_len + name_size;
    let dt_end = name_end + dt_size;
    let ds_end = dt_end + ds_size;
    if body.len() < ds_end {
        return Err(crate::io::IoError::Format(FormatError::BufferTooShort {
            needed: ds_end,
            available: body.len(),
        }));
    }

    let resolve = |handle: &mut FileHandle, msg_type: u8, range: std::ops::Range<usize>| {
        resolve_shared_body(
            handle,
            meta,
            msg_type,
            &body[range],
            siblings,
            self_addr,
            depth,
        )
    };
    let datatype = if flags & ATTR_FLAG_TYPE_SHARED != 0 {
        resolve(handle, MSG_DATATYPE, name_end..dt_end)?
    } else {
        body[name_end..dt_end].to_vec()
    };
    let dataspace = if flags & ATTR_FLAG_SPACE_SHARED != 0 {
        resolve(handle, MSG_DATASPACE, dt_end..ds_end)?
    } else {
        body[dt_end..ds_end].to_vec()
    };

    let (Ok(dt_len), Ok(ds_len)) = (
        u16::try_from(datatype.len()),
        u16::try_from(dataspace.len()),
    ) else {
        return Err(crate::io::IoError::Format(FormatError::InvalidData(
            "shared attribute datatype/dataspace does not fit an attribute message".into(),
        )));
    };

    let mut out = Vec::with_capacity(hdr_len + name_size + datatype.len() + dataspace.len());
    out.extend_from_slice(&body[..hdr_len]);
    // The spliced-in bodies are literal, so the attribute is no longer sharing
    // either field.
    out[1] = 0;
    out[4..6].copy_from_slice(&dt_len.to_le_bytes());
    out[6..8].copy_from_slice(&ds_len.to_le_bytes());
    out.extend_from_slice(&body[hdr_len..name_end]);
    out.extend_from_slice(&datatype);
    out.extend_from_slice(&dataspace);
    out.extend_from_slice(&body[ds_end..]);
    Ok(Some(out))
}

/// Parse the messages out of a single object-header continuation block.
///
/// For v2 (`is_v2`) the block is `"OCHK"(4) + messages + checksum(4)`;
/// for v1 it is bare messages. Null/padding messages (type 0) are skipped.
///
/// A v2 block is checked whole before any message is taken from it, the way
/// `H5O__cache_chk_verify_chksum` and `H5O__chunk_deserialize` check it:
/// signature first, then the Jenkins checksum over the entire chunk image.
/// Version 1 has neither.
fn parse_continuation_block(
    cont_buf: &[u8],
    is_v2: bool,
    track_creation_order: bool,
    out: &mut Vec<ObjectHeaderMessage>,
) -> crate::format::FormatResult<()> {
    if is_v2 {
        // "OCHK"(4) signature + messages + checksum(4).
        if cont_buf.len() < 8 {
            return Err(FormatError::BufferTooShort {
                needed: 8,
                available: cont_buf.len(),
            });
        }
        if cont_buf[0..4] != *b"OCHK" {
            return Err(FormatError::InvalidSignature);
        }
        let msgs_end = cont_buf.len() - 4; // strip trailing checksum
        let stored = u32::from_le_bytes([
            cont_buf[msgs_end],
            cont_buf[msgs_end + 1],
            cont_buf[msgs_end + 2],
            cont_buf[msgs_end + 3],
        ]);
        let computed = crate::format::checksum::checksum_metadata(&cont_buf[..msgs_end]);
        if stored != computed {
            return Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }
        let mut pos = 4; // skip "OCHK" signature
                         // v2 message header: type(1) + size(2) + flags(1) [+ crt_order(2)]
        let hdr_size = if track_creation_order { 6 } else { 4 };
        while pos + hdr_size <= msgs_end {
            let msg_type = cont_buf[pos];
            let data_size = u16::from_le_bytes([cont_buf[pos + 1], cont_buf[pos + 2]]) as usize;
            let msg_flags = cont_buf[pos + 3];
            let creation_index = if track_creation_order {
                u16::from_le_bytes([cont_buf[pos + 4], cont_buf[pos + 5]])
            } else {
                0
            };
            pos += hdr_size;
            if pos + data_size > msgs_end {
                break;
            }
            if msg_type != 0 {
                out.push(ObjectHeaderMessage {
                    msg_type,
                    flags: msg_flags,
                    creation_index,
                    data: cont_buf[pos..pos + data_size].to_vec(),
                });
            }
            pos += data_size;
        }
    } else {
        // v1 continuation: bare messages, 8-byte aligned, no prefix.
        let mut pos = 0;
        while pos + 8 <= cont_buf.len() {
            let msg_type = u16::from_le_bytes([cont_buf[pos], cont_buf[pos + 1]]);
            let data_size = u16::from_le_bytes([cont_buf[pos + 2], cont_buf[pos + 3]]) as usize;
            let msg_flags = cont_buf[pos + 4];
            pos += 8; // type(2) + size(2) + flags(1) + reserved(3)
            if pos + data_size > cont_buf.len() {
                break;
            }
            if msg_type != 0 {
                out.push(ObjectHeaderMessage {
                    msg_type: msg_type as u8,
                    flags: msg_flags,
                    // A version-1 message envelope has no creation index.
                    creation_index: 0,
                    data: cont_buf[pos..pos + data_size].to_vec(),
                });
            }
            pos += data_size;
            pos = (pos + 7) & !7; // v1 8-byte alignment
        }
    }
    Ok(())
}

/// Decode the datatype a message carries.
///
/// Every read path that wants a type from an object-header message goes
/// through here. The message it is handed came out of
/// [`read_object_header_full`], so a stored-shared datatype has already been
/// substituted for its literal body; a message still carrying the shared flag
/// at this point never passed through the resolver, and decoding its bytes as
/// a datatype would read `H5O_shared_t`'s version byte as a version and class
/// of its own rather than fail.
pub(crate) fn read_datatype_message(
    handle: &mut FileHandle,
    meta: &FileMeta,
    msg: &ObjectHeaderMessage,
) -> IoResult<DatatypeMessage> {
    if msg.flags & MSG_FLAG_SHARED != 0 {
        // Not reachable through `read_object_header_full`; a caller that
        // assembled the message some other way gets told, not mis-decoded.
        let body = resolve_shared_body(handle, meta, msg.msg_type, &msg.data, &[], UNDEF_ADDR, 0)?;
        let (dt, _) = DatatypeMessage::decode(&body, &meta.ctx)?;
        return Ok(dt);
    }
    let (dt, _) = DatatypeMessage::decode(&msg.data, &meta.ctx)?;
    Ok(dt)
}

/// A [`BlockReader`] over the file handle, for the fractal heaps a shared
/// message index keeps its bodies in.
struct HandleBlockReader<'a> {
    handle: &'a mut FileHandle,
}

impl BlockReader for HandleBlockReader<'_> {
    fn read_block(&mut self, offset: u64, len: usize) -> crate::format::FormatResult<Vec<u8>> {
        self.handle.read_at(offset, len).map_err(|e| {
            FormatError::InvalidData(format!(
                "fractal heap block read failed at {offset:#x}: {e}"
            ))
        })
    }
}
