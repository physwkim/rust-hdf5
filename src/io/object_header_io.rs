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

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::shared::{SharedMessage, MSG_FLAG_SHARED};
use crate::format::messages::{MSG_DATATYPE, MSG_OBJ_HEADER_CONTINUATION};
use crate::format::object_header::{ObjectHeader, ObjectHeaderMessage};
use crate::format::{FormatContext, UNDEF_ADDR};
use crate::io::file_handle::FileHandle;
use crate::io::IoResult;

/// Bound on the number of continuation blocks followed per header.
const MAX_CONT_BLOCKS: usize = 4096;

/// An object header's chunk-0 can hold more than 8 KiB of inline messages
/// (many/large attributes), but reading the whole file tail would allocate
/// gigabytes per object on a large valid file. Probe a bounded prefix; if the
/// header declares a larger chunk-0, `decode_any` reports the exact byte count
/// via `BufferTooShort` and we read precisely that much.
const HEADER_PROBE: usize = 8192;

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
pub(crate) fn read_object_header_full(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    addr: u64,
) -> IoResult<ObjectHeader> {
    let mut buf = handle.read_at_most(addr, HEADER_PROBE)?;
    if let Err(crate::format::FormatError::BufferTooShort { needed, .. }) =
        ObjectHeader::decode_any(&buf)
    {
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

        let cont_buf = handle.read_at_most(cont_addr, cont_len as usize)?;
        let mut new_msgs = Vec::new();
        parse_continuation_block(&cont_buf, is_v2, track_creation_order, &mut new_msgs);
        collect(&new_msgs, &mut pending);
        header.messages.extend(new_msgs);
    }

    Ok(header)
}

/// Parse the messages out of a single object-header continuation block.
///
/// For v2 (`is_v2`) the block is `"OCHK"(4) + messages + checksum(4)`;
/// for v1 it is bare messages. Null/padding messages (type 0) are skipped.
fn parse_continuation_block(
    cont_buf: &[u8],
    is_v2: bool,
    track_creation_order: bool,
    out: &mut Vec<ObjectHeaderMessage>,
) {
    if is_v2 {
        // "OCHK"(4) signature + messages + checksum(4).
        if cont_buf.len() < 8 || cont_buf[0..4] != *b"OCHK" {
            return;
        }
        let msgs_end = cont_buf.len() - 4; // strip trailing checksum
        let mut pos = 4; // skip "OCHK" signature
                         // v2 message header: type(1) + size(2) + flags(1) [+ crt_order(2)]
        let hdr_size = if track_creation_order { 6 } else { 4 };
        while pos + hdr_size <= msgs_end {
            let msg_type = cont_buf[pos];
            let data_size = u16::from_le_bytes([cont_buf[pos + 1], cont_buf[pos + 2]]) as usize;
            let msg_flags = cont_buf[pos + 3];
            pos += hdr_size;
            if pos + data_size > msgs_end {
                break;
            }
            if msg_type != 0 {
                out.push(ObjectHeaderMessage {
                    msg_type,
                    flags: msg_flags,
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
                    data: cont_buf[pos..pos + data_size].to_vec(),
                });
            }
            pos += data_size;
            pos = (pos + 7) & !7; // v1 8-byte alignment
        }
    }
}

/// Decode the datatype a message carries, following it into the object header
/// it shares when the message is a reference rather than a body.
///
/// Every read path that wants a type from an object header message goes
/// through here. Decoding `msg.data` directly is only correct for a message
/// that is not shared, and the shared form does not fail loudly enough to
/// notice: `H5O_shared_t`'s first byte is its version, which a datatype
/// decoder reads as a version and a class of its own.
pub(crate) fn read_datatype_message(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    msg: &ObjectHeaderMessage,
) -> IoResult<DatatypeMessage> {
    read_datatype_message_at(handle, ctx, msg, MAX_SHARE_HOPS)
}

/// A committed datatype's own message is a body, not another reference, so
/// one hop is all a well-formed file needs. The bound is here because a
/// crafted file can point a shared message at itself.
const MAX_SHARE_HOPS: usize = 8;

fn read_datatype_message_at(
    handle: &mut FileHandle,
    ctx: &FormatContext,
    msg: &ObjectHeaderMessage,
    hops: usize,
) -> IoResult<DatatypeMessage> {
    if msg.flags & MSG_FLAG_SHARED == 0 {
        let (dt, _) = DatatypeMessage::decode(&msg.data, ctx)?;
        return Ok(dt);
    }
    if hops == 0 {
        return Err(crate::io::IoError::InvalidState(
            "a shared datatype message points at another shared datatype message more than \
             8 times over; the references may form a cycle"
                .into(),
        ));
    }
    match SharedMessage::decode(&msg.data, ctx)? {
        SharedMessage::Committed { object_header } => {
            let header = read_object_header_full(handle, ctx, object_header)?;
            let shared = header
                .messages
                .iter()
                .find(|m| m.msg_type == MSG_DATATYPE)
                .ok_or_else(|| {
                    crate::io::IoError::InvalidState(format!(
                        "a shared datatype message names the object header at {object_header}, \
                         which holds no datatype message"
                    ))
                })?
                // The borrow ends before the recursive call reads the file
                // again, so the followed message is cloned out of it.
                .clone();
            read_datatype_message_at(handle, ctx, &shared, hops - 1)
        }
        SharedMessage::Sohm { .. } => Err(crate::io::IoError::Unsupported(
            "its datatype is stored in the file's shared object header message heap, \
             which this crate does not read"
                .into(),
        )),
    }
}
