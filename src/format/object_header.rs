/// Object Header v2 encode/decode.
///
/// The Object Header is the primary metadata container in HDF5. Every named
/// object (group, dataset, committed datatype) has one. Version 2 headers use
/// the "OHDR" signature and end with a Jenkins checksum.
///
/// Layout of the header prefix (before messages):
/// ```text
/// "OHDR" (4 bytes)
/// Version: 2 (1 byte)
/// Flags (1 byte):
///   bits 0-1: chunk#0 data-size encoding (0=1B, 1=2B, 2=4B, 3=8B)
///   bit 2:    attribute creation order tracked
///   bit 3:    attribute creation order indexed
///   bit 4:    non-default attribute storage phase-change thresholds
///   bit 5:    store access/modify/change/birth timestamps
/// [if bit 5 set: 4x uint32 timestamps (16 bytes)]
/// [if bit 4 set: max_compact(u16) + min_dense(u16) (4 bytes)]
/// chunk0_data_size: 1/2/4/8 bytes depending on bits 0-1
/// <messages>
/// Checksum (4 bytes)
/// ```
///
/// Each message (v2 format):
/// ```text
/// msg_type:       u8
/// msg_data_size:  u16 LE
/// msg_flags:      u8
/// [if obj header flags bit 2: creation_order: u16 LE]
/// msg_data:       [u8; msg_data_size]
/// ```
use crate::format::checksum::checksum_metadata;
use crate::format::creation_order::CreationOrder;
use crate::format::{FormatContext, FormatError, FormatResult};

/// The 4-byte object header v2 signature.
pub const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// The 4-byte signature of a version-2 object header continuation chunk.
pub const OCHK_SIGNATURE: [u8; 4] = *b"OCHK";

/// `H5O_NULL_ID` — the message that covers space a chunk holds but does not
/// use.
const MSG_NIL: u8 = 0x00;

/// `H5O_CONT_ID` — the message naming a continuation chunk.
const MSG_CONTINUATION: u8 = 0x10;

/// Object header version 2.
pub const OHDR_VERSION: u8 = 2;

/// Largest payload one object header message can carry.
///
/// The message envelope encodes the payload length in a `u16`, so this is a
/// hard on-disk ceiling, not a policy: libhdf5 refuses the same sizes through
/// `H5O_MESG_MAX_SIZE` (65536) and moves anything that reaches it out of the
/// header — `H5O__attr_create` switches such an attribute to dense storage.
pub const MAX_MESSAGE_SIZE: usize = u16::MAX as usize;

// Flag bit masks
const FLAG_SIZE_MASK: u8 = 0x03;
const FLAG_ATTR_CREATION_ORDER_TRACKED: u8 = 0x04;
const FLAG_ATTR_CREATION_ORDER_INDEXED: u8 = 0x08;
const FLAG_NON_DEFAULT_ATTR_THRESHOLDS: u8 = 0x10;
const FLAG_STORE_TIMESTAMPS: u8 = 0x20;

/// A single message within an object header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeaderMessage {
    /// Message type ID (e.g., 0x01 = Dataspace, 0x03 = Datatype, etc.)
    pub msg_type: u8,
    /// Per-message flags (bit 0 = constant, bit 1 = shared, etc.)
    pub flags: u8,
    /// Creation index, written only when the header tracks attribute creation
    /// order (flags bit 2). Only the attribute message class has one in
    /// libhdf5 — `H5O_msg_class_t::get_crt_index` is null for every other
    /// type, leaving the field zero (`H5O_msg_append_real`).
    pub creation_index: u16,
    /// Raw message payload.
    pub data: Vec<u8>,
}

/// The four times a version-2 object header stores, in the order
/// `H5O__cache_serialize` writes them: seconds since the epoch, as `H5_now`
/// produces them.
///
/// Their presence *is* the `H5O_HDR_STORE_TIMES` flag — see
/// [`ObjectHeader::times`] — so an object created with
/// `H5Pset_obj_track_times(true)` cannot be encoded with the flag set and no
/// times behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectTimes {
    /// Access time (`oh->atime`).
    pub access: u32,
    /// Modification time (`oh->mtime`).
    pub modification: u32,
    /// Change time (`oh->ctime`).
    pub change: u32,
    /// Birth time (`oh->btime`).
    pub birth: u32,
}

impl ObjectTimes {
    /// All four set to `now` — what `H5O_create_ohdr` does for an object
    /// created with timestamps enabled.
    pub fn created_at(now: u32) -> Self {
        Self {
            access: now,
            modification: now,
            change: now,
            birth: now,
        }
    }

    /// These times after a real modification of the object.
    ///
    /// `H5O_touch_oh` moves access and change time to `now` for a version-2
    /// header and leaves modification and birth time as they were — the
    /// modification time is what its own `XXX` comment says is not updated
    /// yet. Following it means a rewrite reports the same times libhdf5 would.
    pub fn touched(self, now: u32) -> Self {
        Self {
            access: now,
            change: now,
            ..self
        }
    }
}

/// How an object header's messages divide between chunk 0 and one
/// continuation chunk, produced by [`ObjectHeader::plan_chunks`] and consumed
/// by [`ObjectHeader::encode_chunked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    /// Messages before this index go in chunk 0, the rest in the continuation
    /// chunk.
    split: usize,
    /// Bytes chunk 0 occupies, prefix and checksum included.
    pub chunk0_size: usize,
    /// Bytes the continuation chunk occupies, signature and checksum
    /// included; zero when every message fits chunk 0.
    pub continuation_size: usize,
}

/// Object Header v2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeader {
    /// Header flags byte. Bits 0-1 control chunk0 size encoding. Other bits
    /// control optional fields (attr thresholds, creation order).
    ///
    /// Bit 5 (`H5O_HDR_STORE_TIMES`) is *not* held here: it is derived from
    /// [`times`](Self::times) at encode and stripped at decode, so the flag and
    /// the four values it announces cannot disagree. Setting it by hand here
    /// does nothing — the encoder's flag byte comes from `times`.
    pub flags: u8,
    /// The stored times, when this object tracks them.
    pub times: Option<ObjectTimes>,
    /// The ordered list of header messages.
    pub messages: Vec<ObjectHeaderMessage>,
}

impl ObjectHeader {
    /// Create a new, empty object header with default flags.
    ///
    /// Defaults: bits 0-1 = 2 (4-byte chunk size encoding), no timestamps,
    /// no attribute creation order, no non-default thresholds.
    pub fn new() -> Self {
        Self {
            flags: 0x02, // bits 0-1 = 2 => 4-byte chunk0 size
            times: None,
            messages: Vec::new(),
        }
    }

    /// Append a message to the object header.
    pub fn add_message(&mut self, msg_type: u8, flags: u8, data: Vec<u8>) {
        self.add_message_indexed(msg_type, flags, data, 0);
    }

    /// Append a message carrying a creation index.
    ///
    /// The index reaches the file only when the header's flags bit 2 says the
    /// creation order is tracked; libhdf5 does not encode the field otherwise
    /// (`H5O_SIZEOF_MSGHDR_OH`).
    pub fn add_message_indexed(
        &mut self,
        msg_type: u8,
        flags: u8,
        data: Vec<u8>,
        creation_index: u16,
    ) {
        self.messages.push(ObjectHeaderMessage {
            msg_type,
            flags,
            creation_index,
            data,
        });
    }

    /// Declare `order` as this object's attribute creation-order policy.
    ///
    /// `H5Pget_attr_creation_order` reads these two bits back out of the
    /// header, not out of the Attribute Info message (`H5Pocpl.c`), so they
    /// are what makes an object report its attributes as creation-ordered.
    /// Setting `TRACKED` also widens every message envelope by the two-byte
    /// creation index.
    pub fn set_attribute_creation_order(&mut self, order: CreationOrder) {
        self.flags &= !(FLAG_ATTR_CREATION_ORDER_TRACKED | FLAG_ATTR_CREATION_ORDER_INDEXED);
        if order.is_tracked() {
            self.flags |= FLAG_ATTR_CREATION_ORDER_TRACKED;
        }
        if order.is_indexed() {
            self.flags |= FLAG_ATTR_CREATION_ORDER_INDEXED;
        }
    }

    /// This object's attribute creation-order policy, as its flag bits
    /// declare it — the reverse of
    /// [`set_attribute_creation_order`](Self::set_attribute_creation_order),
    /// and what a reopen must consult so a rewrite re-declares what the file
    /// already says.
    pub fn attribute_creation_order(&self) -> CreationOrder {
        CreationOrder::from_flags(
            self.flags & FLAG_ATTR_CREATION_ORDER_TRACKED != 0,
            self.flags & FLAG_ATTR_CREATION_ORDER_INDEXED != 0,
        )
    }

    /// The flags byte as it reaches the file: everything [`flags`](Self::flags)
    /// holds, with `H5O_HDR_STORE_TIMES` taken from
    /// [`times`](Self::times) — the one place the two are joined, so a header
    /// can neither claim times it does not have nor carry times it does not
    /// declare.
    fn encoded_flags(&self) -> u8 {
        let base = self.flags & !FLAG_STORE_TIMESTAMPS;
        match self.times {
            Some(_) => base | FLAG_STORE_TIMESTAMPS,
            None => base,
        }
    }

    /// Returns the number of bytes used to encode chunk0's data size, based on
    /// flags bits 0-1.
    fn chunk0_size_bytes(&self) -> usize {
        match self.flags & FLAG_SIZE_MASK {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        }
    }

    /// Whether attribute creation order tracking is enabled (flags bit 2).
    pub fn has_creation_order(&self) -> bool {
        self.flags & FLAG_ATTR_CREATION_ORDER_TRACKED != 0
    }

    /// Bytes a message envelope takes: type, size and flags, plus the creation
    /// index when this header tracks one (`H5O_SIZEOF_MSGHDR_OH`).
    pub fn message_envelope_size(&self) -> usize {
        if self.has_creation_order() {
            1 + 2 + 1 + 2 // type + size + flags + creation_order
        } else {
            1 + 2 + 1 // type + size + flags
        }
    }

    /// Bytes `messages` occupy in a chunk, envelopes included.
    fn messages_size(&self, messages: &[ObjectHeaderMessage]) -> usize {
        messages
            .iter()
            .map(|m| self.message_envelope_size() + m.data.len())
            .sum()
    }

    /// Compute the byte size of the messages region (chunk0 data).
    fn messages_data_size(&self) -> usize {
        self.messages_size(&self.messages)
    }

    /// Bytes chunk 0 spends before its message area: signature, version,
    /// flags, the optional prefix fields, and the chunk-0 size field.
    fn prefix_size(&self) -> usize {
        let mut size = 4 + 1 + 1; // OHDR + version + flags
        if self.times.is_some() {
            size += 16; // 4 x u32
        }
        if self.flags & FLAG_NON_DEFAULT_ATTR_THRESHOLDS != 0 {
            size += 4; // max_compact(u16) + min_dense(u16)
        }
        size + self.chunk0_size_bytes()
    }

    /// Reject a message whose payload the `u16` size field cannot express.
    ///
    /// Writing such a message would record its length modulo 65536: the reader
    /// would then take the payload's own tail for the next message envelope and
    /// every message after it would decode as garbage. Nothing downstream can
    /// detect that — the checksum is computed over the truncated image and
    /// matches — so the check has to happen before any bytes are produced.
    fn check_message_sizes(messages: &[ObjectHeaderMessage]) -> FormatResult<()> {
        for msg in messages {
            if msg.data.len() > MAX_MESSAGE_SIZE {
                return Err(FormatError::InvalidData(format!(
                    "object header message type 0x{:02X} is {} bytes, over the \
                     {MAX_MESSAGE_SIZE}-byte limit the message size field can express",
                    msg.msg_type,
                    msg.data.len()
                )));
            }
        }
        Ok(())
    }

    /// Append `messages` in wire form, then pad to `data_size` bytes.
    ///
    /// Space left over is filled the way `H5O__chunk_serialize` leaves a
    /// partly used chunk: a NIL message covering the rest when its envelope
    /// fits, and otherwise a "gap" of zero bytes too short to hold any message
    /// header at all (`H5O_SIZEOF_MSGHDR_OH`, H5Ocache.c).
    fn write_messages(
        &self,
        buf: &mut Vec<u8>,
        messages: &[ObjectHeaderMessage],
        data_size: usize,
    ) {
        let envelope = self.message_envelope_size();
        let mut write = |msg_type: u8, flags: u8, creation_index: u16, data: &[u8]| {
            buf.push(msg_type);
            // Checked against MAX_MESSAGE_SIZE by `check_message_sizes`.
            buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
            buf.push(flags);
            if self.has_creation_order() {
                buf.extend_from_slice(&creation_index.to_le_bytes());
            }
            buf.extend_from_slice(data);
        };
        for msg in messages {
            write(msg.msg_type, msg.flags, msg.creation_index, &msg.data);
        }
        let spare = data_size - self.messages_size(messages);
        if spare >= envelope {
            write(MSG_NIL, 0x00, 0, &vec![0u8; spare - envelope]);
        } else {
            buf.extend(std::iter::repeat_n(0u8, spare));
        }
    }

    /// Encode the object header to a byte vector, including "OHDR" signature
    /// and trailing checksum, with every message in chunk 0.
    ///
    /// Fails when any message payload exceeds [`MAX_MESSAGE_SIZE`] — see
    /// [`check_message_sizes`](Self::check_message_sizes).
    pub fn encode(&self) -> FormatResult<Vec<u8>> {
        self.encode_chunk0(&self.messages, self.messages_data_size())
    }

    /// Chunk 0 holding `messages`, with a message area of exactly `data_size`
    /// bytes — the sole producer of a version-2 chunk-0 image.
    fn encode_chunk0(
        &self,
        messages: &[ObjectHeaderMessage],
        data_size: usize,
    ) -> FormatResult<Vec<u8>> {
        Self::check_message_sizes(messages)?;
        debug_assert!(data_size >= self.messages_size(messages));
        let total = self.prefix_size() + data_size + 4; // + checksum
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&OHDR_SIGNATURE);
        buf.push(OHDR_VERSION);
        buf.push(self.encoded_flags());

        // Optional timestamps (bit 5), in `H5O__cache_serialize` order.
        if let Some(t) = self.times {
            for field in [t.access, t.modification, t.change, t.birth] {
                buf.extend_from_slice(&field.to_le_bytes());
            }
        }

        // Optional attr storage thresholds (bit 4) -- write defaults if enabled
        if self.flags & FLAG_NON_DEFAULT_ATTR_THRESHOLDS != 0 {
            // max_compact = 8, min_dense = 6 (HDF5 defaults)
            buf.extend_from_slice(&8u16.to_le_bytes());
            buf.extend_from_slice(&6u16.to_le_bytes());
        }

        let csb = self.chunk0_size_bytes();
        buf.extend_from_slice(&(data_size as u64).to_le_bytes()[..csb]);

        self.write_messages(&mut buf, messages, data_size);

        // Checksum over everything before the checksum
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), total);
        Ok(buf)
    }

    /// A continuation chunk holding `messages`: the `"OCHK"` signature, the
    /// messages, and the Jenkins checksum over both, exactly as
    /// `H5O__chunk_serialize` writes one. Sized to fit, so it needs no
    /// padding.
    fn encode_continuation(&self, messages: &[ObjectHeaderMessage]) -> FormatResult<Vec<u8>> {
        Self::check_message_sizes(messages)?;
        let data_size = self.messages_size(messages);
        let mut buf = Vec::with_capacity(OCHK_SIGNATURE.len() + data_size + 4);
        buf.extend_from_slice(&OCHK_SIGNATURE);
        self.write_messages(&mut buf, messages, data_size);
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        Ok(buf)
    }

    /// How this header divides between chunk 0 and its continuation chunk
    /// when chunk 0's message area holds at most `capacity` bytes.
    ///
    /// The whole point of a plan is that both sizes are known before either
    /// chunk has an address: the caller allocates the continuation from
    /// [`continuation_size`](ChunkPlan::continuation_size) and hands the
    /// address back to [`encode_chunked`](Self::encode_chunked), which is the
    /// only way chunk 0 can name a block that does not exist yet.
    ///
    /// Messages fill chunk 0 in order and the rest go to the continuation, so
    /// a message never moves ahead of one that was written before it.
    pub fn plan_chunks(&self, capacity: usize, ctx: &FormatContext) -> FormatResult<ChunkPlan> {
        let exact = self.messages_data_size();
        if exact <= capacity {
            return Ok(ChunkPlan {
                split: self.messages.len(),
                chunk0_size: self.prefix_size() + exact + 4,
                continuation_size: 0,
            });
        }
        // Chunk 0 has to keep room for the message naming the continuation,
        // whose body is the block's address and length (`H5O_CONT_ID`).
        let envelope = self.message_envelope_size();
        let continuation_message = envelope + ctx.sizeof_addr as usize + ctx.sizeof_size as usize;
        if capacity < continuation_message {
            return Err(FormatError::InvalidData(format!(
                "an object header chunk-0 capacity of {capacity} bytes cannot hold the \
                 {continuation_message}-byte message naming its continuation chunk"
            )));
        }
        let mut used = continuation_message;
        let mut split = 0;
        for msg in &self.messages {
            let size = envelope + msg.data.len();
            if used + size > capacity {
                break;
            }
            used += size;
            split += 1;
        }
        let spilled = self.messages_size(&self.messages[split..]);
        Ok(ChunkPlan {
            split,
            chunk0_size: self.prefix_size() + capacity + 4,
            continuation_size: OCHK_SIGNATURE.len() + spilled + 4,
        })
    }

    /// Encode this header as `plan` divides it, with the continuation chunk at
    /// `continuation_addr`.
    ///
    /// Returns chunk 0 and, when the plan spills, the continuation chunk's
    /// image. `continuation_addr` is ignored for a plan that does not spill.
    pub fn encode_chunked(
        &self,
        plan: &ChunkPlan,
        ctx: &FormatContext,
        continuation_addr: u64,
    ) -> FormatResult<(Vec<u8>, Option<Vec<u8>>)> {
        if plan.continuation_size == 0 {
            return Ok((self.encode()?, None));
        }
        let mut chunk0: Vec<ObjectHeaderMessage> = self.messages[..plan.split].to_vec();
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let mut body = Vec::with_capacity(sa + ss);
        body.extend_from_slice(&continuation_addr.to_le_bytes()[..sa]);
        body.extend_from_slice(&(plan.continuation_size as u64).to_le_bytes()[..ss]);
        chunk0.push(ObjectHeaderMessage {
            msg_type: MSG_CONTINUATION,
            flags: 0x00,
            creation_index: 0,
            data: body,
        });
        let data_size = plan.chunk0_size - self.prefix_size() - 4;
        let continuation = self.encode_continuation(&self.messages[plan.split..])?;
        debug_assert_eq!(continuation.len(), plan.continuation_size);
        Ok((self.encode_chunk0(&chunk0, data_size)?, Some(continuation)))
    }

    /// Decode an object header from a byte buffer. Returns the parsed header
    /// and the number of bytes consumed from the buffer.
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        // Minimum: OHDR(4) + version(1) + flags(1) + chunk0_size(1) + checksum(4) = 11
        if buf.len() < 11 {
            return Err(FormatError::BufferTooShort {
                needed: 11,
                available: buf.len(),
            });
        }

        // Signature
        if buf[0..4] != OHDR_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        // Version
        let version = buf[4];
        if version != OHDR_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        // Bit 5 is stripped here and carried by `times` instead, so the two
        // can only ever agree — see [`ObjectHeader::flags`].
        let flags = buf[5] & !FLAG_STORE_TIMESTAMPS;
        let mut pos: usize = 6;

        // Optional timestamps (bit 5)
        let times = if buf[5] & FLAG_STORE_TIMESTAMPS != 0 {
            if buf.len() < pos + 16 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 16,
                    available: buf.len(),
                });
            }
            let read = |off: usize| {
                u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
            };
            let t = ObjectTimes {
                access: read(pos),
                modification: read(pos + 4),
                change: read(pos + 8),
                birth: read(pos + 12),
            };
            pos += 16;
            Some(t)
        } else {
            None
        };

        // Optional attr storage thresholds (bit 4)
        if flags & FLAG_NON_DEFAULT_ATTR_THRESHOLDS != 0 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            // Skip thresholds for now
            pos += 4;
        }

        // Chunk0 data size
        let chunk0_size_bytes = match flags & FLAG_SIZE_MASK {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };

        if buf.len() < pos + chunk0_size_bytes {
            return Err(FormatError::BufferTooShort {
                needed: pos + chunk0_size_bytes,
                available: buf.len(),
            });
        }

        let chunk0_data_size =
            crate::format::bytes::read_le_uint(&buf[pos..], chunk0_size_bytes) as usize;
        pos += chunk0_size_bytes;

        // We need chunk0_data_size bytes of messages + 4 bytes of checksum.
        // chunk0_data_size is a file field up to 8 bytes wide; guard the
        // addition so a crafted absurd value yields a clean error instead of
        // an overflow panic (debug) or wrap (release).
        let total_consumed = pos
            .checked_add(chunk0_data_size)
            .and_then(|x| x.checked_add(4))
            .ok_or_else(|| {
                FormatError::InvalidData("object header chunk-0 size overflows usize".into())
            })?;
        if buf.len() < total_consumed {
            return Err(FormatError::BufferTooShort {
                needed: total_consumed,
                available: buf.len(),
            });
        }

        // Verify checksum: covers everything from start up to (but not
        // including) the 4-byte checksum.
        let data_end = total_consumed - 4;
        let stored_cksum = u32::from_le_bytes([
            buf[data_end],
            buf[data_end + 1],
            buf[data_end + 2],
            buf[data_end + 3],
        ]);
        let computed_cksum = checksum_metadata(&buf[..data_end]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }

        // Parse messages
        let has_creation_order = flags & FLAG_ATTR_CREATION_ORDER_TRACKED != 0;
        let messages_end = pos + chunk0_data_size;
        let mut messages = Vec::new();

        while pos < messages_end {
            // Each message: type(1) + size(2) + flags(1) [+ creation_order(2)]
            let msg_header_size = if has_creation_order { 6 } else { 4 };
            if pos + msg_header_size > messages_end {
                // libhdf5 (H5O__chunk_deserialize) permits a gap smaller than
                // one message header at the end of a v2 chunk; treat the
                // remaining bytes as such a gap rather than an error.
                break;
            }

            let msg_type = buf[pos];
            let msg_data_size = u16::from_le_bytes([buf[pos + 1], buf[pos + 2]]) as usize;
            let msg_flags = buf[pos + 3];
            pos += 4;

            let creation_index = if has_creation_order {
                let v = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
                pos += 2;
                v
            } else {
                0
            };

            if pos + msg_data_size > messages_end {
                return Err(FormatError::InvalidData(format!(
                    "message data ({} bytes) extends past chunk0 boundary",
                    msg_data_size
                )));
            }

            let data = buf[pos..pos + msg_data_size].to_vec();
            pos += msg_data_size;

            messages.push(ObjectHeaderMessage {
                msg_type,
                flags: msg_flags,
                creation_index,
                data,
            });
        }

        Ok((
            ObjectHeader {
                flags,
                times,
                messages,
            },
            total_consumed,
        ))
    }
}

impl Default for ObjectHeader {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Object Header v1 — for reading and writing legacy HDF5 files
// =========================================================================

/// `H5O_ALIGN_OLD` (H5Opkg.h:57): version 1 rounds every message body, and the
/// header prefix, up to a multiple of 8.
fn align_old(n: usize) -> usize {
    (n + 7) & !7
}

/// `H5O_SIZEOF_HDR` for version 1 (H5Opkg.h:85): `H5O_ALIGN_OLD(1 + 1 + 2 + 4
/// + 4)` — the four trailing bytes are the alignment pad, not a field.
const V1_PREFIX_SIZE: usize = 16;

/// `H5O_SIZEOF_MSGHDR_VERS` for version 1 (H5Opkg.h:112):
/// `H5O_ALIGN_OLD(2 + 2 + 1 + 3)`.
const V1_MSG_HEADER_SIZE: usize = 8;

impl ObjectHeader {
    /// Encode this header in the version-1 format, with `nlink` as the object
    /// reference count.
    ///
    /// The differences from [`encode`](Self::encode) are all
    /// `H5O_ALIGN_OLD`'s doing: no signature and no checksum, a two-byte
    /// message type, three reserved bytes where version 2 puts the optional
    /// creation index, and every message body padded out to eight bytes with
    /// the *padded* length in the size field (`H5O_msg_flush` writes
    /// `mesg->raw_size`, which `H5O__alloc` already aligned). The message
    /// count is the count of messages in the whole header, and this writer
    /// emits one chunk, so it is `self.messages.len()`.
    ///
    /// Refuses a header carrying attribute-creation-order flags: those bits
    /// live in the version-2 flags byte, which version 1 does not have, so
    /// encoding such a header would silently drop the policy the caller set.
    ///
    /// Refuses a header carrying [`times`](Self::times) for the same reason.
    /// The four times and the `H5O_HDR_STORE_TIMES` bit that announces them
    /// are version-2 prefix fields; a version-1 header records only its
    /// modification time, in an `H5O_MTIME_NEW` message, which this writer
    /// does not emit. Refusing keeps the version gate at the one encoder that
    /// knows which prefix it is writing, instead of letting a v2-shaped header
    /// reach `encode_v1` and come back out with its times gone.
    pub fn encode_v1(&self, nlink: u32) -> FormatResult<Vec<u8>> {
        if self.flags & (FLAG_ATTR_CREATION_ORDER_TRACKED | FLAG_ATTR_CREATION_ORDER_INDEXED) != 0 {
            return Err(FormatError::InvalidData(
                "a version-1 object header cannot record attribute creation order: \
                 the tracking flags exist only in the version-2 header prefix"
                    .into(),
            ));
        }
        if self.times.is_some() {
            return Err(FormatError::InvalidData(
                "a version-1 object header cannot store access/modification/change/birth \
                 times: they are version-2 prefix fields, and version 1 carries only a \
                 modification time, as an H5O_MTIME_NEW message"
                    .into(),
            ));
        }
        let mut data_size = 0usize;
        for msg in &self.messages {
            let padded = align_old(msg.data.len());
            if padded > MAX_MESSAGE_SIZE {
                return Err(FormatError::InvalidData(format!(
                    "object header message type 0x{:02X} is {} bytes, {padded} once \
                     aligned to 8, over the {MAX_MESSAGE_SIZE}-byte limit the message \
                     size field can express",
                    msg.msg_type,
                    msg.data.len()
                )));
            }
            data_size += V1_MSG_HEADER_SIZE + padded;
        }
        let Ok(chunk0_size) = u32::try_from(data_size) else {
            return Err(FormatError::InvalidData(format!(
                "version-1 object header chunk 0 is {data_size} bytes, over the 4-byte \
                 size field's range"
            )));
        };
        let Ok(nmesgs) = u16::try_from(self.messages.len()) else {
            return Err(FormatError::InvalidData(format!(
                "version-1 object header holds {} messages, over the 2-byte count \
                 field's range",
                self.messages.len()
            )));
        };

        let total = V1_PREFIX_SIZE + data_size;
        let mut buf = Vec::with_capacity(total);
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&nmesgs.to_le_bytes());
        buf.extend_from_slice(&nlink.to_le_bytes());
        buf.extend_from_slice(&chunk0_size.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // pad to H5O_ALIGN_OLD(12)

        for msg in &self.messages {
            let padded = align_old(msg.data.len());
            buf.extend_from_slice(&u16::from(msg.msg_type).to_le_bytes());
            buf.extend_from_slice(&(padded as u16).to_le_bytes());
            buf.push(msg.flags);
            buf.extend_from_slice(&[0u8; 3]); // reserved
            buf.extend_from_slice(&msg.data);
            buf.resize(buf.len() + (padded - msg.data.len()), 0);
        }

        debug_assert_eq!(buf.len(), total);
        Ok(buf)
    }

    /// Encode this header in the version `format` calls for.
    ///
    /// `nlink` reaches the file only in the version-1 layout; the version-2
    /// header has no reference-count field (an object with more than one hard
    /// link carries an Object Reference Count message instead).
    pub fn encode_for(
        &self,
        format: crate::format::ObjectFormat,
        nlink: u32,
    ) -> FormatResult<Vec<u8>> {
        match format {
            crate::format::ObjectFormat::Legacy => self.encode_v1(nlink),
            crate::format::ObjectFormat::Modern => self.encode(),
        }
    }
}

impl ObjectHeader {
    /// Decode a v1 object header from a byte buffer.
    ///
    /// v1 headers do NOT have the "OHDR" signature or a checksum. The layout is:
    /// ```text
    /// Byte 0: version = 1
    /// Byte 1: reserved
    /// Bytes 2-3: num_messages (u16 LE)
    /// Bytes 4-7: obj_ref_count (u32 LE)
    /// Bytes 8-11: header_data_size (u32 LE) — size of message data in first chunk
    /// Messages follow, each:
    ///   type: u16 LE
    ///   data_size: u16 LE
    ///   flags: u8
    ///   reserved: 3 bytes
    ///   data: data_size bytes (padded to 8-byte alignment)
    /// ```
    pub fn decode_v1(buf: &[u8]) -> FormatResult<(Self, usize)> {
        // V1 header prefix is 16 bytes: version(1) + reserved(1) + num_msg(2)
        // + ref_count(4) + chunk0_data_size(4) + reserved_padding(4)
        if buf.len() < 16 {
            return Err(FormatError::BufferTooShort {
                needed: 16,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != 1 {
            return Err(FormatError::InvalidVersion(version));
        }

        // buf[1] = reserved
        let num_messages = u16::from_le_bytes([buf[2], buf[3]]) as usize;
        let _obj_ref_count = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let header_data_size = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        // buf[12..16] = reserved alignment padding

        let total_consumed = 16 + header_data_size;
        if buf.len() < total_consumed {
            return Err(FormatError::BufferTooShort {
                needed: total_consumed,
                available: buf.len(),
            });
        }

        let msg_data_start = 16; // offset where message data begins (after 16-byte prefix)
        let mut pos = msg_data_start;
        let messages_end = msg_data_start + header_data_size;
        let mut messages = Vec::with_capacity(num_messages);

        for _ in 0..num_messages {
            if pos + 8 > messages_end {
                break; // no more room for a message header
            }

            let msg_type = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            let data_size = u16::from_le_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
            let msg_flags = buf[pos + 4];
            // bytes pos+5..pos+8 are reserved
            pos += 8;

            if pos + data_size > messages_end {
                return Err(FormatError::InvalidData(format!(
                    "v1 message data ({} bytes) extends past header boundary",
                    data_size
                )));
            }

            let data = buf[pos..pos + data_size].to_vec();
            pos += data_size;

            // In v1, messages are padded to 8-byte alignment relative to
            // the start of the message data region.
            let rel = pos - msg_data_start;
            let aligned_rel = (rel + 7) & !7;
            let aligned_pos = msg_data_start + aligned_rel;
            if aligned_pos <= messages_end {
                pos = aligned_pos;
            }

            // Skip null/padding messages (type 0)
            if msg_type == 0 {
                continue;
            }

            messages.push(ObjectHeaderMessage {
                msg_type: msg_type as u8,
                flags: msg_flags,
                // A version-1 message envelope has no creation index.
                creation_index: 0,
                data,
            });
        }

        Ok((
            ObjectHeader {
                flags: 0x02, // default flags (not meaningful for v1)
                // A v1 header keeps its modification time in a message
                // (`H5O_MSG_MTIME`), never in the prefix.
                times: None,
                messages,
            },
            total_consumed,
        ))
    }

    /// Auto-detect and decode either v1 or v2 object header.
    ///
    /// Checks for the "OHDR" signature to decide v2; otherwise tries v1.
    pub fn decode_any(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() >= 4 && buf[0..4] == OHDR_SIGNATURE {
            Self::decode(buf)
        } else if !buf.is_empty() && buf[0] == 1 {
            Self::decode_v1(buf)
        } else {
            // Try v2 first (will fail with proper error)
            Self::decode(buf)
        }
    }
}

#[cfg(test)]
mod tests_v1 {
    use super::*;

    /// Build a minimal v1 object header with given messages.
    fn build_v1_header(messages: &[(u16, u8, &[u8])]) -> Vec<u8> {
        let mut msg_data = Vec::new();
        for (msg_type, flags, data) in messages {
            msg_data.extend_from_slice(&msg_type.to_le_bytes());
            msg_data.extend_from_slice(&(data.len() as u16).to_le_bytes());
            msg_data.push(*flags);
            msg_data.extend_from_slice(&[0u8; 3]); // reserved
            msg_data.extend_from_slice(data);
            // Pad to 8-byte alignment
            let aligned = (msg_data.len() + 7) & !7;
            msg_data.resize(aligned, 0);
        }

        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(messages.len() as u16).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ref count
        buf.extend_from_slice(&(msg_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // reserved padding (align to 16 bytes)
        buf.extend_from_slice(&msg_data);
        buf
    }

    #[test]
    fn test_decode_v1_empty() {
        let buf = build_v1_header(&[]);
        let (hdr, consumed) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(consumed, 16); // 16-byte prefix, no messages
        assert!(hdr.messages.is_empty());
    }

    #[test]
    fn test_decode_v1_single_message() {
        let data = vec![0xAA, 0xBB, 0xCC];
        let buf = build_v1_header(&[(0x03, 0x00, &data)]);
        let (hdr, _consumed) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, 0x03);
        assert_eq!(hdr.messages[0].data, data);
    }

    #[test]
    fn test_decode_v1_multiple_messages() {
        let buf = build_v1_header(&[
            (0x01, 0x00, &[1, 2, 3, 4]),
            (0x03, 0x01, &[10, 20]),
            (0x08, 0x00, &[0xFF; 16]),
        ]);
        let (hdr, _) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(hdr.messages.len(), 3);
        assert_eq!(hdr.messages[0].msg_type, 0x01);
        assert_eq!(hdr.messages[1].msg_type, 0x03);
        assert_eq!(hdr.messages[2].msg_type, 0x08);
        assert_eq!(hdr.messages[2].data, vec![0xFF; 16]);
    }

    /// The exact bytes h5py 3.15/libhdf5 1.14.6 wrote for the header of a
    /// contiguous `<i4` dataset of shape (6,) in a default (superblock-0)
    /// file: five messages, the last a null pad, chunk 0 of 256 bytes. Only
    /// the first four are re-encoded here — the pad is libhdf5 pre-allocating
    /// room to grow, not content — so the assertion is on the prefix shape and
    /// on each message's aligned envelope.
    #[test]
    fn an_encoded_v1_header_matches_the_envelope_libhdf5_writes() {
        let dataspace = vec![
            0x01, 0x01, 0x01, 0x00, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ];
        let datatype = vec![0x10, 0x08, 0, 0, 0x04, 0, 0, 0, 0, 0, 0x20, 0, 0, 0, 0, 0];
        let fill = vec![0x02, 0x02, 0x02, 0x01, 0, 0, 0, 0];
        // 18 raw bytes: version 3 contiguous layout, address then size.
        let layout = vec![
            0x03, 0x01, 0, 0x08, 0, 0, 0, 0, 0, 0, 0x18, 0, 0, 0, 0, 0, 0, 0,
        ];
        let mut header = ObjectHeader::new();
        header.add_message(0x01, 0x00, dataspace);
        header.add_message(0x03, 0x01, datatype);
        header.add_message(0x05, 0x01, fill);
        header.add_message(0x08, 0x00, layout);

        let buf = header.encode_v1(1).unwrap();
        assert_eq!(buf[0], 1, "version");
        assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), 4, "message count");
        assert_eq!(
            u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            1,
            "reference count"
        );
        // (8 + 24) + (8 + 16) + (8 + 8) + (8 + 24), the layout body aligned
        // from 18 to 24.
        assert_eq!(
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            104,
            "chunk 0 data size"
        );
        assert_eq!(buf.len(), 16 + 104);
        // The layout message's size field records the aligned length.
        let layout_at = 16 + 32 + 24 + 16;
        assert_eq!(u16::from_le_bytes([buf[layout_at], buf[layout_at + 1]]), 8);
        assert_eq!(
            u16::from_le_bytes([buf[layout_at + 2], buf[layout_at + 3]]),
            24
        );
        assert_eq!(&buf[layout_at + 8 + 18..layout_at + 8 + 24], &[0u8; 6]);
    }

    #[test]
    fn a_v1_header_round_trips_through_its_own_decoder() {
        let mut header = ObjectHeader::new();
        header.add_message(0x11, 0x00, vec![0xAB; 16]);
        header.add_message(0x0C, 0x00, vec![0xCD; 21]);
        let buf = header.encode_v1(3).unwrap();
        let (back, consumed) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.messages[0].data, vec![0xAB; 16]);
        // The 21-byte body came back padded to 24, so re-encoding is stable.
        assert_eq!(back.messages[1].data.len(), 24);
        assert_eq!(back.encode_v1(3).unwrap(), buf);
        // `decode_any` must not mistake it for a version-2 header.
        assert_eq!(ObjectHeader::decode_any(&buf).unwrap().1, buf.len());
    }

    #[test]
    fn a_v1_header_refuses_to_drop_attribute_creation_order() {
        let mut header = ObjectHeader::new();
        header.set_attribute_creation_order(CreationOrder::Tracked);
        assert!(matches!(
            header.encode_v1(1).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    /// The times gate is the header version, not the caller: a version-1
    /// prefix has nowhere to put them, so `encode_v1` says so instead of
    /// returning a header whose times are gone.
    #[test]
    fn a_v1_header_refuses_to_drop_its_stored_times() {
        let mut header = ObjectHeader::new();
        header.add_message(0x11, 0x00, vec![0u8; 16]);
        assert!(header.encode_v1(1).is_ok());

        header.times = Some(ObjectTimes::created_at(0x1234_5678));
        assert!(matches!(
            header.encode_v1(1).unwrap_err(),
            FormatError::InvalidData(_)
        ));
        // The same header is fine as version 2, where the prefix holds them.
        let v2 = header.encode().unwrap();
        assert_eq!(v2[5] & FLAG_STORE_TIMESTAMPS, FLAG_STORE_TIMESTAMPS);
    }

    #[test]
    fn encode_for_picks_the_version_the_format_calls_for() {
        use crate::format::ObjectFormat;
        let mut header = ObjectHeader::new();
        header.add_message(0x11, 0x00, vec![0u8; 16]);
        assert_eq!(header.encode_for(ObjectFormat::Legacy, 1).unwrap()[0], 1);
        assert_eq!(
            &header.encode_for(ObjectFormat::Modern, 1).unwrap()[0..4],
            &OHDR_SIGNATURE
        );
    }

    #[test]
    fn test_decode_v1_skips_null_messages() {
        let buf = build_v1_header(&[
            (0x00, 0x00, &[0; 8]), // null message (type 0)
            (0x03, 0x00, &[1, 2]),
        ]);
        let (hdr, _) = ObjectHeader::decode_v1(&buf).unwrap();
        assert_eq!(hdr.messages.len(), 1);
        assert_eq!(hdr.messages[0].msg_type, 0x03);
    }

    #[test]
    fn test_decode_any_v2() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);
        let encoded = hdr.encode().unwrap();
        let (decoded, _) = ObjectHeader::decode_any(&encoded).unwrap();
        assert_eq!(decoded.messages.len(), 1);
    }

    #[test]
    fn test_decode_any_v1() {
        let buf = build_v1_header(&[(0x03, 0x00, &[1, 2])]);
        let (decoded, _) = ObjectHeader::decode_any(&buf).unwrap();
        assert_eq!(decoded.messages.len(), 1);
        assert_eq!(decoded.messages[0].msg_type, 0x03);
    }

    #[test]
    fn test_decode_v1_bad_version() {
        let mut buf = build_v1_header(&[]);
        buf[0] = 5;
        assert!(matches!(
            ObjectHeader::decode_v1(&buf).unwrap_err(),
            FormatError::InvalidVersion(5)
        ));
    }

    #[test]
    fn test_decode_v1_buffer_too_short() {
        assert!(matches!(
            ObjectHeader::decode_v1(&[1, 0, 0]).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_header_roundtrip() {
        let hdr = ObjectHeader::new();
        let encoded = hdr.encode().unwrap();

        // OHDR(4) + version(1) + flags(1) + chunk0_size(4) + checksum(4) = 14
        assert_eq!(encoded.len(), 14);
        assert_eq!(&encoded[..4], b"OHDR");
        assert_eq!(encoded[4], 2); // version

        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_single_message_roundtrip() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![0xAA, 0xBB, 0xCC]);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages.len(), 1);
        assert_eq!(decoded.messages[0].msg_type, 0x01);
        assert_eq!(decoded.messages[0].flags, 0x00);
        assert_eq!(decoded.messages[0].data, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_multiple_messages_roundtrip() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3, 4]);
        hdr.add_message(0x03, 0x01, vec![10, 20]);
        hdr.add_message(0x0C, 0x00, vec![]);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages.len(), 3);
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_with_creation_order() {
        let mut hdr = ObjectHeader {
            flags: 0x02 | FLAG_ATTR_CREATION_ORDER_TRACKED,
            times: None,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![0xFF; 8]);
        hdr.add_message(0x03, 0x00, vec![0xEE; 4]);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages.len(), 2);
        assert_eq!(decoded.messages[0].data, vec![0xFF; 8]);
        assert_eq!(decoded.messages[1].data, vec![0xEE; 4]);
    }

    /// The per-message creation index survives a round trip, and lands where
    /// libhdf5 puts it: right after the message flags byte, ahead of the
    /// message data.
    #[test]
    fn a_tracked_header_round_trips_each_message_creation_index() {
        let mut hdr = ObjectHeader::new();
        hdr.set_attribute_creation_order(CreationOrder::Indexed);
        hdr.add_message_indexed(0x0C, 0x00, vec![0xAA; 6], 0);
        hdr.add_message_indexed(0x0C, 0x00, vec![0xBB; 6], 1);
        hdr.add_message_indexed(0x0C, 0x00, vec![0xCC; 6], 2);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, hdr);
        let indices: Vec<u16> = decoded.messages.iter().map(|m| m.creation_index).collect();
        assert_eq!(indices, vec![0, 1, 2]);

        // Off: no index is written, and every message decodes with 0.
        let mut plain = ObjectHeader::new();
        plain.add_message(0x0C, 0x00, vec![0xAA; 6]);
        assert!(plain.encode().unwrap().len() < encoded.len());
        let (plain_back, _) = ObjectHeader::decode(&plain.encode().unwrap()).unwrap();
        assert_eq!(plain_back.messages[0].creation_index, 0);
    }

    /// The two flag bits are set and read back independently, so a header that
    /// tracks without indexing survives a decode as exactly that — the state
    /// `H5Pset_attr_creation_order(H5P_CRT_ORDER_TRACKED)` produces.
    #[test]
    fn each_attribute_creation_order_state_round_trips_through_the_flags() {
        for order in [
            CreationOrder::Untracked,
            CreationOrder::Tracked,
            CreationOrder::Indexed,
        ] {
            let mut hdr = ObjectHeader::new();
            hdr.set_attribute_creation_order(order);
            hdr.add_message_indexed(0x0C, 0x00, vec![0xAA; 6], 3);
            let (decoded, _) = ObjectHeader::decode(&hdr.encode().unwrap()).unwrap();
            assert_eq!(decoded.attribute_creation_order(), order);
            let want_index = if order.is_tracked() { 3 } else { 0 };
            assert_eq!(decoded.messages[0].creation_index, want_index);
        }
    }

    /// Setting a policy clears whatever the previous one left behind, so a
    /// header recovered as indexed and re-declared untracked does not keep a
    /// stale bit.
    #[test]
    fn setting_a_weaker_policy_clears_the_stronger_one() {
        let mut hdr = ObjectHeader::new();
        hdr.set_attribute_creation_order(CreationOrder::Indexed);
        hdr.set_attribute_creation_order(CreationOrder::Untracked);
        assert_eq!(hdr.attribute_creation_order(), CreationOrder::Untracked);
        // Bits 0-1 (the chunk0 size encoding) are untouched.
        assert_eq!(hdr.flags, 0x02);
    }

    /// The four times survive a round trip in `H5O__cache_serialize` order,
    /// and their presence is what sets `H5O_HDR_STORE_TIMES` in the file.
    #[test]
    fn stored_times_round_trip_and_set_the_flag() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);
        let without = hdr.encode().unwrap();

        hdr.times = Some(ObjectTimes {
            access: 0x0A0A_0A0A,
            modification: 0x0B0B_0B0B,
            change: 0x0C0C_0C0C,
            birth: 0x0D0D_0D0D,
        });
        let with = hdr.encode().unwrap();

        assert_eq!(with.len(), without.len() + 16);
        assert_eq!(with[5] & FLAG_STORE_TIMESTAMPS, FLAG_STORE_TIMESTAMPS);
        assert_eq!(&with[6..10], &0x0A0A_0A0Au32.to_le_bytes());
        assert_eq!(&with[10..14], &0x0B0B_0B0Bu32.to_le_bytes());
        assert_eq!(&with[14..18], &0x0C0C_0C0Cu32.to_le_bytes());
        assert_eq!(&with[18..22], &0x0D0D_0D0Du32.to_le_bytes());

        let (decoded, consumed) = ObjectHeader::decode(&with).expect("decode failed");
        assert_eq!(consumed, with.len());
        assert_eq!(decoded, hdr);
    }

    /// The flag cannot be set without the times behind it: bit 5 poked into
    /// `flags` by hand is dropped at encode rather than announcing sixteen
    /// bytes that are not there — the shape that made a rewrite emit zero
    /// timestamps.
    #[test]
    fn the_timestamps_flag_is_never_written_without_times() {
        let mut hdr = ObjectHeader::new();
        hdr.flags |= FLAG_STORE_TIMESTAMPS;
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);

        let encoded = hdr.encode().unwrap();
        assert_eq!(encoded[5] & FLAG_STORE_TIMESTAMPS, 0);
        let (decoded, _) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.times, None);
        assert_eq!(decoded.flags & FLAG_STORE_TIMESTAMPS, 0);
    }

    /// `H5O_touch_oh` on a version-2 header moves access and change time to
    /// now and leaves modification and birth time alone.
    #[test]
    fn touching_moves_access_and_change_time_only() {
        let before = ObjectTimes {
            access: 100,
            modification: 200,
            change: 300,
            birth: 400,
        };
        assert_eq!(
            before.touched(999),
            ObjectTimes {
                access: 999,
                modification: 200,
                change: 999,
                birth: 400,
            }
        );
        assert_eq!(
            ObjectTimes::created_at(7).touched(7),
            ObjectTimes::created_at(7)
        );
    }

    #[test]
    fn test_chunk0_size_1byte() {
        // flags bits 0-1 = 0 => 1-byte chunk0 size
        let mut hdr = ObjectHeader {
            flags: 0x00,
            times: None,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![42]);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data, vec![42]);
    }

    #[test]
    fn test_chunk0_size_2byte() {
        // flags bits 0-1 = 1 => 2-byte chunk0 size
        let mut hdr = ObjectHeader {
            flags: 0x01,
            times: None,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn test_chunk0_size_8byte() {
        // flags bits 0-1 = 3 => 8-byte chunk0 size
        let mut hdr = ObjectHeader {
            flags: 0x03,
            times: None,
            messages: Vec::new(),
        };
        hdr.add_message(0x01, 0x00, vec![0xDE, 0xAD]);

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_decode_bad_signature() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"XHDR");
        let err = ObjectHeader::decode(&data).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn test_decode_bad_version() {
        let hdr = ObjectHeader::new();
        let mut encoded = hdr.encode().unwrap();
        encoded[4] = 99; // corrupt version
        let err = ObjectHeader::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(99)));
    }

    #[test]
    fn test_decode_checksum_mismatch() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![1, 2, 3]);
        let mut encoded = hdr.encode().unwrap();
        // Corrupt a message byte
        let last_data = encoded.len() - 5;
        encoded[last_data] ^= 0xFF;
        let err = ObjectHeader::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let err = ObjectHeader::decode(&[0u8; 5]).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn test_decode_with_trailing_data() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![7, 8, 9]);
        let mut encoded = hdr.encode().unwrap();
        let original_len = encoded.len();
        encoded.extend_from_slice(&[0xBB; 50]); // trailing garbage

        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, original_len);
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_large_message_payload() {
        let mut hdr = ObjectHeader::new();
        let big_data = vec![0x42; 1000];
        hdr.add_message(0x0C, 0x00, big_data.clone());

        let encoded = hdr.encode().unwrap();
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data.len(), 1000);
        assert_eq!(decoded.messages[0].data, big_data);
    }

    /// The largest payload the size field can express still round-trips.
    #[test]
    fn test_message_payload_at_size_limit() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x0C, 0x00, vec![0x42; MAX_MESSAGE_SIZE]);

        let encoded = hdr.encode().expect("encode at the limit must succeed");
        let (decoded, consumed) = ObjectHeader::decode(&encoded).expect("decode failed");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.messages[0].data.len(), MAX_MESSAGE_SIZE);
    }

    /// One byte past the limit is refused. Encoding it would store the length
    /// modulo 65536 — here 0 — and every message after it would decode from
    /// the middle of this one's payload, with a checksum that still matches.
    #[test]
    fn test_message_payload_over_size_limit_is_refused() {
        let mut hdr = ObjectHeader::new();
        hdr.add_message(0x01, 0x00, vec![7; 4]);
        hdr.add_message(0x0C, 0x00, vec![0x42; MAX_MESSAGE_SIZE + 1]);

        let err = hdr.encode().expect_err("over the limit must not encode");
        let msg = err.to_string();
        assert!(msg.contains("0x0C"), "{msg}");
        assert!(msg.contains(&(MAX_MESSAGE_SIZE + 1).to_string()), "{msg}");
    }

    #[test]
    fn test_default() {
        let hdr = ObjectHeader::default();
        assert_eq!(hdr.flags, 0x02);
        assert!(hdr.messages.is_empty());
    }

    /// A header with three 40-byte messages, for the chunking tests below.
    fn chunked_header() -> ObjectHeader {
        let mut hdr = ObjectHeader::new();
        for (i, t) in [0x02u8, 0x0A, 0x0C].iter().enumerate() {
            hdr.add_message(*t, 0x00, vec![i as u8; 40]);
        }
        hdr
    }

    /// A capacity that covers every message leaves one chunk, and the image is
    /// the one `encode` produces on its own.
    #[test]
    fn a_capacity_that_fits_every_message_plans_one_chunk() {
        let hdr = chunked_header();
        let ctx = FormatContext::default_v3();
        let plan = hdr.plan_chunks(1024, &ctx).unwrap();
        assert_eq!(plan.continuation_size, 0);
        let (chunk0, continuation) = hdr.encode_chunked(&plan, &ctx, 0x1000).unwrap();
        assert!(continuation.is_none());
        assert_eq!(chunk0, hdr.encode().unwrap());
        assert_eq!(chunk0.len(), plan.chunk0_size);
    }

    /// Past the capacity the tail of the message list moves into an `OCHK`
    /// chunk, chunk 0 names it by address and length, and both chunks carry a
    /// checksum over their own image.
    #[test]
    fn messages_past_the_capacity_move_into_a_continuation_chunk() {
        let hdr = chunked_header();
        let ctx = FormatContext::default_v3();
        // Room for one 44-byte message and the 20-byte continuation message.
        let plan = hdr.plan_chunks(64, &ctx).unwrap();
        assert_eq!(plan.continuation_size, 4 + 2 * 44 + 4);
        let (chunk0, continuation) = hdr.encode_chunked(&plan, &ctx, 0x2000).unwrap();
        let continuation = continuation.unwrap();
        assert_eq!(chunk0.len(), plan.chunk0_size);
        assert_eq!(continuation.len(), plan.continuation_size);
        assert_eq!(&continuation[..4], &OCHK_SIGNATURE);

        let (decoded, consumed) = ObjectHeader::decode(&chunk0).unwrap();
        assert_eq!(consumed, chunk0.len());
        assert_eq!(
            decoded.messages.len(),
            2,
            "the first message and the pointer"
        );
        assert_eq!(decoded.messages[0], hdr.messages[0]);
        let pointer = &decoded.messages[1];
        assert_eq!(pointer.msg_type, MSG_CONTINUATION);
        assert_eq!(
            u64::from_le_bytes(pointer.data[..8].try_into().unwrap()),
            0x2000
        );
        assert_eq!(
            u64::from_le_bytes(pointer.data[8..16].try_into().unwrap()),
            plan.continuation_size as u64
        );

        let body = &continuation[..continuation.len() - 4];
        let stored = u32::from_le_bytes(continuation[continuation.len() - 4..].try_into().unwrap());
        assert_eq!(stored, checksum_metadata(body));
    }

    /// Space left in chunk 0 becomes a NIL message when a message envelope
    /// fits in it, and zero bytes — a "gap" — when it does not.
    #[test]
    fn leftover_chunk_zero_space_is_a_nil_message_or_a_gap() {
        let hdr = chunked_header();
        let ctx = FormatContext::default_v3();

        // 44 (message) + 20 (continuation) + 8 leaves room for a NIL.
        let (chunk0, _) = hdr
            .encode_chunked(&hdr.plan_chunks(72, &ctx).unwrap(), &ctx, 0x2000)
            .unwrap();
        let tail = &chunk0[chunk0.len() - 4 - 8..chunk0.len() - 4];
        assert_eq!(tail, [MSG_NIL, 4, 0, 0, 0, 0, 0, 0]);

        // Three bytes over is one short of an envelope, so they stay a gap.
        let (chunk0, _) = hdr
            .encode_chunked(&hdr.plan_chunks(67, &ctx).unwrap(), &ctx, 0x2000)
            .unwrap();
        assert_eq!(&chunk0[chunk0.len() - 4 - 3..chunk0.len() - 4], [0, 0, 0]);
        // A gap is not a message: the decoder stops at it.
        let (decoded, _) = ObjectHeader::decode(&chunk0).unwrap();
        assert_eq!(decoded.messages.len(), 2);
    }

    /// A capacity with no room for the message naming the continuation cannot
    /// be planned: chunk 0 would spill with nothing pointing at the spill.
    #[test]
    fn a_capacity_below_the_continuation_message_is_refused() {
        let hdr = chunked_header();
        let err = hdr
            .plan_chunks(19, &FormatContext::default_v3())
            .expect_err("19 bytes cannot hold a 20-byte continuation message");
        assert!(err.to_string().contains("continuation chunk"), "{err}");
    }

    /// The times a version-2 header stores sit in its prefix, ahead of the
    /// message area a plan divides — so a header carrying them reserves
    /// sixteen more bytes for chunk 0 and spills at exactly the same message.
    ///
    /// `chunk0_capacity` in the writer budgets the *message* area, and the
    /// prefix is added on top of it here; a plan that folded the times into
    /// that budget would size chunk 0 sixteen bytes short of the image
    /// `encode_chunked` then produces, which `check_header_size` refuses.
    #[test]
    fn the_times_prefix_widens_chunk_zero_without_moving_the_split() {
        let ctx = FormatContext::default_v3();
        let plain = chunked_header();
        let mut timed = chunked_header();
        timed.times = Some(ObjectTimes::created_at(0x5EED_1234));

        for capacity in [72usize, 120, 1024] {
            let a = plain.plan_chunks(capacity, &ctx).unwrap();
            let b = timed.plan_chunks(capacity, &ctx).unwrap();
            assert_eq!(a.split, b.split, "capacity {capacity}: same split");
            assert_eq!(
                a.continuation_size, b.continuation_size,
                "capacity {capacity}: same continuation"
            );
            assert_eq!(
                b.chunk0_size,
                a.chunk0_size + 16,
                "capacity {capacity}: four times of four bytes"
            );
        }

        // And the plan describes the image: chunk 0 is the length the plan
        // said, times included.
        let plan = timed.plan_chunks(72, &ctx).unwrap();
        let (chunk0, continuation) = timed.encode_chunked(&plan, &ctx, 0x2000).unwrap();
        assert_eq!(chunk0.len(), plan.chunk0_size);
        assert_eq!(continuation.unwrap().len(), plan.continuation_size);
        let (decoded, _) = ObjectHeader::decode(&chunk0).unwrap();
        assert_eq!(decoded.times, timed.times);
    }

    /// A version-1 header has its own continuation rules (`H5O__chunk_deserialize`
    /// reads a v1 chunk with no `OCHK` signature and no checksum), so the
    /// version-2 planner must never be applied to one. `encode_for` is where
    /// that holds — and it is the whole of `Hdf5Writer::encode_header_at`'s
    /// legacy arm: the plan is never built, every message goes in chunk 0, and
    /// the two version-2 prefix fields are refused outright rather than
    /// planned around (`tests_v1::a_v1_header_refuses_to_drop_its_stored_times`,
    /// `tests_v1::a_v1_header_refuses_to_drop_attribute_creation_order`).
    #[test]
    fn a_version_one_header_is_never_planned_into_chunks() {
        let hdr = chunked_header();
        // Far past any plausible chunk-0 estimate, and still one chunk.
        let v1 = hdr
            .encode_for(crate::format::ObjectFormat::Legacy, 1)
            .unwrap();
        assert_eq!(v1[0], 1, "version-1 prefix");
        assert!(
            !v1.windows(4).any(|w| w == OCHK_SIGNATURE),
            "a version-1 header holds no OCHK chunk"
        );
        let (decoded, _) = ObjectHeader::decode_v1(&v1).unwrap();
        assert_eq!(decoded.messages.len(), hdr.messages.len());
        assert!(
            !decoded
                .messages
                .iter()
                .any(|m| m.msg_type == MSG_CONTINUATION),
            "no continuation message"
        );
    }
}
