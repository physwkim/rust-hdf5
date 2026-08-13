//! Global Heap Collection (GCOL) -- stores variable-length data such as
//! variable-length strings.
//!
//! Binary layout of a Global Heap Collection:
//! ```text
//! "GCOL"              (4 bytes, signature)
//! version             (1 byte, must be 1)
//! reserved            (3 bytes)
//! collection_size     (sizeof_size bytes LE, total including header)
//!
//! Followed by heap objects:
//!   index             (u16 LE, 0 = free space / end marker, 1+ = object)
//!   ref_count          (u16 LE)
//!   reserved           (u32 LE)
//!   size               (sizeof_size bytes LE)
//!   data               (size bytes, padded to 8-byte alignment)
//! ```
//!
//! A variable-length reference stored in dataset raw data is:
//! ```text
//! sequence_length     (u32 LE, length of the vlen sequence)
//! collection_address  (sizeof_addr bytes LE, address of the GCOL)
//! object_index        (u32 LE, index within the collection)
//! ```
//! Total vlen reference size = 4 + sizeof_addr + 4 bytes.

use crate::format::bytes::read_le_uint as read_size;
use crate::format::{FormatContext, FormatError, FormatResult};

/// Signature for a global heap collection.
const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// Global heap collection version.
const GCOL_VERSION: u8 = 1;

/// Minimum collection size required by the HDF5 C library (H5HG_MINALLOC).
const GCOL_MIN_SIZE: usize = 4096;

/// Ceiling to which the CWFS second pass grows a collection in place
/// (`H5HG_MAXSIZE`): `H5F_cwfs_find_free_heap` extends a listed collection
/// only while `size + new_need <= H5HG_MAXSIZE`. Collections *created*
/// larger than this (one oversized object) are legal — they are just never
/// extended.
pub const GCOL_MAX_SIZE: usize = 65536;

/// A single object within a global heap collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalHeapObject {
    /// Object index (1-based). Index 0 is reserved for the free-space marker.
    pub index: u16,
    /// On-disk reference count. libhdf5 writes 0 on insert (`H5HG_insert`)
    /// and only its virtual-dataset layer ever raises it via `H5HG_link`,
    /// so new objects carry 0 — but a decoded object keeps what the file
    /// says, or rewriting a foreign collection after a removal would reset
    /// a VDS-linked object's count.
    pub ref_count: u16,
    /// Raw data stored in this object.
    pub data: Vec<u8>,
}

/// A global heap collection, containing a set of heap objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalHeapCollection {
    /// The heap objects in this collection (index > 0).
    pub objects: Vec<GlobalHeapObject>,
}

impl GlobalHeapCollection {
    /// Create an empty global heap collection.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Add a data blob to the collection. Returns the 1-based object index.
    pub fn add_object(&mut self, data: Vec<u8>) -> FormatResult<u16> {
        let max_index = self.objects.iter().map(|o| o.index).max().unwrap_or(0);
        // Object index 0 is the reserved free-space marker, so the usable
        // range is 1..=u16::MAX. Refuse to wrap past it.
        if max_index == u16::MAX {
            return Err(FormatError::InvalidData(
                "global heap collection is full (65535 objects)".into(),
            ));
        }
        let index = max_index + 1;
        self.objects.push(GlobalHeapObject {
            index,
            ref_count: 0,
            data,
        });
        Ok(index)
    }

    /// Retrieve the data for an object by its 1-based index.
    pub fn get_object(&self, index: u16) -> Option<&[u8]> {
        self.objects
            .iter()
            .find(|o| o.index == index)
            .map(|o| o.data.as_slice())
    }

    /// Drop the object with this 1-based index, reporting whether it was
    /// there.
    ///
    /// The surviving objects keep their indices, so the vlen references that
    /// name them stay valid; only their offsets within the collection move.
    /// This is what libhdf5's `H5HG_remove` does — it compacts the objects and
    /// gives the recovered bytes to the free-space marker — and re-encoding at
    /// the collection's existing size ([`encode_at_size`](Self::encode_at_size))
    /// reproduces that layout. Removing an index that is already gone is not
    /// an error: two elements of one dataset may name the same object, so a
    /// range update can reach it twice (libhdf5 tolerates the same case, see
    /// its HDFFV-10635 note).
    pub fn remove_object(&mut self, index: u16) -> bool {
        match self.objects.iter().position(|o| o.index == index) {
            Some(at) => {
                self.objects.remove(at);
                true
            }
            None => false,
        }
    }

    /// True when the collection holds no objects, so its block is free.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// The bytes the free-space marker owns (its own 16-byte header
    /// included) when this collection is encoded into a `collection_size`
    /// block, or `None` when the objects (plus that marker header) do not
    /// fit — the fit test [`encode_at_size`](Self::encode_at_size) applies.
    pub fn free_space_at(&self, ctx: &FormatContext, collection_size: usize) -> Option<usize> {
        let (header_size, objhdr_size, objects_size) = self.layout(ctx);
        let content_size = header_size + objects_size + objhdr_size;
        if collection_size < content_size {
            return None;
        }
        Some(collection_size - header_size - objects_size)
    }

    /// The bytes an object of `data_len` occupies on disk: its aligned
    /// header plus the 8-byte-aligned data. What one insert takes from a
    /// collection's free space.
    pub fn object_disk_size(ctx: &FormatContext, data_len: usize) -> usize {
        let ss = ctx.sizeof_size as usize;
        pad_to_8(2 + 2 + 4 + ss) + pad_to_8(data_len)
    }

    /// The highest object index in use (0 when the collection is empty).
    pub fn max_index(&self) -> u16 {
        self.objects.iter().map(|o| o.index).max().unwrap_or(0)
    }

    /// Encode the collection into a byte vector.
    ///
    /// The encoded blob includes the GCOL header and all heap objects,
    /// followed by a free-space marker (index=0 object).
    /// The total size is padded to at least 4096 bytes (H5HG_MINALLOC)
    /// for compatibility with the HDF5 C library.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        // Unwrap: the size asked for is the one `encode_at_size` computes as
        // the minimum, so it always fits.
        self.encode_at_size(ctx, self.encoded_size(ctx)).unwrap()
    }

    /// The smallest collection size that holds these objects — what
    /// [`encode`](Self::encode) uses.
    pub fn encoded_size(&self, ctx: &FormatContext) -> usize {
        let (header_size, objhdr_size, objects_size) = self.layout(ctx);
        (header_size + objects_size + objhdr_size).max(GCOL_MIN_SIZE)
    }

    /// Encode the collection into a block of exactly `collection_size` bytes,
    /// with everything not used by an object given to the free-space marker.
    ///
    /// A collection is rewritten in place after [`remove_object`](Self::remove_object),
    /// and its block does not shrink — libhdf5's `H5HG_remove` keeps the
    /// collection's size and grows its free space instead, and shortening the
    /// image would leave the tail of the previous, longer one on disk. Fails
    /// if the objects do not fit in `collection_size`.
    pub fn encode_at_size(
        &self,
        ctx: &FormatContext,
        collection_size: usize,
    ) -> FormatResult<Vec<u8>> {
        let ss = ctx.sizeof_size as usize;
        let (header_size, objhdr_size, objects_size) = self.layout(ctx);

        // The free-space marker's own header has to fit after the objects.
        let content_size = header_size + objects_size + objhdr_size;
        if collection_size < content_size {
            return Err(FormatError::InvalidData(format!(
                "global heap collection needs {content_size} bytes but was given {collection_size}"
            )));
        }
        // HDF5 convention: free marker size = collection_size - header - objects
        // (includes the free marker's own header in the "free space")
        let free_space = collection_size - header_size - objects_size;

        let mut buf = Vec::with_capacity(collection_size);

        // Header
        buf.extend_from_slice(&GCOL_SIGNATURE);
        buf.push(GCOL_VERSION);
        buf.extend_from_slice(&[0u8; 3]); // reserved
        buf.extend_from_slice(&(collection_size as u64).to_le_bytes()[..ss]);
        buf.resize(header_size, 0); // pad header to 8-byte alignment

        // Objects
        for obj in &self.objects {
            let obj_start = buf.len();
            buf.extend_from_slice(&obj.index.to_le_bytes());
            buf.extend_from_slice(&obj.ref_count.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
            buf.extend_from_slice(&(obj.data.len() as u64).to_le_bytes()[..ss]);
            buf.resize(obj_start + objhdr_size, 0); // pad object header
            buf.extend_from_slice(&obj.data);
            buf.resize(buf.len() + (pad_to_8(obj.data.len()) - obj.data.len()), 0);
        }

        // Free-space marker (index = 0) with remaining space
        buf.extend_from_slice(&0u16.to_le_bytes()); // index = 0
        buf.extend_from_slice(&0u16.to_le_bytes()); // ref_count = 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&(free_space as u64).to_le_bytes()[..ss]); // free space size

        // Zero-fill remaining space
        buf.resize(collection_size, 0);

        debug_assert_eq!(buf.len(), collection_size);
        Ok(buf)
    }

    /// Aligned header size, aligned object-header size, and the total the
    /// objects occupy — the three numbers every size calculation here needs.
    ///
    /// libhdf5 (H5HGpkg.h) 8-byte-aligns both the collection header and every
    /// object header (`H5HG_ALIGN`). For `ss == 8` the raw sizes are already
    /// multiples of 8, so the alignment is a no-op there and only matters for
    /// files with 4-byte lengths.
    fn layout(&self, ctx: &FormatContext) -> (usize, usize, usize) {
        let ss = ctx.sizeof_size as usize;
        let header_size = pad_to_8(4 + 1 + 3 + ss); // GCOL + version + reserved + collection_size
        let objhdr_size = pad_to_8(2 + 2 + 4 + ss); // index + ref_count + reserved + size
        let objects_size = self
            .objects
            .iter()
            .map(|obj| objhdr_size + pad_to_8(obj.data.len()))
            .sum();
        (header_size, objhdr_size, objects_size)
    }

    /// Read just the collection's declared total size from its header.
    ///
    /// [`decode`](Self::decode) needs the whole collection in `buf`, and the
    /// size that says how much to read is in the header — so a caller reading
    /// from a file asks here first, with only the header in hand.
    pub fn decode_size(buf: &[u8], ctx: &FormatContext) -> FormatResult<usize> {
        let ss = ctx.sizeof_size as usize;
        let header_size = pad_to_8(4 + 1 + 3 + ss);
        if buf.len() < header_size {
            return Err(FormatError::BufferTooShort {
                needed: header_size,
                available: buf.len(),
            });
        }
        if buf[0..4] != GCOL_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }
        if buf[4] != GCOL_VERSION {
            return Err(FormatError::InvalidVersion(buf[4]));
        }
        Ok(read_size(&buf[8..], ss) as usize)
    }

    /// Decode a global heap collection from a byte buffer.
    ///
    /// Returns the collection and the number of bytes consumed.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        let ss = ctx.sizeof_size as usize;
        let header_size = pad_to_8(4 + 1 + 3 + ss);
        let objhdr_size = pad_to_8(2 + 2 + 4 + ss);

        if buf.len() < header_size {
            return Err(FormatError::BufferTooShort {
                needed: header_size,
                available: buf.len(),
            });
        }

        // Signature
        if buf[0..4] != GCOL_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        // Version
        let version = buf[4];
        if version != GCOL_VERSION {
            return Err(FormatError::InvalidVersion(version));
        }

        // Reserved (bytes 5..8) -- skip

        // Collection size
        let collection_size = read_size(&buf[8..], ss) as usize;

        if buf.len() < collection_size {
            return Err(FormatError::BufferTooShort {
                needed: collection_size,
                available: buf.len(),
            });
        }

        // Parse objects
        let mut pos = header_size;
        let mut objects = Vec::new();

        while pos + objhdr_size <= collection_size {
            let obj_start = pos;
            let index = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let ref_count = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let _reserved =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
            pos += 4;
            let size = read_size(&buf[pos..], ss) as usize;
            // Skip any object-header alignment padding.
            pos = obj_start + objhdr_size;

            if index == 0 {
                // Free-space marker -- end of used objects
                break;
            }

            // `size` is a file field up to 8 bytes wide; use a checked add
            // so a crafted value cannot wrap `pos + size` into a small (or
            // `< pos`) end offset that bypasses the bound check or panics
            // the slice below.
            let obj_end = pos
                .checked_add(size)
                .filter(|&end| end <= collection_size)
                .ok_or_else(|| {
                    FormatError::InvalidData(format!(
                        "global heap object {} extends past collection boundary",
                        index,
                    ))
                })?;

            let data = buf[pos..obj_end].to_vec();
            let padded = pad_to_8(size);
            pos += padded;

            objects.push(GlobalHeapObject {
                index,
                ref_count,
                data,
            });
        }

        Ok((Self { objects }, collection_size))
    }
}

impl Default for GlobalHeapCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a variable-length reference (used in dataset raw data).
///
/// On-disk format per element:
///   sequence_length (u32 LE) + collection_address (sizeof_addr bytes) + object_index (u32 LE).
pub fn encode_vlen_reference(
    sequence_length: u32,
    collection_addr: u64,
    object_index: u32,
    ctx: &FormatContext,
) -> Vec<u8> {
    let sa = ctx.sizeof_addr as usize;
    let mut buf = Vec::with_capacity(4 + sa + 4);
    buf.extend_from_slice(&sequence_length.to_le_bytes());
    buf.extend_from_slice(&collection_addr.to_le_bytes()[..sa]);
    buf.extend_from_slice(&object_index.to_le_bytes());
    buf
}

/// Decode a variable-length reference from dataset raw data.
///
/// Returns `(sequence_length, collection_address, object_index)`.
pub fn decode_vlen_reference(buf: &[u8], ctx: &FormatContext) -> FormatResult<(u32, u64, u32)> {
    let sa = ctx.sizeof_addr as usize;
    let total = 4 + sa + 4;
    if buf.len() < total {
        return Err(FormatError::BufferTooShort {
            needed: total,
            available: buf.len(),
        });
    }
    let seq_len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let addr = read_size(&buf[4..], sa);
    let index = u32::from_le_bytes([
        buf[4 + sa],
        buf[4 + sa + 1],
        buf[4 + sa + 2],
        buf[4 + sa + 3],
    ]);
    Ok((seq_len, addr, index))
}

/// Return the size of a vlen reference in bytes: 4 + sizeof_addr + 4.
pub fn vlen_reference_size(ctx: &FormatContext) -> usize {
    4 + ctx.sizeof_addr as usize + 4
}

/// The `u32` sequence length a vlen reference stores, or an error when the
/// data is longer than the on-disk field can say.
///
/// The single owner of this conversion: a bare `as u32` at a write site
/// silently wraps a 4 GiB sequence to its low 32 bits — the heap object
/// keeps every byte, but each read returns the wrapped length.
pub fn vlen_seq_len(data_len: usize) -> FormatResult<u32> {
    u32::try_from(data_len).map_err(|_| {
        FormatError::InvalidData(format!(
            "a {data_len}-byte vlen sequence does not fit the 32-bit length field"
        ))
    })
}

/// Round `n` up to the next multiple of 8.
fn pad_to_8(n: usize) -> usize {
    (n + 7) & !7
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    fn ctx4() -> FormatContext {
        FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        }
    }

    #[test]
    fn empty_collection_roundtrip() {
        let coll = GlobalHeapCollection::new();
        let encoded = coll.encode(&ctx());
        let (decoded, consumed) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, coll);
        assert!(decoded.objects.is_empty());
    }

    #[test]
    fn single_object_roundtrip() {
        let mut coll = GlobalHeapCollection::new();
        let idx = coll.add_object(b"hello".to_vec()).unwrap();
        assert_eq!(idx, 1);

        let encoded = coll.encode(&ctx());
        let (decoded, consumed) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.objects.len(), 1);
        assert_eq!(decoded.objects[0].index, 1);
        assert_eq!(decoded.objects[0].data, b"hello");
    }

    #[test]
    fn multiple_objects_roundtrip() {
        let mut coll = GlobalHeapCollection::new();
        let i1 = coll.add_object(b"alpha".to_vec()).unwrap();
        let i2 = coll.add_object(b"beta".to_vec()).unwrap();
        let i3 = coll.add_object(b"gamma delta".to_vec()).unwrap();
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
        assert_eq!(i3, 3);

        let encoded = coll.encode(&ctx());
        let (decoded, _) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(decoded.objects.len(), 3);
        assert_eq!(decoded.get_object(1), Some(b"alpha".as_slice()));
        assert_eq!(decoded.get_object(2), Some(b"beta".as_slice()));
        assert_eq!(decoded.get_object(3), Some(b"gamma delta".as_slice()));
    }

    #[test]
    fn get_object_not_found() {
        let coll = GlobalHeapCollection::new();
        assert_eq!(coll.get_object(1), None);
    }

    #[test]
    fn padding_to_8() {
        assert_eq!(pad_to_8(0), 0);
        assert_eq!(pad_to_8(1), 8);
        assert_eq!(pad_to_8(7), 8);
        assert_eq!(pad_to_8(8), 8);
        assert_eq!(pad_to_8(9), 16);
        assert_eq!(pad_to_8(16), 16);
    }

    #[test]
    fn vlen_reference_roundtrip() {
        let c = ctx();
        let encoded = encode_vlen_reference(5, 0x1234_5678_9ABC_DEF0, 42, &c);
        assert_eq!(encoded.len(), vlen_reference_size(&c));
        let (seq_len, addr, idx) = decode_vlen_reference(&encoded, &c).unwrap();
        assert_eq!(seq_len, 5);
        assert_eq!(addr, 0x1234_5678_9ABC_DEF0);
        assert_eq!(idx, 42);
    }

    #[test]
    fn vlen_reference_4byte_roundtrip() {
        let c = ctx4();
        let encoded = encode_vlen_reference(10, 0x1234_5678, 7, &c);
        assert_eq!(encoded.len(), 12); // 4 + 4 + 4
        let (seq_len, addr, idx) = decode_vlen_reference(&encoded, &c).unwrap();
        assert_eq!(seq_len, 10);
        assert_eq!(addr, 0x1234_5678);
        assert_eq!(idx, 7);
    }

    #[test]
    fn vlen_seq_len_bounds() {
        assert_eq!(vlen_seq_len(0).unwrap(), 0);
        assert_eq!(vlen_seq_len(u32::MAX as usize).unwrap(), u32::MAX);
        #[cfg(target_pointer_width = "64")]
        assert!(vlen_seq_len(u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn vlen_reference_size_check() {
        assert_eq!(vlen_reference_size(&ctx()), 16);
        assert_eq!(vlen_reference_size(&ctx4()), 12);
    }

    #[test]
    fn decode_bad_signature() {
        let mut buf = vec![0u8; 32];
        buf[0..4].copy_from_slice(b"XYZW");
        let err = GlobalHeapCollection::decode(&buf, &ctx()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn decode_bad_version() {
        let coll = GlobalHeapCollection::new();
        let mut encoded = coll.encode(&ctx());
        encoded[4] = 99;
        let err = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(99)));
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [0u8; 4];
        let err = GlobalHeapCollection::decode(&buf, &ctx()).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn ctx4_roundtrip() {
        let c = ctx4();
        let mut coll = GlobalHeapCollection::new();
        coll.add_object(b"test data".to_vec()).unwrap();
        let encoded = coll.encode(&c);
        let (decoded, consumed) = GlobalHeapCollection::decode(&encoded, &c).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.get_object(1), Some(b"test data".as_slice()));
    }

    #[test]
    fn object_data_alignment() {
        // Verify that data of odd sizes still roundtrips correctly due to padding
        let mut coll = GlobalHeapCollection::new();
        coll.add_object(vec![1]).unwrap(); // 1 byte -> padded to 8
        coll.add_object(vec![2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap(); // 9 bytes -> padded to 16
        coll.add_object(vec![11, 12, 13, 14, 15, 16, 17, 18])
            .unwrap(); // 8 bytes -> stays 8

        let encoded = coll.encode(&ctx());
        let (decoded, _) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(decoded.get_object(1), Some([1u8].as_slice()));
        assert_eq!(
            decoded.get_object(2),
            Some([2, 3, 4, 5, 6, 7, 8, 9, 10].as_slice())
        );
        assert_eq!(
            decoded.get_object(3),
            Some([11, 12, 13, 14, 15, 16, 17, 18].as_slice())
        );
    }

    #[test]
    fn empty_data_object() {
        let mut coll = GlobalHeapCollection::new();
        coll.add_object(vec![]).unwrap();
        let encoded = coll.encode(&ctx());
        let (decoded, _) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(decoded.get_object(1), Some([].as_slice()));
    }

    /// Removing an object keeps the other indices valid, so the vlen
    /// references that name them do not have to be rewritten.
    #[test]
    fn remove_object_keeps_the_survivors_indices() {
        let mut coll = GlobalHeapCollection::new();
        assert_eq!(coll.add_object(b"first".to_vec()).unwrap(), 1);
        assert_eq!(coll.add_object(b"second".to_vec()).unwrap(), 2);
        assert_eq!(coll.add_object(b"third".to_vec()).unwrap(), 3);

        assert!(coll.remove_object(2));
        // Already gone: two elements of one dataset may name the same object.
        assert!(!coll.remove_object(2));

        let encoded = coll.encode(&ctx());
        let (decoded, _) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(decoded.get_object(1), Some(b"first".as_slice()));
        assert_eq!(decoded.get_object(2), None);
        assert_eq!(decoded.get_object(3), Some(b"third".as_slice()));

        assert!(coll.remove_object(1) && coll.remove_object(3));
        assert!(coll.is_empty());
    }

    /// A collection above the 4096-byte minimum keeps its size when an object
    /// leaves it, the way libhdf5's `H5HG_remove` gives the recovered bytes to
    /// the free-space marker rather than shortening the collection. Re-encoding
    /// at the natural size would declare a smaller collection than the block
    /// the file allocated, so a later free would return less than was taken.
    #[test]
    fn encode_at_size_holds_an_oversized_collection_open() {
        let mut coll = GlobalHeapCollection::new();
        for _ in 0..4 {
            coll.add_object(vec![0xab; 2000]).unwrap();
        }
        let block = coll.encode(&ctx()).len();
        assert!(block > GCOL_MIN_SIZE, "{block} is not above the minimum");

        coll.remove_object(2);
        assert!(
            coll.encoded_size(&ctx()) < block,
            "the test needs the natural size to have shrunk"
        );

        let held = coll.encode_at_size(&ctx(), block).unwrap();
        assert_eq!(held.len(), block);
        assert_eq!(
            GlobalHeapCollection::decode(&held, &ctx()).unwrap().1,
            block,
            "the collection must still declare the whole block"
        );
        let (decoded, _) = GlobalHeapCollection::decode(&held, &ctx()).unwrap();
        assert_eq!(decoded.get_object(3), Some(vec![0xab; 2000].as_slice()));

        // Asking for less than the objects need is refused, not truncated.
        assert!(coll.encode_at_size(&ctx(), 64).is_err());
    }

    /// New objects encode a zero reference count, matching `H5HG_insert`
    /// (`UINT16ENCODE(p, 0)` — only the virtual-dataset layer's `H5HG_link`
    /// ever raises it). It used to be hardcoded to 1.
    #[test]
    fn new_objects_encode_a_zero_ref_count() {
        let mut coll = GlobalHeapCollection::new();
        coll.add_object(b"x".to_vec()).unwrap();
        let img = coll.encode(&ctx());
        let header_size = pad_to_8(4 + 1 + 3 + ctx().sizeof_size as usize);
        // Object header: index (2) then ref_count (2).
        assert_eq!(&img[header_size + 2..header_size + 4], &[0, 0]);
    }

    /// A decoded object keeps the reference count the file declares, and a
    /// removal's rewrite re-encodes it unchanged — resetting it would strip
    /// a foreign file's VDS link count from the surviving objects.
    #[test]
    fn rewrite_preserves_a_foreign_objects_ref_count() {
        let mut coll = GlobalHeapCollection::new();
        coll.add_object(b"doomed".to_vec()).unwrap();
        coll.add_object(b"vds linked".to_vec()).unwrap();
        coll.objects[1].ref_count = 3;
        let img = coll.encode(&ctx());

        let (mut back, _) = GlobalHeapCollection::decode(&img, &ctx()).unwrap();
        assert_eq!(back.objects[1].ref_count, 3, "decode must keep the count");

        assert!(back.remove_object(1));
        let rewritten = back.encode_at_size(&ctx(), img.len()).unwrap();
        let (again, _) = GlobalHeapCollection::decode(&rewritten, &ctx()).unwrap();
        assert_eq!(again.get_object(2), Some(b"vds linked".as_slice()));
        assert_eq!(again.objects[0].ref_count, 3, "rewrite must keep the count");
    }
}
