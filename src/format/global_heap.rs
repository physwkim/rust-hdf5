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

/// A single object within a global heap collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalHeapObject {
    /// Object index (1-based). Index 0 is reserved for the free-space marker.
    pub index: u16,
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
        self.objects.push(GlobalHeapObject { index, data });
        Ok(index)
    }

    /// Retrieve the data for an object by its 1-based index.
    pub fn get_object(&self, index: u16) -> Option<&[u8]> {
        self.objects
            .iter()
            .find(|o| o.index == index)
            .map(|o| o.data.as_slice())
    }

    /// Encode the collection into a byte vector.
    ///
    /// The encoded blob includes the GCOL header and all heap objects,
    /// followed by a free-space marker (index=0 object).
    /// The total size is padded to at least 4096 bytes (H5HG_MINALLOC)
    /// for compatibility with the HDF5 C library.
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let ss = ctx.sizeof_size as usize;

        // libhdf5 (H5HGpkg.h) 8-byte-aligns both the collection header and
        // every object header (H5HG_ALIGN). For ss == 8 the raw sizes are
        // already multiples of 8, so this is a no-op there and only matters
        // for files with 4-byte lengths.
        let header_size = pad_to_8(4 + 1 + 3 + ss); // GCOL + version + reserved + collection_size
        let objhdr_size = pad_to_8(2 + 2 + 4 + ss); // index + ref_count + reserved + size
        let mut objects_size: usize = 0;
        for obj in &self.objects {
            objects_size += objhdr_size + pad_to_8(obj.data.len());
        }
        // Free-space marker carries the same (aligned) object header.
        let free_marker_size = objhdr_size;
        let content_size = header_size + objects_size + free_marker_size;

        // HDF5 C library requires collection_size >= 4096 (H5HG_MINALLOC)
        let collection_size = content_size.max(GCOL_MIN_SIZE);
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
            buf.extend_from_slice(&1u16.to_le_bytes()); // ref_count = 1
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
        buf
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
            let _ref_count = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
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

            objects.push(GlobalHeapObject { index, data });
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
}
