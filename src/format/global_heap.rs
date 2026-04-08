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
//! collection_address  (sizeof_addr bytes LE, address of the GCOL)
//! object_index        (u32 LE, index within the collection)
//! ```
//! Total vlen reference size = sizeof_addr + 4 bytes.

use crate::format::{FormatContext, FormatError, FormatResult};

/// Signature for a global heap collection.
const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// Global heap collection version.
const GCOL_VERSION: u8 = 1;

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
    pub fn add_object(&mut self, data: Vec<u8>) -> u16 {
        let index = if self.objects.is_empty() {
            1
        } else {
            self.objects.iter().map(|o| o.index).max().unwrap_or(0) + 1
        };
        self.objects.push(GlobalHeapObject { index, data });
        index
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
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let ss = ctx.sizeof_size as usize;

        // Compute body size: sum of all object encodings + free-space marker
        // Each object: 2 (index) + 2 (ref_count) + 4 (reserved) + ss (size) + padded_data
        let header_size = 4 + 1 + 3 + ss; // GCOL + version + reserved + collection_size
        let mut objects_size: usize = 0;
        for obj in &self.objects {
            let padded = pad_to_8(obj.data.len());
            objects_size += 2 + 2 + 4 + ss + padded;
        }
        // Free-space marker: index(2) + ref_count(2) + reserved(4) + size(ss) = 8 + ss
        let free_marker_size = 2 + 2 + 4 + ss;
        let collection_size = header_size + objects_size + free_marker_size;

        let mut buf = Vec::with_capacity(collection_size);

        // Header
        buf.extend_from_slice(&GCOL_SIGNATURE);
        buf.push(GCOL_VERSION);
        buf.extend_from_slice(&[0u8; 3]); // reserved
        buf.extend_from_slice(&(collection_size as u64).to_le_bytes()[..ss]);

        // Objects
        for obj in &self.objects {
            buf.extend_from_slice(&obj.index.to_le_bytes());
            buf.extend_from_slice(&1u16.to_le_bytes()); // ref_count = 1
            buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
            buf.extend_from_slice(&(obj.data.len() as u64).to_le_bytes()[..ss]);
            buf.extend_from_slice(&obj.data);
            // Pad to 8-byte alignment
            let pad = pad_to_8(obj.data.len()) - obj.data.len();
            if pad > 0 {
                buf.extend_from_slice(&vec![0u8; pad]);
            }
        }

        // Free-space marker (index = 0)
        buf.extend_from_slice(&0u16.to_le_bytes()); // index = 0
        buf.extend_from_slice(&0u16.to_le_bytes()); // ref_count = 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]); // size = 0

        debug_assert_eq!(buf.len(), collection_size);
        buf
    }

    /// Decode a global heap collection from a byte buffer.
    ///
    /// Returns the collection and the number of bytes consumed.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        let ss = ctx.sizeof_size as usize;
        let header_size = 4 + 1 + 3 + ss;

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

        while pos + 2 + 2 + 4 + ss <= collection_size {
            let index = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let _ref_count = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let _reserved =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
            pos += 4;
            let size = read_size(&buf[pos..], ss) as usize;
            pos += ss;

            if index == 0 {
                // Free-space marker -- end of used objects
                break;
            }

            if pos + size > collection_size {
                return Err(FormatError::InvalidData(format!(
                    "global heap object {} extends past collection boundary",
                    index,
                )));
            }

            let data = buf[pos..pos + size].to_vec();
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
/// A vlen reference is: collection_address (sizeof_addr bytes) + object_index (u32).
pub fn encode_vlen_reference(
    collection_addr: u64,
    object_index: u32,
    ctx: &FormatContext,
) -> Vec<u8> {
    let sa = ctx.sizeof_addr as usize;
    let mut buf = Vec::with_capacity(sa + 4);
    buf.extend_from_slice(&collection_addr.to_le_bytes()[..sa]);
    buf.extend_from_slice(&object_index.to_le_bytes());
    buf
}

/// Decode a variable-length reference from dataset raw data.
///
/// Returns `(collection_address, object_index)`.
pub fn decode_vlen_reference(buf: &[u8], ctx: &FormatContext) -> FormatResult<(u64, u32)> {
    let sa = ctx.sizeof_addr as usize;
    if buf.len() < sa + 4 {
        return Err(FormatError::BufferTooShort {
            needed: sa + 4,
            available: buf.len(),
        });
    }
    let addr = read_size(buf, sa);
    let index = u32::from_le_bytes([buf[sa], buf[sa + 1], buf[sa + 2], buf[sa + 3]]);
    Ok((addr, index))
}

/// Return the size of a vlen reference in bytes: sizeof_addr + 4.
pub fn vlen_reference_size(ctx: &FormatContext) -> usize {
    ctx.sizeof_addr as usize + 4
}

/// Round `n` up to the next multiple of 8.
fn pad_to_8(n: usize) -> usize {
    (n + 7) & !7
}

/// Read a little-endian unsigned integer of `n` bytes (1..=8) into a `u64`.
fn read_size(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
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
        let idx = coll.add_object(b"hello".to_vec());
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
        let i1 = coll.add_object(b"alpha".to_vec());
        let i2 = coll.add_object(b"beta".to_vec());
        let i3 = coll.add_object(b"gamma delta".to_vec());
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
        let encoded = encode_vlen_reference(0x1234_5678_9ABC_DEF0, 42, &c);
        assert_eq!(encoded.len(), vlen_reference_size(&c));
        let (addr, idx) = decode_vlen_reference(&encoded, &c).unwrap();
        assert_eq!(addr, 0x1234_5678_9ABC_DEF0);
        assert_eq!(idx, 42);
    }

    #[test]
    fn vlen_reference_4byte_roundtrip() {
        let c = ctx4();
        let encoded = encode_vlen_reference(0x1234_5678, 7, &c);
        assert_eq!(encoded.len(), 8); // 4 + 4
        let (addr, idx) = decode_vlen_reference(&encoded, &c).unwrap();
        assert_eq!(addr, 0x1234_5678);
        assert_eq!(idx, 7);
    }

    #[test]
    fn vlen_reference_size_check() {
        assert_eq!(vlen_reference_size(&ctx()), 12);
        assert_eq!(vlen_reference_size(&ctx4()), 8);
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
        coll.add_object(b"test data".to_vec());
        let encoded = coll.encode(&c);
        let (decoded, consumed) = GlobalHeapCollection::decode(&encoded, &c).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.get_object(1), Some(b"test data".as_slice()));
    }

    #[test]
    fn object_data_alignment() {
        // Verify that data of odd sizes still roundtrips correctly due to padding
        let mut coll = GlobalHeapCollection::new();
        coll.add_object(vec![1]); // 1 byte -> padded to 8
        coll.add_object(vec![2, 3, 4, 5, 6, 7, 8, 9, 10]); // 9 bytes -> padded to 16
        coll.add_object(vec![11, 12, 13, 14, 15, 16, 17, 18]); // 8 bytes -> stays 8

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
        coll.add_object(vec![]);
        let encoded = coll.encode(&ctx());
        let (decoded, _) = GlobalHeapCollection::decode(&encoded, &ctx()).unwrap();
        assert_eq!(decoded.get_object(1), Some([].as_slice()));
    }
}
