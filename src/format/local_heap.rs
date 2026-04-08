//! Local heap decode (for reading legacy HDF5 files).
//!
//! The local heap is used by v0/v1 groups to store link names as
//! null-terminated strings. The heap header lives at a known address and
//! points to a contiguous data block.
//!
//! Header layout:
//! ```text
//! "HEAP" (4 bytes)
//! version: 1 byte (0)
//! reserved: 3 bytes
//! data_size: sizeof_size bytes LE
//! free_list_offset: sizeof_size bytes LE (0xFFFFFFFFFFFFFFFF = none)
//! data_addr: sizeof_addr bytes LE
//! ```

use crate::format::{FormatError, FormatResult};

/// The 4-byte local heap signature.
pub const LOCAL_HEAP_SIGNATURE: [u8; 4] = *b"HEAP";

/// Decoded local heap header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalHeapHeader {
    /// Total size of the data segment.
    pub data_size: u64,
    /// Offset into the data segment of the first free block, or u64::MAX if none.
    pub free_list_offset: u64,
    /// File address of the data segment.
    pub data_addr: u64,
}

impl LocalHeapHeader {
    /// Decode a local heap header from `buf`.
    ///
    /// `sizeof_addr` and `sizeof_size` come from the superblock.
    pub fn decode(buf: &[u8], sizeof_addr: usize, sizeof_size: usize) -> FormatResult<Self> {
        let min_size = 4 + 1 + 3 + sizeof_size * 2 + sizeof_addr;
        if buf.len() < min_size {
            return Err(FormatError::BufferTooShort {
                needed: min_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != LOCAL_HEAP_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != 0 {
            return Err(FormatError::InvalidVersion(version));
        }

        // buf[5..8] reserved
        let mut pos = 8;

        let data_size = read_uint(&buf[pos..], sizeof_size);
        pos += sizeof_size;

        let free_list_offset = read_uint(&buf[pos..], sizeof_size);
        pos += sizeof_size;

        let data_addr = read_uint(&buf[pos..], sizeof_addr);

        Ok(LocalHeapHeader {
            data_size,
            free_list_offset,
            data_addr,
        })
    }
}

/// Look up a null-terminated string in the heap data block by offset.
///
/// `heap_data` is the raw bytes of the local heap data segment.
/// `offset` is the byte offset into that segment where the string starts.
pub fn local_heap_get_string(heap_data: &[u8], offset: u64) -> FormatResult<String> {
    let start = offset as usize;
    if start >= heap_data.len() {
        return Err(FormatError::InvalidData(format!(
            "local heap offset {} out of range (heap size {})",
            offset,
            heap_data.len()
        )));
    }

    // Find the null terminator
    let end = heap_data[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|p| start + p)
        .unwrap_or(heap_data.len());

    String::from_utf8(heap_data[start..end].to_vec())
        .map_err(|e| FormatError::InvalidData(format!("invalid UTF-8 in local heap string: {}", e)))
}

/// Read a little-endian unsigned integer of `n` bytes into a u64.
fn read_uint(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn build_heap_header(
        data_size: u64,
        free_list_offset: u64,
        data_addr: u64,
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&LOCAL_HEAP_SIGNATURE);
        buf.push(0); // version
        buf.extend_from_slice(&[0u8; 3]); // reserved
        buf.extend_from_slice(&data_size.to_le_bytes()[..sizeof_size]);
        buf.extend_from_slice(&free_list_offset.to_le_bytes()[..sizeof_size]);
        buf.extend_from_slice(&data_addr.to_le_bytes()[..sizeof_addr]);
        buf
    }

    #[test]
    fn decode_basic() {
        let buf = build_heap_header(128, u64::MAX, 0x1000, 8, 8);
        let hdr = LocalHeapHeader::decode(&buf, 8, 8).unwrap();
        assert_eq!(hdr.data_size, 128);
        assert_eq!(hdr.free_list_offset, u64::MAX);
        assert_eq!(hdr.data_addr, 0x1000);
    }

    #[test]
    fn decode_4byte() {
        let buf = build_heap_header(64, 0xFFFFFFFF, 0x800, 4, 4);
        let hdr = LocalHeapHeader::decode(&buf, 4, 4).unwrap();
        assert_eq!(hdr.data_size, 64);
        assert_eq!(hdr.free_list_offset, 0xFFFFFFFF);
        assert_eq!(hdr.data_addr, 0x800);
    }

    #[test]
    fn decode_bad_sig() {
        let mut buf = build_heap_header(64, 0, 0x800, 8, 8);
        buf[0] = b'X';
        assert!(matches!(
            LocalHeapHeader::decode(&buf, 8, 8).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn decode_bad_version() {
        let mut buf = build_heap_header(64, 0, 0x800, 8, 8);
        buf[4] = 1;
        assert!(matches!(
            LocalHeapHeader::decode(&buf, 8, 8).unwrap_err(),
            FormatError::InvalidVersion(1)
        ));
    }

    #[test]
    fn decode_too_short() {
        let buf = [0u8; 4];
        assert!(matches!(
            LocalHeapHeader::decode(&buf, 8, 8).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    #[test]
    fn get_string_basic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"\0"); // offset 0: empty
        data.extend_from_slice(b"hello\0");
        data.extend_from_slice(b"world\0");

        assert_eq!(local_heap_get_string(&data, 0).unwrap(), "");
        assert_eq!(local_heap_get_string(&data, 1).unwrap(), "hello");
        assert_eq!(local_heap_get_string(&data, 7).unwrap(), "world");
    }

    #[test]
    fn get_string_out_of_range() {
        let data = b"hello\0";
        assert!(matches!(
            local_heap_get_string(data, 100).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }
}
