/// Superblock encode/decode for HDF5 files.
///
/// The superblock is always at offset 0 (or at a user-hint offset) and
/// contains the file-level metadata: version, size parameters, and addresses
/// of the root group and end-of-file.
///
/// This module supports:
/// - v2/v3 superblocks (encode + decode)
/// - v0/v1 superblocks (decode only, for reading legacy files)
use crate::format::checksum::checksum_metadata;
use crate::format::{FormatError, FormatResult, UNDEF_ADDR};

/// The 8-byte HDF5 file signature that begins every superblock.
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Superblock version 2.
pub const SUPERBLOCK_V2: u8 = 2;

/// Superblock version 3 (adds SWMR support).
pub const SUPERBLOCK_V3: u8 = 3;

/// File consistency flag: file was opened for write access.
pub const FLAG_WRITE_ACCESS: u8 = 0x01;

/// File consistency flag: file is consistent / was properly closed.
pub const FLAG_FILE_OK: u8 = 0x02;

/// File consistency flag: file was opened for single-writer/multi-reader.
pub const FLAG_SWMR_WRITE: u8 = 0x04;

/// Superblock v2/v3 structure.
///
/// Layout (O = sizeof_offsets):
/// ```text
/// [0..8]              Signature (8 bytes)
/// [8]                 Version (1 byte)
/// [9]                 Size of Offsets (1 byte)
/// [10]                Size of Lengths (1 byte)
/// [11]                File Consistency Flags (1 byte)
/// [12..12+O]          Base Address (O bytes)
/// [12+O..12+2O]       Superblock Extension Address (O bytes)
/// [12+2O..12+3O]      End of File Address (O bytes)
/// [12+3O..12+4O]      Root Group Object Header Address (O bytes)
/// [12+4O..12+4O+4]    Checksum (4 bytes)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockV2V3 {
    /// Superblock version: 2 or 3.
    pub version: u8,
    /// Size of file offsets in bytes (typically 8).
    pub sizeof_offsets: u8,
    /// Size of file lengths in bytes (typically 8).
    pub sizeof_lengths: u8,
    /// File consistency flags (see `FLAG_*` constants).
    pub file_consistency_flags: u8,
    /// Base address of the file (usually 0).
    pub base_address: u64,
    /// Address of the superblock extension object header, or UNDEF.
    pub superblock_extension_address: u64,
    /// End-of-file address.
    pub end_of_file_address: u64,
    /// Address of the root group object header.
    pub root_group_object_header_address: u64,
}

impl SuperblockV2V3 {
    /// Returns the total encoded size in bytes: 12 + 4*O + 4 (checksum).
    pub fn encoded_size(&self) -> usize {
        12 + 4 * (self.sizeof_offsets as usize) + 4
    }

    /// Encode the superblock to a byte vector, including the trailing checksum.
    pub fn encode(&self) -> Vec<u8> {
        let size = self.encoded_size();
        let mut buf = Vec::with_capacity(size);

        // Signature
        buf.extend_from_slice(&HDF5_SIGNATURE);
        // Version
        buf.push(self.version);
        // Size of Offsets
        buf.push(self.sizeof_offsets);
        // Size of Lengths
        buf.push(self.sizeof_lengths);
        // File Consistency Flags
        buf.push(self.file_consistency_flags);

        // Addresses -- encode as little-endian with sizeof_offsets bytes
        let o = self.sizeof_offsets as usize;
        encode_offset(&mut buf, self.base_address, o);
        encode_offset(&mut buf, self.superblock_extension_address, o);
        encode_offset(&mut buf, self.end_of_file_address, o);
        encode_offset(&mut buf, self.root_group_object_header_address, o);

        // Checksum over everything before the checksum field
        debug_assert_eq!(buf.len(), size - 4);
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());

        debug_assert_eq!(buf.len(), size);
        buf
    }

    /// Decode a superblock from a byte buffer. Verifies the signature, version,
    /// and checksum. Returns the parsed superblock.
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        // Minimum size check: we need at least the fixed 12-byte header to
        // read sizeof_offsets before computing the full size.
        if buf.len() < 12 {
            return Err(FormatError::BufferTooShort {
                needed: 12,
                available: buf.len(),
            });
        }

        // Signature
        if buf[0..8] != HDF5_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        // Version
        let version = buf[8];
        if version != SUPERBLOCK_V2 && version != SUPERBLOCK_V3 {
            return Err(FormatError::InvalidVersion(version));
        }

        let sizeof_offsets = buf[9];
        let sizeof_lengths = buf[10];
        let file_consistency_flags = buf[11];

        let o = sizeof_offsets as usize;
        let total_size = 12 + 4 * o + 4;
        if buf.len() < total_size {
            return Err(FormatError::BufferTooShort {
                needed: total_size,
                available: buf.len(),
            });
        }

        // Verify checksum
        let data_end = total_size - 4;
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

        // Decode addresses
        let mut pos = 12;
        let base_address = decode_offset(buf, &mut pos, o);
        let superblock_extension_address = decode_offset(buf, &mut pos, o);
        let end_of_file_address = decode_offset(buf, &mut pos, o);
        let root_group_object_header_address = decode_offset(buf, &mut pos, o);

        Ok(SuperblockV2V3 {
            version,
            sizeof_offsets,
            sizeof_lengths,
            file_consistency_flags,
            base_address,
            superblock_extension_address,
            end_of_file_address,
            root_group_object_header_address,
        })
    }
}

/// Encode a u64 address as `size` little-endian bytes and append to `buf`.
fn encode_offset(buf: &mut Vec<u8>, value: u64, size: usize) {
    let bytes = value.to_le_bytes();
    buf.extend_from_slice(&bytes[..size]);
}

/// Decode a little-endian address of `size` bytes from `buf` at `*pos`,
/// advancing `*pos` past the consumed bytes.
fn decode_offset(buf: &[u8], pos: &mut usize, size: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..size].copy_from_slice(&buf[*pos..*pos + size]);
    *pos += size;
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::UNDEF_ADDR;

    #[test]
    fn test_encoded_size() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        // 12 + 4*8 + 4 = 48
        assert_eq!(sb.encoded_size(), 48);
    }

    #[test]
    fn test_roundtrip_v3_offset8() {
        let original = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: FLAG_FILE_OK,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 0x1_0000,
            root_group_object_header_address: 48,
        };

        let encoded = original.encode();
        assert_eq!(encoded.len(), original.encoded_size());

        // Verify signature
        assert_eq!(&encoded[..8], &HDF5_SIGNATURE);

        let decoded = SuperblockV2V3::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_v2_offset4() {
        let original = SuperblockV2V3 {
            version: SUPERBLOCK_V2,
            sizeof_offsets: 4,
            sizeof_lengths: 4,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: 0xFFFF_FFFF,
            end_of_file_address: 8192,
            root_group_object_header_address: 28,
        };

        let encoded = original.encode();
        // 12 + 4*4 + 4 = 32
        assert_eq!(encoded.len(), 32);

        let decoded = SuperblockV2V3::decode(&encoded).expect("decode failed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_bad_signature() {
        let mut data = vec![0u8; 48];
        // Wrong signature
        data[0] = 0x00;
        let err = SuperblockV2V3::decode(&data).unwrap_err();
        assert!(matches!(err, FormatError::InvalidSignature));
    }

    #[test]
    fn test_decode_bad_version() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let mut encoded = sb.encode();
        // Corrupt version to 1
        encoded[8] = 1;
        let err = SuperblockV2V3::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(1)));
    }

    #[test]
    fn test_decode_checksum_mismatch() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let mut encoded = sb.encode();
        // Corrupt a data byte
        encoded[12] = 0xFF;
        let err = SuperblockV2V3::decode(&encoded).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let err = SuperblockV2V3::decode(&[0u8; 4]).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn test_flags() {
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let encoded = sb.encode();
        let decoded = SuperblockV2V3::decode(&encoded).unwrap();
        assert_eq!(
            decoded.file_consistency_flags,
            FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE
        );
    }

    #[test]
    fn test_roundtrip_with_extra_trailing_data() {
        // decode should succeed even if the buffer is longer than needed
        let sb = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let mut encoded = sb.encode();
        encoded.extend_from_slice(&[0xAA; 100]); // trailing garbage
        let decoded = SuperblockV2V3::decode(&encoded).unwrap();
        assert_eq!(decoded, sb);
    }
}

// =========================================================================
// Superblock v0/v1 — decode only (for reading legacy HDF5 files)
// =========================================================================

/// Symbol table entry, as stored in the root group's superblock (v0/v1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolTableEntry {
    /// Offset of the name in the local heap.
    pub name_offset: u64,
    /// Address of the object header.
    pub obj_header_addr: u64,
    /// Cache type: 0 = nothing cached, 1 = symbol table (group).
    pub cache_type: u32,
    /// If cache_type == 1: B-tree address for group children.
    pub btree_addr: u64,
    /// If cache_type == 1: local heap address for group names.
    pub heap_addr: u64,
}

/// Superblock v0/v1 structure (decode only).
///
/// Layout after 8-byte signature:
/// ```text
/// Byte 0: superblock version (0 or 1)
/// Byte 1: free-space version (0)
/// Byte 2: root group STE version (0)
/// Byte 3: reserved (0)
/// Byte 4: shared header version (0)
/// Byte 5: sizeof_addr
/// Byte 6: sizeof_size
/// Byte 7: reserved (0)
/// Bytes 8-9: sym_leaf_k (u16 LE)
/// Bytes 10-11: btree_internal_k (u16 LE)
/// Bytes 12-15: file_consistency_flags (u32 LE)
/// [v1 only: bytes 16-17: indexed_storage_k (u16 LE), bytes 18-19: reserved]
/// Then: base_addr(O), extension_addr(O), eof_addr(O), driver_addr(O)
/// Then: root group symbol table entry
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockV0V1 {
    pub version: u8,
    pub sizeof_offsets: u8,
    pub sizeof_lengths: u8,
    pub file_consistency_flags: u32,
    pub sym_leaf_k: u16,
    pub btree_internal_k: u16,
    pub indexed_storage_k: Option<u16>,
    pub base_address: u64,
    pub superblock_extension_address: u64,
    pub end_of_file_address: u64,
    pub driver_info_address: u64,
    pub root_symbol_table_entry: SymbolTableEntry,
}

impl SuperblockV0V1 {
    /// Decode a v0/v1 superblock from `buf`. The buffer must start at the
    /// 8-byte HDF5 signature. Returns the parsed superblock.
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        // Minimum: 8 (sig) + 8 (fixed header before addresses) = 16
        if buf.len() < 16 {
            return Err(FormatError::BufferTooShort {
                needed: 16,
                available: buf.len(),
            });
        }

        // Signature
        if buf[0..8] != HDF5_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[8];
        if version != 0 && version != 1 {
            return Err(FormatError::InvalidVersion(version));
        }

        // buf[9] = free-space version (must be 0)
        // buf[10] = root group STE version (must be 0)
        // buf[11] = reserved
        // buf[12] = shared header version (must be 0)
        let sizeof_offsets = buf[13];
        let sizeof_lengths = buf[14];
        // buf[15] = reserved

        let o = sizeof_offsets as usize;
        let mut pos = 16;

        // Check we have enough for the remaining fixed fields
        if buf.len() < pos + 4 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 4,
                available: buf.len(),
            });
        }

        let sym_leaf_k = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let btree_internal_k = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;

        if buf.len() < pos + 4 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 4,
                available: buf.len(),
            });
        }
        let file_consistency_flags =
            u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        pos += 4;

        let indexed_storage_k = if version == 1 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let k = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            // 2 bytes reserved
            pos += 2;
            Some(k)
        } else {
            None
        };

        // 4 addresses, each sizeof_offsets bytes
        let needed = pos + 4 * o;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }

        let base_address = decode_offset(buf, &mut pos, o);
        let superblock_extension_address = decode_offset(buf, &mut pos, o);
        let end_of_file_address = decode_offset(buf, &mut pos, o);
        let driver_info_address = decode_offset(buf, &mut pos, o);

        // Root group symbol table entry
        let ste = decode_symbol_table_entry(buf, &mut pos, o, sizeof_lengths as usize)?;

        Ok(SuperblockV0V1 {
            version,
            sizeof_offsets,
            sizeof_lengths,
            file_consistency_flags,
            sym_leaf_k,
            btree_internal_k,
            indexed_storage_k,
            base_address,
            superblock_extension_address,
            end_of_file_address,
            driver_info_address,
            root_symbol_table_entry: ste,
        })
    }
}

/// Decode a symbol table entry from buf at the given position.
pub fn decode_symbol_table_entry(
    buf: &[u8],
    pos: &mut usize,
    sizeof_addr: usize,
    sizeof_size: usize,
) -> FormatResult<SymbolTableEntry> {
    let needed = *pos + sizeof_size + sizeof_addr + 4 + 4 + 16;
    if buf.len() < needed {
        return Err(FormatError::BufferTooShort {
            needed,
            available: buf.len(),
        });
    }

    let name_offset = decode_offset(buf, pos, sizeof_size);
    let obj_header_addr = decode_offset(buf, pos, sizeof_addr);
    let cache_type = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos += 4;
    // reserved u32
    *pos += 4;

    // Scratch pad: 16 bytes
    let (btree_addr, heap_addr) = if cache_type == 1 {
        let btree = decode_offset(buf, pos, sizeof_addr);
        let heap = decode_offset(buf, pos, sizeof_addr);
        // Skip remaining scratch pad bytes
        let used = 2 * sizeof_addr;
        if used < 16 {
            *pos += 16 - used;
        }
        (btree, heap)
    } else {
        *pos += 16;
        (UNDEF_ADDR, UNDEF_ADDR)
    };

    Ok(SymbolTableEntry {
        name_offset,
        obj_header_addr,
        cache_type,
        btree_addr,
        heap_addr,
    })
}

/// Detect the superblock version from the first 9+ bytes of a file.
/// Returns the version byte (0, 1, 2, or 3).
pub fn detect_superblock_version(buf: &[u8]) -> FormatResult<u8> {
    if buf.len() < 9 {
        return Err(FormatError::BufferTooShort {
            needed: 9,
            available: buf.len(),
        });
    }
    if buf[0..8] != HDF5_SIGNATURE {
        return Err(FormatError::InvalidSignature);
    }
    Ok(buf[8])
}

#[cfg(test)]
mod tests_v0v1 {
    use super::*;

    /// Build a minimal v0 superblock for testing.
    fn build_v0_superblock(
        root_obj_header_addr: u64,
        btree_addr: u64,
        heap_addr: u64,
        eof: u64,
    ) -> Vec<u8> {
        let sizeof_addr: usize = 8;
        let sizeof_size: usize = 8;
        let mut buf = Vec::new();

        // Signature (8 bytes)
        buf.extend_from_slice(&HDF5_SIGNATURE);
        // Version 0
        buf.push(0);
        // Free-space version
        buf.push(0);
        // Root group STE version
        buf.push(0);
        // Reserved
        buf.push(0);
        // Shared header version
        buf.push(0);
        // sizeof_addr
        buf.push(sizeof_addr as u8);
        // sizeof_size
        buf.push(sizeof_size as u8);
        // Reserved
        buf.push(0);

        // sym_leaf_k = 4
        buf.extend_from_slice(&4u16.to_le_bytes());
        // btree_internal_k = 32
        buf.extend_from_slice(&32u16.to_le_bytes());
        // file_consistency_flags = 0
        buf.extend_from_slice(&0u32.to_le_bytes());

        // base_addr = 0
        buf.extend_from_slice(&0u64.to_le_bytes()[..sizeof_addr]);
        // extension_addr = UNDEF
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);
        // eof_addr
        buf.extend_from_slice(&eof.to_le_bytes()[..sizeof_addr]);
        // driver_info_addr = UNDEF
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);

        // Root group symbol table entry:
        // name_offset (sizeof_size)
        buf.extend_from_slice(&0u64.to_le_bytes()[..sizeof_size]);
        // obj_header_addr (sizeof_addr)
        buf.extend_from_slice(&root_obj_header_addr.to_le_bytes()[..sizeof_addr]);
        // cache_type = 1 (stab)
        buf.extend_from_slice(&1u32.to_le_bytes());
        // reserved
        buf.extend_from_slice(&0u32.to_le_bytes());
        // scratch pad: btree_addr + heap_addr
        buf.extend_from_slice(&btree_addr.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&heap_addr.to_le_bytes()[..sizeof_addr]);

        buf
    }

    #[test]
    fn test_decode_v0() {
        let buf = build_v0_superblock(0x100, 0x200, 0x300, 0x1000);
        let sb = SuperblockV0V1::decode(&buf).expect("decode failed");
        assert_eq!(sb.version, 0);
        assert_eq!(sb.sizeof_offsets, 8);
        assert_eq!(sb.sizeof_lengths, 8);
        assert_eq!(sb.sym_leaf_k, 4);
        assert_eq!(sb.btree_internal_k, 32);
        assert_eq!(sb.file_consistency_flags, 0);
        assert_eq!(sb.indexed_storage_k, None);
        assert_eq!(sb.base_address, 0);
        assert_eq!(sb.end_of_file_address, 0x1000);
        assert_eq!(sb.root_symbol_table_entry.obj_header_addr, 0x100);
        assert_eq!(sb.root_symbol_table_entry.cache_type, 1);
        assert_eq!(sb.root_symbol_table_entry.btree_addr, 0x200);
        assert_eq!(sb.root_symbol_table_entry.heap_addr, 0x300);
    }

    #[test]
    fn test_decode_v1() {
        // Build a v1 superblock (includes indexed_storage_k)
        let sizeof_addr: usize = 8;
        let sizeof_size: usize = 8;
        let mut buf = Vec::new();
        buf.extend_from_slice(&HDF5_SIGNATURE);
        buf.push(1); // version 1
        buf.push(0);
        buf.push(0);
        buf.push(0);
        buf.push(0);
        buf.push(sizeof_addr as u8);
        buf.push(sizeof_size as u8);
        buf.push(0);
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&32u16.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        // indexed_storage_k = 16
        buf.extend_from_slice(&16u16.to_le_bytes());
        // reserved
        buf.extend_from_slice(&0u16.to_le_bytes());
        // addresses
        buf.extend_from_slice(&0u64.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&0x2000u64.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);
        // STE
        buf.extend_from_slice(&0u64.to_le_bytes()[..sizeof_size]);
        buf.extend_from_slice(&0x100u64.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0x200u64.to_le_bytes()[..sizeof_addr]);
        buf.extend_from_slice(&0x300u64.to_le_bytes()[..sizeof_addr]);

        let sb = SuperblockV0V1::decode(&buf).expect("decode failed");
        assert_eq!(sb.version, 1);
        assert_eq!(sb.indexed_storage_k, Some(16));
        assert_eq!(sb.root_symbol_table_entry.btree_addr, 0x200);
    }

    #[test]
    fn test_detect_version() {
        let v0 = build_v0_superblock(0x100, 0x200, 0x300, 0x1000);
        assert_eq!(detect_superblock_version(&v0).unwrap(), 0);

        let sb_v3 = SuperblockV2V3 {
            version: SUPERBLOCK_V3,
            sizeof_offsets: 8,
            sizeof_lengths: 8,
            file_consistency_flags: 0,
            base_address: 0,
            superblock_extension_address: UNDEF_ADDR,
            end_of_file_address: 4096,
            root_group_object_header_address: 48,
        };
        let v3 = sb_v3.encode();
        assert_eq!(detect_superblock_version(&v3).unwrap(), 3);
    }

    #[test]
    fn test_bad_sig() {
        let mut buf = build_v0_superblock(0x100, 0x200, 0x300, 0x1000);
        buf[0] = 0;
        assert!(matches!(
            SuperblockV0V1::decode(&buf).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn test_bad_version() {
        let mut buf = build_v0_superblock(0x100, 0x200, 0x300, 0x1000);
        buf[8] = 5; // invalid version
        assert!(matches!(
            SuperblockV0V1::decode(&buf).unwrap_err(),
            FormatError::InvalidVersion(5)
        ));
    }
}
