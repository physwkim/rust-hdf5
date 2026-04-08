//! Symbol Table Node (SNOD) decode (for reading legacy HDF5 files).
//!
//! In v0/v1 groups, child objects are stored in a B-tree that points to
//! symbol table nodes (SNODs). Each SNOD contains an array of symbol
//! table entries, each describing one child object.
//!
//! Layout:
//! ```text
//! "SNOD" (4 bytes)
//! version: 1 byte (1)
//! reserved: 1 byte
//! num_symbols: u16 LE
//! entries: num_symbols * symbol_table_entry
//! ```

use crate::format::superblock::{decode_symbol_table_entry, SymbolTableEntry};
use crate::format::{FormatError, FormatResult};

/// The 4-byte SNOD signature.
pub const SNOD_SIGNATURE: [u8; 4] = *b"SNOD";

/// A decoded symbol table node.
#[derive(Debug, Clone)]
pub struct SymbolTableNode {
    /// The symbol table entries in this node.
    pub entries: Vec<SymbolTableEntry>,
}

impl SymbolTableNode {
    /// Decode a symbol table node from `buf`.
    ///
    /// `sizeof_addr` and `sizeof_size` come from the superblock.
    pub fn decode(buf: &[u8], sizeof_addr: usize, sizeof_size: usize) -> FormatResult<Self> {
        if buf.len() < 8 {
            return Err(FormatError::BufferTooShort {
                needed: 8,
                available: buf.len(),
            });
        }

        if buf[0..4] != SNOD_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let version = buf[4];
        if version != 1 {
            return Err(FormatError::InvalidVersion(version));
        }

        // buf[5] reserved
        let num_symbols = u16::from_le_bytes([buf[6], buf[7]]) as usize;

        // Each entry: sizeof_size + sizeof_addr + 4 + 4 + 16 bytes
        let entry_size = sizeof_size + sizeof_addr + 4 + 4 + 16;
        let needed = 8 + num_symbols * entry_size;
        if buf.len() < needed {
            return Err(FormatError::BufferTooShort {
                needed,
                available: buf.len(),
            });
        }

        let mut pos = 8;
        let mut entries = Vec::with_capacity(num_symbols);

        for _ in 0..num_symbols {
            let entry = decode_symbol_table_entry(buf, &mut pos, sizeof_addr, sizeof_size)?;
            entries.push(entry);
        }

        Ok(SymbolTableNode { entries })
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::UNDEF_ADDR;

    fn build_snod(
        entries: &[(u64, u64, u32, u64, u64)],
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SNOD_SIGNATURE);
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());

        for &(name_offset, obj_header_addr, cache_type, btree_addr, heap_addr) in entries {
            // name_offset
            buf.extend_from_slice(&name_offset.to_le_bytes()[..sizeof_size]);
            // obj_header_addr
            buf.extend_from_slice(&obj_header_addr.to_le_bytes()[..sizeof_addr]);
            // cache_type
            buf.extend_from_slice(&cache_type.to_le_bytes());
            // reserved
            buf.extend_from_slice(&0u32.to_le_bytes());
            // scratch pad (16 bytes)
            if cache_type == 1 {
                buf.extend_from_slice(&btree_addr.to_le_bytes()[..sizeof_addr]);
                buf.extend_from_slice(&heap_addr.to_le_bytes()[..sizeof_addr]);
                let used = 2 * sizeof_addr;
                if used < 16 {
                    buf.extend_from_slice(&vec![0u8; 16 - used]);
                }
            } else {
                buf.extend_from_slice(&[0u8; 16]);
            }
        }

        buf
    }

    #[test]
    fn decode_basic() {
        let snod = build_snod(
            &[
                (8, 0x100, 0, UNDEF_ADDR, UNDEF_ADDR), // dataset
                (16, 0x200, 1, 0x300, 0x400),          // group
            ],
            8,
            8,
        );
        let node = SymbolTableNode::decode(&snod, 8, 8).unwrap();
        assert_eq!(node.entries.len(), 2);
        assert_eq!(node.entries[0].name_offset, 8);
        assert_eq!(node.entries[0].obj_header_addr, 0x100);
        assert_eq!(node.entries[0].cache_type, 0);
        assert_eq!(node.entries[1].cache_type, 1);
        assert_eq!(node.entries[1].btree_addr, 0x300);
        assert_eq!(node.entries[1].heap_addr, 0x400);
    }

    #[test]
    fn decode_empty() {
        let snod = build_snod(&[], 8, 8);
        let node = SymbolTableNode::decode(&snod, 8, 8).unwrap();
        assert!(node.entries.is_empty());
    }

    #[test]
    fn decode_bad_sig() {
        let mut snod = build_snod(&[], 8, 8);
        snod[0] = b'X';
        assert!(matches!(
            SymbolTableNode::decode(&snod, 8, 8).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn decode_bad_version() {
        let mut snod = build_snod(&[], 8, 8);
        snod[4] = 2;
        assert!(matches!(
            SymbolTableNode::decode(&snod, 8, 8).unwrap_err(),
            FormatError::InvalidVersion(2)
        ));
    }

    #[test]
    fn decode_too_short() {
        assert!(matches!(
            SymbolTableNode::decode(&[0u8; 4], 8, 8).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    #[test]
    fn decode_4byte() {
        let snod = build_snod(&[(4, 0x80, 0, UNDEF_ADDR, UNDEF_ADDR)], 4, 4);
        let node = SymbolTableNode::decode(&snod, 4, 4).unwrap();
        assert_eq!(node.entries.len(), 1);
        assert_eq!(node.entries[0].name_offset, 4);
        assert_eq!(node.entries[0].obj_header_addr, 0x80);
    }
}
