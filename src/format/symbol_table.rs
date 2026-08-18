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
    /// Encode this node into a block of exactly `node_size` bytes.
    ///
    /// A SNOD is a fixed-size record — `H5G_NODE_SIZE(f)`, which
    /// [`BTreeV1Config::symbol_table_node_size`] computes — whatever fraction
    /// of it the entries fill, because the B-tree above it allocates every
    /// leaf the same width. The unused tail is zeroed rather than left as it
    /// was: `H5G__node_create` calloc's the entry array, so a node libhdf5
    /// wrote never carries a stale entry past `num_symbols`.
    ///
    /// [`BTreeV1Config::symbol_table_node_size`]: crate::format::btree_v1::BTreeV1Config::symbol_table_node_size
    pub fn encode(
        &self,
        node_size: usize,
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> FormatResult<Vec<u8>> {
        let entry_size =
            crate::format::superblock::symbol_table_entry_size(sizeof_addr, sizeof_size);
        let needed = 8 + self.entries.len() * entry_size;
        if needed > node_size {
            return Err(FormatError::InvalidData(format!(
                "symbol table node holds {} entries, {needed} bytes, more than the \
                 {node_size}-byte record the file's 'sym_leaf_k' allows",
                self.entries.len()
            )));
        }
        let Ok(num_symbols) = u16::try_from(self.entries.len()) else {
            return Err(FormatError::InvalidData(format!(
                "symbol table node holds {} entries, over the 2-byte count field",
                self.entries.len()
            )));
        };

        let mut buf = Vec::with_capacity(node_size);
        buf.extend_from_slice(&SNOD_SIGNATURE);
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&num_symbols.to_le_bytes());
        for entry in &self.entries {
            crate::format::superblock::encode_symbol_table_entry(
                &mut buf,
                entry,
                sizeof_addr,
                sizeof_size,
            );
        }
        buf.resize(node_size, 0);
        Ok(buf)
    }

    /// Decode a symbol table node from `buf`.
    ///
    /// `sizeof_addr` and `sizeof_size` come from the superblock, and
    /// `max_entries` is `2 * sym_leaf_k` — the node's fixed capacity
    /// (`H5Gpkg.h` `H5G_NODE_SIZE`). A node declaring more than that is
    /// corrupt, not merely unusual, so it is rejected before its entries are
    /// read.
    pub fn decode(
        buf: &[u8],
        sizeof_addr: usize,
        sizeof_size: usize,
        max_entries: u16,
    ) -> FormatResult<Self> {
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
        if num_symbols > max_entries as usize {
            return Err(FormatError::InvalidData(format!(
                "symbol table node declares {num_symbols} entries, capacity is {max_entries}"
            )));
        }

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
    use crate::format::superblock::SymbolTableCache;
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
            let mut scratch = Vec::new();
            match cache_type {
                1 => {
                    scratch.extend_from_slice(&btree_addr.to_le_bytes()[..sizeof_addr]);
                    scratch.extend_from_slice(&heap_addr.to_le_bytes()[..sizeof_addr]);
                }
                // A soft-link entry caches the 4-byte heap offset of its
                // value string; `btree_addr` carries it in this builder.
                2 => scratch.extend_from_slice(&(btree_addr as u32).to_le_bytes()),
                _ => {}
            }
            scratch.resize(16, 0);
            buf.extend_from_slice(&scratch);
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
        let node = SymbolTableNode::decode(&snod, 8, 8, 8).unwrap();
        assert_eq!(node.entries.len(), 2);
        assert_eq!(node.entries[0].name_offset, 8);
        assert_eq!(node.entries[0].obj_header_addr, 0x100);
        assert_eq!(node.entries[0].cache, SymbolTableCache::Nothing);
        assert_eq!(
            node.entries[1].cache,
            SymbolTableCache::SymbolTable {
                btree_addr: 0x300,
                heap_addr: 0x400,
            }
        );
    }

    /// A `H5G_CACHED_SLINK` entry names no object; its scratch pad holds the
    /// 4-byte offset of the link's value in the group's local heap. Reading
    /// it as a cached B-tree/heap pair (the old shape) lost the offset and
    /// left an entry whose object header address is undefined — which is how
    /// a soft link in a v0/v1 group vanished from the listing.
    #[test]
    fn decode_soft_link_entry() {
        let snod = build_snod(&[(16, UNDEF_ADDR, 2, 24, 0)], 8, 8);
        let node = SymbolTableNode::decode(
            &snod,
            8,
            8,
            crate::format::btree_v1::BTreeV1Config::default().sym_leaf_max_entries(),
        )
        .unwrap();
        assert_eq!(node.entries.len(), 1);
        assert_eq!(node.entries[0].name_offset, 16);
        assert_eq!(node.entries[0].obj_header_addr, UNDEF_ADDR);
        assert_eq!(
            node.entries[0].cache,
            SymbolTableCache::SoftLink { value_offset: 24 }
        );
        assert_eq!(node.entries[0].cached_symbol_table(), None);
    }

    #[test]
    fn decode_empty() {
        let snod = build_snod(&[], 8, 8);
        let node = SymbolTableNode::decode(&snod, 8, 8, 8).unwrap();
        assert!(node.entries.is_empty());
    }

    #[test]
    fn decode_bad_sig() {
        let mut snod = build_snod(&[], 8, 8);
        snod[0] = b'X';
        assert!(matches!(
            SymbolTableNode::decode(&snod, 8, 8, 8).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn decode_bad_version() {
        let mut snod = build_snod(&[], 8, 8);
        snod[4] = 2;
        assert!(matches!(
            SymbolTableNode::decode(&snod, 8, 8, 8).unwrap_err(),
            FormatError::InvalidVersion(2)
        ));
    }

    #[test]
    fn decode_too_short() {
        assert!(matches!(
            SymbolTableNode::decode(&[0u8; 4], 8, 8, 8).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    /// The root group's SNOD in a file h5py wrote with no `libver` argument,
    /// byte for byte: three plain hard links, and 208 zero bytes of unused
    /// capacity behind them.
    #[test]
    fn an_encoded_snod_matches_the_bytes_libhdf5_wrote() {
        let node = SymbolTableNode {
            entries: vec![
                SymbolTableEntry {
                    name_offset: 8,
                    obj_header_addr: 0x320,
                    cache: SymbolTableCache::Nothing,
                },
                SymbolTableEntry {
                    name_offset: 16,
                    obj_header_addr: 0x578,
                    cache: SymbolTableCache::Nothing,
                },
                SymbolTableEntry {
                    name_offset: 24,
                    obj_header_addr: 0x688,
                    cache: SymbolTableCache::Nothing,
                },
            ],
        };
        let node_size =
            crate::format::btree_v1::BTreeV1Config::default().symbol_table_node_size(8, 8);
        assert_eq!(node_size, 328);
        let encoded = node.encode(node_size, 8, 8).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(b"SNOD");
        expected.extend_from_slice(&[1, 0, 3, 0]); // version 1, reserved, 3 symbols
        for (name_offset, addr) in [(8u64, 0x320u64), (16, 0x578), (24, 0x688)] {
            expected.extend_from_slice(&name_offset.to_le_bytes());
            expected.extend_from_slice(&addr.to_le_bytes());
            expected.extend_from_slice(&0u32.to_le_bytes()); // cache type: nothing
            expected.extend_from_slice(&0u32.to_le_bytes()); // reserved
            expected.extend_from_slice(&[0u8; 16]); // scratch pad
        }
        assert_eq!(expected.len(), 8 + 3 * 40);
        assert_eq!(&encoded[..expected.len()], &expected[..]);
        assert!(encoded[expected.len()..].iter().all(|&b| b == 0));
        assert_eq!(encoded.len(), node_size);
    }

    /// The cached-symbol-table and soft-link scratch pads survive the round
    /// trip: a group child keeps its B-tree/heap pair, a soft link keeps the
    /// heap offset of its value.
    #[test]
    fn an_encoded_snod_round_trips_every_scratch_pad_shape() {
        let node = SymbolTableNode {
            entries: vec![
                SymbolTableEntry {
                    name_offset: 8,
                    obj_header_addr: 0x320,
                    cache: SymbolTableCache::SymbolTable {
                        btree_addr: 0x348,
                        heap_addr: 0x568,
                    },
                },
                SymbolTableEntry {
                    name_offset: 16,
                    obj_header_addr: UNDEF_ADDR,
                    cache: SymbolTableCache::SoftLink { value_offset: 24 },
                },
            ],
        };
        let cfg = crate::format::btree_v1::BTreeV1Config::default();
        let node_size = cfg.symbol_table_node_size(8, 8);
        let encoded = node.encode(node_size, 8, 8).unwrap();
        let decoded = SymbolTableNode::decode(&encoded, 8, 8, cfg.sym_leaf_max_entries()).unwrap();
        assert_eq!(decoded.entries.len(), 2);
        assert_eq!(decoded.entries[0].cache, node.entries[0].cache);
        assert_eq!(decoded.entries[1].cache, node.entries[1].cache);
        assert_eq!(decoded.entries[1].obj_header_addr, UNDEF_ADDR);
    }

    /// A SNOD is a fixed-width record, so overfilling it is a layout error the
    /// encoder must refuse rather than a buffer it can grow.
    #[test]
    fn a_snod_refuses_more_entries_than_its_record_holds() {
        let cfg = crate::format::btree_v1::BTreeV1Config::default();
        let node = SymbolTableNode {
            entries: (0..=cfg.sym_leaf_max_entries())
                .map(|i| SymbolTableEntry {
                    name_offset: u64::from(i) * 8,
                    obj_header_addr: 0x100,
                    cache: SymbolTableCache::Nothing,
                })
                .collect(),
        };
        assert!(matches!(
            node.encode(cfg.symbol_table_node_size(8, 8), 8, 8)
                .unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    #[test]
    fn decode_4byte() {
        let snod = build_snod(&[(4, 0x80, 0, UNDEF_ADDR, UNDEF_ADDR)], 4, 4);
        let node = SymbolTableNode::decode(&snod, 4, 4, 8).unwrap();
        assert_eq!(node.entries.len(), 1);
        assert_eq!(node.entries[0].name_offset, 4);
        assert_eq!(node.entries[0].obj_header_addr, 0x80);
    }
}
