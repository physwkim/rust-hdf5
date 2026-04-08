//! B-tree v1 decode (for reading legacy HDF5 files).
//!
//! The B-tree v1 is used in v0/v1 groups to index symbol table entries.
//! For group B-trees (type 0), each key is a name offset into the local
//! heap, and each child pointer is an address of either a SNOD (leaf
//! level) or another TREE node (internal level).
//!
//! Layout:
//! ```text
//! "TREE" (4 bytes)
//! type: 1 byte (0 = group)
//! level: 1 byte (0 = leaf)
//! entries_used: u16 LE
//! left_sibling: sizeof_addr bytes LE
//! right_sibling: sizeof_addr bytes LE
//! Then interleaved keys and children:
//!   key[0], child[0], key[1], child[1], ..., key[entries_used]
//! ```
//!
//! For type-0 (group) B-trees:
//! - Each key is sizeof_size bytes (name offset into local heap)
//! - Each child is sizeof_addr bytes (address of SNOD or sub-TREE)

use crate::format::{FormatError, FormatResult, UNDEF_ADDR};

/// The 4-byte B-tree v1 signature.
pub const BTREE_V1_SIGNATURE: [u8; 4] = *b"TREE";

/// A decoded B-tree v1 node.
#[derive(Debug, Clone)]
pub struct BTreeV1Node {
    /// Node type: 0 = group, 1 = raw data chunk.
    pub node_type: u8,
    /// Node level: 0 = leaf (children are SNODs), >0 = internal (children are sub-TREE).
    pub level: u8,
    /// Number of entries used in this node.
    pub entries_used: u16,
    /// Address of left sibling, or UNDEF_ADDR if none.
    pub left_sibling: u64,
    /// Address of right sibling, or UNDEF_ADDR if none.
    pub right_sibling: u64,
    /// Keys (entries_used + 1 entries for type-0 group trees).
    pub keys: Vec<u64>,
    /// Child addresses (entries_used entries).
    pub children: Vec<u64>,
}

impl BTreeV1Node {
    /// Decode a B-tree v1 node from `buf`.
    ///
    /// `sizeof_addr` and `sizeof_size` come from the superblock.
    pub fn decode(buf: &[u8], sizeof_addr: usize, sizeof_size: usize) -> FormatResult<Self> {
        let header_size = 4 + 1 + 1 + 2 + sizeof_addr * 2;
        if buf.len() < header_size {
            return Err(FormatError::BufferTooShort {
                needed: header_size,
                available: buf.len(),
            });
        }

        if buf[0..4] != BTREE_V1_SIGNATURE {
            return Err(FormatError::InvalidSignature);
        }

        let node_type = buf[4];
        let level = buf[5];
        let entries_used = u16::from_le_bytes([buf[6], buf[7]]);

        let mut pos = 8;
        let left_sibling = read_addr(&buf[pos..], sizeof_addr);
        pos += sizeof_addr;
        let right_sibling = read_addr(&buf[pos..], sizeof_addr);
        pos += sizeof_addr;

        // For group B-trees (type 0):
        // Interleaved: key[0], child[0], key[1], child[1], ..., key[n]
        // That's (entries_used + 1) keys and entries_used children.
        let n = entries_used as usize;

        if node_type == 0 {
            // Group B-tree
            let key_size = sizeof_size;
            let child_size = sizeof_addr;
            // Total data: (n+1) keys interleaved with n children
            let data_size = (n + 1) * key_size + n * child_size;
            let needed = pos + data_size;
            if buf.len() < needed {
                return Err(FormatError::BufferTooShort {
                    needed,
                    available: buf.len(),
                });
            }

            let mut keys = Vec::with_capacity(n + 1);
            let mut children = Vec::with_capacity(n);

            for _i in 0..n {
                // key[i]
                keys.push(read_uint(&buf[pos..], key_size));
                pos += key_size;
                // child[i]
                children.push(read_uint(&buf[pos..], child_size));
                pos += child_size;
            }
            // final key[n]
            keys.push(read_uint(&buf[pos..], key_size));

            Ok(BTreeV1Node {
                node_type,
                level,
                entries_used,
                left_sibling,
                right_sibling,
                keys,
                children,
            })
        } else {
            // Raw data chunk B-tree (type 1) - we only need to parse the structure
            // For now, return empty keys/children (we primarily support type 0)
            Err(FormatError::UnsupportedFeature(format!(
                "B-tree v1 type {} not supported (only type 0 for groups)",
                node_type
            )))
        }
    }
}

/// Read a little-endian unsigned integer of `n` bytes into a u64.
fn read_uint(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

/// Read a little-endian address of `n` bytes, mapping all-ones to UNDEF_ADDR.
fn read_addr(buf: &[u8], n: usize) -> u64 {
    if buf[..n].iter().all(|&b| b == 0xFF) {
        UNDEF_ADDR
    } else {
        read_uint(buf, n)
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a group B-tree v1 node for testing.
    fn build_group_btree(
        level: u8,
        keys: &[u64],
        children: &[u64],
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Vec<u8> {
        assert_eq!(keys.len(), children.len() + 1);
        let entries_used = children.len() as u16;

        let mut buf = Vec::new();
        buf.extend_from_slice(&BTREE_V1_SIGNATURE);
        buf.push(0); // type = group
        buf.push(level);
        buf.extend_from_slice(&entries_used.to_le_bytes());
        // left sibling = UNDEF
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);
        // right sibling = UNDEF
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()[..sizeof_addr]);

        // Interleaved keys and children
        for i in 0..children.len() {
            buf.extend_from_slice(&keys[i].to_le_bytes()[..sizeof_size]);
            buf.extend_from_slice(&children[i].to_le_bytes()[..sizeof_addr]);
        }
        // Final key
        buf.extend_from_slice(&keys[children.len()].to_le_bytes()[..sizeof_size]);

        buf
    }

    #[test]
    fn decode_leaf_node() {
        let buf = build_group_btree(
            0,               // leaf
            &[0, 8, 16],     // 3 keys
            &[0x100, 0x200], // 2 children (SNOD addresses)
            8,
            8,
        );
        let node = BTreeV1Node::decode(&buf, 8, 8).unwrap();
        assert_eq!(node.node_type, 0);
        assert_eq!(node.level, 0);
        assert_eq!(node.entries_used, 2);
        assert_eq!(node.keys, vec![0, 8, 16]);
        assert_eq!(node.children, vec![0x100, 0x200]);
        assert_eq!(node.left_sibling, UNDEF_ADDR);
        assert_eq!(node.right_sibling, UNDEF_ADDR);
    }

    #[test]
    fn decode_internal_node() {
        let buf = build_group_btree(
            1,         // internal
            &[0, 100], // 2 keys
            &[0x500],  // 1 child (sub-TREE address)
            8,
            8,
        );
        let node = BTreeV1Node::decode(&buf, 8, 8).unwrap();
        assert_eq!(node.level, 1);
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![0x500]);
    }

    #[test]
    fn decode_single_entry() {
        let buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        let node = BTreeV1Node::decode(&buf, 8, 8).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn decode_4byte() {
        let buf = build_group_btree(0, &[0, 4], &[0x80], 4, 4);
        let node = BTreeV1Node::decode(&buf, 4, 4).unwrap();
        assert_eq!(node.entries_used, 1);
        assert_eq!(node.children, vec![0x80]);
    }

    #[test]
    fn decode_bad_sig() {
        let mut buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        buf[0] = b'X';
        assert!(matches!(
            BTreeV1Node::decode(&buf, 8, 8).unwrap_err(),
            FormatError::InvalidSignature
        ));
    }

    #[test]
    fn decode_too_short() {
        assert!(matches!(
            BTreeV1Node::decode(&[0u8; 4], 8, 8).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    #[test]
    fn decode_unsupported_type() {
        let mut buf = build_group_btree(0, &[0, 8], &[0x100], 8, 8);
        buf[4] = 1; // type = raw data chunks
        assert!(matches!(
            BTreeV1Node::decode(&buf, 8, 8).unwrap_err(),
            FormatError::UnsupportedFeature(_)
        ));
    }
}
