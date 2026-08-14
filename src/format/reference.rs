//! Reference elements: what an element of a class-7 datatype names.
//!
//! The pre-1.12 kinds, both of which h5py 3.x writes today, store a file
//! address directly (`H5Tref.c`):
//!
//! ```text
//! H5R_OBJECT1          sizeof_addr bytes   the target's object header address
//! H5R_DATASET_REGION1  sizeof_addr + 4     a global-heap id: collection
//!                                          address then a u32 object index
//! ```

use crate::format::bytes::read_le_addr;
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// The address a reference element leads with, or `None` when it names
/// nothing.
///
/// Both element layouts start with a file address, and both spell "no target"
/// the same two ways: the all-ones undefined address `H5F_addr_decode`
/// produces, and 0 — the superblock's own address, so never an object header,
/// and what an unwritten (fill-value) element holds. `H5R__decode_heap`
/// rejects both together (`!H5_addr_defined(hobjid.addr) || hobjid.addr == 0`),
/// so this crate applies the one rule to both kinds rather than per element
/// layout.
fn target_address(elem: &[u8], sizeof_addr: usize) -> Option<u64> {
    match read_le_addr(elem, sizeof_addr) {
        0 | UNDEF_ADDR => None,
        addr => Some(addr),
    }
}

/// The address a `H5R_OBJECT1` element names, or `None` for a null reference.
pub fn decode_object_element(elem: &[u8], ctx: &FormatContext) -> FormatResult<Option<u64>> {
    let sa = ctx.sizeof_addr as usize;
    if elem.len() < sa {
        return Err(FormatError::BufferTooShort {
            needed: sa,
            available: elem.len(),
        });
    }
    Ok(target_address(elem, sa))
}

/// One reference element, decoded and resolved against the file it came from.
///
/// `path` is the target's absolute path when the file's link structure names
/// it, and `None` when nothing in the traversed structure points at that
/// address — a reference into an untraversed part of the file, or a stale one
/// left by a deletion. The address is reported either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reference {
    /// An element naming no object: the undefined address libhdf5 writes for
    /// an unset reference, or the zeroed element h5py leaves in an
    /// unwritten slot.
    Null,
    /// `H5R_OBJECT1`: a whole object.
    Object {
        /// Object header address of the target.
        address: u64,
        /// Absolute path of the target.
        path: Option<String>,
    },
}

impl Reference {
    /// The target's absolute path, when the file names it.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Null => None,
            Self::Object { path, .. } => path.as_deref(),
        }
    }

    /// The target's object header address, or `None` for a null reference.
    pub fn address(&self) -> Option<u64> {
        match self {
            Self::Null => None,
            Self::Object { address, .. } => Some(*address),
        }
    }

    /// Whether this element names no object.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    #[test]
    fn object_elements_report_their_address_or_null() {
        assert_eq!(
            decode_object_element(&0x320u64.to_le_bytes(), &ctx()).unwrap(),
            Some(0x320)
        );
        assert_eq!(
            decode_object_element(&[0xFF; 8], &ctx()).unwrap(),
            None,
            "an undefined address is a null reference"
        );
        assert_eq!(
            decode_object_element(&[0; 8], &ctx()).unwrap(),
            None,
            "so is address 0, which h5py writes for an unset element"
        );
    }

    #[test]
    fn a_truncated_object_element_is_refused() {
        let err = decode_object_element(&[0u8; 4], &ctx()).unwrap_err();
        assert!(
            matches!(err, FormatError::BufferTooShort { .. }),
            "unexpected error: {err:?}"
        );
    }
}
