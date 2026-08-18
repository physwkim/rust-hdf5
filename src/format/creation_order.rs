//! How an object records the creation order of one of its collections.
//!
//! HDF5 keeps two of these per object and never merges them: a group's links
//! are governed by `H5Pset_link_creation_order` and read back from the **Link
//! Info** message's flag bits (`H5Gint.c::H5G__get_create_plist`), while every
//! object's attributes are governed by `H5Pset_attr_creation_order` and read
//! back from the **object header's own** flag bits (`H5Pocpl.c`, bits
//! `H5O_HDR_ATTR_CRT_ORDER_TRACKED` / `..._INDEXED`). A file is free to set
//! either one alone, so anything that carries "creation order is tracked" as a
//! single fact for a whole object is already wrong for half the files libhdf5
//! writes.

/// The creation-order policy of one collection — a group's links, or an
/// object's attributes.
///
/// Both property-list setters accept exactly three values: nothing,
/// `H5P_CRT_ORDER_TRACKED`, or `H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED`.
/// `INDEXED` on its own is rejected (`H5Pset_link_creation_order`,
/// `H5Pset_attr_creation_order`), which is why this is one enum rather than
/// two booleans: the fourth combination cannot be constructed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CreationOrder {
    /// No creation order is recorded. Members carry the "no creation index"
    /// sentinel and the object declares neither flag.
    #[default]
    Untracked,
    /// Every member carries its creation index and the object declares a
    /// running maximum, but no creation-order B-tree exists.
    Tracked,
    /// Tracked, plus a v2 B-tree indexing the members by creation order.
    Indexed,
}

impl CreationOrder {
    /// Whether members carry a creation index — the `TRACKED` flag bit.
    pub fn is_tracked(self) -> bool {
        !matches!(self, Self::Untracked)
    }

    /// Whether a creation-order index exists — the `INDEXED` flag bit.
    pub fn is_indexed(self) -> bool {
        matches!(self, Self::Indexed)
    }

    /// Recover the policy from the two on-disk flag bits.
    ///
    /// A lone `indexed` bit is read as [`Untracked`](Self::Untracked): the
    /// library refuses to set it, and an index over records that carry no
    /// creation index cannot be walked in creation order anyway.
    pub fn from_flags(tracked: bool, indexed: bool) -> Self {
        match (tracked, indexed) {
            (true, true) => Self::Indexed,
            (true, false) => Self::Tracked,
            (false, _) => Self::Untracked,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_implies_tracked() {
        assert!(CreationOrder::Indexed.is_tracked());
        assert!(CreationOrder::Indexed.is_indexed());
        assert!(CreationOrder::Tracked.is_tracked());
        assert!(!CreationOrder::Tracked.is_indexed());
        assert!(!CreationOrder::Untracked.is_tracked());
        assert!(!CreationOrder::Untracked.is_indexed());
    }

    #[test]
    fn a_lone_indexed_flag_reads_as_untracked() {
        assert_eq!(
            CreationOrder::from_flags(false, true),
            CreationOrder::Untracked
        );
        assert_eq!(
            CreationOrder::from_flags(false, false),
            CreationOrder::Untracked
        );
        assert_eq!(
            CreationOrder::from_flags(true, false),
            CreationOrder::Tracked
        );
        assert_eq!(
            CreationOrder::from_flags(true, true),
            CreationOrder::Indexed
        );
    }

    #[test]
    fn the_default_is_untracked() {
        assert_eq!(CreationOrder::default(), CreationOrder::Untracked);
    }
}
