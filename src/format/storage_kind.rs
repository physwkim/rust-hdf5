//! Storage-kind facts read off the object-header messages that split an
//! object's metadata into an inline (compact) shape versus one moved out to
//! a fractal heap plus a v2 B-tree index (dense) once it grows past the
//! phase-change threshold. Attribute storage is governed by
//! [`AttributeInfoMessage`](crate::format::messages::attr_info::AttributeInfoMessage);
//! link storage adds a third, legacy shape (the pre-1.8 symbol table) and is
//! governed by
//! [`LinkInfoMessage`](crate::format::messages::link_info::LinkInfoMessage).

/// Where an object's attributes physically live: inline as `Attribute`
/// messages in the object header (`Compact`), or split into a fractal heap
/// plus a v2 B-tree name index once the set has grown past
/// `H5O__attr_create`'s phase-change threshold (`Dense`). Mirrors h5py's
/// `h5py.h5o.get_info(...).meta_size.attr.index_size` check: nonzero means
/// dense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributeStorage {
    /// Attribute messages sit directly in the object header.
    #[default]
    Compact,
    /// Attributes live in a fractal heap, indexed by a v2 B-tree.
    Dense,
}

impl AttributeStorage {
    /// Whether storage has moved out to the fractal heap.
    pub fn is_dense(self) -> bool {
        matches!(self, Self::Dense)
    }
}
