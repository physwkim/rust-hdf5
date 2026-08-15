//! Committed (named) datatypes.
//!
//! `H5Tcommit` writes a datatype into the file as an object of its own, with
//! a name and an object header, so that many datasets and attributes can be
//! declared to share one type. It is the third kind of object the HDF5 model
//! has, beside groups and datasets — h5py hands it back as a `Datatype`, and
//! a file that has one lists a name that is neither a group nor a dataset.
//!
//! This crate reads them: see [`H5File::named_datatype`] and
//! [`H5Group::named_datatype`].
//!
//! [`H5File::named_datatype`]: crate::file::H5File::named_datatype
//! [`H5Group::named_datatype`]: crate::group::H5Group::named_datatype

use crate::attribute::H5Attribute;
use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
use crate::format::messages::datatype::DatatypeMessage;

/// A handle to a committed (named) datatype in a file open for reading.
///
/// Obtained from [`H5File::named_datatype`](crate::file::H5File::named_datatype)
/// or [`H5Group::named_datatype`](crate::group::H5Group::named_datatype).
pub struct H5NamedDatatype {
    file_inner: SharedInner,
    /// The path this handle was opened by, as the caller wrote it.
    name: String,
}

impl H5NamedDatatype {
    pub(crate) fn new_reader(file_inner: SharedInner, name: String) -> Self {
        Self { file_inner, name }
    }

    /// The path this datatype was opened by.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The committed type itself.
    ///
    /// # Errors
    ///
    /// [`Hdf5Error::Unsupported`] when the object is a committed datatype
    /// whose type this crate cannot decode — the object is still listed and
    /// still opens, because the name is in the file either way.
    pub fn datatype(&self) -> Result<DatatypeMessage> {
        self.with_reader(|reader, name| Ok(reader.named_datatype(name)?.clone()))
    }

    /// The names of the attributes attached to this committed datatype.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        self.with_reader(|reader, name| Ok(reader.named_datatype_attr_names(name)?))
    }

    /// Open one attribute of this committed datatype by name.
    pub fn attr(&self, attr_name: &str) -> Result<H5Attribute> {
        let msg = self
            .with_reader(|reader, name| Ok(reader.named_datatype_attr(name, attr_name)?.clone()))?;
        Ok(H5Attribute::new_reader(clone_inner(&self.file_inner), msg))
    }

    /// This committed datatype's own object-header attribute count — the
    /// equivalent of `h5py.h5o.get_info(o.id).num_attrs`.
    pub fn header_attr_count(&self) -> Result<u64> {
        self.with_reader(|reader, name| Ok(reader.named_datatype_header_attr_count(name)?))
    }

    /// Run `f` against the reader behind this handle.
    fn with_reader<T>(
        &self,
        f: impl FnOnce(&mut crate::io::reader::Hdf5Reader, &str) -> Result<T>,
    ) -> Result<T> {
        // Mutable: a path that crosses an external link opens the file that
        // link names, and the reader caches that handle for the next one.
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => f(reader, &self.name),
            H5FileInner::Writer(_) => Err(Hdf5Error::InvalidState(
                "committed datatypes are readable only in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }
}
