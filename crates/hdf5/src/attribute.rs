//! Attribute support.
//!
//! Attributes are small metadata items attached to datasets (or groups).
//! They are created via the [`AttrBuilder`] API obtained from
//! [`H5Dataset::new_attr`](crate::dataset::H5Dataset::new_attr).
//!
//! # Example
//!
//! ```no_run
//! use hdf5::H5File;
//! use hdf5::types::VarLenUnicode;
//!
//! let file = H5File::create("attrs.h5").unwrap();
//! let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
//! let attr = ds.new_attr::<VarLenUnicode>().shape(()).create("units").unwrap();
//! attr.write_scalar(&VarLenUnicode("meters".to_string())).unwrap();
//! ```

use std::marker::PhantomData;

use hdf5_format::messages::attribute::AttributeMessage;

use crate::error::{Hdf5Error, Result};
use crate::file::{H5FileInner, SharedInner, borrow_inner_mut, clone_inner};
use crate::types::VarLenUnicode;

/// A handle to an HDF5 attribute.
///
/// After creating an attribute via [`AttrBuilder::create`], use
/// [`write_scalar`](Self::write_scalar) or [`write_string`](Self::write_string)
/// to set its value.
///
/// In read mode, use [`read_string`](Self::read_string) to read string attributes.
pub struct H5Attribute {
    file_inner: SharedInner,
    ds_index: usize,
    name: String,
    /// Cached data for read-mode attributes.
    read_data: Option<Vec<u8>>,
}

impl H5Attribute {
    /// Create a read-mode attribute handle with cached data.
    pub(crate) fn new_reader(
        file_inner: SharedInner,
        name: String,
        data: Vec<u8>,
    ) -> Self {
        Self {
            file_inner,
            ds_index: usize::MAX,
            name,
            read_data: Some(data),
        }
    }

    /// Return the attribute name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Write a scalar value to the attribute.
    ///
    /// For `VarLenUnicode`, this writes a fixed-length string attribute
    /// whose size is determined by the string value.
    pub fn write_scalar(&self, value: &VarLenUnicode) -> Result<()> {
        let attr_msg = AttributeMessage::scalar_string(&self.name, &value.0);

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.add_dataset_attribute(self.ds_index, attr_msg)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot write attributes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Write a string value to the attribute (convenience method).
    pub fn write_string(&self, value: &str) -> Result<()> {
        self.write_scalar(&VarLenUnicode(value.to_string()))
    }

    /// Write a numeric scalar attribute.
    ///
    /// ```no_run
    /// # use hdf5::H5File;
    /// let file = H5File::create("num_attr.h5").unwrap();
    /// let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
    /// ds.write_raw(&[0.0f32; 10]).unwrap();
    /// let attr = ds.new_attr::<f64>().shape(()).create("scale").unwrap();
    /// attr.write_numeric(&3.14f64).unwrap();
    /// ```
    pub fn write_numeric<T: crate::types::H5Type>(&self, value: &T) -> Result<()> {
        let es = T::element_size();
        let raw = unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, es)
        };
        let attr_msg = AttributeMessage::scalar_numeric(
            &self.name,
            T::hdf5_type(),
            raw.to_vec(),
        );

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.add_dataset_attribute(self.ds_index, attr_msg)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot write attributes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Read a numeric scalar attribute.
    ///
    /// ```no_run
    /// # use hdf5::H5File;
    /// let file = H5File::open("num_attr.h5").unwrap();
    /// let ds = file.dataset("data").unwrap();
    /// let attr = ds.attr("scale").unwrap();
    /// let val: f64 = attr.read_numeric().unwrap();
    /// ```
    pub fn read_numeric<T: crate::types::H5Type>(&self) -> Result<T> {
        let data = self.read_data.as_ref().ok_or_else(|| {
            Hdf5Error::InvalidState("attribute has no read data".into())
        })?;
        let es = T::element_size();
        if data.len() < es {
            return Err(Hdf5Error::TypeMismatch(format!(
                "attribute data {} bytes, need {} for type", data.len(), es
            )));
        }
        unsafe {
            let mut val = std::mem::MaybeUninit::<T>::uninit();
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                val.as_mut_ptr() as *mut u8,
                es,
            );
            Ok(val.assume_init())
        }
    }

    /// Read the attribute value as a string.
    ///
    /// Works for fixed-length string attributes (as written by this library)
    /// and returns the string value with any trailing null bytes stripped.
    pub fn read_string(&self) -> Result<String> {
        let data = self.read_data.as_ref().ok_or_else(|| {
            Hdf5Error::InvalidState("attribute has no read data (write-mode handle?)".into())
        })?;
        // Strip trailing null bytes
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        Ok(String::from_utf8_lossy(&data[..end]).to_string())
    }

    /// Read the raw attribute data bytes.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.read_data.clone().ok_or_else(|| {
            Hdf5Error::InvalidState("attribute has no read data (write-mode handle?)".into())
        })
    }
}

/// A fluent builder for creating attributes on datasets.
///
/// Obtained from [`H5Dataset::new_attr::<T>()`](crate::dataset::H5Dataset::new_attr).
pub struct AttrBuilder<'a, T> {
    file_inner: &'a SharedInner,
    ds_index: usize,
    _shape_set: bool,
    _marker: PhantomData<T>,
}

impl<'a, T> AttrBuilder<'a, T> {
    pub(crate) fn new(
        file_inner: &'a SharedInner,
        ds_index: usize,
    ) -> Self {
        Self {
            file_inner,
            ds_index,
            _shape_set: false,
            _marker: PhantomData,
        }
    }

    /// Set the attribute shape. Use `()` for a scalar attribute.
    #[must_use]
    pub fn shape<S>(mut self, _shape: S) -> Self {
        // For now we only support scalar attributes.
        self._shape_set = true;
        self
    }

    /// Create the attribute with the given name.
    ///
    /// The attribute is created but does not yet have a value.
    /// Call [`H5Attribute::write_scalar`] to set the value.
    pub fn create(self, name: &str) -> Result<H5Attribute> {
        Ok(H5Attribute {
            file_inner: clone_inner(self.file_inner),
            ds_index: self.ds_index,
            name: name.to_string(),
            read_data: None,
        })
    }
}
