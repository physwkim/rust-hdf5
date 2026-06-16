//! Attribute support.
//!
//! Attributes are small metadata items attached to datasets (or groups).
//! They are created via the [`AttrBuilder`] API obtained from
//! [`H5Dataset::new_attr`](crate::dataset::H5Dataset::new_attr).
//!
//! # Example
//!
//! ```no_run
//! use rust_hdf5::H5File;
//! use rust_hdf5::types::VarLenUnicode;
//!
//! let file = H5File::create("attrs.h5").unwrap();
//! let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
//! let attr = ds.new_attr::<VarLenUnicode>().shape(()).create("units").unwrap();
//! attr.write_scalar(&VarLenUnicode("meters".to_string())).unwrap();
//! ```

use std::marker::PhantomData;

use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::datatype::DatatypeMessage;

use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
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
    /// Dimensions for write-mode array attributes (empty = scalar). Set from
    /// [`AttrBuilder::shape`] and consumed by [`write_array`](Self::write_array).
    write_dims: Vec<usize>,
    /// The decoded attribute message for read-mode handles (carries the
    /// datatype, needed to resolve variable-length string values).
    read_attr: Option<AttributeMessage>,
}

impl H5Attribute {
    /// Create a read-mode attribute handle from a decoded attribute message.
    pub(crate) fn new_reader(file_inner: SharedInner, attr_msg: AttributeMessage) -> Self {
        Self {
            file_inner,
            ds_index: usize::MAX,
            name: attr_msg.name.clone(),
            write_dims: Vec::new(),
            read_attr: Some(attr_msg),
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
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("num_attr.h5").unwrap();
    /// let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
    /// ds.write_raw(&[0.0f32; 10]).unwrap();
    /// let attr = ds.new_attr::<f64>().shape(()).create("scale").unwrap();
    /// attr.write_numeric(&3.14f64).unwrap();
    /// ```
    pub fn write_numeric<T: crate::types::H5Type>(&self, value: &T) -> Result<()> {
        let es = T::element_size();
        let raw = unsafe { std::slice::from_raw_parts(value as *const T as *const u8, es) };
        let attr_msg = AttributeMessage::scalar_numeric(&self.name, T::hdf5_type(), raw.to_vec());

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

    /// Write a numeric array attribute.
    ///
    /// The number of `values` must equal the product of the dimensions set
    /// via [`AttrBuilder::shape`]; if no shape was set the attribute is a
    /// scalar and exactly one value is required. The on-disk datatype is
    /// `T::hdf5_type()` and the dataspace is the simple dataspace described by
    /// the shape — matching the 1-D `int32` array attributes AreaDetector
    /// writes (e.g. `NDArrayDimOffset`, `Binning`, `Reverse`).
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("arr_attr.h5").unwrap();
    /// let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
    /// ds.write_raw(&[0.0f32; 10]).unwrap();
    /// let attr = ds.new_attr::<i32>().shape([3]).create("dim_offset").unwrap();
    /// attr.write_array(&[0i32, 4, 8]).unwrap();
    /// ```
    pub fn write_array<T: crate::types::H5Type>(&self, values: &[T]) -> Result<()> {
        // Product of an empty shape is 1 (a scalar holds one element).
        let expected: usize = self.write_dims.iter().product();
        if values.len() != expected {
            return Err(Hdf5Error::InvalidState(format!(
                "attribute '{}' shape {:?} needs {} elements, got {}",
                self.name,
                self.write_dims,
                expected,
                values.len()
            )));
        }

        let es = T::element_size();
        // Safety: `T: H5Type` is a `Copy` numeric primitive with a defined
        // byte representation; `element_size()` matches `size_of::<T>()`. The
        // slice borrows `values` only for this call.
        let raw =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * es) };
        let dims_u64: Vec<u64> = self.write_dims.iter().map(|&d| d as u64).collect();
        let attr_msg =
            AttributeMessage::array_numeric(&self.name, T::hdf5_type(), &dims_u64, raw.to_vec());

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
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open("num_attr.h5").unwrap();
    /// let ds = file.dataset("data").unwrap();
    /// let attr = ds.attr("scale").unwrap();
    /// let val: f64 = attr.read_numeric().unwrap();
    /// ```
    pub fn read_numeric<T: crate::types::H5Type>(&self) -> Result<T> {
        let data = self
            .read_attr
            .as_ref()
            .map(|a| &a.data)
            .ok_or_else(|| Hdf5Error::InvalidState("attribute has no read data".into()))?;
        let es = T::element_size();
        if data.len() < es {
            return Err(Hdf5Error::TypeMismatch(format!(
                "attribute data {} bytes, need {} for type",
                data.len(),
                es
            )));
        }
        unsafe {
            let mut val = std::mem::MaybeUninit::<T>::uninit();
            std::ptr::copy_nonoverlapping(data.as_ptr(), val.as_mut_ptr() as *mut u8, es);
            Ok(val.assume_init())
        }
    }

    /// Read the attribute value as a string.
    ///
    /// Handles both fixed-length string attributes and variable-length
    /// string attributes (h5py's default), resolving a vlen value through
    /// the global heap.
    pub fn read_string(&self) -> Result<String> {
        let attr = self.read_attr.as_ref().ok_or_else(|| {
            Hdf5Error::InvalidState("attribute has no read data (write-mode handle?)".into())
        })?;
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => Ok(reader.attr_string_value(attr)?),
            _ => {
                // No reader available — fall back to the raw fixed-length
                // interpretation.
                let end = attr
                    .data
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(attr.data.len());
                Ok(String::from_utf8_lossy(&attr.data[..end]).to_string())
            }
        }
    }

    /// Return the attribute datatype as parsed from the file (read mode only).
    ///
    /// Mirrors [`H5Dataset::datatype`](crate::dataset::H5Dataset::datatype):
    /// it exposes the full datatype — class (integer vs floating-point vs
    /// string vs compound …), signedness, byte order and bit precision — so
    /// callers mapping an attribute to a NumPy / Arrow dtype need not infer a
    /// type from the byte width, which cannot distinguish `u8` from `i8` (both
    /// 1 byte) or `i32` from `f32` (both 4 bytes).
    ///
    /// # Errors
    ///
    /// Returns an error for a write-mode handle, which carries no decoded
    /// attribute message.
    ///
    /// ```no_run
    /// # use rust_hdf5::{H5File, DatatypeMessage};
    /// let file = H5File::open("data.h5").unwrap();
    /// let ds = file.dataset("image").unwrap();
    /// let attr = ds.attr("scale").unwrap();
    /// match attr.datatype().unwrap() {
    ///     DatatypeMessage::FloatingPoint { size, .. } => println!("float: {size} bytes"),
    ///     other => println!("other type: {other}"),
    /// }
    /// ```
    pub fn datatype(&self) -> Result<DatatypeMessage> {
        self.read_attr
            .as_ref()
            .map(|a| a.datatype.clone())
            .ok_or_else(|| {
                Hdf5Error::InvalidState("attribute has no read data (write-mode handle?)".into())
            })
    }

    /// Read the raw attribute data bytes.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.read_attr
            .as_ref()
            .map(|a| a.data.clone())
            .ok_or_else(|| {
                Hdf5Error::InvalidState("attribute has no read data (write-mode handle?)".into())
            })
    }
}

/// Shapes accepted by [`AttrBuilder::shape`].
///
/// `()` selects a scalar attribute (rank 0). A slice, array, or `Vec` of
/// `usize` selects a simple dataspace with those dimension sizes — e.g.
/// `[3]` for a 1-D array attribute of length 3.
pub trait AttrShape {
    /// Dimension sizes for the attribute; empty for a scalar.
    fn attr_dims(&self) -> Vec<usize>;
}

impl AttrShape for () {
    fn attr_dims(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl AttrShape for &[usize] {
    fn attr_dims(&self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> AttrShape for [usize; N] {
    fn attr_dims(&self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> AttrShape for &[usize; N] {
    fn attr_dims(&self) -> Vec<usize> {
        self.to_vec()
    }
}

impl AttrShape for Vec<usize> {
    fn attr_dims(&self) -> Vec<usize> {
        self.clone()
    }
}

/// A fluent builder for creating attributes on datasets.
///
/// Obtained from [`H5Dataset::new_attr::<T>()`](crate::dataset::H5Dataset::new_attr).
pub struct AttrBuilder<'a, T> {
    file_inner: &'a SharedInner,
    ds_index: usize,
    dims: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<'a, T> AttrBuilder<'a, T> {
    pub(crate) fn new(file_inner: &'a SharedInner, ds_index: usize) -> Self {
        Self {
            file_inner,
            ds_index,
            dims: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Set the attribute shape.
    ///
    /// Use `()` for a scalar attribute, or an array/slice of dimension sizes
    /// for an array attribute (e.g. `[3]` for a 1-D array of length 3). The
    /// shape is consumed by [`H5Attribute::write_array`]; the scalar writers
    /// ([`write_scalar`](H5Attribute::write_scalar),
    /// [`write_numeric`](H5Attribute::write_numeric)) ignore it.
    #[must_use]
    pub fn shape<S: AttrShape>(mut self, shape: S) -> Self {
        self.dims = shape.attr_dims();
        self
    }

    /// Create the attribute with the given name.
    ///
    /// The attribute is created but does not yet have a value.
    /// Call [`H5Attribute::write_scalar`] or
    /// [`H5Attribute::write_array`] to set the value.
    pub fn create(self, name: &str) -> Result<H5Attribute> {
        Ok(H5Attribute {
            file_inner: clone_inner(self.file_inner),
            ds_index: self.ds_index,
            name: name.to_string(),
            write_dims: self.dims,
            read_attr: None,
        })
    }
}
