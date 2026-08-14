//! HDF5 file handle — the main entry point for the public API.
//!
//! ```no_run
//! use rust_hdf5::H5File;
//!
//! // Write
//! let file = H5File::create("example.h5").unwrap();
//! let ds = file.new_dataset::<u8>().shape(&[10, 20]).create("data").unwrap();
//! ds.write_raw(&vec![0u8; 200]).unwrap();
//! drop(file);
//!
//! // Read
//! let file = H5File::open("example.h5").unwrap();
//! let ds = file.dataset("data").unwrap();
//! let data = ds.read_raw::<u8>().unwrap();
//! assert_eq!(data.len(), 200);
//! ```

use std::path::Path;

use crate::io::locking::FileLocking;
use crate::io::{Hdf5Reader, Hdf5Writer};

use crate::dataset::{DatasetBuilder, H5Dataset};
use crate::error::{Hdf5Error, Result};
use crate::format::messages::filter::FilterPipeline;
use crate::group::H5Group;
use crate::types::H5Type;

// ---------------------------------------------------------------------------
// Thread-safety: choose between Rc<RefCell<>> and Arc<Mutex<>> based on
// the `threadsafe` feature flag.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "threadsafe"))]
pub(crate) type SharedInner = std::rc::Rc<std::cell::RefCell<H5FileInner>>;

#[cfg(feature = "threadsafe")]
pub(crate) type SharedInner = std::sync::Arc<std::sync::RwLock<H5FileInner>>;

/// Helper to borrow/lock the inner state immutably.
#[cfg(not(feature = "threadsafe"))]
pub(crate) fn borrow_inner(inner: &SharedInner) -> std::cell::Ref<'_, H5FileInner> {
    inner.borrow()
}

/// Helper to borrow/lock the inner state mutably.
#[cfg(not(feature = "threadsafe"))]
pub(crate) fn borrow_inner_mut(inner: &SharedInner) -> std::cell::RefMut<'_, H5FileInner> {
    inner.borrow_mut()
}

/// Helper to clone a SharedInner.
#[cfg(not(feature = "threadsafe"))]
pub(crate) fn clone_inner(inner: &SharedInner) -> SharedInner {
    std::rc::Rc::clone(inner)
}

/// Helper to wrap an H5FileInner in SharedInner.
#[cfg(not(feature = "threadsafe"))]
pub(crate) fn new_shared(inner: H5FileInner) -> SharedInner {
    std::rc::Rc::new(std::cell::RefCell::new(inner))
}

/// Acquire a shared (read) lock on the inner state. The fine-grained writer
/// (atomic allocator, positioned handle, per-dataset `Slot` mutexes) is safe to
/// drive through `&H5FileInner`, so the non-extending chunk-write path takes
/// this read guard and lets writes to different datasets proceed concurrently.
#[cfg(feature = "threadsafe")]
pub(crate) fn borrow_inner(inner: &SharedInner) -> std::sync::RwLockReadGuard<'_, H5FileInner> {
    inner.read().unwrap()
}

/// Acquire an exclusive (write) lock on the inner state. Required by paths that
/// mutate shared writer state directly — create, extend/`set_extent`, append,
/// metadata, finalize/close.
#[cfg(feature = "threadsafe")]
pub(crate) fn borrow_inner_mut(
    inner: &SharedInner,
) -> std::sync::RwLockWriteGuard<'_, H5FileInner> {
    inner.write().unwrap()
}

#[cfg(feature = "threadsafe")]
pub(crate) fn clone_inner(inner: &SharedInner) -> SharedInner {
    std::sync::Arc::clone(inner)
}

#[cfg(feature = "threadsafe")]
pub(crate) fn new_shared(inner: H5FileInner) -> SharedInner {
    std::sync::Arc::new(std::sync::RwLock::new(inner))
}

/// The inner state of an HDF5 file, shared with datasets via reference counting.
///
/// By default, this uses `Rc<RefCell<>>` for zero-overhead single-threaded use.
/// Enable the `threadsafe` feature to use `Arc<Mutex<>>` instead, making
/// `H5File` `Send + Sync`.
pub(crate) enum H5FileInner {
    Writer(Hdf5Writer),
    Reader(Hdf5Reader),
    /// Sentinel value used during `close()` to take ownership of the writer.
    Closed,
}

/// An HDF5 file opened for reading or writing.
///
/// Datasets created from this file hold a shared reference to the underlying
/// I/O handle, so the file does not need to outlive its datasets (they share
/// ownership via reference counting).
pub struct H5File {
    pub(crate) inner: SharedInner,
}

impl H5File {
    /// Create a new HDF5 file at `path`. Truncates if the file already exists.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let writer = Hdf5Writer::create(path.as_ref())?;
        Ok(Self {
            inner: new_shared(H5FileInner::Writer(writer)),
        })
    }

    /// Open an existing HDF5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = Hdf5Reader::open(path.as_ref())?;
        Ok(Self {
            inner: new_shared(H5FileInner::Reader(reader)),
        })
    }

    /// Open an existing HDF5 file for appending new datasets.
    ///
    /// Existing datasets are preserved. New datasets can be added and will
    /// be written after the current end of file. Existing chunked datasets
    /// can be extended with `write_chunk` and `extend_dataset`.
    ///
    /// ```no_run
    /// use rust_hdf5::H5File;
    /// let file = H5File::open_rw("existing.h5").unwrap();
    /// let ds = file.new_dataset::<f64>().shape(&[100]).create("new_data").unwrap();
    /// ds.write_raw(&vec![0.0f64; 100]).unwrap();
    /// file.close().unwrap();
    /// ```
    pub fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self> {
        let writer = Hdf5Writer::open_append(path.as_ref())?;
        Ok(Self {
            inner: new_shared(H5FileInner::Writer(writer)),
        })
    }

    /// Start building open options for an HDF5 file.
    ///
    /// Use this to control file-locking behavior explicitly:
    ///
    /// ```no_run
    /// use rust_hdf5::{H5File, FileLocking};
    /// // Open with locking disabled (e.g. on NFS without lock support).
    /// let file = H5File::options()
    ///     .locking(FileLocking::Disabled)
    ///     .open_rw("existing.h5")
    ///     .unwrap();
    /// # let _ = file;
    /// ```
    pub fn options() -> H5FileOptions {
        H5FileOptions::default()
    }

    /// Opt in to the latest file format for datasets created after this call —
    /// the equivalent of libhdf5's `H5Pset_libver_bounds(low = H5F_LIBVER_V200)`.
    ///
    /// With `latest` set, filtered chunked datasets get a version-5 data layout
    /// message, whose chunk indexes store on-disk chunk sizes in fixed-width
    /// (`sizeof_size`, i.e. 8-byte) fields instead of fields sized from the
    /// uncompressed chunk size. That removes the overflow risk when a filter
    /// *expands* a chunk, but the file is only readable by libhdf5 ≥ 2.0
    /// (h5py bundling hdf5 1.14 rejects it with "bad version number").
    ///
    /// Off by default; unfiltered and contiguous datasets are unaffected.
    /// Independent of this setting, a chunk larger than 4 GiB forces version 5
    /// because version 4 cannot represent its size field, matching libhdf5.
    ///
    /// Errors in read mode.
    pub fn set_libver_latest(&self, latest: bool) -> Result<()> {
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.set_libver_latest(latest);
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot write in read mode".into())),
        }
    }

    /// Return a handle to the root group.
    ///
    /// The root group can be used to create datasets and sub-groups.
    pub fn root_group(&self) -> H5Group {
        H5Group::new(clone_inner(&self.inner), "/".to_string())
    }

    /// Create a group in the root of the file.
    ///
    /// ```no_run
    /// use rust_hdf5::H5File;
    /// let file = H5File::create("groups.h5").unwrap();
    /// let grp = file.create_group("detector").unwrap();
    /// ```
    pub fn create_group(&self, name: &str) -> Result<H5Group> {
        self.root_group().create_group(name)
    }

    /// Start building a new dataset with the given element type.
    ///
    /// This returns a fluent builder. Call `.shape(...)` to set dimensions and
    /// `.create("name")` to finalize.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("build.h5").unwrap();
    /// let ds = file.new_dataset::<f64>().shape(&[3, 4]).create("matrix").unwrap();
    /// ```
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilder<T> {
        DatasetBuilder::new(clone_inner(&self.inner))
    }

    /// Add a string attribute to the file (root group).
    ///
    /// The value is stored as a variable-length UTF-8 string (read back as a
    /// Python `str` by h5py), not a fixed-length string.
    pub fn set_attr_string(&self, name: &str, value: &str) -> Result<()> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.set_vlen_string_attribute(
                    crate::io::writer::AttrTarget::Root,
                    name,
                    value,
                )?;
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot write in read mode".into())),
        }
    }

    /// Add a numeric attribute to the file (root group).
    pub fn set_attr_numeric<T: crate::types::H5Type>(&self, name: &str, value: &T) -> Result<()> {
        use crate::format::messages::attribute::AttributeMessage;
        let es = T::element_size();
        let raw = unsafe { std::slice::from_raw_parts(value as *const T as *const u8, es) };
        let attr = AttributeMessage::scalar_numeric(name, T::hdf5_type(), raw.to_vec());
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.add_root_attribute(attr)?;
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot write in read mode".into())),
        }
    }

    /// Add a numeric (or bool) **array** attribute to the file (root group).
    ///
    /// The values are written as a 1-D HDF5 array attribute (simple dataspace
    /// `[values.len()]`, on-disk type `T::hdf5_type()`), read back by h5py as a
    /// numpy array — the array counterpart of [`set_attr_numeric`](Self::set_attr_numeric).
    /// For a multi-dimensional shape use
    /// [`set_attr_array_numeric_nd`](Self::set_attr_array_numeric_nd).
    pub fn set_attr_array_numeric<T: crate::types::H5Type>(
        &self,
        name: &str,
        values: &[T],
    ) -> Result<()> {
        self.set_attr_array_numeric_nd(name, values, &[values.len()])
    }

    /// Add a numeric (or bool) **N-dimensional array** attribute to the file
    /// (root group).
    ///
    /// `shape` gives the dataspace dimensions; `values` is the row-major data
    /// and its length must equal the product of `shape` (an empty `shape` is a
    /// scalar, requiring exactly one value). Read back by h5py as a numpy array
    /// of that shape. [`set_attr_array_numeric`](Self::set_attr_array_numeric)
    /// is the 1-D convenience form.
    pub fn set_attr_array_numeric_nd<T: crate::types::H5Type>(
        &self,
        name: &str,
        values: &[T],
        shape: &[usize],
    ) -> Result<()> {
        use crate::format::messages::attribute::AttributeMessage;
        let n: usize = shape.iter().product();
        if values.len() != n {
            return Err(Hdf5Error::InvalidState(format!(
                "attribute '{name}' shape {shape:?} needs {n} elements, got {}",
                values.len()
            )));
        }
        let es = T::element_size();
        // Safety: `T: H5Type` is a `Copy` POD numeric whose byte width is `es`.
        let raw =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * es) };
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let attr = AttributeMessage::array_numeric(name, T::hdf5_type(), &dims, raw.to_vec());
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.add_root_attribute(attr)?;
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot write in read mode".into())),
        }
    }

    /// Add a variable-length UTF-8 string **array** attribute to the file (root
    /// group), read back by h5py as a 1-D array of `str` — the array counterpart
    /// of [`set_attr_string`](Self::set_attr_string). For a multi-dimensional
    /// shape use [`set_attr_string_array_nd`](Self::set_attr_string_array_nd).
    pub fn set_attr_string_array(&self, name: &str, values: &[&str]) -> Result<()> {
        self.set_attr_string_array_nd(name, values, &[values.len()])
    }

    /// Add a variable-length UTF-8 string **N-dimensional array** attribute to
    /// the file (root group).
    ///
    /// `shape` gives the dataspace dimensions; `values` is the row-major data
    /// and its length must equal the product of `shape` (an empty `shape` is a
    /// scalar, requiring exactly one value). Read back by h5py as a numpy array
    /// of Python `str` with that shape.
    /// [`set_attr_string_array`](Self::set_attr_string_array) is the 1-D
    /// convenience form.
    pub fn set_attr_string_array_nd(
        &self,
        name: &str,
        values: &[&str],
        shape: &[usize],
    ) -> Result<()> {
        let n: usize = shape.iter().product();
        if values.len() != n {
            return Err(Hdf5Error::InvalidState(format!(
                "attribute '{name}' shape {shape:?} needs {n} elements, got {}",
                values.len()
            )));
        }
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.set_vlen_string_array_attribute(
                    crate::io::writer::AttrTarget::Root,
                    name,
                    values,
                    &dims,
                )?;
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot write in read mode".into())),
        }
    }

    /// Return the names of file-level (root group) attributes.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Reader(reader) => Ok(reader.root_attr_names()),
            _ => Ok(vec![]),
        }
    }

    /// Read a file-level string attribute.
    pub fn attr_string(&self, name: &str) -> Result<String> {
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                let attr = reader
                    .root_attr(name)
                    .ok_or_else(|| Hdf5Error::NotFound(name.to_string()))?
                    .clone();
                Ok(reader.attr_string_value(&attr)?)
            }
            _ => Err(Hdf5Error::InvalidState("not in read mode".into())),
        }
    }

    /// Check if the file is in write/append mode.
    pub fn is_writable(&self) -> bool {
        let inner = borrow_inner(&self.inner);
        matches!(&*inner, H5FileInner::Writer(_))
    }

    /// Create a variable-length string dataset and write data.
    ///
    /// This is a convenience method for writing h5py-compatible vlen string
    /// datasets using global heap storage.
    pub fn write_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<H5Dataset> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_string_dataset(name, strings)?;
                // If the name contains '/', assign the dataset to its parent group
                if let Some(slash_pos) = name.rfind('/') {
                    let group_path = &name[..slash_pos];
                    let abs_group_path = if group_path.starts_with('/') {
                        group_path.to_string()
                    } else {
                        format!("/{}", group_path)
                    };
                    writer.assign_dataset_to_group(&abs_group_path, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.inner),
                    idx,
                    shape,
                    element_size,
                    chunked,
                    btree2,
                    fixed_array,
                ))
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Create a variable-length byte-array dataset and write data.
    ///
    /// Each `&[u8]` becomes one element of variable length, stored as a vlen
    /// sequence of `u8` in global heap storage. h5py reads it back as an array
    /// of `uint8` arrays. Returns a writer-mode handle so attributes can be
    /// attached, like [`write_vlen_strings`](Self::write_vlen_strings).
    pub fn write_vlen_bytes(&self, name: &str, items: &[&[u8]]) -> Result<H5Dataset> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_bytes_dataset(name, items)?;
                // If the name contains '/', assign the dataset to its parent group
                if let Some(slash_pos) = name.rfind('/') {
                    let group_path = &name[..slash_pos];
                    let abs_group_path = if group_path.starts_with('/') {
                        group_path.to_string()
                    } else {
                        format!("/{}", group_path)
                    };
                    writer.assign_dataset_to_group(&abs_group_path, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.inner),
                    idx,
                    shape,
                    element_size,
                    chunked,
                    btree2,
                    fixed_array,
                ))
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Create a chunked, compressed variable-length string dataset.
    ///
    /// Like `write_vlen_strings`, but stores the vlen references in chunked
    /// layout with the given filter pipeline (e.g., `FilterPipeline::deflate(6)`
    /// or `FilterPipeline::zstd(3)`). `chunk_size` is the number of strings
    /// per chunk.
    pub fn write_vlen_strings_compressed(
        &self,
        name: &str,
        strings: &[&str],
        chunk_size: usize,
        pipeline: FilterPipeline,
    ) -> Result<H5Dataset> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer
                    .create_vlen_string_dataset_compressed(name, strings, chunk_size, pipeline)?;
                if let Some(slash_pos) = name.rfind('/') {
                    let group_path = &name[..slash_pos];
                    let abs_group_path = if group_path.starts_with('/') {
                        group_path.to_string()
                    } else {
                        format!("/{}", group_path)
                    };
                    writer.assign_dataset_to_group(&abs_group_path, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.inner),
                    idx,
                    shape,
                    element_size,
                    chunked,
                    btree2,
                    fixed_array,
                ))
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Create an empty chunked vlen string dataset ready for incremental appends.
    ///
    /// Use `append_vlen_strings` to add data. If `pipeline` is `Some`, chunks
    /// are compressed (e.g., `Some(FilterPipeline::lz4())`).
    pub fn create_appendable_vlen_dataset(
        &self,
        name: &str,
        chunk_size: usize,
        pipeline: Option<FilterPipeline>,
    ) -> Result<H5Dataset> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx =
                    writer.create_appendable_vlen_string_dataset(name, chunk_size, pipeline)?;
                if let Some(slash_pos) = name.rfind('/') {
                    let group_path = &name[..slash_pos];
                    let abs_group_path = if group_path.starts_with('/') {
                        group_path.to_string()
                    } else {
                        format!("/{}", group_path)
                    };
                    writer.assign_dataset_to_group(&abs_group_path, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.inner),
                    idx,
                    shape,
                    element_size,
                    chunked,
                    btree2,
                    fixed_array,
                ))
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Append variable-length strings to an existing chunked vlen string dataset.
    pub fn append_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<()> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let ds_index = writer.open_dataset_index(name)?;
                writer.append_vlen_strings(ds_index, strings)?;
                Ok(())
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Delete a dataset name, with libhdf5's `H5Ldelete` semantics: a
    /// path naming a hard link removes just that link, and if a hard link
    /// still names the object whose tree name is deleted, the dataset
    /// lives on under the link. Deleting the last name unlinks the
    /// dataset on close and the file space it owned — data blocks,
    /// chunk-index structures, and the global-heap objects of
    /// variable-length values — is freed for reuse by later writes in this
    /// session (the file itself does not shrink).
    pub fn delete_dataset(&self, name: &str) -> Result<()> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.delete_dataset(name)?;
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot delete in read mode".into())),
        }
    }

    /// Delete a group and all its child datasets/sub-groups, freeing their
    /// file space the way [`delete_dataset`](Self::delete_dataset) does.
    ///
    /// Hard links reaching in from outside the deleted subtree keep their
    /// targets alive: a dataset or group named by such a link survives
    /// under the link's path (a group brings its whole subtree with it),
    /// and a `name` that is itself a hard link's path removes just that
    /// link.
    pub fn delete_group(&self, name: &str) -> Result<()> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.delete_group(name)?;
                Ok(())
            }
            _ => Err(Hdf5Error::InvalidState("cannot delete in read mode".into())),
        }
    }

    /// Open an existing dataset by name (read mode).
    pub fn dataset(&self, name: &str) -> Result<H5Dataset> {
        // Mutable: a name that crosses an external link opens the file that
        // link names, and the reader caches that handle for the next one.
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                // The reader's open gate reports *why* a name cannot be
                // opened — a dangling soft link and an unsupported object are
                // both present in the listing, and neither is an absence.
                let info = reader.open_dataset(name)?;
                let shape: Vec<usize> = info.dataspace.dims.iter().map(|&d| d as usize).collect();
                let element_size = info.datatype.element_size() as usize;
                Ok(H5Dataset::new_reader(
                    clone_inner(&self.inner),
                    name.to_string(),
                    shape,
                    element_size,
                ))
            }
            H5FileInner::Writer(_) => Err(Hdf5Error::InvalidState(
                "cannot open a dataset by name in write mode; use new_dataset() instead"
                    .to_string(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".to_string())),
        }
    }

    /// Reopen an existing dataset by name in write mode.
    ///
    /// [`dataset`](Self::dataset) only works in read mode; in write mode a
    /// dataset is normally created via [`new_dataset`](Self::new_dataset).
    /// This returns a write-mode handle to a dataset created earlier in the
    /// same session, so you can attach attributes or append chunks to it
    /// without keeping the original handle around — e.g. to flush cached
    /// first/last values onto a dataset at file-close time.
    ///
    /// # Errors
    ///
    /// Returns [`Hdf5Error::NotFound`] if no live dataset with that name
    /// exists, and an error in read mode (use [`dataset`](Self::dataset)).
    pub fn dataset_writer(&self, name: &str) -> Result<H5Dataset> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let index = writer.open_dataset_index(name)?;
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(index);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.inner),
                    index,
                    shape,
                    element_size,
                    chunked,
                    btree2,
                    fixed_array,
                ))
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot open a dataset_writer in read mode; use dataset() instead".to_string(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".to_string())),
        }
    }

    /// Return the names of all datasets in the root group.
    ///
    /// Works in both read and write mode: in write mode, returns the names of
    /// datasets created so far; in read mode, returns the names discovered
    /// during file open.
    pub fn dataset_names(&self) -> Vec<String> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Reader(reader) => reader
                .dataset_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            H5FileInner::Writer(writer) => writer
                .dataset_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            H5FileInner::Closed => Vec::new(),
        }
    }

    /// The paths of every committed (named) datatype in this file.
    ///
    /// A committed datatype is an object in its own right, in neither
    /// [`dataset_names`](Self::dataset_names) nor the group listing. Write
    /// mode answers empty: this crate does not commit types.
    pub fn named_datatype_names(&self) -> Vec<String> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Reader(reader) => reader
                .named_datatype_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            H5FileInner::Writer(_) | H5FileInner::Closed => Vec::new(),
        }
    }

    /// Open a committed (named) datatype by path (read mode).
    ///
    /// The handle opens whenever the object is there; a type this crate
    /// cannot decode reports why from
    /// [`H5NamedDatatype::datatype`](crate::named_datatype::H5NamedDatatype::datatype),
    /// so its attributes stay reachable.
    ///
    /// # Errors
    ///
    /// [`Hdf5Error::NotFound`] when no committed datatype is at that path.
    pub fn named_datatype(&self, path: &str) -> Result<crate::H5NamedDatatype> {
        // Mutable: a path that crosses an external link opens the file that
        // link names, and the reader caches that handle for the next one.
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                reader.named_datatype_info(path)?;
                drop(inner);
                Ok(crate::H5NamedDatatype::new_reader(
                    clone_inner(&self.inner),
                    path.to_string(),
                ))
            }
            H5FileInner::Writer(_) => Err(Hdf5Error::InvalidState(
                "committed datatypes are readable only in read mode".to_string(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".to_string())),
        }
    }

    /// Explicitly close the file. For a writer, this finalizes the file
    /// (writes superblock, headers, etc.). For a reader, this is a no-op.
    ///
    /// The file is also auto-finalized on drop, but calling `close()` lets
    /// you handle errors.
    pub fn close(self) -> Result<()> {
        let old = {
            let mut inner = borrow_inner_mut(&self.inner);
            std::mem::replace(&mut *inner, H5FileInner::Closed)
        };
        match old {
            H5FileInner::Writer(writer) => {
                writer.close()?;
                Ok(())
            }
            H5FileInner::Reader(_) => Ok(()),
            H5FileInner::Closed => Ok(()),
        }
    }

    /// Close the file without a final `fsync` (write mode only).
    ///
    /// Like [`close`](Self::close), this finalizes the file — object headers
    /// and superblock are written, so on return it is a complete, valid HDF5
    /// file readable by any process — but the trailing `sync_all` (fsync) is
    /// skipped. The bytes are handed to the OS but are not guaranteed durable
    /// against power loss or an OS crash until the OS flushes its page cache.
    ///
    /// This trades durability for speed (the fsync typically dominates close
    /// latency); use it for bulk output that can be regenerated. Prefer
    /// [`close`](Self::close) when durability matters. Dropping the file
    /// without calling either finalizes durably.
    ///
    /// For a reader or an already-closed file this is a no-op, matching
    /// [`close`](Self::close).
    pub fn close_no_sync(self) -> Result<()> {
        let old = {
            let mut inner = borrow_inner_mut(&self.inner);
            std::mem::replace(&mut *inner, H5FileInner::Closed)
        };
        match old {
            H5FileInner::Writer(writer) => {
                writer.close_no_sync()?;
                Ok(())
            }
            H5FileInner::Reader(_) => Ok(()),
            H5FileInner::Closed => Ok(()),
        }
    }

    /// Flush the file to disk. Only meaningful in write mode.
    pub fn flush(&self) -> Result<()> {
        // The underlying writer does not expose a standalone flush; data is
        // written to disk immediately via pwrite. This is a compatibility
        // method that does nothing for now.
        Ok(())
    }
}

/// Builder controlling how an [`H5File`] is opened.
///
/// The default policy follows the HDF5 C library: an exclusive lock is
/// acquired for write-mode opens and a shared lock for read-mode opens,
/// honoring the `HDF5_USE_FILE_LOCKING` environment variable. Calling
/// [`Self::locking`] overrides the env-var value.
#[derive(Debug, Default, Clone)]
pub struct H5FileOptions {
    locking: Option<FileLocking>,
}

impl H5FileOptions {
    /// Construct a fresh options builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the locking policy. Bypasses the `HDF5_USE_FILE_LOCKING`
    /// environment variable for the resulting open call.
    pub fn locking(mut self, policy: FileLocking) -> Self {
        self.locking = Some(policy);
        self
    }

    /// Disable OS-level file locking entirely (equivalent to
    /// `HDF5_USE_FILE_LOCKING=FALSE`).
    pub fn no_locking(self) -> Self {
        self.locking(FileLocking::Disabled)
    }

    /// Try to acquire the lock but do not fail if the filesystem rejects it
    /// (equivalent to `HDF5_USE_FILE_LOCKING=BEST_EFFORT`).
    pub fn best_effort_locking(self) -> Self {
        self.locking(FileLocking::BestEffort)
    }

    fn resolved_locking(&self) -> FileLocking {
        match self.locking {
            Some(p) => p,
            None => FileLocking::from_env_or(FileLocking::default()),
        }
    }

    /// Create a new HDF5 file at `path` with the configured options.
    pub fn create<P: AsRef<Path>>(self, path: P) -> Result<H5File> {
        let writer = Hdf5Writer::create_with_locking(path.as_ref(), self.resolved_locking())?;
        Ok(H5File {
            inner: new_shared(H5FileInner::Writer(writer)),
        })
    }

    /// Open an existing HDF5 file for reading with the configured options.
    pub fn open<P: AsRef<Path>>(self, path: P) -> Result<H5File> {
        let reader = Hdf5Reader::open_with_locking(path.as_ref(), self.resolved_locking())?;
        Ok(H5File {
            inner: new_shared(H5FileInner::Reader(reader)),
        })
    }

    /// Open an existing HDF5 file for read/write with the configured options.
    pub fn open_rw<P: AsRef<Path>>(self, path: P) -> Result<H5File> {
        let writer = Hdf5Writer::open_append_with_locking(path.as_ref(), self.resolved_locking())?;
        Ok(H5File {
            inner: new_shared(H5FileInner::Writer(writer)),
        })
    }
}

#[cfg(test)]
fn unique_test_path(name: &str) -> std::path::PathBuf {
    // PID + atomic counter so each test invocation uses a distinct path,
    // preventing collisions across concurrent cargo runs and any
    // flock/LockFileEx race where a previous close()'d file's lock
    // remains briefly visible when reopening the same path.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rust_hdf5_test_{}_{}_{}.h5",
        name,
        std::process::id(),
        n
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        super::unique_test_path(name)
    }

    #[test]
    fn create_and_close_empty() {
        let path = temp_path("create_empty");
        let file = H5File::create(&path).unwrap();
        file.close().unwrap();

        // Should be readable
        let file = H5File::open(&path).unwrap();
        file.close().unwrap();

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_and_drop_empty() {
        let path = temp_path("drop_empty");
        {
            let _file = H5File::create(&path).unwrap();
            // drop auto-finalizes
        }
        // Verify the file is valid by opening it
        let file = H5File::open(&path).unwrap();
        file.close().unwrap();

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dataset_not_found() {
        let path = temp_path("ds_not_found");
        {
            let _file = H5File::create(&path).unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let result = file.dataset("nonexistent");
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_and_read_roundtrip() {
        let path = temp_path("write_read_rt");

        // Write
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<u8>()
                .shape([4, 4])
                .create("data")
                .unwrap();
            ds.write_raw(&[0u8; 16]).unwrap();
            file.close().unwrap();
        }

        // Read
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![4, 4]);
            let data = ds.read_raw::<u8>().unwrap();
            assert_eq!(data.len(), 16);
            assert!(data.iter().all(|&b| b == 0));
            file.close().unwrap();
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_no_sync_produces_valid_readable_file() {
        let path = temp_path("close_no_sync_rt");
        let payload: Vec<u8> = (0u8..16).collect();

        // Write and finalize WITHOUT the trailing fsync.
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<u8>()
                .shape([4, 4])
                .create("data")
                .unwrap();
            ds.write_raw(&payload).unwrap();
            // The only difference from `write_and_read_roundtrip`: no fsync.
            // The file must still be a complete, valid, readable HDF5 file.
            file.close_no_sync().unwrap();
        }

        // Reopen and verify the full content survived (same-machine reader sees
        // the OS page cache regardless of whether fsync ran).
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![4, 4]);
            let data = ds.read_raw::<u8>().unwrap();
            assert_eq!(data, payload);
            file.close().unwrap();
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn create_over_existing_file_truncates() {
        // The create path skips the ftruncate on a brand-new empty file (it
        // arms ext4's auto_da_alloc and turns close(2) into an implicit
        // writeback, defeating close_no_sync). This pins the other side of
        // that guard: creating over an existing non-empty file must still
        // truncate it, so no stale content survives.
        let path = temp_path("create_truncates");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<u8>()
                .shape([4, 4])
                .create("old_data")
                .unwrap();
            ds.write_raw(&[7u8; 16]).unwrap();
            file.close().unwrap();
        }
        assert!(std::fs::metadata(&path).unwrap().len() > 0);

        // Re-create over the non-empty file, write nothing.
        {
            let file = H5File::create(&path).unwrap();
            file.close().unwrap();
        }

        // The old dataset must be gone.
        let file = H5File::open(&path).unwrap();
        assert!(file.dataset("old_data").is_err());
        file.close().unwrap();

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_no_sync_chunked_dataset_valid() {
        // Exercises flush_dataset_synced(sync=false): a chunked (EA-indexed)
        // dataset closed with close_no_sync must skip the per-dataset
        // sync_data yet still write valid index structures, so the reopened
        // file reconstructs every frame.
        let path = temp_path("close_no_sync_chunked");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0usize, 3])
                .chunk(&[1, 3])
                .max_shape(&[None, Some(3)])
                .create("data")
                .unwrap();
            // 10 frames exceeds idx_blk_elmts=4, so data blocks are exercised.
            for frame in 0..10u64 {
                let vals: Vec<i32> = (0..3).map(|i| (frame * 3 + i) as i32).collect();
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk(frame as usize, &raw).unwrap();
            }
            ds.extend(&[10, 3]).unwrap();
            file.close_no_sync().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![10, 3]);
            let data = ds.read_raw::<i32>().unwrap();
            let expected: Vec<i32> = (0..30).collect();
            assert_eq!(data, expected);
            file.close().unwrap();
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_and_read_f64() {
        let path = temp_path("write_read_f64");

        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Write
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([2, 3])
                .create("matrix")
                .unwrap();
            ds.write_raw(&values).unwrap();
            file.close().unwrap();
        }

        // Read
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("matrix").unwrap();
            assert_eq!(ds.shape(), vec![2, 3]);
            let readback = ds.read_raw::<f64>().unwrap();
            assert_eq!(readback, values);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn multiple_datasets() {
        let path = temp_path("multi_ds");

        {
            let file = H5File::create(&path).unwrap();
            let ds1 = file.new_dataset::<i32>().shape([3]).create("ints").unwrap();
            ds1.write_raw(&[10i32, 20, 30]).unwrap();

            let ds2 = file
                .new_dataset::<f32>()
                .shape([2, 2])
                .create("floats")
                .unwrap();
            ds2.write_raw(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();

            let ds_ints = file.dataset("ints").unwrap();
            assert_eq!(ds_ints.shape(), vec![3]);
            let ints = ds_ints.read_raw::<i32>().unwrap();
            assert_eq!(ints, vec![10, 20, 30]);

            let ds_floats = file.dataset("floats").unwrap();
            assert_eq!(ds_floats.shape(), vec![2, 2]);
            let floats = ds_floats.read_raw::<f32>().unwrap();
            assert_eq!(floats, vec![1.0f32, 2.0, 3.0, 4.0]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_is_idempotent() {
        let path = temp_path("close_idemp");
        let file = H5File::create(&path).unwrap();
        file.close().unwrap();
        // File is consumed by close(), so no double-close possible at the type level.
        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        super::unique_test_path(name)
    }

    #[test]
    fn write_file_for_h5dump() {
        let path = temp_path("integration");
        let file = H5File::create(&path).unwrap();

        let ds = file
            .new_dataset::<u8>()
            .shape([4usize, 4])
            .create("data_u8")
            .unwrap();
        let data: Vec<u8> = (0..16).collect();
        ds.write_raw(&data).unwrap();

        let ds2 = file
            .new_dataset::<f64>()
            .shape([3usize, 2])
            .create("data_f64")
            .unwrap();
        let fdata: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        ds2.write_raw(&fdata).unwrap();

        let ds3 = file
            .new_dataset::<i32>()
            .shape([5usize])
            .create("values")
            .unwrap();
        let idata: Vec<i32> = vec![-10, -5, 0, 5, 10];
        ds3.write_raw(&idata).unwrap();

        file.close().unwrap();

        // File exists
        assert!(path.exists());
    }

    #[test]
    fn write_chunked_file_for_h5dump() {
        let path = temp_path("chunked");
        let file = H5File::create(&path).unwrap();

        // Create a chunked dataset with unlimited first dimension
        let ds = file
            .new_dataset::<f64>()
            .shape([0usize, 4])
            .chunk(&[1, 4])
            .max_shape(&[None, Some(4)])
            .create("streaming_data")
            .unwrap();

        // Write 5 frames of data
        for frame in 0..5u64 {
            let values: Vec<f64> = (0..4).map(|i| (frame * 4 + i) as f64).collect();
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            ds.write_chunk(frame as usize, &raw).unwrap();
        }

        // Extend dimensions to reflect the 5 written frames
        ds.extend(&[5, 4]).unwrap();
        ds.flush().unwrap();

        file.close().unwrap();

        assert!(path.exists());
    }

    #[test]
    fn write_chunked_many_frames_for_h5dump() {
        let path = temp_path("chunked_many");
        let file = H5File::create(&path).unwrap();

        let ds = file
            .new_dataset::<i32>()
            .shape([0usize, 3])
            .chunk(&[1, 3])
            .max_shape(&[None, Some(3)])
            .create("data")
            .unwrap();

        // Write 10 frames (exceeds idx_blk_elmts=4, uses data blocks)
        for frame in 0..10u64 {
            let vals: Vec<i32> = (0..3).map(|i| (frame * 3 + i) as i32).collect();
            let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            ds.write_chunk(frame as usize, &raw).unwrap();
        }
        ds.extend(&[10, 3]).unwrap();
        file.close().unwrap();

        assert!(path.exists());
    }

    #[test]
    fn write_dataset_with_attributes() {
        use crate::types::VarLenUnicode;

        let path = temp_path("attributes");
        let file = H5File::create(&path).unwrap();

        let ds = file
            .new_dataset::<f32>()
            .shape([10usize])
            .create("temperature")
            .unwrap();
        let data: Vec<f32> = (0..10).map(|i| i as f32 * 1.5).collect();
        ds.write_raw(&data).unwrap();

        // Add string attributes
        let attr = ds
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("units")
            .unwrap();
        attr.write_scalar(&VarLenUnicode("kelvin".to_string()))
            .unwrap();

        let attr2 = ds
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("description")
            .unwrap();
        attr2
            .write_scalar(&VarLenUnicode("Temperature measurements".to_string()))
            .unwrap();

        // Use write_string convenience method
        let attr3 = ds
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("source")
            .unwrap();
        attr3.write_string("sensor_01").unwrap();

        // Also test parse -> write_scalar pattern
        let attr4 = ds
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create("label")
            .unwrap();
        let s: VarLenUnicode = "test_label".parse().unwrap_or_default();
        attr4.write_scalar(&s).unwrap();

        file.close().unwrap();

        assert!(path.exists());
    }

    #[test]
    fn dataset_writer_reopens_for_attributes() {
        // Reopen a dataset by name in write mode (the original handle is gone)
        // and attach attributes to it — the close-time flush pattern.
        let path = temp_path("dataset_writer");
        {
            let file = H5File::create(&path).unwrap();
            {
                let ds = file
                    .new_dataset::<u16>()
                    .shape([8])
                    .create("image")
                    .unwrap();
                ds.write_raw(&[0u16; 8]).unwrap();
                // original handle dropped here
            }

            // Reopen by name; dataset() would error in write mode.
            assert!(file.dataset("image").is_err());
            let ds = file.dataset_writer("image").unwrap();
            assert_eq!(ds.shape(), vec![8]);
            ds.new_attr::<i32>()
                .shape([3])
                .create("NDArrayDimOffset")
                .unwrap()
                .write_array(&[0i32, 4, 8])
                .unwrap();
            ds.new_attr::<i32>()
                .shape(())
                .create("NDUniqueId")
                .unwrap()
                .write_numeric(&42i32)
                .unwrap();

            // Missing dataset is reported.
            assert!(matches!(
                file.dataset_writer("nope"),
                Err(crate::error::Hdf5Error::NotFound(_))
            ));

            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("image").unwrap();
            let off = ds.attr("NDArrayDimOffset").unwrap().read_raw().unwrap();
            let got: Vec<i32> = off
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(got, vec![0, 4, 8]);
            let uid: i32 = ds.attr("NDUniqueId").unwrap().read_numeric().unwrap();
            assert_eq!(uid, 42);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn chunked_write_read_roundtrip() {
        let path = temp_path("chunked_roundtrip");

        // Write
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0usize, 3])
                .chunk(&[1, 3])
                .max_shape(&[None, Some(3)])
                .create("table")
                .unwrap();

            for frame in 0..8u64 {
                let vals: Vec<i32> = (0..3).map(|i| (frame * 3 + i) as i32).collect();
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk(frame as usize, &raw).unwrap();
            }
            ds.extend(&[8, 3]).unwrap();
            file.close().unwrap();
        }

        // Read
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("table").unwrap();
            assert_eq!(ds.shape(), vec![8, 3]);
            let data = ds.read_raw::<i32>().unwrap();
            assert_eq!(data.len(), 24);
            for (i, val) in data.iter().enumerate() {
                assert_eq!(*val, i as i32);
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn compressed_chunked_roundtrip() {
        let path = temp_path("compressed_roundtrip");

        // Write compressed
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([0usize, 4])
                .chunk(&[1, 4])
                .max_shape(&[None, Some(4)])
                .deflate(6)
                .create("compressed")
                .unwrap();

            for frame in 0..10u64 {
                let vals: Vec<f64> = (0..4).map(|i| (frame * 4 + i) as f64).collect();
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk(frame as usize, &raw).unwrap();
            }
            ds.extend(&[10, 4]).unwrap();
            file.close().unwrap();
        }

        // Read back and verify
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("compressed").unwrap();
            assert_eq!(ds.shape(), vec![10, 4]);
            let data = ds.read_raw::<f64>().unwrap();
            assert_eq!(data.len(), 40);
            for (i, val) in data.iter().enumerate() {
                assert!(
                    (val - i as f64).abs() < 1e-10,
                    "mismatch at {}: {} != {}",
                    i,
                    val,
                    i
                );
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn compressed_chunked_many_frames() {
        let path = temp_path("compressed_many");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0usize, 3])
                .chunk(&[1, 3])
                .max_shape(&[None, Some(3)])
                .deflate(6)
                .create("stream")
                .unwrap();

            for frame in 0..100u64 {
                let vals: Vec<i32> = (0..3).map(|i| (frame * 3 + i) as i32).collect();
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk(frame as usize, &raw).unwrap();
            }
            ds.extend(&[100, 3]).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("stream").unwrap();
            assert_eq!(ds.shape(), vec![100, 3]);
            let data = ds.read_raw::<i32>().unwrap();
            assert_eq!(data.len(), 300);
            for (i, val) in data.iter().enumerate() {
                assert_eq!(*val, i as i32, "mismatch at {}", i);
            }
        }

        std::fs::remove_file(&path).ok();
    }
    #[test]
    fn append_mode() {
        let path = temp_path("append");

        // Create initial file
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([3usize])
                .create("first")
                .unwrap();
            ds.write_raw(&[1i32, 2, 3]).unwrap();
            file.close().unwrap();
        }

        // Append new dataset
        {
            let file = H5File::open_rw(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([2usize])
                .create("second")
                .unwrap();
            ds.write_raw(&[4.0f64, 5.0]).unwrap();
            file.close().unwrap();
        }

        // Read back both
        {
            let file = H5File::open(&path).unwrap();
            let names = file.dataset_names();
            assert!(names.contains(&"first".to_string()));
            assert!(names.contains(&"second".to_string()));

            let ds1 = file.dataset("first").unwrap();
            assert_eq!(ds1.read_raw::<i32>().unwrap(), vec![1, 2, 3]);

            let ds2 = file.dataset("second").unwrap();
            assert_eq!(ds2.read_raw::<f64>().unwrap(), vec![4.0, 5.0]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rw_set_attr_preserves_file() {
        let path = temp_path("open_rw_attr");
        // Create file with a dataset and an attribute
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([3usize])
                .create("data")
                .unwrap();
            ds.write_raw(&[10i32, 20, 30]).unwrap();
            file.set_attr_string("version", "1.0").unwrap();
            file.close().unwrap();
        }
        // Open rw and modify the attribute
        {
            let file = H5File::open_rw(&path).unwrap();
            file.set_attr_string("version", "2.0").unwrap();
            file.close().unwrap();
        }
        // Verify: dataset intact, attribute updated
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.read_raw::<i32>().unwrap(), vec![10, 20, 30]);
            let ver = file.attr_string("version").unwrap();
            assert_eq!(ver, "2.0");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn open_rw_attr_with_compressed_dataset() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("open_rw_compressed");
        let input: Vec<&str> = (0..50).map(|_| "test string data").collect();
        // Create file with compressed vlen strings
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings_compressed("texts", &input, 16, FilterPipeline::deflate(6))
                .unwrap();
            file.set_attr_string("version", "1.0").unwrap();
            file.close().unwrap();
        }
        // Open rw and modify attribute only
        {
            let file = H5File::open_rw(&path).unwrap();
            file.set_attr_string("version", "2.0").unwrap();
            file.close().unwrap();
        }
        // Verify: compressed dataset still readable, attribute updated
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("texts").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings.len(), 50);
            assert_eq!(strings[0], "test string data");
            let ver = file.attr_string("version").unwrap();
            assert_eq!(ver, "2.0");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn append_vlen_strings_basic() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("append_vlen");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("names", 4, Some(FilterPipeline::lz4()))
                .unwrap();
            file.append_vlen_strings("names", &["alice", "bob", "charlie"])
                .unwrap();
            file.append_vlen_strings("names", &["dave", "eve"]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("names").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings, vec!["alice", "bob", "charlie", "dave", "eve"]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn append_vlen_strings_large() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("append_vlen_large");
        let batch1: Vec<String> = (0..5000).map(|i| format!("node-{:06}", i)).collect();
        let batch2: Vec<String> = (5000..7189).map(|i| format!("node-{:06}", i)).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("data", 512, Some(FilterPipeline::lz4()))
                .unwrap();
            let r1: Vec<&str> = batch1.iter().map(|s| s.as_str()).collect();
            file.append_vlen_strings("data", &r1).unwrap();
            let r2: Vec<&str> = batch2.iter().map(|s| s.as_str()).collect();
            file.append_vlen_strings("data", &r2).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings.len(), 7189);
            assert_eq!(strings[0], "node-000000");
            assert_eq!(strings[7188], "node-007188");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_vlen_strings_uncompressed() {
        let path = temp_path("append_vlen_unc");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("texts", 8, None)
                .unwrap();
            file.append_vlen_strings("texts", &["hello", "world"])
                .unwrap();
            file.append_vlen_strings("texts", &["foo", "bar", "baz"])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("texts").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings, vec!["hello", "world", "foo", "bar", "baz"]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_dataset_roundtrip() {
        let path = temp_path("delete_ds");
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("keep", &["a", "b"]).unwrap();
            file.write_vlen_strings("remove", &["x", "y"]).unwrap();
            file.delete_dataset("remove").unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let names = file.dataset_names();
            assert!(names.contains(&"keep".to_string()));
            assert!(!names.contains(&"remove".to_string()));
            let ds = file.dataset("keep").unwrap();
            assert_eq!(ds.read_vlen_strings().unwrap(), vec!["a", "b"]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_group_roundtrip() {
        let path = temp_path("delete_grp");
        {
            let file = H5File::create(&path).unwrap();
            let g1 = file.create_group("keep").unwrap();
            g1.write_vlen_strings("data", &["a"]).unwrap();
            let g2 = file.create_group("remove").unwrap();
            g2.write_vlen_strings("data", &["x"]).unwrap();
            file.delete_group("remove").unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let names = file.dataset_names();
            assert!(names.contains(&"keep/data".to_string()));
            assert!(!names.contains(&"remove/data".to_string()));
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rw_delete_recreate_group() {
        let path = temp_path("rw_delete_recreate");
        // Step 1: create file with groups
        {
            let file = H5File::create(&path).unwrap();
            let n = file.create_group("nodes").unwrap();
            n.write_vlen_strings("id", &["a", "b", "c"]).unwrap();
            let e = file.create_group("edges").unwrap();
            e.write_vlen_strings("src", &["x", "y"]).unwrap();
            file.close().unwrap();
        }
        // Step 2: open_rw, delete one group, recreate with new data
        {
            let file = H5File::open_rw(&path).unwrap();
            file.delete_group("nodes").unwrap();
            let n = file.create_group("nodes").unwrap();
            n.write_vlen_strings("id", &["new1", "new2"]).unwrap();
            file.close().unwrap();
        }
        // Step 3: verify
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("nodes/id").unwrap();
            let s = ds.read_vlen_strings().unwrap();
            assert_eq!(s, vec!["new1", "new2"]);
            // edges should still be intact
            let ds = file.dataset("edges/src").unwrap();
            let s = ds.read_vlen_strings().unwrap();
            assert_eq!(s, vec!["x", "y"]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_and_recreate_group() {
        let path = temp_path("delete_recreate");
        {
            let file = H5File::create(&path).unwrap();
            let g = file.create_group("nodes").unwrap();
            g.write_vlen_strings("id", &["old1", "old2"]).unwrap();
            file.delete_group("nodes").unwrap();
            let g = file.create_group("nodes").unwrap();
            g.write_vlen_strings("id", &["new1", "new2", "new3"])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("nodes/id").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings, vec!["new1", "new2", "new3"]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn vlen_string_compressed_large_roundtrip() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("vlen_large");
        // Simulate kodex scenario: 7189 strings, chunk_size 512
        let input: Vec<String> = (0..7189)
            .map(|i| format!("node-{:08x}-{}", i, "a".repeat(20 + (i % 30))))
            .collect();
        let input_refs: Vec<&str> = input.iter().map(|s| s.as_str()).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.create_group("nodes").unwrap();
            file.write_vlen_strings_compressed(
                "nodes/id",
                &input_refs,
                512,
                FilterPipeline::deflate(6),
            )
            .unwrap();
            file.close().unwrap();
        }
        // Read back
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("nodes/id").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings.len(), 7189);
            assert_eq!(strings[0], input[0]);
            assert_eq!(strings[7188], input[7188]);
        }
        // Also test open_rw then re-read
        {
            let file = H5File::open_rw(&path).unwrap();
            file.set_attr_string("version", "1.0").unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("nodes/id").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings.len(), 7189);
            assert_eq!(strings[0], input[0]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_string_write_read() {
        let path = temp_path("vlen_wr");
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("names", &["alice", "bob", "charlie"])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("names").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings, vec!["alice", "bob", "charlie"]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_bytes_write_read() {
        let path = temp_path("vlen_bytes_wr");
        let items: [&[u8]; 4] = [b"abc", b"", &[0u8, 1, 2, 255], b"hi"];
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_bytes("blobs", &items).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("blobs").unwrap();
            let got = ds.read_vlen_bytes().unwrap();
            let expected: Vec<Vec<u8>> = items.iter().map(|s| s.to_vec()).collect();
            assert_eq!(got, expected);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_bytes_in_group_with_attribute() {
        use crate::types::VarLenUnicode;
        let path = temp_path("vlen_bytes_grp");
        let items: [&[u8]; 2] = [&[1u8, 2, 3], &[9u8, 8, 7, 6]];
        {
            let file = H5File::create(&path).unwrap();
            let grp = file.root_group().create_group("payloads").unwrap();
            let ds = grp.write_vlen_bytes("frames", &items).unwrap();
            ds.new_attr::<VarLenUnicode>()
                .shape(())
                .create("codec")
                .unwrap()
                .write_string("raw")
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("payloads/frames").unwrap();
            let got = ds.read_vlen_bytes().unwrap();
            let expected: Vec<Vec<u8>> = items.iter().map(|s| s.to_vec()).collect();
            assert_eq!(got, expected);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_dataset_returns_handle_for_attributes() {
        use crate::types::VarLenUnicode;
        let path = temp_path("vlen_attr");
        {
            let file = H5File::create(&path).unwrap();
            let grp = file.root_group().create_group("ch").unwrap();
            // The vlen helper now returns the dataset handle, so attributes can
            // be attached directly — the issue the mdfr reporter hit.
            let ds = grp
                .write_vlen_strings("labels", &["a", "bb", "ccc"])
                .unwrap();
            ds.new_attr::<VarLenUnicode>()
                .shape(())
                .create("unit")
                .unwrap()
                .write_string("volt")
                .unwrap();
            // The same dataset can also be reopened by name within the group.
            let ds2 = grp.dataset_writer("labels").unwrap();
            ds2.new_attr::<VarLenUnicode>()
                .shape(())
                .create("desc")
                .unwrap()
                .write_string("channel labels")
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("ch/labels").unwrap();
            assert_eq!(ds.read_vlen_strings().unwrap(), vec!["a", "bb", "ccc"]);
            assert_eq!(ds.attr("unit").unwrap().read_string().unwrap(), "volt");
            assert_eq!(
                ds.attr("desc").unwrap().read_string().unwrap(),
                "channel labels"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn vlen_string_deflate_roundtrip() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("vlen_deflate");
        let input: Vec<&str> = (0..100)
            .map(|i| match i % 3 {
                0 => "hello world",
                1 => "compressed vlen string test",
                _ => "rust-hdf5",
            })
            .collect();
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings_compressed("texts", &input, 16, FilterPipeline::deflate(6))
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("texts").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings.len(), 100);
            for (i, s) in strings.iter().enumerate() {
                assert_eq!(s, input[i]);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "zstd")]
    fn vlen_string_zstd_roundtrip() {
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("vlen_zstd");
        let input: Vec<&str> = (0..200)
            .map(|i| match i % 4 {
                0 => "zstandard compression test",
                1 => "variable length string",
                2 => "rust-hdf5 chunked storage",
                _ => "hello zstd world",
            })
            .collect();
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings_compressed("data", &input, 32, FilterPipeline::zstd(3))
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            let strings = ds.read_vlen_strings().unwrap();
            assert_eq!(strings.len(), 200);
            for (i, s) in strings.iter().enumerate() {
                assert_eq!(s, input[i]);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn shuffle_deflate_roundtrip() {
        let path = temp_path("shuf_defl");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([0usize, 4])
                .chunk(&[1, 4])
                .max_shape(&[None, Some(4)])
                .shuffle_deflate(6)
                .create("data")
                .unwrap();
            for frame in 0..20u64 {
                let vals: Vec<f64> = (0..4).map(|i| (frame * 4 + i) as f64).collect();
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk(frame as usize, &raw).unwrap();
            }
            ds.extend(&[20, 4]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![20, 4]);
            let data = ds.read_raw::<f64>().unwrap();
            assert_eq!(data.len(), 80);
            for (i, val) in data.iter().enumerate() {
                assert!((val - i as f64).abs() < 1e-10);
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_level_attributes() {
        let path = temp_path("file_attr");
        {
            let file = H5File::create(&path).unwrap();
            file.set_attr_string("title", "Test File").unwrap();
            file.set_attr_numeric("version", &42i32).unwrap();
            let ds = file
                .new_dataset::<u8>()
                .shape([1usize])
                .create("dummy")
                .unwrap();
            ds.write_raw(&[0u8]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            assert!(file.dataset_names().contains(&"dummy".to_string()));

            // Read file-level attributes
            let names = file.attr_names().unwrap();
            assert!(names.contains(&"title".to_string()));

            let title = file.attr_string("title").unwrap();
            assert_eq!(title, "Test File");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn scalar_dataset_roundtrip() {
        let path = temp_path("scalar");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f64>().scalar().create("pi").unwrap();
            ds.write_raw(&[std::f64::consts::PI]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("pi").unwrap();
            assert_eq!(ds.shape(), Vec::<usize>::new());
            assert_eq!(ds.total_elements(), 1);
            let data = ds.read_raw::<f64>().unwrap();
            assert_eq!(data.len(), 1);
            assert!((data[0] - std::f64::consts::PI).abs() < 1e-15);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_mode_extend_chunked() {
        let path = temp_path("append_extend");

        // Create with 5 frames
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0usize, 3])
                .chunk(&[1, 3])
                .max_shape(&[None, Some(3)])
                .create("stream")
                .unwrap();
            for i in 0..5u64 {
                let vals: Vec<i32> = (0..3).map(|j| (i * 3 + j) as i32).collect();
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk(i as usize, &raw).unwrap();
            }
            ds.extend(&[5, 3]).unwrap();
            file.close().unwrap();
        }

        // Reopen and add 5 more frames
        {
            let file = H5File::open_rw(&path).unwrap();
            // Find the stream dataset index (it's the first one)
            let names = file.dataset_names();
            assert!(names.contains(&"stream".to_string()));

            // Write more chunks via the writer directly
            let mut inner = crate::file::borrow_inner_mut(&file.inner);
            if let crate::file::H5FileInner::Writer(writer) = &mut *inner {
                let ds_idx = writer.dataset_index("stream").unwrap();
                for i in 5..10u64 {
                    let vals: Vec<i32> = (0..3).map(|j| (i * 3 + j) as i32).collect();
                    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                    writer.write_chunk(ds_idx, i, &raw).unwrap();
                }
                writer.extend_dataset(ds_idx, &[10, 3]).unwrap();
            }
            drop(inner);
            file.close().unwrap();
        }

        // Read back all 10 frames
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("stream").unwrap();
            assert_eq!(ds.shape(), vec![10, 3]);
            let data = ds.read_raw::<i32>().unwrap();
            assert_eq!(data.len(), 30);
            for (i, val) in data.iter().enumerate() {
                assert_eq!(*val, i as i32, "mismatch at {}", i);
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn group_hierarchy_roundtrip() {
        let path = temp_path("groups_rt");

        {
            let file = H5File::create(&path).unwrap();
            let root = file.root_group();

            // Create groups
            let det = root.create_group("detector").unwrap();
            let raw = det.create_group("raw").unwrap();

            // Create datasets in groups
            let ds1 = det
                .new_dataset::<f32>()
                .shape([10usize])
                .create("temperature")
                .unwrap();
            ds1.write_raw(&[1.0f32; 10]).unwrap();

            let ds2 = raw
                .new_dataset::<u16>()
                .shape([4usize, 4])
                .create("image")
                .unwrap();
            ds2.write_raw(&[42u16; 16]).unwrap();

            // Root-level dataset
            let ds3 = file
                .new_dataset::<i32>()
                .shape([3usize])
                .create("version")
                .unwrap();
            ds3.write_raw(&[1i32, 0, 0]).unwrap();

            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let names = file.dataset_names();
            assert!(names.contains(&"version".to_string()));
            assert!(names.contains(&"detector/temperature".to_string()));
            assert!(names.contains(&"detector/raw/image".to_string()));

            // Read datasets
            let ds = file.dataset("version").unwrap();
            assert_eq!(ds.read_raw::<i32>().unwrap(), vec![1, 0, 0]);

            let ds = file.dataset("detector/temperature").unwrap();
            assert_eq!(ds.read_raw::<f32>().unwrap(), vec![1.0f32; 10]);

            let ds = file.dataset("detector/raw/image").unwrap();
            assert_eq!(ds.shape(), vec![4, 4]);
            assert_eq!(ds.read_raw::<u16>().unwrap(), vec![42u16; 16]);

            // Group traversal
            let root = file.root_group();
            let group_names = root.group_names().unwrap();
            assert!(group_names.contains(&"detector".to_string()));
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nested_groups_via_file_create_group() {
        let path = temp_path("file_create_group");

        {
            let file = H5File::create(&path).unwrap();

            // Use the H5File::create_group convenience method
            let grp = file.create_group("sensors").unwrap();
            let sub = grp.create_group("accel").unwrap();

            let ds = sub
                .new_dataset::<f64>()
                .shape([3usize])
                .create("xyz")
                .unwrap();
            ds.write_raw(&[1.0f64, 2.0, 3.0]).unwrap();

            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let names = file.dataset_names();
            assert!(names.contains(&"sensors/accel/xyz".to_string()));

            let ds = file.dataset("sensors/accel/xyz").unwrap();
            assert_eq!(ds.read_raw::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);

            // Open group in read mode
            let root = file.root_group();
            let sensors = root.group("sensors").unwrap();
            assert_eq!(sensors.name(), "/sensors");

            let accel = sensors.group("accel").unwrap();
            assert_eq!(accel.name(), "/sensors/accel");

            // list_groups from root
            let top_groups = root.group_names().unwrap();
            assert!(top_groups.contains(&"sensors".to_string()));

            // list_groups from sensors
            let sub_groups = sensors.group_names().unwrap();
            assert!(sub_groups.contains(&"accel".to_string()));
        }

        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod h5py_compat_tests {
    use super::*;

    // Only the deflate-gated h5dump test creates files; without that
    // feature this helper has no callers.
    #[cfg(feature = "deflate")]
    fn temp_path(name: &str) -> std::path::PathBuf {
        super::unique_test_path(name)
    }

    /// Verify our files can be read by h5dump (if available).
    #[test]
    #[cfg(feature = "deflate")]
    fn h5dump_validates_our_files() {
        // Check if h5dump is available
        let h5dump = std::process::Command::new("h5dump")
            .arg("--version")
            .output();
        if h5dump.is_err() {
            eprintln!("skipping: h5dump not found");
            return;
        }

        let path = temp_path("h5dump_validate");

        // Write a comprehensive test file
        {
            let file = H5File::create(&path).unwrap();

            // Contiguous
            let ds = file
                .new_dataset::<f64>()
                .shape([3usize, 4])
                .create("matrix")
                .unwrap();
            let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
            ds.write_raw(&data).unwrap();

            // Chunked + compressed
            let ds2 = file
                .new_dataset::<i32>()
                .shape([0usize, 2])
                .chunk(&[1, 2])
                .max_shape(&[None, Some(2)])
                .deflate(6)
                .create("stream")
                .unwrap();
            for i in 0..5u64 {
                let vals: Vec<i32> = vec![i as i32 * 2, i as i32 * 2 + 1];
                let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                ds2.write_chunk(i as usize, &raw).unwrap();
            }
            ds2.extend(&[5, 2]).unwrap();

            // Group
            let grp = file.create_group("meta").unwrap();
            let ds3 = grp
                .new_dataset::<u8>()
                .shape([4usize])
                .create("flags")
                .unwrap();
            ds3.write_raw(&[1u8, 0, 1, 0]).unwrap();

            // String attribute
            use crate::types::VarLenUnicode;
            let attr = ds
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("units")
                .unwrap();
            attr.write_string("meters").unwrap();

            file.close().unwrap();
        }

        // Run h5dump and verify exit code
        let output = std::process::Command::new("h5dump")
            .arg("-H") // header only (faster)
            .arg(path.to_str().unwrap())
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "h5dump failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );

        // Full dump (with data) should also work
        let output2 = std::process::Command::new("h5dump")
            .arg(path.to_str().unwrap())
            .output()
            .unwrap();

        assert!(
            output2.status.success(),
            "h5dump (full) failed:\nstderr: {}",
            String::from_utf8_lossy(&output2.stderr),
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_h5py_generated_file() {
        let path = "/tmp/test_h5py_default.h5";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: h5py test file not found");
            return;
        }
        let file = H5File::open(path).unwrap();

        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.shape(), vec![4, 5]);
        let data = ds.read_raw::<f64>().unwrap();
        assert_eq!(data.len(), 20);
        assert!((data[0]).abs() < 1e-10);
        assert!((data[19] - 19.0).abs() < 1e-10);

        let ds2 = file.dataset("images").unwrap();
        assert_eq!(ds2.shape(), vec![3, 64, 64]);
        let images = ds2.read_raw::<u16>().unwrap();
        assert_eq!(images.len(), 3 * 64 * 64);
    }
}
