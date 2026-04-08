//! Dataset creation and I/O.
//!
//! Datasets are created via the fluent [`DatasetBuilder`] API obtained from
//! [`H5File::new_dataset`](crate::file::H5File::new_dataset). Once created,
//! the [`H5Dataset`] handle can read or write raw typed data.

use crate::attribute::AttrBuilder;
use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner, borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
use crate::types::H5Type;

// ---------------------------------------------------------------------------
// DatasetBuilder
// ---------------------------------------------------------------------------

/// A fluent builder for creating datasets.
///
/// Obtained from [`H5File::new_dataset::<T>()`](crate::file::H5File::new_dataset).
///
/// ```no_run
/// # use rust_hdf5::H5File;
/// let file = H5File::create("builder.h5").unwrap();
/// let ds = file.new_dataset::<f32>()
///     .shape(&[10, 20])
///     .create("temperatures")
///     .unwrap();
/// ```
pub struct DatasetBuilder<T: H5Type> {
    file_inner: SharedInner,
    shape: Option<Vec<usize>>,
    chunk_dims: Option<Vec<usize>>,
    max_shape: Option<Vec<Option<usize>>>,
    deflate_level: Option<u32>,
    shuffle_deflate_level: Option<u32>,
    custom_pipeline: Option<crate::format::messages::filter::FilterPipeline>,
    group_path: Option<String>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: H5Type> DatasetBuilder<T> {
    pub(crate) fn new(file_inner: SharedInner) -> Self {
        Self {
            file_inner,
            shape: None,
            chunk_dims: None,
            max_shape: None,
            deflate_level: None,
            shuffle_deflate_level: None,
            custom_pipeline: None,
            group_path: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn new_in_group(file_inner: SharedInner, group_path: String) -> Self {
        Self {
            file_inner,
            shape: None,
            chunk_dims: None,
            max_shape: None,
            deflate_level: None,
            shuffle_deflate_level: None,
            custom_pipeline: None,
            group_path: Some(group_path),
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the dataset dimensions.
    ///
    /// This is required before calling [`create`](Self::create).
    /// Use an empty slice `&[]` for a scalar (0-dimensional) dataset.
    #[must_use]
    pub fn shape<S: AsRef<[usize]>>(mut self, dims: S) -> Self {
        self.shape = Some(dims.as_ref().to_vec());
        self
    }

    /// Create a scalar (0-dimensional) dataset holding a single value.
    #[must_use]
    pub fn scalar(mut self) -> Self {
        self.shape = Some(vec![]);
        self
    }

    /// Set chunk dimensions for chunked storage.
    ///
    /// When set, the dataset uses chunked storage with the extensible array
    /// index. You should also call [`max_shape`](Self::max_shape) or
    /// [`resizable`](Self::resizable) to allow extending.
    #[must_use]
    pub fn chunk(mut self, chunk_dims: &[usize]) -> Self {
        self.chunk_dims = Some(chunk_dims.to_vec());
        self
    }

    /// Make all dimensions unlimited (resizable).
    ///
    /// This sets max_dims to u64::MAX for all dimensions.
    #[must_use]
    pub fn resizable(mut self) -> Self {
        self.max_shape = Some(vec![None; self.shape.as_ref().map_or(0, |s| s.len())]);
        self
    }

    /// Set maximum dimensions. `None` means unlimited for that dimension.
    #[must_use]
    pub fn max_shape(mut self, max: &[Option<usize>]) -> Self {
        self.max_shape = Some(max.to_vec());
        self
    }

    /// Enable deflate (gzip) compression with the given level (0-9).
    ///
    /// Requires chunked storage (call `.chunk()` before `.create()`).
    /// Level 0 = no compression, 9 = maximum compression. Default is 6.
    #[must_use]
    pub fn deflate(mut self, level: u32) -> Self {
        self.deflate_level = Some(level);
        self
    }

    /// Enable shuffle + deflate compression.
    ///
    /// Shuffle reorders bytes by position within elements before compression,
    /// which typically improves compression ratios for numeric data.
    /// Requires chunked storage.
    #[must_use]
    pub fn shuffle_deflate(mut self, level: u32) -> Self {
        self.shuffle_deflate_level = Some(level);
        self
    }

    /// Enable Zstandard compression with the given level (1-22, default 3).
    ///
    /// Requires chunked storage (call `.chunk()` before `.create()`).
    #[must_use]
    pub fn zstd(mut self, level: u32) -> Self {
        self.custom_pipeline = Some(crate::format::messages::filter::FilterPipeline::zstd(level));
        self
    }

    /// Set a custom filter pipeline for compression.
    ///
    /// This takes precedence over [`deflate`](Self::deflate) and
    /// [`shuffle_deflate`](Self::shuffle_deflate). Requires chunked storage.
    #[must_use]
    pub fn filter_pipeline(
        mut self,
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> Self {
        self.custom_pipeline = Some(pipeline);
        self
    }

    /// Finalize and create the dataset with the given `name`.
    ///
    /// The name is the link name within the root group (e.g. `"data"` or
    /// `"group1/data"` once nested groups are supported).
    pub fn create(self, name: &str) -> Result<H5Dataset> {
        let shape = self.shape.ok_or_else(|| {
            Hdf5Error::InvalidState("shape must be set before calling create()".into())
        })?;

        // Build the full name: if created within a group, prefix with group path
        let full_name = if let Some(ref gp) = self.group_path {
            if gp == "/" {
                name.to_string()
            } else {
                let trimmed = gp.trim_start_matches('/');
                format!("{}/{}", trimmed, name)
            }
        } else {
            name.to_string()
        };
        let group_path = self.group_path.clone();

        let dims_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let datatype = T::hdf5_type();
        let element_size = T::element_size();

        if let Some(ref chunk_dims) = self.chunk_dims {
            // Chunked dataset
            let chunk_u64: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
            let max_u64: Vec<u64> = if let Some(ref max) = self.max_shape {
                max.iter()
                    .map(|m| m.map_or(u64::MAX, |v| v as u64))
                    .collect()
            } else {
                // Default: max = current
                dims_u64.clone()
            };

            let index = {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        let idx = if let Some(pipeline) = self.custom_pipeline {
                            writer.create_chunked_dataset_with_pipeline(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64, pipeline,
                            )?
                        } else if let Some(level) = self.shuffle_deflate_level {
                            let pipeline =
                                crate::format::messages::filter::FilterPipeline::shuffle_deflate(
                                    T::element_size() as u32,
                                    level,
                                );
                            writer.create_chunked_dataset_with_pipeline(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64, pipeline,
                            )?
                        } else if let Some(level) = self.deflate_level {
                            writer.create_chunked_dataset_compressed(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64, level,
                            )?
                        } else {
                            writer.create_chunked_dataset(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64,
                            )?
                        };
                        if let Some(ref gp) = group_path {
                            if gp != "/" {
                                writer.assign_dataset_to_group(gp, idx)?;
                            }
                        }
                        idx
                    }
                    H5FileInner::Reader(_) => {
                        return Err(Hdf5Error::InvalidState(
                            "cannot create a dataset in read mode".into(),
                        ));
                    }
                    H5FileInner::Closed => {
                        return Err(Hdf5Error::InvalidState("file is closed".into()));
                    }
                }
            };

            Ok(H5Dataset {
                file_inner: clone_inner(&self.file_inner),
                info: DatasetInfo::Writer {
                    index,
                    shape,
                    element_size,
                    chunked: true,
                },
            })
        } else {
            // Contiguous dataset (original path)
            let index = {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        let idx = writer.create_dataset(&full_name, datatype, &dims_u64)?;
                        if let Some(ref gp) = group_path {
                            if gp != "/" {
                                writer.assign_dataset_to_group(gp, idx)?;
                            }
                        }
                        idx
                    }
                    H5FileInner::Reader(_) => {
                        return Err(Hdf5Error::InvalidState(
                            "cannot create a dataset in read mode".into(),
                        ));
                    }
                    H5FileInner::Closed => {
                        return Err(Hdf5Error::InvalidState("file is closed".into()));
                    }
                }
            };

            Ok(H5Dataset {
                file_inner: clone_inner(&self.file_inner),
                info: DatasetInfo::Writer {
                    index,
                    shape,
                    element_size,
                    chunked: false,
                },
            })
        }
    }
}

// ---------------------------------------------------------------------------
// DatasetInfo
// ---------------------------------------------------------------------------

/// Internal metadata about a dataset handle.
enum DatasetInfo {
    /// A dataset created via `new_dataset().create()` in write mode.
    Writer {
        /// Index into the writer's dataset list.
        index: usize,
        /// Shape (current dimensions).
        shape: Vec<usize>,
        /// Size of one element in bytes.
        element_size: usize,
        /// Whether this is a chunked dataset.
        chunked: bool,
    },
    /// A dataset opened by name in read mode.
    Reader {
        /// The link name of the dataset.
        name: String,
        /// Shape (current dimensions).
        shape: Vec<usize>,
        /// Size of one element in bytes.
        element_size: usize,
    },
}

// ---------------------------------------------------------------------------
// H5Dataset
// ---------------------------------------------------------------------------

/// A handle to an HDF5 dataset, supporting typed read and write operations.
///
/// The dataset holds a shared reference to the file's I/O backend, so it
/// remains valid even if the originating [`H5File`](crate::file::H5File) is
/// moved or dropped (they share ownership via `Rc`).
pub struct H5Dataset {
    file_inner: SharedInner,
    info: DatasetInfo,
}

impl H5Dataset {
    /// Create a reader-mode dataset handle (called internally by `H5File::dataset`).
    pub(crate) fn new_reader(
        file_inner: SharedInner,
        name: String,
        shape: Vec<usize>,
        element_size: usize,
    ) -> Self {
        Self {
            file_inner,
            info: DatasetInfo::Reader {
                name,
                shape,
                element_size,
            },
        }
    }

    /// Return the dataset dimensions.
    pub fn shape(&self) -> Vec<usize> {
        match &self.info {
            DatasetInfo::Writer { shape, .. } => shape.clone(),
            DatasetInfo::Reader { shape, .. } => shape.clone(),
        }
    }

    /// Return the number of dimensions (rank) of the dataset.
    pub fn ndims(&self) -> usize {
        match &self.info {
            DatasetInfo::Writer { shape, .. } => shape.len(),
            DatasetInfo::Reader { shape, .. } => shape.len(),
        }
    }

    /// Return the total number of elements in the dataset.
    pub fn total_elements(&self) -> usize {
        match &self.info {
            DatasetInfo::Writer { shape, .. } => shape.iter().product(),
            DatasetInfo::Reader { shape, .. } => shape.iter().product(),
        }
    }

    /// Return the size of one element in bytes.
    pub fn element_size(&self) -> usize {
        match &self.info {
            DatasetInfo::Writer { element_size, .. } => *element_size,
            DatasetInfo::Reader { element_size, .. } => *element_size,
        }
    }

    /// Return the chunk dimensions, if this is a chunked dataset.
    pub fn chunk_dims(&self) -> Option<Vec<usize>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let inner = borrow_inner(&self.file_inner);
                if let H5FileInner::Reader(reader) = &*inner {
                    if let Some(info) = reader.dataset_info(name) {
                        if let crate::format::messages::data_layout::DataLayoutMessage::ChunkedV4 {
                            chunk_dims,
                            ..
                        } = &info.layout
                        {
                            // Strip trailing element-size dimension
                            return Some(
                                chunk_dims[..chunk_dims.len() - 1]
                                    .iter()
                                    .map(|&d| d as usize)
                                    .collect(),
                            );
                        }
                    }
                }
                None
            }
            DatasetInfo::Writer { .. } => None,
        }
    }

    /// Return whether this is a chunked dataset.
    pub fn is_chunked(&self) -> bool {
        match &self.info {
            DatasetInfo::Writer { chunked, .. } => *chunked,
            DatasetInfo::Reader { name, .. } => {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Reader(reader) => {
                        if let Some(info) = reader.dataset_info(name) {
                            matches!(
                                info.layout,
                                crate::format::messages::data_layout::DataLayoutMessage::ChunkedV4 { .. }
                            )
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            }
        }
    }

    /// Return the names of all attributes on this dataset (read mode only).
    pub fn attr_names(&self) -> Result<Vec<String>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Reader(reader) => Ok(reader.dataset_attr_names(name)?),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "attr_names not available in write mode".into(),
            )),
        }
    }

    /// Open an attribute by name (read mode only).
    pub fn attr(&self, attr_name: &str) -> Result<crate::attribute::H5Attribute> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Reader(reader) => {
                        let attr_msg = reader.dataset_attr(name, attr_name)?;
                        Ok(crate::attribute::H5Attribute::new_reader(
                            clone_inner(&self.file_inner),
                            attr_msg.name.clone(),
                            attr_msg.data.clone(),
                        ))
                    }
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "attr() not available in write mode".into(),
            )),
        }
    }

    /// Start building a new attribute on this dataset.
    ///
    /// Returns a fluent builder. Call `.shape(())` for a scalar attribute
    /// and `.create("name")` to finalize.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// # use rust_hdf5::types::VarLenUnicode;
    /// let file = H5File::create("attr.h5").unwrap();
    /// let ds = file.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
    /// let attr = ds.new_attr::<VarLenUnicode>().shape(()).create("units").unwrap();
    /// attr.write_scalar(&VarLenUnicode("meters".to_string())).unwrap();
    /// ```
    pub fn new_attr<T: 'static>(&self) -> AttrBuilder<'_, T> {
        let ds_index = match &self.info {
            DatasetInfo::Writer { index, .. } => *index,
            DatasetInfo::Reader { .. } => {
                // Reader mode: we'll return a builder that will error on create.
                // Using usize::MAX as sentinel.
                usize::MAX
            }
        };
        AttrBuilder::new(&self.file_inner, ds_index)
    }

    /// Write a typed slice to the dataset (contiguous datasets only).
    ///
    /// The slice length must match the total number of elements declared by
    /// the dataset shape. The data is reinterpreted as raw bytes and written
    /// to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file is in read mode.
    /// - The data length does not match the declared shape.
    pub fn write_raw<T: H5Type>(&self, data: &[T]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                shape,
                element_size,
                chunked,
            } => {
                if *chunked {
                    return Err(Hdf5Error::InvalidState(
                        "use write_chunk for chunked datasets".into(),
                    ));
                }

                let total_elements: usize = shape.iter().product();
                if data.len() != total_elements {
                    return Err(Hdf5Error::InvalidState(format!(
                        "data length {} does not match dataset size {}",
                        data.len(),
                        total_elements,
                    )));
                }

                // Verify element size matches
                if T::element_size() != *element_size {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "write type has element size {} but dataset expects {}",
                        T::element_size(),
                        element_size,
                    )));
                }

                // Safety: T: Copy + 'static (numeric primitive) with well-defined
                // byte representation. The resulting slice borrows `data` and
                // lives only as long as this block.
                let byte_len = data.len() * T::element_size();
                let raw =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };

                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_dataset_raw(*index, raw)?;
                        Ok(())
                    }
                    _ => Err(Hdf5Error::InvalidState(
                        "file is no longer in write mode".into(),
                    )),
                }
            }
            DatasetInfo::Reader { .. } => Err(Hdf5Error::InvalidState(
                "cannot write to a dataset opened in read mode".into(),
            )),
        }
    }

    /// Write a single chunk to a chunked dataset.
    ///
    /// `chunk_idx` is the linear chunk index (typically the frame number for
    /// streaming datasets). `data` is the raw byte data for one chunk.
    pub fn write_chunk(&self, chunk_idx: usize, data: &[u8]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, chunked, .. } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "write_chunk is only for chunked datasets".into(),
                    ));
                }

                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_chunk(*index, chunk_idx as u64, data)?;
                        Ok(())
                    }
                    _ => Err(Hdf5Error::InvalidState(
                        "file is no longer in write mode".into(),
                    )),
                }
            }
            DatasetInfo::Reader { .. } => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
        }
    }

    /// Write multiple chunks in a batch, optionally compressing in parallel.
    ///
    /// `chunks` is a slice of `(chunk_index, raw_data)` pairs. When a filter
    /// pipeline is configured and the `parallel` feature is enabled, all
    /// chunks are compressed concurrently via rayon.
    pub fn write_chunks_batch(&self, chunks: &[(usize, &[u8])]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, chunked, .. } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "write_chunks_batch is only for chunked datasets".into(),
                    ));
                }
                let pairs: Vec<(u64, &[u8])> = chunks
                    .iter()
                    .map(|(idx, data)| (*idx as u64, *data))
                    .collect();
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_chunks_batch(*index, &pairs)?;
                        Ok(())
                    }
                    _ => Err(Hdf5Error::InvalidState(
                        "file is no longer in write mode".into(),
                    )),
                }
            }
            DatasetInfo::Reader { .. } => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
        }
    }

    /// Extend the dimensions of a chunked dataset.
    pub fn extend(&self, new_dims: &[usize]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, chunked, .. } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "extend is only for chunked datasets".into(),
                    ));
                }

                let dims_u64: Vec<u64> = new_dims.iter().map(|&d| d as u64).collect();
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.extend_dataset(*index, &dims_u64)?;
                        Ok(())
                    }
                    _ => Err(Hdf5Error::InvalidState(
                        "file is no longer in write mode".into(),
                    )),
                }
            }
            DatasetInfo::Reader { .. } => {
                Err(Hdf5Error::InvalidState("cannot extend in read mode".into()))
            }
        }
    }

    /// Flush a chunked dataset's index structures to disk.
    pub fn flush(&self) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.flush_dataset(*index)?;
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
            DatasetInfo::Reader { .. } => Ok(()),
        }
    }

    /// Read a slice (hyperslab) of the dataset as a typed vector.
    ///
    /// `starts` and `counts` define the N-dimensional selection:
    /// `starts[d]` = first index along dim d, `counts[d]` = how many elements.
    pub fn read_slice<T: H5Type>(&self, starts: &[usize], counts: &[usize]) -> Result<Vec<T>> {
        match &self.info {
            DatasetInfo::Reader {
                name, element_size, ..
            } => {
                if T::element_size() != *element_size {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "read type has element size {} but dataset has element size {}",
                        T::element_size(),
                        element_size,
                    )));
                }
                let starts_u64: Vec<u64> = starts.iter().map(|&s| s as u64).collect();
                let counts_u64: Vec<u64> = counts.iter().map(|&c| c as u64).collect();

                let raw = {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => {
                            reader.read_slice(name, &starts_u64, &counts_u64)?
                        }
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()))
                        }
                    }
                };

                if raw.len() % T::element_size() != 0 {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "raw data size {} is not a multiple of element size {}",
                        raw.len(),
                        T::element_size(),
                    )));
                }

                let count = raw.len() / T::element_size();
                let mut result = Vec::<T>::with_capacity(count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw.as_ptr(),
                        result.as_mut_ptr() as *mut u8,
                        raw.len(),
                    );
                    result.set_len(count);
                }
                Ok(result)
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read_slice from a dataset in write mode".into(),
            )),
        }
    }

    /// Write a typed slice to a sub-region of a contiguous dataset.
    ///
    /// `starts` and `counts` define the N-dimensional selection.
    pub fn write_slice<T: H5Type>(
        &self,
        starts: &[usize],
        counts: &[usize],
        data: &[T],
    ) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                element_size,
                chunked,
                ..
            } => {
                if *chunked {
                    return Err(Hdf5Error::InvalidState(
                        "write_slice is only for contiguous datasets".into(),
                    ));
                }
                if T::element_size() != *element_size {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "write type has element size {} but dataset expects {}",
                        T::element_size(),
                        element_size,
                    )));
                }

                let expected: usize = counts.iter().product();
                if data.len() != expected {
                    return Err(Hdf5Error::InvalidState(format!(
                        "data length {} does not match slice size {}",
                        data.len(),
                        expected,
                    )));
                }

                let starts_u64: Vec<u64> = starts.iter().map(|&s| s as u64).collect();
                let counts_u64: Vec<u64> = counts.iter().map(|&c| c as u64).collect();

                let byte_len = data.len() * T::element_size();
                let raw =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };

                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_slice(*index, &starts_u64, &counts_u64, raw)?;
                        Ok(())
                    }
                    _ => Err(Hdf5Error::InvalidState(
                        "file is no longer in write mode".into(),
                    )),
                }
            }
            DatasetInfo::Reader { .. } => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
        }
    }

    /// Read variable-length strings from a dataset.
    ///
    /// This handles h5py-style vlen string datasets that store strings
    /// as global heap references. Returns one String per element.
    pub fn read_vlen_strings(&self) -> Result<Vec<String>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader.read_vlen_strings(name)?),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read vlen strings from a dataset in write mode".into(),
            )),
        }
    }

    /// Read the entire dataset as a typed vector.
    ///
    /// The raw bytes are read from the file and reinterpreted as `T`. The
    /// caller must ensure that `T` matches the datatype used when the dataset
    /// was written.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file is in write mode.
    /// - The raw data size is not a multiple of `T::element_size()`.
    pub fn read_raw<T: H5Type>(&self) -> Result<Vec<T>> {
        match &self.info {
            DatasetInfo::Reader {
                name, element_size, ..
            } => {
                if T::element_size() != *element_size {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "read type has element size {} but dataset has element size {}",
                        T::element_size(),
                        element_size,
                    )));
                }

                let raw = {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => reader.read_dataset_raw(name)?,
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()));
                        }
                    }
                };

                if raw.len() % T::element_size() != 0 {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "raw data size {} is not a multiple of element size {}",
                        raw.len(),
                        T::element_size(),
                    )));
                }

                let count = raw.len() / T::element_size();
                let mut result = Vec::<T>::with_capacity(count);

                // Safety: T is Copy + 'static (required by H5Type). We verified
                // the byte count matches count * size_of::<T>() above.
                // copy_nonoverlapping fills the memory with valid bit patterns
                // for all H5Type implementors (numeric primitives).
                // We call set_len AFTER the copy so that if an unexpected panic
                // occurs, uninitialized memory is never exposed.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw.as_ptr(),
                        result.as_mut_ptr() as *mut u8,
                        raw.len(),
                    );
                    result.set_len(count);
                }

                Ok(result)
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read from a dataset in write mode".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::H5File;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hdf5_dataset_test_{}.h5", name))
    }

    #[test]
    fn builder_requires_shape() {
        let path = temp_path("no_shape");
        let file = H5File::create(&path).unwrap();
        let result = file.new_dataset::<u8>().create("data");
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_raw_size_mismatch() {
        let path = temp_path("size_mismatch");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<u8>().shape([4]).create("data").unwrap();
        // Provide 3 elements instead of 4
        let result = ds.write_raw(&[1u8, 2, 3]);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_u8_1d() {
        let path = temp_path("rt_u8_1d");
        let data: Vec<u8> = (0..10).collect();

        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<u8>().shape([10]).create("seq").unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("seq").unwrap();
            assert_eq!(ds.shape(), vec![10]);
            let readback = ds.read_raw::<u8>().unwrap();
            assert_eq!(readback, data);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_i32_2d() {
        let path = temp_path("rt_i32_2d");
        let data: Vec<i32> = vec![-1, 0, 1, 2, 3, 4];

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([2, 3])
                .create("matrix")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("matrix").unwrap();
            assert_eq!(ds.shape(), vec![2, 3]);
            let readback = ds.read_raw::<i32>().unwrap();
            assert_eq!(readback, data);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_f64_3d() {
        let path = temp_path("rt_f64_3d");
        let data: Vec<f64> = (0..24).map(|i| i as f64 * 0.5).collect();

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([2, 3, 4])
                .create("cube")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("cube").unwrap();
            assert_eq!(ds.shape(), vec![2, 3, 4]);
            let readback = ds.read_raw::<f64>().unwrap();
            assert_eq!(readback, data);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cannot_read_in_write_mode() {
        let path = temp_path("no_read_write");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<u8>().shape([4]).create("x").unwrap();
        ds.write_raw(&[1u8, 2, 3, 4]).unwrap();
        let result = ds.read_raw::<u8>();
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cannot_write_in_read_mode() {
        let path = temp_path("no_write_read");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<u8>().shape([4]).create("x").unwrap();
            ds.write_raw(&[1u8, 2, 3, 4]).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("x").unwrap();
            let result = ds.write_raw(&[5u8, 6, 7, 8]);
            assert!(result.is_err());
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn numeric_attr_roundtrip() {
        let path = temp_path("num_attr");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f32>().shape([4]).create("data").unwrap();
            ds.write_raw(&[1.0f32; 4]).unwrap();

            let a1 = ds.new_attr::<f64>().shape(()).create("scale").unwrap();
            a1.write_numeric(&1.2345f64).unwrap();

            let a2 = ds.new_attr::<i32>().shape(()).create("count").unwrap();
            a2.write_numeric(&42i32).unwrap();

            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();

            let scale = ds.attr("scale").unwrap();
            let val: f64 = scale.read_numeric().unwrap();
            assert!((val - 1.2345).abs() < 1e-10);

            let count = ds.attr("count").unwrap();
            let val: i32 = count.read_numeric().unwrap();
            assert_eq!(val, 42);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cannot_create_dataset_in_read_mode() {
        let path = temp_path("no_create_read");

        {
            let _file = H5File::create(&path).unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let result = file.new_dataset::<u8>().shape([4]).create("x");
            assert!(result.is_err());
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn shape_accessor() {
        let path = temp_path("shape_acc");

        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<f32>()
            .shape([5, 10, 3])
            .create("tensor")
            .unwrap();
        assert_eq!(ds.shape(), vec![5, 10, 3]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn slice_roundtrip_2d() {
        let path = temp_path("slice_2d");

        // Create a 4x5 dataset, write full, then read a slice
        let data: Vec<i32> = (0..20).collect();
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([4, 5])
                .create("mat")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("mat").unwrap();
            // Read rows 1..3, cols 2..4 (2x2 slice)
            let slice = ds.read_slice::<i32>(&[1, 2], &[2, 2]).unwrap();
            // Row 1: [5,6,7,8,9] -> cols 2..4 = [7,8]
            // Row 2: [10,11,12,13,14] -> cols 2..4 = [12,13]
            assert_eq!(slice, vec![7, 8, 12, 13]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_slice_2d() {
        let path = temp_path("write_slice_2d");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f32>()
                .shape([3, 4])
                .create("data")
                .unwrap();
            ds.write_raw(&[0.0f32; 12]).unwrap();
            // Overwrite a 2x2 sub-region
            ds.write_slice(&[1, 1], &[2, 2], &[10.0f32, 20.0, 30.0, 40.0])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            let full = ds.read_raw::<f32>().unwrap();
            // Row 0: [0,0,0,0]
            // Row 1: [0,10,20,0]
            // Row 2: [0,30,40,0]
            assert_eq!(
                full,
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 30.0, 40.0, 0.0,]
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn attr_read_roundtrip() {
        use crate::types::VarLenUnicode;
        let path = temp_path("attr_read");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<u8>().shape([4]).create("data").unwrap();
            ds.write_raw(&[1u8, 2, 3, 4]).unwrap();
            let a1 = ds
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("units")
                .unwrap();
            a1.write_string("meters").unwrap();
            let a2 = ds
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("desc")
                .unwrap();
            a2.write_string("test data").unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();

            let names = ds.attr_names().unwrap();
            assert!(names.contains(&"units".to_string()));
            assert!(names.contains(&"desc".to_string()));

            let units = ds.attr("units").unwrap();
            assert_eq!(units.read_string().unwrap(), "meters");

            let desc = ds.attr("desc").unwrap();
            assert_eq!(desc.read_string().unwrap(), "test data");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn type_mismatch_element_size() {
        let path = temp_path("type_mismatch");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f64>().shape([4]).create("data").unwrap();
            ds.write_raw(&[1.0f64, 2.0, 3.0, 4.0]).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            // Try to read as u8 (element_size = 1) from a f64 dataset (element_size = 8)
            let result = ds.read_raw::<u8>();
            assert!(result.is_err());
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dataset_survives_file_move() {
        let path = temp_path("ds_survives");

        let ds = {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<u8>().shape([4]).create("x").unwrap()
        };
        // file is dropped here, but ds still holds Rc to the inner state
        ds.write_raw(&[1u8, 2, 3, 4]).unwrap();
        // The writer will finalize on drop of the last Rc

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn new_attr_scalar_string() {
        use crate::types::VarLenUnicode;

        let path = temp_path("attr_scalar_string");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<u8>().shape([4]).create("data").unwrap();
            ds.write_raw(&[1u8, 2, 3, 4]).unwrap();

            let attr = ds
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("name")
                .unwrap();
            attr.write_scalar(&VarLenUnicode("test_value".to_string()))
                .unwrap();

            file.close().unwrap();
        }

        // Verify the file is still valid and readable
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![4]);
            let readback = ds.read_raw::<u8>().unwrap();
            assert_eq!(readback, vec![1u8, 2, 3, 4]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn all_numeric_types_roundtrip() {
        let path = temp_path("all_types");

        {
            let file = H5File::create(&path).unwrap();

            let ds = file.new_dataset::<u8>().shape([2]).create("u8").unwrap();
            ds.write_raw(&[1u8, 2]).unwrap();

            let ds = file.new_dataset::<i8>().shape([2]).create("i8").unwrap();
            ds.write_raw(&[-1i8, 1]).unwrap();

            let ds = file.new_dataset::<u16>().shape([2]).create("u16").unwrap();
            ds.write_raw(&[100u16, 200]).unwrap();

            let ds = file.new_dataset::<i16>().shape([2]).create("i16").unwrap();
            ds.write_raw(&[-100i16, 100]).unwrap();

            let ds = file.new_dataset::<u32>().shape([2]).create("u32").unwrap();
            ds.write_raw(&[1000u32, 2000]).unwrap();

            let ds = file.new_dataset::<i32>().shape([2]).create("i32").unwrap();
            ds.write_raw(&[-1000i32, 1000]).unwrap();

            let ds = file.new_dataset::<u64>().shape([2]).create("u64").unwrap();
            ds.write_raw(&[10000u64, 20000]).unwrap();

            let ds = file.new_dataset::<i64>().shape([2]).create("i64").unwrap();
            ds.write_raw(&[-10000i64, 10000]).unwrap();

            let ds = file.new_dataset::<f32>().shape([2]).create("f32").unwrap();
            ds.write_raw(&[1.5f32, 2.5]).unwrap();

            let ds = file.new_dataset::<f64>().shape([2]).create("f64").unwrap();
            ds.write_raw(&[1.23456f64, 7.89012]).unwrap();

            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();

            assert_eq!(
                file.dataset("u8").unwrap().read_raw::<u8>().unwrap(),
                vec![1u8, 2]
            );
            assert_eq!(
                file.dataset("i8").unwrap().read_raw::<i8>().unwrap(),
                vec![-1i8, 1]
            );
            assert_eq!(
                file.dataset("u16").unwrap().read_raw::<u16>().unwrap(),
                vec![100u16, 200]
            );
            assert_eq!(
                file.dataset("i16").unwrap().read_raw::<i16>().unwrap(),
                vec![-100i16, 100]
            );
            assert_eq!(
                file.dataset("u32").unwrap().read_raw::<u32>().unwrap(),
                vec![1000u32, 2000]
            );
            assert_eq!(
                file.dataset("i32").unwrap().read_raw::<i32>().unwrap(),
                vec![-1000i32, 1000]
            );
            assert_eq!(
                file.dataset("u64").unwrap().read_raw::<u64>().unwrap(),
                vec![10000u64, 20000]
            );
            assert_eq!(
                file.dataset("i64").unwrap().read_raw::<i64>().unwrap(),
                vec![-10000i64, 10000]
            );
            assert_eq!(
                file.dataset("f32").unwrap().read_raw::<f32>().unwrap(),
                vec![1.5f32, 2.5]
            );
            assert_eq!(
                file.dataset("f64").unwrap().read_raw::<f64>().unwrap(),
                vec![1.23456f64, 7.89012]
            );
        }

        std::fs::remove_file(&path).ok();
    }
}
