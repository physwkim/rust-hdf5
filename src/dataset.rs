//! Dataset creation and I/O.
//!
//! Datasets are created via the fluent [`DatasetBuilder`] API obtained from
//! [`H5File::new_dataset`](crate::file::H5File::new_dataset). Once created,
//! the [`H5Dataset`] handle can read or write raw typed data.

use crate::attribute::AttrBuilder;
use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner, borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
use crate::format::messages::datatype::DatatypeMessage;
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
    fill_value: Option<Vec<u8>>,
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
            fill_value: None,
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
            fill_value: None,
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

    /// Set a user-defined fill value for unwritten elements.
    ///
    /// Without this, datasets use the HDF5 default zero-fill. When set,
    /// the value is written into the dataset's fill-value message
    /// (`fill_defined = 2`), so HDF5 readers treat unallocated chunks and
    /// unwritten regions as this value rather than zero.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("fv.h5").unwrap();
    /// let ds = file.new_dataset::<f32>()
    ///     .shape(&[100])
    ///     .fill_value(f32::NAN)
    ///     .create("data")
    ///     .unwrap();
    /// ```
    #[must_use]
    pub fn fill_value(mut self, value: T) -> Self {
        let es = T::element_size();
        // Safety: `T: H5Type` is a `Copy` numeric primitive with a
        // well-defined byte representation; `element_size()` matches
        // `size_of::<T>()`. The slice borrows `value` only for this call.
        let raw = unsafe { std::slice::from_raw_parts(&value as *const T as *const u8, es) };
        self.fill_value = Some(raw.to_vec());
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
        let fill_value = self.fill_value.clone();

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

            // libhdf5 selects the chunk index from the dataspace: a v2
            // B-tree for two or more unlimited dimensions, an extensible
            // array for exactly one, and a fixed array when there are none.
            let n_unlimited = max_u64.iter().filter(|&&m| m == u64::MAX).count();
            let is_btree2 = n_unlimited >= 2;
            let is_fixed_array = n_unlimited == 0;
            let wants_filter = self.custom_pipeline.is_some()
                || self.shuffle_deflate_level.is_some()
                || self.deflate_level.is_some();

            let index = {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        let idx = if is_btree2 {
                            if wants_filter {
                                return Err(Hdf5Error::InvalidState(
                                    "compression of v2 B-tree (multi-unlimited-dimension) \
                                     datasets is not yet supported"
                                        .into(),
                                ));
                            }
                            writer.create_btree_v2_dataset(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64,
                            )?
                        } else if is_fixed_array {
                            // A chunked dataset with no unlimited dimension
                            // must use the fixed-array index — libhdf5
                            // rejects an extensible-array index here. A
                            // compressed fixed-shape dataset uses a *filtered*
                            // fixed array (FA client id 1).
                            if wants_filter {
                                let pipeline = if let Some(p) = self.custom_pipeline {
                                    p
                                } else if let Some(level) = self.shuffle_deflate_level {
                                    crate::format::messages::filter::FilterPipeline::shuffle_deflate(
                                        T::element_size() as u32,
                                        level,
                                    )
                                } else {
                                    // deflate_level (checked by wants_filter).
                                    crate::format::messages::filter::FilterPipeline::deflate(
                                        self.deflate_level.unwrap(),
                                    )
                                };
                                writer.create_fixed_array_dataset_with_pipeline(
                                    &full_name, datatype, &dims_u64, &chunk_u64, pipeline,
                                )?
                            } else {
                                writer.create_fixed_array_dataset(
                                    &full_name, datatype, &dims_u64, &chunk_u64,
                                )?
                            }
                        } else if let Some(pipeline) = self.custom_pipeline {
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
                        if let Some(ref fv) = fill_value {
                            writer.set_dataset_fill_value(idx, fv.clone())?;
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
                    btree2: is_btree2,
                    fixed_array: is_fixed_array,
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
                        if let Some(ref fv) = fill_value {
                            writer.set_dataset_fill_value(idx, fv.clone())?;
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
                    btree2: false,
                    fixed_array: false,
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
        /// Whether the chunk index is a v2 B-tree (multiple unlimited dims).
        btree2: bool,
        /// Whether the chunk index is a Fixed Array (no unlimited dims).
        fixed_array: bool,
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

    /// Return the element datatype as parsed from the file (read mode only).
    ///
    /// Unlike [`element_size`](Self::element_size), which reports only the
    /// byte width, this exposes the full datatype: its class (integer vs
    /// floating-point vs string vs compound …), signedness, byte order and
    /// bit precision. Callers that must reconstruct the exact stored type —
    /// for example to map it to a NumPy / Arrow dtype — should use this
    /// instead of inferring a type from the byte width, which cannot
    /// distinguish `u8` from `i8` (both 1 byte) or `i32` from `f32` (both 4
    /// bytes).
    ///
    /// # Errors
    ///
    /// Returns an error if the file is in write mode, or if the dataset can
    /// no longer be found in the reader's metadata.
    ///
    /// ```no_run
    /// # use rust_hdf5::{H5File, DatatypeMessage};
    /// let file = H5File::open("data.h5").unwrap();
    /// let ds = file.dataset("image").unwrap();
    /// match ds.datatype().unwrap() {
    ///     DatatypeMessage::FixedPoint { size, signed, .. } => {
    ///         println!("integer: {} bytes, signed={}", size, signed);
    ///     }
    ///     DatatypeMessage::FloatingPoint { size, .. } => {
    ///         println!("float: {} bytes", size);
    ///     }
    ///     other => println!("other type: {other}"),
    /// }
    /// ```
    pub fn datatype(&self) -> Result<DatatypeMessage> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Reader(reader) => reader
                        .dataset_info(name)
                        .map(|info| info.datatype.clone())
                        .ok_or_else(|| Hdf5Error::NotFound(name.clone())),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "datatype() is only available in read mode".into(),
            )),
        }
    }

    /// Return the chunk dimensions, if this is a chunked dataset.
    pub fn chunk_dims(&self) -> Option<Vec<usize>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let inner = borrow_inner(&self.file_inner);
                if let H5FileInner::Reader(reader) = &*inner {
                    if let Some(info) = reader.dataset_info(name) {
                        use crate::format::messages::data_layout::DataLayoutMessage;
                        let chunk_dims = match &info.layout {
                            DataLayoutMessage::ChunkedV4 { chunk_dims, .. }
                            | DataLayoutMessage::ChunkedV3 { chunk_dims, .. } => Some(chunk_dims),
                            _ => None,
                        };
                        if let Some(chunk_dims) = chunk_dims {
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
                            use crate::format::messages::data_layout::DataLayoutMessage;
                            matches!(
                                info.layout,
                                DataLayoutMessage::ChunkedV4 { .. }
                                    | DataLayoutMessage::ChunkedV3 { .. }
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
                        let attr_msg = reader.dataset_attr(name, attr_name)?.clone();
                        Ok(crate::attribute::H5Attribute::new_reader(
                            clone_inner(&self.file_inner),
                            attr_msg,
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
                btree2: _,
                fixed_array: _,
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
    ///
    /// For datasets with two or more unlimited dimensions (v2 B-tree index),
    /// use [`write_chunk_at`](Self::write_chunk_at) instead.
    pub fn write_chunk(&self, chunk_idx: usize, data: &[u8]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                chunked,
                btree2,
                fixed_array,
                ..
            } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "write_chunk is only for chunked datasets".into(),
                    ));
                }
                if *btree2 {
                    return Err(Hdf5Error::InvalidState(
                        "this dataset uses a v2 B-tree chunk index; use write_chunk_at \
                         with the chunk's grid coordinates"
                            .into(),
                    ));
                }

                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        if *fixed_array {
                            // Fixed-array dataset: convert the linear chunk
                            // index into row-major grid coordinates.
                            let chunk_dims = writer
                                .dataset_chunk_dims(*index)
                                .ok_or_else(|| {
                                    Hdf5Error::InvalidState("dataset has no chunk info".into())
                                })?
                                .to_vec();
                            let dims = writer.dataset_dims(*index).to_vec();
                            let mut grid = vec![0u64; dims.len()];
                            for d in 0..dims.len() {
                                grid[d] = if chunk_dims[d] > 0 {
                                    dims[d].div_ceil(chunk_dims[d])
                                } else {
                                    1
                                };
                            }
                            // A zero-extent dimension yields a grid of 0
                            // chunks — there is no chunk to write.
                            if grid.contains(&0) {
                                return Err(Hdf5Error::InvalidState(
                                    "dataset has a zero-extent dimension and no chunks".into(),
                                ));
                            }
                            let mut rem = chunk_idx as u64;
                            let mut coords = vec![0u64; dims.len()];
                            for d in (0..dims.len()).rev() {
                                coords[d] = rem % grid[d];
                                rem /= grid[d];
                            }
                            // A leftover means chunk_idx exceeded the grid.
                            if rem != 0 {
                                return Err(Hdf5Error::InvalidState(format!(
                                    "chunk index {chunk_idx} is out of range for this dataset"
                                )));
                            }
                            writer.write_chunk_fixed_array(*index, &coords, data)?;
                        } else {
                            writer.write_chunk(*index, chunk_idx as u64, data)?;
                        }
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

    /// Write an already-filtered (pre-compressed) chunk **verbatim**, recording
    /// the caller-supplied `filter_mask`. The bytes are stored as-is without
    /// running the dataset's filter pipeline — the HDF5 "direct chunk write"
    /// (`H5Dwrite_chunk`, formerly `H5DOwrite_chunk`) operation.
    ///
    /// `chunk_idx` is the linear chunk index (the frame number for streaming
    /// datasets), exactly as for [`write_chunk`](Self::write_chunk). `data` is
    /// the already-filtered bytes of one chunk — its length is the *stored*
    /// (compressed) size, not the uncompressed chunk size.
    ///
    /// `filter_mask` is a bitfield: bit *i* set means filter *i* of the
    /// dataset's pipeline was **not** applied to this chunk and must be skipped
    /// on read. Pass 0 when the full pipeline was already applied upstream (the
    /// common case: a codec plugin handed you compressed frames).
    ///
    /// The dataset must be chunked **and** filtered; an unfiltered chunk index
    /// has no slot to record a stored size or mask. For a v2-B-tree-indexed
    /// dataset (two or more unlimited dimensions) direct chunk writes are not
    /// supported.
    ///
    /// # Reading back
    ///
    /// Both this crate's reader and libhdf5/h5py honor the per-chunk
    /// `filter_mask`: a chunk written with any mask round-trips correctly, with
    /// the reader skipping exactly the filters the mask marks as not applied.
    pub fn write_chunk_raw(&self, chunk_idx: usize, data: &[u8], filter_mask: u32) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                chunked,
                btree2,
                fixed_array,
                ..
            } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "write_chunk_raw is only for chunked datasets".into(),
                    ));
                }
                if *btree2 {
                    return Err(Hdf5Error::InvalidState(
                        "direct chunk writes are not supported for v2-B-tree-indexed \
                         datasets (two or more unlimited dimensions)"
                            .into(),
                    ));
                }

                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        if *fixed_array {
                            // Fixed-array dataset: convert the linear chunk
                            // index into row-major grid coordinates.
                            let chunk_dims = writer
                                .dataset_chunk_dims(*index)
                                .ok_or_else(|| {
                                    Hdf5Error::InvalidState("dataset has no chunk info".into())
                                })?
                                .to_vec();
                            let dims = writer.dataset_dims(*index).to_vec();
                            let mut grid = vec![0u64; dims.len()];
                            for d in 0..dims.len() {
                                grid[d] = if chunk_dims[d] > 0 {
                                    dims[d].div_ceil(chunk_dims[d])
                                } else {
                                    1
                                };
                            }
                            // A zero-extent dimension yields a grid of 0
                            // chunks — there is no chunk to write.
                            if grid.contains(&0) {
                                return Err(Hdf5Error::InvalidState(
                                    "dataset has a zero-extent dimension and no chunks".into(),
                                ));
                            }
                            let mut rem = chunk_idx as u64;
                            let mut coords = vec![0u64; dims.len()];
                            for d in (0..dims.len()).rev() {
                                coords[d] = rem % grid[d];
                                rem /= grid[d];
                            }
                            // A leftover means chunk_idx exceeded the grid.
                            if rem != 0 {
                                return Err(Hdf5Error::InvalidState(format!(
                                    "chunk index {chunk_idx} is out of range for this dataset"
                                )));
                            }
                            writer.write_compressed_chunk_fixed_array(
                                *index,
                                &coords,
                                data,
                                filter_mask,
                            )?;
                        } else {
                            writer.write_compressed_chunk(
                                *index,
                                chunk_idx as u64,
                                data,
                                filter_mask,
                            )?;
                        }
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

    /// Write a single chunk to a v2-B-tree-indexed dataset, addressed by its
    /// chunk-grid coordinates (one per dimension).
    ///
    /// This is the entry point for datasets with two or more unlimited
    /// dimensions. The dataset's logical dimensions are extended to cover
    /// the written chunk. `data` is the raw bytes of one full chunk.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("bt2.h5").unwrap();
    /// let ds = file.new_dataset::<i32>()
    ///     .shape(&[0, 0])
    ///     .chunk(&[2, 2])
    ///     .max_shape(&[None, None])
    ///     .create("grid")
    ///     .unwrap();
    /// let chunk = [0i32, 1, 2, 3];
    /// let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
    /// ds.write_chunk_at(&[0, 0], &bytes).unwrap();
    /// ```
    pub fn write_chunk_at(&self, chunk_coords: &[usize], data: &[u8]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                chunked,
                btree2,
                fixed_array,
                ..
            } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "write_chunk_at is only for chunked datasets".into(),
                    ));
                }
                let coords: Vec<u64> = chunk_coords.iter().map(|&c| c as u64).collect();
                let btree2 = *btree2;
                let fixed_array = *fixed_array;
                let mut inner = borrow_inner_mut(&self.file_inner);
                let writer = match &mut *inner {
                    H5FileInner::Writer(w) => w,
                    _ => {
                        return Err(Hdf5Error::InvalidState(
                            "file is no longer in write mode".into(),
                        ))
                    }
                };
                let chunk_dims = writer
                    .dataset_chunk_dims(*index)
                    .ok_or_else(|| Hdf5Error::InvalidState("dataset has no chunk info".into()))?
                    .to_vec();
                let dims = writer.dataset_dims(*index).to_vec();
                if coords.len() != dims.len() {
                    return Err(Hdf5Error::InvalidState(format!(
                        "chunk_coords has {} entries but the dataset has {} dimensions",
                        coords.len(),
                        dims.len()
                    )));
                }
                if chunk_dims.len() != dims.len() {
                    return Err(Hdf5Error::InvalidState(format!(
                        "dataset chunk shape has {} dimensions but the dataspace has {}",
                        chunk_dims.len(),
                        dims.len()
                    )));
                }

                // Validate coordinates and compute the grown dimensions
                // up-front, before any chunk is written, so an overflowing
                // coordinate cannot leave an orphaned chunk in the file.
                let mut new_dims = dims.clone();
                for d in 0..dims.len() {
                    let needed = coords[d]
                        .checked_add(1)
                        .and_then(|c| c.checked_mul(chunk_dims[d]))
                        .ok_or_else(|| {
                            Hdf5Error::InvalidState(format!(
                                "chunk coordinate {} in dimension {} is too large",
                                coords[d], d
                            ))
                        })?;
                    if needed > new_dims[d] {
                        new_dims[d] = needed;
                    }
                }

                if fixed_array {
                    // Fixed-array (fixed-shape) dataset: no dimension growth.
                    writer.write_chunk_fixed_array(*index, &coords, data)?;
                    return Ok(());
                }

                if btree2 {
                    writer.write_chunk_btree_v2(*index, &coords, data)?;
                } else {
                    // Extensible array: linearize the chunk-grid coordinates
                    // (row-major) into the array's chunk index.
                    let mut linear = 0u64;
                    for d in 0..dims.len() {
                        let grid = if chunk_dims[d] > 0 {
                            dims[d].div_ceil(chunk_dims[d])
                        } else {
                            1
                        };
                        linear = linear
                            .checked_mul(grid)
                            .and_then(|l| l.checked_add(coords[d]))
                            .ok_or_else(|| {
                                Hdf5Error::InvalidState(
                                    "chunk coordinates overflow the array index".into(),
                                )
                            })?;
                    }
                    writer.write_chunk(*index, linear, data)?;
                }

                if new_dims != dims {
                    writer.extend_dataset(*index, &new_dims)?;
                }
                Ok(())
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

    /// Append data along the first dimension of a chunked dataset.
    ///
    /// `data` must contain a whole number of "frames" — slices along
    /// dimension 0. For example, if the dataset has shape `[N, H, W]`
    /// and `chunk_dims = [1, H, W]`, then `data.len()` must be a
    /// multiple of `H * W`.
    ///
    /// This method writes the necessary chunks and extends the dataset
    /// shape automatically.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("append.h5").unwrap();
    /// let ds = file.new_dataset::<f64>()
    ///     .shape(&[0, 3])
    ///     .chunk(&[1, 3])
    ///     .max_shape(&[None, Some(3)])
    ///     .create("data")
    ///     .unwrap();
    /// ds.append(&[1.0, 2.0, 3.0]).unwrap();       // shape becomes [1, 3]
    /// ds.append(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap(); // shape becomes [3, 3]
    /// ```
    pub fn append<T: H5Type>(&self, data: &[T]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                element_size,
                chunked,
                ..
            } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(
                        "append is only for chunked datasets".into(),
                    ));
                }
                if T::element_size() != *element_size {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "append type has element size {} but dataset expects {}",
                        T::element_size(),
                        element_size,
                    )));
                }

                let ds_index = *index;
                let es = *element_size;

                let mut inner = borrow_inner_mut(&self.file_inner);
                let writer = match &mut *inner {
                    H5FileInner::Writer(w) => w,
                    _ => {
                        return Err(Hdf5Error::InvalidState(
                            "file is no longer in write mode".into(),
                        ))
                    }
                };

                let chunk_dims = writer
                    .dataset_chunk_dims(ds_index)
                    .ok_or_else(|| Hdf5Error::InvalidState("dataset has no chunk info".into()))?
                    .to_vec();
                let dims = writer.dataset_dims(ds_index).to_vec();

                // Frame size = product of dims[1..]
                let frame_elems: usize = if dims.len() > 1 {
                    dims[1..].iter().map(|&d| d as usize).product()
                } else {
                    1
                };

                if frame_elems == 0 {
                    return Err(Hdf5Error::InvalidState(
                        "cannot append to dataset with zero-size trailing dimensions".into(),
                    ));
                }

                if !data.len().is_multiple_of(frame_elems) {
                    return Err(Hdf5Error::InvalidState(format!(
                        "data length {} is not a multiple of frame size {}",
                        data.len(),
                        frame_elems,
                    )));
                }

                let n_new_frames = data.len() / frame_elems;
                let current_dim0 = dims[0] as usize;

                // Chunk size along first dimension
                let chunk_dim0 = chunk_dims[0] as usize;
                let frame_bytes = frame_elems * es;

                let raw = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * es)
                };

                // Merge buffered data with new data
                let ds = &mut writer.datasets[ds_index];
                let buffered_frames = ds.append_buffered_frames as usize;
                let mut combined = std::mem::take(&mut ds.append_buffer);
                combined.extend_from_slice(raw);
                ds.append_buffered_frames = 0;

                let total_frames = buffered_frames + n_new_frames;
                let total_bytes = combined.len();

                // Base chunk index: account for buffered frames
                let base_dim0 = current_dim0 - buffered_frames;
                let mut byte_pos = 0usize;
                let mut frame_pos = 0usize;

                while frame_pos < total_frames {
                    let abs_frame = base_dim0 + frame_pos;
                    let chunk_idx = abs_frame / chunk_dim0;
                    let remaining_frames = total_frames - frame_pos;
                    let frames_to_fill = chunk_dim0 - (abs_frame % chunk_dim0);

                    if remaining_frames >= frames_to_fill {
                        // Full chunk — write
                        let end = byte_pos + frames_to_fill * frame_bytes;
                        if frames_to_fill == chunk_dim0 {
                            writer.write_chunk(
                                ds_index,
                                chunk_idx as u64,
                                &combined[byte_pos..end],
                            )?;
                        } else {
                            // Partial-chunk write: this branch only runs with
                            // offset_in_chunk > 0, meaning the chunk already
                            // holds earlier frames on disk. Read-modify-write
                            // so those frames survive — a fresh fill buffer
                            // would erase them.
                            let offset_in_chunk = (abs_frame % chunk_dim0) * frame_bytes;
                            let mut chunk_buf =
                                match writer.read_chunk_if_present(ds_index, chunk_idx as u64)? {
                                    Some(existing) => existing,
                                    None => {
                                        return Err(Hdf5Error::InvalidState(format!(
                                            "cannot append into partially-written chunk {}: \
                                         its existing content was not found in the chunk \
                                         index (the file may be inconsistent)",
                                            chunk_idx
                                        )));
                                    }
                                };
                            chunk_buf
                                [offset_in_chunk..offset_in_chunk + frames_to_fill * frame_bytes]
                                .copy_from_slice(&combined[byte_pos..end]);
                            writer.write_chunk(ds_index, chunk_idx as u64, &chunk_buf)?;
                        }
                        byte_pos = end;
                        frame_pos += frames_to_fill;
                    } else {
                        // Partial chunk — buffer for next append
                        let ds = &mut writer.datasets[ds_index];
                        ds.append_buffer = combined[byte_pos..total_bytes].to_vec();
                        ds.append_buffered_frames = remaining_frames as u64;
                        frame_pos = total_frames;
                    }
                }

                // Extend dims to include all frames (buffered + new)
                let logical_dim0 = base_dim0 + total_frames;
                let mut new_dims: Vec<u64> = dims;
                new_dims[0] = logical_dim0 as u64;
                writer.extend_dataset(ds_index, &new_dims)?;

                Ok(())
            }
            DatasetInfo::Reader { .. } => {
                Err(Hdf5Error::InvalidState("cannot append in read mode".into()))
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

    /// Set the logical extent of a chunked dataset, growing **or
    /// shrinking** any dimension.
    ///
    /// Unlike [`extend`](Self::extend), which only grows, this can reduce a
    /// dimension — for example to correct an over-extended frame count
    /// after writing a partial multi-frame chunk. Shrinking changes the
    /// logical dataspace only: data in chunks beyond the new extent stays
    /// in the file but is no longer visible on read, exactly as libhdf5's
    /// `H5Dset_extent` behaves. The new extent must not exceed the
    /// dataset's maximum dimensions.
    pub fn set_extent(&self, new_dims: &[usize]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, .. } => {
                let dims_u64: Vec<u64> = new_dims.iter().map(|&d| d as u64).collect();
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Writer(writer) => {
                        writer.set_dataset_extent(*index, &dims_u64)?;
                        Ok(())
                    }
                    _ => Err(Hdf5Error::InvalidState(
                        "file is no longer in write mode".into(),
                    )),
                }
            }
            DatasetInfo::Reader { .. } => Err(Hdf5Error::InvalidState(
                "cannot set extent in read mode".into(),
            )),
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
        // Include PID + a per-call atomic counter so that concurrent
        // cargo invocations and any kernel-level "lock not yet
        // released" races between sequential opens cannot collide.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "hdf5_dataset_test_{}_{}_{}.h5",
            name,
            std::process::id(),
            n
        ))
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
    fn write_slice_out_of_bounds_rejected() {
        let path = temp_path("write_slice_oob");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<i32>().shape([4]).create("d").unwrap();
        ds.write_raw(&[0i32; 4]).unwrap();
        // start 2 + count 6 = 8 > extent 4 -> must error, not corrupt.
        assert!(ds.write_slice(&[2], &[6], &[9i32; 6]).is_err());
        // An in-bounds slice still works.
        assert!(ds.write_slice(&[1], &[2], &[7i32, 8]).is_ok());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn duplicate_dataset_name_rejected() {
        let path = temp_path("dup_name");
        let file = H5File::create(&path).unwrap();
        let _ = file.new_dataset::<i32>().shape([2]).create("d").unwrap();
        assert!(file.new_dataset::<i32>().shape([2]).create("d").is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn extend_cannot_shrink() {
        let path = temp_path("extend_shrink");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0])
            .chunk(&[2])
            .max_shape(&[None])
            .create("d")
            .unwrap();
        ds.append(&[1i32, 2, 3, 4]).unwrap();
        // Shrinking below the written extent must be rejected.
        assert!(ds.extend(&[2]).is_err());
        // Growing is fine.
        assert!(ds.extend(&[6]).is_ok());
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

    #[test]
    fn append_chunked_roundtrip() {
        let path = temp_path("append_chunked");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([0, 3])
                .chunk(&[1, 3])
                .max_shape(&[None, Some(3)])
                .create("data")
                .unwrap();

            // Append one frame
            ds.append(&[1.0f64, 2.0, 3.0]).unwrap();
            // Append two frames at once
            ds.append(&[4.0f64, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![3, 3]);
            let all = ds.read_raw::<f64>().unwrap();
            assert_eq!(all, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_1d_chunked() {
        let path = temp_path("append_1d");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .create("values")
                .unwrap();

            ds.append(&[10i32, 20, 30]).unwrap(); // partial chunk
            ds.append(&[40i32]).unwrap(); // fills chunk boundary
            ds.append(&[50i32, 60, 70, 80]).unwrap(); // full chunk

            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("values").unwrap();
            assert_eq!(ds.shape(), vec![8]);
            let all = ds.read_raw::<i32>().unwrap();
            assert_eq!(all, vec![10, 20, 30, 40, 50, 60, 70, 80]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_partial_chunk_flushed_on_close() {
        let path = temp_path("append_partial_close");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .create("vals")
                .unwrap();

            // Append 5 elements: chunk 0 = full [1,2,3,4], chunk 1 = partial [5,0,0,0]
            ds.append(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
            file.close().unwrap();
        }

        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("vals").unwrap();
            assert_eq!(ds.shape(), vec![5]);
            let all = ds.read_raw::<f64>().unwrap();
            // The full dataset is 2 chunks * 4 = 8 elements; shape says 5
            // read_raw reads total shape elements
            assert_eq!(all.len(), 5);
            assert_eq!(all, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        }

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn vlen_append_after_reopen_filtered() {
        // Reopen + append into a partially-written *compressed* vlen chunk
        // (index-block chunk). Exercises filtered-index-block reconstruction
        // in open_append plus filtered read-modify-write.
        let path = temp_path("vlen_reopen_filtered");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset(
                "strs",
                4,
                Some(crate::format::messages::filter::FilterPipeline::deflate(6)),
            )
            .unwrap();
            file.append_vlen_strings("strs", &["alpha", "beta", "gamma"])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.append_vlen_strings("strs", &["delta"]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let got = file.dataset("strs").unwrap().read_vlen_strings().unwrap();
            assert_eq!(
                got.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                vec!["alpha", "beta", "gamma", "delta"]
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_append_after_reopen_data_block() {
        // Reopen + append into a partial chunk that lives in an extensible-
        // array *data block* (chunk index >= idx_blk_elmts). Exercises
        // data-block resolution in read_chunk_if_present and write_chunk.
        let path = temp_path("vlen_reopen_datablk");
        let labels: Vec<String> = (0..9).map(|i| format!("s{i}")).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("strs", 2, None)
                .unwrap();
            let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            file.append_vlen_strings("strs", &refs).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.append_vlen_strings("strs", &["s9"]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let got = file.dataset("strs").unwrap().read_vlen_strings().unwrap();
            let want: Vec<String> = (0..10).map(|i| format!("s{i}")).collect();
            assert_eq!(got, want);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_append_after_reopen_super_block() {
        // Reopen + append into a partial chunk whose index falls in an
        // extensible-array *super block* (chunk index 244 with the default
        // EA geometry: idx_blk_elmts=4, data_blk_min_elmts=16,
        // sup_blk_min_data_ptrs=4 -> chunks 0..=243 are reached via the
        // index block or its direct data blocks, so chunk 244 is reached
        // via a super block read from disk). Exercises the ViaSblk branch
        // of read_chunk_if_present.
        let path = temp_path("vlen_reopen_super");
        // 489 strings, chunk size 2 -> chunk 244 holds one string only
        // (partially filled) and is flushed to disk on close.
        let labels: Vec<String> = (0..489).map(|i| format!("v{i}")).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("strs", 2, None)
                .unwrap();
            let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            file.append_vlen_strings("strs", &refs).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.append_vlen_strings("strs", &["v489"]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let got = file.dataset("strs").unwrap().read_vlen_strings().unwrap();
            let want: Vec<String> = (0..490).map(|i| format!("v{i}")).collect();
            assert_eq!(got, want);
        }
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn vlen_append_after_reopen_filtered_data_block() {
        // The hardest path: compressed + chunk in a data block + partial
        // read-modify-write across a reopen.
        let path = temp_path("vlen_reopen_filt_datablk");
        let labels: Vec<String> = (0..9).map(|i| format!("item{i:02}")).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset(
                "strs",
                2,
                Some(crate::format::messages::filter::FilterPipeline::deflate(6)),
            )
            .unwrap();
            let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            file.append_vlen_strings("strs", &refs).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.append_vlen_strings("strs", &["item09"]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let got = file.dataset("strs").unwrap().read_vlen_strings().unwrap();
            let want: Vec<String> = (0..10).map(|i| format!("item{i:02}")).collect();
            assert_eq!(got, want);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn group_nx_class_attribute_roundtrip() {
        // Non-root groups carry attributes (NeXus `NX_class`) in their
        // own object header, and the reader reads them back by path.
        let path = temp_path("group_nx_class");
        {
            let file = H5File::create(&path).unwrap();
            let entry = file.create_group("entry").unwrap();
            entry.set_attr_string("NX_class", "NXentry").unwrap();
            let det = entry.create_group("detector").unwrap();
            det.set_attr_string("NX_class", "NXdetector").unwrap();
            det.set_attr_numeric("frame_count", &7i32).unwrap();
            det.new_dataset::<f32>()
                .shape([4])
                .create("data")
                .unwrap()
                .write_raw(&[1.0f32; 4])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let entry = file.root_group().group("entry").unwrap();
            assert_eq!(entry.attr_string("NX_class").unwrap(), "NXentry");
            let det = entry.group("detector").unwrap();
            assert_eq!(det.attr_string("NX_class").unwrap(), "NXdetector");
            let names = det.attr_names().unwrap();
            assert!(names.contains(&"NX_class".to_string()));
            assert!(names.contains(&"frame_count".to_string()));
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ea_super_block_roundtrip() {
        // 2000 chunks span several extensible-array super blocks. Before
        // super-block support the writer errored at chunk index 228.
        let path = temp_path("ea_super_rt");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[1])
                .max_shape(&[None])
                .create("v")
                .unwrap();
            ds.append(&(0..2000).collect::<Vec<i32>>()).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v.len(), 2000);
            assert!(v.iter().enumerate().all(|(i, &x)| x == i as i32));
        }
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn ea_filtered_super_block_roundtrip() {
        // Compressed chunks across super blocks.
        let path = temp_path("ea_filt_super");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[1])
                .max_shape(&[None])
                .deflate(4)
                .create("v")
                .unwrap();
            ds.append(&(0..600).collect::<Vec<i32>>()).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v, (0..600).collect::<Vec<i32>>());
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ea_super_block_open_append() {
        // Reopen a dataset and append chunks that fall in super blocks.
        let path = temp_path("ea_super_append");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[1])
                .max_shape(&[None])
                .create("v")
                .unwrap();
            ds.append(&(0..300).collect::<Vec<i32>>()).unwrap();
            file.close().unwrap();
        }
        {
            let mut w = crate::io::writer::Hdf5Writer::open_append(&path).unwrap();
            let idx = w.dataset_index("v").unwrap();
            for c in 300..900u64 {
                w.write_chunk(idx, c, &(c as i32).to_le_bytes()).unwrap();
            }
            w.extend_dataset(idx, &[900]).unwrap();
            w.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v.len(), 900);
            assert!(v.iter().enumerate().all(|(i, &x)| x == i as i32));
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn btree_v2_multi_unlimited_roundtrip() {
        // A dataset with two unlimited dimensions uses the v2 B-tree chunk
        // index; chunks are written by grid coordinates with write_chunk_at.
        let path = temp_path("bt2_multi");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, 0])
                .chunk(&[2, 2])
                .max_shape(&[None, None])
                .create("grid")
                .unwrap();
            assert!(ds.is_chunked());
            // 4x4 logical grid, value[r][c] = r*4 + c, in 2x2 chunks.
            for cr in 0..2usize {
                for cc in 0..2usize {
                    let mut bytes = Vec::new();
                    for i in 0..2usize {
                        for j in 0..2usize {
                            let v = ((cr * 2 + i) * 4 + (cc * 2 + j)) as i32;
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    ds.write_chunk_at(&[cr, cc], &bytes).unwrap();
                }
            }
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("grid").unwrap();
            assert_eq!(ds.shape(), vec![4, 4]);
            assert_eq!(ds.read_raw::<i32>().unwrap(), (0..16).collect::<Vec<i32>>());
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn subframe_chunking_roundtrip() {
        // A chunk smaller than a frame: shape [N,8,8], chunk [1,4,4], so each
        // frame is tiled into a 2x2 grid of 4x4 chunks. write_chunk_at takes
        // the chunk-grid coordinates.
        let path = temp_path("subframe");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, 8, 8])
                .chunk(&[1, 4, 4])
                .max_shape(&[None, Some(8), Some(8)])
                .create("v")
                .unwrap();
            for f in 0..3usize {
                for cr in 0..2usize {
                    for cc in 0..2usize {
                        let mut bytes = Vec::new();
                        for i in 0..4usize {
                            for j in 0..4usize {
                                let v = (f * 64 + (cr * 4 + i) * 8 + (cc * 4 + j)) as i32;
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                        ds.write_chunk_at(&[f, cr, cc], &bytes).unwrap();
                    }
                }
            }
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("v").unwrap();
            assert_eq!(ds.shape(), vec![3, 8, 8]);
            assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                (0..192).collect::<Vec<i32>>()
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fill_value_contiguous_roundtrip() {
        let path = temp_path("fill_value_contig");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f32>()
                .shape([4])
                .fill_value(2.5f32)
                .create("data")
                .unwrap();
            ds.write_raw(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
            file.close().unwrap();
        }
        // open_append decodes the fill-value message back from the header.
        {
            let writer = crate::io::writer::Hdf5Writer::open_append(&path).unwrap();
            let idx = writer.dataset_index("data").unwrap();
            assert_eq!(
                writer.datasets[idx].fill_value,
                Some(2.5f32.to_le_bytes().to_vec())
            );
        }
        // Data still reads back correctly.
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.read_raw::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fill_value_chunked_roundtrip() {
        let path = temp_path("fill_value_chunked");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .fill_value(-7i32)
                .create("vals")
                .unwrap();
            ds.append(&[1i32, 2, 3, 4]).unwrap();
            file.close().unwrap();
        }
        {
            let writer = crate::io::writer::Hdf5Writer::open_append(&path).unwrap();
            let idx = writer.dataset_index("vals").unwrap();
            assert_eq!(
                writer.datasets[idx].fill_value,
                Some((-7i32).to_le_bytes().to_vec())
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fill_value_read_missing_chunks() {
        // A chunked dataset with chunk 1 left unwritten must read that
        // gap back as the user-defined fill value, not zero.
        fn i32_bytes(vals: &[i32]) -> Vec<u8> {
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        let path = temp_path("fill_value_read_missing");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[2])
                .max_shape(&[None])
                .fill_value(-1i32)
                .create("vals")
                .unwrap();
            // chunk 0 = [10,20]; chunk 1 unwritten; chunk 2 = [50,60].
            ds.write_chunk(0, &i32_bytes(&[10, 20])).unwrap();
            ds.write_chunk(2, &i32_bytes(&[50, 60])).unwrap();
            ds.extend(&[6]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("vals").unwrap();
            let all = ds.read_raw::<i32>().unwrap();
            assert_eq!(all, vec![10, 20, -1, -1, 50, 60]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fill_value_partial_chunk_padded_with_fill() {
        // A partial trailing chunk flushed at close must pad its unwritten
        // tail with the fill value. That pad sits beyond the logical shape,
        // so it is verified by scanning the on-disk chunk bytes directly.
        let path = temp_path("fill_value_partial_pad");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .fill_value(-9i32)
                .create("vals")
                .unwrap();
            // 3 of 4 frames -> flushed as a partial chunk on close.
            ds.append(&[1i32, 2, 3]).unwrap();
            file.close().unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        // Locate the chunk: i32 LE of [1, 2, 3] written contiguously.
        let needle: Vec<u8> = [1i32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let pos = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("chunk data [1,2,3] not found in file");
        let pad = &bytes[pos + needle.len()..pos + needle.len() + 4];
        assert_eq!(
            pad,
            &(-9i32).to_le_bytes(),
            "partial chunk tail must be padded with fill value -9, got {:?}",
            pad
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vlen_append_after_reopen_preserves_existing() {
        // Reopening and appending into a partially-written vlen chunk must
        // read-modify-write: the strings already on disk must survive.
        let path = temp_path("vlen_append_reopen");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("strs", 4, None)
                .unwrap();
            // 3 of 4 frames -> flushed as a partial chunk on close.
            file.append_vlen_strings("strs", &["a", "b", "c"]).unwrap();
            file.close().unwrap();
        }
        {
            // Append a 4th string -> partial-chunk write into chunk 0.
            let file = H5File::open_rw(&path).unwrap();
            file.append_vlen_strings("strs", &["d"]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("strs").unwrap();
            let got = ds.read_vlen_strings().unwrap();
            assert_eq!(
                got.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                vec!["a", "b", "c", "d"]
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn fill_value_size_mismatch_errors() {
        let path = temp_path("fill_value_mismatch");
        let mut writer = crate::io::writer::Hdf5Writer::create(&path).unwrap();
        let dt = <f64 as crate::types::H5Type>::hdf5_type();
        let idx = writer.create_dataset("d", dt, &[4u64]).unwrap();
        // f64 element size is 8; a 4-byte fill value must be rejected.
        assert!(writer.set_dataset_fill_value(idx, vec![0u8; 4]).is_err());
        // The correct width succeeds.
        writer.set_dataset_fill_value(idx, vec![0u8; 8]).unwrap();
        writer.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn datatype_exposes_class_sign_and_byteorder() {
        // The byte width alone cannot tell u8 from i8 (both 1 byte) or i32
        // from f32 (both 4 bytes). datatype() must report the real class and
        // signedness so a reader does not have to guess from element_size.
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};

        let path = temp_path("datatype_accessor");
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<u8>().shape([3]).create("u8d").unwrap();
            file.new_dataset::<i8>().shape([3]).create("i8d").unwrap();
            file.new_dataset::<i32>().shape([3]).create("i32d").unwrap();
            file.new_dataset::<f32>().shape([3]).create("f32d").unwrap();
            file.close().unwrap();
        }

        let file = H5File::open(&path).unwrap();

        match file.dataset("u8d").unwrap().datatype().unwrap() {
            DatatypeMessage::FixedPoint {
                size,
                signed,
                byte_order,
                ..
            } => {
                assert_eq!(size, 1);
                assert!(!signed, "u8 must be unsigned");
                assert_eq!(byte_order, ByteOrder::LittleEndian);
            }
            other => panic!("expected FixedPoint for u8, got {other:?}"),
        }

        match file.dataset("i8d").unwrap().datatype().unwrap() {
            DatatypeMessage::FixedPoint { size, signed, .. } => {
                assert_eq!(size, 1);
                assert!(signed, "i8 must be signed");
            }
            other => panic!("expected FixedPoint for i8, got {other:?}"),
        }

        match file.dataset("i32d").unwrap().datatype().unwrap() {
            DatatypeMessage::FixedPoint { size, signed, .. } => {
                assert_eq!(size, 4);
                assert!(signed, "i32 must be signed");
            }
            other => panic!("expected FixedPoint for i32, got {other:?}"),
        }

        match file.dataset("f32d").unwrap().datatype().unwrap() {
            DatatypeMessage::FloatingPoint { size, .. } => assert_eq!(size, 4),
            other => panic!("expected FloatingPoint for f32, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn datatype_in_write_mode_errors() {
        let path = temp_path("datatype_write_mode");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<f32>().shape([4]).create("d").unwrap();
        assert!(ds.datatype().is_err());
        std::fs::remove_file(&path).ok();
    }

    // --- write_chunk_raw (HDF5 direct chunk write) ---------------------------

    /// Extensible-array path: pre-compress with the dataset's pipeline, write
    /// the bytes verbatim via write_chunk_raw (filter_mask = 0), and confirm
    /// the data round-trips through the reader unchanged.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_ea_roundtrip_mask0() {
        use crate::format::messages::filter::{apply_filters, FilterPipeline};
        let path = temp_path("wcr_ea_mask0");
        let original: Vec<i32> = (0..12).collect();
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .deflate(4)
                .create("v")
                .unwrap();
            assert!(ds.is_chunked());
            let pipeline = FilterPipeline::deflate(4);
            for c in 0..3usize {
                let raw: Vec<u8> = original[c * 4..c * 4 + 4]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let compressed = apply_filters(&pipeline, &raw).unwrap();
                ds.write_chunk_raw(c, &compressed, 0).unwrap();
            }
            ds.set_extent(&[12]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v, original);
        }
        std::fs::remove_file(&path).ok();
    }

    /// Fixed-array path (all dimensions bounded): same verbatim write through
    /// the linear-index dispatch, round-tripped through the reader.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_fixed_array_roundtrip_mask0() {
        use crate::format::messages::filter::{apply_filters, FilterPipeline};
        let path = temp_path("wcr_fa_mask0");
        let original: Vec<i32> = (0..12).collect();
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([12])
                .chunk(&[4])
                .deflate(4)
                .create("v")
                .unwrap();
            assert!(ds.is_chunked());
            let pipeline = FilterPipeline::deflate(4);
            for c in 0..3usize {
                let raw: Vec<u8> = original[c * 4..c * 4 + 4]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let compressed = apply_filters(&pipeline, &raw).unwrap();
                ds.write_chunk_raw(c, &compressed, 0).unwrap();
            }
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v, original);
        }
        std::fs::remove_file(&path).ok();
    }

    /// The caller-supplied filter_mask must reach the on-disk filtered index
    /// entry (not be hardcoded to 0). Store one chunk uncompressed in a
    /// filtered dataset with mask = 1 (deflate skipped), then reopen and decode
    /// the extensible-array filtered entry to read the mask back at the format
    /// level (independent of the data reader's mask handling).
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_records_filter_mask() {
        let path = temp_path("wcr_records_mask");
        let raw: Vec<u8> = [10i32, 20, 30, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(raw.len(), 16);
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .deflate(4)
                .create("v")
                .unwrap();
            // mask = 1: bit 0 set => filter 0 (deflate) was skipped, so the
            // chunk is stored uncompressed (its raw bytes).
            ds.write_chunk_raw(0, &raw, 1).unwrap();
            ds.set_extent(&[4]).unwrap();
            file.close().unwrap();
        }
        // Reopen the writer; open_append decodes the filtered index block from
        // disk, so the entry reflects exactly what was committed.
        {
            let w = crate::io::writer::Hdf5Writer::open_append(&path).unwrap();
            let idx = w.dataset_index("v").unwrap();
            let entry = &w.datasets[idx]
                .chunked
                .as_ref()
                .unwrap()
                .filt_iblk
                .as_ref()
                .unwrap()
                .elements[0];
            assert_eq!(entry.filter_mask, 1, "filter_mask must round-trip to disk");
            assert_eq!(entry.nbytes, 16, "uncompressed chunk stored verbatim");
        }
        std::fs::remove_file(&path).ok();
    }

    /// Reader honors a per-chunk filter_mask (EA): one chunk is stored
    /// compressed (mask 0), the next stored raw with deflate skipped (mask 1),
    /// in the same dataset. A correct reader skips deflate for chunk 1 only;
    /// ignoring the mask would feed raw bytes through inflate and corrupt them.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_ea_per_chunk_mask_roundtrip() {
        use crate::format::messages::filter::{apply_filters, FilterPipeline};
        let path = temp_path("wcr_ea_per_chunk_mask");
        let original: Vec<i32> = (0..8).collect();
        let pipeline = FilterPipeline::deflate(4);
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .deflate(4)
                .create("v")
                .unwrap();
            let raw0: Vec<u8> = original[0..4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            // chunk 0: compressed through the pipeline, mask 0.
            ds.write_chunk_raw(0, &apply_filters(&pipeline, &raw0).unwrap(), 0)
                .unwrap();
            let raw1: Vec<u8> = original[4..8]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            // chunk 1: stored uncompressed, mask 1 (deflate skipped).
            ds.write_chunk_raw(1, &raw1, 1).unwrap();
            ds.set_extent(&[8]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v, original);
        }
        std::fs::remove_file(&path).ok();
    }

    /// Reader honors a per-chunk filter_mask (fixed array): same mixed
    /// compressed/raw chunks as the EA case, through the fixed-array index.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_fixed_array_per_chunk_mask_roundtrip() {
        use crate::format::messages::filter::{apply_filters, FilterPipeline};
        let path = temp_path("wcr_fa_per_chunk_mask");
        let original: Vec<i32> = (0..8).collect();
        let pipeline = FilterPipeline::deflate(4);
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([8])
                .chunk(&[4])
                .deflate(4)
                .create("v")
                .unwrap();
            let raw0: Vec<u8> = original[0..4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            ds.write_chunk_raw(0, &apply_filters(&pipeline, &raw0).unwrap(), 0)
                .unwrap();
            let raw1: Vec<u8> = original[4..8]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            ds.write_chunk_raw(1, &raw1, 1).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let v = file.dataset("v").unwrap().read_raw::<i32>().unwrap();
            assert_eq!(v, original);
        }
        std::fs::remove_file(&path).ok();
    }

    /// An unfiltered chunk index has no slot for a stored size or mask, so a
    /// direct chunk write must be rejected rather than silently dropping them.
    #[test]
    fn write_chunk_raw_rejects_unfiltered() {
        let path = temp_path("wcr_unfiltered");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0])
            .chunk(&[4])
            .max_shape(&[None])
            .create("v")
            .unwrap();
        let err = ds.write_chunk_raw(0, &[0u8; 16], 0).unwrap_err();
        assert!(
            err.to_string().contains("filtered dataset"),
            "expected a filtered-dataset error, got: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// v2-B-tree-indexed datasets (two or more unlimited dimensions) do not
    /// support direct chunk writes.
    #[test]
    fn write_chunk_raw_rejects_btree_v2() {
        let path = temp_path("wcr_btree2");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0, 0])
            .chunk(&[2, 2])
            .max_shape(&[None, None])
            .create("grid")
            .unwrap();
        let err = ds.write_chunk_raw(0, &[0u8; 16], 0).unwrap_err();
        assert!(
            err.to_string().contains("v2-B-tree"),
            "expected a v2-B-tree rejection, got: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A stored size that does not fit the index's chunk-size field must error
    /// (libhdf5 H5D_CHUNK_ENCODE_SIZE_CHECK) instead of truncating silently.
    /// A 4-byte chunk (chunk[1] of i32) has chunk_size_len = 2 (max 65535), so
    /// a 70000-byte stored chunk overflows it.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_rejects_oversized_chunk() {
        let path = temp_path("wcr_oversized");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0])
            .chunk(&[1])
            .max_shape(&[None])
            .deflate(4)
            .create("v")
            .unwrap();
        let err = ds.write_chunk_raw(0, &vec![0u8; 70000], 0).unwrap_err();
        assert!(
            err.to_string().contains("does not fit"),
            "expected a chunk-size-field overflow error, got: {err}"
        );
        std::fs::remove_file(&path).ok();
    }
}
