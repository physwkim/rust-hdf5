//! Dataset creation and I/O.
//!
//! Datasets are created via the fluent [`DatasetBuilder`] API obtained from
//! [`H5File::new_dataset`](crate::file::H5File::new_dataset). Once created,
//! the [`H5Dataset`] handle can read or write raw typed data.

use crate::attribute::AttrBuilder;
use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner, borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
use crate::format::reference::Reference;
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
    is_null: bool,
    chunk_dims: Option<Vec<usize>>,
    max_shape: Option<Vec<Option<usize>>>,
    is_compact: bool,
    deflate_level: Option<u32>,
    shuffle: bool,
    custom_pipeline: Option<crate::format::messages::filter::FilterPipeline>,
    group_path: Option<String>,
    fill_value: Option<Vec<u8>>,
    datatype_override: Option<crate::format::messages::datatype::DatatypeMessage>,
    object_references: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: H5Type> DatasetBuilder<T> {
    pub(crate) fn new(file_inner: SharedInner) -> Self {
        Self {
            file_inner,
            shape: None,
            is_null: false,
            chunk_dims: None,
            max_shape: None,
            is_compact: false,
            deflate_level: None,
            shuffle: false,
            custom_pipeline: None,
            group_path: None,
            fill_value: None,
            datatype_override: None,
            object_references: false,
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn new_in_group(file_inner: SharedInner, group_path: String) -> Self {
        Self {
            file_inner,
            shape: None,
            is_null: false,
            chunk_dims: None,
            max_shape: None,
            is_compact: false,
            deflate_level: None,
            shuffle: false,
            custom_pipeline: None,
            group_path: Some(group_path),
            fill_value: None,
            datatype_override: None,
            object_references: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the dataset dimensions.
    ///
    /// This is required before calling [`create`](Self::create), unless
    /// [`null`](Self::null) was called instead.
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

    /// Create a dataset with the NULL dataspace: no elements at all.
    ///
    /// Distinct from [`scalar`](Self::scalar), which holds exactly one
    /// element. A NULL dataset holds zero bytes of data and cannot be
    /// written to — [`write_raw`](H5Dataset::write_raw) and
    /// [`write_raw_bytes`](H5Dataset::write_raw_bytes) return an error, and
    /// it cannot be chunked or filtered, matching h5py's `h5py.Empty`.
    #[must_use]
    pub fn null(mut self) -> Self {
        self.is_null = true;
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

    /// Store the raw data inside the dataset's object header —
    /// `H5Pset_layout(dcpl, H5D_COMPACT)`.
    ///
    /// A compact dataset costs no data block and no second seek to read, which
    /// suits the small per-run constants an analysis file is full of. It is
    /// bounded by what one object header message can hold
    /// ([`MAX_COMPACT_DATA`](crate::MAX_COMPACT_DATA) bytes) and it
    /// is fixed in size: [`chunk`](Self::chunk), a filter, and an unlimited
    /// [`max_shape`](Self::max_shape) are all rejected at
    /// [`create`](Self::create), as libhdf5 rejects them.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("compact.h5").unwrap();
    /// let ds = file.new_dataset::<i32>()
    ///     .shape([16])
    ///     .compact()
    ///     .create("data")
    ///     .unwrap();
    /// ds.write_raw(&(0..16i32).collect::<Vec<_>>()).unwrap();
    /// ```
    #[must_use]
    pub fn compact(mut self) -> Self {
        self.is_compact = true;
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

    /// Enable the shuffle filter — `H5Pset_shuffle(dcpl)`, h5py's
    /// `shuffle=True`.
    ///
    /// Shuffle reorders a chunk's bytes by their position within an element,
    /// which typically improves how well a compressor behind it does on
    /// numeric data. It is a permutation, not a compressor: on its own it
    /// leaves the chunk exactly as large as it was, which is what
    /// `H5Pset_shuffle` without a compressor writes. Combine it with
    /// [`deflate`](Self::deflate) to compress the shuffled stream. Requires
    /// chunked storage.
    ///
    /// The element width the filter records is the dataset's, so a
    /// [`datatype`](Self::datatype) override is what it follows when the
    /// stored element is not `T` itself.
    #[must_use]
    pub fn shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    /// Enable shuffle + deflate compression — the same pipeline as
    /// `.shuffle().deflate(level)`.
    ///
    /// Shuffle reorders bytes by position within elements before compression,
    /// which typically improves compression ratios for numeric data.
    /// Requires chunked storage.
    #[must_use]
    pub fn shuffle_deflate(mut self, level: u32) -> Self {
        self.shuffle = true;
        self.deflate_level = Some(level);
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

    /// Override the stored element datatype.
    ///
    /// By default the dataset is created with the datatype derived from the
    /// Rust type parameter `T` ([`H5Type::hdf5_type`]). Use this to store a
    /// different on-disk datatype than the in-memory element type — for
    /// example a reduced-precision fixed-point type that matches an N-bit
    /// filter (see [`FilterPipeline::nbit`]). The element *byte* size of the
    /// override must equal `T::element_size()`; the N-bit filter packs the
    /// significant bits within that fixed footprint.
    ///
    /// [`H5Type::hdf5_type`]: crate::H5Type::hdf5_type
    /// [`FilterPipeline::nbit`]: crate::FilterPipeline::nbit
    #[must_use]
    pub fn datatype(mut self, dt: crate::format::messages::datatype::DatatypeMessage) -> Self {
        self.datatype_override = Some(dt);
        self
    }

    /// Store object references — h5py's `h5py.ref_dtype`.
    ///
    /// The elements are written with
    /// [`write_object_references`](H5Dataset::write_object_references) and
    /// name objects by path. The element width is the file's address size, so
    /// the datatype is resolved at [`create`](Self::create) rather than here;
    /// it overrides both `T` and any [`datatype`](Self::datatype) call.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::create("refs.h5").unwrap();
    /// file.new_dataset::<i32>().shape([4]).create("target").unwrap();
    /// let refs = file.new_dataset::<u64>()
    ///     .object_references()
    ///     .shape([1])
    ///     .create("refs")
    ///     .unwrap();
    /// refs.write_object_references(&["/target"]).unwrap();
    /// file.close().unwrap();
    /// ```
    #[must_use]
    pub fn object_references(mut self) -> Self {
        self.object_references = true;
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

        let datatype = if self.object_references {
            // The element is one file address wide, and only the writer knows
            // how wide that is for this file.
            let inner = borrow_inner(&self.file_inner);
            match &*inner {
                H5FileInner::Writer(writer) => {
                    crate::format::messages::datatype::DatatypeMessage::object_reference(
                        writer.ctx(),
                    )
                }
                H5FileInner::Reader(_) => {
                    return Err(Hdf5Error::InvalidState(
                        "cannot create a dataset in read mode".into(),
                    ))
                }
                H5FileInner::Closed => {
                    return Err(Hdf5Error::InvalidState("file is closed".into()))
                }
            }
        } else {
            self.datatype_override.clone().unwrap_or_else(T::hdf5_type)
        };
        // Size one element from the on-disk datatype, not the carrier `T`. For
        // the default path this equals `T::element_size()`; when a `datatype()`
        // override is set (N-bit, or a runtime `CompoundType`), the stored type
        // — not `T` — defines the element width, so the dataspace, the raw
        // allocation, and the `write_raw` length check all agree with the bytes
        // libhdf5/h5py will read.
        let element_size = datatype.element_size() as usize;
        // `fill_value` took the host image of a `T`; the fill-value message
        // holds one element in the dataset's own datatype, so it is converted
        // here — the order is only known once the override is resolved, and
        // the builder's calls can arrive in either order.
        let fill_value = match self.fill_value.as_deref() {
            Some(bytes) => Some(to_stored_byte_order(bytes, &datatype, element_size)?.into_owned()),
            None => None,
        };

        let wants_filter =
            self.custom_pipeline.is_some() || self.shuffle || self.deflate_level.is_some();

        if self.is_null {
            // A NULL dataspace holds no elements at all: no chunk grid to
            // scatter into, no raw image to put in an object header, no fill
            // value to apply to unwritten elements (there are none), matching
            // upstream's rejection of these combinations (`H5Dchunk.c`'s
            // chunked-layout dataspace check).
            if self.chunk_dims.is_some() || wants_filter || self.is_compact {
                return Err(Hdf5Error::InvalidState(
                    "a NULL dataspace dataset cannot be chunked, filtered or compact".into(),
                ));
            }
            if fill_value.is_some() {
                return Err(Hdf5Error::InvalidState(
                    "a NULL dataspace dataset cannot have a fill value".into(),
                ));
            }

            let index = {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        let idx = writer.create_null_dataset(&full_name, datatype)?;
                        if let Some(ref gp) = self.group_path {
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

            return Ok(H5Dataset {
                file_inner: clone_inner(&self.file_inner),
                info: DatasetInfo::Writer {
                    index,
                    shape: Vec::new(),
                    element_size,
                    chunked: false,
                    btree2: false,
                    fixed_array: false,
                    is_null: true,
                },
            });
        }

        let shape = self.shape.ok_or_else(|| {
            Hdf5Error::InvalidState("shape must be set before calling create()".into())
        })?;
        let dims_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

        if self.is_compact {
            // The raw data is the layout message, so there is no chunk grid to
            // filter and no room to grow into: `H5D__compact_construct` refuses
            // a max dimension above the current one, and `H5Pset_layout` and
            // `H5Pset_chunk` overwrite each other rather than combining.
            if self.chunk_dims.is_some() || wants_filter {
                return Err(Hdf5Error::InvalidState(
                    "a compact dataset stores its data in the object header, so it \
                     cannot be chunked or filtered"
                        .into(),
                ));
            }
            if self
                .max_shape
                .as_ref()
                .is_some_and(|max| max.iter().zip(&shape).any(|(m, &d)| *m != Some(d)))
            {
                return Err(Hdf5Error::InvalidState(
                    "a compact dataset cannot be extendible: its maximum shape must \
                     equal its shape"
                        .into(),
                ));
            }

            let index = {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        let idx = writer.create_compact_dataset(&full_name, datatype, &dims_u64)?;
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

            return Ok(H5Dataset {
                file_inner: clone_inner(&self.file_inner),
                info: DatasetInfo::Writer {
                    index,
                    shape,
                    element_size,
                    chunked: false,
                    btree2: false,
                    fixed_array: false,
                    is_null: false,
                },
            });
        }

        // A filter pipeline requires chunked storage. When a filter is
        // requested without explicit chunk dimensions, store the whole
        // dataset as a single chunk instead of silently dropping the filter
        // on the contiguous path. (This is one whole-dataset chunk, not
        // h5py's ~1 MiB chunk-size heuristic; pass explicit chunk dimensions
        // for large datasets.)
        let auto_chunk: Option<Vec<usize>> =
            if self.chunk_dims.is_none() && wants_filter && !shape.is_empty() {
                Some(shape.iter().map(|&d| d.max(1)).collect())
            } else {
                None
            };

        if let Some(chunk_dims) = self.chunk_dims.as_ref().or(auto_chunk.as_ref()) {
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

            let index = {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        // The requested filter pipeline, if any. Every index
                        // builds it from the same options, so one owner
                        // resolves it: a second construction site is what let
                        // a request naming no compressor — shuffle on its own
                        // — fall through to unfiltered storage.
                        let explicit_pipeline = || {
                            use crate::format::messages::filter::FilterPipeline;
                            if let Some(p) = self.custom_pipeline.clone() {
                                return p;
                            }
                            // Shuffle records the width of the element it
                            // permutes, which is the stored one — a `datatype`
                            // override moves that away from `T`.
                            let es = element_size as u32;
                            match (self.shuffle, self.deflate_level) {
                                (true, Some(level)) => FilterPipeline::shuffle_deflate(es, level),
                                (true, None) => FilterPipeline::shuffle(es),
                                // deflate_level (checked by wants_filter).
                                (false, level) => FilterPipeline::deflate(level.unwrap()),
                            }
                        };
                        let idx = if is_btree2 {
                            // Two or more unlimited dimensions: a v2 B-tree,
                            // whose records carry the stored size and filter
                            // mask when the dataset is compressed (libhdf5
                            // H5D_BT2_FILT).
                            if wants_filter {
                                writer.create_btree_v2_dataset_with_pipeline(
                                    &full_name,
                                    datatype,
                                    &dims_u64,
                                    &max_u64,
                                    &chunk_u64,
                                    explicit_pipeline(),
                                )?
                            } else {
                                writer.create_btree_v2_dataset(
                                    &full_name, datatype, &dims_u64, &max_u64, &chunk_u64,
                                )?
                            }
                        } else if is_fixed_array {
                            // A chunked dataset with no unlimited dimension
                            // must use the fixed-array index — libhdf5
                            // rejects an extensible-array index here. A
                            // compressed fixed-shape dataset uses a *filtered*
                            // fixed array (FA client id 1). The maximum shape
                            // sizes the array, so a finite max above the
                            // current shape stays growable.
                            let pipeline = wants_filter.then(explicit_pipeline);
                            writer.create_fixed_array_dataset_with_max(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64, pipeline,
                            )?
                        } else if wants_filter {
                            // The extensible-array index takes the pipeline
                            // the same way, so it goes through the one owner
                            // too — `create_chunked_dataset_compressed` is
                            // this call with a deflate-only pipeline.
                            writer.create_chunked_dataset_with_pipeline(
                                &full_name,
                                datatype,
                                &dims_u64,
                                &max_u64,
                                &chunk_u64,
                                explicit_pipeline(),
                            )?
                        } else {
                            writer.create_chunked_dataset(
                                &full_name, datatype, &dims_u64, &max_u64, &chunk_u64,
                            )?
                        };
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
                    is_null: false,
                },
            })
        } else {
            // Contiguous dataset (original path)
            let index = {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        let idx = writer.create_dataset(&full_name, datatype, &dims_u64)?;
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
                    is_null: false,
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
        /// Whether this is a NULL dataspace (no elements at all — distinct
        /// from a scalar, which holds exactly one). Always `false` when
        /// `chunked` is `true`: a NULL dataspace can never be chunked.
        is_null: bool,
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

/// One chunk's bytes on the way to the file, and who filtered them.
///
/// This is what separates a normal chunk write from a direct one; everything
/// else about placing a chunk is identical, so the two share a single dispatch.
#[derive(Clone, Copy)]
enum ChunkBytes<'a> {
    /// The chunk's raw bytes; the dataset's filter pipeline runs before they
    /// are stored.
    Unfiltered(&'a [u8]),
    /// Bytes already in their stored form, with `filter_mask` naming the
    /// filters that were skipped.
    Prefiltered { data: &'a [u8], filter_mask: u32 },
}

/// The byte order this build reads and writes natively.
pub(crate) const HOST_BYTE_ORDER: ByteOrder = if cfg!(target_endian = "big") {
    ByteOrder::BigEndian
} else {
    ByteOrder::LittleEndian
};

/// The byte order this build does not read or write natively.
pub(crate) const FOREIGN_BYTE_ORDER: ByteOrder = match HOST_BYTE_ORDER {
    ByteOrder::LittleEndian => ByteOrder::BigEndian,
    ByteOrder::BigEndian => ByteOrder::LittleEndian,
};

/// What a typed access has to do with an element image of a given datatype.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ByteOrderAction {
    /// Stored order is the host's: the image is already the typed value.
    Keep,
    /// The whole element is one scalar in the foreign order: reverse it.
    SwapElements,
    /// A composite storing something in the foreign order.
    Refuse,
}

/// Classify a datatype for a typed access of element width `width`.
///
/// The single owner of the rule; both directions ask it, so a type a read
/// converts is exactly a type a write converts.
///
/// A composite element cannot be swapped as a unit — its members have their
/// own orders and offsets — so one that touches the foreign order is refused
/// rather than silently passed through in the wrong order.
fn byte_order_action(datatype: &DatatypeMessage, width: usize) -> ByteOrderAction {
    match datatype.scalar_byte_order() {
        Some(order) if order == FOREIGN_BYTE_ORDER && width > 1 => ByteOrderAction::SwapElements,
        Some(_) => ByteOrderAction::Keep,
        None if datatype.contains_byte_order(FOREIGN_BYTE_ORDER) => ByteOrderAction::Refuse,
        None => ByteOrderAction::Keep,
    }
}

/// Put a raw element image into host byte order, in place, for a typed read.
///
/// Every path that reinterprets the on-disk image as `T` — `read_raw`,
/// `read_slice`, `read_raw_into`, `read_slice_into` and their SWMR
/// counterparts — passes through here. Reinterpretation only yields the
/// stored value when the stored order is the host's.
///
/// A refused datatype is one no reinterpretation can decode;
/// [`H5Dataset::read_raw_bytes`] hands over the image for the caller to
/// decode member by member.
///
/// `width` is the element size, already checked equal to `T::element_size()`.
pub(crate) fn to_host_byte_order(
    bytes: &mut [u8],
    datatype: &DatatypeMessage,
    width: usize,
) -> Result<()> {
    match byte_order_action(datatype, width) {
        ByteOrderAction::Keep => {}
        ByteOrderAction::SwapElements => {
            for elem in bytes.chunks_exact_mut(width) {
                elem.reverse();
            }
        }
        ByteOrderAction::Refuse => {
            return Err(Hdf5Error::TypeMismatch(format!(
                "dataset datatype {datatype} stores {FOREIGN_BYTE_ORDER:?} values, which a \
                 typed read cannot reinterpret element by element; read_raw_bytes() returns \
                 the image to decode member by member"
            )))
        }
    }
    Ok(())
}

/// The stored byte image of each variable-length sequence in a batch.
///
/// The vlen writers take `&[&[T]]` and store one global-heap object per
/// sequence, so each sequence needs the same host-image-to-stored-image step
/// [`to_stored_byte_order`] performs for a fixed-shape write — a `T` is
/// written from its host bytes, and `T::hdf5_type()` declares little-endian.
/// Borrows on a little-endian host, which is every machine that does not have
/// to swap.
pub(crate) fn vlen_sequence_images<'a, T: H5Type>(
    items: &'a [&'a [T]],
) -> Result<Vec<std::borrow::Cow<'a, [u8]>>> {
    let base = T::hdf5_type();
    items
        .iter()
        .map(|item| {
            // Safety: the same contract `write_raw` relies on — `T: Copy +
            // 'static` is a numeric primitive whose byte image is its value —
            // and the extent comes from the slice itself, so it cannot name
            // memory past it. The result borrows `items` and outlives nothing.
            let host = unsafe {
                std::slice::from_raw_parts(item.as_ptr() as *const u8, std::mem::size_of_val(*item))
            };
            to_stored_byte_order(host, &base, T::element_size())
        })
        .collect()
}

/// Put a typed value's host-order image into the order the datatype declares.
///
/// The write-side counterpart of [`to_host_byte_order`], and the one place
/// every path that hands a `&[T]` to the file — `write_raw`, `write_slice`,
/// `append`, and the builder's fill value — turns those bytes into stored
/// bytes. A `T` is written from its host image, so a dataset declaring the
/// foreign order would otherwise hold host bytes under that declaration: a
/// file that is wrong by its own header.
///
/// Borrows when the declared order is the host's, which is every write that
/// does not set a [`datatype`](DatasetBuilder::datatype) override.
///
/// `width` is the element size, already checked equal to `T::element_size()`.
pub(crate) fn to_stored_byte_order<'a>(
    bytes: &'a [u8],
    datatype: &DatatypeMessage,
    width: usize,
) -> Result<std::borrow::Cow<'a, [u8]>> {
    match byte_order_action(datatype, width) {
        ByteOrderAction::Keep => Ok(std::borrow::Cow::Borrowed(bytes)),
        ByteOrderAction::SwapElements => {
            let mut owned = bytes.to_vec();
            for elem in owned.chunks_exact_mut(width) {
                elem.reverse();
            }
            Ok(std::borrow::Cow::Owned(owned))
        }
        ByteOrderAction::Refuse => Err(Hdf5Error::TypeMismatch(format!(
            "dataset datatype {datatype} stores {FOREIGN_BYTE_ORDER:?} values, which a typed \
             write cannot lay out element by element; write_raw_bytes() takes the image the \
             caller encodes member by member"
        ))),
    }
}

/// Strip a fixed-string element's padding, leaving the bytes that carry the
/// value.
///
/// The rule itself lives with the datatype message
/// ([`fixed_string_content`]); this adds the element index a reserved padding
/// rule needs to be reported against.
fn trim_fixed_string(elem: &[u8], padding: u8, index: usize) -> Result<&[u8]> {
    crate::format::messages::datatype::fixed_string_content(elem, padding).ok_or_else(|| {
        Hdf5Error::InvalidState(format!(
            "string {index} uses padding rule {padding}, which the format reserves"
        ))
    })
}

/// Decode one string element's bytes under the datatype's character set.
///
/// `lossy` replaces what it cannot decode with U+FFFD instead of failing;
/// `index` names the element in the error otherwise.
fn decode_string(bytes: &[u8], charset: u8, lossy: bool, index: usize) -> Result<String> {
    if lossy {
        return Ok(String::from_utf8_lossy(bytes).into_owned());
    }
    match charset {
        // ASCII. Bytes are 7-bit, which makes them UTF-8 as well.
        0 => match bytes.iter().position(|&b| b >= 0x80) {
            None => Ok(String::from_utf8_lossy(bytes).into_owned()),
            Some(at) => Err(Hdf5Error::InvalidState(format!(
                "string {index} declares the ASCII character set but byte {at} is {:#04x}",
                bytes[at]
            ))),
        },
        1 => String::from_utf8(bytes.to_vec()).map_err(|e| {
            Hdf5Error::InvalidState(format!(
                "string {index} declares UTF-8 but is not valid UTF-8: {e}"
            ))
        }),
        other => Err(Hdf5Error::InvalidState(format!(
            "string {index} uses character set {other}, which the format reserves"
        ))),
    }
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

    /// Create a writer-mode dataset handle for an already-created dataset
    /// (called internally by [`H5File::dataset_writer`](crate::file::H5File::dataset_writer)).
    ///
    /// Reconstructs the same handle `new_dataset().create()` returns, so the
    /// reopened dataset supports attribute writes and chunk appends.
    ///
    /// `is_null` is always `false` here: reopening an existing NULL-dataspace
    /// dataset for further writes is not a case this constructor's caller
    /// distinguishes (a NULL dataset has nothing to append or chunk-write in
    /// the first place).
    pub(crate) fn new_writer(
        file_inner: SharedInner,
        index: usize,
        shape: Vec<usize>,
        element_size: usize,
        chunked: bool,
        btree2: bool,
        fixed_array: bool,
    ) -> Self {
        Self {
            file_inner,
            info: DatasetInfo::Writer {
                index,
                shape,
                element_size,
                chunked,
                btree2,
                fixed_array,
                is_null: false,
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
    ///
    /// 0 for a NULL dataspace ([`is_null`](Self::is_null)) — unlike a scalar,
    /// whose `shape()` is the same empty `Vec` but which holds exactly one
    /// element, so `shape().iter().product()` cannot be used here.
    pub fn total_elements(&self) -> usize {
        if self.is_null() {
            return 0;
        }
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

    /// Return whether this dataset has the NULL dataspace: no elements at
    /// all, distinct from a scalar dataset (rank 0, exactly one element) —
    /// both report the same empty [`shape`](Self::shape). See
    /// [`DatasetBuilder::null`].
    pub fn is_null(&self) -> bool {
        match &self.info {
            DatasetInfo::Writer { is_null, .. } => *is_null,
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => reader
                        .dataset_info(name)
                        .map(|info| info.dataspace.is_null())
                        .unwrap_or(false),
                    _ => false,
                }
            }
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
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
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
                let mut inner = borrow_inner_mut(&self.file_inner);
                if let H5FileInner::Reader(reader) = &mut *inner {
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
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
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
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader.dataset_attr_names(name)?),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "attr_names not available in write mode".into(),
            )),
        }
    }

    /// Why the attribute `attr_name` on this dataset cannot be read, or `None`
    /// when it can be.
    ///
    /// An attribute whose message this crate cannot decode is still listed by
    /// [`attr_names`](Self::attr_names) — the object header carries it — and
    /// this says what stands in the way. Opening it through
    /// [`attr`](Self::attr) fails with the same text.
    pub fn attr_unreadable_reason(&self, attr_name: &str) -> Result<Option<String>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader
                        .dataset_attr_unreadable_reason(name, attr_name)
                        .map(str::to_string)),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "attr_unreadable_reason not available in write mode".into(),
            )),
        }
    }

    /// Why this dataset's attribute *set* cannot be listed, or `None` when it
    /// can be.
    ///
    /// The object-scope counterpart of
    /// [`attr_unreadable_reason`](Self::attr_unreadable_reason). A dense
    /// attribute set is indexed by name hash, so a heap or index that will not
    /// read yields no names to hang a per-attribute reason on;
    /// [`attr_names`](Self::attr_names) then returns the failure rather than a
    /// short list, and this reports it without an attribute name.
    pub fn attrs_unreadable_reason(&self) -> Result<Option<String>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader
                        .dataset_attrs_unreadable_reason(name)
                        .map(str::to_string)),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "attrs_unreadable_reason not available in write mode".into(),
            )),
        }
    }

    /// Open an attribute by name (read mode only).
    pub fn attr(&self, attr_name: &str) -> Result<crate::attribute::H5Attribute> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
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

    /// Write a typed slice holding the dataset's whole image.
    ///
    /// The slice length must match the total number of elements declared by
    /// the dataset shape. The data is reinterpreted as raw bytes and written
    /// to the file: to the contiguous data block, or — for a chunked dataset —
    /// scattered across its chunk grid, through the filter pipeline if one is
    /// set. To write only part of a dataset, use
    /// [`write_slice`](Self::write_slice).
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
                btree2,
                fixed_array,
                is_null,
            } => {
                if *is_null {
                    return Err(Hdf5Error::InvalidState(
                        "cannot write to a NULL dataspace dataset".into(),
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
                let host =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
                let datatype = {
                    let inner = borrow_inner(&self.file_inner);
                    match &*inner {
                        H5FileInner::Writer(writer) => writer.dataset_datatype(*index),
                        _ => {
                            return Err(Hdf5Error::InvalidState(
                                "file is no longer in write mode".into(),
                            ))
                        }
                    }
                };
                let stored = to_stored_byte_order(host, &datatype, T::element_size())?;

                if *chunked {
                    // A chunked dataset has no contiguous data block; scatter
                    // the full row-major image into its chunk grid and write
                    // each chunk through the dataset's filter pipeline.
                    return self.write_full_image_chunked(
                        *index,
                        *btree2,
                        *fixed_array,
                        &stored,
                        *element_size,
                    );
                }

                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_dataset_raw(*index, &stored)?;
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

    /// Write the raw byte image of the whole dataset directly.
    ///
    /// Takes the same layouts as [`write_raw`](Self::write_raw): a contiguous
    /// data block, or a chunk grid the image is scattered across.
    ///
    /// Unlike [`write_raw`](Self::write_raw), this is not generic over an
    /// `H5Type` carrier, so it works for element types that have no matching
    /// Rust primitive — in particular a runtime
    /// [`CompoundType`](crate::types::CompoundType) of arbitrary size set via
    /// [`DatasetBuilder::datatype`]. `bytes.len()` must equal
    /// `product(shape) * element_size`, where `element_size` is taken from the
    /// dataset's on-disk datatype.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// # use rust_hdf5::types::{CompoundType, H5Type};
    /// let file = H5File::create("c.h5").unwrap();
    /// let ct = CompoundType {
    ///     members: vec![
    ///         ("id".to_string(), i32::hdf5_type(), 0),
    ///         ("val".to_string(), f64::hdf5_type(), 4),
    ///     ],
    ///     total_size: 12,
    /// };
    /// let ds = file
    ///     .new_dataset::<u8>()
    ///     .datatype(ct.to_datatype())
    ///     .shape(&[2])
    ///     .create("records")
    ///     .unwrap();
    /// let mut bytes = Vec::new();
    /// bytes.extend_from_slice(&1i32.to_le_bytes());
    /// bytes.extend_from_slice(&2.5f64.to_le_bytes());
    /// bytes.extend_from_slice(&2i32.to_le_bytes());
    /// bytes.extend_from_slice(&3.5f64.to_le_bytes());
    /// ds.write_raw_bytes(&bytes).unwrap();
    /// ```
    pub fn write_raw_bytes(&self, bytes: &[u8]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                shape,
                element_size,
                chunked,
                btree2,
                fixed_array,
                is_null,
            } => {
                if *is_null {
                    return Err(Hdf5Error::InvalidState(
                        "cannot write to a NULL dataspace dataset".into(),
                    ));
                }
                let expected: usize = shape.iter().product::<usize>() * *element_size;
                if bytes.len() != expected {
                    return Err(Hdf5Error::InvalidState(format!(
                        "raw byte length {} does not match dataset size {} \
                         (product(shape) * element_size {})",
                        bytes.len(),
                        expected,
                        element_size,
                    )));
                }
                if *chunked {
                    // Scatter the full row-major image into the chunk grid
                    // (same path as write_raw, carrier-agnostic bytes).
                    return self.write_full_image_chunked(
                        *index,
                        *btree2,
                        *fixed_array,
                        bytes,
                        *element_size,
                    );
                }
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_dataset_raw(*index, bytes)?;
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

    /// Scatter a full row-major dataset image into its chunk grid, writing
    /// every chunk through the dataset's filter pipeline.
    ///
    /// This is the chunked counterpart of a single contiguous `write_dataset_raw`
    /// — it is how [`write_raw`](Self::write_raw) and
    /// [`write_raw_bytes`](Self::write_raw_bytes) populate a chunked dataset
    /// (including the single auto-chunk created when a filter is set without
    /// explicit chunk dimensions). Edge chunks are zero-padded to the full
    /// chunk footprint, exactly as libhdf5 stores them.
    fn write_full_image_chunked(
        &self,
        index: usize,
        btree2: bool,
        fixed_array: bool,
        bytes: &[u8],
        element_size: usize,
    ) -> Result<()> {
        let inner = borrow_inner(&self.file_inner);
        let writer = match &*inner {
            H5FileInner::Writer(w) => w,
            _ => {
                return Err(Hdf5Error::InvalidState(
                    "file is no longer in write mode".into(),
                ))
            }
        };
        // Whole-operation guard: the flush, the grid snapshot and the chunk
        // writes below must not interleave with a concurrent same-dataset
        // operation.
        let cell = writer.ds(index);
        let _op = cell.op.lock();
        // A buffered append tail would flush over the image at close; hand
        // it to the chunks first, the image below overwrites everything.
        writer.flush_append_buffer(index)?;
        let chunk_dims = writer
            .dataset_chunk_dims(index)
            .ok_or_else(|| Hdf5Error::InvalidState("dataset has no chunk info".into()))?
            .to_vec();
        let dims = writer.dataset_dims(index).to_vec();
        let rank = dims.len();

        // Chunk grid: number of chunks along each dimension (row-major).
        let mut grid = vec![0u64; rank];
        for d in 0..rank {
            grid[d] = if chunk_dims[d] > 0 {
                dims[d].div_ceil(chunk_dims[d])
            } else {
                0
            };
        }
        let total_chunks: u64 = grid.iter().product();

        // Decode the iteration counter into row-major coordinates over the
        // *current* image's chunk grid. This is only an odometer over the
        // chunks the image spans — the slot a chunk is recorded under comes
        // from the index grid (`Hdf5Writer::chunk_slot`), which the maximum
        // extent decides.
        let coords_of = |linear: u64| -> Vec<u64> {
            let mut rem = linear;
            let mut coords = vec![0u64; rank];
            for d in (0..rank).rev() {
                coords[d] = rem % grid[d];
                rem /= grid[d];
            }
            coords
        };

        if btree2 {
            // B-tree v2 stores chunks verbatim (unfiltered only in this
            // codebase), so there is no compression to parallelize; write one
            // chunk at a time.
            for linear in 0..total_chunks {
                let coords = coords_of(linear);
                let chunk_buf =
                    Self::gather_chunk(bytes, &dims, &chunk_dims, &coords, element_size);
                writer.write_chunk_btree_v2_inner(index, &coords, &chunk_buf)?;
            }
        } else {
            // Extensible array and fixed array both compress each chunk through
            // the filter pipeline. Gather chunks and write them through the
            // per-index batch path so the pipeline compresses them in parallel
            // (with the `parallel` feature). A fixed-size window bounds peak
            // memory instead of materializing every chunk at once; 256 keeps
            // every rayon worker fed while capping the transient buffers to
            // window * chunk bytes. The two indexes differ only in how a chunk
            // is addressed: EA by its linear grid index, FA by grid coordinates.
            const BATCH_WINDOW: u64 = 256;
            let mut start = 0u64;
            while start < total_chunks {
                let end = (start + BATCH_WINDOW).min(total_chunks);
                let items: Vec<(Vec<u64>, Vec<u8>)> = (start..end)
                    .map(|counter| {
                        let coords = coords_of(counter);
                        let buf =
                            Self::gather_chunk(bytes, &dims, &chunk_dims, &coords, element_size);
                        (coords, buf)
                    })
                    .collect();
                if fixed_array {
                    let pairs: Vec<(&[u64], &[u8])> = items
                        .iter()
                        .map(|(c, d)| (c.as_slice(), d.as_slice()))
                        .collect();
                    writer.write_chunks_fixed_array_batch_inner(index, &pairs)?;
                } else {
                    let mut pairs: Vec<(u64, &[u8])> = Vec::with_capacity(items.len());
                    for (c, d) in &items {
                        pairs.push((writer.chunk_slot(index, c)?, d.as_slice()));
                    }
                    writer.write_chunks_batch_inner(index, &pairs)?;
                }
                start = end;
            }
        }
        Ok(())
    }

    /// Gather one chunk's bytes from a row-major full-dataset image.
    ///
    /// `coords` are the chunk's grid coordinates. The returned buffer is
    /// exactly `product(chunk_dims) * element_size` bytes, zero-padded where
    /// the chunk extends past the dataset edge.
    fn gather_chunk(
        source: &[u8],
        dims: &[u64],
        chunk_dims: &[u64],
        coords: &[u64],
        element_size: usize,
    ) -> Vec<u8> {
        let rank = dims.len();
        let chunk_elems: u64 = chunk_dims.iter().product();
        let mut out = vec![0u8; chunk_elems as usize * element_size];
        if rank == 0 {
            // Scalar dataset: a single element, no chunking dimension.
            if source.len() >= element_size {
                out[..element_size].copy_from_slice(&source[..element_size]);
            }
            return out;
        }

        // Actual extent of this chunk along each dimension (edge chunks are
        // smaller than the nominal chunk shape).
        let mut extent = vec![0u64; rank];
        for d in 0..rank {
            let start = coords[d] * chunk_dims[d];
            let end = ((coords[d] + 1) * chunk_dims[d]).min(dims[d]);
            extent[d] = end.saturating_sub(start);
        }
        if extent.contains(&0) {
            return out; // nothing of the dataset falls in this chunk
        }

        // Row-major strides (in elements) for the source (over `dims`) and the
        // destination chunk buffer (over `chunk_dims`).
        let mut src_stride = vec![1u64; rank];
        let mut dst_stride = vec![1u64; rank];
        for d in (0..rank - 1).rev() {
            src_stride[d] = src_stride[d + 1] * dims[d + 1];
            dst_stride[d] = dst_stride[d + 1] * chunk_dims[d + 1];
        }

        // Copy one contiguous run along the last axis per outer multi-index.
        let last = rank - 1;
        let run = extent[last] as usize * element_size;
        let outer: u64 = extent[..last].iter().product::<u64>().max(1);
        let mut idx = vec![0u64; rank]; // local indices within the chunk extent
        for _ in 0..outer {
            let mut src_off = 0u64;
            let mut dst_off = 0u64;
            for d in 0..rank {
                let global = coords[d] * chunk_dims[d] + idx[d];
                src_off += global * src_stride[d];
                dst_off += idx[d] * dst_stride[d];
            }
            let s = src_off as usize * element_size;
            let dpos = dst_off as usize * element_size;
            out[dpos..dpos + run].copy_from_slice(&source[s..s + run]);

            // Advance the multi-index over axes [0..last); the last axis is the
            // contiguous run handled above.
            let mut d = last;
            while d > 0 {
                d -= 1;
                idx[d] += 1;
                if idx[d] < extent[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
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

                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        // One op: the slot decode and the write see the same
                        // extents.
                        let cell = writer.ds(*index);
                        let _op = cell.op.lock();
                        if *fixed_array {
                            // Fixed-array dataset: decode the index-grid slot
                            // into row-major grid coordinates.
                            let coords = writer.chunk_coords_from_slot(*index, chunk_idx as u64)?;
                            writer.write_chunk_fixed_array_inner(*index, &coords, data)?;
                        } else {
                            writer.write_chunk_inner(*index, chunk_idx as u64, data)?;
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
    /// has no slot to record a stored size or mask. A v2-B-tree-indexed dataset
    /// (two or more unlimited dimensions) has no fixed chunk grid to linearize
    /// against, so address its chunks with
    /// [`write_chunk_raw_at`](Self::write_chunk_raw_at) instead.
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
                        "this dataset uses a v2 B-tree chunk index; use \
                         write_chunk_raw_at with the chunk's grid coordinates"
                            .into(),
                    ));
                }

                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        // One op: the slot decode and the write see the same
                        // extents.
                        let cell = writer.ds(*index);
                        let _op = cell.op.lock();
                        if *fixed_array {
                            // Fixed-array dataset: decode the index-grid slot
                            // into row-major grid coordinates.
                            let coords = writer.chunk_coords_from_slot(*index, chunk_idx as u64)?;
                            writer.write_compressed_chunk_fixed_array_inner(
                                *index,
                                &coords,
                                data,
                                filter_mask,
                            )?;
                        } else {
                            writer.write_compressed_chunk_inner(
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
        self.write_chunk_at_inner(chunk_coords, ChunkBytes::Unfiltered(data), "write_chunk_at")
    }

    /// Write an already-filtered chunk **verbatim** to a chunked dataset,
    /// addressed by its chunk-grid coordinates.
    ///
    /// The coordinate-addressed twin of
    /// [`write_chunk_raw`](Self::write_chunk_raw), and the form a
    /// v2-B-tree-indexed dataset needs: with two or more unlimited dimensions
    /// there is no fixed chunk grid for a linear index to mean anything against.
    /// As with `write_chunk_at`, the dataset's logical dimensions are extended
    /// to cover the written chunk.
    ///
    /// `data` is the already-filtered bytes of one chunk — its length is the
    /// *stored* size — and `filter_mask` bit *i* set means filter *i* of the
    /// pipeline was **not** applied and must be skipped on read. Pass 0 when the
    /// full pipeline already ran upstream.
    ///
    /// The dataset must be chunked **and** filtered; an unfiltered chunk index
    /// has no slot to record a stored size or mask.
    pub fn write_chunk_raw_at(
        &self,
        chunk_coords: &[usize],
        data: &[u8],
        filter_mask: u32,
    ) -> Result<()> {
        self.write_chunk_at_inner(
            chunk_coords,
            ChunkBytes::Prefiltered { data, filter_mask },
            "write_chunk_raw_at",
        )
    }

    /// The single owner of coordinate-addressed chunk writes: validates the
    /// coordinates, grows the dataspace to cover them, and routes the bytes to
    /// whichever chunk index the dataset uses. Whether the filter pipeline runs
    /// here or already ran upstream is carried by `bytes`, not by a second copy
    /// of this dispatch.
    fn write_chunk_at_inner(
        &self,
        chunk_coords: &[usize],
        bytes: ChunkBytes<'_>,
        what: &str,
    ) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer {
                index,
                chunked,
                btree2,
                fixed_array,
                ..
            } => {
                if !*chunked {
                    return Err(Hdf5Error::InvalidState(format!(
                        "{what} is only for chunked datasets"
                    )));
                }
                let coords: Vec<u64> = chunk_coords.iter().map(|&c| c as u64).collect();
                let btree2 = *btree2;
                let fixed_array = *fixed_array;
                let inner = borrow_inner(&self.file_inner);
                let writer = match &*inner {
                    H5FileInner::Writer(w) => w,
                    _ => {
                        return Err(Hdf5Error::InvalidState(
                            "file is no longer in write mode".into(),
                        ))
                    }
                };
                // Whole-operation guard: the dims snapshot, the chunk write
                // and the extend below must not interleave with a concurrent
                // same-dataset operation.
                let cell = writer.ds(*index);
                let _op = cell.op.lock();
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
                    match bytes {
                        ChunkBytes::Unfiltered(data) => {
                            writer.write_chunk_fixed_array_inner(*index, &coords, data)?
                        }
                        ChunkBytes::Prefiltered { data, filter_mask } => writer
                            .write_compressed_chunk_fixed_array_inner(
                                *index,
                                &coords,
                                data,
                                filter_mask,
                            )?,
                    }
                    return Ok(());
                }

                if btree2 {
                    match bytes {
                        ChunkBytes::Unfiltered(data) => {
                            writer.write_chunk_btree_v2_inner(*index, &coords, data)?
                        }
                        ChunkBytes::Prefiltered { data, filter_mask } => writer
                            .write_compressed_chunk_btree_v2_inner(
                                *index,
                                &coords,
                                data,
                                filter_mask,
                            )?,
                    }
                } else {
                    // Extensible array: the chunk's index-grid slot (row-major
                    // against the maximum extent).
                    let linear = writer.chunk_slot(*index, &coords)?;
                    match bytes {
                        ChunkBytes::Unfiltered(data) => {
                            writer.write_chunk_inner(*index, linear, data)?
                        }
                        ChunkBytes::Prefiltered { data, filter_mask } => writer
                            .write_compressed_chunk_inner(*index, linear, data, filter_mask)?,
                    }
                }

                if new_dims != dims {
                    writer.extend_dataset_inner(*index, &new_dims)?;
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
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
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

                let inner = borrow_inner(&self.file_inner);
                let writer = match &*inner {
                    H5FileInner::Writer(w) => w,
                    _ => {
                        return Err(Hdf5Error::InvalidState(
                            "file is no longer in write mode".into(),
                        ))
                    }
                };

                // Whole-operation guard: the buffer take, the frame writes,
                // the re-buffer and the extend below are separate slot
                // acquisitions that a concurrent same-dataset append must not
                // interleave with.
                let cell = writer.ds(ds_index);
                let _op = cell.op.lock();

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

                let host = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * es)
                };
                let datatype = writer.dataset_datatype(ds_index);
                let raw = to_stored_byte_order(host, &datatype, es)?;

                // Merge the buffer with the new frames when it is the
                // dataset's tail; a buffer left mid-extent (the extent moved
                // past it) keeps its recorded place — flush it and start
                // fresh at the current end.
                let taken = { writer.ds(ds_index).lock().append.take() };
                let (base_dim0, buffered_frames, mut combined) = match taken {
                    Some(b) if b.base + b.frames == current_dim0 as u64 => {
                        (b.base as usize, b.frames as usize, b.bytes)
                    }
                    Some(b) => {
                        writer.write_append_frames(ds_index, b.base, b.frames, &b.bytes)?;
                        (current_dim0, 0, Vec::new())
                    }
                    None => (current_dim0, 0, Vec::new()),
                };
                combined.extend_from_slice(&raw);

                let total_frames = buffered_frames + n_new_frames;

                // Rows up to the last chunk boundary are written now; the
                // tail that does not complete a chunk goes back in the
                // buffer for the next append (or the flush at close). The
                // boundary can precede `base_dim0` — a reopened file's
                // flushed partial chunk leaves the base mid-chunk — in
                // which case everything is tail.
                let last_boundary = ((base_dim0 + total_frames) / chunk_dim0) * chunk_dim0;
                let write_frames = last_boundary.saturating_sub(base_dim0);
                let tail_frames = total_frames - write_frames;
                if write_frames > 0 {
                    writer.write_append_frames(
                        ds_index,
                        base_dim0 as u64,
                        write_frames as u64,
                        &combined[..write_frames * frame_bytes],
                    )?;
                }
                if tail_frames > 0 {
                    let ds = writer.ds(ds_index);
                    let mut m = ds.lock();
                    m.append = Some(crate::io::writer::AppendBuffer {
                        base: (base_dim0 + write_frames) as u64,
                        frames: tail_frames as u64,
                        bytes: combined[write_frames * frame_bytes..].to_vec(),
                    });
                }

                // Extend dims to include all frames (buffered + new)
                let logical_dim0 = base_dim0 + total_frames;
                let mut new_dims: Vec<u64> = dims;
                new_dims[0] = logical_dim0 as u64;
                writer.extend_dataset_inner(ds_index, &new_dims)?;

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
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
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
    /// after writing a partial multi-frame chunk. Shrinking prunes the
    /// stored chunks the way libhdf5's `H5Dset_extent` does: a chunk
    /// entirely beyond the new extent is removed from the chunk index and
    /// its storage freed for reuse, and a chunk the new extent cuts
    /// through has its out-of-extent region overwritten with the fill
    /// value — so growing the extent back exposes fill values, not the
    /// old data. The new extent must not exceed the dataset's maximum
    /// dimensions.
    pub fn set_extent(&self, new_dims: &[usize]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, .. } => {
                let dims_u64: Vec<u64> = new_dims.iter().map(|&d| d as u64).collect();
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
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
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
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
                let datatype = self.datatype()?;
                let starts_u64: Vec<u64> = starts.iter().map(|&s| s as u64).collect();
                let counts_u64: Vec<u64> = counts.iter().map(|&c| c as u64).collect();

                let mut raw = {
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
                to_host_byte_order(&mut raw, &datatype, T::element_size())?;

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

    /// Write a typed slice to a sub-region of the dataset.
    ///
    /// `starts` and `counts` define the N-dimensional selection, which must lie
    /// inside the dataset's current extent.
    ///
    /// Works for both contiguous and chunked datasets. For a chunked dataset
    /// only the chunks the selection touches are rewritten — a partially
    /// covered chunk is read back, patched, and written again, so updating one
    /// row of an appendable dataset costs the chunks that row crosses rather
    /// than the whole dataset. Elements of a touched chunk that the selection
    /// does not cover keep their stored value, or the dataset's fill value if
    /// the chunk did not exist yet.
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
                ..
            } => {
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
                let host =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };

                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        let datatype = writer.dataset_datatype(*index);
                        let stored = to_stored_byte_order(host, &datatype, T::element_size())?;
                        writer.write_slice(*index, &starts_u64, &counts_u64, &stored)?;
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

    /// Replace elements `start .. start + strings.len()` of a 1-D
    /// variable-length string dataset.
    ///
    /// The extent and every element outside the range are left alone, and the
    /// cost is the new strings plus the chunks holding their references — not
    /// the column. The dataset's character set is enforced: a non-ASCII
    /// replacement in a dataset that declares ASCII is rejected rather than
    /// stored under a datatype that misdescribes it.
    ///
    /// The global heap objects the replaced references pointed at are freed —
    /// the same reclaim libhdf5 performs on an overwrite — so updating one
    /// element repeatedly reuses space rather than growing the file. A
    /// collection emptied by the update returns its block to the allocator.
    /// Under SWMR nothing is freed, because a reader may still be following
    /// those references.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open_rw("meta.h5").unwrap();
    /// let ds = file.dataset_writer("notes").unwrap();
    /// ds.write_vlen_strings_slice(42, &["replacement"]).unwrap();
    /// file.close().unwrap();
    /// ```
    pub fn write_vlen_strings_slice(&self, start: usize, strings: &[&str]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, .. } => {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_vlen_strings_slice(*index, start as u64, strings)?;
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

    /// Read variable-length byte arrays from a dataset.
    ///
    /// This handles vlen byte-array datasets (a vlen sequence of `u8`, e.g.
    /// those written by [`write_vlen_bytes`](crate::H5File::write_vlen_bytes))
    /// that store each element as a global heap reference. Returns one
    /// `Vec<u8>` per element.
    pub fn read_vlen_bytes(&self) -> Result<Vec<Vec<u8>>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader.read_vlen_bytes(name)?),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read vlen bytes from a dataset in write mode".into(),
            )),
        }
    }

    /// Write object references naming `paths` into elements `0..paths.len()`
    /// — h5py's `refs[i] = f['/target'].ref`.
    ///
    /// The dataset must have been created with
    /// [`object_references`](DatasetBuilder::object_references). A path names
    /// a dataset or a group (`/` is the root group) and must already exist;
    /// what reaches the file is the target's object header address, which is
    /// assigned when the file is finalized. Elements left unwritten read back
    /// as null references.
    pub fn write_object_references(&self, paths: &[&str]) -> Result<()> {
        match &self.info {
            DatasetInfo::Writer { index, .. } => {
                let inner = borrow_inner(&self.file_inner);
                match &*inner {
                    H5FileInner::Writer(writer) => {
                        writer.write_object_references(*index, 0, paths)?;
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

    /// Read a reference dataset's elements, each resolved to the object it
    /// names.
    ///
    /// Every reference kind is read: the pre-1.12 pair h5py writes —
    /// `Reference` (an object header address) and `RegionReference` (a heap id
    /// whose heap object holds the target plus a serialized selection) — and
    /// the 1.12 `H5T_STD_REF` trio, `H5R_OBJECT2`, `H5R_DATASET_REGION2` and
    /// `H5R_ATTR`. An object reference comes back as [`Reference::Object`]
    /// carrying the target's path, a region reference as
    /// [`Reference::Region`], whose [`bounds`](Reference::bounds) is the
    /// selection's bounding box — libhdf5's `H5Sget_select_bounds` — and an
    /// attribute reference as [`Reference::Attr`], which adds the attribute's
    /// name.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open("refs.h5").unwrap();
    /// for r in file.dataset("refs").unwrap().read_references().unwrap() {
    ///     println!("{:?} {:?}", r.path(), r.bounds());
    /// }
    /// ```
    pub fn read_references(&self) -> Result<Vec<Reference>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader.read_references(name)?),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read references from a dataset in write mode".into(),
            )),
        }
    }

    /// Read a string dataset, fixed-width or variable-length, as one `String`
    /// per element.
    ///
    /// The width of a `FixedString` dataset is whatever the file says, so a
    /// 24-byte label column and a 100-byte one are read by the same call. The
    /// padding rule the datatype declares decides where each element ends —
    /// null-terminated (0), null-padded (1) or space-padded (2) — and its
    /// character set decides how the remaining bytes are decoded: ASCII (0)
    /// requires 7-bit bytes, UTF-8 (1) requires valid UTF-8. An element that
    /// violates either is an error naming the element, not a silent
    /// substitution; [`read_strings_lossy`](Self::read_strings_lossy) is the
    /// call that accepts such a file, replacing what it cannot decode.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open("labels.h5").unwrap();
    /// let labels = file.dataset("names").unwrap().read_strings().unwrap();
    /// ```
    pub fn read_strings(&self) -> Result<Vec<String>> {
        self.read_strings_inner(false)
    }

    /// [`read_strings`](Self::read_strings), but bytes that do not decode
    /// under the dataset's character set become U+FFFD instead of an error.
    ///
    /// Producers do mislabel the character set — a file that declares ASCII
    /// while storing Latin-1 or UTF-8 bytes reads here and not there.
    pub fn read_strings_lossy(&self) -> Result<Vec<String>> {
        self.read_strings_inner(true)
    }

    /// The single owner of string decoding for both string datatypes: the
    /// element bytes are found differently, the padding and character-set
    /// rules that turn them into a `String` are the same.
    fn read_strings_inner(&self, lossy: bool) -> Result<Vec<String>> {
        if matches!(self.info, DatasetInfo::Writer { .. }) {
            return Err(Hdf5Error::InvalidState(
                "cannot read strings from a dataset in write mode".into(),
            ));
        }
        match self.datatype()? {
            DatatypeMessage::VarLenString { charset, .. } => self
                .read_vlen_bytes()?
                .iter()
                .enumerate()
                .map(|(i, bytes)| decode_string(bytes, charset, lossy, i))
                .collect(),
            DatatypeMessage::FixedString {
                size,
                padding,
                charset,
            } => {
                let width = size as usize;
                if width == 0 {
                    // A corrupt file can declare it; `chunks_exact(0)` panics.
                    return Err(Hdf5Error::InvalidState(
                        "fixed-string datatype has zero width".into(),
                    ));
                }
                // `read_raw_bytes` returns `product(dims) * width` bytes, so
                // `chunks_exact` leaves no remainder.
                let raw = self.read_raw_bytes()?;
                raw.chunks_exact(width)
                    .enumerate()
                    .map(|(i, elem)| {
                        decode_string(trim_fixed_string(elem, padding, i)?, charset, lossy, i)
                    })
                    .collect()
            }
            other => Err(Hdf5Error::InvalidState(format!(
                "read_strings is only for string datasets, this one is {other:?}"
            ))),
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

                let datatype = self.datatype()?;
                let mut raw = {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => reader.read_dataset_raw(name)?,
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()));
                        }
                    }
                };
                to_host_byte_order(&mut raw, &datatype, T::element_size())?;

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

    /// Read the raw byte image of a dataset without an `H5Type` carrier.
    ///
    /// The counterpart to [`write_raw_bytes`](Self::write_raw_bytes): returns
    /// the element bytes verbatim regardless of the on-disk element type, so a
    /// runtime [`CompoundType`](crate::types::CompoundType) whose records have
    /// no matching Rust primitive can be read back and decoded by the caller.
    pub fn read_raw_bytes(&self) -> Result<Vec<u8>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let mut inner = borrow_inner_mut(&self.file_inner);
                match &mut *inner {
                    H5FileInner::Reader(reader) => Ok(reader.read_dataset_raw(name)?),
                    _ => Err(Hdf5Error::InvalidState("file is not in read mode".into())),
                }
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read from a dataset in write mode".into(),
            )),
        }
    }

    /// Read a numeric dataset as `T`, converting each element from the
    /// on-disk datatype.
    ///
    /// Unlike [`read_raw`](Self::read_raw), which requires `T`'s size to match
    /// the stored element size exactly, this inspects the dataset's datatype
    /// message — class, signedness, byte order, width — and converts per
    /// element:
    ///
    /// - integer → integer: checked; a stored value that does not fit in `T`
    ///   is an error naming the element index and value, never a silent wrap.
    /// - `f32` source → `f64`: exact widening.
    /// - `f64` source → `f32`, float → integer, and integer → float are
    ///   rejected as [`TypeMismatch`](Hdf5Error::TypeMismatch).
    ///
    /// Big-endian sources are decoded according to the datatype's byte order,
    /// which [`read_raw`](Self::read_raw)'s size-only check would misread.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open("data.h5").unwrap();
    /// let ds = file.dataset("counts").unwrap(); // stored as e.g. i16
    /// let counts = ds.read_numeric_as::<i64>().unwrap();
    /// ```
    pub fn read_numeric_as<T: ReadNumeric>(&self) -> Result<Vec<T>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let (kind, raw) = {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => {
                            let info = reader
                                .dataset_info(name)
                                .ok_or_else(|| Hdf5Error::NotFound(name.clone()))?;
                            let kind = numeric::classify(&info.datatype)?;
                            (kind, reader.read_dataset_raw(name)?)
                        }
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()))
                        }
                    }
                };
                numeric::convert(kind, &raw)
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read from a dataset in write mode".into(),
            )),
        }
    }

    /// Read a slice (hyperslab) of a numeric dataset as `T`, with the same
    /// per-element datatype conversion as
    /// [`read_numeric_as`](Self::read_numeric_as).
    ///
    /// `starts` and `counts` define the N-dimensional selection exactly as in
    /// [`read_slice`](Self::read_slice).
    pub fn read_numeric_slice_as<T: ReadNumeric>(
        &self,
        starts: &[usize],
        counts: &[usize],
    ) -> Result<Vec<T>> {
        match &self.info {
            DatasetInfo::Reader { name, .. } => {
                let starts_u64: Vec<u64> = starts.iter().map(|&s| s as u64).collect();
                let counts_u64: Vec<u64> = counts.iter().map(|&c| c as u64).collect();
                let (kind, raw) = {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => {
                            let info = reader
                                .dataset_info(name)
                                .ok_or_else(|| Hdf5Error::NotFound(name.clone()))?;
                            let kind = numeric::classify(&info.datatype)?;
                            (kind, reader.read_slice(name, &starts_u64, &counts_u64)?)
                        }
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()))
                        }
                    }
                };
                numeric::convert(kind, &raw)
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read_slice from a dataset in write mode".into(),
            )),
        }
    }

    /// Read the whole dataset into a caller-provided buffer, with no allocation.
    ///
    /// `out` must have exactly `product(dims)` elements (the dataset's element
    /// count) and `T::element_size()` must match the dataset's on-disk element
    /// size, otherwise an error is returned and `out` is left unspecified. The
    /// zero-copy counterpart of [`read_raw`](Self::read_raw): the bytes are read
    /// straight into `out` rather than into a fresh `Vec`, so a pinned /
    /// page-locked host buffer can be filled in one pass and DMA'd to a GPU
    /// without the extra staging copy a `read_raw` + copy-into-pinned would
    /// incur. Works for every layout (contiguous, compact, and chunked under
    /// any index); for chunked data each decoded chunk is scattered directly
    /// into `out`.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open("data.h5").unwrap();
    /// let ds = file.dataset("frames").unwrap();
    /// let n: usize = ds.shape().iter().product();
    /// let mut buf = vec![0u16; n];           // or a pinned host allocation
    /// ds.read_raw_into(&mut buf).unwrap();
    /// ```
    pub fn read_raw_into<T: H5Type>(&self, out: &mut [T]) -> Result<()> {
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
                let datatype = self.datatype()?;
                // Safety: `T: H5Type` is a `Copy` POD numeric with a defined
                // byte representation; every bit pattern the read writes is a
                // valid `T`. The byte view borrows `out` exclusively for this
                // call, and `out.len() * element_size` cannot overflow because
                // it is the byte length of an existing slice (<= isize::MAX).
                let byte_len = out.len() * T::element_size();
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
                };
                {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => reader.read_dataset_raw_into(name, bytes)?,
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()))
                        }
                    }
                }
                to_host_byte_order(bytes, &datatype, T::element_size())
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read from a dataset in write mode".into(),
            )),
        }
    }

    /// Read a hyperslab into a caller-provided buffer, with no allocation.
    ///
    /// `out` must have exactly `product(counts)` elements and
    /// `T::element_size()` must match the dataset's element size. The zero-copy
    /// counterpart of [`read_slice`](Self::read_slice) and the slice analogue of
    /// [`read_raw_into`](Self::read_raw_into): only chunks overlapping the
    /// selection are read, and the selected bytes land directly in `out` — the
    /// entry point for reading one frame / block straight into a pinned host
    /// buffer for an H2D transfer.
    ///
    /// ```no_run
    /// # use rust_hdf5::H5File;
    /// let file = H5File::open("vol.h5").unwrap();
    /// let ds = file.dataset("vol").unwrap();   // shape [nz, ny, nx]
    /// let (ny, nx) = (ds.shape()[1], ds.shape()[2]);
    /// let mut frame = vec![0f32; ny * nx];     // or a pinned host allocation
    /// ds.read_slice_into(&mut frame, &[5, 0, 0], &[1, ny, nx]).unwrap();
    /// ```
    pub fn read_slice_into<T: H5Type>(
        &self,
        out: &mut [T],
        starts: &[usize],
        counts: &[usize],
    ) -> Result<()> {
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
                let datatype = self.datatype()?;
                let starts_u64: Vec<u64> = starts.iter().map(|&s| s as u64).collect();
                let counts_u64: Vec<u64> = counts.iter().map(|&c| c as u64).collect();
                // Safety: see `read_raw_into` — `T: H5Type` POD, exclusive
                // borrow of `out`, byte length within bounds.
                let byte_len = out.len() * T::element_size();
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len)
                };
                {
                    let mut inner = borrow_inner_mut(&self.file_inner);
                    match &mut *inner {
                        H5FileInner::Reader(reader) => {
                            reader.read_slice_into(name, &starts_u64, &counts_u64, bytes)?
                        }
                        _ => {
                            return Err(Hdf5Error::InvalidState("file is not in read mode".into()))
                        }
                    }
                }
                to_host_byte_order(bytes, &datatype, T::element_size())
            }
            DatasetInfo::Writer { .. } => Err(Hdf5Error::InvalidState(
                "cannot read from a dataset in write mode".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Datatype-aware numeric conversion (read_numeric_as)
// ---------------------------------------------------------------------------

/// Marker trait for the Rust types [`H5Dataset::read_numeric_as`] can convert
/// into: the integer primitives (checked, never wrapping) plus `f32`/`f64`
/// (widening only).
///
/// Sealed — the conversion policy is part of the library contract, so the
/// trait cannot be implemented outside this crate.
pub trait ReadNumeric: numeric::Sealed {}
impl<T: numeric::Sealed> ReadNumeric for T {}

pub(crate) mod numeric {
    //! Per-element decode + checked conversion for `read_numeric_as` (and the
    //! attribute counterpart `H5Attribute::read_numeric_as`).

    use crate::error::{Hdf5Error, Result};
    use crate::format::messages::datatype::{ByteOrder, DatatypeMessage, IeeeFormat};

    /// A source element, normalized: every standard integer width — u64::MAX
    /// included — fits in `i128` without loss.
    pub enum NumericSource {
        Int(i128),
        F32(f32),
        F64(f64),
    }

    /// The on-disk element shape `classify` accepted.
    #[derive(Clone, Copy)]
    pub enum SourceKind {
        Int {
            size: usize,
            signed: bool,
            byte_order: ByteOrder,
        },
        F16(ByteOrder),
        F32(ByteOrder),
        F64(ByteOrder),
    }

    impl SourceKind {
        fn element_size(self) -> usize {
            match self {
                SourceKind::Int { size, .. } => size,
                SourceKind::F16(_) => 2,
                SourceKind::F32(_) => 4,
                SourceKind::F64(_) => 8,
            }
        }
    }

    /// Widen an IEEE 754 binary16 bit pattern to `f32`, which represents every
    /// half — including subnormals, infinities and NaN payloads — exactly.
    ///
    /// Rust has no stable `f16` to convert through.
    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = u32::from(bits >> 15);
        let exponent = u32::from((bits >> 10) & 0x1f);
        let mantissa = u32::from(bits & 0x03ff);
        if exponent == 0 {
            // Zero and subnormals: the value is mantissa * 2^-24, which is a
            // normal f32 for every mantissa, so the multiply is exact. Going
            // through the sign separately keeps -0.0.
            let magnitude = mantissa as f32 * (1.0 / 16_777_216.0);
            return if sign == 1 { -magnitude } else { magnitude };
        }
        let out = if exponent == 0x1f {
            // Infinity and NaN; shifting the mantissa maps the quiet bit onto
            // f32's quiet bit and preserves the rest of the payload.
            (sign << 31) | 0x7f80_0000 | (mantissa << 13)
        } else {
            // Normal: rebias the exponent (127 - 15) and left-align the
            // mantissa.
            (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
        };
        f32::from_bits(out)
    }

    /// Map a datatype message to a supported numeric source shape.
    ///
    /// Accepts standard-width integers (1/2/4/8 bytes, full precision, zero
    /// bit offset) and IEEE binary32/binary64 floats; everything else is a
    /// `TypeMismatch` naming what was found.
    pub fn classify(dt: &DatatypeMessage) -> Result<SourceKind> {
        match *dt {
            DatatypeMessage::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                if !matches!(size, 1 | 2 | 4 | 8)
                    || bit_offset != 0
                    || u32::from(bit_precision) != size * 8
                {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "fixed-point datatype (size {size}, bit offset {bit_offset}, \
                         precision {bit_precision}) is not a standard-width integer",
                    )));
                }
                Ok(SourceKind::Int {
                    size: size as usize,
                    signed,
                    byte_order,
                })
            }
            DatatypeMessage::BitField {
                size,
                byte_order,
                bit_offset,
                bit_precision,
            } => {
                // A bit field has no signed form; a full-width one is the
                // unsigned integer of the stored width. A narrower one would
                // need a shift-and-mask conversion this path does not model.
                if !matches!(size, 1 | 2 | 4 | 8)
                    || bit_offset != 0
                    || u32::from(bit_precision) != size * 8
                {
                    return Err(Hdf5Error::TypeMismatch(format!(
                        "bit-field datatype (size {size}, bit offset {bit_offset}, \
                         precision {bit_precision}) is not a whole-width bit field",
                    )));
                }
                Ok(SourceKind::Int {
                    size: size as usize,
                    signed: false,
                    byte_order,
                })
            }
            DatatypeMessage::FloatingPoint {
                size,
                byte_order,
                exponent_size,
                mantissa_size,
                ..
            } => match dt.ieee_format() {
                Some(IeeeFormat::Binary16) => Ok(SourceKind::F16(byte_order)),
                Some(IeeeFormat::Binary32) => Ok(SourceKind::F32(byte_order)),
                Some(IeeeFormat::Binary64) => Ok(SourceKind::F64(byte_order)),
                None => Err(Hdf5Error::TypeMismatch(format!(
                    "floating-point datatype (size {size}, exponent {exponent_size} bits, \
                     mantissa {mantissa_size} bits) is not an IEEE 754 interchange format",
                ))),
            },
            ref other => Err(Hdf5Error::TypeMismatch(format!(
                "dataset datatype '{other}' is not numeric",
            ))),
        }
    }

    fn decode_element(kind: SourceKind, bytes: &[u8]) -> NumericSource {
        match kind {
            SourceKind::Int {
                size,
                signed,
                byte_order,
            } => {
                let mut le = [0u8; 8];
                match byte_order {
                    ByteOrder::LittleEndian => le[..size].copy_from_slice(bytes),
                    ByteOrder::BigEndian => {
                        for (dst, src) in le[..size].iter_mut().zip(bytes.iter().rev()) {
                            *dst = *src;
                        }
                    }
                }
                let zero_extended = u64::from_le_bytes(le);
                let value = if signed {
                    // Arithmetic right shift sign-extends the low `size` bytes.
                    let shift = 64 - 8 * size as u32;
                    i128::from(((zero_extended as i64) << shift) >> shift)
                } else {
                    i128::from(zero_extended)
                };
                NumericSource::Int(value)
            }
            SourceKind::F16(byte_order) => {
                let arr: [u8; 2] = bytes.try_into().unwrap();
                let bits = match byte_order {
                    ByteOrder::LittleEndian => u16::from_le_bytes(arr),
                    ByteOrder::BigEndian => u16::from_be_bytes(arr),
                };
                NumericSource::F32(f16_bits_to_f32(bits))
            }
            SourceKind::F32(byte_order) => {
                let arr: [u8; 4] = bytes.try_into().unwrap();
                NumericSource::F32(match byte_order {
                    ByteOrder::LittleEndian => f32::from_le_bytes(arr),
                    ByteOrder::BigEndian => f32::from_be_bytes(arr),
                })
            }
            SourceKind::F64(byte_order) => {
                let arr: [u8; 8] = bytes.try_into().unwrap();
                NumericSource::F64(match byte_order {
                    ByteOrder::LittleEndian => f64::from_le_bytes(arr),
                    ByteOrder::BigEndian => f64::from_be_bytes(arr),
                })
            }
        }
    }

    /// Decode and convert every element of `raw` into `T`.
    pub fn convert<T: Sealed>(kind: SourceKind, raw: &[u8]) -> Result<Vec<T>> {
        let size = kind.element_size();
        if !raw.len().is_multiple_of(size) {
            return Err(Hdf5Error::TypeMismatch(format!(
                "raw data size {} is not a multiple of element size {size}",
                raw.len(),
            )));
        }
        raw.chunks_exact(size)
            .enumerate()
            .map(|(index, bytes)| T::from_source(decode_element(kind, bytes), index))
            .collect()
    }

    /// The sealed half of `ReadNumeric`: how one normalized source element
    /// becomes a `Self`, or a `TypeMismatch` explaining why it cannot.
    pub trait Sealed: Sized {
        fn from_source(src: NumericSource, index: usize) -> Result<Self>;
    }

    macro_rules! int_targets {
        ($($t:ty),* $(,)?) => {$(
            impl Sealed for $t {
                fn from_source(src: NumericSource, index: usize) -> Result<Self> {
                    match src {
                        NumericSource::Int(v) => <$t>::try_from(v).map_err(|_| {
                            Hdf5Error::TypeMismatch(format!(
                                concat!(
                                    "value {} at element {} does not fit in ",
                                    stringify!($t),
                                ),
                                v, index,
                            ))
                        }),
                        NumericSource::F32(_) | NumericSource::F64(_) => {
                            Err(Hdf5Error::TypeMismatch(
                                concat!(
                                    "cannot read a floating-point dataset as ",
                                    stringify!($t),
                                    "; read as f64 and convert explicitly",
                                )
                                .into(),
                            ))
                        }
                    }
                }
            }
        )*};
    }
    int_targets!(i8, i16, i32, i64, u8, u16, u32, u64, u128);

    // Not in the macro: `i128::try_from(i128)` is infallible, which trips
    // clippy::unnecessary_fallible_conversions.
    impl Sealed for i128 {
        fn from_source(src: NumericSource, _index: usize) -> Result<Self> {
            match src {
                NumericSource::Int(v) => Ok(v),
                NumericSource::F32(_) | NumericSource::F64(_) => Err(Hdf5Error::TypeMismatch(
                    "cannot read a floating-point dataset as i128; read as f64 and \
                     convert explicitly"
                        .into(),
                )),
            }
        }
    }

    impl Sealed for f32 {
        fn from_source(src: NumericSource, _index: usize) -> Result<Self> {
            match src {
                NumericSource::F32(v) => Ok(v),
                NumericSource::F64(_) => Err(Hdf5Error::TypeMismatch(
                    "narrowing an f64 dataset to f32 loses precision; read as f64".into(),
                )),
                NumericSource::Int(_) => Err(Hdf5Error::TypeMismatch(
                    "cannot read an integer dataset as f32; read as an integer type and \
                     convert explicitly"
                        .into(),
                )),
            }
        }
    }

    impl Sealed for f64 {
        fn from_source(src: NumericSource, _index: usize) -> Result<Self> {
            match src {
                NumericSource::F64(v) => Ok(v),
                // Every f32 is exactly representable as f64.
                NumericSource::F32(v) => Ok(f64::from(v)),
                NumericSource::Int(_) => Err(Hdf5Error::TypeMismatch(
                    "cannot read an integer dataset as f64; integers above 2^53 lose \
                     precision — read as an integer type and convert explicitly"
                        .into(),
                )),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every binary16 bit pattern class widens to the f32 with the same
        /// value: zeros keep their sign, subnormals stay exact, infinities and
        /// NaN payloads survive.
        #[test]
        fn f16_widening_is_exact() {
            let cases: [(u16, f32); 10] = [
                (0x0000, 0.0),
                (0x3c00, 1.0),
                (0xc000, -2.0),
                (0x3555, 0.333_251_95),   // nearest half to 1/3
                (0x0001, 5.960_464_5e-8), // smallest subnormal, 2^-24
                (0x03ff, 6.097_555e-5),   // largest subnormal
                (0x0400, 6.103_515_6e-5), // smallest normal
                (0x7bff, 65504.0),        // largest finite
                (0x7c00, f32::INFINITY),
                (0xfc00, f32::NEG_INFINITY),
            ];
            for (bits, expected) in cases {
                let got = f16_bits_to_f32(bits);
                assert_eq!(got, expected, "0x{bits:04x} widened to {got}");
            }

            let neg_zero = f16_bits_to_f32(0x8000);
            assert_eq!(neg_zero, 0.0);
            assert!(neg_zero.is_sign_negative(), "-0.0 lost its sign");

            let nan = f16_bits_to_f32(0x7e01);
            assert!(nan.is_nan());
            // The quiet bit and the payload land in f32's mantissa.
            assert_eq!(nan.to_bits(), 0x7fc0_2000);
        }

        #[test]
        fn f16_source_converts_and_honors_byte_order() {
            let kind = classify(&DatatypeMessage::f16_type()).unwrap();
            // 1.0, -2.0, 0.333..., 65504
            let raw = [0x00, 0x3c, 0x00, 0xc0, 0x55, 0x35, 0xff, 0x7b];
            assert_eq!(
                convert::<f32>(kind, &raw).unwrap(),
                vec![1.0, -2.0, 0.333_251_95, 65504.0]
            );
            assert_eq!(
                convert::<f64>(kind, &raw).unwrap(),
                vec![1.0, -2.0, 0.333_251_953_125, 65504.0]
            );

            let DatatypeMessage::FloatingPoint { .. } = DatatypeMessage::f16_type() else {
                unreachable!()
            };
            let mut be = DatatypeMessage::f16_type();
            if let DatatypeMessage::FloatingPoint { byte_order, .. } = &mut be {
                *byte_order = ByteOrder::BigEndian;
            }
            let be_kind = classify(&be).unwrap();
            assert_eq!(convert::<f32>(be_kind, &[0x3c, 0x00]).unwrap(), vec![1.0]);
        }

        /// A float whose layout is not an interchange format is refused, not
        /// reinterpreted.
        #[test]
        fn non_ieee_float_is_refused() {
            let mut odd = DatatypeMessage::f32_type();
            if let DatatypeMessage::FloatingPoint { exponent_bias, .. } = &mut odd {
                *exponent_bias = 63;
            }
            assert!(odd.ieee_format().is_none());
            let err = classify(&odd).err().expect("non-IEEE float was accepted");
            assert!(
                err.to_string().contains("IEEE 754 interchange format"),
                "unexpected error: {err}"
            );
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
    fn runtime_compound_via_datatype_override_and_raw_bytes() {
        use crate::format::messages::datatype::DatatypeMessage;
        use crate::types::{CompoundType, H5Type};

        let path = temp_path("compound_raw");
        // A 12-byte packed compound with NO matching Rust primitive carrier,
        // so it can only be written through the datatype() override +
        // write_raw_bytes path (the runtime-CompoundType use case).
        let ct = CompoundType {
            members: vec![
                ("id".to_string(), i32::hdf5_type(), 0),
                ("val".to_string(), f64::hdf5_type(), 4),
            ],
            total_size: 12,
        };
        let recs: [(i32, f64); 3] = [(1, 2.5), (2, 3.5), (3, -4.0)];
        let mut bytes = Vec::new();
        for (id, val) in recs {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<u8>()
                .datatype(ct.to_datatype())
                .shape([recs.len()])
                .create("records")
                .unwrap();
            ds.write_raw_bytes(&bytes).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("records").unwrap();
            // The on-disk element type is the compound we specified (size 12),
            // not the u8 carrier.
            match ds.datatype().unwrap() {
                DatatypeMessage::Compound { size, members } => {
                    assert_eq!(size, 12);
                    assert_eq!(members.len(), 2);
                    assert_eq!(members[0].name, "id");
                    assert_eq!(members[0].offset, 0);
                    assert_eq!(members[1].name, "val");
                    assert_eq!(members[1].offset, 4);
                }
                other => panic!("expected compound datatype, got {other:?}"),
            }
            assert_eq!(ds.read_raw_bytes().unwrap(), bytes);
        }
        std::fs::remove_file(&path).ok();
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

    // A: a filter set without explicit chunk dimensions must auto-chunk (whole
    // dataset = one chunk) rather than silently drop the filter on the
    // contiguous path. write_raw then populates that single chunk.
    #[cfg(feature = "deflate")]
    #[test]
    fn filter_without_chunk_autochunks_and_roundtrips() {
        let path = temp_path("autochunk_filter");
        let data: Vec<i32> = (0..8).collect();
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .deflate(6)
                .shape([8])
                .create("seq")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("seq").unwrap();
            // The filter forced chunked storage: a single whole-dataset chunk.
            assert!(
                ds.is_chunked(),
                "auto-chunk did not produce chunked storage"
            );
            assert_eq!(ds.chunk_dims(), Some(vec![8]));
            assert_eq!(ds.read_raw::<i32>().unwrap(), data);
        }
        std::fs::remove_file(&path).ok();
    }

    // B: write_raw on an explicitly chunked + compressed dataset scatters the
    // full row-major image across a multi-chunk grid, including edge chunks
    // (7/3 -> 3,3,1 along dim0; 5/2 -> 2,2,1 along dim1).
    #[cfg(feature = "deflate")]
    #[test]
    fn write_raw_multichunk_edge_roundtrips() {
        let path = temp_path("multichunk_edge");
        let data: Vec<i32> = (0..35).collect(); // 7 x 5 row-major
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([7, 5])
                .chunk(&[3, 2])
                .deflate(4)
                .create("grid")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("grid").unwrap();
            assert_eq!(ds.shape(), vec![7, 5]);
            assert_eq!(ds.chunk_dims(), Some(vec![3, 2]));
            assert_eq!(ds.read_raw::<i32>().unwrap(), data);
        }
        std::fs::remove_file(&path).ok();
    }

    // B: write_raw on an unfiltered chunked dataset (previously rejected with
    // "use write_chunk for chunked datasets") now gathers and round-trips.
    #[test]
    fn write_raw_unfiltered_chunked_roundtrips() {
        let path = temp_path("chunked_unfiltered");
        let data: Vec<f64> = (0..12).map(|i| i as f64 * 1.5).collect(); // 4 x 3
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([4, 3])
                .chunk(&[2, 2])
                .create("m")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("m").unwrap();
            assert_eq!(ds.chunk_dims(), Some(vec![2, 2]));
            assert_eq!(ds.read_raw::<f64>().unwrap(), data);
        }
        std::fs::remove_file(&path).ok();
    }

    // write_raw on an extensible-array (unlimited first dim) compressed dataset
    // drives write_full_image_chunked's EA branch, which gathers chunks and
    // compresses them through the windowed batch path. Round-trips the full
    // image, including a partial edge chunk along the unlimited dimension.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_raw_ea_compressed_roundtrips() {
        let path = temp_path("write_raw_ea_deflate");
        let data: Vec<i32> = (0..20).collect(); // 5 x 4 row-major
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([5, 4])
                .chunk(&[2, 4])
                .max_shape(&[None, Some(4)]) // unlimited dim 0 -> extensible array
                .deflate(5)
                .create("stream")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("stream").unwrap();
            assert_eq!(ds.shape(), vec![5, 4]);
            assert_eq!(ds.read_raw::<i32>().unwrap(), data);
        }
        std::fs::remove_file(&path).ok();
    }

    // 3D chunked Full read with partial-edge chunks in every dimension. This
    // drives copy_chunk_to_output's multi-dim run-memcpy path with two outer
    // dimensions, exercising the nested outer-coordinate carry and the
    // last-axis edge clamp (chunks hang off the high edge in all three axes).
    #[cfg(feature = "deflate")]
    #[test]
    fn read_full_3d_chunked_edge_roundtrips() {
        let path = temp_path("full_3d_chunked_edge");
        // shape 5x4x3, chunk 2x3x2 -> ceil gives 3x2x2 chunks; the last chunk
        // along each axis is partial (1, 1, and 1 element respectively).
        let total: usize = 5 * 4 * 3;
        let data: Vec<i32> = (0..total as i32).collect();
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([5, 4, 3])
                .chunk(&[2, 3, 2])
                .deflate(4)
                .create("vol")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("vol").unwrap();
            assert_eq!(ds.shape(), vec![5, 4, 3]);
            assert_eq!(ds.chunk_dims(), Some(vec![2, 3, 2]));
            assert_eq!(ds.read_raw::<i32>().unwrap(), data);
        }
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
    fn array_attr_roundtrip() {
        let path = temp_path("array_attr");
        let offsets = [10i32, -20, 30];
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f32>().shape([4]).create("data").unwrap();
            ds.write_raw(&[1.0f32; 4]).unwrap();

            // 1-D int32 array attribute (NDArrayDimOffset-style).
            let a = ds
                .new_attr::<i32>()
                .shape([3])
                .create("dim_offset")
                .unwrap();
            a.write_array(&offsets).unwrap();

            // Wrong element count is rejected.
            let bad = ds.new_attr::<i32>().shape([3]).create("bad").unwrap();
            assert!(bad.write_array(&[1i32, 2]).is_err());

            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            let a = ds.attr("dim_offset").unwrap();
            let raw = a.read_raw().unwrap();
            assert_eq!(raw.len(), 3 * 4);
            let got: Vec<i32> = raw
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(got, offsets);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn attr_datatype_exposes_class_and_sign() {
        // H5Attribute::datatype() must report the stored datatype class and
        // signedness so a generic attr->metadata mapper need not infer it from
        // the byte width (the HDF5-L1 adapter blocker this accessor unblocks).
        use crate::format::messages::datatype::DatatypeMessage;

        let path = temp_path("attr_datatype");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f32>().shape([4]).create("data").unwrap();
            ds.new_attr::<f64>()
                .shape(())
                .create("scale")
                .unwrap()
                .write_numeric(&1.5f64)
                .unwrap();
            ds.new_attr::<i32>()
                .shape(())
                .create("count")
                .unwrap()
                .write_numeric(&7i32)
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();

            match ds.attr("scale").unwrap().datatype().unwrap() {
                DatatypeMessage::FloatingPoint { size, .. } => assert_eq!(size, 8),
                other => panic!("expected FloatingPoint for f64 attr, got {other:?}"),
            }

            match ds.attr("count").unwrap().datatype().unwrap() {
                DatatypeMessage::FixedPoint { size, signed, .. } => {
                    assert_eq!(size, 4);
                    assert!(signed, "i32 attr must be signed");
                }
                other => panic!("expected FixedPoint for i32 attr, got {other:?}"),
            }
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn attr_datatype_in_write_mode_errors() {
        let path = temp_path("attr_datatype_write_mode");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<f32>().shape([4]).create("data").unwrap();
        let attr = ds.new_attr::<f64>().shape(()).create("scale").unwrap();
        assert!(attr.datatype().is_err());
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

    // H2D zero-alloc reads. `read_raw_into` / `read_slice_into` fill a
    // caller-provided buffer and MUST produce byte-for-byte the same data as
    // their Vec-returning counterparts (`read_raw` / `read_slice`) on every
    // creatable layout, since both now share one buffer-filling core.
    fn assert_into_matches<T>(ds: &super::H5Dataset, starts: &[usize], counts: &[usize])
    where
        T: crate::types::H5Type + Copy + std::fmt::Debug + PartialEq + Default,
    {
        let n: usize = ds.shape().iter().product();
        let want_full = ds.read_raw::<T>().unwrap();
        let mut got_full = vec![T::default(); n];
        ds.read_raw_into::<T>(&mut got_full).unwrap();
        assert_eq!(got_full, want_full, "read_raw_into != read_raw");

        let want_slice = ds.read_slice::<T>(starts, counts).unwrap();
        let sn: usize = counts.iter().product();
        let mut got_slice = vec![T::default(); sn];
        ds.read_slice_into::<T>(&mut got_slice, starts, counts)
            .unwrap();
        assert_eq!(got_slice, want_slice, "read_slice_into != read_slice");
    }

    #[test]
    fn read_into_matches_vec_contiguous() {
        let path = temp_path("into_contig");
        let data: Vec<i32> = (0..20).collect(); // 4 x 5 contiguous
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
            assert_eq!(ds.chunk_dims(), None);
            assert_into_matches::<i32>(&ds, &[1, 2], &[2, 2]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_into_matches_vec_chunked_unfiltered() {
        let path = temp_path("into_chunk");
        let data: Vec<f64> = (0..35).map(|i| i as f64 * 1.5).collect(); // 7 x 5
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape([7, 5])
                .chunk(&[3, 2]) // multi-chunk grid with edge chunks
                .create("grid")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("grid").unwrap();
            assert_eq!(ds.chunk_dims(), Some(vec![3, 2]));
            // Slice spans multiple chunks (rows 2..5, cols 1..4).
            assert_into_matches::<f64>(&ds, &[2, 1], &[3, 3]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_into_matches_vec_single_chunk() {
        let path = temp_path("into_single_chunk");
        let data: Vec<i32> = (0..12).collect(); // 3 x 4, one chunk covers all
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([3, 4])
                .chunk(&[3, 4]) // chunk == shape -> SingleChunk index
                .create("g")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("g").unwrap();
            assert_eq!(ds.chunk_dims(), Some(vec![3, 4]));
            assert_into_matches::<i32>(&ds, &[1, 1], &[2, 2]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn read_into_matches_vec_chunked_deflate() {
        let path = temp_path("into_chunk_deflate");
        let data: Vec<i32> = (0..35).collect(); // 7 x 5
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([7, 5])
                .chunk(&[3, 2])
                .deflate(4)
                .create("grid")
                .unwrap();
            ds.write_raw(&data).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("grid").unwrap();
            assert_eq!(ds.chunk_dims(), Some(vec![3, 2]));
            assert_into_matches::<i32>(&ds, &[2, 1], &[3, 3]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_into_wrong_buffer_size_rejected() {
        let path = temp_path("into_badlen");
        let data: Vec<i32> = (0..20).collect(); // 4 x 5
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

            // Too small / too large full-read buffers are both rejected.
            let mut small = vec![0i32; 19];
            assert!(ds.read_raw_into::<i32>(&mut small).is_err());
            let mut large = vec![0i32; 21];
            assert!(ds.read_raw_into::<i32>(&mut large).is_err());

            // Slice buffer must be exactly product(counts) = 4.
            let mut bad_slice = vec![0i32; 3];
            assert!(ds
                .read_slice_into::<i32>(&mut bad_slice, &[1, 2], &[2, 2])
                .is_err());
            // The correctly sized slice buffer succeeds.
            let mut ok_slice = vec![0i32; 4];
            assert!(ds
                .read_slice_into::<i32>(&mut ok_slice, &[1, 2], &[2, 2])
                .is_ok());
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_into_wrong_element_size_rejected() {
        let path = temp_path("into_badtype");
        let data: Vec<i32> = (0..20).collect(); // element size 4
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
            // u8 (size 1) and i64 (size 8) mismatch the dataset's 4-byte
            // element size -> TypeMismatch, even with a "correctly sized" Vec.
            let mut as_u8 = vec![0u8; 20];
            assert!(matches!(
                ds.read_raw_into::<u8>(&mut as_u8),
                Err(crate::Hdf5Error::TypeMismatch(_))
            ));
            let mut as_i64 = vec![0i64; 20];
            assert!(matches!(
                ds.read_slice_into::<i64>(&mut as_i64, &[0, 0], &[4, 5]),
                Err(crate::Hdf5Error::TypeMismatch(_))
            ));
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

    /// One 2x4 i32 chunk whose every element is `v`.
    fn chunk_of(v: i32) -> Vec<u8> {
        (0..8).flat_map(|_| v.to_le_bytes()).collect()
    }

    /// Write chunk (0,0) `rewrites` times — each time with a different value,
    /// so no write can be skipped — and return the closed file's size along
    /// with what the chunk reads back as.
    fn rewrite_chunk(
        tag: &str,
        rewrites: i32,
        build: impl Fn(&H5File) -> crate::H5Dataset,
    ) -> (u64, i32) {
        let path = temp_path(tag);
        {
            let file = H5File::create(&path).unwrap();
            let ds = build(&file);
            for v in 1..=rewrites {
                ds.write_chunk_at(&[0, 0], &chunk_of(v)).unwrap();
            }
            file.close().unwrap();
        }
        let size = std::fs::metadata(&path).unwrap().len();
        let first = {
            let file = H5File::open(&path).unwrap();
            file.dataset("d").unwrap().read_raw::<i32>().unwrap()[0]
        };
        std::fs::remove_file(&path).ok();
        (size, first)
    }

    // An unfiltered chunk's stored size is fixed by the chunk shape, so
    // rewriting it must overwrite the block it already occupies rather than
    // abandoning it and appending a new one (libhdf5 H5D__chunk_flush_entry
    // leaves must_alloc false for exactly this case). The file must therefore
    // be byte-identical in size no matter how many times the chunk is written.
    #[test]
    fn rewriting_an_unfiltered_extensible_array_chunk_stays_in_place() {
        let build = |f: &H5File| {
            f.new_dataset::<i32>()
                .shape([2, 4])
                .chunk(&[2, 4])
                .max_shape(&[None, Some(4)])
                .create("d")
                .unwrap()
        };
        let (once, _) = rewrite_chunk("rewrite_ea_1", 1, build);
        let (many, last) = rewrite_chunk("rewrite_ea_8", 8, build);
        assert_eq!(many, once, "8 rewrites grew the file past a single write");
        assert_eq!(last, 8, "the last write must be the one that survives");
    }

    #[test]
    fn rewriting_an_unfiltered_fixed_array_chunk_stays_in_place() {
        let build = |f: &H5File| {
            f.new_dataset::<i32>()
                .shape([2, 4])
                .chunk(&[2, 4])
                .create("d")
                .unwrap()
        };
        let (once, _) = rewrite_chunk("rewrite_fa_1", 1, build);
        let (many, last) = rewrite_chunk("rewrite_fa_8", 8, build);
        assert_eq!(many, once, "8 rewrites grew the file past a single write");
        assert_eq!(last, 8);
    }

    #[test]
    fn rewriting_an_unfiltered_btree_v2_chunk_stays_in_place() {
        let build = |f: &H5File| {
            f.new_dataset::<i32>()
                .shape([2, 4])
                .chunk(&[2, 4])
                .max_shape(&[None, None])
                .create("d")
                .unwrap()
        };
        let (once, _) = rewrite_chunk("rewrite_bt2_1", 1, build);
        let (many, last) = rewrite_chunk("rewrite_bt2_8", 8, build);
        assert_eq!(many, once, "8 rewrites grew the file past a single write");
        assert_eq!(last, 8);
    }

    // A flush re-serializes the whole v2 B-tree over the dataset's node-block
    // pool. Every node is the same size, so the blocks already on disk are
    // reused and repeated flushes cost nothing; sizing the root to its record
    // count instead would relocate it each time and orphan the block it left.
    #[test]
    fn repeated_flushes_do_not_grow_a_btree_v2_index() {
        let flush_n = |label: &str, flushes: usize| -> u64 {
            let path = temp_path(label);
            {
                let file = H5File::create(&path).unwrap();
                let ds = file
                    .new_dataset::<i32>()
                    .shape([2, 4])
                    .chunk(&[2, 4])
                    .max_shape(&[None, None])
                    .create("d")
                    .unwrap();
                let bytes: Vec<u8> = (0..8i32).flat_map(|v| v.to_le_bytes()).collect();
                ds.write_chunk_at(&[0, 0], &bytes).unwrap();
                for _ in 0..flushes {
                    ds.flush().unwrap();
                }
                file.close().unwrap();
            }
            let size = std::fs::metadata(&path).unwrap().len();
            // The data must survive every rewrite of the index.
            {
                let file = H5File::open(&path).unwrap();
                let ds = file.dataset("d").unwrap();
                assert_eq!(ds.read_raw::<i32>().unwrap(), (0..8).collect::<Vec<i32>>());
            }
            std::fs::remove_file(&path).ok();
            size
        };
        assert_eq!(
            flush_n("bt2_flush_8", 8),
            flush_n("bt2_flush_1", 1),
            "8 index flushes grew the file past a single one"
        );
    }

    // A filtered chunk whose compressed size changes cannot stay put, so it
    // moves and releases its old block (libhdf5 H5D__chunk_file_alloc calls
    // H5MF_xfree). Alternating between two payloads of different compressed
    // size must therefore keep reusing the same two blocks instead of
    // appending a fresh one each time.
    #[cfg(feature = "deflate")]
    #[test]
    fn rewriting_a_filtered_chunk_recycles_the_released_block() {
        // All-equal elements deflate to far fewer bytes than a varied payload,
        // so the two writes below land at different stored sizes.
        let flat: Vec<u8> = (0..8).flat_map(|_| 7i32.to_le_bytes()).collect();
        let varied: Vec<u8> = (0..8i32)
            .flat_map(|i| i.wrapping_mul(0x5bd1_e995).to_le_bytes())
            .collect();

        let sizes: Vec<u64> = [1usize, 8]
            .iter()
            .map(|&rounds| {
                let path = temp_path(&format!("rewrite_filtered_{rounds}"));
                {
                    let file = H5File::create(&path).unwrap();
                    let ds = file
                        .new_dataset::<i32>()
                        .shape([2, 4])
                        .chunk(&[2, 4])
                        .max_shape(&[None, Some(4)])
                        .deflate(6)
                        .create("d")
                        .unwrap();
                    for _ in 0..rounds {
                        ds.write_chunk_at(&[0, 0], &flat).unwrap();
                        ds.write_chunk_at(&[0, 0], &varied).unwrap();
                    }
                    file.close().unwrap();
                }
                let size = std::fs::metadata(&path).unwrap().len();
                {
                    let file = H5File::open(&path).unwrap();
                    let got = file.dataset("d").unwrap().read_raw::<i32>().unwrap();
                    let want: Vec<i32> = (0..8i32).map(|i| i.wrapping_mul(0x5bd1_e995)).collect();
                    assert_eq!(got, want, "the last write must survive the round trip");
                }
                std::fs::remove_file(&path).ok();
                size
            })
            .collect();

        assert_eq!(
            sizes[1], sizes[0],
            "8 alternating rewrites grew the file past a single pair"
        );
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
            use crate::format::messages::datatype::DatatypeMessage;
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("data").unwrap();
            assert_eq!(ds.shape(), vec![4]);
            let readback = ds.read_raw::<u8>().unwrap();
            assert_eq!(readback, vec![1u8, 2, 3, 4]);

            // The string attribute is stored as a true variable-length string
            // (not fixed-length) and round-trips its value.
            let attr = ds.attr("name").unwrap();
            assert!(
                matches!(
                    attr.datatype().unwrap(),
                    DatatypeMessage::VarLenString { .. }
                ),
                "string attribute should have a variable-length string datatype"
            );
            assert_eq!(attr.read_string().unwrap(), "test_value");
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

    /// An append that leaves its chunk partial is buffered until close, and
    /// the flush has to keep the frames that chunk already holds. It built a
    /// fresh fill-value chunk around the buffered frame instead, so reopening
    /// a file and appending one row erased every earlier row of that chunk
    /// (issue #3). Four sessions: the second lands beside an existing row, the
    /// third closes chunk 0 and opens chunk 1, the fourth lands beside the row
    /// the third left in chunk 1.
    #[test]
    fn append_after_reopen_keeps_the_partial_chunk_it_lands_in() {
        let path = temp_path("append_reopen_partial");

        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, 3])
                .chunk(&[4, 3])
                .max_shape(&[None, Some(3)])
                .create("values")
                .unwrap();
            ds.append(&[1, 2, 3]).unwrap();
            file.close().unwrap();
        }
        for rows in [
            vec![4, 5, 6],
            vec![7, 8, 9, 10, 11, 12, 13, 14, 15],
            vec![16, 17, 18],
        ] {
            let file = H5File::open_rw(&path).unwrap();
            file.dataset_writer("values")
                .unwrap()
                .append(&rows)
                .unwrap();
            file.close().unwrap();
        }

        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("values").unwrap();
        assert_eq!(ds.shape(), vec![6, 3]);
        assert_eq!(
            ds.read_raw::<i32>().unwrap(),
            (1..=18).collect::<Vec<i32>>()
        );
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
            let w = crate::io::writer::Hdf5Writer::open_append(&path).unwrap();
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

    // Two or more unlimited dimensions select the v2 B-tree index; with a
    // filter its records become type 11, carrying each chunk's stored size and
    // mask. The payload is highly compressible, so the chunks really are
    // stored smaller than the extent — the file would be at least
    // 6*8*4 = 192 bytes of raw chunk data otherwise.
    #[cfg(feature = "deflate")]
    #[test]
    fn compressed_multi_unlimited_dataset_roundtrips() {
        let path = temp_path("bt2_filtered");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([6, 8])
                .chunk(&[2, 4])
                .max_shape(&[None, None])
                .deflate(6)
                .create("d")
                .unwrap();
            ds.write_slice(&[0, 0], &[6, 8], &[7i32; 48]).unwrap();
            // A partial write forces a decompress-patch-recompress of one
            // chunk, whose new compressed size may not fit its old block.
            ds.write_slice(&[1, 1], &[2, 2], &[1i32, 2, 3, 4]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(ds.shape(), vec![6, 8]);
            let mut want = vec![7i32; 48];
            want[9] = 1;
            want[10] = 2;
            want[17] = 3;
            want[18] = 4;
            assert_eq!(ds.read_raw::<i32>().unwrap(), want);
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
                writer.ds(idx).lock().fill_value,
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
                writer.ds(idx).lock().fill_value,
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
        let writer = crate::io::writer::Hdf5Writer::create(&path).unwrap();
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
            let ds = w.ds(idx);
            let m = ds.lock();
            let entry = &m
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

    /// Two or more unlimited dimensions leave no fixed chunk grid for a linear
    /// index to mean anything against, so the linear entry point points the
    /// caller at the coordinate-addressed one rather than guessing a grid.
    #[test]
    fn write_chunk_raw_sends_btree_v2_to_the_coordinate_form() {
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
            err.to_string().contains("write_chunk_raw_at"),
            "expected a pointer to the coordinate form, got: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Direct chunk writes on a v2-B-tree index: the bytes are stored verbatim
    /// and the type-11 record carries their size and the caller's mask, so a
    /// chunk written with the pipeline skipped (mask 1) reads back as the raw
    /// bytes while one written compressed (mask 0) is decompressed.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_at_round_trips_on_btree_v2() {
        use crate::format::messages::filter::{apply_filters, FilterPipeline};

        let path = temp_path("wcr_at_btree2");
        let raw0: Vec<u8> = (0..4i32).flat_map(|v| v.to_le_bytes()).collect();
        let raw1: Vec<u8> = (100..104i32).flat_map(|v| v.to_le_bytes()).collect();
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, 0])
                .chunk(&[2, 2])
                .max_shape(&[None, None])
                .deflate(6)
                .create("grid")
                .unwrap();
            let pipeline = FilterPipeline::deflate(6);
            // Chunk (0,0): pipeline already applied upstream, mask 0.
            ds.write_chunk_raw_at(&[0, 0], &apply_filters(&pipeline, &raw0).unwrap(), 0)
                .unwrap();
            // Chunk (1,1): stored uncompressed, mask 1 says filter 0 was skipped.
            ds.write_chunk_raw_at(&[1, 1], &raw1, 1).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("grid").unwrap();
        assert_eq!(ds.shape(), vec![4, 4]);
        let all = ds.read_raw::<i32>().unwrap();
        // Chunk (0,0) occupies rows 0..2, columns 0..2.
        assert_eq!([all[0], all[1], all[4], all[5]], [0, 1, 2, 3]);
        // Chunk (1,1) occupies rows 2..4, columns 2..4.
        assert_eq!([all[10], all[11], all[14], all[15]], [100, 101, 102, 103]);
        drop(file);
        std::fs::remove_file(&path).ok();
    }

    /// The coordinate form is not BT2-only: it addresses an extensible- or
    /// fixed-array dataset's grid just as well, and records the same mask.
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_at_round_trips_on_the_array_indexes() {
        for (label, max_shape) in [
            ("wcr_at_ea", Some(vec![None, Some(4usize)])),
            ("wcr_at_fa", None),
        ] {
            let path = temp_path(label);
            let raw: Vec<u8> = (0..8i32).flat_map(|v| v.to_le_bytes()).collect();
            {
                let file = H5File::create(&path).unwrap();
                let mut b = file
                    .new_dataset::<i32>()
                    .shape([4usize, 4])
                    .chunk(&[2, 4])
                    .deflate(6);
                if let Some(ref ms) = max_shape {
                    b = b.max_shape(ms);
                }
                let ds = b.create("grid").unwrap();
                // Row-of-chunks 1, stored uncompressed with filter 0 skipped.
                ds.write_chunk_raw_at(&[1, 0], &raw, 1).unwrap();
                file.close().unwrap();
            }
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("grid").unwrap();
            let all = ds.read_raw::<i32>().unwrap();
            assert_eq!(&all[8..16], &(0..8).collect::<Vec<i32>>()[..], "{label}");
            drop(file);
            std::fs::remove_file(&path).ok();
        }
    }

    /// A direct write hands over caller-supplied bytes, so the v2 B-tree's
    /// chunk-size field can overflow just as the array indexes' can. A 4-byte
    /// chunk gives chunk_size_len = 2 (max 65535).
    #[cfg(feature = "deflate")]
    #[test]
    fn write_chunk_raw_at_rejects_an_oversized_btree_v2_chunk() {
        let path = temp_path("wcr_at_oversized");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0, 0])
            .chunk(&[1, 1])
            .max_shape(&[None, None])
            .deflate(4)
            .create("grid")
            .unwrap();
        let err = ds
            .write_chunk_raw_at(&[0, 0], &vec![0u8; 70000], 0)
            .unwrap_err();
        assert!(
            err.to_string().contains("does not fit"),
            "expected a chunk-size-field overflow error, got: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// An unfiltered v2 B-tree record has no slot for a stored size or mask,
    /// the same reason the array indexes reject a direct write.
    #[test]
    fn write_chunk_raw_at_rejects_an_unfiltered_btree_v2() {
        let path = temp_path("wcr_at_unfiltered");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([0, 0])
            .chunk(&[2, 2])
            .max_shape(&[None, None])
            .create("grid")
            .unwrap();
        let err = ds.write_chunk_raw_at(&[0, 0], &[0u8; 16], 0).unwrap_err();
        assert!(
            err.to_string().contains("filtered dataset"),
            "expected a filtered-dataset error, got: {err}"
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

    // ---- issue #5: runtime-width fixed-string reading ----------------------

    use crate::format::messages::datatype::{CompoundMember, DatatypeMessage};

    /// Build a 1-D fixed-string dataset of `width` bytes per element from raw
    /// element images, optionally chunked and deflated.
    fn write_fixed_string_dataset(
        path: &std::path::Path,
        dt: DatatypeMessage,
        width: usize,
        elems: &[&[u8]],
        compressed: bool,
    ) {
        let mut raw = Vec::with_capacity(elems.len() * width);
        for e in elems {
            assert!(e.len() <= width);
            raw.extend_from_slice(e);
            raw.resize(raw.len() + (width - e.len()), 0);
        }
        let file = H5File::create(path).unwrap();
        let mut b = file.new_dataset::<u8>().datatype(dt).shape([elems.len()]);
        if compressed {
            b = b.chunk(&[2]).deflate(6);
        }
        let ds = b.create("labels").unwrap();
        ds.write_raw_bytes(&raw).unwrap();
        file.close().unwrap();
    }

    /// The width is whatever the file says, so one call reads a 24-byte label
    /// column and a 100-byte one. Producers like VASP pick it per dataset.
    #[test]
    fn read_strings_handles_any_fixed_width() {
        for width in [4usize, 24, 100] {
            let path = temp_path(&format!("fixed_str_{width}"));
            write_fixed_string_dataset(
                &path,
                DatatypeMessage::fixed_string(width as u32),
                width,
                &[b"ab", b"cde", b""],
                false,
            );
            let file = H5File::open(&path).unwrap();
            let got = file.dataset("labels").unwrap().read_strings().unwrap();
            assert_eq!(got, vec!["ab", "cde", ""], "width {width}");
            std::fs::remove_file(&path).ok();
        }
    }

    /// Each padding rule decides where the value ends. Null-terminated and
    /// null-padded both stop at the first NUL and ignore the bytes after it;
    /// space-padded strips only a tail of spaces, so an embedded NUL is
    /// content there.
    ///
    /// Checked against libhdf5 1.14.6: reading this same `"ab\0X\0\0"`
    /// null-padded element into a wider null-terminated destination gives
    /// `"ab"`, and reading a space-padded `"a\0b     "` gives `"a\0b"` —
    /// `H5T__conv_s_s` runs the same `!s[nchars]` loop for both null rules.
    #[test]
    fn read_strings_honors_every_padding_rule() {
        // "ab" then a NUL then trailing junk that both null rules must drop.
        let elem: &[u8] = b"ab\0X\0\0";
        for (padding, want) in [(0u8, "ab"), (1, "ab")] {
            let path = temp_path(&format!("fixed_pad_{padding}"));
            write_fixed_string_dataset(
                &path,
                DatatypeMessage::FixedString {
                    size: 6,
                    padding,
                    charset: 0,
                },
                6,
                &[elem],
                false,
            );
            let file = H5File::open(&path).unwrap();
            let got = file.dataset("labels").unwrap().read_strings().unwrap();
            assert_eq!(got, vec![want.to_string()], "padding {padding}");
            std::fs::remove_file(&path).ok();
        }
        // Space-padded keeps interior spaces and strips only the tail.
        let path = temp_path("fixed_pad_2");
        write_fixed_string_dataset(
            &path,
            DatatypeMessage::FixedString {
                size: 8,
                padding: 2,
                charset: 0,
            },
            8,
            &[b"a b     "],
            false,
        );
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("labels").unwrap().read_strings().unwrap(),
            vec!["a b".to_string()]
        );
        std::fs::remove_file(&path).ok();

        // ... and an embedded NUL, which no space rule marks as an end.
        let path = temp_path("fixed_pad_2_nul");
        write_fixed_string_dataset(
            &path,
            DatatypeMessage::FixedString {
                size: 8,
                padding: 2,
                charset: 0,
            },
            8,
            &[b"a\0b     "],
            false,
        );
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("labels").unwrap().read_strings().unwrap(),
            vec!["a\0b".to_string()]
        );
        std::fs::remove_file(&path).ok();
    }

    /// A reserved padding or character-set code is an error naming the element,
    /// not a guess.
    #[test]
    fn read_strings_rejects_reserved_datatype_codes() {
        for (padding, charset, want) in [(3u8, 0u8, "padding rule 3"), (0, 7, "character set 7")] {
            let path = temp_path(&format!("fixed_reserved_{padding}_{charset}"));
            write_fixed_string_dataset(
                &path,
                DatatypeMessage::FixedString {
                    size: 4,
                    padding,
                    charset,
                },
                4,
                &[b"ab"],
                false,
            );
            let file = H5File::open(&path).unwrap();
            let err = file
                .dataset("labels")
                .unwrap()
                .read_strings()
                .unwrap_err()
                .to_string();
            assert!(err.contains(want), "got: {err}");
            std::fs::remove_file(&path).ok();
        }
    }

    /// The typed read paths reinterpret the element image, so the stored order
    /// has to be the host's first. A scalar is swapped; a composite cannot be
    /// (its members have their own orders and offsets) and is refused.
    #[test]
    fn to_host_byte_order_converts_scalars_and_refuses_composites() {
        use crate::dataset::{to_host_byte_order, HOST_BYTE_ORDER};
        use crate::format::messages::datatype::ByteOrder;

        let foreign = match HOST_BYTE_ORDER {
            ByteOrder::LittleEndian => ByteOrder::BigEndian,
            ByteOrder::BigEndian => ByteOrder::LittleEndian,
        };
        let int = |order, size| DatatypeMessage::FixedPoint {
            size,
            byte_order: order,
            signed: false,
            bit_offset: 0,
            bit_precision: (size * 8) as u16,
        };

        // Foreign order: each element is reversed, elementwise.
        let mut buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        to_host_byte_order(&mut buf, &int(foreign, 4), 4).unwrap();
        assert_eq!(buf, [4, 3, 2, 1, 8, 7, 6, 5]);

        // Host order: untouched.
        let mut buf = [1u8, 2, 3, 4];
        to_host_byte_order(&mut buf, &int(HOST_BYTE_ORDER, 4), 4).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);

        // One byte wide: no order to convert.
        let mut buf = [1u8, 2, 3, 4];
        to_host_byte_order(&mut buf, &int(foreign, 1), 1).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);

        // An enum stores its values in its base type's order.
        let mut buf = [1u8, 2];
        let enumeration = DatatypeMessage::Enum {
            base: Box::new(int(foreign, 2)),
            members: Vec::new(),
        };
        to_host_byte_order(&mut buf, &enumeration, 2).unwrap();
        assert_eq!(buf, [2, 1]);

        // A string has no byte order at all.
        let mut buf = *b"abcd";
        to_host_byte_order(&mut buf, &DatatypeMessage::fixed_string(4), 4).unwrap();
        assert_eq!(&buf, b"abcd");

        // A compound whose members are all host-order is reinterpretable.
        let compound = |order| DatatypeMessage::Compound {
            size: 4,
            members: vec![CompoundMember {
                name: "x".into(),
                offset: 0,
                datatype: int(order, 4),
            }],
        };
        let mut buf = [1u8, 2, 3, 4];
        to_host_byte_order(&mut buf, &compound(HOST_BYTE_ORDER), 4).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);

        // One that is not says so, rather than handing back the raw bytes.
        let mut buf = [1u8, 2, 3, 4];
        let err = to_host_byte_order(&mut buf, &compound(foreign), 4)
            .expect_err("a foreign-order compound was reinterpreted")
            .to_string();
        assert!(err.contains("read_raw_bytes"), "got: {err}");
        assert_eq!(buf, [1, 2, 3, 4], "the refused image is left alone");
    }

    /// The write direction answers for exactly the types the read direction
    /// does — same classifier — and borrows the caller's bytes whenever the
    /// declared order is already the host's.
    #[test]
    fn to_stored_byte_order_converts_scalars_and_refuses_composites() {
        use crate::dataset::{to_stored_byte_order, FOREIGN_BYTE_ORDER, HOST_BYTE_ORDER};
        use std::borrow::Cow;

        let int = |order, size| DatatypeMessage::FixedPoint {
            size,
            byte_order: order,
            signed: false,
            bit_offset: 0,
            bit_precision: (size * 8) as u16,
        };

        // Declared foreign: each element is reversed on the way out.
        let host = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let stored = to_stored_byte_order(&host, &int(FOREIGN_BYTE_ORDER, 4), 4).unwrap();
        assert_eq!(&*stored, &[4, 3, 2, 1, 8, 7, 6, 5]);
        assert!(matches!(stored, Cow::Owned(_)), "a swap needs its own copy");

        // Declared host order: handed through without a copy.
        let stored = to_stored_byte_order(&host, &int(HOST_BYTE_ORDER, 4), 4).unwrap();
        assert!(matches!(stored, Cow::Borrowed(_)), "no copy without a swap");
        assert_eq!(&*stored, &host);

        // One byte wide: no order to lay out.
        let stored = to_stored_byte_order(&host, &int(FOREIGN_BYTE_ORDER, 1), 1).unwrap();
        assert_eq!(&*stored, &host);

        // An enum stores its values in its base type's order.
        let enumeration = DatatypeMessage::Enum {
            base: Box::new(int(FOREIGN_BYTE_ORDER, 2)),
            members: Vec::new(),
        };
        let stored = to_stored_byte_order(&[1u8, 2], &enumeration, 2).unwrap();
        assert_eq!(&*stored, &[2, 1]);

        // A compound cannot be laid out as a unit; one that declares the
        // foreign order for a member is refused, not written host-order.
        let compound = |order| DatatypeMessage::Compound {
            size: 4,
            members: vec![CompoundMember {
                name: "x".into(),
                offset: 0,
                datatype: int(order, 4),
            }],
        };
        let stored = to_stored_byte_order(&[1u8, 2, 3, 4], &compound(HOST_BYTE_ORDER), 4).unwrap();
        assert_eq!(&*stored, &[1, 2, 3, 4]);
        let err = to_stored_byte_order(&[1u8, 2, 3, 4], &compound(FOREIGN_BYTE_ORDER), 4)
            .expect_err("a foreign-order compound was written from host bytes")
            .to_string();
        assert!(err.contains("write_raw_bytes"), "got: {err}");
    }

    /// The declared character set is enforced: a byte that cannot be decoded is
    /// an error naming the element, and the lossy call is what accepts the file
    /// instead of a silent substitution here.
    #[test]
    fn read_strings_enforces_the_character_set_and_lossy_does_not() {
        // Latin-1 "é" (0xE9) in a dataset that declares ASCII, and a lone 0xFF
        // in one that declares UTF-8.
        for (charset, bytes, want) in [
            (0u8, b"caf\xe9".as_slice(), "ASCII character set"),
            (1, b"a\xff".as_slice(), "not valid UTF-8"),
        ] {
            let path = temp_path(&format!("fixed_charset_{charset}"));
            write_fixed_string_dataset(
                &path,
                DatatypeMessage::FixedString {
                    size: 6,
                    padding: 1,
                    charset,
                },
                6,
                &[b"ok", bytes],
                false,
            );
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("labels").unwrap();
            let err = ds.read_strings().unwrap_err().to_string();
            assert!(err.contains(want) && err.contains("string 1"), "got: {err}");
            let lossy = ds.read_strings_lossy().unwrap();
            assert_eq!(lossy[0], "ok");
            assert_eq!(
                lossy[1].chars().next().unwrap(),
                if charset == 0 { 'c' } else { 'a' }
            );
            std::fs::remove_file(&path).ok();
        }
    }

    /// Valid multi-byte UTF-8 survives, and the trailing NUL padding does not
    /// split a character.
    #[test]
    fn read_strings_reads_utf8_fixed_strings() {
        let path = temp_path("fixed_utf8");
        write_fixed_string_dataset(
            &path,
            DatatypeMessage::fixed_string_utf8(12),
            12,
            &["héllo".as_bytes(), "안녕".as_bytes()],
            false,
        );
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("labels").unwrap().read_strings().unwrap(),
            vec!["héllo".to_string(), "안녕".to_string()]
        );
        std::fs::remove_file(&path).ok();
    }

    /// The decode sits on the decoded raw-data path, so a chunked and deflated
    /// dataset reads the same as a contiguous one.
    #[cfg(feature = "deflate")]
    #[test]
    fn read_strings_reads_a_compressed_fixed_string_dataset() {
        let path = temp_path("fixed_str_deflate");
        write_fixed_string_dataset(
            &path,
            DatatypeMessage::fixed_string(16),
            16,
            &[b"alpha", b"beta", b"gamma", b"delta", b"epsilon"],
            true,
        );
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("labels").unwrap().read_strings().unwrap(),
            vec!["alpha", "beta", "gamma", "delta", "epsilon"]
        );
        std::fs::remove_file(&path).ok();
    }

    /// One call covers both string datatypes, so a caller need not branch on
    /// which one the file used.
    #[test]
    fn read_strings_also_reads_variable_length_strings() {
        let path = temp_path("read_strings_vlen");
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("names", &["alpha", "", "안녕"])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("names").unwrap().read_strings().unwrap(),
            vec!["alpha".to_string(), String::new(), "안녕".to_string()]
        );
        std::fs::remove_file(&path).ok();
    }

    /// A file declaring a zero-width fixed string is an error, not the panic
    /// `chunks_exact(0)` would raise. Nothing in this crate writes one, so the
    /// test patches the width in the encoded datatype message down to zero and
    /// re-stamps the object header's checksum over the result.
    #[test]
    fn read_strings_rejects_a_zero_width_fixed_string_dataset() {
        use crate::format::checksum::checksum_metadata;
        use crate::format::object_header::OHDR_SIGNATURE;

        let path = temp_path("fixed_str_zero_width");
        write_fixed_string_dataset(
            &path,
            DatatypeMessage::fixed_string(37),
            37,
            &[b"ab", b"cd"],
            false,
        );

        // Version 1 string datatype: class|version, padding|charset, two
        // reserved bytes, then the width as a little-endian u32. The width is
        // 37 so the eight bytes occur once in the file.
        let mut bytes = std::fs::read(&path).unwrap();
        let needle = [0x13u8, 0, 0, 0, 37, 0, 0, 0];
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("encoded fixed-string datatype message");
        assert!(
            !bytes[at + 1..].windows(needle.len()).any(|w| w == needle),
            "the datatype message pattern is not unique in the file"
        );

        // The enclosing v2 object header ends in a checksum over everything
        // from its signature onwards; find the offset where the stored value
        // still agrees, so the patched header can be re-stamped there.
        let ohdr = bytes[..at]
            .windows(4)
            .rposition(|w| w == OHDR_SIGNATURE)
            .expect("enclosing object header");
        let cksum_at = (at + needle.len()..bytes.len() - 4)
            .find(|&e| {
                u32::from_le_bytes(bytes[e..e + 4].try_into().unwrap())
                    == checksum_metadata(&bytes[ohdr..e])
            })
            .expect("object header checksum");

        bytes[at + 4..at + 8].copy_from_slice(&0u32.to_le_bytes());
        let fixed = checksum_metadata(&bytes[ohdr..cksum_at]);
        bytes[cksum_at..cksum_at + 4].copy_from_slice(&fixed.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let file = H5File::open(&path).unwrap();
        let err = file
            .dataset("labels")
            .unwrap()
            .read_strings()
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero width"), "got: {err}");
        std::fs::remove_file(&path).ok();
    }

    /// A non-string dataset is an error, not an attempt to reinterpret bytes.
    #[test]
    fn read_strings_rejects_a_non_string_dataset() {
        let path = temp_path("read_strings_numeric");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<i32>().shape([3]).create("nums").unwrap();
            ds.write_raw(&[1i32, 2, 3]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let err = file
            .dataset("nums")
            .unwrap()
            .read_strings()
            .unwrap_err()
            .to_string();
        assert!(err.contains("only for string datasets"), "got: {err}");
        std::fs::remove_file(&path).ok();
    }

    // ---- issue #6: random updates to vlen string datasets ------------------

    /// One element changes; the extent and every other element stay as they
    /// were, on a contiguous vlen dataset.
    #[test]
    fn write_vlen_strings_slice_replaces_one_element() {
        let path = temp_path("vlen_slice_contig");
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("notes", &["a", "b", "c", "d"])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.dataset_writer("notes")
                .unwrap()
                .write_vlen_strings_slice(1, &["replacement"])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("notes").unwrap();
        assert_eq!(ds.shape(), vec![4]);
        assert_eq!(
            ds.read_vlen_strings().unwrap(),
            vec!["a", "replacement", "c", "d"]
        );
        std::fs::remove_file(&path).ok();
    }

    /// The same on an appendable chunked dataset, across a reopen, over a range
    /// that spans a chunk boundary.
    #[test]
    fn write_vlen_strings_slice_spans_chunks_after_reopen() {
        let path = temp_path("vlen_slice_chunked");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("notes", 2, None)
                .unwrap();
            let all: Vec<String> = (0..6).map(|i| format!("v{i}")).collect();
            let refs: Vec<&str> = all.iter().map(|s| s.as_str()).collect();
            file.append_vlen_strings("notes", &refs).unwrap();
            file.close().unwrap();
        }
        {
            // Elements 1..4 cross the 2-element chunk boundary twice.
            let file = H5File::open_rw(&path).unwrap();
            file.dataset_writer("notes")
                .unwrap()
                .write_vlen_strings_slice(1, &["x", "y", "z"])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("notes").unwrap();
        assert_eq!(ds.shape(), vec![6]);
        assert_eq!(
            ds.read_vlen_strings().unwrap(),
            vec!["v0", "x", "y", "z", "v4", "v5"]
        );
        std::fs::remove_file(&path).ok();
    }

    /// Elements the append buffer still holds are not on disk yet; the
    /// update flushes them to their chunks first, so the flush at close has
    /// nothing left to write the pre-update reference over.
    #[test]
    fn write_vlen_strings_slice_updates_buffered_elements() {
        let path = temp_path("vlen_slice_buffered");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("notes", 4, None)
                .unwrap();
            // 3 of a 4-element chunk: all three stay in the append buffer.
            file.append_vlen_strings("notes", &["a", "b", "c"]).unwrap();
            file.dataset_writer("notes")
                .unwrap()
                .write_vlen_strings_slice(1, &["patched"])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["a", "patched", "c"]
        );
        std::fs::remove_file(&path).ok();
    }

    /// A range past the end is rejected before anything is written, and an
    /// empty batch costs the file nothing — without the early return it would
    /// still allocate and write an empty global-heap collection.
    #[test]
    fn write_vlen_strings_slice_checks_its_range() {
        let build = |name: &str, empty_call: bool| {
            let path = temp_path(name);
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("notes", &["a", "b"]).unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            let err = ds
                .write_vlen_strings_slice(1, &["x", "y"])
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("outside the dataset's 2 elements"),
                "got: {err}"
            );
            if empty_call {
                ds.write_vlen_strings_slice(0, &[]).unwrap();
            }
            file.close().unwrap();
            path
        };

        let with_empty = build("vlen_slice_range", true);
        let control = build("vlen_slice_range_control", false);
        assert_eq!(
            std::fs::metadata(&with_empty).unwrap().len(),
            std::fs::metadata(&control).unwrap().len(),
            "the rejected and empty calls must leave the file untouched"
        );

        let file = H5File::open(&with_empty).unwrap();
        assert_eq!(
            file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["a", "b"]
        );
        std::fs::remove_file(&with_empty).ok();
        std::fs::remove_file(&control).ok();
    }

    /// The element offset is one-dimensional, so a multi-dimensional dataset is
    /// rejected rather than silently indexed along the first axis.
    #[test]
    fn write_vlen_strings_slice_rejects_a_multidimensional_dataset() {
        let path = temp_path("vlen_slice_2d");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<u8>()
            .datatype(DatatypeMessage::vlen_string_utf8())
            .shape([2, 3])
            .create("grid")
            .unwrap();
        let err = ds
            .write_vlen_strings_slice(0, &["x"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("1-dimension datasets"), "got: {err}");
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A `&str` is UTF-8, so writing a non-ASCII one into a dataset that
    /// declares the ASCII character set would mislabel the bytes.
    #[test]
    fn write_vlen_strings_slice_enforces_the_ascii_character_set() {
        let path = temp_path("vlen_slice_ascii");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<u8>()
            .datatype(DatatypeMessage::vlen_string_ascii())
            .shape([3])
            .create("notes")
            .unwrap();
        let err = ds
            .write_vlen_strings_slice(0, &["ok", "안녕"])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("string 1") && err.contains("is not ASCII"),
            "got: {err}"
        );
        ds.write_vlen_strings_slice(0, &["ok", "fine"]).unwrap();
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A numeric dataset is rejected: its elements are not vlen references and
    /// writing one would corrupt the column.
    #[test]
    fn write_vlen_strings_slice_rejects_a_non_vlen_dataset() {
        let path = temp_path("vlen_slice_numeric");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<i32>().shape([3]).create("nums").unwrap();
        ds.write_raw(&[1i32, 2, 3]).unwrap();
        let err = ds
            .write_vlen_strings_slice(0, &["x"])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("only for variable-length string datasets"),
            "got: {err}"
        );
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    // ---- superseded global heap objects (libhdf5 H5HG_remove parity) -------

    /// Repeatedly replacing the same element must not grow the file per
    /// update: the collection each update supersedes is freed and the next
    /// update's collection lands in that block. Without the release every
    /// update costs another `H5HG_MINALLOC` (4096) bytes.
    #[test]
    fn write_vlen_strings_slice_reuses_the_freed_heap_block() {
        let size_after = |updates: usize| {
            let path = temp_path(&format!("vlen_slice_heap_reuse_{updates}"));
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("notes", &["a", "b"]).unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            for i in 0..updates {
                ds.write_vlen_strings_slice(0, &[&format!("update {i}")])
                    .unwrap();
            }
            file.close().unwrap();
            let n = std::fs::metadata(&path).unwrap().len();
            let read = H5File::open(&path).unwrap();
            assert_eq!(
                read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
                vec![format!("update {}", updates - 1), "b".to_string()]
            );
            drop(read);
            std::fs::remove_file(&path).ok();
            n
        };

        // The allocator settles once a freed block is available to reuse, so
        // every count past that produces the same file.
        let settled = size_after(3);
        assert_eq!(size_after(20), settled, "20 updates against 3");
        assert_eq!(size_after(50), settled, "50 updates against 3");
    }

    /// An empty string is stored as a real heap object under a reference whose
    /// sequence length is zero, so the release must go by the address, not the
    /// length — a length test strands the object and its collection forever.
    #[test]
    fn write_vlen_strings_slice_frees_an_empty_strings_object() {
        let size_after = |updates: usize| {
            let path = temp_path(&format!("vlen_slice_empty_reuse_{updates}"));
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("notes", &["", "b"]).unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            for _ in 0..updates {
                ds.write_vlen_strings_slice(0, &[""]).unwrap();
            }
            file.close().unwrap();
            let n = std::fs::metadata(&path).unwrap().len();
            let read = H5File::open(&path).unwrap();
            assert_eq!(
                read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
                vec!["".to_string(), "b".to_string()]
            );
            drop(read);
            std::fs::remove_file(&path).ok();
            n
        };

        let settled = size_after(3);
        assert_eq!(size_after(20), settled, "20 empty updates against 3");
        assert_eq!(size_after(50), settled, "50 empty updates against 3");
    }

    /// The elements the update does not name keep their strings, so freeing
    /// the superseded objects must not disturb the collection's survivors.
    #[test]
    fn write_vlen_strings_slice_keeps_the_untouched_strings_readable() {
        let path = temp_path("vlen_slice_heap_survivors");
        {
            let file = H5File::create(&path).unwrap();
            file.write_vlen_strings("notes", &["a", "b", "c", "d"])
                .unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            // Two updates inside the one collection the create wrote, so the
            // second reads a collection the first already rewrote.
            ds.write_vlen_strings_slice(1, &["B"]).unwrap();
            ds.write_vlen_strings_slice(3, &["D"]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["a", "B", "c", "D"]
        );
        std::fs::remove_file(&path).ok();
    }

    /// Replacing every element of a chunked dataset empties the collection the
    /// append wrote, and the file must still read back correctly after its
    /// block goes to the allocator.
    #[test]
    fn write_vlen_strings_slice_frees_an_emptied_collection() {
        let path = temp_path("vlen_slice_heap_emptied");
        {
            let file = H5File::create(&path).unwrap();
            file.create_appendable_vlen_dataset("notes", 2, None)
                .unwrap();
            file.append_vlen_strings("notes", &["p", "q", "r", "s"])
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.dataset_writer("notes")
                .unwrap()
                .write_vlen_strings_slice(0, &["w", "x", "y", "z"])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["w", "x", "y", "z"]
        );
        std::fs::remove_file(&path).ok();
    }

    /// A collection larger than the 4096-byte minimum must keep its size when
    /// an object leaves it. Re-encoding at the natural size instead shrinks
    /// what the header declares, so the block's tail stops being part of the
    /// collection and the eventual free returns less than was allocated —
    /// stranding the difference on every cycle.
    #[test]
    fn write_vlen_strings_slice_keeps_an_oversized_collections_block_whole() {
        let big = |tag: char| std::iter::repeat_n(tag, 2000).collect::<String>();
        let size_after = |cycles: usize| {
            let path = temp_path(&format!("vlen_slice_heap_big_{cycles}"));
            let file = H5File::create(&path).unwrap();
            let seed: Vec<String> = "abcd".chars().map(big).collect();
            let refs: Vec<&str> = seed.iter().map(|s| s.as_str()).collect();
            // Four 2000-byte strings do not fit the 4096-byte minimum, so this
            // is one collection well above it.
            file.write_vlen_strings("notes", &refs).unwrap();
            let ds = file.dataset_writer("notes").unwrap();
            for _ in 0..cycles {
                // Partially empty the collection, then finish it off: the
                // block is freed only after it has been rewritten once.
                let head = big('x');
                ds.write_vlen_strings_slice(0, &[&head]).unwrap();
                let tail: Vec<String> = "yzw".chars().map(big).collect();
                let tail_refs: Vec<&str> = tail.iter().map(|s| s.as_str()).collect();
                ds.write_vlen_strings_slice(1, &tail_refs).unwrap();
            }
            file.close().unwrap();
            let n = std::fs::metadata(&path).unwrap().len();
            let read = H5File::open(&path).unwrap();
            assert_eq!(
                read.dataset("notes").unwrap().read_vlen_strings().unwrap(),
                vec![big('x'), big('y'), big('z'), big('w')]
            );
            drop(read);
            std::fs::remove_file(&path).ok();
            n
        };

        let settled = size_after(4);
        assert_eq!(size_after(30), settled, "30 cycles against 4");
    }

    /// An element still in the append buffer has never been on disk, so its
    /// superseded object has to be found in the buffer or it is stranded.
    #[test]
    fn write_vlen_strings_slice_releases_a_buffered_elements_object() {
        let path = temp_path("vlen_slice_heap_buffered");
        let file = H5File::create(&path).unwrap();
        file.create_appendable_vlen_dataset("notes", 4, None)
            .unwrap();
        file.append_vlen_strings("notes", &["a", "b", "c"]).unwrap();
        let ds = file.dataset_writer("notes").unwrap();
        for i in 0..20 {
            ds.write_vlen_strings_slice(1, &[&format!("patch {i}")])
                .unwrap();
        }
        file.close().unwrap();

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("notes").unwrap().read_vlen_strings().unwrap(),
            vec!["a", "patch 19", "c"]
        );
        let size = std::fs::metadata(&path).unwrap().len();
        std::fs::remove_file(&path).ok();
        assert!(
            size < 20 * 4096,
            "20 buffered updates left {size} bytes, one collection per update"
        );
    }

    /// Regression: a typed `write_slice` into rows the append buffer still
    /// held wrote the chunks, and the flush at close wrote the stale buffered
    /// rows back over it — write 99, read 50. The slice now flushes the
    /// buffer first, making the chunks the single authority for those rows.
    #[test]
    fn write_slice_into_the_buffered_tail_survives_close() {
        let path = temp_path("slice_into_buffered_tail");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .create("d")
                .unwrap();
            // 6 rows: 4 land in chunk 0, rows 4 and 5 stay buffered.
            ds.append(&[10, 11, 12, 13, 50, 51]).unwrap();
            ds.write_slice(&[4], &[1], &[99]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(ds.read_raw::<i32>().unwrap(), vec![10, 11, 12, 13, 99, 51]);
        }
        std::fs::remove_file(&path).ok();
    }

    /// Extending a dataset while appends sit in the buffer must not move
    /// them: the buffer records the absolute row its frames belong to, so
    /// the flush at close lands them there, and the grown region reads as
    /// fill.
    #[test]
    fn extend_does_not_move_buffered_appends() {
        let path = temp_path("extend_keeps_buffered_rows");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0])
                .chunk(&[4])
                .max_shape(&[None])
                .create("d")
                .unwrap();
            ds.append(&[10, 11, 12, 13, 50, 51]).unwrap(); // rows 4, 5 buffered
            ds.extend(&[10]).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                vec![10, 11, 12, 13, 50, 51, 0, 0, 0, 0]
            );
        }
        std::fs::remove_file(&path).ok();
    }

    /// Regression: appends to a v2 B-tree indexed dataset (two unlimited
    /// dimensions) buffered fine but close() failed "not a chunked dataset"
    /// and lost the buffered rows — the append's chunk writes required the
    /// extensible-array index. They now go through the index-generic
    /// hyperslab engine.
    #[test]
    fn append_to_a_btree_v2_dataset_survives_close() {
        let path = temp_path("append_bt2_close");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, 3])
                .chunk(&[4, 3])
                .max_shape(&[None, None])
                .create("d")
                .unwrap();
            // One buffered row, then a batch that crosses the chunk
            // boundary: 4 rows fill chunk band 0, one row stays buffered
            // for the flush at close.
            ds.append(&[1, 2, 3]).unwrap();
            ds.append(&(4..=15).collect::<Vec<i32>>()).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(ds.shape(), vec![5, 3]);
            assert_eq!(
                ds.read_raw::<i32>().unwrap(),
                (1..=15).collect::<Vec<i32>>()
            );
        }
        std::fs::remove_file(&path).ok();
    }

    /// A chunk row narrower than the frame row is legal geometry (libhdf5
    /// creates it); appended frames must be scattered across the row's
    /// tiles at the chunk stride, not packed at the frame stride.
    #[test]
    fn append_scatters_frames_across_narrow_chunk_tiles() {
        let path = temp_path("append_narrow_chunks");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([0, 8])
                .chunk(&[2, 4])
                .max_shape(&[None, Some(8)])
                .create("d")
                .unwrap();
            // 3 rows of 8: rows 0..2 complete chunk band 0 (two tiles),
            // row 2 is flushed partial at close.
            ds.append(&(0..24).collect::<Vec<i32>>()).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(ds.shape(), vec![3, 8]);
            assert_eq!(ds.read_raw::<i32>().unwrap(), (0..24).collect::<Vec<i32>>());
        }
        std::fs::remove_file(&path).ok();
    }

    /// A fixed-array dataset has no room to grow: appending must surface an
    /// error naming the chunk grid, not lose rows silently. (Before the
    /// index-generic append it failed as "not a chunked dataset".)
    #[test]
    fn append_to_a_full_fixed_array_dataset_errors() {
        let path = temp_path("append_fa_errors");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 3])
            .chunk(&[2, 3])
            .create("d")
            .unwrap();
        let err = ds.append(&(0..6).collect::<Vec<i32>>()).unwrap_err();
        assert!(
            err.to_string().contains("chunk grid"),
            "unexpected error: {err}"
        );
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// A finite max_shape above the current shape used to be dropped on the
    /// fixed-array path: the array was sized from the current dims and the
    /// stored dataspace had no maximum, so growth failed. The array is now
    /// sized from the maximum's chunk grid (libhdf5 `max_nchunks`), so a
    /// fixed-max dataset appends up to its maximum and roundtrips.
    #[test]
    fn fixed_array_with_a_larger_max_shape_grows_and_survives_close() {
        let path = temp_path("fa_growable_dim0");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([4, 3])
                .chunk(&[2, 3])
                .max_shape(&[Some(10), Some(3)])
                .create("d")
                .unwrap();
            ds.write_raw(&(0..12).collect::<Vec<i32>>()).unwrap();
            ds.append(&(12..18).collect::<Vec<i32>>()).unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(ds.shape(), vec![6, 3]);
            assert_eq!(ds.read_raw::<i32>().unwrap(), (0..18).collect::<Vec<i32>>());
        }
        std::fs::remove_file(&path).ok();
    }

    /// The multiplier-dimension boundary: growing a dimension other than 0
    /// changes the current chunk grid but not the index grid. Chunk slots
    /// must come from the maximum's grid (libhdf5 `max_down_chunks`), or the
    /// chunks written before the extend are looked up under different
    /// indices after it.
    #[test]
    fn fixed_array_growable_inner_dimension_keeps_chunk_slots() {
        let path = temp_path("fa_growable_dim1");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file
                .new_dataset::<i32>()
                .shape([4, 3])
                .chunk(&[2, 3])
                .max_shape(&[Some(4), Some(9)])
                .create("d")
                .unwrap();
            ds.write_raw(&(0..12).collect::<Vec<i32>>()).unwrap();
            ds.extend(&[4, 6]).unwrap();
            ds.write_slice(&[0, 3], &[4, 3], &(12..24).collect::<Vec<i32>>())
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open(&path).unwrap();
            let ds = file.dataset("d").unwrap();
            assert_eq!(ds.shape(), vec![4, 6]);
            // Row-major [4,6]: row r is [r*3 .. r*3+3) from the first write
            // then [12 + r*3 ..) from the second.
            let mut expect = Vec::new();
            for r in 0i32..4 {
                expect.extend((r * 3)..(r * 3 + 3));
                expect.extend((12 + r * 3)..(12 + r * 3 + 3));
            }
            assert_eq!(ds.read_raw::<i32>().unwrap(), expect);
        }
        std::fs::remove_file(&path).ok();
    }

    /// Growth boundaries: past the stored maximum is rejected, and a dataset
    /// without a stored maximum is fixed at its extent (libhdf5 defaults
    /// maxdims to dims at creation).
    #[test]
    fn extend_beyond_the_maximum_is_rejected() {
        let path = temp_path("extend_beyond_max");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 3])
            .chunk(&[2, 3])
            .max_shape(&[Some(6), Some(3)])
            .create("d")
            .unwrap();
        ds.extend(&[6, 3]).unwrap();
        let err = ds.extend(&[8, 3]).unwrap_err();
        assert!(
            err.to_string().contains("exceeds the maximum"),
            "unexpected error: {err}"
        );
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// An unlimited dimension other than 0 has no fixed linear slot without
    /// libhdf5's extensible-array swizzling; `chunk_grid::linear_index` now
    /// implements that swizzle for any dimension, so this creates cleanly
    /// and every extend keeps writing new chunks to new slots, never
    /// re-addressing one already on disk.
    #[test]
    fn builder_accepts_an_unlimited_inner_dimension() {
        let path = temp_path("unlimited_inner_dim");
        let file = H5File::create(&path).unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape([4, 0])
            .chunk(&[2, 2])
            .max_shape(&[Some(4), None])
            .create("d")
            .unwrap();
        assert_eq!(ds.shape(), vec![4, 0]);

        // Write, then extend and write again: if the linear index were
        // recomputed from the *current* extent instead of the maximum one,
        // the second extend would shift every slot number and the first
        // write's chunks would decode under the wrong coordinates below.
        ds.extend(&[4, 2]).unwrap();
        ds.write_slice(&[0, 0], &[4, 2], &[1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        ds.extend(&[4, 4]).unwrap();
        ds.write_slice(&[0, 2], &[4, 2], &[9, 10, 11, 12, 13, 14, 15, 16])
            .unwrap();

        file.close().unwrap();
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        assert_eq!(
            ds.read_slice::<i32>(&[0, 0], &[4, 4]).unwrap(),
            vec![1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16]
        );
        std::fs::remove_file(&path).ok();
    }

    /// Regression: a chunk wider than a fixed max dimension used to be
    /// accepted, and appends then packed rows at the chunk stride — writing
    /// [1, 2, 3, 4] and reading back [1, 2, 0, 0]. libhdf5 rejects the
    /// geometry at create (`H5D__chunk_construct`); so do we now.
    #[test]
    fn builder_rejects_a_chunk_wider_than_a_fixed_max_dimension() {
        let path = temp_path("builder_chunk_wider_than_max");
        let file = H5File::create(&path).unwrap();
        let err = match file
            .new_dataset::<i32>()
            .shape([0, 2])
            .chunk(&[2, 4])
            .max_shape(&[None, Some(2)])
            .create("v5")
        {
            Ok(_) => panic!("create accepted a chunk wider than the fixed max dimension"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("maximum dimension size"),
            "unexpected error: {err}"
        );
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Boundary: `fits in T` vs `does not fit in T`, for both the too-large
    /// (u64::MAX → i64) and the negative-to-unsigned (−1 → u32) directions.
    #[test]
    fn numeric_int_checked_conversion_boundaries() {
        let path = temp_path("numeric_int_bounds");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<u64>().shape([2]).create("u").unwrap();
            ds.write_raw(&[1u64, u64::MAX]).unwrap();
            let ds = file.new_dataset::<i32>().shape([2]).create("i").unwrap();
            ds.write_raw(&[-1i32, 5]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();

        let u = file.dataset("u").unwrap();
        assert_eq!(u.read_numeric_as::<u64>().unwrap(), vec![1, u64::MAX]);
        assert_eq!(
            u.read_numeric_as::<i128>().unwrap(),
            vec![1, i128::from(u64::MAX)]
        );
        let err = u.read_numeric_as::<i64>().unwrap_err();
        assert!(
            err.to_string()
                .contains("value 18446744073709551615 at element 1 does not fit in i64"),
            "unexpected error: {err}"
        );

        let i = file.dataset("i").unwrap();
        assert_eq!(i.read_numeric_as::<i64>().unwrap(), vec![-1, 5]);
        let err = i.read_numeric_as::<u32>().unwrap_err();
        assert!(
            err.to_string()
                .contains("value -1 at element 0 does not fit in u32"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Boundary: f32 → f64 is exact widening; f64 → f32 is rejected.
    #[test]
    fn numeric_float_widening_only() {
        let path = temp_path("numeric_float");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f32>().shape([2]).create("f4").unwrap();
            ds.write_raw(&[1.5f32, -2.25]).unwrap();
            let ds = file.new_dataset::<f64>().shape([1]).create("f8").unwrap();
            ds.write_raw(&[3.75f64]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();

        let f4 = file.dataset("f4").unwrap();
        assert_eq!(f4.read_numeric_as::<f32>().unwrap(), vec![1.5, -2.25]);
        assert_eq!(f4.read_numeric_as::<f64>().unwrap(), vec![1.5, -2.25]);

        let f8 = file.dataset("f8").unwrap();
        assert_eq!(f8.read_numeric_as::<f64>().unwrap(), vec![3.75]);
        let err = f8.read_numeric_as::<f32>().unwrap_err();
        assert!(
            err.to_string().contains("narrowing"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Boundary: cross-class conversions (float ↔ integer) are rejected in
    /// both directions, and a non-numeric datatype is rejected at classify.
    #[test]
    fn numeric_cross_class_and_non_numeric_rejected() {
        let path = temp_path("numeric_cross_class");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f64>().shape([1]).create("f8").unwrap();
            ds.write_raw(&[1.0f64]).unwrap();
            let ds = file.new_dataset::<i32>().shape([1]).create("i4").unwrap();
            ds.write_raw(&[7i32]).unwrap();
            file.write_vlen_strings("s", &["a", "b"]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();

        let err = file
            .dataset("f8")
            .unwrap()
            .read_numeric_as::<i64>()
            .unwrap_err();
        assert!(
            err.to_string().contains("floating-point dataset as i64"),
            "unexpected error: {err}"
        );
        let err = file
            .dataset("i4")
            .unwrap()
            .read_numeric_as::<f64>()
            .unwrap_err();
        assert!(
            err.to_string().contains("integer dataset as f64"),
            "unexpected error: {err}"
        );
        let err = file
            .dataset("s")
            .unwrap()
            .read_numeric_as::<i64>()
            .unwrap_err();
        assert!(
            err.to_string().contains("is not numeric"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Boundary: big-endian sources decode per the datatype's byte order.
    /// Unit-level (the writer only emits little-endian): feed `convert` a
    /// big-endian datatype plus big-endian bytes directly.
    #[test]
    fn numeric_big_endian_decode() {
        use super::numeric;
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};

        let dt = DatatypeMessage::FixedPoint {
            size: 4,
            byte_order: ByteOrder::BigEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let mut raw = Vec::new();
        raw.extend_from_slice(&(-2i32).to_be_bytes());
        raw.extend_from_slice(&(100_000i32).to_be_bytes());
        let kind = numeric::classify(&dt).unwrap();
        assert_eq!(
            numeric::convert::<i64>(kind, &raw).unwrap(),
            vec![-2, 100_000]
        );

        let dt = DatatypeMessage::FloatingPoint {
            size: 8,
            byte_order: ByteOrder::BigEndian,
            sign_location: 63,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        let raw = (-2.25f64).to_be_bytes();
        let kind = numeric::classify(&dt).unwrap();
        assert_eq!(numeric::convert::<f64>(kind, &raw).unwrap(), vec![-2.25]);
    }

    /// `H5Attribute::read_numeric` validates the stored datatype before
    /// reinterpreting bytes: cross-width, cross-class, and non-numeric
    /// attributes error instead of returning bit-garbage, while the exact
    /// type and the HBool / complex-compound paths keep working.
    #[test]
    fn attr_read_numeric_validates_datatype() {
        use crate::types::{Complex64, HBool, VarLenUnicode};
        let path = temp_path("attr_read_numeric_validate");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f32>().shape([2]).create("d").unwrap();
            ds.write_raw(&[1.0f32; 2]).unwrap();
            let a = ds.new_attr::<f64>().shape(()).create("f8").unwrap();
            a.write_numeric(&1.5f64).unwrap();
            let a = ds.new_attr::<i32>().shape(()).create("i4").unwrap();
            a.write_numeric(&-7i32).unwrap();
            let a = ds.new_attr::<HBool>().shape(()).create("b").unwrap();
            a.write_numeric(&HBool::from(true)).unwrap();
            let a = ds.new_attr::<Complex64>().shape(()).create("z").unwrap();
            a.write_numeric(&Complex64 { re: 1.0, im: -2.0 }).unwrap();
            let a = ds
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("s")
                .unwrap();
            a.write_scalar(&VarLenUnicode("text".into())).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();

        let f8 = ds.attr("f8").unwrap();
        assert_eq!(f8.read_numeric::<f64>().unwrap(), 1.5);
        // Previously returned the low half of the f64 image as an f32.
        let err = f8.read_numeric::<f32>().unwrap_err();
        assert!(
            err.to_string().contains("read_numeric_as"),
            "unexpected error: {err}"
        );
        assert!(f8.read_numeric::<i64>().is_err());

        let i4 = ds.attr("i4").unwrap();
        assert_eq!(i4.read_numeric::<i32>().unwrap(), -7);
        assert!(i4.read_numeric::<u32>().is_err());

        assert!(bool::from(
            ds.attr("b").unwrap().read_numeric::<HBool>().unwrap()
        ));
        let z = ds.attr("z").unwrap().read_numeric::<Complex64>().unwrap();
        assert_eq!((z.re, z.im), (1.0, -2.0));

        // A vlen string attribute: read_numeric used to transmute the heap
        // reference bytes into the requested type.
        let s = ds.attr("s").unwrap();
        assert!(s.read_numeric::<f64>().is_err());
        assert!(s.read_numeric_as::<f64>().is_err());
        std::fs::remove_file(&path).ok();
    }

    /// The attribute conversion read applies the dataset rules: checked
    /// int → int naming index and value on overflow, widening-only floats,
    /// cross-class rejected; an array attribute converts every element.
    #[test]
    fn attr_read_numeric_as_converts() {
        let path = temp_path("attr_read_numeric_as");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f32>().shape([2]).create("d").unwrap();
            ds.write_raw(&[1.0f32; 2]).unwrap();
            let a = ds.new_attr::<i32>().shape(()).create("i4").unwrap();
            a.write_numeric(&-7i32).unwrap();
            let a = ds.new_attr::<u64>().shape(()).create("u8max").unwrap();
            a.write_numeric(&u64::MAX).unwrap();
            let a = ds.new_attr::<i16>().shape([3]).create("arr").unwrap();
            a.write_array(&[1i16, -2, 3]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        assert_eq!(
            ds.attr("i4").unwrap().read_numeric_as::<i64>().unwrap(),
            vec![-7]
        );
        let err = ds
            .attr("u8max")
            .unwrap()
            .read_numeric_as::<i64>()
            .unwrap_err();
        assert!(
            err.to_string().contains("does not fit in i64"),
            "unexpected error: {err}"
        );
        assert_eq!(
            ds.attr("arr").unwrap().read_numeric_as::<i32>().unwrap(),
            vec![1, -2, 3]
        );
        assert!(ds.attr("i4").unwrap().read_numeric_as::<f64>().is_err());
        std::fs::remove_file(&path).ok();
    }

    /// The hyperslab variant applies the same conversion to a sub-selection.
    #[test]
    fn numeric_slice_conversion() {
        let path = temp_path("numeric_slice");
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<i16>().shape([2, 3]).create("m").unwrap();
            ds.write_raw(&[1i16, 2, 3, 4, 5, 6]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let m = file.dataset("m").unwrap();
        assert_eq!(
            m.read_numeric_slice_as::<i32>(&[0, 1], &[2, 2]).unwrap(),
            vec![2, 3, 5, 6]
        );
        std::fs::remove_file(&path).ok();
    }

    /// A NULL dataspace round-trips through the public API as `is_null() ==
    /// true`, `shape() == []`, and `read_raw_bytes()` empty — and stays
    /// distinguishable from a scalar dataset, which shares the same empty
    /// `shape()` but holds exactly one element.
    #[test]
    fn null_dataspace_distinct_from_scalar() {
        let path = temp_path("null_vs_scalar");
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<i32>().null().create("empty").unwrap();
            let scalar = file.new_dataset::<i32>().scalar().create("scalar").unwrap();
            scalar.write_raw(&[42i32]).unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();

        let empty = file.dataset("empty").unwrap();
        assert!(empty.is_null());
        assert_eq!(empty.shape(), Vec::<usize>::new());
        assert_eq!(empty.total_elements(), 0);
        assert_eq!(empty.read_raw_bytes().unwrap(), Vec::<u8>::new());

        let scalar = file.dataset("scalar").unwrap();
        assert!(!scalar.is_null());
        assert_eq!(scalar.shape(), Vec::<usize>::new());
        assert_eq!(scalar.total_elements(), 1);
        assert_eq!(scalar.read_raw::<i32>().unwrap(), vec![42]);

        std::fs::remove_file(&path).ok();
    }

    /// A NULL dataspace dataset rejects writes outright — there is nothing
    /// to write into — rather than silently accepting a scalar-shaped
    /// write against unallocated storage.
    #[test]
    fn null_dataspace_rejects_writes() {
        let path = temp_path("null_write_rejected");
        let file = H5File::create(&path).unwrap();
        let ds = file.new_dataset::<i32>().null().create("empty").unwrap();
        assert!(ds.write_raw(&[1i32]).is_err());
        assert!(ds.write_raw_bytes(&[0u8; 4]).is_err());
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// `.null()` combined with `.chunk()` or a fill value is rejected at
    /// `create()` rather than silently dropping the conflicting option —
    /// a NULL dataspace can never be chunked or filtered upstream.
    #[test]
    fn null_dataspace_rejects_chunking_and_fill_value() {
        let path = temp_path("null_chunk_rejected");
        let file = H5File::create(&path).unwrap();
        assert!(file
            .new_dataset::<i32>()
            .null()
            .chunk(&[4])
            .create("a")
            .is_err());
        assert!(file
            .new_dataset::<i32>()
            .null()
            .fill_value(7i32)
            .create("b")
            .is_err());
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// The layout class the reader sees for `name`, plus the image a compact
    /// layout carries — what distinguishes compact storage from contiguous
    /// storage that happens to hold the same bytes.
    fn compact_image(path: &std::path::Path, name: &str) -> Option<Vec<u8>> {
        use crate::format::messages::data_layout::DataLayoutMessage;
        let mut reader = crate::io::reader::Hdf5Reader::open(path).unwrap();
        match &reader.dataset_info(name).unwrap().layout {
            DataLayoutMessage::Compact { data } => Some(data.clone()),
            other => panic!("{name}: expected a compact layout, got {other:?}"),
        }
    }

    /// `.compact()` puts the raw data inside the data layout message: the
    /// dataset has no data block of its own, and the image the layout carries
    /// is what a read returns.
    #[test]
    fn a_compact_dataset_stores_its_data_in_the_layout_message() {
        let path = temp_path("compact_roundtrip");
        let values: Vec<i32> = (0..16).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<i32>()
                .shape([16usize])
                .compact()
                .create("d")
                .unwrap()
                .write_raw(&values)
                .unwrap();
            file.close().unwrap();
        }

        let image = compact_image(&path, "d").unwrap();
        assert_eq!(
            image,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );

        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("d").unwrap();
        assert_eq!(ds.shape(), vec![16]);
        assert_eq!(ds.read_raw::<i32>().unwrap(), values);
        std::fs::remove_file(&path).ok();
    }

    /// A compact dataset created inside a group is linked from that group,
    /// not from the root: its create path goes through the same parent
    /// resolution every other layout uses.
    #[test]
    fn a_compact_dataset_lands_in_its_group() {
        let path = temp_path("compact_in_group");
        {
            let file = H5File::create(&path).unwrap();
            let g = file.root_group().create_group("g").unwrap();
            g.new_dataset::<u8>()
                .shape([4usize])
                .compact()
                .create("d")
                .unwrap()
                .write_raw(&[1u8, 2, 3, 4])
                .unwrap();
            file.close().unwrap();
        }
        assert_eq!(compact_image(&path, "/g/d").unwrap(), vec![1u8, 2, 3, 4]);

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("/g/d").unwrap().read_raw::<u8>().unwrap(),
            vec![1u8, 2, 3, 4]
        );
        std::fs::remove_file(&path).ok();
    }

    /// A compact dataset's storage is the image itself, so the fill value has
    /// to be tiled into it at create — `H5D__compact_fill`'s job. An unwritten
    /// element must read back as the fill value, not as zero.
    #[test]
    fn an_unwritten_compact_dataset_reads_back_as_its_fill_value() {
        let path = temp_path("compact_fill");
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<i32>()
                .shape([4usize])
                .compact()
                .fill_value(-7i32)
                .create("d")
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            vec![-7i32; 4]
        );
        std::fs::remove_file(&path).ok();
    }

    /// The image is the layout message, so anything that rewrites the header
    /// rewrites the data with it. Reopening and attaching an attribute makes
    /// the header stale; the rebuilt one must still carry the image rather
    /// than fall back to an unallocated contiguous layout.
    #[test]
    fn a_reopened_compact_dataset_keeps_its_image() {
        let path = temp_path("compact_reopen");
        let values: Vec<i32> = (100..108).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<i32>()
                .shape([8usize])
                .compact()
                .create("d")
                .unwrap()
                .write_raw(&values)
                .unwrap();
            file.close().unwrap();
        }
        {
            let file = H5File::open_rw(&path).unwrap();
            file.dataset_writer("d")
                .unwrap()
                .new_attr::<i32>()
                .shape(())
                .create("note")
                .unwrap()
                .write_numeric(&1i32)
                .unwrap();
            file.close().unwrap();
        }
        assert_eq!(
            compact_image(&path, "d").unwrap(),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            values
        );
        std::fs::remove_file(&path).ok();
    }

    /// The ceiling is what a data layout message can hold, so it is checked
    /// in bytes and names them: the largest image that fits is accepted and
    /// one element more is refused.
    #[test]
    fn the_compact_ceiling_is_checked_in_bytes() {
        let path = temp_path("compact_ceiling");
        let file = H5File::create(&path).unwrap();

        let fits = crate::MAX_COMPACT_DATA / 4;
        file.new_dataset::<i32>()
            .shape([fits])
            .compact()
            .create("fits")
            .unwrap();

        let over = fits + 1;
        let err = match file
            .new_dataset::<i32>()
            .shape([over])
            .compact()
            .create("over")
        {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an image {} bytes wide must be refused", over * 4),
        };
        assert!(
            err.contains(&(over * 4).to_string())
                && err.contains(&crate::MAX_COMPACT_DATA.to_string()),
            "the error must name both sizes: {err}"
        );

        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Compact storage has no chunk grid to filter and no room to grow, so
    /// each conflicting option is refused at `create()` rather than silently
    /// overriding the layout the way `H5Pset_chunk` does.
    #[test]
    fn compact_rejects_chunking_filters_growth_and_a_null_dataspace() {
        let path = temp_path("compact_rejects");
        let file = H5File::create(&path).unwrap();
        assert!(file
            .new_dataset::<i32>()
            .shape([4usize])
            .compact()
            .chunk(&[4])
            .create("a")
            .is_err());
        assert!(file
            .new_dataset::<i32>()
            .shape([4usize])
            .compact()
            .deflate(4)
            .create("b")
            .is_err());
        assert!(file
            .new_dataset::<i32>()
            .shape([4usize])
            .compact()
            .max_shape(&[None])
            .create("c")
            .is_err());
        assert!(file
            .new_dataset::<i32>()
            .shape([4usize])
            .compact()
            .max_shape(&[Some(8)])
            .create("d")
            .is_err());
        assert!(file
            .new_dataset::<i32>()
            .null()
            .compact()
            .create("e")
            .is_err());
        file.close().unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// The filter pipeline the reader decodes from `name`'s header.
    fn stored_pipeline(
        path: &std::path::Path,
        name: &str,
    ) -> crate::format::messages::filter::FilterPipeline {
        let mut reader = crate::io::reader::Hdf5Reader::open(path).unwrap();
        reader
            .dataset_info(name)
            .unwrap()
            .filter_pipeline
            .clone()
            .unwrap_or_else(|| panic!("{name}: no filter pipeline"))
    }

    /// Shuffle is a permutation, not a compressor, so it is a pipeline on its
    /// own — `H5Pset_shuffle` with nothing behind it. Its stage must reach the
    /// header, and the reader must unpermute what it wrote.
    #[test]
    fn shuffle_alone_is_a_filter_pipeline() {
        use crate::format::messages::filter::{FilterPipeline, FILTER_SHUFFLE};
        let path = temp_path("shuffle_alone");
        let values: Vec<i32> = (0..64).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<i32>()
                .shape([64usize])
                .chunk(&[16])
                .shuffle()
                .create("d")
                .unwrap()
                .write_raw(&values)
                .unwrap();
            file.close().unwrap();
        }
        let pipeline = stored_pipeline(&path, "d");
        assert_eq!(pipeline, FilterPipeline::shuffle(4));
        assert_eq!(pipeline.filters[0].id, FILTER_SHUFFLE);

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            values
        );
        std::fs::remove_file(&path).ok();
    }

    /// `.shuffle()` and `.deflate()` are separate stages that compose, and
    /// `.shuffle_deflate()` is the shorthand for both — one pipeline, built
    /// once, whichever way it was asked for.
    #[cfg(feature = "deflate")]
    #[test]
    fn shuffle_composes_with_deflate() {
        let path = temp_path("shuffle_then_deflate");
        let values: Vec<i32> = (0..64).collect();
        {
            let file = H5File::create(&path).unwrap();
            for (name, ds) in [
                ("split", file.new_dataset::<i32>().shuffle().deflate(6)),
                ("combined", file.new_dataset::<i32>().shuffle_deflate(6)),
            ] {
                ds.shape([64usize])
                    .chunk(&[16])
                    .create(name)
                    .unwrap()
                    .write_raw(&values)
                    .unwrap();
            }
            file.close().unwrap();
        }
        assert_eq!(
            stored_pipeline(&path, "split"),
            crate::format::messages::filter::FilterPipeline::shuffle_deflate(4, 6)
        );
        assert_eq!(
            stored_pipeline(&path, "split"),
            stored_pipeline(&path, "combined")
        );

        let file = H5File::open(&path).unwrap();
        for name in ["split", "combined"] {
            assert_eq!(
                file.dataset(name).unwrap().read_raw::<i32>().unwrap(),
                values,
                "{name}"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    /// The width shuffle permutes by is the stored element's, which a
    /// `datatype` override moves away from the carrier type `T`: recording
    /// `T`'s width would permute a 4-byte element as four 1-byte ones and
    /// hand libhdf5 a chunk it unshuffles into different bytes.
    #[test]
    fn shuffle_records_the_stored_element_width() {
        use crate::format::messages::datatype::DatatypeMessage;
        use crate::format::messages::filter::FilterPipeline;
        let path = temp_path("shuffle_override_width");
        let bytes: Vec<u8> = (0..16u8).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<u8>()
                .datatype(DatatypeMessage::i32_type())
                .shape([4usize])
                .chunk(&[4])
                .shuffle()
                .create("d")
                .unwrap()
                .write_raw_bytes(&bytes)
                .unwrap();
            file.close().unwrap();
        }
        assert_eq!(stored_pipeline(&path, "d"), FilterPipeline::shuffle(4));

        let file = H5File::open(&path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            bytes
                .chunks(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        );
        std::fs::remove_file(&path).ok();
    }
}
