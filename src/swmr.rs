//! Single Writer / Multiple Reader (SWMR) API.
//!
//! Provides a high-level wrapper around the SWMR protocol for streaming
//! frame-based data (e.g., area detector images).

use std::path::Path;

use crate::format::messages::attribute::AttributeMessage;
use crate::io::locking::FileLocking;
use crate::io::Hdf5Reader;
use crate::io::SwmrWriter as IoSwmrWriter;

use crate::error::Result;
use crate::types::H5Type;

/// SWMR writer for streaming frame-based data to an HDF5 file.
///
/// Usage:
/// ```no_run
/// use rust_hdf5::swmr::SwmrFileWriter;
///
/// let mut writer = SwmrFileWriter::create("stream.h5").unwrap();
/// let ds = writer.create_streaming_dataset::<f32>("frames", &[256, 256]).unwrap();
/// writer.start_swmr().unwrap();
///
/// // Write frames
/// let frame_data = vec![0.0f32; 256 * 256];
/// let raw: Vec<u8> = frame_data.iter()
///     .flat_map(|v| v.to_le_bytes())
///     .collect();
/// writer.append_frame(ds, &raw).unwrap();
/// writer.flush().unwrap();
///
/// writer.close().unwrap();
/// ```
pub struct SwmrFileWriter {
    inner: IoSwmrWriter,
}

impl SwmrFileWriter {
    /// Create a new HDF5 file for SWMR streaming using the env-var-derived
    /// locking policy.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = IoSwmrWriter::create(path.as_ref())?;
        Ok(Self { inner })
    }

    /// Create a new HDF5 file for SWMR streaming with an explicit locking
    /// policy. The writer holds an exclusive lock until [`Self::start_swmr`]
    /// is called, at which point the lock is downgraded to shared so
    /// concurrent SWMR readers can attach.
    pub fn create_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self> {
        let inner = IoSwmrWriter::create_with_locking(path.as_ref(), locking)?;
        Ok(Self { inner })
    }

    /// Reopen a cleanly-closed HDF5 file to resume SWMR streaming.
    ///
    /// Existing datasets are reconstructed; locate them with
    /// [`dataset_index`](Self::dataset_index), call [`start_swmr`](Self::start_swmr)
    /// to re-enter SWMR mode, then continue with [`append_frame`](Self::append_frame).
    /// Appending to a multi-frame-chunk dataset (`chunk[0] > 1`) after reopen
    /// is rejected — its final partial band was zero-padded at the original
    /// close. Recovering a crashed (never cleanly closed) file is not supported.
    pub fn open_append<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = IoSwmrWriter::open_append(path.as_ref())?;
        Ok(Self { inner })
    }

    /// Reopen a cleanly-closed HDF5 file to resume SWMR streaming with an
    /// explicit locking policy. See [`Self::open_append`].
    pub fn open_append_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self> {
        let inner = IoSwmrWriter::open_append_with_locking(path.as_ref(), locking)?;
        Ok(Self { inner })
    }

    /// Return the index of a dataset by name, or `None` if absent.
    ///
    /// Mainly used after [`open_append`](Self::open_append) to recover the
    /// index of a reconstructed dataset for [`append_frame`](Self::append_frame).
    pub fn dataset_index(&self, name: &str) -> Option<usize> {
        self.inner.dataset_index(name)
    }

    /// Create a streaming dataset.
    ///
    /// The dataset will have shape `[0, frame_dims...]` initially, with
    /// chunk dimensions `[1, frame_dims...]` and unlimited first dimension.
    ///
    /// Returns the dataset index for use with `append_frame`.
    pub fn create_streaming_dataset<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
    ) -> Result<usize> {
        let datatype = T::hdf5_type();
        let idx = self
            .inner
            .create_streaming_dataset(name, datatype, frame_dims)?;
        Ok(idx)
    }

    /// Create a streaming dataset whose frames are compressed.
    ///
    /// Like [`create_streaming_dataset`](Self::create_streaming_dataset) but
    /// each appended frame is run through `pipeline` (e.g.
    /// `FilterPipeline::deflate(4)`). SWMR appends and in-place header
    /// updates work the same as for uncompressed streaming datasets.
    pub fn create_streaming_dataset_compressed<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> Result<usize> {
        let idx = self.inner.create_streaming_dataset_compressed(
            name,
            T::hdf5_type(),
            frame_dims,
            pipeline,
        )?;
        Ok(idx)
    }

    /// Create a streaming dataset whose frames are split into fixed-size
    /// chunk tiles.
    ///
    /// `frame_dims` is the per-frame shape (e.g. `[1024, 1024]`);
    /// `frame_chunk` is the tile shape within a frame (e.g. `[256, 256]`),
    /// of the same rank. The on-disk chunk shape becomes
    /// `[1, frame_chunk...]`, so each frame is stored as
    /// `product(frame_dims / frame_chunk)` chunks instead of one. This
    /// mirrors area-detector tiling controls such as NDFileHDF5's
    /// `nRowChunks` / `nColChunks`: it changes only the partial-read
    /// granularity and compression unit, not the stored data.
    /// [`append_frame`](Self::append_frame) accepts a whole frame and
    /// splits it into tiles automatically.
    pub fn create_streaming_dataset_tiled<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
        frame_chunk: &[u64],
    ) -> Result<usize> {
        let idx = self.inner.create_streaming_dataset_tiled(
            name,
            T::hdf5_type(),
            frame_dims,
            frame_chunk,
        )?;
        Ok(idx)
    }

    /// Create a compressed streaming dataset whose frames are split into
    /// fixed-size chunk tiles. See
    /// [`create_streaming_dataset_tiled`](Self::create_streaming_dataset_tiled)
    /// for the meaning of `frame_chunk`; each tile is the compression unit.
    pub fn create_streaming_dataset_tiled_compressed<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
        frame_chunk: &[u64],
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> Result<usize> {
        let idx = self.inner.create_streaming_dataset_tiled_compressed(
            name,
            T::hdf5_type(),
            frame_dims,
            frame_chunk,
            pipeline,
        )?;
        Ok(idx)
    }

    /// Create a streaming dataset with full control over the chunk shape,
    /// including the frame axis.
    ///
    /// `chunk` is the complete per-chunk shape, of rank
    /// `frame_dims.len() + 1`: `chunk[0]` frames per chunk (the NDFileHDF5
    /// `nFramesChunks` control) and `chunk[1..]` the per-frame tile shape
    /// (`nRowChunks` / `nColChunks`). When `chunk[0] > 1`,
    /// [`append_frame`](Self::append_frame) buffers whole frames until a
    /// chunk band fills; the final partial band is written (zero-padded) at
    /// [`close`](Self::close), and the dataset's logical frame count always
    /// equals the exact number of frames appended.
    pub fn create_streaming_dataset_chunked<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
        chunk: &[u64],
    ) -> Result<usize> {
        let idx =
            self.inner
                .create_streaming_dataset_chunked(name, T::hdf5_type(), frame_dims, chunk)?;
        Ok(idx)
    }

    /// Compressed variant of
    /// [`create_streaming_dataset_chunked`](Self::create_streaming_dataset_chunked);
    /// each chunk is filtered independently through `pipeline`.
    pub fn create_streaming_dataset_chunked_compressed<T: H5Type>(
        &mut self,
        name: &str,
        frame_dims: &[u64],
        chunk: &[u64],
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> Result<usize> {
        let idx = self.inner.create_streaming_dataset_chunked_compressed(
            name,
            T::hdf5_type(),
            frame_dims,
            chunk,
            pipeline,
        )?;
        Ok(idx)
    }

    /// Create a fixed-shape multi-dimensional grid dataset that fills at
    /// explicit positions as frames arrive.
    ///
    /// Unlike [`create_streaming_dataset`](Self::create_streaming_dataset),
    /// which appends frames along a single unlimited leading axis, this
    /// creates a dataset of the full bounded shape `dims` (no unlimited axis)
    /// and lets you place each frame at an arbitrary chunk position with
    /// [`write_chunk_at`](Self::write_chunk_at). This mirrors AreaDetector's
    /// "extra dimensions" layout, where a scan of known size (e.g.
    /// `[Na, Nb, H, W]`) is filled in odometer order. `chunk` is the per-chunk
    /// shape of the same rank (typically `[1, …, 1, H, W]`). Returns the
    /// dataset index.
    pub fn create_grid_dataset<T: H5Type>(
        &mut self,
        name: &str,
        dims: &[u64],
        chunk: &[u64],
    ) -> Result<usize> {
        let idx = self
            .inner
            .create_grid_dataset(name, T::hdf5_type(), dims, chunk)?;
        Ok(idx)
    }

    /// Create a hard link: an additional name for a dataset or group that
    /// already exists in the file.
    ///
    /// No data is copied — the link and its target share one object header,
    /// exactly as `h5py` / libhdf5 hard links do. This is the NeXus-style way
    /// to expose a streaming dataset at an aliased path.
    ///
    /// * `parent_group_path` — full path of the group that will hold the
    ///   link (`"/"` for the root group).
    /// * `link_name` — leaf name of the new link within that group.
    /// * `target_path` — full path of an existing dataset or group.
    ///
    /// # Visibility relative to SWMR mode
    ///
    /// A link created **before** [`start_swmr`](Self::start_swmr) is committed
    /// by `start_swmr` and is visible to SWMR readers for the whole streaming
    /// window. A link created **after** `start_swmr` is committed only by
    /// [`close`](Self::close); it does not appear to readers that attach
    /// during the live SWMR window. Create layout links before `start_swmr`
    /// when readers must resolve them while streaming.
    pub fn create_hard_link(
        &mut self,
        parent_group_path: &str,
        link_name: &str,
        target_path: &str,
    ) -> Result<()> {
        self.inner
            .writer_mut()
            .create_hard_link(parent_group_path, link_name, target_path)?;
        Ok(())
    }

    /// Create a group in the file hierarchy.
    ///
    /// * `parent_group_path` — full path of the parent group (`"/"` for the
    ///   root group).
    /// * `name` — leaf name of the new group.
    ///
    /// A nested NeXus layout is built one level at a time, parent first:
    ///
    /// ```no_run
    /// # use rust_hdf5::swmr::SwmrFileWriter;
    /// # let mut writer = SwmrFileWriter::create("stream.h5").unwrap();
    /// writer.create_group("/", "entry").unwrap();
    /// writer.create_group("/entry", "data").unwrap();
    /// ```
    ///
    /// Like [`create_hard_link`](Self::create_hard_link), a group created
    /// before [`start_swmr`](Self::start_swmr) is visible to SWMR readers for
    /// the whole streaming window; one created after is committed only by
    /// [`close`](Self::close).
    pub fn create_group(&mut self, parent_group_path: &str, name: &str) -> Result<()> {
        self.inner
            .writer_mut()
            .create_group(parent_group_path, name)?;
        Ok(())
    }

    /// Set a string attribute on a group, or on the root group when
    /// `group_path` is `"/"`.
    ///
    /// This is the NeXus way to tag a group with its class — for example
    /// `set_group_attr_string("/entry", "NX_class", "NXentry")`. An existing
    /// attribute of the same name is replaced.
    ///
    /// Attributes must be set before [`start_swmr`](Self::start_swmr):
    /// object headers are frozen while readers stream, so every attribute
    /// setter is refused once SWMR is active — libhdf5's rule for SWMR
    /// writes too.
    pub fn set_group_attr_string(
        &mut self,
        group_path: &str,
        name: &str,
        value: &str,
    ) -> Result<()> {
        self.inner.writer_mut().set_vlen_string_attribute(
            group_attr_target(group_path),
            name,
            value,
        )?;
        Ok(())
    }

    /// Set a numeric scalar attribute on a group, or on the root group when
    /// `group_path` is `"/"`. An existing attribute of the same name is
    /// replaced. Refused after [`start_swmr`](Self::start_swmr) — see
    /// [`set_group_attr_string`](Self::set_group_attr_string).
    pub fn set_group_attr_numeric<T: H5Type>(
        &mut self,
        group_path: &str,
        name: &str,
        value: &T,
    ) -> Result<()> {
        let attr = AttributeMessage::scalar_numeric(name, T::hdf5_type(), scalar_to_bytes(value));
        self.inner
            .writer_mut()
            .set_attribute(group_attr_target(group_path), attr)?;
        Ok(())
    }

    /// Create a fixed-shape (non-streaming) dataset and write all its data in
    /// one call. Returns the dataset index.
    ///
    /// This is for the NeXus metadata that surrounds the image stream —
    /// coordinate axes, detector geometry, and (with `dims = &[]`) scalar
    /// values such as `/entry/instrument/detector/distance`. Unlike a
    /// streaming dataset, it is written once and not appended to.
    pub fn write_dataset<T: H5Type>(
        &mut self,
        name: &str,
        dims: &[u64],
        data: &[T],
    ) -> Result<usize> {
        let expected: u64 = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };
        if data.len() as u64 != expected {
            return Err(crate::error::Hdf5Error::InvalidState(format!(
                "write_dataset: data has {} elements but shape {dims:?} needs {expected}",
                data.len()
            )));
        }
        let idx = self
            .inner
            .writer_mut()
            .create_dataset(name, T::hdf5_type(), dims)?;
        self.inner
            .writer_mut()
            .write_dataset_raw(idx, slice_to_bytes(data))?;
        Ok(idx)
    }

    /// Create a variable-length string dataset (one element per string).
    /// Returns the dataset index. Useful for NeXus metadata such as
    /// `/entry/start_time` or per-frame timestamp arrays.
    ///
    /// The datatype declares UTF-8;
    /// [`write_string_dataset_ascii`](Self::write_string_dataset_ascii)
    /// declares ASCII instead.
    pub fn write_string_dataset(&mut self, name: &str, strings: &[&str]) -> Result<usize> {
        let idx = self
            .inner
            .writer_mut()
            .create_vlen_string_dataset(name, strings, 1)?;
        Ok(idx)
    }

    /// [`write_string_dataset`](Self::write_string_dataset) under an **ASCII**
    /// datatype, the one h5py's `string_dtype("ascii")` produces. A string
    /// that is not 7-bit is rejected rather than mislabelled.
    pub fn write_string_dataset_ascii(&mut self, name: &str, strings: &[&str]) -> Result<usize> {
        let idx = self
            .inner
            .writer_mut()
            .create_vlen_string_dataset(name, strings, 0)?;
        Ok(idx)
    }

    /// Set a string attribute on a dataset, addressed by its index. The
    /// NeXus way to record `units`, `long_name`, `signal`, etc. An existing
    /// attribute of the same name is replaced. Refused after
    /// [`start_swmr`](Self::start_swmr) — see
    /// [`set_group_attr_string`](Self::set_group_attr_string).
    pub fn set_dataset_attr_string(
        &mut self,
        ds_index: usize,
        name: &str,
        value: &str,
    ) -> Result<()> {
        self.inner.writer_mut().set_vlen_string_attribute(
            crate::io::writer::AttrTarget::Dataset(ds_index),
            name,
            value,
        )?;
        Ok(())
    }

    /// Set a numeric scalar attribute on a dataset, addressed by its index.
    /// An existing attribute of the same name is replaced. Refused after
    /// [`start_swmr`](Self::start_swmr) — see
    /// [`set_group_attr_string`](Self::set_group_attr_string).
    pub fn set_dataset_attr_numeric<T: H5Type>(
        &mut self,
        ds_index: usize,
        name: &str,
        value: &T,
    ) -> Result<()> {
        let attr = AttributeMessage::scalar_numeric(name, T::hdf5_type(), scalar_to_bytes(value));
        self.inner
            .writer_mut()
            .add_dataset_attribute(ds_index, attr)?;
        Ok(())
    }

    /// Set a numeric array attribute on a dataset, addressed by its index.
    ///
    /// `dims` are the dimension sizes (e.g. `&[3]` for a 1-D array) and the
    /// number of `values` must equal their product. This is the SWMR
    /// counterpart of [`H5Attribute::write_array`](crate::H5Attribute::write_array),
    /// for the length-ndims `int32` array attributes AreaDetector writes
    /// (`NDArrayDimOffset`, `NDArrayDimBinning`, `NDArrayDimReverse`). An
    /// existing attribute of the same name is replaced.
    ///
    /// In SWMR mode, attributes can only be added before
    /// [`start_swmr`](Self::start_swmr); HDF5 forbids adding dataset
    /// attributes once the file is in SWMR write mode. Resolve a dataset path
    /// to its index with [`dataset_index`](Self::dataset_index).
    pub fn set_dataset_attr_array<T: H5Type>(
        &mut self,
        ds_index: usize,
        name: &str,
        dims: &[u64],
        values: &[T],
    ) -> Result<()> {
        // Product of an empty shape is 1 (a scalar holds one element).
        let expected: u64 = dims.iter().product();
        if values.len() as u64 != expected {
            return Err(crate::error::Hdf5Error::InvalidState(format!(
                "set_dataset_attr_array: {} values but shape {dims:?} needs {expected}",
                values.len()
            )));
        }
        let attr = AttributeMessage::array_numeric(
            name,
            T::hdf5_type(),
            dims,
            slice_to_bytes(values).to_vec(),
        );
        self.inner
            .writer_mut()
            .add_dataset_attribute(ds_index, attr)?;
        Ok(())
    }

    /// Set the fill value of a streaming dataset, addressed by its index.
    ///
    /// Call this before the first [`append_frame`](Self::append_frame): it
    /// determines the value of chunk regions that are never written (a
    /// partial final band, or unwritten tiles).
    pub fn set_dataset_fill_value<T: H5Type>(&mut self, ds_index: usize, value: &T) -> Result<()> {
        self.inner
            .writer_mut()
            .set_dataset_fill_value(ds_index, scalar_to_bytes(value))?;
        Ok(())
    }

    /// Place an existing dataset inside a group.
    ///
    /// By default a dataset created through this writer lives at the root
    /// level; this moves its link record into `group_path` (which must
    /// already exist). The group must be created before `start_swmr` for the
    /// placement to be visible to readers during streaming.
    pub fn assign_dataset_to_group(&mut self, group_path: &str, ds_index: usize) -> Result<()> {
        self.inner
            .writer_mut()
            .assign_dataset_to_group(group_path, ds_index)?;
        Ok(())
    }

    /// Signal the start of SWMR mode.
    pub fn start_swmr(&mut self) -> Result<()> {
        self.inner.start_swmr()?;
        Ok(())
    }

    /// Append a frame of raw data to a streaming dataset.
    ///
    /// The data size must match one frame (product of frame_dims * element_size).
    pub fn append_frame(&mut self, ds_index: usize, data: &[u8]) -> Result<()> {
        self.inner.append_frame(ds_index, data)?;
        Ok(())
    }

    /// Write one frame at an explicit chunk position of a grid dataset
    /// created with [`create_grid_dataset`](Self::create_grid_dataset).
    ///
    /// `chunk_coords` are in units of chunks (row-major over the chunk grid)
    /// and `data` must be exactly one full chunk (`product(chunk) *
    /// element_size` bytes; edge chunks are zero-padded by the caller). The
    /// logical extent is fixed, so positions may be written in any order and
    /// unwritten positions read back as fill. As with the streaming path,
    /// call [`flush`](Self::flush) to make writes visible to SWMR readers and
    /// set dataset attributes before [`start_swmr`](Self::start_swmr).
    pub fn write_chunk_at(
        &mut self,
        ds_index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> Result<()> {
        self.inner.write_chunk_at(ds_index, chunk_coords, data)?;
        Ok(())
    }

    /// Flush all dataset index structures to disk with SWMR ordering.
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }

    /// Close and finalize the file.
    pub fn close(self) -> Result<()> {
        self.inner.close()?;
        Ok(())
    }
}

/// SWMR reader for monitoring a streaming HDF5 file.
///
/// Opens a file being written by a concurrent [`SwmrFileWriter`] and
/// periodically calls [`refresh`](Self::refresh) to pick up new data.
///
/// ```no_run
/// use rust_hdf5::swmr::SwmrFileReader;
///
/// let mut reader = SwmrFileReader::open("stream.h5").unwrap();
///
/// loop {
///     reader.refresh().unwrap();
///     let names = reader.dataset_names();
///     if let Some(shape) = reader.dataset_shape("frames").ok() {
///         println!("frames shape: {:?}", shape);
///         if shape[0] > 0 {
///             let data = reader.read_dataset_raw("frames").unwrap();
///             println!("got {} bytes", data.len());
///             break;
///         }
///     }
///     std::thread::sleep(std::time::Duration::from_millis(100));
/// }
/// ```
pub struct SwmrFileReader {
    reader: Hdf5Reader,
}

impl SwmrFileReader {
    /// Open an HDF5 file for SWMR reading using the env-var-derived
    /// locking policy. Takes a shared lock so it coexists with the
    /// downgraded shared lock held by [`SwmrFileWriter`] after
    /// `start_swmr`, and with other concurrent SWMR readers.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = Hdf5Reader::open_swmr(path.as_ref())?;
        Ok(Self { reader })
    }

    /// Open an HDF5 file for SWMR reading with an explicit locking policy.
    pub fn open_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self> {
        let reader = Hdf5Reader::open_swmr_with_locking(path.as_ref(), locking)?;
        Ok(Self { reader })
    }

    /// Re-read the superblock and dataset metadata from disk.
    ///
    /// Call this periodically to pick up new data written by the concurrent
    /// SWMR writer.
    pub fn refresh(&mut self) -> Result<()> {
        self.reader.refresh()?;
        Ok(())
    }

    /// Return the names of all datasets.
    pub fn dataset_names(&self) -> Vec<String> {
        self.reader
            .dataset_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Return the current shape of a dataset.
    pub fn dataset_shape(&self, name: &str) -> Result<Vec<u64>> {
        Ok(self.reader.dataset_shape(name)?)
    }

    /// Read the raw bytes of a dataset.
    pub fn read_dataset_raw(&mut self, name: &str) -> Result<Vec<u8>> {
        Ok(self.reader.read_dataset_raw(name)?)
    }

    /// Read a dataset as a typed vector.
    ///
    /// `T` must have the dataset's exact element width — the same
    /// width-checked byte reinterpretation as
    /// [`H5Dataset::read_raw`](crate::dataset::H5Dataset::read_raw), with no
    /// datatype conversion. Use
    /// [`dataset_element_size`](Self::dataset_element_size) to size-check a
    /// type at runtime.
    pub fn read_dataset<T: H5Type>(&mut self, name: &str) -> Result<Vec<T>> {
        self.check_element_width::<T>(name)?;
        bytes_to_typed(self.reader.read_dataset_raw(name)?)
    }

    /// Read a slice (hyperslab) of a dataset as raw bytes.
    ///
    /// `starts[d]` is the first index along dimension `d`, `counts[d]` is how
    /// many. For a streaming dataset this reads only the chunks the slice
    /// overlaps — the efficient way for a live viewer to fetch the latest
    /// frame without re-reading the whole stream.
    pub fn read_slice_raw(
        &mut self,
        name: &str,
        starts: &[u64],
        counts: &[u64],
    ) -> Result<Vec<u8>> {
        Ok(self.reader.read_slice(name, starts, counts)?)
    }

    /// Read a slice (hyperslab) of a dataset as a typed vector.
    /// See [`read_slice_raw`](Self::read_slice_raw); `T`'s width is checked
    /// like [`read_dataset`](Self::read_dataset).
    pub fn read_slice<T: H5Type>(
        &mut self,
        name: &str,
        starts: &[u64],
        counts: &[u64],
    ) -> Result<Vec<T>> {
        self.check_element_width::<T>(name)?;
        bytes_to_typed(self.reader.read_slice(name, starts, counts)?)
    }

    /// The width gate for the typed reads: without it, reinterpreting the
    /// raw bytes splits or merges elements — an f64 dataset read as `i32`
    /// passed the divisibility check and silently returned twice as many
    /// garbage values.
    fn check_element_width<T: H5Type>(&self, name: &str) -> Result<()> {
        let stored = self.dataset_element_size(name)?;
        if T::element_size() != stored {
            return Err(crate::error::Hdf5Error::TypeMismatch(format!(
                "read type has element size {} but dataset has element size {stored}",
                T::element_size(),
            )));
        }
        Ok(())
    }

    /// Read a variable-length string dataset.
    pub fn read_vlen_strings(&mut self, name: &str) -> Result<Vec<String>> {
        Ok(self.reader.read_vlen_strings(name)?)
    }

    /// Element size of a dataset's datatype, in bytes — enough to size a
    /// read buffer without knowing the concrete element type at compile time.
    pub fn dataset_element_size(&self, name: &str) -> Result<usize> {
        self.reader
            .dataset_info(name)
            .map(|i| i.datatype.element_size() as usize)
            .ok_or_else(|| crate::error::Hdf5Error::NotFound(name.to_string()))
    }

    /// All group paths in the file.
    pub fn group_paths(&self) -> Vec<String> {
        self.reader.group_paths().iter().cloned().collect()
    }

    /// Whether a group exists. A leading `/` is tolerated.
    pub fn has_group(&self, group_path: &str) -> bool {
        self.reader.has_group(group_path.trim_start_matches('/'))
    }

    /// Names of the attributes on a dataset.
    pub fn dataset_attr_names(&self, name: &str) -> Result<Vec<String>> {
        Ok(self.reader.dataset_attr_names(name)?)
    }

    /// Read a dataset's attribute as a string (e.g. `units`, `NX_class`).
    pub fn dataset_attr_string(&mut self, dataset: &str, attr: &str) -> Result<String> {
        let a = self.reader.dataset_attr(dataset, attr)?.clone();
        Ok(self.reader.attr_string_value(&a)?)
    }

    /// Names of the attributes on a group, or on the root group when
    /// `group_path` is `"/"`. A leading `/` is tolerated.
    pub fn group_attr_names(&self, group_path: &str) -> Vec<String> {
        if group_path == "/" {
            self.reader.root_attr_names()
        } else {
            self.reader
                .group_attr_names(group_path.trim_start_matches('/'))
        }
    }

    /// Read a group's attribute as a string (the NeXus `NX_class` etc.), or
    /// a root attribute when `group_path` is `"/"`. A leading `/` is tolerated.
    pub fn group_attr_string(&mut self, group_path: &str, attr: &str) -> Result<String> {
        let a = if group_path == "/" {
            self.reader.root_attr(attr)
        } else {
            self.reader
                .group_attr(group_path.trim_start_matches('/'), attr)
        }
        .ok_or_else(|| crate::error::Hdf5Error::NotFound(attr.to_string()))?
        .clone();
        Ok(self.reader.attr_string_value(&a)?)
    }
}

/// The writer-side attribute list a group path names: `"/"` is the
/// file-level (root) list, anything else the group's own.
fn group_attr_target(group_path: &str) -> crate::io::writer::AttrTarget<'_> {
    if group_path == "/" {
        crate::io::writer::AttrTarget::Root
    } else {
        crate::io::writer::AttrTarget::Group(group_path)
    }
}

/// Reinterpret a raw byte buffer as a typed vector. The buffer length must
/// be a whole multiple of `T`'s element size.
fn bytes_to_typed<T: H5Type>(raw: Vec<u8>) -> Result<Vec<T>> {
    let es = T::element_size();
    if es == 0 || !raw.len().is_multiple_of(es) {
        return Err(crate::error::Hdf5Error::TypeMismatch(format!(
            "raw data size {} is not a multiple of element size {es}",
            raw.len()
        )));
    }
    let count = raw.len() / es;
    let mut result = Vec::<T>::with_capacity(count);
    // Safety: `T: H5Type` is a `Copy` POD primitive exactly `element_size()`
    // bytes wide, so the byte buffer is a valid array of `count` `T`s.
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), result.as_mut_ptr() as *mut u8, raw.len());
        result.set_len(count);
    }
    Ok(result)
}

/// Raw bytes of one `H5Type` scalar.
fn scalar_to_bytes<T: H5Type>(value: &T) -> Vec<u8> {
    let es = T::element_size();
    // Safety: `T: H5Type` is a `Copy` POD primitive exactly `element_size()`
    // bytes wide.
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, es) }.to_vec()
}

/// Raw bytes of an `H5Type` slice, in element order.
fn slice_to_bytes<T: H5Type>(data: &[T]) -> &[u8] {
    // Safety: as `scalar_to_bytes`; the slice is a contiguous POD array.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}
