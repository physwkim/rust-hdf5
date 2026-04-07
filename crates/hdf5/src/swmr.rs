//! Single Writer / Multiple Reader (SWMR) API.
//!
//! Provides a high-level wrapper around the SWMR protocol for streaming
//! frame-based data (e.g., area detector images).

use std::path::Path;

use hdf5_io::Hdf5Reader;
use hdf5_io::SwmrWriter as IoSwmrWriter;

use crate::error::Result;
use crate::types::H5Type;

/// SWMR writer for streaming frame-based data to an HDF5 file.
///
/// Usage:
/// ```no_run
/// use hdf5::swmr::SwmrFileWriter;
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
    /// Create a new HDF5 file for SWMR streaming.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = IoSwmrWriter::create(path.as_ref())?;
        Ok(Self { inner })
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
        let idx = self.inner.create_streaming_dataset(name, datatype, frame_dims)?;
        Ok(idx)
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
/// use hdf5::swmr::SwmrFileReader;
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
    /// Open an HDF5 file for SWMR reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = Hdf5Reader::open_swmr(path.as_ref())?;
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
        self.reader.dataset_names().iter().map(|s| s.to_string()).collect()
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
    pub fn read_dataset<T: H5Type>(&mut self, name: &str) -> Result<Vec<T>> {
        let raw = self.reader.read_dataset_raw(name)?;
        if raw.len() % T::element_size() != 0 {
            return Err(crate::error::Hdf5Error::TypeMismatch(format!(
                "raw data size {} is not a multiple of element size {}",
                raw.len(), T::element_size(),
            )));
        }
        let count = raw.len() / T::element_size();
        let mut result = Vec::<T>::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(raw.as_ptr(), result.as_mut_ptr() as *mut u8, raw.len());
            result.set_len(count);
        }
        Ok(result)
    }
}
