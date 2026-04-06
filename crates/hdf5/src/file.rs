//! HDF5 file handle — the main entry point for the public API.
//!
//! ```no_run
//! use hdf5::H5File;
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

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use hdf5_io::{Hdf5Reader, Hdf5Writer};

use crate::dataset::{DatasetBuilder, H5Dataset};
use crate::error::{Hdf5Error, Result};
use crate::group::H5Group;
use crate::types::H5Type;

/// The inner state of an HDF5 file, shared with datasets via `Rc<RefCell<>>`.
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
    pub(crate) inner: Rc<RefCell<H5FileInner>>,
}

impl H5File {
    /// Create a new HDF5 file at `path`. Truncates if the file already exists.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let writer = Hdf5Writer::create(path.as_ref())?;
        Ok(Self {
            inner: Rc::new(RefCell::new(H5FileInner::Writer(writer))),
        })
    }

    /// Open an existing HDF5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = Hdf5Reader::open(path.as_ref())?;
        Ok(Self {
            inner: Rc::new(RefCell::new(H5FileInner::Reader(reader))),
        })
    }

    /// Return a handle to the root group.
    ///
    /// The root group can be used to create datasets and sub-groups.
    pub fn root_group(&self) -> H5Group {
        H5Group::new(Rc::clone(&self.inner), "/".to_string())
    }

    /// Start building a new dataset with the given element type.
    ///
    /// This returns a fluent builder. Call `.shape(...)` to set dimensions and
    /// `.create("name")` to finalize.
    ///
    /// ```no_run
    /// # use hdf5::H5File;
    /// let file = H5File::create("build.h5").unwrap();
    /// let ds = file.new_dataset::<f64>().shape(&[3, 4]).create("matrix").unwrap();
    /// ```
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilder<T> {
        DatasetBuilder::new(Rc::clone(&self.inner))
    }

    /// Open an existing dataset by name (read mode).
    pub fn dataset(&self, name: &str) -> Result<H5Dataset> {
        let inner = self.inner.borrow();
        match &*inner {
            H5FileInner::Reader(reader) => {
                let info = reader
                    .dataset_info(name)
                    .ok_or_else(|| Hdf5Error::NotFound(name.to_string()))?;
                let shape: Vec<usize> = info.dataspace.dims.iter().map(|&d| d as usize).collect();
                let element_size = info.datatype.element_size() as usize;
                Ok(H5Dataset::new_reader(
                    Rc::clone(&self.inner),
                    name.to_string(),
                    shape,
                    element_size,
                ))
            }
            H5FileInner::Writer(_) => Err(Hdf5Error::InvalidState(
                "cannot open a dataset by name in write mode; use new_dataset() instead"
                    .to_string(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState(
                "file is closed".to_string(),
            )),
        }
    }

    /// Explicitly close the file. For a writer, this finalizes the file
    /// (writes superblock, headers, etc.). For a reader, this is a no-op.
    ///
    /// The file is also auto-finalized on drop, but calling `close()` lets
    /// you handle errors.
    pub fn close(self) -> Result<()> {
        let old = {
            let mut inner = self.inner.borrow_mut();
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

    /// Flush the file to disk. Only meaningful in write mode.
    pub fn flush(&self) -> Result<()> {
        // The underlying writer does not expose a standalone flush; data is
        // written to disk immediately via pwrite. This is a compatibility
        // method that does nothing for now.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hdf5_file_test_{}.h5", name))
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
            let ds = file.new_dataset::<u8>().shape([4, 4]).create("data").unwrap();
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
    fn write_and_read_f64() {
        let path = temp_path("write_read_f64");

        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Write
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<f64>().shape([2, 3]).create("matrix").unwrap();
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

            let ds2 = file.new_dataset::<f32>().shape([2, 2]).create("floats").unwrap();
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

    #[test]
    fn write_file_for_h5dump() {
        let path = "/tmp/test_hdf5rs_integration.h5";
        let file = H5File::create(path).unwrap();

        let ds = file.new_dataset::<u8>().shape([4usize, 4]).create("data_u8").unwrap();
        let data: Vec<u8> = (0..16).collect();
        ds.write_raw(&data).unwrap();

        let ds2 = file.new_dataset::<f64>().shape([3usize, 2]).create("data_f64").unwrap();
        let fdata: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        ds2.write_raw(&fdata).unwrap();

        let ds3 = file.new_dataset::<i32>().shape([5usize]).create("values").unwrap();
        let idata: Vec<i32> = vec![-10, -5, 0, 5, 10];
        ds3.write_raw(&idata).unwrap();

        file.close().unwrap();

        // File exists
        assert!(std::path::Path::new(path).exists());
    }

    #[test]
    fn write_chunked_file_for_h5dump() {
        let path = "/tmp/test_hdf5rs_chunked.h5";
        let file = H5File::create(path).unwrap();

        // Create a chunked dataset with unlimited first dimension
        let ds = file.new_dataset::<f64>()
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

        assert!(std::path::Path::new(path).exists());
    }

    #[test]
    fn write_chunked_many_frames_for_h5dump() {
        let path = "/tmp/test_hdf5rs_chunked_many.h5";
        let file = H5File::create(path).unwrap();

        let ds = file.new_dataset::<i32>()
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

        assert!(std::path::Path::new(path).exists());
    }

    #[test]
    fn write_dataset_with_attributes() {
        use crate::types::VarLenUnicode;

        let path = "/tmp/test_hdf5rs_attributes.h5";
        let file = H5File::create(path).unwrap();

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

        assert!(std::path::Path::new(path).exists());
    }

    #[test]
    fn chunked_write_read_roundtrip() {
        let path = std::env::temp_dir().join("hdf5_chunked_roundtrip.h5");

        // Write
        {
            let file = H5File::create(&path).unwrap();
            let ds = file.new_dataset::<i32>()
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
            for i in 0..24 {
                assert_eq!(data[i], i as i32);
            }
        }

        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod h5py_compat_tests {
    use super::*;

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
