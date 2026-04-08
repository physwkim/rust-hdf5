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

use crate::io::{Hdf5Reader, Hdf5Writer};

use crate::dataset::{DatasetBuilder, H5Dataset};
use crate::error::{Hdf5Error, Result};
use crate::group::H5Group;
use crate::types::H5Type;

// ---------------------------------------------------------------------------
// Thread-safety: choose between Rc<RefCell<>> and Arc<Mutex<>> based on
// the `threadsafe` feature flag.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "threadsafe"))]
pub(crate) type SharedInner = std::rc::Rc<std::cell::RefCell<H5FileInner>>;

#[cfg(feature = "threadsafe")]
pub(crate) type SharedInner = std::sync::Arc<std::sync::Mutex<H5FileInner>>;

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

#[cfg(feature = "threadsafe")]
pub(crate) fn borrow_inner(inner: &SharedInner) -> std::sync::MutexGuard<'_, H5FileInner> {
    inner.lock().unwrap()
}

#[cfg(feature = "threadsafe")]
pub(crate) fn borrow_inner_mut(inner: &SharedInner) -> std::sync::MutexGuard<'_, H5FileInner> {
    inner.lock().unwrap()
}

#[cfg(feature = "threadsafe")]
pub(crate) fn clone_inner(inner: &SharedInner) -> SharedInner {
    std::sync::Arc::clone(inner)
}

#[cfg(feature = "threadsafe")]
pub(crate) fn new_shared(inner: H5FileInner) -> SharedInner {
    std::sync::Arc::new(std::sync::Mutex::new(inner))
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
    pub fn set_attr_string(&self, name: &str, value: &str) -> Result<()> {
        use crate::format::messages::attribute::AttributeMessage;
        let attr = AttributeMessage::scalar_string(name, value);
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.add_root_attribute(attr);
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
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.add_root_attribute(attr);
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
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Reader(reader) => {
                let attr = reader
                    .root_attr(name)
                    .ok_or_else(|| Hdf5Error::NotFound(name.to_string()))?;
                let end = attr
                    .data
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(attr.data.len());
                Ok(String::from_utf8_lossy(&attr.data[..end]).to_string())
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
    pub fn write_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<()> {
        let mut inner = borrow_inner_mut(&self.inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.create_vlen_string_dataset(name, strings)?;
                Ok(())
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Open an existing dataset by name (read mode).
    pub fn dataset(&self, name: &str) -> Result<H5Dataset> {
        let inner = borrow_inner(&self.inner);
        match &*inner {
            H5FileInner::Reader(reader) => {
                let info = reader
                    .dataset_info(name)
                    .ok_or_else(|| Hdf5Error::NotFound(name.to_string()))?;
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

    #[test]
    fn write_file_for_h5dump() {
        let path = std::env::temp_dir().join("test_hdf5rs_integration.h5");
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
        let path = std::env::temp_dir().join("test_hdf5rs_chunked.h5");
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
        let path = std::env::temp_dir().join("test_hdf5rs_chunked_many.h5");
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

        let path = std::env::temp_dir().join("test_hdf5rs_attributes.h5");
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
    fn chunked_write_read_roundtrip() {
        let path = std::env::temp_dir().join("hdf5_chunked_roundtrip.h5");

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
        let path = std::env::temp_dir().join("hdf5_compressed_roundtrip.h5");

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
        let path = std::env::temp_dir().join("hdf5_compressed_many.h5");

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
        let path = std::env::temp_dir().join("hdf5_append.h5");

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
    fn vlen_string_write_read() {
        let path = std::env::temp_dir().join("hdf5_vlen_wr.h5");
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
    #[cfg(feature = "deflate")]
    fn shuffle_deflate_roundtrip() {
        let path = std::env::temp_dir().join("hdf5_shuf_defl.h5");
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
        let path = std::env::temp_dir().join("hdf5_file_attr.h5");
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
        let path = std::env::temp_dir().join("hdf5_scalar.h5");
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
        let path = std::env::temp_dir().join("hdf5_append_extend.h5");

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
        let path = std::env::temp_dir().join("hdf5_groups_rt.h5");

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
        let path = std::env::temp_dir().join("hdf5_file_create_group.h5");

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

        let path = std::env::temp_dir().join("hdf5_h5dump_validate.h5");

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
