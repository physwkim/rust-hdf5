//! Group support.
//!
//! Groups are containers for datasets and other groups, forming a
//! hierarchical namespace within an HDF5 file.
//!
//! # Example
//!
//! ```no_run
//! use rust_hdf5::H5File;
//!
//! let file = H5File::create("groups.h5").unwrap();
//! let root = file.root_group();
//! let grp = root.create_group("detector").unwrap();
//! let ds = grp.new_dataset::<f32>()
//!     .shape(&[10])
//!     .create("temperature")
//!     .unwrap();
//! ```

use crate::dataset::DatasetBuilder;
use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner, borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::filter::FilterPipeline;
use crate::types::H5Type;

/// A handle to an HDF5 group.
///
/// Groups are containers for datasets and other groups. The root group
/// is always available via [`H5File::root_group`](crate::file::H5File::root_group).
pub struct H5Group {
    file_inner: SharedInner,
    /// The absolute path of this group (e.g., "/" or "/detector").
    name: String,
}

impl H5Group {
    /// Create a new group handle.
    pub(crate) fn new(file_inner: SharedInner, name: String) -> Self {
        Self { file_inner, name }
    }

    /// Return the name (path) of this group.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Start building a new dataset in this group.
    ///
    /// The dataset will be registered as a child of this group in the
    /// HDF5 file hierarchy.
    pub fn new_dataset<T: H5Type>(&self) -> DatasetBuilder<T> {
        DatasetBuilder::new_in_group(clone_inner(&self.file_inner), self.name.clone())
    }

    /// Create a sub-group within this group.
    ///
    /// Creates a real HDF5 group with its own object header.
    pub fn create_group(&self, name: &str) -> Result<H5Group> {
        let full_name = if self.name == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", self.name, name)
        };

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.create_group(&self.name, name)?;
            }
            H5FileInner::Reader(_) => {
                return Err(Hdf5Error::InvalidState(
                    "cannot create groups in read mode".into(),
                ));
            }
            H5FileInner::Closed => {
                return Err(Hdf5Error::InvalidState("file is closed".into()));
            }
        }
        drop(inner);

        Ok(H5Group {
            file_inner: clone_inner(&self.file_inner),
            name: full_name,
        })
    }

    /// Open an existing sub-group by name (read mode).
    pub fn group(&self, name: &str) -> Result<H5Group> {
        let full_name = if self.name == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", self.name, name)
        };

        // Verify the group exists by checking if any datasets have this prefix
        let inner = borrow_inner(&self.file_inner);
        if let H5FileInner::Reader(reader) = &*inner {
            let prefix = if full_name == "/" {
                String::new()
            } else {
                format!("{}/", full_name.trim_start_matches('/'))
            };
            let has_children = reader
                .dataset_names()
                .iter()
                .any(|n| n.starts_with(&prefix));
            if !has_children {
                return Err(Hdf5Error::NotFound(full_name));
            }
        }
        drop(inner);

        Ok(H5Group {
            file_inner: clone_inner(&self.file_inner),
            name: full_name,
        })
    }

    /// List dataset names that are direct children of this group.
    pub fn dataset_names(&self) -> Result<Vec<String>> {
        let inner = borrow_inner(&self.file_inner);
        let all_names = match &*inner {
            H5FileInner::Reader(reader) => reader
                .dataset_names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            H5FileInner::Writer(writer) => writer
                .dataset_names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            H5FileInner::Closed => return Ok(vec![]),
        };

        let prefix = if self.name == "/" {
            String::new()
        } else {
            format!("{}/", self.name.trim_start_matches('/'))
        };

        let mut result = Vec::new();
        for name in &all_names {
            let stripped = if prefix.is_empty() {
                name.as_str()
            } else if let Some(rest) = name.strip_prefix(&prefix) {
                rest
            } else {
                continue;
            };
            // Only direct children (no further '/')
            if !stripped.contains('/') {
                result.push(stripped.to_string());
            }
        }
        Ok(result)
    }

    /// Create a variable-length string dataset and write data within this group.
    pub fn write_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<()> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_string_dataset(&full_name, strings)?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                Ok(())
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Create a chunked, compressed variable-length string dataset within this group.
    pub fn write_vlen_strings_compressed(
        &self,
        name: &str,
        strings: &[&str],
        chunk_size: usize,
        pipeline: FilterPipeline,
    ) -> Result<()> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_string_dataset_compressed(
                    &full_name, strings, chunk_size, pipeline,
                )?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                Ok(())
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Create an empty chunked vlen string dataset ready for incremental appends.
    pub fn create_appendable_vlen_dataset(
        &self,
        name: &str,
        chunk_size: usize,
        pipeline: Option<FilterPipeline>,
    ) -> Result<()> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                let idx = writer
                    .create_appendable_vlen_string_dataset(&full_name, chunk_size, pipeline)?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                Ok(())
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Append variable-length strings to an existing chunked vlen string dataset.
    pub fn append_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<()> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                let ds_index = writer
                    .dataset_index(&full_name)
                    .ok_or_else(|| Hdf5Error::NotFound(full_name.clone()))?;
                writer.append_vlen_strings(ds_index, strings)?;
                Ok(())
            }
            H5FileInner::Reader(_) => {
                Err(Hdf5Error::InvalidState("cannot write in read mode".into()))
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// List sub-group names that are direct children of this group.
    pub fn group_names(&self) -> Result<Vec<String>> {
        let inner = borrow_inner(&self.file_inner);
        let all_names = match &*inner {
            H5FileInner::Reader(reader) => reader
                .dataset_names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            H5FileInner::Writer(writer) => writer
                .dataset_names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            H5FileInner::Closed => return Ok(vec![]),
        };

        let prefix = if self.name == "/" {
            String::new()
        } else {
            format!("{}/", self.name.trim_start_matches('/'))
        };

        let mut groups = std::collections::BTreeSet::new();
        for name in &all_names {
            let stripped = if prefix.is_empty() {
                name.as_str()
            } else if let Some(rest) = name.strip_prefix(&prefix) {
                rest
            } else {
                continue;
            };
            // If there's a '/', the first part is a group name
            if let Some(pos) = stripped.find('/') {
                groups.insert(stripped[..pos].to_string());
            }
        }
        Ok(groups.into_iter().collect())
    }

    /// Add (or replace) a string attribute on this group.
    ///
    /// This is the standard way to mark a NeXus class, e.g.
    /// `grp.set_attr_string("NX_class", "NXdetector")`.
    pub fn set_attr_string(&self, name: &str, value: &str) -> Result<()> {
        self.add_attr(AttributeMessage::scalar_string(name, value))
    }

    /// Add (or replace) a numeric scalar attribute on this group.
    pub fn set_attr_numeric<T: H5Type>(&self, name: &str, value: &T) -> Result<()> {
        let es = T::element_size();
        // Safety: `T: H5Type` is a `Copy` numeric primitive whose byte
        // representation is exactly `element_size()` wide.
        let raw = unsafe { std::slice::from_raw_parts(value as *const T as *const u8, es) };
        self.add_attr(AttributeMessage::scalar_numeric(
            name,
            T::hdf5_type(),
            raw.to_vec(),
        ))
    }

    /// Route an attribute to the writer: the root group goes to the
    /// file-level attribute list, any other group to its own header.
    fn add_attr(&self, attr: AttributeMessage) -> Result<()> {
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                if self.name == "/" {
                    writer.add_root_attribute(attr);
                } else {
                    writer.add_group_attribute(&self.name, attr)?;
                }
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot write attributes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// List this group's attribute names (read mode).
    pub fn attr_names(&self) -> Result<Vec<String>> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => {
                if self.name == "/" {
                    Ok(reader.root_attr_names())
                } else {
                    Ok(reader.group_attr_names(self.name.trim_start_matches('/')))
                }
            }
            _ => Err(Hdf5Error::InvalidState(
                "attr_names is only available in read mode".into(),
            )),
        }
    }

    /// Read one of this group's attributes as a string (read mode).
    pub fn attr_string(&self, name: &str) -> Result<String> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => {
                let attr = if self.name == "/" {
                    reader.root_attr(name)
                } else {
                    reader.group_attr(self.name.trim_start_matches('/'), name)
                }
                .ok_or_else(|| Hdf5Error::NotFound(name.to_string()))?;
                let end = attr
                    .data
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(attr.data.len());
                Ok(String::from_utf8_lossy(&attr.data[..end]).to_string())
            }
            _ => Err(Hdf5Error::InvalidState(
                "attr_string is only available in read mode".into(),
            )),
        }
    }
}
