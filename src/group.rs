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

use crate::dataset::{DatasetBuilder, H5Dataset};
use crate::error::{Hdf5Error, Result};
use crate::file::{borrow_inner, borrow_inner_mut, clone_inner, H5FileInner, SharedInner};
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::filter::FilterPipeline;
use crate::io::reader::LinkClass;
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

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
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

    /// Create a hard link in this group: an additional name `link_name`
    /// for the object that already exists at `target_path`.
    ///
    /// No data is copied — the link and its target share one object, just
    /// as `h5py` / libhdf5 hard links do. `target_path` may be given with
    /// or without a leading `/` and must name an existing dataset or group.
    /// This is the NeXus-style way to expose a dataset at a second
    /// canonical location (e.g. `/entry/data/data`) without duplicating it.
    ///
    /// ```no_run
    /// use rust_hdf5::H5File;
    ///
    /// let file = H5File::create("nexus.h5").unwrap();
    /// let inst = file.root_group().create_group("instrument").unwrap();
    /// inst.new_dataset::<f32>().shape(&[10]).create("data").unwrap();
    /// let data = file.root_group().create_group("data").unwrap();
    /// // /data/data is now a hard link to /instrument/data — no copy.
    /// data.link("data", "/instrument/data").unwrap();
    /// ```
    pub fn link(&self, link_name: &str, target_path: &str) -> Result<()> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.create_hard_link(&self.name, link_name, target_path)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot create hard links in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Open an existing sub-group by name (read mode).
    pub fn group(&self, name: &str) -> Result<H5Group> {
        let full_name = if self.name == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", self.name, name)
        };

        // Verify the group exists by consulting the reader's actual group
        // set (derived from link records), not inferred dataset prefixes.
        // This opens empty groups, attribute-only groups, and
        // subgroup-only groups, which have no datasets beneath them.
        let inner = borrow_inner(&self.file_inner);
        let full_name = match &*inner {
            H5FileInner::Reader(reader) => {
                let group_path = full_name.trim_start_matches('/');
                if !reader.has_group(group_path) {
                    return Err(Hdf5Error::NotFound(full_name));
                }
                // Store the traversed path, as write mode does below: the
                // handle's listings key off it, so a group reached through a
                // soft link or a group hard link lists the same children as
                // the group it names.
                format!("/{}", reader.canonical_path(group_path))
            }
            // In write mode the handle stores the tree path, so a path
            // through hard links resolves once here and every operation
            // made through the handle lands on the link's target.
            H5FileInner::Writer(writer) => writer.canonical_group_path(&full_name),
            H5FileInner::Closed => full_name,
        };
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
    ///
    /// Returns a writer-mode handle to the created dataset so attributes can be
    /// attached to it (e.g. units, descriptions) just like a dataset created
    /// via [`new_dataset`](Self::new_dataset).
    pub fn write_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<H5Dataset> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_string_dataset(&full_name, strings)?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
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

    /// Create a variable-length byte-array dataset and write data within this
    /// group.
    ///
    /// Each `&[u8]` becomes one element of variable length, stored as a vlen
    /// sequence of `u8`. h5py reads it back as an array of `uint8` arrays.
    /// Returns a writer-mode handle so attributes can be attached, like
    /// [`write_vlen_strings`](Self::write_vlen_strings).
    pub fn write_vlen_bytes(&self, name: &str, items: &[&[u8]]) -> Result<H5Dataset> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_bytes_dataset(&full_name, items)?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
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

    /// Create a chunked, compressed variable-length string dataset within this group.
    ///
    /// Returns a writer-mode handle to the created dataset so attributes can be
    /// attached to it, like [`write_vlen_strings`](Self::write_vlen_strings).
    pub fn write_vlen_strings_compressed(
        &self,
        name: &str,
        strings: &[&str],
        chunk_size: usize,
        pipeline: FilterPipeline,
    ) -> Result<H5Dataset> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer.create_vlen_string_dataset_compressed(
                    &full_name, strings, chunk_size, pipeline,
                )?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
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
    /// Returns a writer-mode handle to the created dataset so attributes can be
    /// attached before or between [`append_vlen_strings`](Self::append_vlen_strings)
    /// calls.
    pub fn create_appendable_vlen_dataset(
        &self,
        name: &str,
        chunk_size: usize,
        pipeline: Option<FilterPipeline>,
    ) -> Result<H5Dataset> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx = writer
                    .create_appendable_vlen_string_dataset(&full_name, chunk_size, pipeline)?;
                if self.name != "/" {
                    writer.assign_dataset_to_group(&self.name, idx)?;
                }
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
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
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
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

    /// Reopen a writer-mode handle to a dataset in this group by name.
    ///
    /// Mirrors [`H5File::dataset_writer`](crate::file::H5File::dataset_writer)
    /// but resolves `name` relative to this group, so a dataset created here
    /// (including via the vlen-string helpers) can be reopened to attach
    /// attributes or append chunks. `name` is the link name within this group.
    pub fn dataset_writer(&self, name: &str) -> Result<H5Dataset> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let index = writer
                    .dataset_index(&full_name)
                    .ok_or_else(|| Hdf5Error::NotFound(full_name.clone()))?;
                let (shape, element_size, chunked, btree2, fixed_array) =
                    writer.dataset_handle_parts(index);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
                    index,
                    shape,
                    element_size,
                    chunked,
                    btree2,
                    fixed_array,
                ))
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot open a dataset_writer in read mode; use dataset() instead".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// List sub-group names that are direct children of this group.
    pub fn group_names(&self) -> Result<Vec<String>> {
        let prefix = if self.name == "/" {
            String::new()
        } else {
            format!("{}/", self.name.trim_start_matches('/'))
        };

        let mut groups = std::collections::BTreeSet::new();
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            // Read mode: list immediate child groups from the reader's
            // actual group set (link records), so empty / attribute-only /
            // subgroup-only child groups are included.
            H5FileInner::Reader(reader) => {
                for path in reader.group_paths() {
                    let stripped = if prefix.is_empty() {
                        path.as_str()
                    } else if let Some(rest) = path.strip_prefix(&prefix) {
                        rest
                    } else {
                        continue;
                    };
                    if stripped.is_empty() {
                        continue;
                    }
                    // Immediate child only: take the first path component.
                    let child = match stripped.find('/') {
                        Some(pos) => &stripped[..pos],
                        None => stripped,
                    };
                    groups.insert(child.to_string());
                }
            }
            // Write mode: no link-record store; infer from dataset paths.
            H5FileInner::Writer(writer) => {
                for name in writer.dataset_names() {
                    let stripped = if prefix.is_empty() {
                        name.as_str()
                    } else if let Some(rest) = name.strip_prefix(&prefix) {
                        rest
                    } else {
                        continue;
                    };
                    if let Some(pos) = stripped.find('/') {
                        groups.insert(stripped[..pos].to_string());
                    }
                }
            }
            H5FileInner::Closed => return Ok(vec![]),
        }
        Ok(groups.into_iter().collect())
    }

    /// List every link that is a direct child of this group — hard, soft and
    /// external alike, in name order.
    ///
    /// This is the listing of *links* (`H5Lget_name_by_idx`, h5py's
    /// `grp.keys()`), not of the objects they reach:
    /// [`dataset_names`](Self::dataset_names) and
    /// [`group_names`](Self::group_names) answer the object question, and a
    /// soft or external link appears here whether or not its target resolves.
    /// Pair it with [`link_class`](Self::link_class) to tell the kinds apart.
    pub fn link_names(&self) -> Result<Vec<String>> {
        let prefix = if self.name == "/" {
            String::new()
        } else {
            format!("{}/", self.name.trim_start_matches('/'))
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => {
                let mut names = std::collections::BTreeSet::new();
                for path in reader.links().keys() {
                    let stripped = if prefix.is_empty() {
                        path.as_str()
                    } else if let Some(rest) = path.strip_prefix(&prefix) {
                        rest
                    } else {
                        continue;
                    };
                    if !stripped.is_empty() && !stripped.contains('/') {
                        names.insert(stripped.to_string());
                    }
                }
                Ok(names.into_iter().collect())
            }
            // The writer creates hard links only, so its link listing is the
            // union of the object listings.
            H5FileInner::Writer(_) => {
                drop(inner);
                let mut names: std::collections::BTreeSet<String> =
                    self.dataset_names()?.into_iter().collect();
                names.extend(self.group_names()?);
                Ok(names.into_iter().collect())
            }
            H5FileInner::Closed => Ok(vec![]),
        }
    }

    /// The class of the link `name` in this group, carrying the value
    /// `H5Lget_val` returns for the classes that have one — the target path
    /// of a soft link, the file and path of an external link.
    ///
    /// # Errors
    ///
    /// [`Hdf5Error::NotFound`] when this group holds no link of that name.
    pub fn link_class(&self, name: &str) -> Result<LinkClass> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            format!("{}/{}", self.name.trim_start_matches('/'), name)
        };
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => reader
                .link_class(&full_name)
                .cloned()
                .ok_or(Hdf5Error::NotFound(full_name)),
            // The writer creates hard links only.
            H5FileInner::Writer(_) => {
                drop(inner);
                if self.link_names()?.iter().any(|n| n == name) {
                    Ok(LinkClass::Hard)
                } else {
                    Err(Hdf5Error::NotFound(full_name))
                }
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Add (or replace) a string attribute on this group.
    ///
    /// This is the standard way to mark a NeXus class, e.g.
    /// `grp.set_attr_string("NX_class", "NXdetector")`. The value is stored as
    /// a variable-length UTF-8 string (read back as a Python `str` by h5py),
    /// not a fixed-length string.
    pub fn set_attr_string(&self, name: &str, value: &str) -> Result<()> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.set_vlen_string_attribute(self.attr_target(), name, value)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot write attributes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
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

    /// Add (or replace) a numeric (or bool) **array** attribute on this group.
    ///
    /// The values are written as a 1-D HDF5 array attribute (simple dataspace
    /// `[values.len()]`), read back by h5py as a numpy array — the array
    /// counterpart of [`set_attr_numeric`](Self::set_attr_numeric). For a
    /// multi-dimensional shape use
    /// [`set_attr_array_numeric_nd`](Self::set_attr_array_numeric_nd).
    pub fn set_attr_array_numeric<T: H5Type>(&self, name: &str, values: &[T]) -> Result<()> {
        self.set_attr_array_numeric_nd(name, values, &[values.len()])
    }

    /// Add (or replace) a numeric (or bool) **N-dimensional array** attribute on
    /// this group.
    ///
    /// `shape` gives the dataspace dimensions; `values` is the row-major data
    /// and its length must equal the product of `shape` (an empty `shape` is a
    /// scalar, requiring exactly one value). Read back by h5py as a numpy array
    /// of that shape. [`set_attr_array_numeric`](Self::set_attr_array_numeric)
    /// is the 1-D convenience form.
    pub fn set_attr_array_numeric_nd<T: H5Type>(
        &self,
        name: &str,
        values: &[T],
        shape: &[usize],
    ) -> Result<()> {
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
        self.add_attr(AttributeMessage::array_numeric(
            name,
            T::hdf5_type(),
            &dims,
            raw.to_vec(),
        ))
    }

    /// Add (or replace) a variable-length UTF-8 string **array** attribute on
    /// this group, read back by h5py as a 1-D array of `str` — the array
    /// counterpart of [`set_attr_string`](Self::set_attr_string). For a
    /// multi-dimensional shape use
    /// [`set_attr_string_array_nd`](Self::set_attr_string_array_nd).
    pub fn set_attr_string_array(&self, name: &str, values: &[&str]) -> Result<()> {
        self.set_attr_string_array_nd(name, values, &[values.len()])
    }

    /// Add (or replace) a variable-length UTF-8 string **N-dimensional array**
    /// attribute on this group.
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
        // The vlen array message needs `&mut writer` (global-heap allocation),
        // so we route it the same way `set_attr_string` does rather than through
        // `add_attr` (which re-borrows the writer).
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Writer(writer) => {
                writer.set_vlen_string_array_attribute(self.attr_target(), name, values, &dims)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot write attributes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// The writer-side attribute list this group's attributes live in: the
    /// root group's is the file-level list, any other group's is its own.
    fn attr_target(&self) -> crate::io::writer::AttrTarget<'_> {
        if self.name == "/" {
            crate::io::writer::AttrTarget::Root
        } else {
            crate::io::writer::AttrTarget::Group(&self.name)
        }
    }

    /// Route an attribute to the writer: the root group goes to the
    /// file-level attribute list, any other group to its own header.
    fn add_attr(&self, attr: AttributeMessage) -> Result<()> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.set_attribute(self.attr_target(), attr)?;
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
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                let attr = if self.name == "/" {
                    reader.root_attr(name)
                } else {
                    reader.group_attr(self.name.trim_start_matches('/'), name)
                }
                .ok_or_else(|| Hdf5Error::NotFound(name.to_string()))?
                .clone();
                Ok(reader.attr_string_value(&attr)?)
            }
            _ => Err(Hdf5Error::InvalidState(
                "attr_string is only available in read mode".into(),
            )),
        }
    }
}
