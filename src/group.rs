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
use crate::format::creation_order::CreationOrder;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::filter::FilterPipeline;
use crate::format::messages::link::LinkTarget;
use crate::format::storage_kind::AttributeStorage;
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

    /// Create a soft link — `H5Lcreate_soft`, h5py's `h5py.SoftLink`.
    ///
    /// The link stores `target_path` as text and HDF5 resolves it on every
    /// traversal, so it may name an object that does not exist yet, or one
    /// that never will: unlike [`link`](Self::link), nothing is checked here
    /// and a dangling soft link is a legal file.
    ///
    /// ```no_run
    /// use rust_hdf5::H5File;
    ///
    /// let file = H5File::create("soft.h5").unwrap();
    /// file.new_dataset::<i32>().shape([8]).create("orig").unwrap();
    /// file.root_group().create_soft_link("alias", "/orig").unwrap();
    /// ```
    pub fn create_soft_link(&self, link_name: &str, target_path: &str) -> Result<()> {
        self.create_symbolic_link(
            link_name,
            LinkTarget::Soft {
                target: target_path.to_string(),
            },
        )
    }

    /// Create an external link — `H5Lcreate_external`, h5py's
    /// `h5py.ExternalLink`.
    ///
    /// The link names `target_path` inside `target_file`; neither is opened
    /// here, and libhdf5 resolves `target_file` against the directory holding
    /// *this* file, so a relative name is the portable form. As with
    /// [`create_soft_link`](Self::create_soft_link) the link may dangle: a
    /// file that is not there and an object that is not there are both legal.
    ///
    /// ```no_run
    /// use rust_hdf5::H5File;
    ///
    /// let file = H5File::create("master.h5").unwrap();
    /// file.root_group()
    ///     .create_external_link("ext", "payload.h5", "/data")
    ///     .unwrap();
    /// ```
    pub fn create_external_link(
        &self,
        link_name: &str,
        target_file: &str,
        target_path: &str,
    ) -> Result<()> {
        self.create_symbolic_link(
            link_name,
            LinkTarget::External {
                file: target_file.to_string(),
                path: target_path.to_string(),
            },
        )
    }

    /// Commit a datatype in this group under `name` — `H5Tcommit2`, h5py's
    /// `group["name"] = dtype`.
    ///
    /// The type becomes an object of its own, so datasets can be built on it
    /// with [`DatasetBuilder::committed_type`](crate::dataset::DatasetBuilder::committed_type)
    /// and share one definition instead of each carrying a copy. A committed
    /// datatype no dataset ever uses is still a complete object, and h5py
    /// reads it back as a `Datatype`.
    ///
    /// ```no_run
    /// use rust_hdf5::H5File;
    /// use rust_hdf5::format::messages::datatype::DatatypeMessage;
    ///
    /// let file = H5File::create("committed.h5").unwrap();
    /// let grp = file.create_group("types").unwrap();
    /// grp.commit_datatype("temperature", DatatypeMessage::f64_type()).unwrap();
    /// ```
    pub fn commit_datatype(
        &self,
        name: &str,
        datatype: crate::format::messages::datatype::DatatypeMessage,
    ) -> Result<()> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            format!("{}/{}", self.name.trim_start_matches('/'), name)
        };
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.commit_datatype(&full_name, datatype)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot commit datatypes in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Both symbolic-link constructors, behind one writer-mode check.
    fn create_symbolic_link(&self, link_name: &str, target: LinkTarget) -> Result<()> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.create_symbolic_link(&self.name, link_name, target)?;
                Ok(())
            }
            H5FileInner::Reader(_) => Err(Hdf5Error::InvalidState(
                "cannot create links in read mode".into(),
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
                // A group handle lists and reads through the reader it was
                // made from, and that reader is one file. Opening the target
                // file directly is the way across; saying so beats handing
                // back a handle whose listings would all come up empty.
                if let Some(edge) = reader.external_edge(group_path) {
                    return Err(Hdf5Error::Unsupported(format!(
                        "'{}' resolves through the external link '{}' to '{}' in '{}'; \
                         a group handle does not cross into another file — open '{}' \
                         and start from there (datasets read through the link)",
                        full_name, edge.link, edge.path, edge.file, edge.file
                    )));
                }
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
    /// via [`new_dataset`](Self::new_dataset). The datatype declares UTF-8;
    /// [`write_vlen_strings_ascii`](Self::write_vlen_strings_ascii) is the
    /// ASCII-declaring twin, as on [`H5File`](crate::H5File).
    pub fn write_vlen_strings(&self, name: &str, strings: &[&str]) -> Result<H5Dataset> {
        self.write_vlen_strings_charset(name, strings, 1)
    }

    /// Create a variable-length **ASCII** string dataset within this group.
    ///
    /// The group-level twin of
    /// [`H5File::write_vlen_strings_ascii`](crate::H5File::write_vlen_strings_ascii),
    /// with the same rejection of a string the ASCII declaration would
    /// misdescribe.
    pub fn write_vlen_strings_ascii(&self, name: &str, strings: &[&str]) -> Result<H5Dataset> {
        self.write_vlen_strings_charset(name, strings, 0)
    }

    /// The single owner of group-level vlen-string dataset creation: the two
    /// public entry points differ only in the character set they declare.
    fn write_vlen_strings_charset(
        &self,
        name: &str,
        strings: &[&str],
        charset: u8,
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
                let idx = writer.create_vlen_string_dataset(&full_name, strings, charset)?;
                let (shape, element_size, chunk_index) = writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
                    idx,
                    shape,
                    element_size,
                    chunk_index,
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
    ///
    /// The `u8` case of [`write_vlen_numeric`](Self::write_vlen_numeric).
    pub fn write_vlen_bytes(&self, name: &str, items: &[&[u8]]) -> Result<H5Dataset> {
        self.write_vlen_numeric(name, items)
    }

    /// Create a variable-length numeric-sequence dataset within this group.
    ///
    /// The group-level twin of
    /// [`H5File::write_vlen_numeric`](crate::H5File::write_vlen_numeric).
    pub fn write_vlen_numeric<T: H5Type>(&self, name: &str, items: &[&[T]]) -> Result<H5Dataset> {
        let images = crate::dataset::vlen_sequence_images(items)?;
        let images: Vec<&[u8]> = images.iter().map(|c| c.as_ref()).collect();
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            let trimmed = self.name.trim_start_matches('/');
            format!("{}/{}", trimmed, name)
        };

        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                let idx =
                    writer.create_vlen_sequence_dataset(&full_name, T::hdf5_type(), &images)?;
                let (shape, element_size, chunk_index) = writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
                    idx,
                    shape,
                    element_size,
                    chunk_index,
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
                let (shape, element_size, chunk_index) = writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
                    idx,
                    shape,
                    element_size,
                    chunk_index,
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
                let (shape, element_size, chunk_index) = writer.dataset_handle_parts(idx);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
                    idx,
                    shape,
                    element_size,
                    chunk_index,
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
                let ds_index = writer.open_dataset_index(&full_name)?;
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
                let index = writer.open_dataset_index(&full_name)?;
                let (shape, element_size, chunk_index) = writer.dataset_handle_parts(index);
                Ok(H5Dataset::new_writer(
                    clone_inner(&self.file_inner),
                    index,
                    shape,
                    element_size,
                    chunk_index,
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
            // A hard link to one of the writer's own objects is in the object
            // listings, so those are the union — plus every link that names a
            // path instead: a soft or external link made this session, or one
            // a reopened file brought in.
            H5FileInner::Writer(writer) => {
                let mut names = std::collections::BTreeSet::new();
                for (path, _) in writer.path_link_classes() {
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
                drop(inner);
                names.extend(self.dataset_names()?);
                names.extend(self.group_names()?);
                names.extend(self.named_datatype_names()?);
                Ok(names.into_iter().collect())
            }
            H5FileInner::Closed => Ok(vec![]),
        }
    }

    /// The committed (named) datatypes that are direct children of this
    /// group, in the order the catalog holds them.
    pub fn named_datatype_names(&self) -> Result<Vec<String>> {
        let inner = borrow_inner(&self.file_inner);
        let all = match &*inner {
            H5FileInner::Reader(reader) => reader
                .named_datatype_names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            H5FileInner::Writer(writer) => writer.committed_datatype_names(),
            H5FileInner::Closed => return Ok(vec![]),
        };
        Ok(all
            .iter()
            .filter_map(|path| self.direct_child(path))
            .collect())
    }

    /// Open a committed (named) datatype that is a child of this group.
    pub fn named_datatype(&self, name: &str) -> Result<crate::H5NamedDatatype> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            format!("{}/{}", self.name.trim_start_matches('/'), name)
        };
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                reader.named_datatype_info(&full_name)?;
                drop(inner);
                Ok(crate::H5NamedDatatype::new_reader(
                    clone_inner(&self.file_inner),
                    full_name,
                ))
            }
            H5FileInner::Writer(_) => Err(Hdf5Error::InvalidState(
                "committed datatypes are readable only in read mode".into(),
            )),
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// `path` as a direct child name of this group, or `None` when it is not
    /// one.
    fn direct_child(&self, path: &str) -> Option<String> {
        let prefix = if self.name == "/" {
            String::new()
        } else {
            format!("{}/", self.name.trim_start_matches('/'))
        };
        let stripped = if prefix.is_empty() {
            path
        } else {
            path.strip_prefix(&prefix)?
        };
        if stripped.is_empty() || stripped.contains('/') {
            None
        } else {
            Some(stripped.to_string())
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
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => reader
                .link_class(&full_name)
                .cloned()
                .ok_or(Hdf5Error::NotFound(full_name)),
            // Anything the writer holds by a path — created here or carried
            // in by a reopen — answers with the class it was made with; every
            // other name it knows reaches an object of its own, so it is hard.
            H5FileInner::Writer(writer) => {
                let by_path = writer
                    .path_link_classes()
                    .into_iter()
                    .find(|(p, _)| *p == full_name)
                    .map(|(_, class)| class);
                drop(inner);
                match by_path {
                    Some(class) => Ok(class),
                    None if self.link_names()?.iter().any(|n| n == name) => Ok(LinkClass::Hard),
                    None => Err(Hdf5Error::NotFound(full_name)),
                }
            }
            H5FileInner::Closed => Err(Hdf5Error::InvalidState("file is closed".into())),
        }
    }

    /// Why the dataset `name` in this group cannot be read, or `None` when it
    /// can be.
    ///
    /// A dataset whose datatype (or any other message its payload depends on)
    /// this crate cannot decode is still listed by
    /// [`dataset_names`](Self::dataset_names) — the file contains it — and
    /// this says what stands in the way. Opening it through
    /// [`H5File::dataset`](crate::H5File::dataset) fails with
    /// [`Hdf5Error::Unsupported`] carrying the same text.
    pub fn unreadable_reason(&self, name: &str) -> Result<Option<String>> {
        let full_name = if self.name == "/" {
            name.to_string()
        } else {
            format!("{}/{}", self.name.trim_start_matches('/'), name)
        };
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                Ok(reader.unreadable_reason(&full_name).map(str::to_string))
            }
            // The writer only holds datasets it built itself.
            H5FileInner::Writer(_) => Ok(None),
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

    /// Add (or replace) an object-reference attribute on this group — h5py's
    /// `g.attrs['owner'] = f['/data'].ref`.
    ///
    /// `path` names a dataset or a group (`/` is the root group) and must
    /// already exist. The attribute takes the scalar shape h5py gives a single
    /// reference; [`set_attr_object_references`](Self::set_attr_object_references)
    /// is the array form. What reaches the file is the target's object header
    /// address, which is assigned when the file is finalized.
    pub fn set_attr_object_reference(&self, name: &str, path: &str) -> Result<()> {
        self.set_reference_attr(name, &[path], &[])
    }

    /// Add (or replace) a 1-D array of object references as an attribute on
    /// this group — the array counterpart of
    /// [`set_attr_object_reference`](Self::set_attr_object_reference).
    pub fn set_attr_object_references(&self, name: &str, paths: &[&str]) -> Result<()> {
        self.set_reference_attr(name, paths, &[paths.len() as u64])
    }

    /// Route a reference attribute to the writer, the way
    /// [`add_attr`](Self::add_attr) routes every other kind.
    fn set_reference_attr(&self, name: &str, paths: &[&str], dims: &[u64]) -> Result<()> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Writer(writer) => {
                writer.set_object_reference_attribute(self.attr_target(), name, paths, dims)?;
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
        let mut inner = borrow_inner_mut(&self.file_inner);
        match &mut *inner {
            H5FileInner::Reader(reader) => {
                if self.name == "/" {
                    Ok(reader.root_attr_names()?)
                } else {
                    Ok(reader.group_attr_names(self.name.trim_start_matches('/'))?)
                }
            }
            _ => Err(Hdf5Error::InvalidState(
                "attr_names is only available in read mode".into(),
            )),
        }
    }

    /// This group's own attribute creation-order policy — the equivalent of
    /// `H5Pget_attr_creation_order(gid.get_create_plist())` — `-` when
    /// neither `TRACKED` nor `INDEXED` is set, and never derived from
    /// whether the group currently holds any attributes: a group can track
    /// creation order and still be empty (read mode).
    pub fn attr_creation_order(&self) -> Result<CreationOrder> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => Ok(if self.name == "/" {
                reader.root_attr_creation_order()
            } else {
                reader.group_attr_creation_order(self.name.trim_start_matches('/'))
            }),
            _ => Err(Hdf5Error::InvalidState(
                "attr_creation_order is only available in read mode".into(),
            )),
        }
    }

    /// This group's own compact-vs-dense attribute storage — the equivalent
    /// of `h5py.h5o.get_info(gid.id).meta_size.attr.index_size` being
    /// nonzero.
    pub fn attr_storage(&self) -> Result<AttributeStorage> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => Ok(if self.name == "/" {
                reader.root_attr_storage()
            } else {
                reader.group_attr_storage(self.name.trim_start_matches('/'))
            }),
            _ => Err(Hdf5Error::InvalidState(
                "attr_storage is only available in read mode".into(),
            )),
        }
    }

    /// This group's own object-header attribute count — the equivalent of
    /// `h5py.h5o.get_info(gid.id).num_attrs`, which need not equal
    /// [`attr_names`](Self::attr_names)'s length when the set could not be
    /// read whole.
    pub fn header_attr_count(&self) -> Result<u64> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => Ok(if self.name == "/" {
                reader.root_header_attr_count()?
            } else {
                reader.group_header_attr_count(self.name.trim_start_matches('/'))?
            }),
            _ => Err(Hdf5Error::InvalidState(
                "header_attr_count is only available in read mode".into(),
            )),
        }
    }

    /// Why this group's attribute `name` cannot be read, or `None` when it
    /// can be. The attribute counterpart of
    /// [`unreadable_reason`](Self::unreadable_reason)'s shape for datasets:
    /// an attribute this crate cannot decode stays in
    /// [`attr_names`](Self::attr_names) and answers here.
    pub fn attr_unreadable_reason(&self, name: &str) -> Result<Option<String>> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => Ok(if self.name == "/" {
                reader.root_attr_unreadable_reason(name)
            } else {
                reader.group_attr_unreadable_reason(self.name.trim_start_matches('/'), name)
            }
            .map(str::to_string)),
            _ => Err(Hdf5Error::InvalidState(
                "attr_unreadable_reason is only available in read mode".into(),
            )),
        }
    }

    /// Why this group's attribute *set* cannot be listed, or `None` when it
    /// can be.
    ///
    /// The object-scope counterpart of
    /// [`attr_unreadable_reason`](Self::attr_unreadable_reason): a failure
    /// that belongs to no single name — a dense set whose heap or name index
    /// will not read — leaves nothing to list, so
    /// [`attr_names`](Self::attr_names) returns it as an error and this
    /// reports it without one.
    pub fn attrs_unreadable_reason(&self) -> Result<Option<String>> {
        let inner = borrow_inner(&self.file_inner);
        match &*inner {
            H5FileInner::Reader(reader) => Ok(if self.name == "/" {
                reader.root_attrs_unreadable_reason()
            } else {
                reader.group_attrs_unreadable_reason(self.name.trim_start_matches('/'))
            }
            .map(str::to_string)),
            _ => Err(Hdf5Error::InvalidState(
                "attrs_unreadable_reason is only available in read mode".into(),
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
                }?
                .clone();
                Ok(reader.attr_string_value(&attr)?)
            }
            _ => Err(Hdf5Error::InvalidState(
                "attr_string is only available in read mode".into(),
            )),
        }
    }
}
