//! Zero-copy views of a dataset's stored bytes, under the `mmap` feature.
//!
//! [`H5Dataset::read_mapped`](crate::H5Dataset::read_mapped) hands back a
//! [`MappedView<T>`] that dereferences to `&[T]` pointing straight into the
//! file's memory map — no read, no copy, no allocation. Whether that is
//! possible is a property of the dataset, not of the caller, so when it is not
//! the answer is a [`ViewRefusal`] naming the reason. This module never falls
//! back to copying: a caller who asked for a view and got `Ok` knows the bytes
//! were never moved.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use memmap2::Mmap;

use crate::io::reader::{DatasetViewSource, ViewStorage};
use crate::types::H5Type;

/// Why a dataset's stored bytes cannot be handed out as a `&[T]` pointing
/// into the file's map.
///
/// Carried by [`Hdf5Error::NotViewable`](crate::Hdf5Error::NotViewable).
/// Every variant is a fact about the dataset or the request, and every one of
/// them is a case where a copying read
/// ([`read_raw`](crate::H5Dataset::read_raw),
/// [`read_slice`](crate::H5Dataset::read_slice)) still works.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewRefusal {
    /// The file holding this dataset is not memory-mapped. A read-only open
    /// takes a whole-file map when it can, but an empty file, a file larger
    /// than the address space, and a filesystem that refuses `mmap` all leave
    /// the handle reading through `pread`, with nothing to point at.
    NotMapped,
    /// The dataset's raw data is not one stretch of the mapped file. The
    /// phrase says what it is instead — chunked, compact, virtual, or held in
    /// external data files.
    Layout(&'static str),
    /// The dataset has no storage allocated: every element reads as the fill
    /// value, which lives in the object header rather than as an image of the
    /// data anywhere in the file.
    Unallocated,
    /// `T` is not as wide as the stored element, so a `&[T]` over the stored
    /// bytes would not be this dataset's elements.
    ElementSize {
        /// `T::element_size()`.
        view: usize,
        /// The width the datatype message declares.
        stored: usize,
    },
    /// The stored bytes are not already the host image of a `T` — reading
    /// them as `T` needs a per-element conversion, which is a copy by
    /// definition. The phrase names the conversion the stored type would
    /// need.
    ElementImage(&'static str),
    /// The data starts at a file offset `T` cannot be read from in place.
    Alignment {
        /// The absolute file offset of the first element.
        offset: u64,
        /// `align_of::<T>()`, which `offset` is not a multiple of.
        align: usize,
    },
    /// The dataset's image runs past the end of the map — the file was
    /// truncated, or grew after the map was taken and the dataset's extent
    /// now names bytes the map does not cover.
    PastMappedEnd {
        /// The file offset one past the last byte the image claims.
        end: u64,
        /// How many bytes the map holds.
        mapped: u64,
    },
    /// The requested range is not one contiguous run of the stored image, or
    /// is not a range of this dataset at all. The message says which.
    Range(String),
}

impl fmt::Display for ViewRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMapped => write!(f, "the file is not memory-mapped"),
            Self::Layout(what) => write!(f, "{what}"),
            Self::Unallocated => write!(
                f,
                "it has no storage allocated; every element reads as the fill value"
            ),
            Self::ElementSize { view, stored } => write!(
                f,
                "the view type is {view} bytes wide but the stored element is {stored}"
            ),
            Self::ElementImage(what) => write!(
                f,
                "the stored elements are not the host image of the view type: {what}"
            ),
            Self::Alignment { offset, align } => write!(
                f,
                "its data starts at file offset {offset}, which is not a multiple of the \
                 view type's {align}-byte alignment"
            ),
            Self::PastMappedEnd { end, mapped } => write!(
                f,
                "its image ends at {end} but the map holds {mapped} bytes"
            ),
            Self::Range(what) => write!(f, "{what}"),
        }
    }
}

/// The part of a dataset a view was asked for.
pub(crate) enum ViewRange<'a> {
    /// Every element, in stored order.
    Whole,
    /// The hyperslab [`read_slice`](crate::H5Dataset::read_slice) would
    /// return, which is viewable only when it is one contiguous run of the
    /// stored image.
    Slab {
        starts: &'a [usize],
        counts: &'a [usize],
    },
}

/// Decide whether `src` can be viewed as `T` over `range`, and where.
///
/// The single owner of the decision, and the only caller of
/// [`MappedView::new`]. `Hdf5Reader::dataset_view_source` reports facts and
/// judges none of them; `MappedView::new` proves the cast is sound and asks
/// nothing about the dataset. Every reason a view is refused for what the
/// dataset *is* is weighed here, so no caller can assemble a view out of
/// parts that were never weighed together.
pub(crate) fn view<T: H5Type>(
    src: &DatasetViewSource,
    range: ViewRange<'_>,
) -> Result<MappedView<T>, ViewRefusal> {
    let Some(map) = src.map.as_ref() else {
        return Err(ViewRefusal::NotMapped);
    };
    let (offset, bytes) = match src.storage {
        ViewStorage::Contiguous { offset, len } => (offset, len),
        ViewStorage::Unallocated => return Err(ViewRefusal::Unallocated),
        ViewStorage::Elsewhere(what) => return Err(ViewRefusal::Layout(what)),
    };

    let width = T::element_size();
    let stored = src.datatype.element_size() as usize;
    if width != stored {
        return Err(ViewRefusal::ElementSize {
            view: width,
            stored,
        });
    }
    if let Some(what) = crate::dataset::stored_image_mismatch(&src.datatype, width) {
        return Err(ViewRefusal::ElementImage(what));
    }

    let (offset, count) = match range {
        // `bytes` is `product(dims) * stored`, and `stored == width` was just
        // checked, so the division is exact.
        ViewRange::Whole => (offset, bytes / width as u64),
        ViewRange::Slab { starts, counts } => {
            let (first, count) = slab_run(&src.dims, starts, counts)?;
            let skip = first.checked_mul(width as u64).ok_or_else(|| {
                ViewRefusal::Range(format!(
                    "the range starts at element {first}, whose byte offset does not fit \
                     in the address space"
                ))
            })?;
            let offset = offset.checked_add(skip).ok_or_else(|| {
                ViewRefusal::Range(format!(
                    "the range starts {skip} bytes into a dataset at file offset {offset}, \
                     which does not fit in the address space"
                ))
            })?;
            (offset, count)
        }
    };
    MappedView::new(Arc::clone(map), offset, count)
}

/// The element index the hyperslab starts at and how many elements it holds,
/// for a hyperslab that is one contiguous run of a row-major image.
///
/// A selection is one run exactly when a trailing group of dimensions is
/// taken whole, the dimension just before it is taken as one span, and every
/// dimension before *that* selects a single index — anything else steps over
/// elements the run would have to include.
fn slab_run(dims: &[u64], starts: &[usize], counts: &[usize]) -> Result<(u64, u64), ViewRefusal> {
    if starts.len() != dims.len() || counts.len() != dims.len() {
        return Err(ViewRefusal::Range(format!(
            "the dataset has {} dimension(s) but the range names {} start(s) and {} count(s)",
            dims.len(),
            starts.len(),
            counts.len()
        )));
    }
    for (d, dim) in dims.iter().enumerate() {
        let end = (starts[d] as u64).checked_add(counts[d] as u64);
        if end.is_none_or(|e| e > *dim) {
            return Err(ViewRefusal::Range(format!(
                "dimension {d} holds {dim} element(s) but the range asks for {} from {}",
                counts[d], starts[d]
            )));
        }
    }

    let mut count = 1u64;
    for &c in counts {
        count = count.checked_mul(c as u64).ok_or_else(|| {
            ViewRefusal::Range("the range holds more elements than fit in a count".into())
        })?;
    }

    // An empty selection is one empty run wherever it is placed, so the
    // shape rule below — which asks what lies *between* selected elements —
    // has nothing to say about it.
    if count != 0 {
        // Walk in from the fastest-varying end over the dimensions taken
        // whole; `d` lands on the dimension that may be taken as a partial
        // span.
        let mut d = dims.len();
        while d > 0 && counts[d - 1] as u64 == dims[d - 1] {
            d -= 1;
        }
        if d > 0 {
            for (i, &c) in counts.iter().enumerate().take(d - 1) {
                if c != 1 {
                    return Err(ViewRefusal::Range(format!(
                        "the range is not one contiguous run of the stored image: it takes \
                         {c} indices along dimension {i} while dimension {} is only \
                         partially selected",
                        d - 1
                    )));
                }
            }
        }
    }

    let mut first = 0u64;
    let mut stride = 1u64;
    for d in (0..dims.len()).rev() {
        first = first.saturating_add((starts[d] as u64).saturating_mul(stride));
        stride = stride.saturating_mul(dims[d]);
    }
    Ok((first, count))
}

/// A borrowed-from-the-file slice of `T`, kept alive by the map it points
/// into.
///
/// Dereferences to `&[T]`. The bytes are the file's own — nothing was read,
/// copied, or allocated to produce them — and the view holds a share of the
/// map, so it stays readable after the dataset, the file handle, and even a
/// [`refresh`](crate::swmr::SwmrFileReader::refresh) that retook the map are
/// gone. What it shows is the file as it was when *that* map was taken.
///
/// See [`H5Dataset::read_mapped`](crate::H5Dataset::read_mapped) for how one
/// is obtained and when it is refused.
pub struct MappedView<T> {
    /// The map the elements live in. Held, not borrowed: this is what keeps
    /// the pages mapped for as long as the view exists.
    map: Arc<Mmap>,
    /// Byte offset of the first element within `map`.
    start: usize,
    /// How many elements the view holds.
    count: usize,
    _elem: PhantomData<T>,
}

impl<T: H5Type> MappedView<T> {
    /// The only constructor: `count` elements of `T` at absolute file offset
    /// `offset` in `map`.
    ///
    /// Refuses rather than builds a view whose cast would not be sound, so a
    /// `MappedView` that exists is one whose [`Deref`] cannot be wrong. It is
    /// private to this module and [`view`] is its only caller, so no code
    /// anywhere can reach [`Deref`] around these checks or around the
    /// viewability decision that precedes them.
    fn new(map: Arc<Mmap>, offset: u64, count: u64) -> Result<Self, ViewRefusal> {
        let mapped = map.len();
        let past = |end: u64| ViewRefusal::PastMappedEnd {
            end,
            mapped: mapped as u64,
        };
        let bytes_u64 = count.saturating_mul(std::mem::size_of::<T>() as u64);
        let claimed_end = offset.saturating_add(bytes_u64);
        let fits = usize::try_from(offset)
            .ok()
            .zip(usize::try_from(bytes_u64).ok())
            .zip(usize::try_from(count).ok())
            .filter(|((start, bytes), _)| start.checked_add(*bytes).is_some_and(|e| e <= mapped));
        let Some(((start, _), count)) = fits else {
            return Err(past(claimed_end));
        };

        if !map[start..].as_ptr().cast::<T>().is_aligned() {
            return Err(ViewRefusal::Alignment {
                offset,
                align: std::mem::align_of::<T>(),
            });
        }
        Ok(Self {
            map,
            start,
            count,
            _elem: PhantomData,
        })
    }
}

impl<T> Deref for MappedView<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: the four things `from_raw_parts` asks for, each established
        // by `MappedView::new`, which is the only way to reach this:
        //
        // * Alignment — `new` checked this exact pointer, `map[start..]`'s,
        //   against `align_of::<T>()`. `map` is an `Arc<Mmap>` that has not
        //   moved since (the mapping's address is fixed for its lifetime, and
        //   `Arc` never relocates the `Mmap`), so the pointer is the same one.
        // * Size — `new` checked `start + count * size_of::<T>() <= map.len()`,
        //   so the `count` elements lie entirely inside the mapping, and
        //   `start <= map.len()` makes the index valid.
        // * Initialization — every byte of a mapped file page is initialized
        //   by the kernel, and every bit pattern is a valid `T`: that is the
        //   `H5Type` contract, the same one a typed read reinterpreting a
        //   stored image already rests on, and `new` requires it.
        // * Lifetime and aliasing — the view owns a share of the map, so the
        //   pages stay mapped for at least as long as the returned reference,
        //   which borrows `self`. The mapping is read-only and nothing in
        //   this crate hands out a `&mut` to it.
        //
        // The one hazard the checks cannot remove is another process
        // truncating the file under the mapping, which faults with `SIGBUS`
        // on the pages that went away. That is the risk taken when the file
        // was mapped at all (see `ReadSource::for_read_only`); a view does
        // not add to it, and no guard inside this process can close it.
        unsafe {
            std::slice::from_raw_parts(self.map[self.start..].as_ptr().cast::<T>(), self.count)
        }
    }
}

impl<T> AsRef<[T]> for MappedView<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T: fmt::Debug> fmt::Debug for MappedView<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedView")
            .field("offset", &self.start)
            .field("elements", &self.count)
            .field("data", &&**self)
            .finish()
    }
}

/// Every reason a dataset can and cannot be viewed in place, and what a view
/// keeps alive.
///
/// The differential cases pin the contract that matters: a view holds
/// *exactly* what the copying read holds, bit for bit. The refusal cases pin
/// the other half — that a dataset which cannot be viewed says so by name
/// instead of quietly copying.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{borrow_inner_mut, H5FileInner};
    use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
    use crate::{FileSpaceStrategy, H5Dataset, H5File, Hdf5Error};
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "hdf5_mapped_view_{}_{}_{}.h5",
            name,
            std::process::id(),
            n
        ))
    }

    /// The refusal a view of `ds` as `T` gives, or a panic naming what it
    /// returned instead.
    fn refusal<T: H5Type>(ds: &H5Dataset) -> ViewRefusal {
        match ds.read_mapped::<T>() {
            Err(Hdf5Error::NotViewable(reason)) => reason,
            Err(other) => panic!("expected a view refusal, got: {other}"),
            Ok(view) => panic!("expected a view refusal, got {} elements", view.len()),
        }
    }

    /// A view holds exactly what `read_raw` holds, for every shape and width
    /// a contiguous dataset comes in.
    #[test]
    fn a_view_matches_read_raw_bit_for_bit() {
        let path = temp_path("differential");
        let f64s: Vec<f64> = (0..24).map(|i| i as f64 * -0.5).collect();
        let i32s: Vec<i32> = (0..12).map(|i| i * 7 - 30).collect();
        let u8s: Vec<u8> = (0..24).map(|i| (i * 11) as u8).collect();
        let u16s: Vec<u16> = (0..6).map(|i| (i * 4097) as u16).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>()
                .shape([24usize])
                .create("flat")
                .unwrap()
                .write_raw(&f64s)
                .unwrap();
            file.new_dataset::<i32>()
                .shape([3usize, 4])
                .create("grid")
                .unwrap()
                .write_raw(&i32s)
                .unwrap();
            file.new_dataset::<u8>()
                .shape([2usize, 3, 4])
                .create("cube")
                .unwrap()
                .write_raw(&u8s)
                .unwrap();
            file.new_dataset::<u16>()
                .shape([1usize, 6])
                .create("row")
                .unwrap()
                .write_raw(&u16s)
                .unwrap();
            file.close().unwrap();
        }

        let file = H5File::open(&path).unwrap();
        for name in ["flat", "grid", "cube", "row"] {
            let ds = file.dataset(name).unwrap();
            match name {
                "flat" => {
                    let view = ds.read_mapped::<f64>().unwrap();
                    assert_eq!(&*view, &ds.read_raw::<f64>().unwrap()[..], "{name}");
                    assert_eq!(&*view, &f64s[..], "{name}");
                }
                "grid" => {
                    let view = ds.read_mapped::<i32>().unwrap();
                    assert_eq!(&*view, &ds.read_raw::<i32>().unwrap()[..], "{name}");
                    assert_eq!(&*view, &i32s[..], "{name}");
                }
                "cube" => {
                    let view = ds.read_mapped::<u8>().unwrap();
                    assert_eq!(&*view, &ds.read_raw::<u8>().unwrap()[..], "{name}");
                    assert_eq!(&*view, &u8s[..], "{name}");
                }
                _ => {
                    let view = ds.read_mapped::<u16>().unwrap();
                    assert_eq!(&*view, &ds.read_raw::<u16>().unwrap()[..], "{name}");
                    assert_eq!(&*view, &u16s[..], "{name}");
                }
            }
        }
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A view of a contiguous sub-range holds exactly what `read_slice`
    /// holds, and a range that would have to gather is refused.
    #[test]
    fn a_range_view_matches_read_slice_where_the_run_is_one_piece() {
        let path = temp_path("range");
        let vals: Vec<i32> = (0..20).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<i32>()
                .shape([4usize, 5])
                .create("grid")
                .unwrap()
                .write_raw(&vals)
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("grid").unwrap();

        // Whole rows, a partial span of one row, one element, and nothing:
        // each is a single run of the row-major image.
        for (starts, counts) in [
            (vec![0usize, 0], vec![4usize, 5]),
            (vec![1, 0], vec![2, 5]),
            (vec![2, 1], vec![1, 3]),
            (vec![3, 4], vec![1, 1]),
            (vec![1, 2], vec![0, 0]),
        ] {
            let view = ds.read_mapped_slice::<i32>(&starts, &counts).unwrap();
            let copied = ds.read_slice::<i32>(&starts, &counts).unwrap();
            assert_eq!(&*view, &copied[..], "{starts:?} {counts:?}");
        }

        // A sub-block of several rows steps over the columns it leaves out.
        let reason = match ds.read_mapped_slice::<i32>(&[1, 1], &[2, 3]) {
            Err(Hdf5Error::NotViewable(r)) => r,
            other => panic!("expected a refusal, got {:?}", other.map(|v| v.len())),
        };
        assert!(
            matches!(&reason, ViewRefusal::Range(m) if m.contains("not one contiguous run")),
            "unexpected refusal: {reason}"
        );

        // A range that is not a range of this dataset at all.
        assert!(matches!(
            ds.read_mapped_slice::<i32>(&[0], &[4]),
            Err(Hdf5Error::NotViewable(ViewRefusal::Range(_)))
        ));
        assert!(matches!(
            ds.read_mapped_slice::<i32>(&[0, 0], &[5, 5]),
            Err(Hdf5Error::NotViewable(ViewRefusal::Range(_)))
        ));

        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A chunked dataset, a compact one, and a virtual one are all refused by
    /// layout — each names what it is instead.
    #[test]
    fn a_layout_that_is_not_one_stretch_of_the_file_is_refused() {
        let path = temp_path("layout");
        let vals: Vec<f64> = (0..64).map(|i| i as f64).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>()
                .shape([64usize])
                .chunk(&[16])
                .create("chunked")
                .unwrap()
                .write_raw(&vals)
                .unwrap();
            file.new_dataset::<f64>()
                .shape([8usize])
                .compact()
                .create("compact")
                .unwrap()
                .write_raw(&vals[..8])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        assert_eq!(
            refusal::<f64>(&file.dataset("chunked").unwrap()),
            ViewRefusal::Layout("it is chunked")
        );
        assert_eq!(
            refusal::<f64>(&file.dataset("compact").unwrap()),
            ViewRefusal::Layout("its raw data is compact, stored inside the object header")
        );
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A dataset whose layout names no address has no image anywhere in the
    /// file, so there is nothing to point at.
    ///
    /// A NULL dataspace is how this crate's writer produces one — it
    /// allocates a fill-value-only dataset's storage at create time, so that
    /// shape cannot be built here. A dataset libhdf5 created with late
    /// allocation and never wrote carries the same undefined address and
    /// takes this same branch, which reads the address and nothing else.
    #[test]
    fn a_dataset_with_no_storage_is_refused() {
        let path = temp_path("unallocated");
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>().null().create("nothing").unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("nothing").unwrap();
        assert_eq!(refusal::<f64>(&ds), ViewRefusal::Unallocated);
        // The copying read still answers.
        assert_eq!(ds.read_raw::<f64>().unwrap(), Vec::<f64>::new());
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A view type that is not as wide as the stored element is refused
    /// before any byte is reinterpreted.
    #[test]
    fn a_view_type_of_the_wrong_width_is_refused() {
        let path = temp_path("width");
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>()
                .shape([8usize])
                .create("doubles")
                .unwrap()
                .write_raw(&[1.0f64; 8])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("doubles").unwrap();
        assert_eq!(
            refusal::<f32>(&ds),
            ViewRefusal::ElementSize { view: 4, stored: 8 }
        );
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A dataset stored in the foreign byte order is refused: reading it as
    /// `T` needs a per-element swap, which is a copy by definition.
    #[test]
    fn a_foreign_byte_order_dataset_is_refused() {
        let path = temp_path("byte_order");
        let vals: Vec<f64> = (0..8).map(|i| i as f64 + 0.25).collect();
        let mut be = DatatypeMessage::f64_type();
        if let DatatypeMessage::FloatingPoint { byte_order, .. } = &mut be {
            *byte_order = ByteOrder::BigEndian;
        }
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>()
                .shape([8usize])
                .datatype(be)
                .create("big")
                .unwrap()
                .write_raw(&vals)
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("big").unwrap();
        assert_eq!(
            refusal::<f64>(&ds),
            ViewRefusal::ElementImage("they are stored in the foreign byte order")
        );
        // The copying read swaps and succeeds, which is what the refusal
        // points the caller back to.
        assert_eq!(ds.read_raw::<f64>().unwrap(), vals);
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A dataset whose data lands at an offset `T` cannot be read from in
    /// place is refused rather than read through an unaligned pointer.
    ///
    /// Paged file-space allocation is what makes this reachable: it aligns to
    /// the page and to nothing else, exactly as `H5Pset_alignment`'s defaults
    /// leave libhdf5, so a byte dataset of odd length pushes whatever follows
    /// off an eight-byte boundary. This crate's unpaged allocator aligns every
    /// block to eight and cannot produce one.
    #[test]
    fn a_misaligned_data_offset_is_refused() {
        let path = temp_path("alignment");
        {
            let file = H5File::options()
                .file_space(FileSpaceStrategy::Page, true, 1)
                .file_space_page_size(512)
                .create(&path)
                .unwrap();
            file.new_dataset::<u8>()
                .shape([7usize])
                .create("shim")
                .unwrap()
                .write_raw(&[1u8; 7])
                .unwrap();
            file.new_dataset::<f64>()
                .shape([8usize])
                .create("doubles")
                .unwrap()
                .write_raw(&[2.0f64; 8])
                .unwrap();
            file.close().unwrap();
        }
        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("doubles").unwrap();
        let reason = refusal::<f64>(&ds);
        assert!(
            matches!(reason, ViewRefusal::Alignment { align: 8, offset } if offset % 8 != 0),
            "unexpected refusal: {reason}"
        );
        // The copying read does not care where the bytes start.
        assert_eq!(ds.read_raw::<f64>().unwrap(), vec![2.0f64; 8]);
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A file truncated below what its datasets claim maps short, and a view
    /// of an image running off the end is refused rather than built.
    ///
    /// Paged again, for the layout rather than the alignment: the raw-data
    /// page is the last thing in the file, so cutting the tail takes data
    /// bytes and leaves the metadata that names them intact — which is the
    /// state a corrupt or half-copied file arrives in.
    #[test]
    fn an_image_past_the_end_of_the_map_is_refused() {
        let path = temp_path("truncated");
        {
            let file = H5File::options()
                .file_space(FileSpaceStrategy::Page, true, 1)
                .file_space_page_size(512)
                .create(&path)
                .unwrap();
            file.new_dataset::<f64>()
                .shape([64usize])
                .create("doubles")
                .unwrap()
                .write_raw(&[3.0f64; 64])
                .unwrap();
            file.close().unwrap();
        }
        // Cut the tail off before anything maps it, so the map is simply
        // short — no live mapping is truncated under a reader.
        let full = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full - 64)
            .unwrap();

        let file = H5File::open(&path).unwrap();
        let ds = file.dataset("doubles").unwrap();
        assert_eq!(
            refusal::<f64>(&ds),
            ViewRefusal::PastMappedEnd {
                end: full,
                mapped: full - 64
            }
        );
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// A file that could not be mapped is refused by name. Reachable only
    /// when the OS declines the mapping, so it is asked of the decision
    /// itself rather than through an open.
    #[test]
    fn an_unmapped_file_is_refused() {
        let src = DatasetViewSource {
            map: None,
            storage: ViewStorage::Contiguous {
                offset: 2048,
                len: 64,
            },
            datatype: DatatypeMessage::f64_type(),
            dims: vec![8],
        };
        assert_eq!(
            view::<f64>(&src, ViewRange::Whole).err(),
            Some(ViewRefusal::NotMapped)
        );
    }

    /// A view is a snapshot that owns its pages: it outlives the dataset, the
    /// file, and a refresh that retakes the handle's map, while the refreshed
    /// handle goes on to read a file the old map does not cover.
    #[test]
    fn a_view_outlives_the_file_and_a_refresh_that_retakes_the_map() {
        let path = temp_path("snapshot");
        let first: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let second: Vec<f64> = (0..8).map(|i| 100.0 + i as f64).collect();
        {
            let file = H5File::create(&path).unwrap();
            file.new_dataset::<f64>()
                .shape([8usize])
                .create("first")
                .unwrap()
                .write_raw(&first)
                .unwrap();
            file.close().unwrap();
        }
        let mapped_len = std::fs::metadata(&path).unwrap().len();

        let file = H5File::options().no_locking().open(&path).unwrap();
        let ds = file.dataset("first").unwrap();
        let view = ds.read_mapped::<f64>().unwrap();

        // Grow the file behind the open reader: the new dataset's bytes are
        // past the end of the map the reader is holding.
        {
            let w = H5File::options().no_locking().open_rw(&path).unwrap();
            w.new_dataset::<f64>()
                .shape([8usize])
                .create("second")
                .unwrap()
                .write_raw(&second)
                .unwrap();
            w.close().unwrap();
        }
        assert!(std::fs::metadata(&path).unwrap().len() > mapped_len);
        assert!(
            file.dataset("second").is_err(),
            "reader saw the write early"
        );

        // Retake the map, which drops the handle's share of the old one.
        {
            let mut inner = borrow_inner_mut(&file.inner);
            let H5FileInner::Reader(reader) = &mut *inner else {
                panic!("not a reader");
            };
            reader.refresh().unwrap();
        }

        // The view still reads its own map...
        assert_eq!(&*view, &first[..]);
        // ...while the refreshed handle reads bytes that map never held.
        let fresh = file.dataset("second").unwrap();
        assert_eq!(&*fresh.read_mapped::<f64>().unwrap(), &second[..]);

        // And the view outlives every handle that produced it.
        drop(fresh);
        drop(ds);
        drop(file);
        assert_eq!(&*view, &first[..]);
        let _ = std::fs::remove_file(&path);
    }
}
