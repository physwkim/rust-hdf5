//! Filter pipeline message (type 0x0B).
//!
//! Describes a pipeline of data filters (compression, checksumming, etc.)
//! applied to chunk data.
//!
//! Version 2 binary layout:
//! ```text
//! Byte 0: version = 2
//! Byte 1: number of filters
//! For each filter:
//!   filter_id:     u16 LE
//!   [if filter_id >= 256: name_length: u16 LE, name: NUL-padded string]
//!   flags:         u16 LE
//!   num_cd_values: u16 LE
//!   cd_values:     num_cd_values * u32 LE
//! ```
//!
//! Version 1 binary layout (written by libhdf5 / h5py with the default
//! `libver` bounds):
//! ```text
//! Byte 0: version = 1
//! Byte 1: number of filters
//! Bytes 2..8: 6 reserved bytes
//! For each filter:
//!   filter_id:     u16 LE
//!   name_length:   u16 LE        (always present, name padded to a
//!                                 multiple of 8 bytes)
//!   flags:         u16 LE
//!   num_cd_values: u16 LE
//!   name:          name_length bytes (NUL-padded to a multiple of 8)
//!   cd_values:     num_cd_values * u32 LE
//!   [if num_cd_values is odd: 4 bytes padding]
//! ```

use crate::format::shuffle::{
    bitshuffle_block_into, bitunshuffle_block_into, trans_byte_elem, untrans_byte_elem,
    BitshuffleScratch,
};
use crate::format::{FormatError, FormatResult};

/// Well-known filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;
pub const FILTER_BZIP2: u16 = 307;
pub const FILTER_LZF: u16 = 32000;
pub const FILTER_BLOSC: u16 = 32001;
pub const FILTER_LZ4: u16 = 32004;
pub const FILTER_BSHUF: u16 = 32008;
/// Bitshuffle compression sub-option (`cd_values[4]`): apply LZ4 after the
/// bit transpose. Matches `BSHUF_H5_COMPRESS_LZ4` in the canonical filter.
pub const BSHUF_COMPRESS_LZ4: u32 = 2;
pub const FILTER_ZFP: u16 = 32013;
pub const FILTER_ZSTD: u16 = 32015;
pub const FILTER_JPEG: u16 = 32019;
pub const FILTER_BITGROOM: u16 = 32022;
pub const FILTER_BITROUND: u16 = 32023;
pub const FILTER_BLOSC2: u16 = 32026;

/// Filter flags (`H5Zpublic.h`). A mandatory filter must run; if it can't
/// (e.g. a compressor whose output would be larger than its input), the
/// write fails. An optional filter is silently skipped for that chunk
/// instead, with the skip recorded in the chunk's filter mask.
///
/// Every builtin and registered filter libhdf5 sets through its own
/// `H5Pset_*` convenience call (`H5Pset_deflate`, `H5Pset_shuffle`,
/// `H5Pset_szip`, `H5Pset_nbit`, `H5Pset_scaleoffset`) or through
/// `H5Pset_filter` for a dynamically loaded one (h5py's `filters.py`
/// `fill_dcpl`) uses `H5Z_FLAG_OPTIONAL`. `H5Pset_fletcher32` is the sole
/// exception: a checksum is meaningless if it can be skipped, so it is
/// `H5Z_FLAG_MANDATORY` (H5Pocpl.c).
pub const FLAG_OPTIONAL: u16 = 1;
/// See [`FLAG_OPTIONAL`]. Matches `H5Z_FLAG_MANDATORY`; used only by
/// `H5Pset_fletcher32`.
pub const FLAG_MANDATORY: u16 = 0;

/// The name libhdf5 registers a filter under, for the filters libhdf5
/// registers itself — the `H5Z_class2_t` name field of `H5Z_DEFLATE`,
/// `H5Z_SHUFFLE`, `H5Z_FLETCHER32`, `H5Z_SZIP`, `H5Z_NBIT` and
/// `H5Z_SCALEOFFSET`.
///
/// Only these. Everything at or above `H5Z_FILTER_RESERVED` (256,
/// H5Zpublic.h:84) is a third-party filter whose registered name belongs to
/// the plugin that registers it, and this crate does not know what the reader
/// on the other side will have loaded — a guessed name would be a claim about
/// someone else's plugin. `None` is what a version-1 pipeline writes as a
/// zero name length, exactly as `H5O__pline_encode` does when `H5Z_find`
/// resolves nothing.
fn registered_name(id: u16) -> Option<&'static str> {
    match id {
        FILTER_DEFLATE => Some("deflate"),
        FILTER_SHUFFLE => Some("shuffle"),
        FILTER_FLETCHER32 => Some("fletcher32"),
        FILTER_SZIP => Some("szip"),
        FILTER_NBIT => Some("nbit"),
        FILTER_SCALEOFFSET => Some("scaleoffset"),
        _ => None,
    }
}

/// A single filter in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// Filter identifier (1 = deflate, 2 = shuffle, etc.).
    pub id: u16,
    /// Filter flags. Bit 0: filter is optional (0 = mandatory).
    pub flags: u16,
    /// Client data values (filter-specific parameters).
    pub cd_values: Vec<u32>,
}

/// A pipeline of data filters applied to chunk data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterPipeline {
    /// Ordered list of filters in the pipeline.
    pub filters: Vec<Filter>,
}

impl FilterPipeline {
    /// Create a pipeline with a single deflate (gzip) filter.
    ///
    /// `level` is the compression level (0-9). A level of 0 means no
    /// compression, 9 is maximum compression. The HDF5 default is 6.
    pub fn deflate(level: u32) -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_DEFLATE,
                flags: FLAG_OPTIONAL, // H5Pset_deflate (H5Pocpl.c)
                cd_values: vec![level],
            }],
        }
    }

    /// Create a pipeline with a single shuffle filter.
    ///
    /// Shuffle reorders a chunk's bytes by their position within an element,
    /// gathering each element's first bytes together, then each element's
    /// second bytes, and so on. It compresses nothing by itself — its output
    /// is a permutation of its input, exactly as long — so it is normally the
    /// first stage before a compressor. `H5Pset_shuffle` sets it alone all the
    /// same, and libhdf5 stores the permuted bytes with no filter behind it.
    ///
    /// `element_size` is the size of each data element in bytes; the filter
    /// carries it as its one client-data value, since the raw chunk bytes do
    /// not say where the element boundaries are.
    pub fn shuffle(element_size: u32) -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_SHUFFLE,
                flags: FLAG_OPTIONAL, // H5Pset_shuffle (H5Pdcpl.c)
                cd_values: vec![element_size],
            }],
        }
    }

    /// Create a pipeline with shuffle + deflate for better compression.
    ///
    /// Shuffle reorders bytes by position within elements, then deflate
    /// compresses the shuffled stream. `element_size` is the size of
    /// each data element in bytes.
    pub fn shuffle_deflate(element_size: u32, level: u32) -> Self {
        Self {
            filters: vec![
                Filter {
                    id: FILTER_SHUFFLE,
                    flags: FLAG_OPTIONAL,
                    cd_values: vec![element_size],
                },
                Filter {
                    id: FILTER_DEFLATE,
                    flags: FLAG_OPTIONAL,
                    cd_values: vec![level],
                },
            ],
        }
    }

    /// Create a pipeline with a single LZ4 filter (registered filter 32004).
    pub fn lz4() -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_LZ4,
                flags: FLAG_OPTIONAL, // dynamically registered filter (H5Pset_filter)
                cd_values: vec![],
            }],
        }
    }

    /// Create a pipeline with a single Zstandard filter.
    ///
    /// `level` is the compression level (1-22, default 3).
    pub fn zstd(level: u32) -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_ZSTD,
                flags: FLAG_OPTIONAL, // dynamically registered filter (H5Pset_filter)
                cd_values: vec![level],
            }],
        }
    }

    /// Create a pipeline with a single bitshuffle filter (registered filter
    /// 32008), bit transpose only (no secondary compression).
    ///
    /// `element_size` is the size of each data element in bytes. The block
    /// size is left at the canonical default (`cd_values[3] = 0`). Output is
    /// byte-for-byte compatible with the canonical bitshuffle HDF5 filter, so
    /// files written this way are readable by h5py / libhdf5.
    pub fn bshuf(element_size: u32) -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_BSHUF,
                flags: FLAG_OPTIONAL, // dynamically registered filter (H5Pset_filter)
                // [major, minor, elem_size, block_size, compression]
                cd_values: vec![0, 0, element_size, 0, 0],
            }],
        }
    }

    /// Create a pipeline with a single bitshuffle filter that applies LZ4
    /// after the bit transpose (the "BSLZ4" encoding).
    ///
    /// `element_size` is the size of each data element in bytes. The block
    /// size is left at the canonical default. Output is byte-for-byte
    /// framing-compatible with the canonical bitshuffle HDF5 filter.
    pub fn bshuf_lz4(element_size: u32) -> Self {
        Self {
            filters: vec![Filter {
                id: FILTER_BSHUF,
                flags: FLAG_OPTIONAL, // dynamically registered filter (H5Pset_filter)
                cd_values: vec![0, 0, element_size, 0, BSHUF_COMPRESS_LZ4],
            }],
        }
    }

    /// Create a pipeline with a single N-bit filter (`H5Z_FILTER_NBIT`, id 5)
    /// for an atomic numeric datatype.
    ///
    /// `dt` is the element datatype to pack (typically a reduced-precision
    /// fixed-point type) and `d_nelmts` is the number of elements per chunk.
    /// The `cd_values` tree mirrors libhdf5's `H5Z__set_local_nbit` for a
    /// single atomic datatype:
    /// `[nparms, need_not_compress, d_nelmts, NBIT_ATOMIC, size, order,
    /// precision, offset]`. Pair this with
    /// [`DatasetBuilder::datatype`](crate::dataset::DatasetBuilder::datatype) so the
    /// stored datatype matches the filter parameters.
    ///
    /// Non-atomic datatypes are emitted with `need_not_compress = 1` (a
    /// pass-through), matching libhdf5's handling of types it does not pack.
    pub fn nbit(dt: &crate::format::messages::datatype::DatatypeMessage, d_nelmts: usize) -> Self {
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
        use crate::format::nbit_scaleoffset::{NBIT_ATOMIC, NBIT_ORDER_BE, NBIT_ORDER_LE};

        let (size, order, precision, offset) = match dt {
            DatatypeMessage::FixedPoint {
                size,
                byte_order,
                bit_offset,
                bit_precision,
                ..
            }
            | DatatypeMessage::FloatingPoint {
                size,
                byte_order,
                bit_offset,
                bit_precision,
                ..
            } => {
                let order = match byte_order {
                    ByteOrder::LittleEndian => NBIT_ORDER_LE,
                    ByteOrder::BigEndian => NBIT_ORDER_BE,
                };
                (*size, order, *bit_precision as u32, *bit_offset as u32)
            }
            // Non-atomic: full footprint, no packing.
            other => {
                let size = other.element_size();
                (size, NBIT_ORDER_LE, size * 8, 0)
            }
        };

        // A full-precision atomic (offset 0, precision == size*8) carries no
        // savings, so libhdf5 flags it pass-through; reduced precision sets
        // need_not_compress = 0 so the bit packing actually runs.
        let need_not_compress = u32::from(offset == 0 && precision == size * 8);

        Self {
            filters: vec![Filter {
                id: FILTER_NBIT,
                flags: FLAG_OPTIONAL, // H5Pset_nbit (H5Pdcpl.c)
                // [nparms, need_not_compress, d_nelmts, class, size, order,
                //  precision, offset] — total 8 (3 base + 5 atomic).
                cd_values: vec![
                    8,
                    need_not_compress,
                    d_nelmts as u32,
                    NBIT_ATOMIC,
                    size,
                    order,
                    precision,
                    offset,
                ],
            }],
        }
    }

    /// Create a pipeline with a single scale-offset filter
    /// (`H5Z_FILTER_SCALEOFFSET`, id 6) for an atomic numeric datatype.
    ///
    /// `d_nelmts` is the number of elements per chunk and `scale_factor` is
    /// what `H5Pset_scaleoffset` takes: for an integer datatype the minimum
    /// number of bits to store each offset (0 lets the filter work it out per
    /// chunk), for a floating-point datatype the number of decimal digits to
    /// keep. The `cd_values` layout mirrors `H5Z__set_local_scaleoffset`,
    /// which fills all 20 entries; the fill value is left at zero and flagged
    /// defined, which is what libhdf5 records for a dataset that never set
    /// one.
    ///
    /// Only fixed-point and floating-point datatypes can be scale-offset
    /// filtered; anything else returns `None`, as `H5Z__set_local_scaleoffset`
    /// errors on it.
    pub fn scaleoffset(
        dt: &crate::format::messages::datatype::DatatypeMessage,
        d_nelmts: usize,
        scale_factor: i32,
    ) -> Option<Self> {
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
        use crate::format::nbit_scaleoffset::{
            SO_CLS_FLOAT, SO_CLS_INTEGER, SO_FLOAT_DSCALE, SO_INT, SO_ORDER_BE, SO_ORDER_LE,
            SO_SGN_2, SO_SGN_NONE, SO_TOTAL_NPARMS,
        };

        let (scale_type, class, size, sign, byte_order) = match dt {
            DatatypeMessage::FixedPoint {
                size,
                byte_order,
                signed,
                ..
            } => (
                SO_INT,
                SO_CLS_INTEGER,
                *size,
                if *signed { SO_SGN_2 } else { SO_SGN_NONE },
                *byte_order,
            ),
            DatatypeMessage::FloatingPoint {
                size, byte_order, ..
            } => (SO_FLOAT_DSCALE, SO_CLS_FLOAT, *size, 0, *byte_order),
            _ => return None,
        };
        let order = match byte_order {
            ByteOrder::LittleEndian => SO_ORDER_LE,
            ByteOrder::BigEndian => SO_ORDER_BE,
        };

        let mut cd_values = vec![0u32; SO_TOTAL_NPARMS];
        cd_values[0] = scale_type;
        cd_values[1] = scale_factor as u32;
        cd_values[2] = d_nelmts as u32;
        cd_values[3] = class;
        cd_values[4] = size;
        cd_values[5] = sign;
        cd_values[6] = order;
        cd_values[7] = 1; // fill value defined, and it is the zero left below
        Some(Self {
            filters: vec![Filter {
                id: FILTER_SCALEOFFSET,
                flags: FLAG_OPTIONAL, // H5Pset_scaleoffset (H5Pdcpl.c)
                cd_values,
            }],
        })
    }

    /// Create an empty pipeline (no filters).
    pub fn none() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Encode at the version `format` calls for (`H5O_pline_ver_bounds`,
    /// H5Opline.c:85): version 1 in a classic file, version 2 otherwise.
    pub fn encode_for(&self, format: crate::format::ObjectFormat) -> Vec<u8> {
        match format.filter_pipeline_version() {
            1 => self.encode_v1(),
            _ => self.encode(),
        }
    }

    /// Encode as a version-1 filter pipeline message, the one libhdf5 writes
    /// at `H5F_LIBVER_EARLIEST`.
    ///
    /// Version 1 differs from version 2 in three ways, all of them alignment
    /// (`H5O__pline_encode`, H5Opline.c:280-350): six reserved bytes follow
    /// the filter count, every filter carries a name length whether or not it
    /// is one of libhdf5's own, and both the name and the client-data array
    /// are padded — the name to a multiple of 8 bytes, the array to an even
    /// number of values.
    ///
    /// The name written is the one the filter is registered under
    /// (`registered_name`), because that is what `H5O__pline_encode` reaches
    /// for when the message carries none of its own. A filter this crate has
    /// no registered name for gets a zero name length, which is what libhdf5
    /// writes when `H5Z_find` does not resolve the id either.
    pub fn encode_v1(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);

        buf.push(1);
        buf.push(self.filters.len() as u8);
        buf.extend_from_slice(&[0u8; 6]);

        for f in &self.filters {
            buf.extend_from_slice(&f.id.to_le_bytes());

            // The stored length counts the NUL terminator and is rounded up
            // to the 8-byte multiple the name is padded to (`H5O_ALIGN_OLD`,
            // H5Opkg.h:57).
            let name = registered_name(f.id);
            let name_len = name.map_or(0, |n| n.len() + 1);
            let padded_len = name_len.div_ceil(8) * 8;
            buf.extend_from_slice(&(padded_len as u16).to_le_bytes());

            buf.extend_from_slice(&f.flags.to_le_bytes());
            buf.extend_from_slice(&(f.cd_values.len() as u16).to_le_bytes());

            if let Some(name) = name {
                buf.extend_from_slice(name.as_bytes());
                buf.resize(buf.len() + (padded_len - name.len()), 0);
            }

            for &cd in &f.cd_values {
                buf.extend_from_slice(&cd.to_le_bytes());
            }
            if !f.cd_values.len().is_multiple_of(2) {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        buf
    }

    /// Encode as a version-2 filter pipeline message.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);

        // Version
        buf.push(2);
        // Number of filters
        buf.push(self.filters.len() as u8);

        for f in &self.filters {
            // Filter ID
            buf.extend_from_slice(&f.id.to_le_bytes());

            // For filter IDs >= 256 (user-defined), a name string follows.
            // Predefined filters (< 256) have no name.
            if f.id >= 256 {
                // Name length = 0 (no name for now)
                buf.extend_from_slice(&0u16.to_le_bytes());
            }

            // Flags
            buf.extend_from_slice(&f.flags.to_le_bytes());

            // Number of client data values
            buf.extend_from_slice(&(f.cd_values.len() as u16).to_le_bytes());

            // Client data values
            for &cd in &f.cd_values {
                buf.extend_from_slice(&cd.to_le_bytes());
            }
        }

        buf
    }

    /// Decode a filter pipeline message (version 1 or version 2).
    ///
    /// Version 1 is what libhdf5 / h5py writes with the default `libver`
    /// bounds; version 2 is written for `libver` >= V18.
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != 1 && version != 2 {
            return Err(FormatError::InvalidVersion(version));
        }

        let nfilters = buf[1] as usize;
        // libhdf5 caps a pipeline at H5Z_MAX_NFILTERS (32).
        if nfilters > 32 {
            return Err(FormatError::InvalidData(format!(
                "filter pipeline declares {nfilters} filters (max 32)"
            )));
        }
        let mut pos = 2;

        // Version 1 carries 6 reserved bytes after the filter count.
        if version == 1 {
            if buf.len() < pos + 6 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 6,
                    available: buf.len(),
                });
            }
            pos += 6;
        }

        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            // filter_id
            if buf.len() < pos + 2 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 2,
                    available: buf.len(),
                });
            }
            let id = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;

            // The name length field is always present in version 1; in
            // version 2 it is present only for user-defined filters
            // (id >= 256, i.e. >= H5Z_FILTER_RESERVED).
            let name_len = if version == 1 || id >= 256 {
                if buf.len() < pos + 2 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 2,
                        available: buf.len(),
                    });
                }
                let n = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                // libhdf5 (H5Opline.c) requires a version-1 filter name
                // length to be a multiple of 8.
                if version == 1 && !n.is_multiple_of(8) {
                    return Err(FormatError::InvalidData(format!(
                        "version-1 filter name length {n} is not a multiple of 8"
                    )));
                }
                n
            } else {
                0
            };

            // flags + num_cd_values
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let flags = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
            let num_cd = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
            pos += 2;

            // Filter name. Version 1 always pads the name to a multiple
            // of 8 bytes; version 2 stores it unpadded.
            if name_len > 0 {
                let stored_len = if version == 1 {
                    (name_len + 7) & !7
                } else {
                    name_len
                };
                if buf.len() < pos + stored_len {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + stored_len,
                        available: buf.len(),
                    });
                }
                pos += stored_len;
            }

            // cd_values
            if buf.len() < pos + num_cd * 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + num_cd * 4,
                    available: buf.len(),
                });
            }
            let mut cd_values = Vec::with_capacity(num_cd);
            for _ in 0..num_cd {
                let v = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
                pos += 4;
                cd_values.push(v);
            }

            // Version 1 pads the client-data array to an even count.
            if version == 1 && num_cd % 2 == 1 {
                if buf.len() < pos + 4 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 4,
                        available: buf.len(),
                    });
                }
                pos += 4;
            }

            filters.push(Filter {
                id,
                flags,
                cd_values,
            });
        }

        Ok((Self { filters }, pos))
    }
}

/// Apply filter pipeline to compress raw chunk data.
///
/// Returns the compressed data. If no filters are configured, returns the
/// input unchanged.
pub fn apply_filters(pipeline: &FilterPipeline, data: &[u8]) -> FormatResult<Vec<u8>> {
    let mut buf: Option<Vec<u8>> = None;
    for filter in &pipeline.filters {
        buf = Some(apply_single_filter(
            filter,
            buf.as_deref().unwrap_or(data),
            true,
        )?);
    }
    // Only an empty pipeline leaves `buf` unset, and only then is a copy of
    // the input the answer; every stage already produces a fresh buffer.
    Ok(buf.unwrap_or_else(|| data.to_vec()))
}

/// Reverse the filter pipeline, skipping any filter whose bit is set in
/// `filter_mask`.
///
/// Filters run in reverse pipeline order. A set bit *i* means filter *i*
/// (counted in forward pipeline order) was **not** applied to this chunk on
/// write — e.g. an HDF5 direct chunk write (`H5Dwrite_chunk`) that handed in
/// already-filtered bytes, or an incompressible chunk libhdf5 chose to store
/// raw — so its reverse must be skipped here too. The mask addresses only the
/// first 32 filters (the HDF5 limit); a position beyond that cannot be marked
/// skipped and is always applied.
pub fn reverse_filters_masked(
    pipeline: &FilterPipeline,
    data: &[u8],
    filter_mask: u32,
) -> FormatResult<Vec<u8>> {
    let mut buf: Option<Vec<u8>> = None;
    for (i, filter) in pipeline.filters.iter().enumerate().rev() {
        if i < 32 && filter_mask & (1u32 << i) != 0 {
            continue;
        }
        buf = Some(apply_single_filter(
            filter,
            buf.as_deref().unwrap_or(data),
            false,
        )?);
    }
    // Reached only when every filter was masked off (or the pipeline is
    // empty): the stored bytes are already the chunk's data.
    Ok(buf.unwrap_or_else(|| data.to_vec()))
}

/// Reverse filter pipeline to decompress raw chunk data (the full pipeline,
/// no per-chunk mask). Equivalent to [`reverse_filters_masked`] with mask 0.
pub fn reverse_filters(pipeline: &FilterPipeline, data: &[u8]) -> FormatResult<Vec<u8>> {
    reverse_filters_masked(pipeline, data, 0)
}

/// Reverse the filter pipeline straight into `out`, returning the length of
/// the image it produced.
///
/// A chunk's decoded image is frequently one contiguous stretch of the read's
/// output — for a whole chunk of a full read it always is — and then the image
/// *is* those output bytes: the last reverse stage writes them where they
/// belong and no staging buffer is allocated, filled and copied out. Stages
/// before the last still materialize their own buffer; only the last one has a
/// destination the caller can name.
///
/// The returned length is the whole image the pipeline produced, which may
/// exceed `out.len()` — the surplus is discarded, exactly as a caller copying
/// `out.len()` bytes out of a materialized image discards it. A length below
/// `out.len()` means the stored chunk decoded short of its image; `out` past
/// that length is then left **unspecified**, because the inflate writes its
/// output in blocks and the last one can run past the byte the stream ends on.
/// A caller whose destination may outlast the image owns every byte of it,
/// not just the tail past the image.
pub fn reverse_filters_masked_into(
    pipeline: &FilterPipeline,
    data: &[u8],
    filter_mask: u32,
    out: &mut [u8],
) -> FormatResult<usize> {
    let masked = |i: usize| i < 32 && filter_mask & (1u32 << i) != 0;
    // The last stage to run is the lowest-numbered filter still in play; it is
    // the one whose output is the image, so it is the one that writes to `out`.
    let Some(last) = (0..pipeline.filters.len()).find(|i| !masked(*i)) else {
        // Every filter masked off (or an empty pipeline): the stored bytes are
        // already the chunk's image.
        return Ok(copy_image_into(data, out));
    };
    let mut buf: Option<Vec<u8>> = None;
    for (i, filter) in pipeline.filters.iter().enumerate().rev() {
        if masked(i) || i == last {
            continue;
        }
        buf = Some(apply_single_filter(
            filter,
            buf.as_deref().unwrap_or(data),
            false,
        )?);
    }
    reverse_single_filter_into(&pipeline.filters[last], buf.as_deref().unwrap_or(data), out)
}

/// Copy an already-decoded image into a caller's destination, returning the
/// image's own length so a short image is visible to the caller as one.
fn copy_image_into(image: &[u8], out: &mut [u8]) -> usize {
    let n = image.len().min(out.len());
    out[..n].copy_from_slice(&image[..n]);
    image.len()
}

/// Reverse one filter into `out`, returning the length of the image it
/// produced (which may exceed `out.len()`; see
/// [`reverse_filters_masked_into`]).
///
/// Only the inflate has a decoder that can be pointed at a caller's buffer;
/// every other filter builds its image and copies it, which is what the caller
/// would have done with it anyway.
fn reverse_single_filter_into(filter: &Filter, data: &[u8], out: &mut [u8]) -> FormatResult<usize> {
    #[cfg(feature = "deflate")]
    if filter.id == FILTER_DEFLATE {
        return inflate_zlib_into(data, out);
    }
    Ok(copy_image_into(
        &apply_single_filter(filter, data, false)?,
        out,
    ))
}

/// Apply the shuffle filter (byte transposition).
///
/// For elements of size `bytesoftype`, the shuffle gathers all first bytes,
/// then all second bytes, etc. This improves subsequent compression ratios
/// because bytes at the same position within elements tend to be correlated.
fn shuffle(data: &[u8], bytesoftype: usize) -> Vec<u8> {
    if bytesoftype <= 1 || data.len() <= bytesoftype {
        return data.to_vec();
    }
    let numofelements = data.len() / bytesoftype;
    let total = numofelements * bytesoftype;
    let mut dest = vec![0u8; data.len()];
    trans_byte_elem(data, &mut dest, numofelements, bytesoftype);
    // Copy any leftover bytes unchanged
    if data.len() > total {
        dest[total..].copy_from_slice(&data[total..]);
    }
    dest
}

/// Reverse the shuffle filter (byte de-transposition).
fn unshuffle(data: &[u8], bytesoftype: usize) -> Vec<u8> {
    if bytesoftype <= 1 || data.len() <= bytesoftype {
        return data.to_vec();
    }
    let numofelements = data.len() / bytesoftype;
    let total = numofelements * bytesoftype;
    let mut dest = vec![0u8; data.len()];
    untrans_byte_elem(data, &mut dest, numofelements, bytesoftype);
    if data.len() > total {
        dest[total..].copy_from_slice(&data[total..]);
    }
    dest
}

/// Inflate one zlib stream, growing a single output buffer.
///
/// The engine is zlib-rs rather than the miniz_oxide behind `flate2`: on the
/// same level-6 streams it inflates 1.2x faster on incompressible chunks and
/// 1.7x faster on text-like ones. Compression stays on miniz_oxide
/// ([`apply_single_filter`]) because zlib-rs's deflate below level 9 gives up
/// a large part of the ratio on periodic data.
///
/// `Read::read_to_end` was the obvious spelling but the wrong one for chunk
/// data: it grows the buffer up from nothing and re-enters the decoder once
/// per growth step, which on a 2 MiB chunk that compresses well costs about
/// as much as the inflate itself.
///
/// `expected` is the uncompressed length when the caller knows it — blosc
/// records one in its own header. The deflate filter does not: HDF5 keeps the
/// uncompressed chunk size nowhere the filter can see, so its buffer starts at
/// the next power of two above the compressed length and doubles. Chunk sizes
/// are themselves powers of two often enough that this usually lands on the
/// exact size in two or three steps; doubling from the compressed length
/// instead overshoots by up to 2x and cost more than `read_to_end` did.
#[cfg(feature = "deflate")]
fn inflate_zlib(data: &[u8], expected: Option<usize>) -> FormatResult<Vec<u8>> {
    use zlib_rs::{Inflate, InflateFlush, Status};

    let err = |what: &str| FormatError::InvalidData(format!("deflate decompress error: {what}"));
    // 15 is the largest LZ77 window; a stream that declares a smaller one in
    // its zlib header still inflates against it.
    let mut inflate = Inflate::new(true, 15);
    let start = expected
        .filter(|n| *n > 0)
        .unwrap_or_else(|| data.len().max(4096).next_power_of_two());
    let mut out = vec![0u8; start];
    loop {
        let consumed = inflate.total_in() as usize;
        let filled = inflate.total_out() as usize;
        if filled == out.len() {
            out.resize(out.len() * 2, 0);
        }
        let status = inflate
            .decompress(&data[consumed..], &mut out[filled..], InflateFlush::NoFlush)
            .map_err(|e| err(e.as_str()))?;
        if status == Status::StreamEnd {
            break;
        }
        // Neither side moved with output still to spare: the stored bytes end
        // before the stream does.
        if inflate.total_in() as usize == consumed && inflate.total_out() as usize == filled {
            return Err(err("truncated stream"));
        }
    }
    out.truncate(inflate.total_out() as usize);
    Ok(out)
}

/// Bytes of scratch the overflow drain works through when a stream turns out
/// to be longer than the destination the caller named. Allocated only when
/// that happens, which for a chunk whose image is the size the layout says it
/// is never does.
#[cfg(feature = "deflate")]
const INFLATE_DRAIN_BYTES: usize = 32 * 1024;

/// Inflate one zlib stream straight into `out`, returning the stream's whole
/// decompressed length.
///
/// The destination is fixed, so nothing is allocated, grown or copied: the
/// [`inflate_zlib`] growth loop exists only because that spelling has no
/// destination to write to. A stream longer than `out` keeps inflating through
/// a small scratch buffer and the surplus is dropped — so a stream that is
/// malformed past the caller's window still fails here, as it does when
/// `inflate_zlib` grows a buffer to the stream's end.
///
/// `out` past the returned length is left unspecified: the decoder copies
/// output in blocks and the block that finishes the stream may reach past the
/// stream's last byte, into slack the caller offered.
#[cfg(feature = "deflate")]
fn inflate_zlib_into(data: &[u8], out: &mut [u8]) -> FormatResult<usize> {
    use zlib_rs::{Inflate, InflateFlush, Status};

    let err = |what: &str| FormatError::InvalidData(format!("deflate decompress error: {what}"));
    let mut inflate = Inflate::new(true, 15);
    let mut drain: Vec<u8> = Vec::new();
    let mut filled = 0usize;
    let mut produced = 0usize;
    loop {
        let consumed = inflate.total_in() as usize;
        let before = inflate.total_out();
        let status = if filled < out.len() {
            inflate.decompress(&data[consumed..], &mut out[filled..], InflateFlush::NoFlush)
        } else {
            if drain.is_empty() {
                drain = vec![0u8; INFLATE_DRAIN_BYTES];
            }
            inflate.decompress(&data[consumed..], &mut drain, InflateFlush::NoFlush)
        }
        .map_err(|e| err(e.as_str()))?;
        let made = (inflate.total_out() - before) as usize;
        if filled < out.len() {
            filled += made;
        }
        produced += made;
        if status == Status::StreamEnd {
            break;
        }
        // Neither side moved with output still to spare: the stored bytes end
        // before the stream does.
        if inflate.total_in() as usize == consumed && made == 0 {
            return Err(err("truncated stream"));
        }
    }
    Ok(produced)
}

fn apply_single_filter(filter: &Filter, data: &[u8], compress: bool) -> FormatResult<Vec<u8>> {
    match filter.id {
        #[cfg(feature = "deflate")]
        FILTER_DEFLATE => {
            if compress {
                use flate2::write::ZlibEncoder;
                use flate2::Compression;
                use std::io::Write;

                let level = filter.cd_values.first().copied().unwrap_or(6);
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
                encoder.write_all(data).map_err(|e| {
                    FormatError::InvalidData(format!("deflate compress error: {}", e))
                })?;
                encoder
                    .finish()
                    .map_err(|e| FormatError::InvalidData(format!("deflate finish error: {}", e)))
            } else {
                inflate_zlib(data, None)
            }
        }
        #[cfg(not(feature = "deflate"))]
        FILTER_DEFLATE => Err(FormatError::UnsupportedFeature(
            "deflate filter requires the 'deflate' feature".into(),
        )),
        FILTER_SHUFFLE => {
            // cd_values[0] = bytesoftype (element size)
            let bytesoftype = filter.cd_values.first().copied().unwrap_or(1) as usize;
            if compress {
                Ok(shuffle(data, bytesoftype))
            } else {
                Ok(unshuffle(data, bytesoftype))
            }
        }
        FILTER_FLETCHER32 => {
            if compress {
                // Fletcher-32 appends a 4-byte checksum trailer. libhdf5
                // (H5Zfletcher32.c) writes it with `UINT32ENCODE`, which is
                // little-endian, over the single u32 returned by
                // H5_checksum_fletcher32 — i.e. `cksum.to_le_bytes()`.
                let cksum = fletcher32(data);
                let mut out = data.to_vec();
                out.extend_from_slice(&cksum.to_le_bytes());
                Ok(out)
            } else {
                // Strip the trailing 4-byte checksum
                if data.len() < 4 {
                    return Err(FormatError::InvalidData(
                        "fletcher32: data too short for checksum".into(),
                    ));
                }
                Ok(data[..data.len() - 4].to_vec())
            }
        }
        FILTER_SZIP => {
            // cd_values layout per H5Zpublic.h: index 0 = options_mask,
            // 1 = pixels_per_block, 2 = bits_per_pixel, 3 = pixels_per_scanline.
            // libhdf5's H5Zszip.c stores exactly 4 cd_values. On the wire it
            // prepends a 4-byte little-endian header holding the uncompressed
            // length (UINT32ENCODE) ahead of the raw AEC bitstream, and reads
            // it back with UINT32DECODE on decompress.
            let options_mask = filter.cd_values.first().copied().unwrap_or(0);
            let pixels_per_block = filter.cd_values.get(1).copied().unwrap_or(32);
            let bits_per_pixel = filter.cd_values.get(2).copied().unwrap_or(8);
            let pixels_per_scanline = filter.cd_values.get(3).copied().unwrap_or(256);
            if compress {
                let compressed = crate::format::szip::compress(
                    data,
                    bits_per_pixel,
                    pixels_per_block,
                    pixels_per_scanline,
                    options_mask,
                )
                .map_err(|e| FormatError::InvalidData(format!("SZIP compress: {}", e)))?;
                // Prepend the 4-byte LE uncompressed-length header.
                let mut out = Vec::with_capacity(compressed.len() + 4);
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(&compressed);
                Ok(out)
            } else {
                if data.len() < 4 {
                    return Err(FormatError::InvalidData(
                        "SZIP: data too short for length header".into(),
                    ));
                }
                // Read the 4-byte LE uncompressed-length header.
                let out_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                crate::format::szip::decompress(
                    &data[4..],
                    out_size,
                    bits_per_pixel,
                    pixels_per_block,
                    pixels_per_scanline,
                    options_mask,
                )
                .map_err(|e| FormatError::InvalidData(format!("SZIP decompress: {}", e)))
            }
        }
        // =====================================================================
        // N-bit (5) — strips unused leading/trailing bits per the datatype's
        // precision/offset (H5Znbit.c). cd_values is the datatype parameter
        // tree; both directions are byte-exact with libhdf5.
        // =====================================================================
        FILTER_NBIT => {
            crate::format::nbit_scaleoffset::apply_nbit(data, &filter.cd_values, compress)
        }

        // =====================================================================
        // Scale-offset (6) — stores values as small integers relative to a
        // per-chunk minimum (H5Zscaleoffset.c). Both directions are byte-exact
        // with libhdf5.
        // =====================================================================
        FILTER_SCALEOFFSET => {
            if compress {
                crate::format::nbit_scaleoffset::forward_scaleoffset(data, &filter.cd_values)
            } else {
                crate::format::nbit_scaleoffset::reverse_scaleoffset(data, &filter.cd_values)
            }
        }
        // =====================================================================
        // LZ4 (32004) — C-compatible block framing: 8-byte BE orig_size +
        // 4-byte BE block_size, then per-block: 4-byte BE compressed_size + data
        // =====================================================================
        #[cfg(feature = "lz4")]
        FILTER_LZ4 => {
            if compress {
                let orig_size = data.len() as u64;
                let block_size = filter.cd_values.first().copied().unwrap_or(1 << 30) as usize;
                let block_size = std::cmp::min(block_size, data.len());
                let block_size = if block_size == 0 {
                    data.len()
                } else {
                    block_size
                };

                let mut out = Vec::with_capacity(12 + data.len());
                out.extend_from_slice(&orig_size.to_be_bytes());
                out.extend_from_slice(&(block_size as u32).to_be_bytes());

                let mut pos = 0;
                while pos < data.len() {
                    let end = std::cmp::min(pos + block_size, data.len());
                    let block = &data[pos..end];
                    let compressed = lz4_flex::compress(block);
                    if compressed.len() >= block.len() {
                        // Incompressible block: store uncompressed
                        out.extend_from_slice(&(block.len() as u32).to_be_bytes());
                        out.extend_from_slice(block);
                    } else {
                        out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
                        out.extend_from_slice(&compressed);
                    }
                    pos = end;
                }
                Ok(out)
            } else {
                if data.len() < 12 {
                    return Err(FormatError::InvalidData("LZ4: header too short".into()));
                }
                let orig_size = u64::from_be_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]) as usize;
                // Sanity check: orig_size should not exceed a reasonable limit.
                // HDF5 chunks are typically < 64MB.
                if orig_size > 64 * 1024 * 1024 {
                    return Err(FormatError::InvalidData(format!(
                        "LZ4: orig_size {} exceeds 64MB limit",
                        orig_size
                    )));
                }
                let mut block_size =
                    u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
                if block_size > orig_size {
                    block_size = orig_size;
                }

                let mut output = Vec::with_capacity(orig_size);
                let mut rpos = 12;
                while output.len() < orig_size {
                    if rpos + 4 > data.len() {
                        break;
                    }
                    let comp_size = u32::from_be_bytes([
                        data[rpos],
                        data[rpos + 1],
                        data[rpos + 2],
                        data[rpos + 3],
                    ]) as usize;
                    rpos += 4;
                    if rpos + comp_size > data.len() {
                        break;
                    }

                    let remaining = orig_size - output.len();
                    let cur_block = std::cmp::min(block_size, remaining);

                    if comp_size == cur_block {
                        // Uncompressed block
                        output.extend_from_slice(&data[rpos..rpos + comp_size]);
                    } else {
                        let decompressed =
                            lz4_flex::decompress(&data[rpos..rpos + comp_size], cur_block)
                                .map_err(|e| {
                                    FormatError::InvalidData(format!("LZ4 decompress: {}", e))
                                })?;
                        output.extend_from_slice(&decompressed);
                    }
                    rpos += comp_size;
                }
                Ok(output)
            }
        }
        #[cfg(not(feature = "lz4"))]
        FILTER_LZ4 => Err(FormatError::UnsupportedFeature(
            "LZ4 filter requires the 'lz4' feature".into(),
        )),

        // =====================================================================
        // ZSTD (32015) — Zstandard compression via pure Rust rust-zstd crate
        // =====================================================================
        #[cfg(feature = "zstd")]
        FILTER_ZSTD => {
            if compress {
                let level = filter.cd_values.first().copied().unwrap_or(3) as i32;
                Ok(rust_zstd::compress(data, level))
            } else {
                rust_zstd::decompress(data)
                    .map_err(|e| FormatError::InvalidData(format!("zstd decompress: {}", e)))
            }
        }
        #[cfg(not(feature = "zstd"))]
        FILTER_ZSTD => Err(FormatError::UnsupportedFeature(
            "ZSTD filter requires the 'zstd' feature".into(),
        )),

        // =====================================================================
        // BZIP2 (307) — raw bzip2 stream
        // =====================================================================
        #[cfg(feature = "bzip2")]
        FILTER_BZIP2 => {
            if compress {
                use bzip2::write::BzEncoder;
                use bzip2::Compression;
                use std::io::Write;
                let level = filter.cd_values.first().copied().unwrap_or(9);
                let mut enc = BzEncoder::new(Vec::new(), Compression::new(level));
                enc.write_all(data)
                    .map_err(|e| FormatError::InvalidData(format!("bzip2 compress: {}", e)))?;
                enc.finish()
                    .map_err(|e| FormatError::InvalidData(format!("bzip2 finish: {}", e)))
            } else {
                use bzip2::read::BzDecoder;
                use std::io::Read;
                let mut dec = BzDecoder::new(data);
                let mut out = Vec::new();
                dec.read_to_end(&mut out)
                    .map_err(|e| FormatError::InvalidData(format!("bzip2 decompress: {}", e)))?;
                Ok(out)
            }
        }
        #[cfg(not(feature = "bzip2"))]
        FILTER_BZIP2 => Err(FormatError::UnsupportedFeature(
            "BZIP2 requires 'bzip2_filter' feature".into(),
        )),

        // =====================================================================
        // LZF (32000) — raw lzf stream, no framing. Pure Rust implementation.
        // =====================================================================
        FILTER_LZF => {
            let chunk_size = filter.cd_values.get(2).copied().unwrap_or(0) as usize;
            if compress {
                Ok(lzf_compress(data))
            } else {
                let out_size = if chunk_size > 0 {
                    chunk_size
                } else {
                    data.len() * 4
                };
                lzf_decompress(data, out_size)
            }
        }

        // =====================================================================
        // Bitshuffle (32008) — bit-level transpose + optional LZ4/ZSTD
        // =====================================================================
        FILTER_BSHUF => {
            let elem_size = filter.cd_values.get(2).copied().unwrap_or(1) as usize;
            let block_size = filter.cd_values.get(3).copied().unwrap_or(0) as usize;
            let comp_type = filter.cd_values.get(4).copied().unwrap_or(0);
            if compress {
                bitshuffle_compress(data, elem_size, block_size, comp_type, filter)
            } else {
                bitshuffle_decompress(data, elem_size, block_size, comp_type)
            }
        }

        // =====================================================================
        // BitGroom (32022) — lossy float quantization (alternating shave/set)
        // =====================================================================
        FILTER_BITGROOM => {
            if compress {
                bitgroom_quantize(data, filter)
            } else {
                Ok(data.to_vec()) // no-op on decompress
            }
        }

        // =====================================================================
        // Granular BitRound (32023) — lossy float quantization (round-then-shave)
        // =====================================================================
        FILTER_BITROUND => {
            if compress {
                bitround_quantize(data, filter)
            } else {
                Ok(data.to_vec()) // no-op on decompress
            }
        }

        // =====================================================================
        // BLOSC (32001) — c-blosc 1.x frames
        // =====================================================================
        #[cfg(feature = "blosc")]
        FILTER_BLOSC => {
            if compress {
                blosc_compress(data, filter)
            } else {
                blosc_decompress(data)
            }
        }
        #[cfg(not(feature = "blosc"))]
        FILTER_BLOSC => Err(FormatError::UnsupportedFeature(
            "BLOSC requires 'blosc' feature".into(),
        )),

        other => Err(FormatError::UnsupportedFeature(format!(
            "filter id {}",
            other
        ))),
    }
}

/// Compute the Fletcher-32 checksum over a byte buffer, byte-exact with
/// libhdf5's `H5_checksum_fletcher32`.
///
/// The accumulators are *not* reduced modulo 65535 per word; instead they
/// are folded with `sum = (sum & 0xffff) + (sum >> 16)` after every group
/// of at most 360 words (the largest run that cannot overflow `u32`) and
/// twice at the end. Per-word `% 65535` is a different function — it maps
/// an exact `0xFFFF` partial sum to `0` where the fold keeps `0xFFFF`.
fn fletcher32(data: &[u8]) -> u32 {
    #[cfg(feature = "simd")]
    let (sum1, sum2, done) =
        fearless_simd::dispatch!(crate::simd::level(), s => fletcher_simd::runs(s, data));
    #[cfg(not(feature = "simd"))]
    let (sum1, sum2, done) = (0, 0, 0);
    fletcher32_from(sum1, sum2, done, data)
}

/// The run loop of [`fletcher32`] from byte `done`, a multiple of 720 (a
/// whole number of 360-word runs), with the accumulators as they stand
/// there.
fn fletcher32_from(mut sum1: u32, mut sum2: u32, done: usize, data: &[u8]) -> u32 {
    let mut words = (data.len() - done) / 2;
    let mut i = done;
    while words > 0 {
        let mut tlen = words.min(360);
        words -= tlen;
        while tlen > 0 {
            let word = ((data[i] as u32) << 8) | (data[i + 1] as u32);
            sum1 = sum1.wrapping_add(word);
            sum2 = sum2.wrapping_add(sum1);
            i += 2;
            tlen -= 1;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Odd trailing byte: contributes only its high byte.
    if data.len() % 2 == 1 {
        sum1 = sum1.wrapping_add((data[i] as u32) << 8);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Second reduction to fold the sums back into 16 bits.
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

/// The whole 360-word runs of [`fletcher32`] on `fearless_simd` lanes.
///
/// The scalar recurrence is wrapping `u32` arithmetic, a ring, so a run of
/// `m` words `w_0..w_m` starting from `(s1, s2)` has the closed form
/// `s1' = s1 + Σ w_i` and `s2' = s2 + m·s1 + Σ (m - i)·w_i`, and any
/// lane-parallel evaluation of those sums lands on the scalar's exact
/// value. The 360-word fold points are kept as they are.
#[cfg(feature = "simd")]
mod fletcher_simd {
    use fearless_simd::{prelude::*, Simd};
    use fearless_simd_macros::simd;

    const RUN_WORDS: usize = 360;

    #[inline(always)]
    fn hsum<S: Simd>(v: S::u32s) -> u32 {
        v.as_slice().iter().fold(0u32, |s, &x| s.wrapping_add(x))
    }

    /// Fold the whole runs at the front of `data`; returns `(sum1, sum2)`
    /// after them and the bytes they covered, a multiple of 720.
    ///
    /// Within a run the vector chunks of `L` words keep three lane
    /// accumulators: `a`, the chunk sums; `e`, the sum over chunks of the
    /// chunk sums before it; and `b`, the chunk words weighted by their
    /// lane index. With `m = C·L` words covered, `Σ (m - i)·w_i =
    /// L·Σa + L·Σe - Σb`, so the run's state is one scalar combine; the
    /// words a run has past its vector chunks (360 is not a multiple of
    /// every lane count) go through the scalar recurrence.
    #[simd]
    pub(super) fn runs<S: Simd>(simd: S, data: &[u8]) -> (u32, u32, usize) {
        let l = S::u16s::LEN;
        let chunks = RUN_WORDS / l;
        let vec_words = chunks * l;
        let k_lo = S::u32s::from_fn(simd, |k| k as u32);
        let k_hi = S::u32s::from_fn(simd, |k| (k + l / 2) as u32);
        let mut sum1 = 0u32;
        let mut sum2 = 0u32;
        let mut pos = 0;
        while pos + 2 * RUN_WORDS <= data.len() {
            let mut a = S::u32s::splat(simd, 0);
            let mut e = a;
            let mut b = a;
            for c in 0..chunks {
                let off = pos + 2 * c * l;
                let w = S::u16s::from_bytes(S::u8s::from_slice(simd, &data[off..off + 2 * l]));
                // Big-endian words on little-endian lanes.
                let w = (w << 8) | (w >> 8);
                let (lo, hi) = w.widen();
                e += a;
                a = a + lo + hi;
                b = b + lo * k_lo + hi * k_hi;
            }
            let (ha, he, hb) = (hsum::<S>(a), hsum::<S>(e), hsum::<S>(b));
            let l32 = l as u32;
            sum2 = sum2
                .wrapping_add((vec_words as u32).wrapping_mul(sum1))
                .wrapping_add(l32.wrapping_mul(ha))
                .wrapping_add(l32.wrapping_mul(he))
                .wrapping_sub(hb);
            sum1 = sum1.wrapping_add(ha);
            for i in vec_words..RUN_WORDS {
                let at = pos + 2 * i;
                let word = ((data[at] as u32) << 8) | (data[at + 1] as u32);
                sum1 = sum1.wrapping_add(word);
                sum2 = sum2.wrapping_add(sum1);
            }
            sum1 = (sum1 & 0xffff) + (sum1 >> 16);
            sum2 = (sum2 & 0xffff) + (sum2 >> 16);
            pos += 2 * RUN_WORDS;
        }
        (sum1, sum2, pos)
    }
}

/// Compress multiple chunks in parallel using rayon.
///
/// Each chunk is independently compressed through the filter pipeline. A
/// compression failure on any chunk is propagated (the whole call returns
/// `Err`), never silently substituted with the raw bytes: a caller that then
/// records the chunk under `filter_mask = 0` would claim the pipeline ran when
/// it did not, so the reader would try to reverse-filter raw data and corrupt
/// it. Serial writes (`apply_filters`) propagate the same way.
#[cfg(feature = "parallel")]
pub fn apply_filters_parallel(
    pipeline: &FilterPipeline,
    chunks: &[&[u8]],
) -> FormatResult<Vec<Vec<u8>>> {
    use rayon::prelude::*;
    // Run on rust-hdf5's private half-cores pool, not rayon's global pool; fall
    // back to serial if the pool could not be built.
    match crate::parallel::io_pool() {
        Some(pool) => pool.install(|| {
            chunks
                .par_iter()
                .map(|chunk| apply_filters(pipeline, chunk))
                .collect()
        }),
        None => chunks
            .iter()
            .map(|chunk| apply_filters(pipeline, chunk))
            .collect(),
    }
}

/// Decompress multiple chunks in parallel using rayon.
///
/// Each chunk is independently decompressed through the reversed filter
/// pipeline. A decompression failure on any chunk is propagated (the whole
/// call returns `Err`), never silently substituted with the still-compressed
/// bytes. Serial reads (`reverse_filters`) propagate the same way.
#[cfg(feature = "parallel")]
pub fn reverse_filters_parallel(
    pipeline: &FilterPipeline,
    chunks: &[Vec<u8>],
) -> FormatResult<Vec<Vec<u8>>> {
    use rayon::prelude::*;
    // Run on rust-hdf5's private half-cores pool, not rayon's global pool; fall
    // back to serial if the pool could not be built.
    match crate::parallel::io_pool() {
        Some(pool) => pool.install(|| {
            chunks
                .par_iter()
                .map(|chunk| reverse_filters(pipeline, chunk))
                .collect()
        }),
        None => chunks
            .iter()
            .map(|chunk| reverse_filters(pipeline, chunk))
            .collect(),
    }
}

// =========================================================================
// LZF — pure Rust implementation of Marc Lehmann's LZF compression
// =========================================================================

fn lzf_compress(input: &[u8]) -> Vec<u8> {
    // Simple LZF compressor. If compression doesn't help, return input unchanged.
    let len = input.len();
    let mut out = Vec::with_capacity(len);
    let mut htab = [0u32; 1 << 14];
    let mut ip = 0usize;
    let mut lit_start = 0usize; // index in `out` of the current literal length byte
    let mut lit = 0usize;
    out.push(0); // placeholder

    while ip < len {
        if len - ip < 3 {
            out.push(input[ip]);
            ip += 1;
            lit += 1;
            if lit == 32 {
                out[lit_start] = (lit - 1) as u8;
                lit = 0;
                lit_start = out.len();
                out.push(0);
            }
            continue;
        }

        let v = ((input[ip] as u32) << 8) | (input[ip + 1] as u32);
        let h = ((v >> 1) ^ (input[ip + 2] as u32)) & 0x3FFF;
        let r = htab[h as usize] as usize;
        htab[h as usize] = ip as u32;

        if r > 0
            && ip - r < (1 << 13)
            && r + 2 < len
            && ip + 2 < len
            && input[r] == input[ip]
            && input[r + 1] == input[ip + 1]
            && input[r + 2] == input[ip + 2]
        {
            if lit > 0 {
                out[lit_start] = (lit - 1) as u8;
                lit = 0;
            } else {
                out.pop();
            }

            let mut ml = 3;
            let max_len = std::cmp::min(len - ip, std::cmp::min(len - r, 264));
            while ml < max_len && input[r + ml] == input[ip + ml] {
                ml += 1;
            }

            let off = ip - r - 1;
            if ml <= 8 {
                out.push(((ml - 2) as u8) << 5 | (off >> 8) as u8);
                out.push((off & 0xFF) as u8);
            } else {
                out.push(7 << 5 | (off >> 8) as u8);
                out.push((ml - 9) as u8);
                out.push((off & 0xFF) as u8);
            }
            ip += ml;
            lit_start = out.len();
            out.push(0);
        } else {
            out.push(input[ip]);
            ip += 1;
            lit += 1;
            if lit == 32 {
                out[lit_start] = (lit - 1) as u8;
                lit = 0;
                lit_start = out.len();
                out.push(0);
            }
        }
    }

    if lit > 0 {
        out[lit_start] = (lit - 1) as u8;
    } else if !out.is_empty() {
        out.pop();
    }

    // Always return valid LZF — even if slightly larger, the format is correct
    out
}

/// Decode a raw LZF stream. `max_output` is the size the filter's
/// `cd_values[2]` declares, and like h5py's `lzf_filter.c` (which grows its
/// buffer on `E2BIG`) it is a sizing hint, not a bound on what the stream may
/// expand to; the stream itself bounds that, at 88 bytes of output per byte
/// of input (a 3-byte back-reference of 264 bytes), so a declared size past
/// what the input could produce sizes nothing.
fn lzf_decompress(input: &[u8], max_output: usize) -> FormatResult<Vec<u8>> {
    const MAX_EXPANSION: usize = 88;
    let reserve = max_output.min(input.len().saturating_mul(MAX_EXPANSION));
    let mut out = Vec::new();
    out.try_reserve_exact(reserve).map_err(|_| {
        FormatError::InvalidData(format!("LZF: cannot reserve {reserve} bytes of output"))
    })?;
    let mut ip = 0;

    while ip < input.len() {
        let ctrl = input[ip] as usize;
        ip += 1;

        if ctrl < 32 {
            // Literal run: ctrl+1 bytes
            let count = ctrl + 1;
            if ip + count > input.len() {
                return Err(FormatError::InvalidData(
                    "LZF: truncated literal run".into(),
                ));
            }
            out.extend_from_slice(&input[ip..ip + count]);
            ip += count;
        } else {
            // Back reference
            let len = ctrl >> 5;
            let ml = if len == 7 {
                if ip >= input.len() {
                    return Err(FormatError::InvalidData("LZF: truncated back-ref".into()));
                }
                let extra = input[ip] as usize;
                ip += 1;
                extra + 7 + 2
            } else {
                len + 2
            };

            if ip >= input.len() {
                return Err(FormatError::InvalidData("LZF: truncated offset".into()));
            }
            let off = ((ctrl & 0x1F) << 8) | (input[ip] as usize);
            ip += 1;

            if out.len() < off + 1 {
                return Err(FormatError::InvalidData(
                    "LZF: invalid back-ref offset".into(),
                ));
            }
            let ref_start = out.len() - off - 1;
            for i in 0..ml {
                out.push(out[ref_start + i]);
            }
        }
    }
    Ok(out)
}

// =========================================================================
// Bitshuffle — bit-level transpose
//
// Byte-for-byte compatible with the canonical bitshuffle HDF5 filter
// (32008, kiyo-masui/bitshuffle, as vendored in libhdf5 plugins and
// c-blosc). Two conventions are load-bearing for interop with h5py / C:
//
//   * Bit order is LSB-first in BOTH dimensions. Bit-plane `q` of an
//     element is `byte_within_elem * 8 + bit_within_byte` (LSB = bit 0),
//     and within an output plane the element bits are packed LSB-first.
//     (The earlier implementation used MSB-first, which produced files
//     libhdf5/h5py could not read and could not read theirs back.)
//   * The block transpose operates on whole blocks of `block_size`
//     elements (a multiple of 8). The trailing elements are split into a
//     final transposed block rounded down to a multiple of 8, plus a raw
//     `size % 8` leftover that is copied verbatim — see
//     `bshuf_block_plan` and the (de)compress framing below.
// =========================================================================

/// Canonical default block size in elements for `elem_size`-byte elements.
/// Mirrors `bshuf_default_block_size`: target 8 KiB, rounded down to a
/// multiple of 8, floored at 128.
fn bshuf_default_block_size(elem_size: usize) -> usize {
    let block = (8192 / elem_size / 8) * 8;
    std::cmp::max(block, 128)
}

/// Resolve the effective block size in elements: the user/header value
/// (0 => canonical default), rounded down to a multiple of 8 (block sizes
/// must be a multiple of 8), floored at 8.
fn bshuf_resolve_block_elems(block_size: usize, elem_size: usize) -> usize {
    let mut block = if block_size == 0 {
        bshuf_default_block_size(elem_size)
    } else {
        block_size
    };
    block = (block / 8) * 8;
    if block < 8 {
        block = 8;
    }
    block
}

/// Canonical block layout for `n_elems` elements split into `block_elems`
/// blocks: a sequence of bit-transposed block element-counts (full blocks
/// followed by an optional final block rounded down to a multiple of 8).
/// The elements not covered (`n_elems - sum`, always `n_elems % 8`) form
/// the raw leftover the caller copies verbatim. Mirrors
/// `bshuf_blocked_wrap_fun`.
fn bshuf_block_plan(n_elems: usize, block_elems: usize) -> Vec<usize> {
    let n_full = n_elems / block_elems;
    let mut blocks = vec![block_elems; n_full];
    let tail = n_elems % block_elems;
    let last_block = tail - tail % 8;
    if last_block > 0 {
        blocks.push(last_block);
    }
    blocks
}

/// Append the bit transpose of one planned block (`inverse` for the
/// untranspose) to `out`, in place in its tail.
fn push_bitshuffled(
    out: &mut Vec<u8>,
    block: &[u8],
    scratch: &mut BitshuffleScratch,
    elem_size: usize,
    inverse: bool,
) {
    let pos = out.len();
    out.resize(pos + block.len(), 0);
    if inverse {
        bitunshuffle_block_into(block, scratch, &mut out[pos..], elem_size);
    } else {
        bitshuffle_block_into(block, scratch, &mut out[pos..], elem_size);
    }
}

fn bitshuffle_compress(
    data: &[u8],
    elem_size: usize,
    block_size: usize,
    comp_type: u32,
    _filter: &Filter,
) -> FormatResult<Vec<u8>> {
    if elem_size == 0 {
        return Ok(data.to_vec());
    }
    let n_elems = data.len() / elem_size;
    let block_elems = bshuf_resolve_block_elems(block_size, elem_size);
    let blocks = bshuf_block_plan(n_elems, block_elems);
    let mut scratch = BitshuffleScratch::new(block_elems * elem_size);

    if comp_type == 0 {
        // Bitshuffle only, no compression, no header. Each planned block is
        // bit-transposed; the trailing `n_elems % 8` elements are copied raw.
        let mut out = Vec::with_capacity(data.len());
        let mut elem_pos = 0;
        for &block in &blocks {
            let start = elem_pos * elem_size;
            let end = start + block * elem_size;
            push_bitshuffled(&mut out, &data[start..end], &mut scratch, elem_size, false);
            elem_pos += block;
        }
        out.extend_from_slice(&data[elem_pos * elem_size..]);
        return Ok(out);
    }

    if comp_type != BSHUF_COMPRESS_LZ4 {
        return Err(FormatError::UnsupportedFeature(format!(
            "bitshuffle: unsupported compression type {comp_type}"
        )));
    }

    #[cfg(not(feature = "lz4"))]
    {
        Err(FormatError::UnsupportedFeature(
            "bitshuffle+LZ4 requires the 'lz4' feature".into(),
        ))
    }

    #[cfg(feature = "lz4")]
    {
        // 12-byte header: BE u64 total uncompressed bytes, BE u32 block bytes.
        let mut out = Vec::with_capacity(12 + data.len());
        out.extend_from_slice(&(data.len() as u64).to_be_bytes());
        out.extend_from_slice(&((block_elems * elem_size) as u32).to_be_bytes());

        // Each planned block: BE u32 LZ4-compressed size, then the raw LZ4
        // block of the bit-transposed data. Trailing `n_elems % 8` elements
        // are copied raw with no length prefix.
        let mut shuffled = vec![0u8; block_elems * elem_size];
        let mut elem_pos = 0;
        for &block in &blocks {
            let start = elem_pos * elem_size;
            let end = start + block * elem_size;
            let shuffled = &mut shuffled[..end - start];
            bitshuffle_block_into(&data[start..end], &mut scratch, shuffled, elem_size);
            let compressed = lz4_flex::compress(shuffled);
            out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
            out.extend_from_slice(&compressed);
            elem_pos += block;
        }
        out.extend_from_slice(&data[elem_pos * elem_size..]);
        Ok(out)
    }
}

fn bitshuffle_decompress(
    data: &[u8],
    elem_size: usize,
    block_size: usize,
    comp_type: u32,
) -> FormatResult<Vec<u8>> {
    if comp_type == 0 {
        // No compression and no header: the stream is the same length as the
        // original, so reconstruct the canonical block plan from its length
        // and the filter's block size, then bitunshuffle each block. The
        // trailing `n_elems % 8` elements are raw.
        if elem_size == 0 {
            return Ok(data.to_vec());
        }
        let n_elems = data.len() / elem_size;
        let block_elems = bshuf_resolve_block_elems(block_size, elem_size);
        let blocks = bshuf_block_plan(n_elems, block_elems);
        let mut scratch = BitshuffleScratch::new(block_elems * elem_size);
        let mut out = Vec::with_capacity(data.len());
        let mut elem_pos = 0;
        for &block in &blocks {
            let start = elem_pos * elem_size;
            let end = start + block * elem_size;
            push_bitshuffled(&mut out, &data[start..end], &mut scratch, elem_size, true);
            elem_pos += block;
        }
        out.extend_from_slice(&data[elem_pos * elem_size..]);
        return Ok(out);
    }

    if comp_type != BSHUF_COMPRESS_LZ4 {
        return Err(FormatError::UnsupportedFeature(format!(
            "bitshuffle: unsupported compression type {comp_type}"
        )));
    }

    #[cfg(not(feature = "lz4"))]
    {
        Err(FormatError::UnsupportedFeature(
            "bitshuffle+LZ4 requires the 'lz4' feature".into(),
        ))
    }

    #[cfg(feature = "lz4")]
    {
        if data.len() < 12 {
            return Err(FormatError::InvalidData(
                "bitshuffle: header too short".into(),
            ));
        }
        let orig_size = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;
        // Block size is read from the header, overriding the filter value
        // (matching the canonical filter), and must be a multiple of 8.
        let block_bytes = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
        if elem_size == 0 {
            return Err(FormatError::InvalidData(
                "bitshuffle: element size is zero".into(),
            ));
        }
        let block_elems = block_bytes / elem_size;
        if block_elems == 0 || !block_elems.is_multiple_of(8) {
            return Err(FormatError::InvalidData(
                "bitshuffle: invalid block size in header".into(),
            ));
        }
        // `orig_size` is file-derived; cap the pre-allocation so a hostile
        // header cannot drive an unbounded allocation.
        let mut output = Vec::with_capacity(orig_size.min(64 * 1024 * 1024));
        let mut rpos = 12;

        let n_elems = orig_size / elem_size;
        let mut scratch = BitshuffleScratch::new(block_bytes);
        let mut decompressed = vec![0u8; block_bytes];
        // Every block (full or final) is a length-prefixed LZ4 block; only the
        // `n_elems % 8` trailing elements are stored raw.
        for &block in &bshuf_block_plan(n_elems, block_elems) {
            if rpos + 4 > data.len() {
                return Err(FormatError::InvalidData(
                    "bitshuffle: truncated block header".into(),
                ));
            }
            let comp_size =
                u32::from_be_bytes([data[rpos], data[rpos + 1], data[rpos + 2], data[rpos + 3]])
                    as usize;
            rpos += 4;
            if rpos + comp_size > data.len() {
                return Err(FormatError::InvalidData(
                    "bitshuffle: truncated block data".into(),
                ));
            }
            let exp_size = block * elem_size;
            let decompressed = &mut decompressed[..exp_size];
            let got = lz4_flex::block::decompress_into(&data[rpos..rpos + comp_size], decompressed)
                .map_err(|e| FormatError::InvalidData(format!("bshuf LZ4: {}", e)))?;
            if got != exp_size {
                return Err(FormatError::InvalidData(format!(
                    "bshuf LZ4: block decompressed to {got} bytes, expected {exp_size}"
                )));
            }
            push_bitshuffled(&mut output, decompressed, &mut scratch, elem_size, true);
            rpos += comp_size;
        }

        // Trailing raw leftover: `orig_size - output.len()` bytes.
        if output.len() < orig_size {
            let remaining = orig_size - output.len();
            if rpos + remaining > data.len() {
                return Err(FormatError::InvalidData(
                    "bitshuffle: truncated leftover bytes".into(),
                ));
            }
            output.extend_from_slice(&data[rpos..rpos + remaining]);
        }
        Ok(output)
    }
}

// =========================================================================
// BitGroom (32022) — alternating shave/set quantization
// =========================================================================

fn bitgroom_quantize(data: &[u8], filter: &Filter) -> FormatResult<Vec<u8>> {
    let nsd = filter.cd_values.first().copied().unwrap_or(3) as usize;
    let datum_size = filter.cd_values.get(1).copied().unwrap_or(4) as usize;
    let has_mss = filter.cd_values.get(2).copied().unwrap_or(0) != 0;
    let mss_val_u32 = filter.cd_values.get(3).copied().unwrap_or(0);

    let prc_bnr_xct = nsd as f64 * std::f64::consts::LOG2_10;
    let prc_bnr_ceil = prc_bnr_xct.ceil() as usize;
    let prc_bnr_xpl_rqr = prc_bnr_ceil + 1;

    let mut out = data.to_vec();

    if datum_size == 4 {
        let bit_xpl_nbr_sgn: usize = 23;
        if prc_bnr_xpl_rqr >= bit_xpl_nbr_sgn {
            return Ok(out);
        }
        let bit_xpl_nbr_zro = bit_xpl_nbr_sgn - prc_bnr_xpl_rqr;
        let msk_zro: u32 = 0xFFFF_FFFFu32 << bit_xpl_nbr_zro;
        let msk_one: u32 = !msk_zro;

        let n = out.len() / 4;
        for i in 0..n {
            let off = i * 4;
            let mut val = u32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
            if has_mss && val == mss_val_u32 {
                continue;
            }
            if val == 0 {
                continue;
            } // skip zero
            if i % 2 == 0 {
                val &= msk_zro; // shave
            } else {
                val |= msk_one; // set
            }
            out[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    } else if datum_size == 8 {
        let bit_xpl_nbr_sgn: usize = 52;
        if prc_bnr_xpl_rqr >= bit_xpl_nbr_sgn {
            return Ok(out);
        }
        let bit_xpl_nbr_zro = bit_xpl_nbr_sgn - prc_bnr_xpl_rqr;
        let msk_zro: u64 = 0xFFFF_FFFF_FFFF_FFFFu64 << bit_xpl_nbr_zro;
        let msk_one: u64 = !msk_zro;

        let n = out.len() / 8;
        for i in 0..n {
            let off = i * 8;
            let mut val = u64::from_le_bytes([
                out[off],
                out[off + 1],
                out[off + 2],
                out[off + 3],
                out[off + 4],
                out[off + 5],
                out[off + 6],
                out[off + 7],
            ]);
            if val == 0 {
                continue;
            }
            if i % 2 == 0 {
                val &= msk_zro;
            } else {
                val |= msk_one;
            }
            out[off..off + 8].copy_from_slice(&val.to_le_bytes());
        }
    }
    Ok(out)
}

// =========================================================================
// Granular BitRound (32023) — per-element rounding quantization
// =========================================================================

fn bitround_quantize(data: &[u8], filter: &Filter) -> FormatResult<Vec<u8>> {
    let nsd = filter.cd_values.first().copied().unwrap_or(3) as i32;
    let datum_size = filter.cd_values.get(1).copied().unwrap_or(4) as usize;

    let mut out = data.to_vec();

    if datum_size == 4 {
        let n = out.len() / 4;
        for i in 0..n {
            let off = i * 4;
            let val = f32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
            if val == 0.0 || val.is_nan() || val.is_infinite() {
                continue;
            }

            let (mnt, xpn) = frexp_f32(val);
            let mnt_log10 = mnt.abs().log10();
            let dgt_nbr =
                ((xpn as f64) * std::f64::consts::LOG10_2 + mnt_log10 as f64).floor() as i32 + 1;
            let qnt_pwr = ((dgt_nbr - nsd) as f64 * std::f64::consts::LOG2_10).floor() as i32;
            let prc_rqr = ((xpn as f64 - (std::f64::consts::LOG2_10 * mnt_log10 as f64)).floor()
                as i32
                - qnt_pwr)
                .unsigned_abs() as usize;
            let prc_rqr = prc_rqr.saturating_sub(1);

            if prc_rqr >= 23 {
                continue;
            }
            let zro_bits = 23 - prc_rqr;
            let msk_zro: u32 = 0xFFFF_FFFFu32 << zro_bits;
            let msk_hshv: u32 = (!msk_zro) & (msk_zro >> 1);

            let mut u = u32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
            u = u.wrapping_add(msk_hshv);
            u &= msk_zro;
            out[off..off + 4].copy_from_slice(&u.to_le_bytes());
        }
    } else if datum_size == 8 {
        let n = out.len() / 8;
        for i in 0..n {
            let off = i * 8;
            let val = f64::from_le_bytes([
                out[off],
                out[off + 1],
                out[off + 2],
                out[off + 3],
                out[off + 4],
                out[off + 5],
                out[off + 6],
                out[off + 7],
            ]);
            if val == 0.0 || val.is_nan() || val.is_infinite() {
                continue;
            }

            let (mnt, xpn) = frexp_f64(val);
            let mnt_log10 = mnt.abs().log10();
            let dgt_nbr = ((xpn as f64) * std::f64::consts::LOG10_2 + mnt_log10).floor() as i32 + 1;
            let qnt_pwr = ((dgt_nbr - nsd) as f64 * std::f64::consts::LOG2_10).floor() as i32;
            let prc_rqr = ((xpn as f64 - std::f64::consts::LOG2_10 * mnt_log10).floor() as i32
                - qnt_pwr)
                .unsigned_abs() as usize;
            let prc_rqr = prc_rqr.saturating_sub(1);

            if prc_rqr >= 52 {
                continue;
            }
            let zro_bits = 52 - prc_rqr;
            let msk_zro: u64 = 0xFFFF_FFFF_FFFF_FFFFu64 << zro_bits;
            let msk_hshv: u64 = (!msk_zro) & (msk_zro >> 1);

            let mut u = u64::from_le_bytes([
                out[off],
                out[off + 1],
                out[off + 2],
                out[off + 3],
                out[off + 4],
                out[off + 5],
                out[off + 6],
                out[off + 7],
            ]);
            u = u.wrapping_add(msk_hshv);
            u &= msk_zro;
            out[off..off + 8].copy_from_slice(&u.to_le_bytes());
        }
    }
    Ok(out)
}

/// Pure Rust frexp for f32.
fn frexp_f32(x: f32) -> (f32, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 126;
    let mnt = f32::from_bits((bits & 0x807F_FFFF) | 0x3F00_0000);
    (mnt, exp)
}

/// Pure Rust frexp for f64.
fn frexp_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32 - 1022;
    let mnt = f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000);
    (mnt, exp)
}

// =========================================================================
// BLOSC (32001) — the c-blosc 1.x frame (blosc.c 1.12.0, format version 2)
//
// A frame is a 16-byte header, one u32 offset per block (bstarts), then the
// blocks. A block is shuffled whole, cut into `typesize` streams (BloscLZ
// and Snappy only) or kept as one, and each stream is stored as a u32
// length followed by the codec's output — or by the raw bytes when that
// length equals the stream's own. A MEMCPYED frame holds the input verbatim
// after the header. The codec of a frame is in its flags, bits 5-7.
// =========================================================================

/// Blosc compressor codes (cd_values[6]).
#[cfg(feature = "blosc")]
const BLOSC_BLOSCLZ: u32 = 0;
#[cfg(feature = "blosc")]
const BLOSC_LZ4: u32 = 1;
#[cfg(feature = "blosc")]
const BLOSC_LZ4HC: u32 = 2;
#[cfg(feature = "blosc")]
const BLOSC_SNAPPY: u32 = 3;
#[cfg(feature = "blosc")]
const BLOSC_ZLIB: u32 = 4;
#[cfg(feature = "blosc")]
const BLOSC_ZSTD: u32 = 5;

/// Compressor library formats, the header's flags bits 5-7 (LZ4HC writes LZ4).
#[cfg(feature = "blosc")]
const BLOSC_BLOSCLZ_FORMAT: u8 = 0;
#[cfg(feature = "blosc")]
const BLOSC_LZ4_FORMAT: u8 = 1;
#[cfg(feature = "blosc")]
const BLOSC_SNAPPY_FORMAT: u8 = 2;
#[cfg(feature = "blosc")]
const BLOSC_ZLIB_FORMAT: u8 = 3;
#[cfg(feature = "blosc")]
const BLOSC_ZSTD_FORMAT: u8 = 4;

#[cfg(feature = "blosc")]
const BLOSC_VERSION_FORMAT: u8 = 2;
/// Every compressor writes format version 1.
#[cfg(feature = "blosc")]
const BLOSC_CODEC_VERSION_FORMAT: u8 = 1;
#[cfg(feature = "blosc")]
const BLOSC_MAX_OVERHEAD: usize = 16;
#[cfg(feature = "blosc")]
const BLOSC_MAX_BUFFERSIZE: usize = i32::MAX as usize - BLOSC_MAX_OVERHEAD;
#[cfg(feature = "blosc")]
const BLOSC_MAX_TYPESIZE: usize = 255;
#[cfg(feature = "blosc")]
const BLOSC_MIN_BUFFERSIZE: usize = 128;
#[cfg(feature = "blosc")]
const BLOSC_MAX_SPLITS: usize = 16;
#[cfg(feature = "blosc")]
const BLOSC_L1: usize = 32 * 1024;

#[cfg(feature = "blosc")]
const BLOSC_DOSHUFFLE: u8 = 0x1;
#[cfg(feature = "blosc")]
const BLOSC_MEMCPYED: u8 = 0x2;
#[cfg(feature = "blosc")]
const BLOSC_DOBITSHUFFLE: u8 = 0x4;
#[cfg(feature = "blosc")]
const BLOSC_DONT_SPLIT: u8 = 0x10;

/// The library format a compressor code writes, or why it cannot be used.
#[cfg(feature = "blosc")]
fn blosc_compformat(compcode: u32) -> FormatResult<u8> {
    match compcode {
        BLOSC_BLOSCLZ => Ok(BLOSC_BLOSCLZ_FORMAT),
        BLOSC_LZ4 | BLOSC_LZ4HC => Ok(BLOSC_LZ4_FORMAT),
        BLOSC_SNAPPY => Ok(BLOSC_SNAPPY_FORMAT),
        #[cfg(feature = "deflate")]
        BLOSC_ZLIB => Ok(BLOSC_ZLIB_FORMAT),
        #[cfg(not(feature = "deflate"))]
        BLOSC_ZLIB => Err(FormatError::UnsupportedFeature(
            "blosc zlib sub-codec requires the 'deflate' feature".into(),
        )),
        #[cfg(feature = "zstd")]
        BLOSC_ZSTD => Ok(BLOSC_ZSTD_FORMAT),
        #[cfg(not(feature = "zstd"))]
        BLOSC_ZSTD => Err(FormatError::UnsupportedFeature(
            "blosc zstd sub-codec requires the 'zstd' feature".into(),
        )),
        other => Err(FormatError::UnsupportedFeature(format!(
            "blosc compressor code {}",
            other
        ))),
    }
}

/// `compute_blocksize` without a forced block size.
#[cfg(feature = "blosc")]
fn blosc_blocksize(compcode: u32, clevel: u32, typesize: usize, nbytes: usize) -> usize {
    if nbytes < typesize {
        return 1;
    }
    let mut blocksize = nbytes;
    if nbytes >= BLOSC_L1 {
        // The high-ratio codecs get blocks twice as large.
        let hcr = matches!(compcode, BLOSC_LZ4HC | BLOSC_ZLIB | BLOSC_ZSTD);
        let base = if hcr { 2 * BLOSC_L1 } else { BLOSC_L1 };
        blocksize = match clevel {
            0 => base / 4,
            1..=3 => base / 2,
            4 | 5 => base,
            6 => base * 2,
            7 => base * 4,
            8 => base * 8,
            _ => base * 16,
        };
    }
    blocksize = blocksize.min(nbytes);
    if blocksize > typesize {
        blocksize = blocksize / typesize * typesize;
    }
    blocksize
}

/// `split_block`: whether the compressor cuts each block into `typesize`
/// streams. c-blosc measured LZ4 faster unsplit.
#[cfg(feature = "blosc")]
fn blosc_split_block(compcode: u32, typesize: usize, blocksize: usize) -> bool {
    matches!(compcode, BLOSC_BLOSCLZ | BLOSC_SNAPPY)
        && typesize <= BLOSC_MAX_SPLITS
        && blocksize / typesize >= BLOSC_MIN_BUFFERSIZE
}

/// One frame's header fields and block geometry.
#[cfg(feature = "blosc")]
#[derive(Clone, Copy)]
struct BloscFrame {
    flags: u8,
    typesize: usize,
    nbytes: usize,
    blocksize: usize,
    nblocks: usize,
}

#[cfg(feature = "blosc")]
impl BloscFrame {
    fn new(flags: u8, typesize: usize, nbytes: usize, blocksize: usize) -> Self {
        Self {
            flags,
            typesize,
            nbytes,
            blocksize,
            nblocks: nbytes.div_ceil(blocksize),
        }
    }

    fn is_leftover(&self, j: usize) -> bool {
        j + 1 == self.nblocks && !self.nbytes.is_multiple_of(self.blocksize)
    }

    fn block_len(&self, j: usize) -> usize {
        if self.is_leftover(j) {
            self.nbytes % self.blocksize
        } else {
            self.blocksize
        }
    }

    /// The streams block `j` is stored as — `blosc_d`'s rule. For a frame
    /// c-blosc wrote it agrees with `blosc_c`'s, which only reads the
    /// DONT_SPLIT flag, because `split_block` set that flag.
    fn nsplits(&self, j: usize) -> usize {
        let split = self.flags & BLOSC_DONT_SPLIT == 0
            && self.typesize <= BLOSC_MAX_SPLITS
            && self.block_len(j) / self.typesize >= BLOSC_MIN_BUFFERSIZE
            && !self.is_leftover(j);
        if split {
            self.typesize
        } else {
            1
        }
    }

    /// Byte shuffle only means something for elements wider than a byte, and
    /// wins over the bit flag (`blosc_c`/`blosc_d` test it first).
    fn byte_shuffled(&self) -> bool {
        self.flags & BLOSC_DOSHUFFLE != 0 && self.typesize > 1
    }

    fn bit_shuffled(&self) -> bool {
        !self.byte_shuffled() && self.flags & BLOSC_DOBITSHUFFLE != 0
    }
}

/// c-blosc `shuffle` (`inverse`: `unshuffle`) of one block: the byte
/// transpose of its whole elements, a ragged tail copied as it is.
#[cfg(feature = "blosc")]
fn blosc_shuffle_block(typesize: usize, src: &[u8], dst: &mut [u8], inverse: bool) {
    let n = src.len() / typesize;
    if inverse {
        untrans_byte_elem(src, dst, n, typesize);
    } else {
        trans_byte_elem(src, dst, n, typesize);
    }
    dst[n * typesize..].copy_from_slice(&src[n * typesize..]);
}

/// c-blosc `bitshuffle` (`inverse`: `bitunshuffle`) of one block. A block
/// whose element count is not a multiple of 8 is copied untransposed. C
/// leaves a ragged tail past the last whole element unwritten; it is copied
/// here.
#[cfg(feature = "blosc")]
fn blosc_bitshuffle_block(
    typesize: usize,
    src: &[u8],
    dst: &mut [u8],
    scratch: &mut BitshuffleScratch,
    inverse: bool,
) {
    let n = src.len() / typesize;
    if !n.is_multiple_of(8) {
        dst.copy_from_slice(src);
        return;
    }
    let whole = n * typesize;
    if inverse {
        bitunshuffle_block_into(&src[..whole], scratch, dst, typesize);
    } else {
        bitshuffle_block_into(&src[..whole], scratch, dst, typesize);
    }
    dst[whole..].copy_from_slice(&src[whole..]);
}

/// The per-frame state of `blosc_c`: the shuffled block and each codec's
/// reusable working memory.
#[cfg(feature = "blosc")]
struct BloscCompressor {
    frame: BloscFrame,
    clevel: u32,
    compcode: u32,
    /// `get_accel`; only BloscLZ honours it here.
    accel: u32,
    tmp: Vec<u8>,
    bitshuffle: Option<BitshuffleScratch>,
    /// LZ4 and zstd write only into room for their worst case.
    staging: Vec<u8>,
    snappy: snap::raw::Encoder,
    #[cfg(feature = "deflate")]
    zlib: Option<flate2::Compress>,
}

#[cfg(feature = "blosc")]
impl BloscCompressor {
    fn new(frame: BloscFrame, clevel: u32, compcode: u32) -> Self {
        let typesize = frame.typesize;
        let accel = if clevel == 9 {
            1
        } else if compcode == BLOSC_BLOSCLZ && typesize.is_power_of_two() && typesize < 32 {
            32
        } else if compcode == BLOSC_LZ4 {
            10 - clevel
        } else {
            1
        };
        let shuffled = frame.byte_shuffled() || frame.bit_shuffled();
        Self {
            frame,
            clevel,
            compcode,
            accel,
            tmp: if shuffled {
                vec![0u8; frame.blocksize]
            } else {
                Vec::new()
            },
            bitshuffle: frame
                .bit_shuffled()
                .then(|| BitshuffleScratch::new(frame.blocksize)),
            staging: Vec::new(),
            snappy: snap::raw::Encoder::new(),
            #[cfg(feature = "deflate")]
            zlib: None,
        }
    }

    /// `blosc_c`: shuffle and compress block `j` into `dest`, the frame's
    /// room left, returning the bytes written or 0 when the block does not
    /// fit — the frame is then stored MEMCPYED.
    fn block(&mut self, j: usize, src: &[u8], dest: &mut [u8]) -> FormatResult<usize> {
        let frame = self.frame;
        let bsize = src.len();
        let src = if frame.byte_shuffled() {
            blosc_shuffle_block(frame.typesize, src, &mut self.tmp[..bsize], false);
            &self.tmp[..bsize]
        } else if let Some(scratch) = self.bitshuffle.as_mut() {
            blosc_bitshuffle_block(frame.typesize, src, &mut self.tmp[..bsize], scratch, false);
            &self.tmp[..bsize]
        } else {
            src
        };
        let nsplits = frame.nsplits(j);
        let neblock = bsize / nsplits;
        let mut pos = 0usize;
        for split in src.chunks_exact(neblock).take(nsplits) {
            pos += 4;
            let mut maxout = if self.compcode == BLOSC_SNAPPY {
                snap::raw::max_compress_len(neblock)
            } else {
                neblock
            };
            if pos + maxout > dest.len() {
                if pos >= dest.len() {
                    return Ok(0);
                }
                maxout = dest.len() - pos;
            }
            let mut cbytes = blosc_codec_compress(
                self.compcode,
                self.clevel,
                self.accel,
                split,
                &mut dest[pos..pos + maxout],
                &mut self.staging,
                &mut self.snappy,
                #[cfg(feature = "deflate")]
                &mut self.zlib,
            )?;
            // A stream as long as its input reads back as raw bytes, so one
            // that did not shrink is stored raw.
            if cbytes == 0 || cbytes == neblock {
                if pos + neblock > dest.len() {
                    return Ok(0);
                }
                dest[pos..pos + neblock].copy_from_slice(split);
                cbytes = neblock;
            }
            dest[pos - 4..pos].copy_from_slice(&(cbytes as u32).to_le_bytes());
            pos += cbytes;
        }
        Ok(pos)
    }
}

/// Compress one stream into `dst`, returning its length, or 0 when it does
/// not fit — c-blosc's `*_wrap_compress` with `maxout = dst.len()`.
#[cfg(feature = "blosc")]
#[allow(clippy::too_many_arguments)]
fn blosc_codec_compress(
    compcode: u32,
    clevel: u32,
    accel: u32,
    src: &[u8],
    dst: &mut [u8],
    staging: &mut Vec<u8>,
    snappy: &mut snap::raw::Encoder,
    #[cfg(feature = "deflate")] zlib: &mut Option<flate2::Compress>,
) -> FormatResult<usize> {
    let fit = |out: &[u8], dst: &mut [u8]| {
        if out.len() <= dst.len() {
            dst[..out.len()].copy_from_slice(out);
            out.len()
        } else {
            0
        }
    };
    match compcode {
        BLOSC_BLOSCLZ => Ok(blosclz_compress(clevel, src, dst, accel)),
        // LZ4HC streams are LZ4 streams; the LZ4 encoder writes them.
        BLOSC_LZ4 | BLOSC_LZ4HC => {
            staging.resize(lz4_flex::block::get_maximum_output_size(src.len()), 0);
            let n = lz4_flex::block::compress_into(src, staging)
                .map_err(|e| FormatError::InvalidData(format!("blosc lz4 compress: {e}")))?;
            Ok(fit(&staging[..n], dst))
        }
        // Like snappy-c, refuses a destination short of the worst case.
        BLOSC_SNAPPY => Ok(snappy.compress(src, dst).unwrap_or(0)),
        #[cfg(feature = "deflate")]
        BLOSC_ZLIB => {
            use flate2::{Compress, Compression, FlushCompress, Status};
            let z = zlib.get_or_insert_with(|| Compress::new(Compression::new(clevel), true));
            z.reset();
            match z.compress(src, dst, FlushCompress::Finish) {
                Ok(Status::StreamEnd) => Ok(z.total_out() as usize),
                _ => Ok(0),
            }
        }
        #[cfg(feature = "zstd")]
        BLOSC_ZSTD => {
            // zstd_wrap_compress; its level-8 adjustment tests the already
            // mapped (odd) level and never applies.
            let level = if clevel < 9 {
                clevel as i32 * 2 - 1
            } else {
                22
            };
            Ok(fit(&rust_zstd::compress(src, level), dst))
        }
        other => blosc_compformat(other).map(|_| 0),
    }
}

/// Decode one stream into all of `dst` — c-blosc's `*_wrap_decompress`
/// with `maxout = dst.len()`, which `blosc_d` requires to come out full.
#[cfg(feature = "blosc")]
fn blosc_codec_decompress(compformat: u8, src: &[u8], dst: &mut [u8]) -> FormatResult<()> {
    let bad = |codec: &str, e: &dyn std::fmt::Display| {
        FormatError::InvalidData(format!("blosc {codec} decompress: {e}"))
    };
    let n = match compformat {
        BLOSC_BLOSCLZ_FORMAT => {
            blosclz_decompress(src, dst).ok_or_else(|| bad("blosclz", &"corrupt stream"))?
        }
        BLOSC_LZ4_FORMAT => {
            lz4_flex::block::decompress_into(src, dst).map_err(|e| bad("lz4", &e))?
        }
        BLOSC_SNAPPY_FORMAT => snap::raw::Decoder::new()
            .decompress(src, dst)
            .map_err(|e| bad("snappy", &e))?,
        #[cfg(feature = "deflate")]
        BLOSC_ZLIB_FORMAT => inflate_zlib_into(src, dst)?,
        #[cfg(feature = "zstd")]
        BLOSC_ZSTD_FORMAT => {
            // rust-zstd decodes only into a buffer of its own.
            let v = rust_zstd::decompress(src).map_err(|e| bad("zstd", &e))?;
            if v.len() == dst.len() {
                dst.copy_from_slice(&v);
            }
            v.len()
        }
        #[cfg(not(feature = "deflate"))]
        BLOSC_ZLIB_FORMAT => {
            return Err(FormatError::UnsupportedFeature(
                "blosc zlib sub-codec requires the 'deflate' feature".into(),
            ))
        }
        #[cfg(not(feature = "zstd"))]
        BLOSC_ZSTD_FORMAT => {
            return Err(FormatError::UnsupportedFeature(
                "blosc zstd sub-codec requires the 'zstd' feature".into(),
            ))
        }
        other => {
            return Err(FormatError::UnsupportedFeature(format!(
                "blosc compressor format {other}"
            )))
        }
    };
    if n != dst.len() {
        return Err(FormatError::InvalidData(format!(
            "blosc: a stream decoded to {n} bytes, its block holds {}",
            dst.len()
        )));
    }
    Ok(())
}

/// `blosc_compress` for the HDF5 filter's `cd_values` (blosc_filter.c).
///
/// The destination has room for the header past the input, so input that
/// does not compress comes out as a MEMCPYED frame. The C filter offers only
/// `nbytes` and fails such a chunk instead, for libhdf5 to store unfiltered;
/// this crate's writers have no optional-filter skip to fall back on.
#[cfg(feature = "blosc")]
fn blosc_compress(data: &[u8], filter: &Filter) -> FormatResult<Vec<u8>> {
    let cd = &filter.cd_values;
    // Slot 2 is the type size; clevel, shuffle and compressor are optional.
    let typesize = cd.get(2).copied().unwrap_or(1) as usize;
    let clevel = cd.get(4).copied().unwrap_or(5);
    let doshuffle = cd.get(5).copied().unwrap_or(1);
    let compcode = cd.get(6).copied().unwrap_or(BLOSC_BLOSCLZ);
    let compformat = blosc_compformat(compcode)?;
    let bad = |what: String| FormatError::InvalidData(format!("blosc compress: {what}"));
    if clevel > 9 {
        return Err(bad(format!("clevel {clevel} is not 0..9")));
    }
    if doshuffle > 2 {
        return Err(bad(format!("shuffle {doshuffle} is not 0, 1 or 2")));
    }
    if typesize == 0 {
        return Err(bad("typesize 0".into()));
    }
    let nbytes = data.len();
    if nbytes > BLOSC_MAX_BUFFERSIZE {
        return Err(bad(format!("{nbytes} bytes exceed BLOSC_MAX_BUFFERSIZE")));
    }
    // A wider type is compressed as a stream of bytes.
    let typesize = if typesize > BLOSC_MAX_TYPESIZE {
        1
    } else {
        typesize
    };
    let blocksize = blosc_blocksize(compcode, clevel, typesize, nbytes);
    let mut flags = compformat << 5;
    if clevel == 0 || nbytes < BLOSC_MIN_BUFFERSIZE {
        flags |= BLOSC_MEMCPYED;
    }
    flags |= match doshuffle {
        1 => BLOSC_DOSHUFFLE,
        2 => BLOSC_DOBITSHUFFLE,
        _ => 0,
    };
    if !blosc_split_block(compcode, typesize, blocksize) {
        flags |= BLOSC_DONT_SPLIT;
    }
    let frame = BloscFrame::new(flags, typesize, nbytes, blocksize);

    let mut out = vec![0u8; nbytes + BLOSC_MAX_OVERHEAD];
    let mut ntbytes = 0;
    if flags & BLOSC_MEMCPYED == 0 {
        ntbytes = blosc_compress_blocks(frame, clevel, compcode, data, &mut out)?;
        if ntbytes == 0 {
            flags |= BLOSC_MEMCPYED;
        }
    }
    if flags & BLOSC_MEMCPYED != 0 {
        out[BLOSC_MAX_OVERHEAD..].copy_from_slice(data);
        ntbytes = out.len();
    }
    out[0] = BLOSC_VERSION_FORMAT;
    out[1] = BLOSC_CODEC_VERSION_FORMAT;
    out[2] = flags;
    out[3] = typesize as u8;
    out[4..8].copy_from_slice(&(nbytes as u32).to_le_bytes());
    out[8..12].copy_from_slice(&(blocksize as u32).to_le_bytes());
    out[12..16].copy_from_slice(&(ntbytes as u32).to_le_bytes());
    out.truncate(ntbytes);
    Ok(out)
}

/// `serial_blosc` compressing: bstarts and every block into `out`,
/// returning the frame's length, or 0 when it does not fit in `out`.
#[cfg(feature = "blosc")]
fn blosc_compress_blocks(
    frame: BloscFrame,
    clevel: u32,
    compcode: u32,
    data: &[u8],
    out: &mut [u8],
) -> FormatResult<usize> {
    let mut ntbytes = BLOSC_MAX_OVERHEAD + 4 * frame.nblocks;
    if ntbytes > out.len() {
        return Ok(0);
    }
    let mut compressor = BloscCompressor::new(frame, clevel, compcode);
    for j in 0..frame.nblocks {
        let at = BLOSC_MAX_OVERHEAD + 4 * j;
        out[at..at + 4].copy_from_slice(&(ntbytes as u32).to_le_bytes());
        let src = &data[j * frame.blocksize..][..frame.block_len(j)];
        let cbytes = compressor.block(j, src, &mut out[ntbytes..])?;
        if cbytes == 0 {
            return Ok(0);
        }
        ntbytes += cbytes;
    }
    Ok(ntbytes)
}

/// The per-frame state of `blosc_d`: where a shuffled block decodes before
/// its unshuffle.
#[cfg(feature = "blosc")]
struct BloscDecompressor {
    tmp: Vec<u8>,
    bitshuffle: Option<BitshuffleScratch>,
}

#[cfg(feature = "blosc")]
impl BloscDecompressor {
    fn new(frame: &BloscFrame) -> Self {
        let shuffled = frame.byte_shuffled() || frame.bit_shuffled();
        // A header's blocksize past nbytes makes one short block.
        let largest = frame.blocksize.min(frame.nbytes);
        Self {
            tmp: if shuffled {
                vec![0u8; largest]
            } else {
                Vec::new()
            },
            bitshuffle: frame
                .bit_shuffled()
                .then(|| BitshuffleScratch::new(largest)),
        }
    }

    /// `blosc_d`: decode block `j` of the frame `src` into `dest`, which is
    /// exactly the block's length.
    fn block(
        &mut self,
        frame: &BloscFrame,
        j: usize,
        src: &[u8],
        dest: &mut [u8],
    ) -> FormatResult<()> {
        let bad = |what: &str| FormatError::InvalidData(format!("blosc: block {j} {what}"));
        let bsize = dest.len();
        let nsplits = frame.nsplits(j);
        let neblock = bsize / nsplits;
        if neblock * nsplits != bsize {
            return Err(bad("is not a whole number of streams"));
        }
        let shuffled = frame.byte_shuffled() || frame.bit_shuffled();
        let mut pos = u32::from_le_bytes(src[16 + 4 * j..20 + 4 * j].try_into().unwrap()) as usize;
        {
            let target = if shuffled {
                &mut self.tmp[..bsize]
            } else {
                &mut *dest
            };
            for part in target.chunks_exact_mut(neblock) {
                let csize = src
                    .get(pos..pos + 4)
                    .ok_or_else(|| bad("runs past the frame"))?;
                let csize = i32::from_le_bytes(csize.try_into().unwrap());
                let csize =
                    usize::try_from(csize).map_err(|_| bad("has a negative stream length"))?;
                pos += 4;
                let stream = src
                    .get(pos..pos + csize)
                    .ok_or_else(|| bad("runs past the frame"))?;
                if csize == neblock {
                    part.copy_from_slice(stream);
                } else {
                    blosc_codec_decompress(frame.flags >> 5, stream, part)?;
                }
                pos += csize;
            }
        }
        if frame.byte_shuffled() {
            blosc_shuffle_block(frame.typesize, &self.tmp[..bsize], dest, true);
        } else if let Some(scratch) = self.bitshuffle.as_mut() {
            blosc_bitshuffle_block(frame.typesize, &self.tmp[..bsize], dest, scratch, true);
        }
        Ok(())
    }
}

/// The header of the frame `src` (`blosc_cbuffer_sizes`), checked against
/// the bytes it claims: a MEMCPYED body, or the bstarts table.
#[cfg(feature = "blosc")]
fn blosc_frame(src: &[u8]) -> FormatResult<BloscFrame> {
    let bad = |what: &str| FormatError::InvalidData(format!("blosc: {what}"));
    if src.len() < BLOSC_MAX_OVERHEAD {
        return Err(bad("header too short"));
    }
    let word = |at: usize| u32::from_le_bytes(src[at..at + 4].try_into().unwrap()) as usize;
    let (flags, typesize, nbytes, blocksize) = (src[2], src[3] as usize, word(4), word(8));
    if nbytes > BLOSC_MAX_BUFFERSIZE {
        return Err(bad("nbytes exceeds BLOSC_MAX_BUFFERSIZE"));
    }
    if flags & BLOSC_MEMCPYED != 0 || nbytes == 0 {
        if flags & BLOSC_MEMCPYED != 0 && src.len() < BLOSC_MAX_OVERHEAD + nbytes {
            return Err(bad("memcpy frame shorter than its nbytes"));
        }
        // nothing to decode block by block
        return Ok(BloscFrame {
            flags,
            typesize,
            nbytes,
            blocksize,
            nblocks: 0,
        });
    }
    if blocksize == 0 || typesize == 0 {
        return Err(bad("zero blocksize or typesize"));
    }
    let frame = BloscFrame::new(flags, typesize, nbytes, blocksize);
    if src.len() < BLOSC_MAX_OVERHEAD + 4 * frame.nblocks {
        return Err(bad("bstarts run past the frame"));
    }
    Ok(frame)
}

/// `blosc_decompress` straight into `out`, returning the frame's whole
/// uncompressed length (see [`reverse_filters_masked_into`]). Blocks past
/// the end of `out` still decode — into scratch — so a frame corrupt past
/// the caller's window fails here too.
#[cfg(feature = "blosc")]
fn blosc_decompress_into(src: &[u8], out: &mut [u8]) -> FormatResult<usize> {
    let frame = blosc_frame(src)?;
    if frame.flags & BLOSC_MEMCPYED != 0 {
        let body = &src[BLOSC_MAX_OVERHEAD..BLOSC_MAX_OVERHEAD + frame.nbytes];
        return Ok(copy_image_into(body, out));
    }
    let mut decompressor = BloscDecompressor::new(&frame);
    let mut spill = Vec::new();
    for j in 0..frame.nblocks {
        let lo = j * frame.blocksize;
        let len = frame.block_len(j);
        if lo + len <= out.len() {
            decompressor.block(&frame, j, src, &mut out[lo..lo + len])?;
        } else {
            spill.resize(len, 0);
            decompressor.block(&frame, j, src, &mut spill)?;
            if lo < out.len() {
                let n = out.len() - lo;
                out[lo..].copy_from_slice(&spill[..n]);
            }
        }
    }
    Ok(frame.nbytes)
}

/// `blosc_decompress` into a buffer of the length the header records.
#[cfg(feature = "blosc")]
fn blosc_decompress(data: &[u8]) -> FormatResult<Vec<u8>> {
    let mut out = vec![0u8; blosc_frame(data)?.nbytes];
    blosc_decompress_into(data, &mut out)?;
    Ok(out)
}

// =========================================================================
// BloscLZ 1.0.5 (blosclz.c of c-blosc 1.12.0), FastLZ-derived LZ77
// =========================================================================

#[cfg(feature = "blosc")]
const BLOSCLZ_MAX_COPY: usize = 32;
#[cfg(feature = "blosc")]
const BLOSCLZ_MAX_DISTANCE: usize = 8191;
#[cfg(feature = "blosc")]
const BLOSCLZ_MAX_FARDISTANCE: usize = 65535 + BLOSCLZ_MAX_DISTANCE - 1;

/// `HASH_FUNCTION` over the 3 bytes at `p`.
#[cfg(feature = "blosc")]
fn blosclz_hash(input: &[u8], p: usize, hash_log: u32) -> usize {
    let v = u16::from_le_bytes([input[p], input[p + 1]]) as u32;
    let v = v ^ u16::from_le_bytes([input[p + 1], input[p + 2]]) as u32 ^ (v >> (16 - hash_log));
    (v & ((1 << hash_log) - 1)) as usize
}

/// `blosclz_compress(opt_level, input, length, output, maxout = output.len(),
/// accel)`: the stream's length, or 0 when it would not come in under
/// `maxlength_[opt_level]` of the input. The match extension compares
/// 8 bytes at a time and, where a whole word matches up to the bound, takes
/// no byte-wise tail — the x86 build (no `BLOSCLZ_STRICT_ALIGN`) — so the
/// output is byte for byte what c-blosc writes there.
#[cfg(feature = "blosc")]
fn blosclz_compress(opt_level: u32, input: &[u8], output: &mut [u8], accel: u32) -> usize {
    const HASH_LOG: [u32; 10] = [0, 11, 11, 11, 12, 13, 13, 13, 13, 13];
    const MAXLENGTH: [f64; 10] = [-1.0, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0];
    const IP_BOUNDARY: usize = 2;
    let opt = opt_level.min(9) as usize;
    let length = input.len();
    let maxlength = ((length as f64 * MAXLENGTH[opt]) as usize).min(output.len());
    if opt == 0 || maxlength < 66 || length < 4 {
        return 0;
    }
    let op_limit = maxlength;
    let ip_bound = length - IP_BOUNDARY;
    let hash_log = HASH_LOG[opt];
    let accel = accel.max(1) - 1;
    // Positions are kept as u16, as C does: past 64 KiB a hit points at the
    // wrong place and the byte compare rejects or accepts it on its merits.
    let mut htab = vec![0u16; 1 << hash_log];

    let mut op = 0usize;
    let mut copy: usize = 2;
    output[op] = (BLOSCLZ_MAX_COPY - 1) as u8;
    output[op + 1] = input[0];
    output[op + 2] = input[1];
    op += 3;
    let mut ip = 2usize;

    while ip + 12 < length {
        let anchor = ip;
        let mut len = 3usize;
        let mut rf: usize;
        let mut distance: usize;
        let mut literal = false;

        if input[ip] == input[ip - 1] && input[ip - 1..ip + 1] == input[ip + 1..ip + 3] {
            // a run
            distance = 1;
            rf = anchor + 2;
        } else {
            let hval = blosclz_hash(input, ip, hash_log);
            rf = htab[hval] as usize;
            distance = anchor - rf;
            if distance & accel as usize == 0 {
                htab[hval] = anchor as u16;
            }
            if distance == 0
                || distance >= BLOSCLZ_MAX_FARDISTANCE
                || input[rf..rf + 3] != input[ip..ip + 3]
            {
                literal = true;
            } else if opt_level >= 5 && distance >= BLOSCLZ_MAX_DISTANCE {
                // far, needs at least 5 bytes
                if input[rf + 3..rf + 5] != input[ip + 3..ip + 5] {
                    literal = true;
                } else {
                    len += 2;
                }
            }
            rf += len;
        }

        if literal {
            if op + 2 > op_limit {
                return 0;
            }
            output[op] = input[anchor];
            op += 1;
            ip = anchor + 1;
            copy += 1;
            if copy == BLOSCLZ_MAX_COPY {
                copy = 0;
                output[op] = (BLOSCLZ_MAX_COPY - 1) as u8;
                op += 1;
            }
            continue;
        }

        // last matched byte
        ip = anchor + len;
        distance -= 1;

        if distance == 0 {
            // a run: compare against the repeated byte
            let x = input[ip - 1];
            while ip + 8 < length {
                if input[rf..rf + 8].iter().any(|&b| b != x) {
                    while ip < ip_bound {
                        let b = input[rf];
                        rf += 1;
                        if b != x {
                            break;
                        }
                        ip += 1;
                    }
                    break;
                }
                ip += 8;
                rf += 8;
            }
        } else {
            while ip + 8 < length {
                if input[rf..rf + 8] != input[ip..ip + 8] {
                    while ip < ip_bound {
                        let (a, b) = (input[rf], input[ip]);
                        rf += 1;
                        ip += 1;
                        if a != b {
                            break;
                        }
                    }
                    break;
                }
                ip += 8;
                rf += 8;
            }
        }
        ip = ip.min(ip_bound);

        // close the literal run, or take back its unused count byte
        if copy > 0 {
            output[op - copy - 1] = (copy - 1) as u8;
        } else {
            op -= 1;
        }
        copy = 0;

        // length is biased, '1' means a match of 3 bytes
        ip -= 3;
        let mut len = ip - anchor;
        if op + len / 255 + 6 > op_limit {
            return 0;
        }

        if distance < BLOSCLZ_MAX_DISTANCE {
            if len < 7 {
                output[op] = ((len << 5) + (distance >> 8)) as u8;
                output[op + 1] = (distance & 255) as u8;
                op += 2;
            } else {
                output[op] = ((7 << 5) + (distance >> 8)) as u8;
                op += 1;
                len -= 7;
                while len >= 255 {
                    output[op] = 255;
                    op += 1;
                    len -= 255;
                }
                output[op] = len as u8;
                output[op + 1] = (distance & 255) as u8;
                op += 2;
            }
        } else {
            distance -= BLOSCLZ_MAX_DISTANCE;
            if len < 7 {
                output[op] = ((len << 5) + 31) as u8;
                op += 1;
            } else {
                output[op] = (7 << 5) + 31;
                op += 1;
                len -= 7;
                while len >= 255 {
                    output[op] = 255;
                    op += 1;
                    len -= 255;
                }
                output[op] = len as u8;
                op += 1;
            }
            output[op] = 255;
            output[op + 1] = (distance >> 8) as u8;
            output[op + 2] = (distance & 255) as u8;
            op += 3;
        }

        // update the hash at the match boundary
        htab[blosclz_hash(input, ip, hash_log)] = ip as u16;
        ip += 1;
        htab[blosclz_hash(input, ip, hash_log)] = ip as u16;
        ip += 1;

        // assuming literal copy
        output[op] = (BLOSCLZ_MAX_COPY - 1) as u8;
        op += 1;
    }

    // left-over as literal copy
    while ip < length {
        if op + 2 > op_limit {
            return 0;
        }
        output[op] = input[ip];
        op += 1;
        ip += 1;
        copy += 1;
        if copy == BLOSCLZ_MAX_COPY {
            copy = 0;
            output[op] = (BLOSCLZ_MAX_COPY - 1) as u8;
            op += 1;
        }
    }
    if copy > 0 {
        output[op - copy - 1] = (copy - 1) as u8;
    } else {
        op -= 1;
    }

    // marker for blosclz
    output[0] |= 1 << 5;
    op
}

/// `blosclz_decompress(input, length, output, maxout = output.len())`: the
/// bytes written, or `None` where C returns 0 or would read past `input`.
#[cfg(feature = "blosc")]
fn blosclz_decompress(input: &[u8], output: &mut [u8]) -> Option<usize> {
    let mut ip = 0usize;
    let mut op = 0usize;
    // the first byte is a literal run, its top bits the BloscLZ marker
    let mut ctrl = (*input.first()? & 31) as usize;
    ip += 1;
    loop {
        if ctrl >= 32 {
            let mut len = (ctrl >> 5) - 1;
            let ofs = (ctrl & 31) << 8;
            if len == 7 - 1 {
                loop {
                    let code = *input.get(ip)? as usize;
                    ip += 1;
                    len += code;
                    if code != 255 {
                        break;
                    }
                }
            }
            let code = *input.get(ip)? as usize;
            ip += 1;
            // distance back to the byte before the match source
            let mut back = ofs + code;
            if code == 255 && ofs == 31 << 8 {
                // match from a 16-bit distance
                let far = ((*input.get(ip)? as usize) << 8) + *input.get(ip + 1)? as usize;
                ip += 2;
                back = far + BLOSCLZ_MAX_DISTANCE;
            }
            let len = len + 3;
            if op + len > output.len() || back + 1 > op {
                return None;
            }
            let from = op - back - 1;
            let period = back + 1;
            if period >= len {
                output.copy_within(from..from + len, op);
            } else {
                // overlapping: the source repeats with `period`; copy whole
                // repeats of what is already written, doubling each time
                let mut done = 0;
                while done < len {
                    let n = (len - done).min(period + done);
                    output.copy_within(from..from + n, op + done);
                    done += n;
                }
            }
            op += len;
        } else {
            let n = ctrl + 1;
            if op + n > output.len() || ip + n > input.len() {
                return None;
            }
            output[op..op + n].copy_from_slice(&input[ip..ip + n]);
            op += n;
            ip += n;
        }
        if ip >= input.len() {
            return Some(op);
        }
        ctrl = input[ip] as usize;
        ip += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte-for-byte against what libhdf5 writes. Each expected slice was
    /// read out of a file h5py created with `libver='earliest'` and a chunked
    /// dataset carrying that pipeline, which is the only authority on the
    /// padding rules — `H5O__pline_encode` states them, but the file is what
    /// a reader parses.
    ///
    /// The pipelines are the ones this crate's own builders produce, so the
    /// flags byte in each expected slice is the flag the builder stored and
    /// not a value written into the test: `H5Z_FLAG_OPTIONAL` for deflate and
    /// shuffle, `H5Z_FLAG_MANDATORY` for fletcher32 ([`FLAG_OPTIONAL`]).
    /// libhdf5 makes the same split, so a hardcoded flags byte on either
    /// side of it would fail here.
    #[test]
    fn version_1_encodes_the_bytes_libhdf5_writes() {
        // gzip level 4: name "deflate\0" is 8 bytes and needs no padding,
        // and the single client-data value is padded out to an even count.
        assert_eq!(
            FilterPipeline::deflate(4).encode_v1(),
            b"\x01\x01\x00\x00\x00\x00\x00\x00\
              \x01\x00\x08\x00\x01\x00\x01\x00\
              deflate\x00\
              \x04\x00\x00\x00\x00\x00\x00\x00"
        );

        // shuffle over 4-byte elements: same shape, and libhdf5 records the
        // element width `H5Z__set_local_shuffle` filled in.
        assert_eq!(
            FilterPipeline::shuffle(4).encode_v1(),
            b"\x01\x01\x00\x00\x00\x00\x00\x00\
              \x02\x00\x08\x00\x01\x00\x01\x00\
              shuffle\x00\
              \x04\x00\x00\x00\x00\x00\x00\x00"
        );

        // fletcher32: "fletcher32\0" is 11 bytes, so the name is padded to
        // 16 and the declared length is the padded one. No client data, so
        // no client-data padding either. Its flags byte is 0 — `H5Pset_fletcher32`
        // is the one libhdf5 setter that asks for a mandatory filter.
        let fletcher = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_FLETCHER32,
                flags: FLAG_MANDATORY,
                cd_values: vec![],
            }],
        };
        assert_eq!(
            fletcher.encode_v1(),
            b"\x01\x01\x00\x00\x00\x00\x00\x00\
              \x03\x00\x10\x00\x00\x00\x00\x00\
              fletcher32\x00\x00\x00\x00\x00\x00"
        );

        // Two filters in one message: the six reserved bytes are the
        // message's, not each filter's, and shuffle precedes deflate the way
        // libhdf5 applies them.
        assert_eq!(
            FilterPipeline::shuffle_deflate(4, 4).encode_v1(),
            b"\x01\x02\x00\x00\x00\x00\x00\x00\
              \x02\x00\x08\x00\x01\x00\x01\x00\
              shuffle\x00\
              \x04\x00\x00\x00\x00\x00\x00\x00\
              \x01\x00\x08\x00\x01\x00\x01\x00\
              deflate\x00\
              \x04\x00\x00\x00\x00\x00\x00\x00"
        );

        // A mandatory filter beside an optional one, `fletcher32=True` with
        // `compression='gzip'`. Each filter carries its own flags byte, so
        // one message holds both values — an encoder emitting a constant
        // cannot produce this.
        let mixed = FilterPipeline {
            filters: vec![
                FilterPipeline::deflate(4).filters[0].clone(),
                fletcher.filters[0].clone(),
            ],
        };
        assert_eq!(
            mixed.encode_v1(),
            b"\x01\x02\x00\x00\x00\x00\x00\x00\
              \x01\x00\x08\x00\x01\x00\x01\x00\
              deflate\x00\
              \x04\x00\x00\x00\x00\x00\x00\x00\
              \x03\x00\x10\x00\x00\x00\x00\x00\
              fletcher32\x00\x00\x00\x00\x00\x00"
        );
    }

    /// A filter this crate has no registered name for gets a zero name
    /// length and no name bytes — what `H5O__pline_encode` writes when
    /// `H5Z_find` resolves nothing — and the client-data padding still
    /// applies.
    #[test]
    fn version_1_names_only_the_filters_libhdf5_registers() {
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_ZSTD,
                flags: 0,
                cd_values: vec![3],
            }],
        };
        assert_eq!(
            pipeline.encode_v1(),
            b"\x01\x01\x00\x00\x00\x00\x00\x00\
              \x0f\x7d\x00\x00\x00\x00\x01\x00\
              \x03\x00\x00\x00\x00\x00\x00\x00"
        );
        let (decoded, consumed) = FilterPipeline::decode(&pipeline.encode_v1()).unwrap();
        assert_eq!(consumed, pipeline.encode_v1().len());
        assert_eq!(decoded, pipeline);
    }

    /// Every pipeline this crate can build survives its own version-1
    /// encoding, whatever the name length and client-data count do to the
    /// padding. The decoder is the one libhdf5 files are read with, so a
    /// round trip through it is what says the two agree on where each field
    /// ends.
    #[test]
    fn version_1_round_trips_through_the_decoder() {
        use crate::format::messages::datatype::DatatypeMessage;

        let dt = DatatypeMessage::i32_type();
        let single = |id, cd_values| FilterPipeline {
            filters: vec![Filter {
                id,
                flags: 0,
                cd_values,
            }],
        };
        for pipeline in [
            FilterPipeline::none(),
            FilterPipeline::deflate(6),
            FilterPipeline::shuffle(4),
            FilterPipeline::shuffle_deflate(4, 9),
            single(FILTER_FLETCHER32, vec![]),
            single(FILTER_SZIP, vec![4, 32, 32, 256]),
            FilterPipeline::nbit(&dt, 16),
            FilterPipeline::scaleoffset(&dt, 16, 0).unwrap(),
            FilterPipeline::zstd(3),
            FilterPipeline::bshuf_lz4(4),
        ] {
            let encoded = pipeline.encode_v1();
            assert_eq!(encoded[0], 1, "{pipeline:?}");
            assert_eq!(encoded[1] as usize, pipeline.filters.len(), "{pipeline:?}");
            let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
            assert_eq!(consumed, encoded.len(), "{pipeline:?}");
            assert_eq!(decoded, pipeline);
        }
    }

    /// The version follows the file's format and nothing else.
    #[test]
    fn the_object_format_picks_the_pipeline_version() {
        use crate::format::ObjectFormat;

        let pipeline = FilterPipeline::deflate(6);
        assert_eq!(
            pipeline.encode_for(ObjectFormat::Legacy),
            pipeline.encode_v1()
        );
        assert_eq!(pipeline.encode_for(ObjectFormat::Modern), pipeline.encode());
    }

    #[test]
    fn encode_decode_deflate() {
        let pipeline = FilterPipeline::deflate(6);
        let encoded = pipeline.encode();

        assert_eq!(encoded[0], 2); // version
        assert_eq!(encoded[1], 1); // 1 filter

        let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, pipeline);
        assert_eq!(decoded.filters[0].id, FILTER_DEFLATE);
        assert_eq!(decoded.filters[0].cd_values, vec![6]);
    }

    #[test]
    fn encode_decode_empty() {
        let pipeline = FilterPipeline::none();
        let encoded = pipeline.encode();
        assert_eq!(encoded.len(), 2);
        let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(decoded, pipeline);
    }

    #[test]
    fn encode_decode_multiple_filters() {
        let pipeline = FilterPipeline {
            filters: vec![
                Filter {
                    id: FILTER_SHUFFLE,
                    flags: 0,
                    cd_values: vec![],
                },
                Filter {
                    id: FILTER_DEFLATE,
                    flags: 0,
                    cd_values: vec![4],
                },
            ],
        };
        let encoded = pipeline.encode();
        let (decoded, consumed) = FilterPipeline::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.filters.len(), 2);
        assert_eq!(decoded.filters[0].id, FILTER_SHUFFLE);
        assert_eq!(decoded.filters[1].id, FILTER_DEFLATE);
        assert_eq!(decoded.filters[1].cd_values, vec![4]);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [3u8, 0]; // version 3 is not supported
        let err = FilterPipeline::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(3)));
    }

    #[test]
    fn decode_version_1_deflate() {
        // Version-1 pipeline with a single deflate filter (id 1), one
        // cd_value (the compression level). h5py default-libver shape.
        let mut buf = vec![1u8, 1]; // version 1, 1 filter
        buf.extend_from_slice(&[0u8; 6]); // 6 reserved bytes
        buf.extend_from_slice(&1u16.to_le_bytes()); // filter id = deflate
        buf.extend_from_slice(&8u16.to_le_bytes()); // name_length (padded)
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&1u16.to_le_bytes()); // num_cd_values = 1
        buf.extend_from_slice(b"deflate\0"); // 8-byte padded name
        buf.extend_from_slice(&6u32.to_le_bytes()); // cd_values[0] = level 6
        buf.extend_from_slice(&[0u8; 4]); // odd cd count -> 4-byte padding
        let (pl, consumed) = FilterPipeline::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].id, FILTER_DEFLATE);
        assert_eq!(pl.filters[0].cd_values, vec![6]);
    }

    #[test]
    fn decode_version_1_shuffle_then_deflate() {
        // Two filters, shuffle (id 2, one cd_value -> odd, padded) then
        // deflate (id 1).
        let mut buf = vec![1u8, 2];
        buf.extend_from_slice(&[0u8; 6]);
        // shuffle
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&8u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(b"shuffle\0");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // odd padding
                                          // deflate
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&8u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(b"deflate\0");
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // odd padding
        let (pl, consumed) = FilterPipeline::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(pl.filters.len(), 2);
        assert_eq!(pl.filters[0].id, FILTER_SHUFFLE);
        assert_eq!(pl.filters[1].id, FILTER_DEFLATE);
        assert_eq!(pl.filters[1].cd_values, vec![5]);
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [2u8]; // missing nfilters
        let err = FilterPipeline::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline::deflate(6);
        let original = vec![42u8; 1024];

        let compressed = apply_filters(&pipeline, &original).unwrap();
        // Compressed should be smaller than original for repeated data.
        assert!(compressed.len() < original.len());

        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn reverse_filters_masked_skips_masked_filter() {
        let pipeline = FilterPipeline::deflate(6);
        let original = vec![42u8; 1024];
        let compressed = apply_filters(&pipeline, &original).unwrap();

        // mask 0: the full pipeline reverses (inflate) compressed -> original.
        assert_eq!(
            reverse_filters_masked(&pipeline, &compressed, 0).unwrap(),
            original
        );
        // mask 1: filter 0 (deflate) is marked skipped, so the bytes are taken
        // verbatim — a chunk stored uncompressed by a direct chunk write.
        assert_eq!(
            reverse_filters_masked(&pipeline, &original, 1).unwrap(),
            original
        );
        // Out-of-range bits (>= 32) cannot mark a real filter; the single
        // deflate filter still runs, recovering the original.
        assert_eq!(
            reverse_filters_masked(&pipeline, &compressed, 1u32 << 31).unwrap(),
            original
        );
    }

    /// Boundaries of the growing inflate buffer in [`inflate_zlib`]: output
    /// below the first guess, exactly on a power-of-two step, and several
    /// doublings past it.
    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_output_sizes_around_the_buffer_boundary() {
        let pipeline = FilterPipeline::deflate(6);
        for len in [0usize, 1, 4095, 4096, 4097, 65_536, 1 << 20] {
            let original: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let compressed = apply_filters(&pipeline, &original).unwrap();
            let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
            assert_eq!(decompressed, original, "length {len}");
        }
    }

    /// Decoding into a caller's destination lands the same bytes the
    /// materializing spelling produces, for every pipeline shape: a lone
    /// deflate, a deflate behind another filter, and a pipeline masked down to
    /// nothing.
    #[cfg(feature = "deflate")]
    #[test]
    fn reverse_into_a_destination_matches_reverse_into_a_vec() {
        let original: Vec<u8> = (0..40_000u32).map(|i| (i % 17) as u8).collect();
        let cases = [
            FilterPipeline::deflate(6),
            FilterPipeline {
                filters: vec![
                    Filter {
                        id: FILTER_SHUFFLE,
                        flags: 0,
                        cd_values: vec![8],
                    },
                    Filter {
                        id: FILTER_DEFLATE,
                        flags: 0,
                        cd_values: vec![6],
                    },
                ],
            },
            FilterPipeline {
                filters: vec![
                    Filter {
                        id: FILTER_DEFLATE,
                        flags: 0,
                        cd_values: vec![6],
                    },
                    Filter {
                        id: FILTER_FLETCHER32,
                        flags: 0,
                        cd_values: vec![],
                    },
                ],
            },
        ];
        for (case, pipeline) in cases.iter().enumerate() {
            let compressed = apply_filters(pipeline, &original).unwrap();
            let want = reverse_filters(pipeline, &compressed).unwrap();
            let mut got = vec![0xAAu8; want.len()];
            assert_eq!(
                reverse_filters_masked_into(pipeline, &compressed, 0, &mut got).unwrap(),
                want.len(),
                "case {case}"
            );
            assert_eq!(got, want, "case {case}");

            // Masked down to nothing: the stored bytes are the image.
            let mask = u32::MAX;
            let mut raw = vec![0u8; compressed.len()];
            assert_eq!(
                reverse_filters_masked_into(pipeline, &compressed, mask, &mut raw).unwrap(),
                compressed.len(),
                "case {case}"
            );
            assert_eq!(
                raw,
                reverse_filters_masked(pipeline, &compressed, mask).unwrap(),
                "case {case}"
            );
        }
    }

    /// The returned length is the image the pipeline produced, not what fitted:
    /// a destination shorter than the image keeps its first bytes and reports
    /// the whole length, a longer one is written only as far as the image goes.
    /// Both are what a caller copying out of a materialized image would see.
    #[cfg(feature = "deflate")]
    #[test]
    fn reverse_into_reports_the_image_length_not_the_destination() {
        let pipeline = FilterPipeline::deflate(6);
        let original: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
        let compressed = apply_filters(&pipeline, &original).unwrap();

        let mut short = vec![0u8; 1000];
        assert_eq!(
            reverse_filters_masked_into(&pipeline, &compressed, 0, &mut short).unwrap(),
            original.len()
        );
        assert_eq!(short, original[..1000]);

        // A destination outlasting the image keeps the image; what the slack
        // past it holds is unspecified (the inflate's last output block can
        // reach past the byte the stream ends on), which is why a caller that
        // sees a short image owns the whole destination, not just that tail.
        let mut long = vec![0xCDu8; original.len() + 4096];
        assert_eq!(
            reverse_filters_masked_into(&pipeline, &compressed, 0, &mut long).unwrap(),
            original.len()
        );
        assert_eq!(&long[..original.len()], &original[..]);
    }

    /// A stream that is malformed past the destination still fails: the
    /// overflow is inflated through a scratch buffer rather than abandoned at
    /// the destination's end, which is where a growing buffer would have found
    /// the same error.
    #[cfg(feature = "deflate")]
    #[test]
    fn reverse_into_a_short_destination_still_validates_the_rest_of_the_stream() {
        let pipeline = FilterPipeline::deflate(6);
        let original: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let compressed = apply_filters(&pipeline, &original).unwrap();
        let cut = &compressed[..compressed.len() / 2];
        let mut small = vec![0u8; 4096];
        assert!(reverse_filters_masked_into(&pipeline, cut, 0, &mut small).is_err());
        assert!(reverse_filters(&pipeline, cut).is_err());
    }

    /// A chunk read `at_most` can carry bytes past the end of the stream (the
    /// reader does not always know the exact stored length); inflate stops at
    /// the stream end and ignores them. A stream cut short is an error.
    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_tolerates_trailing_bytes_but_not_a_cut_stream() {
        let pipeline = FilterPipeline::deflate(6);
        let original: Vec<u8> = (0..40_000).map(|i| (i % 13) as u8).collect();
        let compressed = apply_filters(&pipeline, &original).unwrap();

        let mut padded = compressed.clone();
        padded.extend_from_slice(&[0xAB; 64]);
        assert_eq!(reverse_filters(&pipeline, &padded).unwrap(), original);

        let cut = &compressed[..compressed.len() / 2];
        assert!(reverse_filters(&pipeline, cut).is_err());
        assert!(reverse_filters(&pipeline, &[]).is_err());
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_level_zero() {
        let pipeline = FilterPipeline::deflate(0);
        let original = b"hello world, this is a test of level 0 deflate";

        let compressed = apply_filters(&pipeline, original).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_level_nine() {
        let pipeline = FilterPipeline::deflate(9);
        let original: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();

        let compressed = apply_filters(&pipeline, &original).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn shuffle_unshuffle_roundtrip() {
        // 4-byte elements: [0,1,2,3, 4,5,6,7, 8,9,10,11]
        let data: Vec<u8> = (0..12).collect();
        let shuffled = shuffle(&data, 4);
        // After shuffle: [0,4,8, 1,5,9, 2,6,10, 3,7,11]
        assert_eq!(shuffled, vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_filter_pipeline_roundtrip() {
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_SHUFFLE,
                flags: 0,
                cd_values: vec![4], // 4-byte elements
            }],
        };
        let data: Vec<u8> = (0..64).collect();
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn shuffle_deflate_roundtrip() {
        let pipeline = FilterPipeline::shuffle_deflate(8, 6);
        // Repeating f64 pattern compresses well with shuffle
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let compressed = apply_filters(&pipeline, &data).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn fletcher32_matches_libhdf5() {
        // Reference values computed from a Python port of libhdf5's
        // H5_checksum_fletcher32 (H5checksum.c).
        assert_eq!(fletcher32(b""), 0x0000_0000);
        assert_eq!(fletcher32(b"hi"), 0x6869_6869);
        assert_eq!(fletcher32(b"hello world"), 0xfc27_91ce);
        assert_eq!(fletcher32(b"abc"), 0x25c5_c462);
        assert_eq!(fletcher32(&[0xFF, 0xFF]), 0xffff_ffff);
        assert_eq!(fletcher32(&[0xFF; 4]), 0xffff_ffff);
        let range256: Vec<u8> = (0..=255u8).collect();
        assert_eq!(fletcher32(&range256), 0x5575_c03f);
        assert_eq!(fletcher32(&[0u8; 1000]), 0x0000_0000);
    }

    /// The lane kernel finished by the scalar loop against the scalar loop
    /// alone, on every level the host offers, over lengths on both sides of
    /// every run boundary and on the all-ones input that drives the
    /// accumulators to their fold ceiling.
    #[cfg(feature = "simd")]
    #[test]
    fn fletcher32_lanes_match_the_scalar_loop_on_every_level() {
        use fearless_simd::Level;
        let top = crate::simd::level();
        let mut levels = vec![top, Level::baseline()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            levels.extend(top.as_avx2().map(Level::Avx2));
            levels.extend(top.as_sse4_2().map(Level::Sse4_2));
            levels.extend(top.as_sse2().map(Level::Sse2));
        }
        let mut lengths: Vec<usize> = vec![0, 1, 2, 3, 15, 16, 17];
        for run in 1..=4 {
            lengths.extend([720 * run - 1, 720 * run, 720 * run + 1, 720 * run + 2]);
        }
        lengths.extend([8192, 65_536 + 3, 100_001]);
        let mixed: Vec<u8> = (0..100_001)
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let ones = vec![0xFFu8; 100_001];
        for &level in &levels {
            for &len in &lengths {
                for (name, input) in [("mixed", &mixed[..len]), ("ones", &ones[..len])] {
                    let want = fletcher32_from(0, 0, 0, input);
                    let (s1, s2, done) =
                        fearless_simd::dispatch!(level, s => fletcher_simd::runs(s, input));
                    assert_eq!(done, len / 720 * 720, "{level:?} {name} len={len} done");
                    let got = fletcher32_from(s1, s2, done, input);
                    assert_eq!(got, want, "{level:?} {name} len={len}");
                }
            }
        }
    }

    #[test]
    fn fletcher32_roundtrip() {
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_FLETCHER32,
                flags: 0,
                cd_values: vec![],
            }],
        };
        let data = b"hello world";
        let encoded = apply_filters(&pipeline, data).unwrap();
        assert_eq!(encoded.len(), data.len() + 4);
        let decoded = reverse_filters(&pipeline, &encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn fletcher32_trailer_byte_order_matches_libhdf5() {
        // libhdf5 writes the trailer with UINT32ENCODE (little-endian) over
        // the u32 checksum, so the on-disk trailer is `cksum.to_le_bytes()`.
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_FLETCHER32,
                flags: 0,
                cd_values: vec![],
            }],
        };
        let data: Vec<u8> = (0..8i32).flat_map(|v| v.to_le_bytes()).collect();
        let encoded = apply_filters(&pipeline, &data).unwrap();
        let trailer = &encoded[encoded.len() - 4..];
        let c = fletcher32(&data);
        assert_eq!(trailer, &c.to_le_bytes());
    }

    #[cfg(all(feature = "deflate", feature = "parallel"))]
    #[test]
    fn parallel_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline::deflate(6);
        let chunks: Vec<Vec<u8>> = (0..8)
            .map(|i| vec![(i as u8).wrapping_mul(42); 1024])
            .collect();

        let borrowed: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let compressed = apply_filters_parallel(&pipeline, &borrowed).unwrap();
        assert_eq!(compressed.len(), 8);
        // Each compressed chunk should be smaller (repeated data compresses well)
        for c in &compressed {
            assert!(c.len() < 1024);
        }

        let decompressed = reverse_filters_parallel(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed.len(), 8);
        for (original, decoded) in chunks.iter().zip(decompressed.iter()) {
            assert_eq!(original, decoded);
        }
    }

    /// A compress failure on any chunk must propagate out of the parallel
    /// helpers, never be swallowed into raw bytes. If it were swallowed, a
    /// caller recording the chunk under filter_mask=0 would claim the pipeline
    /// ran, and the reader would try to reverse-filter raw data and corrupt it.
    /// Scale-offset with empty `cd_values` cannot describe a datatype, so
    /// apply_single_filter errors deterministically (no feature flag needed).
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_compress_propagates_filter_error() {
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_SCALEOFFSET,
                flags: 0,
                cd_values: vec![],
            }],
        };
        let chunks: [&[u8]; 2] = [&[1u8; 64], &[2u8; 64]];
        assert!(
            apply_filters_parallel(&pipeline, &chunks).is_err(),
            "scale-offset compress error must propagate, not be swallowed"
        );
    }

    // =================================================================
    // Golden tests — verify against known data patterns
    // =================================================================

    /// Golden test data: 256 f32 values [0.0, 1.0, 2.0, ..., 255.0]
    fn golden_f32_data() -> Vec<u8> {
        (0..256u32).flat_map(|i| (i as f32).to_le_bytes()).collect()
    }

    /// Golden test data: 128 f64 values [0.0, 0.5, 1.0, ..., 63.5]
    fn golden_f64_data() -> Vec<u8> {
        (0..128u32)
            .flat_map(|i| (i as f64 * 0.5).to_le_bytes())
            .collect()
    }

    // --- LZF roundtrip ---
    #[test]
    fn lzf_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_LZF,
                flags: 0,
                cd_values: vec![4, 0, data.len() as u32],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lzf_golden_known_pattern() {
        // All-zeros should compress well
        let data = vec![0u8; 1024];
        let compressed = lzf_compress(&data);
        assert!(compressed.len() < data.len());
        let decompressed = lzf_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    /// The declared output size is a hint: past what the input could expand
    /// to it sizes nothing, so a crafted `cd_values[2]` neither aborts nor
    /// reserves, and a stream that expands past its declaration still
    /// decodes, as h5py's filter does.
    #[test]
    fn lzf_declared_size_is_a_hint_not_a_reservation() {
        assert_eq!(lzf_decompress(&[], usize::MAX).unwrap(), Vec::<u8>::new());
        // One literal byte under a declaration of 2^60 bytes.
        let out = lzf_decompress(&[0x00, 0x41], 1 << 60).unwrap();
        assert_eq!(out, b"A");
        assert!(out.capacity() <= 88 * 2);

        let data = vec![7u8; 4096];
        let compressed = lzf_compress(&data);
        assert_eq!(lzf_decompress(&compressed, 16).unwrap(), data);
        assert_eq!(lzf_decompress(&compressed, 0).unwrap(), data);
    }

    #[test]
    fn lzf_incompressible() {
        // Random-like data shouldn't grow
        let data: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(137)).collect();
        let compressed = lzf_compress(&data);
        let decompressed = if compressed == data {
            data.clone() // returned unchanged
        } else {
            lzf_decompress(&compressed, data.len()).unwrap()
        };
        assert_eq!(decompressed, data);
    }

    // --- Bitshuffle roundtrip ---
    #[test]
    fn bitshuffle_no_compression_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BSHUF,
                flags: 0,
                cd_values: vec![0, 0, 4, 0, 0],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn bitshuffle_lz4_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BSHUF,
                flags: 0,
                cd_values: vec![0, 0, 4, 0, 2],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        // Bitshuffle+LZ4 produces valid output (may not always be smaller)
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    fn bshuf_dummy_filter() -> Filter {
        Filter {
            id: FILTER_BSHUF,
            flags: 0,
            cd_values: vec![0, 0, 0, 0, 0],
        }
    }

    /// Byte-for-byte parity of the no-compression bit transpose against the
    /// canonical bitshuffle filter (kiyo-masui/bitshuffle, HDF5 plugin
    /// `bshuf_bitshuffle`). Vectors generated from that C reference; each
    /// tuple is (elem_size, block_size_elems, input, canonical_shuffled).
    /// Covers full blocks, a final block rounded to a multiple of 8, and the
    /// raw `n_elems % 8` leftover. Guards against the historical MSB-first
    /// (non-canonical) bit order and the old raw-tail framing.
    #[test]
    fn bitshuffle_canonical_nocomp_vectors() {
        let cases: &[(usize, usize, &[u8], &[u8])] = &[
            (
                1,
                8,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204,
                ],
                &[
                    85, 51, 165, 198, 248, 0, 0, 170, 85, 51, 165, 57, 193, 254, 0, 170, 85, 51,
                    165, 198, 7, 7, 248, 170,
                ],
            ),
            (
                1,
                8,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192,
                ],
                &[
                    85, 51, 165, 198, 248, 0, 0, 170, 85, 51, 165, 57, 193, 254, 0, 170, 55, 186,
                    61, 192,
                ],
            ),
            (
                1,
                16,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216,
                ],
                &[
                    85, 85, 51, 51, 165, 165, 198, 57, 248, 193, 0, 254, 0, 0, 170, 170, 85, 51,
                    165, 198, 7, 7, 248, 170, 79, 210, 85, 216,
                ],
            ),
            (
                2,
                8,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252,
                ],
                &[
                    255, 85, 51, 90, 156, 224, 0, 0, 0, 85, 204, 105, 142, 240, 0, 255, 255, 85,
                    51, 90, 99, 131, 252, 0, 0, 85, 204, 105, 113, 129, 254, 255, 103, 234, 109,
                    240, 115, 246, 121, 252,
                ],
            ),
            (
                2,
                16,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                    163, 38, 169, 44,
                ],
                &[
                    255, 255, 85, 85, 51, 51, 90, 90, 156, 99, 224, 131, 0, 252, 0, 0, 0, 0, 85,
                    85, 204, 204, 105, 105, 142, 113, 240, 129, 0, 254, 255, 255, 255, 85, 51, 90,
                    156, 31, 31, 224, 0, 85, 204, 105, 142, 15, 15, 15, 151, 26, 157, 32, 163, 38,
                    169, 44,
                ],
            ),
            (
                4,
                8,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                ],
                &[
                    255, 255, 85, 204, 150, 24, 224, 0, 0, 255, 170, 153, 210, 28, 224, 255, 255,
                    0, 85, 51, 90, 156, 224, 0, 0, 0, 170, 102, 75, 140, 240, 255, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                ],
            ),
            (
                4,
                0,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                    163, 38, 169, 44, 175, 50, 181, 56, 187, 62, 193, 68, 199, 74, 205, 80, 211,
                    86, 217, 92, 223, 98, 229, 104, 235, 110, 241, 116, 247, 122, 253, 128, 3, 134,
                    9, 140, 15, 146, 21, 152, 27, 158, 33, 164,
                ],
                &[
                    255, 255, 255, 255, 255, 255, 85, 85, 85, 204, 204, 204, 150, 150, 150, 24,
                    231, 24, 224, 7, 31, 0, 248, 31, 0, 0, 0, 255, 255, 255, 170, 170, 170, 153,
                    153, 153, 210, 210, 210, 28, 227, 28, 224, 3, 31, 255, 3, 224, 255, 255, 255,
                    0, 0, 0, 85, 85, 85, 51, 51, 51, 90, 90, 90, 156, 99, 156, 224, 131, 31, 0,
                    252, 31, 0, 0, 0, 0, 0, 0, 170, 170, 170, 102, 102, 102, 75, 75, 75, 140, 115,
                    140, 240, 131, 15, 255, 3, 240,
                ],
            ),
            (
                1,
                0,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252,
                ],
                &[
                    85, 85, 85, 85, 85, 51, 51, 51, 51, 51, 165, 165, 165, 165, 165, 198, 57, 198,
                    57, 198, 248, 193, 7, 62, 248, 0, 254, 7, 192, 255, 0, 0, 248, 255, 255, 170,
                    170, 170, 170, 170,
                ],
            ),
        ];
        let filter = bshuf_dummy_filter();
        for (i, &(elem_size, block, input, expected)) in cases.iter().enumerate() {
            let got = bitshuffle_compress(input, elem_size, block, 0, &filter).unwrap();
            assert_eq!(
                got, expected,
                "shuffle mismatch in case {i} (es={elem_size}, block={block})"
            );
            let back = bitshuffle_decompress(&got, elem_size, block, 0).unwrap();
            assert_eq!(back, input, "roundtrip mismatch in case {i}");
        }
    }

    /// rust-hdf5 must decode BSLZ4 streams produced by the canonical C filter.
    /// Vectors are the full HDF5 filter bytes (12-byte header + body) from
    /// `bshuf_compress_lz4`; each tuple is (elem_size, block_arg, input,
    /// filter_bytes). The block size is read from the header, so block_arg is
    /// passed only to mirror how the dataset would advertise it.
    #[cfg(feature = "lz4")]
    #[test]
    fn bitshuffle_canonical_lz4_decode_vectors() {
        let cases: &[(usize, usize, &[u8], &[u8])] = &[
            (
                1,
                16,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216,
                ],
                &[
                    0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 16, 0, 0, 0, 18, 240, 1, 85, 85, 51, 51, 165,
                    165, 198, 57, 248, 193, 0, 254, 0, 0, 170, 170, 0, 0, 0, 9, 128, 85, 51, 165,
                    198, 7, 7, 248, 170, 79, 210, 85, 216,
                ],
            ),
            (
                2,
                16,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                    163, 38, 169, 44,
                ],
                &[
                    0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 32, 0, 0, 0, 34, 240, 17, 255, 255, 85, 85,
                    51, 51, 90, 90, 156, 99, 224, 131, 0, 252, 0, 0, 0, 0, 85, 85, 204, 204, 105,
                    105, 142, 113, 240, 129, 0, 254, 255, 255, 0, 0, 0, 18, 240, 1, 255, 85, 51,
                    90, 156, 31, 31, 224, 0, 85, 204, 105, 142, 15, 15, 15, 151, 26, 157, 32, 163,
                    38, 169, 44,
                ],
            ),
            (
                4,
                8,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                    163, 38, 169, 44, 175, 50, 181, 56, 187, 62, 193, 68, 199, 74, 205, 80, 211,
                    86, 217, 92, 223, 98, 229, 104, 235, 110, 241, 116, 247, 122, 253, 128, 3, 134,
                    9, 140, 15, 146, 21, 152, 27, 158, 33, 164, 39, 170, 45, 176, 51, 182, 57, 188,
                    63, 194, 69, 200, 75, 206, 81, 212, 87, 218, 93, 224, 99, 230, 105, 236, 111,
                    242, 117, 248, 123, 254, 129, 4, 135, 10, 141, 16, 147, 22, 153, 28, 159, 34,
                    165, 40, 171, 46, 177, 52, 183, 58, 189, 64, 195, 70, 201, 76, 207, 82, 213,
                    88, 219, 94, 225, 100,
                ],
                &[
                    0, 0, 0, 0, 0, 0, 0, 160, 0, 0, 0, 32, 0, 0, 0, 34, 240, 17, 255, 255, 85, 204,
                    150, 24, 224, 0, 0, 255, 170, 153, 210, 28, 224, 255, 255, 0, 85, 51, 90, 156,
                    224, 0, 0, 0, 170, 102, 75, 140, 240, 255, 0, 0, 0, 34, 240, 17, 255, 255, 85,
                    204, 150, 231, 7, 248, 0, 255, 170, 153, 210, 227, 3, 3, 255, 0, 85, 51, 90,
                    99, 131, 252, 0, 0, 170, 102, 75, 115, 131, 3, 0, 0, 0, 34, 240, 17, 255, 255,
                    85, 204, 150, 24, 31, 31, 0, 255, 170, 153, 210, 28, 31, 224, 255, 0, 85, 51,
                    90, 156, 31, 31, 0, 0, 170, 102, 75, 140, 15, 240, 0, 0, 0, 34, 240, 17, 255,
                    255, 85, 204, 150, 231, 248, 0, 0, 255, 170, 153, 210, 227, 252, 255, 255, 0,
                    85, 51, 90, 99, 124, 128, 0, 0, 170, 102, 75, 115, 124, 127, 0, 0, 0, 34, 240,
                    17, 255, 255, 85, 204, 150, 24, 224, 255, 0, 255, 170, 153, 210, 28, 224, 0,
                    255, 0, 85, 51, 90, 156, 224, 255, 0, 0, 170, 102, 75, 140, 240, 0,
                ],
            ),
            (
                4,
                0,
                &[
                    7, 138, 13, 144, 19, 150, 25, 156, 31, 162, 37, 168, 43, 174, 49, 180, 55, 186,
                    61, 192, 67, 198, 73, 204, 79, 210, 85, 216, 91, 222, 97, 228, 103, 234, 109,
                    240, 115, 246, 121, 252, 127, 2, 133, 8, 139, 14, 145, 20, 151, 26, 157, 32,
                    163, 38, 169, 44, 175, 50, 181, 56, 187, 62, 193, 68, 199, 74, 205, 80, 211,
                    86, 217, 92, 223, 98, 229, 104, 235, 110, 241, 116, 247, 122, 253, 128, 3, 134,
                    9, 140, 15, 146, 21, 152, 27, 158, 33, 164,
                ],
                &[
                    0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 32, 0, 0, 0, 0, 96, 17, 255, 1, 0, 240, 50, 85,
                    85, 85, 204, 204, 204, 150, 150, 150, 24, 231, 24, 224, 7, 31, 0, 248, 31, 0,
                    0, 0, 255, 255, 255, 170, 170, 170, 153, 153, 153, 210, 210, 210, 28, 227, 28,
                    224, 3, 31, 255, 3, 224, 255, 255, 255, 0, 0, 0, 85, 85, 85, 51, 51, 51, 90,
                    90, 90, 156, 99, 156, 224, 131, 31, 0, 252, 48, 0, 240, 6, 0, 0, 0, 170, 170,
                    170, 102, 102, 102, 75, 75, 75, 140, 115, 140, 240, 131, 15, 255, 3, 240,
                ],
            ),
        ];
        for (i, &(elem_size, block, input, filter_bytes)) in cases.iter().enumerate() {
            let back =
                bitshuffle_decompress(filter_bytes, elem_size, block, BSHUF_COMPRESS_LZ4).unwrap();
            assert_eq!(back, input, "canonical BSLZ4 decode mismatch in case {i}");
        }
    }

    /// The LZ4 framing (header + per-block length-prefixed blocks + raw
    /// `n_elems % 8` leftover) must round-trip for inputs that exercise full
    /// blocks, a rounded final block, and a raw leftover simultaneously.
    #[cfg(feature = "lz4")]
    #[test]
    fn bitshuffle_lz4_framing_roundtrip() {
        let filter = bshuf_dummy_filter();
        // (elem_size, block_size_elems, n_elems): each yields full + last + leftover.
        for &(elem_size, block, n_elems) in &[
            (1usize, 16usize, 28usize),
            (2, 16, 28),
            (4, 8, 45),
            (2, 8, 100),
        ] {
            let data: Vec<u8> = (0..n_elems * elem_size)
                .map(|i| (i * 131 + 7) as u8)
                .collect();
            let comp =
                bitshuffle_compress(&data, elem_size, block, BSHUF_COMPRESS_LZ4, &filter).unwrap();
            let back = bitshuffle_decompress(&comp, elem_size, block, BSHUF_COMPRESS_LZ4).unwrap();
            assert_eq!(
                back, data,
                "lz4 framing roundtrip failed for es={elem_size} block={block} n={n_elems}"
            );
        }
    }

    // --- N-bit filter constructor tests ---
    #[test]
    fn nbit_atomic_cd_values_layout() {
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
        // 12-bit unsigned packed into a 2-byte fixed-point footprint.
        let dt = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 12,
        };
        let pipeline = FilterPipeline::nbit(&dt, 100);
        assert_eq!(pipeline.filters.len(), 1);
        let f = &pipeline.filters[0];
        assert_eq!(f.id, FILTER_NBIT);
        // [nparms, need_not_compress, d_nelmts, NBIT_ATOMIC, size, order,
        //  precision, offset]
        assert_eq!(f.cd_values, vec![8, 0, 100, 1, 2, 0, 12, 0]);

        // A full-precision atomic flags need_not_compress = 1 (pass-through).
        let full = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        };
        assert_eq!(FilterPipeline::nbit(&full, 100).filters[0].cd_values[1], 1);

        // Big-endian maps to order code 1.
        let be = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::BigEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 12,
        };
        assert_eq!(FilterPipeline::nbit(&be, 100).filters[0].cd_values[5], 1);
    }

    #[test]
    fn nbit_atomic_roundtrip_packs_12bit() {
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
        let n = 64usize;
        let dt = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 12,
        };
        let pipeline = FilterPipeline::nbit(&dt, n);
        // Values within 12-bit range, stored little-endian as u16.
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = ((i * 53 + 1) % 4096) as u16;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let packed = apply_filters(&pipeline, &data).unwrap();
        // 12 of every 16 bits are kept, so the packed stream is shorter.
        assert!(
            packed.len() < data.len(),
            "nbit did not pack: {} >= {}",
            packed.len(),
            data.len()
        );
        let back = reverse_filters(&pipeline, &packed).unwrap();
        assert_eq!(back, data, "nbit 12-bit roundtrip mismatch");
    }

    #[test]
    fn scaleoffset_cd_values_layout() {
        use crate::format::messages::datatype::{ByteOrder, DatatypeMessage};
        // int32 LE, library-computed minbits: the layout libhdf5 stores for a
        // dataset created with `H5Pset_scaleoffset(H5Z_SO_INT, 0)`.
        let dt = DatatypeMessage::i32_type();
        let f = &FilterPipeline::scaleoffset(&dt, 100, 0).unwrap().filters[0];
        assert_eq!(f.id, FILTER_SCALEOFFSET);
        // [scale_type, scale_factor, d_nelmts, class, size, sign, order,
        //  filavail, fill value ...]
        assert_eq!(
            f.cd_values,
            vec![2, 0, 100, 0, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );

        // Unsigned drops the sign code; big-endian raises the order code.
        let be = DatatypeMessage::FixedPoint {
            size: 2,
            byte_order: ByteOrder::BigEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        };
        let f = &FilterPipeline::scaleoffset(&be, 8, 0).unwrap().filters[0];
        assert_eq!((f.cd_values[5], f.cd_values[6]), (0, 1));

        // A float carries the D-scale type and the digit count, and no sign.
        let f = &FilterPipeline::scaleoffset(&DatatypeMessage::f64_type(), 8, 3)
            .unwrap()
            .filters[0];
        assert_eq!(f.cd_values[..8], [0, 3, 8, 1, 8, 0, 0, 1]);

        // Classes the filter cannot describe are refused, as H5Z__set_local
        // does.
        assert!(FilterPipeline::scaleoffset(&DatatypeMessage::fixed_string(4), 8, 0).is_none());
    }

    /// The `d_nelmts` a pipeline is built with is what both directions read
    /// the chunk length from, so a pipeline and its chunk must agree.
    #[test]
    fn scaleoffset_pipeline_roundtrips_its_own_chunk() {
        use crate::format::messages::datatype::DatatypeMessage;
        let n = 40usize;
        let pipeline = FilterPipeline::scaleoffset(&DatatypeMessage::i32_type(), n, 0).unwrap();
        let data: Vec<u8> = (0..n as i32)
            .flat_map(|i| (-40 + i * 3).to_le_bytes())
            .collect();
        let packed = apply_filters(&pipeline, &data).unwrap();
        assert!(
            packed.len() < data.len(),
            "scaleoffset did not pack: {} >= {}",
            packed.len(),
            data.len()
        );
        assert_eq!(reverse_filters(&pipeline, &packed).unwrap(), data);
    }

    // --- BitGroom golden test ---
    #[test]
    fn bitgroom_golden_f32() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BITGROOM,
                flags: 0,
                cd_values: vec![3, 4, 0, 0, 0],
            }],
        };
        let quantized = apply_filters(&pipeline, &data).unwrap();
        assert_eq!(quantized.len(), data.len()); // same size

        // Verify values are close to originals (within NSD=3 precision)
        for i in 0..256 {
            let orig = f32::from_le_bytes([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]);
            let quant = f32::from_le_bytes([
                quantized[i * 4],
                quantized[i * 4 + 1],
                quantized[i * 4 + 2],
                quantized[i * 4 + 3],
            ]);
            if orig == 0.0 {
                continue;
            }
            let rel_err = ((quant - orig) / orig).abs();
            assert!(
                rel_err < 0.01,
                "value {} quantized to {}, rel_err={}",
                orig,
                quant,
                rel_err
            );
        }

        // Decompress is no-op
        let decompressed = reverse_filters(&pipeline, &quantized).unwrap();
        assert_eq!(decompressed, quantized);
    }

    // --- BitRound golden test ---
    #[test]
    fn bitround_golden_f64() {
        let data = golden_f64_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BITROUND,
                flags: 0,
                cd_values: vec![4, 8, 0, 0, 0],
            }],
        };
        let quantized = apply_filters(&pipeline, &data).unwrap();
        assert_eq!(quantized.len(), data.len());

        for i in 0..128 {
            let orig = f64::from_le_bytes([
                data[i * 8],
                data[i * 8 + 1],
                data[i * 8 + 2],
                data[i * 8 + 3],
                data[i * 8 + 4],
                data[i * 8 + 5],
                data[i * 8 + 6],
                data[i * 8 + 7],
            ]);
            let quant = f64::from_le_bytes([
                quantized[i * 8],
                quantized[i * 8 + 1],
                quantized[i * 8 + 2],
                quantized[i * 8 + 3],
                quantized[i * 8 + 4],
                quantized[i * 8 + 5],
                quantized[i * 8 + 6],
                quantized[i * 8 + 7],
            ]);
            if orig == 0.0 {
                continue;
            }
            let rel_err = ((quant - orig) / orig).abs();
            assert!(
                rel_err < 0.001,
                "f64 {} quantized to {}, rel_err={}",
                orig,
                quant,
                rel_err
            );
        }
    }

    // --- LZ4 C-compatible framing golden test ---
    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_c_framing_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_LZ4,
                flags: 0,
                cd_values: vec![1 << 20],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        // Verify C-compatible header: 8-byte BE orig_size + 4-byte BE block_size
        assert!(compressed.len() >= 12);
        let orig_from_header = u64::from_be_bytes([
            compressed[0],
            compressed[1],
            compressed[2],
            compressed[3],
            compressed[4],
            compressed[5],
            compressed[6],
            compressed[7],
        ]);
        assert_eq!(orig_from_header, data.len() as u64);

        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_multi_block_roundtrip() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        // Small block size to force multiple blocks
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_LZ4,
                flags: 0,
                cd_values: vec![1024],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BZIP2 roundtrip ---
    #[cfg(feature = "bzip2")]
    #[test]
    fn bzip2_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BZIP2,
                flags: 0,
                cd_values: vec![9],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    /// The fixture is `bzip2 -9` 1.0.8 (C) output for `golden_f64_data`, so
    /// the pure-Rust backend writes the bytes a C-filter file holds.
    #[cfg(feature = "bzip2")]
    #[test]
    fn bzip2_matches_c_libbz2_bytes() {
        let c_bytes = include_bytes!("../../../tests/fixtures/golden_f64_bzip2_9.bz2");
        let data = golden_f64_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BZIP2,
                flags: 0,
                cd_values: vec![9],
            }],
        };
        assert_eq!(apply_filters(&pipeline, &data).unwrap(), c_bytes);
        assert_eq!(reverse_filters(&pipeline, c_bytes).unwrap(), data);
    }

    // --- BLOSC roundtrip (LZ4, default) ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, BLOSC_LZ4],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        assert!(compressed.len() >= 16);
        assert_eq!(compressed[0], 2); // format version
        let nbytes =
            u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]);
        assert_eq!(nbytes as usize, data.len());

        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BLOSC + BloscLZ ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_blosclz_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, BLOSC_BLOSCLZ],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BLOSC + LZ4HC ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_lz4hc_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, BLOSC_LZ4HC],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BLOSC + Snappy ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_snappy_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, BLOSC_SNAPPY],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BLOSC + Zlib (requires deflate feature) ---
    #[cfg(all(feature = "blosc", feature = "deflate"))]
    #[test]
    fn blosc_zlib_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, BLOSC_ZLIB],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BLOSC + Zstd (requires zstd feature) ---
    #[cfg(all(feature = "blosc", feature = "zstd"))]
    #[test]
    fn blosc_zstd_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, BLOSC_ZSTD],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BloscLZ pure roundtrip (no shuffle) ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_pure_roundtrip() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let mut compressed = vec![0u8; data.len()];
        let n = blosclz_compress(5, &data, &mut compressed, 1);
        assert!(n > 0);
        let mut decompressed = vec![0u8; data.len()];
        assert_eq!(
            blosclz_decompress(&compressed[..n], &mut decompressed),
            Some(data.len())
        );
        assert_eq!(decompressed, data);
    }

    // --- BloscLZ with highly compressible data ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_repeated_data() {
        let data = vec![42u8; 8192];
        let mut compressed = vec![0u8; data.len()];
        let n = blosclz_compress(5, &data, &mut compressed, 1);
        assert!(n > 0 && n < data.len());
        let mut decompressed = vec![0u8; data.len()];
        assert_eq!(
            blosclz_decompress(&compressed[..n], &mut decompressed),
            Some(data.len())
        );
        assert_eq!(decompressed, data);
    }

    // --- BloscLZ golden tests: decompress data produced by C blosclz_compress ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_golden_decompress_rle() {
        // C blosclz_compress(5, [0x42; 512], ..., 0) -> 12 bytes
        let compressed: &[u8] = &[
            0x23, 0x42, 0x42, 0x42, 0x42, 0xe0, 0xff, 0xf2, 0x03, 0x01, 0x42, 0x42,
        ];
        let expected = vec![0x42u8; 512];
        let mut decompressed = vec![0u8; 512];
        assert_eq!(blosclz_decompress(compressed, &mut decompressed), Some(512));
        assert_eq!(decompressed, expected);
    }

    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_golden_decompress_pattern16() {
        // C blosclz_compress(9, [i%16; 1024], ..., 0) -> 26 bytes
        let compressed: &[u8] = &[
            0x2f, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
            0x0d, 0x0e, 0x0f, 0xe0, 0xff, 0xff, 0xff, 0xe8, 0x0f, 0x01, 0x0e, 0x0f,
        ];
        let expected: Vec<u8> = (0..1024).map(|i| (i % 16) as u8).collect();
        let mut decompressed = vec![0u8; 1024];
        assert_eq!(
            blosclz_decompress(compressed, &mut decompressed),
            Some(1024)
        );
        assert_eq!(decompressed, expected);
    }

    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_golden_decompress_pattern4() {
        // C blosclz_compress(5, [i%4; 1024], ..., 0) -> 14 bytes
        let compressed: &[u8] = &[
            0x23, 0x00, 0x01, 0x02, 0x03, 0xe0, 0xff, 0xff, 0xff, 0xf4, 0x03, 0x01, 0x02, 0x03,
        ];
        let expected: Vec<u8> = (0..1024).map(|i| (i % 4) as u8).collect();
        let mut decompressed = vec![0u8; 1024];
        assert_eq!(
            blosclz_decompress(compressed, &mut decompressed),
            Some(1024)
        );
        assert_eq!(decompressed, expected);
    }

    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_golden_decompress_mixed() {
        // C blosclz_compress(5, mixed 2048-byte pattern, ..., 0) -> 40 bytes
        let compressed: &[u8] = &[
            0x23, 0xaa, 0xaa, 0xaa, 0xaa, 0xe0, 0xff, 0xf4, 0x03, 0x07, 0x00, 0x01, 0x02, 0x03,
            0x04, 0x05, 0x06, 0x07, 0xe0, 0xff, 0xf0, 0x07, 0x00, 0xbb, 0xe0, 0xff, 0xf7, 0x00,
            0x03, 0x00, 0x01, 0x02, 0x03, 0xe0, 0xff, 0xf2, 0x03, 0x01, 0x02, 0x03,
        ];
        let mut expected = vec![0u8; 2048];
        expected[..512].fill(0xAA);
        for (i, v) in expected[512..1024].iter_mut().enumerate() {
            *v = ((512 + i) % 8) as u8;
        }
        expected[1024..1536].fill(0xBB);
        for (i, v) in expected[1536..2048].iter_mut().enumerate() {
            *v = ((1536 + i) % 4) as u8;
        }
        let mut decompressed = vec![0u8; 2048];
        assert_eq!(
            blosclz_decompress(compressed, &mut decompressed),
            Some(2048)
        );
        assert_eq!(decompressed, expected);
    }

    // --- c-blosc 1.12.0 frames (tests/fixtures/gen_blosc.py) ---

    /// `make_input` of gen_blosc.py: kind 0 smooth, 1 random, 2 smooth with
    /// a random tail.
    #[cfg(feature = "blosc")]
    fn blosc_fixture_input(kind: u8, nbytes: usize, typesize: usize) -> Vec<u8> {
        let mut x = 0x9E37_79B9 ^ nbytes as u32 ^ ((typesize as u32) << 20) ^ ((kind as u32) << 28);
        let mut step = || {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            x >> 16
        };
        let mut out = vec![0u8; nbytes];
        if kind == 1 {
            out.iter_mut().for_each(|b| *b = step() as u8);
            return out;
        }
        let w = typesize.min(8);
        let n = nbytes / w;
        for i in 0..n {
            let v = ((i / 7) % 50) as u64 * 3 + u64::from((step() & 1023) == 0);
            out[i * w..(i + 1) * w].copy_from_slice(&v.to_le_bytes()[..w]);
        }
        out[n * w..].fill(0xA5);
        if kind == 2 {
            out[nbytes - 1500..]
                .iter_mut()
                .for_each(|b| *b = step() as u8);
        }
        out
    }

    /// The records of a gen_blosc.py container: (cd_values, kind, nbytes,
    /// frame).
    #[cfg(feature = "blosc")]
    fn blosc_fixture_records(blob: &[u8]) -> Vec<(Vec<u32>, u8, usize, &[u8])> {
        assert_eq!(&blob[..8], b"BLOSCFX1");
        let mut pos = 8;
        let mut records = Vec::new();
        while pos < blob.len() {
            let h = &blob[pos..pos + 16];
            let u16_at = |at: usize| u16::from_le_bytes([h[at], h[at + 1]]) as u32;
            let u32_at = |at: usize| u32::from_le_bytes(h[at..at + 4].try_into().unwrap());
            let (compcode, shuffle, typesize, clevel, kind) =
                (h[0] as u32, h[1] as u32, u16_at(2), h[4] as u32, h[5]);
            let (nbytes, len) = (u32_at(8), u32_at(12) as usize);
            let cd = vec![2, 2, typesize, nbytes, clevel, shuffle, compcode];
            records.push((cd, kind, nbytes as usize, &blob[pos + 16..pos + 16 + len]));
            pos += 16 + len;
        }
        records
    }

    /// Every c-blosc frame decodes, whole and into a short window; BloscLZ
    /// frames, and every MEMCPYED frame, are what this crate writes too.
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_frames_from_c_blosc_1_12() {
        let blob = include_bytes!("../../../tests/fixtures/blosc_c1120.bin");
        let mut decoded = [0usize; 6];
        for (cd, kind, nbytes, frame) in blosc_fixture_records(blob) {
            let compcode = cd[6];
            if blosc_compformat(compcode).is_err() {
                continue; // a sub-codec feature is off
            }
            let ctx = format!("cd {cd:?} kind {kind}");
            let input = blosc_fixture_input(kind, nbytes, cd[2] as usize);
            let pipeline = FilterPipeline {
                filters: vec![Filter {
                    id: FILTER_BLOSC,
                    flags: 0,
                    cd_values: cd.clone(),
                }],
            };
            assert_eq!(reverse_filters(&pipeline, frame).unwrap(), input, "{ctx}");
            let mut out = vec![0u8; nbytes];
            let n = reverse_filters_masked_into(&pipeline, frame, 0, &mut out).unwrap();
            assert_eq!((n, &out), (nbytes, &input), "{ctx} into");
            let mut short = vec![0u8; nbytes / 3];
            let n = reverse_filters_masked_into(&pipeline, frame, 0, &mut short).unwrap();
            assert_eq!(
                (n, &short[..]),
                (nbytes, &input[..nbytes / 3]),
                "{ctx} short"
            );
            // Only the frame's last stream is past its end.
            assert!(
                reverse_filters(&pipeline, &frame[..frame.len() - 1]).is_err(),
                "{ctx}"
            );

            let ours = apply_filters(&pipeline, &input).unwrap();
            if compcode == BLOSC_BLOSCLZ || frame[2] & BLOSC_MEMCPYED != 0 {
                assert_eq!(ours, frame, "{ctx} encode");
            }
            assert_eq!(
                reverse_filters(&pipeline, &ours).unwrap(),
                input,
                "{ctx} ours"
            );
            decoded[compcode as usize] += 1;
        }
        // blosclz, lz4 and lz4hc, snappy; zlib and zstd with their features
        assert_eq!(decoded[..4], [46, 36, 36, 36]);
        #[cfg(feature = "deflate")]
        assert_eq!(decoded[4], 36);
        #[cfg(feature = "zstd")]
        assert_eq!(decoded[5], 36);
    }

    /// Corrupt frames fail without a panic.
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_corrupt_frames_fail() {
        let blob = include_bytes!("../../../tests/fixtures/blosc_c1120.bin");
        for (cd, _, nbytes, frame) in blosc_fixture_records(blob) {
            if blosc_compformat(cd[6]).is_err() {
                continue;
            }
            let mut out = vec![0u8; nbytes];
            for cut in [0, 15, 16, 17, frame.len() / 2] {
                let cut = cut.min(frame.len() - 1);
                assert!(
                    blosc_decompress_into(&frame[..cut], &mut out).is_err(),
                    "cut {cut}"
                );
            }
            let mut bad = frame.to_vec();
            for at in 16..bad.len().min(16 + 4 * 8 + 32) {
                bad[at] ^= 0x5A;
                let _ = blosc_decompress_into(&bad, &mut out);
                bad[at] ^= 0x5A;
            }
        }
    }

    // --- Shuffle + BZIP2 combined golden test ---
    #[cfg(feature = "bzip2")]
    #[test]
    fn shuffle_bzip2_combined_roundtrip() {
        let data = golden_f64_data();
        let pipeline = FilterPipeline {
            filters: vec![
                Filter {
                    id: FILTER_SHUFFLE,
                    flags: 0,
                    cd_values: vec![8],
                },
                Filter {
                    id: FILTER_BZIP2,
                    flags: 0,
                    cd_values: vec![9],
                },
            ],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- Filter pipeline encode/decode with new IDs ---
    #[test]
    fn encode_decode_all_filter_ids() {
        for &(id, ref cd) in &[
            (FILTER_LZF, vec![4u32, 0, 1024]),
            (FILTER_BSHUF, vec![0, 0, 4, 128, 0]),
            (FILTER_BITGROOM, vec![3, 4, 0, 0, 0]),
            (FILTER_BITROUND, vec![3, 8, 0, 0, 0]),
            (FILTER_BLOSC, vec![2, 2, 4, 1024, 5, 1, 1]),
            (FILTER_BZIP2, vec![9]),
        ] {
            let pipeline = FilterPipeline {
                filters: vec![Filter {
                    id,
                    flags: 0,
                    cd_values: cd.to_vec(),
                }],
            };
            let encoded = pipeline.encode();
            let (decoded, _) = FilterPipeline::decode(&encoded).unwrap();
            assert_eq!(decoded.filters[0].id, id);
        }
    }
}
