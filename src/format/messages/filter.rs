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
/// Bitshuffle compression sub-option (cd_values[4]): apply LZ4 after the
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
    /// ([`registered_name`]), because that is what `H5O__pline_encode` reaches
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

    for i in 0..bytesoftype {
        let dest_start = i * numofelements;
        for j in 0..numofelements {
            dest[dest_start + j] = data[j * bytesoftype + i];
        }
    }
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

    for i in 0..bytesoftype {
        let src_start = i * numofelements;
        for j in 0..numofelements {
            dest[j * bytesoftype + i] = data[src_start + j];
        }
    }
    if data.len() > total {
        dest[total..].copy_from_slice(&data[total..]);
    }
    dest
}

/// Inflate one zlib stream, growing a single output buffer.
///
/// The engine is zlib-rs rather than the miniz_oxide behind `flate2`: on the
/// same level-6 streams it inflates 1.1x faster on incompressible chunks and
/// up to 1.8x faster on text-like ones. Compression stays on miniz_oxide
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
        // BLOSC (32001) — decompress only via sub-codec dispatch
        // =====================================================================
        #[cfg(feature = "blosc")]
        FILTER_BLOSC => {
            if compress {
                blosc_compress(data, filter)
            } else {
                blosc_decompress(data, filter)
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
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let mut words = data.len() / 2;
    let mut i = 0;
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
    chunks: &[Vec<u8>],
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

fn lzf_decompress(input: &[u8], max_output: usize) -> FormatResult<Vec<u8>> {
    let mut out = Vec::with_capacity(max_output);
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

fn bitshuffle_block(input: &[u8], elem_size: usize) -> Vec<u8> {
    let n_elems = input.len() / elem_size;
    let nbits = elem_size * 8;
    let mut out = vec![0u8; input.len()];

    for bit in 0..nbits {
        let byte_idx = bit / 8;
        let bit_idx = bit % 8; // LSB-first within the source byte
        for elem in 0..n_elems {
            let src_byte = input[elem * elem_size + byte_idx];
            let src_bit = (src_byte >> bit_idx) & 1;
            let dst_bit_pos = bit * n_elems + elem;
            let dst_byte_idx = dst_bit_pos / 8;
            let dst_bit_idx = dst_bit_pos % 8; // LSB-first within the output byte
            out[dst_byte_idx] |= src_bit << dst_bit_idx;
        }
    }
    out
}

fn bitunshuffle_block(input: &[u8], elem_size: usize) -> Vec<u8> {
    let n_elems = input.len() / elem_size;
    let nbits = elem_size * 8;
    let mut out = vec![0u8; input.len()];

    for bit in 0..nbits {
        let byte_idx = bit / 8;
        let bit_idx = bit % 8; // LSB-first within the destination byte
        for elem in 0..n_elems {
            let src_bit_pos = bit * n_elems + elem;
            let src_byte_idx = src_bit_pos / 8;
            let src_bit_idx = src_bit_pos % 8; // LSB-first within the source byte
            let src_bit = (input[src_byte_idx] >> src_bit_idx) & 1;
            out[elem * elem_size + byte_idx] |= src_bit << bit_idx;
        }
    }
    out
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

    if comp_type == 0 {
        // Bitshuffle only, no compression, no header. Each planned block is
        // bit-transposed; the trailing `n_elems % 8` elements are copied raw.
        let mut out = Vec::with_capacity(data.len());
        let mut elem_pos = 0;
        for &block in &blocks {
            let start = elem_pos * elem_size;
            let end = start + block * elem_size;
            out.extend_from_slice(&bitshuffle_block(&data[start..end], elem_size));
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
        let mut elem_pos = 0;
        for &block in &blocks {
            let start = elem_pos * elem_size;
            let end = start + block * elem_size;
            let shuffled = bitshuffle_block(&data[start..end], elem_size);
            let compressed = lz4_flex::compress(&shuffled);
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
        let mut out = Vec::with_capacity(data.len());
        let mut elem_pos = 0;
        for &block in &blocks {
            let start = elem_pos * elem_size;
            let end = start + block * elem_size;
            out.extend_from_slice(&bitunshuffle_block(&data[start..end], elem_size));
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
            let decompressed = lz4_flex::decompress(&data[rpos..rpos + comp_size], exp_size)
                .map_err(|e| FormatError::InvalidData(format!("bshuf LZ4: {}", e)))?;
            output.extend_from_slice(&bitunshuffle_block(&decompressed, elem_size));
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
// BLOSC (32001) — sub-codec dispatch: BloscLZ, LZ4, LZ4HC, Snappy, Zlib, Zstd
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

#[cfg(feature = "blosc")]
fn blosc_sub_compress(compressor: u32, data: &[u8]) -> FormatResult<Vec<u8>> {
    match compressor {
        BLOSC_BLOSCLZ => Ok(blosclz_compress(data, 5)),
        BLOSC_LZ4 | BLOSC_LZ4HC => {
            // LZ4HC decompresses identically to LZ4; we compress with standard LZ4
            Ok(lz4_flex::compress(data))
        }
        BLOSC_SNAPPY => {
            let mut enc = snap::raw::Encoder::new();
            enc.compress_vec(data)
                .map_err(|e| FormatError::InvalidData(format!("blosc snappy compress: {}", e)))
        }
        #[cfg(feature = "deflate")]
        BLOSC_ZLIB => {
            use flate2::write::ZlibEncoder;
            use flate2::Compression;
            use std::io::Write;
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
            enc.write_all(data)
                .map_err(|e| FormatError::InvalidData(format!("blosc zlib: {}", e)))?;
            enc.finish()
                .map_err(|e| FormatError::InvalidData(format!("blosc zlib: {}", e)))
        }
        #[cfg(not(feature = "deflate"))]
        BLOSC_ZLIB => Err(FormatError::UnsupportedFeature(
            "blosc zlib sub-codec requires the 'deflate' feature".into(),
        )),
        #[cfg(feature = "zstd")]
        BLOSC_ZSTD => Ok(rust_zstd::compress(data, 3)),
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

#[cfg(feature = "blosc")]
fn blosc_sub_decompress(compressor: u32, data: &[u8], nbytes: usize) -> FormatResult<Vec<u8>> {
    match compressor {
        BLOSC_BLOSCLZ => blosclz_decompress(data, nbytes),
        BLOSC_LZ4 | BLOSC_LZ4HC => lz4_flex::decompress(data, nbytes)
            .map_err(|e| FormatError::InvalidData(format!("blosc lz4: {}", e))),
        BLOSC_SNAPPY => {
            let mut dec = snap::raw::Decoder::new();
            dec.decompress_vec(data)
                .map_err(|e| FormatError::InvalidData(format!("blosc snappy: {}", e)))
        }
        #[cfg(feature = "deflate")]
        BLOSC_ZLIB => inflate_zlib(data, Some(nbytes))
            .map_err(|e| FormatError::InvalidData(format!("blosc zlib: {e}"))),
        #[cfg(not(feature = "deflate"))]
        BLOSC_ZLIB => Err(FormatError::UnsupportedFeature(
            "blosc zlib sub-codec requires the 'deflate' feature".into(),
        )),
        #[cfg(feature = "zstd")]
        BLOSC_ZSTD => rust_zstd::decompress(data)
            .map_err(|e| FormatError::InvalidData(format!("blosc zstd: {}", e))),
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

#[cfg(feature = "blosc")]
fn blosc_compress(data: &[u8], filter: &Filter) -> FormatResult<Vec<u8>> {
    let typesize = filter.cd_values.get(2).copied().unwrap_or(1) as usize;
    let doshuffle = filter.cd_values.get(5).copied().unwrap_or(1);
    let compressor = filter.cd_values.get(6).copied().unwrap_or(BLOSC_LZ4);

    // Apply byte-shuffle if requested
    let shuffled = if doshuffle == 1 && typesize > 1 {
        shuffle(data, typesize)
    } else {
        data.to_vec()
    };

    let compressed = blosc_sub_compress(compressor, &shuffled)?;

    // Build blosc header (16 bytes)
    let flags: u8 = if doshuffle == 1 { 0x01 } else { 0x00 };
    let nbytes = data.len() as u32;
    let blocksize = data.len() as u32;
    let cbytes = (16 + compressed.len()) as u32;

    let mut out = Vec::with_capacity(cbytes as usize);
    out.push(2); // blosc format version
    out.push(1); // compressor format version (always 1, matches C blosc)
    out.push(flags);
    out.push(typesize as u8);
    out.extend_from_slice(&nbytes.to_le_bytes());
    out.extend_from_slice(&blocksize.to_le_bytes());
    out.extend_from_slice(&cbytes.to_le_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

#[cfg(feature = "blosc")]
fn blosc_decompress(data: &[u8], filter: &Filter) -> FormatResult<Vec<u8>> {
    if data.len() < 16 {
        return Err(FormatError::InvalidData("blosc: header too short".into()));
    }
    let _version = data[0];
    let flags = data[2];
    let typesize = data[3] as usize;
    let nbytes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

    let compressed_data = &data[16..];
    let compressor = filter.cd_values.get(6).copied().unwrap_or(BLOSC_LZ4);

    // Check for memcpy flag (flags bit 1)
    let decompressed = if flags & 0x02 != 0 {
        compressed_data[..nbytes].to_vec()
    } else {
        blosc_sub_decompress(compressor, compressed_data, nbytes)?
    };

    // Unshuffle if byte-shuffle flag set
    if flags & 0x01 != 0 && typesize > 1 {
        Ok(unshuffle(&decompressed, typesize))
    } else {
        Ok(decompressed)
    }
}

// =========================================================================
// BloscLZ — pure Rust port of the FastLZ-derived LZ77 compressor
// =========================================================================

#[cfg(feature = "blosc")]
const BLOSCLZ_MAX_COPY: usize = 32;
#[cfg(feature = "blosc")]
const BLOSCLZ_MAX_DISTANCE: usize = 8191;
#[cfg(feature = "blosc")]
const BLOSCLZ_MAX_FARDISTANCE: usize = 65535 + BLOSCLZ_MAX_DISTANCE - 1;

#[cfg(feature = "blosc")]
fn blosclz_hash(seq: u32, hashlog: u32) -> u32 {
    seq.wrapping_mul(2654435761) >> (32 - hashlog)
}

#[cfg(feature = "blosc")]
fn blosclz_compress(input: &[u8], clevel: u32) -> Vec<u8> {
    let length = input.len();
    if length < 16 {
        return Vec::new();
    }

    let hashlog_table: [u32; 10] = [0, 12, 13, 14, 14, 14, 14, 14, 14, 14];
    let clevel = (clevel as usize).clamp(1, 9);
    let hashlog = hashlog_table[clevel];
    let htab_size = 1usize << hashlog;
    let mut htab = vec![0u32; htab_size];

    let ipshift: usize = if clevel <= 2 { 3 } else { 4 };
    let minlen: usize = if clevel <= 2 { 3 } else { 4 };

    let maxout = length + length / 20 + 66;
    let mut out = Vec::with_capacity(maxout);
    let op_limit = maxout;

    let ip_bound = length - 1;
    let ip_limit = if length >= 12 {
        length - 12
    } else {
        return Vec::new();
    };

    // Start with literal copy
    let mut copy: usize = 4;
    out.push((BLOSCLZ_MAX_COPY - 1) as u8);
    out.push(input[0]);
    out.push(input[1]);
    out.push(input[2]);
    out.push(input[3]);
    let mut ip: usize = 4;

    while ip < ip_limit {
        if out.len() + 2 > op_limit {
            return Vec::new();
        }

        let anchor = ip;
        let seq = u32::from_le_bytes([input[ip], input[ip + 1], input[ip + 2], input[ip + 3]]);
        let hval = blosclz_hash(seq, hashlog) as usize;
        let r = htab[hval] as usize;
        let distance = anchor.wrapping_sub(r);

        htab[hval] = anchor as u32;

        if distance == 0 || distance >= BLOSCLZ_MAX_FARDISTANCE {
            // literal
            out.push(input[anchor]);
            ip = anchor + 1;
            copy += 1;
            if copy == BLOSCLZ_MAX_COPY {
                copy = 0;
                if out.len() + 1 > op_limit {
                    return Vec::new();
                }
                out.push((BLOSCLZ_MAX_COPY - 1) as u8);
            }
            continue;
        }

        // Check first 4 bytes for match
        if r + 3 < length
            && input[r] == input[ip]
            && input[r + 1] == input[ip + 1]
            && input[r + 2] == input[ip + 2]
            && input[r + 3] == input[ip + 3]
        {
            // match found
        } else {
            // literal
            out.push(input[anchor]);
            ip = anchor + 1;
            copy += 1;
            if copy == BLOSCLZ_MAX_COPY {
                copy = 0;
                if out.len() + 1 > op_limit {
                    return Vec::new();
                }
                out.push((BLOSCLZ_MAX_COPY - 1) as u8);
            }
            continue;
        }

        // Extend match
        let mut ref_pos = r + 4;
        ip = anchor + 4;
        let dist_biased = distance - 1;

        if dist_biased == 0 {
            // RLE run
            let x = input[ip - 1];
            while ip < ip_bound && ref_pos < length && input[ref_pos] == x {
                ip += 1;
                ref_pos += 1;
            }
        } else {
            while ip < ip_bound && ref_pos < length && input[ref_pos] == input[ip] {
                ip += 1;
                ref_pos += 1;
            }
        }

        // Bias length
        if ip > ipshift {
            ip -= ipshift;
        } else {
            ip = anchor + 1;
            out.push(input[anchor]);
            copy += 1;
            if copy == BLOSCLZ_MAX_COPY {
                copy = 0;
                if out.len() + 1 > op_limit {
                    return Vec::new();
                }
                out.push((BLOSCLZ_MAX_COPY - 1) as u8);
            }
            continue;
        }

        let len = ip - anchor;
        if len < minlen || (len <= 5 && dist_biased >= BLOSCLZ_MAX_DISTANCE) {
            ip = anchor + 1;
            out.push(input[anchor]);
            copy += 1;
            if copy == BLOSCLZ_MAX_COPY {
                copy = 0;
                if out.len() + 1 > op_limit {
                    return Vec::new();
                }
                out.push((BLOSCLZ_MAX_COPY - 1) as u8);
            }
            continue;
        }

        // Adjust copy count
        if copy > 0 {
            let idx = out.len() - copy - 1;
            out[idx] = (copy - 1) as u8;
        } else {
            out.pop();
        }
        copy = 0;

        // Encode match
        if dist_biased < BLOSCLZ_MAX_DISTANCE {
            if len < 7 {
                if out.len() + 2 > op_limit {
                    return Vec::new();
                }
                out.push(((len << 5) + (dist_biased >> 8)) as u8);
                out.push((dist_biased & 255) as u8);
            } else {
                if out.len() + 1 > op_limit {
                    return Vec::new();
                }
                out.push(((7 << 5) + (dist_biased >> 8)) as u8);
                let mut remaining = len - 7;
                while remaining >= 255 {
                    if out.len() + 1 > op_limit {
                        return Vec::new();
                    }
                    out.push(255);
                    remaining -= 255;
                }
                if out.len() + 2 > op_limit {
                    return Vec::new();
                }
                out.push(remaining as u8);
                out.push((dist_biased & 255) as u8);
            }
        } else {
            // Far distance
            let far_dist = dist_biased - BLOSCLZ_MAX_DISTANCE;
            if len < 7 {
                if out.len() + 4 > op_limit {
                    return Vec::new();
                }
                out.push(((len << 5) + 31) as u8);
                out.push(255);
                out.push((far_dist >> 8) as u8);
                out.push((far_dist & 255) as u8);
            } else {
                if out.len() + 1 > op_limit {
                    return Vec::new();
                }
                out.push((7 << 5) + 31);
                let mut remaining = len - 7;
                while remaining >= 255 {
                    if out.len() + 1 > op_limit {
                        return Vec::new();
                    }
                    out.push(255);
                    remaining -= 255;
                }
                if out.len() + 4 > op_limit {
                    return Vec::new();
                }
                out.push(remaining as u8);
                out.push(255);
                out.push((far_dist >> 8) as u8);
                out.push((far_dist & 255) as u8);
            }
        }

        // Update hash at match boundary
        if ip + 3 < length {
            let seq = u32::from_le_bytes([input[ip], input[ip + 1], input[ip + 2], input[ip + 3]]);
            let hval = blosclz_hash(seq, hashlog) as usize;
            htab[hval] = ip as u32;
        }
        ip += 1;
        if clevel == 9 && ip + 3 < length {
            let seq = u32::from_le_bytes([input[ip], input[ip + 1], input[ip + 2], input[ip + 3]]);
            let hval = blosclz_hash(seq, hashlog) as usize;
            htab[hval] = ip as u32;
        }
        ip += 1;

        if out.len() + 1 > op_limit {
            return Vec::new();
        }
        out.push((BLOSCLZ_MAX_COPY - 1) as u8);
    }

    // Left-over as literal copy
    while ip <= ip_bound {
        if out.len() + 2 > op_limit {
            return Vec::new();
        }
        out.push(input[ip]);
        ip += 1;
        copy += 1;
        if copy == BLOSCLZ_MAX_COPY {
            copy = 0;
            if out.len() + 1 > op_limit {
                return Vec::new();
            }
            out.push((BLOSCLZ_MAX_COPY - 1) as u8);
        }
    }

    // Finalize
    if copy > 0 {
        let idx = out.len() - copy - 1;
        out[idx] = (copy - 1) as u8;
    } else {
        out.pop();
    }

    // Set BloscLZ marker (bit 5 of first byte)
    if !out.is_empty() {
        out[0] |= 1 << 5;
    }

    out
}

#[cfg(feature = "blosc")]
fn blosclz_decompress(input: &[u8], maxout: usize) -> FormatResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(maxout);
    let mut ip = 0usize;
    let ip_limit = input.len();

    // First byte: strip BloscLZ marker (bit 5)
    let mut ctrl = (input[ip] & 31) as u32;
    ip += 1;

    loop {
        if ctrl >= 32 {
            // Match
            let mut len = ((ctrl >> 5) - 1) as usize;
            let ofs = ((ctrl & 31) << 8) as usize;

            if len == 6 {
                loop {
                    if ip >= ip_limit {
                        return Err(FormatError::InvalidData("blosclz: truncated".into()));
                    }
                    let code = input[ip] as usize;
                    ip += 1;
                    len += code;
                    if code != 255 {
                        break;
                    }
                }
            }

            if ip >= ip_limit {
                return Err(FormatError::InvalidData("blosclz: truncated".into()));
            }
            let code = input[ip] as usize;
            ip += 1;
            len += 3;

            let mut ref_offset = ofs + code; // distance from current output pos

            // Far distance
            if code == 255 && ofs == (31 << 8) {
                if ip + 1 >= ip_limit {
                    return Err(FormatError::InvalidData("blosclz: truncated far".into()));
                }
                let far_ofs = ((input[ip] as usize) << 8) + input[ip + 1] as usize;
                ip += 2;
                ref_offset = far_ofs + BLOSCLZ_MAX_DISTANCE;
            }

            ref_offset += 1; // distance is biased by 1

            if ref_offset > out.len() {
                return Err(FormatError::InvalidData(
                    "blosclz: bad back-reference".into(),
                ));
            }
            if out.len() + len > maxout {
                return Err(FormatError::InvalidData("blosclz: output overflow".into()));
            }

            let ref_start = out.len() - ref_offset;
            if ref_offset == 1 {
                // RLE: repeat single byte
                let b = out[ref_start];
                out.resize(out.len() + len, b);
            } else {
                // Copy with possible overlap
                for i in 0..len {
                    let b = out[ref_start + i];
                    out.push(b);
                }
            }

            if ip >= ip_limit {
                break;
            }
            ctrl = input[ip] as u32;
            ip += 1;
        } else {
            // Literal
            let count = (ctrl + 1) as usize;
            if out.len() + count > maxout {
                return Err(FormatError::InvalidData("blosclz: output overflow".into()));
            }
            if ip + count > ip_limit {
                return Err(FormatError::InvalidData(
                    "blosclz: truncated literal".into(),
                ));
            }
            out.extend_from_slice(&input[ip..ip + count]);
            ip += count;

            if ip >= ip_limit {
                break;
            }
            ctrl = input[ip] as u32;
            ip += 1;
        }
    }

    Ok(out)
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

        let compressed = apply_filters_parallel(&pipeline, &chunks).unwrap();
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
        let chunks: Vec<Vec<u8>> = vec![vec![1u8; 64], vec![2u8; 64]];
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
        let compressed = blosclz_compress(&data, 5);
        assert!(!compressed.is_empty());
        let decompressed = blosclz_decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- BloscLZ with highly compressible data ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosclz_repeated_data() {
        let data = vec![42u8; 8192];
        let compressed = blosclz_compress(&data, 5);
        assert!(!compressed.is_empty());
        assert!(compressed.len() < data.len());
        let decompressed = blosclz_decompress(&compressed, data.len()).unwrap();
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
        let decompressed = blosclz_decompress(compressed, 512).unwrap();
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
        let decompressed = blosclz_decompress(compressed, 1024).unwrap();
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
        let decompressed = blosclz_decompress(compressed, 1024).unwrap();
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
        let decompressed = blosclz_decompress(compressed, 2048).unwrap();
        assert_eq!(decompressed, expected);
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
