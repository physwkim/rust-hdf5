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
pub const FILTER_ZFP: u16 = 32013;
pub const FILTER_ZSTD: u16 = 32015;
pub const FILTER_JPEG: u16 = 32019;
pub const FILTER_BITGROOM: u16 = 32022;
pub const FILTER_BITROUND: u16 = 32023;
pub const FILTER_BLOSC2: u16 = 32026;

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
                flags: 0, // mandatory
                cd_values: vec![level],
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
                    flags: 0,
                    cd_values: vec![element_size],
                },
                Filter {
                    id: FILTER_DEFLATE,
                    flags: 0,
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
                flags: 0,
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
                flags: 0,
                cd_values: vec![level],
            }],
        }
    }

    /// Create an empty pipeline (no filters).
    pub fn none() -> Self {
        Self {
            filters: Vec::new(),
        }
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

    /// Decode a version-2 filter pipeline message.
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }

        let version = buf[0];
        if version != 2 {
            return Err(FormatError::InvalidVersion(version));
        }

        let nfilters = buf[1] as usize;
        let mut pos = 2;
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

            // For user-defined filters (id >= 256), read and skip the name.
            if id >= 256 {
                if buf.len() < pos + 2 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 2,
                        available: buf.len(),
                    });
                }
                let name_len = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                // Name is padded to a multiple of 8 bytes
                let padded_len = (name_len + 7) & !7;
                if buf.len() < pos + padded_len {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + padded_len,
                        available: buf.len(),
                    });
                }
                pos += padded_len;
            }

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
    let mut buf = data.to_vec();
    for filter in &pipeline.filters {
        buf = apply_single_filter(filter, &buf, true)?;
    }
    Ok(buf)
}

/// Reverse filter pipeline to decompress raw chunk data.
///
/// Filters are applied in reverse order. Returns the decompressed data.
pub fn reverse_filters(pipeline: &FilterPipeline, data: &[u8]) -> FormatResult<Vec<u8>> {
    let mut buf = data.to_vec();
    for filter in pipeline.filters.iter().rev() {
        buf = apply_single_filter(filter, &buf, false)?;
    }
    Ok(buf)
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
                use flate2::read::ZlibDecoder;
                use std::io::Read;

                let mut decoder = ZlibDecoder::new(data);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out).map_err(|e| {
                    FormatError::InvalidData(format!("deflate decompress error: {}", e))
                })?;
                Ok(out)
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
                // Fletcher-32 appends a 4-byte checksum
                let cksum = fletcher32(data);
                let mut out = data.to_vec();
                out.extend_from_slice(&cksum.to_be_bytes());
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
            // cd_values: [options_mask, bits_per_pixel, pixels_per_block, pixels_per_scanline]
            let options_mask = filter.cd_values.first().copied().unwrap_or(0);
            let bits_per_pixel = filter.cd_values.get(1).copied().unwrap_or(8);
            let pixels_per_block = filter.cd_values.get(2).copied().unwrap_or(32);
            let pixels_per_scanline = filter.cd_values.get(3).copied().unwrap_or(256);
            if compress {
                crate::format::szip::compress(
                    data,
                    bits_per_pixel,
                    pixels_per_block,
                    pixels_per_scanline,
                    options_mask,
                )
                .map_err(|e| FormatError::InvalidData(format!("SZIP compress: {}", e)))
            } else {
                let output_size = filter.cd_values.get(4).copied().unwrap_or(0) as usize;
                let out_size = if output_size > 0 {
                    output_size
                } else {
                    data.len() * 4
                };
                crate::format::szip::decompress(
                    data,
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
                bitshuffle_decompress(data, elem_size, comp_type)
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

/// Compute Fletcher-32 checksum over a byte buffer.
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    // Process 16-bit words (big-endian pairs per HDF5 spec)
    let mut i = 0;
    while i + 1 < data.len() {
        let word = ((data[i] as u32) << 8) | (data[i + 1] as u32);
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
        i += 2;
    }
    // Handle odd byte
    if i < data.len() {
        let word = (data[i] as u32) << 8;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }

    (sum2 << 16) | sum1
}

/// Compress multiple chunks in parallel using rayon.
///
/// Each chunk is independently compressed through the filter pipeline.
/// If compression of a chunk fails, the original (uncompressed) data is used.
#[cfg(feature = "parallel")]
pub fn apply_filters_parallel(pipeline: &FilterPipeline, chunks: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    chunks
        .par_iter()
        .map(|chunk| apply_filters(pipeline, chunk).unwrap_or_else(|_| chunk.clone()))
        .collect()
}

/// Decompress multiple chunks in parallel using rayon.
///
/// Each chunk is independently decompressed through the reversed filter pipeline.
/// If decompression of a chunk fails, the original data is used.
#[cfg(feature = "parallel")]
pub fn reverse_filters_parallel(pipeline: &FilterPipeline, chunks: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    chunks
        .par_iter()
        .map(|chunk| reverse_filters(pipeline, chunk).unwrap_or_else(|_| chunk.clone()))
        .collect()
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
// =========================================================================

fn bitshuffle_block(input: &[u8], elem_size: usize) -> Vec<u8> {
    let n_elems = input.len() / elem_size;
    let nbits = elem_size * 8;
    let mut out = vec![0u8; input.len()];

    for bit in 0..nbits {
        let byte_idx = bit / 8;
        let bit_idx = 7 - (bit % 8);
        for elem in 0..n_elems {
            let src_byte = input[elem * elem_size + byte_idx];
            let src_bit = (src_byte >> bit_idx) & 1;
            let dst_bit_pos = bit * n_elems + elem;
            let dst_byte_idx = dst_bit_pos / 8;
            let dst_bit_idx = 7 - (dst_bit_pos % 8);
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
        let bit_idx = 7 - (bit % 8);
        for elem in 0..n_elems {
            let src_bit_pos = bit * n_elems + elem;
            let src_byte_idx = src_bit_pos / 8;
            let src_bit_idx = 7 - (src_bit_pos % 8);
            let src_bit = (input[src_byte_idx] >> src_bit_idx) & 1;
            out[elem * elem_size + byte_idx] |= src_bit << bit_idx;
        }
    }
    out
}

fn bitshuffle_compress(
    data: &[u8],
    elem_size: usize,
    mut block_size: usize,
    comp_type: u32,
    _filter: &Filter,
) -> FormatResult<Vec<u8>> {
    if elem_size == 0 {
        return Ok(data.to_vec());
    }
    let n_elems = data.len() / elem_size;
    if block_size == 0 {
        block_size = std::cmp::max(128, 8192 / elem_size);
    }
    block_size = (block_size / 8) * 8; // round down to multiple of 8
    if block_size < 8 {
        block_size = 8;
    }

    if comp_type == 0 {
        // Bitshuffle only, no compression, no header
        let usable = (n_elems / block_size) * block_size;
        let mut out = Vec::with_capacity(data.len());
        let mut pos = 0;
        while pos < usable * elem_size {
            let end = std::cmp::min(pos + block_size * elem_size, usable * elem_size);
            out.extend_from_slice(&bitshuffle_block(&data[pos..end], elem_size));
            pos = end;
        }
        out.extend_from_slice(&data[usable * elem_size..]);
        return Ok(out);
    }

    // With compression: write 12-byte header
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u64).to_be_bytes());
    out.extend_from_slice(&((block_size * elem_size) as u32).to_be_bytes());

    let usable = (n_elems / block_size) * block_size;
    let mut pos = 0;
    while pos < usable * elem_size {
        let end = std::cmp::min(pos + block_size * elem_size, usable * elem_size);
        let shuffled = bitshuffle_block(&data[pos..end], elem_size);

        #[cfg(feature = "lz4")]
        if comp_type == 2 {
            let compressed = lz4_flex::compress(&shuffled);
            out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
            out.extend_from_slice(&compressed);
            pos = end;
            continue;
        }

        // Fallback: store uncompressed
        out.extend_from_slice(&(shuffled.len() as u32).to_be_bytes());
        out.extend_from_slice(&shuffled);
        pos = end;
    }
    // Leftover
    if usable * elem_size < data.len() {
        out.extend_from_slice(&data[usable * elem_size..]);
    }
    Ok(out)
}

fn bitshuffle_decompress(data: &[u8], elem_size: usize, comp_type: u32) -> FormatResult<Vec<u8>> {
    if comp_type == 0 {
        // No compression header — bitunshuffle block-by-block (matching compress)
        if elem_size == 0 {
            return Ok(data.to_vec());
        }
        let n_elems = data.len() / elem_size;
        let mut block_size = std::cmp::max(128, 8192 / elem_size);
        block_size = (block_size / 8) * 8;
        if block_size < 8 {
            block_size = 8;
        }
        let usable = (n_elems / block_size) * block_size;
        let mut out = Vec::with_capacity(data.len());
        let mut pos = 0;
        while pos < usable * elem_size {
            let end = std::cmp::min(pos + block_size * elem_size, usable * elem_size);
            out.extend_from_slice(&bitunshuffle_block(&data[pos..end], elem_size));
            pos = end;
        }
        out.extend_from_slice(&data[usable * elem_size..]);
        return Ok(out);
    }

    if data.len() < 12 {
        return Err(FormatError::InvalidData(
            "bitshuffle: header too short".into(),
        ));
    }
    let orig_size = u64::from_be_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;
    let block_bytes = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let block_elems = if elem_size > 0 {
        block_bytes / elem_size
    } else {
        0
    };

    let mut output = Vec::with_capacity(orig_size);
    let mut rpos = 12;

    let n_elems = orig_size / elem_size;
    let usable = (n_elems / block_elems) * block_elems;
    let mut elems_done = 0;

    while elems_done < usable {
        if rpos + 4 > data.len() {
            break;
        }
        let comp_size =
            u32::from_be_bytes([data[rpos], data[rpos + 1], data[rpos + 2], data[rpos + 3]])
                as usize;
        rpos += 4;
        if rpos + comp_size > data.len() {
            break;
        }

        let cur_elems = std::cmp::min(block_elems, usable - elems_done);
        let _exp_size = cur_elems * elem_size;

        #[cfg(feature = "lz4")]
        if comp_size < _exp_size {
            let decompressed = lz4_flex::decompress(&data[rpos..rpos + comp_size], _exp_size)
                .map_err(|e| FormatError::InvalidData(format!("bshuf LZ4: {}", e)))?;
            output.extend_from_slice(&bitunshuffle_block(&decompressed, elem_size));
            rpos += comp_size;
            elems_done += cur_elems;
            continue;
        }

        // Uncompressed or no LZ4 available
        output.extend_from_slice(&bitunshuffle_block(
            &data[rpos..rpos + comp_size],
            elem_size,
        ));
        rpos += comp_size;
        elems_done += cur_elems;
    }

    // Leftover raw bytes
    if rpos < data.len() && output.len() < orig_size {
        let remaining = std::cmp::min(data.len() - rpos, orig_size - output.len());
        output.extend_from_slice(&data[rpos..rpos + remaining]);
    }
    Ok(output)
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
// BLOSC (32001) — compress via LZ4 sub-codec, decompress by parsing header
// =========================================================================

#[cfg(feature = "blosc")]
fn blosc_compress(data: &[u8], filter: &Filter) -> FormatResult<Vec<u8>> {
    let typesize = filter.cd_values.get(2).copied().unwrap_or(1) as usize;
    let _clevel = filter.cd_values.get(4).copied().unwrap_or(5);
    let doshuffle = filter.cd_values.get(5).copied().unwrap_or(1);
    let _compressor = filter.cd_values.get(6).copied().unwrap_or(1); // default LZ4

    // Apply byte-shuffle if requested
    let shuffled = if doshuffle == 1 && typesize > 1 {
        shuffle(data, typesize)
    } else {
        data.to_vec()
    };

    // Compress with LZ4
    let compressed = lz4_flex::compress(&shuffled);

    // Build blosc header (16 bytes)
    let flags: u8 = if doshuffle == 1 { 0x01 } else { 0x00 };
    let nbytes = data.len() as u32;
    let blocksize = data.len() as u32;
    let cbytes = (16 + compressed.len()) as u32;

    let mut out = Vec::with_capacity(cbytes as usize);
    out.push(2); // blosc format version
    out.push(1); // lz4 version
    out.push(flags);
    out.push(typesize as u8);
    out.extend_from_slice(&nbytes.to_le_bytes());
    out.extend_from_slice(&blocksize.to_le_bytes());
    out.extend_from_slice(&cbytes.to_le_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

#[cfg(feature = "blosc")]
fn blosc_decompress(data: &[u8]) -> FormatResult<Vec<u8>> {
    if data.len() < 16 {
        return Err(FormatError::InvalidData("blosc: header too short".into()));
    }
    let _version = data[0];
    let _versionlz = data[1];
    let flags = data[2];
    let typesize = data[3] as usize;
    let nbytes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let _blocksize = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let _cbytes = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let compressed_data = &data[16..];

    // Check for memcpy flag (flags bit 1)
    let decompressed = if flags & 0x02 != 0 {
        compressed_data[..nbytes].to_vec()
    } else {
        lz4_flex::decompress(compressed_data, nbytes)
            .map_err(|e| FormatError::InvalidData(format!("blosc lz4: {}", e)))?
    };

    // Unshuffle if byte-shuffle flag set
    if flags & 0x01 != 0 && typesize > 1 {
        Ok(unshuffle(&decompressed, typesize))
    } else {
        Ok(decompressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let buf = [1u8, 0]; // version 1
        let err = FilterPipeline::decode(&buf).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(1)));
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

    #[cfg(all(feature = "deflate", feature = "parallel"))]
    #[test]
    fn parallel_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline::deflate(6);
        let chunks: Vec<Vec<u8>> = (0..8)
            .map(|i| vec![(i as u8).wrapping_mul(42); 1024])
            .collect();

        let compressed = apply_filters_parallel(&pipeline, &chunks);
        assert_eq!(compressed.len(), 8);
        // Each compressed chunk should be smaller (repeated data compresses well)
        for c in &compressed {
            assert!(c.len() < 1024);
        }

        let decompressed = reverse_filters_parallel(&pipeline, &compressed);
        assert_eq!(decompressed.len(), 8);
        for (original, decoded) in chunks.iter().zip(decompressed.iter()) {
            assert_eq!(original, decoded);
        }
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

    // --- BLOSC roundtrip ---
    #[cfg(feature = "blosc")]
    #[test]
    fn blosc_roundtrip() {
        let data = golden_f32_data();
        let pipeline = FilterPipeline {
            filters: vec![Filter {
                id: FILTER_BLOSC,
                flags: 0,
                cd_values: vec![2, 2, 4, data.len() as u32, 5, 1, 1],
            }],
        };
        let compressed = apply_filters(&pipeline, &data).unwrap();
        // Verify blosc header
        assert!(compressed.len() >= 16);
        assert_eq!(compressed[0], 2); // format version
        let nbytes =
            u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]);
        assert_eq!(nbytes as usize, data.len());

        let decompressed = reverse_filters(&pipeline, &compressed).unwrap();
        assert_eq!(decompressed, data);
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
