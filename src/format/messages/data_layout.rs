//! Data layout message (type 0x08) — describes how raw data is stored.
//!
//! Binary layout (version 3):
//!   Byte 0: version = 3
//!   Byte 1: layout class (0=compact, 1=contiguous, 2=chunked)
//!
//!   Contiguous (class 1):
//!     address: sizeof_addr bytes
//!     size:    sizeof_size bytes
//!
//!   Compact (class 0):
//!     compact_size: u16 LE
//!     data:         compact_size bytes
//!
//! Binary layout (version 4, chunked only):
//!   Byte 0: version = 4
//!   Byte 1: layout class = 2 (chunked)
//!   flags(1) + ndims(1) + enc_bytes_per_dim(1)
//!   + dim_sizes(ndims * enc_bytes_per_dim, each LE)
//!   + index_type(1)
//!   + [for earray: 5 param bytes]
//!   + index_address(sizeof_addr)

use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

const VERSION_3: u8 = 3;
const VERSION_4: u8 = 4;
const CLASS_COMPACT: u8 = 0;
const CLASS_CONTIGUOUS: u8 = 1;
const CLASS_CHUNKED: u8 = 2;

/// Chunk index type for version-4 chunked layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChunkIndexType {
    SingleChunk = 1,
    Implicit = 2,
    FixedArray = 3,
    ExtensibleArray = 4,
    BTreeV2 = 5,
}

impl ChunkIndexType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::SingleChunk),
            2 => Some(Self::Implicit),
            3 => Some(Self::FixedArray),
            4 => Some(Self::ExtensibleArray),
            5 => Some(Self::BTreeV2),
            _ => None,
        }
    }
}

/// Parameters for the extensible array chunk index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EarrayParams {
    pub max_nelmts_bits: u8,
    pub idx_blk_elmts: u8,
    pub sup_blk_min_data_ptrs: u8,
    pub data_blk_min_elmts: u8,
    pub max_dblk_page_nelmts_bits: u8,
}

impl EarrayParams {
    /// Default extensible array parameters (from H5Dpkg.h).
    pub fn default_params() -> Self {
        Self {
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            sup_blk_min_data_ptrs: 4,
            data_blk_min_elmts: 16,
            max_dblk_page_nelmts_bits: 10,
        }
    }
}

/// Parameters for the fixed array chunk index (max_dblk_page_nelmts_bits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedArrayParams {
    pub max_dblk_page_nelmts_bits: u8,
}

impl FixedArrayParams {
    pub fn default_params() -> Self {
        Self {
            max_dblk_page_nelmts_bits: 0,
        }
    }
}

/// Data layout message payload.
#[derive(Debug, Clone, PartialEq)]
pub enum DataLayoutMessage {
    /// Contiguous storage — raw data in a single block.
    Contiguous {
        /// Address of raw data.  `UNDEF_ADDR` if not yet allocated.
        address: u64,
        /// Size of raw data in bytes.
        size: u64,
    },
    /// Compact storage — raw data stored within the object header.
    Compact {
        /// The raw data bytes.
        data: Vec<u8>,
    },
    /// Version 4 chunked storage.
    ChunkedV4 {
        flags: u8,
        /// Chunk dimension sizes.
        chunk_dims: Vec<u64>,
        /// Type of chunk index structure.
        index_type: ChunkIndexType,
        /// Extensible array parameters (present when index_type == ExtensibleArray).
        earray_params: Option<EarrayParams>,
        /// Fixed array parameters (present when index_type == FixedArray).
        farray_params: Option<FixedArrayParams>,
        /// Address of the chunk index structure.
        index_address: u64,
    },
}

impl DataLayoutMessage {
    /// Contiguous layout with no data allocated yet.
    pub fn contiguous_unallocated(size: u64) -> Self {
        Self::Contiguous {
            address: UNDEF_ADDR,
            size,
        }
    }

    /// Contiguous layout pointing to allocated data.
    pub fn contiguous(address: u64, size: u64) -> Self {
        Self::Contiguous { address, size }
    }

    /// Compact layout with inline data.
    pub fn compact(data: Vec<u8>) -> Self {
        Self::Compact { data }
    }

    /// Version 4 chunked layout with extensible array index.
    ///
    /// `chunk_dims` should include the trailing element-size dimension.
    /// For example, for a 2D dataset with chunk=(1,4) and element_size=8,
    /// pass chunk_dims = [1, 4, 8].
    pub fn chunked_v4_earray(
        chunk_dims: Vec<u64>,
        earray_params: EarrayParams,
        index_address: u64,
    ) -> Self {
        Self::ChunkedV4 {
            flags: 0,
            chunk_dims,
            index_type: ChunkIndexType::ExtensibleArray,
            earray_params: Some(earray_params),
            farray_params: None,
            index_address,
        }
    }

    /// Version 4 chunked layout with fixed array index.
    ///
    /// `chunk_dims` should include the trailing element-size dimension.
    pub fn chunked_v4_farray(
        chunk_dims: Vec<u64>,
        farray_params: FixedArrayParams,
        index_address: u64,
    ) -> Self {
        Self::ChunkedV4 {
            flags: 0,
            chunk_dims,
            index_type: ChunkIndexType::FixedArray,
            earray_params: None,
            farray_params: Some(farray_params),
            index_address,
        }
    }

    /// Version 4 chunked layout with B-tree v2 index.
    ///
    /// `chunk_dims` should include the trailing element-size dimension.
    pub fn chunked_v4_btree_v2(chunk_dims: Vec<u64>, index_address: u64) -> Self {
        Self::ChunkedV4 {
            flags: 0,
            chunk_dims,
            index_type: ChunkIndexType::BTreeV2,
            earray_params: None,
            farray_params: None,
            index_address,
        }
    }

    /// Version 4 chunked layout with single-chunk index.
    ///
    /// `chunk_dims` should include the trailing element-size dimension.
    pub fn chunked_v4_single(chunk_dims: Vec<u64>, index_address: u64) -> Self {
        Self::ChunkedV4 {
            flags: 0,
            chunk_dims,
            index_type: ChunkIndexType::SingleChunk,
            earray_params: None,
            farray_params: None,
            index_address,
        }
    }

    // ------------------------------------------------------------------ encode

    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        match self {
            Self::Contiguous { address, size } => {
                let sa = ctx.sizeof_addr as usize;
                let ss = ctx.sizeof_size as usize;
                let mut buf = Vec::with_capacity(2 + sa + ss);
                buf.push(VERSION_3);
                buf.push(CLASS_CONTIGUOUS);
                buf.extend_from_slice(&address.to_le_bytes()[..sa]);
                buf.extend_from_slice(&size.to_le_bytes()[..ss]);
                buf
            }
            Self::Compact { data } => {
                let mut buf = Vec::with_capacity(2 + 2 + data.len());
                buf.push(VERSION_3);
                buf.push(CLASS_COMPACT);
                buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
                buf.extend_from_slice(data);
                buf
            }
            Self::ChunkedV4 {
                flags,
                chunk_dims,
                index_type,
                earray_params,
                farray_params,
                index_address,
            } => {
                let sa = ctx.sizeof_addr as usize;
                let ndims = chunk_dims.len() as u8;

                // Compute enc_bytes_per_dim: minimum bytes to represent the
                // max chunk dimension value.
                let max_dim = chunk_dims.iter().copied().max().unwrap_or(1);
                let enc_bytes = enc_bytes_for_value(max_dim);

                let mut buf = Vec::with_capacity(64);
                buf.push(VERSION_4);
                buf.push(CLASS_CHUNKED);
                buf.push(*flags);
                buf.push(ndims);
                buf.push(enc_bytes);

                // Dimension sizes
                for &d in chunk_dims {
                    buf.extend_from_slice(&d.to_le_bytes()[..enc_bytes as usize]);
                }

                // Index type
                buf.push(*index_type as u8);

                // Index-type-specific parameters
                match *index_type {
                    ChunkIndexType::ExtensibleArray => {
                        if let Some(ref params) = earray_params {
                            buf.push(params.max_nelmts_bits);
                            buf.push(params.idx_blk_elmts);
                            buf.push(params.sup_blk_min_data_ptrs);
                            buf.push(params.data_blk_min_elmts);
                            buf.push(params.max_dblk_page_nelmts_bits);
                        }
                    }
                    ChunkIndexType::FixedArray => {
                        if let Some(ref params) = farray_params {
                            buf.push(params.max_dblk_page_nelmts_bits);
                        }
                    }
                    // BTreeV2, SingleChunk, Implicit: no extra parameters
                    _ => {}
                }

                // Index address
                buf.extend_from_slice(&index_address.to_le_bytes()[..sa]);

                buf
            }
        }
    }

    // ------------------------------------------------------------------ decode

    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }

        let version = buf[0];
        let class = buf[1];

        match (version, class) {
            (VERSION_3, CLASS_CONTIGUOUS) => {
                let sa = ctx.sizeof_addr as usize;
                let ss = ctx.sizeof_size as usize;
                let mut pos = 2;
                let needed = pos + sa + ss;
                if buf.len() < needed {
                    return Err(FormatError::BufferTooShort {
                        needed,
                        available: buf.len(),
                    });
                }
                let address = read_addr(&buf[pos..], sa);
                pos += sa;
                let size = read_size(&buf[pos..], ss);
                pos += ss;
                Ok((Self::Contiguous { address, size }, pos))
            }
            (VERSION_3, CLASS_COMPACT) => {
                let mut pos = 2;
                if buf.len() < pos + 2 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 2,
                        available: buf.len(),
                    });
                }
                let compact_size = u16::from_le_bytes([buf[pos], buf[pos + 1]]) as usize;
                pos += 2;
                if buf.len() < pos + compact_size {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + compact_size,
                        available: buf.len(),
                    });
                }
                let data = buf[pos..pos + compact_size].to_vec();
                pos += compact_size;
                Ok((Self::Compact { data }, pos))
            }
            (VERSION_4, CLASS_CHUNKED) => {
                let sa = ctx.sizeof_addr as usize;
                let mut pos = 2;

                // flags(1) + ndims(1) + enc_bytes_per_dim(1)
                if buf.len() < pos + 3 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 3,
                        available: buf.len(),
                    });
                }
                let flags = buf[pos];
                pos += 1;
                let ndims = buf[pos] as usize;
                pos += 1;
                let enc_bytes = buf[pos] as usize;
                pos += 1;

                // dim sizes
                let dim_data_len = ndims * enc_bytes;
                if buf.len() < pos + dim_data_len {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + dim_data_len,
                        available: buf.len(),
                    });
                }
                let mut chunk_dims = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    chunk_dims.push(read_size(&buf[pos..], enc_bytes));
                    pos += enc_bytes;
                }

                // index type
                if buf.len() < pos + 1 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 1,
                        available: buf.len(),
                    });
                }
                let idx_type_raw = buf[pos];
                pos += 1;
                let index_type = ChunkIndexType::from_u8(idx_type_raw).ok_or_else(|| {
                    FormatError::UnsupportedFeature(format!("chunk index type {}", idx_type_raw))
                })?;

                // Index-type-specific parameters
                let mut earray_params = None;
                let mut farray_params = None;

                match index_type {
                    ChunkIndexType::ExtensibleArray => {
                        if buf.len() < pos + 5 {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + 5,
                                available: buf.len(),
                            });
                        }
                        earray_params = Some(EarrayParams {
                            max_nelmts_bits: buf[pos],
                            idx_blk_elmts: buf[pos + 1],
                            sup_blk_min_data_ptrs: buf[pos + 2],
                            data_blk_min_elmts: buf[pos + 3],
                            max_dblk_page_nelmts_bits: buf[pos + 4],
                        });
                        pos += 5;
                    }
                    ChunkIndexType::FixedArray => {
                        if buf.len() < pos + 1 {
                            return Err(FormatError::BufferTooShort {
                                needed: pos + 1,
                                available: buf.len(),
                            });
                        }
                        farray_params = Some(FixedArrayParams {
                            max_dblk_page_nelmts_bits: buf[pos],
                        });
                        pos += 1;
                    }
                    // BTreeV2, SingleChunk, Implicit: no extra parameters
                    _ => {}
                }

                // index address
                if buf.len() < pos + sa {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + sa,
                        available: buf.len(),
                    });
                }
                let index_address = read_addr(&buf[pos..], sa);
                pos += sa;

                Ok((
                    Self::ChunkedV4 {
                        flags,
                        chunk_dims,
                        index_type,
                        earray_params,
                        farray_params,
                        index_address,
                    },
                    pos,
                ))
            }
            (VERSION_3, other) => Err(FormatError::UnsupportedFeature(format!(
                "data layout class {}",
                other
            ))),
            (v, _) => Err(FormatError::InvalidVersion(v)),
        }
    }
}

// ========================================================================= helpers

/// Read a little-endian address of `n` bytes, mapping all-ones to `UNDEF_ADDR`.
fn read_addr(buf: &[u8], n: usize) -> u64 {
    if buf[..n].iter().all(|&b| b == 0xFF) {
        UNDEF_ADDR
    } else {
        let mut tmp = [0u8; 8];
        tmp[..n].copy_from_slice(&buf[..n]);
        u64::from_le_bytes(tmp)
    }
}

/// Read a little-endian size of `n` bytes.
fn read_size(buf: &[u8], n: usize) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

/// Compute the minimum number of bytes (1-8) needed to encode `v`.
fn enc_bytes_for_value(v: u64) -> u8 {
    if v == 0 {
        return 1;
    }
    let bits_needed = 64 - v.leading_zeros(); // 1..=64
    bits_needed.div_ceil(8) as u8
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx8() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    fn ctx4() -> FormatContext {
        FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        }
    }

    #[test]
    fn roundtrip_contiguous() {
        let msg = DataLayoutMessage::contiguous(0x1000, 4096);
        let encoded = msg.encode(&ctx8());
        // 2 + 8 + 8 = 18
        assert_eq!(encoded.len(), 18);
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 18);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_contiguous_ctx4() {
        let msg = DataLayoutMessage::contiguous(0x800, 256);
        let encoded = msg.encode(&ctx4());
        // 2 + 4 + 4 = 10
        assert_eq!(encoded.len(), 10);
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_contiguous_unallocated() {
        let msg = DataLayoutMessage::contiguous_unallocated(1024);
        let encoded = msg.encode(&ctx8());
        let (decoded, _) = DataLayoutMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(decoded, msg);
        match decoded {
            DataLayoutMessage::Contiguous { address, size } => {
                assert_eq!(address, UNDEF_ADDR);
                assert_eq!(size, 1024);
            }
            _ => panic!("expected Contiguous"),
        }
    }

    #[test]
    fn roundtrip_contiguous_undef_ctx4() {
        let msg = DataLayoutMessage::contiguous_unallocated(512);
        let encoded = msg.encode(&ctx4());
        let (decoded, _) = DataLayoutMessage::decode(&encoded, &ctx4()).unwrap();
        match decoded {
            DataLayoutMessage::Contiguous { address, .. } => {
                assert_eq!(address, UNDEF_ADDR);
            }
            _ => panic!("expected Contiguous"),
        }
    }

    #[test]
    fn roundtrip_compact() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let msg = DataLayoutMessage::compact(data.clone());
        let encoded = msg.encode(&ctx8());
        // 2 + 2 + 8 = 12
        assert_eq!(encoded.len(), 12);
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_compact_empty() {
        let msg = DataLayoutMessage::compact(vec![]);
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded.len(), 4); // 2 + 2 + 0
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, 4);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_bad_version() {
        let buf = [2u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = DataLayoutMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::InvalidVersion(2) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_unsupported_class() {
        let buf = [3u8, 3]; // class 3 = unknown
        let err = DataLayoutMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::UnsupportedFeature(_) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        let buf = [3u8];
        let err = DataLayoutMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn decode_contiguous_truncated() {
        // version=3, class=1, but not enough bytes for address+size
        let buf = [3u8, 1, 0, 0];
        let err = DataLayoutMessage::decode(&buf, &ctx8()).unwrap_err();
        match err {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn version_and_class_bytes() {
        let encoded = DataLayoutMessage::contiguous(0, 0).encode(&ctx8());
        assert_eq!(encoded[0], 3);
        assert_eq!(encoded[1], 1);

        let encoded = DataLayoutMessage::compact(vec![]).encode(&ctx8());
        assert_eq!(encoded[0], 3);
        assert_eq!(encoded[1], 0);
    }

    #[test]
    fn roundtrip_chunked_v4_earray() {
        let params = EarrayParams::default_params();
        let msg = DataLayoutMessage::chunked_v4_earray(vec![1, 256, 256], params, 0x2000);
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded[0], 4); // version 4
        assert_eq!(encoded[1], 2); // class chunked
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_chunked_v4_earray_ctx4() {
        let params = EarrayParams::default_params();
        let msg = DataLayoutMessage::chunked_v4_earray(vec![1, 128], params, 0x1000);
        let encoded = msg.encode(&ctx4());
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx4()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_chunked_v4_single() {
        let msg = DataLayoutMessage::chunked_v4_single(vec![100, 200], 0x3000);
        let encoded = msg.encode(&ctx8());
        let (decoded, consumed) = DataLayoutMessage::decode(&encoded, &ctx8()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, msg);
    }

    #[test]
    fn chunked_v4_enc_bytes() {
        // chunk dims [1, 256, 256]: max=256, needs 2 bytes
        let params = EarrayParams::default_params();
        let msg = DataLayoutMessage::chunked_v4_earray(vec![1, 256, 256], params, 0x2000);
        let encoded = msg.encode(&ctx8());
        // version(1) + class(1) + flags(1) + ndims(1) + enc_bytes(1)
        // + 3*2 dim bytes + index_type(1) + 5 earray params + 8 addr = 25
        assert_eq!(encoded.len(), 25);
        assert_eq!(encoded[4], 2); // enc_bytes_per_dim = 2
    }

    #[test]
    fn chunked_v4_large_dims() {
        // Large dims requiring 4 bytes each
        let params = EarrayParams::default_params();
        let msg = DataLayoutMessage::chunked_v4_earray(vec![1, 65536], params, 0x4000);
        let encoded = msg.encode(&ctx8());
        assert_eq!(encoded[4], 3); // enc_bytes_per_dim = 3 (65536 = 0x10000, needs 3 bytes)
    }
}
