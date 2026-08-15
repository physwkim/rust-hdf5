//! Fill value message (type 0x05) — specifies the default fill value for
//! unwritten elements.
//!
//! Three on-disk versions exist for this message and a conforming reader
//! must accept all of them (`H5O__fill_new_decode`, H5Ofill.c):
//!
//! Versions 1 & 2:
//!   Byte 0: version (1 or 2)
//!   Byte 1: space allocation time (1=early, 2=late, 3=incremental)
//!   Byte 2: fill value write time (0=on_alloc, 1=never, 2=if_set)
//!   Byte 3: fill value defined (0=undefined, non-zero=defined)
//!   [if defined != 0]: u32 LE size + `size` bytes of fill data
//!
//! Version 3 (the version this crate writes):
//!   Byte 0: version = 3
//!   Byte 1: flags — bits 0-1 allocation time, bits 2-3 fill write time,
//!           0x10 = fill value undefined, 0x20 = explicit fill value present
//!   [if 0x20 set]: u32 LE size + `size` bytes of fill data
//!
//! libhdf5 2.0.0 still writes version 2 for this message, so decoding must
//! handle every version even though encoding always emits version 3.

use crate::format::{FormatError, FormatResult};

const VERSION: u8 = 3;

// Version-3 flags byte layout (H5Ofill.c).
const FLAG_MASK_ALLOC: u8 = 0x03;
const FLAG_MASK_FILL: u8 = 0x03;
const FLAG_SHIFT_FILL: u8 = 2;
const FLAG_UNDEFINED: u8 = 0x10;
const FLAG_HAVE_VALUE: u8 = 0x20;
const FLAGS_ALL: u8 =
    FLAG_MASK_ALLOC | (FLAG_MASK_FILL << FLAG_SHIFT_FILL) | FLAG_UNDEFINED | FLAG_HAVE_VALUE;

/// `H5D_ALLOC_TIME_EARLY` (`H5Dpublic.h`): space is allocated as soon as the
/// dataset is created. `H5P__set_layout`'s default switch (H5Pdcpl.c:1864-
/// 1877) gives this to a compact dataset — its storage *is* the object
/// header, so it exists as soon as the dataset does.
pub const ALLOC_TIME_EARLY: u8 = 1;

/// `H5D_ALLOC_TIME_LATE`: space is allocated when data is first written.
/// The default for a contiguous dataset (`H5P__set_layout`, H5Pdcpl.c:1870).
pub const ALLOC_TIME_LATE: u8 = 2;

/// `H5D_ALLOC_TIME_INCR`: space is allocated incrementally, as chunks (or
/// virtual source datasets) are written. The default for a chunked or
/// virtual dataset (`H5P__set_layout`, H5Pdcpl.c:1874-1877).
pub const ALLOC_TIME_INCR: u8 = 3;

/// `H5D_FILL_TIME_ALLOC` (`H5Dpublic.h`): fill at allocation regardless of
/// whether a fill value was ever set — an unset value fills with the
/// default (zeros), same as leaving newly allocated space untouched.
pub const FILL_TIME_ALLOC: u8 = 0;

/// `H5D_FILL_TIME_NEVER`: the fill value is never written into allocated
/// storage. `H5D__chunk_lock`'s cache-miss path (H5Dchunk.c:4894) gates on
/// it, and this crate's writer mirrors that gate at its own two eager-fill
/// sites — the immediate tiling `set_dataset_fill_value` does for a
/// compact/contiguous/implicit dataset, and the buffer a chunked partial
/// write builds for a chunk touched for the first time — unlike a shrink's
/// straddler refill (`H5D__chunk_prune_fill`), which fills unconditionally
/// because it is repairing data about to become reachable again, not
/// filling at allocation.
pub const FILL_TIME_NEVER: u8 = 1;

/// `H5D_FILL_TIME_IFSET`, the write time the default dataset creation
/// property list carries (`H5D_CRT_FILL_TIME_DEF`, H5Dpkg.h) and therefore the
/// one every dataset gets unless `H5Pset_fill_time` says otherwise.
///
/// It means "fill at allocation only when the user set a fill value", where
/// `H5D_FILL_TIME_ALLOC` fills at allocation either way. The two differ only
/// for a dataset with no fill value of its own, where `ALLOC` writes the
/// default fill — zeros — and `IFSET` writes nothing into space that reads as
/// zeros regardless. So this is what the writer's own allocation-time fills
/// already do, whichever of the two the message claimed.
pub const FILL_TIME_IFSET: u8 = 2;

/// Fill value message payload.
#[derive(Debug, Clone, PartialEq)]
pub struct FillValueMessage {
    /// Space allocation time: 1=early, 2=late, 3=incremental.
    pub alloc_time: u8,
    /// Fill value write time: 0=on alloc, 1=never, 2=if set.
    pub fill_write_time: u8,
    /// Fill value defined: 0=undefined, 1=default (zeros), 2=user-defined.
    pub fill_defined: u8,
    /// User-defined fill value data.  Present only when `fill_defined == 2`.
    pub fill_value: Option<Vec<u8>>,
}

impl Default for FillValueMessage {
    fn default() -> Self {
        Self {
            alloc_time: ALLOC_TIME_LATE,
            fill_write_time: FILL_TIME_IFSET,
            fill_defined: 1, // default value (zeros)
            fill_value: None,
        }
    }
}

impl FillValueMessage {
    /// A user-defined fill value.
    pub fn with_value(data: Vec<u8>) -> Self {
        Self {
            alloc_time: ALLOC_TIME_LATE,
            fill_write_time: FILL_TIME_IFSET,
            fill_defined: 2,
            fill_value: Some(data),
        }
    }

    /// An undefined fill value (no fill is performed).
    pub fn undefined() -> Self {
        Self {
            alloc_time: ALLOC_TIME_LATE,
            fill_write_time: FILL_TIME_NEVER,
            fill_defined: 0,
            fill_value: None,
        }
    }

    // ------------------------------------------------------------------ encode

    /// Encode as a version-3 fill-value message (`H5O__fill_new_encode`).
    pub fn encode(&self) -> Vec<u8> {
        self.encode_for(crate::format::ObjectFormat::Modern)
    }

    /// Encode at the version a file of this `format` calls for
    /// (`H5O__fill_new_encode`, H5Ofill.c:409).
    ///
    /// Version 2 spells out allocation time, write time and definedness as
    /// three bytes instead of packing them into flags, and — the part a
    /// version-3 reader must not assume — always writes the four-byte size
    /// field when the value is defined, even when the size is zero. That
    /// `defined = 1, size = 0` shape is what libhdf5 writes for the default
    /// fill of a contiguous dataset in a classic file.
    pub fn encode_for(&self, format: crate::format::ObjectFormat) -> Vec<u8> {
        if format.fill_value_version() < VERSION {
            let mut buf = Vec::with_capacity(12);
            buf.push(format.fill_value_version());
            buf.push(self.alloc_time);
            buf.push(self.fill_write_time);
            if self.fill_defined == 0 {
                buf.push(0);
                return buf;
            }
            buf.push(1);
            let value = self.fill_value.as_deref().unwrap_or(&[]);
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buf.extend_from_slice(value);
            return buf;
        }
        let mut buf = Vec::with_capacity(10);
        buf.push(VERSION);

        let flags = (self.alloc_time & FLAG_MASK_ALLOC)
            | ((self.fill_write_time & FLAG_MASK_FILL) << FLAG_SHIFT_FILL);

        if self.fill_defined == 0 {
            // Explicitly undefined: no value follows.
            buf.push(flags | FLAG_UNDEFINED);
        } else if let Some(data) = self.fill_value.as_ref().filter(|d| !d.is_empty()) {
            // Explicit value present.
            buf.push(flags | FLAG_HAVE_VALUE);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
        } else {
            // Defined, but no explicit value (default zero fill).
            buf.push(flags);
        }

        buf
    }

    // ------------------------------------------------------------------ decode

    /// Decode a fill-value message of version 1, 2, or 3.
    pub fn decode(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.is_empty() {
            return Err(FormatError::BufferTooShort {
                needed: 1,
                available: 0,
            });
        }
        match buf[0] {
            1 | 2 => Self::decode_v1v2(buf),
            3 => Self::decode_v3(buf),
            other => Err(FormatError::InvalidVersion(other)),
        }
    }

    /// Decode the version-1/2 layout (separate alloc/fill-time/defined bytes).
    fn decode_v1v2(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 4 {
            return Err(FormatError::BufferTooShort {
                needed: 4,
                available: buf.len(),
            });
        }
        let alloc_time = buf[1];
        let fill_write_time = buf[2];
        let defined_byte = buf[3];

        let mut pos = 4;
        let mut fill_value = None;
        if defined_byte != 0 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let size =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            pos += 4;
            if size > 0 {
                if buf.len() < pos + size {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + size,
                        available: buf.len(),
                    });
                }
                fill_value = Some(buf[pos..pos + size].to_vec());
                pos += size;
            }
        }

        // Normalize `fill_defined` onto this crate's tri-state: an explicit
        // non-empty value is "user-defined", any other defined message is
        // "default zero fill", an undefined message is "undefined".
        let fill_defined = if fill_value.is_some() {
            2
        } else if defined_byte != 0 {
            1
        } else {
            0
        };

        Ok((
            Self {
                alloc_time,
                fill_write_time,
                fill_defined,
                fill_value,
            },
            pos,
        ))
    }

    /// Decode the version-3 layout (packed flags byte).
    fn decode_v3(buf: &[u8]) -> FormatResult<(Self, usize)> {
        if buf.len() < 2 {
            return Err(FormatError::BufferTooShort {
                needed: 2,
                available: buf.len(),
            });
        }
        let flags = buf[1];
        if flags & !FLAGS_ALL != 0 {
            return Err(FormatError::InvalidData(format!(
                "unknown flags 0x{flags:02x} in version-3 fill-value message"
            )));
        }
        let alloc_time = flags & FLAG_MASK_ALLOC;
        let fill_write_time = (flags >> FLAG_SHIFT_FILL) & FLAG_MASK_FILL;

        let mut pos = 2;
        let (fill_defined, fill_value) = if flags & FLAG_UNDEFINED != 0 {
            if flags & FLAG_HAVE_VALUE != 0 {
                return Err(FormatError::InvalidData(
                    "fill-value message sets both the undefined and have-value flags".into(),
                ));
            }
            (0, None)
        } else if flags & FLAG_HAVE_VALUE != 0 {
            if buf.len() < pos + 4 {
                return Err(FormatError::BufferTooShort {
                    needed: pos + 4,
                    available: buf.len(),
                });
            }
            let size =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            pos += 4;
            if buf.len() < pos + size {
                return Err(FormatError::BufferTooShort {
                    needed: pos + size,
                    available: buf.len(),
                });
            }
            let data = buf[pos..pos + size].to_vec();
            pos += size;
            (2, Some(data))
        } else {
            (1, None)
        };

        Ok((
            Self {
                alloc_time,
                fill_write_time,
                fill_defined,
                fill_value,
            },
            pos,
        ))
    }
}

// ================================================================ tiling helper

/// Build a `total`-byte buffer whose contents are `fill_value` tiled one
/// element wide, or all zeros when `fill_value` is `None` or empty.
///
/// This is the single source of truth for materializing fill values:
/// chunked reads use it to initialize output buffers, and the writer uses
/// it to pad partial chunks so that unwritten elements read back as the
/// fill value rather than zero.
pub(crate) fn tiled_fill(total: usize, fill_value: Option<&[u8]>) -> Vec<u8> {
    match fill_value {
        Some(fv) if !fv.is_empty() && total > 0 => {
            let mut buf = vec![0u8; total];
            for slot in buf.chunks_mut(fv.len()) {
                let n = slot.len().min(fv.len());
                slot[..n].copy_from_slice(&fv[..n]);
            }
            buf
        }
        _ => vec![0u8; total],
    }
}

/// Fallible variant of [`tiled_fill`] for reader paths.
///
/// `total` on a read path is derived from untrusted file fields (dataspace
/// dimensions, element size). A crafted file can declare an absurd dataset
/// size; allocating it with `vec![0u8; total]` aborts the process on
/// allocation failure. This variant uses `try_reserve_exact`, returning a
/// `TryReserveError` the caller can surface as a clean error instead.
pub(crate) fn try_tiled_fill(
    total: usize,
    fill_value: Option<&[u8]>,
) -> Result<Vec<u8>, std::collections::TryReserveError> {
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(total)?;
    buf.resize(total, 0);
    if let Some(fv) = fill_value {
        if !fv.is_empty() && total > 0 {
            for slot in buf.chunks_mut(fv.len()) {
                let n = slot.len().min(fv.len());
                slot[..n].copy_from_slice(&fv[..n]);
            }
        }
    }
    Ok(buf)
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_default() {
        let msg = FillValueMessage::default();
        let encoded = msg.encode();
        // Version 3: version byte + flags byte, no value.
        assert_eq!(encoded.len(), 2);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrip_user_defined() {
        let msg = FillValueMessage::with_value(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let encoded = msg.encode();
        // version + flags + u32 size + 4 data = 10
        assert_eq!(encoded.len(), 10);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(decoded, msg);
        assert_eq!(
            decoded.fill_value.as_ref().unwrap(),
            &vec![0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    fn roundtrip_undefined() {
        let msg = FillValueMessage::undefined();
        let encoded = msg.encode();
        assert_eq!(encoded.len(), 2);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(decoded, msg);
    }

    #[test]
    fn version_3_flags_byte_layout() {
        // alloc_time=3 (bits 0-1), fill_write_time=2 (bits 2-3),
        // explicit value present -> 0x20.
        let msg = FillValueMessage {
            alloc_time: 3,
            fill_write_time: 2,
            fill_defined: 2,
            fill_value: Some(vec![0x01, 0x02]),
        };
        let encoded = msg.encode();
        assert_eq!(encoded[0], 3);
        assert_eq!(encoded[1], 0x03 | (0x02 << 2) | FLAG_HAVE_VALUE);
        assert_eq!(&encoded[2..6], &2u32.to_le_bytes());
        assert_eq!(&encoded[6..8], &[0x01, 0x02]);
        assert_eq!(encoded.len(), 8);
    }

    #[test]
    fn version_3_undefined_flag() {
        let encoded = FillValueMessage::undefined().encode();
        assert_eq!(encoded[1] & FLAG_UNDEFINED, FLAG_UNDEFINED);
        assert_eq!(encoded[1] & FLAG_HAVE_VALUE, 0);
    }

    #[test]
    fn empty_user_data_normalizes_to_default() {
        // An empty explicit value is indistinguishable from default fill in
        // the on-disk format, so it round-trips as `fill_defined == 1`.
        let msg = FillValueMessage {
            alloc_time: 1,
            fill_write_time: 2,
            fill_defined: 2,
            fill_value: Some(vec![]),
        };
        let encoded = msg.encode();
        assert_eq!(encoded.len(), 2);
        let (decoded, consumed) = FillValueMessage::decode(&encoded).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(decoded.alloc_time, 1);
        assert_eq!(decoded.fill_write_time, 2);
        assert_eq!(decoded.fill_defined, 1);
        assert_eq!(decoded.fill_value, None);
    }

    #[test]
    fn decode_version_1_message() {
        // Version-1 message with an explicit 4-byte value.
        let buf = [1u8, 2, 0, 1, 4, 0, 0, 0, 0xAA, 0xBB, 0xCC, 0xDD];
        let (decoded, consumed) = FillValueMessage::decode(&buf).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded.alloc_time, 2);
        assert_eq!(decoded.fill_defined, 2);
        assert_eq!(
            decoded.fill_value.as_ref().unwrap(),
            &vec![0xAA, 0xBB, 0xCC, 0xDD]
        );
    }

    #[test]
    fn decode_version_2_message_libhdf5_default() {
        // The body libhdf5 2.0.0 actually writes: version 2, alloc=2,
        // fill_time=2, defined=1, size=4.
        let buf = [2u8, 2, 2, 1, 4, 0, 0, 0, 0x00, 0x00, 0x80, 0xBF];
        let (decoded, consumed) = FillValueMessage::decode(&buf).unwrap();
        assert_eq!(consumed, 12);
        assert_eq!(decoded.alloc_time, 2);
        assert_eq!(decoded.fill_write_time, 2);
        assert_eq!(decoded.fill_defined, 2);
        assert_eq!(
            decoded.fill_value.as_ref().unwrap(),
            &vec![0x00, 0x00, 0x80, 0xBF]
        );
    }

    #[test]
    fn decode_version_2_defined_without_value() {
        // Defined (byte != 0) but size 0: no explicit value.
        let buf = [2u8, 2, 0, 1, 0, 0, 0, 0];
        let (decoded, consumed) = FillValueMessage::decode(&buf).unwrap();
        assert_eq!(consumed, 8);
        assert_eq!(decoded.fill_defined, 1);
        assert_eq!(decoded.fill_value, None);
    }

    #[test]
    fn decode_bad_version() {
        for bad in [0u8, 4, 9] {
            let buf = [bad, 0, 0, 0];
            match FillValueMessage::decode(&buf).unwrap_err() {
                FormatError::InvalidVersion(v) if v == bad => {}
                other => panic!("unexpected error for version {bad}: {other:?}"),
            }
        }
    }

    #[test]
    fn decode_buffer_too_short() {
        // Only the version byte — a v3 message needs the flags byte too.
        let buf = [3u8];
        match FillValueMessage::decode(&buf).unwrap_err() {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decode_v3_unknown_flag_rejected() {
        // Bit 0x40 is not a defined flag.
        let buf = [3u8, 0x40];
        match FillValueMessage::decode(&buf).unwrap_err() {
            FormatError::InvalidData(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decode_v3_truncated_size() {
        // HAVE_VALUE flag set but the u32 size field is missing.
        let buf = [3u8, FLAG_HAVE_VALUE, 0xFF];
        match FillValueMessage::decode(&buf).unwrap_err() {
            FormatError::BufferTooShort { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn decode_v3_truncated_data() {
        // HAVE_VALUE, size=4, but only 2 bytes of data.
        let buf = [3u8, FLAG_HAVE_VALUE, 4, 0, 0, 0, 0xAA, 0xBB];
        match FillValueMessage::decode(&buf).unwrap_err() {
            FormatError::BufferTooShort {
                needed: 10,
                available: 8,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn version_byte() {
        let encoded = FillValueMessage::default().encode();
        assert_eq!(encoded[0], 3);
    }

    #[test]
    fn tiled_fill_repeats_pattern() {
        assert_eq!(tiled_fill(0, Some(&[1, 2])), Vec::<u8>::new());
        assert_eq!(tiled_fill(6, None), vec![0u8; 6]);
        assert_eq!(tiled_fill(6, Some(&[])), vec![0u8; 6]);
        assert_eq!(
            tiled_fill(6, Some(&[0xAB, 0xCD])),
            vec![0xAB, 0xCD, 0xAB, 0xCD, 0xAB, 0xCD]
        );
        // Partial tail when total is not a multiple of the pattern width.
        assert_eq!(tiled_fill(5, Some(&[1, 2])), vec![1, 2, 1, 2, 1]);
    }

    /// The 8 bytes libhdf5 1.14.6 wrote for the default fill of a contiguous
    /// dataset in a default (superblock-0) file: version 2, and the size field
    /// present-but-zero that a version-3 reader never sees.
    #[test]
    fn a_legacy_fill_value_matches_the_bytes_libhdf5_wrote() {
        let fv = FillValueMessage {
            alloc_time: 2,
            fill_write_time: 2,
            fill_defined: 1,
            fill_value: None,
        };
        let buf = fv.encode_for(crate::format::ObjectFormat::Legacy);
        assert_eq!(buf, vec![0x02, 0x02, 0x02, 0x01, 0, 0, 0, 0]);
        let (back, consumed) = FillValueMessage::decode(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(back, fv);
    }

    #[test]
    fn a_legacy_user_fill_value_round_trips() {
        let fv = FillValueMessage::with_value(vec![7, 0, 0, 0]);
        let buf = fv.encode_for(crate::format::ObjectFormat::Legacy);
        assert_eq!(&buf[..4], &[0x02, 0x02, 0x02, 0x01]);
        let (back, _) = FillValueMessage::decode(&buf).unwrap();
        assert_eq!(back, fv);
    }

    #[test]
    fn a_legacy_undefined_fill_value_writes_no_size() {
        let buf = FillValueMessage::undefined().encode_for(crate::format::ObjectFormat::Legacy);
        assert_eq!(buf, vec![0x02, 0x02, 0x01, 0x00]);
        let (back, _) = FillValueMessage::decode(&buf).unwrap();
        assert_eq!(back, FillValueMessage::undefined());
    }
}
