//! Object modification time message (`H5O_MTIME_NEW`, type 0x12).
//!
//! Where a version-1 object header records the one time it keeps. A version-2
//! header has four time fields in its prefix instead and never carries this
//! message — `H5O_touch_oh` branches on `oh->version == H5O_VERSION_1` and
//! writes the prefix fields otherwise (H5Oint.c:1290-1345).
//!
//! The message is only ever *created* by a forced touch, which happens at one
//! place in the library: `H5D__update_oh_info` calls `H5O_touch_oh(file, oh,
//! true)` for a dataset whose file is below the v1.8 bound (H5Dint.c:1022-1026).
//! Every other caller passes `force = false` (H5Oattribute.c:376, :909, :1158,
//! :1517, :1603; H5Omessage.c:1193, :1799), which updates a message that is
//! already there and creates nothing — which is why a version-1 group or
//! committed datatype has no modification time even though its object header
//! is tracking times.
//!
//! There is an older form of the same idea, `H5O_MTIME` (type 0x0E), which
//! stored the time as a formatted date string; nothing since 1.6 writes it and
//! this crate does not either.

use crate::format::{FormatError, FormatResult};

/// `H5O_MTIME_VERSION` (H5Omtime.c) — the only version this message has.
const MTIME_VERSION: u8 = 1;

/// Encoded size of the message: version, three reserved bytes, and a 32-bit
/// time (`H5O__mtime_new_size`, H5Omtime.c returns 8).
pub const MTIME_MESSAGE_SIZE: usize = 8;

/// The seconds-since-the-epoch a modification time message carries.
///
/// A 32-bit count, as `H5O__mtime_new_encode` writes it (H5Omtime.c): the
/// value saturates in 2106 whatever the platform's `time_t` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModificationTime(pub u32);

impl ModificationTime {
    /// The message body: version 1, three reserved zero bytes, then the time
    /// (`H5O__mtime_new_encode`, H5Omtime.c).
    pub fn encode(self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(MTIME_MESSAGE_SIZE);
        buf.push(MTIME_VERSION);
        buf.extend_from_slice(&[0, 0, 0]);
        buf.extend_from_slice(&self.0.to_le_bytes());
        buf
    }

    /// Read a message body (`H5O__mtime_new_decode`, H5Omtime.c).
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        if buf.len() < MTIME_MESSAGE_SIZE {
            return Err(FormatError::BufferTooShort {
                needed: MTIME_MESSAGE_SIZE,
                available: buf.len(),
            });
        }
        if buf[0] != MTIME_VERSION {
            return Err(FormatError::InvalidVersion(buf[0]));
        }
        Ok(Self(u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_the_eight_bytes_upstream_writes() {
        let encoded = ModificationTime(0x5F00_1234).encode();
        assert_eq!(encoded.len(), MTIME_MESSAGE_SIZE);
        assert_eq!(&encoded[..4], &[1, 0, 0, 0], "version then three reserved");
        assert_eq!(ModificationTime::decode(&encoded).unwrap().0, 0x5F00_1234);
    }

    /// `H5O__mtime_new_decode` fails the message rather than guessing when the
    /// version is not the one version it has.
    #[test]
    fn a_foreign_version_is_refused() {
        let mut encoded = ModificationTime(1).encode();
        encoded[0] = 2;
        assert!(ModificationTime::decode(&encoded).is_err());
    }

    #[test]
    fn a_short_body_is_refused() {
        assert!(ModificationTime::decode(&[1, 0, 0, 0, 0]).is_err());
    }
}
