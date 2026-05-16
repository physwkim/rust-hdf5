//! Panic-free little-endian integer decode helpers.
//!
//! HDF5 on-disk structures store offsets, lengths and counts as
//! little-endian unsigned integers of a file-determined width (typically
//! 4 or 8 bytes). These helpers decode such fields into a `u64` without
//! ever panicking on a short buffer or an oversized width, so that a
//! truncated or malformed file produces a clamped value rather than an
//! index-out-of-bounds panic.

use super::UNDEF_ADDR;

/// Read a little-endian unsigned integer of `n` bytes into a `u64`.
///
/// `n` is clamped to `min(8, buf.len())` and the result is zero-padded,
/// so this can never panic. On well-formed input (`n <= 8` and
/// `buf.len() >= n`) the result is identical to the naive
/// `u64::from_le_bytes` of the first `n` bytes.
pub(crate) fn read_le_uint(buf: &[u8], n: usize) -> u64 {
    let n = n.min(8).min(buf.len());
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(tmp)
}

/// Read a little-endian address of `n` bytes, mapping an all-`0xFF`
/// field to [`UNDEF_ADDR`].
///
/// The all-ones test runs over the clamped width (`min(n, buf.len())`),
/// matching the on-disk "undefined address" sentinel. On well-formed
/// input this is byte-identical to the previous per-module `read_addr`
/// helpers.
pub(crate) fn read_le_addr(buf: &[u8], n: usize) -> u64 {
    let n = n.min(8).min(buf.len());
    if n != 0 && buf[..n].iter().all(|&b| b == 0xFF) {
        UNDEF_ADDR
    } else {
        read_le_uint(buf, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint_basic_widths() {
        assert_eq!(read_le_uint(&[0x01, 0x02, 0x03, 0x04], 4), 0x0403_0201);
        assert_eq!(
            read_le_uint(&[1, 0, 0, 0, 0, 0, 0, 0x80], 8),
            0x8000_0000_0000_0001
        );
        assert_eq!(read_le_uint(&[0xAB], 1), 0xAB);
    }

    #[test]
    fn uint_clamps_oversized_n_without_panic() {
        assert_eq!(read_le_uint(&[0xFF; 8], 16), u64::MAX);
        assert_eq!(read_le_uint(&[0x01, 0x02], 8), 0x0201);
        assert_eq!(read_le_uint(&[], 8), 0);
    }

    #[test]
    fn addr_all_ones_is_undef() {
        assert_eq!(read_le_addr(&[0xFF; 8], 8), UNDEF_ADDR);
        assert_eq!(read_le_addr(&[0xFF; 4], 4), UNDEF_ADDR);
        assert_eq!(read_le_addr(&[0xFF, 0xFF, 0x00, 0xFF], 4), 0xFF00_FFFF);
    }

    #[test]
    fn addr_empty_buffer_is_not_undef() {
        assert_eq!(read_le_addr(&[], 8), 0);
    }
}
