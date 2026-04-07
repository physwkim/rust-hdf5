//! Bit-level stream writer for FSE and Huffman encoding.
//!
//! Zstd uses a backward bitstream: bits are written from the end of the buffer
//! toward the beginning. The encoder fills a 64-bit accumulator and flushes
//! full bytes to the output when the accumulator overflows.

/// A forward bitstream writer (bits are appended LSB-first).
/// Used for Huffman literal streams.
pub struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32,     // bits in current partial byte (0..8)
    current: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self { buf: Vec::with_capacity(256), bit_pos: 0, current: 0 }
    }

    /// Write `nbits` (1..=57) from the low bits of `value`.
    pub fn write_bits(&mut self, value: u64, nbits: u32) {
        let mut val = value;
        let mut bits = nbits;

        // Fill current partial byte
        while bits > 0 {
            let space = 8 - self.bit_pos;
            let take = std::cmp::min(space, bits);
            let mask = (1u64 << take) - 1;
            self.current |= ((val & mask) as u8) << self.bit_pos;
            val >>= take;
            bits -= take;
            self.bit_pos += take;
            if self.bit_pos == 8 {
                self.buf.push(self.current);
                self.current = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Flush remaining bits (pad with zeros).
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buf.push(self.current);
        }
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len() * 8 + self.bit_pos as usize
    }
}

/// Backward bitstream writer for FSE sequence encoding.
///
/// Zstd encodes sequences in reverse order using a backward bitstream.
/// Bits accumulate in a register; when flushed, bytes come out in reverse.
pub struct BackwardBitWriter {
    bits: u64,
    bit_pos: u32,
    buf: Vec<u8>,
}

impl Default for BackwardBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardBitWriter {
    pub fn new() -> Self {
        Self { bits: 0, bit_pos: 0, buf: Vec::with_capacity(256) }
    }

    /// Add `nbits` from the low bits of `value` to the accumulator.
    pub fn add_bits(&mut self, value: u64, nbits: u32) {
        debug_assert!(nbits <= 57);
        debug_assert!(self.bit_pos + nbits <= 64);
        self.bits |= value << self.bit_pos;
        self.bit_pos += nbits;
    }

    /// Flush complete bytes from the accumulator to the output buffer.
    pub fn flush_bits(&mut self) {
        let bytes_to_flush = (self.bit_pos / 8) as usize;
        for _ in 0..bytes_to_flush {
            self.buf.push(self.bits as u8);
            self.bits >>= 8;
            self.bit_pos -= 8;
        }
    }

    /// Finalize: write a sentinel 1-bit then flush remaining.
    pub fn finish(mut self) -> Vec<u8> {
        // Add sentinel bit (marks the end for the decoder)
        self.add_bits(1, 1);
        self.flush_bits();
        if self.bit_pos > 0 {
            self.buf.push(self.bits as u8);
        }
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_writer_basic() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b1100, 4);
        w.write_bits(0b1, 1);
        let bytes = w.finish();
        // 8 bits total = 1 byte: 101_1100_1 -> reversed bit order:
        // LSB first: bits 0-2 = 101, bits 3-6 = 1100, bit 7 = 1
        // byte = 0b1_1100_101 = 0xE5
        assert_eq!(bytes, vec![0xE5]);
    }

    #[test]
    fn backward_writer_basic() {
        let mut w = BackwardBitWriter::new();
        w.add_bits(0xFF, 8);
        w.flush_bits();
        w.add_bits(0xAB, 8);
        w.flush_bits();
        let result = w.finish();
        // After sentinel and reverse, the decoder reads from front
        assert!(!result.is_empty());
    }
}
