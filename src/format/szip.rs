// Copyright 2024 Mathis Rosenhauer, Moritz Hanke, Joerg Behrens, Luis Kornblueh
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Pure Rust port of libaec (Adaptive Entropy Coding) for SZIP/HDF5.
// Based on the CCSDS recommended standard 121.0-B-3.

// ---------------------------------------------------------------------------
// SZIP option masks (HDF5 filter interface)
// ---------------------------------------------------------------------------
const SZ_MSB_OPTION_MASK: u32 = 16;
const SZ_NN_OPTION_MASK: u32 = 32;
const SZ_RAW_OPTION_MASK: u32 = 128;

// ---------------------------------------------------------------------------
// AEC flags
// ---------------------------------------------------------------------------
const AEC_DATA_SIGNED: u32 = 1;
#[allow(dead_code)]
const AEC_DATA_3BYTE: u32 = 2;
const AEC_DATA_MSB: u32 = 4;
const AEC_DATA_PREPROCESS: u32 = 8;
const AEC_RESTRICTED: u32 = 16;
const AEC_NOT_ENFORCE: u32 = 64;

// Marker for Remainder Of Segment in zero block encoding
const ROS_ENC: i32 = -1;
const ROS_DEC: u32 = 5;

const SE_TABLE_SIZE: usize = 90;

// ---------------------------------------------------------------------------
// Convert SZIP option mask to AEC flags
// ---------------------------------------------------------------------------
fn convert_options(sz_opts: u32) -> u32 {
    let mut flags: u32 = 0;
    if sz_opts & SZ_MSB_OPTION_MASK != 0 {
        flags |= AEC_DATA_MSB;
    }
    if sz_opts & SZ_NN_OPTION_MASK != 0 {
        flags |= AEC_DATA_PREPROCESS;
    }
    flags
}

fn bits_to_bytes(bits: u32) -> u32 {
    if bits > 16 {
        4
    } else if bits > 8 {
        2
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// SZIP interleaving (for 32/64-bit samples)
// ---------------------------------------------------------------------------
fn interleave_buffer(src: &[u8], wordsize: usize) -> Vec<u8> {
    let n = src.len();
    let count = n / wordsize;
    let mut dest = vec![0u8; n];
    for i in 0..count {
        for j in 0..wordsize {
            dest[j * count + i] = src[i * wordsize + j];
        }
    }
    dest
}

fn deinterleave_buffer(src: &[u8], wordsize: usize) -> Vec<u8> {
    let n = src.len();
    let count = n / wordsize;
    let mut dest = vec![0u8; n];
    for i in 0..count {
        for j in 0..wordsize {
            dest[i * wordsize + j] = src[j * count + i];
        }
    }
    dest
}

// ---------------------------------------------------------------------------
// Scanline padding helpers
// ---------------------------------------------------------------------------
fn add_padding(
    src: &[u8],
    line_size: usize,
    padding_size: usize,
    pixel_size: usize,
    pp: bool,
) -> Vec<u8> {
    let padded_line = line_size + padding_size;
    let num_lines = src.len().div_ceil(line_size);
    let mut dest = vec![0u8; num_lines * padded_line];
    let mut si = 0;
    let mut di = 0;
    while si < src.len() {
        let ls = std::cmp::min(src.len() - si, line_size);
        dest[di..di + ls].copy_from_slice(&src[si..si + ls]);
        di += ls;
        si += ls;
        let pad_pixels = padded_line - ls;
        let pixel: &[u8] = if pp && si >= pixel_size {
            &src[si - pixel_size..si]
        } else {
            &[0u8; 4][..pixel_size]
        };
        for k in (0..pad_pixels).step_by(pixel_size) {
            let end = std::cmp::min(k + pixel_size, pad_pixels);
            dest[di + k..di + end].copy_from_slice(&pixel[..end - k]);
        }
        di += pad_pixels;
    }
    dest.truncate(di);
    dest
}

fn remove_padding(buf: &mut Vec<u8>, line_size: usize, padding_size: usize) {
    let padded = line_size + padding_size;
    if padded == 0 || padding_size == 0 {
        return;
    }
    let mut dst = line_size;
    let mut src_off = padded;
    while src_off < buf.len() {
        let copy_len = std::cmp::min(line_size, buf.len() - src_off);
        buf.copy_within(src_off..src_off + copy_len, dst);
        dst += copy_len;
        src_off += padded;
    }
    buf.truncate(dst);
}

// ---------------------------------------------------------------------------
// Determine id_len from bits_per_sample and flags
// ---------------------------------------------------------------------------
fn compute_id_len(bits_per_sample: u32, flags: u32) -> Result<u32, String> {
    if bits_per_sample > 16 {
        Ok(5)
    } else if bits_per_sample > 8 {
        Ok(4)
    } else if flags & AEC_RESTRICTED != 0 {
        if bits_per_sample <= 2 {
            Ok(1)
        } else if bits_per_sample <= 4 {
            Ok(2)
        } else {
            Err("restricted mode only supports <= 4 bits".into())
        }
    } else {
        Ok(3)
    }
}

// ===========================================================================
//  ENCODER
// ===========================================================================

struct BitWriter {
    buf: Vec<u8>,
    bits: i32, // free bits in current byte (1..8)
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: vec![0u8],
            bits: 8,
        }
    }

    fn emit(&mut self, data: u32, mut nbits: i32) {
        if nbits == 0 {
            return;
        }
        if nbits <= self.bits {
            self.bits -= nbits;
            *self.buf.last_mut().unwrap() |= (data << self.bits) as u8;
        } else {
            nbits -= self.bits;
            *self.buf.last_mut().unwrap() |=
                ((data as u64 >> nbits) as u8) & ((1u16 << self.bits) - 1) as u8;
            while nbits > 8 {
                nbits -= 8;
                self.buf.push((data >> nbits) as u8);
            }
            self.bits = 8 - nbits;
            self.buf.push((data << self.bits) as u8);
        }
    }

    fn emitfs(&mut self, fs: u32) {
        // fs zero bits followed by one 1 bit
        let mut remaining = fs as i32;
        loop {
            if remaining < self.bits {
                self.bits -= remaining + 1;
                *self.buf.last_mut().unwrap() |= 1u8 << self.bits;
                break;
            } else {
                remaining -= self.bits;
                self.buf.push(0);
                self.bits = 8;
            }
        }
    }

    fn emit_block_fs(&mut self, block: &[u32], k: u32, ref_skip: usize) {
        // Emit fundamental sequences for each sample's high bits (sample >> k)
        for &s in &block[ref_skip..] {
            self.emitfs(s >> k);
        }
    }

    fn emit_block(&mut self, block: &[u32], k: u32, ref_skip: usize) {
        // Emit k LSBs of each sample
        if k == 0 {
            return;
        }
        let mask = (1u64 << k) - 1;
        for &s in &block[ref_skip..] {
            self.emit((s as u64 & mask) as u32, k as i32);
        }
    }

    fn flush_to_byte(&mut self) {
        // Pad remaining bits with zeros to byte boundary
        if self.bits < 8 {
            self.bits = 8;
            self.buf.push(0);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Remove trailing empty byte if fully aligned
        if self.bits == 8 && self.buf.len() > 1 {
            // The last byte is fully empty (no bits used)
            if *self.buf.last().unwrap() == 0 {
                self.buf.pop();
            }
        }
        self.buf
    }
}

fn preprocess_unsigned(raw: &[u32], xmax: u32) -> Vec<u32> {
    let n = raw.len();
    let mut d = vec![0u32; n];
    d[0] = 0; // placeholder for ref sample
    for i in 0..n - 1 {
        if raw[i + 1] >= raw[i] {
            let diff = raw[i + 1] - raw[i];
            if diff <= raw[i] {
                d[i + 1] = 2 * diff;
            } else {
                d[i + 1] = raw[i + 1];
            }
        } else {
            let diff = raw[i] - raw[i + 1];
            if diff <= xmax - raw[i] {
                d[i + 1] = 2 * diff - 1;
            } else {
                d[i + 1] = xmax - raw[i + 1];
            }
        }
    }
    d
}

fn preprocess_signed(raw: &[u32], bits_per_sample: u32, xmax: u32) -> Vec<u32> {
    let n = raw.len();
    let mut d = vec![0u32; n];
    let m = 1u32 << (bits_per_sample - 1);

    // Sign-extend all samples into i32 values, then compute deltas
    let mut sx = vec![0i32; n];
    for i in 0..n {
        sx[i] = ((raw[i] ^ m).wrapping_sub(m)) as i32;
    }

    d[0] = 0;
    for i in 0..n - 1 {
        let cur = sx[i];
        let nxt = sx[i + 1];
        if nxt < cur {
            let diff = (cur as u32).wrapping_sub(nxt as u32);
            if diff <= xmax.wrapping_add((cur as u32).wrapping_add(1)) {
                // half_d style: 2*D - 1
                d[i + 1] = 2u32.wrapping_mul(diff).wrapping_sub(1);
            } else {
                d[i + 1] = xmax.wrapping_sub(nxt as u32);
            }
        } else {
            let diff = (nxt as u32).wrapping_sub(cur as u32);
            let xmin_val = (!xmax) as i32; // ~xmax gives xmin for signed
            if diff <= (cur as u32).wrapping_sub(xmin_val as u32) {
                d[i + 1] = 2u32.wrapping_mul(diff);
            } else {
                d[i + 1] = (nxt as u32).wrapping_sub(xmin_val as u32);
            }
        }
    }
    d
}

fn assess_splitting(
    block: &[u32],
    block_size: usize,
    has_ref: bool,
    prev_k: u32,
    kmax: u32,
) -> (u32, u32) {
    let this_bs = if has_ref { block_size - 1 } else { block_size } as u64;
    let effective = if has_ref { &block[1..] } else { block };

    let mut len_min = u64::MAX;
    let mut k = prev_k;
    let mut k_min = k;
    let mut no_turn = k == 0;
    let mut dir = true; // true = increasing k

    loop {
        let fs_len: u64 = effective.iter().map(|&s| (s >> k) as u64).sum();
        let len = fs_len + this_bs * (k as u64 + 1);

        if len < len_min {
            if len_min < u64::MAX {
                no_turn = true;
            }
            len_min = len;
            k_min = k;

            if dir {
                if fs_len < this_bs || k >= kmax {
                    if no_turn {
                        break;
                    }
                    if prev_k == 0 {
                        break;
                    }
                    k = prev_k - 1;
                    dir = false;
                    no_turn = true;
                } else {
                    k += 1;
                }
            } else {
                if fs_len >= this_bs || k == 0 {
                    break;
                }
                k -= 1;
            }
        } else {
            if no_turn {
                break;
            }
            if prev_k == 0 {
                break;
            }
            k = prev_k - 1;
            dir = false;
            no_turn = true;
        }
    }
    (k_min, len_min as u32)
}

fn assess_se(block: &[u32], block_size: usize, uncomp_len: u32) -> u32 {
    let mut len = 1u64;
    let mut i = 0;
    while i < block_size {
        let d = block[i] as u64 + block[i + 1] as u64;
        len += d * (d + 1) / 2 + block[i + 1] as u64 + 1;
        if len > uncomp_len as u64 {
            return u32::MAX;
        }
        i += 2;
    }
    len as u32
}

struct Encoder {
    bits_per_sample: u32,
    block_size: u32,
    rsi: u32,
    flags: u32,
    id_len: u32,
    kmax: u32,
    xmax: u32,
    bytes_per_sample: u32,
}

impl Encoder {
    fn new(bits_per_sample: u32, block_size: u32, rsi: u32, flags: u32) -> Result<Self, String> {
        let id_len = compute_id_len(bits_per_sample, flags)?;
        let kmax = (1u32 << id_len) - 3;
        let xmax = if flags & AEC_DATA_SIGNED != 0 {
            ((1u64 << (bits_per_sample - 1)) - 1) as u32
        } else {
            ((1u64 << bits_per_sample) - 1) as u32
        };
        let bytes_per_sample = bits_to_bytes(bits_per_sample);
        Ok(Self {
            bits_per_sample,
            block_size,
            rsi,
            flags,
            id_len,
            kmax,
            xmax,
            bytes_per_sample,
        })
    }

    fn read_samples(&self, data: &[u8]) -> Vec<u32> {
        let bps = self.bytes_per_sample as usize;
        let msb = self.flags & AEC_DATA_MSB != 0;
        let n = data.len() / bps;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * bps;
            let s = match bps {
                1 => data[off] as u32,
                2 => {
                    if msb {
                        ((data[off] as u32) << 8) | data[off + 1] as u32
                    } else {
                        (data[off] as u32) | ((data[off + 1] as u32) << 8)
                    }
                }
                3 => {
                    if msb {
                        ((data[off] as u32) << 16)
                            | ((data[off + 1] as u32) << 8)
                            | data[off + 2] as u32
                    } else {
                        (data[off] as u32)
                            | ((data[off + 1] as u32) << 8)
                            | ((data[off + 2] as u32) << 16)
                    }
                }
                4 => {
                    if msb {
                        ((data[off] as u32) << 24)
                            | ((data[off + 1] as u32) << 16)
                            | ((data[off + 2] as u32) << 8)
                            | data[off + 3] as u32
                    } else {
                        (data[off] as u32)
                            | ((data[off + 1] as u32) << 8)
                            | ((data[off + 2] as u32) << 16)
                            | ((data[off + 3] as u32) << 24)
                    }
                }
                _ => unreachable!(),
            };
            // Mask to bits_per_sample
            let mask = if self.bits_per_sample == 32 {
                u32::MAX
            } else {
                (1u32 << self.bits_per_sample) - 1
            };
            samples.push(s & mask);
        }
        samples
    }

    fn encode(&self, input_data: &[u8]) -> Result<Vec<u8>, String> {
        let rsi_samples = (self.rsi * self.block_size) as usize;
        let samples = self.read_samples(input_data);
        let total_samples = samples.len();

        let mut writer = BitWriter::new();
        let mut offset = 0;
        let mut prev_k = 0u32;

        while offset < total_samples {
            // Read one RSI worth of samples (pad with last if short)
            let avail = std::cmp::min(rsi_samples, total_samples - offset);
            let mut rsi_buf = Vec::with_capacity(rsi_samples);
            rsi_buf.extend_from_slice(&samples[offset..offset + avail]);
            if avail < rsi_samples {
                let last = *rsi_buf.last().unwrap_or(&0);
                rsi_buf.resize(rsi_samples, last);
            }

            // Number of actual blocks to encode
            let blocks_to_encode = if avail < rsi_samples {
                let b = avail.div_ceil(self.block_size as usize);
                if b == 0 {
                    1
                } else {
                    b
                }
            } else {
                self.rsi as usize
            };

            // Preprocess
            let pp = if self.flags & AEC_DATA_PREPROCESS != 0 {
                if self.flags & AEC_DATA_SIGNED != 0 {
                    preprocess_signed(&rsi_buf, self.bits_per_sample, self.xmax)
                } else {
                    preprocess_unsigned(&rsi_buf, self.xmax)
                }
            } else {
                rsi_buf.clone()
            };

            let ref_sample = rsi_buf[0];
            let has_preprocess = self.flags & AEC_DATA_PREPROCESS != 0;

            // Encode each block
            let mut zero_blocks: i32 = 0;
            let mut zero_ref = false;
            let mut zero_ref_sample = 0u32;

            let bs = self.block_size as usize;

            for b in 0..blocks_to_encode {
                let block_start = b * bs;
                let block = &pp[block_start..block_start + bs];
                let is_first = b == 0;
                let has_ref = has_preprocess && is_first;

                let uncomp_len = if has_ref {
                    (bs as u32 - 1) * self.bits_per_sample
                } else {
                    bs as u32 * self.bits_per_sample
                };

                // Check if block is all zeros
                let all_zero = block.iter().all(|&x| x == 0);

                if all_zero {
                    zero_blocks += 1;
                    if zero_blocks == 1 {
                        zero_ref = has_ref;
                        zero_ref_sample = ref_sample;
                    }
                    // Check if we need to flush zero blocks:
                    // at end of RSI, or every 64 blocks
                    let is_last = b + 1 >= blocks_to_encode;
                    let at_boundary = (b + 1) % 64 == 0;
                    if is_last || at_boundary {
                        if zero_blocks > 4 {
                            zero_blocks = ROS_ENC;
                        }
                        // Encode zero block
                        writer.emit(0, self.id_len as i32 + 1);
                        if zero_ref {
                            writer.emit(zero_ref_sample, self.bits_per_sample as i32);
                        }
                        if zero_blocks == ROS_ENC {
                            writer.emitfs(4);
                        } else if zero_blocks >= 5 {
                            writer.emitfs(zero_blocks as u32);
                        } else {
                            writer.emitfs((zero_blocks - 1) as u32);
                        }
                        zero_blocks = 0;
                    }
                    continue;
                }

                // Non-zero block: first flush any pending zero blocks
                if zero_blocks > 0 {
                    writer.emit(0, self.id_len as i32 + 1);
                    if zero_ref {
                        writer.emit(zero_ref_sample, self.bits_per_sample as i32);
                    }
                    if zero_blocks == ROS_ENC {
                        writer.emitfs(4);
                    } else if zero_blocks >= 5 {
                        writer.emitfs(zero_blocks as u32);
                    } else {
                        writer.emitfs((zero_blocks - 1) as u32);
                    }
                    zero_blocks = 0;
                }

                // Assess coding options
                let (split_len, best_k) = if self.id_len > 1 {
                    let (k, len) = assess_splitting(block, bs, has_ref, prev_k, self.kmax);
                    prev_k = k;
                    (len, k)
                } else {
                    (u32::MAX, 0)
                };

                // SE always operates on the full (even-sized) block
                let se_len = if bs >= 2 {
                    assess_se(block, bs, uncomp_len)
                } else {
                    u32::MAX
                };

                if split_len < uncomp_len {
                    if split_len <= se_len {
                        // Splitting (Golomb-Rice)
                        writer.emit(best_k + 1, self.id_len as i32);
                        if has_ref {
                            writer.emit(ref_sample, self.bits_per_sample as i32);
                        }
                        writer.emit_block_fs(block, best_k, if has_ref { 1 } else { 0 });
                        if best_k > 0 {
                            writer.emit_block(block, best_k, if has_ref { 1 } else { 0 });
                        }
                    } else {
                        // Second extension
                        encode_se(
                            &mut writer,
                            block,
                            bs,
                            has_ref,
                            ref_sample,
                            self.id_len,
                            self.bits_per_sample,
                        );
                    }
                } else if uncomp_len <= se_len {
                    // Uncompressed
                    writer.emit((1u32 << self.id_len) - 1, self.id_len as i32);
                    if has_ref {
                        // For uncompressed, first sample is the raw reference
                        let mut ublock = block.to_vec();
                        ublock[0] = ref_sample;
                        writer.emit_block(&ublock, self.bits_per_sample, 0);
                    } else {
                        writer.emit_block(block, self.bits_per_sample, 0);
                    }
                } else {
                    // Second extension
                    encode_se(
                        &mut writer,
                        block,
                        bs,
                        has_ref,
                        ref_sample,
                        self.id_len,
                        self.bits_per_sample,
                    );
                }
            }

            offset += avail;
        }

        // Pad final byte
        writer.flush_to_byte();
        Ok(writer.finish())
    }
}

fn encode_se(
    writer: &mut BitWriter,
    block: &[u32],
    block_size: usize,
    has_ref: bool,
    ref_sample: u32,
    id_len: u32,
    bits_per_sample: u32,
) {
    // SE uses id_len + 1 bits: 0 followed by 1
    writer.emit(1, id_len as i32 + 1);
    if has_ref {
        writer.emit(ref_sample, bits_per_sample as i32);
    }
    // Always encode the full block (block[0] is 0 for ref blocks from preprocessing)
    let mut i = 0;
    while i < block_size {
        let a = block[i] as u64;
        let b = block[i + 1] as u64;
        let d = a + b;
        let fs = d * (d + 1) / 2 + b;
        writer.emitfs(fs as u32);
        i += 2;
    }
}

// ===========================================================================
//  DECODER
// ===========================================================================

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    bitp: i32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            bitp: 0,
        }
    }

    fn fill(&mut self) {
        while self.bitp <= 56 && self.pos < self.data.len() {
            self.acc = (self.acc << 8) | self.data[self.pos] as u64;
            self.pos += 1;
            self.bitp += 8;
        }
    }

    fn get_bits(&mut self, n: i32) -> u32 {
        while self.bitp < n {
            if self.pos < self.data.len() {
                self.acc = (self.acc << 8) | self.data[self.pos] as u64;
                self.pos += 1;
                self.bitp += 8;
            } else {
                // pad with zeros
                self.acc <<= 8;
                self.bitp += 8;
            }
        }
        self.bitp -= n;
        ((self.acc >> self.bitp) & ((1u64 << n) - 1)) as u32
    }

    fn get_fs(&mut self) -> u32 {
        let mut fs = 0u32;

        // Mask accumulator to valid bits
        if self.bitp > 0 {
            self.acc &= (1u64 << self.bitp) - 1;
        } else {
            self.acc = 0;
        }

        while self.acc == 0 {
            fs += self.bitp as u32;
            self.acc = 0;
            self.bitp = 0;
            // read more bytes
            let to_read = std::cmp::min(7, self.data.len() - self.pos);
            if to_read == 0 {
                return fs;
            }
            for _ in 0..to_read {
                self.acc = (self.acc << 8) | self.data[self.pos] as u64;
                self.pos += 1;
                self.bitp += 8;
            }
        }

        // Find highest set bit
        let highest = 63 - self.acc.leading_zeros() as i32;
        fs += (self.bitp - highest - 1) as u32;
        self.bitp = highest; // consume the 1 bit
        fs
    }
}

fn create_se_table() -> [i32; 2 * (SE_TABLE_SIZE + 1)] {
    let mut table = [0i32; 2 * (SE_TABLE_SIZE + 1)];
    let mut k = 0usize;
    for i in 0..13i32 {
        let ms = k as i32;
        for _j in 0..=i {
            if k <= SE_TABLE_SIZE {
                table[2 * k] = i;
                table[2 * k + 1] = ms;
            }
            k += 1;
        }
    }
    table
}

fn postprocess_unsigned(rsi_buf: &[u32], xmax: u32) -> Vec<u32> {
    let n = rsi_buf.len();
    if n == 0 {
        return vec![];
    }
    let mut out = vec![0u32; n];
    out[0] = rsi_buf[0]; // reference sample
    let med = xmax / 2 + 1;

    let mut data = out[0];
    for i in 1..n {
        let d = rsi_buf[i];
        let half_d = (d >> 1) + (d & 1);
        let mask = if data >= med { xmax } else { 0 };

        if half_d <= (mask ^ data) {
            data = data.wrapping_add((d >> 1) ^ (!((d & 1).wrapping_sub(1))));
        } else {
            data = mask ^ d;
        }
        out[i] = data;
    }
    out
}

fn postprocess_signed(rsi_buf: &[u32], bits_per_sample: u32, xmax: u32) -> Vec<u32> {
    let n = rsi_buf.len();
    if n == 0 {
        return vec![];
    }
    let mut out = vec![0u32; n];
    let m = 1u32 << (bits_per_sample - 1);
    // Sign-extend the reference sample
    let ref_val = (rsi_buf[0] ^ m).wrapping_sub(m);
    out[0] = ref_val;

    let mut data = ref_val;
    for i in 1..n {
        let d = rsi_buf[i];
        let half_d = (d >> 1) + (d & 1);

        if (data as i32) < 0 {
            if half_d <= xmax.wrapping_add(data).wrapping_add(1) {
                data = data.wrapping_add((d >> 1) ^ (!((d & 1).wrapping_sub(1))));
            } else {
                data = d.wrapping_sub(xmax).wrapping_sub(1);
            }
        } else {
            if half_d <= xmax.wrapping_sub(data) {
                data = data.wrapping_add((d >> 1) ^ (!((d & 1).wrapping_sub(1))));
            } else {
                data = xmax.wrapping_sub(d);
            }
        }
        out[i] = data;
    }
    out
}

struct Decoder {
    bits_per_sample: u32,
    block_size: u32,
    rsi: u32,
    flags: u32,
    id_len: u32,
    xmax: u32,
    bytes_per_sample: u32,
}

impl Decoder {
    fn new(bits_per_sample: u32, block_size: u32, rsi: u32, flags: u32) -> Result<Self, String> {
        let id_len = compute_id_len(bits_per_sample, flags)?;
        let xmax = if flags & AEC_DATA_SIGNED != 0 {
            ((1u64 << (bits_per_sample - 1)) - 1) as u32
        } else {
            ((1u64 << bits_per_sample) - 1) as u32
        };
        let bytes_per_sample = bits_to_bytes(bits_per_sample);
        Ok(Self {
            bits_per_sample,
            block_size,
            rsi,
            flags,
            id_len,
            xmax,
            bytes_per_sample,
        })
    }

    fn decode(&self, compressed: &[u8], output_samples: usize) -> Result<Vec<u32>, String> {
        let mut reader = BitReader::new(compressed);
        reader.fill();

        let se_table = create_se_table();
        let rsi_samples = (self.rsi * self.block_size) as usize;
        let pp = self.flags & AEC_DATA_PREPROCESS != 0;

        let mut all_output: Vec<u32> = Vec::with_capacity(output_samples);

        while all_output.len() < output_samples {
            // Decode one RSI
            let mut rsi_buf: Vec<u32> = Vec::with_capacity(rsi_samples);
            let mut first_block_in_rsi = true;

            while rsi_buf.len() < rsi_samples
                && all_output.len() + rsi_buf.len() < output_samples + rsi_samples
            {
                let has_ref = pp && first_block_in_rsi;
                let encoded_block_size = if has_ref {
                    self.block_size - 1
                } else {
                    self.block_size
                } as usize;

                // Read ID
                let id = reader.get_bits(self.id_len as i32);

                if id == 0 {
                    // Low entropy
                    let sub_id = reader.get_bits(1);
                    if sub_id == 1 {
                        // Second extension
                        if has_ref {
                            rsi_buf.push(reader.get_bits(self.bits_per_sample as i32));
                        }
                        // SE decoding: i starts at ref (0 or 1), runs to block_size
                        // Each iteration reads one FS and produces 1 or 2 samples
                        let ref_offset = if has_ref { 1usize } else { 0 };
                        let mut i = ref_offset;
                        while i < self.block_size as usize {
                            let m = reader.get_fs();
                            if m as usize > SE_TABLE_SIZE {
                                return Err("SE table overflow".into());
                            }
                            let d1 = m as i32 - se_table[2 * m as usize + 1];

                            if (i & 1) == 0 {
                                rsi_buf.push((se_table[2 * m as usize] - d1) as u32);
                                i += 1;
                            }
                            rsi_buf.push(d1 as u32);
                            i += 1;
                        }
                    } else {
                        // Zero block
                        if has_ref {
                            rsi_buf.push(reader.get_bits(self.bits_per_sample as i32));
                        }
                        let fs = reader.get_fs();
                        let mut zero_blocks = fs + 1;

                        if zero_blocks == ROS_DEC {
                            let b = rsi_buf.len() / self.block_size as usize;
                            let remaining = self.rsi as usize - b;
                            let boundary = 64 - (b % 64);
                            zero_blocks = std::cmp::min(remaining, boundary) as u32;
                        } else if zero_blocks > ROS_DEC {
                            zero_blocks -= 1;
                        }

                        let zero_samples = zero_blocks as usize * self.block_size as usize
                            - if has_ref { 1 } else { 0 };
                        rsi_buf.extend(std::iter::repeat_n(0, zero_samples));
                    }
                } else if id == (1u32 << self.id_len) - 1 {
                    // Uncompressed
                    for _ in 0..self.block_size {
                        rsi_buf.push(reader.get_bits(self.bits_per_sample as i32));
                    }
                } else {
                    // Split (Golomb-Rice) with k = id - 1
                    let k = id - 1;

                    if has_ref {
                        rsi_buf.push(reader.get_bits(self.bits_per_sample as i32));
                    }

                    // Read FS parts
                    let base = rsi_buf.len();
                    for _ in 0..encoded_block_size {
                        let fs = reader.get_fs();
                        rsi_buf.push(fs << k);
                    }

                    // Read binary parts and add
                    if k > 0 {
                        for j in 0..encoded_block_size {
                            let bits = reader.get_bits(k as i32);
                            rsi_buf[base + j] += bits;
                        }
                    }
                }

                first_block_in_rsi = false;

                // Check if RSI is complete
                if rsi_buf.len() >= rsi_samples {
                    break;
                }
            }

            // Postprocess RSI
            if pp {
                let processed = if self.flags & AEC_DATA_SIGNED != 0 {
                    postprocess_signed(&rsi_buf, self.bits_per_sample, self.xmax)
                } else {
                    postprocess_unsigned(&rsi_buf, self.xmax)
                };
                all_output.extend_from_slice(&processed);
            } else {
                all_output.extend_from_slice(&rsi_buf);
            }
        }

        all_output.truncate(output_samples);
        Ok(all_output)
    }

    fn write_samples(&self, samples: &[u32], output_size: usize) -> Vec<u8> {
        let bps = self.bytes_per_sample as usize;
        let msb = self.flags & AEC_DATA_MSB != 0;
        let mut out = Vec::with_capacity(output_size);
        for &s in samples {
            match bps {
                1 => out.push(s as u8),
                2 => {
                    if msb {
                        out.push((s >> 8) as u8);
                        out.push(s as u8);
                    } else {
                        out.push(s as u8);
                        out.push((s >> 8) as u8);
                    }
                }
                3 => {
                    if msb {
                        out.push((s >> 16) as u8);
                        out.push((s >> 8) as u8);
                        out.push(s as u8);
                    } else {
                        out.push(s as u8);
                        out.push((s >> 8) as u8);
                        out.push((s >> 16) as u8);
                    }
                }
                4 => {
                    if msb {
                        out.push((s >> 24) as u8);
                        out.push((s >> 16) as u8);
                        out.push((s >> 8) as u8);
                        out.push(s as u8);
                    } else {
                        out.push(s as u8);
                        out.push((s >> 8) as u8);
                        out.push((s >> 16) as u8);
                        out.push((s >> 24) as u8);
                    }
                }
                _ => unreachable!(),
            }
            if out.len() >= output_size {
                break;
            }
        }
        out.truncate(output_size);
        out
    }
}

// ===========================================================================
//  Public API
// ===========================================================================

/// Compress data using the SZIP (AEC) algorithm.
///
/// Parameters match the HDF5 SZIP filter interface:
/// - `data`: raw uncompressed bytes
/// - `bits_per_pixel`: sample width (1-32, or 64 for double interleaving)
/// - `pixels_per_block`: block size (must be even, typically 8/16/32)
/// - `pixels_per_scanline`: scanline width in pixels
/// - `options_mask`: SZIP option flags
pub fn compress(
    data: &[u8],
    bits_per_pixel: u32,
    pixels_per_block: u32,
    pixels_per_scanline: u32,
    options_mask: u32,
) -> Result<Vec<u8>, String> {
    if pixels_per_scanline == 0
        || pixels_per_block == 0
        || pixels_per_block & 1 != 0
        || bits_per_pixel == 0
        || (bits_per_pixel > 32 && bits_per_pixel != 64)
    {
        return Err("invalid SZIP parameters".into());
    }

    let flags = AEC_NOT_ENFORCE | convert_options(options_mask);
    let block_size = pixels_per_block;
    let rsi = pixels_per_scanline.div_ceil(pixels_per_block);

    let interleave = bits_per_pixel == 32 || bits_per_pixel == 64;
    let bits_per_sample;
    let input_buf: Vec<u8>;

    if interleave && options_mask & SZ_RAW_OPTION_MASK == 0 {
        bits_per_sample = 8;
        input_buf = interleave_buffer(data, (bits_per_pixel / 8) as usize);
    } else {
        bits_per_sample = if bits_per_pixel == 64 {
            8
        } else {
            bits_per_pixel
        };
        input_buf = data.to_vec();
    }

    let pixel_size = bits_to_bytes(bits_per_sample) as usize;
    if !data.len().is_multiple_of(pixel_size) {
        return Err("input size not a multiple of pixel size".into());
    }

    let line_size_bytes = pixels_per_scanline as usize * pixel_size;
    let padded_line_pixels = rsi * block_size;
    let padding_pixels = padded_line_pixels as usize - pixels_per_scanline as usize;
    let padding_size = padding_pixels * pixel_size;

    let padded_input = if padding_size > 0 {
        add_padding(
            &input_buf,
            line_size_bytes,
            padding_size,
            pixel_size,
            flags & AEC_DATA_PREPROCESS != 0,
        )
    } else {
        input_buf
    };

    let encoder = Encoder::new(bits_per_sample, block_size, rsi, flags)?;
    encoder.encode(&padded_input)
}

/// Decompress SZIP (AEC) compressed data.
///
/// Parameters match the HDF5 SZIP filter interface:
/// - `data`: compressed bytes
/// - `output_size`: expected size of decompressed data in bytes
/// - `bits_per_pixel`: sample width (1-32, or 64 for double interleaving)
/// - `pixels_per_block`: block size (must be even)
/// - `pixels_per_scanline`: scanline width in pixels
/// - `options_mask`: SZIP option flags
pub fn decompress(
    data: &[u8],
    output_size: usize,
    bits_per_pixel: u32,
    pixels_per_block: u32,
    pixels_per_scanline: u32,
    options_mask: u32,
) -> Result<Vec<u8>, String> {
    if pixels_per_scanline == 0
        || pixels_per_block == 0
        || pixels_per_block & 1 != 0
        || bits_per_pixel == 0
        || (bits_per_pixel > 32 && bits_per_pixel != 64)
    {
        return Err("invalid SZIP parameters".into());
    }

    let flags = convert_options(options_mask);
    let block_size = pixels_per_block;
    let rsi = pixels_per_scanline.div_ceil(pixels_per_block);

    let deinterleave =
        (bits_per_pixel == 32 || bits_per_pixel == 64) && options_mask & SZ_RAW_OPTION_MASK == 0;
    let bits_per_sample = if deinterleave || bits_per_pixel == 64 {
        8
    } else {
        bits_per_pixel
    };
    let pixel_size = bits_to_bytes(bits_per_sample) as usize;

    let pad_scanline = !pixels_per_scanline.is_multiple_of(pixels_per_block);
    let _extra_buffer = pad_scanline || deinterleave;

    let decode_output_size = if pad_scanline {
        let scanlines = (output_size / pixel_size).div_ceil(pixels_per_scanline as usize);
        rsi as usize * block_size as usize * pixel_size * scanlines
    } else {
        output_size
    };

    let decoder = Decoder::new(bits_per_sample, block_size, rsi, flags)?;
    let output_samples = decode_output_size / pixel_size;
    let samples = decoder.decode(data, output_samples)?;
    let mut raw_bytes = decoder.write_samples(&samples, decode_output_size);

    if pad_scanline {
        let line_size = pixels_per_scanline as usize * pixel_size;
        let padding_size =
            (rsi as usize * block_size as usize - pixels_per_scanline as usize) * pixel_size;
        remove_padding(&mut raw_bytes, line_size, padding_size);
    }

    let result = if deinterleave {
        let len = std::cmp::min(raw_bytes.len(), output_size);
        deinterleave_buffer(&raw_bytes[..len], (bits_per_pixel / 8) as usize)
    } else {
        raw_bytes.truncate(output_size);
        raw_bytes
    };

    Ok(result)
}

// ===========================================================================
//  Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(
        data: &[u8],
        bits_per_pixel: u32,
        pixels_per_block: u32,
        pixels_per_scanline: u32,
        options_mask: u32,
    ) {
        let compressed = compress(
            data,
            bits_per_pixel,
            pixels_per_block,
            pixels_per_scanline,
            options_mask,
        )
        .expect("compress failed");
        let decompressed = decompress(
            &compressed,
            data.len(),
            bits_per_pixel,
            pixels_per_block,
            pixels_per_scanline,
            options_mask,
        )
        .expect("decompress failed");
        assert_eq!(
            data,
            &decompressed[..],
            "roundtrip mismatch for bpp={bits_per_pixel}"
        );
    }

    #[test]
    fn test_roundtrip_u8() {
        let data: Vec<u8> = (0..256u16).map(|i| (i & 0xFF) as u8).collect();
        // NN (preprocess) + MSB
        roundtrip(&data, 8, 16, 256, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_u8_no_preprocess() {
        let data: Vec<u8> = (0..128).collect();
        roundtrip(&data, 8, 16, 128, SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_u16() {
        let mut data = Vec::new();
        for i in 0..128u16 {
            data.push((i >> 8) as u8);
            data.push((i & 0xFF) as u8);
        }
        roundtrip(&data, 16, 16, 128, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_u16_lsb() {
        let mut data = Vec::new();
        for i in 0..128u16 {
            data.push((i & 0xFF) as u8);
            data.push((i >> 8) as u8);
        }
        roundtrip(&data, 16, 16, 128, SZ_NN_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_u32_interleaved() {
        let values: Vec<u32> = (0..64).collect();
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        roundtrip(&data, 32, 16, 64, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_f32() {
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 1.5).collect();
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        roundtrip(&data, 32, 16, 64, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_f64() {
        let values: Vec<f64> = (0..32).map(|i| i as f64 * 2.5).collect();
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        roundtrip(&data, 64, 16, 32, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_zeros() {
        let data = vec![0u8; 256];
        roundtrip(&data, 8, 16, 256, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_constant() {
        let data = vec![42u8; 128];
        roundtrip(&data, 8, 16, 128, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_scanline_padding() {
        // pixels_per_scanline not a multiple of pixels_per_block
        // 100 pixels, block=16 => rsi=7, padded=112
        let data: Vec<u8> = (0..100).collect();
        roundtrip(&data, 8, 16, 100, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_small_block() {
        let data: Vec<u8> = (0..32).collect();
        roundtrip(&data, 8, 8, 32, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_roundtrip_u8_random_like() {
        // Data with varying patterns to exercise different code paths
        let data: Vec<u8> = (0..256).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        roundtrip(&data, 8, 16, 256, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK);
    }

    #[test]
    fn test_compress_reduces_size() {
        // Highly compressible data
        let data = vec![0u8; 1024];
        let compressed = compress(&data, 8, 16, 1024, SZ_NN_OPTION_MASK | SZ_MSB_OPTION_MASK)
            .expect("compress failed");
        assert!(
            compressed.len() < data.len(),
            "compression should reduce size for zeros"
        );
    }
}
