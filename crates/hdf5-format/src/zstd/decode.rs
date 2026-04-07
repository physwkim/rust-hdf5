#![allow(clippy::manual_is_multiple_of, clippy::identity_op)]
//! Self-contained Zstandard decompressor ported from ruzstd 0.8.2.
//!
//! Public API: `decompress(data: &[u8]) -> Result<Vec<u8>, String>`
//!
//! Supports raw blocks, RLE blocks, and compressed blocks with Huffman
//! literals and FSE sequences. No dictionary support.

#![allow(
    clippy::needless_range_loop,
    clippy::len_without_is_empty,
    clippy::upper_case_acronyms,
    clippy::manual_range_contains,
    dead_code
)]

// ============================================================
// Constants
// ============================================================

const ZSTD_MAGIC: u32 = 0xFD2F_B528;
const MIN_WINDOW_SIZE: u64 = 1024;
const MAX_WINDOW_SIZE: u64 = (1 << 41) + 7 * (1 << 38);
const MAX_BLOCK_SIZE: u32 = 128 * 1024;
const MAXIMUM_ALLOWED_WINDOW_SIZE: u64 = 1024 * 1024 * 100;
const MAX_MAX_NUM_BITS: u8 = 11;
const ACC_LOG_OFFSET: u8 = 5;

const MAX_LITERAL_LENGTH_CODE: u8 = 35;
const MAX_MATCH_LENGTH_CODE: u8 = 52;
const MAX_OFFSET_CODE: u8 = 31;

const LL_MAX_LOG: u8 = 9;
const ML_MAX_LOG: u8 = 9;
const OF_MAX_LOG: u8 = 8;

const LL_DEFAULT_ACC_LOG: u8 = 6;
const ML_DEFAULT_ACC_LOG: u8 = 6;
const OF_DEFAULT_ACC_LOG: u8 = 5;

const LITERALS_LENGTH_DEFAULT_DISTRIBUTION: [i32; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1,
    1, 1, -1, -1, -1, -1,
];

const MATCH_LENGTH_DEFAULT_DISTRIBUTION: [i32; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

const OFFSET_DEFAULT_DISTRIBUTION: [i32; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

// ============================================================
// Public API
// ============================================================

/// Decompress a zstd-compressed byte slice, returning the uncompressed data.
///
/// Supports one or more concatenated zstd frames. Skippable frames are skipped.
/// Dictionary frames are not supported.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut cursor = std::io::Cursor::new(data);
    let mut output = Vec::new();
    let mut decoder = FrameDecoder::new();

    loop {
        // Check if we have consumed all the data
        if cursor.position() as usize >= data.len() {
            break;
        }

        match decoder.reset(&mut cursor) {
            Ok(()) => {}
            Err(e) => {
                if let Some(skip_len) = e.skip_frame_length() {
                    let new_pos = cursor.position() + skip_len as u64;
                    if new_pos as usize > data.len() {
                        return Err("Skippable frame extends past end of input".to_string());
                    }
                    cursor.set_position(new_pos);
                    continue;
                }
                // If we already have output and hit an error, it might just be trailing data
                if !output.is_empty() {
                    break;
                }
                return Err(format!("Frame header error: {}", e));
            }
        }

        // Decode all blocks in this frame
        decoder.decode_all_blocks(&mut cursor)?;

        // Collect the output
        if let Some(mut collected) = decoder.collect() {
            output.append(&mut collected);
        }
    }

    Ok(output)
}

// ============================================================
// BitReader (forward)
// ============================================================

struct BitReader<'s> {
    idx: usize,
    source: &'s [u8],
}

impl<'s> BitReader<'s> {
    fn new(source: &'s [u8]) -> BitReader<'s> {
        BitReader { idx: 0, source }
    }

    fn bits_left(&self) -> usize {
        self.source.len() * 8 - self.idx
    }

    fn bits_read(&self) -> usize {
        self.idx
    }

    fn return_bits(&mut self, n: usize) {
        if n > self.idx {
            panic!("Cannot return more bits than have been read");
        }
        self.idx -= n;
    }

    fn get_bits(&mut self, n: usize) -> Result<u64, String> {
        if n > 64 {
            return Err(format!(
                "Cannot read {} bits, maximum is 64",
                n
            ));
        }
        if self.bits_left() < n {
            return Err(format!(
                "Cannot read {} bits, only {} remaining",
                n,
                self.bits_left()
            ));
        }

        let old_idx = self.idx;
        let bits_left_in_current_byte = 8 - (self.idx % 8);
        let bits_not_needed_in_current_byte = 8 - bits_left_in_current_byte;

        let mut value = u64::from(self.source[self.idx / 8] >> bits_not_needed_in_current_byte);

        if bits_left_in_current_byte >= n {
            value &= (1 << n) - 1;
            self.idx += n;
        } else {
            self.idx += bits_left_in_current_byte;
            let full_bytes_needed = (n - bits_left_in_current_byte) / 8;
            let bits_in_last_byte_needed = n - bits_left_in_current_byte - full_bytes_needed * 8;

            let mut bit_shift = bits_left_in_current_byte;

            for _ in 0..full_bytes_needed {
                value |= u64::from(self.source[self.idx / 8]) << bit_shift;
                self.idx += 8;
                bit_shift += 8;
            }

            if bits_in_last_byte_needed > 0 {
                let val_last_byte =
                    u64::from(self.source[self.idx / 8]) & ((1 << bits_in_last_byte_needed) - 1);
                value |= val_last_byte << bit_shift;
                self.idx += bits_in_last_byte_needed;
            }
        }

        debug_assert!(self.idx == old_idx + n);
        Ok(value)
    }
}

// ============================================================
// BitReaderReversed
// ============================================================

struct BitReaderReversed<'s> {
    index: usize,
    bits_consumed: u8,
    extra_bits: usize,
    source: &'s [u8],
    bit_container: u64,
}

impl<'s> BitReaderReversed<'s> {
    fn bits_remaining(&self) -> isize {
        self.index as isize * 8 + (64 - self.bits_consumed as isize) - self.extra_bits as isize
    }

    fn new(source: &'s [u8]) -> BitReaderReversed<'s> {
        BitReaderReversed {
            index: source.len(),
            bits_consumed: 64,
            source,
            bit_container: 0,
            extra_bits: 0,
        }
    }

    #[cold]
    fn refill(&mut self) {
        let bytes_consumed = self.bits_consumed as usize / 8;
        if bytes_consumed == 0 {
            return;
        }

        if self.index >= bytes_consumed {
            self.index -= bytes_consumed;
            self.bits_consumed &= 7;
            let remaining = self.source.len() - self.index;
            if remaining >= 8 {
                self.bit_container =
                    u64::from_le_bytes(self.source[self.index..][..8].try_into().unwrap());
            } else {
                let mut value = [0u8; 8];
                value[..remaining].copy_from_slice(&self.source[self.index..]);
                self.bit_container = u64::from_le_bytes(value);
            }
        } else if self.index > 0 {
            if self.source.len() >= 8 {
                self.bit_container =
                    u64::from_le_bytes(self.source[..8].try_into().unwrap());
            } else {
                let mut value = [0; 8];
                value[..self.source.len()].copy_from_slice(self.source);
                self.bit_container = u64::from_le_bytes(value);
            }

            self.bits_consumed -= 8 * self.index as u8;
            self.index = 0;

            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else if self.bits_consumed < 64 {
            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else {
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            self.bit_container = 0;
        }

        debug_assert!(self.bits_consumed < 8);
    }

    #[inline(always)]
    fn get_bits(&mut self, n: u8) -> u64 {
        if self.bits_consumed + n > 64 {
            self.refill();
        }
        let value = self.peek_bits(n);
        self.consume(n);
        value
    }

    #[inline(always)]
    fn peek_bits(&mut self, n: u8) -> u64 {
        if n == 0 {
            return 0;
        }
        let mask = (1u64 << n) - 1u64;
        let shift_by = 64 - self.bits_consumed - n;
        (self.bit_container >> shift_by) & mask
    }

    #[inline(always)]
    fn peek_bits_triple(&mut self, sum: u8, n1: u8, n2: u8, n3: u8) -> (u64, u64, u64) {
        if sum == 0 {
            return (0, 0, 0);
        }
        let all_three = self.bit_container >> (64 - self.bits_consumed - sum);

        let mask1 = (1u64 << n1) - 1u64;
        let val1 = (all_three >> (n3 + n2)) & mask1;

        let mask2 = (1u64 << n2) - 1u64;
        let val2 = (all_three >> n3) & mask2;

        let mask3 = (1u64 << n3) - 1u64;
        let val3 = all_three & mask3;

        (val1, val2, val3)
    }

    #[inline(always)]
    fn consume(&mut self, n: u8) {
        self.bits_consumed += n;
        debug_assert!(self.bits_consumed <= 64);
    }

    #[inline(always)]
    fn get_bits_triple(&mut self, n1: u8, n2: u8, n3: u8) -> (u64, u64, u64) {
        let sum = n1 + n2 + n3;
        if sum <= 56 {
            self.refill();
            let triple = self.peek_bits_triple(sum, n1, n2, n3);
            self.consume(sum);
            return triple;
        }
        (self.get_bits(n1), self.get_bits(n2), self.get_bits(n3))
    }
}

// ============================================================
// FSE Table and Decoder
// ============================================================

#[derive(Copy, Clone, Debug)]
struct FSEEntry {
    base_line: u32,
    num_bits: u8,
    symbol: u8,
}

#[derive(Debug, Clone)]
struct FSETable {
    max_symbol: u8,
    decode: Vec<FSEEntry>,
    accuracy_log: u8,
    symbol_probabilities: Vec<i32>,
    symbol_counter: Vec<u32>,
}

impl FSETable {
    fn new(max_symbol: u8) -> FSETable {
        FSETable {
            max_symbol,
            symbol_probabilities: Vec::with_capacity(256),
            symbol_counter: Vec::with_capacity(256),
            decode: Vec::new(),
            accuracy_log: 0,
        }
    }

    fn reinit_from(&mut self, other: &Self) {
        self.reset();
        self.symbol_counter.extend_from_slice(&other.symbol_counter);
        self.symbol_probabilities
            .extend_from_slice(&other.symbol_probabilities);
        self.decode.extend_from_slice(&other.decode);
        self.accuracy_log = other.accuracy_log;
    }

    fn reset(&mut self) {
        self.symbol_counter.clear();
        self.symbol_probabilities.clear();
        self.decode.clear();
        self.accuracy_log = 0;
    }

    fn build_decoder(&mut self, source: &[u8], max_log: u8) -> Result<usize, String> {
        self.accuracy_log = 0;
        let bytes_read = self.read_probabilities(source, max_log)?;
        self.build_decoding_table()?;
        Ok(bytes_read)
    }

    fn build_from_probabilities(&mut self, acc_log: u8, probs: &[i32]) -> Result<(), String> {
        if acc_log == 0 {
            return Err("Accuracy log is zero".to_string());
        }
        self.symbol_probabilities = probs.to_vec();
        self.accuracy_log = acc_log;
        self.build_decoding_table()
    }

    fn build_decoding_table(&mut self) -> Result<(), String> {
        if self.symbol_probabilities.len() > self.max_symbol as usize + 1 {
            return Err(format!(
                "Too many symbols: {}, max: {}",
                self.symbol_probabilities.len(),
                self.max_symbol + 1
            ));
        }

        self.decode.clear();

        let table_size = 1 << self.accuracy_log;
        self.decode.resize(
            table_size,
            FSEEntry {
                base_line: 0,
                num_bits: 0,
                symbol: 0,
            },
        );

        let mut negative_idx = table_size;

        for symbol in 0..self.symbol_probabilities.len() {
            if self.symbol_probabilities[symbol] == -1 {
                negative_idx -= 1;
                let entry = &mut self.decode[negative_idx];
                entry.symbol = symbol as u8;
                entry.base_line = 0;
                entry.num_bits = self.accuracy_log;
            }
        }

        let mut position = 0;
        for idx in 0..self.symbol_probabilities.len() {
            let symbol = idx as u8;
            if self.symbol_probabilities[idx] <= 0 {
                continue;
            }
            let prob = self.symbol_probabilities[idx];
            for _ in 0..prob {
                let entry = &mut self.decode[position];
                entry.symbol = symbol;
                position = fse_next_position(position, table_size);
                while position >= negative_idx {
                    position = fse_next_position(position, table_size);
                }
            }
        }

        self.symbol_counter.clear();
        self.symbol_counter
            .resize(self.symbol_probabilities.len(), 0);
        for idx in 0..negative_idx {
            let symbol = self.decode[idx].symbol;
            let prob = self.symbol_probabilities[symbol as usize];
            let symbol_count = self.symbol_counter[symbol as usize];
            let (bl, nb) =
                fse_calc_baseline_and_numbits(table_size as u32, prob as u32, symbol_count);

            assert!(nb <= self.accuracy_log);
            self.symbol_counter[symbol as usize] += 1;

            self.decode[idx].base_line = bl;
            self.decode[idx].num_bits = nb;
        }
        Ok(())
    }

    fn read_probabilities(&mut self, source: &[u8], max_log: u8) -> Result<usize, String> {
        self.symbol_probabilities.clear();

        let mut br = BitReader::new(source);
        self.accuracy_log = ACC_LOG_OFFSET + (br.get_bits(4)? as u8);
        if self.accuracy_log > max_log {
            return Err(format!(
                "Accuracy log {} exceeds max {}",
                self.accuracy_log, max_log
            ));
        }
        if self.accuracy_log == 0 {
            return Err("Accuracy log is zero".to_string());
        }

        let probability_sum = 1u32 << self.accuracy_log;
        let mut probability_counter = 0u32;

        while probability_counter < probability_sum {
            let max_remaining_value = probability_sum - probability_counter + 1;
            let bits_to_read = highest_bit_set(max_remaining_value);

            let unchecked_value = br.get_bits(bits_to_read as usize)? as u32;

            let low_threshold = ((1 << bits_to_read) - 1) - max_remaining_value;
            let mask = (1 << (bits_to_read - 1)) - 1;
            let small_value = unchecked_value & mask;

            let value = if small_value < low_threshold {
                br.return_bits(1);
                small_value
            } else if unchecked_value > mask {
                unchecked_value - low_threshold
            } else {
                unchecked_value
            };

            let prob = (value as i32) - 1;
            self.symbol_probabilities.push(prob);

            if prob != 0 {
                if prob > 0 {
                    probability_counter += prob as u32;
                } else {
                    // probability -1 counts as 1
                    probability_counter += 1;
                }
            } else {
                loop {
                    let skip_amount = br.get_bits(2)? as usize;
                    self.symbol_probabilities
                        .resize(self.symbol_probabilities.len() + skip_amount, 0);
                    if skip_amount != 3 {
                        break;
                    }
                }
            }
        }

        if probability_counter != probability_sum {
            return Err(format!(
                "Probability counter {} does not match expected sum {}",
                probability_counter, probability_sum
            ));
        }
        if self.symbol_probabilities.len() > self.max_symbol as usize + 1 {
            return Err(format!(
                "Too many symbols: {}",
                self.symbol_probabilities.len()
            ));
        }

        let bytes_read = if br.bits_read() % 8 == 0 {
            br.bits_read() / 8
        } else {
            (br.bits_read() / 8) + 1
        };

        Ok(bytes_read)
    }
}

fn fse_next_position(mut p: usize, table_size: usize) -> usize {
    p += (table_size >> 1) + (table_size >> 3) + 3;
    p &= table_size - 1;
    p
}

fn fse_calc_baseline_and_numbits(
    num_states_total: u32,
    num_states_symbol: u32,
    state_number: u32,
) -> (u32, u8) {
    if num_states_symbol == 0 {
        return (0, 0);
    }
    let num_state_slices = if 1 << (highest_bit_set(num_states_symbol) - 1) == num_states_symbol {
        num_states_symbol
    } else {
        1 << highest_bit_set(num_states_symbol)
    };

    let num_double_width_state_slices = num_state_slices - num_states_symbol;
    let num_single_width_state_slices = num_states_symbol - num_double_width_state_slices;
    let slice_width = num_states_total / num_state_slices;
    let num_bits = highest_bit_set(slice_width) - 1;

    if state_number < num_double_width_state_slices {
        let baseline =
            num_single_width_state_slices * slice_width + state_number * slice_width * 2;
        (baseline, num_bits as u8 + 1)
    } else {
        let index_shifted = state_number - num_double_width_state_slices;
        (index_shifted * slice_width, num_bits as u8)
    }
}

fn highest_bit_set(x: u32) -> u32 {
    assert!(x > 0);
    u32::BITS - x.leading_zeros()
}

struct FSEDecoder<'table> {
    state: FSEEntry,
    table: &'table FSETable,
}

impl<'t> FSEDecoder<'t> {
    fn new(table: &'t FSETable) -> FSEDecoder<'t> {
        FSEDecoder {
            state: table.decode.first().copied().unwrap_or(FSEEntry {
                base_line: 0,
                num_bits: 0,
                symbol: 0,
            }),
            table,
        }
    }

    fn decode_symbol(&self) -> u8 {
        self.state.symbol
    }

    fn init_state(&mut self, bits: &mut BitReaderReversed<'_>) -> Result<(), String> {
        if self.table.accuracy_log == 0 {
            return Err("FSE table is uninitialized".to_string());
        }
        let new_state = bits.get_bits(self.table.accuracy_log);
        self.state = self.table.decode[new_state as usize];
        Ok(())
    }

    fn update_state(&mut self, bits: &mut BitReaderReversed<'_>) {
        let num_bits = self.state.num_bits;
        let add = bits.get_bits(num_bits);
        let base_line = self.state.base_line;
        let new_state = base_line + add as u32;
        self.state = self.table.decode[new_state as usize];
    }
}

// ============================================================
// Huffman Table and Decoder
// ============================================================

#[derive(Copy, Clone, Debug)]
struct HuffmanEntry {
    symbol: u8,
    num_bits: u8,
}

struct HuffmanTable {
    decode: Vec<HuffmanEntry>,
    weights: Vec<u8>,
    max_num_bits: u8,
    bits: Vec<u8>,
    bit_ranks: Vec<u32>,
    rank_indexes: Vec<usize>,
    fse_table: FSETable,
}

impl HuffmanTable {
    fn new() -> HuffmanTable {
        HuffmanTable {
            decode: Vec::new(),
            weights: Vec::with_capacity(256),
            max_num_bits: 0,
            bits: Vec::with_capacity(256),
            bit_ranks: Vec::with_capacity(11),
            rank_indexes: Vec::with_capacity(11),
            fse_table: FSETable::new(255),
        }
    }

    fn reinit_from(&mut self, other: &Self) {
        self.reset();
        self.decode.extend_from_slice(&other.decode);
        self.weights.extend_from_slice(&other.weights);
        self.max_num_bits = other.max_num_bits;
        self.bits.extend_from_slice(&other.bits);
        self.rank_indexes.extend_from_slice(&other.rank_indexes);
        self.fse_table.reinit_from(&other.fse_table);
    }

    fn reset(&mut self) {
        self.decode.clear();
        self.weights.clear();
        self.max_num_bits = 0;
        self.bits.clear();
        self.bit_ranks.clear();
        self.rank_indexes.clear();
        self.fse_table.reset();
    }

    fn build_decoder(&mut self, source: &[u8]) -> Result<u32, String> {
        self.decode.clear();
        let bytes_used = self.read_weights(source)?;
        self.build_table_from_weights()?;
        Ok(bytes_used)
    }

    fn read_weights(&mut self, source: &[u8]) -> Result<u32, String> {
        if source.is_empty() {
            return Err("Huffman source is empty".to_string());
        }
        let header = source[0];
        let mut bits_read = 8;

        match header {
            0..=127 => {
                let fse_stream = &source[1..];
                if (header as usize) > fse_stream.len() {
                    return Err(format!(
                        "Not enough bytes for weights: have {}, need {}",
                        fse_stream.len(),
                        header
                    ));
                }
                let bytes_used_by_fse_header = self.fse_table.build_decoder(fse_stream, 6)?;

                if bytes_used_by_fse_header > header as usize {
                    return Err(format!(
                        "FSE table used {} bytes but only {} available",
                        bytes_used_by_fse_header, header
                    ));
                }

                let mut dec1 = FSEDecoder::new(&self.fse_table);
                let mut dec2 = FSEDecoder::new(&self.fse_table);

                let compressed_start = bytes_used_by_fse_header;
                let compressed_length = header as usize - bytes_used_by_fse_header;

                let compressed_weights = &fse_stream[compressed_start..];
                if compressed_weights.len() < compressed_length {
                    return Err(format!(
                        "Not enough bytes to decompress weights: have {}, need {}",
                        compressed_weights.len(),
                        compressed_length
                    ));
                }
                let compressed_weights = &compressed_weights[..compressed_length];
                let mut br = BitReaderReversed::new(compressed_weights);

                bits_read += (bytes_used_by_fse_header + compressed_length) * 8;

                let mut skipped_bits = 0;
                loop {
                    let val = br.get_bits(1);
                    skipped_bits += 1;
                    if val == 1 || skipped_bits > 8 {
                        break;
                    }
                }
                if skipped_bits > 8 {
                    return Err(format!("Extra padding: {} bits skipped", skipped_bits));
                }

                dec1.init_state(&mut br)?;
                dec2.init_state(&mut br)?;

                self.weights.clear();

                loop {
                    let w = dec1.decode_symbol();
                    self.weights.push(w);
                    dec1.update_state(&mut br);

                    if br.bits_remaining() <= -1 {
                        self.weights.push(dec2.decode_symbol());
                        break;
                    }

                    let w = dec2.decode_symbol();
                    self.weights.push(w);
                    dec2.update_state(&mut br);

                    if br.bits_remaining() <= -1 {
                        self.weights.push(dec1.decode_symbol());
                        break;
                    }
                    if self.weights.len() > 255 {
                        return Err(format!("Too many weights: {}", self.weights.len()));
                    }
                }
            }
            _ => {
                let weights_raw = &source[1..];
                let num_weights = header - 127;
                self.weights.resize(num_weights as usize, 0);

                let bytes_needed = if num_weights % 2 == 0 {
                    num_weights as usize / 2
                } else {
                    (num_weights as usize / 2) + 1
                };

                if weights_raw.len() < bytes_needed {
                    return Err(format!(
                        "Not enough bytes in source: have {}, need {}",
                        weights_raw.len(),
                        bytes_needed
                    ));
                }

                for idx in 0..num_weights {
                    if idx % 2 == 0 {
                        self.weights[idx as usize] = weights_raw[idx as usize / 2] >> 4;
                    } else {
                        self.weights[idx as usize] = weights_raw[idx as usize / 2] & 0xF;
                    }
                    bits_read += 4;
                }
            }
        }

        let bytes_read = if bits_read % 8 == 0 {
            bits_read / 8
        } else {
            (bits_read / 8) + 1
        };
        Ok(bytes_read as u32)
    }

    fn build_table_from_weights(&mut self) -> Result<(), String> {
        self.bits.clear();
        self.bits.resize(self.weights.len() + 1, 0);

        let mut weight_sum: u32 = 0;
        for w in &self.weights {
            if *w > MAX_MAX_NUM_BITS {
                return Err(format!("Weight {} exceeds max {}", w, MAX_MAX_NUM_BITS));
            }
            weight_sum += if *w > 0 { 1_u32 << (*w - 1) } else { 0 };
        }

        if weight_sum == 0 {
            return Err("Missing weights".to_string());
        }

        let max_bits = highest_bit_set(weight_sum) as u8;
        let left_over = (1u32 << max_bits) - weight_sum;

        if !left_over.is_power_of_two() {
            return Err(format!("Leftover {} is not a power of 2", left_over));
        }

        let last_weight = highest_bit_set(left_over) as u8;

        for symbol in 0..self.weights.len() {
            let bits = if self.weights[symbol] > 0 {
                max_bits + 1 - self.weights[symbol]
            } else {
                0
            };
            self.bits[symbol] = bits;
        }

        self.bits[self.weights.len()] = max_bits + 1 - last_weight;
        self.max_num_bits = max_bits;

        if max_bits > MAX_MAX_NUM_BITS {
            return Err(format!("Max bits {} too high", max_bits));
        }

        self.bit_ranks.clear();
        self.bit_ranks.resize((max_bits + 1) as usize, 0);
        for num_bits in &self.bits {
            self.bit_ranks[(*num_bits) as usize] += 1;
        }

        self.decode.resize(
            1 << self.max_num_bits,
            HuffmanEntry {
                symbol: 0,
                num_bits: 0,
            },
        );

        self.rank_indexes.clear();
        self.rank_indexes.resize((max_bits + 1) as usize, 0);

        self.rank_indexes[max_bits as usize] = 0;
        for bits in (1..self.rank_indexes.len() as u8).rev() {
            self.rank_indexes[bits as usize - 1] = self.rank_indexes[bits as usize]
                + self.bit_ranks[bits as usize] as usize * (1 << (max_bits - bits));
        }

        assert!(
            self.rank_indexes[0] == self.decode.len(),
            "rank_idx[0]: {} should be: {}",
            self.rank_indexes[0],
            self.decode.len()
        );

        for symbol in 0..self.bits.len() {
            let bits_for_symbol = self.bits[symbol];
            if bits_for_symbol != 0 {
                let base_idx = self.rank_indexes[bits_for_symbol as usize];
                let len = 1 << (max_bits - bits_for_symbol);
                self.rank_indexes[bits_for_symbol as usize] += len;
                for idx in 0..len {
                    self.decode[base_idx + idx].symbol = symbol as u8;
                    self.decode[base_idx + idx].num_bits = bits_for_symbol;
                }
            }
        }

        Ok(())
    }
}

struct HuffmanDecoder<'table> {
    table: &'table HuffmanTable,
    state: u64,
}

impl<'t> HuffmanDecoder<'t> {
    fn new(table: &'t HuffmanTable) -> HuffmanDecoder<'t> {
        HuffmanDecoder { table, state: 0 }
    }

    fn decode_symbol(&mut self) -> u8 {
        self.table.decode[self.state as usize].symbol
    }

    fn init_state(&mut self, br: &mut BitReaderReversed<'_>) -> u8 {
        let num_bits = self.table.max_num_bits;
        let new_bits = br.get_bits(num_bits);
        self.state = new_bits;
        num_bits
    }

    fn next_state(&mut self, br: &mut BitReaderReversed<'_>) -> u8 {
        let num_bits = self.table.decode[self.state as usize].num_bits;
        let new_bits = br.get_bits(num_bits);
        self.state <<= num_bits;
        self.state &= self.table.decode.len() as u64 - 1;
        self.state |= new_bits;
        num_bits
    }
}

// ============================================================
// Block types and headers
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockType {
    Raw,
    RLE,
    Compressed,
    Reserved,
}

struct BlockHeader {
    last_block: bool,
    block_type: BlockType,
    decompressed_size: u32,
    content_size: u32,
}

// ============================================================
// Literals Section
// ============================================================

enum LiteralsSectionType {
    Raw,
    RLE,
    Compressed,
    Treeless,
}

struct LiteralsSection {
    regenerated_size: u32,
    compressed_size: Option<u32>,
    num_streams: Option<u8>,
    ls_type: LiteralsSectionType,
}

impl LiteralsSection {
    fn new() -> LiteralsSection {
        LiteralsSection {
            regenerated_size: 0,
            compressed_size: None,
            num_streams: None,
            ls_type: LiteralsSectionType::Raw,
        }
    }

    fn section_type(raw: u8) -> Result<LiteralsSectionType, String> {
        let t = raw & 0x3;
        match t {
            0 => Ok(LiteralsSectionType::Raw),
            1 => Ok(LiteralsSectionType::RLE),
            2 => Ok(LiteralsSectionType::Compressed),
            3 => Ok(LiteralsSectionType::Treeless),
            other => Err(format!("Illegal literal section type: {}", other)),
        }
    }

    fn header_bytes_needed(&self, first_byte: u8) -> Result<u8, String> {
        let ls_type = Self::section_type(first_byte)?;
        let size_format = (first_byte >> 2) & 0x3;
        match ls_type {
            LiteralsSectionType::RLE | LiteralsSectionType::Raw => match size_format {
                0 | 2 => Ok(1),
                1 => Ok(2),
                3 => Ok(3),
                _ => unreachable!(),
            },
            LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => match size_format {
                0 | 1 => Ok(3),
                2 => Ok(4),
                3 => Ok(5),
                _ => unreachable!(),
            },
        }
    }

    fn parse_from_header(&mut self, raw: &[u8]) -> Result<u8, String> {
        let mut br = BitReader::new(raw);
        let block_type = br.get_bits(2)? as u8;
        self.ls_type = Self::section_type(block_type)?;
        let size_format = br.get_bits(2)? as u8;

        let byte_needed = self.header_bytes_needed(raw[0])?;
        if raw.len() < byte_needed as usize {
            return Err(format!(
                "Not enough bytes for literals header: have {}, need {}",
                raw.len(),
                byte_needed
            ));
        }

        match self.ls_type {
            LiteralsSectionType::RLE | LiteralsSectionType::Raw => {
                self.compressed_size = None;
                match size_format {
                    0 | 2 => {
                        self.regenerated_size = u32::from(raw[0]) >> 3;
                        Ok(1)
                    }
                    1 => {
                        self.regenerated_size =
                            (u32::from(raw[0]) >> 4) + (u32::from(raw[1]) << 4);
                        Ok(2)
                    }
                    3 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4)
                            + (u32::from(raw[1]) << 4)
                            + (u32::from(raw[2]) << 12);
                        Ok(3)
                    }
                    _ => unreachable!(),
                }
            }
            LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
                match size_format {
                    0 => {
                        self.num_streams = Some(1);
                    }
                    1..=3 => {
                        self.num_streams = Some(4);
                    }
                    _ => unreachable!(),
                };

                match size_format {
                    0 | 1 => {
                        self.regenerated_size =
                            (u32::from(raw[0]) >> 4) + ((u32::from(raw[1]) & 0x3f) << 4);
                        self.compressed_size =
                            Some(u32::from(raw[1] >> 6) + (u32::from(raw[2]) << 2));
                        Ok(3)
                    }
                    2 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4)
                            + (u32::from(raw[1]) << 4)
                            + ((u32::from(raw[2]) & 0x3) << 12);
                        self.compressed_size =
                            Some((u32::from(raw[2]) >> 2) + (u32::from(raw[3]) << 6));
                        Ok(4)
                    }
                    3 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4)
                            + (u32::from(raw[1]) << 4)
                            + ((u32::from(raw[2]) & 0x3F) << 12);
                        self.compressed_size = Some(
                            (u32::from(raw[2]) >> 6)
                                + (u32::from(raw[3]) << 2)
                                + (u32::from(raw[4]) << 10),
                        );
                        Ok(5)
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

// ============================================================
// Sequences Section
// ============================================================

#[derive(Clone, Copy)]
struct Sequence {
    ll: u32,
    ml: u32,
    of: u32,
}

#[derive(Copy, Clone)]
struct CompressionModes(u8);

enum ModeType {
    Predefined,
    RLE,
    FSECompressed,
    Repeat,
}

impl CompressionModes {
    fn decode_mode(m: u8) -> ModeType {
        match m {
            0 => ModeType::Predefined,
            1 => ModeType::RLE,
            2 => ModeType::FSECompressed,
            3 => ModeType::Repeat,
            _ => panic!("Invalid mode value"),
        }
    }
    fn ll_mode(self) -> ModeType {
        Self::decode_mode(self.0 >> 6)
    }
    fn of_mode(self) -> ModeType {
        Self::decode_mode((self.0 >> 4) & 0x3)
    }
    fn ml_mode(self) -> ModeType {
        Self::decode_mode((self.0 >> 2) & 0x3)
    }
}

struct SequencesHeader {
    num_sequences: u32,
    modes: Option<CompressionModes>,
}

impl SequencesHeader {
    fn new() -> SequencesHeader {
        SequencesHeader {
            num_sequences: 0,
            modes: None,
        }
    }

    fn parse_from_header(&mut self, source: &[u8]) -> Result<u8, String> {
        let mut bytes_read = 0;
        if source.is_empty() {
            return Err("Sequences header source is empty".to_string());
        }

        match source[0] {
            0 => {
                self.num_sequences = 0;
                bytes_read += 1;
            }
            1..=127 => {
                if source.len() < 2 {
                    return Err(format!(
                        "Not enough bytes for sequences header: have {}, need 2",
                        source.len()
                    ));
                }
                self.num_sequences = u32::from(source[0]);
                self.modes = Some(CompressionModes(source[1]));
                bytes_read += 2;
            }
            128..=254 => {
                if source.len() < 2 {
                    return Err(format!(
                        "Not enough bytes for sequences header: have {}, need 2",
                        source.len()
                    ));
                }
                self.num_sequences =
                    ((u32::from(source[0]) - 128) << 8) + u32::from(source[1]);
                bytes_read += 2;
                if self.num_sequences != 0 {
                    if source.len() < 3 {
                        return Err(format!(
                            "Not enough bytes for sequences header: have {}, need 3",
                            source.len()
                        ));
                    }
                    self.modes = Some(CompressionModes(source[2]));
                    bytes_read += 1;
                }
            }
            255 => {
                if source.len() < 4 {
                    return Err(format!(
                        "Not enough bytes for sequences header: have {}, need 4",
                        source.len()
                    ));
                }
                self.num_sequences =
                    u32::from(source[1]) + (u32::from(source[2]) << 8) + 0x7F00;
                self.modes = Some(CompressionModes(source[3]));
                bytes_read += 4;
            }
        }

        Ok(bytes_read)
    }
}

// ============================================================
// Decode Buffer (Vec-based, no ringbuffer)
// ============================================================

struct DecodeBuffer {
    buffer: Vec<u8>,
    window_size: usize,
}

impl DecodeBuffer {
    fn new(window_size: usize) -> DecodeBuffer {
        DecodeBuffer {
            buffer: Vec::new(),
            window_size,
        }
    }

    fn reset(&mut self, window_size: usize) {
        self.window_size = window_size;
        self.buffer.clear();
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn push(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    fn repeat(&mut self, offset: usize, match_length: usize) -> Result<(), String> {
        if offset > self.buffer.len() {
            return Err(format!(
                "Offset {} exceeds buffer length {}",
                offset,
                self.buffer.len()
            ));
        }
        if offset == 0 {
            return Err("Zero offset in repeat".to_string());
        }

        let start_idx = self.buffer.len() - offset;
        self.buffer.reserve(match_length);

        for i in 0..match_length {
            let byte = self.buffer[start_idx + (i % offset)];
            self.buffer.push(byte);
        }

        Ok(())
    }

    fn drain(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buffer)
    }
}

// ============================================================
// Scratch space
// ============================================================

struct HuffmanScratch {
    table: HuffmanTable,
}

struct FSEScratch {
    offsets: FSETable,
    of_rle: Option<u8>,
    literal_lengths: FSETable,
    ll_rle: Option<u8>,
    match_lengths: FSETable,
    ml_rle: Option<u8>,
}

struct DecoderScratch {
    huf: HuffmanScratch,
    fse: FSEScratch,
    buffer: DecodeBuffer,
    offset_hist: [u32; 3],
    literals_buffer: Vec<u8>,
    sequences: Vec<Sequence>,
    block_content_buffer: Vec<u8>,
}

impl DecoderScratch {
    fn new(window_size: usize) -> DecoderScratch {
        DecoderScratch {
            huf: HuffmanScratch {
                table: HuffmanTable::new(),
            },
            fse: FSEScratch {
                offsets: FSETable::new(MAX_OFFSET_CODE),
                of_rle: None,
                literal_lengths: FSETable::new(MAX_LITERAL_LENGTH_CODE),
                ll_rle: None,
                match_lengths: FSETable::new(MAX_MATCH_LENGTH_CODE),
                ml_rle: None,
            },
            buffer: DecodeBuffer::new(window_size),
            offset_hist: [1, 4, 8],
            block_content_buffer: Vec::new(),
            literals_buffer: Vec::new(),
            sequences: Vec::new(),
        }
    }

    fn reset(&mut self, window_size: usize) {
        self.offset_hist = [1, 4, 8];
        self.literals_buffer.clear();
        self.sequences.clear();
        self.block_content_buffer.clear();
        self.buffer.reset(window_size);
        self.fse.literal_lengths.reset();
        self.fse.match_lengths.reset();
        self.fse.offsets.reset();
        self.fse.ll_rle = None;
        self.fse.ml_rle = None;
        self.fse.of_rle = None;
        self.huf.table.reset();
    }
}

// ============================================================
// Frame header
// ============================================================

struct FrameDescriptor(u8);

impl FrameDescriptor {
    fn frame_content_size_flag(&self) -> u8 {
        self.0 >> 6
    }

    fn single_segment_flag(&self) -> bool {
        ((self.0 >> 5) & 0x1) == 1
    }

    fn content_checksum_flag(&self) -> bool {
        ((self.0 >> 2) & 0x1) == 1
    }

    fn dict_id_flag(&self) -> u8 {
        self.0 & 0x3
    }

    fn frame_content_size_bytes(&self) -> Result<u8, String> {
        match self.frame_content_size_flag() {
            0 => {
                if self.single_segment_flag() {
                    Ok(1)
                } else {
                    Ok(0)
                }
            }
            1 => Ok(2),
            2 => Ok(4),
            3 => Ok(8),
            other => Err(format!("Invalid frame content size flag: {}", other)),
        }
    }

    fn dictionary_id_bytes(&self) -> Result<u8, String> {
        match self.dict_id_flag() {
            0 => Ok(0),
            1 => Ok(1),
            2 => Ok(2),
            3 => Ok(4),
            other => Err(format!("Invalid dict id flag: {}", other)),
        }
    }
}

struct FrameHeader {
    descriptor: FrameDescriptor,
    window_descriptor: u8,
    frame_content_size: u64,
}

impl FrameHeader {
    fn window_size(&self) -> Result<u64, String> {
        if self.descriptor.single_segment_flag() {
            Ok(self.frame_content_size)
        } else {
            let exp = self.window_descriptor >> 3;
            let mantissa = self.window_descriptor & 0x7;

            let window_log = 10 + u64::from(exp);
            let window_base = 1u64 << window_log;
            let window_add = (window_base / 8) * u64::from(mantissa);

            let window_size = window_base + window_add;

            if window_size < MIN_WINDOW_SIZE {
                Err(format!("Window size {} too small", window_size))
            } else if window_size >= MAX_WINDOW_SIZE {
                Err(format!("Window size {} too big", window_size))
            } else {
                Ok(window_size)
            }
        }
    }

    fn frame_content_size(&self) -> u64 {
        self.frame_content_size
    }
}

// ============================================================
// Error wrapper for skip frames
// ============================================================

struct FrameDecoderError {
    msg: String,
    skip_length: Option<u32>,
}

impl FrameDecoderError {
    fn new(msg: String) -> Self {
        Self {
            msg,
            skip_length: None,
        }
    }

    fn skip(length: u32) -> Self {
        Self {
            msg: format!("Skippable frame with length {}", length),
            skip_length: Some(length),
        }
    }

    fn skip_frame_length(&self) -> Option<u32> {
        self.skip_length
    }
}

impl std::fmt::Display for FrameDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

// ============================================================
// Frame header reading
// ============================================================

fn read_frame_header(r: &mut dyn std::io::Read) -> Result<(FrameHeader, u8), FrameDecoderError> {
    let mut buf = [0u8; 4];

    r.read_exact(&mut buf)
        .map_err(|e| FrameDecoderError::new(format!("Error reading magic number: {}", e)))?;
    let mut bytes_read: usize = 4;
    let magic_num = u32::from_le_bytes(buf);

    // Skippable frames
    if (0x184D2A50..=0x184D2A5F).contains(&magic_num) {
        r.read_exact(&mut buf)
            .map_err(|e| FrameDecoderError::new(format!("Error reading skip frame size: {}", e)))?;
        let skip_size = u32::from_le_bytes(buf);
        return Err(FrameDecoderError::skip(skip_size));
    }

    if magic_num != ZSTD_MAGIC {
        return Err(FrameDecoderError::new(format!(
            "Bad magic number: 0x{:X}",
            magic_num
        )));
    }

    r.read_exact(&mut buf[0..1])
        .map_err(|e| FrameDecoderError::new(format!("Error reading frame descriptor: {}", e)))?;
    let desc = FrameDescriptor(buf[0]);
    bytes_read += 1;

    let mut frame_header = FrameHeader {
        descriptor: FrameDescriptor(desc.0),
        frame_content_size: 0,
        window_descriptor: 0,
    };

    if !desc.single_segment_flag() {
        r.read_exact(&mut buf[0..1]).map_err(|e| {
            FrameDecoderError::new(format!("Error reading window descriptor: {}", e))
        })?;
        frame_header.window_descriptor = buf[0];
        bytes_read += 1;
    }

    let dict_id_len = desc
        .dictionary_id_bytes()
        .map_err(FrameDecoderError::new)?
        as usize;
    if dict_id_len != 0 {
        let buf = &mut buf[..dict_id_len];
        r.read_exact(buf).map_err(|e| {
            FrameDecoderError::new(format!("Error reading dictionary id: {}", e))
        })?;
        bytes_read += dict_id_len;
        // We don't support dictionaries, but we still need to skip these bytes
    }

    let fcs_len = desc
        .frame_content_size_bytes()
        .map_err(FrameDecoderError::new)?
        as usize;
    if fcs_len != 0 {
        let mut fcs_buf = [0u8; 8];
        let fcs_buf = &mut fcs_buf[..fcs_len];
        r.read_exact(fcs_buf).map_err(|e| {
            FrameDecoderError::new(format!("Error reading frame content size: {}", e))
        })?;
        bytes_read += fcs_len;
        let mut fcs = 0u64;
        for i in 0..fcs_len {
            fcs += (fcs_buf[i] as u64) << (8 * i);
        }
        if fcs_len == 2 {
            fcs += 256;
        }
        frame_header.frame_content_size = fcs;
    }

    Ok((frame_header, bytes_read as u8))
}

// ============================================================
// Block header reading
// ============================================================

fn read_block_header(r: &mut dyn std::io::Read) -> Result<(BlockHeader, u8), String> {
    let mut buf = [0u8; 3];
    r.read_exact(&mut buf)
        .map_err(|e| format!("Error reading block header: {}", e))?;

    let last_block = buf[0] & 0x1 == 1;
    let block_type_raw = (buf[0] >> 1) & 0x3;
    let block_type = match block_type_raw {
        0 => BlockType::Raw,
        1 => BlockType::RLE,
        2 => BlockType::Compressed,
        3 => BlockType::Reserved,
        _ => unreachable!(),
    };

    if block_type == BlockType::Reserved {
        return Err("Found reserved block type".to_string());
    }

    let block_size = u32::from(buf[0] >> 3)
        | (u32::from(buf[1]) << 5)
        | (u32::from(buf[2]) << 13);

    if block_size > MAX_BLOCK_SIZE {
        return Err(format!("Block size {} exceeds max {}", block_size, MAX_BLOCK_SIZE));
    }

    let decompressed_size = match block_type {
        BlockType::Raw | BlockType::RLE => block_size,
        BlockType::Compressed | BlockType::Reserved => 0,
    };
    let content_size = match block_type {
        BlockType::Raw | BlockType::Compressed => block_size,
        BlockType::RLE => 1,
        BlockType::Reserved => 0,
    };

    Ok((
        BlockHeader {
            last_block,
            block_type,
            decompressed_size,
            content_size,
        },
        3,
    ))
}

// ============================================================
// Literals section decoder
// ============================================================

fn decode_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, String> {
    match section.ls_type {
        LiteralsSectionType::Raw => {
            target.extend(&source[0..section.regenerated_size as usize]);
            Ok(section.regenerated_size)
        }
        LiteralsSectionType::RLE => {
            target.resize(target.len() + section.regenerated_size as usize, source[0]);
            Ok(1)
        }
        LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
            decompress_literals(section, scratch, source, target)
        }
    }
}

fn decompress_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, String> {
    let compressed_size = section
        .compressed_size
        .ok_or_else(|| "Missing compressed size".to_string())? as usize;
    let num_streams = section
        .num_streams
        .ok_or_else(|| "Missing num_streams".to_string())?;

    target.reserve(section.regenerated_size as usize);
    let source = &source[0..compressed_size];
    let mut bytes_read = 0u32;

    match section.ls_type {
        LiteralsSectionType::Compressed => {
            bytes_read += scratch.table.build_decoder(source)?;
        }
        LiteralsSectionType::Treeless => {
            if scratch.table.max_num_bits == 0 {
                return Err("Uninitialized Huffman table for treeless literals".to_string());
            }
        }
        _ => {}
    }

    let source = &source[bytes_read as usize..];

    if num_streams == 4 {
        if source.len() < 6 {
            return Err(format!(
                "Missing bytes for jump header: have {}",
                source.len()
            ));
        }
        let jump1 = source[0] as usize + ((source[1] as usize) << 8);
        let jump2 = jump1 + source[2] as usize + ((source[3] as usize) << 8);
        let jump3 = jump2 + source[4] as usize + ((source[5] as usize) << 8);
        bytes_read += 6;
        let source = &source[6..];

        if source.len() < jump3 {
            return Err(format!(
                "Missing bytes for literals: have {}, need {}",
                source.len(),
                jump3
            ));
        }

        let stream1 = &source[..jump1];
        let stream2 = &source[jump1..jump2];
        let stream3 = &source[jump2..jump3];
        let stream4 = &source[jump3..];

        for stream in &[stream1, stream2, stream3, stream4] {
            let mut decoder = HuffmanDecoder::new(&scratch.table);
            let mut br = BitReaderReversed::new(stream);
            let mut skipped_bits = 0;
            loop {
                let val = br.get_bits(1);
                skipped_bits += 1;
                if val == 1 || skipped_bits > 8 {
                    break;
                }
            }
            if skipped_bits > 8 {
                return Err(format!("Extra padding: {} bits skipped", skipped_bits));
            }
            decoder.init_state(&mut br);

            while br.bits_remaining() > -(scratch.table.max_num_bits as isize) {
                target.push(decoder.decode_symbol());
                decoder.next_state(&mut br);
            }
            if br.bits_remaining() != -(scratch.table.max_num_bits as isize) {
                return Err(format!(
                    "Bitstream read mismatch: {} vs expected {}",
                    br.bits_remaining(),
                    -(scratch.table.max_num_bits as isize)
                ));
            }
        }

        bytes_read += source.len() as u32;
    } else {
        assert!(num_streams == 1);
        let mut decoder = HuffmanDecoder::new(&scratch.table);
        let mut br = BitReaderReversed::new(source);
        let mut skipped_bits = 0;
        loop {
            let val = br.get_bits(1);
            skipped_bits += 1;
            if val == 1 || skipped_bits > 8 {
                break;
            }
        }
        if skipped_bits > 8 {
            return Err(format!("Extra padding: {} bits skipped", skipped_bits));
        }
        decoder.init_state(&mut br);
        while br.bits_remaining() > -(scratch.table.max_num_bits as isize) {
            target.push(decoder.decode_symbol());
            decoder.next_state(&mut br);
        }
        bytes_read += source.len() as u32;
    }

    if target.len() != section.regenerated_size as usize {
        return Err(format!(
            "Decoded literal count mismatch: {} vs expected {}",
            target.len(),
            section.regenerated_size
        ));
    }

    Ok(bytes_read)
}

// ============================================================
// Sequence section decoder
// ============================================================

fn decode_sequences(
    section: &SequencesHeader,
    source: &[u8],
    scratch: &mut FSEScratch,
    target: &mut Vec<Sequence>,
) -> Result<(), String> {
    let bytes_read = maybe_update_fse_tables(section, source, scratch)?;
    let bit_stream = &source[bytes_read..];

    let mut br = BitReaderReversed::new(bit_stream);

    let mut skipped_bits = 0;
    loop {
        let val = br.get_bits(1);
        skipped_bits += 1;
        if val == 1 || skipped_bits > 8 {
            break;
        }
    }
    if skipped_bits > 8 {
        return Err(format!("Extra padding: {} bits skipped", skipped_bits));
    }

    if scratch.ll_rle.is_some() || scratch.ml_rle.is_some() || scratch.of_rle.is_some() {
        decode_sequences_with_rle(section, &mut br, scratch, target)
    } else {
        decode_sequences_without_rle(section, &mut br, scratch, target)
    }
}

fn decode_sequences_with_rle(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    scratch: &FSEScratch,
    target: &mut Vec<Sequence>,
) -> Result<(), String> {
    let mut ll_dec = FSEDecoder::new(&scratch.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&scratch.match_lengths);
    let mut of_dec = FSEDecoder::new(&scratch.offsets);

    if scratch.ll_rle.is_none() {
        ll_dec.init_state(br)?;
    }
    if scratch.of_rle.is_none() {
        of_dec.init_state(br)?;
    }
    if scratch.ml_rle.is_none() {
        ml_dec.init_state(br)?;
    }

    target.clear();
    target.reserve(section.num_sequences as usize);

    for _seq_idx in 0..section.num_sequences {
        let ll_code = scratch.ll_rle.unwrap_or_else(|| ll_dec.decode_symbol());
        let ml_code = scratch.ml_rle.unwrap_or_else(|| ml_dec.decode_symbol());
        let of_code = scratch.of_rle.unwrap_or_else(|| of_dec.decode_symbol());

        let (ll_value, ll_num_bits) = lookup_ll_code(ll_code)?;
        let (ml_value, ml_num_bits) = lookup_ml_code(ml_code)?;

        if of_code > MAX_OFFSET_CODE {
            return Err(format!("Unsupported offset code: {}", of_code));
        }

        let (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_num_bits, ll_num_bits);
        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err("Zero offset".to_string());
        }

        target.push(Sequence {
            ll: ll_value + ll_add as u32,
            ml: ml_value + ml_add as u32,
            of: offset,
        });

        if target.len() < section.num_sequences as usize {
            if scratch.ll_rle.is_none() {
                ll_dec.update_state(br);
            }
            if scratch.ml_rle.is_none() {
                ml_dec.update_state(br);
            }
            if scratch.of_rle.is_none() {
                of_dec.update_state(br);
            }
        }

        if br.bits_remaining() < 0 {
            return Err("Not enough bytes for number of sequences".to_string());
        }
    }

    if br.bits_remaining() > 0 {
        Err(format!(
            "Extra bits remaining: {}",
            br.bits_remaining()
        ))
    } else {
        Ok(())
    }
}

fn decode_sequences_without_rle(
    section: &SequencesHeader,
    br: &mut BitReaderReversed<'_>,
    scratch: &FSEScratch,
    target: &mut Vec<Sequence>,
) -> Result<(), String> {
    let mut ll_dec = FSEDecoder::new(&scratch.literal_lengths);
    let mut ml_dec = FSEDecoder::new(&scratch.match_lengths);
    let mut of_dec = FSEDecoder::new(&scratch.offsets);

    ll_dec.init_state(br)?;
    of_dec.init_state(br)?;
    ml_dec.init_state(br)?;

    target.clear();
    target.reserve(section.num_sequences as usize);

    for _seq_idx in 0..section.num_sequences {
        let ll_code = ll_dec.decode_symbol();
        let ml_code = ml_dec.decode_symbol();
        let of_code = of_dec.decode_symbol();

        let (ll_value, ll_num_bits) = lookup_ll_code(ll_code)?;
        let (ml_value, ml_num_bits) = lookup_ml_code(ml_code)?;

        if of_code > MAX_OFFSET_CODE {
            return Err(format!("Unsupported offset code: {}", of_code));
        }

        let (obits, ml_add, ll_add) = br.get_bits_triple(of_code, ml_num_bits, ll_num_bits);
        let offset = obits as u32 + (1u32 << of_code);

        if offset == 0 {
            return Err("Zero offset".to_string());
        }

        target.push(Sequence {
            ll: ll_value + ll_add as u32,
            ml: ml_value + ml_add as u32,
            of: offset,
        });

        if target.len() < section.num_sequences as usize {
            ll_dec.update_state(br);
            ml_dec.update_state(br);
            of_dec.update_state(br);
        }

        if br.bits_remaining() < 0 {
            return Err("Not enough bytes for number of sequences".to_string());
        }
    }

    if br.bits_remaining() > 0 {
        Err(format!(
            "Extra bits remaining: {}",
            br.bits_remaining()
        ))
    } else {
        Ok(())
    }
}

fn lookup_ll_code(code: u8) -> Result<(u32, u8), String> {
    let result = match code {
        0..=15 => (u32::from(code), 0),
        16 => (16, 1),
        17 => (18, 1),
        18 => (20, 1),
        19 => (22, 1),
        20 => (24, 2),
        21 => (28, 2),
        22 => (32, 3),
        23 => (40, 3),
        24 => (48, 4),
        25 => (64, 6),
        26 => (128, 7),
        27 => (256, 8),
        28 => (512, 9),
        29 => (1024, 10),
        30 => (2048, 11),
        31 => (4096, 12),
        32 => (8192, 13),
        33 => (16384, 14),
        34 => (32768, 15),
        35 => (65536, 16),
        _ => return Err(format!("Illegal literal length code: {}", code)),
    };
    Ok(result)
}

fn lookup_ml_code(code: u8) -> Result<(u32, u8), String> {
    let result = match code {
        0..=31 => (u32::from(code) + 3, 0),
        32 => (35, 1),
        33 => (37, 1),
        34 => (39, 1),
        35 => (41, 1),
        36 => (43, 2),
        37 => (47, 2),
        38 => (51, 3),
        39 => (59, 3),
        40 => (67, 4),
        41 => (83, 4),
        42 => (99, 5),
        43 => (131, 7),
        44 => (259, 8),
        45 => (515, 9),
        46 => (1027, 10),
        47 => (2051, 11),
        48 => (4099, 12),
        49 => (8195, 13),
        50 => (16387, 14),
        51 => (32771, 15),
        52 => (65539, 16),
        _ => return Err(format!("Illegal match length code: {}", code)),
    };
    Ok(result)
}

fn maybe_update_fse_tables(
    section: &SequencesHeader,
    source: &[u8],
    scratch: &mut FSEScratch,
) -> Result<usize, String> {
    let modes = section
        .modes
        .ok_or_else(|| "Missing compression mode".to_string())?;

    let mut bytes_read = 0;

    match modes.ll_mode() {
        ModeType::FSECompressed => {
            let bytes = scratch.literal_lengths.build_decoder(source, LL_MAX_LOG)?;
            bytes_read += bytes;
            scratch.ll_rle = None;
        }
        ModeType::RLE => {
            if source.is_empty() {
                return Err("Missing byte for RLE LL table".to_string());
            }
            bytes_read += 1;
            if source[0] > MAX_LITERAL_LENGTH_CODE {
                return Err(format!("RLE LL code {} exceeds max", source[0]));
            }
            scratch.ll_rle = Some(source[0]);
        }
        ModeType::Predefined => {
            scratch.literal_lengths.build_from_probabilities(
                LL_DEFAULT_ACC_LOG,
                &LITERALS_LENGTH_DEFAULT_DISTRIBUTION,
            )?;
            scratch.ll_rle = None;
        }
        ModeType::Repeat => { /* Nothing to do */ }
    };

    let of_source = &source[bytes_read..];

    match modes.of_mode() {
        ModeType::FSECompressed => {
            let bytes = scratch.offsets.build_decoder(of_source, OF_MAX_LOG)?;
            bytes_read += bytes;
            scratch.of_rle = None;
        }
        ModeType::RLE => {
            if of_source.is_empty() {
                return Err("Missing byte for RLE OF table".to_string());
            }
            bytes_read += 1;
            if of_source[0] > MAX_OFFSET_CODE {
                return Err(format!("RLE OF code {} exceeds max", of_source[0]));
            }
            scratch.of_rle = Some(of_source[0]);
        }
        ModeType::Predefined => {
            scratch.offsets.build_from_probabilities(
                OF_DEFAULT_ACC_LOG,
                &OFFSET_DEFAULT_DISTRIBUTION,
            )?;
            scratch.of_rle = None;
        }
        ModeType::Repeat => { /* Nothing to do */ }
    };

    let ml_source = &source[bytes_read..];

    match modes.ml_mode() {
        ModeType::FSECompressed => {
            let bytes = scratch.match_lengths.build_decoder(ml_source, ML_MAX_LOG)?;
            bytes_read += bytes;
            scratch.ml_rle = None;
        }
        ModeType::RLE => {
            if ml_source.is_empty() {
                return Err("Missing byte for RLE ML table".to_string());
            }
            bytes_read += 1;
            if ml_source[0] > MAX_MATCH_LENGTH_CODE {
                return Err(format!("RLE ML code {} exceeds max", ml_source[0]));
            }
            scratch.ml_rle = Some(ml_source[0]);
        }
        ModeType::Predefined => {
            scratch.match_lengths.build_from_probabilities(
                ML_DEFAULT_ACC_LOG,
                &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
            )?;
            scratch.ml_rle = None;
        }
        ModeType::Repeat => { /* Nothing to do */ }
    };

    Ok(bytes_read)
}

// ============================================================
// Sequence execution
// ============================================================

fn execute_sequences(scratch: &mut DecoderScratch) -> Result<(), String> {
    let mut literals_copy_counter = 0;
    let old_buffer_size = scratch.buffer.len();
    let mut seq_sum = 0u32;

    for idx in 0..scratch.sequences.len() {
        let seq = scratch.sequences[idx];

        if seq.ll > 0 {
            let high = literals_copy_counter + seq.ll as usize;
            if high > scratch.literals_buffer.len() {
                return Err(format!(
                    "Not enough bytes for sequence: wanted {}, have {}",
                    high,
                    scratch.literals_buffer.len()
                ));
            }
            let literals = &scratch.literals_buffer[literals_copy_counter..high];
            literals_copy_counter += seq.ll as usize;
            scratch.buffer.push(literals);
        }

        let actual_offset = do_offset_history(seq.of, seq.ll, &mut scratch.offset_hist);
        if actual_offset == 0 {
            return Err("Zero offset in sequence execution".to_string());
        }
        if seq.ml > 0 {
            scratch
                .buffer
                .repeat(actual_offset as usize, seq.ml as usize)?;
        }

        seq_sum += seq.ml;
        seq_sum += seq.ll;
    }

    if literals_copy_counter < scratch.literals_buffer.len() {
        let rest_literals = &scratch.literals_buffer[literals_copy_counter..];
        scratch.buffer.push(rest_literals);
        seq_sum += rest_literals.len() as u32;
    }

    let diff = scratch.buffer.len() - old_buffer_size;
    assert!(
        seq_sum as usize == diff,
        "Seq_sum: {} is different from the difference in buffersize: {}",
        seq_sum,
        diff
    );
    Ok(())
}

fn do_offset_history(offset_value: u32, lit_len: u32, scratch: &mut [u32; 3]) -> u32 {
    let actual_offset = if lit_len > 0 {
        match offset_value {
            1..=3 => scratch[offset_value as usize - 1],
            _ => offset_value - 3,
        }
    } else {
        match offset_value {
            1..=2 => scratch[offset_value as usize],
            3 => scratch[0].wrapping_sub(1),
            _ => offset_value - 3,
        }
    };

    if lit_len > 0 {
        match offset_value {
            1 => { /* nothing */ }
            2 => {
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
            _ => {
                scratch[2] = scratch[1];
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
        }
    } else {
        match offset_value {
            1 => {
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
            _ => {
                scratch[2] = scratch[1];
                scratch[1] = scratch[0];
                scratch[0] = actual_offset;
            }
        }
    }

    actual_offset
}

// ============================================================
// Block decoder
// ============================================================

fn decode_block_content(
    header: &BlockHeader,
    workspace: &mut DecoderScratch,
    source: &mut dyn std::io::Read,
) -> Result<u64, String> {
    match header.block_type {
        BlockType::RLE => {
            const BATCH_SIZE: usize = 512;
            let mut buf = [0u8; BATCH_SIZE];
            let full_reads = header.decompressed_size / BATCH_SIZE as u32;
            let single_read_size = header.decompressed_size % BATCH_SIZE as u32;

            source
                .read_exact(&mut buf[0..1])
                .map_err(|e| format!("Error reading RLE byte: {}", e))?;

            for i in 1..BATCH_SIZE {
                buf[i] = buf[0];
            }

            for _ in 0..full_reads {
                workspace.buffer.push(&buf[..]);
            }
            let smaller = &buf[..single_read_size as usize];
            workspace.buffer.push(smaller);

            Ok(1)
        }
        BlockType::Raw => {
            const BATCH_SIZE: usize = 128 * 1024;
            let mut buf = [0u8; BATCH_SIZE];
            let full_reads = header.decompressed_size / BATCH_SIZE as u32;
            let single_read_size = header.decompressed_size % BATCH_SIZE as u32;

            for _ in 0..full_reads {
                source
                    .read_exact(&mut buf[..])
                    .map_err(|e| format!("Error reading raw block: {}", e))?;
                workspace.buffer.push(&buf[..]);
            }

            let smaller = &mut buf[..single_read_size as usize];
            source
                .read_exact(smaller)
                .map_err(|e| format!("Error reading raw block: {}", e))?;
            workspace.buffer.push(smaller);

            Ok(u64::from(header.decompressed_size))
        }
        BlockType::Reserved => {
            Err("Reserved block type encountered".to_string())
        }
        BlockType::Compressed => {
            decompress_block(header, workspace, source)?;
            Ok(u64::from(header.content_size))
        }
    }
}

fn decompress_block(
    header: &BlockHeader,
    workspace: &mut DecoderScratch,
    source: &mut dyn std::io::Read,
) -> Result<(), String> {
    workspace
        .block_content_buffer
        .resize(header.content_size as usize, 0);

    source
        .read_exact(workspace.block_content_buffer.as_mut_slice())
        .map_err(|e| format!("Error reading compressed block: {}", e))?;
    let raw = workspace.block_content_buffer.as_slice();

    let mut section = LiteralsSection::new();
    let bytes_in_literals_header = section.parse_from_header(raw)?;
    let raw = &raw[bytes_in_literals_header as usize..];

    let upper_limit_for_literals = match section.compressed_size {
        Some(x) => x as usize,
        None => match section.ls_type {
            LiteralsSectionType::RLE => 1,
            LiteralsSectionType::Raw => section.regenerated_size as usize,
            _ => return Err("Bug: unexpected literals section type".to_string()),
        },
    };

    if raw.len() < upper_limit_for_literals {
        return Err(format!(
            "Malformed section header: expected {} bytes, have {}",
            upper_limit_for_literals,
            raw.len()
        ));
    }

    let raw_literals = &raw[..upper_limit_for_literals];

    workspace.literals_buffer.clear();
    let bytes_used_in_literals_section = decode_literals(
        &section,
        &mut workspace.huf,
        raw_literals,
        &mut workspace.literals_buffer,
    )?;
    assert!(
        section.regenerated_size == workspace.literals_buffer.len() as u32,
        "Wrong number of literals: {}, Should have been: {}",
        workspace.literals_buffer.len(),
        section.regenerated_size
    );
    assert!(bytes_used_in_literals_section == upper_limit_for_literals as u32);

    let raw = &raw[upper_limit_for_literals..];

    let mut seq_section = SequencesHeader::new();
    let bytes_in_sequence_header = seq_section.parse_from_header(raw)?;
    let raw = &raw[bytes_in_sequence_header as usize..];

    assert!(
        u32::from(bytes_in_literals_header)
            + bytes_used_in_literals_section
            + u32::from(bytes_in_sequence_header)
            + raw.len() as u32
            == header.content_size
    );

    if seq_section.num_sequences != 0 {
        decode_sequences(
            &seq_section,
            raw,
            &mut workspace.fse,
            &mut workspace.sequences,
        )?;
        execute_sequences(workspace)?;
    } else {
        if !raw.is_empty() {
            return Err(format!(
                "Extra bits remaining: {} bits",
                raw.len() as isize * 8
            ));
        }
        workspace.buffer.push(&workspace.literals_buffer);
        workspace.sequences.clear();
    }

    Ok(())
}

// ============================================================
// Frame Decoder (top-level)
// ============================================================

struct FrameDecoder {
    scratch: Option<DecoderScratch>,
    frame_header: Option<FrameHeader>,
    frame_finished: bool,
}

impl FrameDecoder {
    fn new() -> FrameDecoder {
        FrameDecoder {
            scratch: None,
            frame_header: None,
            frame_finished: false,
        }
    }

    fn reset(&mut self, source: &mut dyn std::io::Read) -> Result<(), FrameDecoderError> {
        let (frame_header, _header_size) = read_frame_header(source)?;
        let window_size = frame_header
            .window_size()
            .map_err(FrameDecoderError::new)?;

        if window_size > MAXIMUM_ALLOWED_WINDOW_SIZE {
            return Err(FrameDecoderError::new(format!(
                "Window size {} exceeds maximum allowed {}",
                window_size, MAXIMUM_ALLOWED_WINDOW_SIZE
            )));
        }

        match &mut self.scratch {
            Some(s) => s.reset(window_size as usize),
            None => {
                self.scratch = Some(DecoderScratch::new(window_size as usize));
            }
        }

        self.frame_header = Some(frame_header);
        self.frame_finished = false;
        Ok(())
    }

    fn decode_all_blocks(&mut self, source: &mut dyn std::io::Read) -> Result<(), String> {
        let scratch = self
            .scratch
            .as_mut()
            .ok_or_else(|| "Decoder not initialized".to_string())?;

        loop {
            let (block_header, _block_header_size) = read_block_header(source)?;

            decode_block_content(&block_header, scratch, source)?;

            if block_header.last_block {
                self.frame_finished = true;

                // Read and discard checksum if present
                if let Some(ref fh) = self.frame_header {
                    if fh.descriptor.content_checksum_flag() {
                        let mut chksum = [0u8; 4];
                        source
                            .read_exact(&mut chksum)
                            .map_err(|e| format!("Error reading checksum: {}", e))?;
                        // We skip checksum verification in this simplified decoder
                    }
                }
                break;
            }
        }

        Ok(())
    }

    fn collect(&mut self) -> Option<Vec<u8>> {
        self.scratch.as_mut().map(|s| s.buffer.drain())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let result = decompress(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_bad_magic() {
        let result = decompress(&[0, 0, 0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_raw() {
        // A minimal zstd frame: magic + frame header + single raw block
        // This test builds a valid frame with a raw block containing "hello"
        let data = b"hello";
        let mut frame = Vec::new();
        // Magic number
        frame.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        // Frame descriptor: single_segment=1, no checksum, no dict, fcs_flag=0
        // So FCS field = 1 byte
        frame.push(0x20); // single_segment_flag set
        // FCS = 5 (length of "hello")
        frame.push(5);
        // Block header: last_block=1, type=raw(0), size=5
        // Encoding: bit0=last(1), bit1-2=type(0), bit3-20=size(5)
        let bh = 1u32 | (0u32 << 1) | (5u32 << 3);
        frame.push((bh & 0xFF) as u8);
        frame.push(((bh >> 8) & 0xFF) as u8);
        frame.push(((bh >> 16) & 0xFF) as u8);
        // Block content
        frame.extend_from_slice(data);

        let result = decompress(&frame).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roundtrip_rle() {
        // Frame with an RLE block: 10 copies of byte 0x42
        let mut frame = Vec::new();
        frame.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        frame.push(0x20); // single_segment_flag set
        frame.push(10); // FCS = 10
        // Block header: last_block=1, type=RLE(1), size=10
        let bh = 1u32 | (1u32 << 1) | (10u32 << 3);
        frame.push((bh & 0xFF) as u8);
        frame.push(((bh >> 8) & 0xFF) as u8);
        frame.push(((bh >> 16) & 0xFF) as u8);
        // Single RLE byte
        frame.push(0x42);

        let result = decompress(&frame).unwrap();
        assert_eq!(result, vec![0x42; 10]);
    }

    #[test]
    fn test_roundtrip_with_compressor() {
        // Use the crate's own compressor to produce a valid zstd frame,
        // then decompress with our decoder.
        let data = b"Hello, world! This is a test of the zstd compression and decompression round-trip. \
                      The quick brown fox jumps over the lazy dog. \
                      AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA \
                      BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB \
                      Hello, world! This is a test of the zstd compression and decompression round-trip.";
        let compressed = crate::zstd::compress::compress_to_vec(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_roundtrip_larger() {
        // Test with larger data that triggers compressed blocks.
        let mut data = Vec::with_capacity(16384);
        for i in 0..16384u32 {
            data.push((i % 251) as u8);
        }
        let compressed = crate::zstd::compress::compress_to_vec(&data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
}
