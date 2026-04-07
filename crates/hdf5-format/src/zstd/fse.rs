//! FSE (Finite State Entropy) encoder.
//! Ported from zstd C source: lib/common/fse.h, lib/compress/fse_compress.c.

use super::bitstream::BackwardBitWriter;

/// Per-symbol compression transform (matches C FSE_symbolCompressionTransform).
#[derive(Clone, Copy, Default)]
pub struct SymbolTT {
    pub delta_find_state: i32,
    pub delta_nb_bits: u32,
}

/// Compiled FSE compression table.
pub struct FseCTable {
    pub table_log: u32,
    pub state_table: Vec<u16>,
    pub symbol_tt: Vec<SymbolTT>,
    pub max_symbol: usize,
}

impl FseCTable {
    /// Build an FSE compression table from normalized counts.
    /// Ported from FSE_buildCTable_wksp() in fse_compress.c.
    pub fn build(norm: &[i16], max_symbol: usize, table_log: u32) -> Self {
        let table_size = 1u32 << table_log;
        let table_mask = table_size - 1;

        // 1. Build cumulative counts and place low-probability symbols.
        let mut cumul = vec![0u16; max_symbol + 2];
        let mut high_threshold = table_size - 1;
        let mut table_symbol = vec![0u8; table_size as usize];

        for s in 0..=max_symbol {
            if norm[s] == -1 {
                cumul[s + 1] = cumul[s] + 1;
                table_symbol[high_threshold as usize] = s as u8;
                high_threshold = high_threshold.wrapping_sub(1);
            } else {
                cumul[s + 1] = cumul[s] + norm[s] as u16;
            }
        }
        cumul[max_symbol + 1] = (table_size + 1) as u16;

        // 2. Spread symbols into the table using the exact C formula.
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mut pos = 0u32;
        for s in 0..=max_symbol {
            let count = if norm[s] <= 0 { 0 } else { norm[s] as u32 };
            for _ in 0..count {
                table_symbol[pos as usize] = s as u8;
                pos = (pos + step) & table_mask;
                while pos > high_threshold {
                    pos = (pos + step) & table_mask;
                }
            }
        }
        debug_assert_eq!(pos, 0);

        // 3. Build state transition table sorted by symbol order.
        let mut state_table = vec![0u16; table_size as usize];
        for u in 0..table_size {
            let s = table_symbol[u as usize] as usize;
            let idx = cumul[s] as usize;
            state_table[idx] = (table_size + u) as u16;
            cumul[s] += 1;
        }

        // 4. Build per-symbol compression transforms.
        let mut symbol_tt = vec![SymbolTT::default(); max_symbol + 1];
        let mut total = 0u32;
        for s in 0..=max_symbol {
            match norm[s] {
                0 => {
                    symbol_tt[s].delta_nb_bits = ((table_log + 1) << 16) - table_size;
                }
                -1 | 1 => {
                    symbol_tt[s].delta_nb_bits = (table_log << 16) - table_size;
                    symbol_tt[s].delta_find_state = total as i32 - 1;
                    total += 1;
                }
                n => {
                    let n = n as u32;
                    let max_bits_out = table_log - highest_bit(n - 1);
                    let min_state_plus = n << max_bits_out;
                    symbol_tt[s].delta_nb_bits = (max_bits_out << 16).wrapping_sub(min_state_plus);
                    symbol_tt[s].delta_find_state = total as i32 - n as i32;
                    total += n;
                }
            }
        }

        Self { table_log, state_table, symbol_tt, max_symbol }
    }

    /// Initialize FSE state for the first symbol (FSE_initCState2).
    pub fn init_state(&self, symbol: usize) -> u32 {
        let stt = &self.symbol_tt[symbol];
        let nb_bits = ((stt.delta_nb_bits as u64 + (1 << 15)) >> 16) as u32;
        let base_val = (nb_bits << 16).wrapping_sub(stt.delta_nb_bits);
        self.state_table[((base_val >> nb_bits) as i32 + stt.delta_find_state) as usize] as u32
    }

    /// Encode a symbol: output bits from current state, then transition.
    /// Returns (bits_to_output, nb_bits, new_state).
    pub fn encode_symbol(&self, state: u32, symbol: usize) -> (u32, u32, u32) {
        let stt = &self.symbol_tt[symbol];
        let nb_bits = (state.wrapping_add(stt.delta_nb_bits)) >> 16;
        let bits_out = state & ((1 << nb_bits) - 1);
        let new_state = self.state_table[((state >> nb_bits) as i32 + stt.delta_find_state) as usize] as u32;
        (bits_out, nb_bits, new_state)
    }
}

fn highest_bit(v: u32) -> u32 {
    if v == 0 { return 0; }
    31 - v.leading_zeros()
}

#[allow(clippy::too_many_arguments)]
/// Encode sequences using predefined FSE tables.
/// Exact port of ZSTD_encodeSequences_body from zstd_compress_sequences.c.
pub fn encode_sequences(
    ll_table: &FseCTable,
    off_table: &FseCTable,
    ml_table: &FseCTable,
    ll_codes: &[u8],
    off_codes: &[u8],
    ml_codes: &[u8],
    ll_values: &[u32],  // literal length values (for extra bits)
    ml_values: &[u32],  // match length - MINMATCH values (for extra bits)
    off_values: &[u32], // offset values (for extra bits)
) -> Vec<u8> {
    use super::constants::*;

    let nb_seq = ll_codes.len();
    if nb_seq == 0 { return vec![]; }

    let mut bw = BackwardBitWriter::new();

    // Initialize states from the last sequence (first in encoding order)
    let last = nb_seq - 1;
    let mut state_ll = ll_table.init_state(ll_codes[last] as usize);
    let mut state_off = off_table.init_state(off_codes[last] as usize);
    let mut state_ml = ml_table.init_state(ml_codes[last] as usize);

    // Encode extra bits for the last sequence
    let ll_bits_n = LL_BITS[ll_codes[last] as usize] as u32;
    bw.add_bits(ll_values[last] as u64, ll_bits_n);
    if ll_bits_n > 0 { bw.flush_bits(); }

    let ml_bits_n = ML_BITS[ml_codes[last] as usize] as u32;
    bw.add_bits(ml_values[last] as u64, ml_bits_n);
    if ml_bits_n > 0 { bw.flush_bits(); }

    let of_bits_n = off_codes[last] as u32;
    bw.add_bits(off_values[last] as u64, of_bits_n);
    bw.flush_bits();

    // Encode remaining sequences in reverse order
    if nb_seq >= 2 {
        for n in (0..last).rev() {
            let llc = ll_codes[n] as usize;
            let ofc = off_codes[n] as usize;
            let mlc = ml_codes[n] as usize;

            // FSE encode: OFF, ML, LL (order matters!)
            let (bits, nb, new_state) = off_table.encode_symbol(state_off, ofc);
            bw.add_bits(bits as u64, nb);
            state_off = new_state;

            let (bits, nb, new_state) = ml_table.encode_symbol(state_ml, mlc);
            bw.add_bits(bits as u64, nb);
            state_ml = new_state;

            let (bits, nb, new_state) = ll_table.encode_symbol(state_ll, llc);
            bw.add_bits(bits as u64, nb);
            state_ll = new_state;

            bw.flush_bits();

            // Extra bits: LL, ML, OFF
            let ll_eb = LL_BITS[llc] as u32;
            bw.add_bits(ll_values[n] as u64, ll_eb);

            let ml_eb = ML_BITS[mlc] as u32;
            bw.add_bits(ml_values[n] as u64, ml_eb);

            let of_eb = ofc as u32;
            bw.add_bits(off_values[n] as u64, of_eb);
            bw.flush_bits();
        }
    }

    // Flush final states
    bw.add_bits(state_ml as u64, ml_table.table_log);
    bw.flush_bits();
    bw.add_bits(state_off as u64, off_table.table_log);
    bw.flush_bits();
    bw.add_bits(state_ll as u64, ll_table.table_log);
    bw.flush_bits();

    bw.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zstd::constants::*;

    #[test]
    fn build_ll_default_table() {
        let table = FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        assert_eq!(table.table_log, 6);
        assert_eq!(table.state_table.len(), 64);
    }

    #[test]
    fn build_ml_default_table() {
        let table = FseCTable::build(&ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);
        assert_eq!(table.table_log, 6);
        assert_eq!(table.state_table.len(), 64);
    }

    #[test]
    fn init_state_in_range() {
        let table = FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        // Encoder states are stored in the [table_size, 2 * table_size) range.
        let state = table.init_state(0);
        let table_size = 1u32 << table.table_log;
        assert!((table_size..(table_size * 2)).contains(&state));
    }
}
