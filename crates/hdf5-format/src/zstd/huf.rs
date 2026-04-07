//! Huffman encoder for zstd literal compression.
//!
//! Ported from zstd C source (lib/compress/huf_compress.c).
//! Implements canonical Huffman coding for the literals section.

/// Huffman code entry: (code_bits, code_length).
#[derive(Clone, Copy, Default)]
pub struct HufCode {
    pub bits: u32,
    pub nbits: u8,
}

/// Build canonical Huffman codes from symbol frequencies.
///
/// Returns (codes[256], max_bits, tree_description_bytes).
/// `tree_description_bytes` is the serialized Huffman tree for the block header.
pub fn build_huffman_table(counts: &[u32; 256]) -> Option<(Vec<HufCode>, u8, Vec<u8>)> {
    let mut symbols: Vec<(u32, u8)> = counts.iter().enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(s, &c)| (c, s as u8))
        .collect();

    if symbols.len() <= 1 {
        return None; // Use raw/RLE instead
    }

    // Sort by frequency (ascending), then by symbol
    symbols.sort();

    // Build Huffman tree using package-merge or simple tree
    let n = symbols.len();
    let max_bits = std::cmp::min(11, highest_bit_u32(n as u32) as u8 + 2);

    // Simple length-limited Huffman: assign bit lengths
    let mut lengths = vec![0u8; 256];
    assign_lengths(&symbols, max_bits, &mut lengths);

    // Build canonical codes from lengths
    let mut codes = vec![HufCode::default(); 256];
    let actual_max = *lengths.iter().max().unwrap_or(&0);

    // Count lengths
    let mut bl_count = vec![0u32; (actual_max + 1) as usize];
    for &l in &lengths {
        if l > 0 { bl_count[l as usize] += 1; }
    }

    // Generate start codes for each length
    let mut next_code = vec![0u32; (actual_max + 2) as usize];
    for bits in 1..=actual_max {
        next_code[bits as usize] = (next_code[(bits - 1) as usize] + bl_count[(bits - 1) as usize]) << 1;
    }

    // Assign codes
    for s in 0..256 {
        let l = lengths[s];
        if l > 0 {
            codes[s] = HufCode { bits: next_code[l as usize], nbits: l };
            next_code[l as usize] += 1;
        }
    }

    // Encode tree description (FSE-based weight table or direct)
    let tree_desc = encode_tree_description(&lengths, actual_max);

    Some((codes, actual_max, tree_desc))
}

/// Assign bit lengths using a simple greedy method.
fn assign_lengths(symbols: &[(u32, u8)], max_bits: u8, lengths: &mut [u8]) {
    let n = symbols.len();
    if n == 0 { return; }

    // Simple: assign decreasing lengths based on frequency rank
    // Start from max_bits for least frequent, decrease toward 1 for most frequent
    let total_freq: u64 = symbols.iter().map(|&(f, _)| f as u64).sum();
    if total_freq == 0 { return; }

    // Use a simple heuristic: log2(total/freq) clamped to [1, max_bits]
    for &(freq, sym) in symbols {
        if freq == 0 { continue; }
        let ideal = if total_freq / (freq as u64) <= 1 { 1 }
                    else { std::cmp::min(max_bits as u64, 1 + highest_bit_u64(total_freq / freq as u64)) };
        lengths[sym as usize] = std::cmp::max(1, std::cmp::min(max_bits, ideal as u8));
    }

    // Kraft inequality check and adjustment
    loop {
        let kraft: u64 = (0..256)
            .filter(|&s| lengths[s] > 0)
            .map(|s| 1u64 << (max_bits - lengths[s]))
            .sum();
        let target = 1u64 << max_bits;
        if kraft == target { break; }
        if kraft < target {
            // Under-full: increase lengths of least frequent symbols
            let mut best_s = 0usize;
            let mut best_freq = u32::MAX;
            for s in 0..256 {
                if lengths[s] > 0 && lengths[s] < max_bits {
                    let f = symbols.iter().find(|&&(_, sym)| sym as usize == s).map(|&(f, _)| f).unwrap_or(u32::MAX);
                    if f < best_freq { best_freq = f; best_s = s; }
                }
            }
            if best_freq == u32::MAX { break; }
            lengths[best_s] += 1;
        } else {
            // Over-full: decrease lengths of most frequent symbols
            let mut best_s = 0usize;
            let mut best_freq = 0u32;
            for s in 0..256 {
                if lengths[s] > 1 {
                    let f = symbols.iter().find(|&&(_, sym)| sym as usize == s).map(|&(f, _)| f).unwrap_or(0);
                    if f > best_freq { best_freq = f; best_s = s; }
                }
            }
            if best_freq == 0 { break; }
            lengths[best_s] -= 1;
        }
    }
}

/// Encode tree description as direct representation (weights).
/// Zstd uses weights = max_bits + 1 - length (0 for unused symbols).
fn encode_tree_description(lengths: &[u8], max_bits: u8) -> Vec<u8> {
    // Find the highest used symbol
    let max_sym = (0..256).rev().find(|&s| lengths[s] > 0).unwrap_or(0);

    // Weights
    let weights: Vec<u8> = (0..=max_sym)
        .map(|s| if lengths[s] > 0 { max_bits + 1 - lengths[s] } else { 0 })
        .collect();

    // Direct representation: header byte = (max_sym) (if < 128)
    // Each pair of weights packed into one byte (4 bits each)
    let mut desc = Vec::with_capacity(1 + weights.len().div_ceil(2));
    let packed_size = weights.len().div_ceil(2);
    desc.push(packed_size as u8); // header byte: number of bytes of weights

    for i in (0..weights.len()).step_by(2) {
        let w0 = weights[i];
        let w1 = if i + 1 < weights.len() { weights[i + 1] } else { 0 };
        desc.push((w0 << 4) | w1);
    }

    desc
}

fn highest_bit_u32(v: u32) -> u64 {
    if v == 0 { 0 } else { (31 - v.leading_zeros()) as u64 }
}

fn highest_bit_u64(v: u64) -> u64 {
    if v == 0 { 0 } else { (63 - v.leading_zeros()) as u64 }
}
