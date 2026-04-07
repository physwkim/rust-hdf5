//! Zstandard frame compressor.
//!
//! Produces valid zstd frames decompressible by any standard decoder.
//! Uses greedy hash-based matching (equivalent to zstd level 1).

use super::constants::*;

/// Compress data into a zstd frame.
///
/// `level` controls the compression strategy:
/// - 0: no compression (raw blocks, fastest)
/// - 1-2: greedy matching (fast, hash_log=14)
/// - 3-5: lazy matching (better ratio, hash_log=15)
/// - 6-8: lazy matching + deeper search (hash_log=16)
/// - 9-11: lazy matching + deepest search (hash_log=17)
///
/// Returns a valid zstd frame decompressible by any conformant decoder.
pub fn compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    write_frame_header(&mut out, data.len() as u64);

    if data.is_empty() {
        write_raw_block(&mut out, &[], true);
        return out;
    }

    let blocks: Vec<&[u8]> = data.chunks(ZSTD_BLOCKSIZE_MAX).collect();
    let n_blocks = blocks.len();

    if level <= 0 {
        // Level 0: raw blocks only
        for (i, block) in blocks.iter().enumerate() {
            write_raw_block(&mut out, block, i == n_blocks - 1);
        }
        return out;
    }

    // Level 1+: compress blocks (optionally in parallel)
    let params = MatchParams::from_level(level);

    #[cfg(feature = "parallel")]
    let compressed_blocks: Vec<Option<Vec<u8>>> = {
        use rayon::prelude::*;
        blocks.par_iter()
            .map(|block| compress_block(block, &params))
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let compressed_blocks: Vec<Option<Vec<u8>>> = blocks.iter()
        .map(|block| compress_block(block, &params))
        .collect();

    for (i, compressed) in compressed_blocks.iter().enumerate() {
        let is_last = i == n_blocks - 1;
        match compressed {
            Some(c) if c.len() < blocks[i].len() => {
                write_compressed_block(&mut out, c, is_last);
            }
            _ => write_raw_block(&mut out, blocks[i], is_last),
        }
    }

    out
}

/// Convenience wrapper.
pub fn compress_to_vec(data: &[u8]) -> Vec<u8> {
    compress(data, 1)
}

// =========================================================================
// Frame header
// =========================================================================

fn write_frame_header(out: &mut Vec<u8>, content_size: u64) {
    // Magic number (LE)
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

    // Frame_Header_Descriptor:
    // bit 7-6: Frame_Content_Size_flag (determines FCS field size)
    // bit 5:   Single_Segment_flag (1 = no Window_Descriptor)
    // bit 4:   unused
    // bit 3:   reserved
    // bit 2:   Content_Checksum_flag (0 = no checksum)
    // bit 1-0: Dictionary_ID_flag (0 = no dict)

    let (fcs_flag, fcs_bytes) = if content_size <= 255 {
        (0u8, 1) // 1 byte FCS (but flag 0 means 0 bytes normally...)
    } else if content_size <= 65535 + 256 {
        (1u8, 2) // 2 bytes
    } else if content_size <= u32::MAX as u64 {
        (2u8, 4)
    } else {
        (3u8, 8)
    };

    // For single-segment, FCS_flag=0 means 1 byte (when single_segment=1)
    let single_segment = 1u8; // always single-segment for simplicity
    let descriptor = (fcs_flag << 6) | (single_segment << 5);
    out.push(descriptor);

    // No Window_Descriptor (single_segment = 1)

    // Frame_Content_Size
    match fcs_bytes {
        1 => out.push(content_size as u8),
        2 => out.extend_from_slice(&((content_size - 256) as u16).to_le_bytes()),
        4 => out.extend_from_slice(&(content_size as u32).to_le_bytes()),
        8 => out.extend_from_slice(&content_size.to_le_bytes()),
        _ => {}
    }
}

// =========================================================================
// Block writing
// =========================================================================

fn write_raw_block(out: &mut Vec<u8>, data: &[u8], is_last: bool) {
    let header = (is_last as u32)
        | ((BLOCK_TYPE_RAW as u32) << 1)
        | ((data.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.extend_from_slice(data);
}

fn write_compressed_block(out: &mut Vec<u8>, compressed: &[u8], is_last: bool) {
    let header = (is_last as u32)
        | ((BLOCK_TYPE_COMPRESSED as u32) << 1)
        | ((compressed.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.extend_from_slice(compressed);
}

// =========================================================================
// Block compression (greedy matching + raw literals + predefined FSE)
// =========================================================================

/// A sequence: (literal_length, offset, match_length).
struct Sequence {
    ll: u32,   // literal length
    off: u32,  // offset (1-based, as stored in zstd)
    ml: u32,   // match length (raw, not -3)
}

/// Match finder parameters, derived from compression level.
struct MatchParams {
    hash_log: u32,
    lazy_depth: u32,   // 0=greedy, 1=lazy, 2=lazy2
    search_depth: u32, // hash chain search depth
}

impl MatchParams {
    fn from_level(level: i32) -> Self {
        match level {
            0..=2 => Self { hash_log: 14, lazy_depth: 0, search_depth: 4 },
            3..=5 => Self { hash_log: 15, lazy_depth: 1, search_depth: 16 },
            6..=8 => Self { hash_log: 16, lazy_depth: 1, search_depth: 64 },
            _     => Self { hash_log: 17, lazy_depth: 1, search_depth: 128 },
        }
    }
}

fn compress_block(data: &[u8], params: &MatchParams) -> Option<Vec<u8>> {
    let sequences = find_matches(data, params);

    if sequences.is_empty() {
        return None; // No matches found, use raw block
    }

    // Compute trailing literals
    let last_seq_end = sequences.iter()
        .fold(0u32, |acc, s| acc + s.ll + s.ml);
    let _trailing_lits = data.len() as u32 - last_seq_end;

    // Collect all literals
    let mut literals = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    for seq in &sequences {
        literals.extend_from_slice(&data[pos..pos + seq.ll as usize]);
        pos += seq.ll as usize + seq.ml as usize;
    }
    literals.extend_from_slice(&data[pos..]);

    // Encode block content
    let mut block = Vec::with_capacity(data.len());

    // === Literals section ===
    if literals.len() >= 64 {
        let huf_result = encode_literals_huffman(&literals);
        if !huf_result.is_empty() && huf_result.len() < literals.len() {
            block.extend_from_slice(&huf_result);
        } else {
            encode_literals_raw(&mut block, &literals);
        }
    } else {
        encode_literals_raw(&mut block, &literals);
    }

    // === Sequences section ===
    encode_sequences_section(&mut block, &sequences);

    Some(block)
}

/// Hash chain match finder with configurable lazy depth.
///
/// - `lazy_depth=0`: greedy — take first match (level 1-2)
/// - `lazy_depth=1`: lazy — check next position for better match (level 3+)
fn find_matches(data: &[u8], params: &MatchParams) -> Vec<Sequence> {
    if data.len() < ZSTD_MINMATCH + 1 {
        return vec![];
    }

    let hash_size = 1usize << params.hash_log;
    let hash_mask = (hash_size - 1) as u32;
    let mut hash_table = vec![0u32; hash_size];
    let mut chain = vec![0u32; data.len()];
    let mut sequences = Vec::new();
    let mut anchor = 0usize;
    let mut ip = 0usize;

    while ip + ZSTD_MINMATCH < data.len() {
        // Try to find a match at current position
        let best = find_best_at(data, ip, &hash_table, &chain, hash_mask, params.search_depth);

        if let Some((offset, match_len)) = best {
            let mut final_off = offset;
            let mut final_len = match_len;
            let mut final_ip = ip;

            // Lazy matching: check if next position gives a better match
            if params.lazy_depth >= 1 && ip + 1 + ZSTD_MINMATCH < data.len() {
                insert_hash(&mut hash_table, &mut chain, data, ip, hash_mask);
                if let Some((off2, len2)) = find_best_at(data, ip + 1, &hash_table, &chain, hash_mask, params.search_depth) {
                    if len2 > final_len + 1 {
                        final_off = off2;
                        final_len = len2;
                        final_ip = ip + 1;
                    }
                }
            }

            let ll = (final_ip - anchor) as u32;
            sequences.push(Sequence { ll, off: final_off as u32, ml: final_len as u32 });

            // Insert all positions within the match for future matching
            for p in ip..std::cmp::min(final_ip + final_len, data.len().saturating_sub(ZSTD_MINMATCH)) {
                insert_hash(&mut hash_table, &mut chain, data, p, hash_mask);
            }

            ip = final_ip + final_len;
            anchor = ip;
        } else {
            insert_hash(&mut hash_table, &mut chain, data, ip, hash_mask);
            ip += 1;
        }
    }

    sequences
}

/// Insert position into hash chain.
#[inline]
fn insert_hash(hash_table: &mut [u32], chain: &mut [u32], data: &[u8], pos: usize, mask: u32) {
    if pos + 4 > data.len() { return; }
    let h = hash4(&data[pos..], mask);
    chain[pos] = hash_table[h];
    hash_table[h] = pos as u32;
}

/// Find the best match at `pos` by walking the hash chain.
fn find_best_at(
    data: &[u8], pos: usize,
    hash_table: &[u32], chain: &[u32],
    mask: u32, max_depth: u32,
) -> Option<(usize, usize)> {
    if pos + ZSTD_MINMATCH > data.len() { return None; }
    let h = hash4(&data[pos..], mask);
    let mut candidate = hash_table[h] as usize;
    let mut best_len = ZSTD_MINMATCH - 1;
    let mut best_off = 0;

    for _ in 0..max_depth {
        if candidate >= pos || pos - candidate > (1 << 24) { break; }
        if candidate + ZSTD_MINMATCH > data.len() { break; }

        // Quick 4-byte check
        if data[candidate..candidate + 4] == data[pos..pos + 4] {
            let ml = common_prefix_len(&data[candidate..], &data[pos..]);
            if ml > best_len {
                best_len = ml;
                best_off = pos - candidate;
            }
        }

        let next = chain[candidate] as usize;
        if next >= candidate { break; }
        candidate = next;
    }

    if best_len >= ZSTD_MINMATCH { Some((best_off, best_len)) } else { None }
}

/// Fast common prefix length using 8-byte chunks.
#[inline]
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let max = std::cmp::min(a.len(), b.len());
    let mut i = 0;
    while i + 8 <= max {
        let va = u64::from_le_bytes(a[i..i+8].try_into().unwrap());
        let vb = u64::from_le_bytes(b[i..i+8].try_into().unwrap());
        if va != vb {
            return i + ((va ^ vb).trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < max && a[i] == b[i] { i += 1; }
    i
}

/// 4-byte multiplicative hash, result masked to table size.
#[inline]
fn hash4(data: &[u8], mask: u32) -> usize {
    let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    (v.wrapping_mul(0x9E3779B1) as usize) & (mask as usize)
}

// =========================================================================
// Literals section encoding (Huffman compressed)
// =========================================================================

/// Encode literals using Huffman compression.
/// Returns the complete literals section (header + tree + compressed streams).
fn encode_literals_huffman(literals: &[u8]) -> Vec<u8> {
    // 1. Count symbol frequencies
    let mut counts = [0u32; 256];
    let mut max_sym = 0u8;
    for &b in literals {
        counts[b as usize] += 1;
        if b > max_sym { max_sym = b; }
    }
    let n_symbols = counts[..=max_sym as usize].iter().filter(|&&c| c > 0).count();
    if n_symbols < 2 {
        // Can't Huffman-encode with < 2 symbols
        let mut out = Vec::new();
        encode_literals_raw(&mut out, literals);
        return out;
    }

    // 2. Build Huffman codes (canonical, max 11 bits per spec)
    let (codes, max_bits) = build_huffman_codes(&counts, max_sym as usize);

    // 3. Encode the Huffman tree description (weights)
    let tree_desc = encode_huffman_tree(&codes, max_bits, max_sym as usize);

    // 4. Encode literal data using single stream (< 1KB) or 4 streams (>= 1KB)
    let use_4_streams = literals.len() >= 1024;
    let compressed_streams = if use_4_streams {
        encode_huf_4streams(literals, &codes)
    } else {
        encode_huf_1stream(literals, &codes)
    };

    // 5. Build the header
    if tree_desc.is_empty() {
        // Huffman tree encoding failed (too many symbols), fall back to raw
        return Vec::new();
    }
    let regen_size = literals.len();
    let compressed_size = tree_desc.len() + compressed_streams.len();

    let mut out = Vec::with_capacity(5 + compressed_size);

    // Literals_Section_Header: type=2 (Compressed)
    let size_format = if regen_size < 1024 && compressed_size < 1024 { 0b00u8 }
        else if regen_size < 16384 { 0b01 }
        else { 0b10 };

    match size_format {
        0b00 => {
            // 3-byte header: type(2) + size_format(2) + regen(10) + comp(10)
            let h = (LIT_TYPE_COMPRESSED as u32)
                | ((size_format as u32) << 2)
                | (((use_4_streams as u32) & 1) << 4) // single/4-streams flag in bit 4
                | ((regen_size as u32) << 4)
                | ((compressed_size as u32) << 14);
            out.extend_from_slice(&h.to_le_bytes()[..3]);
        }
        0b01 => {
            // 4-byte header
            let h = (LIT_TYPE_COMPRESSED as u32)
                | ((size_format as u32) << 2)
                | ((regen_size as u32) << 4)
                | ((compressed_size as u32) << 18);
            out.extend_from_slice(&h.to_le_bytes()[..4]);
        }
        _ => {
            // 5-byte header
            let h = (LIT_TYPE_COMPRESSED as u64)
                | ((size_format as u64) << 2)
                | ((regen_size as u64) << 4)
                | ((compressed_size as u64) << 22);
            out.extend_from_slice(&h.to_le_bytes()[..5]);
        }
    }

    out.extend_from_slice(&tree_desc);
    out.extend_from_slice(&compressed_streams);
    out
}

/// Build canonical Huffman codes using the package-merge approach.
/// Returns (codes[256], max_bits). max_bits <= 11 (zstd spec).
fn build_huffman_codes(counts: &[u32; 256], max_sym: usize) -> ([(u32, u8); 256], u8) {
    const MAX_BITS: u8 = 11;

    let mut syms: Vec<(u32, u8)> = (0..=max_sym)
        .filter(|&s| counts[s] > 0)
        .map(|s| (counts[s], s as u8))
        .collect();
    syms.sort();

    let n = syms.len();
    if n == 0 { return ([(0, 0); 256], 0); }
    if n == 1 {
        let mut codes = [(0u32, 0u8); 256];
        codes[syms[0].1 as usize] = (0, 1);
        return (codes, 1);
    }

    // Build optimal length-limited Huffman using simple approach:
    // Start with equal-length codes, then iteratively optimize.
    let mut lengths = [0u8; 256];
    let base_bits = ((n as f64).log2().ceil() as u8).clamp(1, MAX_BITS);

    for &(_, sym) in &syms {
        lengths[sym as usize] = base_bits;
    }

    // Iteratively adjust lengths based on frequency
    let total: u64 = syms.iter().map(|&(c, _)| c as u64).sum();
    for &(freq, sym) in &syms {
        let ideal = (total as f64 / freq as f64).log2().round() as u8;
        lengths[sym as usize] = ideal.clamp(1, MAX_BITS);
    }

    // Fix Kraft inequality: sum(2^(max-len)) must be a power of 2
    // Use iterative adjustment
    for _ in 0..200 {
        let kraft: i64 = (0..256).filter(|&s| lengths[s] > 0)
            .map(|s| 1i64 << (MAX_BITS - lengths[s]))
            .sum();
        let target = 1i64 << MAX_BITS;
        let diff = kraft - target;
        if diff == 0 { break; }

        if diff > 0 {
            // Over-full: decrease length of most frequent (shortest code) symbol
            let mut best = (0usize, 0u32);
            for &(freq, sym) in syms.iter().rev() {
                if lengths[sym as usize] > 1 && freq > best.1 {
                    best = (sym as usize, freq);
                    break;
                }
            }
            if best.1 > 0 { lengths[best.0] -= 1; }
            else { break; }
        } else {
            // Under-full: increase length of least frequent (longest code) symbol
            let mut best = (0usize, u32::MAX);
            for &(freq, sym) in &syms {
                if lengths[sym as usize] < MAX_BITS && freq < best.1 {
                    best = (sym as usize, freq);
                    break;
                }
            }
            if best.1 < u32::MAX { lengths[best.0] += 1; }
            else { break; }
        }
    }

    // Final force: make Kraft sum exactly a power of 2
    let kraft: u64 = (0..256).filter(|&s| lengths[s] > 0)
        .map(|s| 1u64 << (MAX_BITS - lengths[s]))
        .sum();
    if !kraft.is_power_of_two() {
        // Fallback to uniform lengths
        let bits = (n as f64).log2().ceil() as u8;
        for &(_, sym) in &syms { lengths[sym as usize] = bits; }
        // Adjust last symbol to make power of 2
        let kraft: u64 = (0..256).filter(|&s| lengths[s] > 0)
            .map(|s| 1u64 << (MAX_BITS - lengths[s]))
            .sum();
        if !kraft.is_power_of_two() {
            // Give up on Huffman, caller will fall back to raw
            return ([(0, 0); 256], 0);
        }
    }

    // Build canonical codes
    let max_bits = *lengths.iter().max().unwrap_or(&0);
    let mut bl_count = [0u32; 16];
    for &l in &lengths { if l > 0 { bl_count[l as usize] += 1; } }

    let mut next_code = [0u32; 16];
    for bits in 1..=max_bits {
        next_code[bits as usize] = (next_code[(bits - 1) as usize] + bl_count[(bits - 1) as usize]) << 1;
    }

    let mut codes = [(0u32, 0u8); 256];
    for s in 0..256 {
        let l = lengths[s];
        if l > 0 {
            codes[s] = (next_code[l as usize], l);
            next_code[l as usize] += 1;
        }
    }

    (codes, max_bits)
}

/// Encode Huffman tree description as weights.
/// zstd uses weight = max_bits + 1 - num_bits (0 for unused).
fn encode_huffman_tree(codes: &[(u32, u8); 256], max_bits: u8, max_sym: usize) -> Vec<u8> {
    // Compute weights
    let mut weights: Vec<u8> = (0..=max_sym)
        .map(|s| if codes[s].1 > 0 { max_bits + 1 - codes[s].1 } else { 0 })
        .collect();

    // Trim trailing zeros
    while weights.last() == Some(&0) && weights.len() > 1 { weights.pop(); }
    // The last weight is implicit (not stored)
    if !weights.is_empty() { weights.pop(); }

    // Direct representation: header byte = numSymbols + 127
    // Each pair of weights packed into one byte (high nibble, low nibble)
    // Max numSymbols for direct mode: 128 (header = 255)
    let num_weights = weights.len();
    if num_weights == 0 || num_weights > 128 {
        // Too many symbols for direct mode — return empty to trigger raw fallback
        return vec![];
    }
    let mut desc = Vec::with_capacity(1 + num_weights.div_ceil(2));
    desc.push((num_weights as u8) + 127);

    for pair in weights.chunks(2) {
        let w0 = pair[0];
        let w1 = if pair.len() > 1 { pair[1] } else { 0 };
        desc.push((w0 << 4) | (w1 & 0x0F));
    }

    desc
}

/// Encode literals using a single Huffman stream.
/// Symbols are encoded in reverse order, padded with a sentinel bit.
fn encode_huf_1stream(literals: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
    let mut bits: u64 = 0;
    let mut bit_pos: u32 = 0;
    let mut out = Vec::with_capacity(literals.len());

    // Encode in reverse
    for &sym in literals.iter().rev() {
        let (code, nbits) = codes[sym as usize];
        if nbits == 0 { continue; }
        bits |= (code as u64) << bit_pos;
        bit_pos += nbits as u32;
        while bit_pos >= 8 {
            out.push(bits as u8);
            bits >>= 8;
            bit_pos -= 8;
        }
    }

    // Sentinel: add 1-bit to mark end
    bits |= 1u64 << bit_pos;
    bit_pos += 1;
    while bit_pos > 0 {
        out.push(bits as u8);
        bits >>= 8;
        bit_pos = bit_pos.saturating_sub(8);
    }

    // Reverse because zstd reads backward
    out.reverse();
    out
}

/// Encode literals using 4 interleaved Huffman streams.
fn encode_huf_4streams(literals: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
    let split = literals.len().div_ceil(4);
    let s1 = &literals[..split];
    let s2 = &literals[split..std::cmp::min(split * 2, literals.len())];
    let s3 = &literals[std::cmp::min(split * 2, literals.len())..std::cmp::min(split * 3, literals.len())];
    let s4 = &literals[std::cmp::min(split * 3, literals.len())..];

    let c1 = encode_huf_1stream(s1, codes);
    let c2 = encode_huf_1stream(s2, codes);
    let c3 = encode_huf_1stream(s3, codes);
    let c4 = encode_huf_1stream(s4, codes);

    // Jump table: 3 x u16 LE (sizes of first 3 streams)
    let mut out = Vec::with_capacity(6 + c1.len() + c2.len() + c3.len() + c4.len());
    out.extend_from_slice(&(c1.len() as u16).to_le_bytes());
    out.extend_from_slice(&(c2.len() as u16).to_le_bytes());
    out.extend_from_slice(&(c3.len() as u16).to_le_bytes());
    out.extend_from_slice(&c1);
    out.extend_from_slice(&c2);
    out.extend_from_slice(&c3);
    out.extend_from_slice(&c4);
    out
}

// =========================================================================
// Literals section encoding (Raw mode)
// =========================================================================

fn encode_literals_raw(out: &mut Vec<u8>, literals: &[u8]) {
    let size = literals.len();

    if size <= 31 {
        // 1-byte header: type=0 (raw), size in 5 bits
        out.push(LIT_TYPE_RAW | ((size as u8) << 3));
    } else if size <= 4095 {
        // 2-byte header
        let h = (LIT_TYPE_RAW as u16) | (1 << 2) | ((size as u16) << 4);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        // 3-byte header
        let h = (LIT_TYPE_RAW as u32) | (3 << 2) | ((size as u32) << 4);
        out.extend_from_slice(&h.to_le_bytes()[..3]);
    }

    out.extend_from_slice(literals);
}

// =========================================================================
// Sequences section encoding using exact C-compatible FSE tables
// =========================================================================

fn encode_sequences_section(out: &mut Vec<u8>, sequences: &[Sequence]) {
    let nb_seq = sequences.len();

    // Number of sequences header
    if nb_seq < 128 {
        out.push(nb_seq as u8);
    } else if nb_seq < 0x7F00 {
        out.push(((nb_seq >> 8) as u8) + 128);
        out.push(nb_seq as u8);
    } else {
        out.push(255);
        out.extend_from_slice(&((nb_seq - 0x7F00) as u16).to_le_bytes());
    }

    if nb_seq == 0 { return; }

    // Symbol compression modes: all predefined
    out.push(0x00);

    // Build predefined FSE tables (exact C-compatible implementation)
    let ll_table = super::fse::FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
    let ml_table = super::fse::FseCTable::build(&ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);
    let of_table = super::fse::FseCTable::build(
        &OF_DEFAULT_NORM, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG,
    );

    // Convert sequences to codes + extra bit values
    let mut ll_codes_v = Vec::with_capacity(nb_seq);
    let mut ml_codes_v = Vec::with_capacity(nb_seq);
    let mut off_codes_v = Vec::with_capacity(nb_seq);
    let mut ll_values = Vec::with_capacity(nb_seq);
    let mut ml_values = Vec::with_capacity(nb_seq);
    let mut off_values = Vec::with_capacity(nb_seq);

    for seq in sequences {
        let llc = ll_code(seq.ll);
        let ml_base = seq.ml - ZSTD_MINMATCH as u32;
        let mlc = ml_code(ml_base);
        // Sequence OF stores an "offset value", not the raw back-reference
        // distance. Values 1..=3 are reserved for repeat offsets, so new
        // offsets must be encoded as actual_offset + 3.
        let of_value = seq.off + 3;
        let ofc = off_code(of_value);

        // Extra bit values = value - base_for_code
        ll_codes_v.push(llc);
        ml_codes_v.push(mlc);
        off_codes_v.push(ofc);
        ll_values.push(seq.ll - LL_BASE[llc as usize]);
        ml_values.push(seq.ml - ML_BASE[mlc as usize]);
        // Offset extra bits reconstruct the full offset value, which the
        // decoder later maps back to the actual offset using repcode rules.
        off_values.push(if ofc > 0 { of_value - (1u32 << ofc) } else { 0 });
    }

    // Encode with the exact C-compatible FSE sequence encoder
    let bitstream = super::fse::encode_sequences(
        &ll_table, &of_table, &ml_table,
        &ll_codes_v, &off_codes_v, &ml_codes_v,
        &ll_values, &ml_values, &off_values,
    );
    out.extend_from_slice(&bitstream);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_empty() {
        let compressed = compress(&[], 1);
        assert!(compressed.len() >= 5); // magic + header + empty block
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    #[test]
    fn compress_small() {
        let data = b"hello world";
        let compressed = compress(data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert!(compressed.len() > 5);
    }

    #[test]
    fn compress_repetitive() {
        let data = vec![42u8; 4096];
        let compressed = compress(&data, 1);
        // Valid zstd frame (raw blocks are larger than input due to framing)
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    #[test]
    fn compress_real_data() {
        let data: Vec<u8> = (0..1024u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    /// Golden test: roundtrip through our compressor → our decompressor.
    #[test]
    fn roundtrip_self_contained() {
        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros", vec![0u8; 4096]),
            ("sequential", (0..4096u32).flat_map(|i| i.to_le_bytes()).collect()),
            ("f32_data", (0..256u32).flat_map(|i| (i as f32 * 1.5).to_le_bytes()).collect()),
            ("repetitive", b"hello world! ".repeat(100)),
            ("small", b"abc".to_vec()),
        ];

        for (name, data) in &test_cases {
            let compressed = compress(data, 1);
            let decompressed = crate::zstd::decompress(&compressed)
                .unwrap_or_else(|e| panic!("{}: decompress failed: {}", name, e));

            assert_eq!(decompressed.len(), data.len(), "{}: length mismatch", name);
            assert_eq!(&decompressed, data, "{}: data mismatch", name);
        }
    }
}
