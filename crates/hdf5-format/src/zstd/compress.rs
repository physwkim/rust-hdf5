//! Zstandard frame compressor.
//!
//! Produces valid zstd frames decompressible by any standard decoder.
//! Uses greedy hash-based matching (equivalent to zstd level 1).

use super::constants::*;

/// Compress data into a zstd frame.
///
/// Returns a valid zstd frame that can be decompressed by any conformant decoder
/// or any other conformant decoder.
pub fn compress(data: &[u8], #[allow(unused)] _level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);

    // Frame header
    write_frame_header(&mut out, data.len() as u64);

    // Split into blocks (max 128KB each)
    let blocks: Vec<&[u8]> = data.chunks(ZSTD_BLOCKSIZE_MAX).collect();
    let n_blocks = blocks.len();

    for (i, block) in blocks.iter().enumerate() {
        let is_last = i == n_blocks - 1;
        match compress_block(block) {
            Some(compressed) if compressed.len() < block.len() => {
                write_compressed_block(&mut out, &compressed, is_last);
            }
            _ => write_raw_block(&mut out, block, is_last),
        }
    }

    // Handle empty input
    if data.is_empty() {
        write_raw_block(&mut out, &[], true);
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

fn compress_block(data: &[u8]) -> Option<Vec<u8>> {
    let sequences = find_matches(data);

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

    // === Literals section (Raw mode — no Huffman for simplicity) ===
    encode_literals_raw(&mut block, &literals);

    // === Sequences section ===
    encode_sequences_section(&mut block, &sequences);

    Some(block)
}

/// Greedy hash-based match finder (zstd level 1 strategy).
fn find_matches(data: &[u8]) -> Vec<Sequence> {
    if data.len() < ZSTD_MINMATCH + 1 {
        return vec![];
    }

    let hash_log = 14u32;
    let hash_size = 1usize << hash_log;
    let mut hash_table = vec![0u32; hash_size];
    let mut sequences = Vec::new();
    let mut anchor = 0usize; // start of current literal run
    let mut ip = 0usize;

    while ip + ZSTD_MINMATCH < data.len() {
        let h = hash4(&data[ip..]) & ((hash_size - 1) as u32);
        let match_pos = hash_table[h as usize] as usize;
        hash_table[h as usize] = ip as u32;

        // Check match
        if match_pos > 0
            && ip - match_pos < (1 << 24) // max offset
            && ip - match_pos > 0
            && data[match_pos..].len() >= ZSTD_MINMATCH
            && data[match_pos..match_pos + 4] == data[ip..ip + 4]
        {
            // Found a match — extend it
            let mut ml = 4;
            let max_ml = std::cmp::min(data.len() - ip, data.len() - match_pos);
            while ml < max_ml && data[match_pos + ml] == data[ip + ml] {
                ml += 1;
            }

            if ml >= ZSTD_MINMATCH {
                let ll = (ip - anchor) as u32;
                let offset = (ip - match_pos) as u32;

                sequences.push(Sequence { ll, off: offset, ml: ml as u32 });
                ip += ml;
                anchor = ip;
                continue;
            }
        }

        ip += 1;
    }

    sequences
}

/// 4-byte hash function (from zstd fast strategy).
fn hash4(data: &[u8]) -> u32 {
    let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    v.wrapping_mul(0x9E3779B1) >> 18
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
