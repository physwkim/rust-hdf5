//! Higher compression levels for zstd via custom Matchers.
//!
//! Implements lazy and optimal matching strategies for levels 3, 7, and 11
//! using ruzstd's Matcher trait.

use std::vec::Vec;
use ruzstd::encoding::{CompressionLevel, Matcher, Sequence};

const MIN_MATCH: usize = 4;

// =========================================================================
// Hash chain match finder — supports lazy matching (levels 3-7)
// =========================================================================

/// Hash chain-based matcher with configurable lazy depth.
///
/// - `lazy_depth=0`: greedy (level 1)
/// - `lazy_depth=1`: lazy (level 3)
/// - `lazy_depth=2`: lazy2 (level 7)
pub struct HashChainMatcher {
    window: Vec<u8>,
    hash_table: Vec<u32>,
    chain: Vec<u32>,
    hash_log: u32,
    window_log: u32,
    pos: usize,
    lazy_depth: u32,
    max_window: usize,
    slice_size: usize,
    // Buffering for the Matcher trait
    spaces: Vec<Vec<u8>>,
    current_space: usize,
}

impl HashChainMatcher {
    pub fn new(level: i32) -> Self {
        let (hash_log, window_log, lazy_depth) = match level {
            0..=2 => (14, 17, 0),    // greedy
            3..=5 => (15, 19, 1),    // lazy
            6..=8 => (16, 21, 1),    // lazy + deeper search
            9..=11 => (17, 22, 1),   // lazy + deepest search
            _ => (17, 22, 1),
        };
        let hash_size = 1usize << hash_log;
        let max_window = 1usize << window_log;
        let slice_size = 128 * 1024; // 128KB blocks

        Self {
            window: Vec::with_capacity(max_window),
            hash_table: vec![0u32; hash_size],
            chain: Vec::new(),
            hash_log,
            window_log,
            pos: 0,
            lazy_depth,
            max_window,
            slice_size,
            spaces: Vec::new(),
            current_space: 0,
        }
    }

    fn hash4(&self, data: &[u8]) -> usize {
        let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        (v.wrapping_mul(0x9E3779B1) >> (32 - self.hash_log)) as usize
    }

    fn insert(&mut self, pos: usize) {
        if pos + 4 > self.window.len() { return; }
        let h = self.hash4(&self.window[pos..]);
        if pos < self.chain.len() {
            self.chain[pos] = self.hash_table[h];
        }
        self.hash_table[h] = pos as u32;
    }

    fn find_best_match(&self, pos: usize, max_depth: u32) -> Option<(usize, usize)> {
        if pos + MIN_MATCH > self.window.len() { return None; }
        let h = self.hash4(&self.window[pos..]);
        let mut candidate = self.hash_table[h] as usize;
        let mut best_len = MIN_MATCH - 1;
        let mut best_off = 0;
        let mut depth = 0;
        let max_back = std::cmp::min(pos, self.max_window);

        while candidate > 0 && candidate < pos && depth < max_depth && pos - candidate <= max_back {
            let match_len = common_prefix_len(
                &self.window[candidate..],
                &self.window[pos..],
            );
            if match_len > best_len {
                best_len = match_len;
                best_off = pos - candidate;
            }
            if candidate >= self.chain.len() { break; }
            let next = self.chain[candidate] as usize;
            if next >= candidate { break; } // prevent loops
            candidate = next;
            depth += 1;
        }

        if best_len >= MIN_MATCH {
            Some((best_off, best_len))
        } else {
            None
        }
    }

    fn generate_sequences(&mut self, data: &[u8]) -> Vec<(usize, usize, usize)> {
        // Reset window per block to keep offsets valid within the block
        self.window.clear();
        self.chain.clear();
        self.hash_table.fill(0);

        let base = 0;
        self.window.extend_from_slice(data);
        self.chain.resize(self.window.len(), 0);

        let search_depth = match self.lazy_depth {
            0 => 4u32,
            1 => 16,
            _ => 64,
        };

        let mut sequences = Vec::new();
        let mut ip = base;
        let mut anchor = base;

        while ip + MIN_MATCH < self.window.len() {
            if let Some((offset, match_len)) = self.find_best_match(ip, search_depth) {
                self.insert(ip);
                // Lazy matching: check if next position gives a better match
                let mut final_offset = offset;
                let mut final_len = match_len;
                let mut final_ip = ip;

                if self.lazy_depth >= 1 && ip + 1 + MIN_MATCH < self.window.len() {
                    if let Some((off2, len2)) = self.find_best_match(ip + 1, search_depth) {
                        self.insert(ip + 1);
                        if len2 > final_len + 1 {
                            final_offset = off2;
                            final_len = len2;
                            final_ip = ip + 1;
                        }
                    }
                }

                if self.lazy_depth >= 2 && final_ip + 1 + MIN_MATCH < self.window.len() {
                    self.insert(final_ip + 1);
                    if let Some((off3, len3)) = self.find_best_match(final_ip + 1, search_depth) {
                        if len3 > final_len + 1 {
                            final_offset = off3;
                            final_len = len3;
                            final_ip += 1;
                        }
                    }
                }

                let lit_len = final_ip - anchor;
                sequences.push((lit_len, final_offset, final_len));

                // Insert positions within the match for future matching
                for p in (final_ip + 1)..std::cmp::min(final_ip + final_len, self.window.len().saturating_sub(4)) {
                    self.insert(p);
                }

                ip = final_ip + final_len;
                anchor = ip;
            } else {
                self.insert(ip);
                ip += 1;
            }
        }

        // Trim window if too large
        if self.window.len() > self.max_window * 2 {
            let trim = self.window.len() - self.max_window;
            self.window.drain(..trim);
            self.chain.drain(..trim);
            // Rebuild hash table (simplified: just clear)
            self.hash_table.fill(0);
            for i in 0..self.window.len().saturating_sub(4) {
                self.insert(i);
            }
        }

        sequences
    }
}

impl Matcher for HashChainMatcher {
    fn get_next_space(&mut self) -> Vec<u8> {
        vec![0u8; self.slice_size]
    }

    fn get_last_space(&mut self) -> &[u8] {
        self.spaces.last().map(|s| s.as_slice()).unwrap_or(&[])
    }

    fn commit_space(&mut self, space: Vec<u8>) {
        self.spaces.push(space);
    }

    fn skip_matching(&mut self) {
        if let Some(space) = self.spaces.last() {
            let data = space.clone();
            let base = self.window.len();
            self.window.extend_from_slice(&data);
            self.chain.resize(self.window.len(), 0);
            for i in base..self.window.len().saturating_sub(4) {
                self.insert(i);
            }
        }
    }

    fn start_matching(&mut self, mut handle_sequence: impl for<'a> FnMut(Sequence<'a>)) {
        let space = match self.spaces.last() {
            Some(s) => s.clone(),
            None => return,
        };

        let base = self.window.len();
        let seqs = self.generate_sequences(&space);

        let mut pos = 0usize;
        for (lit_len, offset, match_len) in &seqs {
            let literals = &space[pos..pos + lit_len];
            handle_sequence(Sequence::Triple {
                literals,
                offset: *offset,
                match_len: *match_len,
            });
            pos += lit_len + match_len;
        }

        // Trailing literals
        if pos < space.len() {
            handle_sequence(Sequence::Literals {
                literals: &space[pos..],
            });
        }
    }

    fn reset(&mut self, _level: CompressionLevel) {
        self.window.clear();
        self.hash_table.fill(0);
        self.chain.clear();
        self.spaces.clear();
        self.pos = 0;
    }

    fn window_size(&self) -> u64 {
        self.max_window as u64
    }
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let max = std::cmp::min(a.len(), b.len());
    let mut i = 0;
    // Process 8 bytes at a time
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
