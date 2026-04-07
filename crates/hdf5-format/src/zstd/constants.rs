//! Zstandard format constants, tables, and symbol coding definitions.
//! Ported from zstd C source (lib/common/zstd_internal.h, decompress_internal.h).

pub const ZSTD_MAGIC: u32 = 0xFD2FB528;
pub const ZSTD_BLOCKSIZELOG_MAX: u32 = 17;
pub const ZSTD_BLOCKSIZE_MAX: usize = 1 << ZSTD_BLOCKSIZELOG_MAX; // 128 KiB
pub const ZSTD_WINDOWLOG_MAX: u32 = 31;
pub const ZSTD_MINMATCH: usize = 3;

// Block types
pub const BLOCK_TYPE_RAW: u8 = 0;
pub const BLOCK_TYPE_RLE: u8 = 1;
pub const BLOCK_TYPE_COMPRESSED: u8 = 2;

// Literal block types
pub const LIT_TYPE_RAW: u8 = 0;
pub const LIT_TYPE_RLE: u8 = 1;
pub const LIT_TYPE_COMPRESSED: u8 = 2;
pub const LIT_TYPE_TREELESS: u8 = 3;

// Sequence encoding modes
pub const SEQ_MODE_PREDEFINED: u8 = 0;
pub const SEQ_MODE_RLE: u8 = 1;
pub const SEQ_MODE_FSE: u8 = 2;
pub const SEQ_MODE_REPEAT: u8 = 3;

// Maximum symbol values
pub const MAX_LL: usize = 35;
pub const MAX_ML: usize = 52;
pub const MAX_OFF: usize = 31;

// FSE table log sizes
pub const LL_FSE_LOG: u32 = 9;
pub const ML_FSE_LOG: u32 = 9;
pub const OFF_FSE_LOG: u32 = 8;

// Default norm log
pub const LL_DEFAULT_NORM_LOG: u32 = 6;
pub const ML_DEFAULT_NORM_LOG: u32 = 6;
pub const OF_DEFAULT_NORM_LOG: u32 = 5;

/// Extra bits for each literal length code.
pub const LL_BITS: [u8; MAX_LL + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3,
    4, 6, 7, 8, 9,10,11,12,
   13,14,15,16,
];

/// Base value for each literal length code.
pub const LL_BASE: [u32; MAX_LL + 1] = [
     0,  1,  2,   3,   4,   5,   6,    7,
     8,  9, 10,  11,  12,  13,  14,   15,
    16, 18, 20,  22,  24,  28,  32,   40,
    48, 64, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000,
    0x2000, 0x4000, 0x8000, 0x10000,
];

/// Extra bits for each match length code.
pub const ML_BITS: [u8; MAX_ML + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3,
    4, 4, 5, 7, 8, 9,10,11,
   12,13,14,15,16,
];

/// Base value for each match length code.
pub const ML_BASE: [u32; MAX_ML + 1] = [
     3,  4,  5,  6,  7,  8,  9, 10,
    11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34,
    35, 37, 39, 41, 43, 47, 51, 59,
    67, 83, 99, 0x83, 0x103, 0x203, 0x403, 0x803,
    0x1003, 0x2003, 0x4003, 0x8003, 0x10003,
];

/// Extra bits for each offset code. offset_code = OF_bits[code].
pub const OF_BITS: [u8; MAX_OFF + 1] = [
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
];

/// Default normalized count for literal length FSE table.
pub const LL_DEFAULT_NORM: [i16; MAX_LL + 1] = [
     4, 3, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2,
     2, 3, 2, 1, 1, 1, 1, 1,
    -1,-1,-1,-1,
];

/// Default normalized count for match length FSE table.
pub const ML_DEFAULT_NORM: [i16; MAX_ML + 1] = [
     1, 4, 3, 2, 2, 2, 2, 2,
     2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1,-1,-1,
    -1,-1,-1,-1,-1,
];

/// Default normalized count for offset FSE table.
pub const OF_DEFAULT_NORM: [i16; MAX_OFF - 2] = [
     1, 1, 1, 1, 1, 1, 2, 2,
     2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1,
    -1,-1,-1,-1,-1,
];

/// Lookup table for literal length code (values 0..63).
const LL_CODE_TABLE: [u8; 64] = [
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21,
    22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
];

/// Lookup table for match length code (values 0..127, where value = matchLength - 3).
const ML_CODE_TABLE: [u8; 128] = [
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
    38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
];

/// Convert a literal length value to its code.
/// Ported from ZSTD_LLcode() in zstd_compress_internal.h.
pub fn ll_code(litlen: u32) -> u8 {
    const LL_DELTA: u32 = 19;
    if litlen <= 63 { LL_CODE_TABLE[litlen as usize] }
    else { (highbit32(litlen) + LL_DELTA) as u8 }
}

/// Convert a match length base (matchLength - 3) to its code.
/// Ported from ZSTD_MLcode() in zstd_compress_internal.h.
pub fn ml_code(ml_base: u32) -> u8 {
    const ML_DELTA: u32 = 36;
    if ml_base <= 127 { ML_CODE_TABLE[ml_base as usize] }
    else { (highbit32(ml_base) + ML_DELTA) as u8 }
}

/// Convert an offset value to its code (highest bit position + 1).
pub fn off_code(offset: u32) -> u8 {
    highbit32(offset) as u8
}

/// Highest set bit (0-indexed). Returns 0 for input 0 or 1.
fn highbit32(v: u32) -> u32 {
    if v <= 1 { return 0; }
    31 - v.leading_zeros()
}
