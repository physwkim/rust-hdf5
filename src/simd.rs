//! The run-time SIMD level the crate's kernels dispatch on.
//!
//! `fearless_simd` picks AVX2, AVX-512 or NEON at run time; the loops the
//! compiler vectorizes on its own only reach the baseline the binary was
//! compiled for (SSE2 on x86_64). The level is detected once per process.

use fearless_simd::Level;
use std::sync::OnceLock;

/// The detected level, once per process.
pub(crate) fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}
