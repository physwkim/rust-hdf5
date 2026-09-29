//! The byte and bit transposes behind the shuffle-family filters.
//!
//! One owner for every "gather byte `k` of each element" loop in the crate:
//! the shuffle filter, the blosc byte shuffle, the szip interleave and the
//! first stage of bitshuffle are all [`trans_byte_elem`] (or its inverse
//! [`untrans_byte_elem`]); the bit-level stages of bitshuffle follow the
//! canonical library (kiyo-masui/bitshuffle `bitshuffle_core.c`) stage for
//! stage, so the on-disk image is the one h5py and libhdf5 produce.
//!
//! Every stage has a scalar loop that handles any element size. With the
//! `simd` feature the 1-, 2-, 4- and 8-byte element sizes run a
//! `fearless_simd` kernel over the vector-sized prefix first; the kernel
//! returns where it stopped and the scalar loop finishes the rest, so the
//! two paths are one function by construction and the kernels are tested
//! against the loops on every level the host offers.

/// 8x8 bit-matrix transpose of a quadword, little-endian convention
/// (library macro `TRANS_BIT_8X8`, bitshuffle_core.c:89).
#[inline]
fn trans_bit_8x8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x = x ^ t ^ (t << 28);
    x
}

/// Read 8 bytes at `off` as a little-endian quadword.
#[inline]
fn read_u64_le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// The SIMD level every kernel in this module dispatches on.
#[cfg(feature = "simd")]
macro_rules! dispatch {
    ($($tt:tt)*) => {
        fearless_simd::dispatch!(crate::simd::level(), $($tt)*)
    };
}

// ---------------------------------------------------------------------------
// Byte transpose: element-major <-> plane-major
// ---------------------------------------------------------------------------

/// Gather byte `k` of every element into plane `k`: `out[k * size + i] =
/// input[i * elem_size + k]` for `size` elements of `elem_size` bytes.
///
/// This is the HDF5 shuffle filter (`H5Z__filter_shuffle`), blosc's byte
/// shuffle, szip's interleave and the library's `bshuf_trans_byte_elem_scal`
/// (bitshuffle_core.c:174). Only the first `size * elem_size` bytes of
/// either slice are touched.
pub(crate) fn trans_byte_elem(input: &[u8], out: &mut [u8], size: usize, elem_size: usize) {
    if elem_size == 1 {
        out[..size].copy_from_slice(&input[..size]);
        return;
    }
    #[cfg(feature = "simd")]
    let done = dispatch!(s => simd::trans_byte_elem(s, input, out, size, elem_size));
    #[cfg(not(feature = "simd"))]
    let done = 0;
    trans_byte_elem_from(done, input, out, size, elem_size);
}

/// The element loops of [`trans_byte_elem`] from element `from`, a
/// multiple of 8.
fn trans_byte_elem_from(from: usize, input: &[u8], out: &mut [u8], size: usize, elem_size: usize) {
    let mut ii = from;
    while ii + 7 < size {
        for jj in 0..elem_size {
            for kk in 0..8 {
                out[jj * size + ii + kk] = input[ii * elem_size + kk * elem_size + jj];
            }
        }
        ii += 8;
    }
    let mut ii = ii.max(size - size % 8);
    while ii < size {
        for jj in 0..elem_size {
            out[jj * size + ii] = input[ii * elem_size + jj];
        }
        ii += 1;
    }
}

/// The inverse of [`trans_byte_elem`]: `out[i * elem_size + k] =
/// input[k * size + i]`. The HDF5 unshuffle, blosc unshuffle and szip
/// deinterleave.
pub(crate) fn untrans_byte_elem(input: &[u8], out: &mut [u8], size: usize, elem_size: usize) {
    if elem_size == 1 {
        out[..size].copy_from_slice(&input[..size]);
        return;
    }
    #[cfg(feature = "simd")]
    let done = dispatch!(s => simd::untrans_byte_elem(s, input, out, size, elem_size));
    #[cfg(not(feature = "simd"))]
    let done = 0;
    untrans_byte_elem_from(done, input, out, size, elem_size);
}

/// The element loop of [`untrans_byte_elem`] from element `from`.
fn untrans_byte_elem_from(
    from: usize,
    input: &[u8],
    out: &mut [u8],
    size: usize,
    elem_size: usize,
) {
    for jj in 0..elem_size {
        let plane = &input[jj * size..(jj + 1) * size];
        for ii in from..size {
            out[ii * elem_size + jj] = plane[ii];
        }
    }
}

// ---------------------------------------------------------------------------
// Bitshuffle: the canonical three-stage bit transpose and its inverse
// ---------------------------------------------------------------------------

/// Transpose bits within bytes (library `bshuf_trans_bit_byte_scal`,
/// bitshuffle_core.c:219, little-endian path): bit `r` of every input byte
/// lands in bit row `r`, `nbyte / 8` bytes long.
fn trans_bit_byte(input: &[u8], out: &mut [u8], nbyte: usize) {
    #[cfg(feature = "simd")]
    let done = dispatch!(s => simd::trans_bit_byte(s, input, out, nbyte));
    #[cfg(not(feature = "simd"))]
    let done = 0;
    trans_bit_byte_from(done, input, out, nbyte);
}

/// The quadword loop of [`trans_bit_byte`] from byte `from`, a multiple of
/// 8.
fn trans_bit_byte_from(from: usize, input: &[u8], out: &mut [u8], nbyte: usize) {
    let nbyte_bitrow = nbyte / 8;
    for ii in from / 8..nbyte_bitrow {
        let mut x = trans_bit_8x8(read_u64_le(input, ii * 8));
        for kk in 0..8 {
            out[kk * nbyte_bitrow + ii] = x as u8;
            x >>= 8;
        }
    }
}

/// Transpose rows of shuffled bits within groups of eight (library
/// `bshuf_trans_bitrow_eight` -> `bshuf_trans_elem`, lda=8, ldb=elem_size).
fn trans_bitrow_eight(input: &[u8], out: &mut [u8], size: usize, elem_size: usize) {
    let nbyte_bitrow = size / 8;
    for ii in 0..8 {
        for jj in 0..elem_size {
            let src = (ii * elem_size + jj) * nbyte_bitrow;
            let dst = (jj * 8 + ii) * nbyte_bitrow;
            out[dst..dst + nbyte_bitrow].copy_from_slice(&input[src..src + nbyte_bitrow]);
        }
    }
}

/// Transpose bytes for data organized as one row per bit (library
/// `bshuf_trans_byte_bitrow_scal`, bitshuffle_core.c:281).
fn trans_byte_bitrow(input: &[u8], out: &mut [u8], size: usize, elem_size: usize) {
    #[cfg(feature = "simd")]
    let done = dispatch!(s => simd::trans_byte_bitrow(s, input, out, size, elem_size));
    #[cfg(not(feature = "simd"))]
    let done = 0;
    trans_byte_bitrow_from(done, input, out, size, elem_size);
}

/// The column loops of [`trans_byte_bitrow`] from column `from`.
fn trans_byte_bitrow_from(
    from: usize,
    input: &[u8],
    out: &mut [u8],
    size: usize,
    elem_size: usize,
) {
    let nbyte_row = size / 8;
    for jj in 0..elem_size {
        for ii in from..nbyte_row {
            for kk in 0..8 {
                out[ii * 8 * elem_size + jj * 8 + kk] = input[(jj * 8 + kk) * nbyte_row + ii];
            }
        }
    }
}

/// Shuffle bits within the bytes of eight-element groups (library
/// `bshuf_shuffle_bit_eightelem_scal`, bitshuffle_core.c:308, LE path).
fn shuffle_bit_eightelem(input: &[u8], out: &mut [u8], nbyte: usize, elem_size: usize) {
    #[cfg(feature = "simd")]
    let done = dispatch!(s => simd::shuffle_bit_eightelem(s, input, out, nbyte, elem_size));
    #[cfg(not(feature = "simd"))]
    let done = 0;
    shuffle_bit_eightelem_from(done, input, out, nbyte, elem_size);
}

/// The quadword loop of [`shuffle_bit_eightelem`] from byte `from`, a
/// multiple of 8: the library walks the quadwords group-column first, this
/// walks them in address order, and every quadword lands in the same place.
fn shuffle_bit_eightelem_from(
    from: usize,
    input: &[u8],
    out: &mut [u8],
    nbyte: usize,
    elem_size: usize,
) {
    let group = 8 * elem_size;
    let mut p = from;
    while p + 7 < nbyte {
        let mut x = trans_bit_8x8(read_u64_le(input, p));
        let base = p / group * group + p % group / 8;
        for kk in 0..8 {
            out[base + kk * elem_size] = x as u8;
            x >>= 8;
        }
        p += 8;
    }
}

/// The stage buffers of one block's transposes, sized for the largest block
/// of a stream and reused across its blocks.
pub(crate) struct BitshuffleScratch {
    a: Vec<u8>,
    b: Vec<u8>,
}

impl BitshuffleScratch {
    /// Scratch for blocks of up to `nbyte` bytes.
    pub(crate) fn new(nbyte: usize) -> Self {
        Self {
            a: vec![0u8; nbyte],
            b: vec![0u8; nbyte],
        }
    }
}

/// Bit-transpose one block of `input.len() / elem_size` elements, a
/// multiple of 8, into `out` — library `bshuf_trans_bit_elem_scal`
/// (bitshuffle_core.c:256): byte transpose, bit-within-byte transpose, then
/// bit-row transpose. Bit plane `q` of an element is `byte * 8 + bit`
/// (LSB first) and within a plane the elements pack LSB first.
pub(crate) fn bitshuffle_block_into(
    input: &[u8],
    scratch: &mut BitshuffleScratch,
    out: &mut [u8],
    elem_size: usize,
) {
    let size = input.len() / elem_size;
    debug_assert_eq!(size % 8, 0);
    let nbyte = size * elem_size;
    let (a, b) = (&mut scratch.a[..nbyte], &mut scratch.b[..nbyte]);
    trans_byte_elem(input, a, size, elem_size);
    trans_bit_byte(a, b, nbyte);
    trans_bitrow_eight(b, &mut out[..nbyte], size, elem_size);
}

/// The inverse of [`bitshuffle_block_into`] — library
/// `bshuf_untrans_bit_elem_scal` (bitshuffle_core.c:349).
pub(crate) fn bitunshuffle_block_into(
    input: &[u8],
    scratch: &mut BitshuffleScratch,
    out: &mut [u8],
    elem_size: usize,
) {
    let size = input.len() / elem_size;
    debug_assert_eq!(size % 8, 0);
    let nbyte = size * elem_size;
    let tmp = &mut scratch.b[..nbyte];
    trans_byte_bitrow(input, tmp, size, elem_size);
    shuffle_bit_eightelem(tmp, &mut out[..nbyte], nbyte, elem_size);
}

/// The transposes on `fearless_simd` lanes. Each kernel covers the
/// vector-sized prefix of its input and returns where it stopped, so the
/// scalar loop after it finishes the rest.
#[cfg(feature = "simd")]
mod simd {
    use fearless_simd::{prelude::*, Simd};
    use fearless_simd_macros::simd;

    /// [`super::trans_bit_8x8`] on every quadword lane of `x`, back as
    /// bytes: byte `8 * k + r` holds bit `r` of each byte of quadword `k`.
    #[inline(always)]
    fn bit_transposed<S: Simd>(simd: S, x: S::u8s) -> S::u8s {
        let mut x = S::u64s::from_bytes(x);
        let m = S::u64s::splat(simd, 0x00AA_00AA_00AA_00AA);
        let t = (x ^ (x >> 7)) & m;
        x = x ^ t ^ (t << 7);
        let m = S::u64s::splat(simd, 0x0000_CCCC_0000_CCCC);
        let t = (x ^ (x >> 14)) & m;
        x = x ^ t ^ (t << 14);
        let m = S::u64s::splat(simd, 0x0000_0000_F0F0_F0F0);
        let t = (x ^ (x >> 28)) & m;
        x = x ^ t ^ (t << 28);
        x.to_bytes()
    }

    /// The byte shuffle that turns [`bit_transposed`] output into its eight
    /// bit rows, `LEN / 8` bytes each: byte `r * q + k` takes byte
    /// `8 * k + r`.
    #[inline(always)]
    fn row_order<S: Simd>(simd: S) -> S::u8s {
        let q = S::u8s::LEN / 8;
        S::u8s::from_fn(simd, |i| (8 * (i % q) + i / q) as u8)
    }

    /// [`super::trans_byte_elem`] for 2-, 4- and 8-byte elements: the
    /// vector's worth of elements is `elem_size` byte vectors, and each
    /// deinterleave level halves the byte stride, so after `log2(elem_size)`
    /// levels vector `k` holds byte `k` of every element. Returns the
    /// elements done, 0 for any other element size.
    #[simd]
    pub(super) fn trans_byte_elem<S: Simd>(
        simd: S,
        input: &[u8],
        out: &mut [u8],
        size: usize,
        elem_size: usize,
    ) -> usize {
        if !matches!(elem_size, 2 | 4 | 8) {
            return 0;
        }
        let n = S::u8s::LEN;
        let mut v = [S::u8s::splat(simd, 0); 8];
        let mut ii = 0;
        while ii + n <= size {
            let chunk = &input[ii * elem_size..(ii + n) * elem_size];
            for (k, x) in v[..elem_size].iter_mut().enumerate() {
                *x = S::u8s::from_slice(simd, &chunk[k * n..(k + 1) * n]);
            }
            let mut stride = elem_size;
            while stride > 1 {
                let mut next = v;
                for k in 0..elem_size / 2 {
                    let (lo, hi) = v[2 * k].deinterleave(v[2 * k + 1]);
                    next[k] = lo;
                    next[elem_size / 2 + k] = hi;
                }
                v = next;
                stride /= 2;
            }
            for (k, x) in v[..elem_size].iter().enumerate() {
                x.store_slice(&mut out[k * size + ii..k * size + ii + n]);
            }
            ii += n;
        }
        ii
    }

    /// [`super::untrans_byte_elem`] for 2-, 4- and 8-byte elements: the
    /// deinterleave tree of [`trans_byte_elem`] run backwards, one
    /// interleave level per halving of the plane stride, so after
    /// `log2(elem_size)` levels vector `k` holds elements `k * LEN /
    /// elem_size ..` in element order. Returns the elements done, 0 for any
    /// other element size.
    #[simd]
    pub(super) fn untrans_byte_elem<S: Simd>(
        simd: S,
        input: &[u8],
        out: &mut [u8],
        size: usize,
        elem_size: usize,
    ) -> usize {
        if !matches!(elem_size, 2 | 4 | 8) {
            return 0;
        }
        let n = S::u8s::LEN;
        let mut v = [S::u8s::splat(simd, 0); 8];
        let mut ii = 0;
        while ii + n <= size {
            for (k, x) in v[..elem_size].iter_mut().enumerate() {
                *x = S::u8s::from_slice(simd, &input[k * size + ii..k * size + ii + n]);
            }
            let mut stride = 1;
            while stride < elem_size {
                let mut prev = v;
                for k in 0..elem_size / 2 {
                    let (lo, hi) = v[k].interleave(v[elem_size / 2 + k]);
                    prev[2 * k] = lo;
                    prev[2 * k + 1] = hi;
                }
                v = prev;
                stride *= 2;
            }
            let chunk = &mut out[ii * elem_size..(ii + n) * elem_size];
            for (k, x) in v[..elem_size].iter().enumerate() {
                x.store_slice(&mut chunk[k * n..(k + 1) * n]);
            }
            ii += n;
        }
        ii
    }

    /// [`super::trans_bit_byte`]: a vector of bytes at a time, its bit rows
    /// stored as `LEN / 8` bytes into the eight output rows. Returns the
    /// bytes done.
    #[simd]
    pub(super) fn trans_bit_byte<S: Simd>(
        simd: S,
        input: &[u8],
        out: &mut [u8],
        nbyte: usize,
    ) -> usize {
        let n = S::u8s::LEN;
        let q = n / 8;
        let nbyte_bitrow = nbyte / 8;
        let order = row_order(simd);
        let mut ii = 0;
        while ii + n <= nbyte {
            let rows = bit_transposed(simd, S::u8s::from_slice(simd, &input[ii..ii + n]))
                .swizzle_dyn(order);
            for (r, bytes) in rows.as_slice().chunks_exact(q).enumerate() {
                let o = r * nbyte_bitrow + ii / 8;
                out[o..o + q].copy_from_slice(bytes);
            }
            ii += n;
        }
        ii
    }

    /// One zip level of the row transpose: units `w` bytes wide, row groups
    /// `2m` and `2m + 1` of the same column range interleaved into the two
    /// column halves of group `m`. Units of 16 bytes and up are the quadword
    /// zip followed by `perm`, the lane permutation that regroups its
    /// alternating quadwords into alternating units.
    #[inline(always)]
    fn zip<S: Simd>(a: S::u8s, b: S::u8s, w: usize, perm: S::u8s) -> (S::u8s, S::u8s) {
        match w {
            1 => a.interleave(b),
            2 => {
                let (lo, hi) = S::u16s::from_bytes(a).interleave(S::u16s::from_bytes(b));
                (lo.to_bytes(), hi.to_bytes())
            }
            4 => {
                let (lo, hi) = S::u32s::from_bytes(a).interleave(S::u32s::from_bytes(b));
                (lo.to_bytes(), hi.to_bytes())
            }
            8 => {
                let (lo, hi) = S::u64s::from_bytes(a).interleave(S::u64s::from_bytes(b));
                (lo.to_bytes(), hi.to_bytes())
            }
            _ => {
                let (lo, hi) = S::u64s::from_bytes(a).interleave(S::u64s::from_bytes(b));
                (
                    lo.to_bytes().swizzle_dyn(perm),
                    hi.to_bytes().swizzle_dyn(perm),
                )
            }
        }
    }

    /// The permutation [`zip`] applies after the quadword zip for a unit of
    /// `w` bytes: unit `t`'s `u = w / 8` quadwords of the first operand sit
    /// at even lanes `2 * (t * u + s)`, the second operand's at the odd
    /// lanes after them.
    #[inline(always)]
    fn zip_perm<S: Simd>(simd: S, w: usize) -> S::u8s {
        let u = (w / 8).max(1);
        S::u8s::from_fn(simd, |i| {
            let lane = i / 8;
            let (t, s) = (lane / (2 * u), lane % (2 * u));
            let src = if s < u {
                2 * (t * u + s)
            } else {
                2 * (t * u + s - u) + 1
            };
            (src * 8 + i % 8) as u8
        })
    }

    /// [`super::trans_byte_bitrow`], a plain `8 * elem_size` rows by
    /// `size / 8` columns byte transpose: a vector of columns at a time, the
    /// rows zipped level by level (bytes, then 16-, 32-, 64-bit units, and
    /// on up) until each vector holds whole columns of a group of `g` row
    /// groups, or one column piece when the column is taller than the
    /// vector; the vectors then go out column by column, row group by row
    /// group. Returns the columns done.
    #[simd]
    pub(super) fn trans_byte_bitrow<S: Simd>(
        simd: S,
        input: &[u8],
        out: &mut [u8],
        size: usize,
        elem_size: usize,
    ) -> usize {
        if !matches!(elem_size, 1 | 2 | 4 | 8) {
            return 0;
        }
        let n = S::u8s::LEN;
        let nbyte_row = size / 8;
        let nrows = 8 * elem_size;
        let perms = [zip_perm(simd, 16), zip_perm(simd, 32)];
        let mut a = [S::u8s::splat(simd, 0); 64];
        let mut b = [S::u8s::splat(simd, 0); 64];
        let mut ii = 0;
        while ii + n <= nbyte_row {
            for (k, x) in a[..nrows].iter_mut().enumerate() {
                let s = k * nbyte_row + ii;
                *x = S::u8s::from_slice(simd, &input[s..s + n]);
            }
            let (mut src, mut dst) = (&mut a, &mut b);
            let (mut g, mut c, mut w) = (nrows, 1, 1);
            while g > 1 && w < n {
                let perm = perms[usize::from(w == 32)];
                for m in 0..g / 2 {
                    for j in 0..c {
                        let (lo, hi) =
                            zip::<S>(src[2 * m * c + j], src[(2 * m + 1) * c + j], w, perm);
                        dst[2 * m * c + 2 * j] = lo;
                        dst[2 * m * c + 2 * j + 1] = hi;
                    }
                }
                std::mem::swap(&mut src, &mut dst);
                g /= 2;
                c *= 2;
                w *= 2;
            }
            for j in 0..c {
                for rg in 0..g {
                    let o = ii * nrows + (j * g + rg) * n;
                    src[rg * c + j].store_slice(&mut out[o..o + n]);
                }
            }
            ii += n;
        }
        ii
    }

    /// [`super::shuffle_bit_eightelem`]: a vector of bytes at a time. When a
    /// vector sits inside one eight-element group its bit rows are runs of
    /// consecutive quadwords, one copy per row; when it covers whole groups
    /// the output bytes are a permutation of the transposed bytes, one byte
    /// shuffle and one store. Both need the group and the vector to divide
    /// one another, so an element size that leaves a vector straddling a
    /// group boundary returns 0. Returns the bytes done.
    #[simd]
    pub(super) fn shuffle_bit_eightelem<S: Simd>(
        simd: S,
        input: &[u8],
        out: &mut [u8],
        nbyte: usize,
        elem_size: usize,
    ) -> usize {
        let n = S::u8s::LEN;
        let q = n / 8;
        let group = 8 * elem_size;
        if group % n != 0 && n % group != 0 {
            return 0;
        }
        let mut ii = 0;
        if group >= n {
            let order = row_order(simd);
            while ii + n <= nbyte {
                let rows = bit_transposed(simd, S::u8s::from_slice(simd, &input[ii..ii + n]))
                    .swizzle_dyn(order);
                let base = ii / group * group + ii % group / 8;
                for (r, bytes) in rows.as_slice().chunks_exact(q).enumerate() {
                    let o = base + r * elem_size;
                    out[o..o + q].copy_from_slice(bytes);
                }
                ii += n;
            }
        } else {
            let order = S::u8s::from_fn(simd, |i| {
                let (t, rem) = (i / group, i % group);
                let (r, e) = (rem / elem_size, rem % elem_size);
                (8 * (t * elem_size + e) + r) as u8
            });
            while ii + n <= nbyte {
                bit_transposed(simd, S::u8s::from_slice(simd, &input[ii..ii + n]))
                    .swizzle_dyn(order)
                    .store_slice(&mut out[ii..ii + n]);
                ii += n;
            }
        }
        ii
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(nbyte: usize) -> Vec<u8> {
        (0..nbyte)
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect()
    }

    /// The per-bit definition of the bitshuffle image: bit plane `bit` of
    /// element `elem` goes to bit `bit * n_elems + elem`, LSB first on both
    /// sides. The three-stage transpose must equal this on every input.
    fn bitshuffle_reference(input: &[u8], elem_size: usize) -> Vec<u8> {
        let n_elems = input.len() / elem_size;
        let mut out = vec![0u8; input.len()];
        for bit in 0..elem_size * 8 {
            for elem in 0..n_elems {
                let src_bit = (input[elem * elem_size + bit / 8] >> (bit % 8)) & 1;
                let dst = bit * n_elems + elem;
                out[dst / 8] |= src_bit << (dst % 8);
            }
        }
        out
    }

    fn byte_transpose_reference(input: &[u8], size: usize, elem_size: usize) -> Vec<u8> {
        let mut out = vec![0u8; size * elem_size];
        for i in 0..size {
            for k in 0..elem_size {
                out[k * size + i] = input[i * elem_size + k];
            }
        }
        out
    }

    const ELEM_SIZES: [usize; 8] = [1, 2, 3, 4, 5, 8, 12, 16];
    const SIZES: [usize; 10] = [8, 16, 24, 64, 128, 136, 520, 1024, 1032, 4096];

    #[test]
    fn three_stage_transpose_matches_the_per_bit_reference() {
        for elem_size in ELEM_SIZES {
            for size in SIZES {
                let input = pattern(size * elem_size);
                let mut got = vec![0u8; input.len()];
                let mut scratch = BitshuffleScratch::new(input.len());
                bitshuffle_block_into(&input, &mut scratch, &mut got, elem_size);
                assert_eq!(
                    got,
                    bitshuffle_reference(&input, elem_size),
                    "elem_size={elem_size} size={size}"
                );
                let mut back = vec![0u8; input.len()];
                bitunshuffle_block_into(&got, &mut scratch, &mut back, elem_size);
                assert_eq!(back, input, "inverse elem_size={elem_size} size={size}");
            }
        }
    }

    #[test]
    fn byte_transpose_matches_the_reference_for_any_element_count() {
        for elem_size in ELEM_SIZES {
            for size in [1usize, 5, 7, 8, 9, 31, 33, 64, 100, 1000, 1031] {
                let input = pattern(size * elem_size);
                let mut got = vec![0u8; input.len()];
                trans_byte_elem(&input, &mut got, size, elem_size);
                let want = byte_transpose_reference(&input, size, elem_size);
                assert_eq!(got, want, "trans elem_size={elem_size} size={size}");
                let mut back = vec![0u8; input.len()];
                untrans_byte_elem(&got, &mut back, size, elem_size);
                assert_eq!(back, input, "untrans elem_size={elem_size} size={size}");
            }
        }
    }

    /// Every kernel, finished by its scalar tail, against the scalar loops
    /// alone, on every level the host offers, for the element sizes the
    /// kernels take and sizes that leave every kind of tail.
    #[cfg(feature = "simd")]
    #[test]
    fn kernels_match_the_scalar_loops_on_every_level() {
        use fearless_simd::Level;
        let top = crate::simd::level();
        let mut levels = vec![top, Level::baseline()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            levels.extend(top.as_avx2().map(Level::Avx2));
            levels.extend(top.as_sse4_2().map(Level::Sse4_2));
            levels.extend(top.as_sse2().map(Level::Sse2));
        }
        for elem_size in [1usize, 2, 3, 4, 8, 16] {
            for size in SIZES {
                let nbyte = size * elem_size;
                let input = pattern(nbyte);
                let mut want = vec![0u8; nbyte];
                let mut got = vec![0u8; nbyte];
                for &level in &levels {
                    let what = format!("{level:?} elem_size={elem_size} size={size}");

                    trans_byte_elem_from(0, &input, &mut want, size, elem_size);
                    got.fill(0);
                    let done = fearless_simd::dispatch!(level, s => simd::trans_byte_elem(s, &input, &mut got, size, elem_size));
                    assert_eq!(done % 8, 0, "{what} trans_byte_elem done");
                    trans_byte_elem_from(done, &input, &mut got, size, elem_size);
                    assert_eq!(got, want, "{what} trans_byte_elem");

                    untrans_byte_elem_from(0, &input, &mut want, size, elem_size);
                    got.fill(0);
                    let done = fearless_simd::dispatch!(level, s => simd::untrans_byte_elem(s, &input, &mut got, size, elem_size));
                    untrans_byte_elem_from(done, &input, &mut got, size, elem_size);
                    assert_eq!(got, want, "{what} untrans_byte_elem");

                    trans_bit_byte_from(0, &input, &mut want, nbyte);
                    got.fill(0);
                    let done = fearless_simd::dispatch!(level, s => simd::trans_bit_byte(s, &input, &mut got, nbyte));
                    assert_eq!(done % 8, 0, "{what} trans_bit_byte done");
                    trans_bit_byte_from(done, &input, &mut got, nbyte);
                    assert_eq!(got, want, "{what} trans_bit_byte");

                    trans_byte_bitrow_from(0, &input, &mut want, size, elem_size);
                    got.fill(0);
                    let done = fearless_simd::dispatch!(level, s => simd::trans_byte_bitrow(s, &input, &mut got, size, elem_size));
                    trans_byte_bitrow_from(done, &input, &mut got, size, elem_size);
                    assert_eq!(got, want, "{what} trans_byte_bitrow");

                    shuffle_bit_eightelem_from(0, &input, &mut want, nbyte, elem_size);
                    got.fill(0);
                    let done = fearless_simd::dispatch!(level, s => simd::shuffle_bit_eightelem(s, &input, &mut got, nbyte, elem_size));
                    assert_eq!(done % 8, 0, "{what} shuffle_bit_eightelem done");
                    shuffle_bit_eightelem_from(done, &input, &mut got, nbyte, elem_size);
                    assert_eq!(got, want, "{what} shuffle_bit_eightelem");
                }
            }
        }
    }
}
