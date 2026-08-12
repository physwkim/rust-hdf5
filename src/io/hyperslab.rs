//! Hyperslab (strideless) selection geometry shared by the reader and writer.
//!
//! Every N-dimensional hyperslab transfer — reading a slice out of a
//! contiguous/chunked dataset, writing a slice into a contiguous dataset, and
//! scattering a slice across a chunked dataset's chunk buffers — decomposes
//! the selection into the same set of maximal contiguous byte-runs.
//! [`for_each_dual_run`] is the single owner of that geometry, so the "one
//! transfer per last-axis row" defect cannot reappear in one path while
//! another is fixed. [`for_each_contiguous_run`] is the special case where the
//! second array *is* the selection.

use crate::io::IoResult;

/// Compute row-major byte strides for an N-dimensional array.
pub(crate) fn compute_strides(dims: &[u64], element_size: u64) -> Vec<u64> {
    let ndims = dims.len();
    if ndims == 0 {
        return vec![];
    }
    let mut strides = vec![0u64; ndims];
    strides[ndims - 1] = element_size;
    for d in (0..ndims - 1).rev() {
        strides[d] = strides[d + 1] * dims[d + 1];
    }
    strides
}

/// Visit the maximal contiguous byte-runs of a strideless hyperslab.
///
/// For a row-major dataset of shape `dims`, the selection `[starts, counts)`
/// is laid out on disk as a set of contiguous runs. A trailing dimension that
/// is *fully* selected (`counts[d] == dims[d]`) coalesces with the dimension
/// above it into one run, because advancing the outer coordinate by one lands
/// exactly on the next run with no gap. This finds the largest such run and
/// calls `f(src_off, out_off, len)` once per run:
///
/// - `src_off` — byte offset of the run within the full row-major dataset
///   (for a contiguous read, add the dataset's base address; for an in-memory
///   full-dataset buffer, index directly; for a contiguous write, it is the
///   destination file offset relative to the base address),
/// - `out_off` — byte offset of the run within the row-major buffer shaped
///   like `counts` (the slice output when reading, or the caller's source
///   data when writing),
/// - `len` — run length in bytes.
///
/// Runs are visited in output order, so `out_off` advances by `len` each call.
pub(crate) fn for_each_contiguous_run(
    dims: &[u64],
    starts: &[u64],
    counts: &[u64],
    element_size: u64,
    mut f: impl FnMut(u64, usize, usize) -> IoResult<()>,
) -> IoResult<()> {
    // The selection buffer is exactly `counts`-shaped and read from its
    // origin, which is what makes this the degenerate case of the dual walk.
    let src_starts = vec![0u64; counts.len()];
    for_each_dual_run(
        dims,
        starts,
        counts,
        &src_starts,
        counts,
        element_size,
        |dst_off, src_off, len| f(dst_off, src_off as usize, len),
    )
}

/// Visit the maximal contiguous byte-runs shared by the *same* logical region
/// as it sits in two differently-shaped row-major arrays.
///
/// The region is `counts` elements wide, located at `dst_starts` within an
/// array of shape `dst_dims` and at `src_starts` within an array of shape
/// `src_dims`. `f(dst_off, src_off, len)` is called once per run with the byte
/// offsets of that run in each array.
///
/// A run may only coalesce across a dimension that is fully selected in
/// **both** arrays: a trailing dimension that is full on one side but partial
/// on the other is contiguous there and strided here, so the shorter run wins.
/// This is what lets one walker serve a whole-dataset transfer (where the
/// second array is the selection itself, see [`for_each_contiguous_run`]) and a
/// chunk scatter (where the second array is one chunk of the grid) without
/// either path re-deriving stride arithmetic.
pub(crate) fn for_each_dual_run(
    dst_dims: &[u64],
    dst_starts: &[u64],
    src_dims: &[u64],
    src_starts: &[u64],
    counts: &[u64],
    element_size: u64,
    mut f: impl FnMut(u64, u64, usize) -> IoResult<()>,
) -> IoResult<()> {
    let ndims = counts.len();
    debug_assert_eq!(dst_dims.len(), ndims);
    debug_assert_eq!(dst_starts.len(), ndims);
    debug_assert_eq!(src_dims.len(), ndims);
    debug_assert_eq!(src_starts.len(), ndims);
    if ndims == 0 || counts.contains(&0) {
        return Ok(());
    }
    let dst_strides = compute_strides(dst_dims, element_size);
    let src_strides = compute_strides(src_dims, element_size);
    // Largest fully-selected trailing block: walk inward while the dimension
    // just inside the run boundary is fully selected on both sides. `m` ends as
    // the outermost dimension folded into a single run; dims `(m, ndims)` are
    // full in both arrays, dim `m` may be partial and forms the run's outer
    // stride.
    let mut m = ndims - 1;
    while m > 0 && counts[m] == dst_dims[m] && counts[m] == src_dims[m] {
        m -= 1;
    }
    let run_elems: u64 = counts[m..].iter().product();
    let run_bytes = (run_elems * element_size) as usize;
    // Offset of the run's first element within dims `[m, ndims)` is constant
    // across outer iterations (dim `m` starts at its start coordinate, deeper
    // dims at 0 since a fully-selected dimension must start at 0).
    let dst_base: u64 = (m..ndims).map(|d| dst_starts[d] * dst_strides[d]).sum();
    let src_base: u64 = (m..ndims).map(|d| src_starts[d] * src_strides[d]).sum();
    let n_outer: u64 = counts[..m].iter().product(); // empty product == 1
    let mut coords = vec![0u64; m];
    for _ in 0..n_outer {
        let mut dst_off = dst_base;
        let mut src_off = src_base;
        for d in 0..m {
            dst_off += (dst_starts[d] + coords[d]) * dst_strides[d];
            src_off += (src_starts[d] + coords[d]) * src_strides[d];
        }
        f(dst_off, src_off, run_bytes)?;
        for d in (0..m).rev() {
            coords[d] += 1;
            if coords[d] < counts[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect `(dst_off, src_off, len)` for every run of a dual walk.
    fn dual(
        dst_dims: &[u64],
        dst_starts: &[u64],
        src_dims: &[u64],
        src_starts: &[u64],
        counts: &[u64],
        es: u64,
    ) -> Vec<(u64, u64, usize)> {
        let mut runs = Vec::new();
        for_each_dual_run(
            dst_dims,
            dst_starts,
            src_dims,
            src_starts,
            counts,
            es,
            |d, s, l| {
                runs.push((d, s, l));
                Ok(())
            },
        )
        .unwrap();
        runs
    }

    #[test]
    fn dual_run_coalesces_when_trailing_dim_is_full_on_both_sides() {
        // A 2x6 region occupying whole rows of both a 4x6 array and a 2x6 one:
        // both sides are contiguous end to end, so it is a single run.
        let runs = dual(&[4, 6], &[2, 0], &[2, 6], &[0, 0], &[2, 6], 4);
        assert_eq!(runs, vec![(48, 0, 48)]);
    }

    #[test]
    fn dual_run_splits_when_trailing_dim_is_partial_on_the_source() {
        // Trailing dim is full in the destination (6 of 6) but only part of the
        // 12-wide source, so the source stride breaks the run.
        let runs = dual(&[4, 6], &[1, 0], &[4, 12], &[1, 3], &[2, 6], 4);
        assert_eq!(runs, vec![(24, 60, 24), (48, 108, 24)]);
    }

    #[test]
    fn dual_run_splits_when_trailing_dim_is_partial_on_the_destination() {
        // Mirror of the previous case: full on the source side, partial on the
        // destination side. The run must still break.
        let runs = dual(&[4, 12], &[1, 3], &[4, 6], &[1, 0], &[2, 6], 4);
        assert_eq!(runs, vec![(60, 24, 24), (108, 48, 24)]);
    }

    #[test]
    fn dual_run_walks_three_dimensions() {
        // 3-D selection whose innermost dim is full on both sides but whose
        // middle dim is not: one run per (outer, middle) pair.
        let runs = dual(
            &[2, 4, 3],
            &[0, 1, 0],
            &[2, 2, 3],
            &[0, 0, 0],
            &[2, 2, 3],
            2,
        );
        assert_eq!(runs, vec![(6, 0, 12), (30, 12, 12)]);
    }

    #[test]
    fn empty_selection_visits_no_runs() {
        assert!(dual(&[4, 6], &[0, 0], &[4, 6], &[0, 0], &[0, 6], 4).is_empty());
        assert!(dual(&[4, 6], &[0, 0], &[4, 6], &[0, 0], &[2, 0], 4).is_empty());
    }

    #[test]
    fn contiguous_run_is_the_dual_walk_against_the_selection_itself() {
        // The wrapper must agree with the general walker for every case the
        // old single-array implementation covered.
        for (dims, starts, counts) in [
            (vec![4u64, 6], vec![1u64, 2], vec![2u64, 3]),
            (vec![4, 6], vec![2, 0], vec![2, 6]),
            (vec![5, 3, 2], vec![1, 0, 0], vec![3, 3, 2]),
            (vec![7], vec![2], vec![4]),
        ] {
            let mut got = Vec::new();
            for_each_contiguous_run(&dims, &starts, &counts, 4, |dst, src, len| {
                got.push((dst, src as u64, len));
                Ok(())
            })
            .unwrap();
            let zeros = vec![0u64; counts.len()];
            assert_eq!(got, dual(&dims, &starts, &counts, &zeros, &counts, 4));
            // The selection buffer is filled exactly once, front to back.
            let total: usize = got.iter().map(|r| r.2).sum();
            assert_eq!(total as u64, counts.iter().product::<u64>() * 4);
        }
    }
}
