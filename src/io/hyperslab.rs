//! Hyperslab (strideless) selection geometry shared by the reader and writer.
//!
//! Both directions of an N-dimensional hyperslab transfer — reading a slice
//! out of a contiguous/chunked dataset and writing a slice into a contiguous
//! dataset — decompose the selection into the same set of maximal contiguous
//! byte-runs. [`for_each_contiguous_run`] is the single owner of that
//! geometry, so the "one transfer per last-axis row" defect cannot reappear in
//! one path while another is fixed.

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
    let ndims = dims.len();
    debug_assert_eq!(starts.len(), ndims);
    debug_assert_eq!(counts.len(), ndims);
    if ndims == 0 {
        return Ok(());
    }
    let strides = compute_strides(dims, element_size);
    // Largest fully-selected trailing block: walk inward while the dimension
    // just inside the run boundary is fully selected. `m` ends as the
    // outermost dimension folded into a single run; dims `(m, ndims)` are all
    // full, dim `m` may be partial and forms the run's outer stride.
    let mut m = ndims - 1;
    while m > 0 && counts[m] == dims[m] {
        m -= 1;
    }
    let run_elems: u64 = counts[m..].iter().product();
    let run_bytes = (run_elems * element_size) as usize;
    // Offset of the run's first element within dims `[m, ndims)` is constant
    // across outer iterations (dim `m` starts at `starts[m]`, deeper dims at 0
    // since they are full).
    let inner_base: u64 = (m..ndims).map(|d| starts[d] * strides[d]).sum();
    let n_outer: u64 = counts[..m].iter().product(); // empty product == 1
    let mut coords = vec![0u64; m];
    for outer in 0..n_outer {
        let mut src_off = inner_base;
        for d in 0..m {
            src_off += (starts[d] + coords[d]) * strides[d];
        }
        f(src_off, outer as usize * run_bytes, run_bytes)?;
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
