//! Row-major chunk linear indexing against the maximum-extent grid.
//!
//! libhdf5 indexes a chunk by its row-major position in the chunk grid of the
//! *maximum* dataspace extent, never the current one: `H5D__chunk_set_info_real`
//! (H5Dchunk.c) computes `max_chunks[d] = ceil(max_dims[d] / chunk_dims[d])`
//! and every linearly-addressed index — fixed array (H5Dfarray.c), extensible
//! array (H5Dearray.c), implicit (H5Dnone.c) — feeds `max_down_chunks` to
//! `H5VM_array_offset_pre`. That makes a chunk's index permanent: extending
//! the dataset never re-addresses chunks already written.
//!
//! Row-major offsets never multiply by the slowest dimension's own extent, so
//! dimension 0 may grow — or be unlimited — without entering the arithmetic.
//! An *unlimited* dimension other than 0 has no finite multiplier; libhdf5
//! handles it by swizzling that dimension to the slowest position
//! (H5Dearray.c), which this crate does not implement, so such coordinates
//! are rejected here and such dataspaces are rejected at dataset create.
//!
//! This module is the single owner of that arithmetic for the writer, the
//! reader, and the high-level dataset API.

use crate::io::{IoError, IoResult};

/// Chunk count along each dimension of the *index* grid: the maximum extent
/// where one is declared (absent maximum means the shape is fixed, so the
/// current extent is the maximum), the current extent for an unlimited
/// dimension. Rejects a zero chunk dimension and a rank mismatch.
pub(crate) fn index_grid(
    dims: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u64],
) -> IoResult<Vec<u64>> {
    let ndims = dims.len();
    if chunk_dims.len() != ndims {
        return Err(IoError::InvalidState(format!(
            "dataset chunk shape has {} dimensions but the dataspace has {}",
            chunk_dims.len(),
            ndims
        )));
    }
    if let Some(max) = max_dims {
        if max.len() != ndims {
            return Err(IoError::InvalidState(format!(
                "dataset maximum shape has {} dimensions but the dataspace has {}",
                max.len(),
                ndims
            )));
        }
    }
    let mut grid = Vec::with_capacity(ndims);
    for d in 0..ndims {
        if chunk_dims[d] == 0 {
            return Err(IoError::InvalidState(format!(
                "chunk dimension {d} is zero"
            )));
        }
        let extent = match max_dims {
            Some(max) if max[d] != u64::MAX => max[d],
            _ => dims[d],
        };
        grid.push(extent.div_ceil(chunk_dims[d]));
    }
    Ok(grid)
}

/// Whether dimension `d`'s maximum extent is unlimited.
fn is_unlimited(max_dims: Option<&[u64]>, d: usize) -> bool {
    max_dims.is_some_and(|m| m[d] == u64::MAX)
}

/// Row-major linear index of the chunk at grid `coords` — the slot an
/// extensible or fixed array records the chunk under.
///
/// Bounds every coordinate by the index grid (an unlimited dimension is
/// unbounded); an out-of-grid coordinate on a bounded dimension would
/// otherwise silently alias another chunk's slot.
pub(crate) fn linear_index(
    dims: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u64],
    coords: &[u64],
) -> IoResult<u64> {
    let ndims = dims.len();
    if coords.len() != ndims {
        return Err(IoError::InvalidState(format!(
            "chunk_coords has {} entries but the dataset has {} dimensions",
            coords.len(),
            ndims
        )));
    }
    let grid = index_grid(dims, max_dims, chunk_dims)?;
    let mut linear = 0u64;
    for d in 0..ndims {
        if is_unlimited(max_dims, d) {
            if d != 0 {
                return Err(IoError::InvalidState(format!(
                    "unlimited dimension {d} is not the first: its chunks have \
                     no fixed linear index (extensible-array swizzling is not \
                     supported)"
                )));
            }
        } else if coords[d] >= grid[d] {
            return Err(IoError::InvalidState(format!(
                "chunk coordinate {} in dimension {} is outside the chunk grid (0..{})",
                coords[d], d, grid[d]
            )));
        }
        // The multiplier for coords[d] is the grid extent of the dimensions
        // after it; dimension 0's own extent is never multiplied in, which is
        // what lets it grow without re-indexing.
        linear = if d == 0 {
            coords[0]
        } else {
            linear
                .checked_mul(grid[d])
                .and_then(|l| l.checked_add(coords[d]))
                .ok_or_else(|| {
                    IoError::InvalidState("chunk coordinates overflow the array index".into())
                })?
        };
    }
    Ok(linear)
}

/// Grid coordinates of the chunk at row-major `linear` — the inverse of
/// [`linear_index`]. Dimension 0 takes the leftover quotient, so an index
/// beyond the current extent (a slot written before a shrink, or one that
/// only becomes visible after an extend) still decodes to its true position.
pub(crate) fn coords_of(
    dims: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u64],
    linear: u64,
) -> IoResult<Vec<u64>> {
    let ndims = dims.len();
    let grid = index_grid(dims, max_dims, chunk_dims)?;
    let mut coords = vec![0u64; ndims];
    let mut rem = linear;
    for d in (1..ndims).rev() {
        if is_unlimited(max_dims, d) {
            return Err(IoError::InvalidState(format!(
                "unlimited dimension {d} is not the first: its chunks have \
                 no fixed linear index (extensible-array swizzling is not \
                 supported)"
            )));
        }
        if grid[d] == 0 {
            return Err(IoError::InvalidState(format!(
                "chunk grid is empty in dimension {d}"
            )));
        }
        coords[d] = rem % grid[d];
        rem /= grid[d];
    }
    if ndims > 0 {
        if !is_unlimited(max_dims, 0) && rem >= grid[0] {
            return Err(IoError::InvalidState(format!(
                "chunk index {linear} is outside the chunk grid"
            )));
        }
        coords[0] = rem;
    } else if rem != 0 {
        return Err(IoError::InvalidState(format!(
            "chunk index {linear} is outside the chunk grid"
        )));
    }
    Ok(coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The multiplier for every coordinate comes from the maximum extent,
    /// so an index assigned while the dataset was small stays valid after
    /// an extend — the libhdf5 `max_down_chunks` rule.
    #[test]
    fn indices_come_from_the_maximum_grid() {
        // dims [4,3], max [10,9], chunks [2,3]: max grid is [5,3].
        let dims = [4, 3];
        let max = [10, 9];
        let chunks = [2, 3];
        let li = |c: &[u64]| linear_index(&dims, Some(&max), &chunks, c).unwrap();
        assert_eq!(li(&[0, 0]), 0);
        assert_eq!(li(&[0, 2]), 2);
        assert_eq!(li(&[1, 0]), 3); // 3 columns of chunks in the MAX grid, not 1
        assert_eq!(li(&[4, 2]), 14);
        assert_eq!(
            coords_of(&dims, Some(&max), &chunks, 14).unwrap(),
            vec![4, 2]
        );
    }

    /// Without a stored maximum the shape is fixed, so the current extent is
    /// the maximum and the grids coincide.
    #[test]
    fn absent_maximum_means_the_current_extent() {
        let dims = [4, 4];
        let chunks = [2, 2];
        assert_eq!(linear_index(&dims, None, &chunks, &[1, 1]).unwrap(), 3);
        assert_eq!(coords_of(&dims, None, &chunks, 3).unwrap(), vec![1, 1]);
        let err = linear_index(&dims, None, &chunks, &[2, 0]).unwrap_err();
        assert!(err.to_string().contains("outside the chunk grid"));
    }

    /// Dimension 0 never enters the multiplication, so an unlimited first
    /// dimension is indexable and its coordinate is unbounded; decoding an
    /// index beyond the current extent recovers the true position.
    #[test]
    fn unlimited_dimension_zero_is_unbounded() {
        let dims = [4, 8];
        let max = [u64::MAX, 8];
        let chunks = [2, 4];
        assert_eq!(
            linear_index(&dims, Some(&max), &chunks, &[100, 1]).unwrap(),
            201
        );
        assert_eq!(
            coords_of(&dims, Some(&max), &chunks, 201).unwrap(),
            vec![100, 1]
        );
    }

    /// An unlimited dimension other than 0 has no finite multiplier; libhdf5
    /// swizzles it to the slowest position, which this crate does not
    /// implement, so both directions reject it.
    #[test]
    fn unlimited_inner_dimension_is_rejected() {
        let dims = [4, 0];
        let max = [4, u64::MAX];
        let chunks = [2, 2];
        let err = linear_index(&dims, Some(&max), &chunks, &[0, 0]).unwrap_err();
        assert!(err.to_string().contains("not the first"));
        let err = coords_of(&dims, Some(&max), &chunks, 0).unwrap_err();
        assert!(err.to_string().contains("not the first"));
    }
}
