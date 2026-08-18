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
//! the slowest dimension may grow — or be unlimited — without entering the
//! arithmetic. For a fixed (non-unlimited) dataset that slot is always
//! dimension 0. A dataset with one unlimited dimension puts *that* dimension
//! in the slot instead, whichever position it is declared at: libhdf5
//! computes the same address by swizzling the unlimited dimension to the
//! slowest position before linearizing (`H5VM_swizzle_coords`, H5Dearray.c),
//! which amounts to the same "grows without a multiplier" slot this module
//! seeds directly, without materializing a swizzled coordinate array. A
//! dataspace with more than one unlimited dimension has no finite grid at
//! all and is addressed by a v2 B-tree instead — this module rejects it.
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

/// The one dimension whose maximum extent is unlimited, if any — the
/// dimension libhdf5 swizzles to the slowest position before linearizing
/// (`H5VM_swizzle_coords`, H5Dearray.c). More than one has no finite grid at
/// all and belongs to a v2 B-tree instead, so this rejects it rather than
/// picking one arbitrarily.
fn single_unlimited_dim(max_dims: Option<&[u64]>, ndims: usize) -> IoResult<Option<usize>> {
    let mut found = None;
    for d in 0..ndims {
        if is_unlimited(max_dims, d) {
            if let Some(prev) = found {
                return Err(IoError::InvalidState(format!(
                    "dimensions {prev} and {d} are both unlimited: chunks have no \
                     finite linear index with more than one unlimited dimension \
                     (that needs a v2 B-tree index instead)"
                )));
            }
            found = Some(d);
        }
    }
    Ok(found)
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
    // The unlimited dimension (if any) is the slot that grows without a
    // multiplier; a fixed dataset has none, so dimension 0 fills that slot
    // instead — the same convention libhdf5 reaches via an identity swizzle.
    let seed_dim = single_unlimited_dim(max_dims, ndims)?.unwrap_or(0);

    for d in 0..ndims {
        if !is_unlimited(max_dims, d) && coords[d] >= grid[d] {
            return Err(IoError::InvalidState(format!(
                "chunk coordinate {} in dimension {} is outside the chunk grid (0..{})",
                coords[d], d, grid[d]
            )));
        }
    }

    // Horner's method over every dimension but the seed: the seed's own
    // extent is never a multiplier, which is what lets it grow (or be
    // unlimited) without re-indexing chunks already written.
    let mut linear = coords[seed_dim];
    for d in 0..ndims {
        if d == seed_dim {
            continue;
        }
        linear = linear
            .checked_mul(grid[d])
            .and_then(|l| l.checked_add(coords[d]))
            .ok_or_else(|| {
                IoError::InvalidState("chunk coordinates overflow the array index".into())
            })?;
    }
    Ok(linear)
}

/// Grid coordinates of the chunk at row-major `linear` — the inverse of
/// [`linear_index`]. The seed dimension (the unlimited one, or dimension 0
/// for a fixed dataset) takes the leftover quotient, so an index beyond the
/// current extent (a slot written before a shrink, or one that only becomes
/// visible after an extend) still decodes to its true position.
pub(crate) fn coords_of(
    dims: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u64],
    linear: u64,
) -> IoResult<Vec<u64>> {
    let ndims = dims.len();
    let grid = index_grid(dims, max_dims, chunk_dims)?;
    let unlim_dim = single_unlimited_dim(max_dims, ndims)?;
    let mut coords = vec![0u64; ndims];
    coords_into(&grid, unlim_dim, linear, &mut coords)?;
    Ok(coords)
}

/// Grid coordinates of every chunk slot `0..count`, packed row-major as
/// `count * dims.len()` values — [`coords_of`] for a whole index at once,
/// resolving the grid and the seed dimension once for the lot.
///
/// A chunked read decodes the position of every slot its index records, so
/// what `coords_of` spends per slot is what the read spends per chunk; the
/// packed table is one allocation for all of them.
pub(crate) fn coords_table(
    dims: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u64],
    count: usize,
) -> IoResult<Vec<u64>> {
    let ndims = dims.len();
    let grid = index_grid(dims, max_dims, chunk_dims)?;
    let unlim_dim = single_unlimited_dim(max_dims, ndims)?;
    let mut table = vec![0u64; count.saturating_mul(ndims)];
    for (linear, coords) in table.chunks_mut(ndims.max(1)).enumerate() {
        coords_into(&grid, unlim_dim, linear as u64, &mut coords[..ndims])?;
    }
    Ok(table)
}

/// Write the grid coordinates of chunk slot `linear` into `coords`, against a
/// grid and unlimited dimension already resolved. The arithmetic both
/// [`coords_of`] and [`coords_table`] read through, so a slot decodes the same
/// way however many of them a caller asks for at once.
fn coords_into(
    grid: &[u64],
    unlim_dim: Option<usize>,
    linear: u64,
    coords: &mut [u64],
) -> IoResult<()> {
    let ndims = coords.len();
    let seed_dim = unlim_dim.unwrap_or(0);
    let mut rem = linear;
    for d in (0..ndims).rev() {
        if d == seed_dim {
            continue;
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
        if unlim_dim.is_none() && rem >= grid[seed_dim] {
            return Err(IoError::InvalidState(format!(
                "chunk index {linear} is outside the chunk grid"
            )));
        }
        coords[seed_dim] = rem;
    } else if rem != 0 {
        return Err(IoError::InvalidState(format!(
            "chunk index {linear} is outside the chunk grid"
        )));
    }
    Ok(())
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

    /// An unlimited dimension anywhere but 0 is the seed instead — the same
    /// slot libhdf5 reaches by swizzling it to the slowest position
    /// (`H5VM_swizzle_coords`, H5Dearray.c) — so its coordinate is unbounded
    /// and every *other* dimension enters the multiplication in its own
    /// original order. dims[1] is unlimited here, with grid = [2, _, 3], so
    /// `linear` is `coords[1] * 2 * 3 + coords[0] * 3 + coords[2]` — a
    /// hand-derived swizzle of [1, 0, 2].
    #[test]
    fn unlimited_inner_dimension_is_indexable() {
        let dims = [4, 5, 6];
        let max = [4, u64::MAX, 6];
        let chunks = [2, 3, 2];
        assert_eq!(
            linear_index(&dims, Some(&max), &chunks, &[1, 2, 1]).unwrap(),
            16
        );
        assert_eq!(
            coords_of(&dims, Some(&max), &chunks, 16).unwrap(),
            vec![1, 2, 1]
        );
        // The unlimited dimension's coordinate is unbounded, exactly like an
        // unlimited dimension 0.
        assert_eq!(
            linear_index(&dims, Some(&max), &chunks, &[0, 100, 0]).unwrap(),
            600
        );
        assert_eq!(
            coords_of(&dims, Some(&max), &chunks, 600).unwrap(),
            vec![0, 100, 0]
        );
    }

    /// Two unlimited dimensions have no finite grid at all — that shape
    /// belongs to a v2 B-tree index, which never calls into this module.
    #[test]
    fn two_unlimited_dimensions_are_rejected() {
        let dims = [4, 5];
        let max = [u64::MAX, u64::MAX];
        let chunks = [2, 3];
        let err = linear_index(&dims, Some(&max), &chunks, &[0, 0]).unwrap_err();
        assert!(err.to_string().contains("both unlimited"), "{err}");
        let err = coords_of(&dims, Some(&max), &chunks, 0).unwrap_err();
        assert!(err.to_string().contains("both unlimited"), "{err}");
    }
}
