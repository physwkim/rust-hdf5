//! SWMR (single-writer / multi-reader) protocol.
//!
//! Implements ordered flush semantics:
//! 1. Write chunk data -> fsync
//! 2. Update extensible array (new chunk address) -> fsync
//! 3. Update dataset object header (new dataspace dims) -> fsync
//! 4. Update superblock (new EOF) -> fsync

use std::collections::HashMap;
use std::path::Path;

use crate::format::messages::datatype::DatatypeMessage;
use crate::format::superblock::{FLAG_SWMR_WRITE, FLAG_WRITE_ACCESS};

use crate::io::writer::Hdf5Writer;
use crate::io::IoResult;

/// Per-dataset accumulation state for a streaming dataset whose chunks span
/// more than one frame (`chunk[0] > 1`, i.e. NDFileHDF5 `nFramesChunks`).
/// `append_frame` buffers whole frames here until a chunk band is full.
struct BandBuffer {
    /// Frames per chunk along the frame axis, `chunk[0]`.
    frames_per_chunk: u64,
    /// Per-frame (spatial) dimensions.
    frame_dims: Vec<u64>,
    /// Per-frame tile dimensions, `chunk[1..]`.
    tile_dims: Vec<u64>,
    /// Element size in bytes.
    elem_size: usize,
    /// Whole frames accumulated for the current (not-yet-full) band.
    frames: Vec<Vec<u8>>,
}

/// SWMR writer wrapping an Hdf5Writer.
///
/// After calling `start_swmr()`, each `append_frame()` writes a chunk and
/// updates the index structures with ordered flushes.
pub struct SwmrWriter {
    writer: Hdf5Writer,
    swmr_active: bool,
    /// Band buffers keyed by dataset index, for streaming datasets created
    /// with `chunk[0] > 1`.
    band_buffers: HashMap<usize, BandBuffer>,
}

impl SwmrWriter {
    /// Create a new HDF5 file configured for SWMR using the env-var-derived
    /// locking policy.
    pub fn create(path: &Path) -> IoResult<Self> {
        let writer = Hdf5Writer::create(path)?;
        Ok(Self {
            writer,
            swmr_active: false,
            band_buffers: HashMap::new(),
        })
    }

    /// Create a new HDF5 file configured for SWMR with an explicit locking
    /// policy. The writer takes an exclusive lock initially; once
    /// [`Self::start_swmr`] is called, the lock is downgraded to shared so
    /// concurrent SWMR readers can attach.
    pub fn create_with_locking(
        path: &Path,
        locking: crate::io::locking::FileLocking,
    ) -> IoResult<Self> {
        let writer = Hdf5Writer::create_with_locking(path, locking)?;
        Ok(Self {
            writer,
            swmr_active: false,
            band_buffers: HashMap::new(),
        })
    }

    /// Reopen a cleanly-closed HDF5 file to resume SWMR streaming.
    ///
    /// Existing datasets are reconstructed so `append_frame` can continue
    /// extending them after a fresh `start_swmr`. Multi-frame-chunk datasets
    /// (`chunk[0] > 1`) are reopened read-as-is; appending to one is rejected
    /// because its final partial band was already zero-padded at the original
    /// close.
    pub fn open_append(path: &Path) -> IoResult<Self> {
        let writer = Hdf5Writer::open_append(path)?;
        Ok(Self {
            writer,
            swmr_active: false,
            band_buffers: HashMap::new(),
        })
    }

    /// Reopen a cleanly-closed HDF5 file to resume SWMR streaming with an
    /// explicit locking policy. See [`Self::open_append`].
    pub fn open_append_with_locking(
        path: &Path,
        locking: crate::io::locking::FileLocking,
    ) -> IoResult<Self> {
        let writer = Hdf5Writer::open_append_with_locking(path, locking)?;
        Ok(Self {
            writer,
            swmr_active: false,
            band_buffers: HashMap::new(),
        })
    }

    /// Return the index of a dataset by name, or `None` if absent. Used to
    /// locate datasets reconstructed by [`Self::open_append`] for
    /// [`append_frame`](Self::append_frame).
    pub fn dataset_index(&self, name: &str) -> Option<usize> {
        self.writer.dataset_index(name)
    }

    /// Create a streaming dataset (chunked, unlimited first dim).
    ///
    /// `frame_dims` are the spatial dimensions per frame (e.g., [H, W]).
    /// The dataset will have shape [0, H, W] initially, with chunk = [1, H, W].
    pub fn create_streaming_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
    ) -> IoResult<usize> {
        // Dataset shape: [0, dim1, dim2, ...]
        let mut dims = vec![0u64];
        dims.extend_from_slice(frame_dims);

        // Max dims: [unlimited, dim1, dim2, ...]
        let mut max_dims = vec![u64::MAX];
        max_dims.extend_from_slice(frame_dims);

        // Chunk dims: [1, dim1, dim2, ...]
        let mut chunk_dims = vec![1u64];
        chunk_dims.extend_from_slice(frame_dims);

        self.writer
            .create_chunked_dataset(name, datatype, &dims, &max_dims, &chunk_dims)
    }

    /// Create a streaming dataset whose frames are split into fixed-size
    /// chunk tiles.
    ///
    /// `frame_dims` is the per-frame shape (e.g. `[1024, 1024]`);
    /// `frame_chunk` is the tile shape within a frame (e.g. `[256, 256]`),
    /// of the same rank. The dataset chunk shape becomes
    /// `[1, frame_chunk...]`. This is the equivalent of an area-detector
    /// writer's tiling controls (NDFileHDF5 `nRowChunks` / `nColChunks`):
    /// it changes the partial-read granularity and compression unit, not
    /// the stored data.
    pub fn create_streaming_dataset_tiled(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
        frame_chunk: &[u64],
    ) -> IoResult<usize> {
        validate_frame_chunk(frame_dims, frame_chunk)?;
        let mut dims = vec![0u64];
        dims.extend_from_slice(frame_dims);
        let mut max_dims = vec![u64::MAX];
        max_dims.extend_from_slice(frame_dims);
        let mut chunk_dims = vec![1u64];
        chunk_dims.extend_from_slice(frame_chunk);

        self.writer
            .create_chunked_dataset(name, datatype, &dims, &max_dims, &chunk_dims)
    }

    /// Create a streaming dataset whose frames are compressed with the given
    /// filter pipeline.
    pub fn create_streaming_dataset_compressed(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> IoResult<usize> {
        let mut dims = vec![0u64];
        dims.extend_from_slice(frame_dims);
        let mut max_dims = vec![u64::MAX];
        max_dims.extend_from_slice(frame_dims);
        let mut chunk_dims = vec![1u64];
        chunk_dims.extend_from_slice(frame_dims);

        self.writer.create_chunked_dataset_with_pipeline(
            name,
            datatype,
            &dims,
            &max_dims,
            &chunk_dims,
            pipeline,
        )
    }

    /// Create a compressed streaming dataset whose frames are split into
    /// fixed-size chunk tiles. See [`create_streaming_dataset_tiled`] for
    /// the meaning of `frame_chunk`; each tile is the compression unit.
    ///
    /// [`create_streaming_dataset_tiled`]: Self::create_streaming_dataset_tiled
    pub fn create_streaming_dataset_tiled_compressed(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
        frame_chunk: &[u64],
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> IoResult<usize> {
        validate_frame_chunk(frame_dims, frame_chunk)?;
        let mut dims = vec![0u64];
        dims.extend_from_slice(frame_dims);
        let mut max_dims = vec![u64::MAX];
        max_dims.extend_from_slice(frame_dims);
        let mut chunk_dims = vec![1u64];
        chunk_dims.extend_from_slice(frame_chunk);

        self.writer.create_chunked_dataset_with_pipeline(
            name,
            datatype,
            &dims,
            &max_dims,
            &chunk_dims,
            pipeline,
        )
    }

    /// Create a streaming dataset with full control over the chunk shape,
    /// including the frame axis.
    ///
    /// `chunk` is the complete per-chunk shape, of rank
    /// `frame_dims.len() + 1`: `chunk[0]` is the number of frames per chunk
    /// (the NDFileHDF5 `nFramesChunks` control) and `chunk[1..]` is the
    /// per-frame tile shape (`nRowChunks` / `nColChunks`). When
    /// `chunk[0] > 1`, [`append_frame`](Self::append_frame) buffers whole
    /// frames until a chunk band fills; the final partial band is written
    /// (zero-padded) at [`close`](Self::close). The dataset's logical frame
    /// count always tracks the exact number of frames appended, so a
    /// partial last chunk does not over-extend it.
    pub fn create_streaming_dataset_chunked(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
        chunk: &[u64],
    ) -> IoResult<usize> {
        self.create_streaming_dataset_chunked_inner(name, datatype, frame_dims, chunk, None)
    }

    /// Compressed variant of [`create_streaming_dataset_chunked`]; each
    /// chunk is filtered independently through `pipeline`.
    ///
    /// [`create_streaming_dataset_chunked`]: Self::create_streaming_dataset_chunked
    pub fn create_streaming_dataset_chunked_compressed(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
        chunk: &[u64],
        pipeline: crate::format::messages::filter::FilterPipeline,
    ) -> IoResult<usize> {
        self.create_streaming_dataset_chunked_inner(
            name,
            datatype,
            frame_dims,
            chunk,
            Some(pipeline),
        )
    }

    fn create_streaming_dataset_chunked_inner(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        frame_dims: &[u64],
        chunk: &[u64],
        pipeline: Option<crate::format::messages::filter::FilterPipeline>,
    ) -> IoResult<usize> {
        validate_streaming_chunk(frame_dims, chunk)?;
        let elem_size = datatype.element_size() as usize;
        let mut dims = vec![0u64];
        dims.extend_from_slice(frame_dims);
        let mut max_dims = vec![u64::MAX];
        max_dims.extend_from_slice(frame_dims);

        let idx = match pipeline {
            Some(p) => self
                .writer
                .create_chunked_dataset_with_pipeline(name, datatype, &dims, &max_dims, chunk, p)?,
            None => self
                .writer
                .create_chunked_dataset(name, datatype, &dims, &max_dims, chunk)?,
        };

        // A chunk that spans more than one frame needs frame buffering in
        // `append_frame`; a single-frame chunk is written immediately.
        if chunk[0] > 1 {
            self.band_buffers.insert(
                idx,
                BandBuffer {
                    frames_per_chunk: chunk[0],
                    frame_dims: frame_dims.to_vec(),
                    tile_dims: chunk[1..].to_vec(),
                    elem_size,
                    frames: Vec::new(),
                },
            );
        }
        Ok(idx)
    }

    /// Create a fixed-shape multi-dimensional grid dataset addressed by
    /// chunk coordinates.
    ///
    /// Unlike [`create_streaming_dataset`](Self::create_streaming_dataset),
    /// which grows a single unlimited leading axis via `append_frame`, this
    /// creates a dataset of the full bounded shape `dims` (no unlimited axis)
    /// whose chunks are placed at explicit positions with
    /// [`write_chunk_at`](Self::write_chunk_at). `chunk` is the per-chunk
    /// shape of the same rank. A fully bounded dataspace uses the fixed-array
    /// chunk index — the index libhdf5 requires for a dataset with no
    /// unlimited dimension; an extensible array would make the file
    /// unreadable ("didn't find unlimited dimension"). Returns the dataset
    /// index.
    pub fn create_grid_dataset(
        &mut self,
        name: &str,
        datatype: DatatypeMessage,
        dims: &[u64],
        chunk: &[u64],
    ) -> IoResult<usize> {
        if dims.is_empty() {
            return Err(crate::io::IoError::InvalidState(
                "create_grid_dataset requires at least one dimension".into(),
            ));
        }
        if dims.contains(&0) {
            return Err(crate::io::IoError::InvalidState(
                "grid dataset dims must be non-zero".into(),
            ));
        }
        // Fixed array index (rank/chunk validation happens inside).
        self.writer
            .create_fixed_array_dataset(name, datatype, dims, chunk)
    }

    /// Set the SWMR flag in the superblock.
    ///
    /// This performs a full finalize: writes all dataset object headers, the
    /// root group, and the superblock with SWMR flags. After this call,
    /// readers can open the file in SWMR mode. Subsequent data writes use
    /// in-place header updates via `flush()`.
    pub fn start_swmr(&mut self) -> IoResult<()> {
        self.writer.finalize_for_swmr()?;
        // Release the writer's exclusive lock so concurrent SWMR readers
        // can attach. Note: the SWMR protocol assumes a single writer —
        // the caller is responsible for ensuring no second writer
        // attaches once SWMR mode starts. (Holding a shared lock here
        // would block other writers but breaks subsequent writes on
        // Windows due to LockFileEx semantics.)
        self.writer.handle().release_lock()?;
        self.swmr_active = true;
        Ok(())
    }

    /// Append a frame of data to a streaming dataset.
    ///
    /// This writes the chunk data, updates the extensible array index,
    /// and extends the dataset dimensions. For a tiled streaming dataset
    /// (created via [`create_streaming_dataset_tiled`]) the frame buffer is
    /// split into its chunk tiles, each written as a separate chunk; for a
    /// one-chunk-per-frame dataset the frame is written as a single chunk.
    /// For a dataset created with `chunk[0] > 1` (via
    /// [`create_streaming_dataset_chunked`]) the frame is buffered until a
    /// chunk band fills.
    ///
    /// [`create_streaming_dataset_tiled`]: Self::create_streaming_dataset_tiled
    /// [`create_streaming_dataset_chunked`]: Self::create_streaming_dataset_chunked
    pub fn append_frame(&mut self, ds_index: usize, data: &[u8]) -> IoResult<()> {
        // Multi-frame-chunk datasets buffer whole frames per chunk band.
        if self.band_buffers.contains_key(&ds_index) {
            return self.append_frame_banded(ds_index, data);
        }

        // Current frame count (dim 0).
        let frame_idx = self.writer.ds(ds_index).lock().dataspace.dims[0];

        let dims = self.writer.dataset_dims(ds_index).to_vec();
        let chunk_dims = self
            .writer
            .dataset_chunk_dims(ds_index)
            .ok_or_else(|| {
                crate::io::IoError::InvalidState("append_frame requires a chunked dataset".into())
            })?
            .to_vec();

        // A multi-frame chunk (`chunk[0] > 1`) not tracked in `band_buffers`
        // can only be a dataset reopened via `open_append`: its final partial
        // band was already zero-padded and written at the original close, so
        // frame-aligned appending cannot resume. (Datasets created in this
        // session with `chunk[0] > 1` are routed through `band_buffers` above
        // and never reach here.)
        if chunk_dims[0] > 1 {
            return Err(crate::io::IoError::InvalidState(format!(
                "dataset {ds_index} uses multi-frame chunks (chunk[0] = {}); \
                 appending to it after open_append is not supported",
                chunk_dims[0]
            )));
        }

        // `data` must hold exactly one frame.
        let elem_size = self.writer.ds(ds_index).lock().datatype.element_size() as usize;
        let frame_elems: u64 = dims[1..].iter().product();
        let expected = frame_elems as usize * elem_size;
        if data.len() != expected {
            return Err(crate::io::IoError::InvalidState(format!(
                "append_frame: data is {} bytes, expected {expected} for one frame",
                data.len()
            )));
        }

        // 1. Write the chunk data. The fast path (one chunk == one whole
        // frame) is taken only when the chunk shape exactly equals the
        // frame shape; otherwise the frame is split into chunk tiles,
        // including the case of a chunk larger than the frame, which still
        // produces one zero-padded tile of the full chunk size.
        if chunk_dims[1..] == dims[1..] {
            self.writer.write_chunk(ds_index, frame_idx, data)?;
        } else {
            // Sub-frame tiling: split the row-major frame buffer into its
            // chunk tiles and write each as a separate chunk. The linear
            // chunk index is row-major over the whole chunk grid, so the
            // tiles of frame `f` occupy `f * tiles_per_frame ..` .
            let mut tiles_per_frame = 1u64;
            for d in 1..dims.len() {
                tiles_per_frame *= dims[d].div_ceil(chunk_dims[d].max(1));
            }
            let tiles = split_frame_into_tiles(data, &dims[1..], &chunk_dims[1..], elem_size);
            let base = frame_idx * tiles_per_frame;
            for (i, tile) in tiles.iter().enumerate() {
                self.writer.write_chunk(ds_index, base + i as u64, tile)?;
            }
        }

        // 2. Extend dimensions.
        let mut new_dims = self.writer.ds(ds_index).lock().dataspace.dims.clone();
        new_dims[0] = frame_idx + 1;
        self.writer.extend_dataset(ds_index, &new_dims)?;

        Ok(())
    }

    /// Write one chunk at explicit chunk-grid coordinates of a fixed grid
    /// dataset (see [`create_grid_dataset`](Self::create_grid_dataset)).
    ///
    /// `chunk_coords` are in units of chunks; `data` must be exactly one full
    /// chunk (`product(chunk) * element_size` bytes — edge chunks are
    /// zero-padded by the caller). The logical extent is fixed, so positions
    /// may be written in any order and unwritten positions read back as fill.
    /// Make writes visible to SWMR readers with [`flush`](Self::flush).
    pub fn write_chunk_at(
        &mut self,
        ds_index: usize,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> IoResult<()> {
        if self.writer.ds(ds_index).lock().fixed_array.is_none() {
            return Err(crate::io::IoError::InvalidState(
                "write_chunk_at requires a fixed grid dataset (create_grid_dataset)".into(),
            ));
        }
        self.writer
            .write_chunk_fixed_array(ds_index, chunk_coords, data)
    }

    /// `append_frame` for a dataset whose chunk spans `chunk[0] > 1` frames:
    /// buffer the frame, grow the logical extent by exactly one, and write
    /// the band's chunks once `frames_per_chunk` frames have accumulated.
    fn append_frame_banded(&mut self, ds_index: usize, data: &[u8]) -> IoResult<()> {
        let (frame_bytes, frames_per_chunk) = {
            let bb = &self.band_buffers[&ds_index];
            let elems: u64 = bb.frame_dims.iter().product();
            (elems as usize * bb.elem_size, bb.frames_per_chunk)
        };
        if data.len() != frame_bytes {
            return Err(crate::io::IoError::InvalidState(format!(
                "append_frame: data is {} bytes, expected {frame_bytes} for one frame",
                data.len()
            )));
        }

        self.band_buffers
            .get_mut(&ds_index)
            .expect("band buffer present")
            .frames
            .push(data.to_vec());

        // The logical frame count tracks the exact number appended.
        let mut new_dims = self.writer.ds(ds_index).lock().dataspace.dims.clone();
        let frame_idx = new_dims[0];
        new_dims[0] = frame_idx + 1;
        self.writer.extend_dataset(ds_index, &new_dims)?;

        // A full band is written immediately.
        if self.band_buffers[&ds_index].frames.len() as u64 == frames_per_chunk {
            self.write_band(ds_index)?;
        }
        Ok(())
    }

    /// Assemble and write every chunk of the currently buffered band of a
    /// multi-frame-chunk dataset, then clear the buffer. A band shorter than
    /// `frames_per_chunk` (the final band at close) yields zero-padded
    /// chunks; the dataset's logical extent is not changed.
    fn write_band(&mut self, ds_index: usize) -> IoResult<()> {
        let (frames, n, frame_dims, tile_dims, elem_size) = {
            let bb = self
                .band_buffers
                .get_mut(&ds_index)
                .expect("band buffer present");
            if bb.frames.is_empty() {
                return Ok(());
            }
            (
                std::mem::take(&mut bb.frames),
                bb.frames_per_chunk,
                bb.frame_dims.clone(),
                bb.tile_dims.clone(),
                bb.elem_size,
            )
        };

        let count = frames.len() as u64;
        // Band index: the logical extent already counts every appended
        // frame, so the band starts at `dim0 - count`.
        let dim0 = self.writer.ds(ds_index).lock().dataspace.dims[0];
        let band = (dim0 - count) / n;

        let k = frame_dims.len();
        let grid: Vec<u64> = (0..k)
            .map(|d| frame_dims[d].div_ceil(tile_dims[d]))
            .collect();
        let cells: u64 = grid.iter().product();
        let tile_bytes = tile_dims.iter().product::<u64>() as usize * elem_size;

        // Split each frame into its tiles once (row-major cell order).
        let per_frame_tiles: Vec<Vec<Vec<u8>>> = frames
            .iter()
            .map(|f| split_frame_into_tiles(f, &frame_dims, &tile_dims, elem_size))
            .collect();

        // For each tile-grid cell, assemble the [n, tile...] chunk: frame
        // `s` occupies frame-slot `s`; slots `count..n` stay zero-padded.
        for cell in 0..cells as usize {
            let mut chunk = vec![0u8; n as usize * tile_bytes];
            for (s, frame_tiles) in per_frame_tiles.iter().enumerate() {
                chunk[s * tile_bytes..(s + 1) * tile_bytes].copy_from_slice(&frame_tiles[cell]);
            }
            let linear = band * cells + cell as u64;
            self.writer.write_chunk(ds_index, linear, &chunk)?;
        }
        Ok(())
    }

    /// Write the final partial band of every multi-frame-chunk dataset.
    fn flush_band_buffers(&mut self) -> IoResult<()> {
        let indices: Vec<usize> = self.band_buffers.keys().copied().collect();
        for ds_index in indices {
            self.write_band(ds_index)?;
        }
        Ok(())
    }

    /// Flush with ordered semantics for SWMR safety.
    ///
    /// Performs ordered fsyncs:
    /// 1. Flush EA index structures -> fsync
    /// 2. Re-write dataset object headers in place (updated dataspace) -> fsync
    /// 3. Re-write superblock (updated EOF) -> fsync
    ///
    /// A deleted dataset is skipped by both loops, the rule both finalize
    /// paths already follow: `release_dataset_storage` returned that dataset's
    /// header block and index blocks to the allocator, so writing either back
    /// puts an object header over whatever now owns the block.
    pub fn flush(&mut self) -> IoResult<()> {
        // Step 1: Flush chunk index structures for all chunked datasets —
        // extensible array (streaming), fixed array (fixed grid), and v2
        // B-tree alike, matching finalize_for_swmr.
        for i in 0..self.writer.dataset_count() {
            let is_indexed = {
                let ds = self.writer.ds(i);
                let m = ds.lock();
                !m.deleted
                    && (m.chunked.is_some() || m.fixed_array.is_some() || m.btree_v2.is_some())
            };
            if is_indexed {
                self.writer.flush_dataset(i)?;
            }
        }
        self.writer.handle().sync_data()?;

        if self.swmr_active {
            // Step 2: Re-write dataset object headers in place with updated dims.
            for i in 0..self.writer.dataset_count() {
                let rewrite = {
                    let ds = self.writer.ds(i);
                    let m = ds.lock();
                    !m.deleted && m.obj_header_written_addr.is_some()
                };
                if rewrite {
                    self.writer.write_dataset_header_inplace(i)?;
                }
            }
            self.writer.handle().sync_data()?;

            // Step 3: Re-write superblock with updated EOF.
            self.writer
                .write_superblock(FLAG_WRITE_ACCESS | FLAG_SWMR_WRITE)?;
            self.writer.handle().sync_data()?;
        }

        Ok(())
    }

    /// Provide access to the underlying writer for creating non-streaming datasets.
    pub fn writer_mut(&mut self) -> &mut Hdf5Writer {
        &mut self.writer
    }

    /// Close and finalize the file.
    ///
    /// Any partially filled multi-frame chunk band is written (zero-padded)
    /// before the file is finalized.
    pub fn close(mut self) -> IoResult<()> {
        self.flush_band_buffers()?;
        self.writer.close()
    }
}

/// Reject a streaming chunk shape that does not have rank
/// `frame_dims.len() + 1`, or that (with `frame_dims`) has a zero dimension.
fn validate_streaming_chunk(frame_dims: &[u64], chunk: &[u64]) -> IoResult<()> {
    if chunk.len() != frame_dims.len() + 1 {
        return Err(crate::io::IoError::InvalidState(format!(
            "chunk rank {} must be the frame rank + 1 ({})",
            chunk.len(),
            frame_dims.len() + 1
        )));
    }
    if chunk.contains(&0) {
        return Err(crate::io::IoError::InvalidState(
            "chunk dimensions must be non-zero".into(),
        ));
    }
    if frame_dims.contains(&0) {
        return Err(crate::io::IoError::InvalidState(
            "frame_dims dimensions must be non-zero".into(),
        ));
    }
    Ok(())
}

/// Reject a `frame_chunk` whose rank does not match `frame_dims`, or that
/// has a zero dimension.
fn validate_frame_chunk(frame_dims: &[u64], frame_chunk: &[u64]) -> IoResult<()> {
    if frame_chunk.len() != frame_dims.len() {
        return Err(crate::io::IoError::InvalidState(format!(
            "frame_chunk rank {} does not match frame_dims rank {}",
            frame_chunk.len(),
            frame_dims.len()
        )));
    }
    if frame_chunk.contains(&0) {
        return Err(crate::io::IoError::InvalidState(
            "frame_chunk dimensions must be non-zero".into(),
        ));
    }
    if frame_dims.contains(&0) {
        return Err(crate::io::IoError::InvalidState(
            "frame_dims dimensions must be non-zero".into(),
        ));
    }
    Ok(())
}

/// Row-major linear element offset of `coords` within an array of `dims`.
fn lin_offset(coords: &[u64], dims: &[u64]) -> u64 {
    let mut off = 0u64;
    for d in 0..dims.len() {
        off = off * dims[d] + coords[d];
    }
    off
}

/// Split a row-major frame buffer into its chunk tiles, in row-major chunk
/// grid order. A frame dimension that is not a whole multiple of the tile
/// dimension produces partial edge tiles, which are zero-padded to a full
/// tile (HDF5 chunks are fixed-size; the dataset extent clips them on read).
fn split_frame_into_tiles(
    frame: &[u8],
    frame_dims: &[u64],
    tile_dims: &[u64],
    elem_size: usize,
) -> Vec<Vec<u8>> {
    let k = frame_dims.len();
    let grid: Vec<u64> = (0..k)
        .map(|d| frame_dims[d].div_ceil(tile_dims[d].max(1)))
        .collect();
    let n_tiles: u64 = grid.iter().product();
    let tile_elems: u64 = tile_dims.iter().product();
    // Number of "rows" within a tile: every axis but the last.
    let last = k - 1;
    let tile_rows: u64 = tile_dims[..last].iter().product();
    let run = tile_dims[last] as usize; // contiguous run length along the last axis

    let mut tiles = Vec::with_capacity(n_tiles as usize);
    for t in 0..n_tiles {
        // Grid coordinates of tile `t`.
        let mut tg = vec![0u64; k];
        let mut rem = t;
        for d in (0..k).rev() {
            tg[d] = rem % grid[d];
            rem /= grid[d];
        }
        let mut tile = vec![0u8; tile_elems as usize * elem_size];
        for row in 0..tile_rows {
            // Within-tile coordinates for axes 0..last.
            let mut src = vec![0u64; k];
            let mut r = row;
            let mut oob = false;
            for d in (0..last).rev() {
                let tc = r % tile_dims[d];
                r /= tile_dims[d];
                src[d] = tg[d] * tile_dims[d] + tc;
                if src[d] >= frame_dims[d] {
                    oob = true;
                }
            }
            let last_base = tg[last] * tile_dims[last];
            if oob || last_base >= frame_dims[last] {
                continue;
            }
            src[last] = last_base;
            let copy = run.min((frame_dims[last] - last_base) as usize);
            let src_off = lin_offset(&src, frame_dims) as usize * elem_size;
            let dst_off = row as usize * run * elem_size;
            tile[dst_off..dst_off + copy * elem_size]
                .copy_from_slice(&frame[src_off..src_off + copy * elem_size]);
        }
        tiles.push(tile);
    }
    tiles
}

#[cfg(test)]
mod swmr_hardlink_tests {
    use super::*;
    use crate::format::messages::datatype::DatatypeMessage;
    use crate::io::reader::Hdf5Reader;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique temp directory, removed on drop, so parallel runs cannot collide.
    struct TmpDir(std::path::PathBuf);

    impl TmpDir {
        fn new(label: &str) -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "rh5_swmr_hl_{}_{}_{}",
                label,
                std::process::id(),
                n
            ));
            std::fs::create_dir_all(&dir).unwrap();
            TmpDir(dir)
        }

        fn file(&self) -> std::path::PathBuf {
            self.0.join("stream.h5")
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A hard link created BEFORE `start_swmr` is committed by `start_swmr`
    /// and is visible to readers for the whole SWMR window and after close.
    #[test]
    fn hardlink_before_start_swmr_visible() {
        let dir = TmpDir::new("before");
        let path = dir.file();
        {
            let mut w = SwmrWriter::create(&path).unwrap();
            let idx = w
                .create_streaming_dataset("frames", DatatypeMessage::u8_type(), &[2, 2])
                .unwrap();
            w.writer_mut()
                .create_hard_link("/", "alias", "frames")
                .unwrap();
            w.start_swmr().unwrap();
            w.append_frame(idx, &[1u8, 2, 3, 4]).unwrap();
            w.close().unwrap();
        }
        let mut r = Hdf5Reader::open(&path).unwrap();
        let names: Vec<String> = r.dataset_names().iter().map(|s| s.to_string()).collect();
        assert!(
            names.iter().any(|n| n == "alias"),
            "hard link 'alias' missing from final file: {names:?}"
        );
        assert_eq!(r.read_dataset_raw("frames").unwrap(), vec![1u8, 2, 3, 4]);
        assert_eq!(r.read_dataset_raw("alias").unwrap(), vec![1u8, 2, 3, 4]);
    }

    /// Regression: a hard link created AFTER `start_swmr` is a structural
    /// change made during the SWMR window. `close` must commit it via a full
    /// re-finalize (rebuilding group/root headers and the grown target
    /// header), not reject it with an in-place-rewrite error.
    #[test]
    fn hardlink_after_start_swmr_committed_at_close() {
        let dir = TmpDir::new("after");
        let path = dir.file();
        {
            let mut w = SwmrWriter::create(&path).unwrap();
            let idx = w
                .create_streaming_dataset("frames", DatatypeMessage::u8_type(), &[2, 2])
                .unwrap();
            w.start_swmr().unwrap();
            w.append_frame(idx, &[1u8, 2, 3, 4]).unwrap();
            w.append_frame(idx, &[5u8, 6, 7, 8]).unwrap();
            // Structural change during the SWMR window.
            w.writer_mut()
                .create_hard_link("/", "alias", "frames")
                .unwrap();
            // Before the fix this returned Err("dataset header grew ...
            // cannot rewrite in place") and left the file SWMR-dirty.
            w.close().unwrap();
        }
        let mut r = Hdf5Reader::open(&path).unwrap();
        let names: Vec<String> = r.dataset_names().iter().map(|s| s.to_string()).collect();
        assert!(
            names.iter().any(|n| n == "alias"),
            "hard link created after start_swmr missing from final file: {names:?}"
        );
        // The streaming data must survive the full re-finalize intact.
        let expect = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(r.read_dataset_raw("frames").unwrap(), expect);
        assert_eq!(r.read_dataset_raw("alias").unwrap(), expect);
        assert_eq!(r.dataset_shape("frames").unwrap(), vec![2, 2, 2]);
    }

    /// Regression: `flush` must leave a deleted dataset alone. Both finalize
    /// paths skip one, because `release_dataset_storage` has already returned
    /// its header block to the allocator — a flush that rewrote the header
    /// would stamp an object header over whatever now owns that block.
    ///
    /// The deleted state is set directly rather than through
    /// `delete_dataset`, which refuses while SWMR streaming is active and
    /// clears the header address on the way out: what is under test is the
    /// flush loop's own rule, not the delete path that currently keeps it
    /// from being exercised.
    #[test]
    fn flush_leaves_a_deleted_datasets_header_alone() {
        let dir = TmpDir::new("deleted");
        let path = dir.file();

        let mut w = SwmrWriter::create(&path).unwrap();
        let keep = w
            .create_streaming_dataset("frames", DatatypeMessage::u8_type(), &[2, 2])
            .unwrap();
        let gone = w
            .create_streaming_dataset("scratch", DatatypeMessage::u8_type(), &[2, 2])
            .unwrap();
        w.start_swmr().unwrap();
        w.append_frame(keep, &[1u8, 2, 3, 4]).unwrap();
        w.append_frame(gone, &[9u8, 9, 9, 9]).unwrap();

        // Stand in for the block having been reused: the header address holds
        // bytes that are no longer this dataset's header.
        let (addr, size) = {
            let ds = w.writer_mut().ds(gone);
            let m = ds.lock();
            let &(addr, size) = m.obj_header_blocks.first().expect("header written");
            (addr, size as usize)
        };
        let sentinel = vec![0xABu8; size];
        w.writer_mut().handle().write_at(addr, &sentinel).unwrap();
        w.writer_mut().ds(gone).lock().deleted = true;

        w.flush().unwrap();

        let after = w.writer_mut().handle().read_at(addr, size).unwrap();
        assert_eq!(
            after, sentinel,
            "flush rewrote the object header of a deleted dataset at {addr:#x}"
        );

        w.close().unwrap();
        let mut r = Hdf5Reader::open(&path).unwrap();
        let names: Vec<String> = r.dataset_names().iter().map(|s| s.to_string()).collect();
        assert_eq!(names, vec!["frames".to_string()]);
        assert_eq!(r.read_dataset_raw("frames").unwrap(), vec![1u8, 2, 3, 4]);
    }

    /// A reader attached during the SWMR window must still resolve the file
    /// after `close`, even though the full re-finalize relocates every object
    /// header. `refresh` re-reads the superblock and follows the new root
    /// address, so the relocation is transparent.
    #[test]
    fn reader_refresh_after_close_follows_relocated_headers() {
        use crate::io::locking::FileLocking;

        let dir = TmpDir::new("refresh");
        let path = dir.file();

        let mut w = SwmrWriter::create_with_locking(&path, FileLocking::Disabled).unwrap();
        let idx = w
            .create_streaming_dataset("frames", DatatypeMessage::u8_type(), &[2, 2])
            .unwrap();
        w.start_swmr().unwrap();
        w.append_frame(idx, &[1u8, 2, 3, 4]).unwrap();
        w.flush().unwrap();

        // Reader attaches mid-stream and sees the first frame.
        let mut r = Hdf5Reader::open_swmr_with_locking(&path, FileLocking::Disabled).unwrap();
        assert_eq!(r.dataset_shape("frames").unwrap(), vec![1, 2, 2]);

        // A second frame plus a structural change, then close: the full
        // re-finalize writes every header at a fresh address.
        w.append_frame(idx, &[5u8, 6, 7, 8]).unwrap();
        w.writer_mut()
            .create_hard_link("/", "alias", "frames")
            .unwrap();
        w.close().unwrap();

        // The already-attached reader follows the relocated headers via the
        // superblock and sees the final state, including the new hard link.
        r.refresh().unwrap();
        assert_eq!(r.dataset_shape("frames").unwrap(), vec![2, 2, 2]);
        assert_eq!(
            r.read_dataset_raw("frames").unwrap(),
            vec![1u8, 2, 3, 4, 5, 6, 7, 8]
        );
        let names: Vec<String> = r.dataset_names().iter().map(|s| s.to_string()).collect();
        assert!(
            names.iter().any(|n| n == "alias"),
            "refreshed reader missing hard link: {names:?}"
        );
    }

    /// A group created after `start_swmr` is a structural change; the full
    /// re-finalize at close rebuilds every group/root header, so the group
    /// reaches the final file.
    #[test]
    fn group_created_after_start_swmr_committed_at_close() {
        let dir = TmpDir::new("group_after");
        let path = dir.file();
        {
            let mut w = SwmrWriter::create(&path).unwrap();
            let idx = w
                .create_streaming_dataset("frames", DatatypeMessage::u8_type(), &[2, 2])
                .unwrap();
            w.start_swmr().unwrap();
            w.append_frame(idx, &[1u8, 2, 3, 4]).unwrap();
            w.writer_mut().create_group("/", "entry").unwrap();
            w.close().unwrap();
        }
        let r = Hdf5Reader::open(&path).unwrap();
        assert!(
            r.has_group("entry"),
            "group created after start_swmr missing: {:?}",
            r.group_paths()
        );
    }
}
