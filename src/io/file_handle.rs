use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "mmap")]
use std::sync::Arc;
use std::sync::Mutex;

use crate::io::locking::{self, FileLocking, LockMode};

// The `threadsafe` writer's safety rests on positioned I/O: concurrent
// pread/pwrite at distinct offsets on a shared `&File` never race because each
// call carries its own explicit offset and never consults a shared file cursor.
// (On Windows, `seek_read`/`seek_write` may update the file position as a side
// effect, but since no concurrent positioned op reads that position, distinct
// offsets still cannot race.) On a target that is neither Unix nor
// Windows there is no positioned API, so the pwrite_all/pread fallbacks below
// fall back to seek+read/write on the shared cursor — which IS a data race
// when the writer methods run concurrently through the shared read guard. The
// in-fallback comment assumes "the writer must stay serialized," but under
// `threadsafe` it is not. Rather than let that silent-corruption path be
// reachable, refuse to build the feature where its premise does not hold.
#[cfg(all(feature = "threadsafe", not(any(unix, windows))))]
compile_error!(
    "the `threadsafe` feature requires positioned file I/O (pread/pwrite), \
     which is only available on Unix and Windows targets; on this target the \
     seek-based fallback would race the shared file cursor across threads"
);

/// Wraps `std::fs::File` with positioned (pread / pwrite) I/O convenience
/// methods.
///
/// Every read and write takes an explicit byte offset and never consults a
/// shared file cursor. Positioned operations at distinct offsets on a
/// shared `&File` are safe to issue concurrently — on Windows the cursor may
/// move as a side effect, but nothing consults it — which is what lets the
/// `threadsafe` fine-grained writer read and write chunk data from `&self`
/// without holding a global lock (see
/// `docs/threadsafe-fine-grained-locking.md`).
///
/// There is no application-level read buffer. The structural reads the
/// reader makes are at scattered offsets and already over-read whole structures
/// per call (`read_at_most` grabs e.g. a 64 KiB window), so a `BufReader` —
/// whose buffer is discarded on every non-adjacent `seek` — added little for
/// this access pattern, and a buffer would have blocked the `&self` positioned
/// reads the concurrent write path needs.
///
/// Writes do go through one: [`Accum`], the write accumulator, which is the
/// same trade `H5Faccum.c` makes for the C library's sec2 driver. Metadata
/// leaves the writer in 30-to-500-byte pieces laid down in address order, and
/// one `pwrite` per piece costs far more in syscalls than the memcpy that
/// coalesces them.
pub struct FileHandle {
    file: File,
    /// Where this handle's reads come from. Every read entry point goes
    /// through it; `file` is private, so no caller can reach the descriptor
    /// around it.
    source: ReadSource,
    /// The write accumulator. Only [`write_at`](Self::write_at) puts bytes in
    /// it and only [`flush`](Self::flush) takes them out; `file` is private, so
    /// no caller can reach the descriptor around it.
    accum: Mutex<Accum>,
    /// Whether `accum` holds anything, kept outside the mutex so that a read,
    /// or a write large enough to go straight out, can tell there is nothing
    /// to flush without taking it. That matters under `threadsafe`, where many
    /// threads issue positioned writes at once and must not queue behind one
    /// another on a lock they have no bytes in.
    accum_dirty: AtomicBool,
    /// False for files opened read-only; [`write_at`](Self::write_at) then
    /// errors instead of attempting a write.
    writable: bool,
    /// Active locking policy. Used when downgrading the lock for SWMR.
    /// When the policy is [`FileLocking::Disabled`], no lock was taken.
    lock_policy: FileLocking,
    /// True if a lock is currently held on the underlying file.
    lock_held: bool,
    /// Byte offset of the HDF5 address space within the file: 0 unless the
    /// file has a userblock. Every offset passed to this handle is relative to
    /// it, which is what `H5FD_set_base_addr` does for the C library's
    /// drivers — no caller can forget to add it.
    base: u64,
}

/// Where a [`FileHandle`]'s bytes come from — the one place that decides, so
/// that no read entry point picks for itself and none can be added that skips
/// the choice.
///
/// INVARIANT: a handle is `Mapped` only when it was opened read-only.
/// [`FileHandle::new_read_only`] is the only constructor that installs a map
/// and the only one that passes `writable: false`, and
/// [`FileHandle::refresh_read_source`] can only retake a map on a handle that
/// already has one. So a writable handle is never mapped, and a map can never
/// go stale against the write accumulator: on a mapped handle
/// [`FileHandle::write_at`] refuses before it can stage a byte.
///
/// A mapped handle serves the file as it was when the map was taken, and its
/// [`ReadSource::len`] is the mapped length — so a read past it fails with
/// `UnexpectedEof` exactly as a `pread` past the end of a file of that length
/// does. Bytes a concurrent SWMR writer appends are outside the map until
/// [`FileHandle::refresh_read_source`] retakes it.
enum ReadSource {
    /// Positioned reads (`pread`) against the descriptor.
    Pread,
    /// A read-only map of the whole file as of the moment it was taken.
    /// Reads up to [`MAP_MAX_READ`] come out of it; larger ones still go
    /// through `pread` — see there for why.
    ///
    /// Held behind an `Arc` because a zero-copy view handed to a caller
    /// borrows nothing: it clones this handle, so the pages stay mapped for
    /// as long as the view lives even after
    /// [`FileHandle::refresh_read_source`] has retaken the map or the handle
    /// itself is gone.
    #[cfg(feature = "mmap")]
    Mapped(Arc<memmap2::Mmap>),
}

/// Largest read a mapped handle serves out of its map.
///
/// A mapped read trades one syscall entry for faulting the source pages in.
/// Which side of that trade wins depends on the size of the read and on
/// whether the buffer it lands in is one the caller already owns. Measured on
/// this box (tmpfs, 128 MiB file read through in pieces of the given size,
/// fresh map per round, minimum of eight), as the mapped read's time over the
/// same read through `pread`:
///
/// ```text
/// piece      4 KiB  16 KiB  64 KiB  256 KiB  1 MiB   4 MiB  128 MiB
/// warm dst    0.69    0.84    0.94     0.94   0.94    0.94     0.65
/// cold dst    0.92    1.08    1.12     1.14   1.14    1.13     1.24
/// ```
///
/// "Cold" is a buffer freshly allocated for the read, whose own pages have to
/// be faulted in as they are filled; "warm" is one the reader already had.
/// With a cold destination the fault cost of the destination dominates
/// everything, and the map's source-side faults are pure addition — `pread`
/// reaches the same page cache through the kernel's own huge-page mapping and
/// pays nothing per page for it. So the map only pays off for reads small
/// enough that the syscall is the larger half of the work.
///
/// 8 KiB is where this library's reads divide. Metadata windows, small
/// datasets, and filtered chunk images are at or below it — opening and
/// reading 2000 small datasets is 2000 reads of 1 KiB and 2000 of 8 KiB, and
/// it runs 27% faster mapped. A selected run out of a chunked dataset is
/// 64 KiB and a whole chunk is megabytes; those stay on `pread`, and it
/// matters that they do: at a 64 KiB ceiling the 1000-random-slice read
/// workload lost 7.7%, which this ceiling gives back.
///
/// The ceiling only prices the cold row of the table. A read whose
/// destination the caller keeps — [`ReadDst::Reused`], asserted by the
/// public `*_into` dataset reads — is the warm row, where the map wins at
/// every size, so it is served from the map with no ceiling at all.
#[cfg(feature = "mmap")]
const MAP_MAX_READ: usize = 8 << 10;

/// Where a read's bytes land, as far as the fault cost of the destination
/// is concerned. The table on [`MAP_MAX_READ`] is priced by this fact:
/// `Fresh` is its cold row (the read's own allocation, faulted in as it
/// fills), `Reused` its warm row (a buffer the caller holds across calls,
/// already faulted). Callers state the fact; [`ReadSource::map_for`] alone
/// turns it into a choice.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReadDst {
    /// The destination was allocated for this read.
    Fresh,
    /// The destination is a buffer the caller reuses across reads.
    Reused,
}

impl ReadSource {
    /// The best source a read-only handle can have: a whole-file map when one
    /// can be had, positioned reads otherwise.
    ///
    /// Mapping failure is not an open failure. An empty file (`mmap` rejects a
    /// zero length), a file longer than the address space holds, a filesystem
    /// or platform that refuses `mmap` — each lands on `Pread`, which every
    /// caller already cannot tell from a map.
    fn for_read_only(file: &File) -> Self {
        #[cfg(feature = "mmap")]
        {
            let len = file.metadata().map(|m| m.len()).unwrap_or(0);
            if len > 0 && usize::try_from(len).is_ok() {
                // SAFETY: mapping a file is unsafe because another process
                // can change its bytes, or truncate it out from under the
                // mapping, while it is mapped. The read-only opener took a
                // shared lock first (unless the policy waived it), which is
                // the same protection the C library's `H5FD_lock` gives its
                // drivers, and the reader treats the mapped bytes as a
                // snapshot: nothing here caches a borrowed slice, so a
                // concurrent modification can only be seen or not seen, never
                // half-seen.
                if let Ok(map) = unsafe { memmap2::Mmap::map(file) } {
                    return ReadSource::Mapped(Arc::new(map));
                }
            }
        }
        let _ = file;
        ReadSource::Pread
    }

    /// The map to serve a read of `len` bytes from, or `None` when this read
    /// goes to the descriptor — either because the handle has no map, or
    /// because the read is too big to be worth taking out of one
    /// ([`MAP_MAX_READ`]) for the destination it lands in ([`ReadDst`]).
    /// The one place that decides, so that no read entry point can pick for
    /// itself and none can be added that skips the choice.
    #[cfg(feature = "mmap")]
    fn map_for(&self, len: usize, dst: ReadDst) -> Option<&memmap2::Mmap> {
        match self {
            ReadSource::Mapped(map) if dst == ReadDst::Reused || len <= MAP_MAX_READ => Some(map),
            _ => None,
        }
    }

    /// Bytes this source can serve, counted from the start of the file.
    fn len(&self, file: &File) -> std::io::Result<u64> {
        match self {
            #[cfg(feature = "mmap")]
            ReadSource::Mapped(map) => Ok(map.len() as u64),
            ReadSource::Pread => Ok(file.metadata()?.len()),
        }
    }

    /// Exactly `len` bytes at absolute offset `at`, in a fresh `Vec`.
    ///
    /// The map gets its own copy rather than the `pread` path's
    /// zero-then-fill: the bytes are already there to be copied, and zeroing
    /// a buffer that is about to be overwritten is a second pass over it.
    fn read_vec(&self, file: &File, at: u64, len: usize) -> std::io::Result<Vec<u8>> {
        #[cfg(feature = "mmap")]
        if let Some(map) = self.map_for(len, ReadDst::Fresh) {
            return Ok(mapped_range(map, at, len)?.to_vec());
        }
        let mut buf = vec![0u8; len];
        pread_exact(file, at, &mut buf)?;
        Ok(buf)
    }

    /// Up to `max_len` bytes at absolute offset `at`, stopping at the end of
    /// what the source holds.
    fn read_vec_upto(&self, file: &File, at: u64, max_len: usize) -> std::io::Result<Vec<u8>> {
        #[cfg(feature = "mmap")]
        if let Some(map) = self.map_for(max_len, ReadDst::Fresh) {
            let avail = (map.len() as u64).saturating_sub(at) as usize;
            return Ok(mapped_range(map, at, max_len.min(avail))?.to_vec());
        }
        let mut buf = vec![0u8; max_len];
        let mut total = 0;
        while total < buf.len() {
            match pread(file, at + total as u64, &mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        buf.truncate(total);
        Ok(buf)
    }

    /// Exactly `buf.len()` bytes at absolute offset `at`, straight into
    /// `buf`. Fails with `UnexpectedEof` when the source is too short, which
    /// is what a short `pread` reports.
    fn read_exact_into(
        &self,
        file: &File,
        at: u64,
        buf: &mut [u8],
        dst: ReadDst,
    ) -> std::io::Result<()> {
        #[cfg(feature = "mmap")]
        if let Some(map) = self.map_for(buf.len(), dst) {
            buf.copy_from_slice(mapped_range(map, at, buf.len())?);
            return Ok(());
        }
        let _ = dst;
        pread_exact(file, at, buf)
    }
}

/// `len` bytes of `map` at absolute offset `at`, or the `UnexpectedEof` a
/// `pread` that could not fill its buffer returns.
///
/// Every mapped read comes through here, so a length or address the file
/// itself supplied can only ever produce that error — never an out-of-bounds
/// slice, and never the `SIGBUS` a raw pointer walk off the end of the
/// mapping would take.
#[cfg(feature = "mmap")]
fn mapped_range(map: &memmap2::Mmap, at: u64, len: usize) -> std::io::Result<&[u8]> {
    if len == 0 {
        // A read of nothing succeeds wherever it is asked for, which is what
        // `read_exact_at` does with an empty buffer: it fills it without
        // looking at the offset at all.
        return Ok(&[]);
    }
    let end = at.checked_add(len as u64);
    let range = usize::try_from(at)
        .ok()
        .zip(end.and_then(|e| usize::try_from(e).ok()));
    match range.and_then(|(s, e)| map.get(s..e)) {
        Some(slice) => Ok(slice),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "failed to fill whole buffer",
        )),
    }
}

/// Upper bound on the buffered range, as `H5F_ACCUM_MAX_SIZE` bounds
/// libhdf5's accumulator.
const ACCUM_MAX: usize = 1 << 20;

/// A write of at least this many bytes goes straight to `pwrite`.
///
/// Raw data must not be copied through the buffer — a chunk write is already
/// one large syscall and staging it would only add a pass over memory — but
/// `FileHandle` is handed offsets and bytes, never the `H5FD_mem_t` class that
/// tells libhdf5 metadata from raw data. Size stands in for the class: every
/// metadata structure the writer emits is far below this, and every raw write
/// worth passing through is above it.
#[cfg(not(feature = "threadsafe"))]
const ACCUM_PASSTHROUGH: usize = 8 << 10;

/// Whether a write of `len` bytes is worth staging rather than issuing.
#[cfg(not(feature = "threadsafe"))]
fn stageable(len: usize) -> bool {
    len < ACCUM_PASSTHROUGH
}

/// Never, under `threadsafe`: every write goes straight out.
///
/// Coalescing is a single-writer trade. This build's whole premise is that
/// many threads issue positioned writes at distinct offsets at once
/// (`docs/threadsafe-fine-grained-locking.md`), and one shared run is exactly
/// the queue that design exists to remove — two threads writing unrelated
/// datasets would take turns flushing each other's bytes, and a run bridged
/// across a gap would read back and rewrite bytes another thread owns.
/// libhdf5 can accumulate because its accumulator sits behind the global API
/// lock; this build has no such lock, so it keeps the behaviour it had before
/// the accumulator existed.
#[cfg(feature = "threadsafe")]
fn stageable(_len: usize) -> bool {
    false
}

/// Widest gap the accumulator will bridge to keep one run going.
///
/// The writer aligns each object header block, so consecutive headers land a
/// few bytes apart with the slack in between never written. Bridging that
/// slack is what turns 2000 header writes into one; a page bounds how much of
/// the file a bridge can pull into the run.
const ACCUM_MAX_GAP: u64 = 4096;

/// The write accumulator: one contiguous run of bytes waiting for a single
/// `pwrite`, the trade `H5F__accum_write` (H5Faccum.c) makes for the C
/// library's sec2 driver.
///
/// INVARIANT: `buf` holds the bytes the file must end up with over
/// `start .. start + buf.len()`, and `gaps` names every sub-range of it that
/// no write filled. A gap is a placeholder, never a value: [`Accum::flush`]
/// reads each one back off the file and writes it out unchanged, so joining a
/// run across a gap cannot change a byte of the file — no matter what the
/// gap held, and whether or not it lies past the end of the file.
struct Accum {
    /// Absolute file offset of `buf[0]`. Meaningless while `buf` is empty.
    start: u64,
    buf: Vec<u8>,
    /// Bridged gaps, as offsets into `buf`. Disjoint and ascending.
    gaps: Vec<std::ops::Range<usize>>,
}

impl Accum {
    fn new() -> Self {
        Accum {
            start: 0,
            buf: Vec::new(),
            gaps: Vec::new(),
        }
    }

    /// Absolute file offset one past the last buffered byte.
    fn end(&self) -> u64 {
        self.start + self.buf.len() as u64
    }

    /// Write the buffered run out and empty it.
    fn flush(&mut self, file: &File) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let result = self.write_out(file);
        // Empty the run whatever happened: a failed write has already been
        // reported to the caller, and retrying it from `Drop` would only
        // report it a second time.
        self.buf.clear();
        self.gaps.clear();
        result
    }

    /// One `pwrite` of the whole run, after every bridged gap is put back to
    /// the bytes the file already holds there.
    fn write_out(&mut self, file: &File) -> std::io::Result<()> {
        if !self.gaps.is_empty() {
            // Zeroed, so the tail of a run that reaches past the end of the
            // file keeps the zeros a hole reads as.
            let mut on_disk = vec![0u8; self.buf.len()];
            pread_upto(file, self.start, &mut on_disk)?;
            for gap in std::mem::take(&mut self.gaps) {
                self.buf[gap.clone()].copy_from_slice(&on_disk[gap]);
            }
        }
        pwrite_all(file, self.start, &self.buf)
    }

    /// Put `data` at absolute file offset `at` into the run, breaking the run
    /// and starting a new one where it cannot join.
    fn stage(&mut self, file: &File, at: u64, end: u64, data: &[u8]) -> std::io::Result<()> {
        if !self.buf.is_empty() && end.saturating_sub(self.start) as usize <= ACCUM_MAX {
            // Overwrites part of the run, or extends it: splice in place.
            if at >= self.start && at <= self.end() {
                let off = (at - self.start) as usize;
                if off + data.len() > self.buf.len() {
                    self.buf.resize(off + data.len(), 0);
                }
                self.buf[off..off + data.len()].copy_from_slice(data);
                self.fill_gaps(off..off + data.len());
                return Ok(());
            }
            // Sits a short way past the run: bridge the untouched bytes in
            // between rather than pay a syscall to break the run here.
            if at > self.end() && at - self.end() <= ACCUM_MAX_GAP {
                let gap = self.buf.len()..(at - self.start) as usize;
                self.buf.resize(gap.end, 0);
                self.buf.extend_from_slice(data);
                self.gaps.push(gap);
                return Ok(());
            }
        }
        self.flush(file)?;
        self.buf.extend_from_slice(data);
        self.start = at;
        Ok(())
    }

    /// Drop `written` out of the recorded gaps: those bytes now carry a value
    /// the flush must keep, not a placeholder for what the file holds.
    fn fill_gaps(&mut self, written: std::ops::Range<usize>) {
        match self.gaps.last() {
            Some(last) if written.start < last.end => {}
            _ => return,
        }
        let mut kept = Vec::with_capacity(self.gaps.len() + 1);
        for gap in self.gaps.drain(..) {
            if written.end <= gap.start || written.start >= gap.end {
                kept.push(gap);
                continue;
            }
            if gap.start < written.start {
                kept.push(gap.start..written.start);
            }
            if written.end < gap.end {
                kept.push(written.end..gap.end);
            }
        }
        self.gaps = kept;
    }
}

impl FileHandle {
    /// Wrap an already-opened file. Every constructor goes through here, so
    /// every handle starts with an empty accumulator over its own descriptor.
    fn new(file: File, writable: bool, lock_policy: FileLocking, lock_held: bool) -> Self {
        Self {
            file,
            source: ReadSource::Pread,
            accum: Mutex::new(Accum::new()),
            accum_dirty: AtomicBool::new(false),
            writable,
            lock_policy,
            lock_held,
            base: 0,
        }
    }

    /// Wrap a file opened read-only. The only constructor that can install a
    /// map, and the only one that hands `new` a `writable` of `false` — which
    /// is what makes `Mapped` imply read-only (see [`ReadSource`]).
    fn new_read_only(file: File, lock_policy: FileLocking, lock_held: bool) -> Self {
        let source = ReadSource::for_read_only(&file);
        let mut handle = Self::new(file, false, lock_policy, lock_held);
        handle.source = source;
        handle
    }

    /// Create a new file with the env-var-derived locking policy.
    #[cfg(test)]
    pub fn create(path: &Path) -> std::io::Result<Self> {
        Self::create_with_locking(path, FileLocking::from_env_or(FileLocking::default()))
    }

    /// Create a new file (truncating if it already exists) opened for
    /// read/write access, with an explicit locking policy.
    ///
    /// The lock is acquired *before* the file is truncated, so a lock
    /// conflict on an existing file does not destroy its contents.
    pub fn create_with_locking(path: &Path, policy: FileLocking) -> std::io::Result<Self> {
        // Open without O_TRUNC first so that we can validate the lock
        // before destroying any existing data.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let lock_held = locking::try_acquire(&file, LockMode::Exclusive, policy)?;
        // Now that the lock is held (or skipped per policy), truncate — but
        // only when there is something to truncate. An ftruncate-to-0, even on
        // a brand-new empty file, arms ext4's auto_da_alloc
        // (replace-via-truncate protection), which turns the final close(2)
        // into an implicit writeback of everything written since (~330 ms for
        // a 512^3 f32 dataset) and silently defeats `close_no_sync`.
        if file.metadata()?.len() > 0 {
            file.set_len(0)?;
        }
        Ok(Self::new(file, true, policy, lock_held))
    }

    /// Open a file for read/write access, creating it when it does not exist
    /// and keeping whatever it already holds — `O_CREAT | O_RDWR` with no
    /// `O_TRUNC`.
    ///
    /// What `H5D__efl_write` (H5Defl.c) opens an external raw-data file with.
    /// A dataset's External File List slot owns one byte range of a file this
    /// library does not otherwise manage, so a write to it must neither
    /// require the file to exist already nor destroy the ranges its other
    /// slots — or another dataset's — hold in the same file.
    pub fn open_or_create_readwrite_with_locking(
        path: &Path,
        policy: FileLocking,
    ) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let lock_held = locking::try_acquire(&file, LockMode::Exclusive, policy)?;
        Ok(Self::new(file, true, policy, lock_held))
    }

    /// Open an existing file for read-only access with the env-var-derived
    /// locking policy.
    #[cfg(test)]
    pub fn open_read(path: &Path) -> std::io::Result<Self> {
        Self::open_read_with_locking(path, FileLocking::from_env_or(FileLocking::default()))
    }

    /// Open an existing file for read-only access with an explicit locking
    /// policy. A shared lock is taken so multiple readers can coexist.
    pub fn open_read_with_locking(path: &Path, policy: FileLocking) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        // The lock goes on before the map is taken, so the map cannot capture
        // a file another opener is in the middle of writing.
        let lock_held = locking::try_acquire(&file, LockMode::Shared, policy)?;
        Ok(Self::new_read_only(file, policy, lock_held))
    }

    /// Open an existing file for read/write access with an explicit locking
    /// policy. An exclusive lock is taken.
    pub fn open_readwrite_with_locking(path: &Path, policy: FileLocking) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let lock_held = locking::try_acquire(&file, LockMode::Exclusive, policy)?;
        Ok(Self::new(file, true, policy, lock_held))
    }

    /// Byte offset of the HDF5 address space within the file — the userblock
    /// size. Zero for a file without one.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Move this handle's address space to start at `base`
    /// (`H5FD_set_base_addr`). Every later offset is taken relative to it.
    pub fn set_base(&mut self, base: u64) {
        self.base = base;
    }

    /// Absolute file offset of the HDF5 superblock signature, or `None` when
    /// the file holds no signature at all.
    ///
    /// `H5FD_locate_signature` looks at offset 0 and then at every power of
    /// two from 512 up to the file size: a userblock is a multiple-of-512
    /// power-of-two prefix, so those are the only places a superblock can
    /// start. Called before [`set_base`](Self::set_base), so it reads absolute
    /// offsets.
    pub fn locate_signature(&self) -> std::io::Result<Option<u64>> {
        use crate::format::superblock::HDF5_SIGNATURE;
        self.flush()?;
        let file_len = self.source.len(&self.file)?;
        let mut buf = [0u8; HDF5_SIGNATURE.len()];
        let mut addr = 0u64;
        loop {
            if addr + HDF5_SIGNATURE.len() as u64 <= file_len {
                self.source
                    .read_exact_into(&self.file, addr, &mut buf, ReadDst::Fresh)?;
                if buf == HDF5_SIGNATURE {
                    return Ok(Some(addr));
                }
            }
            // 0, then 512, 1024, 2048, ... while the probe is inside the file.
            addr = if addr == 0 { 512 } else { addr * 2 };
            if addr >= file_len {
                return Ok(None);
            }
        }
    }

    /// Release the OS-level lock so concurrent SWMR readers (and other
    /// openers) can attach. No-op if the policy is
    /// [`FileLocking::Disabled`] or no lock is held.
    ///
    /// We don't try to *downgrade* the exclusive lock to shared here:
    /// Windows' `LockFileEx` is a mandatory range lock, and an
    /// `unlock` followed by `try_lock_shared` on the same handle leaves
    /// the file in a state where subsequent `WriteFile` calls through
    /// that handle can fail with `ERROR_LOCK_VIOLATION`. Instead we
    /// release the lock entirely — matching the HDF5 C library, which
    /// also doesn't enforce reader/writer separation purely through
    /// OS locks during SWMR streaming.
    pub fn release_lock(&mut self) -> std::io::Result<()> {
        if !self.lock_held || matches!(self.lock_policy, FileLocking::Disabled) {
            return Ok(());
        }
        // A reader attaches the moment the lock is gone, so the accumulator
        // must not still be holding bytes it would then miss.
        self.flush()?;
        locking::release(&self.file)?;
        self.lock_held = false;
        Ok(())
    }

    /// Write `data` at the given byte offset.
    ///
    /// Uses a positioned write (`pwrite`) that neither moves nor depends on the
    /// file's seek cursor, so distinct, non-overlapping offsets are safe to
    /// write concurrently. Takes `&self` for exactly that reason.
    pub fn write_at(&self, offset: u64, data: &[u8]) -> std::io::Result<()> {
        if !self.writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "file opened read-only",
            ));
        }
        let at = self.abs(offset)?;
        let end = at.checked_add(data.len() as u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "write of {} bytes at {offset} overflows the address space",
                    data.len()
                ),
            )
        })?;
        if !stageable(data.len()) {
            // Buffered bytes under this write are older than it, so they have
            // to reach the file first or the flush would undo it. The write
            // itself then goes out with no lock held, which is what keeps
            // concurrent positioned writes at distinct offsets overlapping.
            if self.accum_dirty.load(Ordering::Acquire) {
                let mut accum = self.accum.lock().unwrap();
                if at < accum.end() && end > accum.start {
                    let result = accum.flush(&self.file);
                    self.accum_dirty.store(false, Ordering::Release);
                    result?;
                }
            }
            return pwrite_all(&self.file, at, data);
        }
        let mut accum = self.accum.lock().unwrap();
        let result = accum.stage(&self.file, at, end, data);
        self.accum_dirty
            .store(!accum.buf.is_empty(), Ordering::Release);
        result
    }

    /// Write out everything the accumulator holds.
    ///
    /// The one finalizer: every read, every look at the file's length, every
    /// `fsync`, the lock release SWMR opens its window with, and `Drop` all
    /// come through here, so no exit leaves buffered bytes unwritten and no
    /// observer sees a file the accumulator is still holding bytes back from.
    pub fn flush(&self) -> std::io::Result<()> {
        if !self.accum_dirty.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut accum = self.accum.lock().unwrap();
        let result = accum.flush(&self.file);
        self.accum_dirty.store(false, Ordering::Release);
        result
    }

    /// Read exactly `len` bytes starting at the given byte offset.
    pub fn read_at(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.flush()?;
        // `offset`/`len` are often file-derived; reject a request larger than
        // the file before allocating, so a corrupt size field cannot drive an
        // unbounded allocation.
        let file_len = self.source.len(&self.file)?;
        let start = self.abs(offset)?;
        let end = start.checked_add(len as u64);
        if end.is_none_or(|e| e > file_len) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("read past end: offset={offset} len={len} file_size={file_len}"),
            ));
        }
        self.source.read_vec(&self.file, start, len)
    }

    /// Read exactly `buf.len()` bytes at `offset` directly into `buf`.
    ///
    /// Like [`read_at`](Self::read_at) but writes into a caller-owned buffer,
    /// so a coalesced block read can land straight in the output without an
    /// intermediate `Vec` allocation per block.
    ///
    /// Unlike `read_at`, this does not `fstat` the file first: `read_at`'s
    /// up-front length check exists to bound an allocation sized from a
    /// possibly-corrupt on-disk length field, but here the buffer is owned by
    /// the caller and already sized from the validated selection, so there is
    /// no allocation to bound. A read past EOF still fails — the positioned
    /// read returns `UnexpectedEof` when it cannot fill `buf` — but without
    /// paying a per-call `fstat` on the hot coalesced-read path.
    ///
    /// `dst` is the caller's word on where `buf` came from ([`ReadDst`]):
    /// a reused buffer lets a mapped handle serve the read from its map at
    /// any size. The fact is a required argument, not a defaulted wrapper,
    /// so no call site can leave it unstated.
    pub fn read_exact_at_into(
        &self,
        offset: u64,
        buf: &mut [u8],
        dst: ReadDst,
    ) -> std::io::Result<()> {
        self.flush()?;
        self.source
            .read_exact_into(&self.file, self.abs(offset)?, buf, dst)
    }

    /// Read up to `max_len` bytes starting at the given byte offset.
    pub fn read_at_most(&self, offset: u64, max_len: usize) -> std::io::Result<Vec<u8>> {
        self.flush()?;
        // Clamp the allocation to what the file can actually hold.
        let file_len = self.source.len(&self.file)?;
        let start = self.abs(offset)?;
        let avail = file_len.saturating_sub(start);
        let max_len = (max_len as u64).min(avail) as usize;
        self.source.read_vec_upto(&self.file, start, max_len)
    }

    /// Retake this handle's read source against the file as it is now.
    ///
    /// A mapped handle serves the file as it was when the map was taken, so a
    /// SWMR writer's appends are past its end — and read as such — until the
    /// map is retaken. Every refresh comes through here before it decodes
    /// anything, so the bytes a refreshed reader decodes are as new as the
    /// metadata it decodes them with.
    ///
    /// Only a handle that already has a map takes a new one: `Mapped` implies
    /// read-only (see [`ReadSource`]), so this cannot hand a map to a writable
    /// handle, and a handle that never mapped has nothing that can go stale.
    /// A remap that fails leaves the handle on `Pread`, which no caller can
    /// tell from a mapped one.
    pub fn refresh_read_source(&mut self) {
        #[cfg(feature = "mmap")]
        if matches!(self.source, ReadSource::Mapped(_)) {
            // Drop the old mapping before taking the new one so the address
            // space of a large file is not held twice.
            self.source = ReadSource::Pread;
            self.source = ReadSource::for_read_only(&self.file);
        }
    }

    /// A share of the map this handle reads through, or `None` when it reads
    /// through `pread`.
    ///
    /// The one way the map leaves this type. A caller that holds the clone
    /// holds the pages: [`refresh_read_source`](Self::refresh_read_source)
    /// dropping this handle's share, or the handle itself going away, unmaps
    /// nothing while a share is outstanding, so a view built on it stays
    /// readable and keeps showing the file as it was when the map was taken.
    ///
    /// The bytes are the file's own, so a caller must not assume they stop
    /// changing: another process writing the file in place through a shared
    /// mapping is seen through this one too, and a truncation past the map's
    /// end takes `SIGBUS` on the pages that went away — the same risk
    /// [`ReadSource::for_read_only`] takes when it maps at all, and one no
    /// guard in this process can remove.
    #[cfg(feature = "mmap")]
    pub fn map_snapshot(&self) -> Option<Arc<memmap2::Mmap>> {
        match &self.source {
            ReadSource::Mapped(map) => Some(Arc::clone(map)),
            ReadSource::Pread => None,
        }
    }

    /// Whether this handle reads through a map. Only the tests that pin the
    /// map's boundary behaviour ask; every other caller is meant not to be
    /// able to tell.
    #[cfg(test)]
    fn is_mapped(&self) -> bool {
        // Written against `Pread` rather than `Mapped` so it reads the same
        // in a build that has no `Mapped` variant to name.
        !matches!(self.source, ReadSource::Pread)
    }

    /// Translate an HDF5 address into a file offset. Fails rather than wraps
    /// when a corrupt address plus the userblock would overflow.
    fn abs(&self, offset: u64) -> std::io::Result<u64> {
        offset.checked_add(self.base).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "address {offset} overflows past the userblock at {}",
                    self.base
                ),
            )
        })
    }

    /// Flush file data (not necessarily metadata) to disk.
    pub fn sync_data(&self) -> std::io::Result<()> {
        self.flush()?;
        self.file.sync_data()
    }

    /// Flush both file data and metadata to disk.
    pub fn sync_all(&self) -> std::io::Result<()> {
        self.flush()?;
        self.file.sync_all()
    }

    /// Return the current file size, in the same address space as every offset
    /// this handle takes: the userblock is not part of it, so the result is
    /// directly comparable against an HDF5 address (which is what
    /// `H5FD_get_eof` returns for the same reason).
    pub fn file_size(&self) -> std::io::Result<u64> {
        self.flush()?;
        Ok(self.source.len(&self.file)?.saturating_sub(self.base))
    }

    /// Set the file's length to `eof` in this handle's address space —
    /// `H5FD_truncate`, which every close calls so the file on disk ends where
    /// the superblock says the address space does.
    ///
    /// A file shorter than its recorded end of file is one libhdf5 refuses as
    /// truncated (`H5F__super_read`, H5Fsuper.c:573); a longer one has bytes
    /// no structure in it accounts for, which is what `h5stat -S` reports as
    /// unaccounted space. Both are closed here rather than by a rule that
    /// every allocation must be written, which nothing can enforce.
    pub fn set_eof(&self, eof: u64) -> std::io::Result<()> {
        // Buffered bytes first: they can be what makes the file the length it
        // should already be, and a truncation that ran before them would cut
        // off bytes this handle then wrote back past the new end.
        self.flush()?;
        let want = eof + self.base;
        if self.file.metadata()?.len() != want {
            self.file.set_len(want)?;
        }
        Ok(())
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        // Backstop. Every ordinary exit — close, `sync_data`/`sync_all`,
        // `set_eof`, `release_lock`, any read — has already been through
        // `flush` and returned its error to a caller who could act on it.
        // Reaching here with bytes still buffered means the handle was
        // abandoned without one of those, so report rather than swallow, the
        // way `Hdf5Writer::drop` reports a failed finalize.
        if let Err(e) = self.flush() {
            eprintln!(
                "rust-hdf5: failed to flush buffered file writes on drop: {e}. \
                 The file may be incomplete or corrupt; call H5File::close() \
                 to handle this error explicitly."
            );
        }
    }
}

/// Write all of `data` to `file` starting at `offset` using a positioned
/// write.
///
/// A positioned write neither consults nor moves the file's seek cursor, so
/// concurrent calls at distinct, non-overlapping offsets against a shared
/// `&File` do not race — the property the `threadsafe` fine-grained writer
/// relies on to write chunk data without holding a global lock (see
/// `docs/threadsafe-fine-grained-locking.md`).
#[cfg(unix)]
fn pwrite_all(file: &File, offset: u64, data: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(data, offset)
}

#[cfg(windows)]
fn pwrite_all(file: &File, mut offset: u64, mut data: &[u8]) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    // `seek_write` is positioned but may write fewer bytes than requested, so
    // loop until the whole buffer lands.
    while !data.is_empty() {
        match file.seek_write(data, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            Ok(n) => {
                data = &data[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn pwrite_all(file: &File, offset: u64, data: &[u8]) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    // No positioned-write API on this platform; fall back to seek+write on a
    // shared `&File`. This moves the shared cursor and is therefore NOT
    // concurrency-safe — on such a platform the writer must stay serialized.
    let mut f = file;
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(data)
}

/// Positioned read of up to `buf.len()` bytes from `file` at `offset`,
/// returning the number of bytes read (may be short). Does not move the seek
/// cursor; safe to issue concurrently with positioned writes at other offsets.
#[cfg(unix)]
fn pread(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn pread(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(not(any(unix, windows)))]
fn pread(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = file;
    f.seek(SeekFrom::Start(offset))?;
    f.read(buf)
}

/// Positioned read of as much of `buf` as the file holds at `offset`, leaving
/// the bytes past the end of the file untouched.
///
/// The accumulator reads a run's gaps back through this before flushing, and a
/// run may reach past the end of the file — where a plain `pread_exact` would
/// fail on bytes that a `pwrite` is about to create as a zero-filled hole.
fn pread_upto(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    let mut total = 0;
    while total < buf.len() {
        match pread(file, offset + total as u64, &mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Positioned read of exactly `buf.len()` bytes from `file` at `offset`,
/// failing with `UnexpectedEof` if the file is too short.
#[cfg(unix)]
fn pread_exact(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn pread_exact(file: &File, mut offset: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn pread_exact(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = file;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp(label: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rust_hdf5_accum_{}_{}_{}",
            label,
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("f.bin")
    }

    /// Hand one piece straight to the accumulator, the way `write_at` does
    /// for a write below the pass-through size.
    fn stage(accum: &mut Accum, file: &File, at: u64, data: &[u8]) {
        accum.stage(file, at, at + data.len() as u64, data).unwrap();
    }

    fn scratch() -> (std::path::PathBuf, File) {
        let path = tmp("accum_unit");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        (path, file)
    }

    /// The owner path: consecutive pieces land in one run, so one `pwrite`
    /// carries what used to be three.
    #[test]
    fn adjoining_writes_join_one_run() {
        let (_p, file) = scratch();
        let mut accum = Accum::new();
        stage(&mut accum, &file, 100, &[1u8; 10]);
        stage(&mut accum, &file, 110, &[2u8; 10]);
        stage(&mut accum, &file, 120, &[3u8; 10]);
        assert_eq!(accum.start, 100);
        assert_eq!(accum.buf.len(), 30);
        assert!(accum.gaps.is_empty());
        assert_eq!(file.metadata().unwrap().len(), 0, "nothing written yet");
        accum.flush(&file).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 130);
    }

    /// A short hop past the run is bridged rather than breaking it, and the
    /// bridged bytes are recorded as a gap, not as a value.
    #[test]
    fn a_short_hop_is_bridged_and_recorded() {
        let (_p, file) = scratch();
        let mut accum = Accum::new();
        stage(&mut accum, &file, 100, &[1u8; 10]);
        stage(&mut accum, &file, 116, &[2u8; 10]);
        assert_eq!(accum.start, 100);
        assert_eq!(accum.gaps, vec![10..16]);
        // Beyond the bridge width the run breaks instead.
        stage(&mut accum, &file, 116 + 10 + ACCUM_MAX_GAP + 1, &[3u8; 10]);
        assert_eq!(accum.start, 116 + 10 + ACCUM_MAX_GAP + 1);
        assert!(accum.gaps.is_empty());
    }

    /// The reason a bridge is safe: whatever the file already holds under the
    /// gap is read back and written out unchanged. Reclaimed space can hold
    /// anything, so zero fill would corrupt it.
    #[test]
    fn a_bridge_keeps_the_bytes_already_under_it() {
        let path = tmp("bridge_keeps");
        let handle = FileHandle::create(&path).unwrap();
        handle.write_at(0, &[0xABu8; 64]).unwrap();
        handle.flush().unwrap();

        handle.write_at(0, &[1, 2, 3, 4]).unwrap();
        handle.write_at(60, &[5, 6, 7, 8]).unwrap();
        handle.flush().unwrap();

        let got = handle.read_at(0, 64).unwrap();
        assert_eq!(&got[0..4], &[1, 2, 3, 4]);
        assert_eq!(&got[4..60], &[0xABu8; 56]);
        assert_eq!(&got[60..64], &[5, 6, 7, 8]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A write landing inside an already-bridged gap owns those bytes: the
    /// flush must keep them, not restore what the file held.
    #[test]
    fn a_write_into_a_bridged_gap_wins_over_the_read_back() {
        let path = tmp("gap_overwrite");
        let handle = FileHandle::create(&path).unwrap();
        handle.write_at(0, &[0xABu8; 64]).unwrap();
        handle.flush().unwrap();

        handle.write_at(0, &[1, 2, 3, 4]).unwrap();
        handle.write_at(60, &[5, 6, 7, 8]).unwrap();
        handle.write_at(30, &[9, 9]).unwrap();
        handle.flush().unwrap();

        let got = handle.read_at(0, 64).unwrap();
        assert_eq!(&got[28..34], &[0xAB, 0xAB, 9, 9, 0xAB, 0xAB]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// `read_at` used to go straight to the descriptor. It must not be able
    /// to read around bytes the accumulator is still holding.
    #[test]
    fn a_read_sees_writes_still_in_the_accumulator() {
        let path = tmp("read_through");
        let handle = FileHandle::create(&path).unwrap();
        handle.write_at(0, &[7u8; 32]).unwrap();
        assert_eq!(handle.read_at(0, 32).unwrap(), vec![7u8; 32]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// So must `file_size`, which used to `fstat` straight through, and
    /// `set_eof`, which used to truncate against that stale length.
    #[test]
    fn the_file_length_accounts_for_buffered_writes() {
        let path = tmp("file_len");
        let handle = FileHandle::create(&path).unwrap();
        handle.write_at(0, &[7u8; 300]).unwrap();
        assert_eq!(handle.file_size().unwrap(), 300);
        handle.write_at(300, &[8u8; 100]).unwrap();
        handle.set_eof(400).unwrap();
        assert_eq!(handle.file_size().unwrap(), 400);
        assert_eq!(handle.read_at(299, 2).unwrap(), vec![7, 8]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A write too large to be worth staging goes straight out, but only
    /// after the older bytes it covers have. Under `threadsafe` nothing is
    /// ever staged, so there is no older run for it to overtake.
    #[cfg(not(feature = "threadsafe"))]
    #[test]
    fn a_pass_through_write_lands_after_the_bytes_it_covers() {
        let path = tmp("pass_through");
        let handle = FileHandle::create(&path).unwrap();
        handle.write_at(0, &[1u8; 64]).unwrap();
        handle.write_at(0, &vec![2u8; ACCUM_PASSTHROUGH]).unwrap();
        handle.flush().unwrap();
        assert_eq!(handle.read_at(0, 64).unwrap(), vec![2u8; 64]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// Dropping a handle without any of the explicit exits still flushes.
    #[test]
    fn drop_flushes_what_is_left() {
        let path = tmp("drop_flush");
        {
            let handle = FileHandle::create(&path).unwrap();
            handle.write_at(0, &[42u8; 16]).unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), vec![42u8; 16]);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A userblock shifts the address space; the accumulator works in
    /// absolute offsets, so a run spanning the shift is still one write.
    #[test]
    fn a_userblock_does_not_break_the_run() {
        let path = tmp("userblock");
        let mut handle = FileHandle::create(&path).unwrap();
        handle.write_at(0, &[0u8; 512]).unwrap();
        handle.set_base(512);
        handle.write_at(0, &[3u8; 8]).unwrap();
        handle.write_at(8, &[4u8; 8]).unwrap();
        handle.flush().unwrap();
        assert_eq!(handle.file_size().unwrap(), 16);
        // The handle's lock is mandatory on Windows: reading through a
        // second handle needs it gone first.
        drop(handle);
        assert_eq!(std::fs::read(&path).unwrap().len(), 528);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}

/// The read source's boundary behaviour.
///
/// A read-only handle reads through a memory map when the `mmap` feature is
/// on and the platform allows one, and through `pread` otherwise. The two
/// must be indistinguishable: same bytes, same error at the same boundary.
/// Every test here runs in both builds and asserts the behaviour that must
/// hold either way; the `mmap`-gated assertions pin what is specific to the
/// map (that it is taken at all, and that it is a snapshot until retaken).
#[cfg(test)]
mod read_source_tests {
    use super::*;

    fn dir(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::AtomicU64;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rust_hdf5_source_{}_{}_{}",
            label,
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A read-only open maps; a writable one never does, which is what keeps
    /// a map from ever coexisting with buffered writes.
    #[test]
    fn only_a_read_only_open_maps() {
        let d = dir("only_read_only");
        let path = d.join("f.bin");
        std::fs::write(&path, vec![7u8; 4096]).unwrap();

        let writable =
            FileHandle::open_readwrite_with_locking(&path, FileLocking::Disabled).unwrap();
        assert!(!writable.is_mapped());
        drop(writable);

        let created = FileHandle::create(&d.join("created.bin")).unwrap();
        assert!(!created.is_mapped());
        drop(created);

        let read_only = FileHandle::open_read(&path).unwrap();
        assert_eq!(read_only.is_mapped(), cfg!(feature = "mmap"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A read that ends exactly at the end of the file is a read, not an
    /// error — including the zero-length one that starts there.
    #[test]
    fn a_read_ending_at_eof_succeeds() {
        let d = dir("at_eof");
        let path = d.join("f.bin");
        let bytes: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &bytes).unwrap();

        let handle = FileHandle::open_read(&path).unwrap();
        assert_eq!(handle.file_size().unwrap(), 1000);
        assert_eq!(handle.read_at(990, 10).unwrap(), bytes[990..].to_vec());
        assert_eq!(handle.read_at(1000, 0).unwrap(), Vec::<u8>::new());
        assert_eq!(handle.read_at_most(1000, 64).unwrap(), Vec::<u8>::new());
        assert_eq!(handle.read_at_most(990, 64).unwrap(), bytes[990..].to_vec());
        let mut out = [0u8; 10];
        handle
            .read_exact_at_into(990, &mut out, ReadDst::Fresh)
            .unwrap();
        assert_eq!(&out, &bytes[990..]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A read reaching past the end fails with the error a short `pread`
    /// gives — never a panic, and never the `SIGBUS` a walk off the end of a
    /// mapping would take.
    #[test]
    fn a_read_past_eof_fails_as_unexpected_eof() {
        let d = dir("past_eof");
        let path = d.join("f.bin");
        std::fs::write(&path, vec![3u8; 1000]).unwrap();

        let handle = FileHandle::open_read(&path).unwrap();
        for (offset, len) in [(1000u64, 1usize), (999, 2), (0, 1001)] {
            let err = handle.read_at(offset, len).unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::UnexpectedEof,
                "read_at({offset}, {len})"
            );
            let mut out = vec![0u8; len];
            let err = handle
                .read_exact_at_into(offset, &mut out, ReadDst::Fresh)
                .unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::UnexpectedEof,
                "read_exact_at_into({offset}, {len})"
            );
        }
        // An offset past what the OS will take an address for is refused by
        // both sources, though not with the same word for it: the descriptor
        // reports `InvalidInput` where the map reports `UnexpectedEof`. What
        // the two owe each other is that neither reads.
        assert!(handle.read_at(u64::MAX, 8).is_err());
        assert!(handle
            .read_exact_at_into(u64::MAX, &mut [0u8; 8], ReadDst::Fresh)
            .is_err());

        // `read_at_most` is the one that reports a short read by returning
        // what there was, so past the end it returns nothing.
        assert!(handle.read_at_most(1000, 64).unwrap().is_empty());
        assert!(handle.read_at_most(u64::MAX, 64).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A userblock shifts the address space for the map exactly as it does
    /// for `pread`: the same address reads the same bytes through either.
    #[test]
    fn a_userblock_shifts_the_source_the_same_way() {
        let d = dir("userblock");
        let path = d.join("f.bin");
        let mut bytes = vec![0xEEu8; 512];
        bytes.extend((0..500u32).map(|i| i as u8));
        std::fs::write(&path, &bytes).unwrap();

        let mut handle = FileHandle::open_read(&path).unwrap();
        handle.set_base(512);
        assert_eq!(handle.file_size().unwrap(), 500);
        assert_eq!(handle.read_at(0, 4).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(handle.read_at(499, 1).unwrap(), vec![243]);
        assert_eq!(
            handle.read_at(500, 1).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A file that grows under a SWMR writer: the new bytes are past the end
    /// of what the handle reads from until the source is retaken, and a read
    /// of them until then fails the way a read past the end of a shorter file
    /// does.
    #[test]
    fn growth_is_picked_up_when_the_source_is_retaken() {
        use std::io::Write;
        let d = dir("growth");
        let path = d.join("f.bin");
        std::fs::write(&path, vec![1u8; 4096]).unwrap();

        // No lock: the appender below writes through a second handle while
        // this one is open, which a shared lock forbids on Windows — the
        // topology a SWMR reader opens with (`open_swmr_with_locking`).
        let mut handle = FileHandle::open_read_with_locking(&path, FileLocking::Disabled).unwrap();
        assert_eq!(handle.file_size().unwrap(), 4096);

        let mut appender = OpenOptions::new().append(true).open(&path).unwrap();
        appender.write_all(&vec![2u8; 4096]).unwrap();
        appender.flush().unwrap();

        // A map is a snapshot: the appended bytes are outside it, and reading
        // them errors rather than faulting.
        #[cfg(feature = "mmap")]
        {
            assert!(handle.is_mapped());
            assert_eq!(handle.file_size().unwrap(), 4096);
            assert_eq!(
                handle.read_at(4096, 4096).unwrap_err().kind(),
                std::io::ErrorKind::UnexpectedEof
            );
        }

        handle.refresh_read_source();
        assert_eq!(handle.file_size().unwrap(), 8192);
        assert_eq!(handle.read_at(4096, 4096).unwrap(), vec![2u8; 4096]);
        assert_eq!(handle.read_at(0, 4096).unwrap(), vec![1u8; 4096]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// An empty file cannot be mapped. The handle falls back and reads from
    /// the descriptor, so an opener cannot tell the mapping failed.
    #[test]
    fn an_empty_file_falls_back_to_the_descriptor() {
        let d = dir("empty");
        let path = d.join("f.bin");
        std::fs::write(&path, []).unwrap();

        let mut handle = FileHandle::open_read(&path).unwrap();
        assert!(!handle.is_mapped());
        assert_eq!(handle.file_size().unwrap(), 0);
        assert!(handle.read_at_most(0, 64).unwrap().is_empty());
        assert_eq!(
            handle.read_at(0, 1).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        assert!(handle.locate_signature().unwrap().is_none());
        // Retaking the source on a handle that never mapped leaves it alone:
        // a map begets a map, so a fallback can never turn into one.
        handle.refresh_read_source();
        assert!(!handle.is_mapped());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The two sources return the same bytes for the same request, at every
    /// shape of request the reader makes — including the reads either side of
    /// the size above which a mapped handle goes back to the descriptor, so
    /// that the rule about where bytes come from cannot change what they are.
    #[test]
    fn a_map_and_a_descriptor_read_the_same_bytes() {
        let d = dir("differential");
        let path = d.join("f.bin");
        let bytes: Vec<u8> = (0..70_000u32).map(|i| (i * 31) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();

        let handle = FileHandle::open_read(&path).unwrap();
        for (offset, len) in [
            (0u64, 8usize),
            (1, 1),
            (4095, 4098),
            // Either side of the size above which a mapped handle hands the
            // read back to the descriptor.
            (100, 8 * 1024 - 1),
            (100, 8 * 1024),
            (100, 8 * 1024 + 1),
            (65_536, 4464),
            (69_999, 1),
            (0, 70_000),
        ] {
            let want = &bytes[offset as usize..offset as usize + len];
            assert_eq!(handle.read_at(offset, len).unwrap(), want, "read_at");
            assert_eq!(
                handle.read_at_most(offset, len + 100).unwrap(),
                &bytes[offset as usize..(offset as usize + len + 100).min(bytes.len())],
                "read_at_most"
            );
            let mut out = vec![0u8; len];
            handle
                .read_exact_at_into(offset, &mut out, ReadDst::Fresh)
                .unwrap();
            assert_eq!(out, want, "read_exact_at_into");
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
