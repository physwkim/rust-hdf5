use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
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
            accum: Mutex::new(Accum::new()),
            accum_dirty: AtomicBool::new(false),
            writable,
            lock_policy,
            lock_held,
            base: 0,
        }
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
        let lock_held = locking::try_acquire(&file, LockMode::Shared, policy)?;
        Ok(Self::new(file, false, policy, lock_held))
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
        let file_len = self.file.metadata()?.len();
        let mut buf = [0u8; HDF5_SIGNATURE.len()];
        let mut addr = 0u64;
        loop {
            if addr + HDF5_SIGNATURE.len() as u64 <= file_len {
                pread_exact(&self.file, addr, &mut buf)?;
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
        let file_len = self.file.metadata()?.len();
        let start = self.abs(offset)?;
        let end = start.checked_add(len as u64);
        if end.is_none_or(|e| e > file_len) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("read past end: offset={offset} len={len} file_size={file_len}"),
            ));
        }
        let mut buf = vec![0u8; len];
        pread_exact(&self.file, start, &mut buf)?;
        Ok(buf)
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
    pub fn read_exact_at_into(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.flush()?;
        pread_exact(&self.file, self.abs(offset)?, buf)
    }

    /// Read up to `max_len` bytes starting at the given byte offset.
    pub fn read_at_most(&self, offset: u64, max_len: usize) -> std::io::Result<Vec<u8>> {
        self.flush()?;
        // Clamp the allocation to what the file can actually hold.
        let file_len = self.file.metadata()?.len();
        let start = self.abs(offset)?;
        let avail = file_len.saturating_sub(start);
        let max_len = (max_len as u64).min(avail) as usize;
        let mut buf = vec![0u8; max_len];
        let mut total = 0;
        while total < buf.len() {
            match pread(&self.file, start + total as u64, &mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        buf.truncate(total);
        Ok(buf)
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
        Ok(self.file.metadata()?.len().saturating_sub(self.base))
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

/// A memory-mapped read-only file handle for zero-copy reads.
///
/// Available when the `mmap` feature is enabled.
#[cfg(feature = "mmap")]
pub struct MmapFileHandle {
    mmap: memmap2::Mmap,
    /// Keep the underlying file alive so the OS lock survives for the
    /// lifetime of this handle. (The mmap itself doesn't pin the fd.)
    _file: File,
    /// Userblock size, as in [`FileHandle::base`]: this handle shares the
    /// address space of the [`FileHandle`] the same file was opened with, so
    /// an address is read from the same bytes through either.
    base: u64,
}

#[cfg(feature = "mmap")]
impl MmapFileHandle {
    /// Open a file with memory mapping for read-only access, using the
    /// env-var-derived locking policy.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        Self::open_with_locking(path, FileLocking::from_env_or(FileLocking::default()))
    }

    /// Open a file with memory mapping with an explicit locking policy.
    /// A shared lock is taken (mmap is read-only) so the handle blocks
    /// concurrent writers as long as it lives.
    pub fn open_with_locking(path: &Path, policy: FileLocking) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // Take the shared lock BEFORE mmapping so the mmap doesn't
        // capture a snapshot of a file that's being concurrently
        // modified.
        let _ = locking::try_acquire(&file, LockMode::Shared, policy)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self {
            mmap,
            _file: file,
            base: 0,
        })
    }

    /// Move this handle's address space to start at `base`, as
    /// [`FileHandle::set_base`] does.
    pub fn set_base(&mut self, base: u64) {
        self.base = base;
    }

    /// Return the size of the mapped file's HDF5 address space — the mapping
    /// less the userblock, so the result bounds the offsets this handle takes.
    pub fn len(&self) -> usize {
        self.mmap.len().saturating_sub(self.base as usize)
    }

    /// Return whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `len` bytes at `offset`. Zero-copy: returns a slice.
    pub fn read_at(&self, offset: u64, len: usize) -> std::io::Result<&[u8]> {
        // `offset`/`len` are file-derived; compute the end in u64 and reject
        // overflow so a hostile value cannot wrap past the bounds check.
        let start = offset.checked_add(self.base);
        let end = start
            .and_then(|s| s.checked_add(len as u64))
            .filter(|&e| e <= self.mmap.len() as u64);
        match (start, end) {
            (Some(start), Some(end)) => Ok(&self.mmap[start as usize..end as usize]),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "mmap read past end: offset={} len={} file_size={}",
                    offset,
                    len,
                    self.mmap.len()
                ),
            )),
        }
    }

    /// Read up to `max_len` bytes at `offset`. Returns a slice.
    pub fn read_at_most(&self, offset: u64, max_len: usize) -> &[u8] {
        let Some(start) = offset
            .checked_add(self.base)
            .filter(|&s| s < self.mmap.len() as u64)
        else {
            return &[];
        };
        let end = start
            .saturating_add(max_len as u64)
            .min(self.mmap.len() as u64) as usize;
        &self.mmap[start as usize..end]
    }
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
        assert_eq!(std::fs::read(&path).unwrap().len(), 528);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
