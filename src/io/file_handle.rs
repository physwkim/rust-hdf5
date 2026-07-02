use std::fs::{File, OpenOptions};
use std::path::Path;

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
/// There is no application-level read or write buffer. The structural reads the
/// reader makes are at scattered offsets and already over-read whole structures
/// per call (`read_at_most` grabs e.g. a 64 KiB window), so a `BufReader` —
/// whose buffer is discarded on every non-adjacent `seek` — added little for
/// this access pattern, and a buffer would have blocked the `&self` positioned
/// reads the concurrent write path needs.
pub struct FileHandle {
    file: File,
    /// False for files opened read-only; [`write_at`](Self::write_at) then
    /// errors instead of attempting a write.
    writable: bool,
    /// Active locking policy. Used when downgrading the lock for SWMR.
    /// When the policy is [`FileLocking::Disabled`], no lock was taken.
    lock_policy: FileLocking,
    /// True if a lock is currently held on the underlying file.
    lock_held: bool,
}

impl FileHandle {
    /// Create a new file with the env-var-derived locking policy.
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
        // Now that the lock is held (or skipped per policy), truncate.
        file.set_len(0)?;
        Ok(Self {
            file,
            writable: true,
            lock_policy: policy,
            lock_held,
        })
    }

    /// Open an existing file for read-only access with the env-var-derived
    /// locking policy.
    pub fn open_read(path: &Path) -> std::io::Result<Self> {
        Self::open_read_with_locking(path, FileLocking::from_env_or(FileLocking::default()))
    }

    /// Open an existing file for read-only access with an explicit locking
    /// policy. A shared lock is taken so multiple readers can coexist.
    pub fn open_read_with_locking(path: &Path, policy: FileLocking) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let lock_held = locking::try_acquire(&file, LockMode::Shared, policy)?;
        Ok(Self {
            file,
            writable: false,
            lock_policy: policy,
            lock_held,
        })
    }

    /// Open an existing file for read/write access with the env-var-derived
    /// locking policy.
    pub fn open_readwrite(path: &Path) -> std::io::Result<Self> {
        Self::open_readwrite_with_locking(path, FileLocking::from_env_or(FileLocking::default()))
    }

    /// Open an existing file for read/write access with an explicit locking
    /// policy. An exclusive lock is taken.
    pub fn open_readwrite_with_locking(path: &Path, policy: FileLocking) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let lock_held = locking::try_acquire(&file, LockMode::Exclusive, policy)?;
        Ok(Self {
            file,
            writable: true,
            lock_policy: policy,
            lock_held,
        })
    }

    /// Locking policy this handle was opened with.
    pub fn lock_policy(&self) -> FileLocking {
        self.lock_policy
    }

    /// Whether a lock is currently held on this handle.
    pub fn lock_held(&self) -> bool {
        self.lock_held
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
        // Positioned writes hit the OS directly (no application buffer), so
        // there is nothing to flush before the lock window opens.
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
        pwrite_all(&self.file, offset, data)
    }

    /// Read exactly `len` bytes starting at the given byte offset.
    pub fn read_at(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        // `offset`/`len` are often file-derived; reject a request larger than
        // the file before allocating, so a corrupt size field cannot drive an
        // unbounded allocation.
        let file_len = self.file.metadata()?.len();
        let end = offset.checked_add(len as u64);
        if end.is_none_or(|e| e > file_len) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("read past end: offset={offset} len={len} file_size={file_len}"),
            ));
        }
        let mut buf = vec![0u8; len];
        pread_exact(&self.file, offset, &mut buf)?;
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
        pread_exact(&self.file, offset, buf)
    }

    /// Read up to `max_len` bytes starting at the given byte offset.
    pub fn read_at_most(&self, offset: u64, max_len: usize) -> std::io::Result<Vec<u8>> {
        // Clamp the allocation to what the file can actually hold.
        let file_len = self.file.metadata()?.len();
        let avail = file_len.saturating_sub(offset);
        let max_len = (max_len as u64).min(avail) as usize;
        let mut buf = vec![0u8; max_len];
        let mut total = 0;
        while total < buf.len() {
            match pread(&self.file, offset + total as u64, &mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        buf.truncate(total);
        Ok(buf)
    }

    /// Flush file data (not necessarily metadata) to disk.
    pub fn sync_data(&self) -> std::io::Result<()> {
        self.file.sync_data()
    }

    /// Flush both file data and metadata to disk.
    pub fn sync_all(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    /// Return the current file size.
    pub fn file_size(&self) -> std::io::Result<u64> {
        Ok(self.file.metadata()?.len())
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
        Ok(Self { mmap, _file: file })
    }

    /// Return the total size of the mapped file.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Return whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Read exactly `len` bytes at `offset`. Zero-copy: returns a slice.
    pub fn read_at(&self, offset: u64, len: usize) -> std::io::Result<&[u8]> {
        // `offset`/`len` are file-derived; compute the end in u64 and reject
        // overflow so a hostile value cannot wrap past the bounds check.
        let end = offset
            .checked_add(len as u64)
            .filter(|&e| e <= self.mmap.len() as u64);
        match end {
            Some(end) => Ok(&self.mmap[offset as usize..end as usize]),
            None => Err(std::io::Error::new(
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
        if offset >= self.mmap.len() as u64 {
            return &[];
        }
        let start = offset as usize;
        let end = (start as u64)
            .saturating_add(max_len as u64)
            .min(self.mmap.len() as u64) as usize;
        &self.mmap[start..end]
    }
}
