use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::io::locking::{self, FileLocking, LockMode};

/// Default buffer size for buffered I/O (64 KiB).
const DEFAULT_BUF_SIZE: usize = 64 * 1024;

/// Wraps `std::fs::File` with buffered, positioned I/O convenience methods.
///
/// Write operations go through a `BufWriter` to merge small writes.
/// Read operations go through a `BufReader` to reduce syscall overhead.
/// The buffers are flushed automatically when switching between
/// read and write operations.
pub struct FileHandle {
    file: Option<File>,
    mode: Mode,
    /// Active locking policy. Used when downgrading the lock for SWMR.
    /// When the policy is [`FileLocking::Disabled`], no lock was taken.
    lock_policy: FileLocking,
    /// True if a lock is currently held on the underlying file.
    lock_held: bool,
}

enum Mode {
    /// Read/write capable, currently buffering writes.
    Writer(BufWriter<File>),
    /// Read/write capable, currently buffering reads.
    Reader(BufReader<File>),
    /// Read-only file.
    ReadOnly(BufReader<File>),
    /// Transitional state while swapping buffers.
    Transitioning,
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
            file: None,
            mode: Mode::Writer(BufWriter::with_capacity(DEFAULT_BUF_SIZE, file)),
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
            file: None,
            mode: Mode::ReadOnly(BufReader::with_capacity(DEFAULT_BUF_SIZE, file)),
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
            file: None,
            mode: Mode::Writer(BufWriter::with_capacity(DEFAULT_BUF_SIZE, file)),
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

    /// Downgrade an exclusive lock to a shared lock so concurrent SWMR
    /// readers can attach. No-op if the policy is [`FileLocking::Disabled`]
    /// or no lock is held.
    pub fn downgrade_lock_to_shared(&mut self) -> std::io::Result<()> {
        if !self.lock_held || matches!(self.lock_policy, FileLocking::Disabled) {
            return Ok(());
        }
        // Flush any pending writes so they hit disk before the lock window.
        self.flush_buffers()?;
        let file_ref = self.get_file_ref();
        let acquired = locking::downgrade_to_shared(file_ref, self.lock_policy)?;
        self.lock_held = acquired;
        Ok(())
    }

    /// Extract the raw `File` from the current mode, flushing if needed.
    fn take_file(&mut self) -> std::io::Result<File> {
        let old = std::mem::replace(&mut self.mode, Mode::Transitioning);
        match old {
            Mode::Writer(w) => w.into_inner().map_err(|e| e.into_error()),
            Mode::Reader(r) => Ok(r.into_inner()),
            Mode::ReadOnly(r) => Ok(r.into_inner()),
            Mode::Transitioning => {
                // Use stashed file if available
                self.file
                    .take()
                    .ok_or_else(|| std::io::Error::other("no file available"))
            }
        }
    }

    /// Ensure we are in writer mode.
    fn ensure_writer(&mut self) -> std::io::Result<()> {
        match &self.mode {
            Mode::Writer(_) => return Ok(()),
            Mode::ReadOnly(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "file opened read-only",
                ));
            }
            _ => {}
        }
        let file = self.take_file()?;
        self.mode = Mode::Writer(BufWriter::with_capacity(DEFAULT_BUF_SIZE, file));
        Ok(())
    }

    /// Ensure we are in reader mode.
    fn ensure_reader(&mut self) -> std::io::Result<()> {
        match &self.mode {
            Mode::Reader(_) | Mode::ReadOnly(_) => return Ok(()),
            _ => {}
        }
        let file = self.take_file()?;
        self.mode = Mode::Reader(BufReader::with_capacity(DEFAULT_BUF_SIZE, file));
        Ok(())
    }

    /// Write `data` at the given byte offset.
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> std::io::Result<()> {
        self.ensure_writer()?;
        if let Mode::Writer(w) = &mut self.mode {
            w.seek(SeekFrom::Start(offset))?;
            w.write_all(data)?;
        }
        Ok(())
    }

    /// Read exactly `len` bytes starting at the given byte offset.
    pub fn read_at(&mut self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        self.ensure_reader()?;
        match &mut self.mode {
            Mode::Reader(r) | Mode::ReadOnly(r) => {
                r.seek(SeekFrom::Start(offset))?;
                let mut buf = vec![0u8; len];
                r.read_exact(&mut buf)?;
                Ok(buf)
            }
            _ => unreachable!(),
        }
    }

    /// Read up to `max_len` bytes starting at the given byte offset.
    pub fn read_at_most(&mut self, offset: u64, max_len: usize) -> std::io::Result<Vec<u8>> {
        self.ensure_reader()?;
        match &mut self.mode {
            Mode::Reader(r) | Mode::ReadOnly(r) => {
                r.seek(SeekFrom::Start(offset))?;
                let mut buf = vec![0u8; max_len];
                let mut total = 0;
                loop {
                    match r.read(&mut buf[total..]) {
                        Ok(0) => break,
                        Ok(n) => total += n,
                        Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => return Err(e),
                    }
                }
                buf.truncate(total);
                Ok(buf)
            }
            _ => unreachable!(),
        }
    }

    /// Flush file data (not necessarily metadata) to disk.
    pub fn sync_data(&mut self) -> std::io::Result<()> {
        self.flush_buffers()?;
        self.get_file_ref().sync_data()
    }

    /// Flush both file data and metadata to disk.
    pub fn sync_all(&mut self) -> std::io::Result<()> {
        self.flush_buffers()?;
        self.get_file_ref().sync_all()
    }

    /// Return the current file size by seeking to the end.
    pub fn file_size(&mut self) -> std::io::Result<u64> {
        self.ensure_reader()?;
        match &mut self.mode {
            Mode::Reader(r) | Mode::ReadOnly(r) => {
                let pos = r.seek(SeekFrom::End(0))?;
                Ok(pos)
            }
            _ => unreachable!(),
        }
    }

    /// Flush any pending buffered writes.
    fn flush_buffers(&mut self) -> std::io::Result<()> {
        if let Mode::Writer(w) = &mut self.mode {
            w.flush()?;
        }
        Ok(())
    }

    /// Get a reference to the underlying File for sync operations.
    fn get_file_ref(&self) -> &File {
        match &self.mode {
            Mode::Writer(w) => w.get_ref(),
            Mode::Reader(r) | Mode::ReadOnly(r) => r.get_ref(),
            Mode::Transitioning => unreachable!("sync during transition"),
        }
    }
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
        let start = offset as usize;
        let end = start + len;
        if end > self.mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "mmap read past end: offset={} len={} file_size={}",
                    offset,
                    len,
                    self.mmap.len()
                ),
            ));
        }
        Ok(&self.mmap[start..end])
    }

    /// Read up to `max_len` bytes at `offset`. Returns a slice.
    pub fn read_at_most(&self, offset: u64, max_len: usize) -> &[u8] {
        let start = offset as usize;
        let end = std::cmp::min(start + max_len, self.mmap.len());
        if start >= self.mmap.len() {
            return &[];
        }
        &self.mmap[start..end]
    }
}
