use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

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
    /// Create a new file (truncating if it already exists) opened for
    /// read/write access. Uses a `BufWriter` by default.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            file: None,
            mode: Mode::Writer(BufWriter::with_capacity(DEFAULT_BUF_SIZE, file)),
        })
    }

    /// Open an existing file for read-only access. Uses a `BufReader`.
    pub fn open_read(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        Ok(Self {
            file: None,
            mode: Mode::ReadOnly(BufReader::with_capacity(DEFAULT_BUF_SIZE, file)),
        })
    }

    /// Open an existing file for read/write access.
    pub fn open_readwrite(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Self {
            file: None,
            mode: Mode::Writer(BufWriter::with_capacity(DEFAULT_BUF_SIZE, file)),
        })
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
}

#[cfg(feature = "mmap")]
impl MmapFileHandle {
    /// Open a file with memory mapping for read-only access.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { mmap })
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
