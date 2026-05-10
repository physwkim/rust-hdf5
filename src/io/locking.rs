//! OS-level advisory file locking for HDF5 files.
//!
//! Mirrors the locking semantics of the HDF5 C library:
//!
//! - A read-only opener takes a **shared** lock; multiple readers are allowed.
//! - A read/write opener takes an **exclusive** lock; conflicts with any
//!   other lock holder.
//! - SWMR writers initially take an exclusive lock and **release** it once
//!   SWMR mode starts so concurrent SWMR readers can attach. (We don't
//!   downgrade exclusive→shared on the same handle: Windows'
//!   `LockFileEx` is mandatory and a same-handle unlock+shared-relock
//!   leaves subsequent `WriteFile` calls failing with
//!   `ERROR_LOCK_VIOLATION`. The HDF5 C library similarly relies on the
//!   SWMR file-format sentinel rather than OS locks during streaming.)
//!
//! Locks are released automatically when the underlying [`std::fs::File`]
//! is dropped (i.e. when the [`crate::io::file_handle::FileHandle`] closes).
//!
//! Locking can be controlled via:
//! - The `HDF5_USE_FILE_LOCKING` environment variable (`TRUE` / `FALSE` /
//!   `BEST_EFFORT`).
//! - The [`FileLocking`] enum passed to a `*_with_locking` constructor or
//!   to [`crate::file::H5FileOptions`].

use std::fs::File;
use std::io;

/// Whether the file should be locked shared (multiple readers) or
/// exclusive (sole owner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Shared lock — multiple holders allowed; conflicts with any
    /// exclusive lock.
    Shared,
    /// Exclusive lock — sole holder; conflicts with any shared or
    /// exclusive lock.
    Exclusive,
}

/// File-locking policy applied at file open time.
///
/// # Platform notes
///
/// On Unix (`flock(2)` / `fcntl(F_OFD_SETLK)`) the lock is **advisory**:
/// a handle without a lock can still read and write a file that another
/// handle has locked. Setting [`FileLocking::Disabled`] or
/// [`FileLocking::BestEffort`] therefore lets the opener bypass another
/// process's lock at the cost of safety.
///
/// On Windows (`LockFileEx`) the lock is **mandatory**: while one
/// handle holds an exclusive range lock, no other handle (regardless
/// of locking policy) can read or write that range — `WriteFile` and
/// `ReadFile` return `ERROR_LOCK_VIOLATION` (33). `Disabled` and
/// `BestEffort` only control whether *we* try to acquire a lock, not
/// whether the OS enforces locks held by other handles. The HDF5 C
/// library has the same limitation on Windows.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FileLocking {
    /// Acquire the lock; fail to open if it cannot be acquired
    /// (the HDF5 C-library default).
    #[default]
    Enabled,
    /// Skip locking entirely. On Windows, OS-level locks held by other
    /// handles still apply.
    Disabled,
    /// Try to acquire the lock; if the filesystem doesn't support
    /// locking (e.g. NFS), proceed without one. On Unix this also
    /// proceeds when the lock is contended; on Windows the resulting
    /// reads/writes still fail at the OS level if another handle holds
    /// a conflicting `LockFileEx` lock.
    BestEffort,
}

impl FileLocking {
    /// Returns the policy implied by the `HDF5_USE_FILE_LOCKING`
    /// environment variable, falling back to [`FileLocking::Enabled`].
    pub fn from_env() -> Self {
        Self::parse_env_value(std::env::var("HDF5_USE_FILE_LOCKING").ok().as_deref())
    }

    /// Returns the policy implied by the env var if set, otherwise `default`.
    pub fn from_env_or(default: FileLocking) -> Self {
        match std::env::var("HDF5_USE_FILE_LOCKING").ok().as_deref() {
            None => default,
            Some(v) => Self::parse_env_value(Some(v)),
        }
    }

    pub(crate) fn parse_env_value(value: Option<&str>) -> Self {
        match value {
            None => FileLocking::Enabled,
            Some(v) => {
                let trimmed = v.trim();
                if trimmed.eq_ignore_ascii_case("FALSE")
                    || trimmed == "0"
                    || trimmed.eq_ignore_ascii_case("OFF")
                    || trimmed.eq_ignore_ascii_case("NO")
                {
                    FileLocking::Disabled
                } else if trimmed.eq_ignore_ascii_case("BEST_EFFORT")
                    || trimmed.eq_ignore_ascii_case("BEST-EFFORT")
                    || trimmed.eq_ignore_ascii_case("BESTEFFORT")
                {
                    FileLocking::BestEffort
                } else {
                    // Any other value (TRUE/1/ON/YES or unrecognized) → enabled.
                    FileLocking::Enabled
                }
            }
        }
    }
}

/// Attempt to acquire the requested lock on `file`.
///
/// Returns `Ok(true)` if the lock was acquired, `Ok(false)` if locking
/// was skipped (policy = Disabled) or the attempt failed under
/// [`FileLocking::BestEffort`]. Returns `Err` only when policy is
/// [`FileLocking::Enabled`] and the lock could not be obtained.
pub fn try_acquire(file: &File, mode: LockMode, policy: FileLocking) -> io::Result<bool> {
    if matches!(policy, FileLocking::Disabled) {
        return Ok(false);
    }

    let attempt = match mode {
        LockMode::Shared => file.try_lock_shared(),
        LockMode::Exclusive => file.try_lock(),
    };

    match attempt {
        Ok(()) => Ok(true),
        Err(std::fs::TryLockError::WouldBlock) => match policy {
            FileLocking::Enabled => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "unable to lock file: another process holds a conflicting lock",
            )),
            FileLocking::BestEffort => Ok(false),
            FileLocking::Disabled => unreachable!(),
        },
        Err(std::fs::TryLockError::Error(e)) => match policy {
            FileLocking::Enabled => Err(e),
            FileLocking::BestEffort => Ok(false),
            FileLocking::Disabled => unreachable!(),
        },
    }
}

/// Release any lock currently held on `file`. Safe to call when
/// no lock is held — the underlying syscall is idempotent in
/// practice on the platforms we target.
pub fn release(file: &File) -> io::Result<()> {
    file.unlock()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_value_defaults_to_enabled() {
        assert_eq!(FileLocking::parse_env_value(None), FileLocking::Enabled);
    }

    #[test]
    fn parse_env_value_recognizes_disabled() {
        for v in ["FALSE", "false", "0", "off", "no"] {
            assert_eq!(
                FileLocking::parse_env_value(Some(v)),
                FileLocking::Disabled,
                "value: {v}",
            );
        }
    }

    #[test]
    fn parse_env_value_recognizes_best_effort() {
        for v in ["BEST_EFFORT", "best_effort", "best-effort", "BestEffort"] {
            assert_eq!(
                FileLocking::parse_env_value(Some(v)),
                FileLocking::BestEffort,
                "value: {v}",
            );
        }
    }

    #[test]
    fn parse_env_value_recognizes_enabled() {
        for v in ["TRUE", "true", "1", "on", "yes", "garbage"] {
            assert_eq!(
                FileLocking::parse_env_value(Some(v)),
                FileLocking::Enabled,
                "value: {v}",
            );
        }
    }

    #[test]
    fn try_acquire_disabled_is_noop() {
        let dir =
            std::env::temp_dir().join(format!("rust_hdf5_lock_disabled_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("noop.bin");
        let f = std::fs::File::create(&path).unwrap();
        let acquired = try_acquire(&f, LockMode::Exclusive, FileLocking::Disabled).unwrap();
        assert!(!acquired, "Disabled policy must not acquire a lock");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn try_acquire_exclusive_then_shared_fails() {
        let dir = std::env::temp_dir().join(format!("rust_hdf5_lock_excl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("conflict.bin");
        let f1 = std::fs::File::create(&path).unwrap();
        assert!(try_acquire(&f1, LockMode::Exclusive, FileLocking::Enabled).unwrap());
        let f2 = std::fs::OpenOptions::new().read(true).open(&path).unwrap();
        let res = try_acquire(&f2, LockMode::Shared, FileLocking::Enabled);
        assert!(res.is_err(), "expected lock conflict");
        // Best-effort should silently fall through.
        let res2 = try_acquire(&f2, LockMode::Shared, FileLocking::BestEffort).unwrap();
        assert!(!res2, "best-effort must report unsuccessful lock as false");
        release(&f1).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shared_locks_coexist() {
        let dir =
            std::env::temp_dir().join(format!("rust_hdf5_lock_shared_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shared.bin");
        std::fs::File::create(&path).unwrap();
        let f1 = std::fs::OpenOptions::new().read(true).open(&path).unwrap();
        let f2 = std::fs::OpenOptions::new().read(true).open(&path).unwrap();
        assert!(try_acquire(&f1, LockMode::Shared, FileLocking::Enabled).unwrap());
        assert!(try_acquire(&f2, LockMode::Shared, FileLocking::Enabled).unwrap());
        release(&f1).unwrap();
        release(&f2).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn release_then_relock_works() {
        let dir =
            std::env::temp_dir().join(format!("rust_hdf5_lock_release_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("release.bin");
        let f1 = std::fs::File::create(&path).unwrap();
        assert!(try_acquire(&f1, LockMode::Exclusive, FileLocking::Enabled).unwrap());
        // Release; another opener should now be able to take a fresh lock.
        release(&f1).unwrap();

        let f2 = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(try_acquire(&f2, LockMode::Exclusive, FileLocking::Enabled).unwrap());

        release(&f2).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
