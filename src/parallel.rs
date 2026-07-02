//! Process-wide thread pool for rust-hdf5's internal parallelism.
//!
//! rust-hdf5 parallelizes chunk I/O, compression, and decompression with rayon
//! (the `parallel` feature). Rather than run on rayon's *global* pool — which
//! defaults to every logical core and which the host application also owns —
//! all of rust-hdf5's parallel sections run on this private pool, sized to
//! **half** the logical cores by default. This keeps a single HDF5 read or
//! write from saturating the machine and starving co-running processes, and it
//! never calls [`rayon::ThreadPoolBuilder::build_global`], so it never fights
//! the application for configuration of the global pool. If the pool cannot be
//! built (the OS refuses to spawn its worker threads), each parallel section
//! falls back to serial execution rather than panicking.
//!
//! The thread count can be overridden with the `RUST_HDF5_IO_THREADS`
//! environment variable (a positive integer). An unset, zero, or unparseable
//! value falls back to the half-the-cores default (at least one thread).

use std::sync::OnceLock;

use rayon::ThreadPool;

/// Worker-thread count for [`io_pool`]: `RUST_HDF5_IO_THREADS` when set to a
/// positive integer, otherwise half the logical cores (at least 1).
fn thread_count() -> usize {
    if let Ok(v) = std::env::var("RUST_HDF5_IO_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|c| c.get() / 2)
        .unwrap_or(1)
        .max(1)
}

/// The shared, lazily-built I/O thread pool, or `None` if the OS refused to
/// spawn its worker threads (e.g. `RLIMIT_NPROC` exhaustion, a very large
/// `RUST_HDF5_IO_THREADS`). Every rust-hdf5 rayon section runs inside
/// `io_pool().install(...)` when the pool is present, capping the crate's
/// parallelism at [`thread_count`] threads regardless of the global rayon pool;
/// when it is `None` the caller falls back to serial execution, so a pool-build
/// failure degrades performance instead of panicking across the library
/// boundary. The `None` decision is cached — the build is not retried, and the
/// crate runs serially for the rest of the process.
pub(crate) fn io_pool() -> Option<&'static ThreadPool> {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count())
            .thread_name(|i| format!("hdf5-io-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_is_capped_and_reused() {
        let n = thread_count();
        assert!(n >= 1, "pool must have at least one thread");
        // Default (no env override) must not exceed half the logical cores.
        if std::env::var_os("RUST_HDF5_IO_THREADS").is_none() {
            let cores = std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1);
            assert!(
                n <= (cores / 2).max(1),
                "default pool must be <= half the cores"
            );
        }
        // The pool is a single shared instance across calls (when it builds;
        // build only fails if the OS refuses threads, in which case callers run
        // serially and there is nothing to assert here).
        if let Some(pool) = io_pool() {
            let a = io_pool().unwrap() as *const ThreadPool;
            let b = io_pool().unwrap() as *const ThreadPool;
            assert_eq!(a, b, "io_pool must return the same shared pool");
            assert_eq!(pool.current_num_threads(), n);
        }
    }
}
