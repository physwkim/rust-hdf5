//! Process-wide thread pool for rust-hdf5's internal parallelism.
//!
//! rust-hdf5 parallelizes chunk I/O, compression, and decompression with rayon
//! (the `parallel` feature). Rather than run on rayon's *global* pool — which
//! defaults to every logical core and which the host application also owns —
//! all of rust-hdf5's parallel sections run on this private pool, sized to
//! **half** the logical cores by default. This keeps a single HDF5 read or
//! write from saturating the machine and starving co-running processes, and it
//! never calls [`rayon::ThreadPoolBuilder::build_global`], so it never fights
//! the application for configuration of the global pool.
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

/// The shared, lazily-built I/O thread pool. Every rust-hdf5 rayon section runs
/// inside `io_pool().install(...)`, so all of the crate's parallelism is capped
/// at [`thread_count`] threads regardless of the global rayon pool.
pub(crate) fn io_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count())
            .thread_name(|i| format!("hdf5-io-{i}"))
            .build()
            .expect("failed to build rust-hdf5 I/O thread pool")
    })
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
        // The pool is a single shared instance across calls.
        let a = io_pool() as *const ThreadPool;
        let b = io_pool() as *const ThreadPool;
        assert_eq!(a, b, "io_pool must return the same shared pool");
        assert_eq!(io_pool().current_num_threads(), n);
    }
}
