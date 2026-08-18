#!/usr/bin/env python3
"""Build and run the two mirrored performance probes and print a table.

The C side compiles perf/probe.c with h5cc (default: the pinned libhdf5
1.14.6 in ~/micromamba/envs/tomo); the Rust side is the crate's own
src/bin/perf_probe.rs built --release. Both time each workload in-process;
this runner reports the per-workload minimum, which is the least-noise
estimate of the work itself.

Environment:
  RUST_HDF5_PERF_PREFIX   libhdf5 install prefix to compile the C probe
                          against (default ~/micromamba/envs/tomo)
  RUST_HDF5_PERF_WORKDIR  scratch dir (default /dev/shm/rust-hdf5-perf)
  RUST_HDF5_PERF_ONLY     comma-separated workload subset
"""

import os
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKLOADS = [
    ("contig-write", 5),
    ("contig-read", 5),
    ("chunked-write", 5),
    ("chunked-read", 5),
    ("deflate-write", 3),
    ("deflate-read", 5),
    ("slice-read", 5),
    ("small-write", 3),
    ("small-read", 5),
    ("attr-write", 3),
    ("append", 5),
]


def build(prefix):
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "perf_probe"],
        cwd=ROOT,
        check=True,
    )
    c_bin = ROOT / "target" / "release" / "perf_probe_c"
    # Conda's h5cc insists on the conda-internal compiler; link with the
    # system cc against the env's libhdf5 directly instead.
    subprocess.run(
        [
            "cc",
            "-O2",
            str(ROOT / "perf" / "probe.c"),
            f"-I{prefix}/include",
            f"-L{prefix}/lib",
            f"-Wl,-rpath,{prefix}/lib",
            "-lhdf5",
            "-o",
            str(c_bin),
        ],
        check=True,
    )
    return ROOT / "target" / "release" / "perf_probe", c_bin


def run_probe(binary, workdir, workload, reps):
    out = subprocess.run(
        [str(binary), str(workdir), workload, str(reps)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ns = [
        int(line.split()[-1])
        for line in out.splitlines()
        if line.startswith(f"BENCH {workload} ")
    ]
    if len(ns) != reps:
        raise RuntimeError(f"{binary} {workload}: expected {reps} reps, got {ns}")
    return ns


def main():
    prefix = os.environ.get(
        "RUST_HDF5_PERF_PREFIX",
        str(pathlib.Path.home() / "micromamba" / "envs" / "tomo"),
    )
    workdir = pathlib.Path(
        os.environ.get("RUST_HDF5_PERF_WORKDIR", "/dev/shm/rust-hdf5-perf")
    )
    workdir.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("RUST_HDF5_PERF_ONLY")
    matrix = [
        (w, r) for w, r in WORKLOADS if only is None or w in only.split(",")
    ]

    rust_bin, c_bin = build(prefix)
    print(f"{'workload':<14} {'C min ms':>10} {'rust min ms':>12} "
          f"{'ratio':>7}  {'C med':>10} {'rust med':>10}")
    for workload, reps in matrix:
        c_ns = run_probe(c_bin, workdir, workload, reps)
        r_ns = run_probe(rust_bin, workdir, workload, reps)
        cmin, rmin = min(c_ns) / 1e6, min(r_ns) / 1e6
        cmed = statistics.median(c_ns) / 1e6
        rmed = statistics.median(r_ns) / 1e6
        print(f"{workload:<14} {cmin:>10.2f} {rmin:>12.2f} "
              f"{rmin / cmin:>6.2f}x  {cmed:>10.2f} {rmed:>10.2f}")
        sys.stdout.flush()
    for leftover in workdir.iterdir():
        leftover.unlink()


if __name__ == "__main__":
    main()
