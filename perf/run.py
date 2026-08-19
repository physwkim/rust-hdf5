#!/usr/bin/env python3
"""Build and run the two mirrored performance probes and print a table.

The C side compiles perf/probe.c with h5cc (default: the pinned libhdf5
1.14.6 in ~/micromamba/envs/tomo); the Rust side is the crate's own
src/bin/perf_probe.rs built --release, three times over: the default
build, the `parallel` build, and the `mmap` build, each into its own
target dir so they coexist. Both time each workload in-process; this
runner reports the per-workload minimum, which is the least-noise
estimate of the work itself.

Environment:
  RUST_HDF5_PERF_PREFIX   libhdf5 install prefix to compile the C probe
                          against (default ~/micromamba/envs/tomo)
  RUST_HDF5_PERF_WORKDIR  scratch dir (default /dev/shm/rust-hdf5-perf)
  RUST_HDF5_PERF_ONLY     comma-separated workload subset
"""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKLOADS = [
    ("contig-write", 5),
    ("contig-read", 5),
    # The mmap column runs read_mapped, every other column read_raw: the
    # workload is "obtain the data and sum it", and the view is what changes
    # between builds. See src/bin/perf_probe.rs.
    ("contig-view", 5),
    # Buffer-reuse steady state: open and the first read are setup, timed
    # reads land in an already-faulted buffer.
    ("into-read", 5),
    ("into-slice", 5),
    ("chunked-write", 5),
    ("chunked-read", 5),
    # The buffer-reuse pair on an unfiltered chunked dataset: full rereads
    # and slice-read's random selections into a kept buffer.
    ("chunked-into-read", 5),
    ("chunked-into-slice", 5),
    ("deflate-write", 3),
    ("deflate-read", 5),
    ("deflate-slice", 5),
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
    # The rayon variant builds into its own target dir so the two probes
    # coexist; libhdf5's sec2 driver is single-threaded by design, so this
    # column shows what the thread pool buys on a 96-core box.
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--bin",
            "perf_probe",
            "--features",
            "parallel",
            "--target-dir",
            str(ROOT / "target" / "parallel"),
        ],
        cwd=ROOT,
        check=True,
    )
    # Likewise for the mmap variant: a read-only open serves its reads from a
    # whole-file map instead of pread, so this column is the read side's.
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--bin",
            "perf_probe",
            "--features",
            "mmap",
            "--target-dir",
            str(ROOT / "target" / "mmap"),
        ],
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
    return (
        ROOT / "target" / "release" / "perf_probe",
        ROOT / "target" / "parallel" / "release" / "perf_probe",
        ROOT / "target" / "mmap" / "release" / "perf_probe",
        c_bin,
    )


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

    rust_bin, par_bin, mmap_bin, c_bin = build(prefix)
    print(f"{'workload':<14} {'C min ms':>10} {'rust min ms':>12} "
          f"{'ratio':>7} {'par min ms':>11} {'par':>7} "
          f"{'mmap min ms':>12} {'mmap':>7}")
    for workload, reps in matrix:
        # Interleave C and Rust invocations: this box swings whole-process
        # throughput between invocations, so running all C reps before all
        # Rust reps folds that drift into the ratio.
        per_round = max(2, (reps + 1) // 2)
        c_ns, r_ns, p_ns, m_ns = [], [], [], []
        for _ in range(2):
            c_ns += run_probe(c_bin, workdir, workload, per_round)
            r_ns += run_probe(rust_bin, workdir, workload, per_round)
            p_ns += run_probe(par_bin, workdir, workload, per_round)
            m_ns += run_probe(mmap_bin, workdir, workload, per_round)
        cmin, rmin = min(c_ns) / 1e6, min(r_ns) / 1e6
        pmin, mmin = min(p_ns) / 1e6, min(m_ns) / 1e6
        print(f"{workload:<14} {cmin:>10.2f} {rmin:>12.2f} "
              f"{rmin / cmin:>6.2f}x {pmin:>11.2f} {pmin / cmin:>6.2f}x "
              f"{mmin:>12.2f} {mmin / cmin:>6.2f}x")
        sys.stdout.flush()
    for leftover in workdir.iterdir():
        leftover.unlink()


if __name__ == "__main__":
    main()
