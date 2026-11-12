//! Chunk lookup against libhdf5 on the index shapes this crate's writer does
//! not produce: a paged fixed array, an extensible array past its first paged
//! data block (131k chunks), and two-level v2 and v1 B-trees. h5py writes
//! them, with gaps and one chunk written raw under filter mask 1, and records
//! what `read_direct_chunk` and a whole-dataset read return; both reads here
//! must return the same.

#![cfg(feature = "deflate")]

use std::path::{Path, PathBuf};

use rust_hdf5::H5File;

/// Interpreters tried in order when `RUST_HDF5_TEST_PYTHON` is unset, matching
/// `h5py_cross_validation`.
const TEST_PYTHONS: [&str; 2] = [
    "/Users/stevek/mamba/envs/bs2026.1/bin/python",
    "/home/stevek/micromamba/envs/tomo/bin/python",
];

fn python() -> Option<&'static str> {
    static PY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PY.get_or_init(|| {
        let candidates: Vec<String> = match std::env::var("RUST_HDF5_TEST_PYTHON") {
            Ok(p) => vec![p],
            Err(_) => TEST_PYTHONS.iter().map(|p| p.to_string()).collect(),
        };
        let found = candidates.iter().find(|c| Path::new(c).exists()).cloned();
        if found.is_none() {
            eprintln!("skipping chunk lookup cross-check: none of {candidates:?} present");
        }
        found
    })
    .as_deref()
}

/// Writes `latest.h5` (fixed array, extensible array, v2 B-tree) and
/// `earliest.h5` (v1 B-tree) into `argv[1]`, and per dataset `<name>.decoded`
/// (the whole dataset) and `<name>.chunks` (`coords mask hex` per sampled
/// chunk, `coords -` for one never written). Fails unless both B-trees are
/// at least two levels deep.
const GENERATE: &str = r#"
import os, sys
import h5py, numpy as np

out = sys.argv[1]

def values(n):
    return (np.arange(n) * 7 % 251).astype('u1')

def dump(f, name, coords):
    d = f[name]
    d[...].tofile(os.path.join(out, name + '.decoded'))
    # get_chunk_info_by_coord walks every chunk; one H5Dchunk_iter does not.
    allocated = set()
    d.id.chunk_iter(lambda info: allocated.add(info.chunk_offset))
    with open(os.path.join(out, name + '.chunks'), 'w') as m:
        for c in coords:
            off = tuple(ci * ch for ci, ch in zip(c, d.chunks))
            key = ','.join(map(str, c))
            if off not in allocated:
                m.write(f'{key} -\n')
            else:
                mask, raw = d.id.read_direct_chunk(off)
                m.write(f'{key} {mask} {raw.hex()}\n')

def deepest(path, sig, at, fmt):
    data = open(path, 'rb').read()
    found, i = [], data.find(sig)
    while i >= 0:
        found.append(data[i + at:i + at + np.dtype(fmt).itemsize])
        i = data.find(sig, i + 1)
    return max(int(np.frombuffer(b, fmt)[0]) for b in found)

N_EA = 140_000
grid2 = [(r, c) for r in range(200) for c in range(100) if (r * 100 + c) % 89 == 0]
grid2 += [(189, 99), (190, 0), (195, 4), (195, 5), (199, 99), (0, 0)]
lin = lambda n, extra: [(c,) for c in sorted(set(range(0, n, 97)) | {n - 1} | set(extra))]

latest = os.path.join(out, 'latest.h5')
with h5py.File(latest, 'w', libver='latest') as f:
    d = f.create_dataset('fa_paged', shape=(4100,), chunks=(1,), dtype='u1', compression='gzip')
    d[:2900] = values(2900)
    d.id.write_direct_chunk((2950,), b'\x2a', filter_mask=1)
    d = f.create_dataset('ea_paged', shape=(N_EA + 50,), maxshape=(None,), chunks=(1,), dtype='u1')
    d[:N_EA] = values(N_EA)
    d = f.create_dataset('bt2_deep', shape=(200, 100), maxshape=(None, None), chunks=(1, 1),
                         dtype='u1', compression='gzip')
    d[:190] = values(19000).reshape(190, 100)
    d.id.write_direct_chunk((195, 5), b'\x2a', filter_mask=1)
with h5py.File(latest, 'r') as f:
    dump(f, 'fa_paged', lin(4100, [1023, 1024, 1025, 2047, 2048, 2899, 2900, 2950, 3071, 3072, 3500]))
    dump(f, 'ea_paged', lin(N_EA + 50, list(range(130_900, 131_400, 7)) + [N_EA - 1, N_EA]))
    dump(f, 'bt2_deep', grid2)
assert deepest(latest, b'BTHD', 12, '<u2') >= 2, 'bt2_deep is not multi-level'

earliest = os.path.join(out, 'earliest.h5')
with h5py.File(earliest, 'w', libver='earliest') as f:
    d = f.create_dataset('v1_deep', shape=(200, 100), chunks=(1, 1), dtype='u1', compression='gzip')
    d[:190] = values(19000).reshape(190, 100)
    d.id.write_direct_chunk((195, 5), b'\x2a', filter_mask=1)
with h5py.File(earliest, 'r') as f:
    dump(f, 'v1_deep', grid2)
assert deepest(earliest, b'TREE', 5, 'u1') >= 2, 'v1_deep is not multi-level'
"#;

fn tmp_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("rust_hdf5_chunk_lookup_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Every mismatch between this crate's reads of `name` and h5py's.
fn mismatches(dir: &Path, file: &str, name: &str) -> Vec<String> {
    let f = H5File::open(dir.join(file)).unwrap();
    let ds = f.dataset(name).unwrap();
    let mut bad = Vec::new();
    let decoded = std::fs::read(dir.join(format!("{name}.decoded"))).unwrap();
    match ds.read_raw::<u8>() {
        Ok(got) if got == decoded => {}
        Ok(got) => {
            let at = got.iter().zip(&decoded).position(|(a, b)| a != b);
            bad.push(format!(
                "{name}: decoded read differs (first at {at:?}, lengths {} and {})",
                got.len(),
                decoded.len()
            ));
        }
        Err(e) => bad.push(format!("{name}: decoded read failed: {e}")),
    }
    let chunks = std::fs::read_to_string(dir.join(format!("{name}.chunks"))).unwrap();
    for line in chunks.lines() {
        let mut parts = line.split(' ');
        let key = parts.next().unwrap();
        let coords: Vec<usize> = key.split(',').map(|c| c.parse().unwrap()).collect();
        let got = ds.read_chunk_raw_at(&coords);
        match (parts.next().unwrap(), got) {
            ("-", Err(e)) if e.to_string().contains("has not been written") => {}
            ("-", got) => bad.push(format!("{name} [{key}]: never written, read {got:?}")),
            (mask, Ok((bytes, m))) => {
                let want = (unhex(parts.next().unwrap()), mask.parse::<u32>().unwrap());
                if (bytes.clone(), m) != want {
                    bad.push(format!(
                        "{name} [{key}]: read {:?}, h5py {want:?}",
                        (bytes, m)
                    ));
                }
            }
            (_, Err(e)) => bad.push(format!("{name} [{key}]: written, read failed: {e}")),
        }
    }
    bad
}

#[test]
fn every_index_shape_reads_as_h5py_reads_it() {
    let Some(py) = python() else { return };
    let dir = tmp_dir();
    let out = std::process::Command::new(py)
        .arg("-c")
        .arg(GENERATE)
        .arg(&dir)
        .output()
        .expect("spawn python");
    assert!(
        out.status.success(),
        "h5py generation failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bad: Vec<String> = [
        ("latest.h5", "fa_paged"),
        ("latest.h5", "ea_paged"),
        ("latest.h5", "bt2_deep"),
        ("earliest.h5", "v1_deep"),
    ]
    .iter()
    .flat_map(|(file, name)| mismatches(&dir, file, name))
    .collect();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
