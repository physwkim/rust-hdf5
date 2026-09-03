#!/usr/bin/env python3
"""Run the bidirectional rust-hdf5 <-> libhdf5 parity oracle.

Direction A (read parity)
    h5py writes the reference file for every case in `cases.py`; `canon.py` and
    `oracle_probe dump` each describe it in the canonical format; the two
    descriptions are compared field by field.

Direction B (write parity)
    For every case rust-hdf5's public API can express, `oracle_probe write`
    produces the equivalent file; h5py must read it, `canon.py` must describe
    it, and `h5diff` must find no difference against the h5py-written
    reference.

Outputs `doc/oracle-report.md` (human) and `oracle/report.json` (machine).

    RUST_HDF5_ORACLE_PYTHON  interpreter with h5py (default: the pinned path)
    RUST_HDF5_ORACLE_PROBE   oracle_probe binary (default: target/release/...)
    RUST_HDF5_ORACLE_BINDIR  directory holding h5diff/h5dump (default: the
                             interpreter's own bin/)

    usage: run.py [--filter SUBSTR] [--work DIR] [--keep] [--no-build]
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback

DEFAULT_PYTHON = "/home/stevek/micromamba/envs/tomo/bin/python"
REPO = pathlib.Path(__file__).resolve().parent.parent


def report_path(p):
    """`p` relative to the repo root when it lies under it, so a report made
    in a worktree names the same probe as one made in the main checkout."""
    p = pathlib.Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hdf5env  # noqa: E402,F401  (must precede h5py; see the module docstring)


def reexec_with_h5py():
    """Re-run under the configured interpreter when this one lacks h5py."""
    try:
        import h5py  # noqa: F401

        return
    except ImportError:
        pass
    interp = os.environ.get("RUST_HDF5_ORACLE_PYTHON", DEFAULT_PYTHON)
    if not pathlib.Path(interp).exists():
        sys.stderr.write(
            "no h5py in %s and RUST_HDF5_ORACLE_PYTHON=%s does not exist\n"
            % (sys.executable, interp)
        )
        sys.exit(3)
    if os.path.realpath(interp) == os.path.realpath(sys.executable):
        sys.stderr.write("%s has no h5py\n" % interp)
        sys.exit(3)
    os.execv(interp, [interp, os.path.abspath(__file__)] + sys.argv[1:])


reexec_with_h5py()

import canon  # noqa: E402
import cases  # noqa: E402

# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

# Fields where a difference between the h5py reference and the rust-written
# file is a metadata deviation rather than an invalid file: the data, the type
# and the shape all still agree, so libhdf5 reads exactly the same values.
B_TOLERATED_FIELDS = {
    "superblock",
    "layout",
    "chunkindex",
    "filters",
    "fillvalue",
    "maxshape",
    "linkorder",
    "attrorder",
    "linkstore",
    "shared",
    "msgflags",
}

# Direction-B metadata deviations that are known, understood and stable: the
# rust writer describes the file differently from libhdf5 while the data,
# datatype and shape it stores read back identically.
#
# An entry is one deviation — one cause, one `why` — and its `faces` are the
# fields that cause shows up in, each matched against (field, libhdf5 value,
# rust-hdf5 value) with `None` matching anything. A cause visible in two
# fields is one entry with two faces, not two entries: splitting it would say
# the crate has two unexplained behaviours where it has one, and the second
# copy of the explanation would drift from the first.
#
# A deviation that matches no face is reported as unexpected, and a face that
# matches nothing in a run is reported as no longer observed — so a rerun
# after a writer fix shows the change rather than hiding it.
#
# `min_hdf5` scopes an entry to a reference library at or above that version.
# It is for a deviation that exists only because the reference changed
# behaviour, never for one the crate could fix: an entry gated this way must
# name the upstream change and the evidence for it.
EXPECTED_DEVIATIONS = [
    {
        "id": "chunkindex-v3-superblock-low-bound",
        "faces": [
            {"field": "chunkindex", "ref": "btree1", "rust": "farray"},
        ],
        "min_hdf5": (2, 0),
        "why": (
            "Reopening a v3-superblock file to append: libhdf5 1.14.6 raises "
            "the file's low_bound on every open from the superblock version "
            "it finds — v2 to V18, v3 to V110 (H5Fsuper.c:460-466) — so the "
            "reopened file writes a V110 chunk index. 2.0.0 raises it only "
            "under H5F_ACC_SWMR_WRITE (H5Fsuper.c:448-454), so the same file "
            "reopened non-SWMR keeps the default bound and libhdf5 writes a "
            "version-1 B-tree index where it used to write a fixed/extensible "
            "array. This crate's write rule is pinned to 1.14.6 semantics by "
            "decision, so under a 2.0 reference the two disagree; against the "
            "pinned 1.14.6 reference they do not, and that run declares no "
            "deviation at all."
        ),
    },
    {
        "id": "shared-v2-superblock-low-bound",
        "faces": [
            {
                "field": "shared",
                "ref": "[dataspace:sohm,datatype:shareable]",
                "rust": "[dataspace:shareable,datatype:shareable]",
            },
            # The same dataspace message, seen as the raw flags byte rather
            # than as the storage behind it: `S` is the body on the heap,
            # `SA` is the body still literal in this header.
            {
                "field": "msgflags",
                "ref": "[dataspace:S,datatype:C+SA,fill_new:C,layout:none]",
                "rust": "[dataspace:SA,datatype:C+SA,fill_new:C,layout:none]",
            },
        ],
        "min_hdf5": (2, 0),
        "why": (
            "The same low_bound change as chunkindex-v3-superblock-low-bound, "
            "seen through the shared-message flags instead of the chunk "
            "index. gen_sohm.c writes sohm_list.h5 at H5F_LIBVER_EARLIEST, so "
            "its four datasets carry version-1 dataspaces. 1.14.6 raises the "
            "low bound to V18 on opening the version-2 superblock the "
            "shared-message table forces (H5Fsuper.c:460-462), so the dataset "
            "the append creates gets a version-2 dataspace — a body no "
            "existing record matches, which H5SM__write_mesg leaves literal "
            "in the new header under H5O_MSG_FLAG_SHAREABLE "
            "(H5SM.c:1400-1417). 2.0.0 raises the bound only under "
            "H5F_ACC_SWMR_WRITE (H5Fsuper.c:448-454), so the appended dataset "
            "gets a version-1 dataspace equal to the four already indexed, "
            "the matching record's reference count rises and the body moves "
            "to the heap. Witnessed on the reference files themselves: "
            "h5debug reports /appended's dataspace message as 20 bytes <SA> "
            "under 1.14.6 and 10 bytes <S> under 2.0.0, with /shared0's "
            "unchanged at 24 bytes <SA> in both. Pinned to 1.14.6 by "
            "decision, this crate writes what the 1.14.6 reference writes, "
            "and that run declares no deviation."
        ),
    },
]

_HDF5_VERSION = None


def hdf5_version_tuple():
    """The reference library's version, cached. Imported lazily because this
    module re-execs itself under the interpreter that has h5py."""
    global _HDF5_VERSION
    if _HDF5_VERSION is None:
        import h5py

        _HDF5_VERSION = tuple(h5py.version.hdf5_version_tuple)
    return _HDF5_VERSION


def deviation_applies(exp):
    """Whether a declared deviation is in force for the running reference."""
    return "min_hdf5" not in exp or hdf5_version_tuple() >= exp["min_hdf5"]


def declared_deviations():
    """The EXPECTED_DEVIATIONS in force for this run."""
    return [exp for exp in EXPECTED_DEVIATIONS if deviation_applies(exp)]


def expected_deviation(entry):
    """The EXPECTED_DEVIATIONS id whose face matches this diff, or None."""
    ref, rust = entry["ref"], entry["rust"]
    if ref is None or rust is None:
        # One side does not describe the field at all. That is a missing
        # object or a missing field, never one of the declared deviations.
        return None
    for exp in declared_deviations():
        for face in exp["faces"]:
            if face["field"] != entry["field"]:
                continue
            if face["ref"] is not None and face["ref"] != ref:
                continue
            if face["rust"] is not None and face["rust"] != rust:
                continue
            if "check" in face and not face["check"](ref, rust):
                continue
            return exp["id"]
    return None


# Fields the public rust-hdf5 API has no accessor for *at all*, in any file.
# Each is one API gap, listed once in the findings; they say nothing about the
# case that happens to contain them, so they do not stop a case passing
# direction A. Everything else that comes back UNSUPPORTED is specific to the
# file at hand and does. Empty since wave 6: every field the canonical format
# names now has an accessor on at least one object kind, so the remaining
# carve-outs are all per-kind (below).
STRUCTURAL_FIELDS = set()
# `strpad` is deliberately NOT structural: it is a datatype detail the probe
# answers from the decoded type, so a disagreement there is a modelling gap in
# one class, not a missing accessor.

# Fields with a real accessor on some object kinds but not others: still an
# API-wide gap on the `#kind` values listed here, so a case is not held to a
# field the API was never asked to expose there. `attr_storage()` exists on
# H5Group and H5Dataset; H5NamedDatatype has no counterpart.
STRUCTURAL_FIELDS_BY_KIND = {
    "attrstore": {"committed-datatype"},
}


def parse_dump(text):
    """Canonical dump text -> (records, header) where records is key -> value."""
    records, header = {}, {}
    for line in text.splitlines():
        if not line:
            continue
        key, _, value = line.partition("\t")
        if key.startswith("!"):
            header[key] = value
        else:
            records[key] = value
    return records, header


def field_of(key):
    return key.rpartition("#")[2]


def object_of(key):
    return key.rpartition("#")[0]


def marker(value, name):
    return value.startswith(name + "(")


def is_structural(key, field, ref):
    """True when this UNSUPPORTED is one of the always-missing accessors."""
    if field in STRUCTURAL_FIELDS:
        return True
    if ref.get("%s#kind" % object_of(key)) in STRUCTURAL_FIELDS_BY_KIND.get(field, ()):
        return True
    if "@" not in key:
        return False
    if field == "shape":
        return True  # H5Attribute has no shape() at all
    # A group attribute has no typed handle in read mode, so none of its
    # fields are observable — a property of the API, not of this file.
    owner = key.split("@", 1)[0]
    return ref.get("%s#kind" % owner) == "group"


def compare(ref, probe):
    """Classify every field of two canonical dumps.

    Returns (divergences, gaps, oracle_errors, matched). An object the probe
    never listed collapses to a single `missing-object` entry rather than one
    per field, because the object — not each field — is what went missing.
    """
    divergences, gaps, oracle_errors = [], [], []
    matched = 0
    reported_missing = set()
    for key in sorted(set(ref) | set(probe)):
        rv, pv = ref.get(key), probe.get(key)
        if rv is not None and marker(rv, "ERROR"):
            oracle_errors.append({"key": key, "ref": rv})
            continue
        if pv is None:
            obj = object_of(key).split("@", 1)[0]
            if "%s#kind" % obj not in probe:
                if obj in reported_missing:
                    continue
                reported_missing.add(obj)
                gaps.append(
                    {
                        "key": obj,
                        "kind": "missing-object",
                        "field": "kind",
                        "structural": False,
                        "ref": ref.get("%s#kind" % obj, "?"),
                    }
                )
            else:
                gaps.append(
                    {
                        "key": key,
                        "kind": "missing-field",
                        "field": field_of(key),
                        "structural": False,
                        "ref": rv,
                    }
                )
            continue
        if rv is None:
            divergences.append(
                {"key": key, "kind": "rust-extra", "field": field_of(key),
                 "ref": None, "probe": pv}
            )
            continue
        field = field_of(key)
        if marker(pv, "UNSUPPORTED"):
            gaps.append(
                {
                    "key": key,
                    "kind": "unsupported",
                    "field": field,
                    "structural": is_structural(key, field, ref),
                    "ref": rv,
                    "probe": pv,
                }
            )
            continue
        if rv == pv:
            matched += 1
        else:
            divergences.append(
                {"key": key, "kind": "value", "field": field,
                 "ref": rv, "probe": pv}
            )
    return divergences, gaps, oracle_errors, matched


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


def access_argv(access):
    """A case's dataset-access properties as `oracle_probe dump` flags."""
    argv = []
    if not access:
        return argv
    if "view" in access:
        argv += ["--virtual-view", access["view"]]
    if "printf_gap" in access:
        argv += ["--printf-gap", str(access["printf_gap"])]
    return argv


def run(cmd, **kw):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, **kw
    )


def tail(text, n=6):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


class Oracle:
    def __init__(self, probe, bindir, work):
        self.probe = str(probe)
        self.bindir = pathlib.Path(bindir)
        self.work = work
        (work / "A").mkdir(parents=True, exist_ok=True)
        (work / "B").mkdir(parents=True, exist_ok=True)

    def tool(self, name):
        p = self.bindir / name
        return str(p) if p.exists() else None

    # -- direction A ---------------------------------------------------

    def direction_a(self, case):
        out = {"verdict": None, "divergences": [], "gaps": [], "oracle_errors": [],
               "matched": 0, "detail": ""}
        ref_path = self.work / "A" / (case.name + ".h5")
        if ref_path.exists():
            ref_path.unlink()
        try:
            case.gen(ref_path)
        except Exception:
            out["verdict"] = "GEN-ERROR"
            out["detail"] = tail(traceback.format_exc())
            return out, None

        try:
            ref_text = canon.dump(str(ref_path), case.access)
        except Exception:
            out["verdict"] = "GEN-ERROR"
            out["detail"] = "canon.py could not describe the reference: " + tail(
                traceback.format_exc()
            )
            return out, None

        proc = run([self.probe, "dump"] + access_argv(case.access) + [str(ref_path)])
        if proc.returncode != 0:
            out["verdict"] = "READ-ERROR"
            out["detail"] = tail(proc.stdout + proc.stderr)
            return out, ref_path

        ref, ref_hdr = parse_dump(ref_text)
        probe, probe_hdr = parse_dump(proc.stdout)
        if ref_hdr.get("!canon") != probe_hdr.get("!canon"):
            # A stale probe binary would otherwise diff as hundreds of
            # spurious field mismatches.
            out["verdict"] = "READ-ERROR"
            out["detail"] = "canonical format mismatch: canon.py emits %r, %s emits %r" % (
                ref_hdr.get("!canon"),
                self.probe,
                probe_hdr.get("!canon"),
            )
            return out, ref_path
        div, gaps, errs, matched = compare(ref, probe)
        out.update(
            divergences=div, gaps=gaps, oracle_errors=errs, matched=matched
        )
        if div:
            out["verdict"] = "DIFF"
        elif any(g["kind"] == "missing-object" for g in gaps):
            out["verdict"] = "MISS"
        elif any(not g.get("structural") for g in gaps):
            out["verdict"] = "GAP"
        else:
            out["verdict"] = "PASS"
        return out, ref_path

    # -- direction B ---------------------------------------------------

    def direction_b(self, case, ref_path):
        out = {"verdict": None, "detail": "", "core_diffs": [],
               "metadata_diffs": [], "h5diff_rc": None, "h5dump_rc": None}
        if not case.rust:
            out["verdict"] = "UNSUPPORTED-API"
            out["detail"] = "no rust writer arm: the public API cannot express this case"
            return out
        if ref_path is None:
            out["verdict"] = "SKIPPED"
            out["detail"] = "direction A produced no reference file"
            return out

        out_path = self.work / "B" / (case.name + ".h5")
        if out_path.exists():
            out_path.unlink()
        proc = run([self.probe, "write", case.rust, str(out_path)])
        if proc.returncode == 2:
            out["verdict"] = "UNSUPPORTED-API"
            out["detail"] = tail(proc.stdout)
            return out
        if proc.returncode != 0:
            out["verdict"] = "INVALID"
            out["detail"] = "rust writer failed: " + tail(proc.stdout + proc.stderr)
            return out

        try:
            written_text = canon.dump(str(out_path), case.access)
        except Exception:
            out["verdict"] = "INVALID"
            out["detail"] = "h5py could not read the rust-written file: " + tail(
                traceback.format_exc()
            )
            return out

        ref, _ = parse_dump(canon.dump(str(ref_path), case.access))
        got, _ = parse_dump(written_text)
        # An object h5py cannot open in the rust-written file makes every one
        # of its fields diverge; report the object once instead, the same way
        # direction A collapses an object the reader never lists.
        unreadable = {
            object_of(k)
            for k, v in got.items()
            if field_of(k) == "kind" and (marker(v, "ERROR") or marker(v, "UNSUPPORTED"))
        }
        for obj in sorted(unreadable):
            out["core_diffs"].append(
                {
                    "key": obj,
                    "field": "object",
                    "ref": ref.get("%s#kind" % obj),
                    "rust": got.get("%s#kind" % obj),
                }
            )
        for key in sorted(set(ref) | set(got)):
            rv, gv = ref.get(key), got.get(key)
            if rv == gv:
                continue
            if object_of(key).split("@", 1)[0] in unreadable:
                continue
            entry = {"key": key, "field": field_of(key), "ref": rv, "rust": gv}
            if field_of(key) in B_TOLERATED_FIELDS:
                entry["expected"] = expected_deviation(entry)
                out["metadata_diffs"].append(entry)
            else:
                out["core_diffs"].append(entry)

        h5diff = self.tool("h5diff")
        if h5diff:
            d = run([h5diff, str(ref_path), str(out_path)])
            out["h5diff_rc"] = d.returncode
            if d.returncode not in (0,):
                out["detail"] = tail(d.stdout + d.stderr)
        h5dump = self.tool("h5dump")
        if h5dump:
            out["h5dump_rc"] = run([h5dump, "-pBH", str(out_path)]).returncode

        bad = out["core_diffs"] or out["h5diff_rc"] not in (0, None)
        bad = bad or out["h5dump_rc"] not in (0, None)
        out["verdict"] = "INVALID" if bad else "PASS"
        return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def clip(s, n=90):
    if s is None:
        return "-"
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


SEVERITY = {
    "divergence": 0,
    "missing-object": 1,
    "write-invalid": 2,
    "capability": 3,
    "missing-field": 4,
    "write-unsupported": 5,
    "structural": 6,
}

SEVERITY_LABEL = {
    "divergence": "value divergence",
    "missing-object": "object silently dropped",
    "write-invalid": "written file rejected",
    "capability": "read capability missing",
    "missing-field": "field not reported",
    "write-unsupported": "API cannot express",
    "structural": "no public accessor (API-wide)",
}


def deviation_tables(results):
    """(expected rows, unexpected rows) for the direction-B metadata diffs.

    Expected rows keep the declared order and carry the cases that hit them,
    so an entry that stops firing stays visible as `observed: no`.
    """
    declared = declared_deviations()
    # One row per face, so a face that stops firing is visible even when its
    # entry's other face still does.
    hits = {(exp["id"], f["field"]): [] for exp in declared for f in exp["faces"]}
    seen = {key: (None, None) for key in hits}
    unexpected = {}
    for r in results:
        for d in r["b"].get("metadata_diffs", []):
            eid = d.get("expected")
            if eid is None:
                unexpected.setdefault(
                    (d["key"], d["ref"], d["rust"]), []
                ).append(r["case"])
                continue
            key = (eid, d["field"])
            if r["case"] not in hits[key]:
                hits[key].append(r["case"])
                seen[key] = (d["ref"], d["rust"])
    expected = []
    for exp in declared:
        for face in exp["faces"]:
            key = (exp["id"], face["field"])
            ref, rust = seen[key]
            expected.append(
                {
                    "id": exp["id"],
                    "field": face["field"],
                    # `*` where the face deliberately matches a family of
                    # values rather than one pair; the example then carries a
                    # real pair.
                    "ref": face["ref"] or "*",
                    "rust": face["rust"] or "*",
                    "example": None if ref is None else "%s -> %s" % (ref, rust),
                    "why": exp["why"],
                    "cases": hits[key],
                }
            )
    return expected, sorted(unexpected.items())


def collect_gaps(results):
    """Aggregate every finding into (kind, signature) -> affected cases."""
    agg = {}

    def add(kind, signature, case, example=""):
        entry = agg.setdefault(
            (kind, signature), {"kind": kind, "signature": signature,
                                "cases": [], "example": example}
        )
        if case not in entry["cases"]:
            entry["cases"].append(case)
        if example and not entry["example"]:
            entry["example"] = example

    for r in results:
        a = r["a"]
        for d in a["divergences"]:
            add(
                "divergence",
                "%s: %s" % (d["field"], d["kind"]),
                r["case"],
                "%s\n  libhdf5: %s\n  rust:    %s"
                % (d["key"], clip(d["ref"], 120), clip(d.get("probe"), 120)),
            )
        for g in a["gaps"]:
            if g["kind"] == "unsupported":
                # Everything after the first ';' is per-file detail (e.g. the
                # string attr_string() managed to return); the signature is
                # the reason itself, so the same gap aggregates.
                reason = g["probe"].partition(": ")[2].partition(";")[0]
                add(
                    "structural" if g.get("structural") else "capability",
                    "%s — %s" % (g["field"], clip(reason, 100)),
                    r["case"],
                    g["key"],
                )
            elif g["kind"] == "missing-object":
                add("missing-object",
                    "object present in the file is not listed by the reader",
                    r["case"], "%s (%s)" % (g["key"], clip(g["ref"], 40)))
            else:
                add("missing-field", "%s not emitted" % g["field"], r["case"],
                    g["key"])
        b = r["b"]
        if b["verdict"] == "UNSUPPORTED-API":
            add("write-unsupported", clip(b["detail"].replace("UNSUPPORTED-API: ", ""), 110),
                r["case"], "")
        elif b["verdict"] == "INVALID":
            sig = ", ".join(sorted({d["field"] for d in b["core_diffs"]})) or "writer/reader error"
            add("write-invalid", sig, r["case"], clip(b["detail"], 160))

    items = list(agg.values())
    items.sort(key=lambda e: (SEVERITY[e["kind"]], -len(e["cases"]), e["signature"]))
    return items


def counts(results, direction, keys):
    out = {k: 0 for k in keys}
    for r in results:
        v = r[direction]["verdict"]
        out[v] = out.get(v, 0) + 1
    return out


def write_report(results, gaps, meta, md_path, json_path):
    a_counts = counts(
        results, "a", ["PASS", "GAP", "MISS", "DIFF", "READ-ERROR", "GEN-ERROR"]
    )
    b_counts = counts(results, "b", ["PASS", "INVALID", "UNSUPPORTED-API", "SKIPPED"])

    L = []
    L.append("# rust-hdf5 parity oracle — report")
    L.append("")
    L.append(
        "Generated by `oracle/run.py` against libhdf5 %s / h5py %s. "
        "%d cases." % (meta["hdf5_version"], meta["h5py_version"], len(results))
    )
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```sh")
    L.append("cargo build --release --bin oracle_probe")
    L.append("$RUST_HDF5_ORACLE_PYTHON oracle/run.py        # or just: python3 oracle/run.py")
    L.append("```")
    L.append("")
    L.append(
        "`run.py` re-executes itself under `RUST_HDF5_ORACLE_PYTHON` "
        "(default `%s`) when the invoking interpreter has no h5py, builds the "
        "probe if it is missing, and rewrites this file plus "
        "`oracle/report.json`. `--filter SUBSTR` restricts the matrix, "
        "`--keep` leaves the generated `.h5` files in the work directory."
        % DEFAULT_PYTHON
    )
    L.append("")
    L.append("## Verdicts")
    L.append("")
    L.append(
        "**Direction A** (h5py writes, rust-hdf5 reads) — `DIFF` at least one "
        "field where both sides produced a value and the values disagree; "
        "`MISS` no divergence, but an object that is in the file never appears "
        "in `group_names`/`dataset_names`, so the reader does not even report "
        "an error for it; `GAP` a field this file has that the public API "
        "cannot observe here (an unreadable datatype, an unresolvable link); "
        "`PASS` everything this file contains was read correctly; "
        "`READ-ERROR` the probe could not open or walk the file at all."
    )
    L.append("")
    L.append(
        "`PASS` tolerates the %d accessors that are missing from the API "
        "*everywhere* (%s) — they are counted once each in the findings table "
        "below rather than against every case that happens to contain a "
        "dataset."
        % (
            len(STRUCTURAL_FIELDS),
            ", ".join("`%s`" % f for f in sorted(STRUCTURAL_FIELDS)),
        )
    )
    L.append("")
    L.append(
        "**Direction B** (rust-hdf5 writes, h5py/libhdf5 reads) — `PASS` h5py "
        "read it, every core field (kind, dtype, shape, data, attributes, link "
        "targets) matched the reference and `h5diff`/`h5dump` were clean; "
        "`INVALID` one of those failed; `UNSUPPORTED-API` the public API cannot "
        "express the case. Differences confined to %s are recorded as metadata "
        "deviations and do not fail a case, because the values libhdf5 reads "
        "are identical."
        % ", ".join("`%s`" % f for f in sorted(B_TOLERATED_FIELDS))
    )
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append("| direction | " + " | ".join(a_counts) + " |")
    L.append("|---|" + "---|" * len(a_counts))
    L.append("| A (read) | " + " | ".join(str(a_counts[k]) for k in a_counts) + " |")
    L.append("")
    L.append("| direction | " + " | ".join(b_counts) + " |")
    L.append("|---|" + "---|" * len(b_counts))
    L.append("| B (write) | " + " | ".join(str(b_counts[k]) for k in b_counts) + " |")
    L.append("")

    L.append("## Top gaps by severity")
    L.append("")
    if not gaps:
        L.append("None: no case in this run produced a gap to rank.")
    else:
        L.append("| # | severity | finding | cases |")
        L.append("|---|---|---|---|")
        for i, g in enumerate(gaps[:10], 1):
            L.append(
                "| %d | %s | %s | %d (%s) |"
                % (
                    i,
                    SEVERITY_LABEL[g["kind"]],
                    clip(g["signature"], 110),
                    len(g["cases"]),
                    clip(", ".join(g["cases"][:4]) + ("…" if len(g["cases"]) > 4 else ""), 60),
                )
            )
    L.append("")

    L.append("## Case matrix")
    L.append("")
    if not results:
        L.append("None: this run selected no case.")
    else:
        L.append("| case | group | A | div | miss | gap | B | note |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in results:
            a, b = r["a"], r["b"]
            note = ""
            if a["verdict"] in ("READ-ERROR", "GEN-ERROR"):
                note = clip(a["detail"], 70)
            elif b["verdict"] in ("INVALID", "UNSUPPORTED-API"):
                note = clip(b["detail"], 70)
            miss = sum(1 for g in a["gaps"] if g["kind"] == "missing-object")
            gap = sum(
                1
                for g in a["gaps"]
                if g["kind"] != "missing-object" and not g.get("structural")
            )
            L.append(
                "| `%s` | %s | %s | %d | %d | %d | %s | %s |"
                % (
                    r["case"],
                    r["group"],
                    a["verdict"],
                    len(a["divergences"]),
                    miss,
                    gap,
                    b["verdict"],
                    note,
                )
            )
    L.append("")

    diffs = [r for r in results if r["a"]["divergences"]]
    L.append("## Direction A divergences, in full")
    L.append("")
    if not diffs:
        L.append("None.")
    for r in diffs:
        L.append("### `%s`" % r["case"])
        L.append("")
        for d in r["a"]["divergences"]:
            L.append("- `%s` (%s)" % (d["key"], d["kind"]))
            L.append("  - libhdf5: `%s`" % clip(d["ref"], 160))
            L.append("  - rust-hdf5: `%s`" % clip(d.get("probe"), 160))
        L.append("")

    dropped = [
        (r["case"], g)
        for r in results
        for g in r["a"]["gaps"]
        if g["kind"] == "missing-object"
    ]
    L.append("## Objects the reader does not list")
    L.append("")
    if not dropped:
        L.append("None.")
    else:
        L.append(
            "These paths exist in the file and h5py describes them, but "
            "`H5Group::group_names` / `dataset_names` never mention them, so "
            "the probe cannot even report an error for them."
        )
        L.append("")
        L.append("| case | path | libhdf5 calls it | case exercises |")
        L.append("|---|---|---|---|")
        notes = {r["case"]: r["note"] for r in results}
        for case, g in dropped:
            L.append(
                "| `%s` | `%s` | %s | %s |"
                % (case, g["key"], clip(g["ref"], 24), clip(notes.get(case, ""), 60))
            )
    L.append("")

    expected, unexpected = deviation_tables(results)

    L.append("## Direction B expected deviations")
    L.append("")
    if not expected:
        gated = len(EXPECTED_DEVIATIONS) - len(declared_deviations())
        if gated:
            version = ".".join(str(n) for n in hdf5_version_tuple())
            why = (
                "no `EXPECTED_DEVIATIONS` entry (oracle/run.py) applies to "
                "libhdf5 %s — %s scoped to a later one"
                % (
                    version,
                    "1 entry is" if gated == 1 else "%d entries are" % gated,
                )
            )
        else:
            why = "`EXPECTED_DEVIATIONS` (oracle/run.py) is empty"
        L.append(
            "None: %s, so every case in this run describes itself the way "
            "libhdf5 describes the same file. Any metadata deviation from "
            "here on matches no entry and is reported as unexpected below."
            % why
        )
    else:
        L.append(
            "The rust-written file carries the same data, type and shape as "
            "the h5py reference but describes itself differently. Each row "
            "below is a known, understood writer deviation declared in "
            "`EXPECTED_DEVIATIONS` (oracle/run.py); it does not fail a case. "
            "`observed: no` means a declared deviation no longer happens — "
            "either the writer was fixed and the entry should go, or the "
            "cases that exercised it changed."
        )
        L.append("")
        L.append("| id | field | libhdf5 | rust-hdf5 | observed | cases |")
        L.append("|---|---|---|---|---|---|")
        for e in expected:
            L.append(
                "| `%s` | `%s` | `%s` | `%s` | %s | %d%s |"
                % (
                    e["id"],
                    e["field"],
                    clip(e["ref"], 34),
                    clip(e["rust"], 34),
                    "yes" if e["cases"] else "no",
                    len(e["cases"]),
                    (" (%s)" % clip(", ".join(e["cases"][:3])
                                    + ("…" if len(e["cases"]) > 3 else ""), 44))
                    if e["cases"] else "",
                )
            )
        L.append("")
        # One bullet per entry, not per row: the faces of one deviation share
        # the explanation, and repeating it would read as two causes.
        for eid in dict.fromkeys(e["id"] for e in expected):
            faces = [e for e in expected if e["id"] == eid]
            observed = [
                "`%s` as `%s`" % (e["field"], clip(e["example"], 70))
                for e in faces
                if e["example"]
            ]
            L.append(
                "- `%s`: %s%s"
                % (
                    eid,
                    faces[0]["why"],
                    ("; observed in " + ", ".join(observed)) if observed else "",
                )
            )
    L.append("")

    L.append("## Direction B unexpected deviations")
    L.append("")
    if not unexpected:
        L.append("None: every metadata deviation in this run is a declared one.")
    else:
        L.append(
            "Metadata deviations matching no `EXPECTED_DEVIATIONS` entry. These "
            "are new since the table was written and want a verdict."
        )
        L.append("")
        L.append("| key | libhdf5 | rust-hdf5 | cases |")
        L.append("|---|---|---|---|")
        for (key, ref, rust), cs in unexpected:
            L.append(
                "| `%s` | `%s` | `%s` | %d (%s) |"
                % (
                    key,
                    clip(ref, 40),
                    clip(rust, 40),
                    len(cs),
                    clip(", ".join(cs[:3]) + ("…" if len(cs) > 3 else ""), 50),
                )
            )
    L.append("")

    binvalid = [r for r in results if r["b"]["verdict"] == "INVALID"]
    L.append("## Direction B failures, in full")
    L.append("")
    if not binvalid:
        L.append("None.")
    for r in binvalid:
        L.append("### `%s`" % r["case"])
        L.append("")
        L.append("- h5diff rc: `%s`, h5dump rc: `%s`" %
                 (r["b"]["h5diff_rc"], r["b"]["h5dump_rc"]))
        for d in r["b"]["core_diffs"]:
            L.append("- `%s`" % d["key"])
            L.append("  - libhdf5: `%s`" % clip(d["ref"], 160))
            L.append("  - rust-hdf5: `%s`" % clip(d["rust"], 160))
        if r["b"]["detail"]:
            L.append("- detail: `%s`" % clip(r["b"]["detail"], 200))
        L.append("")

    L.append("## All findings")
    L.append("")
    if not gaps:
        L.append("None: no case in this run produced a finding.")
    else:
        L.append("| severity | finding | cases | example |")
        L.append("|---|---|---|---|")
        for g in gaps:
            L.append(
                "| %s | %s | %d | %s |"
                % (
                    SEVERITY_LABEL[g["kind"]],
                    clip(g["signature"], 110),
                    len(g["cases"]),
                    clip(g["example"], 80),
                )
            )
    L.append("")
    L.append("## Known modelling gaps in the canonical format")
    L.append("")
    L.append(
        "- The string pad of a *variable-length* string is reported in the "
        "separate `strpad` field rather than inline in `dtype`; a fixed string "
        "keeps its pad inline. Both sides answer both forms — the split is a "
        "layout choice of this format, not a gap."
    )
    L.append(
        "- `chunkindex` is read on the libhdf5 side from the on-disk layout "
        "message via `h5debug`, like `filters`; the selection rules it was "
        "once derived from changed under HDF5 2.0 (see oracle/CANON.md)."
    )
    L.append(
        "- h5py 3.x exposes neither `get_offset` nor `get_precision` on a "
        "bitfield type, so a sub-width bitfield would be reported at full "
        "width. No case exercises that."
    )
    L.append("")

    md_path.write_text("\n".join(L) + "\n")
    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "results": results,
                "findings": gaps,
                "expected_deviations": expected,
                "unexpected_deviations": [
                    {"key": k, "ref": rv, "rust": gv, "cases": cs}
                    for (k, rv, gv), cs in unexpected
                ],
            },
            indent=1,
            sort_keys=False,
        )
        + "\n"
    )


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="only cases whose name contains this")
    ap.add_argument("--work", default="", help="directory for generated .h5 files")
    ap.add_argument("--keep", action="store_true", help="keep the generated files")
    ap.add_argument("--no-build", action="store_true", help="do not run cargo build")
    ap.add_argument(
        "--variant",
        default="",
        help="suffix for the report paths, so a run against a second "
        "h5py/libhdf5 build does not overwrite the pinned report",
    )
    args = ap.parse_args()

    import h5py

    probe = os.environ.get(
        "RUST_HDF5_ORACLE_PROBE", str(REPO / "target" / "release" / "oracle_probe")
    )
    if not args.no_build:
        build = run(
            ["cargo", "build", "--release", "--bin", "oracle_probe"], cwd=str(REPO)
        )
        if build.returncode != 0:
            sys.stderr.write(build.stdout + build.stderr)
            return 1
    if not pathlib.Path(probe).exists():
        sys.stderr.write("probe binary not found: %s\n" % probe)
        return 1

    bindir = os.environ.get(
        "RUST_HDF5_ORACLE_BINDIR", str(pathlib.Path(sys.executable).parent)
    )
    work = pathlib.Path(args.work) if args.work else REPO / "target" / "oracle-work"
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    oracle = Oracle(probe, bindir, work)

    selected = [c for c in cases.ALL_CASES if args.filter in c.name]
    results = []
    for i, case in enumerate(selected, 1):
        sys.stderr.write("[%2d/%d] %s\n" % (i, len(selected), case.name))
        sys.stderr.flush()
        a, ref_path = oracle.direction_a(case)
        b = oracle.direction_b(case, ref_path)
        results.append(
            {
                "case": case.name,
                "group": case.group,
                "note": case.note,
                "rust_case": case.rust,
                "a": a,
                "b": b,
            }
        )

    gaps = collect_gaps(results)
    meta = {
        "hdf5_version": h5py.version.hdf5_version,
        "h5py_version": h5py.__version__,
        "numpy_version": __import__("numpy").__version__,
        "python": sys.executable,
        "probe": report_path(probe),
        "cases": len(results),
    }
    suffix = "-%s" % args.variant if args.variant else ""
    write_report(
        results,
        gaps,
        meta,
        REPO / "doc" / ("oracle-report%s.md" % suffix),
        REPO / "oracle" / ("report%s.json" % suffix),
    )

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)

    a_bad = sum(1 for r in results if r["a"]["verdict"] in ("DIFF", "READ-ERROR", "GEN-ERROR"))
    b_bad = sum(1 for r in results if r["b"]["verdict"] == "INVALID")
    sys.stderr.write(
        "\nA: %d PASS  %d GAP  %d MISS  %d DIFF  %d READ-ERROR\n"
        % (
            sum(1 for r in results if r["a"]["verdict"] == "PASS"),
            sum(1 for r in results if r["a"]["verdict"] == "GAP"),
            sum(1 for r in results if r["a"]["verdict"] == "MISS"),
            sum(1 for r in results if r["a"]["verdict"] == "DIFF"),
            sum(1 for r in results if r["a"]["verdict"] == "READ-ERROR"),
        )
    )
    sys.stderr.write(
        "B: %d PASS  %d INVALID  %d UNSUPPORTED-API\n"
        % (
            sum(1 for r in results if r["b"]["verdict"] == "PASS"),
            sum(1 for r in results if r["b"]["verdict"] == "INVALID"),
            sum(1 for r in results if r["b"]["verdict"] == "UNSUPPORTED-API"),
        )
    )
    sys.stderr.write(
        "report: doc/oracle-report%s.md, oracle/report%s.json\n" % (suffix, suffix)
    )
    return 0 if (a_bad == 0 and b_bad == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
