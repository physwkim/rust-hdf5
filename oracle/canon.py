#!/usr/bin/env python3
"""Canonical h5py-side dump of an HDF5 file — the reference half of the oracle.

Emits the `!canon 7` format described in oracle/CANON.md. The rust-hdf5 side
(`src/bin/oracle_probe.rs`, `dump` subcommand) emits the same format from the
same file, so the two are comparable line by line and field by field.

Usage:  canon.py <file.h5>

Only the standard library, numpy and h5py are used.
"""

import hashlib
import os
import pathlib
import re
import subprocess
import sys
import traceback

import numpy as np

import hdf5env  # noqa: F401  (must precede h5py; see the module docstring)

import h5py
from h5py import h5d, h5o, h5p, h5s, h5t

CANON_VERSION = "8"
RAW_LIMIT = 1024
MAX_DEPTH = 32

# Object header message type of the **Symbol Table** message (`H5Opkg.h`),
# used as a bit position in `H5O_hdr_info_t.mesg.present`.
H5O_STAB_ID = 0x11

# --------------------------------------------------------------------------
# canonical scalar / string encoding
# --------------------------------------------------------------------------


def esc(s):
    """Canonical quoted-string encoding, identical on the rust side."""
    if isinstance(s, (bytes, bytearray, np.bytes_)):
        try:
            s = bytes(s).decode("utf-8")
        except UnicodeDecodeError:
            return "b0x" + bytes(s).hex()
    if isinstance(s, np.str_):
        s = str(s)
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        elif o <= 0xFF:
            out.append("\\x%02x" % o)
        elif o <= 0xFFFF:
            out.append("\\u%04x" % o)
        else:
            out.append("\\U%08x" % o)
    out.append('"')
    return "".join(out)


def float_bits(value, size):
    """IEEE bits of `value` as big-endian hex — never a formatted decimal."""
    arr = np.asarray(value).astype(">f%d" % size, copy=False)
    return "0x" + arr.tobytes().hex()


def dims_str(dims):
    return "[" + ",".join(str(d) for d in dims) + "]"


def maxdims_str(dims):
    return "[" + ",".join("U" if d is None else str(d) for d in dims) + "]"


# --------------------------------------------------------------------------
# canonical datatype strings
# --------------------------------------------------------------------------

_ORDER = {
    h5t.ORDER_LE: "le",
    h5t.ORDER_BE: "be",
    h5t.ORDER_VAX: "vax",
    h5t.ORDER_NONE: "none",
}

_CSET = {h5t.CSET_ASCII: "ascii", h5t.CSET_UTF8: "utf8"}

_STRPAD = {
    h5t.STR_NULLTERM: "null",
    h5t.STR_NULLPAD: "nullpad",
    h5t.STR_SPACEPAD: "spacepad",
}

# (sign_pos, exp_pos, exp_size, mant_pos, mant_size, bias) for IEEE 754.
_FLOAT_STD = {
    2: (15, 10, 5, 0, 10, 15),
    4: (31, 23, 8, 0, 23, 127),
    8: (63, 52, 11, 0, 52, 1023),
}


def _order(tid):
    return _ORDER.get(tid.get_order(), "ord%d" % tid.get_order())


def canon_dtype(tid):
    cls = tid.get_class()
    size = tid.get_size()

    if cls == h5t.INTEGER:
        sgn = "i" if tid.get_sign() == h5t.SGN_2 else "u"
        s = "%s%d%s" % (sgn, size * 8, _order(tid))
        off, prec = tid.get_offset(), tid.get_precision()
        if off != 0 or prec != size * 8:
            s += "+off%dp%d" % (off, prec)
        return s

    if cls == h5t.FLOAT:
        s = "f%d%s" % (size * 8, _order(tid))
        spos, epos, esize, mpos, msize = tid.get_fields()
        bias = tid.get_ebias()
        off, prec = tid.get_offset(), tid.get_precision()
        std = _FLOAT_STD.get(size)
        if std != (spos, epos, esize, mpos, msize, bias) or off != 0 or prec != size * 8:
            s += "+s%de%d,%dm%d,%db%doff%dp%d" % (
                spos,
                epos,
                esize,
                mpos,
                msize,
                bias,
                off,
                prec,
            )
        return s

    if cls == h5t.STRING:
        cset = _CSET.get(tid.get_cset(), "cset%d" % tid.get_cset())
        if tid.is_variable_str():
            # The pad of a variable-length string travels in the separate
            # `strpad` field, not inline here; see oracle/CANON.md.
            return "vstr,cset=%s" % cset
        pad = _STRPAD.get(tid.get_strpad(), "pad%d" % tid.get_strpad())
        return "str[%d],pad=%s,cset=%s" % (size, pad, cset)

    if cls == h5t.BITFIELD:
        s = "bits[%d]%s" % (size, _order(tid))
        # h5py 3.x's TypeBitfieldID exposes neither get_offset nor
        # get_precision, so a bitfield whose precision is narrower than its
        # size is reported as full width. No case exercises that today.
        off = tid.get_offset() if hasattr(tid, "get_offset") else 0
        prec = tid.get_precision() if hasattr(tid, "get_precision") else size * 8
        if off != 0 or prec != size * 8:
            s += "+off%dp%d" % (off, prec)
        return s

    if cls == h5t.OPAQUE:
        return "opaque[%d],tag=%s" % (size, esc(tid.get_tag()))

    if cls == h5t.COMPOUND:
        parts = []
        for i in range(tid.get_nmembers()):
            parts.append(
                "%s@%d:%s"
                % (
                    _name(tid.get_member_name(i)),
                    tid.get_member_offset(i),
                    canon_dtype(tid.get_member_type(i)),
                )
            )
        return "compound[%d]{%s}" % (size, ";".join(parts))

    if cls == h5t.ENUM:
        base = tid.get_super()
        parts = []
        for i in range(tid.get_nmembers()):
            parts.append(
                "%s=%d" % (_name(tid.get_member_name(i)), tid.get_member_value(i))
            )
        return "enum(%s){%s}" % (canon_dtype(base), ";".join(parts))

    if cls == h5t.VLEN:
        return "vlen(%s)" % canon_dtype(tid.get_super())

    if cls == h5t.ARRAY:
        return "array%s(%s)" % (
            dims_str(tid.get_array_dims()),
            canon_dtype(tid.get_super()),
        )

    if cls == h5t.REFERENCE:
        return "objref" if size == 8 else "regref"

    if cls == h5t.TIME:
        return "time[%d]%s" % (size, _order(tid))

    return "class%d[%d]" % (cls, size)


def _name(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


# --------------------------------------------------------------------------
# value rendering
# --------------------------------------------------------------------------


def has_heap_type(tid):
    """True when the datatype has no flat on-disk byte image we can compare."""
    cls = tid.get_class()
    if cls == h5t.VLEN:
        return True
    if cls == h5t.REFERENCE:
        return True
    if cls == h5t.STRING and tid.is_variable_str():
        return True
    if cls == h5t.ARRAY:
        return has_heap_type(tid.get_super())
    if cls == h5t.COMPOUND:
        return any(
            has_heap_type(tid.get_member_type(i)) for i in range(tid.get_nmembers())
        )
    return False


def vlen_strpads(tid, where=""):
    """`where=pad` for every variable-length string in the type tree.

    A fixed string carries its pad inline in the dtype, where both sides
    agree. A variable-length one cannot, so it is reported here. `where` is
    the position in the type tree: `.` is the type itself, `.m` a compound
    member, `[]` an array element, `()` a vlen element.
    """
    cls = tid.get_class()
    if cls == h5t.STRING and tid.is_variable_str():
        pad = _STRPAD.get(tid.get_strpad(), "pad%d" % tid.get_strpad())
        return ["%s=%s" % (where or ".", pad)]
    if cls == h5t.ARRAY:
        return vlen_strpads(tid.get_super(), where + "[]")
    if cls == h5t.VLEN:
        return vlen_strpads(tid.get_super(), where + "()")
    if cls == h5t.COMPOUND:
        out = []
        for i in range(tid.get_nmembers()):
            name = tid.get_member_name(i).decode("utf-8", "replace")
            out.extend(vlen_strpads(tid.get_member_type(i), where + "." + name))
        return out
    return []


def strpad_str(tid):
    pads = vlen_strpads(tid)
    return ";".join(pads) if pads else "-"


def render_elem(value, tid):
    """Canonical rendering of one element of type `tid`."""
    cls = tid.get_class()

    if cls == h5t.FLOAT:
        return float_bits(value, tid.get_size())

    if cls in (h5t.INTEGER, h5t.ENUM, h5t.BITFIELD, h5t.TIME):
        return str(int(value))

    if cls == h5t.STRING:
        return esc(value)

    if cls == h5t.OPAQUE:
        return "0x" + np.asarray(value).tobytes().hex()

    if cls == h5t.REFERENCE:
        return render_ref(value)

    if cls == h5t.VLEN:
        base = tid.get_super()
        arr = np.asarray(value)
        return "[" + ",".join(render_elem(v, base) for v in arr.ravel()) + "]"

    if cls == h5t.ARRAY:
        base = tid.get_super()
        arr = np.asarray(value)
        return "[" + ",".join(render_elem(v, base) for v in arr.ravel()) + "]"

    if cls == h5t.COMPOUND:
        parts = []
        for i in range(tid.get_nmembers()):
            name = _name(tid.get_member_name(i))
            parts.append(render_elem(value[name], tid.get_member_type(i)))
        return "{" + ",".join(parts) + "}"

    return "0x" + np.asarray(value).tobytes().hex()


_REF_FILE = None


def render_ref(ref):
    """Object / region references render as their resolved target path.

    A raw address would not be comparable between two writers, so the
    canonical form is the target path — plus the selection bounds for a
    region reference, which is what distinguishes two region references into
    the same dataset.
    """
    if ref is None:
        return "objref:null"
    try:
        if isinstance(ref, h5py.RegionReference):
            target = _REF_FILE[ref]
            space = h5py.h5r.get_region(ref, target.id)
            lo, hi = space.get_select_bounds()
            return "regref:%s:%s-%s" % (target.name, dims_str(lo), dims_str(hi))
        target = _REF_FILE[ref]
        return "objref:%s" % target.name
    except Exception as exc:  # pragma: no cover - defensive
        return "ERROR(ref): %s" % oneline(exc)


def encode_payload(text_vals=None, raw=None):
    """Apply the size policy and return the canonical `data`/`value` string."""
    if raw is not None:
        if len(raw) <= RAW_LIMIT:
            return "raw:" + raw.hex()
        return "sha256:" + hashlib.sha256(raw).hexdigest()
    body = "vals:[" + ",".join(text_vals) + "]"
    if len(body) <= RAW_LIMIT:
        return body
    return "valsha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def raw_image(read_into, tid, sid):
    """The on-disk byte image, read with the file type as the memory type.

    Going through an untyped `V<size>` buffer with `mtype` pinned to the file
    type means libhdf5 performs no conversion at all, so the bytes are exactly
    what is stored. That matters for compounds with gaps, for opaque data
    (which has no numpy conversion path), and for big-endian types, all of
    which a plain `dset[()]` either reorders or refuses.
    """
    size = tid.get_size()
    dims = tuple(sid.get_simple_extent_dims())
    arr = np.empty(dims, dtype="V%d" % size)
    if arr.size:
        read_into(arr)
    return arr.tobytes()


def flatten_heap(data):
    if isinstance(data, np.ndarray):
        return [data[()]] if data.ndim == 0 else list(data.ravel())
    return [data]


def heap_payload(data, tid):
    return encode_payload(
        text_vals=[render_elem(v, tid) for v in flatten_heap(data)]
    )


def dataset_payload(dset):
    dsid = dset.id
    tid, sid = dsid.get_type(), dsid.get_space()
    if sid.get_simple_extent_type() == h5s.NULL:
        return "empty"
    if has_heap_type(tid):
        return heap_payload(dset[()], tid)
    return encode_payload(
        raw=raw_image(
            lambda arr: dsid.read(h5s.ALL, h5s.ALL, arr, mtype=tid), tid, sid
        )
    )


def resolve_extlink(grp, name):
    """What crossing an external link lands on, in one line.

    `target` above is only the link's stored value, so a reader that never
    opens the other file dumps exactly the same text as one that does. This
    follows the link the way `H5Dopen` on a path through it does, and reports
    the shape and payload digest of what it finds — the same functions the
    dataset dump uses, so resolving to the wrong object or reading the wrong
    bytes both show up here.
    """
    try:
        obj = grp[name]
    except Exception:
        # Target file missing, target object missing, or the file is not
        # readable: one answer, because a reader cannot tell them apart
        # without reporting on the other file's behalf.
        return "dangling"
    if isinstance(obj, h5py.Dataset):
        return "dataset %s %s" % (dims_str(obj.shape), dataset_payload(obj))
    if isinstance(obj, h5py.Group):
        return "group"
    if isinstance(obj, h5py.Datatype):
        return "committed-datatype"
    return "unknown"


def attr_payload(obj, name, aid, tid, sid):
    if sid.get_simple_extent_type() == h5s.NULL:
        return "empty"
    if has_heap_type(tid):
        return heap_payload(obj.attrs[name], tid)
    return encode_payload(
        raw=raw_image(lambda arr: aid.read(arr, mtype=tid), tid, sid)
    )


# --------------------------------------------------------------------------
# chunk index derivation
# --------------------------------------------------------------------------


_H5DEBUG_LAYOUT_MSG_RE = re.compile(
    r"`layout'.*?(?=\nMessage \d+\.\.\.|\Z)", re.DOTALL
)
_H5DEBUG_INDEX_TYPE_RE = re.compile(r"Index Type:\s*(.+)")

# H5O__layout_debug's names for H5D_chunk_index_t, printed for every chunked
# layout message regardless of its version (a v3 message decodes as a v1
# B-tree index).
_INDEX_TYPES = {
    "v1 B-tree": "btree1",
    "Implicit": "implicit",
    "Single Chunk": "single",
    "Fixed Array": "farray",
    "Extensible Array": "earray",
    "v2 B-tree": "btree2",
}


def chunk_index_str(dset):
    """The chunk index stored in the on-disk layout message, via `h5debug`.

    This used to be derived from the superblock version and the dataspace,
    mirroring H5D__layout_set_version: through 1.14 `H5F__super_read` raised
    the file's low bound for every open (superblock v3 implies the v1.10
    indexes). HDF5 2.0 raises it only for SWMR-write opens, so one file can
    hold a Fixed Array dataset from its libver=latest session next to a v1
    B-tree dataset appended through a default reopen — no static property of
    the dataset decides it. Measured, not derived, same route as `filters`.
    """
    addr = h5o.get_info(dset.id).addr
    proc = subprocess.run(
        [_h5debug_bin(), dset.file.filename, str(addr)],
        capture_output=True,
        text=True,
        check=True,
    )
    msg = _H5DEBUG_LAYOUT_MSG_RE.search(proc.stdout)
    if not msg:
        raise RuntimeError("no layout message in h5debug output")
    name = _H5DEBUG_INDEX_TYPE_RE.search(msg.group(0)).group(1).strip()
    return _INDEX_TYPES.get(name, name)


_FILTER_NAMES = {
    1: "deflate",
    2: "shuffle",
    3: "fletcher32",
    4: "szip",
    5: "nbit",
    6: "scaleoffset",
    307: "bzip2",
    32000: "lzf",
    32001: "blosc",
    32004: "lz4",
    32008: "bshuf",
    32015: "zstd",
}


# `dcpl.get_filter()` (H5Pget_filter2) does not always describe the on-disk
# filter-pipeline message: opening a scale-offset-filtered dataset
# re-invokes `H5Z__set_local_scaleoffset`, which reconstructs the pipeline's
# cd_values in memory — and for the reserved/unused tail slots past the
# packed fill-value bytes, that reconstruction leaves them holding whatever
# was there before rather than the zeros actually stored on disk. Verified
# on a real file two ways independent of any Python binding: a byte-exact
# manual parse of the filter-pipeline message (its declared size fully and
# exactly accounts for 20 zero-filled cd_values, no leftover bytes), and
# `h5debug <file> <header_addr>` — which calls libhdf5's own
# `H5O_pline_debug` directly — reporting `CD value 16` through `19` as `0`.
# `dcpl.get_filter()` instead reports `(1818321779, 1717989221, 7628147, 0)`
# for that same slot range, which decodes as the ASCII bytes
# `"scaleoffset\0"` — the filter's own name string bleeding through
# uncleared memory in the reconstruction. Full writeup: oracle/CANON.md,
# under `filters`.
#
# So `filters` is always measured from the on-disk message via `h5debug`,
# for every case, filtered or not — one route, not a special case carved
# out for scaleoffset.

_H5DEBUG_FILTER_MSG_RE = re.compile(
    r"`filter pipeline'.*?(?=\nMessage \d+\.\.\.|\Z)", re.DOTALL
)
_H5DEBUG_FILTER_BLOCK_RE = re.compile(
    r"Filter at position \d+.*?(?=Filter at position \d+|\Z)", re.DOTALL
)
_H5DEBUG_ID_RE = re.compile(r"Filter identification:\s*0x([0-9a-fA-F]+)")
_H5DEBUG_FLAGS_RE = re.compile(r"Flags:\s*0x([0-9a-fA-F]+)")
_H5DEBUG_CD_RE = re.compile(r"CD value \d+\s+(-?\d+)")


def _tool_bin(name):
    """Locate an HDF5 command-line tool the way `run.py` locates
    `h5dump`/`h5diff`: alongside this interpreter unless
    `RUST_HDF5_ORACLE_BINDIR` overrides it."""
    bindir = os.environ.get(
        "RUST_HDF5_ORACLE_BINDIR", str(pathlib.Path(sys.executable).parent)
    )
    return str(pathlib.Path(bindir) / name)


def _h5debug_bin():
    return _tool_bin("h5debug")


def filters_str(dset):
    """The `filters` field, read from the on-disk filter-pipeline message
    via `h5debug` rather than `dcpl.get_filter()` — see the note above.

    Uses `dset.file.filename`, not the top-level file path: `dset` may be
    reached through an external link, in which case its object header lives
    in a different file than the one this dump started from.
    """
    addr = h5o.get_info(dset.id).addr
    proc = subprocess.run(
        [_h5debug_bin(), dset.file.filename, str(addr)],
        capture_output=True,
        text=True,
        check=True,
    )
    msg = _H5DEBUG_FILTER_MSG_RE.search(proc.stdout)
    if not msg:
        return "[]"
    parts = []
    for block in _H5DEBUG_FILTER_BLOCK_RE.findall(msg.group(0)):
        fid = int(_H5DEBUG_ID_RE.search(block).group(1), 16)
        flags = int(_H5DEBUG_FLAGS_RE.search(block).group(1), 16)
        cd = _H5DEBUG_CD_RE.findall(block)
        parts.append("%s(%s)@%d" % (_FILTER_NAMES.get(fid, str(fid)), "|".join(cd), flags))
    return "[" + ",".join(parts) + "]"


# `hdr.mesg.shared` from `H5Oget_info` would answer most of this, but only as
# a bitmask of classes stored as a *pointer*: it cannot see the first copy of a
# share-in-object-header class, which `H5SM__write_mesg` leaves literal in the
# header that offered it under `H5O_MSG_FLAG_SHAREABLE` rather than
# `H5O_MSG_FLAG_SHARED` (H5SM.c:1112, H5SM.c:1400-1417) — and that flag is the
# whole difference between "the body is the message" and "the body is a
# reference". `h5debug` prints the flags byte itself (`H5O__debug_real`,
# H5Odbg.c:409-455) and, beneath a shared message, which storage the pointer
# names (`H5O__shared_debug`, H5Oshared.c:682-706), so the field is read from
# there.
#
# Deliberately *not* `hdr.mesg.present`: that mask counts the null and
# continuation messages too (H5Oint.c:2067-2069), and where those fall is the
# writer's allocation strategy rather than anything a reader sees. libhdf5
# creates a dataset header at a 256-byte size hint (H5D_MINHDR_SIZE,
# H5Dpkg.h:42; H5Dint.c:898, :993) whose unused tail stays one null message
# (H5Oint.c:516-522), and a committed datatype at the exact size of its
# datatype message (H5Tcommit.c:468, :475), so the reference count message
# added afterwards forces a second chunk and a continuation. A null message has
# no decode, encode, size or copy method at all (H5Onull.c:28-49) — it is free
# space wearing a message header — so a field that measured the present mask
# would be comparing padding.

_H5DEBUG_MSG_SPLIT_RE = re.compile(r"^Message \d+\.\.\.$", re.M)
_H5DEBUG_MSG_NAME_RE = re.compile(
    r"Message ID \(sequence number\):\s+0x[0-9a-fA-F]+ `([^']+)'"
)
_H5DEBUG_MSG_FLAGS_RE = re.compile(r"Message flags:\s+<([^>]*)>")
_H5DEBUG_SHARED_TYPE_RE = re.compile(r"Shared Message type:\s+(.+?)\s*$", re.M)

# `H5O_msg_class_t.name` for the classes a shared-message index can cover,
# mapped to the names `oracle_probe`'s `message_class_name` uses.
_SHARED_CLASS_NAMES = {
    "filter pipeline": "filter_pipeline",
}

# What `H5O__shared_debug` prints for each `H5O_shared_t.type`.
_SHARED_LOCATIONS = {
    "SOHM": "sohm",
    "Obj Hdr": "committed",
    "Here": "here",
    "Unshared": "unshared",
}


def shared_str(obj_id, filename):
    """The `shared` field: how the object header stores each message it does
    not hold privately, as sorted `class:storage` pairs."""
    addr = h5o.get_info(obj_id).addr
    proc = subprocess.run(
        [_h5debug_bin(), filename, str(addr)],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = []
    for block in _H5DEBUG_MSG_SPLIT_RE.split(proc.stdout)[1:]:
        name = _H5DEBUG_MSG_NAME_RE.search(block)
        flags = _H5DEBUG_MSG_FLAGS_RE.search(block)
        if not name or not flags:
            continue
        tokens = [t.strip() for t in flags.group(1).split(",")]
        if "S" in tokens:
            where = _H5DEBUG_SHARED_TYPE_RE.search(block)
            where = _SHARED_LOCATIONS.get(where.group(1), where.group(1)) if where else "?"
        elif "SA" in tokens:
            where = "shareable"
        else:
            continue
        cls = name.group(1)
        parts.append("%s:%s" % (_SHARED_CLASS_NAMES.get(cls, cls), where))
    return "[" + ",".join(sorted(parts)) + "]"


_LAYOUTS = {
    h5d.COMPACT: "compact",
    h5d.CONTIGUOUS: "contiguous",
    h5d.CHUNKED: "chunked",
}
if hasattr(h5d, "VIRTUAL"):
    _LAYOUTS[h5d.VIRTUAL] = "virtual"


def external_str(dcpl):
    """External file storage segments, `-` when the data lives in the file."""
    count = dcpl.get_external_count()
    if not count:
        return "-"
    parts = []
    for i in range(count):
        name, offset, size = dcpl.get_external(i)
        parts.append("%s@%d+%d" % (esc(name), offset, size))
    return "[" + ",".join(parts) + "]"


_H5DUMP_EXTENT_RE = re.compile(r"DATASPACE\s+SIMPLE\s*\{\s*\(([^)]*)\)")
_H5DUMP_MAPPING_RE = re.compile(r"MAPPING\s+\d+\s*")
_H5DUMP_TUPLE_RE = re.compile(r"(START|STRIDE|COUNT|BLOCK)\s*\(([^)]*)\)")


def _brace_block(text, pos):
    """Body of the block whose opening brace is the first one at or after
    `pos`, and the offset just past its close.

    Brace-matched rather than pattern-matched because `h5dump` nests these:
    a `MAPPING` holds a `VIRTUAL` and a `SOURCE`, each holding a `SELECTION`.
    """
    open_at = text.index("{", pos)
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i], i + 1
    raise ValueError("unbalanced braces in h5dump output")


def _h5dump_mappings(dump):
    """The body of every `MAPPING n` block, in order."""
    out = []
    pos = 0
    while True:
        m = _H5DUMP_MAPPING_RE.search(dump, pos)
        if not m:
            return out
        body, pos = _brace_block(dump, m.end())
        out.append(body)


def _h5dump_side(mapping, name):
    """The `VIRTUAL` or `SOURCE` half of one mapping."""
    m = re.search(r"\b%s\b\s*" % name, mapping)
    if not m:
        return ""
    return _brace_block(mapping, m.end())[0]


def _h5dump_layout(dset):
    """The `STORAGE_LAYOUT` block `h5dump -p -H` prints for one dataset, and
    the dataset's own extent, as text.

    The virtual mapping is read from here rather than from the creation
    property list because libhdf5 2.0 resolves the layout before handing the
    DCPL back — `H5D__flush_layout_to_dcpl` (H5Dint.c:4086), called from
    `H5D__get_create_plist` (H5Dint.c:3643) — so `H5Pget_virtual_srcspace`
    reports the extent the mapping currently resolves to rather than the one
    the file stores. `h5dump` prints the stored message, and prints it
    identically under 1.14.6 and 2.1.
    """
    proc = subprocess.run(
        [_tool_bin("h5dump"), "-p", "-H", "-d", dset.name, dset.file.filename],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _selection_bounds(text, extent):
    """The bounds of one `SELECTION` block as `lo-hi`, or `?` when the file
    does not pin them down.

    `ALL` needs an extent to have bounds at all: the virtual side has one (the
    dataset's own), the source side does not — the mapping stores a selection
    and no source extent — so a source `ALL` is `?`, which is what
    `H5Sget_select_bounds` used to raise on. A hyperslab that runs to
    `H5S_UNLIMITED` has no upper bound until it is resolved against the files
    that exist, so it is `?` as well.
    """
    if "SELECTION ALL" in text:
        if extent is None:
            return "?"
        return "%s-%s" % (dims_str([0] * len(extent)), dims_str([d - 1 for d in extent]))
    if "REGULAR_HYPERSLAB" not in text:
        return "?"
    got = {}
    for name, body in _H5DUMP_TUPLE_RE.findall(text):
        got[name] = [v.strip() for v in body.split(",")]
    if set(got) != {"START", "STRIDE", "COUNT", "BLOCK"}:
        return "?"
    if any("UNLIMITED" in v for vs in got.values() for v in vs):
        return "?"
    start, stride, count, block = (
        [int(v) for v in got[k]] for k in ("START", "STRIDE", "COUNT", "BLOCK")
    )
    hi = [
        st + sr * (c - 1) + b - 1 for st, sr, c, b in zip(start, stride, count, block)
    ]
    return "%s-%s" % (dims_str(start), dims_str(hi))


def virtual_str(dset, dcpl):
    """Virtual dataset mappings, `-` when the dataset is not virtual."""
    # get_virtual_count() raises on any other layout.
    if not hasattr(h5d, "VIRTUAL") or dcpl.get_layout() != h5d.VIRTUAL:
        return "-"
    count = dcpl.get_virtual_count()
    if not count:
        return "-"
    dump = _h5dump_layout(dset)
    extent_m = _H5DUMP_EXTENT_RE.search(dump)
    extent = None
    if extent_m and extent_m.group(1).strip():
        extent = [int(v) for v in extent_m.group(1).split(",")]
    mappings = _h5dump_mappings(dump)
    if len(mappings) != count:
        raise ValueError(
            "h5dump printed %d mappings where the DCPL has %d"
            % (len(mappings), count)
        )
    parts = []
    for i in range(count):
        parts.append(
            "%s::%s %s->%s"
            % (
                esc(dcpl.get_virtual_filename(i)),
                esc(dcpl.get_virtual_dsetname(i)),
                _selection_bounds(_h5dump_side(mappings[i], "SOURCE"), None),
                _selection_bounds(_h5dump_side(mappings[i], "VIRTUAL"), extent),
            )
        )
    return "[" + ",".join(parts) + "]"


def link_storage_str(gid):
    """How a group stores its links: `symtab`, `compact` or `dense`.

    libhdf5 exposes `H5G_info_t.storage_type` for exactly this, but h5py has
    no binding for `H5Gget_info`, so it is reconstructed from what h5py does
    expose. A **Symbol Table** message is the pre-1.8 group format — a v1
    B-tree and local heap — and its presence is the observable, not the object
    header version: anything else in the file can raise the header past
    version 1 over a symbol table underneath, and a file with shared messages
    does exactly that (`H5SM_init` sets `store_msg_crt_idx`, which
    `H5O__create_ohdr` turns into a version-2 header for every object).
    Otherwise the links are messages until `H5G_obj_insert`'s phase change
    moves the whole set into a fractal heap plus a name index, which is when
    libhdf5 starts sizing them. The sizes themselves are not observables: a
    bulk-loaded index and heap are legitimately smaller than ones grown insert
    by insert.
    """
    info = h5py.h5o.get_info(gid)
    if info.hdr.mesg.present & (1 << H5O_STAB_ID):
        return "symtab"
    return "dense" if info.meta_size.obj.index_size else "compact"


def crt_order_str(flags):
    """Creation-order tracking flags of a group or file creation plist."""
    parts = []
    if flags & h5p.CRT_ORDER_TRACKED:
        parts.append("tracked")
    if flags & h5p.CRT_ORDER_INDEXED:
        parts.append("indexed")
    return "+".join(parts) if parts else "-"


# --------------------------------------------------------------------------
# the dumper
# --------------------------------------------------------------------------


def oneline(exc):
    return " ".join(str(exc).split())


class Dumper:
    def __init__(self, path, access=None):
        self.path = path
        self.access = access
        self.out = []
        self.superblock, self.userblock = read_superblock(path)

    def dapl(self):
        """The `H5P_DATASET_ACCESS` list this dump opens datasets with.

        `None` when the case names no access properties, which is `H5Dopen2`
        with `H5P_DEFAULT`. The two properties it can carry are read only by
        a virtual dataset (`H5D__virtual_init`, H5Dvirtual.c:2178-2188), so
        applying the list to every dataset changes nothing else.
        """
        if not self.access:
            return None
        plist = h5p.create(h5p.DATASET_ACCESS)
        if "view" in self.access:
            plist.set_virtual_view(
                {"first_missing": h5d.VDS_FIRST_MISSING,
                 "last_available": h5d.VDS_LAST_AVAILABLE}[self.access["view"]]
            )
        if "printf_gap" in self.access:
            plist.set_virtual_printf_gap(self.access["printf_gap"])
        return plist

    def emit(self, key, value):
        self.out.append("%s\t%s" % (key, str(value).replace("\t", " ")))

    def field(self, path, name, fn):
        """Emit `path#name`, turning an oracle-side failure into ERROR(...)."""
        try:
            self.emit("%s#%s" % (path, name), fn())
        except Exception as exc:
            self.emit("%s#%s" % (path, name), "ERROR(%s): %s" % (name, oneline(exc)))

    def run(self):
        global _REF_FILE
        self.emit("!canon", CANON_VERSION)
        self.emit("#superblock", self.superblock)
        self.emit("#userblock", self.userblock)
        self.field("", "fspace", lambda: fspace_str(self.path))
        self.field("", "freespace", lambda: freespace_str(self.path))
        with h5py.File(self.path, "r") as f:
            _REF_FILE = f
            try:
                self.dump_group("/", f, 0)
            finally:
                _REF_FILE = None
        return "\n".join(self.out) + "\n"

    # -- objects ----------------------------------------------------------

    def dump_group(self, path, grp, depth):
        self.emit("%s#kind" % path, "group")
        # A File is a Group, but its `.id` is the file id, whose creation
        # plist is the FCPL and does not carry the root group's own
        # creation-order flags.
        gid = grp["/"].id if isinstance(grp, h5py.File) else grp.id
        self.field(
            path, "linkorder", lambda: crt_order_str(
                gid.get_create_plist().get_link_creation_order()
            )
        )
        self.field(
            path, "attrorder", lambda: crt_order_str(
                gid.get_create_plist().get_attr_creation_order()
            )
        )
        self.field(path, "linkstore", lambda: link_storage_str(gid))
        self.field(path, "shared", lambda: shared_str(gid, grp.file.filename))
        self.dump_attrs(path, grp)
        if depth >= MAX_DEPTH:
            self.emit("%s#truncated" % path, "depth")
            return
        for name in sorted(grp.keys()):
            child = path.rstrip("/") + "/" + name
            try:
                link = grp.get(name, getlink=True)
            except Exception as exc:
                self.emit("%s#kind" % child, "ERROR(kind): %s" % oneline(exc))
                continue
            if isinstance(link, h5py.SoftLink):
                self.emit("%s#kind" % child, "softlink")
                self.emit("%s#target" % child, link.path)
                continue
            if isinstance(link, h5py.ExternalLink):
                self.emit("%s#kind" % child, "extlink")
                self.emit("%s#target" % child, "%s::%s" % (link.filename, link.path))
                self.emit("%s#resolved" % child, resolve_extlink(grp, name))
                continue
            try:
                obj = self.open_child(grp, name)
            except Exception as exc:
                self.emit("%s#kind" % child, "ERROR(kind): %s" % oneline(exc))
                continue
            if isinstance(obj, h5py.Group):
                self.dump_group(child, obj, depth + 1)
            elif isinstance(obj, h5py.Dataset):
                self.dump_dataset(child, obj)
            elif isinstance(obj, h5py.Datatype):
                self.emit("%s#kind" % child, "committed-datatype")
                self.field(child, "dtype", lambda o=obj: canon_dtype(o.id))
                self.field(child, "strpad", lambda o=obj: strpad_str(o.id))
                self.dump_attrs(child, obj)
            else:
                self.emit("%s#kind" % child, "unknown")

    def open_child(self, grp, name):
        """Open `name` under `grp` with the case's dataset access list.

        The list is read by the open that first brings the dataset into
        memory and by no later one: `H5D_open` hands `dapl_id` to
        `H5D__open_oid` only when the object is not already in the file's
        open-object list, and otherwise adopts the existing shared struct
        unexamined (H5Dint.c:1496-1525). Reopening a dataset h5py has
        already handed out under a second list therefore changes nothing,
        so the list has to ride on the first open.
        """
        dapl = self.dapl()
        if dapl is None or grp.get(name, getclass=True) is not h5py.Dataset:
            return grp[name]
        return h5py.Dataset(h5d.open(grp.id, name.encode("utf-8"), dapl=dapl))

    def dump_dataset(self, path, dset):
        self.emit("%s#kind" % path, "dataset")
        dsid = dset.id
        tid = dsid.get_type()
        sid = dsid.get_space()
        dcpl = dsid.get_create_plist()
        space_type = sid.get_simple_extent_type()

        self.field(path, "dtype", lambda: canon_dtype(tid))
        self.field(path, "strpad", lambda: strpad_str(tid))

        if space_type == h5s.NULL:
            shape, maxshape = None, None
            self.emit("%s#shape" % path, "null")
            self.emit("%s#maxshape" % path, "null")
        else:
            shape = sid.get_simple_extent_dims()
            maxdims = sid.get_simple_extent_dims(True)
            maxshape = tuple(None if d == h5s.UNLIMITED else d for d in maxdims)
            self.emit("%s#shape" % path, dims_str(shape))
            self.emit("%s#maxshape" % path, maxdims_str(maxshape))

        layout = _LAYOUTS.get(dcpl.get_layout(), "layout%d" % dcpl.get_layout())
        self.emit("%s#layout" % path, layout)

        if layout == "chunked":
            self.emit("%s#chunk" % path, dims_str(dcpl.get_chunk()))
            self.field(path, "chunkindex", lambda: chunk_index_str(dset))
        else:
            self.emit("%s#chunk" % path, "-")
            self.emit("%s#chunkindex" % path, "-")

        self.field(path, "external", lambda: external_str(dcpl))
        self.field(path, "virtual", lambda: virtual_str(dset, dcpl))
        self.field(path, "filters", lambda: filters_str(dset))
        self.field(path, "fillvalue", lambda: self.fill_value(dset, dcpl))
        self.field(path, "filltime", lambda: self.fill_time(dcpl))
        self.field(path, "alloctime", lambda: self.alloc_time(dcpl))
        self.field(path, "shared", lambda: shared_str(dsid, dset.file.filename))
        self.dump_attrs(path, dset)
        self.field(path, "data", lambda: dataset_payload(dset))

    def fill_value(self, dset, dcpl):
        defined = dcpl.fill_value_defined()
        if defined == h5d.FILL_VALUE_UNDEFINED:
            return "undefined"
        if defined == h5d.FILL_VALUE_DEFAULT:
            return "default"
        buf = np.zeros((1,), dtype=dset.dtype)
        dcpl.get_fill_value(buf)
        return "0x" + buf.tobytes().hex()

    def fill_time(self, dcpl):
        # `H5Pget_fill_time` reads back from the on-disk fill-value message,
        # the same message `fill_value` above reads — not a client-side
        # property-list echo.
        return {
            h5d.FILL_TIME_ALLOC: "alloc",
            h5d.FILL_TIME_NEVER: "never",
            h5d.FILL_TIME_IFSET: "ifset",
        }[dcpl.get_fill_time()]

    def alloc_time(self, dcpl):
        # `H5Pget_alloc_time` reads back from the same fill-value message
        # `fill_time` above reads — not a client-side property-list echo.
        return {
            h5d.ALLOC_TIME_EARLY: "early",
            h5d.ALLOC_TIME_LATE: "late",
            h5d.ALLOC_TIME_INCR: "incr",
        }[dcpl.get_alloc_time()]

    def dump_attrs(self, path, obj):
        try:
            names = sorted(obj.attrs.keys())
        except Exception as exc:
            self.emit("%s#nattrs" % path, "ERROR(nattrs): %s" % oneline(exc))
            return
        self.emit("%s#nattrs" % path, len(names))
        # The object header's own attribute count, which is what H5Oget_info
        # and therefore h5diff/h5repack trust. Iteration walks the messages
        # and can disagree with it, so the two are separate observables.
        self.field(
            path,
            "nattrs_hdr",
            lambda o=obj: h5py.h5o.get_info(o.id).num_attrs,
        )
        # Compact vs dense storage. libhdf5 sizes the name index and the
        # fractal heap only when the whole set has moved out of the object
        # header, so a nonzero index size is `H5O__attr_create`'s phase change
        # having fired. The sizes themselves are not observables: a bulk-loaded
        # index and heap are legitimately smaller than ones grown insert by
        # insert.
        self.field(
            path,
            "attrstore",
            lambda o=obj: (
                "dense"
                if h5py.h5o.get_info(o.id).meta_size.attr.index_size
                else "compact"
            ),
        )
        for name in names:
            key = "%s@%s" % (path, name)
            try:
                aid = h5py.h5a.open(obj.id, name.encode("utf-8"))
            except Exception as exc:
                self.emit("%s#dtype" % key, "ERROR(dtype): %s" % oneline(exc))
                continue
            tid = aid.get_type()
            sid = aid.get_space()
            self.field(key, "dtype", lambda t=tid: canon_dtype(t))
            self.field(key, "strpad", lambda t=tid: strpad_str(t))
            if sid.get_simple_extent_type() == h5s.NULL:
                self.emit("%s#shape" % key, "null")
            else:
                self.emit("%s#shape" % key, dims_str(sid.get_simple_extent_dims()))
            self.field(
                key,
                "value",
                lambda o=obj, n=name, a=aid, t=tid, s=sid: attr_payload(o, n, a, t, s),
            )


_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def read_superblock(path):
    """(superblock version, user block size).

    A user block displaces the superblock to the first power-of-two offset at
    or after 512 bytes, so the signature has to be searched for rather than
    read at zero.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    offset = 0
    while offset < len(data):
        if data[offset : offset + 8] == _SIGNATURE:
            return data[offset + 8], offset
        offset = 512 if offset == 0 else offset * 2
    return "ERROR(superblock): no signature at any user-block offset", 0


# H5F_fspace_strategy_t (H5Fpublic.h) as the canon names it.
_FS_STRATEGY_NAMES = {0: "fsmaggr", 1: "page", 2: "aggr", 3: "none"}


def fspace_str(path):
    """The `fspace` field: `<strategy>/<persist>/<threshold>/<page size>`.

    Read from the file creation property list, which `H5F__super_read` fills
    from the on-disk file-space info message when the file carries one and
    leaves at the library defaults (`H5F_FILE_SPACE_STRATEGY_DEF` = FSM_AGGR,
    no persist, threshold 1, page size 4096) when it does not. That is the
    same question the rust side answers from the message itself, so a file
    without the message reports the defaults on both sides.

    All four are here because all four are what `H5F__super_init` compares
    against those defaults to decide whether the file needs the message at
    all: a file differing only in its page size still carries one.
    """
    with h5py.File(path, "r") as f:
        plist = f.id.get_create_plist()
        strategy, persist, threshold = plist.get_file_space_strategy()
        page_size = plist.get_file_space_page_size()
    return "%s/%s/%d/%d" % (
        _FS_STRATEGY_NAMES.get(strategy, "unknown(%d)" % strategy),
        "true" if persist else "false",
        threshold,
        page_size,
    )


_H5STAT_FREE_RE = re.compile(r"tracked free space:\s*(\d+)\s*bytes")


def freespace_str(path):
    """The `freespace` field: `tracked` or `none`.

    Whether the file's on-disk free-space managers record any space at all,
    read from `h5stat -S`'s "Amount/Percent of tracked free space" line —
    `H5Fget_freespace` has no h5py binding.

    A *count* would not be comparable: the two writers lay a file out
    differently, so the same sequence of creates and appends frees different
    numbers of bytes. Whether any manager holds anything is the property the
    two must agree on, and it is the one that separates a file whose freed
    space is recorded from a file that leaked it.
    """
    proc = subprocess.run(
        [_tool_bin("h5stat"), "-S", path],
        capture_output=True,
        text=True,
        check=True,
    )
    match = _H5STAT_FREE_RE.search(proc.stdout)
    if match is None:
        raise ValueError("h5stat -S printed no tracked-free-space line")
    return "tracked" if int(match.group(1)) > 0 else "none"


def dump(path, access=None):
    return Dumper(path, access).run()


def main(argv):
    """usage: canon.py [--virtual-view V] [--printf-gap N] <file.h5>"""
    access, args = {}, []
    it = iter(argv[1:])
    for arg in it:
        if arg == "--virtual-view":
            access["view"] = next(it)
        elif arg == "--printf-gap":
            access["printf_gap"] = int(next(it))
        else:
            args.append(arg)
    if len(args) != 1:
        sys.stderr.write(main.__doc__.partition(": ")[2] + "\n")
        return 64
    try:
        sys.stdout.write(dump(args[0], access or None))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
