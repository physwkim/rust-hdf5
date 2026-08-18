#!/usr/bin/env python3
"""Canonical h5py-side dump of an HDF5 file — the reference half of the oracle.

Emits the `!canon 2` format described in oracle/CANON.md. The rust-hdf5 side
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

CANON_VERSION = "5"
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


def _h5debug_bin():
    """Locate `h5debug` the way `run.py` locates `h5dump`/`h5diff`: alongside
    this interpreter unless `RUST_HDF5_ORACLE_BINDIR` overrides it."""
    bindir = os.environ.get(
        "RUST_HDF5_ORACLE_BINDIR", str(pathlib.Path(sys.executable).parent)
    )
    return str(pathlib.Path(bindir) / "h5debug")


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


def _bounds_str(sid):
    try:
        lo, hi = sid.get_select_bounds()
    except Exception:
        return "?"
    return "%s-%s" % (dims_str(lo), dims_str(hi))


def virtual_str(dcpl):
    """Virtual dataset mappings, `-` when the dataset is not virtual."""
    # get_virtual_count() raises on any other layout.
    if not hasattr(h5d, "VIRTUAL") or dcpl.get_layout() != h5d.VIRTUAL:
        return "-"
    count = dcpl.get_virtual_count()
    if not count:
        return "-"
    parts = []
    for i in range(count):
        parts.append(
            "%s::%s %s->%s"
            % (
                esc(dcpl.get_virtual_filename(i)),
                esc(dcpl.get_virtual_dsetname(i)),
                _bounds_str(dcpl.get_virtual_srcspace(i)),
                _bounds_str(dcpl.get_virtual_vspace(i)),
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
    def __init__(self, path):
        self.path = path
        self.out = []
        self.superblock, self.userblock = read_superblock(path)

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
                obj = grp[name]
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
        self.field(path, "virtual", lambda: virtual_str(dcpl))
        self.field(path, "filters", lambda: filters_str(dset))
        self.field(path, "fillvalue", lambda: self.fill_value(dset, dcpl))
        self.field(path, "filltime", lambda: self.fill_time(dcpl))
        self.field(path, "alloctime", lambda: self.alloc_time(dcpl))
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


def dump(path):
    return Dumper(path).run()


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: canon.py <file.h5>\n")
        return 64
    try:
        sys.stdout.write(dump(argv[1]))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
