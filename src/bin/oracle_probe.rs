//! The rust-hdf5 half of the parity oracle.
//!
//! Two subcommands:
//!
//! * `dump <file.h5>` walks any HDF5 file through the **public** rust-hdf5 API
//!   only and prints the canonical description defined in `oracle/CANON.md`.
//!   `oracle/canon.py` prints the same description of the same file from
//!   h5py/libhdf5, so the two are comparable field by field. Where the public
//!   API exposes no way to observe a field, the value is
//!   `UNSUPPORTED(<field>): <why>` — that is the measurement, not a failure.
//!   Nothing here may panic or abort the walk: a per-object guard turns a
//!   panic or an error into a marker line and the walk continues.
//!
//! * `write <case> <file.h5>` writes the rust-hdf5 equivalent of one case from
//!   `oracle/cases.py`, so the runner can check the other direction. A case the
//!   public API cannot express prints `UNSUPPORTED-API: <why>` and exits 2.
//!
//! Deliberately restricted to the crate's public surface plus `std`; the
//! internal reader/writer types are never touched, because the point of the
//! oracle is to measure what a user of the published API can see and produce.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use rust_hdf5::format::messages::datatype::{
    ByteOrder, CompoundMember, DatatypeMessage, EnumMember,
};
use rust_hdf5::format::messages::filter::{Filter, FilterPipeline, FILTER_FLETCHER32};
use rust_hdf5::types::VarLenUnicode;
use rust_hdf5::{
    H5Attribute, H5Dataset, H5File, H5FileOptions, H5Group, H5NamedDatatype, Hdf5Error, Hyperslab,
    HyperslabBlock, LibverBound, LinkClass, Reference, Selection,
};

const CANON_VERSION: &str = "3";
const RAW_LIMIT: usize = 1024;
const MAX_DEPTH: usize = 32;

// ===========================================================================
// canonical encoding — mirrors oracle/canon.py exactly
// ===========================================================================

/// Canonical quoted-string encoding; the twin of `canon.py`'s `esc`.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let o = c as u32;
                if o <= 0xff {
                    out.push_str(&format!("\\x{o:02x}"));
                } else if o <= 0xffff {
                    out.push_str(&format!("\\u{o:04x}"));
                } else {
                    out.push_str(&format!("\\U{o:08x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn dims_str(dims: &[usize]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(","))
}

/// Apply the size policy shared with `canon.py`: raw bytes up to 1 KiB, the
/// SHA-256 of those same bytes beyond it.
fn encode_raw(bytes: &[u8]) -> String {
    if bytes.len() <= RAW_LIMIT {
        format!("raw:{}", hex(bytes))
    } else {
        format!("sha256:{}", hex(&sha256(bytes)))
    }
}

fn encode_vals(vals: &[String]) -> String {
    let body = format!("vals:[{}]", vals.join(","));
    if body.len() <= RAW_LIMIT {
        body
    } else {
        format!("valsha256:{}", hex(&sha256(body.as_bytes())))
    }
}

// ---------------------------------------------------------------------------
// datatype canonicalisation
// ---------------------------------------------------------------------------

fn order_str(o: &ByteOrder) -> &'static str {
    match o {
        ByteOrder::LittleEndian => "le",
        ByteOrder::BigEndian => "be",
    }
}

/// IEEE 754 parameters per width: (sign, exp_pos, exp_size, mant_pos,
/// mant_size, bias). A float that matches these prints without a suffix.
fn ieee_params(size: u32) -> Option<(u8, u8, u8, u8, u8, u32)> {
    match size {
        2 => Some((15, 10, 5, 0, 10, 15)),
        4 => Some((31, 23, 8, 0, 23, 127)),
        8 => Some((63, 52, 11, 0, 52, 1023)),
        _ => None,
    }
}

fn charset_str(c: u8) -> String {
    match c {
        0 => "ascii".into(),
        1 => "utf8".into(),
        n => format!("cset{n}"),
    }
}

fn strpad_str(p: u8) -> String {
    match p {
        0 => "null".into(),
        1 => "nullpad".into(),
        2 => "spacepad".into(),
        n => format!("pad{n}"),
    }
}

fn canon_dtype(dt: &DatatypeMessage) -> String {
    match dt {
        DatatypeMessage::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            let mut s = format!(
                "{}{}{}",
                if *signed { "i" } else { "u" },
                size * 8,
                order_str(byte_order)
            );
            if *bit_offset != 0 || u32::from(*bit_precision) != size * 8 {
                s.push_str(&format!("+off{bit_offset}p{bit_precision}"));
            }
            s
        }
        DatatypeMessage::FloatingPoint {
            size,
            byte_order,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } => {
            let mut s = format!("f{}{}", size * 8, order_str(byte_order));
            let actual = (
                *sign_location,
                *exponent_location,
                *exponent_size,
                *mantissa_location,
                *mantissa_size,
                *exponent_bias,
            );
            let standard = ieee_params(*size);
            if standard != Some(actual) || *bit_offset != 0 || u32::from(*bit_precision) != size * 8
            {
                s.push_str(&format!(
                    "+s{sign_location}e{exponent_location},{exponent_size}\
                     m{mantissa_location},{mantissa_size}b{exponent_bias}\
                     off{bit_offset}p{bit_precision}"
                ));
            }
            s
        }
        DatatypeMessage::BitField {
            size,
            byte_order,
            bit_offset,
            bit_precision,
        } => {
            let mut s = format!("bits[{size}]{}", order_str(byte_order));
            if *bit_offset != 0 || u32::from(*bit_precision) != size * 8 {
                s.push_str(&format!("+off{bit_offset}p{bit_precision}"));
            }
            s
        }
        DatatypeMessage::Opaque { size, tag } => format!("opaque[{size}],tag={}", esc(tag)),
        DatatypeMessage::FixedString {
            size,
            padding,
            charset,
        } => format!(
            "str[{size}],pad={},cset={}",
            strpad_str(*padding),
            charset_str(*charset)
        ),
        // The pad is deliberately absent here: it travels in the separate
        // `strpad` field, and canon.py omits it too. See oracle/CANON.md.
        DatatypeMessage::VarLenString { charset, .. } => {
            format!("vstr,cset={}", charset_str(*charset))
        }
        DatatypeMessage::Compound { size, members } => {
            let parts: Vec<String> = members
                .iter()
                .map(|m| format!("{}@{}:{}", m.name, m.offset, canon_dtype(&m.datatype)))
                .collect();
            format!("compound[{size}]{{{}}}", parts.join(";"))
        }
        DatatypeMessage::Enum { base, members } => {
            let parts: Vec<String> = members
                .iter()
                .map(|m| format!("{}={}", m.name, enum_value(base, &m.value)))
                .collect();
            format!("enum({}){{{}}}", canon_dtype(base), parts.join(";"))
        }
        // canon.py splits the class by element width, not by the stored
        // reference type: an 8-byte element is an object reference, anything
        // else a region one. That rule only covers what h5py can express —
        // the 1.12 kinds, which it refuses outright, get their own name so a
        // file holding them is never reported as a pre-1.12 region reference.
        DatatypeMessage::Reference { size, kind } => if kind.is_revised() {
            "stdref"
        } else if *size == 8 {
            "objref"
        } else {
            "regref"
        }
        .to_string(),
        DatatypeMessage::VarLenSequence { base } => format!("vlen({})", canon_dtype(base)),
        DatatypeMessage::Array { dims, base } => {
            let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
            format!("array[{}]({})", parts.join(","), canon_dtype(base))
        }
    }
}

/// Decode an enum member's raw bytes to the decimal libhdf5 reports.
fn enum_value(base: &DatatypeMessage, raw: &[u8]) -> String {
    if let DatatypeMessage::FixedPoint {
        size,
        byte_order,
        signed,
        ..
    } = base
    {
        let n = (*size as usize).min(raw.len()).min(16);
        let mut le = [0u8; 16];
        match byte_order {
            ByteOrder::LittleEndian => le[..n].copy_from_slice(&raw[..n]),
            ByteOrder::BigEndian => {
                for (i, slot) in le[..n].iter_mut().enumerate() {
                    *slot = raw[n - 1 - i];
                }
            }
        }
        if *signed {
            if n == 0 || n >= 16 {
                return i128::from_le_bytes(le).to_string();
            }
            if le[n - 1] & 0x80 != 0 {
                for slot in le[n..].iter_mut() {
                    *slot = 0xff;
                }
            }
            i128::from_le_bytes(le).to_string()
        } else {
            u128::from_le_bytes(le).to_string()
        }
    } else {
        format!("0x{}", hex(raw))
    }
}

/// One element of a variable-length sequence, rendered as canon.py renders it.
fn render_elem(base: &DatatypeMessage, bytes: &[u8]) -> String {
    match base {
        DatatypeMessage::FixedPoint { .. } => enum_value(base, bytes),
        DatatypeMessage::FloatingPoint {
            size, byte_order, ..
        } => {
            // Big-endian IEEE bits, exactly as canon.py's `float_bits`.
            let mut be: Vec<u8> = bytes[..(*size as usize).min(bytes.len())].to_vec();
            if matches!(byte_order, ByteOrder::LittleEndian) {
                be.reverse();
            }
            format!("0x{}", hex(&be))
        }
        _ => format!("0x{}", hex(bytes)),
    }
}

/// True when the element image is not comparable between two writers, so the
/// canonical form is the rendered values rather than the raw bytes: a
/// variable-length payload lives in a heap the element only points at, and a
/// reference names a file address whose value is an allocation detail.
fn renders_as_values(dt: &DatatypeMessage) -> bool {
    match dt {
        DatatypeMessage::VarLenString { .. }
        | DatatypeMessage::VarLenSequence { .. }
        | DatatypeMessage::Reference { .. } => true,
        DatatypeMessage::Array { base, .. } => renders_as_values(base),
        DatatypeMessage::Compound { members, .. } => {
            members.iter().any(|m| renders_as_values(&m.datatype))
        }
        _ => false,
    }
}

/// One reference element in the form `oracle/canon.py`'s `render_ref` prints:
/// the target's path, plus the selection's bounding box for a region
/// reference. An address the reader could not name is printed as the address,
/// which compares unequal to h5py's path — a difference, not a silent match.
fn render_ref(r: &Reference) -> String {
    fn target(path: &Option<String>, address: u64) -> String {
        path.clone().unwrap_or_else(|| format!("<{address:#x}>"))
    }
    fn coords(dims: &[u64]) -> String {
        let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
        format!("[{}]", parts.join(","))
    }
    match r {
        Reference::Null => "objref:null".to_string(),
        Reference::Object { address, path } => format!("objref:{}", target(path, *address)),
        Reference::Region {
            address,
            path,
            selection,
        } => match selection.bounds() {
            Some((lo, hi)) => format!(
                "regref:{}:{}-{}",
                target(path, *address),
                coords(&lo),
                coords(&hi)
            ),
            None => format!("regref:{}:unbounded", target(path, *address)),
        },
        // No h5py-generated case can reach this arm: h5py 3.x refuses the
        // `H5T_STD_REF` datatype an attribute reference needs.
        Reference::Attr {
            address,
            path,
            name,
        } => format!("attrref:{}:{name}", target(path, *address)),
    }
}

/// `where=pad` for every variable-length string in the type tree, appended to
/// `out`.
///
/// `where` is the position in the type tree, as `oracle/CANON.md` defines it:
/// `.` is the type itself, `.m` a compound member, `[]` an array element, `()`
/// a vlen element.
fn vlen_strpads(dt: &DatatypeMessage, whence: &str, out: &mut Vec<String>) {
    match dt {
        DatatypeMessage::VarLenString { padding, .. } => {
            let at = if whence.is_empty() { "." } else { whence };
            out.push(format!("{at}={}", strpad_str(*padding)));
        }
        DatatypeMessage::Array { base, .. } => vlen_strpads(base, &format!("{whence}[]"), out),
        DatatypeMessage::VarLenSequence { base } => vlen_strpads(base, &format!("{whence}()"), out),
        DatatypeMessage::Compound { members, .. } => {
            for m in members {
                vlen_strpads(&m.datatype, &format!("{whence}.{}", m.name), out);
            }
        }
        _ => {}
    }
}

/// The `strpad` field: `-` when the type tree holds no variable-length string,
/// otherwise one `position=pad` entry per such string.
fn strpad_field(dtype: Option<&DatatypeMessage>) -> std::result::Result<String, String> {
    match dtype {
        Some(dt) => {
            let mut pads = Vec::new();
            vlen_strpads(dt, "", &mut pads);
            Ok(if pads.is_empty() {
                "-".into()
            } else {
                pads.join(";")
            })
        }
        None => Err("datatype unavailable, so the string pad cannot be classified".into()),
    }
}

// ===========================================================================
// SHA-256 (no external crates are allowed in this repository)
// ===========================================================================

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    let mut w = [0u32; 64];
    for block in msg.chunks_exact(64) {
        for (i, slot) in w.iter_mut().take(16).enumerate() {
            *slot = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

// ===========================================================================
// the dump side
// ===========================================================================

fn unsupported(field: &str, why: &str) -> String {
    format!("UNSUPPORTED({field}): {why}")
}

fn oneline(e: impl std::fmt::Display) -> String {
    e.to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run `f`, converting a panic into an `Err(message)` so one bad object never
/// aborts the walk.
fn guarded<T>(f: impl FnOnce() -> T) -> std::result::Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_string());
        oneline(msg)
    })
}

struct Dump {
    lines: Vec<String>,
}

impl Dump {
    fn new() -> Self {
        Self { lines: Vec::new() }
    }

    fn emit(&mut self, key: &str, value: impl AsRef<str>) {
        self.lines
            .push(format!("{key}\t{}", value.as_ref().replace('\t', " ")));
    }

    /// Emit `path#field`, turning an error into the canonical marker.
    fn field(
        &mut self,
        path: &str,
        field: &str,
        f: impl FnOnce() -> std::result::Result<String, String>,
    ) {
        let value = match guarded(f) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => unsupported(field, &e),
            Err(p) => unsupported(field, &format!("panic: {p}")),
        };
        self.emit(&format!("{path}#{field}"), value);
    }
}

/// What the walk will say about one child name of a group.
enum Child {
    Group,
    Dataset,
    /// A committed (named) datatype: an object, and neither of the above.
    NamedDatatype,
    Soft(String),
    External(String, String),
    /// The name is linked but the public API answers neither "which kind of
    /// object" nor "which kind of link"; the reason rides along.
    Unclassified(String),
}

fn child_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn dump_file(path: &str) -> std::result::Result<String, String> {
    let mut d = Dump::new();
    d.emit("!canon", CANON_VERSION);
    // The superblock version is what pins the libver bound a file was written
    // under, and there is no public accessor for it on H5File.
    d.emit(
        "#superblock",
        unsupported("superblock", "H5File exposes no superblock/libver accessor"),
    );

    let file = match guarded(|| H5File::open(path)) {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => return Err(format!("H5File::open failed: {}", oneline(e))),
        Err(p) => return Err(format!("H5File::open panicked: {p}")),
    };

    // `H5File::userblock_size` answers in either mode, so this is a value the
    // canon can be compared against rather than an API gap. It has to come
    // after the open — it is a property of the file, not of the path.
    d.field("", "userblock", || Ok(file.userblock_size().to_string()));

    let root = file.root_group();
    dump_group(&mut d, &file, "/", &root, 0);
    Ok(d.lines.join("\n") + "\n")
}

fn dump_group(d: &mut Dump, file: &H5File, path: &str, group: &H5Group, depth: usize) {
    d.emit(&format!("{path}#kind"), "group");
    d.field(path, "linkorder", || {
        Err("H5Group exposes no link creation-order tracking flags".into())
    });
    d.field(path, "attrorder", || {
        Err("H5Group exposes no attribute creation-order tracking flags".into())
    });
    d.emit(
        &format!("{path}#linkstore"),
        unsupported(
            "linkstore",
            "H5Group exposes no compact/dense link storage accessor",
        ),
    );
    dump_group_attrs(d, path, group);

    if depth >= MAX_DEPTH {
        d.emit(&format!("{path}#truncated"), "depth");
        return;
    }

    // canon.py walks one sorted list of link names; merge the typed listings
    // back into that single order so the text diffs line up.
    let mut children: BTreeMap<String, Child> = BTreeMap::new();
    match guarded(|| group.group_names()) {
        Ok(Ok(names)) => {
            for n in names {
                children.insert(n, Child::Group);
            }
        }
        Ok(Err(e)) => d.emit(
            &format!("{path}#group_names"),
            unsupported("group_names", &oneline(e)),
        ),
        Err(p) => d.emit(
            &format!("{path}#group_names"),
            unsupported("group_names", &format!("panic: {p}")),
        ),
    }
    match guarded(|| group.dataset_names()) {
        Ok(Ok(names)) => {
            for n in names {
                children.insert(n, Child::Dataset);
            }
        }
        Ok(Err(e)) => d.emit(
            &format!("{path}#dataset_names"),
            unsupported("dataset_names", &oneline(e)),
        ),
        Err(p) => d.emit(
            &format!("{path}#dataset_names"),
            unsupported("dataset_names", &format!("panic: {p}")),
        ),
    }
    match guarded(|| group.named_datatype_names()) {
        Ok(Ok(names)) => {
            for n in names {
                children.insert(n, Child::NamedDatatype);
            }
        }
        Ok(Err(e)) => d.emit(
            &format!("{path}#named_datatype_names"),
            unsupported("named_datatype_names", &oneline(e)),
        ),
        Err(p) => d.emit(
            &format!("{path}#named_datatype_names"),
            unsupported("named_datatype_names", &format!("panic: {p}")),
        ),
    }
    // canon.py classifies by link kind first (`grp.get(name, getlink=True)`)
    // and only falls through to the object type for a hard link, so the link
    // listing both adds names the typed listings cannot answer for and
    // overrides the ones that are not hard links.
    match guarded(|| group.link_names()) {
        Ok(Ok(names)) => {
            for n in names {
                let class = match guarded(|| group.link_class(&n)) {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => {
                        children.insert(n, Child::Unclassified(oneline(e)));
                        continue;
                    }
                    Err(p) => {
                        children.insert(n, Child::Unclassified(format!("panic: {p}")));
                        continue;
                    }
                };
                match class {
                    // A hard link is described by the object it reaches, which
                    // the typed listings have already classified; only when
                    // they have not does the name need a marker of its own.
                    LinkClass::Hard => {
                        children.entry(n).or_insert_with(|| {
                            Child::Unclassified(
                                "the link is hard but the object it reaches is in neither \
                                 group_names() nor dataset_names()"
                                    .into(),
                            )
                        });
                    }
                    LinkClass::Soft { path } => {
                        children.insert(n, Child::Soft(path));
                    }
                    LinkClass::External { file, path } => {
                        children.insert(n, Child::External(file, path));
                    }
                    LinkClass::UserDefined { link_type } => {
                        children.insert(
                            n,
                            Child::Unclassified(format!(
                                "user-defined link of type {link_type}, which this crate \
                                 does not interpret"
                            )),
                        );
                    }
                }
            }
        }
        Ok(Err(e)) => d.emit(
            &format!("{path}#link_names"),
            unsupported("link_names", &oneline(e)),
        ),
        Err(p) => d.emit(
            &format!("{path}#link_names"),
            unsupported("link_names", &format!("panic: {p}")),
        ),
    }

    for (name, child) in children {
        let cpath = child_path(path, &name);
        match child {
            Child::Group => match guarded(|| group.group(&name)) {
                Ok(Ok(sub)) => dump_group(d, file, &cpath, &sub, depth + 1),
                Ok(Err(e)) => d.emit(&format!("{cpath}#kind"), unsupported("kind", &oneline(e))),
                Err(p) => d.emit(
                    &format!("{cpath}#kind"),
                    unsupported("kind", &format!("panic: {p}")),
                ),
            },
            Child::Dataset => {
                let lookup = cpath.trim_start_matches('/').to_string();
                match guarded(|| file.dataset(&lookup)) {
                    Ok(Ok(ds)) => dump_dataset(d, &cpath, &ds),
                    Ok(Err(e)) => {
                        d.emit(&format!("{cpath}#kind"), unsupported("kind", &oneline(e)))
                    }
                    Err(p) => d.emit(
                        &format!("{cpath}#kind"),
                        unsupported("kind", &format!("panic: {p}")),
                    ),
                }
            }
            Child::NamedDatatype => match guarded(|| group.named_datatype(&name)) {
                Ok(Ok(t)) => dump_named_datatype(d, &cpath, &t),
                Ok(Err(e)) => d.emit(&format!("{cpath}#kind"), unsupported("kind", &oneline(e))),
                Err(p) => d.emit(
                    &format!("{cpath}#kind"),
                    unsupported("kind", &format!("panic: {p}")),
                ),
            },
            // A soft link is reported by its value and never followed here,
            // exactly as canon.py reports it.
            Child::Soft(target) => {
                d.emit(&format!("{cpath}#kind"), "softlink");
                d.emit(&format!("{cpath}#target"), target);
            }
            // An external link reports its value and then what crossing it
            // lands on, so the dump distinguishes a reader that follows the
            // link from one that only lists it.
            Child::External(efile, epath) => {
                d.emit(&format!("{cpath}#kind"), "extlink");
                d.emit(&format!("{cpath}#target"), format!("{efile}::{epath}"));
                d.field(&cpath, "resolved", || resolve_extlink(file, &cpath));
            }
            Child::Unclassified(why) => d.emit(&format!("{cpath}#kind"), unsupported("kind", &why)),
        }
    }
}

/// What crossing an external link lands on — canon.py's `resolve_extlink`.
///
/// `H5File::dataset` is the only public entry point that crosses a link, so a
/// target that is a group answers as a capability gap rather than as `group`;
/// that gap is real and is what the field is here to measure.
fn resolve_extlink(file: &H5File, cpath: &str) -> std::result::Result<String, String> {
    let lookup = cpath.trim_start_matches('/').to_string();
    // A committed datatype is an object of its own, so it is asked about
    // first; every other answer comes from the dataset entry point.
    if let Ok(Ok(_)) = guarded(|| file.named_datatype(&lookup)) {
        return Ok("committed-datatype".into());
    }
    match guarded(|| file.dataset(&lookup)) {
        Ok(Ok(ds)) => {
            let dims = guarded(|| ds.shape()).map_err(|p| format!("panic: {p}"))?;
            let dtype = guarded(|| ds.datatype()).ok().and_then(|r| r.ok());
            let payload = dataset_payload(&ds, dtype.as_ref())?;
            Ok(format!("dataset {} {}", dims_str(&dims), payload))
        }
        // A missing target file and a missing target object are one answer,
        // as they are on the h5py side.
        Ok(Err(Hdf5Error::DanglingLink { .. }))
        | Ok(Err(Hdf5Error::ExternalFileNotFound { .. })) => Ok("dangling".into()),
        Ok(Err(e)) => Err(oneline(e)),
        Err(p) => Err(format!("panic: {p}")),
    }
}

fn dump_dataset(d: &mut Dump, path: &str, ds: &H5Dataset) {
    d.emit(&format!("{path}#kind"), "dataset");

    let dtype = guarded(|| ds.datatype()).ok().and_then(|r| r.ok());

    d.field(path, "dtype", || match &dtype {
        Some(dt) => Ok(canon_dtype(dt)),
        None => Err("H5Dataset::datatype() failed or is unavailable".into()),
    });

    let is_null = guarded(|| ds.is_null()).unwrap_or(false);

    d.field(path, "strpad", || strpad_field(dtype.as_ref()));

    // H5Dataset::shape() returns Vec<usize>, so a scalar dataspace and a NULL
    // dataspace are both the empty vector and cannot be told apart.
    d.field(path, "shape", || {
        if is_null {
            return Ok("null".into());
        }
        Ok(dims_str(
            &guarded(|| ds.shape()).map_err(|p| format!("panic: {p}"))?,
        ))
    });

    d.field(path, "maxshape", || {
        Err("H5Dataset exposes no max_shape() accessor".into())
    });

    let chunked = guarded(|| ds.is_chunked()).unwrap_or(false);

    d.field(path, "layout", || {
        if chunked {
            Ok("chunked".into())
        } else {
            Err(
                "H5Dataset::is_chunked() is false; contiguous and compact are \
                 not distinguishable through the public API"
                    .into(),
            )
        }
    });

    d.field(path, "chunk", || {
        if !chunked {
            return Ok("-".into());
        }
        match guarded(|| ds.chunk_dims()).map_err(|p| format!("panic: {p}"))? {
            Some(dims) => Ok(dims_str(&dims)),
            None => Err("is_chunked() is true but chunk_dims() returned None".into()),
        }
    });

    d.field(path, "chunkindex", || {
        if chunked {
            Err("H5Dataset exposes no chunk index type".into())
        } else {
            Ok("-".into())
        }
    });

    d.field(path, "external", || {
        Err("H5Dataset exposes no external file list accessor".into())
    });

    d.field(path, "virtual", || {
        Err("H5Dataset exposes no virtual mapping accessor".into())
    });

    d.field(path, "filters", || {
        Err("H5Dataset exposes no filter pipeline accessor".into())
    });

    d.field(path, "fillvalue", || {
        Err("H5Dataset exposes no fill value accessor".into())
    });

    dump_object_attrs(d, path, ds);

    d.field(path, "data", || dataset_payload(ds, dtype.as_ref()));
}

fn dataset_payload(
    ds: &H5Dataset,
    dtype: Option<&DatatypeMessage>,
) -> std::result::Result<String, String> {
    if guarded(|| ds.is_null()).unwrap_or(false) {
        return Ok("empty".into());
    }
    let dt = dtype.ok_or("datatype unavailable, so the payload cannot be classified")?;
    if !renders_as_values(dt) {
        let bytes = guarded(|| ds.read_raw_bytes())
            .map_err(|p| format!("panic: {p}"))?
            .map_err(oneline)?;
        return Ok(encode_raw(&bytes));
    }
    match dt {
        DatatypeMessage::VarLenString { .. } => {
            let strings = guarded(|| ds.read_vlen_strings())
                .map_err(|p| format!("panic: {p}"))?
                .map_err(oneline)?;
            let vals: Vec<String> = strings.iter().map(|s| esc(s)).collect();
            Ok(encode_vals(&vals))
        }
        DatatypeMessage::VarLenSequence { base } => {
            let width = base.element_size() as usize;
            if width == 0 {
                return Err("variable-length sequence with a zero-width base".into());
            }
            let items = guarded(|| ds.read_vlen_bytes())
                .map_err(|p| format!("panic: {p}"))?
                .map_err(oneline)?;
            let vals: Vec<String> = items
                .iter()
                .map(|item| {
                    let elems: Vec<String> =
                        item.chunks(width).map(|c| render_elem(base, c)).collect();
                    format!("[{}]", elems.join(","))
                })
                .collect();
            Ok(encode_vals(&vals))
        }
        DatatypeMessage::Reference { .. } => {
            let refs = guarded(|| ds.read_references())
                .map_err(|p| format!("panic: {p}"))?
                .map_err(oneline)?;
            let vals: Vec<String> = refs.iter().map(render_ref).collect();
            Ok(encode_vals(&vals))
        }
        other => Err(format!(
            "no public reader for a {} payload",
            canon_dtype(other)
        )),
    }
}

/// An object whose attributes are readable through a typed handle. Datasets
/// and committed datatypes both are, and canon.py dumps their attributes with
/// one function, so this side does too.
trait AttrSource {
    fn attr_names(&self) -> rust_hdf5::Result<Vec<String>>;
    fn attr(&self, name: &str) -> rust_hdf5::Result<H5Attribute>;
    /// What stands in the way of the object-header attribute count.
    fn nattrs_hdr_gap() -> &'static str;
}

impl AttrSource for H5Dataset {
    fn attr_names(&self) -> rust_hdf5::Result<Vec<String>> {
        H5Dataset::attr_names(self)
    }
    fn attr(&self, name: &str) -> rust_hdf5::Result<H5Attribute> {
        H5Dataset::attr(self, name)
    }
    fn nattrs_hdr_gap() -> &'static str {
        "H5Dataset exposes no object-header attribute count"
    }
}

impl AttrSource for H5NamedDatatype {
    fn attr_names(&self) -> rust_hdf5::Result<Vec<String>> {
        H5NamedDatatype::attr_names(self)
    }
    fn attr(&self, name: &str) -> rust_hdf5::Result<H5Attribute> {
        H5NamedDatatype::attr(self, name)
    }
    fn nattrs_hdr_gap() -> &'static str {
        "H5NamedDatatype exposes no object-header attribute count"
    }
}

/// A committed (named) datatype: the type it commits, then its attributes.
fn dump_named_datatype(d: &mut Dump, path: &str, t: &H5NamedDatatype) {
    d.emit(&format!("{path}#kind"), "committed-datatype");

    let dtype = guarded(|| t.datatype()).ok().and_then(|r| r.ok());
    d.field(path, "dtype", || match &dtype {
        Some(dt) => Ok(canon_dtype(dt)),
        None => Err("H5NamedDatatype::datatype() failed or is unavailable".into()),
    });
    d.field(path, "strpad", || strpad_field(dtype.as_ref()));

    dump_object_attrs(d, path, t);
}

fn dump_object_attrs<T: AttrSource>(d: &mut Dump, path: &str, ds: &T) {
    let names = match guarded(|| ds.attr_names()) {
        Ok(Ok(mut n)) => {
            n.sort();
            n
        }
        Ok(Err(e)) => {
            d.emit(
                &format!("{path}#nattrs"),
                unsupported("nattrs", &oneline(e)),
            );
            return;
        }
        Err(p) => {
            d.emit(
                &format!("{path}#nattrs"),
                unsupported("nattrs", &format!("panic: {p}")),
            );
            return;
        }
    };
    d.emit(&format!("{path}#nattrs"), names.len().to_string());
    d.emit(
        &format!("{path}#nattrs_hdr"),
        unsupported("nattrs_hdr", T::nattrs_hdr_gap()),
    );
    d.emit(
        &format!("{path}#attrstore"),
        unsupported(
            "attrstore",
            "H5Dataset exposes no compact/dense attribute storage accessor",
        ),
    );

    for name in names {
        let key = format!("{path}@{name}");
        let attr = match guarded(|| ds.attr(&name)) {
            Ok(Ok(a)) => Some(a),
            _ => None,
        };
        let dtype = attr
            .as_ref()
            .and_then(|a| guarded(|| a.datatype()).ok())
            .and_then(|r| r.ok());

        d.field(&key, "dtype", || match &dtype {
            Some(dt) => Ok(canon_dtype(dt)),
            None => Err("H5Attribute::datatype() failed or is unavailable".into()),
        });

        d.field(&key, "strpad", || strpad_field(dtype.as_ref()));

        d.field(&key, "shape", || {
            Err("H5Attribute exposes no shape() accessor".into())
        });

        d.field(&key, "value", || {
            let a = attr.as_ref().ok_or("attr() did not return a handle")?;
            let dt = dtype
                .as_ref()
                .ok_or("datatype unavailable, so the value cannot be classified")?;
            if !renders_as_values(dt) {
                let bytes = guarded(|| a.read_raw())
                    .map_err(|p| format!("panic: {p}"))?
                    .map_err(oneline)?;
                return Ok(encode_raw(&bytes));
            }
            match dt {
                DatatypeMessage::VarLenString { .. } => {
                    let s = guarded(|| a.read_string())
                        .map_err(|p| format!("panic: {p}"))?
                        .map_err(oneline)?;
                    Ok(encode_vals(&[esc(&s)]))
                }
                DatatypeMessage::Reference { .. } => {
                    let refs = guarded(|| a.read_references())
                        .map_err(|p| format!("panic: {p}"))?
                        .map_err(oneline)?;
                    let vals: Vec<String> = refs.iter().map(render_ref).collect();
                    Ok(encode_vals(&vals))
                }
                other => Err(format!(
                    "no public reader for a {} attribute",
                    canon_dtype(other)
                )),
            }
        });
    }
}

/// Group attributes are read-only through `attr_names` / `attr_string`: there
/// is no `H5Group::attr()` returning a typed handle, so neither the datatype
/// nor the shape of a group attribute is observable.
fn dump_group_attrs(d: &mut Dump, path: &str, group: &H5Group) {
    let names = match guarded(|| group.attr_names()) {
        Ok(Ok(mut n)) => {
            n.sort();
            n
        }
        Ok(Err(e)) => {
            d.emit(
                &format!("{path}#nattrs"),
                unsupported("nattrs", &oneline(e)),
            );
            return;
        }
        Err(p) => {
            d.emit(
                &format!("{path}#nattrs"),
                unsupported("nattrs", &format!("panic: {p}")),
            );
            return;
        }
    };
    d.emit(&format!("{path}#nattrs"), names.len().to_string());
    d.emit(
        &format!("{path}#nattrs_hdr"),
        unsupported(
            "nattrs_hdr",
            "H5Group exposes no object-header attribute count",
        ),
    );
    d.emit(
        &format!("{path}#attrstore"),
        unsupported(
            "attrstore",
            "H5Group exposes no compact/dense attribute storage accessor",
        ),
    );

    for name in names {
        let key = format!("{path}@{name}");
        d.emit(
            &format!("{key}#dtype"),
            unsupported("dtype", "H5Group has no attr() handle in read mode"),
        );
        d.emit(
            &format!("{key}#strpad"),
            unsupported("strpad", "H5Group has no attr() handle in read mode"),
        );
        d.emit(
            &format!("{key}#shape"),
            unsupported("shape", "H5Group has no attr() handle in read mode"),
        );
        let detail = match guarded(|| group.attr_string(&name)) {
            Ok(Ok(s)) => format!(
                "H5Group has no attr() handle; attr_string() gave {}",
                esc(&s)
            ),
            Ok(Err(e)) => format!(
                "H5Group has no attr() handle; attr_string() failed: {}",
                oneline(e)
            ),
            Err(p) => format!("H5Group has no attr() handle; attr_string() panicked: {p}"),
        };
        d.emit(&format!("{key}#value"), unsupported("value", &detail));
    }
}

// ===========================================================================
// the write side
// ===========================================================================

/// A case the public API cannot express. Reported as its own verdict rather
/// than as a failure.
struct Unsupported(String);

type WriteResult = std::result::Result<(), Unsupported>;

fn unsup(why: &str) -> WriteResult {
    Err(Unsupported(why.to_string()))
}

fn be_bytes_ramp(width: usize, n: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * n as usize);
    for i in 0..n {
        let full = i.to_be_bytes();
        out.extend_from_slice(&full[8 - width..]);
    }
    out
}

fn fixed_string_bytes(strings: &[&str], width: usize, pad: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(strings.len() * width);
    for s in strings {
        let mut cell = vec![pad; width];
        let b = s.as_bytes();
        cell[..b.len()].copy_from_slice(b);
        out.extend_from_slice(&cell);
    }
    out
}

const STRINGS: [&str; 4] = ["alpha", "b", "", "delta12"];
const UNISTR: [&str; 4] = ["été", "日本", "", "café"];

/// f16 bit patterns for 0.0 .. 7.0 — the reference ramp under `<f2`.
const F16_RAMP: [u16; 8] = [
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
];

/// The eight f64 bit patterns `float_specials` writes.
const SPECIAL_BITS: [u64; 8] = [
    0x7FF8_0000_0000_0001,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x0000_0000_0000_0000,
];

fn f16_dtype() -> DatatypeMessage {
    DatatypeMessage::FloatingPoint {
        size: 2,
        byte_order: ByteOrder::LittleEndian,
        sign_location: 15,
        bit_offset: 0,
        bit_precision: 16,
        exponent_location: 10,
        exponent_size: 5,
        mantissa_location: 0,
        mantissa_size: 10,
        exponent_bias: 15,
    }
}

fn be_of(dt: DatatypeMessage) -> DatatypeMessage {
    match dt {
        DatatypeMessage::FixedPoint {
            size,
            signed,
            bit_offset,
            bit_precision,
            ..
        } => DatatypeMessage::FixedPoint {
            size,
            byte_order: ByteOrder::BigEndian,
            signed,
            bit_offset,
            bit_precision,
        },
        DatatypeMessage::FloatingPoint {
            size,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
            ..
        } => DatatypeMessage::FloatingPoint {
            size,
            byte_order: ByteOrder::BigEndian,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        },
        other => other,
    }
}

/// Create a dataset whose on-disk element type is `dt` and fill it from a raw
/// byte image. `u8` is only the carrier: `DatasetBuilder::create` sizes the
/// element from the override, so the carrier width is irrelevant here.
fn raw_typed(
    file: &H5File,
    name: &str,
    dt: DatatypeMessage,
    shape: &[usize],
    bytes: &[u8],
) -> rust_hdf5::Result<()> {
    let ds = file
        .new_dataset::<u8>()
        .datatype(dt)
        .shape(shape)
        .create(name)?;
    ds.write_raw_bytes(bytes)
}

fn write_case(case: &str, path: &str) -> rust_hdf5::Result<WriteResult> {
    // Every arm below mirrors the h5py generator of the same name in
    // oracle/cases.py, byte for byte.
    match case {
        // ---- integers ---------------------------------------------------
        "int_i8" => simple_ramp::<i8>(path, ramp_n::<i8>(8)),
        "int_u8" => simple_ramp::<u8>(path, ramp_n::<u8>(8)),
        "int_i16le" => simple_ramp::<i16>(path, ramp_n::<i16>(8)),
        "int_u16le" => simple_ramp::<u16>(path, ramp_n::<u16>(8)),
        "int_i32le" => simple_ramp::<i32>(path, ramp_n::<i32>(8)),
        "int_u32le" => simple_ramp::<u32>(path, ramp_n::<u32>(8)),
        "int_i64le" => simple_ramp::<i64>(path, ramp_n::<i64>(8)),
        "int_u64le" => simple_ramp::<u64>(path, ramp_n::<u64>(8)),
        "int_i16be" => be_ramp(path, DatatypeMessage::i16_type(), 2),
        // The one big-endian case written through the *typed* path: the
        // values handed over are host-order `i32`s, and the file has to hold
        // their big-endian image. Its siblings keep writing a pre-swapped
        // byte image, so both write styles stay covered.
        "int_i32be" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .datatype(be_of(DatatypeMessage::i32_type()))
                .shape([8usize])
                .create("data")?;
            ds.write_raw(&(0..8i32).collect::<Vec<_>>())?;
            file.close()?;
            Ok(Ok(()))
        }
        "int_u64be" => be_ramp(path, DatatypeMessage::u64_type(), 8),

        // ---- floats -----------------------------------------------------
        "float_f16le" => {
            let bytes: Vec<u8> = F16_RAMP.iter().flat_map(|b| b.to_le_bytes()).collect();
            let file = H5File::create(path)?;
            raw_typed(&file, "data", f16_dtype(), &[8], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "float_f32le" => simple_ramp::<f32>(path, (0..8).map(|i| i as f32).collect()),
        "float_f64le" => simple_ramp::<f64>(path, (0..8).map(|i| i as f64).collect()),
        "float_f64be" => {
            let bytes: Vec<u8> = (0..8u64).flat_map(|i| (i as f64).to_be_bytes()).collect();
            let file = H5File::create(path)?;
            raw_typed(
                &file,
                "data",
                be_of(DatatypeMessage::f64_type()),
                &[8],
                &bytes,
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        "float_specials" => {
            let vals: Vec<f64> = SPECIAL_BITS.iter().map(|b| f64::from_bits(*b)).collect();
            simple_ramp::<f64>(path, vals)
        }

        // ---- strings ----------------------------------------------------
        "str_fixed_ascii" => {
            let file = H5File::create(path)?;
            raw_typed(
                &file,
                "data",
                DatatypeMessage::FixedString {
                    size: 8,
                    padding: 1,
                    charset: 0,
                },
                &[4],
                &fixed_string_bytes(&STRINGS, 8, 0),
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        // The two pad rules the reference writes explicitly: the declared
        // rule and the bytes actually stored have to agree.
        "str_fixed_nullpad" | "str_fixed_spacepad" => {
            let (padding, pad_byte) = if case == "str_fixed_spacepad" {
                (2u8, b' ')
            } else {
                (1u8, 0u8)
            };
            let file = H5File::create(path)?;
            raw_typed(
                &file,
                "data",
                DatatypeMessage::FixedString {
                    size: 8,
                    padding,
                    charset: 0,
                },
                &[4],
                &fixed_string_bytes(&STRINGS, 8, pad_byte),
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        "str_fixed_utf8" => {
            let file = H5File::create(path)?;
            raw_typed(
                &file,
                "data",
                DatatypeMessage::FixedString {
                    size: 16,
                    padding: 1,
                    charset: 1,
                },
                &[4],
                &fixed_string_bytes(&UNISTR, 16, 0),
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        "str_vlen_ascii" => {
            let file = H5File::create(path)?;
            file.write_vlen_strings_ascii("data", &STRINGS)?;
            file.close()?;
            Ok(Ok(()))
        }
        "str_vlen_utf8" => {
            let file = H5File::create(path)?;
            file.write_vlen_strings("data", &UNISTR)?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- composite types --------------------------------------------
        "compound_simple" => {
            let dt = DatatypeMessage::Compound {
                size: 8,
                members: vec![
                    member("x", 0, DatatypeMessage::f32_type()),
                    member("y", 4, DatatypeMessage::f32_type()),
                ],
            };
            let mut bytes = Vec::new();
            for i in 0..4u32 {
                bytes.extend_from_slice(&(i as f32).to_le_bytes());
                bytes.extend_from_slice(&((100 + i) as f32).to_le_bytes());
            }
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[4], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "compound_nested" => {
            let inner = DatatypeMessage::Compound {
                size: 4,
                members: vec![
                    member("u", 0, DatatypeMessage::i16_type()),
                    member("v", 2, DatatypeMessage::i16_type()),
                ],
            };
            let dt = DatatypeMessage::Compound {
                size: 8,
                members: vec![
                    member("a", 0, DatatypeMessage::i32_type()),
                    member("inner", 4, inner),
                ],
            };
            let mut bytes = Vec::new();
            for i in 0..4i32 {
                bytes.extend_from_slice(&i.to_le_bytes());
                bytes.extend_from_slice(&((10 + i) as i16).to_le_bytes());
                bytes.extend_from_slice(&((20 + i) as i16).to_le_bytes());
            }
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[4], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "compound_with_string" => {
            let dt = DatatypeMessage::Compound {
                size: 12,
                members: vec![
                    member("id", 0, DatatypeMessage::i32_type()),
                    member(
                        "name",
                        4,
                        DatatypeMessage::FixedString {
                            size: 8,
                            padding: 1,
                            charset: 0,
                        },
                    ),
                ],
            };
            let names = ["aa", "bbb", "cccc"];
            let mut bytes = Vec::new();
            for (i, n) in names.iter().enumerate() {
                bytes.extend_from_slice(&(i as i32).to_le_bytes());
                bytes.extend_from_slice(&fixed_string_bytes(&[n], 8, 0));
            }
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[3], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "compound_padded" => {
            let dt = DatatypeMessage::Compound {
                size: 12,
                members: vec![
                    member("a", 0, DatatypeMessage::i16_type()),
                    member("b", 4, DatatypeMessage::i32_type()),
                ],
            };
            let mut bytes = Vec::new();
            for i in 0..4i32 {
                bytes.extend_from_slice(&(i as i16).to_le_bytes());
                bytes.extend_from_slice(&[0, 0]);
                bytes.extend_from_slice(&(1000 + i).to_le_bytes());
                bytes.extend_from_slice(&[0, 0, 0, 0]);
            }
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[4], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "compound_dtype_v4" => {
            // Same compound as `compound_simple`, written into a file whose
            // low libver bound is v1.12, which is what makes the datatype
            // message version 4. Chunked, matching the h5py generator.
            let dt = DatatypeMessage::Compound {
                size: 8,
                members: vec![
                    member("x", 0, DatatypeMessage::f32_type()),
                    member("y", 4, DatatypeMessage::f32_type()),
                ],
            };
            let mut bytes = Vec::new();
            for i in 0..4u32 {
                bytes.extend_from_slice(&(i as f32).to_le_bytes());
                bytes.extend_from_slice(&((100 + i) as f32).to_le_bytes());
            }
            let file = H5File::create(path)?;
            file.set_libver_bound(LibverBound::V112)?;
            let ds = file
                .new_dataset::<u8>()
                .datatype(dt)
                .shape([4usize])
                .chunk(&[4])
                .create("data")?;
            ds.write_raw_bytes(&bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "array_dtype" => {
            let dt = DatatypeMessage::Array {
                dims: vec![2, 3],
                base: Box::new(DatatypeMessage::f64_type()),
            };
            let bytes: Vec<u8> = (0..12u64).flat_map(|i| (i as f64).to_le_bytes()).collect();
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[2], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "enum_i8" => {
            let dt = DatatypeMessage::Enum {
                base: Box::new(DatatypeMessage::i8_type()),
                members: vec![
                    EnumMember {
                        name: "BLUE".into(),
                        value: vec![2],
                    },
                    EnumMember {
                        name: "GREEN".into(),
                        value: vec![1],
                    },
                    EnumMember {
                        name: "RED".into(),
                        value: vec![0],
                    },
                ],
            };
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[4], &[0u8, 1, 2, 1])?;
            file.close()?;
            Ok(Ok(()))
        }
        "enum_i32" => {
            let dt = DatatypeMessage::Enum {
                base: Box::new(DatatypeMessage::i32_type()),
                members: vec![
                    EnumMember {
                        name: "HIGH".into(),
                        value: 1000i32.to_le_bytes().to_vec(),
                    },
                    EnumMember {
                        name: "LOW".into(),
                        value: (-1i32).to_le_bytes().to_vec(),
                    },
                    EnumMember {
                        name: "MID".into(),
                        value: 0i32.to_le_bytes().to_vec(),
                    },
                ],
            };
            let mut bytes = Vec::new();
            for v in [-1i32, 0, 1000, 0] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let file = H5File::create(path)?;
            raw_typed(&file, "data", dt, &[4], &bytes)?;
            file.close()?;
            Ok(Ok(()))
        }
        "vlen_bytes" => {
            let file = H5File::create(path)?;
            let a: &[u8] = &[0, 1, 2];
            let b: &[u8] = &[];
            let c: &[u8] = &[255];
            file.write_vlen_bytes("data", &[a, b, c])?;
            file.close()?;
            Ok(Ok(()))
        }
        "vlen_numeric" => {
            let file = H5File::create(path)?;
            let a: &[i32] = &[1, 2, 3];
            let b: &[i32] = &[];
            let c: &[i32] = &[-7];
            file.write_vlen_numeric("data", &[a, b, c])?;
            file.close()?;
            Ok(Ok(()))
        }
        "named_datatype" => {
            let file = H5File::create(path)?;
            file.commit_datatype("t", DatatypeMessage::i32_type())?;
            // `data` describes its own type; `shared` points at /t.
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("data")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.new_dataset::<i32>()
                .committed_type("t")
                .shape([8usize])
                .create("shared")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.close()?;
            Ok(Ok(()))
        }
        "opaque" => {
            let file = H5File::create(path)?;
            let bytes: Vec<u8> = (0u8..12).collect();
            raw_typed(
                &file,
                "data",
                DatatypeMessage::Opaque {
                    size: 4,
                    tag: "raw4".into(),
                },
                &[3],
                &bytes,
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        "bitfield" => {
            let file = H5File::create(path)?;
            raw_typed(
                &file,
                "data",
                DatatypeMessage::BitField {
                    size: 1,
                    byte_order: ByteOrder::LittleEndian,
                    bit_offset: 0,
                    bit_precision: 8,
                },
                &[4],
                &[0x01, 0x80, 0xFF, 0x00],
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        "ref_object" => {
            let file = H5File::create(path)?;
            let target = file.new_dataset::<i32>().shape([8]).create("target")?;
            target.write_raw(&ramp_n::<i32>(8))?;
            file.create_group("grp")?;
            let refs = file
                .new_dataset::<u64>()
                .object_references()
                .shape([2])
                .create("refs")?;
            refs.write_object_references(&["/target", "/grp"])?;
            file.close()?;
            Ok(Ok(()))
        }
        "ref_region" => {
            let file = H5File::create(path)?;
            let target = file.new_dataset::<i32>().shape([8]).create("target")?;
            target.write_raw(&ramp_n::<i32>(8))?;
            let refs = file
                .new_dataset::<u64>()
                .region_references()
                .shape([2])
                .create("refs")?;
            // The two slices h5py's `t.regionref[0:3]` and `[4:8]` select.
            let slice = |start: u64, end: u64| Selection::Hyperslab {
                rank: 1,
                form: Hyperslab::Blocks(vec![HyperslabBlock {
                    start: vec![start],
                    end: vec![end],
                }]),
            };
            refs.write_region_references(&[("/target", slice(0, 2)), ("/target", slice(4, 7))])?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- layouts and chunk indexes ----------------------------------
        "layout_contiguous" => simple_ramp::<i32>(path, ramp_n::<i32>(16)),
        "layout_compact" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([16usize])
                .compact()
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(16))?;
            file.close()?;
            Ok(Ok(()))
        }
        "external_storage" => {
            // The reference names the raw file by its bare name, built from
            // this file's stem, so both resolve it against the directory the
            // HDF5 file is in. The bytes go through the dataset rather than
            // being written to the raw file directly: that is the external
            // write path under test.
            let raw = format!(
                "{}_ext.raw",
                std::path::Path::new(path)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([16usize])
                .external(&[(raw.as_str(), 0, 64)])
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(16))?;
            file.close()?;
            Ok(Ok(()))
        }
        "layout_contiguous_v108" => layout_at_libver(path, LibverBound::V18, None),
        "layout_contiguous_v110" => layout_at_libver(path, LibverBound::V110, None),
        "layout_chunked_v108" => layout_at_libver(path, LibverBound::V18, Some(&[16])),
        "layout_chunked_v110" => layout_at_libver(path, LibverBound::V110, Some(&[16])),
        "chunkidx_btree1" => {
            let file = H5File::create(path)?;
            file.set_libver_latest(false)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([8usize])
                .chunk(&[4])
                .max_shape(&[None])
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(8))?;
            file.close()?;
            Ok(Ok(()))
        }
        "chunkidx_single" => chunked_ramp(path, 8, &[8], &[Some(8)]),
        "chunkidx_farray" => chunked_ramp(path, 16, &[4], &[Some(16)]),
        "chunkidx_earray" => chunked_ramp(path, 16, &[4], &[None]),
        "chunkidx_earray_unlim_inner" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([4usize, 4])
                .chunk(&[2, 2])
                .max_shape(&[Some(4), None])
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(16))?;
            file.close()?;
            Ok(Ok(()))
        }
        "chunkidx_earray_dim1" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([4usize, 4])
                .chunk(&[2, 4])
                .max_shape(&[Some(4), None])
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(16))?;
            file.close()?;
            Ok(Ok(()))
        }
        "chunkidx_btree2" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([4usize, 4])
                .chunk(&[2, 2])
                .max_shape(&[None, None])
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(16))?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- filters ------------------------------------------------------
        "filter_deflate" => filtered(path, |b| b.deflate(6)),
        "filter_shuffle" => filtered(path, |b| b.shuffle()),
        "filter_deflate_shuffle" => filtered(path, |b| b.shuffle_deflate(6)),
        "filter_fletcher32" => filtered(path, |b| {
            b.filter_pipeline(FilterPipeline {
                filters: vec![Filter {
                    id: FILTER_FLETCHER32,
                    flags: 0,
                    cd_values: vec![],
                }],
            })
        }),
        "filter_scaleoffset" => filtered(path, |b| {
            // `filtered` writes 64 i32 elements in chunks of 16, and the
            // filter parameters carry that per-chunk element count.
            b.filter_pipeline(
                FilterPipeline::scaleoffset(&DatatypeMessage::i32_type(), 16, 0)
                    .expect("i32 is scale-offset filterable"),
            )
        }),

        // ---- fill values --------------------------------------------------
        "fill_default" => {
            let file = H5File::create(path)?;
            file.new_dataset::<i32>()
                .shape([16usize])
                .chunk(&[4])
                .create("data")?;
            file.close()?;
            Ok(Ok(()))
        }
        "fill_set_int" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([16usize])
                .chunk(&[4])
                .fill_value(-1i32)
                .create("data")?;
            ds.write_slice::<i32>(&[0], &[4], &ramp_n::<i32>(4))?;
            file.close()?;
            Ok(Ok(()))
        }
        "fill_set_float_nan" => {
            let file = H5File::create(path)?;
            file.new_dataset::<f64>()
                .shape([16usize])
                .chunk(&[4])
                .fill_value(f64::NAN)
                .create("data")?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- dataspaces ---------------------------------------------------
        "space_scalar" => {
            let file = H5File::create(path)?;
            let ds = file.new_dataset::<i32>().scalar().create("data")?;
            ds.write_raw(&[42i32])?;
            file.close()?;
            Ok(Ok(()))
        }
        "space_null" => {
            // Mirrors h5py.Empty("<i4"): the dataset holds no elements at all.
            let file = H5File::create(path)?;
            file.new_dataset::<i32>().null().create("data")?;
            file.close()?;
            Ok(Ok(()))
        }
        "space_zerosized" => {
            // Nothing to write: the h5py reference only creates the dataset.
            let file = H5File::create(path)?;
            file.new_dataset::<i32>().shape([0usize]).create("data")?;
            file.close()?;
            Ok(Ok(()))
        }
        "space_unlimited_resized" => {
            let file = H5File::create(path)?;
            let ds = file
                .new_dataset::<i32>()
                .shape([4usize])
                .chunk(&[4])
                .max_shape(&[None])
                .create("data")?;
            ds.write_raw(&ramp_n::<i32>(4))?;
            ds.extend(&[12])?;
            let tail: Vec<i32> = (0..8).map(|i| i + 100).collect();
            ds.write_slice::<i32>(&[4], &[8], &tail)?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- groups and links ---------------------------------------------
        "groups_nested" => {
            let file = H5File::create(path)?;
            let a = file.root_group().create_group("a")?;
            let b = a.create_group("b")?;
            b.create_group("c")?;
            b.new_dataset::<i32>()
                .shape([8usize])
                .create("leaf")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("top")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.close()?;
            Ok(Ok(()))
        }
        "link_hard" => {
            let file = H5File::create(path)?;
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("orig")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.root_group().link("alias", "/orig")?;
            file.close()?;
            Ok(Ok(()))
        }
        "link_soft" => {
            let file = H5File::create(path)?;
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("orig")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.create_soft_link("alias", "/orig")?;
            file.close()?;
            Ok(Ok(()))
        }
        "link_external" => {
            // The reference builds the sibling's name from this file's stem,
            // and stores the bare file name so the link resolves against the
            // directory holding the master.
            let target = std::path::Path::new(path).with_file_name(format!(
                "{}_ext.h5",
                std::path::Path::new(path)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));
            let ext = H5File::create(&target)?;
            ext.new_dataset::<i32>()
                .shape([8usize])
                .create("payload")?
                .write_raw(&ramp_n::<i32>(8))?;
            ext.close()?;

            let file = H5File::create(path)?;
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("orig")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.create_external_link(
                "ext",
                &target.file_name().unwrap_or_default().to_string_lossy(),
                "/payload",
            )?;
            file.close()?;
            Ok(Ok(()))
        }
        "link_external_read" => {
            // The whole payload lives in the sibling; the master holds only
            // links, two of which are deliberately dangling — a target object
            // that is not there and a target file that is not there.
            let target = std::path::Path::new(path).with_file_name(format!(
                "{}_data.h5",
                std::path::Path::new(path)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));
            let data = H5File::create(&target)?;
            data.new_dataset::<f64>()
                .shape([8usize])
                .create("top")?
                .write_raw(&(0..8).map(|i| i as f64).collect::<Vec<_>>())?;
            data.root_group()
                .create_group("deep")?
                .new_dataset::<i16>()
                .shape([8usize])
                .create("inner")?
                .write_raw(&ramp_n::<i16>(8))?;
            data.close()?;

            let name = target.file_name().unwrap_or_default().to_string_lossy();
            let file = H5File::create(path)?;
            file.create_external_link("direct", &name, "/top")?;
            file.create_external_link("nested", &name, "/deep/inner")?;
            file.create_external_link("gone_object", &name, "/absent")?;
            file.create_external_link("gone_file", "no_such_file.h5", "/top")?;
            file.close()?;
            Ok(Ok(()))
        }
        "links_dense" => {
            // The reference makes `g` with `track_order=True`, so the dense
            // storage it spills into carries a creation-order index beside
            // the name index.
            let file = H5File::create(path)?;
            file.set_track_order(true)?;
            let g = file.root_group().create_group("g")?;
            for i in 0..12i32 {
                g.new_dataset::<i32>()
                    .shape([1usize])
                    .create(&format!("d{i:02}"))?
                    .write_raw(&[i])?;
            }
            file.close()?;
            Ok(Ok(()))
        }
        "track_order" => {
            // h5py's `File(track_order=True)` is a file-creation property, so
            // it reaches the root group only; the three plain `create_group`
            // calls take h5py's default policy, and `g` turns it back on for
            // itself.
            let file = H5FileOptions::new().track_order(true).create(path)?;
            file.set_track_order(false)?;
            let root = file.root_group();
            for name in ["zebra", "apple", "mango"] {
                root.create_group(name)?;
            }
            for (i, key) in ["zeta", "alpha", "mu"].iter().enumerate() {
                file.set_attr_numeric(key, &(i as i32))?;
            }
            file.set_track_order(true)?;
            let g = root.create_group("g")?;
            g.new_dataset::<i32>()
                .shape([8usize])
                .create("data")?
                .write_raw(&ramp_n::<i32>(8))?;
            g.set_attr_numeric("second", &2i32)?;
            g.set_attr_numeric("first", &1i32)?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- attributes ----------------------------------------------------
        "attr_scalar_num" => {
            let file = H5File::create(path)?;
            let ds = file.new_dataset::<i32>().shape([8usize]).create("data")?;
            ds.write_raw(&ramp_n::<i32>(8))?;
            ds.new_attr::<f64>()
                .shape(())
                .create("gain")?
                .write_numeric(&2.5f64)?;
            ds.new_attr::<i32>()
                .shape(())
                .create("count")?
                .write_numeric(&7i32)?;
            file.close()?;
            Ok(Ok(()))
        }
        "attr_array_num" => {
            let file = H5File::create(path)?;
            let ds = file.new_dataset::<i32>().shape([8usize]).create("data")?;
            ds.write_raw(&ramp_n::<i32>(8))?;
            ds.new_attr::<i32>()
                .shape([4usize])
                .create("offsets")?
                .write_array(&ramp_n::<i32>(4))?;
            let matrix: Vec<f64> = (0..6).map(|i| i as f64).collect();
            ds.new_attr::<f64>()
                .shape([2usize, 3])
                .create("matrix")?
                .write_array(&matrix)?;
            file.close()?;
            Ok(Ok(()))
        }
        "attr_large" => {
            let file = H5File::create(path)?;
            let ds = file.new_dataset::<i32>().shape([8usize]).create("data")?;
            ds.write_raw(&ramp_n::<i32>(8))?;
            let big: Vec<i32> = (0..25600i32).collect();
            // An attribute this large has no compact form: the object header
            // message size field is a u16. The writer answers the way
            // `H5O__attr_create` does and spills the object's whole attribute
            // set to dense storage.
            ds.new_attr::<i32>()
                .shape([25600usize])
                .create("big")?
                .write_array(&big)?;
            file.close()?;
            Ok(Ok(()))
        }
        "attr_string" => {
            let file = H5File::create(path)?;
            let ds = file.new_dataset::<i32>().shape([8usize]).create("data")?;
            ds.write_raw(&ramp_n::<i32>(8))?;
            ds.new_attr::<VarLenUnicode>()
                .shape(())
                .create("units")?
                .write_string("volt")?;
            file.root_group()
                .create_group("g")?
                .set_attr_string("NX_class", "NXdetector")?;
            file.close()?;
            Ok(Ok(()))
        }
        "attrs_dense" => {
            let file = H5File::create(path)?;
            let ds = file.new_dataset::<i32>().shape([8usize]).create("data")?;
            ds.write_raw(&ramp_n::<i32>(8))?;
            for i in 0..12i32 {
                ds.new_attr::<i32>()
                    .shape(())
                    .create(&format!("a{i:02}"))?
                    .write_numeric(&i)?;
            }
            file.close()?;
            Ok(Ok(()))
        }
        "attrs_dense_group" => {
            let file = H5File::create(path)?;
            let g = file.root_group().create_group("g")?;
            for i in 0..12i32 {
                g.set_attr_numeric(&format!("g{i:02}"), &i)?;
            }
            for i in 0..12i32 {
                file.set_attr_numeric(&format!("r{i:02}"), &i)?;
            }
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("data")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.close()?;
            Ok(Ok(()))
        }
        "attr_on_root" => {
            let file = H5File::create(path)?;
            file.set_attr_string("title", "root")?;
            file.set_attr_numeric("version", &3i64)?;
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("data")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.close()?;
            Ok(Ok(()))
        }

        // ---- library version bounds -----------------------------------------
        "libver_earliest" => libver_case(path, LibverBound::Earliest),
        "libver_v108" => libver_case(path, LibverBound::V18),
        "libver_v110" => libver_case(path, LibverBound::V110),
        "libver_latest" => libver_case(path, LibverBound::V200),
        "userblock" => {
            let file = H5File::options().userblock(512).create(path)?;
            file.new_dataset::<i32>()
                .shape([8usize])
                .create("data")?
                .write_raw(&ramp_n::<i32>(8))?;
            file.root_group().create_group("g")?;
            file.close()?;
            // The h5py arm fills the block with a shebang line afterwards, as
            // an application that keeps a script there would; the block is the
            // application's, so this is a plain write to the front of the file.
            let prefix = b"#!/bin/sh\n# userblock\n";
            let mut block = prefix.to_vec();
            block.resize(511, b'#');
            block.push(b'\n');
            let mut fh = std::fs::OpenOptions::new().write(true).open(path)?;
            std::io::Write::write_all(&mut fh, &block)?;
            Ok(Ok(()))
        }

        // ---- SWMR and bulk ---------------------------------------------------
        "swmr_created" => {
            use rust_hdf5::swmr::SwmrFileWriter;
            let mut w = SwmrFileWriter::create(path)?;
            let idx = w.create_streaming_dataset::<f32>("stream", &[4u64])?;
            w.start_swmr()?;
            for i in 0..8u32 {
                let frame: Vec<u8> = (0..4u32)
                    .flat_map(|j| ((i * 4 + j) as f32).to_le_bytes())
                    .collect();
                w.append_frame(idx, &frame)?;
            }
            w.close()?;
            Ok(Ok(()))
        }
        "large_multi_mb" => {
            let file = H5File::create(path)?;
            let data: Vec<f64> = (0..512u32 * 512).map(|i| i as f64).collect();
            let ds = file
                .new_dataset::<f64>()
                .shape([512usize, 512])
                .chunk(&[64, 512])
                .create("big")?;
            ds.write_raw(&data)?;
            file.close()?;
            Ok(Ok(()))
        }

        _ => Ok(unsup(&format!("no rust writer arm for case '{case}'"))),
    }
}

fn member(name: &str, offset: u32, datatype: DatatypeMessage) -> CompoundMember {
    CompoundMember {
        name: name.to_string(),
        offset,
        datatype,
    }
}

/// `0, 1, .. n-1` in `T` — the ramp every reference generator writes. Only
/// called with `n <= 16`, well inside every integer width used here.
fn ramp_n<T: TryFrom<u8>>(n: u8) -> Vec<T>
where
    <T as TryFrom<u8>>::Error: std::fmt::Debug,
{
    (0..n)
        .map(|i| T::try_from(i).expect("ramp index fits the element type"))
        .collect()
}

fn simple_ramp<T: rust_hdf5::H5Type>(path: &str, data: Vec<T>) -> rust_hdf5::Result<WriteResult> {
    let file = H5File::create(path)?;
    let ds = file.new_dataset::<T>().shape([data.len()]).create("data")?;
    ds.write_raw(&data)?;
    file.close()?;
    Ok(Ok(()))
}

fn be_ramp(path: &str, le: DatatypeMessage, width: usize) -> rust_hdf5::Result<WriteResult> {
    let file = H5File::create(path)?;
    raw_typed(&file, "data", be_of(le), &[8], &be_bytes_ramp(width, 8))?;
    file.close()?;
    Ok(Ok(()))
}

fn chunked_ramp(
    path: &str,
    n: usize,
    chunk: &[usize],
    max: &[Option<usize>],
) -> rust_hdf5::Result<WriteResult> {
    let file = H5File::create(path)?;
    let ds = file
        .new_dataset::<i32>()
        .shape([n])
        .chunk(chunk)
        .max_shape(max)
        .create("data")?;
    ds.write_raw(&ramp_n::<i32>(n as u8))?;
    file.close()?;
    Ok(Ok(()))
}

fn filtered(
    path: &str,
    configure: impl FnOnce(
        rust_hdf5::dataset::DatasetBuilder<i32>,
    ) -> rust_hdf5::dataset::DatasetBuilder<i32>,
) -> rust_hdf5::Result<WriteResult> {
    let file = H5File::create(path)?;
    let builder = file.new_dataset::<i32>().shape([64usize]).chunk(&[16]);
    let ds = configure(builder).create("data")?;
    let data: Vec<i32> = (0..64).collect();
    ds.write_raw(&data)?;
    file.close()?;
    Ok(Ok(()))
}

fn libver_case(path: &str, libver: LibverBound) -> rust_hdf5::Result<WriteResult> {
    let file = H5File::options().libver(libver).create(path)?;
    file.new_dataset::<i32>()
        .shape([8usize])
        .create("data")?
        .write_raw(&ramp_n::<i32>(8))?;
    file.root_group().create_group("g")?;
    file.close()?;
    Ok(Ok(()))
}

/// A single 16-element i32 ramp under an explicit libver bound: contiguous
/// when `chunk` is `None`, one whole-dataset chunk when it is `Some`.
fn layout_at_libver(
    path: &str,
    libver: LibverBound,
    chunk: Option<&[usize]>,
) -> rust_hdf5::Result<WriteResult> {
    let file = H5File::options().libver(libver).create(path)?;
    let mut builder = file.new_dataset::<i32>().shape([16usize]);
    if let Some(chunk) = chunk {
        builder = builder.chunk(chunk);
    }
    builder.create("data")?.write_raw(&ramp_n::<i32>(16))?;
    file.close()?;
    Ok(Ok(()))
}

// ===========================================================================

fn usage() -> i32 {
    eprintln!("usage: oracle_probe dump <file.h5>");
    eprintln!("       oracle_probe write <case> <file.h5>");
    64
}

fn main() {
    // A panic message on stderr would be noise; `guarded` captures the payload
    // and reports it in-band, so silence the default hook.
    std::panic::set_hook(Box::new(|_| {}));

    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("dump") if args.len() == 3 => match dump_file(&args[2]) {
            Ok(text) => {
                print!("{text}");
                0
            }
            Err(e) => {
                println!("!open-error\t{e}");
                1
            }
        },
        Some("write") if args.len() == 4 => {
            let outcome = guarded(|| write_case(&args[2], &args[3]));
            match outcome {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(Unsupported(why)))) => {
                    println!("UNSUPPORTED-API: {why}");
                    2
                }
                Ok(Err(e)) => {
                    println!("WRITE-ERROR: {}", oneline(e));
                    1
                }
                Err(p) => {
                    println!("WRITE-PANIC: {p}");
                    1
                }
            }
        }
        _ => usage(),
    };
    std::process::exit(code);
}
