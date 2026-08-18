//! Reference elements and the dataspace selections region references carry.
//!
//! Two element layouts exist for the pre-1.12 reference kinds, both written by
//! h5py 3.x today (`H5Tref.c`):
//!
//! ```text
//! H5R_OBJECT1          sizeof_addr bytes   the target's object header address
//! H5R_DATASET_REGION1  sizeof_addr + 4     a global-heap id: collection
//!                                          address then a u32 object index
//! ```
//!
//! The heap object a region reference points at is the target's object header
//! address followed by the serialized selection
//! (`H5R__encode_token_region_compat`), whose wire format lives in the `H5S`
//! serializers (`H5S__hyper_serialize`, `H5S__point_serialize`,
//! `H5S__all_serialize`, `H5S__none_serialize`).
//!
//! The 1.12 kinds — `H5R_OBJECT2`, `H5R_DATASET_REGION2` and `H5R_ATTR`, all
//! written as `H5T_STD_REF` — share one element layout instead
//! (`H5T__ref_disk_getsize`):
//!
//! ```text
//! type (1) | flags (1) | encoded reference           an H5R_OBJECT2 with no
//!                                                    external file, stored
//!                                                    inline
//! type (1) | flags (1) | size (4) | heap id          everything else, whose
//!                                                    encoded reference is a
//!                                                    global-heap blob
//! ```
//!
//! and the encoded reference itself is `H5R__encode`: a token (its length, then
//! the target's object header address), then — when the flags carry
//! `H5R_IS_EXTERNAL` — the name of the file the target lives in, then the
//! serialized selection for a region and the attribute name for an attribute
//! reference.

use crate::format::bytes::{read_le_addr, read_le_uint};
use crate::format::messages::datatype::ReferenceKind;
use crate::format::selection::Selection;
use crate::format::{FormatContext, FormatError, FormatResult, UNDEF_ADDR};

/// The address a reference element leads with, or `None` when it names
/// nothing.
///
/// Both element layouts start with a file address, and both spell "no target"
/// the same two ways: the all-ones undefined address `H5F_addr_decode`
/// produces, and 0 — the superblock's own address, so never an object header,
/// and what an unwritten (fill-value) element holds. `H5R__decode_heap`
/// rejects both together (`!H5_addr_defined(hobjid.addr) || hobjid.addr == 0`),
/// so this crate applies the one rule to both kinds rather than per element
/// layout.
fn target_address(elem: &[u8], sizeof_addr: usize) -> Option<u64> {
    match read_le_addr(elem, sizeof_addr) {
        0 | UNDEF_ADDR => None,
        addr => Some(addr),
    }
}

/// The address a `H5R_OBJECT1` element names, or `None` for a null reference.
pub fn decode_object_element(elem: &[u8], ctx: &FormatContext) -> FormatResult<Option<u64>> {
    let sa = ctx.sizeof_addr as usize;
    if elem.len() < sa {
        return Err(FormatError::BufferTooShort {
            needed: sa,
            available: elem.len(),
        });
    }
    Ok(target_address(elem, sa))
}

/// The `(collection address, object index)` a `H5R_DATASET_REGION1` element
/// names, or `None` when the element is a null reference.
pub fn decode_region_element(elem: &[u8], ctx: &FormatContext) -> FormatResult<Option<(u64, u32)>> {
    let sa = ctx.sizeof_addr as usize;
    if elem.len() < sa + 4 {
        return Err(FormatError::BufferTooShort {
            needed: sa + 4,
            available: elem.len(),
        });
    }
    let Some(addr) = target_address(elem, sa) else {
        return Ok(None);
    };
    let idx = u32::from_le_bytes([elem[sa], elem[sa + 1], elem[sa + 2], elem[sa + 3]]);
    Ok(Some((addr, idx)))
}

/// Split a region reference's heap object into the target's object header
/// address and the selection over it.
pub fn decode_region_heap_object(
    data: &[u8],
    ctx: &FormatContext,
) -> FormatResult<(u64, Selection)> {
    let sa = ctx.sizeof_addr as usize;
    if data.len() < sa {
        return Err(FormatError::BufferTooShort {
            needed: sa,
            available: data.len(),
        });
    }
    let addr = read_le_addr(data, sa);
    let (selection, _) = Selection::decode(&data[sa..])?;
    Ok((addr, selection))
}

/// `H5R_IS_EXTERNAL`: the encoded reference carries the name of the file the
/// target lives in, after the token.
const REVISED_FLAG_EXTERNAL: u8 = 0x01;

/// The two bytes every 1.12 element leads with, `H5R_ENCODE_HEADER_SIZE`.
const REVISED_HEADER: usize = 2;

/// Where a 1.12 element keeps its encoded reference.
///
/// `H5T__ref_disk_getsize` makes the same split: an `H5R_OBJECT2` naming an
/// object in this file is short enough to sit in the element, and every other
/// element holds a global-heap blob id instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisedElement<'a> {
    /// An element naming nothing: reference type 0 over a nil blob id.
    Null,
    /// The encoded reference, less its two-byte header, stored in the element.
    /// Never external: `H5T__ref_disk_getsize` takes the direct-copy arm only
    /// for an `H5R_OBJECT2` whose flags are clear (H5Tref.c:890).
    Inline {
        /// The kind the element's own type byte names.
        kind: ReferenceKind,
        /// The encoded reference from the token onwards.
        body: &'a [u8],
    },
    /// The encoded reference lives in a global-heap collection.
    Heap {
        /// The kind the element's own type byte names.
        kind: ReferenceKind,
        /// Whether the element's flags carry `H5R_IS_EXTERNAL`, in which case
        /// the encoded reference names the file the target lives in.
        external: bool,
        /// Address of the collection holding the blob.
        collection: u64,
        /// Index of the blob's object within that collection.
        index: u32,
    },
}

/// Split a 1.12 reference element into its kind and the encoded reference.
///
/// The element's type byte is the authority on the kind, not the datatype
/// message: libhdf5 stores every `H5T_STD_REF` as `H5R_OBJECT2` and lets each
/// element say what it actually holds.
pub fn decode_revised_element<'a>(
    elem: &'a [u8],
    ctx: &FormatContext,
) -> FormatResult<RevisedElement<'a>> {
    let sa = ctx.sizeof_addr as usize;
    let heap_id = REVISED_HEADER + 4;
    if elem.len() < REVISED_HEADER {
        return Err(FormatError::BufferTooShort {
            needed: REVISED_HEADER,
            available: elem.len(),
        });
    }
    let (code, flags) = (elem[0], elem[1]);

    // Reference type 0 is `H5R_BADTYPE`; `H5T__ref_disk_isnull` reads such an
    // element as null when — and only when — the blob id it carries is the nil
    // one (a zero collection address), which is what an unwritten element
    // holds.
    if code == 0 {
        if elem.len() < heap_id + sa {
            return Err(FormatError::BufferTooShort {
                needed: heap_id + sa,
                available: elem.len(),
            });
        }
        return match read_le_addr(&elem[heap_id..], sa) {
            0 => Ok(RevisedElement::Null),
            addr => Err(FormatError::InvalidData(format!(
                "reference type 0 over a blob at {addr:#x}, which is not the nil id a null \
                 reference carries"
            ))),
        };
    }

    let kind = ReferenceKind::from_code(code)
        .filter(|k| k.is_revised())
        .ok_or_else(|| {
            FormatError::InvalidData(format!("reference type {code} in a 1.12 element"))
        })?;

    let external = flags & REVISED_FLAG_EXTERNAL != 0;

    if !external && kind == ReferenceKind::Object2 {
        return Ok(RevisedElement::Inline {
            kind,
            body: &elem[REVISED_HEADER..],
        });
    }

    if elem.len() < heap_id + sa + 4 {
        return Err(FormatError::BufferTooShort {
            needed: heap_id + sa + 4,
            available: elem.len(),
        });
    }
    let collection = read_le_addr(&elem[heap_id..], sa);
    let index = read_le_uint(&elem[heap_id + sa..heap_id + sa + 4], 4) as u32;
    Ok(RevisedElement::Heap {
        kind,
        external,
        collection,
        index,
    })
}

/// What one reference element holds — the encode-side mirror of
/// [`RevisedElement`], which is what a reader splits an element back into.
///
/// The variant carries everything its own layout needs, so
/// [`encode_reference_element`] is total: there is no kind it cannot write,
/// and no way to hand it a body its layout has no room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceElementImage {
    /// `H5R_OBJECT1`: the element is the target's object header address and
    /// nothing else.
    Legacy(u64),
    /// `H5R_OBJECT2` naming an object in this file: short enough that the
    /// whole encoded reference sits in the element
    /// (`H5T__ref_disk_getsize`'s direct-copy arm).
    Inline(u64),
    /// Every other 1.12 element: the encoded reference lives in a global-heap
    /// blob and the element carries the blob's byte count and its id
    /// (`H5T__ref_disk_write`, which copies the two-byte header across, then
    /// the size, then what `H5VL__native_blob_put` leaves).
    Blob {
        /// The kind the element's own type byte names.
        kind: ReferenceKind,
        /// The blob's byte count: the encoded reference less its two-byte
        /// header, which the element keeps instead.
        size: u32,
        /// Address of the collection holding the blob.
        collection: u64,
        /// Index of the blob's object within that collection.
        index: u32,
    },
}

/// The image one reference element has, at the `size` its datatype declares.
///
/// The single owner of reference-element encoding, so every element layout
/// this module documents is written where it is read. Anything past what the
/// layout needs is left zero, which is where libhdf5 leaves it too: it sizes
/// every element for the largest reference the dataset may hold and writes
/// only as much of it as the reference uses.
pub fn encode_reference_element(
    image: &ReferenceElementImage,
    size: usize,
    ctx: &FormatContext,
) -> FormatResult<Vec<u8>> {
    let sa = ctx.sizeof_addr as usize;
    let mut elem = vec![0u8; size];
    let (what, needed) = match image {
        ReferenceElementImage::Legacy(_) => ("an H5R_OBJECT1", sa),
        ReferenceElementImage::Inline(_) => ("an H5R_OBJECT2", REVISED_HEADER + 1 + sa),
        ReferenceElementImage::Blob { kind, .. } => (
            match kind {
                ReferenceKind::DatasetRegion2 => "an H5R_DATASET_REGION2",
                ReferenceKind::Attr => "an H5R_ATTR",
                _ => "a blob-backed",
            },
            REVISED_HEADER + 4 + sa + 4,
        ),
    };
    if needed > size {
        return Err(FormatError::InvalidData(format!(
            "{what} element needs {needed} bytes but its datatype declares {size}"
        )));
    }
    match *image {
        ReferenceElementImage::Legacy(address) => {
            elem[..sa].copy_from_slice(&address.to_le_bytes()[..sa]);
        }
        // Type, flags, then the encoded reference inline: the token's length
        // and the token itself (`H5R__encode_obj_token`).
        ReferenceElementImage::Inline(address) => {
            elem[0] = ReferenceKind::Object2.code();
            elem[1] = 0;
            elem[2] = sa as u8;
            let at = REVISED_HEADER + 1;
            elem[at..at + sa].copy_from_slice(&address.to_le_bytes()[..sa]);
        }
        ReferenceElementImage::Blob {
            kind,
            size: blob_size,
            collection,
            index,
        } => {
            elem[0] = kind.code();
            elem[1] = 0;
            elem[REVISED_HEADER..REVISED_HEADER + 4].copy_from_slice(&blob_size.to_le_bytes());
            let at = REVISED_HEADER + 4;
            elem[at..at + sa].copy_from_slice(&collection.to_le_bytes()[..sa]);
            elem[at + sa..at + sa + 4].copy_from_slice(&index.to_le_bytes());
        }
    }
    Ok(elem)
}

/// The encoded reference a 1.12 blob holds, less the two-byte header the
/// element keeps — `H5R__encode` from the token onwards, which is exactly
/// what [`decode_revised_body`] reads back.
///
/// The token is written as `address`; a caller that does not know the
/// target's object header address yet passes 0 and patches those `sa` bytes
/// at offset [`REVISED_BLOB_TOKEN_OFFSET`] once it does.
///
/// `extent_rank` is the rank of the dataspace the reference's selection is
/// over, which is what `H5R__encode_region` encodes and where it takes it
/// from — `H5S_get_simple_extent_ndims` of the space, not the serialized
/// selection, which is why an `H5S_SEL_ALL` region still says a rank. It is 0
/// for the two kinds that carry no selection.
pub fn encode_revised_blob(
    address: u64,
    target: &ReferenceTarget,
    extent_rank: usize,
    ctx: &FormatContext,
) -> FormatResult<Vec<u8>> {
    let sa = ctx.sizeof_addr as usize;
    let mut blob = Vec::with_capacity(1 + sa + 16);
    blob.push(sa as u8);
    blob.extend_from_slice(&address.to_le_bytes()[..sa]);
    match target {
        ReferenceTarget::Object => {}
        // `H5R__encode_region`: the serialized selection's length, then the
        // extent's rank, then the selection.
        ReferenceTarget::Region(selection) => {
            let bytes = selection.encode()?;
            blob.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            blob.extend_from_slice(&(extent_rank as u32).to_le_bytes());
            blob.extend_from_slice(&bytes);
        }
        // `H5R__encode_string`: a 16-bit length, then the unterminated name.
        ReferenceTarget::Attribute(name) => {
            let len = u16::try_from(name.len()).map_err(|_| {
                FormatError::InvalidData(format!(
                    "attribute name of {} bytes does not fit a reference's 16-bit length",
                    name.len()
                ))
            })?;
            blob.extend_from_slice(&len.to_le_bytes());
            blob.extend_from_slice(name.as_bytes());
        }
    }
    Ok(blob)
}

/// Where the object token sits inside a blob [`encode_revised_blob`] built:
/// behind the one byte that gives its length.
pub const REVISED_BLOB_TOKEN_OFFSET: usize = 1;

/// What a reference names beyond the object its token points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceTarget {
    /// The object itself.
    Object,
    /// A selection over the target dataset.
    Region(Selection),
    /// An attribute of the target, by name.
    Attribute(String),
}

/// Everything `H5R__decode` recovers from an encoded reference: where the
/// target is, and what is named there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedReference {
    /// Object header address the token names.
    pub address: u64,
    /// The file the target lives in, as the reference records it — the name
    /// that file was open under when the reference was written
    /// (`H5F_get_name`, not a canonical path). `None` when the flags do not
    /// carry `H5R_IS_EXTERNAL`, which means the file holding the reference.
    pub file: Option<String>,
    /// What the reference names at that address.
    pub target: ReferenceTarget,
}

/// Decode an encoded 1.12 reference — `H5R__decode` from the token onwards —
/// into where its target is and what it names there, or `None` when the token
/// names no object.
///
/// `external` is the element's `H5R_IS_EXTERNAL` flag, which decides whether a
/// file name sits between the token and the kind's own payload; the flag lives
/// in the element, not in the encoded reference, so it has to be handed in
/// (`H5R__decode`, H5Rint.c:991-999).
pub fn decode_revised_body(
    kind: ReferenceKind,
    external: bool,
    body: &[u8],
    ctx: &FormatContext,
) -> FormatResult<Option<DecodedReference>> {
    let sa = ctx.sizeof_addr as usize;
    let mut r = Cursor::new(body);

    // `H5R__decode_obj_token` stores the token's length ahead of it. The
    // native VOL's token is the object header address (`H5VL_native_addr_to_
    // token`), so a file whose tokens are some other width came from a
    // connector this crate cannot follow.
    let token_size = r.u8()? as usize;
    if token_size != sa {
        return Err(FormatError::UnsupportedFeature(format!(
            "object tokens {token_size} bytes wide, not the {sa}-byte file addresses the \
             native format uses"
        )));
    }
    let token = r.take(token_size)?;
    let Some(address) = target_address(token, sa) else {
        return Ok(None);
    };

    // `H5R__encode` writes the file name straight after the token and before
    // the kind's own payload (H5Rint.c:903-905).
    let file = if external {
        Some(decode_string(&mut r)?)
    } else {
        None
    };

    let target = match kind {
        ReferenceKind::Object2 => ReferenceTarget::Object,
        ReferenceKind::DatasetRegion2 => {
            // `H5R__encode_region` prefixes the serialized selection with its
            // length and the extent's rank; the selection carries the rank
            // again, so only the length is needed to bound it.
            let len = r.u32()? as usize;
            let _rank = r.u32()?;
            ReferenceTarget::Region(Selection::decode(r.take(len)?)?.0)
        }
        ReferenceKind::Attr => ReferenceTarget::Attribute(decode_string(&mut r)?),
        ReferenceKind::Object1 | ReferenceKind::DatasetRegion1 => {
            return Err(FormatError::InvalidData(format!(
                "{kind:?} is not a 1.12 encoded reference"
            )))
        }
    };
    Ok(Some(DecodedReference {
        address,
        file,
        target,
    }))
}

/// One `H5R__encode_string` field: a 16-bit length, then that many unterminated
/// bytes. Both a reference's file name and an attribute reference's name are
/// written this way, so both are read back through here.
fn decode_string(r: &mut Cursor<'_>) -> FormatResult<String> {
    let len = r.u16()? as usize;
    let bytes = r.take(len)?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| FormatError::InvalidData("a string in a reference is not UTF-8".into()))
}

/// One reference element, decoded and resolved against the file it came from.
///
/// `path` is the target's absolute path when the file's link structure names
/// it, and `None` when nothing in the traversed structure points at that
/// address — a reference into an untraversed part of the file, or a stale one
/// left by a deletion. The address is reported either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reference {
    /// An element naming no object: the undefined address libhdf5 writes for
    /// an unset object reference, or a zeroed region-reference heap id.
    Null,
    /// A whole object — `H5R_OBJECT1` or `H5R_OBJECT2`.
    Object {
        /// Object header address of the target.
        address: u64,
        /// The file the target lives in; see [`Reference::file`].
        file: Option<String>,
        /// Absolute path of the target.
        path: Option<String>,
    },
    /// A dataset plus a selection over it — `H5R_DATASET_REGION1` or
    /// `H5R_DATASET_REGION2`.
    Region {
        /// Object header address of the target dataset.
        address: u64,
        /// The file the target dataset lives in; see [`Reference::file`].
        file: Option<String>,
        /// Absolute path of the target dataset.
        path: Option<String>,
        /// The selection the reference carries.
        selection: Selection,
    },
    /// `H5R_ATTR`: one attribute of an object, by name. Only the 1.12
    /// encodings can express it.
    Attr {
        /// Object header address of the object the attribute belongs to.
        address: u64,
        /// The file that object lives in; see [`Reference::file`].
        file: Option<String>,
        /// Absolute path of that object.
        path: Option<String>,
        /// Name of the attribute.
        name: String,
    },
}

impl Reference {
    /// The file the target lives in, for a reference that names another file
    /// — `H5Rget_file_name` on an un-opened external reference. `None` means
    /// the file holding the reference, which is every reference libhdf5 can
    /// write without the `H5R_IS_EXTERNAL` flag.
    ///
    /// The name is the one the target file was open under when the reference
    /// was written, recorded verbatim, and [`path`](Self::path) is a path
    /// inside *that* file.
    pub fn file(&self) -> Option<&str> {
        match self {
            Self::Null => None,
            Self::Object { file, .. } | Self::Region { file, .. } | Self::Attr { file, .. } => {
                file.as_deref()
            }
        }
    }

    /// The target's absolute path, when the file names it.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Null => None,
            Self::Object { path, .. } | Self::Region { path, .. } | Self::Attr { path, .. } => {
                path.as_deref()
            }
        }
    }

    /// The target's object header address, or `None` for a null reference.
    pub fn address(&self) -> Option<u64> {
        match self {
            Self::Null => None,
            Self::Object { address, .. }
            | Self::Region { address, .. }
            | Self::Attr { address, .. } => Some(*address),
        }
    }

    /// The attribute an attribute reference names; `None` for the other kinds.
    pub fn attribute_name(&self) -> Option<&str> {
        match self {
            Self::Attr { name, .. } => Some(name),
            _ => None,
        }
    }

    /// The selection a region reference carries; `None` for the other kinds.
    pub fn selection(&self) -> Option<&Selection> {
        match self {
            Self::Region { selection, .. } => Some(selection),
            _ => None,
        }
    }

    /// The inclusive bounding box of a region reference's selection —
    /// `H5Sget_select_bounds` on the dereferenced region.
    pub fn bounds(&self) -> Option<(Vec<u64>, Vec<u64>)> {
        self.selection()?.bounds()
    }

    /// Whether this element names no object.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

/// Little-endian cursor over a serialized selection.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> FormatResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(FormatError::BufferTooShort {
            needed: usize::MAX,
            available: self.buf.len(),
        })?;
        if end > self.buf.len() {
            return Err(FormatError::BufferTooShort {
                needed: end,
                available: self.buf.len(),
            });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> FormatResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> FormatResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> FormatResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::format::selection::{Hyperslab, HyperslabBlock, PointSelection};

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    /// The heap object libhdf5 1.14.6 writes for `dset.regionref[0:3]` on a
    /// 1-D 8-element dataset: the target's object header address followed by a
    /// version-1 hyperslab naming one block, [0]-[2].
    const REGION_HEAP_OBJECT: [u8; 40] = [
        0x20, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // target address 0x320
        0x02, 0x00, 0x00, 0x00, // H5S_SEL_HYPERSLABS
        0x01, 0x00, 0x00, 0x00, // version 1
        0x00, 0x00, 0x00, 0x00, // padding
        0x10, 0x00, 0x00, 0x00, // length 16
        0x01, 0x00, 0x00, 0x00, // rank 1
        0x01, 0x00, 0x00, 0x00, // one block
        0x00, 0x00, 0x00, 0x00, // start [0]
        0x02, 0x00, 0x00, 0x00, // end [2]
    ];

    #[test]
    fn region_heap_object_from_libhdf5_decodes() {
        let (addr, selection) = decode_region_heap_object(&REGION_HEAP_OBJECT, &ctx()).unwrap();
        assert_eq!(addr, 0x320);
        assert_eq!(
            selection,
            Selection::Hyperslab {
                rank: 1,
                form: Hyperslab::Blocks(vec![HyperslabBlock {
                    start: vec![0],
                    end: vec![2],
                }]),
            }
        );
        assert_eq!(selection.bounds(), Some((vec![0], vec![2])));
    }

    #[test]
    fn object_and_region_elements_report_null() {
        assert_eq!(
            decode_object_element(&0x320u64.to_le_bytes(), &ctx()).unwrap(),
            Some(0x320)
        );
        assert_eq!(
            decode_object_element(&[0xFF; 8], &ctx()).unwrap(),
            None,
            "an undefined address is a null reference"
        );
        assert_eq!(
            decode_object_element(&[0; 8], &ctx()).unwrap(),
            None,
            "so is address 0, which h5py writes for an unset element"
        );
        let mut elem = [0u8; 12];
        elem[..8].copy_from_slice(&0x820u64.to_le_bytes());
        elem[8..].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode_region_element(&elem, &ctx()).unwrap(),
            Some((0x820, 2))
        );
        assert_eq!(
            decode_region_element(&[0u8; 12], &ctx()).unwrap(),
            None,
            "a zeroed element carries no heap id"
        );
    }

    // The four element/blob captures below come from a file libhdf5 1.14.6
    // wrote with `H5F_LIBVER_V112` as its low bound
    // (`tests/fixtures/gen_revised_refs.c latest`), the combination that puts
    // the newest selection encodings inside a reference: `matrix` is a 4x6
    // dataset at object header address 0xC3, the region references select the
    // hyperslab (1,2)-(2,4) and the points (0,1) and (3,5), and the attribute
    // reference names `note`.

    /// `H5R_OBJECT2`: type, flags, then the token inline.
    const OBJ2_ELEMENT: [u8; 18] = [
        0x02, 0x00, 0x08, 0xC3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00,
    ];

    /// `H5R_DATASET_REGION2`: type, flags, blob size, then the heap id.
    const REGION2_ELEMENT: [u8; 18] = [
        0x03, 0x00, 0x39, 0x00, 0x00, 0x00, 0xA8, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00,
    ];

    /// `H5R_ATTR`, whose blob ends in the attribute name.
    const ATTR_ELEMENT: [u8; 18] = [
        0x04, 0x00, 0x0F, 0x00, 0x00, 0x00, 0xA8, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x00,
    ];

    /// The blob behind `REGION2_ELEMENT`: token, then a version-3 regular
    /// hyperslab. libhdf5 sizes the blob for the largest reference the dataset
    /// may hold, so the encoding stops short of the object's end.
    const REGION2_HYPERSLAB_BLOB: [u8; 57] = [
        0x08, 0xC3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x02, 0x02, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x02, 0x00, 0x02, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    /// The blob of the second region reference: a version-2 point list.
    const REGION2_POINT_BLOB: [u8; 57] = [
        0x08, 0xC3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    /// The blob of the attribute reference: token, name length, name.
    const ATTR_BLOB: [u8; 15] = [
        0x08, 0xC3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x6E, 0x6F, 0x74, 0x65,
    ];

    /// The name every external reference in `tests/fixtures/ext_refs.h5`
    /// carries: the path its target file was created under, relative to the
    /// crate root the generator ran in.
    const EXT_FILE: &str = "tests/fixtures/ext_ref_target.h5";

    /// The first element of that fixture's `extobjrefs`: an `H5R_OBJECT2`
    /// whose flags carry `H5R_IS_EXTERNAL`, which sends it to the heap
    /// although its kind alone would keep it inline.
    const EXT_OBJ2_ELEMENT: [u8; 18] = [
        0x02, 0x01, 0x2B, 0x00, 0x00, 0x00, 0x24, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00,
    ];

    /// Its blob: token, then the file name, and nothing after it.
    const EXT_OBJ2_BLOB: [u8; 43] = [
        0x08, 0x20, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x74, 0x65, 0x73, 0x74,
        0x73, 0x2F, 0x66, 0x69, 0x78, 0x74, 0x75, 0x72, 0x65, 0x73, 0x2F, 0x65, 0x78, 0x74, 0x5F,
        0x72, 0x65, 0x66, 0x5F, 0x74, 0x61, 0x72, 0x67, 0x65, 0x74, 0x2E, 0x68, 0x35,
    ];

    /// The blob of the same fixture's `extattrrefs`: the file name sits
    /// between the token and the attribute name, so the two strings are only
    /// told apart by their order.
    const EXT_ATTR_BLOB: [u8; 49] = [
        0x08, 0x20, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x74, 0x65, 0x73, 0x74,
        0x73, 0x2F, 0x66, 0x69, 0x78, 0x74, 0x75, 0x72, 0x65, 0x73, 0x2F, 0x65, 0x78, 0x74, 0x5F,
        0x72, 0x65, 0x66, 0x5F, 0x74, 0x61, 0x72, 0x67, 0x65, 0x74, 0x2E, 0x68, 0x35, 0x04, 0x00,
        0x6E, 0x6F, 0x74, 0x65,
    ];

    /// An `H5R_OBJECT2` element keeps its encoded reference inline; the other
    /// two kinds point at a heap blob.
    #[test]
    fn revised_elements_from_libhdf5_split_into_kind_and_body() {
        let RevisedElement::Inline { kind, body } =
            decode_revised_element(&OBJ2_ELEMENT, &ctx()).unwrap()
        else {
            panic!("an object reference is stored in the element");
        };
        assert_eq!(kind, ReferenceKind::Object2);
        assert_eq!(
            decode_revised_body(kind, false, body, &ctx()).unwrap(),
            Some(DecodedReference {
                address: 0xC3,
                file: None,
                target: ReferenceTarget::Object,
            })
        );

        assert_eq!(
            decode_revised_element(&REGION2_ELEMENT, &ctx()).unwrap(),
            RevisedElement::Heap {
                kind: ReferenceKind::DatasetRegion2,
                external: false,
                collection: 0x8A8,
                index: 1,
            }
        );
        assert_eq!(
            decode_revised_element(&ATTR_ELEMENT, &ctx()).unwrap(),
            RevisedElement::Heap {
                kind: ReferenceKind::Attr,
                external: false,
                collection: 0x8A8,
                index: 3,
            }
        );
    }

    /// The bounds of both region blobs are what `H5Sget_select_bounds` reports
    /// for the selections the fixture generator made.
    #[test]
    fn revised_region_blobs_decode_to_their_selections() {
        let hyper = decode_revised_body(
            ReferenceKind::DatasetRegion2,
            false,
            &REGION2_HYPERSLAB_BLOB,
            &ctx(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(hyper.address, 0xC3);
        let ReferenceTarget::Region(selection) = &hyper.target else {
            panic!("a region reference names a selection");
        };
        assert_eq!(
            selection.bounds(),
            Some((vec![1, 2], vec![2, 4])),
            "{selection:?}"
        );

        let points = decode_revised_body(
            ReferenceKind::DatasetRegion2,
            false,
            &REGION2_POINT_BLOB,
            &ctx(),
        )
        .unwrap()
        .unwrap();
        let ReferenceTarget::Region(selection) = &points.target else {
            panic!("a region reference names a selection");
        };
        assert_eq!(
            selection,
            &Selection::Points(PointSelection {
                rank: 2,
                points: vec![vec![0, 1], vec![3, 5]],
            })
        );
        assert_eq!(selection.bounds(), Some((vec![0, 1], vec![3, 5])));
    }

    #[test]
    fn an_attribute_reference_carries_its_name() {
        assert_eq!(
            decode_revised_body(ReferenceKind::Attr, false, &ATTR_BLOB, &ctx()).unwrap(),
            Some(DecodedReference {
                address: 0xC3,
                file: None,
                target: ReferenceTarget::Attribute("note".into()),
            })
        );
    }

    /// An unwritten element is reference type 0 over a nil blob id
    /// (`H5T__ref_disk_isnull`); a type byte outside the 1.12 kinds and a
    /// foreign token width are both reported rather than read as something
    /// else.
    #[test]
    fn revised_elements_that_name_nothing_or_cannot_be_followed() {
        assert_eq!(
            decode_revised_element(&[0u8; 18], &ctx()).unwrap(),
            RevisedElement::Null
        );

        let mut stale = [0u8; 18];
        stale[6] = 0xA8;
        stale[7] = 0x08;
        assert!(
            matches!(
                decode_revised_element(&stale, &ctx()).unwrap_err(),
                FormatError::InvalidData(_)
            ),
            "reference type 0 over a live blob is not a null reference"
        );

        let mut old_code = OBJ2_ELEMENT;
        old_code[0] = ReferenceKind::DatasetRegion1.code();
        assert!(matches!(
            decode_revised_element(&old_code, &ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));

        let mut foreign = ATTR_BLOB;
        foreign[0] = 16;
        assert!(
            matches!(
                decode_revised_body(ReferenceKind::Attr, false, &foreign, &ctx()).unwrap_err(),
                FormatError::UnsupportedFeature(_)
            ),
            "a token that is not a file address belongs to another VOL connector"
        );

        let mut unset = OBJ2_ELEMENT;
        unset[3] = 0;
        let RevisedElement::Inline { kind, body } = decode_revised_element(&unset, &ctx()).unwrap()
        else {
            panic!("an object reference is stored in the element");
        };
        assert_eq!(
            decode_revised_body(kind, false, body, &ctx()).unwrap(),
            None,
            "a zero token names no object"
        );
    }

    /// An element flagged `H5R_IS_EXTERNAL` names the file its target lives
    /// in, and takes the heap even when its kind would otherwise be stored
    /// inline (`H5T__ref_disk_getsize`, H5Tref.c:890).
    #[test]
    fn an_external_reference_names_the_file_its_target_is_in() {
        assert_eq!(
            decode_revised_element(&EXT_OBJ2_ELEMENT, &ctx()).unwrap(),
            RevisedElement::Heap {
                kind: ReferenceKind::Object2,
                external: true,
                collection: 0x824,
                index: 1,
            }
        );
        assert_eq!(
            decode_revised_body(ReferenceKind::Object2, true, &EXT_OBJ2_BLOB, &ctx()).unwrap(),
            Some(DecodedReference {
                address: 0x320,
                file: Some(EXT_FILE.into()),
                target: ReferenceTarget::Object,
            })
        );

        // The file name precedes the kind's own payload, so reading it as if
        // the reference were internal would take the name for the payload.
        assert_eq!(
            decode_revised_body(ReferenceKind::Attr, true, &EXT_ATTR_BLOB, &ctx()).unwrap(),
            Some(DecodedReference {
                address: 0x320,
                file: Some(EXT_FILE.into()),
                target: ReferenceTarget::Attribute("note".into()),
            })
        );
        let internal = decode_revised_body(ReferenceKind::Attr, false, &EXT_ATTR_BLOB, &ctx())
            .unwrap()
            .unwrap();
        assert_eq!(
            internal.target,
            ReferenceTarget::Attribute(EXT_FILE.into()),
            "the flag is what tells the file name from the attribute name"
        );
    }

    /// Every 1.12 element this crate writes is the image libhdf5 wrote for the
    /// same reference: the inline object form, and the blob id the other two
    /// kinds carry.
    #[test]
    fn revised_elements_encode_to_the_libhdf5_images() {
        assert_eq!(
            encode_reference_element(&ReferenceElementImage::Inline(0xC3), 18, &ctx()).unwrap(),
            OBJ2_ELEMENT
        );
        assert_eq!(
            encode_reference_element(
                &ReferenceElementImage::Blob {
                    kind: ReferenceKind::DatasetRegion2,
                    size: REGION2_HYPERSLAB_BLOB.len() as u32,
                    collection: 0x8A8,
                    index: 1,
                },
                18,
                &ctx()
            )
            .unwrap(),
            REGION2_ELEMENT
        );
        assert_eq!(
            encode_reference_element(
                &ReferenceElementImage::Blob {
                    kind: ReferenceKind::Attr,
                    size: ATTR_BLOB.len() as u32,
                    collection: 0x8A8,
                    index: 3,
                },
                18,
                &ctx()
            )
            .unwrap(),
            ATTR_ELEMENT
        );
        // The pre-1.12 element is the address and nothing else.
        assert_eq!(
            encode_reference_element(&ReferenceElementImage::Legacy(0xC3), 8, &ctx()).unwrap(),
            [0xC3, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    /// A blob-backed element needs more room than an inline one; a datatype
    /// too narrow for the layout is reported rather than truncated.
    #[test]
    fn an_element_narrower_than_its_layout_is_refused() {
        let err = encode_reference_element(
            &ReferenceElementImage::Blob {
                kind: ReferenceKind::Attr,
                size: 15,
                collection: 0x8A8,
                index: 3,
            },
            11,
            &ctx(),
        )
        .unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)), "{err:?}");
        assert!(encode_reference_element(&ReferenceElementImage::Inline(0xC3), 11, &ctx()).is_ok());
    }

    /// A blob this crate encodes says the same reference the libhdf5 blob it
    /// came from says, and is read back by the same decoder.
    ///
    /// Byte equality holds for everything but the serialized selection, which
    /// this crate writes in the version-1 block-list form every bounded
    /// selection takes here while the fixture's `latest` bound produced the
    /// version-3 one. Both are `H5S_decode` input; the length field ahead of
    /// the selection is what bounds it, and it is written from the bytes
    /// actually produced.
    #[test]
    fn revised_blobs_encode_to_what_libhdf5_reads_back() {
        let attr = decode_revised_body(ReferenceKind::Attr, false, &ATTR_BLOB, &ctx())
            .unwrap()
            .unwrap();
        assert_eq!(
            encode_revised_blob(attr.address, &attr.target, 0, &ctx()).unwrap(),
            ATTR_BLOB
        );

        for (kind, golden) in [
            (ReferenceKind::DatasetRegion2, &REGION2_HYPERSLAB_BLOB[..]),
            (ReferenceKind::DatasetRegion2, &REGION2_POINT_BLOB[..]),
        ] {
            let DecodedReference {
                address, target, ..
            } = decode_revised_body(kind, false, golden, &ctx())
                .unwrap()
                .unwrap();
            let blob = encode_revised_blob(address, &target, 2, &ctx()).unwrap();
            // Token and rank are byte-identical; the selection is re-encoded.
            assert_eq!(blob[..9], golden[..9]);
            assert_eq!(blob[13..17], golden[13..17], "the extent rank");
            let selection_len = u32::from_le_bytes(blob[9..13].try_into().unwrap()) as usize;
            assert_eq!(blob.len(), 17 + selection_len);
            let back = decode_revised_body(kind, false, &blob, &ctx())
                .unwrap()
                .unwrap();
            assert_eq!(back.address, address);
            let (ReferenceTarget::Region(was), ReferenceTarget::Region(now)) =
                (&target, &back.target)
            else {
                panic!("a region reference names a selection");
            };
            // Version 1 spells a regular hyperslab as the blocks it covers, so
            // what survives is the region, not the form it was written in.
            assert_eq!(
                was.to_boxes(&[4, 6]).unwrap(),
                now.to_boxes(&[4, 6]).unwrap()
            );
            assert_eq!(was.bounds(), now.bounds());
        }

        // `H5S_SEL_ALL` serializes without a rank, and the blob says one
        // anyway, because `H5R__encode_region` reads it from the dataspace.
        let all =
            encode_revised_blob(0xC3, &ReferenceTarget::Region(Selection::All), 3, &ctx()).unwrap();
        assert_eq!(u32::from_le_bytes(all[13..17].try_into().unwrap()), 3);
        assert_eq!(
            decode_revised_body(ReferenceKind::DatasetRegion2, false, &all, &ctx()).unwrap(),
            Some(DecodedReference {
                address: 0xC3,
                file: None,
                target: ReferenceTarget::Region(Selection::All),
            })
        );

        // An object reference's blob is the token alone; that is also the body
        // an `H5R_OBJECT2` element carries inline.
        assert_eq!(
            encode_revised_blob(0xC3, &ReferenceTarget::Object, 0, &ctx()).unwrap(),
            OBJ2_ELEMENT[2..11]
        );
    }

    /// A truncated selection is reported, not read past.
    #[test]
    fn a_truncated_selection_is_refused() {
        let err = Selection::decode(&REGION_HEAP_OBJECT[8..20]).unwrap_err();
        assert!(
            matches!(err, FormatError::BufferTooShort { .. }),
            "unexpected error: {err:?}"
        );
    }
}
