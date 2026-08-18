//! Virtual Dataset mapping list — the global-heap-resident payload a
//! Virtual layout message
//! ([`crate::format::messages::data_layout::DataLayoutMessage::Virtual`])
//! points at by heap address and object index
//! (`H5D__virtual_load_layout`, H5Dvirtual.c). Unlike the External File
//! List message's slot names (which index into a *local* heap that needs
//! a second address-based read), everything here — including both source
//! and destination names — is inline in the one heap object this module
//! decodes: no further indirection.
//!
//! Binary layout (the heap object's raw bytes, in full):
//! ```text
//! heap_version: 1 byte (0 or 1)
//! num_entries:  sizeof_size bytes LE
//! num_entries * entry {
//!     if heap_version >= 1: flags: 1 byte
//!         (bit0 = SOURCE_FILE_SHARED, bit1 = SOURCE_DSET_SHARED,
//!          bit2 = SOURCE_SAME_FILE — heap_version 0 has no flags byte at
//!          all, so every name is always an inline string)
//!
//!     source file name:
//!       if flags & SAME_FILE: no bytes on the wire; name is "."
//!       elif flags & FILE_SHARED: origin_index (sizeof_size bytes LE),
//!           must be < this entry's index; reuse that entry's name
//!       else: NUL-terminated string
//!
//!     source dataset name: same three forms, using
//!       DSET_SHARED/origin_index instead of FILE_SHARED (no "same file"
//!       form — a dataset is never "the VDS itself")
//!
//!     source_selection:  a serialized H5S selection (see
//!       [`crate::format::selection::Selection::decode`])
//!     virtual_selection:  a serialized H5S selection, immediately after
//! }
//! checksum: 4 bytes LE (Jenkins lookup3 / H5_checksum_metadata over
//!           every byte before it)
//! ```
//!
//! Empirically confirmed byte-for-byte against a real h5py-written VDS
//! (`heap_version` 0 in every fixture h5py's `create_virtual_dataset`
//! produces — h5py never emits the version-1 name-sharing optimizations,
//! though a spec-conformant reader still has to decode them).

use crate::format::checksum::checksum_metadata;
use crate::format::selection::Selection;
use crate::format::{FormatContext, FormatError, FormatResult};

const ENC_VERS_0: u8 = 0;
const ENC_VERS_1: u8 = 1;

const SOURCE_FILE_SHARED: u8 = 0x01;
const SOURCE_DSET_SHARED: u8 = 0x02;
const SOURCE_SAME_FILE: u8 = 0x04;
const ALL_FLAGS: u8 = SOURCE_FILE_SHARED | SOURCE_DSET_SHARED | SOURCE_SAME_FILE;

/// One (source, virtual) mapping entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualMapping {
    /// The source file's name, exactly as stored (or `"."` for
    /// `SOURCE_SAME_FILE`) — resolved against `HDF5_VDS_PREFIX` /
    /// the VDS file's own directory at read time, not here.
    pub source_file_name: String,
    /// The source dataset's path within that file.
    pub source_dset_name: String,
    /// Which elements of the source dataset this mapping reads.
    pub source_selection: Selection,
    /// Which elements of the virtual dataset this mapping fills.
    pub virtual_selection: Selection,
}

/// A source name split around its `printf`-style block substitutions —
/// `H5D_virtual_parse_source_name` (H5Dvirtual.c).
///
/// A virtual dataset whose virtual selection is unlimited and whose source
/// selection is not draws each block of the virtual selection from a
/// *different* source dataset, named by substituting the block index into
/// the stored name. Only two conversions are legal: `%b`, the block index,
/// and `%%`, an escaped literal `%`. Anything else is "invalid format
/// specifier", and libhdf5 raises it both when the mapping is set and when
/// the layout is loaded back out of the file, so a name that does not parse
/// makes the dataset unopenable rather than merely unwritable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSourceName {
    /// The literal text around the substitutions, `%%` already unescaped to
    /// a single `%` — upstream's `H5O_storage_virtual_name_seg_t` chain.
    /// Always exactly one longer than the substitution count, so joining the
    /// segments with the printed block index is the whole of
    /// `H5D__virtual_build_source_name`.
    segments: Vec<String>,
}

impl ParsedSourceName {
    /// How many `%b` substitutions the name carries — upstream's `nsubs`,
    /// the quantity `H5D_virtual_check_mapping_post` tests to decide whether
    /// a mapping is a printf mapping at all.
    pub fn nsubs(&self) -> usize {
        self.segments.len() - 1
    }

    /// The name block `blockno` resolves to —
    /// `H5D__virtual_build_source_name`. With no substitutions this is the
    /// unescaped name, which is what upstream uses for an ordinary mapping
    /// too (`H5D__virtual_load_layout` takes `parsed_name->name_segment`,
    /// not the stored string, whenever the name parsed into one).
    pub fn build(&self, blockno: u64) -> String {
        self.segments.join(&blockno.to_string())
    }
}

/// Split a source file or dataset name around its `%b` substitutions —
/// `H5D_virtual_parse_source_name` (H5Dvirtual.c). See [`ParsedSourceName`].
pub fn parse_source_name(name: &str) -> FormatResult<ParsedSourceName> {
    let mut segments = vec![String::new()];
    let mut rest = name;
    while let Some(pct) = rest.find('%') {
        let (literal, tail) = rest.split_at(pct);
        segments.last_mut().expect("never empty").push_str(literal);
        match tail.as_bytes().get(1) {
            Some(b'b') => segments.push(String::new()),
            Some(b'%') => segments.last_mut().expect("never empty").push('%'),
            _ => {
                return Err(FormatError::InvalidData(format!(
                    "invalid format specifier in virtual dataset source name {name:?}: only \
                     %b (block index) and %% (escaped percent) are legal"
                )))
            }
        }
        rest = &tail[2.min(tail.len())..];
    }
    segments.last_mut().expect("never empty").push_str(rest);
    Ok(ParsedSourceName { segments })
}

/// A decoded Virtual Dataset mapping list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualMappingList {
    pub mappings: Vec<VirtualMapping>,
}

impl VirtualMappingList {
    /// Encode this mapping list into the bytes one global heap object holds
    /// — `H5D__virtual_store_layout` (H5Dvirtual.c).
    ///
    /// Always at heap encoding version 0, because that is the only version
    /// libhdf5 writes short of `H5F_LIBVER_V200`: version 1 exists solely to
    /// shorten repeated names (`SOURCE_SAME_FILE` / the two `_SHARED` forms),
    /// and `H5D__virtual_store_layout` picks it only when the file's *low*
    /// bound is at least V200 *and* it measures the version-1 block as no
    /// larger. Nothing here is lost by staying at 0: the same mappings decode
    /// back identically, one inline name per entry.
    ///
    /// A name holding an interior NUL is refused rather than written: the
    /// wire form terminates each name with one, so a name containing another
    /// would read back truncated — and everything after it in the block would
    /// decode as some other field.
    pub fn encode(&self, ctx: &FormatContext) -> FormatResult<Vec<u8>> {
        let ss = ctx.sizeof_size as usize;
        let mut buf = Vec::new();
        buf.push(ENC_VERS_0);
        buf.extend_from_slice(&(self.mappings.len() as u64).to_le_bytes()[..ss]);
        for m in &self.mappings {
            push_cstr(&mut buf, &m.source_file_name, "source file")?;
            push_cstr(&mut buf, &m.source_dset_name, "source dataset")?;
            buf.extend_from_slice(&m.source_selection.encode()?);
            buf.extend_from_slice(&m.virtual_selection.encode()?);
        }
        let cksum = checksum_metadata(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        Ok(buf)
    }

    /// Decode a mapping list from a global heap object's raw bytes.
    ///
    /// Unlike most message decoders here, this does not return a
    /// "bytes consumed" count: `buf` is expected to be exactly one heap
    /// object's data (as `GlobalHeapCollection::get_object` returns it),
    /// and every byte in it belongs to this structure — trailing bytes
    /// left over after the checksum are a corrupt block, not data for a
    /// caller to continue decoding, so that case is an error instead.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let ss = ctx.sizeof_size as usize;
        if buf.is_empty() {
            return Err(FormatError::BufferTooShort {
                needed: 1,
                available: 0,
            });
        }
        let heap_version = buf[0];
        if heap_version != ENC_VERS_0 && heap_version != ENC_VERS_1 {
            return Err(FormatError::InvalidVersion(heap_version));
        }
        let mut pos = 1;

        if buf.len() < pos + ss {
            return Err(FormatError::BufferTooShort {
                needed: pos + ss,
                available: buf.len(),
            });
        }
        let num_entries = crate::format::bytes::read_le_uint(&buf[pos..], ss) as usize;
        pos += ss;

        let mut mappings: Vec<VirtualMapping> = Vec::new();
        for i in 0..num_entries {
            let flags = if heap_version >= ENC_VERS_1 {
                if buf.len() < pos + 1 {
                    return Err(FormatError::BufferTooShort {
                        needed: pos + 1,
                        available: buf.len(),
                    });
                }
                let f = buf[pos];
                pos += 1;
                if f & !ALL_FLAGS != 0 {
                    return Err(FormatError::InvalidData(format!(
                        "unknown virtual dataset mapping flag bits in {f:#x}"
                    )));
                }
                f
            } else {
                0
            };

            let source_file_name = if flags & SOURCE_SAME_FILE != 0 {
                ".".to_string()
            } else if flags & SOURCE_FILE_SHARED != 0 {
                let origin = read_origin_index(buf, &mut pos, ss, i)?;
                mappings[origin].source_file_name.clone()
            } else {
                read_cstr(buf, &mut pos)?
            };

            let source_dset_name = if flags & SOURCE_DSET_SHARED != 0 {
                let origin = read_origin_index(buf, &mut pos, ss, i)?;
                mappings[origin].source_dset_name.clone()
            } else {
                read_cstr(buf, &mut pos)?
            };

            let (source_selection, consumed) = Selection::decode(&buf[pos..])?;
            pos += consumed;
            let (virtual_selection, consumed) = Selection::decode(&buf[pos..])?;
            pos += consumed;

            // `H5D__virtual_load_layout` parses both names as it decodes the
            // entry, so a name with an illegal conversion fails the load
            // rather than surfacing later as a source that cannot be found.
            parse_source_name(&source_file_name)?;
            parse_source_name(&source_dset_name)?;

            mappings.push(VirtualMapping {
                source_file_name,
                source_dset_name,
                source_selection,
                virtual_selection,
            });
        }

        if buf.len() < pos + 4 {
            return Err(FormatError::BufferTooShort {
                needed: pos + 4,
                available: buf.len(),
            });
        }
        let stored_cksum = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        let computed_cksum = checksum_metadata(&buf[..pos]);
        if stored_cksum != computed_cksum {
            return Err(FormatError::ChecksumMismatch {
                expected: stored_cksum,
                computed: computed_cksum,
            });
        }
        pos += 4;

        if pos != buf.len() {
            return Err(FormatError::InvalidData(format!(
                "virtual dataset mapping list declares {pos} bytes but the heap object holds {}",
                buf.len()
            )));
        }

        Ok(Self { mappings })
    }
}

/// Append a name and its NUL terminator, refusing one that already holds a
/// NUL (see [`VirtualMappingList::encode`]).
fn push_cstr(buf: &mut Vec<u8>, name: &str, what: &str) -> FormatResult<()> {
    if name.as_bytes().contains(&0) {
        return Err(FormatError::InvalidData(format!(
            "virtual dataset {what} name {name:?} contains a NUL, which terminates a \
             name on the wire"
        )));
    }
    buf.extend_from_slice(name.as_bytes());
    buf.push(0);
    Ok(())
}

/// Read a `sizeof_size`-byte origin-entry index and validate it points
/// strictly before the current entry (`H5D__virtual_load_layout`'s own
/// check — a forward or self reference is malformed, not just unusual).
fn read_origin_index(
    buf: &[u8],
    pos: &mut usize,
    ss: usize,
    this_entry: usize,
) -> FormatResult<usize> {
    if buf.len() < *pos + ss {
        return Err(FormatError::BufferTooShort {
            needed: *pos + ss,
            available: buf.len(),
        });
    }
    let origin = crate::format::bytes::read_le_uint(&buf[*pos..], ss) as usize;
    *pos += ss;
    if origin >= this_entry {
        return Err(FormatError::InvalidData(format!(
            "virtual dataset mapping entry {this_entry} shares a name with entry {origin}, \
             which is not an earlier entry"
        )));
    }
    Ok(origin)
}

/// Read a NUL-terminated string starting at `*pos`, requiring the
/// terminator to appear within `buf` (an unterminated string is a
/// truncated/corrupt block, matching `H5D__virtual_load_layout`'s own
/// "ran off end of input buffer... unterminated" check).
fn read_cstr(buf: &[u8], pos: &mut usize) -> FormatResult<String> {
    let start = *pos;
    let nul = buf[start..].iter().position(|&b| b == 0).ok_or_else(|| {
        FormatError::InvalidData(
            "virtual dataset mapping entry has an unterminated name string".into(),
        )
    })?;
    let s = String::from_utf8_lossy(&buf[start..start + nul]).into_owned();
    *pos = start + nul + 1;
    Ok(s)
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::selection::{Hyperslab, HyperslabBlock};

    fn ctx8() -> FormatContext {
        FormatContext {
            sizeof_addr: 8,
            sizeof_size: 8,
        }
    }

    fn all_selection_bytes() -> Vec<u8> {
        let mut b = vec![0x03, 0, 0, 0]; // SEL_ALL
        b.extend_from_slice(&1u32.to_le_bytes()); // version 1
        b.extend_from_slice(&[0u8; 8]); // reserved
        b
    }

    /// A single-entry, both-sides-ALL heap block, built to exactly match
    /// what h5debug reported for a real h5py-written VDS
    /// (`layout[...] = VirtualSource(...)`): heap_version 0 (no flags
    /// byte), inline names, checksum computed over the real body.
    fn single_entry_all_block() -> Vec<u8> {
        let mut body = vec![ENC_VERS_0];
        body.extend_from_slice(&1u64.to_le_bytes()); // num_entries
        body.extend_from_slice(b"src.h5\0");
        body.extend_from_slice(b"data\0");
        body.extend_from_slice(&all_selection_bytes()); // source selection
        body.extend_from_slice(&all_selection_bytes()); // virtual selection
        let cksum = checksum_metadata(&body);
        body.extend_from_slice(&cksum.to_le_bytes());
        body
    }

    #[test]
    fn decode_single_all_mapping() {
        let buf = single_entry_all_block();
        let list = VirtualMappingList::decode(&buf, &ctx8()).unwrap();
        assert_eq!(list.mappings.len(), 1);
        let m = &list.mappings[0];
        assert_eq!(m.source_file_name, "src.h5");
        assert_eq!(m.source_dset_name, "data");
        assert_eq!(m.source_selection, Selection::All);
        assert_eq!(m.virtual_selection, Selection::All);
    }

    #[test]
    fn decode_empty_mapping_list() {
        let mut body = vec![ENC_VERS_0];
        body.extend_from_slice(&0u64.to_le_bytes());
        let cksum = checksum_metadata(&body);
        body.extend_from_slice(&cksum.to_le_bytes());
        let list = VirtualMappingList::decode(&body, &ctx8()).unwrap();
        assert!(list.mappings.is_empty());
    }

    #[test]
    fn decode_rejects_bad_checksum() {
        let mut buf = single_entry_all_block();
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;
        let err = VirtualMappingList::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::ChecksumMismatch { .. }));
    }

    #[test]
    fn decode_rejects_bad_heap_version() {
        let mut buf = single_entry_all_block();
        buf[0] = 2;
        let err = VirtualMappingList::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidVersion(2)));
    }

    #[test]
    fn decode_rejects_unterminated_name() {
        let mut body = vec![ENC_VERS_0];
        body.extend_from_slice(&1u64.to_le_bytes());
        body.extend_from_slice(b"no_nul_here"); // never terminated
        let err = VirtualMappingList::decode(&body, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    /// heap_version 1's `SOURCE_SAME_FILE` flag emits no bytes for the
    /// source file name at all — it always means the literal `"."` — and
    /// two entries can each independently set it.
    #[test]
    fn decode_heap_version_1_same_file() {
        let mut body = vec![ENC_VERS_1];
        body.extend_from_slice(&2u64.to_le_bytes()); // num_entries

        // Entry 0: SAME_FILE.
        body.push(SOURCE_SAME_FILE);
        body.extend_from_slice(b"a\0");
        body.extend_from_slice(&all_selection_bytes());
        body.extend_from_slice(&all_selection_bytes());

        // Entry 1: also SAME_FILE, different dataset.
        body.push(SOURCE_SAME_FILE);
        body.extend_from_slice(b"b\0");
        body.extend_from_slice(&all_selection_bytes());
        body.extend_from_slice(&all_selection_bytes());

        let cksum = checksum_metadata(&body);
        body.extend_from_slice(&cksum.to_le_bytes());

        let list = VirtualMappingList::decode(&body, &ctx8()).unwrap();
        assert_eq!(list.mappings.len(), 2);
        assert_eq!(list.mappings[0].source_file_name, ".");
        assert_eq!(list.mappings[1].source_file_name, ".");
        assert_eq!(list.mappings[0].source_dset_name, "a");
        assert_eq!(list.mappings[1].source_dset_name, "b");
    }

    /// heap_version 1's `SOURCE_FILE_SHARED`/`SOURCE_DSET_SHARED` flags
    /// reference an earlier entry's already-decoded name by index.
    #[test]
    fn decode_heap_version_1_shared_names() {
        let mut body = vec![ENC_VERS_1];
        body.extend_from_slice(&2u64.to_le_bytes());

        // Entry 0: literal names.
        body.push(0);
        body.extend_from_slice(b"shared.h5\0");
        body.extend_from_slice(b"data\0");
        body.extend_from_slice(&all_selection_bytes());
        body.extend_from_slice(&all_selection_bytes());

        // Entry 1: both names shared from entry 0.
        body.push(SOURCE_FILE_SHARED | SOURCE_DSET_SHARED);
        body.extend_from_slice(&0u64.to_le_bytes()); // origin for file
        body.extend_from_slice(&0u64.to_le_bytes()); // origin for dset
        body.extend_from_slice(&all_selection_bytes());
        body.extend_from_slice(&all_selection_bytes());

        let cksum = checksum_metadata(&body);
        body.extend_from_slice(&cksum.to_le_bytes());

        let list = VirtualMappingList::decode(&body, &ctx8()).unwrap();
        assert_eq!(list.mappings[1].source_file_name, "shared.h5");
        assert_eq!(list.mappings[1].source_dset_name, "data");
    }

    /// A shared-name origin index that is not strictly earlier than the
    /// current entry (self or forward reference) is malformed, matching
    /// `H5D__virtual_load_layout`'s own check.
    #[test]
    fn decode_rejects_non_earlier_shared_origin() {
        let mut body = vec![ENC_VERS_1];
        body.extend_from_slice(&1u64.to_le_bytes());
        body.push(SOURCE_FILE_SHARED);
        body.extend_from_slice(&0u64.to_le_bytes()); // origin == this entry's own index (0)
        let err = VirtualMappingList::decode(&body, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    #[test]
    fn decode_rejects_unknown_flag_bits() {
        let mut body = vec![ENC_VERS_1];
        body.extend_from_slice(&1u64.to_le_bytes());
        body.push(0x08); // no such flag bit
        let err = VirtualMappingList::decode(&body, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }

    /// A mapping with a real (non-ALL) hyperslab selection round-trips
    /// through the same block, matching `layout[4:12] = ...`.
    #[test]
    fn decode_mapping_with_hyperslab_virtual_selection() {
        let mut hyper = vec![0x02, 0, 0, 0]; // SEL_HYPERSLABS
        hyper.extend_from_slice(&1u32.to_le_bytes()); // version 1
        hyper.extend_from_slice(&[0u8; 8]);
        hyper.extend_from_slice(&1u32.to_le_bytes()); // rank
        hyper.extend_from_slice(&1u32.to_le_bytes()); // num_blocks
        hyper.extend_from_slice(&4u32.to_le_bytes()); // start
        hyper.extend_from_slice(&11u32.to_le_bytes()); // end

        let mut body = vec![ENC_VERS_0];
        body.extend_from_slice(&1u64.to_le_bytes());
        body.extend_from_slice(b"src.h5\0");
        body.extend_from_slice(b"data\0");
        body.extend_from_slice(&all_selection_bytes()); // source: ALL
        body.extend_from_slice(&hyper); // virtual: [4:12]
        let cksum = checksum_metadata(&body);
        body.extend_from_slice(&cksum.to_le_bytes());

        let list = VirtualMappingList::decode(&body, &ctx8()).unwrap();
        match &list.mappings[0].virtual_selection {
            Selection::Hyperslab {
                rank: 1,
                form: Hyperslab::Blocks(blocks),
            } => {
                assert_eq!(
                    blocks,
                    &vec![HyperslabBlock {
                        start: vec![4],
                        end: vec![11],
                    }]
                );
            }
            other => panic!("expected a rank-1 hyperslab, got {other:?}"),
        }
    }

    #[test]
    fn decode_truncated_num_entries() {
        let buf = [ENC_VERS_0, 0, 0, 0];
        let err = VirtualMappingList::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    #[test]
    fn decode_empty_buffer() {
        let err = VirtualMappingList::decode(&[], &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::BufferTooShort { .. }));
    }

    /// The 60 bytes libhdf5 1.14 actually wrote for the oracle's `vds` case
    /// (`h5py.VirtualLayout(shape=(16,))[...] = VirtualSource("vds_src.h5",
    /// "src", shape=(16,))`), lifted out of the global heap object the layout
    /// message points at — heap version 0, one entry, both selections ALL.
    /// `encode` must reproduce it byte for byte, checksum included.
    #[test]
    fn encode_matches_the_captured_libhdf5_block() {
        let list = VirtualMappingList {
            mappings: vec![VirtualMapping {
                source_file_name: "vds_src.h5".into(),
                source_dset_name: "src".into(),
                source_selection: Selection::All,
                virtual_selection: Selection::All,
            }],
        };
        let captured = [
            0x00, // heap encoding version 0
            0x01, 0, 0, 0, 0, 0, 0, 0, // num_entries = 1
            b'v', b'd', b's', b'_', b's', b'r', b'c', b'.', b'h', b'5', 0x00, b's', b'r', b'c',
            0x00, //
            0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // source: ALL
            0x03, 0, 0, 0, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // virtual: ALL
            0xcd, 0xe5, 0xe5, 0xed, // checksum
        ];
        assert_eq!(list.encode(&ctx8()).unwrap(), captured);
    }

    #[test]
    fn encode_roundtrips_a_hyperslab_mapping_at_ctx4() {
        let ctx4 = FormatContext {
            sizeof_addr: 4,
            sizeof_size: 4,
        };
        let list = VirtualMappingList {
            mappings: vec![
                VirtualMapping {
                    source_file_name: "a.h5".into(),
                    source_dset_name: "one".into(),
                    source_selection: Selection::All,
                    virtual_selection: Selection::Hyperslab {
                        rank: 1,
                        form: Hyperslab::Blocks(vec![HyperslabBlock {
                            start: vec![0],
                            end: vec![7],
                        }]),
                    },
                },
                VirtualMapping {
                    source_file_name: "b.h5".into(),
                    source_dset_name: "two".into(),
                    source_selection: Selection::All,
                    virtual_selection: Selection::Hyperslab {
                        rank: 1,
                        form: Hyperslab::Blocks(vec![HyperslabBlock {
                            start: vec![8],
                            end: vec![15],
                        }]),
                    },
                },
            ],
        };
        let encoded = list.encode(&ctx4).unwrap();
        assert_eq!(VirtualMappingList::decode(&encoded, &ctx4).unwrap(), list);
    }

    #[test]
    fn encode_empty_list_roundtrips() {
        let list = VirtualMappingList {
            mappings: Vec::new(),
        };
        let encoded = list.encode(&ctx8()).unwrap();
        assert_eq!(encoded.len(), 1 + 8 + 4);
        assert_eq!(VirtualMappingList::decode(&encoded, &ctx8()).unwrap(), list);
    }

    #[test]
    fn encode_rejects_a_name_holding_a_nul() {
        let list = VirtualMappingList {
            mappings: vec![VirtualMapping {
                source_file_name: "sr\0c.h5".into(),
                source_dset_name: "src".into(),
                source_selection: Selection::All,
                virtual_selection: Selection::All,
            }],
        };
        let err = list.encode(&ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)), "{err:?}");
    }

    #[test]
    fn decode_rejects_trailing_garbage() {
        let mut buf = single_entry_all_block();
        buf.push(0xAB);
        let err = VirtualMappingList::decode(&buf, &ctx8()).unwrap_err();
        assert!(matches!(err, FormatError::InvalidData(_)));
    }
    /// `H5D_virtual_parse_source_name`: `%b` splits the name, `%%` is an
    /// escaped literal, and anything else after a `%` is an error. The build
    /// side is `H5D__virtual_build_source_name`.
    #[test]
    fn source_names_parse_and_build_the_way_libhdf5_does() {
        for (name, nsubs, block7) in [
            ("plain.h5", 0, "plain.h5"),
            ("f%b.h5", 1, "f7.h5"),
            ("%b", 1, "7"),
            ("a%b%bc", 2, "a77c"),
            // `%%` is a literal percent and no substitution at all, so the
            // name a mapping resolves against is the unescaped one.
            ("od%%d", 0, "od%d"),
            ("%%%b%%", 1, "%7%"),
        ] {
            let parsed = parse_source_name(name).unwrap();
            assert_eq!(parsed.nsubs(), nsubs, "{name}");
            assert_eq!(parsed.build(7), block7, "{name}");
        }
        // Two-digit block numbers are printed in full, once per specifier.
        assert_eq!(parse_source_name("b%b_%b").unwrap().build(123), "b123_123");
        for bad in ["%z", "50%", "%d.h5", "%"] {
            let err = parse_source_name(bad).unwrap_err();
            assert!(
                matches!(&err, FormatError::InvalidData(m) if m.contains("invalid format specifier")),
                "{bad}: {err:?}"
            );
        }
    }

    /// `H5D__virtual_load_layout` parses both names while decoding, so a
    /// stored name with an illegal conversion makes the layout unreadable
    /// rather than surfacing later as a source that cannot be found.
    #[test]
    fn decode_rejects_an_illegal_format_specifier_in_a_stored_name() {
        let list = VirtualMappingList {
            mappings: vec![VirtualMapping {
                source_file_name: "src.h5".into(),
                source_dset_name: "d".into(),
                source_selection: Selection::All,
                virtual_selection: Selection::All,
            }],
        };
        let mut buf = list.encode(&ctx8()).unwrap();
        // Rewrite "src.h5" as "s%z.h5" in place (same length), then fix the
        // trailing checksum so only the name is what decode objects to.
        let at = buf
            .windows(6)
            .position(|w| w == b"src.h5")
            .expect("name is inline");
        buf[at..at + 6].copy_from_slice(b"s%z.h5");
        let end = buf.len() - 4;
        let cksum = checksum_metadata(&buf[..end]);
        buf[end..].copy_from_slice(&cksum.to_le_bytes());
        let err = VirtualMappingList::decode(&buf, &ctx8()).unwrap_err();
        assert!(
            matches!(&err, FormatError::InvalidData(m) if m.contains("invalid format specifier")),
            "{err:?}"
        );
    }
}
