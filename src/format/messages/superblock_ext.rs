//! The object-header messages that live in the superblock extension.
//!
//! A v2/v3 superblock (and a v0/v1 one, whose second address field is the
//! extension address in every libhdf5 that writes one) may point at an object
//! header holding file-level metadata that does not fit in the fixed
//! superblock: the shared-message table, non-default v1 B-tree "K" values, the
//! driver-info block, and the file-space strategy. They are grouped in one
//! module because they are exactly the set of messages that only ever appear
//! there.
//!
//! Upstream: `H5Oshmesg.c`, `H5Obtreek.c`, `H5Odrvinfo.c`, `H5Ofsinfo.c`;
//! the read side that consumes them is `H5Fsuper.c::H5F__super_read`.

use crate::format::bytes::read_le_uint as read_uint;
use crate::format::{FormatContext, FormatError, FormatResult};

/// Shared Message Table message (0x000F) — `H5Oshmesg.c`.
///
/// Points at the SOHM master table and says how many indexes it holds; the
/// count is not stored in the table itself, so this message is the only way
/// to size it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessageTableMessage {
    /// Message version (0 is the only version defined).
    pub version: u8,
    /// File address of the SOHM master table (`SMTB`).
    pub table_address: u64,
    /// Number of index headers stored in that table.
    pub nindexes: u8,
}

impl SharedMessageTableMessage {
    /// Decode the message body.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        need(buf, 1 + sa + 1)?;
        let version = buf[0];
        if version != 0 {
            return Err(FormatError::InvalidVersion(version));
        }
        Ok(Self {
            version,
            table_address: read_uint(&buf[1..], sa),
            nindexes: buf[1 + sa],
        })
    }

    /// Encode the message body (`H5O__shmesg_encode`).
    pub fn encode(&self, ctx: &FormatContext) -> Vec<u8> {
        let sa = ctx.sizeof_addr as usize;
        let mut buf = Vec::with_capacity(1 + sa + 1);
        buf.push(self.version);
        buf.extend_from_slice(&self.table_address.to_le_bytes()[..sa]);
        buf.push(self.nindexes);
        buf
    }
}

/// v1 B-tree "K" values message (0x0013) — `H5Obtreek.c`.
///
/// Present only when the file was created with non-default split ranks; its
/// three values replace the superblock's own (v0/v1) or the library defaults
/// (v2/v3, whose superblock has no room for them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtreeKMessage {
    /// Internal-node 1/2 rank for chunked-storage (type 1) B-trees.
    pub chunk_internal_k: u16,
    /// Internal-node 1/2 rank for symbol-table (type 0) B-trees.
    pub snode_internal_k: u16,
    /// Symbol-table leaf-node (SNOD) 1/2 rank.
    pub sym_leaf_k: u16,
}

impl BtreeKMessage {
    /// Decode the message body.
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        need(buf, 1 + 6)?;
        let version = buf[0];
        if version != 0 {
            return Err(FormatError::InvalidVersion(version));
        }
        Ok(Self {
            chunk_internal_k: u16::from_le_bytes([buf[1], buf[2]]),
            snode_internal_k: u16::from_le_bytes([buf[3], buf[4]]),
            sym_leaf_k: u16::from_le_bytes([buf[5], buf[6]]),
        })
    }
}

/// Driver info message (0x0014) — `H5Odrvinfo.c`.
///
/// Carries the VFD-specific superblock payload (`H5FD_MULTI` / `H5FD_FAMILY`)
/// that a v0/v1 file keeps in a separate driver-info block. The payload is
/// handed back verbatim: this crate opens files through a single sec2-like
/// handle and does not implement the multi/family drivers, so the message is
/// exposed for inspection rather than acted on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverInfoMessage {
    /// The 8-byte driver identifier, trailing NULs trimmed (e.g. `NCSAmulti`
    /// is stored as `NCSAmult`).
    pub name: String,
    /// Driver-specific payload.
    pub data: Vec<u8>,
}

impl DriverInfoMessage {
    /// Decode the message body.
    pub fn decode(buf: &[u8]) -> FormatResult<Self> {
        need(buf, 1 + 8 + 2)?;
        let version = buf[0];
        if version != 0 {
            return Err(FormatError::InvalidVersion(version));
        }
        let name_bytes: Vec<u8> = buf[1..9].iter().copied().take_while(|&b| b != 0).collect();
        let name = String::from_utf8_lossy(&name_bytes).into_owned();
        let len = u16::from_le_bytes([buf[9], buf[10]]) as usize;
        if len == 0 {
            return Err(FormatError::InvalidData(
                "driver info message declares a zero-length payload".into(),
            ));
        }
        need(buf, 11 + len)?;
        Ok(Self {
            name,
            data: buf[11..11 + len].to_vec(),
        })
    }
}

/// File-space strategy (`H5F_fspace_strategy_t`), as stored in the file-space
/// info message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSpaceStrategy {
    /// Free-space managers plus aggregators (the library default).
    FsmAggr,
    /// Paged aggregation.
    Page,
    /// Aggregators only, no persisted free-space manager.
    Aggr,
    /// No cross-close reuse.
    None,
    /// A strategy byte this crate does not recognize.
    Unknown(u8),
}

impl FileSpaceStrategy {
    fn from_byte(b: u8) -> Self {
        // `H5F_fspace_strategy_t` in H5Fpublic.h, which is zero-based; the
        // one-based `H5F_file_space_type_t` below it is the deprecated
        // version-0 encoding and is mapped separately.
        match b {
            0 => Self::FsmAggr,
            1 => Self::Page,
            2 => Self::Aggr,
            3 => Self::None,
            other => Self::Unknown(other),
        }
    }

    /// The `H5F_fspace_strategy_t` value, the inverse of
    /// [`from_byte`](Self::from_byte).
    fn to_byte(self) -> u8 {
        match self {
            Self::FsmAggr => 0,
            Self::Page => 1,
            Self::Aggr => 2,
            Self::None => 3,
            Self::Unknown(b) => b,
        }
    }
}

/// Number of page-type free-space managers whose addresses a persisting
/// version-1 message stores (`H5F_MEM_PAGE_SUPER` .. `H5F_MEM_PAGE_LARGE_OHDR`).
pub const FS_ADDR_COUNT_V1: usize = 12;

/// Number of memory-type free-space managers a persisting version-0 message
/// stores (`H5FD_MEM_SUPER` .. `H5FD_MEM_OHDR`). They are the first six of
/// the twelve slots, which is why the same array serves both encodings.
const FS_ADDR_COUNT_V0: usize = 6;

/// Maximum file-space page size libhdf5 accepts
/// (`H5F_FILE_SPACE_PAGE_SIZE_MAX`, H5Fprivate.h:337) — the ceiling
/// `H5Pset_file_space_page_size` enforces, and what the decoder refuses above.
pub const PAGE_SIZE_MAX: u64 = 1024 * 1024 * 1024;

/// Smallest file-space page size libhdf5 accepts
/// (`H5F_FILE_SPACE_PAGE_SIZE_MIN`, H5Fprivate.h:336) — the floor
/// `H5Pset_file_space_page_size` enforces, and what a paged file's allocation
/// rules assume.
pub const PAGE_SIZE_MIN: u64 = 512;

/// `H5F_FILE_SPACE_PAGE_SIZE_DEF` (H5Fprivate.h:335) — the page size a file
/// gets when nothing says otherwise, which is also the one the version-0
/// encoding implies.
pub(crate) const DEFAULT_FILE_SPACE_PAGE_SIZE: u64 = 4096;

/// File-space info message (0x0017) — `H5Ofsinfo.c`.
///
/// The version-0 form is the deprecated 1.10.0 encoding. Its four strategy
/// values are mapped onto the version-1 `strategy`/`persist`/`threshold`
/// fields on decode (`H5O__fsinfo_decode`), so every consumer of this struct
/// sees one shape whatever the file carries.
///
/// What decode does *not* discard is which encoding the file used:
/// [`version`](Self::version) keeps it and [`encode`](Self::encode) re-emits
/// at that version, so appending to a version-0 file leaves it a version-0
/// file. Upstream instead upgrades — `H5O__fsinfo_encode` has no version-0
/// branch, and `H5F__super_read` removes a mapped message and writes a
/// version-1 replacement on read-write open (H5Fsuper.c:843-885) — which is a
/// format change this crate has no reason to make on the user's behalf, and
/// which libhdf5 will still make on its own next open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSpaceInfoMessage {
    /// The encoding this message uses on disk: 0 for the deprecated 1.10.0
    /// form, 1 for the current one. Decode records what it read and encode
    /// writes that same form back; nothing else in the crate branches on it,
    /// because decode has already normalized the fields below.
    pub version: u8,
    /// File-space handling strategy.
    pub strategy: FileSpaceStrategy,
    /// Whether free-space manager state is persisted across close/reopen.
    pub persist: bool,
    /// Smallest free-space section tracked by a manager.
    pub threshold: u64,
    /// File-space page size (paged aggregation).
    pub page_size: u64,
    /// Page-end metadata threshold.
    pub pgend_meta_thres: u16,
    /// End-of-address before free-space header/section allocation.
    pub eoa_pre_fsm_fsalloc: u64,
    /// Addresses of the free-space managers, indexed by
    /// [`FreeSpaceClass::message_slot`](crate::format::free_space::FreeSpaceClass::message_slot).
    ///
    /// Always [`FS_ADDR_COUNT_V1`] entries, undefined where there is no
    /// manager — including for a non-persisting message, which stores none of
    /// them, and for a version-0 message, which stores only the first
    /// [`FS_ADDR_COUNT_V0`]. Upstream keeps the same full-width array for the
    /// same reason (`H5O__fsinfo_decode` fills all twelve with `HADDR_UNDEF`
    /// before reading any): a caller indexes a slot without first asking which
    /// encoding the slot came from.
    pub fs_addr: Vec<u64>,
}

impl FileSpaceInfoMessage {
    /// Decode the message body.
    pub fn decode(buf: &[u8], ctx: &FormatContext) -> FormatResult<Self> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        need(buf, 1)?;
        let version = buf[0];
        let mut pos = 1;

        if version == 0 {
            // Deprecated 1.10.0 encoding: strategy byte + threshold, and the
            // free-space manager addresses only for the "all persist" value.
            need(buf, pos + 1 + ss)?;
            let legacy_strategy = buf[pos];
            pos += 1;
            let threshold = read_uint(&buf[pos..], ss);
            pos += ss;

            // H5F_file_space_type_t -> H5F_fspace_strategy_t, per
            // H5O__fsinfo_decode's version-0 branch.
            let (strategy, persist, threshold) = match legacy_strategy {
                1 => (FileSpaceStrategy::FsmAggr, true, threshold),
                2 => (FileSpaceStrategy::FsmAggr, false, threshold),
                3 => (FileSpaceStrategy::Aggr, false, 1),
                4 => (FileSpaceStrategy::None, false, 1),
                other => {
                    return Err(FormatError::InvalidData(format!(
                        "invalid file-space strategy {other} in version-0 file-space info message"
                    )))
                }
            };

            // The six memory-type managers this encoding knows land in the
            // first six of the twelve slots; the rest stay undefined.
            let mut fs_addr = vec![crate::format::UNDEF_ADDR; FS_ADDR_COUNT_V1];
            if persist {
                need(buf, pos + FS_ADDR_COUNT_V0 * sa)?;
                for slot in fs_addr.iter_mut().take(FS_ADDR_COUNT_V0) {
                    *slot = read_uint(&buf[pos..], sa);
                    pos += sa;
                }
            }

            return Ok(Self {
                version: 0,
                strategy,
                persist,
                threshold,
                // Values the version-0 encoding does not carry, set to the
                // library defaults `H5O__fsinfo_decode` gives them. The one
                // that is not a constant is `eoa_pre_fsm_fsalloc`, which
                // upstream fills with the file's current end of allocation:
                // the field means "end of allocation before the managers were
                // laid out", and a file that never recorded one has no
                // narrower answer than the whole file.
                page_size: DEFAULT_FILE_SPACE_PAGE_SIZE,
                pgend_meta_thres: 0,
                eoa_pre_fsm_fsalloc: crate::format::UNDEF_ADDR,
                fs_addr,
            });
        }

        if version != 1 {
            return Err(FormatError::InvalidVersion(version));
        }

        need(buf, pos + 2 + ss + ss + 2 + sa)?;
        let strategy = FileSpaceStrategy::from_byte(buf[pos]);
        pos += 1;
        let persist = buf[pos] != 0;
        pos += 1;
        let threshold = read_uint(&buf[pos..], ss);
        pos += ss;
        let page_size = read_uint(&buf[pos..], ss);
        pos += ss;
        if page_size == 0 || page_size > PAGE_SIZE_MAX {
            return Err(FormatError::InvalidData(format!(
                "invalid file-space page size {page_size}"
            )));
        }
        let pgend_meta_thres = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let eoa_pre_fsm_fsalloc = read_uint(&buf[pos..], sa);
        pos += sa;

        let mut fs_addr = vec![crate::format::UNDEF_ADDR; FS_ADDR_COUNT_V1];
        if persist {
            need(buf, pos + FS_ADDR_COUNT_V1 * sa)?;
            for slot in fs_addr.iter_mut() {
                *slot = read_uint(&buf[pos..], sa);
                pos += sa;
            }
        }

        Ok(Self {
            version: 1,
            strategy,
            persist,
            threshold,
            page_size,
            pgend_meta_thres,
            eoa_pre_fsm_fsalloc,
            fs_addr,
        })
    }
    /// Encode the message body at the version it was decoded at
    /// (`H5O__fsinfo_encode`, plus the version-0 layout upstream only reads).
    ///
    /// Fails only for a version-0 message carrying something that encoding
    /// cannot express, which decode cannot produce and only a caller that
    /// edited the message can reach — see [`encode_v0`](Self::encode_v0).
    pub fn encode(&self, ctx: &FormatContext) -> FormatResult<Vec<u8>> {
        if self.version == 0 {
            return self.encode_v0(ctx);
        }
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        let mut buf = Vec::with_capacity(3 + 2 * ss + 2 + sa + FS_ADDR_COUNT_V1 * sa);
        buf.push(1);
        buf.push(self.strategy.to_byte());
        buf.push(self.persist as u8);
        buf.extend_from_slice(&self.threshold.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.page_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&self.pgend_meta_thres.to_le_bytes());
        buf.extend_from_slice(&self.eoa_pre_fsm_fsalloc.to_le_bytes()[..sa]);
        if self.persist {
            for addr in &self.fs_addr {
                buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
            }
        }
        Ok(buf)
    }

    /// Encode the deprecated 1.10.0 body, the inverse of what
    /// `H5O__fsinfo_decode`'s version-0 branch reads.
    ///
    /// Upstream has no such encoder: it decodes version 0, marks the message
    /// `mapped`, and rewrites it as version 1 on the next read-write open
    /// (H5Fsuper.c:866-880). This crate re-emits instead, so appending to a
    /// version-0 file does not silently change its on-disk format — but that
    /// is only sound while every field still fits, so the ones the encoding
    /// cannot hold are checked rather than assumed:
    ///
    /// * `strategy`/`persist` must be one of the four `H5F_file_space_type_t`
    ///   combinations; paged aggregation postdates this encoding.
    /// * `threshold` is stored only for the two `FSM_AGGR` values; the other
    ///   two are decoded as 1 and must still be 1.
    /// * the managers past the sixth must be undefined, there being no slot
    ///   for them.
    ///
    /// `page_size`, `pgend_meta_thres` and `eoa_pre_fsm_fsalloc` are *not*
    /// checked, because the version-0 encoding never carried them: decode
    /// synthesizes all three, and the next decode synthesizes them again from
    /// the same rule. Writing back the first two would be writing back
    /// constants; the third is re-derived from the file's end of allocation,
    /// which is the value it would have anyway.
    pub fn encode_v0(&self, ctx: &FormatContext) -> FormatResult<Vec<u8>> {
        let sa = ctx.sizeof_addr as usize;
        let ss = ctx.sizeof_size as usize;
        // The inverse of the version-0 branch's strategy mapping.
        let legacy = match (self.strategy, self.persist) {
            (FileSpaceStrategy::FsmAggr, true) => 1,
            (FileSpaceStrategy::FsmAggr, false) => 2,
            (FileSpaceStrategy::Aggr, false) => 3,
            (FileSpaceStrategy::None, false) => 4,
            (strategy, persist) => {
                return Err(FormatError::InvalidData(format!(
                    "strategy {strategy:?} with persist {persist} has no version-0 \
                     file-space info encoding"
                )))
            }
        };
        if legacy > 2 && self.threshold != 1 {
            return Err(FormatError::InvalidData(format!(
                "version-0 file-space strategy {legacy} cannot carry threshold {}",
                self.threshold
            )));
        }
        if self.fs_addr.len() > FS_ADDR_COUNT_V0
            && self.fs_addr[FS_ADDR_COUNT_V0..]
                .iter()
                .any(|&a| a != crate::format::UNDEF_ADDR)
        {
            return Err(FormatError::InvalidData(
                "a version-0 file-space info message has no slot for a page-type \
                 free-space manager"
                    .into(),
            ));
        }

        let mut buf = Vec::with_capacity(2 + ss + FS_ADDR_COUNT_V0 * sa);
        buf.push(0);
        buf.push(legacy);
        buf.extend_from_slice(&self.threshold.to_le_bytes()[..ss]);
        if self.persist {
            let undef = [crate::format::UNDEF_ADDR];
            for slot in 0..FS_ADDR_COUNT_V0 {
                let addr = self.fs_addr.get(slot).unwrap_or(&undef[0]);
                buf.extend_from_slice(&addr.to_le_bytes()[..sa]);
            }
        }
        Ok(buf)
    }
}

fn need(buf: &[u8], n: usize) -> FormatResult<()> {
    if buf.len() < n {
        Err(FormatError::BufferTooShort {
            needed: n,
            available: buf.len(),
        })
    } else {
        Ok(())
    }
}

// ======================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FormatContext {
        FormatContext::default_v3()
    }

    #[test]
    fn shmesg_roundtrip() {
        let mut buf = vec![0u8];
        buf.extend_from_slice(&88u64.to_le_bytes());
        buf.push(3);
        let m = SharedMessageTableMessage::decode(&buf, &ctx()).unwrap();
        assert_eq!(m.table_address, 88);
        assert_eq!(m.nindexes, 3);
    }

    /// The ten bytes `sohm_list.h5` carries in its superblock extension: the
    /// table at 88, one index.
    #[test]
    fn shmesg_encodes_the_fixture_body() {
        let m = SharedMessageTableMessage {
            version: 0,
            table_address: 88,
            nindexes: 1,
        };
        assert_eq!(
            m.encode(&ctx()),
            vec![0, 0x58, 0, 0, 0, 0, 0, 0, 0, 1],
            "H5O__shmesg_encode order: version, table address, index count"
        );
        assert_eq!(
            SharedMessageTableMessage::decode(&m.encode(&ctx()), &ctx()).unwrap(),
            m
        );
    }

    #[test]
    fn shmesg_rejects_bad_version() {
        let mut buf = vec![9u8];
        buf.extend_from_slice(&88u64.to_le_bytes());
        buf.push(1);
        assert!(matches!(
            SharedMessageTableMessage::decode(&buf, &ctx()).unwrap_err(),
            FormatError::InvalidVersion(9)
        ));
    }

    #[test]
    fn shmesg_rejects_short_buffer() {
        assert!(matches!(
            SharedMessageTableMessage::decode(&[0u8; 4], &ctx()).unwrap_err(),
            FormatError::BufferTooShort { .. }
        ));
    }

    #[test]
    fn btreek_field_order_matches_upstream() {
        // version, chunk K, snode K, sym leaf K — H5O__btreek_decode order.
        let buf = [0u8, 0x40, 0x00, 0x20, 0x00, 0x08, 0x00];
        let m = BtreeKMessage::decode(&buf).unwrap();
        assert_eq!(m.chunk_internal_k, 64);
        assert_eq!(m.snode_internal_k, 32);
        assert_eq!(m.sym_leaf_k, 8);
    }

    #[test]
    fn btreek_rejects_bad_version() {
        let buf = [1u8, 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            BtreeKMessage::decode(&buf).unwrap_err(),
            FormatError::InvalidVersion(1)
        ));
    }

    #[test]
    fn drvinfo_decodes_name_and_payload() {
        let mut buf = vec![0u8];
        buf.extend_from_slice(b"NCSAmult");
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&[1, 2, 3, 4]);
        let m = DriverInfoMessage::decode(&buf).unwrap();
        assert_eq!(m.name, "NCSAmult");
        assert_eq!(m.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn drvinfo_rejects_zero_length() {
        let mut buf = vec![0u8];
        buf.extend_from_slice(b"NCSAfami");
        buf.extend_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            DriverInfoMessage::decode(&buf).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    fn fsinfo_v1(persist: bool) -> Vec<u8> {
        // version 1, strategy PAGE (H5F_FSPACE_STRATEGY_PAGE == 1).
        let mut buf = vec![1u8, 1u8, persist as u8];
        buf.extend_from_slice(&1u64.to_le_bytes()); // threshold
        buf.extend_from_slice(&4096u64.to_le_bytes()); // page size
        buf.extend_from_slice(&0u16.to_le_bytes()); // pgend meta threshold
        buf.extend_from_slice(&0x1000u64.to_le_bytes()); // eoa_pre_fsm_fsalloc
        if persist {
            for i in 0..FS_ADDR_COUNT_V1 {
                buf.extend_from_slice(&((0x2000 + i as u64) * 8).to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn fsinfo_v1_paged_non_persisting() {
        let m = FileSpaceInfoMessage::decode(&fsinfo_v1(false), &ctx()).unwrap();
        assert_eq!(m.strategy, FileSpaceStrategy::Page);
        assert!(!m.persist);
        assert_eq!(m.page_size, 4096);
        assert!(m.fs_addr.iter().all(|&a| a == crate::format::UNDEF_ADDR));
    }

    #[test]
    fn fsinfo_v1_persisting_reads_twelve_addresses() {
        let m = FileSpaceInfoMessage::decode(&fsinfo_v1(true), &ctx()).unwrap();
        assert!(m.persist);
        assert_eq!(m.fs_addr.len(), FS_ADDR_COUNT_V1);
        assert_eq!(m.fs_addr[0], 0x2000 * 8);
        assert_eq!(m.eoa_pre_fsm_fsalloc, 0x1000);
    }

    #[test]
    fn fsinfo_v1_rejects_absurd_page_size() {
        let mut buf = fsinfo_v1(false);
        buf[11..19].copy_from_slice(&(PAGE_SIZE_MAX + 1).to_le_bytes());
        assert!(matches!(
            FileSpaceInfoMessage::decode(&buf, &ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    #[test]
    fn fsinfo_v0_all_persist_maps_onto_the_version_one_fields() {
        let mut buf = vec![0u8, 1u8];
        buf.extend_from_slice(&7u64.to_le_bytes()); // threshold
        for i in 0..FS_ADDR_COUNT_V0 {
            buf.extend_from_slice(&(0x100u64 + i as u64).to_le_bytes());
        }
        let m = FileSpaceInfoMessage::decode(&buf, &ctx()).unwrap();
        assert_eq!(m.version, 0);
        assert_eq!(m.strategy, FileSpaceStrategy::FsmAggr);
        assert!(m.persist);
        assert_eq!(m.threshold, 7);
        // Twelve slots whatever the encoding: the six the version-0 body
        // carries, then the page-type ones it has no room for.
        assert_eq!(m.fs_addr.len(), FS_ADDR_COUNT_V1);
        assert_eq!(m.fs_addr[5], 0x105);
        assert!(m.fs_addr[FS_ADDR_COUNT_V0..]
            .iter()
            .all(|&a| a == crate::format::UNDEF_ADDR));
    }

    #[test]
    fn fsinfo_v0_vfd_maps_to_none() {
        let mut buf = vec![0u8, 4u8];
        buf.extend_from_slice(&0u64.to_le_bytes());
        let m = FileSpaceInfoMessage::decode(&buf, &ctx()).unwrap();
        assert_eq!(m.strategy, FileSpaceStrategy::None);
        assert!(!m.persist);
        assert!(m.fs_addr.iter().all(|&a| a == crate::format::UNDEF_ADDR));
    }

    /// The four version-0 bodies, hand-built: every `H5F_file_space_type_t`
    /// value `H5O__fsinfo_decode` accepts. No writer produces these — libhdf5
    /// has emitted version 1 since 1.10.1 and h5py cannot ask for the older
    /// form at any `libver` bound — so a byte fixture is the only way to
    /// exercise the path a 1.10.0-written file takes.
    fn fsinfo_v0(legacy: u8, threshold: u64) -> Vec<u8> {
        let mut buf = vec![0u8, legacy];
        buf.extend_from_slice(&threshold.to_le_bytes());
        if legacy == 1 {
            for i in 0..FS_ADDR_COUNT_V0 {
                buf.extend_from_slice(&(0x400u64 + 0x40 * i as u64).to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn fsinfo_v0_round_trips_as_version_zero() {
        for (legacy, threshold) in [(1u8, 9u64), (2, 9), (3, 1), (4, 1)] {
            let bytes = fsinfo_v0(legacy, threshold);
            let m = FileSpaceInfoMessage::decode(&bytes, &ctx()).unwrap();
            assert_eq!(m.version, 0, "strategy {legacy}");
            assert_eq!(
                m.encode(&ctx()).unwrap(),
                bytes,
                "strategy {legacy} did not re-emit the body it was read from"
            );
        }
    }

    #[test]
    fn a_version_zero_message_re_emits_a_moved_manager() {
        // What an append does to the message: the managers move, and nothing
        // else about it changes.
        let mut m = FileSpaceInfoMessage::decode(&fsinfo_v0(1, 9), &ctx()).unwrap();
        m.fs_addr[0] = 0x2000;
        m.fs_addr[2] = 0x3000;
        m.eoa_pre_fsm_fsalloc = 0x9000;
        let bytes = m.encode(&ctx()).unwrap();
        assert_eq!(bytes.len(), fsinfo_v0(1, 9).len());
        let again = FileSpaceInfoMessage::decode(&bytes, &ctx()).unwrap();
        assert_eq!(again.fs_addr[0], 0x2000);
        assert_eq!(again.fs_addr[2], 0x3000);
        // The one field the older encoding cannot carry: decode synthesizes it
        // from the file's end of allocation, as `H5O__fsinfo_decode` does.
        assert_eq!(again.eoa_pre_fsm_fsalloc, crate::format::UNDEF_ADDR);
        assert_eq!(
            FileSpaceInfoMessage {
                eoa_pre_fsm_fsalloc: m.eoa_pre_fsm_fsalloc,
                ..again
            },
            m
        );
    }

    #[test]
    fn fsinfo_v0_refuses_what_it_cannot_encode() {
        let paged = FileSpaceInfoMessage::decode(&fsinfo_v1(true), &ctx()).unwrap();
        assert!(matches!(
            paged.encode_v0(&ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));

        let mut aggr = FileSpaceInfoMessage::decode(&fsinfo_v0(3, 1), &ctx()).unwrap();
        aggr.threshold = 64;
        assert!(matches!(
            aggr.encode(&ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));

        let mut paged_manager = FileSpaceInfoMessage::decode(&fsinfo_v0(1, 9), &ctx()).unwrap();
        paged_manager.fs_addr[FS_ADDR_COUNT_V0] = 0x800;
        assert!(matches!(
            paged_manager.encode(&ctx()).unwrap_err(),
            FormatError::InvalidData(_)
        ));
    }

    #[test]
    fn fsinfo_v1_round_trips() {
        for persist in [false, true] {
            let bytes = fsinfo_v1(persist);
            let m = FileSpaceInfoMessage::decode(&bytes, &ctx()).unwrap();
            assert_eq!(m.encode(&ctx()).unwrap(), bytes);
        }
    }

    #[test]
    fn fsinfo_rejects_unknown_version() {
        let buf = vec![9u8; 40];
        assert!(matches!(
            FileSpaceInfoMessage::decode(&buf, &ctx()).unwrap_err(),
            FormatError::InvalidVersion(9)
        ));
    }
}
