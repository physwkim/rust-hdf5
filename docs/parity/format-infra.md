# File-format infrastructure — C-parity inventory

Upstream: `/home/stevek/work/hdf5`, branch `develop` (2.0-dev), commit `eeba6ab8a5a62d170fb8921183760887401f615f` (2026-08-11).
Rust port: `/home/stevek/work/rust-hdf5`, commit `f08df756d4073c3e2ced3c01f6dfab94d3f0ced8` (main, v0.4.3).

Parity target is the **1.14.x on-disk format** (what libhdf5/h5py/pytables/netCDF4/MATLAB actually produce
today). Items confirmed 2.0-dev-only (`H5O_LAYOUT_VERSION_5`, datatype message v5/`H5T_COMPLEX`) are noted
inline and never count against the port. No other 2.0-dev-only wire divergence was found in the files this
slice covers (H5Fsuper*, H5O*, H5B*, H5G*, H5HL/H5HG/H5HF, H5SM*, H5FS*, H5MF*, H5D{btree,single,none,farray,
earray,btree2}.c, H5checksum.c) — the 1.14.x format applies unchanged unless a row says otherwise.

Rust code covered: `src/format/*.rs` (wire structs/codecs) and `src/io/{reader,writer,allocator}.rs` (the
actual byte-level read/write logic — `format/*.rs` frequently defines a struct's codec while `reader.rs`/
`writer.rs` decide whether and how it is actually invoked; both were checked for every row, and a
discrepancy between "codec exists" and "codec is ever called" is called out explicitly).

Status values: **Implemented** / **Partial** (what's missing) / **Missing** / **UNVERIFIED**. Impact:
**H** = files commonly produced by h5py/pytables/netCDF4/MATLAB become unreadable, corrupted, or silently
lose data; **M** = less common read paths or write-side fidelity; **L** = exotic/deprecated.

Method: nine parallel read-only research passes (one per subsystem below), each required to cite an exact
upstream `file:symbol` and an exact rust `file:line` per claim or mark `UNVERIFIED`. Three of the highest-
severity claims (superblock-extension address never dereferenced, `MSG_ATTR_INFO` never consumed, compound
datatype v2 padding condition) were independently re-verified against both repos before being promoted into
the top-10 list.

---

## 1. Superblock (v0/v1/v2/v3, driver info block, superblock extension)

Upstream: `H5Fsuper.c`, `H5Fsuper_cache.c`, version macros `H5Fprivate.h:300-305`.
Rust: `src/format/superblock.rs`, dispatch in `src/io/reader.rs` (`open_v0v1`/`RootGroupInfo`), construction
in `src/io/writer.rs`.

| # | Feature | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| SB.1 | Superblock v0 (original baseline) | `H5Fprivate.h:300` `HDF5_SUPERBLOCK_VERSION_DEF`; `H5Fsuper_cache.c:140` `H5F__superblock_prefix_decode` (v<2 body :427-514) | Implemented | Missing — writer hardcodes `SUPERBLOCK_V3` | `src/format/superblock.rs:406-527` (`SuperblockV0V1::decode`); `src/io/reader.rs:387,444-445` (`open_v0v1`); `src/io/writer.rs:7092-7093` (only write path, always v3) | H |
| SB.2 | Superblock v1 (indexed-storage internal-node K) | `H5Fprivate.h:301`; `H5Fsuper.c:1150-1151` (version bumped to 1 for non-default chunk-B-tree K); `H5Fsuper_cache.c:503-514` | Implemented | Missing | `src/format/superblock.rs:414,479-493` (`indexed_storage_k: Option<u16>` decoded); no v1 write path anywhere | M |
| SB.3 | Superblock v2 (compact, extension, no root symbol-table entry) | `H5Fprivate.h:302`; `H5Fsuper_cache.c:403` `H5F__cache_superblock_deserialize` (v≥2 body :582-627) | Implemented | Partial — encode is generic over version 2/3, but `write_superblock` never passes 2 | `src/format/superblock.rs:47-175` (`SuperblockV2V3` encode/decode, version param); `src/io/writer.rs:7093` (hardcoded `SUPERBLOCK_V3`) | M |
| SB.4 | Superblock v3 (adds SWMR-writer bit) | `H5Fprivate.h:303-305`; `H5Fsuper.c:1129-1131` (SWMR write forces v3); `H5Fpkg.h:52` `H5F_SUPER_SWMR_WRITE_ACCESS` | Implemented | Implemented | `src/format/superblock.rs:20,29` (`SUPERBLOCK_V3`, `FLAG_SWMR_WRITE`); `src/io/writer.rs:7093,7210` | H |
| SB.5 | Driver info block (`H5FD_MULTI`/`H5FD_FAMILY`) | `H5Fsuper_cache.c:66-72,103-118` (`H5AC_DRVRINFO`); `H5Fsuper.c:589-627` (read), `:216` `H5F__update_super_ext_driver_msg` (write) | Missing — address decoded, block content never fetched | Missing — no field/path exists on `SuperblockV2V3` | `src/format/superblock.rs:418,507` (`driver_info_address` decoded, never dereferenced); `rg -n "NCSAmulti\|NCSAfami\|driver_info_block" src/` → 0 hits | L |
| SB.6 | Superblock extension object header (create/locate/read its messages) | `H5Fsuper.c:89` `H5F__super_ext_create`; `:141` `H5F__super_ext_open` (called from `H5F__super_read` at `:666`); `:1311` (create on write) | Missing — address decoded, extension OH never opened | Missing — always written as `UNDEF_ADDR` | `src/format/superblock.rs:160,505` (field decoded, never consumed downstream); `src/io/writer.rs:967,7098` (`superblock_extension_address: UNDEF_ADDR`, no other write site); zero OH-open/create call for it in `reader.rs`/`writer.rs` | **H** |
| SB.7 | File consistency flags / checksum trailer (v2/v3) | `H5Fpkg.h:50-53`; `H5Fsuper_cache.c:403` (verify), `:707` `H5F__cache_superblock_serialize` (write) | Implemented | Implemented | `src/format/superblock.rs:22-29` (flag consts), `:96-98` (checksum computed+appended), `:141-155` (verified on decode); `src/format/checksum.rs` (Jenkins lookup3) | H |
| SB.8 | Root addressing: symbol-table entry (v0/v1) vs root-group OH address (v2/v3) | `H5Fsuper_cache.c:605-627`; `H5Groot.c:188` (`super_vers < 2` branch uses STE) | Implemented (both paths, via `RootGroupInfo` enum) | Partial — only the v2/v3 root-OH-address form is ever written | `src/format/superblock.rs:372-385,406-420`; `src/io/reader.rs:125-143,432-434,452-524`; `src/io/writer.rs:7092-7101` (only `root_group_object_header_address` ever set) | H |

**Notes.** SB.5 and SB.6 are exhaustively confirmed Missing on both read and write (repo-wide `rg` for the
block signatures and for any OH-open/create call site at those addresses — no hits). **SB.6 is the
structural root cause behind several other sections below**: because the superblock extension is never
opened, the v1-B-tree-K message (MSG.20), driver-info message (MSG.21), free-space-manager-info message
(MSG.24 / §6), and the SOHM table pointer (MSG.17 / §5) are all unreachable regardless of whether their own
codecs exist. Writing `superblock_extension_address = UNDEF_ADDR` is spec-legal (the extension is optional
at v2/v3), so rust never *falsely claims* an extension exists — this is a completeness gap, not a
corruption risk, for files rust itself writes.

---

## 2. Object header — structure (v1/v2, continuation, per-message flags, checksums)

Upstream: `H5Ocache.c` (`H5O__prefix_deserialize`, `H5O__chunk_deserialize/serialize`), `H5Opkg.h`.
Rust: `src/format/object_header.rs`, continuation-chunk handling in `src/io/reader.rs`.

| # | Feature | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| OH.1 | Object header v1 (variable prefix, 8-byte message alignment, no flags/times) | `H5Ocache.c:1095-1140` (v1 branch); `H5Opkg.h:58` `H5O_ALIGN_OLD` | Implemented | Missing — no v1 encoder anywhere | `src/format/object_header.rs:359-442` (`decode_v1`, alignment :414-421); zero `encode_v1` | L |
| OH.2 | Object header v2 (`OHDR` sig, flags incl. track-times/attr-crt-order/attr-phase-change/timestamps-present, optional times/thresholds, gap+checksum per chunk) | `H5Ocache.c:1007-1093` (v2 branch); `H5Opublic.h:61-66` (flag bits); `H5Opkg.h:85-106` | Partial — sig/version/flags/chunk0-size parsed; timestamps/thresholds skipped by correct byte count but discarded, never exposed | Partial — encode has conditional paths, but no writer call site ever sets those flag bits; `ObjectHeader::new()` hardcodes flags=0x02, so STORE_TIMES/ATTR_CRT_ORDER_TRACKED/ATTR_STORE_PHASE_CHANGE are never written | `src/format/object_header.rs:195-230` (parse+discard), `:269-284` (checksum verify); `:74-79` (`new()` hardcodes flags=0x02) | M |
| OH.3 | Object header continuation (msg 0x0010 CONT; v2's own `OCHK`-prefixed continuation chunk) — multi-chunk headers | `H5Ocache.c:1168-1240` `H5O__chunk_deserialize`, `:1417-1433` (CONT detection), `:1559-1619` (serialize) | Partial — follows CONT chains, parses bare-v1 and `OCHK`-prefixed v2 continuation blocks, but **the v2 branch strips the trailing 4-byte checksum without ever comparing it** | Missing — no code ever emits a CONT message or an `OCHK` block; a header that outgrows chunk 0 is a hard write error, not a spill into a continuation chunk | `src/io/reader.rs:1017-1088` (`read_object_header_full`), `:1094-1148` (`parse_continuation_block`, checksum stripped not verified at `:1100-1125`); `src/io/writer.rs:7126-7132` ("header grew … cannot rewrite in place") | **H** |
| OH.4 | Attribute creation-order tracking (v2 flag bit 2, 2-byte per-message index) | `H5Ocache.c:1299-1306`; `H5Opkg.h:113-126` | Partial — flag recognized, index bytes skipped structurally but discarded, no field to hold it | Missing — flag bit 2 never set by the writer | `src/format/object_header.rs:43,103-105,287,306-309` | L |
| OH.5 | Per-message flags byte (constant/shared/don't-share/fail-if-unknown-\*/mark-if-unknown/shareable) | `H5Oprivate.h:73-84`; `H5Ocache.c:1280-1287,1373-1399,1404-1407` | Partial — raw byte captured on both chunk0 and continuation decode, **but no bit is ever validated or acted on**: no fail-if-unknown enforcement, no shared-message (SOHM) dereferencing | Partial — byte round-tripped as given; only `CONSTANT` (0x01) is ever emitted (on datatype messages) | `src/format/object_header.rs:50-57,301-303`; `src/io/writer.rs:7415-7417`; `rg "SharedMessage\|SOHM" src/` → 0 hits | M |
| OH.6 | Checksum on v2 OH chunks (Jenkins lookup3, 4-byte trailer per chunk) | `H5checksum.c:365`; `H5Ocache.c:217-245` (chunk0 verify), `:624-649` (continuation verify), `:1608-1614` (write) | Partial — chunk0 verified before parsing; `OCHK` continuation trailer stripped but never verified (same gap as OH.3) | Implemented for chunk0; N/A for continuation chunks (never written) | `src/format/object_header.rs:175-177,269-284`; `src/io/reader.rs:1102-1105` | M |

**Notes.** OH.3 is one of the highest-severity findings in this slice: any real HDF5 file whose object
header spans multiple chunks (routine once an object accumulates enough attributes/links to exceed chunk 0)
has its continuation-chunk integrity **silently unverified** by rust — a bit-flipped or truncated
continuation chunk is accepted rather than rejected. It does not corrupt anything rust itself writes (rust
never emits continuation chunks), but it is a read-side safety hole against arbitrary upstream-written
files.

---

## 3. Object header messages — the H5O message-class table

Upstream ground truth: `H5Opkg.h:409` `H5O_msg_class_g[H5O_MSG_TYPES=27]`, one row per class
`H5O_MSG_NULL` … `H5O_MSG_DELETED` (`H5Opkg.h:427-534`). 25 real on-disk message types (0x0000-0x0018) plus
two in-memory-only placeholders (`UNKNOWN`=0x0019, `DELETED`=0x001a). Rust constants: `src/format/messages/
mod.rs:14-29`.

### 3a. Core data messages (dataspace/datatype/layout/filter/fill/attribute/…)

| # | Message (type ID) | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| MSG.1 | NULL (0x0000) | `H5Onull.c:28-49` (opaque gap-filler) | Implemented — v1 skips type-0; v2 stores it, transparently ignored downstream | Partial — never explicitly constructed as padding | `src/format/object_header.rs:424` (v1 skip), `:321-325` (v2 push) | L |
| MSG.2 | Dataspace (0x0001), v1/v2, SCALAR/SIMPLE/NULL | `H5Osdspace.c:104` decode, `:249` encode | Partial — SCALAR/SIMPLE round-trip; **H5S_NULL unsupported**, no class field on the struct, v2 decode reads-then-discards the type byte | Partial — type derived purely from ndims==0 vs >0; no way to emit H5S_NULL | `src/format/messages/dataspace.rs:22-29` (no class field), `:118` (`let _ds_type = buf[3];`) | M |
| MSG.3 | Datatype (0x0003) — dispatch overview | `H5Odtype.c:119/955` (helpers), `:1493/1567` (msg decode/encode) | Partial — upstream accepts version 1-5 uniformly; rust has **no class entries for Time(2)/Bitfield(4)/Opaque(5)/Reference(7)/Complex(11, 2.0-dev-only)** | Partial — missing classes can't be constructed | `src/format/messages/datatype.rs:33-39` (`CLASS_*` consts — only 0,1,3,6,8,9,10 defined); `:1021-1025` (`UnsupportedFeature` fallback) | H |
| MSG.3a | Datatype: fixed-point/integer | `H5Odtype.c:167-187/974-1044` | Implemented (v1-3) | Implemented (always v1) | `datatype.rs:703-736,386-419` | L |
| MSG.3b | Datatype: float, incl. VAX order | `H5Odtype.c:189-273/1046-1149` | Partial — IEEE LE/BE correct; **VAX order (flag bit 0x40) silently misread as big-endian**, no VAX variant | Partial — VAX cannot be constructed | `datatype.rs:737-778,42-46,420-466` | L |
| MSG.3c | Datatype: string (fixed + vlen) | `H5Odtype.c:285-297,757-786` | Fixed: Implemented. Vlen: Partial — pad-type nibble (NULLPAD/SPACEPAD) never extracted, struct has no field | Fixed: Implemented. Vlen: Partial — always hardcodes NULLTERM | `datatype.rs:779-796,467-489,113-117,938-964,568-602` | M |
| MSG.3d | Datatype: bitfield + opaque (classes 4, 5) | `H5Odtype.c:299-343/1185-1259` | Missing — no enum variant, falls to `UnsupportedFeature` | Missing | `datatype.rs:33-39,1021-1025` | M |
| MSG.3e | Datatype: compound, incl. legacy member packing | `H5Odtype.c:345-635/1261-1331` | **Partial — real bug**: member-name 8-byte padding applied only `if version == 1`; upstream pads for v1 **and** v2 (`H5Odtype.c:414-428`, only v3 drops padding). A genuine v2 compound (used whenever a member is `H5T_ARRAY`, or produced under an older libver bound) is mis-parsed — every subsequent member's name/offset desyncs. Independently re-verified against `H5Odtype.c:414-451` in this session. | Partial — always emits v3 only, so the read bug is upstream-file-only | `datatype.rs:826-832` (`if version == 1`, should be `version < 3`) vs `H5Odtype.c:415,451` | **H** |
| MSG.3f | Datatype: reference (object + region, old and revised) | `H5Odtype.c:637-672/1333-1337`; revised forms need msg v4 | Missing — class 7 entirely absent; decode failure is **swallowed silently** one layer up (`if let Ok(...)`), so a reference-typed dataset/attribute vanishes from the catalog instead of erroring | Missing | `datatype.rs:33-39,1021-1025`; `src/io/reader.rs:1170-1220` (silent drop) | **H** |
| MSG.3g | Datatype: enum | `H5Odtype.c:674-755/1339-1373` | Implemented (v1-3, incl. v1/v2-pads-to-8 vs v3-no-pad) | Implemented (always v1) | `datatype.rs:877-937,528-567` | L |
| MSG.3h | Datatype: vlen-sequence + array | `H5Odtype.c:757-863/1375-1426` | Array: Implemented. **Vlen-sequence: Partial — rejects any version other than exactly 1**, but upstream legitimately bumps a vlen's version to match its nested base (e.g. base is array or v3 compound) | Array: Implemented (always v3). Vlen: Partial — always hardcodes v1 regardless of base's actual version, can violate parent-version-≥-child-version | `datatype.rs:938-963` (`version != DT_VERSION`), `:965-1021`, `:603-631` | **H** |
| MSG.4 | Old Fill Value (0x0004) | `H5Ofill.c:328/502`; emitted only for `H5F_LIBVER_EARLIEST`-bound files | Missing — constant defined, never matched | Missing — only 0x0005 ever built | `src/format/messages/mod.rs:17`; `src/io/reader.rs:1204` `_ => {}`; `src/io/writer.rs:7419-7441` | M |
| MSG.5 | New Fill Value (0x0005), v1-3 incl. undefined/have-value flags | `H5Ofill.c:189/409` | Implemented | Implemented (always v3) | `src/format/messages/fill_value.rs:84,110,125,183` | L |
| MSG.6 | Data Layout (0x0008), v3/4/5 overview | `H5Olayout.c:83/592-593` | Implemented (v3 compact/contiguous/chunked-v1btree; v4/5 chunked incl. FixedArray/ExtensibleArray/BTreeV2) | Partial (see 6a-6c) | `src/format/messages/data_layout.rs:454,328`; `src/io/reader.rs:1180-1184` | L (default path) |
| MSG.6a | Layout: v1-B-tree index & Single-Chunk index | `H5Olayout.c:142-155/637-649` (v1-Btree); `:392-409/678-684` (Single-Chunk) | Implemented for both | **Missing for both on write** — codec constructors `chunked_v3_btree_v1`/`chunked_v4_single` exist but writer.rs never calls either; writer always upgrades to v4 Fixed/Extensible-Array/BTreeV2 | `data_layout.rs:503,234,148-155,312`; `src/io/writer.rs:1010` (`chunk_layout_version` only returns 4/5) | L |
| MSG.6b | Layout: Implicit index | `H5Olayout.c:388-390/675-676` (auto-selected: no filters, `alloc_time==EARLY`) | **Missing — hard errors on read** | Missing — writer never sets `alloc_time=EARLY` | `data_layout.rs:53` (parses structurally); `src/io/reader.rs:1810-1813` (`_ => Err(...)`) | **H** (see also CI.3, §8) |
| MSG.6c | Legacy layout message v1/v2 (`H5F_LIBVER_EARLIEST` files) | `H5Olayout.c:106-217/608` | Missing — decode only matches v3/4/5 | Missing — encoder has no libver-bound concept | `data_layout.rs:767` (`InvalidVersion`); `src/io/reader.rs:1181` (swallowed) | M |
| MSG.7 | Filter Pipeline (0x000b), v1/2 overview | `H5Opline.c:109/267` | Partial — id/flags/cd_values preserved; filter *name string* for user-defined IDs (≥256) parsed then discarded (struct has no name field) | Partial — v2 framing correct, name never encoded for id≥256; v1 never written | `src/format/messages/filter.rs:60-68,289,373-386,251,263-268` | M |
| MSG.7a | Standard filters: deflate/shuffle/fletcher32/szip/nbit/scaleoffset | `H5Zdeflate.c`/`H5Zshuffle.c`/`H5Zfletcher32.c`/`H5Zszip.c`/`H5Znbit.c`/`H5Zscaleoffset.c` | deflate/shuffle/szip/nbit: Implemented both directions. **fletcher32: Partial — checksum trailer stripped, never recomputed/compared.** scaleoffset: decompress only | deflate/shuffle/szip/nbit/fletcher32: Implemented. **scaleoffset: Missing — compress direction explicitly unimplemented** | `filter.rs:516-575,576-619,625-627,634-642`; `src/format/nbit_scaleoffset.rs:648,928` (no forward scaleoffset) | M |
| MSG.8 | Attribute (0x000c) | `H5Oattr.c:120/330` | **Partial — compact v1/2/3 decoded; dense/fractal-heap storage (Attribute Info 0x0015) never read at all — objects with >8 attributes (h5py's own default max-compact threshold) silently yield zero attributes, no error** | Partial — only compact v3 ever emitted, non-shared | `src/format/messages/attribute.rs:152-264,99-144`; `src/format/messages/mod.rs:28` (`MSG_ATTR_INFO`, zero other refs — confirmed); `src/io/reader.rs:419-425` (no dense fallback) | **H** |
| MSG.9 | Object name/comment (0x000d) | `H5Oname.c:74/114` | Missing | Missing | no `MSG_NAME` constant anywhere; `rg` negative | L |
| MSG.10 | Modification time — old (0x000e) + new (0x0012) | `H5Omtime.c:168/260` (old); `:110/224` (new) | Missing (both) | Missing (both) | `src/format/messages/mod_time.rs:1` (entire file is a stub comment, no code); OH v2 embedded-timestamps path also stubbed (see OH.2) | M |
| MSG.11 | Object Header Continuation (0x0010) | `H5Ocont.c:79/128` | Implemented | Missing | `src/io/reader.rs:1051-1109` (follows chains, cycle-guarded, bounded); `MSG_OBJ_HEADER_CONTINUATION` zero refs in writer.rs | M |
| MSG.12 | External File List (0x0007) | `H5Oefl.c:74/209` | Missing | Missing | no `MSG_EFL` constant; no `efl.rs`; `rg` negative | M |

### 3b. Link / group / superblock-extension / misc messages

| # | Message (type ID) | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| MSG.13 | Link Information (0x0002) | `H5Olinfo.c:102/192` | Partial — decoded, fractal-heap addr drives dense-link discovery; `max_creation_order`/`creation_order_btree_address` decoded but never consulted | Partial — only `LinkInfoMessage::compact()` ever emitted | `src/format/messages/link_info.rs:87,56`; `src/io/reader.rs:625-629`; `src/io/writer.rs:7509-7511,7580-7582` | M |
| MSG.14 | Link (0x0006) | `H5Olink.c:104/290` | Partial — hard links fully surfaced; **soft links decoded then silently dropped by the group-walk**; external/UD links (type ≥64) rejected with `UnsupportedFeature` and silently skipped by caller | Partial/Missing — only hard links ever constructed; `HardLinkTarget` has only Dataset/Group variants, no soft/external/UD constructor | `src/format/messages/link.rs:128,213-218`; `src/io/reader.rs:634-635` (soft-link discard); `link.rs:49,68` | M |
| MSG.15 | Group Information (0x000a) | `H5Oginfo.c:85/162` | Partial — used only as a group-classifier marker; `decode()` never actually called from reader.rs | Partial — always `GroupInfoMessage::default()` (flags=0); no threshold ever set → compact→dense promotion never triggers (see GRP.6) | `src/format/messages/mod.rs:21`; `src/io/reader.rs:665,898` (marker use only); `src/io/writer.rs:7514-7516,7585-7587` | L |
| MSG.16 | Shared Message table info (0x000f) | `H5Oshmesg.c:68/117` | Missing — no `MSG_SHMESG` module; blocked by SB.6 | Missing — blocked by SB.6 | `rg "shmesg\|SHMESG" src/` → no dedicated module | L (gated behind §5 SOHM, itself M/H) |
| MSG.17 | Symbol Table (0x0011) | `H5Ostab.c:84/126` | Implemented — field layout matches upstream exactly, resolves v0/v1 root/old-style subgroups | Missing — never emitted; every group is unconditionally new-style | `src/io/reader.rs:541-551,501,941`; `src/format/symbol_table.rs:33`; zero `MSG_SYMBOL_TABLE` write sites | L |
| MSG.18 | v1 B-tree 'K' values (0x0013) | `H5Obtreek.c:71/124` | Missing — const defined, never referenced again; blocked by SB.6 | Missing — no encode fn/struct | `src/format/messages/mod.rs:27` (sole occurrence in crate) | L |
| MSG.19 | Driver info (0x0014) | `H5Odrvinfo.c:71/132` | Missing — no module; blocked by SB.6 | Missing — `SuperblockV2V3` has no such field | `src/format/superblock.rs:418,507` (only the legacy v0/v1 address field exists, unconsumed) | L |
| MSG.20 | Attribute Info (0x0015) | `H5Oainfo.c:93/185` | **Missing — const defined, never matched, falls to catch-all**; same root cause as MSG.8's dense-attribute gap | Missing | `src/format/messages/mod.rs:28`; `src/io/reader.rs:1169-1206` (`_ => {}` at :1204) | **H** (duplicate of MSG.8) |
| MSG.21 | Reference Count (0x0016) | `H5Orefcount.c:82/130` | **Missing on read** — falls to catch-all; even the v1-header `obj_ref_count` prefix is parsed and explicitly discarded | **Implemented on write** — `encode_refcount()` invoked on hard-link creation, gated on `rc > 1`, matching upstream's "only when refcount != 1" rule | decode: absent (reader.rs catch-all); `src/format/object_header.rs:376` (discard); `src/io/writer.rs:853-861,7404,7495-7499,7568-7570` | M |
| MSG.22 | Free-space Manager Info (0x0017) | `H5Ofsinfo.c:90/230` | Missing — no constant; blocked by SB.6 | Missing — allocator is documented session-only/non-persistent | no "0x17"/FSINFO hits in `src/`; `src/io/allocator.rs:11-30` (non-persistence doc) | L (see §6, mostly M there) |
| MSG.23 | Metadata Cache Image (0x0018) | `H5Ocache_image.c:88/146` | Implemented in a degenerate/harmless sense — no dedicated logic, but the generic message store round-trips unknown bytes safely and the dispatch's `_ => {}` skips it; moot since the extension it lives in (SB.6) is never opened | Missing (expected — pure perf optimization, upstream itself treats absence as normal) | `src/format/object_header.rs:291-326` (unvalidated generic store) | L |
| MSG.24/25 | Unknown (0x0019) / Deleted (0x001a) placeholders + unrecognized-ID dispatch | `H5Ounknown.c`/`H5Odeleted.c`; flag-driven fail/mark semantics in `H5Ocache.c:1352-1399` | Rust has **no exhaustive dispatch table** analogous to `H5O_msg_class_g[27]`: every byte 0x00-0xFF is stored generically regardless of type; two `match msg.msg_type` blocks (reader.rs, writer.rs) cover only ~10 of the 25 real IDs, everything else — recognized-but-unimplemented or genuinely future — is silently skipped via `_ => {}`, never an error or a per-flag fail/mark decision | N/A — rust never constructs an unknown/deleted sentinel | `src/io/reader.rs:1169-1206` (:1204 catch-all); `src/io/writer.rs:1247-1284` (:1282 catch-all); `src/format/messages/mod.rs:14-29` (only 10 of 25 IDs have a named constant with a live consumer) | L |

**Notes.** MSG.8 and MSG.20 are the same defect from two angles: the Attribute Info message (0x0015) that
points at dense/fractal-heap attribute storage is never decoded anywhere in `reader.rs` (confirmed:
`MSG_ATTR_INFO` has exactly one reference in the whole crate — its own `const` definition). Any object with
more attributes than h5py's default `max_compact` threshold (8) silently reports **zero** attributes on
read. This is corroborated independently by the B-tree agent (§4, items BT2.12/BT2.13) and the fractal-heap
agent (§4 Heaps, HEAP.11) from three unrelated code paths, and re-verified directly in this session
(`rg -n MSG_ATTR_INFO src/` → 1 hit total).

---

## 4. B-trees — v1 (`H5B.c`) and v2 (`H5B2*.c`, all record types)

Rust: `src/format/btree_v1.rs` (v1); `src/format/chunk_index/btree_v2.rs` (v2 nodes + the 2 record types
rust implements). Ground truth for v2 record types: `H5B2private.h:38-58` `H5B2_subid_t`, 11 real registered
subtypes (2 more, `H5B2_TEST_ID`/`H5B2_TEST2_ID`, are test-only and excluded).

| # | Feature | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| BT1.1 | v1 B-tree node structure (`TREE` sig, type, level, entries_used, sibling addrs) | `H5B.c`; `H5Bcache.c:139-229`; `H5Fprivate.h:372` (`H5B_MAGIC`) | Implemented | **Missing — no `encode()` exists anywhere in the crate** (only test-only builders) | `src/format/btree_v1.rs:60-134,172-244`; encode absent | H |
| BT1.2 | v1 B-tree type 0 (group/symbol-table, keys = local-heap offsets) | `H5Gnode.c:90` (`H5B_SNODE` class) | Implemented | Missing | `src/io/reader.rs:819-963,966-1000`; `src/format/symbol_table.rs:33`; writer never emits `MSG_SYMBOL_TABLE` | H |
| BT1.3 | v1 B-tree type 1 (chunk offset + filter_mask + size, legacy layout v1-3) | `H5Dbtree.c:169` (`H5B_BTREE` class) | Implemented | Missing — `chunk_layout_version()` only ever returns 4 or 5 | `src/format/btree_v1.rs:172-244`; `src/io/reader.rs:2250-2354,2364-2410`; `src/io/writer.rs:1010-1023` | M |
| BT1.4 | v1 B-tree checksum (must be absent — pre-dates format checksumming) | `H5Bcache.c:139-229` — no checksum field | Implemented correctly (no checksum expected or required) | N/A (no writer) | `src/format/btree_v1.rs` — no checksum code anywhere in the file | L |
| BT1.5 | v1 B-tree K-value / split control honored | `H5Obtreek.c` (msg 0x0013); `H5B.c:449,471` (`shared->two_k` split bound); `H5Bcache.c:184` | Partial — `sym_leaf_k`/`btree_internal_k`/`indexed_storage_k` decoded from superblock but never read again anywhere; no `entries_used > 2K` corruption bound-check | Missing — no v1 writer, so no split/K logic at all | `src/format/superblock.rs:412-414,464-467,479-493` (decoded, unread elsewhere) | L |
| BT2.6 | v2 B-tree header (`BTHD`): version/type/node_size/record_size/depth/split-merge%/root-ptr/num-records | `H5B2hdr.c`; `H5B2cache.c:214-306` | Implemented | Implemented | `src/format/chunk_index/btree_v2.rs:128-146,180-296` (checksummed) | H |
| BT2.7 | v2 B-tree internal node (`BTIN`), checksummed | `H5B2int.c`; `H5B2cache.c:584-701` | Implemented | Implemented | `btree_v2.rs:427-585` | H |
| BT2.8 | v2 B-tree leaf node (`BTLF`), checksummed | `H5B2leaf.c`; `H5B2cache.c:984-1076` | Implemented | Implemented | `btree_v2.rs:318-396` | H |
| BT2.9 | Record type: fractal-heap huge-object **indirect** tracking (`H5HF_HUGE_BT2_INDIR`/`_FILT_INDIR`) | `H5B2private.h:40-43`; `H5HFbtree2.c:103,118` | Missing | Missing | `src/format/fractal_heap.rs:36-64,135-138` (`huge_bt2_addr` parsed then discarded) | M |
| BT2.10 | Record type: dense-group-link **name** index (`H5G_BT2_NAME`) | `H5B2private.h:46`; `H5Gbtree2.c:92` | Partial — no B2-record decoder exists; `read_dense_links` bypasses the index entirely via a linear fractal-heap scan, recovering data for common cases without ever touching this type | Missing | `src/format/chunk_index/btree_v2.rs` (only implements chunk-index record types); `src/io/reader.rs:726-769` | **H** |
| BT2.11 | Record type: dense-group-link **creation-order** index (`H5G_BT2_CORDER`) | `H5B2private.h:47-48`; `H5Gbtree2.c:107` | **Missing — `read_dense_links` returns links in fractal-heap physical order, not creation order**, no way to reconstruct it | Missing | `src/io/reader.rs:726-769` | **H** |
| BT2.12 | Record type: dense-attribute **name** index (`H5A_BT2_NAME`) | `H5B2private.h:50-51`; `H5Abtree2.c:94` | Missing — same root cause as MSG.8/MSG.20 | Missing | `src/format/messages/mod.rs:28` (`MSG_ATTR_INFO`, never decoded) | **H** |
| BT2.13 | Record type: dense-attribute **creation-order** index (`H5A_BT2_CORDER`) | `H5B2private.h:52-53`; `H5Abtree2.c:109` | Missing — same root cause | Missing | same as BT2.12 | **H** |
| BT2.14 | Record type: chunked dataset, non-filtered, >1 unlimited dim (`H5D_BT2`) | `H5B2private.h:54`; `H5Dbtree2.c:194` | Implemented | Implemented | `btree_v2.rs:47,132-146,1146-1195`; `src/io/reader.rs:2057-2150`; `src/io/writer.rs:5521-5610` | H |
| BT2.15 | Record type: chunked dataset, filtered, >1 unlimited dim (`H5D_BT2_FILT`) | `H5B2private.h:55`; `H5Dbtree2.c:209` | Implemented | Implemented | `btree_v2.rs:49,152-167` | H |
| BT2.16 | Record type: SOHM index (`H5SM_INDEX`) | `H5B2private.h:49`; `H5SMbtree2.c:51` | **Missing — and worse than absent**: the per-message "shared" flag bit is stored but never inspected on read (see OH.5), so a SOHM-enabled file has shared messages **misdecoded as literal content**, not merely un-optimized | Missing | zero `SOHM`/`H5SM` hits anywhere in `src/` | **H** |
| BT2.17 | Fractal-heap huge-object **direct**-block tracking (`H5HF_HUGE_BT2_DIR`/`_FILT_DIR`) | `H5B2private.h:44-45`; `H5HFbtree2.c:133,148` | Missing | Missing | same evidence as BT2.9 | M |
| BT2.18 | v2 B-tree checksum (Jenkins lookup3) on all 3 node types | `H5B2cache.c` (verify+serialize in each `*_deserialize`/`*_serialize`) | Implemented | Implemented | `src/format/checksum.rs`; applied at all encode/decode sites in `btree_v2.rs` | H |

**Notes.** v1-B-tree write support is entirely absent because the writer only ever emits superblock v3,
new-style link storage, and v4/v5 chunk layouts — never v0/v1 superblocks, SNOD/symbol-table groups, or
legacy chunked layout v3 (BT1.1-BT1.3). SOHM's list→B-tree conversion (`H5SM.c:682`) reuses the same
`H5SM_INDEX` class, not a distinct wire subtype, so it is folded into BT2.16 rather than broken out.

---

## 5. Shared Object Header Messages (SOHM)

Upstream: `H5SM.c`, `H5SMbtree2.c`, `H5SMcache.c`, `H5SMmessage.c`. Ground truth search
(`rg -in 'sohm|shared.?message' src/`) returns **zero matches** anywhere in the port.

| # | Feature | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| SOHM.1 | SOHM master table decode/encode | `H5SM.c:H5SM_init`; `H5SMcache.c` `H5SM_TABLE_MAGIC` | Missing | Missing | blocked by SB.6 (`superblock_extension_address` always `UNDEF_ADDR`, never followed) — `src/io/writer.rs:967,7098` | M |
| SOHM.2 | SOHM index — "list" form | `H5SMcache.c` `H5SM_LIST_MAGIC`; `H5SM.c:610` `H5SM__create_list` | Missing | Missing | no list-decode code anywhere in `src/` | M |
| SOHM.3 | SOHM index — "btree" form + list→btree threshold | `H5SMbtree2.c:51` (`H5SM_INDEX`); `H5SM.c:682,120,145-147` | Missing | Missing | see BT2.16 | M |
| SOHM.4 | Shared-message hashing/dedup on write (flag 0x02) | `H5SM.c:1041` `H5SM_try_share`; `:1224` `H5SM__write_mesg` | N/A | Missing — rust always duplicates messages per object header, never sets flag 0x02 anywhere | `rg -n 'flags\s*=\s*0x02' src/io/writer.rs src/format/messages/*.rs` → 0 matches | M |
| SOHM.5 | Reading a message with the shared flag (0x02) set | `H5Oprivate.h:74` `H5O_MSG_FLAG_SHARED`; `H5Oshared.c:289` `H5O__shared_decode` | **Missing, silently** — `msg.flags` captured but never tested against 0x02 anywhere; shared-message-pointer bytes are fed straight into the normal per-type decoder, whose failures are swallowed by `if let Ok(...)` call sites | N/A | `src/io/reader.rs:1168-1201` (no flag check; decode-failure swallow) | M |

**Notes.** SOHM is opt-in upstream (not enabled by h5py by default), so the baseline impact of "feature
absent" is M rather than H. SOHM.5 is the more dangerous half: a file that *does* use SOHM doesn't just miss
an optimization, it can hand a shared-message pointer's raw bytes to a datatype/dataspace/attribute decoder
that has no idea it's looking at a pointer, not a payload — the failure mode is call-site-dependent (some
silently drop the field, none currently produce wrong-but-plausible data in the traced paths, but this was
not exhaustively fuzzed).

---

## 6. Free-space manager (H5FS) + file-space strategies incl. paged aggregation

Upstream: `H5FS.c`, `H5FScache.c`, `H5FSsection.c`; strategies in `H5Fpublic.h:196-201`; `H5MFaggr.c`,
`H5MF.c`, `H5MFsection.c`. Rust allocator: `src/io/allocator.rs` — read in full: a pure in-process
bump-the-EOF allocator with an **in-memory-only** free list, never serialized to disk.

| # | Feature | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| FS.6 | Free-space manager header (`FSHD`) | `H5FScache.c:242,695` `H5FS_HDR_MAGIC` | Missing | Missing | `rg -in 'H5FS\|free.?space.?manager\|FreeSpace' src/` → only a comment in `allocator.rs:30` | M |
| FS.7 | Free-space serialized sections (`FSSE`) | `H5FScache.c:953,1221` `H5FS_SINFO_MAGIC` | Missing | Missing | no section-block (de)serialization code anywhere | M |
| FS.8 | Free-space section classes (`H5MF_FSPACE_SECT_CLS_*` / `H5HF_FSPACE_SECT_CLS_*`) | `H5MFsection.c:81,107,133`; `H5HFsection.c:154,183,209,238` | Missing | Missing | `src/io/allocator.rs:1-442` — pure bump allocator + in-memory `Vec<FreeBlock>`, no on-disk section machinery at all | M |
| FS.9 | Strategy `FSM_AGGR` (upstream default: FSM + small aggregators) | `H5Fpublic.h:196` | N/A | Not implemented — only the in-memory half exists | `allocator.rs:25-30` (explicit comment: session-only, no persisted FSM) | M |
| FS.10 | Strategy `AGGR` (aggregators only, no persisted FSM) | `H5Fpublic.h:200` | N/A | Partial — architecturally closest match, but never recorded via an FSINFO message | same as FS.9 | M |
| FS.11 | Strategy `PAGE` (paged aggregation) | `H5Fpublic.h:198`; `H5MFaggr.c` page-alignment logic | Missing | Missing | `rg -in 'fsinfo\|paged.?aggregation\|page.?buffer' src/` → 0 hits; allocator has one fixed 8-byte alignment, no page-size concept | M |
| FS.12 | Strategy `NONE` (no cross-close reuse) | `H5Fpublic.h:201` | N/A | Effectively matches on close/reopen (space never reclaimed across sessions), but **stronger within one open session** (rust does reuse freed blocks in-process, which real NONE would not) | `allocator.rs` free_list is session-only in-memory state | M |
| FS.13 | Persisting free space across close/reopen (fsinfo `persist` flag) | `H5Ofsinfo.c:121,178,245`; default `H5F_FREE_SPACE_PERSIST_DEF = false` | Missing | Missing — no FSINFO message ever written (blocked by SB.6) | `src/io/writer.rs:967,7098` (`superblock_extension_address` always `UNDEF_ADDR`) | M |

**Notes.** Rust never *falsely claims* a persisted strategy — `superblock_extension_address` is always
`UNDEF_ADDR`, which is spec-legal, so this is a completeness/size-bloat gap (files never shrink or reclaim
space across a close+reopen cycle by a *different* rust-hdf5 process — within one open `File` handle, space
*is* reused via the in-memory free list) rather than a corruption risk. All 8 rows are M, not H, per the
rubric: no evidence rust writes a superblock/fsinfo that claims a strategy or persistence it doesn't
implement.

---

## 7. Groups: symbol table (v1) vs v2 link messages vs dense link storage

Upstream: `H5Gstab.c`/`H5Gnode.c`/`H5Gent.c` (v1); `H5Gdense.c`/`H5Gbtree2.c` (v2 dense). Rust:
`src/format/symbol_table.rs`, `src/format/messages/{link,link_info,group_info}.rs`.

| # | Feature | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| GRP.1 | Symbol table entry decode/encode (`H5G_entry_t`, cache types 0/1/2) | `H5Gent.c` (`H5G_ent_decode/encode`); cache types `H5Gpkg.h:65-67` | **Partial — cache_type 2 (symbolic-link scratch pad, `lval_offset`) not decoded at all; v1 soft links are silently dropped, not just unsupported** | Missing — no encode fn | `src/format/superblock.rs:373-385` (no lval_offset field), `:530-573` (`decode_symbol_table_entry`, cache_type==1 only) | M |
| GRP.2 | Symbol table node (`SNOD`) decode/encode | `H5Gnode.c` | Implemented | Missing — no encode fn | `src/format/symbol_table.rs:1-73` (decode only) | L |
| GRP.3 | Group traversal: v1 B-tree(type 0) + local heap + SNOD end-to-end | `H5Gstab.c` `H5G__stab_insert/lookup`; `H5Gnode.c` `H5G__node_found/insert` | Implemented for hard links/datasets/subgroups; soft links dropped per GRP.1 | Missing — no v1 group creation path anywhere | `src/io/reader.rs:444-537,819-925,966` | M |
| GRP.4 | Creating a new v1-style group on write | `H5Gstab.c` `H5G__stab_create` | N/A | Missing — writer always emits v2/link-message groups; `open_append` explicitly rejects v0/v1-superblock files | `src/io/writer.rs:1146-1164` (rejection); zero SNOD/local-heap/btree-v1 hits in writer.rs | L |
| GRP.5 | Compact link storage (inline Link msgs, below max-compact threshold) | `H5Oginfo.c`/`H5Gobj.c` | Implemented | Implemented | `src/io/reader.rs:592-718`; `src/io/writer.rs:7508-7627` | L |
| GRP.6 | Compact→dense promotion on write (max-compact/min-dense threshold) | `H5Gobj.c` | N/A | **Missing — `GroupInfoMessage::default()` used unconditionally; every group is written compact regardless of link count** | `src/io/writer.rs:7514,7585` (only 2 occurrences, both defaults) | L |
| GRP.7 | Dense link storage: fractal heap + Link Info (0x0002) | `H5Gdense.c` `H5G__dense_insert/lookup` | Implemented | Missing (see GRP.6) | `src/io/reader.rs:726-769` (`read_dense_links`); regression test at `:3843` | L |
| GRP.8 | Dense link storage: v2-B-tree **name** index (`H5G_BT2_NAME`) | `H5Gbtree2.c:92` | Missing — no code parses/traverses this B-tree; enumeration still works via linear fractal-heap scan (GRP.7), but no indexed name lookup | Missing (dense storage never written) | `src/io/reader.rs:726-745` (no `name_btree_address` param); `src/format/messages/link_info.rs:27-28` (field decoded, never consumed) | M |
| GRP.9 | Dense link storage: v2-B-tree **creation-order** index (`H5G_BT2_CORDER`) + the `creation_order` value itself | `H5Gbtree2.c:107`; `H5Olink.c` creation-order encode | **Missing — the 8-byte creation_order value is decoded only to be skipped; `LinkMessage` has no field for it at all, so original insertion order can never be recovered on read** | Missing — `FLAG_CREATION_ORDER` never set by the writer | `src/format/messages/link.rs:42-45` (no field), `:159-164` (skip), `:85-87` (flag never set) | **H** |
| GRP.10 | Link message codec shared between inline and dense-heap-serialized paths | `H5Olink.c` (same codec both paths) | Implemented — same `decode()` used for both, no divergence found | Missing — no dense-path encode call exists (writer is inline-only) | `src/format/messages/link.rs:47-223`; call sites `src/io/reader.rs:620-623,758` | L |

**Notes.** GRP.9 is user-visible and silent: `h5py.Group(..., track_order=True)` and NeXus-format files
(which conventionally rely on creation order for instrument/scan ordering) lose their intended order on
read with no error or warning — iteration silently falls back to fractal-heap physical order. `LinkMessage`
also rejects any external/UD link (type ≥64) with `UnsupportedFeature`, and the dense-heap scanner
(`read_dense_links`) `break`s out of the whole payload on a decode failure, silently truncating links packed
after it in the same heap block (evidence: `src/io/reader.rs:753-765`).

---

## 8. Chunk index types — all 6

Upstream: `H5Dbtree.c` (legacy v1-B-tree), `H5Dsingle.c`, `H5Dnone.c` (Implicit), `H5Dfarray.c`,
`H5Dearray.c`, `H5Dbtree2.c`. Selection logic in `H5Dchunk.c:938-1037`. Rust: `ChunkIndexType` enum at
`src/format/messages/data_layout.rs:51-56` (SingleChunk=1, Implicit=2, FixedArray=3, ExtensibleArray=4,
BTreeV2=5); codecs in `src/format/btree_v1.rs` and `src/format/chunk_index/{btree_v2,fixed_array,
extensible_array}.rs`.

| # | Index type | Upstream anchor (file:symbol) | READ | WRITE | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| CI.1 | Legacy v1 B-tree (layout msg v1-3) | `H5Dbtree.c:141-162` (`H5D_COPS_BTREE`); `H5Dchunk.c:876-1078` (version gate); `H5Olayout.c:382-386,669-673` (rejected inside v4/v5 msgs, matching rust's enum starting at 1) | Implemented | Missing (by design — writer never emits layout msg version <4) | `src/io/reader.rs:2250`; `src/format/btree_v1.rs:172` (no encode); `src/io/writer.rs:1010-1016` (only returns 4/5) | L |
| CI.1a | v1-btree: layout message versions 1-2 (pre-1.6, not produced by modern tools) | `H5Olayout.c:150-154,311-315` | Missing — only version 3 handled | Missing | `src/format/messages/data_layout.rs:37-43,465-767` (no v1/v2 arms) | L |
| CI.2 | Single Chunk (index_type=1) | `H5Dsingle.c:74-95`; selection `H5Dchunk.c:938-1009`; inline in layout msg, no index structure | Implemented (incl. filtered via `SingleChunkFilter`) | Missing — never emitted | `src/io/reader.rs:1619-1677`; zero `ChunkIndexType::SingleChunk` writer.rs references | L |
| CI.2a | Single Chunk: writer substitutes Fixed Array instead | `H5Dchunk.c:999-1009` | N/A | Partial — emits index_type=3 (Fixed Array, num_chunks=1) instead of true index_type=1; self-consistent (rust reads its own output fine), non-idiomatic not wrong | `src/dataset.rs:272-274,317-328` | L |
| CI.3 | Implicit / No Index (index_type=2) | `H5Dnone.c:75-96`; selection `H5Dchunk.c:996-1019` (no filters AND `alloc_time==EARLY`); pure arithmetic addressing, zero on-disk index | **Missing — hard errors on read** | Missing — `alloc_time` is always incremental, never EARLY | `src/io/reader.rs:1810-1813` (`_ => Err("unsupported chunk index type")`); `src/io/writer.rs:7421` | **H** |
| CI.4 | Fixed Array (index_type=3) | `H5Dfarray.c:196-223`; selection `H5Dchunk.c:997,1020-1037`; paging `H5FAdblock.c:87-158` | Implemented, incl. paged data blocks, filtered+unfiltered | Implemented, incl. paged data blocks | `src/io/reader.rs:1821,1896-1950`; `src/io/writer.rs:5373,51,138` | L |
| CI.5 | Extensible Array (index_type=4) | `H5Dearray.c:194-221`; selection `H5Dchunk.c:955-976` (`unlim_count==1`) | Implemented, incl. super blocks + paged data blocks | Partial — paged data blocks rejected on write (see CI.5a) | `src/io/reader.rs:2721,2861-2893`; `src/io/writer.rs:2929,5619,5727` | M |
| CI.5a | Extensible Array: paged data blocks unsupported on write | `H5EAdblock.c:110-113,390-398` (default page threshold 1024 chunks) | Implemented | **Missing — explicit error, fails cleanly past ~a few thousand chunks on the unlimited dim** | `src/io/writer.rs:3248-3253` (explicit "not yet supported" error) | M |
| CI.5b | Extensible Array: unlimited dim must be dimension 0 | `H5Dearray.c:1259-1297,1159-1171,1469-1481` (`H5VM_swizzle_coords` upstream supports **any** dimension) | **Partial — rust fails to read EA files whose unlimited dim isn't dim 0** | Missing for that case (rejected at create) | `src/io/chunk_grid.rs:13-16,93-100,139-145` ("swizzling is not supported"); `src/io/writer.rs:730-741` | **H** |
| CI.6 | v2 B-tree (index_type=5) | `H5Dbtree2.c:194-221`; selection `H5Dchunk.c:977-995` (`unlim_count>1`) | Implemented | Implemented | `src/io/reader.rs:2057`; `src/io/writer.rs:5521,6052` | L |

**Notes.** `H5O_LAYOUT_VERSION_5` is confirmed 2.0.0-only (absent from every `hdf5_1_14_*` tag, present on
`hdf5_2_0_0`) — rust's acceptance of it is forward-compat, not required for 1.14.x parity. Edge/partial-chunk
filter handling is index-type-agnostic upstream (lives in `H5Dchunk.c`, not per-index-type) so no separate
per-type edge-chunk gap exists beyond the rows above. `open_append` explicitly refuses v0/v1-superblock
files — a deliberate scope boundary, not a silent bug, but it does mean CI.1's read-only support is
unreachable from the one entry point (`open_append`) that would otherwise exercise it against a real old
file being extended.

---

## 9. Metadata checksums (`H5_checksum_*`)

Upstream: `H5checksum.c` `H5_checksum_lookup3` (Jenkins lookup3). Rust: `src/format/checksum.rs`.

| # | Structure | Upstream anchor (file:symbol) | READ (validates?) | WRITE (computes correctly?) | Evidence (rust file:line) | Impact |
|---|---|---|---|---|---|---|
| CKS.1 | Superblock v2/v3 | `H5Fsuper_cache.c` `H5F__cache_superblock_verify_chksum`/`_serialize` | Validates, rejects on mismatch | Correct | `src/format/superblock.rs:97` (write), `:149-155` (verify) | L |
| CKS.2 | Object header v2 chunk 0 | `H5Ocache.c` `H5O__cache_verify_chksum`/`H5O__chunk_serialize` | Validates, rejects on mismatch | Correct | `src/format/object_header.rs:176` (write), `:278-284` (verify) | L |
| CKS.3 | Object header v2 continuation chunk (`OCHK`) | `H5Ocache.c` `H5O__cache_chk_verify_chksum` | **Not validated — trailing 4 bytes stripped but never recomputed/compared (= OH.3)** | N/A — writer never emits `OCHK` chunks | `src/io/reader.rs:1094-1125` (no `checksum_metadata` call) | M |
| CKS.4 | Local heap (classic) | `H5HLcache.c` — `verify_chksum` callback is `NULL`, confirmed no checksum upstream | Correct (matches upstream: no checksum) | Correct | `src/format/local_heap.rs` — no checksum code, matches upstream by design | L |
| CKS.5 | Global heap collection (v1) | `H5HGcache.c` — `verify_chksum` is `NULL` | Correct | Correct | `src/format/global_heap.rs` — no checksum code | L |
| CKS.6 | v1 B-tree | `H5Bcache.c` — `verify_chksum` is `NULL` | Correct | Correct | `src/format/btree_v1.rs` — no checksum code | L |
| CKS.7 | v2 B-tree header/internal/leaf | `H5B2cache.c` (verify+serialize ×3) | Validates all three, rejects on mismatch, regression-tested | Correct for all three | `src/format/chunk_index/btree_v2.rs` (header/leaf/internal encode+decode sites) | L |
| CKS.8 | Fractal heap header/direct-block/indirect-block | `H5HFcache.c` (verify+serialize ×3) | Validates all three, rejects on mismatch | N/A — **read-only module** (`fractal_heap.rs` doc comment: "Fractal heap reader (read-only)"); writer never emits any fractal-heap structure | `src/format/fractal_heap.rs:1,175,354,482` | L (read correct; write gap is completeness, not a checksum-correctness defect) |
| CKS.9 | Free-space manager header + sections | `H5FScache.c` (verify+serialize) | N/A — unimplemented (§6) | N/A — unimplemented (§6) | no `H5FS` code in `src/format` | L |
| CKS.10 | SOHM table + list/btree index | `H5SMcache.c` (verify+serialize) | N/A — unimplemented (§5); message type 0x0f falls into the generic skip arm, no error | N/A — unimplemented | `src/format/messages/mod.rs:14-29` (no `MSG_SHMESG` const); `src/io/reader.rs:1204` skip | M (silently ignoring is correct for checksum purposes specifically, but shared-message semantics are unhandled — see SOHM.5) |
| CKS.11 | Group Info / Link Info messages (inline in v2 OH) | `H5Ocache.c` — checksum coverage is chunk-level only, no per-message checksum field exists upstream | Correctly relies on the containing OH-chunk checksum; no erroneous independent checksum added | Correct | `src/format/messages/{group_info,link_info}.rs` — no checksum hits, confirmed no erroneous addition | L |
| CKS.12 | Fixed Array header/data-block/page | `H5FAcache.c` (verify+serialize ×3) | Validates all three, regression-tested | Correct for all three | `src/format/chunk_index/fixed_array.rs` (multiple encode/decode sites) | L |
| CKS.13 | Extensible Array header/index-block/super-block/data-block/page | `H5EAcache.c` (verify+serialize ×5) | Validates all five, regression-tested | Correct for all five | `src/format/chunk_index/extensible_array.rs` (multiple encode/decode sites) | L |
| CKS.14 | **Jenkins lookup3 algorithm fidelity** | `H5checksum.c:271-334,364-458` (`H5_lookup3_mix`/`_rot`/`_final`, `H5_checksum_lookup3`) | N/A (algorithm, not a structure) | **No divergence found** — byte-for-byte match on initval (`0xdeadbeef + length + initval`), little-endian 4-byte accumulation, mix constants (4,6,8,16,19,4), final constants (14,11,25,16,4,14,24), and the 12→1 tail-byte fallthrough switch, all confirmed by direct side-by-side line read of both files in this session | `src/format/checksum.rs:8-9,14-38,42-63,72-182` vs `H5checksum.c:271-334,364-458` | L (verified correct — this row would be H if any divergence existed) |

**Notes.** No checksum-algorithm divergence was found anywhere — this is the one subsystem in this slice
with a clean bill of health end-to-end. The only real defect is CKS.3/OH.3 (continuation-chunk checksum
parsed but not verified on read), which is a read-side safety gap against arbitrary upstream files, not a
write-side corruption risk.

---

## Top-10 gap list (ranked)

1. **Dense attribute storage is never read at all (`MSG_ATTR_INFO`/0x0015 has exactly one reference in the
   whole crate — its own definition).** Any object with more than h5py's default `max_compact` threshold (8)
   attributes silently reports **zero** attributes on read, with no error. Confirmed independently from three
   angles (MSG.8/MSG.20, BT2.12/BT2.13, HEAP.11) and directly re-verified in this session. *(§3a MSG.8, §3b
   MSG.20, §4 BT2.12/13)*

2. **The superblock extension object header is never opened, on read or write** (`superblock_extension_address`
   is decoded but never dereferenced; always written as `UNDEF_ADDR`). This is the single structural gate
   behind SOHM (§5), persisted free-space info (§6 FS.13), the driver-info message (MSG.19), and the v1-B-tree-K
   message (MSG.18) all at once — closing it is higher-leverage than fixing any one of those independently.
   Independently re-verified in this session (`rg -n superblock_extension_address` → only `UNDEF_ADDR` write
   sites, zero reader.rs references). *(§1 SB.6)*

3. **Datatype class 7 (object/region reference) is entirely unimplemented, and the decode failure is silently
   swallowed** (`if let Ok(...)` at the object-header layer) — a dataset or attribute using a reference dtype
   simply vanishes from the catalog instead of surfacing an error. Reference types are common in real-world
   scientific HDF5 files (region references, dataset cross-links). *(§3a MSG.3f)*

4. **Compound datatype v2 member-name padding uses the wrong version guard** (`if version == 1` where upstream
   pads for v1 **and** v2, only v3 drops padding — `H5Odtype.c:414-451`). A genuine v2 compound message (any
   compound containing an array-typed member, or written under an older libver bound) has every subsequent
   member's name/offset desynced on read — this is byte-level mis-parsing, not a clean rejection. Independently
   re-verified against upstream source in this session. *(§3a MSG.3e)*

5. **Object header continuation chunk (`OCHK`) checksums are parsed but never verified on read** — any real
   HDF5 file whose object header spans multiple chunks (routine once an object accumulates enough
   attributes/links) has its continuation-chunk integrity silently unchecked; a corrupted continuation chunk
   from an upstream file is accepted rather than rejected. *(§2 OH.3, §9 CKS.3)*

6. **Link creation-order is dropped entirely — no field decodes it, and the dense creation-order B-tree index
   is unimplemented.** `h5py(..., track_order=True)` and NeXus-format files silently lose their intended link
   ordering on read, falling back to physical fractal-heap order with no warning. *(§4 BT2.11, §7 GRP.9)*

7. **The Implicit chunk index (index_type=2, "no index") hard-errors on read instead of degrading gracefully.**
   A legitimate, simple h5py option (no filters + early allocation) makes the whole dataset unreadable rather
   than silently failing one attribute or falling back. *(§3a MSG.6b, §8 CI.3)*

8. **SOHM is completely unimplemented, and the per-message "shared" flag bit is never even checked on read** —
   worse than a missing optimization: a file using SOHM has shared-message-pointer bytes fed straight into the
   normal per-type decoder as if they were literal payload. *(§5 SOHM.5, §4 BT2.16)*

9. **Vlen-sequence datatype rejects any wire version other than exactly 1**, but upstream legitimately bumps a
   vlen's version to match its nested base type (vlen-of-array, vlen-of-v3-compound) — such files fail to
   decode entirely even though rust otherwise understands both the vlen and the base class. *(§3a MSG.3h)*

10. **Extensible Array chunk index requires the unlimited dimension to be dimension 0**, while upstream
    supports an unlimited dimension in any position via `H5VM_swizzle_coords`. Resizable datasets whose
    unlimited axis isn't the first dimension fail to read. *(§8 CI.5b)*

**Runners-up not in the top 10** (still real, lower-frequency or lower-severity): dense-group-link name index
missing (§4 BT2.10, mitigated by a working linear-scan fallback); v1-groups' symbolic-link scratch pad
silently dropped (§7 GRP.1); fletcher32 filter checksum stripped without verification (§3a MSG.7a); free-space
manager / persistent file-space strategy entirely absent (§6, mostly M — no false claims of a strategy rust
doesn't implement, so it degrades to file-size bloat across close/reopen rather than corruption).
