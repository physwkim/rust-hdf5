# API-level feature parity: rust-hdf5 vs upstream HDF5 C

Scope: the public API surface enumerated in upstream `src/H5{F,G,L,A,D,O,R,Z,P}public.h`
(HDF5 C, `/home/stevek/work/hdf5`, branch `develop` = 2.0-dev), mapped against the
Rust port's public surface (`/home/stevek/work/rust-hdf5`: `src/lib.rs` exports,
`file.rs`, `group.rs`, `dataset.rs`, `attribute.rs`, `swmr.rs`, `parallel.rs`,
`src/io/*`, `src/format/*`).

**Parity target is libhdf5 1.14.x behavior.** Upstream `develop` is 2.0-dev;
every section calls out 2.0-only additions separately so they aren't scored
against the 1.14.x target.

Status values: **Implemented** / **Partial(missing what)** / **Missing** /
**UNVERIFIED**. Interop impact: **H** = common h5py/pytables/netcdf4/MATLAB
files or workflows break; **M** = write fidelity/less common; **L** = exotic.

This is a feature-presence inventory (does the API/on-disk capability exist),
not a correctness/bug audit — known write-path correctness bugs (e.g. the
append-buffer/chunk-index interactions tracked separately in the project's
knowledge base) are out of scope here unless they bear directly on whether a
feature is usable at all.

---

## H5F — file-level API

Upstream anchor: `H5Fpublic.h` (2009 lines).

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| `H5F_ACC_TRUNC` (H5Fcreate flag) | H5Fpublic.h:30,381 (`H5F_ACC_TRUNC`, `H5Fcreate`) | Implemented (as the only behavior — not an opt-in flag) | src/io/file_handle.rs:65-84 (`create_with_locking` always does `if file.metadata()?.len() > 0 { file.set_len(0)?; }`) | L |
| `H5F_ACC_EXCL` (fail if file exists) | H5Fpublic.h:31,339-341,381 | Missing | `H5File::create`/`H5FileOptions::create` (src/file.rs:117-122,803-808) always truncate via `create_with_locking` (src/io/file_handle.rs:65-84); no "fail if exists" option anywhere in src/ | M |
| `H5F_ACC_RDONLY` / `H5F_ACC_RDWR` (H5Fopen flags) | H5Fpublic.h:28-29,402-404,497 | Implemented (as separate methods, not bit flags) | src/file.rs:125-130 (`open` → `Hdf5Reader::open`), src/file.rs:145-150 (`open_rw` → `Hdf5Writer::open_append`); src/io/file_handle.rs:100 (`open_read_with_locking`, shared lock), :119 (`open_readwrite_with_locking`, exclusive lock) | L |
| `H5F_ACC_CREAT` (create-if-missing, no truncate) | H5Fpublic.h:33 | Missing | No "open or create without truncating" path in src/file.rs or src/io/file_handle.rs; only `create` (always truncates) and `open_rw` (requires existing file, errors if missing) | L |
| Userblock (`H5Pset_userblock`, superblock `base_address`) | H5Ppublic.h:3378 (`H5Pset_userblock`); on-disk effect is the superblock base-address field | Missing (write); UNVERIFIED (read offsetting) | src/format/superblock.rs:57,415 decode `base_address` but src/io/writer.rs:966,7097 hardcode `base_address: 0` with no setter anywhere in src/ | M |
| Libver bounds (`H5F_LIBVER_EARLIEST/V18/V110/V112/V114/LATEST`) | H5Fpublic.h:179-189 (`H5F_libver_t`); H5Ppublic.h:5138 (`H5Pset_libver_bounds`) | Partial(missing granular bound selection; only a binary latest/default switch) | src/file.rs:169-193 (`set_libver_latest(bool)`); src/io/writer.rs:1000-1016 (`libver_latest` gates only filtered-chunk layout version 4→5); superblock is unconditionally V3 (src/io/writer.rs:961-971,7092-7100) and object headers unconditionally v2 (src/format/object_header.rs:39,141, `OHDR_VERSION: u8 = 2`) regardless of the setting — no V18/V110/V112/V114-style (pre-1.8) output is possible | M |
| SWMR write mode (`H5Fstart_swmr_write`) | H5Fpublic.h:1392-1415 | Implemented (via a dedicated SWMR writer type, not general `H5File`) | src/io/swmr.rs:348-359 (`SwmrWriter::start_swmr`); src/swmr.rs:487-488 (`SwmrFileWriter::start_swmr`) | H |
| `H5F_ACC_SWMR_READ`/`H5F_ACC_SWMR_WRITE` flags | H5Fpublic.h:36-49 | Partial(superblock flag bit exists; access-mode flag is not distinctly enforced) | src/format/superblock.rs:29 (`FLAG_SWMR_WRITE: u8 = 0x04`), used at src/io/writer.rs:7210; `open_swmr`/`open_swmr_with_locking` (src/io/reader.rs:349,355-358) are aliases of `open`/`open_with_locking` with no distinct SWMR-read validation | M |
| Refresh (`H5Orefresh` interplay with SWMR) | H5Fpublic.h SWMR section; H5Orefresh is in H5Opublic.h | Implemented | src/io/reader.rs:1549 (`fn refresh`); src/swmr.rs:577-584 (`SwmrFileReader::refresh`) | H |
| `H5Fflush` scope (`H5F_SCOPE_LOCAL`/`H5F_SCOPE_GLOBAL`) | H5Fpublic.h:94-101 (`H5F_scope_t`), 588 (`H5Fflush`) | Missing (scope concept) / Partial (flush itself, at other layers) | src/file.rs:750-756 (`H5File::flush` is a documented no-op stub: "does nothing for now"); real flush exists only at writer/dataset/SWMR level: src/io/writer.rs:6938 (`flush_dataset`), src/io/swmr.rs:575, src/dataset.rs:1728 — no `H5F_scope_t`/local-vs-global distinction anywhere | M |
| File mount (`H5Fmount`/`H5Funmount`) | H5Fpublic.h:902-928, 930-952 | Missing | No "mount" match anywhere in src/ | L |
| File image (`H5Pset_file_image`/`H5Fget_file_image`, in-memory open) | H5Ppublic.h:4705; H5Fpublic.h:1038-1084 | Missing | src/file.rs:117,125,145 (`create`/`open`/`open_rw`) take `P: AsRef<Path>` only; src/io/file_handle.rs wraps `std::fs::File` exclusively, no in-memory byte-buffer constructor | M |
| Alignment (`H5Pset_alignment`: threshold + alignment) | H5Ppublic.h:4205-4228 | Missing (configurable); Partial (fixed internal alignment) | src/io/allocator.rs:33 (`alignment: u64` field), :51 (hardcoded `alignment: 8`), :58-60 (`align_up`) — always 8-byte, no threshold, no public setter | M |
| Meta block size (`H5Pset_meta_block_size`) | H5Ppublic.h:5228-5249 | Missing | No "meta_block" match anywhere in src/ | M |
| Small data block size (`H5Pset_small_data_block_size`) | H5Ppublic.h:5435-5461 | Missing | No "small_data_block" match anywhere in src/ | M |
| Paged buffering (`H5Pset_page_buffer_size`) | H5Ppublic.h:5746-5801 | Missing | No "page_buffer" match anywhere in src/ | L |
| File space strategy (`H5Pset_file_space_strategy`, `H5F_FSPACE_STRATEGY_*`) | H5Fpublic.h:191-204; H5Ppublic.h:3107-3126 | Missing (configurable); implicit behavior resembles non-persistent `FSM_AGGR` | No "file_space_strategy"/"FSPACE_STRATEGY" match; closest analog is the session-only, non-persistent free list in src/io/allocator.rs:34-44,93-141, never written to disk as a persistent free-space manager | M |

**2.0-dev-only H5F additions (informational, not scored):** `H5F_LIBVER_V200`
(H5Fpublic.h:186) redefines `H5F_LIBVER_LATEST = 5` instead of 1.14.x's
`H5F_LIBVER_V114 = 4` — rust-hdf5's own `set_libver_latest` doc comment
(src/file.rs:170,176) explicitly targets the 2.0-only value, meaning the
port's one libver-bounds feature is itself scoped to 2.0-dev, not 1.14.x
`LATEST`. `H5F_PAGE_BUFFER_SIZE_DEFAULT` (H5Fpublic.h:81-82, `\since 2.0.0`)
is moot since paged buffering is unimplemented either way.

---

## H5G / H5L — groups and links

Upstream anchors: `H5Gpublic.h` (1217 lines), `H5Lpublic.h` (1956 lines).

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| Auto-create intermediate groups | H5Gpublic.h:119 (`H5Gcreate2`); property `H5Pset_create_intermediate_group` H5Ppublic.h:8851 | Missing | src/io/writer.rs:2582-2596 (`create_group` errors `NotFound` if parent doesn't already exist, no split-and-create-ancestors loop); src/group.rs:59-86 (single level only); src/dataset.rs:208-227 (concatenates path, no auto-creation) | H |
| Hard links (`H5Lcreate_hard`) | H5Lpublic.h:279 | Implemented | src/group.rs:107-119 (`H5Group::link` → `writer.create_hard_link`); src/io/writer.rs:2652-2712; src/format/messages/link.rs:47-54 (`LinkMessage::hard`); refcount at src/io/writer.rs:2829-2848 | L |
| Soft links (`H5Lcreate_soft`) | H5Lpublic.h:358 | Partial(no creation API; reader silently drops soft targets) | Format-level encode/decode only, unit-tested: src/format/messages/link.rs:33-38,56-64,292-298; no `create_soft_link` in src/io/writer.rs (confirmed absent by repo-wide search); src/io/reader.rs:634-635 (`if let LinkTarget::Hard { address } = &link.target` — soft targets fall through and are never surfaced via `group_names`/`dataset_names`) | H |
| External links (`H5Lcreate_external`) | H5Lpublic.h:1417 | Missing | No `ExternalLink`/`H5L_TYPE_EXTERNAL` anywhere in src/; src/format/messages/link.rs:192-218 `decode` matches only `LINK_TYPE_HARD`/`LINK_TYPE_SOFT`, other types → `FormatError::UnsupportedFeature`; callers swallow that error and skip the link (src/io/reader.rs:621,753-764) — external links are silently omitted on read, not surfaced as an error | M |
| User-defined links (`H5Lcreate_ud`/`H5Lregister`) | H5Lpublic.h:1218; `H5Lregister` in H5Ldevelop.h:272 (not public) | Missing | No `H5L_class_t`/UD-link registration concept anywhere in src/ | L |
| Link creation-order tracking (`H5Pset_link_creation_order`) | H5Ppublic.h:9070; H5Lpublic.h:76-79 (`corder`/`corder_valid` in `H5L_info2_t`) | Missing | src/format/messages/link_info.rs:44-52 (`compact_with_creation_order` exists but is never called — writer.rs:7509,7580 always use `LinkInfoMessage::compact()`); src/format/messages/link.rs:85-87 (`encode` never sets `FLAG_CREATION_ORDER`); :159-164 (decode skips and discards the 8-byte corder field) | M |
| Dense link storage threshold (`H5Pset_link_phase_change`) | H5Ppublic.h:9103; H5Gpublic.h:41-47 (`H5G_STORAGE_TYPE_COMPACT`/`DENSE`) | Partial(write side Missing; read side works) | Write: `GroupInfoMessage::with_phase_change` (src/format/messages/group_info.rs:30-39) exists but unused — writer.rs:7514-7515,7585-7586 always emit `GroupInfoMessage::default()`, no fractal-heap link-storage writer path exists. Read: src/io/reader.rs:726-769 (`read_dense_links`) decodes fractal-heap-backed link storage correctly | M |
| Iterate/visit by name (`H5Literate2`/`H5Lvisit2`, `H5_INDEX_NAME`) | H5Lpublic.h:913, 1085 | Partial(no callback/resume API; non-recursive listing only) | src/group.rs:432-483 (`group_names`), :157-194 (`dataset_names`) — collect into `BTreeSet<String>` (name order), immediate children only, no operator callback, no resume cursor; no recursive `H5Lvisit`-equivalent anywhere | M |
| Iterate/visit by creation order (`H5_INDEX_CRT_ORDER`) | H5Lpublic.h:913 | Missing | Listings always go through a `BTreeSet<String>` (name order only); creation-order values are dropped on decode (link.rs:159-164), so there's no data to sort by even internally | M |
| Link delete (`H5Ldelete`/`H5Ldelete_by_idx`) | H5Lpublic.h:409, 447 | Partial(typed delete only; no `by_idx`) | src/file.rs:584-592 (`delete_dataset`), :603-612 (`delete_group`) → src/io/writer.rs:1947, :2019 — caller must already know the object kind, no link-name-agnostic delete; no `by_idx` variant anywhere | M |
| Link move (`H5Lmove`) | H5Lpublic.h:167 | Missing | No `move`/`rename` link function anywhere in src/ | M |
| Link copy (`H5Lcopy`) | H5Lpublic.h:227 | Missing | No `copy_link` function anywhere in src/ | M |
| Link existence/info query (`H5Lexists`, `H5Lget_info2`) | H5Lpublic.h:659, 739 | Partial(implicit existence only; no info/type struct) | src/io/reader.rs:1321 (`has_group`), src/file.rs:615-630 (`dataset()` `NotFound` as implicit probe) stand in for `H5Lexists`, but there is no `H5L_info`-equivalent public type exposing link type/corder/cset/token/val_size | M |
| Get link name by index (`H5Lget_name_by_idx`) | H5Lpublic.h:841 | Missing | No `by_idx`/indexed accessor anywhere; `group_names()`/`dataset_names()` return the full `Vec<String>` only | L |

**2.0-dev-only H5G/H5L additions:** none found — `\since`/`\version` tags and
`git log` on both headers show only doc/style churn (Doxygen fixes,
`hbool_t`→`bool`, copyright updates) post-1.14. Deprecated 1.x symbols
(`H5Gcreate1`, `H5L_info1_t`, `H5Literate1`, etc.) are still present behind
`H5_NO_DEPRECATED_SYMBOLS` in 2.0-dev.

---

## H5A — attributes

Upstream anchor: `H5Apublic.h` (1263 lines).

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| Create attribute on a dataset | H5Apublic.h:146 `H5Acreate2` | Implemented | src/dataset.rs:770 (`new_attr`); src/attribute.rs:421 (`AttrBuilder::create`), :74,117,149,202 (`AttrTarget::Dataset`) | H |
| Create attribute on a group | H5Apublic.h:146 `H5Acreate2` | Implemented | src/group.rs:491-503 (`set_attr_string`), :613-619 (`attr_target` → `Group`/`Root`); src/io/writer.rs:4774 | H |
| Create attribute on a committed/named datatype | H5Apublic.h:146 `H5Acreate2` | Missing | Named datatypes aren't a supported object at all: src/io/reader.rs:658-670,891-899 detect a "committed (named) datatype" only to `continue` (skip it); no `commit`/`NamedDatatype` concept anywhere in src/ | M |
| Dense attribute storage (`H5Pset_attr_phase_change`) | H5Ppublic.h:2427 | Missing | No property-list abstraction exists at all; src/format/object_header.rs:45 (`FLAG_NON_DEFAULT_ATTR_THRESHOLDS`) is read-only (:130,151,221), never set on write; `src/format/fractal_heap.rs` is wired only to dense **links** (src/io/reader.rs:626-627), no dense-attribute path exists; every attribute is always emitted inline (src/io/writer.rs:7489-7491,7562-7564,7634-7636) | M |
| Creation-order tracking + iteration (`H5Pset_attr_creation_order`, `H5Aiterate2` by `H5_INDEX_CRT_ORDER`) | H5Ppublic.h:2389; H5Apublic.h:671 | Missing | Write side never sets tracked/indexed flags (same as row above); `FLAG_ATTR_CREATION_ORDER_INDEXED` is commented out (object_header.rs:44); read side discards the creation-order field ("Skip creation_order for now", object_header.rs:306-308); writer always writes `0` for it (object_header.rs:168-169); no `H5Aiterate2`-style callback API — only `attr_names() -> Vec<String>` in on-disk order (src/dataset.rs:718-731, src/io/reader.rs:1262-1267) | M |
| Delete attribute (`H5Adelete`/`H5Adelete_by_name`/`H5Adelete_by_idx`) | H5Apublic.h:243,282,309 | Missing | No public delete API; the only removal logic is a private `fn evict_attr` (src/io/writer.rs:3744-3758, not `pub`), used only internally before overwriting a vlen-string attribute, not exposed for outright deletion | M |
| Rename attribute (`H5Arename`/`H5Arename_by_name`) | H5Apublic.h:953,1041 | Missing | No rename function exists; `AttributeMessage` has no in-place name mutation (src/format/messages/attribute.rs) | L |
| Large-attribute size heuristic (upstream forces dense/global-heap storage above `H5O_MESG_MAX_SIZE`) | H5Ppublic.h:2427 interplay; upstream `H5Oattribute.c:259` (`raw_size >= H5O_MESG_MAX_SIZE`) | Missing — and a correctness risk, not just a gap | Attribute bytes are always stored inline with no size check on write (src/io/writer.rs:7489-7491, unconditional); the object-header encoder casts message length straight to `u16` with no bounds check (src/format/object_header.rs:184, `(msg.data.len() as u16).to_le_bytes()`) — an attribute (or any message) whose encoded size exceeds 65535 bytes **silently truncates the length field into a corrupt file** instead of falling back to dense storage | M |
| Read/write attribute value (`H5Awrite`/`H5Aread`) | H5Apublic.h:1006, 920 | Implemented | src/attribute.rs:69 (`write_scalar`), :141, :176, :221, :265, :279, :333 — scalar/array numeric + vlen-UTF8 strings; no compound/enum/opaque attribute values | L |
| Get attribute info (`H5Aget_info`/`H5Aget_name`) | H5Apublic.h:425, 513 | Partial(missing dataspace/shape getter and `H5A_info_t`'s `corder`/`cset`/`data_size`) | src/attribute.rs:60 (`name()`), :323 (`datatype()`); no `shape()`/rank accessor, no `H5A_info_t`-equivalent struct (creation order can't be surfaced since it's discarded — see row above) | M |

**2.0-dev-only H5A note:** `git log -- src/H5Apublic.h` on upstream shows only
documentation/formatting churn (Doxygen fixes, `hbool_t`→`bool`,
license-header renames) — no new or removed H5A prototypes since 1.14.x.

---

## H5D — dataset I/O

Upstream anchor: `H5Dpublic.h` (2059 lines).

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| Full-extent I/O (`H5S_ALL`) | H5Dpublic.h:948 `H5Dread`, :1173 `H5Dwrite` | Implemented | src/dataset.rs:796 (`write_raw`), :898 (`write_raw_bytes`), :2035 (`read_raw`), :2098 (`read_raw_bytes`) | L |
| Hyperslab selection I/O (start/stride/count/block) | H5Spublic.h:1244 `H5Sselect_hyperslab` | Partial(no stride/block; only start+count) | src/io/hyperslab.rs:47-137 (`for_each_dual_run`/`for_each_contiguous_run` take only `starts`/`counts`); src/dataset.rs:1748 (`read_slice`), :1813 (`write_slice`); src/io/writer.rs:3447-3499 (same start/count-only signature) | M |
| Point selection I/O (`H5Sselect_elements`) | H5Spublic.h:1137 | Missing | No "select_elements"/point-selection/coordinate-list I/O anywhere in src/dataset.rs, src/io/reader.rs, src/io/writer.rs | M |
| Set extent — GROW | H5Dpublic.h:1543 `H5Dset_extent` | Implemented | src/dataset.rs:1666 (`extend`), :1706 (`set_extent`); src/io/writer.rs:6406 (`set_dataset_extent`, validates against `max_dims`) | L |
| Set extent — SHRINK | H5Dpublic.h:1543 | Implemented | src/io/writer.rs:6460-6540 (`prune_chunks_beyond` frees chunks entirely beyond the new extent, refills straddling chunks with fill value, releases vlen heap refs), called from `set_dataset_extent` (:6406,6464-6466) | L |
| Fill value (`H5Pset_fill_value`) | H5Ppublic.h:6727 | Implemented | src/dataset.rs:198 (`DatasetBuilder::fill_value`); src/io/writer.rs:7422-7429; src/format/messages/fill_value.rs:37-79 | L |
| Fill time (`H5Pset_fill_time`) | H5Ppublic.h:6677 | Missing | No `fill_time` setter on `DatasetBuilder`; `fill_write_time` hardcoded to `0` (on-alloc) everywhere: src/io/writer.rs:7426,7433; src/format/messages/fill_value.rs:53 | M |
| Allocation time (`H5Pset_alloc_time`) | H5Ppublic.h:6436 | Partial(not user-settable; and contiguous behavior contradicts its own declared value) | No `alloc_time` setter on `DatasetBuilder`; src/io/writer.rs:7420-7439 hardcodes `alloc_time` (3=incr for chunked, 2=late for contiguous) but :2884-2889 `create_dataset` eagerly allocates every contiguous dataset at creation time regardless of the "late" value it just wrote | M |
| External file storage (`H5Pset_external` / EFL) | H5Ppublic.h:6635 | Missing | No "external"/"efl" file-layout handling anywhere in src/; src/format/messages/data_layout.rs:44-46 implements only Compact/Contiguous/Chunked classes, no EFL class | L |
| Virtual datasets (VDS) | H5Ppublic.h:7297 `H5Pset_virtual`; H5Dpublic.h:63 `H5D_VIRTUAL` | Missing | No `H5D_VIRTUAL`/`VirtualLayout`/`set_virtual` anywhere in src/ | M |
| Direct chunk write (`H5Dwrite_chunk`) | H5Dpublic.h:1311 | Implemented | src/dataset.rs:1222 (`write_chunk_raw`), :1323 (`write_chunk_raw_at`) — write pre-filtered bytes verbatim with caller-supplied `filter_mask` | L |
| Direct chunk read (`H5Dread_chunk2`/`H5Dread_chunk1`) | H5Dpublic.h:1377 (2.0.0), :2051 (deprecated 1.x) | Missing | `pub fn` list in src/dataset.rs has no `read_chunk`/`read_chunk_raw`; the only chunk-byte readers, src/io/writer.rs:5015 (`read_chunk_if_present`) and :5183 (`read_chunk_at_coords`), are `pub(crate)`-only | M |
| Chunk query APIs (`H5Dget_num_chunks`/`get_chunk_info`/`get_chunk_info_by_coord`) | H5Dpublic.h:689, 808, 723 | Missing | No `num_chunks`/`chunk_info`/`chunk_info_by_coord` method in src/dataset.rs; only `chunk_dims()` (:663) and `is_chunked()` (:693) expose any chunk metadata | M |
| Layout types (`H5D_COMPACT`/`CONTIGUOUS`/`CHUNKED`, `H5Pset_layout`) | H5Dpublic.h:58-65; H5Ppublic.h:6800 | Partial(read-only Compact support; writer cannot create Compact datasets) | Reader: src/format/messages/data_layout.rs:167-168,226-228,491-501 and src/io/reader.rs:1471,2656,3062 handle `DataLayoutMessage::Compact`. Writer: no code path in `DatasetBuilder::create` (dataset.rs:212-423) ever emits `Compact` — only contiguous (:384-422) and chunked (:257-370) exist | M |

**2.0-dev-only H5D additions:** `H5Dread_chunk2` (H5Dpublic.h:1377, `\since
2.0.0`) replaces deprecated `H5Dread_chunk1` (:2051) with an added bounds
parameter — a signature change, not a new capability; rust-hdf5 has neither
variant (see "Direct chunk read" row). `H5Dchunk_iter` and
`H5Dread_multi`/`H5Dwrite_multi` (+ async variants) are `\since 1.14.0`, so
in-scope for the parity target but not requested by this slice's capability
list.

---

## H5O / H5R — objects and references

Upstream anchors: `H5Opublic.h` (2480 lines), `H5Rpublic.h` (964 lines).

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| Object copy (deep-copy group/dataset/named-datatype incl. attrs, intra/cross-file) | H5Opublic.h:935 `H5Ocopy` (flags H5Opublic.h:33-40) | Missing | No `copy_object`/H5Ocopy-equivalent anywhere in src/; src/file.rs and src/group.rs `pub fn` lists expose only create/link/dataset/write methods | M |
| Get object info (type, address, ref count, timestamps, header size) | H5Opublic.h:506,543,602 (`H5Oget_info3`/`_by_name3`/`_by_idx3`, struct `H5O_info2_t`) | Missing | No `ObjectInfo`/`H5O_info` analogue anywhere; timestamps unimplemented — src/format/messages/mod_time.rs is a one-line stub ("will be implemented later"); object-header v2 optional timestamps explicitly skipped (src/format/object_header.rs:145,216, "MVP writes zeros"/"skip timestamps"); refcount is written internally (src/io/writer.rs:853-861) but never surfaced through a public getter | H |
| Object flush (`H5Oflush`, per-object cache flush) | H5Opublic.h:1400 (`\since 1.10.0`) | Partial(dataset-only, different semantics) | src/dataset.rs:1727-1728 (`flush`, doc'd "flush a chunked dataset's index structures") — dataset-level only, no group/named-datatype equivalent; whole-file flush exists separately (src/file.rs:750-751, src/swmr.rs:520-522) but that's file-wide, not object-scoped | M |
| Object refresh (`H5Orefresh`, SWMR per-object metadata re-read) | H5Opublic.h:1436 (`\since 1.10.0`) | Partial(whole-reader refresh only, not per-object) | src/swmr.rs:579-583 (`SwmrFileReader::refresh`, "re-read the superblock and dataset metadata") refreshes the whole reader, not one object identifier; no `Dataset::refresh`/`Group::refresh` method exists | M |
| User comments on an object (`H5Oset_comment`/`H5Oget_comment`) | H5Opublic.h:985,1034,1074,1126 | Missing | Zero HDF5-comment-message hits anywhere in src/ (`rg -ni "comment"` matches only an unrelated code comment at src/io/file_handle.rs:15) | L |
| Object exists / visit (`H5Oexists_by_name`, `H5Ovisit3`) | H5Opublic.h:463, 1220 | Missing | No `pub fn ... exist`; no recursive-visit API in src/group.rs (only single-level `dataset_names`/`group_names`); existence can only be inferred indirectly via `Err` from `group()`/`dataset()` (src/group.rs:122, src/file.rs:615) | L |
| Create object reference (`H5Rcreate_object`, `H5R_OBJECT`) | H5Rpublic.h:150 (`\since 1.12.0`) | Missing | No `H5R_OBJECT`/object-reference construction anywhere; `H5Type` trait (src/types.rs:8-15) has no reference-type impl; `DatatypeMessage` enum (src/format/messages/datatype.rs:70-134) has no `Reference` variant | H |
| Create region reference (`H5Rcreate_region`, `H5R_DATASET_REGION`) | H5Rpublic.h:189 (`\since 1.12.0`) | Missing | Same evidence as above; dataspace-selection serialization exists (src/format/messages/dataspace.rs) but is never paired with an object address to build a reference blob | H |
| Create attribute reference (`H5Rcreate_attr`) | H5Rpublic.h:229 (`\since 1.12.0`, new-API-only, no back-compat) | Missing | No `H5R`-anything in src/attribute.rs | M |
| Dereference (`H5Rdereference`/`H5Ropen_object`/`H5Ropen_region`) | H5Rpublic.h:366,417,464 (new API); H5Rpublic.h:734,876 (legacy) | Missing | No open/dereference-by-reference API in src/dataset.rs, src/group.rs, or src/file.rs — consistent with no reference-creation support above | H |
| On-disk reference encoding (`H5T_REFERENCE` datatype class: object address, or address + global-heap-stored serialized selection) | H5Rpublic.h:44-56 (`H5R_type_t`); referenced via `H5T_STD_REF_OBJ`/`H5T_STD_REF_DSETREG` | Missing | src/format/messages/datatype.rs:33-39 defines datatype classes 0,1,3,6,8,9,10 — class 7 (Reference) is absent and falls through to the catch-all error at :1022-1025 (`UnsupportedFeature(format!("datatype class {}", class))`); vlen global-heap support (src/format/global_heap.rs:377,394) is exclusively for vlen strings/sequences, not region-reference selections | H |

**2.0-dev-only H5O/H5R notes:** H5O's `H5Oopen_by_token`/`H5Otoken_*`
(H5Opublic.h:299,1617,1639,1660) are 1.12+ opaque-token replacements for the
deprecated `haddr_t`-based `H5Oopen_by_addr` — introduced before 2.0-dev, not
2.0-exclusive, and moot either way since rust-hdf5 has no H5O-info API at
all. `H5Oare_mdc_flushes_disabled`/`_disable_`/`_enable_mdc_flushes`
(H5Opublic.h:1490,1531,1566) and all `_async` variants are 1.10/1.14
additions, not 2.0-only. For H5R specifically: the entire "new" reference API
(`H5R_ref_t`, `H5Rcreate_object/_region/_attr`, `H5Rdestroy`, `H5Ropen_*`,
`H5Rget_*`, H5Rpublic.h:150-597) is `\since 1.12.0`, not 2.0-only; the old API
(`H5Rcreate`, `H5Rdereference1/2`, `H5Rget_region`, `H5Rget_name`) remains
shipped for back-compat behind `H5_NO_DEPRECATED_SYMBOLS`. rust-hdf5
implements neither generation, so the 1.14.x-vs-2.0-dev distinction is moot
for this port.

---

## H5Z — filter pipeline

Upstream anchor: `H5Zpublic.h` (372 lines), cross-checked against
`H5Zdeflate.c`, `H5Zshuffle.c`, `H5Zfletcher32.c`, `H5Zszip.c`, `H5Znbit.c`,
`H5Zscaleoffset.c`.

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| Filter pipeline ORDER (forward on write, reverse on read) | H5Z.c:1527 (write, forward loop); H5Z.c:1405 (read, reverse loop) | Implemented | src/format/messages/filter.rs:428-434 (`apply_filters`, forward); :446-459 (`reverse_filters_masked`, `.rev()`); called from src/io/writer.rs:5877 (write) and src/io/reader.rs:233,1648 (read) | H |
| `H5Z_FLAG_OPTIONAL` semantics (skip-on-failure at write) | H5Ppublic.h:2752 flags param; H5Z.c:1523-1608 (write loop sets a `failed` bit and continues rather than erroring) | Missing | src/format/messages/filter.rs:64 documents `flags` bit 0 as "filter is optional" but never reads it; `apply_filters`/`apply_single_filter` (:428-434,514) have no `filter.flags` check — any filter `Err` aborts the whole write; src/io/writer.rs:5877-5881 always records `filter_mask = 0` ("whole pipeline ran"), no optional-skip path | H |
| Deflate/zlib (ID=1) | H5Zdeflate.c:57; H5Ppublic.h:2485 `H5Pset_deflate` | Implemented | src/format/messages/filter.rs:38 (`FILTER_DEFLATE=1`), :516-546 (`flate2` compress/decompress, feature-gated, default-on); src/dataset.rs:120-128 (`DatasetBuilder::deflate`) | H |
| Shuffle (ID=2) | H5Zshuffle.c:108; H5Ppublic.h:6763 | Implemented | src/format/messages/filter.rs:39 (`FILTER_SHUFFLE=2`), :472-491/:494-512 (`shuffle`/`unshuffle`), :547-555 dispatch; src/dataset.rs:130-139 (`shuffle_deflate`) | H |
| Fletcher32 checksum (ID=3) | H5Zfletcher32.c:49 (read recomputes and compares, errors unless `H5Z_FLAG_SKIP_EDC`) | Partial(write-side checksum byte-exact; read side strips the trailer without verifying it — no corruption detection) | src/format/messages/filter.rs:873-906 (`fletcher32()`, byte-exact); write arm :556-565 (appends checksum); read arm :566-574 only does `data[..len-4].to_vec()`, no recomputation/comparison; confirmed via `grep fletcher` in reader.rs/writer.rs (no other check) | M |
| SZIP (ID=4) | H5Zszip.c:251; H5Ppublic.h:7159 | Implemented | src/format/messages/filter.rs:41 (`FILTER_SZIP=4`), dispatch :576-619 calls `format::szip::compress`/`decompress` (src/format/szip.rs:1111,1200) — wired into `apply_single_filter`, not dead code, unconditionally compiled | M |
| N-bit (ID=5) | H5Znbit.c:922; H5Ppublic.h:6892 | Implemented | src/format/messages/filter.rs:42 (`FILTER_NBIT=5`), dispatch :625-627 calls `format::nbit_scaleoffset::apply_nbit` (src/format/nbit_scaleoffset.rs:648) for both directions; builder helper `FilterPipeline::nbit` (:172-241) | M |
| Scale-offset (ID=6) | H5Zscaleoffset.c:1098; H5Ppublic.h:7001 | Partial(decode-only; compress unimplemented) | src/format/messages/filter.rs:43 (`FILTER_SCALEOFFSET=6`); dispatch :634-642 — the `compress == true` branch returns `Err(UnsupportedFeature("scale-offset filter compression is not implemented"))`; only decompress calls `format::nbit_scaleoffset::reverse_scaleoffset` (:928), confirmed intentional by test comment :2413 | M |
| Dynamically-loaded plugin IDs LZ4=32004, ZSTD=32015, BLOSC=32001, BZIP2=307 | H5Z.c:1416-1431 (`H5PL_load` dynamic-plugin path upstream) | Implemented (as native feature-gated code, not via dynamic plugin loading) | src/format/messages/filter.rs:44,53,47,46 (ID constants); dispatch: LZ4 :647-739 (`#[cfg(feature="lz4")]`), ZSTD :744-757, BZIP2 :762-787, BLOSC :845-856 — all feature-gated, Cargo.toml:29-33 | M |
| Custom/unknown filter passthrough — mandatory unregistered filter must error, never produce garbage | H5Z.c:1416-1445 (`H5Z__find_idx` fails → hard error regardless of `H5Z_FLAG_OPTIONAL`) | Implemented | src/format/messages/filter.rs:858-861, final match arm `other => Err(UnsupportedFeature(...))`, hit for any unrecognized ID (e.g. ZFP=32013, JPEG=32019, BLOSC2=32026 are defined as constants but not matched) on both directions — decode always propagates `Err`, never corrupted bytes | H |

**src/format/szip.rs and src/format/nbit_scaleoffset.rs are live, not dead
code** — both are called from the filter-ID dispatch table in
`apply_single_filter` (src/format/messages/filter.rs), confirmed above. The
one live gap inside that wiring is scale-offset **compression**, which is a
hard error rather than a silent no-op.

**2.0-dev-only H5Z note:** no new public `H5Z*`/`H5Pset_filter*` symbol was
added to `H5Zpublic.h` itself since 1.14.6 (doc/`\since`-tag churn only).
2.0-dev does add internal "structured chunk filter" plumbing (new
`H5Dstruct_chunk.c`/`H5SC.c`, commit `af23786b72c`) and a new
`H5Z_ignore_filters` gate that changes optional-pipeline behavior for
`H5S_NULL`/`H5S_SCALAR` dataspaces at creation time — internal semantics
changes, not new public surface, and out of scope for the 1.14.x target.

---

## H5P — creation properties with on-disk effect

Upstream anchor: `H5Ppublic.h` (10544 lines) — only the properties listed in
the task scope are covered here (not the full header).

| Feature | Upstream anchor (header:function) | rust-hdf5 status | Evidence (rust file:line) | Interop impact |
|---|---|---|---|---|
| `H5Pset_chunk` (chunk dimensions, baseline) | H5Ppublic.h:6486 | Implemented | src/dataset.rs:99-101 (`DatasetBuilder::chunk`); src/format/messages/data_layout.rs:185-193 (`ChunkedV4`) | L |
| `H5Pset_chunk_opts` (`H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS`) | H5Ppublic.h:6525; on-disk bit `H5O_LAYOUT_CHUNK_DONT_FILTER_PARTIAL_BOUND_CHUNKS=0x01` | Missing | No `chunk_opts`/`DONT_FILTER` symbol anywhere; layout-message `flags` byte is only ever `0` or `0x02` (src/format/messages/data_layout.rs:254,276,298,315,1012) — bit `0x01` is never set or read; `DatasetBuilder` has no edge-chunk-filter option | M |
| `H5Pset_alignment` (threshold, alignment) | H5Ppublic.h:4228 | Missing | `H5FileOptions` exposes only `locking` (src/file.rs:766-793) — no `alignment`/`threshold`; allocator alignment is fixed at 8 with no threshold concept (src/io/allocator.rs:33,51) | M |
| `H5Pset_istore_k` (chunk-index v1 B-tree K) | H5Ppublic.h:3157 | Missing (no on-disk hook — v1 B-tree chunk indexing is never written) | src/format/btree_v1.rs has only a `decode` path (no `encode`); `DataLayoutMessage::ChunkedV3` (the v1-B-tree layout) is constructed only in decode/test helpers and src/io/reader.rs:1478,3077, never in src/io/writer.rs — rust always writes `ChunkedV4` (FA/EA/v2-B-tree), so `istore_k` has nothing to attach to | L |
| `H5Pset_sym_k` (symbol-table v1 B-tree K, old-style groups) | H5Ppublic.h:3360 | Missing | rust never writes old-style (v1-B-tree + local-heap) groups — group creation always emits new-style Link-Info/Group-Info messages (src/io/writer.rs:7503-7515); src/format/btree_v1.rs and src/format/symbol_table.rs are decode-only (legacy-file read support) | L |
| `H5Pset_attr_phase_change` (compact↔dense attribute threshold) | H5Ppublic.h:2427 | Partial(hardcoded, not configurable) | `FLAG_NON_DEFAULT_ATTR_THRESHOLDS` plumbing exists (src/format/object_header.rs:45,130-134,151-155,221) but is never set — `ObjectHeader::new()` hardcodes `flags: 0x02` (:74-79), used unconditionally by writer.rs:7409-7414,7503-7508; no dense (fractal-heap+v2-B-tree) attribute storage exists anywhere, so attributes are always compact regardless of count | M |
| `H5Pset_link_phase_change` (compact↔dense link threshold) | H5Ppublic.h:9103 | Partial(hardcoded, not configurable) | `GroupInfoMessage::with_phase_change` (src/format/messages/group_info.rs:31-38) exists but the writer never calls it — every group uses `GroupInfoMessage::default()`/`LinkInfoMessage::compact()` unconditionally (writer.rs:7508-7515,7579-7586); `fractal_heap_address` is only ever `UNDEF_ADDR` on write (src/format/messages/link_info.rs:35-40) — groups are always compact regardless of link count | M |
| `H5Pset_shared_mesg_nindexes`/`_index` (SOHM) | H5Ppublic.h:3225, 3258 | Missing | Zero matches for "SOHM"/"shared_mesg"/"SharedMessage" anywhere in src/ | M |
| `H5Pset_obj_track_times` | H5Ppublic.h:2825 | Partial(hardcoded, not configurable) | `FLAG_STORE_TIMESTAMPS=0x20` plumbing exists (src/format/object_header.rs:46,127-129,146-150,209) but is never set — `ObjectHeader::new()` hardcodes no-timestamp `flags: 0x02` (:74-79), used unconditionally (writer.rs:7409,7506,7577); legacy `H5O_MTIME` message is an explicit stub (src/format/messages/mod_time.rs:1) — rust never writes any object timestamps and offers no way to request them | M |
| `H5Pset_layout` (`H5D_COMPACT`/`CONTIGUOUS`/`CHUNKED` selection) | H5Ppublic.h:6800 | Partial(missing explicit selection + Compact write support) | `DatasetBuilder` has no `.layout()` selector; layout is inferred purely from `chunk_dims` presence (src/dataset.rs:257 chunked branch, :384 contiguous fallback); `DataLayoutMessage::Compact` exists for decode/round-trip only — never constructed in src/io/writer.rs or src/dataset.rs | M |

**2.0-dev-only H5P additions relevant to this list (informational):**
`H5Pset_chunk`'s doc notes 2.0.0 chunks >4 GiB when paired with
`H5F_LIBVER_V200` — layered on the 1.14.x baseline, out of scope here.
`H5Pset_virtual_spatial_tree`/`get_` (H5Ppublic.h:6060,6581) are new 2.0-only
VDS-acceleration properties, not part of the 1.14.x-parity surface and
outside this slice (VDS itself is scored under H5D, above, as Missing).
`H5Pget_external`'s `off_t`→`HDoff_t` widening (H5Ppublic.h:6092-6096,6631)
is a 2.0 ABI change with no on-disk format implication.

---

## Out of scope (runtime-only concerns)

Per the audit brief, the following are **not** scored as table rows because
they have no on-disk-format implication — they are process/runtime behavior:

- **H5E (error stack).** rust-hdf5 uses its own `Hdf5Error`/`Result` types
  (`src/error.rs`), not an HDF5-style pushable error stack — a deliberate API
  design difference, not a format gap.
- **H5I (identifiers).** No `hid_t`-style identifier/reference-counting layer
  exists or is needed — Rust ownership replaces it.
- **Chunk cache tuning** (`H5Pset_chunk_cache`, `H5Pset_cache`,
  `H5Fget/set_mdc_config`). Pure performance knobs with no on-disk effect;
  not evaluated here.
- **MPI / parallel HDF5** (`H5Pset_fapl_mpio`, `H5FD_MPIO`, collective I/O).
  **Entirely absent**, worth flagging explicitly even though out-of-scope:
  `src/parallel.rs` is a private, in-process **rayon** thread pool
  (`io_pool()`, src/parallel.rs:47-57) used to parallelize chunk
  compression/decompression on one node — it has no relationship to
  MPI-based collective I/O across processes/nodes. There is no
  `H5FD_MPIO`-equivalent file driver anywhere in the crate. Any workflow
  depending on parallel HDF5 (multi-rank collective writes) is unsupported,
  not just partially implemented.

---

## Top-10 ranked gap list for this slice

1. **Soft and external links are silently dropped when *reading* any file
   that contains them** (H5G/L rows: soft-link read, external-link read).
   This isn't a Rust-side write limitation — it's silent data loss when
   *opening ordinary third-party HDF5 files* (e.g. NeXus hierarchies that
   commonly use soft/external links for default-plot pointers or federated
   datasets). No error is raised; the link just disappears from
   `group_names()`/`dataset_names()`.
2. **`H5T_REFERENCE` (object and region references) is entirely
   unimplemented** (H5O/R rows 7-11). Any dataset or attribute whose
   datatype class is Reference (class 7) hits a hard decode error on read
   and cannot be constructed on write at all — this is a whole datatype
   class missing, not a single API gap, and it blocks every workflow that
   uses HDF5 references (common in MATLAB-authored files and h5py low-level
   reference usage).
3. **Attribute values ≥64KB silently corrupt the file on write** (H5A "large
   attribute" row): the object-header message-length field is cast straight
   to `u16` with no bounds check, so the length wraps instead of erroring or
   falling back to dense/global-heap storage. This is the one item in this
   slice that produces a corrupt file with no diagnostic, rather than merely
   an absent feature.
4. **No auto-creation of intermediate groups** (H5G/L row 1): `create_group`
   requires every ancestor to already exist. This breaks the single most
   common h5py/pytables nested-path idiom
   (`create_dataset('a/b/c/data', ...)`).
5. **Compact→dense storage thresholds are hardcoded off for both links and
   attributes**, and creation-order tracking is dropped on both read and
   write (H5G/L row 6-7, H5A rows 4-5, H5P `attr_phase_change`/
   `link_phase_change` rows). Groups/objects with many children never get
   space-efficient dense storage, and any consumer that depends on
   insertion-order iteration (common in NeXus/config-style files) gets
   name-sorted order instead — silently, since the order data is discarded
   at decode time rather than just left at defaults.
6. **No object-reference-free introspection surface**: `H5Oget_info`/
   `H5Ocopy`/`H5Oset_comment` are all entirely missing (H5O rows 1-2, 5).
   `H5Ocopy` in particular is the mechanism h5py's `.copy()` and most
   "consolidate several files into one" scripts rely on.
7. **External file storage (EFL, `H5Pset_external`) is entirely missing**
   (H5D row). Files that keep raw data in sibling files — common in legacy
   NetCDF-classic-in-HDF5 bridges and some instrument pipelines — can be
   neither created nor read.
8. **Virtual datasets (VDS) are entirely unimplemented** (H5D row). Common
   in detector/synchrotron NeXus pipelines that stitch many physical files
   into one logical dataset (e.g. Eiger/DECTRIS acquisition software).
9. **Hyperslab I/O has no stride/block support (start+count only), and point
   selection is entirely missing** (H5D rows 2-3). This caps the port's own
   slicing flexibility relative to `H5Sselect_hyperslab`/
   `H5Sselect_elements` — strided access (`ds[::2]`-style) and scattered
   point I/O can't be expressed at all.
10. **Direct chunk read (`H5Dread_chunk`) is missing while direct chunk
    write exists** (H5D row) — the asymmetry blocks chunk-copy/rechunk
    tooling (e.g. cloud-optimized-HDF5/kerchunk-style pipelines) that need
    to move compressed chunk bytes without re-running the filter pipeline.

**Also notable but outside the top 10:** `H5Z_FLAG_OPTIONAL` is unimplemented
(H5Z row) — a write aborts instead of silently skipping a failed optional
filter, which is spec-deviant but rare in practice since most common
compressors (deflate/shuffle) aren't marked optional by typical writers;
`H5Pset_alloc_time`/`H5Pset_fill_time` are hardcoded rather than
configurable (H5D rows), including one internal inconsistency where
contiguous datasets are eagerly allocated despite the fill-value message
declaring "late" allocation.
