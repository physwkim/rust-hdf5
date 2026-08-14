# Unified parity gap ranking — Round 1 synthesis (2026-08-14)

Sources:
- Executable oracle: `doc/oracle-report.md` + `oracle/report.json` on branch
  `caucus/QAS5ZKN5CY/oracle-harness-d1c0a356-1` (72 cases × 2 directions vs
  libhdf5 1.14.6 / h5py 3.15.1, `~/micromamba/envs/tomo`).
  Headline: A (h5py writes → rust reads) 59 PASS / 2 DIFF / 9 MISS / 2 GAP;
  B (rust writes → libhdf5 reads) 49 PASS / 6 INVALID / 17 UNSUPPORTED-API.
- Inventory slices: `format-infra.md`, `datatypes.md`, `api-features.md`
  (upstream anchors + rust file:line evidence in each).

Parity target: the 1.14.x file format as produced by libhdf5/h5py.
2.0-dev-only features are flagged in the slices and excluded from ranking.

## Tier 1 — silent data loss / file corruption

| # | Gap | Evidence | Round 2 owner |
|---|-----|----------|---------------|
| 1 | Dense attribute storage never read: >8 attrs show as 0 (`MSG_ATTR_INFO` unconsumed; dense *links* read fine at the same libver) | oracle `attrs_dense` DIFF (nattrs 12→0); format-infra #1 | wp-attrs |
| 2 | Soft/external links silently dropped on read | oracle `link_soft`/`link_external` MISS; api-features #1 | wp-catalog |
| 3 | Contiguous dataset in a libver≥1.10 file dropped from the listing (chunked at same libver PASSes — bisected) | oracle `layout_contiguous_v110`, `libver_v110`, `libver_latest` MISS | wp-catalog |
| 4 | Attribute ≥64 KiB silently corrupts the file on write (u16 message-length truncation, no bounds check) | api-features #3 | wp-attrs |
| 5 | Rust-written files carry object-header attribute count 0 (`h5a.get_num_attrs`=0 while `h5a.iterate` yields them) | oracle B INVALID ×5 (`attr_scalar_num` etc.) | wp-attrs |
| 6 | Unsupported-datatype decode errors silently swallowed → dataset/attribute vanishes from the catalog | datatypes #3; oracle `opaque`/`bitfield`/`ref_*` MISS | wp-catalog (surface) + wp-dtype (decode) |
| 7 | Compound v2 member-name padding uses the wrong version guard — byte-level mis-parse of real files (found independently by two auditors) | datatypes #10; format-infra #4 | wp-dtype |
| 8 | Shared-message (SOHM) flag bit never checked — shared payloads misdecoded as literal message bodies | format-infra #8 | wp-sbext |
| 9 | Unchecked `read_raw::<T>()` ignores byte order — silent misread of big-endian files (invisible to the raw-image oracle; needs typed-read tests) | datatypes #4 | wp-dtype |
| 10 | NULL dataspace read as a 4-byte scalar | oracle `space_null` DIFF | wp-space |
| 11 | Object-header continuation-chunk (OCHK) checksum parsed but never verified | format-infra #5 | wp-sbext |

## Tier 2 — hard errors / rejections on valid 1.14 files

| # | Gap | Evidence | Round 2 owner |
|---|-----|----------|---------------|
| 12 | Implicit chunk index (type 2) hard-errors instead of reading (address arithmetic only, `H5Dnone.c`) | oracle `chunkidx_implicit` GAP; format-infra #7 | wp-space |
| 13 | IEEE f16 unreadable ("non-standard floating-point bit layout") | oracle `float_f16le` GAP | wp-dtype |
| 14 | Datatype message v4 rejected on read even for known classes; vlen-sequence rejects any version but 1 | datatypes #8; format-infra #9 | wp-dtype |
| 15 | Extensible-array index restricted to unlimited dim == 0, unlike upstream | format-infra #10 | wp-space |
| 16 | Superblock extension OH never opened — gates SOHM, FSINFO, driver-info, btree-K all at once | format-infra #2 | wp-sbext |
| 17 | String fidelity: fixed-string SPACEPAD mishandled on read; vlen-string cset forced utf8 on write (`str_vlen_ascii` B INVALID); h5py `bytes` convention unmatched | datatypes #5/#7; oracle B | wp-dtype |

## Tier 3 — feature gaps (Round 3 queue, ranked)

1. Object/region references (`H5T_REFERENCE`, old + revised) — whole class missing
2. Committed (named) datatypes + shared-datatype message
3. Virtual datasets (VDS); external file storage (EFL)
4. `H5Ocopy` / `H5Oget_info` / comments
5. Hyperslab stride/block; point selection; direct chunk **read** (write exists)
6. Link/attr creation-order tracking + by-order iteration; dense-storage write
   thresholds (phase change); intermediate-group auto-create
7. Write-fidelity deviations (libhdf5 reads the data fine, bytes differ):
   superblock always v3 regardless of libver, btree-v1 index written as
   extensible array, filter flags byte 0 vs 1
8. Scale-offset compression (decode exists, compress direction missing)

## Round 2 work packages

All implementation workers branch from main in their own worktree, merge
`caucus/QAS5ZKN5CY/oracle-harness-d1c0a356-1` first for the oracle, verify
each fix flips its oracle verdict, one commit per finding.

- **wp-catalog** — listing/traversal completeness: #2, #3, #6(surface)
- **wp-attrs** — attribute subsystem: #1, #4, #5
- **wp-dtype** — datatype codec: #7, #9, #13, #14, #17, opaque/bitfield decode
- **wp-space** — dataspace/chunk-index: #10, #12, #15
- **wp-sbext** — superblock extension + SOHM + metadata integrity: #8, #11, #16
- **oracle-harness** (reuse) — matrix extension for Tier 3 coverage + modeling fixes
