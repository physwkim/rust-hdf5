# Integration checklist — merging the round-2/3 worker branches

Six branches, all off `main` (f08df75), all sharing the oracle-harness commits
as a common ancestor. None are merged; that is the user's call.

## Recommended merge order

1. `caucus/QAS5ZKN5CY/oracle-harness-d1c0a356-1` — harness only, everyone's base
2. `caucus/QAS5ZKN5CY/wp-catalog-54353370-1` — deepest reader/writer structure
   (classifier, `object_header_io.rs`, preserved links)
3. `caucus/QAS5ZKN5CY/wp-sbext-040abf4b-1` — shared-message resolution,
   superblock extension, userblock base address
4. `caucus/QAS5ZKN5CY/wp-attrs-f7883ceb-1` — attribute collector typestate,
   dense read/write, fractal-heap writer
5. `caucus/QAS5ZKN5CY/wp-dtype-bad7c793-1` — datatype codec, references
6. `caucus/QAS5ZKN5CY/wp-space-500431fd-1` — dataspace/chunk-index/EFL/VDS

Conflict hotspots: `src/io/reader.rs` (2,3,4,5), `src/io/writer.rs` (2,3,4),
`src/format/object_header.rs` (3,4), `src/bin/oracle_probe.rs` (all).

## Semantic merge requirements (not just textual resolution)

1. **One full object-header reader, one shared-message resolver.** catalog
   hoisted `read_object_header_full` to `src/io/object_header_io.rs` (reader
   AND writer use it); sbext added shared-message resolution inside the
   reader's copy (incl. SOHM table→heap); catalog's round 4 (`87596e2`)
   added `SharedMessage::decode` in `read_datatype_message` (committed case
   only, SOHM = named error). The merged tree must have ONE full reader, in
   the hoisted location, with ONE shared-message resolver handling both
   `H5O_SHARE_TYPE_COMMITTED` and `H5O_SHARE_TYPE_SOHM`, used by both
   reader and writer. No single branch has all three properties.
2. **Writer-side message swallow — CLOSED on catalog's branch**
   (`2097dbb` `ReopenWalk::plan` preserve-by-bytes, `9a6b383` chunk-index
   decode failure preserves instead of substituting an empty index).
   At integration this is verification only: open_rw on a file containing
   a VAX / shared-datatype dataset must preserve the object byte-identically
   on rewrite. Two cross-branch completions to verify after merge:
   (a) catalog's branch measured a dirty rewrite losing the compact `ainfo`
   (0x15) message and the v2 header Timestamps flag — wp-attrs' AINFO
   emission should close it; assert ainfo survives a dirty rewrite.
   (b) catalog's reader-side attribute swallows (`reader.rs:524,956,999,
   1006,1219,1338` at `9a6b383`) are closed by wp-attrs' `ObjectAttributes`
   typestate; confirm no `if let Ok` attr swallow survives the merged
   reader.
3. **Chunk-0 root-attribute preservation on append.** sbext measured
   `open_append` losing root attrs (chunk 0, v1.8-bounds files, userblock or
   not) on its branch. attrs' throwaway-worktree check proved catalog's
   `ca978ee` preserves a continuation-block fixture; re-run sbext's exact
   chunk-0 repro on the integration branch.
4. **Error-variant unification.** catalog added `Hdf5Error::Unsupported` /
   `DanglingLink`; attrs deliberately reused `FormatError::UnsupportedFeature`
   to avoid a textual collision. Unify to one surface after merge.
5. **One selection module.** space built `src/format/selection.rs`
   (VDS, full deserializer); dtype's references carry their own
   point/hyperslab/all/none deserializer in `format::reference`. Deduplicate
   into the standalone module; region-reference WRITE later needs its
   serializer half.
6. **SOHM/named-datatype case convergence.** `sohm_list`/`sohm_btree` stay
   A=MISS on sbext's branch only for `/named_i32` (committed-datatype object
   not enumerated); catalog's classifier lists such objects. Expect MISS→GAP
   after merge; full committed-datatype support is the deferred follow-up.
7. **Dense-link groups across reopen.** attrs demonstrated the writer's
   reopen walk (`collect_links_recursive`, MSG_LINK only) returning a
   dense-link group EMPTY and orphaning its children; catalog's
   `ReopenWalk::plan` classifies dense links as Preserve. Post-merge,
   reopening a dense-link group must preserve or fully recover it — never
   orphan. Also verify creation-order values survive a reopen (attrs
   measured link corder re-stamped as discovery order, dense-attr corder
   as name-hash order).
8. **Oracle artifacts.** Each branch left `doc/oracle-report.md` /
   `oracle/report.json` at its own baseline. Regenerate once on the
   integration branch for the true union numbers; also fix the stale
   `UNSUPPORTED(strpad)` prose in `oracle/run.py` (~line 872) and
   `oracle/CANON.md` §134–137.

## Deferred work items (post-integration queue)

- Committed (named) datatypes as first-class objects (listing kind, type
  exposure; sbext's shared resolution + catalog's classifier are the bases)
- Attribute-stored reference write — blocked: finalize builds attribute bytes
  before header addresses exist; needs allocation/content split in the writer
- Region-reference and revised-reference write (needs selection serializer)
- Fractal-heap indirect blocks (attrs' >32-direct-blocks ceiling, ~8k attrs)
- Dense-storage heap reclamation on rewrite (old heap leaks per session)
- Superblock version fidelity on write (always v3 today; also rewrites v2
  files as v3 on append) — assigned to wp-sbext
- Layout message v1/v2 read (pre-1.8 legacy files, listed-with-reason today)
- External-link write-mode traversal; preserved-link delete/rename
- `open_append` on v0/v1-superblock (symbol-table) files
- Direct chunk read; hyperslab stride/block; point selection; H5Ocopy
- h5py `bytes` (vlen bytes) convention; `Reference::path()` leading-slash vs
  `H5File::dataset()` name convention
- `src/io/reader.rs` internal test module's hardcoded macOS `TEST_PYTHON`
  (ignores `RUST_HDF5_TEST_PYTHON`)
- EFL `H5O_EFL_UNLIMITED` slot
