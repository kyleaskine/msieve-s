# Benchmark evidence retained in Git

Keep measured results, settings, winning polynomial coefficients, provenance
hashes and small reusable seeds here. Full logs, per-command records, candidate
pools and copies of source trees belong in the ignored `stage23_gpu/runs/`
area. Export full snapshots there with `save.py` or `archive.py`; only promote
the summaries and fixtures needed to review or reproduce a specific result.

## October 27–28 evidence consolidation

The pending change originally contained 651 files: 30 source, test, config,
documentation and reproduction-script files, plus 621 generated evidence files.
The evidence is now nine files, with this index making 40 pending files total.

| Study | Original Git-visible evidence files | Retained files | Contents |
| --- | ---: | ---: | --- |
| [C170 full GPU run](c170/gpu_ropt/20261027/summary.json) | 78 | 1 | Timing, migration audit, sieve settings, CPU/GPU winners and scores |
| [C184 full GPU run](c184/gpu_ropt/20261027/summary.json) | 90 | 1 | Same, including preferred and alternative sieve profiles |
| [C179 reference retention](c179/ropt_coverage/live_20261027/result.json) | 15 | 1 | Depth and Murphy checks, reference and GPU winners |
| [Root-search coverage study](root_coverage/20261027/summary.json) | 212 | 5 | Timing/quality results, loss diagnostics, winners and four new seed fixtures |
| [Coverage review validation](root_coverage/review_20261027/summary.json) | 226 | 1 | Host/GPU checks, unchanged legacy plans, export parity and provenance |
| **Total** | **621** | **9** | |

The root-search timing/quality results describe the frozen **pre-review**
prototype. The later review checks establish correctness and parity, not new
performance measurements. The numerical implementation remains unchanged by
this consolidation.

No evidence was deleted. Full snapshots were moved, with every file's SHA-256
verified, to `stage23_gpu/runs/20261028_commit_evidence/`, preserving the table's
relative directory paths. Its `inventory.json` records all original file hashes.
There are 631 physical snapshot files; ten binary metadata files were already
ignored and were not part of the original 651-file pending change.

Each retained result records the local snapshot path and original manifest hash
under `evidence`. Full original runs, including larger inputs and pools, remain
at their existing paths under `stage23_gpu/runs/`. These ignored artifacts are
local only; a fresh Git checkout has the summaries and fixtures, not the full
historical replay environment. See the [run index](../runs/README.md) for study
scripts, validation details and reproduction requirements.

## Older evidence compacted on October 28

Eighteen bulk artifacts from the October 21–25 studies have been moved out of
the checkout. Their 101.8 MiB of original data became 3.1 MiB of compressed
tables, fixtures and metadata. Including the small reader/documentation changes,
versioned file contents shrink from 114.4 MiB to about 15.7 MiB relative to
commit `ca2a3d0`; historical Git objects are unchanged.
The [manifest](historical_evidence.json) records original paths,
sizes, SHA-256 hashes, Git blobs and the source commit. Full originals remain in
the ignored `stage23_gpu/runs/20261028_historical_evidence/files/` area, preserving
their repository-relative paths. This cleanup changes the current tree only.

- Six AS660 score tables retain all 60,000 rows, byte for byte after
  decompression, as adjacent `scores*.jsonl.gz` files. The statistical comparison
  readers accept these automatically.
- Six discovery/sample ledgers are local archive material. Eighteen
  [representative raw seeds](as660/size_sample_fixtures.json) retain each
  coefficient sample's best, middle and worst score for small reproductions.
  These were selected after scoring and are not an unbiased evaluation set.
- The full CPU-review checkpoint and top-1000 list are archived. Aggregate
  results, winner origin proofs and the winning stage1 record remain versioned.
- The full filesystem/Git inventories and retained-file inventory are archived.
  Cleanup decisions, the exact deletion list, execution results and verification
  totals remain versioned in the storage-audit directory.

The affected scripts use `stage23_bench/tools/run_evidence.py` to read an existing
plain file, its gzip replacement, or the mapped local original. The helper does
not recreate large files. A fresh checkout can reproduce the score-distribution
and discovery-quarter comparisons from the gzip tables; full historical replay
still requires the original inputs, binaries and other local run artifacts.

If an archived original is unavailable, recover its bytes with
`git show SOURCE_COMMIT:ORIGINAL_PATH`, using the commit and path in the manifest,
and save them under the manifest's `archive_root` plus `ORIGINAL_PATH`. Verify
the recorded SHA-256 before using it. The exact old bulk paths are ignored so
regenerating them does not add them to a future commit accidentally.
