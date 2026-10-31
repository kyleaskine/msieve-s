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

## October 28 pipeline comparisons

The [six-run report](gpu_pipeline/20261028.json) retains CPU/GPU winners, common
scoring settings, validation totals and the two isolated size-optimization
regression inputs. It also compares the separate 60060 and 138600 C190 CPU runs
with the GPU search of their union. Detailed records and reproduction scripts
remain local under `stage23_gpu/runs/20261028_results_review/`.

The [October 29 follow-up](gpu_pipeline/20261029_followup.json) records the new
C170 (`418691…`) and C152 jobs. Both completed their audits: 1,142,421 of
1,142,423 initial outputs match CPU coefficients exactly, the two C170 differences
are within the 1e-6 score tolerance and outside the selected top 1000, and all
6,400 depth exports match CPU replay byte for byte. The report retains both raw
differing inputs, winning polynomials, settings, binary hashes and evidence hashes.

| Job | GPU pipeline wall time | Best GPU shortlist vs CPU: standard MurphyE | Best GPU shortlist vs CPU: lattice model |
|---|---|---|---|
| C170 (`418691…`) | 1h 05m 29s | +0.000085% | tied |
| C152 | 38m 01s | -0.00313% | +0.00918% |

**October 30 correction:** the earlier comparison reoptimized declared skews
before scoring them. It therefore did not measure the actual exported
configurations. At its exported skew C170 scores 582,868,400 in the independent
lattice model, above the old CPU control's 582,832,700; the claimed export
regression was incorrect. The table now retains declared configurations as well
as proposed skews for both methods. Historical evidence is preserved, with the
correction in [the integration report](gpu_pipeline/20261030_integration.json).

Final export now preserves all declared finalists and adds standard/job Murphy
skew proposals before independent fixed-skew scoring. Replaying the saved
finalists improves C170's export to 582,950,000 (+0.0140%) and C152's to
173,296,800 (+0.000866%). Both exports reproduce their scores on readback.
These are small model differences under inherited, provisional profiles,
without test sieving or matched CPU timings. Original run outputs remain intact;
replay artifacts are local under `stage23_gpu/runs/20261030_shipping/`.

The [size-descent diagnosis](gpu_pipeline/20261029_sopt_descent.json) records
the synthetic perturbation search (one exact hit among 16 seeds per case),
fresh-build reproduction, original-batch extraction and the host's FMA log
implementation. GPU validation reproduces both losses with the original code;
the matched-log prototype matches all 16,384 context outputs, all 46 historical
differences and all 13,000 C146 acceptance outputs to CADO. Full sources/build
manifests and traces remain ignored. The October 30 implementation selects and
checks the host's FMA/non-FMA log variant, checks the device before sopt, and
uses CADO on the CPU for unsupported host math. Host tests pass, as do new-build
device log checks, all 48 historical differences against current CADO, and both
8192-input batch contexts and target traces against CPU. All 28,000 C146/C161
effort-0/50 fixture outputs and all 8,028,269 outputs in the eight-job replay
also match their CPU references exactly. The subsequently added C150 job is a
separate comparison and is not part of that corpus.

The [C150 follow-up](gpu_pipeline/20261030_c150.json) finds that all three root
optimizers reached the same winning rotation, `u=3, v=2189298`. Msieve exported
it with coefficient gcd 2 and printed MurphyE `5.668e-12`, below its reported
leader `6.543e-12`. Dividing out that content reveals a common-score result of
`7.230907e-12`, effectively tied with GPU `7.230667e-12` and CADO `7.228422e-12`.
The comparison now retains every distinct nonprimitive msieve export before
common scoring, alongside three distinct printed-score leaders per family.
This corrects a shortlist loss; the prior C150 comparison remains preserved.

C150's initial audit differs from its imported CADO reference on 2,782 of
362,672 outputs, including 147 worse size scores. All 2,782 disputed rows match
the local CPU port and linked CADO exactly; the precise cross-build/environment
cause remains unisolated. All 3,200 depth exports match CPU replay byte for byte.
The original queue validation error is retained, and no new test sieve was run.

The [October 31 focused validation](gpu_pipeline/20261031_validation.json)
covers the subsequent review fixes: the CPU suite passes, and all 47,214
GPU polynomial outputs match their references, including the 2,782 C150
discrepancies against fresh CPU and linked CADO. Original batch traces, both
log variants, NDEBUG checks and aggregate fallback logging also pass. The
CPU suite found a refinement-output parsing bug, now fixed and covered by the
successful rerun. Builds and logs remain ignored. The eight-million-input
corpus was not rerun for this build.
