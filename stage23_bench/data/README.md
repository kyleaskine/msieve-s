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
