# Class-planner review fixes — October 27–28, 2026

The thirteen review findings are addressed. Dense and legacy class defaults
are unchanged; `-class-size-aware` remains a standalone opt-in experiment.

| Findings | Resolution |
| --- | --- |
| 1: wrong rows in the work limit | The size-aware path bounds actual cached coordinates, including coarse/hinted probes, at 65536. Remaining capacity is apportioned across selected rows; each row visits at most `maxlines` coordinates. Large `umax` no longer fails from hypothetical scans of unselected rows. The review's M=30, K=16, umax=100000 example passes with default maxlines. |
| 2, 13: distant basins and duplicated probing | A shared cached probe in `rscan.h` serves basin discovery and class planning. Selected progressions expand in both directions from their best discovered basin. The synthetic u=400000 test checks both neighboring directions, caching and the work cap. Documentation now calls these bounded local scans. |
| 3: non-finite starts | Warm starts must be finite. A finite cold result replaces an invalid warm result; invalid evaluations normalize to infinity. Shared-path tests inject NaN/INF and verify recovery. |
| 4: empty profile lookup | `nearest()` returns an empty list; the CLI reports a useful argument error without a traceback. Assumed-only catalogs still work with explicit `--include-assumed`. |
| 5, 6: misleading diagnostics | Trace output labels the scalar rank as score-only and separately reports actual selection position, policy and exploration status. Logs distinguish coarse/hinted/scanned coordinates, cache reuse, evaluations, retained lines and capped rows. The proposal bound respects M² and is explicitly an upper bound. |
| 7, 8: serial/repeated probes | Coarse and hinted batches run in parallel against an immutable cache snapshot, followed by serial insertion. Both stages and selected-row scans reuse evaluated coordinates. |
| 9, 10: avoidable work | The unconditional lookup was removed from the legacy scan, and rows with non-finite size hints are skipped before constructing their v proposals. |
| 11: export-test gaps | Tests check the input n, exact selected/exploration counts, class membership, resultants, deterministic exports and the reported actual selection position. |
| 12: weak heap test | A full-enumeration oracle checks the entire expected union of global, row-reserve and exploration proposals. It explicitly requires a heap-only expected candidate, so the other reserves cannot mask a broken global heap. |

The work limit is a coverage cap: a selected row can stop on patience, its
`maxlines` quota or its share of remaining cache capacity. Capped rows are
reported. These scans do not claim exhaustive coverage of a residue progression.

Validation completed:

- Full host suite, including the new profile tests, shared-probe tests and the
  existing 2100 scalar CADO evaluator comparisons.
- All 60 saved planning cases, with all 30 dense/legacy plans preserving their
  selected classes, exact cell counts, budget steps and trace boundaries.
- The large-umax regression with default maxlines; it uses 571 cached
  coordinates rather than rejecting the hypothetical full progressions.
- Six live GPU searches: 16/32 classes for C208 seed 222, C192's reference seed
  and C197 seed 1. CPU/GPU class selection, basin centers, probe accounting and
  admitted geometry agree. Sampled roots and full-refinement checks pass with
  zero device-refinement fallbacks. All 1200 exports match CPU pool replay byte
  for byte and their resultants verify.

Historical reference admission is unchanged on these 60 plans. This is a
correctness/coverage-control check, not a fresh equal-time quality study. The
October 27 prototype's earlier speed and MurphyE measurements remain attached
to its frozen binary and are explicitly marked pre-review.

Reproduce after building both binaries, using a fresh run directory:

```sh
make -C stage23_gpu test s23_ropt
python3 stage23_bench/runs/20261027_coverage_review/run.py host --run-dir NEW_RUN
python3 stage23_bench/runs/20261027_coverage_review/run.py gpu --run-dir NEW_RUN
```

The GPU phase checks availability before each search. Full snapshots, command
records, logs and pools are under `stage23_gpu/runs/20261027_coverage_review/`.
Compact evidence is in `stage23_bench/data/root_coverage/review_20261027/`.
