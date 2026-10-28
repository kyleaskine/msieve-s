# Size-aware class proposals — October 27, 2026

This is the frozen **pre-review prototype** study. The subsequent review fixed
basin centering, probe reuse/guards, work limits and diagnostics. The measurements
here remain valid for the archived binary and do not establish the corrected
planner's quality or speed. Follow-up validation is recorded in
`../20261027_coverage_review/`.

This experiment adds standalone `-class-size-aware` to `s23_ropt -classes M K`.
It addresses the root-only proposal cutoff diagnosed in the October 25 study.
Dense search and the existing class policy retain their defaults.

The policy samples at most 129 u coordinates on a cubic grid, including zero
and the requested endpoints. Linear interpolation estimates the attainable
size on each u residue. These estimates enter the combined size/root score
before proposal truncation. The bounded pool combines the leading 2*K classes
with the best class from each of the leading 2*K distinct u residues, plus the
existing deterministic exploration proposals. Guaranteed content is corrected
in the score. The interpolation is a heuristic, not a certified lower bound.

Each proposed row receives a local size probe at its estimated best lift,
using nearby and zero starts. Nearby admissible v coordinates replace the
coarse estimates before class selection. Half of the ranked slots preserve u
diversity. The prototype scans selected progressions outward from zero, keeping
the earlier hinted probe if better; the review identified that this does not
expand distant hinted basins. All selected classes share the existing cell,
candidate, refinement and output budgets. Neither proposals nor outputs are
nested as K changes; more classes need not preserve a particular winner.

The planning audit includes the ten previous controls, the C179 msieve-original
winner, and the first two depth seeds from each new C170/C184 run. The latter
four seeds were selected before measuring this policy and were not used to
design it. Reference coordinates enable tracing only; they never force a
proposal, scan point or survivor. All 20 overlapping dense/legacy plans match
the October 25 audit's classes, cell counts, budget steps and loss boundaries.

The GPU comparison uses eight seeds: C192, both C197 controls, C179 and the four
new-input seeds. It compares dense breadth, the existing 32-class policy, and
the new 16/32-class policies modulo 2520. Each gets a 2e11-cell cap, 4096 exact
refinements and 200 exports. Three repetitions rotate mode order, with one GPU
process and two CPU threads. Timings exclude subsequent scoring and CPU replay.
The configurations have equal caps/quotas, not equal actual elapsed budgets.

Independent scoring evaluates every unique export at 16k search/64k evaluation
angles and validates the leaders at 256k. The provisional lattice model uses
two independent validation streams with 256 lattices and 4096 points per q.
C179 inherits the nearest C181 profile provisionally; the other cases retain
their saved job profiles. No test sieving is performed. Every first-repetition
pool is also replayed through CPU refinement and its exported bytes compared.
For quality losses, `trace_quality.py` follows the dense search's actual MurphyE
leader through the new policy. This is separate from the historical traces,
which sometimes identify an exact-alpha winner rather than the MurphyE leader.

Reproduce from the repository root with a fresh run directory after building
the CPU and CUDA tools:

```sh
python3 stage23_bench/runs/20261027_root_coverage/study.py plan --run-dir NEW_RUN
python3 stage23_bench/runs/20261027_root_coverage/study.py pilot --run-dir NEW_RUN --repeats 3
python3 stage23_bench/runs/20261027_root_coverage/validate.py --run-dir NEW_RUN
python3 stage23_bench/runs/20261027_root_coverage/study.py score --run-dir NEW_RUN
python3 stage23_bench/runs/20261027_root_coverage/trace_quality.py --run-dir NEW_RUN
python3 stage23_bench/runs/20261027_root_coverage/archive.py --run-dir NEW_RUN --archive-dir NEW_ARCHIVE
```

The study freezes inputs, binaries, source headers and profile snapshots. It
reuses the previous planning/timing/scoring harnesses with explicit study
configurations and verifies their hashes on resume. GPU availability is checked
between searches; this is not an exclusive hardware reservation. Full artifacts
live in `stage23_gpu/runs/20261027_root_coverage/`.

## Results

All 60 CPU plans and 96 GPU searches completed. The three repetitions reproduce
their pools and exports exactly; 19,200 exported resultants verify, 128 sampled
root scores per search agree with brute force, and there are no device-refinement
fallbacks. CPU replay of every first-repetition pool reproduces all 6,400 exports
byte for byte, including the selected classes and admitted class geometry.
The full host suite and four CUDA automatic-class CLI tests pass.

Independent scoring covers 5,273 unique configurations. Times below are process
medians in seconds; MurphyE changes are against each seed's dense breadth run.

| Seed | Dense s | New 16 s | New 16 MurphyE | New 32 s | New 32 MurphyE |
| --- | ---: | ---: | ---: | ---: | ---: |
| C192 winner seed | 6.733 | 1.877 | -3.687% | 2.837 | -3.687% |
| C197 s0005 | 3.879 | 1.890 | tied | 2.698 | tied |
| C197 s0001 | 3.915 | 1.773 | -5.184% | 2.426 | -4.409% |
| C179 winner seed | 4.355 | 1.470 | -11.229% | 2.151 | -0.00009% |
| C170 s0009 | 2.992 | 1.141 | tied | 1.235 | tied |
| C170 s0010 | 4.808 | 1.264 | -4.552% | 1.963 | -4.552% |
| C184 s0001 | 3.072 | 1.413 | -3.168% | 2.316 | -3.168% |
| C184 s0003 | 3.463 | 1.468 | tied | 2.043 | tied |

The new policy improves several old class-policy losses: C192 goes from a
14.080% MurphyE deficit to 3.687%; C197 s0001 from 11.972% to 4.409–5.184%;
C170 s0010 from 14.852% to 4.552%; and C184 s0003 from 7.144% to a tie.
However, C179's 16-class run is worse than the old class policy, and C184 s0001
loses more lattice-model quality despite a small MurphyE improvement. The new
32-class mode's largest lattice deficit is 3.49–3.57% on C170 s0010. Neither
mode is a generally safe substitute for dense search.

Tracing the original references alone would overstate success. For example,
the new 16-class mode retains C197 s0001's historical exact-alpha winner, but
its dense MurphyE leader is a different v rotation and the quality gap remains.
The size-aware policy also drops C208 seeds 15/27 at proposal and seed 103 at
class selection, where the old 32-class policy admitted their references.

Keep dense as the production default. The next experiment should preserve a
dense control and a root-driven proposal reserve while adding bounded size/u
diversity. It must track actual MurphyE/lattice leaders and competing v residues,
not just the first exact-alpha finalist. Select the next policy using these
diagnostics, then evaluate it on new holdout seeds and matched elapsed budgets.

Results, settings, winner coefficients, provenance hashes and four new seed
fixtures are retained in `stage23_bench/data/root_coverage/20261027/`. Full logs
and source snapshots are local under
`stage23_gpu/runs/20261028_commit_evidence/root_coverage/20261027/`;
inputs/pools stay in the original run directory. The [evidence index](../../data/README.md)
describes the consolidation. A fresh checkout is not a complete historical
replay environment. Use an ignored destination under `stage23_gpu/runs/` when
exporting another full snapshot with `archive.py`.
