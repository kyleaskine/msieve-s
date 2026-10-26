# C197 GPU refinement production check — 2026-10-24

The new full GPU refinement and MurphyE quadrature paths completed an independent
root search after importing the finished CPU initial-size-optimization pass.
The search recovered the CPU winner's rotation family at a different translation
and skew. Quality is effectively tied: the GPU winner is 0.0103% lower in standard
MurphyE, and 0.0567–0.0938% higher in the provisional lattice model. No test
sieving was performed.

User settings: LPB 32/33, factor-base limits 160M/240M, algebraic special-q
75M–400M, and 2^30 sieve positions. The modeled cuda-sieve arguments are
`--logI 16 --J 16384`; MurphyE's equivalent I is 15.5. The lattice coordinates
are i in [-32768,32768), j in [0,16384), before the special-q basis transformation.
Factor-base limits are recorded for future test sieving; the model uses LPB,
area and lattice geometry. These settings remain provisional.

## Correctness and quality

- Imported 1,680,020 initial outputs; effort-50 GPU resopt used the same 1,000
  raw inputs as CPU and reproduced all 1,000 optimized polynomials exactly.
- Completed 314 breadth searches, 16 depth searches and four full-pool MurphyE
  refinements, with no GPU fallbacks or partial MurphyE failures.
- All 3,200 exports from the 16 depth pools match CPU bytes. Six intermediate
  flat-tie translations differ; none changes an export. Maximum checked alpha
  error is 7.11e-15 and weighted-key error is 1.42e-14.
- All 64 sampled MurphyE cells have identical independently evaluated scores
  at printed precision. Maximum checked integral relative error is 2.08e-14.
- The independent CPU comparison screened every CADO export and the three best
  msieve exports per rational family in each orientation, not every msieve export.
  CPU winners were never inserted into the GPU search.
- At 1,024,000 angles and frozen skews, standard MurphyE is 1.082618e-14 CPU
  versus 1.082506e-14 GPU. The separate lattice winners have GPU/CPU ratios
  1.000566712, 1.000938154 and 1.000881771 on streams 0, 1 and 3 with 512
  lattices and 8,192 points per lattice. Stream 0 overlaps the initial screen;
  streams 1 and 3 provide additional samples.

The normal pipeline's output is `pipeline_best.poly`. Independent comparison
also adjusted skew: `lattice_score_gpu.poly` is the GPU representative used in
the dense lattice checks, and `standard_murphy_gpu.poly` uses its standard-Murphy
skew. These are the same integer polynomial. The corresponding CPU controls
are included for a future matched test sieve.

## Timing and next work

Pipeline wall time was 3,376.8 s (56m 17s), excluding the initial CPU sopt,
221.4 s import, and subsequent audits/comparison. It includes GPU resopt
(74.8 s). With two workers and two host threads per process:

| Phase | Wall time |
| --- | ---: |
| Breadth search | ~1,119.7 s |
| Depth search | ~1,766.2 s |
| Full-pool MurphyE refinement | ~160.3 s |
| Independent CPU ranking, all pipeline phases | 249.7 s |

Search-phase wall spans are estimates from batch completion timestamps and
recorded process durations; overlapping seed timings are not wall times.
Breadth plus depth accounts for about 85.5% of pipeline wall time. The depth
logs attribute 3,395.7 of 3,435.9 aggregate seed-process seconds to GPU sieving.
Dense root-search work is therefore the leading speed target on this input.
An equal-time coverage comparison of selective residue-class searching against
this dense baseline should precede changing search defaults. Remaining host
numerical work includes Murphy search control and exact translations, rational
alpha, size/band planning, and lattice scoring.

There is no matched full CPU timing, so this run does not establish an
end-to-end GPU speedup. The initial CPU sopt pass was shared.

## Evidence and reproduction

- `execution.json`: completion, GPU use/fallback counts, binary hashes.
- `resopt_comparison.json`, `migration_audit.json`: numerical migration checks.
- `comparison.json`, `winner_validation.json`: independently scored CPU/GPU winners.
- `scoring.json`, `timing_summary.json`: settings and timing scope.
- Full artifacts: `stage23_gpu/runs/20261024_c197_gpu/`.
- Reproduction: `stage23_bench/runs/20261024_c197/README.md`.

Frozen CPU references were verified unchanged. Both new GPU paths remain opt-in.
