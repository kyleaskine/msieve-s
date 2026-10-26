# Root-search coverage and timing pilot — October 25, 2026

This study uses the validated grouped-lookup CUDA binary as the new dense
baseline. It measures why the experimental class search cannot yet replace it.
No production search code or defaults change.

The CPU audit runs nine configurations on ten saved seeds: seven C208 loss
controls, the C192 effort-5 winner's seed, and C197 seeds s0005/s0001. It compares
dense breadth/depth, coarse-basin breadth/depth, 16/32/64 classes modulo 2520,
and 16 classes modulo 420/210. Every plan records its exact selected-cell count,
budget staircase, class diversity and first loss boundary. The reference
coordinates only enable tracing; they never force a proposal or survivor.

The proposal probe enumerates the unchanged conditional-root ranking separately
from the CLI. Its thirty full-enumeration audits agree with all fifty applicable
class plans. At modulus 2520, the C192 winner ranks 1723rd and C197 s0001's
dense winner ranks 539th before size is considered. Each ranks first within
its own u residue. Neither enters even the 64-class proposal pool. C197 s0001's
root-score gap from the maximum is only 2.38e-8, so the early cut also makes
very small table-rounding differences consequential.

The GPU pilot compares dense breadth against 16 and 32 classes modulo 2520 on
C192 and the two C197 seeds. It uses one GPU process, two host threads, three
rotated-order repetitions, 4096 fully refined candidates, and 200 exports per
search. All 27 searches pass 128 sampled brute-root checks, use zero refinement
fallbacks, and reproduce their pools and exports exactly across repetitions.
All 5400 exported resultants verify.

| Seed | Dense seconds | 16-class seconds | 32-class seconds | Class MurphyE / dense | Class lattice / dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| C192 winner seed | 4.939 | 2.808 | 4.969 | 0.85920 | 0.91490–0.91511 |
| C197 s0005, overall winner | 2.784 | 2.667 | 4.658 | 1.00000 | 1.00000 |
| C197 s0001 | 3.002 | 1.861 | 3.466 | 0.88028 | 0.92920–0.92931 |

Both class counts have the same displayed quality ratios. Times are process
medians and exclude independent scoring. All modes have the same refinement
and scoring quotas; elapsed time was measured, not forced equal. C192's dense
and 32-class runs happen to be within 1% in time. This is a diagnostic pilot,
not a holdout or general equal-time comparison.

Independent CADO scoring covers every export: 1334 unique configurations across
the three cases, at 16k search/64k evaluation angles. The top four per mode and
objective are validated at 256k angles and 256 lattices × 4096 points per q.
Lattice stream 2 chooses the leader; stream 3 checks that fixed choice. The
recorded provisional C192/C197 geometry and q ranges are reused. Each model
uses its own bounds for skew optimization; lattice skew is chosen by job-bounds
MurphyE, not directly optimized for lattice yield. There is no test-sieve result.

Planning now dominates the small class searches: C197 s0005's 32-class run
spends 4.43 seconds in the size model and only 0.02 seconds sieving. Simply
adding classes can increase cost without restoring excluded families. The
next implementation should make proposals consider attainable size before
truncation, retain diversity across u residues, and bound the extra modeling
cost. Keep dense coverage as a control and evaluate the new policy on independent
seeds before changing defaults. The coarse-basin configurations and smaller
moduli did not solve the observed losses across this corpus.

Reproduce from the repository root with fresh output directories:

```
python3 stage23_bench/runs/20261025_root_coverage/audit.py plan --run-dir NEW_RUN
python3 stage23_bench/runs/20261025_root_coverage/proposals.py --run-dir NEW_RUN
python3 stage23_bench/runs/20261025_root_coverage/pilot.py --run-dir NEW_RUN
python3 stage23_bench/runs/20261025_root_coverage/score.py --run-dir NEW_RUN
python3 stage23_bench/runs/20261025_root_coverage/archive.py --run-dir NEW_RUN --archive-dir NEW_ARCHIVE
```

Preparation uses repository C192/C208 fixtures and the retained C197 seeds/depth
exports under `stage23_gpu/runs/20261024_c197_gpu/`; it needs no preprocessing
indexes or raw-stage input. It freezes binaries, source headers and input bytes.
The probe uses the configured CADO build to compile its small host helper.
GPU execution checks for other compute processes between searches and stops if
one appears; this is not exclusive hardware reservation. Successful records can
be reused, while incomplete or changed outputs require inspection.

Full artifacts: `stage23_gpu/runs/20261025_root_coverage/` (about 30 MiB).
Compact measurements, finalist coefficients and all ten seed fixtures:
`stage23_bench/data/root_coverage/20261025/`. These are sufficient to preserve
the numerical evidence without retaining another large preprocessing dataset.
