# C197 production test of GPU refinement

This study imports the completed CPU initial-size-optimization output, then runs
GPU effort-50 size optimization on its best 1,000 raw inputs, 300 parent seeds
plus content seeds, breadth/depth root searches, and full-pool MurphyE refinement.
Both `--refine-gpu` and `--murphy-gpu` are enabled, with two workers, two host
threads per process and batches of at most 16 seeds. Screening and selective-class
search remain disabled. CPU winners are comparison controls and are not inserted
into the GPU search.

User-supplied provisional settings (October 24): LPB 32/33, rlim/alim 160M/240M,
2^30 sieve positions, q roughly 75M–400M. The explicit cuda-sieve geometry is
`--logI 16 --J 16384`: i in [-32768,32768), j in [0,16384), followed by the
special-q basis transformation. MurphyE uses equivalent I=15.5, so its area is
2^30 * 75M. `scoring.json` records the geometry and is the source for comparisons.

Run from the repository root, choosing new run and archive directories:

```
python3 stage23_bench/runs/20261024_c197/run.py prepare --run-dir RUN
python3 stage23_bench/runs/20261024_c197/run.py preprocess --run-dir RUN
python3 stage23_bench/runs/20261024_c197/run.py pipeline --run-dir RUN
python3 stage23_bench/runs/20261024_c197/finish.py --run-dir RUN --archive-dir ARCHIVE
```

Preparation freezes the current `pipeline_results/`, `pipeline_work/`,
configuration and binaries. The driver snapshots the consumed initial-sopt bytes.
The standalone `breadth` phase is available while scoring parameters are pending;
the regular pipeline can subsequently reuse those checkpoints. The `pipeline`
phase can use `--wait-preprocess` when an import is already running. `finish.py`
waits for the pipeline, checks all depth pools against CPU full refinement,
audits sampled MurphyE pools and independently scores the final comparison.

`compare.py --run-dir RUN --cpu-only` prepares the CPU scoring cache during the
search. Its CPU shortlist includes every CADO output and three msieve outputs
per rational family from both original/inverted runs. It deduplicates sign-equivalent
polynomials for scoring. It is not an exhaustive rescore of all msieve exports.
Both standard MurphyE and the job lattice score choose their own skew; leaders
are re-evaluated at 256k angles and an independent lattice sample stream.

`validate_winners.py --run-dir RUN` holds those winners and skews fixed for
1,024,000-angle MurphyE and three 512-lattice/8,192-point sample streams. The
finish script runs this check and archives it alongside a timing summary.
Stream 0 overlaps the initial screen; streams 1 and 3 are additional samples.

The completed C197 study found an effective quality tie: GPU standard MurphyE
is 0.0103% lower; lattice score is 0.0567–0.0938% higher across those streams.
All 1,000 resopt outputs and 3,200 audited depth exports match CPU. Pipeline
wall time was 56m 17s, with about 48m 6s in breadth/depth searches. Compact
results and interpretation are in `stage23_bench/data/c197/gpu_ropt/README.md`.

Full artifacts: `stage23_gpu/runs/20261024_c197_gpu/`. The initial CPU sopt pass
is shared; the pipeline wall time is not a full raw-input GPU comparison, and
no matched CPU end-to-end speedup or actual sieve-yield result is implied.
