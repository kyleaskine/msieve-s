# stage23_bench: fixtures and analysis tools for GPU size-opt / root-opt

Companion to `GPU_STAGE23_PLAN.md` (the plan, the M0 results, and the design constraints).
This directory holds what is needed to rerun the M0 checks and to test a GPU sopt (M1) or
GPU ropt (M2) against the current msieve + CADO pipeline.

Conventions used throughout:

- **`exp_E` lower is better.** MurphyE / msieve `e` higher is better.
- **Y1 identifies a seed.** Translation and rotation never change Y1, so every sopt and ropt
  result of one stage-1 polynomial shares it.
- **A raw polynomial is identified by (Y1, Y0, c_d).** Use this to join sopt outputs.
- **CADO sopt returns a·f + r(x)·g, not just a translation and rotation of f.** Its LLL
  lattice includes f, so the result has c_d = a·c_d(raw) and Res(f, g) = a·N. Most results
  have |a| > 1. ropt usually keeps a, but CADO ropt divides out content, which shrank a
  on 7 of 150 c146 seeds per pass (e.g. −9 → −3); `rotation_between` returns None
  for those, and the tools skip them.

## Layout

```
tools/            Python 3, no dependencies beyond the standard library
  polyfmt.py        parsers (CADO sopt / ropt, msieve -npr .p, CADO poly files), exact helpers
  sopt_compare.py   join sopt outputs on the raw poly: exp_E change, multiplier stats, top-K shifts
  seed_funnel.py    per-seed best ropt score vs exp_E rank; writes the per-seed table
  rescore.py        MurphyE under CADO default vs job parameters, skew optimized per poly
  sopt_sweep.py     forced-multiplier and pre-rotation sweeps through CADO sopt
  rotation_diff.py  exact (t, r(y)) between two results of one seed, and Res/N
  alpha_proto.py    CPU reference alpha (exact, matches CADO) and the root-sieve counting
                    rule a GPU ropt must use at g's root; validate / sieve-check / benchmark
  alpha_window.c    C line sieve: affine alpha over p < 2000 for every cell of a u-line
                    window (CPU reference for the GPU ropt sieve; built on first use)
  prefilter_window.py  exhaustive windows around the benchmark optima: does a small-prime
                    prefilter (classes mod 2520) lose the best cell?
  make_fixture.py   builds data/<job>/ from a pipeline run
  fixture_raw.py    writes a fixture's raw polys as CADO-format input for any sopt
  cado_expe.c       full-precision lognorm / exp_E / alpha from CADO's own code (built on
                    first use against the configured CADO build); M1's scorer
data/
  c146/             2026-10-02 c146 run (full data below)
  c161/             2026-10-03 c161 run (full data below)
  c204/             2026-10-02 c204 run (the three winners only; the rest was deleted)
```

Run the tools from the repository root; their defaults are the pipeline's file names
(`cado_sopt_unsorted.txt`, `pipeline_work/`, `pipeline_results/`). CADO binaries are found
from `$CADO_BUILD_DIR`, else `cado_build_dir` in `nfs_config.ini`; override with `--sopt` /
`--score`. `sopt_compare.py` rejects any result that is not a valid polynomial for N (Y1
changed, a zero or non-multiple leading coefficient, or Res(f, g) not a nonzero multiple
of N) and exits 1. Plain runs compare CADO's printed 2-decimal exp_E; `--rescore`
rescores with `cado_expe` at full precision (M1's rule) and also exits 1 if any row is
worse or a fixture row of a covered set is missing.

## data/c146

N = 3662801383…034597099 (146 digits, in `n.txt`), degree 5, 371,526 deduped raw polys from
msieve `-np1 -nps`. Pipeline run with top 1000 re-sopt at effort 50, msieve ropt on the best
300, CADO ropt (effort 5) on the best 150, 8 threads. CADO 0574bc39d, CADO parameters
`params.c145` (lpb 30/30, I = 14, qmin = 2e6). Extra runs for M0: effort 50 on ranks
1001–3000, msieve ropt on ranks 301–1000, CADO ropt on the 10 best seeds past rank 150.

| File | Contents |
|---|---|
| `n.txt` | N |
| `sopt_sample.tsv.gz` | 13,000 raw polys: the top 3000 by effort-0 exp_E (`set=top`) and a random 10,000 of the rest (`set=random`, seed 1). Each row: raw coefficients, CADO's effort-0 result, and the effort-50 result for the top 3000. Columns are listed in `make_fixture.py`. **M1's acceptance set.** |
| `ropt_seeds.tsv.gz` | Every seed of effort-50 exp_E ranks 1–1000, in the pipeline's own order (`resopt_msieve_sorted.ms`): rank, Y1, exp_E, multiplier, `msieve_ran`, `cado_ran`, best msieve orig / inv `e`, best CADO orig / inv MurphyE (CADO default parameters). An empty score with ran = 1 means the tool ran and printed nothing (msieve drops results below its `min_e`; 476 of the 1000 seeds have no msieve result); ran = 0 means not run. **M2's per-seed baseline.** |
| `winners/` | The winning seed (Y1 = 18436463496569028067, exp_E rank 84, multiplier −2): `raw.poly`, `sopt.poly`, and the best `msieve_best`, `cado_orig_best`, `cado_inv_best` results. |
| `test_sieve/` | Seven finalists test-sieved with CADO `las` and CADO's c145 parameters (a stand-in for a real job file): the polys, the exact command, and per-range relations per special-q and CPU s/rel. See its README.txt. |

Benchmark for this seed: msieve's winner scores **1.2391e-11** under skewopt (CADO-inv found
the same poly, translated). CADO-orig found a separate optimum at u = 120,
1.2133e-11. The second-best seed overall is exp_E rank 407 (1.225e-11), outside the
pipeline's top 300.

## data/c161

N = 7326172377…774589164086067 (161 digits, in `n.txt`), degree 5, 814,307 deduped raw
polys from msieve `-np1 -nps` (2026-10-03 run). Same pipeline settings as c146: top 1000
re-sopt at effort 50, msieve ropt on the best 300, CADO ropt (effort 5) on the best 150,
8 threads, CADO default parameters for MurphyE.

| File | Contents |
|---|---|
| `n.txt` | N |
| `sopt_sample.tsv.gz` | 11,000 raw polys: the top 1000 by effort-0 exp_E (`set=top`, with CADO's effort-50 results from the pipeline's re-sopt) and a random 10,000 of the rest (`set=random`, seed 1). Built before any stage23_gpu code saw this number: a clean second acceptance set. |
| `ropt_seeds.tsv.gz` | As for c146: every seed of effort-50 exp_E ranks 1–1000 in the pipeline's order, with ran flags and best scores. msieve ran on ranks 1–300 (3 printed nothing), CADO on 1–150. |
| `winners/` | The winning seed (Y1 = 419093078612002505953, exp_E rank 16 of 1000, 42.12 vs 41.34 best, multiplier −8): `raw.poly`, `sopt.poly`, and the best `msieve_best`, `cado_orig_best`, `cado_inv_best` results. |

Benchmark for this seed: CADO's winner scores **1.6246e-12** under skewopt (CADO-inv found
the identical poly, negated). msieve's best is a separate optimum, 1.6101e-12; it is at
u = 0 like CADO's (t = −658,490, v = −1,219,680 from CADO's), so on this seed the two
optima differ by translation and v only, not by a large u as on c146 and c204. Both
tools rank the same seed first and exp_E rank 2 second (1.49e-12, 8% behind).

## data/c204

N in `n.txt`. The three winners of the 2026-10-02 c204 run, all from one seed (Y1 =
552226208798178007565897, multiplier 2): `A_msieve.poly` (5.019e-15), `B_cado_orig.poly`
(5.160e-15, 5.194e-15 after skewopt, at u = 90 from A) and `C_cado_inv.poly`. **Target: beat
5.194e-15.** The raw polys and other results of this run were not kept.

## M1 acceptance (GPU sopt vs CADO)

```bash
python3 stage23_bench/tools/fixture_raw.py stage23_bench/data/c146/sopt_sample.tsv.gz /tmp/raw.poly
# ... run the GPU sopt (or CADO's sopt -sopteffort 0) on /tmp/raw.poly -> RESULT ...
python3 stage23_bench/tools/sopt_compare.py stage23_bench/data/c146/sopt_sample.tsv.gz RESULT --rescore
```

A row passes if the polynomial is identical to CADO's effort-0 result, or valid and no
worse in exp_E when both are recomputed at full precision by CADO's code (`cado_expe`).
Self-test: CADO's own sopt on the exported raws gives 13,000/13,000 identical (pass
100%). Against the effort-50 results (`--effort e50`, or effort-50 outputs as OTHER),
CADO's effort 50 is slightly worse than its effort 0 on 2 of 3000 at full precision
(by 0.0015 and 0.0006), so higher effort is not monotone either.

## Reproducing the M0 checks (c146 numbers in GPU_STAGE23_PLAN.md)

```bash
T=stage23_bench/tools
# effort 0 vs effort 50 (and the multiplier distribution)
python3 $T/sopt_compare.py cado_sopt_unsorted.txt pipeline_work/resopt_output.txt
# exp_E rank vs ropt outcome, in the pipeline's seed order; --ran says which seeds each
# tool ran on (its selection file), so seeds without a result count as worst instead of
# being dropped; --tsv writes the table
python3 $T/seed_funnel.py --sopt pipeline_work/resopt_output.txt \
    --rank-file pipeline_work/resopt_msieve_sorted.ms \
    --msieve-orig pipeline_results/msieve_ropt_orig.p --msieve-inv pipeline_results/msieve_ropt_inv.p \
    --cado-orig pipeline_results/cado_ropt_orig.txt --cado-inv pipeline_results/cado_ropt_inv.txt \
    --ran msieve=pipeline_results/best300_msieve.ms --ran cado=pipeline_results/best150_cado.txt
# default vs job MurphyE parameters
python3 $T/rescore.py --n-from pipeline_results/best150_cado.txt \
    --msieve pipeline_results/msieve_ropt_*.p --cado pipeline_results/cado_ropt_*.txt \
    --params ~/cado-nfs/parameters/factor/params.c145
# forced multipliers (about 10 min on 6 cores) and pre-rotation (about 8 min on 8)
python3 $T/sopt_sweep.py --mode mult --range 500 --top 300 --out /tmp/mult --emit-alt /tmp/alt.ms
python3 $T/sopt_sweep.py --mode prerot --range 256 --top 300 --out /tmp/prerot --procs 8
# exact relation between two results of one seed
python3 $T/rotation_diff.py stage23_bench/data/c146/winners/msieve_best.poly \
    stage23_bench/data/c146/winners/cado_orig_best.poly
```

Test sieving uses the GPU siever's script, `~/code/cuda-sieve/bench/testsieve.sh`
(job file with the parameters you would actually sieve with; q in millions; it can also
sweep sieve geometry and factor-base bounds). It keeps the whole flow on the GPU and is
much faster than the CPU sievers. The c146 test sieve in `data/c146/test_sieve/` was run
earlier with CADO `las` and is kept for reference.

To run sopt or ropt on more ranks, cut the ranked files the way the pipeline does
(`pipeline_work/resopt_msieve_sorted.ms` is the effort-50 ranking in msieve format; the
pipeline's top 300 is its first 300 lines; the CADO top 150 comes from
`resopt_sorted.txt`, whose tie order can differ, so use `best150_cado.txt` itself as the
record of what CADO ran) and use `scripts/run_msieve_ropt_annotated.sh`
or CADO's `polyselect_ropt` directly.

## Fixtures from a new run

```bash
python3 stage23_bench/tools/make_fixture.py --out stage23_bench/data/cNNN \
    --base cado_sopt_unsorted.txt --e50 pipeline_work/resopt_output.txt --ropt-dir pipeline_results
python3 stage23_bench/tools/seed_funnel.py --sopt pipeline_work/resopt_output.txt \
    --msieve-orig pipeline_results/msieve_ropt_orig.p --msieve-inv pipeline_results/msieve_ropt_inv.p \
    --cado-orig pipeline_results/cado_ropt_orig.txt --cado-inv pipeline_results/cado_ropt_inv.txt \
    --rank-file pipeline_work/resopt_msieve_sorted.ms \
    --ran msieve=pipeline_results/best300_msieve.ms --ran cado=pipeline_results/best150_cado.txt \
    --tsv /tmp/seeds.tsv && gzip -n -c /tmp/seeds.tsv > stage23_bench/data/cNNN/ropt_seeds.tsv.gz
```

The full sopt outputs are too large to check in (`cado_sopt_unsorted.txt` was 373 MB for
c146). Keep them locally while a job's M1 comparison is open.
