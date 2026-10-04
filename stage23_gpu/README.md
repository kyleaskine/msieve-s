# stage23_gpu: GPU size optimization (M1) and root optimization (M2)

Standalone tools, built separately from msieve. The plan, decisions and benchmarks are in
`../GPU_STAGE23_PLAN.md`; fixtures and the acceptance tooling are in `../stage23_bench/`.

## How this is tested

All device code lives in `include/*.h` as `__host__ __device__` templates. The same
source compiles with g++ and runs on the CPU, so it is unit-tested against GMP and
against CADO's own code without a GPU (CUDA has no built-in emulator any more). The
device harness then only has to show the GPU computes what the host build computes.

```bash
make test                 # host: Int<L> vs GMP, sopt LLL vs CADO's utils/lll.c, root-sieve
                          # tables vs brute force and projective alpha vs CADO, block top K
                          # vs an exhaustive sort, bench regressions (regress.py)
make device CUDA=120      # compile the device harness (does not touch the GPU)
make run-device           # run it: GPU results vs the host build (needs a free GPU)
make accept-cpu           # M1 acceptance: s23_sopt (CPU build) on 13,000 c146 polys vs CADO
make accept-gpu           # the same with the GPU build (needs a free GPU)
make accept-cpu EFFORT=50 SET=top   # effort 50 on the 3000 polys CADO ran at effort 50
```

`s23_sopt` (`tools/s23_sopt.cu`) reads raw polynomials and writes CADO sopt's format,
so the pipeline and `stage23_bench/tools/sopt_compare.py` read it unchanged:

```bash
build/s23_sopt -sopteffort 0 -inputpolys msieve.dat.ms > sopt_out.txt     # GPU
build/s23_sopt_cpu -t 8 -sopteffort 0 -inputpolys msieve.dat.ms > ...      # CPU build
```

Each batch runs the five phases of `include/sopt.h` (prepare, lll, dedupe, descent,
reduce: CADO's loop split into per-polynomial and per-candidate work, with CADO's order
kept where it matters, so ties come out the same). The GPU runs each phase over the
whole batch; the CPU build runs the same five functions back to back for one polynomial
at a time (polynomials in parallel), which keeps a polynomial's candidates in cache. Any polynomial the port flags
(overflow, loop or recursion caps) is redone by CADO's own `size_optimization` on the
CPU and reported on stderr; `-compare` also runs CADO on every polynomial and counts
identical results. Options:

- `-t N` (default 8): host threads for the CPU path, the CADO fallbacks, and CADO's
  stats and formatting of each batch (in parallel, written in input order).
- `-dev N`: the GPU to use (default 0).
- `-slice S` (default 10): on the GPU each phase is a work queue that warps take 32 items
  at a time from; a launch stops taking items after S seconds and is relaunched until the
  queue is empty, so no launch comes near the 60 s WSL2 watchdog. A warp's first chunk
  ignores the slice only while its launch has taken nothing yet, so every launch makes
  progress, and blocks that start late (a phase that runs in waves) stop at once.
- `-scratch F` (default 0.5): the GPU's per-worker scratch (about 220 KB each) is capped
  at this fraction of the VRAM free after the candidate slots; fewer workers only means
  fewer warps in flight. A candidate whose lll or descent never ran (an item the driver
  missed) fails its polynomial (redone by CADO) instead of being skipped or read stale.
- `-batch N` (default 4096): polynomials per batch. On the GPU it is lowered so that the
  candidate slots (`sopt_max_cand(effort)` per poly, 600 B each) stay under 1.5 GB and
  40% of the free VRAM (about 1,500 polys at effort 50 on an idle 5070).
- `-sopteffort E`: 0 to 100 (the translation list is sized for 100; CADO's sopt above
  that).
- A malformed input block stops the run (exit 1) after the polynomials before it, as
  CADO's sopt does; blank lines between blocks are skipped.

The Makefile gets CADO's include flags and libraries from
`stage23_bench/tools/polyfmt.py` (`CADO_BUILD_DIR`, else `cado_build_dir` in
`../nfs_config.ini`), the same code the bench tools use, and relinks when CADO's
libraries change.

## Layout

```
include/
  sopt.h        CADO's size_optimization (degree 5) as five phases (prepare, lll,
                dedupe, descent, reduce): translation candidates, best_norm2 (iterative,
                frame stack in scratch), local descent, expected_growth,
                expected_rotation_gain, exp_E selection; effort > 0 adds Farey candidates
  rsieve.h      M2 root sieve: per-prime tables of the alpha of f + (u x + v) g along
                one u-line (affine roots to prime powers <= 200 with the special rule at
                g's root; projective roots to p^(e+3), tabulated once per seed), and a
                brute-force reference
  rsize.h       M2 size model (host): lognorm after re-translation, per-line minimum,
                band and knots
  rseed.h       host helper: a CADO-format poly to the root sieve's per-prime residues
                (shared by s23_ropt and the tests)
  rblock.h      M2 sieve blocks, kept cells, the order of cells in a block and the CPU
                block choice (exact top K), shared by s23_ropt and test_rselect
  norms.h       CADO's L2 lognorm and L2 skewness (polyselect_norms.c)
  dpoly.h       CADO's double polynomial root finder (double_poly.cpp)
  alpha_proj.h  CADO's projective alpha (p < 100): exact discriminant (Bareiss),
                special_val0 recursion, both valuation code paths; expected_alpha
  hd.h          host/device macros (S23_HD inlined; S23_HD_CALL real calls on the device)
  mpint.h       Int<L>: L x 32-bit two's-complement integers; add/sub/neg/mul with
                overflow reporting, floor and exact division (Knuth D), mpz_get_d-style
                truncating conversion
  lll_exact.h   exact-integer LLL, a port of CADO utils/lll.c (NTL LLL_ZZ) for
                independent rows, delta = 1
  sopt_lll.h    CADO's best_norm: translate by k, build the skew-scaled rotation lattice,
                LLL, choose the shortest row with a degree-d term, extract a*f + r*g
tools/
  s23_sopt.cu      the size optimization tool (GPU build with nvcc, CPU build with g++)
  s23_ropt.cu      root-sieve experiments: window, search (size model + sieve + proxy
                   ranking, -out for CADO scoring), cell
  cado_io.c/.h     C shim to CADO: read polys, print sopt format, CADO's skew, CADO fallback
tests/
  test_mpint.cpp   Int<L> vs GMP on random operands (L = 2 .. 128), overflow edges
  test_lll.cpp     sopt_best_norm vs CADO's LLL on the same matrices
  gen_cases.py     LLL cases from the c146 fixture and the c204 winners
  test_device.cu   the same sopt_best_norm on the GPU vs the host build
  test_rsieve.cpp  root-sieve tables (affine and projective) vs brute-force root counting on
                   the benchmark winners; projective alpha vs CADO's get_alpha_projective
  test_rselect.cpp the CPU block choice vs an exhaustive sort (ties, short blocks, slopes)
```

## Status (2026-10-03, M1 steps 3-4)

- `s23_sopt`, CPU build, M1 acceptance on the c146 fixture:
  - effort 0, 13,000 polys: **13,000 identical to CADO** (bit for bit), 0 redone by
    CADO. (Before the rounding fix below: 12,996 identical and 4 off by a constant
    rotation, exp_E within 1e-7, which were rounding differences, not ties.)
  - effort 50, the 3000 polys CADO ran at effort 50: **3,000 identical** after the
    rounding fix (2,993 identical and 7 no worse before it), 0 redone.
  - speed: 13,000 polys in 22 s on 8 threads (about 14 ms of CPU per poly, CADO 11);
    effort 50 about 1 s of CPU per poly (CADO 0.67). The fixed 4096-bit LLL integers
    cost this on the CPU; the GPU build is the one that matters.
- GPU build, first runs (RTX 5070):
  - LLL harness: all 24,343 cases identical to the host build (0.2 s).
  - M1 acceptance, effort 0, 13,000 polys: after the rounding fix, **13,000 identical to
    CADO**, 0 redone by CADO (before it: 12,999 identical, 1 no worse).
  - Speed: 248 s for 13,000 polys (19 ms each), about 11x slower than the CPU build on
    8 threads. One thread per poly: a single thread needs up to ~20 s for one poly, so a
    1,024-poly launch takes ~29 s; the default batch is 1,024 to stay under the 60 s
    WSL2 watchdog. Peak VRAM ~6.5 GB, mostly CUDA's stack reservation (64 KB x every
    thread that can be resident).
  - Effort 50 has not been run on the GPU: ~100x the per-thread work would take far
    longer than the watchdog allows. Splitting the work per translation candidate across
    threads (design constraint 4) fixes this and is the first speed step.
- c161 (a second number, unseen during development): CPU build on all 814,307 polys,
  effort 0: all identical to CADO, 0 fallbacks (3,110 s optimizing on 8 shared threads,
  ~1,570 s more in the single-threaded CADO stats printing); effort 50 on the top 1000:
  1,000/1,000 identical. GPU build on the c161 fixture (11,000): all identical.
- Caps (each falls back to CADO): best_norm2 depth 128 (one c146 poly needs more than
  32 and at most 64), special_val0 depth 12 (the sample needs at most 10), translation
  candidates sized for effort 100 (3,265).
- Superseded below: the whole-poly-per-thread kernel and its `-budget` guard.
- Review fixes (2026-10-03, CPU build; the GPU was busy with LA): stats and formatting
  now run on all host threads; the c146 13,000 at effort 0 went from 59 s to 45 s with
  byte-identical output. `-budget` tested on the CPU (every poly out of time -> CADO,
  output identical). On the GPU: `make accept-gpu` 13,000/13,000 identical; default
  budget 40 s, 219 s optimizing (17 ms per poly), 0 out of time; `-budget 1`: 7,315 of
  13,000 out of time and redone by CADO, output still identical to the CPU build's;
  `-dev` with a missing GPU exits 2.

## Status (2026-10-04, M2: breadth run on the c168 job)

Details and numbers in `../GPU_STAGE23_PLAN.md` (M2, "Breadth run, c168").

- `s23_sopt` on the whole c168 job: effort 0, 772,402 raw polys, byte-identical to CADO,
  14:32 wall vs CADO's 25.6 min on 8 threads; effort 50 on the top 1000, identical,
  about 65 s vs 2.5 min (idle GPU).
- `s23_ropt -search` over the 1000 re-sopt seeds plus 31 content seeds: no seed beats
  the job's winner (found as the search's #1). Against CADO's ropt on its 150 seeds it is
  competitive, with identifiable gaps: better 31, tie 106, worse 13 at 16,000 MurphyE
  sample points (30/106/14 at CADO's 1,000). It is ahead of msieve's on most of its 300.
  Idle-GPU sieve
  3.4e10 cells/s; the host size model is now the bottleneck: seeds 0-299 (plus 13
  content seeds) take 35.6 min wall with 4 workers, about 75% of it size model
  (msieve's two ropt passes over the same 300: 45 min).
- New search options: `-aw W` (proxy lognorm + W alpha; 1.3 fits MurphyE), `-maxcells C`
  and `-maxlines M` (bounded work per seed, see the comment in `tools/s23_ropt.cu`);
  blocks are split at the knots and chosen by the proxy, not by alpha alone; content is
  divided out of written cells (content lattices are searched as their own seeds,
  (f + (u0 x + v0) g)/p).
- Code review fixes (all retested: c146 acceptance at effort 0 and 50 on CPU and GPU,
  `make test`, GPU vs CPU ropt, brute-force checks): duplicate cells from the top-K
  reduction; sieve launches bounded to about 2^36 cells whatever `-seg`; `-check` now
  checks the sieve's own scores; sopt scratch capped (`-scratch`), every relaunch makes
  progress, a skipped descent falls back to CADO; the CPU sopt build runs the phases per
  polynomial again: c146 effort 0 on 8 threads 40.6 s (43.3 s wall), back to the old
  sequential speed (42 s; the batch-phased CPU path took 54 s), still 13,000/13,000
  identical.
- **Clocks on this WSL2 machine:** CLOCK_MONOTONIC (`steady_clock`, which the tools'
  own timings use) runs 3-5% fast against the Windows host; wall time from `date` or
  `/usr/bin/time` matches the host. So the tools' "optimizing", sieve and size-model
  times read 3-5% high (a GPU effort-50 run printed 64.5 s optimizing in 63.3 s of wall
  time). Compare wall times with wall times.
- **Review checks and fixes (2026-10-04, details in the plan's "Review checks" and
  "Fixes"):**
  - MurphyE at CADO's 1,000 sample points is noisy: it reads high by a median 0.17%, up
    to 1.4%. The translation gains and multiple skew peaks found earlier were that noise.
    `cado_murphy -K/-Keval` and `score_polys.py --points/--eval-points` now score at
    more points.
  - Projective alpha changes under linear rotation, by up to 0.81 nats on 306 of the 1000
    c168 seeds. The tables now include projective roots to p^(e+3), tabulated once per
    seed; `-noproj` turns them off. The alpha column is now `alpha_s`. On 100 c168 seeds
    2 seeds' best improved (+3.1%, +1.4%) and none got worse.
  - Each sieve block used to keep the best cell of each of its 256 threads, then the
    best K of those. It now keeps its exact best K: each warp holds its list one cell per
    lane. On 100 c168 seeds no seed's best cell changed.
  - Content-seed generation is a tool, `stage23_bench/tools/content_seeds.py`.
  - The sieve runs at the same 3.5e10 cells/s with both changes (idle GPU, c168 window).
  - The CUDA context is now created in the background during the size model. On this
    WSL2 box it took 10–13 s on 2026-10-04 afternoon, and the sieve timings had
    included it.
  - A code review of these changes found no math errors. Its small fixes are in, retested
    with M1 acceptance (GPU effort 0 and 50, CPU effort 0) all identical:
    - sopt slices and stale candidate slots;
    - VRAM-aware slot sizing;
    - the poly reader;
    - the trim off-by-one;
    - the `-out` count.

    Open items are listed in the plan ("Code review of the fixes").

## Status (2026-10-03, M2 first steps: root sieve and size model)

Timings were taken while an ECM job used the GPU: retime on an idle GPU.

- **Root sieve** (`rsieve.h`, `s23_ropt`): for a fixed u, one pass over x mod p^e gives
  the root count of f + (u x + v) g for every v mod p^e (with the special rule where p
  divides g(x)), folded per prime into one table of period p^e_max <= 200. A cell's
  score is 46 table lookups. Checks: tables vs brute-force root counting on 324,054 cells
  of the six benchmark polys (0 mismatches; `make test`), three cells vs exact Python
  big-integer counting (alpha_proto's evaluator), GPU vs CPU build bit-identical, and
  `-check N` rescores N random cells of the sieve's own output (GPU or CPU scores) plus
  every printed cell by brute force (0 mismatches so far).
- **GPU speed:** 1.2e10 cells/s on large runs with ECM sharing the GPU (one block per
  2^20-cell segment, tables in shared memory, 48 registers, no spills). Small runs are
  launch-bound.
- **Size model** (`rsize.h`): lognorm after the best integer re-translation. It recovers
  the known optima's translations (c204 B: t within 3 of CADO's; C likewise). With
  re-translation the size landscape is shallow in u: on the c204 seed the line minimum
  at u = 90 is lower than at u = 0, and the deepest basin found is at u = -690
  (61.99 vs 62.90 at u = 0), a region neither CADO nor msieve searched.
- **First band searches** (`-search`: every cell whose re-translated lognorm is
  within B of the best line minimum; ranked by lognorm + affine alpha; best 150-200
  scored by CADO's MurphyE at best skew, `stage23_bench/tools/score_polys.py`). They
  cover every cell of the size model's bands, which is not a proof that every
  competitive basin is covered:
  - c161, B = 1.0 (4 lines, 1.8e7 cells, 1 s): both known optima found (CADO's #2,
    msieve's cell #3); one new cell, (0, -941220), is 0.04% above CADO's winner, a tie.
  - c146, B = 2.0 (416 lines, 1.5e11 cells, 16 s + 9 s size model): msieve's cell #1,
    CADO-orig's u = 120 optimum #2; nothing better in the band. B = 1.0 missed msieve's
    winner, whose lognorm is 1.76 above the best line minimum but whose alpha is
    exceptional: the budget must be about 2 nats.
  - c204, B = 1.5 (869 lines, 8.7e12 cells, 505 s): CADO's B is #1 by the proxy, at
    CADO's exact translation; C and A found too; nothing better in the band.
  - Under job parameters (c200: lpb 32/33, I 15, qmin 5e7; c160: 31/32, 14, 1e7; c145:
    29/30, 14, 1e7) and with translation tuned for MurphyE, no seed has a cell more than
    0.4% above the best known poly: the exhaustive search matches CADO + msieve. CADO's
    default bounds overrate low-skew cells on c204 (+11% that disappears with c200
    parameters), so scoring must use the job's parameters.
  - Lessons: the proxy (L2 lognorm + affine alpha to 200) ranks the true top within the
    best ~20 but not in order (the c161 winner was 17th by proxy), so survivors need
    MurphyE on the GPU. Two lessons from this run were wrong, as shown 2026-10-04:
    - "The translation should be optimized for MurphyE" (msieve's translation of its own
      cell scored 0.25-0.4% higher than ours).
    - "MurphyE has several peaks in skew" (skewopt's 1.6246e-12 for the c161 winner
      looked like a lower peak than the grid's 1.6295e-12).

    Both were sampling noise of MurphyE's 1,000 points. At 64,000 points the skew curve
    has one peak, skewopt's value is its maximum, and translation is worth at most 0.1%.

## Status (2026-10-03, M1 speed step 1: per-candidate phases)

GPU numbers below were taken **while an ECM job was using the GPU (94%)**: retime them
on an idle GPU.

- Correctness, phased GPU build: c146 effort 0 **13,000/13,000** identical, effort 50
  (top 3000) **3,000/3,000**; c161 effort 0 **11,000/11,000**, effort 50 (top 1000)
  **1,000/1,000**; 0 redone by CADO anywhere. The phased CPU build is byte-identical to
  the sequential one. Dedupe now compares the whole final k (CADO's `mpz_cmp`); the
  sequential code compared its low 64 bits.
- Work: c146 has 9.7 candidates per poly at effort 0 and 806 at effort 50; dedupe drops
  about 1% (k after best_norm2 rarely repeats).
- GPU speed (12,288 workers = 8 warps x 48 SMs, 2.7 GB scratch):
  - effort 0: c146 13,000 polys in 31 s optimizing (2.4 ms per poly; before: 219 s,
    17 ms); lll 17.6 s, descent 12.8 s. 8 CPU threads: 42 s (sequential), 54 s (phased).
  - effort 50: c146 top 3000 in 371 s (124 ms per poly; lll 165 s, descent 202 s), c161
    top 1000 in 154 s. CADO on 8 threads would take about 250 s for the 3000 (0.67 s of
    CPU per poly), so effort 50 on the GPU is not yet faster than the CPU.
- Where the time goes: lll is multi-limb integer work on `Int<128>`, where every add,
  sub, negate, compare and copy costs all 128 limbs however small the value (mul's inner
  loop is bounded, its set-up is not). Descent is mostly FP64 root finding (the
  L2 skewness of each step), and FP64 is 1/64 of FP32 on this GPU, so exactness with CADO
  caps that phase near CPU speed. Next steps: length-aware `Int` (a used-limb count), then
  measure again; descent needs an exact-decision shortcut (FP32 with a rigorous margin,
  FP64 only when a comparison is close) to go much faster. Every decision inside the root
  finding and the descent steps needs that margin, not only the final comparison, or the
  path stops being CADO's.

## Status (2026-10-03, M1 step 2)

- `Int<L>`: all operations match GMP (about 660,000 random cases, L = 2 .. 128).
- Exact LLL / `best_norm`: identical to CADO on all 24,343 cases (every raw poly of the
  c146 acceptance sample at k = 0 and at CADO's final translation, plus the c204
  winners): the whole reduced basis, the chosen row and the polynomial.
- Widest intermediate: 1,826 bits on c146 cases, 2,433 bits on c204, so `Int<80>` to
  `Int<96>` would do for c204; the tests use `Int<128>`.
- Device harness: compiles for sm_120 (255 registers, ~18 KB stack per thread with the
  multi-limb routines as real calls; it raises `cudaLimitStackSize` to 24 KB). Run on the
  GPU since: all 24,343 cases identical to the host build (`make run-device`).

Correctness first: no carry intrinsics, one thread per LLL, one width for everything.
Speed comes after M1 matches (smaller widths per variable, warp-cooperative arithmetic,
or floating-point LLL with exactness checks, measured against this reference).

Design notes:

- Rounding, matched to CADO: CADO's C code (norms, alpha, size optimization) is compiled
  without fused multiply-adds, but its C++ root finder (`utils/double_poly.cpp`) is
  compiled with them (GCC, `-O3 -march=native`: every Horner step of `double_poly_eval`
  and `a*pb` in false position are FMAs). Both our builds disable automatic contraction
  (`-ffp-contract=off`, `--fmad=false`) and `include/dpoly.h` writes exactly those
  operations as `fma()`. The device's `log` is CUDA's (within 1 ulp of glibc), the one
  known remaining source of last-bit differences on the GPU.

- The skew for the LLL is computed on the host exactly as CADO's `sopt_get_skewness`
  (double `pow`, then truncation), so the GPU never depends on its own `pow` agreeing
  with glibc's in the last bit. Translation candidates (also doubles in CADO) will get
  the same treatment.
- NTL's linearly-dependent-vector branch is not ported; the sopt lattice is always
  independent, and the port returns `LLL_DEPENDENT` (CPU fallback) if that ever fails.
