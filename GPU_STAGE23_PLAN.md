# GPU size-opt + root-opt (stages 2 and 3): plan

> Started 2026-10-02. Companion docs:
> - `POLYSELECT_RANKING_IMPROVEMENT_PLAN.md`: ranking and objective work (Murphy-E blind spots, E′, lattice geometry).
> - `POLYSELECT_OPTIMIZATION_NOTES.md`: stage-1 GPU kernels.
>
> Numbers marked *(est.)* are back-of-envelope and have not been measured.

## Goal

Run size optimization (sopt) and root optimization (ropt) on the GPU with our own code,
not ports of CADO or msieve.

1. **One device.** The whole post-stage-1 pipeline runs on the 5070 and the CPU is free.
2. **Better polys.** Search the rotation space of the top seeds much more thoroughly than
   either CPU tool can afford, and control the objective used *inside* the search.
3. **Independent code.** Keep it separate from CADO and msieve so the result can be compared
   against both. The current msieve + CADO hybrid stays as the reference.

Speed alone is not the motivation. The CPU pipeline takes about 2 h, while stage 1 takes
about 24 h on a rented 5090.

## Baseline: c204 degree-5 run, 2026-10-02

Stage 1 took about 24 h on a rented RTX 5090 (2–3× that on the 5070). The post-processing
below ran on the local desktop with 8 threads, on 1,944,118 deduped raw polys.

| Stage | Wall | Per poly, per core | Share |
|---|---:|---:|---:|
| CADO sopt, effort 0 | 2779 s | 11.4 ms | 38% |
| Re-sopt top 1000, effort 50 | 150 s | 1.2 s | 2% |
| msieve `-npr`, 300 seeds × (orig + inv) | 1701 s | 22.7 s | 23% |
| CADO ropt effort 5, 150 seeds × (orig + inv) | 2652 s | 70.7 s | 36% |

- Preprocessing runs CADO sopt at effort **0**. `dedupe_and_sopt.sh:222` never passes
  `-sopteffort`, and `sopt_effort` in `nfs_config.ini` is not wired up.
- Effort 50 costs about 105× effort 0. Running it on all 1.94M polys would take about 81 h
  on 8 cores. Only a GPU can afford that.
- Final results (MurphyE, CADO default parameters):
  - msieve: 5.019e-15
  - CADO original: 5.160e-15 (5.194e-15 after skewopt)
  - CADO inverted: 5.154e-15 (5.172e-15 after skewopt)
- The winning seed was rank 5 by `exp_E` (54.40, against 53.30 for rank 1).
  - Only about 10 seeds were within 1.5 nats of the best `exp_E`, while the top 300 spanned
    2.7 nats.
  - From this run alone, the funnel looked wide enough and depth per seed looked like the
    priority. **The c146 run below contradicts this:** there, ropt outcome barely tracks
    `exp_E` rank, so funnel width matters too.

**The value bar.** Plot `exp_E` against ln(rank) for ranks 10–300; the slope is about
0.35 nats (Gumbel tail scale).
- So each doubling of stage 1 improves the expected best by about 0.35·ln 2 ≈ 0.24 nats.
- At about 0.13 log-yield per nat, that is roughly **3%** for another 24 h of 5090.
- The same seed already gave 3.5% different results depending on which tool searched it
  (msieve 5.019 vs CADO 5.194 after skewopt).
- So a better search of the top seeds is worth about as much as a second rental day.
  *(est.: a few order statistics from one run, plus the plan doc's back-of-envelope
  sensitivity)*

## What the current tools actually do (verified in source)

CADO is at `~/cado-nfs` (master, 0574bc39d). msieve is this tree.

### `exp_E` and CADO sopt

- `exp_E = lognorm + expected_rotation_gain(f, g)` (`polyselect/auxiliary.c:211`, `:346`).
  - The gain is projective alpha + `expected_alpha(ln S)` + 0.1 for each rotation degree
    with at least 2 allowed values.
  - S is the product, over rotations x^i·g (i = 0..2 for degree 5), of the coefficient
    range that keeps lognorm within +0.2 (`NORM_MARGIN`) at **fixed skew and translation**.
  - `expected_alpha(K) = −0.824·(sqrt(2K) − (ln K + 1.3766)/(2·sqrt(2K)))`
    (`polyselect_alpha.c:505`).
- **sopt picks its winning translation by lognorm + expected_rotation_gain**, not by
  lognorm (`size_optimization.c:973`, `:1091`). So the preference for higher skew is
  built into sopt itself, not only into the sort.
  - At fixed lognorm, the constant-rotation range grows like s^2.5.
  - That is roughly 0.8–1.1 nats of `exp_E` bonus per decade of skew. *(est.)*
- **How sopt finds translations.** It builds candidate translations k from the roots of
  Res_k(c2, c3) in q2, for f(x+k) + q2·x²·g(x+k).
  - Each q2 root is replaced by its 16 best rational approximations (denominator ≤ 100).
  - Each approximation gives k as a root of c3(k, q2) = 0.
  - Each k then gets an LLL over rotations up to x^(d−2)·g, followed by a local descent
    of up to 300 steps.
  - q2 is used only to choose k. Its rotation is never applied directly.
- **The LLL lattice contains f itself, so sopt returns a·f + r(x)·g with any integer a.**
  - Rows are f(x+k), g, x·g, x²·g, x³·g, scaled by skew powers
    (`LLL_set_matrix_from_polys`). `best_norm` (`size_optimization.c:822`) keeps the
    shortest reduced row with a nonzero leading coefficient and never checks a = ±1.
  - So c5 becomes a·c5_raw and Res(f, g) = a·N. CADO only requires N | Res
    (`cado_poly.c:175`), and the common root mod N is preserved.
  - Equivalently: rotation by rational polynomials with denominator a. For p | a,
    f' ≡ r·g (mod p), which adds a projective root and g's root mod p, so a's small
    factors also feed alpha.
  - **This is the normal case, not an edge case** (c146 run, 371,526 polys): |a| > 1 for
    85% of all outputs, 92% of the top 150, and 94% of the top 1000. Common values among
    the top: 3, 7, 6, 5, 9, 4, 2. The best |a| = 1 poly is 37.15 vs 36.88 overall.
  - ropt usually keeps a: all three c146 winners have c5 = ±5040 = −2·2520 (raw),
    Res = −2N. But CADO ropt divides out content, so a rotation that makes every
    coefficient divisible by a prime factor of a shrinks the multiplier: on c146, |c5|
    changed on 7 of 150 seeds in each CADO pass (e.g. c5 −34020 → −11340, a −9 → −3).
    A GPU ropt should check the content of each candidate and divide it out the same
    way, and its validity checks must allow a to shrink by such a factor.
- **What sopteffort adds.** Each unit of effort adds 16 more q2 values: reduced fractions
  a/b in **[−1, 1]**, in order of increasing denominator (`size_optimization.c:502–521`).
- **Kleinjung's integer pre-rotation, f + δ·x²·g with |δ| ≤ 256, adds nothing** (M0:
  best over all 513 δ matches CADO's effort-50 result on 296 of 300 seeds).
  - CADO already solves for the real-valued q2 analytically.
  - The lower-degree rotations that pre-rotation introduces are already in the LLL lattice.

### CADO ropt

- Degree 5 uses linear rotation only (`ropt.c:177`). Quadratic rotation is used only for
  degrees 6 and 7.
- **Sieve:** primes below 200 (`ROPT_NPRIMES 46`), with prime powers capped at 200
  (`ropt_stage2.c:668`). Scores are int16, scaled by 1000.
- **Pruning:** only the top 32 cells per sublattice (`NUM_TOPALPHA_SIEVEARRAY`) get
  translation re-optimization and MurphyE. The top 32 MurphyE per sublattice are kept
  (`NUM_TOPE_SUBLATTICE`).
- **Funnel:** a tuning stage, then stage 1 (sublattices via Hensel lifting and CRT), then
  stage 2 (root sieve), then the final pqueue. Effort scales the sublattice counts linearly
  (`ropt_param.c`).
- **MurphyE parameters default to Bf=1e7, Bg=5e6, area=1e16** (`area.h`), which is what the
  pipeline uses. With `-I`, area becomes Bf·2^(2I−1) (`ropt_main.c:300`). CADO's own c200
  parameters are lim0=1.3e8, lim1=1e8, I=15.

### msieve

- **In this fork, `-nps` does no size-opt.** `poly_sizeopt_run` writes the expanded raw
  poly and returns at `gnfs/poly/stage2/stage2.c:247`.
  - The only filter on what reaches `.ms` is `pol_expand`'s |c_{d−2}| ≤ coeff_bound check.
  - That function is the natural place to plug in an inline GPU sopt.
- **msieve ropt:**
  - `root_sieve_run` sweeps `rootopt_stage2_*` norm bounds, which change the lattice
    spacing (see the deep-ropt notes).
  - It sieves in x/y for degree 5 (`root_sieve_deg5_xy.c`).
  - It finishes with `optimize_final`, which uses Powell's method (`common/minimize.c:264`).
- **The inverted pass (f → −f) stays.**
  - It usually matches the original pass, but not always.
  - The likely reason: Powell line-searches along positive unit directions, and negating f
    flips the signs of the rotation coordinates, so the search can reach different optima.
  - The GPU version should keep this kind of start variety on purpose.

## Benchmark: one seed, three known optima

The three winners from the 2026-10-02 run share Y1, so they are the same seed (same
rotation and translation family). I recovered the exact transformations between them
(exact integer division, zero remainder). The deleted seed is not needed: any member of the
family is a valid origin, and the search space is identical.

```
n: 439720410267491320573316170888951374332497845760232566953110544200627951472089897471620444241890147987198378179131262589288584983229132379313942580162752137149693081334177098806011632087906654257033579

# A: msieve -npr winner, e 5.019e-15, alpha -8.56, skew 230988105.72
Y1: 552226208798178007565897
Y0: -656580574523506307168091152512134301360
c5: 7207200
c4: -2009879558140500
c3: -1170924462759770419238197
c2: -320830019316735857875472855599094
c1: 5867295737650435671309091195521271756137
c0: 756148725619987276579013549995276036729050526134

# B: CADO ropt (orig) winner, MurphyE 5.160e-15 (skewopt 5.194e-15 at s=278605872)
#    lognorm 63.21, alpha -8.90 (proj -3.12), skew 205696414.754
Y1: 552226208798178007565897
Y0: -656580607096863569697910357664601469461
c5: 7207200
c4: -4135482225328500
c3: -445949582208271311284197
c2: -170376108911627472710638855147361
c1: -25512103124933873178031205261993266280708
c0: 2347343878381630657926552692744033467962846389216

# C: CADO ropt (inverted) winner, MurphyE 5.154e-15 (skewopt 5.172e-15 at s=284918331)
#    lognorm 63.20, alpha -8.87, skew 223338680.059
#    Printed in the inverted convention (c0..c5 negated); negate to compare with A and B.
Y1: 552226208798178007565897
Y0: -656580585488393389334657255180988729093
c5: -7207200
c4: 2725402770544500
c3: 982878929212174404308197
c2: 256399724056172669311613998652795
c1: -17291576216685926082949204651290941366758
c0: -3865979372010771630711957374746814426055516189560
```

Each result is written as `f_X(x) = f_A(x+t) + (u·x + v)·g_X(x)`, where `g_X(x) = g_A(x+t)`:

| Result | t | (u, v) in X's frame | (u, v) in A's frame | MurphyE |
|---|---:|---|---|---:|
| A (msieve) | 0 | (0, 0) | (0, 0) | 5.019e-15 |
| B (CADO orig) | −58,985,533 | (90, −4,329,571,680) | (90, 979,126,290) | 5.160e-15 |
| C (CADO inv, un-negated) | −19,855,789 | (0, −5,093,033,400) | (0, −5,093,033,400) | 5.154e-15 |

To convert to A's frame: v_A = v − u·t.

These are three separate local optima within 3% of each other. Neither tool covers this
seed's rotation space fully. All three have Res = 2N: CADO sopt gave this seed multiplier 2
(see "The LLL lattice contains f itself"). The polys are in `stage23_bench/data/c204/`.

**Targets for the GPU ropt on this seed:**
- First, find B or C (or better) on its own.
- Then beat **5.194e-15**, scored the same way: skewopt/cownoise, which matches msieve's
  `e` scale.
- If an exhaustive search can't beat B, that is a cheap and important negative result.

### Second seed: c146, 2026-10-02

n = 3662801383…034597099 (146 digits, in `pipeline_results/`). Y1 = 18436463496569028067,
raw c5 = 2520, sopt multiplier a = −2 (c5 = −5040, Res = −2N). The seed was `exp_E` rank 84
(38.04 vs 36.88 best).

| Result | t vs msieve | (u, v) | skewopt MurphyE |
|---|---:|---|---:|
| msieve `-npr` (orig and inv identical) | 0 | (0, 0) | **1.2391e-11** |
| CADO inv | 617,246 | (0, 0): same poly, translated | 1.2379e-11 |
| CADO orig | −1,232,902 | (120, −219,882,480) in its own frame; (120, −71,934,240) in msieve's | 1.2133e-11 |

Again a distinct optimum at large u (120 here, 90 on the c204 seed). Target for this seed:
beat 1.2391e-11. The raw poly, its sopt result and the three winners are in
`stage23_bench/data/c146/winners/`; the per-seed ropt baseline for ranks 1–1000 is
`stage23_bench/data/c146/ropt_seeds.tsv.gz`.

### Third seed: c161, 2026-10-03

n = 7326172377…774589164086067 (161 digits). Y1 = 419093078612002505953, raw c5 = 22680,
sopt multiplier a = −8 (c5 = −181440, Res = −8N). The seed was `exp_E` rank 16 (42.12
vs 41.34 best). All four ropt runs (msieve and CADO, orig and inv) picked this seed.

| Result | t vs CADO orig | (u, v) | skewopt MurphyE |
|---|---:|---|---:|
| CADO orig (inv identical, negated) | 0 | (0, 0) | **1.6246e-12** |
| msieve `-npr` orig | −658,490 | (0, −1,219,680) | 1.6101e-12 |
| msieve `-npr` inv | | | 1.532e-12 (msieve's `e`) |

Here the two tools' optima are both at u = 0 relative to the sopt poly; they differ by
translation and v only. The next seed (exp_E rank 2) is 8% behind under both tools.
Target for this seed: beat 1.6246e-12. Files: `stage23_bench/data/c161/winners/` and
`ropt_seeds.tsv.gz`.

Appendix A has the recovery math. The original fixture in
`POLYSELECT_RANKING_IMPROVEMENT_PLAN.md` (a different c204, E 4.237e-15) is a second test
case. Its alternative optima are unknown.

## Design constraints

1. **FP64 is 1/64 of FP32 speed on the 5070 (GeForce sm_120).**
   - CADO and msieve do all numeric work in double.
   - Keep the polynomial in exact integers and do scores in FP32 in the log domain.
   - Range, not just precision: c0 (2.3e48 on the c204 seed) and c1 (~1e40) already
     exceed FP32's 3.4e38 before any squaring. Convert with a separate exponent, or scale
     by skew powers (a_i·s^(i−d/2)) in integer arithmetic before converting.
   - CADO's translation-candidate resultant (`size_optimization.c:466-476`) and its L2
     skew root-finding run in double with cancellation. How sopt splits work between
     exact integers, FP64 and FP32 is decided under Open questions.
2. **The polynomial must be exact; only the score can be approximate.**
   - Translation and rotation work by cancellation.
   - Degree 5 at c204: c0 is about 153–161 bits, and translation intermediates reach about
     200 bits.
   - Fixed-width integers with an overflow flag that falls back to the CPU path. As
     built (`stage23_gpu/include/mpint.h`): 512-bit polynomial coefficients, 4096-bit
     for the exact LLL and the discriminant, whose intermediates measured up to 2,433
     bits on c204 (so ~2,600-3,072 bits would do). Degree 6 at c250+ will need more.
   - The existing `stage1_core_gpu/cuda_intrinsics.h` only covers 64-bit modular
     arithmetic, so the multiprecision layer is new.
3. **No general LLL, but the multiplier a must be searched.**
   - The rotation rows x^i·g(x+k), scaled by skew powers, each have only two nonzeros, so
     the Gram matrix is tridiagonal.
   - With s much smaller than m = |Y0/Y1|, the basis is nearly orthogonal:
     - this seed: 2.1e8 vs 1.2e15;
     - the plan-doc fixture: 1.15e8 vs 3.9e15.
   - Babai nearest-plane then reduces to a sequential back-substitution, with fixed cost
     and no branches.
   - **Babai with target f only covers a = 1.** CADO's LLL also searches the multiplier
     (see "The LLL lattice contains f itself" above), and over 90% of top polys use
     |a| > 1. Restricting to a = 1 would lose roughly 0.3 nats at the very top and more
     in the bulk.
   - So run Babai on target a·f for each a in 1..A (sign is symmetric) and keep the best few
     per translation candidate. That is A independent fixed-cost solves, which suits the
     GPU better than LLL.
   - CADO's |a| on c146: top 150 median 15, 95% ≤ 132, max 315; top 1000 median 32,
     95% ≤ 466, max 12,474; all polys median 753, max 98,295. A ≈ 500 covers the top 150
     but not M1's acceptance set: 10% of the top 3000 and 59% of the random sample in
     `sopt_sample.tsv.gz` have |a| > 500.
     Picking a is simultaneous Diophantine approximation (a·f's lower coefficients close
     to the rotation lattice), which is exactly what LLL solves. **Decided: a fixed
     5-row LLL** (Open questions).
   - LLL picks a by L2 norm, but that is already almost exactly the exp_E-best a: a forced
     sweep of a = 1..500 on 300 seeds improved exp_E by ≥ 0.05 nats on only 8 seeds (M0).
     So the GPU needs a to match CADO; searching a harder for exp_E gains almost nothing.
   - Runner-up multipliers (typically 0.3 nats behind in exp_E) don't make useful extra
     ropt starts: after msieve ropt they are a median 6% worse than CADO's choice (M0).
   - Measure how often the result disagrees with CADO's LLL on a sample.
4. **Avoid warp divergence.**
   - CADO's descent has data-dependent step counts, `best_norm2` recurses, and the number
     of translation candidates varies.
   - Flatten this into (poly, candidate) work lists, run a descent kernel with a fixed
     iteration cap and early-exit masks, then reduce per poly.
   - Same pattern as the stage-1 collision engine.
   - **The reduction must keep CADO's sequential semantics, or ties stop matching.**
     CADO walks the sorted, deduplicated translation list in order; after best_norm2 a
     candidate whose resulting k was already produced by an earlier candidate is skipped
     (first one wins); and `lognorm < best_lognorm` keeps the first of equal exp_E. So
     dedupe the post-best_norm2 k in list order before the descent, and reduce per poly
     by (exp_E, candidate index) with strict-less semantics.
5. **Alpha is invariant under translation, so search in (u, v) with re-translation.**
   - At fixed translation, a +0.2 lognorm budget allows only u ∈ {−1, 0, 1} for this seed.
     Yet B differs from A by u = 90.
   - The reason: both tools re-optimize translation for each candidate, and translation by
     about s/4 cancels the u·g0 change to c1.
   - Design:
     - Sieve alpha over (u, v) in one fixed frame. Alpha does not depend on the
       translation.
     - Separately, tabulate the size after re-translating and re-optimizing skew,
       size(u, v). It varies smoothly with v along each u line.
     - Sieve only where size(u, v) ≤ L0 + budget, which is a curved band, not a box.
   - Region size for this seed: u at least ±100 and v up to about ±1e10, which is
     1e12 cells or more per seed. *(est.)*
6. **Sieve cost grows with the number of prime powers up to B.**
   - Each cell gets about one hit per prime power on average: about 60 at B = 200 and
     about 334 at B = 2000.
   - So sieve with p < 200, plus small-prime patterns for 2, 3, 4, 5, 7, 8, 9, ….
   - Compute exact alpha (to B = 2000, or to lim) only for survivors: root counting mod p,
     plus the exact p-adic recursion for primes with a multiple root (constraint 7). That
     is about 1.3M integer ops per candidate, cheap for 1e5–1e6 survivors.
7. **Root counting at g's root, and multiple roots.** CPU reference:
   `stage23_bench/tools/alpha_proto.py`.
   - Sieve formulation: for a fixed u, one pass over x mod p^e gives N_e(v) for every v.
     An x with p ∤ g(x) is a root for exactly one v class mod p^e.
   - **At g's root mod p that is false.** With k = v_p(g(x)) and h = f + u·x·g: if
     e ≤ k, x is a root for every v when p^e | h(x) and for none otherwise; if e > k, it
     is a root for one v class mod p^(e−k) when p^k | h(x). Whenever p | a, f(x_g) ≡ 0
     mod p for every cell; both benchmark seeds have a = ±2.
   - Checked against brute force on 67,370 cells (p^e ≤ 256, both seeds, several u): the
     special rule matches everywhere; the naive one-class rule is wrong on 1,280 cells
     and misstates alpha by 0.23–0.46 depending on the cell, so it would mis-rank.
   - Truncating at small p^e is fine inside the sieve, but not for the exact alpha on
     survivors. A double root at p = 577 whose lifts all survived mod p² moved one poly's
     alpha by 0.011, and point counting to p^e ≤ 4096 still differed from the exact value
     by up to 5e-3 in E_p. With CADO's p-adic recursion (`special_val0`) for those primes,
     the prototype matches CADO's printed alpha on 400 of 400 sopt outputs and all six
     benchmark winners.
8. **The tail-aware score is not additive within a prime.**
   - E[p^(kν_p)] is affine in the per-level root counts N_e(u, v), but the score is its log.
   - So the sieve needs a fold per prime: sieve p into a scratch tile, then
     `score += (1/k)·ln(1 + Σ c_e·N_e)`, then clear the tile.
   - In shared memory this is cheap. On a CPU it is an extra full pass per prime.
9. **The objective sits inside the pruning.**
   - Both CPU ropts discard candidates by MurphyE inside the search.
   - v·g rotation grows c0, which raises the optimal skew. Reranking afterwards cannot
     recover moderate-skew alternatives that were already discarded.
   - Make the objective pluggable, and output a Pareto set (alpha, lognorm, skew) per
     seed, not a single score.
   - Score with the job's sieving parameters (Bf, Bg, I or area), not CADO's defaults.
     Low priority for the final ranking: on c146 it barely changed it (M0), but it still
     matters for what gets pruned inside the search.
10. **Start variety.** Keep several starts and both signs on purpose (see the msieve
    inverted pass above).
11. **Runtime environment.**
    - WSL2 TDR is raised to 60 s. Even so, keep kernels short by chunking sublattices and
      u lines.
    - Never run during LA (VRAM).
    - Contention with stage 1 is a non-issue while stage 1 runs on rented hardware.

## Milestones

### M0: CPU checks before building anything

Run on the c146 job, 2026-10-02 (371,526 deduped polys, CADO c145 parameters). The tools
and fixtures to rerun these are in `stage23_bench/` (see its README); the join key is the
raw poly (Y1, Y0, c5).

- [x] **Does higher sopt effort move polys into the top?** Barely. Ranks 1001–3000 at
      effort 50 (168 s on 8 processes, 0.67 s per poly): mean gain 0.013 nats, max 0.73.
      None enter the top 150; 2 enter the top 300 (from ranks 1302 and 1391).
- [x] **How much does effort 0 → 50 change `exp_E`?** Little. On the top 1000, 745 are
      identical; mean gain 0.015 nats, max 0.60, never worse at CADO's printed precision
      (at full precision, 2 of the top 3000 are slightly worse, by ≤ 0.0015); 75 change
      their multiplier a.
      The effort-50 top 150 has 147 from effort-0 ranks ≤ 250; the deepest is rank 607.
- [x] **Do job parameters reorder the finalists?** No. 43 distinct finalists, each with
      skew optimized separately under CADO defaults and under c145 job parameters
      (Bf = Bg = 2^30, area = 2^27·qmin = 2.7e14): Spearman 0.99. The only top-5 change
      is two copies of the same poly at different translations, 0.01% apart.
- [x] **Pre-rotation test.** Nothing to gain. For each of the top 300 seeds, fed
      f + δ·x²·g with δ = −256..256 (Kleinjung's range) into sopt at effort 0: 153,900 runs,
      487 s on 8 processes. Best over δ vs CADO's effort-50 result: mean gain −0.001 nats,
      4 of 300 seeds gain > 0.05, max 0.29. Most δ give identical results. A δ sweep at
      effort 0 is worth about the same as effort 50 (mean 0.024 over δ = 0).
- [x] **Does searching the multiplier a beat CADO's LLL choice?** Not for exp_E. For each
      of the effort-50 top 300 seeds, fed a·f_raw + g (forces a multiple of a) into sopt at
      effort 0 for a = 1..500: 150,000 runs, 592 s on 6 processes. Best over all a vs
      CADO's effort-50 result: 8 of 300 seeds gain ≥ 0.05 nats, 1 gains > 0.3 (max 0.34).
      a = 1 reproduces CADO exactly. The best alternative |a| per seed is a median 0.32
      nats behind (25–75%: 0.22–0.43).
- [x] **Do runner-up multipliers ropt better?** No. msieve ropt (orig pass) on the best
      alternative-|a| poly of each of the 300 seeds (310 s on 6 threads), paired per seed
      against the pipeline's orig-pass results: median ratio 0.942, wins on 31 of 125
      seeds, best 1.074e-11 vs 1.239e-11. The runner-up is usually 2× CADO's multiplier.
- [x] **Is the ropt funnel cutting off contenders?** Yes, but not the winner on this job.
      msieve ropt on effort-50 ranks 301–1000, both passes (1586 s on 8 threads, with other
      jobs competing):
      - The second-best seed overall is rank 407 (1.225e-11, 1.1% behind the winner).
        4 of the top 20 seeds and 10 of the top 50 are from ranks > 300.
      - Mean best score per rank band declines slowly: 9.61 (1–150), 9.26 (151–300),
        9.04 (301–500), 8.84 (501–750), 8.85 (751–1000), all e-12. The best per band is
        noisy: 1.239, 1.183, 1.225, 1.059, 1.110.
      - CADO ropt (effort 5, both passes) on the 10 best seeds from ranks 183–849, which it
        never saw in the pipeline: best 1.220e-11 (rank 407). No new winner.
      - Orig and inverted passes differ a lot per seed for CADO too: rank 407 gave 1.045 vs
        1.220, and rank 241 gave 1.178 vs 1.042 (e-11, orig vs inv). msieve's passes
        differed by > 0.1% on 28 of about 288 seeds in ranks 301–1000.

- [x] **CPU alpha prototype before any CUDA.** Done; results in constraint 7.
- [x] **Does MurphyE track real sieving at the 1–3% level?** Partly. Test-sieved seven
      c146 polys with `las` and the c145 parameters (lim 15M/23M, lpb 30, mfb 58, I = 14)
      over q ∈ [3M, 3.004M], [8M, 8.004M], [16M, 16.004M], about 700 special-q per poly;
      the standard error of relations per special-q is about 0.3% per poly (polys, exact
      command and results in `stage23_bench/data/c146/test_sieve/`). CADO's c145
      parameters stand in for a real job file here; real validation should test-sieve
      on the GPU with the job's parameters (`~/code/cuda-sieve/bench/testsieve.sh`,
      the user's preferred tool: GPU-only like the rest of this work, and much faster):

      | Poly (exp_E rank, tool) | skew | MurphyE vs winner | yield vs winner | speed vs winner |
      |---|---:|---:|---:|---:|
      | 84, msieve (winner) | 4.9e6 | 1.000 | 1.000 | 1.000 |
      | 407, msieve | 3.0e6 | 0.989 | 0.980 | 0.977 |
      | 84, CADO orig | 3.0e6 | 0.979 | 0.964 | 0.976 |
      | 241, msieve | 3.3e6 | 0.955 | 0.959 | 0.960 |
      | 250, msieve | 2.0e6 | 0.893 | 0.973 | 0.983 |
      | 657, msieve | 2.9e6 | 0.808 | 0.824 | 0.795 |
      | 397, msieve | 1.7e6 | 0.726 | 0.778 | 0.752 |

      (yield = relations per special-q; speed = inverse CPU seconds per relation.)
      - Large gaps are right, and the winner really is the fastest. The 1–2% gaps at the
        top have the right sign.
      - Rank 250 is a clear miss: MurphyE puts it 11% behind, it sieves 1.7% slower,
        ahead of ranks 407, 241 and CADO's optimum. It has the lowest skew of the top
        group, which fits the ranking plan's high-skew blind spot.
      - No MurphyE parameter set fixes it (0.90–0.93 predicted). Against measured speed,
        RMS error: CADO defaults 3.8% (top five 4.2%); Bf = Bg = 2^lpb with area
        2^(2I−1)·qmin 4.8% (top five 2.7%); Bf, Bg = lim with the same area 3.5% (3.8%).
        msieve's `e` uses the same bounds as CADO's defaults (rational 5e6, algebraic
        1e7, area 1e16; `gnfs/poly/size_score.c:555`).

Findings that change the design:

- **CADO sopt searches the multiplier a** (`a·f + r·g`, Res = a·N), and over 90% of top
  polys use |a| > 1. See constraint 3. A GPU sopt limited to a = 1 would not even match
  CADO.
- **Deeper sopt doesn't reorder the funnel.** Neither higher CADO effort nor an explicit
  multiplier search moves exp_E meaningfully. On exp_E, CADO sopt at effort 0 is already
  close to the ceiling for these raw polys. GPU sopt's value is the one-device goal and
  feeding the GPU ropt, not better seeds by exp_E.
- **`exp_E` rank only weakly predicts ropt outcome:**
  - Spearman between a seed's exp_E rank and its best msieve ropt score: 0.35 over the
    top 300 and 0.42 over the top 1000, counting seeds msieve returned nothing for as
    worst. msieve drops results below its `min_e`: 67 of the top 300 and 476 of the top
    1000 have none, and they cluster at worse ranks. Over only the seeds with results the
    figures are 0.24 and 0.34, which are biased low (`seed_funnel.py --ran`, with the
    pipeline's own seed order from `--rank-file`).
  - The winner was rank 84. 4 of the top 10 ropt seeds came from ranks 151–300, which
    CADO ropt (top 150) never saw. Best score per band of 75 ranks: 1.207, 1.239, 1.143,
    1.183 (e-11), with no clear decline by rank 300.
  - Extending to ranks 301–1000 found the #2 seed (rank 407) but no new winner. So the
    funnel matters, contrary to the c204 conclusion, though on this job the 300 cutoff
    didn't cost anything. GPU ropt throughput should target 1000+ seeds, not only depth on
    the top 10–30.
- **MurphyE has a 3–4% error floor against test sieving at the top**, the same size as
  the gains being chased. Final selection among the best candidates should be by test
  sieve (`~/code/cuda-sieve/bench/testsieve.sh`, on the GPU), and the
  GPU's job is to hand it a short, diverse list rather than a single MurphyE winner.

### M1: GPU sopt reproducing CADO's objective

- **Components:**
  - fixed-width polynomial arithmetic (translate, rotate; 512-bit as built)
  - L2 lognorm (closed form) and best skew (numeric: roots of a degree-d polynomial in s²,
    comparing the lognorm at each extremum, as `polyselect_norms.c` does)
  - `expected_rotation_gain`
  - degree-5 resultant translation candidates (all closed-form quadratics)
  - multiplier and rotation search: a fixed 5-row LLL per translation candidate
    (constraint 3): exact integer LLL with delta = 1, as CADO's `utils/lll.c` does
  - bounded local descent
- **Interface:** start as a standalone binary that reads the raw `.ms` and writes CADO
  `sopt` format, so `utils/sort_cado_by_expe.py` and the pipeline work unchanged. Inline
  integration (the `poly_sizeopt_run` hook) comes later.
- **Done when:** on the acceptance set (top 3000 + random 10,000), every poly is either
  identical to CADO's result or valid and no worse (within 1e-6) in `exp_E`, with both
  outputs rescored at full precision by CADO's own code (`cado_expe`; CADO prints 2
  decimals), joined on the raw poly (Y1, Y0, c5). `sopt_compare.py --rescore` applies
  this and fails its exit status otherwise. `stage23_bench/data/c146/sopt_sample.tsv.gz`
  is the first target.
- **Progress (2026-10-03):** `stage23_gpu/` holds the host/device code. `Int<L>`
  (32-bit-limb fixed-width integers) matches GMP; the exact-integer LLL and CADO's
  `best_norm` (translate, lattice, LLL, row choice) are identical to CADO on 24,343
  cases from the c146 fixture and the c204 winners, tested as a host build of the same
  source. Widest intermediate 2,433 bits (c204). On the GPU (2026-10-03) all 24,343
  cases are identical to the host build.
- **M1 steps 3-4 (2026-10-03, CPU build):** `stage23_gpu/tools/s23_sopt` ports CADO's
  whole size optimization (translation candidates, best_norm2, local descent,
  expected_rotation_gain with the exact projective alpha, effort > 0) to the host/device
  code. On the CPU build it passes M1 acceptance: effort 0 on all 13,000 polys (12,996
  identical, 4 no worse, 0 worse) and effort 50 on the top 3000 (2,993 identical, 7 no
  worse, 0 worse), with no polynomial needing the CADO fallback.
- **Rounding parity (2026-10-03, after review):** the remaining differences were not
  ties. CADO's C code (norms, alpha, size optimization) is unfused, but its C++ root
  finder (`double_poly.cpp`) is compiled with fused multiply-adds (objdump: every Horner
  step of `double_poly_eval`, and `a*pb` in false position). Both our builds now disable
  automatic contraction and write exactly those fused operations as `fma()`. The CPU
  build is then **13,000/13,000 identical** to CADO at effort 0 (and 3,000/3,000 at
  effort 50, top set), and so is the GPU build at effort 0
  (`s23_sopt`, effort 0, all 13,000 identical, 0 fallbacks). The GPU uses CUDA's `log`
  (within 1 ulp of glibc's), the one known remaining source of last-bit differences; it
  changed nothing on this set.
- **Second number (c161, 2026-10-03):** a fixture built from a new pipeline run, unseen
  during development (`stage23_bench/data/c161/`, top 1000 + random 10,000): the GPU
  build gives **11,000/11,000 identical** to CADO at effort 0, 0 fallbacks. The CPU build
  on the **whole set, 814,307 polys: all identical** to CADO's effort-0 results, 0
  fallbacks; effort 50 on the top 1000: 1,000/1,000 identical.
- **Output cost to fix:** in that full run, 3,110 s of 4,681 s were optimization; most of
  the rest is printing CADO's stats line (lognorm, exp_E, alpha to 2000, real roots) for
  every output poly on one host thread via CADO's code. Now (review fix) the stats and
  formatting run on all host threads (c146 13,000 at effort 0: 59 s -> 45 s, same
  bytes). For a one-device pipeline they still need computing on the GPU (alpha to 2000,
  the real-root count and the combined skew are new; see below), or overlapping with the
  next launch.
- **M1 on the GPU (2026-10-03):** `make accept-gpu`, effort 0: 13,000/13,000 identical to
  CADO on c146 and 11,000/11,000 on c161, 0 fallbacks, so M1's correctness bar is met on
  the GPU. (Before the `fma()` fix it was 12,999 identical + 1 no worse.) Speed is not yet:
  19 ms per poly (11x slower than 8 CPU threads), with one thread per poly taking up to
  ~20 s, so effort 50 cannot run under the 60 s watchdog. Next: per-candidate parallelism
  (constraint 4), then narrower integers and fewer spills, re-checked with
  `make accept-gpu` after each step.
- **Acceptance tooling (ready, 2026-10-03):** `fixture_raw.py` exports the fixture's raw
  polys; `sopt_compare.py FIXTURE RESULT --rescore` applies the rule, rescoring with
  `cado_expe` (CADO's own `cado_poly_compute_expected_stats` at the skew sopt prints,
  `L2_combined_skewness2(g, f)`; it rounds to CADO's printed exp_E on 1000/1000 polys).
  Self-test: CADO sopt on the exported raws passes 13,000/13,000 as identical.
- **Printed exp_E is not the objective sopt minimizes.** `size_optimization` keeps the
  lowest lognorm at f's own L2 skew + `expected_rotation_gain`; the exp_E it prints (and
  the pipeline ranks by, and `cado_expe` computes) is the lognorm at the combined f/g skew
  + the same gain, which is never lower and not always in the same order. So more effort
  always lowers the objective but can raise the printed exp_E: c161 top 1000, effort 50
  vs 0, one poly (rank 261) has objective 43.0524 vs 43.0686 but printed exp_E 0.0023
  worse. Consequences: (1) M1's "no worse" judges by the printed value, which is what
  the pipeline uses; a result that is better by sopt's own objective can still fail it,
  so once the GPU stops being bit-exact, check such rows against both. (2) Stats on the
  GPU need `L2_combined_skewness2` as well as the port's objective.
- **Throughput target:** all 1.94M polys at effort 0 in minutes, and effort-50 equivalent
  on all of them in under an hour. *(est.)*

### M2: GPU ropt core

- **Search:**
  - alpha sieve (p < 200, prime powers ≤ 200, matching CADO first) over the size-limited
    (u, v) band from constraint 5
  - top-K per u-line block
  - exact alpha to B = 2000 on survivors
  - translation and skew re-optimization
  - FP32 MurphyE with the pruning objective (pending: see Open questions)
- **Done when:**
  - on the benchmark seed, it finds B or C (or better) from A as the origin;
  - it reaches or beats 5.194e-15 under skewopt scoring;
  - it reports how many separate optima above 5.0e-15 exist;
  - on the c161 seed it beats 1.6246e-12;
  - on the c146 seed it beats 1.2391e-11, and its best few candidates test-sieve at least
    as fast as the pipeline's winner (`~/code/cuda-sieve/bench/testsieve.sh`, with the
    job's real parameters), since MurphyE alone can't resolve 1–3% differences.
- **Then:** run the top 10–30 seeds of a real job and compare against the
  msieve + CADO union.

### M3: Objectives

- E′ (David–Zimmermann) and the exact-distribution E′ (ranking plan, workstream 4)
- the tail-aware sieve score (constraint 8)
- lattice-aware E (ranking plan, workstream 1)
- change one objective at a time against the M2 baseline

### M4: Integration

- inline GPU sopt in `-nps`
- a pipeline mode that runs entirely on one device
- keep the CPU hybrid as the reference

## Open questions

**Still pending:**

- **Pruning objective inside ropt:** which MurphyE parameters, whether to keep a Pareto
  set per seed, and the size of the shortlist handed to test sieving. Measured
  2026-10-03 on c146:
  - The parameter set barely changes a shortlist. The best poly of each of 72 seeds,
    skew optimized per set, under CADO defaults, 2^lpb with qmin 10M or 2M (lpbr 29,
    lpba 30, I = 14), and c145 lims: every set keeps 10/10 of the default top 10 and
    19–20/20 of its top 20 (Spearman ≥ 0.995).
  - The q range matters more than MurphyE's parameters. Test-sieving the seven finalists
    again with lpbr 29, lpba 30, I = 14 at q = 10M–30M: the winner leads the next four
    by about 4% in relations per special-q (MurphyE said 1–4.5%), and the low-skew seed
    250 is 5% behind at every q (it was +0.7% at q = 3M in the first run). Low skew pays
    at the bottom of a low-qmin range and loses higher up, as the user has seen. The CPU
    s/rel of this run overlapped another job and was not compared.
  - Proposal: prune with the job's expected parameters when known (CADO defaults
    otherwise; it hardly matters), hand a shortlist of 10–20 to test sieving over the
    job's real q range, and decide there.
- **ropt budget numbers:** cells per seed and seeds per tier, once a GPU sieve kernel's
  throughput is measured. The exhaustive run on a benchmark seed (below) is the first M2
  experiment.

Decided (2026-10-03):

- **Multiplier search (M1): a fixed 5-row LLL**, as CADO does (f, g, x·g, x²·g, x³·g,
  skew-scaled), keeping the shortest row with a nonzero leading coefficient. Babai-only
  over a ≤ A was rejected: |a| > 500 on 10% of the top 3000 and 59% of the random sample.
- **M1 acceptance:** the top 3000 plus the random 10,000 in `sopt_sample.tsv.gz`. A row
  passes if the GPU poly is identical to CADO's, or valid and no worse in exp_E when both
  are rescored by one full-precision implementation (CADO prints 2 decimals).
- **Precision:** sopt keeps polynomials in exact multi-limb integers (32-bit limbs) and
  runs the LLL exactly in integers too: CADO's sopt LLL (`best_norm` →
  `utils/lll.c`, "exact multiprecision arithmetic", delta = 1/1) is not floating point,
  and a floating-point LLL at delta = 1 can pick a different reduced basis on near-ties
  (and has no termination guarantee), which would show up as M1 mismatches. The scalar
  steps (translation candidates, skew root-finding) use native FP64, as CADO does in
  double; their volume is small enough that FP64's 1/64 rate does not matter. The ropt
  sieve scores in FP32 log domain (or int16); exact alpha on survivors uses integers and
  the p-adic recursion. (Corrected 2026-10-03: the first version of this decision had
  FP64 Gram–Schmidt for the LLL.)
- **ropt search shape (provisional):** a small-prime prefilter (classes of (u, v) mod
  2520 or similar) inside each seed, two tiers across seeds as `select_deep_seeds` does
  now, with the deep tier keeping more classes and a wider band. **Gate:** one exhaustive
  (no-prefilter) GPU run on a benchmark seed, compared with the prefiltered run, before
  the prefilter is relied on.
  - CPU check so far (`prefilter_window.py`, 2026-10-03): every cell of size-flat windows
    around the four known optima (±2 u-lines; 460M cells; lognorm within 0.03 of the
    centre) scored by alpha over p < 2000. In each window the known optimum is the best
    cell, and keeping only the top 0.01% of classes loses nothing. This cannot show
    what a prefilter would lose elsewhere: the windows are centred on optima that CPU
    tools found, and both of them already favour good small-prime classes, so these
    optima sit in top classes by construction. Hence the gate.
- **Code location and build:** M1 and M2 are standalone tools; revisit before M4.
  - Stage 1 loads PTX via `cuModuleLoad` (`stage1_sieve_gpu.c:1658`).
  - The CUB engines are `.so` files loaded with `dlopen` (`common/util.c:547`).

Other open questions:

- **Re-translation inside the sieve.** Is a per-u-line size table accurate enough, or does
  the band need a 2-D size surface?
- **How many separate optima does a seed have, and how far apart are they?** This sets
  how far the search must reach, and whether multi-start beats exhaustive.
- **Is int16 score precision enough once B grows and the tail-aware score is added?**

## Appendix A: recovering the rotation between two ropt outputs from one seed

Two outputs from the same seed share Y1. Compute the translation `t = (Y0_X − Y0_A) / Y1`,
which must divide exactly. Then compute `D(x) = f_X(x) − f_A(x+t)` and divide it by
`g_X(x) = Y1·x + Y0_X`. The remainder must be zero, and the quotient is `u·x + v`. For an
inverted-pipeline output, negate c0..c5 first. Exact Python integers suffice:

```python
from math import comb
def translate(c, t):  # coefficients of f(x+t), c[i] = coeff of x^i
    d = len(c) - 1
    return [sum(c[j] * comb(j, i) * t**(j - i) for j in range(i, d + 1)) for i in range(d + 1)]
def div_linear(c, g1, g0):  # exact c(x) / (g1*x + g0) -> (quotient, remainder)
    r = list(c); q = [0] * (len(c) - 1)
    for i in range(len(c) - 1, 0, -1):
        assert r[i] % g1 == 0
        q[i-1] = r[i] // g1; r[i] -= q[i-1] * g1; r[i-1] -= q[i-1] * g0
    return q, r[0]
```

For ranking-plan workstream 5: when the two outputs come from different tools, this tells
you whether one tool's winner lies outside the other's search region.

## References

- CADO-NFS `polyselect/`: `size_optimization.c`, `auxiliary.c`, `polyselect_alpha.c`,
  `polyselect_norms.c`, `ropt*.c`, `area.h`
- msieve stage 2: `gnfs/poly/stage2/` (`stage2.c`, `optimize.c`, `root_sieve*.c`),
  `common/minimize.c`
- Bai, Brent, Thomé, "Root optimization of polynomials in the NFS" (2015):
  https://arxiv.org/abs/1212.1958
- Bai, PhD thesis (2011): https://maths-people.anu.edu.au/~brent/pd/Bai-thesis.pdf
- David, Zimmermann, "A New Ranking Function for Polynomial Selection in the NFS" (2020):
  https://inria.hal.science/hal-02151093
- Other references: see `POLYSELECT_RANKING_IMPROVEMENT_PLAN.md`
