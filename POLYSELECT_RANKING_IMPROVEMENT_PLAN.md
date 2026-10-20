# GNFS polynomial selection: improvement work, handoff brief

> Brought over from a browser chat on 2026-10-02. This covers ranking and
> post-stage-1 work (size-opt, ropt, Murphy-E blind spots, sieve geometry).
> Stage-1 GPU kernel work lives in `POLYSELECT_OPTIMIZATION_NOTES.md`.
>
> **Status 2026-10-03:** the M0 checks in `GPU_STAGE23_PLAN.md` (run on a c146 job)
> settled several items below; each is marked **Done** with the result. Tools and test
> data for rerunning them are in `stage23_bench/`.
>
> **2026-10-04:** checks from an outside review (`GPU_STAGE23_PLAN.md`, "Review checks")
> qualify conclusions 1 and 5, answer the `las` question under "Verify first", and move a
> modest lattice-aware calculator (workstream 1) earlier.

## Stage-1 geometry and quality audit (2026-10-19)

The hypothesis that a large stage-1/raw norm rules out an excellent final
polynomial needs measurement. The first saved-fixture check gives counterexamples
to a hard cutoff on raw optimized-skew L2 norm; it does not establish that the
current stage-1 search policy is optimal.

### What the current controls actually do

- The January 4 change (`0a1b7a3`) sets the effective stage-1 bound to
  `min(configured_stage1_norm, 1637.8 * m0^0.7)` for degree 5. It caps the
  configured bound; it does not raise it. The automatic expression is described
  as pushing special-q to a representation limit, not as maximizing final quality.
- `stage1_bounds_update` uses that same bound to derive the estimated third-highest
  coefficient limit and leading rational coefficient scale. The GPU host also
  uses the coefficient limit in `check_found_array` to accept/reject collisions.
  Thus geometry and admission currently share a parameter.
- There is an additional, separate geometry expansion: the GPU computes q_max
  from the base p_max, then multiplies p_max by **64**. This was already 32 before
  the automatic-norm change, and became 64 in `32dcfe5` on March 12. q_max is not
  recomputed after this expansion. Products p1*p2 can therefore be up to 4096
  times the base scale. This is not proof that every resulting polynomial is
  poor; it is a reason to measure the actual rational coefficient and quality
  rather than treating the nominal norm as a complete description of the search.
- A further bound exists in `pol_expand`: the absolute expanded a_(d-2) is checked
  against the *configured* stage1_norm, not the per-leading-coefficient effective
  bound. This is also not the optimized-skew L2 norm. Current `poly_sizeopt_run`
  writes raw polynomials and returns before the legacy optimization call.

For fixed m0 in degree 5, writing B for the effective stage-1 bound gives:

```
skew_min = (m0/B)^(2/3)
coeff_max = B^(4/3) / m0^(1/3)
P_scale = B^2 / m0
```

When the GPU's `sqrt(0.5/target)` constraint selects base p_max, and before
integer rounding or clipping, base p_max scales as B^(-2/3) and q_max as
B^(10/3). Halving B changes q_max by about 10x and increases p_max by about 1.59x.
A norm sweep changes which regions are searched, not merely which hits survive.
The automatic cap's comment refers to CPU constraints; it should not be assumed
to put the GPU q_max just below 2^32 in every regime.

### Check against saved raw, size-optimized, and root-optimized results

Reproducer: `stage23_bench/runs/20261019_stage1_audit/audit.py`, with
`norm_probe.cpp` and `norm_audit.json` in the same directory. CPU only; no new
stage-1, size-optimization, or root-optimization search was run.

Used all 13,000 c146 and 11,000 c161 saved raw/sopt pairs. Each fixture includes
10,000 random rows from outside its saved top-sopt group, plus that top group
(3,000 and 1,000 respectively). Percentiles and correlations below use only the
10,000 random reference rows, not the deliberately oversampled top group.
Joined the saved per-seed ropt results by Y1; each sample has unique Y1 values.

Raw lognorm is the circular algebraic L2 lognorm at the raw polynomial's own
optimal skew, calculated using `stage23_gpu/include/norms.h`. As a check, the
same norm implementation evaluated all 24,000 saved optimized polynomials at
their printed CADO skews within 0.005 of the printed two-decimal lognorm. CADO's
independent `cado_expe -keep-skew` agrees with both winners' raw norms to the nine
printed decimal places.

| Fixture | Known winning seed raw lognorm | Fraction of random references with a smaller raw norm | Saved effort-0 sopt lognorm |
|---|---:|---:|---:|
| c146 | 53.228062808 | **99.94%** | 43.95 |
| c161 | 58.847751649 | **74.18%** | 47.77 |

Both tools' best recorded seeds agree within each fixture. Therefore discarding
the highest-norm 10% of the c146 random-reference distribution would discard its
known winning seed; keeping only the lower-norm half loses both fixtures' winners.
These are percentile diagnostics within already-generated candidates, not claims
that changing stage1_norm reproduces these exact subsets.

Raw lognorm versus post-sopt exp_E has Spearman correlation **0.0713 (c146)** and
**0.0696 (c161)** in the random references. Keeping the lower-raw-norm half would
retain only 508/1000 and 502/1000 of the best saved effort-0 sopt candidates.
This is little enrichment for a filter that discards half the candidates.

Also checked the coefficient-derived diagnostic
`B_proxy = (m0 * abs(raw_a3)^3)^(1/4)`, obtained by inverting the degree-5
coeff_max formula. It is not the exact collision gate: the host uses an
approximation to the transformed correction, and expansion can alter coefficients.
Its correlations with post-sopt exp_E are 0.0151 and 0.0332; the winners are at
the 92.64th and 79.77th percentiles. This diagnostic likewise gives no support for
assuming that a large raw high coefficient is automatically hopeless.

Limitations: only two numbers, an already-selected stage-1 population, and ropt
results only for selected sopt seeds. These data cannot measure what a tighter
geometry would discover in the time saved, or candidates excluded by the original
generation policy. Raw algebraic L2 norm, the stage-1 B parameter, and final
size/root/sieve quality are different measurements. Translation and rotation
can change size substantially; root properties also matter
([Bai, Brent, Thome, section 2.4](https://members.loria.fr/EThome/files/rs.pdf)).

### Next experiment: separate geometry, admission, and time allocation

1. Log each hit's actual search geometry and collision-gate value, alongside the
   raw identity `(a_d, P, m)`, and carry this identity through sopt and ropt.
2. Hold geometry fixed while varying an independent acceptance threshold.
   This isolates the cost/quality of admitting marginal hits. A filter applied
   after collision search cannot recover the stage-1 work already performed.
3. Separately sweep geometry (effective B, p-range expansion, q range, leading
   coefficient allocation) with a permissive common admission policy. This
   measures whether time spent elsewhere finds stronger seeds.
4. Compare best common-parameter final scores versus **equal total wall time**,
   not equal raw hit counts. Preserve a stratified sample of candidates that a
   proposed early filter would reject and run them through the same downstream
   effort, so validation does not hide false rejections. Test-sieve finalists.

Keep the present policy until such a comparison supports changing it. The
optimization target is final polynomial quality per search budget, with hardware
range limits treated as feasibility constraints rather than quality objectives.

### Leading-coefficient allocation audit (2026-10-19)

The distributed project's local checkout is
`/home/kylea/code/polyselect-distributed`. Its `polylocal/__main__.py` table uses
the following `(min_coeff = high_coeff_mult, raw-count target)` choices:
420/400k for C145–158; 2520/800k for C159–172; 27720/1.2M for C173–186;
138600/1.6M for C187–200; 360360/2M for C201–215. These are starting values and
mandatory divisors, not upper bounds on a_d. The defaults do not force 17 or 19.
Range enumeration additionally filters the quotient by small-prime factors up to
97, with a power limit of 2 (4 above 200 digits). `coeff_list=1` bypasses this
enumerator and is the appropriate existing control for experimental coefficient
families. The current GPU stopping policy finishes the coefficient in progress;
the raw-count target does not provide equal time or equal per-coefficient depth.

Magnitude, prime factorization, and effort per coefficient must be varied
separately. A larger a_d can reduce skew, but minimizing skew is not the objective:
size, root optimization opportunity, and the actual sieving region all matter.
Do not assume a historical preference for larger a_d was solely a skew choice.

For a prime p dividing a_d but not a_(d-1), the projective root is simple and
has contribution `-p*log(p)/(p*p-1)` to projective alpha. This contribution does
not increase merely because v_p(a_d) increases. If p also divides a_(d-1), the
root is multiple and lifting depends on further coefficients. Thus additional
powers of 2, 3, or 5 are not automatically better than introducing 17 or 19.
The latter each give a projective root; the simple-root contributions are about
-0.167 and -0.155 respectively. These are isolated contributions, not guaranteed
net improvements between different polynomials. See
[Bai, Brent, Thome, section 2.3.1](https://members.loria.fr/EThome/files/rs.pdf).
Actual projective alpha after size optimization is more informative than a
smoothness label; size optimization may also change the leading coefficient.

`stage23_bench/runs/20261019_stage1_audit/coeff_audit.py` and its JSON output
summarize the saved fixtures by *original* a_d. c146 covers only eight coefficients
from 420 to 3780; c161 covers nine from 2520 to 25200. All are 7-smooth, so these
fixtures cannot test introducing/removing 11, 13, 17, or 19. As a descriptive
example, c161 a_d=5040 contributes 16.23% of the random remainder and 23.1% of
the saved top-1000 sopt group, whereas 22680 contributes 19.58% and 13.7%.
Nevertheless, 22680 produces the known final winner. Neither mean/top-group
enrichment nor one winning coefficient establishes quality per GPU-hour.
Per-coefficient search exposure and timing are absent from this comparison.

Suggested experiment, not a new default policy:

1. Use explicit coefficient lists in several overlapping magnitude bands around
   today's choices, including smaller values. Compare near-equal magnitudes with
   different factorizations, e.g. 16M versus 17M and 18M versus 19M where M has
   small-prime factors but neither 17 nor 19. Use several such pairs and numbers.
2. Give each family an initial equal time budget; log a_d, q/p ranges and root
   counts, stage-1 time, unique raw hits, optimized norm/projective alpha/skew,
   and final scores. Split q work into resumable chunks for a genuine depth
   comparison. The current q enumerator traverses a factor-combination tree,
   not increasing numerical q. An early prefix is a particular part of that
   traversal, not a random sample or necessarily a low-q sample; measure the
   actual q/factorization distribution as well as depth.
3. Compare broader/shallow allocation with fewer/deeper coefficients at equal
   total cost, including initialization. Retain exploration across families;
   do not eliminate one solely because a short pilot lacks a rare winner.
4. Track best final score versus cumulative time and exceedances of strong fixed
   score thresholds. Use common downstream effort and test-sieve finalists.
   A raw-count target alone cannot distinguish productive search from abundant
   mediocre candidates.

No leading-coefficient bounds, scheduling, or production code changed.

### Diminishing returns within one leading coefficient

Code inspection does not establish that later hits have worse final quality.
`get_next_enum` in `stage1_roots.c` traverses a stack of small-prime factors and
powers. The GPU builds the p progression set once, then processes successive
q-root batches, applying the same collision acceptance bound throughout.
The optional q-piece randomization is currently unreachable: num_pieces is
capped at 50, while its branch requires more than 51. Small trailing batches
can also search additional progression offsets. Thus position in the run can
correlate with search geometry, but this is not a quality-ranked traversal.
GPU collision ordering and asynchronous stage-2 tasks also mean literal output
line order is less useful than a recorded batch identifier.

Distinguish declining record-breaking probability from declining quality of
the draws. Under the illustrative model of independent, identically distributed
continuous final scores, the nth candidate has probability 1/n of being the best
so far, although its score distribution is unchanged. Switching a_d does not
reset the existing global best: if the distribution and cost are identical,
switching provides no statistical advantage. Breadth helps only to the extent
that it discovers better distributions, reduces dependence, or improves cost
per useful candidate; setup cost can favor depth.

To test actual deterioration, compare equal-work batches at successive depths
using fixed quality thresholds and standardized downstream optimization,
logging q factors, P=q*p1*p2, hit counts and elapsed time. A flattening running
maximum alone is not evidence of worse late candidates. The saved sopt fixtures
are ranked/sampled and do not carry original batch IDs, so they do not answer
this temporal question directly.

## Context

Pipeline (c200-class jobs, currently degree 5):
1. msieve GPU stage 1 produces millions of raw polys (much faster than CADO).
2. Every raw poly goes through CADO size optimization (cheap, far better than msieve's).
3. Sort by CADO `exp_E`; keep the top 150–300.
4. Root-optimize those with BOTH msieve and CADO ropt. They diverge in interesting
   ways; neither is strictly better.
5. Final pick by Murphy-E and/or test sieving.

Field observations:
- The best polys come from unusually small lognorm, not exceptional alpha. A top
  poly is often ~20% better than anything else from 2M+ raw polys.
- Known leak: polys with very high skew (~1e9 at c204) rank well on Murphy-E but
  sieve poorly, with relations at low special-q coming in much lighter than expected.
- Skew ~1e8 at c204 sieves fine.

Reference fixture (record-class c204, degree 5):

```
n: 167607202751713520755395238547505175579109747181766065729512289906719395455186020967457134128228376355120680855258479360585078682339636368676745445285657085644774094525261305605926676266686577843002833121
Y0: -3704498996216684411842794304917527627729
Y1: 937553014235954675840893
c0: 5985437540624063729983612458960011252682473616
c1: -1144032245234742198596820070888957088452
c2: 51877342678250806689871851475359
c3: 211913712046660820491492
c4: -1584472727122620
c5: 3603600
skew: 114791685.398
# lognorm 61.27, E 53.62, alpha -7.65 (proj -3.08), 3 real roots
# CADO MurphyE 4.237e-15
```

## Conclusions so far

1. **For this workflow, Murphy-E's biggest blind spot is lattice-sieve geometry,
   not root properties.**
   - Murphy-E integrates over an idealized skewed ellipse of fixed area,
     independent of special-q.
   - Real lattice sieving reduces each special-q lattice in skewed coordinates
     (a/sqrt(s), b·sqrt(s)) and sieves an I × J box (i ∈ [−I/2, I/2), j ∈ [0, J))
     in the reduced basis.
   - When s ≫ q, reduction cannot balance the basis. It degenerates to
     u = (q, 0), v = (r′, 1), so i steps a by q and j is b itself.
   - The box then covers |a| ≤ I·q/2, 0 ≤ b < J. The ideal same-area half-disk
     has half-width R·sqrt(s) and height R/sqrt(s).
   - The aspects match when s = I·q/(2J), which is s = q for J = I/2. Beyond that,
     both dimensions are off by a factor k = sqrt(s · 2J / (I·q)): the a-range is
     k times too short and the b-range k times too tall.
   - Example: s = 1e9, q = 1e8 gives k ≈ 3.2. Rows at b up to ~3× optimal have
     degree-5 algebraic norms up to ~3^5 times larger.
   - The damage is concentrated at low q, which should be the most profitable range.
   - *Qualified (2026-10-04):* CADO's `las` does not sieve one fixed box for every
     special-q.
     - With `adjust-strategy` 0 (the default) it keeps logI and sets J per special-q so
       that the boundary is capped.
     - With 1 it caps the norm in the (a, b) plane instead.
     - With 2 it chooses logI and a skewed basis per special-q by estimated yield,
       trying a few alternative bases (`sieve/las-choose-sieve-area.cpp:94`,
       `estimate_yield_in_sieve_area` in `sieve/las-norms.cpp:1073`).

     So the k-factor analysis above describes a fixed I × J box. What matters is the
     region the siever in use actually covers. The user's GPU siever
     (`~/code/cuda-sieve`) is the one to model. It is also no reason for a universal
     high-skew penalty: the question is how a given poly meets the job's q range and
     region.

2. **The root-property variance blind spot is already addressed by David–Zimmermann's
   E′ (2020).**
   - E′ replaces alpha's constant shift with the distribution of
     Y = Σ_{p ≤ B} ν_p · ln p, fitted as a non-central chi-squared.
   - At c120, E′ picked a different poly in 44 of 100 numbers, mostly faster. Their
     RSA-155 example shows a ~15% ranking swing.
   - Weaknesses:
     - The chi-squared tail is wrong. They truncate the integral at 6c because a
       larger cutoff made results worse.
     - B = 2000 is too small.
     - The root sieve still optimizes mean alpha.
     - Only tested at c120.
   - Implemented on CADO branch `dist-alpha`; unknown whether merged into master.

3. **`exp_E` is lognorm + expected_rotation_gain, not lognorm + alpha.** **Done**
   (verified in CADO source, `polyselect/auxiliary.c:211`, `:346`): projective alpha +
   expected best affine alpha given the rotation-space size + 0.1 per rotation degree.
   - It already includes a root-potential estimate.
   - It rewards skew directly, because a larger rotation space gives better expected
     alpha. CADO sopt also picks its translation by this score
     (`size_optimization.c:973`, `:1091`), so the lean toward high skew starts in
     size optimization, not only in the sort.

4. **Useful numbers from Bai's thesis (ANU 2011):**
   - Expected best alpha from a rotation space of ~s^6 is about
     −0.257 − 1.648·sqrt(3 ln s) (alpha modeled as normal, σ ≈ 0.82). Going from
     s = 1e6 to 1e9 is worth ~2.4 nats.
   - Size-then-root beats root-then-size.
   - Translation during size-opt inflates c_{d−1} and shrinks the rotation space.
   - Kleinjung's pre-rotation trick (try f + δ·x³·g, |δ| ≤ 256, before
     translation) moved the mean log L2 norm from 70.34 to 69.84 on an RSA-768
     sextic sample.

5. **Rough sensitivity at c204:** ~0.13 log-yield per nat of lognorm (back-of-envelope:
   ξ(u)/ln B). A 20% outlier is ~1.4 nats. Calibrate with real test sieving.
   - First calibration (c146, 7 finalists, c145 parameters, `GPU_STAGE23_PLAN.md`):
     MurphyE gets 20–27% gaps right, but has a 3–4% RMS error at the top. One poly
     MurphyE put 11% behind the winner sieved only 1.7% slower; it had the lowest
     skew of the top group, which fits the high-skew blind spot above.
   - The 3–4% is a residual on 7 polys from one job, not a fundamental floor. It mixes
     three separate errors:
     - numerical error: CADO's 1,000-point integral reads high by about 0.2% and up to
       1.4% (`GPU_STAGE23_PLAN.md`, "Review checks");
     - the idealized region against the special-q regions actually sieved;
     - a smoothness model that does not describe the real relation-acceptance rules
       (lims, large-prime counts, mfb, cofactorization cost).

## Verify first (CADO source)

- [x] `grep -rn exp_E polyselect/` and find `expected_rotation_gain`: what exactly
      feeds the sort? **Done:** see conclusion 3. The pipeline sorts on the `exp_E`
      that CADO sopt prints (`utils/sort_cado_by_expe.py`, then the exp_E column of the
      msieve-format file); CADO ropt ranks by MurphyE at its default parameters.
- [ ] Is E′ (`dist-alpha`) in master? What does the final ranking use?
- [x] `las` adjust-strategy modes: does any reshape the sieve region per special-q?
      **Done (2026-10-04):** yes. Mode 0 (default) sets J per special-q, 1 caps the
      (a, b)-plane norm, 2 picks logI and the basis by estimated yield, 3 combines 2 and
      0. See conclusion 1.
- [ ] Exactly how `las` reduces the q-lattice with skew (match it in the calculator).
- [x] `sopteffort` semantics. Does size-opt already do any pre-rotation? **Done:** each
      unit of effort adds 16 rational q2 values in [−1, 1] for translation candidates;
      CADO solves for real q2 analytically, and its LLL lattice already contains the
      lower rotations and f itself (so it also searches the multiplier, Res = a·N).

## Workstreams (priority order)

### 1. Lattice-aware E calculator (highest value)

*2026-10-04 (review):* start a modest version now, alongside the GPU ropt (M2), rather
than after it.
- Reuse the GPU siever's own q-lattice reduction (`qlat_build` in
  `~/code/cuda-sieve/bench/poly.c:905`, skewed Gauss reduction as las does) and its
  norm evaluation. Sample the region that siever actually visits, with the forced
  special-q factor on the correct side, over several q bands.
- Prior art: las's yield estimator (`estimate_yield_in_sieve_area`) already integrates
  ρ·ρ over a candidate sieve region to choose between bases (conclusion 1).
- Keep the integration accurate (K ≥ 16,000 or split at the real roots). Otherwise
  geometry effects of 1% drown in the quadrature noise.

Inputs:
- CADO .poly file (n, c_i, Y0, Y1, skew).
- logI (I = 2^logI, J = I/2), special-q side, q range [q0, q1].
- lim0/lim1 (later lpb/mfb for semismooth).
- Sample counts.

Algorithm:
1. Sample primes q across the range. For each root r of f mod q (one special-q
   ideal each), take the lattice basis (q, 0), (r, 1).
2. Gauss-reduce under the skewed norm, matching las.
3. Map an (i, j) grid or Monte Carlo sample of the box to (a, b) = i·u + j·v.
4. Score each point:
   ρ((ln|F(a,b)| − ln q + α_f) / ln B_f) · ρ((ln|G(a,b)| + α_g) / ln B_g)
   (Murphy-style first). Later, replace ρ with semismooth probabilities using
   the actual lpb/mfb.
5. Per-q yield = mean score × #points. Integrate over q with the special-q ideal
   density (~1/ln q).
6. Also compute the same integral over an ideal same-area half-disk to isolate the
   geometry loss.

Outputs:
- Per-q predicted yield curve, total, and comparison against Murphy-E and the
  ideal-shape integral.

Validation:
- Test-sieve the 1.15e8-skew fixture and a ~1e9-skew candidate at several q (las
  with q0/q1 or `-random-sample`). Compare per-q relation counts to the predicted
  curve shape.

Key experiment:
- Compare both polys at q ≥ ~1e9, where neither is distorted. If their yield ratio
  matches the Murphy-E ratio, the low-q deficit is purely geometry.

### 2. Funnel analysis on existing logs (no new algorithm)

**Partly done** (c146): seed level only. exp_E rank vs best ropt score has Spearman
0.35 (top 300) / 0.42 (top 1000); the winner was exp_E rank 84 and the second-best
seed rank 407 (`stage23_bench/tools/seed_funnel.py`). The stage-1 score questions
below are still open.

Join records across stages:
- msieve stage-1 raw score
- after CADO size-opt: lognorm, skew, exp_E
- after ropt: alpha, MurphyE (msieve and CADO)
- test-sieve yield where available

Questions:
- How well does msieve's raw score predict CADO size-opt lognorm?
- Where did eventual winners rank at each stage?
- Were winners near msieve's actual collision/expansion cutoffs? Track the
  independent effects on geometry, admission, and final quality per unit time;
  changing `stage1_norm` can substantially change GPU work. The 2026-10-19
  audit above finds that raw norm alone is a poor filter in the saved fixtures.
- What predicts extreme lognorm (leading coeff c_d, raw skew, msieve score)?

### 3. Size-optimization headroom

**Done, negative result** (c146, M0 in `GPU_STAGE23_PLAN.md`): effort 50 on the top
3000 gains 0.014 nats on average and puts nothing new in the top 150; pre-rotation with
|δ| ≤ 256 on the top 300 seeds gains > 0.05 nats on 4 of 300. CADO sopt at effort 0 is
already close to the exp_E ceiling for these raw polys. Not a priority any more.

- Rerun CADO size-opt on the top few thousand at higher sopteffort. Measure lognorm
  deltas and rank changes.
- Implement the Kleinjung pre-rotation before translation: f + δ·x^(d−3)·g
  (δ·x²·g for quintics, δ·x³·g for sextics), |δ| ≤ 256, keeping the
  best result.
- Any mean lognorm shift applies to every poly, including the eventual winner.

### 4. Exact-distribution E′ (improves on David–Zimmermann)

- Extend their Algorithm 1 (Hensel-lifting recursion) to return the full pmf of
  ν_p, affine plus projective via rev(f)(p·x), not just two moments.
- Convolve the per-prime pmfs of ν_p · ln p on a fine grid (~0.01 nat) for
  p ≤ B. Use larger B (up to lim) for finalists.
- E′_exact = ∫ over the region and y of
  ρ((ln|F| + c − y)/ln B_f) · ρ(…g…) dP(y), where c = Σ ln p/(p−1).
  No chi-squared fit, no 6c truncation.
- Validate rank agreement of E, E′(χ²), and E′(exact) against test sieving at
  c150–c200.
- Later: a tail-aware root-sieve score. Per prime, (1/k)·ln E[p^(k·ν_p)] with
  k ≈ 0.1 is additive across primes, so it drops into the root sieve.

### 5. msieve vs CADO root-opt diagnostic

- When msieve ropt beats CADO ropt on the same input, recover its rotation
  (u, v[, w]) from (f_msieve − f) / g.
- Check whether that rotation lies outside CADO ropt's search bounds.
  - Outside: a coverage problem. Search the ellipsoid where size stays within
    budget; size is a positive-definite quadratic in the rotation parameters.
  - Inside: a scoring problem.

### 6. Windowed lattice siever (only if #1's validation says geometry explains the deficit)

- For q ≪ s, sieve k ≈ sqrt(s·2J/(I·q)) windows of I × J/k, offset in i, instead
  of one I × J box. Same point count, correct shape.
- The Franke–Kleinjung per-prime enumeration basis depends only on window width I;
  each window needs only new per-prime start points.
- Payoff: high skew is where root-opt finds the best alpha, so traps could become
  winners.

### Parked

- BBKZ resultant multiplier c (Res = c·N) as a root lever. RSA-240 had 120N and
  RSA-250 had 48N, both smooth. Test whether smooth c is over-represented among
  top-ranked vs all size-opt outputs. **Mostly answered:** CADO sopt already searches
  c (its LLL includes f), and |c| > 1 for 85% of outputs; this doc's fixture has
  Res = 15N and both 2026-10-02 winners have 2N. Top polys have much smaller c
  (median 15 in the top 150 vs 753 overall). Forcing other c values (1..500) on the
  top 300 seeds almost never beat CADO's choice on exp_E, and runner-up c values ropt
  a median 6% worse. Smoothness specifically (vs size) is not yet tested.
- Cross-side small-prime divisibility correlation: dead end (~0.1% at p = 2).
- Non-linear / Joux–Lercier-style pairs for SNFS cofactors: separate topic.

## References

- David, Zimmermann, "A New Ranking Function for Polynomial Selection in the NFS"
  (2020): https://inria.hal.science/hal-02151093
- Bai, PhD thesis (2011): https://maths-people.anu.edu.au/~brent/pd/Bai-thesis.pdf
- Bai, Brent, Thomé, "Root optimization of polynomials in the NFS" (2015):
  https://arxiv.org/abs/1212.1958
- Bai, Bouvier, Kruppa, Zimmermann, "Better polynomials for GNFS" (2016):
  https://hal.archives-ouvertes.fr/hal-01089507
- Boudot et al., RSA-240/DLP-240 (2020), polyselect section:
  https://arxiv.org/abs/2006.06197
- Kleinjung 2006: https://www.ams.org/journals/mcom/2006-75-256/S0025-5718-06-01870-9
- Kleinjung 2008 slides: https://members.loria.fr/EThome/cado2008/slides/kleinjung.pdf
- Thomé, CSE291 lecture 9 (pipeline overview):
  https://members.loria.fr/EThome/teaching/2022-cse-291-14/slides/cse-291-14-lecture-09.pdf
