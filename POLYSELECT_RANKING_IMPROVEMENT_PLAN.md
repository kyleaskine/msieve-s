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
- Were winners near msieve's `stage1_norm` cutoff? If the correlation is loose or
  winners sit near the cutoff, loosen `stage1_norm`. This costs little GPU time
  but adds CADO size-opt load.
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
