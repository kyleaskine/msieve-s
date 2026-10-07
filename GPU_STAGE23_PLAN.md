# GPU size-opt + root-opt (stages 2 and 3): plan

> Started 2026-10-02. Companion docs:
> - `POLYSELECT_RANKING_IMPROVEMENT_PLAN.md`: ranking and objective work (Murphy-E blind spots, E′, lattice geometry).
> - `POLYSELECT_OPTIMIZATION_NOTES.md`: stage-1 GPU kernels.
>
> Numbers marked *(est.)* are back-of-envelope and have not been measured.

## Goal

Run size optimization (sopt) and root optimization (ropt) on the GPU with our own code.
(As built, M1's sopt is a deliberate bit-exact port of CADO's, so it could be checked
poly by poly; the ropt of M2 is our own design.)

1. **One device.** The whole post-stage-1 pipeline runs on the 5070 and the CPU is free.
2. **Better polys.** Search the rotation space of the top seeds much more thoroughly than
   either CPU tool can afford, and control the objective used *inside* the search.
3. **Independent code.** Keep it separate from CADO and msieve so the result can be compared
   against both. The current msieve + CADO hybrid stays as the reference.

Speed alone is not the motivation. The CPU pipeline takes about 2 h, while stage 1 takes
about 24 h on a rented 5090.

## Current status (2026-10-05)

This section supersedes the conclusions written along the way in the milestones below.
Where an older claim turned out wrong or weaker than first written, it is marked in place.

- **M1 (GPU sopt) is done.**
  - It is byte-identical to CADO on c146, c161 and all 772,402 polys of c168. On c168
    with an idle GPU:
    - effort 0 takes 14:32 wall, against CADO's 25.6 min on 8 threads (1.8×);
    - effort 50 on the top 1000 takes 63 s, against about 2.5 min.
  - On c208 (2,198,964 polys, CPU shared with the pipeline; M2, "Breadth run, c208"):
    - effort 0 took 51 min wall, against about 111 min for CADO on 8 threads;
    - 12 outputs differ from CADO's: the first differences on any job. They have the
      same printed exp_E and a different translation or constant rotation, all are no
      worse at full precision, and none is in the top 2000;
    - effort 50 on the top 2000 took 145 s and was identical 2000/2000.
- **M2 (GPU ropt) is competitive with CADO's ropt at c168 size, but not at c208 size.**
  - **c208 (2026-10-05, rerun 2026-10-06):** on MurphyE the GPU's best is 2.6% behind msieve's best. It
    loses 5–30% on msieve's best seeds. Every such loss checked is a coverage loss, from
    the cell cap (winners at +1.3 to +2.1 nats with the budget cut to 0.75–1.75) or from
    the 1000-line scan (a basin at u = −16152). Covering those cells densely costs
    1e13–1e14 cells per seed, so at this size the selective search (Next, item 3) is
    required.
    - *But (test sieve, 2026-10-06):* MurphyE overrates high-skew polys under a lattice
      siever (known problem 7), and every one of those msieve cells has skew
      3e8–1.6e9. So the losses are overstated.
    - The two best polys by test sieve are both from seed 10, where the GPU's best cell
      also is.
    - *C181 (2026-10-07; M2, "C181"):* breadth (27 min) and pass 2 on the top 16 (31 min)
      find msieve's winning cell, 0.9993 against 1.0000: the GPU writes it at another
      translation (known problem 10). CADO's best is 0.9786.
    - *Rescored with the lattice-aware score (2026-10-06):* the GPU's best is 3.3% behind
      the best (CADO orig, seed 10). On that seed the gap is coverage: at budget 1.5
      (5e12 cells) the GPU finds CADO's exact cell and ties it.
    - *Two-pass test (2026-10-06; M2, "Progressive allocation, a first two-pass test"):*
      a second pass on the 16 leading seeds (27 min, cap 6e12) **ties the best poly**:
      CADO orig's exact cell, lattice score 1.0000.
      - Scoring only 32 of the 200 outputs had hidden a 0.9841 poly in the breadth pass.
        So the gap before pass 2 was 1.6%, not 3.3%.
      - The next loss found was retention by the proxy, whose truncated alpha is off by up
        to 0.9 nats (known problem 9). It is fixed by re-ranking with CADO's alpha
        (`-rerank`, the default since 2026-10-06).
      - *2026-10-07:* the re-ranked breadth pass (36 min) puts seed 10 first. A third
        pass, the next budget step on the three leaders (16 min), finds nothing better
        than CADO's cell.
  - It recovers every known optimum of the benchmark jobs.
  - On c168, scored with MurphyE under the job's parameters at 16,000 sample points (see
    "Review checks"; run before the 2026-10-04 fixes):
    - against CADO's ropt on its 150 seeds: better 31, tie 106, worse 13;
    - against msieve's on its 300: better 136, tie 110, worse 53.
  - Some losses are large: up to 15%, on flat seeds under the cell cap.
  - It has not shown a better final poly. No GPU-found poly beats a pipeline winner by more
    than MurphyE's noise, and none has been test-sieved.
  - It is not one-device yet. These still run on the CPU:
    - the size model, about 75% of ropt time;
    - the re-rank of the best 4096 cells by CADO's alpha (about 0.5 s a search);
    - MurphyE scoring of the survivors;
    - content-seed generation (`stage23_bench/tools/content_seeds.py`).
- **Known problems, verified 2026-10-04** (details in "Review checks"; items 1–3 fixed
  the same day, see "Fixes"):
  1. CADO's MurphyE samples only 1,000 points, so it is noisy at the 0.2–1% level.
     Searching skew or translation on it fits the noise. The findings "translation is
     worth 1–2%" and "MurphyE has several peaks in skew" were artifacts of this.
     *Fixed:* `cado_murphy -K/-Keval` and `score_polys.py --points/--eval-points` score at
     more points.
  2. Projective alpha was assumed constant under linear rotation, but it changes. Across
     the 1000 c168 seeds it varies on 306, by up to 0.81 nats, and on 207 it varies
     within a single u-line too. *Fixed:* projective roots are in the sieve tables. On 100
     c168 seeds, 2 seeds' best improved (+3.1%, +1.4%) and none got worse.
  3. The sieve's per-block choice was not the exact top K. Each thread kept its best
     cell and the block kept the best K of the 256 thread winners, so a third of blocks
     lost part of their true top 8. *Fixed:* it is exact now, at the same speed. On 100
     c168 seeds no seed's best changed.
  4. The cell cap lowers the budget. Flat seeds therefore lose high-alpha cells that sit
     far above the size minimum. *On c208 it sets the budget for most seeds (median 1.5
     nats) and causes every large loss.*
  5. The band search is not exhaustive. The bands come from the size model's local
     searches, they assume one v interval per line, and stopping rules end the scan in u.
     *On c208 the 1000-line scan missed basins thousands of lines out. msieve's winners
     sit at u = 2014 and u = −16152, and with 4000 lines seed 32's best line minimum is
     1.06 nats lower.*
  6. *(2026-10-05)* The sieve's block list was unbounded and also built by `-plan`: nine
     runs took 107 GB and crashed the machine. *Fixed 2026-10-06:* the blocks are
     generated a launch at a time, so memory no longer grows with the search ("Memory
     guard").
  7. *(2026-10-06)* **MurphyE ignores the lattice siever's geometry.** A special-q lattice
     can't realize a skew much above (I/J)·q. On c208 two skew-1.33e9 polys that MurphyE
     put within 1.5% of the best sieved 15–20% worse at q = 80M, and 5–12% worse over
     80M–1G (M2, "Breadth run, c208", test sieve). Every ranking here inherits this:
     msieve's and CADO's ropt, and this project's comparisons. *Fixed for scoring
     (2026-10-06):* `score_polys.py --lattice` reproduces the test sieves within 0.04
     (Next, item 5). The ropt proxies still rank by size and alpha only.
  8. *(2026-10-06)* Each `s23_ropt` GPU launch ran about 2^36 cells, seconds per kernel.
     Under WSL2 the Windows desktop shares the GPU, and long kernels made it lag.
     *Fixed:* a launch holds about 2^32 cells (`-launch`, about 0.1 s each) and at least
     four waves of blocks. Throughput is the same at the default `-seg` (3.53e10 against
     3.46e10 cells/s), and outputs are byte-identical. `s23_sopt`'s 10 s `-slice` would
     lag the same way; a short slice is a flag away, untested.
  9. *(2026-10-06)* **The proxy's alpha is too coarse to choose the survivors.** alpha_s
     stops at p < 200 and p^e ≤ 200. Against CADO's alpha to 2000 it is off by −0.24 to
     +0.93 nats per cell, so 0.3–1.2 nats of E at `-aw 1.3`. The 200 kept cells span
     only about 0.9 nats of E.
     - A c208 seed-10 cell scoring 0.9841 sat at proxy rank 57 at budget 0.75, and it
       fell out of the 200 at budget 1.25.
     - Scoring only the first 32 outputs (the c208 runs before 2026-10-06) loses more.
     - *Fixed (2026-10-06):* `s23_ropt -rerank N`, default 4096, re-ranks the best N by
       the proxy using the exact lognorm and CADO's alpha. Within a run, rank correlation
       with the lattice score rises from 0.52 to 0.91. The lost cell is kept, and on 16
       c208 searches the best cell is at rank 7 at worst, against 95. See M2, "Re-ranking
       by CADO's alpha".
     - *Fixed 2026-10-07 after a code review:* content cells were re-ranked with their
       content left in, (aw − 1)·log d too well. On a c168 seed with d = 2 they filled 73
       of the 200 written cells, against 40 after the fix. Each block's best K are still
       chosen by alpha_s, but K = 16 wrote the same 200 cells as K = 8 on the two leading
       seeds tried.
  10. *(2026-10-07)* **The written translation is the lognorm's, not the score's.** For
      each cell the size model takes the integer t that minimizes the L2 lognorm. On the
      C181 job, on the five leading seeds checked (5, 6, 10, 0, 11), msieve's best cell is
      one the GPU also wrote, at another translation (tens of millions apart, at skews of
      about 5e7). msieve's scores 0.07–0.9% better, by the lattice score and by MurphyE
      alike. Every per-seed lead msieve has there over the GPU is this (M2, "C181").
      - This is not what c168 showed ("translation is worth about nothing"). There the
        gains came from tuning t on 1,000-point MurphyE, and vanished at 64,000 points.
        Here the two scores agree at 16,000 points and in the lattice score.
      - Not fixed. A search over t by the accurate score for the finalists is the
        candidate. Check it against fitting the score's fixed sample points, and keep
        the lognorm's t as a control.
- **Content seeds** are now generated by a tool, `stage23_bench/tools/content_seeds.py`.
- **Exactness is not retention.** The block choice is now the exact top K *by the proxy*.
  A cell ranked K+1 in its block can still be the seed's best by accurate MurphyE: the
  proxy interpolates the size between knots and truncates the root information. Every
  pruning boundary below can still lose a winner.
- **Accurate scoring is opt-in at the CADO level.** `cado_murphy` defaults to CADO's
  1,000 points so it can reproduce CADO. `score_polys.py` defaults to 4,000/16,000 points
  and needs `--cado` for CADO's numbers.
  - `-Keval` re-evaluates the configuration the coarse search chose; it does not search
    again. A configuration discarded at the coarse level is not recovered.
  - For very close finalists, refine several configurations at high accuracy and keep
    the original as a control.
  - "16,000 points is within 0.02% of 256,000" was measured on the c168 winners. It is
    evidence, not an error guarantee.
- **Next, as a concrete work package** (revised 2026-10-04 after the second and third
  reviews, and 2026-10-05 after the c208 run):
  1. **A loss-audit corpus.** Gather the known losses: the large capped losses (flat
     seeds), the loss at the full budget (seed 234), the content cases, and any seed
     close enough to the incumbent to matter.
     - For each missed CADO/msieve winner, record the first boundary where it drops out
       and its margin to that cutoff. The boundaries are: band construction (size model
       and stopping rules), the cell cap, the block top K, the global shortlist (`cap`),
       exact size refinement (`-refine`), and the final MurphyE scoring.
     - Each case needs a reproducible explanation. This decides between wider coverage,
       better interpolation, more survivors, and a better proxy: a larger K helps one of
       those, a larger budget another.
     - The corpus is also the acceptance test for later speed work, rather than "the
       best seed still works".
     - Also check the size model's translation search, which is local (a pattern search
       started from a neighbouring line's optimum). On quadratic-rotated seeds it
       stayed 5–6 nats above a far basin that a wide search found (see item 6). On
       ordinary seeds the two agree; sample some lines to confirm none of them has a
       far basin either.
     - *c208 (2026-10-05)* adds six cases whose boundary is already known (M2, "Breadth
       run, c208"). Seeds 15, 27, 32, 103 and 151 were lost to the cell cap, and seed 222
       to the line scan. The scan in u has the same weakness as the translation search: it
       walks outwards line by line and stops on a count. 160 of 343 c208 searches hit its
       1000-line limit, and seed 32's minimum lies further out. A coarse scan of u (every
       k-th line, or the closed-form minimum over u from item 2) would find far basins
       before the fine scan.
     - *Retention cases (2026-10-06):* c208 seed 10, cell (7, −174864538), lost at the
       200-cell cut at budget 1.25; and seed 20, a cell at proxy rank 86. Both are proxy
       alpha errors (known problem 9), not coverage.
  2. **A faster size model.** Its ~75% is of summed per-process time, which is not the
     share of elapsed time a GPU port would remove: with 4 workers the size model of one
     seed overlaps the sieve of another. So measure complete searches under the intended
     concurrency.
     - At fixed translation and skew, the squared norm is an exact quadratic in (u, v).
       That gives cheap estimates, closed-form minima, band proposals and warm starts.
       `stage23_bench/tools/slice_size.py` uses it (the best integer (u, v) at fixed
       (t, s) is a 2-D closest-vector problem). It reproduces the size model's best line
       minima to 1e-3.
     - **A fixed-configuration value is an upper bound.** The size model's L(u, v) is a
       minimum over translation and skew, so any one fixed (t, s) overestimates it. A
       band built from one configuration therefore lies inside the true band: safe for
       admitting cells, unsafe for excluding them. Different configurations are optimal
       in different regions, so one global quadratic surface is not enough.
     - So use several local models (anchors), refine adaptively, and compute exact
       values near every pruning threshold. Refining only the survivors cannot recover a
       cell the model wrongly excluded.
  3. **A selective search over wider regions**, replacing the budget-lowering cap. Run
     the dense GPU kernel inside promising residue classes (u, v) = (u0, v0) + M·(i, j),
     rather than meeting the budget mainly by narrowing the norm range.
     - **c208 makes this the first priority.** msieve's winners on its best seeds sit
       1.3–2.1 nats above the best line minimum, at u = 189 to 2014. Reaching them
       densely costs 3e12–4e13 cells per seed at budget 2.25 (seed 27: about 20 min of
       sieving). At c168 size the dense search could afford its bands; at c208 size it
       cannot. A class of density 1/M costs 1/M as much.
     - **Both CPU tools already do this.** CADO's ropt stage 1 chooses sublattices by
       Hensel lifting and CRT. msieve's degree-5 sieve uses M = 2^3·3^2·5·7 = 2520 for
       lines longer than 1e5 (`find_lattice_size_y`). Its `find_hits` keeps only each
       prime power's top-scoring classes, with no exploration allowance: that is the
       class-selection boundary at its narrowest.
     - Treat 2520 as a starting point. Tune M and the number of classes kept to the
       seed's geometry and budget, and rank classes by attainable size and root
       potential, projective contributions included.
     - Reusing the kernel: a class is the derived seed f + (u0 x + v0) g rotated in steps
       of M·g. For p ∤ M the tables are the same root counts, re-indexed. For p^k ‖ M,
       levels up to k are constant within the class, and higher levels need lifting
       conditioned on the class. Content seeds are already a special case: one class mod
       d, with d divided out.
     - Class selection adds a pruning boundary to the loss audit. Keep an exploration
       allowance.
     - Validate on independently chosen seeds and windows. The CPU winners came from
       searches that favour good small-prime classes, so tests centred on them are
       biased towards the classes those searches already prefer
       (`prefilter_window.py`'s windows were).
  4. **Progressive allocation across seeds.**
     - Give every seed a modest, geographically broad first pass.
     - Then spend more where it is likely to pay. The guide is the expected improvement
       over the incumbent per extra second, with an exploration reserve, estimated from:
       - the observed size/root trade-off;
       - the accurately scored survivors;
       - the remaining opportunity and its cost.
     - Basin flatness or small-prime root density alone can overfund mediocre seeds.
     - A hard cutoff (say 1000 seeds down to 50–100) needs its own retention analysis.
     - Compare policies at equal elapsed time on several jobs, tracking the best
       accurately scored candidates found as time increases.
     - Per-seed win/loss counts stay a diagnostic. Final competitive quality is the
       objective.
     - *First test, c208 (2026-10-06):*
       - The breadth pass, then the top 16 by lattice score at cap 6e12: 27 min more, and
         the GPU ties the best poly. 3 of the 16 gained.
       - The breadth ranking was good enough: the winning seed was 2nd.
       - Two changes for the next test:
         - set each seed's cap to its next budget step (two of 16 got no new budget);
         - fix the survivor choice (known problem 9) before judging how much depth is
           worth, since retention loses cells that the budget pays to cover. *Done:*
           `-rerank` (M2, "Re-ranking by CADO's alpha").
       - *Second test (2026-10-07; M2, "The breadth pass re-ranked, and a third pass by
         budget step"):*
         - re-ranked breadth (36 min), then pass 2 on the top 16 (27 min), ties the best;
         - pass 3, the next step for the three leaders (16 min, from `-plan`'s step
           table), finds nothing better.
         - Untested so far: a policy that stops earlier, for instance pass 2 on fewer than
           16 seeds. The c208 winner was rank 1 after the re-ranked breadth pass.
  5. **GPU test sieving and a modest lattice-aware calculator, alongside M2** (ranking
     plan workstream 1, reusing the siever's basis construction and visited region).
     - Test 10–20 diverse finalists with the job's parameters, over representative
       q bands. Diverse means across seed families, skews, size/root trade-offs, and
       content-derived families.
     - Measure relations per second and per special-q. Use matched q blocks and
       repeated or interleaved runs, and estimate the uncertainty across blocks. Where
       practical, record duplicate rates and cofactor behaviour.
     - That shows whether the next gain is in coverage, geometry-aware ranking, or a
       richer smoothness model.
     - Geometry stays a hypothesis to measure, including how it changes across q bands.
     - *Measured on c208 (2026-10-06; known problem 7):* geometry is real and large. A
       skew far above (I/J)·q costs 15–20% of the yield at low q, and MurphyE does not see
       it.
       - *Done (2026-10-06):* a lattice-aware score, `cado_murphy -lattice LOGI,J` with
         `-qband`/`-qpoints`, behind `score_polys.py --lattice LOGI,J --qband
         QMIN,QMAX,NQ`.
         - It keeps MurphyE's smoothness model: ρ of the log norm plus alpha over
           log B, on both sides, with the job's bounds.
         - It averages that over the region the siever actually covers. Each special-q
           lattice is reduced at the poly's skew exactly as the user's siever reduces it
           (`qlat_build`: Gauss reduction under the skewed norm, the shorter vector on
           i). The special-q side's norm is divided by q. Every poly gets the same
           random special-q and sample points, so ratios between polys carry no sampling
           noise.
         - Nothing is fitted. It reproduces all 30 test-sieve points within 0.04 in
           yield ratio, against measured effects of up to 0.20.
         - Its absolute relations per special-q read about 1.9× the test sieve's, so
           compare ratios only.
         - `regress.py` checks it against the fixture `stage23_bench/data/c208/testsieve`.
         - Cost: about the same as job MurphyE scoring, since the skew search dominates,
           and one pass gives both numbers.
       - It was used to rescore the c208 comparison (M2, "Rerun with the lattice-aware
         score").
       - Outside this project: the siever could choose the region's aspect per
         special-q, at fixed area, to recover what high-skew polys lose at low q.
  6. **Quadratic rotation, measured; not worth running on degree 5 at this size.** The
     proposed pilot was f_w = f + w x²·g for fixed w, searched by the linear ropt.
     `slice_size.py` measures its size cost first: the best lognorm over the whole slice,
     against w = 0.
     - On 8 c168 seeds, w = ±1 costs **+5.8 to +8.3 nats**, and w = ±2 only slightly
       more. The seeds were 0, 1 (the winner's), 48 and 234, plus the four that a
       Y1·√s screen picked as cheapest (they were not: the screen does not predict
       the cost).
     - Any w ≠ 0 forces a translation of 1e8 to 2.5e9 and a skew of about 1e4, against
       1e5 to 7e6 for the seeds themselves.
     - No plausible alpha repays 6 nats. Here is how much alpha linear ropt already
       finds:
       - On 300 c168 seeds, the best cells have alpha −4.2 to −8.6 (median −6.9). The
         seeds themselves have a median of −1.7.
       - The winner is at −7.73, from its seed's −2.40.
       - To beat it, a w ≠ 0 slice would need alpha near −14.
     - These are the best points found, so they bound the cost from above. The size model
       itself reported +12.4 to +13.6, because its local translation search missed the
       far basin.
     - Revisit only for degree 6 (where CADO uses quadratic rotation) or a job where
       `slice_size.py` shows a cost under about 2 nats.
     - Large E′ work stays behind the test-sieve measurements.
  7. Choose the reference scorer for a GPU MurphyE: more points, or integration split
     at the real roots.

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
      *Qualified later:* this holds for ropt outputs of similar skew. A wider search
      finds low-skew cells that CADO's default area overrates (M2, c204), so scoring
      uses the job's parameters.
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
- **MurphyE has a 3–4% error against test sieving at the top**, the same size as
  the gains being chased. Final selection among the best candidates should be by test
  sieve (`~/code/cuda-sieve/bench/testsieve.sh`, on the GPU), and the
  GPU's job is to hand it a short, diverse list rather than a single MurphyE winner.
  - The 3–4% is a residual measured on 7 polys from one job, not a fundamental floor.
    Part of it may be reducible:
    - numerical error, since 1,000 sample points are too few (see "Review checks");
    - the idealized region against the special-q regions actually sieved;
    - the smoothness model. Bf = Bg = 2^lpb plus the job's area still leaves out the
      factor-base bounds, large-prime counts, mfb and cofactorization cost.

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
  the GPU. (Before the `fma()` fix it was 12,999 identical + 1 no worse.) That version ran
  one polynomial per thread: 19 ms per poly, 11x slower than 8 CPU threads, and effort 50
  could not run under the 60 s watchdog.
- **Per-candidate phases (constraint 4 done, 2026-10-03):** sopt is now five phases
  (prepare, lll, dedupe, descent, reduce) with CADO's order kept in dedupe and reduce;
  on the GPU each phase is a work queue run in time slices (10 s), so any effort fits
  the watchdog. Identical to CADO on everything: c146 13,000 (effort 0) and 3,000
  (effort 50), c161 11,000 and 1,000. Speed *(measured with ECM sharing the GPU; retime)*:
  effort 0 2.4 ms per poly (8 CPU threads: 3.2), effort 50 124 ms per poly (CADO on 8
  threads: ~84). lll is bound by full-width `Int<128>` operations, descent by FP64
  root finding (1/64 rate on GeForce). Next for sopt: length-aware integers; then an
  exact-decision FP32 shortcut in the descent (FP64 only for close comparisons). See
  stage23_gpu/README.md.
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
- **Throughput target:** all 1.94M polys at effort 0 in minutes. *(est.)* The earlier
  second target, effort-50 equivalent on all of them, is dropped. M0 showed effort 50
  barely moves exp_E, so its quality gain does not justify the work.
- **Speed work keeps exact CADO reproduction as a reference mode.**
  - Length-aware integers (or a few width classes with overflow retry) come before any
    redesign of the LLL.
  - An FP32 shortcut in the descent must keep every decision exact: root bracketing and
    step choices as well as the final comparison. Otherwise it no longer follows CADO's
    path, and an approximate root-finding decision is not caught by protecting only the
    last comparison.
  - A faster mode that is not bit-exact is judged by validity, candidate retention, and
    final quality under a reliable scorer. Record both sopt's internal objective and the
    printed exp_E (see "Printed exp_E is not the objective sopt minimizes" above).

### M2: GPU ropt core

- **Search:**
  - alpha sieve (p < 200, prime powers ≤ 200, matching CADO first) over the size-limited
    (u, v) band from constraint 5
  - top-K per u-line block (as built since 2026-10-04: the exact best K by the proxy, ties
    to the lower v; before that, each thread's best cell, then the best K of those)
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
  - *Proposed split (review, 2026-10-04):* these targets mix engineering completion with
    a research result. A seed's family may have nothing better than its saved winner, so
    "beats X" may never be met. Two separate criteria instead:
    - **Engineering done:** the search is bounded, reproducible and fast enough, and it
      reliably keeps every competitive candidate, with each known loss explained.
    - **Research success:** a GPU-found poly beats a pipeline winner by test sieve.
- **Then:** run the top 10–30 seeds of a real job and compare against the
  msieve + CADO union.
- **Progress (2026-10-03; GPU timings with ECM sharing the GPU, retime):**
  - *Root sieve* (`stage23_gpu/include/rsieve.h`, `tools/s23_ropt`): per u-line, one pass
    over x mod p^e gives every v's root count (constraint 7's special rule included),
    folded per prime into a table of period p^e_max <= 200; a cell is 46 shared-memory
    lookups. Exact against brute-force counting (324,054 cells, `make test`) and Python
    big-integer counting; GPU and CPU builds bit-identical. 1.2e10 cells/s.
  - *Size model* (`include/rsize.h`): lognorm after the best integer re-translation,
    per line its minimum, the band within a budget B, and knots. It reproduces the
    known optima's translations (c204 B and C to within 3). With re-translation the
    landscape is shallow in u, which is why CADO finds optima at u = 90-120.
  - *Search* (`s23_ropt -search`): sieve every band, rank by L2 lognorm + affine alpha,
    exact re-translation for the best 500, then MurphyE on the CPU
    (`stage23_bench/tools/score_polys.py`, which uses `cado_murphy`: CADO's MurphyE loop
    in-process, bit-identical to CADO's `MurphyE()`, on a fixed skew grid).
  - *c161* (B = 1.0: 4 lines, 1.8e7 cells, about 1 s): both known optima found; best new
    cell (0, -941220) ties CADO's winner (1.6301 vs 1.6295e-12).
  - *c146* (B = 2.0: 416 lines, 1.5e11 cells, 16 s + 9 s size model): msieve's cell is
    #1 and CADO-orig's u = 120 optimum #2; nothing better in the band. At B = 1.0
    msieve's winner was outside the band (lognorm +1.76 over the best line minimum, but
    an exceptional alpha): **the budget must be about 2 nats.** *Superseded:* c168 has
    competitive cells 2.1–3.14 nats above the best line minimum, so run 2 used 3.
  - ~~**Translation matters to MurphyE by 1-2%, and neither tool optimizes it for
    MurphyE.**~~ *Artifact (2026-10-04, "Review checks"):* these gains come from the
    sampling noise of MurphyE's 1,000 points.
    - With the integral done accurately, tuning the translation is worth at most 0.1%.
      For example, CADO's c161 winner: +0.06%, where the 1,000-point tuning claimed
      +1.25%.
    - Original note: `cado_murphy -trans` (pattern search on MurphyE over t): CADO's c161
      winner 1.6295 -> 1.6539e-12; msieve's cell 1.6129 -> 1.6441; our best cell
      1.6301 -> 1.6658e-12.
  - ~~*Scoring note:* MurphyE has several peaks in skew; skewopt's 1.6246e-12 for the
    c161 CADO winner is a lower peak than CADO's score finds on a fixed grid
    (1.6295e-12).~~ *Artifact:* at 64,000 sample points the skew curve has one peak, and
    skewopt's 1.6246e-12 is its maximum (1.6248e-12). The grid's 1.6295e-12 was noise.
    All comparisons here use one scorer.
  - *Proxy quality:* lognorm + affine alpha (p^e <= 200) gets the true best into its top
    ~20 but not in order (the c161 best was 17th), so survivors need MurphyE; that is
    the next GPU piece (with exact alpha to 2000 and the translation search), replacing
    the CPU scorer.
  - *c204* (B = 1.5: 869 lines, u -697..171, 8.7e12 cells, 505 s sieve at 1.7e10
    cells/s + 16 s size model): **finds B on its own as #1** by the proxy (cell (90,
    979126290), CADO's exact translation), C's cell #2 and A #4 by MurphyE; nothing in
    the band beats B. The size model's deepest basin (u near -690, line minimum 61.99
    vs 62.90 at u = 0) has no strong alpha cells.
  - **MurphyE must use the job's parameters** (constraint 9). Under CADO's default
    bounds (area 1e16, Bf 1e7) a low-skew c204 cell, (46, 1901148914), with tuned
    translation scored 5.79e-15, +11% over B; under CADO's c200 parameters (lpb 32/33,
    I = 15, qmin 5e7) it is not in the top 8, and B is best again (6.68e-9; 6.77e-9 with
    translation tuned, +1.3%). M0's "parameters barely matter" held for ropt outputs of
    similar skew; a wider search finds candidates the default area overrates. ~~With job
    parameters translation tuning is worth about 1% (c146 c145 params: msieve's winner
    +1.4%; c161 c160 params: +0.1-0.4%).~~ *Artifact:* the translation-tuned figures
    here are sampling noise (see the translation note above). The c146 winner tuned with
    1,000 points read +0.5%, and at 64,000 points it is -0.01%.
  - **Result of the band-search experiment:** on all three seeds the band search recovers
    every known optimum, and with job parameters nothing in the band beats the best of
    CADO + msieve by more than 0.4% (c161: our cell (0, -941220) 2.9593e-7 vs CADO's
    winner 2.9474e-7, both with tuned translation: a tie under MurphyE's 3-4% error).
    ("Exhaustive" in earlier notes meant every cell of the size model's bands. The
    bands come from local searches with stopping rules, so they are not proof that every
    competitive basin was covered.) ~~So depth per seed is not where the gains are;
    breadth is (the c146 #2 seed was exp_E rank 407, outside the pipeline's funnel).~~
    *Too strong:* c146 argues against a narrow funnel, but on c168 breadth to 1000 seeds
    found nothing better either. Neither result sets a general policy (see "Allocation"
    under Open questions). A seed's whole band takes about 1 s
    (c161) to 25 s (c146) on the GPU, so the next step is to run the search over the
    pipeline's 1000 seeds (`ropt_seeds.tsv.gz` has msieve's and CADO's per-seed
    results to compare) with MurphyE under job parameters on the GPU, replacing the CPU
    scorer, and a translation search in it (since found to be worth nothing).
  - *Budget:* about 2 nats over the best line minimum (c146's winner sits at +1.76).
    c204 at 2 nats is 5.9e13 cells (about an hour); per-seed cost at that size needs the
    prefilter or a faster kernel. *Superseded:* c168 needed 3 (CADO winners at +2.1 to
    +3.14).
- **Breadth run, c168 (2026-10-03/04; the user's job, poly not yet submitted).** Seeds:
  the pipeline's 1000 effort-50 re-sopt polys (`pipeline_work/resopt_sorted.txt`).
  Baselines: msieve ropt on the top 300 (both passes, best 3 per seed) and CADO ropt
  effort 5 on the top 150, all rescored by `cado_murphy` with the job's parameters
  (lpb 30/31, I 14, qmin 25M: Bf 2^31, Bg 2^30, area 2^27 x 2.5e7). Winner seed is
  exp_E rank 2 (Y1 879314489222145128657): msieve 4.5604e-8, CADO 4.5504e-8 (same cell,
  different translation).
  - *GPU sopt on the job* (idle GPU): effort 0, all 772,402 raw polys, **byte-identical
    to CADO** (772,402/772,402), 14:32 wall (1.03 ms per poly on the GPU) vs CADO's
    25.6 min on 8 threads; effort 50, top 1000, identical 1000/1000, about 65 s vs
    2.5 min.
  - *Run 1* (budget 2, proxy lognorm + alpha, top 8 per 2^20 block by alpha alone;
    stopped after 133 seeds): idle-GPU sieve 3.44e10 cells/s (5.9e13 cells in 1720 s).
    It matched CADO on most seeds but missed by 1-15% on some, for three reasons, all
    fixed for run 2:
    - **Block selection by alpha alone loses the cells near a line's minimum.** Near the
      minimum the lognorm changes a lot within a 2^20-cell block, so the 8 best-alpha
      cells of a block were not the 8 best by the proxy (seed 36: CADO's cell had a
      better proxy than our #1 but was never kept). Fix: blocks split at the knots, and
      the kernel ranks by aw * score - (L0 + dL * offset), the proxy itself.
    - **MurphyE weighs alpha about 1.3x lognorm.** Regressing log MurphyE on lognorm and
      affine alpha within seeds (20k scored cells): coefficients -0.115 and -0.146
      (ratio 1.27; 1.57 with log skew added); R^2 only 0.47, so the proxy stays a
      prefilter. Run 2 uses `-aw 1.3`. Several CADO winners sit 2.1-3.1 nats above the
      best line minimum with alpha -4.2 to -4.9; run 2 uses budget 3.
    - **Content.** When p | the sopt multiplier a and f = (linear) g (mod p), the cells
      u = u0, v = v0 (mod p) have content p, and CADO's ropt divides it out (multiplier
      a/p, lognorm lower by about log p, a different alpha at p). 31 derived seeds
      (f + (u0 x + v0) g)/p from 30 of the 1000 seeds, searched like any other seed,
      recover CADO's cells exactly (seed 75: 3.498e-8) or beat them (seed 68: +5.5%).
      `s23_ropt` now divides content out of any cell it writes.
  - *Run 2* (budget 3, `-aw 1.3`, `-maxcells 2e11` per seed, the 31 content seeds;
    timings unusable after ECM started on the GPU at 23:13): each seed's MurphyE-best cell
    is proxy rank median 1, 90% 17 (run 1: 30).
    - **No seed beats the known winner.** Best cell overall is the winner cell (CADO's
      translation); the best seed beyond msieve's 300 is rank 320 at 4.06e-8 (89%).
    - vs CADO on its 150 seeds: better 30, tie (0.1%) 106, worse 14. vs msieve on its
      300: better 152, tie 77, worse 70. The large losses (up to 15%) are flat seeds
      where the cell cap lowered the budget to 1.0-1.5 (670 of 1031 seeds had the cap
      bind, about 90 down to <= 1.25); seed 234 lost 5% at the full budget.
    - Translation tuned for MurphyE (job params): the winner cell 4.5979e-8 (from CADO's
      translation, t + 197246) vs msieve's 4.5604e-8 as submitted (+0.8%; msieve's
      translation tuned the same way: 4.5853e-8). Saved in
      `stage23_bench/data/c168/winners/`. *Artifact:* at 256,000 sample points the tuned
      poly is 4.5471e-8 and msieve's 4.5499e-8 (-0.06%). All three translations of the
      cell are a tie.
    - **Flat seeds.** Two seeds (787, 974; lognorm 54.3, every line minimum within
      0.001 of each other, 3.7e11 cells per line even at budget 0.5) ran the line scan to
      umax and built every block at the floor budget: 36 GB and 5 hours before being
      killed. Fixed: the scan stops after `-maxlines`/2 lines within budget per direction
      (default 1000 lines in total), and if the floor budget still exceeds `-maxcells` only
      the lines nearest the best line are kept, the last trimmed around its minimum. Both
      seeds now take about 2 minutes and 14 MB.
    - The size model (host code, one thread for the line scan) is now the bottleneck.
      Clean timing (2026-10-04, idle machine, bounded binary, 4 workers): seeds 0-299 plus
      their 13 content seeds in **35.6 min wall**, 3.0e13 cells; summed per process, size
      model 5,568 s vs sieve 1,869 s (internal clocks, 3-5% high on this WSL2 box). For
      comparison the pipeline's msieve ropt took 45 min for the same 300 seeds (two
      passes) and CADO's 40 min for 150 (two passes). The slowest seeds (80-95 s) are
      flat ones hitting the cell cap, nearly all of it size model. Top cells match run 2
      on 307 of 313 seeds; the 6 others are flat seeds now cut at 1000 lines (3 better,
      3 worse, all 1-15% behind msieve/CADO): flat seeds remain the weak spot.
  - **Conclusions for c168:**
    - Breadth to 1000 seeds found nothing better here; the winner was exp_E rank 2.
    - Once the three selection fixes were in, the GPU search was competitive with CADO's
      ropt on its seeds, with identifiable gaps, and ahead of msieve's on most.
    - Remaining gaps:
      - flat seeds under the cell cap (an allocation that gives flat seeds more cells,
        not fewer, or ranks seeds before spending);
      - out-of-band cells (CADO's seed 48 cell is at +3.14);
      - GPU MurphyE for the survivors.
    - Later checks ("Review checks") struck the translation gap from this list: MurphyE
      only appeared to reward tuned translations. They added the three problems listed
      under "Current status".
- **Breadth run, c208 (2026-10-05; the user's job, run while its CPU pipeline ran).**
  - *The job:* 2,198,964 deduped raw polys, degree 5. The pipeline ran `--size big`:
    CADO effort 50 on the top 2000, msieve ropt on 300 and CADO ropt on 150. The CADO
    passes were lost when the machine crashed (see "Memory guard").
  - *Scoring:* the user's planned parameters, lpbr 33, lpba 34, sieve area 2^32 per q,
    qmin 80M. That is `score_polys --params job:33,34,16.5,80e6` (Bf 2^34, Bg 2^33, area
    2^32 × 8e7) at 4,000/16,000 points.
  - The run's files were in /tmp and were lost in the crash. The numbers below are from
    the session log. The seeds can be regenerated from the pipeline's
    `pipeline_work/resopt_sorted.txt`, which the GPU's effort-50 output matched exactly.
  - *GPU sopt (M1) at this size:*
    - Effort 0 on all 2,198,964 polys took 50:59 wall (1.04 ms each; 2 redone by CADO),
      sharing the CPU with CADO's own sopt for 27 min. CADO's took about 111 min on 8
      threads.
    - **2,198,952 outputs were identical and 12 differ**, the first differences on any
      job. Each pair has the same printed exp_E, but a different translation or constant
      rotation. All 12 are no worse at full-precision exp_E (`sopt_compare.py --rescore`
      passes 100%), and none is in the top 2000.
    - Effort 50 on the top 2000 took 145 s wall and was identical 2000/2000 to the
      pipeline's CADO re-sopt, so seed ranks join exactly.
    - Multipliers are large: |a| has median 224 in the top 2000 (15,077 over the whole
      job). The top 2000 yield 47 content seeds.
  - *GPU ropt with the c168 run-3 settings* (budget 3, `-aw 1.3`, `-maxcells 2e11`, 4
    workers). Paused after 325 seeds plus their content seeds: 343 searches in about
    70 min.
    - **The cell cap sets the budget at this size.** Over 325 seeds the effective budget
      has median 1.5: 142 seeds are below 1.5, 214 below 2, and only 43 keep the full 3.
      On c168 most seeds kept 3.
    - The best GPU cell is 3.2979e-9 (seed 1), against msieve's best of 3.4104e-9
      (seed 27): −3.3%.
    - Against msieve on its 300 seeds (296 scored): better 118, tie (0.1%) 59, worse 119.
      Every large loss (ratios 0.70–0.83) is on a seed cut to the 0.5 floor. Seeds left
      at budget ≥ 2 have median ratio 1.000; those below 2, 0.992.
    - Each seed's MurphyE-best cell has proxy rank median 1, 90% 11, max 29 (32 are
      scored per seed).
    - Scoring takes about 0.9 s per poly at this size, against 0.28 s on c168.
  - *Depth pass:* `-maxcells 2e12` on the 26 best seeds by either method, plus one
    content seed; 2 workers, 22 min.
    - Budgets rose only to 1.25–2.5; seed 222 stayed at 0.5.
    - Per-seed gains reach +8.8%, and +5–6% on seeds 27, 32 and 140.
    - The best GPU cell is 3.3201e-9 (seed 10, proxy rank 29): −2.6% against msieve.
    - The GPU is still behind msieve on seeds 27 (0.936), 32 (0.939), 15 (0.953),
      151 (0.914), 103 (0.856) and 222 (0.696).
  - *Where msieve's winners are.* Rotations from `rotation_diff.py`. The lognorm is at
    msieve's own translation, so it bounds the re-translated value from above.

    | seed | msieve u | lognorm − best line minimum | deep pass: budget, u range |
    |---|---|---|---|
    | 15 | 306 | +1.93 | 1.25, −7..109 |
    | 27 | −295 | +2.13 | 1.75, −98..7 |
    | 32 | 295 | +1.31 | 1.00, −4..61 |
    | 103 | 2014 | +2.01 | 1.25, −7..113 |
    | 151 | 189 | +1.49 | 1.25, −6..91 |
    | 222 | −16152 | −0.39 | 0.50, −500..−499 |

    - **Every one is a coverage loss**: the cell was never in the sieved region; it was
      not ranked out.
    - Five lie inside a 3-nat budget that the cap lowered.
    - Seed 222's cell is *below* the size model's best line minimum. The line scan
      stopped at its 1000-line limit (|u| ≤ 500) while the minima were still falling.
  - **The 1000-line scan misses deeper minima.** 160 of the 343 searches hit the limit.
    With `-maxlines 4000`, seed 32's best line minimum is 62.94, not 64.00. A 40-seed
    rescan to count how often this happens was running when the machine crashed.
  - **Covering those cells densely costs too much.**
    - At budget 2.25, seed 27 has 4.1e13 cells (3.9e7 blocks, u −793..153), about 20 min
      of sieving. Seed 32 has 2.8e12 (u −44..1912).
    - One poor poly at budget 3 had 2e14 cells, almost all in a far, flat basin at
      |u| ≈ 500 with 1.7e11 cells per line.
    - msieve reaches these regions by sieving only good classes (mod 2520 on long lines).
  - *The pipeline's final polys* (CADO default MurphyE; CADO's passes rerun after the
    crash, plus the user's deep msieve and CADO runs):
    - CADO orig: seed 10, skew 5.0e8 after skewopt, MurphyE 1.798e-15.
    - The user's deep msieve inverted pass: also seed 10, skew 5.0e8, e 1.799e-15.
    - msieve's original winner and CADO inv's winner: both seed 27, skew 1.33e9, about
      1.767e-15.
  - **Test sieve (2026-10-06):** the user's GPU siever (`testsieve.sh`), the job's
    parameters (rlim 200M, alim 300M, lpb 33/34, mfb 64/96), 3 points at q ≈ 80M, 540M
    and 1G, two shapes of area 2^32. The table gives the yield relative to CADO orig;
    files are in `pipeline_results/testsieve_c208/` of the job.

    | poly | skew | I17×J2^15: 80M / 540M / 1G, projected | I16×J2^16: 80M / 540M / 1G, projected |
    |---|---|---|---|
    | CADO orig (seed 10) | 5.0e8 | 1 / 1 / 1, 1.000 | 1 / 1 / 1, 1.000 |
    | deep msieve (seed 10) | 5.0e8 | 0.999 / 0.996 / 0.999, 0.997 | 0.995 / 1.004 / 1.002, 1.000 |
    | msieve (seed 27) | 1.33e9 | 0.852 / 1.015 / 0.993, 0.949 | 0.801 / 0.878 / 1.042, 0.883 |
    | CADO inv (seed 27) | 1.33e9 | 0.854 / 1.015 / 0.994, 0.949 | 0.792 / 0.879 / 1.042, 0.883 |
    | msieve (seed 27), declared skew 5e8 | — | 0.852 / 0.927 / 0.942, 0.902 | 0.801 / 0.965 / 1.027, 0.924 |

    - **MurphyE overrates high-skew polys under a lattice siever.** It had seed 27's
      polys within 1.5% of seed 10's. The test sieve has them 5% behind over the band on
      the user's shape and 12% behind on the square one, nearly all of it at low q (−15%
      and −20% at 80M).
    - **The mechanism.** Every special-q lattice contains (q, 0). At skew s that vector
      has skewed length q/√s, below √q once s > q. The reduced basis is then
      unbalanced, and the fixed I×J rectangle covers a region of aspect about (I/J)·q,
      whatever the poly's skew.
      - The squarer shape (I/J = 1) is hurt more than I/J = 4 (−12% against +1.5% at
        540M, where s/q ≈ 2.5), as predicted.
      - Declaring a lower skew changes nothing at 80M: the yield is identical, because
        the lattice forces the same region. It costs 5–7% at higher q, where the
        poly's own skew is realizable.
    - Each point is one 2000-wide window (90–120 special-q), so differences of 1–3% are
      noise; the 15–20% gaps at low q are not.
    - Both seed-10 polys tie. On CADO orig the I17×J2^15 shape projects 12% more
      relations than I16×J2^16 over [80M, 1G].
  - **Conclusions for c208:**
    - GPU sopt scales: identical or no worse on all 2.2M polys, about 2× faster than
      CADO on 8 threads, and effort 50 on 2000 seeds in under 3 minutes.
    - The dense band search under a fixed cell budget does not scale. On MurphyE, the
      GPU loses 5–30% on msieve's best seeds, because the cells it needs cost 1e13–1e14
      cells per seed: they sit 1.3–2.1 nats up, or in a basin thousands of lines out.
    - *Correction (2026-10-06, test sieve):* those msieve cells all have skew 3e8–1.6e9,
      and MurphyE overrates exactly such polys at low q. *Superseded by the rerun below:*
      the lattice-aware score reorders seed 27 against seed 10, but most per-seed
      losses remain, and they are coverage losses.
    - Selective search (Next, item 3) is still needed to reach cells far from the
      minimum at this size. Whether those cells are worth reaching should be judged by
      the lattice-aware score, not MurphyE.
    - The line scan needs far more lines, or a coarse scan of u first (Next, item 1).
  - **Rerun with the lattice-aware score (2026-10-06).** The same breadth and deep passes
    were rerun, because the first run's files were lost in the crash. They reproduced it
    exactly. The rerun's breadth pass has 341 searches (325 seeds and 16 content seeds); the
    lost run's 343 is from the session log and was not checked. Every output was scored by job MurphyE and by `score_polys --lattice 17,32768
    --qband 80e6,1e9,9` (the user's siever shape). The baseline adds the user's deep
    msieve and CADO runs.
    - Best by the lattice score: CADO orig 1.0000 and the deep msieve 0.9998, both
      seed 10 at skew about 5e8, as the test sieve found. Seed 27's best drops to 0.963
      (0.989 by MurphyE). The GPU's best was 0.9668 (seed 10, deep pass), 3.3% behind
      under either score.
    - Per seed the lattice score barely moves the counts. GPU vs msieve on its 300:
      better 120, tie 61, worse 117 (MurphyE 117 / 60 / 121). GPU vs CADO on its 150:
      27 / 73 / 50 (MurphyE 19 / 72 / 59). The big losses (−25 to −30%: seeds 78, 222,
      71) are seeds whose budget the cap cut to the floor.
    - **On the winning seed the loss is coverage, and more cells recover it.**
      - CADO orig's cell and the deep msieve's are the same rotation, u = 147, 1.36 nats
        above seed 10's best line minimum. The deep pass's 2e12 cap had stopped at budget
        1.25 (u −8..122).
      - At budget 1.5 (5.0e12 cells, 146 s of sieving, shared with ECM) the GPU finds
        CADO's exact cell, (147, 4704054682), at a translation within 7 of CADO's.
      - It is the GPU's best of 200 by the lattice score, 1.0000, and proxy rank 2. Its job
        MurphyE is 3.4489e-9, against 3.4502e-9 for the best poly.
      - The sieve and the proxy work once the cell is in the band. The budget has to go
        to the right seeds (Next, item 4): a broad pass, then deeper budgets on the few
        seeds that lead.
  - **Progressive allocation, a first two-pass test (2026-10-06).** Data in
    `pipeline_results/gpu_ropt_c208/pass2/` (`run.sh`, `compare.py`, `rescore.py`).
    - *Policy:* rank the breadth pass's 341 searches by their best lattice score. Rerun the
      top 16 one at a time with `-maxcells 6e12`, otherwise as in the breadth pass (budget
      3, `-aw 1.3`, 1000 lines). Score all 200 outputs of each search, not the first 32.
    - Seed 10 was second after the breadth pass alone, so this policy picks it.
    - *Cost:* 27.4 min wall for the 16, GPU otherwise idle (one CPU-only ECM running).
      Budgets rose to 1.25–2.75, at 0.4e12–5.0e12 cells and 33–169 s a search.
    - **The GPU now ties the best poly.** On seed 10 it finds CADO orig's exact cell,
      lattice score 1.0000, ahead of the deep msieve (0.9998). It is the same poly as CADO
      orig's (same cell and skew, translation within 7), which is already test-sieved, so
      it needs no new test sieve.
    - Against pass 1 scored the same way (all 200 outputs), three searches gained: seed 10
      +1.6%, seed 19 +2.1%, seed 11 +2.7%. The other 13 are unchanged.
    - Every search's lattice-best cell lies 0.6–2.2 nats above its best line minimum
      (median 1.2). Budgets the cap cuts below about 1.25 lose cells at this size.
    - *A fixed cap wastes time.* Seeds 17 and 140 got budget 1.25 again, the same as the
      deep pass (156 s for nothing). On seed 10 each 0.25-nat step costs 3.6× the cells,
      and seed 186's step from 1.5 to 1.75 costs more than 14×. The cap should be set per
      seed, as the next budget step's cell count from `-plan`.
  - **Scoring 32 of 200 hid the GPU's own polys.**
    - All 200 breadth and deep outputs of the 16 searches were rescored
      (`pass2/rescore/`).
    - Seed 10's breadth run already held a 0.9841 poly at proxy rank 57, cell
      (7, −174864538). So the GPU's pass-1 best was 0.9841, not 0.9668: 1.6% behind, not
      3.3%.
    - The deep pass lost that cell. At budget 1.25 more cells compete for the 200 kept, and
      it fell out.
    - Seed 20's pass-2 gain (+0.6%) was a cell the deep pass had already output, at proxy
      rank 86.
  - **The proxy's alpha is the next retention loss.**
    - Over the 48 runs (16 searches × 3 passes), the accurately best cell's proxy gap to
      its search's proxy best has a median of 0.02–0.14 nats and a maximum of 0.69. Five
      exceed 0.5.
    - The 200 kept cells span only 0.9–1.0 nats of proxy, so a cell at a gap of 0.6–0.7
      survives only while the band is small.
    - The cause is alpha_s, the proxy's truncated alpha. On cell (7, −174864538) alpha_s is
      −7.38, against CADO's alpha (p ≤ 2000) of −8.31:
      - −0.39 from higher powers of the primes below 200 (the affine tables stop at
        p^e ≤ 200);
      - −0.54 from the primes 200 to 2000.
    - At `-aw 1.3` that is 1.2 nats of proxy error. On CADO's cell the error is 0.52, and
      on the breadth run's proxy best 0.19.
    - Across 50 outputs each of seeds 10 and 20, alpha_s − alpha(2000) runs from −0.24 to
      +0.72 (medians 0.16 and 0.21). That spread is as wide as the kept set.
    - *Fix to test:* keep a few thousand cells by the proxy, recompute alpha to 2000 and
      the exact lognorm on the CPU, and keep the best 200 by the corrected E for scoring.
    - Acceptance: it keeps cell (7, −174864538) on seed 10 at budgets 1.25 and 1.5, and
      seed 20's rank-86 cell, with no search's lattice-best cell lost.
  - **Re-ranking by CADO's alpha (2026-10-06): `s23_ropt -rerank N`, default 4096.**
    - The best N cells by the proxy get the exact lognorm (as `-refine` computed for 200
      before) and CADO's own alpha: `get_alpha`, p ≤ 2000, through
      `cio_alpha_rot` in `tools/cado_io.c`. They are ranked by E = lognorm + aw·alpha, and
      the best max(top, refine) are written.
    - `-rerank 0` is the old ranking; its output is byte-identical to the previous binary
      (c168 s0051, c208 seed 10).
    - `test_rsieve` checks that the alpha of each rotation equals that of the polynomial
      `s23_ropt` writes, built through `rs_rotate` and translated.
    - *Calibration:* 9,600 c208 outputs (16 searches × 3 passes × 200), each with the
      lattice score and CADO's alpha. Mean within-run Spearman correlation with the
      lattice score:

      | proxy | ρ | worst rank of each run's best |
      |---|---:|---:|
      | lognorm + 1.3 alpha_s | 0.52 | 95 |
      | lognorm + 1.0 alpha | 0.90 | 8 |
      | lognorm + 1.2 alpha | 0.91 | 8 |
      | lognorm + 1.3 alpha | 0.91 | 8 |
      | lognorm + 2.0 alpha | 0.85 | 8 |

      The weight is flat from 1.1 to 1.4, so `-aw 1.3` is kept.
    - *Acceptance:*
      - Seed 10 at the deep pass's setting (cap 2e12, budget 1.25): cell (7, −174864538) is
        rank 2 (it had dropped out of the 200), and the search's best rises from 0.9668 to
        0.9841.
      - Re-ranking 16,384 instead of 4096 changes nothing in the top 50.
      - The 16 pass-2 searches rerun with `-rerank 4096` (27.2 min, as before;
        `pipeline_results/gpu_ropt_c208/pass2r/`): every search's best is the same poly
        as pass 2 found with all 200 outputs scored.
      - The best's rank in the output is at most 7 (seed 15; seed 20's is 3, from 95), and
        rank 0 in 10 of 16. The first 20 outputs hold the best in all 16.
    - *Cost:* 0.46–0.63 s a search on 4 threads (CADO's alpha is about 0.4 ms a cell).
    - With the re-rank, scoring the first 32 outputs is enough: 6× less scoring than 200.
  - **The breadth pass re-ranked, and a third pass by budget step (2026-10-07).** Data in
    `pipeline_results/gpu_ropt_c208/`: `breadth_r/`, `pass3/`, `progressive.py`,
    `run_caps.sh`.
    - *Breadth again with `-rerank 4096`:* the same 341 searches at cap 2e11 with 4
      workers.
      - It took 36 min wall. The 2026-10-06 breadth pass took 64 min, but shared the CPU
        with the job's pipeline. The re-rank took 243 s of CPU in all.
      - Each search's best against the old breadth pass (32 outputs scored in both):
        better 20 (up to +3.4%; seed 10 +2.9%), the same 321, worse 0.
      - Seed 10 is now first (0.9841, against 0.9565). The top 16 are the same set, so
        pass 2 is `pass2r`, run with the same settings.
    - *Budget steps:* `s23_ropt -plan` now prints each budget step's cells
      ("# step B: N cells"). `-maxcells N` then gives exactly budget B, so a later pass
      can give a seed its next step. *Since 2026-10-07,* `-band B` does it directly
      (code review, item 2).
      - Checked on seed 10: step 1.75 is 1.79e13 cells over 569 lines, and the run capped
        at that count sieved exactly those.
    - *Pass 3:* the searches within 5% of the best poly after pass 2 (seeds 10, 1 and 17)
      each got their next step above pass 2's.

      | search | pass-2 budget | next step | cells | result |
      |---|---:|---:|---:|---|
      | s0010 | 1.50 | 1.75 | 1.79e13 (8.5 min) | 1.0000, the same cell |
      | s0001 | 1.75 | 2.00 | 6.39e12 (3.2 min) | 0.9582, the same cell |
      | s0017 | 1.25 | 1.50 | 8.51e12 (4.3 min) | 0.9538, the same cell |

      - None improved, in 16 min. The one new cell near the top is seed 10's
        (227, 26611975762), at 0.978.
      - Each best is at output rank 0–2.
    - **So on c208 the progressive scheme is:**
      - a breadth pass (36 min);
      - pass 2 on the 16 leaders (27 min), which ties the best poly of any tool;
      - one more step on the leaders, which finds nothing better.
    - Per-seed steps keep a later pass from re-sieving a budget already covered. From
      breadth to pass 2 every seed advances anyway, so the step table matters from pass 3
      on.

- **C181 (a backup job, 2026-10-07).** The user's earlier C181 pipeline run in
  `~/msieve-s-backup/backup_20261006_205527`, run through the current GPU ropt and
  compared. Data in `pipeline_results/gpu_ropt_c181/`.
  - *The job:*
    - 12.7M raw polys;
    - CADO sopt;
    - the top 1000 re-optimized at effort 50;
    - msieve's ropt on the best 300 (both passes), CADO's (effort 5, both passes) on the
      best 150 (the same seeds, joined on Y1).
  - *Seeds:* the same 300 (`best300_msieve.ms`), plus 16 content seeds. GPU sopt was not
    rerun: M1 is exact on every earlier job.
  - *Scoring:* the lattice-aware score and job MurphyE, with the parameters of the user's
    test-sieve job (`~/code/test-sieve/input.job`): lpbr 31, lpba 32, rlim 120M,
    alim 160M (3LP algebraic). That is `--params job:31,32,15,45e6 --lattice 15,16384
    --qband 45e6,2e8,7`.
    - The siever region (ggnfs 15e) and the q range (CADO's c180 qmin 45M up to 200M) are
      assumptions. The job file names neither, and they are not test-sieve validated
      for this job.
  - *Runs, GPU otherwise idle, CPU ECM running:*
    - breadth, as c208's (cap 2e11, `-rerank 4096`, 4 workers): 316 searches, 27 min;
    - pass 2, the top 16 by lattice score at cap 6e12, one at a time: 31 min.
    - Both ran before the content fix of known problem 9 (content cells ranked 0.3·log d
      too well). None of the five seeds compared below has a content lattice.
  - *Results* (ratios to the best poly, msieve's on seed 5, the poly the user
    test-sieved):

    | | lattice | MurphyE |
    |---|---:|---:|
    | msieve | 1.0000 (seed 5) | 1.0000 |
    | GPU, pass 2 | 0.9993 (seed 5, msieve's cell) | 0.9995 |
    | GPU, breadth | 0.9786 (seed 6) | |
    | CADO | 0.9786 (seed 6, the GPU breadth's cell) | 0.9886 |

    - msieve's winner is at budget +1.76 nats over seed 5's best line minimum. The
      breadth pass stopped at 1.75. Seed 5 ranked 3rd after breadth, and pass 2 (budget
      2.75) finds the exact cell at proxy rank 1.
    - The 0.07% left is the translation (known problem 10).
    - Per seed, GPU vs CADO on its 150: better 51, tie 83, worse 16. GPU vs msieve on its
      300 (179 that msieve output): better 65, tie 51, worse 63.
    - Pass 2 raised 5 of the 16 (seed 5 +2.6%, 8 +2.3%, 11 +1.7%, 181 +0.9%, 115 +0.7%);
      the other 11 were already at their best.
  - **Translation, not coverage.** On seeds 6, 10, 11, 0 and 5, msieve's best cell is
    among the GPU's 200 outputs, with another translation:

    | seed | cell | GPU's t | msieve's t |
    |---|---|---:|---:|
    | 6 | (−83, −1250377327) | 0.9738 | 0.9827 |
    | 10 | (−30, 220886514) | 0.9765 | 0.9804 |
    | 11 | (−82, −568152767) | 0.9205 | 0.9267 |
    | 0 | (16, −270465760) | 0.9666 | 0.9680 |
    | 5 | (48, 472598559) | 0.9993 | 1.0000 |

  - **The per-block cut, measured.** c208 seed 10 (budget 1.5) and C181 seed 5 (budget
    2.75), rerun with `-k 16`, wrote exactly the same 200 cells as with K = 8, at the
    same sieve speed.

### M3: Objectives

- E′ (David–Zimmermann) and the exact-distribution E′ (ranking plan, workstream 4).
  - Prototype it on an existing finalist pool first.
  - Build the full distribution out to large prime bounds only if it predicts test
    sieving better on held-out jobs. An "exact" distribution still carries truncation,
    discretization, geometry and smoothness-model assumptions.
- the tail-aware sieve score (constraint 8)
- lattice-aware E (ranking plan, workstream 1). A modest version can run alongside M2
  rather than wait for it: sample the regions the GPU siever actually visits (see the
  ranking plan).
- change one objective at a time against the M2 baseline
- A bounded new search dimension: quadratic rotation f + (w x² + u x + v)·g with w held
  fixed in ropt, on a few selected quintic seeds. CADO's degree-5 ropt is linear only.
  M0's pre-rotation test does not settle this: it fed f + δ x² g back through sopt, whose
  LLL can undo δ. Holding w fixed and measuring real root properties is a different
  experiment.
  - *Measured (2026-10-04):* not worth it for degree 5 at c168 size. w = ±1 costs 5.8 to
    8.3 nats of size on 8 seeds (`stage23_bench/tools/slice_size.py`; details in
    "Current status", Next item 6). Revisit for degree 6, or for a job where the size
    cost measures under about 2 nats.

### M4: Integration

- inline GPU sopt in `-nps`
- a pipeline mode that runs entirely on one device
- keep the CPU hybrid as the reference

## Review checks (2026-10-04)

An outside review of the project (GPT, 2026-10-04) raised three scoring and selection
problems and several points about direction. Each problem was re-checked here with
separate code before being accepted:
- a build of `cado_murphy` with a settable sample count;
- a Python projective root counter;
- a selection harness around `cpu_block`.

These scratch tools are not in the tree. The accepted direction points are folded into
the sections they belong to, each marked "2026-10-04".

### 1. MurphyE at 1,000 points is too noisy to tune on

**Mechanism.** CADO's `MurphyE()` integrates over the half-ellipse with K = 1000
equally spaced angles. This is the score behind `cado_murphy`, CADO's ropt and the
pipeline's final ranking.
- The integrand has a narrow peak wherever F or G has a real root: the norm goes to 0,
  so ρ goes to 1.
- A sample that lands near a root counts that peak at full weight.
- Translation moves the roots and skew moves the samples, so a search over either fits
  the noise.

**The c168 winner cell** (job parameters, each poly at its K = 1000 best skew):

| Poly | K = 1000 | K = 16,000 | K = 256,000 |
|---|---:|---:|---:|
| msieve winner | 4.5604e-8 | 4.5499e-8 | 4.5499e-8 |
| CADO winner | 4.5504e-8 | 4.5475e-8 | 4.5475e-8 |
| translation-tuned (`gpu_tuned_best.poly`) | 4.5979e-8 | 4.5468e-8 | 4.5471e-8 |

- The tuned poly's +0.8% is gone; it is 0.06% behind msieve's.
- With translation tuned at K = 16,000 instead, the three translations of this one
  cell converge to the same value (4.5499–4.5501e-8 at 64,000 points).

**Translation, re-checked on two older jobs.** Each poly's translation was tuned at
K = 1000 as before, then the tuned poly was evaluated at 64,000 points:

| Poly | Gain claimed at K = 1000 | Same poly at 64,000 points |
|---|---:|---:|
| c161 CADO winner, CADO defaults | +1.25% | +0.06% |
| c161 msieve winner, CADO defaults | +2.1% | −0.02% |
| c146 msieve winner, c145 parameters | +0.5% | −0.01% |

Tuning at K = 16,000 instead either keeps t = 0 (c161 CADO) or changes E by at most
0.07%, part of which is the better skew. Translation is worth about nothing to MurphyE
once it is integrated accurately.

**Skew.** On the c161 CADO winner (CADO defaults), MurphyE over the 2^(j/32) skew grid
has:
- 42 local maxima at K = 1000, 11 of them within 1% of the top;
- one flat peak at 64,000 points (two adjacent maxima, 0.01% apart).

skewopt's 1.6246e-12 is the true maximum (1.6248e-12). The grid scorer's 1.6295e-12 was
noise, and at the skew it picked the true value is 0.2–0.3% below the maximum.

**Size of the error.** Over 12,097 c168 polys (every baseline poly, and each seed's top 10
GPU cells, at the K = 1000 best skew), K = 1000 reads high:
- median +0.17%;
- 5–95%: +0.06% to +0.38%;
- max +1.4%.

The bias is upward because each poly's score is a maximum over a noisy skew grid. At a
random offset the 1,000-point sum would be unbiased; it is the maximum that picks out
the high readings.

**The c168 tool comparison barely moves** (better / tie / worse, tie = 0.1%):

| Comparison | K = 1000 | K = 16,000 (skew searched at 4,000) |
|---|---|---|
| GPU vs CADO, 150 seeds | 30 / 106 / 14 | 31 / 106 / 13 |
| GPU vs msieve, 300 seeds | 152 / 77 / 70 | 136 / 110 / 53 |

- Evaluating at 16,000 points at the K = 1000 skew gives nearly the same counts
  (msieve 136 / 108 / 55).
- The best five seeds are the same under every scorer. The winner seed is still first,
  for both tools and for the GPU.
- K = 4,000 is already much closer: it reads high by a median 0.05% (max 0.4%) against
  16,000.

**Consequences:**
- Compare finalists at K ≥ 16,000; 16,000 and 256,000 agree within 0.02%.
- Alternatively, integrate between the real roots of F·G, as CADO's experimental
  `MurphyE_int_cut` does (`polyselect/E.sage:54`).
- Tune nothing at K = 1000. Use it only to prune, with a margin above its noise (about
  1.5%).
- A GPU MurphyE needs this decided before it is built.
- `cado_murphy` needs a sample-count option, and the reference scorer should be
  validated by convergence (not yet done in the tree).

### 2. Projective alpha changes under linear rotation

The root sieve scores affine roots only. The projective part was taken as constant
because c5, c4 and c3 do not change under f + (u x + v)·g, but that reasoning is
incomplete.
- The projective roots are the roots y ≡ 0 (mod p) of rev f(y) = c5 + c4 y + c3 y² +
  c2 y³ + c1 y⁴ + c0 y⁵.
- When p divides c5, c4, … to high enough powers, those roots lift far enough for the
  rotated coefficients to matter: c2 (+ u Y1), c1 (+ u Y0 + v Y1) and c0 (+ v Y0).

**Evidence:**
- **c168 winner** (c5 = 37800 = 2³·3³·5²·7): projective alpha is -2.7025 at even u
  and -2.4715 at odd u, 0.231 nats, all of it from p = 2.
  - The reviewer's value is CADO's `get_alpha_projective`. An independent Hensel lift
    of rev f to 2^40 gives the same numbers to 6 decimals.
- **All 1000 c168 seeds** (u = 0..23, v = 0..3, primitive cells): the projective part
  varies on 306 seeds, by more than 0.2 nats on 188 and more than 0.3 on 104 (max 0.81).
  - On 207 seeds it also varies with v within a line (up to 0.58), so it is not a
    per-line constant.
- **Weight:** MurphyE weighs alpha about 1.3× lognorm. That makes 0.2–0.8 nats large
  next to the proxy's other errors at the top, so the sieve can mis-rank cells across
  and within lines.
- **CADO's ropt has the same blind spot:** it computes projective alpha once per input
  poly (`ropt_str.c:333`).

**Fix:** for each p dividing c5 (p < 200), add the projective root counts to that
prime's per-line table, with period p^j in v, as the affine counts already are. The exact
alpha of the survivors is already right.

### 3. The sieve keeps each thread's best cell, not the block's top K

Each of a block's 256 threads keeps only its best cell, and the block emits the best
K = 8 of those 256.
- A thread owns the cells at one offset mod 32 (in one warp slot), so all its cells
  share v mod 32 and with it their 2-adic root behaviour.
- Good cells therefore cluster in one thread, and its second-best is never kept.
- CPU and GPU agree because both make the same choice, so the GPU-vs-CPU check cannot
  catch this.

**Measured** on 256 random 2^20-cell blocks (u in -100..100, random v) of the c146,
c161, c204 and c168 winners:
- 84 blocks (33%) miss part of their true top 8. The reviewer found 29 of 64 on fixed
  windows.
- The true #1 is never lost, #2 is lost in 3 blocks, and #3–#8 in the rest.
- The returned 8th cell scores on average 0.03 below the true 8th.
- If each thread kept its best 2, the top 8 would be exact in 253 of 256 blocks; with
  its best 4, in all 256.

The end-to-end cost is not measured. It is probably small, since each seed's
MurphyE-best cell is usually the proxy's #1 (median rank 1, 90% within 17). The fix is
cheap, and a selection reference test belongs next to the root-counting tests.

### Fixes (2026-10-04)

All four fixes are in the tree. The c146/c161/c204/c168 checks below were run after the
last change.

- **Scorer sample count.** `cado_murphy -K N` sets the sample angles for the skew and
  translation search, and `-Keval N` recomputes the printed value at the chosen skew.
  `score_polys.py` takes the same as `--points` and `--eval-points`.
  - The defaults are unchanged: 1000, bit-identical to before and to CADO, per
    `-selftest`, which also passes at K = 4000.
  - `-Keval 256000` reproduces the reviewer's table exactly.
  - For close finalists use `-K 4000 -Keval 16000`.
- **Exact top K per block.** Each warp keeps its best K cells, one per lane. A cell that
  beats the warp's K-th is inserted with a warp vote and shuffles, which is rare after
  the first chunks. Thread 0 merges the 8 warp lists, and the chosen cells' scores are
  summed again in the sieve's order, so they are bit-identical. The order is key
  descending, then v ascending.
  - Correctness:
    - the CPU path equals an exact brute-force top K on 192 random blocks (short ones
      included, K = 1..16, random aw, L0, dL);
    - GPU = CPU on windows of 4 polys at K = 1, 3, 5, 8, 13, 16 and `-seg` 12, 13, 20,
      and on a c161 search (`-out` byte-identical);
    - every `-check` reports 0 brute-force differences.
  - Speed: 59 registers, no spills.
    - A first version kept each thread's best 8 in registers. It took 64 registers,
      which cost a block per SM, and ran 8% slower.
    - The warp version runs the c168 window (2.05e11 cells) at 3.5e10 cells/s on the
      idle GPU, the same as before.
  - Effect on c168: 100 seeds (every 3rd of 0–299, run-3 options) against run 3.
    - The #1 cell is the same on all 100.
    - The top-200 lists overlap a median 99% (minimum 84%; 31 identical).
    - The best MurphyE per seed (16,000 points) is the same on all 100.

    The approximation cost nothing measurable here, but the selection now matches its
    description.
- **Projective alpha in the tables** (`rsieve.h`). For each p < 200 dividing c5, the
  projective roots are counted to level emax + 3. The value depends only on
  (u mod q_max, v mod p^(emax−1)), so it is tabulated once per seed on the host
  (`rs_proj_tables`, at most a few thousand floats per prime), and a line's table adds
  its row.
  - Levels to emax + 3 capture about 98% of the full-depth variation on c168 seeds
    (median error 0.003 nats, max 0.04). Levels to p^e ≤ 200 alone would capture about
    75%.
  - The output's alpha is now `alpha_s` (affine to p^e ≤ 200 plus projective). `-noproj`
    restores the old affine-only tables, but not the old block selection.
  - Correctness:
    - `test_rsieve` checks tables against a brute force that uses the full u and v, so
      no periodicity is assumed. It enumerates while there are at most 2^10 candidates,
      then lifts the roots found. Result: 378,063 cells on 7 polys, now including the
      c168 winner, 0 mismatches.
    - It matches the Python counter, which matches CADO's `get_alpha_projective`, to
      1e-6 at the same depth (e.g. c168 winner (0, 0) 2.699366, (1, 0) 2.470123).
    - GPU = CPU with it on and off.
  - Speed: the same 3.5e10 cells/s; computing the lifts inside every block had cost 35%.
  - Effect on c168: the same 100 seeds, against the top-K-only run.
    - 30 seeds change their top-200 list (8 change their #1 cell).
    - The best MurphyE per seed (16,000 points) is better on 2 seeds and worse on none:
      - seed 216: +3.1%, now 5% ahead of msieve's;
      - seed 249: +1.4%, from 0.2% behind msieve's to 1.2% ahead.
    - These are two of the seeds with the largest projective spread. Neither is near the
      winner (2.7e-8 vs 4.55e-8).
    - The 100 seeds took 13.3 min wall with 4 workers, waiting 0 s for the GPU context on
      92 of them.
- **Content seeds in the toolchain.** `stage23_bench/tools/content_seeds.py SEED.poly ...
  [--out DIR]` writes `<stem>_d<d>.poly`. On the c168 seeds it reproduces the 31 scratch
  seeds exactly.
- **GPU start-up.** On this WSL2 box, creating the CUDA context took 10–13 s on the
  afternoon of 2026-10-04, against about 1.5 s that morning. Even a bare
  `cudaFree(0)` program took that long.
  - `s23_ropt`'s sieve time included it, which is why window timings swung 8–18 s for
    the same 5.9 s kernel.
  - `s23_ropt` now creates the context on a background thread while it reads the seed
    and runs the size model, and prints "waited X s for the GPU context". In searches
    the wait is 0.

### Second outside review (2026-10-04)

GPT re-checked the fixes on the CPU. Its checks:
- the host suite and 13,000/13,000 CPU acceptance;
- its own exhaustive top-K comparison on 280 blocks of 7 polys, with 0 mismatches;
- the scorer at 16,000 points against CADO, and searching skew at 4,000, 16,000 and
  64,000 points, which still leaves the tuned c168 poly no better;
- the 31 c168 content seeds.

It raised four points, all taken:
- **"Exact" is not "no loss".** The top K is exact by the proxy only. The status section
  says so, and the next work package audits every pruning boundary with its margin.
- **Accurate scoring should be the normal path.** `score_polys.py` now defaults to
  4,000/16,000 points, with `--cado` for CADO's numbers. `cado_murphy` keeps CADO's
  default.
- **The evidence should be durable tests**, now part of `make test`:
  - `test_rselect`: the block choice against an exhaustive sort on 168 blocks, including
    25 decided by the tie rule. A reversed tie rule fails it.
  - `test_rsieve`: projective alpha against CADO's `get_alpha_projective` at 7
    rotations per poly (worst 0.005, differences within 0.002), including the c168
    counterexample.
  - `stage23_bench/tools/regress.py`:
    - scorer self-tests at K = 1000 and 4000, with the pinned K = 1000 values;
    - the c168 quadrature trap: +0.82% at 1,000 points, −0.04% at 4,000/16,000;
    - the content-seed cases below.
- **A content-seed bug.** `content_seeds.py` skipped any d with Y1 not invertible mod d.
  GPT's example is N = 77, g = 2x + 1, f = 2x⁵ + 2x⁴ + 2x³ + 5: Res = 2N, and
  (f + g)/2 is a valid seed. The tool now solves the congruences per prime power with
  whichever of Y1, Y0 is a unit there (g is primitive), then combines them by CRT.
  - It finds that seed.
  - It reproduces the 31 c168 seeds exactly; GPT also found no other c168 families.
  - Five c168 seeds and their expected derived seeds are fixtures in
    `stage23_bench/data/c168/content/`.

### Third review (GPT and Gemini, 2026-10-04)

A joint note on the next objective: better candidate retention and better final
polynomial quality per unit of elapsed time. Most of it restated the work package already
written after the second review. Additions accepted (now in "Current status", Next):
- **The fixed-configuration size model is an upper bound**, so it can safely admit cells
  but not exclude them. Use anchors, refine adaptively, and check exactly near
  thresholds; refining survivors cannot recover what the model dropped.
- **Selective residue-class search** is Gemini's strongest recommendation. Checked
  against the CPU tools:
  - 2520 is exactly msieve's degree-5 lattice modulus (`find_lattice_size_y`);
  - msieve keeps only the top-scoring classes per prime power (`find_hits`), with no
    exploration allowance;
  - CADO's ropt stage 1 also searches sublattices.
  Class selection is a new pruning boundary with a validation bias, and the kernel can
  be reused per class.
- **Allocation by expected improvement per second**, with an exploration reserve. Hard
  seed cutoffs need retention analysis.
- **Test-sieve protocol details:** 10–20 diverse finalists, matched q blocks,
  repeated or interleaved runs, duplicates and cofactor behaviour.
- **A bounded quadratic-rotation pilot.** Its first step was measuring the size cost, and
  that settled it for c168 (Next item 6):
  - w = ±1 costs 5.8–8.3 nats on 8 seeds, against linear ropt's alpha gains of about 5.
  - The pilot is dropped for degree 5 at this size.
  - The tool that measured it, `stage23_bench/tools/slice_size.py`, also prototypes the
    closed-form (u, v) minimum for the size-model work.
  - The same check showed that the size model's local translation search can miss a far
    basin. That was by 5–6 nats on those slices, and on ordinary seeds it agrees to
    1e-3. It is now an item for the loss audit.

### Code review of the fixes (2026-10-04)

A code review of the working tree found no error in the core math. It checked the sopt
phase split against CADO's order, the table rules, the projective periodicity and
overflow bounds, the warp top-K against the CPU path, and `cado_murphy -selftest`.

**Fixed:**
- **`s23_sopt` slices.** Late-starting blocks of a lower-occupancy phase each ran one
  chunk past the deadline. A warp's first chunk now ignores the deadline only while its
  launch has taken nothing yet.
- **Stale candidate slots.** `prepare` now marks its slots `CAND_NEW`, so a slot whose lll
  never ran fails its poly (redone by CADO) instead of reading the previous batch's state.
- **Candidate-slot sizing.** The GPU's candidate slots are capped at 40% of free VRAM as
  well as 1.5 GB, and the CPU build no longer shrinks its batch for them.
- **Poly reader.** `score_polys.py` crashed on lines such as `type: gnfs`. It now reads
  through a shared `polyfmt.read_cado_blocks`.
- **`-maxcells` trim.** It could sieve maxcells + 1 cells.
- **`-out` count.** It printed the wrong count when a translation was skipped, and the
  documented count is now max(-top, -refine).
- **GPU context start.** It now begins only after the seed is read and valid.

**Retested:**
- `make accept-gpu`: 13,000/13,000 at effort 0 and 3,000/3,000 at effort 50 (top set).
- `make accept-cpu`: 13,000/13,000.
- A 1e-12 s slice: identical output.
- A trimmed c161 search: GPU = CPU, `-out` byte-identical, 0 brute-force differences.

**Open (not done):**
- *Size model:* each budget step recomputes every band edge from scratch, though the new
  band lies inside the old one.
- *Narrow bands:* they are split at all 64 knots, so tiny blocks each rebuild 46 tables.
- *sopt descent list:* it is rebuilt on the host from a strided state copy.
- *Content seeds:* `s23_ropt` could find content lattices itself, or at least warn,
  instead of relying on `content_seeds.py`.
- *Duplicated helpers:* `CUDA_CHECK` and the xorshift rng.
- *Test data:* `make test` now reads `stage23_bench/data/c168/`, so that directory must
  be committed.

### Memory guard (2026-10-05)

During the c208 run, nine `s23_ropt` processes held 107.5 GB of RAM and swap, 6–16 GB
each by the kernel's OOM report. Eight were `-plan -maxlines 4000` runs started in
parallel; the ninth was a search with the cell cap lifted. The OOM killer fired, WSL
thrashed and was then powered off, and the user's CADO ropt passes were lost.
- **Cause.** `s23_ropt` built the whole list of sieve blocks before sieving: 32 bytes
  each, about 44 with the vector's growth. `-plan` built it too, only to print its length.
  - Without `-maxcells` nothing bounds the list. c208 seeds with a far, flat basin need
    1e8–2e8 blocks at budget 2–3.
  - The c168 flat seeds that reached 36 GB ("Breadth run, c168") were the same mechanism.
- **First fix (2026-10-05):** the blocks were counted first. `-plan` only counted them,
  and a run needing more than `-maxmem` MB (default 2048) of blocks stopped with exit
  status 3.
- **Final fix (2026-10-06, after the code review below): the blocks are streamed.**
  - The kept lines are cut into segments, the pieces between knots (at most 64 per
    line). A generator splits them into blocks, in the same order as before, a launch's
    worth at a time.
  - Memory no longer grows with the search. Nothing else does either: the size model
    keeps 64 knots per line, and the hits are trimmed per launch.
  - So `-maxmem` is gone. Its refusal would have silently dropped the far, flat seeds
    from an uncapped batch, since each such seed would just exit 3.
  - `-plan` only counts, in window mode too.
  - After the sieve, the number of blocks generated must equal the count from
    `seg_blocks`; a mismatch is an internal error.
- **Verified, first fix (2026-10-06):**
  - `make test` passes.
  - Against the previous binary (built from HEAD), stdout (less timings and the output
    path) and `-out` files are byte-identical on three c168 seeds with run-3 settings,
    one with the cap binding (seed 51, budget lowered to 2.75, lines trimmed). The same
    holds for a window box and an uncapped search at budget 1.
  - c208 seed 32 at budget 2, under an 8 GB `ulimit -v`:
    - `-plan` reports 344,781,297 blocks (10.5 GB) using 7 MB, where it used to hit
      `bad_alloc`;
    - the search exits with status 3 and the message, using 6 MB.
  - c208 seed 1 at budget 2: the same block count as the old binary (6,104,637).
- **Verified, streaming:**
  - Against the first fix's binary, stdout and `-out` are byte-identical on eight cases:
    - the five above;
    - the CPU path;
    - `-seg 24`;
    - the c208 seed-10 deep search, where `-check 2000` found 0 brute-force differences.
  - A window of 2.1M blocks peaks at 185 MB, against 248 MB before. The difference is
    the old list's 64 MB; the rest is CUDA and the launch buffers.
- **Operating rule.** Try any new setting with a single `-plan` first, and keep the
  total of concurrent runs well under the free RAM. Keep long runs' files outside /tmp,
  which a WSL restart wipes.

### Code review of the c208 changes (2026-10-06)

A `/code-review` of the memory guard, the `-launch` change and the lattice-aware score
found 15 issues. All were confirmed and all are fixed:
- **Blocks streamed** (see "Memory guard"). This also removes `-maxmem`'s silent exit 3
  in batches, and the count-vs-build duplication: one `seg_blocks` plus a runtime check.
- **Launch sizing.**
  - Launches were sized from the longest block, so a large `-seg` left most SMs idle
    (`-seg 30`: 4 blocks per launch). Many short knot blocks also made launches carry far
    fewer cells than intended.
  - A launch now takes blocks until it holds about 2^LOG2 cells, at least four waves'
    worth (sms × resident blocks per SM × 4; on the 5070, 48 × 4 × 4 = 768 blocks), and
    at most 65,536.
  - The kept set is now chosen by a total order (key, u, v), so results cannot depend on
    how hits arrive in launches.
  - With huge blocks a kernel still takes long: one block is one CUDA block.
  - Throughput on an idle GPU (2026-10-06, old = git HEAD, interleaved, cells/s):
    - c168 seed 51 at `-seg 20`: 3.54e10 against 3.53e10 (three runs each). No
      difference.
    - c208 seed 10 deep search (1.26e12 cells): 3.54e10 against 3.48e10 (−1.7%).
      `-launch 36` (the old size) gives 3.53e10, so streaming costs nothing; the short
      launches do.
    - `-seg 26`: 3.73e10 against 3.63e10 (−3%).
    - The cost is each launch's partly empty last wave and the synchronous gap between
      launches. Two CUDA streams (double buffering) would hide both while keeping the
      kernels short. That is about 0.5% of a whole search, where the size model
      dominates, so it is not done.
  - (While ECM shared the GPU, `-seg 26` had read 5% lower; that was mostly contention.)
- **`cado_murphy -lattice`.**
  - `-useskew` now warns when polys have no skew line. Every `s23_ropt -out` file has
    none, and those polys get MurphyE's best skew.
  - ρ's argument is clamped at 0, so a norm below e^-alpha counts as smooth (CADO's
    `dickman_rho` returns 0 below 0).
  - q is limited to 2^52, where the reduction's doubles stay exact.
  - The band comment now says 1/ln q special-q per unit q.
  - MurphyE is not recomputed when Keval equals K.
  - The header documents the lattice mode and its output columns.
- **`score_polys.py`:** one helper runs cado_murphy for both modes, and passes on its
  warnings. The accuracy statement now gives "24 ratios within 0.04 (worst 0.038)".
- **`regress.py`:** a missing c208 fixture fails the check instead of crashing, and
  cado_murphy's errors appear in the detail. The bounds come from `rescore.derived`, and
  the docstring counts 24 ratios.

### Code review of the re-rank and budget steps (2026-10-07)

The user's `/code-review xhigh` of `-rerank`, `cio_alpha_rot`, the `-plan` step table and the
new test found 13 issues. All were addressed:

1. **Content cells were ranked too well.** The re-rank scored f_{u,v} with its content d
   left in, but wrote it divided out: lognorm log d higher, alpha log d lower (exactly, by
   CADO's get_alpha), so Ea was (aw − 1)·log d too low.
   - *Fixed:* `cio_alpha_rot` divides the content out and returns its log, and
     `rs_exact_cell` (new `include/rhost.h`) takes it off the lognorm. Both values are now
     those of the polynomial written.
   - On c168 s0051 (d = 2), content cells in the 200 written fell from 73 to 40, and
     every other cell kept its order.
2. **Choosing a budget step by an exact `-maxcells` count was fragile.** *Fixed:* `-band B`
   sieves the bands at B while the line scan still runs to `-budget`. Checked on c208
   seed 10: `-band 1.75` and `-band 1.5` give the step table's 1.79e13 and 4.96e12 cells.
   `run_caps.sh` and `progressive.py` use it.
3. **The re-rank fixes only the final cut.** Each block's best K are still chosen by
   alpha_s.
   - *Measured:* `-k 16` on c208 seed 10 (budget 1.5) and C181 seed 5 (budget 2.75) wrote
     the same 200 cells as K = 8, at the same sieve speed. K stays 8.
   - The header comment states the limit.
4. **Negative `-top`/`-refine` wrapped the cap and disabled pruning.** *Fixed:* `-top ≥ 1`,
   `-refine ≥ 0` and `-rerank` from 0 to 2^24 are checked, with their own message, and
   the cap is computed in int64.
5. **The new test rebuilt the rotation instead of testing what `s23_ropt` writes.**
   - *Fixed:* `translated_poly` and `seed_from_poly` moved to `include/rhost.h`
     (`rs_written_poly`, `rs_seed_and_size`, with `RS_LP`).
   - `test_rsieve` compares `rs_exact_cell` against the written polynomial's alpha and
     lognorm at 56 cells a polynomial. `make test` adds two c168 seeds with content
     lattices (s0051, s0068): 17 content cells, and the test fails if none is checked.
   - With the fix removed, the test reports a mismatch of 0.693 (log 2).
6. **A NaN difference passed the test** (`std::max` drops NaN). *Fixed:* `!(x < tol)`, and
   a NaN sets the worst value to infinity.
7. **`-plan`'s size-model time included the step table.** *Fixed:* the table is computed
   after the timed part and the line list; it rewrites the bands, so it comes last.
8. **The alpha bound was hard-coded to 2000.** *Fixed:* `cio_alpha_bound()` returns CADO's
   `get_alpha_bound()`, as `cado_murphy` uses.
9. **The written key changed meaning under the same name.** *Fixed:* with `-rerank` the key
   is labelled `Ea`; `-rerank 0` writes `E` as before.
10. **The step table duplicated the band loop.** *Fixed:* one helper, `bands_at`, serves
    the `-maxcells` loop and the table.
11. **Three near-identical printf branches.** *Fixed:* the shared fields are printed once,
    with `alpha` appended under `-rerank`.
12. **The `-rerank < 0` error did not name the option, and `atoi` took "1e4" as 1.**
    *Fixed:* strict parsing and a message naming each option.
13. **"The same 341 searches" against 343 elsewhere.** Both breadth passes run since the
    crash have 341 searches (325 seeds and 16 content seeds). 343 is the lost 2026-10-05
    run's count, from the session log. The policy text now says 341.

Verified:
- `make test` passes;
- `-rerank 0` output is byte-identical to the binary before `-rerank` on c168 s0051 (a
  content seed) and c208 seed 10;
- `-band` reproduces the step table;
- the C181 and c208 seeds compared have no content lattice, so their results stand.

### Other points accepted

- **Status summary.** The chronological conclusions needed one current summary; see
  "Current status".
- **Corrected claims**, each now marked where it was written:
  - "not ports of CADO" (Goal);
  - "the budget must be about 2 nats";
  - "parameters barely matter" (M0);
  - "depth is not where the gains are; breadth is";
  - the FP64-volume note (Decided: Precision);
  - "exhaustive" for the band search;
  - the 1–2% translation gains.
- **MurphyE's 3–4% error** is a residual measured on 7 polys, not a floor (M0).
- **M2's done criteria** (proposed split under M2).
- **Selective search, allocation, diverse shortlist, test-sieve protocol** (Open
  questions).
- **Sopt speed work:** exact CADO mode kept as the reference, FP32 decisions exact,
  effort 50 on every poly dropped (M1).
- **Size model, the next performance target** (~75% of ropt time):
  - At fixed translation and skew, the squared L2 norm is a quadratic in (u, v). That can
    give cheap minima, band edges and starting points before exact refinement.
  - Batch seeds and knots on the GPU, and reuse work between neighbouring points.
  - Place knots adaptively, and check the interpolation error near selection thresholds:
    an underestimated lognorm wastes work, and an overestimated one can drop the winner.
- **Content seeds** changed the c168 results (seed 68 +5.5%). They are generated by a
  scratch script (`content_seeds.py`), and their generation belongs in the toolchain.
- **New dimensions and objectives:** quadratic rotation with w held fixed, and E′ on a
  finalist pool first (M3). (Quadratic rotation since measured: not for degree 5 at this
  size; see "Third review".)
- **Lattice geometry earlier** (ranking plan, workstream 1).

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
  - *Added 2026-10-04 (review):*
    - **Keep the shortlist diverse.** A single score eliminates alternatives before the
      final comparison. Keep representatives across seed families, skews, size/root
      trade-offs and content-derived families. A few objective weightings or bounded
      quotas are easier to control than an unrestricted Pareto front.
    - **How to test-sieve the finalists.** Use the job's parameters and measure both
      relations per second and relations per special-q, over several representative
      q bands. For close differences, alternate candidate and baseline runs and
      estimate the uncertainty over q blocks. Later, also watch duplicate rates and
      large-prime distributions: the fastest raw producer need not minimize the rest of
      the factorization.
    - **Compare finalists at 16,000 or more sample points.** At CADO's 1,000 points,
      MurphyE differences under about 1% are noise ("Review checks").
- **ropt budget numbers:** cells per seed and seeds per tier, once a GPU sieve kernel's
  throughput is measured. The exhaustive run on a benchmark seed (below) is the first M2
  experiment.
  - **Allocation (added 2026-10-04).** The c168 cell cap bound on 670 of 1031 seeds, so
    capping is the normal regime, not a safeguard. Equal cells per seed, enforced by
    lowering the budget, systematically drops cells whose alpha makes up for a higher
    norm. A flat size landscape is also an opportunity: many rotations of similar size
    means more chances of an unusually good alpha. Proposal:
    - give every seed a modest, geographically broad first pass;
    - spend more on a seed according to its observed root-score tail, its size
      landscape, its best fully scored cells and the estimated cost, roughly the chance
      of beating the incumbent per extra second;
    - compare allocation policies at equal elapsed time, not at equal seed counts.
    - *Added (third review):* keep an exploration reserve. Flatness or root density
      alone can overfund mediocre seeds, and a hard seed cutoff needs its own retention
      analysis (see "Current status", Next item 4).

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
  double. ~~Their volume is small enough that FP64's 1/64 rate does not matter.~~
  *Corrected (profile, 2026-10-03):* the descent's FP64 root finding is the largest
  phase on the GPU (c146 effort 50: descent 202 s of 371 s), so FP64 rate does matter.
  See the FP32 caution under M1's throughput target. The ropt
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
  - *Direction (2026-10-04, review):* this is the strongest algorithmic lever, because
    the dense search grows fast (c204: 8.7e12 cells at budget 1.5, 5.9e13 at 2), much
    faster than kernel speedups can recover. It is CADO's sublattice idea (the two-stage
    method in Bai–Brent–Thomé): search the classes (u, v) = (u0, v0) + M·(i, j) of good
    small-prime behaviour over a *wider* region, instead of every cell of a band
    narrowed by the cap. That keeps coverage while cutting cells.
    - Validate class selection on independently chosen seeds and windows before relying
      on it, keep an exploration allowance, and measure what survives the whole scoring
      pipeline, not just the window's best cell.
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
