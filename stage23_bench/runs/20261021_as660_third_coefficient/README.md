> Historical run record moved from the main notes on 2026-10-22. See the [current conclusions](../../../POLYSELECT_OPTIMIZATION_NOTES.md) for subsequent evidence and revised priorities. Statements about a running or stopped process describe the time of this record.

## Third complete C196 coefficient (2026-10-21)

The user completed a_d=415800 and the worker exited cleanly. Analysis artifacts
are in `stage23_bench/runs/20261021_as660_third_coefficient/`. The original run
header plus the exact third-coefficient suffix, including its coefficient and
worker endings, were extracted without synthetic records; `snapshot.json`
records the source byte ranges and hashes. Production inputs, outputs, and
executable were not changed, and no GPU work was launched.

The complete streaming audit independently reconstructed all 309,570 collisions
and joined all 309,570 raw outputs with no unmatched records or incomplete
markers. It covered 3,440,522 batches and 199,550,235 q roots. There were no
saturation or hash-cap events; 3,779 match-arena fallbacks used the existing
fallback path. The corpus contains 296,586 distinct candidate identities and
12,984 repeat discoveries (4.19%). Distinct yield is 1,486.27 per million q
roots, 2.46% above a_d=277200 and 7.47% above a_d=138600. This normalization is
not a GPU throughput comparison because entries per q and launch shapes differ.

Coefficient duration was 19,677.32 s RAW and 19,677.28 s wall time. It searched
84.94% more q roots than the second coefficient; the larger workload explains
much of the longer duration. All returned collisions again passed the admission
gate. Across the three coefficients there are 656,516 output rows and 633,097
distinct `(N,a_d,P,m)` identities; this identity count does not claim all later
optimized polynomials are inequivalent.

The same 10,000-lowest-candidate-hash sampling method and previously frozen CPU
CADO sopt binary were used at effort 0, one low-priority thread. All 10,000
optimized outputs joined exactly and passed the integer resultant check. The
original first/second sample results were reused, not rerun or resampled.
`quality-manifest.json` records the binary/input hashes; `comparison.json`
contains the combined three-coefficient results and bootstrap intervals.

| Measure | a_d=138600 | a_d=277200 | a_d=415800 |
|---|---:|---:|---:|
| Distinct candidates / million q roots | 1,383.03 | 1,450.57 | 1,486.27 |
| Sample mean sopt exp_E | 58.59247 | 58.59795 | 58.64309 |
| Sample median sopt exp_E | 58.27 | 58.28 | 58.32 |
| Sample 10th percentile sopt exp_E | 56.95 | 57.00 | 57.04 |
| Sample 1st percentile sopt exp_E | 55.86 | 55.98 | 55.96 |
| Best sampled sopt exp_E | 53.34 | 54.41 | 53.74 |

Lower exp_E is better. Third-minus-first mean difference is +0.05061 (descriptive
95% bootstrap interval +0.01103 to +0.08585); third-minus-second is +0.04514
(+0.00260 to +0.08195). The third's 1st-percentile difference versus the second is
-0.02 (-0.12 to +0.08). Thus higher candidate yield is accompanied by a small
average sopt-quality penalty in this sample, while its good tail is close to
the second's. This is not a ranking by final Murphy E, and the uncorrected
exploratory intervals and isolated sample minima do not justify choosing a
leading-coefficient policy from three related coefficients.

For the third coefficient, median exp_E by increasing q quarter is
58.38, 58.35, 58.26, 58.29; by discovery quarter it is
58.35, 58.38, 58.28, 58.31. The q tables omit 113 sampled identities with
discoveries spanning q quarters (overall results retain them); no sampled
identity spans discovery quarters. There is still no consistent deterioration
with increasing q or later discovery across the three samples. This does not
exclude differences in the extreme tail or establish an early stopping rule.
The largest returned gate ratio is 0.00136953, again far below the cutoff.

The next quality step is matched root optimization with preserved candidate
identity and common scoring parameters, to test whether these small sopt
differences persist in final quality. No change to the reference search policy
is supported yet. All analysis processes for this third-coefficient check have
finished; no GPU profiling or tuning was performed during this check.
