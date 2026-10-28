> Historical run record moved from the main notes on 2026-10-22. See the [current conclusions](../../../POLYSELECT_OPTIMIZATION_NOTES.md) for subsequent evidence and revised priorities. Statements about a running or stopped process describe the time of this record.

Evidence was compacted on October 28: all six score tables remain available as
lossless `.jsonl.gz` files, while full sample ledgers are in the local archive.
Updated readers resolve both forms. See the [evidence index](../../data/README.md)
for hashes, representative fixtures and recovery instructions.

## Six complete C196 coefficients (2026-10-22)

The user's next AS660 run completed a_d=554400, 693000, and 831600 and ended
cleanly. Artifacts are in `stage23_bench/runs/20261022_as660_six_coefficients/`.
Each per-coefficient snapshot preserves the original run header and exact source
section, with hashes and half-open source byte ranges in `snapshot.json`. No
synthetic records were added. The first two extracted sections necessarily lack
`worker_end`; their coefficient endings are complete, and only those expected
missing-worker diagnostics are accepted. The final section contains the clean
worker ending. All original production files are preserved.

| Leading coefficient | Returned/raw outputs | Distinct candidates | q roots | Match-arena fallbacks |
|---|---:|---:|---:|---:|
| 554400 | 151,003 | 147,028 | 102,004,491 | 683 |
| 693000 | 230,703 | 218,988 | 136,534,031 | 3,602 |
| 831600 | 199,223 | 191,004 | 125,196,933 | 3,857 |

All 580,929 new collisions were independently reconstructed and joined exactly
to raw outputs. No saturation, hash caps, unmatched raw records, or unexpected
incomplete markers occurred. Match-arena fallbacks use the existing fallback
path and do not censor output. All returned collisions again passed the gate.
Across the six coefficients there are 1,237,445 output rows and 1,190,117
distinct `(N,a_d,P,m)` identities. This does not assert inequivalence of later
optimized polynomials. These remain related multiples of 138600, not a broad
test of leading-coefficient smoothness policies.

The matched CPU analysis selected the same 10,000-lowest-SHA-256 identities per
new coefficient, deduplicated before sampling, and used the same frozen CADO
sopt binary at effort 0 with one low-priority thread. All 30,000 new optimized
outputs joined to their printed inputs and passed exact resultant checks. The
previous 30,000 sample results were reused without resampling. Each directory
records binary/input hashes, raw and optimized samples, scores, and audit
results. The combined `comparison.json` contains all six coefficients.

| Leading coefficient | Distinct candidates / million q roots | Sopt median | Sopt 1st percentile | Best sampled sopt score |
|---|---:|---:|---:|---:|
| 138600 | 1,383.03 | 58.27 | 55.86 | 53.34 |
| 277200 | 1,450.57 | 58.28 | 55.98 | 54.41 |
| 415800 | 1,486.27 | 58.32 | 55.96 | 53.74 |
| 554400 | 1,441.39 | 58.34 | 56.0199 | 53.93 |
| 693000 | 1,603.91 | 58.41 | 56.09 | 54.69 |
| 831600 | 1,525.63 | 58.38 | 56.13 | 53.00 |

Lower exp_E is better. The new coefficients have modestly worse typical and
lower-tail sopt scores than the first. Their mean-score differences relative to
138600 are +0.04683, +0.09759, and +0.05869; descriptive 95% bootstrap intervals
are [+0.00815,+0.08702], [+0.05867,+0.13499], and [+0.02267,+0.09710]. These
400-resample comparisons have no multiplicity correction. The largest
coefficient nevertheless supplies the best individual sampled score, 53.00:
distribution summaries are not a rule that a larger coefficient cannot win.
These are still sopt scores, not final root-optimized Murphy E.

`compare_position.py` adds a within-coefficient comparison using the 60,000
saved scores. For each coefficient, compare the last and first quarters, then
average the six mean-score differences with equal coefficient weight. Resample
candidates independently within coefficient/quarter strata for 400 bootstrap
replicates. Identities discovered in multiple quarters are excluded from those
quarter comparisons only. There are 61, 84, 113, 65, 128, and 95 such q-quarter
ambiguities; no sampled identity spans chronology quarters.

- **Q magnitude:** largest-quarter minus smallest-quarter mean exp_E is
  -0.05775, with descriptive 95% interval [-0.08770,-0.02911]. The individual
  differences are -0.09436, -0.04878, -0.08232, -0.02116, -0.04842, and -0.05144.
  All six favor the larger-q quarter on average. Thus the data now suggest a
  small association in the opposite direction to smaller q being better. This
  does not establish causation, behavior of the extreme tail, or a benefit to
  changing q coverage. Q factorization/root populations can differ by magnitude.
- **Discovery chronology:** last-quarter minus first-quarter mean exp_E is
  +0.00425, with interval [-0.02394,+0.03941]. Individual signs are mixed. There
  is no detected average-quality decay across batch traversal. This does not
  eliminate ordinary diminishing improvements to a running best as more draws
  are taken, nor does it establish equality of final-quality extreme tails.

Those intervals describe candidate-sampling uncertainty for these six fixed
coefficients, not variation over future coefficients. Results and per-group
sample counts are in `position-comparison.json`.

To combine candidate quantity with preliminary quality, that file also reports
estimated distinct candidates at or below two common sopt thresholds per million
searched q roots. Multiply the uniform distinct-candidate sample fraction by
the full-corpus distinct-candidate density. Thresholds 56.0 and 57.0 were chosen
after seeing preliminary summaries, so this is an exploratory check, not a
confirmatory selection rule. The cutoff comparison is inclusive and uses the
printed exp_E precision.

| Leading coefficient | Sample count exp_E <= 56.0 / 10,000 | Estimated qualifying candidates / million q roots | Approx. 95% interval | Estimate at exp_E <= 57.0 |
|---|---:|---:|---|---:|
| 138600 | 141 | 19.50 | 16.55–22.96 | 153.79 |
| 277200 | 110 | 15.96 | 13.25–19.21 | 147.09 |
| 415800 | 109 | 16.20 | 13.44–19.52 | 138.52 |
| 554400 | 99 | 14.27 | 11.73–17.35 | 134.77 |
| 693000 | 87 | 13.95 | 11.32–17.19 | 132.00 |
| 831600 | 73 | 11.14 | 8.86–13.99 | 129.83 |

Intervals are Wilson binomial approximations scaled by measured candidate
density; finite-population correction is omitted. Higher raw candidate density
does not offset the observed sopt-quality shift at these thresholds. This is
not a wall-time productivity measure: entries per q, launch shapes, contention,
and first-run clock differences prevent treating q counts as identical cost.
The largest gate ratio across all six remains 0.00136953, so these runs still
do not supply a rejected population for a fixed-geometry admission sweep.

Next prioritize matched root optimization and common final scoring parameters,
preserving identities and representing both q ranges and coefficients. The
first versus sixth coefficient is a useful contrast, but do not select inputs
only from one sopt tail and then claim an unbiased final-quality distribution.
Keep the reference search policy pending that check; do not discard larger q
or infer that later discoveries are intrinsically worse. All analysis processes
have finished; no production changes or GPU work were performed.
