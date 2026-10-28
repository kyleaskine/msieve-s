# AS660 completed CPU pipeline review — 2026-10-23

The full checkpoint and top-1000 list were moved to the local archive on
October 28. `analyze.py` resolves the archived checkpoint automatically.
Summaries, winner proofs and the winning stage1 record remain here; see the
[evidence index](../../data/README.md) for hashes and recovery instructions.

Subsequent archival work completed the independent collision audits for 180180
and 360360 and preserved compact log2 q exposure for all eight coefficients.
See [the archive record](../20261023_stage1_archive/README.md); the pending-audit
statements below describe the state at this earlier CPU review.

The user completed CPU processing and is running GPU sopt/ropt separately.
This review reads the frozen CPU reference under
`stage23_gpu/runs/20261022_three_jobs/c196/cpu_reference/` and the completed
`pipeline_work/resopt_output.txt`. No GPU work or production-file changes were
performed. `analyze.py`, `summary.json`, and `root-results.json` reproduce the
analysis; the archived `corpus-checkpoint.json` retains selected-seed state so the
full output need not be rescanned for origin joins.

## Results

All **1,687,829** printed raw/optimized pairs have distinct raw candidate IDs and
valid optimized resultants modulo AS660. Raw and optimized header counts agree.
The original six coefficient counts match the independently audited corpus.
The two added coefficients contribute 161,826 and 335,886 distinct candidates.
Their sidecars have complete coefficient and worker endings; this review did
**not** independently reconstruct their full collision populations.

Inclusive effort-0 cutoffs use printed exp_E precision. Rates divide the exact
observed qualifying count by recorded stage-1 elapsed hours; they are not GPU
busy-time or end-to-end pipeline rates. The first coefficient is not directly
comparable because its clock and contention differ.

| Original a_d | Distinct scored | Count <=56 | Count <=57 | <=56 / hour | <=57 / hour | Global top 300 |
|---|---:|---:|---:|---:|---:|---:|
| 138600* | 179,993 | 2,313 | 20,242 | 530 | 4,634 | 43 |
| 180180 | 161,826 | 2,034 | 18,490 | 594 | 5,397 | 33 |
| 277200 | 156,518 | 1,755 | 16,063 | 586 | 5,365 | 20 |
| 360360 | 335,886 | 4,227 | 37,951 | 697 | 6,259 | 71 |
| 415800 | 296,586 | 3,040 | 28,051 | 556 | 5,132 | 52 |
| 554400 | 147,028 | 1,434 | 13,703 | 555 | 5,305 | 21 |
| 693000 | 218,988 | 1,773 | 17,210 | 520 | 5,047 | 31 |
| 831600 | 191,004 | 1,751 | 16,680 | 591 | 5,628 | 29 |

The effort-0 top-300 cutoff is 54.40. Ties use original concatenated CPU output
order. These are top-300 effort-0 origins, not the 300 seeds selected after
effort-50 reoptimization. Full scoring removes subsampling error for this
corpus, not variability across future searches. In particular, the sampled
831600 estimate of 470/hour at <=56 becomes 591/hour with the full population.
360360 has the highest observed rates at both cutoffs, while 180180 is close
to 277200. This is evidence to retain those families, not a causal conclusion
about a particular prime factor or an established allocation optimum.

The completed pipeline extracted the top 1,000, re-ran sopt at effort 50, used
300 seeds for msieve ropt and 150 for CADO ropt, and ran original and sign-inverted
passes. CADO ropteffort was 5; the recorded worker count was 8. Every effort-50
input joined to an exact printed raw polynomial from the full-corpus output.

| Output | Best reported Murphy E | Verified original stage-1 a_d |
|---|---:|---:|
| CADO original | 1.071e-14 | 831600 |
| CADO inverted | 1.045e-14 | 360360 |
| msieve original | 1.046e-14 | 360360 |
| msieve inverted | 1.046e-14 | 360360 |

The saved skewopt report raises the CADO-original winner to 1.0753195700e-14,
with skew 158376848. This is the saved report, not a new scoring run. CADO's
printed scores use Bf=1e7, Bg=5e6, area=1e16. The GPU workflow uses inherited
job-specific settings and already schedules CPU/GPU rescoring with common
parameters; wait for that before comparing headline CPU/GPU numbers.

All 300 CADO outputs and the top 10 from each msieve pass have exact origin
proofs and valid resultants. No CADO output has ambiguous candidate origins.
The joins account for translations, rotations, signs, and changes of overall
polynomial multiplier. The initial rotation-only check failed on a valid CADO
output; the final analysis clears the multiplier denominator and checks the
resulting integer polynomial identity. CADO's `polyselect/ropt.c` also performs
local size descent and divides out content. The original failed log is retained;
the completed run is `process-retry.log` (exit 0).

The winning CADO seed is
`ab44896ee8d004bc758d86799f5e240a0781f4bca83c35be3238687ffc02fea2`.
It scored 52.77 at effort 0 and 52.62 at effort 50. Its previously audited
stage-1 discovery is retained in `cado-winner-stage1.json`: q=3,685,044,133,
batch 1,880,118 of 1,987,253 (about 94.6% through traversal), gate ratio
5.26777e-5. A strong late, large-q candidate exists here; one winner does not
establish a general q-quality trend or eliminate diminishing returns.

The msieve winner is
`1f46b368a7d64d3047fadfbadf71358e9da914faefe56d608cf59be56a0ac846`,
from 360360, with effort-0 and effort-50 score 52.46.

## Validation and remaining work

All frozen CPU reference hashes were verified against `prepared.json`.
The effort-50 file was hashed before and after analysis. The full effort-0
output was streamed using one low-priority CPU worker; no optimizers were rerun.
The memory index stores candidate hashes, histograms, and selected seeds, not
all 1.69 million polynomial pairs. Saved per-coefficient counts and rates are
exact for the observed corpus and printed precision.

Still pending: independent full collision audits for 180180/360360, logarithmic
q-bin analysis, unbiased controls outside the selected sopt tail, a quiet
138600 timing control, and the already-running GPU workflow's common-parameter
final comparison. CPU root optimization here is conditional on sopt selection;
it does not characterize each coefficient's entire final-quality distribution.
