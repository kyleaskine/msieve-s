> Historical run record moved from the main notes on 2026-10-22. See the [current conclusions](../../../POLYSELECT_OPTIMIZATION_NOTES.md) for subsequent evidence and revised priorities. Statements about a running or stopped process describe the time of this record.

## Second complete C196 coefficient and CPU quality samples (2026-10-21)

Inspected the completed a_d=277200 while the user's third coefficient continued.
No GPU work, production changes, or process control was performed. Artifacts and
reproduction scripts are in `stage23_bench/runs/20261021_as660_two_coefficients/`.

`snapshot.json` identifies the byte-exact schema-2 sidecar prefix through the
second coefficient's completion marker (4,748,715,449 bytes), plus its SHA-256 and
the extracted raw rows. No synthetic completion records were added. The audit
reports the expected missing `worker_end` because the source worker continued;
the completed coefficient itself passes all reconstruction and count checks.
All 161,745 outputs join to valid collisions, covering 1,926,806 batches and
107,901,096 q roots. There are no saturation or hash-cap events. The 705 match
arena fallbacks use the existing fallback path and do not truncate output.

| Measure | a_d=138600 | a_d=277200 |
|---|---:|---:|
| Returned/raw outputs | 185,201 | 161,745 |
| Distinct `(N,a_d,P,m)` candidates | 179,993 | 156,518 |
| Repeat discoveries | 5,208 (2.81%) | 5,227 (3.23%) |
| Distinct candidates / million q roots | 1,383.03 | 1,450.57 |
| Largest `coeff/coeff_max` | 0.00136910 | 0.00135234 |
| Sampled sopt median exp_E | 58.27 | 58.28 |
| Sampled sopt 10th percentile exp_E | 56.95 | 57.00 |
| Sampled sopt 1st percentile exp_E | 55.86 | 55.98 |
| Best sampled sopt exp_E | 53.34 | 54.41 |

The second coefficient yields 4.88% more distinct candidates per searched q
root. This is a work-normalized count, not a throughput or quality estimate:
entries per q and batch shapes also change. Its duration is 10,779.25 s RAW versus
10,779.48 s wall time. The first run had contention and a different duration
clock, so these runs do not establish a GPU speedup. Every collision again
passes the norm gate, with the largest value only 0.135% of its cutoff. This
still does not test the effect of changing the norm-dependent search geometry.

Quality methodology: select the 10,000 lowest SHA-256 candidate IDs independently
from each complete deduplicated ledger, then run the same frozen CPU CADO sopt
binary at effort 0, one thread, low process priority. `quality-manifest.json`
records selection and binary/input hashes. Both processes exit successfully;
all 20,000 outputs join exactly to their printed input polynomials and their
integer resultants remain divisible by N. Lower sopt exp_E is better; these are
not final root-optimized Murphy E values. The sample extremes are especially
noisy and do not rank the best candidates in the full populations.

`comparison.json` contains scores, full-corpus q exposure/yields, and descriptive
400-resample bootstrap intervals. Second-minus-first mean exp_E is +0.00548
(95% interval -0.03254 to +0.04175); median difference +0.01 (-0.05 to +0.0651).
The 1st-percentile difference is +0.12 (+0.0201 to +0.2102), a modest sampled
lower-tail advantage for the first coefficient. These exploratory intervals
have no multiple-comparison correction and do not establish a search policy.

Separately split q magnitude into equal-width quarters of its uint32 range and
discovery chronology into quarters of batch traversal. Q-bin tables exclude
61/84 sampled identities discovered in multiple q bins; overall scores retain
them. No sampled identity crosses chronology quarters. Neither coefficient
shows a consistent monotone quality decline with larger q or later discovery:

| Coefficient | Median exp_E by increasing q quarter | Median exp_E by discovery quarter |
|---|---|---|
| 138600 | 58.38, 58.20, 58.28, 58.22 | 58.31, 58.30, 58.27, 58.20 |
| 277200 | 58.30, 58.30, 58.31, 58.22 | 58.23, 58.325, 58.22, 58.38 |

This is no evidence for an early stopping rule at the sensitivity of these
samples; it is not proof of identical extreme tails. Each q bin contains about
2,300–2,900 sampled candidates. Next compare the completed third coefficient
using the same procedure, then apply matched root optimization to establish
whether the size-optimization differences survive into final quality. Keep the
reference search policy and provenance in the meantime. The third coefficient
has 199,550,235 q roots versus the second's 107,901,096, so it also represents
substantially more work; elapsed time alone is not a kernel regression test.
