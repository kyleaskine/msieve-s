> Historical run record moved from the main notes on 2026-10-22. See the [current conclusions](../../../POLYSELECT_OPTIMIZATION_NOTES.md) for subsequent evidence and revised priorities. Statements about a running or stopped process describe the time of this record.

## First complete C196 coefficient (2026-10-20)

The user's AS660 search completed a_d=138600 and stopped at the coefficient
boundary. The original `as660.ms`, `as660.log`, and `as660-stage1.w0.jsonl` are
preserved. Analysis and isolated profiling live in
`stage23_bench/runs/20261020_as660_first_coefficient/`.

- The complete streaming audit reconstructed every returned collision and joined
  all 185,201 raw outputs. It covered 2,502,772 batches and 130,144,102 q roots;
  no missing endings, saturation, hash caps, or unmatched raw records occurred.
  There were 1,043 match-arena fallbacks, which select the engine's existing
  fallback path rather than censoring output.
- There are 179,993 unique `(N,a_d,P,m)` identities: 5,208 repeat discoveries,
  or 2.81% of output rows. Of the repeated identities, 3,385 occurred with more
  than one q. Deduplicate downstream work while retaining every discovery.
- Every returned hit passed the admission gate. The median `coeff/coeff_max`
  was 0.000213713, the 99th percentile 0.000920289, and the maximum 0.001369096.
  This is evidence that the gate is inactive for this geometry, not evidence
  that changing `stage1_norm` (which also changes geometry) is harmless.
- Full batches contain 52 q roots and three trans groups:
  `(1903,1)/256`, `(135,2)/256`, and `(1,3)/192`. The last group still contains
  only 132 p values with 25 roots each. The final partial batch searches 10 roots
  with five progressions; the rest use one.
- An isolated 57-batch Nsight Systems capture attributed 38.2% of summed kernel
  time to scatter, 21.6% to bucket filtering, 27.0% to the two larger trans
  groups, and 2.6% to the sparse trans group. Matched-value collection used
  another 10.1%. Thus eliminating the sparse launch entirely would save only
  about 2.6% of kernel time in this window. Kernel fractions are not end-to-end
  speedup estimates.
- Nsight Compute measured the dominant trans launch at 56 registers/thread,
  63.64% achieved occupancy, 74.93% DRAM throughput, 74.37% compute throughput,
  67.66% aggregate L2 hit rate, and no spills. The medium launch had 69.9%
  long-scoreboard stalls. These are three sampled launches with unlocked clocks
  and `--cache-control none`; aggregate L2 hit rate does not isolate the prefix
  load. Keep the reports for a targeted follow-up.
- The traced timeline has median batch spacing about 12.4 ms, whereas late
  production batches were about 5.1 ms. Do not interpret its apparent idle
  fraction as recoverable production overhead. Late production q generation
  took about 29 microseconds/batch: overlapping that alone has a small ceiling.
  CUDA event spans include host-mediated gaps and are not kernel busy time.
- Five alternating logging-on/off pairs used a_d=138600,
  q=2140000000..2150000000, fixed one progression, one worker, and the frozen
  pilot binary/PTX (hash-identical to the user's current build). Timing ran from
  the coefficient's start line to its completion line, with line-buffered
  stdout, after one warm-up. Every run produced the same 432-polynomial multiset.
  Pooled logging overhead was 2.71%, median paired overhead 2.93%, with pairs
  ranging from 0.36% to 4.49%. This does not establish the <=1% goal. Retain
  provenance for the research run; reducing repeated formatting/flush work is
  a follow-up, with output parity and another paired measurement required.

The auditor now has `--stream`, which retains active batches and candidates while
binning q exposure into sixteenths of the uint32 range. The full audit took 155 s
and peaked at 606 MiB RSS. Exact repeated-q/root detection is explicitly
unavailable in this mode (`null`), rather than silently reported as zero. Eight
CPU tests cover reconstruction, joins, streaming parity, retired-batch replay,
incomplete batches, count mismatches, and hash-cap censorship. Reproduce with:

```
python3 stage23_bench/tools/stage1_provenance.py as660-stage1.w0.jsonl \
  --raw as660.ms --stream --require-complete --ledger NEW-ledger.jsonl
```

`completed-corpus.json` records hashes of the original sidecar and saved
`first-as660.ms` / `first-as660.log` snapshots. `analyze.py` writes `unique.ms`
with one raw polynomial per candidate, checking that repeated identities have
identical raw expansions. `first-analysis.json` contains discovery multiplicity,
gate ratios, q exposure/yield bins, and all five timing pairs. Across 16 q bins,
raw hit yields range from 1,399 to 1,445 per million searched q roots; these
include repeat discoveries and do not measure downstream quality.

Next: run common-setting CPU sopt on the deduplicated corpus and compare
q-conditioned quality tails per searched
q/root work. Raw hit yield alone cannot answer whether small q produces better
polynomials. No launch heuristic, norm, or coefficient-family change has been
made on the strength of this first corpus.

For the user's next production invocation, start at `min_coeff=277200`, retain
`high_coeff_mult=138600`, and choose a fresh provenance prefix. Stage 1 does not
subtract old `.ms` rows from `num_polys`; `num_polys=1414799` requests the remainder
of the original 1,600,000-output target (with the existing coefficient-boundary
overshoot). Keeping `-s as660 -l as660.log` appends to the existing outputs. The
per-run audit corpus must use the saved first-coefficient raw snapshot after
that append, rather than trying to join new rows against the old sidecar.
