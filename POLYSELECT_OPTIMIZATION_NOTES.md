# Polyselect Stage-1 GPU Optimization Log

> **Consolidates four prior plan docs** (`POLYSELECT_OPTIMIZATION_NOTES.md` original, `TRANS_KERNEL_PLAN.md`, `TWO_LEVEL_SCATTER_PLAN.md`, `FUSED_TRANS_SCATTER_PLAN.md`). The full version history is in git; this file is the working reference going forward.

## Current conclusions (2026-10-23)

The CPU pipeline is complete on **1,687,829 distinct candidates across eight
coefficients**, including 180180 and 360360. The full output passed resultant
checks; the top-1,000 effort-50 inputs joined exactly, and all 300 CADO ropt
outputs plus the top 10 from each msieve pass have verified origin proofs.
The best reported CADO result comes from **831600** (Murphy E 1.071e-14,
1.07531957e-14 in the saved skewopt report); the msieve winner comes from
**360360** (1.046e-14). The CADO winner's audited discovery has q=3,685,044,133,
about 94.6% through that coefficient's batch traversal.

Full scoring revises the 831600 <=56 yield from the sampled 470/hour to
**591/hour**. The new 360360 run gives **697/hour** and 71 of the global top 300
effort-0 seeds; 180180 gives **594/hour**, close to 277200's 586/hour. These are
observed stage-1 elapsed-hour rates, not estimates of isolated GPU speed or
proof of a general coefficient policy. The first timing remains unsuitable
for direct comparisons. The GPU workflow is running separately and includes
CPU/GPU rescoring under common job parameters; no new GPU work was started here.

See the [completed CPU review](stage23_bench/runs/20261023_as660_cpu_review/README.md)
for exact tail counts, top-300 origins, proofs, settings, and limitations. The
subsequent archive work completed the full collision audits for 180180 and
360360: all 517,661 new returned hits reconstruct and join to raw output, with
no saturation, hash caps, or unmatched rows. All eight coefficients are now
audited: 1,755,106 raw outputs / 1,687,829 distinct identities. Outside-sopt-cutoff
controls remain pending. The October 22 proposal to launch full CPU scoring/ropt below
is now superseded by the user's completed pipeline.

The [archive and compact research index](stage23_bench/runs/20261023_stage1_archive/README.md)
retain all candidate discoveries in 149.45 MiB and logarithmic exposure for
1,141,047,425 searched q-root occurrences. All candidate IDs were recomputed
from N/ad/P/m during readback. Full sidecars are losslessly preserved as zstd
archives; the manifest records hashes, reconstruction ranges, and cleanup state.
The main auditor supports compressed originals and archived extracted paths.
Cleanup is complete: five verified archives occupy 15.527 GiB; the five original
files and five redundant extracts were removed, reclaiming about 59.0 GiB after
accounting for compact provenance. All ten old paths resolve through the archive
reader after deletion. GPU work and polynomial inputs/results were untouched.

| Question | Current evidence | Consequence |
|---|---|---|
| Corpus integrity | All eight coefficients independently collision-audited and CPU-scored; 1,687,829 distinct identities | Use the compact provenance and exposure index for further analysis |
| Coefficient productivity | Full-corpus cutoff rates are available; 360360 leads the observed <=56 and <=57 rates | Promising family; no established general allocation optimum |
| Final quality | CPU selection/ropt complete; CADO winner originates from 831600, msieve winner from 360360 | Await common-parameter CPU/GPU comparison; retain outside-cutoff controls as a research task |
| Q magnitude | Larger linear q quarters have slightly better algebraic exp_E on average; log2 exposure is now saved for all eight coefficients | Join final scores to the saved logarithmic bins; do not change q coverage yet |
| Discovery order | No detected mean-score deterioration across traversal quarters | No demonstrated quality-based early stopping rule |
| Admission norm | First-six maximum returned gate ratio 0.00136953; all eight coefficients' returned hits pass | No rejected population here; changing norm also changes geometry |
| Stage-1 speed | Representative C196 profiles exist; no launch/metadata speed change accepted | Reprofile and benchmark only with an available, uncontended GPU |

### October 22 review: sample-based elapsed-time normalization

Claude's per-hour table is reproduced exactly by
`stage23_bench/runs/20261022_as660_six_coefficients/compare_productivity.py`;
inputs, hashes, rational-side diagnostics, and results are saved in
`productivity-comparison.json` there. Rate is
`distinct_count * sampled_fraction * 3600 / recorded_stage1_seconds`.
These are recorded stage-1 elapsed hours, including host work and logging within
that timing scope, not isolated GPU-busy hours or total stage-1/2/3 pipeline cost.

| a_d | q roots/s | Estimated exp_E <=56 / elapsed hour (95%) | Estimated exp_E <=57 / elapsed hour |
|---|---:|---:|---:|
| 138600* | 8,276 | 581 (493–684) | 4,582 |
| 277200 | 10,010 | 575 (478–692) | 5,300 |
| 415800 | 10,141 | 591 (491–713) | 5,057 |
| 554400 | 10,970 | 564 (463–685) | 5,322 |
| 693000 | 11,121 | 559 (453–688) | 5,285 |
| 831600 | 11,734 | 470 (374–591) | 5,484 |

*The first coefficient used CLOCK_MONOTONIC under contention; later coefficients
use CLOCK_MONOTONIC_RAW. Do not rank the first rate against the others or treat
an extrapolated ~670/hour as a corrected measurement. Clock effects and contention
cannot be separated from these totals. Later nominal entry throughput spans
6.11–6.50 billion entries/s (entries/q times q roots/s), explaining much of the
q-root rate increase as entry counts fall. Wilson intervals include only sample
uncertainty conditional on the recorded times. They omit timing uncertainty and
variation between future runs; overlapping intervals are not an equivalence test.

The earlier statement that extra candidate density does not compensate for the
quality shift applies **per q root**, not per elapsed hour. At <=57 there is no
monotone loss with increasing a_d; at <=56 the later estimates overlap widely.
The 831600 point estimate is lower at <=56 but does not establish a reliable loser.
These exploratory cutoffs use printed score precision and are not final Murphy E.

### Score interpretation and next work

The local CADO `polyselect/auxiliary.c:211` defines algebraic exp_E as lognorm
plus expected rotation gain; `:350` and `:363` include projective alpha in that
gain. Do not add current alpha or projective alpha again. Claude's temporary
`alpha.py` has a stale contrary comment and its exp_E+alpha column is not a
corrected score. The rational L2 diagnostic at the printed algebraic skew
reproduces: coefficient means range 79.869–79.900, with differences from 138600
between -0.0234 and +0.0079. This supports first-order comparability, not exact
equality of final sieving quality. Within q bins, adding this diagnostic reverses
the high-minus-low mean sign for 554400 and 693000. The sum itself is not Murphy E.

Next score all 1,190,117 distinct candidates using the same frozen CPU sopt binary,
explicit effort 0, and candidate IDs. Report exact observed tail counts, elapsed-hour
yields, and origins of the global top 300, with deterministic handling of rounded
score ties. Full scoring removes subsampling uncertainty for this corpus, not
uncertainty about future coefficients. The current C196 samples took roughly
98–191 s per 10,000 on one low-priority worker: budget roughly 3–6 serial hours
before overhead, not the unverified C184 runtime. A few CPU workers may reduce
wall time, but reserve CPU capacity for the active stage-1 producer. The generic
`dedupe_and_sopt.sh` does not explicitly fix effort or retain these audit IDs;
use an isolated, identity-preserving run rather than its default output paths.

Then run broad/deep root optimization on the top seeds with common Murphy-E
parameters and explicit parent IDs. For research, supplement the global top 300
with a fixed per-coefficient quota and a stratified control sample outside that
cutoff, including q strata. Top-300 origins alone mix corpus size, runtime, and
score selection and cannot settle allocation or measure missed finalists.
Keep polynomial selection for the real C196 and inference about search policy
as distinct outputs of the same experiment. Full scoring/ropt have not started.

The user's 180180 run is a useful new coefficient: it is 13/10 times 138600,
replacing one factor of 2 and one of 5 with 13. Size and divisibility both change;
it is not an isolated factorial comparison. Audit its completed output and apply
the same scoring pipeline. Add log2 q bins (q=1 separate), with searched-root
exposure from the sidecars and explicit multiple-discovery attribution. Neither
the present linear-quarter association nor log bins within q<=2^32 justify an
extrapolation beyond that clip or predict the B_transition norm-sweep result.
Once the GPU is available, time identical fixed 138600 work without contention
using the current clock, retaining setup separately and matching q/workload shape.

### Correctness and artifact handling

The shutdown drain applies to **all GPU stage-1 runs**, including runs without
provenance and graceful Ctrl-C shutdown. `gpu_data_free` now drains queued stage-2
tasks after stopping the GPU producer; `threadpool_free` alone discards queued
tasks. This can extend graceful shutdown while preserving pending polynomial
expansion. It does not guarantee survival of SIGKILL or a second forced interrupt.
The earlier review log mentioned the drain but did not make its default-path scope
clear. Keep three logical commits when preparing changes: prefix correctness,
stage-2 shutdown correctness, then provenance/q-window/aprog instrumentation.
No commits were made as part of this review.

Per-run narratives now live in their run-directory READMEs, linked below. Original
production files and analysis snapshots remain intact; no compression or deletion
was performed. There is currently ample free disk space. If reclaiming storage,
compress only closed artifacts, verify decompression hashes, and update readers
and manifests before removing redundant copies; preserve active production files.
This review used only saved CPU results and source; it did not touch the GPU run.

## Six complete C196 coefficients (2026-10-22)

All six completed coefficients passed the audit: 1,237,445 outputs and 1,190,117 distinct identities. The 60,000-candidate sopt sample is complete; elapsed-time normalization supersedes interpreting per-q yield as productivity.

[Full run record](stage23_bench/runs/20261022_as660_six_coefficients/README.md).

## Third complete C196 coefficient (2026-10-21)

The third coefficient passed reconstruction and output joins; its 10,000-candidate CPU sample completed.

[Full run record](stage23_bench/runs/20261021_as660_third_coefficient/README.md).

## Second complete C196 coefficient and CPU quality samples (2026-10-21)

The second coefficient passed the audit. Frozen CPU sopt samples for the first two coefficients preserve exact candidate identities.

[Full run record](stage23_bench/runs/20261021_as660_two_coefficients/README.md).

## Provenance code-review follow-up (2026-10-20)

All 15 review items were checked and addressed or qualified. Schema 2, runtime PTX hashing, RAW timing, bounded streaming, and graceful error handling passed the documented regressions. The <=1% logging-overhead goal remains unproven.

[Full run record](stage23_bench/runs/20261020_stage1_review/README.md).

## First complete C196 coefficient (2026-10-20)

The first complete coefficient and representative GPU profiles are archived. Its elapsed time is affected by contention and the older clock; it is not a clean performance baseline.

[Full run record](stage23_bench/runs/20261020_as660_first_coefficient/README.md).

## Prefix fix and provenance implementation (2026-10-20)

The prefix fix, optional provenance, q windows, and fixed-progression controls passed arithmetic, parser, and bounded GPU checks. Later review fixes and production results supersede the original pending-work list.

[Full run record](stage23_bench/runs/20261020_stage1_provenance/README.md).

## Profile and source audit (2026-10-19)

The ordered implementation plan, provenance requirements, and proposed C196
quality experiments are in
[the execution roadmap](POLYSELECT_RANKING_IMPROVEMENT_PLAN.md#execution-roadmap-and-c196-search-2026-10-19).
After the review, start with the isolated prefix-width correctness issue,
provenance, and a representative C196 baseline. Use measured C196 launch shapes
to select tuning targets; the c157 sparse-group finding is not a universal priority.

Read-only performance investigation of the saved reports and current production
source; no new GPU timings or kernel changes. The May conclusions below describe
the experiments performed then, not proof that these kernels cannot improve.
In particular, the dominant trans capture is memory-latency-heavy, and the
previous wave-alignment experiment did not address the very small trans launch.

### Evidence and timing scope

Reimported `trans_full_post_memset.ncu-rep`,
`trans_full_post_memset_60060.ncu-rep`, `trans_full.ncu-rep`,
`trans_full_medium.ncu-rep`, `filter_420.ncu-rep`,
`filter_60060.ncu-rep`, and `scatter_roots_ncu.ncu-rep`. Also queried
`profile_gerbicz_420_memset.sqlite` and the earlier arena trace directly.
The trans report has no CUDA line information; the load identifications below
use the captured SASS, kernel argument offsets, and matching source control flow.
Stall sample percentages are shares of sampled warp stalls, not wall-time savings.
See NVIDIA's [stall attribution guidance](https://docs.nvidia.com/nsight-compute/ComputeTriage/).

The memset-cleanup Systems trace contains 12,456 batches. Its measured kernel
totals differ slightly from the rounded May summary: trans 25.923 s, scatter
24.342 s, filter 12.979 s, count-and-store 6.258 s. The span from first to last GPU
activity is 106.855 s; the union of kernels, copies, and clears occupies 70.391 s.
These are instrumented trace times, not the approximately 88 s unprofiled run.

### 1. Small trans launch leaves almost the whole GPU unused

The host launch heuristic in `stage1_sieve_gpu.c:780` targets an estimated amount
of work per block, without a minimum device-coverage requirement. The three
degree-5 groups have 1, 5, and 25 roots per p and run serially in the same stream.
Grouping the Systems trace by launch shape exposes this distribution:

| Grid / block | Launch count | Total GPU time | Mean launch |
|---|---:|---:|---:|
| `(317,3)` / 224 | 12,255 | 16.056 s | 1,310 us |
| `(21,10)` / 256 | 6,898 | 2.932 s | 425 us |
| `(21,11)` / 256 | 5,557 | 2.251 s | 405 us |
| **`(1,5)` / 128** | **12,455** | **4.420 s** | **355 us** |

The startup copies, interpreted using `p_soa_array_start`, identify the p counts
as 71,001, 5,296, and **30**. The third group's 120-byte p copy and 6,000-byte root
copy correspond to 30 p values with 25 roots each. Thus each of its five blocks
has only 30 useful threads; the other 98 return immediately. At most five of 48
SMs can work on this launch, with only one useful warp per block. The group's
750 entries are only 0.76% of 98,231 entries per q, but consume 17.1% of trans
time and about 6.3% of total kernel time. The earlier arena trace independently
shows 4.261 s for the same shape.

**Candidate experiment:** increase q-axis subdivision for this sparse
group, and test a warp-sized x block. There are 356 q records in a saved normal batch,
so considerably more independent q tiles are available. More tiles increase the
number of modular inversions and may split runs of equal q; benchmark that
tradeoff. This is different from the reverted *decrease-only* wave-alignment
experiment on the 951-block launch. The measured 4.42 s is an upper bound on
what eliminating this launch's entire cost could save in this trace.

### 2. Repeated p/q metadata stores can be avoided on the Gerbicz path

The trans kernels write `p_out[j] = (q << shift) | p` for every generated root
(`stage1_core.cu:546`, with equivalent sites in all variants). For fixed p arrays,
shift, and output layout, this value depends on the **batch-local q index**, not
the actual special-q value or the computed root. Gerbicz only reads this array;
unlike the sort path, it does not reorder it.

A normal saved batch has 98,231 entries per q and 356 q records: 34,970,236 slots,
or **139.88 MB of 32-bit metadata capacity**. Nearly all valid slots are rewritten
every batch. The collision scan only fetches a slot's metadata after matching a
candidate; the May counters record roughly 2,826 matched values per batch.

**Preferred experiment after review:** reconstruct metadata only for matches.
For global slot `tid`, `E = num_entries`, `Q = current batch root count`, and
`plane = E*Q`, decode
`q_index = (tid % plane) / E` and `p = p_of_entry[tid % E]`.
The per-entry table follows the concatenated root-group layout, including repeated
p values for their roots. It is about 393 KB for the saved 98,231 entries. This
removes both the repeated store stream and the roughly 140 MB metadata buffer.
Update both normal and fallback match paths and the host/DSO interface together;
validate partial batches, progression planes, root groups, and sort behavior.

One caveat to the review's small-change estimate: `sieve_kernel_trans_pp32_r64`
uses `p_out` as prefix scratch as well as final metadata (`stage1_core.cu:299,314`).
Removing the array for every Gerbicz kernel variant requires preserving or
replacing that scratch use. The c157/C196 pp64 path uses `roots_out` instead.
Pre-initialization remains a fallback experiment; it requires initialization of
even initially invalid slots and refresh on layout/shift changes.

### 3. Trans repeatedly reads its own prefix-product scratch

The dominant trans capture reports 68.53% DRAM throughput, 52.81% SM throughput,
51.3% long-scoreboard stalls, 28.99 active threads per warp, and 56 registers
without spills. Calling it simply math-chain-saturated is too strong.

The forward pass stores prefix products into the first root plane
(`stage1_core.cu:486`). The backward pass loads them (`:517`) and subsequently
overwrites those locations with final roots. This is intentional batch inversion
amortization, not an unnecessary duplicate calculation, but it creates an
expensive memory dependency:

- SASS load at `0x1701106090` followed by the nonzero test at `0x17011060b0`:
  **46,919 / 75,182 long-scoreboard samples (62.4%)**. The pointer is the
  `roots_out` kernel argument, matching the backward prefix-product read.
- Load at `0x1701103db0` and test at `0x1701103dd0`: **16,308 samples (21.7%)**,
  matching the forward `curr_q->p` read and q-change test.
- Another 4,507 samples occur after loading the q root, and 4,490 after loading
  a start root. Together these four sites account for about 96% of the captured
  long-scoreboard samples.

**First experiment within the launch sweep:** increase q subdivision. The saved
dominant launch gives each thread about 119 q records; 48 SMs × 5 resident blocks
× 224 threads × 119 prefixes × 8 bytes is about 51.2 MB, versus the GPU's 48 MiB
L2. Twelve q tiles instead of three would reduce that nominal per-wave prefix
footprint to about 12.9 MB. These are working-set estimates, not measured cache
residency: other arrays, overlapping blocks, scheduling, and cache policy matter.
Measure prefix-load L2 hits/traffic, DRAM traffic, and elapsed time for both
variants. One capture or the 62.4% stall attribution alone cannot establish a
speedup. Extra tiles also repeat modular inversions.

The medium `(21,10–11)` launch has 210–231 blocks versus 192 resident blocks
under a four-block/SM limit: test its possible partial-wave tail too. Treat this
as one launch-heuristic sweep across groups, rather than hard-coding a fix for
the old five-block group.

**Subsequent experiments:** prefetch the next prefix while working on the current one;
investigate read-only/alias annotations for q and start-root inputs; consider
bounded q tiles that retain some prefixes on chip. Preserve the batch-inversion
algorithm and track registers: 56 to 64 registers would cross the five-block
occupancy boundary for 224-thread blocks. The previous forced-register and
scatter-fusion failures do not directly test these memory-access changes.

### 4. Filter: repeated passes, underfilled warps, and an allocation boundary

The filter first reads every bucket item to build hash tables, then reads them
again to filter. Survivors are repeatedly compacted into global `arr_a/arr_b`
buffers and read on the next iteration (`collision_engine.cu:198,235,264`).
The capture reports 19.509 million global-load sectors at L1 (624.3 MB requested)
but 282.2 MB DRAM reads: cache does absorb part of the repeated traffic. Global
stores use only 11.3 of 32 bytes per sector on average. The repeated-loop item
load accounts for 30,440 / 48,710 long-scoreboard samples (62.5%). Shared loads
also show about three wavefronts per request due to bank conflicts.

The block contains 128 threads, has 24.576 KB dynamic plus 96 bytes static shared
memory, and incurs 1,024 bytes of driver shared memory. Allocation rounds to
**25,728 bytes per block**. Four blocks would require 102,912 bytes, just over the
102,400-byte SM budget; only three fit. This is a real allocation boundary, not
just a vague need for less shared memory. Removing the 80-byte history array
alone does not establish that a fourth block will fit: counters and allocation
rounding still matter.

**Low-cost experiment:** sweep 128/256 threads without changing hash resolution
or convergence. At three resident blocks, 256 threads could provide 24 rather
than 12 warps per SM, subject to the compiled register usage. The tradeoff is
extra idle warps and synchronization once survivor counts become small. No such
sweep is recorded in the May log. Higher occupancy alone is not a performance
guarantee ([CUDA guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)).

**Larger experiment:** after the bucket shrinks, process a bounded survivor list
on chip, with a fallback for larger lists. This differs from putting the entire
initial bucket in shared memory. Reuse space released as hash tables shrink if
possible, rather than increasing the launch's shared-memory allocation.

The report shows 24.64% achieved versus 25% theoretical occupancy, and average
SM-active cycles are 99.2% of elapsed cycles. It therefore does **not** support a
claim that most filter time is a globally idle GPU waiting for a few final
blocks. Average active lanes are 24.55/32 (21.90 after predication), and 12.5% of
sampled stalls are at barriers: there is inefficiency *within* blocks. The
iteration histogram establishes varying work, but not its share of total time.
Changing the history/barrier scheme must preserve the barrier that prevents the
next iteration from resetting `s_cnt_out` before every thread has read it.

### 5. Host/device round trips remain visible

`collision_engine_run` has four explicit stream synchronizations per normal
batch: after bucket sizing, candidate count, dedup count, and match count. The
trace confirms **49,824** runtime stream synchronizations and **87,192 four-byte
DtoH copies** (seven per batch), in addition to the final found-array copy.
The candidate count is copied to pageable stack storage, which can block inside
`cudaMemcpyAsync` itself. Attribute queries/settings are also repeated per batch.

The 36.464 s of GPU-idle intervals in the instrumented trace includes CPU work,
launch/readback latency, OS scheduling and profiler overhead. It is not a claim
that 36 s can be recovered, and API time spent waiting overlaps GPU execution.
Nevertheless, keep invariant setup out of the batch loop, consider deferring
safe arena-success/overflow decisions to device code, and measure an unprofiled
run before deciding whether broader orchestration changes are worthwhile.

The next batch's q roots are generated on the host after the preceding collision
work and found-array synchronization. Measure that uncovered interval in the
initial C196 baseline, alongside p setup and the upfront q-root count. Preparing
the next batch concurrently may outrank kernel tuning if its measured share is
large; it needs independent mutable state and buffers. Do not estimate an 18 s
host gap by subtracting the 70 s instrumented kernel total from the unrelated
88 s unprofiled run. Existing batch CUDA events span host-mediated launch gaps
as well as GPU work; distinguish elapsed event spans from active GPU intervals.

### Correctness and allocation findings from the review

- `sieve_kernel_trans_pp64_r64` loads a 64-bit prefix into a `uint32` at
  `stage1_core.cu:502`, then tests for zero. A nonzero prefix with zero low bits
  is skipped incorrectly. Fix separately with a targeted arithmetic-reference
  regression before setting the new parity baseline. The asserted rate of
  about 0.6 occurrences per coefficient-420 run is unverified; output effects
  should be measured, not assumed to be uniform across all roots in a tile.
- Both sort scratch arrays are allocated even for Gerbicz
  (`stage1_sieve_gpu.c:1707,1709`), but their uses are in the sort branch.
  At single-worker caps, 50M × 4 plus 35M × 8 bytes is **480 MB decimal**.
  Allocate/free them conditionally, preserving sort pointer-swapping and cleanup.
  This saves capacity; it does not by itself demonstrate faster execution.

### Revised experiment order

1. Isolate and validate the prefix-width fix; establish corrected references.
2. Add provenance/controlled q windows before a substantial C196 run.
3. Measure C196 launch shapes and host/device costs; start the useful search.
4. Sweep trans launch geometry and test index-derived metadata independently.
   Remove sort-only scratch independently; prioritize host overlap if measured
   uncovered work justifies it. Tune in gaps without contaminating benchmarks.
5. Retain filter/prefetch experiments as follow-ups selected by remaining cost.

Use c157 coefficients 420/60060 for regression diagnostics, plus representative
C196 fixed work. The roadmap specifies >=5 interleaved A/B pairs, >=2% median
coefficient-time improvement with a 95% interval favoring the variant, and
investigation of >2% regressions. Measure startup separately. Require corrected
canonical polynomial-set parity, including partial batches and multiple
progressions. None of the proposed speedups has been measured yet.

## Current state (2026-05-26)

Polyselect stage 1 on the test 157-digit composite, degree 5, RTX 5070 (sm_120), CUDA 12.8. Coeff 420 baseline timing on the production gerbicz path:

- **Wall:** ~1m28s (current Gerbicz path, no collstats)
- **Engine kernel time:** ~56s (collstats=1)
- **Polynomial output:** 12,739 polynomials, md5 `ef7d7676615235c41caa228698940511` after canonical record-sort
- **Parity reference file:** `profiling/msieve.gerbicz.dat.ms`

### Top kernels (coeff 420, post-memset cleanup, current baseline)

| Kernel | GPU time | % | Avg/launch |
|---|---:|---:|---:|
| `sieve_kernel_trans_pp64_r64` | 25.40s | **37.4%** | 680 µs |
| `scatter_roots_kernel` | 23.05s | **33.9%** | 1.85 ms |
| `filter_per_bucket_kernel` | 12.70s | **18.7%** | 1.02 ms |
| `count_and_store_matched_values_kernel` | 6.07s | 8.9% | 487 µs |
| CUB DeviceRadixSortOnesweep | 0.46s | 0.7% | — |
| `emit_found_arena_kernel` | 0.05s | 0.1% | — |
| Memsets | 0.12s | 0.2% | — |

**Top 3 kernels = 90% of GPU time. All three confirmed at structural ceilings** against every implementation strategy attempted (see "Optimization log" below).

### Coeff 60060 baseline (for scale comparison)

92,454 batches, max_bucket 2417, 217,073 found polynomials, engine 407.5s, wall 10:33. Same parity / collstats counters scale linearly.

## Foundation that survived

Pieces from the optimization push that landed in tree (all zero-overhead or actively useful):

| Piece | Location | Purpose |
|---|---|---|
| Gerbicz collision engine | `cub/collision_engine.{cu,h,so}` | The big landed win — ~40% wall improvement over CUB radix sort path on coeff 420 (sort 2:50 → gerbicz 1:39). Opt-in via `collengine=gerbicz`. |
| B1 bounded-arena fast path | `cub/collision_engine.cu` (`count_and_store_matched_values_kernel` + arena emit) | `MATCH_ARENA_WIDTH=8`. Matched-value path 11.7s → 6.2s on coeff 420. <0.003% fallback rate. |
| Memset cleanup | `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu` + `stage1_sieve_gpu.c:702` | Per-batch 267 MB `gpu_root_array` clear is 98.7% of memset time. Skip when `num_aprog_vals == 1` + refine trans kernel to write plane 0 inline. Memset GPU time 5.6s → 0.12s on coeff 420; engine 469 → 419s on coeff 60060. |
| Shared bucket-hash header | `cub/collision_bucket.h` | Extracted `NUM_BUCKETS`, `LOG2_NUM_BUCKETS=14`, `BUCKET_MASK`, `BUCKET_HASH_MIX`, `compute_bucket()` so engine + any future bucket-aware kernel use one source of truth. |
| `GPU_MAX_KERNEL_ARGS` 15 → 20 | `include/cuda_xface.h:90` | Headroom for kernels with more args. From the (reverted) fusion spike, but harmless. |
| `gpu_kernel_args_idx[]` refactor | `gnfs/poly/stage1/stage1_sieve_gpu.c:72` | Replaces the fragile `(i / 3)` arg-grouping shortcut. Pure infra, no behavior change. |
| Filter iters histogram | `cub/collision_engine.cu` + `stage1_sieve_gpu.c` collstats print | 21-iter stop distribution × 3 reason categories (cnt==0 / converged / cap-hit) + 3-category bucket-size log2 histogram. Gated under `collstats=1`, ~0% overhead off. |
| Engine collstats counters | `cub/collision_engine.h` | bucket_max, candidates, dedup, matched, arena_attempts/fallbacks/capacity_skips, bucket_grow_count, hash_cap_count. Surface engine-internal health. |
| `found_array` saturation stats | `stage1_sieve_gpu.c` (`check_found_array`) | Always-on `peak/saturated/total` print. Confirms no silent drops (peak 11 on coeff 420, well under 1000 cap). |
| Makefile rebuild deps | `Makefile` | `cub/collision_bucket.h` added to `NFS_GPU_HDR` and `cub/built` prereqs; `-I.` on the `stage1_core_sm{89,90}.ptx` nvcc rule. |

### CLI flags (in `-nps` argument string)

- `collengine=gerbicz` — use the Gerbicz collision engine (vs default CUB sort)
- `colllib=PATH` — override the engine .so path
- `collhash=1` (default) / `collhash=0` — bucket hashing on / off
- `collstats=1` — log full collstats + filter histogram at end of coeff
- `colldebug=1` — verbose per-batch growth diagnostics

## Optimization experiment log

Chronological summary of every lever tried after the Gerbicz engine landed. All experiments **passed byte-exact polynomial parity** unless noted. The "Verdict" column says whether the change is in tree (✓) or reverted (✗).

| # | Date | Lever | Measured outcome | Verdict | Key lesson |
|---|---|---|---|---|---|
| 1 | 2026-05-23 | Initial profiling (nsys + ncu) | Sort = 83.7% of GPU time pre-Gerbicz | — | Confirmed sort was the lever to attack |
| 2 | 2026-05-24 | **B1 bounded-arena fast path** | Matched-value path 11.7s → 6.2s on coeff 420 | ✓ | Arena width 8 covers >99.9% of slots; exact fallback for the rare overflow |
| 3 | 2026-05-24 | A1: `__stcs` cache hint on `bucket_storage` write | L2 sector traffic -3pp, duration flat | ✗ | scatter_roots is L2-throughput-ceiling-bound, not write-bandwidth-bound |
| 4 | 2026-05-24 | A.5: direct-atomic scatter (no warp aggregation) | Compute throughput -25pp, duration flat | ✗ | Kernel is L2-throughput-limited regardless of compute load. Warp aggregation contributes ~0 useful work at 16K buckets. |
| 5 | 2026-05-24 | LOG2_NUM_BUCKETS sweep (12..17) | LOG2=14 confirmed empirically optimal | ✓ (no change) | Bucket count is a balance between scatter cost and filter cost, not a coalescing knob. Bucket counts cluster in [1024, 2414] regardless. |
| 6 | 2026-05-25 | **Memset cleanup (v6)** | 267 MB `gpu_root_array` clear was 98.7% of memset time. Skip when `num_aprog_vals == 1`; refine trans to write plane 0 inline. Memset 5.6s → 0.12s on coeff 420; engine 469s → 419s on coeff 60060. | ✓ | Profiling showed CUB-internal scratch was the assumed culprit; actual was a single fat clear. Reading the data wins. |
| 7 | 2026-05-25 | T3: `__launch_bounds__(256, 6)` on trans pp64_r64 | Compiler hit target (56 → 40 regs, 73% → 87.5% theoretical occupancy, zero spills) but per-launch 1.20 → 1.26 ms (+5%) and Stall Long Scoreboard 51% → 58%. Wall +2.9%. | ✗ | The math chain needs its registers. Cutting them pushes loop state into dependent memory ops; lost cycles dominate the gained warps. |
| 8 | 2026-05-25 | Step 2: tail-effect wave-alignment (`cuOccupancyMaxActiveBlocksPerMultiprocessor` + decrease-only blocks_y) | nsys per-kernel: trans +0.7% (essentially flat). Wall A/B 6 pairs: mean -3.5% / median -0.9% / std ~7s (not distinguishable from zero). | ✗ | ncu's "up to 25% tail effect" is the worst-case ceiling, not the realistic delta. The modinv-amortization theory didn't materialize because forward/backward montmul loops already dominate per-thread work. |
| 9 | 2026-05-25 | Variant B: filter convergence 3-iter → 2-iter stability rule | Polynomial parity byte-exact, arena_fallbacks unchanged, dedup/matched/found identical. Filter -1.1%, count_and_store +4.5% (candidate growth absorbed downstream). Net flat. | ✗ | Phase B filter is **slowest-block-bound, not average-bounded**. Shifting the bulk of iter-4 converged buckets down to iter 3 saved nothing on wall — the long-tail iter-5+ blocks still dictate kernel duration. |
| 10 | 2026-05-25 | Tail-bucket-size diagnostic (coeff 420 + 60060) | Slow buckets (it ≥ 5) have **same size distribution as fast buckets** — at coeff 60060, slow are *slightly smaller* on average (9.0% in [1024, 2047] vs fast's 6.2%). | ✗ (closes another lever) | Slow buckets aren't oversized — they're slow because their *keys* cluster in hash subspaces. "Split oversized buckets" lever is closed. |
| 11 | 2026-05-25 | **Option A fusion: inline trans + bucket-scatter** | Byte-exact polynomial parity, collstats within 4 keys of 35.2M (~10⁻⁷, harmless intra-warp atomic-ordering drift between `0xFFFFFFFFu` and `__activemask()`). nsys: trans 25.4s → 65.0s (+39.6s); scatter saved 23.0s; **net +16.6s GPU kernel time (+25%)**. Wall A/B 3 pairs: median +15.8s (+18%). ncu: regs 56 → 72 (+16), block_limit 5 → 4, per-launch 1.20 → 2.59 ms (+116%), zero spills. | ✗ | Same lesson as T3, opposite direction: the math chain genuinely needs its 56 registers. Inlining scatter logic into trans paid for the saved kernel twice over in lost occupancy. Direct-atomic rescue variant also stuck at 72 regs — the growth is from persistent bucket-arg state + bucket_storage write, not the aggregation. |
| 12 | 2026-05-25 | Bucket-hash salt sweep (3 alternative `BUCKET_HASH_MIX` multipliers) | All byte-exact polynomial parity. Slow-tail count varied only 7.5% across 4 well-chosen 64-bit multipliers. Golden-ratio default `0x9E3779B97F4A7C15` is best. | ✗ | Slow buckets are **structurally intrinsic to the key distribution** (special-q lattice geometry), not hash-induced. Different multipliers shuffle non-colliders between buckets but the proportion of "hard" buckets stays roughly constant. |
| 13 | 2026-05-26 | Filter ncu at scale (coeff 60060 vs coeff 420 head-to-head) | Both scales **identical SOL**: duration 1.03/1.04 ms, DRAM 42%, L2 16%, theoretical occupancy 25% (shmem-limited at 3 blocks/SM via 24.58 KB dynamic shmem/block), 38% of issue cycles in L1TEX scoreboard stalls. | — (diagnostic) | Filter is **shmem-limited-occupancy + L1TEX-scoreboard-latency-bound**, not L2-throughput-bound (that was scatter_roots). The 5.1% → 8.8% growth in slow-tail proportion at coeff 60060 does NOT shift the per-launch bottleneck shape — just adds cycles in the same stall pattern. Closes "does filter behave differently at scale?" |

## Closed levers and why

After 13 experiments across the trans, scatter, and filter kernels, this is the cumulative picture:

**Trans kernel (37% of GPU time):** at its math-chain-bound ceiling.
- T3 (lower regs → more occupancy) hurts because the math chain spills state into memory ops.
- Tail-effect alignment is essentially a no-op because ncu's "25% tail effect" estimate is the worst-case ceiling, not the realistic delta — at 3.96 waves with 96.25%-full last wave, actual tail waste is ~1%.
- Modinv-amortization (reducing blocks_y to give each thread more specialqs) saves <1% because the modinv share is small relative to the per-specialq forward/backward montmul loops.
- Compute-side levers (T1 unroll, T2 templates, T4 modinv64 rewrite) untried but unlikely to move the needle by more than 1-2% given the math chain saturation.

**scatter_roots_kernel (34% of GPU time):** at the L2-throughput ceiling (~77%).
- Standalone scatter is bounded by 280 MB writes to `bucket_storage[]` (445 MB working set vs 40 MB L2). Per-line merging is already saturated.
- A1 cache hint and A.5 direct-atomic both confirmed: no in-kernel restructure can reduce per-key transaction count meaningfully.
- LOG2_NUM_BUCKETS sweep confirmed 14 is the optimum balance between scatter and filter work.

**filter_per_bucket_kernel (19% of GPU time):** shmem-limited-occupancy + L1TEX-scoreboard-latency-bound; also slowest-block-bound at the workload level.
- Per-launch SOL (identical at coeff 420 and 60060 — confirmed by paired ncu captures): theoretical occupancy 25%, achieved 24.6%, capped at 3 blocks/SM by 24.58 KB dynamic shmem/block (the per-bucket hash tables). DRAM 42%, **L2 only 16%** (not L2-bound), L1TEX scoreboard stalls eat 38% of issue cycles.
- `MAX_FILTER_ITERS=20` is dead-letter (max observed iter 13, single block out of 204M).
- Convergence-rule loosening (variant B, 3-iter → 2-iter window) shifted the bulk of iter-4 buckets down to iter 3 but saved zero wall — the iter-5+ tail still dictates duration. The candidate growth (+2.88%) shifted cost into count_and_store.
- Slow buckets are NOT oversized (tail-bucket-size diagnostic confirmed at both coeff 420 and 60060). They're slow because their keys cluster in hash subspaces.
- Salt sweep confirmed the clustering is structural to the key distribution, not hash-induced.
- Theoretical levers that could move the per-launch SOL — both rejected:
  - Smaller per-bucket hash table → more blocks/SM → higher occupancy. But smaller table = more false positives = candidate growth = same Phase-C cost-shift failure mode variant B hit.
  - Bring `bucket_storage` data into shmem during filter loop. Doesn't fit at current bucket sizes (~2400 × 8 bytes = 19 KB per bucket plus the hash table blows the shmem budget).

**Option A fusion (inline trans + bucket-scatter):** ruled out by register pressure.
- Bringing the scatter compute into trans grows registers 56 → 72, dropping block_limit_registers 5 → 4 (lost a block/SM). Per-launch duration doubles. Net regression worse than the 23s scatter saving.
- Direct-atomic rescue variant didn't help — the register growth is from persistent bucket-arg state, not the warp aggregation.
- The design (active-mask, exact-stored-key emit, ONLY-final-write-sites guidance, shared bucket header, overflow fallback through scatter_roots) was correctness-clean; the lever is just intrinsically wrong-direction.

## Future directions

**Option B (full restructure):** the only remaining lever in this design space. Eliminate `gpu_root_array` entirely; pack `(key, value)` into `bucket_storage` (12 bytes/slot vs current 8); rewrite `count_and_store_matched_values_kernel` to walk buckets instead of scanning the trans output. Memory delta: ~700 MB per batch eliminated.

Effort: **2-4 weeks of focused work** with uncertain ceiling. Caveats:
- The trans kernel currently uses `gpu_root_array` as scratch during the qq_prod walkback (lines ~500-510 in `stage1_core.cu`), so Option B has to solve a scratch/state problem too.
- count_and_store's bucket-walk replacement has unknown perf vs the current linear scan (depends on per-bucket hash lookup structure).

**Not pursued right now.** The Gerbicz engine + B1 arena + memset cleanup already delivered the big win (~40% wall improvement vs CUB sort). Diminishing returns are clear from the closed levers and the coeff-60060 scale diagnostics.

**Small things still worth doing if curiosity strikes:**
- Compute-side trans tuning (T1 unroll factor sweep, T2 per-num_roots templates, T4 modinv64 rewrite). Each likely 1-2%, but cumulative could matter.
- Wider salt-sweep with more exotic multipliers, or a hash that XOR-folds high+low bits to fight key clustering — gambling odds.
- Additional ncu only for materially different workloads (larger N, different degree, or a new GPU architecture); coeff 60060 already confirmed the current kernel landscape at scale.

## Reference appendix

### Parity fingerprints

- Polynomial md5 (coeff 420, gerbicz path, canonical record-sort): **`ef7d7676615235c41caa228698940511`** (12,739 polynomials)
- Polynomial md5 (coeff 420, sort path): same — both paths produce identical poly sets
- Parity baseline files: `profiling/msieve.gerbicz.dat.ms`, `profiling/msieve.normal.dat.ms`

### Collstats baselines

**Coeff 420 (gerbicz, post-memset, post-B1):**
- batches: 12,456
- engine: ~56s (with collstats; ~70s without arena)
- max_bucket: 2,414
- candidates: 69,228,524
- dedup: 17,344,961
- matched: 35,196,265
- found: 12,739
- arena_attempts: 12,456 / arena_fallbacks: 3 / arena_capacity_skips: 0
- bucket_grows: 0 / hash_caps: 0

**Coeff 60060 (gerbicz, post-memset, post-B1):**
- batches: 92,454
- engine: ~408s
- max_bucket: 2,417
- candidates: 818,966,939
- dedup: 338,682,731
- matched: 684,476,386
- found: 217,073
- arena_attempts: 92,454 / arena_fallbacks: 0 / arena_capacity_skips: 0

### Filter iter histogram baseline (coeff 420)

Total stops: 204,079,104 (= 12,456 batches × 16,384 buckets)

| Stop iter | Total | Zero (cnt==0) | Converged | Cap (hit MAX) |
|---:|---:|---:|---:|---:|
| 1 | 105,519,247 | 105,519,247 | 0 | 0 |
| 2 | 27,756,836 | 27,756,836 | 0 | 0 |
| 3 | 36,623,471 | 36,623,471 | 0 | 0 |
| **4** | **23,819,369** | 1,503,977 | **22,315,392** | 0 |
| 5 | 4,212,690 | 1,639 | 4,211,051 | 0 |
| 6 | 5,897,059 | 360 | 5,896,699 | 0 |
| 7 | 246,512 | 2,576 | 243,936 | 0 |
| 8 | 3,100 | 8 | 3,092 | 0 |
| 9-13 | 820 | 0 | 820 | 0 |

- Max observed iter: 13 (single block out of 204M)
- `MAX_FILTER_ITERS=20` cap is dead-letter
- Convergence rule (`it >= 3 && s_nsize[it-3] == s_nsize[it]`) becomes evaluable for the first time at iter 4 → the 22.3M "convergence elbow"

### Bucket-size by stop category (coeff 420, log2 bins)

13-bin log2 histogram. Bins cover {1, 2, 4, ..., 4096+}. Bucket counts cap at `max_per_bucket = 2,414` so only bins 10-11 fire materially.

| Category | Bin 1024+ | Bin 2048+ | Total |
|---|---:|---:|---:|
| Fast (it ≤ 3) | 7.65M (4.5%) | 162.25M (95.5%) | 169.9M |
| Medium (it == 4) | 0.63M (2.6%) | 23.19M (97.4%) | 23.8M |
| Slow (it ≥ 5) | 0.67M (6.5%) | 9.69M (93.5%) | 10.4M |

**Slow buckets are NOT oversized** — distribution within ±2pp of fast buckets, and at coeff 60060 slow buckets are *slightly smaller* on average. Closes the "split oversized buckets" lever.

### Cached `blocks_per_sm` table (pp64_r64 on sm_120, RTX 5070)

From `cuOccupancyMaxActiveBlocksPerMultiprocessor` queried at init time during the (reverted) tail-effect experiment. Useful for any future occupancy-driven decision:

| size_x | 128 | 160 | 192 | 224 | 256 |
|---:|---:|---:|---:|---:|---:|
| blocks/SM | 9 | 7 | 6 | **5** | 4 |

Dominant launch is size_x=224 → 5 blocks/SM × 48 SMs = 240 blocks/wave. Coeff 420 grid is 951 blocks ≈ 3.96 waves.

### Reference artifacts in `profiling/`

| File | What |
|---|---|
| `profile_gerbicz_420.{nsys-rep,sqlite}` | Initial post-integration baseline (pre-B1, pre-memset) |
| `profile_gerbicz_420_arena.{nsys-rep,sqlite}` | After B1 arena landed |
| `profile_gerbicz_420_log15.{nsys-rep,sqlite}` | LOG2_NUM_BUCKETS=15 sweep point |
| `profile_gerbicz_420_memset.{nsys-rep,sqlite}` | After memset cleanup (current baseline shape) |
| `trans_full.ncu-rep` | Trans pp64_r64 full ncu, coeff 120 (early compute-bound regime) |
| `trans_full_medium.ncu-rep` | Trans medium-launch ncu (latency-bound regime) |
| `trans_full_post_memset.ncu-rep` | Trans dominant launch coeff 420, post-memset |
| `trans_full_post_memset_60060.ncu-rep` | Trans dominant launch coeff 60060 |
| `trans_full_t3.ncu-rep` | T3 `__launch_bounds__(256,6)` capture |
| `filter_420.ncu-rep` | filter_per_bucket_kernel ncu on coeff 420 dominant launch |
| `filter_60060.ncu-rep` | filter_per_bucket_kernel ncu on coeff 60060 (head-to-head with 420 — identical SOL) |
| `scatter_roots_ncu.ncu-rep` | scatter_roots baseline ncu |
| `scatter_roots_ncu_stcs.ncu-rep` | A1 `__stcs` cache hint capture |
| `scatter_roots_ncu_direct.ncu-rep` | A.5 direct-atomic capture |
| `sort_sol.ncu-rep` | Original CUB sort onesweep ncu (pre-Gerbicz path) |
| `trans_sol.ncu-rep` | Trans speed-of-light pass |
| `msieve.gerbicz.dat.ms` | Polynomial parity baseline (md5 `ef7d7676...`) |
| `msieve.normal.dat.ms` | Sort-path polynomial baseline (same md5) |

### Source pointers

- Trans kernel: `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu` (pp32_r32 ~23, pp32_r64 ~219, pp64_r64 ~419)
- Trans launch host: `gnfs/poly/stage1/stage1_sieve_gpu.c:765` (`handle_special_q_batch`)
- Collision engine: `cub/collision_engine.{cu,h}` (kernels, `collision_engine_run`)
- Shared bucket constants: `cub/collision_bucket.h`
- B1 arena fast path: search for `count_and_store_matched_values_kernel` + `emit_found_arena_kernel` in `cub/collision_engine.cu`
- collstats accumulators and printing: `stage1_sieve_gpu.c` (`device_thread_data_t` + `check_found_array` + the `logprintf` block around line 1075)
- gerbicz_bench standalone harness: `gerbicz_bench/` (separate Makefile, independent of msieve build — sandbox for algorithm prototyping)

### mersenneforum / Gerbicz algorithm references

Background reading for the collision-detection approach implemented in `cub/collision_engine.cu`:
- The algorithmic reference implementation: `gerbicz_bench/collision_ref.c` (Gerbicz's CPU prototype, source line numbers cited throughout the engine code)
- mersenneforum.org threads on the Gerbicz / Papadopoulos / Broman polyselect work (collision-detection vs sort-based) — predates this codebase

---

## Git history

Original plan docs that were consolidated here:
- `POLYSELECT_OPTIMIZATION_NOTES.md` (original 43K — initial profiling, integration plan, post-integration history)
- `TRANS_KERNEL_PLAN.md` (28K — T3 / tail-effect plan + v2.1/v2.2 status)
- `TWO_LEVEL_SCATTER_PLAN.md` (36K — scatter A1/A.5 + filter variant B + tail-bucket diagnostics v1-v7)
- `FUSED_TRANS_SCATTER_PLAN.md` (34K — fusion design v0-v3 with Codex review threads)

All preserved in git history before this consolidation commit. Reach for them if you need the full step-by-step plans, risk tables, or implementation sketches for reverted code.
