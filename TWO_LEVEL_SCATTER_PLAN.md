# Scatter-roots optimization plan

> **Status (2026-05-24):**
> - **v1:** original two-level scatter plan, based on inferred metrics.
> - **v2:** rewritten after targeted ncu profiling showed the kernel is **L2-throughput-ceiling-bound**, not DRAM-bound and not instruction-bound. Original Phase 1/2/3 framing preserved as appendix.
> - **v3:** experiments A1 (`__stcs`) and A.5 (direct-atomic) both ran with byte-exact parity but no perf win. A.5 confirmed the L2 ceiling is the wall — removing 25 pp of compute throughput didn't move duration. **Conclusion: no in-kernel restructure of `scatter_roots_kernel` will substantially help.** Pivot to Experiment B (count/scatter fusion).
> - **v4:** Experiment B bounded-arena fast path implemented with `MATCH_ARENA_WIDTH=8`. Full coeff-420 validation under `cuda-gdb` produced byte-exact parity against `msieve.gerbicz.dat.ms` and reduced collision-engine time to **75.070s** from the documented **81.6s** baseline. A separate crash was seen only in an alternate-output validation harness using `-s`/`-l`; the normal invocation path is not blocked.
> - **v5:** nsys recapture confirms B removed the old second matched-value scan in the common case. Value-path GPU kernel time dropped **11.698s -> 6.181s** on coeff 420, `scatter_matched_values_kernel` dropped from **12,456 launches to 3 fallback launches**, and CUB scan ranges halved. Whole-profile kernel time moved **71.665s -> 69.404s**; unrelated trans/scatter/filter kernels were slower in this capture, so the cleanest claim is the targeted **5.5s value-path saving** plus a collstats engine time of **65.821s** in the profiled run.
> - **v6:** Memset cleanup landed. The per-batch 267 MB `gpu_root_array` clear at `stage1_sieve_gpu.c:696` was 98.7% of all memset time (5.48s of 5.6s on coeff 420). Skipped it when `num_aprog_vals == 1` and refined the three `sieve_kernel_trans_*` variants in `stage1_core.cu` so plane 0 is fully written even for zero-`qq_prod` slots. Coeff 420 collstats now match B1 baseline exactly (12,456 batches, 69,228,524 candidates, 17,344,961 dedup, 35,196,265 matched, 12,739 found). Memset total GPU time dropped **5.598s -> 0.121s** (−97.8%). Wall **1m39s -> 1m35s** (~4% wall improvement). Coeff 60060 also exact: engine **469.080s -> 418.534s**, all counters match.
> - **v7 (current, 2026-05-25):** Phase B filter-aggressiveness sweep ran and was **NEUTRAL**. Added stop-iteration histogram (gated behind `collstats=1`, ~0% overhead off) with 3-way breakdown: total / cnt==0 / converged / cap-hit. Coeff 420 histogram showed `MAX_FILTER_ITERS=20` cap is **dead-letter** (max observed iter 13, single bucket); iter-1/2/3 stops are 100% `cnt==0` (convergence rule literally can't fire at it<3); the convergence rule fires for the first time at iter 4 with **22.3M buckets** (the "convergence elbow"). Tested variant B (`it >= 2 && s_nsize[it-2] == s_nsize[it]` — loosens stability window from 3 to 2 iters). All gates passed: byte-exact polynomial parity, arena_fallbacks unchanged at 3, dedup/matched/found EXACTLY identical (extra candidates were all dedup duplicates). Per-kernel nsys: filter -1.1% (-0.14s), count_and_store +4.5% (+0.26s, candidate growth absorbed downstream), **net effectively flat**. **Lesson: Phase B is slowest-block-bounded, not average-bounded** — shifting the bulk of iter-4 converged buckets down to iter 3 saved nothing on wall because the long-tail blocks (iter 5,6) still dictated kernel duration. Variant B reverted; histogram instrumentation kept as a permanent diagnostic. Future filter ideas need to attack the tail (oversized buckets), not the average. **Sweep retired.**

## Goal

Original goal: reduce `scatter_roots_kernel` GPU time from **31.2%** of total GPU time toward **~15-20%**. Concrete per-launch target: drop from 1.795 ms -> <= 1.2 ms. Expected total wall improvement if hit: **5-12%** on the engine path.

Current status: scatter-local work is deprioritized after A1/A.5. The active lever is reducing fixed full-array passes elsewhere in the collision engine, starting with Experiment B's matched-value count/scatter fusion.

## Ground truth: ncu profile of `scatter_roots_kernel`

Captured 2026-05-24 via `ncu --set basic -k regex:scatter_roots_kernel -c 5` on coeff 420, LOG2_NUM_BUCKETS=14. Five captures, all within ±2% — numbers below are representative.

| Section | Metric | Value | Read |
|---|---|---|---|
| **SOL** | Memory throughput | **76-78%** | Memory-bound |
| | DRAM throughput | 50-52% | DRAM has headroom |
| | L2 cache throughput | **76-78%** | **L2 is the wall** |
| | Compute (SM) throughput | 39% | Not instruction-bound |
| | L1/TEX throughput | 67% | Significant traffic but not the bottleneck |
| **L2 breakdown** | L2 D Atomic Input Cycles Active | **36.87%** | Atomic ops on `bucket_count[]` consume large slice of L2 |
| | L2 T Sectors / D Sectors | 42% / 35% | Sector traffic (the `bucket_storage[]` writes) is the rest |
| **Launch** | Grid × block | 136,603 × 256 | **N = 34.97M keys per launch** (not 10M as plan assumed) |
| | Registers/thread | 18 | Low — substantial headroom for richer kernels |
| | Shared mem/block | 0 (dynamic) | Currently unused |
| | Theoretical occupancy | 100% | |
| | **Achieved occupancy** | **85.5%** | Already high — any variant that loses occupancy is risky |
| | Block limit shared mem | 16 blocks | Driving factor if shared mem is added |
| | # SMs / # TPCs | 48 / 24 | RTX 5070 |
| | Duration | 1.70-1.72 ms | Matches nsys avg 1.795 ms |

**The headline finding:** the kernel is L2-bandwidth-bound. Memory throughput 76-78% with DRAM only 50-52% means L2 traffic — not DRAM bandwidth — is what's keeping the kernel from going faster.

> **Reviewer note:** This reframing is the right one. Treat any scatter-local improvement as provisional until the full engine is profiled, though: `bucket_storage` is not dead output. Phase B reads `arr_a` immediately after scatter, so a store policy that makes scatter faster can still be a net loss if it evicts useful lines or changes write-combining behavior before `filter_per_bucket_kernel`.

## What this changes vs the original analysis

| Original assumption | Reality |
|---|---|
| N ≈ 10M keys/launch | **N ≈ 35M keys/launch** (3.5× higher) |
| DRAM writes are the wall, coalescing is the lever | **L2 is the wall**; cache policy / atomic pressure / coalescing all flow into the same L2 budget |
| Atomic latency on `bucket_count[]` is masked | Atomics consume **~37% of L2 active cycles** — they're a real fraction of the bottleneck |
| Phase 1 privatized counter could win 5-10% | Phase 1 occupancy hit (~14× drop) outweighs any plausible atomic-count savings |
| Phase 2 at T=16K → ~625 blocks/launch, fills GPU well | At N=35M: T=16K → 2,138 blocks/launch but **1 block/SM** at 64 KB shared mem → ~44 waves × ~110 µs/block ≈ **4.9 ms regression** |
| Current `__match_any_sync` warp aggregation is a perf win | Probably a wash at 16K buckets (~0.03 dup pairs/warp), but the kernel is memory-bound so trimming it doesn't free L2 bandwidth |

## What's ruled out (and why)

1. **Original Phase 1 (privatized shared-mem counter, 64 KB/block).** Sparsity math: 256 keys into 16K buckets → only ~2 duplicate buckets per block. Privatization can eliminate at most ~2 global atomics per block. The cost is 64 KB shared/block, which drops occupancy from 85.5% to ~6% (block-limit-shared-mem goes from 16 to 1). Net regression highly likely.

2. **Direct-atomic kernel (drop warp aggregation).** Tempting because `__match_any_sync` saves so few atomics at 16K buckets, but the kernel is memory-bound at 76% L2 throughput. Instruction trims don't free L2 bandwidth. Compute throughput is only 39% — there's nothing to gain on the SM side.

   > **Reviewer note:** I agree this should not lead the work, but it remains a cheap control experiment if Experiment A is a no-op. The value would be diagnostic more than strategic: confirm whether warp-collective overhead is truly invisible at this bucket density.

3. **Original Phase 2 at T=16K with 64 KB+ shared mem.** Wave-budget regression at the actual N=35M. Kernel would run slower, not faster.

4. **Bucket-count retuning.** Already swept; confirmed LOG2_NUM_BUCKETS=14 is optimal (POLYSELECT_OPTIMIZATION_NOTES.md "Bucket-count sweep").

5. **A1 `__stcs` cache hint** (empirically ruled out). Functions correctly (L2 sector traffic drops 2-3pp) but doesn't reduce duration. L2 atomic cycles dropped too, indicating atomics weren't on the critical path. See Experiment results below.

6. **A.5 direct-atomic kernel** (empirically ruled out). Removes 25pp of compute throughput, doesn't change duration. Confirms the kernel is L2-throughput-ceiling-bound, not instruction-bound or atomic-bound. See Experiment results below.

7. **Experiment C (revised Phase 2, block-sort + grouped flush)** (retired). A.5 showed that reducing per-key transaction count locally doesn't help; the L2 ceiling is set by total transaction count on a 445 MB working set, and the block-sort restructure doesn't reduce that.

## The actual lever: L2 cache policy on the write

The `bucket_storage[]` array is **~445 MB** (16,384 buckets × ~3,400 slots × 8 bytes). L2 on RTX 5070 is ~40 MB. Each launch writes 35M × 8 = **280 MB** to this array. Two consequences:

- L2 hit rate on the writes is bounded by ~40/280 ≈ 14% even in the best case. Most writes miss L2.
- But the writes still traverse L2 sector caches on their way to DRAM, consuming L2 bandwidth — bandwidth that competes with the atomic ops on `bucket_count[]`.

If we mark the writes as **streaming** (evict-first cache policy, or write-through bypassing L2 allocation), L2 sector traffic drops sharply and the atomics get more bandwidth. This is essentially free to try — one-line change, no semantic effect, no occupancy impact.

> **Reviewer note:** The `40/280` cache-hit bound is a useful sanity check but not a complete model. The writes advance sequentially within each bucket, so the active "current cache line per bucket" footprint is roughly `NUM_BUCKETS * line_size`, which can fit in L2. That means normal stores may be getting useful line merging before Phase B reads the same buffer. Expect `__stcs` to be plausible, but treat `__stwt` as high-risk because true write-through/no-allocate behavior can defeat that merging and push more pressure to DRAM.

## Action plan

### Experiment A: cache-streaming hint on `bucket_storage` write — **completed, negative**

**Escalation sequence (do these in order, not as alternatives — each step is gated on the previous):**

**A1 — `__stcs` intrinsic (lowest risk, start here).** Change `cub/collision_engine.cu:151`:
```cuda
bucket_storage[(size_t)bucket * max_per_bucket + slot] = k;
```
to:
```cuda
__stcs(&bucket_storage[(size_t)bucket * max_per_bucket + slot], k);
```
Streaming hint: line is still allocated in L2 (so write-merging within hot head-of-bucket lines is preserved — see Reviewer Note 3 above) but is marked for first eviction. This is the safe lever.

**A2 — inline PTX `st.global.cs.u64` (only if A1 helps, as a verification).** Equivalent to A1 but explicit. Run only to confirm the compiler emitted the cache-streaming op as expected; if A1's SASS is already correct, skip A2.

**A3 — `__stwt` (only if A1 helps materially and we want more, and only with a DRAM-side check).** Write-through, no L2 allocation. **Higher risk:** may defeat the line-merging that 8 consecutive 8-byte writes into one cache line currently benefit from (16K buckets × 64-byte lines ≈ 1 MB hot footprint, fits L2). Could turn 1 line-burst into 8 partial-line DRAM transactions and slam DRAM. Run only after A1 has shown the framing is correct, and only with explicit before/after DRAM-throughput comparison.

**Mechanism:** these all reduce L2 bandwidth consumed by the store traffic on `bucket_storage[]`, freeing L2 for the atomics on `bucket_count[]` (currently 36.87% of L2 active cycles).

**Effort:** ~30 min for A1 including parity check. A2/A3 only if A1 lands.

**Expected win:** 10-25% on scatter duration if writes currently default to write-back; 0% if sm_120 already uses streaming policy. ncu will tell us either way.

**Validation (for each step):**
1. Rebuild (`make CUDA=1 all`) per `msieve_build_command.md`.
2. Parity diff against the LOG2=14 baseline `.ms` (12,739 polynomials on coeff 420). **Must be byte-exact after canonical record-sort.**
3. Re-run ncu — compare `L2 D Sectors`, `L2 Cache Throughput`, **and `DRAM Throughput`** (the DRAM check is critical for A3) and per-launch duration.
4. Re-run nsys — compare `scatter_roots_kernel`, `filter_per_bucket_kernel`, and full collision-engine time. Do not ship based on scatter time alone.
5. If ≥5% full-engine improvement and parity holds: ship it, stop here.

### Experiment A.5: diagnostic direct-atomic kernel — only if A1 is a no-op

If A1 changes neither scatter duration nor L2 metrics, the writes were already streaming by default — but we still don't know why scatter is at 76% L2 throughput. Run a one-off `scatter_roots_direct_kernel` variant: drop `__match_any_sync` and friends, do `slot = atomicAdd(&bucket_count[bucket], 1)` directly. **Diagnostic value only:** confirms whether warp-collective overhead is genuinely invisible at 16K buckets (expected: yes, kernel is memory-bound; observed regression would suggest atomic pressure is more nuanced than the ncu breakdown shows). Result either way redirects us to Experiment B without further scatter work.

**Effort:** ~1 hour including parity. No flag plumbing needed — gate via a compile-time `#ifdef`.

### Experiment B: bounded per-slot arena fast path — **implemented, nsys-confirmed**

`count_matched_values_kernel` and `scatter_matched_values_kernel` are **8.2% + 8.1% = 16.3% of GPU time** combined, both with N-scan structure over the same input arrays. A bounded per-slot arena eliminates the second full pass, the value-count exclusive scan, and the `d_value_cursor` copy in the common case.

**Chosen structure:** bounded per-slot arena fast path + exact fallback.

1. Clear `d_value_counts`, `d_value_match_total`, and `d_value_overflow`.
2. Launch a fused count/store kernel. It does the same candidate lookup as `count_matched_values_kernel`, increments `value_counts[slot]`, and stores the first `MATCH_ARENA_WIDTH` packed values at `matched_values[slot * MATCH_ARENA_WIDTH + pos]`.
3. If no per-slot arena overflow occurs, launch an arena emit kernel that reads each slot's `value_counts[slot]` directly. This skips the old exclusive scan, cursor copy, and `scatter_matched_values_kernel`.
4. If any slot exceeds the arena width, keep exactness by using the already-correct `value_counts[]`, running the existing exclusive scan and `scatter_matched_values_kernel`, then using the existing `emit_found_kernel`.

The fast path is gated by capacity: `dedup_cnt * MATCH_ARENA_WIDTH <= VALUE_MATCH_CAP`. If this fails, use the old path unchanged.

**Why this is safe:** observed `matched/dedup ≈ 2.0`, so nearly all slots should fit in a small arena. The fallback preserves exact output for heavy slots or unusual coefficients.

**Effort:** medium (1-2 days). Keep the old path intact and use it as the fallback.

**Expected win:** 4-7% wall (less than the additive 16% because fused kernel won't be 2× faster, only ~30-50% of the combined cost).

**Why this beats restructuring scatter:** lower risk, direct transaction-count reduction on a confirmed fixed-cost pair of N-scans, and exact fallback when the arena is too small.

### Experiment C: revised Phase 2 (block-sort + grouped flush) — speculative, only if A and B don't deliver enough

If A + B together don't get us to the 15-20% target, the scatter coalescing path is the next option. Original Phase 2 has to be rebuilt for the actual N:

**Constraints from ncu:**
- Must keep ≥4 blocks/SM to stay near current 85% occupancy → shared mem ≤ 16 KB/block.
- Block-limit-shared-mem on RTX 5070: 16 KB shared/block × 16 blocks = ~256 KB max per SM; at 4 blocks/SM that's 64 KB shared per SM in use.
- 18 reg/thread current → can grow to ~32 reg/thread for `BlockRadixSort` state without spilling.

**Recommended config:**
- BLOCK_THREADS=512, KEYS_PER_THREAD=8 → **T = 4,096 keys/block**
- 14-bit radix sort only (sort by `bucket` field, drop the high key bits — they're not needed for grouping)
- Shared mem budget: BlockRadixSort temp + small `s_base[]` for group bases → target 12-16 KB total
- At N=35M, T=4K: **8,536 blocks/launch**; at 4 blocks/SM × 48 SMs = 192 concurrent → ~45 waves
- Expected per-block: ~50 µs (radix sort 14-bit on 4K keys is fast on Blackwell)
- Total: ~45 × 50 µs ≈ **2.25 ms** — still worse than current

**This is concerning.** The wave math says even the revised Phase 2 may not win. At T=4K the coalescing factor is ~1.10 (per the original plan's table) — barely any benefit, so the sort cost has to come from somewhere else (faster DRAM utilization?) and that's not obvious.

**Conclusion for Phase 2:** likely not worth the engineering cost given the wave math. Mark as deprioritized; revisit only if A + B fall meaningfully short and there's no better lever.

## Experiment results (2026-05-24)

### A1: `__stcs` cache-streaming hint — **NEGATIVE (no win)**

| Metric | Baseline (5 runs) | A1 `__stcs` (5 runs) | Δ |
|---|---|---|---|
| Duration (median) | 1.71 ms | 1.80 ms | +5% (slight regression) |
| Duration (mean) | 1.71 ms | 1.98 ms | +16% (skewed by 2 outliers) |
| L2 Cache Throughput | 76.57% | 73.58% | **-3.0 pp** |
| L2: T Sectors | 42.73% | 40.10% | -2.6 pp |
| L2: D Sectors | 34.80% | 32.77% | -2.0 pp |
| **L2: D Atomic Input Cycles** | **36.87%** | **35.16%** | **-1.7 pp** |
| DRAM throughput | 51.94% | 49.37% | -2.6 pp |
| Compute (SM) | 38.85% | 37.04% | -1.8 pp |
| Achieved occupancy | 85.52% | 84.99% | flat |
| Full-engine wall (1 run) | ~1m39s (per memory) | 1m50s | ~+11s |

Parity: byte-exact (12,739 polynomials, canonical record-sort identical).

**Interpretation:** `__stcs` is functioning — L2 sector traffic dropped 2-3pp in the expected direction. But every metric moved down in lockstep, including duration which stayed flat or went slightly up. **L2 atomic cycles went DOWN, not up** — if atomics had been the limit and we'd freed L2 for them, atomic % should have risen. Strong evidence the bucket_count atomics were never on the critical path, even though they consume 37% of L2 active cycles.

### A.5: direct-atomic diagnostic — **NEGATIVE but informative**

Replaced warp-aggregated atomic (`__match_any_sync` + leader atomicAdd + `__shfl_sync`) with a direct `atomicAdd(&bucket_count[bucket], 1)` per thread. Gated by `#ifdef SCATTER_DIRECT_ATOMIC` in `cub/collision_engine.cu`. Built with `nvcc -DSCATTER_DIRECT_ATOMIC ...`.

| Metric | Baseline | A.5 direct-atomic | Δ |
|---|---|---|---|
| Duration (median) | 1.71 ms | 1.71 ms | **flat** |
| Duration (4 of 5 captures) | 1.70-1.72 ms | 1.69-1.71 ms | flat |
| L2 Cache Throughput | 76.57% | **77.86%** | +1.3 pp |
| L2: T Sectors | 42.73% | 42.42% | flat |
| **L2: D Atomic Input Cycles** | **36.87%** | **37.19%** | **flat (atomic count up 32×, L2 cycles flat)** |
| DRAM throughput | 51.94% | 52.12% | flat |
| **Compute (SM) Throughput** | **38.85%** | **13.62%** | **-25 pp (huge drop)** |
| Registers/thread | 18 | 16 | -2 |
| Achieved occupancy | 85.52% | 86.62% | +1.1 pp |

Parity: byte-exact (12,739 polynomials).

**Interpretation:** This is the key diagnostic result. Removing the warp-aggregation logic dropped compute throughput by **25 percentage points** (from 39% to 14%) — i.e., the warp aggregation was ~64% of the kernel's instruction load. **Duration didn't change at all.** That tells us:

1. **The kernel is genuinely memory-bound (L2-throughput-limited).** All those instructions are hidden behind the L2 wait.
2. **Warp aggregation is doing essentially zero useful work at 16K buckets.** 32× more global atomics per warp moved L2 atomic input cycles by only +0.32pp.
3. **The L2 throughput ceiling (~77%) is a hard wall for this kernel.** It isn't atomics, it isn't instructions, and it isn't write coalescing — all those move freely without changing duration.

### What this means for the action plan

**No in-kernel restructure of `scatter_roots_kernel` will substantially help.** The L2 ceiling appears to be set by the total transaction count (atomics + writes) on a 445 MB working set against a 40 MB L2, and the only way under it is to issue fewer total transactions. That can't be done locally.

Three remaining options ranked by expected wall win:

1. **Pivot to Experiment B (count/scatter fusion)** — reduces one full N-scan + intermediate buffer. Direct attack on transaction count for the *other* 16% of GPU time, not scatter. Most likely to pay.
2. **Memset cleanup** (8% GPU time in tiny counter clears). Easy, ~1-3% wall.
3. **Trans math-chain restructure** (34.3% of GPU time, 30% Montgomery dependency stalls). High ceiling, high effort.

Experiment C (block-sort scatter) is now retired entirely — even if the wave math worked out, A.5 shows that reducing the kernel's per-key transaction count isn't the lever. Don't pursue.

**A2 and A3 are skipped.** A2 (PTX) was only useful as verification that A1 emitted the right op; since A1 emitted the right op and didn't win, no value in A2. A3 (`__stwt`) would aggravate the line-merging concern (Reviewer Note 3) without any signal it would help, since A1 already shows duration is L2-throughput-ceiling-bound, not write-bandwidth-bound.

### B1: bounded per-slot arena fast path — **IMPLEMENTED, parity + nsys + diagnostics pass**

Implemented in `cub/collision_engine.cu`:

- `MATCH_ARENA_WIDTH=8`.
- New `count_and_store_matched_values_kernel`: one N-scan that both counts matches and stores the first 8 packed values per dedup slot into `matched_values[slot * MATCH_ARENA_WIDTH + pos]`.
- New `emit_found_arena_kernel`: emits directly from the per-slot arena when no slot overflowed.
- Exact fallback: if `dedup_cnt * MATCH_ARENA_WIDTH > VALUE_MATCH_CAP` or any slot exceeds width 8, reuse the exact `value_counts[]`, run the old exclusive scan and `scatter_matched_values_kernel`, then use the old `emit_found_kernel`.
- Cheap arena diagnostics in `collstats=1`: `arena_attempts`, `arena_fallbacks`, and `arena_capacity_skips`. These are host-side branch counters, not an extra GPU reduce.

Fallback correctness detail: `count_and_store_matched_values_kernel` increments `value_counts[slot]` for every matched value, even when the bounded arena overflows. That means the old fallback scan/scatter path sees complete counts and can rebuild the exact compact `matched_values[]` layout. Any stale bounded-arena entries are ignored by `emit_found_kernel`, which reads only the exclusive-scan ranges.

Validation:

| Check | Result |
|---|---|
| Build | `make CUDA=1 all` passed |
| Full coeff-420 run | completed under `cuda-gdb` |
| Output size | 12,739 polynomial records |
| Canonical parity | byte-exact vs `msieve.gerbicz.dat.ms` |
| Collision stats | `batches 12456`, `candidates 69228524`, `dedup 17344961`, `matched 35196265` |
| Engine time | **75.070s** |
| Found stats | `peak 11`, `saturated 0`, `total 12739` |

Baseline state match:

| Metric | LOG2=14 baseline | B1 arena | Result |
|---|---:|---:|---|
| Batches | 12,456 | 12,456 | exact |
| Max bucket | 2,414 | 2,414 | exact |
| Candidates | 69,228,524 | 69,228,524 | exact |
| Dedup keys | 17,344,961 | 17,344,961 | exact |
| Matched values | 35,196,265 | 35,196,265 | exact |
| Found records | 12,739 | 12,739 | exact |

nsys recapture:

Command:

```bash
nsys profile --stats=true --force-overwrite=true --output profile_gerbicz_420_arena \
  ./msieve -np1 -nps "min_coeff=420 max_coeff=420 collengine=gerbicz collstats=1"
```

Artifacts: `profile_gerbicz_420_arena.nsys-rep` and `profile_gerbicz_420_arena.sqlite`.

| Metric | LOG2=14 baseline | B1 arena | Delta |
|---|---:|---:|---:|
| Collstats engine time | 81.6s documented baseline | 65.821s in nsys run | -15.8s observed |
| Total CUDA kernel time | 71.665s | 69.404s | -2.261s |
| Old value path kernels | 11.698s | n/a | n/a |
| Arena value path kernels | n/a | 6.181s | -5.517s vs old path |
| `count_matched_values_kernel` | 5.866s / 12,456 calls | replaced | - |
| `count_and_store_matched_values_kernel` | n/a | 6.128s / 12,456 calls | +0.262s vs count-only |
| `scatter_matched_values_kernel` | 5.782s / 12,456 calls | 0.001s / 3 calls | eliminated except fallback |
| Arena fallback batches | n/a | 3 / 12,456 | 0.024% |
| Emit kernel | 0.050s / 12,456 calls | 0.051s arena + 0.000s fallback | flat |
| CUB `DeviceScan::ExclusiveSum` NVTX | 1.770s / 24,912 ranges | 0.835s / 12,459 ranges | roughly halved |

Important caveat: the targeted B win is clearer than the total-profile win. In the arena capture, unrelated major kernels were slower than the baseline capture (`sieve_kernel_trans_pp64_r64` +1.526s, `scatter_roots_kernel` +1.380s, `filter_per_bucket_kernel` +0.365s). That variance masks part of the B saving in total kernel time. The robust conclusion is that B removed the second matched-value N-scan in **12,453 / 12,456** batches and saved about **5.5s** of GPU kernel time on the matched-value path for coeff 420.

Arena diagnostic sweep:

| Coeff | Runs | Batches | Arena attempts | Arena fallbacks | Capacity skips | Matched values | Found records | Engine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 420 | 1 | 12,456 | 12,456 | 3 | 0 | 35,196,265 | 12,739 | 59.714s |
| 60,060 | 2 | 92,454 each | 92,454 each | 0 each | 0 each | 684,476,386 | 217,073 | 470.229s / 469.080s |

Interpretation: width 8 looks safe on both the easy coeff and a much larger 60060 sample. Combined observed fallback rate across the 420 + one 60060 run is **3 / 104,910 batches = 0.0029%**, with no capacity skips. There is no evidence yet to raise `MATCH_ARENA_WIDTH` or add the more expensive per-slot peak reduce.

Note on `-s`/`-l`: one attempted alternate-output validation command using these flags segfaulted before the collision engine started. We do not know whether that invocation worked before B1, and it is not part of the normal workflow used for these profiles. Treat it as a separate CLI/harness issue only if that path becomes relevant.

### Source state after experiments

- `cub/collision_engine.cu:133-149` now contains a `#ifdef SCATTER_DIRECT_ATOMIC` gate kept in source for future diagnostic use. Default build (without `-DSCATTER_DIRECT_ATOMIC`) is the original warp-aggregated path, identical to pre-experiment behavior.
- A1's `__stcs` change has been reverted; no source change remains from A1.
- B1's bounded-arena fast path is active by default with exact fallback to the old value scan/scatter path.
- nsys artifacts: `profile_gerbicz_420.nsys-rep` / `.sqlite` for baseline and `profile_gerbicz_420_arena.nsys-rep` / `.sqlite` for B1.
- Baseline ncu artifact: `scatter_roots_ncu.ncu-rep`. A1 artifact: `scatter_roots_ncu_stcs.ncu-rep`. A.5 artifact: `scatter_roots_ncu_direct.ncu-rep`. All in repo root.

### Other levers, ranked by expected wall win

(For context — if scatter optimization stalls, these are next.)

1. ~~**Memset cleanup**~~ — landed (v6). Lever turned out to be one 267 MB `gpu_root_array` clear per batch (not the tiny counter clears the v5 framing assumed). 5.48s -> 0.0s on coeff 420 once the trans kernel was refined to fully write plane 0 for `num_aprog_vals == 1`. Coeff 60060 engine **−50s (−11%)** as the same change scales to the larger batch count.
2. **Trans math-chain restructure** (sieve_kernel_trans_pp64_r64 at ~37% of GPU time post-memset, 30% Stall Wait on Montgomery dependency chains per the original trans_full ncu). High ceiling but high effort.
3. **Optional fused trans + bucket-scatter** — eliminates the ~280 MB intermediate root array between trans and scatter. Large refactor.

Full ordering in POLYSELECT_OPTIMIZATION_NOTES.md "Next optimization" section.

## Validation plan

**Parity is mandatory before any perf claim, for every variant.**

```bash
rm -f msieve.dat.ms
./msieve -np1 -nps "min_coeff=420 max_coeff=420 collengine=gerbicz" \
  && mv msieve.dat.ms msieve.scatterN.ms
```

Then byte-exact diff against the LOG2=14 baseline after canonical record-sort. Variant **fails** if any polynomial differs. (For deterministic kernels with fixed bucket layout, exact match is required — earlier "within 1%" framing was too loose.)

`collstats=1` should also show identical `dedup`, `matched`, `candidates`, and `value_match` counts. Found-array `peak/total` per coeff should match baseline.

## Integration plan

### Source changes (cumulative across experiments)

1. **Experiment A:** single-line change at `cub/collision_engine.cu:151`. No new kernel, no flag plumbing.
2. **Experiment B:** new fused count/store fast path and arena emit kernel in `cub/collision_engine.cu`; the old count/scan/scatter/emit path remains as the exact fallback.
3. **Experiment C (if pursued):** new `scatter_roots_v2_kernel` + a `collscatter=N` flag in `stage1_sieve_gpu.c` mirroring existing `collengine=` and `collstats=`. Keep v1 as fallback.

### Build

`make CUDA=1 all` per [memory: msieve build command](../.claude/projects/-home-kylea-msieve-s/memory/msieve_build_command.md). Verify `.so` mtime updated after each kernel change.

### nsys recapture for each variant

```bash
nsys profile --stats=true --force-overwrite=true -o profile_gerbicz_420_<variant> \
  ./msieve -np1 -nps "min_coeff=420 max_coeff=420 collengine=gerbicz collstats=1"
```

For scatter experiments, compare `scatter_roots_kernel`, `filter_per_bucket_kernel`, and full engine time. For B-style value-path experiments, compare `count_matched_values_kernel + scatter_matched_values_kernel + CUB scan` against the replacement path, and separately note any movement in unrelated kernels.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `__stcs` already the default on sm_120 → Experiment A is a no-op | medium | Cheap to find out (30 min). If no change, move to B. |
| `__stcs` changes ordering semantics in some subtle way (e.g., breaks atomicity guarantees in the engine) | low | Streaming hint is cache-policy only, not ordering. Verify via parity diff. |
| Streaming stores improve scatter but regress Phase B reads | medium | Compare full engine time and `filter_per_bucket_kernel`, not only scatter duration. |
| Fused B kernel has different occupancy profile that loses other wins | medium | Profile with ncu before/after. If fused kernel regresses on any sub-metric, keep them separate. |
| Arena overflow rate climbs on other coefficients | medium | Monitor fallback rate across more coefficients. Raise `MATCH_ARENA_WIDTH` only with measured evidence, not proactively. |
| Fallback repeats the full matched-value lookup | low | Accept this while fallback stays rare. At 3/12,456 batches, optimizing the fallback is not worth new complexity. |
| Arena sizing lacks per-slot peak visibility | medium | Current diagnostics expose attempts/fallbacks/capacity skips only. If coefficient sweeps show overflow, add stats/debug-only `arena_peak_value_count` before changing width. |
| Phase C revised config still regresses due to wave math | high | Acknowledged above. Don't pursue C unless A + B underdeliver and there's specific reason to believe coalescing helps. |
| Targeted optimization saves L2 bandwidth but DRAM bandwidth becomes the new wall | low | DRAM is at 50% — substantial headroom. Even doubling DRAM utilization stays under peak. |

## Decision criteria for moving on

- **After Experiment A:** if full engine time drops by ≥5%, ship and stop. If <5%, move to B without further scatter work. **→ A1 ran, ≈0% improvement (slight regression). A.5 confirmed kernel is L2-throughput-ceiling-bound. Moved on.**
- **After Experiment B:** if cumulative wall improvement ≥10%, stop. If <10%, decide whether C is worth pursuing based on remaining wall budget. **→ B1 is a real targeted win: old value path 11.698s → 6.181s and fallback only 3/12,456 batches. Do not pursue C next; the remaining easy lever is memset cleanup, then trans.**
- **Before Experiment C:** require a standalone microbench (using a dumped real `roots` buffer from coeff 420) showing ≥20% scatter-kernel improvement before integrating. **→ C now retired per A.5 finding; kernel-local restructure cannot reduce L2 transaction count.**
- **At any point:** if a variant breaks parity, halt and root-cause before continuing.

## Next starting point

When picking this up:

1. Treat B1 width 8 as accepted unless a future coefficient shows frequent `arena_fallbacks` or nonzero `arena_capacity_skips`.
2. If future coefficients do show frequent arena overflow, add stats/debug-only peak instrumentation before comparing width 8 vs 16. Width 16 doubles arena footprint, so only raise it with measured overflow evidence.
3. ~~Move to memset cleanup next.~~ Done (v6). The next lever is the trans kernel math-chain restructure — see "Other levers" above and the original trans_full ncu in `POLYSELECT_OPTIMIZATION_NOTES.md`.
4. The remaining memset GPU time is 121 ms total on coeff 420 (vs 5.6s before), most of it CUB-internal scratch init. Not worth chasing further; the explicit small-counter clears in `cub/collision_engine.cu` would only save tens of ms even if fully coalesced.
5. If alternate `-s`/`-l` output/state paths are needed for automated parity runs, debug that startup crash independently; it is not currently evidence against B1.

Existing baseline artifacts in repo root:
- `profile_gerbicz_420.nsys-rep` / `.sqlite` — baseline nsys, LOG2=14
- `profile_gerbicz_420_arena.nsys-rep` / `.sqlite` — B1 arena nsys, LOG2=14
- `profile_gerbicz_420_memset.nsys-rep` / `.sqlite` — v6 memset-cleanup nsys, LOG2=14
- `scatter_roots_ncu.ncu-rep` — baseline ncu for scatter_roots
- `scatter_roots_ncu_stcs.ncu-rep` — A1 capture (`__stcs`, no win)
- `scatter_roots_ncu_direct.ncu-rep` — A.5 capture (direct-atomic, no win)
- `msieve.gerbicz.dat.ms` — baseline parity `.ms` (12,739 polynomials on coeff 420)

## References

- `cub/collision_engine.cu:114-152` — current `scatter_roots_kernel`
- `gnfs/poly/stage1/stage1_sieve_gpu.c:763-797` — engine invocation site
- `POLYSELECT_OPTIMIZATION_NOTES.md` — full profiling history, bucket sweep data, next-lever ranking
- CUDA Programming Guide §10.x — cache operators (`.cs`, `.wt`, etc.) and PTX store instructions
- CUDA C++ Programming Guide — `__stcs` / `__stwt` intrinsics

---

## Appendix: original analysis (superseded by ncu profile)

Preserved for reference; do not act on this section without re-checking against the ground truth above.

The original plan framed the bottleneck as DRAM write throughput (~35% of theoretical peak inferred from per-launch wall time) and proposed three phases:

- **Phase 1: shared-mem privatized counter** — clear `s_count[16384]` per block, do shared-mem atomics, one global atomic per non-empty bucket. **Ruled out** because: (a) at 256 keys/block into 16K buckets, only ~2 duplicate buckets/block, so ≤2 atomics saved; (b) 64 KB shared/block drops occupancy by ~14×; (c) memory-bound kernel doesn't benefit from atomic reductions when atomics are only 37% of the L2 budget.

- **Phase 2: block-sort + grouped flush** — `BlockRadixSort` to group keys by bucket, then coalesced writes within each group. The original config (T=16K, 1 block/SM at 64 KB shared) regresses at the actual N=35M. A revised config (T=4K, 4 blocks/SM, 14-bit radix sort only) is described as Experiment C above but the wave math still doesn't clearly win.

- **Phase 3: tile spanning + register accumulation** — speculative, not pursued.

The original bucket-count sweep finding still holds and is the basis for using LOG2_NUM_BUCKETS=14:

> Doubling NUM_BUCKETS to 32K made `scatter_roots` 39% slower per launch even though filter got 37% faster. Smaller NUM_BUCKETS made filter much slower with no scatter improvement. The scatter bottleneck is in the kernel itself, not in NUM_BUCKETS choice.

The two-LLM review (this document + a second model) is what drove the decision to validate the original plan's assumptions with ncu before implementing, which is what surfaced the L2-bound finding and reshaped the action plan.
