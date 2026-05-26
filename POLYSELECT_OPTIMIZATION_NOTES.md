# Polyselect Stage-1 GPU Optimization Log

> **Consolidates four prior plan docs** (`POLYSELECT_OPTIMIZATION_NOTES.md` original, `TRANS_KERNEL_PLAN.md`, `TWO_LEVEL_SCATTER_PLAN.md`, `FUSED_TRANS_SCATTER_PLAN.md`). The full version history is in git; this file is the working reference going forward.

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
