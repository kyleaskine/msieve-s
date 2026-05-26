# Trans kernel optimization plan

> **Status (2026-05-25, v2):** Re-profiled on coeff 420 and coeff 60060 after the memset cleanup landed. The pre-session ncu baseline was on coeff 120 and showed compute-bound (math chain) behavior — but the production-shaped workloads on this 186-digit composite are **memory-latency-bound and register-limited**. Plan substantially reordered: occupancy and memory traffic come first, ILP/math-chain work is deferred until those settle.

> **Status (2026-05-25, v2.1):** **T3 `__launch_bounds__(256, 6)` ran and REGRESSED.** Compiler hit the occupancy target cleanly (56 → 40 regs/thread, theoretical occupancy 72.9% → 87.5%, achieved 67.8% → 81.2%, zero spills), but per-launch trans duration grew **1.20 ms → 1.26 ms (+5%)** and Stall Long Scoreboard climbed **51.3% → 58.3%**. The tighter register budget forced the compiler to turn register-resident loop state into more dependent memory ops — adding more resident warps couldn't hide the extra latency. Wall A/B on coeff 420 gerbicz (no collstats), 3 runs each: baseline median **1:18.4** (range 0.7s), T3 median **1:20.7** (range 2.6s) → ~+2.8% wall regression. **Reverted.** Lesson: occupancy is not the gating lever here when raising it costs the math chain. Move to Step 2 (tail-effect fix) next — it's independent of occupancy. T3 details preserved below for reference.

> **Status (2026-05-25, v2.2):** **Step 2 tail-effect alignment ran and was effectively NEUTRAL.** Implemented init-time-cached `cuOccupancyMaxActiveBlocksPerMultiprocessor` (3 kernels × 5 size_x in {128,160,192,224,256}); env-tunable `TRANS_DESIRED_WAVES`; runtime override of `blocks_y` rounded to nearest wave-aligned target, gated on heuristic `total_blocks >= blocks_per_wave/2` AND `wave_blocks_y < blocks_y` (decrease-only — growing inflated medium/small SOA launches and regressed wall). Polynomial parity byte-exact (md5 `ef7d7676...`); collstats counters drift by 4/35M (~10⁻⁷, from block-boundary effects on zero-padding) which is harmless. **Per-kernel nsys A/B was the clean signal:** `sieve_kernel_trans_pp64_r64` 25.41s baseline → 25.59s wave-aligned-3 → **+0.7% regression on the trans kernel itself.** Wall A/B over 6 paired runs: mean -3.5%, median -0.9%, std ~7s — within noise. **The modinv-amortization theory didn't pan out**: pushing blocks_y 3 → 2 (each thread does 1.5× more specialq) saved <1% because the per-thread work was already dominated by the forward/backward montmul loops, not the single modinv64. **Diagnostic cache values (kernel 2 = PP64_R64, sm_120, post-JIT):** size_x=128 → 9, 160 → 7, 192 → 6, 224 → 5, 256 → 4 blocks/SM. **Reverted.** Lesson: ncu's "up to 25% tail effect" was the WORST-case ceiling assuming all blocks take identical time. Actual partial-wave waste at 3.96 waves with 96.25%-full last wave is ~1% of total runtime. The lever is just too small at this register-pressured occupancy. Both T3 and Step 2 retired — the v2 plan's framing of "raise occupancy / reduce tail waste" doesn't survive measurement on this hardware/workload. Step 2 details preserved below for reference.

## Baselines

### Post-memset-cleanup nsys (coeff 420, `profile_gerbicz_420_memset.sqlite`)

| Kernel | GPU time | % | Avg/launch |
|---|---:|---:|---:|
| `sieve_kernel_trans_pp64_r64` | 25.92s | **36.9%** | 694 µs |
| `scatter_roots_kernel` | 24.34s | 34.7% | 1.95 ms |
| `filter_per_bucket_kernel` | 12.98s | 18.5% | 1.04 ms |
| `count_and_store_matched_values_kernel` | 6.26s | 8.9% | 502 µs |
| Memsets | 0.12s | 0.2% | — |

### ncu on the dominant trans launch — coeff 120 → coeff 420 → coeff 60060

| Metric | Coeff 120 (historical) | Coeff 420 | Coeff 60060 |
|---|---|---|---|
| Launch shape (gridX, gridY) × blockX | (4282, 1) × 160 | (317, 3) × 224 | (182, 5) × 224 |
| Duration | 1.06 ms | 1.20 ms | 1.08 ms |
| Compute SM % | 74.9% | 52.8% | 54.5% |
| **DRAM %** | 51% | **68.5%** | **70.2%** |
| L1/TEX % | 67% | 27.5% | 23.8% |
| L2 % | 36.2% | 31.9% | 27.6% |
| **Top stall** | Wait 30.4% (math chain) | **Long SB 51.3%** (L1TEX) | **Long SB 48.8%** (L1TEX) |
| Cycles per issued inst | 12.2 | 20.2 | 18.8 |
| Registers / thread | 64 | 56 | 56 |
| Theoretical occupancy | 100% | **72.9%** | **72.9%** |
| Achieved occupancy | 85.5% | 67.8% | 67.7% |
| Eligible warps / scheduler / cycle | several | **0.89** of 8.34 active | similar |
| Local-mem spills | 0 | **0** | **0** |
| Tail waste (partial wave) | n/a | **231/240 blocks (~24%)** | **190/240 blocks (~21%)** |
| Threads | 685K | 213K | 204K |

Artifacts: `trans_full.ncu-rep` (coeff 120, pre-memset), `trans_full_post_memset.ncu-rep` (coeff 420), `trans_full_post_memset_60060.ncu-rep` (coeff 60060).

**Diagnosis:** coeff 120 was a small-coefficient outlier with lots of parallel work per launch and little memory work per thread — that pushed it compute-bound. Production coefficients (420, 60060) have fewer threads per launch but more memory work per thread, flipping the kernel into a memory-latency regime. The kernel can't hide L1TEX latency because the register-limited occupancy keeps fewer warps in flight than needed.

## Where the memory pressure is

In the inner write loop:

```c
for (j = qq_prod_offset, k = p_offset, m = 0;
            m < num_roots;
            j += num_p, k += num_p, m++) {
    newroot = modsub64(start_roots[k], qroot, pp);     /* global load */
    newroot = montmul64(newroot, curr_inv, pp, pp_w);
    /* writes to p_out[j], roots_out[j] (num_aprog_vals == 1 path) */
}
```

Major loads / stores in the kernel:

- `start_roots[k]` — coalesced uint64 loads per `m` iteration.
- `q_batch[curr_q->...]` — warp-broadcast specialq_t reads.
- `roots_out[curr_offset]`, `p_out[qq_prod_offset]` — global reads in the walk-back and main backwards loops of values the first/second loops just wrote.
- `roots_out[j]`, `p_out[j]` — coalesced writes (~86–89% sector utilization per ncu — small 7.6% uncoalesced share, ~5% potential win).

The first/second loops write a temporary `write_val` (`qq_prod` or 0) to plane-0 col-0 of each specialq slot, then the walk-back / main backwards loops read those values to detect qq_prod transitions. That write-then-read pattern goes through DRAM (L1/L2 may evict between phases) and is the most likely contributor to the Long Scoreboard stalls.

## GPU portability

The changes proposed below are designed to work across modern NVIDIA GPUs (compute capability ≥ 8.0 — Ampere, Ada, Hopper, Blackwell). Specifically:

| Change | Hardware-specific? | How it stays portable |
|---|---|---|
| Trans refinement (already shipped) | No | Pure logic, no hardware constants. |
| T0.5 branch hoist | No | Pure source restructure. |
| T3 `__launch_bounds__(T, B)` | One compile-time pair. | Correctly portable, but empirically negative at `(256, 6)` on the RTX 5070. Keep only as a preserved negative result unless the math chain is restructured first. |
| Tail-effect block-count tune | Yes, queried at runtime. | Uses `cudaGetDeviceProperties(...).multiProcessorCount` (same query the rest of msieve uses) to compute target `total_blocks` as a multiple of `numSMs × blocksPerSM`. Fully portable; the formula adapts to any GPU and should use the current no-launch-bounds kernel occupancy. |
| T1 `#pragma unroll N` | No | Pure source-level hint, compiler emits arch-specific SASS. |
| T2 per-num_roots templates | No | Compile-time specialization on a polyselect-side value (degree 5 → {1, 5, 25}), independent of GPU. |
| T4 modinv64 rewrite | No | Pure source/PTX. |
| Fused trans + bucket-scatter | No | Pure restructure. Inherits the existing `LOG2_NUM_BUCKETS=14` from `cub/collision_engine.cu`, which is GPU-tuned but out of scope here. |

**Not touched by this work:**

- `LOG2_NUM_BUCKETS=14` in `cub/collision_engine.cu` — already in repo, tuned to user's RTX 5070 L2. Memory note "don't change it" stands.
- Any hardcoded grid size, block size, or shared-memory size matching a specific card.

## Plan: experiments ordered by latest evidence

### Step 0 — Re-profile baseline

**Done** for coeff 420 and coeff 60060. Both show memory-bound, register-limited, ~25% tail waste. The numbers above feed the rest of the plan.

> **Reviewer note (carried from v1):** Add one legacy radix-sort sanity run to the validation loop for any trans experiment. The trans kernels feed both `collengine=gerbicz` and the legacy sort/final path, so Gerbicz parity alone is not enough to prove the trans change is safe for the default path.

### Step 1 — T3: raise occupancy via `__launch_bounds__` — **DONE, NEGATIVE (2026-05-25)**

> **Result:** `__launch_bounds__(256, 6)` on `sieve_kernel_trans_pp64_r64` only. Built clean, byte-exact parity against `msieve.gerbicz.dat.ms` on coeff 420 (both gerbicz and legacy sort paths, md5 `ef7d7676615235c41caa228698940511`). ncu re-capture on the dominant launch (951 blocks × 224 threads, coeff 420):
>
> | Metric | Baseline | T3 (256, 6) | Δ |
> |---|---|---|---|
> | Registers / thread | 56 | **40** | -16 |
> | Block Limit Registers | 5 | **6** | +1 (target hit) |
> | Theoretical Occupancy | 72.9% | **87.5%** | +14.6 pp |
> | Achieved Occupancy | 67.8% | **81.2%** | +13.4 pp |
> | Local Memory Spilling Requests | 0 | **0** | gate passed |
> | Stall Long Scoreboard | 51.3% | **58.3%** | **+7 pp** |
> | Stall Wait | 30.4% (coeff 120 ref) | 15.1% | (different regime — see note) |
> | Warp Cycles per Issued Inst | 20.2 | **24.45** | +4.25 |
> | Duration (per launch) | 1.20 ms | **1.26 ms** | **+5%** |
>
> Per-launch ncu artifact: `trans_full_t3.ncu-rep`.
>
> Wall A/B, coeff 420 gerbicz, no collstats, 3 runs each:
> - Baseline: 1:18.9, 1:18.2, 1:18.4 — median **1:18.4**, range 0.7s
> - T3: 1:20.3, 1:22.9, 1:20.7 — median **1:20.7**, range 2.6s
> - Net wall: ~+2.3s (+2.9%) regression on coeff 420
>
> User's nvidia-smi observation: Volatile GPU-Util is low/mid 80s now — i.e., the GPU is already mostly busy. The occupancy raise added resident warps but the new warps had nothing useful to do because the compiler turned register state into more memory ops to fit the lower budget. The kernel was already finding eligible warps (0.89 → 0.98 per scheduler); the wall is bounded by memory latency on the math chain, not by warp shortage.
>
> **Reverted.** Source restored to no annotation; PTX checksum back to original size. The occupancy lever is dead at this register-pressure point. The math chain *needs* its registers — try restructuring the math chain (Step 7 / T4-style) first if we want to revisit occupancy.

(Original Step 1 description preserved below for reference.)

**Direct attack on the bottleneck.** The kernel is register-limited to 5 blocks/SM (theoretical occupancy 72.9%). Forcing the compiler to fit in fewer registers lets 6 blocks/SM fit, raising theoretical occupancy to ~87.5%. More resident warps means the scheduler has more eligible candidates when one is stalled on memory.

**Touch points:** Add `__launch_bounds__(256, 6)` to `sieve_kernel_trans_pp64_r64` (and the pp32 variants once pp64 is proven). The constant is conservative — works on any CC 8.0+ card.

> **Reviewer note (carried from v1):** `__launch_bounds__(160, 4)` would be unsafe because the host can launch with up to `size_x = MIN(256, launch->threads_per_block)` threads/block. **256 must be the maximum** in the launch_bounds first argument, since smaller-block launches respect it but the kernel must remain callable from 256-thread launches.

**Risks:**

| Risk | Likelihood | Mitigation |
|---|---|---|
| Compiler spills registers to local memory at the lower budget | medium-high | Re-profile with ncu: check `Local Memory Spilling Requests` (currently 0). If non-zero, back off to `(256, 5)` or revert. |
| Spills add DRAM traffic and make memory pressure worse | medium | Same as above — explicit ncu check. |
| Lower regs/thread limits the compiler's ability to keep loop state in registers, hurting the math chain | low–medium | ncu Stall Wait will rise if so; compare against baseline. |
| Different cards have different SM counts → different total throughput, but same per-SM behavior | n/a | This is fine; portability is preserved. |

**Effort:** 10 minutes to add the annotation. ~30 min for parity check + ncu re-capture.

**Stop criterion:** ship if wall improvement ≥ 3% with parity and no local-mem spills. If spills appear, back off the `B` value.

### Step 2 — Tail-effect fix — **DONE, NEUTRAL (2026-05-25)**

> **Result:** Implemented in `gnfs/poly/stage1/stage1_sieve_gpu.c` (now reverted). Approach:
> - Init-time cache of `cuOccupancyMaxActiveBlocksPerMultiprocessor(kernel, size_x)` for the 3 trans kernels at 5 size_x values (128, 160, 192, 224, 256). Eliminates per-batch driver-call overhead (37k+ calls/coeff).
> - Init-time read of `TRANS_DESIRED_WAVES` env var (1..16; default 3).
> - At each launch, after the existing size_x adjustment loop: compute `wave_target = numSMs * blocks_per_sm * desired_waves`, then `wave_blocks_y = round(wave_target / blocks_x)`, gated on `total_blocks >= blocks_per_wave/2` AND `wave_blocks_y < blocks_y` (decrease-only — see lesson below).
>
> **Cached blocks_per_sm values (kernel 2 = PP64_R64, post-JIT to sm_120 on RTX 5070):**
>
> | size_x | 128 | 160 | 192 | 224 | 256 |
> |---|---|---|---|---|---|
> | blocks/SM | 9 | 7 | 6 | **5** | 4 |
>
> Confirms ncu's earlier 5 blocks/SM at size_x=224 (the dominant launch shape).
>
> **For coeff 420 dominant launch (size_x=224, blocks_x=317, baseline blocks_y=3, total=951):**
> - desired_waves=4: wave_target = 48×5×4 = 960; wave_blocks_y = round(960/317) = 3 = baseline → no-op
> - desired_waves=3: wave_target = 720; wave_blocks_y = round(720/317) = 2 → blocks_y 3 → 2, total 951 → 634 (-33%). Tested.
>
> **Parity:** byte-exact polynomial output (md5 `ef7d7676...`, 12,739 polys). Collstats counters drift by 4/35,196,265 (~10⁻⁷) due to block-boundary effects on zero-padding rows in `roots_out` — the filter sees slightly different intermediate state but produces the same polynomial set.
>
> **Per-kernel nsys A/B (clean signal, low noise):**
>
> | Kernel | Baseline | desired_waves=3 | Δ |
> |---|---|---|---|
> | `sieve_kernel_trans_pp64_r64` | 25.41s | 25.59s | **+0.7%** |
> | trans avg/launch | 680 µs | 685 µs | +0.7% |
> | scatter_roots_kernel | 23.42s | 22.78s | -2.7% (noise, kernel untouched) |
> | filter_per_bucket_kernel | 12.80s | 12.63s | -1.3% (noise) |
>
> **Wall A/B over 6 paired back-to-back runs (no collstats, coeff 420 gerbicz):**
>
> | Pair | Baseline | wave-aligned-3 | Δ |
> |---|---|---|---|
> | 1 | 1:38.0 | 1:29.9 | -8.1s |
> | 2 | 1:31.1 | 1:27.5 | -3.6s |
> | 3 | 1:28.9 | 1:30.7 | +1.8s |
> | 4 | 1:42.9 | 1:28.0 | -14.9s |
> | 5 | 1:30.4 | 1:34.7 | +4.3s |
> | 6 | 1:30.5 | 1:31.1 | +0.6s |
>
> Mean delta: -3.3s (-3.5%); median delta: -0.85s (-0.9%); std of delta: ~7s. Not statistically distinguishable from zero with 6 pairs (95% CI: -3.3 ± 5.8s).
>
> **Lessons:**
> - **ncu's "up to 25%" tail-effect is the WORST-case ceiling**, assuming all blocks take identical time. Realistic partial-wave waste at 951 blocks / 240-per-wave = 3.96 waves with 96.25%-full last wave is only ~1% of total runtime.
> - **The modinv-amortization theory didn't pan out.** Pushing blocks_y 3 → 2 (each thread does 1.5× more specialq) saved only 0.7% on the trans kernel itself, because per-thread work was already dominated by the forward/backward montmul loops (which scale with size_y) — not by the single modinv64 (which is fixed per thread).
> - **The wave-align must only DECREASE blocks_y.** Growing inflates the medium/small SOA launches: my first attempt (desired_waves=4, increase allowed) regressed wall by ~25-30% because the medium SOA's blocks_y went 1 → 4, multiplying its grid by 4× for the same total work.
> - **Init-time caching was critical.** The first wave-align attempt queried `cuOccupancyMaxActiveBlocksPerMultiprocessor` and `getenv` per-call (37k×/coeff) — substantial overhead even though the values never change.
> - **Per-kernel nsys is the right signal when wall noise is large.** The 6-pair wall A/B couldn't distinguish ±4% from zero; nsys-per-kernel showed clearly the trans kernel didn't move.
>
> **Reverted.** Source restored to no-overrides; collstats fingerprint exact-match baseline (12,456/69,228,524/17,344,961/35,196,265/12,739). Both T3 and Step 2 retired — the v2 plan's framing of "raise occupancy / reduce tail waste" doesn't survive measurement on this hardware/workload. Step 2 details preserved below for reference.

(Original Step 2 description preserved below for reference.)

**Direct attack on the second-largest single source of waste.** ncu estimates **up to 25%** per-launch potential just from the partial-wave tail. The grid is currently 951 blocks (coeff 420) / 910 blocks (coeff 60060), which works out to ~3.79–3.96 waves with the current 5 blocks/SM occupancy. The partial wave occupies the GPU for the same wall time as a full wave.

**Touch points:** `stage1_sieve_gpu.c:728`, the `total_blocks` formula:

```c
total_blocks = (3 * num_p * num_q +
                num_p * soa->num_roots * num_specialq) /
                50000;
total_blocks = MIN(total_blocks, 1000);
total_blocks = MAX(total_blocks, 1);
```

The `50000` divisor and `1000` cap are static. Replace with a wave-aligned target based on `d->gpu_info->num_sms × blocks_per_sm × desired_waves`. `blocks_per_sm` should come from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (runtime query) for the current no-launch-bounds trans kernel.

**Risks:**

| Risk | Likelihood | Mitigation |
|---|---|---|
| Wave count depends on actual occupancy | medium | Query/cache occupancy for the current kernel instead of assuming either 5 or 6 blocks/SM. |
| Different per-SOA launch shapes need different block counts | medium | Compute target per-SOA, not globally. |
| Querying `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at every batch is slow | low | Cache the result per kernel function pointer at init time. |

**Effort:** ~2 hours including measurement.

**Stop criterion:** ship if tail alignment gives a clear wall improvement (target >= 3-5%) with parity and no regression in the dominant launch. Revert if it only moves launch shape without measurable wall gain.

### Step 3 — T0.5: hoist `num_aprog_vals == 1` branch

Slim cleanup, low priority but useful while we're touching the trans kernel. The two inner write loops branch on `num_aprog_vals == 1` per iteration; that branch is loop-invariant and could be hoisted to give the compiler simpler bodies. The `num_aprog_vals > 1` path is the rare batch (end-of-coeff sizing), `num_aprog_vals == 1` is overwhelmingly the common case.

> **Reviewer note (carried from v1):** This also gives a cleaner baseline for unroll factors when we eventually get to T1.

**Effort:** ~1 hour.

**Stop criterion:** included for code-cleanliness. Even a 0% perf change is fine; revert if it regresses (it shouldn't).

### Step 4 — Re-profile

Capture ncu on the dominant trans launch for both coeff 420 and coeff 60060 again. Verify:

- Tail waste / partial-wave exposure dropped meaningfully
- Stall Long Scoreboard and per-launch duration did not regress
- No local memory traffic appeared
- Whether the bottleneck has shifted (e.g., compute now leading, or memory bandwidth saturated)

**This re-profile is mandatory.** Plan the rest of the work based on what the new ncu shows, not on the historical assumptions.

### Step 5 — T1: `#pragma unroll N` on the inner write loops (demoted to "only if compute becomes the wall")

The original v1 plan put T1 first. Per the new evidence, it should only be tried if Step 4's re-profile shows compute leading and Stall Wait climbing. Otherwise the increased register pressure could lower the effective quality of the generated code, as T3 already demonstrated.

**Touch points (when triggered):** the two inner write for-loops in each trans kernel variant (six sites across pp32_r32 / pp32_r64 / pp64_r64).

> **Reviewer note (carried from v1):** Sweep `#pragma unroll` factor `2 → 4 → 5 → 8`. Factor 5 is included because degree-5 SOA 1 has `num_roots = 5`.

> **Reviewer note (carried from v1):** Treat `nvcc -Xptxas -v` register counts as secondary for this path. The stage1 kernels are built as PTX and JITed to sm_120 at runtime, so ncu's launch/register/spill metrics on the actual RTX 5070 run are the authoritative signal.

**Stop criterion:** ship if wall improvement >= 3% on top of the accepted tail/T0.5 baseline. If `#pragma unroll 2` worsens or matches baseline, skip the higher factors.

### Step 6 — Fused trans + bucket-scatter (only if memory-bound persists)

The big restructure. Trans currently writes ~280 MB/launch to `gpu_root_array` and ~140 MB to `gpu_p_array`. The collision engine immediately reads `gpu_root_array` back to bucket-scatter into `bucket_storage`. Fusing eliminates the intermediate DRAM round-trip.

**Effort:** large refactor — possibly 1–2 weeks of focused work. Major architectural change.

**Trigger condition:** re-profile after tail+T0.5 still shows >= 50% Long Scoreboard stalls and >= 60% DRAM utilization, and the wall improvement target has not been met by T1.

### Step 7 — T2 per-`num_roots` specialization, T4 modinv64 rewrite (low priority)

Both target compute. Compute has headroom in the current profile, so these are not the lever right now. Keep them in the plan as a tail option if the bottleneck shifts to compute in a future iteration.

> **Reviewer note (carried from v1):** `modinv64` runs **once per thread** (after the forward product scan at `stage1_core.cu:466`), not per qq_prod transition. The backward transitions use the pair of `montmul64`s around `stage1_core.cu:527-528`. That makes `modinv64` even less likely to be the first lever; T4 stays last-resort unless the post-memset ncu source counters prove otherwise.

## Validation (every experiment)

1. **Build:** `make CUDA=1 all`; verify PTX mtimes (`stage1_core_sm{89,90}.ptx`) and `msieve` mtime updated.
2. **Parity, gerbicz path:** rerun `./msieve -np1 -nps "min_coeff=420 max_coeff=420 collengine=gerbicz collstats=1"`, canonical-sort, byte-exact diff vs `msieve.gerbicz.dat.ms`. *Required.*
3. **Parity, legacy sort path** (per reviewer note): rerun on the same coeff without `collengine=gerbicz`, canonical-sort, byte-exact diff. *Required for any trans-kernel-touching experiment.*
4. **Engine counters:** collstats must match B1 baseline exactly (12,456 batches / 69,228,524 candidates / 17,344,961 dedup / 35,196,265 matched / 12,739 found / max_bucket 2,414).
5. **nsys recapture:** compare `sieve_kernel_trans_pp64_r64` total time and per-launch duration; compare full engine wall.
6. **ncu re-capture:** Stall Long Scoreboard %, Stall Wait %, Issue Slots Busy, Active Warps, register count, occupancy, **Local Memory Spilling Requests** on the dominant launch.
7. **Sanity on 60060:** if 420 parity holds and wall improves, rerun 60060 to confirm at scale.

## Decision criteria for moving on

- **After T3:** done, negative, reverted. Do not retry lower-register `__launch_bounds__` without first reducing/restructuring the math-chain state.
- **After tail-effect:** if wall improvement is clear (target >= 3-5%), ship that and re-profile before deciding T1.
- **After T0.5:** functional cleanup, no perf gate.
- **After re-profile:** branch on bottleneck nature. Memory still leading → consider fused trans+bucket-scatter (Step 6) before T1. Compute now leading → T1 sweep makes sense.
- **Stop entirely** if tail+T0.5 deliver < 3% combined — the kernel may already be near its memory-side ceiling without the big restructure.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| `__launch_bounds__` value `(256, 6)` regresses despite no spills | Observed and reverted; do not pursue occupancy-by-register-cap as the next lever |
| Compiler's reg-count varies between sm_89 and sm_90 PTX → JIT to sm_120 | Test on the live device via ncu, not via static `-Xptxas -v` |
| Tail-effect tune assumes wrong occupancy | Use runtime occupancy query on the current no-launch-bounds kernel |
| Soa 0 (num_roots=1) regresses from added unroll bookkeeping at T1 time | Templated dispatch (T2) avoids; until then `#pragma unroll 1` is a no-op |
| Stall shape moves to a different chain after tail/T0.5 | Step 4 re-profile catches this; iterate |
| Fused trans+bucket-scatter (Step 6) is high-effort and may not be worth it if memory pressure drops below the threshold first | Use Step 4 re-profile to gate the decision |
| Medium and tiny trans launch shapes have different bottlenecks | Big launch is 62% of trans wall on coeff 420; targeting it first is the right call. Re-check smaller shapes if needed. |

## Next starting point

Both T3 and Step 2 retired as negatives/no-ops. The v2 plan's two "easy" levers didn't deliver. Re-strategize before more code:

1. ~~T3: `__launch_bounds__(256, 6)`~~ — Done 2026-05-25, regressed +2.9% wall, reverted (v2.1).
2. ~~Step 2: tail-effect via cuOccupancyMaxActiveBlocksPerMultiprocessor + wave alignment~~ — Done 2026-05-25, +0.7% on the trans kernel per nsys (neutral within noise), wall ±4% within session noise. Reverted (v2.2).
3. **Re-profile to see where the bottleneck actually is now.** Both T3 and Step 2 retired the "raise resident warps" framing of v2. A fresh ncu on the dominant trans launch after the memset cleanup (with no T3/Step-2 hangover) would tell us whether Long Scoreboard is still the dominant stall or if it has shifted. Cheap to run; high information.
4. **T0.5 branch hoist** as a cleanup pass — perf-neutral but simpler code; sets a cleaner baseline for any future ILP work.
5. **Step 5/T1: `#pragma unroll`** on the inner write loops. Only useful if compute becomes the wall (Stall Wait > Stall Long Scoreboard). Decide after Step 3 re-profile.
6. **Step 6: fused trans + bucket-scatter.** Big restructure (1-2 weeks), eliminates the 280 MB intermediate root-array round-trip. The biggest remaining lever per the analysis but high effort. Only worth pursuing if no smaller win is found.
7. **Step 7: T2 / T4** (per-num_roots templates / modinv64 rewrite). Compute-side levers — low priority unless re-profile shows compute leading.

The post-T3/Step-2 evidence increasingly suggests the trans kernel is closer to its achievable optimum than the v2 plan thought. The biggest remaining levers may actually be on **other** kernels (scatter_roots at 34.7%, filter_per_bucket at 18.5%) which we previously concluded were L2-throughput-ceiling-bound. That conclusion was reached when scatter was 31.2% of GPU time; if trans is at its floor and the engine kernels have more headroom than the L2-ceiling framing suggested, the priority ordering may want a fresh look.

## References

- `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu` — trans kernel source. Inner write loops to target are around lines 503–518 and 549–578 (pp64_r64), with parallel sites in pp32_r32 / pp32_r64.
- `gnfs/poly/stage1/stage1_core_gpu/cuda_intrinsics.h` — `montmul64` (32-bit decomposed, no DP), `modinv64` (64-bit, probably routes some ops to DP — but called only once per thread per the reviewer note).
- `gnfs/poly/stage1/stage1_sieve_gpu.c:728` — host-side `total_blocks` computation for trans kernel launch.
- `trans_full.ncu-rep` — historical full-detail ncu (coeff 120, pre-memset). Compute-bound regime.
- `trans_full_post_memset.ncu-rep` — current ncu on coeff 420 dominant launch (memory-bound).
- `trans_full_post_memset_60060.ncu-rep` — current ncu on coeff 60060 dominant launch (memory-bound, confirms 420 finding).
- `profile_gerbicz_420_memset.nsys-rep` / `.sqlite` — current nsys baseline (coeff 420, post-memset).
- `POLYSELECT_OPTIMIZATION_NOTES.md` — full ncu interpretation under "Step 3 — Full profile on the big trans launch".
- `msieve.gerbicz.dat.ms` — parity baseline (coeff 420, 12,739 polynomials).
