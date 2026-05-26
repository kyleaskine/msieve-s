# Profile the polyselect stage-1 trans kernel to direct next optimization

## Context

You're tracking the Gerbicz / Papadopoulos / Broman work on a much faster collision-detection algorithm for polyselect stage 1. JasonP reports ~4× speedup on GPU vs the existing CUB radix-sort approach. Before deciding whether the trans kernel (which runs *before* the collision search) is also worth attacking, you want to know what it's actually bottlenecked on: compute (the modular inverse + Montgomery multiplies) or memory (strided writes to `roots_out` / `p_out`).

The answer determines which optimization to chase from the discussion section at the bottom of this file.

### System under test

- GPU: RTX 5070, Blackwell, compute capability 12.0
- Tools: CUDA 12.8, Nsight Compute 2026.1.1, Nsight Systems available
- PTX shipped: `stage1_core_sm89.ptx` / `stage1_core_sm90.ptx` (both JIT to sm_120 at runtime)
- Active factorization: 186-digit composite (`worktodo.ini`), degree 5, max stage-1 norm 1.08e+28, using `coeff_list.txt` driving with leading coefficients 9240 → ... → 116396280

### Kernels to profile (from `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu`)

- `sieve_kernel_trans_pp64_r64` (most likely dominant for a 186-digit N — p² typically > 2³²)
- `sieve_kernel_trans_pp32_r{32,64}` (smaller-p path)
- `sieve_kernel_final_{32,64}` (post-sort collision scan)
- `cub::DeviceRadixSort*` (current sort, hosted in `cub/sort_engine.so`)

## Plan

### Step 1 — Timeline overview with nsys

Get the per-kernel time split before drilling into one kernel. Run msieve under nsys for ~30 seconds:

```bash
nsys profile --stats=true --force-overwrite=true \
    --output profile_baseline \
    ./msieve -np1 -g 0 -t 1 -v
```

Stop with Ctrl-C once a coeff has fully run (or set `--duration=60`). Inspect the stats output for kernel time breakdown:

```
nsys stats --report cuda_gpu_kern_sum profile_baseline.nsqlite
```

The numbers I want from this:
- Total GPU time per kernel
- Total time in `sieve_kernel_trans*` vs `cub::DeviceRadixSort*` vs `sieve_kernel_final*`
- Average launch duration per kernel

This calibrates *how big a lever* the trans kernel is. If trans = 20% of GPU time and sort = 70%, even doubling trans speed only gives a ~10% total win — confirms the Gerbicz collision port (which kills the sort) is the bigger lever.

### Step 2 — Speed-of-Light pass on the trans kernels

Light, fast metric set to classify the bound type:

```bash
ncu --target-processes all \
    --kernel-name regex:sieve_kernel_trans \
    --launch-skip 20 --launch-count 5 \
    --set speedoflight \
    --export trans_sol.ncu-rep \
    ./msieve -np1 -g 0 -t 1 -v
```

`--launch-skip 20` avoids the first 20 launches so JIT-compile noise doesn't pollute. `--launch-count 5` gets us five representative launches with low overhead.

Open in `ncu-ui` (or `ncu --import trans_sol.ncu-rep --page details`) and read the **Speed of Light** section. Interpretation:

| SM throughput | Memory throughput | Diagnosis |
|---------------|-------------------|-----------|
| > 80%         | < 50%             | Compute-bound |
| < 50%         | > 80%             | Memory-bound |
| 50–80%        | 50–80%            | Balanced — go to Step 3 for stall analysis |
| both low      | —                 | Latency-bound (occupancy / divergence) — also Step 3 |

### Step 3 — Full profile on one representative launch

Once we know the dominant variant, get the deep metrics on one launch:

```bash
ncu --target-processes all \
    --kernel-name sieve_kernel_trans_pp64_r64 \
    --launch-skip 30 --launch-count 1 \
    --set full \
    --export trans_full.ncu-rep \
    ./msieve -np1 -g 0 -t 1 -v
```

`--set full` replays the kernel many times for different metric groups; do *not* do this with `--launch-count > 2` or it will take forever. One launch with full metrics is enough.

Sections to inspect, in order of importance:

1. **Warp State Statistics** — stall reasons.
   - `stall_long_scoreboard` high → waiting on DRAM/L2 → memory-bound
   - `stall_math_pipe_throttle` / `stall_wait` high → math units saturated → compute-bound
   - `stall_short_scoreboard` → shared-memory dependency
   - `stall_barrier` → block-level sync stalls
2. **Memory Workload Analysis** — L1 / L2 / DRAM read+write throughput, sectors-per-request.
   - Sectors-per-request > 4 on global stores → uncoalesced (expected given the `num_entries`-stride writes in trans kernel).
   - DRAM throughput close to peak (~600+ GB/s on RTX 5070) → memory-bound.
3. **Instruction Statistics** — which instruction types dominate. Look for IMAD/IADD3 (arithmetic), LDG/STG (memory), ISETP/branch (divergence).
4. **Source Counters** — if SASS is attributed cleanly, identifies the specific lines with most stalls. Expected hotspots: `modinv32`/`modinv64` (inside `cuda_intrinsics.h`) and the strided write loop around `stage1_core.cu:127–153`.
5. **Compute Workload Analysis** — which pipes are saturated (INT, ADU, LSU, MIO).
6. **Achieved Occupancy** — is the kernel running enough warps in flight? Trans kernels use ~256 threads/block from what I saw in `stage1_sieve_gpu.c:704`; if occupancy is low, register pressure may be the issue.

### Step 4 — Comparator pass on the CUB sort

For context, run the same light profile on the sort:

```bash
ncu --target-processes all \
    --kernel-name "regex:cub::Device" \
    --launch-skip 30 --launch-count 5 \
    --set speedoflight \
    --export sort_sol.ncu-rep \
    ./msieve -np1 -g 0 -t 1 -v
```

Confirms that the sort is memory-bound (expected) and gives a baseline to compare against once a Gerbicz-style collision kernel exists.

## Reporting back

After running the above, paste back:

1. From Step 1: trans vs sort vs final time split
2. From Step 2: which trans variant dominates; SoL bound type
3. From Step 3: top 3 stall reasons by %; L1/L2/DRAM throughput %; sectors-per-request on global stores
4. From Step 4: sort SoL bound type (for comparison)

That's enough to pick the next optimization with confidence.

## Decision table for next optimization

Based on what the profile reveals:

| Trans kernel bound | Sort dominant? | Recommended next step |
|---|---|---|
| Compute (math throttle) | Yes | Wait for JasonP's collision port; trans tuning is secondary |
| Memory (long scoreboard) | Yes | Wait for collision port, then plan a fused trans+bucket-scatter kernel to eliminate the intermediate DRAM round-trip (~1+ GB per batch at RSA scale) |
| Compute | No (trans dominant) | Modular arithmetic: warp-cooperative modinv, better SASS for `modinv64`, possible reorganization to enable batch-inversion across q |
| Memory | No (trans dominant) | Output layout: per-warp staging in shared memory then coalesced flush; or fuse-with-bucketing |
| Latency / low occupancy | — | Reduce register pressure (check `--print-summary all` register usage), tune block size; possibly split kernel by `num_aprog_vals` path |

## Pitfalls

- `--set full` replays the kernel ~10× internally per launch. Restrict to `--launch-count 1` or 2.
- First kernel launches include JIT-compile time (sm_89/sm_90 PTX → sm_120 SASS). Always `--launch-skip 20+`.
- nsys and ncu both attached to the same process will conflict. Run them as separate msieve invocations.
- If a coeff finishes before ncu collects enough launches, pick a larger leading coefficient from `coeff_list.txt` (the uncommented ones starting at 9240; bigger values run longer per batch). Or temporarily set a longer deadline via msieve args.
- Profiling a long-running polyselect during your actual factorization will interfere with the production run. Either pause the production run first, or do the profile in a separate worktodo / output directory.

## Verification that the profile data is real

Sanity checks before trusting the results:

- nsys timeline should show the kernels in the expected order: trans → cub sort kernels → final → trans → ... (one cycle per special-q batch)
- Total time per kernel × launch count should roughly match wall time minus CPU work
- `sieve_kernel_trans_pp64_r64` should appear (degree 5 + 186-digit ⇒ pp_is_64 = 1)
- The `gpu_elapsed` printed by msieve (it logs cumulative GPU time per coeff) should agree with nsys totals to within a few %

---

## RESULTS (2026-05-23, run against active 186-digit coeff 120)

### Step 1 — nsys timeline (60 s capture, 3,872 special-q batches processed)

| Kernel | GPU time | % | Launches | Avg/launch |
|---|---|---|---|---|
| `cub::DeviceRadixSortOnesweepKernel` | 41.0 s | **80.0** | 27,111 | 1.5 ms |
| `sieve_kernel_trans_pp64_r64` | 5.4 s | 10.6 | 11,619 | 467 µs |
| `sieve_kernel_final_64` | 2.9 s | 5.7 | 3,872 | 749 µs |
| `cub::DeviceRadixSortHistogramKernel` | 1.9 s | 3.7 | 3,873 | 493 µs |
| `cub::DeviceRadixSortExclusiveSumKernel` | 4 ms | 0.0 | 3,873 | 1 µs |

Sort total = **83.7%**. Trans = 10.6%. Final scan = 5.7%.

Other:
- Only `pp64_r64` fires (degree 5 × 186-digit ⇒ all p² > 2³²).
- 3 trans launches per batch (one per SoA: roots = 1, 5, 25). Structural.
- 11,619 / 3 ≈ 3,873 batches, matching the sort/final launch count.
- ~16 ms wall per batch.
- 1240 GB cumulative `cudaMemsetAsync` over 60 s ≈ **21 GB/s** of memset bandwidth (~3% of RTX 5070's ~675 GB/s peak DRAM). Asynchronous, mostly overlaps compute. Total memset wall time: 1.68 s of the 60 s window (~2.8%).
- Trans kernel launches use 64 registers/thread, configured occupancy ~62.5% (32 of 48 max warps/SM).

**Verdict for the original question:** the sort is decisively the bottleneck. The Gerbicz collision port (what JasonP is doing) is unambiguously the biggest lever. A 2× speedup of the trans kernel alone would buy only ~5% total.

### Step 2/3/4 — ncu Speed-of-Light + full + sort comparator

First attempt was blocked by `ERR_NVGPUCTRPERM`; permission counter access on WSL2 is gated by the Windows NVIDIA Control Panel toggle (see [[wsl2-ncu-perfcounters]] memory). Re-ran after the toggle + WSL restart; counters now work. Two practical lessons from the second attempt:

1. **`--set speedoflight` silently emits no metrics on sm_120** in ncu 2026.1.1 ("No metrics to collect found in sections"). Use `--set basic` instead — it contains the SpeedOfLight section plus Occupancy and WorkloadDistribution and works fine on consumer Blackwell.
2. **ncu only writes metric data on clean exit.** msieve doesn't terminate after `--launch-count` is satisfied (polyselect runs forever), so killing it with `SIGTERM` strands a metric-empty report — exactly what happened the first time. Fix: launch msieve with `-d 1` (1-minute deadline) so it cleans up on its own; ncu finalizes the report when the host process exits.
3. **CUB kernels appear without the `cub::` prefix** in ncu's kernel filter: use `regex:DeviceRadixSort`, not `regex:cub::Device`.

#### Step 2 — Speed-of-Light on the three trans launch shapes

`ncu --kernel-name regex:sieve_kernel_trans --launch-skip 20 --launch-count 5 --set basic` against `./msieve -np1 -g 0 -t 1 -d 1`. All five captures are `sieve_kernel_trans_pp64_r64`, three distinct launch shapes — the pattern within a batch is (big, medium, small):

| Shape (block × grid) | Threads | Duration | Compute SM% | DRAM% | L1% | L2% | Achieved occ | SoL verdict |
|---|---|---|---|---|---|---|---|---|
| (160) × (4282) — **big** | 685,120 | 1.06 ms | **74.9%** | **55.1%** | 31.3% | 36.2% | 60.0% | "Compute more heavily utilized than Memory" |
| (128) × (385) — medium | 49,280 | ~100 µs | low | low | ~9% | low | 55.7% | "low compute throughput and memory bandwidth … latency issues" |
| (256) × (1, 3) — small | 768 | 99 µs | 1.65% | 1.22% | 8.7% | 1.86% | 13.2% | "kernel grid too small … 0.02 full waves" |

The **big** launch carries essentially all the trans-kernel wall time and is the only one worth tuning. Occupancy is register-limited: 64 regs/thread caps theoretical at 62.5%; achieved 60% is close to that ceiling. The "small" launch is a 3-block setup/edge launch — too tiny to ever utilize the GPU, ignore.

#### Step 3 — Full profile on the big trans launch

`ncu --kernel-name sieve_kernel_trans_pp64_r64 --launch-skip 30 --launch-count 1 --set full`. 40 replay passes for the (4282 × 160) launch.

**Warp State Statistics — 12.2 cycles per issued instruction, top stalls:**

| Stall reason | cycles | % of issued-cycle time | What it means here |
|---|---|---|---|
| Stall Wait | 3.71 | **30.4%** | Fixed-latency execution dependency. Back-to-back math instructions waiting on the previous result — the Montgomery chain. |
| Stall Not Selected | 1.76 | 14.4% | Warp ready but scheduler picked another. Plenty of parallelism. |
| Stall Long Scoreboard | 1.48 | 12.1% | DRAM/L2 latency. Present but secondary. |
| Stall Short Scoreboard | 1.16 | 9.5% | Shared-mem / register dep. |
| Stall Math Pipe Throttle | 1.13 | 9.3% | Math units saturating but not the top stall. |
| Selected (issued) | 1.00 | 8.2% | Useful work. |

**Compute Workload Analysis:** Executed IPC = 2.36 / 4.0 (59%); Issue Slots Busy = 58.7%; SM Busy = 75.3%. Pipe utilization (% of elapsed cycles): ALU 37.9%, FP64 34.1%, FMA 15.8%, Tensor 0%. The FP64 number is surprising for an integer kernel — likely some 64-bit integer ops (modinv64 / 64-bit shifts / mulwide) routed to the DP unit on consumer Blackwell. Worth a separate look if trans becomes the next bottleneck.

**Memory Workload Analysis:** DRAM 345 GB/s (51% of ~675 GB/s peak); Mem Busy 20.9%; Max Bandwidth 51.5%; L1/TEX hit rate 44.8%; L2 hit rate 69.3%; Mem Pipes Busy 9.7%. Memory is being used hard enough to matter but is not the limiter.

**Uncoalesced global accesses:** 8% of sectors are excessive (2.5M / 33M sectors). Expected given the `num_entries`-stride writes at `stage1_core.cu:138-153`. Not the dominant cost, but real.

**Diagnosis:** the big trans launch is **compute-bound on the modular-arithmetic dependency chain**, not throughput-bound on any single pipe. The remedy is shortening or unblocking the chain — exposing more independent work per thread, batching inverses across q to reduce the modinv share, or reorganizing Montgomery multiplies so back-to-back instructions don't depend on each other. Adding warps won't help (scheduler already has eligible warps competing — see Stall Not Selected 14.4%).

#### Step 3b — Bonus: the medium launch is a different regime

Initially mis-targeted `--launch-skip 31` and captured the (128) × (385) launch. Kept it as `trans_full_medium.ncu-rep` because the contrast is informative:

| Metric | Big (4282 grid) | Medium (385 grid) |
|---|---|---|
| Top stall | Stall Wait 30.4% (math chain) | **Stall Long Scoreboard 70.4%** (DRAM latency) |
| Issue Slot Util | 58.7% | 22% (1 inst / 4.5 cycles) |
| Eligible warps/cycle | many | 0.48 |
| Compute pipe util | moderate | "All compute pipelines under-utilized" |

Same source code, two different bottlenecks depending on grid size. The big launch has enough warps in flight to hide DRAM latency and gets pinned on the math chain; the medium launch can't hide DRAM latency at all. This means a future trans optimization should be evaluated on *both* regimes — what helps the big launch may not help the medium one, and vice versa.

#### Step 4 — Sort comparator: DeviceRadixSortOnesweepKernel

`ncu --kernel-name regex:DeviceRadixSort --launch-skip 30 --launch-count 5 --set basic`. All five captures identical to within noise:

| Metric | Value |
|---|---|
| Compute SM throughput | 22.0% |
| **DRAM throughput** | **85.5%** |
| L1/TEX throughput | 24.8% |
| L2 throughput | 26.4% |
| Duration | 1.44 ms |
| Achieved occupancy | 49.5% |
| Registers/thread | 80 |
| Grid × block | 6014 × 384 |
| SoL verdict | ">80% of available compute or memory … memory most utilized" |

This is **decisively memory-bound at ~85% of peak DRAM bandwidth** (~575 GB/s of the ~675 GB/s peak). Combined with Drill 2's earlier finding that the key width is already at the minimum (50-bit minimum, rounded up to 56-bit = 7 Onesweep passes), the existing sort path is essentially at its theoretical ceiling. No amount of further sort tuning will move this number meaningfully.

#### Updated synthesis

The Step 1 nsys data (sort = 83.7% of GPU time) said *where* the time is going. The Steps 2-4 ncu data says *why*:

- **Sort: memory-bound at 85% DRAM.** Already near the architectural ceiling. The only way to make this kernel faster is to do less work; the only way to do less work is to replace the algorithm — which is exactly what the Gerbicz port does.
- **Trans (big launch): compute-bound on the Montgomery dependency chain.** Stall Wait dominates at 30.4%. Not throughput-limited on any pipe (FP64 34%, ALU 38%, FMA 16% — all moderate). DRAM at 51% of peak. The big lever here would be reducing or restructuring the per-thread math chain.
- **Trans (medium launch): latency-bound on DRAM.** A different bottleneck regime at small grid sizes. Anything claiming to optimize the trans kernel should be tested against both regimes.

This **confirms** the decision-table call from earlier: Gerbicz port is the right next step, trans tuning is secondary. Post-port, when trans becomes dominant, the priorities flip: math-chain restructuring (batch inversion across q, better SASS for modmul/modinv) becomes more interesting than the strided-write coalescing or the modinv share.

Report files in repo for later re-import: `trans_sol.ncu-rep` (Step 2), `trans_full.ncu-rep` (Step 3 big launch), `trans_full_medium.ncu-rep` (Step 3b medium launch), `sort_sol.ncu-rep` (Step 4). Each ~400 KB to 4 MB.

### Drill 1 — investigate the 1.24 TB memset

Of the 38,921 memset operations:
- **34,857 are `cudaMemsetAsync`** (Runtime API): CUB-internal scratch initialization. ~9 memsets per sort. Goes away if CUB sort is removed (Gerbicz port).
- **4,063 are `cuMemsetD8Async`** (Driver API): the explicit msieve calls in `stage1_sieve_gpu.c:660` clearing `gpu_root_array` between batches, plus the small `gpu_found_array` clear.

The explicit `gpu_root_array` clear is *necessary* only for `num_aprog_vals > 1`. Re-reading the trans kernel (`stage1_core.cu:429-464`): when a thread reaches `if (write_val == 0) return;` at line 444, the preceding loop has already written 0 to every slot in plane 0 for this thread. So for `num_aprog_vals == 1` (single plane), every slot is written by some thread — the memset is **redundant**.

For `num_aprog_vals > 1`, the additional planes are written only inside the post-early-return loop, so early-returning threads leave planes 1..(N-1) untouched — the memset clears them.

**Possible micro-optimization:** guard the memset with `if (num_aprog_vals > 1) { ... }`. But the impact is tiny: total memset wall time is 1.68 s out of 60 s (2.8%), and the explicit `gpu_root_array` clears are maybe a quarter of that. So saving ~0.4 s out of 60 s = ~0.7% total. Not worth pursuing — the Gerbicz port subsumes it (no fixed output array to zero between batches).

### Drill 2 — investigate the sort launch parameters

3,873 sorts × 7 Onesweep passes = 27,111 launches. CUB Onesweep does 1 pass per radix digit (8 bits). So 7 passes ⇒ ~56-bit keys.

Computing what `key_bits` should be from `stage1_sieve_gpu.c:917`:
```
key_bits = ceil(log2(p_max² × (num_aprog_vals+1)/2)) + (num_aprog_vals > 1 ? 1 : 0)
```
For this run: p_max = 22,972,992; p_max² ≈ 5.27e14; log2 ≈ 49.
Add log2((num_aprog_vals+1)/2) — for the common large-batch case `num_aprog_vals == 1`, this is 0; for end-of-coeff small-batch `num_aprog_vals` up to 5, this is ~1.5. So `key_bits` is 49–52 most of the time. Round up to next multiple of 8 ⇒ 56 bits ⇒ 7 Onesweep passes. **Matches observation.**

Can `key_bits` be reduced? The root values are in `(-pp/2, pp/2)` where pp = p² is up to ~5.3e14. So key_bits = 49 + 1 (sign) = 50 minimum to avoid wrap. Below that, sort produces false positives that the final kernel walks via the `if (root1 != root2) break;` early-exit. Estimated false positive cost at 48 bits with 50M entries: 50M² / 2^49 ≈ 4.4M false positives per batch — way too many. **Already at the practical minimum.**

**Conclusion:** the sort path is already well-tuned. CUB Onesweep is doing close to the minimum work for the key distribution. No easy reduction available within the sort path.

### Synthesis

Both drills confirm: the sort is the bottleneck (83.7%) and is already operating near its theoretical minimum given the problem's key width. There are no easy 2× wins within the existing sort-based pipeline. The Gerbicz collision port is the right lever — it bypasses the sort entirely and replaces it with a hash-bucket-and-filter chain whose work scales with the *number of candidate collisions*, not the total entry count.

For a follow-up after the Gerbicz port lands, the trans kernel (currently 10.6%) becomes the new dominant cost and *then* ncu-driven profiling is worth doing. At that point the questions become:
- Is the modular inverse really worth ~30 instructions per p per batch, or could it be amortized further?
- Is the strided `roots_out` write coalesced as well as possible?
- Can the trans+collision kernels be fused to avoid the intermediate DRAM round-trip?

Steps 2-4 have now been run with counters enabled — see the "ncu Speed-of-Light + full + sort comparator" section above. Re-run the same commands once the Gerbicz port is in tree; the questions will have moved but the methodology stays.

---

## Integration status update (2026-05-23)

The Gerbicz collision path described in the implementation plan below has been ported into msieve and is available behind opt-in NFS arguments:

```bash
./msieve -np1 "collengine=gerbicz" ...
```

Keep the quoted NFS argument string immediately after `-np1`; otherwise msieve may not route the options into the polyselect argument parser.

Current flags:

- `collengine=gerbicz` loads `cub/collision_engine.so` and replaces CUB sort plus `sieve_kernel_final_*`.
- `colllib=PATH` overrides the collision DSO path.
- `collhash=1` is the default bucket-hash mode; `collhash=0` uses low-bit bucketing.
- `collstats=1` prints compact per-coefficient collision stats.
- `colldebug=1` prints the per-batch growth/cap diagnostics that were useful while debugging real-data crashes.

What is now done:

- Added `cub/collision_engine.{h,cu}` with `collision_engine_init/free/run`.
- Updated the top-level CUDA build to produce `cub/collision_engine.so`.
- Hooked `stage1_sieve_gpu.c` so the new engine consumes the trans-kernel root/value arrays and emits directly to the existing `found_array`.
- Preserved the old sort path as the default.
- Added exact-value standalone verification with injected duplicates, including comparison against the msieve hit predicate.

Real-data fixes made during integration:

- The collision scatter skips zero roots because the root array is pre-cleared and the old final kernels ignore `root == 0`.
- Bucket capacity grows and retries based on observed max bucket size.
- Phase-B hash-table words are capped to the device opt-in shared-memory limit.
- Bucket growth and hash-cap prints are quiet by default and gated behind `colldebug=1`.

First fixed-coefficient result from the user:

| Path | Real time |
|---|---|
| Existing sort path | 4m39.169s |
| `collengine=gerbicz` | 2m35.957s |
| `collengine=gerbicz collhash=0` | 2m37.059s |

All three output files had the exact same size. That is a useful smoke test, but the next correctness pass should compare contents or run several fixed ranges through downstream validation.

Next work: all four items from this list have been completed. See the post-integration profiling section immediately below for the results, and [TWO_LEVEL_SCATTER_PLAN.md](./TWO_LEVEL_SCATTER_PLAN.md) for the next optimization.

---

## Post-integration profiling (2026-05-24)

The integration's "Next work" list above has been fully retired. Summary of what we found.

### Parity diff (item 1 — done)

Ran sort path, `collengine=gerbicz`, and `collengine=gerbicz collhash=0` on the test composite at coeff 420. Canonical-sorted each `.ms` polynomial output (each polynomial is a 10-line record beginning with `n:`) and diff'd. All three runs produce the **exact same 12,739-polynomial set**, byte-equivalent up to ordering. Sort vs. gerbicz hit-set parity confirmed at the polynomial level.

### `collstats=1` headroom (item 2 — done)

Added in `stage1_sieve_gpu.c`. Two coeffs sampled:

| Metric | Per batch (coeff 420, 12,456 batches) | Per batch (coeff 180180, 202,846 batches) | Cap | Headroom |
|---|---|---|---|---|
| candidates | 5,558 | 11,074 | 4,194,304 | 380×+ |
| dedup | 1,393 | 4,897 | 4,194,304 | 850×+ |
| matched | 2,827 | 9,847 | 4,194,304 | 425×+ |
| max_bucket | 2,414 (max) | 2,438 (max) | heuristic | grows=0 across both runs |
| **matched/dedup** | **2.03** | **2.01** | — | filter converges to ~2 matches/slot |
| hash_caps | 0 | 0 | — | shared-mem cap never triggered |

A new `found_array stats` block was added to `check_found_array` (always-on, prints when `t->found_batches != 0`) — surfaces `peak`, `saturated`, `total` so silent-drop saturation at `FOUND_ARRAY_SIZE=1000` is visible:

- coeff 420: peak **11** hits/batch, saturated 0, total 12,739
- coeff 180180: peak **23** hits/batch, saturated 0, total 560,042 (in 30:25 wall)

All six pre-trans concerns from the original review are dead:

| Concern | Result |
|---|---|
| `found_array` silent drops | peak ≤ 23 of 1000 cap (43×+ headroom) |
| `CANDIDATE_CAP` overflow | 380×+ headroom |
| `VALUE_MATCH_CAP` overflow | 425×+ headroom |
| `emit_found_kernel` hot-slot O(n²) | matched/dedup = 2.0 → trivial inner loop |
| `ensure_capacity` realloc churn | `bucket_grow_count = 0` |
| Hash-table opt-in shared-mem cap | `hash_caps = 0` |

### nsys kernel breakdown (item 3 — done)

Coeff 420, `LOG2_NUM_BUCKETS=14` (the default), captured under nsys:

| Kernel | % GPU | ms/launch | Notes |
|---|---|---|---|
| `sieve_kernel_trans_pp64_r64` | 34.3% | 0.66 | Independent host trans path (unchanged) |
| **`scatter_roots_kernel`** | **31.2%** | **1.80** | Phase A — biggest engine kernel |
| `filter_per_bucket_kernel` | 17.3% | 1.00 | Phase B (ping-pong filter) |
| `count_matched_values_kernel` | 8.2% | 0.47 | Phase C count pass |
| `scatter_matched_values_kernel` | 8.1% | 0.46 | Phase C scatter pass |
| `emit_found_kernel` | 0.1% | 0.004 | Microscopic |
| All CUB sort/scan/reduce inside engine | <1% | — | Negligible — candidate set is small |
| Memsets | 8% (cuda_gpu_mem_time_sum) | — | Mostly tiny per-batch counter clears |

Engine kernels collectively = **65% of GPU time**, trans = 34%. `scatter_roots` is the next optimization target, not trans.

### Bucket-count sweep — `NUM_BUCKETS` is the wrong knob (item 4 reframed — done)

Tested `LOG2_NUM_BUCKETS ∈ {12, 13, 14, 15, 16, 17}` (4K → 131K buckets). All produce **identical dedup (17,344,961) and matched (35,196,265) counts** — strong correctness signal that bucket count doesn't affect the hit-set.

| LOG2 | NUM_BUCKETS | Wall | Engine | max_bucket | candidates emitted |
|---|---|---|---|---|---|
| 12 | 4,096 | 147s | 105.8s | 9,023 | 95.4M |
| 13 | 8,192 | 126s | 82.3s | 4,619 | 97.8M |
| **14** | **16,384** | **124s** | **81.6s** | 2,414 | 69.2M |
| 15 | 32,768 | 121s | 79.7s | 1,273 | 52.2M |
| 16 | 65,536 | 141s | 98.6s | 676 | 41.8M |
| 17 | 131,072 | 161s | 119.8s | 370 | 37.7M |

**LOG2=14 is empirically optimal.** LOG2=15 looks 3% better on wall but it's within the run-to-run noise (~5%). A fresh nsys at LOG2=15 confirmed the wall "win" was illusory: `scatter_roots` got **39% slower per launch** (1.80 → 2.50 ms) because doubling the bucket count doubles the `bucket_count[]` array (now 128 KB, no longer fits in L1) and fragments writes across more L2 cache lines. Filter dropped 37% per launch as expected, but the engine kernels in total got **4.5s worse**, not better. Net wall improvement was noise.

**Implication: bucket count is a balance lever between scatter cost and filter cost, not a coalescing knob.** The original choice of 16K buckets was correct. Two-level scatter must attack the kernel itself, not via bucket count.

The interesting non-perf finding from the sweep is the **2.5× candidate-volume swing** (95M at LOG2=12 → 38M at LOG2=17). With fewer buckets, Phase B's filter has more items per bucket and lets more false positives through. That extra candidate volume is real work for the downstream small CUB sort, dedup, and value-match kernels — but their fixed-cost dominates and the swing doesn't translate to a wall difference at the current 16K operating point.

### Lessons from the sweep process (worth preserving)

- **`make CUDA=1` without `all` does nothing** — prints the help banner and exits 0. Silent rebuild failure burned an entire sweep before we caught it. The fix: always use `make CUDA=1 all`, and verify `.so` mtime changed before trusting the run. Saved to memory.
- **`-np1 "collengine=gerbicz" -nps "..."`** does not route `collengine=` correctly — must put the engine flag inside the `-nps` quote string: `-np1 -nps "min_coeff=420 max_coeff=420 collengine=gerbicz collstats=1"`. Earlier integration update had this wrong; runs that look like gerbicz actually fell back to the sort path. The msieve.log line "using Gerbicz GPU collision engine" is the load-bearing confirmation.

### Next optimization

See **[TWO_LEVEL_SCATTER_PLAN.md](./TWO_LEVEL_SCATTER_PLAN.md)** for the detailed scatter/value-path history. Current status from 2026-05-24:

1. **Scatter-local work is deprioritized.** `__stcs` and direct-atomic diagnostics were parity-clean but did not move `scatter_roots_kernel`; ncu shows it is already at an L2 throughput ceiling.
2. **Fuse `count_matched_values` + `scatter_matched_values` is implemented.** The bounded arena path (`MATCH_ARENA_WIDTH=8`) reduced coeff-420 matched-value path GPU kernel time from **11.698s to 6.181s**. Diagnostics show **3 / 12,456** fallbacks on coeff 420 and **0 / 92,454** fallbacks on coeff 60,060, with no capacity skips.
3. **Next low-risk lever: memset cleanup.** nsys still shows large fixed per-batch memset activity. Coalesce contiguous engine clears first; only revisit the old `gpu_root_array` memset if it still shows up materially.
4. **Next high-ceiling lever: trans kernel math-chain restructure.** Per the original ncu drill: 30.4% Stall Wait on the Montgomery dependency chain. Batch-inversion across q, or warp-cooperative `modinv`. Expected ~5-10% wall win, but higher effort.
5. **Fused trans + bucket-scatter** — eliminates the ~80 MB intermediate root-array DRAM round-trip. Bigger restructure, deferred until trans tuning is also done.

---

# Implementation plan: porting Gerbicz collision search to CUDA

If JasonP's port doesn't land and you want to do it yourself. The Step 1 profile data establishes that the sort is 83.7% of GPU time — the right place to invest. Goal: replace `cub::DeviceRadixSort::SortPairs` + `sieve_kernel_final_64` with a CUDA implementation of the bucket-and-ping-pong-filter algorithm in `collision_ref.c`.

The trans kernel stays unchanged. Only sort+final get replaced.

## Architecture

Three CUDA kernels mirroring `collision_ref.c`'s phases, plus an integration shim.

### Phase A — bucket scatter kernel
New file: `gnfs/poly/stage1/stage1_core_gpu/collision_bucket.cu` → `__global__ void bucket_scatter(...)`

Input: `(keys[N], values[N])` produced by trans kernel. `keys` are 32 or 64 bit roots; `values` are packed `(q_idx << shift) | p`.
Output: linked-list-of-chunks structure, one linked list per bucket.

Design choices to lock in early:
- **1024 buckets** (cf. 256 in `collision_ref.c`). More buckets reduce atomic contention on the bucket counter and produce smaller per-bucket hash tables that fit in shared memory in Phase B. Bucket index = `key & 0x3FF`.
- **256-entry chunks** (match `collision_ref.c`). Each chunk holds 256 × (key, value) pairs ≈ 3 KB for 64-bit keys + 32-bit values.
- **Warp-aggregated atomic** for bucket counter increment. Use `__match_any_sync` (CC ≥ 7.0, supported on Blackwell):
  ```
  mask = __match_any_sync(0xffffffff, bucket)
  leader = __ffs(mask) - 1
  count = __popc(mask)
  if (lane == leader) base = atomicAdd(&bucket_offset[bucket], count)
  base = __shfl_sync(0xffffffff, base, leader)
  my_offset = base + __popc(mask & ((1u << lane) - 1))
  ```
  This collapses 32-way same-bucket contention to one atomic per matched group per warp.
- **Chunk allocation**: single global counter `next_chunk_id`. When a bucket's offset wraps to a new chunk (`offset & 0xff == 0`), do `new_chunk = atomicAdd(&next_chunk_id, 1)` and `nextblock[old_chunk] = new_chunk`.

Storage budget for 50M entries (typical RSA-768 batch):
- Chunks: 50M / 256 ≈ 200k chunks × ~3 KB = ~600 MB
- `nextblock[]`: 200k × 4 bytes = 0.8 MB
- Bucket heads: 1024 × 4 bytes = 4 KB
- Total: well within 12 GB on RTX 5070

### Phase B — per-bucket ping-pong filter kernel
New file: `gnfs/poly/stage1/stage1_core_gpu/collision_filter.cu` → `__global__ void filter_per_bucket(...)`

Input: Phase A's chunk structure.
Output: per-bucket candidate list appended to a global `candidate_array`.

Design choices:
- **One block per bucket** (1024 blocks total). Each block walks its bucket's linked-list chunks via `nextblock[]`.
- **Hash table in shared memory**. For ~50M / 1024 ≈ 50k items per bucket, the hash table at 4× sizing is ~16k bits ≈ 2 KB — tiny. Plenty of shared mem headroom for the no-loss `T` + `T2` + `T3` arrays (cf. `collision_ref.c:317-322`).
- **No-loss variant** to start (`collision_ref.c:271-282` path) for parity with the current sort-based path. Add allow-loss as a tunable later.
- **5 ping-pong rounds**, each shifting the hash by +6 bits (matches `collision_ref.c:324`).
- **Candidate emit**: surviving items appended via `atomicAdd` to a global counter on the candidate array. Expected output: ~tens of candidates per bucket = ~tens of thousands globally.

### Phase C — verify kernel
New file: `gnfs/poly/stage1/stage1_core_gpu/collision_verify.cu` → `__global__ void verify_candidates(...)`

Input: candidate list from Phase B (small — fits in DRAM trivially).
Output: `found_array` entries in the same format `handle_collision` expects.

Implementation:
1. Sort the candidate list with a small CUB sort (low launch count; small data; cheap).
2. Dedup adjacent entries (mirror `collision_ref.c:386-390`).
3. Build a hash on the small sorted list (mirror `collision_ref.c:395-415`).
4. Scan the original `keys[N]` array; for each match, retrieve the corresponding `values[N]` entry (which has packed q-index + p).
5. Apply the same checks as `sieve_kernel_final_64` (`stage1_core.cu:669-678`):
   ```
   if ((p1 >> shift) == (p2 >> shift) && gcd32(p1 & mask, p2 & mask) == 1)
       store_hit(...)
   ```

### Integration shim
New file: `cub/collision_engine.cu` — analogous to existing `cub/sort_engine.cu`. Builds to `cub/collision_engine.so`.

Exports (mirror `sort_engine.h`):
```c
void *collision_engine_init(void);
void  collision_engine_free(void *);
void  collision_engine_run(void *, collision_data_t *);
```

Where `collision_data_t` is `sort_data_t` plus a `found_array` pointer (since the engine now does both sort-replacement and final-scan in one call).

In `gnfs/poly/stage1/stage1_sieve_gpu.c`, branch on a CLI flag analogous to `sortlib=`. Default to existing CUB path; add `collengine=gerbicz` to opt into the new path. In `handle_special_q_batch`, when the new engine is selected, skip the CUB sort call and the `sieve_kernel_final_*` launch — call `collision_engine_run` instead.

## Build system

`cub/Makefile` already builds `sort_engine.so` separately. Add parallel rules for `collision_engine.so`. Targets: `-arch=sm_120` for your RTX 5070, plus `sm_89`/`sm_90` for portability.

## Testing strategy

### 1. Standalone correctness (do this first, before any msieve integration)
Port `collision_ref.c`'s `main()` driver to a CUDA test executable that runs both the GPU kernels and `collision_ref.c`'s slow-quicksort verification on the same synthetic data. Must match hit counts exactly in no-loss mode. Run at the user's actual problem sizes (~50M 56-bit keys for the 186-digit composite).

### 2. Integration parity
Build msieve with both paths available. Factor a small composite (e.g., a 130-digit RSA challenge) end-to-end with each path. Polynomial outputs should be equivalent within polyselect's randomness.

### 3. Performance validation
On the user's 186-digit composite, measure wall time per coeff with both paths. Target: ≥3× speedup on the collision portion (JasonP's reported 4× minus implementation tax).

## Risks and open questions

1. **Bucket contention.** With 1024 buckets and ~50M keys, average is ~50k items per bucket but the *peak* matters more. Worst-case unbalanced bucket might be 5–10× the mean — that block has to do more work, and the kernel launch waits for it. Measure bucket size variance first; if peak/mean > 4×, consider hashing the bucket index instead of taking low bits.

2. **Memory pressure at larger N.** Bucket chunks ~600 MB for 50M entries; scales linearly. For RSA-1024-scale (N ≈ 10⁹) this would blow past 12 GB. Need a strategy for processing in tiles. Not a Day-1 concern but worth flagging.

3. **Allow-loss tradeoff.** 6-12% missed collisions in allow-loss mode means 6-12% fewer stage-1 hits per unit time. But the throughput is roughly 2× higher, so net wins. Start with no-loss for parity; add allow-loss as `-collopt=allow_loss` once the no-loss path is solid.

4. **Output ordering.** Current pipeline produces hits in sort order (root-ascending). The new pipeline produces hits in candidate-scan order (arbitrary). If anything downstream relies on hit ordering, that breaks. Unlikely but check `handle_collision`'s callers.

5. **Found-array overflow.** The current `FOUND_ARRAY_SIZE` budget assumes a sort-based pipeline. Verify it's still sized appropriately for the new path — the candidate-scan may produce hits more bursty than the sorted-adjacency scan.

## Estimated effort

| Component | Effort | Notes |
|---|---|---|
| Standalone CUDA collision_ref.c port (test harness) | 3 days | Critical first step; catches bugs cheaply |
| Phase A bucket scatter kernel | 3 days | Warp aggregation tuning is the hard part |
| Phase B per-bucket filter kernel | 2 days | Shared-mem layout iteration |
| Phase C verify kernel | 1 day | Mostly mechanical |
| Integration shim + msieve hookup | 2 days | Including build system, CLI flag |
| Integration testing + perf tuning | 3 days | Includes parity test on small composite |

**Total: ~2 weeks of focused work** assuming you don't get sidetracked by Blackwell-specific tuning issues (block sizes, occupancy targets, shared mem budget tradeoffs).

## Day 1 starting point

When you decide to pick this up:

1. **Establish CPU baseline.** Build `gerbicz_bench/collision_ref.c` (compilation line is on line 5 of the file) and run it at your real problem size:
   ```
   cd gerbicz_bench
   g++ -m64 -O2 -fomit-frame-pointer -mtune=corei7 -march=corei7 -mavx2 -o collision collision_ref.c -lm
   ./collision 56 50000000 0   # 56-bit keys, 50M entries, no-loss
   ```
   The "found N near-collisions in T seconds" line is your CPU reference. Compare against the slow-qsort verification in the same output.

2. **Profile that CPU run.** `perf stat -e cache-misses,cache-references,instructions ./collision 56 50000000 0` tells you where it spends time. The bucket scatter and the per-bucket filter rounds dominate on CPU; on GPU they'll have different characteristics but knowing the CPU shape helps.

3. **Write the standalone CUDA port** in something like `cub/collision_engine_test.cu`. Mirror `collision_ref.c`'s `main()`. Make it produce a hit count that matches both the CPU implementation and the slow qsort. Don't touch msieve yet.

4. **Iterate on the CUDA port** until it beats the multithreaded CPU `collision_ref.c` on your hardware. If it doesn't beat CPU, integration is pointless.

5. **Then start the msieve integration** following the Architecture section above.

## References worth keeping handy

- `gerbicz_bench/collision_ref.c` — the algorithmic reference; **don't delete**, the plan cites its line numbers.
- `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu` — existing trans + final kernels for the integration interface.
- `gnfs/poly/stage1/stage1_sieve_gpu.c:660` — the `cuMemsetD8Async` of `gpu_root_array` (need to drop when sort goes away).
- `cub/sort_engine.cu` — model for the new `collision_engine.cu` integration shim.
- mersenneforum.org threads on Gerbicz's algorithm — discussion has been going for over a year; any prototypes JasonP or Ben Broman shared are worth pulling in before starting from scratch.

---

## Background — discussion of CUDA-side improvement ideas

(Kept here so the plan file is self-contained for sharing.)

### What the current msieve GPU pipeline does

Per special-q batch:

1. **trans kernel** (`sieve_kernel_trans_pp{32,64}_r{32,64}` in `stage1_core.cu`) — one thread per p in the progression set, iterating over the q-batch. Computes packed key `(q_idx << shift) | p` and root `(start_root − q_root) · (q² mod p²)⁻¹ mod p²`. One `modinv` per p per batch plus many Montgomery multiplies. Writes strided by `num_entries`.
2. **CUB radix sort** (`sort_engine.cu`) on ~10⁸ key-value pairs for RSA-scale.
3. **final kernel** (`sieve_kernel_final_{32,64}`) — linear scan for adjacent equal sort keys, with same-q + gcd(p1,p2)==1 check.

### What Gerbicz does (per `collision_ref.c`)

1. **Bucket scatter** (lines 213–223): 256 buckets indexed by low 8 bits of key; each bucket is a linked list of 256-entry chunks. Chunks atomically claimed from a global counter. Total storage O(n).
2. **Per-bucket ping-pong filter** (lines 247–382): build a small bit-array hash table per bucket; mark slots with ≥2 items; filter; change hash bits; repeat ~5 times. ~99% reduction per round.
3. **Verify** (lines 384–429): sort tiny candidate list, scan original keys.

Win: most items eliminated by cheap local hash ops on bucket-sized arrays that fit in cache. No global sort.

### CUDA improvement ideas (ranked)

1. **Direct port of Gerbicz** (what JasonP is doing). Warp-aggregate atomics via `__match_any_sync` for the bucket scatter; choose bucket count (256 vs 1024 vs more) so per-bucket hash table fits in shared memory (~64 KB ⇒ 1024 buckets for n ≈ 30M); persistent block-per-bucket for L2 residency.
2. **Warp `__match_any_sync` shortcut** in late filter rounds — once candidate set is dense, intra-warp matches are common and skippable past the ping-pong.
3. **Fuse trans + collision into one persistent kernel** — produce keys, bucket as produced, run ping-pong, emit hits, all in one launch. Avoids materializing ~1+ GB of intermediate keys per batch.
4. **Stream pipelining / CUDA Graphs** — overlap CPU q-batch production with GPU collision search. Most of the framework already exists; need to overlap host-side `sieve_fb_next` with GPU compute.
5. **Trans kernel batched modinv** — initially thought promising via Montgomery's trick, but `stage1_core.cu:88–164` already chains the inversion across q for fixed p. So per-thread the trick is already exploited; cross-thread batching is blocked by per-p moduli.
