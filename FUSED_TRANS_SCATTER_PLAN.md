# Fused trans + bucket-scatter design

> **Status (2026-05-25, v3 — Option A spike RAN, REVERTED):** Full implementation passed all correctness gates but failed the first GPU performance gate decisively. Reverted. Foundation retained. Summary:
> - **Polynomial parity:** byte-exact (md5 `ef7d7676...`, 12,739 polys). The design (emit-after-store, active-mask aggregation, ONLY at final-write sites, shared bucket header, exact-bit-pattern key) was correct — no parity bugs.
> - **Collstats:** matched baseline within 4 keys of 35.2M (~10⁻⁷), exactly the variant-B-flavor intra-warp atomic-ordering drift between `0xFFFFFFFFu` and `__activemask()`. Harmless.
> - **First GPU gate FAILED:** `fused_trans - trans_baseline = +39.6s; scatter_roots_baseline = -23.0s saved.` Trans grew by **more than** scatter saved — net +16.6s GPU kernel time (+25%).
> - **ncu on dominant launch (224×951):** registers 56 → **72 (+16)**; block_limit_registers 5 → **4 (lost a block/SM)**; theoretical occupancy 72.9% → 58.3% (-14.6 pp); per-launch duration 1.20 ms → **2.59 ms (+116%)**; zero local-mem spills.
> - **Codex rescue (direct atomic, no warp aggregation) FAILED**: regs stayed at 72, duration unchanged at 2.59 ms — the register growth comes from the persistent bucket-arg state and bucket_storage write, not the aggregation logic.
> - **Wall A/B 3 pairs (collfuse=0 vs collfuse=1, no collstats):** baseline 1:29.3 / 1:27.7 / 1:28.0 → median **1:28.0**; fused 1:39.9 / 1:44.6 / 1:43.8 → median **1:43.8** = **+15.8s wall regression (+18%)**.
> - **Verdict:** Option A is closed. The math chain genuinely needs its 56 registers — same lesson as T3 (going the opposite direction). Foundation kept (shared bucket header, GPU_MAX_KERNEL_ARGS=20, Makefile deps, filter iters histogram, gpu_kernel_args_idx[] refactor) since all are zero-overhead or actively useful for diagnostics.

> **Status (2026-05-25, v0 — design only, no code):** Last structural lever standing after T3, tail-effect, A1/A.5, variant B, and oversized-bucket-split all retired. Top 3 kernels (trans 37%, scatter_roots 34%, filter 19%) are at their respective ceilings against in-kernel tuning. Fusion is the only remaining lever that *removes an entire intermediate write/read path* rather than shaving saturated kernels. This document is the contained-spike design per Codex's framing; user reviews before any implementation commitment.

> **Status (2026-05-25, v1 — Codex review folded in, still no code):** v0 had 12 actionable Codex comments — 5 correctness-critical, 3 framing-accuracy, 4 API/portability. All folded in below. Summary of substantive changes from v0:
> - **Memory framing tightened** — "scatter goes to 0 as work" was wrong; the atomic + bucket_storage write moves into trans. Real savings: 1 read of gpu_root_array + 1 kernel launch + cache/scheduling. Wall ceiling not budgeted at 15-30% until measured.
> - **Option B accounting corrected** from -560 MB to -700 MB (after subtracting the bucket_storage payload growth).
> - **Kernel sketch rewritten** with `__activemask()`-based participating mask (the `0xFFFFFFFFu` mask is unsafe inside divergent trans loops), an `emit_fused_key(stored_key, ...)` helper called *after* `roots_out[...] = ...` to bucket the exact-bit-pattern key, and explicit guidance to emit ONLY at final write sites (NOT at the qq_prod scratch writes or zero-fill columns).
> - **Shared bucket-hash header** — fused kernel must use identical `NUM_BUCKETS`, `LOG2_NUM_BUCKETS`, `BUCKET_MASK`, `BUCKET_HASH_MIX`, `compute_bucket()` as `collision_engine.cu`; design now requires a shared header (or a validation assert if duplication is unavoidable).
> - **Overflow fallback fixed** — old "reset bucket_count + re-run scatter_roots" would loop forever at the same `max_per_bucket`. Now: read observed max from `d_bucket_count`, call `ensure_capacity()` to grow, then re-run scatter_roots from gpu_root_array.
> - **API hygiene** — exported DSO boundary stays in `CUdeviceptr` (not raw host pointers); pp64_r64-only spike has an explicit `pp_is_64 && root_bytes == sizeof(uint64)` gate; launch-table touchup (`GPU_*` enum, `gpu_kernel_names`, `gpu_kernel_args[i/3]` grouping); multi-SOA wording uses `p_array->num_arrays` not the degree-5-specific `{1, 5, 25}`.
> - **Timing measure** — don't compare `collision_engine_ms` after fusion (some former engine work now runs inside the trans launch); authoritative measures are nsys total CUDA kernel time and whole-batch wall.

> **Status (2026-05-25, v2 — Codex second-pass folded in, ready for implementation):** v1's 3 second-pass blockers all verified against codebase and folded into the relevant sections. None changed the design strategy; all were "v1 underspecified the actual mechanics" gaps that would have failed at build/load time:
> - **Makefile rebuild dependencies (Codex #13):** `cub/collision_bucket.h` must be added to `NFS_GPU_HDR` (so PTX rebuilds when bucket constants change) AND to `cub/built` prereqs (so engine .so rebuilds). Critically the existing PTX rule (`stage1_core_sm{89,90}.ptx`) invokes `nvcc` *without* any `-I.` — needs matching `-I. -Ignfs -Ignfs/poly/stage1` added so the shared header is findable. Without these fixes, header edits silently produce stale builds.
> - **GPU_MAX_KERNEL_ARGS=15 is a runtime memory-corruption blocker (Codex #14):** `include/cuda_xface.h:86` currently caps kernel args at 15; `gpu_launch_t::arg_offsets[]` and `gpu_arg_type_list_t::arg_type[]` use that fixed size. A 17-arg fused signature would silently overflow at runtime. Bump to **20** (cleanest spike move — small, no immediate need to repack); revisit packing later if more growth happens.
> - **Loader symbol resolution (Codex #15):** `stage1_sieve_gpu.c:498-500` only stores `collision_engine_{init,free,run}` function pointers; `load_collision_engine():1424-1427` only resolves those three via `get_lib_symbol`. The new `collision_engine_prepare_fused` export needs: function-pointer typedef in `collision_engine.h`, new field in `device_data_t`, and a `get_lib_symbol` resolve in `load_collision_engine()` (non-fatal NULL check so older engine .so files still load and just disable `collfuse`).
>
> Effort estimate bumped from 8 → 8.5 days. Files-to-touch list extended (`include/cuda_xface.h`, `Makefile`). Ready for implementation.

## Ceiling estimate

### Current memory traffic per batch (coeff 420, ~140M trans entries)

| Op | Direction | Size | Where |
|---|---|---|---|
| trans → gpu_root_array | W | 280 MB | `sieve_kernel_trans_pp64_r64` |
| trans → gpu_p_array | W | 140 MB | same |
| scatter_roots ← gpu_root_array | R | 280 MB | `scatter_roots_kernel` |
| scatter_roots → bucket_storage | W | 280 MB | same |
| count_and_store ← gpu_root_array | R | 280 MB | `count_and_store_matched_values_kernel` |
| count_and_store ← gpu_p_array | R | 140 MB | same |
| **Total trans+scatter+count_and_store path** | | **1400 MB** | |

### After Option A (minimal fusion)

| Op | Direction | Size | Where |
|---|---|---|---|
| fused_trans → gpu_root_array | W | 280 MB | new `sieve_kernel_trans_fused_*` |
| fused_trans → gpu_p_array | W | 140 MB | same |
| fused_trans → bucket_storage | W | 280 MB | same (warp-aggregated atomic, but now inside trans) |
| (scatter_roots_kernel SKIPPED) | | -560 MB | -1 read (280 MB), -1 write (280 MB) |
| count_and_store ← gpu_root_array | R | 280 MB | unchanged |
| count_and_store ← gpu_p_array | R | 140 MB | unchanged |
| **Total** | | **1120 MB** | |

**Memory delta: -280 MB per batch (the gpu_root_array READ that scatter_roots used to do).** At ~12,456 batches/coeff this is ~3.5 TB of saved memory traffic per coeff 420 run.

What is **not** saved: the bucket compute, atomic-add to `bucket_count`, and the 280 MB `bucket_storage` write — these all happen now, just inside trans rather than inside scatter_roots. The clean wins are: the 280 MB read, one kernel launch (and the ~12k launch-overhead saving per coeff), plus whatever cache/scheduling benefit comes from doing the bucket write while the key is still hot in registers.

### Realistic wall ceiling

The first gate is *measured*: whether `fused_trans - trans_baseline < scatter_roots_baseline`. If yes, fusion is a net win on the GPU. If no, the atomic/write work absorbed into trans cost more than the saved kernel — net regression. Below that gate, there are three plausible regimes:

- **Net win:** trans absorbs the scatter work with some hide-behind-existing-latency benefit. Engine wall improves; magnitude TBD.
- **Net neutral:** trans grows by approximately scatter_roots's full cost. Kernel-launch overhead saved, but per-batch wall change is in the noise.
- **Net regression:** trans's register budget tightens enough to hurt the math chain, OR the added atomics serialize. Trans grows by more than scatter saves. **Primary risk.**

The v0 numbers ("15-30% engine win") were budgeted before measurement. I'm dropping that until we have an nsys A/B. Decision criteria below are framed in deltas, not absolutes.

## The two fusion options

### Option A — minimal (recommended for first spike)

- New kernel `sieve_kernel_trans_fused_pp64_r64` (initially only pp64_r64; pp32 variants later if A wins).
- Same as current trans kernel, but at each *final* write to `roots_out[j]` (NOT at the qq_prod scratch writes or zero-fill columns — see kernel structure section), also:
  - Compute bucket from the *exact bit pattern* just stored in `roots_out`
  - Warp-aggregated atomicAdd to `bucket_count[bucket]` using the *active participating mask* (NOT a hard-coded `0xFFFFFFFFu`)
  - Write key to `bucket_storage[bucket * max_per_bucket + slot]`
- gpu_root_array and gpu_p_array writes UNCHANGED (count_and_store still reads them).
- scatter_roots_kernel SKIPPED on the flag-on happy path; retained as the overflow fallback.

**Pros:** isolated change, no downstream rework, exact-fallback path available (revert to scatter_roots_kernel after grow on overflow), incremental testing.

**Cons:** trans output is still materialized (we don't save the 280 MB gpu_root_array write or the 280 MB count_and_store read of it). Caps the ceiling at the scatter elimination.

### Option B — full restructure (deferred)

- Eliminate `gpu_root_array` and `gpu_p_array` entirely.
- bucket_storage holds packed `(key, value)` pairs (12 bytes/slot vs current 8) OR holds `(key, original_trans_index)` (still 12 bytes).
- count_and_store rewritten to walk bucket-by-bucket instead of linear scan over trans output.
- Memory delta: gpu_root_array R+W (560 MB) + gpu_p_array R (140 MB) eliminated = **840 MB savings**, offset by the bucket_storage payload growing from 8→12 bytes per slot ≈ **+140 MB** at this scale. **Net: ~700 MB/batch** (not 560 MB as v0 estimated).

If the design instead keeps an `original_trans_index` and still has count_and_store read `gpu_p_array` by index, then gpu_p_array is NOT eliminated and the savings analysis needs another revision.

**Cons:** count_and_store rewrite is a significant chunk of work; bucket-walk vs linear-scan may not be faster (depends on per-bucket hash lookup structure). High effort, uncertain win.

**Recommendation:** Option A first. If A passes all gates AND the saved memory traffic is bounded by the remaining gpu_root_array+gpu_p_array reads, then consider B.

## Concrete kernel structure (Option A)

Edit `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu`. Add a new kernel `sieve_kernel_trans_fused_pp64_r64`. The body is the existing kernel's body with a single helper-call insertion at the *final* write sites.

### The helper

Define a `__device__` helper called only by lanes that have a non-zero key to emit. It captures the active participating mask (NOT `0xFFFFFFFFu`) and uses that same mask for both the match and the shuffle. The key passed in must be the exact `uint64` bit pattern that was just stored in `roots_out` — including the centered-negative pp64 reduction.

```cuda
__device__ static __forceinline__ void
emit_fused_key(uint64 stored_key, int hash_mode,
		uint32 *bucket_count, uint32 *bucket_overflow,
		uint64 *bucket_storage, uint32 max_per_bucket)
{
	/* Skip-zero contract matches scatter_roots_kernel:151 — count_and_store
	   also drops zero keys, so bucketing zeros would be wasted work. */
	if (stored_key == 0)
		return;

	uint32 bucket = compute_bucket(stored_key, hash_mode);

	/* Use the active mask. Fused emission lives inside trans's divergent
	   walk-back / backwards loops; lanes that exited early or hit
	   different `if (curr_qq_prod == 0)` branches must not be assumed
	   present. The same mask must be passed to both __match_any_sync
	   and __shfl_sync. */
	uint32 active = __activemask();
	uint32 lane = threadIdx.x & 31u;
	uint32 mask = __match_any_sync(active, bucket);
	uint32 leader = __ffs(mask) - 1u;
	uint32 count = __popc(mask);

	uint32 base = 0;
	if (lane == leader)
		base = atomicAdd(&bucket_count[bucket], count);
	base = __shfl_sync(active, base, leader);

	uint32 within = __popc(mask & ((1u << lane) - 1u));
	uint32 slot = base + within;
	if (slot >= max_per_bucket) {
		atomicOr(bucket_overflow, 1u);
		return;
	}
	bucket_storage[(size_t)bucket * max_per_bucket + slot] = stored_key;
}
```

### Where to emit

**ONLY** at the final write sites that also write the matching `p_out[j] = (q << shift) | p`. These are the writes inside the walk-back / backwards loops (around `stage1_core.cu:545-548` and `:555-558` for pp64_r64) — the keys at these sites are the actual collision-input data.

**Do NOT emit** from:

1. The early `roots_out[qq_prod_offset] = write_val` writes in the first/second loops (line 460, 486). Those store the qq_prod scratch value (or 0), not a real candidate key. count_and_store ignores them because their matching `p_out` entry is never set; bucketing them would inject garbage into the filter input.
2. The zero-fill column writes (the `for (mm = 1; mm < num_roots; mm++) roots_out[qq_prod_offset + mm * num_p] = 0;` blocks). These are part of the v6 memset-cleanup correctness fix, not candidate emission. emit_fused_key's `if (stored_key == 0) return;` would catch them anyway, but lifting them out of the call entirely keeps the bucketing surface tight.

After each real write:

```cuda
/* num_aprog_vals == 1 path (single root column) */
roots_out[j] = newroot;
p_out[j] = (q << shift) | p;
emit_fused_key((uint64)newroot, hash_mode,
		bucket_count, bucket_overflow,
		bucket_storage, max_per_bucket);
```

```cuda
/* num_aprog_vals > 1 path (multi-r emission) */
for (n = 0; n < num_aprog_vals; n++) {
	p_out[r] = (q << shift) | p;
	roots_out[r] = newroot;
	emit_fused_key((uint64)newroot, hash_mode,
			bucket_count, bucket_overflow,
			bucket_storage, max_per_bucket);
	r += aprog_stride;
	newroot += pp;
}
```

The cast `(uint64)newroot` preserves the post-centered bit pattern stored to `roots_out` (which is `int64`). Storing the not-yet-centered value or a different cast would put a different key in bucket_storage than what count_and_store later reads from gpu_root_array — filter would miss collisions and parity would break in confusing ways.

### Shared bucket-hash header

`compute_bucket`, `NUM_BUCKETS`, `LOG2_NUM_BUCKETS`, `BUCKET_MASK`, `BUCKET_HASH_MIX` are currently defined in `cub/collision_engine.cu`. The fused trans kernel lives in `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu` — a different translation unit. Three ways to keep them in sync:

1. **Shared header (preferred):** extract bucket-hash constants and `compute_bucket()` into `cub/collision_bucket.h` (or `gnfs/poly/stage1/stage1_core_gpu/collision_bucket.h`), include from both. Single source of truth.
2. **Duplicate + assert:** copy the definitions into stage1_core.cu with a strong comment. Add a runtime assert in `collision_engine_init` that NUM_BUCKETS / etc. match a value exported from the trans-kernel side.
3. **Engine exposes a small constant struct** that fused trans reads at launch. Most flexible, most boilerplate.

Recommendation: #1. Header is a few-dozen-line file; the trans kernel is already a heavy include surface.

### Makefile changes the shared header forces

The current PTX rule at `Makefile:338-342` is:
```
stage1_core_sm90.ptx: $(NFS_GPU_HDR)
	$(NVCC) -arch sm_90 -ptx -o $@ $<
```
No `-I.` flag. The engine rule (`cub/built` at line 344) already has `-I. -Ignfs -Ignfs/poly/stage1`, so engine-side includes work, but the PTX rule can't currently find a header outside `gnfs/poly/stage1/stage1_core_gpu/`. Three Makefile edits land together with the shared header:

1. **Add the header to `NFS_GPU_HDR`** (line 218) so the PTX rules rebuild when bucket constants change. Otherwise header-only edits produce stale builds.
2. **Add the header to `cub/built` prereqs** (line 344) so `collision_engine.so` also rebuilds.
3. **Add `-I.` (and/or matching `-Icub` / `-Ignfs/...`) to the PTX nvcc invocation** at lines 339, 342 so the shared header is reachable. Match the engine rule's include flags for consistency.

Without all three, the build will either fail or silently use stale objects.

## Plumbing

### CLI flag

Mirror existing `collengine=gerbicz` pattern. Add `collfuse=1` to the `-nps` argument parser. Defaults off. When on:

- `handle_special_q_batch` in `stage1_sieve_gpu.c` uses the new fused trans kernel.
- The engine's `scatter_roots_kernel` launch is skipped (controlled inside `collision_engine_run` via a new `collision_data_t` field `fused_input`).
- **Explicit support gate:** the spike implements only `pp64_r64`. If `p_array->pp_is_64` is false or `root_bytes != sizeof(uint64)`, the host either errors clearly (preferred) or silently falls back to the unfused trans + scatter_roots path with a log warning. Document the gate behavior in the help text.

### Launch table touchup

The current `stage1_sieve_gpu.c` launch table has 5 GPU functions (3 trans variants + 2 final). The arg-list grouping `gpu_kernel_args + (i / 3)` selects `gpu_kernel_args[0]` for the first 3 (trans) and `gpu_kernel_args[1]` for the last 2 (final). Adding a 4th trans-like kernel (`sieve_kernel_trans_fused_pp64_r64`) needs:

- New `GPU_TRANS_FUSED_PP64_R64` entry in the enum, inserted BEFORE `GPU_FINAL_32` so the trans group stays contiguous.
- New entry in `gpu_kernel_names[]`.
- New row in `gpu_kernel_args[]` for the extended signature (12 original + 5 fusion params = 17 args).
- Update the `(i / 3)` grouping — either generalize to a per-kernel arg-index table, or split `gpu_kernel_args` into per-kernel rows (so each index has its own arg list).

Don't add the kernel by patching the dispatch with magic numbers — the (i/3) shortcut was already fragile.

### Common-code blocker: GPU_MAX_KERNEL_ARGS=15

`include/cuda_xface.h:86` defines `#define GPU_MAX_KERNEL_ARGS 15`, and `gpu_launch_t::arg_offsets[GPU_MAX_KERNEL_ARGS]` + `gpu_arg_type_list_t::arg_type[GPU_MAX_KERNEL_ARGS]` use that fixed size. The fused kernel signature is 12 original trans args + 5 fusion args = **17 args**. Without raising the cap, the launch wrapper silently overflows at runtime (no compile error — `arg_offsets[15]` would write into adjacent struct fields).

Bump to `GPU_MAX_KERNEL_ARGS 20` (cleanest spike move — small, leaves headroom, no need to repack the fused signature). This requires rebuilding everything that includes `cuda_xface.h` (the whole tree, basically — `make clean && make CUDA=1 all`). Revisit packing later only if a future kernel needs more than 20 args.

### Engine API

Add to `collision_data_t`:
```c
int fused_input;  /* if 1, bucket_storage + bucket_count already populated;
                     skip scatter_roots_kernel */
```

Add two new exported entry points (DSO boundary stays in `CUdeviceptr` plus scalar metadata — `stage1_sieve_gpu.c` is driver API, `collision_engine.cu` is runtime API, and the existing interop relies on this convention):

```c
typedef struct {
	CUdeviceptr bucket_storage;
	CUdeviceptr bucket_count;
	CUdeviceptr bucket_overflow;
	unsigned int max_per_bucket;
	unsigned int num_buckets;
} collision_fused_ptrs_t;

void collision_engine_prepare_fused(void *engine, collision_data_t *data,
		collision_fused_ptrs_t *out);

typedef void (*collision_engine_prepare_fused_func)(void *engine,
		collision_data_t *data, collision_fused_ptrs_t *out);
```

### Host loader plumbing for the new symbol

The current loader in `stage1_sieve_gpu.c` only resolves `init/free/run`:

- `device_data_t` (line 498-500) needs a new field: `collision_engine_prepare_fused_func collision_engine_prepare_fused;`
- `load_collision_engine()` (line 1392) needs an additional `get_lib_symbol(..., "collision_engine_prepare_fused")` call. **Resolve with a non-fatal NULL check** — if the loaded engine .so is older and doesn't export the symbol, `d->collision_engine_prepare_fused = NULL` and the host should fall back to the unfused path with a warning when `collfuse=1` is requested. This keeps the engine .so / host binary loosely versioned.

`prepare_fused` must:

1. Call `ensure_capacity(n, key_bits, /*min_per_bucket*/ 0)`. Without this, `bucket_storage` / `bucket_count` may be undersized for the current batch.
2. Zero `d_bucket_count` and `d_bucket_overflow` (the existing scatter path's first two `cudaMemsetAsync` calls).
3. Fill `out` with the device pointers and current `max_per_bucket` / `num_buckets`.

`collision_engine_run` then branches on `data->fused_input`:
- `fused_input == 0`: current behavior (memsets + scatter_roots → filter → ...).
- `fused_input == 1`: skip the memsets + scatter_roots launch, jump straight to the filter launch.

### Multi-SOA handling

Trans is called `p_array->num_arrays` times per batch (3× for degree 5 with num_roots ∈ {1, 5, 25}; different SOA counts for other degrees per `p_soa_array_init` at `stage1_sieve_gpu.c:127`). Each call writes a different region of gpu_root_array (offset by `j`). Currently scatter_roots runs ONCE on the concatenated array.

Fused: each SOA call accumulates into the same bucket_count[] and bucket_storage[]. The SOA launches are **sequential on the same stream** — there's no concurrent inter-SOA atomic contention, just accumulation across launches.

```
[host] engine->prepare_fused(data, &ptrs)   // ensure_capacity + zero + return ptrs
[host] for (i = 0; i < p_array->num_arrays; i++) {
[host]     soa = p_array->soa[i];
[host]     launch fused_trans(..., ptrs.bucket_count,
[host]                              ptrs.bucket_storage,
[host]                              ptrs.bucket_overflow,
[host]                              ptrs.max_per_bucket);
[host] }
[host] data->fused_input = 1;
[host] engine->run(data)                     // filter → dedup → ...
```

### Bucket overflow / retry

Currently scatter_roots sets `bucket_overflow` on slot >= max_per_bucket; host detects this after sync, calls `engine->grow_buckets()` to enlarge max_per_bucket, and re-runs scatter_roots. The trans kernel is NOT re-run (it's expensive and the trans output is still good).

With fusion, if overflow fires inside fused_trans, **don't re-run trans**. The fused kernel always writes gpu_root_array regardless of bucket-write overflow, so the trans output is intact. The fallback flow:

1. Sync; check `bucket_overflow`. If clear, continue normally.
2. If set, read `d_bucket_count` back to host and find `max(bucket_count[0..NUM_BUCKETS-1])`. That's the per-bucket count the just-finished fused trans would have wanted, which is greater than the current `max_per_bucket` (since overflow fired).
3. Call `ensure_capacity(n, key_bits, /*min_per_bucket*/ observed_max)`. This grows `max_per_bucket` and re-allocates `bucket_storage`.
4. Zero `d_bucket_count` and `d_bucket_overflow` again.
5. Launch the ORIGINAL `scatter_roots_kernel` (which reads `gpu_root_array`) — now with the grown `max_per_bucket`. This is the trusted scatter path.
6. Continue with filter as normal.

This means the engine keeps `scatter_roots_kernel` compiled and reachable as a fallback path. The trans-output writes always happen, so the fallback is always safe. The "naive reset-and-retry without growing" version would loop on the same overflow forever.

Surface a `fused_overflow_count` in collstats to monitor how often the fallback fires.

**Alternative (Choice Y, not pursued):** re-run fused_trans with grown max_per_bucket. More expensive (re-computes trans) and complicates the kernel launch logic. The trans-output-already-good property makes Choice X strictly better.

## Validation strategy

Identical to all prior engine experiments, with one timing caveat:

1. **Build:** `make CUDA=1 all`. Verify PTX/binary mtimes updated.
2. **Parity on coeff 420 (gerbicz path):** byte-exact polynomial output vs `msieve.gerbicz.dat.ms` (md5 `ef7d7676615235c41caa228698940511`, 12,739 polys). **Hard gate.**
3. **Parity on coeff 420 (legacy sort path):** unchanged — fused trans only fires on `collfuse=1` + `collengine=gerbicz`. Sort path uses the original trans kernel; should be byte-identical.
4. **collstats:** exact match on `batches`, `candidates`, `dedup`, `matched`, `value_match`, `bucket_max`, `arena_fallbacks`, `arena_capacity_skips`. Fused trans should produce identical bucket contents to scatter_roots (same keys, same hash, same buckets) — so this is a stronger correctness signal than variant-B-style harmless drift.
5. **filter_iters_hist:** should be IDENTICAL to baseline. Different histograms indicate a subtle bucket-assignment divergence.
6. **nsys per-kernel A/B on coeff 420 (no collstats):**
   - First gate: `sieve_kernel_trans_fused_pp64_r64 - sieve_kernel_trans_pp64_r64_baseline < scatter_roots_kernel_baseline`. If false, fusion loses on the GPU.
   - `scatter_roots_kernel` time → 0 (or near-zero if overflow fallback fires occasionally).
   - `filter_per_bucket_kernel`, count_and_store, emit should be unchanged within run-to-run noise.
   - **Total CUDA kernel time delta is the headline number.**
7. **Wall A/B on coeff 420 + coeff 60060:** 3-pair tight back-to-back per the established protocol (msieve.baseline binary + msieve.fused binary).
8. **Do NOT trust `collision_engine_ms` after fusion.** Some former engine work (bucket compute, atomic, bucket_storage write) now runs inside the trans launch *before* `collision_engine_run` is called, so the engine-elapsed timer no longer captures the same scope. Authoritative measures: nsys total CUDA kernel time + whole-batch wall.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Register pressure from added scatter compute pushes trans into local-mem spills | medium-high | ncu check `Local Memory Spilling Requests` after first build. If non-zero, consider keeping fused_trans separate from the size_x sweep (force size_x=224 to keep block_limit constant). |
| `__match_any_sync` adds enough instruction count that trans's compute-bound path slows materially | medium | ncu compare trans's Stall Wait + Stall Math Pipe Throttle before/after. If trans Stall Wait climbs 30%+, fusion hurts more than scatter elimination helps. |
| Warp-sync mask misuse inside divergent trans loops | medium (now mitigated in design) | The `emit_fused_key` helper uses `__activemask()` for both match and shuffle. Code review must verify no `0xFFFFFFFFu` sneaks into the emit path. |
| Bucketed key differs from stored key (e.g., pre-centered value bucketed) | medium (now mitigated in design) | Helper takes `stored_key` and is called *after* `roots_out[...] = ...`. Code review must verify the call sites pass the post-cast value. |
| Bucket-hash constants drift between fused trans and engine | medium (now mitigated in design) | Shared header (preferred). If duplicated, add init-time assert that the constants match. |
| Overflow path becomes hot (frequently fires fallback) | low | Current scatter_roots rarely overflows (`bucket_grow_count = 0` on both coeffs). Overflow fallback should remain rare. Surface a `fused_overflow_count` in collstats for monitoring. |
| Overflow fallback loops because retry uses the same capacity | medium (now mitigated in design) | Choice X flow grows via `ensure_capacity(min_per_bucket = observed_max)` before re-running scatter_roots. |
| Inter-SOA atomic contention on bucket_count[] becomes worse | low | SOAs run sequentially on the stream — no actual concurrent atomic contention between SOAs. Within a single SOA, warp aggregation already handles intra-warp contention well. |
| Bucket layout assumptions change between fused and scatter_roots (e.g., key ordering within a bucket) | low | filter reads bucket_storage as an unordered set (hash-based filter); ordering doesn't matter. Verified by reading filter_per_bucket_kernel. |
| `num_aprog_vals > 1` path emits multiple keys per source thread → atomics serialized per thread | low | Same as current scatter_roots; not new behavior. |
| Launch-table refactor (`gpu_kernel_args[i/3]`) breaks an adjacent kernel by accident | medium | Update `GPU_*` enum + `gpu_kernel_names[]` + `gpu_kernel_args[]` together; smoke-test the legacy sort path before any perf measurement. |
| count_and_store sees identical gpu_root_array but slight per-key timing changes break some downstream cache pattern | very low | If we see any count_and_store regression, can diagnose via nsys; the data it reads is byte-identical so any timing shift is microarchitectural noise. |

## Effort estimate

| Task | Effort |
|---|---|
| Shared bucket-hash header (extract from collision_engine.cu) + Makefile dep wiring + -I. on PTX rule | 0.5 day |
| Bump `GPU_MAX_KERNEL_ARGS` 15→20 + full `make clean && make CUDA=1 all` rebuild | 0.25 day |
| Engine API additions (`collision_fused_ptrs_t`, `prepare_fused` + impl, `fused_input` branch in `run`, overflow grow path) | 1.5 days |
| Host loader plumbing (`device_data_t` field, `get_lib_symbol` resolve with NULL fallback) | 0.25 day |
| New kernel `sieve_kernel_trans_fused_pp64_r64` + `emit_fused_key` helper in stage1_core.cu | 1 day |
| Host wiring in `handle_special_q_batch` + `collfuse=1` flag parsing + launch-table touchup | 1 day |
| Build system + PTX rebuild verification | 0.5 day |
| Parity + collstats + filter_iters_hist + nsys validation cycle | 1 day |
| Wall A/B + 60060 confirmation + ncu spill check | 1 day |
| pp32 variants (after pp64 is proven) | 1 day |
| Write up findings + plan v8 update | 0.5 day |
| **Total: ~8.5 days** for Option A from start to ship-or-revert decision. |

v0 was 7 days; v1 bumped to 8 (shared header + overflow grow path); v2 adds 0.5 day for the GPU_MAX_KERNEL_ARGS bump and loader plumbing.

## Decision criteria

**Ship Option A if:**
- Polynomial parity byte-exact
- collstats exact match (all 8 counters)
- filter_iters_hist exact match
- nsys total CUDA kernel time -5% or better on coeff 420
- 60060 nsys confirms similar magnitude
- No local-mem spills in fused trans (ncu check)
- Overflow fallback fires <0.1% of batches (matches current `bucket_grow_count = 0` regime)

**Revert Option A if:**
- Any parity failure
- `fused_trans - trans_baseline >= scatter_roots_baseline` (first GPU gate from validation step 6)
- Spills appear
- Inter-batch behavior changes (e.g., overflow fallback becomes hot)

**Consider Option B if:**
- A wins ≥ 8% and stays clean
- AND nsys shows that gpu_root_array R+W + gpu_p_array R are still substantial (i.e., the count_and_store path is now a larger share of GPU time post-fusion)
- AND there's appetite for the count_and_store rewrite (2-4 weeks)

## Files to touch (Option A)

- `cub/collision_bucket.h` (new) — extracted `compute_bucket`, `NUM_BUCKETS`, `LOG2_NUM_BUCKETS`, `BUCKET_MASK`, `BUCKET_HASH_MIX` shared by engine and fused trans
- `include/cuda_xface.h` — bump `GPU_MAX_KERNEL_ARGS` 15 → 20 (fused signature is 17 args; bump leaves spike headroom without packing)
- `Makefile` — add `cub/collision_bucket.h` to `NFS_GPU_HDR` (so PTX rebuilds on header edits); add to `cub/built` prereqs (so engine .so rebuilds); add `-I.` (and matching `-Ignfs`/`-Icub`) to `stage1_core_sm{89,90}.ptx` nvcc invocation so the shared header is reachable
- `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu` — new `sieve_kernel_trans_fused_pp64_r64`, `emit_fused_key` helper, include shared bucket header
- `cub/collision_engine.h` — `int fused_input` field in `collision_data_t`, `collision_fused_ptrs_t` struct, `collision_engine_prepare_fused` prototype + matching `collision_engine_prepare_fused_func` typedef
- `cub/collision_engine.cu` — new `prepare_fused` impl + `fused_input` branch in `run`; overflow grow-and-retry-from-gpu_root_array path; include the shared bucket header (so engine and fused kernel use one source of truth)
- `gnfs/poly/stage1/stage1_sieve_gpu.c` — new launch path, `collfuse=1` parsing, GPU_* enum + launch-table additions, `pp_is_64 && root_bytes == sizeof(uint64)` gate, bucket-ptr handoff, `device_data_t` field for `collision_engine_prepare_fused`, `get_lib_symbol(..., "collision_engine_prepare_fused")` in `load_collision_engine()` with **non-fatal NULL fallback** (older engine .so still loads, collfuse silently disabled with warning)

## References

- `cub/collision_engine.cu:115-157` — current `scatter_roots_kernel` (template for `emit_fused_key`)
- `gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu:418-680` — current `sieve_kernel_trans_pp64_r64` (final write sites at ~545-548 and ~555-558 inside the walk-back loop are the emit insertion points)
- `cub/collision_engine.cu:850-940` — current scatter+filter run flow (where the `fused_input` branch lands)
- `cub/collision_engine.cu:74-80` (or wherever `compute_bucket` lives) — bucket-hash constants for the shared header
- `gnfs/poly/stage1/stage1_sieve_gpu.c:127-164` — `p_soa_array_init` showing per-degree SOA structure (degree-4 has 3 SOAs at {2,4,8}, degree-5 at {1,5,25}, degree-6 at {2,4,6,12,36}, degree-7 at {1,7,49})
- `POLYSELECT_OPTIMIZATION_NOTES.md` — historical L2-bound analysis of scatter_roots
- `TWO_LEVEL_SCATTER_PLAN.md` v3-v7 — record of why in-kernel scatter tuning was retired (A1, A.5)
- `TRANS_KERNEL_PLAN.md` v2.2 — record of why occupancy + tail-effect levers were retired
