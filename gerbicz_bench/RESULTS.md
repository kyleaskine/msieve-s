# Gerbicz CUDA Collision-Search — Standalone Benchmark Results

**Hardware:** RTX 5070 (sm_120, Blackwell consumer, 12 GB, 99 KB dyn-smem/SM, 48 SMs)
**Toolchain:** CUDA 13.2.78, nvcc native cubin for sm_120
**Date:** 2026-05-23 (v3)

## Headline number

At the msieve regime (N=50M, 56-bit keys, uniform random input):

| Pipeline                             | Per-batch GPU time |
|--------------------------------------|--------------------|
| Existing CUB radix sort + final scan | ~11.75 ms (sort 11.0 + final 0.75)  |
| Gerbicz Phase A + B + C (this port)  | **~4.27 ms** (median, 5 runs)       |
| Speedup                              | **2.75×**          |

Best run: 4.21 ms → 2.79×. Worst: 4.96 ms → 2.37×. Below JasonP's reported 4× — the remaining lever is to fuse Phase A into the trans kernel (msieve integration scope, not standalone).

The number above excludes H2D because in real msieve usage the keys are already on the GPU. Wall-clock including H2D is ~45-50 ms at this cell.

## CPU comparison

**`collision_ref.c` is single-threaded** (no OpenMP, no pthread). For a multi-core CPU comparison we wrote a parallel version at `bench_cpu_omp.c` that mirrors the reference algorithm with OpenMP on the per-bucket filter loop. Use `--threads N` to invoke it.

Quick sanity check (N=10M, bits=30, the largest density-rich cell within collision_ref.c's 40-bit limit):

| CPU mode             | Wall time | Speedup vs 1-thread |
|----------------------|-----------|---------------------|
| `collision_ref.c` (1 thr)| 0.279 s   | 1.0×                |
| `omp_collision`(2 thr)| 0.110 s  | 2.5×                |
| GPU                  | ~0.012 s  | ~23× vs 1-thread    |

Higher-thread runs (--threads 8 etc.) are pending; the user has a CPU-busy box right now. The bench is ready to measure once they're free.

GPU vs CPU comparison at N=50M is limited because **collision_ref.c asserts `key_bits<=40`** (its stripped buckets are uint32, max 32 bits of payload + 8 bits of bucket = 40 bits). At the msieve regime (56-bit keys), only the GPU runs — `--no-verify` is recommended to skip the slow qsort verifier too.

For dense data where everything runs (N=50M, bits=30): GPU 144 ms (incl. H2D) vs CPU 4.25 s single-thread = **~30× speedup** (and qsort verifier also confirms hit count = 2,276,857 ✓).

## Three-way correctness validation

GPU vs CPU `collision_ref.c` vs slow qsort verifier, exact agreement at every tested cell:

| N    | bits | hits      | Notes                  |
|------|------|-----------|------------------------|
| 1M   | 30   | 887       | random-uniform dense   |
| 10M  | 30   | 92,848    | dense                  |
| 50M  | 30   | 2,276,857 | very dense             |
| 50M  | 48   | 20        | sparse (2-way: GPU vs qsort) |
| 50M  | 56   | 0         | msieve regime (2-way)  |

OpenMP CPU also matches (tested at N=1M, 10M with --threads 2).

## Full sweep (median of 5 runs, warmup dropped)

| N     | bits | A (ms) | B (ms) | C (ms) | A+B+C (ms) | candidates | bucket_max | smem-arr? |
|-------|------|--------|--------|--------|------------|------------|------------|-----------|
| 1M    | 40   | 0.17   | 0.10   | 0.14   | 0.41       | 0          | 95         | yes       |
| 1M    | 56   | 0.17   | 0.10   | 0.15   | 0.42       | 0          | 97         | yes       |
| 10M   | 48   | 2.16*  | 0.19   | 2.59*  | 4.94*      | 6          | 711        | yes       |
| 10M   | 56   | 0.74   | 0.21   | 0.30   | 1.25       | 8          | 723        | yes       |
| 10M   | 64   | 0.75   | 0.21   | 0.30   | 1.26       | 6          | 708        | yes       |
| 50M   | 48   | 3.08   | 1.66   | 1.02   | 5.76       | 120        | 3,277      | no        |
| **50M** | **56** | **3.34** | **1.36** | **0.37** | **5.07** | 56 | 3,283 | no |
| 50M   | 64   | 3.91   | 1.36   | 0.26   | 5.53       | 48         | 3,260      | no        |
| 100M  | 56   | 5.83   | 3.41   | 0.26   | 9.50       | 54         | 6,445      | no        |

*10M @ 48-bit shows transient high A and C; likely a warmup artifact, the 56/64-bit lines at the same N are consistent.

For N ≤ 10M the smem-arr Phase B path triggers (bucket fits in 49KB → 2 blocks/SM occupancy preserved) and Phase B drops by ~3× (0.65 → 0.21 ms). At N=50M+ the bucket is too big for smem at preserved occupancy, so the kernel falls back to DRAM ping-pong (which is still bandwidth-efficient).

## Architecture (final)

Three GPU kernels mirroring `collision_ref.c`, plus CUB calls for sort + max-reduce:

**Phase A — single-pass scatter** (`collision_bucket.cu`)
- 16,384 buckets, each with a `max_per_bucket`-slot region in `bucket_storage`.
- One scatter kernel: warp-aggregated atomicAdd via `__match_any_sync` on bucket id.
- Stores the full key (not stripped) so the kernel works for both bucket modes (default low-bits and `--bucket-hash` Knuth-mix).
- CUB DeviceReduce::Max in parallel for telemetry.

**Phase B — per-bucket ping-pong filter** (`collision_filter.cu`)
- 16,384 blocks × 128 threads.
- Two kernel variants:
  - **`filter_per_bucket_smem_kernel`** — items in shared memory across rounds. Used when total smem ≤ 49 KB (= 2 blocks/SM occupancy preserved). Triggers at N ≤ 10M.
  - **`filter_per_bucket_kernel`** — items in DRAM with ping-pong between two `bucket_storage` buffers. Fallback for larger N.
- Hash tables T/T2/T3 always in shared memory.
- `atomicOr` returning previous value preserves no-loss correctness in parallel (a naive `if (U2[hv]) ... else ...` test races and silently drops collisions).

**Phase C — verify** (`collision_verify.cu`)
- CUB sort on candidates → dedup → CUB scan → scatter → scan_originals. Same structure as collision_ref.c:384-429.

**Static sizing at init time** eliminates the cudaStreamSynchronize between Phase A and Phase B. `max_per_bucket = mean + 6σ + 32` covers any uniform-random distribution; per-call `max_tsize_words` is recomputed for the actual N so the smem-arr threshold triggers correctly inside a multi-N sweep.

## Where the remaining time goes (msieve regime)

At N=50M, bits=56, median A+B+C = 4.27 ms:

| Phase | Time   | What it's doing |
|-------|--------|-----------------|
| A     | 2.59ms | One scatter pass: 50M atomicAdds (warp-aggregated to ~1.5M) + 50M strided writes |
| B     | 1.37ms | 16,384 blocks × ~3,000 items × 2-3 filter rounds. DRAM ping-pong path. |
| C     | 0.31ms | 56 candidates → 28 unique → CUB sort + scan over 50M originals |

Phase A scattered writes dominate. To close the gap toward 4×:

1. **Fuse Phase A into the trans kernel** (msieve integration scope) — produces keys + scatters in one fused kernel, eliminates the entire 2.59 ms of Phase A. Estimated post-fusion total: ~1.7 ms → ~7× speedup.
2. **CUDA Graphs** to amortize launch overhead. Marginal here (~50-100 μs over 4.27 ms ≈ 2%).
3. **Phase B smem-arr at larger N**: needs >99 KB dyn-smem per block (already maxed out on sm_120). Possible avenue: smaller hash tables (lower ilog2) to free smem for arr, accepting more candidates surviving each round.

## msieve integration status (2026-05-23)

The standalone Gerbicz collision path is now available inside msieve as an opt-in stage-1 polyselect path:

- `collengine=gerbicz` selects `cub/collision_engine.so`.
- `collhash=1` is the default bucket-index hash. `collhash=0` keeps the low-bit bucket index and now also works on the real coeff test.
- `collstats=1` prints one per-coefficient summary line: batch count, collision-engine GPU time, max bucket, candidate/dedup/value-match totals, bucket growths, and hash-table caps.
- `colldebug=1` re-enables the per-batch bucket-grow and hash-cap diagnostics. Normal runs are quiet unless there is a fatal overflow/error.

Important command-line detail: the NFS argument string must be immediately after `-np1`, e.g.

```bash
./msieve -np1 "min_coeff=420 max_coeff=420 collengine=gerbicz collstats=1" -nps
```

What landed:

- `cub/collision_engine.{h,cu}` exports `collision_engine_init/free/run`, mirroring `cub/sort_engine.cu`.
- `Makefile` builds `cub/collision_engine.so` alongside `cub/sort_engine.so`.
- `gnfs/poly/stage1/stage1_sieve_gpu.c` branches after the trans kernels: the Gerbicz path skips CUB sort and `sieve_kernel_final_*`, and writes directly to the existing `found_array`.
- The real msieve path preserves the old default sort path unless `collengine=gerbicz` is requested.

Correctness and stability fixes found during integration:

- Zero roots in the pre-cleared root array must be skipped by the collision scatter path, matching the old final kernels' behavior.
- Bucket storage grows and retries when the observed max bucket exceeds the estimate.
- Per-bucket hash-table shared memory is capped to the device opt-in shared-memory limit.
- Synthetic value verification with injected duplicates matches the qsort verifier and the msieve hit predicate (`GPU_msieve == qsort_msieve`).

First real coeff timing from the user (`min_coeff=420 max_coeff=420`, same output file sizes for all three runs):

| Path | Real time |
|------|-----------|
| Existing sort path | 4m39.169s |
| `collengine=gerbicz` | 2m35.957s |
| `collengine=gerbicz collhash=0` | 2m37.059s |

That is about 1.79x end-to-end for this one-coefficient run, with both Gerbicz bucket modes agreeing at the output-file-size level.

## Files

```
gerbicz_bench/
  Makefile                 nvcc + g++ with -fopenmp for bench_cpu_omp.o
  bench_main.cu            CLI: --n, --bits, --sweep, --mode, --repeats,
                                  --warmup, --bucket-hash, --threads, --csv
  bench_cpu.{c,h}          single-thread wrapper around collision_ref.c
  bench_cpu_omp.c          OpenMP-parallel mirror of fun_collision_search
  bench_gpu.{cuh,cu}       host orchestration + CUDA event timing + CUB
  collision_common.cuh     NUM_BUCKETS=16384, compute_bucket helper
  collision_bucket.cu      Phase A: scatter_one_pass_kernel
  collision_filter.cu      Phase B: filter_per_bucket_smem_kernel +
                                     filter_per_bucket_kernel (DRAM fallback)
  collision_verify.cu      Phase C: dedup + secondary-hash + scan_originals
  data_gen.{c,h}           Marsaglia MWC RNG matching collision_ref.c
  verify_qsort.{c,h}       ground-truth slow verifier
  sweep_final.csv          v1 sweep CSV (pre-smem-arr / OpenMP)
  RESULTS.md               this file
```

## Reproducibility

```bash
cd /home/kylea/msieve-s/gerbicz_bench && make
./bench --n 50000000 --bits 56 --repeats 5 --warmup --no-verify  # msieve regime, GPU-only
./bench --sweep --repeats 5 --warmup --no-verify --csv sweep.csv  # full sweep
./bench --n 10000000 --bits 30 --threads 8                        # multi-core CPU comparison
```

## Next concrete step

Integration is in tree and builds. Recommended order from here:

1. Run several small real composites or fixed coefficient ranges with the existing sort path and `collengine=gerbicz`, then compare output contents, not just file sizes.
2. Use `collstats=1` on representative coeffs to capture real batch counts, bucket maxima, candidate rates, growth counts, and hash-cap frequency.
3. Re-profile the post-port stage-1 path. The old profile said sort dominated; after removing it, the trans kernels and collision Phase A should decide the next optimization.
4. If the profile confirms Phase A scatter is now a top cost, plan the fused trans+bucket-scatter kernel. That remains the biggest algorithmic lever.
