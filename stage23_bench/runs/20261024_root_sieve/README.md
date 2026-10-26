# Root-sieve grouped lookups — October 24–25, 2026

The frozen baseline is the October 24 C197 production binary. Profiling a dense
C197 window on the RTX 5070 found high instruction and shared-memory lookup
utilization, with little DRAM traffic. Changing thread-block layout alone slowed
the sieve. Reusing raw u-line tables gave a small improvement; grouping four
consecutive selected cells into aligned float4 lookups gave about 1.8x throughput
on large dense and strided windows. Short blocks retain the scalar kernel.

The production implementation preserves the original blocks, every per-cell
prime-summation order, and the total ordering of retained candidates. It caches
at most 1,024 raw u-line tables per launch. Insufficient device resources, cache
overflow or optional-allocation failure uses the scalar kernel. Neither search
budgets nor residue-class selection changed.

To reproduce against a saved original binary/source, first freeze them into a
new directory, then build the current implementation:

```
python3 stage23_bench/runs/20261024_root_sieve/kernel_layout.py prepare \
  --run-dir NEW_RUN --baseline-binary ORIGINAL_BINARY --baseline-source ORIGINAL_SOURCE
make -C stage23_gpu s23_ropt s23_ropt_cpu build/test_rsieve_gpu
python3 stage23_bench/runs/20261024_root_sieve/validate.py checks --run-dir NEW_RUN
python3 stage23_bench/runs/20261024_root_sieve/validate.py sanitize --run-dir NEW_RUN
python3 stage23_bench/runs/20261024_root_sieve/validate.py block-sizes --run-dir NEW_RUN
python3 stage23_bench/runs/20261024_root_sieve/validate.py timings --run-dir NEW_RUN --repeats 3
```

The full-depth experiment needs the saved C197 seeds, original depth pools,
exports and command records in `stage23_gpu/runs/20261024_c197_gpu/`. It runs
seeds s0005 (the overall winner) and s0001, using the original depth settings,
one GPU process and two CPU threads. Every repetition must reproduce both the
4,096-candidate saved pool and all 200 exported polynomials byte for byte.
All exported resultants are checked. Process and sieve times are separate;
these are root-search component comparisons, not whole-pipeline speedups.

The direct kernel test covers four polynomials, affine/projective modes,
dense/mixed strides, signed coordinates near both int64 limits, short tails,
varying K, ties, slopes and cache rejection/reuse. It checks 28,672 retained
hits against the CPU implementation, including the exact float bit patterns.
The full host suite and six GPU CLI suites are logged separately. Memcheck and
racecheck run on the C168 content fixture, with 7,168 comparisons each.

`kernel_layout.py build` and `benchmark` reproduce the initial thread-layout
study from the frozen source. The original table/vector prototypes are kept in
the local run's `source/` and `bin/` directories; the final implementation is
in `stage23_gpu/tools/rsieve_cuda.h`. `--variants` benchmarks named frozen
executables, and `--name` chooses a fresh output directory.

Full artifacts are in `stage23_gpu/runs/20261024_root_sieve/`. Validation phases
require fresh output directories and never overwrite the original C197 run.

Completed medians: s0005 sieve 62.27 -> 32.90 s (1.893x), s0001 108.37 ->
57.55 s (1.883x). Process times are 64.43 -> 34.99 s and 110.05 -> 59.29 s.
All twelve pool/export comparisons pass, as do the host suite, 32 CUDA CLI
tests and both sanitizer runs. The compact record is
`stage23_bench/data/root_sieve/20261025.json`.
