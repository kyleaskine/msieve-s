# Refinement review fixes — October 26, 2026

This follow-up addresses the correctness and resume findings in the supplied
review. The original dirty working tree was preserved; a baseline snapshot of
the affected sources and executables is under
`stage23_gpu/runs/20261026_review/before/`. No root-sieve kernel or search policy
was changed.

| Finding | Resolution |
| --- | --- |
| Storage README claimed no deletion | Corrected to 101 deleted files, 2247 pruned objects, and required preprocessing reconstruction before full resume. |
| Invalid CUDA MurphyE integral silently falls back under audit | NaN, infinity and negative results set the audit failure and shared cancellation flag. |
| Audit failure finishes the remaining pool | Workers skip unstarted cells; the active evaluator cancels search immediately at the next objective boundary. Audit failure never starts CPU fallback. |
| Alpha discrepancy is invisible to integral audit | All MurphyE paths now rank with canonical CADO algebraic/rational alpha. Checked GPU work also compares the supplied refinement alpha against CADO. |
| Persistent failure/error state crosses batch jobs | Each job resets availability, audit state, counters and error maxima. Failed allocations are released and retried; healthy buffers and grids remain cached. |
| Unchecked final translation can follow another device minimum | Every final size/translation is canonicalized on the CPU from its saved start. This is deliberately stronger than an empirical cutoff guard. |
| Final-refinement messages incorrectly say screen | Resource messages now identify alpha/size work; numerical disagreement identifies final refinement or alpha screen explicitly. |
| Copies the entire candidate pool before shrinking | Construct only the nref prefix. |
| Empty CPU refinement loop and duplicated Murphy fallback | Skip the parallel region after successful device refinement; use one recoverable Murphy fallback path. |
| Converts coefficients for each of 221 initial skews | Convert once and reuse that immutable polynomial across the initial grids. |
| Audit oracle can drift from the CPU integral | CPU scoring and audit share the scalar loop; CUDA and CPU share homogeneous evaluation and angle generation. Compare separately against CADO's original integral. |
| Audit flags invalidate numerical checkpoints | `refine_check` and `murphy_check`, including their boolean argv switches, are execution settings. Completed checkpoints are reused; checks apply to newly executed work. |

Three performance/maintenance proposals remain separate follow-ups: batching
pattern-search neighbors while preserving cache/budget/tie ordering, reusing
root tables across launches, and extracting the scalar/vector top-K merge into
shared device helpers. The root workspace already promised a bounded *launch*
cache; its README now explicitly says that tables are rebuilt next launch.
These deferrals do not leave a known root-sieve mismatch unresolved.

Validation completed:

- Full host `make test`, including 41 pipeline tests and nine Murphy
  invalid-cell/fallback/alpha/cancellation scenarios.
- 2100 bit-for-bit scalar integral comparisons against CADO, including
  translated coefficients, multiple degrees, cache eviction and multiple workers.
- A new CUDA audit test rejects NaN/infinity/negative integrals, propagates
  cancellation between evaluators, rejects alpha disagreement, resets per-job
  statistics while retaining healthy allocations, and recovers on the same
  device after an injected out-of-memory status.
- All six CUDA refinement CLI tests pass without skips, including exact
  checked/unchecked/CPU fixture exports, resource fallback and batch recovery.
- Replayed 26 saved pools, 167936 candidates. All 5200 exports match CPU bytes,
  including four candidates with different device translation choices that are
  canonicalized before export. Maximum raw alpha/key differences are
  7.11e-15/1.42e-14. All exported resultants verify.
- Independent 256k-angle evaluation of the bounded C192/C208 Murphy tests
  matches CPU best scores at the scorer's printed precision; checked and
  unchecked CUDA exports match.

The GPU was available outside the sandbox but shared with another workload.
Recorded durations are validation timings, not a controlled speed benchmark.
CPU size canonicalization and canonical alpha add host work; historical
refinement speedups should not be applied to this implementation without a new
matched benchmark. This preserves correctness while GPU numerical migration
continues.

Reproduce the key checks from the repository root:

```
make -C stage23_gpu -j2 s23_ropt s23_ropt_cpu build/test_murphy_cuda
make -C stage23_gpu test
stage23_gpu/build/test_murphy_cuda
S23_ROPT_BIN="$PWD/stage23_gpu/build/s23_ropt" python3 stage23_gpu/tests/test_refinement_gpu.py
python3 stage23_bench/runs/20261023_gpu_refine/run.py --run-dir NEW_DIRECTORY --repeats 1
```

Full logs, frozen corpus inputs/outputs and binary hashes:
`stage23_gpu/runs/20261026_review/`. Compact provenance and results:
`stage23_bench/data/gpu_refinement/review_20261026.json`.
