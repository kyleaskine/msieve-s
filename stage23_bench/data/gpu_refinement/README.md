# GPU finalist refinement, 2026-10-23

`20261023.json` records the full-alpha corpus audit and bounded MurphyE comparison
for the opt-in `-refine-gpu` and `-murphy-gpu` paths. It includes frozen binary and
input hashes, commands/manifests, timing records, output hashes and independent
256k-angle best scores. CPU and GPU preserve all 5,200 final exports across 26
saved pools. The original source artifacts were verified unchanged.

The full-alpha one-pass timings are diagnostic. The MurphyE medians use three
alternating CPU/GPU repetitions on 32 candidates, with two CPU threads and
persistent processes; they do not measure whole-pipeline speed or sieve yield.

Reproduce with `stage23_bench/runs/20261023_gpu_refine/run.py --run-dir NEW_DIRECTORY`.
The original saved pools are needed for the full corpus. The smaller regression
suite uses tracked fixtures and runs with:

```
S23_ROPT_BIN=$PWD/stage23_gpu/build/s23_ropt python3 -m unittest discover \
  -s stage23_gpu/tests -p 'test_*gpu.py' -v
```

Full run: `stage23_gpu/runs/20261023_gpu_refine/`. Implementation and migration
limits are described in `ROPT_OPTIMIZATION_PLAN.md` and `stage23_gpu/README.md`.
