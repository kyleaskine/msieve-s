# C208 root-search coverage regressions

Exact seeds recovered on 2026-10-13 from the archived GPU/MSIEVE comparison.
`manifest.json` records source paths and SHA256 checksums. Seed numbers are the
archive's zero-based `sNNNN` identifiers. All seed and reference resultants are
nonzero multiples of the common n.

`references.poly` preserves MSIEVE's original-pass, j=0 outputs for the six
historical coverage losses. Each manifest `reference_rotation` means
`f_ref(x) = [f_seed + (u*x + v)*g_seed](x+t)`, with the rational polynomial
translated by the same t. These exact relations were reconstructed using
`polyfmt.rotation_between`. The archive's MurphyE labels are historical scores;
they are not new validation results.

| Seed | Historical loss |
|---|---|
| 15, 27, 32, 103, 151 | Cell cap excluded MSIEVE's rotation |
| 222 | Consecutive-line scan stopped before MSIEVE's u=-16152 |
| 10 | Proxy cutoff at (7,-174864538), subsequently fixed by exact-alpha reranking |

Reproduce the local and coarse model plans from the repository root:

```bash
make -C stage23_gpu s23_ropt_cpu
stage23_gpu/build/s23_ropt_cpu \
  -poly stage23_bench/data/c208/ropt_coverage/s0222.poly \
  -search -plan -t 2 -budget 3 -maxcells 2e11 -maxlines 1000 -umax 20000
# Repeat with the same arguments plus:
# -u-scan 256 -u-basins 4 -u-radius 128
```

The first R1 comparison found lower minimum lognorms on seed 222
(67.7577 -> 65.3283, refined center u=-18647) and seed 32
(64.0027 -> 62.9378). Seed 222 retains 50 consecutive lines near the new center
at this cap. It does **not** recover the specific MSIEVE rotation at u=-16152;
that remains outside the added neighborhoods. Discovering a deeper size basin
does not establish higher MurphyE or better sieve yield.

`stage23_gpu/tests/test_ropt_scan.py` also shifts a real C161 polynomial by
4096*x*g: the ordinary scan misses its original minimum, and the coarse scan
recovers it at u=-4096. The test checks the exact resultant, capped coverage,
CLI errors and exported polynomials. `make -C stage23_gpu test` includes it.

The development run (frozen baseline binaries, all commands, input/binary hashes,
plan logs, timings and bounded CPU/GPU comparisons) is saved locally under
`stage23_gpu/runs/20261013_ropt_basins/`, outside the tracked fixture corpus.
See [ROPT_OPTIMIZATION_PLAN.md](../../../../ROPT_OPTIMIZATION_PLAN.md) for the
remaining residue-class and equal-time quality experiments.

The subsequent R2a explicit-class path admits all six reference rotations when
given their own classes modulo 2520. For seed 222, a real GPU search with
`-residue 2520 -16152 6992188045104 -budget 3 -maxcells 2e11 -umax 20000`
retains the historical rotation as the first of 200 exports after exact-alpha
reranking (`-aw 1.3 -rerank 4096 -refine 200`). The class can also be written
`-residue 2520 1488 1704`. It sieved 1.26e11 selected cells in 3.97 seconds on the
RTX 5070, with `-t 2 -launch 30`; the full process took 5.84 seconds.

These are **supplied-class** tests, using the known winners as input to class
choice. They do not validate automatic class selection or final MurphyE/yield.
`test_ropt_class.py` checks all six plan coverages, selected-cell caps and exact
content exports. The R2a artifacts are in
`stage23_gpu/runs/20261013_ropt_classes/`.
