# stage23_gpu: GPU size optimization (M1) and root optimization (M2)

Standalone tools, built separately from msieve. The plan, decisions and benchmarks are in
`../GPU_STAGE23_PLAN.md`; fixtures and the acceptance tooling are in `../stage23_bench/`.

## How this is tested

All device code lives in `include/*.h` as `__host__ __device__` templates. The same
source compiles with g++ and runs on the CPU, so it is unit-tested against GMP and
against CADO's own code without a GPU (CUDA has no built-in emulator any more). The
device harness then only has to show the GPU computes what the host build computes.

```bash
make test                 # host: Int<L> vs GMP, sopt LLL vs CADO's utils/lll.c
make device CUDA=120      # compile the device harness (does not touch the GPU)
make run-device           # run it: GPU results vs the host build (needs a free GPU)
make accept-cpu           # M1 acceptance: s23_sopt (CPU build) on 13,000 c146 polys vs CADO
make accept-gpu           # the same with the GPU build (needs a free GPU)
make accept-cpu EFFORT=50 SET=top   # effort 50 on the 3000 polys CADO ran at effort 50
```

`s23_sopt` (`tools/s23_sopt.cu`) reads raw polynomials and writes CADO sopt's format,
so the pipeline and `stage23_bench/tools/sopt_compare.py` read it unchanged:

```bash
build/s23_sopt -sopteffort 0 -inputpolys msieve.dat.ms > sopt_out.txt     # GPU
build/s23_sopt_cpu -t 8 -sopteffort 0 -inputpolys msieve.dat.ms > ...      # CPU build
```

Any polynomial the port flags (overflow, loop or recursion caps) or runs out of time on
is redone by CADO's own `size_optimization` on the CPU and reported on stderr;
`-compare` also runs CADO on every polynomial and counts identical results. Options:

- `-t N` (default 8): host threads for the CPU path, the CADO fallbacks, and CADO's
  stats and formatting of each batch (in parallel, written in input order).
- `-dev N`: the GPU to use (default 0).
- `-budget S` (default 40 on the GPU, none on the CPU): a polynomial still running S
  seconds after its launch started gives up at its next translation candidate and is
  redone by CADO, so no launch reaches the 60 s WSL2 watchdog.
- `-sopteffort E`: 0 to 100 (the translation list is sized for 100; CADO's sopt above
  that). Effort > 0 is refused on the GPU unless `-force-gpu-effort`.
- A malformed input block stops the run (exit 1) after the polynomials before it, as
  CADO's sopt does; blank lines between blocks are skipped.

The Makefile gets CADO's include flags and libraries from
`stage23_bench/tools/polyfmt.py` (`CADO_BUILD_DIR`, else `cado_build_dir` in
`../nfs_config.ini`), the same code the bench tools use, and relinks when CADO's
libraries change.

## Layout

```
include/
  sopt.h        CADO's size_optimization (degree 5): translation candidates, best_norm2
                (iterative, frame stack in scratch), local descent, expected_growth,
                expected_rotation_gain, exp_E selection; effort > 0 adds Farey candidates
  norms.h       CADO's L2 lognorm and L2 skewness (polyselect_norms.c)
  dpoly.h       CADO's double polynomial root finder (double_poly.cpp)
  alpha_proj.h  CADO's projective alpha (p < 100): exact discriminant (Bareiss),
                special_val0 recursion, both valuation code paths; expected_alpha
  hd.h          host/device macros (S23_HD inlined; S23_HD_CALL real calls on the device)
  mpint.h       Int<L>: L x 32-bit two's-complement integers; add/sub/neg/mul with
                overflow reporting, floor and exact division (Knuth D), mpz_get_d-style
                truncating conversion
  lll_exact.h   exact-integer LLL, a port of CADO utils/lll.c (NTL LLL_ZZ) for
                independent rows, delta = 1
  sopt_lll.h    CADO's best_norm: translate by k, build the skew-scaled rotation lattice,
                LLL, choose the shortest row with a degree-d term, extract a*f + r*g
tools/
  s23_sopt.cu      the size optimization tool (GPU build with nvcc, CPU build with g++)
  cado_io.c/.h     C shim to CADO: read polys, print sopt format, CADO's skew, CADO fallback
tests/
  test_mpint.cpp   Int<L> vs GMP on random operands (L = 2 .. 128), overflow edges
  test_lll.cpp     sopt_best_norm vs CADO's LLL on the same matrices
  gen_cases.py     LLL cases from the c146 fixture and the c204 winners
  test_device.cu   the same sopt_best_norm on the GPU vs the host build
```

## Status (2026-10-03, M1 steps 3-4)

- `s23_sopt`, CPU build, M1 acceptance on the c146 fixture:
  - effort 0, 13,000 polys: **13,000 identical to CADO** (bit for bit), 0 redone by
    CADO. (Before the rounding fix below: 12,996 identical and 4 off by a constant
    rotation, exp_E within 1e-7, which were rounding differences, not ties.)
  - effort 50, the 3000 polys CADO ran at effort 50: **3,000 identical** after the
    rounding fix (2,993 identical and 7 no worse before it), 0 redone.
  - speed: 13,000 polys in 22 s on 8 threads (about 14 ms of CPU per poly, CADO 11);
    effort 50 about 1 s of CPU per poly (CADO 0.67). The fixed 4096-bit LLL integers
    cost this on the CPU; the GPU build is the one that matters.
- GPU build, first runs (RTX 5070):
  - LLL harness: all 24,343 cases identical to the host build (0.2 s).
  - M1 acceptance, effort 0, 13,000 polys: after the rounding fix, **13,000 identical to
    CADO**, 0 redone by CADO (before it: 12,999 identical, 1 no worse).
  - Speed: 248 s for 13,000 polys (19 ms each), about 11x slower than the CPU build on
    8 threads. One thread per poly: a single thread needs up to ~20 s for one poly, so a
    1,024-poly launch takes ~29 s; the default batch is 1,024 to stay under the 60 s
    WSL2 watchdog. Peak VRAM ~6.5 GB, mostly CUDA's stack reservation (64 KB x every
    thread that can be resident).
  - Effort 50 has not been run on the GPU: ~100x the per-thread work would take far
    longer than the watchdog allows. Splitting the work per translation candidate across
    threads (design constraint 4) fixes this and is the first speed step.
- c161 (a second number, unseen during development): CPU build on all 814,307 polys,
  effort 0: all identical to CADO, 0 fallbacks (3,110 s optimizing on 8 shared threads,
  ~1,570 s more in the single-threaded CADO stats printing); effort 50 on the top 1000:
  1,000/1,000 identical. GPU build on the c161 fixture (11,000): all identical.
- Caps (each falls back to CADO): best_norm2 depth 128 (one c146 poly needs more than
  32 and at most 64), special_val0 depth 12 (the sample needs at most 10), translation
  candidates sized for effort 100 (3,265).
- Review fixes (2026-10-03, CPU build; the GPU was busy with LA): stats and formatting
  now run on all host threads; the c146 13,000 at effort 0 went from 59 s to 45 s with
  byte-identical output. `-budget` tested on the CPU (every poly out of time -> CADO,
  output identical). On the GPU: `make accept-gpu` 13,000/13,000 identical; default
  budget 40 s, 219 s optimizing (17 ms per poly), 0 out of time; `-budget 1`: 7,315 of
  13,000 out of time and redone by CADO, output still identical to the CPU build's;
  `-dev` with a missing GPU exits 2.

## Status (2026-10-03, M1 step 2)

- `Int<L>`: all operations match GMP (about 660,000 random cases, L = 2 .. 128).
- Exact LLL / `best_norm`: identical to CADO on all 24,343 cases (every raw poly of the
  c146 acceptance sample at k = 0 and at CADO's final translation, plus the c204
  winners): the whole reduced basis, the chosen row and the polynomial.
- Widest intermediate: 1,826 bits on c146 cases, 2,433 bits on c204, so `Int<80>` to
  `Int<96>` would do for c204; the tests use `Int<128>`.
- Device harness: compiles for sm_120 (255 registers, ~18 KB stack per thread with the
  multi-limb routines as real calls; it raises `cudaLimitStackSize` to 24 KB). Since run
  on the GPU: all 24,343 cases identical to the host build (`make run-device`). Not run
  yet.

Correctness first: no carry intrinsics, one thread per LLL, one width for everything.
Speed comes after M1 matches (smaller widths per variable, warp-cooperative arithmetic,
or floating-point LLL with exactness checks, measured against this reference).

Design notes:

- Rounding, matched to CADO: CADO's C code (norms, alpha, size optimization) is compiled
  without fused multiply-adds, but its C++ root finder (`utils/double_poly.cpp`) is
  compiled with them (GCC, `-O3 -march=native`: every Horner step of `double_poly_eval`
  and `a*pb` in false position are FMAs). Both our builds disable automatic contraction
  (`-ffp-contract=off`, `--fmad=false`) and `include/dpoly.h` writes exactly those
  operations as `fma()`. The device's `log` is CUDA's (within 1 ulp of glibc), the one
  known remaining source of last-bit differences on the GPU.

- The skew for the LLL is computed on the host exactly as CADO's `sopt_get_skewness`
  (double `pow`, then truncation), so the GPU never depends on its own `pow` agreeing
  with glibc's in the last bit. Translation candidates (also doubles in CADO) will get
  the same treatment.
- NTL's linearly-dependent-vector branch is not ported; the sopt lattice is always
  independent, and the port returns `LLL_DEPENDENT` (CPU fallback) if that ever fails.
