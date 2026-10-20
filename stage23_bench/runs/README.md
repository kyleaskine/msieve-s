# Run scripts

The scripts that produced the GPU comparisons recorded in `../data/` and in
`../../GPU_STAGE23_PLAN.md`, kept here because the runs themselves happened in
directories that do not last: `stage23_gpu/build/` (deleted by `make clean`) and
`pipeline_results/` (moved to `previous/` and later deleted by the next pipeline run).
They take the run directory as an argument; each file's docstring or header says how.

- `20261010_c185_c168b/`: the C185 and second C168 breadth runs (2026-10-10), made in
  `stage23_gpu/build/runs_20261010/{c185,c168}`: `prepare.py` (seeds from the CPU
  pipeline's `best300_msieve.ms` plus content seeds), `run_breadth.sh`, `score.py`,
  `analyze.py`, and `save_fixtures.py`, which wrote `../data/c185` and `../data/c168b`
  (it now takes the run directory as argument 1).
- `20261011_c184/`: the whole GPU path on the C184 (2026-10-10/11), made in
  `pipeline_results/gpu_c184/`: `run_e0.sh` (GPU sopt effort 0 on every raw poly),
  `run_after_e0.sh` (top 1000, GPU effort 50, top 300, seeds, breadth), `prepare_gpu.py`,
  `run_breadth.sh` (4 workers, as run), `run_breadth_ab.sh` (another binary or worker
  count; 2 workers with `-launch 30` is the desktop-safe setting), `run_pass2.sh`,
  `score.py`, `score_pass2.py`, `analyze.py`, `compare_ab.py`. Paths inside are the run's
  (`~/msieve-s`, `pipeline_results/gpu_c184`).
- `20261012_c196/run.py`: the C196 full GPU size-optimization and breadth run,
  in `stage23_gpu/build/runs_20261012/c196/`. Reads the run's frozen `input.ms`,
  `bin/`, CPU seed reference and `settings.json`; uses the CPU pipeline's top 2000
  for effort 50, then 300 parent seeds plus content seeds. Breadth uses two workers,
  eight host threads each, and `-launch 30`. `status.json`, `run.log` and per-command
  JSON/log files record progress and failures. `compare.py RUN_DIRECTORY` waits
  for both paths, snapshots the completed CPU results by matching n, scores breadth,
  runs in-process MurphyE on the two leading GPU searches, and independently checks
  finalists at 256k points. It reads `scoring.json`: lpbr/lpba 32/33, rectangle
  17,32768, and provisional q band 80M–450M. MurphyE's equivalent I=16.5 matches
  that rectangle's area; limr/lima 200M/300M are retained for future test sieving.
  Progress is in `comparison_status.json`, results in `comparison.json` and
  `finalists.poly`. `verify_sopt.py RUN CPU_E0_OUTPUT` streams the full effort-0
  comparison and rescores differences at full precision. `save_results.py RUN`
  preserves compact results under `data/c196/` after all checks finish.
  The run directory survives CPU pipeline output rotation, but
  `make clean` deletes it; retain compact results in `data/` after comparison.
  `testsieve.py RUN --prepare-only` freezes the CADO/msieve jobs and siever binaries
  for the 2026-10-13 comparison (lpbr/lpba 32/33, mfbr/mfba 62/95); run without that
  flag for three alternating 4000-wide q windows over 80M–450M. `--resume` reuses
  successful commands. Every relation is verified and duplicate coordinates are
  excluded from yield counts. `save_testsieve.py RUN` archives the completed
  measurements and logs to `data/c196/testsieve/`. This run used
  `stage23_gpu/build/runs_20261013/c196_testsieve/`.
- `20261016_loss_trace/run.py`: bounded C208 seed-10/103 retention experiments
  using `-trace-cell`. Default mode makes CPU plans; `--gpu` runs four sequential
  CUDA variants and verifies all exports. `--wide-pool` instead runs the two
  seed-10 pool-size follow-ups. Use a fresh `--run-dir` to freeze a new binary.
  Full artifacts live in `stage23_gpu/runs/20261016_loss_trace/`; compact evidence
  is retained in `data/c208/ropt_coverage/trace_20261016/`.
- `20261016_alpha_screen/run.py`: CPU-only replay audit of the intermediate
  alpha-screen stage. Alternates three C208 seed-10 policies for three repeats,
  then audits all 16 saved C171 depth pools with the selected bound, checking
  every exported resultant. Freezes the binary and records commands, hashes,
  timings and retention under `stage23_gpu/runs/20261016_alpha_screen/replay/`.
  Use a fresh `--run-dir` for another build. Requires the saved local pools;
  compact results, including a separate live CUDA validation, are in
  `data/c208/ropt_coverage/alpha_screen_20261016.json`.
- `20261017_alpha_quality/`: CPU audit of MurphyE/lattice quality after
  screening. `run.py` scores every distinct exported finalist, freezes skews,
  and validates leading retained/discarded configurations at higher accuracy
  on three independent lattice streams. `assess_exact.py` checks the corrected
  exact-size screen against those winners by identical integer coefficients.
  `holdout.py` replays a fixed policy on the first four unused depth pools from
  each of C147/C166; pass its directory via `run.py --replay-dir` and a fresh
  `--run-dir` to score them. The October 16 replay script additionally supports
  `--exact-size` and `--control-only`. Full artifacts are under
  `stage23_gpu/runs/20261017_alpha_quality/`; compact results are in
  `data/c208/ropt_coverage/alpha_quality_20261017.json`.
- `20261018_gpu_screen/run.py`: CUDA port audit against the 25 corrected CPU
  screen cases. Compares every screened score with CADO, all 200 exports per
  case byte for byte, and every resultant. `--timings --repeats 3` alternates
  CPU screen, GPU screen and full broad CPU refinement on C208 seed 10 and
  C171 seed 1, without the expensive oracle. Use a fresh `--run-dir` to freeze
  each build. Context wait and process wall time are recorded separately from
  refinement. Full artifacts: `stage23_gpu/runs/20261018_gpu_screen/`; compact
  results: `data/c208/ropt_coverage/gpu_screen_20261018.json`.
- `20261018_ropt_batch/run.py`: runs the same 25-case corpus in one persistent
  CUDA session and verifies all 5,000 exports; `--check` also compares every
  screened score with CADO. `--timings --repeats 3` alternates individual and
  batched CPU/GPU execution of all 16 C171 seeds with two host threads. Supply
  a fresh `--run-dir`; each run freezes its binaries and records the manifest,
  commands, hashes, phase/process timings and GPU workspace reuse counts.
  Artifacts live under `stage23_gpu/runs/20261018_ropt_batch/`.
- `20261019_c192/`: freezes the new C192 CPU references and executables;
  traces the original effort-5 winner through dense breadth/depth and automatic
  residue classes, and repeats effort 5/10 on both signs of the exact seed.
  `run.py pipeline` launches the full GPU path with provisional 32/33, 15e,
  q=80M–500M settings; `run.py finish` waits for it, compares all frozen CPU
  methods and archives the result. `compare.py --target` scores the focused
  seed experiments independently. `run.py screen` checks the GPU alpha screen
  against CPU on the recovered depth pool. `run.py driver-check` validates
  separate/batched live searches, checkpoint reuse and corruption repair.
  `run.py driver-screen-check` audits checked/unchecked GPU screening against
  CPU on all sixteen depth pools. `validate_winners.py --run-dir RUN` checks the small
  GPU/CADO model difference with 1,024,000 angles and three further lattice
  streams; `save.py` archives those reports and the independently scored winners.
  Driver batching now needs `ropt-batch-events-v1`: use a fresh run directory
  with rebuilt binaries for new batching experiments; existing frozen pipeline
  runs remain resumable with `--ropt-batch-size 1`.
  `review_gpu.py --run-dir NEW_DIRECTORY` copies the completed C192 fixtures
  and sixteen depth pools, freezes current CPU/CUDA binaries, and validates
  live batch/resume/repair plus CPU/GPU screening parity with two workers.
  It rejects existing destinations and verifies that the source inputs remain
  unchanged. `run.py driver-screen-check --workers 2` also selects two workers.
  Full artifacts: `stage23_gpu/runs/20261019_c192/`; compact evidence:
  `data/c192/ropt_coverage/`.
