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
