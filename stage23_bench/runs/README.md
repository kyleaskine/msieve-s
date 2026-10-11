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
