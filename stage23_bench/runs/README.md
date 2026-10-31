# Run scripts

The scripts that produced the GPU comparisons recorded in `../data/` and in
`../../GPU_STAGE23_PLAN.md`, kept here because the runs themselves happened in
directories that do not last: `stage23_gpu/build/` (deleted by `make clean`) and
`pipeline_results/` (moved to `previous/` and later deleted by the next pipeline run).
They take the run directory as an argument; each file's docstring or header says how.

- `20261026_review/README.md`: audit/fallback/resume review resolutions and
  reproduction of host, CUDA fault/state and 26-pool parity checks. Compact
  record: `../data/gpu_refinement/review_20261026.json`.

- `20261025_c179/inspect.py --source-dir CPU_RESULTS --run-dir NEW_DIRECTORY`:
  inspects the msieve-original-only winner, verifies common seed inputs, locates
  equivalent rotations across all four output sets, independently rescores a
  documented shortlist, and runs four CPU coverage plans. No GPU search.
  Fixtures and compact results: `../data/c179/ropt_coverage/`.

- `20261025_root_coverage/`: 90 CPU plans, thirty proposal-rank audits and 27
  paired GPU searches on the optimized dense baseline, followed by independent
  MurphyE/lattice scoring of every export. Shows proposal losses on C192/C197
  and significant class-modeling overhead; defaults stay unchanged. All scripts
  require `--run-dir`; `archive.py` also requires a fresh `--archive-dir`.
  Full artifacts: `stage23_gpu/runs/20261025_root_coverage/`; compact results
  and seeds: `../data/root_coverage/20261025/`. See its README for reproduction.

- `20261023_gpu_refine/run.py --run-dir NEW_DIRECTORY`: freezes binaries and
  copies the existing 25-case saved-pool corpus plus C192's winning depth pool;
  audits full CUDA alpha/translation refinement on 167,936 candidates and 5,200
  exports. Then compares CPU/CUDA MurphyE on 16 C208 and 16 C192 finalists with
  warmup, three alternating repetitions, and independent 256k-angle scoring.
  Requires the saved pools referenced by `20261018_gpu_screen/run.py` and the
  completed C192 run. Compact record: `../data/gpu_refinement/20261023.json`.

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
- `20261024_c197/`: production check of both new GPU paths after shared CPU
  initial sopt. Freezes CPU references, runs GPU resopt and independent root
  searches, audits all depth pools and sampled MurphyE cells, and compares CPU
  and GPU winners under standard MurphyE and the provisional job lattice.
  `validate_winners.py --run-dir RUN` rechecks the small model gap with fixed
  skews and denser samples. Compact results: `data/c197/gpu_ropt/`.
- `20261024_root_sieve/`: profiler-guided root-sieve optimization. Isolated
  thread-layout and table-lookup experiments precede exact CPU/GPU hit checks,
  memory/race checks and three alternating full C197 depth comparisons against
  frozen binaries and the original production pools/exports. `validate.py`
  takes `--run-dir`; its phases are `checks`, `sanitize`, `block-sizes`, and
  `timings`. Full artifacts: `stage23_gpu/runs/20261024_root_sieve/`.
- `20261025_storage_audit/`: storage inventory followed by authorized cleanup.
  Deleted all 101 manifested files (38.0 GiB) and pruned 2,247 unreachable Git
  objects (8.2 GiB); the execution and integrity checks are recorded in
  `cleanup_result.json` and `deletion_result.json`. Removed preprocessing indexes
  and sorted intermediates must be rebuilt before a full driver resume; retained
  seeds/pools support direct replay. Original `.zst` archives were preserved.
  The large inventories were archived locally on October 28; execution records
  and deletion lists remain versioned. Earlier AS660 score tables are now
  lossless gzip files, with bulky ledgers/checkpoints also archived locally.
  See the [evidence index](../data/README.md) for access and recovery details.
- `20261027_jobs/`: live C179 depth retention and bounded Murphy CPU/GPU audit,
  followed by independent full GPU runs for the newer C170 and C184 inputs.
  `retention.py` freezes the C179 fixtures and traces the known rotation without
  inserting it; `murphy.py` audits the first 32 saved proxy candidates. Both have
  `--score-only` for independent rescoring of completed checks. `jobs.py prepare`
  takes explicit source, destination and historical profile; pipeline/audit/compare
  phases preserve raw inputs, completed CPU references, pools and provenance.
  `compare.py --profile ID` rescores the same finalists under another profile;
  it does not repeat seed selection under those alternative settings. The catalog
  is `config/sieve_profiles.json`. C179 compact evidence is in
  `data/c179/ropt_coverage/live_20261027/`.
  Both full new-job runs completed: C170 in 59m 26s and C184 in 81m 31s,
  including independent GPU initial sopt. All 6,400 depth exports match CPU
  replay; C184 has seven initial-sopt coefficient differences within the
  recorded size-objective tolerance. The preferred C184 scenario is 31/32,
  q=40M–200M; its 32/33, q=35M–175M alternative rescores the same finalists.
  `save.py --run-dir RUN --destination NEW_DIRECTORY` exports a full diagnostic
  snapshot with verified hashes; use a destination under `stage23_gpu/runs/`.
  Versioned summaries in `data/{c170,c184}/gpu_ropt/20261027/` retain conclusions,
  settings, winner coefficients and provenance. Full replay still needs the
  local inputs and pools. See the [evidence index](../data/README.md) for retained
  files and relocated snapshots. No matched full CPU timing or test sieving
  was performed.
- `20261027_root_coverage/`: bounded size-aware class proposals with reserved
  u diversity, evaluated on the loss corpus and four unused C170/C184 seeds.
  `study.py` runs preparation, CPU plans, paired GPU searches and independent
  scoring using the prior audit harnesses with frozen study configurations.
  `validate.py` compares GPU class geometry with CPU plans and replays every
  first-repetition pool. `trace_quality.py` diagnoses losses of the actual dense
  MurphyE leaders; `archive.py` exports full diagnostic and source snapshots
  to an ignored destination under `stage23_gpu/runs/`.
  All 60 plans and 96 GPU searches completed; 6,400 CPU replay exports match.
  The new opt-in modes are faster but still lose quality on several seeds;
  dense defaults and driver forwarding are unchanged. Results and limitations
  are in the study README and `data/root_coverage/20261027/`.
- `20261027_coverage_review/`: shared cached probing, bounded scans around
  selected basin centers, corrected work limits/diagnostics and stronger
  regression tests. `run.py host` compares all 60 plans with the frozen study;
  `run.py gpu` checks six live searches against CPU plans and pool replay.
  All 30 dense/legacy plans are unchanged and all 1200 GPU exports match CPU.
  The study README maps all thirteen findings to fixes. Compact evidence is
  `data/root_coverage/review_20261027/`. Previous prototype speed/quality numbers
  are historical; this check does not establish new performance.
- `20261028_existing_jobs/`: the first six completed GPU pipelines cover C148, C164, two
  C179 inputs, C181 and the union of both C190 stage1 runs. The follow-up
  [compact report](../data/gpu_pipeline/20261028.json) compares 6,885,846 initial
  outputs. All 19,200 depth exports match CPU replay exactly. Two initial
  size-optimization outputs are worse than CADO beyond tolerance; both are
  outside the selected top 1000, and CPU execution of the frozen port matches
  CADO on their isolated inputs. The October 29 matched-log prototype fixes
  both in GPU batch replay; the October 30 production integration is implemented,
  with focused new-build GPU checks and all 8,028,269 saved corpus outputs
  matching their CPU references exactly.
  Common rescoring favors C190's 60060 CPU run by 10.49% in standard MurphyE;
  the GPU union result is 0.11% above that CPU winner. Lattice settings are
  provisional and no test sieving was performed. Full review scripts and
  artifacts are local under `stage23_gpu/runs/20261028_results_review/`.
  Queue preparation now pins C190 to its two dated October 28 backups and
  checks that all inputs have the same composite. C164 uses the saved input
  and CPU references under `stage23_gpu/runs/20261029_queue_sources/c164/`
  (hard links to the immutable completed run, not live repository files).
  `--only` refreshes all per-job results before aggregating status. Background
  launch records are kept separately in `background/PID.json`; `owner.json`
  identifies the process that acquired the queue lock and records its finish.
  The [October 29 follow-up](../data/gpu_pipeline/20261029_followup.json) adds
  C170 (`418691…`) and C152: both jobs and their audits completed successfully.
  Of 1,142,423 initial outputs, two C170 coefficient differences are within
  score tolerance and neither enters the top 1000; all 6,400 depth exports match
  CPU replay exactly. The October 30 correction withdraws the claimed C170
  export regression: the old comparison changed the exported skew. Actual
  fixed-skew scoring puts the export above that control. Corrected comparisons
  and new final-ranking replays are in the
  [integration report](../data/gpu_pipeline/20261030_integration.json).
  Pipeline times were 1h 05m 29s and 38m 01s;
  no matched CPU timing or test sieve was performed. The report preserves the
  two raw differing inputs and distinguishes shortlist winners from exports.
  The aggregate queue still reports the older C181/C179 validation failures;
  neither new job introduced a validation failure. The matched-log prototype
  was not used by these runs.
  The later [C150 review](../data/gpu_pipeline/20261030_c150.json) recovers a
  msieve candidate that the printed-score shortlist discarded: its coefficients
  have gcd 2, and after normalization it ties the CADO/GPU winning family near
  `7.23e-12` standard MurphyE. Comparison now includes every distinct
  content-reduced msieve export and deduplicates its three per-family leaders.
  C150's 2,782 initial differences against imported CADO all match local CADO
  on CPU replay; 147 remain worse than the imported reference, so that queue
  audit error is preserved. All 3,200 depth exports match CPU replay exactly.
  Corrected comparison and diagnostics remain under ignored run directories.
- `20261029_sopt_descent/`: CPU phase diagnosis of the two initial-sopt losses.
  `trace.py perturb --run-dir NEW_DIRECTORY` reproduces both saved GPU outputs
  with one-ULP lognorm perturbations in copied headers, while the integer
  translation/LLL/deduplication phases stay unchanged. The seed-zero controls
  match all CPU phases. This is a sensitivity experiment, not a CUDA math
  simulation. `trace.py gpu` explicitly runs a device comparison when the GPU
  is free. Full traces and binaries
  stay under ignored `stage23_gpu/runs/`; the
  [compact result](../data/gpu_pipeline/20261029_sopt_descent.json) records the
  experiment and provenance. That diagnostic initially left production math
  unchanged; the later October 30 integration is recorded separately.
  Context replay accepts `--case NAME --context-run ORIGINAL_RUN` and preserves
  the failing 4096-input batch plus preceding batches (`--full-prefix` for all).
  `--prepare-only` extracted and verified 8192 inputs per case without GPU work.
  The post-hoc perturbation search tried 16 seeds per case, with one exact
  saved-GPU match each; both denominators are retained in the compact result.
  New runs compile from frozen sources and record dependency/library/compiler
  provenance. The earlier report's source hashes alone did not prove that its
  copied executables matched those headers.
  GPU follow-up reproduced both losses in their batch contexts; the matched-log
  prototype matches all 16,384 outputs and both complete target traces to CPU.
  It also matches all 46 historically differing inputs and all 13,000 C146
  acceptance outputs to CADO. The original binary reproduced all 46 historical
  GPU outputs. Detailed execution/audit scripts are local under
  `stage23_gpu/runs/20261029_host_log/`. October 30 adds production host-variant
  selection, startup device validation and unsupported-host CPU fallback. Host
  checks pass. The new build also passes device log checks, all 48 historically
  differing inputs against current CADO, and both 8192-input batch contexts and
  target traces against CPU. All 28,000 C146/C161 effort-0/50 fixture outputs and
  all 8,028,269 outputs in the eight-job replay also match their CPU references
  exactly. The later C150 job is outside that corpus.
  Final-ranking replays on C170/C152 preserve all
  declared controls and independently score additional skew proposals, with
  exported-file readback checks. Full artifacts remain ignored under
  `stage23_gpu/runs/20261030_shipping/`.
  A subsequent October 30 code-review follow-up changes the remaining sopt
  objective logs, common content normalization, skew precision and report
  provenance. It also pins C152 to saved queue inputs under
  `stage23_gpu/runs/20261030_queue_sources/c152` (hard links, no bulk data copy).
  October 31 CPU and focused GPU validation passed after fixing a refinement
  output-parsing bug. All 47,214 focused outputs and complete target batch
  traces match their references; see
  [the compact report](../data/gpu_pipeline/20261031_validation.json).
  Logs and fresh frozen builds are under
  `stage23_gpu/runs/20261031_focused_validation/`. The full corpus was not rerun;
  its results above remain evidence for the earlier implementation.
