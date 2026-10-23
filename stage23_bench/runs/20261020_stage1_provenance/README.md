> Historical run record moved from the main notes on 2026-10-22. See the [current conclusions](../../../POLYSELECT_OPTIMIZATION_NOTES.md) for subsequent evidence and revised priorities. Statements about a running or stopped process describe the time of this record.

## Prefix fix and provenance implementation (2026-10-20)

Implemented with authorization while the GPU was occupied. These are correctness
results, not representative timing or overhead measurements. No C196 search was
started during those implementation tests and no launch/metadata performance
changes were included. The subsequent user-run search is recorded above.

- Fixed the pp64 prefix narrowing load. The independent arithmetic regression
  `stage23_bench/tests/stage1_prefix.cu` exercises prefixes exactly equal to 2^32,
  repeated q roots, invalid trailing q, tile boundaries, and multiple offsets.
  The old kernel gives 80 mismatches across 12 cases; the corrected kernel gives
  zero. The source fix is isolated from the host instrumentation.
- Added optional buffered per-worker JSONL, inclusive q-window controls, and a
  fixed-progression control. The default search retains adaptive progressions.
  The q factor base is initialized from the natural bound before intersecting
  the requested window. Empty intersections are recorded and skipped.
- Added independent CPU collision reconstruction and an identity ledger in
  `stage23_bench/tools/stage1_provenance.py`. Rejected returned hits can be
  reconstructed offline, without submitting them to production callbacks.
  Raw outputs join by N/P/m; sopt joins through its printed raw input; ropt joins
  require an exact translation/rotation from a linked sopt polynomial, including
  sign changes. Ambiguous origins remain explicitly ambiguous. Preserve the
  ledger's candidate_id on future submissions when possible.

New GPU options (inside the existing whitespace-delimited NFS argument string):

```
stage1_provenance=/path/to/new-prefix
stage1_qmin=100000000 stage1_qmax=100200000
stage1_aprog=1
```

`stage1_provenance` creates `PREFIX.w0.jsonl` etc. Existing files are refused;
use a fresh prefix for each process/replay. Parent directories must already exist.
Omit q limits for the natural range. Bounds are inclusive, so adjacent windows
are `[lo, mid-1]` and `[mid, hi]`. Values must fit 1..4294967295. `stage1_aprog`
accepts 1..5; omit it for the existing adaptive policy. A fixed policy is required
for split/unsplit candidate-set equivalence; otherwise boundary batches can add
extra offsets. This is resumability at completed-window granularity, not an
arbitrary interrupted-batch checkpoint. Replays need fresh output plus deduplication.

Sidecar schema 1 records:

- `run`: N, degree, worker/prefix identity, engine/GPU, arguments, build timestamp,
  CUDA header version, capacities. Also archive source revision/diff, build command,
  binary/PTX/DSO hashes, driver/toolkit versions, and intended sieving settings in
  the experiment manifest; the sidecar build timestamp is not a source hash.
- `coefficient`, `setup`, `batch`, `launch`: effective/configured bounds, p/q
  windows, entries per q, every searched q/root (including no-hit work), progression
  count, launch layout, and setup/q-generation costs. A q may span batches.
- `hit`: exact collision integers, gate ratio, admission decision, and validated
  original ad/P/m for admitted valid hits. Rejected-hit `valid` is null until
  offline reconstruction. Callback submission does not guarantee raw expansion;
  the optional raw-file join identifies actual expanded output.
- `engine`, `batch_end`: candidate/match/cap/fallback counters, returned versus
  total hits, saturation, CUDA event span, and host elapsed scopes. Event spans
  include host-mediated gaps and are not active-kernel time. Fatal engine overflow
  leaves an unfinished batch; keep the process log for its cause.
- `coefficient_end`, `worker_end`: completion markers. Wait for process completion
  and stage-2 queue drain before joining raw output. Missing endings or saturation
  make the recorded population incomplete. Timing ends before the final log flush;
  use external coefficient/process timing to measure total logging overhead.

Example CPU audit after the process finishes:

```
python3 stage23_bench/tools/stage1_provenance.py /path/to/prefix.w*.jsonl \
  --raw /path/to/search.ms --ledger /path/to/new-ledger.jsonl --require-complete
```

Optional `--sopt CADO_SOPT_OUTPUT` and `--ropt ROPT_OUTPUT` add downstream joins;
`--ropt-format cado|msieve|poly` selects the root-optimized input format. Several
files can be supplied by repeating these options. The ledger retains duplicate
hit provenance. It does not infer q exposure or cost from successful hits alone.
The audit is CPU-only but holds work/candidate indexes in memory; split large
analyses by completed coefficient/window if necessary.

Validation: CUDA build/PTX and CPU parser/reconstruction tests passed. The bounded
GPU suite `stage23_bench/tests/stage1_integration.py` uses an isolated directory,
128 MB batch-memory setting, and the saved c157 input. Logging on/off both found
12 polynomials; split windows found 6+6 with exactly matching combined coverage
(6,035 q roots), and sort/Gerbicz returned identical polynomial multisets.
Multi-progression runs both found 8; adaptive batches, two workers, and an empty
intersection passed. All 12 accepted hits independently reconstructed and joined
to raw output, then to actual CADO effort-0 sopt output. One actual CADO
ropteffort-0.1 output also joined through an exact transformation. Synthetic CPU cases cover
rejected hits, truncated/saturated logs, and downstream multiplier/sign/translation
changes. Results and reproducible commands are in
`stage23_bench/runs/20261020_stage1_provenance/validation.json`.

Still pending: quiet-device logging-overhead measurements, representative C196
workloads, production-scale provenance/quality analysis, and performance tuning.
The <=1% logging-overhead target has not been established on this shared GPU.
