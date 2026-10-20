# GPU root optimization: wider search at equal elapsed time

Started 2026-10-13. Implementation follows `GPU_STAGE23_PLAN.md`; numerical
baselines remain the existing dense root sieve and CPU hybrid. The immediate
target is degree five on one GPU with at most two concurrent searches and short
kernel launches.

## Priorities after pipeline integration

1. **Wider selective root search:** discover distant size basins and search good
   small-prime residue classes without lowering the norm budget so aggressively.
   This is the implementation work started here.
2. **Size-optimization speed:** profile descent and exact LLL; investigate a
   certified warm-started skew solver and smaller/length-aware integers with
   fallback. Preserve full-precision acceptance and finalist quality.
3. **Search allocation:** replace fixed breadth/depth counts with measured
   improvement per additional second and an exploration allowance outside the
   initial seed shortlist.
4. **GPU scoring and ranking:** batch accurate MurphyE, retain diverse finalists,
   and validate ranking changes with actual test sieves. Evaluate alternative
   root-distribution scores on saved, measured finalists before using them to
   discard candidates.

Increasing sopt effort, explicit multiplier sweeps and quadratic rotations are
lower priorities for the tested degree-five jobs: existing experiments found
little useful gain. These conclusions do not cover degree six.

## Baseline and problem

The integrated driver has passed a real, bounded GPU run and split/all/resume
checks. Full saved-input driver runs subsequently completed on C147/C166
(October 15) and C171 (October 16); these establish integration and model
comparisons, but controlled end-to-end speed and test-sieve comparisons remain
pending. Recent component runs, made by the earlier
job scripts, took 13.9/31.1 minutes for initial sopt on C165/C187 and 14.7/17.2
minutes for their root-search breadth passes. Root size modeling has already
been accelerated; the measured dense breadth pass is now sieve-bound.

The current root optimizer scans consecutive u lines from zero, warm-starting
each line's v/translation search from its neighbor. It stops after enough lines
outside the size band, or after the per-direction line allowance is exhausted.
The cell cap then reduces the norm budget; at its floor it drops whole lines
and trims a remaining interval near the best line.

Documented losses on C208 include:

| Case | First known loss boundary |
|---|---|
| Seeds 15, 27, 32, 103, 151 | Cell cap excludes rotations whose alpha compensates for their size |
| Seed 222, CPU rotation u = -16152 | Line scan stops near absolute u = 500 before reaching the basin |
| Seed 32 | Extending the line scan lowers the best found lognorm from 64.00 to 62.94 |
| Seed 10, cell (7, -174864538) | Earlier proxy retention loss; exact-alpha reranking subsequently fixed it |

Recovering a missed family is evidence of better coverage, not proof of a better
overall polynomial. A new search must be evaluated through the entire retention,
translation/skew refinement and ranking sequence.

## R0: reproducible baseline and loss corpus

- Freeze the current CPU/CUDA binaries before changing the search.
- Preserve available loss seeds, exact coefficients, n, reference rotations,
  job parameters and the commands that expose each loss. Locate the old C208
  inputs; if unavailable, label those cases as historical rather than claiming
  to have rerun them.
- Add a controlled shifted-polynomial regression: adding U*x*g moves the same
  rotation landscape by -U without changing the resultant. This tests whether
  the scan reaches a known distant basin; it is not independent evidence that
  a residue-class filter preserves winners.
- Include unshifted seeds from multiple sizes and content-derived seeds.
- Record model evaluations, lines sampled/expanded, cells admitted after the
  cap, elapsed times and retained candidates. Separate a discovered basin from
  one whose cells are actually sieved.

## R1: coarse basin discovery — first implementation

Add an opt-in coarse scan across the requested [-umax, umax] range. Preserve the
existing local scan and its default output when the option is absent.

1. Sample a bounded grid in u, including zero and both endpoints. Evaluate the
   existing exact-coefficient size model at each sample; these remain local
   minima over v and translation, not certified global minima.
2. Use warm starts plus an independent start where useful. Never use a failed
   or nonfinite evaluation as evidence that all of a region is uncompetitive.
3. Select geographically distinct promising coarse basins, then refine their
   u locations and expand bounded neighborhoods with consecutive lines.
4. Merge added lines into the existing model before band construction. Keep
   the existing sieve, exact-alpha reranking and MurphyE implementation.
5. Expose the options in `s23_ropt`, `-plan`, and the reusable Python driver;
   record them in checkpoints. Report sampled/added lines and selected basins.

Bound the extra model work explicitly. Do not silently reinterpret `-maxlines`
as the coarse-scan allowance. Under a fixed cell cap the new best size minimum
can change which old lines survive, so enabling this experimental mode does not
guarantee preservation of the old winner. Compare both paths before promoting it.

R1 acceptance:

- Disabled mode preserves the frozen binary's exported polynomials.
- Unit checks cover grid endpoints, limits, duplicate/overlapping neighborhoods,
  deterministic selection, invalid options and nonfinite probe results.
- A shifted real-polynomial case exposes a distant basin missed by the local
  scan and recovered by the new scan; preserve the original as a control.
- Exact resultants and CPU/GPU output agreement on a bounded search.
- Measure overhead and basin changes on ordinary real seeds. Do not describe a
  recovered minimum as improved sieve yield.

## R2: root sieve on residue classes

Implement a separate opt-in path for
`(u,v) = (u0,v0) + M*(i,j)`, beginning with a small configurable smooth modulus
and treating 2520 as a candidate, not a fixed optimal choice.

- Rank classes using small-prime affine and projective root information and
  attainable size estimates. Reserve some classes for exploration.
- For primes not dividing M, reindex the existing periodic tables. For primes
  dividing M, handle constant valuations and conditional lifting correctly.
- Keep original u/v coordinates for size interpolation, exact export and all
  retention diagnostics. Handle content before final scoring as today.
- Stream work in bounded launches. Account for actual selected cells rather
  than charging the enclosing dense rectangle.
- First compare one class against direct brute-force root counting, then prove
  that the union of all classes reproduces a small dense reference window.
- Only then test reduced class selections on independent seeds/windows and the
  loss corpus. Measure losses at class selection and every later cutoff.

R2 is what should make distant high-alpha regions affordable. R1 alone leaves
the existing budget-lowering cap in place and cannot solve those coverage losses.

## R3: allocation and integration experiment

- Use plan tables to select meaningful next budget steps instead of repeating
  equivalent capped searches.
- Compare dense, coarse-basin and selective-class searches at equal elapsed
  time, GPU concurrency and scoring effort on several saved jobs.
- Track best independently scored candidates against time, plus per-family
  coverage diagnostics and survivor diversity.
- Replay the production driver on at least one full saved raw-input job, with
  all hashes, settings and final exports retained outside disposable directories.
- Keep experimental options off by default until the comparisons justify a
  default change. A wider seed funnel is a subsequent allocation experiment.

## Final quality gate

Use at least 16k search / 64k validation angles during MurphyE refinement and
256k final evaluation. Preserve control configurations. Where lattice scoring
is used, validate on an independent sample stream and the actual job region/q
range. Test-sieve diverse leading candidates using the actual lim/lpb/mfb
settings; alternate candidate/control q windows and report both unique relations
per special-q and per second with uncertainty across windows.

Success means the same quality sooner or a better validated polynomial at equal
time. No numeric speedup or yield improvement is promised before measurement.

## Implementation record

### 2026-10-13: R0 corpus and R1 first implementation

- Frozen CPU/CUDA control binaries and hashes in the local development run
  `stage23_gpu/runs/20261013_ropt_basins/`.
- Recovered seven exact C208 seeds and six historical MSIEVE reference outputs;
  repository fixtures live in
  [`stage23_bench/data/c208/ropt_coverage/`](stage23_bench/data/c208/ropt_coverage/README.md).
  The manifest includes original u/v/t coordinates, provenance and seed hashes.
- Implemented bounded sampling, cached warm/independent starts, deterministic
  basin selection, local refinement and dense expansion in `stage23_gpu/include/rscan.h`.
  The root tool logs both discovered centers and center lines retained after
  the cap. The Python driver forwards and checkpoints the settings in every
  root-search phase, for both one-command and split workflows.
- Opt in with `--u-scan 256 --u-basins 4 --u-radius 128` in a new driver run;
  use single hyphens for the standalone tool. `u_scan=0` remains the default.
- Full `make -C stage23_gpu test` passed, plus fixture checksum/reference tests.
  New checks cover bounded work, caching, failed probes, alternate starts,
  distant-basin recovery, invalid CLI settings and pipeline resume behavior.
- Eighteen bounded searches across C161, a shifted C161 control and C208 seed
  222 verified CPU/CUDA byte-identical exports. With the scan disabled, both
  builds also matched their frozen controls. All exported resultants and sampled
  brute-force root counts passed. Each search used at most 100,000 cells, two
  host threads and `-launch 30`; this is correctness validation, not throughput
  measurement.

Nine sequential plan comparisons used `-budget 3 -maxcells 2e11 -maxlines 1000
-umax 20000 -t 2`, adding `-u-scan 256 -u-basins 4 -u-radius 128` for the new path:

| Fixture | Local minimum -> coarse minimum | Model seconds, local -> coarse | Coarse cells retained |
|---|---|---|---|
| C208 seed 32 | 64.0027 -> 62.9378 | 2.93 -> 4.84 | 9.96e10 |
| C208 seed 222 | 67.7577 -> 65.3283 | 3.96 -> 5.27 | 3.73e10 |

The seven controls (C146, C161, C168 winner plus two content seeds, C185 and
C204) found no lower minimum. Added model time ranged from 0.25 to 1.61 seconds.
These are single-run CPU model timings; the plan command does not run the sieve,
and its additional budget-table printing is outside the reported model time.
Commands, hashes, complete plan tables and wall times are in `measurements.json`
in the development run; bounded export comparisons are in `validation.json`.

Seed 222 retains 50 consecutive lines around the discovered u=-18647 basin.
The specific historical winning rotation at u=-16152 remains outside the added
neighborhoods. The shifted C161 control recovers its original minimum at
u=-4096 (47.7495, versus 53.4497 from the shifted local scan).

**R1 result.** The first milestone improves size-basin coverage; it is not a
measured pipeline speedup or a demonstrated improvement in final polynomial
quality. Equal-time production replay and test-sieve comparisons remain pending.

### 2026-10-13: R2a explicit residue-class execution

Implemented `s23_ropt -residue M U V` for window and search modes, on CPU and
CUDA. This is one explicitly chosen class `(u,v)=(U,V) mod M`, with M in
1..65536. It is off by default; M=1 reproduces dense behavior. The Python driver
accepts `--residue M,U,V` in both workflows and saves normalized class settings
for resume. The same explicit class applies to every seed; it is not yet an
automatic polynomial-selection policy.

- Original prime-power/projective tables are indexed at `v_start + M*j`.
  For table period q, the progression has period `q/gcd(q,M)`: it becomes
  constant if q divides M and retains conditional lifts otherwise. No inverse
  of M is taken, including for primes dividing the modulus.
- CPU and CUDA use separate dense/strided specializations; dense cells do not
  pay the per-cell stride arithmetic. Block memory remains bounded by the
  launch size. Returned hits, size interpolation and exact polynomial exports
  retain original u/v coordinates. Existing content division and exact-alpha
  and MurphyE refinement apply to these original-coordinate hits.
- Class search visits matching u lines at spacing M, from near zero in each
  direction, using warm and independent starts for M>1. Existing patience and
  line-count limits apply to these lines. Its reference minimum is the best
  found on these visited lines. Coarse `-u-scan` and explicit classes are
  currently mutually exclusive; composing the two remains future work.
- Bands are intersected with the v progression before counting, trimming and
  knot/segment generation. Search cell caps, block lengths and launch counts
  charge actual selected cells, including partial segments and the final
  trimmed line. Empty intersections are skipped.

Validation:

- Full host `make test` passed, including original dense/root/size/MurphyE tests.
- New geometry tests cover negative coordinates, empty intersections, clipping,
  cap trimming, signed-integer endpoints, partial blocks and interpolation in
  original coordinates. Exhaustive top-k checks include short blocks, ties,
  CPU chunk crossings and moduli 1, 2, 8, 128, 2520 and 65536.
- All classes for small moduli reproduce each cell of a dense reference window
  exactly once, with identical root scores. Affine-only and projective modes
  pass on ten real fixtures, including both content-lattice seeds and C208.
  Independent brute root counts validate large signed coordinates and primes
  dividing the modulus.
- All five CLI acceptance tests also passed on CUDA: union/dense equivalence,
  selected-cell caps, historical coverage, exact content exports and invalid
  options. Separate CPU/CUDA comparisons across four fixtures produced identical
  class exports; dense exports match the frozen pre-R2 CPU/CUDA binaries.

Known-class coverage experiment: supply each historical winner's class modulo
2520, with `-budget 3 -maxcells 2e11 -maxlines 1000 -umax 20000 -t 2`.

| C208 seed | Budget after cap | Selected cells | Historical rotation admitted |
|---|---|---|---|
| 15 | 3.00 | 7.90e10 | yes |
| 27 | 2.50 | 2.00e11 | yes |
| 32 | 2.75 | 1.88e11 | yes |
| 103 | 1.00 | 1.00e11 | yes |
| 151 | 2.75 | 1.29e11 | yes |
| 222 | 1.50 | 1.26e11 | yes |

For seed 222, an actual GPU search with class `(1488,1704) mod 2520` visited
16 u lines and 1.26e11 selected cells. Sieve time was 3.97 seconds, model time
0.23 seconds, exact-alpha refinement 1.36 seconds, total process wall time
5.84 seconds, using one worker, two host threads and `-launch 30`. All 128
sampled hits plus printed finalists matched brute force. The previously missed
rotation `(-16152,6992188045104)` survives into the 200 exported polynomials.
This is a single execution measurement on the RTX 5070, not a comparative
throughput claim. All 272 exported polynomials across 25 validation runs have
nonzero resultants divisible by n.

These classes were taken from known winners. They prove reachability and
retention for this supplied-class experiment, not the effectiveness of an
automatic class selector or improved overall polynomial quality. Frozen
baseline and experiment binaries, source/input/binary hashes, commands,
timings, plan tables and exports are saved locally under
`stage23_gpu/runs/20261013_ropt_classes/` (`coverage.json`, `validation.json`,
`host-tests.log`).

**R2b follow-up (implemented below): class selection and allocation.** Rank classes using conditional
small-prime affine/projective information and size estimates; handle content
classes through the existing content-seed path. Reserve an exploration share.
Use one total cell/time allowance and one global candidate pool across all
selected classes for a seed. Validate selection on held-out seeds and windows,
including explicit accounting for winners lost by class selection and later
cutoffs. Only then run the R3 equal-time comparisons and final quality gates.

### 2026-10-14: R2b automatic selection and shared budget

Implemented `s23_ropt -classes M K [-class-explore E]` and driver
`--classes M,K [--class-explore E]`, off by default. Both one-command and split
raw-input workflows support these settings; pipeline checkpoints reject changes
within an existing run. The explicit single-class mode remains available.

- Compute conditional mean scores from the existing prime-power tables,
  averaging unfixed u/v lifts modulo gcd(M,q). Projective contributions are
  included. Class-independent primes do not affect proposal ordering.
- Keep at most 4*K high-root-score proposals plus max(16,K) deterministic
  exploration proposals, bounded by M*M. Correct guaranteed content dividing M
  in both the estimated norm and root score. Actual content division remains
  part of exact final scoring; content-derived pipeline seeds remain enabled.
- Scan the proposed u residues with warm and independent starts. For each
  proposal, evaluate admissible v coordinates near its best sampled line's
  minimum. Rank the resulting local size/root estimates; reserve E slots outside
  the leading ranking. Default E=floor(K/4); pipeline -1 selects this default.
- Bound M to 2..4096, K to 1..min(64,M*M), and proposal rows to 65536 possible
  u lines in the requested range. Memory for proposals is bounded independently
  of M*M. Automatic mode cannot combine with explicit classes or coarse scans.
- Merge all selected classes into one size-band/cell-cap pass and one streamed
  sieve. Count disjoint residue progressions exactly when trimming mixed-class
  lines. Use one global proxy pool, exact-alpha refinement and final output cut.
  Print per-class cells and survivors at both retention cuts.

Validation passed: full `make -C stage23_gpu test`, including independent brute
roots for the conditional means, guaranteed-content checks, deterministic
selection, exploration, mixed-class geometry, CLI caps and exports, and both
pipeline entry points/resume. All three new CLI tests also passed on CUDA.
Forty bounded CPU/CUDA searches over C161, two C168 content seeds and C208 seed
222 verified identical exports between CPU and GPU. Dense and explicit-class
exports match the frozen R2a binaries. Seven additional full-cap CUDA searches
passed sampled brute-root and exact resultant checks. Across those 47 searches,
all 1520 exported polynomials have nonzero resultants divisible by the input n.

Automatic coverage experiment used **no reference classes as inputs**:
`-classes 2520 16 -class-explore 4 -budget 3 -maxcells 2e11 -maxlines 1000
-umax 20000 -aw 1.3 -rerank 4096 -refine 200 -top 8 -check 128 -t 2 -launch 30`.

| C208 loss seed | Reference class selected / rotation admitted | Rank among 200 exact-alpha finalists | Selected cells | Process seconds |
|---|---|---|---|---|
| 15 | yes / yes | 1 | 6.94e10 | 7.61 |
| 27 | yes / yes | 1 | 1.92e11 | 16.58 |
| 32 | yes / yes | 1 | 1.67e11 | 14.64 |
| 103 | no / no | absent | 1.98e11 | 16.54 |
| 151 | yes / yes | 1 | 1.20e11 | 11.95 |
| 222 | yes / yes | 1 | 1.14e11 | 10.36 |

These ranks use the existing lognorm + 1.3*exact-alpha objective. They are not
MurphyE ranks or measured test-sieve yield. Seed 222 used 3.96 seconds for its
size/class model, 5.24 seconds for sieving and 1.07 seconds for exact refinement.
Timings are single sequential runs on the development GPU, not equal-time
comparisons or a claimed speedup. The additional seed-10 control admits its
historical rotation but does not retain it in the final 200; its individual
block/proxy/exact-cut loss still needs diagnosis.

Six independent planning controls (C146, C161, C185, C204 and two C168 content
seeds) found that M=2520/K=16 can be sparse and can miss the best local size
minimum. Automatic model time was 1.30..2.60 seconds versus 0.04..4.80 seconds
for the dense controls. These are planning checks, not final-quality tests.
Exploration reserves class slots only: a tight norm/cell cap can still remove
an entire class, and a shared proxy cut can remove all its candidates.

Frozen R2a/experiment binaries, source/input/binary hashes, commands, all plan
tables, exports and logs are saved locally in
`stage23_gpu/runs/20261013_ropt_class_selection/` (`validate.py`,
`measurements.json`, `validation.json`, `host-tests.log`, `gpu-cli-tests.log`).
The directory retains the experiment's start date; it completed on October 14.

**Next:** diagnose seed 103's class-selection loss and seed 10's later retention
loss, compare moduli/proposal diversity and class-specific budget allocation on
independent seeds, then run R3 equal-time MurphyE and test-sieve comparisons.
Keep the dense control and experimental opt-in defaults until those comparisons
justify changing the production policy.

### 2026-10-14: pipeline review corrections

Addressed the 15-item review of the uncommitted driver, wrapper and class sieve:

- Partial MurphyE failures now have a distinct tool status (3 when valid
  finalists remain). The driver also understands the older frozen binary's
  status 1, but only accepts a completed partial export with valid nonzero
  resultants and no reported brute-root disagreement. Empty, malformed or
  unrelated failures still fail. Partial results are checkpointed and reported.
- Breadth/depth save the full pre-refinement candidate pool. Accurate MurphyE
  replays the deepest completed pool without repeating the size model or sieve.
  Pool files bind exact seed coefficients, alpha weight and projective mode;
  truncation, incompatible metadata and duplicate cells are rejected. Older
  frozen binaries retain the existing combined path, so old runs remain usable.
- Deduplicate polynomial/skew configurations before the finalist limit.
  Keep execution settings out of checkpoint identities and record their latest
  values separately. Use explicit stage schema versions for compatibility and
  source hashes for provenance; unrelated code edits no longer invalidate sopt.
- Copy and hash raw/imported input bytes into the run before consuming them.
  Resume uses the snapshot even if the source grows or disappears. A bounded
  initial byte length prevents following a live append indefinitely; incomplete
  polynomial blocks still fail validation. Old runs can reuse verified outputs
  even when their original input is unavailable.
- Convert explicit parent residue classes through each content seed's affine
  coordinate map, skipping disjoint lattices. Validate a conservative automatic
  class row bound before preprocessing. Unify explicit/automatic sparse band
  geometry, preserving dense/M=1 behavior and single-class trimming.
- Count sopt headers during the existing parse, percent-encode SQLite paths,
  and remove the unused score-mode condition.
- Validate and route CPU `all` options before preprocessing; forward standalone
  preprocess options. Guard empty Bash arrays. Help and wrapper tests work
  without a local configuration file.

Full host tests passed, including 24 driver regressions and real MurphyE pool
replay/invalid-pool checks. The bounded CUDA acceptance run covered three real
fixtures, CPU/GPU output and pool agreement, dense/explicit exports against the
frozen R2b binary, and direct-versus-replayed MurphyE equality. Real GPU `all`
and `preprocess`/`pipeline` runs produced the same winner. Resuming after live
input growth with changed device/thread/worker/launch/slice settings reused all
command outputs. Forty acceptance commands checked 110 exported polynomials.
Artifacts, frozen binaries, hashes, commands and logs are saved locally in
`stage23_gpu/runs/20261014_pipeline_review/`; these are bounded correctness
checks, not production throughput or polynomial-quality measurements.

### 2026-10-16: C171 review and first-loss tracing

Reviewed the completed C171 driver run: 786,898 raw polynomials, 317 breadth
searches, 16 depth searches and four full-pool MurphyE refinements. Its best
GPU candidate scores **0.8733% above the best CPU finalist by the saved lattice
model**. Ordinary MurphyE ranks the CPU CADO winner higher (GPU is 2.2458%
lower on that metric). All 172 comparison polynomials have valid nonzero
resultants divisible by n. Scoring uses inherited C168 job settings; subsequent
user-reported test sieves are recorded below. The CPU screen considered 2,100 candidates,
retaining three cells per Y1 and pass before finalist validation. Initial sopt
needed three CADO fallbacks. Compact evidence and winner polynomials are in
`stage23_bench/data/c171/gpu_ropt/summary.json`; the original run remains under
`stage23_gpu/runs/20261016_c171/`. This review rechecked resultants and saved
reports; it did not rerun the full job or its scoring.

The user's later GGNFS 14e algebraic test sieves cover q=40,62,84,106,128,150M
with rlim/alim=80M/120M, lpb=31/32, mfb=60/92 and lambda=2.35/3.5. GPU is
faster at all six samples: the reported aggregate rate is 35.625 versus 33.448
rel/s (+6.51%), while projected relations are 349,684,018 versus 348,374,783
(+0.376%). The CADO job's active skew is 12,994,190.161, versus 16,417,714.9
in the saved model; GPU uses 39,903,169.3 in both. A fresh fixed-configuration
lattice check with lpb=31/32 predicts +2.09–2.14% GPU yield at the supplied
skews across three streams, or +1.53–1.56% using the saved CADO skew. This
model has no lim/mfb/lambda acceptance or execution-cost model, so it cannot
explain the measured speed gain. The GPU's smaller rational norms are a
possible cost explanation, pending phase timings. Preserve the distinction
between short test-sieve timing, extrapolated relation counts and model
predictions. User reports, settings and rerun commands are saved in
`stage23_bench/data/c171/gpu_ropt/testsieve_user_20261016.json`; raw sieve logs
and repeated timings were not supplied.

Added optional `s23_ropt -trace-cell U V` diagnostics. A watched cell is never
inserted into proposals, model starts or survivor lists. The trace follows the
first failing boundary through class proposals/selection, geometry/cell caps,
block top K, the global proxy pool, refinement and export. It supports plans
and saved-pool replay, explicitly limiting what each can establish. A streamed
counter measures proxy rank among **all block survivors**, even when the watched
cell has already been removed from the retained global pool.

Actual CUDA searches with `-budget 3 -maxcells 2e11 -umax 20000 -maxlines 1000
-aw 1.3 -class-explore 4 -t 2 -launch 30` diagnosed:

| Case | First loss / recovery |
|---|---|
| Seed 103, `-classes 2520 16` | Its proposed class `(2014,2408)` ranks 15th by size/root estimate; only 12 ranked slots remain after four exploration slots. |
| Seed 103, `-classes 2520 20` | Reference admitted and retained as exact-alpha finalist **3**, within the same cell cap. |
| Seed 10, `-classes 2520 16 -seg 20 -k 8` | Geometry admits the reference; it is lost at **block top K**, before reranking. |
| Seed 10, `-k 16` or `-seg 18 -k 8` | Still lost at block top K. |
| Seed 10, `-seg 16 -k 8` | Passes block top K, but has global proxy rank **17,719**, outside pools of 4,096 and 16,384. |
| Seed 10, `-seg 16 -k 8 -rerank 65536` | Reference retained as exact-alpha finalist **4**. |

Replaying that saved pool with `-rerank 32768` also retains the reference at
rank 4 and produces the same 200 exported polynomials byte for byte. This is
a replay result; a separate 32,768-pool sieve run was not performed.

Seed 10's cheap proxy uses alpha_s=-7.38354 and interpolated lognorm=65.81150;
the earlier alpha audit already measured the substantial error from omitted
higher prime powers and primes above 200. Wider sparse coverage exposes this
retention problem again. Merely increasing the final export count cannot recover
a cell removed inside a sieve block or before exact refinement.

These are diagnostic recovery experiments, not a production-policy change.
The GPU was shared with another workload. In these runs, seed 10's 65,536-cell
exact refinement took 18.90 s versus about 1.2 s for 4,096 cells; blindly widening
the pool has a real cost. Equal-time quality and test-sieve comparisons remain
necessary before choosing smaller blocks, wider pools, or a stronger proxy.

The full host suite and five bounded trace tests on CUDA passed. Trace-on/off
checks preserve polynomial and candidate-pool bytes for dense, explicit-class
and automatic-class searches; replay, invalid options and cutoff distinctions
are covered. Commands, frozen binaries, pools, logs and hashes are under
`stage23_gpu/runs/20261016_loss_trace/`; the repeatable experiment script is
`stage23_bench/runs/20261016_loss_trace/run.py`. All 1,200 exports from its six
GPU variant searches passed exact resultant checks and sampled brute-root checks.
Nine bounded CPU searches (three fixtures across dense/explicit/automatic modes)
also match the frozen pre-change binary's polynomial and pool bytes. Compact
measurements and trace logs are retained under
`stage23_bench/data/c208/ropt_coverage/trace_20261016/`.

**Next:** evaluate a cheaper way to preserve candidates with weak truncated
alpha, using these losses as controls and independent seeds to measure false
rejections. Compare class diversity/budget choices and retention policies at
equal elapsed time. The driver's 32-of-200 scoring cut still needs its own audit.

### 2026-10-16: intermediate alpha screening prototype

Added opt-in `s23_ropt -alpha-screen N B`: retain N proxy cells, recompute CADO
alpha through prime B on the host, rank by interpolated lognorm plus weighted
alpha with content divided out, then fully refine only
`max(-rerank,-refine,-top)`. The saved v1 pool remains the original proxy pool
before screening. Trace output distinguishes `alpha_screen_cut`; screening
time is reported separately and included in total refinement time. Defaults
and driver settings are unchanged. This does not change block selection.

The C208 seed-10 control used N=32768, B=500, rerank=4096 and smaller blocks
(`-seg 16`), with the earlier class/budget settings. The reference moves from
proxy rank 17,719 to screening rank 443 and finishes as exact-alpha finalist 4.
Three CPU-only replays, with alternating policy order and two threads, gave:

| Policy | Median refinement | Reference retained | Broad control top 32 / top 200 retained |
|---|---:|---|---:|
| Original 4096-cell pool | 1.38 s | No | 30 / 178 |
| Full refinement of 32768 | 10.75 s | Yes | 32 / 200 |
| Screen 32768 through 500, fully refine 4096 | 4.51 s | Yes | 32 / 187 |

The screen is 58% cheaper than full refinement of the wide pool in this replay,
but is slower than the original narrow pool. It adds a heuristic cutoff and
does not preserve every strong finalist. A single complete CUDA search confirms
the recovery on 2.73e10 selected cells: sieve 2.03 s, size model 1.03 s, refinement
4.37 s including 3.00 s screening. The GPU appeared idle before this bounded
run. These phases are not an equal-time comparison with the earlier shared-GPU
experiments. All 200 exports match CPU replay byte for byte, and the saved proxy
pool matches the first 32,768 cells of the original wide GPU pool.

Without retuning B, replayed all 16 C171 depth pools. Because those pools contain
4096 cells, this audit screens 4096 to 512, rather than running wider searches.
Every best exact-score cell survived, with 124/128 top-eight, 496/512 top-32 and
2923/3200 top-200 cells retained. Median refinement times were 1.205 s full and
0.530 s screened. These are alpha/size rankings, not MurphyE or test-sieve results.

Validation: full host suite, four focused tests on CPU and CUDA, nine default
searches with byte-identical pools/exports versus the frozen pre-screen binary,
and 8,400 exact-resultant checks across replay/live exports. The shared screen
score also matches the primitive written polynomial on 17 content cells; live
sampled root checks find zero differences. Reproduction script:
`stage23_bench/runs/20261016_alpha_screen/run.py`. Full records:
`stage23_gpu/runs/20261016_alpha_screen/`. Compact results:
`stage23_bench/data/c208/ropt_coverage/alpha_screen_20261016.json`.

**Next:** score the lost finalists by MurphyE/lattice quality, evaluate keeping
some original-proxy candidates alongside screened candidates, and compare at
equal total runtime on independent wider searches. GPU implementation of the
screen should follow a validated retention policy. The separate block-top-K
loss still requires smaller blocks or a better within-block policy.

### 2026-10-17: quality audit and exact-size screening

Scored the union of the broad/screened top-200 exports: 3,701 distinct
polynomials across C208 seed 10 and all 16 C171 depth seeds. Each skew was
searched at 16k MurphyE angles. Leading retained and discarded configurations
(168 total) were then frozen and validated at 256k angles and three fresh
lattice streams, each with 512 lattices and 4096 points per q. C171 uses the
user's supplied lpb 31/32, 14x13-bit region and provisional q=40M–150M; C208
uses archived lpb 33/34, I=17/J=32768 and q=80M–1G. No translations were
optimized during this audit, and the lattice model still omits lim/mfb/lambda
acceptance and processing cost.

The interpolated-size screen retains every checked lattice-model winner, but
loses C171 seed 1's best ordinary MurphyE cell `(0,293874)`. Its best retained
MurphyE is 1.2637% lower. This is a per-seed loss, not a measured whole-job yield
loss. The pool's interpolated lognorm for this cell is 51.1425966, versus exact
50.066483. That pessimistic size estimate is large enough to discard it despite
the improved alpha estimate.

Added opt-in `-alpha-screen-exact-size`, requiring `-alpha-screen`. Compute
the same local integer-translation size search used in full refinement before
the screen cut, with content removed consistently. Reuse the resulting size
and translation for survivors; recompute only alpha to the full bound. This
avoids a second size search that could change the selected configuration.
The cheaper interpolated mode and all previous defaults remain available.

With the same bound 500 and candidate counts, the corrected stage retains
all 200 broad-control C208 finalists. On C171 it retains 128/128 top-eight,
512/512 top-32 and 3114/3200 top-200 exact-score cells. Every previously
validated MurphyE and lattice winner survives in the identical integer
polynomial configuration (17/17 for each metric). These cases now serve as
regression controls for the correction.

For a fresh check, fixed the corrected policy before inspecting eight unused
depth pools: first four filenames each from C147 and C166. Screen 4096 through
prime 500 with exact size, fully refine 512, export 200; no retuning. Scored
1,616 unique finalists and independently validated 57 leading configurations
at the same higher accuracy. Both quality winners survive in all eight cases.
These smaller jobs use archived inherited model settings, not newly confirmed
sieve parameters. Preservation of these winners does not establish lossless
retention or guarantee gains after a later translation search.

After scoring and test processes finished, repeated the C208 replay timings
three times per policy in alternating order, on two CPU threads:

| Policy | Median refinement | Broad control top 200 retained |
|---|---:|---:|
| Original 4096-cell proxy pool | 1.20 s | 178 |
| Fully refine 32768 | 9.42 s | 200 |
| Interpolated-size screen 32768 to 4096, alpha 500 | 4.00 s | 187 |
| Exact-size screen 32768 to 4096, alpha 500 | 5.44 s | 200 |

The exact-size policy costs 42.3% less than full refinement of the broad pool
and reproduces its 200 exports byte for byte on this case. It costs more than
the original narrow pool, which misses the known reference. These are CPU
refinement timings, not end-to-end GPU or sieve throughput; unrelated system
activity was not controlled. The earlier replay timings taken alongside
quality scoring remain in the artifacts but are not used for this comparison.

Validation: full host suite; five focused tests on the CPU build and the
CUDA-built host path; a CUDA-executable pool replay matching CPU exports;
35 original-policy export files byte-identical to the pre-correction run.
The full-alpha exact-size screen also matches ordinary full refinement when
cutting 128 cells to eight, including a fixture with content. All new replay
exports and quality-audit inputs pass exact resultant checks. GPU kernels were
not changed or run during this follow-up.

Reproduction scripts are in `stage23_bench/runs/20261017_alpha_quality/`:
`run.py` for model scoring, `assess_exact.py` for winner preservation, and
`holdout.py` for fresh fixed-policy replay. The earlier replay script accepts
`--exact-size` and `--control-only` for repeated phase timings. Full artifacts:
`stage23_gpu/runs/20261017_alpha_quality/`. Compact results:
`stage23_bench/data/c208/ropt_coverage/alpha_quality_20261017.json`.

**Next:** use the exact-size screen as the correctness reference for GPU
implementation, checking primitive-content handling, translation identity and
prime-power/projective alpha before timing it. Preserve this corpus for
regressions and require broader equal-time search/yield checks before changing
defaults. Block-top-K losses and the driver's 32-of-200 cut remain separate work.

### 2026-10-18: corrected screen on CUDA

Implemented opt-in `-alpha-screen-gpu`, requiring `-alpha-screen-exact-size`
and a screen bound at most 2000. Shared fixed-width arithmetic performs exact
rotation, the same local translation size search, primitive-content division,
and affine/projective prime-power valuations. The alpha calculation uses
Hanrot's recurrence at every prime, avoiding the large discriminant needed
only for CADO's shortcuts. It is mathematically equivalent on the supported
square-free inputs; floating-point addition order can differ.

The device processes 2048 candidates per batch, with separate preparation/size
and per-prime alpha kernels. Overflow, recursion-depth exhaustion or search
exhaustion requests CPU fallback, counted in the log. A 1e-8 band around the
screen cut is rescored on the host. This is an empirical numerical guard,
not a universal error bound. Full-alpha refinement still runs on the host
from each survivor's original saved translation; device translations are not
used for export. `-alpha-screen-check` compares every successful device score
with the independent CPU/CADO reference and fails on key error above 1e-8.
It is intentionally excluded from timing runs. Defaults, pool v1 format and
pipeline forwarding remain unchanged.

The same 25 retention cases (C208 seed 10, 16 C171 pools and eight C147/C166
holdouts) reproduce all 5000 previously validated exports byte for byte.
All 131072 screened candidates were audited. Worst discrepancies were zero
for lognorm, 5.33e-15 for alpha and 1.42e-14 for the combined key. Two device
translations differed at indistinguishable norms before host refinement;
the final pass preserves the exact validated configurations. No corpus case
needed fallback. This preserves the previously checked MurphyE and lattice
winners by identity; no new quality-model scoring or test sieving is claimed.

Three alternating replays per policy on the RTX 5070, with two host workers:

| Case/policy | Screen | Screen + final refinement | Process wall |
|---|---:|---:|---:|
| C208, CPU screen 32768 to 4096 | 5.151 s | 6.22 s | 6.343 s |
| C208, GPU screen 32768 to 4096 | 0.847 s | 2.20 s | 2.652 s |
| C208, full CPU refinement of 32768 | — | 10.96 s | 11.076 s |
| C171 seed 1, CPU screen 4096 to 512 | 0.552 s | 0.68 s | 0.704 s |
| C171 seed 1, GPU screen 4096 to 512 | 0.175 s | 0.37 s | 0.760 s |
| C171 seed 1, full CPU refinement of 4096 | — | 1.27 s | 1.288 s |

These are medians of saved-pool replays; each GPU process creates its own
context. C208 gains 2.83x in the refinement phase versus CPU screening and
4.98x versus broad full refinement. C171's phase is faster, but its short
process loses that gain to context startup. These are not end-to-end search
or relation-throughput measurements, and the machine is not an isolated lab
host. Broad C171 refinement retains additional lower-ranked candidates; the
GPU comparison preserves the corrected screen's policy, not every broad
finalist.

Validation includes the full host suite, 1350 shared-arithmetic/CADO comparisons
across nine fixtures and bounds 2/3/31/200/500/2000, real CUDA screening tests
(including a square-free polynomial whose deep root forces CPU fallback),
and live GPU sieve/screen/replay export agreement. The optional GPU tests are
`S23_ROPT_BIN=stage23_gpu/build/s23_ropt python3 -m unittest discover
-s stage23_gpu/tests -p 'test_alpha_screen*.py'` from the repository root.
Reproduction: `stage23_bench/runs/20261018_gpu_screen/run.py`; full artifacts
under `stage23_gpu/runs/20261018_gpu_screen/`, compact results in
`stage23_bench/data/c208/ropt_coverage/gpu_screen_20261018.json`.

**Next:** amortize GPU context/allocation costs across seeds, then migrate the
remaining full-alpha refinement and final scoring with the same export/quality
checks. Add explicit driver forwarding before pipeline use. Require equal-time
search and test-sieve comparisons before changing defaults; block-top-K loss
and the driver's 32-of-200 cut still require separate work.


### 2026-10-18 review corrections to GPU screening

The cutoff guard is now 0.125 rather than 1e-8 to cover local-search path
differences, but remains empirical. Unchecked GPU selection is not guaranteed
identical for every input. Check mode uses all host-reference keys and
translations for its cut and export, while still rejecting discrepancies.

Screen stack reservation is 32 KiB, before the sieve. CUDA setup and buffer
failures fall back to the CPU screen without changing the saved starts;
non-finite cells rank last instead of aborting valid exports. Device prime
capacity and bound checks share constants. Preparation batches scale with SM
count. Host fallbacks/rechecks and full-bound alpha are reused, and screening
is skipped when it would cut nothing (except explicit audits). The cold size
search, projective polynomial construction, prime predicate and trace/retention
proxy construction now share implementations. The preceding benchmark numbers
are historical; these review changes have not been retimed on an idle GPU.

Validation after these fixes: the full `make test` host suite passes, as do
six CUDA regression tests (including invalid-device CPU fallback and two
concurrent workers). Both checked and unchecked audits of all 25 saved cases
retain all 5,000 reference exports byte for byte. Checked maximum key error
is 1.42e-14. Compact evidence is in
`stage23_bench/data/c208/ropt_coverage/gpu_screen_review_20261018.json`.
Unaudited GPU finalists still need host size canonicalization; preparation
now uses larger batches but does not yet overlap host work and transfers.

### 2026-10-18: persistent multi-seed execution

Added `s23_ropt -batch FILE`, advertised as `ropt-batch-v1` in CPU and CUDA
builds. A versioned manifest contains complete, tab-separated argv rows for
independent searches/replays. One process runs them serially. CUDA retains its
context and screening workspace; scratch grows only when needed, and prime
tables upload only when the bound changes. Seed coefficients upload on each
job. Device switches release scratch on its owning device, and a failed setup
still permits CPU fallback followed by a later valid GPU job.

Each job resets the host model switches and sampling RNG to fresh-process
defaults. Polynomial storage has per-job lifetime, including error paths.
Preflight validates all options and rejects duplicate outputs or output/input
aliases (relative paths, symlinks and existing hard links) before execution.
Paths are relative to the working directory; no shell parsing or expansion.
Runtime errors stop the batch while preserving earlier completed exports.
Partial MurphyE status 3 continues subsequent jobs and propagates to the batch.
The interface is standalone; production-driver forwarding/checkpoint integration
remains a separate step. Root-sieve scratch still allocates per search; this
change retains the screen scratch and device context.

Both checked and unchecked single-session audits of the fixed 25-case corpus
reproduce all 5000 reference exports byte for byte, with exact resultant
validation. The checked audit covers all 131072 candidates; largest key error
remains 1.42e-14, with two device translations corrected by the existing host
path. Each audit reports one device setup, one screen allocation and one prime
upload across all 25 seeds. Search/ranking policies and the review's 0.125
cutoff guard are unchanged.

Three alternating repetitions of all 16 C171 depth pools on the idle RTX 5070,
using two CPU threads, bound 500, 4096-to-512 screening and 200 exports per seed:

| Execution mode | Median process wall, all 16 seeds | Refinement total | Context wait total |
|---|---:|---:|---:|
| CPU, separate processes | 10.981 s | 10.71 s | — |
| CPU, one batch process | 11.174 s | 10.92 s | — |
| GPU screen, separate processes | 12.203 s | 5.02 s | 5.40 s |
| GPU screen, one batch process | 5.165 s | 4.54 s | 0.30 s |

This is 2.36x faster than separate GPU processes and 2.13x faster than separate
CPU screening processes for the same work and outputs. Every timed repetition
reproduces all 3200 reference exports. GPU batches took 5.161–5.176 s; these
are shared-desktop measurements, with more variation in CPU timings. No CPU
build/test work or other GPU compute process was running when timing began.
The earlier `/bench` job had finished before CUDA validation and timing.
These are serial saved-pool comparisons, not complete searches, a comparison
against the driver's two-worker scheduling, or relation-throughput results.

Validation: full host suite; six batch tests on the CUDA executable plus twelve
existing screening regressions; final CPU batch checks (four pass, two CUDA-only
skips). Tests cover live GPU search versus fresh processes, changed per-job
options, scratch growth and reuse, prime-bound changes, invalid-device recovery,
malformed manifests, file aliases and failure after a completed job.
Reproduction: `stage23_bench/runs/20261018_ropt_batch/run.py`, with `--check`
for the CPU oracle or `--timings --repeats 3` for the four execution modes.
Full artifacts: `stage23_gpu/runs/20261018_ropt_batch/`; compact record:
`stage23_bench/data/c208/ropt_coverage/ropt_batch_20261018.json`.

**Next:** connect persistent execution to the resumable driver while preserving
per-seed checkpoints and worker limits. Remaining numerical migration is full
alpha/final scoring and CPU size canonicalization; boundary rechecks and those
host passes still account for much of the 4.54-second batched refinement phase.
Equal-time search and sieve-yield validation remain necessary before changing
search defaults.

### 2026-10-19: C192 coverage regression and driver batching

The new C192's original CADO effort-5 winner prints MurphyE 1.864e-14.
Effort 10 returns only 1.665e-14 for the identical seed, while its overall
original-side best is 1.757e-14. Both logs identify build 0574bc39d. Four
fresh isolated original/inverted runs reproduce the stored effort-5/10
results. This demonstrates non-monotonic search quality as effort changes;
it does not identify the precise internal CADO cutoff responsible.

The winning seed is original index 11 and is unchanged in the deep input.
Its rotation is `(u,v)=(-34,-5754737141)`. Dense GPU breadth loses it during
size-budget tightening to meet the 2e11 cell cap. Dense depth (6e12 cap,
4.58e12 cells visited) retains it at proxy rank 12 and exact-refinement rank 6.
The exported representative differs from CADO by translation -7 only.
Automatic 16-class selection modulo 2520 misses its class at proposal time.
The known cell was traced, not inserted. All 600 exports validate and the
three live searches pass their brute-force root checks.

The recovered depth pool also passes a new CPU/GPU exact-size screening check:
4096 candidates to 512, bound 500. The winner remains rank 6; all 200 exports
match the CPU screen byte for byte. Maximum alpha/key discrepancies are
3.55e-15/7.11e-15, with zero translation differences, fallbacks or boundary
rechecks on this fixture. Final canonicalization/refinement still uses CPU.

Independent scoring yields standard MurphyE 1.864455e-14 for both CADO and
the GPU-recovered representative. Their lattice scores also match at printed
precision for provisional LPB 32/33, 15e and q=80M–500M. Factor-base limits
160M/240M are recorded for future test sieving. This is a focused seed
experiment, separate from the independently launched full GPU pipeline over
1,642,161 distinct inputs. The full run's comparison is automatically archived
when completed. Fixtures, hashes, measurements and reproduction instructions
are in `stage23_bench/data/c192/ropt_coverage/`; full artifacts are under
`stage23_gpu/runs/20261019_c192/`.

The resumable Python driver now supports opt-in `--ropt-batch-size 16`, or
`[gpu] ropt_batch_size=16`, with default 1. Pending searches are grouped into
persistent processes within the existing one/two-worker limit. Each flushed
completion gets an individually validated export, pool and checkpoint before
later jobs finish. Failures and interruption preserve completed checkpoints.
Resume repairs only pending/corrupt outputs; batch size is execution-only and
can change without invalidating results. Every attempt retains a distinct
manifest/process log; per-seed logs preserve partial MurphyE reporting.
Unsupported frozen binaries reject the opt-in. Search policy is unchanged;
screening remains off by default.

Validation covers all 29 driver regressions, including batch parity, resume,
corruption, failure after a completed seed, interruption while the next seed
runs, valid/invalid partial MurphyE, missing status markers and old binaries.
Three real C192 CUDA searches reproduce separate-process exports and pools
byte for byte, including resume and one repaired output. The full host suite
also passes. The live driver check shares the GPU with the full run and makes
no throughput claim.

The full independent C192 run is now complete: 1,642,161 inputs, 314 breadth
searches, 16 depth searches, four Murphy-refinement families, no partial
Murphy failures. It independently selects the known winning family (`s0012`)
and refines the same rotation to a translation 31,970,404 units from the
original CADO representative. Separate standard-Murphy scoring improves from
1.864455e-14 to 1.868093e-14 (0.195%). The provisional lattice score improves
0.204%; three additional streams with 512 lattices and 8192 samples each
give gains of 0.196%, 0.191%, and 0.208%. These are model improvements, not
measured sieve throughput. `comparison.json` and `winner_validation.json`
preserve the full comparison and additional checks.
The 1,024,000-angle standard-Murphy check gives 1.868128e-14 versus
1.864518e-14, retaining a 0.194% advantage with frozen skews.

Driver screening is now exposed via `--alpha-screen N,B`,
`--alpha-screen-exact-size`, `--alpha-screen-gpu`, and `--alpha-screen-check`,
with matching configuration keys. It applies only to breadth/depth. Their
pools are saved before screening, and final Murphy refinement raises its
candidate count to N so every saved start remains eligible. Screening options
are semantic settings; legacy runs with screening off keep valid checkpoints.
Unsupported frozen binaries reject requested screen capabilities.

All 32 driver regressions pass. On all sixteen completed C192 depth pools,
both checked and unchecked batched GPU screening reproduce the CPU screen's
3200 exports exactly. All 512 original top-32 exports survive; 3068 of the
3200 total original exports survive. This is not a full screened-pipeline
quality guarantee. Defaults remain unchanged. Remaining work: migrate full
alpha/final scoring and improve coverage while retaining prior winners.

The follow-up driver review is addressed: small batches split across workers,
feature and semantic checks precede saving execution settings, missing legacy
screen settings retain their historical defaults, and blank screen Booleans
mean false. Legacy pool capability caches remain authoritative for ordinary
resume; failed probes do not overwrite them or persist an unsupported result.
Screen option defaults, dependencies and capabilities share one registry.

Batch completion now uses a separate inherited event descriptor with byte
offsets into the shared process log. Per-seed logs preserve arbitrary bytes,
CRLF and unterminated lines; marker-like diagnostics cannot create checkpoints.
Progress updates per completed/reused seed, including within a running batch.
Driver batching requires `ropt-batch-events-v1`; older frozen binaries resume
with `--ropt-batch-size 1`, and new batched experiments use a fresh run directory
with rebuilt binaries. `validate_winners.py --run-dir RUN` keeps reproduction
outputs within the selected study.

Review validation: CPU and CUDA builds pass, all 40 driver regressions and the
full host suite pass, and three real C192 seeds give byte-identical standalone
and batched CPU exports/pools with two workers. Resume and repair of one corrupt
export also pass. Batch transport tests cover opaque diagnostic bytes, failure
boundaries and descriptor validation.

The GPU follow-up also passes: all eight batch tests and six GPU screening
tests run without skips. Fresh C192 artifacts reproduce standalone/batched
live exports and pools, resume, and one corrupt-output repair. All sixteen
saved depth pools run as two batches of eight per mode; checked and unchecked
GPU screening each reproduce the CPU screen's 3200 exports byte for byte,
including all 512 baseline top-32 exports. Every GPU batch uses one device
setup, one buffer allocation and one prime upload; no screen falls back to CPU.
The completed source run's inputs remain unchanged. This is correctness
validation, not a new throughput result. Reproduce with
`stage23_bench/runs/20261019_c192/review_gpu.py --run-dir NEW_DIRECTORY`.
Compact evidence: `stage23_bench/data/c192/ropt_coverage/review_gpu_20261019.json`;
full artifacts: `stage23_gpu/runs/20261019_c192_review_gpu/`.
