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
checks. A production-size replay of that driver is still needed before claiming
its end-to-end speed or search quality. Recent component runs, made by the earlier
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
