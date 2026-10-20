# C192 effort and coverage regression, 2026-10-19

The October 19 CPU job uses
`n=970137671032597847775481145969913289969436931189728453724304265243876304116067045536862610616610038007662192714697454871633967746240559613115499546579676197722961272656328233636785191370677953`.
Its original CADO effort-5 pass printed MurphyE 1.864e-14. The effort-10
original pass printed 1.757e-14 overall, and only 1.665e-14 for the winning
seed. Both logs identify CADO build 0574bc39d. Fresh, isolated runs reproduce
all four results for that seed: original 1.864e-14/1.665e-14 and inverted
1.698e-14/1.724e-14 at effort 5/10. More effort is not a nested continuation.

The seed is index 11 (zero-based) in the original 150 inputs and is present
unchanged in the deep input. Its winning rotation in original seed coordinates
is `(u,v)=(-34,-5754737141)`, followed by translation `t=113836240`.

| GPU search on that seed | Fate of the known rotation |
|---|---|
| Dense breadth, cell cap 2e11 | Excluded when the size budget is tightened to meet the cap |
| Dense depth, cap 6e12 | Retained: proxy rank 12/4096, full-refinement rank 6/200 |
| Automatic 16 classes modulo 2520, cap 2e11 | Its class is absent from proposals |

The depth run searches 4.58e12 cells and exports the same rotation, translated
seven units from CADO's representative. It was traced, never injected.
All 600 exports across the three searches have valid exact resultants, and
each search passes its 128-cell brute-force root check. The saved timings
are from a shared machine and are not performance comparisons.

The recovered pool also passes exact-size GPU alpha screening from 4096 to
512 candidates at bound 500. The winner stays rank 6; all 200 outputs match
the CPU screen byte for byte. Maximum alpha/key differences are
3.55e-15/7.11e-15, with no translation differences or host fallbacks.

Independent validation gives standard MurphyE **1.864455e-14** for both
representatives. They also have equal lattice scores to the scorer's printed
precision with the user's provisional LPB 32/33, 15e rectangle and 80M–500M
special-q range. Factor-base limits 160M/240M are recorded for future actual
sieving; they are not inputs to this smoothness model. Screening uses 4000
optimization angles/16000 evaluation angles and 32 lattices; final checks
use 256000 angles, 256 lattices and 4096 samples on an independent stream.
Each objective selects its own skew before validation. No test sieving.

`seed.poly`, `seed_inv.poly`, `cado_winner.poly`, and `gpu_recovered.poly` are
exact fixtures. `measurements.json` records commands, input/binary hashes,
traces, resultant checks, all validated target finalists, and the driver
batch correctness check. Full logs, pools and frozen CPU references live in
`stage23_gpu/runs/20261019_c192/`.

The independent full GPU pipeline completed all 1,642,161 distinct raw inputs,
314 breadth searches, 16 depth searches, and four Murphy-refinement families,
with no partial Murphy failures. It independently selected the same winning
family, `s0012`. Final Murphy refinement keeps the same rotation and changes
translation by -31,970,404 from the original CADO representative.

At 256000 angles, standard MurphyE is 1.868093e-14 for GPU versus
1.864455e-14 for CADO (0.195% higher). An additional 1,024,000-angle check
gives 1.868128e-14 versus 1.864518e-14 (0.194% higher). The provisional
lattice model improves 0.204%; three further streams, each with 512 lattices
and 8192 points, give gains of 0.196%, 0.191%, and 0.208%. These small model
gains have not been measured by actual sieving. `comparison.json` and
`winner_validation.json` contain the evidence; `gpu_best.poly` and
`cado_best.poly` use the independently selected lattice skews.

The driver screen audit (`driver_screen_check.json`) also covers all sixteen
completed depth pools. CPU, checked GPU and unchecked GPU screens produce
identical 3200 exports. All 512 original top-32 exports survive; 132 of the
lower-ranked baseline exports are replaced. This does not establish quality
equivalence for a complete screened pipeline. Final Murphy refinement in the
driver always replays all saved candidates without the alpha cut.

Reproduction scripts: `stage23_bench/runs/20261019_c192/`. `run.py trace`
creates CPU plans; add `--gpu` for the three searches. `run.py cado` repeats
the four isolated CPU runs. `compare.py --target` scores the focused checks.
`run.py pipeline` runs the complete GPU path and `run.py finish` waits for it,
compares all frozen methods, then updates this archive with `save.py`.
`run.py driver-check` compares separate/batched live searches, checkpoint
reuse and corrupted-output repair. `run.py driver-screen-check` audits the
sixteen depth pools. `validate_winners.py` runs the additional model checks.

The batch-driver review was subsequently checked on the available GPU with
fresh artifacts. All 14 GPU batch/screening regressions pass without skips.
The driver replays the sixteen depth pools as two batches of eight: checked
and unchecked GPU modes each match all 3200 CPU-screen exports exactly, with
no GPU screen falling back to CPU. Live-search batch parity, resume and corrupt
output repair also pass. `review_gpu_20261019.json` records the current binary
hashes, input hashes and results. Reproduce using
`review_gpu.py --run-dir NEW_DIRECTORY`; it preserves the original study.
