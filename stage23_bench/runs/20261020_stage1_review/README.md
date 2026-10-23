> Historical run record moved from the main notes on 2026-10-22. See the [current conclusions](../../../POLYSELECT_OPTIMIZATION_NOTES.md) for subsequent evidence and revised priorities. Statements about a running or stopped process describe the time of this record.

## Provenance code-review follow-up (2026-10-20)

Checked Claude's 15 findings against the implementation before restarting AS660.
The production search remains stopped. The original coefficient corpus and frozen
pilot binary/PTX are unchanged. Review artifacts, including the clock probe, are
in `stage23_bench/runs/20261020_stage1_review/`.

Disposition, in the review's order:

1. **Fixed:** reject commas and equals signs in provenance prefixes. Validation
   also runs in `find_poly_core` before the legacy leading-coefficient range
   parser, not merely when GPU logging starts. Both supplied bad-prefix forms
   were exercised against the executable and rejected before a coefficient began.
2. **Fixed:** schema 2 emits a launch record only when that group's shape changes
   within a coefficient. Audit write errors request batch-boundary stops, prevent
   queued coefficients from starting, close the failed stream, and report a
   nonzero process status only after worker shutdown and queued polynomial work
   has drained. An explicit stage-2 drain was necessary: `threadpool_free` alone
   discards queued tasks. This cannot make a genuinely full filesystem writable;
   it prevents immediate worker-thread `exit()` from abandoning queued callbacks.
3. **Fixed:** both the CLI and Python `audit()` stream by default. Small-fixture
   exact q/root-repeat analysis requires `--exact-exposure` or `streaming=False`.
   Candidate identities remain in memory for joins; this is bounded batch/root
   storage, not constant memory regardless of candidate count.
4. **Fixed:** clear `gpu_context` immediately after destruction. A logging close
   failure follows the graceful error path rather than exiting its worker.
5. **Fixed:** record resolved PTX path, byte count, and SHA-256. The audited path
   loads the same in-memory bytes it hashed via `cuModuleLoadData`, avoiding a
   hash/load file-replacement race. SHA-256 has empty, `abc`, multi-block, and
   million-byte reference-vector tests; runtime PTX hashes also match Python.
6. **Fixed:** ignore worker JSONL sidecars, generated large candidate ledgers,
   `.ncu-repz` reports, and the local AS660 log in Git. Original files remain local.
7. **Fixed:** start `search_seconds` after pre-batch record formatting. Its scope
   is still the dispatch/engine call, including any launch/engine instrumentation
   inside that call; it is not pure GPU busy time. New `audit_prepare_seconds`
   accounts for the pre-batch record separately.
8. **Confirmed clock disagreement; qualified interpretation:** a 1,500-sample
   host probe measured 30.480670 s MONOTONIC, 29.760321 s MONOTONIC_RAW, and
   29.349207 s REALTIME. MONOTONIC advanced 2.42% more than RAW. One REALTIME
   sample went backward by 1.111 s while both monotonic clocks advanced.
   Linux audit durations now use MONOTONIC_RAW; Windows retains QPC and other
   POSIX systems fall back to MONOTONIC. Schema 2 records the clock name and a
   separate coefficient wall duration. This does not calibrate either host clock
   against CUDA events. A constant rate factor cancels from same-clock on/off
   ratios, but changing rates and run noise still limit small-gain claims.
9. **Fixed:** one post-readback timestamp defines both that component and total
   batch duration. Generation + preparation + search + readback/submission add
   to `seconds`, within JSON rounding. Writing the batch-end record, periodic
   flushing, and loop housekeeping remain outside that total.
10. **Fixed excessive flushing; reject the stated rationale:** flush every 64
    batches and at coefficient/worker boundaries, checking buffered-write errors
    every batch. `exit()` flushing does not cover crashes, SIGKILL, or power loss.
    The unflushed tail can be lost; missing completion markers and partial JSON
    are explicitly incomplete. No per-batch durability guarantee is claimed.
11. **Fixed:** fixed progressions choose the capacity for the actual key width.
    With p_max=12544 and 28-bit keys, default and forced-one-progression searches
    both used 10,010 q roots per full batch (23 batches / 222,411 q roots total).
    Their final adaptive offsets legitimately differ, so whole raw sets need not
    match. CPU cases also exercise progression-driven transitions to 64-bit keys.
12. **Fixed:** empty q intersections return before either factor-base setup, while
    nonempty windows still initialize factors from the original natural bound.
13. **Fixed:** a malformed unterminated final JSON record is reported as an
    incomplete tail, retaining earlier candidates. Malformed interior or
    newline-terminated records remain hard errors. `--require-complete` rejects
    the incomplete corpus.
14. **Fixed:** reserve the ledger with exclusive creation before reading inputs;
    an existing path fails immediately and remains intact.
15. **Fixed:** normalize roots and build a set once per batch for hit membership.

Validation includes the rebuilt CUDA=120 executable, 11 Python tests, strict C
helper tests, the existing 11-scenario GPU integration suite, schema-1 pilot
compatibility (all 20 polynomials joined), and a 64 KiB file-limit injection.
The injection returned status 1, preserved both found raw polynomials, printed
the completed-drain diagnostic, and left a sidecar the auditor classified as
truncated/incomplete. The initial synthetic small-input test needed explicit norm
bounds and an in-range q window; those were harness setup issues, preserved in
the temporary run directories.

The short pre-RAW-clock comparison produced the same 86-polynomial multiset in
each of three C196 pairs. Pooled logging overhead was 0.98%, with individual
pairs -0.80%, +3.28%, and +0.48%. This is encouraging but does not establish the
<=1% goal. Two pairs also crossed a REALTIME correction. Final-build results are
recorded alongside the earlier measurements; do not mix their duration clocks.

The final RAW-clock build passed the same review regressions. Its three timing
pairs again returned identical 86-polynomial multisets, with pooled overhead
1.27% and paired overheads +1.79%, -0.014%, and +2.07%. These short runs still do
not establish <=1% overhead or a precise improvement over the original logger.
Each C196 window wrote six launch records for 1,134 batches, rather than three
per batch. The final executable/PTX hashes and test outputs are in
`final-summary.json` and `final-regressions/`. All GPU test processes have exited;
the previously supplied production restart command needs no new parameters.
