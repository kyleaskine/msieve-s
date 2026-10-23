# AS660 stage-1 archive and compact research data

This directory records the user-authorized archival work of 2026-10-23.
Completion and exact byte totals are recorded in `manifest.json`; the compact
research index is `research-index.json`. No GPU work or polynomial production
outputs are part of this cleanup.

**Completed:** the five originals (47.477 GiB) are now five verified archives
(15.527 GiB); 27.204 GiB of redundant extracts were also removed. After allowing
for the new 149.45 MiB compact candidate files, the net saving is **59.008 GiB**
(small reports/manifests excluded). `completion.json` confirms all ten old paths
resolve correctly through the archive reader after removal. The real reader
also reproduced an entire archive and a noncontiguous extract with matching
SHA-256 hashes; 16 regression tests passed.

All eight populations are independently collision-audited: 1,755,106 raw outputs,
1,687,829 distinct identities, and 1,141,047,425 searched q-root occurrences.
The new 180180/360360 audits found no saturation, hash caps, or unmatched raw
outputs. Compact candidate IDs and all exposure totals were checked on readback.

Use fresh provenance prefixes for future searches; retired original filenames
continue to identify these archived runs.

## Retained evidence

- Five complete original sidecars are preserved losslessly as
  `/home/kylea/msieve-s/as660-stage1*.w0.jsonl.zst`.
- Each compressed archive was fully decompressed and checked against the
  original SHA-256 and byte count. `*.verified.json` records both compressed
  and uncompressed hashes and the original run header.
- The five redundant extracted sidecars are represented by their original
  source byte ranges and SHA-256 hashes. Their bytes were checked against both
  the existing extraction and the decompressed source before removal. Original
  extraction manifests remain in the earlier run directories.
- `AD.candidates.jsonl.zst` retains each distinct candidate's ID, ad/P/m,
  raw-output count, and every discovery's run, worker, coefficient, batch, hit,
  q, admission decision, and gate ratio. Repeated N is held in the research index.
  These files were independently read back and their totals verified.
- `research-index.json` contains each coefficient's bounds, setup, ending,
  compact-candidate artifact hash, and searched q-root exposure by log2 bin and
  progression count. It includes unsuccessful work. q=1 has its own bin;
  subsequent bins cover `[2^b, 2^(b+1)-1]`.
- The first six compact populations come from previously audited ledgers.
  `180180.audit.json` and `360360.audit.json` record fresh full independent
  collision reconstruction, batch-root membership checks, and raw-output joins.
  Their source files contain 167,046 and 350,615 returned collisions respectively.

The compact records are sufficient to join quality results by candidate ID,
compare q magnitude and discovery order, and count unsuccessful exposure. They
do not assign mixed-q batch time to individual successful candidates. Original
per-root/per-batch detail, qroot, offset, and p1/p2 remain recoverable from the
lossless archives when another hypothesis needs them.

## Reading and restoring

The main auditor accepts `.zst` directly and resolves the old original paths
to their `.zst` siblings. Registered extracted paths resolve through
`stage23_bench/sidecar_archives.json` without recreating large files. For example,
from the repository root:

```sh
python3 stage23_bench/tools/stage1_provenance.py as660-stage1-d.w0.jsonl --require-complete
```

That command reruns the full collision audit, so normally use the saved audit
JSON instead. Existing historical capture scripts that use raw `open()` or
seek directly are snapshots of the original experiment, not archive-aware
entry points. Use `sidecar_io` for new analysis or restore a file when needed:

```sh
python3 stage23_bench/tools/sidecar_io.py \
  stage23_bench/runs/20261021_as660_two_coefficients/second.w0.jsonl \
  --output /tmp/restored-second.w0.jsonl
```

The destination must be new and different from the source alias. Omitting
`--output` computes only the restored byte count and SHA-256. Extract reading
decompresses the containing archive sequentially and checks decompressor status,
so it can take time even for a limited byte range. For Python use
`sidecar_io.open_sidecar(path)` for JSONL or `sidecar_io.chunks(path)` for bytes.

Never treat the compact candidate file as a replacement for unsuccessful-work
exposure. Candidate discoveries can span several q bins; preserve that ambiguity
or choose an explicit attribution rule when deduplicating.

## Reproduction and safeguards

`prepare.py` performs the two remaining audits, creates and reads back all eight
compact populations, and extracts log2 exposure from the earlier originals.
`archive.py` compresses and verifies originals and all reconstruction ranges;
it does not remove sources in normal mode. `finalize.py` cross-checks the compact
index totals. Only `archive.py --remove-verified` removes the ten manifest-listed
uncompressed files. It requires complete preparation, verifies all compact and
compressed artifact hashes, and rejects changed original/extract file identities.
The manifest journals each removal and records net bytes reclaimed.

The main auditor's tests cover compressed/plain parity, truncated compression,
virtual extracts, and logarithmic exposure. Cleanup tests cover changed-source
and changed-archive rejection and resumable removal. Historical audit reports,
raw polynomial outputs, CPU/GPU optimization inputs/results, and existing full
candidate ledgers are retained. Nothing is recomputed on the GPU.
