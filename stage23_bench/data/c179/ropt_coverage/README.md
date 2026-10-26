# C179: winner exported only by msieve original

The completed October 25 CPU job factors
`57620329943072851234467695661532076333504874396552000279206114783643367173277603126879399724731783963048827698518286205989191152664289350141501384171806521091095958947776845847733`.
Its reported winner is from msieve original, with printed MurphyE 1.217e-13.

The winning family is seed **9** (zero-based index 8), present identically in
both tools' inputs. Its inverted input differs only by negating c0..c5. Thus
this case is not explained by the 300-versus-150 seed shortlist sizes. The
winning linear rotation, in original-seed coordinates before translation, is
`u=-55, v=34188178`, with the reported representative translated by `t=14690493`.

There are five representatives of that rotation among 954 msieve-original
exports for the seed. None occurs among its 982 msieve-inverted exports or
either CADO export. All these family outputs have the same leading-coefficient
magnitude, so a content reduction is not hiding the rotation comparison.
CADO original and msieve inverted instead select `u=-109, v=-850387982` for
this family, at different translations. CADO inverted selects
`u=-55, v=461195878`. These are observed search-output differences; the internal
CPU-tool rejection point was not traced.

The lead survives one common scorer and skew search:

| Method | Standard MurphyE, 1,024,000 angles | msieve-original advantage |
| --- | ---: | ---: |
| msieve original | 1.217340e-13 | — |
| msieve inverted | 1.193176e-13 | 2.025% |
| CADO original | 1.198309e-13 | 1.588% |
| CADO inverted | 1.198309e-13 | 1.588% |

The scoring shortlist covers every CADO output, each msieve orientation's top
20 printed scores, the top ten from the winning family, and every exported
representative of the target rotation. After within-method deduplication it has
314 records and 196 unique configurations. All selected resultants verify.
Skew search uses 16k angles with 64k evaluation; the top five per method and all
target representatives are checked at 256k and 1,024,000 angles with skew fixed.
This is not an exhaustive rescore of all 120,857 msieve exports. No lattice
model or actual sieve-yield comparison was made for this new job.

CPU plans using our frozen root optimizer show:

| Search | First boundary for the known rotation |
| --- | --- |
| Dense breadth, cap 2e11 | Lost when the size budget is tightened from 3 to 1.75 |
| Dense depth, cap 6e12 | Admitted, with budget 2.75 and about 4.57e12 cells |
| 16 classes modulo 2520 | Never proposed |
| 32 classes modulo 2520 | Never proposed |

The independent proposal probe ranks the class 1290th on conditional roots
alone and seventh within its u residue. Even the 64-class proposal pool omits
it. This strengthens the case for considering size before the proposal cut,
but also shows that keeping only one root-favored v class per u would be
insufficient. A coverage/exploration allowance remains necessary.

These are planning observations, not a live GPU recovery: depth admission does
not prove survival of block, proxy, refinement or output cuts. The exact seed
and winner are preserved here for that follow-up. Production code and defaults
are unchanged.

`prepared.json` records original source hashes and family counts;
`summary.json` contains independent scores, finalist coefficients and trace
commands; `proposal_audit.json` records the separate rank probe. `shortlist.json`
and `screen.json` preserve the compact comparison inputs and scores.
`manifest.json` hashes these evidence files. Full logs and frozen executables
are under `stage23_gpu/runs/20261025_c179_glance/`.

Reproduce against the saved CPU job with a fresh destination:

```
python3 stage23_bench/runs/20261025_c179/inspect.py \
  --source-dir CPU_RESULTS_DIRECTORY --run-dir NEW_DIRECTORY
```

The separate proposal probe is built by
`stage23_bench/runs/20261025_root_coverage/proposals.py`; its standalone invocation
and binary hash are recorded in `proposal_audit.json`.
