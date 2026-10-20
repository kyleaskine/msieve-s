"""Summarize leading-coefficient coverage; no new searches or timing estimates.

Run from any directory. Writes coeff_audit.json beside this script.
The random fixture rows exclude the deliberately saved top group. Enrichment
is a ratio of group shares, not a success probability or a rate per GPU-hour.
"""
import collections
import csv
import gzip
import json
from pathlib import Path


def factor(n):
    factors = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[str(p)] = factors.get(str(p), 0) + 1
            n //= p
        p += 1
    if n > 1:
        factors[str(n)] = factors.get(str(n), 0) + 1
    return factors


here = Path(__file__).resolve().parent
root = here.parents[2]
report = {}
for job in ("c146", "c161"):
    with gzip.open(root / "stage23_bench/data" / job / "sopt_sample.tsv.gz", "rt") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    counts = {s: collections.Counter(int(r["raw_c5"]) for r in rows if r["set"] == s)
              for s in ("random", "top")}
    totals = {s: sum(c.values()) for s, c in counts.items()}
    coefficients = []
    for a in sorted(set(counts["random"]) | set(counts["top"])):
        random_share = counts["random"][a] / totals["random"]
        top_share = counts["top"][a] / totals["top"]
        coefficients.append({
            "raw_c5": a, "factorization": factor(a),
            "random_count": counts["random"][a], "top_count": counts["top"][a],
            "random_share": random_share, "top_share": top_share,
            "top_share_over_random_share": top_share / random_share if random_share else None,
        })
    report[job] = {"group_sizes": totals, "coefficients": coefficients}
output = json.dumps(report, indent=2) + "\n"
(here / "coeff_audit.json").write_text(output)
print(output)
