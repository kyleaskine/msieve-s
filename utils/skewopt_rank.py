#!/usr/bin/env python3
"""Re-score polynomials onto one common MurphyE scale and rank them.

CADO reports MurphyE at its own chosen skew; cownoise/skewopt re-optimizes the
skew and reports MurphyE there, which is the trustworthy, community-standard
number -- and it often differs noticeably from CADO's. msieve's own 'e' already
matches cownoise, so msieve outputs are taken at face value (no skewopt run).

Give it CADO ropt output files (.txt -> rescored through skewopt) and/or msieve
ropt output files (.p -> 'e' used directly). It prints one leaderboard on the
common scale, with CADO's original number alongside for comparison.

  skewopt_rank.py [--skewopt PATH] [--top N] [--jobs N] [--out FILE] FILE...
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MURPHY_RE = re.compile(r"MurphyE\([^)]*\)\s*=\s*([-+0-9.eE]+)")
NORM_E_RE = re.compile(r"\be\s+([-+0-9.eE]+)\s+rroots")
SKEWOPT_SKEW_RE = re.compile(r"Best Skew:\s*([-+0-9.eE]+)")
SKEWOPT_MURPHY_RE = re.compile(r"MurphyE:\s*([-+0-9.eE]+)")


def parse_cado(path, label):
    """Yield dicts for each ### root-optimized polynomial ### block."""
    polys = []
    cur = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("### root-optimized polynomial"):
            if cur:
                polys.append(cur)
            cur = {"label": label, "coeffs": {}, "cado_e": None}
        elif line.startswith("###"):
            if cur:
                polys.append(cur)
            cur = None
        elif cur is not None:
            m = re.match(r"^(Y[01]|c\d+|skew):\s*(.+)$", line)
            if m:
                cur["coeffs"][m.group(1)] = m.group(2).strip()
            elif "MurphyE" in line:
                mm = MURPHY_RE.search(line)
                if mm:
                    cur["cado_e"] = float(mm.group(1))
    if cur:
        polys.append(cur)
    # signature for dedup / identity = Y1 + algebraic coeffs
    for p in polys:
        c = p["coeffs"]
        p["y1"] = c.get("Y1", "")
        p["sig"] = (c.get("Y1", ""),) + tuple(c.get(f"c{i}", "0") for i in range(7))
    return polys


def parse_msieve(path, label, keep):
    """Read msieve .p, return the top `keep` results by 'e' (already on the
    cownoise scale; no skewopt needed)."""
    results = []
    cur = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("# norm"):
            if cur:
                results.append(cur)
            m = NORM_E_RE.search(line)
            e = float(m.group(1)) if m else None
            cur = {"label": label, "coeffs": {}, "cado_e": None,
                   "cownoise_e": e, "skewopt": False}
        elif cur is not None:
            m = re.match(r"^(Y[01]|c\d+|skew):\s*(.+)$", line)
            if m:
                cur["coeffs"][m.group(1)] = m.group(2).strip()
    if cur:
        results.append(cur)
    for p in results:
        c = p["coeffs"]
        p["y1"] = c.get("Y1", "")
        p["sig"] = (c.get("Y1", ""),) + tuple(c.get(f"c{i}", "0") for i in range(7))
    results.sort(key=lambda p: (p["cownoise_e"] is not None, p["cownoise_e"] or -1),
                 reverse=True)
    return results[:keep]


def run_skewopt(skewopt, poly):
    c = poly["coeffs"]
    args = [skewopt, c.get("Y0", "0"), c.get("Y1", "0")] + \
           [c.get(f"c{i}", "0") for i in range(7)] + ["0", "0"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=60)
        skew = SKEWOPT_SKEW_RE.search(out.stdout)
        murphy = SKEWOPT_MURPHY_RE.search(out.stdout)
        poly["opt_skew"] = float(skew.group(1)) if skew else None
        poly["cownoise_e"] = float(murphy.group(1)) if murphy else None
    except Exception as exc:  # noqa: BLE001
        print(f"skewopt failed: {exc}", file=sys.stderr)
        poly["opt_skew"] = None
        poly["cownoise_e"] = None
    poly["skewopt"] = True
    return poly


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help=".txt = CADO (rescored), .p = msieve (e used directly)")
    ap.add_argument("--skewopt", default=os.path.expanduser("~/code/SkewOptimizer/skewopt"))
    ap.add_argument("--top", type=int, default=25, help="rows to print (default 25)")
    ap.add_argument("--msieve-keep", type=int, default=50, help="top-N per msieve file to consider")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None, help="write full ranked TSV here")
    args = ap.parse_args()

    cado_polys, msieve_polys = [], []
    for f in args.files:
        label = f"{Path(f).parent.name}/{Path(f).name}"
        if f.endswith(".p"):
            msieve_polys += parse_msieve(f, label, args.msieve_keep)
        else:
            cado_polys += parse_cado(f, label)

    if cado_polys:
        if not os.path.exists(args.skewopt):
            sys.exit(f"Error: skewopt not found: {args.skewopt}")
        print(f"Rescoring {len(cado_polys)} CADO poly(s) through skewopt ({args.jobs} jobs)...",
              file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            cado_polys = list(ex.map(lambda p: run_skewopt(args.skewopt, p), cado_polys))

    allp = [p for p in cado_polys + msieve_polys if p.get("cownoise_e") is not None]
    # dedup by signature, keep best cownoise_e
    best_by_sig = {}
    for p in allp:
        cur = best_by_sig.get(p["sig"])
        if cur is None or p["cownoise_e"] > cur["cownoise_e"]:
            best_by_sig[p["sig"]] = p
    ranked = sorted(best_by_sig.values(), key=lambda p: p["cownoise_e"], reverse=True)

    hdr = f"{'#':>3}  {'cownoise_E':>12}  {'cado_E':>10}  {'Δ%':>6}  {'opt_skew':>14}  source"
    print(hdr)
    print("-" * len(hdr))
    for i, p in enumerate(ranked[:args.top], 1):
        cado = p.get("cado_e")
        if cado:
            delta = f"{100.0 * (p['cownoise_e'] - cado) / cado:+.1f}"
            cado_s = f"{cado:.3e}"
        else:
            delta, cado_s = "", "(msieve)"
        skew = p.get("opt_skew")
        skew_s = f"{skew:.2f}" if skew is not None else "—"
        print(f"{i:>3}  {p['cownoise_e']:>12.4e}  {cado_s:>10}  {delta:>6}  {skew_s:>14}  {p['label']}")

    if ranked:
        b = ranked[0]
        print(f"\nBest on common (cownoise) scale: {b['cownoise_e']:.6e}  "
              f"[{'skewopt' if b.get('skewopt') and b.get('cado_e') else 'msieve e'}]  {b['label']}  Y1={b['y1']}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("rank\tcownoise_E\tcado_E\topt_skew\tsource\tY1\n")
            for i, p in enumerate(ranked, 1):
                fh.write(f"{i}\t{p['cownoise_e']:.6e}\t"
                         f"{'' if p.get('cado_e') is None else f'{p['cado_e']:.6e}'}\t"
                         f"{'' if p.get('opt_skew') is None else f'{p['opt_skew']:.3f}'}\t"
                         f"{p['label']}\t{p['y1']}\n")
        print(f"\nFull ranking -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
