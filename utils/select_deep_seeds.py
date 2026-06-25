#!/usr/bin/env python3
"""Select a unified deep-ropt seed list from round-1 ropt results.

Seed identity is the leading rational coefficient ``Y1``. Root optimization in
both msieve and CADO only translates (x -> x + t), which shifts Y0 by t*Y1 but
leaves Y1 untouched; the pipeline's "inverted" sign flip negates the algebraic
coefficients and also leaves Y1 untouched. So Y1 is a single stable, unique key
that joins every round-1 result back to its seed across both tools and both
sign orientations -- verified unique in best300_msieve (300/300),
best150_cado (150/150) and resopt_sorted (2000/2000).

Selection (the design agreed with the user):
  * From each of the four round-1 sources -- msieve_orig, msieve_inv,
    cado_orig, cado_inv -- take the top ``--per-source`` seeds by that source's
    own best post-ropt score. Ranking happens WITHIN a source, so the msieve
    ``e`` scale and the CADO MurphyE scale are never compared directly.
  * Union those four sets (dedup by Y1), recording a vote count = in how many of
    the four lists a seed appeared. A seed proven by both orig and inv (or by
    both tools) is a reliable winner rather than a lucky single ropt grid.
  * Fill the remaining slots up to ``--total`` with the best (lowest) exp_E
    seeds not already chosen. exp_E is a (log) norm-style score where SMALLER is
    better; best300_msieve is stored ascending, i.e. best-first.

Emits the selected seeds in msieve (.ms) and CADO (size-opt block) formats, both
orig and inverted, ordered by exp_E descending, plus a manifest. The output
files are drop-in --source-orig/--source-inv inputs for deep_msieve_ropt.sh and
deep_cado_ropt.sh, feeding the SAME unified seed set into both deep passes.
"""

import argparse
import math
import re
import sys
from pathlib import Path

E_RE = re.compile(r"\be\s+([-+0-9.eEnN]+)\s+rroots")
MURPHY_RE = re.compile(r"MurphyE\([^)]*\)\s*=\s*([-+0-9.eE]+)")
CCOEFF_RE = re.compile(r"^(c\d+):\s*(-?\d+)\s*$")


def parse_msieve_results(path):
    """msieve .p -> {Y1: best 'e' score}. Each result block is a '# norm' line
    (carrying the e score) followed by skew/c*/Y0/Y1 lines."""
    best = {}
    if not path or not Path(path).exists():
        return best
    cur_e = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("# norm"):
            m = E_RE.search(line)
            cur_e = None
            if m:
                try:
                    cur_e = float(m.group(1))
                except ValueError:
                    cur_e = None
        elif line.startswith("Y1:") and cur_e is not None:
            y1 = line.split(":", 1)[1].strip()
            if cur_e > best.get(y1, -math.inf):
                best[y1] = cur_e
    return best


def parse_cado_results(path):
    """CADO ropt output -> {Y1: best MurphyE}. Root-optimized blocks have an
    uncommented 'Y1:' line followed by a '# side 1 MurphyE(...)=' line; the
    echoed input block uses '# Y1:' (commented) so it is ignored."""
    best = {}
    if not path or not Path(path).exists():
        return best
    cur_y1 = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("Y1:"):
            cur_y1 = line.split(":", 1)[1].strip()
        elif "MurphyE" in line and cur_y1 is not None:
            m = MURPHY_RE.search(line)
            if m:
                score = float(m.group(1))
                if score > best.get(cur_y1, -math.inf):
                    best[cur_y1] = score
    return best


def parse_msieve_universe(path):
    """best300_msieve.ms -> {Y1: (raw_line, exp_E)}. Columns: c5 c4 c3 c2 c1 c0
    Y1 Y0 proj_alpha exp_E ..."""
    seeds = {}
    if not Path(path).exists():
        return seeds
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) < 10:
            continue
        y1 = cols[6]
        try:
            exp_e = float(cols[9])
        except ValueError:
            exp_e = math.inf
        seeds[y1] = (line, exp_e)
    return seeds


def parse_msieve_universe_inv(path):
    """best300_msieve_inv.ms -> {Y1: raw_line}."""
    seeds = {}
    if not Path(path).exists():
        return seeds
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) < 7:
            continue
        seeds[cols[6]] = line
    return seeds


def parse_cado_universe(path):
    """resopt_sorted.txt -> {Y1: [body lines]} (n: .. through '# side 1 ...'),
    header line dropped so it can be renumbered on emit."""
    seeds = {}
    if not Path(path).exists():
        return seeds
    body = []
    y1 = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("### Size-optimized polynomial"):
            if y1 is not None and body:
                seeds[y1] = body
            body, y1 = [], None
        elif line.strip() == "":
            if y1 is not None and body:
                seeds[y1] = body
            body, y1 = [], None
        else:
            body.append(line)
            if line.startswith("Y1:"):
                y1 = line.split(":", 1)[1].strip()
    if y1 is not None and body:
        seeds[y1] = body
    return seeds


def invert_cado_block(body):
    """Negate c-coefficient lines; leave n/Y0/Y1/skew/comments unchanged."""
    out = []
    for line in body:
        m = CCOEFF_RE.match(line)
        if m:
            val = m.group(2)
            val = val[1:] if val.startswith("-") else "-" + val
            out.append(f"{m.group(1)}: {val}")
        else:
            out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="pipeline_results")
    ap.add_argument("--work-dir", default="pipeline_work")
    ap.add_argument("--msieve-orig", default=None)
    ap.add_argument("--msieve-inv", default=None)
    ap.add_argument("--cado-orig", default=None)
    ap.add_argument("--cado-inv", default=None)
    ap.add_argument("--msieve-universe", default=None)
    ap.add_argument("--msieve-universe-inv", default=None)
    ap.add_argument("--cado-universe", default=None,
                    help="Full CADO size-opt universe with skew (default: work-dir/resopt_sorted.txt)")
    ap.add_argument("--per-source", type=int, default=8,
                    help="Top-N proven winners taken from each of the 4 sources (default 8)")
    ap.add_argument("--total", type=int, default=32,
                    help="Total seeds in the deep list; remainder filled by exp_E (default 32)")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: results-dir/deep_seeds)")
    args = ap.parse_args()

    rd, wd = Path(args.results_dir), Path(args.work_dir)
    msieve_orig = args.msieve_orig or rd / "msieve_ropt_orig.p"
    msieve_inv = args.msieve_inv or rd / "msieve_ropt_inv.p"
    cado_orig = args.cado_orig or rd / "cado_ropt_orig.txt"
    cado_inv = args.cado_inv or rd / "cado_ropt_inv.txt"
    uni_ms = args.msieve_universe or rd / "best300_msieve.ms"
    uni_ms_inv = args.msieve_universe_inv or rd / "best300_msieve_inv.ms"
    uni_cado = args.cado_universe or wd / "resopt_sorted.txt"
    out_dir = Path(args.out_dir) if args.out_dir else rd / "deep_seeds"

    universe = parse_msieve_universe(uni_ms)
    universe_inv = parse_msieve_universe_inv(uni_ms_inv)
    cado_blocks = parse_cado_universe(uni_cado)
    if not universe:
        sys.exit(f"Error: empty/missing msieve universe: {uni_ms}")
    if not cado_blocks:
        sys.exit(f"Error: empty/missing CADO universe: {uni_cado}")

    sources = {
        "msieve_orig": parse_msieve_results(msieve_orig),
        "msieve_inv": parse_msieve_results(msieve_inv),
        "cado_orig": parse_cado_results(cado_orig),
        "cado_inv": parse_cado_results(cado_inv),
    }

    # top per_source winners from each source (ranked within that source)
    votes, src_seen = {}, {}
    for name, scores in sources.items():
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.per_source]
        for y1, _score in top:
            votes[y1] = votes.get(y1, 0) + 1
            src_seen.setdefault(y1, []).append(name)

    winners = set(votes)
    # If winners somehow exceed the budget, keep the most-corroborated ones.
    if len(winners) > args.total:
        def best_any(y1):
            return max((s.get(y1, -math.inf) for s in sources.values()), default=-math.inf)
        winners = set(sorted(winners,
                             key=lambda y: (votes[y], best_any(y)),
                             reverse=True)[:args.total])

    selected = set(winners)
    # exp_E fill: best (lowest) exp_E first -- smaller exp_E is the better seed
    for y1, (_line, exp_e) in sorted(universe.items(),
                                     key=lambda kv: kv[1][1]):
        if len(selected) >= args.total:
            break
        if y1 not in selected:
            selected.add(y1)

    missing_ms = [y for y in selected if y not in universe]
    missing_cado = [y for y in selected if y not in cado_blocks]
    if missing_ms:
        print(f"WARNING: {len(missing_ms)} selected seed(s) absent from msieve universe; skipped", file=sys.stderr)
    if missing_cado:
        print(f"WARNING: {len(missing_cado)} selected seed(s) absent from CADO universe; skipped", file=sys.stderr)
    selected = [y for y in selected if y in universe and y in cado_blocks]

    # order output by exp_E ascending (best first -- smaller is better)
    selected.sort(key=lambda y: universe[y][1])

    out_dir.mkdir(parents=True, exist_ok=True)
    ms_out = out_dir / "deep_seeds_msieve.ms"
    ms_inv_out = out_dir / "deep_seeds_msieve_inv.ms"
    cado_out = out_dir / "deep_seeds_cado.txt"
    cado_inv_out = out_dir / "deep_seeds_cado_inv.txt"
    manifest = out_dir / "deep_seeds_manifest.tsv"

    with ms_out.open("w") as f:
        for y1 in selected:
            f.write(universe[y1][0] + "\n")
    with ms_inv_out.open("w") as f:
        for y1 in selected:
            if y1 in universe_inv:
                f.write(universe_inv[y1] + "\n")

    def write_cado(path, invert):
        with path.open("w") as f:
            for i, y1 in enumerate(selected):
                body = cado_blocks[y1]
                if invert:
                    body = invert_cado_block(body)
                f.write(f"### Size-optimized polynomial ({i}) ###\n")
                for line in body:
                    f.write(line + "\n")
                f.write("\n")
    write_cado(cado_out, invert=False)
    write_cado(cado_inv_out, invert=True)

    with manifest.open("w") as f:
        f.write("rank\texp_E\tvotes\tsources\tmsieve_orig\tmsieve_inv\tcado_orig\tcado_inv\treason\tY1\n")
        for i, y1 in enumerate(selected):
            exp_e = universe[y1][1]
            reason = "proven" if y1 in winners else "exp_E"

            def s(name):
                v = sources[name].get(y1)
                return "" if v is None else f"{v:.3e}"
            f.write(f"{i}\t{exp_e:.2f}\t{votes.get(y1, 0)}\t{','.join(src_seen.get(y1, []))}\t"
                    f"{s('msieve_orig')}\t{s('msieve_inv')}\t{s('cado_orig')}\t{s('cado_inv')}\t"
                    f"{reason}\t{y1}\n")

    n_proven = sum(1 for y in selected if y in winners)
    print(f"Selected {len(selected)} seeds: {n_proven} proven-winner(s) (union of top-{args.per_source} "
          f"from 4 sources), {len(selected) - n_proven} exp_E fill.")
    print(f"  msieve : {ms_out}  +  {ms_inv_out}")
    print(f"  cado   : {cado_out}  +  {cado_inv_out}")
    print(f"  manifest: {manifest}")


if __name__ == "__main__":
    main()
