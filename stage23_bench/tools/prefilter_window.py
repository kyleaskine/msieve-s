#!/usr/bin/env python3
"""How much does a small-prime prefilter lose? Exhaustive check on windows around the
known optima of the benchmark seeds (a CPU stand-in for the exhaustive GPU run).

  prefilter_window.py [--lines 2] [--frac 0.1] [--wcap 5e7] [--procs 4]

A ropt prefilter scores every residue class (u, v mod 2520) by its affine root counts
at 8, 9, 5 and 7 and sieves only the best classes. Here every cell of each window is
scored by affine alpha over p < 2000 (prime powers <= 2000, with the g-root rule from
alpha_proto.py), using the C line sieve alpha_window.c. For each window it reports the
best cell overall and the best cell inside the top 0.01% / 0.1% / 1% / 10% of classes
(cutoffs from all classes with |u| <= 150), re-scored with exact alpha.

Windows: for each benchmark optimum, its u-line +- --lines, and on each line the v range
where c0 (in that optimum's own translation frame) changes by at most --frac, capped at
--wcap. Size is roughly constant there, so a better alpha means a better polynomial;
the lognorm at the window corners is checked with CADO sopt and printed.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alpha_proto as ap  # noqa: E402
from polyfmt import coeffs, rotation_between, translate, parse_sopt, cado_binary  # noqa: E402

PCAP = 2000
SMALL = {2: 3, 3: 2, 5: 1, 7: 1}
M = 2520
CUTS = (1e-4, 1e-3, 1e-2, 1e-1)


def weights(c, g, u, p, e):
    """Score added to each v class mod p^e: N_e(v) / (p^e + p^(e-1)) * log p."""
    m = p ** e
    cnt = ap.counts_by_v(c, g, u, p, e)
    return [cnt[v] / (m + m // p) * math.log(p) for v in range(m)]


def class_table(c, g, u):
    """The prefilter's class score for every v mod M on line u (prime powers dividing M)."""
    cls = [0.0] * M
    for p, emax in SMALL.items():
        for e in range(1, emax + 1):
            w, m = weights(c, g, u, p, e), p ** e
            for v in range(M):
                cls[v] += w[v % m]
    return cls


def tables_for_u(c, g, u):
    """[(m, weights)] for every prime power m = p^e <= PCAP, p < 2000."""
    tabs = []
    for p in ap.PRIMES:
        e = 1
        while p ** e <= PCAP:
            tabs.append((p ** e, weights(c, g, u, p, e)))
            e += 1
    return tabs


def work(job):
    """One u-line: build its tables (pure Python, so in a worker process) and sieve it."""
    binary, c, g, u, v_lo, n, cuts = job
    return u, run_line(binary, u, v_lo, n, tables_for_u(c, g, u), class_table(c, g, u), cuts)


def run_line(binary, u, v_lo, n, tabs, cls, cuts):
    spec = [f"W {u} {v_lo} {n}", "S %d %s" % (M, ' '.join('%.7f' % x for x in cls)),
            "THR %d %s" % (len(cuts), ' '.join('%.7f' % t for t in cuts))]
    spec += ["T %d %s" % (m, ' '.join('%.7f' % x for x in w)) for m, w in tabs]
    out = subprocess.run([binary], input='\n'.join(spec + ['END']) + '\n',
                         capture_output=True, text=True, check=True).stdout
    tops, thr = [], {}
    for line in out.splitlines():
        f = line.split()
        if f[0] == 'TOP':
            tops.append((float(f[3]), int(f[1]), int(f[2]), float(f[4])))
        else:
            thr[int(f[1])] = (float(f[4]), int(f[2]), int(f[3]), int(f[5]))
    return tops, thr


def own_frame(c, g, u, v, t):
    """f_A + (u x + v) g_A, translated by t (the optimum's own frame); returns (coeffs, Y0)."""
    return translate(ap.rotated(c, g, u, v), t), g[0] + g[1] * t


def lognorms(n, polys, y1):
    """Input lognorm (at its L2 skew) of each poly, as CADO sopt prints it."""
    fd, path = tempfile.mkstemp(suffix='.poly')
    with os.fdopen(fd, 'w') as fh:
        for cs, y0 in polys:
            fh.write(f"n: {n}\nY0: {y0}\nY1: {y1}\n" + ''.join(f"c{i}: {x}\n" for i, x in enumerate(cs)) + '\n')
    out = subprocess.run([cado_binary('sopt'), '-sopteffort', '0', '-inputpolys', path],
                         capture_output=True, text=True).stdout
    os.remove(path)
    return [float(l.split('lognorm ')[1].split(',')[0]) for l in out.splitlines() if l.startswith('# # side 1 lognorm')]


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument('--lines', type=int, default=2)
    ap_.add_argument('--frac', type=float, default=0.1)
    ap_.add_argument('--wcap', type=float, default=5e7)
    ap_.add_argument('--procs', type=int, default=4)
    args = ap_.parse_args()

    binary = os.path.join(HERE, 'alpha_window')
    src = os.path.join(HERE, 'alpha_window.c')
    if not os.path.exists(binary) or os.path.getmtime(binary) < os.path.getmtime(src):
        subprocess.run(['gcc', '-O3', '-march=native', '-o', binary, src], check=True)

    for name, polys in ap.benchmark_seeds().items():
        a = polys[0]
        n = parse_n(name)
        c, g = coeffs(a), [a['Y0'], a['Y1']]
        allcls = sorted((x for u in range(-150, 151) for x in class_table(c, g, u)), reverse=True)
        cuts = [allcls[max(0, int(f * len(allcls)) - 1)] for f in CUTS]

        def pct(sv):
            lo, hi = 0, len(allcls)                 # allcls is descending: count entries > sv
            while lo < hi:
                mid = (lo + hi) // 2
                if allcls[mid] > sv + 1e-9:
                    lo = mid + 1
                else:
                    hi = mid
            return lo / len(allcls)

        print(f"== {name} (frame: msieve winner)")
        for label, x in zip(('msieve', 'cado orig'), polys[:2]):
            r = rotation_between(a, x)
            if r is None:   # e.g. ropt divided out content and changed the multiplier
                print(f"  {label}: not a translation + rotation of the msieve winner; skipped")
                continue
            ux = r['rotation'][1] if len(r['rotation']) > 1 else 0
            vx = r['rotation'][0] - ux * r['t']
            W = int(min(args.frac * abs(coeffs(x)[0]) / abs(x['Y0']), args.wcap))
            lines = range(ux - args.lines, ux + args.lines + 1)
            centers = {u: vx - (u - ux) * r['t'] for u in lines}   # same own-frame v as the optimum

            corners = [own_frame(c, g, u, centers[u] + dv, r['t']) for u in (lines[0], lines[-1]) for dv in (-W, W)]
            ln = lognorms(n, [own_frame(c, g, ux, vx, r['t'])] + corners, a['Y1'])

            jobs = [(binary, c, g, u, centers[u] - W, 2 * W + 1, cuts) for u in lines]
            with ProcessPoolExecutor(args.procs) as ex:
                res = dict(ex.map(work, jobs))

            def exact(u, v):
                return ap.alpha(ap.rotated(c, g, u, v))
            tops = sorted((t for u in lines for t in res[u][0]), reverse=True)[:5]
            top_exact = sorted(((exact(u, v), u, v, s) for _, u, v, s in tops))
            best_alpha, bu, bv, bs = top_exact[0]
            print(f"  {label}: (u, v) = ({ux}, {vx}), alpha {exact(ux, vx):.3f}; window {len(lines)} lines x "
                  f"{2 * W + 1:,} cells; lognorm centre {ln[0]:.2f}, corners {min(ln[1:]):.2f}-{max(ln[1:]):.2f}")
            print(f"    best cell in window: alpha {best_alpha:.3f} at (u, v) = ({bu}, {bv}), "
                  f"its class is in the top {pct(bs):.3%}")
            for i, f in enumerate(CUTS):
                cells = sum(res[u][1][i][3] for u in lines)
                if cells == 0:      # alpha_window reports a placeholder cell when nothing passes
                    print(f"    prefilter top {f:.2%} of classes: no cell in the window kept")
                    continue
                cand = max((res[u][1][i] for u in lines if res[u][1][i][3] > 0), key=lambda t: t[0])
                ea = exact(cand[1], cand[2])
                print(f"    prefilter top {f:.2%} of classes ({cells:,} cells kept): best alpha {ea:.3f}, "
                      f"loss {ea - best_alpha:+.3f}")


def parse_n(name):
    with open(os.path.join(HERE, '..', 'data', name, 'n.txt')) as fh:
        return int(fh.read())


if __name__ == '__main__':
    main()
