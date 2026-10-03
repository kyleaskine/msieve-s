#!/usr/bin/env python3
"""Sweep CADO sopt over transformed copies of the top seeds' raw polynomials.

  sopt_sweep.py --mode mult   --range 500 --top 300 --out DIR [--procs 6]
  sopt_sweep.py --mode prerot --range 256 --top 300 --out DIR
  sopt_sweep.py ... --analyze-only          # re-analyze DIR without rerunning sopt

mult:   feeds a*f_raw + g for a = 1..range. Content stays 1, so sopt's LLL can only
        return multipliers that are multiples of a; sweeping a shows whether CADO's
        own choice of multiplier is the exp_E-best one (M0: it nearly always is).
prerot: feeds f_raw + v*x^(d-3)*g for v = -range..range (Kleinjung's pre-rotation,
        x^2*g for degree 5),
        which changes the translation candidates sopt starts from (M0: no gain).

Seeds and the baseline come from --baseline (default: pipeline_work/resopt_output.txt,
the pipeline's effort-50 re-sopt), ranked by exp_E. Reports, per seed, the best result
over the sweep against the baseline. With --emit-alt FILE (mult mode), writes each
seed's best result whose |multiplier| differs from the baseline's, in the pipeline's
msieve .ms format, ready for scripts/run_msieve_ropt_annotated.sh.
"""

import argparse
import collections
import os
import statistics
import subprocess

from polyfmt import parse_sopt, multiplier, degree, msieve_ms_line, cado_binary

def candidates(raw, mode, rng):
    d = degree(raw)
    for v in (range(1, rng + 1) if mode == 'mult' else range(-rng, rng + 1)):
        c = [raw['c%d' % i] for i in range(d + 1)]
        if mode == 'mult':
            c = [v * x for x in c]
            c[0] += raw['Y0']          # + g keeps the content 1 (CADO divides it out)
            c[1] += raw['Y1']
        else:
            c[d - 2] += v * raw['Y1']  # + v * x^(d-3) * g
            c[d - 3] += v * raw['Y0']
        yield v, c


def generate(seeds, args):
    os.makedirs(args.out, exist_ok=True)
    outs = [open(os.path.join(args.out, f'in_{i:02d}.poly'), 'w') for i in range(args.procs)]
    j = 0
    for raw, _ in seeds:
        for _, c in candidates(raw, args.mode, args.range):
            outs[j % args.procs].write(f"n: {raw['n']}\nY0: {raw['Y0']}\nY1: {raw['Y1']}\n"
                                       + ''.join(f"c{i}: {x}\n" for i, x in enumerate(c)) + '\n')
            j += 1
    for fh in outs:
        fh.close()
    return j


def run(args):
    procs = []
    for i in range(args.procs):
        inp = os.path.join(args.out, f'in_{i:02d}.poly')
        out = open(os.path.join(args.out, f'out_{i:02d}.txt'), 'w')
        procs.append(subprocess.Popen([args.sopt, '-sopteffort', str(args.effort), '-inputpolys', inp],
                                      stdout=out, stderr=subprocess.STDOUT))
    for p in procs:
        if p.wait():
            raise SystemExit(f"sopt failed (exit {p.returncode})")


def analyze(seeds, args):
    info = {raw['Y1']: dict(rank=i + 1, raw=raw, base=opt, a0=multiplier(raw, opt))
            for i, (raw, opt) in enumerate(seeds)}
    results = collections.defaultdict(list)
    for i in range(args.procs):
        for inp, opt in parse_sopt(os.path.join(args.out, f'out_{i:02d}.txt')):
            s = info[opt['Y1']]
            d = degree(inp)
            if args.mode == 'mult':
                v = inp['c%d' % d] // s['raw']['c%d' % d]
            else:
                v = (inp['c%d' % (d - 2)] - s['raw']['c%d' % (d - 2)]) // inp['Y1']
            results[opt['Y1']].append((opt['exp_E'], v, multiplier(s['raw'], opt), opt))

    gains, rows = [], []
    for y, s in info.items():
        r = sorted(results[y], key=lambda t: t[0])
        if not r:
            continue
        gains.append(s['base']['exp_E'] - r[0][0])
        alt = [t for t in r if abs(t[2]) != abs(s['a0'])]
        s['alt'] = alt[0] if alt else None
        rows.append((s['rank'], s['base']['exp_E'], r[0][0], r[0][1], r[0][2], s['a0']))
    print(f"{len(rows)} seeds, {statistics.mean(len(v) for v in results.values()):.0f} results each")
    print(f"best over sweep vs baseline: mean gain {statistics.mean(gains):.3f}  "
          f"median {statistics.median(gains):.3f}  max {max(gains):.3f}")
    for th in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"  seeds gaining > {th}: {sum(g > th for g in gains)}")
    print("best 15 after the sweep: rank  baseline  best  sweep-value  multiplier  baseline-multiplier")
    for r in sorted(rows, key=lambda r: r[2])[:15]:
        print(f"  {r[0]:4d}  {r[1]:.2f}  {r[2]:.2f}  {r[3]:5d}  {r[4]:6d}  {r[5]:6d}")
    if args.mode == 'mult':
        gaps = sorted(s['alt'][0] - s['base']['exp_E'] for s in info.values() if s.get('alt'))
        print(f"best alternative |multiplier| is behind the baseline by: median {statistics.median(gaps):.2f}"
              f" (25% {gaps[len(gaps) // 4]:.2f}, 75% {gaps[3 * len(gaps) // 4]:.2f})")
        if args.emit_alt:
            with open(args.emit_alt, 'w') as fh:
                for s in sorted(info.values(), key=lambda s: s['rank']):
                    if s.get('alt'):
                        fh.write(msieve_ms_line(s['alt'][3]) + '\n')
            print(f"wrote {args.emit_alt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('mult', 'prerot'), required=True)
    ap.add_argument('--range', type=int, required=True)
    ap.add_argument('--top', type=int, default=300)
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline', default='pipeline_work/resopt_output.txt')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--effort', type=int, default=0)
    ap.add_argument('--sopt', help='CADO sopt binary (default: from nfs_config.ini)')
    ap.add_argument('--emit-alt', help='mult mode: write best alternative-multiplier polys (.ms)')
    ap.add_argument('--analyze-only', action='store_true')
    args = ap.parse_args()

    pairs = sorted(parse_sopt(args.baseline), key=lambda p: p[1]['exp_E'])
    seeds, seen = [], set()
    for raw, opt in pairs:              # one entry per seed
        if opt['Y1'] not in seen:
            seen.add(opt['Y1'])
            seeds.append((raw, opt))
    seeds = seeds[:args.top]
    if not args.analyze_only:
        args.sopt = args.sopt or cado_binary('sopt')
        n = generate(seeds, args)
        print(f"running {n} sopt candidates on {args.procs} processes ...", flush=True)
        run(args)
    analyze(seeds, args)


if __name__ == '__main__':
    main()
