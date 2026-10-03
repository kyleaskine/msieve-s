#!/usr/bin/env python3
"""CPU reference for the alpha computation a GPU root sieve has to reproduce.

  alpha_proto.py validate [--sopt cado_sopt_unsorted.txt] [--count 300]
  alpha_proto.py sieve-check [--cap 256]
  alpha_proto.py benchmark

Definitions (CADO's get_alpha): for primes p <= B (2000),
    alpha = sum_p (1/(p-1) - E_p) * log p,
    E_p   = expected p-valuation of F(a, b) over coprime (a, b)
          = sum_{e >= 1} N_e / (p^e + p^(e-1)),
where N_e counts projective points (a:b) mod p^e with F(a, b) = 0 mod p^e.

validate     exact alpha vs the alpha CADO printed for sopt outputs (2 decimals),
             including the multiplier cases (Res = a*N). Primes with a multiple root
             use the exact p-adic recursion (CADO's special_val0, after Hanrot); point
             counting mod p^e is cross-checked against it for small p. Truncating at
             small p^e is not enough: a double root at p = 577 whose p lifts all
             survive mod p^2 moved one poly's alpha by 0.011.
sieve-check  the sieve formulation: for f_{u,v} = f + (u*x + v)*g and a fixed u, one
             pass over x mod p^e gives N_e(v) for every v mod p^e at once. The x where
             p | g(x) need the special rule below; checked against brute force on both
             benchmark seeds, for every v and several u, for p^e <= --cap.
benchmark    alpha of each benchmark winner from its rotation (u, v) relative to
             another winner of the same seed, vs a direct computation, and the error
             the naive rule (each x gives exactly one v class) would make.

Special rule at the root of g (k = v_p(g(x)) >= 1, h(x) = f(x) + u*x*g(x)):
    if e <= k: x is a root for every v if p^e | h(x), for no v otherwise;
    else:      x is a root iff p^k | h(x) and
               v = -(h(x)/p^k) * (g(x)/p^k)^(-1) mod p^(e-k)   (p^k classes mod p^e).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polyfmt import iter_sopt, parse_cado_poly, coeffs, rotation_between, translate  # noqa: E402
from math import gcd  # noqa: E402

ALPHA_BOUND = 2000
COUNT_CAP = 1 << 12         # point-counting cross-check: p^e up to this


def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


PRIMES = primes_upto(ALPHA_BOUND)


def ev(c, x, m):
    """f(x) mod m, c = [c0..cd]."""
    r = 0
    for a in reversed(c):
        r = (r * x + a) % m
    return r


def ev_rev(c, y, m):
    """F(1, y) mod m: the homogenized poly at (1 : y)."""
    r = 0
    for a in c:
        r = (r * y + a) % m
    return r


def count_points(c, p, e):
    """N_e: projective points (x:1), x mod p^e, and (1:y), p | y, with F = 0 mod p^e."""
    m = p ** e
    n = sum(1 for x in range(m) if ev(c, x, m) == 0)
    n += sum(1 for y in range(0, m, p) if ev_rev(c, y, m) == 0)
    return n


def all_roots_simple(c, p):
    """True if every root of F mod p (affine and projective) is simple, so N_e = N_1."""
    d = len(c) - 1
    dc = [i * c[i] for i in range(1, d + 1)]
    for x in range(p):
        if ev(c, x, p) == 0 and ev(dc, x, p) == 0:
            return False
    if c[d] % p == 0 and c[d - 1] % p == 0:     # projective root of multiplicity > 1
        return False
    return True


def val0(f, p):
    """Average p-valuation of f(x) for x uniform in Z_p (CADO's special_val0): exact for
    square-free f, any multiplicity, any p."""
    v, cont = 0, 0
    for x in f:
        cont = gcd(cont, x)
    g = list(f)
    while cont and cont % p == 0:
        v += 1
        cont //= p
        g = [x // p for x in g]
    d = len(g) - 1
    dg = [i * g[i] for i in range(1, d + 1)]
    for r in range(p):
        if ev(g, r, p):
            continue
        if ev(dg, r, p):
            v += 1 / (p - 1)
        else:                                   # multiple root: recurse on g(p*x + r)
            t = translate(g, r)
            v += val0([t[i] * p ** i for i in range(d + 1)], p) / p
    return v


def expected_valuation(c, p):
    """E_p over coprime (a, b), affine and projective."""
    if all_roots_simple(c, p):
        return count_points(c, p, 1) * p / (p * p - 1)
    d = len(c) - 1
    v = val0(c, p) * p
    if c[d] % p == 0:                           # projective part: reciprocal poly at p*x
        v += val0([c[d - i] * p ** i for i in range(d + 1)], p)
    return v / (p + 1)


def expected_valuation_counting(c, p, cap=COUNT_CAP):
    """The same by counting points mod p^e for p^e <= cap, plus a tail that assumes the
    count has stabilized. Only a cross-check: wrong when lifts still change past the cap."""
    total, e, n = 0.0, 1, 0
    while p ** e <= cap:
        n = count_points(c, p, e)
        total += n / (p ** e + p ** (e - 1))
        e += 1
    return total + n / (p ** (e - 1) * (p + 1)) / (p - 1) * p


def alpha(c, bound=ALPHA_BOUND):
    return sum((1 / (p - 1) - expected_valuation(c, p)) * math.log(p) for p in PRIMES if p <= bound)


def rotated(c, g, u, v):
    """f + (u*x + v) * g for g = [Y0, Y1]."""
    r = list(c)
    r[0] += v * g[0]
    r[1] += v * g[1] + u * g[0]
    r[2] += u * g[1]
    return r


def vp(x, p, cap):
    if x == 0:
        return cap
    k = 0
    while x % p == 0 and k < cap:
        x //= p
        k += 1
    return k


def counts_by_v(c, g, u, p, e, naive=False):
    """Affine N_e(v) for every v mod p^e, for f_{u,v} = f + (u x + v) g, in one pass over x."""
    m = p ** e
    out = [0] * m
    for x in range(m):
        gx = (g[0] + g[1] * x) % m
        hx = (ev(c, x, m) + u * x * gx) % m
        k = vp(gx, p, e)
        if k == 0:
            out[(-hx * pow(gx, -1, m)) % m] += 1
        elif naive:
            continue                                  # the plan's "one v class per x" rule
        elif k >= e:
            if hx == 0:
                for v in range(m):
                    out[v] += 1
        elif vp(hx, p, e) >= k:
            mk = p ** (e - k)
            v0 = (-(hx // p ** k) * pow((gx // p ** k) % mk, -1, mk)) % mk
            for v in range(v0, m, mk):
                out[v] += 1
    return out


def cmd_validate(args):
    # two passes: rank on exp_E alone, then fetch an evenly spaced sample by rank
    order = sorted((opt['exp_E'], i) for i, (_, opt) in enumerate(iter_sopt(args.sopt)))
    step = max(1, len(order) // args.count)
    want = {i for _, i in order[::step][:args.count]}
    sample = [pr for i, pr in enumerate(iter_sopt(args.sopt)) if i in want]
    worst, bad, xworst = 0.0, 0, 0.0
    for raw, opt in sample:
        c = coeffs(opt)
        a = alpha(c)
        err = abs(a - opt['alpha'])
        worst = max(worst, err)
        bad += err > 0.0051
        for p in (2, 3, 5, 7):                  # recursion vs point counting
            xworst = max(xworst, abs(expected_valuation(c, p) - expected_valuation_counting(c, p)))
    mults = sorted({abs(opt['c%d' % (len(coeffs(opt)) - 1)] // raw['c%d' % (len(coeffs(raw)) - 1)])
                    for raw, opt in sample})
    print(f"{len(sample)} sopt outputs (multipliers seen: {mults[:12]}{' ...' if len(mults) > 12 else ''})")
    print(f"alpha vs CADO's printed value: max |diff| {worst:.4f}, over 0.005: {bad}")
    print(f"recursion vs point counting (p <= 7, p^e <= {COUNT_CAP}): max |E_p diff| {xworst:.2e}")


def benchmark_seeds():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    return {
        'c146': [parse_cado_poly(os.path.join(here, 'c146', 'winners', f + '.poly'))
                 for f in ('msieve_best', 'cado_orig_best', 'cado_inv_best')],
        'c204': [parse_cado_poly(os.path.join(here, 'c204', f + '.poly'))
                 for f in ('A_msieve', 'B_cado_orig', 'C_cado_inv')],
    }


def cmd_sieve_check(args):
    total = mism = naive_wrong = 0
    for name, polys in benchmark_seeds().items():
        a = polys[0]
        c, g = coeffs(a), [a['Y0'], a['Y1']]
        for p in [q for q in PRIMES if q <= args.cap]:
            e = 1
            while p ** (e + 1) <= args.cap and p <= 7:
                e += 1
            m = p ** e
            for u in (0, 1, 7, 120 % m, 90 % m):
                fast = counts_by_v(c, g, u, p, e)
                naive = counts_by_v(c, g, u, p, e, naive=True)
                for v in range(m):
                    rc = rotated(c, g, u, v)
                    brute = sum(1 for x in range(m) if ev(rc, x, m) == 0)
                    total += 1
                    mism += fast[v] != brute
                    naive_wrong += naive[v] != brute
        print(f"{name}: g's root mod 2 is x = {(-a['Y0'] * pow(a['Y1'], -1, 2)) % 2}, "
              f"multiplier makes F(x_g) = 0 mod 2: {ev(c, (-a['Y0'] * pow(a['Y1'], -1, 2)) % 2, 2) == 0}")
    print(f"cells checked: {total}; special-rule mismatches: {mism}; naive-rule wrong cells: {naive_wrong}")


def cmd_benchmark(args):
    for name, polys in benchmark_seeds().items():
        a = polys[0]
        ca, ga = coeffs(a), [a['Y0'], a['Y1']]
        print(f"== {name}")
        for x in polys:
            r = rotation_between(a, x)
            if r is None:   # e.g. ropt divided out content and changed the multiplier
                print(f"  {x.get('_path', 'poly')}: not a translation + rotation of the first winner; skipped")
                continue
            cx = coeffs(x)
            direct = alpha(cx)
            # alpha is translation-invariant: compute from (u, v) in A's frame
            u = r['rotation'][1] if len(r['rotation']) > 1 else 0
            v = r['rotation'][0] - u * r['t']
            via = alpha(rotated(ca, ga, u, v))
            # naive-rule error: affine counts at small p^e with and without the special rule
            err = 0.0
            for p in (2, 3, 5, 7):
                e = 1
                while p ** (e + 1) <= 256:
                    e += 1
                for ee in range(1, e + 1):
                    m = p ** ee
                    good = counts_by_v(ca, ga, u, p, ee)[v % m]
                    bad = counts_by_v(ca, ga, u, p, ee, naive=True)[v % m]
                    err += (good - bad) / (m + m // p) * math.log(p)
            sign = -1 if r['negated'] else 1
            print(f"  u={u:5d}  alpha direct {direct:8.4f}  via (u, v) {via:8.4f}  "
                  f"naive rule overstates alpha by {err:.3f} (p <= 7, p^e <= 256)"
                  + ('  [inverted]' if sign < 0 else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--sopt', default='cado_sopt_unsorted.txt')
    v.add_argument('--count', type=int, default=300)
    s = sub.add_parser('sieve-check')
    s.add_argument('--cap', type=int, default=256)
    sub.add_parser('benchmark')
    args = ap.parse_args()
    {'validate': cmd_validate, 'sieve-check': cmd_sieve_check, 'benchmark': cmd_benchmark}[args.cmd](args)


if __name__ == '__main__':
    main()
