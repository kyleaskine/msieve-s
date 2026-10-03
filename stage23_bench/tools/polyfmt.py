"""Parsers and exact integer helpers for polyselect outputs.

Formats handled:
  - CADO `sopt` output: "### Input raw polynomial" / "### Size-optimized polynomial" pairs
  - CADO `polyselect_ropt` output: "### root-optimized polynomial" blocks
  - msieve `-npr` .p output, as annotated by scripts/run_msieve_ropt_annotated.sh
  - single CADO-format polynomial files (n:, Y0:, Y1:, c0: ..., skew:)

Polynomials are dicts with exact Python ints: 'Y0', 'Y1', 'c0'..'cd' and, when known,
'n'. Scores (skew, lognorm, exp_E, ...) are floats. f(x) = sum c_i x^i, g(x) = Y1 x + Y0.
"""

import os
import re
from math import comb

_SOPT_STATS = re.compile(r'lognorm ([-\d.]+), exp_E ([-\d.]+), alpha ([-\d.]+) '
                         r'\(proj ([-\d.]+)\), (\d+) real')
_COEFF = re.compile(r'(n|Y0|Y1|c\d+): (-?\d+)\s*$')


def degree(p):
    return max(int(k[1:]) for k in p if k[0] == 'c' and k[1:].isdigit())


def coeffs(p):
    """[c0, c1, ..., cd]"""
    return [p['c%d' % i] for i in range(degree(p) + 1)]


def parse_sopt(path):
    """List of (raw, opt) pairs from CADO sopt output, in file order (see iter_sopt).
    Holds every pair in memory; use iter_sopt for multi-million-poly files."""
    return list(iter_sopt(path))


def iter_sopt(path):
    """Yield (raw, opt) pairs from CADO sopt output, in file order.

    raw has the input coefficients (and 'exp_E' of the input as printed by sopt);
    opt has the output coefficients plus skew, lognorm, exp_E, alpha, proj, rroots.
    """
    raw, opt, mode = {}, {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith('### Input raw polynomial'):
                raw, mode = {}, 'in'
                continue
            if line.startswith('### Size-optimized polynomial'):
                opt, mode = {}, 'out'
                continue
            if mode == 'in':
                m = _COEFF.match(line[2:]) if line.startswith('# ') else None
                if m:
                    raw[m.group(1)] = int(m.group(2))
                m = _SOPT_STATS.search(line)
                if m:
                    raw['exp_E'] = float(m.group(2))
            elif mode == 'out':
                m = _COEFF.match(line)
                if m:
                    opt[m.group(1)] = int(m.group(2))
                elif line.startswith('skew:'):
                    opt['skew'] = float(line.split()[1])
                m = _SOPT_STATS.search(line)
                if m and not line.startswith('# #'):
                    opt.update(lognorm=float(m.group(1)), exp_E=float(m.group(2)),
                               alpha=float(m.group(3)), proj=float(m.group(4)),
                               rroots=int(m.group(5)))
                    yield raw, opt
                    mode = None


def raw_key(raw):
    """Identity of a raw (stage-1) polynomial. Y1 alone is shared by translations and
    rotations of the same seed, so it identifies the seed, not the raw poly."""
    return (raw['Y1'], raw['Y0'], raw['c%d' % degree(raw)])


def multiplier(raw, opt):
    """The integer a with opt = a*raw(x+k) + r(x)*g(x+k). CADO sopt's LLL includes f in
    its lattice, so |a| > 1 is common; Res(opt) = a*Res(raw)."""
    d = degree(raw)
    a, rem = divmod(opt['c%d' % d], raw['c%d' % d])
    assert rem == 0, "leading coefficients are not related by an integer multiplier"
    return a


def parse_msieve_p(path):
    """Blocks of msieve -npr output: dicts with e (Murphy E), norm, alpha, rroots,
    exp_E (the seed's sopt exp_E, if annotated), skew, Y0, Y1, c0..cd."""
    out, cur = [], None
    head = re.compile(r'# norm (\S+) alpha (\S+) e (\S+) rroots (\d+)(?: exp_E (\S+))?')
    with open(path) as fh:
        for line in fh:
            m = head.match(line)
            if m:
                cur = dict(norm=float(m.group(1)), alpha=float(m.group(2)), e=float(m.group(3)),
                           rroots=int(m.group(4)))
                if m.group(5):
                    cur['exp_E'] = float(m.group(5))
                continue
            if cur is None:
                continue
            m = _COEFF.match(line)
            if m:
                cur[m.group(1)] = int(m.group(2))
                if m.group(1) == 'Y1':
                    out.append(cur)
                    cur = None
            elif line.startswith('skew:'):
                cur['skew'] = float(line.split()[1])
    return out


def parse_cado_ropt(path):
    """Blocks of CADO polyselect_ropt output: dicts with MurphyE, lognorm, alpha, skew,
    n, Y0, Y1, c0..cd. Echoed input lines ('# Y1: ...') are ignored."""
    out, cur = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith('### root-optimized polynomial'):
                cur = {}
                continue
            if cur is None:
                continue
            m = _COEFF.match(line)
            if m:
                cur[m.group(1)] = int(m.group(2))
            elif line.startswith('skew:'):
                cur['skew'] = float(line.split()[1])
            elif line.startswith('# side 1 lognorm'):
                m = re.search(r'lognorm ([-\d.]+), E [-\d.]+, alpha ([-\d.]+)', line)
                if m:
                    cur.update(lognorm=float(m.group(1)), alpha=float(m.group(2)))
            elif line.startswith('# side 1 MurphyE'):
                cur['MurphyE'] = float(line.rsplit('=', 1)[1])
                out.append(cur)
                cur = None
    return out


def parse_cado_poly(path):
    """First polynomial in a CADO-format file (n:, Y0:, Y1:, c0: ..., skew:)."""
    p = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            m = _COEFF.match(line)
            if m:
                p[m.group(1)] = int(m.group(2))
            elif line.startswith('skew:'):
                p['skew'] = float(line.split()[1])
            elif not line.strip() and 'Y1' in p and 'c0' in p:
                break
    return p


def cado_poly_text(p, n=None):
    """CADO-format text for p (n: from p or the argument)."""
    n = p.get('n', n)
    lines = [] if n is None else ['n: %d' % n]
    lines += ['Y0: %d' % p['Y0'], 'Y1: %d' % p['Y1']]
    lines += ['c%d: %d' % (i, c) for i, c in enumerate(coeffs(p))]
    if 'skew' in p:
        lines.append('skew: %s' % p['skew'])
    return '\n'.join(lines) + '\n'


def msieve_ms_line(p):
    """One line of the pipeline's msieve .ms format: c_d .. c0 Y1 Y0 proj exp_E 0."""
    cs = ' '.join(str(c) for c in reversed(coeffs(p)))
    return f"{cs} {p['Y1']} {p['Y0']} {p['proj']:.2f} {p['exp_E']:.2f} 0"


def resultant(p):
    """Res(f, g) up to sign: sum c_i (-Y0)^i Y1^(d-i)."""
    c = coeffs(p)
    d = len(c) - 1
    return sum(c[i] * (-p['Y0']) ** i * p['Y1'] ** (d - i) for i in range(d + 1))


def resultant_multiple(p, n):
    """Res(f, g) / n, or None if n does not divide it."""
    q, r = divmod(resultant(p), n)
    return q if r == 0 else None


def translate(c, t):
    """Coefficients of f(x + t), c[i] = coefficient of x^i."""
    d = len(c) - 1
    return [sum(c[j] * comb(j, i) * t ** (j - i) for j in range(i, d + 1)) for i in range(d + 1)]


def div_linear(c, g1, g0):
    """Exact c(x) / (g1 x + g0) -> (quotient, remainder)."""
    r = list(c)
    q = [0] * (len(c) - 1)
    for i in range(len(c) - 1, 0, -1):
        if r[i] % g1:
            return None, None
        q[i - 1] = r[i] // g1
        r[i] -= q[i - 1] * g1
        r[i - 1] -= q[i - 1] * g0
    return q, r[0]


def rotation_between(a, x):
    """Express x as a(y + t) + r(y) * g_x(y), for two polys of the same seed (same Y1)
    and the same multiplier. Returns (t, r) with r = [r0, r1, ...], or None.
    If the leading coefficients have opposite signs (an inverted-pass output),
    x's f is negated first; the returned dict says so."""
    if a['Y1'] != x['Y1']:
        return None
    cx = coeffs(x)
    negated = (cx[-1] > 0) != (coeffs(a)[-1] > 0)
    if negated:
        cx = [-c for c in cx]
    t, rem = divmod(x['Y0'] - a['Y0'], a['Y1'])
    if rem:
        return None
    diff = [u - v for u, v in zip(cx, translate(coeffs(a), t))]
    q, r = div_linear(diff, x['Y1'], x['Y0'])
    if q is None or r != 0:
        return None
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return dict(t=t, rotation=q, negated=negated)


def _average_ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation, with tied values given their average rank (matters
    when many entries share a value, e.g. seeds with no ropt result scored as 0)."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)
    return cov / var ** 0.5 if var else 0.0


def load_fixture(path, effort='e0'):
    """(raw, opt, rank, set) rows of a make_fixture.py sopt_sample.tsv.gz; opt is the CADO
    result at the given effort ('e0' or 'e50'; None where that effort was not run).
    N comes from n.txt in the same directory."""
    import gzip
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), 'n.txt')) as fh:
        n = int(fh.read())
    rows = []
    with gzip.open(path, 'rt') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        for line in fh:
            r = dict(zip(header, line.rstrip('\n').split('\t')))
            d = max(int(k[len('raw_c'):]) for k in r if k.startswith('raw_c'))
            raw = {'n': n, 'Y1': int(r['Y1']), 'Y0': int(r['raw_Y0'])}
            raw.update({'c%d' % i: int(r['raw_c%d' % i]) for i in range(d + 1)})
            opt = None
            if r.get(effort + '_Y0'):
                opt = {'n': n, 'Y1': raw['Y1'], 'Y0': int(r[effort + '_Y0'])}
                opt.update({'c%d' % i: int(r['%s_c%d' % (effort, i)]) for i in range(d + 1)})
                opt.update(skew=float(r[effort + '_skew']), lognorm=float(r[effort + '_lognorm']),
                           exp_E=float(r[effort + '_exp_E']), alpha=float(r[effort + '_alpha']),
                           proj=float(r[effort + '_proj']), rroots=int(r[effort + '_rroots']))
            rows.append((raw, opt, int(r['rank_e0']), r['set']))
    return rows


def cado_build_dir():
    """CADO build directory: $CADO_BUILD_DIR, else cado_build_dir in the repository's
    nfs_config.ini (the setting nfs_optimize.sh uses)."""
    build = os.environ.get('CADO_BUILD_DIR')
    if not build:
        ini = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'nfs_config.ini')
        if os.path.exists(ini):
            with open(ini) as fh:
                for line in fh:
                    m = re.match(r'\s*cado_build_dir\s*=\s*(\S+)', line)
                    if m:
                        build = m.group(1)
                        break
    if not build:
        raise SystemExit("can't locate CADO: set CADO_BUILD_DIR or cado_build_dir in nfs_config.ini")
    return os.path.expanduser(os.path.expandvars(build))


def cado_binary(name):
    """Path of CADO's polyselect/<name>."""
    return os.path.join(cado_build_dir(), 'polyselect', name)


def cado_src_dir(build=None):
    """CADO source directory, from the build directory's CMakeCache.txt."""
    build = build or cado_build_dir()
    cache = os.path.join(build, 'CMakeCache.txt')
    try:
        with open(cache) as fh:
            for line in fh:
                if line.startswith('CMAKE_HOME_DIRECTORY'):
                    return line.split('=', 1)[1].strip()
    except OSError as e:
        raise SystemExit(f"can't read {cache}: {e}")
    raise SystemExit(f"no CMAKE_HOME_DIRECTORY in {cache}")


def cado_compile_args(build=None):
    """(include flags, static libs, link flags) for code using CADO's polyselect and utils
    libraries. The one definition shared with stage23_gpu/Makefile (see main below)."""
    build = build or cado_build_dir()
    src = cado_src_dir(build)
    incs = [f'-I{src}', f'-I{src}/utils', f'-I{src}/polyselect', f'-I{build}']
    libs = [f'{build}/polyselect/libpolyselect_common.a', f'{build}/utils/libutils.a']
    return incs, libs, ['-lgmp', '-lm', '-lpthread', '-lstdc++']


def cado_expe_binary():
    """Path of cado_expe (full-precision exp_E from CADO's own code), built on first use
    against the configured CADO build, and rebuilt when its source or CADO's libs change."""
    here = os.path.dirname(os.path.abspath(__file__))
    binary, src = os.path.join(here, 'cado_expe'), os.path.join(here, 'cado_expe.c')
    incs, libs, ldflags = cado_compile_args()
    newest = max(os.path.getmtime(f) for f in [src] + libs)
    if os.path.exists(binary) and os.path.getmtime(binary) >= newest:
        return binary
    import subprocess
    subprocess.run(['gcc', '-O2', '-std=c99', '-fopenmp'] + incs + ['-o', binary, src] + libs + ldflags,
                   check=True)
    return binary


def cado_expe(polys, n):
    """Full-precision (skew, lognorm, exp_E, alpha, alpha_proj) for each poly, computed by
    CADO's own code at CADO's combined f/g skew, exactly as sopt computes the exp_E it
    prints (rounded) - see cado_expe.c."""
    import subprocess
    if not polys:
        return []
    text = '\n'.join(cado_poly_text({k: v for k, v in p.items()
                                     if k in ('Y0', 'Y1') or (k[0] == 'c' and k[1:].isdigit())}, n)
                     for p in polys) + '\n'
    out = subprocess.run([cado_expe_binary()], input=text, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    if len(out) != len(polys):
        raise RuntimeError(f"cado_expe scored {len(out)} of {len(polys)} polynomials")
    keys = ('skew', 'lognorm', 'exp_E', 'alpha', 'alpha_proj')
    return [dict(zip(keys, (float(x) for x in line.split('\t')[3:]))) for line in out]


if __name__ == '__main__':
    # make variables for stage23_gpu/Makefile: polyfmt.py --cado-inc|--cado-libs [BUILD]
    import sys
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ('--cado-inc', '--cado-libs'):
        raise SystemExit('usage: polyfmt.py --cado-inc|--cado-libs [CADO_BUILD_DIR]')
    incs, libs, ldflags = cado_compile_args(sys.argv[2] if len(sys.argv) == 3 and sys.argv[2] else None)
    print(' '.join(incs if sys.argv[1] == '--cado-inc' else libs + ldflags))
