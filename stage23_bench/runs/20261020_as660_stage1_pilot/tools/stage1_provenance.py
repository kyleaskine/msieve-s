#!/usr/bin/env python3
"""Validate GPU stage-1 JSONL, reconstruct rejected hits, and write an identity ledger.

Usage: stage1_provenance.py sidecar.w*.jsonl --raw run.ms --ledger candidates.jsonl
Optional --sopt CADO_SOPT_OUTPUT joins through its printed RAW input, not optimized
Y0 or leading coefficient. Keep candidate_id with every later ropt submission.
Missing batch/worker endings and saturation are reported; --require-complete
rejects such corpora. All arithmetic reconstruction is CPU only.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from polyfmt import (read_cado_blocks, iter_sopt, degree, resultant,
                     parse_cado_ropt, parse_msieve_p, rotation_between)


def integer_root(n, d):
    x = 1 << ((n.bit_length() + d - 1) // d)
    while True:
        y = ((d - 1) * x + n // x ** (d - 1)) // d
        if y >= x:
            return x
        x = y


@lru_cache(maxsize=1024)
def transform(n, ad, d):
    t = d ** d * ad ** (d - 1) * n
    return t, integer_root(t, d)


def tdiv(a, b):
    return (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)


def reconstruct(run, coefficient, hit):
    """Independent integer version of handle_collision; no admission cutoff."""
    n, ad, d = int(run['N']), int(coefficient['ad']), int(run['degree'])
    q, p1, p2 = (int(hit[k]) for k in ('q', 'p1', 'p2'))
    P = q * p1 * p2
    if math.gcd(p1, p2) != 1 or math.gcd(q, p1 * p2) != 1:
        return None
    trans_n, m0 = transform(n, ad, d)
    m = m0 + int(hit['qroot']) + int(hit['offset']) * q * q
    if (pow(m, d, P * P) - trans_n) % (P * P):
        return None
    mod = d * ad
    if math.gcd(P, mod) != 1:
        return None
    b = m - tdiv(m, mod) * mod
    b *= pow(P, -1, mod)
    b -= tdiv(b, mod) * mod
    if b > mod // 2:
        b -= mod
    m = tdiv(m - b * P, mod)
    return n, ad, P, m


def candidate_id(key):
    return hashlib.sha256(':'.join(map(str, key)).encode('ascii')).hexdigest()


def audit(paths):
    candidates = {}
    counts = Counter()
    exposure = Counter()
    incomplete = []
    for path in paths:
        run = None
        coefficients, batches, endings, coeff_end = {}, {}, set(), set()
        worker_end = False
        hit_counts = Counter()
        with open(path) as stream:
            for lineno, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise ValueError(f'{path}:{lineno}: invalid/truncated JSON') from e
                kind = row['type']
                if kind == 'run':
                    if run is not None or row['schema'] != 1:
                        raise ValueError('duplicate run or unsupported schema')
                    run = row
                    continue
                if run is None:
                    raise ValueError('missing run record')
                if kind == 'coefficient':
                    cid = row['coefficient']
                    if cid in coefficients:
                        raise ValueError('duplicate coefficient ID')
                    coefficients[cid] = row
                elif kind == 'batch':
                    key = row['coefficient'], row['batch']
                    if key in batches or len(row['roots']) != row['qroots']:
                        raise ValueError('duplicate batch or root count mismatch')
                    c = coefficients[key[0]]
                    for q, root in row['roots']:
                        if not c['qmin'] <= q <= c['qmax'] or not 0 <= int(root) < q*q:
                            raise ValueError('q/root outside recorded bounds')
                        exposure[(run['N'], c['ad'], q, int(root), row['aprog'])] += 1
                    batches[key] = row
                    counts['batches'] += 1
                    counts['qroots'] += row['qroots']
                elif kind == 'hit':
                    key = row['coefficient'], row['batch']
                    batch = batches[key]
                    c = coefficients[key[0]]
                    if [row['q'], row['qroot']] not in batch['roots']:
                        raise ValueError('hit q/root absent from batch')
                    identity = reconstruct(run, c, row)
                    counts['returned_hits'] += 1
                    hit_counts[key] += 1
                    counts['admitted_hits' if row['admitted'] else 'rejected_hits'] += 1
                    if row['valid'] is not None:
                        if bool(identity) != row['valid']:
                            raise ValueError('host/offline collision validity disagreement')
                        if identity and tuple(map(int, (row['ad'],row['P'],row['m']))) != identity[1:]:
                            raise ValueError('host/offline raw identity disagreement')
                    if identity:
                        counts['valid_hits'] += 1
                        ident = candidate_id(identity)
                        entry = candidates.setdefault(ident, dict(candidate_id=ident,
                            N=str(identity[0]), ad=str(identity[1]), P=str(identity[2]),
                            m=str(identity[3]), discoveries=[], raw=[], sopt=[]))
                        entry['discoveries'].append(dict(sidecar=str(path), run=run['run'],
                            worker=run['worker'], coefficient=key[0], batch=key[1],
                            hit=row['hit'], q=row['q'], ratio=row['ratio'], admitted=row['admitted']))
                elif kind == 'batch_end':
                    key = row['coefficient'], row['batch']
                    if key not in batches or key in endings or row['returned'] != hit_counts[key]:
                        raise ValueError('batch ending/count mismatch')
                    endings.add(key)
                    if row['saturated']:
                        incomplete.append(f'{path}: saturated batch {key}')
                        counts['saturated_batches'] += 1
                elif kind == 'coefficient_end':
                    coeff_end.add(row['coefficient'])
                    if not row['complete']:
                        incomplete.append(f'{path}: incomplete coefficient {row["coefficient"]}')
                elif kind == 'worker_end':
                    worker_end = True
        for key in batches.keys() - endings:
            incomplete.append(f'{path}: missing batch_end {key}')
        for cid in coefficients.keys() - coeff_end:
            incomplete.append(f'{path}: missing coefficient_end {cid}')
        if not worker_end:
            incomplete.append(f'{path}: missing worker_end')
    counts['unique_candidates'] = len(candidates)
    counts['repeated_qroot_work'] = sum(n-1 for n in exposure.values())
    return candidates, counts, exposure, incomplete


def join_raw(candidates, paths):
    # Rational identity survives raw expansion even if the leading coefficient
    # changes. Preserve all ambiguous candidate links rather than guessing.
    index = defaultdict(list)
    for entry in candidates.values():
        index[(int(entry['N']), int(entry['P']), -int(entry['m']))].append(entry)
    missing = 0
    for path in paths:
        for number, (_, raw) in enumerate(read_cado_blocks(path)):
            if resultant(raw) % raw['n']:
                raise ValueError('invalid raw polynomial resultant')
            matches = index.get((raw['n'],raw['Y1'],raw['Y0']), [])
            if not matches:
                missing += 1
            for entry in matches:
                entry['raw'].append(dict(path=str(path), index=number, polynomial=raw,
                                         ambiguous=len(matches)>1))
    return missing


def join_sopt(candidates, paths):
    index = defaultdict(list)
    def key(p):
        return tuple(p.get(k) for k in ['n','Y1','Y0']+[f'c{i}' for i in range(degree(p)+1)])
    for entry in candidates.values():
        for row in entry['raw']:
            index[key(row['polynomial'])].append(entry)
    missing = 0
    for path in paths:
        for number, (raw, opt) in enumerate(iter_sopt(path)):
            matches = index.get(key(raw), [])
            if not matches:
                missing += 1
            for entry in {e['candidate_id']: e for e in matches}.values():
                entry['sopt'].append(dict(path=str(path), index=number, polynomial=opt))
    return missing


def join_ropt(candidates, paths, fmt):
    """Verify an exact translation/rotation from a saved sopt input.

    Y1 is only an index into possible inputs. An exact algebraic transformation
    and compatible N are required. Retain ambiguous origins explicitly.
    """
    index = defaultdict(list)
    for entry in candidates.values():
        for row in entry['sopt']:
            index[row['polynomial']['Y1']].append((entry, row))
    missing = 0
    for path in paths:
        outputs = (parse_cado_ropt(path) if fmt == 'cado' else
                   parse_msieve_p(path) if fmt == 'msieve' else
                   [p for _, p in read_cado_blocks(path)])
        for number, opt in enumerate(outputs):
            matches = []
            for entry, seed in index.get(opt['Y1'], []):
                n = int(entry['N'])
                if opt.get('n', n) != n or resultant(opt) % n:
                    continue
                if degree(seed['polynomial']) != degree(opt):
                    continue
                proof = rotation_between(seed['polynomial'], opt)
                if proof is not None:
                    matches.append((entry, seed, proof))
            origins = {entry['candidate_id'] for entry, _, _ in matches}
            if not origins:
                missing += 1
            for entry, seed, proof in matches:
                entry.setdefault('ropt', []).append(dict(path=str(path), index=number,
                    polynomial=opt, sopt_path=seed['path'], sopt_index=seed['index'],
                    transformation=proof, ambiguous=len(origins)>1))
    return missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sidecars',nargs='+')
    p.add_argument('--raw',action='append',default=[])
    p.add_argument('--sopt',action='append',default=[])
    p.add_argument('--ropt',action='append',default=[])
    p.add_argument('--ropt-format',choices=('cado','msieve','poly'),default='cado')
    p.add_argument('--ledger')
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args()
    candidates,counts,_,incomplete=audit(a.sidecars)
    counts['unmatched_raw']=join_raw(candidates,a.raw)
    counts['unmatched_sopt']=join_sopt(candidates,a.sopt)
    counts['unmatched_ropt']=join_ropt(candidates,a.ropt,a.ropt_format)
    counts['raw_links']=sum(len(e['raw']) for e in candidates.values())
    counts['sopt_links']=sum(len(e['sopt']) for e in candidates.values())
    counts['ropt_links']=sum(len(e.get('ropt', [])) for e in candidates.values())
    if a.ledger:
        with open(a.ledger,'x') as f:
            for entry in candidates.values():
                f.write(json.dumps(entry)+'\n')
    print(json.dumps(dict(counts=counts,incomplete=incomplete),indent=2))
    if a.require_complete and (incomplete or any(counts[k] for k in
            ('unmatched_raw','unmatched_sopt','unmatched_ropt'))):
        raise SystemExit(1)

if __name__=='__main__':
    main()
