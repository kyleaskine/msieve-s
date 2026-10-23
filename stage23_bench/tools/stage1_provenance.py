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
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from sidecar_io import open_sidecar
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


def audit(paths, streaming=True, summary=None):
    """Streaming mode retains one batch per worker and binned q exposure.

    Candidate identities are still retained for exact joins. Exact duplicate
    q/root detection requires explicit streaming=False; it can consume many GB.
    """
    if summary is None:
        summary = {}
    summary.update(exposure_mode='q_uint32_sixteenths' if streaming else 'exact',
                   coefficients=[], timing=[], launches=[])
    candidates = {}
    counts = Counter()
    exposure = Counter()
    log_exposure = Counter()
    incomplete = []
    for path in paths:
        run = None
        coefficients, batches, endings, coeff_end = {}, {}, set(), set()
        last_batch, qdone, found, setups = Counter(), Counter(), Counter(), {}
        timing, launches = defaultdict(Counter), Counter()
        active_launches = {}
        worker_end = False
        hit_counts = Counter()
        with open_sidecar(path) as stream:
            for lineno, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except ValueError as e:
                    if not line.endswith('\n'):
                        incomplete.append(f'{path}:{lineno}: truncated final JSON record')
                        break
                    raise ValueError(f'{path}:{lineno}: invalid/truncated JSON') from e
                kind = row['type']
                if kind == 'run':
                    if run is not None or row['schema'] not in (1, 2):
                        raise ValueError('duplicate run or unsupported schema')
                    run = row
                    continue
                if run is None:
                    raise ValueError('missing run record')
                if worker_end:
                    raise ValueError('records after worker_end')
                if kind == 'coefficient':
                    cid = row['coefficient']
                    if cid in coefficients:
                        raise ValueError('duplicate coefficient ID')
                    coefficients[cid] = row
                    active_launches[cid] = {}
                elif kind == 'setup':
                    setups[row['coefficient']] = row
                elif kind == 'batch':
                    key = row['coefficient'], row['batch']
                    if (key in batches or len(row['roots']) != row['qroots']
                            or key[1] != last_batch[key[0]] + 1):
                        raise ValueError('duplicate batch or root count mismatch')
                    if streaming and batches:
                        raise ValueError('new batch before previous batch ended')
                    if key[0] in coeff_end:
                        raise ValueError('batch after coefficient_end')
                    last_batch[key[0]] = key[1]
                    c = coefficients[key[0]]
                    for q, root in row['roots']:
                        if not c['qmin'] <= q <= c['qmax'] or not 0 <= int(root) < q*q:
                            raise ValueError('q/root outside recorded bounds')
                        ekey = ((run['N'], c['ad'], q >> 28, row['aprog']) if streaming
                                else (run['N'], c['ad'], q, int(root), row['aprog']))
                        exposure[ekey] += 1
                        log_exposure[(run['N'], c['ad'], q.bit_length()-1, row['aprog'])] += 1
                    batches[key] = row
                    # Normalize decimal-string roots for constant-time hit lookup.
                    row['_root_set'] = {(q, int(root)) for q, root in row['roots']}
                    qdone[key[0]] += row['qroots']
                    t = timing[(key[0], (key[1]-1)//100000)]
                    t['generation_seconds'] += row.get('generation_seconds', 0)
                    counts['batches'] += 1
                    counts['qroots'] += row['qroots']
                elif kind == 'hit':
                    key = row['coefficient'], row['batch']
                    batch = batches[key]
                    c = coefficients[key[0]]
                    if (row['q'], int(row['qroot'])) not in batch['_root_set']:
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
                    if row['valid']:
                        found[key[0]] += 1
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
                    for shape in active_launches[key[0]].values():
                        launches[shape] += 1
                    if streaming:
                        del batches[key]
                        hit_counts.pop(key, None)
                    else:
                        endings.add(key)
                    t = timing[(key[0], (key[1]-1)//100000)]
                    t['batches'] += 1
                    for name in ('seconds', 'audit_prepare_seconds',
                                 'search_seconds', 'readback_submit_seconds'):
                        t[name] += row.get(name, 0)
                    t['gpu_span_seconds'] += row.get('gpu_span_ms', 0)/1000
                    if row['saturated']:
                        incomplete.append(f'{path}: saturated batch {key}')
                        counts['saturated_batches'] += 1
                elif kind == 'engine':
                    for name in ('bucket_grows', 'hash_caps', 'arena_fallbacks'):
                        counts[name] += row.get(name, 0)
                    if row.get('hash_caps', 0):
                        incomplete.append(f'{path}: hash cap in batch {row["batch"]}')
                elif kind == 'launch':
                    active_launches[row['coefficient']][row['group']] = tuple(row[k]
                        for k in ('coefficient', 'group', 'num_p', 'roots_per_p',
                                  'blocks_x', 'blocks_y', 'threads', 'q_tile'))
                elif kind == 'coefficient_end':
                    cid = row['coefficient']
                    if cid not in coefficients or cid in coeff_end:
                        raise ValueError('unknown/duplicate coefficient ending')
                    coeff_end.add(cid)
                    if row.get('qroots_done', qdone[cid]) != qdone[cid]:
                        raise ValueError('coefficient qroot count mismatch')
                    if row.get('found', found[cid]) != found[cid]:
                        raise ValueError('coefficient hit count mismatch')
                    setup = setups.get(cid, {})
                    if row['complete'] and setup.get('qroots_total', qdone[cid]) != qdone[cid]:
                        raise ValueError('complete coefficient did not cover setup qroots')
                    summary['coefficients'].append(dict(sidecar=str(path),
                        parameters=coefficients[cid], setup=setup, ending=row))
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
        summary['timing'].extend(dict(sidecar=str(path), coefficient=k[0],
            batch_first=k[1]*100000+1, **v) for k,v in timing.items())
        names = ('coefficient', 'group', 'num_p', 'roots_per_p', 'blocks_x',
                 'blocks_y', 'threads', 'q_tile')
        summary['launches'].extend(dict(sidecar=str(path), **dict(zip(names,k)),
            calls=v) for k,v in launches.items())
    counts['unique_candidates'] = len(candidates)
    counts['repeated_qroot_work'] = None if streaming else sum(n-1 for n in exposure.values())
    if streaming:
        summary['exposure'] = [dict(N=k[0], ad=k[1], q_min=max(1,k[2]<<28),
            q_max=((k[2]+1)<<28)-1, aprog=k[3], qroots=v)
            for k,v in sorted(exposure.items())]
    summary['log2_exposure'] = [dict(N=k[0], ad=k[1], log2_bin=k[2],
        q_min=1 << k[2], q_max=(1 << (k[2]+1))-1, aprog=k[3], qroots=v,
        progression_qroots=v*k[3]) for k,v in sorted(log_exposure.items())]
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
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--stream',dest='stream',action='store_true',default=True,
                      help='default: retain active batches and binned q exposure')
    mode.add_argument('--exact-exposure',dest='stream',action='store_false',
                      help='retain all q/root exposure; small fixtures only, can use tens of GB')
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args()
    # Reserve the output before any expensive reading, also closing on errors.
    with open(a.ledger,'x') if a.ledger else nullcontext(None) as f:
        summary={}
        candidates,counts,_,incomplete=audit(a.sidecars,streaming=a.stream,summary=summary)
        counts['unmatched_raw']=join_raw(candidates,a.raw)
        counts['unmatched_sopt']=join_sopt(candidates,a.sopt)
        counts['unmatched_ropt']=join_ropt(candidates,a.ropt,a.ropt_format)
        counts['raw_links']=sum(len(e['raw']) for e in candidates.values())
        counts['sopt_links']=sum(len(e['sopt']) for e in candidates.values())
        counts['ropt_links']=sum(len(e.get('ropt', [])) for e in candidates.values())
        if f:
            for entry in candidates.values():
                f.write(json.dumps(entry)+'\n')
    print(json.dumps(dict(counts=counts,incomplete=incomplete,summary=summary),indent=2))
    if a.require_complete and (incomplete or any(counts[k] for k in
            ('unmatched_raw','unmatched_sopt','unmatched_ropt'))):
        raise SystemExit(1)

if __name__=='__main__':
    main()
