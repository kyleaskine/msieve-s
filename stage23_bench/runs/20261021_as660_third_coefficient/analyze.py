#!/usr/bin/env python3
"""Audit the completed third coefficient and extend the frozen CPU sopt pilot."""
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / '20261021_as660_two_coefficients'
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from stage1_provenance import audit, join_raw, join_sopt
from polyfmt import cado_poly_text, resultant


def save(name, value):
    with (HERE / name).open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot_and_audit():
    source = ROOT / 'as660-stage1-b.w0.jsonl'
    offset = json.loads((PREVIOUS / 'snapshot.json').read_text())['prefix_bytes']
    initial = source.stat()
    digest = hashlib.sha256()
    with source.open('rb') as src, (HERE / 'third.w0.jsonl').open('xb') as out:
        header = src.readline()
        assert json.loads(header)['type'] == 'run'
        src.seek(offset)
        first = src.readline()
        row = json.loads(first)
        assert row['type'] == 'coefficient' and row['ad'] == '415800'
        for piece in (header, first):
            out.write(piece)
            digest.update(piece)
        while chunk := src.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    final = source.stat()
    assert (initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns)
    skip, take, number, block = 346946, 309570, 0, []
    with (ROOT / 'as660.ms').open('rb') as src, (HERE / 'third.ms').open('xb') as out:
        for line in src:
            block.append(line)
            if line.strip():
                continue
            if not any(x.startswith(b'n:') for x in block):
                block = []
                continue
            if skip <= number < skip + take:
                out.write(b''.join(block))
            number += 1
            block = []
            if number == skip + take:
                break
    assert number == skip + take
    save('snapshot.json', dict(source=str(source), source_size=initial.st_size,
        source_mtime_ns=initial.st_mtime_ns,
        source_byte_ranges=[[0, len(header)], [offset, initial.st_size]],
        byte_ranges='half-open; original run header followed by third coefficient through worker_end',
        sha256=digest.hexdigest(), synthetic_records_added=False,
        raw_start_index=skip, raw_rows=take, raw_sha256=file_hash(HERE / 'third.ms')))
    print('snapshot saved; auditing completed third coefficient', flush=True)
    summary = {}
    candidates, counts, _, incomplete = audit([HERE / 'third.w0.jsonl'], summary=summary)
    assert not incomplete, incomplete
    assert len(summary['coefficients']) == 1
    assert summary['coefficients'][0]['parameters']['ad'] == '415800'
    counts['unmatched_raw'] = join_raw(candidates, [HERE / 'third.ms'])
    counts['raw_links'] = sum(len(e['raw']) for e in candidates.values())
    assert counts['unmatched_raw'] == 0
    assert counts['raw_links'] == counts['valid_hits'] == counts['returned_hits'] == take
    with (HERE / 'ledger.jsonl').open('x') as out:
        for e in candidates.values():
            out.write(json.dumps(e) + '\n')
    save('audit.json', dict(counts=counts, summary=summary, incomplete=incomplete,
        completed_coefficient_validated=True))
    print(json.dumps(dict(counts=counts, incomplete=incomplete)), flush=True)


def quality():
    with (HERE / 'ledger.jsonl').open() as ledger:
        entries = heapq.nsmallest(10000, (json.loads(l) for l in ledger),
                                 key=lambda e: e['candidate_id'])
    assert len(entries) == 10000
    with (HERE / 'sample-415800.ms').open('x') as raw, (HERE / 'sample-415800.jsonl').open('x') as ledger:
        for e in entries:
            p = e['raw'][0]['polynomial']
            assert all(r['polynomial'] == p for r in e['raw'])
            raw.write(cado_poly_text(p) + '\n')
            ledger.write(json.dumps(e) + '\n')
    binary = PREVIOUS / 'bin/sopt'
    expected = json.loads((PREVIOUS / 'quality-manifest.json').read_text())['binary_sha256']
    assert file_hash(binary) == expected
    save('quality-manifest.json', dict(binary=str(binary), binary_sha256=expected,
        effort=0, threads=1, sample_count=10000,
        selection='10,000 lexicographically smallest SHA-256 candidate IDs; one raw per identity',
        sample_sha256=file_hash(HERE / 'sample-415800.ms'),
        previous_comparison=str(PREVIOUS / 'comparison.json')))
    cmd = [str(binary), '-sopteffort', '0', '-inputpolys', str(HERE / 'sample-415800.ms')]
    print('starting 10,000-candidate CPU sopt sample', flush=True)
    start = time.monotonic()
    with (HERE / 'sopt-415800.out').open('x') as out, (HERE / 'sopt-415800.err').open('x') as err:
        done = subprocess.run(cmd, stdout=out, stderr=err,
                              env=dict(os.environ, OMP_NUM_THREADS='1'))
    save('quality-status.json', dict(command=cmd, returncode=done.returncode,
        seconds=time.monotonic() - start))
    assert done.returncode == 0
    print('CPU sopt finished', flush=True)


def quantile(values, q):
    pos = (len(values) - 1) * q
    i, f = int(pos), pos - int(pos)
    return values[i] * (1 - f) + values[min(i + 1, len(values) - 1)] * f


def stats(values):
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), best=values[0],
                q01=quantile(values, .01), q10=quantile(values, .1), median=quantile(values, .5))


def summarize():
    audit_data = json.loads((HERE / 'audit.json').read_text())
    counts = audit_data['counts']
    batches = counts['batches']
    ratio_max, unique, repeated = 0, 0, 0
    raw_hits = Counter()
    with (HERE / 'ledger.jsonl').open() as ledger:
        for line in ledger:
            e = json.loads(line)
            unique += 1
            repeated += len(e['discoveries']) - 1
            for d in e['discoveries']:
                ratio_max = max(ratio_max, d['ratio'])
                raw_hits[d['q'] >> 30] += 1
    exposure = Counter()
    for e in audit_data['summary']['exposure']:
        exposure[e['q_min'] >> 30] += e['qroots']
    with (HERE / 'sample-415800.jsonl').open() as ledger:
        candidates = {e['candidate_id']: e for e in map(json.loads, ledger)}
    assert join_sopt(candidates, [HERE / 'sopt-415800.out']) == 0
    qvalues, timevalues, scores = defaultdict(list), defaultdict(list), []
    ambiguous_q = ambiguous_time = 0
    with (HERE / 'scores-415800.jsonl').open('x') as out:
        for e in candidates.values():
            assert len(e['sopt']) == 1
            opt = e['sopt'][0]['polynomial']
            assert resultant(opt) % int(e['N']) == 0
            score = opt['exp_E']
            scores.append(score)
            qb = {d['q'] >> 30 for d in e['discoveries']}
            tb = {min(3, (d['batch'] - 1) * 4 // batches) for d in e['discoveries']}
            if len(qb) == 1:
                qvalues[next(iter(qb))].append(score)
            else:
                ambiguous_q += 1
            if len(tb) == 1:
                timevalues[next(iter(tb))].append(score)
            else:
                ambiguous_time += 1
            out.write(json.dumps(dict(candidate_id=e['candidate_id'], exp_E=score,
                q_bins=sorted(qb), chronology_bins=sorted(tb), lognorm=opt['lognorm'], skew=opt['skew'])) + '\n')
    assert len(scores) == 10000
    result = json.loads((PREVIOUS / 'comparison.json').read_text())
    result['415800'] = dict(returned_hits=counts['returned_hits'], unique_candidates=unique,
        repeat_discoveries=repeated, repeat_fraction=repeated/counts['returned_hits'],
        admitted_hits=counts['admitted_hits'], qroots=counts['qroots'],
        hits_per_million_qroots=counts['returned_hits']/counts['qroots']*1e6,
        unique_per_million_qroots=unique/counts['qroots']*1e6,
        gate_ratio_max=ratio_max, coefficient=audit_data['summary']['coefficients'][0],
        sopt=stats(scores),
        q_bins=[dict(bin=b, qroots=exposure[b], raw_hits=raw_hits[b],
            hits_per_million_qroots=raw_hits[b]/exposure[b]*1e6, sample=stats(qvalues[b])) for b in range(4)],
        chronology_bins=[dict(bin=b, sample=stats(timevalues[b])) for b in range(4)],
        sampled_identities_spanning_q_bins=ambiguous_q,
        sampled_identities_spanning_chronology_bins=ambiguous_time)
    rng = random.Random(660)
    for ad in ('138600', '277200'):
        with (PREVIOUS / f'scores-{ad}.jsonl').open() as f:
            baseline = [json.loads(line)['exp_E'] for line in f]
        differences = defaultdict(list)
        for _ in range(400):
            a, b = (sorted(rng.choices(s, k=len(s))) for s in (baseline, scores))
            differences['mean'].append(statistics.mean(b) - statistics.mean(a))
            for name, q in (('median', .5), ('q01', .01), ('q10', .1)):
                differences[name].append(quantile(b, q) - quantile(a, q))
        result[f'bootstrap_415800_minus_{ad}'] = {k: dict(
            estimate=result['415800']['sopt'][k] - result[ad]['sopt'][k],
            percentile95=[quantile(sorted(v), .025), quantile(sorted(v), .975)])
            for k, v in differences.items()}
    save('comparison.json', result)
    print(json.dumps(result['415800'], indent=2), flush=True)


if __name__ == '__main__':
    snapshot_and_audit()
    quality()
    summarize()
