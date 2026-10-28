#!/usr/bin/env python3
"""CPU-only extension of AS660's frozen, equal-sample quality comparison."""
from collections import Counter, defaultdict
import gc
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
PREV = HERE.parent / '20261021_as660_two_coefficients'
THIRD = HERE.parent / '20261021_as660_third_coefficient'
ADS = (554400, 693000, 831600)
EXPECTED = (151003, 230703, 199223)
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from stage1_provenance import audit, join_raw, join_sopt
from polyfmt import cado_poly_text, resultant
from run_evidence import open_evidence


def save(path, data):
    with path.open('x') as out:
        json.dump(data, out, indent=2)
        out.write('\n')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot():
    source = ROOT / 'as660-stage1-c.w0.jsonl'
    before = source.stat()
    records, current, out, pos = [], None, None, 0
    with source.open('rb') as src:
        header = src.readline()
        assert json.loads(header)['type'] == 'run'
        pos = len(header)
        for line in src:
            if line.startswith(b'{"type":"coefficient",'):
                if out:
                    out.close()
                    current.update(end=pos, sha256=hash_state.hexdigest())
                row = json.loads(line)
                ad = int(row['ad'])
                assert ad == ADS[len(records)]
                folder = HERE / str(ad)
                folder.mkdir()
                out = (folder / 'coefficient.w0.jsonl').open('xb')
                out.write(header)
                hash_state = hashlib.sha256(header)
                current = dict(ad=ad, start=pos)
                records.append(current)
            assert out is not None
            out.write(line)
            hash_state.update(line)
            if line.startswith(b'{"type":"coefficient_end",'):
                current['ending'] = json.loads(line)
                assert current['ending']['complete']
            pos += len(line)
    out.close()
    current.update(end=pos, sha256=hash_state.hexdigest())
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert len(records) == 3 and line == b'{"type":"worker_end"}\n'
    assert tuple(r['ending']['found'] for r in records) == EXPECTED
    skip, number, block, index = 656516, 0, [], 0
    raw_before = (ROOT / 'as660.ms').stat()
    with (ROOT / 'as660.ms').open('rb') as src:
        out = (HERE / str(ADS[0]) / 'raw.ms').open('xb')
        stop = skip + EXPECTED[0]
        for line in src:
            block.append(line)
            if line.strip():
                continue
            if not any(x.startswith(b'n:') for x in block):
                block = []
                continue
            if number >= skip:
                out.write(b''.join(block))
            number += 1
            block = []
            if number == stop:
                out.close()
                index += 1
                if index == len(ADS):
                    break
                stop += EXPECTED[index]
                out = (HERE / str(ADS[index]) / 'raw.ms').open('xb')
    assert number == skip + sum(EXPECTED)
    raw_after = (ROOT / 'as660.ms').stat()
    assert (raw_before.st_size, raw_before.st_mtime_ns) == (raw_after.st_size, raw_after.st_mtime_ns)
    raw_start = skip
    for r, count in zip(records, EXPECTED):
        folder = HERE / str(r['ad'])
        save(folder / 'snapshot.json', dict(source=str(source), source_size=before.st_size,
            source_mtime_ns=before.st_mtime_ns,
            source_byte_ranges=[[0, len(header)], [r['start'], r['end']]],
            byte_ranges='half-open; original run header and coefficient section; only final section includes worker_end',
            synthetic_records_added=False, sha256=r['sha256'], ending=r['ending'],
            raw_start_index=raw_start, raw_rows=count, raw_sha256=digest(folder / 'raw.ms')))
        raw_start += count
    save(HERE / 'snapshot.json', dict(source=str(source), coefficients=records))
    print('snapshots saved for all three completed coefficients', flush=True)


def analyze(ad):
    folder = HERE / str(ad)
    path = folder / 'coefficient.w0.jsonl'
    summary = {}
    print(f'{ad}: auditing', flush=True)
    candidates, counts, _, incomplete = audit([path], summary=summary)
    expected_missing = [] if ad == ADS[-1] else [f'{path}: missing worker_end']
    assert incomplete == expected_missing, incomplete
    assert len(summary['coefficients']) == 1
    assert int(summary['coefficients'][0]['parameters']['ad']) == ad
    counts['unmatched_raw'] = join_raw(candidates, [folder / 'raw.ms'])
    counts['raw_links'] = sum(len(e['raw']) for e in candidates.values())
    assert counts['unmatched_raw'] == 0
    assert counts['raw_links'] == counts['valid_hits'] == counts['returned_hits'] == EXPECTED[ADS.index(ad)]
    save(folder / 'audit.json', dict(counts=counts, summary=summary, incomplete=incomplete,
        completed_coefficient_validated=True))
    with (folder / 'ledger.jsonl').open('x') as out:
        for e in candidates.values():
            out.write(json.dumps(e) + '\n')
    entries = heapq.nsmallest(10000, candidates.values(), key=lambda e: e['candidate_id'])
    del candidates
    gc.collect()
    print(f'{ad}: audit passed {json.dumps(dict(counts))}', flush=True)
    with (folder / 'sample.ms').open('x') as raw, (folder / 'sample.jsonl').open('x') as ledger:
        for e in entries:
            p = e['raw'][0]['polynomial']
            assert all(r['polynomial'] == p for r in e['raw'])
            raw.write(cado_poly_text(p) + '\n')
            ledger.write(json.dumps(e) + '\n')
    binary = PREV / 'bin/sopt'
    expected_hash = json.loads((PREV / 'quality-manifest.json').read_text())['binary_sha256']
    assert digest(binary) == expected_hash
    save(folder / 'quality-manifest.json', dict(binary=str(binary), binary_sha256=expected_hash,
        effort=0, threads=1, sample_count=len(entries), sample_sha256=digest(folder / 'sample.ms'),
        selection='10,000 lexicographically smallest SHA-256 candidate IDs; one raw per identity'))
    command = [str(binary), '-sopteffort', '0', '-inputpolys', str(folder / 'sample.ms')]
    print(f'{ad}: CPU sopt started', flush=True)
    start = time.monotonic()
    with (folder / 'sopt.out').open('x') as out, (folder / 'sopt.err').open('x') as err:
        p = subprocess.run(command, stdout=out, stderr=err, env=dict(os.environ, OMP_NUM_THREADS='1'))
    save(folder / 'quality-status.json', dict(command=command, returncode=p.returncode,
        seconds=time.monotonic() - start))
    assert p.returncode == 0
    print(f'{ad}: CPU sopt finished', flush=True)


def quantile(values, q):
    pos = (len(values) - 1) * q
    i, f = int(pos), pos - int(pos)
    return values[i] * (1-f) + values[min(i+1, len(values)-1)] * f


def stats(values):
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), best=values[0],
        q01=quantile(values, .01), q10=quantile(values, .1), median=quantile(values, .5))


def summarize(ad):
    folder = HERE / str(ad)
    audit_data = json.loads((folder / 'audit.json').read_text())
    counts, summary = audit_data['counts'], audit_data['summary']
    raw_hits, exposure = Counter(), Counter()
    unique = repeated = ratio_max = 0
    with (folder / 'ledger.jsonl').open() as f:
        for e in map(json.loads, f):
            unique += 1
            repeated += len(e['discoveries']) - 1
            for d in e['discoveries']:
                ratio_max = max(ratio_max, d['ratio'])
                raw_hits[d['q'] >> 30] += 1
    for e in summary['exposure']:
        exposure[e['q_min'] >> 30] += e['qroots']
    with open_evidence(folder / 'sample.jsonl') as f:
        candidates = {e['candidate_id']: e for e in map(json.loads, f)}
    assert join_sopt(candidates, [folder / 'sopt.out']) == 0
    qvalues, timevalues, scores = defaultdict(list), defaultdict(list), []
    ambiguous_q = ambiguous_time = 0
    with (folder / 'scores.jsonl').open('x') as out:
        for e in candidates.values():
            assert len(e['sopt']) == 1
            opt = e['sopt'][0]['polynomial']
            assert resultant(opt) % int(e['N']) == 0
            score = opt['exp_E']
            scores.append(score)
            qb = {d['q'] >> 30 for d in e['discoveries']}
            tb = {min(3, (d['batch']-1)*4//counts['batches']) for d in e['discoveries']}
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
    result = dict(returned_hits=counts['returned_hits'], unique_candidates=unique,
        repeat_discoveries=repeated, repeat_fraction=repeated/counts['returned_hits'],
        admitted_hits=counts['admitted_hits'], qroots=counts['qroots'],
        hits_per_million_qroots=counts['returned_hits']/counts['qroots']*1e6,
        unique_per_million_qroots=unique/counts['qroots']*1e6,
        gate_ratio_max=ratio_max, coefficient=summary['coefficients'][0], sopt=stats(scores),
        q_bins=[dict(bin=b, qroots=exposure[b], raw_hits=raw_hits[b],
            hits_per_million_qroots=raw_hits[b]/exposure[b]*1e6, sample=stats(qvalues[b])) for b in range(4)],
        chronology_bins=[dict(bin=b, sample=stats(timevalues[b])) for b in range(4)],
        sampled_identities_spanning_q_bins=ambiguous_q,
        sampled_identities_spanning_chronology_bins=ambiguous_time)
    save(folder / 'comparison.json', result)
    print(f'{ad}: validated scores {json.dumps(result["sopt"])}', flush=True)


def combine():
    result = json.loads((THIRD / 'comparison.json').read_text())
    with open_evidence(PREV / 'scores-138600.jsonl') as f:
        baseline = [e['exp_E'] for e in map(json.loads, f)]
    rng = random.Random(660)
    for ad in ADS:
        folder = HERE / str(ad)
        result[str(ad)] = json.loads((folder / 'comparison.json').read_text())
        with open_evidence(folder / 'scores.jsonl') as f:
            scores = [e['exp_E'] for e in map(json.loads, f)]
        differences = defaultdict(list)
        for _ in range(400):
            a, b = (sorted(rng.choices(s, k=len(s))) for s in (baseline, scores))
            differences['mean'].append(statistics.mean(b)-statistics.mean(a))
            for name, q in (('median', .5), ('q01', .01), ('q10', .1)):
                differences[name].append(quantile(b, q)-quantile(a, q))
        result[f'bootstrap_{ad}_minus_138600'] = {k: dict(
            estimate=result[str(ad)]['sopt'][k]-result['138600']['sopt'][k],
            percentile95=[quantile(sorted(v), .025), quantile(sorted(v), .975)])
            for k, v in differences.items()}
    save(HERE / 'comparison.json', result)
    print('six-coefficient comparison saved', flush=True)


if __name__ == '__main__':
    snapshot()
    for ad in ADS:
        analyze(ad)
        summarize(ad)
    combine()
