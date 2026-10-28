#!/usr/bin/env python3
"""CPU-only, streamed review of the completed AS660 CPU pipeline."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT/'stage23_gpu/runs/20261022_three_jobs/c196'
CPU = FROZEN/'cpu_reference'
sys.path.insert(0, str(ROOT/'stage23_bench/tools'))
from polyfmt import iter_sopt, resultant, rotation_between, parse_cado_ropt
from stage1_provenance import candidate_id
from run_evidence import evidence_exists, open_evidence


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def key(p):
    return tuple(p[k] for k in ('n', 'Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5'))


def save(name, data):
    with (HERE/name).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def msieve_best(path):
    # Read one block at a time instead of keeping 300K output polynomials.
    total, best, cur = 0, [], None
    for line in path.open():
        if line.startswith('# norm '):
            m = re.search(r' e (\S+)', line)
            cur = {'MurphyE': float(m.group(1))}
        elif cur is not None and ':' in line:
            k, v = line.strip().split(':', 1)
            if k in ('Y0', 'Y1') or re.fullmatch(r'c\d+', k):
                cur[k] = int(v)
            elif k == 'skew':
                cur[k] = float(v)
            if k == 'Y1':
                total += 1
                item = (cur['MurphyE'], -total, cur)
                if len(best) < 10:
                    heapq.heappush(best, item)
                elif item[:2] > best[0][:2]:
                    heapq.heapreplace(best, item)
                cur = None
    return total, [p for _, _, p in sorted(best, reverse=True)]


def main():
    manifest = json.loads((FROZEN/'prepared.json').read_text())
    # Verify immutable CPU references, not mutable GPU outputs.
    for filename, expected in manifest['cpu_sha256'].items():
        assert digest(CPU/filename) == expected, filename
    print('Frozen CPU reference hashes verified', flush=True)
    if evidence_exists(HERE/'corpus-checkpoint.json'):
        with open_evidence(HERE/'corpus-checkpoint.json') as stream:
            checkpoint = json.load(stream)
        assert digest(ROOT/'pipeline_work/resopt_output.txt') == checkpoint['summary']['resopt_input_sha256']
        finish(manifest, checkpoint['summary'], checkpoint['resopt'],
               {int(i): r for i,r in checkpoint['selected'].items()}, checkpoint['ordered'])
        return
    resopt_path = ROOT/'pipeline_work/resopt_output.txt'
    resopt_hash = digest(resopt_path)
    resopt = list(iter_sopt(resopt_path))
    assert len(resopt) == 1000
    targets = defaultdict(list)
    for i, (raw, opt) in enumerate(resopt):
        assert resultant(opt) % manifest['n'] == 0
        targets[key(raw)].append(i)
    counts, hist, seen, selected, top = Counter(), defaultdict(Counter), set(), {}, []
    duplicate_counts = Counter()
    header_counts = {}
    for index, (raw, opt) in enumerate(iter_sopt(CPU/'cado_sopt_unsorted.txt', header_counts)):
        assert raw['n'] == opt['n'] == manifest['n']
        assert resultant(opt) % manifest['n'] == 0
        ad = raw['c5']
        ident = candidate_id((raw['n'], ad, raw['Y1'], -raw['Y0']))
        binary_id = bytes.fromhex(ident)
        if binary_id in seen:
            duplicate_counts[ad] += 1
            continue
        seen.add(binary_id)
        counts[ad] += 1
        hist[ad][round(opt['exp_E']*100)] += 1
        record = dict(candidate_id=ident, ad=ad, index=index, exp_E=opt['exp_E'], raw=raw, opt=opt)
        item = (-opt['exp_E'], -index, record)
        if len(top) < 1000:
            heapq.heappush(top, item)
        elif item[:2] > top[0][:2]:
            heapq.heapreplace(top, item)
        for i in targets.get(key(raw), []):
            assert i not in selected
            selected[i] = record
        if (index+1) % 250000 == 0:
            print(f'Scored pairs checked: {index+1}', flush=True)
    assert header_counts['raw'] == header_counts['opt'] == index+1
    assert len(selected) == 1000
    assert digest(resopt_path) == resopt_hash
    old = json.loads((HERE.parent/'20261022_as660_six_coefficients/comparison.json').read_text())
    for ad in (138600,277200,415800,554400,693000,831600):
        assert counts[ad] == old[str(ad)]['unique_candidates']
    timing = {int(ad): r['coefficient']['ending'] for ad,r in old.items() if ad.isdigit()}
    # Only extract already-completed metadata; do not re-audit huge sidecars here.
    for ad, name in ((180180, 'as660-stage1-d.w0.jsonl'), (360360, 'as660-stage1-e.w0.jsonl')):
        archived_audit = HERE.parent/'20261023_stage1_archive'/f'{ad}.audit.json'
        if archived_audit.exists():
            record = json.loads(archived_audit.read_text())['summary']['coefficients'][0]
            assert int(record['parameters']['ad']) == ad and record['ending']['complete']
            timing[ad] = record['ending']
            continue
        with (ROOT/name).open('rb') as f:
            header = json.loads(f.readline())
            coefficient = json.loads(f.readline())
            assert int(coefficient['ad']) == ad
            f.seek(-4096, 2)
            tail = f.read().splitlines()
        assert json.loads(tail[-1])['type'] == 'worker_end'
        end = json.loads(tail[-2])
        assert end['type'] == 'coefficient_end' and end['complete']
        timing[ad] = end
    ordered = [r for _,_,r in sorted(top, reverse=True)]
    top_counts = {str(k): dict(Counter(r['ad'] for r in ordered[:k])) for k in (100,300,1000)}
    summary = dict(total_pairs=index+1, distinct_identities=len(seen),
        duplicate_identity_counts=dict(duplicate_counts), top_counts=top_counts,
        top300_cutoff=ordered[299]['exp_E'], top1000_cutoff=ordered[999]['exp_E'],
        top1000_cutoff_tie_population=sum(h[round(ordered[999]['exp_E']*100)] for h in hist.values()),
        header_counts=header_counts, coefficients={},
        resopt_input_sha256=resopt_hash,
        frozen_manifest_sha256=digest(FROZEN/'prepared.json'),
        interpretation=[
            'Counts are exact for printed CPU effort-0 scores in this saved corpus, not future-search expectations.',
            'Top sets use printed exp_E then original concatenated output order to break ties.',
            'Original coefficient comes from the printed raw input, not optimized c5.',
            'First coefficient timing has contention and a different clock. Other rates use recorded stage-1 elapsed seconds, not GPU busy time.',
            'The two new sidecars were checked for completed endings only; a full independent collision audit remains pending.',
            'CPU root optimization used sopt-selected seeds; it is not an unbiased comparison of coefficient populations.',
            'Scores from different evaluators/settings must be rescored together before a definitive CPU/GPU comparison.'
        ])
    for ad in sorted(counts):
        h = hist[ad]
        cuts = {str(c): sum(n for score,n in h.items() if score <= c*100) for c in (54,55,56,57)}
        summary['coefficients'][str(ad)] = dict(count=counts[ad], minimum=min(h)/100,
            thresholds=cuts, seconds=timing[ad]['seconds'],
            per_elapsed_hour={c: n*3600/timing[ad]['seconds'] for c,n in cuts.items()})
    save('corpus-checkpoint.json', dict(summary=summary, resopt=resopt,
                                      selected=selected, ordered=ordered))
    finish(manifest, summary, resopt, selected, ordered)


def finish(manifest, summary, resopt, selected, ordered):
    # Join each resopt seed to its exact raw input from the full-corpus run.
    origin_index = defaultdict(list)
    for i, (_, opt) in enumerate(resopt):
        origin_index[opt['Y1']].append((selected[i], opt))
    def origins(poly):
        assert resultant(poly) % manifest['n'] == 0
        matches = []
        for rec, seed in origin_index[poly['Y1']]:
            proof = rotation_between(seed, poly)
            if proof is None:
                # CADO can primitive-normalize f or otherwise change its
                # multiplier. Prove the relation after clearing denominators;
                # an unchanged Y1 alone is never accepted as a provenance join.
                scale = Fraction(abs(poly['c5']), abs(seed['c5']))
                a, b = dict(seed), dict(poly)
                for k in range(6):
                    a[f'c{k}'] *= scale.numerator
                    b[f'c{k}'] *= scale.denominator
                proof = rotation_between(a, b)
                if proof is not None:
                    proof.update(seed_multiplier=scale.numerator,
                                 output_multiplier=scale.denominator,
                                 identity='(-1 if negated else 1) * output_multiplier * output = seed_multiplier * translated_seed + rotation * g')
            if proof is not None:
                matches.append(dict(candidate_id=rec['candidate_id'], ad=rec['ad'],
                                    effort0_exp_E=rec['exp_E'], effort50_exp_E=seed['exp_E'], proof=proof))
        assert matches, poly
        return matches
    results = {}
    for side in ('orig', 'inv'):
        polys = parse_cado_ropt(CPU/f'cado_ropt_{side}.txt')
        assert len(polys) == 150
        joined = [dict(polynomial=p, origins=origins(p)) for p in polys]
        joined.sort(key=lambda e:e['polynomial']['MurphyE'], reverse=True)
        results['cado_'+side] = joined
        total, best = msieve_best(CPU/f'msieve_ropt_{side}.p')
        results['msieve_'+side] = dict(total=total, top10=[dict(polynomial=p,origins=origins(p)) for p in best])
    save('root-results.json', results)
    with (HERE/'top1000.jsonl').open('x') as f:
        for rec in ordered:
            f.write(json.dumps(rec)+'\n')
    save('summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    for name, entries in results.items():
        best = entries[0] if isinstance(entries,list) else entries['top10'][0]
        print(name, best['polynomial']['MurphyE'], best['origins'], flush=True)


if __name__ == '__main__':
    main()
