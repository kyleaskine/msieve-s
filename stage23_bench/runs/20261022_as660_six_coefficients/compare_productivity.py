#!/usr/bin/env python3
"""Reproduce the review's elapsed-time normalization using saved CPU samples.

No GPU work or production-file writes. Intervals include sampling uncertainty
only; recorded coefficient elapsed time is not an isolated GPU-busy duration.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / '20261021_as660_two_coefficients'
THIRD = HERE.parent / '20261021_as660_third_coefficient'
sys.path.insert(0, str(HERE.parents[2] / 'stage23_bench/tools'))
from run_evidence import open_evidence


def sha256(path):
    with open_evidence(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    comparison = json.loads((HERE / 'comparison.json').read_text())
    positions = json.loads((HERE / 'position-comparison.json').read_text())
    paths = {
        '138600': (PREV/'sopt-138600.out', PREV/'scores-138600.jsonl'),
        '277200': (PREV/'sopt-277200.out', PREV/'scores-277200.jsonl'),
        '415800': (THIRD/'sopt-415800.out', THIRD/'scores-415800.jsonl'),
    }
    paths.update({str(ad): (HERE/str(ad)/'sopt.out', HERE/str(ad)/'scores.jsonl')
                  for ad in (554400, 693000, 831600)})
    report = {'coefficients': {}, 'input_sha256': {
        name: sha256(HERE/name) for name in ('comparison.json', 'position-comparison.json')},
        'limitations': [
            'Coefficient elapsed hours, including host and logging costs, not isolated GPU busy hours or full pipeline hours.',
            '138600 used CLOCK_MONOTONIC under contention; later runs use CLOCK_MONOTONIC_RAW. Do not rank the first rate against later rates.',
            'No speculative correction for first-run contention or clock drift is applied.',
            'Wilson intervals reflect sample uncertainty conditional on measured time, not timing uncertainty or future-run variability.',
            'Inclusive cutoffs use printed exp_E precision and were chosen after preliminary analysis.',
            'Rational lognorm is an L2 diagnostic up to an additive constant, evaluated at the printed algebraic skew; it is not Murphy E.',
            'Q-root throughput and entry throughput do not measure equal downstream quality or prove a causal coefficient effect.',
        ]}
    for ad, (sopt_path, scores_path) in paths.items():
        c = comparison[ad]
        seconds = c['coefficient']['ending']['seconds']
        q_rate = c['qroots'] / seconds
        with open_evidence(scores_path) as f:
            scores = list(map(json.loads, f))
        assert len(scores) == 10000
        row = dict(seconds=seconds,
                   clock='CLOCK_MONOTONIC' if ad == '138600' else 'CLOCK_MONOTONIC_RAW',
                   comparable_timing=ad != '138600',
                   qroots_per_second=q_rate,
                   entries_per_second=q_rate*c['coefficient']['setup']['entries_per_q'],
                   thresholds={})
        for cutoff in (56.0, 57.0):
            source = positions['exploratory_threshold_yield'][str(cutoff)][ad]
            count = sum(s['exp_E'] <= cutoff for s in scores)
            assert count == source['sample_count_at_or_below']
            factor = q_rate * 3600 / 1e6
            estimate = source['estimated_candidates_per_million_qroots'] * factor
            direct = count/len(scores) * c['unique_candidates'] / seconds * 3600
            assert math.isclose(estimate, direct, rel_tol=1e-12)
            row['thresholds'][str(cutoff)] = dict(sample_count=count,
                estimated_distinct_candidates_per_elapsed_hour=estimate,
                approximate_wilson95=[v*factor for v in source['approximate_wilson95']])
        # Cross-check the rational side without adding alpha a second time to exp_E.
        blocks = re.split(r'### Size-optimized polynomial \(\d+\) ###', sopt_path.read_text())[1:]
        assert len(blocks) == len(scores)
        rational, log_skew = [], []
        q_groups = {0: [], 3: []}
        for block, score in zip(blocks, scores):
            block = block.split('### Input raw')[0]
            fields = dict(re.findall(r'^(Y\d|skew): (-?[\d.]+)', block, re.M))
            exp_e = float(re.search(r'exp_E ([\d.]+)', block).group(1))
            skew = float(fields['skew'])
            assert exp_e == score['exp_E'] and math.isclose(skew, score['skew'], rel_tol=.001)
            y1, y0 = int(fields['Y1']), int(fields['Y0'])
            rat = .5*math.log(y1*y1*skew + y0*y0/skew)
            rational.append(rat)
            log_skew.append(math.log(skew))
            if score['q_bins'] in ([0], [3]):
                q_groups[score['q_bins'][0]].append((exp_e, rat))
        row['fairness_diagnostic'] = dict(mean_rational_lognorm=statistics.mean(rational),
            mean_log_skew=statistics.mean(log_skew),
            q_high_minus_low_rational_lognorm=statistics.mean(r[1] for r in q_groups[3])
                - statistics.mean(r[1] for r in q_groups[0]),
            q_high_minus_low_exp_E_plus_rational=statistics.mean(sum(r) for r in q_groups[3])
                - statistics.mean(sum(r) for r in q_groups[0]))
        row['input_sha256'] = {str(p.relative_to(HERE.parent)): sha256(p) for p in (sopt_path, scores_path)}
        report['coefficients'][ad] = row
    with (HERE/'productivity-comparison.json').open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    for ad, row in report['coefficients'].items():
        a, b = (row['thresholds'][k] for k in ('56.0', '57.0'))
        print(ad, f"q roots/s={row['qroots_per_second']:.0f}",
              f"<=56/hour={a['estimated_distinct_candidates_per_elapsed_hour']:.0f}",
              f"95%={[round(v) for v in a['approximate_wilson95']]}",
              f"<=57/hour={b['estimated_distinct_candidates_per_elapsed_hour']:.0f}")


if __name__ == '__main__':
    main()
