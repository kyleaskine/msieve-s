#!/usr/bin/env python3
"""Exploratory within-coefficient comparisons of q and discovery quarters."""
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / '20261021_as660_two_coefficients'
THIRD = HERE.parent / '20261021_as660_third_coefficient'
sys.path.insert(0, str(HERE.parents[2] / 'stage23_bench/tools'))
from run_evidence import open_evidence


def quantile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    i, f = int(pos), pos-int(pos)
    return values[i]*(1-f) + values[min(i+1, len(values)-1)]*f


def main():
    paths = {138600: PREV/'scores-138600.jsonl', 277200: PREV/'scores-277200.jsonl',
             415800: THIRD/'scores-415800.jsonl'}
    paths.update({ad: HERE/str(ad)/'scores.jsonl' for ad in (554400, 693000, 831600)})
    data = {}
    for ad, path in paths.items():
        with open_evidence(path) as f:
            data[ad] = list(map(json.loads, f))
        assert len(data[ad]) == 10000
    result = {}
    rng = random.Random(66022)
    for label, key in [('q', 'q_bins'), ('chronology', 'chronology_bins')]:
        groups, per_coefficient = [], {}
        for ad, entries in data.items():
            a = [e['exp_E'] for e in entries if e[key] == [0]]
            b = [e['exp_E'] for e in entries if e[key] == [3]]
            groups.append((a, b))
            per_coefficient[str(ad)] = dict(first_n=len(a), last_n=len(b),
                first_mean=statistics.mean(a), last_mean=statistics.mean(b),
                mean_difference=statistics.mean(b)-statistics.mean(a),
                first_median=statistics.median(a), last_median=statistics.median(b))
        bootstrap = []
        for _ in range(400):
            differences = []
            for a, b in groups:
                differences.append(sum(rng.choices(b, k=len(b)))/len(b)
                                   - sum(rng.choices(a, k=len(a)))/len(a))
            bootstrap.append(statistics.mean(differences))
        result[label] = dict(per_coefficient=per_coefficient,
            equally_weighted_mean_difference=statistics.mean(
                e['mean_difference'] for e in per_coefficient.values()),
            stratified_bootstrap95=[quantile(bootstrap, .025), quantile(bootstrap, .975)])
    comparison = json.loads((HERE/'comparison.json').read_text())
    threshold_yield = {}
    for cutoff in (56.0, 57.0):
        rows = {}
        for ad, entries in data.items():
            n = len(entries)
            count = sum(e['exp_E'] <= cutoff for e in entries)
            fraction = count/n
            z2 = 1.96**2
            center = (fraction + z2/(2*n))/(1+z2/n)
            half = 1.96*math.sqrt(fraction*(1-fraction)/n+z2/(4*n*n))/(1+z2/n)
            density = comparison[str(ad)]['unique_per_million_qroots']
            rows[str(ad)] = dict(sample_count_at_or_below=count, sample_size=n,
                estimated_candidates_per_million_qroots=density*fraction,
                approximate_wilson95=[density*(center-half), density*(center+half)])
        threshold_yield[str(cutoff)] = rows
    result['exploratory_threshold_yield'] = threshold_yield
    result['interpretation'] = [
        'Last minus first quarter; lower exp_E is better. Each of the six coefficients gets equal weight.',
        'Only candidates whose discoveries all belong to one quarter enter that quarter.',
        'Bootstrap independently resamples candidates within each coefficient/quarter; 400 resamples.',
        'Intervals describe sample uncertainty for these six fixed coefficients, not variation across future coefficients.',
        'Exploratory associations, without multiplicity correction or a causal claim about q magnitude.',
        'Mean-score comparisons do not establish equality of extreme tails or final root-optimized quality.',
        'Chronology is batch traversal order, not q magnitude or elapsed wall time.',
        'Exploratory sopt thresholds 56.0 and 57.0 were selected after seeing preliminary summaries; comparisons are not confirmatory.',
        'Threshold yield multiplies the uniform distinct-candidate sample fraction by full-corpus distinct candidates per million q roots.',
        'Wilson intervals approximate sample uncertainty using a binomial model; finite-population correction is omitted.',
        'Threshold yield is not a runtime-normalized or final Murphy-E productivity measure.']
    with (HERE/'position-comparison.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
