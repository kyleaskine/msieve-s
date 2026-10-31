"""Compare declared polynomial configurations with proposed skews.

Keep every declared/proposed skew through independent validation. A skew search
is a proposal, not permission to replace a better exported configuration.
"""
import copy
import math

from polyfmt import coeffs, iter_msieve_p
import score_polys as scores


def polynomial_key(poly):
    sign = 1 if coeffs(poly)[-1] > 0 else -1
    return poly['n'], poly['Y0'], poly['Y1'], *(sign * c for c in coeffs(poly))


def normalize_row(row, n=None):
    """Use primitive algebraic coefficients for every producer; keep provenance.

    source_content reconstructs the original coefficients. Drop norm/alpha
    metadata that described them, while retaining the declared skew.
    """
    original = row['poly']
    n = original.get('n') if n is None else n
    if n is None or n <= 1 or original.get('n', n) != n:
        raise ValueError('invalid polynomial or mixed input n')
    cs = coeffs(original)
    content = math.gcd(*cs)
    if not content or math.gcd(content, n) != 1:
        raise ValueError('polynomial content is zero or shares a factor with N')
    poly = dict(n=n, Y0=original['Y0'], Y1=original['Y1'],
                **{f'c{i}': c // content for i, c in enumerate(cs)})
    if 'skew' in original:
        poly['skew'] = original['skew']
    return dict(row, poly=poly, source_content=row.get('source_content', 1) * content)


def msieve_shortlist(path, n, per_family=3):
    """Keep printed-score leaders plus every distinct content-reduced export.

    Msieve's printed score can hide the best primitive polynomial when the
    exported coefficients have a common factor. Those candidates must reach
    common scoring regardless of their printed rank (the C150 regression).
    """
    families, content_rows = {}, {}
    for index, original in enumerate(iter_msieve_p(path)):
        row = normalize_row(dict(index=index, reported_murphy=original['e'], poly=original), n)
        poly, content = row['poly'], row['source_content']
        key = polynomial_key(poly), poly.get('skew')
        family = families.setdefault(poly['Y1'], {})
        family.setdefault(key, row)
        if len(family) > per_family:
            del family[min(family, key=lambda k: family[k]['reported_murphy'])]
        if content > 1:
            content_rows.setdefault(key, row)
    selected = {k:r for group in families.values() for k,r in group.items()}
    for key, row in content_rows.items():
        selected.setdefault(key, row)
    return list(selected.values())


def score_candidates(rows, n, args, *, propose=False):
    """Return scores per row, retaining each metric's chosen skew explicitly.

    propose=True discovers Murphy skews under standard and job bounds, then
    screens those and the original declared skew. propose=False independently
    scores the same frozen configuration set without another search.
    """
    rows = [normalize_row(row, n) for row in rows]
    params = copy.copy(args)
    job_params = params.params
    proposals = {}
    if propose:
        unique = {polynomial_key(r['poly']): r['poly'] for r in rows}
        polys = list(unique.values())
        params.useskew = False
        params.params = 'default'
        standard = scores.murphy(polys, n, params)
        params.params = job_params
        lattice = scores.lattice_scores(polys, n, params)
        proposals = {key: [m[1], l[1]] for key, m, l in zip(unique, standard, lattice)}

    configurations, row_keys = {}, []
    for row in rows:
        poly = row['poly']
        declared = poly.get('skew')
        skews = [declared] if declared is not None else []
        skews += proposals[polynomial_key(poly)] if propose else row.get('skew_candidates', [])
        keys = []
        for skew in skews:
            if not math.isfinite(skew) or skew <= 0:
                raise ValueError('final comparison needs finite positive declared/proposed skews')
            key = polynomial_key(poly), skew
            if key not in keys:
                keys.append(key)
                configurations.setdefault(key, dict(poly, skew=skew))
        if not keys:
            raise ValueError('no configurations retained for final comparison')
        row_keys.append(keys)

    polys = list(configurations.values())
    params.useskew = True
    params.params = 'default'
    standard = scores.murphy(polys, n, params)
    params.params = job_params
    lattice = scores.lattice_scores(polys, n, params)
    values = {}
    for key, m, l in zip(configurations, standard, lattice):
        if m[1] != key[1] or l[1] != key[1]:
            raise ValueError('fixed-configuration scorer changed the declared skew')
        if not all(math.isfinite(x) and x > 0 for x in (m[0], l[0], l[2])):
            raise ValueError('invalid final comparison score')
        values[key] = dict(standard_murphy=m[0], lattice_score=l[0], job_murphy=l[2], per_q=l[3])

    result = []
    for row, keys in zip(rows, row_keys):
        # Stable ties prefer the declared configuration, which is first.
        best_m = max(keys, key=lambda k: values[k]['standard_murphy'])
        best_l = max(keys, key=lambda k: values[k]['lattice_score'])
        declared = row['poly'].get('skew')
        result.append(dict(row, skew_candidates=[k[1] for k in keys],
            standard_murphy=values[best_m]['standard_murphy'], standard_skew=best_m[1],
            lattice_score=values[best_l]['lattice_score'], lattice_skew=best_l[1],
            job_murphy=values[best_l]['job_murphy'], per_q=values[best_l]['per_q'],
            declared_scores=dict(values[keys[0]], skew=declared) if declared is not None else None))
    return result


def winner_configuration(row, metric):
    field = 'standard_skew' if metric == 'standard_murphy' else 'lattice_skew'
    return dict(row, poly=dict(row['poly'], skew=row[field]))
