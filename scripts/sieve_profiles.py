#!/usr/bin/env python3
"""Inspect historical sieve settings without changing current-job configuration.

Use --digits N to list the nearest historical sizes (ties preserved), or
--profile ID to print a scoring.json snapshot suitable for a new run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'config/sieve_profiles.json'


def load(path=CATALOG):
    data = json.loads(path.read_text())
    if data['schema_version'] != 1:
        raise ValueError('unsupported sieve profile schema')
    ids = set()
    for p in data['profiles']:
        if p['id'] in ids: raise ValueError('duplicate profile ID: ' + p['id'])
        ids.add(p['id'])
        r, q = p['region'], p['q']
        if not (p['degree'] == 5 and p['side'] == 'algebraic' and p['digits'] > 0):
            raise ValueError('unsupported profile type: ' + p['id'])
        if not (r['points'] == 2**r['logI'] * r['J'] and r['J'] > 0
                and 3 <= q['min'] < q['max'] and q['model_samples'] >= 2):
            raise ValueError('invalid geometry: ' + p['id'])
        if not (p['lpbr'] > 1 and p['lpba'] > 1): raise ValueError('invalid LPB')
        for k in ('rlim', 'alim'):
            if p[k] is not None and p[k] <= 1: raise ValueError('invalid limit')
    return data['profiles']


def nearest(profiles, digits, include_assumed=False):
    eligible = [p for p in profiles if include_assumed or p['evidence'] != 'assumed']
    if not eligible:
        return []
    distance = min(abs(p['digits'] - digits) for p in eligible)
    return [p for p in eligible if abs(p['digits'] - digits) == distance]


def scoring(profile, catalog=CATALOG):
    p, r, q = profile, profile['region'], profile['q']
    logi = (math.log2(r['points']) + 1) / 2
    return dict(params=f"job:{p['lpbr']},{p['lpba']},{logi:g},{q['min']}",
                lattice=f"{r['logI']},{r['J']}", qband=f"{q['min']},{q['max']},{q['model_samples']}",
                lpbr=p['lpbr'], lpba=p['lpba'], rlim=p['rlim'], alim=p['alim'],
                profile_id=p['id'], profile_snapshot=p,
                catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                provisional=True, source='Inherited historical profile; not confirmed for this input.',
                note='Model comparison only; no new test sieving.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', type=Path, default=CATALOG)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument('--digits', type=int)
    choice.add_argument('--profile')
    choice.add_argument('--validate', action='store_true')
    ap.add_argument('--include-assumed', action='store_true')
    args = ap.parse_args()
    profiles = load(args.catalog)
    if args.validate:
        print(f'{len(profiles)} valid historical profiles')
    elif args.profile:
        found = [p for p in profiles if p['id'] == args.profile]
        if not found: ap.error('unknown profile: ' + args.profile)
        print(json.dumps(scoring(found[0], args.catalog), indent=2))
    else:
        if args.digits < 1: ap.error('--digits must be positive')
        matches = nearest(profiles, args.digits, args.include_assumed)
        if not matches:
            ap.error('no eligible historical profiles; choose another catalog or use --include-assumed')
        print(json.dumps(matches, indent=2))


if __name__ == '__main__': main()
