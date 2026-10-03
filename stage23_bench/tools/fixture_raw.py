#!/usr/bin/env python3
"""Write the raw polynomials of a fixture as CADO-format input for any size optimizer.

  fixture_raw.py stage23_bench/data/c146/sopt_sample.tsv.gz OUT.poly [--set top|random|all]

The output is what CADO's sopt (`sopt -inputpolys OUT.poly`) or a GPU sopt reads; compare
its result against the fixture with `sopt_compare.py FIXTURE RESULT --rescore`.
"""

import argparse

from polyfmt import load_fixture, cado_poly_text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fixture')
    ap.add_argument('out')
    ap.add_argument('--set', choices=('top', 'random', 'all'), default='all')
    args = ap.parse_args()
    rows = [r for r in load_fixture(args.fixture) if args.set in ('all', r[3])]
    with open(args.out, 'w') as fh:
        for raw, _, _, _ in rows:
            fh.write(cado_poly_text(raw) + '\n')
    print(f"wrote {len(rows)} raw polynomials to {args.out}")


if __name__ == '__main__':
    main()
