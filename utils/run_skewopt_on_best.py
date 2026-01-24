#!/usr/bin/env python3
"""
Run skewopt on the best polynomials from CADO ropt output files.

This script:
1. Reads CADO ropt output files (original and inverted)
2. Extracts the best polynomial from each (highest MurphyE)
3. Runs them through skewopt to get optimized skew and Murphy score
4. Prints the polynomials with optimized skew values

Usage:
  run_skewopt_on_best.py --skewopt /path/to/skewopt [orig.txt inv.txt]
  run_skewopt_on_best.py [orig.txt inv.txt]  # uses default skewopt path
"""

import argparse
import subprocess
import sys
import os
import re

DEFAULT_SKEWOPT_PATH = os.path.expanduser("~/code/SkewOptimizer/skewopt")

# Global variable set by argument parsing
SKEWOPT_PATH = DEFAULT_SKEWOPT_PATH


def extract_best_polynomial(filepath):
    """
    Extract the best polynomial (highest MurphyE) from a CADO ropt output file.
    Returns a dict with the polynomial data.
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Find all MurphyE scores and their locations
    murphy_pattern = r'# side 1 MurphyE\([^)]+\)=([0-9.e+-]+)'
    murphy_matches = list(re.finditer(murphy_pattern, content))

    if not murphy_matches:
        return None

    # Find the best MurphyE
    best_murphy = 0.0
    best_match = None
    for match in murphy_matches:
        murphy_val = float(match.group(1))
        if murphy_val > best_murphy:
            best_murphy = murphy_val
            best_match = match

    if not best_match:
        return None

    # Find the polynomial block containing this MurphyE
    murphy_pos = best_match.start()

    # Search backwards for "### root-optimized polynomial"
    before_murphy = content[:murphy_pos]
    root_opt_match = None
    for m in re.finditer(r'### root-optimized polynomial \d+', before_murphy):
        root_opt_match = m

    if not root_opt_match:
        return None

    # Extract the polynomial block
    start_pos = root_opt_match.start()
    # Find the end (next ### or end of file)
    end_match = re.search(r'\n### ', content[murphy_pos:])
    if end_match:
        end_pos = murphy_pos + end_match.start()
    else:
        end_pos = len(content)

    poly_block = content[start_pos:end_pos].strip()

    # Parse the polynomial coefficients
    poly_data = {
        'block': poly_block,
        'murphy_e': best_murphy,
        'murphy_line': best_match.group(0)
    }

    # Extract individual values
    for line in poly_block.split('\n'):
        line = line.strip()
        if line.startswith('n:'):
            poly_data['n'] = line.split(':')[1].strip()
        elif line.startswith('Y0:'):
            poly_data['Y0'] = line.split(':')[1].strip()
        elif line.startswith('Y1:'):
            poly_data['Y1'] = line.split(':')[1].strip()
        elif line.startswith('c0:'):
            poly_data['c0'] = line.split(':')[1].strip()
        elif line.startswith('c1:'):
            poly_data['c1'] = line.split(':')[1].strip()
        elif line.startswith('c2:'):
            poly_data['c2'] = line.split(':')[1].strip()
        elif line.startswith('c3:'):
            poly_data['c3'] = line.split(':')[1].strip()
        elif line.startswith('c4:'):
            poly_data['c4'] = line.split(':')[1].strip()
        elif line.startswith('c5:'):
            poly_data['c5'] = line.split(':')[1].strip()
        elif line.startswith('c6:'):
            poly_data['c6'] = line.split(':')[1].strip()
        elif line.startswith('skew:'):
            poly_data['skew'] = line.split(':')[1].strip()
        elif '# side 1 lognorm' in line:
            poly_data['lognorm_line'] = line

    return poly_data


def run_skewopt(poly_data):
    """
    Run skewopt on the polynomial and return optimized skew and Murphy score.
    """
    # Build the command: skewopt y0 y1 c0 c1 c2 c3 c4 c5 c6 c7 c8
    y0 = poly_data.get('Y0', '0')
    y1 = poly_data.get('Y1', '0')
    c0 = poly_data.get('c0', '0')
    c1 = poly_data.get('c1', '0')
    c2 = poly_data.get('c2', '0')
    c3 = poly_data.get('c3', '0')
    c4 = poly_data.get('c4', '0')
    c5 = poly_data.get('c5', '0')
    c6 = poly_data.get('c6', '0')
    c7 = '0'  # Degree 7 coefficient (usually 0)
    c8 = '0'  # Degree 8 coefficient (usually 0)

    cmd = [SKEWOPT_PATH, y0, y1, c0, c1, c2, c3, c4, c5, c6, c7, c8]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None, None

        # Parse output
        output = result.stdout
        skew = None
        murphy = None
        for line in output.split('\n'):
            if line.startswith('Best Skew:'):
                skew = float(line.split(':')[1].strip())
            elif line.startswith('Murphy Score:'):
                murphy = float(line.split(':')[1].strip())

        return skew, murphy
    except Exception as e:
        print(f"Error running skewopt: {e}", file=sys.stderr)
        return None, None


def format_optimized_poly(poly_data, opt_skew, opt_murphy):
    """
    Format the polynomial with optimized skew, replacing the ### Best line.
    """
    lines = []

    # Add polynomial header
    lines.append(f"n: {poly_data.get('n', '')}")
    lines.append(f"Y0: {poly_data.get('Y0', '')}")
    lines.append(f"Y1: {poly_data.get('Y1', '')}")
    lines.append(f"c0: {poly_data.get('c0', '')}")
    lines.append(f"c1: {poly_data.get('c1', '')}")
    lines.append(f"c2: {poly_data.get('c2', '')}")
    lines.append(f"c3: {poly_data.get('c3', '')}")
    lines.append(f"c4: {poly_data.get('c4', '')}")
    lines.append(f"c5: {poly_data.get('c5', '')}")
    if 'c6' in poly_data and poly_data['c6'] != '0':
        lines.append(f"c6: {poly_data.get('c6', '')}")

    # Keep original CADO skew
    lines.append(f"skew: {poly_data.get('skew', '')}")

    # Add lognorm line if present
    if 'lognorm_line' in poly_data:
        lines.append(poly_data['lognorm_line'])

    # Add original CADO MurphyE for reference
    lines.append(f"# CADO MurphyE: {poly_data['murphy_e']:.3e}")

    # Add skewopt results
    lines.append(f"# skewopt optimized skew: {opt_skew:.5f}")
    lines.append(f"# skewopt MurphyE: {opt_murphy:.10e}")

    return '\n'.join(lines)


def main():
    global SKEWOPT_PATH

    parser = argparse.ArgumentParser(
        description='Run skewopt on best polynomials from CADO ropt output files'
    )
    parser.add_argument('--skewopt', '-s', default=DEFAULT_SKEWOPT_PATH,
                        help=f'Path to skewopt binary (default: {DEFAULT_SKEWOPT_PATH})')
    parser.add_argument('orig_file', nargs='?',
                        default=os.path.expanduser("~/msieve-s/pipeline_results/cado_ropt_orig.txt"),
                        help='CADO ropt original output file')
    parser.add_argument('inv_file', nargs='?',
                        default=os.path.expanduser("~/msieve-s/pipeline_results/cado_ropt_inv.txt"),
                        help='CADO ropt inverted output file')

    args = parser.parse_args()

    # Set global skewopt path
    SKEWOPT_PATH = os.path.expanduser(args.skewopt)

    # Check skewopt exists
    if not os.path.exists(SKEWOPT_PATH):
        print(f"Error: skewopt not found at {SKEWOPT_PATH}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("SKEWOPT OPTIMIZATION OF BEST CADO POLYNOMIALS")
    print("=" * 60)
    print()

    # Process original file
    print("1. CADO ropt (original) - Best polynomial:")
    print("-" * 40)
    poly_orig = extract_best_polynomial(args.orig_file)
    if poly_orig:
        opt_skew, opt_murphy = run_skewopt(poly_orig)
        if opt_skew and opt_murphy:
            print(format_optimized_poly(poly_orig, opt_skew, opt_murphy))
        else:
            print("  Error: skewopt failed")
    else:
        print("  (No polynomial found or file missing)")
    print()

    # Process inverted file
    print("2. CADO ropt (inverted) - Best polynomial:")
    print("-" * 40)
    poly_inv = extract_best_polynomial(args.inv_file)
    if poly_inv:
        opt_skew, opt_murphy = run_skewopt(poly_inv)
        if opt_skew and opt_murphy:
            print(format_optimized_poly(poly_inv, opt_skew, opt_murphy))
        else:
            print("  Error: skewopt failed")
    else:
        print("  (No polynomial found or file missing)")
    print()

    print("=" * 60)


if __name__ == '__main__':
    main()
