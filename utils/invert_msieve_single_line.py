#!/usr/bin/env python3
"""
Invert C coefficients in single-line msieve format
Format: c_d .. c0 Y1 Y0 alpha exp_E norm (d+6 columns; 11 for degree 5, 12 for degree 6)
"""

import sys

def invert_line(line):
    """Invert the C coefficients of one line: c_d .. c0 Y1 Y0 alpha exp_E norm, any degree
    (the last five fields are kept as they are)."""
    parts = line.strip().split()
    if len(parts) < 7:
        return line.rstrip('\n')
    c_coeffs, rest = parts[:-5], parts[-5:]
    inverted_c = [c[1:] if c.startswith('-') else '-' + c for c in c_coeffs]
    return ' '.join(inverted_c + rest)


def main():
    if len(sys.argv) != 3:
        print("Usage: invert_msieve_single_line.py <input_file> <output_file>", file=sys.stderr)
        print("  Inverts C coefficients in single-line msieve format", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading from {input_file}...")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} line(s)...")

    inverted_lines = [invert_line(line) for line in lines]

    with open(output_file, 'w') as f:
        for line in inverted_lines:
            f.write(line + '\n')

    print(f"Wrote inverted polynomials to {output_file}")

if __name__ == '__main__':
    main()
