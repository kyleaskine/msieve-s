#!/usr/bin/env python3
"""
Invert C coefficients in single-line msieve format
Format: c6 c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm (deg 6, 12 columns)
     or c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm (deg 5, 11 columns)
"""

import sys

def invert_line(line):
    """Invert C coefficients in a single line"""
    parts = line.strip().split()

    if len(parts) == 12:
        # Degree 6: c6 c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm
        # Invert first 7 values (c6 through c0)
        c_coeffs = parts[:7]
        rest = parts[7:]

        inverted_c = []
        for c in c_coeffs:
            if c.startswith('-'):
                inverted_c.append(c[1:])  # Remove minus
            else:
                inverted_c.append('-' + c)  # Add minus

        return ' '.join(inverted_c + rest)

    elif len(parts) == 11:
        # Degree 5: c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm
        # Invert first 6 values (c5 through c0)
        c_coeffs = parts[:6]
        rest = parts[6:]

        inverted_c = []
        for c in c_coeffs:
            if c.startswith('-'):
                inverted_c.append(c[1:])  # Remove minus
            else:
                inverted_c.append('-' + c)  # Add minus

        return ' '.join(inverted_c + rest)

    else:
        # Unknown format, return unchanged
        return line

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
