#!/usr/bin/env python3
"""
Extract the top N size-optimized polynomials from CADO sopt output
"""

import sys

def extract_top_polys(filename, n=1):
    """Extract top N size-optimized polynomials from sorted CADO output"""
    polynomials = []
    current_poly = []
    in_poly = False

    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            # Start of size-optimized polynomial
            if line_stripped.startswith('### Size-optimized polynomial'):
                if current_poly:
                    polynomials.append('\n'.join(current_poly))
                    if len(polynomials) >= n:
                        break
                current_poly = [line_stripped]
                in_poly = True
                continue

            # Blank line marks end
            if in_poly and line_stripped == '':
                if current_poly:
                    polynomials.append('\n'.join(current_poly))
                    if len(polynomials) >= n:
                        break
                current_poly = []
                in_poly = False
                continue

            # Collect lines
            if in_poly:
                current_poly.append(line_stripped)

        # Don't forget last polynomial
        if current_poly and len(polynomials) < n:
            polynomials.append('\n'.join(current_poly))

    return polynomials[:n]

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: extract_top_cado_poly.py <input_file> <output_file> [n]", file=sys.stderr)
        print("  n: number of polynomials to extract (default: 1)", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) == 4 else 1

    print(f"Extracting top {n} polynomial(s) from {input_file}...")
    polynomials = extract_top_polys(input_file, n)

    if not polynomials:
        print("Error: No polynomials found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(polynomials)} polynomial(s)")

    # Write to output
    with open(output_file, 'w') as f:
        for i, poly in enumerate(polynomials):
            f.write(poly)
            if i < len(polynomials) - 1:
                f.write('\n\n')

    print(f"Wrote to {output_file}")

if __name__ == '__main__':
    main()
