#!/usr/bin/env python3
"""
Invert C coefficients in msieve polynomial format
Flips the sign of all c0, c1, c2, c3, c4, c5, c6 coefficients
Leaves n, Y0, Y1, and skew unchanged
"""

import sys

def invert_polynomial(poly_lines):
    """Invert C coefficients in a polynomial block"""
    inverted = []

    for line in poly_lines:
        if line.startswith('c') and ':' in line:
            # This is a C coefficient line like "c0: -123456"
            parts = line.split(':', 1)
            if len(parts) == 2:
                coef_name = parts[0].strip()
                coef_value = parts[1].strip()

                # Flip the sign
                if coef_value.startswith('-'):
                    new_value = coef_value[1:]  # Remove the minus
                else:
                    new_value = '-' + coef_value  # Add minus

                inverted.append(f"{coef_name}: {new_value}")
            else:
                inverted.append(line)
        else:
            # Keep n, Y0, Y1, skew, etc. unchanged
            inverted.append(line)

    return inverted

def process_file(input_file, output_file):
    """Process entire file, inverting all polynomials"""
    polynomials = []
    current_poly = []

    with open(input_file, 'r') as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            if line_stripped == '':
                # Blank line marks end of polynomial
                if current_poly:
                    inverted = invert_polynomial(current_poly)
                    polynomials.append(inverted)
                    current_poly = []
            else:
                current_poly.append(line_stripped)

        # Don't forget last polynomial
        if current_poly:
            inverted = invert_polynomial(current_poly)
            polynomials.append(inverted)

    # Write output
    with open(output_file, 'w') as f:
        for i, poly in enumerate(polynomials):
            for line in poly:
                f.write(line + '\n')
            # Add blank line separator between polynomials (except after last)
            if i < len(polynomials) - 1:
                f.write('\n')

    return len(polynomials)

def main():
    if len(sys.argv) != 3:
        print("Usage: invert_c_coefficients.py <input_file> <output_file>", file=sys.stderr)
        print("  Flips the sign of all C coefficients (c0-c6)", file=sys.stderr)
        print("  Leaves n, Y0, Y1, skew unchanged", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading polynomials from {input_file}...")
    count = process_file(input_file, output_file)
    print(f"Inverted {count} polynomials")
    print(f"Wrote to {output_file}")

if __name__ == '__main__':
    main()
