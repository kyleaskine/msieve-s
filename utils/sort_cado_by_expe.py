#!/usr/bin/env python3
"""
Sort CADO sopt output by exp_E (expected difficulty)
Parses multi-line polynomial blocks and sorts them by exp_E value
"""

import sys

def parse_cado_output(filename):
    """Parse CADO sopt output into polynomial blocks with exp_E values"""
    polynomials = []
    current_poly = []
    current_exp_e = None
    in_poly = False

    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            # Check if this is the start of a new size-optimized polynomial
            if line_stripped.startswith('### Size-optimized polynomial'):
                # Save previous polynomial if exists
                if current_poly and current_exp_e is not None:
                    polynomials.append((current_exp_e, '\n'.join(current_poly)))

                # Start new polynomial
                current_poly = [line_stripped]
                current_exp_e = None
                in_poly = True

            # Check for blank line - marks end of polynomial block
            elif in_poly and line_stripped == '':
                # Don't add the blank line to the polynomial
                # It will be added back when we write output
                if current_poly and current_exp_e is not None:
                    polynomials.append((current_exp_e, '\n'.join(current_poly)))
                    current_poly = []
                    current_exp_e = None
                    in_poly = False

            # Add line to current polynomial
            elif in_poly:
                current_poly.append(line_stripped)

                # Extract exp_E value from line like:
                # "# side 1 lognorm 63.22, exp_E 55.89, alpha -1.48 (proj -2.55), 4 real roots"
                if 'exp_E' in line_stripped:
                    try:
                        # Split by comma and find the exp_E part
                        parts = line_stripped.split(',')
                        for part in parts:
                            if 'exp_E' in part:
                                # Extract the number after "exp_E"
                                exp_e_str = part.split('exp_E')[1].strip().split()[0]
                                current_exp_e = float(exp_e_str)
                                break
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse exp_E from line: {line_stripped}", file=sys.stderr)
                        print(f"  Error: {e}", file=sys.stderr)

        # Don't forget the last polynomial
        if current_poly and current_exp_e is not None:
            polynomials.append((current_exp_e, '\n'.join(current_poly)))

    return polynomials

def main():
    if len(sys.argv) != 3:
        print("Usage: sort_cado_by_expe.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Parsing CADO output from {input_file}...")
    polynomials = parse_cado_output(input_file)

    if not polynomials:
        print("Error: No polynomials found in input file", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(polynomials)} polynomials")

    # Sort by exp_E (lower is better)
    print("Sorting by exp_E (lower is better)...")
    polynomials.sort(key=lambda x: x[0])

    # Get statistics
    best_exp_e = polynomials[0][0]
    worst_exp_e = polynomials[-1][0]
    print(f"exp_E range: {best_exp_e:.6f} (best) to {worst_exp_e:.6f} (worst)")

    # Write sorted output
    print(f"Writing sorted output to {output_file}...")
    with open(output_file, 'w') as f:
        for i, (_, poly_text) in enumerate(polynomials):
            f.write(poly_text)
            # Add blank line separator between polynomials (except after last)
            if i < len(polynomials) - 1:
                f.write('\n\n')

    print(f"Successfully wrote {len(polynomials)} sorted polynomials")

if __name__ == '__main__':
    main()
