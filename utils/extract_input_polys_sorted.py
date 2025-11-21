#!/usr/bin/env python3
"""
Extract input polynomials in sorted order
Uses sorted CADO output to determine order, then finds corresponding input polys in unsorted output
"""

import sys

def parse_polynomial_identifier(poly_lines):
    """
    Extract identifying information from a polynomial block
    Use n, Y0, Y1 as unique identifier
    """
    n = None
    y0 = None
    y1 = None

    for line in poly_lines:
        if line.startswith('n:'):
            n = line.split(':', 1)[1].strip()
        elif line.startswith('Y0:'):
            y0 = line.split(':', 1)[1].strip()
        elif line.startswith('Y1:'):
            y1 = line.split(':', 1)[1].strip()

    if n and y0 and y1:
        return (n, y0, y1)
    return None

def parse_unsorted_output(filename):
    """
    Parse unsorted CADO output to build mapping of optimized poly -> input poly
    Returns dict: identifier -> input_poly_text
    """
    poly_map = {}
    current_input_poly = []
    current_optimized_poly = []
    in_input = False
    in_optimized = False

    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            # Start of input polynomial
            if line_stripped.startswith('### Input raw polynomial'):
                in_input = True
                in_optimized = False
                current_input_poly = []
                continue

            # Start of size-optimized polynomial
            if line_stripped.startswith('### Size-optimized polynomial'):
                in_input = False
                in_optimized = True
                current_optimized_poly = []
                continue

            # Blank line - end of current block
            if line_stripped == '':
                if in_optimized and current_optimized_poly and current_input_poly:
                    # Parse identifier from optimized poly
                    identifier = parse_polynomial_identifier(current_optimized_poly)
                    if identifier:
                        # Store input poly text
                        poly_map[identifier] = '\n'.join(current_input_poly)
                    current_input_poly = []
                    current_optimized_poly = []
                in_input = False
                in_optimized = False
                continue

            # Collect input polynomial lines
            if in_input:
                # Skip comment lines and metadata
                if line_stripped.startswith('# #'):
                    continue
                if any(x in line_stripped for x in ['Reading polynomials', 'Compiled with', 'Compilation flags']):
                    continue
                # Convert "# n:" to "n:", etc.
                if line_stripped.startswith('# '):
                    current_input_poly.append(line_stripped[2:])
                else:
                    current_input_poly.append(line_stripped)

            # Collect optimized polynomial lines
            if in_optimized:
                current_optimized_poly.append(line_stripped)

    # Don't forget last polynomial
    if in_optimized and current_optimized_poly and current_input_poly:
        identifier = parse_polynomial_identifier(current_optimized_poly)
        if identifier:
            poly_map[identifier] = '\n'.join(current_input_poly)

    return poly_map

def parse_sorted_output(filename, max_polys=None):
    """
    Parse sorted CADO output to get polynomial identifiers in sorted order
    Returns list of identifiers in sorted order (by exp_E)
    """
    identifiers = []
    current_poly = []
    in_poly = False

    with open(filename, 'r') as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            # Start of size-optimized polynomial
            if line_stripped.startswith('### Size-optimized polynomial'):
                if current_poly:
                    identifier = parse_polynomial_identifier(current_poly)
                    if identifier:
                        identifiers.append(identifier)
                        if max_polys and len(identifiers) >= max_polys:
                            break
                current_poly = []
                in_poly = True
                continue

            # Blank line
            if line_stripped == '':
                if current_poly:
                    identifier = parse_polynomial_identifier(current_poly)
                    if identifier:
                        identifiers.append(identifier)
                        if max_polys and len(identifiers) >= max_polys:
                            break
                current_poly = []
                in_poly = False
                continue

            # Collect polynomial lines
            if in_poly:
                current_poly.append(line_stripped)

    # Don't forget last polynomial
    if current_poly:
        identifier = parse_polynomial_identifier(current_poly)
        if identifier:
            identifiers.append(identifier)

    return identifiers[:max_polys] if max_polys else identifiers

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: extract_input_polys_sorted.py <sorted_output> <unsorted_output> <output_file> [max_polys]", file=sys.stderr)
        print("  sorted_output: CADO sopt output sorted by exp_E", file=sys.stderr)
        print("  unsorted_output: CADO sopt output (unsorted, with input polynomials)", file=sys.stderr)
        print("  output_file: Output file for extracted input polynomials", file=sys.stderr)
        print("  max_polys: Maximum number of polynomials to extract (optional)", file=sys.stderr)
        sys.exit(1)

    sorted_file = sys.argv[1]
    unsorted_file = sys.argv[2]
    output_file = sys.argv[3]
    max_polys = int(sys.argv[4]) if len(sys.argv) == 5 else None

    print(f"Building polynomial map from unsorted output: {unsorted_file}...")
    poly_map = parse_unsorted_output(unsorted_file)
    print(f"  Found {len(poly_map)} polynomials in unsorted output")

    print(f"Reading sorted order from: {sorted_file}...")
    if max_polys:
        print(f"  Limiting to top {max_polys} polynomials...")
    identifiers = parse_sorted_output(sorted_file, max_polys)
    print(f"  Found {len(identifiers)} polynomials in sorted output")

    # Extract input polynomials in sorted order
    print("Matching and extracting input polynomials in sorted order...")
    input_polys = []
    not_found = 0
    for identifier in identifiers:
        if identifier in poly_map:
            input_polys.append(poly_map[identifier])
        else:
            not_found += 1
            print(f"Warning: Could not find input poly for identifier: {identifier[:3]}", file=sys.stderr)

    if not_found > 0:
        print(f"Warning: {not_found} polynomials not found in unsorted output", file=sys.stderr)

    if not input_polys:
        print("Error: No input polynomials extracted", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully matched {len(input_polys)} input polynomials")

    # Write to output file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        for i, poly in enumerate(input_polys):
            f.write(poly)
            # Add blank line separator between polynomials (except after last)
            if i < len(input_polys) - 1:
                f.write('\n\n')

    print(f"Successfully wrote {len(input_polys)} polynomials in sorted order")

if __name__ == '__main__':
    main()
