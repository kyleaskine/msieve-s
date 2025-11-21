#!/bin/bash

# Extract top polynomials from pipeline results and prepare for iterative_ropt.sh
# Converts msieve .p format to single-line format

set -euo pipefail

# Configuration
PIPELINE_DIR="pipeline_results"
TOP_N=20  # Extract top N polynomials by Murphy-E

# Show help
show_help() {
    cat << EOF
Usage: prepare_for_iterative_ropt.sh [OPTIONS]

Extract top polynomials from full_optimization_pipeline results
and prepare them for iterative_ropt.sh

Options:
  -h, --help              Show this help message
  -n, --top N             Extract top N polynomials (default: 20)
  -d, --dir DIR           Pipeline results directory (default: pipeline_results)

Input files (from full_optimization_pipeline.sh):
  \$DIR/msieve_ropt_orig.p
  \$DIR/msieve_ropt_inv.p

Output:
  top_for_iterative_ropt.ms - Top N polynomials in single-line format

Example:
  prepare_for_iterative_ropt.sh -n 10
  ./scripts/iterative_ropt.sh top_for_iterative_ropt.ms

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -n|--top)
            TOP_N="$2"
            shift 2
            ;;
        -d|--dir)
            PIPELINE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check input files
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

echo "======================================"
echo "PREPARE FOR ITERATIVE ROPT"
echo "======================================"
echo "Pipeline directory: $PIPELINE_DIR"
echo "Extracting top: $TOP_N polynomials"
echo ""

# Combine original and inverted results
COMBINED_P="combined_ropt.p"
rm -f "$COMBINED_P"

if [ -f "$PIPELINE_DIR/msieve_ropt_orig.p" ]; then
    cat "$PIPELINE_DIR/msieve_ropt_orig.p" >> "$COMBINED_P"
    echo "Added: msieve_ropt_orig.p"
fi

if [ -f "$PIPELINE_DIR/msieve_ropt_inv.p" ]; then
    cat "$PIPELINE_DIR/msieve_ropt_inv.p" >> "$COMBINED_P"
    echo "Added: msieve_ropt_inv.p"
fi

if [ ! -f "$COMBINED_P" ]; then
    echo "Error: No .p files found to process"
    exit 1
fi

TOTAL_POLYS=$(grep -c "^# norm" "$COMBINED_P" || echo 0)
echo "Total polynomials: $TOTAL_POLYS"

if [ $TOTAL_POLYS -eq 0 ]; then
    echo "Error: No polynomials found in .p files"
    exit 1
fi

# Extract Murphy-E scores and sort
echo ""
echo "Extracting and sorting by Murphy-E..."

# Create temporary file with Murphy-E and polynomial blocks
TEMP_SCORED="temp_scored.txt"
rm -f "$TEMP_SCORED"

python3 << 'PYEOF'
import re
import sys

# Read the combined .p file
with open('combined_ropt.p', 'r') as f:
    content = f.read()

# Split into polynomial blocks (each starts with "# norm")
poly_blocks = re.split(r'(?=^# norm)', content, flags=re.MULTILINE)
poly_blocks = [block.strip() for block in poly_blocks if block.strip()]

print(f"Total polynomial blocks: {len(poly_blocks)}")

# Extract Murphy-E and polynomial signature for deduplication
scored_polys = []
seen_polys = set()  # Track unique polynomials by coefficients

for block in poly_blocks:
    # Extract Murphy-E from "# norm ... e 9.656e-15 ..." format
    match = re.search(r'\se\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', block)
    if match:
        murphy_e = float(match.group(1))

        # Create signature from coefficients for deduplication
        # Extract all c0-c5, Y0, Y1 values
        coeffs = []
        for line in block.split('\n'):
            if re.match(r'c\d+:', line):
                coeffs.append(line.strip())
            elif line.startswith('Y0:') or line.startswith('Y1:'):
                coeffs.append(line.strip())

        poly_signature = tuple(sorted(coeffs))

        # Only keep if not a duplicate
        if poly_signature not in seen_polys:
            seen_polys.add(poly_signature)
            scored_polys.append((murphy_e, block))

print(f"Unique polynomials after deduplication: {len(scored_polys)}")

# Sort by Murphy-E (higher is better)
scored_polys.sort(key=lambda x: x[0], reverse=True)

# Write sorted blocks with scores
with open('temp_scored.txt', 'w') as f:
    for murphy_e, block in scored_polys:
        f.write(f"MURPHY_E={murphy_e}\n")
        f.write(block)
        f.write("\n\n")

print(f"Sorted {len(scored_polys)} polynomials by Murphy-E")
if scored_polys:
    print(f"Best Murphy-E: {scored_polys[0][0]:.6e}")
    print(f"Worst Murphy-E: {scored_polys[-1][0]:.6e}")
PYEOF

# Extract top N and convert to single-line format
echo ""
echo "Converting top $TOP_N to single-line format..."

python3 << PYEOF
import re

# Read scored polynomials
with open('temp_scored.txt', 'r') as f:
    content = f.read()

# Split by MURPHY_E markers
blocks = re.split(r'MURPHY_E=', content)
blocks = [b.strip() for b in blocks if b.strip()]

output_lines = []
count = 0

for block in blocks:
    if count >= $TOP_N:
        break

    lines = block.split('\n')
    murphy_e = None

    # Parse the polynomial
    poly_data = {}
    for line in lines:
        if line.startswith('n:'):
            poly_data['n'] = line.split(':', 1)[1].strip()
        elif line.startswith('Y0:'):
            poly_data['Y0'] = line.split(':', 1)[1].strip()
        elif line.startswith('Y1:'):
            poly_data['Y1'] = line.split(':', 1)[1].strip()
        elif re.match(r'c\d+:', line):
            coeff_match = re.match(r'c(\d+):\s*(.+)', line)
            if coeff_match:
                degree = coeff_match.group(1)
                value = coeff_match.group(2).strip()
                poly_data[f'c{degree}'] = value
        elif line.startswith('# norm'):
            # Parse comment line: "# norm 4.791305e-19 alpha -6.153462 e 9.656e-15 ..."
            # Extract alpha
            alpha_match = re.search(r'alpha\s+([-+]?[0-9]*\.?[0-9]+)', line)
            if alpha_match:
                poly_data['alpha'] = alpha_match.group(1)
            # Extract norm
            norm_match = re.search(r'norm\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if norm_match:
                poly_data['norm'] = norm_match.group(1)

    # Detect degree
    max_degree = 0
    for key in poly_data:
        if key.startswith('c'):
            degree = int(key[1:])
            if degree > max_degree:
                max_degree = degree

    # Build single-line format
    # Format: c5 c4 c3 c2 c1 c0 Y1 Y0 alpha norm
    if max_degree >= 4 and 'Y0' in poly_data and 'Y1' in poly_data:
        parts = []

        # Coefficients in descending order
        for i in range(max_degree, -1, -1):
            parts.append(poly_data.get(f'c{i}', '0'))

        # Linear polynomial coefficients
        parts.append(poly_data.get('Y1', '0'))
        parts.append(poly_data.get('Y0', '0'))

        # Alpha (estimate if not present)
        parts.append(poly_data.get('alpha', '0'))

        # Norm (from comment line)
        parts.append(poly_data.get('norm', '1e30'))

        output_lines.append(' '.join(parts))
        count += 1

# Write output
with open('top_for_iterative_ropt.ms', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

print(f"Converted {len(output_lines)} polynomials to single-line format")
PYEOF

# Cleanup
rm -f "$COMBINED_P" "$TEMP_SCORED"

echo ""
echo "======================================"
echo "PREPARATION COMPLETE"
echo "======================================"
echo "Output: top_for_iterative_ropt.ms"

if [ -f "top_for_iterative_ropt.ms" ]; then
    LINES=$(wc -l < "top_for_iterative_ropt.ms")
    echo "Polynomials prepared: $LINES"
    echo ""
    echo "Next step:"
    echo "  ./scripts/iterative_ropt.sh top_for_iterative_ropt.ms"
    echo ""
else
    echo "Error: Failed to create output file"
    exit 1
fi
