#!/bin/bash

# Compare root optimization methods:
# 1. msieve -npr (original)
# 2. msieve -npr (inverted)
# 3. CADO polyselect_ropt ropteffort=10 (original)
# 4. CADO polyselect_ropt ropteffort=10 (inverted)

set -euo pipefail

# Configuration
CADO_SOPT_SORTED="cado_sopt_output.txt"
MSIEVE_SOPT_SORTED="cado_results_sorted.ms"
TOP_N=10  # Number of top polynomials to test
THREADS=4  # Number of parallel threads

# Paths
CADO_ROPT="$HOME/cado-nfs/build/Kyle-PC-V2/polyselect/polyselect_ropt"
MSIEVE="./msieve"

# Output files
CADO_ORIG="top${TOP_N}_cado_orig.txt"
CADO_INV="top${TOP_N}_cado_inv.txt"
MSIEVE_ORIG="top${TOP_N}_msieve_orig.ms"
MSIEVE_INV="top${TOP_N}_msieve_inv.ms"

# Result files
MSIEVE_ORIG_RESULT="ropt_msieve_orig.p"
MSIEVE_INV_RESULT="ropt_msieve_inv.p"
CADO_ORIG_RESULT="ropt_cado_orig_e10.txt"
CADO_INV_RESULT="ropt_cado_inv_e10.txt"

COMPARISON_FILE="ropt_comparison.txt"

# Show help
show_help() {
    cat << EOF
Usage: test_ropt_comparison.sh [OPTIONS]

Compare msieve -npr vs CADO polyselect_ropt (with and without inversion)

Options:
  -h, --help              Show this help message and exit
  -n, --top N             Number of top polynomials to test (default: 10)
  -t, --threads N         Number of parallel threads (default: 4)
  -c, --cado FILE         Sorted CADO sopt output (default: cado_sopt_output.txt)
  -m, --msieve FILE       Sorted msieve format (default: cado_results_sorted.ms)

Tests 4 combinations:
  1. msieve -npr (original)
  2. msieve -npr (inverted C coefficients)
  3. CADO polyselect_ropt ropteffort=10 (original)
  4. CADO polyselect_ropt ropteffort=10 (inverted C coefficients)

Output:
  ropt_comparison.txt - Comparison of all methods

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -n|--top)
            TOP_N="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -c|--cado)
            CADO_SOPT_SORTED="$2"
            shift 2
            ;;
        -m|--msieve)
            MSIEVE_SOPT_SORTED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check dependencies
if [ ! -f "$CADO_ROPT" ]; then
    echo "Error: CADO polyselect_ropt not found at $CADO_ROPT"
    exit 1
fi

if [ ! -f "$MSIEVE" ]; then
    echo "Error: msieve not found at $MSIEVE"
    exit 1
fi

for script in utils/extract_top_cado_poly.py utils/invert_c_coefficients.py utils/invert_msieve_single_line.py scripts/run_msieve_ropt_annotated.sh; do
    if [ ! -f "$script" ]; then
        echo "Error: $script not found"
        exit 1
    fi
done

if [ ! -f "$CADO_SOPT_SORTED" ]; then
    echo "Error: CADO sopt output $CADO_SOPT_SORTED not found"
    exit 1
fi

if [ ! -f "$MSIEVE_SOPT_SORTED" ]; then
    echo "Error: msieve sopt output $MSIEVE_SOPT_SORTED not found"
    exit 1
fi

echo "======================================"
echo "Root Optimization Comparison"
echo "======================================"
echo "Testing top $TOP_N polynomial(s)"
echo "Parallel threads: $THREADS"
echo "CADO input:   $CADO_SOPT_SORTED"
echo "msieve input: $MSIEVE_SOPT_SORTED"
echo ""
echo "Testing 4 methods:"
echo "  1. msieve -npr (original)"
echo "  2. msieve -npr (inverted)"
echo "  3. CADO polyselect_ropt ropteffort=10 (original)"
echo "  4. CADO polyselect_ropt ropteffort=10 (inverted)"
echo ""

# Clean up old results
rm -f "$MSIEVE_ORIG_RESULT" "$MSIEVE_INV_RESULT" "$CADO_ORIG_RESULT" "$CADO_INV_RESULT"
rm -f top${TOP_N}_*.ms top${TOP_N}_*.txt ropt_*.p ropt_*.txt

# STEP 1: Extract top polynomials
echo "=== STEP 1: EXTRACT TOP POLYNOMIALS ==="

# Extract from CADO format
echo "Extracting top $TOP_N from CADO output..."
if python3 utils/extract_top_cado_poly.py "$CADO_SOPT_SORTED" "$CADO_ORIG" "$TOP_N"; then
    echo "  Extracted to: $CADO_ORIG"
else
    echo "Error: Failed to extract CADO polynomial"
    exit 1
fi

# Extract from msieve format (just take first N lines)
echo "Extracting top $TOP_N from msieve output..."
head -n "$TOP_N" "$MSIEVE_SOPT_SORTED" > "$MSIEVE_ORIG"
echo "  Extracted to: $MSIEVE_ORIG"

echo ""

# STEP 2: Create inverted versions
echo "=== STEP 2: CREATE INVERTED VERSIONS ==="

echo "Inverting CADO format..."
if python3 utils/invert_c_coefficients.py "$CADO_ORIG" "$CADO_INV"; then
    echo "  Created: $CADO_INV"
else
    echo "Error: Failed to invert CADO polynomial"
    exit 1
fi

echo "Inverting msieve format..."
if python3 utils/invert_msieve_single_line.py "$MSIEVE_ORIG" "$MSIEVE_INV"; then
    echo "  Created: $MSIEVE_INV"
else
    echo "Error: Failed to invert msieve polynomial"
    exit 1
fi

echo ""

# STEP 3: Run msieve -npr on original
echo "=== STEP 3: MSIEVE -NPR (ORIGINAL) ==="
echo "Running msieve -npr on original ($THREADS threads)..."

# Detect polynomial degree from msieve format
# Format for deg5: c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm (11 columns)
# Format for deg6: c6 c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm (12 columns)
NUM_COLS=$(head -n 1 "$MSIEVE_ORIG" | wc -w)
echo "  Detected $NUM_COLS columns"
if [ "$NUM_COLS" -eq 12 ]; then
    POLY_DEGREE=6
    echo "  Polynomial degree: 6"
else
    POLY_DEGREE=5
    echo "  Polynomial degree: 5"
fi

START_TIME=$(date +%s)

# Use annotated script with multithreading
if ./scripts/run_msieve_ropt_annotated.sh "$MSIEVE_ORIG" "$MSIEVE_ORIG_RESULT" "$POLY_DEGREE" "$THREADS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  Completed in ${DURATION}s"
    echo "  Result: $MSIEVE_ORIG_RESULT"
else
    echo "  Warning: Processing failed"
fi
echo ""

# STEP 4: Run msieve -npr on inverted
echo "=== STEP 4: MSIEVE -NPR (INVERTED) ==="
echo "Running msieve -npr on inverted ($THREADS threads)..."
START_TIME=$(date +%s)

# Use annotated script with multithreading
if ./scripts/run_msieve_ropt_annotated.sh "$MSIEVE_INV" "$MSIEVE_INV_RESULT" "$POLY_DEGREE" "$THREADS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  Completed in ${DURATION}s"
    echo "  Result: $MSIEVE_INV_RESULT"
else
    echo "  Warning: Processing failed"
fi
echo ""

# Helper function to run CADO ropt in parallel
run_cado_parallel() {
    local input_file=$1
    local output_file=$2
    local threads=$3

    # If only 1 thread, just run directly
    if [ "$threads" -eq 1 ]; then
        "$CADO_ROPT" -ropteffort 10 -inputpolys "$input_file" > "$output_file" 2>&1
        return
    fi

    # Split input into exactly <threads> number of chunks
    # Each chunk will contain multiple polynomials
    local chunk_dir="${input_file}_chunks"
    rm -rf "$chunk_dir"
    mkdir -p "$chunk_dir"

    # Split polynomials into <threads> chunks to minimize CADO startup overhead
    python3 -c "
import sys

# Read all polynomials first
polynomials = []
current_poly = []

with open('$input_file', 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('### Size-optimized polynomial'):
            if current_poly:
                polynomials.append('\n'.join(current_poly))
            current_poly = [line]
        elif line == '':
            if current_poly:
                polynomials.append('\n'.join(current_poly))
                current_poly = []
        elif current_poly:
            current_poly.append(line)

    # Don't forget last polynomial
    if current_poly:
        polynomials.append('\n'.join(current_poly))

# Split into exactly $threads chunks
num_polys = len(polynomials)
threads = $threads
polys_per_chunk = (num_polys + threads - 1) // threads

for chunk_idx in range(threads):
    start_idx = chunk_idx * polys_per_chunk
    end_idx = min(start_idx + polys_per_chunk, num_polys)

    if start_idx >= num_polys:
        break

    chunk_file = '$chunk_dir/chunk_%02d.txt' % chunk_idx
    with open(chunk_file, 'w') as out:
        for poly in polynomials[start_idx:end_idx]:
            out.write(poly + '\n\n')
"

    # Process chunks in parallel (one CADO process per chunk)
    PIDS=()
    for chunk_file in "$chunk_dir"/chunk_*.txt; do
        if [ -f "$chunk_file" ]; then
            result_file="${chunk_file%.txt}_result.txt"
            (
                "$CADO_ROPT" -ropteffort 10 -inputpolys "$chunk_file" > "$result_file" 2>&1
            ) &
            PIDS+=($!)
        fi
    done

    # Wait for all CADO processes
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    # Merge results
    for result_file in "$chunk_dir"/chunk_*_result.txt; do
        if [ -f "$result_file" ]; then
            cat "$result_file" >> "$output_file"
        fi
    done

    # Cleanup
    rm -rf "$chunk_dir"
}

# STEP 5: Run CADO polyselect_ropt (ropteffort=10) on original
echo "=== STEP 5: CADO POLYSELECT_ROPT EFFORT=10 (ORIGINAL) ==="
echo "Running CADO polyselect_ropt with ropteffort=10 on original ($THREADS threads)..."
START_TIME=$(date +%s)

run_cado_parallel "$CADO_ORIG" "$CADO_ORIG_RESULT" "$THREADS"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Completed in ${DURATION}s"
echo "  Result: $CADO_ORIG_RESULT"
echo ""

# STEP 6: Run CADO polyselect_ropt (ropteffort=10) on inverted
echo "=== STEP 6: CADO POLYSELECT_ROPT EFFORT=10 (INVERTED) ==="
echo "Running CADO polyselect_ropt with ropteffort=10 on inverted ($THREADS threads)..."
START_TIME=$(date +%s)

run_cado_parallel "$CADO_INV" "$CADO_INV_RESULT" "$THREADS"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Completed in ${DURATION}s"
echo "  Result: $CADO_INV_RESULT"
echo ""

# STEP 7: Compare results
echo "=== STEP 7: COMPARISON ==="
echo "Creating comparison report..."

{
    echo "======================================"
    echo "ROOT OPTIMIZATION COMPARISON"
    echo "======================================"
    echo "Top $TOP_N polynomial(s) tested"
    echo ""
    echo "Methods tested:"
    echo "  1. msieve -npr (original)"
    echo "  2. msieve -npr (inverted C coefficients)"
    echo "  3. CADO polyselect_ropt ropteffort=10 (original)"
    echo "  4. CADO polyselect_ropt ropteffort=10 (inverted C coefficients)"
    echo ""
    echo "======================================"
    echo "RESULTS"
    echo "======================================"
    echo ""

    # Extract Murphy E scores from each result
    echo "1. msieve -npr (original):"
    if [ -f "$MSIEVE_ORIG_RESULT" ]; then
        MSIEVE_ORIG_COUNT=$(grep -c "^# norm" "$MSIEVE_ORIG_RESULT" || echo 0)
        echo "  Found $MSIEVE_ORIG_COUNT root-optimized polynomial(s)"
        echo ""
        echo "  Top 10 results (showing exp_E → Murphy E correlation):"
        grep "# norm" "$MSIEVE_ORIG_RESULT" | head -n 10 || echo "  (No results found)"
        echo ""
        # Extract best Murphy E if present (format: "e 3.992e-16")
        BEST_MURPHY=$(grep "# norm" "$MSIEVE_ORIG_RESULT" | grep -o " e [0-9.e+-]*" | awk '{print $2}' | sort -g | tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    echo "2. msieve -npr (inverted):"
    if [ -f "$MSIEVE_INV_RESULT" ]; then
        MSIEVE_INV_COUNT=$(grep -c "^# norm" "$MSIEVE_INV_RESULT" || echo 0)
        echo "  Found $MSIEVE_INV_COUNT root-optimized polynomial(s)"
        echo ""
        echo "  Top 10 results (showing exp_E → Murphy E correlation):"
        grep "# norm" "$MSIEVE_INV_RESULT" | head -n 10 || echo "  (No results found)"
        echo ""
        # Extract best Murphy E if present (format: "e 3.992e-16")
        BEST_MURPHY=$(grep "# norm" "$MSIEVE_INV_RESULT" | grep -o " e [0-9.e+-]*" | awk '{print $2}' | sort -g | tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    echo "3. CADO polyselect_ropt ropteffort=10 (original):"
    if [ -f "$CADO_ORIG_RESULT" ]; then
        CADO_ORIG_COUNT=$(grep -c "### Root-optimized polynomial" "$CADO_ORIG_RESULT" 2>/dev/null || echo 0)
        if [ "$CADO_ORIG_COUNT" -gt 0 ]; then
            echo "  Found $CADO_ORIG_COUNT root-optimized polynomial(s)"
        fi
        grep -E "lognorm|exp_E|MurphyE" "$CADO_ORIG_RESULT" || echo "  (No results found)"
    else
        echo "  (No output file)"
    fi
    echo ""

    echo "4. CADO polyselect_ropt ropteffort=10 (inverted):"
    if [ -f "$CADO_INV_RESULT" ]; then
        CADO_INV_COUNT=$(grep -c "### Root-optimized polynomial" "$CADO_INV_RESULT" 2>/dev/null || echo 0)
        if [ "$CADO_INV_COUNT" -gt 0 ]; then
            echo "  Found $CADO_INV_COUNT root-optimized polynomial(s)"
        fi
        grep -E "lognorm|exp_E|MurphyE" "$CADO_INV_RESULT" || echo "  (No results found)"
    else
        echo "  (No output file)"
    fi
    echo ""

    echo "======================================"
    echo "Full output files for detailed inspection:"
    echo "  $MSIEVE_ORIG_RESULT"
    echo "  $MSIEVE_INV_RESULT"
    echo "  $CADO_ORIG_RESULT"
    echo "  $CADO_INV_RESULT"
    echo "======================================"

} > "$COMPARISON_FILE"

cat "$COMPARISON_FILE"

echo ""
echo "======================================"
echo "SUCCESS!"
echo "======================================"
echo "Comparison saved to: $COMPARISON_FILE"
echo ""

# Cleanup intermediate files (most are cleaned up by run_msieve_ropt_annotated.sh)
rm -f msieve.fb
