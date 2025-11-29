#!/bin/bash

# Test the impact of -sopteffort 10 on top N polynomials
# Compares default sopt vs sopt with -sopteffort 10

set -euo pipefail

# Configuration
SORTED_SOPT="cado_sopt_output.txt"      # Sorted CADO output (from dedupe_and_sopt.sh)
UNSORTED_SOPT="cado_sopt_unsorted.txt"  # Unsorted CADO output (has input polys)
TOP_N=100                               # Number of top polynomials to test
SOPTEFFORT=10                           # CADO sopt effort level

# Output files
INPUT_POLYS="top${TOP_N}_input.ms"
SOPT_EFFORT_OUTPUT="sopt_effort${SOPTEFFORT}_output.txt"
SOPT_EFFORT_SORTED="sopt_effort${SOPTEFFORT}_sorted.txt"
COMPARISON_FILE="sopteffort_comparison.txt"

# CADO-NFS path
CADO_SOPT="$HOME/cado-nfs/build/Kyle-PC-V2/polyselect/sopt"

# Show help
show_help() {
    cat << EOF
Usage: test_sopteffort.sh [OPTIONS]

Test the impact of -sopteffort on polynomial optimization

Options:
  -h, --help              Show this help message and exit
  -n, --top N             Number of top polynomials to test (default: 100)
  -e, --effort N          CADO sopteffort value (default: 10)
  -s, --sorted FILE       Sorted CADO sopt output (default: cado_sopt_output.txt)
  -u, --unsorted FILE     Unsorted CADO sopt output (default: cado_sopt_unsorted.txt)

Steps:
  1. Extract top N input polynomials from sorted sopt output
  2. Run sopt with -sopteffort on them
  3. Sort by exp_E and compare with original

Output:
  sopteffort_comparison.txt - Comparison of exp_E values

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
        -e|--effort)
            SOPTEFFORT="$2"
            shift 2
            ;;
        -s|--sorted)
            SORTED_SOPT="$2"
            shift 2
            ;;
        -u|--unsorted)
            UNSORTED_SOPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Update output filenames based on parameters
INPUT_POLYS="top${TOP_N}_input.ms"
SOPT_EFFORT_OUTPUT="sopt_effort${SOPTEFFORT}_output.txt"
SOPT_EFFORT_SORTED="sopt_effort${SOPTEFFORT}_sorted.txt"

# Check dependencies
if [ ! -f "$CADO_SOPT" ]; then
    echo "Error: CADO sopt not found at $CADO_SOPT"
    exit 1
fi

if [ ! -f "utils/extract_input_polys_sorted.py" ]; then
    echo "Error: extract_input_polys_sorted.py not found in utils directory"
    exit 1
fi

if [ ! -f "utils/sort_cado_by_expe.py" ]; then
    echo "Error: sort_cado_by_expe.py not found in utils directory"
    exit 1
fi

if [ ! -f "$SORTED_SOPT" ]; then
    echo "Error: Sorted sopt output $SORTED_SOPT not found"
    echo "Run dedupe_and_sopt.sh first to generate this file"
    exit 1
fi

if [ ! -f "$UNSORTED_SOPT" ]; then
    echo "Error: Unsorted sopt output $UNSORTED_SOPT not found"
    echo "Run dedupe_and_sopt.sh first to generate this file"
    exit 1
fi

echo "======================================"
echo "Testing -sopteffort $SOPTEFFORT Impact"
echo "======================================"
echo "Sorted input:   $SORTED_SOPT"
echo "Unsorted input: $UNSORTED_SOPT"
echo "Testing top $TOP_N polynomials"
echo "CADO sopteffort: $SOPTEFFORT"
echo ""

# STEP 1: Extract top N input polynomials
echo "=== STEP 1: EXTRACT INPUT POLYNOMIALS ==="
echo "Extracting top $TOP_N input polynomials (in sorted order)..."

if python3 utils/extract_input_polys_sorted.py "$SORTED_SOPT" "$UNSORTED_SOPT" "$INPUT_POLYS" "$TOP_N"; then
    echo "Extracted to: $INPUT_POLYS"
else
    echo "Error: Failed to extract input polynomials"
    exit 1
fi
echo ""

# STEP 2: Run sopt with -sopteffort
echo "=== STEP 2: RUN SOPT WITH -sopteffort $SOPTEFFORT ==="
echo "Running CADO sopt with -sopteffort $SOPTEFFORT..."
START_TIME=$(date +%s)

if "$CADO_SOPT" -sopteffort "$SOPTEFFORT" -inputpolys "$INPUT_POLYS" > "$SOPT_EFFORT_OUTPUT" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    OPTIMIZED_COUNT=$(grep -c "^### Size-optimized polynomial" "$SOPT_EFFORT_OUTPUT" || true)
    echo "  Optimized $OPTIMIZED_COUNT polynomials in ${DURATION}s"
else
    echo "Error: CADO sopt failed"
    exit 1
fi
echo ""

# STEP 3: Sort by exp_E
echo "=== STEP 3: SORT RESULTS ==="
echo "Sorting by exp_E..."

if python3 utils/sort_cado_by_expe.py "$SOPT_EFFORT_OUTPUT" "$SOPT_EFFORT_SORTED"; then
    echo "Sorted output: $SOPT_EFFORT_SORTED"
else
    echo "Error: Failed to sort results"
    exit 1
fi
echo ""

# STEP 4: Compare results
echo "=== STEP 4: COMPARISON ==="
echo "Comparing original vs sopteffort=$SOPTEFFORT..."

# Extract exp_E values from both files
# Note: Using || true to avoid SIGPIPE errors from head with set -o pipefail
echo "Extracting exp_E values from original (default sopt)..."
grep "exp_E" "$SORTED_SOPT" | head -n "$TOP_N" | grep -o "exp_E [0-9.]*" | awk '{print $2}' > /tmp/original_expe.txt || true

echo "Extracting exp_E values from sopteffort=$SOPTEFFORT..."
grep "exp_E" "$SOPT_EFFORT_SORTED" | grep -o "exp_E [0-9.]*" | awk '{print $2}' > /tmp/effort_expe.txt || true

# Create comparison report
{
    echo "======================================"
    echo "SOPTEFFORT COMPARISON REPORT"
    echo "======================================"
    echo "Top $TOP_N polynomials"
    echo "Original: default sopt settings"
    echo "Improved: sopt with -sopteffort $SOPTEFFORT"
    echo ""
    echo "Statistics:"
    echo "----------"

    # Calculate statistics
    ORIGINAL_BEST=$(head -n 1 /tmp/original_expe.txt)
    ORIGINAL_WORST=$(tail -n 1 /tmp/original_expe.txt)
    ORIGINAL_AVG=$(awk '{sum+=$1; count++} END {print sum/count}' /tmp/original_expe.txt)

    EFFORT_BEST=$(head -n 1 /tmp/effort_expe.txt)
    EFFORT_WORST=$(tail -n 1 /tmp/effort_expe.txt)
    EFFORT_AVG=$(awk '{sum+=$1; count++} END {print sum/count}' /tmp/effort_expe.txt)

    echo "Original (default):"
    echo "  Best exp_E:    $ORIGINAL_BEST"
    echo "  Worst exp_E:   $ORIGINAL_WORST"
    echo "  Average exp_E: $ORIGINAL_AVG"
    echo ""
    echo "With -sopteffort $SOPTEFFORT:"
    echo "  Best exp_E:    $EFFORT_BEST"
    echo "  Worst exp_E:   $EFFORT_WORST"
    echo "  Average exp_E: $EFFORT_AVG"
    echo ""

    # Calculate improvement
    BEST_DIFF=$(echo "$ORIGINAL_BEST - $EFFORT_BEST" | bc)
    AVG_DIFF=$(echo "$ORIGINAL_AVG - $EFFORT_AVG" | bc)

    echo "Improvement (lower exp_E is better):"
    echo "  Best exp_E delta:    $BEST_DIFF"
    echo "  Average exp_E delta: $AVG_DIFF"
    echo ""
    echo "======================================"
    echo "DETAILED COMPARISON (first 20)"
    echo "======================================"
    echo "Poly#  Original   Improved   Delta"
    echo "-----  --------   --------   -----"

    # Show first 20 detailed comparisons
    paste /tmp/original_expe.txt /tmp/effort_expe.txt | head -n 20 | awk '{
        delta = $1 - $2
        printf "%5d  %8.4f   %8.4f   %+.4f\n", NR, $1, $2, delta
    }'

} > "$COMPARISON_FILE"

# Display comparison
cat "$COMPARISON_FILE"

echo ""
echo "======================================"
echo "SUCCESS!"
echo "======================================"
echo "Comparison saved to: $COMPARISON_FILE"
echo ""
echo "Files generated:"
echo "  Input polys:      $INPUT_POLYS"
echo "  Sopt output:      $SOPT_EFFORT_OUTPUT"
echo "  Sorted output:    $SOPT_EFFORT_SORTED"
echo "  Comparison:       $COMPARISON_FILE"
echo ""

# Cleanup temp files
rm -f /tmp/original_expe.txt /tmp/effort_expe.txt
