#!/bin/bash

# Deduplicate msieve.dat.ms, run CADO sopt (multithreaded), and sort by exp_E
# This is a standalone preprocessing step that can replace Phase 1 of process_batches.sh

set -euo pipefail

# Configuration
INPUT_FILE="msieve.dat.ms"
DEDUPED_FILE="msieve.dat.deduped.ms"
SOPT_THREADS=4  # Number of parallel sopt workers
CADO_OUTPUT_DIR="sopt_output"
CADO_SOPT_OUTPUT="cado_sopt_output.txt"          # CADO format output (sorted by exp_E)
CADO_SOPT_UNSORTED="cado_sopt_unsorted.txt"      # CADO format (unsorted, for reference)
FINAL_SORTED="cado_results_sorted.ms"            # Msieve format sorted by exp_E

# CADO-NFS path (use environment variable if set via nfs_optimize.sh, otherwise use default)
CADO_SOPT="${CADO_SOPT:-$HOME/cado-nfs/build/localhost/polyselect/sopt}"

# Show help
show_help() {
    cat << EOF
Usage: dedupe_and_sopt.sh [OPTIONS]

Deduplicate msieve.dat.ms, run CADO sopt (multithreaded), and sort by exp_E

Options:
  -h, --help           Show this help message and exit
  -t, --threads N      Set number of parallel sopt workers (default: 4)
  -i, --input FILE     Input file (default: msieve.dat.ms)
  -o, --output FILE    Final sorted output (default: cado_results_sorted.ms)

Steps:
  1. Deduplicate input polynomials
  2. Run CADO sopt on deduplicated data (multithreaded)
  3. Convert to msieve format with exp_E column
  4. Sort by exp_E (lower is better)

Outputs:
  cado_sopt_output.txt      - CADO format sorted by exp_E (for CADO ropt)
  cado_results_sorted.ms    - Msieve format sorted by exp_E (for msieve ropt)

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -t|--threads)
            SOPT_THREADS="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            FINAL_SORTED="$2"
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
if [ ! -f "$CADO_SOPT" ]; then
    echo "Error: CADO sopt not found at $CADO_SOPT"
    exit 1
fi

if [ ! -f "utils/cado_to_msieve.py" ]; then
    echo "Error: cado_to_msieve.py not found in utils directory"
    exit 1
fi

if [ ! -f "utils/sort_cado_by_expe.py" ]; then
    echo "Error: sort_cado_by_expe.py not found in utils directory"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

echo "======================================"
echo "Deduplicate and CADO Size Optimization"
echo "======================================"
echo "Input: $INPUT_FILE"
echo "Threads: $SOPT_THREADS"
echo "Output: $FINAL_SORTED"
echo ""

# STEP 1: Deduplicate polynomials
echo "=== STEP 1: DEDUPLICATION ==="
echo "Reading input file..."

TOTAL_LINES=$(wc -l < "$INPUT_FILE")
echo "Total lines: $TOTAL_LINES"

# Count original polynomials
ORIGINAL_POLYS=$(grep -c "^n:" "$INPUT_FILE" || true)
echo "Original polynomials: $ORIGINAL_POLYS"

# Deduplicate by using awk to track unique polynomials
# Strategy: Hash each complete polynomial block and keep only first occurrence
echo "Deduplicating polynomials..."

awk '
BEGIN {
    poly = ""
    in_poly = 0
}
/^n:/ {
    # Starting new polynomial
    if (poly != "" && !(poly in seen)) {
        # Print previous polynomial if not seen
        print poly
        seen[poly] = 1
    }
    poly = $0
    in_poly = 1
    next
}
/^[[:space:]]*$/ {
    # Blank line marks end of polynomial
    if (in_poly && poly != "" && !(poly in seen)) {
        print poly
        print ""  # Add blank line separator
        seen[poly] = 1
    }
    poly = ""
    in_poly = 0
    next
}
{
    # Accumulate polynomial lines
    if (in_poly) {
        poly = poly "\n" $0
    }
}
END {
    # Handle last polynomial
    if (poly != "" && !(poly in seen)) {
        print poly
    }
}
' "$INPUT_FILE" > "$DEDUPED_FILE"

DEDUPED_POLYS=$(grep -c "^n:" "$DEDUPED_FILE" || true)
DUPLICATES=$((ORIGINAL_POLYS - DEDUPED_POLYS))
echo "Deduplicated polynomials: $DEDUPED_POLYS"
echo "Duplicates removed: $DUPLICATES ($((DUPLICATES * 100 / ORIGINAL_POLYS))%)"
echo ""

if [ $DEDUPED_POLYS -eq 0 ]; then
    echo "Error: No polynomials remaining after deduplication"
    exit 1
fi

# STEP 2: Split into chunks for parallel sopt processing
echo "=== STEP 2: PARALLEL CADO SOPT ==="
echo "Splitting into $SOPT_THREADS chunks..."

# Create output directory
mkdir -p "$CADO_OUTPUT_DIR"
rm -f "$CADO_OUTPUT_DIR"/chunk_*.ms "$CADO_OUTPUT_DIR"/sopt_*.out

# Count polynomials and calculate chunk size
POLYS_PER_CHUNK=$(( (DEDUPED_POLYS + SOPT_THREADS - 1) / SOPT_THREADS ))
echo "Polynomials per chunk: ~$POLYS_PER_CHUNK"

# Split by complete polynomials (each polynomial is multiple lines)
# Use awk to ensure we split on polynomial boundaries
awk -v threads="$SOPT_THREADS" -v outdir="$CADO_OUTPUT_DIR" -v per_chunk="$POLYS_PER_CHUNK" '
BEGIN {
    chunk_num = 0
    poly_count = 0
    filename = outdir "/chunk_" sprintf("%02d", chunk_num) ".ms"
}
/^n:/ {
    # Check if we need to start a new chunk
    if (poly_count >= per_chunk && chunk_num < threads - 1) {
        close(filename)
        chunk_num++
        poly_count = 0
        filename = outdir "/chunk_" sprintf("%02d", chunk_num) ".ms"
    }
    poly_count++
}
{
    print > filename
}
' "$DEDUPED_FILE"

CHUNK_COUNT=$(ls "$CADO_OUTPUT_DIR"/chunk_*.ms 2>/dev/null | wc -l)
echo "Created $CHUNK_COUNT chunks"
echo ""

# STEP 3: Run sopt on each chunk in parallel
echo "Running CADO sopt on $CHUNK_COUNT chunks in parallel..."
START_TIME=$(date +%s)

PIDS=()
WORKER_NUM=0
for chunk in "$CADO_OUTPUT_DIR"/chunk_*.ms; do
    chunk_basename=$(basename "$chunk" .ms)
    output_file="$CADO_OUTPUT_DIR/sopt_${chunk_basename}.out"
    WORKER_NUM=$((WORKER_NUM + 1))

    echo "  Worker $WORKER_NUM: Processing $chunk_basename..."

    (
        "$CADO_SOPT" -inputpolys "$chunk" > "$output_file" 2>&1
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Worker $WORKER_NUM failed with exit code $exit_code"
            exit $exit_code
        fi
    ) &
    PIDS+=($!)
done

# Wait for all sopt processes
echo "  Waiting for $CHUNK_COUNT parallel sopt processes..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $FAILED -gt 0 ]; then
    echo "Error: $FAILED sopt process(es) failed"
    echo "Check output files in $CADO_OUTPUT_DIR/ for details"
    exit 1
fi

echo "  CADO sopt completed in ${DURATION}s"
echo ""

# STEP 4: Merge and sort CADO sopt outputs
echo "=== STEP 3: MERGING AND SORTING CADO OUTPUT ==="
echo "Merging all sopt outputs..."

cat "$CADO_OUTPUT_DIR"/sopt_chunk_*.out > "$CADO_SOPT_UNSORTED"

SOPT_POLYS=$(grep -c "^### Size-optimized polynomial" "$CADO_SOPT_UNSORTED" || true)
echo "Total size-optimized polynomials: $SOPT_POLYS"

echo "Sorting CADO output by exp_E..."
if python3 utils/sort_cado_by_expe.py "$CADO_SOPT_UNSORTED" "$CADO_SOPT_OUTPUT"; then
    echo "CADO format output (sorted): $CADO_SOPT_OUTPUT"
else
    echo "Error: Failed to sort CADO output"
    exit 1
fi
echo ""

# STEP 5: Convert to msieve format with exp_E
echo "=== STEP 4: CONVERT TO MSIEVE FORMAT ==="
echo "Converting to msieve format with exp_E column..."

MSIEVE_WITH_EXPE="$CADO_OUTPUT_DIR/msieve_with_expe.ms"
if python3 utils/cado_to_msieve.py "$CADO_SOPT_OUTPUT" "$MSIEVE_WITH_EXPE" 2>&1 | grep -E "Found|Wrote|Best|Worst"; then
    echo "Conversion successful"
else
    echo "Error: Conversion failed"
    exit 1
fi
echo ""

# STEP 6: Sort by exp_E
echo "=== STEP 5: SORT BY EXP_E ==="
echo "Sorting by exp_E (lower is better)..."

# Detect polynomial degree from number of columns
NUM_COLS=$(head -n 1 "$MSIEVE_WITH_EXPE" | wc -w)

if [ "$NUM_COLS" -eq 12 ]; then
    POLY_DEGREE=6
    EXPE_COL=11
    echo "  Detected degree 6 polynomial format (12 columns)"
else
    POLY_DEGREE=5
    EXPE_COL=10
    echo "  Detected degree 5 polynomial format (11 columns)"
fi

# Sort by exp_E column (lower is better)
sort -k${EXPE_COL},${EXPE_COL}n "$MSIEVE_WITH_EXPE" > "$FINAL_SORTED"

# Get statistics
TOTAL_OUTPUT=$(wc -l < "$FINAL_SORTED")
BEST_EXPE=$(head -n 1 "$FINAL_SORTED" | awk -v col=$EXPE_COL '{print $col}')
WORST_EXPE=$(tail -n 1 "$FINAL_SORTED" | awk -v col=$EXPE_COL '{print $col}')

echo "  Total sorted polynomials: $TOTAL_OUTPUT"
echo "  exp_E range: $BEST_EXPE (best) to $WORST_EXPE (worst)"
echo ""

# Cleanup intermediate files (optional)
echo "=== CLEANUP ==="
echo "Intermediate files saved in $CADO_OUTPUT_DIR/"
echo "Deduplicated input: $DEDUPED_FILE"
echo ""

echo "======================================"
echo "SUCCESS!"
echo "======================================"
echo "Output files:"
echo "  CADO format:   $CADO_SOPT_OUTPUT (sorted by exp_E)"
echo "  Msieve format: $FINAL_SORTED (sorted by exp_E)"
echo ""
echo "Next steps:"
echo "  - Use $FINAL_SORTED for msieve root optimization (msieve -npr)"
echo "  - Use $CADO_SOPT_OUTPUT for CADO root optimization (ropt)"
echo "  - Both files sorted by exp_E: top polynomials have best expected difficulty"
echo ""
