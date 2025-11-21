#!/bin/bash

# Complete optimization pipeline:
# 1. Extract top N polynomials from initial sopt results
# 2. Re-run sopt with higher effort on those polynomials
# 3. Sort by exp_E and take top M unique polynomials
# 4. Run root optimization (msieve + CADO) on the best M

set -euo pipefail

# Configuration
INITIAL_SOPT_SORTED="cado_sopt_output.txt"       # From dedupe_and_sopt.sh
INITIAL_SOPT_UNSORTED="cado_sopt_unsorted.txt"   # From dedupe_and_sopt.sh
MSIEVE_SOPT_SORTED="cado_results_sorted.ms"      # From dedupe_and_sopt.sh

TOP_N_EXTRACT=100        # Extract top 100 from initial sopt
RESOPT_EFFORT=10         # Re-run sopt with effort 10
TOP_M_MSIEVE=10          # Run msieve ropt on best 10 from re-sopt
TOP_M_CADO=100           # Run CADO ropt on best 100 from re-sopt
ROPT_EFFORT=10           # CADO ropt effort
THREADS=4                # Number of parallel threads for ropt

# Paths
CADO_SOPT="$HOME/cado-nfs/build/Kyle-PC-V2/polyselect/sopt"
CADO_ROPT="$HOME/cado-nfs/build/Kyle-PC-V2/polyselect/polyselect_ropt"
MSIEVE="./msieve"

# Working directories
WORK_DIR="pipeline_work"
FINAL_DIR="pipeline_results"

# Show help
show_help() {
    cat << EOF
Usage: full_optimization_pipeline.sh [OPTIONS]

Complete polynomial optimization pipeline

Options:
  -h, --help              Show this help message and exit
  -n, --extract N         Extract top N from initial sopt (default: 100)
  --msieve-ropt M         Run msieve ropt on top M after re-sopt (default: 10)
  --cado-ropt M           Run CADO ropt on top M after re-sopt (default: 100)
  --resopt-effort E       Re-sopt effort level (default: 10)
  --ropt-effort E         CADO ropt effort level (default: 10)
  -t, --threads N         Number of parallel threads for ropt (default: 4)
  --initial-sorted FILE   Initial sorted CADO sopt output (default: cado_sopt_output.txt)
  --initial-unsorted FILE Initial unsorted CADO sopt output (default: cado_sopt_unsorted.txt)
  --msieve-sorted FILE    Initial msieve sorted output (default: cado_results_sorted.ms)

Pipeline steps:
  1. Extract top N polynomials from initial sopt results
  2. Re-run sopt with --resopt-effort on those N polynomials
  3. Sort by exp_E
  4. Run root optimization on best polynomials:
     - msieve -npr on top M_msieve (original + inverted)
     - CADO polyselect_ropt on top M_cado (original + inverted)

Output:
  $FINAL_DIR/ - All final results and comparison

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -n|--extract)
            TOP_N_EXTRACT="$2"
            shift 2
            ;;
        --msieve-ropt)
            TOP_M_MSIEVE="$2"
            shift 2
            ;;
        --cado-ropt)
            TOP_M_CADO="$2"
            shift 2
            ;;
        --resopt-effort)
            RESOPT_EFFORT="$2"
            shift 2
            ;;
        --ropt-effort)
            ROPT_EFFORT="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --initial-sorted)
            INITIAL_SOPT_SORTED="$2"
            shift 2
            ;;
        --initial-unsorted)
            INITIAL_SOPT_UNSORTED="$2"
            shift 2
            ;;
        --msieve-sorted)
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
for cmd in "$CADO_SOPT" "$CADO_ROPT" "$MSIEVE"; do
    if [ ! -f "$cmd" ]; then
        echo "Error: Required binary not found: $cmd"
        exit 1
    fi
done

for script in utils/extract_input_polys_sorted.py utils/extract_top_cado_poly.py utils/sort_cado_by_expe.py \
              utils/invert_c_coefficients.py utils/invert_msieve_single_line.py scripts/run_msieve_ropt_annotated.sh; do
    if [ ! -f "$script" ]; then
        echo "Error: Required script not found: $script"
        exit 1
    fi
done

for file in "$INITIAL_SOPT_SORTED" "$INITIAL_SOPT_UNSORTED" "$MSIEVE_SOPT_SORTED"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required input file not found: $file"
        exit 1
    fi
done

# Create working directories
mkdir -p "$WORK_DIR" "$FINAL_DIR"

echo "======================================"
echo "FULL OPTIMIZATION PIPELINE"
echo "======================================"
echo "Extract top $TOP_N_EXTRACT from initial sopt"
echo "Re-run sopt with effort $RESOPT_EFFORT"
echo "Run msieve ropt on best $TOP_M_MSIEVE after re-sopt"
echo "Run CADO ropt on best $TOP_M_CADO after re-sopt"
echo "CADO ropt effort: $ROPT_EFFORT"
echo "Parallel threads: $THREADS"
echo ""

# PHASE 1: Extract top N input polynomials from initial sopt
echo "=== PHASE 1: EXTRACT TOP $TOP_N_EXTRACT INPUT POLYNOMIALS ==="

echo "Extracting input polynomials in sorted order..."
if python3 utils/extract_input_polys_sorted.py "$INITIAL_SOPT_SORTED" "$INITIAL_SOPT_UNSORTED" \
   "$WORK_DIR/top${TOP_N_EXTRACT}_input.ms" "$TOP_N_EXTRACT"; then
    echo "  Extracted to: $WORK_DIR/top${TOP_N_EXTRACT}_input.ms"
else
    echo "Error: Failed to extract input polynomials"
    exit 1
fi
echo ""

# PHASE 2: Re-run sopt with higher effort
echo "=== PHASE 2: RE-RUN SOPT WITH EFFORT $RESOPT_EFFORT ==="

echo "Running CADO sopt with -sopteffort $RESOPT_EFFORT on top $TOP_N_EXTRACT polynomials ($THREADS threads)..."
START_TIME=$(date +%s)

# If threads = 1, run directly
if [ "$THREADS" -eq 1 ]; then
    if "$CADO_SOPT" -sopteffort "$RESOPT_EFFORT" -inputpolys "$WORK_DIR/top${TOP_N_EXTRACT}_input.ms" \
       > "$WORK_DIR/resopt_output.txt" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        COUNT=$(grep -c "^### Size-optimized polynomial" "$WORK_DIR/resopt_output.txt" || true)
        echo "  Optimized $COUNT polynomials in ${DURATION}s"
    else
        echo "Error: CADO sopt failed"
        exit 1
    fi
else
    # Parallel sopt processing
    SOPT_CHUNK_DIR="$WORK_DIR/sopt_chunks"
    rm -rf "$SOPT_CHUNK_DIR"
    mkdir -p "$SOPT_CHUNK_DIR"

    # Calculate polynomials per chunk
    TOTAL_RESOPT_POLYS=$(grep -c "^n:" "$WORK_DIR/top${TOP_N_EXTRACT}_input.ms" || echo 0)
    POLYS_PER_CHUNK=$(( (TOTAL_RESOPT_POLYS + THREADS - 1) / THREADS ))
    echo "  Splitting $TOTAL_RESOPT_POLYS polynomials into $THREADS chunks (~$POLYS_PER_CHUNK per chunk)..."

    # Split into chunks
    awk -v threads="$THREADS" -v outdir="$SOPT_CHUNK_DIR" -v per_chunk="$POLYS_PER_CHUNK" '
    BEGIN {
        chunk_num = 0
        poly_count = 0
        filename = outdir "/chunk_" sprintf("%02d", chunk_num) ".ms"
    }
    /^n:/ {
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
    ' "$WORK_DIR/top${TOP_N_EXTRACT}_input.ms"

    # Run sopt on each chunk in parallel
    PIDS=()
    WORKER_NUM=0
    for chunk in "$SOPT_CHUNK_DIR"/chunk_*.ms; do
        chunk_basename=$(basename "$chunk" .ms)
        output_file="$SOPT_CHUNK_DIR/sopt_${chunk_basename}.out"
        WORKER_NUM=$((WORKER_NUM + 1))

        (
            "$CADO_SOPT" -sopteffort "$RESOPT_EFFORT" -inputpolys "$chunk" > "$output_file" 2>&1
        ) &
        PIDS+=($!)
    done

    # Wait for all sopt processes
    echo "  Waiting for $THREADS parallel sopt processes..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "Error: $FAILED sopt process(es) failed"
        exit 1
    fi

    # Merge results
    cat "$SOPT_CHUNK_DIR"/sopt_chunk_*.out > "$WORK_DIR/resopt_output.txt"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    COUNT=$(grep -c "^### Size-optimized polynomial" "$WORK_DIR/resopt_output.txt" || true)
    echo "  Optimized $COUNT polynomials in ${DURATION}s"

    # Cleanup
    rm -rf "$SOPT_CHUNK_DIR"
fi

# Sort re-sopt results by exp_E
echo "Sorting re-sopt results by exp_E..."
if python3 utils/sort_cado_by_expe.py "$WORK_DIR/resopt_output.txt" "$WORK_DIR/resopt_sorted.txt"; then
    echo "  Sorted output: $WORK_DIR/resopt_sorted.txt"
else
    echo "Error: Failed to sort re-sopt results"
    exit 1
fi

# Convert to msieve format
echo "Converting to msieve format..."
if python3 utils/cado_to_msieve.py "$WORK_DIR/resopt_output.txt" "$WORK_DIR/resopt_msieve.ms" 2>&1 | grep -E "Found|Wrote"; then
    echo "  Converted to: $WORK_DIR/resopt_msieve.ms"
else
    echo "Error: Conversion failed"
    exit 1
fi

# Sort msieve format by exp_E
echo "Sorting msieve format by exp_E..."
# Detect degree
NUM_COLS=$(head -n 1 "$WORK_DIR/resopt_msieve.ms" | wc -w)
if [ "$NUM_COLS" -eq 12 ]; then
    EXPE_COL=11
else
    EXPE_COL=10
fi
sort -k${EXPE_COL},${EXPE_COL}n "$WORK_DIR/resopt_msieve.ms" > "$WORK_DIR/resopt_msieve_sorted.ms"
echo "  Sorted msieve format: $WORK_DIR/resopt_msieve_sorted.ms"
echo ""

# PHASE 3: Extract polynomials for root optimization
echo "=== PHASE 3: EXTRACT POLYNOMIALS FOR ROOT OPTIMIZATION ==="

# Extract for msieve (smaller count typically)
echo "Extracting top $TOP_M_MSIEVE for msieve ropt (msieve format)..."
head -n "$TOP_M_MSIEVE" "$WORK_DIR/resopt_msieve_sorted.ms" > "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms"
echo "  Extracted: $FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms"

# Get exp_E range for msieve
MSIEVE_BEST_EXPE=$(head -n 1 "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" | awk -v col=$EXPE_COL '{print $col}')
MSIEVE_WORST_EXPE=$(tail -n 1 "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" | awk -v col=$EXPE_COL '{print $col}')
echo "  exp_E range: $MSIEVE_BEST_EXPE (best) to $MSIEVE_WORST_EXPE (worst)"

# Extract for CADO (larger count typically)
echo "Extracting top $TOP_M_CADO for CADO ropt (CADO format)..."
if python3 utils/extract_top_cado_poly.py "$WORK_DIR/resopt_sorted.txt" "$FINAL_DIR/best${TOP_M_CADO}_cado.txt" "$TOP_M_CADO"; then
    echo "  Extracted: $FINAL_DIR/best${TOP_M_CADO}_cado.txt"
else
    echo "Error: Failed to extract CADO polynomials"
    exit 1
fi

# Get exp_E range for CADO (from msieve format)
head -n "$TOP_M_CADO" "$WORK_DIR/resopt_msieve_sorted.ms" > "$WORK_DIR/temp_cado_check.ms"
CADO_BEST_EXPE=$(head -n 1 "$WORK_DIR/temp_cado_check.ms" | awk -v col=$EXPE_COL '{print $col}')
CADO_WORST_EXPE=$(tail -n 1 "$WORK_DIR/temp_cado_check.ms" | awk -v col=$EXPE_COL '{print $col}')
echo "  exp_E range: $CADO_BEST_EXPE (best) to $CADO_WORST_EXPE (worst)"
rm -f "$WORK_DIR/temp_cado_check.ms"
echo ""

# PHASE 4: Create inverted versions
echo "=== PHASE 4: CREATE INVERTED VERSIONS ==="

echo "Inverting msieve format..."
if python3 utils/invert_msieve_single_line.py "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve_inv.ms"; then
    echo "  Created: $FINAL_DIR/best${TOP_M_MSIEVE}_msieve_inv.ms"
fi

echo "Inverting CADO format..."
if python3 utils/invert_c_coefficients.py "$FINAL_DIR/best${TOP_M_CADO}_cado.txt" "$FINAL_DIR/best${TOP_M_CADO}_cado_inv.txt"; then
    echo "  Created: $FINAL_DIR/best${TOP_M_CADO}_cado_inv.txt"
fi
echo ""

# PHASE 5: Root optimization with msieve
echo "=== PHASE 5: ROOT OPTIMIZATION WITH MSIEVE ==="

# Detect polynomial degree
if [ "$NUM_COLS" -eq 12 ]; then
    POLY_DEGREE=6
    echo "Polynomial degree: 6"
else
    POLY_DEGREE=5
    echo "Polynomial degree: 5"
fi

echo "Running msieve -npr on original (annotated with exp_E, $THREADS threads)..."
echo "  Processing $TOP_M_MSIEVE polynomials..."
START_TIME=$(date +%s)
if ./scripts/run_msieve_ropt_annotated.sh "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" \
   "$FINAL_DIR/msieve_ropt_orig.p" "$POLY_DEGREE" "$THREADS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  Completed in ${DURATION}s"
fi

echo "Running msieve -npr on inverted (annotated with exp_E, $THREADS threads)..."
echo "  Processing $TOP_M_MSIEVE polynomials..."
START_TIME=$(date +%s)
if ./scripts/run_msieve_ropt_annotated.sh "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve_inv.ms" \
   "$FINAL_DIR/msieve_ropt_inv.p" "$POLY_DEGREE" "$THREADS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  Completed in ${DURATION}s"
fi
echo ""

# PHASE 6: Root optimization with CADO
echo "=== PHASE 6: ROOT OPTIMIZATION WITH CADO (EFFORT $ROPT_EFFORT) ==="

# Helper function to run CADO ropt in parallel
run_cado_parallel() {
    local input_file=$1
    local output_file=$2
    local threads=$3

    # If only 1 thread, just run directly
    if [ "$threads" -eq 1 ]; then
        "$CADO_ROPT" -ropteffort "$ROPT_EFFORT" -inputpolys "$input_file" > "$output_file" 2>&1
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

    # Count chunks created
    local num_chunks=$(ls -1 "$chunk_dir"/chunk_*.txt 2>/dev/null | wc -l)

    if [ "$num_chunks" -eq 0 ]; then
        echo "  Warning: No polynomials found to process"
        touch "$output_file"
        return
    fi

    # Process chunks in parallel (one CADO process per chunk)
    PIDS=()
    for chunk_file in "$chunk_dir"/chunk_*.txt; do
        if [ -f "$chunk_file" ]; then
            result_file="${chunk_file%.txt}_result.txt"
            (
                "$CADO_ROPT" -ropteffort "$ROPT_EFFORT" -inputpolys "$chunk_file" > "$result_file" 2>&1
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

echo "Running CADO polyselect_ropt on original ($THREADS threads)..."
echo "  Processing $TOP_M_CADO polynomials..."
START_TIME=$(date +%s)
run_cado_parallel "$FINAL_DIR/best${TOP_M_CADO}_cado.txt" "$FINAL_DIR/cado_ropt_orig.txt" "$THREADS"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Completed in ${DURATION}s"

echo "Running CADO polyselect_ropt on inverted ($THREADS threads)..."
echo "  Processing $TOP_M_CADO polynomials..."
START_TIME=$(date +%s)
run_cado_parallel "$FINAL_DIR/best${TOP_M_CADO}_cado_inv.txt" "$FINAL_DIR/cado_ropt_inv.txt" "$THREADS"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Completed in ${DURATION}s"
echo ""

# PHASE 7: Generate comparison report
echo "=== PHASE 7: GENERATE COMPARISON REPORT ==="

{
    echo "======================================"
    echo "FULL OPTIMIZATION PIPELINE RESULTS"
    echo "======================================"
    echo "Pipeline configuration:"
    echo "  Extracted top $TOP_N_EXTRACT from initial sopt"
    echo "  Re-ran sopt with effort $RESOPT_EFFORT"
    echo "  Selected best $TOP_M_MSIEVE for msieve ropt"
    echo "  Selected best $TOP_M_CADO for CADO ropt"
    echo "  CADO ropt effort: $ROPT_EFFORT"
    echo "  Parallel threads: $THREADS"
    echo ""
    echo "msieve exp_E range: $MSIEVE_BEST_EXPE to $MSIEVE_WORST_EXPE"
    echo "CADO exp_E range:   $CADO_BEST_EXPE to $CADO_WORST_EXPE"
    echo ""
    echo "======================================"
    echo "ROOT OPTIMIZATION RESULTS"
    echo "======================================"
    echo ""

    # Msieve original
    echo "1. msieve -npr (original):"
    if [ -f "$FINAL_DIR/msieve_ropt_orig.p" ]; then
        COUNT=$(grep -c "^# norm" "$FINAL_DIR/msieve_ropt_orig.p" || echo 0)
        echo "  Found $COUNT root-optimized polynomial(s)"
        echo ""
        echo "  Top 10 results (with exp_E correlation):"
        grep "# norm" "$FINAL_DIR/msieve_ropt_orig.p" | head -n 10 || echo "  (No results)"
        echo ""
        BEST_MURPHY=$(grep "# norm" "$FINAL_DIR/msieve_ropt_orig.p" | grep -o " e [0-9.e+-]*" | awk '{print $2}' | sort -g | tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    # Msieve inverted
    echo "2. msieve -npr (inverted):"
    if [ -f "$FINAL_DIR/msieve_ropt_inv.p" ]; then
        COUNT=$(grep -c "^# norm" "$FINAL_DIR/msieve_ropt_inv.p" || echo 0)
        echo "  Found $COUNT root-optimized polynomial(s)"
        echo ""
        echo "  Top 10 results (with exp_E correlation):"
        grep "# norm" "$FINAL_DIR/msieve_ropt_inv.p" | head -n 10 || echo "  (No results)"
        echo ""
        BEST_MURPHY=$(grep "# norm" "$FINAL_DIR/msieve_ropt_inv.p" | grep -o " e [0-9.e+-]*" | awk '{print $2}' | sort -g | tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    # CADO original
    echo "3. CADO polyselect_ropt ropteffort=$ROPT_EFFORT (original):"
    if [ -f "$FINAL_DIR/cado_ropt_orig.txt" ]; then
        COUNT=$(grep -c "### Root-optimized polynomial" "$FINAL_DIR/cado_ropt_orig.txt" 2>/dev/null || echo 0)
        if [ "$COUNT" -gt 0 ]; then
            echo "  Found $COUNT root-optimized polynomial(s)"
        fi
        grep -E "lognorm|exp_E|MurphyE" "$FINAL_DIR/cado_ropt_orig.txt" || echo "  (No results)"
    else
        echo "  (No output file)"
    fi
    echo ""

    # CADO inverted
    echo "4. CADO polyselect_ropt ropteffort=$ROPT_EFFORT (inverted):"
    if [ -f "$FINAL_DIR/cado_ropt_inv.txt" ]; then
        COUNT=$(grep -c "### Root-optimized polynomial" "$FINAL_DIR/cado_ropt_inv.txt" 2>/dev/null || echo 0)
        if [ "$COUNT" -gt 0 ]; then
            echo "  Found $COUNT root-optimized polynomial(s)"
        fi
        grep -E "lognorm|exp_E|MurphyE" "$FINAL_DIR/cado_ropt_inv.txt" || echo "  (No results)"
    else
        echo "  (No output file)"
    fi
    echo ""

    echo "======================================"
    echo "OUTPUT FILES"
    echo "======================================"
    echo "Working directory: $WORK_DIR/"
    echo "  top${TOP_N_EXTRACT}_input.ms - Top $TOP_N_EXTRACT input polynomials"
    echo "  resopt_sorted.txt - Re-sopt results (CADO format, sorted)"
    echo "  resopt_msieve_sorted.ms - Re-sopt results (msieve format, sorted)"
    echo ""
    echo "Final results directory: $FINAL_DIR/"
    echo "  best${TOP_M_MSIEVE}_msieve.ms - Best $TOP_M_MSIEVE for msieve ropt"
    echo "  best${TOP_M_MSIEVE}_msieve_inv.ms - Inverted version"
    echo "  best${TOP_M_CADO}_cado.txt - Best $TOP_M_CADO for CADO ropt"
    echo "  best${TOP_M_CADO}_cado_inv.txt - Inverted version"
    echo "  msieve_ropt_orig.p - msieve ropt results (original, $TOP_M_MSIEVE polys)"
    echo "  msieve_ropt_inv.p - msieve ropt results (inverted, $TOP_M_MSIEVE polys)"
    echo "  cado_ropt_orig.txt - CADO ropt results (original, $TOP_M_CADO polys)"
    echo "  cado_ropt_inv.txt - CADO ropt results (inverted, $TOP_M_CADO polys)"
    echo "======================================"

} > "$FINAL_DIR/pipeline_report.txt"

cat "$FINAL_DIR/pipeline_report.txt"

echo ""
echo "======================================"
echo "PIPELINE COMPLETE!"
echo "======================================"
echo "Full report: $FINAL_DIR/pipeline_report.txt"
echo "All results in: $FINAL_DIR/"
echo ""
