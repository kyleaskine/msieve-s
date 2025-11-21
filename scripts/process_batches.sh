#!/bin/bash

# Batch processor for msieve polynomial optimization
# Two-phase approach:
# 1. CADO all new polynomials (cheap, accumulate with alphas)
# 2. Root-optimize top N best from CADO results (expensive, remove after processing)

# Flag for graceful shutdown
SHUTDOWN_REQUESTED=false

# Trap SIGUSR1 for graceful shutdown (use: kill -USR1 <pid>)
# Display PID at startup so user knows what to send signal to
trap 'echo ""; echo "Graceful shutdown requested. Will stop after current cycle completes..."; SHUTDOWN_REQUESTED=true' SIGUSR1

# Show help
show_help() {
    cat << EOF
Usage: process_batches.sh [OPTIONS]

Two-phase polynomial processor for msieve/CADO-NFS integration

Phase 1: CADO size optimization (processes all new polynomials)
Phase 2: Root optimization (processes top N polynomials at a time)

Options:
  -h, --help           Show this help message and exit
  -t, --threads N      Set number of parallel msieve workers (default: 4)
  -b, --batch-size N   Set root optimization batch size (default: 100)
  -s, --sleep N        Set sleep interval between cycles in seconds (default: 0)

Configuration files:
  Input:  msieve.dat.ms         (stage1 polynomials)
  Output: outMsieve.p           (root-optimized results)
  Tracking: .cado_processed_lines (progress tracking)
  Intermediate: cado_results.ms  (accumulated CADO results)

Examples:
  process_batches.sh                    # Use default settings
  process_batches.sh -t 8               # Use 8 parallel workers
  process_batches.sh -t 8 -b 200        # 8 workers, batch size 200

EOF
    exit 0
}

# Default configuration
INPUT_FILE="msieve.dat.ms"
CADO_PROCESSED_LINES=".cado_processed_lines"
CADO_RESULTS="cado_results.ms"  # Accumulated CADO results with alphas
ROOTOPT_BATCH_SIZE=100  # Process top N per cycle
PARALLEL_THREADS=4  # Number of parallel workers (both CADO and msieve)
SLEEP_INTERVAL=0  # Sleep between cycles (default: 0 seconds)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -t|--threads)
            PARALLEL_THREADS="$2"
            shift 2
            ;;
        -b|--batch-size)
            ROOTOPT_BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--sleep)
            SLEEP_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Intermediate files
BATCH_INPUT="batch_input.ms"
BATCH_LINES="batch_lines.tmp"
CADO_OUTPUT="outCado.ms"
MSIEVE_INPUT="outMsieve.ms"

# Final output
FINAL_OUTPUT="outMsieve.p"

# CADO-NFS path
CADO_SOPT="$HOME/cado-nfs/build/Kyle-PC-V2/polyselect/sopt"

# Check dependencies
if [ ! -f "$CADO_SOPT" ]; then
    echo "Error: CADO sopt not found at $CADO_SOPT"
    exit 1
fi

if [ ! -f "utils/cado_to_msieve.py" ]; then
    echo "Error: cado_to_msieve.py not found in utils directory"
    exit 1
fi

if [ ! -f "./msieve" ]; then
    echo "Error: msieve not found in utils directory"
    exit 1
fi

echo "======================================"
echo "Two-Phase Polynomial Processor"
echo "======================================"
echo "Process ID: $$"
echo "For graceful shutdown: kill -USR1 $$"
echo ""
echo "Phase 1: CADO size optimization (all new polynomials, $PARALLEL_THREADS threads)"
echo "Phase 2: Root optimization (top $ROOTOPT_BATCH_SIZE at a time, $PARALLEL_THREADS threads)"
echo "Input: $INPUT_FILE"
echo "CADO results: $CADO_RESULTS"
echo "Final output: $FINAL_OUTPUT"
echo ""

# Initialize tracking files
touch "$CADO_PROCESSED_LINES"
touch "$CADO_RESULTS"

# Function to count complete polynomials in file
count_complete_polys() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo 0
        return
    fi
    # Count "n:" lines (each poly starts with "n:")
    grep -c "^n:" "$file" || echo 0
}

# Function to extract complete polynomials by line range
extract_complete_polys() {
    local first_line=$1
    local last_line=$2

    # Adjust FIRST_LINE to start of polynomial (find previous "n:" line)
    while [ $first_line -gt 1 ]; do
        LINE_CONTENT=$(sed -n "${first_line}p" "$INPUT_FILE")
        if [[ "$LINE_CONTENT" == n:* ]]; then
            break
        fi
        first_line=$((first_line - 1))
    done

    # Adjust LAST_LINE to end of polynomial (find next blank line or EOF)
    TOTAL_LINES_IN_FILE=$(wc -l < "$INPUT_FILE")
    while [ $last_line -le $TOTAL_LINES_IN_FILE ]; do
        LINE_CONTENT=$(sed -n "${last_line}p" "$INPUT_FILE")
        if [[ -z "$LINE_CONTENT" ]]; then
            break
        fi
        last_line=$((last_line + 1))
    done

    echo "$first_line $last_line"
}

# Main loop
while true; do
    echo ""
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting cycle"
    echo "=========================================="

    # PHASE 1: Process new polynomials through CADO
    echo ""
    echo "=== PHASE 1: CADO SIZE OPTIMIZATION ==="

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Waiting for $INPUT_FILE to be created..."
        sleep "$SLEEP_INTERVAL"
        continue
    fi

    TOTAL_LINES=$(wc -l < "$INPUT_FILE")
    CADO_PROCESSED_COUNT=$(wc -l < "$CADO_PROCESSED_LINES")

    # Find unprocessed lines
    if [ $CADO_PROCESSED_COUNT -eq 0 ]; then
        # First run, process everything
        UNPROCESSED_START=1
        UNPROCESSED_END=$TOTAL_LINES
    else
        # Get last processed line
        LAST_CADO_LINE=$(sort -n "$CADO_PROCESSED_LINES" | tail -n 1)
        UNPROCESSED_START=$((LAST_CADO_LINE + 1))
        UNPROCESSED_END=$TOTAL_LINES
    fi

    if [ $UNPROCESSED_START -gt $TOTAL_LINES ]; then
        echo "No new polynomials to process through CADO"
        echo "Processed: $CADO_PROCESSED_COUNT/$TOTAL_LINES lines"
    else
        NEW_LINES=$((UNPROCESSED_END - UNPROCESSED_START + 1))
        echo "Found $NEW_LINES new lines (lines $UNPROCESSED_START-$UNPROCESSED_END)"

        # Extract complete polynomials
        read FIRST LAST < <(extract_complete_polys $UNPROCESSED_START $UNPROCESSED_END)
        echo "Adjusted to complete polynomials: lines $FIRST-$LAST"

        sed -n "${FIRST},${LAST}p" "$INPUT_FILE" > "$BATCH_INPUT"
        BATCH_LINES_COUNT=$((LAST - FIRST + 1))
        POLY_COUNT=$(count_complete_polys "$BATCH_INPUT")

        echo "Extracted $BATCH_LINES_COUNT lines ($POLY_COUNT polynomials)"

        if [ $POLY_COUNT -gt 0 ]; then
            # Deduplicate batch before CADO processing
            echo "Deduplicating polynomials..."
            BATCH_DEDUPED="${BATCH_INPUT}.deduped"
            awk '
            BEGIN {
                poly = ""
                in_poly = 0
            }
            /^n:/ {
                if (poly != "" && !(poly in seen)) {
                    print poly
                    seen[poly] = 1
                }
                poly = $0
                in_poly = 1
                next
            }
            /^[[:space:]]*$/ {
                if (in_poly && poly != "" && !(poly in seen)) {
                    print poly
                    print ""
                    seen[poly] = 1
                }
                poly = ""
                in_poly = 0
                next
            }
            {
                if (in_poly) {
                    poly = poly "\n" $0
                }
            }
            END {
                if (poly != "" && !(poly in seen)) {
                    print poly
                }
            }
            ' "$BATCH_INPUT" > "$BATCH_DEDUPED"

            DEDUPED_COUNT=$(grep -c "^n:" "$BATCH_DEDUPED" || echo 0)
            DUPLICATES=$((POLY_COUNT - DEDUPED_COUNT))
            echo "  After dedup: $DEDUPED_COUNT polynomials ($DUPLICATES duplicates removed)"

            if [ $DEDUPED_COUNT -eq 0 ]; then
                echo "  No unique polynomials to process"
                rm -f "$BATCH_DEDUPED"
            else
                # Run CADO (multithreaded if > 1 thread)
                echo "Running CADO size optimization ($PARALLEL_THREADS threads)..."

                if [ "$PARALLEL_THREADS" -eq 1 ]; then
                    # Single-threaded
                    if "$CADO_SOPT" -sopteffort 2 -inputpolys "$BATCH_DEDUPED" > "$CADO_OUTPUT" 2>&1; then
                        CADO_SUCCESS=1
                    else
                        CADO_SUCCESS=0
                    fi
                else
                    # Multi-threaded: split and process in parallel
                    SOPT_CHUNK_DIR="sopt_batch_chunks"
                    rm -rf "$SOPT_CHUNK_DIR"
                    mkdir -p "$SOPT_CHUNK_DIR"

                    POLYS_PER_CHUNK=$(( (DEDUPED_COUNT + PARALLEL_THREADS - 1) / PARALLEL_THREADS ))

                    # Split into chunks
                    awk -v threads="$PARALLEL_THREADS" -v outdir="$SOPT_CHUNK_DIR" -v per_chunk="$POLYS_PER_CHUNK" '
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
                    ' "$BATCH_DEDUPED"

                    # Run sopt on each chunk in parallel
                    PIDS=()
                    for chunk in "$SOPT_CHUNK_DIR"/chunk_*.ms; do
                        chunk_basename=$(basename "$chunk" .ms)
                        output_file="$SOPT_CHUNK_DIR/sopt_${chunk_basename}.out"

                        (
                            "$CADO_SOPT" -sopteffort 2 -inputpolys "$chunk" > "$output_file" 2>&1
                        ) &
                        PIDS+=($!)
                    done

                    # Wait for all sopt processes
                    FAILED=0
                    for pid in "${PIDS[@]}"; do
                        if ! wait "$pid"; then
                            FAILED=$((FAILED + 1))
                        fi
                    done

                    if [ $FAILED -eq 0 ]; then
                        # Merge results
                        cat "$SOPT_CHUNK_DIR"/sopt_chunk_*.out > "$CADO_OUTPUT"
                        CADO_SUCCESS=1
                    else
                        echo "  Warning: $FAILED CADO sopt process(es) failed"
                        CADO_SUCCESS=0
                    fi

                    # Cleanup chunks
                    rm -rf "$SOPT_CHUNK_DIR"
                fi

                if [ $CADO_SUCCESS -eq 1 ]; then
                    CADO_POLYS=$(grep -c "^### Size-optimized polynomial" "$CADO_OUTPUT" || true)
                    echo "  CADO optimized $CADO_POLYS polynomials"

                    # Convert and append to accumulated results
                    echo "Converting to msieve format and appending to results..."
                    if python3 utils/cado_to_msieve.py "$CADO_OUTPUT" /tmp/cado_batch.ms 2>&1 | grep -E "Found|Wrote|Best|Worst"; then
                        cat /tmp/cado_batch.ms >> "$CADO_RESULTS"
                        rm /tmp/cado_batch.ms

                        # Mark lines as processed
                        for ((line=$FIRST; line<=$LAST; line++)); do
                            echo $line >> "$CADO_PROCESSED_LINES"
                        done

                        TOTAL_CADO_POLYS=$(wc -l < "$CADO_RESULTS")
                        echo "  Total CADO results accumulated: $TOTAL_CADO_POLYS polynomials"
                    else
                        echo "  Warning: Conversion failed, skipping this batch"
                    fi
                else
                    echo "  Warning: CADO failed, skipping this batch"
                fi

                rm -f "$BATCH_DEDUPED"
            fi
        fi
    fi

    # Check if shutdown was requested after Phase 1
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        echo ""
        echo "Shutdown requested - skipping Phase 2"
        echo "=========================================="
        echo "Graceful shutdown complete"
        echo "=========================================="
        exit 0
    fi

    # PHASE 2: Root optimize best N from accumulated CADO results
    echo ""
    echo "=== PHASE 2: ROOT OPTIMIZATION ==="

    CADO_RESULTS_COUNT=$(wc -l < "$CADO_RESULTS")

    if [ $CADO_RESULTS_COUNT -eq 0 ]; then
        echo "No CADO results available for root optimization"
        echo "Sleeping for $SLEEP_INTERVAL seconds..."
        sleep "$SLEEP_INTERVAL"
        continue
    fi

    echo "CADO results available: $CADO_RESULTS_COUNT polynomials"

    # Sort by alpha (column 9) and take top N
    BATCH_SIZE=$ROOTOPT_BATCH_SIZE
    if [ $CADO_RESULTS_COUNT -lt $BATCH_SIZE ]; then
        BATCH_SIZE=$CADO_RESULTS_COUNT
    fi

    echo "Selecting top $BATCH_SIZE polynomials by exp_E..."

    # Clean up old input files only (NOT .p files yet - we need those after msieve runs)
    rm -f "$MSIEVE_INPUT" "$MSIEVE_INPUT.tmp"

    # Detect polynomial degree from number of columns
    # Format for deg5: c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm (11 columns)
    # Format for deg6: c6 c5 c4 c3 c2 c1 c0 Y1 Y0 alpha exp_E norm (12 columns)
    NUM_COLS=$(head -n 1 "$CADO_RESULTS" | wc -w)

    if [ "$NUM_COLS" -eq 12 ]; then
        POLY_DEGREE=6
        EXPE_COL=11
        echo "  Detected degree 6 polynomial format (12 columns)"
    else
        POLY_DEGREE=5
        EXPE_COL=10
        echo "  Detected degree 5 polynomial format (11 columns)"
    fi

    # Sort by exp_E column, take top N (single-line format)
    # Lower exp_E is better (expected difficulty)
    sort -k${EXPE_COL},${EXPE_COL}n "$CADO_RESULTS" | head -n "$BATCH_SIZE" > "$MSIEVE_INPUT.tmp"

    # Get exp_E range for reporting
    BEST_EXPE=$(head -n 1 "$MSIEVE_INPUT.tmp" | awk -v col=$EXPE_COL '{print $col}')
    WORST_EXPE=$(tail -n 1 "$MSIEVE_INPUT.tmp" | awk -v col=$EXPE_COL '{print $col}')
    echo "  exp_E range: $BEST_EXPE (best) to $WORST_EXPE (worst)"

    # Strip exp_E column to create msieve-compatible format
    if [ "$POLY_DEGREE" -eq 6 ]; then
        # Format for deg6: c6 c5 c4 c3 c2 c1 c0 Y1 Y0 alpha norm (columns 1-10, 12 of cado output)
        awk '{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $12}' "$MSIEVE_INPUT.tmp" > "$MSIEVE_INPUT"
    else
        # Format for deg5: c5 c4 c3 c2 c1 c0 Y1 Y0 alpha norm (columns 1-9, 11 of cado output)
        awk '{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $11}' "$MSIEVE_INPUT.tmp" > "$MSIEVE_INPUT"
    fi

    # Run msieve root optimization (in parallel)
    echo "Running msieve root optimization on $BATCH_SIZE polynomials ($PARALLEL_THREADS parallel workers)..."
    START_TIME=$(date +%s)

    # Split into chunks for parallel processing
    # Single-line format: 1 line per polynomial
    CHUNK_SIZE=$(( (BATCH_SIZE + PARALLEL_THREADS - 1) / PARALLEL_THREADS ))

    # Clean up old chunk files and msieve intermediate files from previous batch
    # NOTE: Do NOT delete msieve.dat.ms (stage1 input) or any .ms/.dat files!
    rm -f chunk_*.ms chunk_*.p chunk_*.log chunk_*.fb msieve.fb

    # Split input into chunks (1 line per polynomial in single-line format)
    split -l "$CHUNK_SIZE" -d "$MSIEVE_INPUT" chunk_
    CHUNK_COUNT=$(ls chunk_* 2>/dev/null | wc -l)
    echo "  Split into $CHUNK_COUNT chunks of ~$CHUNK_SIZE polynomials each"

    # Process chunks in parallel
    PIDS=()
    WORKER_NUM=0
    for chunk in chunk_*; do
        chunk_name="${chunk}"
        WORKER_NUM=$((WORKER_NUM + 1))
        (
            # Rename to .ms extension for msieve
            mv "$chunk" "${chunk_name}.ms"
            # IMPORTANT: NFS args must come immediately after -npr, before -s
            # See demo.c:466-471 - parser only looks for args right after -npr
            if [ "$POLY_DEGREE" -eq 6 ]; then
                ./msieve -npr "polydegree=6" -s "$chunk_name" > "${chunk_name}.log" 2>&1
            else
                ./msieve -npr -s "$chunk_name" > "${chunk_name}.log" 2>&1
            fi
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "Worker $WORKER_NUM failed with exit code $exit_code" >> rootopt_errors.log
                cat "${chunk_name}.log" >> rootopt_errors.log
            fi
            exit $exit_code
        ) &
        PIDS+=($!)
    done

    # Wait for all processes
    echo "  Waiting for $CHUNK_COUNT parallel msieve processes..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=$((FAILED + 1))
        fi
    done

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $FAILED -eq 0 ]; then
        echo "  Root optimization completed in ${DURATION}s"
    else
        if [ "$SHUTDOWN_REQUESTED" = false ]; then
            echo "  Error: $FAILED msieve -npr process(es) failed"
            echo "  Chunk files preserved for debugging"
        else
            echo "  Warning: $FAILED worker(s) failed during shutdown - continuing cleanup"
        fi
    fi

    # Always merge results and cleanup during graceful shutdown, even if some workers failed
    if [ $FAILED -eq 0 ] || [ "$SHUTDOWN_REQUESTED" = true ]; then
        # Merge all .p files that were successfully created
        cat chunk_*.p >> "$FINAL_OUTPUT" 2>/dev/null || true

        if [ -f "$FINAL_OUTPUT" ]; then
            TOTAL_ROOTOPT=$(grep -c "^# norm" "$FINAL_OUTPUT" || echo "0")
            echo "  Total root-optimized polynomials: $TOTAL_ROOTOPT"
        fi

        # Remove processed polynomials from CADO results
        echo "Removing processed polynomials from CADO results..."
        sort -k${EXPE_COL},${EXPE_COL}n "$CADO_RESULTS" | tail -n +$((BATCH_SIZE + 1)) > "$CADO_RESULTS.tmp"
        mv "$CADO_RESULTS.tmp" "$CADO_RESULTS"

        REMAINING=$(wc -l < "$CADO_RESULTS")
        echo "  Remaining CADO results: $REMAINING polynomials"

        # Cleanup chunk files and msieve intermediate files
        rm -f chunk_*.ms chunk_*.p chunk_*.log chunk_*.fb msieve.fb
    fi

    # Check if shutdown was requested
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        echo ""
        echo "=========================================="
        echo "Graceful shutdown complete"
        echo "=========================================="
        exit 0
    fi

    # Sleep before next cycle (if configured)
    if [ $SLEEP_INTERVAL -gt 0 ]; then
        echo ""
        echo "Sleeping for $SLEEP_INTERVAL seconds before next cycle..."
        sleep $SLEEP_INTERVAL
    fi

    echo ""
    echo "Cycle complete. Starting next cycle..."
done
