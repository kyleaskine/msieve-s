#!/bin/bash

# Run msieve -npr on polynomials and annotate output with exp_E
# Supports parallel processing with multiple threads

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: run_msieve_ropt_annotated.sh <input_ms_file> <output_p_file> <poly_degree> [threads]" >&2
    echo "  Processes polynomials and annotates with original exp_E" >&2
    echo "  threads: Number of parallel msieve processes (default: 1)" >&2
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
POLY_DEGREE="$3"
THREADS="${4:-1}"  # Default to 1 thread if not specified

MSIEVE="./msieve"

if [ ! -f "$MSIEVE" ]; then
    echo "Error: msieve not found at $MSIEVE" >&2
    exit 1
fi

# Clean up output file and work directory
rm -f "$OUTPUT_FILE"
WORK_DIR="msieve_ropt_work_$$"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Convert msieve path to absolute
if [[ "$MSIEVE" = /* ]]; then
    MSIEVE_ABS="$MSIEVE"
else
    MSIEVE_ABS="$(pwd)/$MSIEVE"
fi

# Get absolute path to worktodo.ini
ROOT_DIR="$(pwd)"
WORKTODO_PATH="$ROOT_DIR/worktodo.ini"

# Get total number of polynomials
TOTAL_POLYS=$(wc -l < "$INPUT_FILE")
echo "Processing $TOTAL_POLYS polynomial(s) with $THREADS thread(s)..."

# Read input file line by line and process
POLY_NUM=0
ACTIVE_JOBS=0

while IFS= read -r line; do
    POLY_NUM=$((POLY_NUM + 1))

    # Extract exp_E from the line
    if [ "$POLY_DEGREE" -eq 6 ]; then
        EXP_E=$(echo "$line" | awk '{print $11}')
    else
        EXP_E=$(echo "$line" | awk '{print $10}')
    fi

    # Create work subdirectory for this polynomial
    WORK_SUBDIR="$WORK_DIR/poly_$POLY_NUM"
    mkdir -p "$WORK_SUBDIR"

    # Create input file
    echo "$line" > "$WORK_SUBDIR/poly.ms"

    # Launch background job
    (
        cd "$WORK_SUBDIR"

        # Copy worktodo.ini from root directory (msieve needs this)
        if [ -f "$WORKTODO_PATH" ]; then
            cp "$WORKTODO_PATH" .
        fi

        # Run msieve
        if [ "$POLY_DEGREE" -eq 6 ]; then
            "$MSIEVE_ABS" -npr "polydegree=6" -s poly > poly.log 2>&1
        else
            "$MSIEVE_ABS" -npr -s poly > poly.log 2>&1
        fi

        # Annotate results
        if [ -f "poly.p" ]; then
            awk -v expe="$EXP_E" -v pnum="$POLY_NUM" -v total="$TOTAL_POLYS" '{
                if ($0 ~ /^# norm/) {
                    print $0 " exp_E " expe
                } else {
                    print $0
                }
            }' "poly.p" > "result_$POLY_NUM.p"

            result_count=$(grep -c "^# norm" "poly.p" || echo 0)
            echo "  Polynomial $POLY_NUM/$TOTAL_POLYS (exp_E=$EXP_E): Generated $result_count result(s)"
        else
            touch "result_$POLY_NUM.p"
            echo "  Polynomial $POLY_NUM/$TOTAL_POLYS (exp_E=$EXP_E): Warning - No output generated"
        fi
    ) &

    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    # Limit concurrent jobs
    if [ "$ACTIVE_JOBS" -ge "$THREADS" ]; then
        wait -n  # Wait for any job to complete
        ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    fi
done < "$INPUT_FILE"

# Wait for all remaining jobs
wait

# Collect all results in order
for ((i=1; i<=TOTAL_POLYS; i++)); do
    result_file="$WORK_DIR/poly_$i/result_$i.p"
    if [ -f "$result_file" ] && [ -s "$result_file" ]; then
        cat "$result_file" >> "$OUTPUT_FILE"
    fi
done

# Cleanup work directory
rm -rf "$WORK_DIR"

if [ -f "$OUTPUT_FILE" ]; then
    TOTAL_RESULTS=$(grep -c "^# norm" "$OUTPUT_FILE" || echo 0)
    echo "Total root-optimized results: $TOTAL_RESULTS"
    echo "Output written to: $OUTPUT_FILE"
else
    echo "Error: No output file created" >&2
    exit 1
fi
