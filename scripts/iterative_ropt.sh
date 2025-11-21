#!/bin/bash

# Iterative root optimization following Gimarel's strategy
# Runs msieve -npr multiple times with progressively relaxed stage2_norm

# Signal handling for Ctrl+C
trap 'echo -e "\n\nInterrupted by user. Cleaning up..."; rm -rf "$WORK_DIR" 2>/dev/null; exit 130' INT TERM

show_help() {
    cat << EOF
Usage: iterative_ropt.sh INPUT_FILE [OPTIONS]

Performs fine iterative root optimization on polynomials using Gimarel's strategy.
Assumes msieve already has the 5-pass progressive expansion built in.
This script only does the 100 fine refinement passes with tiny increments.

Arguments:
  INPUT_FILE              File containing polynomials in msieve single-line format
                          (one poly per line: c5 c4 c3 c2 c1 c0 Y1 Y0 alpha norm)

Options:
  -h, --help              Show this help message
  -d, --degree N          Polynomial degree (default: 5)
  -t, --threads N         Parallel workers (default: 4)
  -r, --refinement N      Number of fine refinement passes (default: 100)
  -f, --fine-mult X       Fine refinement multiplier (default: 1.0327)

Output:
  iterative_ropt_best.p   Best polynomial from all iterations

Examples:
  iterative_ropt.sh top_polys.ms                    # Use defaults (100 passes)
  iterative_ropt.sh top_polys.ms -r 200             # 200 refinement passes
  iterative_ropt.sh top_polys.ms -d 6 -t 8          # Degree 6, 8 threads

EOF
    exit 0
}

# Default configuration
DEGREE=5
THREADS=4
REFINEMENT_PASSES=100
REFINEMENT_MULT=1.0327

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
fi

INPUT_FILE="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -d|--degree)
            DEGREE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -r|--refinement)
            REFINEMENT_PASSES="$2"
            shift 2
            ;;
        -f|--fine-mult)
            REFINEMENT_MULT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate input
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

POLY_COUNT=$(wc -l < "$INPUT_FILE")
if [ $POLY_COUNT -eq 0 ]; then
    echo "Error: Input file is empty"
    exit 1
fi

echo "========================================"
echo "Iterative Root Optimization (Fine Refinement)"
echo "========================================"
echo "Input: $INPUT_FILE ($POLY_COUNT polynomials)"
echo "Degree: $DEGREE"
echo "Threads: $THREADS"
echo "Refinement passes: $REFINEMENT_PASSES (multiplier: $REFINEMENT_MULT)"
echo "(msieve already does 5 progressive expansion passes)"
echo ""

# Create working directory
WORK_DIR="iterative_ropt_work"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Function to extract Murphy-E from .p file
get_best_murphy() {
    local pfile=$1
    if [ ! -f "$pfile" ]; then
        echo "1e100"
        return
    fi
    # Extract Murphy-E score from "# norm ... e <value>" line
    # Format: # norm 4.791305e-19 alpha -6.153462 e 9.656e-15 rroots 1 exp_E 52.27
    local murphy=$(grep "^# norm" "$pfile" | head -n 1 | awk '{
        for(i=1; i<=NF; i++) {
            if($i == "e" && i < NF) {
                print $(i+1)
                exit
            }
        }
    }')

    # Return default if extraction failed
    if [ -z "$murphy" ] || [ "$murphy" = "" ]; then
        echo "1e100"
    else
        echo "$murphy"
    fi
}

# Process each polynomial
FINAL_OUTPUT="iterative_ropt_best.p"
rm -f "$FINAL_OUTPUT"

poly_num=0
while IFS= read -r poly_line; do
    poly_num=$((poly_num + 1))

    echo ""
    echo "=========================================="
    echo "Polynomial $poly_num / $POLY_COUNT"
    echo "=========================================="

    # Extract the norm from the input (field 10) to use as base for refinement
    BASE_NORM=$(echo "$poly_line" | awk '{print $10}')
    if [ -z "$BASE_NORM" ]; then
        echo "  Error: No norm found in input"
        continue
    fi

    echo "  Input norm: $BASE_NORM"

    # Extract initial Murphy-E (field 9)
    INITIAL_ALPHA=$(echo "$poly_line" | awk '{print $9}')
    echo "  Input alpha: $INITIAL_ALPHA"

    CURRENT_OUTPUT="$WORK_DIR/poly_${poly_num}_pass"

    # Iterative refinement passes
    echo ""
    echo "Iterative refinement ($REFINEMENT_PASSES passes)"
    echo "  Starting norm: $BASE_NORM"

    # Create all input files first (each with progressively higher stage2_norm)
    for pass in $(seq 1 $REFINEMENT_PASSES); do
        # Calculate stage2_norm = BASE_NORM * REFINEMENT_MULT^pass
        REFINE_NORM=$(awk -v base="$BASE_NORM" -v mult="$REFINEMENT_MULT" -v pass="$pass" 'BEGIN {
            result = base
            for (i = 1; i <= pass; i++) result *= mult
            printf "%.15e", result
        }')

        # Update field 10 with new stage2_norm
        echo "$poly_line" | awk -v norm="$REFINE_NORM" '{$10=norm; print}' > "${CURRENT_OUTPUT}_refine_${pass}.ms"
    done

    echo "  Created $REFINEMENT_PASSES input files"

    # Get absolute path to msieve
    MSIEVE_ABS="$(pwd)/msieve"

    # Run msieve on all passes in parallel
    echo "  Running $REFINEMENT_PASSES passes with $THREADS threads..."

    for pass in $(seq 1 $REFINEMENT_PASSES); do
        # Wait if we've reached the thread limit
        while [ $(jobs -r | wc -l) -ge $THREADS ]; do
            sleep 0.1
        done

        # Create subdirectory for this pass (to isolate msieve.fb)
        PASS_DIR="${CURRENT_OUTPUT}_refine_${pass}_dir"
        mkdir -p "$PASS_DIR"

        # Copy input file to subdirectory
        cp "${CURRENT_OUTPUT}_refine_${pass}.ms" "$PASS_DIR/poly.ms"

        # Copy worktodo.ini if it exists (msieve needs this)
        if [ -f worktodo.ini ]; then
            cp worktodo.ini "$PASS_DIR/"
        fi

        # Run msieve in background from its own subdirectory
        (
            cd "$PASS_DIR"
            "$MSIEVE_ABS" -npr -s poly > poly.log 2>&1

            # Move result back to main work directory
            if [ -f "poly.p" ]; then
                mv poly.p "../$(basename ${CURRENT_OUTPUT}_refine_${pass}).ms.p"
            fi
        ) &

        # Progress reporting
        if [ $((pass % 10)) -eq 0 ]; then
            echo "    Launched pass $pass / $REFINEMENT_PASSES"
        fi
    done

    # Wait for all background jobs to complete
    wait

    echo "  All passes complete"

    # Merge all .p files into final output
    for pass in $(seq 1 $REFINEMENT_PASSES); do
        if [ -f "${CURRENT_OUTPUT}_refine_${pass}.ms.p" ]; then
            cat "${CURRENT_OUTPUT}_refine_${pass}.ms.p" >> "$FINAL_OUTPUT"
            echo "" >> "$FINAL_OUTPUT"
        fi
    done

done < "$INPUT_FILE"

echo ""
echo "=========================================="
echo "Iterative optimization complete"
echo "=========================================="
echo "Output: $FINAL_OUTPUT"

if [ -f "$FINAL_OUTPUT" ]; then
    FINAL_COUNT=$(grep -c "^# norm" "$FINAL_OUTPUT")
    echo "Total polynomials optimized: $FINAL_COUNT"

    # Show best Murphy-E
    BEST=$(grep "^# norm" "$FINAL_OUTPUT" | awk '{
        for(i=1; i<=NF; i++) {
            if($i == "e" && i < NF) {
                print $(i+1)
            }
        }
    }' | sort -g | head -n 1)
    echo "Best Murphy-E: $BEST"
fi

# Cleanup
echo ""
read -p "Remove working directory? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$WORK_DIR"
    echo "Cleaned up working directory"
fi
