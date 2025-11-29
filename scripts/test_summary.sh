#!/bin/bash

# Standalone script to test the summary section
# Uses existing pipeline results without re-running the pipeline

set -euo pipefail

# Configuration - adjust these if your setup is different
FINAL_DIR="pipeline_results"
WORK_DIR="pipeline_work"

# Default values (can be overridden via command line)
TOP_N_EXTRACT=100
TOP_M_MSIEVE=10
TOP_M_CADO=100
RESOPT_EFFORT=10
ROPT_EFFORT=10
THREADS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --final-dir)
            FINAL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if results exist
if [ ! -d "$FINAL_DIR" ]; then
    echo "Error: Results directory not found: $FINAL_DIR"
    exit 1
fi

echo "Testing summary with results from: $FINAL_DIR"
echo ""

# Detect polynomial degree from msieve file if it exists
EXPE_COL=10
if [ -f "$WORK_DIR/resopt_msieve_sorted.ms" ]; then
    NUM_COLS=$(head -n 1 "$WORK_DIR/resopt_msieve_sorted.ms" | wc -w)
    if [ "$NUM_COLS" -eq 12 ]; then
        EXPE_COL=11
        POLY_DEGREE=6
    else
        EXPE_COL=10
        POLY_DEGREE=5
    fi
elif [ -f "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" ]; then
    NUM_COLS=$(head -n 1 "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" | wc -w)
    if [ "$NUM_COLS" -eq 12 ]; then
        EXPE_COL=11
        POLY_DEGREE=6
    else
        EXPE_COL=10
        POLY_DEGREE=5
    fi
else
    POLY_DEGREE=5
fi

# Get exp_E ranges
if [ -f "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" ]; then
    MSIEVE_BEST_EXPE=$(head -n 1 "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" | awk -v col=$EXPE_COL '{print $col}')
    MSIEVE_WORST_EXPE=$(tail -n 1 "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" | awk -v col=$EXPE_COL '{print $col}')
else
    MSIEVE_BEST_EXPE="N/A"
    MSIEVE_WORST_EXPE="N/A"
fi

if [ -f "$WORK_DIR/resopt_msieve_sorted.ms" ]; then
    head -n "$TOP_M_CADO" "$WORK_DIR/resopt_msieve_sorted.ms" > /tmp/temp_cado_check.ms 2>/dev/null || true
    if [ -f /tmp/temp_cado_check.ms ]; then
        CADO_BEST_EXPE=$(head -n 1 /tmp/temp_cado_check.ms | awk -v col=$EXPE_COL '{print $col}')
        CADO_WORST_EXPE=$(tail -n 1 /tmp/temp_cado_check.ms | awk -v col=$EXPE_COL '{print $col}')
        rm -f /tmp/temp_cado_check.ms
    else
        CADO_BEST_EXPE="N/A"
        CADO_WORST_EXPE="N/A"
    fi
else
    CADO_BEST_EXPE="N/A"
    CADO_WORST_EXPE="N/A"
fi

# Generate the summary (same as in full_optimization_pipeline.sh)
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
        echo "  Top 10 results (sorted by Murphy E, column 7):"
        grep '^#' "$FINAL_DIR/msieve_ropt_orig.p" | LANG=C sort -rgk7 | uniq | head -n10 || echo "  (No results)"
        echo ""
        BEST_MURPHY=$(grep '^#' "$FINAL_DIR/msieve_ropt_orig.p" | awk '{print $7}' | sort -g | tail -n 1 || echo "N/A")
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
        echo "  Top 10 results (sorted by Murphy E, column 7):"
        grep '^#' "$FINAL_DIR/msieve_ropt_inv.p" | LANG=C sort -rgk7 | uniq | head -n10 || echo "  (No results)"
        echo ""
        BEST_MURPHY=$(grep '^#' "$FINAL_DIR/msieve_ropt_inv.p" | awk '{print $7}' | sort -g | tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    # CADO original
    echo "3. CADO polyselect_ropt ropteffort=$ROPT_EFFORT (original):"
    if [ -f "$FINAL_DIR/cado_ropt_orig.txt" ]; then
        COUNT=$(grep -c "### Root-optimized polynomial" "$FINAL_DIR/cado_ropt_orig.txt" 2>/dev/null || true)
        if [ -n "$COUNT" ] && [ "$COUNT" -gt 0 ] 2>/dev/null; then
            echo "  Found $COUNT root-optimized polynomial(s)"
        fi
        echo ""
        echo "  Top 10 results (sorted by Murphy E, highest first):"
        grep '^# side 1 MurphyE' "$FINAL_DIR/cado_ropt_orig.txt" 2>/dev/null | \
            awk -F= '{val=$NF; $0=$0; print val " ||| " $0}' | \
            sort -k1 -gr | \
            head -n10 | \
            cut -d'|' -f4- | \
            sed 's/^/ /' || echo "  (No results)"
        echo ""
        BEST_MURPHY=$(grep '^# side 1 MurphyE' "$FINAL_DIR/cado_ropt_orig.txt" 2>/dev/null | \
            awk -F= '{print $NF}' | \
            sort -g | \
            tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    # CADO inverted
    echo "4. CADO polyselect_ropt ropteffort=$ROPT_EFFORT (inverted):"
    if [ -f "$FINAL_DIR/cado_ropt_inv.txt" ]; then
        COUNT=$(grep -c "### Root-optimized polynomial" "$FINAL_DIR/cado_ropt_inv.txt" 2>/dev/null || true)
        if [ -n "$COUNT" ] && [ "$COUNT" -gt 0 ] 2>/dev/null; then
            echo "  Found $COUNT root-optimized polynomial(s)"
        fi
        echo ""
        echo "  Top 10 results (sorted by Murphy E, highest first):"
        grep '^# side 1 MurphyE' "$FINAL_DIR/cado_ropt_inv.txt" 2>/dev/null | \
            awk -F= '{val=$NF; $0=$0; print val " ||| " $0}' | \
            sort -k1 -gr | \
            head -n10 | \
            cut -d'|' -f4- | \
            sed 's/^/ /' || echo "  (No results)"
        echo ""
        BEST_MURPHY=$(grep '^# side 1 MurphyE' "$FINAL_DIR/cado_ropt_inv.txt" 2>/dev/null | \
            awk -F= '{print $NF}' | \
            sort -g | \
            tail -n 1 || echo "N/A")
        echo "  Best Murphy E: $BEST_MURPHY"
    else
        echo "  (No output file)"
    fi
    echo ""

    echo "======================================"
    echo "BEST POLYNOMIAL FROM EACH METHOD"
    echo "======================================"
    echo ""

    # Helper function to extract complete msieve polynomial
    extract_best_msieve_poly() {
        local file=$1
        local method_name=$2

        if [ ! -f "$file" ]; then
            echo "  (No output file)"
            echo ""
            return
        fi

        # Find the line with best Murphy E (column 7)
        local best_line=$(grep '^#' "$file" | LANG=C sort -rgk7 | head -n1)

        if [ -z "$best_line" ]; then
            echo "  (No polynomials found)"
            echo ""
            return
        fi

        echo "  $best_line"
        echo ""

        # Extract polynomial body (lines after the # line until next # or end)
        # Find line number of best polynomial
        local line_num=$(grep -n "^$best_line$" "$file" | head -n1 | cut -d: -f1)

        if [ -n "$line_num" ]; then
            # Print the polynomial starting from line after the comment
            tail -n +$((line_num + 1)) "$file" | awk '
                /^#/ { exit }
                /^skew:/ || /^[cnY][0-9]+:/ { print $0 }
            ' || true
        fi
        echo ""
    }

    # Helper function to extract complete CADO polynomial
    extract_best_cado_poly() {
        local file=$1
        local method_name=$2

        if [ ! -f "$file" ]; then
            echo "  (No output file)"
            echo ""
            return
        fi

        # Find the Murphy E line with best score (using proper numeric sort on last field)
        local best_murphy_line=$(grep '^# side 1 MurphyE' "$file" 2>/dev/null | \
            awk -F= '{val=$NF; $0=$0; print val " ||| " $0}' | \
            sort -k1 -gr | \
            head -n1 | \
            cut -d'|' -f4- | \
            sed 's/^ //')

        if [ -z "$best_murphy_line" ]; then
            echo "  (No polynomials found)"
            echo ""
            return
        fi

        # Find the polynomial block containing this Murphy E line
        local line_num=$(grep -n -F "$best_murphy_line" "$file" | head -n1 | cut -d: -f1)

        if [ -n "$line_num" ]; then
            echo "  Murphy E: $(echo "$best_murphy_line" | grep -o '=[0-9.e+-]*$' | tr -d '=')"
            echo ""

            # Go backwards to find the start (### root-optimized polynomial)
            local start_line=$(head -n $line_num "$file" | grep -n '### root-optimized polynomial' | tail -n1 | cut -d: -f1)

            if [ -n "$start_line" ]; then
                # Extract from start to just after this Murphy E line (including exp_E and lognorm)
                tail -n +$start_line "$file" | awk '
                    BEGIN { in_poly = 0; lines_after_murphy = 0 }
                    /^### root-optimized polynomial/ { in_poly = 1 }
                    in_poly {
                        if (/^$/ && lines_after_murphy > 2) { exit }
                        if (/^# side 1 MurphyE/) { lines_after_murphy = 1 }
                        if (lines_after_murphy > 0) { lines_after_murphy++ }
                        print $0
                    }
                ' | head -n 20 || true
            fi
        fi
        echo ""
    }

    # Extract best from each method
    echo "1. msieve -npr (original):"
    extract_best_msieve_poly "$FINAL_DIR/msieve_ropt_orig.p" "msieve -npr (original)"

    echo "2. msieve -npr (inverted):"
    extract_best_msieve_poly "$FINAL_DIR/msieve_ropt_inv.p" "msieve -npr (inverted)"

    echo "3. CADO ropt (original):"
    extract_best_cado_poly "$FINAL_DIR/cado_ropt_orig.txt" "CADO ropt (original)"

    echo "4. CADO ropt (inverted):"
    extract_best_cado_poly "$FINAL_DIR/cado_ropt_inv.txt" "CADO ropt (inverted)"

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
}

echo ""
echo "Summary test complete!"
