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

# Paths (use environment variables if set via nfs_optimize.sh, otherwise use defaults)
CADO_SOPT="${CADO_SOPT:-$HOME/cado-nfs/build/localhost/polyselect/sopt}"
CADO_ROPT="${CADO_ROPT:-$HOME/cado-nfs/build/localhost/polyselect/polyselect_ropt}"
MSIEVE="${MSIEVE:-./msieve}"
SKEWOPT="${SKEWOPT:-}"  # Optional: path to skewopt binary
REPORT_ONLY=0            # If 1, skip phases 1-6 and only regenerate the report

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
  --report-only           Skip phases 1-6; regenerate the report from results
                          already in $FINAL_DIR/ (skewopt results are reprinted)
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

Report-only mode:
  --report-only re-runs just the final reporting phase against the results
  already on disk. It is safe to use after the pipeline has finished (or if
  you lost the terminal it was printing to). Counts and efforts are read from
  $FINAL_DIR/pipeline_settings.txt, which the pipeline writes when it starts
  filling $FINAL_DIR/ and marks complete when root optimization finishes. For
  results from before that file existed, they are recovered from the previous
  pipeline_report.txt; anything that can't be recovered is shown as "unknown".
  Values passed explicitly apply to this one report only. Existing skewopt
  results are reprinted; skewopt only runs if they are missing.

Output:
  $FINAL_DIR/ - All final results and comparison. When a new run starts writing
  there (phase 3), the previous run's files are moved to $FINAL_DIR/previous/<timestamp>/
  (the newest KEEP_PREVIOUS, default 3, are kept).

EOF
    exit 0
}

SETTINGS_FILE="$FINAL_DIR/pipeline_settings.txt"
SETTINGS_KEYS="TOP_N_EXTRACT RESOPT_EFFORT TOP_M_MSIEVE TOP_M_CADO ROPT_EFFORT THREADS"

# Record the settings of the run that is filling $FINAL_DIR/ ($1 = running|complete).
# The values come from the variables named by SETTINGS_KEYS, with prefix $2 if given.
write_settings() {
    local key src
    {
        echo "# Written by full_optimization_pipeline.sh; read by --report-only"
        echo "STATUS=$1"
        echo "UPDATED=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "FAILED=${ROPT_FAILED:-}"
        for key in $SETTINGS_KEYS; do
            src="${2:-}$key"
            echo "$key=${!src}"
        done
    } > "$SETTINGS_FILE.tmp"
    mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"
}

# Record a root-optimization pass ($1) as failed if its output ($2) is missing or has
# no line matching $3.
ROPT_FAILED=""
check_ropt_output() {
    if [ ! -f "$2" ] || ! grep -q "$3" "$2"; then
        echo "  WARNING: $1 produced no results ($2)"
        ROPT_FAILED="${ROPT_FAILED:+$ROPT_FAILED,}$1"
    fi
}

# A new run starts from an empty $FINAL_DIR/: move the previous run's files into
# $FINAL_DIR/previous/<timestamp>/, so nothing from an older run (other-size best*
# files, skewopt results, report backups) can be reported as this run's. Only the newest
# KEEP_PREVIOUS archives are kept (environment, default 3).
KEEP_PREVIOUS="${KEEP_PREVIOUS:-3}"
archive_previous_results() {
    local f dest moved=0
    dest="$FINAL_DIR/previous/$(date +%Y%m%d_%H%M%S)"
    for f in "$FINAL_DIR"/*; do
        if [ ! -e "$f" ] || [ "$(basename "$f")" = previous ]; then continue; fi
        mkdir -p "$dest"
        mv "$f" "$dest/"
        moved=1
    done
    if [ "$moved" -eq 1 ]; then
        echo "  Previous results moved to $dest/"
    fi
    # Keep only the newest $KEEP_PREVIOUS archives (timestamped names sort by age)
    local old
    old=$(ls -d "$FINAL_DIR"/previous/*/ 2>/dev/null | sort | head -n -"$KEEP_PREVIOUS" || true)
    if [ -n "$old" ]; then
        echo "$old" | while IFS= read -r d; do rm -rf "$d"; done
        echo "  Removed $(echo "$old" | wc -l) older archive(s); keeping the newest $KEEP_PREVIOUS"
    fi
}

# Print the value of KEY ($1) from the settings file, or nothing
read_setting() {
    sed -n "s/^$1=\([A-Za-z0-9._:,-]*\)\$/\1/p" "$SETTINGS_FILE" | head -n1
}

# Print "best worst" exp_E of an msieve-format selection file. exp_E is the
# second-to-last column for every degree (..., Y1, Y0, proj_alpha, exp_E, 0).
msieve_expe_range() {
    awk '{v = $(NF-1); if (NR == 1 || v+0 < lo+0) lo = v; if (NR == 1 || v+0 > hi+0) hi = v}
         END {if (NR) print lo, hi}' "$1"
}

# Print "best worst" exp_E of a CADO-format selection file
cado_expe_range() {
    sed -n 's/.*, exp_E \([-0-9.]*\),.*/\1/p' "$1" | \
        awk '{if (NR == 1 || $1+0 < lo+0) lo = $1; if (NR == 1 || $1+0 > hi+0) hi = $1}
             END {if (NR) print lo, hi}'
}

# Track which values the user set explicitly (used by --report-only)
SET_TOP_N_EXTRACT=0
SET_TOP_M_MSIEVE=0
SET_TOP_M_CADO=0
SET_RESOPT_EFFORT=0
SET_ROPT_EFFORT=0
SET_THREADS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -n|--extract)
            TOP_N_EXTRACT="$2"
            SET_TOP_N_EXTRACT=1
            shift 2
            ;;
        --msieve-ropt)
            TOP_M_MSIEVE="$2"
            SET_TOP_M_MSIEVE=1
            shift 2
            ;;
        --cado-ropt)
            TOP_M_CADO="$2"
            SET_TOP_M_CADO=1
            shift 2
            ;;
        --resopt-effort)
            RESOPT_EFFORT="$2"
            SET_RESOPT_EFFORT=1
            shift 2
            ;;
        --ropt-effort)
            ROPT_EFFORT="$2"
            SET_ROPT_EFFORT=1
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            SET_THREADS=1
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
        --report-only)
            REPORT_ONLY=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check dependencies (report-only mode reuses results already on disk, so it
# needs none of the binaries or the large sopt inputs)
if [ "$REPORT_ONLY" -eq 0 ]; then
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
fi

if [ "$REPORT_ONLY" -eq 0 ]; then
    mkdir -p "$WORK_DIR" "$FINAL_DIR"
fi

echo "======================================"
if [ "$REPORT_ONLY" -eq 1 ]; then
    echo "FULL OPTIMIZATION PIPELINE (REPORT ONLY)"
else
    echo "FULL OPTIMIZATION PIPELINE"
fi
echo "======================================"

# Report-only mode: recover the settings of the run that produced the results
# on disk. Anything passed explicitly on the command line wins.
if [ "$REPORT_ONLY" -eq 1 ]; then
    PREV_REPORT="$FINAL_DIR/pipeline_report.txt"

    have_results=0
    for f in msieve_ropt_orig.p msieve_ropt_inv.p cado_ropt_orig.txt cado_ropt_inv.txt; do
        if [ -f "$FINAL_DIR/$f" ]; then have_results=1; fi
    done
    if [ "$have_results" -eq 0 ]; then
        echo "Error: no root-optimization results in $FINAL_DIR/ (msieve_ropt_*.p, cado_ropt_*.txt)." >&2
        echo "Run this from the directory the pipeline ran in." >&2
        exit 1
    fi

    # REC_<key> = the value recorded for the run, "" if unknown
    for key in $SETTINGS_KEYS; do printf -v "REC_$key" '%s' ""; done
    if [ -f "$SETTINGS_FILE" ]; then
        for key in $SETTINGS_KEYS; do printf -v "REC_$key" '%s' "$(read_setting "$key")"; done
        echo "Settings read from $SETTINGS_FILE"
        case "$(read_setting STATUS)" in
            complete) ;;
            incomplete)
                echo "WARNING: the last pipeline run finished, but these root-optimization passes"
                echo "         produced no results: $(read_setting FAILED)" ;;
            *)
                echo "WARNING: the last pipeline run did not finish (or is still running);"
                echo "         its results in $FINAL_DIR/ may be incomplete." ;;
        esac
    elif [ -f "$PREV_REPORT" ]; then
        # Results from before the settings file existed: scrape the old report
        scrape_report() {
            sed -n "s/$1/\\1/p" "$PREV_REPORT" | head -n1
        }
        REC_TOP_N_EXTRACT=$(scrape_report '^  Extracted top \([0-9][0-9]*\) from initial sopt$')
        REC_RESOPT_EFFORT=$(scrape_report '^  Re-ran sopt with effort \([0-9][0-9]*\)$')
        REC_TOP_M_MSIEVE=$(scrape_report '^  Selected best \([0-9][0-9]*\) for msieve ropt$')
        REC_TOP_M_CADO=$(scrape_report '^  Selected best \([0-9][0-9]*\) for CADO ropt$')
        REC_ROPT_EFFORT=$(scrape_report '^  CADO ropt effort: \([0-9][0-9]*\)$')
        REC_THREADS=$(scrape_report '^  Parallel threads: \([0-9][0-9]*\)$')
        echo "No $SETTINGS_FILE (results predate it); settings recovered from $PREV_REPORT"
        # Save them once, so later reports start from these values rather than
        # from a report that explicit overrides may have changed
        write_settings complete REC_
        echo "Recovered settings saved to $SETTINGS_FILE"
    else
        echo "WARNING: no $SETTINGS_FILE or previous report; unrecorded settings are shown as unknown"
    fi

    for key in $SETTINGS_KEYS; do
        set_flag="SET_$key"
        rec="REC_$key"
        if [ "${!set_flag}" -eq 0 ]; then
            printf -v "$key" '%s' "${!rec:-unknown}"
        fi
    done

    # If the selection file for a count is missing, use the newest one present
    for kind in msieve cado; do
        if [ "$kind" = msieve ]; then key=TOP_M_MSIEVE; ext=ms; else key=TOP_M_CADO; ext=txt; fi
        set_flag="SET_$key"
        if [ "${!set_flag}" -eq 0 ] && [ ! -f "$FINAL_DIR/best${!key}_${kind}.$ext" ]; then
            v=$(ls -t "$FINAL_DIR"/best*_"$kind".$ext 2>/dev/null | \
                sed -n "s#.*/best\([0-9][0-9]*\)_$kind\.$ext\$#\1#p" | head -n1 || true)
            if [ -n "$v" ]; then
                echo "Using newest selection file $FINAL_DIR/best${v}_${kind}.$ext (recorded count: ${!key})"
                printf -v "$key" '%s' "$v"
            fi
        fi
    done

    echo "Reusing results in $FINAL_DIR/ (phases 1-6 skipped)"
    echo ""
fi

echo "Extract top $TOP_N_EXTRACT from initial sopt"
echo "Re-run sopt with effort $RESOPT_EFFORT"
echo "Run msieve ropt on best $TOP_M_MSIEVE after re-sopt"
echo "Run CADO ropt on best $TOP_M_CADO after re-sopt"
echo "CADO ropt effort: $ROPT_EFFORT"
echo "Parallel threads: $THREADS"
echo ""

if [ "$REPORT_ONLY" -eq 0 ]; then

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
# Lines are c_d .. c0, Y1, Y0, proj_alpha, exp_E, 0: exp_E is column NF-1, degree NF-6
NUM_COLS=$(head -n 1 "$WORK_DIR/resopt_msieve.ms" | wc -w)
EXPE_COL=$((NUM_COLS - 1))
POLY_DEGREE=$((NUM_COLS - 6))
sort -k${EXPE_COL},${EXPE_COL}n "$WORK_DIR/resopt_msieve.ms" > "$WORK_DIR/resopt_msieve_sorted.ms"
echo "  Sorted msieve format: $WORK_DIR/resopt_msieve_sorted.ms"
echo ""

# PHASE 3: Extract polynomials for root optimization
echo "=== PHASE 3: EXTRACT POLYNOMIALS FOR ROOT OPTIMIZATION ==="

# From here on $FINAL_DIR/ holds this run's results; record its settings
archive_previous_results
write_settings running

# Extract for msieve (smaller count typically)
echo "Extracting top $TOP_M_MSIEVE for msieve ropt (msieve format)..."
head -n "$TOP_M_MSIEVE" "$WORK_DIR/resopt_msieve_sorted.ms" > "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms"
echo "  Extracted: $FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms"

# Get exp_E range for msieve
read -r MSIEVE_BEST_EXPE MSIEVE_WORST_EXPE < <(msieve_expe_range "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms") || true
echo "  exp_E range: $MSIEVE_BEST_EXPE (best) to $MSIEVE_WORST_EXPE (worst)"

# Extract for CADO (larger count typically)
echo "Extracting top $TOP_M_CADO for CADO ropt (CADO format)..."
if python3 utils/extract_top_cado_poly.py "$WORK_DIR/resopt_sorted.txt" "$FINAL_DIR/best${TOP_M_CADO}_cado.txt" "$TOP_M_CADO"; then
    echo "  Extracted: $FINAL_DIR/best${TOP_M_CADO}_cado.txt"
else
    echo "Error: Failed to extract CADO polynomials"
    exit 1
fi

# Get exp_E range for CADO
read -r CADO_BEST_EXPE CADO_WORST_EXPE < <(cado_expe_range "$FINAL_DIR/best${TOP_M_CADO}_cado.txt") || true
echo "  exp_E range: $CADO_BEST_EXPE (best) to $CADO_WORST_EXPE (worst)"
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

echo "Polynomial degree: $POLY_DEGREE"

echo "Running msieve -npr on original (annotated with exp_E, $THREADS threads)..."
echo "  Processing $TOP_M_MSIEVE polynomials..."
START_TIME=$(date +%s)
if ./scripts/run_msieve_ropt_annotated.sh "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" \
   "$FINAL_DIR/msieve_ropt_orig.p" "$POLY_DEGREE" "$THREADS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  Completed in ${DURATION}s"
fi
check_ropt_output msieve_orig "$FINAL_DIR/msieve_ropt_orig.p" '^# norm'

echo "Running msieve -npr on inverted (annotated with exp_E, $THREADS threads)..."
echo "  Processing $TOP_M_MSIEVE polynomials..."
START_TIME=$(date +%s)
if ./scripts/run_msieve_ropt_annotated.sh "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve_inv.ms" \
   "$FINAL_DIR/msieve_ropt_inv.p" "$POLY_DEGREE" "$THREADS"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  Completed in ${DURATION}s"
fi
check_ropt_output msieve_inv "$FINAL_DIR/msieve_ropt_inv.p" '^# norm'
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
threads = max(1, min($threads, num_polys))

for chunk_idx in range(threads):
    start_idx = chunk_idx * num_polys // threads
    end_idx = (chunk_idx + 1) * num_polys // threads

    if start_idx == end_idx:
        continue

    chunk_file = '$chunk_dir/chunk_%02d.txt' % chunk_idx
    with open(chunk_file, 'w') as out:
        for poly in polynomials[start_idx:end_idx]:
            out.write(poly + '\n\n')
"

    # Count chunks created
    local num_chunks=$(ls -1 "$chunk_dir"/chunk_*.txt 2>/dev/null | wc -l)

    if [ "$num_chunks" -eq 0 ]; then
        echo "  Warning: No polynomials found to process"
        : > "$output_file"
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
    : > "$output_file"
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
check_ropt_output cado_orig "$FINAL_DIR/cado_ropt_orig.txt" 'MurphyE'

echo "Running CADO polyselect_ropt on inverted ($THREADS threads)..."
echo "  Processing $TOP_M_CADO polynomials..."
START_TIME=$(date +%s)
run_cado_parallel "$FINAL_DIR/best${TOP_M_CADO}_cado_inv.txt" "$FINAL_DIR/cado_ropt_inv.txt" "$THREADS"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "  Completed in ${DURATION}s"
check_ropt_output cado_inv "$FINAL_DIR/cado_ropt_inv.txt" 'MurphyE'
echo ""

if [ -z "$ROPT_FAILED" ]; then
    write_settings complete
else
    write_settings incomplete
    echo "WARNING: root-optimization passes with no results: $ROPT_FAILED"
    echo ""
fi

else

# Report-only: the exp_E ranges come from the selection files themselves, so
# they always describe the polynomials that were root-optimized.
MSIEVE_BEST_EXPE=""
MSIEVE_WORST_EXPE=""
if [ -s "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms" ]; then
    read -r MSIEVE_BEST_EXPE MSIEVE_WORST_EXPE < <(msieve_expe_range "$FINAL_DIR/best${TOP_M_MSIEVE}_msieve.ms") || true
fi
CADO_BEST_EXPE=""
CADO_WORST_EXPE=""
if [ -s "$FINAL_DIR/best${TOP_M_CADO}_cado.txt" ]; then
    read -r CADO_BEST_EXPE CADO_WORST_EXPE < <(cado_expe_range "$FINAL_DIR/best${TOP_M_CADO}_cado.txt") || true
fi
MSIEVE_BEST_EXPE="${MSIEVE_BEST_EXPE:-N/A}"
MSIEVE_WORST_EXPE="${MSIEVE_WORST_EXPE:-N/A}"
CADO_BEST_EXPE="${CADO_BEST_EXPE:-N/A}"
CADO_WORST_EXPE="${CADO_WORST_EXPE:-N/A}"

fi

# PHASE 7: Generate comparison report
echo "=== PHASE 7: GENERATE COMPARISON REPORT ==="

# Find the composite N so the msieve polynomials in the report are complete
# (msieve ropt output omits the "n:" line; CADO output already has it).
POLY_N=""
for n_src in "$FINAL_DIR/best${TOP_M_CADO}_cado.txt" \
             "$FINAL_DIR/cado_ropt_orig.txt" \
             "$WORK_DIR/top${TOP_N_EXTRACT}_input.ms" \
             "$WORK_DIR/resopt_output.txt"; do
    if [ -f "$n_src" ]; then
        POLY_N=$(grep -m1 '^n: [0-9][0-9]*$' "$n_src" 2>/dev/null | awk '{print $2}' || true)
        [ -n "$POLY_N" ] && break
    fi
done
if [ -z "$POLY_N" ] && [ -f "worktodo.ini" ]; then
    POLY_N=$(grep -m1 -o '[0-9]\{20,\}' worktodo.ini 2>/dev/null || true)
fi
if [ -z "$POLY_N" ]; then
    echo "  Warning: could not determine N; msieve polynomials in the report will omit the n: line"
fi

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

    # Print the top-10 list ($1), or a placeholder if it is empty. The lists are
    # captured first: under pipefail, "... | head || echo" also fires when head
    # exits early and sort gets SIGPIPE, which happens on any long result file.
    print_top_or_none() {
        if [ -n "$1" ]; then printf '%s\n' "$1"; else echo "  (No results)"; fi
    }

    report_msieve_section() {
        local file=$1 count top best
        if [ ! -f "$file" ]; then
            echo "  (No output file)"
            return
        fi
        count=$(grep -c "^# norm" "$file" || true)
        echo "  Found $count root-optimized polynomial(s)"
        echo ""
        echo "  Top 10 results (sorted by Murphy E, column 7):"
        top=$(grep '^#' "$file" | LANG=C sort -rgk7 | uniq | head -n10 || true)
        print_top_or_none "$top"
        echo ""
        best=$(grep '^#' "$file" | awk '{print $7}' | sort -g | tail -n 1 || true)
        echo "  Best Murphy E: ${best:-N/A}"
    }

    report_cado_section() {
        local file=$1 count top best
        if [ ! -f "$file" ]; then
            echo "  (No output file)"
            return
        fi
        count=$(grep -ci "### root-optimized polynomial" "$file" || true)
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
            echo "  Found $count root-optimized polynomial(s)"
        fi
        echo ""
        echo "  Top 10 results (sorted by Murphy E, highest first):"
        top=$(grep '^# side 1 MurphyE' "$file" | \
            awk -F= '{val=$NF; $0=$0; print val " ||| " $0}' | \
            sort -k1 -gr | \
            head -n10 | \
            cut -d'|' -f4- | \
            sed 's/^/ /' || true)
        print_top_or_none "$top"
        echo ""
        best=$(grep '^# side 1 MurphyE' "$file" | awk -F= '{print $NF}' | sort -g | tail -n 1 || true)
        echo "  Best Murphy E: ${best:-N/A}"
    }

    echo "1. msieve -npr (original):"
    report_msieve_section "$FINAL_DIR/msieve_ropt_orig.p"
    echo ""

    echo "2. msieve -npr (inverted):"
    report_msieve_section "$FINAL_DIR/msieve_ropt_inv.p"
    echo ""

    echo "3. CADO polyselect_ropt ropteffort=$ROPT_EFFORT (original):"
    report_cado_section "$FINAL_DIR/cado_ropt_orig.txt"
    echo ""

    echo "4. CADO polyselect_ropt ropteffort=$ROPT_EFFORT (inverted):"
    report_cado_section "$FINAL_DIR/cado_ropt_inv.txt"
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
        if [ -n "$POLY_N" ]; then
            echo "n: $POLY_N"
        else
            echo ""
        fi

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

} > "$FINAL_DIR/pipeline_report.txt.new"

# Report-only keeps the report it replaces as .prev, but only when the content
# changed, so re-running report never discards the last differing version.
REPORT_ROTATED=0
if [ "$REPORT_ONLY" -eq 1 ] && [ -f "$FINAL_DIR/pipeline_report.txt" ]; then
    if cmp -s "$FINAL_DIR/pipeline_report.txt" "$FINAL_DIR/pipeline_report.txt.new"; then
        rm -f "$FINAL_DIR/pipeline_report.txt.new"
    else
        mv "$FINAL_DIR/pipeline_report.txt" "$FINAL_DIR/pipeline_report.prev.txt"
        REPORT_ROTATED=1
    fi
fi
if [ -f "$FINAL_DIR/pipeline_report.txt.new" ]; then
    mv "$FINAL_DIR/pipeline_report.txt.new" "$FINAL_DIR/pipeline_report.txt"
fi

cat "$FINAL_DIR/pipeline_report.txt"

echo ""
echo "======================================"
if [ "$REPORT_ONLY" -eq 1 ]; then
    echo "REPORT REGENERATED"
else
    echo "PIPELINE COMPLETE!"
fi
echo "======================================"
echo "Full report: $FINAL_DIR/pipeline_report.txt"
if [ "$REPORT_ROTATED" -eq 1 ]; then
    echo "Previous version: $FINAL_DIR/pipeline_report.prev.txt"
elif [ "$REPORT_ONLY" -eq 1 ]; then
    echo "(unchanged)"
fi
echo "All results in: $FINAL_DIR/"
echo ""

# =============================================================================
# PHASE 8: SKEWOPT OPTIMIZATION (if configured)
# =============================================================================

if [ "$REPORT_ONLY" -eq 1 ] && [ -f "$FINAL_DIR/skewopt_results.txt" ]; then
    # Report-only recomputes nothing: reprint the skewopt results already on disk
    echo "======================================"
    echo "PHASE 8: SKEWOPT RESULTS (from $FINAL_DIR/skewopt_results.txt)"
    echo "======================================"
    echo ""
    cat "$FINAL_DIR/skewopt_results.txt"
    echo ""
elif [ -n "$SKEWOPT" ] && [ -f "$SKEWOPT" ]; then
    echo "======================================"
    echo "PHASE 8: SKEWOPT OPTIMIZATION"
    echo "======================================"
    echo ""

    SKEWOPT_SCRIPT="${UTILS_DIR:-./utils}/run_skewopt_on_best.py"

    if [ -f "$SKEWOPT_SCRIPT" ]; then
        python3 "$SKEWOPT_SCRIPT" \
            --skewopt "$SKEWOPT" \
            "$FINAL_DIR/cado_ropt_orig.txt" \
            "$FINAL_DIR/cado_ropt_inv.txt" \
            | tee "$FINAL_DIR/skewopt_results.txt"
        echo ""
        echo "Skewopt results saved to: $FINAL_DIR/skewopt_results.txt"
    else
        echo "Warning: skewopt script not found at $SKEWOPT_SCRIPT"
    fi
    echo ""
elif [ -n "$SKEWOPT" ]; then
    echo "Warning: skewopt binary not found at $SKEWOPT"
    echo ""
fi
