#!/bin/bash

# Build one unified deep-ropt seed list from the round-1 ropt results and feed
# the SAME set into both deep_msieve_ropt.sh and deep_cado_ropt.sh.
#
# Selection (see utils/select_deep_seeds.py): union of the top --per-source
# seeds by post-ropt score from each of msieve_orig / msieve_inv / cado_orig /
# cado_inv, then fill up to --total with the best (lowest) exp_E seeds. Seeds
# are keyed by Y1, which both tools preserve through ropt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS_DIR="pipeline_results"
WORK_DIR="pipeline_work"
PER_SOURCE=8
TOTAL=32
THREADS=8
EFFORT=50
RUN=0
DO_MSIEVE=1
DO_CADO=1

show_help() {
    cat <<EOF
Usage: select_deep_seeds.sh [OPTIONS]

Select a unified deep-ropt seed list and (optionally) run both deep passes.

Options:
  --results-dir DIR   Pipeline results dir (default: pipeline_results)
  --work-dir DIR      Dir holding resopt_sorted.txt (default: pipeline_work)
  --per-source N      Top-N proven winners per source (default: 8)
  --total N           Total seeds in the deep list (default: 32)
  -t, --threads N     Parallel workers for the deep passes (default: 8)
  --effort N          CADO deep ropteffort (default: 50)
  --run               Run both deep passes now (default: just select + print commands)
  --no-msieve         Skip the deep msieve pass
  --no-cado           Skip the deep CADO pass
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --work-dir)    WORK_DIR="$2"; shift 2;;
        --per-source)  PER_SOURCE="$2"; shift 2;;
        --total)       TOTAL="$2"; shift 2;;
        -t|--threads)  THREADS="$2"; shift 2;;
        --effort)      EFFORT="$2"; shift 2;;
        --run)         RUN=1; shift;;
        --no-msieve)   DO_MSIEVE=0; shift;;
        --no-cado)     DO_CADO=0; shift;;
        -h|--help)     show_help; exit 0;;
        *) echo "Unknown option: $1" >&2; show_help >&2; exit 1;;
    esac
done

cd "$ROOT_DIR"

SEED_DIR="$RESULTS_DIR/deep_seeds"

echo "======================================"
echo "UNIFIED DEEP SEED SELECTION"
echo "======================================"
python3 utils/select_deep_seeds.py \
    --results-dir "$RESULTS_DIR" \
    --work-dir "$WORK_DIR" \
    --per-source "$PER_SOURCE" \
    --total "$TOTAL"

MSIEVE_CMD=( ./scripts/deep_msieve_ropt.sh
    --source-orig "$SEED_DIR/deep_seeds_msieve.ms"
    --source-inv  "$SEED_DIR/deep_seeds_msieve_inv.ms"
    --top "$TOTAL" -t "$THREADS" )

# exp-top >= TOTAL and murphy-top 0 make deep_cado pass the pre-selected set
# through unchanged (no second exp_E trim, no extra wildcards).
CADO_CMD=( ./scripts/deep_cado_ropt.sh
    --source-orig "$SEED_DIR/deep_seeds_cado.txt"
    --source-inv  "$SEED_DIR/deep_seeds_cado_inv.txt"
    --exp-top "$TOTAL" --murphy-top 0 --effort "$EFFORT" -t "$THREADS" )

echo ""
echo "Deep passes (same unified seed set):"
[ "$DO_MSIEVE" -eq 1 ] && printf '  %q ' "${MSIEVE_CMD[@]}" && echo
[ "$DO_CADO" -eq 1 ]   && printf '  %q ' "${CADO_CMD[@]}"   && echo

if [ "$RUN" -eq 1 ]; then
    echo ""
    if [ "$DO_MSIEVE" -eq 1 ]; then
        echo "--- running deep msieve ---"
        "${MSIEVE_CMD[@]}"
    fi
    if [ "$DO_CADO" -eq 1 ]; then
        echo "--- running deep CADO ---"
        "${CADO_CMD[@]}"
    fi
else
    echo ""
    echo "Re-run with --run to execute both deep passes now."
fi
