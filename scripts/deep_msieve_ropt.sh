#!/bin/bash

# Deep msieve root-optimization pass for an existing pipeline run.
#
# This is a convenience wrapper around run_msieve_ropt_annotated.sh. It keeps
# the low-level helper usable by the main pipeline, while giving the manual
# deep pass sensible defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/nfs_config.ini"

RESULTS_DIR="pipeline_results"
WORK_DIR=""
SOURCE_ORIG=""
SOURCE_INV=""
TOP_COUNT=24
STAGE2_STEPS=100
STAGE2_START=1.0327
STAGE2_MULT=1.0327
THREADS=""
POLY_DEGREE="auto"
INCLUDE_INVERTED=1
DRY_RUN=0
HELPER="$SCRIPT_DIR/run_msieve_ropt_annotated.sh"

parse_config() {
    local section=$1
    local key=$2
    local default=${3:-}

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$default"
        return
    fi

    local value
    value=$(awk -F= -v section="[$section]" -v key="$key" '
        $0 == section { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section {
            gsub(/^[ \t]+|[ \t]+$/, "", $1)
            if ($1 == key) {
                gsub(/#.*$/, "", $2)
                gsub(/^[ \t]+|[ \t]+$/, "", $2)
                print $2
                exit
            }
        }
    ' "$CONFIG_FILE")

    value=$(eval echo "$value")
    echo "${value:-$default}"
}

show_help() {
    cat <<EOF
Usage: deep_msieve_ropt.sh [OPTIONS]

Run a fine msieve stage2 root-optimization sweep on the top candidates from an
existing pipeline_results directory.

Options:
  --results-dir DIR       Existing pipeline results directory (default: pipeline_results)
  --work-dir DIR          Output directory (default: DIR/msieve_deep_ropt_top<N>)
  --source-orig FILE      Original msieve input file (default: newest best*_msieve.ms)
  --source-inv FILE       Inverted msieve input file (default: newest best*_msieve_inv.ms)
  --top N                 Process only the first N candidates (default: 24)
  --steps N               Stage2 sweep steps (default: 100)
  --start X               Stage2 starting multiplier (default: 1.0327)
  --mult X                Stage2 per-step multiplier (default: 1.0327)
  --poly-degree N         Polynomial degree 5 or 6 (default: auto)
  -t, --threads N         Parallel msieve processes (default: config/system threads or 8; 0=nproc)
  --helper FILE           run_msieve_ropt_annotated.sh path
  --no-inverted           Only process original signs
  --dry-run               Print commands without running msieve
  -h, --help              Show this help

Typical use:
  ./scripts/deep_msieve_ropt.sh
  ./scripts/deep_msieve_ropt.sh --top 16 -t 8

EOF
}

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_positive_number() {
    awk -v value="$1" 'BEGIN { exit !(value + 0 == value && value > 0) }'
}

newest_source_orig() {
    find "$RESULTS_DIR" -maxdepth 1 -name 'best*_msieve.ms' \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-
}

newest_source_inv() {
    find "$RESULTS_DIR" -maxdepth 1 -name 'best*_msieve_inv.ms' \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-
}

detect_poly_degree() {
    local source_file=$1
    local num_cols

    num_cols=$(awk 'NF { print NF; exit }' "$source_file")
    if [ -z "$num_cols" ]; then
        echo "Error: cannot detect polynomial degree from empty input: $source_file" >&2
        exit 1
    fi

    if [ "$num_cols" -eq 12 ]; then
        echo 6
    else
        echo 5
    fi
}

summarize_results() {
    local file=$1
    local label=$2

    if [ ! -f "$file" ]; then
        echo "$label: no output file"
        return
    fi

    local count
    local best
    count=$(grep -c "^# norm" "$file" || true)
    best=$(grep "^# norm" "$file" | awk '{ print $7 }' | sort -g | tail -n1 || true)
    echo "$label: $count result(s), best MurphyE ${best:-N/A}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --source-orig)
            SOURCE_ORIG="$2"
            shift 2
            ;;
        --source-inv)
            SOURCE_INV="$2"
            shift 2
            ;;
        --top)
            TOP_COUNT="$2"
            shift 2
            ;;
        --steps)
            STAGE2_STEPS="$2"
            shift 2
            ;;
        --start)
            STAGE2_START="$2"
            shift 2
            ;;
        --mult)
            STAGE2_MULT="$2"
            shift 2
            ;;
        --poly-degree)
            POLY_DEGREE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --helper)
            HELPER="$2"
            shift 2
            ;;
        --no-inverted)
            INCLUDE_INVERTED=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

cd "$ROOT_DIR"

if ! is_positive_int "$TOP_COUNT"; then
    echo "Error: --top must be a positive integer, got: $TOP_COUNT" >&2
    exit 1
fi

if ! is_positive_int "$STAGE2_STEPS"; then
    echo "Error: --steps must be a positive integer, got: $STAGE2_STEPS" >&2
    exit 1
fi

if ! is_positive_number "$STAGE2_START"; then
    echo "Error: --start must be a positive number, got: $STAGE2_START" >&2
    exit 1
fi

if ! is_positive_number "$STAGE2_MULT"; then
    echo "Error: --mult must be a positive number, got: $STAGE2_MULT" >&2
    exit 1
fi

if [ "$POLY_DEGREE" != "auto" ] && [ "$POLY_DEGREE" != "5" ] && [ "$POLY_DEGREE" != "6" ]; then
    echo "Error: --poly-degree must be 5, 6, or auto, got: $POLY_DEGREE" >&2
    exit 1
fi

if [ -z "$THREADS" ]; then
    THREADS=$(parse_config "system" "threads" "8")
elif [ "$THREADS" = "0" ]; then
    THREADS=$(nproc)
fi

if ! is_positive_int "$THREADS"; then
    echo "Error: --threads must be a positive integer or 0 for nproc, got: $THREADS" >&2
    exit 1
fi

if [ ! -f "$HELPER" ]; then
    echo "Error: helper script not found: $HELPER" >&2
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: results directory not found: $RESULTS_DIR" >&2
    exit 1
fi

if [ -z "$SOURCE_ORIG" ]; then
    SOURCE_ORIG=$(newest_source_orig)
fi
if [ -z "$SOURCE_INV" ]; then
    SOURCE_INV=$(newest_source_inv)
fi

if [ -z "$SOURCE_ORIG" ] || [ ! -f "$SOURCE_ORIG" ]; then
    echo "Error: original msieve source not found; expected $RESULTS_DIR/best*_msieve.ms" >&2
    exit 1
fi

if [ "$INCLUDE_INVERTED" -eq 1 ] && { [ -z "$SOURCE_INV" ] || [ ! -f "$SOURCE_INV" ]; }; then
    echo "Error: inverted msieve source not found; expected $RESULTS_DIR/best*_msieve_inv.ms" >&2
    echo "Use --no-inverted to run only the original signs." >&2
    exit 1
fi

if [ "$POLY_DEGREE" = "auto" ]; then
    POLY_DEGREE=$(detect_poly_degree "$SOURCE_ORIG")
fi

if [ -z "$WORK_DIR" ]; then
    WORK_DIR="$RESULTS_DIR/msieve_deep_ropt_top${TOP_COUNT}"
fi

NFS_ARGS="rootopt_stage2_steps=$STAGE2_STEPS rootopt_stage2_start=$STAGE2_START rootopt_stage2_mult=$STAGE2_MULT"
ORIG_OUTPUT="$WORK_DIR/msieve_deep_ropt_orig.p"
INV_OUTPUT="$WORK_DIR/msieve_deep_ropt_inv.p"

echo "======================================"
echo "DEEP MSIEVE ROPT"
echo "======================================"
echo "Source original: $SOURCE_ORIG"
if [ "$INCLUDE_INVERTED" -eq 1 ]; then
    echo "Source inverted: $SOURCE_INV"
else
    echo "Source inverted: disabled"
fi
echo "Top candidates:  $TOP_COUNT"
echo "Stage2 sweep:    $STAGE2_STEPS steps, start $STAGE2_START, multiplier $STAGE2_MULT"
echo "Polynomial deg:  $POLY_DEGREE"
echo "Parallel jobs:   $THREADS"
echo "Output dir:      $WORK_DIR"
echo ""

run_ropt() {
    local source_file=$1
    local output_file=$2
    local label=$3

    echo "Running deep msieve ropt on $label candidates..."
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %q %q %q %q %q %q %q\n' \
            "$HELPER" "$source_file" "$output_file" "$POLY_DEGREE" "$THREADS" "$NFS_ARGS" "$TOP_COUNT"
    else
        "$HELPER" "$source_file" "$output_file" "$POLY_DEGREE" "$THREADS" "$NFS_ARGS" "$TOP_COUNT"
    fi
}

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$WORK_DIR"
fi

run_ropt "$SOURCE_ORIG" "$ORIG_OUTPUT" "original"
if [ "$INCLUDE_INVERTED" -eq 1 ]; then
    run_ropt "$SOURCE_INV" "$INV_OUTPUT" "inverted"
fi

if [ "$DRY_RUN" -eq 0 ]; then
    {
        echo "======================================"
        echo "DEEP MSIEVE SUMMARY"
        echo "======================================"
        summarize_results "$ORIG_OUTPUT" "original"
        if [ "$INCLUDE_INVERTED" -eq 1 ]; then
            summarize_results "$INV_OUTPUT" "inverted"
        fi
    } | tee "$WORK_DIR/deep_msieve_report.txt"

    echo ""
    echo "Report written to: $WORK_DIR/deep_msieve_report.txt"
fi
