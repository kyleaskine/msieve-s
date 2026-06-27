#!/bin/bash

# One-liner: rescore the latest CADO ropt outputs onto cownoise's MurphyE scale
# and rank them against the latest deep-msieve outputs (msieve 'e' already
# matches cownoise, so those are used as-is). Auto-discovers the standard files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/nfs_config.ini"

RESULTS_DIR="pipeline_results"
SKEWOPT=""
TOP=25
OUT=""
EXTRA=()

show_help() {
    cat <<EOF
Usage: skewopt_rank.sh [OPTIONS] [extra files...]

Rescore the latest CADO ropt outputs through skewopt (cownoise) and rank them
against the latest deep-msieve outputs on one common MurphyE scale.

Auto-discovered (if present):
  <results>/cado_ropt_orig.txt, cado_ropt_inv.txt          (pipeline CADO)
  newest <results>/cado_deep_ropt_effort*/cado_deep_ropt_{orig,inv}.txt
  newest <results>/msieve_deep_ropt_top*/msieve_deep_ropt_{orig,inv}.p

Options:
  --results-dir DIR   Pipeline results dir (default: pipeline_results)
  --skewopt FILE      skewopt binary (default: config paths.skewopt_binary)
  --top N             Rows to print (default: 25)
  --out FILE          Full ranked TSV (default: <results>/skewopt_leaderboard.tsv)
  -h, --help          Show this help
Any extra file arguments are appended to the ranking.
EOF
}

parse_config() {
    local section=$1 key=$2 default=${3:-}
    [ -f "$CONFIG_FILE" ] || { echo "$default"; return; }
    local value
    value=$(awk -F= -v section="[$section]" -v key="$key" '
        $0 == section { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section { gsub(/^[ \t]+|[ \t]+$/, "", $1)
            if ($1 == key) { gsub(/#.*$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit } }
    ' "$CONFIG_FILE")
    value=$(eval echo "$value")
    echo "${value:-$default}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --skewopt)     SKEWOPT="$2"; shift 2;;
        --top)         TOP="$2"; shift 2;;
        --out)         OUT="$2"; shift 2;;
        -h|--help)     show_help; exit 0;;
        *)             EXTRA+=("$1"); shift;;
    esac
done

cd "$ROOT_DIR"
[ -d "$RESULTS_DIR" ] || { echo "Error: results dir not found: $RESULTS_DIR" >&2; exit 1; }
[ -z "$SKEWOPT" ] && SKEWOPT=$(parse_config "paths" "skewopt_binary" "$HOME/code/SkewOptimizer/skewopt")
SKEWOPT=$(eval echo "$SKEWOPT")
[ -z "$OUT" ] && OUT="$RESULTS_DIR/skewopt_leaderboard.tsv"

newest_dir() {
    find "$RESULTS_DIR" -maxdepth 1 -type d -name "$1" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

FILES=()
add() { [ -f "$1" ] && FILES+=("$1"); }

add "$RESULTS_DIR/cado_ropt_orig.txt"
add "$RESULTS_DIR/cado_ropt_inv.txt"

CADO_DEEP=$(newest_dir 'cado_deep_ropt_effort*')
[ -n "$CADO_DEEP" ] && { add "$CADO_DEEP/cado_deep_ropt_orig.txt"; add "$CADO_DEEP/cado_deep_ropt_inv.txt"; }

MSIEVE_DEEP=$(newest_dir 'msieve_deep_ropt_top*')
[ -n "$MSIEVE_DEEP" ] && { add "$MSIEVE_DEEP/msieve_deep_ropt_orig.p"; add "$MSIEVE_DEEP/msieve_deep_ropt_inv.p"; }

if [ "${#EXTRA[@]}" -gt 0 ]; then
    FILES+=("${EXTRA[@]}")
fi

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "Error: no ropt output files found under $RESULTS_DIR" >&2
    exit 1
fi

echo "Ranking files:"
printf '  %s\n' "${FILES[@]}"
echo ""

python3 utils/skewopt_rank.py --skewopt "$SKEWOPT" --top "$TOP" --out "$OUT" "${FILES[@]}"
