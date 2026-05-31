#!/bin/bash

# Deep CADO root-optimization pass for an existing pipeline run.
#
# This script is intended to run after ./nfs_optimize.sh pipeline. It uses the
# size-optimized CADO inputs as the source of truth, selects a short list by
# exp_E plus low-effort ropt wildcards, then reruns CADO polyselect_ropt at a
# higher effort on only those candidates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/nfs_config.ini"

RESULTS_DIR="pipeline_results"
WORK_DIR=""
SOURCE_ORIG=""
SOURCE_INV=""
BROAD_ORIG=""
BROAD_INV=""
EXP_TOP=32
MURPHY_TOP=8
DEEP_EFFORT=50
THREADS=8
INCLUDE_INVERTED=1
CADO_ROPT="${CADO_ROPT:-}"
SKEWOPT="${SKEWOPT:-}"
RUN_SKEWOPT="auto"
SELECT_ONLY=0
REPORT_ONLY=0

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
Usage: deep_cado_ropt.sh [OPTIONS]

Run a high-effort CADO ropt pass on a short candidate list from an existing
pipeline_results directory.

Options:
  --results-dir DIR       Existing pipeline results directory (default: pipeline_results)
  --work-dir DIR          Output directory (default: DIR/cado_deep_ropt_effort<E>)
  --source-orig FILE      Size-optimized original CADO input file
  --source-inv FILE       Size-optimized inverted CADO input file
  --broad-orig FILE       Low-effort CADO original ropt output
  --broad-inv FILE        Low-effort CADO inverted ropt output
  --exp-top N             Always include top N by CADO exp_E (default: 32)
  --murphy-top N          Also include top N by broad-pass MurphyE (default: 8)
  --effort N              Deep CADO ropteffort (default: 50)
  -t, --threads N         Parallel CADO processes (default: config/system threads or 8)
  --cado-ropt FILE        CADO polyselect_ropt binary
  --skewopt FILE          Optional skewopt binary for final best-polynomial check
  --select-only           Write selected candidate inputs and manifest, then stop
  --report-only           Rebuild report from existing deep ropt outputs, then stop
  --no-inverted           Only process original side
  --no-skewopt            Do not run skewopt even if configured
  -h, --help              Show this help

Typical use:
  ./nfs_optimize.sh --size small pipeline
  ./scripts/deep_cado_ropt.sh --exp-top 32 --murphy-top 8 --effort 50 -t 8

EOF
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
        --broad-orig)
            BROAD_ORIG="$2"
            shift 2
            ;;
        --broad-inv)
            BROAD_INV="$2"
            shift 2
            ;;
        --exp-top)
            EXP_TOP="$2"
            shift 2
            ;;
        --murphy-top)
            MURPHY_TOP="$2"
            shift 2
            ;;
        --effort)
            DEEP_EFFORT="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --cado-ropt)
            CADO_ROPT="$2"
            shift 2
            ;;
        --skewopt)
            SKEWOPT="$2"
            shift 2
            ;;
        --select-only)
            SELECT_ONLY=1
            shift
            ;;
        --report-only)
            REPORT_ONLY=1
            shift
            ;;
        --no-inverted)
            INCLUDE_INVERTED=0
            shift
            ;;
        --no-skewopt)
            RUN_SKEWOPT="false"
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

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: results directory not found: $RESULTS_DIR" >&2
    exit 1
fi

if [ "$THREADS" = "0" ]; then
    THREADS=$(nproc)
elif [ -z "$THREADS" ]; then
    THREADS=$(parse_config "system" "threads" "8")
fi

if [ -z "$CADO_ROPT" ]; then
    CADO_BUILD_DIR=$(parse_config "paths" "cado_build_dir" "$HOME/cado-nfs/build/localhost")
    CADO_ROPT="$CADO_BUILD_DIR/polyselect/polyselect_ropt"
fi

if [ -z "$SKEWOPT" ]; then
    SKEWOPT=$(parse_config "paths" "skewopt_binary" "")
fi

if [ -z "$WORK_DIR" ]; then
    WORK_DIR="$RESULTS_DIR/cado_deep_ropt_effort${DEEP_EFFORT}"
fi

newest_source_orig() {
    find "$RESULTS_DIR" -maxdepth 1 -name 'best*_cado.txt' ! -name '*_inv.txt' \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-
}

newest_source_inv() {
    find "$RESULTS_DIR" -maxdepth 1 -name 'best*_cado_inv.txt' \
        -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-
}

if [ -z "$SOURCE_ORIG" ]; then
    SOURCE_ORIG=$(newest_source_orig)
fi
if [ -z "$SOURCE_INV" ]; then
    SOURCE_INV=$(newest_source_inv)
fi
if [ -z "$BROAD_ORIG" ]; then
    BROAD_ORIG="$RESULTS_DIR/cado_ropt_orig.txt"
fi
if [ -z "$BROAD_INV" ]; then
    BROAD_INV="$RESULTS_DIR/cado_ropt_inv.txt"
fi

if [ "$SELECT_ONLY" -eq 0 ] && [ "$REPORT_ONLY" -eq 0 ] && [ ! -x "$CADO_ROPT" ]; then
    echo "Error: CADO polyselect_ropt not found or not executable: $CADO_ROPT" >&2
    exit 1
fi
if [ ! -f "$SOURCE_ORIG" ]; then
    echo "Error: original CADO source input not found: $SOURCE_ORIG" >&2
    exit 1
fi
if [ "$INCLUDE_INVERTED" -eq 1 ] && [ ! -f "$SOURCE_INV" ]; then
    echo "Error: inverted CADO source input not found: $SOURCE_INV" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"

echo "======================================"
echo "DEEP CADO ROPT"
echo "======================================"
echo "Source original:  $SOURCE_ORIG"
if [ "$INCLUDE_INVERTED" -eq 1 ]; then
    echo "Source inverted:  $SOURCE_INV"
fi
echo "Broad original:   $BROAD_ORIG"
if [ "$INCLUDE_INVERTED" -eq 1 ]; then
    echo "Broad inverted:   $BROAD_INV"
fi
echo "exp_E top:        $EXP_TOP"
echo "MurphyE wildcards:$MURPHY_TOP"
echo "Deep ropteffort:  $DEEP_EFFORT"
echo "Parallel workers: $THREADS"
echo "Output dir:       $WORK_DIR"
if [ "$SELECT_ONLY" -eq 1 ]; then
    echo "Mode:             selection only"
elif [ "$REPORT_ONLY" -eq 1 ]; then
    echo "Mode:             report only"
fi
echo ""

export WORK_DIR SOURCE_ORIG SOURCE_INV BROAD_ORIG BROAD_INV EXP_TOP MURPHY_TOP INCLUDE_INVERTED

if [ "$REPORT_ONLY" -eq 0 ]; then
python3 <<'PY'
import math
import os
import re
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
exp_top = int(os.environ["EXP_TOP"])
murphy_top = int(os.environ["MURPHY_TOP"])
include_inverted = os.environ["INCLUDE_INVERTED"] == "1"


def parse_source(path):
    path = Path(path)
    lines = path.read_text().splitlines()
    blocks = []
    current = []

    def finish(block):
        if not block:
            return
        exp_e = None
        coeffs = []
        for line in block:
            m = re.search(r"\bexp_E\s+([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)", line)
            if m:
                exp_e = float(m.group(1))
            if re.match(r"^(n|Y[01]|c\d+):", line):
                coeffs.append(line.strip())
        blocks.append({
            "index": len(blocks),
            "lines": block[:],
            "exp_e": exp_e if exp_e is not None else math.inf,
            "signature": tuple(coeffs),
        })

    for line in lines:
        if line.startswith("### Size-optimized polynomial"):
            finish(current)
            current = [line]
        elif line == "":
            finish(current)
            current = []
        elif current:
            current.append(line)
    finish(current)
    return blocks


def parse_broad(path, source_len, source_name):
    path = Path(path)
    if not path.exists():
        return {}

    events = []
    current_chunk = None
    current_local = None
    current_run = True
    saw_chunk = False

    for line in path.read_text().splitlines():
        if line.startswith("# ") and " -inputpolys " in line:
            current_run = source_name in line if source_name else True
            m = re.search(r"-inputpolys\s+(\S+)", line)
            current_chunk = None
            current_local = None
            if m:
                cm = re.search(r"chunk_(\d+)", m.group(1))
                if cm:
                    current_chunk = int(cm.group(1))
                    saw_chunk = True
        elif not current_run:
            continue
        elif line.startswith("### input polynomial"):
            parts = line.split()
            current_local = int(parts[3])
        elif line.startswith("# side 1 MurphyE"):
            score = float(line.rsplit("=", 1)[1])
            if current_local is not None:
                events.append((current_chunk, current_local, score))

    if not events:
        return {}

    if saw_chunk:
        chunks = sorted({chunk for chunk, _local, _score in events if chunk is not None})
        num_chunks = max(chunks) + 1
        per_chunk = (source_len + num_chunks - 1) // num_chunks
    else:
        per_chunk = None

    by_index = {}
    for chunk, local, score in events:
        if chunk is None:
            index = local
        else:
            index = chunk * per_chunk + local
        if 0 <= index < source_len:
            by_index[index] = max(score, by_index.get(index, 0.0))
    return by_index


def select_side(side, source_path, broad_path):
    source = parse_source(source_path)
    broad = parse_broad(broad_path, len(source), Path(source_path).name) if broad_path else {}

    selected = {}
    for block in sorted(source, key=lambda item: (item["exp_e"], item["index"]))[:min(exp_top, len(source))]:
        selected.setdefault(block["index"], {"reasons": [], "murphy": None})
        selected[block["index"]]["reasons"].append("exp_E")

    for index, score in sorted(broad.items(), key=lambda item: item[1], reverse=True)[:murphy_top]:
        selected.setdefault(index, {"reasons": [], "murphy": None})
        selected[index]["reasons"].append("broad_MurphyE")
        selected[index]["murphy"] = score

    seen = set()
    ordered = []
    for index in sorted(selected):
        block = source[index]
        if block["signature"] in seen:
            continue
        seen.add(block["signature"])
        ordered.append((block, selected[index]))

    output = work_dir / f"deep_cado_{side}_input.txt"
    with output.open("w") as f:
        for block, meta in ordered:
            for line in block["lines"]:
                f.write(line + "\n")
            f.write("\n")

    return output, ordered


manifest = work_dir / "selection_manifest.tsv"
with manifest.open("w") as mf:
    mf.write("side\tindex\texp_E\tbroad_MurphyE\treason\n")

    sides = [("orig", os.environ["SOURCE_ORIG"], os.environ["BROAD_ORIG"])]
    if include_inverted:
        sides.append(("inv", os.environ["SOURCE_INV"], os.environ["BROAD_INV"]))

    for side, source_path, broad_path in sides:
        output, selected = select_side(side, source_path, broad_path)
        for block, meta in selected:
            murphy = meta["murphy"]
            mf.write(
                f"{side}\t{block['index']}\t{block['exp_e']:.2f}\t"
                f"{'' if murphy is None else f'{murphy:.3e}'}\t"
                f"{'+'.join(meta['reasons'])}\n"
            )
        print(f"Selected {len(selected)} {side} candidates -> {output}")

print(f"Selection manifest -> {manifest}")
PY

    if [ "$SELECT_ONLY" -eq 1 ]; then
        echo "Selection-only mode complete; skipping deep CADO ropt."
        exit 0
    fi
else
    echo "Report-only mode: skipping candidate selection and deep CADO ropt."
fi

split_cado_input() {
    local input_file=$1
    local chunk_dir=$2
    local chunks=$3

    python3 - "$input_file" "$chunk_dir" "$chunks" <<'PY'
import sys
from pathlib import Path

input_file = Path(sys.argv[1])
chunk_dir = Path(sys.argv[2])
chunks = int(sys.argv[3])

polys = []
current = []
for line in input_file.read_text().splitlines():
    if line.startswith("# deep-cado-select "):
        continue
    if line.startswith("### Size-optimized polynomial"):
        if current:
            polys.append("\n".join(current))
        current = [line]
    elif line == "":
        if current:
            polys.append("\n".join(current))
            current = []
    elif current:
        current.append(line)
if current:
    polys.append("\n".join(current))

if not polys:
    sys.exit(0)

chunks = max(1, min(chunks, len(polys)))
chunk_dir.mkdir(parents=True, exist_ok=True)

for chunk_idx in range(chunks):
    start = chunk_idx * len(polys) // chunks
    end = (chunk_idx + 1) * len(polys) // chunks
    if start == end:
        continue
    with (chunk_dir / f"chunk_{chunk_idx:02d}.txt").open("w") as f:
        for poly in polys[start:end]:
            f.write(poly)
            f.write("\n\n")
PY
}

run_cado_side() {
    local side=$1
    local input_file="$WORK_DIR/deep_cado_${side}_input.txt"
    local output_file="$WORK_DIR/cado_deep_ropt_${side}.txt"
    local chunk_dir="$WORK_DIR/${side}_chunks"

    local count
    count=$(grep -c "^### Size-optimized polynomial" "$input_file" || true)
    if [ "$count" -eq 0 ]; then
        echo "No $side candidates; skipping"
        : > "$output_file"
        return
    fi

    echo "Running CADO ropt effort $DEEP_EFFORT on $count $side candidates..."
    rm -rf "$chunk_dir"
    mkdir -p "$chunk_dir"
    split_cado_input "$input_file" "$chunk_dir" "$THREADS"

    local pids=()
    local chunk
    for chunk in "$chunk_dir"/chunk_*.txt; do
        [ -f "$chunk" ] || continue
        (
            "$CADO_ROPT" -ropteffort "$DEEP_EFFORT" -inputpolys "$chunk" \
                > "${chunk%.txt}_result.txt" 2> "${chunk%.txt}_stderr.txt"
        ) &
        pids+=($!)
    done

    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
        fi
    done

    if [ "$failed" -gt 0 ]; then
        echo "Error: $failed CADO ropt worker(s) failed for $side" >&2
        echo "Chunk logs preserved in $chunk_dir" >&2
        local log
        for log in "$chunk_dir"/chunk_*_stderr.txt; do
            [ -s "$log" ] || continue
            echo "--- $(basename "$log") ---" >&2
            sed -n '1,12p' "$log" >&2
        done
        exit 1
    fi

    : > "$output_file"
    for chunk in "$chunk_dir"/chunk_*_result.txt; do
        [ -f "$chunk" ] || continue
        cat "$chunk" >> "$output_file"
        printf '\n' >> "$output_file"
    done
    echo "  Wrote $output_file"
}

START_TIME=$(date +%s)
if [ "$REPORT_ONLY" -eq 0 ]; then
    run_cado_side "orig"
    if [ "$INCLUDE_INVERTED" -eq 1 ]; then
        run_cado_side "inv"
    fi
fi
END_TIME=$(date +%s)

export DEEP_EFFORT BROAD_ORIG BROAD_INV SOURCE_ORIG SOURCE_INV INCLUDE_INVERTED WORK_DIR
python3 <<'PY'
import os
import re
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
include_inverted = os.environ["INCLUDE_INVERTED"] == "1"
effort = os.environ["DEEP_EFFORT"]


def scores(path, source_name=None):
    path = Path(path)
    if not path.exists():
        return []
    vals = []
    current_run = True
    for line in path.read_text().splitlines():
        if line.startswith("# ") and " -inputpolys " in line:
            current_run = source_name in line if source_name else True
            continue
        if not current_run:
            continue
        if line.startswith("# side 1 MurphyE"):
            vals.append(float(line.rsplit("=", 1)[1]))
    return vals


def best(vals):
    return max(vals) if vals else None


def fmt(value):
    return "N/A" if value is None else f"{value:.3e}"


report = work_dir / "deep_cado_report.txt"
with report.open("w") as f:
    f.write("DEEP CADO ROPT REPORT\n")
    f.write("=====================\n")
    f.write(f"Deep ropteffort: {effort}\n")
    f.write(f"Selection manifest: {work_dir / 'selection_manifest.tsv'}\n\n")

    sides = [("original", os.environ["BROAD_ORIG"], os.environ["SOURCE_ORIG"], work_dir / "cado_deep_ropt_orig.txt")]
    if include_inverted:
        sides.append(("inverted", os.environ["BROAD_INV"], os.environ["SOURCE_INV"], work_dir / "cado_deep_ropt_inv.txt"))

    for label, broad_path, source_path, deep_path in sides:
        broad_scores = scores(broad_path, Path(source_path).name)
        deep_scores = scores(deep_path)
        f.write(f"{label}:\n")
        f.write(f"  broad count: {len(broad_scores)}\n")
        f.write(f"  broad best:  {fmt(best(broad_scores))}\n")
        f.write(f"  deep count:  {len(deep_scores)}\n")
        f.write(f"  deep best:   {fmt(best(deep_scores))}\n")
        f.write("  deep top 10:\n")
        for val in sorted(deep_scores, reverse=True)[:10]:
            f.write(f"    {val:.3e}\n")
        f.write("\n")

print(report.read_text())
PY

echo "Total deep CADO wall time: $((END_TIME - START_TIME))s"
echo "Report: $WORK_DIR/deep_cado_report.txt"

if [ "$RUN_SKEWOPT" != "false" ] && [ -n "$SKEWOPT" ] && [ -x "$SKEWOPT" ]; then
    if [ "$INCLUDE_INVERTED" -eq 1 ]; then
        echo ""
        echo "Running skewopt on deep CADO best results..."
        python3 utils/run_skewopt_on_best.py \
            --skewopt "$SKEWOPT" \
            "$WORK_DIR/cado_deep_ropt_orig.txt" \
            "$WORK_DIR/cado_deep_ropt_inv.txt" \
            | tee "$WORK_DIR/skewopt_deep_results.txt"
        echo "Skewopt report: $WORK_DIR/skewopt_deep_results.txt"
    else
        echo "Skipping skewopt: inverted result file is not available"
    fi
elif [ "$RUN_SKEWOPT" != "false" ] && [ -n "$SKEWOPT" ]; then
    echo "Skipping skewopt: binary not executable at $SKEWOPT"
fi
