#!/bin/bash

# NFS Polynomial Optimization - Main Orchestration Script
# This script provides a unified interface to all NFS polyselect optimization workflows

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/nfs_config.ini"

# ============================================================================
# CONFIG PARSER
# ============================================================================

parse_config() {
    local section=$1
    local key=$2
    local default=${3:-}

    # Read value from INI file
    local value=$(awk -F= -v section="[$section]" -v key="$key" '
        $0 == section { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section {
            # Trim key name (before =)
            gsub(/^[ \t]+|[ \t]+$/, "", $1);
            if ($1 == key) {
                # Trim value (after =)
                gsub(/^[ \t]+|[ \t]+$/, "", $2);
                # Remove comments
                gsub(/#.*$/, "", $2);
                # Trim again after removing comments
                gsub(/^[ \t]+|[ \t]+$/, "", $2);
                print $2;
                exit
            }
        }
    ' "$CONFIG_FILE")

    # Expand environment variables
    value=$(eval echo "$value")

    # Return value or default
    echo "${value:-$default}"
}

load_config() {
    # Paths
    CADO_BUILD_DIR=$(parse_config "paths" "cado_build_dir" "$HOME/cado-nfs/build/localhost")
    MSIEVE=$(parse_config "paths" "msieve_binary" "./msieve")
    UTILS_DIR=$(parse_config "paths" "utils_dir" "./utils")
    SKEWOPT=$(parse_config "paths" "skewopt_binary" "")

    # System
    THREADS=$(parse_config "system" "threads" "4")
    if [ "$THREADS" = "0" ]; then
        THREADS=$(nproc)
    fi

    # Preprocessing
    INPUT_FILE=$(parse_config "preprocessing" "input_file" "msieve.dat.ms")
    SOPT_THREADS=$(parse_config "preprocessing" "sopt_threads" "4")
    SOPT_EFFORT=$(parse_config "preprocessing" "sopt_effort" "0")

    # Batch processing
    BATCH_ENABLED=$(parse_config "batch_processing" "enabled" "true")
    BATCH_SIZE=$(parse_config "batch_processing" "batch_size" "100")
    SLEEP_INTERVAL=$(parse_config "batch_processing" "sleep_interval" "0")
    POLY_DEGREE=$(parse_config "batch_processing" "poly_degree" "0")

    # Pipeline
    EXTRACT_TOP_N=$(parse_config "pipeline" "extract_top_n" "100")
    RESOPT_EFFORT=$(parse_config "pipeline" "resopt_effort" "10")
    MSIEVE_ROPT_COUNT=$(parse_config "pipeline" "msieve_ropt_count" "10")
    CADO_ROPT_COUNT=$(parse_config "pipeline" "cado_ropt_count" "100")
    ROPT_EFFORT=$(parse_config "pipeline" "ropt_effort" "10")

    # Output
    WORK_DIR=$(parse_config "output" "work_dir" "pipeline_work")
    RESULTS_DIR=$(parse_config "output" "results_dir" "pipeline_results")
    FINAL_OUTPUT=$(parse_config "output" "final_output" "outMsieve.p")

    # Files
    CADO_PROCESSED_LINES=$(parse_config "files" "cado_processed_lines" ".cado_processed_lines")
    CADO_RESULTS=$(parse_config "files" "cado_results" "cado_results.ms")
    CADO_SOPT_OUTPUT=$(parse_config "files" "cado_sopt_output" "cado_sopt_output.txt")
    CADO_SOPT_UNSORTED=$(parse_config "files" "cado_sopt_unsorted" "cado_sopt_unsorted.txt")
    MSIEVE_SOPT_SORTED=$(parse_config "files" "msieve_sopt_sorted" "cado_results_sorted.ms")

    # CADO binaries
    CADO_SOPT="$CADO_BUILD_DIR/polyselect/sopt"
    CADO_ROPT="$CADO_BUILD_DIR/polyselect/polyselect_ropt"

    # Export for use by subscripts
    export CADO_SOPT CADO_ROPT MSIEVE UTILS_DIR THREADS SKEWOPT
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

check_dependencies() {
    local missing=0

    if [ ! -f "$CADO_SOPT" ]; then
        echo "Error: CADO sopt not found at $CADO_SOPT" >&2
        missing=1
    fi

    if [ ! -f "$MSIEVE" ]; then
        echo "Error: msieve not found at $MSIEVE" >&2
        missing=1
    fi

    if [ ! -d "$UTILS_DIR" ]; then
        echo "Error: Utils directory not found at $UTILS_DIR" >&2
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        echo "" >&2
        echo "Please check your configuration in $CONFIG_FILE" >&2
        exit 1
    fi
}

show_config() {
    echo "======================================"
    echo "CURRENT CONFIGURATION"
    echo "======================================"
    echo "Config file: $CONFIG_FILE"
    echo ""
    echo "Paths:"
    echo "  CADO build:  $CADO_BUILD_DIR"
    echo "  CADO sopt:   $CADO_SOPT"
    echo "  CADO ropt:   $CADO_ROPT"
    echo "  msieve:      $MSIEVE"
    echo "  Utils:       $UTILS_DIR"
    echo "  skewopt:     ${SKEWOPT:-<not configured>}"
    echo ""
    echo "System:"
    echo "  Threads:     $THREADS"
    echo ""
    echo "Preprocessing:"
    echo "  Input:       $INPUT_FILE"
    echo "  sopt effort: $SOPT_EFFORT"
    echo ""
    echo "Batch Processing:"
    echo "  Enabled:     $BATCH_ENABLED"
    echo "  Batch size:  $BATCH_SIZE"
    echo "  Sleep:       ${SLEEP_INTERVAL}s"
    echo ""
    echo "Pipeline:"
    echo "  Extract top: $EXTRACT_TOP_N"
    echo "  Re-sopt effort: $RESOPT_EFFORT"
    echo "  msieve ropt: $MSIEVE_ROPT_COUNT polys"
    echo "  CADO ropt:   $CADO_ROPT_COUNT polys"
    if [ -n "$SIZE_PRESET" ]; then
        echo "  Size preset: $SIZE_PRESET"
    fi
    echo "======================================"
}

# ============================================================================
# WORKFLOW COMMANDS
# ============================================================================

cmd_preprocess() {
    echo "Running preprocessing (dedupe + CADO sopt)..."
    cd "$SCRIPT_DIR"
    ./scripts/dedupe_and_sopt.sh -t "$SOPT_THREADS" -i "$INPUT_FILE"
}

cmd_batch() {
    echo "Starting continuous batch processing..."
    cd "$SCRIPT_DIR"
    ./scripts/process_batches.sh -t "$THREADS" -b "$BATCH_SIZE" -s "$SLEEP_INTERVAL" "$@"
}

cmd_pipeline() {
    echo "Running full optimization pipeline..."
    cd "$SCRIPT_DIR"
    ./scripts/full_optimization_pipeline.sh \
        -n "$EXTRACT_TOP_N" \
        --resopt-effort "$RESOPT_EFFORT" \
        --msieve-ropt "$MSIEVE_ROPT_COUNT" \
        --cado-ropt "$CADO_ROPT_COUNT" \
        --ropt-effort "$ROPT_EFFORT" \
        -t "$THREADS"
}

cmd_cleanup() {
    echo "Running cleanup..."
    cd "$SCRIPT_DIR"
    ./scripts/cleanup.sh "$@"
}

cmd_watch() {
    echo "Watching results (Ctrl+C to exit)..."
    cd "$SCRIPT_DIR"
    watch -n 60 "grep '^#' outMsieve.p 2>/dev/null | LANG=C sort -rgk7 | uniq | head -n25 || echo 'No results yet'"
}

# ============================================================================
# MAIN MENU
# ============================================================================

show_help() {
    cat << EOF
NFS Polynomial Optimization - Unified Interface

Usage: $0 [--size SIZE] <command> [options]

Global Options:
  --size SIZE          Select ropt count preset based on composite size
                         Reads from [size_small], [size_medium], [size_big]
                         sections in nfs_config.ini

Commands:
  preprocess           Run preprocessing: deduplicate and CADO size optimization
  batch [-c N]         Start batch processing (Phase 1: sopt, Phase 2: ropt)
                         -c N  Stop after N cycles (default: unlimited)
  pipeline             Run full optimization pipeline (extract top N, re-sopt, ropt)

  cleanup [--deep]     Clean up intermediate files (--deep removes all outputs)
  watch                Monitor results in real-time

  config               Show current configuration
  help                 Show this help message

Configuration:
  Edit $CONFIG_FILE to customize settings for your system

Examples:
  # Standard workflow
  $0 preprocess              # First, preprocess your polynomials
  $0 batch                   # Then run continuous batch processing
  $0 batch -c 3              # Run 3 batch cycles then stop

  # Advanced workflow
  $0 --size small pipeline   # Pipeline with small ropt counts
  $0 --size big pipeline     # Pipeline with big ropt counts
  $0 pipeline                # Pipeline using config file values

  # Monitoring
  $0 watch                  # Watch results in real-time

  # Cleanup
  $0 cleanup                # Clean intermediate files (keeps results)
  $0 cleanup --deep         # Deep clean (removes everything)

For more information, see README.md

EOF
    exit 0
}

# ============================================================================
# SIZE PRESETS
# ============================================================================

# Read ropt counts from [size_small], [size_medium], or [size_big] config sections
apply_size_preset() {
    local size=$1
    local section="size_${size}"

    local extract_val=$(parse_config "$section" "extract_top_n" "")
    local msieve_val=$(parse_config "$section" "msieve_ropt_count" "")
    local cado_val=$(parse_config "$section" "cado_ropt_count" "")

    if [ -z "$msieve_val" ] || [ -z "$cado_val" ]; then
        echo "Error: Unknown or incomplete size preset: $size"
        echo "Expected [${section}] section in $CONFIG_FILE with msieve_ropt_count and cado_ropt_count"
        exit 1
    fi

    [ -n "$extract_val" ] && EXTRACT_TOP_N="$extract_val"
    MSIEVE_ROPT_COUNT="$msieve_val"
    CADO_ROPT_COUNT="$cado_val"
    echo "Size preset '$size': extract_top=$EXTRACT_TOP_N, msieve_ropt=$MSIEVE_ROPT_COUNT, cado_ropt=$CADO_ROPT_COUNT"
}

# ============================================================================
# ENTRY POINT
# ============================================================================

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo ""
    echo "Please create your local configuration file:"
    echo "  cp nfs_config.ini.template nfs_config.ini"
    echo ""
    echo "Then edit nfs_config.ini to match your system paths."
    exit 1
fi

load_config

# Parse global options
SIZE_PRESET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --size)
            if [ $# -lt 2 ]; then
                echo "Error: --size requires an argument (small, medium, big)"
                exit 1
            fi
            SIZE_PRESET="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Apply size preset if specified
if [ -n "$SIZE_PRESET" ]; then
    apply_size_preset "$SIZE_PRESET"
fi

# Parse command
if [ $# -eq 0 ]; then
    show_help
fi

COMMAND=$1
shift

case "$COMMAND" in
    preprocess)
        check_dependencies
        cmd_preprocess "$@"
        ;;
    batch)
        check_dependencies
        cmd_batch "$@"
        ;;
    pipeline)
        check_dependencies
        cmd_pipeline "$@"
        ;;
    cleanup)
        cmd_cleanup "$@"
        ;;
    watch)
        cmd_watch "$@"
        ;;
    config)
        show_config
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "Error: Unknown command: $COMMAND"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
