#!/bin/bash

# Cleanup script for polynomial optimization pipeline
# Backs up important results and cleans intermediate files

set -euo pipefail

# Configuration
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
KEEP_RESULTS=true
DEEP_CLEAN=false

show_help() {
    cat << EOF
Usage: cleanup.sh [OPTIONS]

Cleanup intermediate files from polynomial optimization pipeline

Options:
  -h, --help           Show this help message and exit
  -d, --deep           Deep clean: remove all results including pipeline_results/
  -b, --backup         Create backup before cleaning (default: yes)
  --no-backup          Skip backup creation

By default:
  - Backs up important results to backup_YYYYMMDD_HHMMSS/
  - Keeps final outputs (cado_sopt_output.txt, cado_results_sorted.ms, pipeline_results/)
  - Removes intermediate work files

Deep clean:
  - Backs up everything to backup directory
  - Removes all generated files including final outputs

EOF
    exit 0
}

# Parse arguments
CREATE_BACKUP=true
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -d|--deep)
            DEEP_CLEAN=true
            KEEP_RESULTS=false
            shift
            ;;
        -b|--backup)
            CREATE_BACKUP=true
            shift
            ;;
        --no-backup)
            CREATE_BACKUP=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "CLEANUP SCRIPT"
echo "======================================"
echo "Mode: $([ "$DEEP_CLEAN" = true ] && echo "DEEP CLEAN" || echo "STANDARD")"
echo "Backup: $([ "$CREATE_BACKUP" = true ] && echo "YES" || echo "NO")"
echo ""

# Create backup if requested
if [ "$CREATE_BACKUP" = true ]; then
    echo "=== CREATING BACKUP ==="
    mkdir -p "$BACKUP_DIR"

    # Backup important files
    BACKUP_COUNT=0

    # Main data files
    for file in msieve.dat.ms msieve.dat.deduped.ms outMsieve.p; do
        if [ -f "$file" ]; then
            echo "  Backing up $file..."
            cp "$file" "$BACKUP_DIR/"
            BACKUP_COUNT=$((BACKUP_COUNT + 1))
        fi
    done

    # CADO sopt outputs
    for file in cado_sopt_output.txt cado_sopt_unsorted.txt cado_results_sorted.ms; do
        if [ -f "$file" ]; then
            echo "  Backing up $file..."
            cp "$file" "$BACKUP_DIR/"
            BACKUP_COUNT=$((BACKUP_COUNT + 1))
        fi
    done

    # Pipeline results
    if [ -d "pipeline_results" ]; then
        echo "  Backing up pipeline_results/..."
        cp -r pipeline_results "$BACKUP_DIR/"
        BACKUP_COUNT=$((BACKUP_COUNT + 1))
    fi

    # Test comparison results
    if [ -f "ropt_comparison.txt" ]; then
        echo "  Backing up ropt_comparison.txt..."
        cp ropt_comparison.txt "$BACKUP_DIR/"
        BACKUP_COUNT=$((BACKUP_COUNT + 1))
    fi

    if [ $BACKUP_COUNT -gt 0 ]; then
        echo "  Backed up $BACKUP_COUNT item(s) to $BACKUP_DIR/"
    else
        echo "  No files to backup"
        rmdir "$BACKUP_DIR"
    fi
    echo ""
fi

# Clean up intermediate files
echo "=== CLEANING INTERMEDIATE FILES ==="
CLEAN_COUNT=0

# Work directories
for dir in pipeline_work sopt_output sopt_batch_chunks msieve_ropt_work_*; do
    if [ -d "$dir" ]; then
        echo "  Removing $dir/..."
        rm -rf "$dir"
        CLEAN_COUNT=$((CLEAN_COUNT + 1))
    fi
done

# Main data files (keep if we don't have backup)
if [ "$CREATE_BACKUP" = true ]; then
    for file in msieve.dat.ms msieve.dat.deduped.ms outMsieve.p; do
        if [ -f "$file" ]; then
            echo "  Removing $file..."
            rm -f "$file"
            CLEAN_COUNT=$((CLEAN_COUNT + 1))
        fi
    done
fi

# Test files and intermediate outputs
for pattern in "top*_*.ms" "top*_*.txt" "top*_*.p" "ropt_*.p" "ropt_*.txt" "best*_*.ms" "best*_*.txt"; do
    FILES=$(ls $pattern 2>/dev/null || true)
    if [ -n "$FILES" ]; then
        echo "  Removing $pattern..."
        rm -f $pattern
        CLEAN_COUNT=$((CLEAN_COUNT + 1))
    fi
done

# Msieve temporary files
for file in msieve.fb msieve.dat.chk msieve.log; do
    if [ -f "$file" ]; then
        echo "  Removing $file..."
        rm -f "$file"
        CLEAN_COUNT=$((CLEAN_COUNT + 1))
    fi
done

# Intermediate conversion files (from cado_to_msieve.py cleanup and process_batches.sh)
for file in outMsieve.ms outMsieve.ms.tmp outCado.ms poly.ms poly.p batch_input.ms batch_input.ms.deduped cado_results.ms .cado_processed_lines batch_lines.tmp; do
    if [ -f "$file" ]; then
        echo "  Removing $file..."
        rm -f "$file"
        CLEAN_COUNT=$((CLEAN_COUNT + 1))
    fi
done

echo "  Cleaned $CLEAN_COUNT item(s)"
echo ""

# Deep clean: remove final outputs too
if [ "$DEEP_CLEAN" = true ]; then
    echo "=== DEEP CLEAN: REMOVING FINAL OUTPUTS ==="
    DEEP_COUNT=0

    for file in cado_sopt_output.txt cado_sopt_unsorted.txt cado_results_sorted.ms ropt_comparison.txt; do
        if [ -f "$file" ]; then
            echo "  Removing $file..."
            rm -f "$file"
            DEEP_COUNT=$((DEEP_COUNT + 1))
        fi
    done

    if [ -d "pipeline_results" ]; then
        echo "  Removing pipeline_results/..."
        rm -rf pipeline_results
        DEEP_COUNT=$((DEEP_COUNT + 1))
    fi

    echo "  Removed $DEEP_COUNT final output(s)"
    echo ""
fi

echo "======================================"
echo "CLEANUP COMPLETE!"
echo "======================================"

if [ "$CREATE_BACKUP" = true ] && [ -d "$BACKUP_DIR" ]; then
    echo "Backup saved to: $BACKUP_DIR/"
fi

if [ "$KEEP_RESULTS" = true ]; then
    echo ""
    echo "Kept final outputs:"
    [ -f "cado_sopt_output.txt" ] && echo "  - cado_sopt_output.txt"
    [ -f "cado_results_sorted.ms" ] && echo "  - cado_results_sorted.ms"
    [ -d "pipeline_results" ] && echo "  - pipeline_results/"
    [ -f "ropt_comparison.txt" ] && echo "  - ropt_comparison.txt"
fi

echo ""
