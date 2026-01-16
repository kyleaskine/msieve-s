#!/bin/bash

# Verification script for NFS Optimization System
# Checks that all files are in place and configuration is valid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "NFS Optimization System - Setup Verification"
echo "======================================"
echo ""

ERRORS=0
WARNINGS=0

# Check main files
echo "Checking main files..."
FILES_TO_CHECK=(
    "nfs_optimize.sh:Main orchestration script"
    "nfs_config.ini:Configuration file"
    "README_NFS_OPTIMIZE.md:Main documentation"
    "MIGRATION_GUIDE.md:Migration guide"
)

for item in "${FILES_TO_CHECK[@]}"; do
    IFS=':' read -r file desc <<< "$item"
    if [ -f "$file" ]; then
        echo "  ✓ $desc ($file)"
    else
        echo "  ✗ MISSING: $desc ($file)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check directories
echo "Checking directories..."
DIRS_TO_CHECK=(
    "scripts:Shell scripts directory"
    "utils:Python utilities directory"
)

for item in "${DIRS_TO_CHECK[@]}"; do
    IFS=':' read -r dir desc <<< "$item"
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  ✓ $desc ($dir) - $count files"
    else
        echo "  ✗ MISSING: $desc ($dir)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check shell scripts
echo "Checking shell scripts..."
SHELL_SCRIPTS=(
    "scripts/dedupe_and_sopt.sh"
    "scripts/process_batches.sh"
    "scripts/full_optimization_pipeline.sh"
    "scripts/run_msieve_ropt_annotated.sh"
    "scripts/cleanup.sh"
)

for script in "${SHELL_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "  ✓ $script (executable)"
        else
            echo "  ⚠ $script (not executable)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ✗ MISSING: $script"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check Python utilities
echo "Checking Python utilities..."
PYTHON_SCRIPTS=(
    "utils/cado_to_msieve.py"
    "utils/sort_cado_by_expe.py"
    "utils/extract_input_polys_sorted.py"
    "utils/extract_top_cado_poly.py"
    "utils/invert_c_coefficients.py"
    "utils/invert_msieve_single_line.py"
)

for script in "${PYTHON_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "  ✓ $script (executable)"
        else
            echo "  ⚠ $script (not executable)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ✗ MISSING: $script"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check main script is executable
echo "Checking main script..."
if [ -x "nfs_optimize.sh" ]; then
    echo "  ✓ nfs_optimize.sh is executable"
else
    echo "  ✗ nfs_optimize.sh is not executable"
    echo "    Run: chmod +x nfs_optimize.sh"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check configuration
echo "Checking configuration..."
if [ -f "nfs_config.ini" ]; then
    # Parse CADO build dir from config
    CADO_DIR=$(awk -F= '/^\s*cado_build_dir/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/#.*$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' nfs_config.ini | head -n1)

    if [ -n "$CADO_DIR" ]; then
        # Expand environment variables
        CADO_DIR=$(eval echo "$CADO_DIR")

        if [ -d "$CADO_DIR" ]; then
            echo "  ✓ CADO build directory exists: $CADO_DIR"

            # Check for sopt
            if [ -f "$CADO_DIR/polyselect/sopt" ]; then
                echo "    ✓ CADO sopt found"
            else
                echo "    ✗ CADO sopt not found at $CADO_DIR/polyselect/sopt"
                ERRORS=$((ERRORS + 1))
            fi

            # Check for ropt
            if [ -f "$CADO_DIR/polyselect/polyselect_ropt" ]; then
                echo "    ✓ CADO polyselect_ropt found"
            else
                echo "    ⚠ CADO polyselect_ropt not found at $CADO_DIR/polyselect/polyselect_ropt"
                WARNINGS=$((WARNINGS + 1))
            fi
        else
            echo "  ✗ CADO build directory not found: $CADO_DIR"
            echo "    Update cado_build_dir in nfs_config.ini"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "  ✗ cado_build_dir not configured in nfs_config.ini"
        ERRORS=$((ERRORS + 1))
    fi

    # Check msieve
    MSIEVE=$(awk -F= '/^\s*msieve_binary/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/#.*$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' nfs_config.ini | head -n1)
    MSIEVE=${MSIEVE:-"./msieve"}

    if [ -f "$MSIEVE" ]; then
        if [ -x "$MSIEVE" ]; then
            echo "  ✓ msieve binary found and executable: $MSIEVE"
        else
            echo "  ⚠ msieve binary found but not executable: $MSIEVE"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ✗ msieve binary not found: $MSIEVE"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ nfs_config.ini not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  ✓ Python3 available: $PYTHON_VERSION"
else
    echo "  ✗ Python3 not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "======================================"
echo "VERIFICATION SUMMARY"
echo "======================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "Your system is ready to use."
    echo ""
    echo "Next steps:"
    echo "  1. Review configuration: ./nfs_optimize.sh config"
    echo "  2. Start processing: ./nfs_optimize.sh batch"
    echo "  3. Read documentation: README_NFS_OPTIMIZE.md"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠ Warnings: $WARNINGS"
    echo ""
    echo "Your system should work, but some files aren't executable."
    echo "To fix, run: chmod +x nfs_optimize.sh scripts/*.sh utils/*.py"
    echo ""
    exit 0
else
    echo "✗ Errors: $ERRORS"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠ Warnings: $WARNINGS"
    fi
    echo ""
    echo "Please fix the errors above before using the system."
    echo ""
    echo "Common fixes:"
    echo "  - Update cado_build_dir in nfs_config.ini"
    echo "  - Make scripts executable: chmod +x nfs_optimize.sh scripts/*.sh utils/*.py"
    echo "  - Compile msieve if not present"
    echo ""
    exit 1
fi
