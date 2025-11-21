#!/bin/bash

# Auto-restart msieve with incremented min_coeff every N minutes
# Reads the last coefficient processed and restarts from there

# Configuration
RESTART_INTERVAL=$((20 * 60))  # 20 minutes in seconds
MSIEVE_ARGS="-np1 -nps \"stage1_norm=3e28\""  # Your base arguments
INITIAL_MIN_COEFF=271828183  # Starting coefficient
LOGFILE="msieve_auto.log"
COEFF_TRACKING=".last_coeff"
LAST_MSIEVE_PID=""  # Track the last spawned msieve PID for safety

# Flag for graceful shutdown
SHUTDOWN_REQUESTED=false

# Trap signals
trap 'echo ""; echo "Shutdown requested. Will stop after current msieve exits..."; SHUTDOWN_REQUESTED=true' SIGINT SIGTERM
trap 'echo ""; echo "Graceful shutdown requested (SIGUSR1)..."; SHUTDOWN_REQUESTED=true' SIGUSR1

# Show help
show_help() {
    cat << EOF
Usage: auto_restart_msieve.sh [OPTIONS]

Automatically restart msieve with incremented min_coeff every N minutes.

Options:
  -h, --help              Show this help message
  -i, --interval MINUTES  Restart interval in minutes (default: 20)
  -m, --min-coeff VALUE   Initial min_coeff (default: 271828183)
  -n, --norm VALUE        stage1_norm value (optional, omitted if not specified)
  -a, --args "ARGS"       Additional msieve arguments (default: none)

Examples:
  auto_restart_msieve.sh                          # Use defaults
  auto_restart_msieve.sh -i 30 -m 280000000       # 30 min interval, different start
  auto_restart_msieve.sh -n 2e28                  # Use stage1_norm=2e28

Control:
  Process ID: Will be displayed at startup
  Graceful shutdown: kill -USR1 <pid>  (finishes current msieve run)
  Immediate shutdown: Ctrl+C or kill -INT <pid>

EOF
    exit 0
}

# Parse arguments
RESTART_MINUTES=20
MIN_COEFF=$INITIAL_MIN_COEFF
STAGE1_NORM=""
EXTRA_ARGS=""
USER_SET_MIN_COEFF=false
USER_SET_NORM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -i|--interval)
            RESTART_MINUTES="$2"
            RESTART_INTERVAL=$((RESTART_MINUTES * 60))
            shift 2
            ;;
        -m|--min-coeff)
            MIN_COEFF="$2"
            USER_SET_MIN_COEFF=true
            shift 2
            ;;
        -n|--norm)
            STAGE1_NORM="$2"
            USER_SET_NORM=true
            shift 2
            ;;
        -a|--args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Auto-Restart Msieve Manager"
echo "========================================"
echo "Process ID: $$"
echo "For graceful shutdown: kill -USR1 $$"
echo ""
echo "Configuration:"
echo "  Restart interval: $RESTART_MINUTES minutes"
echo "  Initial min_coeff: $MIN_COEFF"
if [ "$USER_SET_NORM" = true ]; then
    echo "  stage1_norm: $STAGE1_NORM"
fi
echo "  Log file: $LOGFILE"
echo ""

# Function to extract last coefficient from msieve output
get_last_coeff() {
    # Look for lines like: "coeff 272579700 specialq 1 - 87706707 other 6180080 - 14832192"
    # Extract the first number after "coeff"
    if [ -f "$LOGFILE" ]; then
        last_coeff=$(grep "^coeff " "$LOGFILE" | tail -n 1 | awk '{print $2}')
        if [ -n "$last_coeff" ]; then
            echo "$last_coeff"
            return 0
        fi
    fi

    # Fallback to tracking file
    if [ -f "$COEFF_TRACKING" ]; then
        cat "$COEFF_TRACKING"
        return 0
    fi

    # No previous run, return initial
    echo "$MIN_COEFF"
    return 1
}

# Function to run msieve for specified duration
run_msieve_timed() {
    local min_coeff=$1
    local duration=$2

    # Safety check: ensure no stale msieve from previous iteration
    # This should never trigger if cleanup works, but provides extra safety
    if [ -n "$LAST_MSIEVE_PID" ] && kill -0 "$LAST_MSIEVE_PID" 2>/dev/null; then
        echo "ERROR: Previous msieve (PID $LAST_MSIEVE_PID) still running!"
        echo "Emergency kill..."
        kill -9 "$LAST_MSIEVE_PID"
        sleep 2
        if kill -0 "$LAST_MSIEVE_PID" 2>/dev/null; then
            echo "CRITICAL: Cannot kill previous msieve. Aborting to prevent GPU lock."
            exit 1
        fi
        echo "Previous msieve killed successfully."
    fi

    echo ""
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting msieve"
    echo "  min_coeff: $min_coeff"
    echo "  Duration: $((duration / 60)) minutes"
    echo "=========================================="

    # Build command
    local nps_params="min_coeff=$min_coeff"
    if [ "$USER_SET_NORM" = true ]; then
        nps_params="stage1_norm=$STAGE1_NORM $nps_params"
    fi
    local cmd="./msieve -np1 -nps \"$nps_params\""
    if [ -n "$EXTRA_ARGS" ]; then
        cmd="$cmd $EXTRA_ARGS"
    fi

    echo "Command: $cmd"
    echo ""

    # Get current log file size to tail only NEW lines
    local log_start_size=0
    if [ -f "$LOGFILE" ]; then
        log_start_size=$(wc -c < "$LOGFILE")
    fi

    # Run msieve in background and capture PID
    # Use stdbuf to force line buffering for real-time log output
    eval "stdbuf -oL -eL $cmd" >> "$LOGFILE" 2>&1 &
    local msieve_pid=$!
    LAST_MSIEVE_PID=$msieve_pid  # Track globally for safety checks

    echo "Msieve PID: $msieve_pid"
    echo "Running for $((duration / 60)) minutes..."
    echo ""

    # Start tailing log file for coeff lines in background (only new content)
    tail -c +${log_start_size} -f "$LOGFILE" 2>/dev/null | grep --line-buffered "^coeff " &
    local tail_pid=$!

    # Wait for duration or shutdown signal
    local elapsed=0
    local check_interval=5  # Check every 5 seconds

    while [ $elapsed -lt $duration ]; do
        if [ "$SHUTDOWN_REQUESTED" = true ]; then
            echo ""
            echo "Shutdown requested during wait period"
            break
        fi

        # Check if msieve is still running
        if ! kill -0 $msieve_pid 2>/dev/null; then
            echo ""
            echo "Warning: msieve exited early (after ${elapsed}s)"
            # Kill tail process
            kill $tail_pid 2>/dev/null
            wait $tail_pid 2>/dev/null
            wait $msieve_pid
            local exit_code=$?
            echo "Exit code: $exit_code"
            return $exit_code
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))

        # Progress indicator every minute
        if [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -lt $duration ]; then
            echo "  ... $((elapsed / 60))/$((duration / 60)) minutes elapsed"
        fi
    done

    # Stop the tail process
    kill $tail_pid 2>/dev/null
    wait $tail_pid 2>/dev/null

    # Time's up or shutdown requested - use gentle escalating shutdown
    # This is critical for GPU cleanup to prevent crashes/hangs
    # Based on testing: SIGINT often fails, SIGTERM reliably works without GPU crashes
    if kill -0 $msieve_pid 2>/dev/null; then
        echo ""
        echo "====== Gentle Shutdown Sequence ======"

        # Step 1: SIGINT (Ctrl+C) - try briefly (often doesn't work for msieve)
        echo "[1/4] Sending SIGINT (Ctrl+C) to msieve..."
        kill -INT -$msieve_pid 2>/dev/null || kill -INT $msieve_pid

        echo "      Waiting up to 15 seconds..."
        local wait_count=0
        while kill -0 $msieve_pid 2>/dev/null && [ $wait_count -lt 15 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        # Step 2: SIGTERM - most effective signal for msieve, allows GPU cleanup
        if kill -0 $msieve_pid 2>/dev/null; then
            echo "[2/4] SIGINT timeout. Sending SIGTERM (preferred for msieve)..."
            kill -TERM -$msieve_pid 2>/dev/null || kill -TERM $msieve_pid

            echo "      Waiting up to 45 seconds for GPU cleanup..."
            wait_count=0
            while kill -0 $msieve_pid 2>/dev/null && [ $wait_count -lt 45 ]; do
                sleep 1
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 10)) -eq 0 ]; then
                    echo "      ... ${wait_count}s elapsed (GPU may be cleaning up)"
                fi
            done
        fi

        # Step 3: SIGTERM again with shorter wait
        if kill -0 $msieve_pid 2>/dev/null; then
            echo "[3/4] Still running. Sending SIGTERM again..."
            kill -TERM -$msieve_pid 2>/dev/null || kill -TERM $msieve_pid

            echo "      Waiting up to 20 seconds..."
            wait_count=0
            while kill -0 $msieve_pid 2>/dev/null && [ $wait_count -lt 20 ]; do
                sleep 1
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 10)) -eq 0 ]; then
                    echo "      ... ${wait_count}s elapsed"
                fi
            done
        fi

        # Step 4: SIGKILL - last resort only (should rarely happen now)
        if kill -0 $msieve_pid 2>/dev/null; then
            echo "[4/4] WARNING: Resorting to SIGKILL (may cause GPU issues)..."
            echo "      This should rarely happen. Consider increasing SIGTERM wait times."
            kill -9 -$msieve_pid 2>/dev/null || kill -9 $msieve_pid

            # Extra delay after SIGKILL to let system fully cleanup
            echo "      Waiting 5 seconds for system cleanup..."
            sleep 5

            # Verify process is actually dead after SIGKILL
            local verify_count=0
            while kill -0 $msieve_pid 2>/dev/null && [ $verify_count -lt 10 ]; do
                sleep 0.5
                verify_count=$((verify_count + 1))
            done

            if kill -0 $msieve_pid 2>/dev/null; then
                echo "CRITICAL ERROR: Failed to kill msieve PID $msieve_pid even with SIGKILL!"
                echo "System may be unstable. Exiting to prevent multiple -np1 processes."
                exit 1
            fi

            echo "      Process terminated (forced)"
        else
            echo "Process exited gracefully"
        fi

        echo "======================================"

        wait $msieve_pid 2>/dev/null || true
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Msieve stopped"

    # Extract last coefficient processed
    local last_coeff=$(get_last_coeff)
    echo "Last coefficient processed: $last_coeff"

    # Save for next iteration
    echo "$last_coeff" > "$COEFF_TRACKING"

    return 0
}

# Main loop
echo "Starting main loop..."
echo ""

# Get starting coefficient (resume from last run if available, unless user specified -m)
if [ "$USER_SET_MIN_COEFF" = true ]; then
    CURRENT_COEFF=$MIN_COEFF
    echo "Starting from user-specified coefficient: $CURRENT_COEFF"
    echo "(Ignoring previous saved state)"
else
    CURRENT_COEFF=$(get_last_coeff)
    if [ $CURRENT_COEFF -ne $MIN_COEFF ]; then
        echo "Resuming from last saved coefficient: $CURRENT_COEFF"
    else
        echo "Starting from coefficient: $CURRENT_COEFF"
    fi
fi

iteration=1
while true; do
    echo ""
    echo "=========================================="
    echo "Iteration #$iteration"
    echo "=========================================="

    # Run msieve for the configured interval
    run_msieve_timed "$CURRENT_COEFF" "$RESTART_INTERVAL"

    # Check for shutdown
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        echo ""
        echo "=========================================="
        echo "Graceful shutdown complete"
        echo "=========================================="
        exit 0
    fi

    # Get the last coefficient and increment by 1
    # Msieve will find the next valid smooth number
    LAST_COEFF=$(get_last_coeff)
    CURRENT_COEFF=$((LAST_COEFF + 1))

    echo ""
    echo "Next iteration will start from: $CURRENT_COEFF"

    iteration=$((iteration + 1))

    # Longer delay between restarts to ensure GPU fully resets
    echo "Waiting 10 seconds before next iteration (GPU cooldown)..."
    sleep 10
done
