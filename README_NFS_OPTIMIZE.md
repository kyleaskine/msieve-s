# NFS Polynomial Optimization System

Unified workflow for optimizing NFS polynomials using msieve and CADO-NFS.

## Overview

This system provides a streamlined interface for polynomial optimization in the Number Field Sieve (NFS). It integrates:
- **msieve** for initial polynomial generation and root optimization
- **CADO-NFS** for size optimization (sopt) and root optimization (ropt)

## Quick Start

### 1. Configuration

Create your local configuration file from the template:

```bash
cp nfs_config.ini.template nfs_config.ini
```

Then edit `nfs_config.ini` to match your system:

```ini
[paths]
# Update this to your CADO-NFS build directory
cado_build_dir = $HOME/cado-nfs/build/YOUR-BUILD-DIR
```

**Note**: `nfs_config.ini` is in `.gitignore` and won't be committed to version control.

### 2. Basic Usage

```bash
# View current configuration
./nfs_optimize.sh config

# Continuous batch processing (standalone)
./nfs_optimize.sh batch

# Or: preprocess first, then run full pipeline
./nfs_optimize.sh preprocess
./nfs_optimize.sh pipeline

# Watch results in real-time (in another terminal)
./nfs_optimize.sh watch
```

## Directory Structure

```
msieve-s/
├── nfs_optimize.sh          # Main entry point
├── nfs_config.ini.template  # Configuration template (copy to nfs_config.ini)
├── nfs_config.ini           # Local configuration (gitignored)
├── scripts/                 # Workflow scripts
│   ├── dedupe_and_sopt.sh
│   ├── process_batches.sh
│   ├── full_optimization_pipeline.sh
│   ├── run_msieve_ropt_annotated.sh
│   └── cleanup.sh
├── utils/                   # Python utilities
│   ├── cado_to_msieve.py
│   ├── sort_cado_by_expe.py
│   ├── extract_input_polys_sorted.py
│   ├── extract_top_cado_poly.py
│   ├── invert_c_coefficients.py
│   └── invert_msieve_single_line.py
└── README_NFS_OPTIMIZE.md   # This file
```

## Workflows

### Workflow 1: Continuous Batch Processing (Recommended)

Best for ongoing optimization of large polynomial sets. Runs standalone without preprocessing.

```bash
# Start continuous batch processing
./nfs_optimize.sh batch
```

**What it does:**
1. **Phase 1**: Continuously monitors `msieve.dat.ms` for new polynomials
2. Runs CADO size optimization (sopt) on all new polynomials
3. **Phase 2**: Takes top N polynomials by exp_E and runs root optimization
4. Outputs final results to `outMsieve.p`
5. Repeats continuously

**Configuration** (in `nfs_config.ini`):
- `batch_processing.batch_size`: Number of polynomials to root-optimize per cycle (default: 100)
- `batch_processing.sleep_interval`: Seconds between cycles (0 = continuous)
- `system.threads`: Number of parallel workers

**To stop gracefully:**
```bash
# Find the process ID
ps aux | grep process_batches.sh

# Send graceful shutdown signal
kill -USR1 <PID>
```

### Workflow 2: Full Optimization Pipeline

Best for targeted optimization of a specific polynomial set. Requires preprocessing first.

```bash
# Step 1: Preprocess your input polynomials
./nfs_optimize.sh preprocess

# Step 2: Run full optimization pipeline
./nfs_optimize.sh pipeline
```

**What it does:**
1. Extracts top N polynomials from initial sopt results
2. Re-runs sopt with higher effort on those N
3. Sorts by exp_E
4. Runs msieve root optimization on best M₁ polynomials
5. Runs CADO root optimization on best M₂ polynomials
6. Tests both original and inverted C coefficients

**Configuration**:
- `pipeline.extract_top_n`: How many to extract for re-optimization (default: 100)
- `pipeline.resopt_effort`: Sopteffort level for re-optimization (default: 10)
- `pipeline.msieve_ropt_count`: How many for msieve ropt (default: 10)
- `pipeline.cado_ropt_count`: How many for CADO ropt (default: 100)
- `pipeline.ropt_effort`: CADO ropteffort level (default: 10)

### Workflow 3: Simple Preprocessing

Just deduplicate and run CADO size optimization once.

```bash
./nfs_optimize.sh preprocess
```

**What it does:**
1. Deduplicates polynomials from `msieve.dat.ms`
2. Runs CADO sopt with multithreading
3. Sorts results by exp_E
4. Outputs:
   - `cado_sopt_output.txt` (CADO format, sorted)
   - `cado_results_sorted.ms` (msieve format, sorted)

## Testing & Analysis

### Test sopteffort Parameter

Compare default sopt vs higher effort levels:

```bash
./nfs_optimize.sh test-sopteffort
```

**Configuration**:
- `testing.test_poly_count`: Number of polynomials to test (default: 100)
- `testing.test_sopt_effort`: Effort level to test (default: 10)

### Compare Root Optimization Methods

Compare msieve vs CADO, original vs inverted:

```bash
./nfs_optimize.sh test-ropt
```

**Configuration**:
- `testing.ropt_comparison_count`: Number of polynomials to test (default: 10)

## Monitoring

### Watch Results

Monitor root-optimized results in real-time:

```bash
./nfs_optimize.sh watch
```

Displays top 25 polynomials sorted by Murphy E score, refreshing every 60 seconds.

## Maintenance

### Cleanup

Remove intermediate files while keeping results:

```bash
./nfs_optimize.sh cleanup
```

Deep clean (removes everything including results):

```bash
./nfs_optimize.sh cleanup --deep
```

**Note**: Cleanup automatically creates backups in `backup_YYYYMMDD_HHMMSS/`

## Key Files

### Input Files
- `msieve.dat.ms` - Stage1 polynomials (msieve format)
- `worktodo.ini` - Work configuration for msieve

### Output Files
- `outMsieve.p` - Final root-optimized polynomials (msieve format)
- `cado_sopt_output.txt` - CADO sopt results (sorted by exp_E)
- `cado_results_sorted.ms` - CADO sopt results in msieve format
- `pipeline_results/` - Complete pipeline results

### Tracking Files
- `.cado_processed_lines` - Tracks which lines have been processed by CADO
- `cado_results.ms` - Accumulated CADO results waiting for root optimization

### Log Files
- `msieve.log` - msieve execution log
- `rootopt_errors.log` - Root optimization errors (if any)

## Configuration Reference

### Critical Settings

#### `[paths]`
- `cado_build_dir`: **MUST BE UPDATED** for your system
- `msieve_binary`: Path to msieve (default: `./msieve`)

#### `[system]`
- `threads`: Number of CPU threads (0 = auto-detect)

#### `[batch_processing]`
- `batch_size`: Polynomials per root optimization batch
- `sleep_interval`: Seconds between cycles (0 = continuous)

### Performance Tuning

**For faster processing:**
- Increase `system.threads` (up to your CPU core count)
- Decrease `batch_processing.batch_size` (processes smaller batches more frequently)
- Set `preprocessing.sopt_effort = 0` (fastest sopt)

**For better quality:**
- Increase `preprocessing.sopt_effort` (0-10, higher = better but slower)
- Increase `pipeline.resopt_effort` (re-optimize top polys with higher effort)
- Increase `pipeline.ropt_effort` (more thorough root optimization)

**For testing:**
- Decrease `testing.test_poly_count` for faster tests
- Increase for more comprehensive analysis

## Porting to Another PC

1. Copy the entire directory to the new PC
2. Update `nfs_config.ini`:
   ```ini
   [paths]
   cado_build_dir = /path/to/your/cado-nfs/build/YOUR-BUILD-NAME
   ```
3. Ensure msieve binary is compiled for the new system
4. Run: `./nfs_optimize.sh config` to verify settings
5. Start processing: `./nfs_optimize.sh batch`

## Understanding Polynomial Metrics

### exp_E (Expected Difficulty)
- **Lower is better**
- Predicts how hard the polynomial will be to factor
- Used as primary sorting criterion

### Murphy E
- **Higher is better**
- Root optimization goal
- Final quality metric for factorization

### Alpha (Projective Alpha)
- **More negative is better**
- Size optimization goal
- Intermediate metric during sopt

## Troubleshooting

### "CADO sopt not found"
Update `cado_build_dir` in `nfs_config.ini` to match your CADO-NFS build directory.

### "msieve not found"
Ensure msieve is compiled in the current directory, or update `msieve_binary` path.

### Scripts fail with "command not found"
Run: `chmod +x nfs_optimize.sh scripts/*.sh utils/*.py`

### Batch processing stuck
Check `msieve.dat.ms` exists and contains polynomials in the correct format.

### Out of disk space
Run `./nfs_optimize.sh cleanup` to remove intermediate files.

## Advanced Usage

### Manual Script Execution

If you need to run individual scripts directly:

```bash
cd scripts

# Run preprocessing with custom settings
./dedupe_and_sopt.sh -t 8 -i ../msieve.dat.ms

# Run batch processing with custom batch size
./process_batches.sh -t 8 -b 200

# Run full pipeline with custom parameters
./full_optimization_pipeline.sh -n 200 --msieve-ropt 20 -t 8
```

### Custom Python Utilities

All Python utilities are in `utils/` and can be used standalone:

```bash
# Convert CADO output to msieve format
python3 utils/cado_to_msieve.py input.txt output.ms

# Sort CADO output by exp_E
python3 utils/sort_cado_by_expe.py input.txt sorted.txt

# Extract top N polynomials
python3 utils/extract_top_cado_poly.py input.txt output.txt 100
```

Run any script with `-h` or `--help` for detailed usage.

## Summary of Changes

This unified system consolidates 8 shell scripts and 6 Python utilities into:
- **1 main interface** (`nfs_optimize.sh`)
- **1 config file** (`nfs_config.ini`)
- **Organized structure** (`scripts/` and `utils/` directories)

### Benefits
- ✅ Single entry point for all operations
- ✅ Portable configuration file
- ✅ Consistent parameter handling
- ✅ Better organization
- ✅ Easier to use on multiple PCs

### Script Status

**Active Production Scripts:**
- `dedupe_and_sopt.sh` - Preprocessing
- `process_batches.sh` - Continuous batch processing
- `full_optimization_pipeline.sh` - Complete pipeline
- `run_msieve_ropt_annotated.sh` - Helper for msieve

**Testing Scripts:**
- `test_sopteffort.sh` - Parameter testing
- `test_ropt_comparison.sh` - Method comparison

**Utilities:**
- `cleanup.sh` - Maintenance
- `watcher.sh` - Monitoring (now integrated into main script)

**All Python utilities are active and used by the workflows.**

## Performance Characteristics

Understanding the speed of each stage helps optimize your workflow:

| Stage | Typical Speed | Recommended Threads | Notes |
|-------|---------------|---------------------|-------|
| **Deduplication** | Very fast (seconds) | N/A (single-threaded) | Removes 5-20% duplicates typically |
| **CADO sopt** | Very fast (~1-5 seconds/poly) | Use all cores (8-16+) | Highly parallelizable |
| **msieve -npr** | Slow (~1-10 minutes/poly) | 4-8 threads | CPU/memory intensive |
| **CADO ropt** | Medium (~10-60 seconds/poly) | 4-8 threads | Faster than msieve |

### Multithreading Recommendations

**For fastest initial processing:**
```ini
[preprocessing]
sopt_threads = 16  # Use all available cores
```

**For balanced continuous processing:**
```ini
[system]
threads = 8  # Balance with other tasks

[batch_processing]
batch_size = 100  # Process 100 at a time
```

**For overnight runs:**
```ini
[batch_processing]
batch_size = 200  # Larger batches
sleep_interval = 300  # 5 minutes between cycles
```

### Expected Timings (Example: 100 Polynomials)

On a typical 16-core system:
- Deduplication: ~1 second
- CADO sopt (16 threads): ~5-30 seconds total
- msieve ropt (8 threads): ~10-60 minutes total
- CADO ropt (8 threads): ~2-10 minutes total

## Expanded Troubleshooting

### Setup Issues

**"CADO sopt not found at..."**
```bash
# Find your CADO build directory
find ~/cado-nfs -type d -name "build" 2>/dev/null

# Update config
nano nfs_config.ini
# Change: cado_build_dir = /path/to/your/build/dir
```

**"msieve not found"**
```bash
# Check if msieve is compiled
ls -la msieve

# If not, compile it
make

# Or update path in config
nano nfs_config.ini
# Change: msieve_binary = /full/path/to/msieve
```

**"Python3 not found"**
```bash
# Install Python 3
sudo apt install python3  # Ubuntu/Debian
sudo yum install python3  # CentOS/RHEL
```

### Runtime Issues

**"cannot open worktodo.ini"**
```bash
# Create worktodo.ini with your number
echo "N 123456789..." > worktodo.ini
```

**Batch processing seems stuck**
```bash
# Check if input file exists
ls -lh msieve.dat.ms

# Check file format (should be multi-line msieve format)
head -20 msieve.dat.ms

# Check for new polynomials
wc -l msieve.dat.ms
```

**Deduplication removes too many polynomials**
```bash
# Review deduplicated output
diff msieve.dat.ms msieve.dat.deduped.ms | less

# Check for exact duplicates in input
# (This is normal - duplicates waste processing time)
```

**Out of disk space**
```bash
# Check disk usage
df -h .

# Clean intermediate files
./nfs_optimize.sh cleanup

# For emergency, deep clean
./nfs_optimize.sh cleanup --deep
```

**Multithreading not working / system overloaded**
```bash
# Check CPU usage
top

# Check available cores
nproc

# Reduce threads in config
nano nfs_config.ini
# Set: threads = 4  (or lower)
```

**Pipeline fails during sopt/ropt**
```bash
# Check for errors in log files
tail -100 msieve.log
tail -100 rootopt_errors.log

# Check intermediate work directories
ls -la pipeline_work/
ls -la sopt_batch_chunks/

# Try reducing thread count to isolate issues
```

**Results look wrong / Murphy E scores too low**
```bash
# Check polynomial degree detection
head -5 cado_results_sorted.ms | wc -w
# Should be 11 columns (deg 5) or 12 columns (deg 6)

# Verify exp_E values are present
head -5 outMsieve.p | grep exp_E

# Run comparison test to verify quality
./nfs_optimize.sh test-ropt
```

### Performance Issues

**Processing is too slow**
```bash
# Increase threads for sopt (cheap operation)
nano nfs_config.ini
[preprocessing]
sopt_threads = 16

# But keep ropt threads moderate (expensive)
[system]
threads = 8
```

**CPU usage is too high**
```bash
# Reduce thread counts
nano nfs_config.ini
[system]
threads = 4

# Add sleep between cycles
[batch_processing]
sleep_interval = 60
```

**Disk I/O is bottleneck**
```bash
# Reduce batch size
nano nfs_config.ini
[batch_processing]
batch_size = 50

# Check disk speed
iostat -x 5
```

## Support

For issues or questions:
1. Check configuration: `./nfs_optimize.sh config`
2. Run verification: `./verify_setup.sh`
3. Review logs: `msieve.log`, `rootopt_errors.log`
4. Run cleanup if disk space is low: `./nfs_optimize.sh cleanup`
5. Check the Troubleshooting section above

## License

Same as msieve and CADO-NFS parent projects.
