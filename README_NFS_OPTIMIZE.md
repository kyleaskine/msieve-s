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

## GPU workflow

The GPU mode runs degree-5 size and root optimization from raw stage-1 output.
Build its tools once with `make -C stage23_gpu s23_sopt s23_ropt CUDA=120`
(choose the CUDA architecture for your device). The host scorer builds on demand
using the configured CADO libraries. Size modeling, exact-alpha reranking,
MurphyE scoring and content-seed generation still use the CPU.

One command from raw polynomials to a final polynomial:

```bash
./nfs_optimize.sh --gpu --size big all \
    --input msieve.dat.ms --run-dir stage23_gpu/runs/my-job \
    --params job:32,33,16.5,80e6 \
    --lattice 17,32768 --qband 80e6,450e6,7
```

These scoring numbers are examples: supply the current job's large-prime bounds,
region and q band. `--params` is `job:lpbr,lpba,I,qmin`; for a `2^LOGI × J`
rectangle, the equivalent MurphyE `I` is `(LOGI + log2(J))/2`. Set `--lattice`
and `--qband` together to rank by the lattice band model. Omitting both ranks by
MurphyE. The `[gpu]` config section can supply these values and search settings.

The existing two-step workflow is also available:

```bash
./nfs_optimize.sh --gpu preprocess \
    --input msieve.dat.ms --run-dir stage23_gpu/runs/my-job
./nfs_optimize.sh --gpu --size big pipeline \
    --run-dir stage23_gpu/runs/my-job \
    --params job:32,33,16.5,80e6 \
    --lattice 17,32768 --qband 80e6,450e6,7
```

`preprocess` only deduplicates raw polynomials and runs GPU sopt at effort 0;
it needs no scoring parameters. `pipeline` takes those results, selects the
original raw inputs for the higher-effort sopt pass, generates parent/content
seeds, searches them, and selects the final polynomial. `pipeline --input FILE`
also performs preprocessing, just like `all`. Without `--input`, `preprocess`
and a new `all` run use `[preprocessing] input_file` (normally `msieve.dat.ms`).
Input files use the existing multiline `n:`, `c0:`…`c5:`, `Y0:`, `Y1:` format.

To reuse a completed CPU preprocessing pass, supply its **unsorted paired**
output; the raw/output pairs are needed to recover the original shortlist:

```bash
./nfs_optimize.sh --gpu pipeline \
    --initial-unsorted cado_sopt_unsorted.txt \
    --run-dir stage23_gpu/runs/from-cpu \
    --params job:32,33,16.5,80e6
```

The default search uses two workers with short `-launch 30` kernels, a breadth
cap of `2e11` cells per search, and depth at `6e12` on the best 16 distinct parent
families. `--depth-seeds 0` disables depth. Accurate MurphyE translation/skew
refinement runs on the four leading families over the entire retained pool
(4096 cells by default), before cutting the outputs. Breadth and depth save
those pools, so refinement reuses the completed search without another sieve.
Old runs retain their frozen binaries; binaries without pool support continue
using the older combined search/refinement path. This is followed by
independent final scoring at 256k angles; lattice validation uses a different
sample stream with 256 lattices and 4096 points. Coarse-pass controls remain in
the finalist pool. These are model rankings, not measured test-sieve gains.

Add `--ropt-batch-size 16` to run groups of pending seeds in a persistent ropt
process. This requires a binary advertising `ropt-batch-v1` and
`ropt-batch-events-v1`; the default is 1. Small phases split across the requested
workers. Each completed seed receives its own validated
checkpoint, so an interruption or a later failed seed preserves earlier work.
Batch size may change on resume. This option changes process scheduling only.
Older frozen binaries can resume with `--ropt-batch-size 1`; use a new run
directory with rebuilt binaries for batching. Rejected execution settings are
not saved. Progress updates for each completed or reused seed, and completion
events are separate from the byte-preserving command logs.

For an experimental GPU screen, add `--alpha-screen 4096,500
--alpha-screen-exact-size --alpha-screen-gpu --rerank 512` to a new run.
Breadth/depth then screen 4096 proxy candidates down to 512 before full
refinement. Their saved pools still contain all 4096 starts, which final
MurphyE refinement evaluates without the alpha cut. `--alpha-screen-check`
compares GPU scores against the CPU oracle and is intended for validation.
Screening is off by default; changing its policy requires a new run directory.
Blank screen Boolean configuration values mean false. Runs predating these
options keep screening disabled when resumed, regardless of the global config.

Every run stores its own preprocessing files, inputs, frozen executables,
command logs and verified checkpoints. No CPU pipeline output rotation is used.
Paths are relative to this repository unless absolute. The default run directory
is `stage23_gpu/runs/current`; use a named directory for each job. Resume and
report commands need no repeated scoring options:

```bash
./nfs_optimize.sh --gpu pipeline --run-dir stage23_gpu/runs/my-job --resume
./nfs_optimize.sh --gpu report --run-dir stage23_gpu/runs/my-job
```

Successful checkpoints are reused automatically after checking their hashes.
Failed commands and missing or changed outputs are rerun. A continuation inherits
the saved pipeline settings and frozen binaries, so it can finish after the
original raw file is moved or grows. New runs copy and hash the consumed input
bytes into `RUN/inputs/` before preprocessing; a live source is read only up to
its initial byte length, and the snapshot must contain complete polynomial blocks.
Resuming uses that snapshot. Use a new run directory to consume more input or
change search/scoring settings. Execution settings (`--device`, `--threads`,
`--workers`, `--launch`, `--slice`, `--ropt-batch-size`) can change on resume. Logs record actual
commands, and `execution.json` saves the latest execution settings.

Source hashes are provenance. Explicit per-stage schema versions determine
checkpoint compatibility; unrelated script edits do not invalidate completed
work. Old checkpoints with matching parameters and output hashes remain usable.
Old runs without snapshots can still reuse their verified preprocessing outputs;
repairing those outputs requires the original bytes. A lock prevents concurrent
writers to the same run.

Partial MurphyE scoring failures retain valid, exactly checked outputs and record
the failed search in the summary/report. Empty, malformed, or unrelated failed
outputs remain errors. Finalist limits count unique polynomial/skew configurations,
so repeated breadth/depth candidates do not consume extra slots.

The winner is `RUN/results/best.poly`; all validated finalists, scores and a
readable report are in `RUN/results/`. Progress is in `RUN/status.json` and
per-command logs are in `RUN/logs/`. `RUN/preprocess/` includes the familiar
`cado_sopt_unsorted.txt`, `cado_sopt_output.txt`, and `cado_results_sorted.ms`.
Use `./nfs_optimize.sh --gpu pipeline --help` for all options.

Experimental distant-basin discovery is available with
`--u-scan 256 --u-basins 4 --u-radius 128` on `all` or `pipeline`, or the matching
`u_scan`, `u_basins`, `u_radius` keys in `[gpu]`. It samples across `--umax`, refines
promising locations, and adds nearby lines before the existing cell cap. It is
off by default (`--u-scan 0`). Compare in a separate run directory: the added size
model work and changed cell allocation can help some seeds and hurt others.
The [root optimization plan](ROPT_OPTIMIZATION_PLAN.md) records the coverage
tests and subsequent residue-class work.

The first explicit residue-class mode is available as `--residue M,U,V`, with
`--u-scan 0`. For example, `--residue 2520,1488,1704` restricts each seed's search
to that original-coordinate class. It counts selected cells against the budget
and supports both `all` and `pipeline`; settings are saved for resume. This is
an experimental single-class mode, and the same class need not suit different
seeds. For a content-derived seed, the driver converts the parent class through
`u=u0+d*u'`, `v=v0+d*v'`; incompatible content lattices are skipped.

Automatic selection is available with `--classes 2520,16 --class-explore 4`,
on either `all` or `pipeline`, including a pipeline starting from raw input or
an earlier preprocess pass. It chooses classes separately for each seed and
shares one cell cap and candidate pool across them. The matching config keys
are `[gpu] classes=2520,16` and `class_explore=4`; leave `residue` empty and
`u_scan=0`. Exploration reserves class slots; later size and retention cuts can
still remove their cells. This mode is experimental and off by default. Use a
new run directory for changed search settings. See the
[root-sieve details](stage23_gpu/README.md) and
[measured results](ROPT_OPTIMIZATION_PLAN.md#2026-10-14-r2b-automatic-selection-and-shared-budget).

## Current Wrapper State

`./nfs_optimize.sh` currently supports these commands:

```bash
./nfs_optimize.sh preprocess
./nfs_optimize.sh batch [-c N]
./nfs_optimize.sh pipeline
./nfs_optimize.sh all
./nfs_optimize.sh report
./nfs_optimize.sh cleanup [--deep]
./nfs_optimize.sh watch
./nfs_optimize.sh config
./nfs_optimize.sh help
```

`report` re-runs only the final phase of the pipeline (the comparison report)
against the results already in `pipeline_results/`. It recomputes nothing
(existing skewopt results are reprinted; skewopt only runs if they are missing),
takes a couple of seconds, and is the thing to run when the pipeline finished but
you lost the terminal it was printing to.

- The counts and effort levels come from `pipeline_results/pipeline_settings.txt`,
  which the pipeline writes when it starts filling `pipeline_results/` and marks
  complete when root optimization finishes. `report` warns if the last run did not
  finish.
- For results from before that file existed, the settings are recovered from the
  previous `pipeline_report.txt`. Anything that can't be recovered is shown as
  `unknown` rather than guessed.
- Values you pass explicitly (e.g. `report -t 8`) apply to that one report only.
- When the regenerated report differs from the existing one, the old one is kept as
  `pipeline_report.prev.txt`.
- `report` exits with an error if `pipeline_results/` holds no ropt results.
- Each run owns `pipeline_results/`: when a new pipeline run starts writing there, the
  previous run's files are moved to `pipeline_results/previous/<timestamp>/`.
  `cleanup --deep` removes them along with everything else.
- `./nfs_optimize.sh pipeline --report-only` is the same as `report`. The underlying
  flag is `./scripts/full_optimization_pipeline.sh --report-only`.

The wrapper does **not** currently implement `test-sopteffort` or `test-ropt`; those commands are not present in `nfs_optimize.sh`, and the old standalone test scripts are not in the current tree.

Configuration is partly centralized. The wrapper uses the path settings, thread counts, batch size/sleep, pipeline counts/efforts, and size presets. Some template keys are parsed or documented but not wired through to the scripts yet; see "Configuration Reference" for details.

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
│   ├── deep_cado_ropt.sh
│   ├── deep_msieve_ropt.sh
│   ├── run_msieve_ropt_annotated.sh
│   ├── fix_cuda13_ctxcreate.sh
│   └── cleanup.sh
├── utils/                   # Python utilities
│   ├── cado_to_msieve.py
│   ├── sort_cado_by_expe.py
│   ├── extract_input_polys_sorted.py
│   ├── extract_top_cado_poly.py
│   ├── invert_c_coefficients.py
│   ├── invert_msieve_single_line.py
│   └── run_skewopt_on_best.py
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

### Optional: Post-Pipeline Deep CADO Ropt

After a low-effort pipeline pass, run a deeper CADO root-optimization pass on
only the most promising candidates:

```bash
./nfs_optimize.sh --size small pipeline
./scripts/deep_cado_ropt.sh --exp-top 32 --murphy-top 8 --effort 50 -t 8
```

The deep pass selects candidates from the size-optimized CADO files, using the
top `exp_E` entries plus any broad-pass MurphyE wildcards, then reruns
`polyselect_ropt` at the requested effort on both original and inverted
algebraic signs. Outputs go under `pipeline_results/cado_deep_ropt_effort50/`.
Use `--select-only` to write the candidate manifest without running ropt.

### Experimental: Deep Msieve Ropt Sweep

The local msieve build accepts root-optimization stage 2 sweep controls:

```bash
rootopt_stage2_steps=100 rootopt_stage2_start=1.0327 rootopt_stage2_mult=1.0327
```

The default remains the original coarse sweep of 5 steps at 1.5x. For a
Gimarel-style fine pass after the pipeline, use the wrapper:

```bash
./scripts/deep_msieve_ropt.sh
```

It auto-selects the newest `pipeline_results/best*_msieve.ms` and inverted
mate, auto-detects degree, processes the top 24, and runs the 100-step
`1.0327x` sweep. Common overrides are short:

```bash
./scripts/deep_msieve_ropt.sh --top 16 -t 8
./scripts/deep_msieve_ropt.sh --top 24 --no-inverted
```

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

**Current caveat:** `preprocessing.sopt_effort` is present in `nfs_config.ini.template`, but `preprocess` does not pass it to `scripts/dedupe_and_sopt.sh`. That script currently runs CADO `sopt` without an explicit `-sopteffort` argument.

## Testing & Analysis Status

The unified wrapper does not currently expose dedicated testing commands. These documented commands are **not implemented**:

```bash
./nfs_optimize.sh test-sopteffort
./nfs_optimize.sh test-ropt
```

There are also no `test_sopteffort.sh` or `test_ropt_comparison.sh` scripts in the current `scripts/` directory. For now, use `pipeline` with small counts, or run the lower-level scripts manually with custom inputs, when comparing sopt/ropt settings.

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

### Currently Wired Settings

These settings are actively used by `./nfs_optimize.sh`:

- `[paths]`: `cado_build_dir`, `msieve_binary`, `utils_dir`, `skewopt_binary`
- `[system]`: `threads`
- `[preprocessing]`: `input_file`, `sopt_threads`
- `[batch_processing]`: `batch_size`, `sleep_interval`
- `[pipeline]`: `extract_top_n`, `resopt_effort`, `msieve_ropt_count`, `cado_ropt_count`, `ropt_effort`
- `[size_small]`, `[size_medium]`, `[size_big]`: `extract_top_n`, `msieve_ropt_count`, `cado_ropt_count`

These settings exist in the template but are not currently wired through by the wrapper:

- `[preprocessing]`: `sopt_effort`
- `[batch_processing]`: `enabled`, `poly_degree`
- `[output]`: `work_dir`, `results_dir`, `final_output`, `keep_intermediate`, `auto_backup`
- `[files]`: all file override keys
- `[testing]`: all keys

### Performance Tuning

**For faster processing:**
- Increase `system.threads` (up to your CPU core count)
- Decrease `batch_processing.batch_size` (processes smaller batches more frequently)
- Use smaller pipeline counts while testing changes

**For better quality:**
- Increase `pipeline.resopt_effort` (re-optimize top polys with higher effort)
- Increase `pipeline.ropt_effort` (more thorough root optimization)

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

This unified system provides:
- **1 main interface** (`nfs_optimize.sh`)
- **1 primary config file** (`nfs_config.ini`) for the main workflows
- **Organized structure** (`scripts/` and `utils/` directories)

### Benefits
- ✅ Single entry point for all operations
- ✅ Portable configuration file
- ✅ Consistent parameter handling for the active wrapper commands
- ✅ Better organization
- ✅ Easier to use on multiple PCs

### Script Status

**Active Production Scripts:**
- `dedupe_and_sopt.sh` - Preprocessing
- `process_batches.sh` - Continuous batch processing
- `full_optimization_pipeline.sh` - Complete pipeline
- `run_msieve_ropt_annotated.sh` - Helper for msieve

**Utilities:**
- `cleanup.sh` - Maintenance
- `fix_cuda13_ctxcreate.sh` - CUDA 13 compatibility helper
- `watch` command - Monitoring is integrated into `nfs_optimize.sh`

Some Python utilities are workflow helpers, and some are standalone/manual tools.

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

# For quality comparisons, run a small pipeline or invoke lower-level
# scripts manually; there is no wrapper-level test-ropt command today.
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

---

## Migration from legacy scripts

Earlier revisions of this tree kept all the NFS optimization scripts loose in the repo root. They've been reorganized into the unified system documented above. This section is preserved for users coming from the old layout.

### Old structure (pre-reorg)

```
msieve-s/
├── watcher.sh
├── dedupe_and_sopt.sh
├── test_sopteffort.sh
├── test_ropt_comparison.sh
├── full_optimization_pipeline.sh
├── run_msieve_ropt_annotated.sh
├── process_batches.sh
├── cleanup.sh
├── cado_to_msieve.py
├── sort_cado_by_expe.py
├── extract_input_polys_sorted.py
├── invert_c_coefficients.py
├── extract_top_cado_poly.py
└── invert_msieve_single_line.py
```

### New structure

```
msieve-s/
├── nfs_optimize.sh          ← main entry point
├── nfs_config.ini.template  ← configuration template
├── nfs_config.ini           ← local config (gitignored)
├── scripts/                 ← all shell scripts moved here
│   ├── dedupe_and_sopt.sh
│   ├── process_batches.sh
│   ├── full_optimization_pipeline.sh
│   ├── run_msieve_ropt_annotated.sh
│   └── cleanup.sh
└── utils/                   ← all Python scripts moved here
    ├── cado_to_msieve.py
    ├── sort_cado_by_expe.py
    ├── extract_input_polys_sorted.py
    ├── extract_top_cado_poly.py
    ├── invert_c_coefficients.py
    └── invert_msieve_single_line.py
```

### Command mapping

| Old command | New command |
|---|---|
| `./dedupe_and_sopt.sh` | `./nfs_optimize.sh preprocess` |
| `./process_batches.sh -t 8` | `./nfs_optimize.sh batch` (threads from config) |
| `./full_optimization_pipeline.sh -n 100` | `./nfs_optimize.sh pipeline` (settings from config) |
| `./cleanup.sh` | `./nfs_optimize.sh cleanup` |
| `./watcher.sh` | `./nfs_optimize.sh watch` |

`test_sopteffort.sh` and `test_ropt_comparison.sh` are not currently wired into `nfs_optimize.sh`; the test-sopteffort / test-ropt subcommands are not implemented.

### Configuration change

Old: settings hard-coded in scripts or passed as CLI args.
New: workflow settings in `nfs_config.ini`. Some template keys remain documentation placeholders pending downstream-script integration — in particular `preprocessing.sopt_effort`, `[testing]`, `[output]`, and `[files]` aren't all wired through yet.

### First-time migration setup

1. **Create local configuration:**
   ```bash
   cp nfs_config.ini.template nfs_config.ini
   ```
2. **Update CADO-NFS path** in `nfs_config.ini`:
   ```ini
   [paths]
   cado_build_dir = $HOME/cado-nfs/build/YOUR-BUILD-DIR
   ```
   Find the right value with:
   ```bash
   find ~/cado-nfs -type d -name "build" 2>/dev/null
   ```
   `nfs_config.ini` is in `.gitignore` and isn't committed.
3. **Verify setup:**
   ```bash
   ./nfs_optimize.sh config
   ```
4. **Test:**
   ```bash
   ./nfs_optimize.sh preprocess  # if you have data files
   # or just verify scripts are found:
   ls -la scripts/ utils/
   ```

### What didn't change

- Active workflows behave the same internally
- Input/output file formats unchanged
- No changes to msieve or CADO-NFS usage
- Data files stay in the same location

### Migration troubleshooting

- **`nfs_optimize.sh: command not found`** — `chmod +x nfs_optimize.sh`
- **`CADO sopt not found`** — update `cado_build_dir` in `nfs_config.ini`
- **Python scripts not found** — they moved to `utils/` but path references are updated automatically.
- **Want to use old commands directly?** `cd scripts && ./script_name.sh` still works, you just have to pass parameters manually. The unified interface is recommended.

### Rolling back

The active scripts are still in `scripts/`. To go back to the old way:
1. `cd scripts && ./script_name.sh` — direct invocation
2. `cp scripts/*.sh . && cp utils/*.py .` — copy them back to root
3. Mix and match — use `nfs_optimize.sh` for some tasks, direct scripts for others.
