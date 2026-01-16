# Migration Guide - NFS Optimization System

## What Changed

Your NFS polynomial optimization scripts have been reorganized into a unified system.

### Before (Old Structure)
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

### After (New Structure)
```
msieve-s/
├── nfs_optimize.sh          ← NEW: Main entry point
├── nfs_config.ini.template  ← NEW: Configuration template
├── nfs_config.ini           ← NEW: Local configuration (gitignored)
├── scripts/                 ← NEW: All shell scripts moved here
│   ├── dedupe_and_sopt.sh
│   ├── process_batches.sh
│   ├── full_optimization_pipeline.sh
│   ├── run_msieve_ropt_annotated.sh
│   └── cleanup.sh
└── utils/                   ← NEW: All Python scripts moved here
    ├── cado_to_msieve.py
    ├── sort_cado_by_expe.py
    ├── extract_input_polys_sorted.py
    ├── extract_top_cado_poly.py
    ├── invert_c_coefficients.py
    └── invert_msieve_single_line.py
```

## How to Use the New System

### Old Way → New Way

| Old Command | New Command |
|-------------|-------------|
| `./dedupe_and_sopt.sh` | `./nfs_optimize.sh preprocess` |
| `./process_batches.sh -t 8` | `./nfs_optimize.sh batch` (threads configured in config) |
| `./full_optimization_pipeline.sh -n 100` | `./nfs_optimize.sh pipeline` (settings in config) |
| `./test_sopteffort.sh` | `./nfs_optimize.sh test-sopteffort` |
| `./test_ropt_comparison.sh` | `./nfs_optimize.sh test-ropt` |
| `./cleanup.sh` | `./nfs_optimize.sh cleanup` |
| `./watcher.sh` | `./nfs_optimize.sh watch` |

### Configuration Changes

**Before**: Settings hard-coded in each script or passed as command-line arguments

**After**: All settings in `nfs_config.ini` - edit once, works everywhere

## First-Time Setup

### Step 1: Create Local Configuration

Copy the template to create your local configuration:

```bash
cp nfs_config.ini.template nfs_config.ini
```

### Step 2: Update Configuration

Edit `nfs_config.ini` and update the CADO-NFS path:

```ini
[paths]
cado_build_dir = $HOME/cado-nfs/build/YOUR-BUILD-DIR  ← UPDATE THIS
```

To find your CADO build directory:
```bash
find ~/cado-nfs -type d -name "build" 2>/dev/null
```

**Note**: `nfs_config.ini` is in `.gitignore` and won't be committed to version control.

### Step 3: Verify Setup

```bash
./nfs_optimize.sh config
```

This will show your current configuration and verify all paths are correct.

### Step 4: Test Basic Functionality

```bash
# If you have existing data files, run:
./nfs_optimize.sh preprocess

# Otherwise, verify the scripts are found:
ls -la scripts/ utils/
```

## What Stayed the Same

- **All workflows still work exactly the same way internally**
- **Input/output file formats unchanged**
- **No changes to msieve or CADO-NFS usage**
- **Data files remain in the same location**

## Benefits of the New System

### 1. Portability
Copy to a new PC and just edit one config file:
```bash
# On new PC:
scp -r msieve-s/ newpc:~/
ssh newpc
cd ~/msieve-s
nano nfs_config.ini  # Update cado_build_dir
./nfs_optimize.sh config  # Verify
./nfs_optimize.sh batch   # Start working!
```

### 2. Consistency
All settings in one place:
- Thread counts
- Batch sizes
- Effort levels
- File paths

### 3. Simplicity
One command to rule them all:
```bash
./nfs_optimize.sh <command>
```

### 4. Organization
- All shell scripts in `scripts/`
- All Python utilities in `utils/`
- Main interface at root level

## Troubleshooting Migration

### "nfs_optimize.sh: command not found"
```bash
chmod +x nfs_optimize.sh
```

### "CADO sopt not found"
Update `cado_build_dir` in `nfs_config.ini`

### "Python scripts not found"
Scripts have been moved to `utils/` but all path references are updated automatically.

### Want to use old commands?
You can still run scripts directly:
```bash
cd scripts
./process_batches.sh -t 8 -b 100
```

But you'll need to manually specify parameters. The new unified interface is recommended.

## What If I Don't Like the New Structure?

The old scripts are unchanged (just moved to `scripts/` directory). You can:

1. **Use the old way**: `cd scripts && ./script_name.sh`
2. **Copy scripts back to root**: `cp scripts/*.sh . && cp utils/*.py .`
3. **Mix and match**: Use `nfs_optimize.sh` for some tasks, direct scripts for others

## Questions?

See `README_NFS_OPTIMIZE.md` for comprehensive documentation.

Quick reference:
```bash
./nfs_optimize.sh help        # Show all commands
./nfs_optimize.sh config      # Show configuration
./nfs_optimize.sh preprocess  # Run preprocessing
./nfs_optimize.sh batch       # Start batch processing
```
