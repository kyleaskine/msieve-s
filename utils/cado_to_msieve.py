#!/usr/bin/env python3
"""
Convert CADO-NFS size optimization output to msieve .ms format
"""

import re
import sys
import os
import shutil
from datetime import datetime

def parse_cado_output(filename):
    """Parse CADO output and extract size-optimized polynomials"""
    polynomials = []
    current_n = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Capture n from any section (it's the same for all polys)
        if line.startswith('n:') and current_n is None:
            current_n = line.split(':', 1)[1].strip()

        # Look for size-optimized polynomial sections (not raw input sections)
        if line.startswith('### Size-optimized polynomial'):
            poly = {'n': current_n}  # Store the n value
            i += 1

            # Parse polynomial data
            while i < len(lines):
                line = lines[i].strip()

                if line.startswith('###'):
                    # Next section, back up one line
                    i -= 1
                    break

                if line.startswith('n:'):
                    poly['n'] = line.split(':', 1)[1].strip()
                elif line.startswith('Y0:'):
                    poly['Y0'] = line.split(':', 1)[1].strip()
                elif line.startswith('Y1:'):
                    poly['Y1'] = line.split(':', 1)[1].strip()
                elif line.startswith('c0:'):
                    poly['c0'] = line.split(':', 1)[1].strip()
                elif line.startswith('c1:'):
                    poly['c1'] = line.split(':', 1)[1].strip()
                elif line.startswith('c2:'):
                    poly['c2'] = line.split(':', 1)[1].strip()
                elif line.startswith('c3:'):
                    poly['c3'] = line.split(':', 1)[1].strip()
                elif line.startswith('c4:'):
                    poly['c4'] = line.split(':', 1)[1].strip()
                elif line.startswith('c5:'):
                    poly['c5'] = line.split(':', 1)[1].strip()
                elif line.startswith('c6:'):
                    poly['c6'] = line.split(':', 1)[1].strip()
                elif line.startswith('skew:'):
                    poly['skew'] = line.split(':', 1)[1].strip()
                elif 'proj' in line:
                    # Extract projective alpha from comment line
                    # Format: # side 1 lognorm X, exp_E Y, alpha Z (proj W), ...
                    match = re.search(r'\(proj\s+([-\d.]+)\)', line)
                    if match:
                        poly['proj_alpha'] = match.group(1)
                    # Extract exp_E (expected difficulty)
                    match = re.search(r'exp_E\s+([-\d.]+)', line)
                    if match:
                        poly['exp_E'] = match.group(1)
                    # Also extract Murphy alpha for reference
                    match = re.search(r'alpha\s+([-\d.]+)\s+\(proj', line)
                    if match:
                        poly['murphy_alpha'] = match.group(1)
                    break

                i += 1

            # Only add if we have all required fields
            if all(k in poly for k in ['c0', 'Y0', 'Y1', 'proj_alpha']):
                polynomials.append(poly)

        i += 1

    return polynomials

def poly_to_msieve_format(poly, include_n=False):
    """Convert polynomial to msieve .ms format

    Args:
        poly: Polynomial dictionary
        include_n: If True, return multi-line format with n: header
                  If False, return single line format
    """
    # Determine degree
    degree = 0
    for d in range(7, -1, -1):
        if f'c{d}' in poly:
            degree = d
            break

    # Build coefficient list from high to low degree
    coeffs = []
    for d in range(degree, -1, -1):
        key = f'c{d}'
        coeffs.append(poly.get(key, '0'))

    if include_n:
        # Multi-line format for msieve input
        lines = [f"n: {poly['n']}"]
        for d in range(degree, -1, -1):
            lines.append(f"c{d}: {poly.get(f'c{d}', '0')}")
        lines.append(f"Y1: {poly['Y1']}")
        lines.append(f"Y0: {poly['Y0']}")
        lines.append("")  # Blank line between polynomials
        return '\n'.join(lines)
    else:
        # Single line format: c_deg c_deg-1 ... c1 c0 Y1 Y0 proj_alpha exp_E stage1_norm
        exp_E = poly.get('exp_E', '0')
        parts = coeffs + [poly['Y1'], poly['Y0'], poly['proj_alpha'], exp_E, '0']
        return ' '.join(parts)

def cleanup_old_run():
    """Backup important files and delete intermediate files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'backup_{timestamp}'

    # Files to backup (and then delete)
    backup_files = ['outMsieve.p', 'msieve.dat.ms']

    # Files to delete (intermediate files)
    delete_files = [
        'outMsieve.ms',
        'outMsieve.ms.tmp',
        'outCado.ms',
        'poly.ms',
        'poly.p',
        'batch_input.ms',
        'msieve.log',
        'cado_results.ms',
        '.cado_processed_lines',
        'batch_lines.tmp'
    ]

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Created backup directory: {backup_dir}", file=sys.stderr)

    # Backup important files
    backed_up = []
    for filename in backup_files:
        if os.path.exists(filename):
            shutil.copy2(filename, os.path.join(backup_dir, filename))
            backed_up.append(filename)
            print(f"  Backed up: {filename}", file=sys.stderr)
        else:
            print(f"  Warning: {filename} not found, skipping backup", file=sys.stderr)

    # Delete backed up files
    for filename in backup_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"  Deleted: {filename}", file=sys.stderr)

    # Delete intermediate files
    deleted = []
    for filename in delete_files:
        if os.path.exists(filename):
            os.remove(filename)
            deleted.append(filename)
            print(f"  Deleted: {filename}", file=sys.stderr)

    print(f"\nCleanup complete:", file=sys.stderr)
    print(f"  {len(backed_up)} files backed up to {backup_dir}/", file=sys.stderr)
    print(f"  {len(backed_up) + len(deleted)} files deleted from working directory", file=sys.stderr)

    return backup_dir

def main():
    # Check for help flag
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: cado_to_msieve.py <cado_output.ms> [output.ms] [--top-n N] [--msieve-format] [--cleanup]")
        print()
        print("Convert CADO-NFS size optimization output to msieve .ms format")
        print()
        print("Arguments:")
        print("  cado_output.ms           Input file with CADO-NFS size-optimized polynomials")
        print("  output.ms                Output file (optional, writes to stdout if not specified)")
        print()
        print("Options:")
        print("  -h, --help               Show this help message and exit")
        print("  --top-n N                Only output the N best polynomials (sorted by alpha)")
        print("  --msieve-format          Output multi-line format for msieve -npr (includes n: header)")
        print("  --cleanup                Backup outMsieve.p and msieve.dat.ms, then delete intermediate files")
        print()
        print("Examples:")
        print("  cado_to_msieve.py cado_output.ms output.ms")
        print("  cado_to_msieve.py cado_output.ms --top-n 10 --msieve-format")
        print("  cado_to_msieve.py --cleanup")
        sys.exit(0)

    # Check for --cleanup flag first (can be used standalone)
    if '--cleanup' in sys.argv:
        cleanup_old_run()
        if len(sys.argv) == 2:  # Only --cleanup was specified
            sys.exit(0)
        # Otherwise continue with normal operation

    if len(sys.argv) < 2 or (len(sys.argv) == 2 and sys.argv[1].startswith('--')):
        print("Usage: cado_to_msieve.py <cado_output.ms> [output.ms] [--top-n N] [--msieve-format] [--cleanup]")
        print("If output.ms not specified, writes to stdout")
        print("--top-n N: Only output the N best polynomials (sorted by alpha)")
        print("--msieve-format: Output multi-line format for msieve -npr (includes n: header)")
        print("--cleanup: Backup outMsieve.p and msieve.dat.ms, then delete intermediate files")
        print()
        print("Use -h or --help for more information")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None

    # Parse --top-n option
    top_n = None
    msieve_format = '--msieve-format' in sys.argv
    for i, arg in enumerate(sys.argv):
        if arg == '--top-n' and i + 1 < len(sys.argv):
            try:
                top_n = int(sys.argv[i + 1])
            except ValueError:
                print(f"Error: --top-n requires an integer argument", file=sys.stderr)
                sys.exit(1)

    # Parse CADO output
    polynomials = parse_cado_output(input_file)

    if not polynomials:
        print("Error: No size-optimized polynomials found in input", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(polynomials)} polynomials", file=sys.stderr)

    # Sort by projective alpha (ascending = better/more negative alpha first)
    polynomials.sort(key=lambda p: float(p['proj_alpha']))

    # Apply top-n filter if requested
    if top_n and top_n < len(polynomials):
        print(f"Filtering to top {top_n} of {len(polynomials)} polynomials", file=sys.stderr)
        polynomials = polynomials[:top_n]

    # Convert to msieve format
    if msieve_format:
        # Multi-line format
        msieve_output = '\n'.join([poly_to_msieve_format(p, include_n=True) for p in polynomials])
    else:
        # Single line format
        msieve_lines = [poly_to_msieve_format(p, include_n=False) for p in polynomials]
        msieve_output = '\n'.join(msieve_lines)

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(msieve_output)
            if not msieve_output.endswith('\n'):
                f.write('\n')
        print(f"Wrote {len(polynomials)} polynomials to {output_file}", file=sys.stderr)
        print(f"Best alpha: {polynomials[0]['proj_alpha']}", file=sys.stderr)
        print(f"Worst alpha: {polynomials[-1]['proj_alpha']}", file=sys.stderr)

        # Report exp_E range if available
        if 'exp_E' in polynomials[0]:
            # Sort by exp_E to get true best/worst
            exp_e_sorted = sorted(polynomials, key=lambda p: float(p.get('exp_E', '0')))
            print(f"Best exp_E: {exp_e_sorted[0]['exp_E']}", file=sys.stderr)
            print(f"Worst exp_E: {exp_e_sorted[-1]['exp_E']}", file=sys.stderr)
    else:
        print(msieve_output)

if __name__ == '__main__':
    main()
