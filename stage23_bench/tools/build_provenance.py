#!/usr/bin/env python3
"""Hash manifests recording what a stage23_gpu binary was built from.

A manifest, BINARY.provenance.json, holds the binary's SHA-256, the commands
`make -n -B` prints for it, the make variables it was recorded with and the
SHA-256 of every file those commands read: the sources and headers the
compilers report (-M) and the linked libraries. `verify` reruns make and the
compilers' dependency listing with the recorded variables. It reports a binary
whose bytes, inputs, list of inputs or build commands no longer match. Make's
timestamps cannot do this reliably: the ROPT rules always look stale to a dry
run, and copying, checkout or touch changes timestamps freely.

  build_provenance.py record BINARY...    after `make`, for a build that is current
  build_provenance.py import BINARY --validation JSON --expect-sha256 HASH [--map OLD=NEW]
  build_provenance.py verify BINARY...

`record` and `import` take the make variables the Makefile reads (CUDA, NVCC,
CXX, CADO_BUILD_DIR, ROPT_PROFILE, NVTX_INCLUDE) from the environment, so set
them as for make (`CUDA=89 build_provenance.py record ...` after
`make CUDA=89 ...`). `verify` ignores the caller's environment and uses the
recorded values.

`record` refuses a binary older than any input or the Makefile, one whose flag
stamp differs from the current configuration (ROPT), and bytes an existing
manifest recorded under other build commands. The SOPT rules have no flag stamp,
so make cannot see their flag or variable changes. Without an earlier manifest
`record` cannot tell how such a binary was built, so run it right after `make`.
`import` instead accepts a validated artifact whose recorded input hashes all
match the current files. Paths inside the repository are stored relative to it;
external ones (CADO, CUDA, system headers) are absolute, so a moved CADO or
toolchain needs a new record.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAKE_DIR = ROOT / 'stage23_gpu'
FORMAT = 1
# gcc, g++-13, x86_64-linux-gnu-gcc, clang++-17, nvcc, ...; launchers such as ccache are skipped.
COMPILER = re.compile(r'(?:[\w.]+-)*(?:gcc|g\+\+|cc|c\+\+|clang|clang\+\+|nvcc)(?:-[\d.]+)?')
LAUNCHERS = ('ccache', 'sccache', 'distcc')
SOURCES = ('.c', '.cc', '.cpp', '.cxx', '.cu')
TOOLCHAIN = ('/usr/include/', '/usr/lib/', '/usr/local/cuda')  # system headers (after realpath)
# Variables the stage23_gpu Makefile reads from the environment, and make's own
# controls, which could carry command-line overrides from a parent make.
MAKE_VARIABLES = ('CUDA', 'NVCC', 'CXX', 'CADO_BUILD_DIR', 'ROPT_PROFILE', 'NVTX_INCLUDE')
MAKE_CONTROL = ('MAKEFLAGS', 'MFLAGS', 'GNUMAKEFLAGS', 'MAKELEVEL', 'MAKEFILES', 'MAKEOVERRIDES')


class ProvenanceError(Exception):
    pass


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def manifest_path(binary):
    binary = Path(binary)
    return binary.with_name(binary.name + '.provenance.json')


def stored(path):
    """Repository paths relative to the repository, others absolute."""
    path = Path(os.path.normpath(path))
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path):
    return Path(os.path.normpath(ROOT / path))


def make_target(binary, make_dir=MAKE_DIR):
    binary = Path(os.path.normpath(Path(binary).resolve()))
    try:
        rel = binary.relative_to(make_dir)
    except ValueError:
        raise ProvenanceError(f'{binary} is not a build output under {make_dir}') from None
    return [f'BUILD={rel.parent}', str(rel)]


def environment_variables():
    """The make variables set in this process's environment, as make would read them."""
    return {k: os.environ[k] for k in MAKE_VARIABLES if k in os.environ}


def build_env(variables):
    """This environment without make variables or controls, plus the given variables."""
    env = {k: v for k, v in os.environ.items() if k not in MAKE_VARIABLES + MAKE_CONTROL}
    return dict(env, **variables, LC_ALL='C')


def run(argv, make_dir, variables):
    try:
        return subprocess.run(argv, cwd=make_dir, capture_output=True, text=True, env=build_env(variables))
    except OSError as error:
        raise ProvenanceError(f'cannot run {argv[0]}: {error}') from None


def build_commands(binary, make_dir=MAKE_DIR, variables=None):
    """Commands a full rebuild would run, one argument list per shell command."""
    proc = run(['make', '--no-print-directory', '-n', '-B', *make_target(binary, make_dir)], make_dir, variables or {})
    if proc.returncode:
        raise ProvenanceError('make -n failed: ' + (proc.stderr.strip().splitlines() or ['?'])[-1])
    return [tokens(line) for line in proc.stdout.replace('\\\n', ' ').splitlines() if line.strip()]


def tokens(line):
    """Shell words, with operators such as ; && > as separate tokens."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def simple_commands(commands):
    """Each recipe line split at ; && || | &, so `gcc ... && ar ...` yields both."""
    for argv in commands:
        current = []
        for token in argv + [';']:
            if token in (';', '&&', '||', '|', '&'):
                if current:
                    yield current
                current = []
            else:
                current.append(token)


def is_compiler_name(name):
    return COMPILER.fullmatch(Path(name).name) is not None


def compiler_command(argv):
    """The compiler invocation in a command, without a launcher such as ccache, or None."""
    while argv and Path(argv[0]).name in LAUNCHERS:
        argv = argv[1:]
    return argv if argv and is_compiler_name(argv[0]) else None


def output(argv):
    return argv[argv.index('-o') + 1] if '-o' in argv[:-1] else None


def dependency_command(argv):
    """The compile command with its outputs and link inputs removed, plus -M."""
    out, i = [argv[0]], 1
    while i < len(argv):
        a = argv[i]
        if a == '-o' or (a == '-x' and argv[i + 1:i + 2] == ['none']):
            i += 2
            continue
        if a != '-c' and not a.startswith(('-l', '-L', '-Wl,')) and not a.endswith(('.o', '.a', '.so')):
            out.append(a)
        i += 1
    return out + ['-M']


def parse_dependencies(text):
    deps = []
    for line in text.replace('\\\n', ' ').splitlines():
        if ':' in line:
            deps += line.split(':', 1)[1].split()
    return deps


def flag_stamps(commands):
    """{file: content} for the Makefile's printf flag stamps (build/ropt-build-flags)."""
    stamps = {}
    for argv in simple_commands(commands):
        if argv[:2] == ['printf', '%s\\n'] and '>' in argv:
            target = argv[argv.index('>') + 1]
            stamps[target.removesuffix('.tmp')] = ''.join(a + '\n' for a in argv[2:argv.index('>')])
    return stamps


def collect(binary, make_dir=MAKE_DIR, variables=None):
    """(commands, dependency paths, intermediate objects, flag stamps) for a build output."""
    binary = Path(binary).resolve()
    variables = variables or {}
    commands = build_commands(binary, make_dir, variables)
    compiles = []
    for argv in simple_commands(commands):
        compiler = compiler_command(argv)
        if compiler:
            compiles.append(compiler)
        elif output(argv) or any(a.endswith(SOURCES) for a in argv):
            # An unrecognized compiler would leave its sources and headers untracked.
            raise ProvenanceError(f'cannot list the inputs of build command {argv[0]!r}: not a recognized '
                                  'compiler (gcc, g++, cc, c++, clang, clang++ or nvcc, optionally versioned)')
    if not compiles:
        raise ProvenanceError(f'make prints no compiler command for {binary}')
    produced = {os.path.realpath(make_dir / out) for out in map(output, compiles) if out}
    if str(binary) not in produced:
        raise ProvenanceError(f'no make command writes {binary}')
    deps = set()
    for argv in compiles:
        proc = run(dependency_command(argv), make_dir, variables)
        if proc.returncode:
            raise ProvenanceError(f'{Path(argv[0]).name} -M failed: ' + proc.stderr.strip()[-300:])
        deps.update(os.path.realpath(make_dir / d) for d in parse_dependencies(proc.stdout))
        deps.update(os.path.realpath(make_dir / a) for a in argv[1:] if a.endswith(('.a', '.so')))
    stamps = {os.path.realpath(make_dir / k): v for k, v in flag_stamps(commands).items()}
    deps.update(stamps)
    intermediates = produced - {str(binary)}
    return commands, sorted(deps - intermediates), sorted(intermediates), stamps


def check_stamps(stamps):
    for path, expected in stamps.items():
        if not Path(path).is_file() or Path(path).read_text() != expected:
            raise ProvenanceError(f'{stored(path)} does not match the current build flags; rebuild with make, '
                                  'and record with the same make variables (CUDA=... in the environment)')


def write(binary, commands, dependencies, origin, variables):
    binary = Path(binary)
    manifest = dict(format=FORMAT, binary=stored(binary.resolve()), sha256=sha256(binary),
                    recorded=time.strftime('%Y-%m-%dT%H:%M:%S%z'), origin=origin,
                    make_variables=variables, commands=commands, dependencies=dependencies)
    path = manifest_path(binary)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest, indent=1) + '\n')
    tmp.replace(path)
    return path


def record(binary, make_dir=MAKE_DIR, variables=None):
    """Manifest for a binary make just built: refuse inputs newer than it, and
    bytes an earlier manifest recorded under other commands."""
    binary = Path(binary).resolve()
    variables = environment_variables() if variables is None else variables
    commands, deps, intermediates, stamps = collect(binary, make_dir, variables)
    check_stamps(stamps)
    built = binary.stat().st_mtime_ns
    # Rules without a Makefile dependency (SOPT) are not rebuilt when a recipe changes.
    newer = [p for p in deps + intermediates + [str(make_dir / 'Makefile')] if Path(p).stat().st_mtime_ns > built]
    if newer:
        raise ProvenanceError(f'{stored(binary)} is older than {len(newer)} input(s), e.g. {stored(newer[0])}; '
                              f'rebuild it before recording (make -B {make_target(binary, make_dir)[1]} if make '
                              'reports it up to date, as for the Makefile or system headers)')
    previous = manifest_path(binary)
    if previous.is_file():
        try:
            old = json.loads(previous.read_text())
        except ValueError:
            old = {}
        if (isinstance(old, dict) and old.get('sha256') == sha256(binary) and old.get('commands') != commands
                and built <= previous.stat().st_mtime_ns):
            raise ProvenanceError(f'{stored(previous)} records these same bytes with other build commands, and '
                                  'the binary has not been rebuilt since. make cannot see every flag or variable '
                                  'change: delete the binary, run make, then record')
    hashes = {stored(p): sha256(p) for p in deps}
    return write(binary, commands, hashes, dict(kind='make'), variables)


def import_validation(binary, validation, expect_sha256, path_map=(), make_dir=MAKE_DIR, variables=None):
    """Manifest for a validated artifact: every current input must have the hash
    the validation recorded (after mapping its source paths onto this tree)."""
    binary = Path(binary).resolve()
    variables = environment_variables() if variables is None else variables
    actual = sha256(binary)
    if actual != expect_sha256:
        raise ProvenanceError(f'{stored(binary)} has SHA-256 {actual[:12]}, not the validated {expect_sha256[:12]}')
    data = json.loads(Path(validation).read_text())
    recorded = data.get('dependencies_sha256') or data.get('dependencies')
    if not isinstance(recorded, dict) or not recorded:
        raise ProvenanceError(f'{validation} records no dependency hashes')
    mapped = {}
    for path, digest in recorded.items():
        for old, new in path_map:
            if path.startswith(old):
                path = new + path[len(old):]
                break
        mapped[os.path.realpath(make_dir / path)] = digest
    commands, deps, _, stamps = collect(binary, make_dir, variables)
    check_stamps(stamps)
    # Validations that listed host dependencies with -MM omitted system headers.
    # Accept those only from an unchanged toolchain, and list them in the manifest.
    unrecorded = [p for p in deps if p not in mapped and p not in stamps]
    if any(not p.startswith(TOOLCHAIN) for p in unrecorded):
        p = next(p for p in unrecorded if not p.startswith(TOOLCHAIN))
        raise ProvenanceError(f'{len(unrecorded)} current input(s) absent from the validation, e.g. {stored(p)}')
    compilers = check_toolchain(data) if unrecorded else []
    hashes = {stored(p): sha256(p) for p in deps}
    changed = [p for p in deps if p in mapped and hashes[stored(p)] != mapped[p]]
    if changed:
        raise ProvenanceError(f'{len(changed)} input(s) changed since validation, e.g. {stored(changed[0])}')
    origin = dict(kind='validation', file=stored(Path(validation).resolve()), sha256=sha256(validation),
                  map=[list(m) for m in path_map])
    if unrecorded:
        origin.update(unvalidated_toolchain_inputs=unrecorded, toolchain_versions_matched=compilers)
    if isinstance(data.get('commands'), list):
        origin['commands'] = data['commands']
    return write(binary, commands, hashes, origin, variables)


def check_toolchain(data):
    """Compare the current compilers with the versions a validation recorded."""
    recorded = data.get('compilers') or data.get('tools') or {}
    checked = []
    for name, text in recorded.items():
        if not is_compiler_name(name):
            continue
        proc = subprocess.run([name, '--version'], capture_output=True, text=True)
        if proc.returncode or proc.stdout.strip() != text.strip():
            raise ProvenanceError(f'{name} differs from the version the validation recorded')
        checked.append(name)
    if not checked:
        raise ProvenanceError('the validation records no compiler versions to compare')
    return checked


def listed(paths, limit=5):
    paths = sorted(paths)
    return ', '.join(paths[:limit]) + (', ...' if len(paths) > limit else '')


def verify(binary, make_dir=MAKE_DIR):
    """{verified, problems, sha256, manifest, ...}; never raises for a mismatch,
    an unreadable manifest or a missing make or compiler."""
    binary = Path(binary).resolve()
    path = manifest_path(binary)
    problems: list[str] = []
    result: dict = dict(verified=False, manifest=stored(path), problems=problems)
    if not path.is_file():
        problems.append(f'no provenance manifest {stored(path)}')
        return result
    try:
        result['manifest_sha256'] = sha256(path)
        manifest = json.loads(path.read_text())
        if manifest.get('format') != FORMAT:
            problems.append(f'unsupported manifest format {manifest.get("format")!r}')
            return result
        result.update(origin=manifest['origin'].get('kind'), recorded=manifest['recorded'],
                      dependencies=len(manifest['dependencies']), sha256=manifest['sha256'])
        recorded, commands = dict(manifest['dependencies']), list(manifest['commands'])
        variables = {k: str(v) for k, v in dict(manifest.get('make_variables', {})).items() if k in MAKE_VARIABLES}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        problems.append(f'unreadable manifest {stored(path)}: {error!r}')
        return result
    if not binary.is_file() or sha256(binary) != result['sha256']:
        problems.append(f'{stored(binary)} differs from the binary its manifest describes')
    changed = [p for p, h in recorded.items() if not resolve(p).is_file() or sha256(resolve(p)) != h]
    if changed:
        problems.append(f'{len(changed)} build input(s) changed since it was built: ' + listed(changed))
    # Rerun make and the compilers' dependency listing: an #include can resolve to
    # a different file although every recorded file is unchanged.
    try:
        current_commands, deps, _, _ = collect(binary, make_dir, variables)
    except ProvenanceError as error:
        problems.append(str(error))
    else:
        if current_commands != commands:
            problems.append('make now builds it with different commands')
        current = {stored(p) for p in deps}
        if current != set(recorded):
            problems.append('the build now reads different inputs: ' +
                            '; '.join(f'{label} {listed(paths)}' for label, paths in
                                      (('added', current - set(recorded)), ('dropped', set(recorded) - current))
                                      if paths))
    result['verified'] = not problems
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('record', 'verify'):
        sub.add_parser(name).add_argument('binary', type=Path, nargs='+')
    imp = sub.add_parser('import')
    imp.add_argument('binary', type=Path)
    imp.add_argument('--validation', type=Path, required=True, help="JSON with 'dependencies_sha256' or 'dependencies'")
    imp.add_argument('--expect-sha256', required=True, help='the validated binary hash')
    imp.add_argument('--map', action='append', default=[], metavar='OLD=NEW',
                     help='replace a recorded source path prefix (repeatable)')
    args = ap.parse_args(argv)
    status = 0
    try:
        if args.command == 'record':
            for binary in args.binary:
                print('wrote', stored(record(binary)))
        elif args.command == 'import':
            path_map = [tuple(m.split('=', 1)) for m in args.map]
            print('wrote', stored(import_validation(args.binary, args.validation, args.expect_sha256, path_map)))
        else:
            for binary in args.binary:
                result = verify(binary)
                print(stored(Path(binary).resolve()), 'verified' if result['verified'] else 'NOT verified')
                for problem in result['problems']:
                    print('  ' + problem)
                status |= not result['verified']
    except ProvenanceError as error:
        print(f'build_provenance: {error}', file=sys.stderr)
        return 1
    return status


if __name__ == '__main__':
    sys.exit(main())
