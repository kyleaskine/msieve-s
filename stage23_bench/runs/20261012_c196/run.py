#!/usr/bin/env python3
"""Run the C196 GPU sopt/breadth experiment from a frozen input and binaries.

Usage: run.py RUN_DIRECTORY
The launcher creates input.ms, bin/, cpu/best300_msieve.ms, and settings.json.
Outputs live outside pipeline_results so a later CPU run cannot move them.
Each command, return code and wall time is saved; a failure stops the pipeline.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
from content_seeds import content_seeds
from polyfmt import cado_poly_text, read_cado_blocks, resultant


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    settings = json.loads((run / 'settings.json').read_text())
    records = []

    def state(stage, **extra):
        tmp = run / 'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, updated=time.time(), **extra), indent=2) + '\n')
        tmp.replace(run / 'status.json')
        print(stage, extra, flush=True)

    def command(name, args, output=None):
        start = time.time()
        with (run / (output or f'{name}.log')).open('w') as out:
            if output:
                with (run / f'{name}.log').open('w') as err:
                    result = subprocess.run(list(map(str, args)), cwd=ROOT, stdout=out, stderr=err)
            else:
                result = subprocess.run(list(map(str, args)), cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        record = dict(name=name, command=list(map(str, args)), returncode=result.returncode,
                      started=start, elapsed_seconds=time.time() - start)
        # Concurrent breadth workers each have a separate record file.
        (run / f'{name}.json').write_text(json.dumps(record, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {name}.log')
        return record

    try:
        state('sopt effort 0')
        records.append(command('sopt/e0', [run / 'bin/s23_sopt', '-sopteffort', 0,
            '-t', 8, '-inputpolys', run / 'input.ms'], 'sopt/e0_unsorted.txt'))
        state('selecting effort-50 inputs')
        records.append(command('sopt/sort_e0', [sys.executable, ROOT / 'utils/sort_cado_by_expe.py',
            run / 'sopt/e0_unsorted.txt', run / 'sopt/e0_sorted.txt']))
        records.append(command('sopt/extract', [sys.executable, ROOT / 'utils/extract_input_polys_sorted.py',
            run / 'sopt/e0_sorted.txt', run / 'sopt/e0_unsorted.txt', run / 'sopt/top_input.ms',
            settings['top_n']]))
        state('sopt effort 50')
        records.append(command('sopt/e50', [run / 'bin/s23_sopt', '-sopteffort', 50,
            '-t', 8, '-inputpolys', run / 'sopt/top_input.ms'], 'sopt/e50_output.txt'))
        records.append(command('sopt/convert', [sys.executable, ROOT / 'utils/cado_to_msieve.py',
            run / 'sopt/e50_output.txt', run / 'sopt/e50_msieve.ms']))
        # Match the pipeline's numeric sort and default whole-line tie breaker.
        records.append(command('sopt/sort_e50', ['sort', '-k10,10n', run / 'sopt/e50_msieve.ms'],
                               'sopt/e50_msieve_sorted.ms'))
        lines = (run / 'sopt/e50_msieve_sorted.ms').read_text().splitlines()[:300]
        if len(lines) != 300:
            raise RuntimeError(f'expected 300 seeds, got {len(lines)}')
        top = run / 'sopt/best300_msieve.ms'
        top.write_text('\n'.join(lines) + '\n')
        seeds = []
        n = settings['n']
        for rank, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 11:
                raise RuntimeError('malformed msieve seed')
            p = dict(n=n, Y1=int(fields[6]), Y0=int(fields[7]),
                     **{f'c{i}': int(fields[5-i]) for i in range(6)})
            if not resultant(p) or resultant(p) % n:
                raise RuntimeError(f'invalid seed {rank}')
            stem = f's{rank:04d}'
            (run / f'seeds/{stem}.poly').write_text(cado_poly_text(p))
            seeds.append(dict(search=stem, seed_rank=rank, content=1, exp_E=float(fields[9])))
            for d, u, v, q in content_seeds(p, n):
                name = f'{stem}_d{d}'
                (run / f'seeds/{name}.poly').write_text(cado_poly_text(q))
                seeds.append(dict(search=name, seed_rank=rank, content=d, u=u, v=v, exp_E=float(fields[9])))
        (run / 'seeds.json').write_text(json.dumps(seeds, indent=2) + '\n')
        (run / 'todo_breadth.txt').write_text(''.join(s['search'] + '\n' for s in seeds))
        manifest = dict(n=n, source=str(run / 'cpu'), seed_source=str(top), searches=len(seeds),
                        seed_numbering='one-based GPU effort-50 rank', settings=settings,
                        s23_ropt_sha256=sha(run / 'bin/s23_ropt'), s23_sopt_sha256=sha(run / 'bin/s23_sopt'),
                        same_top300_as_cpu=top.read_bytes() == (run / 'cpu/best300_msieve.ms').read_bytes())
        (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        state('breadth root optimization', completed=0, searches=len(seeds))

        def search(seed):
            name = seed['search']
            dst = run / f'breadth/out/{name}.poly'
            result = command(f'breadth/out/{name}', [run / 'bin/s23_ropt', '-poly', run / f'seeds/{name}.poly',
                '-search', '-budget', 3, '-aw', 1.3, '-maxcells', '2e11', '-rerank', 4096,
                '-top', 3, '-t', 8, '-launch', 30, '-out', dst])
            blocks = read_cado_blocks(dst)
            if not blocks or any(p['n'] != n or not resultant(p) or resultant(p) % n for _, p in blocks):
                raise RuntimeError(f'{name}: invalid or empty polynomial export')
            return result

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for completed, record in enumerate(pool.map(search, seeds), 1):
                records.append(record)
                state('breadth root optimization', completed=completed, searches=len(seeds))
        (run / 'timings.json').write_text(json.dumps(dict(commands=records,
            breadth_seconds=time.time() - start, clock='time.time; WSL wall clock may step'), indent=2) + '\n')
        state('GPU search complete; ready for comparison', searches=len(seeds),
              same_top300_as_cpu=manifest['same_top300_as_cpu'])
    except BaseException as error:
        state('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
