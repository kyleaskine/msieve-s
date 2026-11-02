#!/usr/bin/env python3
"""Summarize warm jobs from an Nsight SQLite export of bench_murphy.py.

Host range totals sum elapsed time across workers, including waits. Nested range
totals overlap; self time subtracts direct children on the same thread. GPU busy
time is the union of kernel/copy intervals, not summed concurrent kernel time.
This is attribution for a saved-pool replay, not a full-pipeline speed estimate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from gpu_pipeline import digest, write_json


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(a, b):
    a, b = merged(a), merged(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def benchmark_layout(db, total_jobs, warmup_jobs=None, cases_per_pass=None, benchmark_result=None):
    """Infer the pass width from the benchmark, never from divisibility alone."""
    source = benchmark_result if benchmark_result is not None else db.parent / 'result.json'
    if source.exists():
        record = json.loads(source.read_text())
        if record.get('returncode', 0) != 0:
            raise ValueError('benchmark did not complete successfully')
        if 'jobs' in record:
            groups = defaultdict(list)
            for job in record['jobs']:
                groups[job['repetition']].append(job['tag'])
            tags = groups.get(-1, [])
            if (not tags or len(set(tags)) != len(tags) or
                    list(groups) != [-1, *range(len(groups)-1)] or
                    any(group != tags for group in groups.values())):
                raise ValueError('benchmark job order/case layout is inconsistent')
            expected_width, expected_warmup = len(tags), len(tags)
            expected_jobs = len(record['jobs'])
            if ('cases_per_pass' in record and record['cases_per_pass'] != expected_width) or (
                    'warmup_jobs' in record and record['warmup_jobs'] != expected_warmup):
                raise ValueError('benchmark layout metadata disagrees with its jobs')
        else:
            expected_width, expected_warmup = record['cases_per_pass'], record['warmup_jobs']
            expected_jobs = expected_warmup + expected_width * record['repeats']
        if total_jobs != expected_jobs:
            raise ValueError('trace job count disagrees with benchmark result')
        if ((warmup_jobs is not None and warmup_jobs != expected_warmup) or
                (cases_per_pass is not None and cases_per_pass != expected_width)):
            raise ValueError('explicit warmup/case counts disagree with benchmark result')
        warmup_jobs, cases_per_pass = expected_warmup, expected_width
        metadata = dict(path=str(source), sha256=digest(source))
    else:
        if benchmark_result is not None:
            raise ValueError('benchmark result file does not exist')
        if warmup_jobs is None or cases_per_pass is None:
            raise ValueError('no benchmark result.json; provide both --warmup-jobs and --cases-per-pass')
        metadata = None
    if warmup_jobs < 0 or cases_per_pass < 1:
        raise ValueError('need nonnegative warmup jobs and positive cases per pass')
    return warmup_jobs, cases_per_pass, metadata


def summarize(db, warmup_jobs=None, cases_per_pass=None, benchmark_result=None):
    c = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    try:
        return _summarize(c, db, warmup_jobs, cases_per_pass, benchmark_result)
    finally:
        c.close()


def _summarize(c, db, warmup_jobs, cases_per_pass, benchmark_result):
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'NVTX_EVENTS' not in tables:
        raise ValueError('no NVTX ranges; build with ROPT_PROFILE=1 and trace nvtx')
    jobs = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='ropt_job' ORDER BY start").fetchall()
    if any(job['end'] is None or job['end'] < job['start'] for job in jobs):
        raise ValueError('trace contains an incomplete or invalid job range')
    warmup_jobs, cases_per_pass, metadata = benchmark_layout(
        db, len(jobs), warmup_jobs, cases_per_pass, benchmark_result)
    jobs = jobs[warmup_jobs:]
    if not jobs or len(jobs) % cases_per_pass:
        raise ValueError('missing ropt_job ranges or incomplete measured passes; build with ROPT_PROFILE=1')
    passes = len(jobs) // cases_per_pass
    totals = defaultdict(lambda: dict(count=0, total_ns=0, self_ns=0))
    kernels, copies, apis = (defaultdict(lambda: dict(count=0, ns=0, work=0)) for _ in range(3))
    job_records = []
    for job in jobs:
        lo, hi = job['start'], job['end']
        rows = [dict(r) for r in c.execute(
            'SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE start>=? AND end<=? '
            'ORDER BY globalTid,start,end DESC', (lo, hi))]
        stack, tid = [], None
        for r in rows:
            r['self_ns'] = r['end'] - r['start']
            if r['globalTid'] != tid:
                stack, tid = [], r['globalTid']
            while stack and r['start'] >= stack[-1]['end']:
                stack.pop()
            if stack:
                if r['end'] > stack[-1]['end']:
                    raise ValueError('overlapping non-nested NVTX ranges')
                stack[-1]['self_ns'] -= r['self_ns']
            stack.append(r)
        for r in rows:
            item = totals[r['text']]
            item['count'] += 1
            item['total_ns'] += r['end'] - r['start']
            item['self_ns'] += r['self_ns']
        gpu = []
        for table, group, name, work in [
            ('KERNEL', kernels, 's.value', 'k.gridY'),
            ('MEMCPY', copies, 'CAST(k.copyKind AS TEXT)', 'k.bytes'),
            ('RUNTIME', apis, 's.value', '0'),
        ]:
            if f'CUPTI_ACTIVITY_KIND_{table}' not in tables:
                continue
            join = (f'JOIN StringIds s ON s.id=k.{"shortName" if table == "KERNEL" else "nameId"}'
                    if table != 'MEMCPY' else '')
            for r in c.execute(f'SELECT k.start,k.end,{name} AS name,{work} AS work '
                               f'FROM CUPTI_ACTIVITY_KIND_{table} k {join} '
                               'WHERE k.start>=? AND k.end<=?', (lo, hi)):
                item = group[r['name']]
                item['count'] += 1
                item['ns'] += r['end'] - r['start']
                item['work'] += r['work']
                if table != 'RUNTIME':
                    gpu.append((r['start'], r['end']))
        evaluation = [(r['start'], r['end']) for r in rows if r['text'] == 'murphy_gpu_evaluate']
        murphy = [(r['start'], r['end']) for r in rows if r['text'] == 'murphy_phase']
        has_gpu_activity = bool({'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY'} & tables)
        job_records.append(dict(
            job_ms=(hi-lo)/1e6, gpu_busy_ms=duration(gpu)/1e6 if has_gpu_activity else None,
            murphy_phase_ms=duration(murphy)/1e6,
            murphy_gpu_busy_ms=intersection(gpu, murphy)/1e6 if has_gpu_activity else None,
            evaluator_active_gpu_idle_ms=(duration(evaluation)-intersection(gpu, evaluation))/1e6
                if has_gpu_activity else None))
    scale = 1e6 * passes
    result = dict(sqlite_sha256=digest(db), warmup_jobs=warmup_jobs, measured_jobs=len(jobs), passes=passes,
                  cases_per_pass=cases_per_pass,
                  benchmark_result=metadata,
                  timing_clock='Nsight timestamps; not independently calibrated to host wall time',
                  cuda_tables_present=sorted(t for t in tables if t.startswith('CUPTI_ACTIVITY_KIND_')),
                  units='Milliseconds per measured pass; counts and work are per pass.',
                  caveats=__doc__.strip(),
                  mean_pass={key: sum(r[key] for r in job_records)/passes if job_records[0][key] is not None
                             else None for key in job_records[0]},
                  median_job_pass_ms=statistics.median(
                      sum(r['job_ms'] for r in job_records[i:i+cases_per_pass])
                      for i in range(0, len(job_records), cases_per_pass)),
                  host_ranges={key: dict(count=v['count']/passes, total_ms=v['total_ns']/scale,
                                         self_ms=v['self_ns']/scale) for key, v in sorted(totals.items())})
    for key, group in [('kernels', kernels), ('copies', copies), ('cuda_apis', apis)]:
        result[key] = {name: dict(count=v['count']/passes, total_ms=v['ns']/scale,
                                 work=v['work']/passes) for name, v in sorted(group.items())}
    result['work_units'] = 'kernel gridY (MurphyE integrals); copy bytes; API work=0; copyKind 1=H2D, 2=D2H'
    if {'COMPOSITE_EVENTS', 'SAMPLING_CALLCHAINS'} <= tables:
        samples = []
        for lo, hi in jobs:
            samples.extend(c.execute(
                'SELECT s.value AS symbol,m.value AS module FROM SAMPLING_CALLCHAINS a '
                'JOIN COMPOSITE_EVENTS e ON e.id=a.id JOIN StringIds s ON s.id=a.symbol '
                'JOIN StringIds m ON m.id=a.module WHERE a.stackDepth=0 AND e.start>=? AND e.start<?',
                (lo, hi)).fetchall())
        modules, symbols = defaultdict(int), defaultdict(int)
        for r in samples:
            modules[r['module']] += 1
            symbols[(r['symbol'], r['module'])] += 1
        result['cpu_samples'] = dict(
            leaf_count=len(samples),
            modules=dict(sorted(modules.items(), key=lambda kv: -kv[1])),
            top_symbols=[dict(symbol=k[0], module=k[1], count=v) for k, v in
                         sorted(symbols.items(), key=lambda kv: -kv[1])[:25]],
            caveat='Leaf samples across all process threads; includes driver busy-waiting, not GPU execution time.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--benchmark-result', type=Path, help='default: result.json beside the SQLite file')
    parser.add_argument('--warmup-jobs', type=int, help='required without benchmark metadata')
    parser.add_argument('--cases-per-pass', type=int, help='required without benchmark metadata')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if ((args.warmup_jobs is not None and args.warmup_jobs < 0) or
            (args.cases_per_pass is not None and args.cases_per_pass < 1)):
        parser.error('need nonnegative warmup jobs and positive cases per pass')
    if args.output.exists():
        parser.error('output already exists')
    try:
        result = summarize(args.sqlite, args.warmup_jobs, args.cases_per_pass, args.benchmark_result)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    write_json(args.output, result)
    print(json.dumps(result['mean_pass'], indent=2))


if __name__ == '__main__':
    main()
