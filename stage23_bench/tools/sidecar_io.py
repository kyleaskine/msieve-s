"""Stream plain/zstd sidecars, including archived byte-range extracts."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'stage23_bench/sidecar_archives.json'


@contextmanager
def binary_stream(path):
    path = Path(path)
    if path.suffix != '.zst':
        with path.open('rb') as stream:
            yield stream
        return
    with subprocess.Popen(['zstd', '-q', '-d', '-c', '--', str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            yield process.stdout
        except BaseException:
            process.terminate()
            process.communicate()
            raise
        else:
            # Drain even after a requested range ends, checking the entire frame.
            while process.stdout.read(1024*1024):
                pass
            error = process.stderr.read()
            if process.wait():
                raise OSError(f'zstd failed for {path}: {error.decode(errors="replace")}')


def resolve(path):
    path = Path(path).resolve()
    if path.exists():
        return path, None
    compressed = Path(str(path) + '.zst')
    if compressed.exists():
        return compressed, None
    if REGISTRY.exists():
        try:
            key = str(path.relative_to(ROOT))
        except ValueError:
            key = str(path)
        row = json.loads(REGISTRY.read_text()).get('extracts', {}).get(key)
        if row:
            return ROOT / row['archive'], row['ranges']
    raise FileNotFoundError(path)


def chunks(path):
    source, ranges = resolve(path)
    with binary_stream(source) as stream:
        position = 0
        if ranges is None:
            while data := stream.read(1024*1024):
                yield data
        else:
            for start, end in ranges:
                if start < position or end < start:
                    raise ValueError('archive ranges must be ordered and nonoverlapping')
                while position < start:
                    data = stream.read(min(1024*1024, start-position))
                    if not data:
                        raise EOFError('archive ended before range start')
                    position += len(data)
                while position < end:
                    data = stream.read(min(1024*1024, end-position))
                    if not data:
                        raise EOFError('archive ended before range end')
                    position += len(data)
                    yield data


def sidecar_lines(path):
    """Yield UTF-8 lines with exact original newline/tail semantics."""
    pending = b''
    for data in chunks(path):
        lines = (pending + data).split(b'\n')
        pending = lines.pop()
        for line in lines:
            yield (line + b'\n').decode('utf-8')
    if pending:
        yield pending.decode('utf-8')


@contextmanager
def open_sidecar(path):
    # Closing the generator also closes any decompressor on parser errors.
    lines = sidecar_lines(path)
    try:
        yield lines
    finally:
        lines.close()


if __name__ == '__main__':
    import argparse
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='original path, .zst archive, or registered extract path')
    parser.add_argument('--output', help='optional new destination; without it only print the SHA-256')
    args = parser.parse_args()
    if args.output and Path(args.output).resolve() == Path(args.source).resolve():
        parser.error('choose a different output path to avoid shadowing an archive alias')
    from contextlib import nullcontext
    state = hashlib.sha256()
    size = 0
    with open(args.output, 'xb') if args.output else nullcontext(None) as output:
        for data in chunks(args.source):
            state.update(data)
            size += len(data)
            if output: output.write(data)
    print(json.dumps(dict(source=args.source, bytes=size, sha256=state.hexdigest())))
