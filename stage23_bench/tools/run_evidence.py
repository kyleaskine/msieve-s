"""Read historical evidence from plain files, gzip tables or the local archive.

Archival mappings and original hashes live in data/historical_evidence.json.
This helper only reads; it never restores or creates large files in the checkout.
"""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'stage23_bench/data/historical_evidence.json'


def evidence_path(path):
    path = Path(path).resolve()
    if path.is_file():
        return path
    compressed = path.with_name(path.name + '.gz')
    if compressed.is_file():
        return compressed
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError:
        relative = None
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    if relative in manifest.get('files', {}):
        archived = ROOT / manifest['archive_root'] / relative
        if archived.is_file():
            return archived
        raise FileNotFoundError(
            f'Archived evidence is unavailable: {archived}. '
            f'Recover {relative} from Git commit {manifest["source_commit"]}; '
            'see stage23_bench/data/README.md.')
    raise FileNotFoundError(f'Missing evidence: {path}')


def evidence_exists(path):
    try:
        evidence_path(path)
        return True
    except FileNotFoundError:
        return False


def open_evidence(path, mode='rt'):
    if mode not in ('rt', 'rb'):
        raise ValueError('Historical evidence is read-only')
    resolved = evidence_path(path)
    opener = gzip.open if resolved.suffix == '.gz' else open
    return opener(resolved, mode, **({'encoding': 'utf-8'} if mode == 'rt' else {}))
