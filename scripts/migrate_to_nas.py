"""Verified NAS migration. Dry-run by default; --apply performs the printed plan.

Run only after stopping all training processes using the source repository.
Only explicit generated directories are deleted, after data verification.
Remote W&B history is not deleted. Original code checkout is retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(directory):
    if not directory.exists():
        return []
    # Refuse directory links: they could point outside the intended migration.
    for p in directory.rglob('*'):
        if p.is_symlink() and (p.is_dir() or not p.exists()):
            raise ValueError(f'Directory/broken symlink requires manual review: {p}')
    return sorted(p for p in directory.rglob('*') if p.is_file())


def is_dataset(path):
    """Identify data content, rather than preserve metrics/prediction tables."""
    suffix = path.suffix.lower()
    data_fields = {'text', 'response', 'completion', 'prompt', 'prompt_text', 'passage'}
    if suffix in {'.csv', '.tsv'}:
        import csv
        with path.open(encoding='utf-8-sig', newline='') as stream:
            columns = next(csv.reader(stream, delimiter='\t' if suffix == '.tsv' else ','), [])
        return bool(data_fields & set(columns))
    if suffix == '.parquet':
        import pyarrow.parquet as pq
        return bool(data_fields & set(pq.read_schema(path).names))
    if suffix in {'.json', '.jsonl'}:
        with path.open(encoding='utf8') as stream:
            value = json.loads(stream.readline()) if suffix == '.jsonl' else json.load(stream)
        if isinstance(value, list):
            value = value[0] if value else {}
        return isinstance(value, dict) and bool(data_fields & set(value))
    return False


def migrate(source, root, apply=False):
    source, root = Path(source).expanduser().resolve(), Path(root).expanduser().resolve()
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError('Source checkout and NAS destination must be separate, non-nested directories')
    if not (source / '.git').exists():
        raise ValueError('Source must be the existing FHF Git checkout')
    if (root / 'code').exists():
        raise ValueError('Destination code already exists; refusing overwrite or repeated cleanup')
    generated = [source / n for n in ('outputs', 'checkpoints', 'results', 'wandb')]
    data = source / 'data'
    for path in [data, *generated]:
        if path.is_symlink():
            raise ValueError(f'Refusing to delete/replace linked directory: {path}')
    # Preserve recognized raw-text datasets and train/validation splits before cleanup.
    mapping = [(p, root / 'csv' / p.relative_to(data)) for p in files(data)]
    for directory in generated:
        for p in files(directory):
            if is_dataset(p):
                prefix = 'splits' if directory.name == 'outputs' else f'preserved-{directory.name}'
                mapping.append((p, root / 'csv' / prefix / p.relative_to(directory)))
    if not mapping:
        raise ValueError('No dataset files found; refusing cleanup')
    tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    for item in filter(None, tracked):
        if any((source / item).is_relative_to(d) for d in [data, *generated]):
            raise ValueError(f'Cleanup directory contains a Git-tracked file: {item}')
    # Protect code edits: cloning committed code must not silently lose local work.
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Commit/stash tracked code changes before migration')
    for src, dst in mapping:
        if dst.is_symlink() or not dst.resolve().is_relative_to(root):
            raise ValueError(f"Destination escapes NAS root or is a symlink: {dst}")
        if dst.exists() and (not dst.is_file() or digest(src) != digest(dst)):
            raise ValueError(f'Conflicting destination: {dst}')
    plan = {'source': str(source), 'destination': str(root), 'files_to_preserve': len(mapping),
            'bytes_to_copy': sum(p.stat().st_size for p, _ in mapping),
            'delete_after_verification': [str(p) for p in generated if p.exists()],
            'replace_with_data_link': str(data),
            'note': 'Text-bearing datasets under old outputs survive in csv/splits; remote W&B unchanged.',
            'preserved_files': [{'source': str(a), 'destination': str(b)} for a, b in mapping]}
    print(json.dumps(plan, indent=2), flush=True)
    if not apply:
        print('Preview only. Add --apply to copy, verify and clean the listed directories.')
        return plan
    root.mkdir(parents=True, exist_ok=True)
    for name in ('csv', 'checkpoints', 'results', 'cache'):
        (root / name).mkdir(exist_ok=True)
    records = []
    for src, dst in mapping:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            tmp = dst.with_name(dst.name + '.fhf-copying')
            shutil.copy2(src, tmp)
            if digest(src) != digest(tmp):
                raise RuntimeError(f'Copy verification failed: {src}; originals untouched')
            tmp.replace(dst)
        checksum = digest(src)
        if checksum != digest(dst):
            raise RuntimeError(f'Verification failed: {src}; originals untouched')
        records.append({'source': str(src), 'destination': str(dst), 'sha256': checksum})
    subprocess.run(['git', 'clone', '--no-hardlinks', str(source), str(root / 'code')], check=True)
    # Local clone otherwise points origin at the old machine directory.
    origin = subprocess.check_output(['git', '-C', str(source), 'remote', 'get-url', 'origin']).decode().strip()
    subprocess.run(['git', '-C', str(root / 'code'), 'remote', 'set-url', 'origin', origin], check=True)
    for name, target in [('data', root / 'csv'), ('outputs', root / 'results'),
                         ('checkpoints', root / 'checkpoints'), ('results', root / 'results'),
                         ('wandb', root / 'cache' / 'wandb')]:
        target.mkdir(parents=True, exist_ok=True)
        (root / 'code' / name).symlink_to(target, target_is_directory=True)
    # Recheck source hashes immediately before deletion to detect active writers.
    for record in records:
        if digest(Path(record['source'])) != record['sha256']:
            raise RuntimeError('Source data changed during migration; originals not deleted')
    (root / 'migration_manifest.json').write_text(json.dumps({'plan': plan, 'files': records}, indent=2))
    for directory in generated:
        if directory.exists():
            shutil.rmtree(directory)
    if data.exists():
        shutil.rmtree(data)
    data.symlink_to(root / 'csv', target_is_directory=True)
    print(f'Completed. Code: {root / "code"}. Original source code retained.')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--root', default='/media/NAS/USERS/yeju/FHF')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    migrate(args.source, args.root, args.apply)
