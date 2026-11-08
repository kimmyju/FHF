import importlib.util
from pathlib import Path
import subprocess
import pytest

spec = importlib.util.spec_from_file_location('migrate', Path(__file__).parents[1] / 'scripts/migrate_to_nas.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def repo(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    def git(*args):
        subprocess.run(['git', '-C', str(source), *args], check=True, capture_output=True)
    git('init')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    (source / 'README.md').write_text('code')
    git('add', 'README.md')
    git('commit', '-m', 'test')
    git('remote', 'add', 'origin', 'https://github.com/kimmyju/FHF.git')
    (source / 'data').mkdir()
    (source / 'data/train.csv').write_text('id,text,model\na,hello,m\n')
    (source / 'outputs/run').mkdir(parents=True)
    (source / 'outputs/run/val.csv').write_text('id,text,model\nb,world,m\n')
    (source / 'outputs/run/heads.pt').write_bytes(b'old weights')
    (source / 'outputs/run/history.json').write_text('{"accuracy": 0.5}')
    return source


def test_preview_and_verified_cleanup(tmp_path):
    source = repo(tmp_path)
    root = tmp_path / 'nas'
    m.migrate(source, root)
    assert not root.exists()
    assert (source / 'outputs/run/heads.pt').exists()
    m.migrate(source, root, True)
    assert (root / 'csv/train.csv').read_text().startswith('id,text')
    assert (root / 'csv/splits/run/val.csv').exists()
    assert not (root / 'csv/splits/run/history.json').exists()
    assert not (source / 'outputs').exists()
    assert (source / 'data').is_symlink()
    assert (root / 'code/data').resolve() == root / 'csv'
    assert (root / 'code/README.md').read_text() == 'code'
    assert (root / 'migration_manifest.json').exists()


def test_conflict_keeps_originals(tmp_path):
    source = repo(tmp_path)
    root = tmp_path / 'nas'
    (root / 'csv').mkdir(parents=True)
    (root / 'csv/train.csv').write_text('different')
    with pytest.raises(ValueError, match='Conflicting'):
        m.migrate(source, root, True)
    assert (source / 'outputs/run/heads.pt').exists()
    assert not (source / 'data').is_symlink()


def test_nested_destination_rejected(tmp_path):
    source = repo(tmp_path)
    with pytest.raises(ValueError, match='non-nested'):
        m.migrate(source, source / 'nas', True)
