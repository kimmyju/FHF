"""Run any repository script from the NAS checkout with NAS-local caches."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(os.environ.get('FHF_ROOT', '/media/NAS/USERS/yeju/FHF')).expanduser().resolve()
if not (ROOT / 'code').is_dir():
    raise SystemExit('Run migrate_to_nas.py first; NAS code directory is missing.')
for key, folder in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                    'XDG_CACHE_HOME': 'xdg', 'WANDB_DIR': 'wandb',
                    'WANDB_CACHE_DIR': 'wandb-cache', 'TMPDIR': 'tmp'}.items():
    path = ROOT / 'cache' / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['FHF_ROOT'] = str(ROOT)
os.chdir(ROOT / 'code')
if len(sys.argv) < 2:
    raise SystemExit('Usage: python scripts/nas_run.py scripts/train.py [arguments...]')
target = Path(sys.argv[1]).resolve()
if not target.is_relative_to(ROOT / 'code') or not target.is_file():
    raise SystemExit('Target must be a script inside the NAS code checkout.')
sys.path.insert(0, str(ROOT / 'code' / 'src'))
sys.path.insert(0, str(target.parent))
sys.argv = sys.argv[1:]
runpy.run_path(str(target), run_name='__main__')
