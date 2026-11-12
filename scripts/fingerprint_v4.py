"""Independent 18-source likelihood experiment; leaves v3 bundles untouched."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', default='configs/fingerprint_v4/qwen3_4b_source_likelihood.json')
    for name in ['train', 'valid', 'external']:
        train.add_argument('--' + name, required=True)
    train.add_argument('--out', required=True, help='New run name below NAS likelihood_v4/, or absolute directory')
    train.add_argument('--wandb', action='store_true')
    audit = sub.add_parser('audit')
    for name in ['train', 'valid', 'external', 'out']:
        audit.add_argument('--' + name, required=True)
    pred = sub.add_parser('predict')
    for name in ['bundle', 'test', 'base', 'out']:
        pred.add_argument('--' + name, required=True)
    pred.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    root = Path(os.environ.get('FHF_ROOT', '/media/NAS/USERS/yeju/FHF'))
    for key, directory in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                            'WANDB_DIR': 'wandb', 'WANDB_CACHE_DIR': 'wandb-cache'}.items():
        path = root / 'cache' / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    import pandas as pd
    from fhf import v4_likelihood as v4
    read = lambda path: pd.read_csv(path, dtype={'id': str, 'prompt_id': str})
    if args.command == 'predict':
        v4.predict(args.bundle, read(args.test), args.base, args.out, args.batch_size)
        return
    if args.command == 'audit':
        _, _, _, report = v4.audit_data(read(args.train), read(args.valid), read(args.external))
        out = Path(args.out)
        if out.exists():
            raise ValueError('Audit output exists')
        out.parent.mkdir(parents=True, exist_ok=True)
        v4.dump(out, report)
        print(f'Data audit saved: {out}')
        return
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    if not out.is_absolute():
        if '..' in out.parts:
            raise ValueError('Run name may not escape NAS root')
        out = root / 'likelihood_v4' / out
    if out.exists():
        raise ValueError('Output exists; choose a new run name')
    tracker = None
    try:
        if args.wandb:
            import wandb
            tracker = wandb.init(project=cfg['wandb_project'], name=out.name, config=cfg,
                                 tags=['fingerprint-v4', 'source-likelihood', '18-models'])
        result = v4.train(read(args.train), read(args.valid), read(args.external), cfg, out, tracker)
        print(json.dumps(result, indent=2))
    finally:
        if tracker:
            tracker.finish()


if __name__ == '__main__':
    main()
