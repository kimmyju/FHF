"""Train a style-only correction on a restored, frozen v3 EOS model."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from fhf.v3_frozen_style import train_frozen, predict_frozen


def main():
    root = Path(os.environ.get('FHF_ROOT', '/media/NAS/USERS/yeju/FHF'))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--source', required=True, help='Original EOS results directory containing manifest.json')
    fit.add_argument('--config', default='configs/fingerprint_v3/qwen3_8b_frozen_style.json')
    fit.add_argument('--train', required=True)
    fit.add_argument('--valid', required=True)
    fit.add_argument('--out', default='v3-qwen8b-frozen-eos-style')
    fit.add_argument('--base', help='Optional local base weights; uses source model/revision otherwise')
    fit.add_argument('--wandb', action='store_true')
    pred = sub.add_parser('predict')
    pred.add_argument('--bundle', required=True)
    pred.add_argument('--test', required=True)
    pred.add_argument('--base', required=True, help='Local original base weights for offline inference')
    pred.add_argument('--out', default='submission.parquet')
    pred.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'predict':
        predict_frozen(args.bundle, pd.read_csv(args.test, dtype={'id': str}),
                       args.base, args.out, args.batch_size)
        return
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.out)
    if not output.is_absolute():
        if '..' in output.parts:
            raise ValueError('Run name may not escape results')
        output = root / 'results' / output
    for key, folder in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                        'WANDB_DIR': 'wandb', 'WANDB_CACHE_DIR': 'wandb-cache'}.items():
        path = root / 'cache' / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    tracker = None
    try:
        if args.wandb:
            import wandb
            tracker = wandb.init(project=cfg['wandb_project'], name=output.name, config=cfg,
                                 tags=['fingerprint-v3', 'frozen-eos', 'style-only'])
        read = lambda path: pd.read_csv(path, dtype={'id': str, 'prompt_id': str})
        result = train_frozen(args.source, read(args.train), read(args.valid), cfg, output, args.base, tracker)
        print(json.dumps(result, indent=2))
    finally:
        if tracker:
            tracker.finish()


if __name__ == '__main__':
    main()
