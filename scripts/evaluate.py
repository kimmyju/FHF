"""Run independent labeled-data evaluation and save classification/recall metrics."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fhf.model import FingerprintModel
from fhf.data import TextDataset, load_csv, make_collator
from fhf.engine import predict
from fhf.metrics import report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='evaluation.json')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-length', type=int, default=384)
    args = p.parse_args()
    df = load_csv(args.data)
    model, tokenizer, _ = FingerprintModel.from_checkpoint(args.checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loader = DataLoader(TextDataset(df, tokenizer, args.max_length), batch_size=args.batch_size,
                        collate_fn=make_collator(tokenizer))
    predictions, ground_truth = predict(model, loader, device)
    summary = report(ground_truth, predictions)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,(dict,list))}, indent=2))


if __name__ == '__main__':
    main()
