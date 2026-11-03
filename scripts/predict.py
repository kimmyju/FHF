"""Load an offline FHF checkpoint and create a validated Kaggle submission.parquet."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from fhf.labels import MODEL_NAMES
from fhf.model import FingerprintModel
from fhf.data import TextDataset, load_csv, make_collator
from fhf.engine import predict


def submit(checkpoint, test_csv, output, batch_size=32, max_length=384):
    """Predict one exact model name per test ID without consulting hidden labels."""
    data = load_csv(test_csv, labeled=False)
    model, tokenizer, _ = FingerprintModel.from_checkpoint(checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loader = DataLoader(TextDataset(data, tokenizer, max_length), batch_size=batch_size,
                        collate_fn=make_collator(tokenizer))
    pred_ids, _ = predict(model, loader, device)
    result = pd.DataFrame({'id': data.id, 'model': [MODEL_NAMES[idx] for idx in pred_ids]})
    if result.id.duplicated().any() or result.isna().any().any() or not result.model.isin(MODEL_NAMES).all():
        raise ValueError('Invalid submission')
    if len(result) != len(data):
        raise AssertionError('Prediction row mismatch')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f'Wrote {len(result)} predictions to {output}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-csv', required=True)
    p.add_argument('--output', default='submission.parquet')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-length', type=int, default=384)
    args = p.parse_args()
    submit(args.checkpoint, args.test_csv, args.output, args.batch_size, args.max_length)


if __name__ == '__main__':
    main()
