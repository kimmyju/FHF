"""Train FHF on the Kaggle labeled CSV with prompt-disjoint validation and ablations."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from fhf.data import load_csv, grouped_split, TextDataset, FamilyBatchSampler, make_collator
from fhf.engine import fit, seed_everything
from fhf.model import FingerprintModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--output-dir', default='outputs/fhf')
    parser.add_argument('--backbone', default='microsoft/deberta-v3-base')
    parser.add_argument('--tokenizer', default=None, help='Optional tokenizer directory if backbone is an encoder export')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--family-weight', type=float, default=0.25)
    parser.add_argument('--contrastive-weight', type=float, default=0.25)
    parser.add_argument('--consistency-weight', type=float, default=0.05)
    parser.add_argument('--hotflip-weight', type=float, default=0.0)
    parser.add_argument('--hotflip-max-examples', type=int, default=2)
    parser.add_argument('--batch-families', type=int, default=3)
    parser.add_argument('--models-per-family', type=int, default=2)
    parser.add_argument('--samples-per-model', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    args = parser.parse_args()
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.train_csv)
    train_df, val_df = grouped_split(df, args.validation_fraction, args.seed)
    train_df.to_csv(out/'train_split.csv', index=False)
    val_df.to_csv(out/'val.csv', index=False)
    (out/'config.json').write_text(json.dumps(vars(args), indent=2))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.backbone)
    train_ds = TextDataset(train_df, tokenizer, args.max_length)
    val_ds = TextDataset(val_df, tokenizer, args.max_length)
    collator = make_collator(tokenizer)
    sampler = FamilyBatchSampler(train_ds.labels, args.batch_families, args.models_per_family,
                                 args.samples_per_model, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, collate_fn=collator)
    model = FingerprintModel(args.backbone)
    history = fit(model, tokenizer, train_loader, val_loader, out, epochs=args.epochs,
                  lr=args.learning_rate, weight_decay=args.weight_decay,
                  family_weight=args.family_weight, contrastive_weight=args.contrastive_weight,
                  consistency_weight=args.consistency_weight, hotflip_weight=args.hotflip_weight,
                  hotflip_max_examples=args.hotflip_max_examples,
                  device='cuda' if torch.cuda.is_available() else 'cpu')
    (out/'history.json').write_text(json.dumps(history, indent=2))
    print(f'Best checkpoint: {out}')


if __name__ == '__main__':
    main()
