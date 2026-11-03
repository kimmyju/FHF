"""Evaluate frozen ModernBERT and Style-only predictions with probability fusion.

Uses the exact saved validation split. Never retrains or alters FHF checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fhf.config import load_config, resolve_path
from fhf.data import TextDataset, load_csv, make_collator
from fhf.fusion import DEFAULT_BERT_WEIGHTS, evaluate_fusion
from fhf.model import FingerprintModel


def checkpoint_probabilities(model, tokenizer, validation, config, device):
    """Return class probabilities in fhf.labels.MODEL_NAMES order."""
    if not model.heads.get("model", False):
        raise ValueError("Fusion requires an 18-class Model Head checkpoint")
    dataset = TextDataset(validation, tokenizer, int(config["model"]["max_length"]))
    loader = DataLoader(
        dataset, batch_size=int(config["inference"]["batch_size"]),
        shuffle=False, collate_fn=make_collator(tokenizer)
    )
    model.to(device).eval()
    result = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch.pop("families", None)
            logits = model(**{k: v.to(device) for k, v in batch.items()})["log_probs"]
            result.append(logits.float().exp().cpu().numpy())
    return np.concatenate(result, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablations/model_only.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--style-predictions",
        default="outputs/style-baseline/val_predictions.csv",
    )
    parser.add_argument("--output-dir", default="outputs/fusion-eval")
    parser.add_argument(
        "--bert-weights", nargs="+", type=float,
        default=list(DEFAULT_BERT_WEIGHTS),
        help="Fixed BERT weights; Style receives 1 - weight.",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)

    validation = load_csv(resolve_path(config, "eval_csv"))
    if "prompt_id" not in validation:
        raise ValueError("Saved validation set must include prompt_id")
    style_path = Path(args.style_predictions).expanduser()
    if not style_path.is_file():
        raise FileNotFoundError(
            f"Missing {style_path}. First run python scripts/style_baseline.py "
            "on the same prompt-disjoint split."
        )
    style_predictions = pd.read_csv(
        style_path, dtype={"id": str, "prompt_id": str}
    )
    base_override = (resolve_path(config, "inference_base_encoder")
                     if config["paths"].get("inference_base_encoder") else None)
    model, tokenizer, _ = FingerprintModel.from_checkpoint(
        resolve_path(config, "checkpoint_dir"),
        base_encoder_override=base_override,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_prob = checkpoint_probabilities(
        model, tokenizer, validation, config, device
    )
    summary, predictions, fusion_grid = evaluate_fusion(
        validation, bert_prob, style_predictions,
        bert_weights=args.bert_weights,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    fusion_grid.to_csv(output_dir / "fusion_grid.csv", index=False)
    print(f"Samples: {summary['n']}")
    print(f"BERT BAcc:   {summary['bert']['balanced_accuracy']:.4f}")
    print(f"Style BAcc:  {summary['style']['balanced_accuracy']:.4f}")
    print(f"Agreement:   {summary['prediction_agreement']:.4f}")
    print(f"Both wrong:  {summary['both_wrong_rate']:.4f}")
    print(f"Oracle ensemble (upper bound): {summary['oracle_ensemble_balanced_accuracy']:.4f}")
    for row in summary["fusion_grid"]:
        print(f"BERT:Style={row['bert_weight']:.2f}:{row['style_weight']:.2f} "
              f"BAcc={row['balanced_accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
