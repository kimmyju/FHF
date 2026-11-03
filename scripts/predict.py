"""Generate a validated 18-class Parquet submission using YAML-configured input and output paths."""
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from fhf.config import load_config, resolve_path
from fhf.labels import MODEL_NAMES
from fhf.model import FingerprintModel
from fhf.data import TextDataset, load_csv, make_collator
from fhf.engine import predict


def submit(checkpoint, test_csv, output, batch_size=32, max_length=384,
           base_encoder_override=None):
    """Disallow family-only checkpoints: they cannot produce valid 18-class submissions."""
    df = load_csv(test_csv, labeled=False)
    model, tokenizer, _ = FingerprintModel.from_checkpoint(checkpoint, base_encoder_override=base_encoder_override)
    if not model.heads["model"]:
        raise ValueError("A family-only checkpoint cannot submit 18-class predictions")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    data = TextDataset(df, tokenizer, int(max_length))
    loader = DataLoader(data, batch_size=int(batch_size), collate_fn=make_collator(tokenizer))
    predicted, _ = predict(model, loader, device)
    result = pd.DataFrame({"id": df.id, "model": [MODEL_NAMES[i] for i in predicted]})
    if len(result) != len(df) or result.isna().any().any() or result.id.duplicated().any():
        raise ValueError("Invalid submission rows")
    if not result.model.isin(MODEL_NAMES).all():
        raise ValueError("Invalid competition labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"Saved {len(result)} predictions at {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    base_override = (resolve_path(config, "inference_base_encoder")
                     if config["paths"].get("inference_base_encoder") else None)
    submit(resolve_path(config, "checkpoint_dir"), resolve_path(config, "test_csv"),
           resolve_path(config, "submission_parquet"), config["inference"]["batch_size"],
           config["model"]["max_length"], base_encoder_override=base_override)


if __name__ == "__main__":
    main()
