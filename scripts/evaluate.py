"""Evaluate a saved model with paths and inference settings from YAML."""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from fhf.config import load_config, resolve_path
from fhf.data import TextDataset, load_csv, make_collator
from fhf.model import FingerprintModel
from fhf.engine import predict, _evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    conf = load_config(args.config, args.set)
    df = load_csv(resolve_path(conf, "eval_csv"))
    base_override = (resolve_path(conf, "inference_base_encoder")
                     if conf["paths"].get("inference_base_encoder") else None)
    model, tokenizer, _ = FingerprintModel.from_checkpoint(
        resolve_path(conf, "checkpoint_dir"), base_encoder_override=base_override)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    data = TextDataset(df, tokenizer, int(conf["model"]["max_length"]))
    loader = DataLoader(data, batch_size=int(conf["inference"]["batch_size"]),
                        collate_fn=make_collator(tokenizer))
    prediction, target = predict(model, loader, device)
    metrics = _evaluate(prediction, target, model)
    dest = resolve_path(conf, "eval_report")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Evaluation written to {dest}")


if __name__ == "__main__":
    main()
