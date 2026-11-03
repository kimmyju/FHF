"""Run reproducible FHF training from an experiment YAML, including independent head ablations."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fhf.config import load_config, resolve_path
from fhf.data import load_csv, grouped_split, TextDataset, FamilyBatchSampler, make_collator
from fhf.engine import fit, seed_everything
from fhf.model import FingerprintModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override any YAML field, e.g. model.heads.family=false")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(int(cfg["seed"]))
    train_csv = resolve_path(cfg, "train_csv")
    checkpoint_dir = resolve_path(cfg, "checkpoint_dir")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    df = load_csv(train_csv)
    train, val = grouped_split(df, float(cfg["training"]["validation_fraction"]), int(cfg["seed"]))
    for key, frame in [("train_split_csv", train), ("validation_csv", val)]:
        destination = resolve_path(cfg, key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    snapshot = resolve_path(cfg, "config_snapshot")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    model_cfg, train_cfg = cfg["model"], cfg["training"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"] or model_cfg["encoder"])
    model = FingerprintModel(model_cfg["encoder"], dropout=float(model_cfg["dropout"]),
                             projection_dim=int(model_cfg["projection_dim"]),
                             heads=model_cfg["heads"])
    train_data = TextDataset(train, tokenizer, int(model_cfg["max_length"]))
    valid_data = TextDataset(val, tokenizer, int(model_cfg["max_length"]))
    collator = make_collator(tokenizer)
    sampler = FamilyBatchSampler(
        train_data.labels, batch_families=int(train_cfg["batch_families"]),
        models_per_family=int(train_cfg["models_per_family"]),
        samples_per_model=int(train_cfg["samples_per_model"]),
        steps_per_epoch=train_cfg.get("steps_per_epoch"), seed=int(cfg["seed"]))
    train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collator,
                              num_workers=int(train_cfg["num_workers"]))
    valid_loader = DataLoader(valid_data, batch_size=int(train_cfg["eval_batch_size"]),
                              collate_fn=collator, num_workers=int(train_cfg["num_workers"]))
    history = fit(model, tokenizer, train_loader, valid_loader, checkpoint_dir, cfg,
                  device="cuda" if torch.cuda.is_available() else "cpu")
    destination = resolve_path(cfg, "history_json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Experiment: {cfg['experiment']}, checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
