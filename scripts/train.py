"""Train FHF from YAML; expose run_training for W&B sweep agents.

All datasets, checkpoints, metrics and W&B metadata use the resolved YAML
configuration. Do not store an API key in source code or experiment YAML.
"""
from __future__ import annotations

import argparse
import json

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fhf.config import load_config, resolve_path
from fhf.data import (
    TextDataset, FamilyBatchSampler,
    PromptAwareFamilyBatchSampler, make_collator,
)
from fhf.external_data import training_frames, write_external_audit
from fhf.full_coverage import FullCoveragePromptFamilyBatchSampler
from fhf.prompt_match import PromptMatchedFullCoverageBatchSampler
from fhf.engine import fit, seed_everything
from fhf.model import FingerprintModel
from fhf.tracking import WandbTracker


def run_training(cfg):
    """Execute one reproducible experiment, logging to W&B when enabled."""
    seed_everything(int(cfg["seed"]))
    hard_path = cfg["paths"].get("hard_pair_weights_json")
    if hard_path:
        weights = json.loads(resolve_path(cfg, "hard_pair_weights_json").read_text(encoding="utf-8"))
        if not isinstance(weights, dict):
            raise ValueError("Hard-pair weights must be a JSON model-name mapping")
        cfg["losses"]["intra_contrastive"]["hard_pair_weights"] = weights
        print(f"Loaded OOF-derived confusion weights from {hard_path}")
    checkpoint_dir = resolve_path(cfg, "checkpoint_dir")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train, val, external, audit = training_frames(cfg)
    for key, frame in (("train_split_csv", train), ("validation_csv", val)):
        destination = resolve_path(cfg, key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    if audit is not None:
        write_external_audit(cfg, external, audit)
        print(
            f"External data: accepted {audit['accepted_external_rows']}/"
            f"{audit['raw_rows']} samples; merged train={audit['merged_train_rows']}; "
            f"frozen validation={audit['frozen_validation_rows']}"
        )
    snapshot = resolve_path(cfg, "config_snapshot")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    model_cfg, train_cfg = cfg["model"], cfg["training"]
    initial = cfg["paths"].get("initial_checkpoint")
    if initial:
        model, tokenizer, metadata = FingerprintModel.from_checkpoint(
            resolve_path(cfg, "initial_checkpoint"), trainable_adapter=True)
        for key, expected in (
            ("base_encoder", model_cfg["encoder"]),
            ("heads", model_cfg["heads"]),
            ("pooling", model_cfg.get("pooling", "mean")),
            ("style", model_cfg.get("style", {"enabled": False})),
            ("residual_fusion", model_cfg.get("residual_fusion", {"enabled": False})),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"Incompatible warm-start checkpoint {key}: "
                                 f"{metadata.get(key)!r} != {expected!r}")
        if model.projection is not None and model.projection[-1].out_features != int(model_cfg["projection_dim"]):
            raise ValueError("Warm-start projection dim mismatch")
        print(f"Warm-started trainable LoRA + heads from {initial}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"] or model_cfg["encoder"])
        model = FingerprintModel(
            model_cfg["encoder"],
            dropout=float(model_cfg["dropout"]),
            projection_dim=int(model_cfg["projection_dim"]),
            heads=model_cfg["heads"],
            lora=model_cfg.get("lora", {"enabled": False}),
            pooling=model_cfg.get("pooling", "mean"),
            style=model_cfg.get("style", {"enabled": False}),
            residual_fusion=model_cfg.get("residual_fusion", {"enabled": False}),
        )
    prompt_aware = (
        cfg["losses"]["intra_contrastive"]["enabled"]
        and float(cfg["losses"]["intra_contrastive"]["weight"]) > 0
        and cfg["losses"]["intra_contrastive"].get("mode", "standard") in ("prompt_aware", "within_family")
    )
    training_ds = TextDataset(
        train, tokenizer, int(model_cfg["max_length"]), style_config=model.style_config,
        include_prompt_id=prompt_aware,
    )
    valid_ds = TextDataset(
        val, tokenizer, int(model_cfg["max_length"]), style_config=model.style_config
    )
    collator = make_collator(tokenizer)
    sampler_mode = train_cfg.get("sampler", "balanced")
    if sampler_mode in ("full_coverage", "prompt_matched_full_coverage"):
        if not prompt_aware:
            raise ValueError("Full coverage requires prompt-aware labels")
        sampler_cls = (PromptMatchedFullCoverageBatchSampler
                       if sampler_mode == "prompt_matched_full_coverage"
                       else FullCoveragePromptFamilyBatchSampler)
    elif sampler_mode == "balanced":
        sampler_cls = PromptAwareFamilyBatchSampler if prompt_aware else FamilyBatchSampler
    else:
        raise ValueError(f"Unknown sampler mode: {sampler_mode}")
    sampler = sampler_cls(
        training_ds.labels,
        **({"prompt_ids": training_ds.prompt_ids} if prompt_aware else {}),
        batch_families=int(train_cfg["batch_families"]),
        models_per_family=int(train_cfg["models_per_family"]),
        samples_per_model=int(train_cfg["samples_per_model"]),
        steps_per_epoch=train_cfg.get("steps_per_epoch"),
        seed=int(cfg["seed"]),
    )
    print(f"Training rows={len(train)}, sampler={sampler_mode}, "
          f"batch size={int(train_cfg['batch_families']) * int(train_cfg['models_per_family']) * int(train_cfg['samples_per_model'])}, "
          f"batches per epoch={len(sampler)}")
    train_loader = DataLoader(
        training_ds, batch_sampler=sampler, collate_fn=collator,
        num_workers=int(train_cfg["num_workers"]),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=int(train_cfg["eval_batch_size"]),
        collate_fn=collator, num_workers=int(train_cfg["num_workers"]),
    )
    tracker = WandbTracker(cfg)
    try:
        history = fit(
            model, tokenizer, train_loader, valid_loader, checkpoint_dir, cfg,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tracker=tracker,
        )
        destination = resolve_path(cfg, "history_json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Experiment: {cfg['experiment']}, checkpoint: {checkpoint_dir}")
        return history
    finally:
        tracker.finish()


def main():
    """Read CLI overrides and start one training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()
    run_training(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
