"""Run one W&B Sweep trial using dotted-key parameters and isolated output paths.

Launched by `wandb agent`, not by Kaggle scoring. Sweep parameters override
YAML values. Each W&B run ID gets a unique output directory, avoiding collisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from fhf.config import load_config
from train import run_training


def sweep_overrides(parameters):
    """Convert W&B sweep parameters to validated --set dotted-key overrides."""
    return [f"{key}={json.dumps(value)}" for key, value in parameters.items()]


def isolate_outputs(config, run_id):
    """Place all writable per-run artifacts under a unique directory."""
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise ValueError("Unsafe or empty run ID")
    root = Path(config.get("sweep", {}).get("output_root", "outputs/sweeps"))
    base = Path(config["paths"].get("base_dir", "."))
    if not root.is_absolute():
        root = base / root
    run_dir = root / run_id
    mapping = {
        "checkpoint_dir": run_dir / "checkpoint",
        "validation_csv": run_dir / "val.csv",
        "train_split_csv": run_dir / "train_split.csv",
        "history_json": run_dir / "history.json",
        "config_snapshot": run_dir / "config_resolved.yaml",
        "eval_csv": run_dir / "val.csv",
        "eval_report": run_dir / "evaluation.json",
    }
    for key, path in mapping.items():
        config["paths"][key] = str(path.resolve())
    return run_dir


def main():
    """Consume one agent-provided parameter choice and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-root", default="outputs/sweeps")
    args = parser.parse_args()
    # Sweep agent sets WANDB_SWEEP_ID; project/entity selection belongs to sweep YAML.
    # wandb.init fetches this trial's parameters from the sweep controller.
    with wandb.init(job_type="sweep-trial") as run:
        config = load_config(args.config, sweep_overrides(dict(run.config)))
        config["wandb"]["enabled"] = True
        config["wandb"]["mode"] = "online"
        config["wandb"]["group"] = run.sweep_id or "sweep"
        config["wandb"]["run_name"] = run.name
        config["sweep"] = {"output_root": args.output_root, "run_id": run.id}
        output_dir = isolate_outputs(config, run.id)
        run.config.update({"resolved_experiment": config}, allow_val_change=True)
        print(f"Sweep run {run.id}, output: {output_dir}")
        run_training(config)


if __name__ == "__main__":
    main()
