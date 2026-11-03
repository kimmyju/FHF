"""Prepare an external-training augmentation safely without starting GPU training."""
from __future__ import annotations

import argparse
import json

from fhf.config import load_config, resolve_path
from fhf.external_data import training_frames, write_external_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablations/dual_path_prompt_aware_ext.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    train, val, external, report = training_frames(config)
    if report is None:
        raise ValueError("Preparation requires external_data.enabled=true")
    # Only write after the full validation and contamination check succeeded.
    for key, frame in (("train_split_csv", train), ("validation_csv", val)):
        dest = resolve_path(config, key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)
    write_external_audit(config, external, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Prepared {len(train)} train, {len(val)} frozen validation examples.")
    print("Inspect external_audit.json and external_clean.csv before training.")


if __name__ == "__main__":
    main()
