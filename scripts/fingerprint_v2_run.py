"""Run the fingerprint-v2 experiment stages in reproducible order.

This driver executes local scripts in the foreground; it does not download
external datasets, train in the background, or submit to Kaggle by itself.
Use --dry-run to inspect the exact commands before running any GPU work.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

STAGES=("select", "classical", "hard", "pretrain", "finetune",
        "replay_select", "replay", "deberta", "ensemble")
DEFAULT=("select", "classical", "hard", "pretrain", "finetune", "ensemble")


def commands(python, include_deberta=False, batch_size=32):
    if batch_size not in (8,16,32,64):
        raise ValueError("batch_size must be 8, 16, 32 or 64")
    # Family batch composition remains 4 families x 2 models x SPM;
    # hardware-specific DeBERTa defaults to its own smaller configuration.
    spm=batch_size//8
    select=[python, "scripts/select_open_pool_full.py"]
    classical=[python, "scripts/fingerprint_v2_classical.py"]
    hard=[python, "scripts/fingerprint_v2_hard_pairs.py"]
    pretrain=[python, "scripts/train.py", "--config",
              "configs/fingerprint_v2/pretrain.yaml",
              "--set", f"training.samples_per_model={spm}"]
    finetune=[python, "scripts/train.py", "--config",
              "configs/fingerprint_v2/finetune.yaml",
              "--set", f"training.samples_per_model={spm}"]
    replay_select=[python, "scripts/select_open_pool_e1.py",
                   "--per-model", "30",
                   "--output", "data/external/replay/selected_30_per_model.csv",
                   "--report", "outputs/fingerprint-v2/replay/selection_report.json"]
    replay=[python, "scripts/train.py", "--config",
            "configs/fingerprint_v2/replay.yaml",
            "--set", f"training.samples_per_model={spm}"]
    deberta=[python, "scripts/train.py", "--config",
             "configs/fingerprint_v2/deberta.yaml"]
    ensemble=[python, "scripts/fingerprint_v2_ensemble.py",
              "--mode", "validation",
              "--checkpoint", "outputs/fingerprint-v2/finetune/checkpoint"]
    if include_deberta:
        ensemble += ["--checkpoint","outputs/fingerprint-v2/deberta/checkpoint"]
    return dict(zip(STAGES,(select,classical,hard,pretrain,
                            finetune,replay_select,replay,deberta,ensemble)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stages",nargs="+",choices=STAGES,default=None)
    p.add_argument("--with-deberta",action="store_true")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    wanted=a.stages or list(DEFAULT)
    if a.with_deberta and "deberta" not in wanted:
        wanted=list(wanted)
        wanted.insert(wanted.index("ensemble") if "ensemble" in wanted
                      else len(wanted),"deberta")
    cmd_map=commands(sys.executable,a.with_deberta,a.batch_size)
    repo=Path(__file__).resolve().parents[1]
    for stage in wanted:
        cmd=cmd_map[stage]
        print(f"[{stage}] {shlex.join(cmd)}",flush=True)
        if not a.dry_run:
            subprocess.run(cmd,cwd=repo,check=True)
    print("Workflow completed." if not a.dry_run else "Dry run only.")


if __name__=="__main__":
    main()
