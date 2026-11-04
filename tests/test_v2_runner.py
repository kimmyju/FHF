from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from fingerprint_v2_run import commands, STAGES


def test_driver_command_order_and_batch_size():
    cmd=commands("python",include_deberta=True,batch_size=64)
    assert set(cmd)==set(STAGES)
    assert "--set" in cmd["pretrain"]
    assert "training.samples_per_model=8" in cmd["pretrain"]
    assert "deberta/checkpoint" in " ".join(cmd["ensemble"])


def test_driver_defaults_no_holdout_training():
    cmd=commands("python")
    assert "val.csv" not in " ".join(cmd["pretrain"])
    assert "configs/fingerprint_v2/finetune.yaml" in cmd["finetune"]
