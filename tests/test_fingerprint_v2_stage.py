"""Stages use the same architecture and separate official evaluation."""
from pathlib import Path
from fhf.config import load_config

ROOT=Path(__file__).resolve().parents[1]
def test_stages():
    p=load_config(ROOT/"configs/fingerprint_v2/pretrain.yaml")
    f=load_config(ROOT/"configs/fingerprint_v2/finetune.yaml")
    assert p["model"] == f["model"]
    assert f["training"]["frozen_split"]
    assert not p["training"].get("frozen_split",False)
    assert f["paths"]["initial_checkpoint"] == p["paths"]["checkpoint_dir"]
    assert f["training"]["steps_per_epoch"] is None
    assert p["training"]["steps_per_epoch"] is None
    assert f["training"]["head_learning_rate"] > f["training"]["learning_rate"]
