"""Replay remains a separate optional experiment."""
from pathlib import Path
from fhf.config import load_config

def test_replay_dataset_is_small_and_uses_pretrain_checkpoint():
    root=Path(__file__).resolve().parents[1]/"configs/fingerprint_v2"
    cfg=load_config(root/"replay.yaml")
    assert cfg["external_data"]["min_accepted_rows"]==540
    assert cfg["paths"]["initial_checkpoint"]=="outputs/fingerprint-v2/pretrain/checkpoint"
    assert cfg["training"]["steps_per_epoch"] is None
