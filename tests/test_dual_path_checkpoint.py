"""Regression test for dual-path checkpoint round-tripping and legacy compatibility."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from fhf.model import FingerprintModel


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 16})()
        self.emb = nn.Embedding(40, 16)

    def forward(self, input_ids, attention_mask, **kwargs):
        return type("Output", (), {"last_hidden_state": self.emb(input_ids)})()

    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_dual_path_checkpoint_roundtrip(tmp_path):
    with patch("fhf.model.AutoModel.from_pretrained", return_value=FakeEncoder()):
        model = FingerprintModel(
            "fake", heads=dict(family=True, model=True, projection=True),
            lora={"enabled": False}, pooling="attention",
            style={"enabled": True, "hash_dim": 64, "hidden_dim": 12},
            residual_fusion={"enabled": True, "flat_weight": 0.4},
        )
        model.save_checkpoint(tmp_path, FakeTokenizer())
        with patch("fhf.model.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
            loaded, _, meta = FingerprintModel.from_checkpoint(tmp_path)
    assert meta["style"]["enabled"]
    assert meta["residual_fusion"]["enabled"]
    assert loaded.style_enabled
    assert loaded.residual_enabled
    assert loaded.pooling == "attention"
    assert loaded.flat_weight == 0.4
    for name in ("family_head", "model_head", "projection",
                 "attention_pool", "style_fusion", "flat_head"):
        original = getattr(model, name).state_dict()
        restored = getattr(loaded, name).state_dict()
        assert set(original) == set(restored)
        for key in original:
            assert torch.equal(original[key], restored[key])


def test_legacy_checkpoint_without_dual_metadata_loads(tmp_path):
    with patch("fhf.model.AutoModel.from_pretrained", return_value=FakeEncoder()):
        model = FingerprintModel("fake", lora={"enabled": False})
        model.save_checkpoint(tmp_path, FakeTokenizer())
        # Mimic a checkpoint written before style-aware fields existed.
        metadata = tmp_path / "metadata.json"
        meta = json.loads(metadata.read_text())
        for key in ("pooling", "style", "residual_fusion"):
            meta.pop(key, None)
        metadata.write_text(json.dumps(meta))
        with patch("fhf.model.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
            loaded, _, _ = FingerprintModel.from_checkpoint(tmp_path)
    assert loaded.pooling == "mean"
    assert not loaded.style_enabled
    assert not loaded.residual_enabled
