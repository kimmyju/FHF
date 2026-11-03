"""Verify LoRA checkpoint layout and trainable module isolation without downloading model weights."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from fhf.model import FingerprintModel


class FakeEncoder(nn.Module):
    """Mimic the minimal pretrained ModernBERT interface for offline tests."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embeddings = nn.Embedding(40, 12)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


class FakePeftModel(nn.Module):
    """Minimal PEFT stand-in to test adapter-only storage behavior."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(12, 2))
        self.lora_B = nn.Parameter(torch.randn(2, 12))

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.base(input_ids, attention_mask, **kwargs)

    def save_pretrained(self, path, safe_serialization=True):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text('{"peft_type":"LORA"}')

    def get_base_model(self):
        return self.base


class FakeTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_lora_leaves_base_frozen_and_heads_trainable():
    """Only LoRA matrices and enabled classifier heads should require gradients."""
    with patch("fhf.model.AutoModel.from_pretrained", return_value=FakeEncoder()), patch(
        "fhf.model.apply_lora", side_effect=lambda base, config: FakePeftModel(base)
    ):
        model = FingerprintModel("fake", lora={"enabled": True})
    assert not any(p.requires_grad for p in model.encoder.base.parameters())
    assert model.encoder.lora_A.requires_grad
    assert all(p.requires_grad for p in model.family_head.parameters())
    assert model.trainable_parameters() > 0


def test_checkpoint_saves_adapter_not_full_base(tmp_path):
    """Default LoRA checkpoints store adapter and heads, not a 395M base model copy."""
    with patch("fhf.model.AutoModel.from_pretrained", return_value=FakeEncoder()), patch(
        "fhf.model.apply_lora", side_effect=lambda base, config: FakePeftModel(base)
    ):
        model = FingerprintModel("answerdotai/ModernBERT-large", lora={"enabled": True})
    model.save_checkpoint(tmp_path, FakeTokenizer())
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "heads.pt").exists()
    assert (tmp_path / "metadata.json").exists()
    assert not (tmp_path / "base_encoder").exists()
    import json
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["base_encoder"] == "answerdotai/ModernBERT-large"
    assert meta["lora"]["enabled"]


def test_default_lora_targets():
    """ModernBERT fused attention projections must be the configured LoRA targets."""
    from fhf.model import DEFAULT_LORA
    assert DEFAULT_LORA["target_modules"] == ["Wqkv", "Wo"]
    assert DEFAULT_LORA["r"] == 16
