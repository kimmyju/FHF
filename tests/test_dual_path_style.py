"""Dual-path style embeddings, attention pooling, gates, and residual heads."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from unittest.mock import patch

from fhf.data import TextDataset, make_collator
from fhf.model import FingerprintModel
from fhf.style import StyleFeatureExtractor, STYLE_SCALAR_NAMES, MaskedAttentionPool


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 16})()
        self.embed = nn.Embedding(40, 16)

    def forward(self, input_ids, attention_mask, **kwargs):
        return type("Output", (), {"last_hidden_state": self.embed(input_ids)})()


class TinyTokenizer:
    def __call__(self, text, truncation=True, max_length=384):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def pad(self, items, padding=True, return_tensors="pt"):
        return {key: torch.tensor([item[key] for item in items])
                for key in ["input_ids", "attention_mask"]}


def create_style_model():
    with patch("fhf.model.AutoModel.from_pretrained", return_value=TinyEncoder()):
        model = FingerprintModel(
            "tiny", lora={"enabled": False}, pooling="attention",
            style={"enabled": True, "hash_dim": 64, "hidden_dim": 12},
            residual_fusion={"enabled": True, "flat_weight": 0.5},
        )
    return model


def test_feature_extraction_deterministic_and_no_fit():
    ex = StyleFeatureExtractor(64)
    texts = ["Hello, world! 1.", "Hello, world! 1.", "- item\n- item"]
    feats = ex.transform(texts)
    assert feats.shape == (3, 64 + len(STYLE_SCALAR_NAMES))
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    np.testing.assert_array_equal(feats[0], feats[1])
    assert not np.array_equal(feats[0], feats[2])
    np.testing.assert_array_equal(ex.transform(texts), feats)


def test_dataset_and_collator_include_style_for_labeled_and_unlabeled():
    df = pd.DataFrame({"id": ["a", "b"], "text": ["Hello.", "Good morning!"],
                       "model": ["gpt-5.5", "gpt-5.6-luna"]})
    tokenizer = TinyTokenizer()
    ds = TextDataset(df, tokenizer, style_config={"enabled": True, "hash_dim": 64})
    batch = make_collator(tokenizer)([ds[0], ds[1]])
    assert batch["style_features"].shape == (2, 64 + len(STYLE_SCALAR_NAMES))
    assert batch["labels"].shape == (2,)
    # Test prediction has no access to labels.
    unlabeled = TextDataset(df.drop(columns="model"), tokenizer,
                            style_config={"enabled": True, "hash_dim": 64})
    batch_unlabeled = make_collator(tokenizer)([unlabeled[0], unlabeled[1]])
    assert "labels" not in batch_unlabeled
    assert batch_unlabeled["style_features"].shape[0] == 2


def test_dual_path_outputs_and_gradient_flow():
    torch.manual_seed(7)
    model = create_style_model()
    extractor = StyleFeatureExtractor(64)
    styles = torch.tensor(extractor.transform(["One, two.", "Three!"]))

    input_ids = torch.tensor([[1, 2, 0], [2, 3, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    outputs = model(input_ids, mask, style_features=styles)
    assert {"log_probs", "flat_log_probs", "hierarchical_log_probs",
            "family_logits", "embeddings", "style_gate_mean"}.issubset(outputs)
    for name in ["log_probs", "flat_log_probs", "hierarchical_log_probs"]:
        assert outputs[name].shape == (2, 18)
        assert torch.allclose(outputs[name].exp().sum(dim=-1), torch.ones(2), atol=1e-5)
    assert outputs["embeddings"].shape == (2, 128)
    assert 0 < float(outputs["style_gate_mean"]) < 1
    assert torch.allclose(
        outputs["log_probs"].exp(),
        0.5 * outputs["flat_log_probs"].exp()
        + 0.5 * outputs["hierarchical_log_probs"].exp(), atol=1e-6
    )
    loss = -outputs["log_probs"][:, 0].mean() - outputs["flat_log_probs"][:, 0].mean()
    loss.backward()
    assert model.style_fusion.style_encoder[1].weight.grad is not None
    assert model.flat_head.weight.grad is not None
    assert model.attention_pool.score.weight.grad is not None


def test_dual_path_requires_style_features():
    model = create_style_model()
    with pytest.raises(ValueError, match="requires raw-text style_features"):
        model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))


def test_padding_attention_pool_ignores_padded_tokens():
    torch.manual_seed(1)
    pool = MaskedAttentionPool(4)
    short = torch.tensor([[[1., 2., 0., 0.], [2., 0., 0., 1.]]])
    long = torch.cat([short, torch.ones(1, 3, 4) * 100], dim=1)
    a = pool(short, torch.tensor([[1, 1]]))
    b = pool(long, torch.tensor([[1, 1, 0, 0, 0]]))
    assert torch.allclose(a, b, atol=1e-6)


def test_original_model_still_has_exact_legacy_outputs():
    with patch("fhf.model.AutoModel.from_pretrained", return_value=TinyEncoder()):
        model = FingerprintModel("tiny", lora={"enabled": False})
    result = model(torch.ones((2, 3), dtype=torch.long),
                   torch.ones((2, 3), dtype=torch.long))
    assert set(result) == {
        "log_probs", "model_logits", "family_logits", "family_log_probs", "embeddings"
    }
