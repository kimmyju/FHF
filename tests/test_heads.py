"""Check independent model and provider heads without downloading Hugging Face weights."""
import torch
from torch import nn
import pytest
from unittest.mock import patch

from fhf.model import FingerprintModel


class TinyEncoder(nn.Module):
    """Toy encoder implementing the minimal AutoModel contract for head tests."""
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 12})()
        self.embedding = nn.Embedding(40, 12)

    def forward(self, input_ids, attention_mask, **kwargs):
        return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()


@pytest.mark.parametrize("heads,expected", [
    (dict(family=True, model=True, projection=True), {"log_probs", "model_logits", "family_logits", "family_log_probs", "embeddings"}),
    (dict(family=False, model=True, projection=False), {"log_probs", "model_logits"}),
    (dict(family=True, model=False, projection=False), {"family_logits", "family_log_probs"}),
])
def test_head_ablation(heads, expected):
    with patch("fhf.model.AutoModel.from_pretrained", return_value=TinyEncoder()):
        model = FingerprintModel("toy", heads=heads, lora={"enabled":False})
    x = torch.randint(1, 40, (4, 7))
    mask = torch.ones_like(x)
    out = model(x, mask)
    assert set(out) == expected
    key = "log_probs" if "log_probs" in out else "family_log_probs"
    assert torch.allclose(out[key].exp().sum(-1), torch.ones(4), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast requires CUDA")
def test_hierarchical_log_probs_under_bf16_autocast():
    with patch("fhf.model.AutoModel.from_pretrained", return_value=TinyEncoder()):
        model = FingerprintModel(
            "toy", heads=dict(family=True, model=True, projection=False), lora={"enabled": False}
        ).cuda()
    x = torch.randint(1, 40, (4, 7), device="cuda")
    mask = torch.ones_like(x)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x, mask)
    assert torch.isfinite(out["log_probs"]).all()


def test_zero_classifier_head_rejected():
    with pytest.raises(ValueError, match="At least one classification"):
        FingerprintModel("unused", heads=dict(family=False, model=False, projection=True))
