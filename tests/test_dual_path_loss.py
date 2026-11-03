"""Independent flat branch receives its own supervised 18-class loss."""
import torch
import torch.nn.functional as F

from fhf.config import load_config
from fhf.losses import objective
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dual_path_objective_trains_independent_flat_head():
    conf = load_config(ROOT / "configs/ablations/dual_path_style.yaml")
    probs = torch.randn(4, 18, requires_grad=True)
    flat_logits = torch.randn(4, 18, requires_grad=True)
    family = torch.randn(4, 6, requires_grad=True)
    embeddings = F.normalize(torch.randn(4, 128), dim=-1)
    labels = torch.tensor([0, 0, 5, 5])
    families = torch.tensor([0, 0, 1, 1])
    out = {
        "log_probs": F.log_softmax(probs, dim=-1),
        "flat_log_probs": F.log_softmax(flat_logits, dim=-1),
        "family_logits": family,
        "embeddings": embeddings,
    }
    loss, parts = objective(out, labels, families, conf)
    assert "flat_classification" in parts
    loss.backward()
    assert flat_logits.grad is not None
    assert probs.grad is not None
    assert family.grad is not None
