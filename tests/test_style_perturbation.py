"""Style-space perturbation sanity and configuration checks."""
from pathlib import Path

import pytest
import torch
from fhf.style import GatedStyleFusion
from fhf.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_style_noise_changes_training_fusion_only():
    torch.manual_seed(8)
    mod = GatedStyleFusion(12, 8, style_hidden=5, dropout=0.0)
    ctx = torch.randn(4, 8)
    style = torch.randn(4, 12)
    mod.train()
    clean, _ = mod(ctx, style, noise_std=0)
    noisy, _ = mod(ctx, style, noise_std=0.05)
    assert clean.shape == noisy.shape
    assert not torch.allclose(clean, noisy)
    mod.eval()
    a, _ = mod(ctx, style, noise_std=0)
    b, _ = mod(ctx, style, noise_std=0.05)
    torch.testing.assert_close(a, b)


def test_new_preset_preserves_within_family_loss_and_validates_noise():
    new = load_config(ROOT / "configs/ablations/dual_path_within_family_style_perturb.yaml")
    old = load_config(ROOT / "configs/ablations/dual_path_within_family.yaml")
    assert new["model"] == old["model"]
    assert new["training"] == old["training"]
    assert new["losses"] == old["losses"]
    assert new["losses"]["intra_contrastive"]["mode"] == "within_family"
    assert new["style_perturbation"] == {"enabled": True, "sigma": 0.05, "weight": 0.10}
    with pytest.raises(ValueError, match="sigma"):
        load_config(ROOT / "configs/ablations/dual_path_within_family_style_perturb.yaml",
                    ["style_perturbation.sigma=0"])
