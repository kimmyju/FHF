"""Isolate within-family contrastive signal from cross-family negatives."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fhf.config import load_config
from fhf.data import PromptAwareFamilyBatchSampler
from fhf.labels import MODEL_FAMILY_IDS
from fhf.losses import (
    within_family_prompt_contrastive_loss, within_family_pair_counts, objective,
)


ROOT = Path(__file__).resolve().parents[1]


def test_within_family_contrastive_ignores_other_families():
    torch.manual_seed(55)
    # 0 and 1 are OpenAI, 5 is Google. Same-family negatives
    # for the two OpenAI models; Google has no within-family negative.
    labels = torch.tensor([0, 0, 1, 1, 5, 5])
    families = torch.tensor([0, 0, 0, 0, 1, 1])
    prompts = torch.tensor([0, 1, 0, 1, 0, 1])
    original = F.normalize(torch.randn(6, 16), dim=-1)
    base = within_family_prompt_contrastive_loss(original, labels, families, prompts)
    modified = original.clone()
    modified[4:] = F.normalize(torch.randn(2, 16) * 40, dim=-1)
    changed = within_family_prompt_contrastive_loss(modified, labels, families, prompts)
    assert base.item() == pytest.approx(changed.item(), abs=1e-6)
    assert torch.isfinite(base)


def test_within_family_gradient_and_no_singleton_gradient():
    torch.manual_seed(56)
    labels = torch.tensor([0, 0, 1, 1, 5, 5])
    families = torch.tensor([0, 0, 0, 0, 1, 1])
    prompts = torch.tensor([0, 1, 0, 1, 0, 1])
    raw = torch.randn(6, 8, requires_grad=True)
    z = F.normalize(raw, dim=-1)
    loss = within_family_prompt_contrastive_loss(z, labels, families, prompts)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[:4].abs().sum() > 0
    assert raw.grad[4:].abs().sum() == 0
    anchors, neg_pairs = within_family_pair_counts(labels, families, prompts)
    assert anchors == 4
    assert neg_pairs == 4


def test_within_family_excludes_same_model_same_prompt_and_zero_pair_cases():
    labels = torch.tensor([0, 0, 1, 1])
    families = torch.zeros(4, dtype=torch.long)
    # No valid positives: same prompt duplicates are NOT positives.
    prompts = torch.tensor([0, 0, 1, 1])
    z = F.normalize(torch.randn(4, 8, requires_grad=True), dim=-1)
    loss = within_family_prompt_contrastive_loss(z, labels, families, prompts)
    assert loss.item() == pytest.approx(0.0)
    assert within_family_pair_counts(labels, families, prompts) == (0, 4)

    # No same-family negatives at all: skip even with positives.
    labels = torch.tensor([0, 0, 5, 5])
    families = torch.tensor([0, 0, 1, 1])
    prompts = torch.tensor([0, 1, 0, 1])
    assert within_family_prompt_contrastive_loss(
        z, labels, families, prompts
    ).item() == pytest.approx(0.0)


def test_within_family_is_invariant_to_cross_family_negative_similarity():
    # Construct two OpenAI models and one Google model with arbitrarily
    # similar embeddings; outside family must never enter loss denominator.
    z = F.normalize(torch.tensor([
        [1., 0.], [1., 0.], [.2, .98], [.2, .98], [1., 0.], [1., 0.],
    ]), dim=-1)
    label = torch.tensor([0, 0, 1, 1, 5, 5])
    family = torch.tensor([0, 0, 0, 0, 1, 1])
    prompt = torch.tensor([0, 1, 0, 1, 0, 1])
    x = within_family_prompt_contrastive_loss(z, label, family, prompt)
    y = within_family_prompt_contrastive_loss(
        z, label, family, prompt, same_prompt_weight=5
    )
    assert y >= x


def test_within_family_preset_keeps_original_hyperparameters_and_sampler():
    cfg = load_config(ROOT / "configs/ablations/dual_path_within_family.yaml")
    baseline = load_config(ROOT / "configs/ablations/dual_path_prompt_aware.yaml")
    assert cfg["losses"]["intra_contrastive"]["mode"] == "within_family"
    assert cfg["training"] == baseline["training"]
    assert cfg["model"] == baseline["model"]
    assert cfg["paths"]["checkpoint_dir"] != baseline["paths"]["checkpoint_dir"]
    assert cfg["losses"]["intra_contrastive"]["weight"] == baseline["losses"]["intra_contrastive"]["weight"]
    assert cfg["losses"]["intra_contrastive"]["temperature"] == baseline["losses"]["intra_contrastive"]["temperature"]

    labels = [m for p in range(3) for m in range(18)]
    prompts = [p for p in range(3) for m in range(18)]
    sampler = PromptAwareFamilyBatchSampler(
        labels, prompts, batch_families=2, models_per_family=2,
        samples_per_model=2, steps_per_epoch=6, seed=42,
    )
    assert len(list(sampler)) == 6


def test_objective_dispatches_within_family_and_requires_prompts():
    cfg = load_config(ROOT / "configs/ablations/dual_path_within_family.yaml")
    labels = torch.tensor([0, 0, 1, 1])
    families = torch.tensor([0, 0, 0, 0])
    prompts = torch.tensor([0, 1, 0, 1])
    embedding = F.normalize(torch.randn(4, 8, requires_grad=True), dim=-1)
    logits = torch.randn(4, 18, requires_grad=True)
    outputs = {
        "embeddings": embedding,
        "log_probs": F.log_softmax(logits, dim=-1),
        "flat_log_probs": F.log_softmax(torch.randn(4, 18, requires_grad=True), dim=-1),
        "family_logits": torch.randn(4, 6, requires_grad=True),
    }
    total, parts = objective(outputs, labels, families, cfg, prompt_ids=prompts)
    assert "intra_contrastive" in parts
    assert total.item() > 0
    with pytest.raises(ValueError, match="prompt_ids"):
        objective(outputs, labels, families, cfg, prompt_ids=None)
    total.backward()
    assert torch.isfinite(logits.grad).all()
