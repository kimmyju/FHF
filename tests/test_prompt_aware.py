"""Prompt-aware supervised contrastive pairs and model-balanced sampling."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from fhf.data import PromptAwareFamilyBatchSampler, TextDataset, make_collator
from fhf.labels import MODEL_FAMILY_IDS, MODEL_NAMES
from fhf.losses import (
    prompt_aware_contrastive_loss, prompt_aware_pair_counts, objective,
)
from fhf.config import load_config
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def synthetic_prompts(num_prompts=12):
    labels = [m for p in range(num_prompts) for m in range(18)]
    prompts = [p for p in range(num_prompts) for m in range(18)]
    return labels, prompts


def test_balanced_sampler_constructs_positive_and_hard_negative_pairs():
    labels, prompts = synthetic_prompts()
    sampler = PromptAwareFamilyBatchSampler(
        labels, prompts, batch_families=2, models_per_family=2,
        samples_per_model=2, steps_per_epoch=27, seed=42,
    )
    first = list(sampler)
    assert len(first) == 27
    assert all(len(batch) == 8 for batch in first)
    counts = Counter(labels[i] for batch in first for i in batch)
    assert set(counts) == set(range(18))
    assert max(counts.values()) - min(counts.values()) <= 2

    for batch in first:
        models = [labels[i] for i in batch]
        ps = [prompts[i] for i in batch]
        f = [MODEL_FAMILY_IDS[m] for m in models]
        assert len(set(f)) == 2
        positives, hard = prompt_aware_pair_counts(
            torch.tensor(models), torch.tensor(ps),
        )
        assert positives >= 4  # every chosen model has a different-prompt positive
        assert hard >= 1  # anchor prompt shared by different models
        for model in set(models):
            assert len(set(ps[j] for j, m in enumerate(models) if m == model)) >= 2

    sampler.set_epoch(0)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_prompt_sampler_requires_two_prompts_per_model():
    with pytest.raises(ValueError, match=">=2 distinct prompts"):
        PromptAwareFamilyBatchSampler([0, 0, 5, 5], [0, 0, 0, 1],
                                      batch_families=2, models_per_family=1,
                                      samples_per_model=2)


def test_prompt_sampler_requires_shared_prompt_across_models():
    with pytest.raises(ValueError, match="shared prompt"):
        PromptAwareFamilyBatchSampler([0, 0, 5, 5], ["a", "b", "c", "d"],
                                      batch_families=2, models_per_family=1,
                                      samples_per_model=2)


def test_prompt_loss_requires_prompt_ids_and_weighting_changes_loss():
    torch.manual_seed(4)
    raw = torch.randn(4, 10, requires_grad=True)
    embeddings = F.normalize(raw, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    families = torch.zeros(4, dtype=torch.long)
    prompts = torch.tensor([0, 1, 0, 1])
    positives, hard = prompt_aware_pair_counts(labels, prompts)
    assert positives == 2
    assert hard == 2
    low = prompt_aware_contrastive_loss(
        embeddings, labels, families, prompts,
        temperature=0.12, family_weight=2.0, same_prompt_weight=1.0,
    )
    high = prompt_aware_contrastive_loss(
        embeddings, labels, families, prompts,
        temperature=0.12, family_weight=2.0, same_prompt_weight=3.0,
    )
    assert torch.isfinite(high)
    assert high > low
    high.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    with pytest.raises(ValueError, match="prompt_ids"):
        prompt_aware_contrastive_loss(embeddings, labels, families, None)


def test_same_model_same_prompt_is_not_positive_or_negative():
    embeddings = F.normalize(torch.randn(4, 12), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    families = torch.tensor([0, 0, 1, 1])
    prompts = torch.tensor([1, 1, 1, 1])
    n_pos, n_neg = prompt_aware_pair_counts(labels, prompts)
    assert n_pos == 0
    assert n_neg == 4
    loss = prompt_aware_contrastive_loss(
        embeddings, labels, families, prompts,
    )
    assert float(loss) == pytest.approx(0.0)


def test_training_objective_switches_to_prompt_aware():
    conf = load_config(ROOT / "configs/ablations/dual_path_prompt_aware.yaml")
    labels = torch.tensor([0, 0, 1, 1])
    families = torch.zeros(4, dtype=torch.long)
    prompts = torch.tensor([0, 1, 0, 1])
    logits = torch.randn(4, 18, requires_grad=True)
    flat = torch.randn(4, 18, requires_grad=True)
    fam_logits = torch.randn(4, 6, requires_grad=True)
    embed_raw = torch.randn(4, 128, requires_grad=True)
    out = {
        "log_probs": F.log_softmax(logits, dim=-1),
        "flat_log_probs": F.log_softmax(flat, dim=-1),
        "family_logits": fam_logits,
        "embeddings": F.normalize(embed_raw, dim=-1),
    }
    value, components = objective(
        out, labels, families, conf, prompt_ids=prompts,
    )
    assert "intra_contrastive" in components
    assert "flat_classification" in components
    value.backward()
    assert embed_raw.grad is not None
    assert torch.isfinite(embed_raw.grad).all()
    with pytest.raises(ValueError, match="prompt_ids"):
        objective(out, labels, families, conf)


class MiniTokenizer:
    def __call__(self, text, truncation=True, max_length=384):
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def pad(self, items, padding=True, return_tensors="pt"):
        return {key: torch.tensor([i[key] for i in items])
                for key in ("input_ids", "attention_mask")}


def test_dataset_prompt_ids_are_collated_but_not_tokenized():
    rows = pd.DataFrame({
        "id": ["a", "b", "c"], "model": [MODEL_NAMES[0]] * 3,
        "text": ["a", "b", "c"], "prompt_id": ["p0", "p1", "p0"],
    })
    t = MiniTokenizer()
    ds = TextDataset(rows, t, include_prompt_id=True)
    batch = make_collator(t)([ds[0], ds[1], ds[2]])
    assert "prompt_ids" in batch
    assert batch["prompt_ids"].tolist() == [0, 1, 0]
    assert batch["input_ids"].shape == (3, 2)
    assert "prompt_ids" not in TextDataset(rows, t)[0]


def test_invalid_prompt_aware_sampler_config():
    with pytest.raises(ValueError, match="samples_per_model"):
        load_config(
            ROOT / "configs/ablations/dual_path_prompt_aware.yaml",
            ["training.samples_per_model=1"],
        )
