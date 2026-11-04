"""Coverage and configuration regressions."""
from pathlib import Path
import pytest
from fhf.full_coverage import FullCoveragePromptFamilyBatchSampler
from fhf.config import load_config

def test_coverage():
    labels = [0]*13 + [1]*13 + [2]*14 + [5]*5 + [6]*8
    prompts = [f"prompt-{i%8}" for i in range(len(labels))]
    sampler = FullCoveragePromptFamilyBatchSampler(
        labels, prompts, batch_families=2, models_per_family=2,
        samples_per_model=4, seed=42)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    all_rows = [i for b in batches for i in b]
    assert sorted(all_rows) == list(range(len(labels)))
    assert max(map(len,batches)) <= 16
    assert list(sampler) == batches
    sampler.set_epoch(1)
    assert sorted(i for b in sampler for i in b) == list(range(len(labels)))

def test_refuse_fixed_steps():
    with pytest.raises(ValueError, match="steps_per_epoch=null"):
        FullCoveragePromptFamilyBatchSampler([0,0], ["a","b"], steps_per_epoch=1)

def test_configs():
    root=Path(__file__).resolve().parents[1]/"configs/ablations"
    for n in (3495,3541):
        conf=load_config(root/f"within_family_style_perturb_openpool_full_{n}.yaml")
        assert conf["training"]["steps_per_epoch"] is None
        assert conf["training"]["sampler"] == "full_coverage"
        assert conf["training"]["batch_families"]*conf["training"]["models_per_family"]*conf["training"]["samples_per_model"] == 32
        assert conf["style_perturbation"]["enabled"]
    e1=load_config(root/"within_family_style_perturb_openpool_e1.yaml")
    assert e1["training"]["steps_per_epoch"] is None
