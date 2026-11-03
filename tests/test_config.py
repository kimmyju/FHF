"""Test YAML experiment inheritance, override parsing and head/loss compatibility."""
from pathlib import Path
import pytest
from fhf.config import load_config, resolve_path


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,enabled", [
    ("configs/base.yaml", (True, True, True)),
    ("configs/ablations/model_only.yaml", (False, True, False)),
    ("configs/ablations/family_only.yaml", (True, False, False)),
    ("configs/ablations/no_contrastive.yaml", (True, True, False)),
])
def test_preset_heads(name, enabled):
    config = load_config(ROOT / name)
    values = config["model"]["heads"]
    assert (values["family"], values["model"], values["projection"]) == enabled


def test_overrides():
    config = load_config(ROOT / "configs/base.yaml",
                         ["model.encoder=distilbert/distilbert-base-uncased",
                          "training.epochs=3", "losses.hotflip.enabled=true"])
    assert config["model"]["encoder"] == "distilbert/distilbert-base-uncased"
    assert config["training"]["epochs"] == 3
    assert config["losses"]["hotflip"]["enabled"] is True


def test_invalid_head_combination():
    with pytest.raises(ValueError, match="At least one classifier"):
        load_config(ROOT / "configs/base.yaml",
                    ["model.heads.family=false", "model.heads.model=false"])


def test_invalid_family_loss():
    with pytest.raises(ValueError, match="Family loss requires"):
        load_config(ROOT / "configs/base.yaml", ["model.heads.family=false"])


def test_invalid_model_loss():
    with pytest.raises(ValueError, match="Exact classification loss requires"):
        load_config(ROOT / "configs/base.yaml", ["model.heads.model=false"])


def test_resolve_path():
    cfg = load_config(ROOT / "configs/base.yaml")
    assert resolve_path(cfg, "train_csv").as_posix().endswith("data/train.csv")


def test_all_presets_load():
    for preset in (ROOT / "configs").rglob("*.yaml"):
        assert load_config(preset)["experiment"]

def test_modernbert_lora_defaults():
    """Check the production YAML activates correct encoder and PEFT parameters."""
    conf = load_config(ROOT / "configs/base.yaml")
    assert conf["model"]["encoder"] == "answerdotai/ModernBERT-large"
    assert conf["model"]["lora"]["enabled"]
    assert conf["model"]["lora"]["target_modules"] == ["Wqkv", "Wo"]
    assert conf["training"]["gradient_accumulation_steps"] >= 1


def test_invalid_lora_rank():
    with pytest.raises(ValueError, match="LoRA r and alpha"):
        load_config(ROOT / "configs/base.yaml", ["model.lora.r=0"])


def test_dual_path_preset_loads():
    conf = load_config(ROOT / "configs/ablations/dual_path_style.yaml")
    assert conf["model"]["pooling"] == "attention"
    assert conf["model"]["style"]["enabled"] is True
    assert conf["model"]["residual_fusion"]["enabled"] is True
    assert conf["losses"]["flat_classification"]["enabled"] is True
    assert conf["model"]["heads"] == {"family": True, "model": True, "projection": True}


def test_flat_aux_loss_requires_residual_head():
    with pytest.raises(ValueError, match="Flat classification loss"):
        load_config(ROOT / "configs/base.yaml",
                    ["losses.flat_classification.enabled=true"])


def test_style_hotflip_incompatibility_is_explicit():
    with pytest.raises(ValueError, match="regenerating perturbed"):
        load_config(ROOT / "configs/ablations/dual_path_style.yaml",
                    ["losses.hotflip.enabled=true"])


def test_prompt_aware_preset_keeps_dual_path_settings():
    control = load_config(ROOT / "configs/ablations/dual_path_style.yaml")
    prompt = load_config(ROOT / "configs/ablations/dual_path_prompt_aware.yaml")
    assert control["model"] == prompt["model"]
    assert control["training"] == prompt["training"]
    assert prompt["losses"]["intra_contrastive"]["mode"] == "prompt_aware"
    assert control["losses"]["intra_contrastive"]["mode"] == "standard"
    assert prompt["losses"]["intra_contrastive"]["weight"] == (
        control["losses"]["intra_contrastive"]["weight"]
    )
