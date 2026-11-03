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
