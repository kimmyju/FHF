"""Test sweep YAML separation, metric compatibility, and unique output paths."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from fhf.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,metric,base", [
    ("lora_bayes.yaml", "val/best_balanced_accuracy", "configs/base.yaml"),
    ("loss_grid.yaml", "val/best_balanced_accuracy", "configs/base.yaml"),
    ("model_only_bayes.yaml", "val/best_balanced_accuracy", "configs/ablations/model_only.yaml"),
    ("family_only_bayes.yaml", "val/best_family_balanced_accuracy", "configs/ablations/family_only.yaml"),
])
def test_sweep_yaml(name, metric, base):
    sweep = yaml.safe_load((ROOT / "sweeps" / name).read_text())
    assert sweep["program"] == "scripts/sweep_train.py"
    assert sweep["metric"]["name"] == metric
    assert sweep["metric"]["goal"] == "maximize"
    assert sweep["command"][-1] == base
    assert all("." in name for name in sweep["parameters"])
    config = load_config(ROOT / base)
    assert config["model"]["lora"]["enabled"]


def test_sweep_paths_are_isolated():
    spec = importlib.util.spec_from_file_location(
        "fhf_sweep_runner", ROOT / "scripts" / "sweep_train.py")
    module = importlib.util.module_from_spec(spec)
    # scripts/train.py is importable when the runner is invoked from scripts/.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    first = load_config(ROOT / "configs/base.yaml")
    second = load_config(ROOT / "configs/base.yaml")
    a = module.isolate_outputs(first, "run-a")
    b = module.isolate_outputs(second, "run-b")
    assert a != b
    assert first["paths"]["checkpoint_dir"] != second["paths"]["checkpoint_dir"]
    assert first["paths"]["train_csv"] == second["paths"]["train_csv"]
    with pytest.raises(ValueError):
        module.isolate_outputs(second, "../unsafe")
