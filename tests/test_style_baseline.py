"""Style-only baseline must use prompt-disjoint validation and save 18-class scores."""
import json
from pathlib import Path

import pandas as pd
import pytest

from fhf.config import load_config
from fhf.labels import MODEL_NAMES
from scripts.style_baseline import load_matching_split, run_style_baseline

ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path):
    config = load_config(ROOT / "configs/ablations/longer_training.yaml")
    config["paths"]["base_dir"] = str(tmp_path)
    config["paths"]["train_csv"] = "train.csv"
    config["paths"]["train_split_csv"] = "train_split.csv"
    config["paths"]["validation_csv"] = "val.csv"
    config["training"]["validation_fraction"] = 0.25
    config["seed"] = 42
    return config


def make_data():
    rows = []
    for prompt in range(8):
        for model_index, name in enumerate(MODEL_NAMES):
            rows.append({
                "id": f"p{prompt}-m{model_index}",
                "prompt_id": f"p{prompt}",
                "model": name,
                "text": f"Text output with author style {name} marker_{model_index} "
                        f"punctuation. Prompt number {prompt}.",
            })
    return pd.DataFrame(rows)


def test_style_baseline_can_regenerate_split_and_save_outputs(tmp_path):
    config = make_config(tmp_path)
    make_data().to_csv(tmp_path / "train.csv", index=False)
    result = run_style_baseline(
        config, output_dir=tmp_path / "style",
        ngram_range=(2, 3), max_features=3000, min_df=1, c=1.0,
    )
    assert result["n"] == 36
    assert result["num_train"] == 108
    assert result["split_source"] == "regenerated_prompt_disjoint_split"
    assert 0 <= result["balanced_accuracy_observed_classes"] <= 1
    assert (tmp_path / "style" / "style_model.joblib").is_file()
    assert (tmp_path / "style" / "evaluation.json").is_file()
    values = json.loads((tmp_path / "style" / "evaluation.json").read_text())
    assert values["n"] == 36
    predictions = pd.read_csv(tmp_path / "style" / "val_predictions.csv")
    score_columns = [f"prob_{name}" for name in MODEL_NAMES]
    assert set(score_columns).issubset(predictions.columns)
    assert len(predictions) == 36
    assert all(abs(x - 1.0) < 1e-5 for x in predictions[score_columns].sum(axis=1))


def test_style_baseline_uses_existing_saved_splits(tmp_path):
    config = make_config(tmp_path)
    data = make_data()
    train = data[data.prompt_id != "p7"]
    valid = data[data.prompt_id == "p7"]
    train.to_csv(tmp_path / "train_split.csv", index=False)
    valid.to_csv(tmp_path / "val.csv", index=False)
    a, b, source = load_matching_split(config)
    assert source == "saved_fhf_split"
    assert len(a) == len(train)
    assert len(b) == len(valid)


def test_style_baseline_rejects_prompt_leakage(tmp_path):
    config = make_config(tmp_path)
    data = make_data()
    data.to_csv(tmp_path / "train_split.csv", index=False)
    data[data.prompt_id == "p7"].to_csv(tmp_path / "val.csv", index=False)
    with pytest.raises(ValueError, match="Prompt leakage"):
        load_matching_split(config)


def test_style_baseline_rejects_missing_half_split(tmp_path):
    config = make_config(tmp_path)
    make_data().to_csv(tmp_path / "train_split.csv", index=False)
    with pytest.raises(FileNotFoundError, match="Only one saved split"):
        load_matching_split(config)
