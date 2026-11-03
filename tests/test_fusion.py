"""Probability fusion diagnostics must align IDs and class probability columns."""
import numpy as np
import pandas as pd
import pytest

from fhf.fusion import evaluate_fusion
from fhf.labels import MODEL_NAMES


def make_data():
    validation = pd.DataFrame({
        "id": ["x0", "x1", "x2", "x3"],
        "prompt_id": ["p0", "p1", "p2", "p3"],
        "model": [MODEL_NAMES[0], MODEL_NAMES[1], MODEL_NAMES[2], MODEL_NAMES[3]],
    })
    bert = np.full((4, 18), 0.0)
    style = np.full((4, 18), 0.0)
    # BERT correct: x0, x1. Style correct: x1, x2.
    for row, label in enumerate([0, 1, 1, 5]):
        bert[row, label] = 1.0
    for row, label in enumerate([5, 1, 2, 6]):
        style[row, label] = 1.0
    data = validation.rename(columns={"model": "true_model"}).copy()
    data["pred_model"] = [MODEL_NAMES[i] for i in style.argmax(axis=1)]
    for i, name in enumerate(MODEL_NAMES):
        data[f"prob_{name}"] = style[:, i]
    return validation, bert, data


def test_fusion_metrics_and_oracle_correctness():
    validation, bert, style = make_data()
    metrics, predictions, grid = evaluate_fusion(
        validation, bert, style, bert_weights=[1.0, 0.5, 0.0]
    )
    assert metrics["n"] == 4
    assert metrics["both_correct"] == 1
    assert metrics["bert_only_correct"] == 1
    assert metrics["style_only_correct"] == 1
    assert metrics["both_wrong"] == 1
    assert metrics["oracle_ensemble_accuracy"] == pytest.approx(0.75)
    assert metrics["oracle_ensemble_balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["prediction_agreement"] == pytest.approx(0.25)
    assert metrics["bert"]["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["style"]["balanced_accuracy"] == pytest.approx(0.5)
    assert grid["bert_weight"].tolist() == [1.0, 0.5, 0.0]
    assert len(predictions) == 4


def test_fusion_aligns_shuffled_rows_and_columns():
    validation, bert, style = make_data()
    permuted = style.iloc[[2, 0, 3, 1]][list(style.columns)[::-1]]
    metrics, predictions, _ = evaluate_fusion(validation, bert, permuted)
    assert metrics["style"]["balanced_accuracy"] == pytest.approx(0.5)
    assert predictions["id"].tolist() == validation["id"].tolist()
    assert predictions["pred_style"].iloc[2] == MODEL_NAMES[2]


def test_fusion_refuses_wrong_ids():
    validation, bert, style = make_data()
    style.loc[0, "id"] = "unknown"
    with pytest.raises(ValueError, match="sample IDs differ"):
        evaluate_fusion(validation, bert, style)


def test_fusion_refuses_wrong_ground_truth():
    validation, bert, style = make_data()
    style.loc[0, "true_model"] = MODEL_NAMES[8]
    with pytest.raises(ValueError, match="ground-truth labels"):
        evaluate_fusion(validation, bert, style)


def test_fusion_refuses_wrong_prompt():
    validation, bert, style = make_data()
    style.loc[0, "prompt_id"] = "unseen"
    with pytest.raises(ValueError, match="prompt_id"):
        evaluate_fusion(validation, bert, style)


def test_fusion_refuses_duplicate_ids():
    validation, bert, style = make_data()
    style.loc[1, "id"] = style.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicate IDs"):
        evaluate_fusion(validation, bert, style)


def test_fusion_refuses_invalid_probabilities():
    validation, bert, style = make_data()
    bert[0, :] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_fusion(validation, bert, style)
    validation, bert, style = make_data()
    style.loc[0, f"prob_{MODEL_NAMES[0]}"] = 0.8
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_fusion(validation, bert, style)


def test_fusion_refuses_missing_class_column_and_bad_weight():
    validation, bert, style = make_data()
    with pytest.raises(ValueError, match="missing"):
        evaluate_fusion(validation, bert, style.drop(columns=f"prob_{MODEL_NAMES[0]}"))
    with pytest.raises(ValueError, match="weights"):
        evaluate_fusion(validation, bert, style, bert_weights=[1.2])
