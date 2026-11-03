"""Verify balanced accuracy and perfect-prediction metric calculation."""
from fhf.metrics import report


def test_report_perfect():
    metrics=report([0,0,1,1],[0,0,1,1])
    assert metrics['balanced_accuracy_observed_classes']==1
    assert metrics['family_accuracy']==1
    assert metrics['within_family_accuracy_given_correct_family']==1
    assert len(metrics['confusion_matrix'])==18


def test_oracle_within_family_accuracy_uses_all_samples():
    """Oracle accuracy must not depend on whether the Family Head was correct."""
    import torch
    from fhf.metrics import oracle_within_family_predictions
    from fhf.labels import MODEL_FAMILY_IDS

    # True labels: 0, 1 (OpenAI), 5 (Google).
    # Ordinary predictions have two wrong families and one wrong model.
    true = [0, 1, 5]
    normal = [5, 5, 0]
    logits = torch.full((3, 18), -10.0)
    # Example 1: Google wins globally, but OpenAI oracle picks exact model 0.
    logits[0, 5], logits[0, 0] = 10.0, 8.0
    # Example 2: Google wins globally, oracle OpenAI picks the wrong sibling (0).
    logits[1, 5], logits[1, 0], logits[1, 1] = 10.0, 8.0, 7.0
    # Example 3: OpenAI wins globally; oracle Google picks exact model 5.
    logits[2, 0], logits[2, 5] = 10.0, 8.0
    oracle = oracle_within_family_predictions(
        logits, [MODEL_FAMILY_IDS[i] for i in true], MODEL_FAMILY_IDS
    ).tolist()
    assert oracle == [0, 0, 5]
    measured = report(true, normal, oracle_predictions=oracle)
    assert measured["oracle_within_family_accuracy"] == 2 / 3
    assert measured["within_family_accuracy_given_correct_family"] is None
    assert measured["accuracy"] == 0.0


def test_oracle_rejects_invalid_family_mask():
    import pytest
    import torch
    from fhf.metrics import oracle_within_family_predictions
    from fhf.labels import MODEL_FAMILY_IDS

    with pytest.raises(ValueError, match="no corresponding model"):
        oracle_within_family_predictions(
            torch.zeros(2, 18), [999, 0], MODEL_FAMILY_IDS
        )


def test_oracle_predictions_must_be_in_true_family():
    import pytest
    with pytest.raises(ValueError, match="within each true family"):
        report([0, 1], [0, 1], oracle_predictions=[5, 1])
