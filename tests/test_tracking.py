"""Validate W&B chart metric names without requiring a network connection."""
from fhf.tracking import WandbTracker, validation_logs


def test_disabled_tracker_requires_no_credentials():
    """Disabled tracking must not instantiate wandb or need an API key."""
    tracker = WandbTracker({"wandb": {"enabled": False}})
    tracker.train_step(1, {"train/loss": 1.0})
    tracker.validation(1, {}, [], [], True)
    tracker.best(1, "balanced_accuracy", 0.5)
    tracker.best_score(1, 0.5, True)
    tracker.finish()
    assert tracker.run is None


def test_exact_model_metric_names():
    metrics = {
        "balanced_accuracy_observed_classes": 0.71,
        "balanced_accuracy_18_classes_zero_for_missing": 0.69,
        "accuracy": 0.73,
        "family_accuracy": 0.87,
        "within_family_accuracy_given_correct_family": 0.84,
        "recall_by_model": {"gpt-5.5": 0.8},
    }
    result = validation_logs(metrics, True)
    assert result["val/balanced_accuracy"] == 0.71
    assert result["val/recall/gpt-5.5"] == 0.8
    assert result["val/family_accuracy"] == 0.87


def test_family_only_metrics():
    result = validation_logs({
        "family_balanced_accuracy": 0.9,
        "family_accuracy": 0.93,
        "recall_by_family": {"OpenAI": 0.95},
    }, False)
    assert result["val/family_balanced_accuracy"] == 0.9
    assert result["val/recall/OpenAI"] == 0.95
