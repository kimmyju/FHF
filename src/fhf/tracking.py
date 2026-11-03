"""Optional Weights & Biases tracking for loss curves, attribution metrics, and confusion matrices.

No API keys or credentials are stored in this module. Set WANDB_API_KEY or use
'wandb login' when switching from the default offline mode to online mode.
"""
from __future__ import annotations

from pathlib import Path
from .labels import MODEL_NAMES, FAMILY_NAMES


def validation_logs(metrics, model_has_version_head: bool):
    """Normalize metric names for scalar charts and W&B sweep optimization."""
    result = {}
    if model_has_version_head:
        result["val/balanced_accuracy"] = metrics["balanced_accuracy_observed_classes"]
        result["val/balanced_accuracy_18_zero_missing"] = metrics[
            "balanced_accuracy_18_classes_zero_for_missing"]
        result["val/accuracy"] = metrics["accuracy"]
        result["val/family_accuracy"] = metrics["family_accuracy"]
        if metrics.get("within_family_accuracy_given_correct_family") is not None:
            result["val/within_family_accuracy"] = metrics[
                "within_family_accuracy_given_correct_family"]
        for model, value in metrics["recall_by_model"].items():
            result[f"val/recall/{model}"] = value
    else:
        result["val/family_balanced_accuracy"] = metrics["family_balanced_accuracy"]
        result["val/family_accuracy"] = metrics["family_accuracy"]
        for family, value in metrics["recall_by_family"].items():
            result[f"val/recall/{family.replace(' ', '_')}"] = value
    return result


class WandbTracker:
    """Adapt optional wandb.Run to FHF while keeping training independent of login."""

    def __init__(self, config):
        settings = config.get("wandb", {})
        self.enabled = bool(settings.get("enabled", False))
        self.run = None
        self._owns_run = False
        self.log_confusion_matrix = bool(settings.get("log_confusion_matrix", True))
        if not self.enabled:
            return
        import wandb

        self._wandb = wandb
        if wandb.run is not None:  # A wandb sweep agent already initialized this run.
            self.run = wandb.run
            self.run.config.update({"resolved_experiment": config}, allow_val_change=True)
        else:
            directory = Path(settings.get("dir", "outputs/wandb")).expanduser()
            if not directory.is_absolute():
                directory = Path(config["paths"].get("base_dir", ".")) / directory
            directory.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=settings.get("project", "fhf-name-that-llm"),
                entity=settings.get("entity") or None,
                name=settings.get("run_name") or config.get("experiment"),
                group=settings.get("group") or None,
                job_type="train",
                tags=list(settings.get("tags") or []),
                mode=settings.get("mode", "offline"),
                dir=str(directory),
                config={"resolved_experiment": config},
            )
            self._owns_run = True
        self.run.define_metric("train/optimizer_step")
        self.run.define_metric("train/*", step_metric="train/optimizer_step")
        self.run.define_metric("epoch")
        self.run.define_metric("val/*", step_metric="epoch")
        self.run.define_metric("epoch_train/*", step_metric="epoch")

    def train_step(self, optimizer_step, metrics):
        """Log optimizer-step loss components and batch accuracy as line charts."""
        if self.enabled:
            self.run.log({"train/optimizer_step": optimizer_step, **metrics})

    def epoch_train(self, epoch, loss, accuracy):
        """Log epoch aggregates on the same x-axis as validation metrics."""
        if self.enabled:
            self.run.log({
                "epoch": epoch,
                "epoch_train/loss": float(loss),
                "epoch_train/accuracy": float(accuracy),
            })

    def validation(self, epoch, metrics, predictions, targets, model_has_version_head):
        """Log validation scores, model recalls and optional 18x18 confusion matrix."""
        if not self.enabled:
            return
        values = validation_logs(metrics, model_has_version_head)
        payload = {
            "epoch": epoch,
            **values,
            "val/loss": float(metrics["validation_loss"]),
        }
        if metrics.get("validation_exact_loss") is not None:
            payload["val/loss_exact"] = float(metrics["validation_exact_loss"])
        if metrics.get("validation_family_loss") is not None:
            payload["val/loss_family"] = float(metrics["validation_family_loss"])
        if self._wandb and model_has_version_head:
            labels = MODEL_NAMES
        else:
            labels = FAMILY_NAMES
        if targets and self.log_confusion_matrix:
            # W&B Table/Plot is logged under a separate key and remains viewable per epoch.
            payload["val/confusion_matrix"] = self._wandb.plot.confusion_matrix(
                probs=None, y_true=targets, preds=predictions, class_names=labels)
        self.run.log(payload)

    def best(self, epoch, name, value):
        """Store best held-out score and corresponding epoch in run summary."""
        if self.enabled:
            self.run.summary["best/epoch"] = epoch
            self.run.summary[f"best/{name}"] = value

    def best_score(self, epoch, score, model_has_version_head):
        """Log the running best score under the exact key Sweep optimizes."""
        if not self.enabled:
            return
        key = ("val/best_balanced_accuracy" if model_has_version_head
               else "val/best_family_balanced_accuracy")
        self.run.log({"epoch": epoch, key: float(score)})
        self.run.summary[key] = float(score)

    def finish(self):
        """Only finish runs created by this tracker, not a parent sweep wrapper."""
        if self.enabled and self._owns_run:
            self.run.finish()
