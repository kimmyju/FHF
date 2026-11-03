"""Experiment-configured optimization, checkpoint selection and hierarchical prediction loops."""
import json
import random
import numpy as np
import torch
from torch.nn import functional as F

from .losses import objective, symmetric_kl
from .hotflip import hotflip_batch
from .metrics import report


def seed_everything(seed):
    """Seed Python, NumPy, CPU and CUDA RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, loader, device):
    """Predict precise model labels if available, else provider IDs for family-only ablations."""
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            if "labels" in batch:
                labels.extend((batch["labels"] if model.heads["model"] else batch["families"]).tolist())
                batch.pop("labels")
                batch.pop("families")
            out = model(**{key: val.to(device) for key, val in batch.items()})
            key = "log_probs" if model.heads["model"] else "family_log_probs"
            predictions.extend(out[key].argmax(dim=-1).cpu().tolist())
    return predictions, labels


def _evaluate(predictions, targets, model):
    """Always compute correct label-space metrics, never map providers to arbitrary versions."""
    if model.heads["model"]:
        return report(targets, predictions)
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score
    from .labels import FAMILY_NAMES
    return {
        "n": len(targets),
        "family_balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "family_accuracy": float(accuracy_score(targets, predictions)),
        "recall_by_family": dict(zip(FAMILY_NAMES,
            [float(x) for x in recall_score(targets, predictions,
                                           labels=list(range(len(FAMILY_NAMES))),
                                           average=None, zero_division=0)])),
    }


def fit(model, tokenizer, train_loader, valid_loader, out_dir, config, device="cpu"):
    """Train according to YAML head/loss switches, saving the best prompt-held-out model."""
    settings, losses = config["training"], config["losses"]
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    model.to(device)
    model._fhf_tokenizer = tokenizer
    best, history = -float("inf"), []
    for epoch in range(int(settings["epochs"])):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            labels, families = batch.pop("labels"), batch.pop("families")
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss, _ = objective(out, labels, families, config)
            if losses["rdrop"]["enabled"] and losses["rdrop"]["weight"] > 0:
                second = model(**batch)
                target = "log_probs" if model.heads["model"] else "family_log_probs"
                loss = loss + float(losses["rdrop"]["weight"]) * symmetric_kl(
                    out[target], second[target])
            if losses["hotflip"]["enabled"] and losses["hotflip"]["weight"] > 0:
                sample = dict(batch, labels=labels, families=families)
                changed = hotflip_batch(model, sample, int(losses["hotflip"]["max_examples"]))
                if changed is not None:
                    new = model(input_ids=changed, attention_mask=batch["attention_mask"])
                    target = "log_probs" if model.heads["model"] else "family_log_probs"
                    loss = loss + float(losses["hotflip"]["weight"]) * F.kl_div(
                        new[target], out[target].detach().exp(), reduction="batchmean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip"]))
            optimizer.step()
            total += float(loss.detach())
            count += 1
        preds, truth = predict(model, valid_loader, device)
        metrics = _evaluate(preds, truth, model)
        score = metrics.get("balanced_accuracy_observed_classes", metrics.get("family_balanced_accuracy"))
        history.append({"epoch": epoch + 1, "training_loss": total / max(count, 1),
                        "metrics": metrics})
        print(f"Epoch {epoch+1}: loss={total/max(count,1):.4f} val balanced accuracy={score:.4f}")
        if score > best:
            best = score
            model.save_checkpoint(out_dir, tokenizer, extra={
                "best_epoch": epoch+1, "validation_balanced_accuracy": score,
                "experiment": config["experiment"],
            })
    return history
