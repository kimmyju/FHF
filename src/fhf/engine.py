"""LoRA-efficient FHF training with mixed precision, gradient accumulation and offline prediction."""
from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from .losses import objective, symmetric_kl
from .hotflip import hotflip_batch
from .metrics import report


def seed_everything(seed):
    """Seed CPU, CUDA, Python and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, loader, device):
    """Return predicted exact-model IDs or provider IDs and optional true labels."""
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            if "labels" in batch:
                labels.extend((batch["labels"] if model.heads["model"] else
                               batch["families"]).tolist())
                batch.pop("labels")
                batch.pop("families")
            out = model(**{key: value.to(device) for key, value in batch.items()})
            key = "log_probs" if model.heads["model"] else "family_log_probs"
            predictions.extend(out[key].argmax(dim=-1).cpu().tolist())
    return predictions, labels


def _evaluate(predictions, targets, model):
    """Use the correct label space for 18-class versus provider-only ablations."""
    if model.heads["model"]:
        return report(targets, predictions)
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score
    from .labels import FAMILY_NAMES
    return {
        "n": len(targets),
        "family_balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "family_accuracy": float(accuracy_score(targets, predictions)),
        "recall_by_family": dict(zip(FAMILY_NAMES, [
            float(x) for x in recall_score(
                targets, predictions, labels=list(range(len(FAMILY_NAMES))),
                average=None, zero_division=0)
        ])),
    }


def _precision(device, setting):
    """Select bf16/fp16 only when supported; never request autocast on CPU."""
    if not str(device).startswith("cuda"):
        if setting in ("fp16", "bf16"):
            raise ValueError("CUDA is required for fp16/bf16 mixed precision")
        return None
    if setting == "auto":
        setting = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if setting == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16; use fp16 or auto")
        return torch.bfloat16
    if setting == "fp16":
        return torch.float16
    if setting == "none":
        return None
    raise ValueError(f"Unsupported mixed precision mode: {setting}")


def fit(model, tokenizer, train_loader, valid_loader, out_dir, config, device="cpu"):
    """Train adapter and heads with weighted losses; retain best prompt-disjoint checkpoint."""
    settings, losses = config["training"], config["losses"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable model parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]))
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.to(device)
    model._fhf_tokenizer = tokenizer
    precision = _precision(device, settings.get("mixed_precision", "none"))
    use_fp16 = precision == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {trainable_params:,}/{total_params:,}"
          f" ({100 * trainable_params / max(total_params,1):.2f}%)")
    best, history = -float("inf"), []
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            labels, families = batch.pop("labels"), batch.pop("families")
            # Re-scale the last partial accumulation window to preserve mean gradients.
            remainder = len(train_loader) % accumulation
            divisor = remainder if remainder and step >= (len(train_loader) - remainder) else accumulation
            autocast = (torch.autocast(device_type="cuda", dtype=precision)
                        if precision is not None else nullcontext())
            with autocast:
                out = model(**batch)
                loss, _ = objective(out, labels, families, config)
                if losses["rdrop"]["enabled"] and losses["rdrop"]["weight"] > 0:
                    second = model(**batch)
                    key = "log_probs" if model.heads["model"] else "family_log_probs"
                    loss = loss + float(losses["rdrop"]["weight"]) * symmetric_kl(
                        out[key], second[key])
                if losses["hotflip"]["enabled"] and losses["hotflip"]["weight"] > 0:
                    changed = hotflip_batch(
                        model, dict(batch, labels=labels, families=families),
                        int(losses["hotflip"]["max_examples"]))
                    if changed is not None:
                        modified = model(input_ids=changed, attention_mask=batch["attention_mask"])
                        key = "log_probs" if model.heads["model"] else "family_log_probs"
                        loss = loss + float(losses["hotflip"]["weight"]) * F.kl_div(
                            modified[key], out[key].detach().exp(), reduction="batchmean")
            scaler.scale(loss / divisor).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(settings["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            count += 1
        preds, truth = predict(model, valid_loader, device)
        metrics = _evaluate(preds, truth, model)
        score = metrics.get("balanced_accuracy_observed_classes",
                            metrics.get("family_balanced_accuracy"))
        history.append({
            "epoch": epoch + 1, "training_loss": total / max(count, 1),
            "metrics": metrics, "trainable_parameters": trainable_params,
        })
        print(f"Epoch {epoch+1}: loss={total/max(count,1):.4f}; "
              f"val balanced accuracy={score:.4f}")
        if score > best:
            best = score
            model.save_checkpoint(out_dir, tokenizer, extra={
                "best_epoch": epoch + 1,
                "validation_balanced_accuracy": score,
                "experiment": config["experiment"],
                "trainable_parameters": trainable_params,
            })
    return history
