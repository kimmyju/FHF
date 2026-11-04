"""LoRA-efficient FHF training with mixed precision, gradient accumulation and offline prediction."""
from __future__ import annotations

import math
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from .losses import (
    objective, symmetric_kl, prompt_aware_pair_counts, within_family_pair_counts,
)
from .hotflip import hotflip_batch
from .metrics import report, oracle_within_family_predictions
from .tracking import WandbTracker


def seed_everything(seed):
    """Seed CPU, CUDA, Python and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, loader, device, return_oracle=False):
    """Predict labels, optionally evaluating exact models using the true family.

    Normal inference remains a (predictions, labels) tuple, unchanged.
    With return_oracle=True, a third list holds teacher-forced model IDs,
    or None for a family-only model.
    """
    model.eval()
    predictions, labels = [], []
    oracle = [] if return_oracle and model.heads["model"] else None
    with torch.inference_mode():
        for batch in loader:
            true_families = None
            if "labels" in batch:
                if oracle is not None:
                    true_families = batch["families"].to(device)
                labels.extend((batch["labels"] if model.heads["model"] else
                               batch["families"]).tolist())
                batch.pop("labels")
                batch.pop("families")
            elif return_oracle:
                raise ValueError("Oracle within-family evaluation requires labeled data")
            out = model(**{key: value.to(device) for key, value in batch.items()})
            key = "log_probs" if model.heads["model"] else "family_log_probs"
            predictions.extend(out[key].argmax(dim=-1).cpu().tolist())
            if oracle is not None:
                oracle.extend(oracle_within_family_predictions(
                    out["model_logits"], true_families, model.family_lookup
                ).cpu().tolist())
    if return_oracle:
        return predictions, labels, oracle
    return predictions, labels


def _evaluate(predictions, targets, model, oracle_predictions=None):
    """Use the correct label space and add oracle diagnostics for 18-class runs."""
    if model.heads["model"]:
        return report(targets, predictions, oracle_predictions=oracle_predictions)
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


def make_scheduler(optimizer, settings, updates_per_epoch):
    """Optional optimizer-step cosine schedule with linear warmup."""
    schedule = settings.get("scheduler", {})
    name = schedule.get("name", "none")
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unknown scheduler: {name}")
    total_updates = int(settings["epochs"]) * int(updates_per_epoch)
    if total_updates < 1:
        raise ValueError("Scheduler needs at least one optimizer step")
    warmup = min(total_updates - 1, int(total_updates * float(schedule.get("warmup_ratio", 0.0))))
    floor = float(schedule.get("min_lr_ratio", 0.0))

    def multiplier(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_updates - warmup)))
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


class ScoreEarlyStopper:
    """Checkpoint on significant metric improvements, stop after patience epochs."""

    def __init__(self, settings):
        self.enabled = bool(settings.get("enabled", False))
        self.patience = int(settings.get("patience", 4))
        self.min_delta = float(settings.get("min_delta", 0.0))
        self.best = -float("inf")
        self.bad_epochs = 0

    def observe(self, score):
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.enabled and self.bad_epochs >= self.patience


def fit(model, tokenizer, train_loader, valid_loader, out_dir, config, device="cpu", tracker=None):
    """Train adapter and heads with weighted losses; retain best prompt-disjoint checkpoint."""
    settings, losses = config["training"], config["losses"]
    tracker = tracker or WandbTracker({"wandb": {"enabled": False}})
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
    stopper = ScoreEarlyStopper(settings.get("early_stopping", {}))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = make_scheduler(optimizer, settings, updates_per_epoch)
    history = []
    optimizer_steps = 0
    for epoch in range(int(settings["epochs"])):
        # Critical fix: do not draw exactly the same balanced batches every epoch.
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        train_correct, train_examples = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            labels, families = batch.pop("labels"), batch.pop("families")
            prompt_ids = batch.pop("prompt_ids", None)
            # Re-scale the last partial accumulation window to preserve mean gradients.
            remainder = len(train_loader) % accumulation
            divisor = remainder if remainder and step >= (len(train_loader) - remainder) else accumulation
            autocast = (torch.autocast(device_type="cuda", dtype=precision)
                        if precision is not None else nullcontext())
            with autocast:
                out = model(**batch)
                loss, components = objective(
                    out, labels, families, config, prompt_ids=prompt_ids
                )
                if losses["rdrop"]["enabled"] and losses["rdrop"]["weight"] > 0:
                    second = model(**batch)
                    key = "log_probs" if model.heads["model"] else "family_log_probs"
                    rdrop_value = symmetric_kl(out[key], second[key])
                    components["rdrop"] = rdrop_value.detach()
                    loss = loss + float(losses["rdrop"]["weight"]) * rdrop_value
                if losses["hotflip"]["enabled"] and losses["hotflip"]["weight"] > 0:
                    changed = hotflip_batch(
                        model, dict(batch, labels=labels, families=families),
                        int(losses["hotflip"]["max_examples"]))
                    if changed is not None:
                        modified = model(input_ids=changed, attention_mask=batch["attention_mask"])
                        key = "log_probs" if model.heads["model"] else "family_log_probs"
                        hotflip_value = F.kl_div(
                            modified[key], out[key].detach().exp(), reduction="batchmean")
                        components["hotflip"] = hotflip_value.detach()
                        loss = loss + float(losses["hotflip"]["weight"]) * hotflip_value
            raw_loss = float(loss.detach())
            batch_accuracy = float((
                out["log_probs"] if model.heads["model"] else out["family_log_probs"]
            ).argmax(dim=-1).eq(labels if model.heads["model"] else families)
            .float().mean().detach())
            scaler.scale(loss / divisor).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(settings["grad_clip"]))
                lr_used = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % int(config.get("wandb", {}).get("log_every_n_steps", 1)) == 0:
                    tracker.train_step(optimizer_steps, {
                        "train/loss": raw_loss,
                        "train/accuracy": batch_accuracy,
                        "train/learning_rate": lr_used,
                        **{f"train/loss_{key}": float(value)
                           for key, value in components.items()},
                        **(dict(zip(
                            ("train/prompt_positive_pairs", "train/prompt_hard_negative_pairs"),
                            prompt_aware_pair_counts(labels, prompt_ids),
                        )) if prompt_ids is not None else {}),
                        **(dict(zip(
                            ("train/within_family_valid_anchors", "train/within_family_negative_pairs"),
                            within_family_pair_counts(labels, families, prompt_ids),
                        )) if prompt_ids is not None and
                            losses["intra_contrastive"].get("mode") == "within_family" else {}),
                    })
            total += raw_loss
            count += 1
            train_correct += batch_accuracy * len(labels)
            train_examples += len(labels)
        if settings.get("log_epoch_metrics", True):
            tracker.epoch_train(epoch + 1, total / max(count, 1),
                                train_correct / max(train_examples, 1))
        preds, truth, oracle_preds = predict(
            model, valid_loader, device, return_oracle=True
        )
        metrics = _evaluate(preds, truth, model, oracle_predictions=oracle_preds)
        score = metrics.get("balanced_accuracy_observed_classes",
                            metrics.get("family_balanced_accuracy"))
        # Compute validation losses in eval mode, so train/val curves are comparable.
        val_loss, val_cls, val_family, val_count = 0.0, 0.0, 0.0, 0
        val_flat, val_hier, val_gate = 0.0, 0.0, 0.0
        model.eval()
        with torch.inference_mode():
            for vb in valid_loader:
                vb = {key: value.to(device) for key, value in vb.items()}
                y = vb.pop("labels")
                f = vb.pop("families")
                vo = model(**vb)
                n = len(y)
                if model.heads["model"]:
                    val_cls += float(F.nll_loss(vo["log_probs"], y)) * n
                if model.heads["family"]:
                    val_family += float(F.cross_entropy(vo["family_logits"], f)) * n
                if "flat_log_probs" in vo:
                    val_flat += float(F.nll_loss(vo["flat_log_probs"], y)) * n
                    val_hier += float(F.nll_loss(vo["hierarchical_log_probs"], y)) * n
                if "style_gate_mean" in vo:
                    val_gate += float(vo["style_gate_mean"]) * n
                val_count += n
        if model.heads["model"]:
            val_loss = val_cls / max(val_count, 1)
        else:
            val_loss = val_family / max(val_count, 1)
        metrics["validation_loss"] = val_loss
        if model.residual_enabled:
            metrics["validation_flat_loss"] = val_flat / max(val_count, 1)
            metrics["validation_hierarchical_loss"] = val_hier / max(val_count, 1)
        if model.style_enabled:
            metrics["style_gate_mean"] = val_gate / max(val_count, 1)
        if model.heads["family"]:
            metrics["validation_family_loss"] = val_family / max(val_count, 1)
        metrics["validation_exact_loss"] = (
            val_cls / max(val_count, 1) if model.heads["model"] else None
        )
        tracker.validation(epoch + 1, metrics, preds, truth, model.heads["model"])
        history.append({
            "epoch": epoch + 1, "training_loss": total / max(count, 1),
            "metrics": metrics, "trainable_parameters": trainable_params,
        })
        print(f"Epoch {epoch+1}: loss={total/max(count,1):.4f}; "
              f"val balanced accuracy={score:.4f}")
        improved, should_stop = stopper.observe(score)
        if improved:
            tracker.best(epoch + 1,
                         "balanced_accuracy" if model.heads["model"] else "family_balanced_accuracy",
                         score)
        tracker.best_score(epoch + 1, stopper.best, model.heads["model"])
        if improved:
            model.save_checkpoint(out_dir, tokenizer, extra={
                "best_epoch": epoch + 1,
                "validation_balanced_accuracy": score,
                "experiment": config["experiment"],
                "trainable_parameters": trainable_params,
            })
        if should_stop:
            print(f"Early stopping at epoch {epoch + 1}: "
                  f"no significant validation BAcc gain for {stopper.patience} epochs")
            break
    return history
