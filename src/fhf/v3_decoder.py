"""Decoder attribution: raw text -> last valid token -> flat 18-way head.

No generated labels, chat template, prompt features or family routing.
CV checkpoints use a prespecified epoch count, not the best outer-fold epoch.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from .labels import MODEL_NAMES, MODEL_TO_ID, MODEL_FAMILY_IDS, check_labels


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")


def normalize(text):
    # For duplicate detection ONLY; the model sees the original text.
    return " ".join(str(text).split()).casefold()


def validate(frame, labeled=True):
    required = {"id", "text"} | ({"model", "prompt_id"} if labeled else set())
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError(f"Missing/null required columns: {sorted(required)}")
    if frame.empty or frame.id.astype(str).duplicated().any():
        raise ValueError("Empty data or duplicate IDs")
    if not frame.text.astype(str).str.strip().all():
        raise ValueError("Empty text")
    if labeled:
        check_labels(frame.model)
        if set(frame.model) != set(MODEL_NAMES):
            raise ValueError("All 18 labels must occur")
        if not frame.prompt_id.astype(str).str.strip().all():
            raise ValueError("Empty prompt_id")
        if frame.text.map(normalize).duplicated().any():
            raise ValueError("Normalized duplicate responses: resolve before splitting")
    return frame.reset_index(drop=True)


def disjoint(train, valid):
    for column in ("id", "prompt_id"):
        if set(train[column].astype(str)) & set(valid[column].astype(str)):
            raise ValueError(f"Train/validation {column} overlap")
    if set(train.text.map(normalize)) & set(valid.text.map(normalize)):
        raise ValueError("Train/validation response overlap")


def split_folds(frame, folds, seed):
    if not 2 <= folds <= frame.prompt_id.nunique():
        raise ValueError("Invalid number of prompt folds")
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    result = []
    for fit, hold in splitter.split(frame, frame.model, frame.prompt_id):
        disjoint(frame.iloc[fit], frame.iloc[hold])
        for ix in (fit, hold):
            if set(frame.iloc[ix].model) != set(MODEL_NAMES):
                raise ValueError("Each fit/hold fold must include all 18 labels")
        result.append((fit, hold))
    return result


def metrics(y, probabilities):
    pred = np.asarray(probabilities).argmax(1)
    matrix = confusion_matrix(y, pred, labels=np.arange(18))
    family = np.asarray(MODEL_FAMILY_IDS)
    family_ok = family[y] == family[pred]
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(np.mean(pred == y)),
            "family_accuracy": float(family_ok.mean()),
            "within_family_given_correct": float(np.mean(pred[family_ok] == y[family_ok])) if family_ok.any() else None,
            "recall": dict(zip(MODEL_NAMES, (matrix.diagonal() / matrix.sum(1)).tolist())),
            "confusion": matrix.tolist(), "labels": MODEL_NAMES}


def encode(texts, tokenizer, max_length, cfg=None):
    if max_length < 4:
        raise ValueError("max_length must be >=4")
    result, lengths = [], []
    for text in texts:
        tokens = tokenizer(str(text), add_special_tokens=False)["input_ids"]
        lengths.append(len(tokens) + 1)
        budget = max_length - 1  # Explicit EOS summarizes the complete paragraph.
        if len(tokens) > budget:
            head = (budget + 1) // 2
            tokens = tokens[:head] + tokens[-(budget - head):]
        tokens = tokens + [tokenizer.eos_token_id]
        row = {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        if (cfg or {}).get("pooling") == "eos_style":
            from .v3_style import style_features
            row["style_features"] = style_features(text)
        result.append(row)
    return result, {"n": len(lengths), "truncated": int(sum(n > max_length for n in lengths)),
                    "token_length_quantiles": np.quantile(lengths, [0, .5, .9, .99, 1]).tolist()}


def prepare_external(external, official, cap, seed):
    """Accept an already audited joined CSV, never recursively discover sources."""
    external = validate(external)
    deny = set(official.text.map(normalize))
    prompt_deny = set(official.prompt_id.astype(str))
    kept = external.loc[~external.text.map(normalize).isin(deny)
                        & ~external.prompt_id.astype(str).isin(prompt_deny)]
    if cap < 1:
        raise ValueError("external_per_class must be positive")
    kept = pd.concat([group.sample(min(len(group), cap), random_state=seed)
                      for _, group in kept.groupby("model")], ignore_index=True)
    validate(kept)
    return kept


def lexical_fit(train):
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                        sublinear_tf=True, min_df=2, max_features=180000, dtype=np.float32),
        LogisticRegression(C=3, class_weight="balanced", max_iter=1200))
    return model.fit(train.text.tolist(), train.model.tolist())


def lexical_predict(model, texts):
    p = model.predict_proba(list(texts))
    return p[:, [list(model.classes_).index(label) for label in MODEL_NAMES]]


def get_base(path, cfg, tokenizer, offline=False):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if offline else torch.float32)
    pooling = cfg.get("pooling", "eos")
    if pooling not in ("eos", "eos_mean", "eos_style"):
        raise ValueError(f"Unknown pooling: {pooling}")
    config = AutoConfig.from_pretrained(path, local_files_only=offline,
                                       revision=cfg.get("revision", "main"))
    config.num_labels = 18
    config.id2label = dict(enumerate(MODEL_NAMES))
    config.label2id = MODEL_TO_ID
    cls = AutoModelForSequenceClassification
    if pooling == "eos_mean":
        if config.model_type != "qwen3":
            raise ValueError("EOS-mean classifier currently supports Qwen3 only")
        from .v3_pooling import Qwen3EOSMeanClassifier
        cls = Qwen3EOSMeanClassifier
        config.pooling_head_dim = int(cfg.get("pooling_head_dim", 512))
        config.pooling_dropout = float(cfg.get("pooling_dropout", 0.1))
        config.pooling_special_ids = list(tokenizer.all_special_ids)
    if pooling == "eos_style":
        from .v3_style import Qwen3EOSStyleClassifier, FEATURE_VERSION
        if config.model_type != "qwen3":
            raise ValueError("EOS-style requires Qwen3")
        if cfg.get("style_feature_version") != FEATURE_VERSION:
            raise ValueError("Missing/unsupported style_feature_version")
        config.style_hidden_dim = int(cfg.get("style_hidden_dim", 128))
        config.style_dropout = float(cfg.get("style_dropout", 0.1))
        cls = Qwen3EOSStyleClassifier
    model = cls.from_pretrained(
        path, config=config, torch_dtype=dtype, local_files_only=offline,
        revision=cfg.get("revision", "main"), attn_implementation="sdpa")
    if pooling == "eos_style":
        # HF initializes missing weights while loading; zero AFTER that pass.
        # The trained head is restored by PeftModel afterwards at inference.
        model.score.zero_residual()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


def collate_rows(rows, tokenizer):
    import torch
    tokens = [{k: v for k, v in row.items() if k != "style_features"} for row in rows]
    batch = tokenizer.pad(tokens, return_tensors="pt")
    if any("style_features" in row for row in rows):
        if not all("style_features" in row for row in rows):
            raise ValueError("Mixed style and non-style rows")
        batch["style_features"] = torch.tensor([r["style_features"] for r in rows], dtype=torch.float32)
    return batch


def predict_neural(model, rows, tokenizer, batch_size):
    import torch
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = collate_rows(rows[start:start + batch_size], tokenizer)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            parts.append(model(**batch).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(parts)


def train_stage(model, rows, labels, tokenizer, cfg, epochs, tracker=None, stage="official", on_epoch_end=None):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    batch_size, accumulation = int(cfg["batch_size"]), int(cfg["accumulation"])
    if min(batch_size, accumulation, epochs) < 1:
        raise ValueError("Batch, accumulation and epochs must be positive")
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=18)
    if (counts == 0).any():
        raise ValueError("Training stage needs all 18 labels")
    # Mean weight is 1 under the training distribution. Avoid per-microbatch
    # weighted-CE normalization: accumulated gradients then match a large batch.
    weights = torch.tensor(len(labels) / (18 * counts), device=model.device, dtype=torch.float32)
    head, adapter = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (head if "modules_to_save" in name else adapter).append(param)
    adapter_lr = float(cfg.get(f"{stage}_learning_rate", cfg["learning_rate"]))
    head_lr = float(cfg.get(f"{stage}_head_learning_rate", cfg["head_learning_rate"]))
    if not math.isfinite(adapter_lr) or not math.isfinite(head_lr) or min(adapter_lr, head_lr) <= 0:
        raise ValueError("Stage learning rates must be finite and positive")
    optimizer = torch.optim.AdamW([
        {"params": adapter, "lr": adapter_lr},
        {"params": head, "lr": head_lr}], weight_decay=float(cfg["weight_decay"]))
    steps = math.ceil(len(rows) / (batch_size * accumulation)) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * cfg["warmup_ratio"]), steps)
    history = []
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(rows))
        loss_total, correct = 0., 0
        # Normalize by actual window size, including the last partial window.
        for start in range(0, len(order), batch_size * accumulation):
            window = order[start:start + batch_size * accumulation]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(window), batch_size):
                ix = window[offset:offset + batch_size]
                batch = collate_rows([rows[i] for i in ix], tokenizer)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                target = torch.tensor(labels[ix], dtype=torch.long, device=model.device)
                logits = model(**batch).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, target, weight=weights, reduction="sum")
                (loss / len(window)).backward()
                loss_total += loss.detach().item()
                correct += (logits.detach().argmax(-1) == target).sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
        record = {"stage": stage, "epoch": epoch + 1, "loss": loss_total / len(rows),
                  "train_accuracy": correct / len(rows),
                  "initial_adapter_lr": adapter_lr, "initial_head_lr": head_lr}
        if on_epoch_end is not None:
            record.update(on_epoch_end(stage, epoch + 1))
        history.append(record)
        print(json.dumps(record), flush=True)
        if tracker:
            tracker.log({f"{stage}/{k}": v for k, v in record.items() if k != "stage"})
    return history


def fit_fold(train, valid, cfg, out, external=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, set_seed
    out = Path(out)
    if out.exists():
        raise ValueError(f"Output already exists; choose a fresh run directory: {out}")
    out.mkdir(parents=True)
    seed = int(cfg["seed"])
    set_seed(seed)
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision", "main"))
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires EOS")
    # EOS must remain a real token for last-token pooling. Qwen has a distinct pad.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Use a decoder tokenizer with distinct PAD and EOS tokens")
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(out / "tokenizer")
    tr, audit = encode(train.text, tokenizer, int(cfg["max_length"]), cfg)
    va, va_audit = encode(valid.text, tokenizer, int(cfg["max_length"]), cfg)
    dump(out / "tokenization.json", {"train": audit, "validation": va_audit})
    model = get_base(cfg["model"], cfg, tokenizer)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    cfg = {**cfg, "revision": resolved_revision or cfg.get("revision", "main")}
    dump(out / "config.json", cfg)
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS", r=int(cfg["lora_rank"]), lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]), bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"]))
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.print_trainable_parameters()
    tracker = None
    if cfg.get("wandb"):
        import wandb
        tracker = wandb.init(project=cfg["wandb_project"], name=out.name, config=cfg, reinit=True,
                             tags=["fingerprint-v3", "decoder", "flat-ce", cfg.get("pooling", "eos")])
    history = []
    select_best = bool(cfg.get("select_best_epoch", False))
    epoch_reports, best = [], {}
    best_probs = None
    final_probs = None

    def evaluate_epoch(stage, epoch):
        nonlocal best_probs, final_probs
        probabilities = predict_neural(model, va, tokenizer, int(cfg["batch_size"]))
        report = metrics(valid.model.map(MODEL_TO_ID).to_numpy(), probabilities)
        folder = out / "epochs" / f"{stage}-{epoch}"
        model.save_pretrained(folder / "adapter")
        np.save(folder / "validation_probabilities.npy", probabilities)
        # External epochs are diagnostic; only the last external checkpoint and
        # official epochs compete. No selection interrupts the training schedule.
        eligible = stage == "official" or (stage == "external" and epoch == int(cfg["external_epochs"]))
        record = {"stage": stage, "epoch": epoch, "eligible": eligible, **report}
        epoch_reports.append(record)
        dump(out / "epoch_metrics.json", epoch_reports)
        if eligible and (not best or report["balanced_accuracy"] > best["balanced_accuracy"]):
            best.update(stage=stage, epoch=epoch, balanced_accuracy=report["balanced_accuracy"],
                        adapter_path=str(folder.relative_to(out) / "adapter"))
            best_probs = probabilities.copy()
            dump(out / "selection.json", best)
        final_probs = probabilities
        return {"val_balanced_accuracy": report["balanced_accuracy"],
                "val_family_accuracy": report["family_accuracy"],
                "val_within_family_given_correct": report["within_family_given_correct"]}

    try:
        if external is not None:
            ex, ex_audit = encode(external.text, tokenizer, int(cfg["max_length"]), cfg)
            dump(out / "external_tokenization.json", ex_audit)
            history += train_stage(model, ex, external.model.map(MODEL_TO_ID), tokenizer, cfg,
                                   int(cfg["external_epochs"]), tracker, "external",
                                   evaluate_epoch if select_best else None)
        history += train_stage(model, tr, train.model.map(MODEL_TO_ID), tokenizer, cfg,
                               int(cfg["epochs"]), tracker, "official",
                               evaluate_epoch if select_best else None)
        if select_best:
            shutil.copytree(out / best["adapter_path"], out / "adapter")
            neural = best_probs
            dump(out / "final_epoch_metrics.json", metrics(valid.model.map(MODEL_TO_ID).to_numpy(), final_probs))
        else:
            model.save_pretrained(out / "adapter")
            neural = predict_neural(model, va, tokenizer, int(cfg["batch_size"]))
        lexical = lexical_fit(train)
        joblib.dump(lexical, out / "lexical.joblib")
        lex = lexical_predict(lexical, valid.text)
        dump(out / "history.json", history)
        train[["id", "prompt_id", "model"]].to_csv(out / "training_ids.csv", index=False)
        return neural, lex
    finally:
        if tracker:
            tracker.finish()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(train, cfg, output, valid=None, external=None, fold_only=None, checkpoint_root=None):
    train = validate(train)
    if cfg.get("select_best_epoch") and valid is None:
        raise ValueError("select_best_epoch requires explicit --valid; outer CV folds cannot select epochs")
    out = Path(output)
    if out.exists():
        raise ValueError("Output exists; use a new path to prevent stale-fold mixing")
    checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else out
    if checkpoint_root != out and checkpoint_root.exists():
        raise ValueError("Checkpoint directory exists; choose a fresh run name")
    if not 0 <= float(cfg["lexical_weight"]) <= 1:
        raise ValueError("lexical_weight must be between 0 and 1")
    if valid is not None:
        valid = validate(valid)
        disjoint(train, valid)
        partitions = [(train, valid)]
        target = valid
    else:
        indices = split_folds(train, int(cfg["folds"]), int(cfg["seed"]))
        partitions = [(train.iloc[a], train.iloc[b]) for a, b in indices]
        target = train
    if fold_only is not None and not 0 <= fold_only < len(partitions):
        raise ValueError("fold-only outside range")
    if external is not None:
        official = pd.concat([train, valid]) if valid is not None else train
        external = prepare_external(external, official, int(cfg["external_per_class"]), int(cfg["seed"]))
    out.mkdir(parents=True)
    dump(out / "config.json", cfg)
    if external is not None:
        dump(out / "external_audit.json", {"rows_after_filter": len(external),
             "counts": external.model.value_counts().to_dict(),
             "warning": "Exact response/ID filtering cannot prove semantic prompt independence or label provenance."})
    all_rows, reports, fold_names = [], {}, []
    for fold, (fit, hold) in enumerate(partitions):
        if fold_only is not None and fold != fold_only:
            continue
        name = f"fold-{fold}"
        neural, lexical = fit_fold(fit, hold, {**cfg, "seed": cfg["seed"] + fold}, checkpoint_root / name, external)
        w = float(cfg["lexical_weight"])
        blend = (1 - w) * neural + w * lexical
        y = hold.model.map(MODEL_TO_ID).to_numpy()
        reports[name] = {key: metrics(y, p) for key, p in [("neural", neural), ("lexical", lexical), ("blend", blend)]}
        rows = hold[["id", "prompt_id", "model"]].copy()
        rows["fold"] = fold
        for prefix, p in [("neural", neural), ("lexical", lexical), ("blend", blend)]:
            for j, label in enumerate(MODEL_NAMES):
                rows[f"{prefix}::{label}"] = p[:, j]
        all_rows.append(rows)
        fold_names.append(os.path.relpath(checkpoint_root / name, out))
        dump(out / "fold_metrics.json", reports)
    oof = pd.concat(all_rows, ignore_index=True)
    oof.to_csv(out / "predictions.csv", index=False)
    y = oof.model.map(MODEL_TO_ID).to_numpy()
    summary = {key: metrics(y, oof[[f"{key}::{c}" for c in MODEL_NAMES]].to_numpy())
               for key in ("neural", "lexical", "blend")}
    summary["evaluation"] = "frozen_holdout" if valid is not None else ("single_fold_pilot" if fold_only is not None else "grouped_oof")
    if cfg.get("select_best_epoch"):
        summary["evaluation"] = "validation_selected_holdout"
        summary["selection"] = json.loads((checkpoint_root / "fold-0" / "selection.json").read_text())
        summary["final_epoch_neural"] = json.loads((checkpoint_root / "fold-0" / "final_epoch_metrics.json").read_text())
    summary["rows"] = len(oof)
    summary["external_used"] = external is not None
    summary["data_sha256"] = hashlib.sha256(target.to_csv(index=False).encode()).hexdigest()
    dump(out / "metrics.json", summary)
    dump(out / "manifest.json", {"folds": fold_names, "labels": MODEL_NAMES,
         "lexical_weight": cfg["lexical_weight"], "evaluation": summary["evaluation"]})
    return summary


def infer(bundle, test, base, output, batch_size=4, lexical_weight=None):
    """Offline only; load one fold at a time. Base folder is explicitly required."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    started = time.monotonic()
    test = validate(test, labeled=False)
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["labels"] != MODEL_NAMES or not manifest["folds"]:
        raise ValueError("Invalid bundle label order/folds")
    weight = manifest["lexical_weight"] if lexical_weight is None else lexical_weight
    if not 0 <= weight <= 1 or batch_size < 1:
        raise ValueError("Invalid inference weight/batch size")
    total = np.zeros((len(test), 18))
    for name in manifest["folds"]:
        path = bundle / name
        cfg = json.loads((path / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        model = PeftModel.from_pretrained(get_base(base, cfg, tokenizer, offline=True), path / "adapter", local_files_only=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        rows, _ = encode(test.text, tokenizer, int(cfg["max_length"]), cfg)
        neural = predict_neural(model, rows, tokenizer, batch_size)
        lex = lexical_predict(joblib.load(path / "lexical.joblib"), test.text)
        total += (1 - weight) * neural + weight * lex
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    total /= len(manifest["folds"])
    if not np.isfinite(total).all() or not np.allclose(total.sum(1), 1, atol=1e-5):
        raise RuntimeError("Invalid probabilities")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test.id, "model": np.asarray(MODEL_NAMES)[total.argmax(1)]}).to_parquet(destination, index=False)
    dump(str(destination) + ".runtime.json", {"seconds": time.monotonic() - started,
         "rows": len(test), "folds": len(manifest["folds"]), "lexical_weight": weight})
