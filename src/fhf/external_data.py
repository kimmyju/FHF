"""Strict external training-data preparation with a frozen official validation split.

Untrusted raw data never changes the official train/validation split. External
generations are filtered, group-prefixed and added to TRAIN ONLY.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .config import resolve_path
from .data import grouped_split, load_csv
from .labels import MODEL_NAMES, check_labels


ALIASES = {
    "text": ("text", "response", "completion", "generated_text", "output",
             "passage", "answer", "content", "generation"),
    "model": ("model", "model_name", "generator", "generator_model",
              "source_model", "model_id", "target_model", "label"),
    "prompt_id": ("prompt_id", "prompt_hash", "question_id", "instruction_id"),
    "prompt_text": ("prompt_text", "prompt", "question", "instruction",
                    "input_prompt", "user_prompt"),
    "id": ("id", "sample_id", "row_id", "uuid"),
}


def normalize_text(text):
    """NFKC, casefold and normalize whitespace for conservative exact matching."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_file(path):
    """Find a single raw file or a recognized passages/prompts CSV pair."""
    path = Path(path).expanduser()
    allowed = {".csv", ".parquet", ".jsonl"}
    if path.is_file():
        if path.suffix.lower() not in allowed:
            raise ValueError(f"External input must be CSV, Parquet, or JSONL: {path}")
        return path
    if not path.exists():
        raise FileNotFoundError(
            f"External dataset not found: {path}. Download and extract the "
            "Kaggle dataset into that directory or override paths.external_source_path."
        )
    if not path.is_dir():
        raise ValueError(f"External source must be a file or directory: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and p.suffix.lower() in allowed)
    if len(files) == 1:
        return files[0]
    if (len(files) == 2 and
        {p.name.lower() for p in files} == {"passages.csv", "prompts.csv"}):
        passage = next(p for p in files if p.name.lower() == "passages.csv")
        prompt = next(p for p in files if p.name.lower() == "prompts.csv")
        return (passage, prompt)
    raise ValueError(
        f"Expected exactly one CSV/Parquet/JSONL, or a passages.csv + prompts.csv "
        f"pair below {path}; found {len(files)}: "
        f"{[str(p) for p in files[:20]]}. "
        "Set paths.external_source_path to the intended file or folder."
    )


def _read_passages_prompts(passage_path, prompt_path, settings):
    """Join responses to prompt metadata by validated many-to-one key.

    Never join by row position. The exact candidate key and prompt text must
    be unambiguous; allow explicit join key overrides for alternate schemas.
    """
    passages = _read_source(passage_path)
    prompts = _read_source(prompt_path)
    if passages.empty or prompts.empty:
        raise ValueError("passages.csv and prompts.csv must both be nonempty")

    aliases = ("prompt_id", "prompt_idx", "prompt_key", "question_id",
               "instruction_id", "prompt_hash", "prompt_uuid")
    override = settings.get("join", {}) or {}
    left_override = override.get("passages_key")
    right_override = override.get("prompts_key")

    if bool(left_override) != bool(right_override):
        raise ValueError(
            "Specify BOTH external_data.join.passages_key and "
            "external_data.join.prompts_key, or neither."
        )
    if left_override:
        options = [(left_override, right_override)]
    else:
        options = []
        for left in aliases:
            if left in passages:
                for right in (left, "id", "prompt_id"):
                    if right in prompts and (left, right) not in options:
                        options.append((left, right))
    def as_keys(frame, col):
        values = frame[col]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            return None
        # Ignore only whitespace, not arbitrary coercion (e.g. 1 vs 1.0).
        return values.astype(str).str.strip()

    valid = []
    for left, right in options:
        if left not in passages or right not in prompts:
            continue
        pk = as_keys(prompts, right)
        fk = as_keys(passages, left)
        if pk is None or fk is None or pk.duplicated().any():
            continue
        if fk.isin(pk).all():
            valid.append((left, right))
    if not valid:
        raise ValueError(
            "Cannot safely JOIN passages.csv to prompts.csv. A unique prompt "
            "lookup key and 100% matching passage keys are required. "
            f"passages columns={list(passages.columns)}; "
            f"prompts columns={list(prompts.columns)}. Set "
            "external_data.join.passages_key and "
            "external_data.join.prompts_key in the YAML using the actual keys."
        )
    # If multiple *different* joins are possible, don't guess which is intended.
    chosen = valid[0]
    if len(valid) > 1:
        candidates = []
        for left, right in valid:
            if not any((as_keys(passages, left).equals(as_keys(passages, p[0]))
                        and as_keys(prompts, right).equals(as_keys(prompts, p[1])))
                       for p in candidates):
                candidates.append((left, right))
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple possible JOIN keys {valid}. Explicitly select "
                "external_data.join.passages_key / prompts_key."
            )
        chosen = candidates[0]
    left, right = chosen

    # In prompt table, 'text' can mean the question, never the passage.
    prompt_field = override.get("prompt_text")
    if prompt_field:
        if prompt_field not in prompts:
            raise ValueError(
                f"external_data.join.prompt_text={prompt_field!r} not in "
                f"prompts columns={list(prompts.columns)}"
            )
    else:
        candidates = [c for c in ("prompt_text", "prompt", "question",
                                  "instruction", "input_prompt", "user_prompt", "text")
                      if c in prompts and c != right]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot identify a unique prompt text column in prompts.csv "
                f"(candidates={candidates}, columns={list(prompts.columns)}). "
                "Set external_data.join.prompt_text explicitly."
            )
        prompt_field = candidates[0]

    lookup = prompts[[right, prompt_field]].copy()
    lookup.columns = ["__join_key", "__prompt_text"]
    lookup["__join_key"] = lookup["__join_key"].astype(str).str.strip()
    response = passages.copy()
    response["__join_key"] = response[left].astype(str).str.strip()
    joined = response.merge(
        lookup, on="__join_key", how="left", validate="many_to_one",
        sort=False, indicator=True,
    )
    if len(joined) != len(passages) or (joined["_merge"] != "both").any():
        raise ValueError("Prompt JOIN changed passage rows or left unmatched responses")
    if joined["__prompt_text"].isna().any() or joined["__prompt_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Joined prompt text contains empty values")
    joined = joined.drop(columns=["_merge", "__join_key"])
    # Avoid triggering the normal single-file prompt_id-consistency check
    # on ambiguous identifiers. For grouped supervision, the normalized
    # prompt *content* is used as the authoritative identity.
    if "prompt_id" in joined:
        joined = joined.drop(columns=["prompt_id"])
    if "prompt_text" in joined:
        joined = joined.drop(columns=["prompt_text"])
    joined["prompt_text"] = joined.pop("__prompt_text")
    return joined, {
        "passages_file": str(passage_path),
        "prompts_file": str(prompt_path),
        "passages_rows": int(len(passages)),
        "prompts_rows": int(len(prompts)),
        "join_passages_key": left,
        "join_prompts_key": right,
        "join_prompt_text_column": prompt_field,
        "join_matched_rows": int(len(joined)),
    }

def _read_source(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True)


def _column_name(frame, kind, settings):
    overrides = settings.get("columns", {})
    choice = overrides.get(kind)
    if choice:
        if choice not in frame.columns:
            raise ValueError(
                f"external_data.columns.{kind}={choice!r} not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return choice
    hits = [name for name in ALIASES[kind] if name in frame.columns]
    if len(hits) > 1:
        raise ValueError(
            f"Ambiguous {kind} columns {hits}; choose one via "
            f"external_data.columns.{kind}."
        )
    return hits[0] if hits else None


def _checked_split(train, val, settings):
    for name, df in (("baseline train", train), ("baseline validation", val)):
        if "prompt_id" not in df or df.prompt_id.isna().any():
            raise ValueError(f"{name} must contain valid prompt_id")
        if df.prompt_id.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains empty prompt_id")
        check_labels(df.model.tolist())
    if set(train.id.astype(str)) & set(val.id.astype(str)):
        raise ValueError("Frozen baseline train/validation IDs overlap")
    if set(train.prompt_id.astype(str)) & set(val.prompt_id.astype(str)):
        raise ValueError("Frozen baseline train/validation prompt_ids overlap")
    expected = settings.get("expected_validation_rows")
    if expected is not None and len(val) != int(expected):
        raise ValueError(
            f"Frozen validation has {len(val)} rows, expected {expected}. "
            "Refusing to change the previous evaluation set."
        )
    expected_train = settings.get("expected_train_rows")
    if expected_train is not None and len(train) != int(expected_train):
        raise ValueError(
            f"Frozen train has {len(train)} rows, expected {expected_train}."
        )


def prepare_external_training(config):
    """Return frozen train+cleaned external, frozen validation, clean external, audit.

    No fitted encoder or label inference occurs. The caller saves artifacts
    only after every leakage and schema check succeeds.
    """
    settings = config.get("external_data", {})
    if not settings.get("enabled", False):
        return None

    original_train = load_csv(resolve_path(config, "frozen_train_csv"))
    validation = load_csv(resolve_path(config, "frozen_validation_csv"))
    _checked_split(original_train, validation, settings)

    raw_source = _source_file(resolve_path(config, "external_source_path"))
    join_audit = {}
    if isinstance(raw_source, tuple):
        raw, join_audit = _read_passages_prompts(
            raw_source[0], raw_source[1], settings
        )
        raw_file = raw_source[0]
    else:
        raw_file = raw_source
        raw = _read_source(raw_file)
    if raw.empty:
        raise ValueError(f"No rows in external dataset: {raw_file}")
    found = {kind: _column_name(raw, kind, settings) for kind in ALIASES}
    if not found["text"] or not found["model"]:
        raise ValueError(
            f"External dataset requires text and model columns. "
            f"Found columns: {list(raw.columns)}. Set external_data.columns."
        )
    if not found["prompt_id"] and not found["prompt_text"]:
        raise ValueError(
            f"No prompt_id or prompt text column in external dataset: "
            f"{list(raw.columns)}. Prompt-aware contrastive cannot use "
            "fabricated one-row prompt IDs."
        )

    # The experiment is generated by explicit opt-in and no model aliases
    # are automatically interpreted as an official competition label.
    label_map = settings.get("label_map") or {}
    if any(value not in MODEL_NAMES for value in label_map.values()):
        raise ValueError("External label_map values must be exact official model names")
    if len(set(label_map.values())) != len(label_map):
        raise ValueError("External label_map may not collapse multiple labels")
    source_labels = raw[found["model"]].fillna("").astype(str).str.strip()
    mapped = source_labels.map(lambda x: label_map.get(x, x))
    known = mapped.isin(MODEL_NAMES)
    all_label_counts = dict(Counter(source_labels))
    report = {
        "source_file": str(raw_file),
        "source_join": join_audit,
        "raw_rows": int(len(raw)),
        "selected_columns": found,
        "raw_model_counts": dict(sorted(all_label_counts.items())),
        "explicit_label_map": label_map,
        "removed_unknown_or_missing_model": int((~known).sum()),
        "unknown_model_counts": dict(sorted(Counter(source_labels[~known]).items())),
        "baseline_train_rows": int(len(original_train)),
        "frozen_validation_rows": int(len(validation)),
        "frozen_validation_id_sha256": digest("\n".join(
            sorted(validation.id.astype(str))
        )),
    }
    work = pd.DataFrame({
        "text": raw[found["text"]],
        "model": mapped,
        "_source_row": range(len(raw)),
    })[known].copy()
    if work.empty:
        raise ValueError(
            "No external labels match the 18 official model names. "
            f"Observed: {list(all_label_counts)}. Do NOT relabel another "
            "model version as an official target; only use verified explicit mappings."
        )

    if found["prompt_text"]:
        prompt_text = raw.loc[known, found["prompt_text"]]
        if prompt_text.isna().any() or prompt_text.astype(str).str.strip().eq("").any():
            raise ValueError("External prompt text contains missing/empty values")
        prompt_norm = prompt_text.astype(str).map(normalize_text)
        prompt_keys = prompt_norm.map(digest)
        if found["prompt_id"]:
            original_prompt = raw.loc[known, found["prompt_id"]]
            if original_prompt.isna().any() or original_prompt.astype(str).str.strip().eq("").any():
                raise ValueError("External prompt_id contains missing/empty values")
            check = pd.DataFrame({
                "source_id": original_prompt.astype(str).values,
                "prompt_hash": prompt_keys.values,
            })
            if check.groupby("source_id").prompt_hash.nunique().max() > 1:
                raise ValueError(
                    "An external prompt_id points to multiple distinct prompt texts; "
                    "fix the raw data or select a trustworthy prompt column."
                )
        # Prefer the prompt text identity even if source prompt IDs differ.
        # Different input prompts cannot accidentally be merged just
        # because their source IDs happen to match.
        keys = prompt_keys
    else:
        raw_prompt = raw.loc[known, found["prompt_id"]]
        if raw_prompt.isna().any() or raw_prompt.astype(str).str.strip().eq("").any():
            raise ValueError("External prompt_id contains missing/empty values")
        keys = raw_prompt.astype(str).map(lambda x: digest(x.strip()))

    work["prompt_id"] = "ntl_v2_" + keys.to_numpy()
    missing_text = work.text.isna() | work.text.astype(str).str.strip().eq("")
    report["removed_empty_text"] = int(missing_text.sum())
    work = work[~missing_text].copy()
    work["text"] = work.text.astype(str)
    work["_text_key"] = work.text.map(normalize_text)
    min_chars = int(settings.get("min_chars", 1))
    min_words = int(settings.get("min_words", 0))
    too_short = (work.text.str.len() < min_chars) | (
        work.text.str.split().str.len() < min_words
    )
    report["removed_too_short"] = int(too_short.sum())
    work = work[~too_short].copy()

    # Holdout (and official train) text overlap must never be used to
    # augment training. Exact normalized matching catches whitespace and
    # case differences; near duplicates/semantic copies need manual review.
    validation_keys = set(validation.text.map(normalize_text))
    train_keys = set(original_train.text.map(normalize_text))
    collision_val = work._text_key.isin(validation_keys)
    report["removed_validation_text_overlap"] = int(collision_val.sum())
    work = work[~collision_val].copy()

    collision_train = work._text_key.isin(train_keys)
    report["removed_official_train_text_overlap"] = int(collision_train.sum())
    work = work[~collision_train].copy()

    # Identical outputs attached to different model labels are ambiguous:
    # remove the entire conflicting group rather than picking an arbitrary
    # label. Repeated same-model text rows keep the first occurrence.
    ambiguous = work.groupby("_text_key").model.nunique()
    ambiguous_keys = set(ambiguous[ambiguous > 1].index)
    conflicting = work._text_key.isin(ambiguous_keys)
    report["removed_conflicting_model_text"] = int(conflicting.sum())
    work = work[~conflicting].copy()
    duplicates = work.duplicated(subset=["_text_key"], keep="first")
    report["removed_internal_duplicate_text"] = int(duplicates.sum())
    work = work[~duplicates].copy()

    cap = settings.get("max_per_model")
    if cap is not None:
        if int(cap) < 1:
            raise ValueError("external_data.max_per_model must be positive or null")
        before = len(work)
        # Retain deterministic first rows to keep source provenance reproducible.
        work = work.groupby("model", sort=False, group_keys=False).head(int(cap))
        report["removed_per_model_cap"] = int(before - len(work))
    else:
        report["removed_per_model_cap"] = 0

    min_rows = int(settings.get("min_accepted_rows", 1))
    if len(work) < min_rows:
        raise ValueError(
            f"Only {len(work)} eligible external examples remain after filtering; "
            f"minimum {min_rows}. Inspect labels, prompts and duplicate counts."
        )

    # Stable IDs based on dataset row number rather than source IDs that may
    # be reused for prompt groups. Prefix both IDs and prompt IDs to prevent
    # collisions with anonymized official identifiers.
    work["id"] = (
        "ntl_v2_row_" + work["_source_row"].astype(str).str.zfill(7)
    )
    work["origin"] = "ntl_claude_gen_v2"
    external = work[["id", "text", "model", "prompt_id", "origin"]].reset_index(drop=True)

    # Ensure that the new dataset is actually useful for prompt-aware pairs.
    multi_model = external.groupby("prompt_id").model.nunique()
    report["external_multi_model_prompt_groups"] = int((multi_model >= 2).sum())
    report["external_unique_prompts"] = int(external.prompt_id.nunique())
    min_multi = int(settings.get("min_multi_model_prompt_groups", 1))
    if report["external_multi_model_prompt_groups"] < min_multi:
        raise ValueError(
            f"External data has only {report['external_multi_model_prompt_groups']} "
            "prompt groups shared across different models. Check whether "
            "prompt_id identifies the question instead of each response."
        )

    official = original_train.copy()
    official["origin"] = "official"
    merged = pd.concat([official, external], ignore_index=True, sort=False)
    # Match the required model fields while retaining external provenance.
    if merged.id.astype(str).duplicated().any():
        raise ValueError("Merged sample IDs are not unique")
    if set(merged.prompt_id.astype(str)) & set(validation.prompt_id.astype(str)):
        raise ValueError("Merged training prompts overlap frozen validation prompts")
    if set(merged.text.map(normalize_text)) & validation_keys:
        raise ValueError("Merged training text overlaps frozen validation text")

    check_labels(merged.model.tolist())
    report.update({
        "accepted_external_rows": int(len(external)),
        "merged_train_rows": int(len(merged)),
        "accepted_external_model_counts": dict(sorted(Counter(external.model).items())),
        "merged_model_counts": dict(sorted(Counter(merged.model).items())),
        "remaining_validation_overlap": 0,
        "validation_split_unchanged": True,
        "warning": (
            "Exact normalized response overlap was removed. Original official "
            "prompts are anonymized: similar prompts, near-duplicate texts, "
            "and mislabeled external generation APIs cannot be ruled out "
            "automatically. Inspect dataset provenance before interpreting gains."
        ),
    })
    return merged, validation, external, report


def training_frames(config):
    """Use the normal split unless an explicitly configured external experiment."""
    if config.get("external_data", {}).get("enabled", False):
        return prepare_external_training(config)
    train, val = grouped_split(
        load_csv(resolve_path(config, "train_csv")),
        float(config["training"]["validation_fraction"]),
        int(config["seed"]),
    )
    return train, val, None, None


def write_external_audit(config, external, audit):
    """Persist prepared, filtered external examples and a provenance report."""
    import json

    for key, payload, is_json in (
        ("external_clean_csv", external, False),
        ("external_audit_json", audit, True),
    ):
        out = resolve_path(config, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        if is_json:
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            payload.to_csv(out, index=False)
