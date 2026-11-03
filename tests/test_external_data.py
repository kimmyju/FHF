"""External generations are joined to frozen train only after strict audit."""
import json
from pathlib import Path

import pandas as pd
import pytest

from fhf.config import load_config
from fhf.external_data import prepare_external_training, training_frames, write_external_audit
from fhf.labels import MODEL_NAMES


ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path):
    cfg = load_config(ROOT / "configs/ablations/dual_path_prompt_aware_ext.yaml")
    cfg["paths"]["base_dir"] = str(tmp_path)
    cfg["paths"]["frozen_train_csv"] = "original_train.csv"
    cfg["paths"]["frozen_validation_csv"] = "original_val.csv"
    cfg["paths"]["external_source_path"] = "raw.csv"
    cfg["paths"]["external_clean_csv"] = "external_clean.csv"
    cfg["paths"]["external_audit_json"] = "audit.json"
    cfg["external_data"]["expected_train_rows"] = None
    cfg["external_data"]["expected_validation_rows"] = None
    return cfg


def write_official(tmp_path):
    # Every target class is present, and official splits have separate prompts.
    train = pd.DataFrame([
        {"id": f"train_{i}", "model": MODEL_NAMES[i % 18],
         "prompt_id": f"official_train_prompt_{i // 18}",
         "text": f"Official training sample {i} with sufficient words."}
        for i in range(54)
    ])
    val = pd.DataFrame([
        {"id": f"val_{i}", "model": MODEL_NAMES[i],
         "prompt_id": "official_heldout",
         "text": f"Official validation sample {i} with separate language."}
        for i in range(18)
    ])
    train.to_csv(tmp_path / "original_train.csv", index=False)
    val.to_csv(tmp_path / "original_val.csv", index=False)
    return train, val


def make_external():
    return pd.DataFrame([
        {"response": "Distinct additional response from the first model.", "model": MODEL_NAMES[0],
         "prompt": "Write a short description of the moon."},
        {"response": "Distinct additional response from the second model.", "model": MODEL_NAMES[1],
         "prompt": "Write a short description of the moon."},
        {"response": "Another additional response from first model.", "model": MODEL_NAMES[0],
         "prompt": "Explain photosynthesis using three sentences."},
        {"response": "Another additional response from second model.", "model": MODEL_NAMES[1],
         "prompt": "Explain photosynthesis using three sentences."},
        {"response": "Official validation sample 0 with separate language.", "model": MODEL_NAMES[0],
         "prompt": "Write a short description of the moon."},
        {"response": "Official training sample 0 with sufficient words.", "model": MODEL_NAMES[0],
         "prompt": "Write a short description of the moon."},
        {"response": "Distinct additional response from the first model.", "model": MODEL_NAMES[0],
         "prompt": "Write a short description of the moon."},
        {"response": "This is a source from an unsupported older model.", "model": "gpt-4o",
         "prompt": "Write a short description of the moon."},
        {"response": "  ", "model": MODEL_NAMES[0],
         "prompt": "Write a short description of the moon."},
    ])


def test_external_train_only_with_frozen_validation_and_audit(tmp_path):
    cfg = make_config(tmp_path)
    old_train, old_val = write_official(tmp_path)
    make_external().to_csv(tmp_path / "raw.csv", index=False)
    train, val, external, audit = prepare_external_training(cfg)
    assert len(train) == len(old_train) + 4
    assert val.equals(old_val)
    assert len(external) == 4
    assert audit["removed_unknown_or_missing_model"] == 1
    assert audit["removed_validation_text_overlap"] == 1
    assert audit["removed_official_train_text_overlap"] == 1
    assert audit["removed_internal_duplicate_text"] == 1
    assert audit["removed_empty_text"] == 1
    assert audit["external_multi_model_prompt_groups"] == 2
    assert audit["validation_split_unchanged"] is True
    assert all(p.startswith("ntl_v2_") for p in external.prompt_id)
    assert not set(train.prompt_id) & set(val.prompt_id)
    assert set(train.model) == set(MODEL_NAMES)
    write_external_audit(cfg, external, audit)
    assert pd.read_csv(tmp_path / "external_clean.csv").shape[0] == 4
    assert json.loads((tmp_path / "audit.json").read_text())["accepted_external_rows"] == 4
    assert training_frames(cfg)[3]["accepted_external_rows"] == 4


def test_fails_if_frozen_validation_count_differs(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    make_external().to_csv(tmp_path / "raw.csv", index=False)
    cfg["external_data"]["expected_validation_rows"] = 540
    with pytest.raises(ValueError, match="expected 540"):
        prepare_external_training(cfg)


def test_fails_if_no_shared_prompt_column(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    frame = make_external().drop(columns="prompt")
    frame.to_csv(tmp_path / "raw.csv", index=False)
    with pytest.raises(ValueError, match="No prompt_id or prompt text"):
        prepare_external_training(cfg)


def test_fails_on_ambiguous_text_columns(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    frame = make_external()
    frame["text"] = frame.response
    frame.to_csv(tmp_path / "raw.csv", index=False)
    with pytest.raises(ValueError, match="Ambiguous text"):
        prepare_external_training(cfg)
    cfg["external_data"]["columns"]["text"] = "response"
    assert prepare_external_training(cfg)[3]["accepted_external_rows"] == 4


def test_fails_if_external_models_cannot_be_verified(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    frame = make_external()
    frame["model"] = "gpt-4o"
    frame.to_csv(tmp_path / "raw.csv", index=False)
    with pytest.raises(ValueError, match="No external labels match"):
        prepare_external_training(cfg)


def test_source_file_detection_never_silently_combines_files(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    root = tmp_path / "external_dir"
    root.mkdir()
    cfg["paths"]["external_source_path"] = "external_dir"
    make_external().to_csv(root / "part1.csv", index=False)
    make_external().to_csv(root / "part2.csv", index=False)
    with pytest.raises(ValueError, match="exactly one"):
        prepare_external_training(cfg)
    cfg["paths"]["external_source_path"] = "external_dir/part1.csv"
    assert prepare_external_training(cfg)[3]["accepted_external_rows"] == 4


def test_prompt_ids_with_distinct_prompts_are_rejected(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    frame = make_external()
    frame["prompt_id"] = ["same"] * len(frame)
    frame.to_csv(tmp_path / "raw.csv", index=False)
    with pytest.raises(ValueError, match="multiple distinct prompt texts"):
        prepare_external_training(cfg)


def test_conflicting_external_labels_are_removed(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    frame = make_external()
    frame.loc[1, "response"] = frame.loc[0, "response"]
    frame.to_csv(tmp_path / "raw.csv", index=False)
    merged, _, external, audit = prepare_external_training(cfg)
    assert audit["removed_conflicting_model_text"] == 3  # two conflict + repeated copy
    assert not external.text.eq(frame.loc[0, "response"]).any()
    assert len(merged) == 54 + 2


def test_no_cross_dataset_false_prompt_id_matching(tmp_path):
    cfg = make_config(tmp_path)
    write_official(tmp_path)
    external = make_external()
    external["prompt_id"] = [
        "official_heldout" if i < 2 else f"original_ext_{i}"
        for i in range(len(external))
    ]
    # The model should use actual prompt text hashes, not untrusted local
    # IDs that happen to collide with an official validation prompt ID.
    external.to_csv(tmp_path / "raw.csv", index=False)
    merged, val, clean, audit = prepare_external_training(cfg)
    assert not set(merged.prompt_id) & set(val.prompt_id)
    assert len(clean) == 4
