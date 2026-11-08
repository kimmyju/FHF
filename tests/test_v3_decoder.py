import json
import numpy as np
import pandas as pd
import pytest

from fhf.labels import MODEL_NAMES
from fhf.v3_decoder import validate, disjoint, split_folds, encode, run, infer


def frame(groups=4):
    return pd.DataFrame([{"id": f"{g}-{k}", "prompt_id": str(g),
                          "text": f"sample group {g} author {k}", "model": name}
                         for g in range(groups) for k, name in enumerate(MODEL_NAMES)])


def test_group_folds_and_leaks():
    data = validate(frame())
    covered = []
    for fit, hold in split_folds(data, 2, 42):
        assert set(data.iloc[fit].prompt_id).isdisjoint(data.iloc[hold].prompt_id)
        covered.extend(hold)
    assert sorted(covered) == list(range(len(data)))
    with pytest.raises(ValueError, match="overlap"):
        disjoint(data, data)
    bad = data.copy()
    bad.loc[1, "text"] = bad.loc[0, "text"].upper()
    with pytest.raises(ValueError, match="duplicate responses"):
        validate(bad)


def test_head_tail_and_eos():
    class Tokenizer:
        eos_token_id = 99
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(int(text)))}
    rows, report = encode(["12", "2"], Tokenizer(), 6)
    assert rows[0]["input_ids"] == [0, 1, 2, 10, 11, 99]
    assert rows[1]["input_ids"] == [0, 1, 99]
    assert report["truncated"] == 1


def test_tiny_qwen_train_save_offline_infer(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config, Qwen3ForSequenceClassification, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "sample": 3,
             "group": 4, "author": 5, **{str(i): i + 6 for i in range(20)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    base = tmp_path / "base"
    tokenizer.save_pretrained(base)
    Qwen3ForSequenceClassification(Qwen3Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_labels=18, pad_token_id=0)).save_pretrained(base)
    cfg = {"model": str(base), "seed": 42, "folds": 2, "epochs": 1,
           "max_length": 16, "batch_size": 7, "accumulation": 3,
           "learning_rate": 1e-3, "head_learning_rate": 1e-3,
           "weight_decay": .01, "warmup_ratio": .1,
           "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.,
           "gradient_checkpointing": True, "lexical_weight": .15}
    bundle = tmp_path / "bundle"
    checkpoint_root = tmp_path / "checkpoints"
    summary = run(frame(), cfg, bundle, fold_only=0, checkpoint_root=checkpoint_root)
    assert summary["evaluation"] == "single_fold_pilot"
    predictions = pd.read_csv(bundle / "predictions.csv", dtype={"id": str})
    source = frame().set_index("id").loc[predictions.id].reset_index()
    output = tmp_path / "submission.parquet"
    infer(bundle, source, str(base), output, batch_size=5)
    submitted = pd.read_parquet(output)
    probs = predictions[[f"blend::{name}" for name in MODEL_NAMES]].to_numpy()
    assert submitted.model.tolist() == np.asarray(MODEL_NAMES)[probs.argmax(1)].tolist()
    assert submitted.id.tolist() == source.id.tolist()
    assert list(submitted.columns) == ["id", "model"]
    assert (checkpoint_root / "fold-0" / "adapter" / "adapter_model.safetensors").exists()
    with pytest.raises(ValueError, match="Output exists"):
        run(frame(), cfg, bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["labels"] == MODEL_NAMES


def test_external_overlap_filter_and_cap():
    from fhf.v3_decoder import prepare_external
    official = frame(2)
    external = frame(5)
    external["id"] = "ext-" + external.id
    external["prompt_id"] = "ext-" + external.prompt_id
    cleaned = prepare_external(external, official, 2, 42)
    assert len(cleaned) == 36
    assert set(cleaned.text).isdisjoint(set(official.text))
    assert cleaned.model.value_counts().eq(2).all()
