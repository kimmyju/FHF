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


@pytest.mark.parametrize("pooling", ["eos", "eos_mean"])
def test_tiny_qwen_train_save_offline_infer(tmp_path, pooling):
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
           "gradient_checkpointing": True, "lexical_weight": .15,
           "pooling": pooling, "pooling_head_dim": 12, "pooling_dropout": 0.0}
    bundle = tmp_path / "bundle"
    checkpoint_root = tmp_path / "checkpoints"
    summary = run(frame(), cfg, bundle, fold_only=0, checkpoint_root=checkpoint_root)
    assert summary["evaluation"] == "single_fold_pilot"
    predictions = pd.read_csv(bundle / "predictions.csv", dtype={"id": str})
    source = frame().set_index("id").loc[predictions.id].reset_index()
    from peft import PeftModel
    from fhf.v3_decoder import get_base, predict_neural
    saved = checkpoint_root / "fold-0"
    loaded_cfg = json.loads((saved / "config.json").read_text())
    restored = PeftModel.from_pretrained(
        get_base(str(base), loaded_cfg, tokenizer, offline=True), saved / "adapter")
    rows, _ = encode(source.text, tokenizer, 16)
    restored_probs = predict_neural(restored, rows, tokenizer, 5)
    expected = predictions[[f"neural::{name}" for name in MODEL_NAMES]].to_numpy()
    np.testing.assert_allclose(restored_probs, expected, atol=1e-6, rtol=1e-5)
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


def test_content_pooling_masks_and_gradients():
    import torch
    from fhf.v3_pooling import pool_content
    h = torch.tensor([[[99., 99.], [2., 4.], [4., 8.], [7., 9.], [88., 88.]]], requires_grad=True)
    ids = torch.tensor([[5, 2, 3, 1, 0]])
    eos, mean = pool_content(h, ids, torch.tensor([[1, 1, 1, 1, 0]]), [0, 1, 5])
    torch.testing.assert_close(eos, torch.tensor([[7., 9.]]))
    torch.testing.assert_close(mean, torch.tensor([[3., 6.]]))
    (eos.sum() + mean.sum()).backward()
    torch.testing.assert_close(h.grad[0, 0], torch.zeros(2))
    torch.testing.assert_close(h.grad[0, 4], torch.zeros(2))
    # Moving padding to the left must not affect either pool.
    e2, m2 = pool_content(h.detach()[:, [4, 0, 1, 2, 3]], ids[:, [4, 0, 1, 2, 3]],
                         torch.tensor([[0, 1, 1, 1, 1]]), [0, 1, 5])
    torch.testing.assert_close(eos, e2)
    torch.testing.assert_close(mean, m2)
    empty_e, empty_m = pool_content(h[:, :1], torch.tensor([[1]]), torch.ones(1, 1), [1])
    torch.testing.assert_close(empty_e, empty_m)
