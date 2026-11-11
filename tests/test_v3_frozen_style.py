import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from fhf.labels import MODEL_NAMES
from fhf.v3_decoder import run
from fhf.v3_frozen_style import fit_residual, train_frozen, predict_frozen, make_head


def config():
    return {'seed': 42, 'style_feature_version': 1, 'hidden_dim': 8, 'dropout': 0.,
            'epochs': 2, 'batch_size': 18, 'learning_rate': .01, 'weight_decay': 0.,
            'alphas': [0., .5, 1.], 'inference_batch_size': 7}


def data(group):
    return pd.DataFrame([{'id': f'{group}-{i}', 'prompt_id': str(group),
                          'text': f'sample {group} author {i}!', 'model': label}
                         for i, label in enumerate(MODEL_NAMES)])


def test_residual_fallback_detaches_logits_and_trains_only_style(tmp_path):
    torch.set_num_threads(1)
    train = data(0)
    logits = (torch.eye(18) * 10).requires_grad_()
    before = logits.detach().clone()
    cfg = config()
    baseline, selected, p, best = fit_residual(logits, logits, train, train, cfg, tmp_path)
    assert baseline['balanced_accuracy'] == selected['balanced_accuracy'] == 1.
    assert best['alpha'] == 0 and best['epoch'] == 0
    assert logits.grad is None
    torch.testing.assert_close(logits, before, atol=0, rtol=0)
    np.testing.assert_allclose(p, logits.detach().softmax(-1).numpy(), atol=0, rtol=0)
    last = torch.load(tmp_path / 'style-epoch-2.pt', weights_only=True)
    assert last['3.weight'].count_nonzero() > 0
    initial = torch.load(tmp_path / 'style.pt', weights_only=True)
    assert initial['3.weight'].count_nonzero() == 0


def test_selected_nonzero_correction_reloads(tmp_path):
    torch.set_num_threads(1)
    # EOS already solves 17 classes; the residual must recover class 1.
    train, valid = data(0), data(99)
    train.loc[1, 'text'] = valid.loc[1, 'text'] = '!' * 100
    cfg = {**config(), 'epochs': 5, 'batch_size': 64, 'learning_rate': .1}
    tr = torch.eye(18) * 10
    tr[1] = 0
    va = tr.clone()
    _, selected, p, best = fit_residual(tr, va, train, valid, cfg, tmp_path)
    # Reload must retain the selected nonzero correction.
    from fhf.v3_style import style_features
    head = make_head(cfg)
    head.load_state_dict(torch.load(tmp_path / 'style.pt', weights_only=True))
    head.eval()
    with torch.no_grad():
        restored = (va + best['alpha'] * head(torch.tensor([style_features(t) for t in valid.text]))).softmax(-1).numpy()
    np.testing.assert_allclose(restored, p, atol=0, rtol=0)
    assert selected['accuracy'] == 1. and best['alpha'] > 0


def test_tiny_frozen_eos_end_to_end(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config, Qwen3ForSequenceClassification, PreTrainedTokenizerFast
    from peft import PeftModelForSequenceClassification
    torch.set_num_threads(1)
    backend = Tokenizer(WordLevel({'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'sample': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    base = tmp_path / 'base'
    tokenizer.save_pretrained(base)
    Qwen3ForSequenceClassification(Qwen3Config(vocab_size=4, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_labels=18, pad_token_id=0)).save_pretrained(base)
    cfg = {'model': str(base), 'seed': 42, 'folds': 2, 'epochs': 1, 'max_length': 16,
           'batch_size': 7, 'accumulation': 2, 'learning_rate': .001, 'head_learning_rate': .001,
           'weight_decay': .01, 'warmup_ratio': .1, 'lora_rank': 2, 'lora_alpha': 4,
           'lora_dropout': .1, 'gradient_checkpointing': False, 'lexical_weight': .15}
    train, valid = data(0), data(1)
    source = tmp_path / 'original'
    run(train, cfg, source, valid=valid)
    adapter = source / 'fold-0/adapter/adapter_model.safetensors'
    checksum = hashlib.sha256(adapter.read_bytes()).hexdigest()
    forward = PeftModelForSequenceClassification.forward
    calls = []
    def checked_forward(self, *args, **kwargs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
        calls.append(1)
        return forward(self, *args, **kwargs)
    monkeypatch.setattr(PeftModelForSequenceClassification, 'forward', checked_forward)
    output = tmp_path / 'style'
    result = train_frozen(source, train, valid, config(), output, str(base))
    assert calls
    assert result['neural']['balanced_accuracy'] >= result['baseline']['balanced_accuracy']
    assert json.loads((output / 'restore_audit.json').read_text())['argmax_mismatches'] == 0
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == checksum
    expected = pd.read_csv(output / 'predictions.csv')[[f'neural::{n}' for n in MODEL_NAMES]].to_numpy()
    # Copied bundle must work even after original checkpoint directory moves.
    source.rename(tmp_path / 'moved-original')
    p = predict_frozen(output, valid, str(base), tmp_path / 'submission.parquet', 5)
    np.testing.assert_allclose(p, expected, atol=1e-6, rtol=1e-5)
    submitted = pd.read_parquet(tmp_path / 'submission.parquet')
    assert submitted.model.tolist() == np.asarray(MODEL_NAMES)[p.argmax(1)].tolist()
    assert submitted.id.tolist() == valid.id.tolist()
    source = tmp_path / 'moved-original'
    changed = train.copy()
    changed.loc[0, 'id'] = 'wrong'
    with pytest.raises(ValueError, match='Training split'):
        train_frozen(source, changed, valid, config(), tmp_path / 'bad', str(base))
    with pytest.raises(ValueError, match='Validation content'):
        train_frozen(source, train, valid.iloc[::-1], config(), tmp_path / 'bad', str(base))
    original_predictions = pd.read_csv(source / 'predictions.csv')
    original_predictions.loc[:, [f'neural::{n}' for n in MODEL_NAMES]] = 0.
    original_predictions['neural::' + MODEL_NAMES[0]] = 1.
    original_predictions.to_csv(source / 'predictions.csv', index=False)
    with pytest.raises(ValueError, match='Restored EOS predictions differ'):
        train_frozen(source, train, valid, config(), tmp_path / 'bad-restore', str(base))
