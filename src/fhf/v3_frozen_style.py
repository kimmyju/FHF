"""Frozen EOS logits + train-only style residual; validation selects epoch/alpha."""
import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .labels import MODEL_NAMES, MODEL_TO_ID
from .v3_decoder import validate, disjoint, encode, collate_rows, get_base, metrics, dump
from .v3_style import FEATURE_NAMES, FEATURE_VERSION, style_features


def make_head(cfg):
    if cfg['style_feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported style feature version')
    head = nn.Sequential(nn.Linear(len(FEATURE_NAMES), int(cfg['hidden_dim'])), nn.GELU(),
                         nn.Dropout(float(cfg['dropout'])), nn.Linear(int(cfg['hidden_dim']), 18))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def frozen_logits(fold, texts, batch_size, base=None, split_at=None):
    from peft import PeftModel
    from transformers import AutoTokenizer
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    fold = Path(fold)
    cfg = json.loads((fold / 'config.json').read_text())
    if cfg.get('pooling', 'eos') != 'eos':
        raise ValueError('Source must be the original EOS checkpoint, not EOS-mean/style')
    tokenizer = AutoTokenizer.from_pretrained(fold / 'tokenizer', local_files_only=True)
    tokenizer.padding_side = 'right'
    model = PeftModel.from_pretrained(
        get_base(base or cfg['model'], cfg, tokenizer, offline=base is not None),
        fold / 'adapter', is_trainable=False, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    rows, _ = encode(texts, tokenizer, int(cfg['max_length']), cfg)
    # Preserve original validation batch boundaries and therefore padding shapes.
    if split_at is not None and not 0 < split_at < len(rows):
        raise ValueError('split_at must separate two nonempty datasets')
    boundaries = [0, len(rows)] if split_at is None else [0, split_at, len(rows)]
    parts = []
    with torch.inference_mode():
        for lower, upper in zip(boundaries, boundaries[1:]):
            for start in range(lower, upper, batch_size):
                batch = collate_rows(rows[start:min(start + batch_size, upper)], tokenizer)
                parts.append(model(**{k: v.to(model.device) for k, v in batch.items()}).logits.float().cpu())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(parts)


def probabilities(logits):
    return logits.softmax(-1).detach().cpu().numpy()


def fit_residual(train_logits, valid_logits, train, valid, cfg, out, tracker=None):
    """CPU-only optimization: EOS is not present in the optimization graph."""
    out = Path(out)
    torch.manual_seed(int(cfg['seed']))
    epochs, batch = int(cfg['epochs']), int(cfg['batch_size'])
    alphas = [float(a) for a in cfg['alphas']]
    if epochs < 1 or batch < 1 or not alphas or any(not np.isfinite(a) or a < 0 for a in alphas):
        raise ValueError('Invalid epochs/batch/alphas')
    if 0.0 not in alphas:
        raise ValueError('Include alpha=0 as baseline')
    x = torch.tensor([style_features(t) for t in train.text], dtype=torch.float32)
    xv = torch.tensor([style_features(t) for t in valid.text], dtype=torch.float32)
    y = torch.tensor(train.model.map(MODEL_TO_ID).to_numpy())
    yv = valid.model.map(MODEL_TO_ID).to_numpy()
    train_logits, valid_logits = train_logits.detach().float(), valid_logits.detach().float()
    head = make_head(cfg)
    head.eval()
    with torch.no_grad():
        torch.testing.assert_close(valid_logits + head(xv), valid_logits, rtol=0, atol=0)
    baseline = metrics(yv, probabilities(valid_logits))
    best = {'epoch': 0, 'alpha': 0., 'balanced_accuracy': baseline['balanced_accuracy']}
    best_state = copy.deepcopy(head.state_dict())
    best_probs = probabilities(valid_logits)
    counts = torch.bincount(y, minlength=18)
    if (counts == 0).any():
        raise ValueError('All 18 training labels required')
    weights = len(y) / (18 * counts.float())
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg['learning_rate']),
                                 weight_decay=float(cfg['weight_decay']))
    history = [{'epoch': 0, 'alpha': 0., **baseline}]
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(y))
        loss_sum = 0.
        for ix in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(train_logits[ix] + head(x[ix]), y[ix],
                                                weight=weights, reduction='sum') / len(ix)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            loss_sum += loss.item() * len(ix)
        head.eval()
        with torch.no_grad():
            correction = head(xv)
        torch.save(head.state_dict(), out / f'style-epoch-{epoch}.pt')
        for alpha in alphas:
            p = probabilities(valid_logits + alpha * correction)
            report = metrics(yv, p)
            history.append({'epoch': epoch, 'alpha': alpha, 'train_loss': loss_sum / len(y), **report})
            if report['balanced_accuracy'] > best['balanced_accuracy']:
                best = {'epoch': epoch, 'alpha': alpha, 'balanced_accuracy': report['balanced_accuracy']}
                best_state, best_probs = copy.deepcopy(head.state_dict()), p.copy()
        print(json.dumps({'epoch': epoch, 'train_loss': loss_sum / len(y), 'best': best}), flush=True)
        if tracker:
            tracker.log({'epoch': epoch, 'train_loss': loss_sum / len(y), 'best_bacc': best['balanced_accuracy']})
        dump(out / 'history.json', history)
    torch.save(best_state, out / 'style.pt')
    dump(out / 'selection.json', best)
    return baseline, metrics(yv, best_probs), best_probs, best


def train_frozen(source, train, valid, cfg, output, base=None, tracker=None):
    source, out = Path(source).resolve(), Path(output).resolve()
    train, valid = validate(train), validate(valid)
    disjoint(train, valid)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['labels'] != MODEL_NAMES or len(manifest['folds']) != 1 or manifest['evaluation'] != 'frozen_holdout':
        raise ValueError('Requires single-fold frozen_holdout source bundle')
    fold = (source / manifest['folds'][0]).resolve()
    source_cfg = json.loads((fold / 'config.json').read_text())
    if source_cfg.get('pooling', 'eos') != 'eos':
        raise ValueError('Source must be original EOS')
    saved_ids = pd.read_csv(fold / 'training_ids.csv', dtype=str)
    columns = ['id', 'prompt_id', 'model']
    if not saved_ids[columns].sort_values('id').reset_index(drop=True).equals(
            train[columns].astype(str).sort_values('id').reset_index(drop=True)):
        raise ValueError('Training split does not match source checkpoint')
    original = json.loads((source / 'metrics.json').read_text())
    data_hash = hashlib.sha256(valid.to_csv(index=False).encode()).hexdigest()
    if data_hash != original['data_sha256']:
        raise ValueError('Validation content/order differs from source run')
    if out.exists():
        raise ValueError('Output exists; choose a new run name')
    out.mkdir(parents=True)
    # Copy the small adapter and tokenizer, never the multi-GB base weights.
    saved = out / 'source'
    saved.mkdir()
    for name in ['adapter', 'tokenizer']:
        shutil.copytree(fold / name, saved / name)
    shutil.copy2(fold / 'config.json', saved / 'config.json')
    dump(out / 'config.json', cfg)
    dump(out / 'provenance.json', {'source_bundle': str(source), 'source_fold': str(fold),
                                  'source_metrics': original['neural'], 'data_sha256': data_hash})
    print('Restoring frozen EOS; caching train and validation logits (no backward through EOS).', flush=True)
    # The original v3 run evaluated valid alone with source_cfg['batch_size'].
    # Do not shift validation batches by the training row count or override size.
    restore_batch_size = int(source_cfg['batch_size'])
    logits = frozen_logits(saved, pd.concat([train.text, valid.text]), restore_batch_size,
                           base, split_at=len(train))
    tr, va = logits[:len(train)], logits[len(train):]
    reference = pd.read_csv(source / 'predictions.csv', dtype={'id': str}).set_index('id').loc[valid.id]
    expected = reference[[f'neural::{n}' for n in MODEL_NAMES]].to_numpy()
    actual = probabilities(va)
    y = valid.model.map(MODEL_TO_ID).to_numpy()
    difference = np.abs(expected - actual)
    changed = expected.argmax(1) != actual.argmax(1)
    import transformers
    import peft
    audit = {'argmax_mismatches': int(changed.sum()),
             'max_probability_difference': float(difference.max()),
             'probability_difference_quantiles': np.quantile(difference, [0.5, 0.9, 0.99, 1.]).tolist(),
             'original_bacc': metrics(y, expected)['balanced_accuracy'],
             'restored_bacc': metrics(y, actual)['balanced_accuracy'],
             'source_batch_size': restore_batch_size,
             'validation_batches_separate': True,
             'model': source_cfg['model'], 'revision': source_cfg.get('revision'),
             'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
             'peft_version': peft.__version__,
             'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
             'all_eos_parameters_frozen': True}
    differences = valid[['id', 'model']].copy()
    differences['original_prediction'] = np.asarray(MODEL_NAMES)[expected.argmax(1)]
    differences['restored_prediction'] = np.asarray(MODEL_NAMES)[actual.argmax(1)]
    differences['max_probability_difference'] = difference.max(1)
    differences['original_top2_margin'] = np.diff(np.sort(expected, axis=1)[:, -2:], axis=1).ravel()
    differences['restored_top2_margin'] = np.diff(np.sort(actual, axis=1)[:, -2:], axis=1).ravel()
    differences.loc[changed].to_csv(out / 'restore_mismatches.csv', index=False)
    dump(out / 'restore_audit.json', audit)
    if audit['argmax_mismatches'] or not np.allclose(actual, expected, atol=2e-3, rtol=2e-3):
        raise ValueError('Restored EOS predictions differ from source; inspect restore_audit.json before training')
    np.savez_compressed(out / 'frozen_logits.npz', train=tr.numpy(), valid=va.numpy())
    baseline, selected, p, best = fit_residual(tr, va, train, valid, cfg, out, tracker)
    result = {'baseline': baseline, 'neural': selected, 'selection': best,
              'evaluation': 'validation_selected_holdout', 'rows': len(valid),
              'note': 'Epoch/alpha selected on this validation set; not an unbiased test score.'}
    dump(out / 'metrics.json', result)
    rows = valid[['id', 'prompt_id', 'model']].copy()
    for i, name in enumerate(MODEL_NAMES):
        rows[f'neural::{name}'] = p[:, i]
    rows.to_csv(out / 'predictions.csv', index=False)
    dump(out / 'manifest.json', {'kind': 'frozen_eos_style_v1', 'source_fold': 'source',
                                'labels': MODEL_NAMES, 'alpha': best['alpha']})
    return result


def predict_frozen(bundle, test, base, output, batch_size=2):
    bundle = Path(bundle)
    test = validate(test, labeled=False)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['kind'] != 'frozen_eos_style_v1' or manifest['labels'] != MODEL_NAMES:
        raise ValueError('Invalid frozen-style bundle')
    cfg = json.loads((bundle / 'config.json').read_text())
    head = make_head(cfg)
    head.load_state_dict(torch.load(bundle / 'style.pt', map_location='cpu', weights_only=True))
    head.eval()
    logits = frozen_logits(bundle / manifest['source_fold'], test.text, batch_size, base)
    with torch.no_grad():
        features = torch.tensor([style_features(t) for t in test.text], dtype=torch.float32)
        p = probabilities(logits + float(manifest['alpha']) * head(features))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': test.id, 'model': np.asarray(MODEL_NAMES)[p.argmax(1)]}).to_parquet(output, index=False)
    return p
