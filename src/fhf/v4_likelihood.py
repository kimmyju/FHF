"""Experimental 18-source likelihood attribution, inspired by ATLAS.

One frozen Qwen3 CausalLM; shared LoRA A, source-specific B and one soft prefix.
No source-name generation, EOS classifier, or online access during inference.
"""
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file, load_file

from .labels import MODEL_NAMES, MODEL_TO_ID
from .v3_decoder import validate, disjoint, prepare_external, normalize, metrics, dump


class SourceLinear(nn.Module):
    def __init__(self, original, rank, alpha, dropout, sources=18):
        super().__init__()
        self.original = original
        self.A = nn.Parameter(torch.empty(rank, original.in_features, dtype=torch.float32,
                                          device=original.weight.device))
        self.B = nn.Parameter(torch.zeros(sources, original.out_features, rank, dtype=torch.float32,
                                          device=original.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        self.source = 0

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x).to(self.A.dtype), self.A), self.B[self.source])
        return self.original(x) + (update * self.scale).to(x.dtype)


class SourceLM(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        if base.config.model_type != 'qwen3':
            raise ValueError('v4 currently supports Qwen3 CausalLM only')
        self.base = base
        base.requires_grad_(False)
        self.cfg = cfg
        targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
        count = 0
        for name, module in list(base.model.named_modules()):
            if name.rsplit('.', 1)[-1] in targets:
                if not isinstance(module, nn.Linear):
                    raise ValueError('Expected dense attention projection; quantization not implemented')
                parent_name, _, attr = name.rpartition('.')
                parent = base.model.get_submodule(parent_name) if parent_name else base.model
                setattr(parent, attr, SourceLinear(module, int(cfg['rank']), float(cfg['alpha']), float(cfg['dropout'])))
                count += 1
        if count != 4 * base.config.num_hidden_layers:
            raise ValueError('Did not wrap all Qwen3 attention projections')
        token_id = base.config.eos_token_id
        token_id = token_id[0] if isinstance(token_id, list) else token_id
        token_id = 0 if token_id is None else token_id
        initial = base.get_input_embeddings().weight[token_id].detach().float()
        self.prefix = nn.Parameter(initial.repeat(18, 1))
        base.config.use_cache = False
        if cfg.get('gradient_checkpointing'):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    @property
    def device(self):
        return self.prefix.device

    def token_logps(self, ids, mask, source):
        if not 0 <= source < 18:
            raise ValueError('Invalid source')
        for module in self.base.model.modules():
            if isinstance(module, SourceLinear):
                module.source = source
        embeds = self.base.get_input_embeddings()(ids)
        prefix = self.prefix[source].to(embeds.dtype)[None, None].expand(len(ids), 1, -1)
        inputs = torch.cat([prefix, embeds], dim=1)
        attention = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        # Prefix predicts token 0; text position t predicts token t+1.
        hidden = self.base.model(inputs_embeds=inputs, attention_mask=attention,
                                 use_cache=False, return_dict=True).last_hidden_state[:, :-1]
        flat, target = hidden.reshape(-1, hidden.shape[-1]), ids.reshape(-1)
        parts = []
        chunk = int(self.cfg['logit_chunk_tokens'])
        def logps(h, y):
            logits = self.base.lm_head(h).float()
            return -F.cross_entropy(logits, y, reduction='none')
        for start in range(0, len(target), chunk):
            h, y = flat[start:start + chunk], target[start:start + chunk]
            value = checkpoint(logps, h, y, use_reentrant=False) if self.training else logps(h, y)
            parts.append(value)
        return torch.cat(parts).reshape_as(ids)

    def save_adapter(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        state = {n: p.detach().cpu().contiguous() for n, p in self.named_parameters() if p.requires_grad}
        save_file(state, str(destination / 'source_adapter.safetensors'))

    def load_adapter(self, destination):
        state = load_file(str(Path(destination) / 'source_adapter.safetensors'))
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(params) != set(state):
            raise ValueError('Source adapter keys do not match architecture')
        with torch.no_grad():
            for name, p in params.items():
                if p.shape != state[name].shape:
                    raise ValueError(f'Adapter shape mismatch: {name}')
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def build(base_path, cfg, offline=False):
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if not torch.cuda.is_available() else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    # FP16 is supported for inference. Training on non-BF16 GPUs is rejected below.
    base = AutoModelForCausalLM.from_pretrained(base_path, revision=cfg.get('revision', 'main'),
        local_files_only=offline, torch_dtype=dtype, attn_implementation='sdpa')
    return SourceLM(base, cfg).to('cuda' if torch.cuda.is_available() else 'cpu')


def encode_texts(texts, tokenizer, maximum):
    if maximum < 2:
        raise ValueError('max_length must be >=2')
    rows, truncated = [], 0
    for text in texts:
        ids = tokenizer(str(text), add_special_tokens=False)['input_ids']
        if not ids:
            raise ValueError('Text encodes to zero tokens')
        truncated += len(ids) > maximum
        # Keep a contiguous prefix; joining head/tail would create fake LM targets.
        rows.append(ids[:maximum])
    return rows, {'rows': len(rows), 'truncated': int(truncated), 'max_text_tokens': maximum}


def batch_rows(rows, pad, device):
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
        mask[i, :len(row)] = 1
    return ids, mask


def relative_scores(logps, mask, temperatures):
    """[batch,source,token] -> bounded ATLAS-style scores; not calibrated posteriors."""
    score = torch.zeros(logps.shape[:2], dtype=torch.float32, device=logps.device)
    for k in range(logps.shape[1]):
        penalty = (logps - logps[:, k:k+1]).clamp_min(0).sum(1)
        for temperature in temperatures:
            score[:, k] += ((-penalty / temperature).exp() * mask).sum(1) / mask.sum(1)
    return score / len(temperatures)


def predict_scores(model, rows, tokenizer, cfg):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), int(cfg['eval_batch_size'])):
            ids, mask = batch_rows(rows[start:start + int(cfg['eval_batch_size'])], tokenizer.pad_token_id, model.device)
            token_scores = torch.stack([model.token_logps(ids, mask, k).cpu() for k in range(18)], dim=1)
            values = relative_scores(token_scores, mask.cpu(), cfg['temperatures']).numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite source likelihood scores')
            parts.append(values)
            done = min(start + int(cfg['eval_batch_size']), len(rows))
            if done % 100 == 0 or done == len(rows):
                print(f'Likelihood evaluation: {done}/{len(rows)} texts, 18 sources each', flush=True)
    return np.concatenate(parts)


def check_config(cfg):
    for key in ['rank', 'max_length', 'logit_chunk_tokens', 'batch_size', 'accumulation',
                'eval_batch_size', 'external_epochs', 'official_epochs', 'external_per_class']:
        if int(cfg[key]) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ['external_learning_rate', 'official_learning_rate', 'alpha']:
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 <= cfg['dropout'] < 1 or not 0 <= cfg['warmup_ratio'] < 1:
        raise ValueError('Invalid dropout/warmup')
    if not cfg['temperatures'] or any(not math.isfinite(t) or t <= 0 for t in cfg['temperatures']):
        raise ValueError('Invalid temperatures')


def audit_data(train, valid, external):
    train, valid, external = validate(train), validate(valid), validate(external)
    disjoint(train, valid)
    report = {'warning': 'Automated checks cannot verify exact generator version or semantic prompt equivalence.',
              'splits': {}}
    for name, frame in [('train', train), ('valid', valid), ('external', external)]:
        words = frame.text.str.split().str.len()
        item = {'rows': len(frame), 'counts': frame.model.value_counts().to_dict(),
                'word_lengths_by_model': {k: words.loc[g.index].quantile([0, .5, .95, 1]).to_dict()
                                          for k, g in frame.groupby('model')},
                'models_per_prompt': frame.groupby('prompt_id').model.nunique().value_counts().to_dict()}
        for column in ['source_dataset', 'origin']:
            if column in frame:
                item[column + '_by_model'] = pd.crosstab(frame.model, frame[column]).to_dict()
        report['splits'][name] = item
    official = pd.concat([train, valid])
    report['external_exact_response_overlap'] = int(external.text.map(normalize).isin(set(official.text.map(normalize))).sum())
    report['external_prompt_id_overlap'] = int(external.prompt_id.isin(set(official.prompt_id)).sum())
    return train, valid, external, report


def train_stage(model, rows, labels, tokenizer, cfg, stage, callback, tracker=None):
    from transformers import get_cosine_schedule_with_warmup
    epochs = int(cfg[stage + '_epochs'])
    batch, accumulation = int(cfg['batch_size']), int(cfg['accumulation'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg[stage + '_learning_rate']), weight_decay=cfg['weight_decay'])
    steps = math.ceil(len(rows) / (batch * accumulation)) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * cfg['warmup_ratio']), steps)
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=18)
    weights = len(labels) / (18 * counts)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(rows))
        total_loss = 0.
        for start in range(0, len(rows), batch * accumulation):
            window = order[start:start + batch * accumulation]
            optimizer.zero_grad(set_to_none=True)
            # Homogeneous source microbatches: same adapter throughout forward
            # AND backward, including gradient-checkpoint recomputation.
            for source in np.unique(labels[window]):
                indices = window[labels[window] == source]
                for offset in range(0, len(indices), batch):
                    ix = indices[offset:offset + batch]
                    ids, mask = batch_rows([rows[i] for i in ix], tokenizer.pad_token_id, model.device)
                    logps = model.token_logps(ids, mask, int(source))
                    per_text = -(logps * mask).sum(1) / mask.sum(1)
                    summed = per_text.sum() * float(weights[source])
                    if not torch.isfinite(summed):
                        raise ValueError(f'Nonfinite training NLL at {stage} epoch {epoch}')
                    (summed / len(window)).backward()
                    total_loss += summed.detach().item()
            nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            scheduler.step()
            update = start // (batch * accumulation) + 1
            if update % 100 == 0:
                print(f'{stage} epoch {epoch}: update {update}/{math.ceil(len(rows) / (batch * accumulation))}', flush=True)
        record = {'stage': stage, 'epoch': epoch, 'train_nll': total_loss / len(rows), **callback(stage, epoch)}
        print(json.dumps(record), flush=True)
        history.append(record)
        if tracker:
            tracker.log({f'{stage}/{k}': v for k, v in record.items() if k != 'stage'})
    return history


def train(train, valid, external, cfg, output, tracker=None):
    from transformers import AutoTokenizer, set_seed
    check_config(cfg)
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError('v4 training requires BF16 GPU (or CPU tests); FP16 GPU inference is supported')
    train, valid, external, audit = audit_data(train, valid, external)
    out = Path(output)
    if out.exists():
        raise ValueError('Output exists; choose a fresh v4 run name')
    external = prepare_external(external, pd.concat([train, valid]), cfg['external_per_class'], cfg['seed'])
    audit['external_after_filter'] = {'rows': len(external), 'counts': external.model.value_counts().to_dict()}
    out.mkdir(parents=True)
    dump(out / 'data_audit.json', audit)
    set_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer needs pad_token_id')
    tokenizer.save_pretrained(out / 'tokenizer')
    datasets = {}
    for name, frame in [('external', external), ('official', train), ('valid', valid)]:
        datasets[name], stats = encode_texts(frame.text, tokenizer, cfg['max_length'])
        dump(out / f'{name}_tokenization.json', stats)
    model = build(cfg['model'], cfg)
    cfg = {**cfg, 'revision': getattr(model.base.config, '_commit_hash', None) or cfg['revision']}
    dump(out / 'config.json', cfg)
    dump(out / 'labels.json', MODEL_NAMES)
    dump(out / 'parameters.json', {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  'total': sum(p.numel() for p in model.parameters())})
    y = valid.model.map(MODEL_TO_ID).to_numpy()
    reports, history, best = [], [], {}
    selected = None
    def evaluate(stage, epoch):
        nonlocal selected
        scores = predict_scores(model, datasets['valid'], tokenizer, cfg)
        report = metrics(y, scores)
        folder = out / 'epochs' / f'{stage}-{epoch}'
        model.save_adapter(folder)
        np.save(folder / 'validation_scores.npy', scores)
        eligible = stage == 'official' or epoch == cfg['external_epochs']
        reports.append({'stage': stage, 'epoch': epoch, 'eligible': eligible, **report})
        dump(out / 'epoch_metrics.json', reports)
        if eligible and (not best or report['balanced_accuracy'] > best['balanced_accuracy']):
            best.update(stage=stage, epoch=epoch, balanced_accuracy=report['balanced_accuracy'],
                        adapter=str(folder.relative_to(out)))
            selected = scores.copy()
            dump(out / 'selection.json', best)
        return {'val_balanced_accuracy': report['balanced_accuracy'], 'val_family_accuracy': report['family_accuracy'],
                'val_within_family_given_correct': report['within_family_given_correct']}
    for name, frame in [('external', external), ('official', train)]:
        history += train_stage(model, datasets[name], frame.model.map(MODEL_TO_ID), tokenizer, cfg, name, evaluate, tracker)
        dump(out / 'history.json', history)
    result = {'neural': metrics(y, selected), 'selection': best, 'rows': len(valid),
              'evaluation': 'validation_selected_holdout',
              'data_sha256': hashlib.sha256(valid.to_csv(index=False).encode()).hexdigest(),
              'score_type': 'bounded_relative_token_likelihood; not calibrated probabilities'}
    dump(out / 'metrics.json', result)
    predicted = valid[['id', 'prompt_id', 'model']].copy()
    for k, name in enumerate(MODEL_NAMES):
        predicted['score::' + name] = selected[:, k]
    predicted.to_csv(out / 'predictions.csv', index=False)
    dump(out / 'manifest.json', {'kind': 'source_likelihood_v1', 'labels': MODEL_NAMES, 'adapter': best['adapter']})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def predict(bundle, test, base, output, batch_size=None):
    from transformers import AutoTokenizer
    started = time.monotonic()
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['kind'] != 'source_likelihood_v1' or manifest['labels'] != MODEL_NAMES:
        raise ValueError('Invalid likelihood bundle')
    cfg = json.loads((bundle / 'config.json').read_text())
    if batch_size is not None:
        cfg['eval_batch_size'] = batch_size
    check_config(cfg)
    test = validate(test, labeled=False)
    tokenizer = AutoTokenizer.from_pretrained(bundle / 'tokenizer', local_files_only=True)
    rows, _ = encode_texts(test.text, tokenizer, cfg['max_length'])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = build(base, cfg, offline=True)
    model.load_adapter(bundle / manifest['adapter'])
    scores = predict_scores(model, rows, tokenizer, cfg)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite attribution scores')
    output = Path(output)
    if output.exists():
        raise ValueError('Submission already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': test.id, 'model': np.asarray(MODEL_NAMES)[scores.argmax(1)]}).to_parquet(output, index=False)
    seconds = time.monotonic() - started
    dump(str(output) + '.runtime.json', {'rows': len(test), 'seconds': seconds,
        'estimated_seconds_4000_rows': seconds / len(test) * 4000,
        'peak_gpu_allocated_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        'note': 'Measure on the actual competition GPU; estimate includes model loading and depends on text lengths.'})
    return scores
