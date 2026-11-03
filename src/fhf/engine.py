"""Reproducible train/validation loops and batch prediction for FHF."""
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import TextDataset, make_collator
from .losses import objective, symmetric_kl
from .hotflip import hotflip_batch
from .metrics import report


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch for comparable experimental runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, loader, device):
    """Return predicted IDs, optionally true labels, without stochastic dropout."""
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            if 'labels' in batch:
                targets.extend(batch.pop('labels').tolist())
                batch.pop('families')
            outputs = model(**{k:v.to(device) for k,v in batch.items()})
            predictions.extend(outputs['log_probs'].argmax(dim=-1).cpu().tolist())
    return predictions, targets


def fit(model, tokenizer, train_loader, valid_loader, out_dir, epochs=5,
        lr=2e-5, weight_decay=0.01, family_weight=0.25,
        contrastive_weight=0.25, consistency_weight=0.05, hotflip_weight=0.,
        hotflip_max_examples=2, grad_clip=1., device='cpu'):
    """Fine-tune classification heads + encoder; retain best prompt-held-out checkpoint."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model._fhf_tokenizer = tokenizer  # Used only by optional HotFlip token filtering.
    best = -float('inf')
    history = []
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            labels, families = batch.pop('labels'), batch.pop('families')
            inputs = batch
            optimizer.zero_grad(set_to_none=True)
            out = model(**inputs)
            loss, _ = objective(out, labels, families, family_weight, contrastive_weight)
            if consistency_weight > 0:
                out2 = model(**inputs)
                loss = loss + consistency_weight * symmetric_kl(out['log_probs'], out2['log_probs'])
            if hotflip_weight > 0:
                sample = dict(inputs, labels=labels, families=families)
                changed_ids = hotflip_batch(model, sample, hotflip_max_examples)
                if changed_ids is not None:
                    altered = model(input_ids=changed_ids, attention_mask=inputs['attention_mask'])
                    # Only use original prediction as a soft consistency target, not as ground truth for edits.
                    loss = loss + hotflip_weight * F.kl_div(
                        altered['log_probs'], out['log_probs'].detach().exp(), reduction='batchmean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item()
            count += 1
        preds, truth = predict(model, valid_loader, device)
        metrics = report(truth, preds)
        score = metrics['balanced_accuracy_observed_classes']
        history.append(dict(epoch=epoch+1, training_loss=total/max(count,1), metrics=metrics))
        print(f'Epoch {epoch+1}: loss={total/max(count,1):.4f}; val balanced acc={score:.4f}')
        if score > best:
            best = score
            model.save_checkpoint(out_dir, tokenizer, extra={'best_epoch':epoch+1,'validation_balanced_accuracy':score})
    return history
