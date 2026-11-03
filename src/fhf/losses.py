"""Exact-class/family classification, hard-negative SupCon and consistency objectives."""
import torch
import torch.nn.functional as F


def contrastive_loss(embeddings, labels, families, temperature=0.12, hard_weight=2.0):
    """Supervised contrastive loss, upweighting other-model negatives in same family."""
    n = len(labels)
    if n < 2:
        return embeddings.sum() * 0
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    neg_same_family = families[:, None].eq(families[None, :]) & ~labels[:, None].eq(labels[None, :])
    # Weighted denominator increases pressure on near-family alternatives.
    weights = torch.ones_like(sim)
    weights = weights + neg_same_family.to(sim.dtype) * (hard_weight - 1)
    weighted = sim + weights.log()
    weighted = weighted.masked_fill(eye, float('-inf'))
    logp = sim - torch.logsumexp(weighted, dim=1, keepdim=True)
    counts = pos.sum(1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0
    return -(logp.masked_fill(~pos, 0).sum(1)[valid] / counts[valid]).mean()


def symmetric_kl(logp, logq):
    """Symmetric KL consistency between two stochastic views of identical text."""
    return 0.5 * (F.kl_div(logp, logq.exp(), reduction='batchmean') +
                  F.kl_div(logq, logp.exp(), reduction='batchmean'))


def objective(outputs, labels, families, family_weight=0.25, contrastive_weight=0.25,
              temperature=0.12, hard_weight=2.0):
    """Return differentiated classification and representation objectives."""
    cls = F.nll_loss(outputs['log_probs'], labels)
    family = F.cross_entropy(outputs['family_logits'], families)
    intra = contrastive_loss(outputs['embeddings'], labels, families, temperature, hard_weight)
    total = cls + family_weight * family + contrastive_weight * intra
    return total, {'cls': cls.detach(), 'family': family.detach(), 'contrastive': intra.detach()}
