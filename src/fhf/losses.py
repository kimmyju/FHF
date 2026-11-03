"""Switchable exact-class, family, hard-negative contrastive and KL consistency losses."""
import torch
import torch.nn.functional as F


def contrastive_loss(embeddings, labels, families, temperature=0.12, hard_weight=2.0):
    """Attract same-exact-model positives; weight negatives from the same provider more."""
    n = len(labels)
    if n < 2:
        return embeddings.sum() * 0
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    within_family_neg = families[:, None].eq(families[None, :]) & ~labels[:, None].eq(labels[None, :])
    weights = torch.ones_like(sim) + within_family_neg.to(sim.dtype) * (hard_weight - 1)
    log_denominator = torch.logsumexp((sim + weights.log()).masked_fill(eye, -float("inf")), dim=1)
    log_probability = sim - log_denominator[:, None]
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0
    return -(log_probability.masked_fill(~positive, 0).sum(1)[valid] / counts[valid]).mean()


def symmetric_kl(logp, logq):
    """Symmetric KL on two dropout views of the same unchanged source text."""
    return 0.5 * (F.kl_div(logp, logq.exp(), reduction="batchmean") +
                  F.kl_div(logq, logp.exp(), reduction="batchmean"))


def objective(outputs, labels, families, config):
    """Evaluate only enabled losses; omit disabled heads from the computational graph."""
    parts = {}
    specs = config["losses"]
    if specs["exact_classification"]["enabled"] and specs["exact_classification"]["weight"] > 0:
        parts["exact_classification"] = F.nll_loss(outputs["log_probs"], labels)
    if specs["family"]["enabled"] and specs["family"]["weight"] > 0:
        parts["family"] = F.cross_entropy(outputs["family_logits"], families)
    if specs["intra_contrastive"]["enabled"] and specs["intra_contrastive"]["weight"] > 0:
        spec = specs["intra_contrastive"]
        parts["intra_contrastive"] = contrastive_loss(
            outputs["embeddings"], labels, families,
            float(spec["temperature"]), float(spec["hard_weight"]))
    if not parts:
        raise ValueError("No enabled base objective")
    loss = sum(parts[name] * float(specs[name]["weight"]) for name in parts)
    return loss, {name: value.detach() for name, value in parts.items()}
