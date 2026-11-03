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


def prompt_aware_pair_counts(labels, prompt_ids):
    """Count unordered different-prompt positives and same-prompt negatives."""
    if labels is None or prompt_ids is None or labels.shape != prompt_ids.shape:
        raise ValueError("Expected matched labels and prompt_ids tensors")
    mask = torch.triu(torch.ones(
        (len(labels), len(labels)), device=labels.device, dtype=torch.bool
    ), diagonal=1)
    same_model = labels[:, None].eq(labels[None, :])
    same_prompt = prompt_ids[:, None].eq(prompt_ids[None, :])
    positives = int((same_model & ~same_prompt & mask).sum().item())
    hard_negatives = int((~same_model & same_prompt & mask).sum().item())
    return positives, hard_negatives


def prompt_aware_contrastive_loss(embeddings, labels, families, prompt_ids,
                                  temperature=0.12, family_weight=2.0,
                                  same_prompt_weight=3.0):
    """Supervised contrastive with prompt-controlled positive/negative pairs.

    Positive: same exact model and *different* prompt.
    Hard negative: different exact models answering the *same* prompt.
    The same-model/same-prompt duplicate is ignored, not a false negative.
    """
    if prompt_ids is None or prompt_ids.shape != labels.shape:
        raise ValueError("Prompt-aware contrastive requires aligned prompt_ids")
    if not 0 < temperature or not family_weight >= 1 or not same_prompt_weight >= 1:
        raise ValueError("Contrastive temperature must be positive; weights >= 1")
    n = len(labels)
    if n < 2:
        return embeddings.sum() * 0
    same_model = labels[:, None].eq(labels[None, :])
    same_prompt = prompt_ids[:, None].eq(prompt_ids[None, :])
    eye = torch.eye(n, device=labels.device, dtype=torch.bool)
    positive = same_model & ~same_prompt & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0

    within_family_negative = (~same_model) & families[:, None].eq(families[None, :])
    same_prompt_negative = (~same_model) & same_prompt
    weights = torch.ones((n, n), device=embeddings.device, dtype=torch.float32)
    weights = weights * (
        1.0 + within_family_negative.to(torch.float32) * (family_weight - 1.0)
    )
    weights = weights * (
        1.0 + same_prompt_negative.to(torch.float32) * (same_prompt_weight - 1.0)
    )
    # A repeated response to the same prompt/model is not a valid positive
    # or negative; exclude it from the contrastive denominator.
    exclude = eye | (same_model & same_prompt)
    scores = torch.matmul(embeddings.float(), embeddings.float().T) / temperature
    denom = torch.logsumexp((scores + weights.log()).masked_fill(exclude, -float("inf")), dim=1)
    log_probs = scores - denom.unsqueeze(1)
    num_pos = positive.sum(dim=1)
    per_anchor = -(log_probs.masked_fill(~positive, 0).sum(dim=1)[valid] /
                   num_pos[valid])
    return per_anchor.mean()


def symmetric_kl(logp, logq):
    """Symmetric KL on two dropout views of the same unchanged source text."""
    return 0.5 * (F.kl_div(logp, logq.exp(), reduction="batchmean") +
                  F.kl_div(logq, logp.exp(), reduction="batchmean"))


def objective(outputs, labels, families, config, prompt_ids=None):
    """Evaluate only enabled losses; omit disabled heads from the computational graph."""
    parts = {}
    specs = config["losses"]
    if specs["exact_classification"]["enabled"] and specs["exact_classification"]["weight"] > 0:
        parts["exact_classification"] = F.nll_loss(outputs["log_probs"], labels)
    flat = specs.get("flat_classification", {"enabled": False, "weight": 0.0})
    if flat.get("enabled", False) and flat.get("weight", 0) > 0:
        parts["flat_classification"] = F.nll_loss(outputs["flat_log_probs"], labels)
    if specs["family"]["enabled"] and specs["family"]["weight"] > 0:
        parts["family"] = F.cross_entropy(outputs["family_logits"], families)
    if specs["intra_contrastive"]["enabled"] and specs["intra_contrastive"]["weight"] > 0:
        spec = specs["intra_contrastive"]
        if spec.get("mode", "standard") == "prompt_aware":
            parts["intra_contrastive"] = prompt_aware_contrastive_loss(
                outputs["embeddings"], labels, families, prompt_ids,
                temperature=float(spec["temperature"]),
                family_weight=float(spec["hard_weight"]),
                same_prompt_weight=float(spec.get("same_prompt_weight", 3.0)),
            )
        else:
            parts["intra_contrastive"] = contrastive_loss(
                outputs["embeddings"], labels, families,
                float(spec["temperature"]), float(spec["hard_weight"]))
    if not parts:
        raise ValueError("No enabled base objective")
    loss = sum(parts[name] * float(specs[name]["weight"]) for name in parts)
    return loss, {name: value.detach() for name, value in parts.items()}
