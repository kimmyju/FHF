"""Switchable exact-class, family, hard-negative contrastive and KL consistency losses."""
import torch
import torch.nn.functional as F
from .labels import MODEL_NAMES, MODEL_TO_ID, MODEL_TO_FAMILY


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



def within_family_prompt_contrastive_loss(
    embeddings, labels, families, prompt_ids,
    temperature=0.12, same_prompt_weight=3.0, hard_pair_weights=None,
):
    """Contrast only among distinct exact models belonging to the same family.

    Positives: same exact model with DIFFERENT prompts.
    Negatives: different exact models of the SAME family.
    Other families and same-model/same-prompt duplicates are excluded.
    Same-prompt negatives may be upweighted. An anchor contributes only
    if it has both a valid positive and at least one within-family negative.
    Singleton-model families therefore have no within-family contrastive
    gradient; their supervised classification losses remain active.
    """
    if prompt_ids is None or prompt_ids.shape != labels.shape:
        raise ValueError("Within-family contrastive requires aligned prompt_ids")
    if not temperature > 0 or same_prompt_weight < 1:
        raise ValueError("Temperature must be positive and same_prompt_weight >= 1")
    n = len(labels)
    if n < 2:
        return embeddings.sum() * 0
    same_model = labels[:, None].eq(labels[None, :])
    same_family = families[:, None].eq(families[None, :])
    same_prompt = prompt_ids[:, None].eq(prompt_ids[None, :])
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    positives = same_model & ~same_prompt & ~eye
    negatives = same_family & ~same_model
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0

    # FP32 logits avoid half precision underflow at low temperature.
    sim = embeddings.float() @ embeddings.float().T / temperature
    allowed = positives | negatives
    log_weights = torch.zeros_like(sim)
    log_weights = log_weights.masked_fill(
        negatives & same_prompt, float(torch.log(torch.tensor(
            same_prompt_weight, dtype=torch.float32
        )))
    )
    if hard_pair_weights:
        # Optional hard model-pair confusion weights, computed strictly from
        # official TRAIN out-of-fold predictions (never held-out validation).
        pair_weight = torch.ones(
            (len(MODEL_NAMES), len(MODEL_NAMES)), dtype=torch.float32,
            device=sim.device)
        for model_a, targets in hard_pair_weights.items():
            if model_a not in MODEL_TO_ID or not isinstance(targets, dict):
                raise ValueError(f"Unknown or malformed hard-pair source: {model_a}")
            for model_b, scale in targets.items():
                if model_b not in MODEL_TO_ID or MODEL_TO_FAMILY[model_a] != MODEL_TO_FAMILY[model_b]:
                    raise ValueError("Hard pairs must use two models in the same provider")
                if not 1 <= float(scale) <= 10:
                    raise ValueError("Hard-pair weights must lie in [1,10]")
                pair_weight[MODEL_TO_ID[model_a], MODEL_TO_ID[model_b]] = float(scale)
        log_weights = log_weights + pair_weight[
            labels[:, None], labels[None, :]].log() * negatives

    denominator = torch.logsumexp(
        (sim + log_weights).masked_fill(~allowed, -float("inf")),
        dim=1,
    )
    log_probs = sim - denominator.unsqueeze(1)
    positive_counts = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(
        log_probs.masked_fill(~positives, 0).sum(dim=1) /
        positive_counts
    )
    return per_anchor[valid].mean()


def within_family_pair_counts(labels, families, prompt_ids):
    """Return eligible anchor count and same-family negative unordered pairs."""
    if prompt_ids is None or labels.shape != prompt_ids.shape or labels.shape != families.shape:
        raise ValueError("Expected aligned labels, families and prompt_ids")
    same_model = labels[:, None].eq(labels[None, :])
    same_family = families[:, None].eq(families[None, :])
    different_prompt = ~prompt_ids[:, None].eq(prompt_ids[None, :])
    positives = same_model & different_prompt
    negatives = same_family & ~same_model
    eligible = int((positives.any(1) & negatives.any(1)).sum().item())
    neg_pairs = int(torch.triu(negatives, diagonal=1).sum().item())
    return eligible, neg_pairs

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
        configured_weights = specs["family"].get("class_weights")
        if configured_weights is None:
            parts["family"] = F.cross_entropy(outputs["family_logits"], families)
        else:
            # Keep loss scale comparable across different family compositions.
            # torch CE(weight=..., reduction='mean') would divide by the sum
            # of in-batch weights, partly cancelling weights in skewed batches.
            weights = torch.as_tensor(
                configured_weights, device=outputs["family_logits"].device,
                dtype=torch.float32,
            )
            if (weights.ndim != 1 or weights.numel() != outputs["family_logits"].shape[-1]
                    or not torch.isfinite(weights).all() or not bool((weights > 0).all())):
                raise ValueError("Invalid provider class_weights")
            per_example = F.cross_entropy(
                outputs["family_logits"].float(), families, reduction="none"
            )
            parts["family"] = (per_example * weights[families]).mean()
    if specs["intra_contrastive"]["enabled"] and specs["intra_contrastive"]["weight"] > 0:
        spec = specs["intra_contrastive"]
        if spec.get("mode", "standard") == "within_family":
            parts["intra_contrastive"] = within_family_prompt_contrastive_loss(
                outputs["embeddings"], labels, families, prompt_ids,
                temperature=float(spec["temperature"]),
                same_prompt_weight=float(spec.get("same_prompt_weight", 3.0)),
                hard_pair_weights=spec.get("hard_pair_weights"),
            )
        elif spec.get("mode", "standard") == "prompt_aware":
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
