"""Constrained, opt-in HotFlip gradient search on single-token discourse markers."""
import torch
import torch.nn.functional as F

# Style-sensitive edits. A replacement can genuinely change authorship cues: this is an ablation.
CANDIDATES = [' however', ' therefore', ' additionally', ' moreover', ' thus',
              ' instead', ' also', ' yet', ' hence', ' consequently']


def marker_token_ids(tokenizer):
    """Only return safe-for-tokenizer marker IDs that each decode to one standalone token."""
    result = []
    special = set(tokenizer.all_special_ids)
    for word in CANDIDATES:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in special:
            result.append(ids[0])
    return sorted(set(result))


def hotflip_batch(model, batch, max_examples=2):
    """Create single-token loss-maximizing edits; never modify original batch in place.

    Gradient wrt embedding vectors approximates the change in NLL for candidate
    token substitutions. This is a heuristic; the true generator label for the
    edited text is *not* guaranteed to remain unchanged.
    """
    ids = batch['input_ids']
    replacement_ids = marker_token_ids(model._fhf_tokenizer)
    if len(replacement_ids) < 2 or max_examples <= 0:
        return None
    layer = model.encoder.get_input_embeddings()
    selected = min(max_examples, len(ids))
    captured = []
    def capture(_module, _inp, output):
        output.retain_grad()
        captured.append(output)
    hook = layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids[:selected], attention_mask=batch['attention_mask'][:selected])
        loss = F.nll_loss(out['log_probs'], batch['labels'][:selected])
        grads = torch.autograd.grad(loss, captured[-1], retain_graph=False)[0].detach()
    finally:
        hook.remove()
    token_embeds = layer.weight.detach()
    modified = ids.clone()
    legal = torch.tensor(replacement_ids, device=ids.device, dtype=torch.long)
    for b in range(selected):
        valid_positions = torch.where(torch.isin(ids[b], legal) & (batch['attention_mask'][b] != 0))[0]
        best = None
        for position in valid_positions.tolist():
            old = ids[b, position]
            # First-order objective: grad . (new_embedding - old_embedding)
            values = (token_embeds[legal] - token_embeds[old]) @ grads[b, position]
            values = values.masked_fill(legal == old, float('-inf'))
            score, index = values.max(dim=0)
            if torch.isfinite(score) and (best is None or score.item() > best[0]):
                best = (score.item(), position, legal[index].item())
        if best is not None:
            modified[b, best[1]] = best[2]
    return modified if not torch.equal(modified, ids) else None
