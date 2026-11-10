"""Qwen3 EOS + content-mean classifier; the full head is saved as `score`."""
import torch
from torch import nn
from transformers import Qwen3ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


def pool_content(hidden, input_ids, attention_mask, special_ids):
    mask = attention_mask.bool()
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("Cannot pool an empty sequence")
    eos = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    content = mask.clone()
    for token_id in special_ids:
        content &= input_ids.ne(token_id)
    # Accumulate the mean in float32 for half-precision inference.
    count = content.sum(dim=1, keepdim=True)
    mean = (hidden.float() * content.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
    mean = torch.where(count > 0, mean, eos.float()).to(hidden.dtype)
    return eos, mean


class EOSMeanHead(nn.Module):
    def __init__(self, hidden, width, labels, dropout):
        super().__init__()
        self.eos_norm = nn.LayerNorm(hidden)
        self.mean_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width, labels))

    def forward(self, eos, mean):
        return self.mlp(torch.cat([self.eos_norm(eos), self.mean_norm(mean)], dim=-1))


class Qwen3EOSMeanClassifier(Qwen3ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = EOSMeanHead(config.hidden_size, config.pooling_head_dim,
                                 config.num_labels, config.pooling_dropout)
        self.score.apply(self._init_weights)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        if input_ids is None or attention_mask is None:
            raise ValueError("EOS-mean pooling requires input_ids and attention_mask")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             inputs_embeds=inputs_embeds, return_dict=True,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, **kwargs)
        eos, mean = pool_content(outputs.last_hidden_state, input_ids, attention_mask,
                                 self.config.pooling_special_ids)
        logits = self.score(eos, mean)
        loss = None if labels is None else nn.functional.cross_entropy(logits.float(), labels)
        result = SequenceClassifierOutputWithPast(loss=loss, logits=logits,
                  past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
                  attentions=outputs.attentions)
        return result.to_tuple() if return_dict is False else result
