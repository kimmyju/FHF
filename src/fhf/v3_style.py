"""Versioned, bounded raw-text style features and residual EOS classifier."""
import math
import re
import torch
from torch import nn
from transformers import Qwen3ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

FEATURE_VERSION = 1
FEATURE_NAMES = ('log_chars', 'log_words', 'log_sentences', 'mean_word_length',
                 'mean_sentence_words', 'std_sentence_words', 'newline_rate',
                 'blank_line_rate', 'upper_rate', 'digit_rate', 'space_rate',
                 'comma_rate', 'colon_rate', 'semicolon_rate', 'dash_rate',
                 'paren_rate', 'quote_rate', 'question_rate', 'exclamation_rate',
                 'period_rate', 'list_line_rate', 'heading_line_rate',
                 'backtick_rate', 'asterisk_rate', 'unique_word_ratio')


def style_features(text):
    text = str(text)
    n = max(1, len(text))
    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences] or [0]
    mean = sum(lengths) / len(lengths)
    std = (sum((v - mean) ** 2 for v in lengths) / len(lengths)) ** .5
    lines = text.splitlines() or ['']
    def logscale(x, scale):
        return min(1., math.log1p(x) / math.log1p(scale))
    def rate(chars):
        return sum(text.count(c) for c in chars) / n
    values = [logscale(len(text), 4096), logscale(len(words), 1024),
              logscale(len(sentences), 128),
              min(1., sum(map(len, words)) / max(1, len(words)) / 20),
              logscale(mean, 256), logscale(std, 256),
              text.count('\n') / n, sum(not x.strip() for x in lines) / len(lines),
              sum(c.isupper() for c in text) / n, sum(c.isdigit() for c in text) / n,
              sum(c.isspace() for c in text) / n,
              *[rate(c) for c in [',', ':', ';', '-–—', '()[]{}', '\'"“”‘’', '?', '!', '.']],
              sum(bool(re.match(r'^\s*(?:[-*+]\s|\d+[.)]\s)', x)) for x in lines) / len(lines),
              sum(bool(re.match(r'^\s*#{1,6}\s', x)) for x in lines) / len(lines),
              rate('`'), rate('*'), len(set(w.casefold() for w in words)) / max(1, len(words))]
    return values


class EOSStyleHead(nn.Module):
    def __init__(self, hidden, width, labels, dropout):
        super().__init__()
        self.eos = nn.Linear(hidden, labels, bias=False)
        self.style = nn.Sequential(nn.Linear(len(FEATURE_NAMES), width), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(width, labels))

    def zero_residual(self):
        nn.init.zeros_(self.style[-1].weight)
        nn.init.zeros_(self.style[-1].bias)

    def forward(self, eos, features):
        return self.eos(eos) + self.style(features.to(dtype=eos.dtype))


class Qwen3EOSStyleClassifier(Qwen3ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = EOSStyleHead(config.hidden_size, config.style_hidden_dim,
                                  config.num_labels, config.style_dropout)
        self.score.apply(self._init_weights)
        self.score.zero_residual()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                style_features=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        if input_ids is None or attention_mask is None or style_features is None:
            raise ValueError('EOS-style requires token IDs, attention mask and style features')
        if style_features.shape != (input_ids.shape[0], len(FEATURE_NAMES)):
            raise ValueError('Wrong style feature shape')
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             inputs_embeds=inputs_embeds, return_dict=True,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, **kwargs)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if (last < 0).any():
            raise ValueError('Empty sequence')
        eos = outputs.last_hidden_state[torch.arange(len(last), device=last.device), last]
        logits = self.score(eos, style_features)
        loss = None if labels is None else nn.functional.cross_entropy(logits.float(), labels)
        result = SequenceClassifierOutputWithPast(loss=loss, logits=logits,
                  past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states,
                  attentions=outputs.attentions)
        return result.to_tuple() if return_dict is False else result
