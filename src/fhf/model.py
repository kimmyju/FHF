"""Shared Transformer encoder with hierarchical provider and conditional exact-model heads."""
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from .labels import MODEL_NAMES, FAMILY_NAMES, MODEL_FAMILY_IDS


class FingerprintModel(nn.Module):
    """Compute P(model|text) = P(family|text) * P(model|family,text)."""
    def __init__(self, backbone, dropout=0.1, projection_dim=128, local_files_only=False):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = AutoModel.from_pretrained(backbone, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection_dim))
        self.family_head = nn.Linear(hidden, len(FAMILY_NAMES))
        self.model_head = nn.Linear(hidden, len(MODEL_NAMES))
        self.register_buffer('family_lookup', torch.tensor(MODEL_FAMILY_IDS, dtype=torch.long))

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state
        weights = attention_mask.unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        pooled = self.dropout(pooled)
        z = nn.functional.normalize(self.projection(pooled), dim=-1)
        f_logit = self.family_head(pooled)
        conditional_logits = self.model_head(pooled)
        family_logprob = torch.log_softmax(f_logit, dim=-1)
        local_logprob = torch.empty_like(conditional_logits)
        for fi in range(len(FAMILY_NAMES)):
            ids = torch.where(self.family_lookup == fi)[0]
            local_logprob[:, ids] = torch.log_softmax(conditional_logits[:, ids], dim=-1)
        joint_logprob = local_logprob + family_logprob[:, self.family_lookup]
        return dict(log_probs=joint_logprob, family_logits=f_logit, embeddings=z)

    def save_checkpoint(self, path, tokenizer, extra=None):
        """Write weights and tokenizer to a relocatable offline Kaggle bundle."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path/'encoder', safe_serialization=True)
        tokenizer.save_pretrained(path/'tokenizer')
        torch.save({
            'projection': self.projection.state_dict(), 'family_head': self.family_head.state_dict(),
            'model_head': self.model_head.state_dict(), 'dropout': self.dropout.p,
            'projection_dim': self.projection[-1].out_features,
        }, path/'heads.pt')
        (path/'metadata.json').write_text(json.dumps({
            'models': MODEL_NAMES, 'families': FAMILY_NAMES, 'extra': extra or {}
        }, indent=2))

    @classmethod
    def from_checkpoint(cls, path, map_location='cpu'):
        """Load saved encoder and heads, verifying the expected competition vocabulary."""
        path = Path(path)
        meta = json.loads((path/'metadata.json').read_text())
        if meta['models'] != MODEL_NAMES or meta['families'] != FAMILY_NAMES:
            raise ValueError('Checkpoint label vocabulary differs from this code')
        state = torch.load(path/'heads.pt', map_location=map_location, weights_only=True)
        model = cls(str(path/'encoder'), dropout=state['dropout'],
                    projection_dim=state['projection_dim'], local_files_only=True)
        for name in ('projection', 'family_head', 'model_head'):
            getattr(model, name).load_state_dict(state[name])
        tokenizer = AutoTokenizer.from_pretrained(str(path/'tokenizer'), local_files_only=True)
        return model, tokenizer, meta
