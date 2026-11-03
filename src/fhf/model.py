"""Configurable shared Transformer encoder with independently switchable provider, model and projection heads."""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .labels import MODEL_NAMES, FAMILY_NAMES, MODEL_FAMILY_IDS


class FingerprintModel(nn.Module):
    """Support family-only, flat model-only and hierarchical attribution.

    When both classifier heads are enabled:
      log P(model|x) = log P(provider|x) + log P(model|provider,x)
    When only the model head is enabled, predict with a flat 18-way softmax.
    Family-only is diagnostic and cannot make an 18-class Kaggle submission.
    """

    def __init__(self, backbone, dropout=0.1, projection_dim=128,
                 heads=None, local_files_only=False):
        super().__init__()
        self.heads = dict(heads or {"family": True, "model": True, "projection": True})
        if not (self.heads.get("family") or self.heads.get("model")):
            raise ValueError("At least one classification head must be enabled")
        self.backbone_name = backbone
        self.encoder = AutoModel.from_pretrained(backbone, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.projection = (nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                        nn.Linear(hidden, projection_dim))
                           if self.heads.get("projection") else None)
        self.family_head = nn.Linear(hidden, len(FAMILY_NAMES)) if self.heads.get("family") else None
        self.model_head = nn.Linear(hidden, len(MODEL_NAMES)) if self.heads.get("model") else None
        self.register_buffer("family_lookup", torch.tensor(MODEL_FAMILY_IDS, dtype=torch.long))

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        weights = attention_mask.unsqueeze(-1)
        pooled = (outputs.last_hidden_state * weights).sum(1) / weights.sum(1).clamp_min(1)
        pooled = self.dropout(pooled)
        result = {}
        if self.projection is not None:
            result["embeddings"] = F.normalize(self.projection(pooled), dim=-1)
        if self.family_head is not None:
            result["family_logits"] = self.family_head(pooled)
            result["family_log_probs"] = F.log_softmax(result["family_logits"], dim=-1)
        if self.model_head is not None:
            logits = self.model_head(pooled)
            if self.family_head is None:
                result["log_probs"] = F.log_softmax(logits, dim=-1)
            else:
                local = torch.empty_like(logits)
                for family_idx in range(len(FAMILY_NAMES)):
                    ids = torch.where(self.family_lookup == family_idx)[0]
                    local[:, ids] = F.log_softmax(logits[:, ids], dim=-1)
                result["log_probs"] = local + result["family_log_probs"][:, self.family_lookup]
        return result

    def save_checkpoint(self, path, tokenizer, extra=None):
        """Save model, optional heads, model configuration and tokenizer for offline inference."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder", safe_serialization=True)
        tokenizer.save_pretrained(path / "tokenizer")
        torch.save({
            "heads": self.heads,
            "projection": self.projection.state_dict() if self.projection is not None else None,
            "family_head": self.family_head.state_dict() if self.family_head is not None else None,
            "model_head": self.model_head.state_dict() if self.model_head is not None else None,
            "dropout": self.dropout.p,
            "projection_dim": self.projection[-1].out_features if self.projection is not None else None,
        }, path / "heads.pt")
        (path / "metadata.json").write_text(json.dumps({
            "models": MODEL_NAMES, "families": FAMILY_NAMES,
            "heads": self.heads, "extra": extra or {},
        }, indent=2), encoding="utf-8")

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        """Restore the exact head architecture, including all disabled modules."""
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta["models"] != MODEL_NAMES or meta["families"] != FAMILY_NAMES:
            raise ValueError("Checkpoint label vocabulary differs from current competition labels")
        state = torch.load(path / "heads.pt", map_location=map_location, weights_only=True)
        model = cls(str(path / "encoder"), dropout=state["dropout"],
                    projection_dim=state["projection_dim"] or 128,
                    heads=state["heads"], local_files_only=True)
        for name in ("projection", "family_head", "model_head"):
            module = getattr(model, name)
            if module is not None:
                module.load_state_dict(state[name])
        tokenizer = AutoTokenizer.from_pretrained(str(path / "tokenizer"), local_files_only=True)
        return model, tokenizer, meta
