"""ModernBERT-compatible, LoRA-adapted shared encoder with independently switchable heads.

LoRA fine-tunes attention Wqkv/Wo matrices while FHF classification/projection
heads remain fully trainable. Adapter-only checkpoints need a separately mounted
frozen base encoder for offline Kaggle inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .labels import MODEL_NAMES, FAMILY_NAMES, MODEL_FAMILY_IDS
from .style import StyleFeatureExtractor, MaskedAttentionPool, GatedStyleFusion


DEFAULT_LORA = {
    "enabled": True,
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["Wqkv", "Wo"],
    "bias": "none",
    "initial_adapter": None,
    "bundle_base_encoder": False,
}


def _lora_config(settings):
    """Create an explicit PEFT feature-extraction adapter for a bidirectional encoder."""
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(settings["r"]),
        lora_alpha=int(settings["alpha"]),
        lora_dropout=float(settings["dropout"]),
        target_modules=list(settings["target_modules"]),
        bias=settings.get("bias", "none"),
        inference_mode=False,
    )


def apply_lora(base, settings):
    """Wrap a pretrained encoder, optionally initializing from a pretrained adapter."""
    from peft import PeftModel, get_peft_model
    prior = settings.get("initial_adapter")
    if prior:
        return PeftModel.from_pretrained(base, prior, is_trainable=True)
    return get_peft_model(base, _lora_config(settings))


class FingerprintModel(nn.Module):
    """Hierarchical (provider/version), flat exact-model, or provider-only classifier."""

    def __init__(self, backbone, dropout=0.1, projection_dim=128,
                 heads=None, lora=None, local_files_only=False, attach_adapter=True,
                 pooling="mean", style=None, residual_fusion=None):
        super().__init__()
        self.heads = dict(heads or {"family": True, "model": True, "projection": True})
        if not (self.heads.get("family") or self.heads.get("model")):
            raise ValueError("At least one classification head must be enabled")
        self.lora = dict(DEFAULT_LORA if lora is None else {**DEFAULT_LORA, **lora})
        self.backbone_name = str(backbone)
        base = AutoModel.from_pretrained(backbone, local_files_only=local_files_only)
        hidden = base.config.hidden_size
        self.encoder = apply_lora(base, self.lora) if self.lora["enabled"] and attach_adapter else base
        self.pooling = str(pooling)
        if self.pooling not in ("mean", "attention"):
            raise ValueError("pooling must be mean or attention")
        self.attention_pool = MaskedAttentionPool(hidden) if self.pooling == "attention" else None
        self.style_config = dict(style or {"enabled": False})
        self.style_enabled = bool(self.style_config.get("enabled", False))
        self.style_fusion = None
        if self.style_enabled:
            style_dim = StyleFeatureExtractor(self.style_config.get("hash_dim", 256)).dim
            self.style_fusion = GatedStyleFusion(
                style_dim, hidden, int(self.style_config.get("hidden_dim", 128)),
                dropout=float(dropout),
            )
        self.residual_config = dict(residual_fusion or {"enabled": False})
        self.residual_enabled = bool(self.residual_config.get("enabled", False))
        self.flat_weight = float(self.residual_config.get("flat_weight", 0.5))
        if not 0 <= self.flat_weight <= 1:
            raise ValueError("flat_weight must lie in [0,1]")
        if self.residual_enabled and not (self.heads.get("model") and self.heads.get("family")):
            raise ValueError("Residual fusion needs both family and model heads")
        self.dropout = nn.Dropout(dropout)
        self.projection = (nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                        nn.Linear(hidden, projection_dim))
                           if self.heads.get("projection") else None)
        self.family_head = nn.Linear(hidden, len(FAMILY_NAMES)) if self.heads.get("family") else None
        self.model_head = nn.Linear(hidden, len(MODEL_NAMES)) if self.heads.get("model") else None
        self.flat_head = nn.Linear(hidden, len(MODEL_NAMES)) if self.residual_enabled else None
        self.register_buffer("family_lookup", torch.tensor(MODEL_FAMILY_IDS, dtype=torch.long))

    def forward(self, input_ids, attention_mask, style_features=None, **kwargs):
        """Encode text with optional style features and combine flat/hierarchical heads."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if self.attention_pool is not None:
            pooled = self.attention_pool(outputs.last_hidden_state, attention_mask)
        else:
            weights = attention_mask.unsqueeze(-1)
            pooled = (outputs.last_hidden_state * weights).sum(1) / weights.sum(1).clamp_min(1)
        result = {}
        if self.style_enabled:
            if style_features is None:
                raise ValueError("Style-enabled model requires raw-text style_features")
            pooled, gate_mean = self.style_fusion(pooled, style_features)
            result["style_gate_mean"] = gate_mean
        pooled = self.dropout(pooled)
        if self.projection is not None:
            result["embeddings"] = F.normalize(self.projection(pooled), dim=-1)
        if self.family_head is not None:
            result["family_logits"] = self.family_head(pooled)
            result["family_log_probs"] = F.log_softmax(result["family_logits"], dim=-1)
        if self.model_head is not None:
            logits = self.model_head(pooled)
            # Raw 18-model logits enable teacher-forced (oracle) within-family
            # evaluation independent of the Family Head's own prediction.
            result["model_logits"] = logits
            if self.family_head is None:
                result["log_probs"] = F.log_softmax(logits, dim=-1)
            else:
                local = torch.empty_like(logits)
                for family_idx in range(len(FAMILY_NAMES)):
                    indices = torch.where(self.family_lookup == family_idx)[0]
                    # autocast runs log_softmax in float32; index assignment does not cast.
                    family_log_probs = F.log_softmax(logits[:, indices], dim=-1)
                    local[:, indices] = family_log_probs.to(dtype=local.dtype)
                result["log_probs"] = local + result["family_log_probs"][:, self.family_lookup]
            if self.flat_head is not None:
                flat_log = F.log_softmax(self.flat_head(pooled), dim=-1)
                result["flat_log_probs"] = flat_log
                result["hierarchical_log_probs"] = result["log_probs"]
                # Mixture is computed in log space for numerical stability.
                if self.flat_weight == 0.0:
                    result["log_probs"] = result["hierarchical_log_probs"]
                elif self.flat_weight == 1.0:
                    result["log_probs"] = flat_log
                else:
                    result["log_probs"] = torch.logaddexp(
                        result["hierarchical_log_probs"].float() + torch.log(
                            flat_log.new_tensor(1.0 - self.flat_weight, dtype=torch.float32)
                        ),
                        flat_log.float() + torch.log(
                            flat_log.new_tensor(self.flat_weight, dtype=torch.float32)
                        ),
                    )
        return result

    def trainable_parameters(self):
        """Expose trainable adapter plus classifier/head parameter counts."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path, tokenizer, extra=None):
        """Save LoRA adapter, custom heads, tokenizer, and optional frozen base weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.lora["enabled"]:
            self.encoder.save_pretrained(path / "adapter", safe_serialization=True)
            if self.lora.get("bundle_base_encoder", False):
                self.encoder.get_base_model().save_pretrained(
                    path / "base_encoder", safe_serialization=True)
        else:
            self.encoder.save_pretrained(path / "encoder", safe_serialization=True)
        tokenizer.save_pretrained(path / "tokenizer")
        torch.save({
            "heads": self.heads,
            "projection": self.projection.state_dict() if self.projection is not None else None,
            "family_head": self.family_head.state_dict() if self.family_head is not None else None,
            "model_head": self.model_head.state_dict() if self.model_head is not None else None,
            "flat_head": self.flat_head.state_dict() if self.flat_head is not None else None,
            "attention_pool": (self.attention_pool.state_dict() if self.attention_pool is not None else None),
            "style_fusion": self.style_fusion.state_dict() if self.style_fusion is not None else None,
            "dropout": self.dropout.p,
            "projection_dim": self.projection[-1].out_features if self.projection is not None else None,
        }, path / "heads.pt")
        (path / "metadata.json").write_text(json.dumps({
            "format_version": 2, "models": MODEL_NAMES, "families": FAMILY_NAMES,
            "base_encoder": self.backbone_name, "lora": self.lora,
            "heads": self.heads, "pooling": self.pooling,
            "style": self.style_config, "residual_fusion": self.residual_config,
            "extra": extra or {},
        }, indent=2), encoding="utf-8")

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", base_encoder_override=None):
        """Restore without network access, loading LoRA on the exact matching frozen base.

        Adapter-only checkpoints require a predownloaded base encoder at
        base_encoder_override or a cached base encoder. For fully offline Kaggle
        runs set paths.inference_base_encoder to an attached model location.
        """
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta["models"] != MODEL_NAMES or meta["families"] != FAMILY_NAMES:
            raise ValueError("Checkpoint labels do not match the competition's 18 classes")
        state = torch.load(path / "heads.pt", map_location=map_location, weights_only=True)
        lora = meta.get("lora", {"enabled": False})
        if lora.get("enabled", False):
            base = base_encoder_override or (
                str(path / "base_encoder") if (path / "base_encoder").is_dir()
                else meta["base_encoder"]
            )
        else:
            base = str(path / "encoder")
        try:
            model = cls(base, dropout=state["dropout"],
                        projection_dim=state["projection_dim"] or 128,
                        heads=state["heads"], lora=lora,
                        local_files_only=True, attach_adapter=False,
                        pooling=meta.get("pooling", "mean"),
                        style=meta.get("style", {"enabled": False}),
                        residual_fusion=meta.get("residual_fusion", {"enabled": False}))
        except (OSError, FileNotFoundError) as error:
            raise FileNotFoundError(
                "Cannot load frozen base encoder offline. Set "
                "paths.inference_base_encoder to the attached ModernBERT-large "
                "directory (or enable model.lora.bundle_base_encoder during training)."
            ) from error
        if lora.get("enabled", False):
            from peft import PeftModel
            model.encoder = PeftModel.from_pretrained(
                model.encoder, str(path / "adapter"), is_trainable=False)
        for name in ("projection", "family_head", "model_head",
                     "flat_head", "attention_pool", "style_fusion"):
            module = getattr(model, name)
            if module is not None:
                module.load_state_dict(state[name])
        tokenizer = AutoTokenizer.from_pretrained(
            str(path / "tokenizer"), local_files_only=True)
        return model, tokenizer, meta
