"""Load YAML experiments, merge inherited presets, validate head/loss switches and resolve I/O paths."""
from __future__ import annotations

import copy
from pathlib import Path
import yaml

HEADS = ("family", "model", "projection")
LOSSES = ("exact_classification", "family", "intra_contrastive", "rdrop", "hotflip")


def _merge(base, extra):
    """Recursively override nested mappings without mutating the preset."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(filename, seen=None):
    """Resolve 'extends' relative to the child YAML file, detecting cycles."""
    path = Path(filename).expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular YAML inheritance: {path}")
    seen.add(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    contents = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(contents, dict):
        raise ValueError("YAML root must be a mapping")
    parent = contents.pop("extends", None)
    if parent:
        merged = _merge(_read_yaml(path.parent / parent, seen), contents)
    else:
        merged = contents
    seen.remove(path)
    return merged


def _override(conf, pair):
    """Parse --set a.b.c=value using YAML scalars, lists and booleans."""
    if "=" not in pair:
        raise ValueError(f"Expected dotted.key=value, got {pair!r}")
    path, raw = pair.split("=", 1)
    keys = path.split(".")
    target = conf
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise KeyError(f"Unknown configuration path: {path}")
        target = target[key]
    if keys[-1] not in target:
        raise KeyError(f"Unknown configuration key: {path}")
    target[keys[-1]] = yaml.safe_load(raw)


def load_config(path, overrides=()):
    """Load and validate an experiment YAML; no I/O file location is hard-coded."""
    conf = _read_yaml(path)
    for pair in overrides:
        _override(conf, pair)
    validate_config(conf)
    return conf


def is_enabled(config, name):
    """Read a loss switch, treating explicit 'enabled' as the sole activation flag."""
    return bool(config["losses"][name]["enabled"])


def validate_config(conf):
    """Prevent incompatible heads, loss targets and undefined experiment objectives."""
    for section in ("paths", "model", "losses", "training", "inference"):
        if not isinstance(conf.get(section), dict):
            raise ValueError(f"Missing YAML section: {section}")
    heads = conf["model"].get("heads", {})
    if set(heads) != set(HEADS) or any(type(heads[h]) is not bool for h in HEADS):
        raise ValueError("Set model.heads.family, model, projection explicitly to true/false")
    if not (heads["family"] or heads["model"]):
        raise ValueError("At least one classifier head (family or model) must be enabled")
    for loss in LOSSES:
        spec = conf["losses"].get(loss)
        if not isinstance(spec, dict) or type(spec.get("enabled")) is not bool:
            raise ValueError(f"losses.{loss}.enabled must be true/false")
        if float(spec.get("weight", -1)) < 0:
            raise ValueError(f"losses.{loss}.weight must be non-negative")
    active = lambda name: is_enabled(conf, name) and float(conf["losses"][name]["weight"]) > 0
    if active("exact_classification") and not heads["model"]:
        raise ValueError("Exact classification loss requires model head")
    if active("family") and not heads["family"]:
        raise ValueError("Family loss requires family head")
    if active("intra_contrastive") and not heads["projection"]:
        raise ValueError("Contrastive loss requires projection head")
    if active("hotflip") and not heads["model"]:
        raise ValueError("HotFlip training currently requires the exact-model head")
    lora = conf["model"].get("lora", {"enabled": False})
    if type(lora.get("enabled", False)) is not bool:
        raise ValueError("model.lora.enabled must be true/false")
    if lora["enabled"]:
        if int(lora["r"]) <= 0 or int(lora["alpha"]) <= 0:
            raise ValueError("LoRA r and alpha must be positive")
        if not 0 <= float(lora["dropout"]) < 1:
            raise ValueError("LoRA dropout must be in [0,1)")
        if not isinstance(lora.get("target_modules"), list) or not lora["target_modules"]:
            raise ValueError("LoRA target_modules must be a nonempty list")
    if int(conf["training"].get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if conf["training"].get("mixed_precision", "none") not in ("auto", "bf16", "fp16", "none"):
        raise ValueError("mixed_precision must be auto, bf16, fp16, or none")
    if not any(active(name) for name in ("exact_classification", "family")):
        raise ValueError("At least one classification loss must be active for supervised training")
    for key in ("encoder", "max_length", "dropout", "projection_dim"):
        if key not in conf["model"]:
            raise ValueError(f"Missing model.{key}")
    if int(conf["model"]["max_length"]) < 8 or int(conf["model"]["projection_dim"]) < 1:
        raise ValueError("Invalid model sequence length or projection dimension")
    if not (0 <= float(conf["model"]["dropout"]) < 1):
        raise ValueError("Dropout must be in [0,1)")
    if int(conf["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be >= 1")
    early = conf["training"].get("early_stopping", {})
    if type(early.get("enabled", False)) is not bool:
        raise ValueError("training.early_stopping.enabled must be true/false")
    if int(early.get("patience", 4)) < 1 or float(early.get("min_delta", 0.0)) < 0:
        raise ValueError("Early stopping patience must be >=1 and min_delta >=0")
    scheduler = conf["training"].get("scheduler", {})
    if scheduler.get("name", "none") not in ("none", "cosine"):
        raise ValueError("training.scheduler.name must be none or cosine")
    if not 0 <= float(scheduler.get("warmup_ratio", 0.0)) < 1:
        raise ValueError("Scheduler warmup_ratio must be in [0,1)")
    if not 0 <= float(scheduler.get("min_lr_ratio", 0.0)) <= 1:
        raise ValueError("Scheduler min_lr_ratio must be in [0,1]")
    if type(conf["training"].get("log_epoch_metrics", True)) is not bool:
        raise ValueError("training.log_epoch_metrics must be true/false")


def resolve_path(conf, key):
    """Paths may be absolute or relative to paths.base_dir (normally project cwd)."""
    raw = conf["paths"].get(key)
    if raw is None or raw == "":
        raise ValueError(f"paths.{key} must be configured")
    base = Path(conf["paths"].get("base_dir", ".")).expanduser()
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else base / path
