"""Load YAML experiments, merge inherited presets, validate head/loss switches and resolve I/O paths."""
from __future__ import annotations

import copy
from pathlib import Path
import yaml

HEADS = ("family", "model", "projection")
LOSSES = ("exact_classification", "flat_classification", "family", "intra_contrastive", "rdrop", "hotflip")


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
    if active("flat_classification") and not (
            heads["family"] and heads["model"] and conf["model"].get("residual_fusion", {}).get("enabled", False)):
        raise ValueError("Flat classification loss requires residual fusion with both classifier heads")
    if active("family") and not heads["family"]:
        raise ValueError("Family loss requires family head")
    balance = conf["losses"]["family"].get("class_balance", {})
    if not isinstance(balance, dict) or type(balance.get("enabled", False)) is not bool:
        raise ValueError("losses.family.class_balance.enabled must be boolean")
    if balance.get("enabled", False):
        if not active("family") or not heads["family"]:
            raise ValueError("Family balance requires enabled family loss and head")
        power = balance.get("power", 1.0)
        if (isinstance(power, bool) or not isinstance(power, (int, float))
                or not 0 <= power <= 1):
            raise ValueError("Family balance power must lie within [0,1]")
    contrastive = conf["losses"]["intra_contrastive"]
    if contrastive.get("mode", "standard") not in ("standard", "prompt_aware", "within_family"):
        raise ValueError("Contrastive mode must be standard, prompt_aware or within_family")
    if not float(contrastive.get("temperature", 0.12)) > 0:
        raise ValueError("Contrastive temperature must be > 0")
    if float(contrastive.get("hard_weight", 2.0)) < 1 or float(
        contrastive.get("same_prompt_weight", 3.0)
    ) < 1:
        raise ValueError("Contrastive negative weights must be >= 1")
    if active("intra_contrastive") and not heads["projection"]:
        raise ValueError("Contrastive loss requires projection head")
    if active("intra_contrastive") and contrastive.get("mode") in ("prompt_aware", "within_family"):
        train = conf["training"]
        if int(train.get("samples_per_model", 2)) < 2:
            raise ValueError("Prompt-aware/within-family sampling needs samples_per_model >= 2")
        if int(train.get("models_per_family", 2)) < 1:
            raise ValueError("Prompt-aware sampling needs models_per_family >= 1")
    pooling = conf["model"].get("pooling", "mean")
    if pooling not in ("mean", "attention"):
        raise ValueError("model.pooling must be mean or attention")
    style = conf["model"].get("style", {})
    if type(style.get("enabled", False)) is not bool:
        raise ValueError("model.style.enabled must be true/false")
    if style.get("enabled", False):
        dim = int(style.get("hash_dim", 4096))
        if dim < 16 or dim & (dim - 1):
            raise ValueError("model.style.hash_dim must be a power of two >=16")
        if int(style.get("hidden_dim", 128)) < 1:
            raise ValueError("model.style.hidden_dim must be positive")
    perturb = conf.get("style_perturbation", {"enabled": False})
    if not isinstance(perturb, dict) or type(perturb.get("enabled", False)) is not bool:
        raise ValueError("style_perturbation.enabled must be boolean")
    if float(perturb.get("sigma", 0.0)) < 0 or float(perturb.get("weight", 0.0)) < 0:
        raise ValueError("style_perturbation sigma/weight must be nonnegative")
    if perturb.get("enabled") and (
        not style.get("enabled", False) or float(perturb.get("sigma", 0)) <= 0
        or float(perturb.get("weight", 0)) <= 0
    ):
        raise ValueError("Style perturbation requires enabled style, sigma > 0 and weight > 0")
    residual = conf["model"].get("residual_fusion", {})
    if type(residual.get("enabled", False)) is not bool:
        raise ValueError("model.residual_fusion.enabled must be true/false")
    if not 0 <= float(residual.get("flat_weight", 0.5)) <= 1:
        raise ValueError("model.residual_fusion.flat_weight must be in [0,1]")
    if residual.get("enabled", False) and not (heads["family"] and heads["model"]):
        raise ValueError("Residual fusion requires both family and model heads")
    if active("hotflip") and style.get("enabled", False):
        raise ValueError("Style-enabled HotFlip requires regenerating perturbed raw-text style features")
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
    head_lr = conf["training"].get("head_learning_rate")
    if head_lr is not None and float(head_lr) <= 0:
        raise ValueError("training.head_learning_rate must be positive")
    if type(conf["training"].get("frozen_split", False)) is not bool:
        raise ValueError("training.frozen_split must be boolean")
    if conf["training"].get("frozen_split"):
        for key in ("frozen_train_csv", "frozen_validation_csv"):
            if not conf["paths"].get(key):
                raise ValueError(f"Frozen split requires paths.{key}")

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
    ext = conf.get("external_data", {"enabled": False})
    if not isinstance(ext, dict) or type(ext.get("enabled", False)) is not bool:
        raise ValueError("external_data.enabled must be true/false")
    if ext.get("enabled", False):
        for key in (
            "external_source_path", "frozen_train_csv", "frozen_validation_csv",
            "external_clean_csv", "external_audit_json",
        ):
            if not conf["paths"].get(key):
                raise ValueError(f"External training requires paths.{key}")
        if not isinstance(ext.get("columns", {}), dict):
            raise ValueError("external_data.columns must be a mapping")
        if not isinstance(ext.get("label_map", {}), dict):
            raise ValueError("external_data.label_map must be a mapping")
        join = ext.get("join", {}) or {}
        if not isinstance(join, dict):
            raise ValueError("external_data.join must be a mapping")
        for key in ("min_chars", "min_words", "min_accepted_rows",
                    "min_multi_model_prompt_groups"):
            if int(ext.get(key, 0)) < 0:
                raise ValueError(f"external_data.{key} must be nonnegative")
        if int(ext.get("min_accepted_rows", 1)) < 1:
            raise ValueError("external_data.min_accepted_rows must be >=1")


def resolve_path(conf, key):
    """Paths may be absolute or relative to paths.base_dir (normally project cwd)."""
    raw = conf["paths"].get(key)
    if raw is None or raw == "":
        raise ValueError(f"paths.{key} must be configured")
    base = Path(conf["paths"].get("base_dir", ".")).expanduser()
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else base / path
