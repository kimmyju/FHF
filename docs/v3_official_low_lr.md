# EOS: lower official-stage LoRA LR and select validation epoch

Run a new Qwen3-8B EOS experiment from the base model: external 2 epochs, then official 4 epochs. This does **not** resume the existing 57.59% final model and does not train the style branch.

| Setting | External | Official |
|---|---:|---:|
| LoRA initial LR | 1e-4 | 2e-5 |
| Classifier initial LR | 3e-4 | 3e-4 |
| Epochs | 2 | 4 |
| Microbatch / accumulation | 2 / 16 | 2 / 16 |

Only official LoRA LR is lowered. Each stage starts a fresh AdamW optimizer and cosine schedule with warmup 0.1, as in the original pipeline. Original configs without stage overrides or `select_best_epoch` retain their behavior.

## Execute

```bash
cd /home/yeju/repos/FHF
git switch experiments/fingerprint-v3-decoder
git pull --ff-only origin experiments/fingerprint-v3-decoder
CUDA_VISIBLE_DEVICES=2 bash scripts/run_v3_official_low_lr.sh
```

The launcher sets `FHF_ROOT` to `/media/NAS/USERS/yeju/FHF` by default, validates input paths and enables W&B. Config: `configs/fingerprint_v3/qwen3_8b_official_low_lr.json`; no YAML is required. Run name defaults to `v3-qwen8b-eos-ext2-official4-low-lr`. Override `FHF_RUN_NAME` for another run; existing output directories are refused.

Inputs:
- `csv/splits/dual-path-prompt-aware/train_split.csv`
- `csv/splits/dual-path-prompt-aware/val.csv`
- `csv/external/full/selected_3495_per_model.csv`

Paths are relative to `FHF_ROOT`. All 18 classes and original split/overlap checks remain enforced.

## Selection and reporting

Evaluate validation after every external and official epoch and save each adapter. External epoch 1 is diagnostic only. Candidates are external epoch 2 (before official adaptation) and official epochs 1–4. Strictly higher **neural BAcc** wins; ties keep the earlier candidate. Every scheduled training epoch still runs. Validation contributes no gradient updates. Outer CV cannot enable selection; explicit `--valid` is required.

This is validation-based model selection, not an unbiased test estimate. The run may score below the original EOS: it is a new training experiment, and the old 57.59% checkpoint is not a candidate. Compare both `neural` (selected) and `final_epoch_neural` (official epoch 4) against the original score to distinguish checkpoint selection from the LR change. A subsequent high-LR control with the same selection policy would isolate LR effects more strictly.

Results: `$FHF_ROOT/results/v3-qwen8b-eos-ext2-official4-low-lr/metrics.json`
- `neural`: selected checkpoint metrics.
- `selection`: stage/epoch and adapter path relative to fold directory.
- `final_epoch_neural`: final official epoch metrics, regardless of selection.
- `evaluation`: `validation_selected_holdout`.

Checkpoints: `$FHF_ROOT/checkpoints/v3-qwen8b-eos-ext2-official4-low-lr/fold-0/`
- `adapter/`: selected weights, used by the standard v3 offline `predict` command.
- `epochs/external-1/adapter/`, `epochs/external-2/adapter/`, `epochs/official-1/adapter/` through `official-4`: adapters for every epoch (base weights are not duplicated).
- Each epoch directory includes `validation_probabilities.npy` in validation CSV row order and canonical label order.
- `epoch_metrics.json`: all validation metrics, updated during training.
- `history.json`: train loss/accuracy, initial stage LRs, validation BAcc/family/within-family.
- `selection.json`, `final_epoch_metrics.json`, tokenizer and config snapshots.

W&B logs include `external/val_balanced_accuracy`, `official/val_balanced_accuracy`, family and within-family metrics, and initial stage LRs. Unlike the frozen-style experiment, this run trains Qwen LoRA on GPU. Small-model CPU tests establish save/reload and selection correctness, not actual 8B performance or Kaggle memory/runtime compliance.
