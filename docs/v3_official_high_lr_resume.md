# Official LR control from the saved external checkpoint

Restore the last external checkpoint from the completed low-LR run, then train only the official stage for 4 epochs at LoRA LR 1e-4 and classifier LR 3e-4. External data is not loaded or trained again. No style branch is used.

## Run

```bash
cd /home/yeju/repos/FHF
git switch experiments/fingerprint-v3-decoder
git pull --ff-only origin experiments/fingerprint-v3-decoder
CUDA_VISIBLE_DEVICES=2 bash scripts/run_v3_official_high_lr_resume.sh
```

Settings are in `configs/fingerprint_v3/qwen3_8b_official_high_lr_resume.json` (JSON, not YAML). W&B is enabled by the launcher.

Default NAS root: `/media/NAS/USERS/yeju/FHF` (`FHF_ROOT` override supported).

| Item | Location relative to NAS root |
|---|---|
| Source results (preserved) | `results/v3-qwen8b-eos-ext2-official4-low-lr` |
| Source checkpoint (read only) | `checkpoints/v3-qwen8b-eos-ext2-official4-low-lr/fold-0/epochs/external-2/adapter` |
| New results | `results/v3-qwen8b-eos-ext2-official4-high-lr-resume` |
| New checkpoints | `checkpoints/v3-qwen8b-eos-ext2-official4-high-lr-resume` |

The actual source fold location is resolved from its results `manifest.json`. The source final/best `adapter/` (60.19% official epoch 2) is NOT used as the starting point and is never overwritten. A source folder must include the saved external epoch adapter and `validation_probabilities.npy`.

For a custom source path or repeat run:

```bash
FHF_SOURCE_BUNDLE=/absolute/path/to/low-lr-results \
FHF_RUN_NAME=v3-qwen8b-eos-high-lr-resume-r2 \
CUDA_VISIBLE_DEVICES=2 bash scripts/run_v3_official_high_lr_resume.sh
```

Official train and valid remain `csv/splits/dual-path-prompt-aware/train_split.csv` and `val.csv`. No `--external` argument is allowed when resuming.

## Restoration and training

- Check canonical labels, one holdout fold, training IDs/prompts/labels and validation content/order hash against the source.
- Inherit base model/revision, architecture, tokenization, batch size, accumulation and other training settings from source. New config overrides official LR, official classifier LR, epochs, seed, selection and W&B options. Source checkpoint metadata is authoritative for model/batch settings.
- Restore existing LoRA and classifier with `is_trainable=True`; do not initialize another adapter.
- Evaluate restored external checkpoint using original validation batch size/order. Save `fold-0/resume_audit.json` with prediction agreement, source checkpoint SHA256 and BAcc. Stop before training if argmax or probability comparison fails.
- Save a copy of the starting external adapter into the NEW run. Create fresh AdamW/cosine schedule for the official stage, matching the original pipeline's stage transition design. No optimizer state is resumed.
- Evaluate and save every official epoch. Select the highest neural validation BAcc among starting external checkpoint and official epochs 1–4; ties retain the earlier model.
- Save selected adapter at the new `fold-0/adapter/`. Standard v3 offline prediction uses this new bundle independently of the old source checkpoint path.

Existing output directories are refused. Outputs inside source results/fold directories are refused. Source data and checkpoint files are read only. No deletion, overwrite or move of the previous experiment is performed.

## Comparison limitation

The previous run saved weights but not RNG state at the external boundary. This run resets the seed before official training. Thus both runs share identical starting weights and settings except official LR, but random sample order/dropout streams need not match the original low-LR run. This is not a bitwise paired continuation. For strict LR isolation, rerun the low-LR official stage from the same source through this resume path with the same seed and `official_learning_rate=2e-5`.

## Results to inspect

```bash
cat /media/NAS/USERS/yeju/FHF/results/v3-qwen8b-eos-ext2-official4-high-lr-resume/metrics.json
```

Compare `neural`, `selection`, and `final_epoch_neural` against the low-LR 60.19% result. This is validation-selected performance, not an unbiased test estimate. New run could be lower than 60.19%; the old best remains available. Epoch metrics and adapters are under the NEW checkpoint fold directory. `history.json` starts with `restored_external`, followed by four official epochs.
