# Frozen EOS + style-only residual

This experiment restores the original EOS run, including its trained LoRA and saved `score` classifier. It does not restart Qwen training or load the EOS-mean/EOS-style experiment. All EOS parameters are frozen and the model stays in eval mode. Only raw-text style MLP parameters are optimized, using official training labels only.

## Run

```bash
cd /home/yeju/repos/FHF
git switch experiments/fingerprint-v3-decoder
git pull --ff-only origin experiments/fingerprint-v3-decoder
CUDA_VISIBLE_DEVICES=2 bash scripts/run_v3_frozen_style.sh
```

Defaults:
- Source results: `/media/NAS/USERS/yeju/FHF/results/v3-qwen8b-ext2-official4`
- Train: `/media/NAS/USERS/yeju/FHF/csv/splits/dual-path-prompt-aware/train_split.csv`
- Validation: same directory, `val.csv`
- Source adapter/tokenizer/config are resolved through the original `manifest.json` (usually NAS `checkpoints/<run>/fold-0`). Keep that directory available.
- New output: `/media/NAS/USERS/yeju/FHF/results/v3-qwen8b-frozen-eos-style`
- All settings: `configs/fingerprint_v3/qwen3_8b_frozen_style.json`. No YAML required.

For a different original results path or a repeat experiment:

```bash
FHF_SOURCE_BUNDLE=/absolute/path/to/original-eos-results \
FHF_RUN_NAME=v3-qwen8b-frozen-eos-style-r2 \
CUDA_VISIBLE_DEVICES=2 bash scripts/run_v3_frozen_style.sh
```

Optional `FHF_BASE_MODEL=/absolute/path/to/local/Qwen3-8B` uses local base weights. They must match the original run's base revision. Otherwise the model ID and revision in the source checkpoint config are used. The new training output is refused if it already exists.

## What is trained

1. Check the original run is one frozen-holdout EOS fold with the canonical 18-label order. Match training IDs/prompts/labels and the original full validation CSV hash.
2. Copy the small adapter, tokenizer and config into the new bundle's `source/`; original files are not overwritten. Base model weights are not copied.
3. Load EOS with `is_trainable=False`, disable gradients on every parameter, and cache float32 train/validation logits in one model load. Train and validation have separate batch boundaries, using the source checkpoint's original batch size to reproduce its evaluation padding shapes. No EOS backward pass or external training occurs.
4. Compare restored validation predictions with the original `neural::*` probabilities. Stop on any changed argmax or a probability difference outside tolerance. `restore_audit.json` records the comparison, original/restored BAcc, batch size and runtime versions. `restore_mismatches.csv` lists changed predictions and top-two probability margins. The parity threshold is not relaxed. Investigate a failure rather than bypassing it.
5. Train a CPU MLP (25 versioned style features → 128 → GELU/dropout → 18) against `EOS_logits + style_logits`. Its last layer starts at zero, so initial combined logits exactly equal restored EOS logits. EOS logits are detached constants; only MLP parameters enter AdamW.
6. Run 20 **small-MLP epochs**, not 20 Qwen epochs. Evaluate alpha `[0, 0.25, 0.5, 1]` each epoch. Retain a candidate only on strictly higher validation BAcc. Ties keep the earlier candidate; if nothing improves, epoch 0/alpha 0 keeps the baseline exactly.

The feature extractor and input scaling are fixed, not fitted on validation. Validation selects epoch and alpha, so report the result as **validation-selected**, not an unbiased held-out test estimate. Baseline fallback protects this validation metric only; it cannot guarantee improvement on unseen test prompts or 70% accuracy.

## Outputs

- `metrics.json`: baseline and selected neural BAcc/family/within-family, plus selection.
- `restore_audit.json`: original vs restored EOS comparison.
- `history.json`: per-epoch/per-alpha validation metrics and train loss.
- `selection.json`, `style.pt`: selected epoch/alpha and style weights.
- `style-epoch-*.pt`: all MLP epoch weights, including non-selected ones.
- `frozen_logits.npz`: cached train/validation EOS logits in supplied CSV row order.
- `predictions.csv`: selected validation probabilities in canonical label order.
- `manifest.json`, `config.json`, `provenance.json`, `source/`: inference settings and copied original adapter.

## Offline prediction

Use this dedicated entry point for the new bundle (not the original v3 `predict` command):

```bash
PYTHONPATH=src python scripts/fingerprint_v3_frozen_style.py predict \
  --bundle /media/NAS/USERS/yeju/FHF/results/v3-qwen8b-frozen-eos-style \
  --base /absolute/path/to/local/Qwen3-8B \
  --test /absolute/path/to/test.csv \
  --out /absolute/path/to/submission.parquet --batch-size 2
```

Prediction restores the copied original EOS checkpoint and adds the saved CPU residual at the selected alpha. No lexical blend is used. The base weights must be available offline. This adds only a small MLP to the existing 8B inference footprint; quantized 16GB inference and competition runtime are not established by the CPU smoke tests.
