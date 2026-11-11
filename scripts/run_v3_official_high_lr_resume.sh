#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FHF_ROOT="${FHF_ROOT:-/media/NAS/USERS/yeju/FHF}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
train="$FHF_ROOT/csv/splits/dual-path-prompt-aware/train_split.csv"
valid="$FHF_ROOT/csv/splits/dual-path-prompt-aware/val.csv"
source_bundle="${FHF_SOURCE_BUNDLE:-$FHF_ROOT/results/v3-qwen8b-eos-ext2-official4-low-lr}"
for file in "$train" "$valid" "$source_bundle/manifest.json" configs/fingerprint_v3/qwen3_8b_official_high_lr_resume.json; do
  if [[ ! -f "$file" ]]; then
    echo "Missing input: $file" >&2
    exit 1
  fi
done
exec python scripts/fingerprint_v3.py train \
  --config configs/fingerprint_v3/qwen3_8b_official_high_lr_resume.json \
  --train "$train" --valid "$valid" --resume-external-from "$source_bundle" \
  --out "${FHF_RUN_NAME:-v3-qwen8b-eos-ext2-official4-high-lr-resume}" --wandb
