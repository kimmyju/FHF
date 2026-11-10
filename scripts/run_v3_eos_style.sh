#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FHF_ROOT="${FHF_ROOT:-/media/NAS/USERS/yeju/FHF}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
train="$FHF_ROOT/csv/splits/dual-path-prompt-aware/train_split.csv"
valid="$FHF_ROOT/csv/splits/dual-path-prompt-aware/val.csv"
external="$FHF_ROOT/csv/external/full/selected_3495_per_model.csv"
for file in "$train" "$valid" "$external" configs/fingerprint_v3/qwen3_8b_eos_style.json; do
  if [[ ! -f "$file" ]]; then
    echo "Missing input: $file" >&2
    exit 1
  fi
done
exec python scripts/fingerprint_v3.py train \
  --config configs/fingerprint_v3/qwen3_8b_eos_style.json \
  --train "$train" --valid "$valid" --external "$external" \
  --out "${FHF_RUN_NAME:-v3-qwen8b-eosstyle-ext2-official4}" --wandb
