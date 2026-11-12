#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FHF_ROOT="${FHF_ROOT:-/media/NAS/USERS/yeju/FHF}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
train="$FHF_ROOT/csv/splits/dual-path-prompt-aware/train_split.csv"
valid="$FHF_ROOT/csv/splits/dual-path-prompt-aware/val.csv"
external="$FHF_ROOT/csv/external/full/selected_3495_per_model.csv"
config="configs/fingerprint_v4/qwen3_4b_source_likelihood.json"
for file in "$train" "$valid" "$external" "$config"; do
  if [[ ! -f "$file" ]]; then echo "Missing input: $file" >&2; exit 1; fi
done
exec python scripts/fingerprint_v4.py train --config "$config" \
  --train "$train" --valid "$valid" --external "$external" \
  --out "${FHF_RUN_NAME:-qwen4b-source-likelihood-ext2-official4}" --wandb
