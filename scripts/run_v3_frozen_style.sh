#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FHF_ROOT="${FHF_ROOT:-/media/NAS/USERS/yeju/FHF}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source_bundle="${FHF_SOURCE_BUNDLE:-$FHF_ROOT/results/v3-qwen8b-ext2-official4}"
train="$FHF_ROOT/csv/splits/dual-path-prompt-aware/train_split.csv"
valid="$FHF_ROOT/csv/splits/dual-path-prompt-aware/val.csv"
for file in "$train" "$valid" "$source_bundle/manifest.json" "$source_bundle/metrics.json" "$source_bundle/predictions.csv" configs/fingerprint_v3/qwen3_8b_frozen_style.json; do
  if [[ ! -f "$file" ]]; then
    echo "Missing input: $file (set FHF_SOURCE_BUNDLE to the original EOS results directory if needed)" >&2
    exit 1
  fi
done
extra=()
if [[ -n "${FHF_BASE_MODEL:-}" ]]; then extra+=(--base "$FHF_BASE_MODEL"); fi
exec python scripts/fingerprint_v3_frozen_style.py train \
  --config configs/fingerprint_v3/qwen3_8b_frozen_style.json \
  --source "$source_bundle" --train "$train" --valid "$valid" \
  --out "${FHF_RUN_NAME:-v3-qwen8b-frozen-eos-style}" --wandb "${extra[@]}"
