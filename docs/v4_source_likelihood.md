# v4: 18-source conditional language modeling

This is a new experiment inspired by ATLAS (https://arxiv.org/abs/2610.19583), adapted from family attribution to **18 exact model labels**. It is not a claimed paper reproduction or a promised 70% result. The existing 60.19% Qwen3-8B EOS model remains the baseline and is not loaded, edited, deleted or overwritten by v4.

## Execute on the training server

```bash
cd /home/yeju/repos/FHF
git switch experiments/fingerprint-v3-decoder
git pull --ff-only origin experiments/fingerprint-v3-decoder
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v4_source_likelihood.sh
```

Uses the existing `fhf` environment and project dependencies (torch, transformers, safetensors, pandas, numpy, sklearn, joblib, pyarrow, wandb). CPU tests used torch 2.15.0, transformers 4.57.1; server-specific 5.x behavior and full 4B GPU execution still need actual validation. No additional PEFT training API is used for v4; source-specific adapters are implemented directly. If needed, install the existing project with `python -m pip install -e .` rather than silently downgrading the working v3 environment.

Config: `configs/fingerprint_v4/qwen3_4b_source_likelihood.json` (JSON, not YAML).

| Setting | Value |
|---|---|
| Base | Qwen/Qwen3-4B CausalLM |
| Trainable parameters | One shared A per attention projection; 18 source-specific B matrices; 18 one-token soft prefixes |
| Frozen parameters | All original model weights, including embeddings and LM head |
| LoRA projections | q/k/v/o in every layer |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| External / official epochs | 2 / 4 |
| External / official LR | 1e-4 / 2e-5 |
| Microbatch / accumulation | 2 / 16 (maximum 32 examples per update) |
| Max text tokens | 512 plus one prefix token |
| Eval batch | 1, all 18 source conditions |
| Vocab-logit chunk | 32 token positions |
| Score temperatures | 1, 2, 4, 8 |

CUDA BF16 training on the user's Blackwell GPU is intended. Non-BF16 GPU training is rejected rather than running unstable unscaled FP16. CPU is supported for small tests. GPU inference uses BF16 where supported, otherwise FP16. Quantized inference is **not implemented**. No claim is made that the unmeasured 4B inference fits a particular GPU or completes within competition runtime.

## Inputs and preservation

Default `FHF_ROOT=/media/NAS/USERS/yeju/FHF`:
- `csv/splits/dual-path-prompt-aware/train_split.csv`
- `csv/splits/dual-path-prompt-aware/val.csv`
- `csv/external/full/selected_3495_per_model.csv`

The launcher checks all files. All v4 outputs, including checkpoints, live in one independent bundle:

`/media/NAS/USERS/yeju/FHF/likelihood_v4/qwen4b-source-likelihood-ext2-official4/`

No writes go into previous v3 result/checkpoint directories. Existing output directories are refused. For a repeated run:

```bash
FHF_RUN_NAME="qwen4b-source-likelihood-$(date +%Y%m%d-%H%M%S)" \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v4_source_likelihood.sh
```

## Training and scoring semantics

A source label chooses the prefix and B matrices; the loss predicts **the original text tokens**, not a label-name token. No prompt_id/source_dataset/origin is fed to the neural model. Every source starts with zero LoRA B and identical prefix initialization from the base EOS embedding. The shared A is randomly initialized. Source identity must therefore be learned, not inferred from arbitrary initial score offsets.

A soft prefix predicts the first real token; text position t predicts token t+1. Padding and the prefix itself are excluded from the loss. No artificial EOS token is scored. Text truncation keeps a contiguous beginning (not head-tail concatenation) to avoid inventing false next-token transitions. Truncation rates are saved.

Each text's NLL is averaged over its valid tokens; per-class weights balance the training distribution. Microbatches are grouped by true source within each accumulation window. Forward AND backward complete before switching source, so gradient checkpoint recomputation uses the correct adapter. Original parameters stay frozen, while shared A, B and prefixes are optimized. Each stage gets a fresh AdamW/cosine schedule.

At inference, score every text under all 18 source conditions. For source k and real token t:

`r[k,t,T] = exp(-sum_j max(logp[j,t] - logp[k,t], 0) / T)`

Average over real tokens and configured temperatures, then choose the maximum source score. This follows the paper's bounded relative evidence aggregation. Saved `score::*` columns are **not calibrated probabilities** and must not be treated as `neural::*` probability blends without separate calibration work.

Evaluation runs at every epoch. External epoch 1 is diagnostic; candidates are external epoch 2 and official epochs 1–4. Select highest validation BAcc, retain earlier candidate on ties. Validation labels are not used in training gradients. Metrics explicitly say `validation_selected_holdout`, not unbiased test performance. The best v3 model is not a selection candidate, so v4 may perform worse.

## Data audit

`data_audit.json` includes class counts, per-model word-length quantiles, source_dataset/origin cross-tabs where present, numbers of models per prompt, exact response overlap and prompt-ID overlap. The v3 overlap filter removes external responses/prompt IDs shared with either official split, then caps classes. Missing labels, duplicate normalized responses within a split, and official split leakage fail loudly. The audit does not prove semantic prompt independence, exact model-version provenance, or near-duplicate absence.

Optional audit-only run (no model download/training):

```bash
PYTHONPATH=src python scripts/fingerprint_v4.py audit \
  --train /media/NAS/USERS/yeju/FHF/csv/splits/dual-path-prompt-aware/train_split.csv \
  --valid /media/NAS/USERS/yeju/FHF/csv/splits/dual-path-prompt-aware/val.csv \
  --external /media/NAS/USERS/yeju/FHF/csv/external/full/selected_3495_per_model.csv \
  --out /media/NAS/USERS/yeju/FHF/likelihood_v4/data_audit_only.json
```

## Outputs and logs

- `metrics.json`: selected BAcc/family/within-family, selected stage/epoch, validation hash.
- `epoch_metrics.json`: metrics for every epoch, updated as the run progresses.
- `selection.json`, `manifest.json`: chosen source adapter.
- `epochs/<stage>-<epoch>/source_adapter.safetensors`: shared A, all B and prefixes only; no full base copies.
- `epochs/<stage>-<epoch>/validation_scores.npy`, `predictions.csv`: 18-source validation scores.
- `data_audit.json`, `*_tokenization.json`, `parameters.json`, `config.json`, `labels.json`, `tokenizer/`.
- `history.json`: completed-stage training NLL and validation metrics.
- W&B: external/official train_nll and validation BAcc/family/within-family.

Unlike a single EOS classification pass, each validation epoch runs 18 source-conditioned passes. Do not interpret the slower evaluation as a hang. This implementation does teacher-forced scoring, not autoregressive text generation.

## Offline inference and GPU measurement

Attach the complete v4 bundle and exact original Qwen3-4B base weights as local inputs. Use the dedicated v4 command (v3 predict cannot read these adapters):

```bash
PYTHONPATH=src python scripts/fingerprint_v4.py predict \
  --bundle /media/NAS/USERS/yeju/FHF/likelihood_v4/qwen4b-source-likelihood-ext2-official4 \
  --base /absolute/path/to/local/Qwen3-4B \
  --test /absolute/path/to/test.csv \
  --out /absolute/path/to/submission.parquet --batch-size 1
```

All inference loads are local-only. Output is id/model parquet. `<output>.runtime.json` records total seconds (including loading), peak GPU allocated GB, and a simple 4000-row runtime estimate. Benchmark on representative text lengths on the actual competition GPU before relying on it; peak allocated memory excludes some driver/library overhead. 18 conditions share one base model but multiply scoring work. No test-time API calls or internet are required.
