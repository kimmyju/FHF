# Fingerprint v3: decoder attribution experiment

**Target: improve toward 0.70 BAcc. No 0.70 result is claimed.** This branch is
based on fingerprint-v2 commit b38987d and keeps the main/v2 entry points intact.
It adds a genuinely different representation, not another loss to the old model.

## Why this experiment

Retrieved user results: prompt-isolated v2 BAcc 0.5296 (previous ~0.5185),
DeBERTa experiment 0.27778 with train accuracy 0.37564. A reported confusion
matrix had GLM recalls 6.7% and 13.3%, often predicted as Claude Fable.
These are prior user reports, not results reproduced by this branch. The exact
0.5296 checkpoint and configuration were not available in the repository.

- A 4B decoder with LoRA in attention **and MLP** provides a new pretrained
  representation. Existing main uses ModernBERT-large; v2 retains that family
  of architecture/losses. Larger capacity is a hypothesis, not evidence of a gain.
- Direct, class-balanced 18-way CE removes the hierarchical contribution and
  auxiliary losses. **v2 already mixes a flat head with its hierarchical head**,
  so provider gating is not proven to cause the plateau. This is a simpler
  alternative objective to test, not a bug fix or a guaranteed improvement.
  No oracle family information is used at inference.
- The decoder reads the raw passage and pools the last EOS token. It never
  generates a model name, uses a chat template, or receives prompt_id as input.
- 768 tokens, with head+tail retention when necessary, preserve openings and
  endings. The competition says **40–250 words**, not characters. Real token
  length and truncation counts are written for every fold. Increase the length
  only if the audit supports it; head+tail still loses middle text when truncated.
- Fixed 85% decoder + 15% case-sensitive character TF-IDF is a conservative
  complementary baseline. All three results are reported. The coefficient is
  not claimed optimal. If lexical hurts, validate `--lexical-weight 0` separately.
- Each prompt stays in one CV fold. Epoch count is fixed BEFORE fitting, and
  outer-fold labels do not select a checkpoint. This avoids reporting the best
  outer-fold epoch as clean OOF. Repeatedly choosing hyperparameters on these
  folds still produces selection bias; keep the old 540-row split for comparison.
- Exact duplicates cause an error. IDs retain string type. All fold train and
  holdout sets must contain all 18 labels. Gradient accumulation handles partial
  windows with correct sample-count normalization.

No new external data collection, API spending, or competition submission occurs.
The existing 62,910-row pool is optional because its model provenance and prompt
independence cannot be established from filenames. More data is not automatically
better data.

## Install

Install a GPU-compatible PyTorch first (your Blackwell server needs a compatible
CUDA build), then from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements-v3.txt
python -m pytest tests/test_v3_decoder.py -q
```

The tested APIs are Transformers 4.57.1 and PEFT 0.17.1. Qwen3-4B weights are
fetched on the training machine. A normal BF16 LoRA path is used, without
bitsandbytes/FlashAttention installation dependencies. CPU fallback is for tests;
4B training requires a suitable GPU in practice. Batch 4 / accumulation 8 is an
initial setting, not a measured memory guarantee. On OOM use batch 2 / accumulation
16, keeping the effective batch at 32. TF32/FP16 assumptions are not required.

## 1. Compare to your old 0.5296 run first

Use the ORIGINAL official 2,159/540 split, not a train CSV already containing
external data. Output directories must be new; partial runs are deliberately not
resumed silently.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v3.py train \
  --train outputs/dual-path-prompt-aware/train_split.csv \
  --valid outputs/dual-path-prompt-aware/val.csv \
  --out outputs/fingerprint-v3/qwen4b-holdout --wandb
```

Read `metrics.json`: neural / lexical / blend BAcc, family accuracy,
within-family accuracy conditional on correct family, each label's recall and
18x18 confusion. `predictions.csv` contains ID-aligned probabilities. Inspect
GLM/Fable, Gemini 3.7/3.8, and GPT Sol/Luna, rather than just aggregate BAcc.
`history.json` has training loss/accuracy per stage; W&B logs these when enabled.
There is one final holdout evaluation, not an epoch-wise early-stopping loop.

## 2. Controlled external warm-up (optional second experiment)

Supply ONE already-audited CSV with `id,text,model,prompt_id`, actual exact labels,
unique IDs/responses, and all 18 classes. The code does not recursively crawl the
mixed ntl_v2/ntl_v3 directory and does not relabel historical models.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v3.py train \
  --train outputs/dual-path-prompt-aware/train_split.csv \
  --valid outputs/dual-path-prompt-aware/val.csv \
  --external /absolute/path/to/audited_external.csv \
  --out outputs/fingerprint-v3/qwen4b-external-holdout --wandb
```

Default warm-up: max 1,000 rows per exact label, one epoch, then four official-only
epochs with a reset optimizer/schedule. This bounds initial cost and prevents
external rows from dominating official adaptation. It transfers the entire
trained adapter AND head. Exact response overlaps with all official rows and
literal prompt_id collisions are removed. Different prompt namespaces do NOT
prove semantic prompt disjointness. Review source provenance yourself and treat
external-assisted validation accordingly. This optional path is not evidence
that external data will improve your score.

## 3. Full grouped CV, then fold ensemble

Once the architecture/epochs are fixed, use all 2,699 OFFICIAL rows:

```bash
# Budget pilot; this is labeled single_fold_pilot, never full OOF.
CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v3.py train \
  --train data/train.csv --fold-only 0 --out outputs/fingerprint-v3/pilot

# All five independent folds; no external data by default.
CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v3.py train \
  --train data/train.csv --out outputs/fingerprint-v3/qwen4b-cv --wandb
```

Each fold starts from the pretrained base. Five adapters are averaged at test
inference; only one full model is in memory at a time. Holdout runs and one-fold
pilots also yield valid single-model bundles, but are not five-fold CV ensembles.
For an 8B capacity ablation, use `--model Qwen/Qwen3-8B --batch-size 2
--accumulation 16` with a different output path. Do not run all variants blindly:
first establish that 4B improves the frozen holdout, then compare costs and errors.
The pipeline intentionally does not learn 18 per-class blend knobs on 540 rows.

## 4. Offline Kaggle inference

Upload the chosen output bundle (manifest + fold adapters/tokenizers/configs +
lexical models), this source package, offline dependency wheels, and ONE complete
Qwen3-4B base snapshot as Kaggle datasets. Use the base revision recorded in
`fold-0/config.json`; never substitute an instruct/base/8B checkpoint. Do not put
training/validation files into the inference notebook inputs unless needed.
The tokenizer and the trained score head are saved with each fold adapter.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/fingerprint_v3.py predict \
  --bundle /kaggle/input/fhf-v3/qwen4b-cv \
  --base /kaggle/input/qwen3-4b \
  --test /kaggle/input/competitions/name-that-llm/test.csv \
  --batch-size 4 --out /kaggle/working/submission.parquet
```

The output is exactly `id,model`, preserving test ID order. No prompt_id is needed.
All model loads are local-only. Only load joblib bundles you created/trust.
A runtime sidecar records elapsed seconds, rows and folds. Benchmark the FULL
~4,000-row workload on the actual Kaggle accelerator before submitting; the
one/few-row interactive test is not a runtime benchmark. The 9-hour limit is not
verified by this implementation. Reducing inference batch size controls memory;
using a single validated fold reduces compute but changes the ensemble. Inference
on non-BF16 CUDA devices uses FP16 to avoid loading 4B weights in FP32.

## Decision rule and limits

Compare official-only decoder against the same frozen split first. Add warm-up
only if it improves that comparison. Check every class, not only family accuracy.
A jump from 0.5296 to 0.70 requires about **92 additional correct predictions** on
an approximately balanced 540-row holdout; GLM improvements alone cannot explain
all of it. If decoder representations do not improve separability, the next
bottleneck is data quality/coverage, not another unvalidated loss weight.

No competition data or 4B GPU is present in the development environment. Tests
use a tiny randomly initialized Qwen3 to verify training, gradient accumulation,
LoRA/head persistence and offline Parquet inference. They establish software
behavior, not attribution accuracy. No leaderboard or 0.70 measurement exists.

Sources: [competition](https://www.kaggle.com/competitions/name-that-llm/overview/description),
[data](https://www.kaggle.com/competitions/name-that-llm/data),
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
[PEFT sequence classification heads](https://huggingface.co/docs/peft/developer_guides/troubleshooting).
