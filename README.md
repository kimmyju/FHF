**New research pipeline:** [Fingerprint V2 guide](docs/fingerprint_v2.md). The original C/E1 experiments are preserved.

# FHF: Fine-grained Hierarchical Fingerprinting

Config-driven research implementation for the [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) competition. This is an **experimental pipeline**, not a trained model or a verified Kaggle leaderboard result.

## Encoder: what is currently configured?

**Default: `answerdotai/ModernBERT-large` (about 395M parameters).** Its publicly pretrained encoder is loaded from Hugging Face, then adapted using PEFT LoRA rather than updating the full backbone. No pre-fine-tuned FHF checkpoint is bundled. Set `model.encoder` and optionally `model.tokenizer` in YAML to use another pretrained encoder. The project uses `AutoModel`, so this path must point to an **encoder-compatible** pretrained model (not an arbitrary generative/decoder-only LLM). The shared output uses masked mean pooling.

## YAML experiments

All paths, training hyperparameters, encoder settings, individual head switches, and individual loss switches are maintained under `configs/`:

```text
configs/
  base.yaml                       default full hierarchical model
  kaggle.yaml                     offline Kaggle submission mounts
  external.yaml                   optional external-data pretraining
  ablations/
    model_only.yaml               direct 18-class head only
    family_only.yaml              6-class provider head only
    no_contrastive.yaml           hierarchy without projection/contrastive
    hotflip.yaml                  hierarchy plus constrained HotFlip
```

`configs/base.yaml` includes:

```yaml
paths:
  base_dir: .
  train_csv: data/train.csv
  checkpoint_dir: outputs/fhf-base
  test_csv: data/test.csv
  submission_parquet: submission.parquet
model:
  encoder: answerdotai/ModernBERT-large
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: [Wqkv, Wo]
    bias: none
    initial_adapter: null
    bundle_base_encoder: false
  tokenizer: null
  max_length: 384
  dropout: 0.10
  projection_dim: 128
  heads:
    family: true
    model: true
    projection: true
losses:
  exact_classification: {enabled: true, weight: 1.0}
  family: {enabled: true, weight: 0.25}
  intra_contrastive:
    enabled: true
    weight: 0.25
    temperature: 0.12
    hard_weight: 2.0
  rdrop: {enabled: true, weight: 0.05}
  hotflip: {enabled: false, weight: 0.05, max_examples: 2}
```

The exact YAML additionally sets every train/validation/report path, seed, learning rate, epochs, batch sampling, gradient clipping, pretraining and inference batch size. Relative paths resolve from `paths.base_dir`; absolute paths also work. Each ablation YAML inherits `../base.yaml`, replacing only selected values. The resolved configuration is saved beside the checkpoint for reproducibility.

### Independent head combinations

| YAML selection | Family head | Model head | Projection head | Prediction space |
|---|---|---|---|---|
| `base.yaml` | ON | ON | ON | hierarchical 18-class |
| `ablations/model_only.yaml` | OFF | ON | OFF | flat 18-class |
| `ablations/family_only.yaml` | ON | OFF | OFF | 6-provider only |
| `ablations/no_contrastive.yaml` | ON | ON | OFF | hierarchical 18-class |
| `ablations/hotflip.yaml` | ON | ON | ON | hierarchical 18-class |

The provider-only checkpoint is useful for ablations but **cannot submit to Kaggle** because the competition requires an exact 18-class label. The prediction script rejects it explicitly. Head and loss flags are validated for incompatible combinations.

When family and model heads are enabled, the exact model distribution is `P(model|x) = P(family|x) * P(model|family,x)`. When only the model head is enabled, a flat 18-class softmax is used. Supervised contrastive learning uses same-model positives and more heavily penalizes within-family different-model negatives.

## Installation

Python 3.10+, PyTorch, Transformers and PyYAML:

```bash
git clone https://github.com/kimmyju/FHF.git
cd FHF
python -m pip install -e ".[dev]"
python -m pytest -q
```

If you use GPU, install the appropriate PyTorch CUDA wheels. Model weights are downloaded by Hugging Face only where Internet is allowed.

## Train

Download the Kaggle competition `train.csv` to the configured path. Expected columns: `id,text,model,prompt_id`.

```bash
# Default 2-head hierarchy and contrastive learning
python scripts/train.py --config configs/base.yaml

# Run separate head experiments
python scripts/train.py --config configs/ablations/model_only.yaml
python scripts/train.py --config configs/ablations/family_only.yaml
python scripts/train.py --config configs/ablations/no_contrastive.yaml
python scripts/train.py --config configs/ablations/hotflip.yaml

# Change encoder, paths, epochs and individual components without editing Python:
python scripts/train.py --config configs/base.yaml \
  --set model.encoder=distilbert/distilbert-base-uncased \
  --set paths.train_csv=/mnt/my_data/train.csv \
  --set paths.checkpoint_dir=/mnt/experiments/run01 \
  --set paths.validation_csv=/mnt/experiments/run01/val.csv \
  --set paths.train_split_csv=/mnt/experiments/run01/train.csv \
  --set paths.history_json=/mnt/experiments/run01/history.json \
  --set paths.config_snapshot=/mnt/experiments/run01/config.yaml \
  --set training.epochs=3 --set losses.intra_contrastive.enabled=false
```

When changing the encoder, ensure `model.tokenizer` is null or points to its compatible tokenizer. Avoid mixing tokenizers between checkpoints. Rather than overriding many paths each run, create a child YAML with `extends: base.yaml` and a different `paths` section.

By default FHF trains the ModernBERT attention projections `Wqkv` and `Wo` through PEFT LoRA with rank 16 and alpha 32. The encoder's frozen original weights do not receive optimizer updates; each enabled FHF head still trains normally. Configure `model.lora.enabled: false` for a full fine-tuning ablation. Mixed precision defaults to `auto` (bf16 if supported, otherwise fp16 on CUDA), and `training.gradient_accumulation_steps: 2`. You can override `model.lora.r`, `model.lora.alpha`, `model.lora.target_modules`, `training.mixed_precision`, and `training.learning_rate` independently. LoRA default LR is 1e-4, a starting point requiring empirical validation.

The default split is **prompt-disjoint**, rather than random row-level. The code uses a family-balanced minibatch sampler so same-model positives exist for contrastive learning. The best validation checkpoint is saved at `paths.checkpoint_dir`.

## External attribution pretraining (optional)

Provide CSV columns `text,model` and optionally `provider`. Keep the actual generator name, even if it is from a historical model; **do not rename it to a Kaggle target label**.

```bash
python scripts/pretrain_external.py --config configs/external.yaml
```

When LoRA is enabled, external pretraining saves `outputs/fhf-external/adapter` and `outputs/fhf-external/tokenizer` rather than a full copy of the base. To transfer the adapter into FHF, keep `model.encoder: answerdotai/ModernBERT-large` and set `model.lora.initial_adapter: outputs/fhf-external/adapter` plus `model.tokenizer: outputs/fhf-external/tokenizer` in your child YAML. The classifier heads are freshly initialized. Externally pretrained models can cause negative transfer, and the simple external-data loader does not yet enforce model-balanced minibatches.

## Evaluate and predict

```bash
python scripts/evaluate.py --config configs/base.yaml
python scripts/evaluate.py --config configs/ablations/family_only.yaml

python scripts/predict.py --config configs/base.yaml
# Or use custom input/output paths:
python scripts/predict.py --config configs/base.yaml \
  --set paths.test_csv=/data/test.csv \
  --set paths.submission_parquet=/tmp/submission.parquet
```

Evaluation uses exact-model recall/balanced accuracy for 18-class checkpoints and separately labeled family metrics for family-only checkpoints. All 18-class predictions are written as a Parquet file with `id` and `model` columns, no missing values, no duplicate IDs.

## Kaggle notebook (Internet OFF)

1. Train externally, then upload the FHF checkpoint folder containing `adapter/`, `tokenizer/`, `heads.pt`, and `metadata.json` to Kaggle as a **private** Dataset/Model. This contains **only LoRA weights and trainable heads**, not the frozen 395M ModernBERT backbone. When `model.lora.bundle_base_encoder: true`, a `base_encoder/` copy is also saved, increasing disk use substantially.
2. For adapter-only checkpoints, attach the original `answerdotai/ModernBERT-large` complete model folder as another Kaggle Dataset at a local path, such as `/kaggle/input/modernbert-large`. The default `configs/kaggle.yaml` provides `paths.inference_base_encoder` for this purpose. **Internet is disabled**, so the base weights must be present locally. Attach this source repository with `src`, `scripts` and `configs` as an additional dataset.
3. Edit `configs/kaggle.yaml` for your Kaggle mount names (or pass `--set paths.checkpoint_dir=...`).
4. Ensure `peft`, `transformers`, `torch`, `PyYAML`, `pandas`, `pyarrow`, and `safetensors` are available offline. Run `notebooks/kaggle_submit.ipynb` or:

```bash
python scripts/kaggle_submit.py --config configs/kaggle.yaml
```

The hidden competition test CSV is never available for local model selection. The interactive Kaggle Notebook may provide a one-row example. Kaggle reruns against the hidden test with Internet disabled; CPU or GPU Notebook runtime must be within nine hours. Make sure PyTorch, Transformers, PyYAML, Pandas, and PyArrow are available offline.

## Project files

- `src/fhf/config.py` - YAML inheritance, overrides, validation and path resolution
- `src/fhf/labels.py` - Competition's exact 18 labels
- `src/fhf/data.py` - Dataset loading, grouped splitting, balanced batches
- `src/fhf/model.py` - Switchable heads, hierarchical/flat inference, offline checkpoints
- `src/fhf/losses.py` - Loss objectives
- `src/fhf/hotflip.py` - Optional gradient-based constrained marker substitution
- `src/fhf/engine.py` - Training and evaluation loops
- `src/fhf/metrics.py` - Balanced accuracy, per-model recall, confusion matrix
- `scripts/train.py`, `evaluate.py`, `predict.py`, `pretrain_external.py`, `kaggle_submit.py`
- `tests/` - Unit tests including YAML presets and independently enabled heads

## Research and limitations

This is inspired by the two-step optimization and gradient-guided token search of *Prompt Injection Attack to Tool Selection in LLM Agents* (ToolHijacker, NDSS 2026), adapted to benign LLM authorship attribution. These are established ingredients, not proof of research novelty. HotFlip is disabled by default; replacing discourse markers may alter the true author fingerprint. Its efficacy and stability on unseen prompts must be established empirically. No external dataset, actual model weights, hidden test labels or benchmark scores are included.

Code license: MIT. External checkpoints and datasets retain their own licenses.

## ModernBERT + LoRA design notes

- Pretrained backbone: [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large), a bidirectional encoder, not a generative LLM.
- LoRA implementation: [PEFT](https://huggingface.co/docs/peft/index). Adapter target modules are `Wqkv` and `Wo`, ModernBERT's attention projections. Model compatibility follows the documented [Sentence Transformers ModernBERT PEFT example](https://www.sbert.net/examples/multi_vector_encoder/training/peft/).
- Fine-tuning: LoRA matrices and enabled family/model/projection heads are updated; the underlying encoder's pretrained weights remain frozen. This reduces optimizer memory requirements but full encoder forward and backward activations still consume GPU memory.
- Precision: `training.mixed_precision: auto` uses bf16 if the GPU supports it, otherwise fp16. CPU uses full precision.
- Checkpoint: adapter-only unless `model.lora.bundle_base_encoder: true`. The frozen original base must be mounted independently for offline Kaggle inference; a missing local model raises a descriptive error.
- All ablations inherit LoRA defaults, ensuring fair comparisons. Set `model.lora.enabled: false` in a new YAML to compare full fine-tuning.
- **Validation required:** the project has no verified competition score or completed real ModernBERT-large training run. Run a minimal dry-run and memory benchmark on your actual GPU before scaling up.

## W&B dashboards: Loss, Accuracy and Model Fingerprints

FHF integrates [Weights & Biases](https://wandb.ai/) training/validation charts and optional hyperparameter sweeps. **Your API key never belongs in a repository file.**

The standard `configs/base.yaml` enables W&B **offline mode** by default. Offline runs save local events under `outputs/wandb/` without credentials or Internet. To upload live metrics:

```bash
wandb login  # enter your API key interactively, or set WANDB_API_KEY externally
python scripts/train.py --config configs/base.yaml --set wandb.mode=online
```

The following charts and summaries are logged:
- `train/loss`, `train/loss_exact_classification`, `train/loss_family`, `train/loss_intra_contrastive`: individual training losses by optimizer step.
- `train/accuracy`, `train/learning_rate`: minibatch accuracy and learning rate by optimizer step.
- `val/loss`, `val/loss_exact`, `val/loss_family`: held-out validation loss by epoch.
- `val/accuracy`, `val/balanced_accuracy`, `val/family_accuracy`, `val/within_family_accuracy`: exact model and family validation metrics.
- `val/recall/<model_name>`: per-model recall on held-out prompts.
- `val/confusion_matrix`: W&B confusion matrix for every epoch (disable with `wandb.log_confusion_matrix: false`).
- `val/best_balanced_accuracy`: best observed held-out balanced accuracy so far, the sweep optimization target; `val/best_family_balanced_accuracy` for a family-only run.

YAML config example:

```yaml
wandb:
  enabled: true
  mode: offline  # online after 'wandb login'
  project: fhf-name-that-llm
  entity: null  # optional W&B team or user account
  run_name: null
  group: null
  tags: [modernbert, lora, attribution]
  dir: outputs/wandb
  log_every_n_steps: 5
  log_confusion_matrix: true
```

To sync local history later (after `wandb login`):

```bash
wandb sync outputs/wandb/wandb/offline-run-*
```

### Hyperparameter sweeps

All sweep-specific YAML files live under **[sweeps/](sweeps/)**, separate from ordinary configurations:

```text
sweeps/
  lora_bayes.yaml        LoRA rank, alpha, dropout, LR, key loss weights
  loss_grid.yaml         Controlled loss ablation matrix
  model_only_bayes.yaml  Flat 18-class head sweep
  family_only_bayes.yaml Six-provider head sweep
  README.md              Sweep usage and safeguards
```

Start one sweep from the repository root:

```bash
wandb login
wandb sweep sweeps/lora_bayes.yaml
wandb agent YOUR_ENTITY/fhf-name-that-llm/SWEEP_ID
```

The sweep agent launches `scripts/sweep_train.py`. It applies W&B-provided dotted-key parameter overrides to the selected normal YAML config and saves every trial under a **unique `outputs/sweeps/<run_id>/` directory** to avoid overwriting checkpoints. The same validated prompt-group split is reused unless you change the seed.

Sweeps run in **online** mode and require a connected W&B account. Restrict the number of trial agents according to available VRAM; concurrently loading multiple ModernBERT-large models can exhaust a GPU. Sweeps repeatedly select against the validation split, so report final performance on a genuinely separate held-out set. Sweep values and ranges are exploratory, not established best hyperparameters.

To disable all W&B telemetry for one run:

```bash
python scripts/train.py --config configs/base.yaml --set wandb.enabled=false
```

Local `history.json` reports remain available regardless of W&B configuration.

## Training beyond 6 epochs: new batches, optional scheduler, early stopping

Previously `FamilyBatchSampler.__iter__` initialized NumPy RNG with the same seed
every epoch, repeating the same balanced minibatch indices and order. The sampler
now accepts `set_epoch(epoch)`, and the training loop calls it at each epoch
boundary. The same seed and epoch still reproduce the same balanced batch
sequence. Note that this is a *with-replacement balanced sampler*, so it does
**not** guarantee every training example is seen once per epoch.

To try a longer run independently of the existing 6-epoch experiment:

```bash
git pull
python scripts/train.py --config configs/ablations/longer_training.yaml
```

The new preset runs up to 15 epochs with `training.scheduler.name: cosine`,
`training.scheduler.warmup_ratio: 0.05`, and early stopping
(`patience: 4`, `min_delta: 0.001`). It has a separate output directory,
`outputs/fhf-15ep-cosine/`, to avoid overwriting the earlier model. A
checkpoint is saved on significant validation BAcc improvement, and training
stops after four consecutive non-improving epochs.

You can also override each option on the command line:

```bash
python scripts/train.py --config configs/base.yaml \
  --set training.epochs=15 \
  --set training.scheduler.name=cosine \
  --set training.early_stopping.enabled=true
```

For a fair assessment, compare **validation BAcc rather than combined training
loss alone**. The latter includes hierarchical and contrastive objectives;
the validation loss reported under `val/loss` is exact-classification NLL
(for 18-class runs) and is not directly the same weighted objective. W&B now
also logs `epoch_train/loss` and `epoch_train/accuracy` by epoch as additional
summary plots.

**No automatic resume:** FHF currently loads its base model at the beginning
of each run. Switching the YAML to 15 epochs starts a new training job.
The existing best LoRA checkpoint may be reused manually as an initial
adapter via `model.lora.initial_adapter`, but that does **not** restore the
optimizer, scheduler, or previously trained classification heads and is not
a proper 6→15-epoch resume. Do not call that a continued training run.


## Oracle within-family accuracy (diagnostic only)

Validation now logs two distinct within-family measures:

- `val/within_family_accuracy`: exact-model accuracy **conditioned on the normal
  prediction's family being correct**. The subset changes with Family Head quality.
- `val/oracle_within_family_accuracy`: for **every** labeled validation example,
  reveal the ground-truth family, restrict the raw 18-class Model Head scores to
  models in that family, then take the highest-scoring one. This directly probes
  within-family ranking independently of family prediction.
- `val/oracle_within_family_accuracy_by_family/<family>`: optional per-family
  diagnostic, averaged over labeled validation examples from that family.

The oracle value is also written to `history.json` and the output of
`scripts/evaluate.py` in `evaluation.json`. It does **not** use gold labels
for training or Kaggle submission; normal predictions and checkpoint selection
remain unchanged. A family-only ablation cannot report oracle within-family
accuracy because it has no 18-class Model Head.

To evaluate an **existing** best checkpoint without retraining:

```bash
git pull
python scripts/evaluate.py --config configs/ablations/longer_training.yaml
```

This writes `outputs/fhf-15ep-cosine/evaluation.json`. It does not retroactively
add points to a completed W&B training run; newly started runs will log
`val/oracle_within_family_accuracy` per epoch.

## Style-only baseline: character n-gram TF-IDF + Logistic Regression

This **independent CPU-friendly baseline** tests whether surface character style
contains extra 18-model attribution information beyond ModernBERT. It does not
modify the FHF model, heads, sampler or training code. It fits a 3–5 character
TF-IDF vocabulary **only on the training split** and trains a multinomial
logistic regression classifier. Character n-grams also capture topical words;
therefore, results should not be interpreted as pure stylistic evidence.

From the repository root:

```bash
git pull
python scripts/style_baseline.py
```

By default the script reads the exact `train_split.csv` and `val.csv`
configured in `configs/ablations/longer_training.yaml`. If **neither**
saved split exists, it regenerates the same prompt-disjoint split from
`data/train.csv` using that config's validation fraction and seed. It rejects
overlapping prompt IDs and does not create a split by randomizing text rows.
For comparison to another experiment:

```bash
python scripts/style_baseline.py --config configs/ablations/model_only.yaml
```

Outputs are isolated in `outputs/style-baseline/`:

- `style_model.joblib`: fitted TF-IDF + Logistic Regression model
- `evaluation.json`: 18-class accuracy, balanced accuracy, per-class recall,
  family accuracy, confusion matrix and split provenance
- `val_predictions.csv`: ID, prompt ID, ground-truth and predicted model plus
  all 18 class probabilities, useful for later fusion experiments

Optional CLI overrides include `--ngram-min`, `--ngram-max`,
`--max-features`, `--min-df`, `--C`, and `--output-dir`.
The default `--output-dir` is shared across configs, so specify another
output directory if keeping several style-only experiments:

```bash
python scripts/style_baseline.py --config configs/ablations/model_only.yaml \
  --output-dir outputs/style-model-only
```

Use **the same prompt-disjoint split** when comparing with the ModernBERT
18-class BAcc of approximately 0.33. This is a baseline only, not an
implemented neural Style Branch or a guaranteed performance improvement.
Only load a `.joblib` checkpoint if you trust its source.

## ModernBERT + Style-only probability fusion (evaluation only)

Run inference from the **saved** 18-class FHF checkpoint, join its validation
probabilities with the Style-only TF-IDF predictions **by sample ID**, then
compare a predeclared BERT:Style weight grid. No training, checkpoint updates,
or neural Style Branch changes occur.

```bash
git pull
python scripts/evaluate_fusion.py \
  --config configs/ablations/model_only.yaml \
  --style-predictions outputs/style-baseline/val_predictions.csv \
  --output-dir outputs/fusion-eval
```

The command expects `outputs/model-only` to hold a local 18-class checkpoint
and the configured `outputs/model-only/val.csv` to be the same held-out set
used for the Style-only CSV. If necessary, first generate the Style-only
predictions using `python scripts/style_baseline.py
--config configs/ablations/model_only.yaml`. Keep original checkpoint
and baseline predictions backed up before rerunning either training script.

Default BERT weights: `1.0 0.8 0.7 0.5 0.0`, with the remaining weight
applied to Style. Override using `--bert-weights 1.0 0.9 0.7 0.5 0.0`.
For each weight, final predictions are the argmax of the weighted average of
**normalized 18-class probabilities**, with class ordering from
`fhf.labels.MODEL_NAMES`.

Validation safeguards: exact matching sample IDs, unique IDs, matching
`prompt_id` and gold `model` labels, all 18 class probabilities present,
finite and normalized values, and consistency of the Style model's saved
`pred_model` with its probabilities. An error terminates the evaluation
rather than silently mixing different examples. This checks *alignment*, not
the history of how either model was trained.

Outputs under `outputs/fusion-eval`:

- `evaluation.json`: original BERT and Style scores, prediction agreement,
  both-wrong count/rate, complementary correct cases, and **oracle ensemble**
  (true-label-based diagnostic upper bound, not deployable).
- `fusion_grid.csv`: accuracy and 18-class BAcc for each fixed weight.
- `val_predictions.csv`: per-example true label, individual predictions,
  correct/incorrect flags, and all fusion-grid predictions.

Selecting the best weight from the same validation data is exploratory:
validate the chosen weight on an independent held-out set before claiming
a generalization gain. Fusion may not improve on either standalone model.


## Dual-path style-aware FHF with residual hierarchy (experimental)

New opt-in architecture in `configs/ablations/dual_path_style.yaml`.
This is **end-to-end training**, not the previous frozen BERT + TF-IDF
probability average. Original FHF config files remain compatible and unchanged
unless the new YAML is selected.

Architecture:

1. Context branch: ModernBERT-large + LoRA with learned padding-aware
   attention pooling instead of masked mean pooling.
2. Style branch: fit-free hashed character 3–5-gram counts plus 24 lexical,
   punctuation, length, repetition, and formatting statistics extracted from
   each **original** text. The small style MLP learns on training data only.
   The hashed representation is distinct from the fitted TF-IDF baseline.
3. Gated style fusion: learned residual style correction to context embeddings.
4. Original **three heads** remain (Family 6-way, Model 18-way, Projection 128-d).
   The Model Head produces within-family scores, while a new independent
   **Flat 18-way Head** learns absolute model scores.
5. Residual probability fusion combines hierarchical and flat predictions:
   `P(final) = (1-flat_weight)*P(hierarchical) + flat_weight*P(flat)`.
   Both the fused prediction and Flat Head have separate supervised losses.

Run:

```bash
git pull
python scripts/train.py --config configs/ablations/dual_path_style.yaml
```

Separate experiment output: `outputs/dual-path-style/`. The new YAML uses
15 epochs, cosine LR, 4-epoch early stopping, the same training/validation
prompt-disjoint splitting and seed 42 as `base.yaml`, and HotFlip disabled.
The model logs `val/balanced_accuracy`, `val/oracle_within_family_accuracy`,
`val/loss_flat`, `val/loss_hierarchical`, and `val/style_gate_mean`.

To evaluate the best saved checkpoint without retraining:

```bash
python scripts/evaluate.py --config configs/ablations/dual_path_style.yaml
```

To generate a competition submission, point `scripts/predict.py` at the
saved dual-path checkpoint, or use the same config and override
`paths.test_csv` / `paths.submission_parquet` as needed.
Original raw text must be available to regenerate the hashed style features.

The checkpoint metadata stores `pooling`, `style`, and
`residual_fusion` settings, and its `heads.pt` stores new Attention Pool,
Style Fusion and Flat Head parameters. Older checkpoints load with the legacy
mean-pooling path. A style-enabled checkpoint requires style features during
inference, which the standard TextDataset + prediction scripts now supply.
HotFlip with the style branch is explicitly blocked because token-only
perturbations would leave the raw-text style features out of sync.

**Research caveat:** this is an unvalidated structural hypothesis. Do not
assume it will achieve BAcc 0.745, and keep the existing Flat and Hierarchical
runs as baselines. This experiment retains the previous contrastive loss and
sampler; prompt-aware contrastive training is *not* included here.

## Prompt-aware contrastive dual-path experiment

`configs/ablations/dual_path_prompt_aware.yaml` extends the
`dual_path_style.yaml` configuration. It keeps ModernBERT, Attention
Pooling, the 4096-hash Style Branch, Gated Fusion, independent Flat Head,
hierarchical and projection heads, R-Drop, the same training budget, and
the same `intra_contrastive.weight: 0.25`. Only the **contrastive pairing
strategy** and additional same-prompt hard-negative weight change.

```bash
git pull
python scripts/train.py --config configs/ablations/dual_path_prompt_aware.yaml
```

Output: `outputs/dual-path-prompt-aware/`. This is a new training run,
**not** a continuation of the 0.378 dual-path checkpoint. The existing
dual-path checkpoint remains intact.

With the prompt-aware switch:
- Positive pairs share the exact generator model but have **different
  prompt IDs**, reducing reward for simply matching a topic.
- Strong hard negatives are answers to the **same prompt from different
  generator models** (factor `same_prompt_weight: 3.0`).
- Other different-model negatives remain, with same-family negatives
  also weighted by `hard_weight: 2.0`.
- Same-model, same-prompt duplicates are excluded from the contrastive
  denominator instead of being treated as negatives.
- `PromptAwareFamilyBatchSampler` retains the old model-balanced choice
  of family/model slots but fills them from aligned prompt IDs, sharing
  an anchor prompt where possible while selecting different prompts for
  same-model positives.
- `prompt_id` stays in **train data metadata only**. It is removed before
  the Transformer forward pass and never supplied during validation,
  standard inference or Kaggle submission.
- W&B logs `train/prompt_positive_pairs` and
  `train/prompt_hard_negative_pairs` per logged optimizer step to
  verify pair availability.

A prompt-disjoint train/validation split is still used, ensuring no
validation prompt is present in training. The sampler checks that each
training model has responses to at least two distinct prompts and that at
least one prompt is shared across different models. If the required
structure is missing, it fails rather than silently treating arbitrary
pairs as matched. For partially overlapping prompt collections, not
every batch is guaranteed to include a same-prompt negative; inspect
the W&B pair-count metrics.

Run the previous `dual_path_style.yaml` command as the control. Any
improvement should be verified over multiple seeds, not attributed to
prompt-aware loss alone on one prompt split. No performance gain is guaranteed.

## Train-only external dataset augmentation (NTL same-prompt v2)

The **opt-in** experiment `configs/ablations/dual_path_prompt_aware_ext.yaml`
keeps the Dual-Path + Prompt-aware model/loss settings unchanged and trains
from the existing ~0.398 experiment's **frozen** split, adding validated external
generations to its TRAIN set only.

### Prepare external data

Download `arxitekton/ntl-claude-gen-v2` from Kaggle and extract the
**`passages.csv` and `prompts.csv` pair** under `data/external/ntl_v2/`.
The loader automatically detects this exact filename pair and performs
a validated many-to-one JOIN on the response table's prompt foreign key
(e.g. `prompt_id`) and the prompt table's unique ID (e.g. `id`).
**Never concatenate these files row-by-row.** It fails with both column lists
and a mapping hint if it cannot prove that every response matches exactly
one prompt. For non-standard schemas set:
`external_data.join.passages_key`, `external_data.join.prompts_key`
and, if needed, `external_data.join.prompt_text` in the YAML.
The actual schema was not supplied, so these checks are deliberate.

A single CSV/Parquet/JSONL file containing the needed joined columns
also remains supported. The local `data/` tree is gitignored, so
downloaded samples are not pushed to GitHub. To choose a different file
use `--set paths.external_source_path=data/your-file.csv`.

The Kaggle dataset's real column schema has **not** been verified by this
repository. The loader detects unambiguous candidate columns (`text`,
`response`, `completion`, `model`, `model_name`, `prompt_id`,
`prompt`, etc.) and rejects missing or ambiguous mappings rather than
guessing. An external prompt text field is preferred for identifying matched
same-prompt generations. `prompt_id` alone is accepted only when it really
identifies a shared question. Use `external_data.columns` to specify actual
column names if auto-detection cannot determine them uniquely.

Only exact names in the official 18-class label vocabulary are accepted.
Other labels are reported and excluded. **Never map one model generation to a
different version because its name looks similar.** For a verified spelling-only
alias, configure `external_data.label_map` explicitly.

### Run

Keep these *existing* files from the original Prompt-aware run unchanged:

```text
outputs/dual-path-prompt-aware/train_split.csv
outputs/dual-path-prompt-aware/val.csv
```

Then run a data-only preparation and quality report **before GPU training**:

```bash
git pull
python scripts/prepare_external_data.py \
  --config configs/ablations/dual_path_prompt_aware_ext.yaml
```

The check writes `outputs/dual-path-prompt-aware-ntl-v2/external_clean.csv`
and `external_audit.json` plus the merged `train_split.csv` and unchanged
`val.csv` under that new experiment folder. It prints model counts and each
filter's exclusion counts. Review unknown model labels and rejected samples
before training.

When the report is satisfactory, run the **new** experiment:

```bash
python scripts/train.py --config configs/ablations/dual_path_prompt_aware_ext.yaml
```

This repeats the same checks and training merge before loading ModernBERT,
then saves a new checkpoint and report in
`outputs/dual-path-prompt-aware-ntl-v2/`.
To evaluate:

```bash
python scripts/evaluate.py --config configs/ablations/dual_path_prompt_aware_ext.yaml
```

### Safety and correctness rules

- Base Train and Val are read from the already-saved original files, not
  re-split after augmentation. Default expected counts are **2,159** and
  **540**; a mismatch stops the new experiment.
- External IDs and prompt identifiers receive `ntl_v2_` prefixes. When
  prompt text is available, normalized prompt text is SHA-256 hashed so
  different responses to the exact same question share a prompt group.
- Exact normalized response duplicates are removed against official Train
  and Val, and internally. Conflicting output labels on identical normalized
  responses cause the conflicting external examples to be removed.
- Each accepted model label must match an exact competition target. Model
  names, provenance, and label exclusions appear in `external_audit.json`.
- If no matching 18-class label or no usable shared prompt groups remain,
  stop before spending GPU time. The external raw format is not silently
  assumed to be compatible.
- Prompt-disjoint validation stays fixed. The content-level leakage checks
  catch exact normalized response overlap, **not semantic near duplicates**.
  Since original official prompts are anonymized, exact prompt equivalence
  with hidden official prompts cannot be guaranteed; manually review source
  provenance and any near-duplicate concerns.

This uses the **same model initialization and hyperparameters**, with no
automatic continuation from the 0.398 checkpoint. Keep the old run for
a comparable ablation. Do not interpret an improvement as causal evidence
without verifying external label authenticity, domain shift and repeated seeds.

## Within-family prompt-aware contrastive ablation

Run `configs/ablations/dual_path_within_family.yaml` to keep the exact
Dual-Path model (ModernBERT + style fusion, attention pooling, heads, LoRA,
R-Drop), official data split, optimizer, epochs, batch sizes, and
`intra_contrastive.weight=0.25`, changing **only** the contrastive loss
from `prompt_aware` to `within_family`.

```bash
git pull
python scripts/train.py --config configs/ablations/dual_path_within_family.yaml
python scripts/evaluate.py --config configs/ablations/dual_path_within_family.yaml
```

Output: `outputs/dual-path-within-family/`; previous ~0.398 checkpoint
remains untouched. W&B run: `dual-path-within-family`.

Within-family learning on the existing **fused Projection Head**:
- Positive: same exact generator, different prompt.
- Negative: different generator **in the same family**.
- A negative answering the same prompt gets weight 3.0.
- Examples from **other families**, plus same-model/same-prompt repeats,
  are excluded from this contrastive denominator.
- Anchors require both a positive AND a within-family negative; singleton
  families (e.g. Qwen, Moonshot AI) receive no contrastive gradient but
  still receive classification/family/R-Drop learning.
- Keep the original `PromptAwareFamilyBatchSampler` unchanged so the
  experiment tests the loss definition rather than also changing batches.
- W&B additional metrics: `train/within_family_valid_anchors` and
  `train/within_family_negative_pairs`. If few anchors are eligible,
  the effective contribution of the contrastive term will be small.

**No perturbation and no ground-truth family input to the model:** family
labels are used only to choose pairs for the training loss. This is
**not** a new style-only encoder or a family-conditioned inference head;
the fused projection remains as in the baseline, so the experiment cleanly
isolates the family-restricted contrastive objective.

Compare BAcc and family/within-family metrics to
`dual_path_prompt_aware.yaml` on the exact same prompt-disjoint validation
split. The model is retrained from initialization; no accuracy improvement
is assumed.

## C: Within-family Contrastive + Style-space perturbation

Run both together, without modifying B's saved checkpoint:

```bash
git pull
python scripts/train.py --config configs/ablations/dual_path_within_family_style_perturb.yaml
python scripts/evaluate.py --config configs/ablations/dual_path_within_family_style_perturb.yaml
```

C keeps B's exact model, sampler, within-family contrastive loss, data and
hyperparameters, adding a **second noisy style forward** during training.
Gaussian noise N(0, 0.05²) is added **after the Style Encoder and before
the Gated Fusion**, not to text or the pretrained Encoder tokens.
KL(clean detached || noisy) receives weight 0.10, logged as
`train/loss_style_perturbation`. Existing R-Drop remains separate. Noise
is disabled in evaluation/inference. The extra pass increases training cost.
Use the same prompt-disjoint validation and compare C to B and baseline.

## E1: 100 external examples per target model (combined Open Pool + NTL v2)

```bash
git pull
python scripts/select_open_pool_e1.py
python scripts/prepare_external_data.py --config configs/ablations/within_family_style_perturb_openpool_e1.yaml
python scripts/train.py --config configs/ablations/within_family_style_perturb_openpool_e1.yaml
python scripts/evaluate.py --config configs/ablations/within_family_style_perturb_openpool_e1.yaml
```

The selection phase audits BOTH downloaded corpora (211,369 pool rows + 3,322
v2 rows), removes exact normalized overlaps including collisions with frozen
official Train/Val, rejects known proxy/routing flags and unverified risk
metadata, retains responses 40-250 words, and picks **100 per official model**
using seeded source-round-robin selection preferring distinct prompt groups.
The Open Pool duplicates 3,313 v2 responses; duplicate responses count once.
Because v2 has no risk flag metadata, its 9 unique responses are excluded by
default from the first conservative E1 run. Both datasets are still audited.
The data pool has not been independently authenticated; flagged-free does not
mean source/model routing verified. Check source/API provenance and licenses.

Inputs untouched; selector writes `data/external/e1/selected_100_per_model.csv`
and `outputs/openpool-e1/selection_report.json`. Preparer writes a frozen
540-row Val + 3,959-row Train under `outputs/openpool-e1/`. E1 retains
C's within-family contrastive, style perturbation, LoRA and optimizer, and
uses a full-coverage sampler and `steps_per_epoch: null`, ensuring all
3,959 training rows appear once per epoch. W&B run `c-openpool-e1-100`.

**Important:** Prompt keys are namespaced by corpus; semantic overlap with
anonymized validation prompts cannot be excluded mechanically.

## Large external full-coverage experiments

Original 3,541 Gemini 3.7 Flash rows are RAW; 3,495 remained after risk,
duplicate, and 40-250 word filters. The conservative first option is 3,495
external examples per model (62,910 external + 2,159 official = 65,069).
Each row is visited exactly once per epoch by `full_coverage`, not sampled
with replacement. Batch 32 (4 families × 2 models × 4 examples), accumulation 1.

```bash
python scripts/select_open_pool_full.py
python scripts/prepare_external_data.py --config configs/ablations/within_family_style_perturb_openpool_full_3495.yaml
python scripts/train.py --config configs/ablations/within_family_style_perturb_openpool_full_3495.yaml
```

Attempt 3,541/model only with a relaxed length constraint, keeping risk flags
and duplicates excluded. It may fail if not enough eligible rows remain:
```bash
python scripts/select_open_pool_full.py --per-model 3541 --min-words 1 --max-words 10000
python scripts/prepare_external_data.py --config configs/ablations/within_family_style_perturb_openpool_full_3541.yaml
python scripts/train.py --config configs/ablations/within_family_style_perturb_openpool_full_3541.yaml
```

To try batch 64, append `--set training.samples_per_model=8` to the
training command. The model does multiple forwards (style perturbation and
R-Drop); check memory. Increasing batch size changes optimization, not just
speed. No GPU training has been run by this repository change.
