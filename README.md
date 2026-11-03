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
