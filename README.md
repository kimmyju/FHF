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
