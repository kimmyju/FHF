# FHF: Fine-grained Hierarchical Fingerprinting

A research implementation for [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). FHF combines a shared Transformer encoder with a hierarchical family/exact-model classifier, same-model supervised contrastive learning with hard within-family negatives, optional stochastic consistency, and an experimental gradient-guided HotFlip objective.

**Status:** Source implementation with unit tests. No pretrained checkpoint, competition dataset, or validated leaderboard score is included. This approach has not been shown to beat a standard classifier. The repository's license applies to the code only.

## Method

Given a text passage, the encoder produces a representation z. A family head estimates P(family|text), and a conditional model head estimates P(model|family,text). Their product forms a normalized distribution over the exact 18 competition models.

```text
L = NLL_18 + alpha * CE_family + beta * SupCon_hard + gamma * KL_RDrop + delta * KL_HotFlip
```

SupCon uses same-version positives and upweights different-version negatives *within the same company*. R-Drop uses two stochastic passes of **unchanged text**. HotFlip is optional, replaces at most one eligible single-token discourse marker with another using a first-order gradient estimate, then penalizes inconsistency. HotFlip can modify stylistic evidence and is not guaranteed to preserve generator identity; treat this as an ablation, not reliable label-preserving augmentation.

## Structure

- `src/fhf/labels.py`: Exact 18 labels and six model providers.
- `src/fhf/data.py`: CSV loader, validation, group split, family-balanced batch sampler.
- `src/fhf/model.py`: Hierarchical Transformer and offline checkpoint saving/loading.
- `src/fhf/losses.py`: Classification, contrastive and KL losses.
- `src/fhf/hotflip.py`: Constrained gradient-based token substitutions.
- `src/fhf/engine.py`: Training, checkpoint selection, inference.
- `src/fhf/metrics.py`: Balanced accuracy, per-class recall and confusion matrix.
- `scripts/train.py`: Fine-tuning on labeled competition CSV.
- `scripts/pretrain_external.py`: Optional real-generator-labeled external contrastive training.
- `scripts/evaluate.py`: Labeled holdout evaluation.
- `scripts/predict.py`: Parquet predictions.
- `scripts/kaggle_submit.py`: Submission entrypoint.
- `notebooks/kaggle_submit.ipynb`: Kaggle template.
- `tests/`: Offline unit tests.

## Install

Python 3.10+ recommended. Install a compatible CUDA PyTorch build if necessary:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Data

Download competition `train.csv` yourself and place it under `data/train.csv`. Expected columns are `id`, `text`, `model`, `prompt_id`. For evaluation, **split by prompt_id** to avoid letting the same prompt appear in both train and validation. Test data needs `id,text`, and is hidden until Kaggle evaluates the submitted notebook. Keep license terms for any external data.

## Training

```bash
python scripts/train.py \
  --train-csv data/train.csv \
  --backbone microsoft/deberta-v3-base \
  --output-dir outputs/fhf-base --epochs 5 \
  --family-weight 0.25 --contrastive-weight 0.25 \
  --consistency-weight 0.05 --batch-families 3 \
  --models-per-family 2 --samples-per-model 2
```

For baseline ablations set `--family-weight 0 --contrastive-weight 0 --consistency-weight 0`. To test HotFlip, add `--hotflip-weight 0.05 --hotflip-max-examples 2`. HotFlip is disabled by default and increases training cost substantially. The contrastive sampler may sample minority models with replacement. For robust model selection, repeat group-disjoint splits and seeds. The `microsoft/deberta-v3-base` backbone is an example, not a benchmark-validated optimal choice.

## External representation pretraining (optional)

CSV format: `text,model,provider` where `model` is the **actual generation model name** in that dataset, not one of the 18 Kaggle labels unless it genuinely matches. `provider` is recommended for within-family hard negatives. Historical model names MUST NOT be relabeled as current versions.

```bash
python scripts/pretrain_external.py \
  --csv data/external.csv \
  --backbone microsoft/deberta-v3-base \
  --output-dir outputs/external --epochs 2

python scripts/train.py \
  --train-csv data/train.csv \
  --backbone outputs/external/encoder \
  --tokenizer outputs/external/tokenizer \
  --output-dir outputs/fhf-external
```

This is optional and requires sufficient same-model positives in each minibatch. External pretraining currently uses random batches; for sparse labels, a family-aware balanced sampler is a logical improvement. Sources to inspect independently include Chatbot Arena, UltraFeedback and SemEval Task 8. Review dataset licenses and competition rules.

## Evaluate and predict

```bash
python scripts/evaluate.py \
  --checkpoint outputs/fhf-base \
  --data outputs/fhf-base/val.csv \
  --output outputs/fhf-base/report.json

python scripts/predict.py \
  --checkpoint outputs/fhf-base \
  --test-csv data/test.csv \
  --output submission.parquet
```

`submission.parquet` has exactly two columns (`id`, `model`), one unique ID per row, and exactly one of the official 18 labels.

## Kaggle code competition submission

1. Train offline (e.g. research GPU server) and evaluate using prompt-disjoint validation.
2. Upload the **full checkpoint folder** (`encoder/`, `tokenizer/`, `heads.pt`, `metadata.json`) as a **private** Kaggle dataset, and attach this source package as another dataset.
3. In the Kaggle notebook adapt checkpoint/source mount locations from `notebooks/kaggle_submit.ipynb`.
4. Ensure dependencies are already installed or attach an offline wheel bundle; Internet must be **OFF**. `torch`, `transformers`, `pandas`, `pyarrow` compatibility must be verified.
5. Commit/submit. Kaggle reruns the notebook on hidden ~4,000-row test data. CPU/GPU notebook runtime must be <= 9 hours including startup and model loading.

### Important limitations

- The competition's ~2,699-row training set is small and may cause fingerprint overfitting.
- Some validation splits may omit rare labels; inspect both observed-class Balanced Accuracy and 18-class diagnostic recall.
- External historical LLM data can hurt, not help, fine-grained version attribution.
- Gradient sensitivity need not correspond to a genuine linguistic fingerprint. Never change the hidden competition text before scoring.
- Prior Kaggle/competition code and datasets are NOT bundled; this is a code-only repository.
- Consult official Kaggle Rules before using paid generators, non-public weights or restricted datasets.

## Attribution

Inspired conceptually by the split retrieval/selection optimization and token-level HotFlip technique described in *Prompt Injection Attack to Tool Selection in LLM Agents* (ToolHijacker, NDSS 2026). FHF targets benign authorship attribution, not a prompt-injection attack. HotFlip, supervised contrastive learning, and hierarchical classifiers are pre-existing methods; demonstrating novelty would require careful experiments.

## License

MIT license applies to FHF source only. External models and datasets retain their own terms.
