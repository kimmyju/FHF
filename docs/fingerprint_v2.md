# Fingerprint V2 — multistage LLM fingerprint research

This branch keeps main unchanged and tests a new end-to-end approach. A 0.74+ BAcc is a goal, not a guaranteed outcome.

## Features

- External-only ModernBERT-large LoRA pretraining with full coverage and prompt-matched negatives.
- Official-only trainable LoRA + custom-head fine-tuning on the frozen 2,159-train / 540-validation split.
- OOF-confusion weighted within-provider negatives: computed from official TRAIN only.
- Character and word TF-IDF LogisticRegression plus separate provider and per-provider specialists.
- GroupKFold lexical OOF, optional five-fold neural OOF, leakage-checked probability ensemble.
- Optional independent DeBERTa-v3-large LoRA encoder; optional 20% external replay ablation.
- Official-vs-external domain-shift ROC-AUC diagnosis.

## Prerequisites

Install dependencies: pip install -e '.[dev]'. Put downloaded NTL v2 and Open Pool under data/external/ntl_v2/. Keep existing outputs/dual-path-prompt-aware/train_split.csv and val.csv. Do not upload dataset rows, checkpoints, secrets, or tokens to GitHub.

## One-command experiment

    git fetch origin
    git switch experiments/fingerprint-v2-multistage
    python scripts/fingerprint_v2_run.py --dry-run
    python scripts/fingerprint_v2_run.py --batch-size 32

The foreground stages run in order: select external data, 5-fold lexical OOF, OOF confusion weights, external pretraining, official fine-tuning, frozen validation ensemble. No fixed steps_per_epoch is used: each completed epoch visits every example once. GPU training runs only when you execute commands locally.

## Manual stages

    python scripts/select_open_pool_full.py
    python scripts/fingerprint_v2_domain.py
    python scripts/fingerprint_v2_classical.py
    python scripts/fingerprint_v2_hard_pairs.py
    python scripts/train.py --config configs/fingerprint_v2/pretrain.yaml
    python scripts/train.py --config configs/fingerprint_v2/finetune.yaml
    python scripts/evaluate.py --config configs/fingerprint_v2/finetune.yaml
    python scripts/fingerprint_v2_ensemble.py --mode validation --checkpoint outputs/fingerprint-v2/finetune/checkpoint

## Optional experiments

20% replay ablation (540 curated external added to official):

    python scripts/fingerprint_v2_run.py --stages replay_select replay
    python scripts/evaluate.py --config configs/fingerprint_v2/replay.yaml

Independent DeBERTa encoder:

    python scripts/train.py --config configs/fingerprint_v2/deberta.yaml
    python scripts/fingerprint_v2_ensemble.py --mode validation --checkpoint outputs/fingerprint-v2/finetune/checkpoint --checkpoint outputs/fingerprint-v2/deberta/checkpoint

Neural GroupKFold OOF: this costs five independent neural training runs.

    python scripts/fingerprint_v2_neural_oof.py --mode prepare
    for i in 0 1 2 3 4; do python scripts/fingerprint_v2_neural_oof.py --mode train --fold "$i"; done
    python scripts/fingerprint_v2_neural_oof.py --mode collect
    python scripts/fingerprint_v2_ensemble.py --mode validation --checkpoint outputs/fingerprint-v2/finetune/checkpoint --neural-weight auto --neural-oof outputs/fingerprint-v2/neural-oof/neural_oof_predictions.csv

OOF neural folds disable global hard-pair weights because those weights otherwise contain information from each fold's holdout. Mixture weights are selected on TRAIN OOF; the score after tuning on the same OOF is optimistic and is not an unbiased evaluation. Frozen official validation is evaluation only.

## Kaggle submission

    python scripts/fingerprint_v2_ensemble.py --mode test --checkpoint outputs/fingerprint-v2/finetune/checkpoint

The output is outputs/fingerprint-v2/ensemble/submission.parquet with only id,model. Nothing reads ground-truth family or labels at test time.

## Validity caveats

- Model labels and generation routes in the external pool are not independently verified.
- Exact response dedup and risk flags do not rule out semantic duplicates or original-prompt leakage.
- Different encoders and larger batches change compute and optimization, not just data volume.
- Report BAcc, Oracle within-family accuracy, per-model recall, family accuracy, domain AUC and error complementarity.
- Validate all improvements on the frozen 540 examples; no claim of 74% without actual scoring.

## Diagnosing GLM-vs-Fable confusion (no GPU)

From the repository root:

    git pull
    python scripts/fingerprint_v2_source_audit.py

Outputs under outputs/fingerprint-v2/source-audit/:

- summary.json: source concentration, exact cross-label conflicts, official response overlap, raw-pool routing metadata flags if available.
- source_by_model.csv: all 18 labels by external source dataset.
- style_comparison.csv: length, punctuation, headings, paragraph counts for official vs external GLM/Fable.
- focus_rows.csv: selected-row IDs, source_dataset, normalized response SHA-256 and raw-metadata checks; **no response texts by default**. Use --include-text only for local manual review.

This is a **read-only heuristic audit**. It cannot prove which API actually generated an example; verify source logs and routing externally. The audit never changes official labels, datasets or checkpoints.

## Proper post-hoc calibration

The saved validation.csv contains 0.5185 BAcc and is **not** used for fitting or selecting class biases. Fitting the current 70% neural / 30% lexical ensemble requires **neural TRAIN OOF predictions**, because lexical-only OOF does not represent the trained neural classifier.

    python scripts/fingerprint_v2_neural_oof.py --mode prepare
    for i in 0 1 2 3 4; do
      python scripts/fingerprint_v2_neural_oof.py --mode train --fold "$i"
    done
    python scripts/fingerprint_v2_neural_oof.py --mode collect
    python scripts/fingerprint_v2_calibration.py --mode fit --neural-weight 0.7
    python scripts/fingerprint_v2_calibration.py --mode apply

Calibration fits a bounded, regularized per-class logit bias on the OOF predictions; the script uses a second prompt-group cross-fit to compare uncalibrated vs calibrated TRAIN OOF without fitting on those fold targets. Both scores remain *training-distribution diagnostics*, not public/private leaderboard promises.

The fitted offsets are stored in outputs/fingerprint-v2/calibration/fit.json. Comparison of saved ensemble validation predictions goes to outputs/fingerprint-v2/calibration/validation_calibrated.csv and .json **without overwriting baseline files**.

For newly generated validation/test predictions, apply the matching calibration automatically:

    python scripts/fingerprint_v2_ensemble.py --mode validation --checkpoint outputs/fingerprint-v2/finetune/checkpoint --bias-json outputs/fingerprint-v2/calibration/fit.json
    python scripts/fingerprint_v2_ensemble.py --mode test --checkpoint outputs/fingerprint-v2/finetune/checkpoint --bias-json outputs/fingerprint-v2/calibration/fit.json

The inference script rejects neural/lexical weight mismatches. If using multiple neural checkpoints (e.g. ModernBERT + DeBERTa), do not reuse biases fitted for a *single-neural* ensemble.

Quick lexical-only sanity check (not valid calibration for the 70% neural ensemble):

    python scripts/fingerprint_v2_calibration.py --mode fit --neural-weight 0 --bias-json outputs/fingerprint-v2/calibration/lexical_only.json

This trains a lexical-only bias for separate experiments. **Do not apply lexical-only biases to the existing neural-heavy ensemble.**
