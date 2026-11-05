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

## Verify the 434-row GLM prompt-group suspicion

    git pull
    python scripts/fingerprint_v2_verify.py

No GPU or model retraining is needed. This joins the selected responses back to the original Open Pool by normalized response SHA-256, exact model, and source name. It then checks whether the large selected prompt group contains multiple original raw prompt groups and, **if present**, multiple actual prompt texts. Results go under outputs/fingerprint-v2/prompt-source-verification/large_prompt_groups.csv and raw_prompt_group_matches.csv.

Interpretation:

- MULTIPLE_ORIGINAL_PROMPTS or MULTIPLE_RAW_PROMPT_GROUPS: grouped samples are not all the same recorded prompt; do not use the collapsed key for same-prompt contrastive training.
- UNVERIFIED_ORIGINAL_PROMPT_TEXT_UNAVAILABLE: source has insufficient prompt metadata; the 434 identical IDs **do not** prove 434 identical prompts.
- SINGLE_RECORDED_PROMPT_NOT_API_VERIFIED: appears internally consistent, but original API requests still need external confirmation.

For an independent GLM/Fable **three-class** source-shift experiment, run:

    python scripts/fingerprint_v2_verify.py --benchmark

The script compares a prompt-group-disjoint random split to leave-one-source-out tests, using a lightweight **external-data-only** char-TFIDF classifier (not the trained ModernBERT). It optionally reports external-trained performance on official TRAIN, never on official Validation. The benchmark source test is not an estimate of final 18-class BAcc; interpret a lower unseen-source score as source/domain shift evidence, not proof of mislabeled examples. Source-specific prompt topics, collection settings, and routing are confounded.

## Non-destructive unverified prompt-group ablation

The raw Open Pool has no verifiable original prompt text for large groups: a single selected prompt ID contains 1,820 Gemini samples, and another contains 434 GLM samples. Neither is proof of a collision. Do not relabel, delete or overwrite samples based on size alone.

The optional ablation keeps **all samples and labels**, and keeps the original prompt IDs for train/validation grouping. Only the prompt identities seen by the contrastive loss and prompt-matched sampler are changed: groups with **more than 32 rows** are replaced by unique IDs *per row*. The number of affected rows is logged in each experiment's prompt_safety_report.json. This removes same-prompt negatives for the unverified large groups, while retaining ordinary class learning and all other samples.

Run the two isolated experiment configs with new, separate checkpoints:

    python scripts/train.py --config configs/fingerprint_v2/pretrain_prompt_isolated.yaml
    python scripts/train.py --config configs/fingerprint_v2/finetune_prompt_isolated.yaml
    python scripts/fingerprint_v2_ensemble.py --mode validation --checkpoint outputs/fingerprint-v2/finetune-isolated/checkpoint --output outputs/fingerprint-v2/ensemble/validation_prompt_isolated.csv

Compare validation_prompt_isolated.json with the baseline validation.json (0.5185 BAcc). This uses the original lexical model and fixed neural weight 0.7. Only one variable changes: prompt matching for groups >32 in external pretraining. You must NOT run the default multi-stage driver to evaluate this ablation, because it would regenerate/overwrite the baseline outputs. We recommend repeating multiple seeds before drawing conclusions.

## Diagnose GLM 5.3 and Flash misclassification as Claude Fable

On the isolated 18-model neural checkpoint, 31/60 real GLM validation rows predict Claude Fable, while only 3/60 are exactly correct. This section separates possible causes without changing training labels or fitting anything to frozen validation.

### Step 1. Check trained heads (single GPU inference; no training)

    CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v2_glm_heads.py \\
      --checkpoint outputs/fingerprint-v2/finetune-isolated/checkpoint

Outputs:

- outputs/fingerprint-v2/glm-root-cause/heads/head_diagnosis.json: GLM and Fable true-label counts, family-head ZAI versus Anthropic, flat and hierarchical predictions, oracle within-ZAI correctness.
- outputs/fingerprint-v2/glm-root-cause/heads/focus_head_probabilities.csv: sample-level scores, the top-two final confidence margin, and separate branch probabilities. This CSV contains IDs, labels and probabilities, **not raw text**.

Interpretations:

- Many GLM samples already classified as Anthropic by family_head: investigate provider-classification head / domain shift.
- family_head correctly chooses ZAI but flat_pred_model chooses Fable: investigate flat branch and fusion; changing the family head alone is unlikely to fix it.
- family_head correctly chooses ZAI, but oracle within-ZAI GLM-vs-Flash accuracy is also low: provider discrimination is separate from GLM version discrimination.
- If GLM has a small top1-vs-top2 probability gap, bias/calibration might help, but only if trained using **matched TRAIN OOF** predictions; do not tune offsets on frozen validation.

### Step 2. Official-TRAIN-only 3-class distinguishability probe

Run the frozen **external-only** pretrained ModernBERT checkpoint so the learned representation has never seen official TRAIN:

    CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v2_glm_probe.py \\
      --checkpoint outputs/fingerprint-v2/pretrain-isolated/checkpoint

For a quick CPU-only lexical baseline, without any checkpoint:

    python scripts/fingerprint_v2_glm_probe.py --checkpoint none \\
      --out outputs/fingerprint-v2/glm-root-cause/probes-lexical

Outputs:

- outputs/fingerprint-v2/glm-root-cause/probes/probe_summary.json: 5-fold prompt-group-disjoint OOF BAcc and confusion matrices for char TF-IDF and frozen neural-feature linear probe.
- outputs/fingerprint-v2/glm-root-cause/probes/official_train_oof_predictions.csv: predictions by each diagnostic method, 359 official TRAIN rows only.

The neural feature is the **input to the model_head**, after the frozen encoder and optional style fusion; it is extracted using a forward hook, without retraining the checkpoint. A LogisticRegression linear probe is trained *inside each official TRAIN GroupKFold fold only*. Do not report a probe on official-fine-tuned features as an unbiased held-out score: the encoder would have seen the train data earlier.

Interpretation matrix:

- Char TF-IDF and frozen feature probe both close to 1/3: official TRAIN contains weak 3-way discriminative evidence under these methods, or provenance/label/prompt issues need scrutiny.
- Frozen feature probe clearly outperforms lexical probe: the pretrained representation contains useful separation; inspect 18-class heads/losses.
- Official-train probe substantially outperforms external 3-class benchmark: external training domain differs from official; source or prompt composition may matter.
- Strong pretrained probe but incorrect family/flat heads on frozen validation: targeted official-only head or fusion ablation is a plausible *future* training experiment, not a validated fix.

Do **not** edit training configs, class labels, or the official validation for these two diagnostics. An independent frozen validation is for final evaluation, not selecting the best of many posthoc configurations. Both scripts are diagnostic only, and baseline checkpoints remain unchanged.
