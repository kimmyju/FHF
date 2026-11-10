# EOS + residual style ablation

Run from the repository: `bash scripts/run_v3_eos_style.sh`.
The launcher validates the three NAS CSV paths and the committed JSON config before launching training. Defaults: FHF_ROOT=/media/NAS/USERS/yeju/FHF, CUDA_VISIBLE_DEVICES=2. Set FHF_RUN_NAME to a NEW name for repeat runs. Existing results/checkpoints are never overwritten.

This v3 entry point reads JSON, not YAML. All settings are in configs/fingerprint_v3/qwen3_8b_eos_style.json. Qwen3-8B, seed 42, max length 768, external cap 3495 per class, external 2 epochs, official 4 epochs, microbatch 2, accumulation 16, LoRA rank 32/alpha 64. The learning rates, schedule, external filtering and lexical baseline remain unchanged. Compare neural BAcc to 0.5759259. No gain is guaranteed.

This run starts from Qwen3's pretrained base, NOT the 57.59% trained adapter. It preserves the EOS linear architecture, not the trained baseline weights. The extra branch is Linear(25,128), GELU, Dropout(0.1), Linear(128,18); its logits are added to the EOS linear logits. Its final layer is zeroed AFTER Hugging Face initializes missing weights and BEFORE PEFT wraps the model. On offline restore the saved PEFT head replaces that initialization. The full EOS head and residual branch are saved under score.modules_to_save and use head_learning_rate.

Feature version 1 uses only raw text: bounded lengths and sentence statistics, whitespace/case/digits, punctuation, list/heading/Markdown rates and lexical diversity. No labels, prompt IDs or source metadata enter the features. Formulas are fixed and bounded; no scaler is fit on validation. Features use the full raw passage, even if token input requires head-tail truncation. Train, validation, external and offline inference all call the same encode and collator. Feature order is defined by FEATURE_NAMES in src/fhf/v3_style.py; a different feature version must fail rather than silently load incompatible weights.

Tests: `python -m pytest tests/test_v3_decoder.py -q`. Tests cover bounded deterministic features, zero correction and gradients, small Qwen training with LoRA, exact probability restoration within tolerance, offline parquet inference and old EOS/EOS-mean modes.

Results: $FHF_ROOT/results/<run>/metrics.json and predictions.csv. Checkpoints and history: $FHF_ROOT/checkpoints/<run>/fold-0/. For offline inference use the existing fingerprint_v3.py predict command with this source version and the complete local base model; config reconstruction selects eos_style automatically. Quantization and multi-GPU inference are NOT added by this experiment. Real 8B quality, GPU memory and Kaggle runtime require separate measurement.
