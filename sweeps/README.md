# FHF W&B Sweeps

Sweep configurations live **only here**. Normal per-run configs remain in `configs/`.

| File | What it optimizes | Metric |
|---|---|---|
| `lora_bayes.yaml` | LoRA rank/alpha/dropout, LR, family and contrastive weights | `val/best_balanced_accuracy` |
| `loss_grid.yaml` | Family, contrastive and R-Drop loss contributions | `val/best_balanced_accuracy` |
| `model_only_bayes.yaml` | Flat 18-class head hyperparameters | `val/best_balanced_accuracy` |
| `family_only_bayes.yaml` | Six-provider head hyperparameters | `val/best_family_balanced_accuracy` |

## Start a sweep

1. Install dependencies with `pip install -e ".[dev]"`.
2. Configure W&B access **outside the repository**: `wandb login` or `export WANDB_API_KEY=...`.
3. Set the project `fhf-name-that-llm` and edit `entity` in the sweep YAML only if your organization requires it.
4. From the repository root execute:

```bash
wandb sweep sweeps/lora_bayes.yaml
wandb agent YOUR_ENTITY/fhf-name-that-llm/SWEEP_ID
```

Run only as many agents as your GPU resources support; each agent can start multiple trials. Stop with Ctrl+C. You can cap one agent to 10 runs with `wandb agent --count 10 YOUR_ENTITY/fhf-name-that-llm/SWEEP_ID`.

Sweep trials automatically use W&B online mode; normal runs default to offline mode. The launcher writes checkpoints, train/validation splits and metric reports to `outputs/sweeps/<run_id>/` so concurrent trials cannot overwrite each other. The base data path remains `paths.train_csv` from `configs/base.yaml`. Every run uses prompt-disjoint validation with the same seed unless the sweep explicitly varies it. The sweep objective is the **best held-out balanced accuracy reached during training**, logged as a scalar.

**Important:** These sweeps optimize the same validation set repeatedly, so the best trial may overfit model selection. Keep a separate truly held-out evaluation set for the final comparison and avoid tuning on Kaggle public leaderboard feedback. LoRA rank/alpha and loss weights here are exploratory, not guaranteed optimal.

No W&B API keys, user tokens or private dataset content are committed.
