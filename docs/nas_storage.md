# NAS storage and one-time migration

Default root: `/media/NAS/USERS/yeju/FHF` (override with `FHF_ROOT`).

| Directory | Contents |
|---|---|
| `code/` | Git checkout of the new branch |
| `csv/` | Original data tree, including external data and non-CSV source files |
| `csv/splits/` | Text-bearing datasets recovered from old `outputs/`, including frozen train/val |
| `checkpoints/<run>/` | v3 fold adapters, tokenizer, lexical model and checkpoint metadata |
| `results/<run>/` | v3 metrics, predictions, manifest |
| `cache/` | Hugging Face weights, Torch, W&B local files and temporary files |

The migration must run ON YOUR SERVER: this development environment cannot
access or delete files on your NAS. Stop training jobs before migration.
Original source code remains; the new code copy is inside the NAS.

## Migrate data and remove old generated artifacts

In the OLD checkout, first pull this branch and install its dependencies:

```bash
git switch experiments/fingerprint-v3-decoder
git pull --ff-only
python -m pip install -e '.[dev]'
python scripts/migrate_to_nas.py
```

This prints a plan including each preserved file and directories to delete.
To execute the requested move and cleanup:

```bash
python scripts/migrate_to_nas.py --apply
```

The default source is the checkout containing this script. The destination must
be separate from it and `NAS/code` must not already exist. If your old checkout
is already `/media/NAS/USERS/yeju/FHF`, DO NOT run this migration: it refuses nested
or identical locations. That layout needs an in-place plan based on the actual
directory inventory rather than moving the repository into itself.

Behavior:
1. Preserve ALL files under old `data/`, retaining subdirectory structure.
2. From old `outputs`, `checkpoints`, `results`, `wandb`, preserve CSV/TSV/Parquet/
   JSON/JSONL datasets that have a text/response/completion/prompt/passage field.
   These include original frozen train/val files. Inspect the preview: unusual
   dataset schemas outside `data/` must first be moved into `data/` yourself.
3. Copy to NAS and verify every file using SHA-256. Conflicting files abort;
   matching destination files may be reused. Code is cloned with no hardlinks.
4. Write `migration_manifest.json`, then DELETE the four old generated directories
   listed in the preview. Replace old `data/` with a link to the NAS copy.

No remote W&B runs are deleted. Existing NAS checkpoints/results are not deleted.
Files outside the four source generated directories are not deleted. The script
will not hunt for `.pt` files elsewhere. Do not run while training writes files.
Directory symlinks, dirty tracked code, or destination conflicts cause a stop.
Untracked source-code files are not cloned; commit wanted code before migrating.

## Run v3 on NAS

```bash
cd /media/NAS/USERS/yeju/FHF/code
python -m pip install -e '.[dev]'
python -m pip install -r requirements-v3.txt

CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v3.py train \
  --train /media/NAS/USERS/yeju/FHF/csv/splits/dual-path-prompt-aware/train_split.csv \
  --valid /media/NAS/USERS/yeju/FHF/csv/splits/dual-path-prompt-aware/val.csv \
  --out qwen4b-holdout --wandb
```

Default train input is `NAS/csv/train.csv`. Relative `--out` is a run name under
`NAS/results`; checkpoints go to `NAS/checkpoints/<run-name>`. All training caches
are set to NAS. Absolute `--out` and `--checkpoint-dir` explicitly override paths.
Use a unique run basename to avoid checkpoint collisions. Runtime errors do not
silently overwrite an existing run.

For all five folds:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/fingerprint_v3.py train \
  --out qwen4b-cv --wandb
```

The results manifest contains relative paths to sibling checkpoint directories.
For Kaggle, upload BOTH `results/<run>` and `checkpoints/<run>` with the same relative
layout, plus the local base model. A results directory alone is not a model bundle.
`predict` keeps explicit Kaggle inputs and `submission.parquet` defaults unchanged.

## Older scripts retained on this branch

Use the NAS runner for old main/v2 scripts so their working directory and caches
also stay on NAS:

```bash
python scripts/nas_run.py scripts/train.py --config configs/base.yaml
```

Compatibility links map `code/data -> csv` and `code/outputs -> results`.
Legacy scripts retain their original internal output structure, including any
checkpoints they place inside `outputs` (now NAS `results`). Only the new v3 runner
splits checkpoints/results into dedicated roots. Old configs referring to deleted
checkpoints must be retrained; original frozen splits now live under `csv/splits`.
Override their paths accordingly. Explicit absolute paths in custom configs always
remain user overrides; no code silently rewrites arbitrary external paths.
