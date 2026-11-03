"""Data ingestion, prompt-disjoint validation, tokenization and family-balanced batches."""
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, Sampler
from .labels import MODEL_TO_ID, MODEL_FAMILY_IDS, check_labels


def load_csv(path, labeled=True):
    """Validate IDs, nonempty text, and exact labels before training/inference."""
    df = pd.read_csv(path)
    cols = {'id', 'text'} | ({'model'} if labeled else set())
    if not cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {cols - set(df.columns)}")
    if df[list(cols)].isna().any().any() or not df.text.astype(str).str.strip().all():
        raise ValueError('Null or empty required data')
    if df.id.duplicated().any():
        raise ValueError('Duplicate IDs')
    if labeled:
        check_labels(df.model.tolist())
    return df.reset_index(drop=True)


def grouped_split(df, validation_fraction=0.2, seed=42):
    """Ensure all samples sharing prompt_id remain in only one split."""
    if 'prompt_id' not in df.columns or df.prompt_id.isna().any():
        raise ValueError('prompt_id is required for prompt-disjoint validation')
    if df.prompt_id.nunique() < 2:
        raise ValueError('Need at least 2 distinct prompt IDs')
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df.prompt_id))
    train, valid = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
    assert set(train.prompt_id).isdisjoint(set(valid.prompt_id))
    return train, valid


class TextDataset(Dataset):
    """Tokenize paragraphs lazily; optionally attach exact model and family IDs."""
    def __init__(self, df, tokenizer, max_length=384):
        self.df, self.tokenizer, self.max_length = df.reset_index(drop=True), tokenizer, max_length
        self.labels = [MODEL_TO_ID[x] for x in df.model] if 'model' in df else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = self.tokenizer(str(self.df.iloc[i].text), truncation=True, max_length=self.max_length)
        if self.labels is not None:
            item['labels'] = self.labels[i]
            item['families'] = MODEL_FAMILY_IDS[self.labels[i]]
        return item


def make_collator(tokenizer):
    """Dynamically pad token sequences while preserving labels."""
    def collate(items):
        labels = [x.pop('labels') for x in items] if 'labels' in items[0] else None
        families = [x.pop('families') for x in items] if 'families' in items[0] else None
        batch = tokenizer.pad(items, padding=True, return_tensors='pt')
        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.long)
            batch['families'] = torch.tensor(families, dtype=torch.long)
        return batch
    return collate


class FamilyBatchSampler(Sampler):
    """Sample multiple labels per family and multiple examples per model for contrastive learning."""
    def __init__(self, labels, batch_families=3, models_per_family=2, samples_per_model=2,
                 steps_per_epoch=None, seed=42):
        self.bf, self.mpf, self.spm, self.seed = batch_families, models_per_family, samples_per_model, seed
        self.by_family = {}
        for idx, model in enumerate(labels):
            family = MODEL_FAMILY_IDS[model]
            self.by_family.setdefault(family, {}).setdefault(model, []).append(idx)
        self.steps = steps_per_epoch or max(1, math.ceil(len(labels)/(batch_families*models_per_family*samples_per_model)))
        if len(self.by_family) < self.bf:
            raise ValueError('Not enough families to create a balanced batch')

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        families = list(self.by_family)
        for _ in range(self.steps):
            chosen_families = rng.choice(families, self.bf, replace=False)
            batch = []
            for family in chosen_families:
                models = list(self.by_family[int(family)])
                selection = rng.choice(models, self.mpf, replace=len(models) < self.mpf)
                for model in selection:
                    indices = self.by_family[int(family)][int(model)]
                    batch.extend(int(i) for i in rng.choice(indices, self.spm, replace=len(indices)<self.spm))
            rng.shuffle(batch)
            yield batch
