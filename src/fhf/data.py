"""Data ingestion, prompt-disjoint validation, tokenization and family-balanced batches."""
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, Sampler
from .labels import MODEL_TO_ID, MODEL_FAMILY_IDS, check_labels
from .style import StyleFeatureExtractor


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
    def __init__(self, df, tokenizer, max_length=384, style_config=None):
        self.df, self.tokenizer, self.max_length = df.reset_index(drop=True), tokenizer, max_length
        self.labels = [MODEL_TO_ID[x] for x in df.model] if 'model' in df else None
        style_config = style_config or {}
        self.style_features = None
        if style_config.get("enabled", False):
            extractor = StyleFeatureExtractor(style_config.get("hash_dim", 4096))
            # Materialize once, not repeatedly for each epoch.
            self.style_features = torch.from_numpy(
                extractor.transform(self.df.text.astype(str).tolist())
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = self.tokenizer(str(self.df.iloc[i].text), truncation=True, max_length=self.max_length)
        if self.style_features is not None:
            item['style_features'] = self.style_features[i]
        if self.labels is not None:
            item['labels'] = self.labels[i]
            item['families'] = MODEL_FAMILY_IDS[self.labels[i]]
        return item


def make_collator(tokenizer):
    """Dynamically pad token sequences while preserving labels."""
    def collate(items):
        labels = [x.pop('labels') for x in items] if 'labels' in items[0] else None
        families = [x.pop('families') for x in items] if 'families' in items[0] else None
        styles = [x.pop('style_features') for x in items] if 'style_features' in items[0] else None
        batch = tokenizer.pad(items, padding=True, return_tensors='pt')
        if styles is not None:
            batch['style_features'] = torch.stack(styles)
        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.long)
            batch['families'] = torch.tensor(families, dtype=torch.long)
        return batch
    return collate


class FamilyBatchSampler(Sampler):
    """Nearly equal *model* exposure with contrastive-friendly family batches.

    Each batch has distinct families, up to models_per_family distinct models
    per family, and samples_per_model text examples per model slot. Families
    receive slots in proportion to their number of observed exact models,
    rather than equally. This avoids oversampling singleton-model families.
    """

    def __init__(self, labels, batch_families=3, models_per_family=2,
                 samples_per_model=2, steps_per_epoch=None, seed=42):
        self.bf, self.mpf, self.spm, self.seed = (
            int(batch_families), int(models_per_family),
            int(samples_per_model), int(seed))
        if min(self.bf, self.mpf, self.spm) < 1:
            raise ValueError("Sampler batch dimensions must be positive")
        self.epoch = 0
        self.by_family = {}
        for idx, model in enumerate(labels):
            model = int(model)
            family = MODEL_FAMILY_IDS[model]
            self.by_family.setdefault(family, {}).setdefault(model, []).append(idx)
        self.steps = (int(steps_per_epoch) if steps_per_epoch is not None else
                      max(1, math.ceil(len(labels) / (self.bf * self.mpf * self.spm))))
        if self.steps < 1:
            raise ValueError("steps_per_epoch must be positive")
        if len(self.by_family) < self.bf:
            raise ValueError("Not enough families to create a balanced batch")
        self.last_model_exposures = {}

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        """Change the draw sequence reproducibly between epochs."""
        if int(epoch) < 0:
            raise ValueError("Sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def _family_quotas(self, rng):
        """Apportion family appearances by model count, max once per batch.

        If one family dominates, its quota is capped at steps_per_epoch:
        exact per-model balance is then impossible with distinct families.
        """
        slots = self.steps * self.bf
        weights = {f: len(models) for f, models in self.by_family.items()}
        remaining = set(weights)
        quotas = {f: 0 for f in weights}
        while remaining:
            total_weight = sum(weights[f] for f in remaining)
            capped = [f for f in remaining
                      if slots * weights[f] > self.steps * total_weight]
            if not capped:
                break
            for f in capped:
                quotas[f] = self.steps
                slots -= self.steps
                remaining.remove(f)
        if remaining:
            denominator = sum(weights[f] for f in remaining)
            desired = {f: slots * weights[f] / denominator for f in remaining}
            for f in remaining:
                quotas[f] = math.floor(desired[f])
            leftover = slots - sum(quotas[f] for f in remaining)
            tie_breaks = {f: rng.random() for f in remaining}
            ranked = sorted(remaining, key=lambda f: (
                desired[f] - quotas[f], tie_breaks[f]), reverse=True)
            for f in ranked[:leftover]:
                quotas[f] += 1
        if sum(quotas.values()) != self.steps * self.bf or any(
                q > self.steps for q in quotas.values()):
            raise RuntimeError("Invalid model-balanced family quotas")
        return quotas

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        quotas = self._family_quotas(rng)
        remaining = quotas.copy()
        family_counts = {f: 0 for f in quotas}
        model_counts = {model: 0 for models in self.by_family.values()
                        for model in models}
        self.last_model_exposures = {model: 0 for model in model_counts}

        # Shuffled per-model pools cover examples before repeating them.
        pools = {}
        cursor = {}
        for models in self.by_family.values():
            for model, indices in models.items():
                pools[model] = rng.permutation(indices).tolist()
                cursor[model] = 0

        def draw_sample(model, used_in_batch):
            indices = self.by_family[MODEL_FAMILY_IDS[model]][model]
            if cursor[model] >= len(pools[model]):
                pools[model] = rng.permutation(indices).tolist()
                cursor[model] = 0
            # Avoid a duplicate within a batch when another candidate remains.
            if len(used_in_batch) < len(indices) and pools[model][cursor[model]] in used_in_batch:
                for pos in range(cursor[model] + 1, len(pools[model])):
                    if pools[model][pos] not in used_in_batch:
                        pools[model][cursor[model]], pools[model][pos] = (
                            pools[model][pos], pools[model][cursor[model]])
                        break
            sample = int(pools[model][cursor[model]])
            cursor[model] += 1
            used_in_batch.add(sample)
            return sample

        for step in range(self.steps):
            steps_left = self.steps - step
            # A quota equal to the remaining steps must appear in every batch.
            chosen = [f for f in remaining if remaining[f] == steps_left]
            candidates = [f for f in quotas if remaining[f] and f not in chosen]
            tie_breaks = {f: rng.random() for f in candidates}
            candidates.sort(key=lambda f: (
                quotas[f] * (step + 1) / self.steps - family_counts[f],
                tie_breaks[f]), reverse=True)
            chosen.extend(candidates[:self.bf - len(chosen)])
            if len(chosen) != self.bf:
                raise RuntimeError("Insufficient distinct families for balanced batch")
            batch = []
            for family in chosen:
                remaining[family] -= 1
                family_counts[family] += 1
                models = list(self.by_family[family])
                selected = []
                for _ in range(self.mpf):
                    available = [m for m in models if m not in selected] or models
                    least = min(model_counts[m] for m in available)
                    tied = [m for m in available if model_counts[m] == least]
                    model = int(rng.choice(tied))
                    selected.append(model)
                    model_counts[model] += 1
                    self.last_model_exposures[model] += self.spm
                used_in_batch = set()
                for model in selected:
                    batch.extend(draw_sample(model, used_in_batch)
                                 for _ in range(self.spm))
            rng.shuffle(batch)
            yield batch
