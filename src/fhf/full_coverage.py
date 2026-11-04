"""Full-coverage batch sampler for prompt-aware attribution."""
from collections import deque
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from .labels import MODEL_FAMILY_IDS

class FullCoveragePromptFamilyBatchSampler(Sampler):
    """Use every input row exactly once each epoch, with model/family grouping."""
    def __init__(self, labels, prompt_ids, batch_families=4,
                 models_per_family=2, samples_per_model=4,
                 steps_per_epoch=None, seed=42):
        if steps_per_epoch is not None:
            raise ValueError("full_coverage requires steps_per_epoch=null")
        if not labels or len(labels) != len(prompt_ids):
            raise ValueError("Expected aligned nonempty labels/prompt IDs")
        if any(pd.isna(x) or not str(x).strip() for x in prompt_ids):
            raise ValueError("Missing prompt IDs")
        self.labels = [int(x) for x in labels]
        self.prompts = [str(x) for x in prompt_ids]
        self.bf, self.mpf, self.spm = map(
            int, (batch_families, models_per_family, samples_per_model)
        )
        if min(self.bf, self.mpf, self.spm) < 1:
            raise ValueError("Batch dimensions must be positive")
        self.batch_size = self.bf * self.mpf * self.spm
        self.seed, self.epoch, self._cache = int(seed), 0, None
        self.by_model = {}
        for i, label in enumerate(self.labels):
            self.by_model.setdefault(label, []).append(i)

    def set_epoch(self, epoch):
        if int(epoch) < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch, self._cache = int(epoch), None

    def _batches(self):
        if self._cache is not None:
            return self._cache
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 614]))
        queues = {m: deque(rng.permutation(indices).tolist())
                  for m, indices in self.by_model.items()}
        left = len(self.labels)
        result = []
        while left:
            batch, visited_families = [], set()
            while len(batch) < self.batch_size and left:
                active_models = [m for m, q in queues.items() if q]
                families = {MODEL_FAMILY_IDS[m] for m in active_models}
                choices = families - visited_families or families
                family = min(choices, key=lambda f: (
                    -sum(len(queues[m]) for m in active_models
                         if MODEL_FAMILY_IDS[m] == f), f
                ))
                visited_families.add(family)
                models = sorted(
                    [m for m in active_models if MODEL_FAMILY_IDS[m] == family],
                    key=lambda m: (-len(queues[m]), m)
                )
                for model in models[:self.mpf]:
                    count = min(self.spm, self.batch_size - len(batch), len(queues[model]))
                    used_prompts = set()
                    for _ in range(count):
                        q = queues[model]
                        for _ in range(min(8, len(q))):
                            if self.prompts[q[0]] not in used_prompts:
                                break
                            q.rotate(-1)
                        idx = q.popleft()
                        batch.append(idx)
                        used_prompts.add(self.prompts[idx])
                        left -= 1
            rng.shuffle(batch)
            result.append(batch)
        flat = [i for batch in result for i in batch]
        if len(flat) != len(self.labels) or len(set(flat)) != len(flat):
            raise RuntimeError("Sampler did not cover every example exactly once")
        self._cache = result
        return result

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())
