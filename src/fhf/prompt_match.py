"""Pair same-prompt responses by swapping same-model rows across batches.

Each source index is still visited exactly once per epoch.
"""
from collections import defaultdict
from .full_coverage import FullCoveragePromptFamilyBatchSampler
from .labels import MODEL_FAMILY_IDS


class PromptMatchedFullCoverageBatchSampler(FullCoveragePromptFamilyBatchSampler):
    def _batches(self):
        if self._cache is not None:
            return self._cache
        batches = [list(row) for row in super()._batches()]
        position = {idx: (batch_idx, slot)
                    for batch_idx, batch in enumerate(batches)
                    for slot, idx in enumerate(batch)}
        by_model_prompt = defaultdict(list)
        for idx, (model, prompt) in enumerate(zip(self.labels, self.prompts)):
            by_model_prompt[(model, prompt)].append(idx)
        def same_prompt_pair_count(batch):
            total = 0
            for slot, a in enumerate(batch):
                for b in batch[slot + 1:]:
                    if (self.labels[a] != self.labels[b]
                            and MODEL_FAMILY_IDS[self.labels[a]] == MODEL_FAMILY_IDS[self.labels[b]]
                            and self.prompts[a] == self.prompts[b]):
                        total += 1
            return total

        swaps = 0
        for bidx, batch in enumerate(batches):
            models_by_family = defaultdict(set)
            for idx in batch:
                models_by_family[MODEL_FAMILY_IDS[self.labels[idx]]].add(self.labels[idx])
            for _, models in sorted(models_by_family.items()):
                models = sorted(models)
                for a_model, b_model in zip(models, models[1:]):
                    a_indices = [idx for idx in batch if self.labels[idx] == a_model]
                    b_indices = [idx for idx in batch if self.labels[idx] == b_model]
                    if not a_indices or not b_indices:
                        continue
                    if any(self.prompts[a] == self.prompts[b]
                           for a in a_indices for b in b_indices):
                        continue
                    target_slot = position[b_indices[0]][1]
                    match = None
                    for a_idx in a_indices:
                        for candidate in by_model_prompt.get((b_model, self.prompts[a_idx]), ()):
                            other_batch, other_slot = position[candidate]
                            if other_batch > bidx:
                                match = (candidate, other_batch, other_slot)
                                break
                        if match is not None:
                            break
                    if match is None:
                        continue
                    candidate, other_batch, other_slot = match
                    evicted = batch[target_slot]
                    before = (same_prompt_pair_count(batch)
                              + same_prompt_pair_count(batches[other_batch]))
                    batch[target_slot] = candidate
                    batches[other_batch][other_slot] = evicted
                    after = (same_prompt_pair_count(batch)
                             + same_prompt_pair_count(batches[other_batch]))
                    if after < before:
                        # Never destroy more same-prompt pairs in a donor batch
                        # than the proposed swap adds to the target batch.
                        batch[target_slot] = evicted
                        batches[other_batch][other_slot] = candidate
                        continue
                    position[candidate] = (bidx, target_slot)
                    position[evicted] = (other_batch, other_slot)
                    swaps += 1
        flat = [idx for batch in batches for idx in batch]
        if len(flat) != len(self.labels) or len(set(flat)) != len(flat):
            raise RuntimeError("Prompt matching lost or duplicated a row")
        self.last_prompt_match_swaps = swaps
        self._cache = batches
        return batches
