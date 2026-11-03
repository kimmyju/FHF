"""Test exact competition label policy and prompt-disjoint splitting."""
import pandas as pd
import pytest
from fhf.labels import MODEL_NAMES, MODEL_FAMILY_IDS, FAMILY_NAMES
from fhf.data import grouped_split, load_csv


def test_labels_and_families():
    assert len(MODEL_NAMES)==18
    assert len(set(MODEL_NAMES))==18
    assert len(FAMILY_NAMES)==6
    assert len(MODEL_FAMILY_IDS)==18


def test_prompt_disjoint():
    df=pd.DataFrame({'id':[f'i{i}' for i in range(20)],
                     'text':['hello world']*20,
                     'model':[MODEL_NAMES[i%18] for i in range(20)],
                     'prompt_id':[i//2 for i in range(20)]})
    train,val=grouped_split(df,0.2,42)
    assert set(train.prompt_id).isdisjoint(val.prompt_id)
    assert len(train)+len(val)==len(df)


def test_unknown_label_rejected(tmp_path):
    path=tmp_path/'data.csv'
    pd.DataFrame({'id':['x'],'text':['hello'],'model':['gpt-4']}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Unknown'):
        load_csv(path)


def test_sampler_changes_between_epochs_but_is_reproducible():
    """Avoid repeating the exact same contrastive batches every epoch."""
    from fhf.data import FamilyBatchSampler
    labels = list(range(18)) * 6
    sampler = FamilyBatchSampler(labels, batch_families=2, models_per_family=2,
                                 samples_per_model=2, steps_per_epoch=20, seed=42)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)
    assert first == list(sampler)
    assert first != second
    assert all(len(batch) == 8 for batch in first)
    with pytest.raises(ValueError, match="non-negative"):
        sampler.set_epoch(-1)


def test_sampler_equalizes_18_model_exposures_and_uses_all_examples():
    """With 120 examples/class and 270 steps each model is seen 120 times."""
    from collections import Counter
    from fhf.data import FamilyBatchSampler
    labels = list(range(18)) * 120
    sampler = FamilyBatchSampler(
        labels, batch_families=2, models_per_family=2,
        samples_per_model=2, steps_per_epoch=270, seed=42)
    batches = list(sampler)
    counts = Counter(labels[i] for batch in batches for i in batch)
    assert len(batches) == 270
    assert all(len(batch) == 8 for batch in batches)
    assert all(counts[model] == 120 for model in range(18))
    assert sampler.last_model_exposures == dict(counts)
    assert len(set(i for batch in batches for i in batch)) == len(labels)

    for batch in batches:
        families = {MODEL_FAMILY_IDS[labels[i]] for i in batch}
        assert len(families) == 2
        for family in families:
            model_counts = Counter(labels[i] for i in batch
                                   if MODEL_FAMILY_IDS[labels[i]] == family)
            assert sum(model_counts.values()) == 4
            assert all(count >= 2 for count in model_counts.values())
            if len({m for m in range(18) if MODEL_FAMILY_IDS[m] == family}) >= 2:
                assert len(model_counts) == 2


def test_sampler_balances_unequal_data_sizes_and_short_epochs():
    """Model balancing must not depend on per-class dataset counts."""
    from collections import Counter
    from fhf.data import FamilyBatchSampler
    # Covers all 6 families with highly unequal numbers of source examples.
    labels = [0] * 3 + [5] * 9 + [9] * 4 + [14] * 13 + [15] * 8 + [17] * 6
    sampler = FamilyBatchSampler(
        labels, batch_families=2, models_per_family=2,
        samples_per_model=2, steps_per_epoch=40, seed=9)
    batches = list(sampler)
    counts = Counter(labels[i] for batch in batches for i in batch)
    assert max(counts.values()) - min(counts.values()) <= 4
    assert all(len(batch) == 8 for batch in batches)
    assert sum(counts.values()) == 320


def test_sampler_rejects_invalid_step_count():
    from fhf.data import FamilyBatchSampler
    with pytest.raises(ValueError, match="steps_per_epoch"):
        FamilyBatchSampler(list(range(18)), steps_per_epoch=0)
