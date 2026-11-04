from collections import Counter
from fhf.full_coverage import FullCoveragePromptFamilyBatchSampler
from fhf.prompt_match import PromptMatchedFullCoverageBatchSampler


def pair_count(batches, labels, prompts):
    return sum(labels[a] != labels[b] and prompts[a] == prompts[b]
               for batch in batches for i, a in enumerate(batch)
               for b in batch[i+1:])


def test_matching_coverage_and_model_slots():
    labels = [0]*24 + [1]*24 + [5]*24 + [6]*24
    prompts = [f"p{i%12}" for _ in range(4) for i in range(24)]
    kwargs = dict(batch_families=2, models_per_family=2,
                  samples_per_model=2, seed=19)
    original = FullCoveragePromptFamilyBatchSampler(labels, prompts, **kwargs)
    paired = PromptMatchedFullCoverageBatchSampler(labels, prompts, **kwargs)
    a, b = list(original), list(paired)
    assert len(a)==len(b)
    assert sorted(i for batch in b for i in batch)==list(range(len(labels)))
    assert all(Counter(labels[i] for i in x)==Counter(labels[i] for i in y)
               for x,y in zip(a,b))
    assert pair_count(b,labels,prompts) >= pair_count(a,labels,prompts)
