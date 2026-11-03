"""Test HotFlip candidate filtering with tokenizer mock and no model downloads."""
from fhf.hotflip import marker_token_ids


class DummyTokenizer:
    all_special_ids=[0]
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [len(text)] if text.startswith(' ') else [0,1]


def test_candidates_single_token_only():
    ids=marker_token_ids(DummyTokenizer())
    assert len(ids)>0 and all(i!=0 for i in ids)
