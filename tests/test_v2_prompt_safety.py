import pandas as pd
import pytest

from fhf.v2_prompt_safety import safe_prompt_ids


def test_only_oversized_training_prompt_ids_are_neutralized():
    data=pd.DataFrame({
        "id":["a","b","c","d","e"],
        "prompt_id":["oversized","oversized","oversized","valid","valid"],
        "model":["glm-5.3"]*5,
        "text":[f"content {i}" for i in range(5)]})
    before=data.copy(deep=True)
    result, stats=safe_prompt_ids(data,max_group_size=2)
    assert len(result)==len(data)
    assert len(set(result[:3]))==3
    assert result[3:]==["valid","valid"]
    assert stats["affected_groups"]==1
    assert stats["affected_rows"]==3
    pd.testing.assert_frame_equal(data,before)


def test_disabled_mode_preserves_original_prompt_ids():
    x=pd.DataFrame({"id":["1","2"],"prompt_id":["same","same"]})
    got, stats=safe_prompt_ids(x)
    assert got==["same","same"]
    assert not stats["enabled"]


def test_invalid_threshold_rejected():
    x=pd.DataFrame({"id":["1","2"],"prompt_id":["same","same"]})
    with pytest.raises(ValueError,match="max_group_size"):
        safe_prompt_ids(x,1)
