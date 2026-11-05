"""Test repeated prompt-group matching without access to real data."""
import importlib.util
from pathlib import Path
import pandas as pd


def loader():
    path=Path(__file__).resolve().parents[1]/"scripts/fingerprint_v2_verify.py"
    spec=importlib.util.spec_from_file_location("v2verify",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_multiple_original_prompts_detected():
    verify=loader()
    selected=pd.DataFrame({
        "id":["x","y","z"],"model":["glm-5.3","glm-5.3-flash","glm-5.3"],
        "prompt_id":["same"]*3,"source_dataset":["src"]*3,
        "text":["response x","response y","response z"]})
    pool=pd.DataFrame({
        "text":selected.text,"model":selected.model,
        "source_dataset":selected.source_dataset,
        "prompt_group":["shared"]*3,
        "prompt_text":["first question","second question","third question"]})
    result,_,col=verify.inspect_groups(selected,pool,2)
    assert col=="prompt_text"
    assert int(result.iloc[0]["n"])==3
    assert result.iloc[0]["verdict"]=="MULTIPLE_ORIGINAL_PROMPTS"


def test_cannot_verify_if_no_prompt_text():
    verify=loader()
    selected=pd.DataFrame({
        "id":["x","y"],"model":["glm-5.3","glm-5.3-flash"],
        "prompt_id":["same","same"],"source_dataset":["src"]*2,
        "text":["response x","response y"]})
    pool=pd.DataFrame({
        "text":selected.text,"model":selected.model,
        "source_dataset":selected.source_dataset,"prompt_group":["shared"]*2})
    result,_,col=verify.inspect_groups(selected,pool,2)
    assert col is None
    assert result.iloc[0]["verdict"]=="UNVERIFIED_ORIGINAL_PROMPT_TEXT_UNAVAILABLE"
