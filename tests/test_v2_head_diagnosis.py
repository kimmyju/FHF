import numpy as np
import pandas as pd
from fhf.labels import MODEL_NAMES, FAMILY_NAMES
from fhf.v2_head_diagnosis import analyze_outputs, save_report


def test_family_mistake_separated_from_flat_mistake(tmp_path):
    n=3
    family=np.full((n,len(FAMILY_NAMES)),.01)
    family[0,FAMILY_NAMES.index("Anthropic")]=.95
    family[1,FAMILY_NAMES.index("ZAI")]=.95
    family[2,FAMILY_NAMES.index("Anthropic")]=.95
    family/=family.sum(1,keepdims=True)
    probs=np.full((n,len(MODEL_NAMES)),.001)
    probs[0,MODEL_NAMES.index("claude-fable-5")]=.9
    probs[1,MODEL_NAMES.index("claude-fable-5")]=.8
    probs[2,MODEL_NAMES.index("claude-fable-5")]=.9
    probs/=probs.sum(1,keepdims=True)
    model_logits=np.zeros_like(probs)
    model_logits[:,MODEL_NAMES.index("glm-5.3")]=3
    summary,table=analyze_outputs(
        ["1","2","3"],["glm-5.3","glm-5.3-flash","claude-fable-5"],
        family,model_logits,probs,flat_probs=probs,hierarchical_probs=probs)
    assert summary["glm_aggregate"]["n"]==2
    assert summary["glm_aggregate"]["family_head_Anthropic"]==1
    assert summary["glm_aggregate"]["final_pred_fable"]==2
    assert summary["glm_aggregate"]["final_wrong_but_family_head_correct"]==1
    assert table.loc[0,"oracle_within_ZAI_pred"]=="glm-5.3"
    save_report(summary,table,tmp_path)
    assert (tmp_path/"head_diagnosis.json").is_file()
    assert (tmp_path/"focus_head_probabilities.csv").is_file()


def test_bad_probabilities_rejected():
    family=np.zeros((1,len(FAMILY_NAMES)))
    probs=np.full((1,len(MODEL_NAMES)),1/len(MODEL_NAMES))
    with np.testing.assert_raises(ValueError):
        analyze_outputs(["1"],["glm-5.3"],family,probs,probs)
