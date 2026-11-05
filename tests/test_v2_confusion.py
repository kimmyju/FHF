import pandas as pd
import numpy as np
import pytest
from fhf.labels import MODEL_NAMES
from fhf.v2_confusion import diagnose


def test_glm_confusion_exports(tmp_path):
    data = pd.DataFrame([
        {"id":"1","prompt_id":"p1","true_model":"glm-5.3","pred_model":"glm-5.3-flash"},
        {"id":"2","prompt_id":"p2","true_model":"glm-5.3","pred_model":"glm-5.3"},
        {"id":"3","prompt_id":"p3","true_model":"glm-5.3-flash","pred_model":"gpt-5.5"},
        {"id":"4","prompt_id":"p4","true_model":"gpt-5.5","pred_model":"glm-5.3"},
    ])
    result = diagnose(data, tmp_path, plot=False)
    matrix = pd.read_csv(tmp_path/"confusion_matrix_counts.csv",index_col=0)
    assert list(matrix.index)==MODEL_NAMES
    assert list(matrix.columns)==MODEL_NAMES
    assert matrix.loc["glm-5.3","glm-5.3-flash"]==1
    assert matrix.loc["gpt-5.5","glm-5.3"]==1
    assert result["glm"]["glm-5.3"]["recall"]==0.5
    assert result["glm"]["glm-5.3-flash"]["recall"]==0.0
    assert len(pd.read_csv(tmp_path/"glm_errors.csv"))==3
    norm = pd.read_csv(tmp_path/"confusion_matrix_row_normalized.csv",index_col=0)
    np.testing.assert_allclose(norm.loc["glm-5.3"].sum(),1)
    assert norm.loc["kimi-k3"].sum()==0


def test_bad_data_rejected(tmp_path):
    data=pd.DataFrame({"id":["x","x"],"true_model":["glm-5.3"]*2,
                       "pred_model":["glm-5.3"]*2})
    with pytest.raises(ValueError,match="duplicated"):
        diagnose(data,tmp_path,plot=False)
