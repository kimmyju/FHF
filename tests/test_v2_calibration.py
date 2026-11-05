"""OOF-only calibration and label alignment regression tests."""
import numpy as np
import pandas as pd
import pytest
from fhf.labels import MODEL_NAMES
from fhf.v2_calibration import (
    apply_bias, fit_bias, build_oof, calibrate_oof_cv,
    fit_and_save, load_bias,
)


def example(n=4):
    rows = len(MODEL_NAMES) * n
    y = np.repeat(np.arange(len(MODEL_NAMES)), n)
    probabilities = np.full((rows, len(MODEL_NAMES)), .1 / (len(MODEL_NAMES)-1))
    probabilities[np.arange(rows), y] = .9
    return y, probabilities


def test_bias_fit_on_balanced_predictors():
    labels, probs = example()
    fit = fit_bias(probs,labels)
    assert fit.shape == (18,)
    np.testing.assert_allclose(fit.sum(),0,atol=1e-6)
    adjusted=apply_bias(probs,fit)
    np.testing.assert_allclose(adjusted.sum(1),1,atol=1e-6)
    result=calibrate_oof_cv(
        probs,labels,[f"prompt-{i%4}" for i in range(len(labels))],
        folds=4)
    assert result["base_bacc"]==1.0
    assert result["crossfit_calibrated_bacc"]==1.0


def test_neural_oof_required_to_calibrate_full_ensemble():
    train=pd.DataFrame({
        "id":[str(i) for i in range(18)],"prompt_id":[str(i) for i in range(18)],
        "model":MODEL_NAMES})
    lex=train.rename(columns={"model":"true_model"}).copy()
    for k in ("char","word","specialist"):
        for model in MODEL_NAMES:
            lex[f"{k}_prob_{model}"]=1/18
    with pytest.raises(ValueError,match="Neural-heavy"):
        build_oof(train,lex,neural_weight=0.7)


def test_unexpected_class_order_rejected(tmp_path):
    file=tmp_path/"bad.json"
    file.write_text('{"schema":"fhf-v2-class-log-bias-1","class_order":["wrong"],"biases":[1]}')
    with pytest.raises(ValueError,match="ordering"):
        load_bias(file)
