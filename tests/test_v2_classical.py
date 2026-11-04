"""Leakage-free lexical and specialist tests on synthetic classes."""
import numpy as np
import pandas as pd

from fhf.labels import MODEL_NAMES
from fhf.v2_classical import LexicalFingerprintEnsemble


def test_specialist_distribution_and_model_order():
    frame=pd.DataFrame({
        "model": [m for m in MODEL_NAMES for i in range(3)],
        "text": [f"Unique response {m} lexical style token{i} suffix end."
                 for m in MODEL_NAMES for i in range(3)]
    })
    model=LexicalFingerprintEnsemble(
        seed=17, char_features=2500, word_features=500).fit(frame)
    result=model.predict_probabilities([
        "Unique response gemini-3.8-flash lexical style token2 suffix end.",
        "Unique response claude-opus-5 lexical style token1 suffix end.",
    ])
    assert set(result)=={"char","word","specialist"}
    for value in result.values():
        assert value.shape==(2,len(MODEL_NAMES))
        assert np.isfinite(value).all()
        assert (value>=0).all()
        np.testing.assert_allclose(value.sum(axis=1),1,rtol=0,atol=1e-6)
