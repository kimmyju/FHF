"""Validate isolated provider-weight experiment and correct gradient scaling."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fhf.config import load_config
from fhf.labels import FAMILY_NAMES, FAMILY_TO_ID, MODEL_NAMES
from fhf.losses import objective
from fhf.v2_family_balance import family_class_weights

ROOT=Path(__file__).resolve().parents[1]


def test_inverse_frequency_on_uneven_providers():
    labels=["claude-fable-5"]*5+["glm-5.3"]*2+[
        name for name in MODEL_NAMES if name not in ("claude-fable-5","glm-5.3")]
    # n per family differs (some families have many individual model labels).
    weights,report=family_class_weights(labels)
    count=report["counts"]
    assert len(weights)==len(FAMILY_NAMES)
    assert sum(count[f]*weights[i] for i,f in enumerate(FAMILY_NAMES))/len(labels)==pytest.approx(1.0)
    assert weights[FAMILY_TO_ID["ZAI"]]>weights[FAMILY_TO_ID["Anthropic"]]
    assert report["n_train"]==len(labels)


def test_half_power_and_identity():
    labels=list(MODEL_NAMES)+["claude-fable-5"]*10
    w0,_=family_class_weights(labels,power=0)
    w1,_=family_class_weights(labels,power=1)
    w_half,_=family_class_weights(labels,power=.5)
    assert w0==pytest.approx([1]*len(FAMILY_NAMES))
    assert max(w0)<max(w_half)<max(w1)
    for power in (-.1,1.1,float("nan"),True):
        with pytest.raises(ValueError,match="power"):
            family_class_weights(labels,power=power)


def test_weighted_ce_has_fixed_batch_mean_and_gradient():
    logits=torch.tensor([[1.,-1.],[1.,-1.]],requires_grad=True)
    family=torch.tensor([0,1])
    w=torch.tensor([.5,2.])
    cfg={"losses":{
        "exact_classification":{"enabled":False,"weight":0},
        "flat_classification":{"enabled":False,"weight":0},
        "family":{"enabled":True,"weight":.25,"class_weights":w.tolist()},
        "intra_contrastive":{"enabled":False,"weight":0},
    }}
    loss,parts=objective({"family_logits":logits},torch.tensor([0,0]),family,cfg)
    expected=((F.cross_entropy(logits,family,reduction="none")*w).mean())*.25
    assert loss.item()==pytest.approx(expected.item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # The weighted batch mean must differ from torch CE(weight,mean).
    wrong=F.cross_entropy(logits.detach(),family,weight=w)
    assert loss.item()!=pytest.approx(wrong.item()*.25)


def test_family_balanced_yaml_differs_only_in_switch_and_paths():
    isolated=load_config(ROOT/"configs/fingerprint_v2/finetune_prompt_isolated.yaml")
    balanced=load_config(ROOT/"configs/fingerprint_v2/finetune_family_balanced.yaml")
    assert isolated["model"]==balanced["model"]
    assert isolated["training"]==balanced["training"]
    assert isolated["paths"]["initial_checkpoint"]==balanced["paths"]["initial_checkpoint"]
    for key in ("exact_classification","flat_classification","intra_contrastive","rdrop","hotflip"):
        assert isolated["losses"][key]==balanced["losses"][key]
    assert balanced["losses"]["family"]["class_balance"]=={"enabled":True,"power":1.0}
    assert isolated["losses"]["family"]["enabled"]==balanced["losses"]["family"]["enabled"]
    assert isolated["losses"]["family"]["weight"]==balanced["losses"]["family"]["weight"]
    assert isolated["paths"]["checkpoint_dir"]!=balanced["paths"]["checkpoint_dir"]


def test_invalid_config_rejected():
    name=str(ROOT/"configs/fingerprint_v2/finetune_family_balanced.yaml")
    for override in ("losses.family.class_balance.power=-1",
                     "losses.family.class_balance.power=2",
                     "losses.family.class_balance.enabled=false"):
        if override.endswith("false"):
            assert not load_config(name,[override])["losses"]["family"]["class_balance"]["enabled"]
        else:
            with pytest.raises(ValueError,match="power"):
                load_config(name,[override])
    with pytest.raises(ValueError,match="Family balance requires"):
        load_config(name,["losses.family.enabled=false"])
