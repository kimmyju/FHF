"""Test numerical behavior of supervised contrastive and consistency losses."""
import torch
from fhf.losses import contrastive_loss, symmetric_kl


def test_contrastive_has_grad():
    embeddings=torch.nn.functional.normalize(torch.randn(6,8,requires_grad=True),dim=-1)
    labels=torch.tensor([0,0,1,1,2,2])
    families=torch.tensor([0,0,0,0,1,1])
    loss=contrastive_loss(embeddings,labels,families)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad_fn is not None


def test_missing_positive_is_zero():
    z=torch.randn(3,4,requires_grad=True)
    loss=contrastive_loss(z,torch.tensor([0,1,2]),torch.tensor([0,0,1]))
    assert loss.item()==0
    loss.backward()


def test_consistency_same_view():
    logp=torch.log_softmax(torch.randn(4,18),-1)
    assert abs(symmetric_kl(logp,logp).item())<1e-6
