"""Check style-fusion dtype reconciliation without autocast."""
import torch
from fhf.style import GatedStyleFusion


def test_fp16_backbone_fp32_style_eval_without_autocast():
    fusion=GatedStyleFusion(feature_dim=16,hidden_size=8,style_hidden=4).eval()
    contextual=torch.randn(3,8,dtype=torch.float16)
    style_features=torch.randn(3,16,dtype=torch.float32)
    with torch.inference_mode():
        fused,gate_mean=fusion(contextual,style_features)
    assert fused.shape==(3,8)
    assert fused.dtype==torch.float32
    assert torch.isfinite(fused).all()
    assert torch.isfinite(gate_mean)


def test_fp16_backbone_fp32_style_backprop_without_autocast():
    fusion=GatedStyleFusion(feature_dim=16,hidden_size=8,style_hidden=4).train()
    contextual=torch.randn(2,8,dtype=torch.float16,requires_grad=True)
    style_features=torch.randn(2,16,dtype=torch.float32)
    fused,_=fusion(contextual,style_features,noise_std=0.05)
    fused.square().mean().backward()
    assert torch.isfinite(fused).all()
    assert contextual.grad is not None
    assert torch.isfinite(contextual.grad).all()
    assert fusion.style_encoder[0].weight.grad is not None
    assert torch.isfinite(fusion.style_encoder[0].weight.grad).all()


def test_original_fp32_path():
    fusion=GatedStyleFusion(feature_dim=8,hidden_size=4,style_hidden=3).eval()
    with torch.inference_mode():
        result,gate=fusion(torch.randn(2,4),torch.randn(2,8))
    assert result.dtype==torch.float32
    assert result.shape==(2,4)
    assert gate.ndim==0
