"""Diagnose GLM->Claude mistakes by model head using a saved checkpoint.

No retraining. Frozen validation labels used only to score and group results.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from fhf.data import load_csv, TextDataset, make_collator
from fhf.model import FingerprintModel
from fhf.v2_head_diagnosis import analyze_outputs, save_report


def run(checkpoint, validation, output, batch_size=16, max_length=384,
        base_encoder_override=None):
    model,tokenizer,_=FingerprintModel.from_checkpoint(
        checkpoint,base_encoder_override=base_encoder_override)
    if not model.heads.get("model") or not model.heads.get("family"):
        raise ValueError("Both model and family heads are required")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    data=TextDataset(validation,tokenizer,max_length=max_length,
                     style_config=model.style_config)
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,
                      collate_fn=make_collator(tokenizer))
    family,local,final,flat,hier=[],[],[],[],[]
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels",None)
            batch.pop("families",None)
            out=model(**{k:v.to(device) for k,v in batch.items()})
            family.append(out["family_log_probs"].float().exp().cpu().numpy())
            local.append(out["model_logits"].float().cpu().numpy())
            final.append(out["log_probs"].float().exp().cpu().numpy())
            if "flat_log_probs" in out:
                flat.append(out["flat_log_probs"].float().exp().cpu().numpy())
                hier.append(out["hierarchical_log_probs"].float().exp().cpu().numpy())
    if bool(flat)!=bool(hier):
        raise RuntimeError("Flat and hierarchical outputs must be paired")
    summary,details=analyze_outputs(
        validation.id.astype(str).tolist(),validation.model.astype(str).tolist(),
        np.concatenate(family),np.concatenate(local),np.concatenate(final),
        flat_probs=np.concatenate(flat) if flat else None,
        hierarchical_probs=np.concatenate(hier) if hier else None)
    folder=save_report(summary,details,output)
    print(f"Output: {folder}")
    print("GLM aggregate:",summary["glm_aggregate"])
    for name,values in summary["per_model"].items():
        print(name, values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",default="outputs/fingerprint-v2/finetune-isolated/checkpoint")
    p.add_argument("--validation",default="outputs/dual-path-prompt-aware/val.csv")
    p.add_argument("--out",default="outputs/fingerprint-v2/glm-root-cause/heads")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--base-encoder-override",default=None)
    a=p.parse_args()
    run(a.checkpoint,load_csv(a.validation),a.out,a.batch_size,
        a.max_length,a.base_encoder_override)


if __name__=="__main__":
    main()
