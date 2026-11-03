"""Optional external-data contrastive pretraining with YAML paths and hyperparameters."""
import argparse
import torch
import pandas as pd
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from fhf.config import load_config, resolve_path
from fhf.losses import contrastive_loss
from fhf.engine import seed_everything


class ExternalDataset(Dataset):
    """Tokenize real external generator labels, never map them to Kaggle version names."""
    def __init__(self, df, tokenizer, model_ids, family_ids, max_length):
        self.df, self.tokenizer, self.max_length = df, tokenizer, max_length
        self.models = df.model.map(model_ids).tolist()
        self.providers = df.provider.map(family_ids).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.tokenizer(str(self.df.iloc[index].text),
                              truncation=True, max_length=self.max_length)
        item["labels"] = self.models[index]
        item["families"] = self.providers[index]
        return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external.yaml")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    config = load_config(args.config, args.set)
    settings, model_cfg = config["external_pretraining"], config["model"]
    seed_everything(int(settings["seed"]))
    df = pd.read_csv(resolve_path(config, "external_csv"))
    if not {"text", "model"}.issubset(df.columns):
        raise ValueError("External CSV requires text and genuine model columns")
    if "provider" not in df:
        df["provider"] = df.model.astype(str)
    df = df.dropna(subset=["text", "model", "provider"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("External data is empty")
    model_ids = {name: i for i, name in enumerate(sorted(df.model.unique()))}
    provider_ids = {name: i for i, name in enumerate(sorted(df.provider.unique()))}
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"] or model_cfg["encoder"])
    encoder = AutoModel.from_pretrained(model_cfg["encoder"])
    projector = nn.Linear(encoder.config.hidden_size, int(model_cfg["projection_dim"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, projector = encoder.to(device), projector.to(device)
    dataset = ExternalDataset(df, tokenizer, model_ids, provider_ids, int(model_cfg["max_length"]))

    def collate(samples):
        labels = torch.tensor([sample.pop("labels") for sample in samples])
        families = torch.tensor([sample.pop("families") for sample in samples])
        return tokenizer.pad(samples, return_tensors="pt"), labels, families

    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True,
                        collate_fn=collate)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()),
                                   lr=float(settings["learning_rate"]))
    for epoch in range(int(settings["epochs"])):
        encoder.train()
        projector.train()
        running = []
        for inputs, labels, families in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels, families = labels.to(device), families.to(device)
            outputs = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings = F.normalize(projector(pooled), dim=-1)
            loss = contrastive_loss(embeddings, labels, families)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach()))
        print(f"External epoch {epoch + 1}: loss={sum(running) / max(len(running), 1):.4f}")
    dest = resolve_path(config, "external_encoder_dir")
    encoder.save_pretrained(dest / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(dest / "tokenizer")
    print(f"External encoder saved: {dest}")


if __name__ == "__main__":
    main()
