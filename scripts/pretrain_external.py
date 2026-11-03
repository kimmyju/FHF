"""Optional supervised contrastive pretraining using genuine external generator labels."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from fhf.losses import contrastive_loss
from fhf.engine import seed_everything


class ExternalDataset(Dataset):
    """Keep historical generator identities separate from competition target labels."""
    def __init__(self, df, tokenizer, identities, providers, max_length):
        self.df, self.tokenizer, self.max_length = df, tokenizer, max_length
        self.models = df.model.map(identities).tolist()
        self.providers = df.provider.map(providers).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        t = self.tokenizer(str(self.df.iloc[index].text), truncation=True, max_length=self.max_length)
        t['labels'] = self.models[index]
        t['families'] = self.providers[index]
        return t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True)
    p.add_argument('--backbone', default='microsoft/deberta-v3-base')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    df = pd.read_csv(args.csv)
    if not {'model','text'}.issubset(df.columns):
        raise ValueError('Required external CSV columns: model,text; optional provider')
    if 'provider' not in df:
        df['provider'] = df.model.map(lambda x: str(x).split('/')[0])
    df = df.dropna(subset=['model','text','provider']).copy()
    model_ids = {k:i for i,k in enumerate(sorted(df.model.unique()))}
    family_ids = {k:i for i,k in enumerate(sorted(df.provider.unique()))}
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    encoder = AutoModel.from_pretrained(args.backbone)
    projector = nn.Linear(encoder.config.hidden_size, 128)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, projector = encoder.to(device), projector.to(device)
    ds = ExternalDataset(df.reset_index(drop=True),tokenizer,model_ids,family_ids,args.max_length)
    def collate(items):
        model_labels = torch.tensor([i.pop('labels') for i in items])
        families = torch.tensor([i.pop('families') for i in items])
        return tokenizer.pad(items, return_tensors='pt'),model_labels,families
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,collate_fn=collate)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(projector.parameters()),lr=args.learning_rate)
    for ep in range(args.epochs):
        encoder.train(); projector.train()
        losses=[]
        for inputs, labels, families in loader:
            inputs = {k:v.to(device) for k,v in inputs.items()}
            labels,families = labels.to(device),families.to(device)
            outputs=encoder(**inputs).last_hidden_state
            weights=inputs['attention_mask'].unsqueeze(-1)
            pooled=(outputs*weights).sum(1)/weights.sum(1).clamp_min(1)
            z=F.normalize(projector(pooled),dim=-1)
            loss=contrastive_loss(z,labels,families)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        print(f'external epoch {ep+1}: contrastive_loss={sum(losses)/max(len(losses),1):.4f}')
    out=Path(args.output_dir)
    encoder.save_pretrained(out/'encoder',safe_serialization=True)
    tokenizer.save_pretrained(out/'tokenizer')
    print('Saved pretrained encoder/tokenizer. FHF classification heads initialize separately.')


if __name__ == '__main__':
    main()
