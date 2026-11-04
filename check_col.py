from pathlib import Path

import pandas as pd

root = Path("data/external/ntl_v2/ntl_v3")
for name in ["ntl_open_pool.parquet", "prompts.parquet", "sources.csv"]:
    path = root / name
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(name, df.shape, df.columns.tolist())