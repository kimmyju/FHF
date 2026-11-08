"""Train and submit the independent decoder attribution experiment."""
import argparse
import json
import pandas as pd
from fhf.v3_decoder import run, infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--config", default="configs/fingerprint_v3/qwen3_4b.json")
    fit.add_argument("--train", default="data/train.csv")
    fit.add_argument("--valid", help="Frozen holdout CSV; otherwise grouped CV")
    fit.add_argument("--external", help="Already audited joined CSV with exact labels")
    fit.add_argument("--out", required=True)
    fit.add_argument("--fold-only", type=int, help="0-based pilot fold; not full OOF")
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--accumulation", type=int)
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--model")
    fit.add_argument("--wandb", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--base", required=True, help="Local complete base model directory")
    predict.add_argument("--test", default="/kaggle/input/competitions/name-that-llm/test.csv")
    predict.add_argument("--out", default="submission.parquet")
    predict.add_argument("--batch-size", type=int, default=4)
    predict.add_argument("--lexical-weight", type=float, help="Override only after validation ablation")
    args = parser.parse_args()
    if args.command == "predict":
        infer(args.bundle, pd.read_csv(args.test, dtype={"id": str}), args.base, args.out,
              args.batch_size, args.lexical_weight)
        return
    with open(args.config, encoding="utf8") as stream:
        cfg = json.load(stream)
    for key in ("batch_size", "accumulation", "epochs", "model"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["wandb"] = args.wandb or cfg.get("wandb", False)
    read = lambda path: pd.read_csv(path, dtype={"id": str, "prompt_id": str}) if path else None
    result = run(read(args.train), cfg, args.out, read(args.valid), read(args.external), args.fold_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
