"""Audit selected external GLM vs Claude Fable training data (no training)."""
import argparse
from pathlib import Path
import pandas as pd
from fhf.v2_source_audit import audit_selected, save_audit


def read(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing data: {p}")
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(
        p, dtype={"id": str, "prompt_id": str}, low_memory=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--selected", default="data/external/full/selected_3495_per_model.csv")
    p.add_argument("--validation", default="outputs/dual-path-prompt-aware/val.csv")
    p.add_argument("--raw-pool", default="data/external/ntl_v2/ntl_v3/ntl_open_pool.parquet")
    p.add_argument("--out", default="outputs/fingerprint-v2/source-audit")
    p.add_argument("--no-raw", action="store_true",
                   help="Skip large raw-pool metadata comparison")
    p.add_argument("--include-text", action="store_true",
                   help="Opt-in only: export actual texts for local manual inspection")
    args = p.parse_args()
    raw = None
    if not args.no_raw:
        if not Path(args.raw_pool).is_file():
            p.error(f"Missing --raw-pool {args.raw_pool}; use --no-raw to skip")
        raw = read(args.raw_pool)
    result = audit_selected(
        read(args.train), read(args.selected), read(args.validation),
        raw_pool=raw, include_text=args.include_text)
    output = save_audit(result, args.out)
    s = result[0]
    print(f"Saved GLM/Claude audit to {output}")
    print(f"Exact label conflicts: {s['selected_hashes_with_conflicting_labels']}")
    for name, metrics in s["focus"].items():
        print(f"{name}: {metrics['selected_rows']} rows, "
              f"dominant source {metrics['dominant_source']} "
              f"({metrics['dominant_source_fraction']:.1%})")
    print("Check raw source/API routing records manually: text statistics cannot verify true generator identity.")


if __name__ == "__main__":
    main()
