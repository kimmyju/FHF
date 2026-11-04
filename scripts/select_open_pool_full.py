"""Select a balanced large external dataset without accepting risky duplicates."""
import argparse
import json
from pathlib import Path
from audit_external_pool import read_table, v2_frame, pool_frame
from select_open_pool_e1 import produce

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-model", type=int, default=3495)
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--val", default="outputs/dual-path-prompt-aware/val.csv")
    p.add_argument("--v2-passages", default="data/external/ntl_v2/passages.csv")
    p.add_argument("--v2-prompts", default="data/external/ntl_v2/prompts.csv")
    p.add_argument("--pool", default="data/external/ntl_v2/ntl_v3/ntl_open_pool.parquet")
    a = p.parse_args()
    if a.per_model < 1 or a.min_words < 1 or a.max_words < a.min_words:
        p.error("Invalid per-model or word limits")
    train, val = read_table(a.train), read_table(a.val)
    v2, _ = v2_frame(a.v2_passages, a.v2_prompts)
    pool = pool_frame(a.pool)
    try:
        selected, report = produce(
            train, val, v2, pool, per_model=a.per_model,
            seed=a.seed, min_words=a.min_words,
            max_words=a.max_words, exclude_unknown_risk=True)
    except ValueError as e:
        raise SystemExit(
            f"Insufficient strictly eligible rows: {e}. Refusing to duplicate "
            "rows or accept flagged data. Use fewer samples per model."
        ) from e
    out = Path(f"data/external/full/selected_{a.per_model}_per_model.csv")
    audit_out = Path(f"outputs/openpool-full-{a.per_model}/selection_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)
    report.update({"min_words": a.min_words, "max_words": a.max_words})
    audit_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(selected)} external + {len(train)} official training rows")
    print(f"Saved {out} and {audit_out}")

if __name__ == "__main__":
    main()
