"""Verify repeated external prompt IDs and source confounding without GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fhf.external_data import normalize_text

FOCUS = ("glm-5.3", "glm-5.3-flash", "claude-fable-5")
PROMPTS = ("prompt_text", "original_prompt", "raw_prompt", "input_prompt",
           "user_prompt", "question", "instruction", "prompt")


def digest_text(text):
    return hashlib.sha256(normalize_text(str(text)).encode("utf8")).hexdigest()


def read(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix==".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path,low_memory=False,dtype={"id":str,"prompt_id":str})


def inspect_groups(selected, pool, minimum=10):
    needed={"id","text","model","prompt_id","source_dataset"}
    raw_needed={"text","model","prompt_group","source_dataset"}
    if not needed.issubset(selected):
        raise ValueError("Selected data missing "+str(needed-set(selected)))
    if not raw_needed.issubset(pool):
        raise ValueError("Raw pool missing "+str(raw_needed-set(pool)))
    if selected.id.astype(str).duplicated().any():
        raise ValueError("Selected ID duplicates")
    if minimum<2:
        raise ValueError("minimum must be >=2")
    groups=selected.groupby("prompt_id").size()
    large_ids=groups.index[groups>=minimum]
    chosen=selected[selected.prompt_id.isin(large_ids)].copy()
    chosen["_hash"]=chosen.text.fillna("").map(digest_text)
    raw=pool.copy()
    raw["_hash"]=raw.text.fillna("").map(digest_text)
    prompt_col=next((c for c in PROMPTS if c in raw.columns),None)
    raw_cols=["_hash","model","source_dataset","prompt_group"]
    if prompt_col:
        raw_cols.append(prompt_col)
    renamed={"prompt_group":"raw_prompt_group"}
    if prompt_col:
        renamed[prompt_col]="raw_prompt_text"
    matched=chosen.drop(columns="text").merge(
        raw[raw_cols].rename(columns=renamed),
        on=["_hash","model","source_dataset"],how="left",
        validate="one_to_many",indicator=True)
    results=[]
    for pid, rows in chosen.groupby("prompt_id"):
        r=matched[matched.prompt_id==pid]
        coverage=r.loc[r["_merge"]=="both","id"].nunique()
        raw_groups=r.loc[r["_merge"]=="both","raw_prompt_group"].dropna().astype(str).nunique()
        text_coverage=0
        prompt_text_count=None
        if prompt_col:
            valid=r.raw_prompt_text.notna() & r.raw_prompt_text.astype(str).str.strip().ne("")
            text_coverage=r.loc[valid,"id"].nunique()
            prompt_text_count=r.loc[valid,"raw_prompt_text"].map(normalize_text).nunique()
        if coverage<len(rows):
            result="UNVERIFIED_RAW_MATCH_INCOMPLETE"
        elif raw_groups>1:
            result="MULTIPLE_RAW_PROMPT_GROUPS"
        elif not prompt_col or text_coverage<len(rows):
            result="UNVERIFIED_ORIGINAL_PROMPT_TEXT_UNAVAILABLE"
        elif prompt_text_count>1:
            result="MULTIPLE_ORIGINAL_PROMPTS"
        else:
            result="SINGLE_RECORDED_PROMPT_NOT_API_VERIFIED"
        results.append({
            "prompt_id":str(pid),"n":int(len(rows)),
            "models":json.dumps(rows.model.value_counts().to_dict()),
            "sources":json.dumps(rows.source_dataset.value_counts().to_dict()),
            "matched_selected_ids":int(coverage),
            "raw_group_values":int(raw_groups),
            "prompt_text_column":prompt_col,
            "unique_original_prompts":prompt_text_count,
            "verdict":result,
        })
    report=pd.DataFrame(results,columns=[
        "prompt_id","n","models","sources","matched_selected_ids",
        "raw_group_values","prompt_text_column","unique_original_prompts","verdict"])
    if len(report):
        report=report.sort_values("n",ascending=False)
    return report,matched.drop(columns="_merge"),prompt_col


def compare_source_generalization(selected, official_train, min_per_class=40,
                                  cap=120, seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline

    subset=selected[selected.model.isin(FOCUS)]
    counts=pd.crosstab(subset.source_dataset,subset.model).reindex(
        columns=list(FOCUS),fill_value=0)
    sources=counts.index[(counts>=min_per_class).all(axis=1)]
    if len(sources)<3:
        raise ValueError("At least 3 sources with all 3 model labels are required")
    segments=[]
    for source in sources:
        for name in FOCUS:
            rows=subset[(subset.source_dataset==source)&(subset.model==name)]
            segments.append(rows.sample(n=min(len(rows),cap),random_state=seed))
    balanced=pd.concat(segments,ignore_index=True)
    def evaluate(train,test):
        if set(train.model)!=set(FOCUS) or set(test.model)!=set(FOCUS):
            raise ValueError("Split must contain all three models")
        classifier=make_pipeline(
            TfidfVectorizer(analyzer="char",ngram_range=(3,5),
                            min_df=2,max_features=30000,
                            sublinear_tf=True,lowercase=False),
            LogisticRegression(max_iter=800,random_state=seed))
        classifier.fit(train.text.astype(str),train.model)
        predicted=classifier.predict(test.text.astype(str))
        return {"n_train":int(len(train)),"n_test":int(len(test)),
                "bacc":float(balanced_accuracy_score(test.model,predicted)),
                "recall":{name:float(np.mean(
                    predicted[test.model.to_numpy()==name]==name))
                    for name in FOCUS}}
    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed)
    a,b=next(splitter.split(balanced,groups=balanced.prompt_id.astype(str)))
    random_split=evaluate(balanced.iloc[a],balanced.iloc[b])
    holdouts={}
    for source in sources:
        holdouts[str(source)]=evaluate(
            balanced[balanced.source_dataset!=source],
            balanced[balanced.source_dataset==source])
    official=official_train[official_train.model.isin(FOCUS)]
    official_result=evaluate(balanced,official) if set(official.model)==set(FOCUS) else None
    return {
        "focus_models":FOCUS,
        "eligible_sources":list(map(str,sources)),
        "external_selected_for_benchmark":int(len(balanced)),
        "prompt_group_holdout":random_split,
        "unseen_source_holdouts":holdouts,
        "unseen_source_mean_bacc":float(np.mean(
            [v["bacc"] for v in holdouts.values()])),
        "official_train_as_independent_external_model_test":official_result,
        "note":"A drop for unseen sources suggests source/domain effects. "
               "It is NOT proof of incorrect generator labels. Never tune on official val.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected",default="data/external/full/selected_3495_per_model.csv")
    parser.add_argument("--pool",default="data/external/ntl_v2/ntl_v3/ntl_open_pool.parquet")
    parser.add_argument("--train",default="outputs/dual-path-prompt-aware/train_split.csv")
    parser.add_argument("--out",default="outputs/fingerprint-v2/prompt-source-verification")
    parser.add_argument("--minimum",type=int,default=10)
    parser.add_argument("--benchmark",action="store_true",
                        help="Also run source-held-out CPU char n-gram classifier")
    parser.add_argument("--min-source-count",type=int,default=40)
    parser.add_argument("--cap",type=int,default=120)
    args=parser.parse_args()
    selected=read(args.selected)
    groups,matched,col=inspect_groups(selected,read(args.pool),args.minimum)
    dest=Path(args.out)
    dest.mkdir(parents=True,exist_ok=True)
    groups.to_csv(dest/"large_prompt_groups.csv",index=False)
    columns=[c for c in ("id","model","source_dataset","prompt_id",
                         "raw_prompt_group","raw_prompt_text") if c in matched]
    matched[columns].to_csv(dest/"raw_prompt_group_matches.csv",index=False)
    print("Original prompt text column:",col or "NONE (cannot verify groups)")
    print(groups.head(15).to_string(index=False))
    if args.benchmark:
        result=compare_source_generalization(
            selected,read(args.train),args.min_source_count,args.cap)
        (dest/"source_holdout.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
        print("Prompt-held-out BAcc:",result["prompt_group_holdout"]["bacc"])
        print("Unseen-source mean BAcc:",result["unseen_source_mean_bacc"])
    print("Saved under",dest)


if __name__=="__main__":
    main()
