from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IN_LABELED = BASE_DIR / "processed_data" / "sentiment" / "labeling_pool_llm_labeled.csv"

OUT_DIR = BASE_DIR / "processed_data" / "sentiment"
OUT_ALL = OUT_DIR / "labeled_all_llm.csv"
OUT_TRAIN = OUT_DIR / "train_llm.csv"
OUT_DEV = OUT_DIR / "dev_llm.csv"
OUT_TEST = OUT_DIR / "test_llm.csv"
OUT_REPORT = OUT_DIR / "label_qc_report_llm.txt"

SEED = 42

def stratified_split(df, label_col="label_llm", train=0.8, dev=0.1, test=0.1, seed=42):
    rng = np.random.RandomState(seed)
    parts_train, parts_dev, parts_test = [], [], []
    for lab, g in df.groupby(label_col):
        g = g.sample(frac=1.0, random_state=rng)
        n = len(g)
        n_train = int(round(n * train))
        n_dev = int(round(n * dev))
        n_test = n - n_train - n_dev
        if n < 10:
            parts_train.append(g)
            continue
        parts_train.append(g.iloc[:n_train])
        parts_dev.append(g.iloc[n_train:n_train+n_dev])
        parts_test.append(g.iloc[n_train+n_dev:])

    train_df = pd.concat(parts_train).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    dev_df = pd.concat(parts_dev).sample(frac=1.0, random_state=rng).reset_index(drop=True) if parts_dev else df.iloc[0:0].copy()
    test_df = pd.concat(parts_test).sample(frac=1.0, random_state=rng).reset_index(drop=True) if parts_test else df.iloc[0:0].copy()
    return train_df, dev_df, test_df

def main():
    if not IN_LABELED.exists():
        raise FileNotFoundError(IN_LABELED)

    df = pd.read_csv(IN_LABELED, encoding="utf-8")
    df = df[df["label_llm"].notna()].copy()

    df["label_llm"] = df["label_llm"].astype(int)

    # optional: filter very low confidence
    # df = df[df["llm_confidence"].fillna(0) >= 0.2].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = stratified_split(df, seed=SEED)

    df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DEV, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

    vc = df["label_llm"].value_counts().sort_index()
    lines = []
    lines.append("LLM LABEL QC REPORT")
    lines.append(f"rows_total_labeled={len(df):,}")
    lines.append("label_dist:")
    for k, v in vc.items():
        lines.append(f"  {int(k)}: {int(v):,}")
    lines.append(f"train={len(train_df):,} dev={len(dev_df):,} test={len(test_df):,}")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()