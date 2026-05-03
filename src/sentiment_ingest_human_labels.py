from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IN_BATCH = BASE_DIR / "processed_data" / "sentiment" / "labeling_batch_01.csv"

OUT_DIR = BASE_DIR / "processed_data" / "sentiment"
OUT_ALL = OUT_DIR / "labeled_all.csv"
OUT_TRAIN = OUT_DIR / "train.csv"
OUT_DEV = OUT_DIR / "dev.csv"
OUT_TEST = OUT_DIR / "test.csv"
OUT_REPORT = OUT_DIR / "label_qc_report.txt"

SEED = 42

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_ARABIC_DIGITS  = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(s: str) -> str:
    return s.translate(_PERSIAN_DIGITS).translate(_ARABIC_DIGITS)


def parse_label(x):
    """
    Accepts:
    - -1 / 0 / 1
    - -1.0 / 0.0 / 1.0
    - Persian digits: ۱، ۰، -۱
    - Strings: neg/negative/منفی, neu/neutral/خنثی, pos/positive/مثبت
    """
    if pd.isna(x):
        return None

    # numeric path (int/float)
    if isinstance(x, (int, np.integer)):
        return int(x) if int(x) in (-1, 0, 1) else None
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        xi = int(round(float(x)))
        return xi if xi in (-1, 0, 1) and abs(float(x) - xi) < 1e-6 else None

    s = normalize_digits(str(x)).strip().lower()
    if s == "":
        return None

    # allow "-1.0"
    try:
        fv = float(s)
        iv = int(round(fv))
        if iv in (-1, 0, 1) and abs(fv - iv) < 1e-6:
            return iv
    except Exception:
        pass

    if s in {"-1", "neg", "negative", "منفی"}:
        return -1
    if s in {"0", "neu", "neutral", "خنثی", "mixed"}:
        return 0
    if s in {"1", "pos", "positive", "مثبت"}:
        return 1

    return None


def safe_split_per_class(g: pd.DataFrame, train=0.8, dev=0.1, test=0.1, seed=42):
    """
    Split one class group safely.
    If too small, keep all in train.
    """
    rng = np.random.RandomState(seed)
    g = g.sample(frac=1.0, random_state=rng)

    n = len(g)
    if n < 5:
        return g, g.iloc[0:0], g.iloc[0:0]  # all train

    n_test = max(1, int(round(n * test)))
    n_dev = max(1, int(round(n * dev)))
    n_train = n - n_dev - n_test

    # ensure at least 1 in train
    if n_train <= 0:
        # shrink dev then test if needed
        while n_train <= 0 and n_dev > 0:
            n_dev -= 1
            n_train = n - n_dev - n_test
        while n_train <= 0 and n_test > 0:
            n_test -= 1
            n_train = n - n_dev - n_test

    if n_train <= 0:
        # ultimate fallback
        return g, g.iloc[0:0], g.iloc[0:0]

    tr = g.iloc[:n_train]
    dv = g.iloc[n_train:n_train + n_dev]
    te = g.iloc[n_train + n_dev:n_train + n_dev + n_test]
    return tr, dv, te


def stratified_split_safe(df: pd.DataFrame, label_col="label_human_int", seed=42):
    parts_train, parts_dev, parts_test = [], [], []
    for lab, g in df.groupby(label_col):
        tr, dv, te = safe_split_per_class(g, seed=seed + int(lab) + 10)
        if len(tr): parts_train.append(tr)
        if len(dv): parts_dev.append(dv)
        if len(te): parts_test.append(te)

    # these should never be empty if df is non-empty, but keep safe:
    train_df = pd.concat(parts_train, ignore_index=True) if parts_train else df.copy()
    dev_df = pd.concat(parts_dev, ignore_index=True) if parts_dev else df.iloc[0:0].copy()
    test_df = pd.concat(parts_test, ignore_index=True) if parts_test else df.iloc[0:0].copy()

    rng = np.random.RandomState(seed)
    train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    dev_df = dev_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return train_df, dev_df, test_df


def main():
    if not IN_BATCH.exists():
        raise FileNotFoundError(IN_BATCH)

    df = pd.read_csv(IN_BATCH, encoding="utf-8")

    # Ensure expected columns exist
    for c in ["task_id", "item_id", "text_label", "label_human", "is_duplicate"]:
        if c not in df.columns:
            df[c] = "" if c != "is_duplicate" else 0

    df["label_human_int"] = df["label_human"].apply(parse_label)

    total = len(df)
    labeled = int(df["label_human_int"].notna().sum())

    if labeled == 0:
        msg = (
            "HUMAN LABEL QC REPORT\n"
            f"input={IN_BATCH.resolve()}\n"
            f"rows_total={total:,}\n"
            "rows_labeled=0\n\n"
            "ERROR: No parseable labels found in column 'label_human'.\n"
            "Fix: fill label_human with -1/0/1 (or neg/neu/pos) and re-save CSV.\n"
            "Also acceptable: -1.0/0.0/1.0.\n"
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(msg, encoding="utf-8")
        print(msg)
        return

    # Duplicate QA (self-consistency)
    dup_df = df[df["is_duplicate"].fillna(0).astype(int) == 1].copy()
    dup_groups = dup_df[dup_df["label_human_int"].notna()].groupby("item_id")["label_human_int"].apply(list)

    agree = 0
    disagree = 0
    for item_id, labs in dup_groups.items():
        if len(labs) < 2:
            continue
        if len(set(labs)) == 1:
            agree += 1
        else:
            disagree += 1

    df_labeled = df[df["label_human_int"].notna()].copy()

    # Trainable: one row per item_id (collapse duplicates)
    df_trainable = df_labeled.sort_values(["item_id", "task_id"]).drop_duplicates("item_id", keep="first").copy()

    # Basic sanity: need at least 2 classes to train a sentiment classifier
    classes_present = sorted(df_trainable["label_human_int"].unique().tolist())
    if len(classes_present) < 2:
        msg = (
            "HUMAN LABEL QC REPORT\n"
            f"input={IN_BATCH.resolve()}\n"
            f"rows_total={total:,}\n"
            f"rows_labeled={labeled:,}\n"
            f"rows_trainable_unique_item_id={len(df_trainable):,}\n\n"
            f"ERROR: Only one class present: {classes_present}\n"
            "Need at least two labels among {-1,0,1} to train.\n"
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(msg, encoding="utf-8")
        print(msg)
        return

    # Split safely
    train_df, dev_df, test_df = stratified_split_safe(df_trainable, seed=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_trainable.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    dev_df.to_csv(OUT_DEV, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

    # Report
    lines = []
    lines.append("HUMAN LABEL QC REPORT")
    lines.append(f"input={IN_BATCH.resolve()}")
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_labeled={labeled:,}")
    lines.append(f"rows_trainable_unique_item_id={len(df_trainable):,}")
    lines.append(f"classes_present={classes_present}")
    lines.append("")
    lines.append("Label distribution (trainable):")
    vc = df_trainable["label_human_int"].value_counts().sort_index()
    for k, v in vc.items():
        lines.append(f"  {int(k)}: {int(v):,}")
    lines.append("")
    lines.append("Split sizes:")
    lines.append(f"  train={len(train_df):,}")
    lines.append(f"  dev={len(dev_df):,}")
    lines.append(f"  test={len(test_df):,}")
    lines.append("")
    lines.append("Duplicate QA (self-consistency):")
    lines.append(f"  duplicate_groups_with_2+labels_agree={agree}")
    lines.append(f"  duplicate_groups_with_2+labels_disagree={disagree}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()