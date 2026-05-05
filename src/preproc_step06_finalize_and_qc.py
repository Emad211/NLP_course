# src/preproc_step06_finalize_and_qc.py
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from retriever_medical_stopwords import (
    MEDICAL_ALLOWLIST_LATIN,
    MEDICAL_STOPWORD_BIGRAMS_V01,
    MEDICAL_STOPWORDS_SEED_V02,
    filter_text_tokens_space_split,
)

BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "preprocess_steps" / "step05_tokenized.csv"

OUT_DIR = BASE_DIR / "processed_data" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing outputs (keep unchanged)
OUT_FULL = OUT_DIR / "comments_preprocessed_full.csv"
OUT_RETRIEVER = OUT_DIR / "comments_for_tfidf_retriever.csv"
OUT_QC = OUT_DIR / "preprocess_qc_report.txt"

# New outputs (medical retriever track)
OUT_RETRIEVER_MEDICAL = OUT_DIR / "comments_for_tfidf_retriever_medical.csv"
OUT_MED_QC = OUT_DIR / "retriever_medical_qc_report.txt"
OUT_MED_SAMPLES = OUT_DIR / "retriever_medical_samples.csv"
OUT_STOPWORDS = OUT_DIR / "retriever_medical_stopwords_seed_v02.txt"


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


def ensure_str_col(df: pd.DataFrame, col: str, default: str = "") -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].astype(str)


def ensure_num_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = np.nan


def build_medical_text_columns(df: pd.DataFrame) -> tuple[Counter, dict]:
    """
    Builds medical retriever columns from df["text_for_tfidf"] (already space-tokenized):
      - text_for_tfidf_medical
      - is_text_for_tfidf_medical_empty
      - medical_removed_token_count
      - medical_removed_tokens_sample
      - tfidf_token_count / tfidf_medical_token_count

    Returns:
      removed_counter_weighted: counts removed tokens weighted by row frequency
      meta: useful uniqueness stats
    """
    vc = df["text_for_tfidf"].astype(str).value_counts(dropna=False)

    med_map = {}
    removed_cnt_map = {}
    removed_sample_map = {}
    removed_counter_weighted = Counter()

    for txt, freq in vc.items():
        med_txt, removed = filter_text_tokens_space_split(
            txt,
            stopwords=MEDICAL_STOPWORDS_SEED_V02,
            allowlist_latin=MEDICAL_ALLOWLIST_LATIN,
            stopword_bigrams=MEDICAL_STOPWORD_BIGRAMS_V01,
        )

        med_map[txt] = med_txt
        removed_cnt_map[txt] = len(removed)

        # readable unique sample list
        if removed:
            seen = set()
            uniq_removed = []
            for r in removed:
                if r not in seen:
                    seen.add(r)
                    uniq_removed.append(r)
            removed_sample_map[txt] = " | ".join(uniq_removed[:30])
        else:
            removed_sample_map[txt] = ""

        if removed:
            for r in removed:
                removed_counter_weighted[r] += int(freq)

    df["text_for_tfidf_medical"] = df["text_for_tfidf"].map(med_map).astype(str)
    df["medical_removed_token_count"] = df["text_for_tfidf"].map(removed_cnt_map).astype(int)
    df["medical_removed_tokens_sample"] = df["text_for_tfidf"].map(removed_sample_map).astype(str)

    df["is_text_for_tfidf_medical_empty"] = df["text_for_tfidf_medical"].str.strip().eq("")

    df["tfidf_token_count"] = df["text_for_tfidf"].str.split().str.len()
    df["tfidf_medical_token_count"] = df["text_for_tfidf_medical"].str.split().str.len()

    meta = {
        "unique_text_for_tfidf": int(df["text_for_tfidf"].nunique(dropna=False)),
        "unique_text_for_tfidf_medical": int(df["text_for_tfidf_medical"].nunique(dropna=False)),
    }
    return removed_counter_weighted, meta


def write_medical_samples(df: pd.DataFrame, out_csv: Path, sample_n: int = 250, seed: int = 42) -> None:
    total = len(df)
    if total <= 0:
        pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
        return

    n = min(sample_n, total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(total), size=n, replace=False)

    sample_df = df.loc[idx, [
        "doctor_id", "label", "rate", "date",
        "text_step04",
        "text_for_tfidf",
        "text_for_tfidf_medical",
        "tfidf_token_count",
        "tfidf_medical_token_count",
        "medical_removed_token_count",
        "medical_removed_tokens_sample",
    ]].copy()

    sample_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()
    df = pd.read_csv(IN_CSV, keep_default_na=False)

    # --- Column hygiene ---
    ensure_str_col(df, "doctor_id", default="")
    df["doctor_id"] = df["doctor_id"].astype(str)

    for col in ["text_step04", "tok_all", "tok_nostop"]:
        ensure_str_col(df, col, default="")

    ensure_num_col(df, "label")
    ensure_num_col(df, "rate")
    ensure_str_col(df, "date", default="")

    ensure_num_col(df, "tok_all_count")
    ensure_num_col(df, "tok_nostop_count")

    total = len(df)

    # --- Flags ---
    df["is_placeholder_negative"] = (df["label"] == -1) & (df["text_step04"].str.strip() == "عدم رضایت")
    df["is_tok_all_empty"] = df["tok_all"].str.strip().eq("")
    df["is_tok_nostop_empty"] = df["tok_nostop"].str.strip().eq("")

    # --- Baseline retriever text (keep EXACT behavior) ---
    df["text_for_tfidf"] = np.where(
        ~df["is_tok_nostop_empty"],
        df["tok_nostop"],
        df["tok_all"],
    ).astype(str)

    df["is_text_for_tfidf_empty"] = df["text_for_tfidf"].str.strip().eq("")

    # --- Length stats ---
    df["clean_char_len"] = df["text_step04"].str.len()
    df["clean_word_len"] = df["text_step04"].str.split().str.len()

    dup_step04 = int(df["text_step04"].duplicated().sum())
    dup_tfidf = int(df["text_for_tfidf"].duplicated().sum())

    # --- By-label summary (baseline) ---
    by_label = (
        df.groupby("label", dropna=False)
        .agg(
            rows=("doctor_id", "size"),
            placeholder=("is_placeholder_negative", "sum"),
            tok_all_empty=("is_tok_all_empty", "sum"),
            tok_nostop_empty=("is_tok_nostop_empty", "sum"),
            tfidf_empty=("is_text_for_tfidf_empty", "sum"),
            mean_tok_all=("tok_all_count", "mean"),
            mean_tok_nostop=("tok_nostop_count", "mean"),
        )
        .reset_index()
    )

    # --- Medical stopwords used in this run (reproducibility) ---
    # store both unigrams and bigrams (readable)
    stop_lines = []
    stop_lines.append("# MEDICAL_STOPWORDS_SEED_V02 (unigrams)")
    stop_lines.extend(sorted(MEDICAL_STOPWORDS_SEED_V02))
    stop_lines.append("")
    stop_lines.append("# MEDICAL_STOPWORD_BIGRAMS_V01 (bigrams)")
    for a, b in sorted(MEDICAL_STOPWORD_BIGRAMS_V01):
        stop_lines.append(f"{a} {b}")
    OUT_STOPWORDS.write_text("\n".join(stop_lines), encoding="utf-8")

    # --- Medical retriever columns ---
    removed_counter_weighted, meta = build_medical_text_columns(df)
    dup_tfidf_med = int(df["text_for_tfidf_medical"].duplicated().sum())

    # --- Write OUT_FULL (keep old columns + add medical columns) ---
    out_full_cols = [
        "doctor_id", "label", "rate", "date",
        "text_step04",
        "tok_all", "tok_nostop",
        "tok_all_count", "tok_nostop_count",
        "text_for_tfidf",
        "is_placeholder_negative",
        "is_tok_all_empty", "is_tok_nostop_empty", "is_text_for_tfidf_empty",
        "clean_char_len", "clean_word_len",
        # new
        "text_for_tfidf_medical",
        "is_text_for_tfidf_medical_empty",
        "medical_removed_token_count",
        "medical_removed_tokens_sample",
        "tfidf_token_count",
        "tfidf_medical_token_count",
    ]
    for c in out_full_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[out_full_cols].to_csv(OUT_FULL, index=False, encoding="utf-8-sig")

    # --- Write baseline retriever csv (as before) ---
    retr = df[~df["is_text_for_tfidf_empty"]].copy()
    retr_cols = [
        "doctor_id", "text_for_tfidf",
        "text_step04",
        "rate", "label", "date",
        "tok_all_count", "tok_nostop_count",
        "is_placeholder_negative",
    ]
    for c in retr_cols:
        if c not in retr.columns:
            retr[c] = np.nan
    retr[retr_cols].to_csv(OUT_RETRIEVER, index=False, encoding="utf-8-sig")

    # --- Write medical retriever csv (drop-in for retriever build) ---
    retr_med = df[~df["is_text_for_tfidf_medical_empty"]].copy()
    retr_med["text_for_tfidf_orig"] = retr_med["text_for_tfidf"]
    retr_med["text_for_tfidf"] = retr_med["text_for_tfidf_medical"]

    retr_med_cols = [
        "doctor_id",
        "text_for_tfidf",          # medical
        "text_for_tfidf_orig",     # baseline audit
        "text_step04",
        "rate", "label", "date",
        "tok_all_count", "tok_nostop_count",
        "is_placeholder_negative",
        "medical_removed_token_count",
        "tfidf_token_count",
        "tfidf_medical_token_count",
    ]
    for c in retr_med_cols:
        if c not in retr_med.columns:
            retr_med[c] = np.nan
    retr_med[retr_med_cols].to_csv(OUT_RETRIEVER_MEDICAL, index=False, encoding="utf-8-sig")

    # --- PREPROCESS QC report (baseline-oriented, like before) ---
    lines = []
    lines.append("PREPROCESS QC REPORT (FINALIZED, v4 + medical retriever track)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_step05={IN_CSV.resolve()}")
    lines.append(f"output_full={OUT_FULL.resolve()}")
    lines.append(f"output_retriever_baseline={OUT_RETRIEVER.resolve()}")
    lines.append(f"output_retriever_medical={OUT_RETRIEVER_MEDICAL.resolve()}")
    lines.append(f"medical_stopwords_file={OUT_STOPWORDS.resolve()}")
    lines.append(f"medical_qc_report={OUT_MED_QC.resolve()}")
    lines.append(f"medical_samples_csv={OUT_MED_SAMPLES.resolve()}")

    lines.append(section("0) Row counts (baseline)"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_placeholder_negative={fmt_pct(int(df['is_placeholder_negative'].sum()), total)}")
    lines.append(f"rows_tok_all_empty={fmt_pct(int(df['is_tok_all_empty'].sum()), total)}")
    lines.append(f"rows_tok_nostop_empty={fmt_pct(int(df['is_tok_nostop_empty'].sum()), total)}")
    lines.append(f"rows_text_for_tfidf_empty={fmt_pct(int(df['is_text_for_tfidf_empty'].sum()), total)}")
    lines.append(f"retriever_rows_written_baseline={len(retr):,}")

    lines.append(section("1) Uniqueness and duplicates (baseline)"))
    lines.append(f"unique_text_step04={df['text_step04'].nunique():,}")
    lines.append(f"unique_text_for_tfidf={df['text_for_tfidf'].nunique():,}")
    lines.append(f"duplicate_rows_text_step04={fmt_pct(dup_step04, total)}")
    lines.append(f"duplicate_rows_text_for_tfidf={fmt_pct(dup_tfidf, total)}")

    lines.append(section("2) By-label summary (baseline)"))
    lines.append(by_label.to_string(index=False))

    dt = time.time() - t0
    lines.append(section("3) Runtime"))
    lines.append(f"seconds={dt:.2f}")

    OUT_QC.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved_qc_report={OUT_QC.resolve()}")

    # --- Medical QC report (retriever-focused) ---
    med_empty = int(df["is_text_for_tfidf_medical_empty"].sum())
    lines2 = []
    lines2.append("RETRIEVER MEDICAL QC REPORT (seed_v02 + bigrams)")
    lines2.append(f"project_root={BASE_DIR.resolve()}")
    lines2.append(f"input_step05={IN_CSV.resolve()}")
    lines2.append(f"baseline_retriever_csv={OUT_RETRIEVER.resolve()}")
    lines2.append(f"medical_retriever_csv={OUT_RETRIEVER_MEDICAL.resolve()}")
    lines2.append(f"stopwords_file={OUT_STOPWORDS.resolve()}")

    lines2.append(section("0) Coverage"))
    lines2.append(f"rows_total={total:,}")
    lines2.append(f"rows_text_for_tfidf_medical_empty={fmt_pct(med_empty, total)}")
    lines2.append(f"medical_retriever_rows_written={len(retr_med):,}")

    lines2.append(section("1) Token counts (before vs after medical stopwording)"))
    lines2.append(f"mean_tokens_before={df['tfidf_token_count'].mean():.2f}")
    lines2.append(f"mean_tokens_after ={df['tfidf_medical_token_count'].mean():.2f}")
    lines2.append(f"unique_text_for_tfidf={meta['unique_text_for_tfidf']:,}")
    lines2.append(f"unique_text_for_tfidf_medical={meta['unique_text_for_tfidf_medical']:,}")

    lines2.append(section("2) Duplicates (medical)"))
    lines2.append(f"unique_text_for_tfidf_medical={df['text_for_tfidf_medical'].nunique():,}")
    lines2.append(f"duplicate_rows_text_for_tfidf_medical={fmt_pct(dup_tfidf_med, total)}")

    lines2.append(section("3) Top removed tokens (weighted by rows)"))
    for tok, cnt in removed_counter_weighted.most_common(100):
        lines2.append(f"{tok}\t{cnt:,}")

    OUT_MED_QC.write_text("\n".join(lines2), encoding="utf-8")
    print(f"\nsaved_medical_qc_report={OUT_MED_QC.resolve()}")

    # --- Medical samples for manual audit ---
    write_medical_samples(df, OUT_MED_SAMPLES, sample_n=250, seed=42)
    print(f"saved_medical_samples={OUT_MED_SAMPLES.resolve()}")


if __name__ == "__main__":
    main()