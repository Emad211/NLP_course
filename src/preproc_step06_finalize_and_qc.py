import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "preprocess_steps" / "step05_tokenized.csv"

OUT_DIR = BASE_DIR / "processed_data" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FULL = OUT_DIR / "comments_preprocessed_full.csv"
OUT_RETRIEVER = OUT_DIR / "comments_for_tfidf_retriever.csv"
OUT_QC = OUT_DIR / "preprocess_qc_report.txt"


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()

    df = pd.read_csv(IN_CSV, keep_default_na=False)

    df["doctor_id"] = df["doctor_id"].astype(str)

    for col in ["text_step04", "tok_all", "tok_nostop"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    total = len(df)

    df["is_placeholder_negative"] = (df["label"] == -1) & (df["text_step04"].str.strip() == "عدم رضایت")

    df["is_tok_all_empty"] = df["tok_all"].str.strip().eq("")
    df["is_tok_nostop_empty"] = df["tok_nostop"].str.strip().eq("")

    df["text_for_tfidf"] = np.where(
        ~df["is_tok_nostop_empty"],
        df["tok_nostop"],
        df["tok_all"],
    ).astype(str)

    df["is_text_for_tfidf_empty"] = df["text_for_tfidf"].str.strip().eq("")

    df["clean_char_len"] = df["text_step04"].str.len()
    df["clean_word_len"] = df["text_step04"].str.split().str.len()

    dup_step04 = int(df["text_step04"].duplicated().sum())
    dup_tfidf = int(df["text_for_tfidf"].duplicated().sum())

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

    out_full_cols = [
        "doctor_id", "label", "rate", "date",
        "text_step04",
        "tok_all", "tok_nostop",
        "tok_all_count", "tok_nostop_count",
        "text_for_tfidf",
        "is_placeholder_negative",
        "is_tok_all_empty", "is_tok_nostop_empty", "is_text_for_tfidf_empty",
        "clean_char_len", "clean_word_len",
    ]
    for c in out_full_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[out_full_cols].to_csv(OUT_FULL, index=False, encoding="utf-8-sig")

    retr = df[~df["is_text_for_tfidf_empty"]].copy()
    retr_cols = [
        "doctor_id", "text_for_tfidf",
        "text_step04",
        "rate", "label", "date",
        "tok_all_count", "tok_nostop_count",
        "is_placeholder_negative",
    ]
    retr[retr_cols].to_csv(OUT_RETRIEVER, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("PREPROCESS QC REPORT (FINALIZED, v2)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_step05={IN_CSV.resolve()}")
    lines.append(f"output_full={OUT_FULL.resolve()}")
    lines.append(f"output_retriever={OUT_RETRIEVER.resolve()}")

    lines.append(section("0) Row counts"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_placeholder_negative={fmt_pct(int(df['is_placeholder_negative'].sum()), total)}")
    lines.append(f"rows_tok_all_empty={fmt_pct(int(df['is_tok_all_empty'].sum()), total)}")
    lines.append(f"rows_tok_nostop_empty={fmt_pct(int(df['is_tok_nostop_empty'].sum()), total)}")
    lines.append(f"rows_text_for_tfidf_empty={fmt_pct(int(df['is_text_for_tfidf_empty'].sum()), total)}")
    lines.append(f"retriever_rows_written={len(retr):,}")

    lines.append(section("1) Uniqueness and duplicates"))
    lines.append(f"unique_text_step04={df['text_step04'].nunique():,}")
    lines.append(f"unique_text_for_tfidf={df['text_for_tfidf'].nunique():,}")
    lines.append(f"duplicate_rows_text_step04={fmt_pct(dup_step04, total)}")
    lines.append(f"duplicate_rows_text_for_tfidf={fmt_pct(dup_tfidf, total)}")

    lines.append(section("2) By-label summary"))
    lines.append(by_label.to_string(index=False))

    dt = time.time() - t0
    lines.append(section("3) Runtime"))
    lines.append(f"seconds={dt:.2f}")

    OUT_QC.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved_qc_report={OUT_QC.resolve()}")


if __name__ == "__main__":
    main()