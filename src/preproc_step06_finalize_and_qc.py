# src/preproc_step06_finalize_and_qc.py
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from retriever_medical_stopwords import (
    HIGH_RISK_KEEP,
    MEDICAL_ALLOWLIST_LATIN,
    MEDICAL_STOPWORD_BIGRAMS_V01,
    MEDICAL_STOPWORDS_SEED_V02,
    filter_text_tokens_space_split,
    load_medical_stopwords,
)

BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "preprocess_steps" / "step05_tokenized.csv"

OUT_DIR = BASE_DIR / "processed_data" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing outputs (keep unchanged)
OUT_FULL = OUT_DIR / "comments_preprocessed_full.csv"
OUT_RETRIEVER = OUT_DIR / "comments_for_tfidf_retriever.csv"
OUT_QC = OUT_DIR / "preprocess_qc_report.txt"

# Medical retriever track outputs
OUT_RETRIEVER_MEDICAL = OUT_DIR / "comments_for_tfidf_retriever_medical.csv"
OUT_MED_QC = OUT_DIR / "retriever_medical_qc_report.txt"
OUT_MED_SAMPLES = OUT_DIR / "retriever_medical_samples.csv"
OUT_STOPWORDS_USED = OUT_DIR / "retriever_medical_stopwords_used_v03.txt"

# The FINAL stopwords list produced by retriever_stopwords_finalize.py
STOPWORDS_FINAL_PATH = OUT_DIR / "retriever_medical_stopwords_final_v03.txt"


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


def build_medical_text_columns(df: pd.DataFrame, stopwords_used: set[str]) -> tuple[Counter, dict]:
    """
    Builds medical retriever columns from df["text_for_tfidf"] (already space-tokenized):
      - text_for_tfidf_medical
      - is_text_for_tfidf_medical_empty
      - medical_removed_token_count
      - medical_removed_tokens_sample
      - tfidf_token_count / tfidf_medical_token_count

    Returns:
      removed_counter_weighted: Counter removed tokens weighted by row frequency
      meta: useful stats (uniqueness etc.)
    """
    vc = df["text_for_tfidf"].astype(str).value_counts(dropna=False)

    med_map = {}
    removed_cnt_map = {}
    removed_sample_map = {}
    removed_counter_weighted = Counter()

    for txt, freq in vc.items():
        med_txt, removed = filter_text_tokens_space_split(
            txt,
            stopwords=stopwords_used,
            allowlist_latin=MEDICAL_ALLOWLIST_LATIN,
            stopword_bigrams=MEDICAL_STOPWORD_BIGRAMS_V01,
        )

        med_map[txt] = med_txt
        removed_cnt_map[txt] = len(removed)

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

    # --- Baseline retriever text (KEEP EXACT behavior) ---
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

    # ----------------------------
    # Load stopwords (FINAL v03 if exists; else fallback to seed)
    # ----------------------------
    stopwords_used, sw_meta = load_medical_stopwords(
        final_path=STOPWORDS_FINAL_PATH,
        fallback_seed=MEDICAL_STOPWORDS_SEED_V02,
    )

    # Write exactly what we USED (effective stopwords after safety filtering)
    sw_lines = []
    sw_lines.append("RETRIEVER MEDICAL STOPWORDS USED (effective, v03)")
    sw_lines.append(f"source={sw_meta.get('source')}")
    sw_lines.append(f"raw_count={sw_meta.get('raw_count')}")
    sw_lines.append(f"effective_count={sw_meta.get('effective_count')}")
    sw_lines.append("")
    sw_lines.append("# excluded_high_risk (never remove):")
    for t in sw_meta.get("excluded_high_risk", []):
        sw_lines.append(f"- {t}")
    sw_lines.append("")
    sw_lines.append("# stopwords (effective):")
    sw_lines.extend(sorted(stopwords_used))
    sw_lines.append("")
    sw_lines.append("# stopword bigrams (always applied):")
    for a, b in sorted(MEDICAL_STOPWORD_BIGRAMS_V01):
        sw_lines.append(f"{a} {b}")

    OUT_STOPWORDS_USED.write_text("\n".join(sw_lines), encoding="utf-8")

    # ----------------------------
    # Build medical text columns
    # ----------------------------
    removed_counter_weighted, meta = build_medical_text_columns(df, stopwords_used=stopwords_used)
    dup_tfidf_med = int(df["text_for_tfidf_medical"].duplicated().sum())

    # ---- Write OUT_FULL (keep old cols + add medical cols) ----
    out_full_cols = [
        "doctor_id", "label", "rate", "date",
        "text_step04",
        "tok_all", "tok_nostop",
        "tok_all_count", "tok_nostop_count",
        "text_for_tfidf",
        "is_placeholder_negative",
        "is_tok_all_empty", "is_tok_nostop_empty", "is_text_for_tfidf_empty",
        "clean_char_len", "clean_word_len",
        # medical
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

    # ---- Baseline retriever CSV (as before) ----
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

    # ---- Medical retriever CSV (drop-in for build) ----
    retr_med = df[~df["is_text_for_tfidf_medical_empty"]].copy()
    retr_med["text_for_tfidf_orig"] = retr_med["text_for_tfidf"]
    retr_med["text_for_tfidf"] = retr_med["text_for_tfidf_medical"]

    retr_med_cols = [
        "doctor_id",
        "text_for_tfidf",          # medical
        "text_for_tfidf_orig",     # baseline for audit
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

    # ----------------------------
    # PREPROCESS QC report (baseline oriented)
    # ----------------------------
    lines = []
    lines.append("PREPROCESS QC REPORT (FINALIZED, v5 + medical stopwords-from-file + safety guards)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_step05={IN_CSV.resolve()}")
    lines.append(f"output_full={OUT_FULL.resolve()}")
    lines.append(f"output_retriever_baseline={OUT_RETRIEVER.resolve()}")
    lines.append(f"output_retriever_medical={OUT_RETRIEVER_MEDICAL.resolve()}")
    lines.append(f"medical_stopwords_final_path={STOPWORDS_FINAL_PATH.resolve()}")
    lines.append(f"medical_stopwords_used_file={OUT_STOPWORDS_USED.resolve()}")
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

    # ----------------------------
    # Medical QC report (retriever oriented + safety checks)
    # ----------------------------
    med_empty = int(df["is_text_for_tfidf_medical_empty"].sum())

    # corpus-like counts (like build does): non-empty & non-placeholder
    baseline_use_mask = (df["text_for_tfidf"].str.strip().ne("")) & (~df["is_placeholder_negative"])
    medical_use_mask = (df["text_for_tfidf_medical"].str.strip().ne("")) & (~df["is_placeholder_negative"])

    baseline_rows_for_corpus = int(baseline_use_mask.sum())
    medical_rows_for_corpus = int(medical_use_mask.sum())

    baseline_doctors_for_corpus = int(df.loc[baseline_use_mask, "doctor_id"].nunique())
    medical_doctors_for_corpus = int(df.loc[medical_use_mask, "doctor_id"].nunique())

    # Safety: did we remove any HIGH_RISK_KEEP tokens? (should be 0 because of safety filter)
    removed_high_risk = []
    for t in sorted(HIGH_RISK_KEEP):
        c = removed_counter_weighted.get(t, 0)
        if c > 0:
            removed_high_risk.append((t, int(c)))

    lines2 = []
    lines2.append("RETRIEVER MEDICAL QC REPORT (v03 stopwords from file + safety guards)")
    lines2.append(f"project_root={BASE_DIR.resolve()}")
    lines2.append(f"input_step05={IN_CSV.resolve()}")
    lines2.append(f"baseline_retriever_csv={OUT_RETRIEVER.resolve()}")
    lines2.append(f"medical_retriever_csv={OUT_RETRIEVER_MEDICAL.resolve()}")
    lines2.append(f"stopwords_final_path={STOPWORDS_FINAL_PATH.resolve()}")
    lines2.append(f"stopwords_used_file={OUT_STOPWORDS_USED.resolve()}")
    lines2.append(f"stopwords_source={sw_meta.get('source')}")
    lines2.append(f"stopwords_raw_count={sw_meta.get('raw_count')}")
    lines2.append(f"stopwords_effective_count={sw_meta.get('effective_count')}")

    lines2.append(section("0) Coverage (rows)"))
    lines2.append(f"rows_total={total:,}")
    lines2.append(f"rows_text_for_tfidf_medical_empty={fmt_pct(med_empty, total)}")
    lines2.append(f"medical_retriever_rows_written={len(retr_med):,}")

    lines2.append(section("1) Coverage (corpus-like, comparable to build.py)"))
    lines2.append(f"baseline_rows_for_corpus(nonempty & !placeholder)={baseline_rows_for_corpus:,}")
    lines2.append(f"medical_rows_for_corpus (nonempty & !placeholder)={medical_rows_for_corpus:,}")
    lines2.append(f"baseline_doctors_for_corpus={baseline_doctors_for_corpus:,}")
    lines2.append(f"medical_doctors_for_corpus ={medical_doctors_for_corpus:,}")
    lines2.append(f"doctors_dropped={baseline_doctors_for_corpus - medical_doctors_for_corpus:,}")

    lines2.append(section("2) Token counts (before vs after medical stopwording)"))
    lines2.append(f"mean_tokens_before={df['tfidf_token_count'].mean():.2f}")
    lines2.append(f"mean_tokens_after ={df['tfidf_medical_token_count'].mean():.2f}")
    lines2.append(f"unique_text_for_tfidf={meta['unique_text_for_tfidf']:,}")
    lines2.append(f"unique_text_for_tfidf_medical={meta['unique_text_for_tfidf_medical']:,}")

    lines2.append(section("3) Duplicates (medical)"))
    lines2.append(f"unique_text_for_tfidf_medical={df['text_for_tfidf_medical'].nunique():,}")
    lines2.append(f"duplicate_rows_text_for_tfidf_medical={fmt_pct(dup_tfidf_med, total)}")

    lines2.append(section("4) Safety check — HIGH_RISK_KEEP removals (must be empty)"))
    if not removed_high_risk:
        lines2.append("OK: no high-risk medical core token was removed.")
    else:
        lines2.append("WARNING: high-risk tokens were removed (should not happen):")
        for t, c in removed_high_risk:
            lines2.append(f"{t}\t{c:,}")

    lines2.append(section("5) Top removed tokens (weighted by rows)"))
    for tok, cnt in removed_counter_weighted.most_common(120):
        lines2.append(f"{tok}\t{cnt:,}")

    OUT_MED_QC.write_text("\n".join(lines2), encoding="utf-8")
    print(f"\nsaved_medical_qc_report={OUT_MED_QC.resolve()}")

    # Samples
    write_medical_samples(df, OUT_MED_SAMPLES, sample_n=250, seed=42)
    print(f"saved_medical_samples={OUT_MED_SAMPLES.resolve()}")
    print(f"saved_stopwords_used={OUT_STOPWORDS_USED.resolve()}")


if __name__ == "__main__":
    main()