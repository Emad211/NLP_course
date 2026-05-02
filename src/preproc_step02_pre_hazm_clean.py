import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV   = BASE_DIR / "processed_data" / "preprocess_steps" / "step01_char_normalized.csv"
OUT_DIR  = BASE_DIR / "processed_data" / "preprocess_steps"
OUT_CSV     = OUT_DIR / "step02_pre_hazm.csv"
OUT_REPORT  = OUT_DIR / "step02_report.txt"
OUT_SAMPLES = OUT_DIR / "step02_samples.csv"

_ELONG_RE      = re.compile(r"(\S)\1{2,}")
_PERS_PUNCT_RE = re.compile(r"[،؛؟!٪«»""]+")
_DOT_SAFE_RE   = re.compile(r"(?<!\d)\.(?!\d)")
_MULTI_SPACE   = re.compile(r"\s+")


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({n / total * 100:.1f}%)"


def desc(s: pd.Series) -> str:
    s = s.astype(float)
    p = np.percentile(s, [10, 50, 90])
    return f"min={s.min():.1f}  mean={s.mean():.1f}  p10={p[0]:.1f}  median={p[1]:.1f}  p90={p[2]:.1f}  max={s.max():.1f}"


def pre_hazm_clean(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(_ELONG_RE,      r"\1\1",  regex=True)
    s = s.str.replace(_PERS_PUNCT_RE, " ",       regex=True)
    s = s.str.replace(_DOT_SAFE_RE,   " ",       regex=True)
    s = s.str.replace(_MULTI_SPACE,   " ",       regex=True)
    s = s.str.strip()
    return s


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()
    df = pd.read_csv(IN_CSV)
    df["doctor_id"]   = df["doctor_id"].astype(str)
    df["text_step01"] = df["text_step01"].astype(str)

    total = len(df)

    has_elong_before = df["text_step01"].str.contains(_ELONG_RE, na=False)
    has_punct_before = df["text_step01"].str.contains(_PERS_PUNCT_RE, na=False)

    df["text_step02"] = pre_hazm_clean(df["text_step01"])

    has_elong_after = df["text_step02"].str.contains(_ELONG_RE, na=False)
    has_punct_after = df["text_step02"].str.contains(_PERS_PUNCT_RE, na=False)

    df["s01_char_len"]  = df["text_step01"].str.len()
    df["s02_char_len"]  = df["text_step02"].str.len()
    df["s01_word_len"]  = df["text_step01"].str.split().str.len()
    df["s02_word_len"]  = df["text_step02"].str.split().str.len()
    df["char_reduction"]= (df["s01_char_len"] - df["s02_char_len"]).clip(lower=0)
    df["changed"]       = df["text_step01"] != df["text_step02"]

    df["is_garbage"] = df["text_step02"].str.strip().eq("")

    placeholder = (df["label"] == -1) & (df["text_step02"] == "عدم رضایت")

    out_cols = [
        "doctor_id", "label", "rate", "date",
        "text_raw", "text_step01", "text_step02",
        "s01_char_len", "s02_char_len",
        "s01_word_len", "s02_word_len",
        "char_reduction", "changed", "is_garbage",
    ]
    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("PREPROCESS STEP 02 — PRE-HAZM CLEAN REPORT")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input={IN_CSV.resolve()}")
    lines.append(f"output={OUT_CSV.resolve()}")

    lines.append(section("0) Snapshot"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_changed={fmt_pct(int(df['changed'].sum()), total)}")
    lines.append(f"rows_garbage_empty={fmt_pct(int(df['is_garbage'].sum()), total)}")
    lines.append(f"rows_placeholder_negative={fmt_pct(int(placeholder.sum()), total)}")

    lines.append(section("1) Elongation  step01 -> step02"))
    lines.append(f"has_elongation_before={fmt_pct(int(has_elong_before.sum()), total)}")
    lines.append(f"has_elongation_after ={fmt_pct(int(has_elong_after.sum()), total)}")

    lines.append(section("2) Persian punctuation  step01 -> step02"))
    lines.append(f"has_persian_punct_before={fmt_pct(int(has_punct_before.sum()), total)}")
    lines.append(f"has_persian_punct_after ={fmt_pct(int(has_punct_after.sum()), total)}")

    lines.append(section("3) Length stats  step01 -> step02"))
    lines.append(f"char_len step01: {desc(df['s01_char_len'])}")
    lines.append(f"char_len step02: {desc(df['s02_char_len'])}")
    lines.append(f"word_len step01: {desc(df['s01_word_len'])}")
    lines.append(f"word_len step02: {desc(df['s02_word_len'])}")

    lines.append(section("4) Garbage rows (will be flagged, NOT deleted yet)"))
    garbage = df[df["is_garbage"]]
    lines.append(f"garbage_count={len(garbage):,}")
    for r in garbage.itertuples(index=False):
        lines.append(f"  doctor_id={r.doctor_id}  label={r.label}  rate={r.rate}  raw='{str(r.text_raw)[:80]}'")

    lines.append(section("5) Most affected samples (char_reduction top 20)"))
    top = df[df["changed"]].sort_values("char_reduction", ascending=False).head(20)
    for r in top.itertuples(index=False):
        s01 = str(r.text_step01).replace("\n", " ")[:200]
        s02 = str(r.text_step02).replace("\n", " ")[:200]
        lines.append(f"  reduce={int(r.char_reduction):>4}  label={r.label}  rate={r.rate}")
        lines.append(f"  STEP1: {s01}")
        lines.append(f"  STEP2: {s02}")
        lines.append("")

    lines.append(section("6) Random samples  step01 -> step02 (20 rows)"))
    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(total), size=min(20, total), replace=False)
    for i in idx:
        row = df.iloc[int(i)]
        s01 = str(row["text_step01"]).replace("\n", " ")[:200]
        s02 = str(row["text_step02"]).replace("\n", " ")[:200]
        lines.append(f"  [{int(i)}] label={row['label']}  rate={row['rate']}")
        lines.append(f"  STEP1: {s01}")
        lines.append(f"  STEP2: {s02}")
        lines.append("")

    dt = time.time() - t0
    lines.append(section("7) Runtime"))
    lines.append(f"seconds={dt:.2f}")

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)

    samples = df.iloc[idx][[
        "doctor_id", "label", "rate", "date",
        "text_step01", "text_step02", "char_reduction", "is_garbage"
    ]].copy()
    samples.to_csv(OUT_SAMPLES, index=False, encoding="utf-8-sig")

    print(f"\nsaved  report : {OUT_REPORT.resolve()}")
    print(f"saved  csv    : {OUT_CSV.resolve()}")
    print(f"saved  samples: {OUT_SAMPLES.resolve()}")


if __name__ == "__main__":
    main()