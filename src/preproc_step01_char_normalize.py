import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV   = BASE_DIR / "processed_data" / "comments_raw_canonical.csv"
OUT_DIR  = BASE_DIR / "processed_data" / "preprocess_steps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV     = OUT_DIR / "step01_char_normalized.csv"
OUT_REPORT  = OUT_DIR / "step01_report.txt"
OUT_SAMPLES = OUT_DIR / "step01_samples.csv"

_DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789"
)

_CONTROL_CHARS_RE = re.compile(
    r"[\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u00ad\u2060\u180e]"
)
_KEEP_CHARS_RE = re.compile(r"[^0-9A-Za-z\u0600-\u06FF\s]")
_ELONG_RE      = re.compile(r"(\S)\1{2,}")
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
    return (
        f"min={s.min():.1f}  mean={s.mean():.1f}  "
        f"p10={p[0]:.1f}  median={p[1]:.1f}  "
        f"p90={p[2]:.1f}  max={s.max():.1f}"
    )


def char_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(str)

    s = s.apply(lambda x: _CONTROL_CHARS_RE.sub(" ", x))

    s = s.apply(lambda x: x.translate(_DIGIT_MAP))

    s = s.str.replace("ي", "ی", regex=False)
    s = s.str.replace("ك", "ک", regex=False)
    s = s.str.replace("أ", "ا", regex=False)
    s = s.str.replace("إ", "ا", regex=False)
    s = s.str.replace("ؤ", "و", regex=False)
    s = s.str.replace("ة", "ه", regex=False)
    s = s.str.replace("ٱ", "ا", regex=False)

    s = s.str.replace(_KEEP_CHARS_RE, " ", regex=True)

    s = s.str.replace(_ELONG_RE, r"\1\1", regex=True)

    s = s.str.replace(_MULTI_SPACE, " ", regex=True)
    s = s.str.strip()

    return s


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()

    df = pd.read_csv(IN_CSV)
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text"]      = df["text"].astype(str)

    total = len(df)
    print(f"rows={total:,}", flush=True)

    df["text_raw"]    = df["text"]
    df["text_step01"] = char_normalize(df["text"])

    df["raw_char_len"]   = df["text_raw"].str.len()
    df["s01_char_len"]   = df["text_step01"].str.len()
    df["raw_word_len"]   = df["text_raw"].str.split().str.len()
    df["s01_word_len"]   = df["text_step01"].str.split().str.len()
    df["char_reduction"] = (df["raw_char_len"] - df["s01_char_len"]).clip(lower=0)
    df["changed"]        = df["text_raw"] != df["text_step01"]
    df["empty_after"]    = df["text_step01"].str.strip().eq("")

    _LATIN_RE    = re.compile(r"[A-Za-z]")
    _DIGIT_RE    = re.compile(r"\d")
    _ELONG_CHECK = re.compile(r"(\S)\1{2,}")

    has_control_raw  = df["text_raw"].str.contains(_CONTROL_CHARS_RE, na=False)
    has_nonstd_raw   = df["text_raw"].str.contains(_KEEP_CHARS_RE, na=False)
    has_nonstd_s01   = df["text_step01"].str.contains(_KEEP_CHARS_RE, na=False)
    has_elong_raw    = df["text_raw"].str.contains(_ELONG_CHECK, na=False)
    has_elong_s01    = df["text_step01"].str.contains(_ELONG_CHECK, na=False)
    has_latin_raw    = df["text_raw"].str.contains(_LATIN_RE, na=False)
    has_latin_s01    = df["text_step01"].str.contains(_LATIN_RE, na=False)
    has_digit_raw    = df["text_raw"].str.contains(_DIGIT_RE, na=False)
    has_digit_s01    = df["text_step01"].str.contains(_DIGIT_RE, na=False)

    arabic_chars     = ["ي", "ك", "أ", "إ", "ئ", "ؤ", "ة", "ٱ"]
    _ARABIC_CHECK    = re.compile("|".join(arabic_chars))
    has_arabic_raw   = df["text_raw"].str.contains(_ARABIC_CHECK, na=False)
    has_arabic_s01   = df["text_step01"].str.contains(_ARABIC_CHECK, na=False)

    placeholder = (df["label"] == -1) & (df["text_step01"] == "عدم رضایت")

    out_cols = [
        "doctor_id", "label", "rate", "date",
        "text_raw", "text_step01",
        "raw_char_len", "s01_char_len",
        "raw_word_len", "s01_word_len",
        "char_reduction", "changed", "empty_after",
    ]
    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("PREPROCESS STEP 01 — CHAR NORMALIZATION REPORT (v2)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input={IN_CSV.resolve()}")
    lines.append(f"output={OUT_CSV.resolve()}")

    lines.append(section("0) Snapshot"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_changed={fmt_pct(int(df['changed'].sum()), total)}")
    lines.append(f"rows_empty_after={fmt_pct(int(df['empty_after'].sum()), total)}")
    lines.append(f"rows_placeholder_negative={fmt_pct(int(placeholder.sum()), total)}")

    lines.append(section("1) Pattern coverage  raw -> step01"))
    lines.append(f"has_control_chars      : {fmt_pct(int(has_control_raw.sum()), total)}  ->  0 (removed)")
    lines.append(f"has_arabic_variants    : {fmt_pct(int(has_arabic_raw.sum()), total)}  ->  {fmt_pct(int(has_arabic_s01.sum()), total)}")
    lines.append(f"has_non_standard_chars : {fmt_pct(int(has_nonstd_raw.sum()), total)}  ->  {fmt_pct(int(has_nonstd_s01.sum()), total)}")
    lines.append(f"has_elongation (>=3x)  : {fmt_pct(int(has_elong_raw.sum()), total)}  ->  {fmt_pct(int(has_elong_s01.sum()), total)}")
    lines.append(f"has_latin_letters      : {fmt_pct(int(has_latin_raw.sum()), total)}  ->  {fmt_pct(int(has_latin_s01.sum()), total)}")
    lines.append(f"has_digits             : {fmt_pct(int(has_digit_raw.sum()), total)}  ->  {fmt_pct(int(has_digit_s01.sum()), total)}")

    lines.append(section("2) Length stats  raw -> step01"))
    lines.append(f"char_len raw   : {desc(df['raw_char_len'])}")
    lines.append(f"char_len step01: {desc(df['s01_char_len'])}")
    lines.append(f"word_len raw   : {desc(df['raw_word_len'])}")
    lines.append(f"word_len step01: {desc(df['s01_word_len'])}")

    lines.append(section("3) Most affected samples (char_reduction top 20)"))
    top = df.sort_values("char_reduction", ascending=False).head(20)
    for r in top.itertuples(index=False):
        raw = str(r.text_raw).replace("\n", " ")[:200]
        s01 = str(r.text_step01).replace("\n", " ")[:200]
        lines.append(f"  reduce={int(r.char_reduction):>4}  label={r.label}  rate={r.rate}")
        lines.append(f"  RAW  : {raw}")
        lines.append(f"  STEP1: {s01}")
        lines.append("")

    lines.append(section("4) Empty-after samples"))
    empties = df[df["empty_after"]]
    if len(empties) == 0:
        lines.append("  none")
    else:
        for r in empties.itertuples(index=False):
            lines.append(
                f"  doctor_id={r.doctor_id}  label={r.label}  "
                f"rate={r.rate}  raw='{str(r.text_raw)[:100]}'"
            )

    lines.append(section("5) Arabic variant samples (before -> after)"))
    arabic_changed = df[has_arabic_raw & df["changed"]].head(15)
    if len(arabic_changed) == 0:
        lines.append("  none")
    else:
        for r in arabic_changed.itertuples(index=False):
            raw = str(r.text_raw).replace("\n", " ")[:200]
            s01 = str(r.text_step01).replace("\n", " ")[:200]
            lines.append(f"  RAW  : {raw}")
            lines.append(f"  STEP1: {s01}")
            lines.append("")

    lines.append(section("6) Random samples (25)"))
    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(total), size=min(25, total), replace=False)
    for i in idx:
        row = df.iloc[int(i)]
        raw = str(row["text_raw"]).replace("\n", " ")[:200]
        s01 = str(row["text_step01"]).replace("\n", " ")[:200]
        lines.append(f"  [{int(i)}] label={row['label']}  rate={row['rate']}")
        lines.append(f"  RAW  : {raw}")
        lines.append(f"  STEP1: {s01}")
        lines.append("")

    dt = time.time() - t0
    lines.append(section("7) Runtime"))
    lines.append(f"seconds={dt:.2f}")

    lines.append(section("8) What this step did (summary)"))
    lines.append("  A. Removed control chars (ZWNJ, ZWNBSP, directional marks, soft-hyphen)")
    lines.append("  B. Normalized Persian/Arabic digits to Latin (0-9)")
    lines.append("  C. Unified Arabic character variants:")
    lines.append("       ي→ی  ك→ک  أ→ا  إ→ا  ئ→ی  ؤ→و  ة→ه  ٱ→ا")
    lines.append("  D. Removed all non-standard chars (emoji, symbols, punctuation)")
    lines.append("  E. Reduced elongation: any char repeated >=3x → 2x")
    lines.append("  F. Collapsed multiple spaces to single space")
    lines.append("")
    lines.append("  NOT done in this step (reserved for later steps):")
    lines.append("  - Persian punctuation removal (،؛؟) → Step 02")
    lines.append("  - Hazm normalization / correct_spacing  → Step 03")
    lines.append("  - Tokenization / stopword removal       → Step 04")

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)

    samples = df.iloc[idx][[
        "doctor_id", "label", "rate", "date",
        "text_raw", "text_step01",
        "raw_word_len", "s01_word_len", "char_reduction"
    ]].copy()
    samples.to_csv(OUT_SAMPLES, index=False, encoding="utf-8-sig")

    print(f"\nsaved  report : {OUT_REPORT.resolve()}")
    print(f"saved  csv    : {OUT_CSV.resolve()}")
    print(f"saved  samples: {OUT_SAMPLES.resolve()}")


if __name__ == "__main__":
    main()