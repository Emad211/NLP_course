import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "comments_raw_canonical.csv"
OUT_TXT = BASE_DIR / "processed_data" / "text_quality_audit.txt"


URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?98|0)?\s*9\d{9}\b")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
PERSIAN_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
NON_STD_RE = re.compile(r"[^\u0600-\u06FF0-9A-Za-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


def normalize_text_key(text: str) -> str:
    if text is None:
        return ""
    t = str(text).replace("\u200c", " ").replace("\ufeff", " ")
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def char_stats(text: str):
    if not text:
        return 0, 0, 0, 0, 0, 0
    total = len(text)
    pers = len(PERSIAN_ARABIC_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    dig = len(DIGIT_RE.findall(text))
    nonstd = len(NON_STD_RE.findall(text))
    spaces = text.count(" ")
    return total, pers, lat, dig, nonstd, spaces


def top_items(counter: Counter, n: int):
    return counter.most_common(n)


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    df = pd.read_csv(IN_CSV)
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text"] = df["text"].astype(str)

    df["text_key"] = df["text"].apply(normalize_text_key)
    df["text_digits_norm"] = df["text_key"].astype(str).apply(lambda x: x.translate(DIGIT_MAP))

    total_rows = len(df)
    empty_rows = int(df["text_key"].str.strip().eq("").sum())

    has_url = df["text_digits_norm"].str.contains(URL_RE, na=False)
    has_email = df["text_digits_norm"].str.contains(EMAIL_RE, na=False)
    has_phone = df["text_digits_norm"].str.contains(PHONE_RE, na=False)
    has_latin = df["text_digits_norm"].str.contains(LATIN_RE, na=False)
    has_digit = df["text_digits_norm"].str.contains(DIGIT_RE, na=False)
    has_nonstd = df["text_digits_norm"].str.contains(NON_STD_RE, na=False)

    df["char_len"] = df["text_key"].str.len()
    df["word_len"] = df["text_key"].str.split().str.len()

    dup_rows = int(df["text_key"].duplicated().sum())
    uniq_texts = int(df["text_key"].nunique())

    placeholder = (df["label"] == -1) & (df["text_key"] == "عدم رضایت")
    placeholder_n = int(placeholder.sum())

    stats = df["text_digits_norm"].apply(char_stats)
    df["n_total"] = stats.apply(lambda x: x[0])
    df["n_pers"] = stats.apply(lambda x: x[1])
    df["n_lat"] = stats.apply(lambda x: x[2])
    df["n_dig"] = stats.apply(lambda x: x[3])
    df["n_nonstd"] = stats.apply(lambda x: x[4])
    df["n_spaces"] = stats.apply(lambda x: x[5])

    def ratio(a, b):
        b = b if b else 1
        return a / b

    df["ratio_pers"] = df.apply(lambda r: ratio(r["n_pers"], r["n_total"]), axis=1)
    df["ratio_lat"] = df.apply(lambda r: ratio(r["n_lat"], r["n_total"]), axis=1)
    df["ratio_nonstd"] = df.apply(lambda r: ratio(r["n_nonstd"], r["n_total"]), axis=1)

    lines = []
    lines.append("TEXT QUALITY AUDIT (CANONICAL, RAW)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_csv={IN_CSV.resolve()}")
    lines.append("")

    lines.append(section("0) Snapshot"))
    lines.append(f"rows={total_rows:,}")
    lines.append(f"empty_text_rows={fmt_pct(empty_rows, total_rows)}")
    lines.append(f"unique_text_key={uniq_texts:,}")
    lines.append(f"duplicate_text_rows={fmt_pct(dup_rows, total_rows)}")
    lines.append(f"placeholder_negative_rows={fmt_pct(placeholder_n, total_rows)}")

    lines.append(section("1) Pattern coverage"))
    lines.append(f"has_url={fmt_pct(int(has_url.sum()), total_rows)}")
    lines.append(f"has_email={fmt_pct(int(has_email.sum()), total_rows)}")
    lines.append(f"has_phone_like_mobile={fmt_pct(int(has_phone.sum()), total_rows)}")
    lines.append(f"has_latin_letters={fmt_pct(int(has_latin.sum()), total_rows)}")
    lines.append(f"has_digits={fmt_pct(int(has_digit.sum()), total_rows)}")
    lines.append(f"has_non_standard_chars={fmt_pct(int(has_nonstd.sum()), total_rows)}")

    lines.append(section("2) Length stats"))
    def desc(s):
        s = s.astype(float)
        return f"min={s.min():.1f} mean={s.mean():.1f} median={s.median():.1f} p90={np.percentile(s,90):.1f} max={s.max():.1f}"

    lines.append(f"char_len: {desc(df['char_len'])}")
    lines.append(f"word_len: {desc(df['word_len'])}")

    lines.append(section("3) Character composition ratios"))
    lines.append(f"ratio_pers: min={df['ratio_pers'].min():.3f} mean={df['ratio_pers'].mean():.3f} p10={np.percentile(df['ratio_pers'],10):.3f} p50={np.percentile(df['ratio_pers'],50):.3f} p90={np.percentile(df['ratio_pers'],90):.3f}")
    lines.append(f"ratio_lat : min={df['ratio_lat'].min():.3f} mean={df['ratio_lat'].mean():.3f} p10={np.percentile(df['ratio_lat'],10):.3f} p50={np.percentile(df['ratio_lat'],50):.3f} p90={np.percentile(df['ratio_lat'],90):.3f}")
    lines.append(f"ratio_nonstd: min={df['ratio_nonstd'].min():.3f} mean={df['ratio_nonstd'].mean():.3f} p10={np.percentile(df['ratio_nonstd'],10):.3f} p50={np.percentile(df['ratio_nonstd'],50):.3f} p90={np.percentile(df['ratio_nonstd'],90):.3f}")

    lines.append(section("4) Most frequent short texts (1-2 words, excluding placeholder)"))
    short = df[(df["word_len"] <= 2) & (~placeholder)].copy()
    vc_short = short["text_key"].value_counts().head(40)
    lines.append(f"short_rows_1_2_words_excluding_placeholder={fmt_pct(int(len(short)), total_rows)}")
    lines.append("")
    for t, c in vc_short.items():
        lines.append(f"count={int(c):>6,}\ttext='{t}'")

    lines.append(section("5) Longest texts"))
    long_df = df.sort_values(["word_len", "char_len"], ascending=False).head(20)
    for r in long_df.itertuples(index=False):
        preview = str(r.text_key).replace("\n", " ")[:260]
        lines.append(f"doctor_id={r.doctor_id}\tlabel={r.label}\trate={r.rate}\twords={int(r.word_len)}\tchars={int(r.char_len)}\ttext='{preview}'")

    lines.append(section("6) Latin token inventory (simple extraction)"))
    latin_rows = df[has_latin].copy()
    token_counter = Counter()
    for txt in latin_rows["text_digits_norm"].astype(str).tolist():
        parts = re.findall(r"[A-Za-z][A-Za-z0-9\._\-]{1,}", txt)
        token_counter.update([p.lower() for p in parts])

    for tok, cnt in top_items(token_counter, 60):
        lines.append(f"{tok}\t{cnt:,}")

    lines.append("\n" + "=" * 72)
    lines.append("END OF AUDIT")
    lines.append("=" * 72)

    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved_report={OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()