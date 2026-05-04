import re
from pathlib import Path
import numpy as np
import pandas as pd

# فیچرهای بی خود پاک بشن 
BASE_DIR = Path(__file__).resolve().parent.parent
DOCTORS_CSV = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"
COMMENTS_CSV = BASE_DIR / "processed_data" / "comments_raw_canonical.csv"
OUT_FILE = BASE_DIR / "processed_data" / "eda_report_canonical.txt"


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def norm_date(d) -> str:
    if d is None:
        return ""
    return str(d).translate(_DIGIT_MAP).strip()


def normalize_text_key(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = t.replace("\u200c", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))
    return pd.read_csv(path)


def label_name(x):
    return {-1: "Negative", 0: "Neutral", 1: "Positive"}.get(x, str(x))


def main():
    doctors = safe_read(DOCTORS_CSV)
    comments = safe_read(COMMENTS_CSV)

    doctors["doctor_id"] = doctors["doctor_id"].astype(str)
    comments["doctor_id"] = comments["doctor_id"].astype(str)

    comments["text"] = comments["text"].astype(str)
    comments["text_key"] = comments["text"].apply(normalize_text_key)

    lines = []
    lines.append("NOBAT DATASET — EDA REPORT (CANONICAL, RAW)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"doctors_csv={DOCTORS_CSV.resolve()}")
    lines.append(f"comments_csv={COMMENTS_CSV.resolve()}")
    lines.append("")

    lines.append(section("0) Snapshot"))
    lines.append(f"doctors_rows={len(doctors):,}")
    lines.append(f"comments_rows={len(comments):,}")
    lines.append(f"unique_doctor_ids_in_doctors={doctors['doctor_id'].nunique():,}")
    lines.append(f"unique_doctor_ids_in_comments={comments['doctor_id'].nunique():,}")

    lines.append(section("1) Key integrity checks"))
    dup_doctor_ids = int(doctors["doctor_id"].duplicated().sum())
    null_doctor_ids = int(doctors["doctor_id"].isna().sum())
    lines.append(f"doctors_doctor_id_null={fmt_pct(null_doctor_ids, len(doctors))}")
    lines.append(f"doctors_doctor_id_duplicate_rows={fmt_pct(dup_doctor_ids, len(doctors))}")

    doc_set = set(doctors["doctor_id"].unique().tolist())
    cdoc_set = set(comments["doctor_id"].unique().tolist())

    missing_in_doctors = sorted(list(cdoc_set - doc_set))
    missing_in_comments = sorted(list(doc_set - cdoc_set))

    lines.append(f"comment_doctor_ids_missing_in_doctors={len(missing_in_doctors):,}")
    if missing_in_doctors:
        lines.append("examples_missing_in_doctors=" + " | ".join(missing_in_doctors[:20]))

    lines.append(f"doctors_with_zero_comments={len(missing_in_comments):,}")
    if missing_in_comments:
        lines.append("examples_zero_comment_doctors=" + " | ".join(missing_in_comments[:20]))

    lines.append(section("2) Label & rating consistency"))
    total_c = len(comments)
    lines.append(f"label_null={fmt_pct(int(comments['label'].isna().sum()), total_c)}")
    lines.append(f"rate_null={fmt_pct(int(comments['rate'].isna().sum()), total_c)}")

    vc_label = comments["label"].value_counts(dropna=False).sort_index()
    lines.append("")
    lines.append("label_distribution:")
    for k, v in vc_label.items():
        lines.append(f"  label={label_name(k):<8} count={int(v):>8,} share={(v/total_c*100):>6.1f}%")

    rate = pd.to_numeric(comments["rate"], errors="coerce")
    lines.append("")
    lines.append(f"rate_unique_values={sorted(rate.dropna().unique().tolist())}")

    xt = (
        comments.assign(rate_num=rate)
        .groupby(["label", "rate_num"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["label", "rate_num"], ascending=[True, True])
    )
    lines.append("")
    lines.append("label_x_rate_counts:")
    lines.append(xt.to_string(index=False))

    lines.append(section("3) Placeholder negative"))
    is_placeholder = (comments["label"] == -1) & (comments["text_key"] == "عدم رضایت")
    ph_n = int(is_placeholder.sum())
    neg_n = int((comments["label"] == -1).sum())
    lines.append(f"negative_total={neg_n:,}")
    lines.append(f"placeholder_negative_total={ph_n:,}")
    if neg_n > 0:
        lines.append(f"placeholder_share_of_negative={(ph_n/neg_n*100):.2f}%")
    lines.append(f"placeholder_share_of_all={(ph_n/total_c*100):.2f}%")

    lines.append(section("4) Text quality (raw)"))
    char_len = comments["text_key"].str.len()
    word_len = comments["text_key"].str.split().str.len()
    is_empty = comments["text_key"].str.strip().eq("")
    dup_rows = comments["text_key"].duplicated()

    lines.append(f"empty_text={fmt_pct(int(is_empty.sum()), total_c)}")
    lines.append(f"duplicate_text_rows={fmt_pct(int(dup_rows.sum()), total_c)}")
    lines.append(f"very_short_lt3w={fmt_pct(int((word_len < 3).sum()), total_c)}")
    lines.append("")
    lines.append("length_stats_all:")
    lines.append(f"  char_len: min={char_len.min():.1f} mean={char_len.mean():.1f} median={char_len.median():.1f} p90={np.percentile(char_len,90):.1f} max={char_len.max():.1f}")
    lines.append(f"  word_len: min={word_len.min():.1f} mean={word_len.mean():.1f} median={word_len.median():.1f} p90={np.percentile(word_len,90):.1f} max={word_len.max():.1f}")

    lines.append("")
    lines.append("length_stats_by_label:")
    rows = []
    for lbl in [-1, 0, 1]:
        sub = comments[comments["label"] == lbl]
        if len(sub) == 0:
            continue
        cl = sub["text_key"].str.len()
        wl = sub["text_key"].str.split().str.len()
        rows.append(
            {
                "label": label_name(lbl),
                "count": len(sub),
                "char_mean": round(float(cl.mean()), 1),
                "char_p90": round(float(np.percentile(cl, 90)), 1),
                "word_mean": round(float(wl.mean()), 2),
                "word_p90": round(float(np.percentile(wl, 90)), 2),
            }
        )
    lines.append(pd.DataFrame(rows).to_string(index=False))

    lines.append(section("5) Duplicate text analysis (top repeated)"))
    counts = comments["text_key"].value_counts()
    lines.append(f"unique_text_key={comments['text_key'].nunique():,}")
    lines.append(f"duplicate_share={(1 - (comments['text_key'].nunique()/total_c))*100:.1f}%")
    lines.append("")
    lines.append("top_25_repeated_texts_overall:")
    for t, c in counts.head(25).items():
        preview = (t[:120] + "…") if len(t) > 120 else t
        lines.append(f"  count={int(c):>6,}  text='{preview}'")

    lines.append("")
    lines.append("top_10_repeated_texts_per_label:")
    for lbl in [-1, 0, 1]:
        sub = comments[comments["label"] == lbl]
        vc = sub["text_key"].value_counts().head(10)
        lines.append(f"\nlabel={label_name(lbl)} total={len(sub):,} unique={sub['text_key'].nunique():,}")
        for t, c in vc.items():
            preview = (t[:120] + "…") if len(t) > 120 else t
            lines.append(f"  count={int(c):>6,}  text='{preview}'")

    coverage = comments.groupby("text_key")["doctor_id"].nunique().rename("unique_doctors").to_frame()
    coverage["count"] = counts
    coverage = coverage.sort_values(["unique_doctors", "count"], ascending=False)

    lines.append("")
    lines.append("most_cross_doctor_repeated_texts:")
    for t, row in coverage.head(20).iterrows():
        preview = (t[:120] + "…") if len(t) > 120 else t
        lines.append(f"  doctors={int(row['unique_doctors']):>4}  count={int(row['count']):>6,}  text='{preview}'")

    lines.append(section("6) Comments per doctor"))
    cpd = comments.groupby("doctor_id").size()
    lines.append(f"doctors_with_any_comment={cpd.shape[0]:,}")
    lines.append(f"comments_per_doctor_min={int(cpd.min()):,}")
    lines.append(f"comments_per_doctor_mean={cpd.mean():.2f}")
    lines.append(f"comments_per_doctor_median={cpd.median():.2f}")
    lines.append(f"comments_per_doctor_p90={np.percentile(cpd.values,90):.2f}")
    lines.append(f"comments_per_doctor_max={int(cpd.max()):,}")

    lines.append(section("7) Specialties (raw doctors table)"))
    total_d = len(doctors)
    lines.append(f"unique_specialties={doctors['specialty'].nunique(dropna=True):,}")
    lines.append(f"specialty_null={fmt_pct(int(doctors['specialty'].isna().sum()), total_d)}")
    lines.append("")
    lines.append("top_30_specialties:")
    spec_counts = doctors["specialty"].value_counts(dropna=False).head(30)
    for spec, cnt in spec_counts.items():
        lines.append(f"  count={int(cnt):>4} share={(cnt/total_d*100):>5.1f}% specialty={str(spec)}")

    lines.append(section("8) Date coverage"))
    dates = comments["date"].dropna().astype(str).apply(norm_date)
    lines.append(f"date_non_null={fmt_pct(int(len(dates)), total_c)}")
    years = dates.str.extract(r"(\d{4})")[0].dropna()
    if len(years) > 0:
        yc = years.value_counts().sort_index()
        lines.append("")
        lines.append("year_counts:")
        for y, cnt in yc.items():
            lines.append(f"  {y}: {int(cnt):,}")

    lines.append(section("9) Samples per label"))
    rng = np.random.default_rng(42)
    for lbl in [-1, 0, 1]:
        sub = comments[comments["label"] == lbl]["text_key"].dropna().astype(str)
        lines.append(f"\nlabel={label_name(lbl)} total={len(sub):,}")
        if len(sub) == 0:
            continue
        idx = rng.choice(np.arange(len(sub)), size=min(6, len(sub)), replace=False)
        for t in sub.iloc[idx].tolist():
            preview = t.replace("\n", " ").strip()
            preview = preview[:200]
            lines.append(f"  - {preview}")

    lines.append("\n" + "=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)

    report = "\n".join(lines)
    OUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved_report={OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()