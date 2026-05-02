import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_FILE = BASE_DIR / "eda_report.txt"


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


def load_all(data_dir: Path):
    doctor_rows = []
    comment_rows = []
    failed = []

    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {data_dir.resolve()}")

    for fpath in files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            failed.append((fpath.name, str(e)))
            continue

        p = data.get("doctor_profile", {}) or {}
        doc_id = p.get("doctor_id")
        doc_id = str(doc_id) if doc_id is not None else None

        doctor_rows.append(
            {
                "doctor_id": doc_id,
                "source_file": fpath.name,
                "name": p.get("name"),
                "specialty": p.get("specialty"),
                "verified": bool(p.get("verified", False)),
                "followers_count": p.get("followers_count"),
                "reviews_count": p.get("reviews_count"),
                "rating_percent": p.get("rating_percent"),
                "rating_score_5": p.get("rating_score_5"),
                "biography": p.get("biography") or "",
                "office_count": len(p.get("offices", []) or []),
            }
        )

        for c in data.get("comments", []) or []:
            comment_rows.append(
                {
                    "doctor_id": doc_id,
                    "specialty": p.get("specialty"),
                    "text": c.get("text", ""),
                    "rate": c.get("rate"),
                    "label": c.get("label"),
                    "date": c.get("date"),
                }
            )

    doctors = pd.DataFrame(doctor_rows)
    comments = pd.DataFrame(comment_rows)
    return doctors, comments, failed


def analyze(doctors: pd.DataFrame, comments: pd.DataFrame, failed: list) -> list[str]:
    lines = []
    lines.append("NOBAT DATASET — EDA REPORT (RAW, NO PREPROCESSING)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"data_dir={DATA_DIR.resolve()}")
    lines.append("")

    lines.append(section("1) File loading"))
    total_files = len(list(DATA_DIR.glob("*.json")))
    loaded_files = total_files - len(failed)
    lines.append(f"json_files_total={total_files:,}")
    lines.append(f"json_files_loaded={loaded_files:,}")
    lines.append(f"json_files_failed={len(failed):,}")
    if failed:
        for name, err in failed[:20]:
            lines.append(f"FAIL {name} -> {err}")
        if len(failed) > 20:
            lines.append(f"... {len(failed)-20} more failures")

    lines.append(section("2) High-level shape"))
    lines.append(f"doctors_rows={len(doctors):,}")
    lines.append(f"comments_rows={len(comments):,}")
    lines.append(f"unique_doctor_ids_in_doctors={doctors['doctor_id'].nunique(dropna=True):,}")
    lines.append(f"unique_doctor_ids_in_comments={comments['doctor_id'].nunique(dropna=True):,}")

    lines.append(section("3) Doctor ID integrity"))
    null_doc_id = doctors["doctor_id"].isna().sum()
    dup_doc_id = doctors["doctor_id"].duplicated().sum()
    lines.append(f"doctor_id_null={fmt_pct(int(null_doc_id), len(doctors))}")
    lines.append(f"doctor_id_duplicate_rows={fmt_pct(int(dup_doc_id), len(doctors))}")
    if dup_doc_id > 0:
        top_dups = doctors["doctor_id"].value_counts(dropna=True).head(10)
        lines.append("")
        lines.append("top_duplicate_doctor_ids:")
        for k, v in top_dups.items():
            if v > 1:
                lines.append(f"  doctor_id={k}  count={int(v):,}")

    lines.append(section("4) Doctor profile missingness"))
    doc_cols = [
        "name",
        "specialty",
        "verified",
        "followers_count",
        "reviews_count",
        "rating_percent",
        "rating_score_5",
        "biography",
        "office_count",
    ]
    for col in doc_cols:
        null_n = int(doctors[col].isna().sum())
        if col == "biography":
            empty_n = int((doctors[col].astype(str).str.strip() == "").sum())
            lines.append(f"{col}_null={fmt_pct(null_n, len(doctors))}  empty_string={fmt_pct(empty_n, len(doctors))}")
        else:
            lines.append(f"{col}_null={fmt_pct(null_n, len(doctors))}")

    lines.append(section("5) Doctor numeric stats"))
    num_cols = ["followers_count", "reviews_count", "rating_percent", "rating_score_5", "office_count"]
    for col in num_cols:
        s = pd.to_numeric(doctors[col], errors="coerce").dropna()
        if len(s) == 0:
            lines.append(f"{col}: no data")
        else:
            lines.append(
                f"{col}: min={s.min():.2f} mean={s.mean():.2f} median={s.median():.2f} p90={np.percentile(s,90):.2f} max={s.max():.2f}"
            )

    lines.append(section("6) Specialties (raw)"))
    total_d = len(doctors)
    lines.append(f"unique_specialties={doctors['specialty'].nunique(dropna=True):,}")
    lines.append(f"specialty_null={fmt_pct(int(doctors['specialty'].isna().sum()), total_d)}")
    spec_counts = doctors["specialty"].value_counts(dropna=False).head(30)
    lines.append("")
    lines.append("top_30_specialties:")
    for spec, cnt in spec_counts.items():
        lines.append(f"  count={int(cnt):>4}  share={(cnt/total_d*100):>5.1f}%  specialty={str(spec)}")

    lines.append(section("7) Comment missingness"))
    total_c = len(comments)
    for col in ["text", "rate", "label", "date"]:
        null_n = int(comments[col].isna().sum())
        lines.append(f"{col}_null={fmt_pct(null_n, total_c)}")

    lines.append(section("8) Label distribution"))
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    vc = comments["label"].value_counts(dropna=False).sort_index()
    for k, v in vc.items():
        lines.append(f"label={label_map.get(k, str(k)):<8} count={int(v):>8,} share={(v/total_c*100):>6.1f}%")

    non_null = comments["label"].dropna()
    if len(non_null) > 0:
        counts = non_null.value_counts()
        if len(counts) >= 2:
            lines.append(f"\nimbalance_ratio_majority_to_minority={(counts.max()/counts.min()):.2f}x")

    lines.append(section("9) Rating distribution"))
    rates = pd.to_numeric(comments["rate"], errors="coerce").dropna()
    lines.append(f"rate_non_null={fmt_pct(int(len(rates)), total_c)}")
    if len(rates) > 0:
        lines.append(f"rate_unique_values={sorted(rates.unique().tolist())}")
        lines.append(f"rate_min={rates.min():.3f}")
        lines.append(f"rate_mean={rates.mean():.3f}")
        lines.append(f"rate_median={rates.median():.3f}")
        lines.append(f"rate_max={rates.max():.3f}")
        bins = [(0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
        lines.append("")
        lines.append("rate_bins:")
        for lo, hi in bins:
            if lo == hi:
                cnt = int((rates == lo).sum())
                lines.append(f"  eq_{lo:g} => {cnt:,} ({(cnt/len(rates)*100):.1f}%)")
            else:
                if hi == 5.0:
                    cnt = int(((rates > lo) & (rates <= hi)).sum())
                else:
                    cnt = int(((rates > lo) & (rates <= hi)).sum())
                lines.append(f"  ({lo:g},{hi:g}] => {cnt:,} ({(cnt/len(rates)*100):.1f}%)")

    lines.append(section("10) Comment text quality (raw)"))
    text = comments["text"].astype(str)
    char_len = text.str.len()
    word_len = text.str.split().str.len()
    is_empty = text.str.strip().eq("")
    dup = text.duplicated()
    very_short = word_len < 3

    lines.append(f"empty_text={fmt_pct(int(is_empty.sum()), total_c)}")
    lines.append(f"duplicate_text_rows={fmt_pct(int(dup.sum()), total_c)}")
    lines.append(f"very_short_text_lt3w={fmt_pct(int(very_short.sum()), total_c)}")
    lines.append("")
    lines.append("length_stats:")
    lines.append(f"  char_len: min={char_len.min():.1f} mean={char_len.mean():.1f} median={char_len.median():.1f} p90={np.percentile(char_len,90):.1f} max={char_len.max():.1f}")
    lines.append(f"  word_len: min={word_len.min():.1f} mean={word_len.mean():.1f} median={word_len.median():.1f} p90={np.percentile(word_len,90):.1f} max={word_len.max():.1f}")

    lines.append(section("11) Comments per doctor"))
    cpd = comments.groupby("doctor_id").size()
    if len(cpd) > 0:
        lines.append(f"doctors_with_any_comment={cpd.shape[0]:,}")
        lines.append(f"comments_per_doctor_min={int(cpd.min()):,}")
        lines.append(f"comments_per_doctor_mean={cpd.mean():.2f}")
        lines.append(f"comments_per_doctor_median={cpd.median():.2f}")
        lines.append(f"comments_per_doctor_p90={np.percentile(cpd.values,90):.2f}")
        lines.append(f"comments_per_doctor_max={int(cpd.max()):,}")

    lines.append(section("12) Date coverage"))
    dates = comments["date"].dropna().astype(str).apply(norm_date)
    lines.append(f"date_non_null={fmt_pct(int(len(dates)), total_c)}")
    years = dates.str.extract(r"(\d{4})")[0].dropna()
    if len(years) > 0:
        yc = years.value_counts().sort_index()
        lines.append("")
        lines.append("year_counts:")
        for y, cnt in yc.items():
            lines.append(f"  {y}: {int(cnt):,}")

    lines.append(section("13) Sample comments per label"))
    rng = np.random.default_rng(42)
    for lbl in [-1, 0, 1]:
        sub = comments[comments["label"] == lbl]["text"].dropna().astype(str)
        lines.append(f"\nlabel={label_map[lbl]} total={len(sub):,}")
        if len(sub) == 0:
            continue
        idx = rng.choice(np.arange(len(sub)), size=min(5, len(sub)), replace=False)
        arr = sub.iloc[idx].tolist()
        for t in arr:
            preview = t.replace("\n", " ").strip()
            preview = preview[:180]
            lines.append(f"  - {preview}")

    lines.append(section("14) Doctors with zero comments (in this dataset)"))
    docs_all = set(doctors["doctor_id"].dropna().astype(str).unique().tolist())
    docs_with = set(comments["doctor_id"].dropna().astype(str).unique().tolist())
    no_comments = docs_all - docs_with
    lines.append(f"doctors_with_zero_comments={len(no_comments):,}")

    lines.append(section("15) Biography coverage"))
    bio = doctors["biography"].astype(str)
    has_bio = int((bio.str.strip() != "").sum())
    lines.append(f"has_biography={fmt_pct(has_bio, total_d)}")
    lines.append(f"empty_biography={fmt_pct(total_d - has_bio, total_d)}")
    bio_len = bio[bio.str.strip() != ""].str.len()
    if len(bio_len) > 0:
        lines.append(f"bio_char_len_min={int(bio_len.min()):,}")
        lines.append(f"bio_char_len_mean={bio_len.mean():.1f}")
        lines.append(f"bio_char_len_max={int(bio_len.max()):,}")

    lines.append("\n" + "=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)
    return lines


def main():
    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"data_dir_exists={DATA_DIR.exists()}")

    doctors, comments, failed = load_all(DATA_DIR)

    print("analyzing...")
    lines = analyze(doctors, comments, failed)

    report = "\n".join(lines)
    print(report)

    OUT_FILE.write_text(report, encoding="utf-8")
    print(f"\nsaved_report={OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()