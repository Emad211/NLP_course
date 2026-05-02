import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCTORS_OUT = OUT_DIR / "doctors_raw_canonical.csv"
COMMENTS_OUT = OUT_DIR / "comments_raw_canonical.csv"
DUP_DOCS_OUT = OUT_DIR / "duplicate_doctor_ids.csv"
REPORT_OUT = OUT_DIR / "raw_dataset_builder_report.txt"


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_file(fpath: Path):
    data = json.loads(fpath.read_text(encoding="utf-8"))
    p = data.get("doctor_profile", {}) or {}
    comments = data.get("comments", []) or []

    doc_id = p.get("doctor_id")
    doc_id = str(doc_id) if doc_id is not None else None

    profile = {
        "doctor_id": doc_id,
        "source_file": fpath.name,
        "profile_url": p.get("profile_url"),
        "nice_id": p.get("nice_id"),
        "name": p.get("name"),
        "specialty": p.get("specialty"),
        "medical_code": p.get("medical_code"),
        "verified": bool(p.get("verified", False)),
        "followers_count": safe_int(p.get("followers_count")),
        "reviews_count": safe_int(p.get("reviews_count")),
        "rating_percent": safe_float(p.get("rating_percent")),
        "rating_score_5": safe_float(p.get("rating_score_5")),
        "biography": (p.get("biography") or ""),
        "office_count": len(p.get("offices", []) or []),
        "comments_count_in_file": len(comments),
    }

    comment_rows = []
    for i, c in enumerate(comments):
        comment_rows.append(
            {
                "doctor_id": doc_id,
                "source_file": fpath.name,
                "comment_index_in_file": i,
                "specialty": p.get("specialty"),
                "text": c.get("text", ""),
                "rate": safe_float(c.get("rate")),
                "label": safe_int(c.get("label")),
                "date": c.get("date"),
            }
        )

    return profile, comment_rows


def main():
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(str(DATA_DIR.resolve()))

    profiles = []
    comments_all = []
    failures = []

    for fpath in files:
        try:
            prof, cmts = load_file(fpath)
            profiles.append(prof)
            comments_all.extend(cmts)
        except Exception as e:
            failures.append((fpath.name, str(e)))

    profiles_df = pd.DataFrame(profiles)
    comments_df = pd.DataFrame(comments_all)

    doc_id_counts = profiles_df["doctor_id"].value_counts(dropna=False)
    dup_doc_ids = doc_id_counts[doc_id_counts >= 2].index.tolist()

    dup_rows = []
    for doc_id in dup_doc_ids:
        sub = profiles_df[profiles_df["doctor_id"] == doc_id].copy()
        sub = sub.sort_values(
            ["comments_count_in_file", "reviews_count", "rating_score_5"],
            ascending=[False, False, False],
            na_position="last",
        )
        for r in sub.itertuples(index=False):
            dup_rows.append(
                {
                    "doctor_id": r.doctor_id,
                    "source_file": r.source_file,
                    "comments_count_in_file": r.comments_count_in_file,
                    "reviews_count": r.reviews_count,
                    "rating_score_5": r.rating_score_5,
                    "specialty": r.specialty,
                    "name": r.name,
                }
            )

    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(DUP_DOCS_OUT, index=False, encoding="utf-8-sig")

    chosen_files = {}
    for doc_id, sub in profiles_df.groupby("doctor_id", dropna=False):
        sub2 = sub.sort_values(
            ["comments_count_in_file", "reviews_count", "rating_score_5"],
            ascending=[False, False, False],
            na_position="last",
        )
        chosen_files[doc_id] = sub2.iloc[0]["source_file"]

    canonical_profiles = []
    canonical_comments = []

    files_by_name = {p["source_file"]: p for p in profiles}
    comments_grouped = comments_df.groupby("source_file")

    for doc_id, chosen_file in chosen_files.items():
        prof = files_by_name.get(chosen_file)
        if prof is None:
            continue
        prof = dict(prof)
        prof["selected_source_file"] = chosen_file

        all_files_for_doc = profiles_df[profiles_df["doctor_id"] == doc_id]["source_file"].tolist()
        prof["all_source_files"] = " | ".join(all_files_for_doc)
        prof["source_files_count"] = len(all_files_for_doc)

        canonical_profiles.append(prof)

        if chosen_file in comments_grouped.groups:
            canonical_comments.append(comments_grouped.get_group(chosen_file))

    doctors_canonical = pd.DataFrame(canonical_profiles)
    comments_canonical = pd.concat(canonical_comments, ignore_index=True) if canonical_comments else pd.DataFrame()

    doctors_canonical = doctors_canonical.sort_values("doctor_id").reset_index(drop=True)
    comments_canonical = comments_canonical.sort_values(["doctor_id", "comment_index_in_file"]).reset_index(drop=True)

    doctors_canonical.to_csv(DOCTORS_OUT, index=False, encoding="utf-8-sig")
    comments_canonical.to_csv(COMMENTS_OUT, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("RAW DATASET BUILDER REPORT")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"data_dir={DATA_DIR.resolve()}")
    lines.append(f"json_files_total={len(files):,}")
    lines.append(section("Load status"))
    lines.append(f"files_loaded={len(files)-len(failures):,}")
    lines.append(f"files_failed={len(failures):,}")
    for name, err in failures[:15]:
        lines.append(f"FAIL {name} -> {err}")
    if len(failures) > 15:
        lines.append(f"... {len(failures)-15} more failures")

    lines.append(section("Before canonicalization"))
    lines.append(f"profiles_rows={len(profiles_df):,}")
    lines.append(f"unique_doctor_ids_in_profiles={profiles_df['doctor_id'].nunique(dropna=True):,}")
    lines.append(f"comments_rows={len(comments_df):,}")
    lines.append(f"unique_doctor_ids_in_comments={comments_df['doctor_id'].nunique(dropna=True):,}")
    lines.append(f"duplicate_doctor_ids_count={(doc_id_counts >= 2).sum():,}")
    lines.append(f"duplicate_profile_rows={(profiles_df['doctor_id'].duplicated()).sum():,}")

    lines.append(section("After canonicalization"))
    lines.append(f"doctors_canonical_rows={len(doctors_canonical):,}")
    lines.append(f"comments_canonical_rows={len(comments_canonical):,}")
    lines.append(f"unique_doctor_ids_in_canonical_comments={comments_canonical['doctor_id'].nunique(dropna=True):,}")

    lines.append(section("Top duplicate doctor_ids (by number of profile files)"))
    if len(dup_doc_ids) == 0:
        lines.append("none")
    else:
        top_dup = profiles_df["doctor_id"].value_counts().head(20)
        for k, v in top_dup.items():
            if v >= 2:
                lines.append(f"doctor_id={k}  profile_files={int(v)}  chosen_file={chosen_files.get(k)}")

    lines.append(section("Outputs"))
    lines.append(f"doctors_csv={DOCTORS_OUT.resolve()}")
    lines.append(f"comments_csv={COMMENTS_OUT.resolve()}")
    lines.append(f"duplicate_doctor_ids_csv={DUP_DOCS_OUT.resolve()}")

    report = "\n".join(lines)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nsaved_report={REPORT_OUT.resolve()}")


if __name__ == "__main__":
    main()