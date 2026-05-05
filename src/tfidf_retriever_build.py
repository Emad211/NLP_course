# src/tfidf_retriever_build.py
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent

FINAL_DIR = BASE_DIR / "processed_data" / "final"
RETR_DIR = BASE_DIR / "processed_data" / "retriever"
RETR_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOCTORS_CSV = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"


def top_terms_for_row(row, feature_names, top_n=20):
    v = row.toarray().ravel()
    if v.size == 0:
        return []
    k = min(top_n, v.size)
    idx = np.argpartition(v, -k)[-k:]
    idx = idx[np.argsort(v[idx])[::-1]]
    return [(feature_names[i], float(v[i])) for i in idx if v[i] > 0]


def resolve_paths(variant: str):
    """
    baseline:
      input:  processed_data/final/comments_for_tfidf_retriever.csv
      output: processed_data/retriever/doctor_documents.csv
              processed_data/retriever/doctor_keywords.csv
              processed_data/retriever/tfidf_retriever_build_report.txt
              models/tfidf_doctor_retriever.joblib

    medical:
      input:  processed_data/final/comments_for_tfidf_retriever_medical.csv
      output: processed_data/retriever/doctor_documents_medical.csv
              processed_data/retriever/doctor_keywords_medical.csv
              processed_data/retriever/tfidf_retriever_build_report_medical.txt
              models/tfidf_doctor_retriever_medical.joblib
    """
    if variant not in {"baseline", "medical"}:
        raise ValueError("variant must be one of: baseline, medical")

    if variant == "baseline":
        in_csv = FINAL_DIR / "comments_for_tfidf_retriever.csv"
        doctor_docs_csv = RETR_DIR / "doctor_documents.csv"
        doctor_keywords_csv = RETR_DIR / "doctor_keywords.csv"
        report_txt = RETR_DIR / "tfidf_retriever_build_report.txt"
        index_joblib = MODELS_DIR / "tfidf_doctor_retriever.joblib"
    else:
        in_csv = FINAL_DIR / "comments_for_tfidf_retriever_medical.csv"
        doctor_docs_csv = RETR_DIR / "doctor_documents_medical.csv"
        doctor_keywords_csv = RETR_DIR / "doctor_keywords_medical.csv"
        report_txt = RETR_DIR / "tfidf_retriever_build_report_medical.txt"
        index_joblib = MODELS_DIR / "tfidf_doctor_retriever_medical.joblib"

    return in_csv, doctor_docs_csv, doctor_keywords_csv, report_txt, index_joblib


def parse_args():
    p = argparse.ArgumentParser(description="Build TF-IDF doctor retriever (baseline/medical).")
    p.add_argument("--variant", type=str, default="baseline", choices=["baseline", "medical"])

    # keep defaults identical to your current script
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-df", type=float, default=0.95)
    p.add_argument("--ngram-max", type=int, default=2, choices=[1, 2])
    p.add_argument("--max-features", type=int, default=60000)

    p.add_argument("--top-n-terms", type=int, default=25)
    p.add_argument("--sample-top-doctors", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    IN_CSV, DOCTOR_DOCS_CSV, DOCTOR_KEYWORDS_CSV, REPORT_TXT, INDEX_JOBLIB = resolve_paths(args.variant)

    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    df = pd.read_csv(IN_CSV, keep_default_na=False)
    df["doctor_id"] = df["doctor_id"].astype(str)

    if "text_for_tfidf" not in df.columns:
        raise ValueError("Input CSV must contain column: text_for_tfidf")

    df["text_for_tfidf"] = df["text_for_tfidf"].astype(str)

    total_rows = len(df)
    placeholder_rows = int(df.get("is_placeholder_negative", pd.Series([False] * total_rows)).astype(bool).sum())

    use = df[df["text_for_tfidf"].str.strip().ne("")].copy()
    if "is_placeholder_negative" in use.columns:
        use = use[~use["is_placeholder_negative"].astype(bool)].copy()

    used_rows = len(use)

    doc_stats = use.groupby("doctor_id").size().rename("comments_used").reset_index()
    doctor_docs = (
        use.groupby("doctor_id")["text_for_tfidf"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index()
        .rename(columns={"text_for_tfidf": "document"})
    )

    doctor_docs = doctor_docs.merge(doc_stats, on="doctor_id", how="left")
    doctor_docs["doc_char_len"] = doctor_docs["document"].str.len()
    doctor_docs["doc_token_len_est"] = doctor_docs["document"].str.split().str.len()

    doctor_docs.to_csv(DOCTOR_DOCS_CSV, index=False, encoding="utf-8-sig")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=True,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
    )

    X = vectorizer.fit_transform(doctor_docs["document"].astype(str).tolist())
    feature_names = np.array(vectorizer.get_feature_names_out())

    kw_rows = []
    for i in range(X.shape[0]):
        terms = top_terms_for_row(X[i], feature_names, top_n=args.top_n_terms)
        kw_rows.append(
            {
                "doctor_id": doctor_docs.loc[i, "doctor_id"],
                "comments_used": int(doctor_docs.loc[i, "comments_used"]),
                "doc_token_len_est": int(doctor_docs.loc[i, "doc_token_len_est"]),
                "top_terms": " | ".join([t for t, _ in terms[:15]]),
                "top_terms_scored": " | ".join([f"{t}:{s:.4f}" for t, s in terms[:15]]),
            }
        )

    keywords = pd.DataFrame(kw_rows)
    keywords.to_csv(DOCTOR_KEYWORDS_CSV, index=False, encoding="utf-8-sig")

    doctors = pd.read_csv(DOCTORS_CSV, keep_default_na=False) if DOCTORS_CSV.exists() else pd.DataFrame()
    name_map = {}
    if not doctors.empty and "doctor_id" in doctors.columns and "name" in doctors.columns:
        tmp = doctors.copy()
        tmp["doctor_id"] = tmp["doctor_id"].astype(str)
        name_map = dict(zip(tmp["doctor_id"], tmp["name"]))

    index_obj = {
        "variant": args.variant,
        "vectorizer": vectorizer,
        "X": X,
        "doctor_ids": doctor_docs["doctor_id"].tolist(),
        "doctor_docs": doctor_docs[["doctor_id", "document", "comments_used", "doc_token_len_est"]].copy(),
        "name_map": name_map,
        "build_meta": {
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_range": (1, args.ngram_max),
            "max_features": args.max_features,
            "rows_total_in_input": int(total_rows),
            "placeholder_rows_in_input": int(placeholder_rows),
            "rows_used_for_doctor_corpus": int(used_rows),
            "doctors_in_corpus": int(len(doctor_docs)),
            "tfidf_matrix_shape": (int(X.shape[0]), int(X.shape[1])),
        },
    }
    joblib.dump(index_obj, INDEX_JOBLIB)

    lines = []
    lines.append(f"TF-IDF DOCTOR RETRIEVER — BUILD REPORT ({args.variant})")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_comments_csv={IN_CSV.resolve()}")
    lines.append(f"doctor_docs_csv={DOCTOR_DOCS_CSV.resolve()}")
    lines.append(f"keywords_csv={DOCTOR_KEYWORDS_CSV.resolve()}")
    lines.append(f"index_joblib={INDEX_JOBLIB.resolve()}")

    lines.append("")
    lines.append("PARAMS")
    lines.append(f"min_df={args.min_df}")
    lines.append(f"max_df={args.max_df}")
    lines.append(f"ngram_range=(1,{args.ngram_max})")
    lines.append(f"max_features={args.max_features}")

    lines.append("")
    lines.append(f"rows_total_in_input={total_rows:,}")
    lines.append(f"placeholder_rows_in_input={placeholder_rows:,}")
    lines.append(f"rows_used_for_doctor_corpus={used_rows:,}")
    lines.append(f"doctors_in_corpus={len(doctor_docs):,}")
    lines.append(f"tfidf_matrix_shape={X.shape[0]} x {X.shape[1]}")

    top_docs = doctor_docs.sort_values("comments_used", ascending=False).head(args.sample_top_doctors)
    lines.append("")
    lines.append(f"SAMPLE TOP KEYWORDS (top {len(top_docs)} doctors by comments_used)")
    for r in top_docs.itertuples(index=False):
        did = str(r.doctor_id)
        nm = name_map.get(did, "")
        kw = keywords[keywords["doctor_id"].astype(str) == did]["top_terms"].values
        kw = kw[0] if len(kw) else ""
        header = f"doctor_id={did}"
        if nm:
            header += f"  name={nm}"
        header += f"  comments_used={int(r.comments_used)}  doc_tokens_est={int(r.doc_token_len_est)}"
        lines.append(header)
        lines.append(f"  top_terms={kw}")
        lines.append("")

    dt = time.time() - t0
    lines.append(f"seconds={dt:.2f}")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()